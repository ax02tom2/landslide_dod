"""core.py — 崩塌地形變異分析核心函式（不依賴 Streamlit，可單獨測試）

符號慣例
    z1 : T1（崩塌前）高程      z2 : T2（崩塌後）高程
    dz = z2 - z1               dz < 0 → 侵蝕/崩落,  dz > 0 → 堆積
"""
from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass

import numpy as np
import rasterio
from PIL import Image
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window, from_bounds, intersection
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import map_coordinates


# --------------------------------------------------------------------------
# 網格
# --------------------------------------------------------------------------
@dataclass
class Grid:
    transform: Affine
    crs_wkt: str
    shape: tuple  # (rows, cols)

    @property
    def crs(self) -> CRS:
        return CRS.from_wkt(self.crs_wkt)

    @property
    def cell_x(self) -> float:
        return abs(self.transform.a)

    @property
    def cell_y(self) -> float:
        return abs(self.transform.e)

    @property
    def cell_area(self) -> float:
        return self.cell_x * self.cell_y

    @property
    def extent(self):
        """(left, right, bottom, top)"""
        h, w = self.shape
        t = self.transform
        return (t.c, t.c + w * t.a, t.f + h * t.e, t.f)

    @property
    def center(self):
        l, r, b, t = self.extent
        return ((l + r) / 2, (b + t) / 2)


def display_grid(grid: Grid, step: int) -> Grid:
    h, w = grid.shape
    return Grid(
        grid.transform * Affine.scale(step, step),
        grid.crs_wkt,
        (math.ceil(h / step), math.ceil(w / step)),
    )


# --------------------------------------------------------------------------
# 讀取 / 對齊
# --------------------------------------------------------------------------
def clean_array(a, nodata=None, override=None):
    """轉 float32 並把 NoData / 非有限值 / 不合理高程設為 NaN。"""
    a = np.asarray(a).astype("float32")
    bad = ~np.isfinite(a) | (a < -1e4) | (a > 1e5)
    for nd in (nodata, override):
        if nd is not None and np.isfinite(nd):
            bad |= a == np.float32(nd)
    a[bad] = np.nan
    return a


def raster_info(path: str) -> dict:
    with rasterio.open(path) as s:
        return dict(
            width=s.width,
            height=s.height,
            bands=s.count,
            res=(abs(s.transform.a), abs(s.transform.e)),
            crs=s.crs.to_string() if s.crs else None,
            projected=bool(s.crs and s.crs.is_projected),
            nodata=s.nodata,
            dtype=s.dtypes[0],
            bounds=tuple(s.bounds),
        )


def warp_band_to_grid(path, grid: Grid, band=1, nodata_override=None, resampling=Resampling.bilinear):
    """把任一 raster 的某波段重投影/重採樣到 grid（滑動面、T2 都用這個）。"""
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(path) as s:
        if s.crs is None:
            raise ValueError(f"{path} 沒有座標系統 (CRS)。")
        src_nd = nodata_override if nodata_override is not None else s.nodata
        reproject(
            source=rasterio.band(s, band),
            destination=dst,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=src_nd,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return clean_array(dst, None, nodata_override)


def load_pair(path1, path2, factor=1, nodata_override=None, max_cells=30_000_000):
    """讀取兩期 DEM，裁切到交集範圍並對齊到 T1 網格（可降採樣）。

    回傳 (z1, z2, Grid)。T1 為基準網格；T2 重投影+重採樣到 T1 網格。
    """
    factor = max(1, int(factor))
    with rasterio.open(path1) as s1, rasterio.open(path2) as s2:
        if s1.crs is None or s2.crs is None:
            raise ValueError("DEM 缺少座標系統 (CRS)，請先在 GIS 軟體指定後再上傳。")
        if not s1.crs.is_projected:
            raise ValueError(
                f"T1 為地理座標系 ({s1.crs})，面積/體積無法以公尺計算。請轉為投影座標 (如 TWD97 TM2 / EPSG:3826)。"
            )
        t1 = s1.transform
        if t1.b != 0 or t1.d != 0:
            raise ValueError("T1 影像含旋轉，不支援。")
        if t1.e > 0:
            raise ValueError("T1 影像為南上北下（e>0），不支援。")

        b2 = transform_bounds(s2.crs, s1.crs, *s2.bounds)
        b1 = s1.bounds
        l, b = max(b1.left, b2[0]), max(b1.bottom, b2[1])
        r, t = min(b1.right, b2[2]), min(b1.top, b2[3])
        if l >= r or b >= t:
            raise ValueError("兩期 DEM 範圍沒有交集，請確認座標系統與位置。")

        win = from_bounds(l, b, r, t, transform=t1)
        win = Window(round(win.col_off), round(win.row_off), round(win.width), round(win.height))
        win = intersection(win, Window(0, 0, s1.width, s1.height))
        out_h = max(1, int(round(win.height / factor)))
        out_w = max(1, int(round(win.width / factor)))
        if out_h * out_w > max_cells:
            need = math.ceil(math.sqrt(win.height * win.width / max_cells))
            raise ValueError(
                f"交集範圍 {int(win.width)}×{int(win.height)} 格太大（超過 {max_cells:,} 格）。"
                f" 請把「降採樣倍率」調到 {need} 以上。"
            )
        if factor == 1:
            res = Resampling.nearest
        elif nodata_override is not None:
            res = Resampling.nearest  # 未宣告的 NoData 不能平均
        else:
            res = Resampling.average
        raw1 = s1.read(1, window=win, out_shape=(out_h, out_w), resampling=res, out_dtype="float32")
        z1 = clean_array(raw1, s1.nodata, nodata_override)
        tr = s1.window_transform(win) * Affine.scale(win.width / out_w, win.height / out_h)
        grid = Grid(tr, s1.crs.to_wkt(), (out_h, out_w))

    rs2 = Resampling.bilinear if factor == 1 else Resampling.average
    z2 = warp_band_to_grid(path2, grid, nodata_override=nodata_override, resampling=rs2)
    return z1, z2, grid


# --------------------------------------------------------------------------
# 兩期對位校正（穩定區）
# --------------------------------------------------------------------------
def fit_correction(dz, stable_mask, grid: Grid, mode="offset", clip_sigma=3.0, max_pts=300_000):
    """用穩定區 dz 擬合系統誤差。

    mode: 'offset' 只修平移（中位數）;  'plane' 平移+傾斜 (a + b·x + c·y)。
    回傳 (corr, stats)；corr 為可直接從 dz 扣除的 float32 陣列/純量。
    """
    if mode == "none" or stable_mask is None:
        return 0.0, {}
    ys, xs = np.nonzero(stable_mask & np.isfinite(dz))
    if len(ys) < 50:
        raise ValueError("穩定區有效像元少於 50 格，無法校正。請擴大穩定區範圍。")
    if len(ys) > max_pts:
        sel = np.random.default_rng(0).choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    tr = grid.transform
    X = tr.c + (xs + 0.5) * tr.a
    Y = tr.f + (ys + 0.5) * tr.e
    x0, y0 = X.mean(), Y.mean()
    v = dz[ys, xs].astype("float64")

    if mode == "offset":
        A = np.ones((len(v), 1))
    else:
        A = np.c_[np.ones(len(v)), X - x0, Y - y0]

    keep = np.ones(len(v), bool)
    coef = np.zeros(A.shape[1])
    for _ in range(4):
        if mode == "offset":
            coef = np.array([np.median(v[keep])])
        else:
            coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        res = v - A @ coef
        s = res[keep].std()
        new_keep = np.abs(res) < clip_sigma * max(s, 1e-9)
        if new_keep.sum() < 50 or (new_keep == keep).all():
            break
        keep = new_keep

    res = v - A @ coef
    stats = dict(
        mode=mode,
        n=int(len(v)),
        n_used=int(keep.sum()),
        before_mean=float(v.mean()),
        before_std=float(v.std()),
        before_rmse=float(np.sqrt((v**2).mean())),
        after_mean=float(res.mean()),
        after_std=float(res.std()),
        after_rmse=float(np.sqrt((res**2).mean())),
        offset=float(coef[0]),
        slope_x=float(coef[1]) if mode == "plane" else 0.0,
        slope_y=float(coef[2]) if mode == "plane" else 0.0,
    )
    h, w = dz.shape
    xc = tr.c + (np.arange(w) + 0.5) * tr.a
    yc = tr.f + (np.arange(h) + 0.5) * tr.e
    corr = np.float32(coef[0]) + np.zeros(dz.shape, dtype="float32")
    if mode == "plane":
        corr += (coef[1] * (xc - x0)).astype("float32")[None, :]
        corr += (coef[2] * (yc - y0)).astype("float32")[:, None]
    return corr, stats


# --------------------------------------------------------------------------
# 範圍 / 統計
# --------------------------------------------------------------------------
def rasterize(geoms, shape, transform):
    """GeoJSON-like geometry list（與網格同 CRS）→ bool mask。無幾何回傳 None。"""
    if not geoms:
        return None
    return features.geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def volume_stats(dz, lod, cell_area, mask, sigma=None):
    d = dz[mask]
    d = d[np.isfinite(d)]
    ero = d[d < -lod]
    dep = d[d > lod]
    A = cell_area
    n = len(d)
    out = dict(
        區域面積_m2=n * A,
        侵蝕面積_m2=len(ero) * A,
        堆積面積_m2=len(dep) * A,
        侵蝕體積_m3=float(-ero.sum(dtype="float64") * A),
        堆積體積_m3=float(dep.sum(dtype="float64") * A),
        平均侵蝕深_m=float(-ero.mean()) if len(ero) else 0.0,
        最大侵蝕深_m=float(-ero.min()) if len(ero) else 0.0,
        平均堆積厚_m=float(dep.mean()) if len(dep) else 0.0,
        最大堆積厚_m=float(dep.max()) if len(dep) else 0.0,
    )
    out["淨變量_m3"] = out["堆積體積_m3"] - out["侵蝕體積_m3"]
    if sigma is not None and n:
        out["不確定度_隨機_m3"] = float(sigma * math.sqrt(n) * A)
        out["不確定度_系統_m3"] = float(sigma * n * A)
    return out


# --------------------------------------------------------------------------
# 滑動面 / 殘餘土體
# --------------------------------------------------------------------------
def slip_uniform(z1, depth):
    """假設一：滑動面 = T1 地表往下等深度 depth。"""
    return z1 - np.float32(depth)


def slip_interpolated(z1, z2, dz, lod, region, max_pts=20000):
    """假設二：由「已崩落像元」的實測深度，內插到範圍內尚未崩落的位置。

    - 已崩落像元 (dz < -LoD)：滑動面 = 現況地表 z2（外露滑動面/崩壁底）
    - 範圍邊界上未崩落處：深度 = 0（滑動面在邊界出露）
    - 其餘像元：以上述已知深度線性內插（範圍外推用最近鄰）
    回傳滑動面高程陣列（範圍外為 NaN）。
    """
    region = region & np.isfinite(z1) & np.isfinite(dz)
    scar = region & (dz < -lod)
    if scar.sum() < 10:
        raise ValueError("範圍內已崩落像元（dz < −LoD）少於 10 格，無法內插滑動面。請改用等深度假設或上傳滑動面。")
    edge = region & ~ndimage.binary_erosion(region) & ~scar

    def _pts(m, n):
        ys, xs = np.nonzero(m)
        if len(ys) > n:
            sel = np.linspace(0, len(ys) - 1, n).astype(int)
            ys, xs = ys[sel], xs[sel]
        return ys, xs

    sy, sx = _pts(scar, max_pts)
    ey, ex = _pts(edge, max_pts // 4)
    pts = np.c_[np.r_[sx, ex], np.r_[sy, ey]].astype("float64")
    vals = np.r_[-dz[sy, sx], np.zeros(len(ey))].astype("float64")

    ty, tx = np.nonzero(region & ~scar)
    slip = np.full(z1.shape, np.nan, dtype="float32")
    slip[scar] = z2[scar]
    if len(ty):
        q = np.c_[tx, ty].astype("float64")
        try:
            d = LinearNDInterpolator(pts, vals)(q)
        except Exception:  # Qhull 退化（點共線等）
            d = np.full(len(q), np.nan)
        bad = ~np.isfinite(d)
        if bad.any():
            d[bad] = NearestNDInterpolator(pts, vals)(q[bad])
        slip[ty, tx] = z1[ty, tx] - d.astype("float32")
    return slip


def residual_thickness(z2, slip, region):
    """殘餘土體厚度 = max(現況地表 − 滑動面, 0)，限於範圍內。"""
    t = (z2 - slip).astype("float32")
    t = np.where(np.isfinite(t), np.maximum(t, 0), np.nan).astype("float32")
    t[~region] = np.nan
    return t


def residual_summary(thick, cell_area, region):
    v = thick[region]
    v = v[np.isfinite(v)]
    pos = v[v > 0]
    return dict(
        殘餘土體體積_m3=float(v.sum(dtype="float64") * cell_area),
        有殘餘土體面積_m2=len(pos) * cell_area,
        範圍面積_m2=len(v) * cell_area,
        平均厚度_m=float(pos.mean()) if len(pos) else 0.0,
        最大厚度_m=float(pos.max()) if len(pos) else 0.0,
    )


def uniform_sensitivity(dz, region, cell_area, depths):
    """等深度假設的敏感度：殘餘體積 = Σ max(d + dz, 0)·A。"""
    v = dz[region & np.isfinite(dz)].astype("float64")
    rows = []
    for d in depths:
        t = np.maximum(d + v, 0)
        rows.append(dict(假設滑動面深度_m=float(d), 殘餘體積_m3=float(t.sum() * cell_area),
                         殘餘面積_m2=float((t > 0).sum() * cell_area)))
    return rows


def scar_depth_hint(dz, lod, region):
    d = -dz[region & np.isfinite(dz) & (dz < -lod)]
    if len(d) == 0:
        return None
    return dict(median=float(np.median(d)), mean=float(d.mean()), p90=float(np.percentile(d, 90)), max=float(d.max()))


# --------------------------------------------------------------------------
# 剖面
# --------------------------------------------------------------------------
def sample_polyline(coords, step):
    pts = np.asarray(coords, dtype="float64")[:, :2]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    L = cum[-1]
    if L <= 0:
        raise ValueError("剖面線長度為 0。")
    n = max(2, int(L / step) + 1)
    d = np.linspace(0, L, n)
    return d, np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])


def sample_raster(arr, transform, xs, ys):
    col = (xs - transform.c) / transform.a - 0.5
    row = (ys - transform.f) / transform.e - 0.5
    return map_coordinates(arr.astype("float32"), [row, col], order=1, mode="constant", cval=np.nan, prefilter=False)


def _trapz(y, x):
    y = np.nan_to_num(y)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)))


def profile_areas(dist, g1, g2, lod, slip=None):
    d = g2 - g1
    ero = np.where(np.isfinite(d) & (d < -lod), -d, 0.0)
    dep = np.where(np.isfinite(d) & (d > lod), d, 0.0)
    out = dict(剖面長度_m=float(dist[-1]), 侵蝕斷面積_m2=_trapz(ero, dist), 堆積斷面積_m2=_trapz(dep, dist))
    if slip is not None:
        res = np.where(np.isfinite(g2 - slip), np.maximum(g2 - slip, 0), 0.0)
        out["殘餘土體斷面積_m2"] = _trapz(res, dist)
    return out


def profile_to_dxf(dist, series: dict) -> bytes:
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    for name, y in series.items():
        if y is None:
            continue
        doc.layers.add(name)
        ok = np.isfinite(y)
        i = 0
        n = len(y)
        while i < n:
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < n and ok[j]:
                j += 1
            if j - i >= 2:
                msp.add_lwpolyline([(float(dist[k]), float(y[k])) for k in range(i, j)], dxfattribs={"layer": name})
            i = j
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# 顯示用影像
# --------------------------------------------------------------------------
def hillshade(z, cell, az=315.0, alt=45.0, zf=1.0):
    zz = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0)
    gy, gx = np.gradient(zz.astype("float32"), cell)
    dzdx, dzdy = gx, -gy  # 東向、北向
    a, h = math.radians(az), math.radians(alt)
    hs = (-zf * dzdx * math.sin(a) * math.cos(h) - zf * dzdy * math.cos(a) * math.cos(h) + math.sin(h)) / np.sqrt(
        1 + (zf * dzdx) ** 2 + (zf * dzdy) ** 2
    )
    hs = np.clip(hs, 0, 1)
    hs[~np.isfinite(z)] = np.nan
    return hs.astype("float32")


def _cmap(name):
    import matplotlib

    return matplotlib.colormaps[name]


def hillshade_rgba(hs):
    g = np.nan_to_num(hs, nan=0.0)
    g8 = (g * 255).astype("uint8")
    a = np.where(np.isfinite(hs), 255, 0).astype("uint8")
    return np.dstack([g8, g8, g8, a])


def colorize_dz(dz, lod, vmax, alpha=0.75):
    norm = np.clip((np.nan_to_num(dz) + vmax) / (2 * vmax), 0, 1)
    rgba = (_cmap("RdBu")(norm) * 255).astype("uint8")  # 負=紅(侵蝕) 正=藍(堆積)
    show = np.isfinite(dz) & (np.abs(dz) > lod)
    rgba[..., 3] = np.where(show, int(alpha * 255), 0)
    return rgba


def colorize_thickness(t, vmax, alpha=0.75):
    norm = np.clip(np.nan_to_num(t) / max(vmax, 1e-6), 0, 1)
    rgba = (_cmap("YlOrBr")(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(t) & (t > 0), int(alpha * 255), 0)
    return rgba


def ortho_rgba(path, dgrid: Grid, max_bands=3):
    """把正射影像重投影到顯示網格，回傳 RGBA uint8。"""
    h, w = dgrid.shape
    with rasterio.open(path) as s:
        if s.crs is None:
            raise ValueError("正射影像沒有座標系統 (CRS)。")
        nb = min(s.count, max_bands)
        bands = []
        for i in range(1, nb + 1):
            dst = np.full((h, w), np.nan, dtype="float32")
            reproject(rasterio.band(s, i), dst, dst_transform=dgrid.transform, dst_crs=dgrid.crs,
                      src_nodata=s.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
            bands.append(dst)
        alpha = None
        if s.count >= 4:
            alpha = np.full((h, w), np.nan, dtype="float32")
            reproject(rasterio.band(s, 4), alpha, dst_transform=dgrid.transform, dst_crs=dgrid.crs,
                      dst_nodata=np.nan, resampling=Resampling.nearest)
        is8 = s.dtypes[0] == "uint8"
    if nb == 1:
        bands = bands * 3
    rgb = np.dstack(bands)
    valid = np.isfinite(rgb).all(axis=2)
    if not is8 and valid.any():
        lo, hi = np.nanpercentile(rgb[valid], [2, 98])
        rgb = (rgb - lo) / max(hi - lo, 1e-6) * 255
    rgb8 = np.clip(np.nan_to_num(rgb), 0, 255).astype("uint8")
    a = np.where(valid, 255, 0).astype("uint8")
    if alpha is not None:
        a = np.where(np.nan_to_num(alpha) > 0, a, 0).astype("uint8")
    return np.dstack([rgb8, a])


def to_wgs84(rgba, transform: Affine, crs):
    """把顯示網格的 RGBA 影像重投影成經緯度，供 Leaflet ImageOverlay 使用。"""
    h, w = rgba.shape[:2]
    left, top = transform * (0, 0)
    right, bottom = transform * (w, h)
    dt, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, left=left, bottom=bottom, right=right, top=top)
    bands = []
    for i in range(4):
        dst = np.zeros((dh, dw), dtype="uint8")
        reproject(np.ascontiguousarray(rgba[..., i]), dst, src_transform=transform, src_crs=crs,
                  dst_transform=dt, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest if i == 3 else Resampling.bilinear)
        bands.append(dst)
    west, north = dt.c, dt.f
    east, south = west + dw * dt.a, north + dh * dt.e
    return np.dstack(bands), [[south, west], [north, east]]


def png_uri(rgba) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# 匯出
# --------------------------------------------------------------------------
def array_to_geotiff_bytes(arr, grid: Grid, nodata=-9999.0) -> bytes:
    h, w = grid.shape
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=grid.crs,
                     transform=grid.transform, nodata=nodata, compress="deflate") as dst:
            dst.write(np.where(np.isfinite(arr), arr, nodata).astype("float32"), 1)
        return mf.read()


# --------------------------------------------------------------------------
# 前處理：裁切 / 縮小（分塊讀寫，低記憶體）
# --------------------------------------------------------------------------
def overview(path, maxpx=1200):
    """低解析度總覽（不載入整檔）。回傳 (陣列, Grid)；陣列為 float32（多波段取第 1 波段）。"""
    with rasterio.open(path) as s:
        if s.crs is None:
            raise ValueError("檔案沒有座標系統 (CRS)。")
        f = max(1, math.ceil(max(s.width, s.height) / maxpx))
        h, w = math.ceil(s.height / f), math.ceil(s.width / f)
        a = s.read(1, out_shape=(h, w), resampling=Resampling.average, out_dtype="float32")
        tr = s.transform * Affine.scale(s.width / w, s.height / h)
        return clean_array(a, s.nodata), Grid(tr, s.crs.to_wkt(), (h, w))


def crop_estimate(path, bounds, bounds_crs, factor):
    with rasterio.open(path) as s:
        b = bounds if s.crs == bounds_crs else transform_bounds(bounds_crs, s.crs, *bounds)
        win = from_bounds(*b, transform=s.transform)
        win = intersection(Window(int(win.col_off), int(win.row_off), math.ceil(win.width), math.ceil(win.height)),
                           Window(0, 0, s.width, s.height))
        w, h = int(win.width) // factor, int(win.height) // factor
        bytes_ = w * h * s.count * np.dtype(s.dtypes[0]).itemsize
        return dict(width=w, height=h, cells=w * h, raw_mb=bytes_ / 1e6, cell=abs(s.transform.a) * factor)


def crop_raster(src_path, dst_path, bounds, bounds_crs, factor=1, block_rows=512, progress=None):
    """依範圍裁切並可縮小（block average），分塊寫出 GeoTIFF。bounds=(left,bottom,right,top)。"""
    factor = max(1, int(factor))
    with rasterio.open(src_path) as s:
        b = bounds if s.crs == bounds_crs else transform_bounds(bounds_crs, s.crs, *bounds)
        win = from_bounds(*b, transform=s.transform)
        win = intersection(Window(int(win.col_off), int(win.row_off), math.ceil(win.width), math.ceil(win.height)),
                           Window(0, 0, s.width, s.height))
        out_w, out_h = int(win.width) // factor, int(win.height) // factor
        if out_w < 1 or out_h < 1:
            raise ValueError("裁切範圍太小或與影像沒有交集。")
        tr = s.window_transform(win) * Affine.scale(factor, factor)
        prof = s.profile.copy()
        prof.pop("photometric", None)
        isfloat = np.dtype(s.dtypes[0]).kind == "f"
        prof.update(driver="GTiff", width=out_w, height=out_h, transform=tr, compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
                    predictor=3 if isfloat else 2)
        res = Resampling.average if factor > 1 else Resampling.nearest
        with rasterio.open(dst_path, "w", **prof) as d:
            for r0 in range(0, out_h, block_rows):
                bh = min(block_rows, out_h - r0)
                sw = Window(int(win.col_off), int(win.row_off) + r0 * factor, out_w * factor, bh * factor)
                data = s.read(window=sw, out_shape=(s.count, bh, out_w), resampling=res)
                d.write(data, window=Window(0, r0, out_w, bh))
                if progress:
                    progress(min(1.0, (r0 + bh) / out_h))
    return dict(path=dst_path, width=out_w, height=out_h, cell=abs(tr.a), size_mb=os.path.getsize(dst_path) / 1e6)
