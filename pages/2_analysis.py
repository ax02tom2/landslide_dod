"""崩塌地形變異分析 (DEM of Difference) — Streamlit 網頁

功能：兩期 DEM/DSM 差分 → 崩塌/堆積量體 → 殘餘土體（含滑動面假設）→ 剖面切割 → 匯出
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from streamlit_folium import st_folium

import core
import demo_data

st.set_page_config(page_title="崩塌地形變異分析", page_icon="⛰️", layout="wide")

TMP = os.path.join(tempfile.gettempdir(), "dod_app")
os.makedirs(TMP, exist_ok=True)

ZONES = {
    # 工程 GIS 配色：紅=已發生、橙=潛勢、藍=堆積；其他為輔助圖層。
    "collapse": ("實際崩塌範圍", "#c62828", "polygon"),
    "potential": ("潛在滑動體範圍", "#ef6c00", "polygon"),
    "deposit": ("堆積區", "#1565c0", "polygon"),
    "stable": ("穩定區 (校正用)", "#2e7d32", "polygon"),
    "exclude": ("排除區 (植被/水體/建物)", "#616161", "polygon"),
    "profile": ("剖面線", "#7b1fa2", "line"),
}

# 工程 GIS 圖徵樣式：實際崩塌用實線、潛在滑動體用虛線、堆積區用藍色。
ZONE_DRAW_STYLES = {
    "collapse": {"color": "#c62828", "weight": 3.5, "opacity": 0.95, "fillColor": "#c62828", "fillOpacity": 0.12},
    "potential": {"color": "#ef6c00", "weight": 3.0, "opacity": 0.95, "fillColor": "#ef6c00", "fillOpacity": 0.08, "dashArray": "8 5"},
    "deposit": {"color": "#1565c0", "weight": 3.0, "opacity": 0.95, "fillColor": "#1565c0", "fillOpacity": 0.10},
    "stable": {"color": "#2e7d32", "weight": 2.5, "opacity": 0.9, "fillColor": "#2e7d32", "fillOpacity": 0.06, "dashArray": "6 4"},
    "exclude": {"color": "#616161", "weight": 2.5, "opacity": 0.9, "fillColor": "#616161", "fillOpacity": 0.05, "dashArray": "4 4"},
    "profile": {"color": "#7b1fa2", "weight": 3.5, "opacity": 0.95},
}
POLY_ZONES = [z for z, v in ZONES.items() if v[2] == "polygon"]

for z in ZONES:
    st.session_state.setdefault(f"geo_{z}", [])  # 已儲存的 GeoJSON features (WGS84)
    st.session_state.setdefault(f"ver_{z}", 0)  # 地圖版本（儲存後重置繪圖層）


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
# 只在「本次 Streamlit rerun」內快取。
# 不把大型 raster / PNG / mask 長期放進 session_state，避免互動幾次後記憶體累積而當機。
_RUN_CACHE = {}

def memo(key, fn):
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    val = fn()
    _RUN_CACHE[key] = val
    return val


def save_upload(uf):
    if uf is None:
        return None
    buf = uf.getbuffer()
    h = hashlib.md5(f"{uf.name}{uf.size}".encode() + bytes(buf[:1 << 20]) + bytes(buf[-(1 << 20):])).hexdigest()[:12]
    path = os.path.join(TMP, f"{h}_{os.path.basename(uf.name)}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(buf)
    return path


def fmt_m3(v):
    return f"{v:,.0f} m³"


def fmt_m2(v):
    return f"{v:,.0f} m²"


def region_bbox(mask):
    """Return a compact bbox around a boolean analysis region."""
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return None
    return int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1


def volume_stats_chunked(dz_arr, lod_, cell_area, mask, sigma=None, rows=512):
    """低記憶體版 DoD 統計；避免 dz[mask] 一次複製整個分析區。"""
    A = float(cell_area)
    n = ne = nd = 0
    se = sd = 0.0
    min_e = None
    max_d = None
    h, w = dz_arr.shape
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        d = dz_arr[r0:r1]
        m = mask[r0:r1] if mask is not None else np.ones(d.shape, dtype=bool)
        ok = m & np.isfinite(d)
        if not ok.any():
            continue
        v = d[ok]  # 只複製一小塊
        n += v.size
        e = v[v < -lod_]
        q = v[v > lod_]
        if e.size:
            ne += e.size
            se += float(e.sum(dtype=np.float64))
            ev = float(e.min())
            min_e = ev if min_e is None else min(min_e, ev)
        if q.size:
            nd += q.size
            sd += float(q.sum(dtype=np.float64))
            qv = float(q.max())
            max_d = qv if max_d is None else max(max_d, qv)
    ve = -se * A
    vd = sd * A
    out = {
        "區域面積_m2": n * A,
        "侵蝕面積_m2": ne * A,
        "堆積面積_m2": nd * A,
        "侵蝕體積_m3": ve,
        "堆積體積_m3": vd,
        "平均侵蝕深_m": (-se / ne) if ne else 0.0,
        "最大侵蝕深_m": (-min_e) if min_e is not None else 0.0,
        "平均堆積厚_m": (sd / nd) if nd else 0.0,
        "最大堆積厚_m": max_d if max_d is not None else 0.0,
    }
    out["淨變量_m3"] = vd - ve
    if sigma is not None and n:
        out["不確定度_隨機_m3"] = float(sigma * math.sqrt(n) * A)
        out["不確定度_系統_m3"] = float(sigma * n * A)
    return out

def scar_depth_hint_sampled(dz_arr, lod_, region, max_values=300_000, rows=512):
    """低記憶體的崩落深度統計；大區域採均勻抽樣估計中位數/P90。"""
    total = int(np.count_nonzero(region & np.isfinite(dz_arr) & (dz_arr < -lod_)))
    if total == 0:
        return None
    stride = max(1, int(math.ceil(total / max_values)))
    vals = []
    seen = 0
    h = dz_arr.shape[0]
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        d = dz_arr[r0:r1]
        m = region[r0:r1] & np.isfinite(d) & (d < -lod_)
        v = (-d[m]).astype(np.float32, copy=False)
        if v.size:
            # 以全域序位近似均勻抽樣，避免保存數百萬個值。
            take = np.arange(0, v.size, stride, dtype=np.int64)
            vals.append(v[take])
    x = np.concatenate(vals) if vals else np.empty(0, dtype=np.float32)
    if x.size > max_values:
        x = x[:max_values]
    return {"median": float(np.median(x)), "mean": float(np.mean(x, dtype=np.float64)),
            "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "n": total, "sampled": int(x.size)}


# --------------------------------------------------------------------------
# 側邊欄：資料與設定
# --------------------------------------------------------------------------
st.title("⛰️ 崩塌地形變異分析")
st.caption("兩期 DEM/DSM 差分（DoD）· 崩塌量體 · 堆積量體 · 殘餘土體 · 剖面切割")

sb = st.sidebar
sb.header("1. 上傳資料")
if sb.button("🧪 載入範例資料（合成）", width="stretch"):
    st.session_state["demo"] = True
if st.session_state.get("demo") and sb.button("結束範例模式", width="stretch"):
    st.session_state["demo"] = False
    st.rerun()

sb.caption(f"目前單檔上傳上限：{int(st.get_option('server.maxUploadSize'))} MB")
f1 = sb.file_uploader("T1（崩塌前）DEM/DSM GeoTIFF", type=["tif", "tiff"], key="f1")
f2 = sb.file_uploader("T2（崩塌後）DEM/DSM GeoTIFF", type=["tif", "tiff"], key="f2")
fo = sb.file_uploader("正射影像 GeoTIFF（選用，僅作底圖）", type=["tif", "tiff"], key="fo")

if st.session_state.get("demo"):
    if "_demo_paths" not in st.session_state:
        st.session_state["_demo_paths"] = demo_data.make_demo(os.path.join(TMP, "demo"))
    dm = st.session_state["_demo_paths"]
    p1, p2, po, demo_slip = dm["t1"], dm["t2"], dm["ortho"], dm["slip"]
    sb.info("目前為範例模式（合成資料）")
else:
    def _local(label, key):
        v = sb.text_input(label, key=key, placeholder=r"D:\\UAV\\1150706_show_dsm.tif").strip().strip('"')
        if v and not os.path.isfile(v):
            sb.warning(f"找不到檔案：{v}")
            return None
        return v or None

    with sb.expander("檔案太大？改用本機路徑（不受上傳大小限制）"):
        st.caption("僅在「自己電腦上執行 streamlit run app.py」時可用；雲端版讀不到你的硬碟。填了路徑會優先於上傳。")
    lp1 = _local("T1 本機路徑", "lp1")
    lp2 = _local("T2 本機路徑", "lp2")
    lpo = _local("正射影像 本機路徑", "lpo")
    p1 = lp1 or save_upload(f1)
    p2 = lp2 or save_upload(f2)
    po = lpo or save_upload(fo)
    demo_slip = None
    pre = st.session_state.get("pre")
    if pre and not (p1 or p2):
        p1, p2, po = pre.get("t1"), pre.get("t2"), pre.get("ortho")
        sb.success("使用「前處理」的裁切成果（左側 **① 前處理｜裁切與縮小** 可重新裁切）")

sb.header("2. 計算設定")
factor = sb.slider("降採樣倍率", 1, 20, 4, help="倍率越高越省記憶體、較不容易因大檔當機，但空間解析度會降低。建議大範圍資料先用 4 倍。")
nd_txt = sb.text_input("額外 NoData 值（選用）", "", help="若 DEM 未宣告 NoData 但用 -9999 之類填空，請在此輸入。")
try:
    nd_override = float(nd_txt) if nd_txt.strip() else None
except ValueError:
    nd_override = None
    sb.warning("NoData 值格式錯誤，已忽略。")

corr_label = sb.radio(
    "兩期對位校正（需先畫穩定區）",
    ["不校正", "平移（中位數）", "平面（平移＋傾斜）"],
    index=0,
    help="穩定區 = 兩期之間地形沒有變動的地方（岩盤、道路、建物屋頂等）。",
)
corr_mode = {"不校正": "none", "平移（中位數）": "offset", "平面（平移＋傾斜）": "plane"}[corr_label]

if not (p1 and p2):
    st.info(
        "👈 請在左側上傳 **T1（崩塌前）** 與 **T2（崩塌後）** 的 DEM/DSM（GeoTIFF，需為投影座標系，單位：公尺），"
        "或到左側的 **① 前處理｜裁切與縮小** 頁先裁切/縮小大檔，或按「載入範例資料」先試用。\n\n"
        "使用流程：① 上傳 → ② 在「範圍與剖面線」視需要畫實際崩塌/潛在滑動體/堆積區/穩定區 → ③ 看「差異與量體」→ "
        "④ 在「殘餘土體」選滑動面假設 → ⑤ 「剖面」→ ⑥ 「匯出」。"
    )
    st.stop()

# --------------------------------------------------------------------------
# 載入 / 對齊
# --------------------------------------------------------------------------
dkey = ("data", p1, p2, factor, nd_override)
if st.session_state.get("_dkey") != dkey:
    st.session_state.pop("_data", None)
    st.session_state["_dkey"] = dkey
try:
    if "_data" not in st.session_state:
        with st.spinner("讀取並對齊兩期 DEM…"):
            st.session_state["_data"] = core.load_pair(p1, p2, factor, nd_override)
    z1, z2, grid = st.session_state["_data"]
except Exception as e:  # noqa: BLE001
    st.error(f"讀取失敗：{e}")
    st.stop()

CA = grid.cell_area
valid0 = np.isfinite(z1) & np.isfinite(z2)
if not valid0.any():
    st.error("兩期 DEM 沒有同時有效的像元，請檢查 NoData 與範圍。")
    st.stop()

with sb.expander("資料資訊"):
    i1, i2 = core.raster_info(p1), core.raster_info(p2)
    st.write(f"**CRS**：{i1['crs']}")
    st.write(f"T1：{i1['width']}×{i1['height']}，解析度 {i1['res'][0]:.3g} m")
    st.write(f"T2：{i2['width']}×{i2['height']}，解析度 {i2['res'][0]:.3g} m")
    st.write(f"分析網格：{grid.shape[1]}×{grid.shape[0]}，格距 {grid.cell_x:.3g} m，"
             f"範圍 {valid0.sum() * CA / 1e4:,.2f} 公頃")
    if i1["crs"] != i2["crs"]:
        st.warning("兩期 CRS 不同，已自動將 T2 重投影到 T1。")

# --------------------------------------------------------------------------
# 範圍檔上傳 + 範圍遮罩
# --------------------------------------------------------------------------
with sb.expander("3. 範圍檔上傳（選用；也可在地圖上畫）"):
    st.caption("支援 GeoJSON、或 shapefile 壓縮成 .zip。需含座標系統。")
    for z in POLY_ZONES:
        st.file_uploader(ZONES[z][0], type=["geojson", "json", "zip"], key=f"up_{z}")


def read_vector_geoms(uf):
    path = save_upload(uf)
    src = f"zip://{path}" if uf.name.lower().endswith(".zip") else path
    gdf = gpd.read_file(src)
    if gdf.crs is None:
        raise ValueError(f"{uf.name} 沒有座標系統")
    gdf = gdf.to_crs(grid.crs)
    return [mapping(g) for g in gdf.geometry if g is not None and g.geom_type in ("Polygon", "MultiPolygon")]


def zone_sig(z):
    feats = json.dumps(st.session_state[f"geo_{z}"], sort_keys=True)
    uf = st.session_state.get(f"up_{z}")
    return hashlib.md5((feats + (f"{uf.name}{uf.size}" if uf else "")).encode()).hexdigest()


def zone_geoms(z):
    geoms = []
    for f in st.session_state[f"geo_{z}"]:
        if f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            geoms.append(transform_geom("EPSG:4326", grid.crs, f["geometry"]))
    uf = st.session_state.get(f"up_{z}")
    if uf is not None:
        try:
            geoms += read_vector_geoms(uf)
        except Exception as e:  # noqa: BLE001
            st.warning(f"範圍檔 {uf.name} 讀取失敗：{e}")
    return geoms


def zone_mask(z):
    return memo(("mask", z, zone_sig(z), dkey), lambda: core.rasterize(zone_geoms(z), grid.shape, grid.transform))


m_collapse, m_potential, m_dep, m_stab, m_exc = (zone_mask(z) for z in ("collapse", "potential", "deposit", "stable", "exclude"))
valid = valid0 & ~m_exc if m_exc is not None else valid0

# --------------------------------------------------------------------------
# 差分 + 對位校正
# --------------------------------------------------------------------------
dz_raw = z2 - z1
stab_valid = (m_stab & valid) if m_stab is not None else None
cstats = {}
corr = 0.0
if corr_mode != "none":
    if stab_valid is None:
        sb.warning("尚未畫「穩定區」，無法校正，已暫時不校正。")
    else:
        try:
            corr, cstats = memo(("corr", dkey, zone_sig("stable"), zone_sig("exclude"), corr_mode),
                                lambda: core.fit_correction(dz_raw, stab_valid, grid, corr_mode))
        except Exception as e:  # noqa: BLE001
            sb.warning(f"校正失敗：{e}")
dz = (dz_raw - corr).astype("float32")
del dz_raw
z2c = None
dz[~valid] = np.nan

sigma = smean = None
if stab_valid is not None and stab_valid.sum() >= 50:
    sv = dz[stab_valid]
    sigma, smean = float(np.std(sv)), float(np.mean(sv))

sb.header("4. 偵測門檻")
sb.number_input("LoD 偵測門檻 (m)", 0.0, 50.0, 0.30, 0.05, key="lod",
                help="|dz| 小於 LoD 視為雜訊、不計入體積。")
lod = float(st.session_state["lod"])
if sigma is not None:
    sugg = abs(smean) + 1.96 * sigma
    sb.caption(f"穩定區 dz：平均 {smean:+.3f} m、σ {sigma:.3f} m → 建議 LoD ≈ {sugg:.2f} m（|平均|+1.96σ）")
    sb.button("套用建議 LoD", on_click=lambda: st.session_state.update(lod=round(sugg, 2)))
else:
    sb.caption("畫出「穩定區」後，這裡會給 LoD 建議值。")
sb.select_slider("地圖預覽解析度（px，越小越不易卡）", options=[500, 700, 900, 1200, 1600], value=900, key="map_px")
sb.number_input("色階範圍 ±(m)", 0.5, 100.0, 5.0, 0.5, key="vmax")
vmax = float(st.session_state["vmax"])

# 範圍定義
# DoD 永遠先在整個有效分析區計算；兩種圈繪範圍只控制後續統計/殘餘分析。
potential_region = (m_potential & valid) if m_potential is not None else None
if m_collapse is not None:
    actual_collapse_region = (m_collapse & valid) & (dz < -lod)
else:
    actual_collapse_region = valid & (dz < -lod)

if m_dep is not None:
    dep_region = m_dep & valid
elif m_potential is not None:
    dep_region = valid & ~m_potential
else:
    dep_region = valid

# --------------------------------------------------------------------------
# 分頁：只計算目前工作區需要的統計，避免每次點按鈕都建立大型暫存陣列。
# --------------------------------------------------------------------------
section = st.radio(
    "分析工作區",
    ["🗺️ 範圍與剖面線", "📊 差異與量體", "🧱 殘餘土體", "📈 剖面", "💾 匯出"],
    horizontal=True,
    key="analysis_section",
)
S_src = S_dep = S_all = None
if section in ("📊 差異與量體", "💾 匯出"):
    S_src = volume_stats_chunked(dz, lod, CA, actual_collapse_region, sigma)
    S_dep = volume_stats_chunked(dz, lod, CA, dep_region, sigma)
    S_all = volume_stats_chunked(dz, lod, CA, valid, sigma)
elif section == "🧱 殘餘土體":
    S_dep = volume_stats_chunked(dz, lod, CA, dep_region, sigma)

# ======================= 差異與量體 =======================
if section == "📊 差異與量體":
    if m_collapse is None:
        st.info("未圈繪「實際崩塌範圍」：目前以整個有效分析區的 DoD（dz < -LoD）自動統計。若只要統計特定崩塌事件，可圈繪實際崩塌範圍。")
    c = st.columns(4)
    c[0].metric("崩塌（侵蝕）體積", fmt_m3(S_src["侵蝕體積_m3"]))
    c[1].metric("堆積體積", fmt_m3(S_dep["堆積體積_m3"]))
    c[2].metric("崩塌面積", fmt_m2(S_src["侵蝕面積_m2"]))
    c[3].metric("最大崩落深度", f"{S_src['最大侵蝕深_m']:.2f} m")

    st.markdown("#### 質量守恆檢核")
    bulk = st.number_input("膨脹係數（崩落土石鬆動後體積 ÷ 原地體積）", 1.0, 2.0, 1.2, 0.05, key="bulk",
                           help="土砂約 1.1–1.3、岩塊約 1.3–1.6。設 1.0 即原地體積直接比較。")
    expect_dep = S_src["侵蝕體積_m3"] * bulk
    diff = expect_dep - S_dep["堆積體積_m3"]
    st.write(
        f"崩落體積 × {bulk:.2f} = **{fmt_m3(expect_dep)}**；範圍內實測堆積 **{fmt_m3(S_dep['堆積體積_m3'])}**；"
        f"差值 **{fmt_m3(diff)}**（{'可能已被水流/下游帶走或超出範圍' if diff > 0 else '堆積多於崩落：可能含上游來源、膨脹係數偏低或對位誤差'}）。"
    )

    st.markdown("#### 統計表")
    df = pd.DataFrame({"實際崩塌（DoD）": S_src, "堆積區（堆積統計）": S_dep, "全區（參考）": S_all})
    st.dataframe(df.style.format("{:,.2f}"), width="stretch")
    if sigma is not None:
        st.caption("不確定度：隨機 = σ√N·A（下限）、系統 = σ·N·A（上限，保守）。實際介於兩者之間。")
    else:
        st.caption("未畫穩定區，無法估計不確定度。")

    if cstats:
        st.markdown("#### 對位校正結果（穩定區）")
        cc = st.columns(4)
        cc[0].metric("校正前 平均/σ", f"{cstats['before_mean']:+.3f} / {cstats['before_std']:.3f} m")
        cc[1].metric("校正後 平均/σ", f"{cstats['after_mean']:+.3f} / {cstats['after_std']:.3f} m")
        cc[2].metric("校正前 RMSE", f"{cstats['before_rmse']:.3f} m")
        cc[3].metric("校正後 RMSE", f"{cstats['after_rmse']:.3f} m")
        if cstats["mode"] == "plane":
            st.caption(f"傾斜係數：dz/dx = {cstats['slope_x']:.2e}，dz/dy = {cstats['slope_y']:.2e}（m/m）")

    st.markdown("#### dz 差異圖")
    ds = max(1, math.ceil(max(grid.shape) / 1000))
    dzd = dz[::ds, ::ds]
    hsd = memo(("hs_plot", dkey, ds), lambda: core.hillshade(z2[::ds, ::ds], grid.cell_x * ds))
    ext = grid.extent
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(hsd, cmap="gray", extent=ext, vmin=0, vmax=1)
    show = np.isfinite(dzd) & (np.abs(dzd) > lod)
    im = ax.imshow(np.ma.masked_where(~show, dzd), cmap="RdBu", vmin=-vmax, vmax=vmax, extent=ext, alpha=0.8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("dz = T2 - T1 (m)   red: erosion / blue: deposition")
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    h1, h2 = st.columns(2)
    sample = dz[valid][:: max(1, int(valid.sum() // 200_000))]
    fh = go.Figure(go.Histogram(x=sample, nbinsx=120, marker_color="#546e7a"))
    fh.add_vline(x=-lod, line_dash="dash", line_color="red")
    fh.add_vline(x=lod, line_dash="dash", line_color="blue")
    fh.update_layout(title="全區 dz 分布", xaxis_title="dz (m)", yaxis_type="log", height=300, margin=dict(t=40, b=30))
    h1.plotly_chart(fh, width="stretch")
    if stab_valid is not None and stab_valid.sum() >= 50:
        ss = dz[stab_valid]
        fs = go.Figure(go.Histogram(x=ss[:: max(1, len(ss) // 100_000)], nbinsx=80, marker_color="#43a047"))
        fs.update_layout(title="穩定區 dz 分布（應接近 0）", xaxis_title="dz (m)", height=300, margin=dict(t=40, b=30))
        h2.plotly_chart(fs, width="stretch")

# 基準深度參考 → 下方數值同步
def _sync_depth_from_basis():
    hint_ = st.session_state.get("_depth_hint")
    basis_ = st.session_state.get("depth_basis", "自訂")
    if basis_.startswith("中位數") and hint_:
        v = hint_["median"]
    elif basis_ == "平均" and hint_:
        v = hint_["mean"]
    elif basis_.startswith("P90") and hint_:
        v = hint_["p90"]
    else:
        v = 2.0
    st.session_state["d_uni"] = max(round(float(v), 1), 0.5)

# ======================= 殘餘土體 =======================
slip = None
thick = None
res_sum = None
sens_df = None
slip_label = ""
if section == "🧱 殘餘土體":
    st.markdown("### A. 堆積土體（崩落後堆積在坡腳）")
    ca = st.columns(4)
    ca[0].metric("堆積體積", fmt_m3(S_dep["堆積體積_m3"]))
    ca[1].metric("堆積面積", fmt_m2(S_dep["堆積面積_m2"]))
    ca[2].metric("平均堆積厚", f"{S_dep['平均堆積厚_m']:.2f} m")
    ca[3].metric("最大堆積厚", f"{S_dep['最大堆積厚_m']:.2f} m")

    st.markdown("### B. 潛在滑動體內殘餘不穩定土體")
    if m_potential is None:
        st.warning("請先到「範圍與剖面線」圈繪完整的「潛在滑動體範圍」。本頁不會在未定義範圍時自動對整張 DEM 計算。")
    else:
        hint = scar_depth_hint_sampled(dz, lod, actual_collapse_region)
        if hint:
            st.markdown("#### 已崩落深度統計參考（DoD 實測，不是滑動面深度）")
            hc = st.columns(4)
            hc[0].metric("中位數", f"{hint['median']:.2f} m")
            hc[1].metric("平均", f"{hint['mean']:.2f} m")
            hc[2].metric("P90", f"{hint['p90']:.2f} m")
            hc[3].metric("最大", f"{hint['max']:.2f} m")
            st.caption("以上是實際崩塌範圍內的 DoD 地表降低深度統計，只能作為滑動面深度假設的參考。")

        methods = ["等深度假設（建議：基準情境 + 敏感度）", "DoD 幾何推估滑動面（研究性）", "上傳滑動面（高程或深度 raster）"]
        default_idx = 2 if demo_slip else 0
        method = st.radio("滑動面來源 / 假設", methods, index=default_idx, key="slip_method")
        err = None
        depth = None
        depth_basis = "自訂"
        if method == methods[0]:
            opts = ["中位數（建議基準情境）", "平均", "P90（較保守情境）", "自訂"]
            default_opt = 0 if hint else 3
            if "depth_basis" not in st.session_state:
                st.session_state["depth_basis"] = opts[default_opt]
            if "d_uni" not in st.session_state:
                if hint:
                    st.session_state["d_uni"] = max(round(float(hint["median"]), 1), 0.5)
                else:
                    st.session_state["d_uni"] = 2.0
            st.session_state["_depth_hint"] = hint
            depth_basis = st.selectbox(
                "基準深度參考",
                opts,
                key="depth_basis",
                on_change=_sync_depth_from_basis,
                help="選擇中位數、平均或 P90 後，下方深度會同步帶入該數值；帶入後仍可手動修改。",
            )
            st.caption("選擇參考值後會同步帶入下方；下方數字仍可自行調整。")
            depth = st.number_input(
                "滑動面深度 d（m，自 T1 地表往下）",
                min_value=0.1, max_value=200.0, step=0.5, key="d_uni",
            )
            slip_label = f"等深度基準情境 d={depth:g} m（參考：{depth_basis}）"
        elif method == methods[1]:
            # DoD 幾何推估：只顯示此方法自己的設定，不要顯示上傳滑動面元件。
            st.info("以實際崩塌（dz < -LoD）的觀測深度，搭配潛在滑動體邊界，幾何內插出滑動面。")
            st.caption("選擇此方法後，按下「計算殘餘土體」才會執行 DoD 幾何推估。")

        else:  # 上傳滑動面
            up = st.file_uploader("滑動面 GeoTIFF", type=["tif", "tiff"], key="slip_up")
            sp = save_upload(up) or demo_slip
            kind = st.radio("檔案內容", ["高程（m，同 DEM 基準面）", "深度（m，自 T1 地表往下）"], horizontal=True, key="slip_kind")
            fb = st.checkbox("滑動面缺值處，以等深度假設補足", value=True, key="slip_fb")
            d_fb = st.number_input("補足用深度 (m)", 0.1, 200.0, 2.0, 0.5, key="d_fb", disabled=not fb)
            if sp is None:
                err = "請上傳滑動面 GeoTIFF。"

        res_sig = (dkey, zone_sig("potential"), zone_sig("collapse"), zone_sig("exclude"), corr_mode, round(lod,4), method,
                   depth, depth_basis, st.session_state.get("slip_up").name if st.session_state.get("slip_up") else None,
                   st.session_state.get("slip_kind"), st.session_state.get("slip_fb"), st.session_state.get("d_fb"))
        stored = st.session_state.get("residual_result")
        ready = stored is not None and stored.get("sig") == res_sig
        if not ready:
            st.caption("設定完成後，按「計算殘餘土體」才會執行計算。")
        if st.button("▶ 計算殘餘土體", type="primary", disabled=bool(err), key="calc_residual"):
            try:
                region = potential_region
                if method == methods[0]:
                    # 分塊計算：不建立整張 vals/t 暫存陣列，避免大 DEM 在按鈕計算時瞬間吃滿 RAM。
                    d0 = float(depth)
                    total_v = total_pos = total_sum = 0.0
                    total_n = 0
                    max_t = 0.0
                    for r0 in range(0, dz.shape[0], 512):
                        r1 = min(dz.shape[0], r0 + 512)
                        dd = dz[r0:r1]
                        rr = region[r0:r1] & np.isfinite(dd)
                        if not rr.any():
                            continue
                        vv = dd[rr].astype(np.float32, copy=False)
                        tt = np.maximum(vv + d0, 0.0)
                        total_v += float(tt.sum(dtype=np.float64))
                        total_pos += float(np.count_nonzero(tt > 0))
                        total_sum += float(tt[tt > 0].sum(dtype=np.float64)) if np.any(tt > 0) else 0.0
                        total_n += vv.size
                        if tt.size:
                            max_t = max(max_t, float(tt.max()))
                    result = {"sig":res_sig, "method":"uniform", "depth":d0,
                              "slip_label":slip_label, "volume":total_v*CA,
                              "area":total_pos*CA, "region_area":total_n*CA,
                              "mean":(total_sum/total_pos) if total_pos else 0.0, "max":max_t}
                elif method == methods[1]:
                    z2c = (z2 - corr).astype("float32")
                    n_region = int(region.sum())
                    if n_region > 4_000_000:
                        raise ValueError("目前分析網格仍過大（潛在滑動體超過 400 萬格）。請先把左側「降採樣倍率」提高到 6～8 倍，再執行 DoD 幾何推估；這可避免 Streamlit 記憶體不足而整頁當機。")
                    slip_arr = core.slip_interpolated(z1, z2c, dz, lod, region)
                    thick_arr = core.residual_thickness(z2c, slip_arr, region)
                    rs = core.residual_summary(thick_arr, CA, region)
                    result = {"sig":res_sig, "method":"raster", "slip":slip_arr, "thick":thick_arr, "slip_label":"DoD 幾何推估滑動面（研究性）", **{"volume":rs["殘餘土體體積_m3"],"area":rs["有殘餘土體面積_m2"],"region_area":rs["範圍面積_m2"],"mean":rs["平均厚度_m"],"max":rs["最大厚度_m"]}}
                else:
                    z2c = (z2 - corr).astype("float32")
                    sp = save_upload(st.session_state.get("slip_up")) or demo_slip
                    if not sp: raise ValueError("請上傳滑動面 GeoTIFF。")
                    if int(region.sum()) > 4_000_000:
                        raise ValueError("目前分析網格仍過大（潛在滑動體超過 400 萬格）。請先提高左側「降採樣倍率」後再套用上傳滑動面。")
                    s_arr = core.warp_band_to_grid(sp, grid, nodata_override=None)
                    slip_arr = (z1 - s_arr) if st.session_state.get("slip_kind", "").startswith("深度") else s_arr.copy()
                    n_missing = int((~np.isfinite(slip_arr) & region).sum())
                    if st.session_state.get("slip_fb", True):
                        miss = ~np.isfinite(slip_arr) & np.isfinite(z1)
                        slip_arr[miss] = z1[miss] - np.float32(st.session_state.get("d_fb", 2.0))
                    thick_arr = core.residual_thickness(z2c, slip_arr, region)
                    rs = core.residual_summary(thick_arr, CA, region)
                    label = f"上傳滑動面（缺值 {n_missing * CA:,.0f} m² 以 d={st.session_state.get('d_fb',2.0):g} m 補足）" if st.session_state.get("slip_fb", True) and n_missing else "上傳滑動面"
                    result = {"sig":res_sig, "method":"raster", "slip":slip_arr, "thick":thick_arr, "slip_label":label, **{"volume":rs["殘餘土體體積_m3"],"area":rs["有殘餘土體面積_m2"],"region_area":rs["範圍面積_m2"],"mean":rs["平均厚度_m"],"max":rs["最大厚度_m"]}}
                st.session_state["residual_result"] = result
                st.rerun()
            except Exception as e:
                st.error(f"殘餘土體計算失敗：{e}")
        stored = st.session_state.get("residual_result")
        if stored is not None and stored.get("sig") == res_sig:
            cr = st.columns(4)
            cr[0].metric("殘餘土體體積", fmt_m3(stored["volume"]))
            cr[1].metric("有殘餘土體面積", fmt_m2(stored["area"]))
            cr[2].metric("平均厚度", f"{stored['mean']:.2f} m")
            cr[3].metric("最大厚度", f"{stored['max']:.2f} m")
            st.caption(f"滑動面來源：**{stored['slip_label']}**")
            if stored.get("method") == "uniform":
                st.caption("等深度情境的殘餘量直接由 DoD 的 dz + d 計算，不建立整張 slip/thickness raster，可大幅降低記憶體使用量。")

        st.markdown("#### 敏感度分析")
        if st.session_state.get("residual_result", {}).get("sig") == res_sig:
            sc = st.columns(3)
            sens_min_default = max(0.5, round(hint["median"], 1)) if hint else 1.0
            sens_max_default = max(sens_min_default + 0.5, round(hint["p90"], 1)) if hint else 10.0
            dmin = sc[0].number_input("最小情境深度 (m)", 0.1, 100.0, sens_min_default, 0.5, key="s_min")
            dmax = sc[1].number_input("最大情境深度 (m)", 0.2, 200.0, sens_max_default, 0.5, key="s_max")
            dstep = sc[2].number_input("情境間隔 (m)", 0.1, 20.0, 0.5, 0.1, key="s_step")
            if st.button("▶ 執行敏感度分析", key="run_sensitivity"):
                depths = np.arange(dmin, dmax + 1e-9, dstep)[:60]
                # 分塊累計：避免 vals.sort()/cumsum() 建立數百 MB 的暫存陣列。
                vol_sum = np.zeros(len(depths), dtype=np.float64)
                pos_n = np.zeros(len(depths), dtype=np.int64)
                for r0 in range(0, dz.shape[0], 512):
                    r1 = min(dz.shape[0], r0 + 512)
                    dd0 = dz[r0:r1]
                    rr = potential_region[r0:r1] & np.isfinite(dd0)
                    if not rr.any():
                        continue
                    vv = dd0[rr].astype(np.float32, copy=False)
                    for j, dd in enumerate(depths):
                        tt = vv + float(dd)
                        pos = tt > 0
                        if np.any(pos):
                            vol_sum[j] += float(tt[pos].sum(dtype=np.float64))
                            pos_n[j] += int(np.count_nonzero(pos))
                rows=[{"假設滑動面深度_m":float(dd),
                       "殘餘體積_m3":float(vol_sum[j]*CA),
                       "殘餘面積_m2":float(pos_n[j]*CA)} for j, dd in enumerate(depths)]
                st.session_state["sensitivity_result"] = {"sig":res_sig,"df":pd.DataFrame(rows)}
            sr = st.session_state.get("sensitivity_result")
            if sr and sr.get("sig") == res_sig:
                sens_df = sr["df"]
                fs = go.Figure(go.Scatter(x=sens_df["假設滑動面深度_m"], y=sens_df["殘餘體積_m3"], mode="lines+markers"))
                fs.update_layout(xaxis_title="情境滑動面深度 d (m)", yaxis_title="殘餘土體體積 (m³)", height=320, margin=dict(t=20,b=30))
                st.plotly_chart(fs, width="stretch")
                with st.expander("敏感度表"):
                    st.dataframe(sens_df.style.format("{:,.1f}"), width="stretch")
        else:
            st.caption("先完成一次殘餘土體計算，才可執行敏感度分析。")

# ======================= 剖面 =======================
profile_rows = None
profile_series = None
if section == "📈 剖面":
    lines = []
    for i, f in enumerate(st.session_state["geo_profile"]):
        g = f["geometry"]
        if g["type"] == "LineString":
            g2 = transform_geom("EPSG:4326", grid.crs, g)
            lines.append((f"地圖剖面 {i + 1}", g2["coordinates"]))
    with st.expander("手動輸入剖面端點（與 DEM 同座標系）", expanded=not lines):
        l_, r_, b_, t_ = grid.extent
        mc = st.columns(4)
        e1 = mc[0].number_input("E1", value=float(l_ + 0.5 * (r_ - l_)), format="%.2f", key="e1")
        n1 = mc[1].number_input("N1", value=float(t_ - 0.1 * (t_ - b_)), format="%.2f", key="n1")
        e2 = mc[2].number_input("E2", value=float(l_ + 0.5 * (r_ - l_)), format="%.2f", key="e2")
        n2 = mc[3].number_input("N2", value=float(b_ + 0.1 * (t_ - b_)), format="%.2f", key="n2")
        if st.checkbox("加入手動剖面", value=not lines, key="use_manual"):
            lines.append(("手動剖面", [(e1, n1), (e2, n2)]))
    if not lines:
        st.info("請到第一個分頁選「剖面線」畫線並儲存，或使用上方手動輸入。")
    else:
        pc = st.columns(3)
        name = pc[0].selectbox("選擇剖面", [n for n, _ in lines])
        step = pc[1].number_input("取樣間距 (m)", 0.01, 100.0, float(grid.cell_x), 0.1, key="pstep")
        ve = pc[2].number_input("垂直誇大倍率", 0.5, 20.0, 1.0, 0.5, key="ve")
        coords = dict(lines)[name]
        try:
            z2c = (z2 - corr).astype("float32")
            dist, xs, ys = core.sample_polyline(coords, step)
            g1 = core.sample_raster(z1, grid.transform, xs, ys)
            g2 = core.sample_raster(z2c, grid.transform, xs, ys)
            stored_res = st.session_state.get("residual_result")
            slip_for_profile = stored_res.get("slip") if stored_res and stored_res.get("method") == "raster" else None
            gs = core.sample_raster(slip_for_profile, grid.transform, xs, ys) if slip_for_profile is not None else None
            if gs is not None:
                gs = np.where(np.isfinite(gs), gs, np.nan)
            pa = core.profile_areas(dist, g1, g2, lod, gs)

            base = np.fmin(g1, g2)
            ero_top = np.where(g1 > g2, g1, base)
            dep_top = np.where(g2 > g1, g2, base)
            fig = go.Figure()
            def helper(y):
                return go.Scatter(x=dist, y=y, showlegend=False, hoverinfo="skip", line=dict(width=0))

            def filled(y, color, nm):
                return go.Scatter(x=dist, y=y, fill="tonexty", fillcolor=color, name=nm, hoverinfo="skip",
                                  line=dict(width=0))

            fig.add_trace(helper(base))
            fig.add_trace(filled(ero_top, "rgba(229,57,53,0.45)", "侵蝕"))
            fig.add_trace(helper(base))
            fig.add_trace(filled(dep_top, "rgba(30,136,229,0.45)", "堆積"))
            if gs is not None:
                fig.add_trace(helper(np.fmin(gs, g2)))
                fig.add_trace(filled(g2, "rgba(161,136,127,0.35)", "殘餘土體"))
                fig.add_trace(go.Scatter(x=dist, y=gs, name="滑動面", line=dict(color="#6d4c41", dash="dash", width=2)))
            fig.add_trace(go.Scatter(x=dist, y=g1, name="T1 崩塌前", line=dict(color="#2e7d32", width=2)))
            fig.add_trace(go.Scatter(x=dist, y=g2, name="T2 崩塌後", line=dict(color="#c62828", width=2)))
            fig.update_layout(xaxis_title="距離 (m)", yaxis_title="高程 (m)", height=520, hovermode="x unified",
                              yaxis=dict(scaleanchor="x", scaleratio=ve), margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")
            mm = st.columns(len(pa))
            for col, (k, v) in zip(mm, pa.items()):
                col.metric(k.replace("_m2", " (m²)").replace("_m", " (m)"), f"{v:,.1f}")
            if stored_res is not None:
                st.caption(f"滑動面：{stored_res.get('slip_label', '')}")
            profile_series = dict(T1_GROUND=g1, T2_GROUND=g2, SLIP=gs)
            profile_rows = pd.DataFrame({"距離_m": dist, "E": xs, "N": ys, "T1_高程": g1, "T2_高程(校正後)": g2,
                                         "dz": g2 - g1, "滑動面": gs if gs is not None else np.nan})
            dl = st.columns(2)
            dl[0].download_button("下載剖面資料（CSV）", profile_rows.to_csv(index=False).encode("utf-8-sig"),
                                  file_name="剖面資料.csv", mime="text/csv")
            try:
                dl[1].download_button("下載剖面圖資（DXF）", core.profile_to_dxf(dist, profile_series),
                                      file_name="剖面圖資.dxf", mime="application/dxf")
            except Exception as e:  # noqa: BLE001
                dl[1].caption(f"DXF 匯出不可用：{e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"剖面計算失敗：{e}")

# ======================= 匯出 =======================
if section == "💾 匯出":
    st.write("按下按鈕才會產生檔案（大範圍時需要數秒）。")
    if st.button("產生匯出檔案", key="mk_exp"):
        out = {}
        out["DoD差異高程.tif"] = core.array_to_geotiff_bytes(dz, grid)
        stored_res = st.session_state.get("residual_result")
        if stored_res and stored_res.get("thick") is not None:
            out["殘餘土體厚度.tif"] = core.array_to_geotiff_bytes(stored_res["thick"], grid)
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as xw:
            pd.DataFrame({"項目": ["降採樣倍率", "格距(m)", "LoD(m)", "校正模式", "膨脹係數", "滑動面來源"],
                          "值": [factor, grid.cell_x, lod, corr_mode, st.session_state.get("bulk", ""), slip_label]}
                         ).to_excel(xw, sheet_name="參數", index=False)
            pd.DataFrame({"實際崩塌（DoD）": S_src, "堆積區": S_dep, "全區": S_all}).to_excel(xw, sheet_name="崩塌與堆積統計")
            if stored_res:
                pd.DataFrame({"殘餘土體": {"殘餘土體體積_m3": stored_res.get("volume",0), "有殘餘土體面積_m2": stored_res.get("area",0), "平均厚度_m": stored_res.get("mean",0), "最大厚度_m": stored_res.get("max",0)}}).to_excel(xw, sheet_name="殘餘土體分析")
            if sens_df is not None:
                sens_df.to_excel(xw, sheet_name="深度敏感度", index=False)
            if cstats:
                pd.DataFrame({"校正": cstats}).to_excel(xw, sheet_name="對位校正結果")
            if profile_rows is not None:
                profile_rows.to_excel(xw, sheet_name="剖面", index=False)
        out["分析結果.xlsx"] = bio.getvalue()
        st.session_state["exports"] = out
    for fn, data in st.session_state.get("exports", {}).items():
        st.download_button(f"下載 {fn}", data, file_name=fn, key=f"dl_{fn}")
    st.caption("GeoTIFF 與輸入 DEM 同座標系；dz = T2(校正後) − T1，NoData = −9999。")


# ======================= 地圖與繪製 =======================
def build_overlays():
    ds_ = max(1, math.ceil(max(grid.shape) / int(st.session_state.get("map_px", 900))))
    dg = core.display_grid(grid, ds_)
    items = []

    def add(name, rgba, show):
        w, b = core.to_wgs84(rgba, dg.transform, dg.crs)
        items.append((name, core.png_uri(w), b, show))

    if po:
        add("正射影像", memo(("ortho", po, dkey, ds_), lambda: core.ortho_rgba(po, dg)), True)
    hs = memo(("hs_map", dkey, ds_), lambda: core.hillshade(z2[::ds_, ::ds_], grid.cell_x * ds_))
    add("T2 地形陰影", core.hillshade_rgba(hs), not po)
    add("dz 差異（紅=侵蝕／藍=堆積）", core.colorize_dz(dz[::ds_, ::ds_], lod, vmax), True)
    stored_res = st.session_state.get("residual_result")
    if stored_res and stored_res.get("thick") is not None:
        th = stored_res["thick"]
        add("殘餘土體厚度", core.colorize_thickness(th[::ds_, ::ds_], max(np.nanmax(th), 0.1)), False)
    return items


if section == "🗺️ 範圍與剖面線":
    zone = st.radio("目前要繪製的類型", list(ZONES), format_func=lambda z: ZONES[z][0], horizontal=True, key="draw_zone")
    st.caption("🔴 實際崩塌＝DoD 已發生變化；🟠 潛在滑動體＝後續殘餘土體評估；🔵 堆積區＝崩落土體堆積位置。三者用途分開，不互相取代。畫完後按「💾 儲存」。")
    with st.spinner("準備地圖圖層…"):
        overlays = memo(("overlays", dkey, zone_sig("exclude"), zone_sig("stable"), corr_mode, round(lod, 4), vmax,
                         slip_label, st.session_state.get("map_px"), zone_sig("potential"), zone_sig("collapse"),
                         str(st.session_state.get("residual_result", {}).get("sig", ""))),
                        build_overlays)
    south, west = overlays[0][2][0]
    north, east = overlays[0][2][1]
    m = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=16, tiles=None,
                   control_scale=True, max_zoom=22)

    # Leaflet Draw 的地圖是在 st_folium iframe 內渲染；CSS 必須注入 Folium 地圖本身，
    # 不能只放在 Streamlit 外層頁面。這裡沿用先前實際顯示效果較好的 6 px 控制點。
    folium.Element("""
    <style>
      .leaflet-editing-icon,
      .leaflet-vertex-icon {
        width: 6px !important;
        height: 6px !important;
        margin-left: -3px !important;
        margin-top: -3px !important;
        border-width: 1px !important;
        box-sizing: border-box !important;
      }
      .leaflet-touch .leaflet-editing-icon,
      .leaflet-touch .leaflet-vertex-icon {
        width: 7px !important;
        height: 7px !important;
        margin-left: -3.5px !important;
        margin-top: -3.5px !important;
      }
    </style>
    """).add_to(m.get_root().header)

    folium.TileLayer("OpenStreetMap", name="OSM", max_zoom=22, max_native_zoom=19).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Esri 衛星", max_zoom=22, max_native_zoom=19).add_to(m)
    for nm, uri, bnds, shw in overlays:
        folium.raster_layers.ImageOverlay(image=uri, bounds=bnds, name=nm, show=shw, opacity=1.0).add_to(m)
    for z, (label, color, _) in ZONES.items():
        feats = st.session_state[f"geo_{z}"]
        if feats:
            stl = ZONE_DRAW_STYLES[z]
            folium.GeoJson(
                {"type": "FeatureCollection", "features": feats},
                name=f"已存：{label}",
                style_function=lambda _f, s=stl: dict(s),
                tooltip=label,
            ).add_to(m)
    draw_style = ZONE_DRAW_STYLES[zone].copy()
    shape_opts = {"shapeOptions": draw_style}
    if ZONES[zone][2] == "line":
        draw_opts = dict(polyline=shape_opts, polygon=False, rectangle=False)
    else:
        draw_opts = dict(polyline=False, polygon=shape_opts, rectangle=shape_opts)
    Draw(export=False, draw_options=dict(circle=False, marker=False, circlemarker=False, **draw_opts),
         edit_options={"edit": True}).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    m.fit_bounds(overlays[0][2])

    out = st_folium(m, key=f"map_{zone}_{st.session_state[f'ver_{zone}']}", height=620,
                    use_container_width=True, returned_objects=["all_drawings"])
    cur = (out or {}).get("all_drawings") or []
    want = ("LineString",) if ZONES[zone][2] == "line" else ("Polygon", "MultiPolygon")
    cur = [f for f in cur if f.get("geometry", {}).get("type") in want]

    bc = st.columns(3)
    if bc[0].button(f"💾 儲存剛畫的 {len(cur)} 個圖形", disabled=not cur, width="stretch"):
        st.session_state[f"geo_{zone}"] += cur
        st.session_state[f"ver_{zone}"] += 1
        st.rerun()
    if bc[1].button("↩️ 刪除此類型最後一個", disabled=not st.session_state[f"geo_{zone}"], width="stretch"):
        st.session_state[f"geo_{zone}"].pop()
        st.session_state[f"ver_{zone}"] += 1
        st.rerun()
    if bc[2].button("🗑️ 清除此類型全部", disabled=not st.session_state[f"geo_{zone}"], width="stretch"):
        st.session_state[f"geo_{zone}"] = []
        st.session_state[f"ver_{zone}"] += 1
        st.rerun()

    cnt = {ZONES[z][0]: len(st.session_state[f"geo_{z}"]) + (1 if st.session_state.get(f"up_{z}") else 0)
           for z in ZONES}
    st.write("已儲存數量：" + "　".join(f"**{k}** {v}" for k, v in cnt.items()))
    st.caption("說明：🔴 實際崩塌範圍只控制已發生崩塌統計；🟠 潛在滑動體範圍只控制殘餘土體；🔵 堆積區控制堆積統計。DoD 本身永遠先計算整個分析區。")
