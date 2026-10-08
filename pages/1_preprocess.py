"""前處理：裁切與縮小大檔 DEM / DSM / 正射影像（分塊處理，低記憶體）

成果會存起來，回到主頁「分析」時自動當作 T1 / T2 / 正射影像使用，不必再上傳。
"""
import hashlib
import os
import tempfile

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import shape
from streamlit_folium import st_folium

import core

st.set_page_config(page_title="前處理：裁切與縮小", page_icon="✂️", layout="wide")
TMP = os.path.join(tempfile.gettempdir(), "dod_app")
OUT = os.path.join(TMP, "pre")
os.makedirs(OUT, exist_ok=True)

st.title("✂️ 前處理：裁切與縮小")
st.caption("大檔先裁到崩塌範圍、必要時降低解析度，再回主頁分析。分塊讀寫，不會把整個檔案讀進記憶體。")


def save_upload(uf):
    buf = uf.getbuffer()
    h = hashlib.md5(f"{uf.name}{uf.size}".encode() + bytes(buf[:1 << 20]) + bytes(buf[-(1 << 20):])).hexdigest()[:12]
    path = os.path.join(TMP, f"{h}_{os.path.basename(uf.name)}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(buf)
    return path


st.info(
    "**雲端版**：上傳整檔仍受主機記憶體（約 1 GB）限制，建議單檔 < 400 MB。\n\n"
    "**超大檔（如 0.8 GB DSM）**：請在自己電腦執行 `streamlit run app.py`，下面直接填本機路徑，就能不上傳、直接裁切。"
)

# ---- 輸入 ----
cols = st.columns(3)
srcs = {}
for col, (key, label, req) in zip(cols, [("t1", "T1（崩塌前）DEM/DSM", True), ("t2", "T2（崩塌後）DEM/DSM", False),
                                          ("ortho", "正射影像（選用）", False)]):
    with col:
        st.markdown(f"**{label}**")
        lp = st.text_input("本機路徑", key=f"pre_lp_{key}", placeholder=r"D:\UAV\1150706_show_dsm.tif").strip().strip('"')
        uf = st.file_uploader("或上傳", type=["tif", "tiff"], key=f"pre_up_{key}")
        if lp:
            if os.path.isfile(lp):
                srcs[key] = lp
            else:
                st.warning("找不到此路徑（雲端版讀不到你的硬碟）")
        elif uf is not None:
            srcs[key] = save_upload(uf)

NAMES = {"t1": "T1", "t2": "T2", "ortho": "正射"}
lock = st.session_state.get("crop_lock")  # {"bounds": (...), "crs_wkt": str}

if not srcs and not lock:
    st.info("請先提供至少一個檔案（建議先從 T1 開始）。")
    st.stop()
if not srcs and lock:
    st.warning("目前沒有任何輸入檔（上傳欄是空的）。範圍已鎖定，請上傳或填入下一個檔案（例如 T2）。")

ref_key = next((k for k in ("t1", "t2") if k in srcs), None)
bounds, crs = None, None

if lock:
    bounds, crs = tuple(lock["bounds"]), CRS.from_wkt(lock["crs_wkt"])
    st.success(f"🔒 已鎖定裁切範圍：E {bounds[0]:,.1f}–{bounds[2]:,.1f}、N {bounds[1]:,.1f}–{bounds[3]:,.1f}"
               "　（T1、T2、正射都會用同一個範圍）")
    if st.button("解除鎖定，重新選範圍"):
        st.session_state.pop("crop_lock")
        st.rerun()
else:
    if ref_key is None:
        st.info("請至少提供 T1 或 T2 以建立總覽並選範圍。")
        st.stop()
    try:
        ov_key = ("ov", srcs[ref_key])
        if st.session_state.get("_ov_key") != ov_key:
            with st.spinner("建立總覽（第一次可能需要一些時間）…"):
                st.session_state["_ov"] = core.overview(srcs[ref_key])
            st.session_state["_ov_key"] = ov_key
        arr, g = st.session_state["_ov"]
    except Exception as e:  # noqa: BLE001
        st.error(f"讀取 {NAMES[ref_key]} 失敗：{e}")
        st.stop()

    crs = g.crs
    if not crs.is_projected:
        st.warning("影像為地理座標系，建議先轉成投影座標（如 EPSG:3826）；裁切仍可進行，但後續分析需要投影座標。")
    l, r, b, t = g.extent
    st.write(f"{NAMES[ref_key]} 範圍：E {l:,.1f} – {r:,.1f}、N {b:,.1f} – {t:,.1f}　（{crs.to_string()}）")

    mode = st.radio("裁切範圍", ["在地圖上畫矩形", "手動輸入座標", "全範圍（只縮小）"], horizontal=True)
    bounds = (l, b, r, t)
    if mode == "手動輸入座標":
        mc = st.columns(4)
        x0 = mc[0].number_input("E 最小", value=float(l), format="%.2f")
        x1 = mc[1].number_input("E 最大", value=float(r), format="%.2f")
        y0 = mc[2].number_input("N 最小", value=float(b), format="%.2f")
        y1 = mc[3].number_input("N 最大", value=float(t), format="%.2f")
        bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    elif mode == "在地圖上畫矩形":
        ds = core.display_grid(g, 1)
        rgba = core.hillshade_rgba(core.hillshade(arr, g.cell_x))
        w, bnds = core.to_wgs84(rgba, ds.transform, ds.crs)
        m = folium.Map(location=[(bnds[0][0] + bnds[1][0]) / 2, (bnds[0][1] + bnds[1][1]) / 2], zoom_start=14,
                       tiles=None, control_scale=True, max_zoom=22)
        folium.TileLayer("OpenStreetMap", name="OSM", max_zoom=22, max_native_zoom=19).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Esri 衛星", max_zoom=22, max_native_zoom=19).add_to(m)
        folium.raster_layers.ImageOverlay(image=core.png_uri(w), bounds=bnds, name="地形陰影", opacity=0.9).add_to(m)
        sh = {"shapeOptions": {"color": "#e53935"}}
        Draw(export=False, draw_options=dict(polyline=False, polygon=False, circle=False, marker=False,
                                             circlemarker=False, rectangle=sh), edit_options={"edit": True}).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        m.fit_bounds(bnds)
        out = st_folium(m, key="crop_map", height=520, use_container_width=True, returned_objects=["all_drawings"])
        drawn = [f for f in ((out or {}).get("all_drawings") or []) if f["geometry"]["type"] == "Polygon"]
        if drawn:
            geom = transform_geom("EPSG:4326", crs, drawn[-1]["geometry"])
            bounds = shape(geom).bounds
            st.success(f"已選範圍：E {bounds[0]:,.1f}–{bounds[2]:,.1f}、N {bounds[1]:,.1f}–{bounds[3]:,.1f}")
        else:
            st.caption("請用地圖左上角的矩形工具框出崩塌範圍（外加一些緩衝與穩定區）。沒畫則使用全範圍。")
    if st.button("🔒 鎖定此範圍（之後的檔案都用它）", type="primary"):
        st.session_state["crop_lock"] = {"bounds": list(bounds), "crs_wkt": crs.to_wkt()}
        st.rerun()

# ---- 縮小與預估 ----
fc = st.columns(2)
factor = fc[0].number_input("DEM 縮小倍率（1 = 保持原解析度）", 1, 50, 1, 1, key="pre_factor")
ofactor = fc[1].number_input("正射影像縮小倍率", 1, 100, 4, 1, key="pre_ofactor", help="正射只當底圖，可縮小很多。")

for k, p in srcs.items():
    try:
        est = core.crop_estimate(p, bounds, crs, int(ofactor if k == "ortho" else factor))
        st.write(f"預估 {NAMES[k]} 輸出：{est['width']:,}×{est['height']:,} 格（{est['cells'] / 1e6:,.1f} M 格）、"
                 f"格距 {est['cell']:.3g} m、未壓縮約 {est['raw_mb']:,.0f} MB")
        if k != "ortho" and est["cells"] > 30_000_000:
            st.warning("超過 3000 萬格，分析頁會要求再降採樣。建議加大倍率或縮小範圍。")
    except Exception as e:  # noqa: BLE001
        st.error(f"{NAMES[k]} 範圍有問題：{e}")

dest = st.text_input("另存資料夾（選用，本機執行時可填，例如 D:\\UAV\\crop）", "").strip().strip('"')


def do_crop(keys):
    res = {}
    for k in keys:
        p = srcs[k]
        stem = os.path.splitext(os.path.basename(p))[0]
        if p.startswith(TMP) and "_" in stem:
            stem = stem.split("_", 1)[1]
        dst = os.path.join(OUT, f"{k.upper()}_{stem}_crop.tif")
        bar = st.progress(0.0, text=f"處理 {NAMES[k]} …")
        try:
            res[k] = core.crop_raster(p, dst, bounds, crs, factor=int(ofactor if k == "ortho" else factor),
                                      progress=lambda v, bar=bar, k=k: bar.progress(v, text=f"處理 {NAMES[k]} … {v:.0%}"))
        except Exception as e:  # noqa: BLE001
            st.error(f"{NAMES[k]} 處理失敗：{e}")
        bar.empty()
    if not res:
        return
    if dest:
        try:
            import shutil

            os.makedirs(dest, exist_ok=True)
            for v in res.values():
                shutil.copy(v["path"], os.path.join(dest, os.path.basename(v["path"])))
            st.success(f"已另存到 {dest}")
        except Exception as e:  # noqa: BLE001
            st.warning(f"另存失敗：{e}")
    st.session_state.setdefault("pre", {}).update({k: v["path"] for k, v in res.items()})
    st.session_state.setdefault("pre_info", {}).update(res)
    for k, v in res.items():
        st.write(f"✅ {NAMES[k]}：{v['width']:,}×{v['height']:,} 格、格距 {v['cell']:.3g} m、{v['size_mb']:,.1f} MB")


if srcs:
    if not lock:
        st.caption("提示：若要分次處理（雲端記憶體不足時），請先按上方「🔒 鎖定此範圍」。")
    bc = st.columns(len(srcs) + (1 if len(srcs) > 1 else 0))
    for col, k in zip(bc, srcs):
        if col.button(f"✂️ 裁切 {NAMES[k]}", key=f"crop_{k}"):
            do_crop([k])
            st.info("完成後請按該檔上傳欄的 ✕ 移除原檔以釋放記憶體，再上傳下一個檔案。")
    if len(srcs) > 1 and bc[-1].button("✂️ 全部一起裁切", type="primary"):
        do_crop(list(srcs))

done = st.session_state.get("pre") or {}
if done:
    st.divider()
    st.markdown("**已完成的前處理成果（主頁 app 會自動使用）：** " + "、".join(NAMES[k] for k in done))
    if not ({"t1", "t2"} <= set(done)):
        st.caption("還缺 " + "、".join(NAMES[k] for k in ("t1", "t2") if k not in done) + "，請接著上傳並裁切。")
    else:
        st.success("T1、T2 都完成了！回到左側選單的 **app** 即可分析。")

for k, v in (st.session_state.get("pre_info") or {}).items():
    if v["size_mb"] < 300:
        with open(v["path"], "rb") as f:
            st.download_button(f"下載 {os.path.basename(v['path'])}（{v['size_mb']:.1f} MB）", f.read(),
                               file_name=os.path.basename(v["path"]), key=f"dlpre_{k}")
if done and st.button("清除前處理成果與鎖定範圍"):
    for key in ("pre", "pre_info", "crop_lock"):
        st.session_state.pop(key, None)
    st.rerun()
