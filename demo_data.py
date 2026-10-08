"""demo_data.py — 產生合成的兩期 DEM / 正射影像 / 滑動面，用來試用與測試。

地形：300 m × 300 m、1 m 格網、TWD97 TM2 (EPSG:3826)、由北向南的坡面
事件：上坡一個 ~3.5 m 深的崩落區 + 坡腳堆積；崩落區旁尚有未崩落的潛在不穩定區
另外在 T2 加入 +0.15 m 平移與微傾斜（模擬兩期對位誤差）與 3 cm 雜訊
"""
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin


def make_demo(outdir: str, seed: int = 0) -> dict:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n, cell = 300, 1.0
    x0, y_top = 250000.0, 2700300.0
    tr = from_origin(x0, y_top, cell, cell)
    yy, xx = np.mgrid[0:n, 0:n].astype("float64")  # yy 向南增加

    # 平滑地形噪聲
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(rng.normal(size=(n, n)), 12)
    noise = noise / np.abs(noise).max() * 3.0
    z1 = 1500.0 - 0.45 * yy + 0.02 * (xx - 150) + noise

    def blob(cy, cx, ry, rx, rot=0.0):
        c, s = np.cos(rot), np.sin(rot)
        u = (xx - cx) * c + (yy - cy) * s
        v = -(xx - cx) * s + (yy - cy) * c
        return np.exp(-((u / rx) ** 2 + (v / ry) ** 2) ** 1.5)

    scar_depth = 3.5 * blob(90, 150, 40, 28)  # 實際崩落深度
    potential = 3.0 * blob(135, 150, 22, 30)  # 旁邊潛在不穩定區（尚未崩落，只有張裂沉陷 ~0.1 m）
    deposit = 2.4 * blob(215, 152, 28, 34)

    z2 = z1 - scar_depth - 0.08 * (potential / 3.0) + deposit
    z2 += 0.15 + 0.0004 * (xx - 150) - 0.0003 * (yy - 150)  # 對位誤差
    z2 += rng.normal(scale=0.03, size=z2.shape)

    prof = dict(driver="GTiff", height=n, width=n, count=1, dtype="float32",
                crs="EPSG:3826", transform=tr, nodata=-9999.0, compress="deflate")
    p1, p2 = os.path.join(outdir, "demo_T1_before.tif"), os.path.join(outdir, "demo_T2_after.tif")
    for p, z in ((p1, z1), (p2, z2)):
        with rasterio.open(p, "w", **prof) as dst:
            dst.write(z.astype("float32"), 1)

    # 滑動面（僅有部分區域有資料，其餘為 NoData → 示範「缺值處以假設補足」）
    slip = z1 - np.maximum(3.5 * blob(90, 150, 42, 30), 3.0 * blob(135, 150, 24, 32))
    slip = slip.astype("float32")
    slip[:, 240:] = -9999.0
    slip[220:, :] = -9999.0
    ps = os.path.join(outdir, "demo_slip_surface.tif")
    with rasterio.open(ps, "w", **prof) as dst:
        dst.write(slip, 1)

    # 正射影像（簡易著色：地形陰影 + 崩落區紅褐色）
    gy, gx = np.gradient(z2)
    shade = np.clip(0.55 + 0.5 * (-gx * 0.7 + gy * 0.7), 0, 1)
    veg = np.stack([60 + 60 * shade, 110 + 80 * shade, 50 + 40 * shade], axis=-1)
    scar = np.clip((scar_depth - 0.3) / 1.0, 0, 1)[..., None]
    img = veg * (1 - scar) + np.array([170, 130, 95]) * (0.6 + 0.4 * shade[..., None]) * scar
    img = np.clip(img + rng.normal(scale=4, size=img.shape), 0, 255).astype("uint8")
    po = os.path.join(outdir, "demo_ortho.tif")
    oprof = dict(driver="GTiff", height=n, width=n, count=3, dtype="uint8",
                 crs="EPSG:3826", transform=tr, compress="deflate")
    with rasterio.open(po, "w", **oprof) as dst:
        for i in range(3):
            dst.write(img[..., i], i + 1)

    return dict(t1=p1, t2=p2, ortho=po, slip=ps)


if __name__ == "__main__":
    print(make_demo("demo_data_out"))
