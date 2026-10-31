# -*- coding: utf-8 -*-
"""3D 模型问题定量诊断：海岸边缘山脉 / 内陆水体 / 其他地球对比问题"""
import numpy as np
from scipy import ndimage

d = np.load('/Users/ltz2024/PycharmProjects/Fantasy_World_Simulator/terrain/output/world_dem.npy').astype(np.float64)
h, w = d.shape
land = d > 0
ocean = ~land
print(f"shape={d.shape}  land_frac={land.mean():.4f}  elev range=({d.min():.0f},{d.max():.0f})")

# ============ 1. 海岸边缘山脉：距海距离 vs 高程 ============
d_coast = ndimage.distance_transform_edt(land)
print("\n=== 1. 距海岸距离 → 平均高程 (m) ===")
print("距海(px)  像素数   平均高程    中位高程    0-200m占比")
prev = 0
for b in [0.5, 5, 15, 30, 60, 120, 240, 480, 960]:
    m = (d_coast > prev) & (d_coast <= b) & land
    prev = b
    if m.sum() == 0: continue
    e = d[m]
    print(f"{prev:>6.0f}   {m.sum():>8d}  {e.mean():8.0f}  {np.median(e):8.0f}  {(e<=200).mean():.2f}")

ring = (d_coast <= 25) & land
inner = (d_coast > 25) & land
print(f"\n海岸25px环带: mean={d[ring].mean():.0f}m median={np.median(d[ring]):.0f}m  n={ring.sum()}")
print(f"内陆>25px:   mean={d[inner].mean():.0f}m median={np.median(d[inner]):.0f}m  n={inner.sum()}")

c0 = (d_coast <= 8) & land
c8 = (d_coast <= 40) & land
print(f"海岸0-8px >400m 占比: {(d[c0]>400).mean():.2f}  0-40px>400m: {(d[c8]>400).mean():.2f}")

# ============ 2. 内陆水体 ============
water = d <= 0
lab, n = ndimage.label(water)
border = np.zeros_like(water)
border[0,:]=border[-1,:]=border[:,0]=border[:, -1] = True
borders = np.unique(lab[border & water])
sea = np.isin(lab, borders)
inland_water = water & ~sea
print("\n=== 2. DEM 内陆水体（低于海平面且不连海）===")
print(f"内陆水体像素: {inland_water.sum()} 占全图 {inland_water.mean():.4f}  占陆地 {(inland_water.sum()/land.sum()):.4f}")
liw, nlw = ndimage.label(inland_water)
if nlw > 0:
    sizes = ndimage.sum(np.ones_like(d), liw, range(1, nlw+1))
    for i in np.argsort(sizes)[-5:][::-1]:
        k = i+1
        print(f"  块{k}: {sizes[i]:.0f}px 面积占比图 {sizes[i]/d.size:.4f} 最深处 {d[liw==k].min():.0f}m")

print("\n=== 3. 纬度带陆地/内陆水体分布 ===")
lat_bands = np.arange(0, h, h//12)
print("纬度带  land_frac  inland_water/land  mean_elev")
for i in range(len(lat_bands)-1):
    sl = d[lat_bands[i]:lat_bands[i+1]]
    ld = land[lat_bands[i]:lat_bands[i+1]]
    iw = inland_water[lat_bands[i]:lat_bands[i+1]]
    if ld.sum() == 0:
        print(f"  {i:2d}  band  land=0"); continue
    print(f"  {i:2d}   {ld.mean():.3f}  {iw[ld].sum()/ld.sum():.4f}  {sl[ld].mean():8.0f}")

print("\n=== 4. 海拔结构 ===")
for lo, hi in [(0,200),(200,400),(400,1000),(1000,2000),(2000,3500),(3500,99999)]:
    m = (d>lo)&(d<=hi)
    print(f"  {lo:>5}-{hi if hi<99999 else '∞':>6}m 占全图 {m.mean():.4f}  占陆地 {m[land].sum()/land.sum():.4f}")

print("\n=== 5. 海岸高程梯度 ===")
coast_px = (d_coast <= 1.5) & land
print(f"沿岸1px: mean={d[coast_px].mean():.0f}m, >400m占比 {(d[coast_px]>400).mean():.2f}")
rim = (d > 500) & (d_coast <= 60) & land
print(f"离海<60px且>500m的岸山: {rim.sum()}px  占总陆地 {rim[land].sum()/land.sum():.3f}")