# -*- coding: utf-8 -*-
"""A/B 3D 形貌诊断：给定 DEM，输出（a）海岸环形山脉指标（b）内陆水体指标。

用法: python terrain/scripts/measure_shape.py <dem.npy> [label]
"""
import sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "/Users/ltz2024/PycharmProjects/Fantasy_World_Simulator")
from terrain.core.terrain_engine import m2v, v2m
# fill_depressions 可能是 native 或 numba；避免 torch 加载慢，直接用 terrain 同款
from terrain.core.generate_world import fill_depressions as _fill


def report(path, label):
    d = np.load(path).astype(np.float64)
    land = d > 0
    dcoast = ndimage.distance_transform_edt(land)
    # (a) 海岸环
    ring5 = d[(dcoast > 4) & (dcoast <= 15) & land]
    ring30 = d[(dcoast > 4) & (dcoast <= 30) & land]
    # (b) fill 内陆水体
    filled = _fill(m2v(d))
    fl = v2m(filled)
    depth = fl - np.maximum(d, -1e4)
    lake = land & (depth > 28)
    n_land = land.sum()
    lab, n = ndimage.label(lake)
    sizes = ndimage.sum(np.ones_like(d, np.float64), lab, range(1, n + 1)) if n else np.array([])
    lake_share = lake.sum() / n_land
    largest = (sizes.max() / n_land) if n else 0.0
    iwa, _ = ndimage.label((d <= 0) & land)
    print(f"\n=== {label}: {path.split('/')[-1]} ===")
    print(f"  land_frac={land.mean():.4f}")
    print(f"  海岸0-5px 均高  {d[(dcoast<=4)&land].mean():.0f}m")
    print(f"  海岸5-15px 均高 {ring5.mean():.0f}m (中位 {np.median(ring5):.0f})   ← 边缘山脉环")
    print(f"  海岸5-30px 均高 {ring30.mean():.0f}m (中位 {np.median(ring30):.0f})")
    print(f"  内陆fill水体占陆地 {lake_share:.3f} (地球≈0.33 口径) . 最大单体 {largest:.3f} (地球≈0.05)")
    print(f"  DEM内陆海(≤0且不连海→cell判据): {iwa.sum()}, fill水体块 {n}")
    return dict(label=label, ring5mean=ring5.mean(), ring15mean=ring5.mean(),
                ring30mean=ring30.mean(), lake_share=lake_share, largest_lake=largest)


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "x")