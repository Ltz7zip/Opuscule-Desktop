# -*- coding: utf-8 -*-
# analyze_center_water.py —— 诊断"大陆中心被水填满": 内陆水域分布形态指标
# 指标:
#   inland_share     : 不与海洋连通的陆地内部水域占比 (湖+内流海)
#   largest_lake     : 最大内陆水域面积 / 陆地面积 (地球: 里海≈0.011 陆地)
#   n_big_lakes      : 面积 > 0.5% 陆地的内陆水域数量 (地球: 几乎 0)
#   center_gap_px    : 最大内陆水域质心到最近海岸的距离 (像素, 4096 口径 → 中心性)
#   water_center_k   : 内陆水域像素的 到海距离/平均 加权 —— 越大越"填满中心"
import os
import sys
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.join(_ROOT, "terrain")

from scipy.ndimage import label as ndi_label, distance_transform_edt
from terrain.modules.hydro_module import fill_depressions
from terrain.core.terrain_engine import m2v
from terrain.scripts.analyze_worlds import load_earth


def analyze(dem_m):
    """dem_m: 米制高程 (float32)。返回内陆水域形态指标。"""
    S = dem_m.shape[0]
    land = dem_m > 0
    sea = ~land
    if land.sum() < 1000 or sea.sum() < 1000:
        return {}
    v01 = m2v(dem_m).astype(np.float32)
    filled = fill_depressions(v01)
    lake = land & ((filled - v01) > 0.012)            # 与 render_world 湖泊渲染同口径
    # v35: 巨型 fill 水面(>6% 陆地)渲染为盐沼而非湖 → 指标与视觉一致剔除
    if lake.any():
        lab_l, _nl = ndi_label(lake)
        sz_l = np.bincount(lab_l.ravel())
        sz_l[0] = 0
        cut_l = int(land.sum() * 0.06)
        big_l = np.zeros_like(lake)
        for bi in range(1, _nl + 1):
            if sz_l[bi] > cut_l:
                big_l |= (lab_l == bi)
        lake = lake & ~big_l
    else:
        big_l = np.zeros_like(land)
    # 关键: 渲染把所有 dem<0 都涂成海蓝 → "内陆海" = 不与外海连通的 dem<0 区域
    # (land_fraction_calibration 淹没出的 -20m 洼地 / M5 endo -28m 终端洼地 等)
    from scipy.ndimage import label as ndi_label2
    deep = dem_m < 0
    if deep.any():
        lab_s, _n_s = ndi_label2(deep)
        border = np.zeros_like(deep)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        border_lab = lab_s[border & deep]
        sea_comp = np.unique(border_lab[border_lab > 0])   # 连通到地图边界=外海
        inland_sea = (lab_s > 0) & ~np.isin(lab_s, sea_comp)
    else:
        inland_sea = np.zeros_like(deep)
    total_well = land.sum() + inland_sea.sum()
    inland_sea_frac = float(inland_sea.sum()) / max(total_well, 1)
    # 内陆水域总掩膜 = 填洼湖 ∪ 内陆海 (两者渲染均为水色)
    water = lake | inland_sea
    if not water.any():
        return dict(inland_share=0.0, largest_lake=0.0, n_big_lakes=0,
                    largest_center_gap_px=0.0, water_center_k=0.0,
                    inland_sea_frac=inland_sea_frac)
    lab, _n = ndi_label2(water)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    fracs = sizes[1:] / max(total_well, 1)
    largest = float(fracs.max()) if fracs.size else 0.0
    n_big = int((fracs > 0.005).sum()) if fracs.size else 0
    # 到外海距离 (外海里=0), 衡量水域"中心化"程度
    d_sea = distance_transform_edt(~deep)
    d_sub = d_sea[water] / (S / 4096.0)
    water_center_k = float(d_sub.mean()) if d_sub.size else 0.0
    center_gap = 0.0
    if largest > 0:
        bi = int(np.argmax(sizes))
        yy, xx = np.nonzero(lab == bi)
        cyc = d_sea[yy, xx]
        center_gap = float(cyc.max()) / (S / 4096.0)
    return dict(inland_share=float(water.sum() / max(total_well, 1)),
                largest_lake=largest,
                n_big_lakes=n_big,
                largest_center_gap_px=center_gap,
                water_center_k=water_center_k,
                inland_sea_frac=inland_sea_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[32, 33, 34, 35, 36, 50, 51, 52])
    ap.add_argument("--dir", type=str, default=None)
    ap.add_argument("--earth", action="store_true")
    a = ap.parse_args()
    datadir = a.dir if a.dir else os.path.join(_DIR, "output")
    rows = []
    if a.earth:
        e = load_earth()
        r = analyze(e)
        print(f"earth     : " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
    for s in a.seeds:
        p = os.path.join(datadir, f"seed{s}_dem.npy")
        if not os.path.exists(p):
            print(f"seed{s} 缺失, 跳过")
            continue
        dem = np.load(p).astype(np.float32)
        r = analyze(dem)
        print(f"seed{s:<4} : " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()