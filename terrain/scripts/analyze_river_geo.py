# -*- coding: utf-8 -*-
# analyze_river_geo.py —— 水系地理指标量化 (内流区/封闭湖/河流弯曲度/入海口)
# 运行: python terrain/scripts/analyze_river_geo.py --seeds 32 33 34 35 36 --tag v34
# 复用 DEM (dem_m 米制高程)。指标:
#   endo_share        : 不流向海洋的陆地像素占比 (地球 ~18%, 但绝大多数在干旱区)
#   n_endo_basins     : 内流流域连通分量数 (>=0.05% 陆地)
#   largest_endo_lake : 最大内流盆地面积 / 陆地面积 -> 巨型内陆海
#   river_sinuosity   : 可见河道蜿蜒度 (前 40 条中心线 弧长/直线长)
#   n_river_mouths    : 入海口的陆地像素数 (acc>=q99.9 且邻海)
#   shelf_share       : 浅海 -200~0m 占比
#   pole_ice_rivers   : 极地冰盖(>65°)河道像素占比
import os
import sys
import json
import argparse
import numpy as np
from numba import njit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DIR = os.path.join(_ROOT, "terrain")

from scipy.ndimage import label as ndi_label, binary_dilation
from terrain.modules.hydro_module import (fill_depressions, d8_flow_accumulation,
                                          extract_river_centerlines)


@njit(cache=True)
def _d8_recv(e, h, w):
    """D8 最陡下坡接收者索引 (n,) int64; -1 = 无。"""
    n = h * w
    recv = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        y = i // w
        x = i - y * w
        best_s = -1e18
        best_r = -1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    j = ny * w + nx
                    d = 1.41421356 if (dy != 0 and dx != 0) else 1.0
                    s = (e[i] - e[j]) / d
                    if s > best_s:
                        best_s = s
                        best_r = j
        recv[i] = best_r
    return recv


@njit(cache=True)
def _reach_sea(recv, sea, h, w):
    """标记每个像素是否最终排入海洋 (DAG 反拓扑传播)。返回 bool (h,w)。"""
    n = h * w
    reach = np.zeros(n, np.int8)
    for i in range(n):
        if sea[i]:
            reach[i] = 1
    # 按高程从低到高的自然处理: 直接链式, 记忆化迭代
    memo = reach.copy()
    order = np.arange(n)
    for i in range(n):
        if memo[i] != 0:
            continue
        path = []
        cur = i
        seen = set()
        while True:
            if memo[cur] != 0:
                m = memo[cur]
                break
            if cur in seen:
                m = -1  # 环 (PnP 填洼后不应出现)
                break
            seen.add(cur)
            path.append(cur)
            r = recv[cur]
            if r < 0 or r == cur:
                m = -1
                break
            cur = r
        for p in path:
            memo[p] = m
    return memo.reshape(h, w) == 1


def analyze(path, step_div=2):
    dem = np.load(path).astype(np.float32)
    Hw, Ww = dem.shape
    step = max(1, Hw // 2048)
    dem = dem[::step, ::step].copy()
    h, w = dem.shape
    land = dem > 0
    sea = ~land
    if land.sum() < 1000:
        return {}

    filled = fill_depressions(dem)
    recv = _d8_recv(filled.ravel().astype(np.float64), h, w)
    rng = np.random.RandomState(7)
    acc = d8_flow_accumulation(filled, rng)

    sea_f = sea.ravel()
    reach = _reach_sea(recv, sea_f, h, w)
    endo = land & ~reach
    endo_share = float(endo.sum()) / max(land.sum(), 1)

    lab, nlab = ndi_label(endo)
    sizes = np.bincount(lab.ravel())
    big = sizes[1:] > (land.sum() * 0.0005)
    n_endo_basins = int(big.sum())
    largest_endo = float(sizes[1:][big].max()) / land.sum() if big.any() else 0.0

    lines = extract_river_centerlines(acc.astype(np.float64), land, q=0.998,
                                      min_len=20, monotonic_repair=False)
    # 蜿蜒度: 只统计长度排名前 40 的主河 (短支流/支流碎片会拉低均值)
    if lines:
        lines_sorted = sorted(lines, key=lambda p: len(p), reverse=True)[:40]
    else:
        lines_sorted = []
    sinu = []
    for pp in lines_sorted:
        arc = float(np.sum(np.hypot(np.diff(pp[:, 0]), np.diff(pp[:, 1]))))
        chord = float(np.hypot(pp[-1, 0] - pp[0, 0], pp[-1, 1] - pp[0, 1]))
        if chord > 5:
            sinu.append(arc / chord)
    river_sinuosity = float(np.mean(sinu)) if sinu else 0.0

    thr_mouth = float(np.quantile(acc[land], 0.999))
    mouth_cand = land & (acc >= thr_mouth)
    near_sea = binary_dilation(sea, iterations=1)
    n_river_mouths = int((mouth_cand & near_sea).sum())

    shelf = (dem <= 0) & (dem > -200)
    shelf_share = float(shelf.sum()) / dem.size

    lats = (90.0 - 90.0 * np.arange(h) / (h - 1))[:, None]
    ice = (np.abs(lats) > 65.0) & land
    river_px = (acc > float(np.quantile(acc[land], 0.985))) & land
    pole_ice_rivers = float((river_px & ice).sum()) / max(ice.sum(), 1)

    return dict(endo_share=endo_share, n_endo_basins=n_endo_basins,
                largest_endo_lake=largest_endo, river_sinuosity=river_sinuosity,
                n_river_mouths=n_river_mouths, shelf_share=shelf_share,
                pole_ice_rivers=pole_ice_rivers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[32, 33, 34, 35, 36])
    ap.add_argument("--tag", type=str, default="v34")
    ap.add_argument("--dir", type=str, default=None)   # 覆盖 DEM 目录 (如 v34_baseline)
    a = ap.parse_args()
    datadir = a.dir if a.dir else os.path.join(_DIR, "output")
    results = {}
    for seed in a.seeds:
        p = os.path.join(datadir, f"seed{seed}_dem.npy")
        if not os.path.exists(p):
            print(f"seed{seed} 缺失, 跳过")
            continue
        results[seed] = analyze(p)
        r = results[seed]
        print(f"seed{seed}: endo={r['endo_share']:.3f} basins={r['n_endo_basins']} "
              f"largest_endo={r['largest_endo_lake']:.3f} sinu={r['river_sinuosity']:.2f} "
              f"mouths={r['n_river_mouths']} shelf={r['shelf_share']:.3f} "
              f"pole_ice_riv={r['pole_ice_rivers']:.3f}")
    out = os.path.join(_DIR, "output", f"rivergeo_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("已保存:", out)


if __name__ == "__main__":
    main()