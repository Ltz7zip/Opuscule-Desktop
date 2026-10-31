# -*- coding: utf-8 -*-
# analyze_river_bigmap.py —— 大图河网评估: GAN 生成图 vs 同坐标 ETOPO1 真值
#
# 运行:
#   .venv/bin/python terrain/scripts/analyze_river_bigmap.py --size 1024 --regions 2
#   PYTHONPATH=. .venv/bin/python terrain/scripts/analyze_river_bigmap.py --json out/river_eval.json
#
# 口径说明(重要, 2026-10 修正):
#   管线实际渲染用的是 fill_depressions(v01) 之后的流量累积
#     - terrain/core/generate_world.py:443 fill_depressions -> :512 paint_rivers_rgb
#     - terrain/modules/hydro_module.py HydroModule.compute 同样先填洼
#   因此"河网密度"等指标必须以 filled 口径统计。raw 口径(不填洼)会把大片洼地
#   造成的断流误读成"河网稀疏", 只作诊断列保留。
#
# 指标: 河网密度‰ / 河流像素 / 连通域数 / 最大连通域占比 / acc 分位 / 入海率
#       另附与真值的空间重合(river_metrics), 仅作参考——不同地形的河道位置
#       本就不重合, 不应作为优化目标。
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
from scipy.ndimage import label as ndi_label, binary_dilation

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from terrain.core.terrain_engine import (
    GlobalDEM, make_tect_provider, UNet, device, m2v, generate_large_map)
from terrain.modules.hydro_module import (
    fill_depressions, d8_flow_accumulation)

# 固定阈值: 上游汇水 ≥300 px。1 px ≈ 1.852 km ≈ 3.43 km²  →  ≈1030 km²
RIVER_ACC = 300.0
# 填洼抬升超过此值才算"洼地池塘"(与 HydroModule 的湖判据同量级)
POND_EPS = 0.002


def parse_args():
    p = argparse.ArgumentParser(description="大图河网评估: GAN vs ETOPO1")
    p.add_argument("--ckpt", default=os.path.join(
        _ROOT, "terrain/checkpoints/best_terrain_gan_v5.pth"))
    p.add_argument("--dem", default=os.path.join(
        _ROOT, "terrain/data/ETOPO1_Ice_g_gmt4.grd"))
    p.add_argument("--size", type=int, default=1024, help="评估窗口边长")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--overlap", type=int, default=48)
    p.add_argument("--regions", type=int, default=2, help="采样区域数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thr", type=float, default=RIVER_ACC)
    p.add_argument("--json", default=None, help="把结果写入该 json 路径")
    p.add_argument("--npz", default=None, help="把河网掩膜 dump 到该目录")
    return p.parse_args()


def pick_regions(dem, size, n, seed, frac=(0.25, 0.75)):
    """选 n 个陆地占比落在 frac 区间、且同时含海岸的窗口(固定种子可复现)。"""
    rng = np.random.RandomState(seed)
    h, w = dem.ny - size, dem.nx - size
    picked = []
    for _ in range(200):
        if len(picked) >= n:
            break
        y0, x0 = int(rng.randint(0, h)), int(rng.randint(0, w))
        m, _ = dem.patch(y0, x0, size)
        lf = float((m > 0).mean())
        if frac[0] <= lf <= frac[1]:
            picked.append((y0, x0, lf))
    return picked


def d8_dirs(e):
    """最陡下降方向 (int8 dy/dx), 无下坡则 (0,0)。"""
    h, w = e.shape
    pad = np.pad(e, 1, mode="edge")
    best = np.full((h, w), -np.inf, np.float32)
    dy = np.zeros((h, w), np.int8)
    dx = np.zeros((h, w), np.int8)
    for ddy, ddx, dist in [(0, 1, 1), (1, 1, 1.41421), (1, 0, 1), (1, -1, 1.41421),
                           (0, -1, 1), (-1, -1, 1.41421), (-1, 0, 1), (-1, 1, 1.41421)]:
        nb = pad[1 + ddy:1 + ddy + h, 1 + ddx:1 + ddx + w]
        s = (e - nb) / dist
        m = s > best
        best = np.where(m, s, best)
        dy = np.where(m, ddy, dy).astype(np.int8)
        dx = np.where(m, ddx, dx).astype(np.int8)
    return dy, dx


def to_sea_rate(v01, river, dy, dx, max_steps=4000):
    """河流像素沿 D8 汇流, 统计最终入海(或出图)的比例。"""
    h, w = v01.shape
    ys, xs = np.nonzero(river)
    sea = v01 < 0.5
    hit = 0
    for y, x in zip(ys, xs):
        for _ in range(max_steps):
            d_y, d_x = dy[y, x], dx[y, x]
            if d_y == 0 and d_x == 0:
                break
            y, x = y + d_y, x + d_x
            if y < 0 or x < 0 or y >= h or x >= w:
                hit += 1
                break
            if sea[y, x]:
                hit += 1
                break
        else:
            hit += 1   # 超过步数上限视为已出路
    return hit / max(len(ys), 1)


def analyze(v01, tag, thr=RIVER_ACC):
    """单图河网指标。filled 为管线口径(主), raw 仅诊断。"""
    land = v01 >= 0.5
    rng = np.random.RandomState(7)
    filled = fill_depressions(v01.astype(np.float32))
    pond = float(((filled - v01) > POND_EPS)[land].mean()) if land.any() else 0.0

    out = {"tag": tag, "land": float(land.mean()), "pond": pond}
    for nm, field in (("raw", v01), ("filled", filled)):
        acc = d8_flow_accumulation(field.astype(np.float32), rng).astype(np.float32)
        river = land & (acc >= thr)
        lab, ncomp = ndi_label(river, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
        out[nm] = dict(
            dens=float(river.sum() / max(land.sum(), 1)),
            nriver=int(river.sum()),
            ncomp=int(ncomp),
            biggest=float(sizes.max() / max(river.sum(), 1)),
            acc_p50=float(np.quantile(acc[land], .50)) if land.any() else 0.0,
            acc_p90=float(np.quantile(acc[land], .90)) if land.any() else 0.0,
            acc_p99=float(np.quantile(acc[land], .99)) if land.any() else 0.0,
            river=river,
        )
    dy, dx = d8_dirs(filled)
    out["filled"]["to_sea"] = to_sea_rate(v01, out["filled"]["river"], dy, dx)
    return out


def print_table(g, t):
    """GAN / ETOPO1 对照表(主列 filled, 附 raw 密度作诊断)。"""
    print(f"{'指标':<18}{'GAN':>12}{'ETOPO1':>12}")
    print(f"{'陆地占比%':<18}{g['land']*100:>12.1f}{t['land']*100:>12.1f}")
    print(f"{'洼地占比%':<18}{g['pond']*100:>12.1f}{t['pond']*100:>12.1f}")
    rows = [("河网密度‰(filled)", "dens", 1000.0), ("河网密度‰(raw诊断)", "dens", 1000.0),
            ("河流像素", "nriver", 1.0), ("入海率%", "to_sea", 100.0),
            ("连通域数", "ncomp", 1.0), ("最大连通域占比%", "biggest", 100.0),
            ("acc P50", "acc_p50", 1.0), ("acc P90", "acc_p90", 1.0),
            ("acc P99", "acc_p99", 1.0)]
    for label, key, scale in rows:
        cal = "raw" if "raw" in label else "filled"
        a, b = g[cal][key] * scale, t[cal][key] * scale
        if key in ("nriver", "ncomp"):
            print(f"{label:<18}{int(a):>12d}{int(b):>12d}")
        else:
            print(f"{label:<18}{a:>12.2f}{b:>12.2f}")


def main():
    args = parse_args()
    t_all = time.time()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    blk = int(ck.get("cfg", {}).get("block", 8))
    stats = {"mean": np.array(ck["ch_stats_mean"], np.float32),
             "std": np.array(ck["ch_stats_std"], np.float32)}
    print(f"checkpoint epoch={ck.get('epoch')} block={blk} "
          f"val={ck.get('val_score')}")

    dem = GlobalDEM(args.dem)
    tp = make_tect_provider(dem)
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ck["gen_ema"])
    gen.eval()

    regions = pick_regions(dem, args.size, args.regions, args.seed)
    if not regions:
        raise SystemExit("未找到含海岸的评估窗口, 放宽 --size 或换 --seed")
    print("评估窗口 (y0, x0, 陆地占比):", regions)

    if args.npz:
        os.makedirs(args.npz, exist_ok=True)
    results = []
    for i, (y0, x0, lf) in enumerate(regions):
        t0 = time.time()
        gan = generate_large_map(gen, dem, stats, y0, x0, out_size=args.size,
                                 tile=args.tile, overlap=args.overlap,
                                 block=blk, seed=i, tect_provider=tp)
        m_t, _ = dem.patch(y0, x0, args.size)
        truth = m2v(m_t)
        g = analyze(gan, f"区{i} GAN", args.thr)
        t = analyze(truth, f"区{i} ETOPO1", args.thr)
        print(f"\n[区{i}] ({y0},{x0}) 陆地 {lf*100:.0f}%  "
              f"用时 {time.time()-t0:.0f}s")
        print_table(g, t)

        try:
            from terrain.dl.physics_metrics import river_metrics
            met = river_metrics(g["filled"]["river"].astype(np.float32),
                                t["filled"]["river"].astype(np.float32), 0.5)
            print(f"空间重合(仅参考): IoU {met['iou']:.4f} | "
                  f"connectivity {met['connectivity']:.4f} | "
                  f"topo_acc {met['topo_acc']:.4f}")
            g["overlap"] = {k: met[k] for k in
                            ("iou", "dice", "precision", "recall", "connectivity", "topo_acc")}
        except Exception as e:
            print(f"river_metrics 跳过: {e}")

        if args.npz:
            np.savez_compressed(
                os.path.join(args.npz, f"river_region{i}.npz"),
                gan_v01=gan.astype(np.float32), truth_v01=truth.astype(np.float32),
                gan_river=g["filled"]["river"], truth_river=t["filled"]["river"])
        results.append({"y0": y0, "x0": x0, "land_frac": lf,
                        "gan": g, "truth": t})
        del gan, truth, g, t

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        clean = []
        for r in results:
            item = {"y0": r["y0"], "x0": r["x0"], "land_frac": r["land_frac"]}
            for side in ("gan", "truth"):
                d = {k: v for k, v in r[side].items() if k != "river"}
                d["raw"] = {k: v for k, v in d["raw"].items() if k != "river"}
                d["filled"] = {k: v for k, v in d["filled"].items() if k != "river"}
                item[side] = d
            clean.append(item)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.json}")

    print(f"\n总用时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
