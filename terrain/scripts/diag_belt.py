# -*- coding: utf-8 -*-
"""v38 诊断: 山带连续性 — plate_sim 的 belt/envS 场 vs 最终 DEM >2000m 山块"""
import os, sys, numpy as np
sys.path.insert(0, "/Users/ltz2024/PycharmProjects/Fantasy_World_Simulator")
from scipy import ndimage
from terrain.core.plate_sim import run_plate_sim

def ar_stats(mask):
    lab, n = ndimage.label(mask)
    ars, areas = [], []
    for i in range(1, n+1):
        ys, xs = np.nonzero(lab == i)
        a = ys.size
        if a < 30: continue
        pts = np.stack([ys-ys.mean(), xs-xs.mean()])
        lam = np.linalg.eigvalsh(np.cov(pts))
        ar = float(np.sqrt(max(lam[1], 1e-9)/max(lam[0], 1e-9)))
        ars.append(ar); areas.append(a)
    belt_a = sum(a for ar, a in zip(ars, areas) if ar > 3.0)
    tot = sum(areas)
    mar = float(np.average(ars, weights=areas)) if ars else 0.0
    return dict(belt_share=belt_a/tot if tot else 0, mean_AR=mar,
                n_ranges=sum(1 for a in areas if a >= 100))

seed = 0
elev, tect, crustm, margin, belt, rift = run_plate_sim(T=512, seed=seed, verbose=False)
print("plate_sim belt 场统计:")
print("  belt>0 占比", (belt>0).mean(), " belt>0.35", (belt>0.35).mean())
print("  belt>0.35 AR 统计:", ar_stats(belt > 0.35))
envS_t = tect[1]
print("  envS(tect[1])>0.2:", (envS_t>0.2).mean(), " AR:", ar_stats(envS_t>0.2))

# 上采样到 2048
from scipy.ndimage import zoom
up = 2048//512
belt_2048 = np.clip(zoom(belt.astype(np.float32), up, order=1), 0, 1)
envS_2048 = np.clip(zoom(envS_t.astype(np.float32), up, order=1), 0, 1)
m = np.maximum(envS_2048, 0.85*belt_2048)
print("2048 合成构造场 m>0.2 AR:", ar_stats(m > 0.2))

# 最终 DEM
d = np.load('terrain/output/world_dem.npy').astype(np.float64)
land = d > 0
mtn = land & (d > 2000)
print("最终 DEM >2000m 山块:", ar_stats(mtn))
print("  mtn 占比", mtn.mean())

# 山块落在构造带上(belt>0.2)的比例
on_belt = mtn & (m > 0.2)
print("  山块∩构造带占比", on_belt.sum()/mtn.sum() if mtn.sum() else 0)
print("  山带中心带(>0.35)内山块占比", (mtn & (m > 0.35)).sum()/mtn.sum() if mtn.sum() else 0)
# 山块尺寸直方
lab, n = ndimage.label(mtn)
sz = np.bincount(lab.ravel()); sz[0]=0
big = np.sort(sz[sz>=100])[::-1]
print("  山块(≥100px) 数量", len(big), " top5:", big[:5])

np.save('/private/tmp/belt_envS_seed0.npy', np.stack([belt_2048, envS_2048, (m>0.2).astype(np.float32)]))
print("saved belt/envS fields for visualization")