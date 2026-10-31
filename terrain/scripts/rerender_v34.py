# -*- coding: utf-8 -*-
# rerender_v34.py —— 从已生成 DEM 快速重渲染 (应用 v34 渲染修复: 蜿蜒/三角洲/融水/群系色)
# 运行: python terrain/scripts/rerender_v34.py --seeds 32 33 34 35 36 50 51 52
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DIR = os.path.join(_PROJECT_ROOT, "terrain")

from terrain.core.terrain_engine import m2v, make_terrain_cmap
from terrain.core.generate_world import render_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[32, 33, 34, 35, 36])
    a = ap.parse_args()
    for s in a.seeds:
        p = os.path.join(_DIR, "output", f"seed{s}_dem.npy")
        out = os.path.join(_DIR, "output", f"seed{s}_map.png")
        if not os.path.exists(p):
            print(f"seed{s} 缺失 DEM, 跳过")
            continue
        d = np.load(p).astype(np.float32)
        S = d.shape[0]
        lats = -90.0 + (np.arange(S) + 0.5) * (180.0 / S)
        img = render_world(m2v(d), lats, d > 0, make_terrain_cmap(),
                           np.random.RandomState(s))
        plt.imsave(out, img)
        print(f"seed{s} 重渲染完成 -> {out}", flush=True)


if __name__ == "__main__":
    main()