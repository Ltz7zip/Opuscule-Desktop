# -*- coding: utf-8 -*-
# make_montage_compare.py —— 真实地球 + 任意种子区间 拼图 (视觉验收)
# 运行: python terrain/scripts/make_montage_compare.py --seeds 11..20 --tag v30
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 21)))
    ap.add_argument("--tag", type=str, default="v30")
    a = ap.parse_args()

    cells = list(a.seeds)
    cols = 3
    rows = (len(cells) + 1) // 2 + 1  # 顶部一行: 真实地球
    fig, axes = plt.subplots(rows + 1, 2, figsize=(14, 7 * (rows + 1)))
    axes = np.atleast_2d(axes)

    # 第一行: 真实地球 ETOPO1 同风格渲染 (复用 analyze_worlds 的渲染口径太慢, 直接存图)
    from terrain.scripts.analyze_worlds import load_earth
    from terrain.core.terrain_engine import m2v, make_terrain_cmap
    from terrain.core.generate_world import render_world
    try:
        earth = load_earth()
        rgb_e = render_world(m2v(earth), None, earth > 0, make_terrain_cmap(),
                             np.random.RandomState(0))
        axes[0, 0].imshow(rgb_e); axes[0, 0].set_title("真实地球 ETOPO1 (同风格渲染)", fontsize=13)
    except Exception as e:
        print("earth render fail:", e)
        axes[0, 0].set_title("(earth 渲染失败)")
    axes[0, 0].axis("off")

    placed = 1  # 已占格数
    for s in cells:
        p = os.path.join(_DIR, "output", f"seed{s}_map.png")
        r = placed // 2
        c = placed % 2
        if os.path.exists(p):
            axes[r, c].imshow(plt.imread(p))
            axes[r, c].set_title(f"Seed {s}", fontsize=12)
        else:
            axes[r, c].set_title(f"(seed{s} 缺失)", fontsize=10)
        axes[r, c].axis("off")
        placed += 1

    # 隐藏多余子图
    total_cells = rows + 1
    for i in range(placed, total_cells * 2):
        axes[i // 2, i % 2].axis("off")

    fig.tight_layout()
    out = os.path.join(_DIR, "output", f"worlds_montage_{a.tag}.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"✅ 拼图已保存: {out}")


if __name__ == "__main__":
    main()