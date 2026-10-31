# -*- coding: utf-8 -*-
# make_montage_v33.py —— v33 通用修改: 旧版 vs 新版 + 新种子 + 真实地球 视觉对比拼图
# 运行: python terrain/scripts/make_montage_v33.py --before-dir v33_before --seeds 32 33 34 35 36 --new-seeds 50 51 52 53 54
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


def render_earth():
    from terrain.scripts.analyze_worlds import load_earth
    from terrain.core.terrain_engine import m2v, make_terrain_cmap
    from terrain.core.generate_world import render_world
    e = load_earth()
    S = e.shape[0]
    lats = -90.0 + (np.arange(S) + 0.5) * (180.0 / S)
    return render_world(m2v(e), lats, e > 0, make_terrain_cmap(),
                        np.random.RandomState(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before-dir", default="v33_before")
    ap.add_argument("--seeds", type=int, nargs="+", default=[32, 33, 34, 35, 36])
    ap.add_argument("--new-seeds", type=int, nargs="+", default=[50, 51, 52, 53, 54])
    ap.add_argument("--tag", default="v33c")
    a = ap.parse_args()

    seeds_new = [s for s in a.new_seeds if s not in a.seeds and s != 0]  # 0=哨兵空
    n_cols, n_rows = 3, 1 + len(a.seeds) + len(seeds_new)
    if n_rows < 1:
        n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * 5.2, n_rows * 4.4))
    axes = np.atleast_2d(axes)

    try:
        earth = render_earth()
        axes[0, 0].imshow(earth)
        axes[0, 0].set_title("真实地球 ETOPO1\n(同渲染管线)", fontsize=11)
    except Exception as ex:
        axes[0, 0].set_title(f"earth fail: {ex}", fontsize=10)
    axes[0, 0].axis("off")
    axes[0, 1].axis("off"); axes[0, 2].axis("off")
    axes[0, 1].text(0.5, 0.5, "真实地球参考\n\n赤道雨林链 / 副热带荒漠带\n海岸平缓·大陆连片",
                    ha="center", va="center", fontsize=10, color="#444")

    r = 1
    for s in a.seeds:
        b = os.path.join(_DIR, "output", a.before_dir, f"seed{s}_map.png")
        new_p = os.path.join(_DIR, "output", f"seed{s}_map.png")
        axes[r, 0].imshow(plt.imread(b)); axes[r, 0].set_title(f"seed{s} 旧版", fontsize=10)
        axes[r, 1].imshow(plt.imread(new_p)); axes[r, 1].set_title(f"seed{s} 新版", fontsize=10)
        axes[r, 2].axis("off")
        for c in range(2):
            axes[r, c].axis("off")
        r += 1
    for s in seeds_new:
        new_p = os.path.join(_DIR, "output", f"seed{s}_map.png")
        axes[r, 0].imshow(plt.imread(new_p)); axes[r, 0].set_title(f"seed{s} 新版(新种子)", fontsize=10)
        axes[r, 1].axis("off"); axes[r, 2].axis("off")
        axes[r, 0].axis("off")
        r += 1

    plt.tight_layout()
    out_png = os.path.join(_DIR, "output", f"{a.tag}_before_after.png")
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    print("saved:", out_png)


if __name__ == "__main__":
    main()