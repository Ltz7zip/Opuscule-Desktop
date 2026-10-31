# -*- coding: utf-8 -*-
# make_montage.py —— 真实地球(同风格渲染) + 10种子生成图 对比拼图
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 允许直接 `python terrain/scripts/make_montage.py` 运行：把项目根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from terrain.scripts.analyze_worlds import load_earth
from terrain.core.terrain_engine import m2v, make_terrain_cmap
from terrain.core.generate_world import render_world

_file_dir = os.path.dirname(os.path.abspath(__file__))
_dir = (os.path.dirname(_file_dir)
        if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)
S = 2048
LATS = (-90.0 + (np.arange(S) + 0.5) * (180.0 / S)).astype(np.float32)


def main() -> int:
    """入口: 渲染真实地球(同风格) + 10 种子生成图拼成对比拼图 worlds_montage.png。"""
    try:
        earth = load_earth()
    except Exception as exc:
        print(f"[make_montage] 加载 ETOPO1 失败：{exc}\n"
              f"请把 ETOPO1_Ice_g_gmt4.grd 放到 {os.path.join(_dir, 'data')}/ 下。",
              file=sys.stderr)
        return 2
    rgb_e = render_world(m2v(earth), LATS, earth > 0, make_terrain_cmap(),
                         np.random.RandomState(0))
    out_earth = os.path.join(_dir, "output", "earth_render.png")
    os.makedirs(os.path.dirname(out_earth), exist_ok=True)
    plt.imsave(out_earth, rgb_e)

    fig, axes = plt.subplots(4, 3, figsize=(21, 22))
    axes[0, 0].imshow(rgb_e)
    axes[0, 0].set_title("真实地球 ETOPO1 (同风格渲染)", fontsize=16)
    axes[0, 0].axis("off")
    cells = [axes[0, 1], axes[0, 2]] + list(axes[1]) + list(axes[2]) + list(axes[3])
    for i, ax in enumerate(cells, start=1):
        p = os.path.join(_dir, "output", f"seed{i}_map.png")
        if not os.path.exists(p):
            ax.set_title(f"(seed{i} 缺失 → 先 python terrain/scripts/batch_gen.py --seeds {i})",
                         fontsize=10)
            ax.axis("off")
            continue
        ax.imshow(plt.imread(p))
        ax.set_title(f"Seed {i}", fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(_dir, "output", "worlds_montage.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"✅ 拼图已保存：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
