# -*- coding: utf-8 -*-
"""
rerender.py —— 只重画, 不动数据

=============================================================
用途：按需加载已生成的 DEM 数据并重新渲染 PNG 图像，不触发任何世界生成逻辑。

CLI 用法示例：
  python -m terrain.core.rerender        # 从项目根目录运行
=============================================================
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from .terrain_engine import m2v, make_terrain_cmap
from .generate_world import render_world

# 兼容移动：本模块现位于 terrain/core/，_dir 指向上级 terrain/ 以命中 terrain/output/
_file_dir = os.path.dirname(os.path.abspath(__file__))
_dir = (os.path.dirname(_file_dir)
        if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)


def main():
    """按需加载 DEM + 重绘 PNG；避免 import 时就强制要求数据存在。"""
    dem_path = os.path.join(_dir, "output", "world_dem.npy")
    if not os.path.exists(dem_path):
        print(f"[rerender] 未找到 DEM：{dem_path}\n请先运行 `python main.py generate`。",
              file=sys.stderr)
        return 2
    d = np.load(dem_path); S = d.shape[0]
    lats = -90.0 + (np.arange(S) + 0.5) * (180.0 / S)
    out_path = os.path.join(_dir, "output", "world_map.png")
    plt.imsave(out_path,
               render_world(m2v(d), lats, d > 0,
                            make_terrain_cmap(), np.random.RandomState(7)))
    print(f"已重绘 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
