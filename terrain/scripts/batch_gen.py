# -*- coding: utf-8 -*-
# batch_gen.py —— 批量生成多种子世界地图 (检验用)
# 运行: python batch_gen.py --seeds 1 2 3 4 5 6 7 8 9 10 --size 2048
import os
import sys
import argparse

# 允许直接 `python terrain/scripts/batch_gen.py` 运行：把项目根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from terrain.core.generate_world import generate_world

_file_dir = os.path.dirname(os.path.abspath(__file__))
_dir = (os.path.dirname(_file_dir)
        if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)

def main():
    """``batch_gen.py`` 命令行入口 —— 批量生成多种子世界地图。

    解析 ``--seeds / --size``，对每个 seed 调用
    :func:`generate_world.generate_world` 分别输出 ``seed{id}_map.png``
    与 ``seed{id}_dem.npy`` 到 ``terrain/output/``。已存在的 DEM 会
    被自动跳过以支持断点续跑。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--size", type=int, default=2048)
    a = ap.parse_args()
    for s in a.seeds:
        png = os.path.join(_dir, "output", f"seed{s}_map.png")
        npy = os.path.join(_dir, "output", f"seed{s}_dem.npy")
        if os.path.exists(npy):
            print(f"[skip] seed{s} 已存在")
            continue
        print(f"\n===== seed {s} =====")
        generate_world(seed=s, size_px=a.size, out_png=png, out_npy=npy)
    print("\n全部完成")


if __name__ == "__main__":
    main()
