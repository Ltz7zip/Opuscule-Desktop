# -*- coding: utf-8 -*-
"""python -m terrain.supervisor CLI 入口.

用法:
    # 默认: 审计指定 seed 的世界 + 反思 + 写日志 + 注入 overrides
    python -m terrain.supervisor --seed 42 --size-px 2048

    # 只跑审计, 不写日志
    python -m terrain.supervisor --seed 42 --audit-only

    # 不自动注入 overrides (默认会注入)
    python -m terrain.supervisor --seed 42 --no-apply

    # 查看最近 5 次记录
    python -m terrain.supervisor --history 5

    # 聚合统计
    python -m terrain.supervisor --stats

    # 清空 overrides.json
    python -m terrain.supervisor --reset-overrides

注意:
    - 审计需要 terrain/output/world_dem*.npy 存在 (由 generate_world 生成)。
    - 若 DEM 不存在, 会自动触发 WorldGeoContext.from_seed 生成 (耗时)。
"""
from __future__ import annotations

import argparse
import sys

# 确保 core/ 在 sys.path 中（terrain.core.* 可裸导入）
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_DIR = os.path.dirname(_HERE)
_CORE = os.path.join(_TERRAIN_DIR, "core")
for _p in (_TERRAIN_DIR, _CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from . import (
    audit_world, reflect_and_learn, reset_overrides,
    query_recent, aggregate_stats,
)


def main(argv=None):
    """CLI 主入口.

    Parameters
    ----------
    argv : list, optional
        命令行参数 (默认 sys.argv[1:])。
    """
    ap = argparse.ArgumentParser(
        prog="python -m terrain.supervisor",
        description="地形监督模块: 审计世界地形/气候/群系")
    ap.add_argument("--seed", type=int, default=0,
                    help="世界种子 (默认 0)")
    ap.add_argument("--size-px", type=int, default=2048, choices=[2048, 4096],
                    help="DEM 单边像素数 (默认 2048)")
    ap.add_argument("--audit-only", action="store_true",
                    help="只跑审计, 不写日志")
    ap.add_argument("--no-apply", action="store_true",
                    help="不自动注入 overrides (默认会注入低风险)")
    ap.add_argument("--history", type=int, default=0, metavar="N",
                    help="查看最近 N 次记录")
    ap.add_argument("--stats", action="store_true",
                    help="聚合统计")
    ap.add_argument("--reset-overrides", action="store_true",
                    help="清空 overrides.json")
    args = ap.parse_args(argv)

    if args.reset_overrides:
        reset_overrides()
        print("overrides.json 已清空")
        return 0
    if args.history:
        for r in query_recent(args.history):
            s = r.get("summary", {})
            print(f"[{r.get('timestamp', '?')}] seed={r.get('seed', '?')} "
                  f"| critical={s.get('critical', 0)} warn={s.get('warn', 0)} "
                  f"info={s.get('info', 0)} | land={r.get('metrics', {}).get('land_ratio', 0)*100:.1f}%")
        return 0
    if args.stats:
        import json
        stats = aggregate_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    # 默认: 审计指定 seed (lazy import 避免循环依赖)
    try:
        from terrain.core.geo_context import WorldGeoContext
    except ImportError as e:
        print(f"错误: 无法导入 geo_context ({e})")
        return 1

    WorldGeoContext.reset()
    ctx = WorldGeoContext.from_seed(seed=args.seed, size=args.size_px)
    dem_m = ctx.dem.astype("float32")
    lats = ctx.lat_grid.astype("float32")

    report = audit_world(dem_m, lats, args.seed, args.size_px)
    if args.audit_only:
        print("[audit-only] 不写日志")
        return 0
    reflect_and_learn(report, apply_suggestions=not args.no_apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
