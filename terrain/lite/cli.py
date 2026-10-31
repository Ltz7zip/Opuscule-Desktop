# -*- coding: utf-8 -*-
"""cli — Terrain Lite 命令行。

用法::

    python -m terrain.lite --pack x.seedpack --out out.png --size 512 \
        [--cx 0.5 --cy 0.5 --zoom 1.0 --quality balanced]

末行打印一行 JSON（含耗时与 ``resource.getrusage`` 峰值内存）。
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from .engine import LiteEngine, LiteParams, load_pack


def _peak_mb():
    """峰值常驻内存（MB）：macOS 的 ru_maxrss 单位是字节，Linux 是 KB。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024.0 ** 2) if sys.platform == "darwin"
                 else rss / 1024.0, 1)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m terrain.lite",
                                description="Terrain Lite：SeedPack → 局部地图")
    p.add_argument("--pack", required=True, help="输入 .seedpack")
    p.add_argument("--out", required=True, help="输出 PNG")
    p.add_argument("--size", type=int, default=512, help="输出边长（默认 512）")
    p.add_argument("--cx", type=float, default=0.5, help="中心 x（0..1）")
    p.add_argument("--cy", type=float, default=0.5, help="中心 y（0..1）")
    p.add_argument("--zoom", type=float, default=1.0, help="基场采样倍率")
    p.add_argument("--quality", default="balanced",
                   choices=["fast", "balanced", "high"])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    pack = load_pack(args.pack)
    t_load = (time.perf_counter() - t0) * 1000.0

    engine = LiteEngine(pack)
    params = LiteParams(size=args.size, zoom=args.zoom, cx=args.cx,
                        cy=args.cy, quality=args.quality)
    res = engine.render(params)

    from PIL import Image
    Image.fromarray(res["rgb"], "RGB").save(args.out, "PNG")

    stats = dict(res["stats"])
    stats.update({"pack": args.pack, "out": args.out,
                  "load_ms": round(t_load, 1),
                  "render_ms": stats.pop("ms"),
                  "total_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                  "peak_rss_mb": _peak_mb()})
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
