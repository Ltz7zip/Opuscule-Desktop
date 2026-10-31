# -*- coding: utf-8 -*-
"""cli — 共享地图包命令行。

用法::

    python -m terrain.pack.cli build-seed --dem world_dem.npy --out x.seedpack
    python -m terrain.pack.cli build-map  --dem world_dem.npy --rgb map.png --out x.mappack
    python -m terrain.pack.cli verify x.seedpack
    python -m terrain.pack.cli info   x.seedpack
    python -m terrain.pack.cli unpack x.seedpack --to out_dir

本模块顶层只导入轻量的 schema / io；``build`` 命令在用到时才导入
``terrain.pack.build``（它依赖 terrain.core）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

from .io import info, verify
from .schema import PackError


def _cmd_build_seed(args):
    from .build import build_seedpack_from_world      # 延迟导入（重依赖）
    path = build_seedpack_from_world(
        args.dem, args.out, size=args.size, seed=args.seed, tier=args.tier,
        license=args.license)
    res = verify(path)
    print(f"已写出 {path}（{os.path.getsize(path)} 字节）")
    print(f"verify: ok={res['ok']} entries={res['n_entries']}")
    return 0 if res["ok"] else 1


def _cmd_build_map(args):
    from .build import build_mappack_from_world
    path = build_mappack_from_world(
        args.dem, args.out, rgb_path=args.rgb, size=args.size, seed=args.seed,
        tier=args.tier, license=args.license)
    res = verify(path)
    print(f"已写出 {path}（{os.path.getsize(path)} 字节）")
    print(f"verify: ok={res['ok']} entries={res['n_entries']}")
    return 0 if res["ok"] else 1


def _cmd_verify(args):
    res = verify(args.path)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


def _cmd_info(args):
    print(json.dumps(info(args.path), ensure_ascii=False, indent=2))
    return 0


def _cmd_unpack(args):
    dest = Path(args.to).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(args.path) as zf:
            for name in zf.namelist():
                parts = Path(name).parts
                if name.startswith("/") or ".." in parts:      # 防 zip-slip
                    raise PackError(f"非法条目路径：{name}")
                out = dest / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(zf.read(name))
                print(f"  {name}")
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"解包失败：{exc}", file=sys.stderr)
        return 1
    print(f"已解包到 {dest}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m terrain.pack.cli",
                                description="共享地图包（SeedPack / MapPack）工具")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-seed", help="从 world_dem.npy 构建 .seedpack")
    b.add_argument("--dem", required=True, help="米制 DEM (.npy)")
    b.add_argument("--out", required=True, help="输出 .seedpack 路径")
    b.add_argument("--size", type=int, default=512, help="基场边长（默认 512）")
    b.add_argument("--seed", type=int, default=None, help="种子（默认按 DEM 派生）")
    b.add_argument("--tier", default="standard", help="质量档位")
    b.add_argument("--license", default="CC-BY-4.0", help="授权标识")
    b.set_defaults(func=_cmd_build_seed)

    m = sub.add_parser("build-map", help="从 world_dem.npy 构建 .mappack")
    m.add_argument("--dem", required=True, help="米制 DEM (.npy)")
    m.add_argument("--rgb", default=None, help="2D 渲染图（缺省用 DEM 自动着色）")
    m.add_argument("--out", required=True, help="输出 .mappack 路径")
    m.add_argument("--size", type=int, default=None, help="可选：降采样边长")
    m.add_argument("--seed", type=int, default=7, help="种子")
    m.add_argument("--tier", default="standard", help="质量档位")
    m.add_argument("--license", default="CC-BY-4.0", help="授权标识")
    m.set_defaults(func=_cmd_build_map)

    v = sub.add_parser("verify", help="校验包的完整性")
    v.add_argument("path")
    v.set_defaults(func=_cmd_verify)

    i = sub.add_parser("info", help="打印包的元信息")
    i.add_argument("path")
    i.set_defaults(func=_cmd_info)

    u = sub.add_parser("unpack", help="解包到目录")
    u.add_argument("path")
    u.add_argument("--to", required=True, help="目标目录")
    u.set_defaults(func=_cmd_unpack)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PackError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
