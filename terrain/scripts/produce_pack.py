# -*- coding: utf-8 -*-
# produce_pack.py —— Terrain Pro 单种子打包 CLI（P1）
# 运行: PYTHONPATH=项目根 .venv/bin/python terrain/scripts/produce_pack.py \
#         --seed 1 --tier draft --reuse-dem terrain/output/world_dem.npy
import argparse
import json
import os
import sys

# 允许直接 `python terrain/scripts/produce_pack.py` 运行：把项目根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from terrain.pro import produce, resolve_tier  # noqa: E402


def _mb(n):
    return f"{(n or 0) / 1048576:.2f} MB"


def main():
    ap = argparse.ArgumentParser(
        description="Terrain Pro：生成 / 打包 SeedPack + MapPack（瓦片 + 3D LOD）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tier", default="standard",
                    help="draft / standard / ultra（research 未实现）")
    ap.add_argument("--size", type=int, default=None,
                    help="覆盖档位的 DEM 边长（如 1024 / 2048）")
    ap.add_argument("--out-dir", default=os.path.join(
        "terrain", "output", "pro"))
    ap.add_argument("--reuse-dem", default=None,
                    help="复用既有 DEM（.npy），跳过 generate_world")
    ap.add_argument("--no-3d", action="store_true", help="不产出 3D LOD")
    ap.add_argument("--no-tiles", action="store_true", help="不产出 2D 瓦片")
    ap.add_argument("--force", action="store_true", help="忽略已有包强制重建")
    a = ap.parse_args()

    tier = resolve_tier(a.tier)
    if a.size:
        tier = tier.with_size(a.size)

    r = produce(a.seed, tier=tier, out_dir=a.out_dir,
                reuse_dem=a.reuse_dem, force=a.force,
                want_3d=not a.no_3d, want_tiles=not a.no_tiles)

    print("=" * 62)
    print(f"Terrain Pro · tier={r['tier']} · seed={r['seed']} · "
          f"dem={os.path.basename(str(r['dem']))}")
    print(f"  SeedPack  {r['seedpack_bytes']:>12,d} B  ({_mb(r['seedpack_bytes'])})")
    print(f"  MapPack   {r['mappack_bytes']:>12,d} B  ({_mb(r['mappack_bytes'])})")
    print(f"  瓦片      {r['tile_total']} 张  {r['tiles']}")
    for lv, st in sorted(r.get("lod_stats", {}).items(), key=lambda kv: int(kv[0])):
        print(f"  LOD {int(lv):>4}  {st['triangles']:>9,d} 三角形  "
              f"grid {st['grid']}²  GLB {_mb(st['bytes'])}")
    if not r.get("lod_stats"):
        print(f"  LOD       已禁用（levels={r.get('lod_levels')}）")
    print(f"  verify    seedpack={r['verify']['seedpack']} "
          f"mappack={r['verify']['mappack']}")
    print("  阶段耗时  " + "  ".join(f"{k}={v:.2f}s"
                                     for k, v in r["stages"].items()))
    print(f"  总计      {r['seconds']:.1f}s · 峰值 {r['peak_rss_mb']:.0f} MB · "
          f"reused={r['reused']}")
    print("=" * 62)

    print(json.dumps({
        "seed": r["seed"],
        "tier": r["tier"],
        "reused": r["reused"],
        "seedpack": r["seedpack"],
        "mappack": r["mappack"],
        "seedpack_bytes": r["seedpack_bytes"],
        "mappack_bytes": r["mappack_bytes"],
        "sha256": r["sha256"],
        "tiles": {str(k): v for k, v in r["tiles"].items()},
        "tile_total": r["tile_total"],
        "lod_levels": r["lod_levels"],
        "lod_triangles": {str(k): v["triangles"]
                          for k, v in r.get("lod_stats", {}).items()},
        "verify_ok": bool(r["verify"]["seedpack"] and r["verify"]["mappack"]),
        "seconds": r["seconds"],
        "peak_rss_mb": r["peak_rss_mb"],
        "stages": r["stages"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
