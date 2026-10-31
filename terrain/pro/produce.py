# -*- coding: utf-8 -*-
"""produce — Pro 侧单种子全链路：生成 → SeedPack → 瓦片 → 3D LOD → MapPack → 校验。

一次调用产出一个可分发单元：

    terrain/output/pro/seed7.seedpack     设备端再生成用的种子包
    terrain/output/pro/seed7.mappack      成品包（DEM + 2D + 瓦片 + 3D LOD/PBR）
    terrain/output/pro/timing.json        各阶段墙钟时间 / 内存 / 体积 / 哈希

复用 / 断点：
    * ``reuse_dem`` 指定既有 DEM 时跳过 ``generate_world``；
    * 两个包都已存在且 ``verify()`` 通过、且未 ``force`` 时直接返回
      ``reused=True``。

内存口径：``resource.getrusage`` 的 ``ru_maxrss`` 在 macOS 是**字节**，
Linux 是 KB，这里按平台换算成 MB，并取 SELF / CHILDREN 的较大值。

本模块依赖 terrain.core（生成器）与 terrain.pack（输出契约）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

from terrain.pack import MapPack, dem_to_rgb, verify
from terrain.pack.build import build_seedpack_from_world

from .lod import build_lods
from .tiers import resolve_tier
from .tiles import build_tiles, count_tiles

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(_ROOT, "terrain", "output", "pro")


# ==================== 小工具 ====================
def _peak_rss_mb():
    """进程（含子进程）峰值常驻内存，MB。macOS 的 ru_maxrss 单位是字节。"""
    import resource
    div = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    mb = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        try:
            mb = max(mb, resource.getrusage(who).ru_maxrss / div)
        except (ValueError, OSError):
            pass
    return round(mb, 1)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    """绝对路径 → 项目内相对路径（对外摘要更短）。"""
    try:
        rel = os.path.relpath(path, _ROOT)
        return path if rel.startswith("..") else rel
    except ValueError:
        return path


def _count_pack_entries(path):
    """从 manifest 里数瓦片与 LOD 条目（复用已有包时用）。"""
    from terrain.pack import info
    meta = info(path)
    tiles, lods = {}, []
    for e in meta["entries"]:
        name = e["path"]
        if name.startswith("map/tiles/z"):
            stem = name[len("map/tiles/z"):-len(".png")]
            lvl = int(stem.split("_")[0])
            tiles[lvl] = tiles.get(lvl, 0) + 1
        elif name.startswith("3d/lod"):
            lods.append(int(name[len("3d/lod"):-len(".glb")]))
    return tiles, sorted(lods)


def _load_rgb(dem_path, dem):
    """找配套 2D 渲染图；找不到就用 ``dem_to_rgb`` 现算。"""
    from PIL import Image
    from terrain.core.export_3d import resolve_map_png
    png = resolve_map_png(dem_path)
    if png:
        with Image.open(png) as im:
            rgb = np.asarray(im.convert("RGB"), np.uint8)
        if rgb.shape[:2] != dem.shape:
            with Image.open(png) as im:
                rgb = np.asarray(im.convert("RGB").resize(
                    (dem.shape[1], dem.shape[0]), Image.BILINEAR), np.uint8)
        return rgb, png
    return dem_to_rgb(dem), None


# ==================== 主流程 ====================
# 阶段 → 累计进度百分比（Studio 任务队列 / --jsonl 事件共用同一口径）
STAGE_PCT = {"start": 1, "dem": 45, "seedpack": 62, "tiles": 76,
             "lods": 92, "mappack": 97, "verify": 100}


def _emit(progress, event, **fields):
    """向调用方推一条进度事件（``None`` 时是空操作）。"""
    if progress is None:
        return
    payload = {"event": event}
    payload.update(fields)
    progress(payload)


def produce(seed, tier="standard", out_dir=DEFAULT_OUT, *, reuse_dem=None,
            force=False, want_3d=True, want_tiles=True, seedpack_size=None,
            verbose=True, progress=None):
    """单种子全链路生产，返回 dict（不返回对象）。

    Parameters
    ----------
    seed : int
        世界种子。
    tier : str | Tier
        档位名（大小写不敏感）或直接给 ``Tier``。
    out_dir : str
        输出目录；相对路径按项目根解析。
    reuse_dem : str or None
        既有 DEM（``.npy`` 米制）。给了就跳过生成。
    force : bool
        为 True 时忽略已存在的包，强制重建。
    want_3d / want_tiles : bool
        是否产出 3D LOD / 2D 瓦片。
    seedpack_size : int or None
        覆盖档位的种子包边长。
    verbose : bool
        打印阶段进度。
    progress : callable or None
        进度回调，收到 ``{"event": "start"|"stage"|"reuse"|"done", ...}``。

    Returns
    -------
    dict
        路径 / 体积 / sha256 / 校验结果 / 阶段计时 / 峰值内存 / 统计。
    """
    t = resolve_tier(tier)
    seed = int(seed)
    out = out_dir if os.path.isabs(str(out_dir)) else os.path.join(
        _ROOT, str(out_dir))
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)

    sp_path = os.path.join(out, f"seed{seed}.seedpack")
    mp_path = os.path.join(out, f"seed{seed}.mappack")
    timing_path = os.path.join(out, "timing.json")

    def _log(msg):
        if verbose:
            print(f"[produce] {msg}", flush=True)

    _emit(progress, "start", seed=seed, tier=t.name, out_dir=out,
          pct=STAGE_PCT["start"])

    # ---- 0. 快速复用 ----
    if not force and os.path.exists(sp_path) and os.path.exists(mp_path):
        vs, vm = verify(sp_path), verify(mp_path)
        if vs["ok"] and vm["ok"]:
            tiles_n, lods = _count_pack_entries(mp_path)
            old = {}
            if os.path.exists(timing_path):
                try:
                    with open(timing_path, "r", encoding="utf-8") as fh:
                        old = json.load(fh)
                except (OSError, ValueError):
                    old = {}
            _log(f"复用已有包 seed={seed}（verify ok）")
            _emit(progress, "reuse", seed=seed, tier=t.name, pct=100)
            _emit(progress, "done", seed=seed, tier=t.name, pct=100,
                  reused=True, seconds=old.get("seconds"),
                  seedpack=sp_path, mappack=mp_path)
            return {
                "seed": seed, "tier": t.name, "out_dir": out,
                "reused": True,
                "dem": old.get("dem"),
                "seedpack": sp_path, "mappack": mp_path,
                "seedpack_bytes": os.path.getsize(sp_path),
                "mappack_bytes": os.path.getsize(mp_path),
                "sha256": {"seedpack": _sha256_file(sp_path),
                           "mappack": _sha256_file(mp_path)},
                "tiles": tiles_n, "tile_total": int(sum(tiles_n.values())),
                "lod_levels": lods, "lod_stats": old.get("lod_stats", {}),
                "verify": {"seedpack": vs["ok"], "mappack": vm["ok"]},
                "stages": old.get("stages", {}),
                "seconds": old.get("seconds"),
                "peak_rss_mb": old.get("peak_rss_mb"),
            }

    t_all = time.time()
    stages = {}

    # ---- 1. DEM：复用或现生成 ----
    t0 = time.time()
    if reuse_dem:
        dem_path = os.path.abspath(str(reuse_dem))
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"reuse_dem 不存在：{dem_path}")
        _log(f"复用 DEM {_rel(dem_path)}，跳过 generate_world")
        generated = False
    else:
        dem_path = os.path.join(out, f"seed{seed}_dem.npy")
        generated = not os.path.exists(dem_path) or force
        if generated:
            from terrain.core.generate_world import generate_world
            _log(f"生成世界 seed={seed} size={t.size}（generate_world）…")
            generate_world(seed=seed, size_px=int(t.size),
                           out_npy=dem_path,
                           out_png=os.path.join(out, f"seed{seed}_map.png"),
                           enable_supervisor=False)
        else:
            _log(f"复用已生成的 DEM {_rel(dem_path)}")
    dem = np.load(dem_path).astype(np.float32)
    if dem.ndim != 2 or dem.shape[0] != dem.shape[1]:
        raise ValueError(f"DEM 不是二维方阵：{dem_path} shape={dem.shape}")
    stages["dem"] = round(time.time() - t0, 3)
    _log(f"DEM {dem.shape[0]}² 就绪（{stages['dem']:.1f}s）")
    _emit(progress, "stage", stage="dem", pct=STAGE_PCT["dem"],
          seconds=stages["dem"], peak_rss_mb=_peak_rss_mb())

    # ---- 2. SeedPack ----
    t0 = time.time()
    sp_size = int(seedpack_size or t.seedpack_size)
    build_seedpack_from_world(dem_path, sp_path, size=sp_size, seed=seed,
                              tier=t.name)
    stages["seedpack"] = round(time.time() - t0, 3)
    _log(f"SeedPack {sp_size}² → {os.path.getsize(sp_path)/1048576:.2f} MB"
         f"（{stages['seedpack']:.1f}s）")
    _emit(progress, "stage", stage="seedpack", pct=STAGE_PCT["seedpack"],
          seconds=stages["seedpack"], peak_rss_mb=_peak_rss_mb())

    # ---- 3. 2D 渲染图 / 瓦片 ----
    t0 = time.time()
    rgb, rgb_src = _load_rgb(dem_path, dem)
    tiles = {}
    if want_tiles:
        tiles = build_tiles(rgb, levels=t.tile_levels, tile_size=t.tile_size)
        _log(f"瓦片 {count_tiles(tiles)} 张 "
             f"(levels={list(tiles)}, {t.tile_size}px)")
    stages["tiles"] = round(time.time() - t0, 3)
    _emit(progress, "stage", stage="tiles", pct=STAGE_PCT["tiles"],
          seconds=stages["tiles"], peak_rss_mb=_peak_rss_mb())

    # ---- 4. 3D LOD（PBR / GLB） ----
    t0 = time.time()
    lods, lod_stats = {}, {}
    if want_3d:
        lods = build_lods(dem, rgb, levels=t.lod_levels, mode="plane",
                          stats_out=lod_stats, verbose=verbose)
    stages["lods"] = round(time.time() - t0, 3)
    _emit(progress, "stage", stage="lods", pct=STAGE_PCT["lods"],
          seconds=stages["lods"], peak_rss_mb=_peak_rss_mb())

    # ---- 5. 打包 ----
    t0 = time.time()
    mp = MapPack.build(
        seed=seed, dem_m=dem, rgb=rgb, tiles=tiles, lods=lods, tier=t.name,
        provenance={
            "source": os.path.basename(dem_path),
            "dem_size": int(dem.shape[0]),
            "rgb_source": (os.path.basename(rgb_src) if rgb_src else "dem_to_rgb"),
            "tile_levels": list(t.tile_levels),
            "tile_size": int(t.tile_size),
            "lod_levels": list(lods),
            "pipeline": "terrain.pro.produce",
        },
        generator={"name": "terrain-pro", "engine": "v13", "tier": t.name},
    )
    mp.write(mp_path)
    stages["mappack"] = round(time.time() - t0, 3)
    _log(f"MapPack → {os.path.getsize(mp_path)/1048576:.2f} MB"
         f"（{stages['mappack']:.1f}s）")
    _emit(progress, "stage", stage="mappack", pct=STAGE_PCT["mappack"],
          seconds=stages["mappack"], peak_rss_mb=_peak_rss_mb())

    # ---- 6. 校验 ----
    t0 = time.time()
    vs, vm = verify(sp_path), verify(mp_path)
    stages["verify"] = round(time.time() - t0, 3)
    if not (vs["ok"] and vm["ok"]):
        _log(f"校验失败：seedpack={vs['errors']} mappack={vm['errors']}")
    _emit(progress, "stage", stage="verify", pct=STAGE_PCT["verify"],
          seconds=stages["verify"], ok=bool(vs["ok"] and vm["ok"]))

    total = round(time.time() - t_all, 3)
    result = {
        "seed": seed,
        "tier": t.name,
        "tier_spec": t.to_dict(),
        "out_dir": out,
        "dem": dem_path,
        "dem_generated": bool(generated),
        "seedpack": sp_path,
        "mappack": mp_path,
        "seedpack_bytes": os.path.getsize(sp_path),
        "mappack_bytes": os.path.getsize(mp_path),
        "sha256": {"seedpack": _sha256_file(sp_path),
                   "mappack": _sha256_file(mp_path)},
        "tiles": {int(k): len(v) for k, v in sorted(tiles.items())},
        "tile_total": count_tiles(tiles),
        "lod_levels": sorted(int(k) for k in lods),
        "lod_stats": lod_stats,
        "verify": {"seedpack": bool(vs["ok"]), "mappack": bool(vm["ok"]),
                   "errors": list(vs["errors"]) + list(vm["errors"])},
        "stages": stages,
        "seconds": total,
        "peak_rss_mb": _peak_rss_mb(),
        "reused": False,
    }

    record = {
        "seed": seed,
        "tier": t.name,
        "tier_spec": t.to_dict(),
        "dem": _rel(dem_path),
        "dem_size": int(dem.shape[0]),
        "generated": bool(generated),
        "seedpack": _rel(sp_path),
        "mappack": _rel(mp_path),
        "seedpack_bytes": result["seedpack_bytes"],
        "mappack_bytes": result["mappack_bytes"],
        "tiles": result["tiles"],
        "tile_total": result["tile_total"],
        "lod_levels": result["lod_levels"],
        "lod_stats": lod_stats,
        "stages": stages,
        "seconds": total,
        "peak_rss_mb": result["peak_rss_mb"],
        "verify": result["verify"],
        "sha256": result["sha256"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(timing_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2, sort_keys=True)

    _log(f"完成 seed={seed} {total:.1f}s，峰值 {result['peak_rss_mb']:.0f} MB，"
         f"verify={'ok' if vm['ok'] and vs['ok'] else 'FAIL'}")
    _emit(progress, "done", seed=seed, tier=t.name, pct=100, reused=False,
          seconds=total, ok=bool(vs["ok"] and vm["ok"]),
          seedpack=sp_path, mappack=mp_path)
    return result


# ==================== CLI ====================
def main(argv=None):
    """命令行入口：``python -m terrain.pro.produce --seed 7 --tier standard``。

    ``--jsonl`` 时逐行向 stdout 打紧凑 JSON（``event``/``pct``/``stage``…），
    供 Studio 用 QProcess 读取进度；此时普通日志改走 stderr。
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="terrain.pro.produce",
        description="Pro 单种子全链路：生成 → SeedPack → 瓦片 → 3D LOD → MapPack")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tier", default="standard",
                    help="draft / standard / ultra（大小写不敏感）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录")
    ap.add_argument("--dem", default=None, help="复用既有 DEM（.npy 米制）")
    ap.add_argument("--size", type=int, default=None, help="覆盖种子包边长")
    ap.add_argument("--seedpack-size", dest="size", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true", help="忽略已有产物强制重建")
    ap.add_argument("--no-3d", dest="want_3d", action="store_false",
                    help="不产出 3D LOD")
    ap.add_argument("--no-tiles", dest="want_tiles", action="store_false",
                    help="不产出 2D 瓦片")
    ap.add_argument("--jsonl", action="store_true", help="逐行 JSON 进度事件")
    ap.add_argument("--quiet", action="store_true", help="静默普通日志")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    jsonl = bool(args.jsonl)
    progress = None
    if jsonl:
        def progress(payload):                # noqa: ANN001
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                  flush=True)

    try:
        result = produce(
            args.seed, tier=args.tier, out_dir=args.out, reuse_dem=args.dem,
            force=args.force, want_3d=args.want_3d, want_tiles=args.want_tiles,
            seedpack_size=args.size, verbose=not (args.quiet or jsonl),
            progress=progress)
    except Exception as exc:                  # noqa: BLE001
        if jsonl:
            print(json.dumps({"event": "error", "message": f"{type(exc).__name__}: {exc}"},
                             ensure_ascii=False, separators=(",", ":")),
                  flush=True)
        else:
            print(f"[produce] 失败：{exc}", file=sys.stderr, flush=True)
        return 1
    if not jsonl and not args.quiet:
        print(json.dumps({k: result[k] for k in
                          ("seed", "tier", "seconds", "seedpack", "mappack")},
                         ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["produce", "main", "DEFAULT_OUT"]
