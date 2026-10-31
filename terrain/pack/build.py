# -*- coding: utf-8 -*-
"""build — 从 Pro 侧产物（``world_dem.npy`` 等）构建 SeedPack / MapPack。

本模块属于 Pro 工具链，允许依赖 ``terrain.core`` / ``terrain.modules``；
Lite / Web 侧只读包，不调这里。

流程（``build_seedpack_from_world``）：
    world_dem.npy（米制 H×W）
      → 降采样到 size×size
      → 温度 / 降水场（与 generate_world 同一套 lats 约定）
      → Whittaker 群系
      → 填洼 + D8 流量累积 → 抽河网折线 / 湖面多边形
      → SeedPack.build + 原子写出
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from scipy.ndimage import binary_erosion, find_objects, label, zoom

from .io import MapPack, SeedPack
from .schema import dem_to_rgb, m2v

# 与 generate_world 保持一致：行 0 ≈ 南极点
def lat_grid(size):
    """``(-90, +90)`` 等距纬度序列（每行中心），行 0 为南极。"""
    return (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)


# 河网阈值：既是抽河的门限，也写进 lite_hints 供 Lite 对齐
HYDRO_MIN_ACC = 300.0


# ==================== 1. 降采样 ====================
def _resize_dem(dem, size):
    """把任意 (H,W) DEM 降到 size×size：整数倍用块均值，否则双线性。"""
    dem = np.asarray(dem, np.float32)
    h, w = dem.shape
    if (h, w) == (size, size):
        return dem.astype(np.float32)
    if h % size == 0 and w % size == 0:
        k, m = h // size, w // size
        return dem.reshape(size, k, size, m).mean(axis=(1, 3)).astype(np.float32)
    return zoom(dem, (size / h, size / w), order=1).astype(np.float32)


def _derive_seed(dem):
    """未给 seed 时按 DEM 内容派生一个稳定的整数种子。"""
    return int.from_bytes(hashlib.sha256(np.ascontiguousarray(dem).tobytes())
                          .digest()[:4], "big") & 0x7FFFFFFF


def _params_hash(**params):
    """参数指纹（provenance.params_hash）：同参数 → 同指纹，便于误差归因。"""
    text = json.dumps(params, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_rev():
    """当前 git 短哈希（不在仓库 / 无 git 时返回 None，不影响产包）。"""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    rev = out.stdout.strip()
    return rev or None


# ==================== 2. 河网 / 湖泊抽取 ====================
def _extract_rivers(acc, land, min_acc=300.0, min_len=4):
    """按连通域从流量最大点顺流向上追踪，得到若干连通像素折线。

    每条折线都是 8 邻接的 ``[y, x]`` 序列（整数），足够 Lite/Web 画示意河道。
    """
    ch = np.asarray(land, bool) & (np.asarray(acc) >= float(min_acc))
    labels, n = label(ch)
    h, w = ch.shape
    lines = []
    if n == 0:
        return lines
    for comp, sl in enumerate(find_objects(labels), start=1):
        if sl is None or (labels[sl] == comp).sum() < min_len:
            continue
        a = np.where(labels[sl] == comp, acc[sl], -1.0)
        ly, lx = np.unravel_index(int(np.argmax(a)), a.shape)
        y = int(sl[0].start) + int(ly)
        x = int(sl[1].start) + int(lx)
        path = [(y, x)]
        visited = {(y, x)}
        cur = float(acc[y, x])
        while True:
            best, best_a = None, -1.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if labels[ny, nx] != comp or (ny, nx) in visited:
                        continue
                    na = float(acc[ny, nx])
                    if na < cur and na > best_a:      # 只向低流量端（上游）走
                        best, best_a = (ny, nx), na
            if best is None:
                break
            y, x = best
            cur = best_a
            visited.add(best)
            path.append((y, x))
        if len(path) >= min_len:
            lines.append([[int(py), int(px)] for py, px in path])
    return lines


def _lake_polygons(lake_mask, land, max_lakes=16, max_pts=192):
    """湖泊 → 闭合多边形：取湖体边界像素并按最近邻连成环。"""
    mask = np.asarray(lake_mask, bool) & np.asarray(land, bool)
    labels, n = label(mask)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())
    keep = [i for i in np.argsort(sizes[1:])[::-1][:max_lakes] + 1
            if sizes[i] >= 12]
    polys = []
    for comp in keep:
        body = labels == comp
        edge = body & ~binary_erosion(body)
        ys, xs = np.nonzero(edge)
        if len(ys) < 3:
            continue
        if len(ys) > max_pts:                    # 等距抽样，避免 O(k²) 过大
            idx = np.linspace(0, len(ys) - 1, max_pts).astype(np.int32)
            ys, xs = ys[idx], xs[idx]
        pts = np.stack([ys, xs], axis=1).astype(np.int64)
        order = [0]
        used = {0}
        for _ in range(len(pts) - 1):
            d = ((pts - pts[order[-1]]) ** 2).sum(axis=1)
            d[list(used)] = np.iinfo(np.int64).max
            order.append(int(np.argmin(d)))
            used.add(order[-1])
        ring = [[int(pts[i, 0]), int(pts[i, 1])] for i in order]
        ring.append(ring[0])                     # 闭合
        polys.append(ring)
    return polys


# ==================== 3. 从世界产物构建 ====================
def build_seedpack_from_world(dem_path, out_path, *, size=512, seed=None,
                              tier="standard", style=None,
                              license="CC-BY-4.0"):
    """读取 ``world_dem.npy``（米制 H×W），构建种子包并写出，返回绝对路径。"""
    from terrain.modules.climate_module import precipitation_field, whittaker_biome
    from terrain.modules.hydro_module import (d8_flow_accumulation,
                                              fill_depressions)
    from terrain.modules.temperature_module import (continentality_index,
                                                    temperature_field)

    dem_full = np.load(dem_path).astype(np.float32)
    if dem_full.ndim != 2:
        raise ValueError(f"{dem_path} 不是二维 DEM：{dem_full.shape}")
    dem = _resize_dem(dem_full, int(size))
    if seed is None:
        seed = _derive_seed(dem)
    seed = int(seed)

    lats = lat_grid(int(size))
    rng = np.random.RandomState(seed + 88)       # 与 generate_world 的降水随机源一致
    ci = continentality_index(dem, lats)
    T = temperature_field(lats, dem, ci=ci).astype(np.float32)
    P = precipitation_field(lats, dem, rng, ci=ci).astype(np.float32)
    biome = whittaker_biome(T, P, lats=lats, ci=ci).astype(np.uint8)

    v01 = m2v(dem)
    filled = fill_depressions(v01)
    acc = d8_flow_accumulation(filled, np.random.RandomState(seed + 5))
    land = dem > 0.0
    hydro = {
        "rivers": _extract_rivers(acc, land),
        "lakes": _lake_polygons((filled - v01) > 0.012, land),
    }

    provenance = {
        "model_version": "v13",
        "seed": seed,
        "tier": tier,
        "params_hash": _params_hash(size=int(size), rain_seed=seed + 88,
                                    river_threshold=HYDRO_MIN_ACC),
        "git": _git_rev(),
        "source": os.path.basename(str(dem_path)),
        "source_shape": [int(dem_full.shape[0]), int(dem_full.shape[1])],
        "method": "terrain.pack.build_seedpack_from_world",
        "downsample": int(size),
        "seed_derived": bool(_derive_seed(dem) == seed),
        "river_threshold": HYDRO_MIN_ACC,
    }
    sp = SeedPack.build(
        seed=seed, dem_m=dem, v01=v01, temperature=T, precipitation=P,
        biome=biome, hydro=hydro, style=style, provenance=provenance,
        license=license, tier=tier,
        generator={"name": "terrain-pro", "engine": "v13", "tier": tier},
        lite_hints={"coast_eps": 8.0, "slope_ref": 2000.0,
                    "hydro_min_acc": HYDRO_MIN_ACC},
    )
    return sp.write(out_path)


def build_mappack_from_world(dem_path, out_path, *, rgb_path=None, size=None,
                             seed=7, tier="standard", style=None,
                             license="CC-BY-4.0"):
    """从世界产物构建成品包（2D 渲染 + DEM 即可分发）。返回绝对路径。"""
    dem_full = np.load(dem_path).astype(np.float32)
    if dem_full.ndim != 2:
        raise ValueError(f"{dem_path} 不是二维 DEM：{dem_full.shape}")
    dem = dem_full if size is None else _resize_dem(dem_full, int(size))

    if rgb_path:
        from PIL import Image
        with Image.open(rgb_path) as im:
            rgb = np.asarray(im.convert("RGB"), np.uint8)
        if rgb.shape[:2] != dem.shape:
            raise ValueError(f"{rgb_path} 尺寸 {rgb.shape[:2]} 与 DEM {dem.shape} 不一致")
    else:
        rgb = dem_to_rgb(dem)                    # 没给渲染图时用 DEM 自动着色

    mp = MapPack.build(
        seed=int(seed), dem_m=dem, rgb=rgb, style=style,
        provenance={"source": os.path.basename(str(dem_path)),
                    "method": "terrain.pack.build_mappack_from_world"},
        tier=tier, license=license,
        generator={"name": "terrain-pro", "engine": "v13", "tier": tier},
    )
    return mp.write(out_path)
