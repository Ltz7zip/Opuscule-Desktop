# -*- coding: utf-8 -*-
"""lod — DEM → 多级 glTF-Binary（GLB）LOD，自包含 PBR 材质。

复用 ``terrain.core.export_3d`` 的 ``build_mesh`` / ``write_gltf``（不改动它），
在此之上补三件事：

1. **重采样**：每个 LOD 级别先把 DEM 重采样到 ``level × level``
   （整数倍用块均值 = 面积平均，否则 scipy 双线性）。
2. **面数控制**：``decimate_for(level)`` 给出抽稀步长，使各级网格顶点数
   落在 256–512² 的可控区间（见下表），三角形数写进统计。
3. **GLB 封装 + PBR**：``write_gltf`` 产出的是 ``.gltf``（JSON + base64
   内嵌 buffer）。这里把它重新封装成标准 GLB（magic ``glTF``，JSON chunk +
   BIN chunk），并对 JSON 做后处理：确保 material 有
   ``pbrMetallicRoughness``（baseColorTexture / metallicFactor /
   roughnessFactor），再按需追加一张从 DEM 生成的法线贴图。

级别 → 网格 / 三角形（plane 模式，grid = level // decimate）：

    level   512  → decimate 2 → grid 256 → 131k 三角形
    level  1024  → decimate 2 → grid 512 → 523k 三角形
    level  2048  → decimate 4 → grid 512 → 523k 三角形
    level  4096  → decimate 8 → grid 512 → 523k 三角形

贴图边长上限 ``texture_max``（缺省 1024）用于压住 GLB 体积。

本模块依赖 terrain.core（Pro 侧允许），Lite / Web 不导入。
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
import tempfile

import numpy as np

_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A          # 'JSON'
_CHUNK_BIN = 0x004E4942           # 'BIN\0'

#: LOD 级别 → 抽稀步长（同时决定网格边长 = level / step）
_DECIMATE = {512: 2, 1024: 2, 2048: 4, 4096: 8, 8192: 16}


def decimate_for(level):
    """该 LOD 级别使用的 DEM 抽稀步长（见模块 docstring 的表）。"""
    level = int(level)
    if level in _DECIMATE:
        return _DECIMATE[level]
    # 非标准级别：按 512 网格做上限约束，最小 2，避免小级别被抽成马赛克
    return max(2, min(16, int(round(level / 512.0))))


def _resample_dem(dem, level):
    """DEM → ``level × level``：整数倍块均值（面积平均），否则双线性。"""
    dem = np.asarray(dem, np.float32)
    h, w = dem.shape
    level = int(level)
    if (h, w) == (level, level):
        return dem.astype(np.float32)
    if h % level == 0 and w % level == 0:
        k, m = h // level, w // level
        return dem.reshape(level, k, level, m).mean(axis=(1, 3)).astype(np.float32)
    from scipy.ndimage import zoom
    return zoom(dem, (level / h, level / w), order=1).astype(np.float32)


def _resize_rgb(rgb, side):
    """RGB → ``side × side``（PIL 双线性，失败时回退 LANCZOS）。"""
    from PIL import Image
    arr = np.asarray(rgb, np.uint8)
    if arr.shape[:2] == (side, side):
        return arr
    im = Image.fromarray(np.ascontiguousarray(arr), "RGB")
    return np.asarray(im.resize((side, side), Image.BILINEAR), np.uint8)


def _normal_map_png(dem, side=512):
    """由 DEM 梯度生成切线空间法线贴图 PNG 字节（PBR normalTexture 用）。"""
    from PIL import Image
    d = np.asarray(dem, np.float32)
    gy, gx = np.gradient(d.astype(np.float64))
    scale = float(np.percentile(np.hypot(gx, gy), 95)) or 1.0
    nx = -gx / scale
    ny = gy / scale
    nz = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / ln, ny / ln, nz / ln], axis=-1)
    rgb = np.clip(n * 0.5 + 0.5, 0.0, 1.0) * 255.0
    arr = rgb.astype(np.uint8)
    if arr.shape[0] != side:
        arr = np.asarray(Image.fromarray(arr, "RGB").resize(
            (side, side), Image.BILINEAR), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _ensure_pbr(gltf):
    """补齐 PBR 材质字段（``write_gltf`` 已是 PBR，这里做成幂等保证）。"""
    mats = gltf.setdefault("materials", [])
    if not mats:
        mats.append({"name": "dem"})
    m = mats[0]
    pbr = m.setdefault("pbrMetallicRoughness", {})
    has_tex = bool(gltf.get("textures"))
    if has_tex and "baseColorTexture" not in pbr:
        pbr["baseColorTexture"] = {"index": 0}
    pbr.setdefault("metallicFactor", 0.0)
    pbr.setdefault("roughnessFactor", 0.85)
    pbr.setdefault("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    return m


def _add_normal_texture(gltf, normal_png):
    """把法线贴图追加进 buffer / bufferViews / images / textures 并挂到材质。"""
    buf = gltf["buffers"][0]
    uri = buf.get("uri", "")
    if not uri.startswith("data:"):
        raise ValueError("glTF buffer 不是内嵌 data URI，无法追加法线贴图")
    blob = base64.b64decode(uri.split(",", 1)[1])
    if len(blob) % 4:                       # 对齐到 4 字节
        blob += b"\x00" * (4 - len(blob) % 4)
    offset = len(blob)
    blob += normal_png
    buf["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
        blob).decode("ascii")
    buf["byteLength"] = len(blob)

    views = gltf.setdefault("bufferViews", [])
    views.append({"buffer": 0, "byteOffset": offset,
                  "byteLength": len(normal_png), "target": 34962})
    images = gltf.setdefault("images", [])
    images.append({"bufferView": len(views) - 1, "mimeType": "image/png",
                   "name": "normal"})
    textures = gltf.setdefault("textures", [])
    textures.append({"sampler": 0, "source": len(images) - 1})
    m = _ensure_pbr(gltf)
    m["normalTexture"] = {"index": len(textures) - 1, "scale": 1.0}


def gltf_to_glb(gltf, normal_png=None):
    """glTF dict（内嵌 base64 buffer）→ 标准 GLB 字节（magic ``glTF``）。"""
    if "buffers" not in gltf or not gltf["buffers"]:
        raise ValueError("glTF 缺少 buffers")
    if normal_png:
        _add_normal_texture(gltf, normal_png)
    _ensure_pbr(gltf)

    uri = gltf["buffers"][0].get("uri", "")
    if not uri.startswith("data:"):
        raise ValueError("只支持自包含（data URI）buffer")
    bin_data = base64.b64decode(uri.split(",", 1)[1])
    gltf["buffers"][0].pop("uri", None)
    gltf["buffers"][0]["byteLength"] = len(bin_data)

    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)                 # JSON chunk 用空格补齐
    bin_pad = bin_data + b"\x00" * ((4 - len(bin_data) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_pad)
    out = bytearray()
    out += _GLB_MAGIC + struct.pack("<II", _GLB_VERSION, total)
    out += struct.pack("<II", len(js), _CHUNK_JSON) + js
    out += struct.pack("<II", len(bin_pad), _CHUNK_BIN) + bin_pad
    return bytes(out)


def build_lods(dem_m, rgb=None, levels=(512, 1024, 2048), mode="plane",
               *, normal_map=True, texture_max=1024, stats_out=None,
               verbose=True):
    """DEM → ``{level: glb_bytes}``，可直接写进 ``MapPack`` 的 ``3d/lod*.glb``。

    Parameters
    ----------
    dem_m : (H,W) float32
        米制高程（行 0 ≈ 南极，与导出 UV 对齐）。
    rgb : (H,W,3) uint8 or None
        2D 渲染图；None 时退回 ``export_3d._elev_texture`` 高程着色。
    levels : Iterable[int]
        LOD 级别（同时是 3D 贴图的边长上限）。
    mode : {"plane", "globe"}
        网格形态。
    normal_map : bool
        是否追加 PBR 法线贴图。
    texture_max : int
        贴图边长上限（压住 GLB 体积）。
    stats_out : dict | None
        传入时填充 ``{level: {triangles, vertices, grid, decimate, ...}}``。
    verbose : bool
        是否打印每级三角形数。

    Returns
    -------
    dict[int, bytes]
        GLB 字节，前 4 字节为 ``b"glTF"``。
    """
    from terrain.core.export_3d import (_elev_texture, _resolve_exagg,
                                        _resize_to, build_mesh, write_gltf)

    dem = np.asarray(dem_m, np.float32)
    if dem.ndim != 2:
        raise ValueError(f"dem_m 必须是二维，实际 {dem.shape}")
    levels = sorted({int(lv) for lv in levels})
    stats = {}
    out = {}

    for lv in levels:
        dem_lv = _resample_dem(dem, lv)
        step = decimate_for(lv)
        dem_mesh = dem_lv if step <= 1 else _resample_dem(dem_lv, lv // step)
        exagg = _resolve_exagg("auto", dem_mesh, mode)
        mesh = build_mesh(dem_mesh, mode, exagg)

        tex_side = int(min(lv, texture_max))
        tex = (_elev_texture(_resize_to(dem_lv, tex_side))
               if rgb is None else _resize_rgb(rgb, tex_side))

        with tempfile.TemporaryDirectory(prefix="terrain_lod_") as td:
            p = write_gltf(os.path.join(td, f"lod{lv}.gltf"), mesh, tex,
                           wrap_u=(mode == "globe"))
            with open(p, "r", encoding="utf-8") as fh:
                gltf = json.load(fh)

        nrm = (_normal_map_png(dem_lv, min(tex_side, 512))
               if normal_map else None)
        glb = gltf_to_glb(gltf, normal_png=nrm)
        out[lv] = glb
        stats[lv] = {
            "triangles": int(mesh["faces"].shape[0]),
            "vertices": int(mesh["verts"].shape[0]),
            "grid": int(dem_mesh.shape[0]),
            "decimate": int(step),
            "texture": int(tex_side),
            "normal_map": bool(normal_map),
            "bytes": int(len(glb)),
        }
        if verbose:
            print(f"[pro.lod] level {lv}: grid {dem_mesh.shape[0]}² "
                  f"(decimate {step}) → {stats[lv]['triangles']} 三角形, "
                  f"贴图 {tex_side}², GLB {len(glb)/1048576:.2f} MB", flush=True)

    if stats_out is not None:
        stats_out.update(stats)
    return out


__all__ = ["build_lods", "decimate_for", "gltf_to_glb"]
