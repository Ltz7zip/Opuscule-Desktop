# -*- coding: utf-8 -*-
"""terrain.pack — 共享地图包格式（P0）。

Pro / Lite / Web 三条产品线共用同一份文件格式：
    - ``.seedpack``  种子包：高程 + 气候 + 群系 + 简化河网，供设备端再生成
    - ``.mappack``   成品包：DEM + 2D 渲染 + 可选瓦片 / 3D LOD / 矢量要素

对外 API（下游直接 ``from terrain.pack import ...``）：
    FORMAT_SEEDPACK / FORMAT_MAPPACK / PACK_VERSION
    PackError / HashMismatch
    SeedPack / MapPack
    verify / info

底层容器（供 MapPack 等复用）：``PackWriter`` / ``PackReader``。
从 Pro 产物构建见 ``terrain.pack.build``（该模块会拉 terrain.core，故不在此导入）。
"""
from __future__ import annotations

from .io import (
    MapPack as MapPack,
    PackReader as PackReader,
    PackWriter as PackWriter,
    SeedPack as SeedPack,
    info as info,
    verify as verify,
)
from .schema import (
    FORMAT_MAPPACK as FORMAT_MAPPACK,
    FORMAT_SEEDPACK as FORMAT_SEEDPACK,
    PACK_VERSION as PACK_VERSION,
    HashMismatch as HashMismatch,
    PackError as PackError,
    dem_to_rgb as dem_to_rgb,
    m2v as m2v,
    v2m as v2m,
)

__all__ = [
    "FORMAT_SEEDPACK", "FORMAT_MAPPACK", "PACK_VERSION",
    "PackError", "HashMismatch",
    "SeedPack", "MapPack",
    "PackWriter", "PackReader",
    "verify", "info",
    "m2v", "v2m", "dem_to_rgb",
]
