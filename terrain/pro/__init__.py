# -*- coding: utf-8 -*-
"""terrain.pro — Terrain Pro（P1）：真实度上限的生产线。

定位：Pro 不是"更快的 Lite"，而是**尽可能真实、尽可能强大**——完整 450 Myr
板块演化、2048/4096 全量 GAN、3D LOD/PBR、瓦片金字塔、批量 seed、本地
Studio。单张 2048 世界地图的预算是"几分钟"（standard <10 分钟），4096 是
"半小时级"（ultra <30 分钟）。

对外 API：

    Tier / TIERS / resolve_tier      真实度阶梯（draft / standard / ultra / research）
    produce                         单种子全链路（生成 → SeedPack → 瓦片 → LOD → MapPack）
    batch_produce                   多种子批量（串行 / 进程池 + ledger 续跑）
    build_tiles                     2D 四叉树瓦片金字塔（PIL）
    build_lods                      DEM → 多级 GLB（复用 export_3d，PBR 材质）

依赖边界：
    * Pro **可以**依赖 ``terrain.core``（生成器 / export_3d）与 ``terrain.pack``；
    * Lite / Web 只读 ``terrain.pack``，不导入本子包；
    * ``studio``（PySide6）不在本 ``__init__`` 里导入，Qt 只在 ``studio.main()``
      内部引入，保证无显示环境也能安全 ``import terrain.pro``。

产物与 P0 契约一致：``seed{N}.seedpack`` / ``seed{N}.mappack`` 可直接用
``terrain.pack.verify()`` 校验、``MapPack.open()`` 读回。
"""
from __future__ import annotations

from .batch import batch_produce as batch_produce
from .lod import build_lods as build_lods
from .produce import DEFAULT_OUT as DEFAULT_OUT, produce as produce
from .tiers import TIERS as TIERS, Tier as Tier, resolve_tier as resolve_tier
from .tiles import build_tiles as build_tiles, count_tiles as count_tiles

__all__ = [
    "Tier", "TIERS", "resolve_tier",
    "produce", "DEFAULT_OUT", "batch_produce",
    "build_tiles", "count_tiles", "build_lods",
]
