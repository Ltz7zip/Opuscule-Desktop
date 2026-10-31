# -*- coding: utf-8 -*-
"""tiers — Terrain Pro 的真实度阶梯（Draft / Standard / Ultra / Research）。

一个档位就是一份"预算合同"：DEM 边长、种子包边长、瓦片金字塔层级与瓦片
边长、3D LOD 级别、目标墙钟时间。``produce()`` 只按 ``Tier`` 字段办事，
不在这里做任何判分逻辑。

档位与规格 §12.1 对应：

    draft     2048 以内的快速预览（这里取 1024 保证 1–3 分钟预算）
    standard  2048 + 9 群系 + 河湖 + 2K 3D（可发布，<10 分钟）
    ultra     4096 + 四级 LOD/PBR（<30 分钟，需 ≥32GB）
    research  8192 / 集合 / 敏感性（小时级；本机 18GB 不可行）

本模块只依赖标准库，不导入 terrain.core / torch。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class Tier:
    """一个真实度档位的全部预算参数（不可变，可用 ``dataclasses.replace`` 派生）。"""

    name: str                    # 档位标识（小写）
    size: int                    # DEM 边长（生成分辨率）
    seedpack_size: int           # 种子包 / 基础场的边长
    tile_levels: Tuple[int, ...]  # 2D 瓦片金字塔层级（level n → 2^n×2^n 张）
    tile_size: int               # 单张瓦片边长（像素）
    lod_levels: Tuple[int, ...]   # 3D LOD 级别（同时也是贴图边长上限）
    target_seconds: int          # 目标墙钟时间（秒），仅作验收口径
    description: str             # 一句话定位
    notes: str                   # 硬件 / 注意事项

    def to_dict(self) -> Dict[str, object]:
        """转成 JSON 友好的 dict（写 timing.json / batch_summary.json 用）。"""
        return {
            "name": self.name,
            "size": int(self.size),
            "seedpack_size": int(self.seedpack_size),
            "tile_levels": [int(v) for v in self.tile_levels],
            "tile_size": int(self.tile_size),
            "lod_levels": [int(v) for v in self.lod_levels],
            "target_seconds": int(self.target_seconds),
            "description": self.description,
            "notes": self.notes,
        }

    def with_size(self, size):
        """派生一个只改 DEM 边长的档位（CLI ``--size`` 用）。"""
        return replace(self, size=int(size))


TIERS: Dict[str, Tier] = {
    "draft": Tier(
        name="draft",
        size=1024,
        seedpack_size=512,
        tile_levels=(0, 1),
        tile_size=256,
        lod_levels=(512, 1024),
        target_seconds=180,
        description="快速看大陆/山脉/气候：低分辨率 DEM + 两级 3D LOD",
        notes="1–3 分钟；内存 <4GB；可 --size 2048 加档但要吃满 10 分钟预算",
    ),
    "standard": Tier(
        name="standard",
        size=2048,
        seedpack_size=512,
        tile_levels=(0, 1, 2),
        tile_size=256,
        lod_levels=(512, 1024, 2048),
        target_seconds=600,
        description="可发布的世界地图：2048 DEM + 瓦片金字塔 + 三级 3D LOD/PBR",
        notes="2048 标准 <10 分钟（规格 §12.3）；16GB 机器可跑",
    ),
    "ultra": Tier(
        name="ultra",
        size=4096,
        seedpack_size=1024,
        tile_levels=(0, 1, 2, 3),
        tile_size=256,
        lod_levels=(512, 1024, 2048, 4096),
        target_seconds=1800,
        description="接近研究级/印刷级：4096 DEM + 四级 LOD/PBR + 9 级瓦片",
        notes="4096 <30 分钟；≥32GB 内存 + 12GB 显存（规格 §12.3）",
    ),
    "research": Tier(
        name="research",
        size=8192,
        seedpack_size=2048,
        tile_levels=(0, 1, 2, 3, 4),
        tile_size=256,
        lod_levels=(512, 1024, 2048, 4096, 8192),
        target_seconds=7200,
        description="8192 全球 / 集合 / 敏感性分析，论文与影视级（离线小时级）",
        notes="8192 DEM ≈ 256MB float32，打包峰值 >8GB；本机 18GB 不可行，"
              "需 64GB 工作站（规格 §12.3）",
    ),
}


def resolve_tier(name):
    """档位解析：大小写不敏感；已是 ``Tier`` 时原样返回。

    Raises
    ------
    ValueError
        档位名不存在（消息里列出可选值）。
    NotImplementedError
        ``research``（8192）：本机 18GB 内存无法承载，仅在规格里占位。
    """
    if isinstance(name, Tier):
        return name
    key = str(name).strip().lower()
    if key not in TIERS:
        raise ValueError(
            f"未知档位 {name!r}；可选：{', '.join(sorted(TIERS))}")
    if key == "research":
        raise NotImplementedError(
            "research 档位（8192 DEM + 64GB 级内存）未在本机实现："
            "本机 18GB，8192² float32 DEM 与打包峰值会触发 OOM。"
            "请用 ultra（4096）或到 ≥64GB 工作站上另行开启。")
    return TIERS[key]


__all__ = ["Tier", "TIERS", "resolve_tier"]
