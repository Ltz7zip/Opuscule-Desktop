# -*- coding: utf-8 -*-
"""terrain_module — 地形模块（高程 / 坡度 / 地貌分区 / 世界地形图）。

依赖：无（直接消费 ``ctx.dem_m``）。

输出键（写入 ``ctx.data``）：
    - ``terrain.v01``        归一化高程 [0, 1]（供下游模块复用）
    - ``terrain.slope``      坡度百分比场（rise/run × 100）
    - ``terrain.landforms``  地貌分类整数标签（位打包，见 classify_landforms）
    - ``terrain.rgb``        世界地形图 RGB（带山体阴影的地形色卡）
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.terrain_engine import m2v, make_terrain_cmap, classify_landforms
from .base import BaseModule, WorldContext


class TerrainModule(BaseModule):
    """地形模块：产出归一化高程、坡度场与地貌分类（无依赖，直接消费 DEM）。

    输出键（写入 ``ctx.data``）：
        - ``terrain.v01``        归一化高程 [0, 1]
        - ``terrain.slope``      坡度百分比场
        - ``terrain.landforms``  地貌分类整数标签
    """
    name = "terrain"
    dependencies: list = []

    def compute(self, ctx: WorldContext) -> None:
        """计算归一化高程 / 坡度场 / 地貌分类并写入 ``ctx.data``。"""
        dem_m = ctx.dem_m
        # 归一化高程（v=0.5 ↔ 0m 海平面），供水系/气温/气候模块复用
        ctx.set("terrain.v01", m2v(dem_m).astype(np.float32))
        # 坡度场（与 geo_context._compute_slope_field 同口径）
        ctx.set("terrain.slope", _slope_field(dem_m, ctx.lats).astype(np.float32))
        # 地貌分类（需要构造场时退化到纯高程分类，tect_ctx 由调用方注入）
        tect_ctx = ctx.get("terrain.tect_ctx") or {}
        lf = classify_landforms(dem_m, ctx.lats, land=ctx.land_mask,
                                envS=tect_ctx.get("envS"),
                                beltS=tect_ctx.get("beltS"),
                                cf=tect_ctx.get("cf"),
                                tect_T=tect_ctx.get("tect_T"),
                                margin_up=tect_ctx.get("margin_up"),
                                riftS=tect_ctx.get("riftS"))
        ctx.set("terrain.landforms", lf)

    def render(self, ctx: WorldContext) -> np.ndarray:
        """渲染世界地形图：地形色卡 + NW 方向山体阴影（缺 v01 时即时回算）。"""
        v01 = ctx.get("terrain.v01")
        if v01 is None:
            v01 = m2v(ctx.dem_m).astype(np.float32)
        cmap = make_terrain_cmap()
        rgb = cmap(np.clip(v01, 0, 1))[..., :3].astype(np.float32)
        # NW 方向山体阴影，增强地形层次感（与 render_world 同口径）
        g = gaussian_filter(v01.astype(np.float32), 1.5)
        gy, gx = np.gradient(g)
        hs = np.clip(0.88 - 22.0 * (gx + gy), 0.60, 1.15)
        shade = (0.78 + 0.55 * np.clip((v01 - 0.5) / 0.5, 0, 1)) * (0.72 + 0.28 * hs)
        land = ctx.land_mask
        rgb = np.where(land[..., None], rgb * shade[..., None], rgb).astype(np.float32)
        return np.clip(rgb, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_METERS_PER_DEG_LAT = 111320.0
_METERS_PER_DEG_LON_EQ = 111320.0


def _slope_field(dem_m: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """基于米制 DEM 计算逐像素坡度百分比（与 geo_context 同口径）。"""
    S = dem_m.shape[0]
    dlat = 180.0 / S
    dlon = 360.0 / S
    dy = _METERS_PER_DEG_LAT * dlat
    g_row, g_col = np.gradient(dem_m.astype(np.float64))
    slope_row = g_row / dy
    cos_lat = np.cos(np.deg2rad(lats)).astype(np.float64)
    dx_per_row = (_METERS_PER_DEG_LON_EQ * dlon) * cos_lat
    slope_col = g_col / dx_per_row[:, None]
    slope_pct = np.sqrt(slope_row ** 2 + slope_col ** 2) * 100.0
    return np.clip(slope_pct, 0.0, 1000.0).astype(np.float32)
