# -*- coding: utf-8 -*-
"""terrain.modules — 地形引擎模块化框架。

将原本集中在 ``terrain.core.terrain_engine`` 中的物理/气候/水系/渲染逻辑
拆分为可独立优化的功能模块：

    - :class:`TerrainModule`      地形（高程 / 坡度 / 地貌 / 地形图）
    - :class:`HydroModule`        水系（流量累积 / 河流 / 湖泊 / 水系图）
    - :class:`TemperatureModule`  气温（大陆度 / 洋流 / 温度场 / 气温图）
    - :class:`ClimateModule`      气候（降水 / 生物群系 / 气候图）

每个模块继承 :class:`BaseModule`，通过 :class:`WorldContext` 共享输入数据
并把产出写入 ``ctx.data``。模块之间通过 ``dependencies`` 声明依赖，
:class:`WorldEngine` 按拓扑序执行，保证「改一个模块不影响其他模块运行」。

快速使用::

    from terrain.modules import WorldEngine, WorldContext
    eng = WorldEngine()
    ctx = eng.run(dem_m, lats)
    maps = eng.render_all(ctx)        # {terrain, hydro, temperature, climate} RGB
"""
from __future__ import annotations

from .base import BaseModule, WorldContext
from .engine import WorldEngine
from .terrain_module import TerrainModule
from .hydro_module import HydroModule
from .temperature_module import TemperatureModule
from .climate_module import ClimateModule

__all__ = [
    "BaseModule", "WorldContext", "WorldEngine",
    "TerrainModule", "HydroModule", "TemperatureModule", "ClimateModule",
]
__version__ = "1.0.0"
