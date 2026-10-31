# -*- coding: utf-8 -*-
"""modules.engine — 模块化引擎编排器。

:class:`WorldEngine` 负责按依赖拓扑序执行各模块，并提供统一的地图生成入口。

扩展性
------
新增功能模块只需：
    1. 继承 :class:`BaseModule` 实现 ``compute`` / ``render``；
    2. 调用 ``engine.register(MyModule())``；
    3. 引擎自动按 ``dependencies`` 把它插入执行序列。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np

from .base import BaseModule, WorldContext
from .terrain_module import TerrainModule
from .hydro_module import HydroModule
from .temperature_module import TemperatureModule
from .climate_module import ClimateModule


class WorldEngine:
    """世界引擎：按依赖序运行模块，产出四类世界地图。

    Parameters
    ----------
    modules : list of BaseModule or None
        自定义模块列表；None 时注册默认的地形/水系/气温/气候四模块。
    """

    DEFAULT_MODULES: List[Type[BaseModule]] = [
        TerrainModule, HydroModule, TemperatureModule, ClimateModule,
    ]

    def __init__(self, modules: Optional[List[BaseModule]] = None) -> None:
        """初始化引擎：None 时注册默认四模块，否则依次注册给定模块。"""
        self._modules: Dict[str, BaseModule] = {}
        if modules is None:
            for cls in self.DEFAULT_MODULES:
                self.register(cls())
        else:
            for m in modules:
                self.register(m)

    # ---- 模块管理 -------------------------------------------------------
    def register(self, module: BaseModule) -> "WorldEngine":
        """注册一个模块；同名模块会被覆盖。"""
        self._modules[module.name] = module
        return self

    def unregister(self, name: str) -> "WorldEngine":
        """注销名为 ``name`` 的模块（不存在时静默忽略），返回自身。"""
        self._modules.pop(name, None)
        return self

    def get(self, name: str) -> Optional[BaseModule]:
        """按名字取回已注册的模块；未注册时返回 None。"""
        return self._modules.get(name)

    @property
    def module_names(self) -> List[str]:
        """返回当前已注册的全部模块名列表。"""
        return list(self._modules.keys())

    # ---- 拓扑排序 -------------------------------------------------------
    def _topo_order(self) -> List[BaseModule]:
        """按 ``dependencies`` 做拓扑排序；缺依赖时抛 ValueError。"""
        names = set(self._modules)
        # 校验依赖存在
        for m in self._modules.values():
            for dep in m.dependencies:
                if dep not in names:
                    raise ValueError(
                        f"模块 {m.name!r} 依赖未注册的模块 {dep!r}")
        # Kahn 算法
        in_degree = {n: 0 for n in names}
        for m in self._modules.values():
            for dep in m.dependencies:
                # dep 必须先于 m 执行 → m 的入度 +1
                in_degree[m.name] = in_degree.get(m.name, 0) + 1
        order: List[BaseModule] = []
        # 反向邻接：dep → 依赖它的模块
        reverse: Dict[str, List[str]] = {n: [] for n in names}
        for m in self._modules.values():
            for dep in m.dependencies:
                reverse[dep].append(m.name)
        ready = [n for n, d in in_degree.items() if d == 0]
        while ready:
            n = ready.pop(0)
            order.append(self._modules[n])
            for nxt in reverse.get(n, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(self._modules):
            raise ValueError("模块依赖存在环，无法排序")
        return order

    # ---- 执行 -----------------------------------------------------------
    def run(self, dem_m: np.ndarray, lats: np.ndarray,
            rng: Optional[np.random.RandomState] = None,
            tect_ctx: Optional[dict] = None) -> WorldContext:
        """依次执行所有模块，返回填充完毕的 :class:`WorldContext`。

        Parameters
        ----------
        dem_m, lats : 见 :class:`WorldContext`。
        rng : 随机数生成器；None 时用 ``RandomState(0)``。
        tect_ctx : 构造场上下文，注入地形模块供地貌分类使用。
        """
        ctx = WorldContext(dem_m, lats, rng=rng)
        if tect_ctx is not None:
            ctx.set("terrain.tect_ctx", tect_ctx)
        for module in self._topo_order():
            module.compute(ctx)
        return ctx

    def render(self, ctx: WorldContext, name: str) -> np.ndarray:
        """渲染指定模块的地图 RGB。"""
        module = self._modules.get(name)
        if module is None:
            raise KeyError(f"未注册模块 {name!r}")
        return module.render(ctx)

    def render_all(self, ctx: WorldContext) -> Dict[str, np.ndarray]:
        """渲染所有已注册模块的地图，返回 ``{module_name: rgb}``。"""
        return {name: m.render(ctx) for name, m in self._modules.items()}

    # ---- 高层便捷接口 ---------------------------------------------------
    def generate_world_maps(self, dem_m: np.ndarray, lats: np.ndarray,
                            rng: Optional[np.random.RandomState] = None,
                            tect_ctx: Optional[dict] = None
                            ) -> Dict[str, np.ndarray]:
        """一站式：执行所有模块并返回四类世界地图 RGB。

        Returns
        -------
        dict with keys ``terrain`` / ``hydro`` / ``temperature`` / ``climate``
            每个值是 ``(S, S, 3)`` float32 RGB 数组（[0, 1]）。
        """
        ctx = self.run(dem_m, lats, rng=rng, tect_ctx=tect_ctx)
        return self.render_all(ctx)
