# -*- coding: utf-8 -*-
"""modules.base — 模块化引擎的基类与共享上下文。

设计目标
--------
1. **解耦**：每个功能模块只通过 :class:`WorldContext` 读写数据，不直接 import
   其他模块的内部实现。模块间唯一的耦合是 ``dependencies`` 中声明的名字。
2. **独立优化**：修改某个模块的 ``compute`` / ``render`` 实现，只要它仍向
   ``ctx.data`` 写入约定的键，其他模块和引擎整体行为不受影响。
3. **可扩展**：新增模块只需继承 :class:`BaseModule`、实现 ``compute`` 与
   ``render``、注册到 :class:`WorldEngine`，引擎会自动按依赖序执行。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class WorldContext:
    """模块间共享的世界上下文。

    保存一份已生成的米制 DEM 以及纬度坐标，作为所有模块的共同输入；
    模块的计算产物通过 ``data`` 字典传递（键约定见各模块文档字符串）。

    Parameters
    ----------
    dem_m : ndarray (S, S), float
        米制高程（正=陆地 / 负=海洋），已由 ``generate_world`` 产出。
    lats : ndarray (S,)
        像素中心对齐的纬度数组（-90° → +90°，递增）。
    rng : numpy.random.RandomState or None
        全局随机数生成器；None 时用 ``RandomState(0)``。
    """

    def __init__(self, dem_m: np.ndarray, lats: np.ndarray,
                 rng: Optional[np.random.RandomState] = None) -> None:
        """校验并保存 DEM / 纬度输入，初始化陆地掩膜、随机源与数据字典。"""
        self.dem_m = np.asarray(dem_m, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        if self.dem_m.ndim != 2 or self.dem_m.shape[0] != self.dem_m.shape[1]:
            raise ValueError(f"dem_m 必须是二维方阵，实际 shape={self.dem_m.shape}")
        if self.lats.shape[0] != self.dem_m.shape[0]:
            raise ValueError(
                f"lats 长度 {self.lats.shape[0]} 与 dem_m 尺寸 "
                f"{self.dem_m.shape[0]} 不匹配")
        self.S = self.dem_m.shape[0]
        self.land_mask = self.dem_m > 0.0
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.data: Dict[str, Any] = {}

    # ---- 便捷访问 -------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """读取 ``data`` 中键 ``key`` 的值，缺省时返回 ``default``。"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """向 ``data`` 写入键值对，供下游模块按约定键读取。"""
        self.data[key] = value

    def has(self, key: str) -> bool:
        """判断 ``data`` 中是否已存在键 ``key``。"""
        return key in self.data


class BaseModule:
    """功能模块基类。

    子类必须实现：
        - :meth:`compute` —— 从 ``ctx`` 读取输入，把产出写入 ``ctx.data``。
        - :meth:`render`  —— 把本模块的产物渲染为 ``(S, S, 3)`` RGB 数组。

    属性：
        - ``name``         —— 模块唯一标识（用于依赖解析与输出键前缀）。
        - ``dependencies`` —— 依赖的模块名列表；引擎保证它们先于本模块执行。
    """

    name: str = "base"
    dependencies: List[str] = []

    def compute(self, ctx: WorldContext) -> None:
        """从 ``ctx`` 读取依赖模块的输出键，计算并把产物写入 ``ctx.data``。"""
        raise NotImplementedError

    def render(self, ctx: WorldContext) -> np.ndarray:
        """返回 ``(S, S, 3)`` float32 RGB 图像（值域 [0, 1]）。"""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        """返回模块的调试用字符串表示（类名与模块名）。"""
        return f"<{self.__class__.__name__} name={self.name!r}>"
