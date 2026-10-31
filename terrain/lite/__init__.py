# -*- coding: utf-8 -*-
"""terrain.lite — P2 Terrain Lite：设备端轻量地图生成（Python 参考实现）。

    LiteEngine   SeedPack 基场 + 程序化残差（fBm / 海岸 SDF / 轻侵蚀 / 河网）
    LiteParams   一次渲染的参数（size / zoom / cx / cy / quality / style）
    load_pack    读 ``.seedpack``
    render_png   一行出图

算法权威实现在 ``engine.py``；浏览器端 ``terrain/web`` 按同一套规则重写。
"""
from __future__ import annotations

from .engine import LiteEngine as LiteEngine
from .engine import LiteParams as LiteParams
from .engine import load_pack as load_pack
from .engine import render_png as render_png

__all__ = ["LiteEngine", "LiteParams", "load_pack", "render_png"]
