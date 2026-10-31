# -*- coding: utf-8 -*-
"""tiles — 2D 瓦片金字塔（Web Mercator 式四叉树，纯 PIL 逐级重采样）。

层级约定与 ``MapPack`` 一致：
    level 0 → 全图 1 张（``tile_size × tile_size``）
    level n → ``2^n × 2^n`` 张，按行主序切分，键为 ``(x, y)``（x 向右、y 向下）

输出结构 ``{level: {(x, y): png_bytes}}`` 可直接交给
``terrain.pack.MapPack.build(tiles=...)``。

本模块只依赖 numpy / PIL，不导入 terrain.core。
"""
from __future__ import annotations

import io

import numpy as np


def _as_image(rgb):
    """uint8 (H,W,3) → PIL.Image（RGB）。"""
    from PIL import Image
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb 必须是 (H,W,3)，实际 {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr), "RGB")


def build_tiles(rgb, levels=(0, 1, 2), tile_size=256):
    """把 RGB 渲染图切成四叉树瓦片金字塔。

    Parameters
    ----------
    rgb : (H,W,3) uint8
        全分辨率 2D 渲染图（可以不是 ``tile_size`` 的整数倍）。
    levels : Iterable[int]
        需要的层级；缺省 ``(0, 1, 2)``。
    tile_size : int
        单张瓦片边长，缺省 256（Web 地图惯例）。

    Returns
    -------
    dict[int, dict[tuple[int,int], bytes]]
        ``{level: {(x, y): png}}``。level 0 若原图不是 ``tile_size`` 整数倍，
        会先 resize 成 ``tile_size × tile_size``。
    """
    from PIL import Image

    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError(f"tile_size 必须为正，实际 {tile_size}")
    levels = sorted({int(lv) for lv in levels})
    if any(lv < 0 for lv in levels):
        raise ValueError(f"level 不能为负：{levels}")
    img = _as_image(rgb)

    out = {}
    for lv in levels:
        n = 1 << lv
        side = tile_size * n
        im = img if img.size == (side, side) else img.resize(
            (side, side), Image.LANCZOS)
        layer = {}
        for y in range(n):
            for x in range(n):
                box = (x * tile_size, y * tile_size,
                       (x + 1) * tile_size, (y + 1) * tile_size)
                buf = io.BytesIO()
                im.crop(box).save(buf, format="PNG", optimize=False)
                layer[(x, y)] = buf.getvalue()
        out[lv] = layer
    return out


def count_tiles(tiles):
    """统计瓦片总数（``sum(len(layer))``）；空输入返回 0。"""
    return int(sum(len(v) for v in (tiles or {}).values()))


__all__ = ["build_tiles", "count_tiles"]
