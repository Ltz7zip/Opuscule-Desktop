# -*- coding: utf-8 -*-
"""Terrain Lite（P2）测试：真包渲染 / 确定性 / 基场骨架保持 / 河网 / 群系。"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from terrain.lite import LiteEngine, LiteParams, load_pack, render_png
from terrain.pack.schema import m2v

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "output", "smoke.seedpack")


@pytest.fixture(scope="module")
def engine():
    return LiteEngine(load_pack(PACK))


@pytest.fixture(scope="module")
def rendered(engine):
    """一次 512² 渲染（后续断言复用，控制总耗时）。"""
    t0 = time.perf_counter()
    res = engine.render(LiteParams(size=512, quality="balanced"))
    return res, time.perf_counter() - t0


def test_shapes_and_dtypes(engine, rendered):
    res, _ = rendered
    rgb, height, biome = res["rgb"], res["height"], res["biome"]
    rivers = res["rivers_rgb"]
    assert rgb.shape == (512, 512, 3) and rgb.dtype == np.uint8
    assert height.shape == (512, 512) and height.dtype == np.float32
    assert biome.shape == (512, 512) and biome.dtype == np.uint8
    assert rivers.shape == (512, 512, 3) and rivers.dtype == np.uint8
    assert rgb.min() >= 0 and rgb.max() <= 255
    assert height.min() > -12000 and height.max() < 9000
    assert biome.max() <= 8
    stats = res["stats"]
    assert stats["size"] == 512 and stats["river_pixels"] > 0


def test_256(engine):
    t0 = time.perf_counter()
    res = engine.render(LiteParams(size=256, quality="fast"))
    assert res["rgb"].shape == (256, 256, 3)
    assert res["height"].shape == (256, 256)
    assert res["stats"]["ms"] > 0
    assert time.perf_counter() - t0 < 30.0


def test_deterministic(engine):
    p = LiteParams(size=256, zoom=1.6, cx=0.42, cy=0.55, quality="balanced")
    a = engine.render(p)
    b = engine.render(p)
    for key in ("rgb", "height", "biome", "rivers_rgb"):
        assert np.array_equal(a[key], b[key]), key
    # 参数变了，随机流必须变
    c = engine.render(LiteParams(size=256, zoom=1.6, cx=0.42, cy=0.55,
                                 quality="fast"))
    assert not np.array_equal(a["rgb"], c["rgb"])


def test_coast_skeleton_preserved(engine, rendered):
    """海岸带内：渲染高程（→v01）与 pack 基场相关性 > 0.9（骨架被保留）。"""
    res, _ = rendered
    p = LiteParams(size=512, quality="balanced")
    base01 = engine.sample_base(p)
    land = base01 >= 0.5
    dist = np.where(land, distance_transform_edt(land),
                    distance_transform_edt(~land))
    band = dist <= 12
    assert band.sum() > 1000
    v = np.asarray(m2v(res["height"]), np.float64)
    corr = float(np.corrcoef(base01[band].astype(np.float64), v[band])[0, 1])
    assert corr > 0.9, f"海岸带相关性过低：{corr:.3f}"


def test_rivers_and_biomes(engine, rendered):
    res, _ = rendered
    river_px = int((res["rivers_rgb"].any(axis=2)).sum())
    assert river_px > 0, "河网未叠加"
    assert np.unique(res["biome"]).size >= 3, "群系种类不足"


def test_render_png(tmp_path):
    out = tmp_path / "lite.png"
    stats = render_png(PACK, out, size=128, quality="fast")
    assert out.exists() and out.stat().st_size > 0
    assert stats["size"] == 128
    assert stats["png"] == str(out)
