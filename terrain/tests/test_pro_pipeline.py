# -*- coding: utf-8 -*-
"""Terrain Pro（P1）流水线测试：档位 / 瓦片 / GLB LOD(PBR) / MapPack 往返 / 批量。

代价控制：不跑 ``generate_world``，而是把现有 ``terrain/output/world_dem.npy``
（2048²）块均值降采样到 512 / 256 后作为 ``reuse_dem`` 使用。
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from terrain.pack import MapPack, dem_to_rgb, verify
from terrain.pro import (TIERS, Tier, batch_produce, build_lods, build_tiles,
                         produce, resolve_tier)
from terrain.pro.tiles import count_tiles

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORLD_DEM = os.path.join(_ROOT, "terrain", "output", "world_dem.npy")


# ==================== 工具 / fixture ====================
def downsample(path, size):
    """2048² DEM → size²（整数倍块均值）；返回 float32 数组。"""
    dem = np.load(path).astype(np.float32)
    k = dem.shape[0] // size
    assert dem.shape[0] % size == 0, f"{dem.shape} 不能被 {size} 整除"
    return dem.reshape(size, k, size, k).mean(axis=(1, 3)).astype(np.float32)


@pytest.fixture(scope="module")
def dem256():
    if not os.path.exists(WORLD_DEM):
        pytest.skip(f"缺少 {WORLD_DEM}")
    return downsample(WORLD_DEM, 256)


@pytest.fixture(scope="module")
def dem512_file(tmp_path_factory):
    if not os.path.exists(WORLD_DEM):
        pytest.skip(f"缺少 {WORLD_DEM}")
    p = tmp_path_factory.mktemp("pro_dem") / "world_dem_512.npy"
    np.save(p, downsample(WORLD_DEM, 512))
    return str(p)


@pytest.fixture(scope="module")
def tiny_tier():
    """极小档位：512 DEM 复用 + 小瓦片 / 小 LOD，几秒跑完。"""
    return Tier(name="test", size=512, seedpack_size=128, tile_levels=(0, 1),
                tile_size=64, lod_levels=(128, 256), target_seconds=60,
                description="unit-test tier", notes="pytest only")


def glb_json(blob):
    """GLB → 内嵌 glTF JSON dict（顺便校验容器头）。"""
    assert blob[:4] == b"glTF", "GLB magic 必须是 glTF"
    assert struct.unpack("<I", blob[4:8])[0] == 2
    total = struct.unpack("<I", blob[8:12])[0]
    assert total == len(blob)
    jlen, jtype = struct.unpack("<II", blob[12:20])
    assert jtype == 0x4E4F534A, "第一个 chunk 必须是 JSON"
    return json.loads(blob[20:20 + jlen].decode("utf-8"))


# ==================== 1. 档位 ====================
def test_tiers():
    assert set(("draft", "standard", "ultra", "research")) <= set(TIERS)
    assert resolve_tier("DRAFT").name == "draft"
    assert resolve_tier(" Ultra ").size == 4096
    std = resolve_tier("standard")
    assert std.size == 2048 and std.tile_levels == (0, 1, 2)
    assert std.lod_levels == (512, 1024, 2048)
    # 非法名 → ValueError，且消息里有可选值
    with pytest.raises(ValueError) as ei:
        resolve_tier("hyper")
    assert "draft" in str(ei.value) and "standard" in str(ei.value)
    # research 明确未实现
    with pytest.raises(NotImplementedError):
        resolve_tier("research")


# ==================== 2. 瓦片金字塔 ====================
def test_build_tiles_levels_and_sizes():
    from PIL import Image
    import io

    rgb = dem_to_rgb(np.random.RandomState(0).randn(200, 200).astype(np.float32)
                     * 500.0)
    tiles = build_tiles(rgb, levels=(0, 1, 2), tile_size=64)
    assert sorted(tiles) == [0, 1, 2]
    assert [len(tiles[lv]) for lv in (0, 1, 2)] == [1, 4, 16]
    assert count_tiles(tiles) == 21
    assert set(tiles[2]) == {(x, y) for x in range(4) for y in range(4)}
    # level 0 原图非 64 整数倍 → 必须被 resize 成 64×64
    with Image.open(io.BytesIO(tiles[0][(0, 0)])) as im:
        assert im.size == (64, 64)
    # 其余层是 tile_size 网格
    with Image.open(io.BytesIO(tiles[2][(3, 3)])) as im:
        assert im.size == (64, 64)


# ==================== 3. 3D LOD：GLB + PBR ====================
def test_build_lods_glb_and_pbr(dem256):
    rgb = dem_to_rgb(dem256)
    stats = {}
    lods = build_lods(dem256, rgb, levels=(128, 256), mode="plane",
                      stats_out=stats, verbose=False)
    assert sorted(lods) == [128, 256]
    for lv, blob in lods.items():
        g = glb_json(blob)
        mat = g["materials"][0]
        pbr = mat["pbrMetallicRoughness"]
        assert "baseColorTexture" in pbr
        assert "metallicFactor" in pbr and "roughnessFactor" in pbr
        assert "normalTexture" in mat
        assert g["buffers"][0]["byteLength"] > 0
        assert "uri" not in g["buffers"][0]          # GLB 用 BIN chunk
        assert stats[lv]["triangles"] > 0
        assert stats[lv]["grid"] == lv // stats[lv]["decimate"]


def test_build_lods_elevation_fallback(dem256):
    """不给 rgb 时退回高程着色，仍应产出合法 PBR GLB。"""
    lods = build_lods(dem256, None, levels=(128,), normal_map=False,
                      verbose=False)
    g = glb_json(lods[128])
    assert "pbrMetallicRoughness" in g["materials"][0]


# ==================== 4. MapPack 写盘 → 读回 → verify ====================
def test_mappack_roundtrip(dem256, tmp_path):
    rgb = dem_to_rgb(dem256)
    tiles = build_tiles(rgb, levels=(0, 1), tile_size=64)
    lods = build_lods(dem256, rgb, levels=(128,), verbose=False)
    mp = MapPack.build(seed=3, dem_m=dem256, rgb=rgb, tiles=tiles, lods=lods,
                       tier="draft")
    p = mp.write(tmp_path / "roundtrip.mappack")

    res = verify(p)
    assert res["ok"] is True, res["errors"]
    mp2 = MapPack.open(p)
    assert mp2.seed == 3 and mp2.size == 256
    assert np.allclose(mp2.dem_m, dem256, atol=1e-4)
    assert set(mp2.tiles[1]) == {(x, y) for x in range(2) for y in range(2)}
    assert mp2.lods[128][:4] == b"glTF"
    assert glb_json(mp2.lods[128])["materials"][0]["pbrMetallicRoughness"]


# ==================== 5. 批量生产 + ledger 续跑 ====================
def test_batch_produce_and_resume(dem512_file, tmp_path, tiny_tier):
    out = tmp_path / "pro"
    r1 = batch_produce([1, 2], tier=tiny_tier, out_dir=str(out),
                       reuse_dem=dem512_file, verbose=False)
    assert r1["success"] == 2 and r1["failed"] == 0
    assert r1["skipped"] == 0
    assert {r["seed"] for r in r1["results"]} == {1, 2}
    assert (out / "batch_summary.json").exists()

    ledger = out / "ledger.jsonl"
    lines = [json.loads(x) for x in ledger.read_text("utf-8").splitlines() if x]
    assert len(lines) == 2
    for rec in lines:
        assert rec["status"] == "ok"
        assert rec["pack"] and os.path.exists(rec["pack"])
        assert rec["lod_levels"] == [128, 256]
        assert rec["error"] is None

    # 两个包都能 verify
    for s in (1, 2):
        assert verify(out / f"seed{s}.mappack")["ok"]
        assert verify(out / f"seed{s}.seedpack")["ok"]
    timing = json.loads((out / "timing.json").read_text("utf-8"))
    assert timing["tier"] == "test" and timing["verify"]["mappack"] is True
    assert "stages" in timing and "peak_rss_mb" in timing

    # 再跑一次：resume 全部跳过，不再追加 ledger
    r2 = batch_produce([1, 2], tier=tiny_tier, out_dir=str(out),
                       reuse_dem=dem512_file, verbose=False)
    assert r2["skipped"] == 2 and r2["ran"] == 0
    assert len([x for x in ledger.read_text("utf-8").splitlines() if x]) == 2


def test_produce_reuse_and_force(dem512_file, tmp_path, tiny_tier):
    out = tmp_path / "reuse"
    r1 = produce(9, tier=tiny_tier, out_dir=str(out),
                 reuse_dem=dem512_file, verbose=False)
    assert r1["reused"] is False and r1["verify"]["mappack"] is True
    assert r1["tile_total"] == 5                      # level0(1) + level1(4)
    assert r1["lod_levels"] == [128, 256]
    r2 = produce(9, tier=tiny_tier, out_dir=str(out),
                 reuse_dem=dem512_file, verbose=False)
    assert r2["reused"] is True
    r3 = produce(9, tier=tiny_tier, out_dir=str(out),
                 reuse_dem=dem512_file, force=True, want_3d=False,
                 want_tiles=False, verbose=False)
    assert r3["reused"] is False and r3["lod_levels"] == []
