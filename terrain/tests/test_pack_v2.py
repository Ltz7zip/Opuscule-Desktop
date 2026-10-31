# -*- coding: utf-8 -*-
"""Pack v2 契约测试：时间切片 / 泥沙 / lite_hints / crs / v1 只读兼容。"""
from __future__ import annotations

import json
import warnings
import zipfile

import numpy as np

from terrain.pack import PACK_VERSION, MapPack, SeedPack, info, verify
from terrain.pack.schema import default_crs, default_lite_hints

S = 32
FIXED_UTC = "2026-01-01T00:00:00Z"


# ==================== 测试数据 ====================
def make_dem(seed=3):
    rng = np.random.RandomState(seed)
    d = np.cumsum(np.cumsum(rng.randn(S, S), 0), 1) * 30.0
    d -= np.median(d)
    return np.clip(d, -4000.0, 5000.0).astype(np.float32)


def make_slices():
    """乱序给出三帧，验证读入后按 time_myr 升序。"""
    return [
        {"time_myr": 450.0, "sea_level": 120.0,
         "base": np.full((S, S), 0.55, np.float32)},
        {"time_myr": 0.0, "sea_level": -200.0,
         "base": np.full((S, S), 0.40, np.float32)},
        {"time_myr": 225.0, "sea_level": 0.0,
         "base": np.full((S, S), 0.50, np.float32)},
    ]


def make_seedpack(**kw):
    dem = make_dem()
    args = dict(
        seed=7, dem_m=dem, provenance={"model_version": "v13", "seed": 7},
        created_utc=FIXED_UTC, tier="standard",
        generator={"name": "terrain-pro", "engine": "v13", "tier": "standard"},
        slices=make_slices(),
        sediment=(np.abs(dem) / 5000.0).astype(np.float32),
        lite_hints={"coast_eps": 10.0, "slope_ref": 1800.0,
                    "hydro_min_acc": 250.0},
        crs={"projection": "equirect", "lat0": -90.0, "lon0": 0.0,
             "res_deg": 180.0 / S},
    )
    args.update(kw)
    return SeedPack.build(**args)


def rewrite_zip(src, dst, *, drop=(), manifest=None):
    """复制 zip：可删条目、可替换 manifest（模拟 v1 旧包）。"""
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            if it.filename in drop:
                continue
            if it.filename == "manifest.json" and manifest is not None:
                payload = json.dumps(manifest, sort_keys=True,
                                     separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")
                zout.writestr(it, payload)
            else:
                zout.writestr(it, zin.read(it.filename))
    return dst


def read_manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json").decode("utf-8"))


# ==================== 1. v2 往返 ====================
def test_v2_seedpack_roundtrip(tmp_path):
    sp = make_seedpack()
    assert sp.manifest["version"] == PACK_VERSION == "2.0.0"
    path = sp.write(tmp_path / "v2.seedpack")

    sp2 = SeedPack.open(path)
    assert sp2.version == "2.0.0"
    assert len(sp2.slices) == 3
    times = [fr["time_myr"] for fr in sp2.slices]
    assert times == sorted(times) == [0.0, 225.0, 450.0]
    assert np.allclose(sp2.slices[2]["base"], 0.55)
    assert sp2.slices[0]["sea_level"] == -200.0
    assert np.allclose(sp2.sediment, sp.sediment)
    assert sp2.lite_hints == {"coast_eps": 10.0, "slope_ref": 1800.0,
                              "hydro_min_acc": 250.0}
    assert sp2.crs["res_deg"] == 180.0 / S

    meta = info(path)
    assert meta["version"] == "2.0.0"
    assert meta["slices"] == 3
    assert meta["has_sediment"] and meta["has_lite_hints"] and meta["has_crs"]
    assert "slices" in meta["sections"]
    assert verify(path)["ok"] is True


def test_v2_mappack_crs(tmp_path):
    dem = make_dem()
    rgb = np.full((S, S, 3), 90, np.uint8)
    mp = MapPack.build(seed=1, dem_m=dem, rgb=rgb,
                       crs={"projection": "equirect", "lat0": -90.0,
                            "lon0": 0.0, "res_deg": 180.0 / S},
                       created_utc=FIXED_UTC)
    path = mp.write(tmp_path / "v2.mappack")
    mp2 = MapPack.open(path)
    assert mp2.crs["res_deg"] == 180.0 / S
    assert info(path)["has_crs"] is True


# ==================== 2. 确定性 ====================
def test_v2_pack_hash_deterministic(tmp_path):
    a = make_seedpack().write(tmp_path / "a.seedpack")
    b = make_seedpack().write(tmp_path / "b.seedpack")
    assert read_manifest(a)["pack_hash"] == read_manifest(b)["pack_hash"]


# ==================== 3. 缺省补全 ====================
def test_v2_defaults_filled(tmp_path):
    sp = make_seedpack(slices=None, sediment=None, lite_hints=None, crs=None)
    assert sp.slices == [] and sp.sediment is None
    assert sp.lite_hints == default_lite_hints()
    assert sp.crs == default_crs(S)
    path = sp.write(tmp_path / "min.seedpack")
    sp2 = SeedPack.open(path)
    assert sp2.slices == []
    assert sp2.lite_hints == default_lite_hints()


# ==================== 4. v1 只读兼容 ====================
def test_v1_readonly_then_write_upgrades(tmp_path):
    v2 = make_seedpack().write(tmp_path / "v2.seedpack")
    m = read_manifest(v2)
    m["version"] = "1.0.0"
    v1 = rewrite_zip(v2, tmp_path / "v1.seedpack",
                     drop=("slices/index.json", "crs.json", "lite_hints.json",
                           "terrain/sediment.npy"),
                     manifest=m)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp = SeedPack.open(v1)
    assert any("旧版" in str(w.message) for w in caught)
    assert sp.slices == [] and sp.sediment is None
    assert sp.crs == default_crs(S)            # 缺省补全
    assert sp.lite_hints == default_lite_hints()

    out = sp.write(tmp_path / "upgraded.seedpack")   # 写回必升 v2
    assert read_manifest(out)["version"] == "2.0.0"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "crs.json" in names and "lite_hints.json" in names
