# -*- coding: utf-8 -*-
"""共享地图包格式（P0）测试：往返 / 确定性 / 篡改检测 / 版本 / CLI。"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from terrain.pack import (
    FORMAT_MAPPACK, FORMAT_SEEDPACK, PACK_VERSION, MapPack, PackError,
    SeedPack, info, m2v, verify,
)
from terrain.pack import cli

S = 64
FIXED_UTC = "2026-01-01T00:00:00Z"


# ==================== 测试数据 ====================
def make_dem(seed=0):
    """随机游走式 DEM（米制），含正负高程。"""
    rng = np.random.RandomState(seed)
    d = np.cumsum(np.cumsum(rng.randn(S, S), 0), 1) * 20.0
    d = d - np.median(d)
    return np.clip(d, -4000.0, 5000.0).astype(np.float32)


def make_fields():
    dem = make_dem()
    temperature = (28.0 - np.abs(dem) / 400.0).astype(np.float32)
    precipitation = (300.0 + np.abs(dem) / 20.0).astype(np.float32)
    biome = (np.arange(S * S).reshape(S, S) % 9).astype(np.uint8)
    hydro = {
        "rivers": [[[1, 1], [1, 2], [2, 3]], [[30, 40], [31, 41], [31, 42], [32, 43]]],
        "lakes": [[[10, 10], [10, 12], [12, 12], [10, 10]]],
    }
    return dem, temperature, precipitation, biome, hydro


def make_seedpack(**kw):
    dem, t, p, b, h = make_fields()
    args = dict(seed=7, dem_m=dem, temperature=t, precipitation=p, biome=b,
                hydro=h, style={"palette": {"plain": "#123456"}},
                provenance={"origin": "unit-test"}, created_utc=FIXED_UTC)
    args.update(kw)
    return SeedPack.build(**args)


def rewrite_entry(src, dst, name, payload):
    """复制 zip 并替换其中一个条目的字节（模拟篡改）。"""
    with zipfile.ZipFile(src) as zin, \
            zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = payload if it.filename == name else zin.read(it.filename)
            zout.writestr(it, data)
    return dst


def read_manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json").decode("utf-8"))


def write_manifest(path, manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return rewrite_entry(path, str(path) + ".mod", "manifest.json", payload)


# ==================== 1. SeedPack 往返 ====================
def test_seedpack_roundtrip(tmp_path):
    dem, t, p, b, h = make_fields()
    sp = make_seedpack()
    assert sp.seed == 7 and sp.size == S
    assert sp.manifest["format"] == FORMAT_SEEDPACK
    assert sp.manifest["version"] == PACK_VERSION

    path = sp.write(tmp_path / "a.seedpack")
    sp2 = SeedPack.open(path)

    assert sp2.seed == 7 and sp2.size == S
    assert np.array_equal(sp2.base, sp.base)
    assert np.allclose(sp2.base, sp.v01)
    assert np.array_equal(sp2.dem_m, dem)
    assert np.array_equal(sp2.temperature, t)
    assert np.array_equal(sp2.precipitation, p)
    assert np.array_equal(sp2.biome, b)
    assert sp2.hydro == h
    assert sp2.style["palette"]["plain"] == "#123456"
    assert sp2.provenance == {"origin": "unit-test"}
    # 未传 preview 时自动用 DEM 着色生成
    assert sp2.preview is not None and sp2.preview.shape == (S, S, 3)
    assert sp2.preview.dtype == np.uint8


def test_seedpack_optional_fields_absent(tmp_path):
    """只给 dem_m 时，气候 / 群系 / 河网读回为 None，base 由 m2v 现算。"""
    dem = make_dem()
    sp = SeedPack.build(seed=3, dem_m=dem, created_utc=FIXED_UTC)
    path = sp.write(tmp_path / "min.seedpack")
    sp2 = SeedPack.open(path)
    assert sp2.temperature is None and sp2.precipitation is None
    assert sp2.biome is None and sp2.hydro is None
    assert np.allclose(sp2.base, m2v(dem), atol=1e-6)
    assert sp2.base.dtype == np.float32


# ==================== 2. 确定性 ====================
def test_deterministic_bytes(tmp_path):
    sp = make_seedpack()
    p1 = sp.write(tmp_path / "one.seedpack")
    p2 = sp.pack(tmp_path / "two.seedpack")      # 别名，同一对象再写一次
    assert (tmp_path / "one.seedpack").read_bytes() == \
           (tmp_path / "two.seedpack").read_bytes()
    assert SeedPack.open(p1).manifest["pack_hash"] == \
           SeedPack.open(p2).manifest["pack_hash"]

    # 两份独立构建、同一 created_utc → 完全相同的字节
    a = make_seedpack().write(tmp_path / "a.seedpack")
    b = make_seedpack().write(tmp_path / "b.seedpack")
    assert (tmp_path / "a.seedpack").read_bytes() == \
           (tmp_path / "b.seedpack").read_bytes()
    assert SeedPack.open(a).manifest["pack_hash"] == \
           SeedPack.open(b).manifest["pack_hash"]


# ==================== 3. 条目篡改检测 ====================
def test_entry_tamper_detected(tmp_path):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    bad = str(tmp_path / "bad.seedpack")
    payload = b"not a real npy file"
    rewrite_entry(path, bad, "global/base.npy", payload)

    res = verify(bad)
    assert res["ok"] is False
    assert any("global/base.npy" in e for e in res["errors"])


# ==================== 4. manifest 篡改检测 ====================
def test_manifest_tamper_detected(tmp_path):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    m = read_manifest(path)
    m["seed"] = 999                       # 改内容，但不动 pack_hash
    bad = write_manifest(path, m)

    res = verify(bad)
    assert res["ok"] is False
    assert any("pack_hash" in e for e in res["errors"])
    assert res["seed"] == 999


# ==================== 5. 版本兼容 ====================
def test_incompatible_version(tmp_path):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    m = read_manifest(path)
    m["version"] = "9.0.0"
    bad = write_manifest(path, m)

    with pytest.raises(PackError):
        SeedPack.open(bad)


# ==================== 6. MapPack 往返 ====================
def png_bytes(rgb):
    import io as _bio
    from PIL import Image
    buf = _bio.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, "PNG")
    return buf.getvalue()


def test_mappack_roundtrip(tmp_path):
    dem = make_dem(1)
    rng = np.random.RandomState(2)
    rgb = rng.randint(0, 256, (S, S, 3)).astype(np.uint8)
    tiles = {0: {(0, 0): png_bytes(rgb[:32, :32]),
                 (1, 0): png_bytes(rgb[:32, 32:])}}
    lods = {512: b"glTF-fake-binary"}
    vector = {"type": "FeatureCollection", "features": []}

    mp = MapPack.build(seed=11, dem_m=dem, rgb=rgb, tiles=tiles, lods=lods,
                       vector=vector, created_utc=FIXED_UTC)
    path = mp.write(tmp_path / "map.mappack")
    mp2 = MapPack.open(path)

    assert mp2.seed == 11 and mp2.size == S
    assert mp2.manifest["format"] == FORMAT_MAPPACK
    assert mp2.rgb.dtype == np.uint8
    assert np.array_equal(mp2.rgb, rgb)                 # 逐字节相等
    assert np.array_equal(mp2.dem_m, dem)
    assert set(mp2.tiles) == {0} and len(mp2.tiles[0]) == 2
    assert mp2.lods[512] == b"glTF-fake-binary"
    assert mp2.vector == vector
    assert verify(path)["ok"] is True


# ==================== 7. CLI ====================
def test_cli_verify_and_info(tmp_path, capsys):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    assert cli.main(["verify", path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["n_entries"] > 0 and out["bytes"] > 0

    assert cli.main(["info", path]) == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["format"] == FORMAT_SEEDPACK
    assert meta["seed"] == 7 and meta["size"] == S
    assert any(e["path"] == "global/base.npy" for e in meta["entries"])


def test_cli_verify_corrupted_returns_1(tmp_path, capsys):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    bad = str(tmp_path / "bad.seedpack")
    rewrite_entry(path, bad, "global/base.npy", b"broken")
    assert cli.main(["verify", bad]) == 1
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] is False


def test_cli_unpack(tmp_path):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    dest = tmp_path / "unpacked"
    assert cli.main(["unpack", path, "--to", str(dest)]) == 0
    assert (dest / "manifest.json").exists()
    assert (dest / "global" / "base.npy").exists()
    assert (dest / "style.json").exists()


# ==================== 附加：manifest 基本结构 ====================
def test_manifest_structure(tmp_path):
    path = make_seedpack().write(tmp_path / "ok.seedpack")
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist()[0] == "manifest.json"
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in zf.infolist())
    m = read_manifest(path)
    paths = [e["path"] for e in m["entries"]]
    assert paths == sorted(paths)
    assert m["generator"]["name"] == "terrain-pro"
    assert m["license"] == "CC-BY-4.0"
    assert info(path)["version"] == PACK_VERSION
