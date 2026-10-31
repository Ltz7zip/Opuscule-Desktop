# -*- coding: utf-8 -*-
"""io — 共享地图包的容器层：zip 读写、manifest、sha256、pack_hash、原子写。

对外：
    - ``PackWriter`` / ``PackReader``  底层条目读写（``add_npy`` / ``add_json`` /
      ``add_bytes`` / ``read`` / ``names`` / ``close``）
    - ``SeedPack`` / ``MapPack``       两种包的高层对象
    - ``verify`` / ``info``            校验与快速元信息

确定性：条目时间戳固定 1980-01-01，条目按路径字典序写入，manifest 规范化
JSON；写入先落 ``*.tmp`` 再 ``os.replace``，避免半截文件。同一份输入（含
``created_utc``）两次打包字节一致；需要完全复现时固定 ``created_utc`` 或
设置环境变量 ``SOURCE_DATE_EPOCH``。

本模块只依赖 numpy / PIL / 标准库，不导入 ``terrain.core``。
"""
from __future__ import annotations

import hashlib
import io as _bio
import json
import os
import warnings
import zipfile
from datetime import datetime, timezone

import numpy as np

from .schema import (
    FORMAT_MAPPACK, FORMAT_SEEDPACK, MANIFEST_NAME, PACK_VERSION, SLICES_INDEX,
    ZIP_EPOCH, HashMismatch, PackError, as_2d, as_rgb, check_version,
    default_style, dem_to_rgb, dumps_canonical, media_type_for, m2v,
    normalize_crs, normalize_lite_hints, normalize_slices_index,
    validate_manifest,
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _utc_now():
    """当前 UTC 时间，秒级：``2026-10-30T12:34:56Z``。

    支持 ``SOURCE_DATE_EPOCH``（可复现构建标准做法）：设定后取该时刻，
    这样同输入重跑也能得到完全相同的字节。
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        ts = datetime.fromtimestamp(int(epoch), timezone.utc)
    else:
        ts = datetime.now(timezone.utc)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _zip_info(name):
    """构造确定性 ZipInfo：固定时间戳、固定权限、固定 create_system。"""
    zi = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.create_system = 3                 # unix，避免跨平台元数据差异
    zi.external_attr = 0o644 << 16       # -rw-r--r--
    return zi


def _png_bytes(rgb):
    """uint8 (S,S,3) → PNG 字节。"""
    from PIL import Image
    buf = _bio.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, np.uint8), "RGB").save(buf, "PNG")
    return buf.getvalue()


def _png_array(data):
    """PNG 字节 → uint8 (S,S,3)。"""
    from PIL import Image
    with Image.open(_bio.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), np.uint8)


def _npy_bytes(arr):
    buf = _bio.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _npy_array(data):
    return np.load(_bio.BytesIO(data), allow_pickle=False)


# ==================== 1. 底层写入 / 读取 ====================
class PackWriter:
    """收集条目 → 生成 manifest → 一次性原子写出 zip。"""

    def __init__(self, path, fmt, *, seed, size, version=PACK_VERSION,
                 generator=None, license="CC-BY-4.0", provenance=None,
                 tier="standard", created_utc=None):
        if fmt not in (FORMAT_SEEDPACK, FORMAT_MAPPACK):
            raise PackError(f"未知格式：{fmt!r}")
        self.path = str(path)
        self.format = fmt
        self.seed = int(seed)
        self.size = int(size)
        self.version = version
        self.created_utc = created_utc or _utc_now()
        self.license = license
        self.provenance = dict(provenance or {})
        gen = {"name": "terrain-pro", "engine": "v12", "tier": tier}
        gen.update(generator or {})
        self.generator = gen
        self._entries = {}               # path -> (bytes, media_type)

    # ---- 添加条目 ----
    def add_bytes(self, path, data, media_type=None):
        if path == MANIFEST_NAME:
            raise PackError("manifest.json 由写入器自动生成，不能手工添加")
        if path in self._entries:
            raise PackError(f"条目重复：{path}")
        blob = bytes(data)
        self._entries[path] = (blob, media_type or media_type_for(path))
        return self

    def add_npy(self, path, arr):
        return self.add_bytes(path, _npy_bytes(np.asarray(arr)),
                              "application/x-npy")

    def add_json(self, path, obj):
        return self.add_bytes(path, dumps_canonical(obj).encode("utf-8"),
                              "application/json")

    # ---- manifest ----
    def manifest(self):
        """按当前条目生成完整 manifest（含 pack_hash）。"""
        entries = []
        for name in sorted(self._entries):
            blob, mt = self._entries[name]
            entries.append({"path": name, "sha256": _sha256(blob),
                            "bytes": len(blob), "media_type": mt})
        m = {
            "format": self.format,
            "version": self.version,
            "seed": self.seed,
            "size": self.size,
            "created_utc": self.created_utc,
            "generator": self.generator,
            "license": self.license,
            "provenance": self.provenance,
            "entries": entries,
        }
        m["pack_hash"] = _sha256(dumps_canonical(m).encode("utf-8"))
        return m

    # ---- 写出 ----
    def write(self):
        """原子写出到 ``self.path``，返回绝对路径。"""
        m = self.manifest()
        out = os.path.abspath(self.path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        tmp = out + ".tmp"
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                zf.writestr(_zip_info(MANIFEST_NAME),
                            dumps_canonical(m).encode("utf-8"))
                for name in sorted(self._entries):       # 条目顺序固定
                    zf.writestr(_zip_info(name), self._entries[name][0])
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return out

    def close(self):
        return self.write()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.write()
        return False


class PackReader:
    """读取一个包：``names()`` / ``read()`` / ``read_npy()`` / ``read_json()``。"""

    def __init__(self, path):
        self.path = str(path)
        try:
            self._zf = zipfile.ZipFile(self.path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise PackError(f"无法打开地图包 {self.path}：{exc}") from exc
        if MANIFEST_NAME not in self._zf.namelist():
            self._zf.close()
            raise PackError(f"{self.path} 不是地图包：缺少 manifest.json")
        try:
            self.manifest = self.read_json(MANIFEST_NAME)
            validate_manifest(self.manifest)
            self.warnings = check_version(self.manifest["version"])
            if self.warnings:                       # minor 更小：只告警不阻断
                warnings.warn("；".join(self.warnings), RuntimeWarning)
        except (ValueError, PackError) as exc:
            self._zf.close()
            if isinstance(exc, PackError):
                raise
            raise PackError(f"manifest.json 非法：{exc}") from exc

    # ---- 读取 ----
    def names(self):
        return [n for n in self._zf.namelist() if n != MANIFEST_NAME]

    def has(self, name):
        return name in self._zf.namelist()

    def read(self, name):
        try:
            return self._zf.read(name)
        except KeyError as exc:
            raise PackError(f"条目缺失：{name}") from exc

    def read_npy(self, name):
        return _npy_array(self.read(name))

    def read_json(self, name):
        return json.loads(self.read(name).decode("utf-8"))

    def entry(self, name):
        for e in self.manifest["entries"]:
            if e["path"] == name:
                return e
        raise PackError(f"清单中无此条目：{name}")

    def close(self):
        self._zf.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ==================== 2. SeedPack ====================
def _norm_slices(frames, size):
    """规整内存中的时间切片：``[{time_myr, sea_level, base}]``（按 time 升序）。"""
    if not frames:
        return []
    out = []
    for i, fr in enumerate(frames):
        if not isinstance(fr, dict) or "base" not in fr:
            raise PackError(f"slices[{i}] 必须是含 base 的对象")
        out.append({
            "time_myr": float(fr.get("time_myr", 0.0)),
            "sea_level": float(fr.get("sea_level", 0.0)),
            "base": as_2d(fr["base"], f"slices[{i}].base", np.float32,
                          (size, size)),
        })
    out.sort(key=lambda f: f["time_myr"])
    return out


def _read_slices(reader):
    """从包里读回时间切片；无 ``slices/index.json`` 时返回 None（v1 包）。"""
    if not reader.has(SLICES_INDEX):
        return None
    frames = normalize_slices_index(reader.read_json(SLICES_INDEX))
    out = []
    for fr in frames:
        if not reader.has(fr["entry"]):
            raise PackError(f"时间切片条目缺失：{fr['entry']}")
        out.append({"time_myr": fr["time_myr"], "sea_level": fr["sea_level"],
                    "base": reader.read_npy(fr["entry"])})
    return out


class SeedPack:
    """种子包：基础高程 + 气候 + 群系 + 简化河网 + 样式，供 Lite 再生成。

    v2 新增（均为可选，缺省自动补全）：``slices``（时间切片）、``sediment``
    （泥沙预算）、``lite_hints``（Lite 参数对齐）、``crs``（地理参照）。
    """

    format = FORMAT_SEEDPACK

    def __init__(self, seed, size, *, v01, dem_m=None, temperature=None,
                 precipitation=None, biome=None, hydro=None, style=None,
                 provenance=None, license="CC-BY-4.0", tier="standard",
                 generator=None, preview=None, created_utc=None, manifest=None,
                 version=PACK_VERSION, slices=None, sediment=None,
                 lite_hints=None, crs=None):
        self.seed = int(seed)
        self.size = int(size)
        self.version = str(version)
        self.v01 = as_2d(v01, "v01", np.float32, (self.size, self.size))
        self.dem_m = (None if dem_m is None else
                      as_2d(dem_m, "dem_m", np.float32, (self.size, self.size)))
        self.temperature = (None if temperature is None else
                            as_2d(temperature, "temperature", np.float32,
                                  (self.size, self.size)))
        self.precipitation = (None if precipitation is None else
                              as_2d(precipitation, "precipitation", np.float32,
                                    (self.size, self.size)))
        self.biome = (None if biome is None else
                      as_2d(biome, "biome", np.uint8, (self.size, self.size)))
        self.hydro = None if hydro is None else dict(hydro)
        self.style = default_style() if style is None else dict(style)
        self.provenance = dict(provenance or {})
        self.license = license
        self.tier = tier
        self.generator = dict(generator or {})
        self.preview = None if preview is None else as_rgb(preview, "preview")
        self.created_utc = created_utc or _utc_now()
        self.manifest = manifest
        # ---- v2 ----
        self.slices = _norm_slices(slices, self.size)
        self.sediment = (None if sediment is None else
                         as_2d(sediment, "sediment", np.float32,
                               (self.size, self.size)))
        self.lite_hints = normalize_lite_hints(lite_hints)
        self.crs = normalize_crs(crs, self.size)

    # ---- 构建 ----
    @property
    def base(self):
        """归一化高程 v01 ∈ [0,1]（对外名 ``base``，内部存 ``v01``）。"""
        return self.v01

    @property
    def shape(self):
        return (self.size, self.size)

    @classmethod
    def build(cls, seed, dem_m, v01=None, temperature=None, precipitation=None,
              biome=None, hydro=None, style=None, provenance=None,
              license="CC-BY-4.0", tier="standard", generator=None,
              preview=None, created_utc=None, slices=None, sediment=None,
              lite_hints=None, crs=None):
        """从内存中的场数据构建种子包。

        ``v01`` 为 None 时用 ``m2v(dem_m)`` 现算；``preview`` 为 None 时
        用 DEM 着色自动生成。
        """
        d = as_2d(dem_m, "dem_m", np.float32)
        if d.shape[0] != d.shape[1]:
            raise PackError(f"dem_m 必须是正方形，实际 {d.shape}")
        s = int(d.shape[0])
        v = m2v(d) if v01 is None else as_2d(v01, "v01", np.float32, (s, s))
        prev = dem_to_rgb(d) if preview is None else preview
        sp = cls(seed, s, v01=v, dem_m=d, temperature=temperature,
                 precipitation=precipitation, biome=biome, hydro=hydro,
                 style=style, provenance=provenance, license=license,
                 tier=tier, generator=generator, preview=prev,
                 created_utc=created_utc, slices=slices, sediment=sediment,
                 lite_hints=lite_hints, crs=crs)
        sp.manifest = sp._writer().manifest()
        return sp

    # ---- 打开 ----
    @classmethod
    def open(cls, path):
        """读取种子包；版本 major 不兼容时抛 ``PackError``。"""
        with PackReader(path) as r:
            m = r.manifest
            if m["format"] != FORMAT_SEEDPACK:
                raise PackError(f"不是种子包：format={m['format']!r}")
            size = int(m["size"])
            sp = cls(
                m["seed"], size,
                v01=r.read_npy("global/base.npy"),
                dem_m=r.read_npy("global/dem_m.npy") if r.has("global/dem_m.npy") else None,
                temperature=(r.read_npy("climate/temperature.npy")
                             if r.has("climate/temperature.npy") else None),
                precipitation=(r.read_npy("climate/precipitation.npy")
                               if r.has("climate/precipitation.npy") else None),
                biome=(r.read_npy("hydro/biome.npy")
                       if r.has("hydro/biome.npy") else None),
                hydro=(r.read_json("hydro/network.json")
                       if r.has("hydro/network.json") else None),
                style=(r.read_json("style.json") if r.has("style.json") else None),
                provenance=m.get("provenance") or {},
                license=m.get("license", "CC-BY-4.0"),
                tier=(m.get("generator") or {}).get("tier", "standard"),
                generator=(m.get("generator") or {}),
                preview=None,
                created_utc=m.get("created_utc"),
                manifest=m,
                version=m.get("version", "1.0.0"),
                slices=_read_slices(r),
                sediment=(r.read_npy("terrain/sediment.npy")
                          if r.has("terrain/sediment.npy") else None),
                lite_hints=(r.read_json("lite_hints.json")
                            if r.has("lite_hints.json") else None),
                crs=(r.read_json("crs.json") if r.has("crs.json") else None),
            )
            if r.has("global/preview.png"):
                sp.preview = _png_array(r.read("global/preview.png"))
            return sp

    # ---- 写盘 ----
    def _writer(self, path=""):
        w = PackWriter(path, self.format,
                       seed=self.seed, size=self.size, version=PACK_VERSION,
                       generator=self.generator,
                       license=self.license, provenance=self.provenance,
                       tier=self.tier, created_utc=self.created_utc)
        w.add_npy("global/base.npy", self.v01)
        if self.dem_m is not None:
            w.add_npy("global/dem_m.npy", self.dem_m)
        if self.preview is not None:
            w.add_bytes("global/preview.png", _png_bytes(self.preview),
                        "image/png")
        if self.temperature is not None:
            w.add_npy("climate/temperature.npy", self.temperature)
        if self.precipitation is not None:
            w.add_npy("climate/precipitation.npy", self.precipitation)
        if self.biome is not None:
            w.add_npy("hydro/biome.npy", self.biome)
        if self.hydro is not None:
            w.add_json("hydro/network.json", self.hydro)
        if self.slices:
            frames = []
            for i, fr in enumerate(self.slices):
                entry = f"slices/{i:03d}_base.npy"
                w.add_npy(entry, fr["base"])
                frames.append({"time_myr": float(fr["time_myr"]),
                               "sea_level": float(fr["sea_level"]),
                               "entry": entry})
            w.add_json(SLICES_INDEX, {"frames": frames})
        if self.sediment is not None:
            w.add_npy("terrain/sediment.npy", self.sediment)
        w.add_json("lite_hints.json", self.lite_hints)
        w.add_json("crs.json", self.crs)
        w.add_json("style.json", self.style)
        if self.provenance:
            w.add_json("provenance.json", self.provenance)
        return w

    def write(self, path):
        """写出到 ``path``；返回绝对路径。"""
        w = self._writer(str(path))
        self.manifest = w.manifest()
        return w.write()

    # 与规格中的 ``sp2.pack(path)`` 等价
    pack = write

    def __repr__(self):
        return f"<SeedPack seed={self.seed} size={self.size}>"


# ==================== 3. MapPack ====================
class MapPack:
    """成品包：DEM + 2D 渲染 + 可选瓦片/3D LOD/矢量要素，供 Web 直接分发。"""

    format = FORMAT_MAPPACK

    def __init__(self, seed, size, *, dem_m, rgb, tiles=None, lods=None,
                 vector=None, style=None, provenance=None, license="CC-BY-4.0",
                 tier="standard", generator=None, created_utc=None,
                 manifest=None, crs=None):
        self.seed = int(seed)
        self.size = int(size)
        self.dem_m = as_2d(dem_m, "dem_m", np.float32, (self.size, self.size))
        self.rgb = as_rgb(rgb)
        if self.rgb.shape[:2] != (self.size, self.size):
            raise PackError(f"rgb 形状应与 dem_m 一致，实际 {self.rgb.shape}")
        self.tiles = dict(tiles or {})        # level -> {(x,y): png bytes}
        self.lods = dict(lods or {})          # level -> glb bytes
        self.vector = None if vector is None else dict(vector)
        self.style = default_style() if style is None else dict(style)
        self.provenance = dict(provenance or {})
        self.crs = normalize_crs(crs, self.size)
        self.license = license
        self.tier = tier
        self.generator = dict(generator or {})
        self.created_utc = created_utc or _utc_now()
        self.manifest = manifest

    @classmethod
    def build(cls, seed, dem_m, rgb, tiles=None, lods=None, vector=None,
              style=None, provenance=None, tier="standard", generator=None,
              license="CC-BY-4.0", created_utc=None, crs=None):
        d = as_2d(dem_m, "dem_m", np.float32)
        if d.shape[0] != d.shape[1]:
            raise PackError(f"dem_m 必须是正方形，实际 {d.shape}")
        s = int(d.shape[0])
        # 统一瓦片键：{level: {(x,y): bytes}} 或扁平 {(x,y): bytes}（视为 level=0）
        tiles_n = {}
        if tiles:
            if all(isinstance(k, tuple) for k in tiles):
                tiles_n = {0: dict(tiles)}
            else:
                tiles_n = {int(k): dict(v) for k, v in tiles.items()}
        mp = cls(seed, s, dem_m=d, rgb=rgb, tiles=tiles_n, lods=lods,
                 vector=vector, style=style, provenance=provenance,
                 tier=tier, generator=generator, license=license,
                 created_utc=created_utc, crs=crs)
        mp.manifest = mp._writer("").manifest()
        return mp

    @classmethod
    def open(cls, path):
        with PackReader(path) as r:
            m = r.manifest
            if m["format"] != FORMAT_MAPPACK:
                raise PackError(f"不是成品包：format={m['format']!r}")
            size = int(m["size"])
            tiles, lods = {}, {}
            for name in r.names():
                if name.startswith("map/tiles/z"):
                    stem = name[len("map/tiles/z"):-len(".png")]
                    lvl, x, y = (int(v) for v in stem.split("_"))
                    tiles.setdefault(lvl, {})[(x, y)] = r.read(name)
                elif name.startswith("3d/lod"):
                    lvl = name[len("3d/lod"):-len(".glb")]
                    lods[int(lvl)] = r.read(name)
            mp = cls(m["seed"], size,
                     dem_m=r.read_npy("map/dem_m.npy"),
                     rgb=_png_array(r.read("map/rgb.png")),
                     tiles=tiles, lods=lods,
                     vector=(r.read_json("vector/features.geojson")
                             if r.has("vector/features.geojson") else None),
                     style=(r.read_json("style.json") if r.has("style.json") else None),
                     provenance=m.get("provenance") or {},
                     license=m.get("license", "CC-BY-4.0"),
                     tier=(m.get("generator") or {}).get("tier", "standard"),
                     generator=(m.get("generator") or {}),
                     created_utc=m.get("created_utc"),
                     crs=(r.read_json("crs.json") if r.has("crs.json") else None),
                     manifest=m)
            return mp

    def _writer(self, path=""):
        w = PackWriter(path, self.format, seed=self.seed, size=self.size,
                       generator=self.generator, license=self.license,
                       provenance=self.provenance, tier=self.tier,
                       created_utc=self.created_utc)
        w.add_npy("map/dem_m.npy", self.dem_m)
        w.add_bytes("map/rgb.png", _png_bytes(self.rgb), "image/png")
        for lvl in sorted(self.tiles):
            for (x, y) in sorted(self.tiles[lvl]):
                w.add_bytes(f"map/tiles/z{int(lvl)}_{int(x)}_{int(y)}.png",
                            self.tiles[lvl][(x, y)], "image/png")
        for lvl in sorted(self.lods):
            w.add_bytes(f"3d/lod{int(lvl)}.glb", self.lods[lvl],
                        "model/gltf-binary")
        if self.vector is not None:
            w.add_json("vector/features.geojson", self.vector)
        w.add_json("style.json", self.style)
        w.add_json("crs.json", self.crs)
        if self.provenance:
            w.add_json("provenance.json", self.provenance)
        return w

    def write(self, path):
        w = self._writer(str(path))
        self.manifest = w.manifest()
        return w.write()

    pack = write

    def __repr__(self):
        return f"<MapPack seed={self.seed} size={self.size}>"


# ==================== 4. 校验 / 元信息 ====================
def verify(path):
    """校验包完整性；任何问题写入 ``errors``，**不抛异常**。"""
    res = {"ok": True, "format": None, "version": None, "seed": None,
           "size": None, "pack_hash": None, "n_entries": 0, "bytes": 0,
           "sections": [], "errors": []}
    errors = res["errors"]
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if MANIFEST_NAME not in names:
                errors.append("缺少 manifest.json")
                res["ok"] = False
                return res
            if names[0] != MANIFEST_NAME:
                errors.append("manifest.json 不是 zip 的首个条目")
            try:
                m = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
                validate_manifest(m)
            except (ValueError, PackError) as exc:
                errors.append(f"manifest.json 非法：{exc}")
                res["ok"] = False
                return res

            res["format"] = m["format"]
            res["version"] = m["version"]
            res["seed"] = m["seed"]
            res["size"] = m["size"]
            res["pack_hash"] = m.get("pack_hash")
            entries = m["entries"]
            res["n_entries"] = len(entries)
            res["sections"] = sorted({n.split("/")[0] for n in names
                                      if n != MANIFEST_NAME})

            # ---- 清单自身未被改动 ----
            body = {k: v for k, v in m.items() if k != "pack_hash"}
            if m.get("pack_hash") != _sha256(dumps_canonical(body).encode("utf-8")):
                errors.append("pack_hash 校验失败：清单内容与哈希不一致")

            # ---- 版本兼容 ----
            try:
                check_version(m["version"])          # minor 偏旧只告警，不算失败
            except PackError as exc:
                errors.append(str(exc))

            # ---- 逐条目哈希 ----
            listed = set()
            for e in entries:
                name = e["path"]
                listed.add(name)
                if name not in names:
                    errors.append(f"{name}: 条目缺失")
                    continue
                data = zf.read(name)
                res["bytes"] += len(data)
                if e.get("bytes") != len(data):
                    errors.append(f"{name}: bytes 字段与实际不一致")
                if e.get("sha256") != _sha256(data):
                    errors.append(f"{name}: sha256 不匹配")
            for name in names:
                if name != MANIFEST_NAME and name not in listed:
                    errors.append(f"{name}: 清单中未登记的多余条目")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"无法作为 zip 读取：{exc}")
    res["ok"] = not errors
    return res


def info(path):
    """读取 manifest 并返回精简元信息（不校验哈希）。"""
    try:
        with zipfile.ZipFile(path) as zf:
            m = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
            names = [n for n in zf.namelist() if n != MANIFEST_NAME]
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise PackError(f"无法读取地图包 {path}：{exc}") from exc
    entries = [{"path": e.get("path"), "bytes": e.get("bytes")}
               for e in m.get("entries", [])]
    try:
        warnings_ = check_version(m.get("version", "1.0.0"))
    except PackError as exc:
        warnings_ = [str(exc)]
    return {
        "format": m.get("format"),
        "version": m.get("version"),
        "seed": m.get("seed"),
        "size": m.get("size"),
        "created_utc": m.get("created_utc"),
        "pack_hash": m.get("pack_hash"),
        "license": m.get("license"),
        "provenance": m.get("provenance") or {},
        "generator": m.get("generator"),
        "sections": sorted({n.split("/")[0] for n in names}),
        "slices": sum(1 for n in names if n.startswith("slices/")
                      and n.endswith(".npy")),
        "has_sediment": "terrain/sediment.npy" in names,
        "has_lite_hints": "lite_hints.json" in names,
        "has_crs": "crs.json" in names,
        "warnings": warnings_,
        "entries": entries,
        "entry_bytes": int(sum(e.get("bytes") or 0 for e in entries)),
    }


__all__ = [
    "PackWriter", "PackReader", "SeedPack", "MapPack", "verify", "info",
    "HashMismatch", "PackError", "m2v", "dem_to_rgb",
]
