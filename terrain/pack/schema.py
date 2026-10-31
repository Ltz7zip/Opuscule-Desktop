# -*- coding: utf-8 -*-
"""schema — 共享地图包（SeedPack / MapPack）的格式常量、版本规则与字段校验。

本模块是 pack 子包里最底层的一层：只依赖 numpy，**不得**导入 ``terrain.core``
或任何需要 torch / numba 的模块（Lite 侧要能单独复用这一层）。

内容：
    - 格式常量：``FORMAT_SEEDPACK`` / ``FORMAT_MAPPACK`` / ``PACK_VERSION``
    - 高程归一化：``m2v`` / ``v2m``（分段线性，与引擎同源）
    - 版本兼容：``parse_version`` / ``check_version``
    - 清单校验：``validate_manifest``
    - 数组与色彩工具：``as_2d`` / ``as_rgb`` / ``dem_to_rgb`` / ``default_style``
"""
from __future__ import annotations

import json

import numpy as np

# ==================== 1. 格式常量 ====================
FORMAT_SEEDPACK = "seedpack"
FORMAT_MAPPACK = "mappack"
PACK_VERSION = "2.0.0"
# 可读取的 major 版本：v1 只读兼容（缺省字段加载时补全），v2 为当前可写版本。
READABLE_MAJORS = (1, 2)

MANIFEST_NAME = "manifest.json"
# zip 条目时间戳固定值（1980-01-01 00:00:00 是 zip 格式允许的最早时间）——
# 保证同一份输入两次打包字节一致。
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


class PackError(Exception):
    """地图包结构 / 版本 / 缺条目等硬错误。"""


class HashMismatch(PackError):
    """条目或清单的 sha256 校验失败。"""


# ==================== 2. 米制高程 <-> 归一化 v01 ====================
# 与 terrain.core.terrain_engine.m2v 同源，复制以避免 pack 依赖 torch。
M_KNOTS = np.array([-11000, -6000, -4000, -2500, -1000, -200, 0,
                    100, 400, 1000, 2000, 3500, 5000, 8849], np.float64)
V_KNOTS = np.array([0.00, 0.08, 0.22, 0.34, 0.42, 0.47, 0.50,
                    0.53, 0.60, 0.70, 0.82, 0.92, 0.97, 1.00], np.float64)


def m2v(m):
    """米制高程 → 归一化 [0,1]（两端由 ``np.interp`` 钳制）。"""
    return np.interp(np.asarray(m, np.float64), M_KNOTS, V_KNOTS).astype(np.float32)


def v2m(v):
    """归一化 [0,1] → 米制高程（``m2v`` 的严格逆映射）。"""
    return np.interp(np.asarray(v, np.float64), V_KNOTS, M_KNOTS)


# ==================== 3. 规范化 JSON / 校验 ====================
def dumps_canonical(obj):
    """规范化 JSON 文本：键排序 + 紧凑分隔符 + 保留中文。

    manifest 的字节与 pack_hash 都基于它，保证同一份内容的哈希稳定。
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def parse_version(version):
    """把 ``"1.2.3"`` 解析为 ``(1, 2, 3)``。"""
    parts = str(version).split(".")
    if len(parts) != 3:
        raise PackError(f"版本号格式非法：{version!r}（期望 major.minor.patch）")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise PackError(f"版本号格式非法：{version!r}") from exc


def check_version(version, current=PACK_VERSION):
    """版本兼容检查。

    * major 不在 ``READABLE_MAJORS`` 或高于当前 major → 抛 ``PackError``；
    * major 更旧（v1 包在 v2 下）→ 返回"只读兼容"告警，加载时补全缺省字段；
    * minor 更小 → 返回向后兼容告警。
    """
    major, minor, _ = parse_version(version)
    cur_major, cur_minor, _ = parse_version(current)
    if major not in READABLE_MAJORS or major > cur_major:
        raise PackError(
            f"不兼容的包版本 {version}（当前 {current}）："
            f"可读 major 为 {', '.join(str(v) for v in READABLE_MAJORS)}")
    if major < cur_major:
        return [f"包版本 {version} 为旧版（当前 {current}），按只读兼容读取，"
                f"缺省字段自动补全；写回将升级为 {current}"]
    if minor < cur_minor:
        return [f"包版本 {version} 低于当前 {current}，按向后兼容方式读取"]
    return []


MANIFEST_KEYS = ("format", "version", "seed", "size", "created_utc",
                 "generator", "license", "provenance", "entries", "pack_hash")


def validate_manifest(m):
    """校验 manifest 的必备字段与基本类型；不合法抛 ``PackError``。"""
    if not isinstance(m, dict):
        raise PackError("manifest 必须是 JSON 对象")
    missing = [k for k in MANIFEST_KEYS if k not in m]
    if missing:
        raise PackError(f"manifest 缺少字段：{', '.join(missing)}")
    if m["format"] not in (FORMAT_SEEDPACK, FORMAT_MAPPACK):
        raise PackError(f"未知格式标识：{m['format']!r}")
    if not isinstance(m["size"], int) or m["size"] <= 0:
        raise PackError(f"size 非法：{m['size']!r}")
    if not isinstance(m["seed"], int):
        raise PackError(f"seed 非法：{m['seed']!r}")
    if not isinstance(m["entries"], list):
        raise PackError("entries 必须是数组")
    for e in m["entries"]:
        if not isinstance(e, dict) or "path" not in e or "sha256" not in e:
            raise PackError(f"条目字段缺失：{e!r}")
    if not isinstance(m["generator"], dict):
        raise PackError("generator 必须是对象")
    if not isinstance(m["provenance"], dict):
        raise PackError("provenance 必须是对象")


# ==================== 3.5 v2 新增字段 ====================
LITE_HINTS_KEYS = ("coast_eps", "slope_ref", "hydro_min_acc")
LITE_HINTS_DEFAULT = {"coast_eps": 8.0, "slope_ref": 2000.0,
                      "hydro_min_acc": 300.0}
CRS_KEYS = ("projection", "lat0", "lon0", "res_deg")
SLICES_INDEX = "slices/index.json"


def default_lite_hints():
    """Lite 侧与 Pro 对齐所需的算法参数（v1 包缺省时用它补全）。"""
    return dict(LITE_HINTS_DEFAULT)


def default_crs(size):
    """等距圆柱（equirectangular）默认地理参照：行 0 为南极。"""
    return {"projection": "equirect", "lat0": -90.0, "lon0": 0.0,
            "res_deg": round(180.0 / float(size), 9)}


def normalize_lite_hints(hints):
    """校验并补全 ``lite_hints``；非法值抛 ``PackError``。"""
    out = default_lite_hints()
    if hints is None:
        return out
    if not isinstance(hints, dict):
        raise PackError("lite_hints 必须是对象")
    for k in LITE_HINTS_KEYS:
        if k in hints:
            try:
                out[k] = float(hints[k])
            except (TypeError, ValueError) as exc:
                raise PackError(f"lite_hints.{k} 非法：{hints[k]!r}") from exc
            if out[k] <= 0:
                raise PackError(f"lite_hints.{k} 必须为正数，实际 {out[k]}")
    return out


def normalize_crs(crs, size):
    """校验并补全 ``crs``；缺省按 ``size`` 生成等距圆柱。"""
    out = default_crs(size)
    if crs is None:
        return out
    if not isinstance(crs, dict):
        raise PackError("crs 必须是对象")
    for k in CRS_KEYS:
        if k in crs:
            out[k] = crs[k] if k == "projection" else float(crs[k])
    if not isinstance(out["projection"], str) or not out["projection"]:
        raise PackError(f"crs.projection 非法：{out['projection']!r}")
    return out


def normalize_slices_index(idx):
    """校验 ``slices/index.json``：``{frames:[{time_myr, sea_level, entry}]}``。

    返回规整后的帧列表（按 ``time_myr`` 升序）；缺 ``frames`` 视为空切片。
    """
    if idx is None:
        return []
    if not isinstance(idx, dict) or not isinstance(idx.get("frames"), list):
        raise PackError("slices/index.json 必须是 {frames: [...]} 对象")
    frames = []
    for i, fr in enumerate(idx["frames"]):
        if not isinstance(fr, dict) or "entry" not in fr:
            raise PackError(f"slices/index.json frames[{i}] 缺少 entry")
        try:
            t = float(fr.get("time_myr", 0.0))
            sea = float(fr.get("sea_level", 0.0))
        except (TypeError, ValueError) as exc:
            raise PackError(f"slices/index.json frames[{i}] 数值非法") from exc
        frames.append({"time_myr": t, "sea_level": sea,
                       "entry": str(fr["entry"])})
    frames.sort(key=lambda f: f["time_myr"])
    return frames


def media_type_for(path):
    """按扩展名推断条目的 media type。"""
    if path.endswith(".npy"):
        return "application/x-npy"
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".geojson"):
        return "application/geo+json"
    if path.endswith(".glb"):
        return "model/gltf-binary"
    if path.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


# ==================== 4. 数组 / 色彩工具 ====================
def as_2d(arr, name, dtype, shape=None):
    """统一转成连续的二维数组并校验形状。"""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise PackError(f"{name} 必须是二维数组，实际 {a.ndim} 维")
    if shape is not None and a.shape != tuple(shape):
        raise PackError(f"{name} 形状应为 {tuple(shape)}，实际 {a.shape}")
    return np.ascontiguousarray(a, dtype=dtype)


def as_rgb(arr, name="rgb"):
    """校验并转成 uint8 (S,S,3) 的 RGB 数组。"""
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise PackError(f"{name} 必须是 (S,S,3) 的 RGB 数组，实际 {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def default_style():
    """默认调色 / 光照 / 标签规则（三家产品线共用的基础样式）。"""
    return {
        "palette": {
            "ocean_deep": "#0b2d5b",
            "ocean_shallow": "#2b6ea8",
            "beach": "#d9d3a6",
            "plain": "#5d8a4a",
            "forest": "#2f5d33",
            "mountain": "#8a7a66",
            "snow": "#f4f6f8",
        },
        "biome_colors": [
            "#b8c7d9",  # 0 苔原
            "#4f7a4a",  # 1 针叶林
            "#b9a35e",  # 2 温带草原
            "#6b8f4e",  # 3 温带落叶林
            "#2f5d33",  # 4 温带雨林
            "#cbb98a",  # 5 荒漠
            "#9cbf5a",  # 6 稀树草原
            "#2e7d32",  # 7 热带季节林
            "#14532d",  # 8 热带雨林
        ],
        "hillshade": {"azimuth": 315.0, "altitude": 45.0, "strength": 0.6},
        "labels": {"max_features": 40, "min_chars": 3, "font": "sans-serif"},
    }


def _ramp(t, colors):
    """按 t∈[0,1] 在颜色停靠点间做线性插值。"""
    t = np.clip(np.asarray(t, np.float32), 0.0, 1.0)
    n = len(colors) - 1
    pos = t * n
    i0 = np.clip(np.floor(pos).astype(np.int32), 0, n - 1)
    frac = (pos - i0)[..., None]
    c = np.asarray(colors, np.float32)
    return c[i0] * (1.0 - frac) + c[i0 + 1] * frac


def dem_to_rgb(dem_m):
    """把米制 DEM 快速着色成 ``(S,S,3)`` uint8 预览图（坡度光照 + 分层设色）。"""
    d = np.asarray(dem_m, np.float32)
    h, w = d.shape
    sea = d <= 0.0

    # ---- 分层设色 ----
    sea_ramp = [[0.04, 0.13, 0.28], [0.06, 0.22, 0.42],
                [0.12, 0.36, 0.58], [0.24, 0.55, 0.72]]
    land_ramp = [[0.62, 0.62, 0.42], [0.35, 0.52, 0.28], [0.27, 0.44, 0.24],
                 [0.45, 0.42, 0.30], [0.55, 0.50, 0.44], [0.86, 0.86, 0.86],
                 [1.00, 1.00, 1.00]]
    sea_t = np.clip(-d / 6000.0, 0.0, 1.0)
    land_t = np.clip(d / 5000.0, 0.0, 1.0)
    rgb = np.where(sea[..., None], _ramp(sea_t, sea_ramp), _ramp(land_t, land_ramp))

    # ---- 山体阴影（西北光, 315°/45°）----
    gy, gx = np.gradient(d)
    scale = float(np.percentile(np.hypot(gx, gy), 99)) or 1.0
    gx = gx / scale
    gy = gy / scale
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    nx, ny, nz = -gx, -gy, np.full_like(gx, 0.6)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    dot = (nx * np.cos(alt) * np.sin(az)
           + ny * np.cos(alt) * np.cos(az)
           + nz * np.sin(alt)) / np.maximum(norm, 1e-6)
    shade = np.clip(0.70 + 0.50 * dot, 0.0, 1.25)
    rgb = rgb * np.where(sea[..., None], 1.0, shade[..., None])
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
