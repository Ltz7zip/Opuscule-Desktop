# -*- coding: utf-8 -*-
"""
geo_context.py — 全局地理信息上下文 (WorldGeoContext Singleton)

=============================================================
用途：将 generate_world.py 输出的 DEM 数据封装为全局可访问的单例，
      供国家外交引擎、兵棋推演引擎等模块进行坐标级查询和区域统计。

依赖：numpy + scipy.interpolate.RegularGridInterpolator
兼容：从项目任意目录运行（基于 __file__ 解析 terrain/output）
=============================================================

快速使用：
    from terrain.geo_context import WorldGeoContext

    # 1) 单例初始化（首次运行加载，后续调用直接命中缓存）
    ctx = WorldGeoContext.from_seed(seed=1, size=2048)

    # 2) 单点查询（经纬度，双线性插值）
    elev = ctx.get_elevation(lat=31.23, lon=121.47)   # 上海
    on_land = ctx.is_land(lat=31.23, lon=121.47)
    slope_pct = ctx.get_slope(lat=31.23, lon=121.47)  # 坡度百分比

    # 3) 区域统计（多边形顶点 [(lat, lon), ...]，支持凹多边形）
    poly = [(45, 5), (55, 5), (55, 20), (45, 20)]    # 大致法国/德国区域
    stats = ctx.get_region_stats(poly)
    print(stats["mean_elevation"], stats["peak"], stats["land_ratio"])
"""

from __future__ import annotations

import os
import sys
import glob
import time
from typing import Tuple, Dict, Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 等距圆柱投影（Plate Carrée）下每纬度度数的米数（WGS84 近似）
_METERS_PER_DEG_LAT = 111320.0            # ~ 20003931 m / 180°
_METERS_PER_DEG_LON_AT_EQUATOR = 111320.0  # ~ 40075017 m / 360°

# 陆地阈值（米制高程 >= 该值视为陆地）
# generate_world.py 以 v01 >= 0.5 为陆地，v01=0.5 对应 elev≈0m，留 1m 余量消浮点噪声
_LAND_THRESHOLD_M = 1.0

# 高海拔阈值（山脉判定，外交引擎用）
_MOUNTAIN_THRESHOLD_M = 1500.0

# 防止自动触发 generate_world 死循环的环境变量
_GEN_LOCK_ENV = "_WORLDGEO_GENERATING_IN_PROGRESS"


# ===========================================================================
# TerrainDataProtocol — 数据协议（可替换为 Perlin 等其他生成器）
# ===========================================================================

@runtime_checkable
class TerrainDataProtocol(Protocol):
    """地形数据提供者协议。任何符合该协议的类都可以替代 WorldGeoContext，
    外交/兵棋模块只依赖接口，不耦合具体实现。"""

    @property
    def shape(self) -> Tuple[int, int]:
        """(rows, cols)，rows 对应纬度方向（0 行 ≈ 南极，末行 ≈ 北极）。"""
        ...

    @property
    def size(self) -> int:
        """网格单边像素数（S）。"""
        ...

    @property
    def lat_grid(self) -> np.ndarray:
        """纬度一维网格 (S,)，递增（-90° → +90°，像素中心对齐）。"""
        ...

    @property
    def lon_grid(self) -> np.ndarray:
        """经度一维网格 (S,)，递增（-180° → +180°，像素中心对齐）。"""
        ...

    def get_elevation(self, lat: float | np.ndarray,
                      lon: float | np.ndarray) -> float | np.ndarray:
        """返回指定经纬度的米制高程，支持向量输入。"""
        ...

    def is_land(self, lat: float | np.ndarray,
                lon: float | np.ndarray) -> bool | np.ndarray:
        """返回指定点是否为陆地（高程 >= LAND_THRESHOLD_M）。"""
        ...

    def get_slope(self, lat: float | np.ndarray,
                  lon: float | np.ndarray) -> float | np.ndarray:
        """返回指定点的坡度百分比（rise/run * 100）。"""
        ...

    def get_region_stats(self, polygon_coords: Sequence[Tuple[float, float]]
                         ) -> Dict[str, Any]:
        """区域统计：mean_elevation / peak / land_ratio / pixel_count / mountain_ratio。"""
        ...


# ===========================================================================
# WorldGeoContext — 全局地理信息单例
# ===========================================================================

class WorldGeoContext(TerrainDataProtocol):
    """
    全局地理信息单例。

    线程/进程备注：
      - 单例是**进程内**的；多进程场景下每个进程会各自持有一份副本（因为 DEM 通过 numpy
        共享对象 fork 时写时复制，通常够用；若超大规模可自行改为 mmap）。
      - 多线程下首次 from_seed() 会加简单锁，避免并发加载。
    """

    # ---- 单例机制 --------------------------------------------------------
    _instance: Optional["WorldGeoContext"] = None
    _instance_key: Optional[Tuple[int, int]] = None  # (seed, size)
    _init_lock = False  # 简化版加载锁（同进程内）

    # ---- 缓存数据 --------------------------------------------------------
    _dem: Optional[np.ndarray] = None          # (S, S) float32, 米制高程
    _land_mask: Optional[np.ndarray] = None    # (S, S) bool, 陆地掩膜
    _mountain_mask: Optional[np.ndarray] = None  # (S, S) bool
    _lat_grid: Optional[np.ndarray] = None     # (S,) 递增，像素中心
    _lon_grid: Optional[np.ndarray] = None     # (S,) 递增，像素中心
    _interp_elev: Optional[RegularGridInterpolator] = None
    _interp_slope: Optional[RegularGridInterpolator] = None

    # 路径
    _terrain_dir: str = ""
    _output_dir: str = ""

    # ---- 构造 ------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        """禁止直接 ``cls()`` 实例化；必须通过 ``WorldGeoContext.from_seed(seed, size)``。

        Raises
        ------
        TypeError
            只要调用就抛出，提示正确入口。"""
        raise TypeError("Use WorldGeoContext.from_seed(seed, size) instead.")

    @classmethod
    def from_seed(cls, seed: int = 0, size: int = 4096,
                  force_reload: bool = False) -> "WorldGeoContext":
        """
        获取或创建全局单例。

        Parameters
        ----------
        seed : int
            世界种子。用于优先查找 `seed{seed}_dem.npy` 或
            `world_dem_{seed}_{size}.npy`，找不到则回退到最新的
            `world_dem*.npy`，最后回退到自动生成。
        size : {2048, 4096}
            网格单边像素数。加载后若实际文件尺寸不匹配会打印警告但继续。
        force_reload : bool
            强制丢弃现有单例并重新加载（调试用）。
        """
        key = (seed, size)

        # 命中缓存直接返回
        if (not force_reload
                and cls._instance is not None
                and cls._instance_key == key):
            return cls._instance

        # 简化并发锁
        if cls._init_lock:
            while cls._init_lock:
                time.sleep(0.05)
            return cls.from_seed(seed, size, force_reload=False)

        cls._init_lock = True
        try:
            # -- 1. 定位 terrain / output 目录 --
            #    geo_context.py 已搬迁到 terrain/core/，向上一级得到 terrain/
            #    （output/ / checkpoints/ / data/ 仍保持 terrain/ 同级，与用户习惯一致）
            _here = os.path.dirname(os.path.abspath(__file__))
            terrain_dir = (os.path.dirname(_here)
                           if os.path.basename(_here) == "core" else _here)
            output_dir = os.path.join(terrain_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # -- 2. 解析 DEM 路径 --
            dem_path = _resolve_dem_path(output_dir, seed, size)

            # -- 3. 缺失时自动生成（防死循环） --
            if dem_path is None:
                dem_path = _generate_world_safe(seed, size, terrain_dir, output_dir)

            # -- 4. 加载 DEM --
            dem = np.load(dem_path).astype(np.float32)
            if dem.ndim != 2 or dem.shape[0] != dem.shape[1]:
                raise ValueError(
                    f"DEM {dem_path} 不是二维方阵，实际 shape={dem.shape}")
            actual_s = dem.shape[0]
            if actual_s != size:
                print(f"[WorldGeoContext] 警告: 请求 size={size} 但实际文件尺寸 "
                      f"为 {actual_s}，将以实际尺寸为准。", file=sys.stderr)
                size = actual_s

            # -- 5. 构造坐标网格（像素中心对齐，与 generate_world.py 保持一致） --
            lats = (-90.0 + (np.arange(size) + 0.5) * (180.0 / size)).astype(np.float32)
            lons = (-180.0 + (np.arange(size) + 0.5) * (360.0 / size)).astype(np.float32)

            # -- 6. 掩膜 --
            land_mask = dem >= _LAND_THRESHOLD_M
            mountain_mask = dem >= _MOUNTAIN_THRESHOLD_M

            # -- 7. 构建双线性高程插值器 --
            #    RegularGridInterpolator 默认 bounds_error=False，填 NaN；
            #    但经纬度超出范围时我们希望循环经度 + 夹取纬度。
            #    lat 必须递增以满足插值器（generate_world 的 lats 就是递增的 ✓）
            interp_elev = RegularGridInterpolator(
                (lats, lons), dem,
                method="linear",
                bounds_error=False,
                fill_value=None,  # 外推时夹取到边界值
            )

            # -- 8. 预计算坡度场并构建插值器 --
            slope_pct_field = _compute_slope_field(dem, lats, lons)
            interp_slope = RegularGridInterpolator(
                (lats, lons), slope_pct_field,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            # -- 9. 写入单例字段（用 Object.__new__ 绕过我们禁止的构造方法） --
            obj = object.__new__(cls)
            obj._dem = dem
            obj._land_mask = land_mask
            obj._mountain_mask = mountain_mask
            obj._lat_grid = lats
            obj._lon_grid = lons
            obj._interp_elev = interp_elev
            obj._interp_slope = interp_slope
            obj._terrain_dir = terrain_dir
            obj._output_dir = output_dir

            cls._instance = obj
            cls._instance_key = (seed, size)
        finally:
            cls._init_lock = False

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """调试用：销毁单例并释放内存。"""
        cls._instance = None
        cls._instance_key = None

    # ======================================================================
    # Protocol 属性实现
    # ======================================================================

    @property
    def shape(self) -> Tuple[int, int]:
        """返回 DEM 二维方阵形状 ``(rows, cols) = (S, S)``。"""
        return self._dem.shape  # type: ignore[union-attr]

    @property
    def size(self) -> int:
        """DEM 单边像素数 ``S``（shape[0]）。"""
        return self._dem.shape[0]  # type: ignore[union-attr]

    @property
    def lat_grid(self) -> np.ndarray:
        """纬度一维网格 ``(S,)``，递增 / 像素中心对齐（-90°+ε → +90°−ε）。"""
        return self._lat_grid  # type: ignore[return-value]

    @property
    def lon_grid(self) -> np.ndarray:
        """经度一维网格 ``(S,)``，递增 / 像素中心对齐（-180°+ε → +180°−ε）。"""
        return self._lon_grid  # type: ignore[return-value]

    @property
    def dem(self) -> np.ndarray:
        """只读访问原始 DEM（不建议修改；若必须修改请先 .copy()）。"""
        return self._dem  # type: ignore[return-value]

    @property
    def land_mask(self) -> np.ndarray:
        """陆地掩膜 ``(S, S) bool`` —— 高程 ≥ ``_LAND_THRESHOLD_M`` (0 m) 的像素为 True。"""
        return self._land_mask  # type: ignore[return-value]

    @property
    def mountain_mask(self) -> np.ndarray:
        """山地掩膜 ``(S, S) bool`` —— 高程 ≥ ``_MOUNTAIN_THRESHOLD_M`` (1500 m) 的像素为 True。"""
        return self._mountain_mask  # type: ignore[return-value]

    # ======================================================================
    # 单点查询接口
    # ======================================================================

    def _normalize_coords(self, lat, lon):
        """将经纬度规范化到插值器接受的范围。
        纬度：夹取到 [-89.999, 89.999]（避免极点刚好越界的 corner case）
        经度：-180~180 循环（wrap），保证 (179.9, 180.1) 相邻。
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.clip(lat, -89.999, 89.999)
        # 经度 wrap 到 [-180, 180)
        lon = ((lon + 180.0) % 360.0) - 180.0
        # 修正正好 = +180 的点（模运算后可能落到 +180）
        lon = np.where(lon >= 180.0, lon - 360.0, lon)
        # 但网格最右端是 -180 + (S-0.5)*360/S < 180，所以 +180 会外推；
        # 把它挪到 -179.999（与 -180 左边界对齐也可以，插值器 fill_value=None 已夹取）
        return lat, lon

    def get_elevation(self, lat, lon):
        """返回米制高程（双线性插值）。

        Parameters
        ----------
        lat, lon : float 或 ndarray
            地理坐标。支持标量 / 任意形状数组（形状相同即可），内部自动广播。
            经度自动 wrap 到 [-180, +180]，纬度 clip 到 ±89.999°。

        Returns
        -------
        float 或 ndarray of float32
            标量输入返回 Python float，数组输入返回同形状 ndarray。
            单位：米（正=陆地 / 负=海洋）。"""
        lat, lon = self._normalize_coords(lat, lon)
        # RegularGridInterpolator 要求最后一维为 (lat, lon) 对
        scalar_in = lat.ndim == 0 and lon.ndim == 0
        pts = np.stack([np.atleast_1d(lat), np.atleast_1d(lon)], axis=-1)
        res = self._interp_elev(pts).astype(np.float32)  # type: ignore[misc]
        return float(res[0]) if scalar_in else res

    def is_land(self, lat, lon):
        """判断 ``(lat, lon)`` 是否为陆地（高程 ≥ ``_LAND_THRESHOLD_M`` = 0 m）。

        返回形状与输入一致：标量 → bool，数组 → bool ndarray。"""
        elev = self.get_elevation(lat, lon)
        return elev >= _LAND_THRESHOLD_M

    def is_mountain(self, lat, lon):
        """判断 ``(lat, lon)`` 是否属于山地（高程 ≥ 1500 m）。

        返回形状与输入一致：标量 → bool，数组 → bool ndarray。"""
        elev = self.get_elevation(lat, lon)
        return elev >= _MOUNTAIN_THRESHOLD_M

    def get_slope(self, lat, lon):
        """返回坡度（百分比：rise/run × 100，双线性插值）。

        0 = 绝对平地；100 ≈ 45°；数值越大越陡（最大值预 clamp 到 1000）。
        已做 cos(lat) 球面经度校正 → 赤道和高纬度的坡度口径一致。
        标量/数组输入返回同形状。"""
        lat, lon = self._normalize_coords(lat, lon)
        scalar_in = lat.ndim == 0 and lon.ndim == 0
        pts = np.stack([np.atleast_1d(lat), np.atleast_1d(lon)], axis=-1)
        res = self._interp_slope(pts).astype(np.float32)  # type: ignore[misc]
        return float(res[0]) if scalar_in else res

    # ======================================================================
    # 网格化便捷接口（给兵棋用：查询网格坐标而非经纬度）
    # ======================================================================

    def idx_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """网格 (row, col) → 该像素中心的 (lat, lon)。"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(f"(row={row}, col={col}) 超出网格 {self.size}x{self.size}")
        return float(self._lat_grid[row]), float(self._lon_grid[col])  # type: ignore[index]

    def latlon_to_idx(self, lat: float, lon: float) -> Tuple[int, int]:
        """(lat, lon) → 最近的网格 (row, col)。

        由于坐标网格是严格等距的，直接反解像素中心公式即可，与 `idx_to_latlon` 严格互逆。
        """
        lat, lon = self._normalize_coords(lat, lon)
        S = self.size
        dlat = 180.0 / S
        dlon = 360.0 / S
        # lat_grid[r] = -90 + (r + 0.5) * dlat  →  r = (lat + 90)/dlat - 0.5
        r = int(round(float(lat + 90.0) / dlat - 0.5))
        c = int(round(float(lon + 180.0) / dlon - 0.5))
        # 修正边界（-180° 与 +180° 同源，但网格是 -180+ε 到 180-ε 中心对齐）
        r = min(max(r, 0), S - 1)
        c = min(max(c, 0), S - 1)
        return r, c

    def get_elevation_at(self, row: int, col: int) -> float:
        """直接获取 DEM[row, col] 的米制高程（跳过插值 / O(1) 取预加载数组）。

        若 row/col 越界会抛 ``IndexError``；0 ≤ row, col < size。"""
        return float(self._dem[row, col])  # type: ignore[index]

    def is_land_at(self, row: int, col: int) -> bool:
        """直接从陆地掩膜返回 ``land_mask[row, col]``（O(1)，无插值）。"""
        return bool(self._land_mask[row, col])  # type: ignore[index]

    def get_slope_at(self, row: int, col: int) -> float:
        """查询网格点的坡度（避免插值，直接用预计算场）。"""
        # 坡度场在内存中，直接取；若用户传入越界索引这里会自然报错
        slope_field = getattr(self, "_slope_field", None)
        if slope_field is None:
            # 懒初始化：从插值器权重重建不现实，这里重新计算并缓存
            self._slope_field = _compute_slope_field(
                self._dem, self._lat_grid, self._lon_grid)
            slope_field = self._slope_field
        return float(slope_field[row, col])

    # ======================================================================
    # 区域统计接口
    # ======================================================================

    def get_region_stats(self, polygon_coords: Sequence[Tuple[float, float]]
                         ) -> Dict[str, Any]:
        """
        返回多边形区域内的统计量。

        Parameters
        ----------
        polygon_coords : list of (lat, lon) tuple
            多边形顶点（最后一个点不需要与第一个点重复；自动闭合）。
            支持凹多边形（使用射线法 PIP 测试）。

        Returns
        -------
        dict with keys:
            mean_elevation : float (m)
            peak            : float (m), 区域内最高点
            trough          : float (m), 区域内最低点
            land_ratio      : float 0~1
            mountain_ratio  : float 0~1
            pixel_count     : int, 落入多边形的网格像素数
            land_pixel_count: int
        """
        poly_lat = np.asarray([p[0] for p in polygon_coords], dtype=np.float64)
        poly_lon = np.asarray([p[1] for p in polygon_coords], dtype=np.float64)
        if poly_lat.size < 3:
            raise ValueError("多边形至少需要 3 个顶点")

        # -- 1. 将多边形顶点映射到网格 (row, col) 坐标 --
        #    注意：此处不做投影变换（Plate Carrée 网格 x 经度线性）
        S = self.size

        def _ll_to_rc(lat_arr, lon_arr):
            """把纬度/经度数组映射为网格 (row, col) 浮点坐标。"""
            # row = (lat - lat_grid[0]) / dlat 整数化
            dlat = 180.0 / S
            dlon = 360.0 / S
            # lat_grid[0] = -90 + 0.5*dlat  → row = (lat - (-90 + 0.5*dlat))/dlat + 0.5?
            # 实际上 lat_grid[r] = -90 + (r+0.5)*dlat
            # => r = (lat + 90)/dlat - 0.5
            r = (lat_arr + 90.0) / dlat - 0.5
            c = (lon_arr + 180.0) / dlon - 0.5
            return r, c

        poly_r, poly_c = _ll_to_rc(poly_lat, poly_lon)

        # -- 2. 取多边形 bounding box，限定检测范围 --
        r_min = int(max(0, np.floor(poly_r.min())))
        r_max = int(min(S - 1, np.ceil(poly_r.max())))
        c_min = int(max(0, np.floor(poly_c.min())))
        c_max = int(min(S - 1, np.ceil(poly_c.max())))
        if r_min >= r_max or c_min >= c_max:
            return dict(mean_elevation=0.0, peak=0.0, trough=0.0,
                        land_ratio=0.0, mountain_ratio=0.0,
                        pixel_count=0, land_pixel_count=0)

        # -- 3. 对 bbox 内所有像素中心做 PIP 判定 --
        rr = np.arange(r_min, r_max + 1, dtype=np.float64)
        cc = np.arange(c_min, c_max + 1, dtype=np.float64)
        CR, CC = np.meshgrid(rr, cc, indexing="ij")  # (H, W)
        inside = _point_in_polygon_2d(CC.ravel(), CR.ravel(), poly_c, poly_r)
        inside = inside.reshape(CR.shape)  # (H, W)

        # -- 4. 提取 mask 内的数据 --
        dem_bbox = self._dem[r_min:r_max + 1, c_min:c_max + 1]  # type: ignore[index]
        land_bbox = self._land_mask[r_min:r_max + 1, c_min:c_max + 1]  # type: ignore[index]
        mont_bbox = self._mountain_mask[r_min:r_max + 1, c_min:c_max + 1]  # type: ignore[index]

        total_px = int(inside.sum())
        if total_px == 0:
            return dict(mean_elevation=0.0, peak=0.0, trough=0.0,
                        land_ratio=0.0, mountain_ratio=0.0,
                        pixel_count=0, land_pixel_count=0)

        in_dem = dem_bbox[inside]
        in_land = land_bbox[inside]
        in_mont = mont_bbox[inside]

        land_px = int(in_land.sum())
        mont_px = int(in_mont.sum())

        return dict(
            mean_elevation=float(in_dem.mean()),
            peak=float(in_dem.max()),
            trough=float(in_dem.min()),
            land_ratio=float(land_px / total_px),
            mountain_ratio=float(mont_px / total_px),
            pixel_count=total_px,
            land_pixel_count=land_px,
        )


# ===========================================================================
# 内部工具函数
# ===========================================================================

def _resolve_dem_path(output_dir: str, seed: int, size: int) -> Optional[str]:
    """
    按优先级寻找 DEM 文件：
      1. world_dem_{seed}_{size}.npy   — 用户显式命名
      2. seed{seed}_dem.npy             — 项目中已存在的命名（seed1_dem.npy ...）
      3. world_dem.npy                  — 默认 fallback
      4. 找 output/ 下最新的 world_dem*.npy / seed*_dem.npy
    找不到返回 None。
    """
    # 1) 精确匹配 seed + size
    p1 = os.path.join(output_dir, f"world_dem_{seed}_{size}.npy")
    if os.path.exists(p1):
        return p1

    # 2) 项目现存命名 seed{seed}_dem.npy
    p2 = os.path.join(output_dir, f"seed{seed}_dem.npy")
    if os.path.exists(p2):
        return p2

    # 3) 默认 world_dem.npy
    p3 = os.path.join(output_dir, "world_dem.npy")
    if os.path.exists(p3):
        return p3

    # 4) 找最新修改的 world_dem*.npy 或 seed*_dem.npy
    candidates = (
        glob.glob(os.path.join(output_dir, "world_dem*.npy"))
        + glob.glob(os.path.join(output_dir, "seed*_dem.npy"))
    )
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    return None


def _generate_world_safe(seed: int, size: int,
                         terrain_dir: str, output_dir: str) -> str:
    """
    当 output/ 中没有任何 DEM 时，触发 terrain.generate_world.generate_world()
    生成一个。加"防死循环"锁：
      - 环境变量 _GEN_LOCK_ENV：若已设置则说明当前调用栈正处于生成过程中
        （可能 generate_world 内部 import 了 geo_context），立刻抛错避免递归。
      - 检查 sys.argv：若当前进程本身就是在跑 generate_world.py（第一参数
        以 generate_world 结尾），也抛错。
    """
    if os.environ.get(_GEN_LOCK_ENV, "") == "1":
        raise RuntimeError(
            "检测到 generate_world 正在执行中（环境变量锁已占用），"
            "为避免死循环，geo_context 停止自动生成 DEM。"
            "请先手动运行 `python -m terrain.core.generate_world --seed {seed} --size {size}`。"
            .format(seed=seed, size=size))

    argv0 = os.path.basename(sys.argv[0]) if sys.argv else ""
    if "generate_world" in argv0:
        raise RuntimeError(
            f"当前进程 ({argv0}) 即 generate_world 脚本本身，为防死循环，"
            f"geo_context 拒绝再次调用生成。请先产出 DEM 再 import geo_context。")

    print(f"[WorldGeoContext] output/ 中未找到 seed={seed} size={size} 的 DEM，"
          f"正在调用 generate_world.generate_world(seed={seed}, size_px={size}) ...",
          file=sys.stderr)

    os.environ[_GEN_LOCK_ENV] = "1"
    try:
        # 延迟 import：避免 geo_context 被 generate_world import 时的循环导入
        from .generate_world import generate_world  # type: ignore
        # 以固定输出名生成；生成后重命名为 seed{seed}_dem.npy 便于下次快速命中
        default_npy = os.path.join(output_dir, "world_dem.npy")
        default_png = os.path.join(output_dir, "world_map.png")
        dem = generate_world(seed=seed, size_px=size,
                             out_png=default_png, out_npy=default_npy)
        # generate_world 已写入 default_npy；这里再存一份命名版
        target_npy = os.path.join(output_dir, f"seed{seed}_dem.npy")
        try:
            np.save(target_npy, dem.astype(np.float32))
        except OSError:
            pass  # 写入失败没关系，default_npy 还在
        return default_npy if os.path.exists(default_npy) else target_npy
    finally:
        os.environ.pop(_GEN_LOCK_ENV, None)


def _compute_slope_field(dem: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    基于 DEM 计算逐像素坡度百分比（rise/run × 100）。

    等距圆柱投影下：
        y 方向（纬度）每像素距离 = METERS_PER_DEG_LAT × dlat  （近似常数）
        x 方向（经度）每像素距离 = METERS_PER_DEG_LON_EQ × dlon × cos(lat)
    用中心差分计算 d(elev)/dy 与 d(elev)/dx，合并后取百分比。
    """
    S = dem.shape[0]
    dlat = 180.0 / S
    dlon = 360.0 / S
    dy = _METERS_PER_DEG_LAT * dlat  # 常数

    # np.gradient 返回 (axis0_grad, axis1_grad)
    # axis0 = 纬度方向（rows）；axis1 = 经度方向（cols）
    g_row, g_col = np.gradient(dem.astype(np.float64))

    # g_row 单位是 m/像素 → 除以 dy 得到 m/m（坡度 = 上升/水平距离）
    slope_row = g_row / dy

    # g_col 需要按纬度校正经度方向每像素的真实米数
    cos_lat = np.cos(np.deg2rad(lats)).astype(np.float64)  # (S,)
    dx_per_row = (_METERS_PER_DEG_LON_AT_EQUATOR * dlon) * cos_lat  # (S,)
    # 广播到 (S, S)：列方向每一行都有不同 dx
    slope_col = g_col / dx_per_row[:, None]  # 安全：赤道处 cos=1，极点 cos≈0 → 无限；但 lat 最多到 ±89.956°

    # 合并：坡度百分比 = sqrt( (dy/d)^2 + (dx/d)^2 ) * 100
    slope_pct = np.sqrt(slope_row ** 2 + slope_col ** 2) * 100.0

    # 极点附近 clamp（cos(lat)≈0 时数值爆炸）：坡度百分比通常不可能超过 1000%
    slope_pct = np.clip(slope_pct, 0.0, 1000.0)
    return slope_pct.astype(np.float32)


def _point_in_polygon_2d(x: np.ndarray, y: np.ndarray,
                         poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """
    向量化射线法（Ray Casting）点在多边形内测试。

    Parameters
    ----------
    x, y : 1D arrays of shape (N,)  —— 待测试点
    poly_x, poly_y : 1D arrays of shape (M,)  —— 多边形顶点（自动闭合）

    Returns
    -------
    1D bool array (N,) ：True = 点在多边形内或边上
    """
    N = x.size
    M = poly_x.size
    inside = np.zeros(N, dtype=bool)
    # 闭合多边形
    px = np.concatenate([poly_x, [poly_x[0]]])
    py = np.concatenate([poly_y, [poly_y[0]]])

    for i in range(M):
        j = i + 1
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        # 边 (i,j) 是否与测试点的水平射线 (y=y0, x→+∞) 相交
        # 条件 1: y 在边的 y 范围之间
        cond_y = ((yi > y) != (yj > y))
        if not cond_y.any():
            continue
        # 条件 2: 射线交点的 x 坐标 < 测试点的 x
        # 求 x_intersect = x 在边与水平线 y=y0 上的交点 x 值
        denom = (yj - yi)
        with np.errstate(divide="ignore", invalid="ignore"):
            # 对 cond_y 的点才计算
            x_intersect = (xj - xi) * (y[cond_y] - yi) / np.where(denom == 0, 1e-18, denom) + xi
        hit = cond_y.copy()
        hit[cond_y] &= (x[cond_y] < x_intersect)
        inside ^= hit
    return inside


# ===========================================================================
# 模块级便捷函数（省去每次 .from_seed()）
# ===========================================================================

def get_ctx(seed: int = 0, size: int = 4096) -> WorldGeoContext:
    """返回全局单例的简写。"""
    return WorldGeoContext.from_seed(seed=seed, size=size)


# ===========================================================================
# __main__ 自测
# ===========================================================================

def main():
    """``geo_context.py`` 命令行入口 —— 单例加载 + 自测。

    解析 ``--seed / --size`` 实例化 :class:`WorldGeoContext`，输出
    内存 / 陆地表面积指标，并执行单点查询、多边形区域统计、
    网格<->经纬度往返一致性自检。
    ``python -m terrain.core.geo_context`` 以模块方式运行时，经由此入口进入。
    """
    # 运行：python -m terrain.core.geo_context  --seed 1 --size 2048
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=2048)
    a = ap.parse_args()

    t0 = time.time()
    ctx = WorldGeoContext.from_seed(seed=a.seed, size=a.size)
    print(f"[自测] DEM 加载耗时 {time.time()-t0:.2f}s | shape={ctx.shape}")
    print(f"       内存占用 ≈ {ctx.dem.nbytes/1024/1024:.1f} MB")
    print(f"       陆地占比 {(ctx.land_mask.mean()*100):.2f}%")
    print(f"       山脉占比 {(ctx.mountain_mask.mean()*100):.2f}%")

    # 单点查询：几个城市附近（在随机世界里不对应真实城市，但用于测试坐标映射）
    tests = [
        ("赤道几内亚湾(海)", 0.0, 5.0),
        ("北非(陆)", 30.0, 5.0),
        ("喜马拉雅(高海拔)", 30.0, 85.0),
        ("南太平洋(海)", -20.0, -140.0),
        ("北极点(边界)", 89.9, 0.0),
        ("日期变更线附近", 0.0, 179.9),
        ("日期变更线附近(-)", 0.0, -179.9),
    ]
    print("\n[自测] 单点查询:")
    for name, lat, lon in tests:
        e = ctx.get_elevation(lat, lon)
        l = ctx.is_land(lat, lon)
        s = ctx.get_slope(lat, lon)
        print(f"  {name:22s} lat={lat:6.1f} lon={lon:7.1f}  "
              f"elev={e:8.0f}m  land={l}  slope={s:6.2f}%")

    # 区域统计
    print("\n[自测] 区域统计 (一个 10°x10° 的方块):")
    poly = [(10, 10), (20, 10), (20, 20), (10, 20)]
    st = ctx.get_region_stats(poly)
    for k, v in st.items():
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.3f}")
        else:
            print(f"  {k:20s} = {v}")

    # 网格往返一致性测试
    r0, c0 = ctx.size // 3, ctx.size // 4
    lat0, lon0 = ctx.idx_to_latlon(r0, c0)
    r1, c1 = ctx.latlon_to_idx(lat0, lon0)
    print(f"\n[自测] 网格<->经纬度往返: ({r0},{c0})→"
          f"({lat0:.3f},{lon0:.3f})→({r1},{c1})  "
          f"一致={'YES' if (r0, c0) == (r1, c1) else 'NO'}")


if __name__ == "__main__":
    main()
