# -*- coding: utf-8 -*-
"""SupervisorMetrics — 从已生成 DEM 计算地形/温度/降水/群系/风带五维指标.

幂等性：从 dem_m + lats + seed 重算气候场，与 render_world（generate_world.py
L200-216）完全同口径、同 RNG 种子，结果 bitwise 一致。

复用关系（避免重复造轮子）：
    - continentality_index / ocean_current_effects / temperature_field /
      precipitation_field / whittaker_biome / smooth_biome_labels / m2v /
      snow_area_by_band / _psd_slope / _fractal_dim / _slope_stats / _coherence
      均从 terrain_engine 扁平导入（terrain/__init__.py 已注入 core/ 到 sys.path）。
    - compute_global_stats 沿用 geo_context.get_region_stats 的 dict 键口径
      （land_ratio/peak/trough/mountain_ratio/mean_elevation）。
    - _band_mean 复用 snow_area_by_band 的 |lats|[:, None] 掩膜思路。

大尺寸下采样（用户已确认）：
    size_px > 2048 时，把 dem_m 用 scipy.ndimage.zoom 下采样到 2048² 再算气候场，
    省 30-60s。地形维度统计仍用原尺寸 dem_m（不损失极值精度）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as ndi_zoom

# 复用 terrain_engine 已有函数 (通过 terrain.core 包导入, 静态可解析)
from terrain.core.terrain_engine import (
    temperature_field, precipitation_field, whittaker_biome,
    continentality_index, ocean_current_effects, smooth_biome_labels,
    snow_area_by_band, m2v,
    BIOME_NAMES,
    _psd_slope, _fractal_dim, _slope_stats, _coherence,
)
from .references import (
    TEMP_LAT, TEMP_C, PRECIP_LAT, PRECIP_MM,
)

# 山地阈值 (与 geo_context._MOUNTAIN_THRESHOLD_M 一致, L55)
_MOUNTAIN_THRESHOLD_M = 1500.0

# 纬度带定义 (与温度/降水参考曲线对齐, 覆盖 0-90° 五带)
LAT_BANDS: List[Tuple[float, float]] = [
    (0, 10),      # 赤道带
    (10, 30),     # 副热带
    (30, 50),     # 温带
    (50, 66.5),   # 亚北极
    (66.5, 90),   # 极地
]
BAND_NAMES: List[str] = ["0-10", "10-30", "30-50", "50-66.5", "66.5-90"]

# 大尺寸审计下采样阈值 (用户已确认: 4096² → 2048²)
_DOWNSAMPLE_THRESHOLD = 2048


@dataclass
class SupervisorMetrics:
    """五维指标容器 (全部 float/dict/list, 便于 JSON 序列化).

    字段分组:
        地形维度: land_ratio / peak_elevation / ocean_trench /
                  mean_land_elevation / mountain_ratio
        温度维度: temp_by_band / temp_ref_by_band (按 5 纬度带)
        降水维度: precip_by_band / precip_ref_by_band (按 5 纬度带)
        群系维度: biome_ratio (9 类面积占比) / biome_ocean_ratio
        风带/雪盖: snow_area_by_band (4 纬度带, 复用 terrain_engine)
        地形纹理: psd_slope / fractal_dim / slope_mean / slope_p95 / coherence
        元数据: seed / size_px / downsampled (是否下采样)
    """
    # 地形维度
    land_ratio: float
    peak_elevation: float
    ocean_trench: float
    mean_land_elevation: float
    mountain_ratio: float
    # 温度维度 (按 5 纬度带)
    temp_by_band: Dict[str, float]
    temp_ref_by_band: Dict[str, float]
    # 降水维度 (按 5 纬度带)
    precip_by_band: Dict[str, float]
    precip_ref_by_band: Dict[str, float]
    # 群系维度 (9 类面积占比, 对齐 BIOME_NAMES)
    biome_ratio: Dict[str, float]
    biome_ocean_ratio: float
    # 风带/雪盖维度 (复用 snow_area_by_band, 4 纬度带)
    snow_area_by_band: List[float]
    # 复用 terrain_metrics 底层指标 (PSD/分形维/坡度/相干性)
    psd_slope: float
    fractal_dim: float
    slope_mean: float
    slope_p95: float
    coherence: float
    # 元数据
    seed: int
    size_px: int
    downsampled: bool = False


def compute_global_stats(dem_m: np.ndarray) -> Dict[str, float]:
    """全局统计 (沿用 geo_context.get_region_stats 的 dict 键口径, 全图掩膜).

    对齐 geo_context.get_region_stats 返回的键:
        mean_elevation / peak / trough / land_ratio / mountain_ratio /
        pixel_count / land_pixel_count / mean_land_elevation

    Parameters
    ----------
    dem_m : ndarray (S, S) float
        米制高程 (正值陆地 / 负值海洋)。

    Returns
    -------
    dict
        含 mean_elevation / peak / trough / land_ratio / mountain_ratio /
        pixel_count / land_pixel_count / mean_land_elevation。
    """
    land = dem_m > 0
    mountain = dem_m >= _MOUNTAIN_THRESHOLD_M
    total = int(dem_m.size)
    land_px = int(land.sum())
    return dict(
        mean_elevation=float(dem_m[land].mean()) if land_px else 0.0,
        peak=float(dem_m.max()),
        trough=float(dem_m.min()),
        land_ratio=float(land_px / total) if total else 0.0,
        mountain_ratio=float(int(mountain.sum()) / total) if total else 0.0,
        pixel_count=total,
        land_pixel_count=land_px,
        mean_land_elevation=float(dem_m[land].mean()) if land_px else 0.0,
    )


def _band_mean(field_arr: np.ndarray, lats: np.ndarray,
               band: Tuple[float, float]) -> float:
    """按纬度带 (|lat|) 计算场均值.

    复用 snow_area_by_band 的 |lats|[:, None] 掩膜思路:

        a = |lats|[:, None]  # (S, 1) 广播到 (S, W)
        mask = (a >= lo) & (a < hi)  # (S, W) 布尔掩膜
        return float(field[mask].mean())

    Parameters
    ----------
    field_arr : ndarray (S, W)
        2D 场 (温度/降水等)。
    lats : ndarray (S,)
        纬度数组 (像素中心对齐)。
    band : (lo, hi)
        纬度带边界 (绝对值, 闭区间 [lo, hi))。

    Returns
    -------
    float
        带内场均值。空带返回 nan。
    """
    lo, hi = band
    # 复用 snow_area_by_band (terrain_engine.py L2769) 的 repeat 广播思路:
    # |lats|[:, None] 是 (S, 1), .repeat(W, 1) 变 (S, W) 才能 mask (S, W) 场
    field_arr = np.asarray(field_arr)
    a = np.abs(np.asarray(lats, np.float64))[:, None].repeat(
        field_arr.shape[1], 1)
    mask = (a >= lo) & (a < hi)
    if not mask.any():
        return float('nan')
    return float(field_arr[mask].mean())


def _interp_ref(lat_band: Tuple[float, float],
                ref_lat: np.ndarray, ref_val: np.ndarray) -> float:
    """参考曲线在纬度带中点的线性插值.

    仅用于温度带 (带内温差小, 中点足够); 降水带请用 ``_area_weighted_ref``。
    """
    mid = 0.5 * (lat_band[0] + min(lat_band[1], 89.99))
    return float(np.interp(mid, ref_lat, ref_val))


def _area_weighted_ref(lat_band: Tuple[float, float],
                       ref_lat: np.ndarray, ref_val: np.ndarray,
                       step: float = 0.05) -> float:
    """参考曲线在纬度带内的**面积加权**均值 (cos(lat) 权重).

    实测 ``_band_mean`` 是带内像素的等面积平均 (像素数 ∝ cos lat), 而
    ``_interp_ref`` 只取带中点。高纬带两者差得最远 (66.5-90° 中点 78.25°
    比面积加权均值高 10°+), 于是把参考曲线尾巴的干旱段放大, 造成假偏差
    (v43: 极地带参考 322mm vs 面积加权 439mm)。
    """
    lo, hi = lat_band
    hi = min(hi, 89.999)
    if hi <= lo:
        return float('nan')
    n = max(int(round((hi - lo) / step)), 2)
    lat = np.linspace(lo, hi, n, endpoint=False) + 0.5 * (hi - lo) / n
    w = np.cos(np.deg2rad(lat))
    v = np.interp(lat, ref_lat, ref_val)
    return float((v * w).sum() / w.sum())


def _downsample_dem(dem_m: np.ndarray, target: int = _DOWNSAMPLE_THRESHOLD
                    ) -> np.ndarray:
    """下采样 DEM 到 target² (双线性插值, 保留低频形态).

    Parameters
    ----------
    dem_m : ndarray (S, S)
        原始米制高程。
    target : int
        目标单边像素数 (默认 2048)。

    Returns
    -------
    ndarray (target, target)
        下采样后的高程。
    """
    S = dem_m.shape[0]
    if S <= target:
        return dem_m
    factor = target / S
    return ndi_zoom(dem_m.astype(np.float64), factor, order=1)


def _downsample_lats(lats: np.ndarray, target: int = _DOWNSAMPLE_THRESHOLD
                     ) -> np.ndarray:
    """下采样纬度数组到 target 长度 (线性插值, 保留纬度范围).

    Parameters
    ----------
    lats : ndarray (S,)
        原始纬度数组。
    target : int
        目标长度 (默认 2048)。

    Returns
    -------
    ndarray (target,)
        下采样后的纬度。
    """
    S = lats.shape[0]
    if S <= target:
        return lats
    old = np.linspace(0, 1, S)
    new = np.linspace(0, 1, target)
    return np.interp(new, old, lats.astype(np.float64))


def compute_metrics(dem_m: np.ndarray,
                    lats: np.ndarray,
                    seed: int,
                    size_px: int,
                    tect_ctx: Optional[dict] = None) -> SupervisorMetrics:
    """主入口: 从 dem_m 计算五维指标.

    复用 render_world 的气候因果链 (generate_world.py L200-216), 幂等重算:
        1. ci = continentality_index(elev_m, lats)
        2. dT, dP = ocean_current_effects(elev_m, lats)
        3. T = temperature_field(lats, elev_m, dT, ci=ci)
        4. P = precipitation_field(lats, elev_m, rng, dP, ci=ci)
        5. T_s = gaussian_filter(T, 2.5)   # 与 render_world L210 一致
        6. P_s = gaussian_filter(P, 1.5)   # 与 render_world L211 一致
        7. bi = whittaker_biome(T_s, P_s, lats=lats, ci=ci)
        8. bi = smooth_biome_labels(bi, land, median_size=7,
                                     min_patch_px=50, iters=3)  # 与 L216 一致

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        最终米制高程 (generate_world 输出, 正值陆地 / 负值海洋)。
    lats : ndarray (S,) float32
        纬度数组 (像素中心对齐, 递增)。
    seed : int
        世界种子 (供 precipitation_field 的 RNG 重建, 与 render_world L785 同口径)。
    size_px : int
        DEM 单边像素数。
    tect_ctx : dict, optional
        构造场上下文 (与 render_world 同口径; None 时退化, 不影响气候计算)。

    Returns
    -------
    SupervisorMetrics
        五维指标容器。
    """
    # ─── 0. 大尺寸下采样 (用户已确认: 4096² → 2048²) ───
    downsampled = size_px > _DOWNSAMPLE_THRESHOLD
    dem_for_climate = dem_m
    lats_for_climate = lats
    if downsampled:
        dem_for_climate = _downsample_dem(dem_m, _DOWNSAMPLE_THRESHOLD)
        lats_for_climate = _downsample_lats(lats, _DOWNSAMPLE_THRESHOLD)

    # ─── 1. 地形维度 (用原尺寸 dem_m, 不损失极值精度) ───
    gs = compute_global_stats(dem_m)

    # ─── 2. 重算气候场 (与 render_world 同口径, 幂等) ───
    elev_m = dem_for_climate.astype(np.float64)
    land_climate = elev_m > 0
    rng = np.random.RandomState(seed)  # 与 render_world L785 一致

    ci = continentality_index(elev_m, lats_for_climate)
    dT, dP = ocean_current_effects(elev_m, lats_for_climate)
    T = temperature_field(lats_for_climate, elev_m, dT, ci=ci)
    P = precipitation_field(lats_for_climate, elev_m, rng=rng, dP=dP, ci=ci)
    # 与 render_world L210-211 一致的高斯平滑参数
    T_s = gaussian_filter(T.astype(np.float32), 2.5)
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    bi = whittaker_biome(T_s, P_s, lats=lats_for_climate, ci=ci)
    # 与 render_world L216 一致的群系平滑参数
    bi = smooth_biome_labels(bi, land_climate, median_size=9,
                            min_patch_px=90, iters=3)

    # ─── 3. 温度维度 (按 5 纬度带) ───
    temp_by_band = {n: _band_mean(T_s, lats_for_climate, b)
                    for n, b in zip(BAND_NAMES, LAT_BANDS)}
    temp_ref_by_band = {n: _interp_ref(b, TEMP_LAT, TEMP_C)
                        for n, b in zip(BAND_NAMES, LAT_BANDS)}

    # ─── 4. 降水维度 ───
    precip_by_band = {n: _band_mean(P_s, lats_for_climate, b)
                      for n, b in zip(BAND_NAMES, LAT_BANDS)}
    precip_ref_by_band = {n: _area_weighted_ref(b, PRECIP_LAT, PRECIP_MM)
                          for n, b in zip(BAND_NAMES, LAT_BANDS)}

    # ─── 5. 群系维度 (9 类面积占比, 对齐 BIOME_NAMES) ───
    # whittaker_biome 对海洋像素也输出标签, 故群系统计必须限制在陆地像素上.
    # v40: 分母改为**陆地像素数** —— references.BIOME_RATIO 是地球"占陆地"口径,
    #      原实现除以全图像素, 相当于把每个群系都乘上陆地占比 (~0.30), 于是所有
    #      群系一律报"偏低 70%"(荒漠 25%→7.5%, 苔原 10%→3.0%), 是纯粹的量纲错误.
    #      海洋占比仍按全图口径单独给出 (biome_ocean_ratio).
    bi_land = bi[land_climate]  # 只取陆地像素的群系标签 (1D)
    total_land = int(bi_land.size)
    biome_ratio = {}
    for i, name in enumerate(BIOME_NAMES):
        biome_ratio[str(name)] = (
            float(int((bi_land == i).sum()) / total_land)
            if total_land else 0.0
        )
    biome_ocean_ratio = float(int((~land_climate).sum()) / int(bi.size)
                              ) if bi.size else 0.0

    # ─── 6. 风带/雪盖维度 (复用 snow_area_by_band, 用下采样后的 dem) ───
    v01_climate = m2v(dem_for_climate)
    snow_bands = snow_area_by_band(v01_climate, lats_for_climate)
    # snow_area_by_band 返回 list[float], 4 元素

    # ─── 7. 复用 terrain_metrics 底层 (PSD/分形维/坡度/相干性, 用下采样后 v01) ───
    sm, sp = _slope_stats(v01_climate)  # 返回 (mean*100, p95*100)
    return SupervisorMetrics(
        land_ratio=gs["land_ratio"],
        peak_elevation=gs["peak"],
        ocean_trench=gs["trough"],
        mean_land_elevation=gs["mean_land_elevation"],
        mountain_ratio=gs["mountain_ratio"],
        temp_by_band=temp_by_band,
        temp_ref_by_band=temp_ref_by_band,
        precip_by_band=precip_by_band,
        precip_ref_by_band=precip_ref_by_band,
        biome_ratio=biome_ratio,
        biome_ocean_ratio=biome_ocean_ratio,
        snow_area_by_band=[float(x) for x in snow_bands],
        psd_slope=_psd_slope(v01_climate),
        fractal_dim=_fractal_dim(v01_climate),
        slope_mean=float(sm),
        slope_p95=float(sp),
        coherence=_coherence(v01_climate),
        seed=int(seed),
        size_px=int(size_px),
        downsampled=bool(downsampled),
    )
