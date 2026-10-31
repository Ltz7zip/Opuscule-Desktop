# -*- coding: utf-8 -*-
"""climate_module — 气候模块（降水 / 生物群系 / 世界气候图）。

本模块持有气候相关的**全部物理实现**，可独立优化。
依赖：``terrain``（高程 / 纬度 / 陆地掩膜）、``temperature``（大陆度 ci、
洋流降水异常 dP、温度场 T）。

输出键：
    - ``climate.precip``    年降水场 (mm/yr)
    - ``climate.biomes``    Whittaker 9 类生物群系整数标签 (int8)
    - ``climate.rgb``       世界气候图 RGB（按生物群系配色）
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter, distance_transform_edt, median_filter, label as ndi_label

from ..core.terrain_engine import value_fbm, _lowres_fbm
from .temperature_module import continentality_index, temperature_field
from .base import BaseModule, WorldContext

# Native 加速 (C++ fws_native 扩展; 不可用时回退 Numba)
try:
    from ..native import fws_native as _native
except ImportError:
    _native = None


# ===========================================================================
# 物理实现（原 terrain_engine.py 中气候相关函数）
# ===========================================================================

# 纬向基础降水 (mm/yr): ITCZ 赤道峰 → 副热带(20°~30°)干旱谷 → 温带峰 → 极地干
_P_LAT = np.array([0, 5, 10, 15, 20, 25, 30, 35, 45, 55, 65, 80, 90], np.float64)
_P_MM = np.array([4600, 4300, 3400, 2300, 1000, 700, 750, 1350, 1700, 1500, 950, 430, 250], np.float64)


# ---------------------------------------------------------------------------
# Numba JIT 加速内核（4096² 地图优化, 2026-09）
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _scan_ew_step(hk, base, ocean, h, w, step):
    """东西向地形雨扫描内核 (替代 Python for + np.where 列循环)。

    hk    : 地形高度 (km) (h, w) float32/64
    base  : 纬向基础降水 (h, w)
    ocean : 海洋掩膜 (h, w) bool
    step  : +1 自西向东, -1 自东向西
    返回 rain (h, w) float32
    """
    rain = np.zeros((h, w), dtype=np.float32)
    m = np.ones(h, dtype=np.float32)
    for _ in range(2):
        if step > 0:
            prev_j = w - 1  # (0 - 1) % w
            hp = hk[:, prev_j].copy()
            for j in range(w):
                for i in range(h):
                    hj_i = hk[i, j]
                    hp_i = hp[i]
                    if hj_i >= 2.0 and hp_i >= 2.0 and not ocean[i, j]:
                        if m[i] > 0.22:
                            m[i] = 0.22
                    diff = hj_i - hp_i
                    if diff < 0.0:
                        diff = 0.0
                    oro = 700.0 * diff * m[i]
                    if oro > 2200.0:
                        oro = 2200.0
                    rain[i, j] = (base[i, j] + oro) * (0.10 + 0.90 * m[i])
                    if ocean[i, j]:
                        m[i] = 1.0
                    else:
                        m[i] = m[i] * np.exp(-oro / 1500.0) * 0.993
                    hp[i] = hj_i
        else:
            prev_j = 0  # (w-1 - (-1)) % w = 0
            hp = hk[:, prev_j].copy()
            for j in range(w - 1, -1, -1):
                for i in range(h):
                    hj_i = hk[i, j]
                    hp_i = hp[i]
                    if hj_i >= 2.0 and hp_i >= 2.0 and not ocean[i, j]:
                        if m[i] > 0.22:
                            m[i] = 0.22
                    diff = hj_i - hp_i
                    if diff < 0.0:
                        diff = 0.0
                    oro = 700.0 * diff * m[i]
                    if oro > 2200.0:
                        oro = 2200.0
                    rain[i, j] = (base[i, j] + oro) * (0.10 + 0.90 * m[i])
                    if ocean[i, j]:
                        m[i] = 1.0
                    else:
                        m[i] = m[i] * np.exp(-oro / 1500.0) * 0.993
                    hp[i] = hj_i
    return rain


@njit(cache=True, fastmath=True)
def _scan_ns_step(hk, base, ocean, h, w, step):
    """南北向地形雨扫描内核 (替代 Python for + np.where 行循环)。

    同 _scan_ew_step 但沿行扫描, m 是 w 长度 (每行内的湿度衰减)。
    """
    rain = np.zeros((h, w), dtype=np.float32)
    m = np.ones(w, dtype=np.float32)
    for _ in range(2):
        if step > 0:
            prev_i = h - 1  # (0 - 1) % h
            hp = hk[prev_i, :].copy()
            for i in range(h):
                for j in range(w):
                    hi_i = hk[i, j]
                    hp_i = hp[j]
                    if hi_i >= 2.0 and hp_i >= 2.0 and not ocean[i, j]:
                        if m[j] > 0.22:
                            m[j] = 0.22
                    diff = hi_i - hp_i
                    if diff < 0.0:
                        diff = 0.0
                    oro = 700.0 * diff * m[j]
                    if oro > 2200.0:
                        oro = 2200.0
                    rain[i, j] = (base[i, j] + oro) * (0.10 + 0.90 * m[j])
                    if ocean[i, j]:
                        m[j] = 1.0
                    else:
                        m[j] = m[j] * np.exp(-oro / 1500.0) * 0.993
                    hp[j] = hi_i
        else:
            prev_i = 0  # (h-1 - (-1)) % h = 0
            hp = hk[prev_i, :].copy()
            for i in range(h - 1, -1, -1):
                for j in range(w):
                    hi_i = hk[i, j]
                    hp_i = hp[j]
                    if hi_i >= 2.0 and hp_i >= 2.0 and not ocean[i, j]:
                        if m[j] > 0.22:
                            m[j] = 0.22
                    diff = hi_i - hp_i
                    if diff < 0.0:
                        diff = 0.0
                    oro = 700.0 * diff * m[j]
                    if oro > 2200.0:
                        oro = 2200.0
                    rain[i, j] = (base[i, j] + oro) * (0.10 + 0.90 * m[j])
                    if ocean[i, j]:
                        m[j] = 1.0
                    else:
                        m[j] = m[j] * np.exp(-oro / 1500.0) * 0.993
                    hp[j] = hi_i
    return rain


@njit(cache=True)
def _mountain_shield(mt_sm, hh, shield, h, w, belt_rows, step_sign, K_max):
    """背风雨影扫描内核 (替代 Python 双重 for 循环)。

    mt_sm     : 山顶掩膜 (h, w) bool
    hh        : 山体高度场 (h, w) float
    shield    : 雨影屏蔽场 (h, w) float, 原地写入
    belt_rows : 需扫描的行索引 (1D int64)
    step_sign : +1 自西向东, -1 自东向西
    K_max     : 雨影最大延伸距离 (像素)
    """
    n_rows = belt_rows.shape[0]
    for ri in range(n_rows):
        i = belt_rows[ri]
        depth = 0
        strength = 0.0
        if step_sign > 0:
            for j in range(w):
                if mt_sm[i, j]:
                    depth = K_max
                    s_here = hh[i, j] * 650.0
                    if s_here > strength:
                        strength = s_here
                elif depth > 0:
                    frac = depth / K_max
                    v = strength * frac
                    if v > shield[i, j]:
                        shield[i, j] = v
                    depth -= 1
                else:
                    strength = strength - 25.0
                    if strength < 0.0:
                        strength = 0.0
        else:
            for j in range(w - 1, -1, -1):
                if mt_sm[i, j]:
                    depth = K_max
                    s_here = hh[i, j] * 650.0
                    if s_here > strength:
                        strength = s_here
                elif depth > 0:
                    frac = depth / K_max
                    v = strength * frac
                    if v > shield[i, j]:
                        shield[i, j] = v
                    depth -= 1
                else:
                    strength = strength - 25.0
                    if strength < 0.0:
                        strength = 0.0
    return shield


@njit(cache=True)
def _mode_filter_njit(bi_m, land, out, r_mode, H, W):
    """单遍 mode 滤波内核 (磁盘半径 r_mode, 排除中心像素).

    对每个陆地像素, 统计半径 r_mode 圆盘内 (排除中心) 出现次数最多的群系标签.
    海洋像素直接复制原值 (后续 where(land) 会覆盖).

    替代原 smooth_biome_labels 中:
      - cnt 数组 (H,W,9) uint16, 在 4096² 上 268MB
      - 24 次 np.roll + 24 次 cnt[yy,xx,nb] += 1 (大量临时数组)
      - cnt.argmax(axis=-1) 在 9 通道上扫描

    输入:
        bi_m  : (H, W) int8, 已中值滤波的群系标签
        land  : (H, W) bool, 陆地掩膜
        out   : (H, W) int8, 输出 (可复用, 循环外预分配)
        r_mode: int, 圆盘半径 (默认 2, 5×5 区域)
    返回:
        out (原地写入)
    """
    counts = np.zeros(9, dtype=np.int32)
    for y in range(H):
        for x in range(W):
            if not land[y, x]:
                out[y, x] = bi_m[y, x]
                continue
            # 重置计数 (避免重新分配)
            for c in range(9):
                counts[c] = 0
            for dy in range(-r_mode, r_mode + 1):
                for dx in range(-r_mode, r_mode + 1):
                    if dy == 0 and dx == 0:
                        continue
                    if dy * dy + dx * dx > r_mode * r_mode:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        counts[bi_m[ny, nx]] += 1
                    else:
                        # nearest 模式: 越界用中心值
                        counts[bi_m[y, x]] += 1
            best_c = 0
            best_cnt = -1
            for c in range(9):
                if counts[c] > best_cnt:
                    best_cnt = counts[c]
                    best_c = c
            out[y, x] = best_c
    return out


@njit(cache=True)
def _median_filter_njit(bi_m, land, out, r_med, H, W):
    """单遍中值滤波 (圆盘半径 r_med, 含中心).

    对每个陆地像素, 收集圆盘内所有邻居值, 排序取中值.
    海洋像素直接复制原值. 替代 scipy median_filter (~3x 加速 on 4096²).

    输入:
        bi_m  : (H, W) int8, 群系标签
        land  : (H, W) bool, 陆地掩膜
        out   : (H, W) int8, 输出 (可复用, 循环外预分配)
        r_med : int, 圆盘半径 (默认 3, 7×7 区域 ≈37 像素)
    返回:
        out (原地写入)
    """
    vals = np.zeros(49, dtype=np.int8)  # 最大 7×7=49
    for y in range(H):
        for x in range(W):
            if not land[y, x]:
                out[y, x] = bi_m[y, x]
                continue
            n = 0
            for dy in range(-r_med, r_med + 1):
                for dx in range(-r_med, r_med + 1):
                    if dy * dy + dx * dx > r_med * r_med:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        vals[n] = bi_m[ny, nx]
                    else:
                        vals[n] = bi_m[y, x]
                    n += 1
            # 插入排序 (n ≤ 49, 小数组比快排快)
            for i in range(n):
                for j in range(i + 1, n):
                    if vals[j] < vals[i]:
                        tmp = vals[i]
                        vals[i] = vals[j]
                        vals[j] = tmp
            out[y, x] = vals[n // 2]
    return out


def precipitation_field(lats, elev_m, rng=None, dP=None, ci=None,
                       _elev_climate_smooth=None, _hf_norm=None, _gate=None,
                       _d_sea_km=None):
    """v7.1 年降水场 (mm/yr) — 六重改进, 打破严格纬向条带感.

    v9.x 优化: 接受预计算的 elev_climate_smooth / hf_norm / gate / d_sea_km 以避免
    与 temperature_field / continentality_index 重复计算.
    """
    h, w = elev_m.shape
    a = np.abs(np.asarray(lats, np.float64))
    if ci is None:
        ci = continentality_index(elev_m, lats, _d_sea_km=_d_sea_km)
    ci64 = ci.astype(np.float64)
    land = elev_m > 0.0
    elev_pos_raw = np.maximum(elev_m, 0.0)
    kmpp = 360.0 / max(w, 1) * 111.32
    if _d_sea_km is not None:
        d_sea_km = _d_sea_km
    else:
        d_sea_km = distance_transform_edt(~((elev_m < 0.0).astype(bool))) * kmpp

    base = np.interp(a, _P_LAT, _P_MM)[:, None] * np.ones((1, w), np.float64)

    lat_penalty = np.ones_like(a[:, None], dtype=np.float64)
    a2d = a[:, None]
    subtrop_band = (a2d >= 18.0) & (a2d <= 34.0)
    westerly_band = (a2d > 34.0) & (a2d <= 60.0)
    polar_band = (a2d > 60.0)
    eq_ci_band = (a2d < 15.0)
    lat_penalty = np.where(subtrop_band, 1.05, lat_penalty)
    lat_penalty = np.where(westerly_band, 0.55, lat_penalty)
    lat_penalty = np.where(polar_band, 0.70, lat_penalty)
    lat_penalty = np.where(eq_ci_band, 0.50, lat_penalty)
    f_ci = 1.0 - (0.35 * ci64) * lat_penalty
    base *= f_ci

    # v20: 副热带降水抑制 1500→2000mm, 扩大副热带荒漠 (v19 desert_in_subtropics 14.5% vs 地球 37%)
    # v39d: 2000→1600mm / σ 7.5→6.5 / sub_mod 1.5→1.3. 旧版把 10°~45° 整条压成荒漠
    #       (荒漠 27.5% 且连成一片), 副热带干旱带应收窄在 18-33°, 南北两侧留出
    #       稀树草原与温带草原的过渡带.
    subtrop_suppress = 1600.0 * np.exp(-((a[:, None] - 25.0) ** 2) / (2 * 6.5 ** 2))
    if rng is not None:
        # v9.x: 低分辨率噪声上采样 (省 ~1.2s on 4096²)
        fbm_raw = np.clip(_lowres_fbm(h, 5.0, 3, rng), -1.0, 1.0)
        sub_mod = 1.3 / (1.0 + np.exp(-3.2 * (fbm_raw + 0.05)))
        subtrop_suppress = subtrop_suppress * sub_mod
    east_coast_relief = np.where(ci64 < 0.15, 0.10,
                                 np.where(ci64 < 0.35, 0.40,
                                          np.clip(0.70 + 0.25 * ci64, 0.60, 0.95)))
    dP_sub = -subtrop_suppress * east_coast_relief * land
    base = np.maximum(base + dP_sub, 0.0)

    # v20: dry_cap 从 <10° 扩展至 <23.5° (整个热带带), 压缩热带季节林
    # v33: 赤道湿带拆分 —— 旧版把整个 <23.5° 都压到 ≤1350mm (沿海), 导致赤道
    #      (0-10°) 也不可能出现热带雨林 (P≥3000mm), 雨林占比≈0 (地球 0.9% 陆地).
    #      现保留 10-23.5° 的稀树草原/荒漠压制, 但 0-10° 湿赤道带放开上界:
    #      沿海对流雨 ~3400mm (雨林), 内陆 2000mm (热带季节林) → 纬度地带性重现.
    wet_eq = np.zeros((h, w), dtype=bool)          # v33b: 防御性初始化 (无赤道陆地时)
    low_eq = np.zeros((h, w), dtype=bool)
    eq_band = (a[:, None] < 23.5) & land
    if eq_band.any():
        # 海岸 1350mm (稀树草原/季节林边界), 内陆快速降至 350mm (荒漠)
        dry_cap = 350.0 + 1000.0 * np.exp(-d_sea_km / 1100.0)
        # v33: 0-10° 湿赤道带 (ITCZ): 上界抬高至 1800(内陆)~3400(沿海)mm
        wet_eq = (a[:, None] < 10.0)
        wet_cap = 1800.0 + 1600.0 * np.exp(-d_sea_km / 2400.0)
        # v39e: 10° / 23.5° 硬切 → 10-23.5° 线性过渡. 旧版在 10-23.5° 一律套
        #       dry_cap(≤1350mm), 把整个热带压成 P≈760mm → 10-30° 全纬度带 72-83%
        #       被判成荒漠, 稀树草原带消失. 现按纬度把干/湿上界插值:
        #       10° ≈ 湿赤道上界, 23.5° ≈ 副热带干上界, 中间是稀树草原的 1400-1800mm.
        _eq_s = np.clip((a[:, None] - 10.0) / 13.5, 0.0, 1.0)
        dry_cap = wet_cap * (1.0 - _eq_s) + dry_cap * _eq_s
        dry_cap = np.where(wet_eq, wet_cap, dry_cap)
        low_eq = (elev_pos_raw < 300.0) & eq_band
        if low_eq.any() and w > 0:
            lon = (np.arange(w, dtype=np.float64) / w) * 2.0 * np.pi
            walker_mosaic = 0.55 + 0.45 * (np.cos(3.0 * lon + 0.7) ** 2)[None, :]
            inland_f = np.clip(d_sea_km / 2800.0, 0.0, 1.0)
            savanna_f = 1.0 - 0.32 * inland_f * (1.0 - walker_mosaic)
            dry_cap = np.where(low_eq, dry_cap * savanna_f, dry_cap)
        hi_eq = (elev_pos_raw > 1000.0) & eq_band
        if hi_eq.any():
            dry_cap = np.where(hi_eq, dry_cap + 700.0, dry_cap)
        base = np.where(eq_band,
                        np.minimum(base, dry_cap * (0.92 + 0.08 * (1.0 - ci64))),
                        base)

    elev_pos_raw = np.maximum(elev_m, 0.0)
    if _elev_climate_smooth is not None and _hf_norm is not None and _gate is not None:
        elev_climate_smooth = _elev_climate_smooth
        hf_norm = _hf_norm
        rugged_gate = _gate
    else:
        elev_climate_smooth = gaussian_filter(elev_pos_raw, 2.2)
        highfreq = elev_pos_raw - gaussian_filter(elev_pos_raw, 0.9)
        highfreq = gaussian_filter(highfreq, 0.5)
        hf_norm = np.clip(highfreq / np.maximum(np.max(np.abs(highfreq)), 1.0), -1.0, 1.0)
        rugged_gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)
    elev_climate = elev_climate_smooth * (1.0 + 0.15 * hf_norm * rugged_gate)
    elev_climate = np.maximum(elev_climate, 0.0)
    hk = elev_climate / 1000.0
    ocean = (elev_m < 0.0) | (hk < 1e-3)

    # Numba JIT 加速东西向/南北向地形雨扫描 (替代 scan_ew / scan_ns 闭包)
    # v10: C++ fws_native 加速 (无 JIT 编译/分配开销), 不可用时回退 Numba
    if _native is not None:
        p_w = _native.scan_ew_step(hk, base, ocean, h, w, +1)
        p_e = _native.scan_ew_step(hk, base, ocean, h, w, -1)
        p_n = _native.scan_ns_step(hk, base, ocean, h, w, +1)
        p_s = _native.scan_ns_step(hk, base, ocean, h, w, -1)
    else:
        p_w = _scan_ew_step(hk, base, ocean, h, w, +1)
        p_e = _scan_ew_step(hk, base, ocean, h, w, -1)
        p_n = _scan_ns_step(hk, base, ocean, h, w, +1)
        p_s = _scan_ns_step(hk, base, ocean, h, w, -1)
    west_belt = ((a >= 30.0) & (a <= 60.0))[:, None]
    P_ew = np.where(west_belt, p_w, p_e).astype(np.float64)
    monsoon_belt = (a < 40.0)[:, None]
    P_ns = np.where(monsoon_belt, p_n, p_s).astype(np.float64)
    ns_w = 0.25 * np.ones((h, 1), np.float64)
    eq_band_ns = (a < 15.0)[:, None]
    pol_band_ns = (a > 65.0)[:, None]
    ns_w[eq_band_ns[:, 0], :] = 0.35
    ns_w[pol_band_ns[:, 0], :] = 0.35
    P = (1.0 - ns_w) * P_ew + ns_w * P_ns

    shield = np.zeros((h, w), np.float64)
    mountain_top = (elev_climate >= 2500.0) & land
    if mountain_top.any():
        mt_sm = gaussian_filter(mountain_top.astype(np.float32), 1.2) > 0.35
        hh = (elev_climate - 2000.0) / 1000.0 * land
        hh = np.maximum(hh, 0.0)
        belt_west = (a >= 30) & (a <= 60)
        belt_east = ~belt_west
        K_max = int(np.ceil(900.0 / max(kmpp, 1.0)))
        # Numba JIT 加速背风雨影扫描 (替代 Python 双重 for 循环); v10: C++ 版优先
        for step_sign, belt_msk in [(+1, belt_west), (-1, belt_east)]:
            belt_rows = np.nonzero(belt_msk)[0].astype(np.int64)
            if belt_rows.size == 0:
                continue
            if _native is not None:
                shield = _native.mountain_shield(mt_sm, hh, shield, h, w,
                                                 belt_rows, step_sign, K_max)
            else:
                shield = _mountain_shield(mt_sm, hh, shield, h, w,
                                           belt_rows, step_sign, K_max)
    P = np.maximum(P - shield, 0.0)
    # v33b: ITCZ 对流雨保底 —— 东西/南北地形雨扫描的水汽走廊衰减 (0.10+0.90·m)
    #       会把赤道内陆降水压到 base 的 ~30% (实际 P 中位 ~1100mm, 全是荒漠/稀疏草原).
    #       但地球热带对流雨 (阿马孙/刚果内陆 2000-3000mm) 由局地蒸发-对流驱动,
    #       不受盛行西风水汽走廊限制. 这里给 0-12° 湿赤道带一个随纬度渐消的对流雨下限:
    #       0° ≈2600mm (雨林) → 12° →0, 使纬度地带性 (赤道雨林带) 真正出现.
    if wet_eq.any():
        _itcz = np.clip(1.0 - (a2d - 1.0) / 11.0, 0.0, 1.0)
        if low_eq.any() and w > 0:
            _lon = (np.arange(w, dtype=np.float64) / w) * 2.0 * np.pi
            _wm = 0.55 + 0.45 * (np.cos(3.0 * _lon + 0.7) ** 2)[None, :]
            _itcz = _itcz * (0.72 + 0.28 * _wm)          # 经度向季节雨调制 (保留稀树草原纹理)
        P = np.maximum(P, 2600.0 * _itcz * land * wet_eq)
    P = gaussian_filter(P.astype(np.float32), [2.2, 1.2]).astype(np.float64)

    if dP is not None:
        P = np.maximum(P + dP, 0.0)

    if rng is not None:
        # v9.x: 低分辨率噪声上采样 (省 ~2.4s on 4096², 2 calls)
        lw = _lowres_fbm(w, 3.2, 2, rng)
        lat_g = 0.08 + 0.20 * np.exp(-((a - 58.0) ** 2) / (2 * 22.0 ** 2))
        P = P * np.exp(lw * lat_g[:, None])
        P = P * np.exp(0.10 * _lowres_fbm(w, 5.5, 3, rng))

    return np.clip(P, 0.0, None).astype(np.float32)


BIOME_NAMES = ["苔原", "北方针叶林", "温带草原", "温带落叶林", "温带雨林",
               "亚热带荒漠", "稀树草原", "热带季节林", "热带雨林"]
BIOME_COLORS = np.array([
    # 顺序: 苔原(0)/北方针叶林(1)/温带草原(2)/温带落叶林(3)/温带雨林(4)/
    #        荒漠(5)/稀树草原(6)/热带季节林(7)/热带雨林(8)
    # v34: 提高气候带区分度 — 荒漠更亮沙、雨林更深绿、针叶林更冷暗,
    #      修复"地图只有绿/棕, 沙漠带和雨林带不明显"的观感.
    # v35b: 提亮森林/草地色系 —— v34 把森林类整体调暗, 叠加 shade 低地压制后
    #      大陆中央低地呈大面积暗绿/暗褐"阴影区"(用户反馈); 卫星图陆地不暗.
    # v39: 去饱和 — 用户反馈"色调像油画, 沙黄/亮绿饱和度过高, 削弱高程表现力".
    #      做法: 每个色向自身亮度靠 18% (保亮度、降彩度), 而不是向白色靠
    #      (向白靠会让荒漠更惨白、整图发灰、山体阴影被冲淡).
    # v46: 温带草原 / 稀树草原 / 荒漠 三色原先几乎是同一个黄 (ΔRGB<0.05),
    #      地图上分不出"草原—稀树草原—荒漠". 改为 橄榄黄绿 / 暖黄褐 / 亮沙 三级,
    #      同时保留 v39 的低饱和度 (不向白靠, 免得荒漠惨白、山体阴影被冲掉).
    # v52: 苔原 0.655/0.630/0.570 (灰米黄) → 0.560/0.575/0.480 (灰橄榄).
    #   极区陆地占全图陆地近半, 旧色比温带草原还亮, 整图因此读成"一大片米黄".
    #   真实苔原夏季影像偏灰绿-橄榄, 与荒漠的亮沙、稀树草原的暖黄褐拉开层级.
    [0.560, 0.575, 0.480],   # 0 苔原: 灰橄榄
    [0.370, 0.444, 0.362],   # 1 北方针叶林: 冷绿
    [0.694, 0.706, 0.420],   # 2 温带草原: 橄榄黄绿
    [0.530, 0.628, 0.398],   # 3 温带落叶林: 明绿
    [0.301, 0.465, 0.301],   # 4 温带雨林: 绿
    [0.906, 0.822, 0.612],   # 5 荒漠: 亮沙
    [0.816, 0.664, 0.402],   # 6 稀树草原: 暖黄褐
    [0.475, 0.614, 0.344],   # 7 热带季节林: 绿
    [0.206, 0.402, 0.214]], np.float32)  # 8 热带雨林: 深绿


def whittaker_biome(t, p, lats=None, ci=None):
    """Whittaker (T, P) → 9 类生物群系 (v41 年均温物理口径).

    v41 标定: 删掉 v23/v39 的**纬度 hack** —— 旧版在 >55° 给针叶林上界叠加
    ``+3·lat_w2 + 2(1−ci)·lat_w2``, 使 66.5° 处针叶林可延伸到 T≈19°C,
    寒温带群系吞掉整个温带 (实测北方针叶林 23.9% / 苔原 20.8%, 参考 5% / 10%).
    现全部改为年均温 + 年降水的物理口径 (Whittaker 1975 图解, 地球群系调查):

    ==============  ==========================================================
    苔原            T ≤ −3 − 2·ci    无夏 (最热月 <10°C ≈ 年均 −3~−5°C)
    北方针叶林       T ≤  4 + 2(1−ci) 凉夏 (大陆 4°C / 海洋性 6°C)
    温带草原/森林    P ≈ 700 + 150·ci 地球林草过渡带 600-900mm
    热带雨林         P ≥ 2200mm       地球雨林下限 ~2000mm (旧版 2600 过高,
                                      赤道陆地降水中位仅 1100mm → 雨林 2.4%)
    ==============  ==========================================================

    标定后 (占陆地): 苔原 18.6 / 针叶林 13.5 / 温带草原 13.9 / 落叶林 14.0 /
    温带雨林 0.5 / 荒漠 24.8 / 稀树草原 6.7 / 热带季节林 1.8 / 热带雨林 6.2.
    """
    t_arr = np.asarray(t)
    p_arr = np.asarray(p)
    if ci is not None:
        ci64 = np.asarray(ci, np.float64)
        t_tundra = -3.0 - 2.0 * ci64
        t_taiga_up = 4.0 + 2.0 * (1.0 - ci64)
        p_stp = 700.0 + 150.0 * ci64
        p_tf = 2400.0 - 200.0 * (1.0 - ci64)
    else:
        t_tundra = -3.0
        t_taiga_up = 5.0
        p_stp = 800.0
        p_tf = 2300.0
    p_des, p_sav, p_rf = 1300.0, 2000.0, 2200.0

    if lats is not None and ci is not None:
        a = np.abs(np.asarray(lats, np.float64))[:, None]
        ci64_2d = np.asarray(ci, np.float64)
        # 副热带冷荒漠: 与热带荒漠阈值一致 (1300mm)
        cold_desert_override = (a >= 18.0) & (a <= 34.0) & (ci64_2d >= 0.15) \
                             & (p_arr < p_des) & (t_arr >= 2.0) & (t_arr <= 22.0)
    else:
        cold_desert_override = np.zeros(t_arr.shape, dtype=bool)

    return np.select(
        [t_arr <= t_tundra, t_arr <= t_taiga_up,
         cold_desert_override,
         (t_arr <= 20) & (p_arr < p_stp),
         (t_arr <= 20) & (p_arr < p_tf),
         t_arr <= 20,
         p_arr < p_des, p_arr < p_sav, p_arr < p_rf],
        [0, 1, 5, 2, 3, 4, 5, 6, 7], default=8).astype(np.int8)


def smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3):
    """v7.9 群系边界平滑 + 碎片清理 — 修复 Whittaker 硬阈值导致的锯齿边界.

    v9.x 优化: 中值滤波改用 Numba njit (替代 scipy median_filter, ~3x 加速);
    mode 滤波改用 Numba njit 单遍 (替代 268MB cnt 数组 + 24 np.roll + argmax).
    """
    bi8 = bi.astype(np.int8).copy()
    H, W = bi8.shape
    land8 = np.ascontiguousarray(land, dtype=np.bool_)
    r_med = max(1, median_size // 2)
    r_mode = 2

    # 预分配输出缓冲 (循环外只算一次)
    mode_bi = np.empty_like(bi8)
    med_bi = np.empty_like(bi8)

    for _it in range(iters):
        # 中值滤波 (v10: C++ fws_native 优先, 回退 Numba njit)
        if _native is not None:
            _native.median_filter_njit(np.ascontiguousarray(bi8), land8,
                                       med_bi, r_med, H, W)
        else:
            _median_filter_njit(np.ascontiguousarray(bi8), land8, med_bi, r_med, H, W)
        bi_m = med_bi.copy()  # copy 防止下轮迭代 bi8 与 med_bi 别名
        if min_patch_px > 0:
            # mode 滤波 (v10: C++ fws_native 优先, 回退 Numba njit)
            bi_m_c = np.ascontiguousarray(bi_m, dtype=np.int8)
            if _native is not None:
                mode_bi = _native.mode_filter_njit(bi_m_c, land8, mode_bi,
                                                   r_mode, H, W)
            else:
                mode_bi = _mode_filter_njit(bi_m_c, land8, mode_bi, r_mode, H, W)
            # 碎片清理 (v11: C++ 单遍 BFS 连通分量, 回退原 9×ndi_label 循环)
            if _native is not None:
                bi_m = _native.smooth_small_patches(bi_m_c, mode_bi, land8,
                                                    min_patch_px, H, W)
            else:
                for idx in range(9):
                    m = land8 & (bi_m == idx)
                    if not m.any():
                        continue
                    lab, n = ndi_label(m)
                    if n == 0:
                        continue
                    sizes = np.bincount(lab.ravel())
                    sizes[0] = 0
                    small_labels = np.where(sizes < min_patch_px)[0]
                    if len(small_labels) == 0:
                        continue
                    small_mask = np.isin(lab, small_labels) & land8
                    bi_m = np.where(small_mask, mode_bi, bi_m)
        bi8 = bi_m
    return bi8.astype(bi.dtype)


# ===========================================================================
# 模块类
# ===========================================================================

class ClimateModule(BaseModule):
    """气候模块：计算降水场与生物群系分区，渲染世界气候图。

    依赖：``terrain``（高程 / 纬度 / 陆地掩膜）、``temperature``
    （大陆度 ci、降水异常 dP、温度场 T）。

    输出键（写入 ``ctx.data``）：
        - ``climate.precip``  年降水场 (mm/yr)
        - ``climate.biomes``  Whittaker 9 类生物群系整数标签 (int8)
    """
    name = "climate"
    dependencies = ["terrain", "temperature"]

    def compute(self, ctx: WorldContext) -> None:
        """计算年降水场并按 Whittaker 方案划分生物群系写入 ``ctx.data``。"""
        dem_m = ctx.dem_m
        lats = ctx.lats
        ci = ctx.get("temperature.ci")
        dP = ctx.get("temperature.dP")
        T = ctx.get("temperature.temp")
        land = ctx.land_mask
        P = precipitation_field(lats, dem_m, rng=ctx.rng, dP=dP, ci=ci)
        ctx.set("climate.precip", P)
        if T is None:
            dT = ctx.get("temperature.dT")
            T = temperature_field(lats, dem_m, dT=dT, ci=ci)
        bi = whittaker_biome(T, P, lats=lats, ci=ci)
        bi = smooth_biome_labels(bi, land, median_size=7, min_patch_px=50, iters=3)
        ctx.set("climate.biomes", bi)

    def render(self, ctx: WorldContext) -> np.ndarray:
        """渲染世界气候图：按生物群系配色，海洋为深蓝色底。"""
        bi = ctx.get("climate.biomes")
        land = ctx.land_mask
        S = ctx.S
        if bi is None:
            self.compute(ctx)
            bi = ctx.get("climate.biomes")
        rgb = np.zeros((S, S, 3), dtype=np.float32)
        sea = ~land
        rgb[sea] = np.array([0.05, 0.15, 0.32], dtype=np.float32)
        if land.any():
            rgb[land] = BIOME_COLORS[bi[land]]
        return np.clip(rgb, 0, 1).astype(np.float32)


__all__ = ["ClimateModule", "BIOME_NAMES", "BIOME_COLORS",
           "precipitation_field", "whittaker_biome", "smooth_biome_labels",
           "_P_LAT", "_P_MM"]
