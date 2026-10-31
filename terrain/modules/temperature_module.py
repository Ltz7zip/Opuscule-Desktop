# -*- coding: utf-8 -*-
"""temperature_module — 气温模块（大陆度 / 洋流 / 温度场 / 世界气温图）。

本模块持有气温相关的**全部物理实现**，可独立优化而不影响地形/水系/气候模块。
依赖：``terrain``（需要米制高程与纬度）。

输出键：
    - ``temperature.ci``        大陆度指数场 [0, 1]
    - ``temperature.dT``        洋流/季风温度异常 (℃)
    - ``temperature.dP``        洋流/季风降水异常 (mm/yr)（供气候模块复用）
    - ``temperature.temp``      年均气温场 (℃)
    - ``temperature.rgb``       世界气温图 RGB（蓝→绿→红 温度色卡）
"""
from __future__ import annotations

import numpy as np
from numba import njit
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, distance_transform_edt, label as ndi_label

from ..core.terrain_engine import snowline_m  # noqa: F401  (再导出, 保持兼容)
from .base import BaseModule, WorldContext

# Native 加速 (C++ fws_native 扩展; 不可用时回退 Numba)
try:
    from ..native import fws_native as _native
except ImportError:
    _native = None


# ===========================================================================
# 物理实现（原 terrain_engine.py 中气温相关函数，搬迁至此便于独立优化）
# ===========================================================================

# ---------------------------------------------------------------------------
# Numba JIT 加速内核（4096² 地图优化, 2026-09）
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ocean_run_lengths_njit(sea, S, W):
    """东西向海洋游程长度内核 (替代 Python for + np.where 列循环).

    sea  : (S, W) bool 海洋掩膜
    返回 run_e, run_w: (S, W) int32, 每像素向东/向西连续海像素数.
    2 遍扫描处理全球经度环绕 (与原 Python 逻辑一致).
    """
    run_e = np.zeros((S, W), dtype=np.int32)
    run_w = np.zeros((S, W), dtype=np.int32)
    for i in range(S):
        # East pass: j 从 W-1 递减到 0
        run = 0
        for _ in range(2):
            for j in range(W - 1, -1, -1):
                if sea[i, j]:
                    run += 1
                else:
                    run = 0
                run_e[i, j] = run
        # West pass: j 从 0 到 W-1
        run = 0
        for _ in range(2):
            for j in range(W):
                if sea[i, j]:
                    run += 1
                else:
                    run = 0
                run_w[i, j] = run
    return run_e, run_w


def _edt_lowres(mask, kmpp, downsample=4):
    """低分辨率 EDT + 最近邻上采样 (海岸距离场专用).

    mask = 前景 (True=远离海岸, False=海岸). EDT 计算每个像素到最近海岸的距离.
    在 size//downsample 分辨率计算 distance_transform_edt, 再上采样回原尺寸.
    海岸距离场平滑且经 exp(-d/D) 衰减 (D≥300km), 4x 下采样误差 ~13km 可忽略.
    v9.x: 原 EDT(4096²) ~0.85s, 本函数 ~0.07s (12x 加速).
    """
    H, W = mask.shape
    lo_h = max(H // downsample, 8)
    lo_w = max(W // downsample, 8)
    H2 = lo_h * downsample
    W2 = lo_w * downsample
    # 下采样: 对海岸掩膜 (~mask, 稀疏) 做 any pooling, 保留稀疏海岸线
    coast_lo = (~mask[:H2, :W2]).reshape(lo_h, downsample, lo_w, downsample).any(axis=(1, 3))
    mask_lo = ~coast_lo  # 前景 (True=远离海岸)
    d_lo = distance_transform_edt(mask_lo) * (kmpp * downsample)
    # 最近邻上采样回原尺寸
    d_hi = np.repeat(np.repeat(d_lo, downsample, axis=0), downsample, axis=1)
    # 若原尺寸大于 H2/W2, 用边缘填充补齐
    if d_hi.shape[0] < H or d_hi.shape[1] < W:
        d_full = np.empty((H, W), dtype=d_hi.dtype)
        d_full[:d_hi.shape[0], :d_hi.shape[1]] = d_hi
        if d_hi.shape[0] < H:
            d_full[d_hi.shape[0]:, :] = d_hi[-1:, :]
        if d_hi.shape[1] < W:
            d_full[:, d_hi.shape[1]:] = d_full[:, d_hi.shape[1]-1:d_hi.shape[1]]
        d_hi = d_full
    return d_hi


def continentality_index(elev_m, lats=None, _d_sea_km=None):
    """v7.1 大陆度指数 Continentality Index (Gorchinsky 式简化版).

    物理含义:
        CI ∈ [0, 1]  —— 0 = 纯海洋性气候(如太平洋小岛、西欧海岸)
                       1 = 极端大陆性气候(西伯利亚腹地、塔里木盆地)

    算法 (与地球口径一致, 合成世界/真实 DEM 同口径):
        1. d_sea_km  = distance_transform_edt(elev_m<0) × km_per_px   # 距最近海(km)
        2. base      = 1 − exp(−d_sea_km / 1800)                      # 1800km 半饱和
        3. lat_weight = 0.55 + 0.45·clip((|lat|−15)/75, 0, 1)         # 高纬放大(极地气团入侵)
        4. CI        = base × lat_weight, 仅陆地像素; 海洋像素 = 0

    大陆度用来修正:
        - 温度: 深内陆 + 高纬 → 冬季更冷(年均温 −2~−8℃); 热带内陆微暖 (+0.8℃)
        - 降水: 深内陆降水打 (1 − 0.55·CI) 折 (中亚戈壁、北美大平原干旱)
        - 生物群系门限: 苔原/针叶林温度阈值在大陆腹地向赤道推移 (雅库茨克式)

    v9.x 优化: 接受预计算的 _d_sea_km 以避免与 precipitation_field 重复 (~1s on 4096²).
    """
    S, W = elev_m.shape
    kmpp = 360.0 / max(W, 1) * 111.32
    sea = elev_m < 0.0
    if _d_sea_km is not None:
        d_sea_km = _d_sea_km
    else:
        d_sea_px = distance_transform_edt(~sea.astype(bool))
        d_sea_km = d_sea_px * kmpp
    base = 1.0 - np.exp(-d_sea_km / 1800.0)
    if lats is not None:
        a = np.abs(np.asarray(lats, np.float64))[:, None]
        lw = 0.55 + 0.45 * np.clip((a - 15.0) / 75.0, 0.0, 1.0)
    else:
        lw = 1.0
    ci = base * lw
    ci[sea] = 0.0
    return np.clip(ci, 0.0, 1.0).astype(np.float32)


def ocean_current_effects(elev_m, lats, warmT=3.5, warmP=250., warmD=400.,
                          coldP=550., coldD=500., monP=750., monD=550.,
                          wprP=400., wprD=300.):
    """M4 洋流/季风气候效应 (海岸几何驱动, 地球/生成图同口径).
    返回 (dT, dP); 陆地外为 0。"""
    land = elev_m > 0
    S, W = elev_m.shape
    a = np.abs(np.asarray(lats, np.float64))[:, None] * np.ones((1, W))
    sea = ~land
    sea_e = np.roll(sea, -1, 1); land_e = np.roll(land, -1, 1)
    sea_w = np.roll(sea, 1, 1);  land_w = np.roll(land, 1, 1)
    east_sea = land & sea_e & land_w
    west_sea = land & sea_w & land_e
    K = 24
    # v9.x: Numba JIT 加速东西游程; v10: C++ fws_native 优先
    if _native is not None:
        run_e, run_w = _native.ocean_run_lengths(sea, S, W)
    else:
        run_e, run_w = _ocean_run_lengths_njit(sea, S, W)
    east_open = east_sea & (np.roll(run_e, -1, 1) >= K)
    west_open = west_sea & (np.roll(run_w, 1, 1) >= K)

    kmpp = 360.0 / W * 111.32
    dT = np.zeros((S, W), np.float64)
    dP = np.zeros((S, W), np.float64)
    lat_gate = np.clip((a - 27.0) / 8.0, 0.0, 1.0)

    lab, n = ndi_label(land)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    big_cont = np.isin(lab, np.nonzero(sz >= 0.015 * land.sum())[0])

    warm = east_open & (a >= 30.0) & (a <= 60.0) & (lat_gate > 0)
    if warm.any():
        dw = _edt_lowres(~warm, kmpp)
        g = np.minimum(lat_gate, 1.0)
        dT += warmT * np.exp(-dw / 600.0) * g * land
        if warmP > 0:
            dP += warmP * np.exp(-dw / warmD) * g * land
    cold = west_open & (a >= 5.0) & (a <= 35.0)
    if cold.any() and coldP > 0:
        dc = _edt_lowres(~cold, kmpp)
        dP -= coldP * np.exp(-dc / coldD) * land
    mon = east_open & (a >= 5.0) & (a <= 28.0) & big_cont
    if mon.any() and monP > 0:
        dm = _edt_lowres(~mon, kmpp)
        dP += monP * np.exp(-dm / monD) * land
        mon_ext = east_open & (a >= 5.0) & (a <= 42.0) & big_cont
        dm_ext = _edt_lowres(~mon_ext, kmpp)
        tailP = 260.0
        tailD = 1400.0
        tail = tailP * np.exp(-dm_ext / tailD) * land
        tail *= np.clip(1.0 - np.clip((a - 28.0) / 14.0, 0.0, 1.0) * 0.7, 0.3, 1.0)
        tail[dm < 550.0] = 0.0
        dP += tail
    if wprP > 0:
        wpr = west_open & (a >= 38.0) & (a <= 62.0)
        if wpr.any():
            dq = _edt_lowres(~wpr, kmpp)
            dP += wprP * np.exp(-dq / wprD) * land
    return dT, dP


def temperature_field(lats, elev_m, dT=None, ci=None,
                      _elev_climate_smooth=None, _hf_norm=None, _gate=None):
    """计算近地面年均气温场（℃，逐像素）。

    v8.0 地球纬向梯度拟合（非线性）：
        T = 27 − 0.05·|lat| − 0.006·|lat|² − 6.5·max(elev,0)/1000
            + dT_continentality(ci, lat)
            + dT_subtropical_sink(lat) + dT

    v9.x 优化: 接受预计算的 elev_climate_smooth / hf_norm / gate 以避免与
    precipitation_field 重复计算 (3 个 gaussian_filter 调用, 4096² 上约 2s).
    """
    a = np.abs(np.asarray(lats, np.float64))[:, None]
    if _elev_climate_smooth is not None and _hf_norm is not None and _gate is not None:
        elev_climate_smooth = _elev_climate_smooth
        hf_norm = _hf_norm
        gate = _gate
    else:
        elev_raw_pos = np.maximum(elev_m, 0.0)
        elev_climate_smooth = gaussian_filter(elev_raw_pos, 2.2)
        hf = elev_raw_pos - gaussian_filter(elev_raw_pos, 0.9)
        hf = gaussian_filter(hf, 0.5)
        hf_norm = np.clip(hf / np.maximum(np.max(np.abs(hf)), 1.0), -1.0, 1.0)
        gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)
    elev_climate = np.maximum(elev_climate_smooth * (1.0 + 0.12 * hf_norm * gate), 0.0)
    # 地球纬向温度梯度（非线性拟合）：热带平缓、极地陡峭
    # T_sea = 27 - 0.05·|lat| - 0.006·|lat|²
    #   0°=27, 30°=20.1, 60°=2.4, 90°=-26.4（与地球一致）
    lat_gradient = 0.05 * a + 0.006 * a ** 2
    if ci is None:
        ci = continentality_index(elev_m, lats)
    ci64 = ci.astype(np.float64)
    cold_term = np.clip((a - 30.0) / 60.0, 0.0, 1.0) * 8.0
    warm_term = np.clip(1.0 - np.abs(a - 24.0) / 14.0, 0.0, 1.0) * 1.5
    polar_w = np.clip((a - 58.0) / 20.0, 0.0, 1.0)
    cold_term_eff = cold_term * (1.0 - 0.60 * polar_w)
    dT_ci = ci64 * (warm_term - cold_term_eff)
    subtrop_warm = 2.3 * np.exp(-((a - 25.0) ** 2) / (2 * 6.5 ** 2))
    dT_sub = subtrop_warm * np.clip(ci64 * 1.6 + 0.1, 0.0, 1.0)
    # 极地冰盖冷却：模拟永久冰盖反照率正反馈
    # 地球南极冰盖因反照率+高程+涡旋，比无冰时低 40-45℃
    # v9.x: 起算 55→50°，上限 38→45℃ (75°约30℃, 60°约15℃)
    # v22: ice_cool 增强至 60℃ — 即使 supervisor override 带均值, 极地最冷端仍更冷, 有助苔原/积雪
    # v39d: 起算 48°→60° 且上限 60→30℃. 旧版在 55° 就砍掉 10℃、60° 砍 17℃,
    #       把整条 50-65° 带压成 T≈-9℃ 的苔原(占陆地 34.8%, 地球 ~7%), 北方针叶林
    #       只剩 0.4%. 反照率正反馈只应作用于真正有冰盖的极区(>60°), 中纬不该付这笔冷.
    ice_cool = np.clip((a - 60.0) / 30.0, 0.0, 1.0) ** 1.3 * 30.0
    T = 29.0 - lat_gradient - 6.5 * elev_climate / 1000.0 \
        + dT_ci + dT_sub - ice_cool
    if dT is not None:
        T = T + dT
    return T


# ===========================================================================
# 模块类
# ===========================================================================

class TemperatureModule(BaseModule):
    """气温模块：计算大陆度 / 洋流异常 / 温度场，渲染世界气温图。

    依赖：``terrain``（米制高程与纬度）。

    输出键（写入 ``ctx.data``）：
        - ``temperature.ci``    大陆度指数场
        - ``temperature.dT``    洋流/季风温度异常 (℃)
        - ``temperature.dP``    洋流/季风降水异常 (mm/yr，供气候模块复用)
        - ``temperature.temp``  年均气温场 (℃)
    """
    name = "temperature"
    dependencies = ["terrain"]

    def compute(self, ctx: WorldContext) -> None:
        """计算大陆度、洋流异常与年均气温场并写入 ``ctx.data``。"""
        dem_m = ctx.dem_m
        lats = ctx.lats
        ci = continentality_index(dem_m, lats)
        dT, dP = ocean_current_effects(dem_m, lats)
        T = temperature_field(lats, dem_m, dT=dT, ci=ci)
        ctx.set("temperature.ci", ci)
        ctx.set("temperature.dT", dT)
        ctx.set("temperature.dP", dP)
        ctx.set("temperature.temp", T.astype(np.float32))

    def render(self, ctx: WorldContext) -> np.ndarray:
        """渲染世界气温图：蓝→绿→红温度色卡，海洋叠加冷色罩。"""
        T = ctx.get("temperature.temp")
        if T is None:
            self.compute(ctx)
            T = ctx.get("temperature.temp")
        cmap = _temp_cmap()
        t_norm = np.clip((T + 40.0) / 85.0, 0.0, 1.0).astype(np.float32)
        rgb = cmap(t_norm)[..., :3].astype(np.float32)
        sea = ~ctx.land_mask
        if sea.any():
            sea_tint = np.array([0.06, 0.12, 0.24], dtype=np.float32)
            rgb[sea] = 0.55 * rgb[sea] + 0.45 * sea_tint
        return np.clip(rgb, 0, 1).astype(np.float32)


def _temp_cmap() -> LinearSegmentedColormap:
    """构造气温图用的蓝→绿→红 512 级线性色卡。"""
    stops = [
        (0.00, (0.02, 0.05, 0.30)),
        (0.20, (0.10, 0.35, 0.65)),
        (0.47, (0.25, 0.65, 0.55)),
        (0.65, (0.45, 0.72, 0.30)),
        (0.82, (0.92, 0.78, 0.20)),
        (1.00, (0.85, 0.18, 0.12)),
    ]
    return LinearSegmentedColormap.from_list("temperature", stops, N=512)
