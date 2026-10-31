# -*- coding: utf-8 -*-
"""hydro_module — 水系模块（流量累积 / 河流 / 湖泊 / 世界水系图）。

本模块持有水系相关的**全部物理实现**，可独立优化。
依赖：``terrain``（需要归一化高程 v01 与陆地掩膜）。

输出键：
    - ``hydro.flow_acc``       D8 流量累积场（像素数）
    - ``hydro.river_mask``     河道二值掩膜（陆地 & 流量 > 99.5 分位）
    - ``hydro.lake_mask``      湖泊掩膜（填洼面与实际地形差 > 0.012v）
    - ``hydro.rgb``            世界水系图 RGB（深蓝底 + 河流 + 湖泊）
"""
from __future__ import annotations

import numpy as np
from numba import njit
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           map_coordinates, label as ndi_label,
                           binary_dilation, binary_closing, binary_opening,
                           minimum_filter, convolve1d, find_objects as ndi_find_objects)

from ..core.terrain_engine import value_fbm
from .base import BaseModule, WorldContext

# Native 加速 (C++ fws_native 扩展; 不可用时回退 Numba)
try:
    from ..native import fws_native as _native
except ImportError:
    _native = None


# ===========================================================================
# 物理实现（原 terrain_engine.py 中水系相关函数）
# ===========================================================================

# D8 八方向邻域: (dy, dx, distance)
_D8 = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356), (0, -1, 1.0),
       (0, 1, 1.0), (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356)]


# ---------------------------------------------------------------------------
# Numba JIT 加速内核（4096² 地图优化, 2026-09）
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _accumulate_flow(acc, flat, order):
    """D8 流量累积串行传播内核 (单线程 JIT, 替代 Python for 循环)。

    acc  : 流量累积场 (1D, float32 或 float64), 原地累加
    flat : 每个像素的接收者索引 (1D, int32), -1 表示无接收者 (边界/洼地)
    order: 处理顺序, 按高程从高到低 (1D, int64), 由 np.argsort(-e) 给出
    """
    n = order.shape[0]
    for k in range(n):
        i = order[k]
        r = flat[i]
        if r >= 0:
            acc[r] += acc[i]
    return acc


@njit(cache=True)
def _trace_centerline(acc, labels, comp_i, h, w, y0, x0, max_steps,
                      visited_comp):
    """河流中心线 D8 贪心追踪内核 (单连通分量)。

    返回反转后的 path (n, 2) float32 数组, 每行 (y, x)。
    visited_comp: int32 (h, w), 记录每像素被哪个 comp_id 访问过, 0=未访问;
                  用 comp_id 标记避免每分量重新分配 HxW 数组。
    """
    path = np.empty((max_steps, 2), dtype=np.float32)
    cy, cx = y0, x0
    n = 0
    path[n, 0] = float(cy)
    path[n, 1] = float(cx)
    n += 1
    visited_comp[cy, cx] = comp_i
    for _ in range(max_steps - 1):
        best_a = -1.0
        best_y = -1
        best_x = -1
        # 八邻域 (含对角)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny = cy + dy
                nx = cx + dx
                if 0 <= ny < h and 0 <= nx < w \
                        and visited_comp[ny, nx] != comp_i \
                        and labels[ny, nx] == comp_i:
                    a = acc[ny, nx]
                    if a > best_a:
                        best_a = a
                        best_y = ny
                        best_x = nx
        if best_y < 0:
            break
        cy = best_y
        cx = best_x
        path[n, 0] = float(cy)
        path[n, 1] = float(cx)
        n += 1
        visited_comp[cy, cx] = comp_i
    # 反转 path (源在上游, 汇于下游)
    result = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        result[i, 0] = path[n - 1 - i, 0]
        result[i, 1] = path[n - 1 - i, 1]
    return result


@njit(cache=True)
def _meander_single_river(path, acc_at, q99, ph, amp, base_freq,
                           head_shrink, tail_shrink, len_fac,
                           scale_adaptive, min_turn_radius_px,
                           S_MIN, S_MAX, dtheta_max, clamp_iters,
                           land, has_land, octaves):
    """meander_centerlines 单条河完整处理 njit 版 (前置+主循环)。

    消除 8773 条河 × 20+ numpy 调用的 overhead。
    path: (n,2) float32 输入路径
    acc_at: (n,) float64 该路径上的流量 (由 Python 层 map_coordinates 算出)
    q99: float 全局 q99 (Python 层算一次, 所有河共享)
    ph: (octaves*5,) float64 随机相位 (Python 层生成)
    返回 (n,2) float32 蜿蜒后的路径
    """
    n_p = path.shape[0]
    if n_p < 6:
        return path.astype(np.float32)

    path_orig = path.astype(np.float64)

    # ---- 前置: dp, seg_len, s, t ----
    dp = np.empty((n_p - 1, 2), dtype=np.float64)
    seg_len = np.empty(n_p - 1, dtype=np.float64)
    for i in range(n_p - 1):
        dp[i, 0] = path_orig[i + 1, 0] - path_orig[i, 0]
        dp[i, 1] = path_orig[i + 1, 1] - path_orig[i, 1]
        seg_len[i] = np.sqrt(dp[i, 0] ** 2 + dp[i, 1] ** 2)
    s = np.empty(n_p, dtype=np.float64)
    s[0] = 0.0
    for i in range(1, n_p):
        s[i] = s[i - 1] + seg_len[i - 1]
    S = s[-1]
    if S < 1e-6:
        S = 1e-6
    t = s / S

    # ---- tang, norm_ ----
    tang = np.zeros((n_p, 2), dtype=np.float64)
    if seg_len[0] > 1e-6:
        tang[0, 0] = dp[0, 0] / seg_len[0]
        tang[0, 1] = dp[0, 1] / seg_len[0]
    if seg_len[-1] > 1e-6:
        tang[-1, 0] = dp[-1, 0] / seg_len[-1]
        tang[-1, 1] = dp[-1, 1] / seg_len[-1]
    for i in range(1, n_p - 1):
        denom = seg_len[i - 1] + seg_len[i] + 1e-6
        tang[i, 0] = (dp[i, 0] + dp[i - 1, 0]) / denom
        tang[i, 1] = (dp[i, 1] + dp[i - 1, 1]) / denom
    for i in range(n_p):
        tn = np.sqrt(tang[i, 0] ** 2 + tang[i, 1] ** 2)
        if tn > 1e-8:
            tang[i, 0] /= tn
            tang[i, 1] /= tn
    norm_ = np.empty((n_p, 2), dtype=np.float64)
    for i in range(n_p):
        norm_[i, 0] = -tang[i, 1]
        norm_[i, 1] = tang[i, 0]

    # ---- scale (尺度自适应) ----
    scale = np.ones(n_p, dtype=np.float64)
    if scale_adaptive:
        acc_max = 0.0
        for i in range(n_p):
            if acc_at[i] > acc_max:
                acc_max = acc_at[i]
        a99p = q99
        candidate = acc_max * 0.2
        if candidate > a99p:
            a99p = candidate
        if a99p < 10.0:
            a99p = 10.0
        # hanning 9-tap 核
        kern = np.empty(9, dtype=np.float64)
        for i in range(9):
            kern[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / 8.0))
        ksum = 0.0
        for i in range(9):
            ksum += kern[i]
        for i in range(9):
            kern[i] /= ksum
        # w_ratio = (acc_at/a99p)^0.5, hanning 平滑
        w_ratio = np.empty(n_p, dtype=np.float64)
        for i in range(n_p):
            a = acc_at[i] / a99p
            if a < 1e-4:
                a = 1e-4
            if a > 1.0:
                a = 1.0
            w_ratio[i] = np.sqrt(a)
        # convolve1d nearest
        w_sm = np.empty(n_p, dtype=np.float64)
        for i in range(n_p):
            s_sum = 0.0
            for k in range(9):
                j = i + k - 4
                if j < 0:
                    j = 0
                elif j >= n_p:
                    j = n_p - 1
                s_sum += kern[k] * w_ratio[j]
            w_sm[i] = s_sum
        for i in range(n_p):
            sc = 0.10 + 2.55 * w_sm[i]
            if sc < 0.10:
                sc = 0.10
            elif sc > 2.65:
                sc = 2.65
            scale[i] = sc
        for i in range(n_p):
            cw = np.sin(np.pi * t[i]) ** 0.45
            scale[i] = scale[i] * (0.35 + 0.65 * cw)

    # ---- off (多 octave 正弦叠加) ----
    off = np.zeros(n_p, dtype=np.float64)
    freq = base_freq * S
    for o in range(octaves):
        for k in range(5):
            p = ph[o * 5 + k]
            f = freq * (1.0 + 0.28 * k) * (2 ** o) * 2.0 * np.pi
            amp_o = 1.0 / (2 ** o)
            for i in range(n_p):
                off[i] += np.sin(t[i] * f + p) * amp_o
    off_max = 0.0
    for i in range(n_p):
        if abs(off[i]) > off_max:
            off_max = abs(off[i])
    if off_max < 1e-6:
        off_max = 1e-6
    for i in range(n_p):
        off[i] = off[i] / off_max * amp

    # ---- end_w, head_w, tail_w, amp_at, path_v1 ----
    end_w = np.empty(n_p, dtype=np.float64)
    head_w = np.empty(n_p, dtype=np.float64)
    tail_w = np.empty(n_p, dtype=np.float64)
    for i in range(n_p):
        end_w[i] = np.sin(np.pi * t[i]) ** 0.85
        if t[i] < head_shrink:
            hw = t[i] / head_shrink
            if hw < 0.0:
                hw = 0.0
            elif hw > 1.0:
                hw = 1.0
            head_w[i] = hw ** 1.3
        else:
            head_w[i] = 1.0
        if t[i] > 1.0 - tail_shrink:
            tw = (1.0 - t[i]) / tail_shrink
            if tw < 0.0:
                tw = 0.0
            elif tw > 1.0:
                tw = 1.0
            tail_w[i] = tw ** 1.4
        else:
            tail_w[i] = 1.0
    off_base = np.empty(n_p, dtype=np.float64)
    off_abs_max = 0.0
    for i in range(n_p):
        if abs(off[i]) > off_abs_max:
            off_abs_max = abs(off[i])
    if off_abs_max > 1e-9:
        for i in range(n_p):
            off_base[i] = off[i] / off_abs_max
    else:
        for i in range(n_p):
            off_base[i] = off[i]
    path_v1 = np.empty((n_p, 2), dtype=np.float64)
    for i in range(n_p):
        amp_at = amp * scale[i] * end_w[i] * head_w[i] * tail_w[i] * len_fac
        if amp_at < 0.0:
            amp_at = 0.0
        off_cur = off_base[i] * amp_at
        path_v1[i, 0] = path_orig[i, 0] + off_cur * norm_[i, 0]
        path_v1[i, 1] = path_orig[i, 1] + off_cur * norm_[i, 1]

    # ---- 主循环 (8×5 平滑 + sinuosity 调节 + 转弯钳制 + hanning + 陆地钳制) ----
    path_v1 = _meander_main_loop(path_v1, path_orig, scale, amp,
                                   min_turn_radius_px, S_MIN, S_MAX,
                                   dtheta_max, clamp_iters, land, has_land)
    return path_v1.astype(np.float32)


@njit(cache=True)
def _meander_main_loop(path_v1, path_orig, scale, amp, min_turn_radius_px,
                       S_MIN, S_MAX, dtheta_max, clamp_iters, land, has_land):
    """meander_centerlines 主循环 njit 版 (替代 8×5 平滑 + sinuosity 调节 + 转弯钳制的 Python+numpy 调用)。

    8773 条河 × 8 外循环 × 5 内循环 = 35 万次 numpy 调用的 overhead 被消除,
    全部在 njit 内用纯循环完成。
    path_v1: (n,2) float64 输入路径, 原地修改并返回
    path_orig: (n,2) float64 原始路径 (陆地钳制时回退用)
    scale: (n,) float64 尺度因子
    land: (H,W) bool (has_land=False 时可传任意数组, 不会被访问)
    has_land: bool 是否执行陆地钳制
    """
    n_p = path_v1.shape[0]
    if n_p < 6:
        return path_v1

    # 9-tap Hanning 核
    kern = np.empty(9, dtype=np.float64)
    for i in range(9):
        kern[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / 8.0))
    ksum = 0.0
    for i in range(9):
        ksum += kern[i]
    for i in range(9):
        kern[i] /= ksum

    dtheta_tight = 35.0 * (np.pi / 180.0)
    TWO_PI = 2.0 * np.pi

    # ---- 8 轮 sinuosity 调节, 每轮 5 次转弯钳制 ----
    for _loop in range(8):
        for _sm in range(clamp_iters):
            dp1 = np.empty((n_p - 1, 2), dtype=np.float64)
            sl1 = np.empty(n_p - 1, dtype=np.float64)
            for i in range(n_p - 1):
                dp1[i, 0] = path_v1[i + 1, 0] - path_v1[i, 0]
                dp1[i, 1] = path_v1[i + 1, 1] - path_v1[i, 1]
                sl1[i] = np.sqrt(dp1[i, 0] ** 2 + dp1[i, 1] ** 2)
            tg1 = np.zeros((n_p, 2), dtype=np.float64)
            if sl1[0] > 1e-6:
                tg1[0, 0] = dp1[0, 0] / sl1[0]
                tg1[0, 1] = dp1[0, 1] / sl1[0]
            if sl1[-1] > 1e-6:
                tg1[-1, 0] = dp1[-1, 0] / sl1[-1]
                tg1[-1, 1] = dp1[-1, 1] / sl1[-1]
            for i in range(1, n_p - 1):
                denom = sl1[i - 1] + sl1[i] + 1e-6
                tg1[i, 0] = (dp1[i, 0] + dp1[i - 1, 0]) / denom
                tg1[i, 1] = (dp1[i, 1] + dp1[i - 1, 1]) / denom
            for i in range(n_p):
                tn = np.sqrt(tg1[i, 0] ** 2 + tg1[i, 1] ** 2)
                if tn > 1e-8:
                    tg1[i, 0] /= tn
                    tg1[i, 1] /= tn
            # 转角
            dtheta = np.zeros(n_p, dtype=np.float64)
            for i in range(1, n_p - 1):
                a = np.arctan2(tg1[i - 1, 0], tg1[i - 1, 1])
                b = np.arctan2(tg1[i + 1, 0], tg1[i + 1, 1])
                d = b - a
                if d > np.pi:
                    d -= TWO_PI
                elif d < -np.pi:
                    d += TWO_PI
                dtheta[i] = abs(d)
            # smooth_mask
            any_smooth = False
            for i in range(1, n_p - 1):
                if dtheta[i] > dtheta_max:
                    any_smooth = True
                else:
                    ds = 0.5 * (sl1[i - 1] + sl1[i])
                    if dtheta[i] > 0.05:
                        R = ds / max(dtheta[i], 1e-8)
                        w_here = 1.5 * amp * scale[i] + min_turn_radius_px * 0.4
                        if R < max(min_turn_radius_px, w_here):
                            any_smooth = True
            if not any_smooth:
                break
            # 平滑 (原地)
            p_new = path_v1.copy()
            for i in range(1, n_p - 1):
                need_smooth = False
                if dtheta[i] > dtheta_max:
                    need_smooth = True
                else:
                    ds = 0.5 * (sl1[i - 1] + sl1[i])
                    if dtheta[i] > 0.05:
                        R = ds / max(dtheta[i], 1e-8)
                        w_here = 1.5 * amp * scale[i] + min_turn_radius_px * 0.4
                        if R < max(min_turn_radius_px, w_here):
                            need_smooth = True
                if need_smooth:
                    iL = i - 1
                    iR = i + 1
                    p_new[i, 0] = 0.25 * path_v1[iL, 0] + 0.50 * path_v1[i, 0] + 0.25 * path_v1[iR, 0]
                    p_new[i, 1] = 0.25 * path_v1[iL, 1] + 0.50 * path_v1[i, 1] + 0.25 * path_v1[iR, 1]
            path_v1 = p_new

        # sinuosity
        Sarc = 0.0
        for i in range(n_p - 1):
            dy = path_v1[i + 1, 0] - path_v1[i, 0]
            dx = path_v1[i + 1, 1] - path_v1[i, 1]
            Sarc += np.sqrt(dy * dy + dx * dx)
        chord_y = path_v1[-1, 0] - path_v1[0, 0]
        chord_x = path_v1[-1, 1] - path_v1[0, 1]
        chord_len = np.sqrt(chord_y * chord_y + chord_x * chord_x)
        if chord_len < 1e-6:
            chord_len = 1e-6
        S_curr = Sarc / chord_len
        if S_MIN <= S_curr <= S_MAX or Sarc < 1e-6:
            break

        # hanning 平滑 path_v1 → path_sm
        path_sm = np.empty((n_p, 2), dtype=np.float64)
        for dim in range(2):
            for i in range(n_p):
                s = 0.0
                for k in range(9):
                    j = i + k - 4
                    if j < 0:
                        j = 0
                    elif j >= n_p:
                        j = n_p - 1
                    s += kern[k] * path_v1[j, dim]
                path_sm[i, dim] = s
        # dp2, sl2, tang2, n2
        dp2 = np.empty((n_p - 1, 2), dtype=np.float64)
        sl2 = np.empty(n_p - 1, dtype=np.float64)
        for i in range(n_p - 1):
            dp2[i, 0] = path_v1[i + 1, 0] - path_v1[i, 0]
            dp2[i, 1] = path_v1[i + 1, 1] - path_v1[i, 1]
            sl2[i] = np.sqrt(dp2[i, 0] ** 2 + dp2[i, 1] ** 2)
            if sl2[i] < 1e-6:
                sl2[i] = 1e-6
        tang2 = np.zeros((n_p, 2), dtype=np.float64)
        tang2[0, 0] = dp2[0, 0] / sl2[0]
        tang2[0, 1] = dp2[0, 1] / sl2[0]
        tang2[-1, 0] = dp2[-1, 0] / sl2[-1]
        tang2[-1, 1] = dp2[-1, 1] / sl2[-1]
        for i in range(1, n_p - 1):
            denom = sl2[i - 1] + sl2[i] + 1e-6
            tang2[i, 0] = (dp2[i, 0] + dp2[i - 1, 0]) / denom
            tang2[i, 1] = (dp2[i, 1] + dp2[i - 1, 1]) / denom
        for i in range(n_p):
            tn = np.sqrt(tang2[i, 0] ** 2 + tang2[i, 1] ** 2)
            if tn > 1e-8:
                tang2[i, 0] /= tn
                tang2[i, 1] /= tn
        # n2 = [-tang2[:,1], tang2[:,0]]
        if S_curr > S_MAX:
            factor = 0.86
        else:
            factor = 1.18
        for i in range(n_p):
            diff_y = path_v1[i, 0] - path_sm[i, 0]
            diff_x = path_v1[i, 1] - path_sm[i, 1]
            n2_y = -tang2[i, 1]
            n2_x = tang2[i, 0]
            proj = diff_y * n2_y + diff_x * n2_x
            new_off = proj * (factor - 1.0)
            path_v1[i, 0] = path_v1[i, 0] + new_off * n2_y
            path_v1[i, 1] = path_v1[i, 1] + new_off * n2_x

    # ---- 3 次紧转弯钳制 ----
    for _h in range(3):
        dp1 = np.empty((n_p - 1, 2), dtype=np.float64)
        sl1 = np.empty(n_p - 1, dtype=np.float64)
        for i in range(n_p - 1):
            dp1[i, 0] = path_v1[i + 1, 0] - path_v1[i, 0]
            dp1[i, 1] = path_v1[i + 1, 1] - path_v1[i, 1]
            sl1[i] = np.sqrt(dp1[i, 0] ** 2 + dp1[i, 1] ** 2)
        tg1 = np.zeros((n_p, 2), dtype=np.float64)
        if n_p >= 3 and sl1[0] > 1e-8:
            tg1[0, 0] = dp1[0, 0] / sl1[0]
            tg1[0, 1] = dp1[0, 1] / sl1[0]
        if n_p >= 3 and sl1[-1] > 1e-8:
            tg1[-1, 0] = dp1[-1, 0] / sl1[-1]
            tg1[-1, 1] = dp1[-1, 1] / sl1[-1]
        if n_p >= 4:
            for i in range(1, n_p - 1):
                denom = sl1[i - 1] + sl1[i]
                if denom < 1e-6:
                    denom = 1e-6
                tg1[i, 0] = (dp1[i, 0] + dp1[i - 1, 0]) / denom
                tg1[i, 1] = (dp1[i, 1] + dp1[i - 1, 1]) / denom
        for i in range(n_p):
            tn = np.sqrt(tg1[i, 0] ** 2 + tg1[i, 1] ** 2)
            if tn > 1e-8:
                tg1[i, 0] /= tn
                tg1[i, 1] /= tn
        bad = False
        for i in range(1, n_p - 1):
            a = np.arctan2(tg1[i - 1, 0], tg1[i - 1, 1])
            b = np.arctan2(tg1[i + 1, 0], tg1[i + 1, 1])
            d = b - a
            if d > np.pi:
                d -= TWO_PI
            elif d < -np.pi:
                d += TWO_PI
            if abs(d) > dtheta_tight:
                bad = True
                break
        if not bad:
            break
        for i in range(1, n_p - 1):
            a = np.arctan2(tg1[i - 1, 0], tg1[i - 1, 1])
            b = np.arctan2(tg1[i + 1, 0], tg1[i + 1, 1])
            d = b - a
            if d > np.pi:
                d -= TWO_PI
            elif d < -np.pi:
                d += TWO_PI
            if abs(d) > dtheta_tight:
                iL = i - 1
                iR = i + 1
                path_v1[i, 0] = 0.5 * path_v1[iL, 0] + 0.5 * path_v1[iR, 0]
                path_v1[i, 1] = 0.5 * path_v1[iL, 1] + 0.5 * path_v1[iR, 1]

    # ---- 最终 hanning 平滑 ----
    if n_p >= 9:
        for dim in range(2):
            tmp = path_v1[:, dim].copy()
            for i in range(n_p):
                s = 0.0
                for k in range(9):
                    j = i + k - 4
                    if j < 0:
                        j = 0
                    elif j >= n_p:
                        j = n_p - 1
                    s += kern[k] * tmp[j]
                path_v1[i, dim] = s

    # ---- 陆地钳制 ----
    if has_land:
        H_L = land.shape[0]
        W_L = land.shape[1]
        for i in range(n_p):
            ir = int(round(path_v1[i, 0]))
            ic = int(round(path_v1[i, 1]))
            if ir < 0:
                ir = 0
            elif ir >= H_L:
                ir = H_L - 1
            if ic < 0:
                ic = 0
            elif ic >= W_L:
                ic = W_L - 1
            if not land[ir, ic]:
                path_v1[i, 0] = path_v1[i, 0] * 0.5 + path_orig[i, 0] * 0.5
                path_v1[i, 1] = path_v1[i, 1] * 0.5 + path_orig[i, 1] * 0.5

    # clip 负值
    for i in range(n_p):
        if path_v1[i, 0] < 0.0:
            path_v1[i, 0] = 0.0
        if path_v1[i, 1] < 0.0:
            path_v1[i, 1] = 0.0

    return path_v1


def d8_flow_accumulation(elev, rng):
    """D8（八方向）流向算法计算流量累积.

    v9.x 优化: 改用 Kahn 拓扑排序单遍累积, 消除 O(N log N) argsort.
    原 argsort 在 4096² 上 ~2s, Kahn 为 O(N) ~0.3s.
    """
    h, w = elev.shape
    e = (elev.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)).ravel()
    acc = np.ones(h * w, dtype=np.float64)
    if _native is not None:
        acc = _native.flow_acc_kahn(e, acc, h, w)      # v10: C++
    else:
        acc = _flow_acc_kahn_njit(e, acc, h, w)        # v9.x: Kahn 拓扑排序
    return acc.reshape(h, w)


@njit(cache=True, fastmath=True)
def _mfd_acc_njit(e, source, order, h, w, p, w_min):
    """多流向 (MFD, Freeman 1991) 流量累积 — 按高程降序单遍分配.

    与 D8 的差别: 每个格点把水量按 ``(局部坡度)^p`` 分摊给**所有**更低的 8 邻域,
    而不是全部塞给最陡的那一个. 结果是: 山坡上形成"辫状/散开的收集面",
    汇流后才收敛成干流 —— 视觉上河网由细密的毛细支流逐级汇聚,
    不会出现 D8 在倾斜面上那种等间距平行直线.

    Parameters
    ----------
    e : (N,) float32
        加了微扰的高程 (1D).
    source : (N,) float64
        每格水源 (降水×融雪门控), 会被原地修改为累积值.
    order : (N,) int64
        按高程**降序**的格点索引 (上游必在下游之后处理).
    p : float
        坡度指数 (1.0=线性, 1.2 更集中).
    w_min : float
        分摊权重下限 (小于此值的下坡分量丢弃, 抑制数值弥散).
    """
    n = h * w
    acc = source.copy()
    sl = np.empty(8, np.float64)
    idx = np.empty(8, np.int64)
    for oi in range(n):
        i = order[oi]
        y = i // w
        x = i - y * w
        ei = e[i]
        k = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                j = yy * w + xx
                dist = 1.4142135623730951 if (dy != 0 and dx != 0) else 1.0
                s = (ei - e[j]) / dist
                if s > 0.0:
                    sl[k] = s ** p
                    idx[k] = j
                    k += 1
        if k == 0:
            continue
        tot = 0.0
        for m in range(k):
            tot += sl[m]
        if tot <= 0.0:
            continue
        a = acc[i]
        for m in range(k):
            wgt = sl[m] / tot
            if wgt >= w_min:
                acc[idx[m]] += a * wgt
    return acc


def mfd_flow_accumulation(elev, rng, p=1.15, source=None, w_min=0.02):
    """多流向流量累积入口 (p=1.15 接近地球实测的汇流集中度)."""
    h, w = elev.shape
    e = (elev.astype(np.float32)
         + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)).ravel()
    order = np.argsort(-e, kind="stable").astype(np.int64)
    if source is None:
        src = np.ones(h * w, np.float64)
    else:
        src = np.ascontiguousarray(source, np.float64).ravel()
    acc = _mfd_acc_njit(e, src, order, h, w, float(p), float(w_min))
    return acc.reshape(h, w).astype(np.float32)


def climate_flow_accumulation(elev, precip, temp, rng, snow_release_t=0.0,
                              p_ref=1500.0, base_floor=0.05, mfd=False):
    """气候加权流量累积：以降水 P 为水源权重，温度 T 门控积雪累积/融雪释放.

    v9.x 优化: 同 d8_flow_accumulation, 改用 Kahn 拓扑排序, 消除 argsort.
    v39: ``mfd=True`` 时改用多流向累积 (支流在坡面上散开、汇流后收敛).
    """
    h, w = elev.shape
    P = np.clip(np.asarray(precip, np.float64) / float(p_ref), 0.0, 1.5)
    T = np.asarray(temp, np.float64)
    release = np.clip((T - snow_release_t) / 5.0, 0.0, 1.0)
    source = (P * release).astype(np.float32)
    source = np.maximum(source, float(base_floor)).astype(np.float32)
    if mfd:
        return mfd_flow_accumulation(elev, rng, source=source)
    e = (elev.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)).ravel()
    acc = source.ravel().astype(np.float64).copy()
    if _native is not None:
        acc = _native.flow_acc_kahn(e, acc, h, w)      # v10: C++
    else:
        acc = _flow_acc_kahn_njit(e, acc, h, w)        # v9.x: Kahn 拓扑排序
    return acc.reshape(h, w).astype(np.float32)


@njit(cache=True)
def _flow_acc_kahn_njit(e, source, h, w):
    """D8 流量累积 (Kahn 拓扑排序, 单遍 O(N)).

    1. 计算 D8 接收者 (最陡下坡邻居) - 内联 8 邻域搜索
    2. 计算每个格点的入度 (有多少上游贡献者指向它)
    3. 初始化队列: 入度为 0 的格点 (局部高点)
    4. Kahn 处理: 弹出 i, 累积到 recv[i], 入度 -1, 入度=0 时入队

    输入:
        e      : (N,) float32, 已加噪声扰动的高程 (1D 视图)
        source : (N,) float64, 每个格点的水源值 (会被原地修改为累积值)
        h, w   : 形状
    返回:
        acc    : (N,) float64, 累积流量 (原地修改 source)
    """
    n = h * w

    # ---- 1. D8 接收者 ----
    recv = np.full(n, -1, dtype=np.int32)
    best_s = np.full(n, 1e-9, dtype=np.float32)

    for i in range(n):
        y = i // w
        x = i - y * w
        ei = e[i]
        # 8 邻域, 对角距离 sqrt(2), 直邻距离 1.0
        # (dy, dx): 顺序不重要, 后更新的会覆盖前值 (取最大坡度)
        # 对角
        for dy in (-1, 1):
            for dx in (-1, 1):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    j = ny * w + nx
                    s = (ei - e[j]) * 0.70710678  # 1/1.41421356
                    if s > best_s[i]:
                        best_s[i] = s
                        recv[i] = j
        # 直邻
        for dy in (-1, 1):
            for dx in (-1, 1):
                # dy, dx 其中一个为 0
                ny1 = y + dy
                nx1 = x
                if 0 <= ny1 < h:
                    j1 = ny1 * w + x
                    s1 = ei - e[j1]
                    if s1 > best_s[i]:
                        best_s[i] = s1
                        recv[i] = j1
                ny2 = y
                nx2 = x + dx
                if 0 <= nx2 < w:
                    j2 = y * w + nx2
                    s2 = ei - e[j2]
                    if s2 > best_s[i]:
                        best_s[i] = s2
                        recv[i] = j2

    # ---- 2. 入度 ----
    in_degree = np.zeros(n, dtype=np.int32)
    for i in range(n):
        r = recv[i]
        if r >= 0:
            in_degree[r] += 1

    # ---- 3. Kahn 拓扑排序累积 ----
    acc = source  # 原地修改
    queue = np.empty(n, dtype=np.int64)
    head = 0
    tail = 0

    # 初始化: 所有入度为 0 的格点入队 (局部高点 / 边界外流)
    for i in range(n):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    while head < tail:
        i = queue[head]
        head += 1
        r = recv[i]
        if r >= 0:
            acc[r] += acc[i]
            in_degree[r] -= 1
            if in_degree[r] == 0:
                queue[tail] = r
                tail += 1

    return acc


def warp_river_corridor(ch, rng, amp=3.0, base_freq=10.0, octaves=3):
    """低频 fBm 方向场对主河道走廊施加横向曲率扰动，打破 D8 单像素直线路径。"""
    h, w = ch.shape
    size = max(h, w)
    wy = value_fbm(size, base_freq, octaves, rng=rng)[:h, :w]
    wx = value_fbm(size, base_freq, octaves, rng=rng)[:h, :w]
    wy = wy / (np.std(wy) + 1e-8) * float(amp)
    wx = wx / (np.std(wx) + 1e-8) * float(amp)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ch_w = map_coordinates(ch.astype(np.float32), [yy + wy, xx + wx],
                           order=1, mode='nearest')
    return ch_w > 0.5


def extract_river_centerlines(acc, land, q=0.999, min_len=12,
                              monotonic_repair=True):
    """v8.3: 骨架提取 + 沿程流量单调修复.

    算法层优化 (2026-09): 用 ndi.find_objects 一次性获取所有连通分量 bbox,
    在 bbox 内做局部 where/argmax, 替代每分量全图扫描.
    复杂度从 O(n × H × W) 降到 O(sum(bbox_area)).
    """
    h, w = acc.shape
    aL = acc[land]
    if aL.size == 0:
        return []
    thr = float(np.quantile(aL, q))
    ch = (land & (acc > thr))
    labels, n = ndi_label(ch)
    if n == 0:
        return []
    lines = []
    # Numba 追踪内核用 comp_id 标记 visited, 避免每分量重置 HxW 数组
    visited_comp = np.zeros((h, w), dtype=np.int32)
    # 一次性获取所有连通分量 bbox: slices[i] = (yslice, xslice) for comp_i=i+1
    comp_slices = ndi_find_objects(labels)
    for comp_i in range(1, n + 1):
        sl = comp_slices[comp_i - 1]
        if sl is None:
            continue
        ysl, xsl = sl
        # 局部 labels 子区域 (bbox 内)
        labels_sub = labels[ysl, xsl]
        acc_sub = acc[ysl, xsl]
        local_mask = labels_sub == comp_i
        if int(local_mask.sum()) < min_len:
            continue
        # 在 bbox 内找最大 acc 像素 (局部索引)
        acc_masked = np.where(local_mask, acc_sub, -1.0)
        idx_local = int(np.argmax(acc_masked))
        ly, lx = divmod(idx_local, xsl.stop - xsl.start)
        # 转全局坐标
        y0 = int(ysl.start) + ly
        x0 = int(xsl.start) + lx
        safety = int(local_mask.sum()) * 2 + 32
        # v10: C++ 贪心追踪优先, 回退 Numba (替代 Python while + set)
        if _native is not None:
            pp = _native.trace_centerline(acc, labels, comp_i, h, w,
                                          y0, x0, safety, visited_comp)
        else:
            pp = _trace_centerline(acc, labels, comp_i, h, w,
                                   y0, x0, safety, visited_comp)
        if len(pp) < min_len:
            continue
        if monotonic_repair and len(pp) >= 4:
            if _native is not None:
                # v10: C++ 单调修复 + 去重 (替代 4 遍 Python+scipy 循环)
                pp = _native.monotonic_repair(acc, pp, land, min_len, h, w)
            else:
                from scipy.ndimage import map_coordinates as _mc
                NEI_R1 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                          (1, -1), (1, 0), (1, 1)]
                NEI_R2 = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0: continue
                        NEI_R2.append((dy, dx))
                NEI_R3 = []
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dy == 0 and dx == 0: continue
                        NEI_R3.append((dy, dx))
                for _pass in range(4):
                    acc_now = _mc(acc, pp.T, order=1, mode='nearest')
                    changed = False
                    for i in range(len(pp) - 1):
                        if acc_now[i + 1] >= acc_now[i] * 0.95:
                            continue
                        cy2, cx2 = int(round(float(pp[i, 0]))), int(round(float(pp[i, 1])))
                        ba, by, bx = -1.0, -1, -1
                        if _pass <= 1:
                            nei_list = NEI_R1
                        elif _pass == 2:
                            nei_list = NEI_R2
                        else:
                            nei_list = NEI_R3
                        for dy, dx in nei_list:
                            ny, nx = cy2 + dy, cx2 + dx
                            if 0 <= ny < h and 0 <= nx < w and land[ny, nx] \
                                    and acc[ny, nx] > ba:
                                ba, by, bx = float(acc[ny, nx]), ny, nx
                        if ba >= acc_now[i] * 0.78 and by >= 0:
                            pp[i + 1, 0] = float(by) + 0.5
                            pp[i + 1, 1] = float(bx) + 0.5
                            changed = True
                    keep = np.ones(len(pp), dtype=bool)
                    for i in range(2, len(pp)):
                        if (np.hypot(pp[i, 0] - pp[i - 1, 0],
                                     pp[i, 1] - pp[i - 1, 1]) < 0.12):
                            keep[i] = False
                    if keep.sum() < min_len:
                        break
                    pp = pp[keep]
                    if not changed:
                        break
                if len(pp) >= 4:
                    keep = np.ones(len(pp), dtype=bool)
                    for i in range(2, len(pp)):
                        if (np.hypot(pp[i, 0] - pp[i - 1, 0],
                                     pp[i, 1] - pp[i - 1, 1]) < 0.12):
                            keep[i] = False
                    if keep.sum() >= min_len:
                        pp = pp[keep]
        if len(pp) >= min_len:
            lines.append(pp)
    return lines


def meander_centerlines(lines, rng, amp=3.0, base_freq=0.015,
                        head_shrink=0.08, tail_shrink=0.06,
                        octaves=3, acc=None, width_ref=None,
                        scale_adaptive=True, max_turn_deg=22.0,
                        min_turn_radius_px=10.0, land=None,
                        clamp_iters=5, sinuosity_bounds=(1.8, 3.0)):
    """v8.3 升级：尺度自适应蜿蜒 + 5轮转弯钳制 + Hack定律摆幅耦合 + Sinuosity闭环。"""
    # 预计算所有河共享的 q99 (避免每条河都调 np.quantile)
    q99 = 1e6
    if scale_adaptive and acc is not None and land is not None and land.any():
        q99 = float(np.quantile(acc[land], 0.99))
    elif scale_adaptive and acc is not None:
        q99 = float(np.quantile(acc.ravel(), 0.99))

    out_lines = []
    has_land = land is not None
    land_arr = land if has_land else np.zeros((1, 1), dtype=bool)
    from scipy.ndimage import map_coordinates as _mc
    for path in lines:
        n_p = len(path)
        if n_p < 6:
            out_lines.append(path.copy())
            continue
        len_fac = float(np.clip((n_p - 30) / 90.0, 0.02, 1.0))
        # Python 层只做 map_coordinates (0.09s) 和生成 ph, 其余全在 njit
        acc_at = np.zeros(n_p, np.float64)
        if scale_adaptive and acc is not None:
            acc_at = _mc(acc, path.T, order=1, mode='nearest').astype(np.float64)
            acc_at = np.maximum(acc_at, 1.0)
        ph = rng.uniform(0, 2 * np.pi, octaves * 5).astype(np.float64)
        S_MIN, S_MAX = float(sinuosity_bounds[0]), float(sinuosity_bounds[1])
        if n_p < 30:
            S_MIN = 1.0; S_MAX = 3.2
        elif n_p < 60:
            S_MIN = 1.3; S_MAX = 3.1
        dtheta_max = float(max_turn_deg) * (np.pi / 180.0)
        # v10: C++ meander 优先 (消除 Python 层 8773 条河的 numpy overhead), 回退 Numba
        if _native is not None:
            result = _native.meander_single_river(
                path.astype(np.float32), acc_at, q99, ph,
                float(amp), float(base_freq),
                float(head_shrink), float(tail_shrink), len_fac,
                bool(scale_adaptive and acc is not None),
                float(min_turn_radius_px), S_MIN, S_MAX, dtheta_max,
                int(clamp_iters), land_arr, has_land, int(octaves))
        else:
            result = _meander_single_river(
                path.astype(np.float32), acc_at, q99, ph,
                float(amp), float(base_freq),
                float(head_shrink), float(tail_shrink), len_fac,
                bool(scale_adaptive and acc is not None),
                float(min_turn_radius_px), S_MIN, S_MAX, dtheta_max,
                int(clamp_iters), land_arr, has_land, int(octaves))
        out_lines.append(result)
    return out_lines


def carve_along_centerlines(v01, lines, acc, land, max_depth=0.045,
                            base_width=3.0, width_per_log_acc=0.4,
                            cutoff_cross=4.0, hack_law=True,
                            oxbow=True, oxbow_thresh_px=4,
                            oxbow_depth_factor=0.6):
    """v8.3 升级：Hack幂律宽/深 + 牛轭湖截断 + 逐级支流雕刻入口。"""
    H, W = v01.shape
    land_flat = land.ravel()
    v_flat = v01.astype(np.float64).ravel().copy()
    a99 = 0.0
    try:
        a99 = float(np.quantile(acc[land], 0.995))
    except Exception:
        a99 = 1000.0
    A_REF = max(a99, 10.0)
    dig = np.zeros_like(v_flat)

    for pth in lines:
        n_p = len(pth)
        if n_p < 3:
            continue
        pp = pth.astype(np.float64)
        dp = np.diff(pp, axis=0)
        seg_len = np.sqrt((dp * dp).sum(axis=1))
        s = np.concatenate([[0], np.cumsum(seg_len)])
        S = max(s[-1], 1e-6)
        t = s / S
        tang = np.zeros_like(pp)
        tang[0] = dp[0] / max(seg_len[0], 1e-6)
        tang[-1] = dp[-1] / max(seg_len[-1], 1e-6)
        tang[1:-1] = (dp[1:] + dp[:-1]) / (
            seg_len[1:] + seg_len[:-1] + 1e-6)[:, None]
        tn = np.linalg.norm(tang, axis=1, keepdims=True)
        tang = tang / np.maximum(tn, 1e-8)
        norm_ = np.concatenate([-tang[:, 1:2], tang[:, 0:1]], axis=1)

        pp_T = pp.T
        A_pix = np.maximum(map_coordinates(acc, pp_T, order=1, mode='nearest'), 1.0)
        A_star = np.clip(A_pix / A_REF, 1e-3, 10.0)
        if hack_law:
            width_h = base_width * np.power(A_star, 0.50)
            depth_h = max_depth * np.clip(0.18 + 0.82 * np.power(A_star, 0.40),
                                          0.02, 1.30)
            slope_mult = np.clip(np.power(A_star, -0.30), 0.72, 1.25)
            depth_h = depth_h * slope_mult
        else:
            acc_log = np.log(A_pix)
            ref_acc_log = float(np.log(A_REF))
            width_h = base_width + width_per_log_acc * acc_log
            depth_h = max_depth * np.clip(acc_log / max(ref_acc_log,1e-6), 0.10, 1.05)
        width_h = np.clip(width_h, max(base_width*0.45, 0.8), 42.0)

        head_wd = np.clip(t / 0.045, 0.0, 1.0) ** 1.5
        head_dp = np.clip(t / 0.035, 0.0, 1.0) ** 1.4
        d_tail = np.clip((t - 0.76) / 0.16, 0, 1)
        tail_wd = 1.0 + 0.35 * np.sin(np.pi * np.clip(d_tail, 0, 1))
        tail_dp = 1.0 + 0.28 * np.clip((t - 0.80) / 0.20, 0, 1)
        width_h = width_h * (0.38 + 0.62 * head_wd) * tail_wd
        depth_h = depth_h * (0.26 + 0.74 * head_dp) * tail_dp
        depth_h = np.clip(depth_h, max_depth * 0.02, max_depth * 1.28)
        width_h = np.clip(width_h, 0.8, 48.0)

        bend = 0.52 * np.sin(s * 0.017 + pp[:, 0] * 0.012 + pp[:, 1] * 0.009)

        if _native is not None:
            # v10: C++ 主雕刻循环 (替代 Python for i + np.maximum.at)
            _native.carve_path_dig(dig, pp, width_h, depth_h, bend, norm_,
                                   H, W, cutoff_cross)
        else:
            cs_step = 0.3
            cs_u = np.arange(-cutoff_cross, cutoff_cross + cs_step * 0.5, cs_step,
                             dtype=np.float64)
            for i in range(0, n_p, 1):
                cy, cx = pp[i, 0], pp[i, 1]
                if not (0.5 <= cy <= H - 1.5 and 0.5 <= cx <= W - 1.5):
                    continue
                w_i = float(width_h[i])
                d_i = float(depth_h[i])
                b_i = float(bend[i])
                ny, nx = float(norm_[i, 0]), float(norm_[i, 1])
                cs_pos = cs_u * w_i
                py = cy + cs_pos * ny
                px = cx + cs_pos * nx
                i0 = np.clip(np.floor(py).astype(np.int32), 0, H - 2)
                j0 = np.clip(np.floor(px).astype(np.int32), 0, W - 2)
                fy, fx = py - i0, px - j0
                base_arr = (i0 * W + j0).astype(np.int64)
                profile_v = np.abs(cs_u) ** 1.35
                trough = np.exp(-((cs_u - b_i) ** 2) / 0.16) * 0.60
                env = np.exp(-(cs_u ** 2) / (2 * 1.05 ** 2))
                rel = np.clip(profile_v - trough, 0.0, None)
                rel = rel / max(rel.max(), 1e-8)
                dig_i = d_i * (1.0 - rel) * env
                wq = dig_i.astype(np.float64)
                omfy, omfx = 1.0 - fy, 1.0 - fx
                np.maximum.at(dig, base_arr,         omfy * omfx * wq)
                np.maximum.at(dig, base_arr + 1,     omfy * fx     * wq)
                np.maximum.at(dig, base_arr + W,     fy   * omfx * wq)
                np.maximum.at(dig, base_arr + W+1,   fy   * fx     * wq)

    if oxbow:
        if _native is not None:
            # v10: C++ 牛轭湖 (替代 Python 双循环 + meshgrid)
            for pth in lines:
                pp = pth.astype(np.float64)
                _native.carve_oxbow(dig, pp, H, W, land,
                                    max_depth, oxbow_thresh_px, oxbow_depth_factor)
        else:
            for pth in lines:
                n_p = len(pth)
                if n_p < 40: continue
                pp = pth.astype(np.float64)
                step = 3
                idxs = list(range(0, n_p, step))
                for a in range(len(idxs)):
                    i = idxs[a]
                    for b in range(a+6, min(a+30, len(idxs))):
                        j = idxs[b]
                        if j - i < 18: continue
                        d_eu = float(np.hypot(pp[j,0]-pp[i,0], pp[j,1]-pp[i,1]))
                        if d_eu > float(oxbow_thresh_px): continue
                        cy, cx = 0.5*(pp[i,0]+pp[j,0]), 0.5*(pp[i,1]+pp[j,1])
                        ry, rx = max(d_eu*1.1, 3.0), max(d_eu*0.6, 1.8)
                        d_oxb = float(max_depth) * float(oxbow_depth_factor)
                        rr = np.arange(int(cy-max(ry,3)-1), int(cy+max(ry,3)+2))
                        cc = np.arange(int(cx-max(rx,3)-1), int(cx+max(rx,3)+2))
                        rrg, ccg = np.meshgrid(rr, cc, indexing='ij')
                        mask_valid = (rrg>=0) & (rrg<H) & (ccg>=0) & (ccg<W)
                        if not mask_valid.any(): continue
                        ell = ((rrg-cy)/max(ry,1e-3))**2 + ((ccg-cx)/max(rx,1e-3))**2
                        fill = np.exp(-ell * 1.8) * d_oxb
                        # 边界掩膜必须参与选点: 椭圆中心贴边时 rrg/ccg 会越界,
                        #   负数索引在 NumPy 里会静默回卷到"错误的像素"(行首/行尾),
                        #   y 越下界则 idx >= H*W 直接 IndexError.
                        mv = mask_valid.ravel()
                        idx_all = np.where(mv, (rrg*W + ccg).ravel(), 0)
                        f_all = fill.ravel().astype(np.float64)
                        sel = mv & land_flat[idx_all] & (f_all > 1e-5)
                        if sel.any():
                            np.maximum.at(dig, idx_all[sel], f_all[sel])

    keep = land_flat & (dig > 0)
    v_flat[keep] = v_flat[keep] - dig[keep]
    return np.clip(v_flat.reshape(H, W), 0, 1).astype(np.float32)


def paint_rivers_rgb(rgb, acc, land, rng, base_q=0.995,
                     color_riv=np.array([0.10, 0.26, 0.42], np.float32),
                     color_big=np.array([0.08, 0.22, 0.38], np.float32),
                     oxbow_color=np.array([0.07, 0.20, 0.33], np.float32),
                     confluence_bleed_px=2.2, oxbow=True, deltas=True,
                     meander=False, meander_amp=2.0, meander_wl=160.0,
                     dryness=None, relief_m=None):
    """v8.3 河流连续蓝度+距离场羽化+交汇色溢+牛轭湖淡染。

    v39 河道几何修正:
        - ``dryness`` (0~1 干燥度): 干旱区河网稀疏(门限抬高 + 透明度衰减),
          解决"平原无河、山区密布"的密度反转.
        - ``relief_m``: 山区河道收窄 (真实山地河谷极窄, 而非宽面),
          解决"河流横切山脊"的观感.
        - 默认关闭全图正弦位移蜿蜒 (该做法会把整层河网刚性扭曲、断成裂纹),
          蜿蜒改由 DEM 河谷雕刻承担.
    """
    H, W = rgb.shape[:2]
    aL = acc[land]
    if aL.size == 0:
        return rgb
    lo = float(np.quantile(aL, base_q))
    A_REF = max(float(np.quantile(aL, min(base_q + 0.004, 0.9999))),
                lo * 1.05, 10.0)
    lo = max(lo, 2.0)

    A_arr = np.maximum(acc, 1.0).astype(np.float64)
    A_star = np.clip(A_arr / A_REF, 1e-3, 10.0)
    s = np.clip(np.log(A_arr) / max(np.log(A_REF), 1e-6), 0.0, 1.0)

    mask_core = land & (acc >= lo)
    if dryness is not None:
        # 干旱区门限抬高 (需更大汇水面积才成河) → 沙漠河网稀疏, 雨林密集
        _dry = np.clip(dryness, 0.0, 1.0)
        mask_core = land & (acc >= lo * (1.0 + 1.6 * _dry))
    else:
        _dry = np.zeros((H, W), np.float32)
    if not mask_core.any():
        return rgb
    mask_core_wide = binary_dilation(mask_core, iterations=1) & land
    D_core = distance_transform_edt(~mask_core_wide)
    # v53: 峰宽再放一档 (1.05→1.15 基宽, 4.2→4.6 增益), 支流从 2px 提到 3px.
    #   全图尺度下这是"河网能不能被看见"的临界值; 干流仍受 A_star 上限约束, 不会失控.
    W_half = 1.15 + 4.6 * np.power(A_star, 0.55)
    if relief_m is not None:
        # 山区河道收窄: 海拔 1200m 以上宽度线性压到 72% (v48: 45%→65%, v53: 65%→72%,
        #   让山地里的河谷也能读出走向)
        _mtn = np.clip((relief_m - 1200.0) / 2600.0, 0.0, 1.0).astype(np.float32)
        W_half = W_half * (1.0 - 0.28 * _mtn)

    ratio = np.clip(D_core / np.maximum(W_half, 1e-4), 0.0, 2.2)
    river_core = (1.0 - np.clip(ratio, 0, 1) ** 1.85)
    # v9.x: edge_ref = ~(mask_core_wide & land) = ~mask_core_wide (因 mask_core_wide 已在 land 内),
    # 所以 D_to_river_edge ≡ D_core. 消除冗余 distance_transform_edt (~1s on 4096²).
    edge_band = np.clip(1.0 - D_core / max(3.0, 1e-6), 0.0, 1.0) ** 1.6
    alpha = np.clip(river_core * (0.6 + 0.4 * edge_band), 0.0, 1.0) * land
    if dryness is not None:
        alpha = alpha * (1.0 - 0.28 * _dry)

    col_s = s[..., None]
    target = (1.0 - col_s) * color_riv.astype(np.float64) + \
        col_s * color_big.astype(np.float64)
    blend_lv = (0.34 + 0.52 * s)
    b_final = (blend_lv * alpha)[..., None]

    s_nb_min = minimum_filter(s.astype(np.float32), size=3)
    diff = s - s_nb_min
    conf_mask = (alpha > 0.25) & (diff >= 0.22) & land
    if conf_mask.any():
        # v9.x: 交汇距离场在 conf_mask 的 bbox 内计算 (稀疏交汇点, bbox ≪ 全图)
        ys_c, xs_c = np.nonzero(conf_mask)
        pad = 10  # bleed 尺度 ~2.2px, 10px padding 足够精确
        y0, y1 = max(0, int(ys_c.min()) - pad), min(H, int(ys_c.max()) + pad + 1)
        x0, x1 = max(0, int(xs_c.min()) - pad), min(W, int(xs_c.max()) + pad + 1)
        D_conf = np.full((H, W), 1e6, dtype=np.float64)  # 远离交汇处 bleed→0
        D_conf[y0:y1, x0:x1] = distance_transform_edt(~conf_mask[y0:y1, x0:x1])
        bleed = np.exp(-(D_conf / max(confluence_bleed_px, 0.5)) ** 2)
        bleed_blend = 0.18 * bleed
        b_final[:, :, 0] = b_final[:, :, 0] * (1.0 + bleed_blend)
        G_boost = (0.020 * bleed)[..., None]
    else:
        G_boost = np.zeros((H, W, 1), dtype=np.float64)

    alpha_px = (b_final > 1e-4)[:, :, 0]
    if alpha_px.any():
        idx = np.where(alpha_px)
        c_add = target[idx] + G_boost[idx].reshape(-1, 1) * np.array([0.0,1.0,0.8],
                                                                       dtype=np.float64)
        c_add = np.clip(c_add, 0.0, 1.0)
        bb = b_final[idx[0], idx[1]]
        rgb[idx] = rgb[idx] * (1.0 - bb) + c_add * bb

    blend_map = np.clip(b_final[..., 0], 0.0, 1.0)
    feather_mask = (blend_map > 0.01) & (blend_map < 0.92)
    if feather_mask.any():
        feather_wide = binary_dilation(feather_mask, iterations=3) & land
        if feather_wide.any():
            # v9.x: 单次 (H, W, 3) gaussian_filter 替代 3 次 for 循环
            # (sigma=[1.6, 1.6, 0] - 仅空间轴模糊, 通道独立处理)
            rgb_s = gaussian_filter(rgb.astype(np.float32),
                                    sigma=[1.6, 1.6, 0.0]).astype(rgb.dtype)
            f = feather_wide.astype(np.float32) * np.clip(0.25 + 0.75 * blend_map, 0.25, 1.0)
            f3 = f[..., None]
            idx_f = np.where(feather_wide)
            rgb[idx_f] = (rgb[idx_f] * (1.0 - f3[idx_f]) +
                          rgb_s[idx_f] * f3[idx_f])

    # ---- v34b: 渲染端河道蜿蜒 (打破 D8/M5 平行直线河) ----
    # 诊断: M5 线性倾斜面使流域内 D8 流线全部平行笔直 (sinu 1.00-1.04);
    #       即使把蜿蜒线雕刻进 DEM, 重跑 fill+D8 后水流仍重新走直线,
    #       DEM 雕刻无法让 D8 重定向 (河道太窄, 倾斜面坡度占优).
    # 方案: 不动 DEM, 对已绘制的河网 RGB 图层施加长波横向位移场扭曲
    #       (波长 ~meander_wl px, 幅度 ~meander_amp px), 使最终画出的
    #       河流呈自然蜿蜒形态 — 显示层蜿蜒, 与水文学一致.
    if meander and meander_amp > 0:
        alpha_a = np.clip(b_final[..., 0], 0.0, 1.0) * land
        idx2 = np.where(alpha_a > 1e-3)
        if idx2[0].size:
            layer = rgb.copy()
            cc = target[idx2] + G_boost[idx2].reshape(-1, 1) * \
                np.array([0.0, 1.0, 0.8], dtype=np.float64)
            layer[idx2] = cc
            # 长波横向位移场: 双轴正弦乘积 + 斜向低频分量, 归一化到 ±amp
            yy0 = np.arange(H)[:, None]
            xx0 = np.arange(W)[None, :]
            p1 = rng.uniform(0, 6.28)
            p2 = p1 + 2.1
            p3 = p1 + 4.2
            wl1 = meander_wl
            f_y = (np.sin(2 * np.pi * (yy0 / wl1) + p1)
                   * np.cos(2 * np.pi * (xx0 / wl1) + p2)
                   + 0.5 * np.sin(2 * np.pi * ((yy0 + xx0) / (wl1 * 1.7)) + p3)
                   + 0.3 * np.cos(2 * np.pi * ((yy0 - xx0) / (wl1 * 2.4)) + p1 * 1.7))
            f_x = (np.cos(2 * np.pi * (yy0 / (wl1 * 0.62)) + p3)
                   * np.sin(2 * np.pi * (xx0 / (wl1 * 0.62)) + p1)
                   + 0.5 * np.cos(2 * np.pi * ((yy0 - xx0) / (wl1 * 1.3)) + p2)
                   + 0.3 * np.sin(2 * np.pi * ((yy0 + xx0) / (wl1 * 2.1)) + p2 * 1.3))
            wy = meander_amp * f_y / max(float(np.std(f_y)), 1e-8)
            wx = meander_amp * f_x / max(float(np.std(f_x)), 1e-8)
            yy = (yy0 + wy).astype(np.float32)
            xx = (xx0 + wx).astype(np.float32)
            layer_w = np.empty_like(layer)
            for ch in range(3):
                layer_w[..., ch] = map_coordinates(layer[..., ch], [yy, xx],
                                                   order=1, mode='nearest')
            alpha_w = map_coordinates(alpha_a, [yy, xx],
                                      order=1, mode='nearest')
            a3w = np.clip(alpha_w, 0.0, 1.0)[..., None] * land[..., None]
            rgb = rgb * (1.0 - a3w) + layer_w * a3w

    # ---- v34: 河口三角洲 (泥沙沉积扇形, 尼罗河/恒河型) ----
    # 大河口 (acc ≥ q99.99 且邻海) 向海方向绘制扇形冲积扇:
    #   半径随流量增大 (6~44px), 方向 = -陆地掩膜梯度 (指向开阔海),
    #   羽化 exp(-(d/R)²) × 方向余弦窗 × 沙色淡染 → 解决"入海口直线戛然而止".
    if deltas and aL.size >= 100:
        _sea = ~land
        mouth_cand = land & (acc >= float(np.quantile(aL, 0.9999)))
        mouth_raw = mouth_cand & binary_dilation(_sea, iterations=1)
        if mouth_raw.any() and _sea.any():
            lab_d, n_d = ndi_label(mouth_raw)
            if n_d > 0:
                ys_d, xs_d = np.nonzero(mouth_raw)
                acc_d = acc[mouth_raw]
                # 只取流量最大的 60 个河口 (避免小支流入海口噪点)
                order = np.argsort(acc_d)[::-1][:60]
                gy, gx = np.gradient(land.astype(np.float32))
                sand = np.array([0.82, 0.70, 0.48], np.float32)
                A_REF_D = max(A_REF * 20.0, 10.0)
                for oi in order:
                    cy, cx = float(ys_d[oi]), float(xs_d[oi])
                    iy0, ix0 = int(round(cy)), int(round(cx))
                    # 开阔海方向 = -(land 梯度) (梯度从海指向陆, 反向即朝海)
                    uy, ux = -gy[iy0, ix0], -gx[iy0, ix0]
                    un = np.hypot(uy, ux)
                    if un < 1e-6:
                        uy, ux = 0.0, -1.0
                    else:
                        uy, ux = uy / un, ux / un
                    R_d = float(np.clip(
                        6.0 + 28.0 * np.log(max(acc_d[oi], 1.0))
                        / np.log(max(A_REF_D, 1.0)), 6.0, 44.0))
                    y0d, y1d = max(0, int(cy - R_d)), min(H, int(cy + R_d) + 1)
                    x0d, x1d = max(0, int(cx - R_d)), min(W, int(cx + R_d) + 1)
                    yy = np.arange(y0d, y1d)
                    xx = np.arange(x0d, x1d)
                    Y, X = np.meshgrid(yy, xx, indexing='ij')
                    dY, dX = Y - cy, X - cx
                    d2 = np.hypot(dY, dX)
                    m_sea = _sea[Y, X] & (d2 <= R_d)
                    if not m_sea.any():
                        continue
                    cosang = (dY * uy + dX * ux) / np.maximum(d2, 1e-6)
                    w_ang = np.clip((cosang + 0.25) / 1.25, 0.0, 1.0) ** 1.6
                    w_d = np.exp(-(d2 / R_d) ** 2)
                    idx_dd = np.where(m_sea)
                    a_dd = (w_d * w_ang)[idx_dd].astype(np.float32) * 0.62
                    if idx_dd[0].size:
                        rgb[Y[idx_dd], X[idx_dd]] = (
                            rgb[Y[idx_dd], X[idx_dd]] * (1.0 - a_dd[:, None])
                            + sand * a_dd[:, None])

    return np.clip(rgb, 0, 1).astype(np.float32)


def fill_depressions(v01, eps=4e-4, max_iters=120):
    """填洼: 抬升局部极小值 → 每个格点都有下游, 河网连通.

    v9.x 优化: 原向量化的形态学重建在 4096² 上每 iter 分配 8 个 (H,W) 临时数组
    (共 8×64MB=512MB 临时内存), 受内存带宽限制. 改用 Numba njit Priority-Flood+ε
    (Barnes 2014): 单遍扫描 + 二叉堆, 复杂度 O(N log N), 不依赖迭代次数.
    4096² 从 6.6s → ~1-2s.
    """
    f = np.ascontiguousarray(v01, np.float32).copy()
    H, W = f.shape
    if _native is not None:
        # v12: Barnes 2014 improved Priority-Flood (堆+FIFO 洼地填充),
        # 输出与标准版逐位一致且更快
        return _native.fill_depressions_bf(f, eps, H, W)
    return _fill_depressions_pf(f, eps, H, W)                 # 回退 Numba


@njit(cache=True)
def _fill_depressions_pf(elev, eps, H, W):
    """Priority-Flood+ε (Barnes 2014) — 二叉堆实现.

    输入:
        elev: (H, W) float32, 地形高程 (会被复制修改)
        eps : 微小增量, 保证每个格点下游严格更低 (drainage)
        H, W: 形状
    返回:
        filled: (H, W) float32, 填洼后的高程
    """
    N = H * W

    # 工作数组 (1D 视图, 与堆索引一致; float64 匹配堆类型)
    filled = elev.ravel().astype(np.float64).copy()
    visited = np.zeros(N, dtype=np.bool_)

    # 二叉堆 (最小堆, 按 height 排序)
    heap_h = np.empty(N, dtype=np.float64)  # 高度键
    heap_i = np.empty(N, dtype=np.int64)     # 线性索引 y*W + x
    heap_size = 0

    # ---- 1. 推入所有边界格点 ----
    # 顶行 y=0
    for x in range(W):
        idx = x
        heap_h[heap_size] = filled[idx]
        heap_i[heap_size] = idx
        heap_size += 1
        visited[idx] = True
    # 底行 y=H-1
    for x in range(W):
        idx = (H - 1) * W + x
        heap_h[heap_size] = filled[idx]
        heap_i[heap_size] = idx
        heap_size += 1
        visited[idx] = True
    # 左右列 (跳过四角, 已在顶/底行推过)
    for y in range(1, H - 1):
        idx = y * W  # x=0
        heap_h[heap_size] = filled[idx]
        heap_i[heap_size] = idx
        heap_size += 1
        visited[idx] = True
        idx = y * W + (W - 1)  # x=W-1
        heap_h[heap_size] = filled[idx]
        heap_i[heap_size] = idx
        heap_size += 1
        visited[idx] = True

    # ---- 2. Floyd 自底向上建堆 ----
    for i in range(heap_size // 2 - 1, -1, -1):
        pos = i
        while True:
            left = 2 * pos + 1
            right = 2 * pos + 2
            smallest = pos
            if left < heap_size and heap_h[left] < heap_h[smallest]:
                smallest = left
            if right < heap_size and heap_h[right] < heap_h[smallest]:
                smallest = right
            if smallest == pos:
                break
            # swap
            th = heap_h[pos]; heap_h[pos] = heap_h[smallest]; heap_h[smallest] = th
            ti = heap_i[pos]; heap_i[pos] = heap_i[smallest]; heap_i[smallest] = ti
            pos = smallest

    # ---- 3. 主循环: 弹堆 → 处理 8 邻域 ----
    while heap_size > 0:
        # 弹出最小 (根)
        h0 = heap_h[0]
        i0 = heap_i[0]
        # 末元素移到根
        heap_size -= 1
        if heap_size > 0:
            heap_h[0] = heap_h[heap_size]
            heap_i[0] = heap_i[heap_size]
            # 下沉
            pos = 0
            while True:
                left = 2 * pos + 1
                right = 2 * pos + 2
                smallest = pos
                if left < heap_size and heap_h[left] < heap_h[smallest]:
                    smallest = left
                if right < heap_size and heap_h[right] < heap_h[smallest]:
                    smallest = right
                if smallest == pos:
                    break
                th = heap_h[pos]; heap_h[pos] = heap_h[smallest]; heap_h[smallest] = th
                ti = heap_i[pos]; heap_i[pos] = heap_i[smallest]; heap_i[smallest] = ti
                pos = smallest

        # 处理 8 邻域
        y0 = i0 // W
        x0 = i0 - y0 * W
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny = y0 + dy
                nx = x0 + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nidx = ny * W + nx
                    if not visited[nidx]:
                        visited[nidx] = True
                        # 若是洼地 (低于当前 + eps), 抬升到当前 + eps
                        if filled[nidx] < h0 + eps:
                            filled[nidx] = h0 + eps
                        # 推入堆
                        nh = filled[nidx]
                        heap_h[heap_size] = nh
                        heap_i[heap_size] = nidx
                        pos = heap_size
                        heap_size += 1
                        # 上浮
                        while pos > 0:
                            parent = (pos - 1) // 2
                            if heap_h[pos] < heap_h[parent]:
                                th = heap_h[pos]; heap_h[pos] = heap_h[parent]; heap_h[parent] = th
                                ti = heap_i[pos]; heap_i[pos] = heap_i[parent]; heap_i[parent] = ti
                                pos = parent
                            else:
                                break

    return filled.reshape(H, W).astype(np.float32)


def droplet_erosion(v01, n_droplets=150000, lifetime=40, inertia=0.10,
                    capacity=4.0, erode=0.45, deposit=0.30, evaporate=0.012,
                    min_slope=0.008, rng=None):
    """粒子水力侵蚀 (Lague 式, 全向量化)."""
    rng = rng or np.random.default_rng(0)
    h = np.ascontiguousarray(v01, np.float32).copy()
    H, W = h.shape
    ys, xs = np.nonzero(h >= 0.5)
    if len(ys) < 16:
        return h
    n = n_droplets
    idx = rng.integers(0, len(ys), n)
    pos = np.stack([ys[idx] + rng.random(n).astype(np.float32),
                    xs[idx] + rng.random(n).astype(np.float32)], 1)
    vel = np.zeros((n, 2), np.float32)
    water = np.ones(n, np.float32)
    sed = np.zeros(n, np.float32)
    act = np.ones(n, bool)

    def bilin(arr, p):
        """对数组 ``arr`` 在点集 ``p`` 处做双线性插值（全向量化）。"""
        i0 = np.clip(np.floor(p[:, 0]).astype(np.int32), 0, H - 2)
        j0 = np.clip(np.floor(p[:, 1]).astype(np.int32), 0, W - 2)
        fy, fx = p[:, 0] - i0, p[:, 1] - j0
        c = (i0 * W + j0).astype(np.int64)
        af = arr.ravel()
        return (af[c] * (1 - fy) * (1 - fx) + af[c + 1] * (1 - fy) * fx
                + af[c + W] * fy * (1 - fx) + af[c + W + 1] * fy * fx)

    for _ in range(lifetime):
        if not act.any():
            break
        gy, gx = np.gradient(h)
        gp = np.stack([bilin(gy, pos), bilin(gx, pos)], 1)
        d = vel * inertia - gp * (1.0 - inertia)
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(dn > 1e-8, d / np.maximum(dn, 1e-8),
                     np.array([1.0, 0.0], np.float32))
        new_pos = pos + d
        h_old = bilin(h, pos)
        h_new = bilin(h, new_pos)
        dh = h_new - h_old
        oob = ((new_pos[:, 0] < 1) | (new_pos[:, 0] > H - 2) |
               (new_pos[:, 1] < 1) | (new_pos[:, 1] > W - 2) | (h_new < 0.5))
        cap = np.maximum(-dh, min_slope) * water * capacity
        rich = (dh > 0) | (sed > cap)
        dep = np.where(rich, np.where(dh > 0, np.minimum(sed, dh),
                                      np.maximum(sed - cap, 0.0) * deposit), 0.0)
        ero = np.where(~rich, np.minimum((cap - sed) * erode,
                                         np.maximum(-dh, 0.0)), 0.0)
        i0 = np.clip(np.floor(pos[:, 0]).astype(np.int32), 0, H - 2)
        j0 = np.clip(np.floor(pos[:, 1]).astype(np.int32), 0, W - 2)
        fy, fx = pos[:, 0] - i0, pos[:, 1] - j0
        c = (i0 * W + j0).astype(np.int64)
        wq = ((dep - ero) * act).astype(np.float32)
        hf = h.ravel()
        np.add.at(hf, c, wq * (1 - fy) * (1 - fx))
        np.add.at(hf, c + 1, wq * (1 - fy) * fx)
        np.add.at(hf, c + W, wq * fy * (1 - fx))
        np.add.at(hf, c + W + 1, wq * fy * fx)
        h = hf.reshape(H, W)
        sed = np.where(rich, sed - dep, sed + ero)
        water *= (1.0 - evaporate)
        vel, pos = d, new_pos
        act &= ~oob
    return np.clip(h, 0, 1)


def carve_trunk_rivers(v01, rng, sigma=6.0, q=0.9985, max_depth=0.045,
                       width=3.2, precip=None, temp=None, warp_amp=3.0,
                       warp_freq=0.015, use_centerline_pipeline=True,
                       q_levels=(0.988, 0.975, 0.945, 0.900),
                       hack_law=True, oxbow=True):
    """v8.3 干流+支流水系雕刻: 四等级骨架 + 尺度自适应蜿蜒 + Hack幂律宽深 + 牛轭湖。"""
    import time as _t
    _c_t0 = _t.time()
    land = v01 >= 0.5
    if not land.any():
        return v01
    v01s = gaussian_filter(v01, sigma)
    print(f"      [ctr] gaussian_filter(sigma={sigma}): {_t.time()-_c_t0:.2f}s"); _c_t0 = _t.time()
    filled = fill_depressions(v01s)
    print(f"      [ctr] fill_depressions: {_t.time()-_c_t0:.2f}s"); _c_t0 = _t.time()
    if precip is not None and temp is not None:
        acc = climate_flow_accumulation(filled, precip, temp, rng)
    else:
        acc = d8_flow_accumulation(filled, rng)
    print(f"      [ctr] flow_accumulation: {_t.time()-_c_t0:.2f}s"); _c_t0 = _t.time()
    aL = acc[land]
    if aL.size < 100:
        return v01

    if use_centerline_pipeline:
        H, W = acc.shape
        all_lines = []
        for qi, q_lv in enumerate(q_levels):
            min_len = max(int(20 - 4 * len(all_lines) // 4), 8)
            _q_t0 = _t.time()
            lines_lv = extract_river_centerlines(acc, land, q=float(q_lv),
                                                  min_len=min_len,
                                                  monotonic_repair=True)
            print(f"      [ctr] extract_river_centerlines q={q_lv} ({len(lines_lv)}条): {_t.time()-_q_t0:.2f}s")
            all_lines.extend(lines_lv)
        print(f"      [ctr] 4级中心线提取总计: {_t.time()-_c_t0:.2f}s (共 {len(all_lines)} 条)"); _c_t0 = _t.time()

        if len(all_lines) > 0:
            if warp_amp > 0.0:
                _m_t0 = _t.time()
                all_lines = meander_centerlines(
                    all_lines, rng, amp=float(warp_amp),
                    base_freq=float(warp_freq),
                    head_shrink=0.08, tail_shrink=0.06, octaves=3,
                    acc=acc, width_ref=None, scale_adaptive=True,
                    max_turn_deg=22.0, min_turn_radius_px=10.0,
                    land=land, clamp_iters=5,
                    sinuosity_bounds=(1.8, 3.0))
                print(f"      [ctr] meander_centerlines: {_t.time()-_m_t0:.2f}s"); _c_t0 = _t.time()
            _v_t0 = _t.time()
            result = carve_along_centerlines(
                v01, all_lines, acc, land,
                max_depth=max_depth, base_width=float(width),
                width_per_log_acc=0.4, cutoff_cross=4.0,
                hack_law=hack_law, oxbow=oxbow)
            print(f"      [ctr] carve_along_centerlines: {_t.time()-_v_t0:.2f}s")
            return result

    thr = float(np.quantile(aL, q))
    ch = land & (acc > thr)
    if not ch.any():
        return v01
    if warp_amp > 0.0:
        ch = warp_river_corridor(ch, rng, amp=warp_amp, base_freq=10.0)
        ch = ch & land
        if not ch.any():
            return v01
    d = distance_transform_edt(~ch)
    w = np.clip(acc / (4.0 * thr), 0.0, 1.0)
    depth = max_depth * (0.45 + 0.55 * w)
    return np.clip(v01 - depth * np.exp(-(d / width) ** 2) * land,
                   0, 1).astype(np.float32)


# ===========================================================================
# v9.0 水系图重渲染：低饱和蓝色系 + 流量宽度分级河流
# ===========================================================================

def _land_blue_color(precip, temp, land):
    """根据降水 P(mm/yr) 与气温 T(℃) 生成蓝色系陆地底色。

    设计：
      - 沙漠（低 P、高 T）→ 深蓝（低明度）
      - 雨林（高 P、高 T）→ 青色（高明度、色相偏青）
      - 寒冷区（低 T）→ 整体压暗，色相更偏蓝
    全程低饱和度，避免纯色。
    """
    P = np.asarray(precip, np.float64)
    T = np.asarray(temp, np.float64)
    p = np.clip(np.log10(np.maximum(P, 1.0) + 10.0) / np.log10(2500.0), 0.0, 1.0)
    t = np.clip((T + 30.0) / 70.0, 0.0, 1.0)
    # 色相：蓝(0.62) → 青(0.50)，降水越多越偏青；寒冷略偏蓝
    H = 0.62 - 0.12 * p + 0.02 * (t - 0.5)
    # 饱和度：整体偏低，降水越多略升
    S = 0.16 + 0.24 * p
    # 明度：降水越多彩亮；寒冷压暗
    V = 0.10 + 0.40 * p
    V = V * (0.55 + 0.45 * t)
    V = np.clip(V, 0.05, 0.62)
    hsv = np.stack([H, S, V], axis=-1).astype(np.float32)
    rgb = hsv_to_rgb(hsv)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _rasterize_polylines(lines, H, W, land):
    """将河道中线折线列表栅格化为布尔掩膜 + 对应流量场。

    用 Bresenham 连线保证相邻点之间无断裂，输出连通的河网骨架。
    v10: C++ fws_native 优先; 不可用时回退 Python 实现。
    """
    if _native is not None:
        return _native.rasterize_polylines(lines, H, W, land)
    mask = np.zeros((H, W), dtype=bool)
    for pp in lines:
        n = len(pp)
        if n < 2:
            continue
        for i in range(n - 1):
            y0, x0 = int(round(float(pp[i, 0]))), int(round(float(pp[i, 1])))
            y1, x1 = int(round(float(pp[i + 1, 0]))), int(round(float(pp[i + 1, 1])))
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            x, y = x0, y0
            while True:
                if 0 <= y < H and 0 <= x < W and land[y, x]:
                    mask[y, x] = True
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
    return mask


def paint_hydro_map_v2(rgb, acc, land, precip, temp, rng,
                       base_q=0.998,
                       ocean_color=np.array([0.10, 0.10, 0.11], np.float32),
                       min_width=0.5, max_width=9.0, width_exp=0.50,
                       feather=2.0, coast_smooth=1.2,
                       smooth_sigma=0.8, min_patch=500,
                       river_q=0.96):
    """v9.4 水系图渲染：流量场阈值 + 高斯衰减 + 严格碎块清理。

    核心改进：
      1. 高斯衰减 strength=exp(-(D/width)^2) → 自然圆润河岸。
      2. 宽度 0.5~9px → 避免河道过度膨胀成斑块。
      3. 形态学开运算移除单像素噪声 + 严格 min_patch=500 → 消除散点。
      4. 入海连通域保留，但需 >= 80 像素（避免入海口处的小碎块）。
      5. 移除重型形态学操作 → 不破坏河道拓扑。
    """
    H, W = rgb.shape[:2]
    aL = acc[land]
    if aL.size == 0:
        return rgb

    A_REF = max(float(np.quantile(aL, 0.998)), 10.0)
    sea = ~land
    sea_dilated = binary_dilation(sea, iterations=1)

    # --- 1. 基于流量累积场构建河道掩膜 ---
    thr = float(np.quantile(aL, float(river_q)))
    river_mask = land & (acc >= thr)
    if not river_mask.any():
        return rgb

    # 开运算：移除单像素噪声和细小突出
    river_mask = binary_opening(river_mask, structure=np.ones((2, 2), dtype=bool))
    # 闭运算：连接邻近水道
    river_mask = binary_closing(river_mask, structure=np.ones((3, 3), dtype=bool))

    # --- 2. 清理孤立碎块 ---
    labeled, n = ndi_label(river_mask)
    if n > 0:
        sizes = np.bincount(labeled.ravel())
        touching = river_mask & sea_dilated
        sea_adj = np.zeros(n + 1, dtype=bool)
        for lbl in np.unique(labeled[touching]):
            if lbl > 0 and sizes[int(lbl)] >= 80:
                sea_adj[int(lbl)] = True
        keep = (sizes >= min_patch) | sea_adj
        keep[0] = False
        river_mask = river_mask & keep[labeled]
    if not river_mask.any():
        return rgb

    # --- 3. 距离变换 + 流量宽度 ---
    D_cl, idx = distance_transform_edt(~river_mask, return_indices=True)
    iy, ix = idx[0], idx[1]
    A_core = np.maximum(acc[iy, ix].astype(np.float64), 1.0)
    A_star = np.clip(A_core / A_REF, 1e-3, 10.0)
    width = min_width + (max_width - min_width) * np.power(A_star, width_exp)
    width = np.clip(width, min_width, max_width)

    # --- 4. 高斯衰减：自然平滑河岸 ---
    strength = np.exp(-(D_cl / np.maximum(width, 0.1)) ** 2)
    strength = strength * land.astype(np.float32)

    if smooth_sigma and smooth_sigma > 0:
        strength = gaussian_filter(strength, sigma=smooth_sigma)

    strength[strength < 0.04] = 0.0

    # --- 5. 按流量着色 ---
    s = np.clip(np.log(A_core) / max(np.log(A_REF), 1e-6), 0.0, 1.0)
    rH = np.full_like(s, 0.57)
    rS = 0.32 + 0.10 * s
    rV = 0.76 + 0.16 * s
    r_hsv = np.stack([rH, rS, rV], axis=-1).astype(np.float32)
    rcol = hsv_to_rgb(r_hsv)
    alpha = strength.astype(np.float32)

    rgb = rgb.astype(np.float32)
    a3 = alpha[..., None]
    rgb = rgb * (1.0 - a3) + rcol * a3

    # --- 6. 海岸线平滑 ---
    if coast_smooth and coast_smooth > 0:
        land_f = land.astype(np.float32)
        land_blur = gaussian_filter(land_f, coast_smooth)
        coast = (land_blur > 0.02) & (land_blur < 0.98)
        if coast.any():
            rgb_s = np.empty_like(rgb)
            for ch in range(3):
                rgb_s[..., ch] = gaussian_filter(rgb[..., ch], coast_smooth)
            f = np.clip(land_blur, 0.0, 1.0)[..., None]
            rgb[coast] = (rgb[coast] * (1.0 - f[coast]) +
                          rgb_s[coast] * f[coast])

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 模块类
# ===========================================================================

class HydroModule(BaseModule):
    """水系模块：计算流量累积、河流与湖泊掩膜，渲染世界水系图。

    依赖：``terrain``（v01 归一化高程）、``temperature``、``climate``
    （降水 / 气温，供渲染底色使用）。

    输出键（写入 ``ctx.data``）：
        - ``hydro.flow_acc``   D8 流量累积场
        - ``hydro.river_mask`` 河道二值掩膜
        - ``hydro.lake_mask``  湖泊二值掩膜
    """
    name = "hydro"
    dependencies = ["terrain", "temperature", "climate"]

    def compute(self, ctx: WorldContext) -> None:
        """填洼后计算流量累积，并据此提取河流 / 湖泊掩膜写入 ``ctx.data``。"""
        v01 = ctx.get("terrain.v01")
        if v01 is None:
            from ..core.terrain_engine import m2v
            v01 = m2v(ctx.dem_m).astype(np.float32)
            ctx.set("terrain.v01", v01)
        land = ctx.land_mask
        rng = ctx.rng
        filled = fill_depressions(v01)
        acc = d8_flow_accumulation(filled, rng)
        ctx.set("hydro.flow_acc", acc)
        a_land = acc[land]
        if a_land.size > 0:
            thr = float(np.quantile(a_land, 0.995))
        else:
            thr = 2.0
        ctx.set("hydro.river_mask", land & (acc >= max(thr, 2.0)))
        lake = land & ((filled - v01) > 0.012)
        ctx.set("hydro.lake_mask", lake)

    def render(self, ctx: WorldContext) -> np.ndarray:
        """渲染世界水系图：深蓝海洋底 + 按降水/气温着色的陆地 + 河流与湖泊。"""
        v01 = ctx.get("terrain.v01")
        acc = ctx.get("hydro.flow_acc")
        lake_mask = ctx.get("hydro.lake_mask")
        land = ctx.land_mask
        S = ctx.S
        rng = ctx.rng

        # 降水 / 气温：供陆地底色使用（依赖中已声明 temperature/climate）
        precip = ctx.get("climate.precip")
        temp = ctx.get("temperature.temp")
        if precip is None or temp is None:
            from .temperature_module import (continentality_index,
                                             ocean_current_effects,
                                             temperature_field)
            from .climate_module import precipitation_field
            ci = continentality_index(ctx.dem_m, ctx.lats)
            dT, dP = ocean_current_effects(ctx.dem_m, ctx.lats)
            temp = temperature_field(ctx.lats, ctx.dem_m, dT=dT, ci=ci)
            precip = precipitation_field(ctx.lats, ctx.dem_m, rng=rng,
                                         dP=dP, ci=ci)

        ocean_color = np.array([0.10, 0.10, 0.11], np.float32)
        rgb = np.zeros((S, S, 3), dtype=np.float32)
        rgb[...] = ocean_color

        if land.any():
            land_rgb = _land_blue_color(precip, temp, land)
            rgb = np.where(land[..., None], land_rgb, ocean_color).astype(np.float32)

        if acc is not None:
            rgb = paint_hydro_map_v2(rgb, acc, land, precip, temp, rng)

        if lake_mask is not None and lake_mask.any():
            # 湖泊：低饱和中蓝，略亮于陆地底色，与河流区分
            lake_col = np.array([0.22, 0.34, 0.46], np.float32)
            rgb[lake_mask] = lake_col

        return np.clip(rgb, 0, 1).astype(np.float32)
