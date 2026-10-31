# -*- coding: utf-8 -*-
"""对拍脚本: 逐函数对比 Numba 原版与 C++ fws_native 版输出。

用法: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 terrain/native/tests/cmp_native.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))  # 项目根

import numpy as np
from numba import njit

from terrain.native import fws_native
from terrain.modules.climate_module import (_scan_ew_step, _scan_ns_step,
                                            _mountain_shield,
                                            _median_filter_njit,
                                            _mode_filter_njit)
from terrain.modules.hydro_module import (_flow_acc_kahn_njit,
                                          _fill_depressions_pf,
                                          _trace_centerline)
from terrain.modules.temperature_module import _ocean_run_lengths_njit

# ---------------------------------------------------------------- helpers
FAILURES = []


def check(name, a, b, exact=False, tol=1e-5):
    """对比两个数组并打印 PASS/FAIL，失败记录到全局 FAILURES。

    exact=True 或整型/布尔输入时要求逐位相等，否则比较浮点最大绝对差 <= tol。"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        FAILURES.append(f"{name}: shape {a.shape} != {b.shape}")
        return
    if exact or a.dtype.kind in "biu":
        ok = np.array_equal(a, b)
        msg = (f"{name}: EXACT OK" if ok else
               f"{name}: MISMATCH n={np.count_nonzero(a != b)}, "
               f"first={np.flatnonzero(~(a == b))[:5].tolist()} "
               f"a={a.ravel()[np.flatnonzero(~(a == b))[:5]]} b={b.ravel()[np.flatnonzero(~(a == b))[:5]]}")
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        ok = np.max(d) <= tol
        msg = (f"{name}: OK max|diff|={np.max(d):.3e}" if ok else
               f"{name}: DIFF max={np.max(d):.3e} mean={np.mean(d):.3e}")
    print(("  PASS " if ok else "  FAIL ") + msg)
    if not ok:
        FAILURES.append(msg)


def rng_arr(shp, dtype, seed=0, scale=1.0):
    """生成随机测试数组: bool 为 0.5 阈值采样, 数值型为 [0,scale) 均匀采样。"""
    r = np.random.RandomState(seed)
    if dtype == bool:
        return r.rand(*shp) > 0.5
    return (r.rand(*shp) * scale).astype(dtype)


# ---------------------------------------------------------------- climate
def cmp_scan(kind):
    """对拍 scan_ew_step / scan_ns_step: 随机 hk(km)/base/ocean 双方向 + float32 hk 路径。"""
    for step in (+1, -1):
        h, w = 96, 128
        hk = rng_arr((h, w), np.float64, seed=1 + step) * 4.0  # km
        base = rng_arr((h, w), np.float64, seed=2 + step) * 2000.0
        ocean = rng_arr((h, w), bool, seed=3 + step)
        if kind == "ew":
            ref = _scan_ew_step(hk, base, ocean, h, w, step)
            got = fws_native.scan_ew_step(hk, base, ocean, h, w, step)
        else:
            ref = _scan_ns_step(hk, base, ocean, h, w, step)
            got = fws_native.scan_ns_step(hk, base, ocean, h, w, step)
        check(f"scan_{kind}_step{step:+d}", ref, got)
    # float32 hk 输入 (precipitation 中 hk 为 float64; 这里测 forcecast 路径)
    hk32 = rng_arr((64, 64), np.float32, seed=5) * 4.0
    base = rng_arr((64, 64), np.float64, seed=6) * 2000.0
    ocean = rng_arr((64, 64), bool, seed=7)
    ref = _scan_ew_step(hk32.astype(np.float64), base, ocean, 64, 64, +1)
    got = fws_native.scan_ew_step(hk32, base, ocean, 64, 64, +1)
    check("scan_ew f32-hk path", ref, got)


def cmp_mountain_shield():
    """对拍 mountain_shield: 随机山地掩膜/山高/山带行, 双扫方向输出屏蔽修正场。"""
    h, w = 128, 200
    for step_sign in (+1, -1):
        mt_sm = rng_arr((h, w), bool, seed=10 + step_sign, ) | (np.abs(np.random.RandomState(88).rand(h, 1)) < 0.05)
        hh = rng_arr((h, w), np.float64, seed=11 + step_sign) * 4.0
        belt_rows = np.nonzero(np.abs(np.random.RandomState(12).randn(h)) < 0.7)[0].astype(np.int64)
        K_max = 37
        sh_ref = np.zeros((h, w), np.float64)
        sh_got = sh_ref.copy()
        ref = _mountain_shield(mt_sm, hh, sh_ref, h, w, belt_rows, step_sign, K_max)
        got = fws_native.mountain_shield(mt_sm, hh, sh_got, h, w, belt_rows, step_sign, K_max)
        check(f"mountain_shield sign{step_sign:+d}", ref, got)


def cmp_filters():
    """对拍 median_filter / mode_filter: 随机生物群系+陆地掩膜, 含边界与全海/全陆极端情形。"""
    H, W = 160, 160
    bi = rng_arr((H, W), np.int8, seed=20) % 9
    land = rng_arr((H, W), bool, seed=21) | (np.random.RandomState(22).rand(H, W) > 0.7)
    per = rng_arr((H, W), bool, seed=23)  # 少量百分比扰动
    land = land | per
    land = land.copy()
    land[0, :] = True  # 边界有陆地, 覆盖 nearest 越界分支

    # median (r=3)
    ref_out = np.empty_like(bi)
    got_out = np.empty_like(bi)
    _median_filter_njit(np.ascontiguousarray(bi), np.ascontiguousarray(land),
                        ref_out, 3, H, W)
    fws_native.median_filter_njit(np.ascontiguousarray(bi), np.ascontiguousarray(land),
                                  got_out, 3, H, W)
    check("median_filter r=3", ref_out, got_out, exact=True)
    # mode (r=2)
    ref_m = np.empty_like(bi)
    got_m = np.empty_like(bi)
    _mode_filter_njit(np.ascontiguousarray(bi), np.ascontiguousarray(land),
                      ref_m, 2, H, W)
    fws_native.mode_filter_njit(np.ascontiguousarray(bi), np.ascontiguousarray(land),
                                got_m, 2, H, W)
    check("mode_filter r=2", ref_m, got_m, exact=True)
    # 全海洋 + 全陆地极端情形
    for nm, ms in (("all_ocean", np.zeros((H, W), bool)), ("all_land", np.ones((H, W), bool))):
        r_o = np.empty_like(bi); g_o = np.empty_like(bi)
        _median_filter_njit(np.ascontiguousarray(bi), ms, r_o, 3, H, W)
        fws_native.median_filter_njit(np.ascontiguousarray(bi), ms, g_o, 3, H, W)
        check(f"median all-{nm}", r_o, g_o, exact=True)


def cmp_small_patches():
    """smooth_small_patches: C++ BFS vs Python 9×ndi_label 语义对拍."""
    H, W = 128, 128
    r = np.random.RandomState(70)
    bi = (r.rand(H, W) * 9).astype(np.int8)
    # 制造小碎片: 细条纹与小块 (面积 < 50)
    bi[(r.rand(H, W) < 0.06)] = 7  # 大量分散点
    bi[10:16, 20:23] = 2           # 6x3=18px 小斑块
    bi[15:16, 20:26] = 2           # 6px 细条 (同一分量 18+6=24? 相邻)
    bi[60:80, 60:62] = 3           # 20x2 竖条 (同一分量 40px)
    land = r.rand(H, W) > 0.35
    land[0:5, :] = True
    mode_bi = (r.rand(H, W) * 9).astype(np.int8)  # 任意 mode 场
    # Python 参考
    from scipy.ndimage import label as _ndi_label
    ref = bi.copy()
    for idx in range(9):
        m = land & (ref == idx)
        if not m.any():
            continue
        lab, n = _ndi_label(m)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        small = np.where(sizes < 50)[0]
        if len(small) == 0:
            continue
        small_mask = np.isin(lab, small) & land
        ref = np.where(small_mask, mode_bi, ref)
    got = fws_native.smooth_small_patches(
        np.ascontiguousarray(bi), np.ascontiguousarray(mode_bi),
        np.ascontiguousarray(land), 50, H, W)
    check("smooth_small_patches", ref, got, exact=True)
    # 全陆地全海洋极端
    for nm, ms in (("all_land", np.ones((H, W), bool)),
                   ("all_ocean", np.zeros((H, W), bool))):
        ref2 = bi.copy()
        if ms.any():
            for idx in range(9):
                m = ms & (ref2 == idx)
                if not m.any():
                    continue
                lab, n = _ndi_label(m)
                if n == 0:
                    continue
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                small = np.where(sizes < 50)[0]
                if len(small) == 0:
                    continue
                small_mask = np.isin(lab, small) & ms
                ref2 = np.where(small_mask, mode_bi, ref2)
        got2 = fws_native.smooth_small_patches(
            np.ascontiguousarray(bi), np.ascontiguousarray(mode_bi),
            np.ascontiguousarray(ms), 50, H, W)
        check(f"smooth_small_patches {nm}", ref2, got2, exact=True)


# ---------------------------------------------------------------- hydro
def cmp_flow_kahn():
    """对拍 flow_acc_kahn: 随机带噪声高程, 输出逐像素流量累积, 要求逐位相等。"""
    h = w = 96
    r = np.random.RandomState(30)
    elev = r.rand(h, w)
    elev += 0.02 * r.rand(h, w)  # 平缓+噪声
    e = (elev.astype(np.float32) + r.rand(h, w).astype(np.float32) * 1e-4).ravel()
    acc_ref = np.ones(h * w, np.float64)
    acc_got = acc_ref.copy()
    _flow_acc_kahn_njit(e.copy(), acc_ref, h, w)
    fws_native.flow_acc_kahn(e.copy(), acc_got, h, w)
    check("flow_acc_kahn", acc_ref.reshape(h, w), acc_got.reshape(h, w), exact=True)


def cmp_fill_depressions():
    """对拍 fill_depressions_pf(Barnes FIFO): 多洼地高斯坑+噪声+斜坡 DEM, 输出填洼后高程。"""
    h = w = 96
    r = np.random.RandomState(40)
    v = np.zeros((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for cy, cx, rr, amp in [(20, 20, 8, -1.0), (70, 60, 10, -0.8), (50, 80, 6, -0.9)]:
        v += (amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * rr ** 2))).astype(np.float32)
    v += (0.05 * r.rand(h, w)).astype(np.float32)
    v += np.linspace(0, 1, h)[:, None].astype(np.float32)
    ref = _fill_depressions_pf(np.ascontiguousarray(v), 4e-4, h, w)
    got = fws_native.fill_depressions_bf(np.ascontiguousarray(v), 4e-4, h, w)  # v12: Barnes FIFO 版
    check("fill_depressions_pf(bf)", ref, got, exact=True)


def cmp_trace_centerline():
    """对拍 trace_centerline: 随机流量场单一连通域, 从上游追踪到下游, 对比路径与 visited。"""
    h = w = 32
    r = np.random.RandomState(50)
    acc = r.rand(h, w).astype(np.float64)
    acc[0, 0] = 0.0  # 上游
    acc[h - 1, w - 1] = 1.0  # 下游
    labels = np.ones((h, w), np.int32)  # 单一 comp
    visited_ref = np.zeros((h, w), np.int32)
    visited_got = np.zeros((h, w), np.int32)
    y0 = x0 = 16
    ref = _trace_centerline(acc, labels, 1, h, w, y0, x0, 64, visited_ref)
    got = fws_native.trace_centerline(acc, labels, 1, h, w, y0, x0, 64, visited_got)
    check("trace_centerline path", ref, got, exact=True)
    check("trace_centerline visited", visited_ref, visited_got, exact=True)


# ---------------------------------------------------------------- temperature
def cmp_ocean_run_lengths():
    """对拍 ocean_run_lengths: 随机海洋掩膜, 输出东西向海洋连续游程, 含全海/全陆极端。"""
    S, W2 = 48, 128
    sea = rng_arr((S, W2), bool, seed=60)
    ref_e, ref_w = _ocean_run_lengths_njit(sea, S, W2)
    got_e, got_w = fws_native.ocean_run_lengths(sea, S, W2)
    check("ocean_run_lengths run_e", ref_e, got_e, exact=True)
    check("ocean_run_lengths run_w", ref_w, got_w, exact=True)
    # 全海 / 全陆
    for nm, ms in (("all_sea", np.ones((S, W2), bool)), ("all_land", np.zeros((S, W2), bool))):
        re, rw = _ocean_run_lengths_njit(ms, S, W2)
        ge, gw = fws_native.ocean_run_lengths(ms, S, W2)
        check(f"ocean_run_lengths {nm} e", re, ge, exact=True)
        check(f"ocean_run_lengths {nm} w", rw, gw, exact=True)


# ---------------------------------------------------------------- main
def main():
    """入口: 依次对拍 climate/hydro/temperature 各内核, 汇总失败项并退出。"""
    t0 = time.time()
    print("== climate ==")
    cmp_scan("ew")
    cmp_scan("ns")
    cmp_mountain_shield()
    cmp_filters()
    cmp_small_patches()
    print("== hydro ==")
    cmp_flow_kahn()
    cmp_fill_depressions()
    cmp_trace_centerline()
    print("== temperature ==")
    cmp_ocean_run_lengths()
    print(f"\n耗时 {time.time() - t0:.1f}s")
    if FAILURES:
        print(f"\n{len(FAILURES)} 项失败:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("\n全部通过")


if __name__ == "__main__":
    main()