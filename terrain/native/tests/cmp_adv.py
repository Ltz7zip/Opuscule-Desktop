# -*- coding: utf-8 -*-
"""对拍脚本(进阶): meander / monotonic_repair / carve / rasterize 与 Python 原版对比。

用法: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 terrain/native/tests/cmp_adv.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))  # 项目根

import numpy as np
from scipy.ndimage import map_coordinates as _mc

from terrain.native import fws_native
from terrain.modules.hydro_module import (_meander_single_river,
                                          _rasterize_polylines)

FAILURES = []


def report(name, ok, msg):
    """打印单项对拍结果，失败时记录到全局 FAILURES。"""
    print(("  PASS " if ok else "  FAIL ") + name + ": " + msg)
    if not ok:
        FAILURES.append(name + ": " + msg)


def cmp_meander():
    """对拍 meander_single_river: 多组长度/蜿蜒幅度/自适应参数 + 无陆地分支, 输出蜿蜒后路径。"""
    r = np.random.RandomState(1)
    # 多种长度与参数组合
    cases = [
        (6, 3.0, 0.015, True),
        (10, 3.0, 0.015, True),
        (35, 3.0, 0.015, True),
        (80, 3.0, 0.015, True),
        (150, 3.0, 0.015, True),
        (40, 0.0, 0.015, False),          # 无蜿蜒幅度
        (40, 4.5, 0.03, True),            # 高蜿蜒
        (40, 3.0, 0.015, True),
    ]
    for idx, (n_p, amp, base_freq, scale_ad) in enumerate(cases):
        # 构造一条围绕斜坡的路径
        path = np.stack([np.linspace(5, 100, n_p),
                         np.linspace(10, 60, n_p) + 3.0 * np.sin(np.linspace(0, 3, n_p))],
                        axis=1).astype(np.float32)
        acc_at = np.abs(np.random.RandomState(10 + idx).randn(n_p)) * 50.0 + 1.0
        q99 = 500.0
        ph = np.random.RandomState(20 + idx).uniform(0, 2 * np.pi, 15)
        land = np.random.RandomState(30 + idx).rand(140, 100) > 0.3
        land[:5, :] = True
        # 与 call site 相同参数
        len_fac = float(np.clip((n_p - 30) / 90.0, 0.02, 1.0))
        S_MIN = 1.0 if n_p < 30 else (1.3 if n_p < 60 else 1.8)
        S_MAX = 3.2 if n_p < 30 else (3.1 if n_p < 60 else 3.0)
        dtheta_max = 22.0 * (np.pi / 180.0)
        ref = _meander_single_river(
            path.copy(), acc_at, q99, ph, float(amp), float(base_freq),
            0.08, 0.06, len_fac, bool(scale_ad), 10.0, S_MIN, S_MAX,
            dtheta_max, 5, land, True, 3)
        got = fws_native.meander_single_river(
            path.copy(), acc_at, q99, ph, float(amp), float(base_freq),
            0.08, 0.06, len_fac, bool(scale_ad), 10.0, S_MIN, S_MAX,
            dtheta_max, 5, land, True, 3)
        d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        ok = np.max(d) <= 1e-3
        report(f"meander n={n_p} amp={amp} scale_ad={scale_ad}",
               ok, f"max|diff|={np.max(d):.3e}, equal_rate={np.mean(d < 1e-12):.3f}")
    # scale_adaptive=False 且 has_land=False
    path = np.stack([np.linspace(3, 80, 50), np.linspace(4, 40, 50)], 1).astype(np.float32)
    acc_at = np.ones(50)
    ph = np.random.RandomState(5).uniform(0, 2 * np.pi, 15)
    q99 = 1e6
    ref = _meander_single_river(path.copy(), acc_at, q99, ph, 3.0, 0.015,
                                0.08, 0.06, 1.0, False, 10.0, 1.0, 3.2, 0.384, 5,
                                np.zeros((1, 1), bool), False, 3)
    got = fws_native.meander_single_river(path.copy(), acc_at, q99, ph, 3.0, 0.015,
                                          0.08, 0.06, 1.0, False, 10.0, 1.0, 3.2, 0.384, 5,
                                          np.zeros((1, 1), bool), False, 3)
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    report("meander no-land", np.max(d) <= 1e-3, f"max|diff|={np.max(d):.3e}")


def _py_monotonic_repair(acc, pp, land, min_len):
    """提取自 extract_river_centerlines 的原始循环 (含 scipy map_coordinates)."""
    h, w = acc.shape
    pp = [list(p) for p in pp]
    pp = np.array(pp, np.float32)
    NEI_R1 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)]
    NEI_R2 = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)
              if not (dy == 0 and dx == 0)]
    NEI_R3 = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)
              if not (dy == 0 and dx == 0)]
    for _pass in range(4):
        acc_now = _mc(acc, pp.T, order=1, mode='nearest')
        changed = False
        for i in range(len(pp) - 1):
            if acc_now[i + 1] >= acc_now[i] * 0.95:
                continue
            cy2, cx2 = int(round(float(pp[i, 0]))), int(round(float(pp[i, 1])))
            ba, by, bx = -1.0, -1, -1
            nei_list = NEI_R1 if _pass <= 1 else (NEI_R2 if _pass == 2 else NEI_R3)
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
            if np.hypot(pp[i, 0] - pp[i - 1, 0], pp[i, 1] - pp[i - 1, 1]) < 0.12:
                keep[i] = False
        if keep.sum() < min_len:
            break
        pp = pp[keep]
        if not changed:
            break
    if len(pp) >= 4:
        keep = np.ones(len(pp), dtype=bool)
        for i in range(2, len(pp)):
            if np.hypot(pp[i, 0] - pp[i - 1, 0], pp[i, 1] - pp[i - 1, 1]) < 0.12:
                keep[i] = False
        if keep.sum() >= min_len:
            pp = pp[keep]
    return pp.astype(np.float32)


def cmp_monotonic_repair():
    """对拍 monotonic_repair: 平滑随机流量场制造非单调路径, 对比修复后的路径点列。"""
    r = np.random.RandomState(100)
    h = w = 80
    # 平滑 acc 场, 制造非单调路径
    yy, xx = np.mgrid[0:h, 0:w]
    acc = (np.sin(yy / 8.0) * np.cos(xx / 7.0) + 2.0).astype(np.float64)
    acc += r.rand(h, w) * 0.2
    land = acc > 1.0
    land = land | (np.abs(yy - 40) < 6) & (np.abs(xx - 40) < 20)
    path = np.stack([np.linspace(38, 58, 30) + 0.3 * r.randn(30),
                     np.linspace(30, 70, 30) + 0.3 * r.randn(30)],
                    axis=1).astype(np.float32)
    ref = _py_monotonic_repair(acc, path, land, 8)
    got = fws_native.monotonic_repair(acc, path.copy(), land, 8, h, w)
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    report("monotonic_repair", np.max(d) <= 1e-4 and ref.shape == got.shape,
           f"shape {ref.shape} vs {got.shape}, max|diff|={np.max(d):.3e}")


def cmp_carve():
    """对拍 carve_path_dig / carve_oxbow: 蜿蜒路径与牛轭湖两条 dig 场, 逐像素比对挖掘量。"""
    r = np.random.RandomState(200)
    H = W = 96
    v = np.ones((H, W), np.float32) * 0.5
    land = np.ones((H, W), bool)
    # 一条蜿蜒路径
    n = 40
    pp = np.stack([np.linspace(12, 82, n) + 2.0 * np.sin(np.linspace(0, 4, n)),
                   np.linspace(10, 90, n)], axis=1).astype(np.float64)
    # 预备计算 (Python 原版)
    dp = np.diff(pp, axis=0)
    seg_len = np.sqrt((dp * dp).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seg_len)])
    S = max(s[-1], 1e-6)
    t = s / S
    tang = np.zeros_like(pp)
    tang[0] = dp[0] / max(seg_len[0], 1e-6)
    tang[-1] = dp[-1] / max(seg_len[-1], 1e-6)
    tang[1:-1] = (dp[1:] + dp[:-1]) / (seg_len[1:] + seg_len[:-1] + 1e-6)[:, None]
    tn = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.maximum(tn, 1e-8)
    norm_ = np.concatenate([-tang[:, 1:2], tang[:, 0:1]], axis=1)
    A_pix = np.maximum(_mc(np.ones_like(v) * 100.0, pp.T, order=1, mode='nearest'), 1.0)
    A_REF = 500.0
    A_star = np.clip(A_pix / A_REF, 1e-3, 10.0)
    width_h = 3.0 * np.power(A_star, 0.50)
    width_h = np.clip(width_h, 0.8, 48.0)
    depth_h = 0.045 * np.clip(0.18 + 0.82 * np.power(A_star, 0.40), 0.02, 1.30)
    depth_h = depth_h * np.clip(np.power(A_star, -0.30), 0.72, 1.25)
    depth_h = np.clip(depth_h, 0.045 * 0.02, 0.045 * 1.28)
    bend = 0.52 * np.sin(s * 0.017 + pp[:, 0] * 0.012 + pp[:, 1] * 0.009)

    # Python 原版雕刻循环 (与 carve_along_centerlines 一致)
    dig_py = np.zeros(H * W, np.float64)
    cutoff_cross = 4.0
    cs_step = 0.3
    cs_u = np.arange(-cutoff_cross, cutoff_cross + cs_step * 0.5, cs_step,
                     dtype=np.float64)
    for i in range(0, n, 1):
        cy, cx = pp[i, 0], pp[i, 1]
        if not (0.5 <= cy <= H - 1.5 and 0.5 <= cx <= W - 1.5):
            continue
        w_i = float(width_h[i]); d_i = float(depth_h[i]); b_i = float(bend[i])
        ny_, nx_ = float(norm_[i, 0]), float(norm_[i, 1])
        cs_pos = cs_u * w_i
        py_ = cy + cs_pos * ny_
        px_ = cx + cs_pos * nx_
        i0 = np.clip(np.floor(py_).astype(np.int32), 0, H - 2)
        j0 = np.clip(np.floor(px_).astype(np.int32), 0, W - 2)
        fy, fx = py_ - i0, px_ - j0
        base_arr = (i0 * W + j0).astype(np.int64)
        profile_v = np.abs(cs_u) ** 1.35
        trough = np.exp(-((cs_u - b_i) ** 2) / 0.16) * 0.60
        env = np.exp(-(cs_u ** 2) / (2 * 1.05 ** 2))
        rel = np.clip(profile_v - trough, 0.0, None)
        rel = rel / max(rel.max(), 1e-8)
        dig_i = d_i * (1.0 - rel) * env
        wq = dig_i.astype(np.float64)
        omfy, omfx = 1.0 - fy, 1.0 - fx
        np.maximum.at(dig_py, base_arr, omfy * omfx * wq)
        np.maximum.at(dig_py, base_arr + 1, omfy * fx * wq)
        np.maximum.at(dig_py, base_arr + W, fy * omfx * wq)
        np.maximum.at(dig_py, base_arr + W + 1, fy * fx * wq)

    dig_cpp = np.zeros(H * W, np.float64)
    fws_native.carve_path_dig(dig_cpp, pp, width_h, depth_h, bend, norm_, H, W, cutoff_cross)
    d = np.abs(dig_py - dig_cpp)
    ok = np.allclose(dig_py, dig_cpp, atol=1e-12)
    report("carve_path_dig", ok, f"max|diff|={np.max(d):.3e}, nonzero={np.count_nonzero(dig_py)}")

    # oxbow
    ox_path = np.stack([np.linspace(30, 66, 80) + 4.0 * np.sin(np.linspace(0, 12, 80)),
                        np.linspace(30, 66, 80)], axis=1).astype(np.float64)
    dig_py2 = np.zeros(H * W, np.float64)
    land_flat = land.ravel()
    for j_ in range(len(ox_path)):
        if len(ox_path) < 40:
            break
        pp2 = ox_path
        n2 = len(pp2)
        step = 3
        idxs = list(range(0, n2, step))
        d_oxb = 0.045 * 0.6
        for a in range(len(idxs)):
            ii = idxs[a]
            for b in range(a + 6, min(a + 30, len(idxs))):
                jj = idxs[b]
                if jj - ii < 18:
                    continue
                d_eu = float(np.hypot(pp2[jj, 0] - pp2[ii, 0], pp2[jj, 1] - pp2[ii, 1]))
                if d_eu > 4.0:
                    continue
                cyy, cxx = 0.5 * (pp2[ii, 0] + pp2[jj, 0]), 0.5 * (pp2[ii, 1] + pp2[jj, 1])
                ry, rx = max(d_eu * 1.1, 3.0), max(d_eu * 0.6, 1.8)
                rr = np.arange(int(cyy - max(ry, 3) - 1), int(cyy + max(ry, 3) + 2))
                cc = np.arange(int(cxx - max(rx, 3) - 1), int(cxx + max(rx, 3) + 2))
                rrg, ccg = np.meshgrid(rr, cc, indexing='ij')
                mask_valid = (rrg >= 0) & (rrg < H) & (ccg >= 0) & (ccg < W)
                if not mask_valid.any():
                    continue
                ell = ((rrg - cyy) / max(ry, 1e-3)) ** 2 + \
                      ((ccg - cxx) / max(rx, 1e-3)) ** 2
                fill = np.exp(-ell * 1.8) * d_oxb
                idx_all = (rrg * W + ccg).ravel()
                f_all = fill.ravel().astype(np.float64)
                land_ok = land_flat[idx_all]
                sel = land_ok & (f_all > 1e-5)
                if sel.any():
                    np.maximum.at(dig_py2, idx_all[sel], f_all[sel])
        break

    dig_cpp2 = np.zeros(H * W, np.float64)
    fws_native.carve_oxbow(dig_cpp2, ox_path, H, W, land, 0.045, 4.0, 0.6)
    d2 = np.abs(dig_py2 - dig_cpp2)
    report("carve_oxbow", np.allclose(dig_py2, dig_cpp2, atol=1e-12),
           f"max|diff|={np.max(d2):.3e}, nonzero={np.count_nonzero(dig_py2)}")


def cmp_rasterize():
    """对拍 rasterize_polylines: 多条随机折线栅格化为像素掩膜, 要求逐位相等。"""
    r = np.random.RandomState(300)
    H = W = 64
    land = np.ones((H, W), bool)
    lines = []
    for k in range(5):
        n = 6 + k * 3
        pts = np.stack([np.linspace(3, 55, n) + 0.4 * r.randn(n),
                        np.linspace(5 + k * 8, 40 + k * 5, n) + 0.4 * r.randn(n)],
                       axis=1).astype(np.float32)
        lines.append(pts)
    ref = _rasterize_polylines(lines, H, W, land)
    got = fws_native.rasterize_polylines(lines, H, W, land)
    report("rasterize_polylines", np.array_equal(ref, got),
           f"diff_px={np.count_nonzero(ref != got)}")


def main():
    """入口: 依次对拍 meander/monotonic_repair/carve/rasterize, 汇总失败项并退出。"""
    t0 = time.time()
    print("== meander ==")
    cmp_meander()
    print("== monotonic_repair ==")
    cmp_monotonic_repair()
    print("== carve ==")
    cmp_carve()
    print("== rasterize ==")
    cmp_rasterize()
    print(f"\n耗时 {time.time() - t0:.1f}s")
    if FAILURES:
        print(f"\n{len(FAILURES)} 项失败:")
        for f in FAILURES:
            print("  -", f)
        sys.exit(1)
    print("\n全部通过")


if __name__ == "__main__":
    main()