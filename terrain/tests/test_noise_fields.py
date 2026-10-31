# -*- coding: utf-8 -*-
"""terrain/tests/test_noise_fields.py —— 噪声场量纲 / 空间连续性回归测试.

v44 修复: ``directional_band_noise`` 的 ``freq_center`` 曾被直接当作
cycles/pixel 使用, 而所有调用方传的是 cycles/image (3~60). 结果掩膜带宽远大于
奈奎斯特频率 → 输出退化成**纯白噪声**, 深海海山成 1px 针尖 (椒盐噪点),
山脉岩石纹理成"斑点".

本测试锁定两件事:
  1. ``freq_center`` 按 cycles/image 解释 → 1px 自相关显著大于 0 (空间连续);
  2. ``ridged_fbm`` 的脊状场同样连续, 且值域在 [0, 1].
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from terrain.core.terrain_engine import directional_band_noise, ridged_fbm  # noqa: E402


def _lag1_autocorr(a):
    """上下相邻行的一阶自相关（白噪声 ≈ 0, 平滑场 → 1）。"""
    return float((a[:-1, :] * a[1:, :]).mean() / (a.std() ** 2 + 1e-12))


def test_directional_band_noise_is_spatially_coherent():
    """方向带通噪声必须是窄带（连续）场，而不是白噪声。"""
    S = 512
    rng = np.random.RandomState(0)
    for freq in (6.0, 20.0, 90.0):
        n = directional_band_noise(S, 45.0, freq, anisotropy=0.3, rng=rng)
        ac = _lag1_autocorr(n)
        assert ac > 0.5, "freq_center=%g 的 1px 自相关 %.3f 过低（疑似退化为白噪声）" % (freq, ac)


def test_directional_band_noise_scale_scales_with_freq():
    """``freq_center`` 越大 → 场越"粗糙"（相邻像素差 / 场标准差 单调上升）。

    白噪声该比值 ≈ √2 ≈ 1.41；被平滑的场远小于 1。若量纲被误当作 cycles/pixel，
    任何 freq 都会给出 ≈1.41（即退化白噪声），本测试会立刻失败。
    """
    S = 512
    rng = np.random.RandomState(1)

    def roughness(freq):
        n = directional_band_noise(S, 0.0, freq, anisotropy=1.0, rng=rng)
        return float(np.diff(n, axis=1).std() / (n.std() + 1e-12))

    r6, r12, r48 = roughness(6.0), roughness(12.0), roughness(48.0)
    assert r48 < 1.30, "freq=48 相邻差/场标准差 %.2f 接近白噪声 √2" % r48
    assert r6 < r12 < r48, "粗糙度未随 freq 单调上升: %.2f / %.2f / %.2f" % (r6, r12, r48)
    assert r48 / r6 > 2.0, "freq 6→48 粗糙度仅放大 %.1fx" % (r48 / r6)


def test_ridged_fbm_continuous_and_bounded():
    """脊状 fBm: 值域 [0,1] 且空间连续（不再产生 1px 针尖）。"""
    S = 512
    rng = np.random.RandomState(2)
    r = ridged_fbm(S, direction=35.0, base_freq=12.0, octaves=3, rng=rng)
    assert r.min() >= 0.0 and r.max() <= 1.0 + 1e-6
    assert _lag1_autocorr(r) > 0.5
