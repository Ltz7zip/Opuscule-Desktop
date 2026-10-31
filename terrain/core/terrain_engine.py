# -*- coding: utf-8 -*-
"""
terrain_engine.py v9.1 —— 物理工具 / 气候模型 / GAN (v5.1 UNet 12 通道条件) / 侵蚀 / v8.3 河流系统 / 渲染 / 训练

=============================================================
v7.0 新增：洋流/季风效应(M4)、峡湾/台地/珊瑚礁(M3)、克拉通大流域(M5)、深洋直方图终检(M6)
v7.1–v7.4：大陆度指数 CI、极地冰盖过渡带、气候去伪影、泰加林修复
v8.3：河流水系系统（骨架+单调修复 / Hack 定律蜿蜒 / 牛轭湖 / 连续蓝度渲染）
数据: ETOPO1 (NOAA) 自动下载(带重试) / 本地 npy / 合成兜底
=============================================================
"""
import os
import time
import socket
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           map_coordinates, zoom as ndi_zoom,
                           median_filter, label as ndi_label,
                           maximum_filter, minimum_filter, uniform_filter)

try:
    from scipy.io import netcdf_file
except Exception:
    netcdf_file = None

# ==================== 0. 环境配置 ====================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

if torch.backends.mps.is_available():
    device = torch.device("mps"); print("Using device: Apple Silicon MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Using device: CUDA")
else:
    device = torch.device("cpu"); print("警告: 未检测到GPU, 将非常缓慢")
print("-" * 60)

# ==================== 1. 物理归一化: 米 <-> [0,1] ====================
M_KNOTS = np.array([-11000, -6000, -4000, -2500, -1000, -200, 0,
                    100, 400, 1000, 2000, 3500, 5000, 8849], np.float64)
V_KNOTS = np.array([0.00, 0.08, 0.22, 0.34, 0.42, 0.47, 0.50,
                    0.53, 0.60, 0.70, 0.82, 0.92, 0.97, 1.00], np.float64)
# 说明：M_KNOTS 与 V_KNOTS 是全局「米制高程 ↔ 归一化 v∈[0,1]」单调分段线性映射。
#   - 深海端压缩（−11000 → 0.08）：−6000m 以内 8% 的值空间就够装，留 42% 给陆壳。
#   - v=0.50 恰好对应 elev=0m（海平面），陆地判定 v01>=0.5 与 elev>=0m 严格一致。
#   - 陆端压缩集中平原 / 丘陵 / 山地：2000m ≈ 0.82、5000m ≈ 0.97、8849m=1.00。


def m2v(m):
    """米制高程 → 归一化 [0, 1]（按 M_KNOTS/V_KNOTS 分段线性插值）。

    例：m=0m → 0.50；m=+2000m → ~0.82；m=−6000m → ~0.08。
    常用于把 DEM 转为 GAN 输入/输出的 v 域（激活函数尾端饱和）。"""
    return np.interp(m, M_KNOTS, V_KNOTS).astype(np.float32)


def v2m(v):
    """归一化 v∈[0,1] → 米制高程（``m2v`` 的严格逆映射）。

    取值限制：v<0 → −11000m，v>1 → +8849m（外推由 np.interp 常数延伸）。"""
    return np.interp(v, V_KNOTS, M_KNOTS)


# 雪线纬度曲线 (米): 副热带(20°~30°)最高——安第斯型; 赤道次之; 极地降至海平面
# v22: 雪线累计下调 35%, 补偿高海拔压缩 (v21 perm_snow 12.0% vs 地球 24.2%)
# A1b: 50-66.5° 带永久雪盖只有 11.4% (地球年均 45%, supervisor 连续 warn)。
#   该带陆地高程 p50=354 / p75=814 / p90=1934m, 旧表 50°=1755 / 60°=910 /
#   66.5°=520m 只盖住最高的一小撮山脊。按"无季节世界的永久雪线"下调中高纬:
#   50° 1755→660 / 60° 910→300 / 66.5° 520→150 / 75° 163→85 / 40° 2730→2680,
#   热带(0-30°)雪占 3.3% 仍在容差内。实测(seed 20261031 / B1 DEM 全量重跑):
#   该带 11.4%→43.3%, A1 雪覆盖 5.11%→8.33% 陆地, 北半球非极区雪块 12→10 个,
#   成片山帽(>=500px)占雪面积 97.4%。
_SL_LAT = np.array([0, 10, 20, 25, 30, 40, 50, 60, 66.5, 75, 90], np.float64)
_SL_M = np.array([3120, 3510, 3738, 3770, 3543, 2680, 660, 300, 150, 85, 0], np.float64)
# 河谷下切/侵蚀用的冰盖判定表 (A1b 之前的 v22 口径)。
# 为什么单列一份: 雪线表同时被 DEM 管线里的 polar_ice_mask 用来屏蔽"冰下不切河",
# 直接跟随新表会让 seed 20261031 的 DEM 漂移(实测陆块掩膜差 6.0万px / p95 |Δ| 385m),
# 使 B1 已验收的地形 (md5 a07375fd.../ >6000m 154px) 失效。A1 属渲染层, 故冻结此表。
_SL_INCISION_M = np.array([3120, 3510, 3738, 3770, 3543, 2730, 1755, 910, 520, 163, 0],
                          np.float64)


def snowline_m(lat):
    """纬度雪线 (米): 副热带最高 ~5800 (干旱→高雪线), 赤道 ~4800, 极地海平面"""
    a = np.abs(np.asarray(lat, np.float64))
    return np.interp(a, _SL_LAT, _SL_M)




# ---- A1: block_reduce 支持 mean/std/min/max ----
def block_reduce(a, block, mode):
    """``skimage.measure.block_reduce`` 的轻量重实现，避免额外依赖。

    把二维数组按 ``block × block`` 非重叠切块，对每块按 ``mode`` 进行归约：
        - ``"mean"``: 算术平均
        - ``"std"`` : 标准差（总体 / ddof=0）
        - ``"min"`` : 最小值
        - ``"max"`` : 最大值

    右下方不足一整块的行列会被**丢弃**（调用方保证 S 是 block 的整数倍即可）。

    Parameters
    ----------
    a : ndarray (H, W)
        二维输入数组。
    block : int
        单边块大小（正方形块）。
    mode : {"mean", "std", "min", "max"}
        归约方式。

    Returns
    -------
    ndarray (H//block, W//block)
        归约结果，dtype 继承自原 dtype（std 会升级到 float）。
    """
    H, W = a.shape
    b = a[:H // block * block, :W // block * block].reshape(
        H // block, block, W // block, block)
    if mode == "mean":
        return b.mean(axis=(1, 3))
    if mode == "std":
        return b.std(axis=(1, 3))
    if mode == "min":
        return b.min(axis=(1, 3))
    if mode == "max":
        return b.max(axis=(1, 3))
    raise ValueError(mode)


# ==================== 2. D8 流向/流量累积 ====================








# ==================== 2b. 河流骨架级蜿蜒 + 沿骨架雕刻 ====================








# ==================== 3. ETOPO1 获取与加载 ====================
ETOPO1_URLS = [
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/ETOPO1_Ice_g_gmt4.grd",
    "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/ETOPO1_Ice_g_gmt4.grd",
]
# ETOPO1 下载大小上限 (~1GB), 文件实际约 450MB
_ETOPO1_MAX_BYTES = 1024 * 1024 * 1024


def download_dem(dest, attempts=3, timeout=180):
    """下载 ETOPO1 全球 DEM 到 dest；已存在直接返回，多次尝试失败后抛异常。"""
    if os.path.exists(dest):
        return dest
    print("开始下载 ETOPO1 全球 DEM (约450MB, 公有领域)...")
    from urllib.parse import urlparse as _urlparse
    last = None
    for url in ETOPO1_URLS:
        # scheme 白名单校验: 仅允许 http/https, 防止 file:// 等 SSRF
        _scheme = _urlparse(url).scheme.lower()
        if _scheme not in ("http", "https"):
            last = ValueError(f"不允许的 URL scheme: {_scheme!r}")
            continue
        for a in range(1, attempts + 1):
            tmp = dest + ".part"
            try:
                socket.setdefaulttimeout(timeout)
                _req = urllib.request.Request(
                    url,
                    headers={"User-Agent":
                             "FantasyWorldSimulator/1.0 (+https://github.com)"},
                )
                req = urllib.request.urlopen(_req, timeout=timeout)  # nosec B310 - scheme 已在循环开头白名单校验
                total = int(req.headers.get('Content-Length', 0))
                # 若服务器声明的大小超过上限, 直接拒绝
                if total and total > _ETOPO1_MAX_BYTES:
                    raise ValueError(
                        f"Content-Length {total/1e6:.0f}MB 超过上限 "
                        f"{_ETOPO1_MAX_BYTES/1e6:.0f}MB")
                done, t0 = 0, time.time()
                with open(tmp, 'wb') as f:
                    while True:
                        chunk = req.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        # 实际下载量也不能超过上限
                        if done > _ETOPO1_MAX_BYTES:
                            raise ValueError(
                                f"下载量超过上限 {_ETOPO1_MAX_BYTES/1e6:.0f}MB")
                        if total:
                            print(f"\r  {done/1e6:5.0f}/{total/1e6:.0f} MB  "
                                  f"{done/max(1e-9, time.time()-t0)/1e6:5.1f} MB/s", end='')
                print()
                os.replace(tmp, dest)
                return dest
            except Exception as e:
                last = e
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"\n  第 {a} 次尝试失败 ({url.split('/')[2]}): {e}")
                time.sleep(5 * a)
    raise RuntimeError(
        f"自动下载失败: {last}\n"
        "请手动获取 ETOPO1 (任选其一):\n"
        f"  1) curl -L -o {dest} '{ETOPO1_URLS[0]}'\n"
        "  2) 浏览器打开上述链接下载, 放到脚本同目录\n"
        "  (需要代理时: export https_proxy=http://127.0.0.1:端口 后重跑)\n"
        "  3) 任何全球米制高程 .npy 文件: 设环境变量 DEM_PATH=/路径/dem.npy")


class GlobalDEM:
    """全球 DEM 提供器。patch(y0,x0,size) -> (米制高程patch, 行纬度数组)"""

    def __init__(self, path):
        """``__init__``: 加载 .npy/.npz 或 .grd 格式的全球 DEM 并做归一化预处理。"""
        print(f"加载全球 DEM: {path}")
        if path.endswith((".npy", ".npz")):
            z = np.load(path)
            if path.endswith(".npz"):
                z = z[list(z.keys())[0]]
            self.z = np.ascontiguousarray(z)
            self.scale, self.offset, self.lon_roll = 1.0, 0.0, 0
        else:
            if netcdf_file is None:
                raise RuntimeError("读取 .grd 需要 scipy.io.netcdf_file")
            f = netcdf_file(path, "r", mmap=False)
            zv = f.variables["z"]
            raw = np.array(zv.data)
            self.scale = float(np.atleast_1d(getattr(zv, "scale_factor", 1.0))[0])
            self.offset = float(np.atleast_1d(getattr(zv, "add_offset", 0.0))[0])
            self.z = raw
            if self.z.shape[0] > self.z.shape[1]:
                self.z = self.z.T
            if "y" in f.variables:
                y = np.array(f.variables["y"][:], np.float64).ravel()
                if y[0] > y[-1]:
                    self.z = np.ascontiguousarray(self.z[::-1])
            self.lon_roll = 0
            if "x" in f.variables:
                x = np.array(f.variables["x"][:], np.float64).ravel()
                if x[0] >= 0 and x[-1] > 180:
                    step = x[1] - x[0]
                    self.lon_roll = int(round((180.0 - x[0]) / step)) % self.z.shape[1]
            f.close()
        self.ny, self.nx = self.z.shape
        sub = self.z[::16, ::16].astype(np.float32) * self.scale + self.offset
        self.land_low = (sub > 0).astype(np.float32)
        print(f"  网格 {self.ny}x{self.nx} | 全球陆地占比 {self.land_low.mean()*100:.1f}%")

    def patch(self, y0, x0, size):
        """裁剪出 (y0,x0) 处 size×size 的米制高程块，并返回对应的行纬度数组。"""
        y0 = int(np.clip(y0, 0, self.ny - size))
        yidx = np.arange(y0, y0 + size)
        xidx = (np.arange(x0, x0 + size) + self.lon_roll) % self.nx
        p = self.z[np.ix_(yidx, xidx)].astype(np.float32) * self.scale + self.offset
        lats = -90.0 + (yidx + 0.5) * (180.0 / self.ny)
        return p.astype(np.float32), lats


# ==================== 4. 合成兜底数据源 ====================
def _fft_gaussian_filter(n, sigma):
    """FFT 实现的 2D 高斯卷积 (周期边界).

    大 sigma 时替代 scipy.ndimage.gaussian_filter (spatial 卷积 O(N*sigma)).
    FFT 路径 O(N log N), 对 sigma=640 / 4096² 可加速 ~10-50 倍.

    scipy gaussian_filter 用 reflect 边界 + truncate=4.0 截断;
    本实现用周期边界, 全长核. 对细节噪声 (value_fbm / 降水扰动)
    两者在图中间区域数值接近, 边界差异可忽略.
    """
    H, W = n.shape
    # 频率 (cycles/pixel), rfft 仅保留正频半轴
    fy = np.fft.fftfreq(H)[:, None]      # (H, 1)
    fx = np.fft.rfftfreq(W)[None, :]     # (1, W//2+1)
    # 高斯传递函数: G(f) = exp(-2 π² σ² (fx² + fy²))
    kernel = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (fy * fy + fx * fx))
    F = np.fft.rfft2(n)
    out = np.fft.irfft2(F * kernel, s=(H, W))
    # 与 scipy gaussian_filter 保持同量级 (scipy 不归一化, FFT 路径需匹配)
    # scipy 高斯核 sum≈1, FFT 路径 DC 增益=1, 量级一致
    return out.astype(n.dtype)


def value_fbm(size, base_freq=4.0, octaves=5, gain=0.5, rng=None,
              fft_threshold=16.0):
    """分形布朗运动噪声 (多 octave 高斯叠加).

    算法层优化 (2026-09): sigma >= fft_threshold 时切换 FFT 卷积,
    替代 scipy gaussian_filter 的 spatial 卷积, 大 sigma 下加速 10-50x.
    """
    rng = rng if rng is not None else np.random
    noise = np.zeros((size, size), np.float32)
    amp, total, freq = 1.0, 0.0, base_freq
    for _ in range(octaves):
        n = rng.randn(size, size).astype(np.float32)
        sigma = max(size / (2.0 * freq), 0.6)
        if sigma >= fft_threshold:
            n = _fft_gaussian_filter(n, sigma)
        else:
            n = gaussian_filter(n, sigma=sigma)
        n /= (np.std(n) + 1e-8)
        noise += amp * n; total += amp; amp *= gain; freq *= 2.0
    return noise / total


def _lowres_fbm(size, base_freq, octaves, rng, gain=0.5, downsample=4):
    """低分辨率 fBm + FFT 零填充上采样 (低频噪声专用).

    在 size//downsample 分辨率生成 fBm, 通过 FFT 频域零填充上采样至 size×size
    (等效 sinc 插值). 省 ~downsample² FFT 开销, 视觉差异可忽略 (低频噪声).
    v9.x: 原 value_fbm(4096, 3.2, 2) 约 1.0s, 本函数约 0.17s (6x 加速).
    """
    lo = max(min(size // downsample, size), 8)
    if lo >= size:
        return value_fbm(size, base_freq, octaves, gain=gain, rng=rng)
    n_lo = value_fbm(lo, base_freq, octaves, gain=gain, rng=rng)
    F = np.fft.rfft2(n_lo)
    F_pad = np.zeros((size, size // 2 + 1), dtype=complex)
    half = lo // 2
    F_pad[:half, :lo // 2 + 1] = F[:half, :]
    F_pad[-half:, :lo // 2 + 1] = F[half:, :]
    n_up = np.fft.irfft2(F_pad, s=(size, size)) * (size / lo) ** 2
    return n_up.astype(np.float32)


def _lowres_directional_noise(size, direction_deg, freq_center, anisotropy=0.35,
                              rng=None, downsample=4):
    """低分辨率方向性条带噪声 + FFT 零填充上采样."""
    lo = max(min(size // downsample, size), 8)
    if lo >= size:
        return directional_band_noise(size, direction_deg, freq_center, anisotropy, rng)
    n_lo = directional_band_noise(lo, direction_deg, freq_center, anisotropy, rng)
    F = np.fft.rfft2(n_lo)
    F_pad = np.zeros((size, size // 2 + 1), dtype=complex)
    half = lo // 2
    F_pad[:half, :lo // 2 + 1] = F[:half, :]
    F_pad[-half:, :lo // 2 + 1] = F[half:, :]
    n_up = np.fft.irfft2(F_pad, s=(size, size)) * (size / lo) ** 2
    return n_up.astype(np.float32)


def _edt_lowres_periodic(mask, downsample=4):
    """低分辨率周期 EDT + np.repeat 上采样 (水平周期边界).

    在 size//downsample 分辨率下 3x 水平铺贴 EDT 取中段, 再 np.repeat 上采样回原尺寸.
    v9.x: 原 np.tile(4096,3) EDT ~3s, 本函数 ~0.08s (37x 加速).
    用于 add_craton_basins 中粗阈值距离场 (≥25px), 4px 误差可接受.
    """
    H, W = mask.shape
    lo_h = max(H // downsample, 8)
    lo_w = max(W // downsample, 8)
    H2 = lo_h * downsample
    W2 = lo_w * downsample
    # 下采样: 格内任一像素属于 mask 即为"内部"(distance=0)
    # 旧版 (~mask).any() 对稀疏 mask (如单点出口) 会丢失种子 → d_o 最小值=size 而非 0
    coast_lo = ~(mask[:H2, :W2].reshape(lo_h, downsample, lo_w, downsample).any(axis=(1, 3)))
    # 3x 水平铺贴 + EDT + 取中段
    # tiled=True 在 mask 外, EDT 计算每个外像素到最近内像素的距离 → 0 内, 正外 (与 _edt_to 一致)
    tiled = np.tile(coast_lo, (1, 3))
    d_tiled = distance_transform_edt(tiled)  # 前景=tiled (mask 外), 距离到最近 mask 内
    d_lo = d_tiled[:, lo_w:2 * lo_w] * downsample  # 恢复像素尺度
    # np.repeat 上采样回原尺寸
    d_hi = np.repeat(np.repeat(d_lo, downsample, axis=0), downsample, axis=1)
    # 边缘填充补齐 (H/W 不被 downsample 整除时)
    if d_hi.shape[0] < H or d_hi.shape[1] < W:
        d_full = np.empty((H, W), dtype=d_hi.dtype)
        d_full[:d_hi.shape[0], :d_hi.shape[1]] = d_hi
        if d_hi.shape[0] < H:
            d_full[d_hi.shape[0]:, :] = d_hi[-1:, :]
        if d_hi.shape[1] < W:
            d_full[:, d_hi.shape[1]:] = d_full[:, d_hi.shape[1]-1:d_hi.shape[1]]
        d_hi = d_full
    return d_hi


def directional_band_noise(size, direction_deg, freq_center, anisotropy=0.35, rng=None):
    """生成具有指定主方向、中心频率与各向异性的方向带通噪声场。

    ``freq_center`` 的量纲是 **整幅图内的周期数**（cycles/image），与
    :func:`value_fbm` 的 ``base_freq`` 一致：``freq_center=6`` 表示沿主方向
    波长 ≈ size/6 像素。

    v44 修正：旧实现把 ``freq_center`` 直接喂给 ``np.fft.fftfreq``（cycles/pixel），
    而所有调用方传的是 3~60 的 cycles/image 值。以 2048² 为例，``freq_center=6``
    对应的掩膜带宽 0.35*6=2.1 cycles/pixel 远大于奈奎斯特 0.5，掩膜在整个频域
    上恒 ≈0.95~1.0（实测覆盖率 100%），输出退化为**纯白噪声**（1px 自相关 0.006）。
    这就是深海海山成 1px 针尖（椒盐噪点）、山脉纹理成"斑点"的根源。
    现统一除以 ``size`` 归一化到 cycles/pixel，输出才真正是窄带方向噪声
    （1px 自相关 ≈0.97~0.99，空间连续）。
    """
    rng = rng if rng is not None else np.random
    d = np.deg2rad(direction_deg)
    f0 = max(freq_center / float(size), 1e-6)   # cycles/pixel
    fx = np.fft.fftfreq(size).reshape(1, -1)
    fy = np.fft.fftfreq(size).reshape(-1, 1)
    u = fx * np.cos(d) + fy * np.sin(d)
    v = -fx * np.sin(d) + fy * np.cos(d)
    mask = np.exp(-(u ** 2 / (2 * max(0.35 * f0, 1e-9) ** 2)
                    + v ** 2 / (2 * max(0.35 * f0 / anisotropy, 1e-9) ** 2)))
    n = rng.randn(size, size).astype(np.float32)
    return np.real(np.fft.ifft2(np.fft.fft2(n) * mask)) / (np.std(n) + 1e-8)


def ridged_fbm(size, direction=None, base_freq=3.0, octaves=4, gain=0.55, rng=None):
    """生成山脊化分形噪声（可选主方向），多倍频叠加后归一化。

    ``base_freq`` 为第一倍频的 **整幅周期数**（cycles/image），后续倍频逐次 ×2。
    ``direction`` 给定时各倍频用 :func:`directional_band_noise`（窄带方向噪声），
    否则用 :func:`value_fbm`（各向同性）。
    """
    noise = np.zeros((size, size), np.float32)
    amp, total, freq = 1.0, 0.0, base_freq
    for _ in range(octaves):
        n = directional_band_noise(size, direction, freq, rng=rng) if direction is not None \
            else value_fbm(size, base_freq=freq, octaves=1, rng=rng)
        n = (n - n.min()) / (n.max() - n.min() + 1e-8)
        n = 1.0 - np.abs(2.0 * n - 1.0)
        noise += amp * (n ** 2); total += amp; amp *= gain; freq *= 2.0
    return noise / total


def warp_field(field, wx, wy, strength):
    """用 (wx,wy) 偏移场对 field 做域扭曲并返回双线性重采样结果。"""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return map_coordinates(field, [yy + wx * strength, xx + wy * strength],
                           order=1, mode='nearest').astype(np.float32)


def periodic_fbm(size, freq=6.0, octaves=3, rng=None):
    """双向周期 fBm (FFT 实现) — 域扭曲专用, 数据线(经度缝)无接缝"""
    rng = rng if rng is not None else np.random
    out = np.zeros((size, size), np.float64)
    amp, tot, f = 1.0, 0.0, float(freq)
    for _ in range(octaves):
        F = np.fft.rfft2(rng.randn(size, size))
        ky = np.fft.fftfreq(size)[:, None] * size
        kx = np.fft.rfftfreq(size)[None, :] * size
        F *= np.exp(-((np.hypot(kx, ky) / f) ** 2) / 2.0)
        n = np.fft.irfft2(F, s=(size, size))
        out += amp * n / (n.std() + 1e-12)
        tot += amp; amp *= 0.5; f *= 2.0
    return (out / tot).astype(np.float32)


def warp_lonlat(field, wx, wy, strength):
    """域扭曲(经度环绕版): 全球场专用, x 向取模回卷, y 向极区钳制"""
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = yy + wx * strength
    cx = (xx + wy * strength) % w
    return map_coordinates(field, [cy, cx], order=1,
                           mode='nearest').astype(np.float32)


def _stamp_disk(lm, cy, cx, rad, value):
    """在掩膜 lm 中以 (cy,cx) 为圆心、rad 为半径的圆内写入 value。"""
    h, w = lm.shape
    y0, y1 = max(0, int(cy - rad) - 1), min(h, int(cy + rad) + 2)
    x0, x1 = max(0, int(cx - rad) - 1), min(w, int(cx + rad) + 2)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    lm[y0:y1, x0:x1][(yy - cy) ** 2 + (xx - cx) ** 2 <= rad * rad] = value


def grow_peninsulas(land_mask, rng, n_range=(1, 4)):
    """从陆地边缘随机长出海岬（摆动圆盘盖章），返回新陆地掩膜。"""
    lm = land_mask.copy()
    h, w = lm.shape
    for _ in range(rng.randint(*n_range)):
        cand = np.argwhere(lm)
        if len(cand) == 0:
            break
        sy, sx = cand[rng.randint(len(cand))]
        ang, length, w0 = rng.uniform(0, 2 * np.pi), rng.randint(5, 14), rng.uniform(1.0, 1.9)
        y, x = float(sy), float(sx)
        for step in range(length):
            ang += rng.normal(0, 0.4)
            y += 1.2 * np.sin(ang); x += 1.2 * np.cos(ang)
            if y < 2 or y > h - 3 or x < 2 or x > w - 3:
                break
            _stamp_disk(lm, y, x, max(w0 * (1 - 0.75 * step / length), 0.3), True)
    return lm


def generate_folded_belt(size, rng):
    """生成褶皱山带场（多支正弦折叠脊线 + 随机游走摆动）。"""
    x = np.arange(size, dtype=np.float32)
    base = rng.uniform(0.3, 1.8) * x + rng.uniform(-size * 0.3, size * 0.3)
    wiggle = rng.uniform(4, 10) * np.sin(rng.uniform(0.03, 0.08) * x + rng.uniform(0, 2 * np.pi))
    walk = gaussian_filter(np.cumsum(rng.normal(0, 0.8, size)).astype(np.float32), 4)
    y0 = base + wiggle + (walk - walk.mean()) * 1.5
    yy = np.mgrid[0:size, 0:size][0].astype(np.float32)
    n_folds, spacing = rng.randint(2, 5), rng.uniform(3.0, 7.0)
    belt = np.zeros((size, size), np.float32)
    for k in range(n_folds):
        off = (k - (n_folds - 1) / 2.0) * spacing + rng.normal(0, 1.5)
        wob = gaussian_filter(np.cumsum(rng.normal(0, 0.5, size)).astype(np.float32), 5)
        d = yy - (y0 + off + wob - wob.mean())[None, :]
        belt = np.maximum(belt, rng.uniform(0.55, 1.0) * np.exp(-(d * d) / (2 * rng.uniform(1.2, 2.6) ** 2)))
    return belt


def hydraulic_erosion(height, iterations=1200, dt=0.05):
    """简易水力侵蚀迭代：按坡度下切、扩散与平滑，返回 [0,1] 高程。"""
    h = height.astype(np.float64).copy()
    for _ in range(iterations):
        gy, gx = np.gradient(h)
        h -= np.clip(np.sqrt(gx ** 2 + gy ** 2) * 0.08 * (h + 0.1) - 0.005, 0, 0.08) * dt
        h += np.clip(gaussian_filter(h, 1.0) - h, 0, 0.03) * 0.15 * dt
        h = gaussian_filter(h, 0.25) * 0.08 + h * 0.92
    return np.clip(h, 0, 1).astype(np.float32)


def tail_stretch(elev, land_all):
    """按海/陆分位点拉伸高程两端并做平滑尾部，返回变换后的数组。"""
    m_sea = ~land_all
    sea_lo = np.quantile(elev[m_sea], 0.002)
    land_hi = np.quantile(elev[land_all], 0.998)
    elev[m_sea] = 0.5 * np.clip((elev[m_sea] - sea_lo) / (0.5 - sea_lo + 1e-8), 0, 1)
    elev[land_all] = 0.5 + 0.5 * np.clip((elev[land_all] - 0.5) / (land_hi - 0.5 + 1e-8), 0, 1)
    return elev * elev * (3 - 2 * elev)


def carve_river_network(elev, rng, river_frac=0.02, max_depth=0.09):
    """基于 D8 流量累积在陆地上沿河网刻蚀河谷，返回修正后的高程。"""
    acc = d8_flow_accumulation(elev, rng)
    thr = np.quantile(acc, 1 - river_frac)
    if thr < 2.0:
        return np.clip(elev, 0, 1).astype(np.float32)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    t[acc <= 1.5] = 0
    landness = np.clip((elev - 0.46) / 0.08, 0, 1)
    return np.clip(elev - max_depth * (t ** 0.6) * landness, 0, 1).astype(np.float32)


class SyntheticDEMSource:
    """兜底: v4 式合成地形, 转'米'后接口与 GlobalDEM 完全一致"""
    def __init__(self, n_tiles=16, tile=288, seed=123):
        """``__init__``: 预生成 n_tiles 块合成地形瓦片作为兜底数据源。"""
        print(f"未找到真实 DEM → 使用合成地形兜底 ({n_tiles} 块 {tile}px, 仅供流程验证)")
        self.tile = tile
        self.tiles = [self._make_tile(tile, seed + i) for i in range(n_tiles)]
        self.ny, self.nx = 4096, 4096
        self.land_low = None

    @staticmethod
    def _make_tile(size, seed):
        """生成单块 size×size 合成地形（米制高程），含陆地/海洋/褶皱山带与河流侵蚀。"""
        rng = np.random.RandomState(seed)
        base = value_fbm(size, 2.0, 4, rng=rng)
        base = warp_field(base, value_fbm(size, 4, 3, rng=rng),
                          value_fbm(size, 4, 3, rng=rng), size * rng.uniform(0.10, 0.18))
        base += 0.40 * value_fbm(size, 10, 4, rng=rng)
        lm = base > np.quantile(base, 1 - rng.uniform(0.34, 0.58))
        lm = grow_peninsulas(lm, rng)
        d_sea = distance_transform_edt(lm)
        d_land = distance_transform_edt(~lm)
        ocean = 0.5 - (0.03 + 0.09 * np.clip(d_land / 4.0, 0, 1)
                       + np.clip((d_land - 3.75) / 6.0, 0, 1) * (0.30 + 0.08 * value_fbm(size, 5, 4, rng=rng)))
        ridged = ridged_fbm(size, direction=rng.uniform(0, 180), rng=rng)
        belt = gaussian_filter(generate_folded_belt(size, rng), 2.0)
        belt /= (belt.max() + 1e-8)
        land = (0.04 + 0.10 * np.clip(value_fbm(size, 6, 5, rng=rng), 0, 1)
                + (ridged ** 1.7) * (belt ** 1.2) * rng.uniform(0.30, 0.40))
        land *= (0.35 + 0.65 * np.clip(d_sea / 5.0, 0, 1))
        v01 = np.where(lm, 0.5 + land, ocean).astype(np.float32)
        v01 = hydraulic_erosion(v01, iterations=400, dt=0.05)
        v01 = np.where(lm, np.maximum(v01, 0.525), np.minimum(v01, 0.475))
        v01 = np.clip(tail_stretch(v01, lm), 0, 1)
        return v2m(carve_river_network(v01, rng))

    def patch(self, y0, x0, size):
        """从随机瓦片裁剪出 size×size 合成高程块，并返回对应的纬度数组。"""
        rng = np.random.RandomState((y0 * 7919 + x0) % (2 ** 31))
        t = self.tiles[rng.randint(len(self.tiles))]
        oy, ox = rng.randint(0, self.tile - size + 1, 2)
        lat_c = -55 + 110 * ((y0 * 37 + x0 * 13) % 256) / 255.0
        return t[oy:oy + size, ox:ox + size], np.linspace(lat_c - 3, lat_c + 3, size)


# ==================== 5. 样本构建: 8通道低清条件 (A2/A3) ====================
# 通道: 0块均值 1块std 2块min 3块max 4陆占比 5河网 6冰川 7纬度
#       8边界邻近 9俯冲/碰撞 10离散(脊/裂谷) 11转换   ← v6.3 构造通道
def build_sample(elev_m, lats, block, rng, ch_stats, tect=None):
    """从米制高程构建 12 通道低清条件样本与归一化目标，返回 (ch, v01)。"""
    H, W = elev_m.shape
    v01 = m2v(elev_m)
    Bs = H // block
    bm = block_reduce(v01, block, "mean")
    bstd = block_reduce(v01, block, "std")
    bmin = block_reduce(v01, block, "min")
    bmax = block_reduce(v01, block, "max")
    lf = block_reduce((elev_m > 0).astype(np.float32), block, "mean")
    river = np.zeros((Bs, Bs), np.float32)
    if lf.max() > 0.05:
        lmask = lf > 0.05
        acc = d8_flow_accumulation(bm, rng)
        thr = max(float(np.quantile(acc[lmask], 0.99)), 2.0)
        river = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1) * lmask
    glac = block_reduce((elev_m > snowline_m(lats)[:, None]).astype(np.float32),
                        block, "mean")
    lat_rows = np.abs(lats[block // 2:: block]) / 90.0
    latn = np.broadcast_to(lat_rows[:, None], (Bs, Bs)).astype(np.float32)
    if tect is None:
        tect = np.zeros((4, H, W), np.float32)
    tch = np.stack([block_reduce(tect[i], block, "mean") for i in range(4)])
    ch = np.concatenate([np.stack([bm, bstd, bmin, bmax, lf, river, glac, latn]),
                         tch], 0).astype(np.float32)
    noise = [0.030, 0.020, 0.020, 0.020, 0.040, 0.040, 0.040, 0.0,
             0.030, 0.030, 0.030, 0.030]
    for c, s in enumerate(noise):
        if s > 0:
            ch[c] += rng.normal(0, s, ch[c].shape).astype(np.float32)
    ch = (ch - ch_stats["mean"][:, None, None]) / ch_stats["std"][:, None, None]
    return ch, v01


def estimate_channel_stats(dem, size, block, n=40, seed=77, tect_provider=None):
    """在 DEM 上随机采样 n 个位置，统计 12 通道条件的均值与标准差。"""
    rng = np.random.RandomState(seed)
    dummy = {"mean": np.zeros(12, np.float32), "std": np.ones(12, np.float32)}
    acc = []
    for _ in range(n):
        y0 = rng.randint(int(dem.ny * 0.05), int(dem.ny * 0.95) - size)
        x0 = rng.randint(0, dem.nx - size)
        m, lats = dem.patch(y0, x0, size)
        tect = tect_provider(y0, x0, size) if tect_provider else None
        ch, _ = build_sample(m, lats, block, np.random.RandomState(0), dummy, tect)
        acc.append(ch)
    acc = np.stack(acc)
    return {"mean": acc.mean(axis=(0, 2, 3)).astype(np.float32),
            "std": (acc.std(axis=(0, 2, 3)) + 1e-4).astype(np.float32)}


def make_tect_provider(dem):
    """真实 DEM → Bird2003 构造条件场; 合成兜底 → None(全零通道)"""
    if type(dem).__name__ != "GlobalDEM":
        return None
    from .tectonics import TectField, download_tectonic
    bjson = download_tectonic()
    proxy = dem.z[::16, ::16].astype(np.float32) * dem.scale + dem.offset
    tf = TectField(dem.ny, dem.nx, bjson, dem_for_proxy=proxy)
    return tf.patch


# ==================== 6. 数据集: 在线随机裁剪 + 陆地偏置 + 经度带切分 ====================
VAL_LON_LO, VAL_LON_HI = 130.0, 310.0   # 0-360 坐标: 130°E → 50°W = 太平洋带 (验证)


class TerrainDataset(Dataset):
    """在线随机裁剪的地形数据集：训练随机采样，验证固定太平洋带中心点。"""

    def __init__(self, dem, size, block, n_items, ch_stats, is_train, tect_provider=None):
        """``__init__``: 保存采样参数；非训练模式预生成固定中心点列表。"""
        self.dem, self.size, self.block = dem, size, block
        self.n_items, self.ch_stats, self.is_train = n_items, ch_stats, is_train
        self.tect_provider = tect_provider
        self.rng = np.random.RandomState(2024 if is_train else 0)
        self.centers = None
        if not is_train:
            rng = np.random.RandomState(1234)
            cands = []
            while len(cands) < n_items:
                y0, x0 = self._sample_center(rng)
                f = 0.5 if self.dem.land_low is None else \
                    self.dem.land_low[min(y0 // 16, self.dem.land_low.shape[0] - 1),
                                      min(x0 // 16, self.dem.land_low.shape[1] - 1)]
                cands.append((y0, x0, float(f)))
            cands.sort(key=lambda t: t[2])
            self.centers = cands

    def _sample_center(self, rng):
        """按经度带归属与陆地占比偏好，采样一个裁剪中心 (y0, x0)。"""
        ny, nx, size = self.dem.ny, self.dem.nx, self.size
        lo_y, hi_y = int(ny * 0.03), int(ny * 0.97) - size
        y0 = x0 = 0
        for _ in range(80):
            y0 = rng.randint(lo_y, max(lo_y + 1, hi_y))
            x0 = rng.randint(0, nx - size)
            l0 = (x0 + size / 2) * 360.0 / nx
            in_val = VAL_LON_LO <= l0 <= VAL_LON_HI
            if in_val != (not self.is_train):
                continue
            if self.dem.land_low is not None:
                f = self.dem.land_low[min(y0 // 16, self.dem.land_low.shape[0] - 1),
                                      min(x0 // 16, self.dem.land_low.shape[1] - 1)]
                p = 0.85 if f > 0.45 else (0.45 if f > 0.08 else 0.10)
                if rng.rand() > p:
                    continue
            return y0, x0
        return y0, x0

    def __len__(self):
        """返回数据集样本数。"""
        return self.n_items

    def __getitem__(self, idx):
        """返回第 idx 个样本：12 通道条件张量与目标高程张量。"""
        if self.is_train:
            y0, x0 = self._sample_center(self.rng)
            rng = self.rng
        else:
            y0, x0 = self.centers[idx][:2]
            rng = np.random.RandomState(9000 + idx)
        m, lats = self.dem.patch(y0, x0, self.size)
        tect = self.tect_provider(y0, x0, self.size) if self.tect_provider else None
        if self.is_train:
            if rng.rand() < 0.5:
                m = m[:, ::-1]
                if tect is not None:
                    tect = tect[:, :, ::-1]
            if rng.rand() < 0.5:
                m = m[::-1]; lats = lats[::-1]
                if tect is not None:
                    tect = tect[:, ::-1]
        ch, v01 = build_sample(m, lats, self.block, rng, self.ch_stats, tect)
        ch = np.ascontiguousarray(
            np.repeat(np.repeat(ch, self.block, axis=1), self.block, axis=2))
        return torch.from_numpy(ch).float(), torch.from_numpy(v01[None]).float()


# ==================== 7. 网络 ====================
class EMA:
    """指数滑动平均（EMA）模型副本，用于获取更稳定的推理权重。"""

    def __init__(self, model, decay=0.999):
        """``__init__``: 深拷贝 model 参数作为 EMA 影子权重。"""
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        """用当前模型参数更新影子权重（浮点参数按 decay 滑动平均）。"""
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


class ResBlock(nn.Module):
    """残差卷积块：两次卷积 + InstanceNorm + ReLU，带可选 Dropout。"""

    def __init__(self, ch, dropout=0.0):
        """``__init__``: 构建两个 3×3 卷积、InstanceNorm 与 Dropout2d 层。"""
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.InstanceNorm2d(ch)
        self.n2 = nn.InstanceNorm2d(ch)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        """前向传播：残差连接 + 双卷积，返回 ReLU 激活结果。"""
        h = F.relu(self.n1(self.c1(x)))
        h = self.drop(h)
        return F.relu(x + self.n2(self.c2(h)))


# ---- A4: resize-conv 上采样 (消除棋盘/半调伪影) ----
class Upsample(nn.Module):
    """resize-conv 上采样层：双线性放大 2 倍后接 3×3 卷积，抑制棋盘伪影。"""

    def __init__(self, cin, cout):
        """``__init__``: 构建放大后接的 3×3 卷积层。"""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)

    def forward(self, x):
        """前向传播：双线性插值放大 2 倍后卷积，返回输出特征。"""
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear',
                                       align_corners=False))


class SelfAttention(nn.Module):
    """自注意力模块：增强长程结构一致性，输出残差加权融合。"""

    def __init__(self, ch):
        """``__init__``: 构建 Q/K/V/投影卷积与可学习残差权重 gamma。"""
        super().__init__()
        qd = max(ch // 8, 8)
        self.q = nn.Conv2d(ch, qd, 1)
        self.k = nn.Conv2d(ch, qd, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.qd = qd

    def forward(self, x):
        """前向传播：计算注意力加权特征并做残差融合，返回同形状张量。"""
        B, C, H, W = x.shape
        q = self.q(x).view(B, self.qd, H * W).permute(0, 2, 1)
        k = self.k(x).view(B, self.qd, H * W)
        attn = torch.softmax(torch.bmm(q, k) * self.qd ** -0.5, dim=-1)
        v = self.v(x).view(B, C, H * W).permute(0, 2, 1)
        out = torch.bmm(attn, v).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.gamma * self.proj(out)


class UNet(nn.Module):
    """条件 GAN 的生成器 G：12 通道条件 → 单通道归一化高程（sigmoid 输出 ∈ [0,1]）。

    架构 (v5, 4 次下采样 + 4 次上采样，与 pix2pix 的 UNet 类似但更轻量)：
        - **编码器 stem + 4 层**：MaxPool(↓2) + Conv(2×ch) + ReLU + ResBlock；
          每一层输出 h1~h4 作为跳跃连接源（保留低频位置信息）。
        - **瓶颈层 bot**：ResBlock(8×b) + SelfAttention（允许长程结构一致性，
          用来避免 GAN 在山脉走向 / 海岸曲率上出现"左右不一致"的斑状伪影）。
        - **解码器 4 层**：Resize-Conv ``bilinear 上采样 → Conv``（代替 ConvTranspose，
          抑制棋盘格伪影），与对应层编码器 feature 在通道维度 concat，再 ResBlock。
        - **输出 head**：1×1 Conv + Sigmoid → 单通道 v∈[0,1]。

    推理用的权重（~44MB）存放在 ``checkpoints/best_terrain_gan_v5.pth``，其中
    ``cfg.in_ch=12, base=32``。训练时保存为 ``{"gen_ema", "cfg", "ch_stats_mean",
    "ch_stats_std", ...}`` 字典，其中 ``gen_ema`` 是指数滑动平均版本（视觉质量
    更稳定，不使用单步模型）。

    Parameters
    ----------
    in_ch : int
        输入条件通道数。v5 训练为 12；v6.x 以前模型可能为 8（AssertionError 提示重训）。
    base : int
        stem 基础通道数。默认 32 足以在 ~2GB 显存上跑 256² 训练 / 推理。
    dropout : float
        ResBlock 内 Dropout2d 比例。默认 0.05，防止 GAN 过拟合小细节导致 PSD 低频化。
    """
    def __init__(self, in_ch=8, base=32, dropout=0.05):
        """``__init__``: 按 in_ch/base 搭建编码器-瓶颈-解码器的 U 型生成网络。"""
        super().__init__()
        b = base
        self.stem = nn.Sequential(nn.Conv2d(in_ch, b, 3, 1, 1), nn.ReLU(True))
        self.e1 = ResBlock(b, dropout)
        self.p1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(b, 2 * b, 3, 1, 1), nn.ReLU(True))
        self.e2 = ResBlock(2 * b, dropout)
        self.p2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(2 * b, 4 * b, 3, 1, 1), nn.ReLU(True))
        self.e3 = ResBlock(4 * b, dropout)
        self.p3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(4 * b, 8 * b, 3, 1, 1), nn.ReLU(True))
        self.e4 = ResBlock(8 * b, dropout)
        self.p4 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(8 * b, 8 * b, 3, 1, 1), nn.ReLU(True))
        self.bot = ResBlock(8 * b, dropout)
        self.attn = SelfAttention(8 * b)
        self.u4 = Upsample(8 * b, 4 * b)
        self.d4 = nn.Sequential(nn.Conv2d(12 * b, 4 * b, 3, 1, 1), nn.ReLU(True), ResBlock(4 * b, dropout))
        self.u3 = Upsample(4 * b, 2 * b)
        self.d3 = nn.Sequential(nn.Conv2d(6 * b, 2 * b, 3, 1, 1), nn.ReLU(True), ResBlock(2 * b, dropout))
        self.u2 = Upsample(2 * b, b)
        self.d2 = nn.Sequential(nn.Conv2d(3 * b, b, 3, 1, 1), nn.ReLU(True), ResBlock(b, dropout))
        self.u1 = Upsample(b, b)
        self.d1 = nn.Sequential(nn.Conv2d(2 * b, b, 3, 1, 1), nn.ReLU(True), ResBlock(b, dropout))
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        """前向传播：UNet 编码-解码加跳跃连接，sigmoid 输出 [0,1] 高程图。"""
        h1 = self.e1(self.stem(x))
        h2 = self.e2(self.p1(h1))
        h3 = self.e3(self.p2(h2))
        h4 = self.e4(self.p3(h3))
        b0 = self.attn(self.bot(self.p4(h4)))
        d4 = self.d4(torch.cat([self.u4(b0), h4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), h3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), h2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), h1], 1))
        return torch.sigmoid(self.out(d1))


class PatchDiscriminator(nn.Module):
    """PatchGAN 判别器：输出 patch 级真假分数图，可选返回中间特征。"""

    def __init__(self, in_channels=9):
        """``__init__``: 构建谱归一化卷积堆叠的 PatchGAN 判别网络。"""
        super().__init__()

        def blk(cin, cout, norm=True):
            """构建一个谱归一化卷积块（可选 InstanceNorm + LeakyReLU）。"""
            layers = [nn.utils.spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1))]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

        self.b1 = blk(in_channels, 32, norm=False)
        self.b2 = blk(32, 64)
        self.b3 = blk(64, 128)
        self.out = nn.utils.spectral_norm(nn.Conv2d(128, 1, 4, 1, 1))

    def forward(self, x, return_feat=False):
        """前向传播：逐级下采样卷积，返回分数图（可选带中间特征）。"""
        f1 = self.b1(x)
        f2 = self.b2(f1)
        o = self.out(self.b3(f2))
        return (o, f2) if return_feat else o


def downsample(x, factor):
    """当 factor>1 时对 x 做平均池化降采样，否则原样返回。"""
    return x if factor <= 1 else F.avg_pool2d(x, factor)


# ==================== 8. 损失 ====================
class EdgeLoss(nn.Module):
    """Sobel 边缘损失：比较真假图在 x/y 方向边缘响应的 L1 距离。"""

    def __init__(self):
        """``__init__``: 注册 Sobel 水平/垂直卷积核为缓冲区。"""
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, fake, real):
        """前向传播：返回真假图 Sobel x/y 边缘响应之差的 L1 损失。"""
        return (F.l1_loss(F.conv2d(fake, self.kx, padding=1), F.conv2d(real, self.kx, padding=1))
                + F.l1_loss(F.conv2d(fake, self.ky, padding=1), F.conv2d(real, self.ky, padding=1)))


class FFTLoss(nn.Module):
    """MPS 对复数 FFT 支持不完整 → 首次失败自动切换到 CPU 计算"""
    def __init__(self):
        """``__init__``: 初始化设备可用标记（MPS 复数 FFT 兼容）。"""
        super().__init__()
        self._device_ok = True

    def forward(self, fake, real):
        """前向传播：比较频谱幅值/相位的 L1 损失，FFT 失败时自动切 CPU 计算。"""
        if self._device_ok:
            try:
                ff, rf = torch.fft.fft2(fake), torch.fft.fft2(real)
                return (F.l1_loss(torch.abs(ff), torch.abs(rf))
                        + 0.1 * F.l1_loss(torch.angle(ff), torch.angle(rf)))
            except (RuntimeError, TypeError):
                self._device_ok = False
        f2, r2 = fake.cpu(), real.cpu()
        ff, rf = torch.fft.fft2(f2), torch.fft.fft2(r2)
        loss = (F.l1_loss(torch.abs(ff), torch.abs(rf))
                + 0.1 * F.l1_loss(torch.angle(ff), torch.angle(rf)))
        return loss.to(fake.device)


class LaplacianLoss(nn.Module):
    """拉普拉斯损失：比较真假图的二阶导数响应的 L1 距离。"""

    def __init__(self):
        """``__init__``: 注册 3×3 拉普拉斯核为缓冲区。"""
        super().__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', k)

    def forward(self, fake, real):
        """前向传播：返回真假图拉普拉斯响应之差的 L1 损失。"""
        return F.l1_loss(F.conv2d(fake, self.kernel, padding=1),
                         F.conv2d(real, self.kernel, padding=1))


class DistributionLoss(nn.Module):
    """W1 距离 + 极值分位匹配, 直接惩罚'削峰填谷'"""
    def forward(self, fake, real):
        """前向传播：排序后计算 W1 距离与极值分位 L1 损失之和。"""
        B = fake.shape[0]
        fs = torch.sort(fake.reshape(B, -1), dim=1).values
        rs = torch.sort(real.reshape(B, -1), dim=1).values
        wass = torch.mean(torch.abs(fs - rs))
        N = fs.shape[1]
        idx = [int(q * (N - 1)) for q in (0.005, 0.01, 0.03, 0.97, 0.99, 0.995)]
        return wass + 2.0 * F.l1_loss(fs[:, idx], rs[:, idx])


class SlopeLoss(nn.Module):
    """坡度分布损失：对齐梯度幅值的均值与上十分位均值，针对"山被摊平"。

    只匹配分布统计量而不逐点匹配梯度——真实地形的高频细节本就不可知，
    强行逐点对齐会把 GAN 拉向条件输入的块均值。实测真值陆地 ≥5° 坡面占
    13.5%，v5(block=8) 仅 3.0%，该损失直接压这一项。
    """

    def __init__(self, top_frac=0.10):
        """``__init__``: top_frac 为参与"陡坡"统计的幅值分位比例。"""
        super().__init__()
        self.top_frac = top_frac

    @staticmethod
    def _grad_mag(x):
        """中心差分梯度幅值 (B,1,H-1,W-1)。"""
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.sqrt(gx[:, :, :-1, :] ** 2 + gy[:, :, :, :-1] ** 2 + 1e-12)

    def forward(self, fake, real):
        """前向传播：返回梯度幅值均值差 + 上十分位均值差。"""
        gf, gr = self._grad_mag(fake), self._grad_mag(real)
        k = max(1, int(gf[0].numel() * self.top_frac))
        tf = torch.topk(gf.flatten(1), k, dim=1).values.mean(1)
        tr = torch.topk(gr.flatten(1), k, dim=1).values.mean(1)
        return (gf.mean() - gr.mean()).abs() + (tf - tr).abs().mean()


def tile_boundary_loss(gen, inp, fake, shift=64, band=64):
    """瓦片边界一致性：同一内容在不同"离边界距离"下的预测应一致。

    把输入左裁 shift 列再前向一次，将原始图的 [shift+band, shift+2band)
    与裁剪图的 [band, 2band) 比较（两者对应同一段内容，但距各自边界的
    距离不同）。实测 v5 的接缝来自 GAN 原始输出（竖/横 1.20~1.26x），
    后处理只是有时把它压平，故该损失作用于模型本身。
    """
    crop = inp[..., shift:]
    p = gen(crop)
    return F.l1_loss(p[..., band:2 * band], fake[..., shift + band:shift + 2 * band])


# ==================== 9. 后处理套件 (A6) ====================
def sharpen_coastline(elev, band=0.025, expand=0.06):
    """单调映射: 0.5=精确海平面, 附近梯度×2.4 → 海岸锐化"""
    return np.interp(elev, [0.0, 0.5 - band, 0.5 + band, 1.0],
                     [0.0, 0.5 - expand, 0.5 + expand, 1.0]).astype(np.float32)


def lowfreq_consistency(v01, bm, block=8, iters=3):
    """引导式低频一致性校正：迭代把 GAN 输出的块均值精确拉回粗底图。

    GAN 在逐瓦片推理时容易出现"云状漂移"（深海整体变浅 / 平原无端起丘），
    而粗底图 ``bm`` 来自构造骨架，其低频是 **权威** 的。本函数执行以下修复循环：

        for each iter:
          1. 对输出 ``out`` 做 block_reduce(mean) → 得到当前块均值 bm_out
          2. ``resid = bm - bm_out``（粗图尺寸的残差）
          3. 对残差上采样 + 高斯平滑（σ=block/3，避免在块边界出现阶跃）
          4. ``out += smoothed resid``（向 bm 方向回拉一块）

    3 次迭代后块均值 RMSE 通常 < 0.001，且高频纹理被高斯核保留。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, [0, 1]
        GAN 输出的归一化高程。
    bm : ndarray (B, B)
        粗块目标均值（来自 ``world_to_channels`` 的同一张骨架图）。
    block : int
        块尺寸，必须与 ``bm`` 生成时的 block 一致（v5=8）。
    iters : int
        迭代次数。默认 3 足够稳定；更多 ≈ 更死锁骨架但纹理更木。

    Returns
    -------
    ndarray (S, S), float32, [0, 1]
        低频对齐后的输出。
    """
    out = np.clip(v01, 0, 1).astype(np.float32)
    for _ in range(iters):
        resid = bm - block_reduce(out, block, "mean")
        out = out + gaussian_filter(ndi_zoom(resid, block, order=1), sigma=block / 3.0)
    return np.clip(out, 0, 1).astype(np.float32)


def envelope_quantile_match(v01, bm, bstd, blend=0.7, seed=0):
    """直方图规格化到'分块高斯混合包络分布'。
    平原尖峰/深海尖峰/山峰厚尾同时恢复; 单调映射不破坏空间结构。"""
    rng = np.random.default_rng(seed)
    h, w = v01.shape
    n = h * w
    idx = rng.integers(0, bm.size, n)
    samples = np.clip(rng.normal(bm.ravel()[idx], bstd.ravel()[idx] + 1e-4), 0, 1)
    target_sorted = np.sort(samples)
    order = np.argsort(v01, axis=None)
    ranked = np.empty(n, np.int64)
    ranked[order] = np.arange(n)
    matched = target_sorted[ranked].reshape(h, w)
    return np.clip(blend * matched + (1 - blend) * v01, 0, 1).astype(np.float32)


def suppress_hallucinated_land(v01, lf, block=8, land_thresh=0.02, cap=0.46):
    """逐像素: 真实陆占比≈0 的深海块, 钳制输出不超过近海面 (消灭凭空浅滩)"""
    lf_up = ndi_zoom(lf, block, order=1)[:v01.shape[0], :v01.shape[1]]
    m = lf_up < land_thresh
    return np.where(m, np.minimum(v01, cap), v01).astype(np.float32)


def _lon_fbm1d(w, sigma=90.0, amp=1.0, seed=7):
    """沿经度方向生成周期 1D fBm (rFFT 高斯低通), 输出零均值、std≈amp.

    用于极地冰盖 / 海冰边界的**经向扰动** —— 纯纬度阈值在地图上就是一条水平直尺边.
    """
    r = np.random.RandomState(seed)
    X = np.fft.rfft(r.randn(w))
    X *= np.exp(-(np.fft.rfftfreq(w) * sigma) ** 2)
    n = np.fft.irfft(X, w)
    return amp * (n / (n.std() + 1e-6)).astype(np.float64)


def polar_ice_mask(lats, v01, sl_tab=None):
    """v7.1 极地冰盖掩膜: 66.5°→72° 过渡, 不再"一刀切"陆地全冰 (问题 #6).

    改动 (地球对应):
      - 66.5° 以下: 完全按纬度雪线 (无强制) → 允许阿拉斯加/北欧 62-66° 仍有森林
      - 66.5°~72° 过渡带: 仅 elev > (雪线 − 400m 软修正) 的高地自动冰,
        低地仍可为苔原 (对应地球斯瓦尔巴 78°N 低地仍有苔原, 格陵兰内陆才冰盖)
      - >72°: 全部陆地强制冰盖 (对应格陵兰冰盖主体 / 南极圈全冰)
    海洋 (v01<0.5) 不受强制 (海冰由纬度 fBm 门控单独渲染).

    ``sl_tab``: 雪线表 (米, 与 ``_SL_LAT`` 对齐)。默认用现行 ``_SL_M``；
    河谷下切/侵蚀两条调用点传入 ``_SL_INCISION_M``（A1b 前的老表），
    使雪线重标定只改渲染与审计口径、不污染 DEM 与 B1 已验收的地形。"""
    tab = _SL_M if sl_tab is None else sl_tab
    sl = m2v(np.interp(np.abs(np.asarray(lats, np.float64)), _SL_LAT, tab))[:, None]
    _w = v01.shape[1]
    # v46: 纬度阈值叠加经向 fBm 扰动 (σ≈2.2°), 冰盖南界不再是水平直边
    a = np.abs(lats)[:, None] + 2.2 * _lon_fbm1d(_w, 70.0, 1.0, seed=17)[None, :]
    land = v01 >= 0.5
    ice = v01 >= sl
    # 北极圈附近 66.5-72° 渐进: 低地保留为苔原 (雪线下 400m 豁免)
    gate_66 = np.clip((a - 66.5) / 5.5, 0.0, 1.0)         # 66.5→0, 72→1
    soft_ice = (v01 >= (sl - m2v(400.0)))                 # 雪线下 400m 近似 "极地平顶/丘陵"
    ice |= (gate_66 > 0) & land & np.where(gate_66 >= 0.5, True, soft_ice)
    ice |= (a >= 72.0) & land                             # 72° 以上: 彻底格陵兰式冰盖
    return ice


def render_snow_field(v01, lats, cmap):
    """v7.1 雪+冰渲染: 放宽 66.5° 全冰 (允许 66-72° 苔原低地)"""
    h, w = v01.shape
    sl01 = m2v(snowline_m(lats))[:, None]
    # v83（用户 v82 第 4 条「底部冰盖边缘是一条平滑波浪线，像一块白布盖在地图底部」）：
    #   旧版这里用**纯纬度水平线**（``a = |lat|``）做雪线/海冰门控，只有 polar_ice_mask
    #   里加了 2.2° 经向扰动，渲染出的冰缘仍是一条干净的等纬度曲线。
    #   现改为多尺度经向扰动：长波 90px（±2.6°，冰架前缘的大尺度凹凸）
    #   + 中波 22px（±1.1°，峡湾/冰舌），雪线、强制冰盖、海冰共用同一条扰动边界，
    #   海冰再叠一层 2D 低频噪声 → 前缘崩解成"海冰—浮冰—开阔洋"的碎边。
    lon_p = (2.6 * _lon_fbm1d(w, 90.0, 1.0, seed=23)
             + 1.1 * _lon_fbm1d(w, 22.0, 1.0, seed=24))
    a = np.abs(lats)[:, None] + lon_p[None, :]
    sea = v01 < 0.5
    land = ~sea
    snow_f = (v01 >= sl01).astype(np.float32)
    # ---- v7.1 极地过渡: 66.5-72° 仅 elev>(雪线-400m) 的极地平顶强制雪; 低地留苔原色 ----
    gate_66 = np.clip((a - 66.5) / 5.5, 0.0, 1.0)
    soft_ice = v01 >= (sl01 - m2v(400.0))
    polar_force = land & ((gate_66 >= 0.5) | (soft_ice & (gate_66 > 0)))
    snow_f[polar_force] = np.maximum(snow_f[polar_force], 1.0)
    # 72°以上彻底冰盖
    snow_f[(a >= 72.0) & land] = 1.0
    # 海冰 (极地带+纬度阈值 fBm 软边 + 2D 噪声碎边)
    _bn = value_fbm(h, 26.0, 3, rng=np.random.RandomState(31))
    sea_ice_f = ((a + 1.2 * _bn >= 66.5) & sea).astype(np.float32)
    snow_m = gaussian_filter(snow_f, 0.8) > 0.5          # 抑制阈值椒盐
    sea_ice = (gaussian_filter(sea_ice_f, 0.8) > 0.5) & ~snow_m
    rgb = cmap(np.clip(v01, 0, 1))[..., :3]
    rgb[snow_m] = rgb[snow_m] * 0.30 + 0.70
    rgb[sea_ice] = rgb[sea_ice] * 0.35 + np.array([0.88, 0.93, 0.97]) * 0.65
    return np.clip(rgb, 0, 1)


# ==================== 9b. 气候因果链: 纬度温度 → 地形降水 → Whittaker 生物群系 ====================




# 纬向基础降水 (mm/yr): ITCZ 赤道峰 → 副热带(20°~30°)干旱谷 → 温带峰 → 极地干










# ==================== 9c. 水文: 填洼 + 粒子水力侵蚀 ====================




def postprocess_pipeline(v01, bm, bstd, lf, lats, rng, block=8):
    """G raw → 低频校正 → 包络分位匹配 → 低频复核 → 幻觉陆地钳制
    → 海岸陡化 → 冰盖感知的 D8 河谷强调"""
    s = lowfreq_consistency(v01, bm, block)
    s = envelope_quantile_match(s, bm, bstd, blend=0.7, seed=int(rng.integers(1e9)))
    s = lowfreq_consistency(s, bm, block, iters=1)
    s = suppress_hallucinated_land(s, lf, block)
    s = sharpen_coastline(s)
    ice = polar_ice_mask(lats, s, _SL_INCISION_M)
    acc = d8_flow_accumulation(s, rng)
    thr = max(float(np.quantile(acc, 0.985)), 2.0)
    t = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((s - 0.5) / 0.02, 0, 1)
    s = np.clip(s - 0.03 * (t ** 0.8) * landness * (~ice), 0, 1)
    return s.astype(np.float32)


# ==================== 9d. v6.6 检验修复层 (七大地理维度对齐) ====================
# 地球陆地高程分位数 —— ETOPO1(Ice) 2048 实测 (含南极/格陵兰冰盖)
# v44 修复尾部口径: 旧表 q=1.0 的结点取的是"最高 2.5% 桶的面积加权均值"
# (6350.8m), 拿它当分位端点 → 分位重排后整个世界最高点被钉死在 6351m,
# 地球最高峰 8848m 永远不可能出现 (supervisor 一直报 peak warn 6351 vs 8848).
# B1 再修尾部口径: 原 4 段尾表把 0.9999→7600m 摊在 0.1% 陆地(≈1176px)上,
# 实测 >6000m 陆地 1527px / >8000m 43px (地球 6000m 以上 <0.05% 陆地)。
# 现拆成 6 段把 6000~8848m 压缩到 0.01% 陆地, 单像素极点仍可达珠峰:
#   0.995 → ~4800m (高原面) / 0.999 → ~5600m (喜马拉雅主体)
#   0.99985 → ~6000m / 0.99995 → ~7000m / 0.99999 → ~8000m / 1.0 → 8848m
_LAND_Q = np.concatenate([np.arange(0, 0.9751, 0.025),
                          np.array([0.99500, 0.99900, 0.99985,
                                    0.99995, 0.99999, 1.00000])])
_LAND_V = np.array([0.0, 14.0, 31.0, 49.0, 68.0, 88.0, 110.0, 130.0, 150.0,
                    170.0, 200.0, 230.0, 260.0, 290.0, 320.0, 350.0, 390.0,
                    420.0, 470.0, 520.0, 570.0, 640.0, 720.0, 820.0, 930.0,
                    1000.0, 1200.0, 1300.0, 1400.0, 1600.0, 1900.0, 2100.0,
                    2300.0, 2500.0, 2700.0, 2800.0, 3000.0, 3100.0, 3300.0,
                    3700.0, 4850.0, 5650.0, 6050.0, 7050.0, 8070.0,
                    8848.0], np.float64)
# 同上, 但按 cos(纬度) 面积加权 (与验收口径一致; 南极冰盖降权)
_LAND_W_V = np.array([0.0, 13.6, 30.5, 48.1, 65.5, 83.7, 101.6, 119.3,
                      137.2, 155.2, 174.3, 195.2, 217.7, 241.9, 266.6,
                      289.8, 312.5, 337.5, 363.9, 392.2, 420.1, 451.5,
                      486.2, 526.8, 572.7, 625.6, 686.5, 758.5, 842.5,
                      932.3, 1024.4, 1119.4, 1225.3, 1358.4, 1531.9,
                      1775.2, 2097.6, 2507.4, 2995.3, 3658.8,
                      4800.0, 5600.0, 6000.0, 7000.0, 8000.0, 8848.0],
                     np.float64)


def land_hist_match(elev_m, lats=None):
    """P0 陆地直方图重映射: 排名保持(单调) → 陆地高程分布对齐地球实测。
    修复 GAN"回归均值"(89%陆地挤在200~1000m) → 28%平原/46%台地/26%山地。
    单调映射不改变海岸线与空间秩序: 骨架最低的大片陆地→平原, 最高→山带。
    lats 给定时按 cos(纬度) 加权分位(目标 _LAND_W_V) → 与面积加权验收口径一致。"""
    land = elev_m > 0
    if not land.any():
        return elev_m
    lv = elev_m[land].astype(np.float64)
    order = np.argsort(lv)
    if lats is not None:
        coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)     # (H,)
        rows = np.nonzero(land)[0]                               # 陆地像素行号(行优先)
        wl = coslat[rows][order]
        wc = np.cumsum(wl)
        wc = (wc - 0.5 * wl) / wc[-1]
        tgt = np.interp(wc, _LAND_Q, _LAND_W_V)
    else:
        wc = (np.arange(lv.size) + 0.5) / lv.size
        tgt = np.interp(wc, _LAND_Q, _LAND_V)
    out = elev_m.copy()
    out[land] = tgt[order.argsort()].astype(np.float32)
    return out.astype(np.float32)


def ocean_margin_fix(elev_m, tect1_up, margin_up=None,
                     km_shelf=250.0, km_slope=70.0, km_rise=250.0,
                     transition_km=15.0):
    """P0 陆架-陆坡-陆隆三段剖面修复（海岸噪声破坏近岸剖面后重新强制）。

    三段几何（v45 按地球实测坡度重构；B3 加宽陆架/陆坡）：
      1) **陆架** 0 → −145m，宽 ``km_shelf``=250km（≈12.8px@2048），内浅外深
         （``d**1.5`` 使内侧近乎平躺，平均坡度 ~0.1°，与地球陆架一致）。
         B3 分两步：先 115→160km 实测浅海只从 5.17%→5.42%（+0.26pp），
         收益不足；据终稿 DEM 的敏感性外推，一步到 250km 才能把浅海推到
         ~8%（地球 7.6%），故定稿 250km。
      2) **陆坡** −145 → −2100m，宽 ``km_slope``=70km（≈3.6px），坡度
         ≈28 m/km = **2.8%**（地球陆坡 3~6%）。这是"坡折"能否被看见的关键：
         v44 及以前 km_slope=175km，同样落差摊到 196km 上只有 1.1%，
         在 v 域里就是一条长而平的渐变，图上完全读不出坡折；
      3) **陆隆** −2100 → −4600m，宽 ``km_rise``=250km（≈12.8px，地球 300~500km），
         之后释放约束由 ``ocean_hist_final`` 接管深渊平原。

    v3.0 per-margin 分化: margin_up=1 被动边缘(大西洋型宽陆架 km_shelf)
    / =2 主动边缘(太平洋型窄陆架 35km+陡坡) / =0 或 None 用默认宽。
    俯冲核带(tect1高)与已有深结构(>-4600m)不抬升 → 保海沟与深渊平原。

    v8.0 平滑过渡: 旧版 ``np.where(d < X, ...)`` 在三个剖断点处产生 200-500m 阶跃,
    在岛屿周围形成同心环方块伪影. 新版用 ``np.tanh`` 软过渡
    (带宽 ``transition_km`` 默认 15km ≈ 0.8px@2048) 在剖断点处 C1 连续."""
    S = elev_m.shape[0]
    land = elev_m > 0
    kmpp = 360.0 / S * 111.32
    d = distance_transform_edt(~land) * kmpp

    def shelf_profile(km_s, km_p):
        """按距岸距离生成陆架-陆坡-陆隆三段 tanh 平滑目标剖面深度场。"""
        # 三段目标深度 (各自 clip 到本段范围, 超出段端点保持端点值)
        # 陆架: d**1.5 让内侧近乎平躺 (地球陆架内缘坡度 ~0.05°, 外缘 ~0.3°)
        p_shelf = -12.0 - 133.0 * np.clip(d / km_s, 0.0, 1.0) ** 1.5
        p_slope = -145.0 - 1955.0 * np.clip((d - km_s) / km_p, 0.0, 1.0)
        p_rise = -2100.0 - 2500.0 * np.clip((d - km_s - km_p) / km_rise, 0.0, 1.0)
        # tanh 软过渡权重 (在 d=km_s / km_s+km_p / km_s+km_p+km_rise 处平滑切换)
        # 注意: w_shelf + w_slope + w_rise 在三段并集内部 = 1, 但在陆隆末端
        # (d≈km_s+km_p+km_rise) 处 → 0.5 (因为 "beyond" 区开始接管). 因此需要归一化,
        # 否则 profile 会降至 p_rise 的一半 (e.g. -2250m 而非 -4600m), 错误填平深海.
        t = transition_km
        w_shelf = 0.5 * (1.0 - np.tanh((d - km_s) / t))
        w_slope = 0.5 * (np.tanh((d - km_s) / t)
                         - np.tanh((d - km_s - km_p) / t))
        w_rise = 0.5 * (np.tanh((d - km_s - km_p) / t)
                        - np.tanh((d - km_s - km_p - km_rise) / t))
        denom = w_shelf + w_slope + w_rise
        # 归一化: 在三段并集内 denom≈1 → profile 为加权平均; 在陆隆末端
        # denom→0 → 超出约束区, 返回 -1e9 (不约束)
        profile = np.where(denom > 1e-3,
                           (w_shelf * p_shelf + w_slope * p_slope + w_rise * p_rise)
                           / np.maximum(denom, 1e-6),
                           -1.0e9)
        return profile

    prof = shelf_profile(km_shelf, km_slope).astype(np.float32)
    ok = (~land) & (tect1_up < 0.30) & (elev_m > -4600.0)
    # v8.0 软地板 (soft floor): 在陆隆末端 (d≈km_s+km_p+km_rise) 处, 旧版 np.maximum
    # 从 -4500m 硬切换到 "不约束" (原 elev_m), 若原海深 -8000m 则产生 3500m 阶跃.
    # 新版用 w_constraint 权重把 max(elev, prof) 与 elev 平滑混合:
    #   内部 (w=1): max(elev, prof) — 硬地板 (与旧版相同)
    #   末端 (w→0): elev — 释放约束, 让海床自然深下去
    #   过渡 (0<w<1): w*max(elev, prof) + (1-w)*elev — 渐进释放
    w_constraint = 0.5 * (1.0 - np.tanh(
        (d - (km_shelf + km_slope + km_rise)) / transition_km)).astype(np.float32)
    floor = np.maximum(elev_m, prof)
    out = np.where(ok, w_constraint * floor + (1.0 - w_constraint) * elev_m, elev_m)
    if margin_up is not None:
        # 主动边缘: 窄陆架 35km + 陡坡 (替换宽剖面, 太平洋型; 坡度 4.6%)
        act = (margin_up >= 2) if margin_up.shape == elev_m.shape else None
        if act is None:
            z = np.zeros(elev_m.shape, bool)
            z[:min(margin_up.shape[0], elev_m.shape[0]),
              :min(margin_up.shape[1], elev_m.shape[1])] = \
                margin_up[:elev_m.shape[0], :elev_m.shape[1]] >= 2
            act = z
        if act.any():
            prof_a = shelf_profile(35.0, 40.0).astype(np.float32)
            ok_a = (~land) & act & (elev_m > -4600.0)
            w_constraint_a = 0.5 * (1.0 - np.tanh(
                (d - (35.0 + 40.0 + km_rise)) / transition_km)).astype(np.float32)
            floor_a = np.maximum(elev_m, prof_a)
            out = np.where(ok_a,
                           w_constraint_a * floor_a + (1.0 - w_constraint_a) * elev_m,
                           out)
    return out.astype(np.float32)


def add_island_arcs(elev_m, tect1_up, cf, rng):
    """P1 岛弧链 (日本/印尼型): 俯冲核带法向偏移 180~340km,
    取大陆侧(crust概率高)且仍为海的点盖章长条岛 → 弧后链状群岛。
    岛长4~11px/宽1.5~3px/峰500~2200m, 沿核线间隔采样 → 链状。
    ---- v7.4: 用户报告"群岛/岛弧数量明显不足"
      俯冲核带分位: 0.995 → 0.990 (放宽准入, 中等强度的俯冲带也能产生岛弧)
      核带最低阈值: 0.25 → 0.18 (进一步降低门槛, 生成 ~2倍 的岛弧链)
      盖章深度下限: -5200m → -5800m (也能在较深盆地边缘盖岛弧, 类似马里亚纳外侧弧形群岛)
    ---- v34: 上游造陆量瘦身回调 (ocean_-200_0 / lake_share 偏高, 被校准淹没成浅海)
      核带分位回 0.985、最低阈值回 0.25、盖章深度回 -5200m → 岛弧链明显减少"""
    from scipy.ndimage import label as ndi_label
    S = elev_m.shape[0]
    thr = float(np.quantile(tect1_up, 0.985))
    if thr <= 0:
        return elev_m
    core = tect1_up > max(thr, 0.25)
    lab, n = ndi_label(core)
    if n == 0:
        return elev_m
    g = gaussian_filter(tect1_up.astype(np.float32), 2.0)
    gy, gx = np.gradient(g)
    out = elev_m.copy()
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < 25:
            continue
        d_arc = rng.uniform(9.0, 17.0)
        L = rng.uniform(4.0, 11.0)
        W = rng.uniform(1.5, 3.0)
        n_st = max(2, ys.size // 4)
        for k in rng.choice(ys.size, size=min(ys.size, n_st), replace=False):
            y, x = int(ys[k]), int(xs[k])
            gn = float(np.hypot(gx[y, x], gy[y, x])) + 1e-9
            nx_, ny_ = gx[y, x] / gn, gy[y, x] / gn
            cands = []
            for sgn in (1.0, -1.0):
                oy = int(round(y + sgn * ny_ * d_arc))
                ox = int(round(x + sgn * nx_ * d_arc)) % S
                if 0 <= oy < S:
                    cands.append((cf[oy, ox], oy, ox))
            if not cands:
                continue
            _, oy, ox = max(cands)                    # 大陆侧
            if out[oy, ox] >= 0.0 or out[oy, ox] < -5200.0:   # v34: -5800→-5200 减少深盆地岛弧
                continue                              # 只要浅海盖章
            ang = rng.uniform(0, 2 * np.pi)           # 岛长轴随机倾斜
            tx, ty = -ny_, nx_
            ca, sa = np.cos(ang), np.sin(ang)
            ux, uy = tx * ca - ty * sa, tx * sa + ty * ca
            pad = int(L) + 3
            if ox - pad < 0 or ox + pad >= S or oy - pad < 0 or oy + pad >= S:
                continue
            yy, xx = np.mgrid[oy - pad:oy + pad + 1, ox - pad:ox + pad + 1]
            dy_, dx_ = yy - oy, xx - ox
            along = (dx_ * ux + dy_ * uy) / L
            across = (dx_ * -uy + dy_ * ux) / W
            H = rng.uniform(500.0, 2200.0) * rng.uniform(0.7, 1.3)
            blob = H * np.exp(-along ** 2 - across ** 2)
            m = (blob > 80.0) & (out[yy, xx] < 0)
            out[yy[m], xx[m]] += blob[m]
    return out


def add_hotspot_chains(elev_m, rng, n_chains=None):
    """P1 热点岛链 (夏威夷型): 开阔洋中随机方向 3~6 岛链,
    单岛半径1.6~3.2px, 峰300~1600m 随链龄衰减 → 线状岛链+海山。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    d_land = distance_transform_edt(~land)
    n_chains = n_chains if n_chains is not None else int(rng.randint(2, 4))  # v34: 3-6 → 2-4 减少热点碎岛
    out = elev_m.copy()
    placed, tries = 0, 0
    while placed < n_chains and tries < 300:
        tries += 1
        y = int(rng.randint(60, S - 60))
        x = int(rng.randint(0, S))
        if land[y, x] or d_land[y, x] < 18.0:
            continue
        placed += 1
        ang = rng.uniform(0, 2 * np.pi)
        dy, dx = np.sin(ang), np.cos(ang)
        n_isl = int(rng.randint(3, 5))   # v34: 3-7 → 3-5 单链岛数微减
        gap = rng.uniform(4.0, 8.0)
        H0 = rng.uniform(900.0, 1600.0)
        for j in range(n_isl):
            cy = int(round(y + dy * gap * j))
            cx = int(round(x + dx * gap * j)) % S
            if not (20 <= cy < S - 20) or cx < 6 or cx >= S - 6:
                break
            h = H0 * (1.0 - 0.55 * j / n_isl) * rng.uniform(0.7, 1.3)
            r = rng.uniform(1.6, 3.2)
            yy, xx = np.mgrid[cy - 5:cy + 6, cx - 5:cx + 6]
            blob = h * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (r * r)))
            m = blob > 60.0
            out[yy[m], xx[m]] += blob[m]
    return out


def carve_fjords(elev_m, lats, margin_up, rng, lat_min=54.0, n_fjords=None):
    """M3 峡湾雕刻 (斯堪的纳维亚/格陵兰型): 高纬被动边缘从海岸向内陆
    刻窄谷 (宽1-2px, 深80-300m, 长4-10px, 二级分叉) → 冰蚀海岸。
    仅 |lat|>54° 被动边缘带; U形谷横剖面。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    pas = (margin_up == 1) if margin_up is not None else None
    if pas is None or not pas.any():
        return elev_m
    # 高纬被动海岸线 = 陆地且邻海
    sea = ~land
    coast = land & pas & (np.abs(lats)[:, None] > lat_min) \
            & (np.roll(sea, 1, 0) | np.roll(sea, -1, 0)
               | np.roll(sea, 1, 1) | np.roll(sea, -1, 1))
    ys, xs = np.nonzero(coast)
    if ys.size == 0:
        return elev_m
    out = elev_m.copy()
    # 内陆方向场: 一次计算 (上坡梯度)
    g = gaussian_filter(out.astype(np.float32), 2.0)
    gy, gx = np.gradient(g)
    gn = np.hypot(gx, gy) + 1e-9
    ux_f, uy_f = gx / gn, gy / gn                # 上坡=内陆
    n_fjords = n_fjords if n_fjords is not None else int(min(ys.size, 900))
    pick = rng.choice(ys.size, size=n_fjords, replace=False)
    for p in pick:
        y, x = int(ys[p]), int(xs[p])
        ux, uy = ux_f[y, x], uy_f[y, x]
        L = rng.uniform(4.0, 10.0)
        depth = rng.uniform(80.0, 300.0)
        W = rng.uniform(1.2, 2.2)
        cy, cx = float(y), float(x)
        for t in range(int(L)):
            cy -= uy; cx += ux                    # 上坡 = 内陆
            iy, ix = int(round(cy)), int(round(cx)) % S
            if not (0 <= iy < S) or out[iy, ix] <= 0:
                break
            pad = int(W) + 2
            y0, y1 = max(0, iy - pad), min(S, iy + pad + 1)
            x0, x1 = max(0, ix - pad), min(S, ix + pad + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            r = np.hypot(yy - iy, xx - ix)
            cut = depth * np.clip(1.0 - (r / W) ** 2, 0, 1)  # U形
            m = (r < W * 1.4) & (out[y0:y1, x0:x1] > 60.0)   # 不切到海平面下
            out[y0:y1, x0:x1] = np.where(
                m, np.maximum(out[y0:y1, x0:x1] - cut * m, 15.0),
                out[y0:y1, x0:x1])
            # 二级分叉 (10%)
            if t > 2 and rng.rand() < 0.10:
                a2 = rng.uniform(0, 2 * np.pi)
                cy2, cx2 = cy, cx
                for _ in range(int(L * 0.5)):
                    cy2 += np.sin(a2); cx2 += np.cos(a2)
                    iy2, ix2 = int(round(cy2)), int(round(cx2)) % S
                    if not (0 <= iy2 < S) or out[iy2, ix2] <= 0:
                        break
                    y0b, y1b = max(0, iy2 - 3), min(S, iy2 + 4)
                    x0b, x1b = max(0, ix2 - 3), min(S, ix2 + 4)
                    yy2, xx2 = np.mgrid[y0b:y1b, x0b:x1b]
                    rr = np.hypot(yy2 - iy2, xx2 - ix2)
                    cut2 = depth * 0.6 * np.clip(
                        1.0 - (rr / (W * 0.7)) ** 2, 0, 1)
                    m2 = (rr < W) & (out[y0b:y1b, x0b:x1b] > 0)
                    out[y0b:y1b, x0b:x1b] = np.where(
                        m2, out[y0b:y1b, x0b:x1b] - cut2 * m2,
                        out[y0b:y1b, x0b:x1b])
    return out.astype(np.float32)


def add_shelf_peninsulas(elev_m, margin_up, cf, rng, lats=None,
                         n_pen=None, n_isl=None):
    """M3 陆架台地半岛 + 大陆岛 (阿拉伯/佛罗里达/不列颠型):
    被动边缘宽陆架台地上, 海平面微抬 +15m → 台地露头成半岛/岛。
    台地 = 被动边缘 0-90km 内 -60~+60m 平缓带 → 加 fBm 起伏后 ±15m 阈值化。"""
    S = elev_m.shape[0]
    land = elev_m > 0
    pas = (margin_up == 1) if margin_up is not None else None
    if pas is None or not pas.any():
        return elev_m, 0
    d = distance_transform_edt(~land)
    kmpp = 360.0 / S * 111.32
    d_km = d * kmpp
    shelf = (~land) & (pas | (d_km < 95.0)) & (elev_m > -170.0)
    if not shelf.any():
        return elev_m, 0
    # 台地起伏: 低频 fBm; 台地基面 = 陆架剖面等效 (-160m 基 + fBm ±70m)
    # (真实陆架多被 ocean_margin_fix 压至 -163m, 阈值需覆盖全陆架)
    # M6: 基频 8→11 (特征 σ128→93px) → 更多中型露头 (大岛数 72.7 vs 地球 113)
    tf = value_fbm(S, 11.0, 3, rng=rng)
    tfc = np.clip(tf, 0, 1)
    out = elev_m.copy()
    plateau = shelf & (elev_m > -165.0) & (d_km < 95.0)
    out = np.where(plateau, np.maximum(out, -160.0 + 130.0 * (tf + 0.5)),
                   out)
    # 海平面等效 +15m: 台地内 -15~+45m 的连通域成陆 (半岛/大陆岛)
    # 近岸带(<25km)阈值放宽(-25m) → 露头连片成大岛/半岛 (不列颠型)
    near = plateau & (d_km < 25.0)
    new_land = (plateau & (out > -15.0)) | (near & (out > -25.0))
    # 清碎屑: 只留 >12px 的露头 (真半岛/岛)
    from scipy.ndimage import label as _lbl
    lab, n = _lbl(new_land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        keep = np.isin(lab, np.nonzero(sz >= 12)[0])
        new_land = new_land & keep
        # 大陆岛增益: 离岸 8-60km 中型露头(20-4000px)抬高确保成岛
        offshore = new_land & (d * kmpp > 8.0) & (d * kmpp < 60.0)
        lab2, n2 = _lbl(offshore)
        if n2:
            sz2 = np.bincount(lab2.ravel()); sz2[0] = 0
            big = np.isin(lab2, np.nonzero((sz2 >= 20) & (sz2 <= 4000))[0])
            offshore = offshore & big
            out = np.where(offshore,
                           np.maximum(out, 8.0 + 45.0 * tfc),
                           out)
    out = np.where(new_land, np.maximum(out, 3.0 + 30.0 * tfc),
                   out).astype(np.float32)
    n_out = int(new_land.sum())
    return out, n_out


def add_coral_reefs(elev_m, lats, rng, temp_fn=None, n_reef=None):
    """M3 珊瑚礁带 (马尔代夫/太平洋环礁型): |lat|<28° 浅海(-60~0m)
    且海表温度>20°C 处, 低概率抬升小岛 (峰3-8m, r=1-2px, 群集3-8个)。
    v33: 减量 (420→160 簇, 簇内 3-8→2-4) —— 旧版在每片热带浅海制造上千个
    1-2px 微岛, 与陆架色带叠加呈 "陨石坑同心圆" 碎点, 而非地球的礁带."""
    S = elev_m.shape[0]
    sea_shallow = (elev_m > -60.0) & (elev_m <= 0.0) \
                  & (np.abs(lats)[:, None] < 28.0)
    if temp_fn is not None:
        sea_shallow &= temp_fn > 20.0
    ys, xs = np.nonzero(sea_shallow)
    if ys.size == 0:
        return elev_m
    out = elev_m.copy()
    n_reef = n_reef if n_reef is not None else int(min(ys.size * 0.004, 120))  # v34: 0.005/160 → 0.004/120
    # 礁盘中心: 稀疏采样 + 群集
    pick = rng.choice(ys.size, size=min(n_reef, ys.size), replace=False)
    for p in pick:
        cy, cx = int(ys[p]), int(xs[p])
        n_isl = int(rng.randint(2, 5))
        for _ in range(n_isl):
            oy = int(round(cy + rng.normal(0, 3.5)))
            ox = int(round(cx + rng.normal(0, 3.5))) % S
            if not (0 <= oy < S):
                continue
            if out[oy, ox] > 0 or out[oy, ox] < -60.0:
                continue
            h = rng.uniform(3.0, 8.0)
            r = rng.uniform(1.0, 2.0)
            y0, y1 = max(0, oy - 3), min(S, oy + 4)
            x0, x1 = max(0, ox - 3), min(S, ox + 4)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            blob = h * np.exp(-((np.hypot(yy - oy, xx - ox) / r) ** 2))
            m = (blob > 1.5) & (out[y0:y1, x0:x1] <= 0)
            out[y0:y1, x0:x1] = np.where(
                m, np.maximum(out[y0:y1, x0:x1],
                              (0.0 + blob * 0.6)),
                out[y0:y1, x0:x1])
    return out.astype(np.float32)


def add_intraplate_orogen(elev_m, cf, tect1_up, rng, n=None):
    """P2 板内造山 (落基山/东非裂谷肩型): 大陆内部远离活跃边界处
    盖1~2条板内山地: 长20~45px 宽4~9px 峰1600~3400m, 平顶脊状+噪声。
    打破"山地100%锁定俯冲带"的过纯耦合 (地球山地过半在板内)。"""
    S = elev_m.shape[0]
    n = n if n is not None else int(rng.randint(1, 3))
    out = elev_m.copy()
    cand = (cf > 0.62) & (tect1_up < 0.15) & (out > 0)
    for _ in range(n):
        ys, xs = np.nonzero(cand)
        if ys.size < 400:
            break
        k = int(rng.randint(ys.size))
        cy, cx = int(ys[k]), int(xs[k])
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(20.0, 45.0)
        W = rng.uniform(4.0, 9.0)
        H = rng.uniform(900.0, 1900.0)   # v38: 1600-3400→900-1900 (板内山不再抢攻高山位)
        ux, uy = np.cos(ang), np.sin(ang)
        pad = int(L) + 4
        yy, xx = np.mgrid[max(0, cy - pad):min(S, cy + pad + 1),
                          max(0, cx - pad):min(S, cx + pad + 1)]
        dy_, dx_ = yy - cy, xx - cx
        along = (dx_ * ux + dy_ * uy) / L
        across = (dx_ * -uy + dy_ * ux) / W
        ridge = np.exp(-along ** 4 - across ** 2)
        nf = value_fbm(max(yy.shape), 8.0, 3, rng=rng)
        noise = 0.65 + 0.35 * np.clip(
            nf[:yy.shape[0], :yy.shape[1]] * 1.6, -1.0, 1.5)
        add = H * ridge * noise
        m = (out[yy, xx] > 0) & (add > 60.0)
        out[yy[m], xx[m]] += add[m]
    return out




def land_hist_final(v01, lats=None):
    """P0 陆地直方图终检 (v01域): GAN blend/后处理漂移后,
    排序CDF重新对齐 m2v(地球陆地分位) → 平原/台地/山地比例精确复原。
    单调重排保留全部空间秩序(谷<脊, V谷细节不损), 陆地不越海平面。
    lats 给定时按 cos(纬度) 加权分位(目标 _LAND_W_V) → 与面积加权验收口径一致。"""
    land = v01 >= 0.5
    if not land.any():
        return v01
    n = int(land.sum())
    order = np.argsort(v01[land])
    if lats is not None:
        coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)     # (H,)
        rows = np.nonzero(land)[0]                               # 陆地像素行号(行优先)
        wl = coslat[rows][order]
        wc = np.cumsum(wl)
        wc = (wc - 0.5 * wl) / wc[-1]
        tgt = m2v(np.interp(wc, _LAND_Q, _LAND_W_V)).astype(np.float32)
    else:
        wc = (np.arange(n) + 0.5) / n
        tgt = m2v(np.interp(wc, _LAND_Q, _LAND_V)).astype(np.float32)
    out = v01.copy()
    out[land] = tgt[order.argsort()]
    return np.clip(out, 0, 1).astype(np.float32)


# 地球深洋 (深度<-1000m) cos(lat) 加权分位 —— ETOPO1(Ice) 2048 实测 (M6 标定)
# 尾部加密网格 (hadal 仅 1.085% 深洋: 粗网格线性插值会把 2% 分位错插到 -6766)
_OCD_Q = np.array([0.0000, 0.0010, 0.0020, 0.0030, 0.0050, 0.0070, 0.0090,
                   0.0110, 0.0130, 0.0160, 0.0200, 0.0240, 0.0250, 0.0500,
                   0.0750, 0.1000, 0.1250, 0.1500, 0.1750, 0.2000, 0.2250,
                   0.2500, 0.2750, 0.3000, 0.3250, 0.3500, 0.3750, 0.4000,
                   0.4250, 0.4500, 0.4750, 0.5000, 0.5250, 0.5500, 0.5750,
                   0.6000, 0.6250, 0.6500, 0.6750, 0.7000, 0.7250, 0.7500,
                   0.7750, 0.8000, 0.8250, 0.8500, 0.8750, 0.9000, 0.9250,
                   0.9500, 0.9750, 0.9760, 0.9840, 0.9880, 0.9910, 0.9930,
                   0.9950, 0.9970, 0.9990, 1.0000])
_OCD_V = np.array([-10517, -7239, -6657, -6353, -6157, -6082, -6034, -5997,
                   -5966, -5923, -5877, -5837, -5828, -5644, -5518, -5417,
                   -5330, -5249, -5173, -5102, -5032, -4962, -4892, -4820,
                   -4748, -4676, -4607, -4538, -4472, -4407, -4342, -4277,
                   -4210, -4140, -4067, -3992, -3915, -3837, -3757, -3673,
                   -3586, -3492, -3391, -3281, -3160, -3016, -2842, -2636,
                   -2356, -2000, -1550, -1530, -1363, -1277, -1210, -1165,
                   -1120, -1073, -1024, -1000], np.float64)


def ocean_hist_final(dem, lats=None):
    """M6 海洋直方图终检 (米域): 深洋(<-1000m)排序 CDF → 地球深洋加权分位。
    GAN v01 域饱和(0 → -11000m)与包络混合使深洋带漂移 (hadal 3.7% vs 0.96%);
    单调重排保结构 (海沟仍最深, 俯冲速率序保留), 陆架-陆坡带(>-1000m)不动。"""
    ocean = dem < -1000.0
    n = int(ocean.sum())
    if n < 1000:
        return dem
    order = np.argsort(dem[ocean])
    if lats is not None:
        rows = np.nonzero(ocean)[0]                      # 行优先 → 与 dem[ocean] 同序
        coslat = np.clip(np.cos(np.deg2rad(np.asarray(lats, np.float64))), 0.0, 1.0)
        w = coslat[rows][order]
        cw = np.cumsum(w)
        wc = ((cw - 0.5 * w) / cw[-1]).astype(np.float64)
    else:
        wc = (np.arange(n) + 0.5) / n
    tgt = np.interp(wc, _OCD_Q, _OCD_V).astype(np.float32)
    tmp = dem[ocean]
    tmp[order] = tgt
    out = dem.copy()
    out[ocean] = tmp
    return out.astype(np.float32)


# ==================== 9e. M7 海底地貌增强 ====================
def enhance_ocean_floor(dem_m, lats, envS=None, rng=None,
                         ridge_amp=320.0, seamount_amp=1500.0,
                         abyssal_hill_amp=85.0, rise_compress_px=10,
                         depth_transition_m=200.0):
    """v7.7 海底地貌增强 — 恢复 GAN/直方图终检弱化的洋中脊/海沟/海山链/深海丘陵纹理.

    在大陆边缘、大洋盆地和大洋中脊三大基本海底地貌单元之上叠加次一级单元:
      1) 深海丘陵 (abyssal hills): -3500~-6000m 深海平原叠加 fBm 起伏 (±85m)
      2) 洋中脊增强 (mid-ocean ridge): envS>0.55 的扩张洋区叠加 +320m 脊状隆起
      3) 海山链 (seamount chains): 热点省包络 × 定向条带噪声 → 弯曲、成群、稀疏的隆起
      4) 大陆隆收窄 (continental rise compress): 过宽的陆隆带压缩 rise_compress_px px

    v8.0: 各深度掩膜改用软 sigmoid 权重 (过渡带宽 ``depth_transition_m`` 默认 200m),
    旧版硬阈值在 -3500/-6000/-2500/-7000/-2000m 等深线处产生纹理突变方块,
    在岛屿周边 (深度跨过阈值时) 尤其明显.

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        海洋后处理后的 DEM (米域). 海陆均传入, 仅修改 sea 部分.
    lats : ndarray (S,)
        每行纬度 (度). 用于极地掩膜.
    envS : float or ndarray, optional
        扩张强度先验 (>0.55 视为活跃洋中脊区). 标量时全局一致.
    rng : numpy.random.Generator, optional
        随机数生成器. None 时用 np.random.
    ridge_amp, seamount_amp, abyssal_hill_amp : float
        洋中脊/海山/深海丘陵的幅度 (米).
    rise_compress_px : int
        大陆隆收窄像素数.
    depth_transition_m : float
        深度掩膜软过渡带宽 (米). 默认 200m ≈ 4px@2048.

    Returns
    -------
    ndarray (S, S) float32
        增强后的 DEM.
    """
    S = dem_m.shape[0]
    rng = rng if rng is not None else np.random
    out = dem_m.astype(np.float32, copy=True)
    sea = dem_m < 0.0
    if not sea.any():
        return out
    sea_f = sea.astype(np.float32)
    dt = float(depth_transition_m)

    def _soft_band(d, lo, hi, trans):
        """软深度带权重: d∈[lo,hi] 时=1, 边缘 trans 米 tanh 过渡.
        (lo, hi 为深度值, lo<hi 表示 lo 是更深的边界; lo<hi 时 d∈[lo,hi] 权重=1)"""
        # w_lo: 1 当 d>=lo (浅于等于 lo), 0 当 d<lo (深于 lo)
        w_lo = 0.5 * (1.0 + np.tanh((d - lo) / trans))
        # w_hi: 1 当 d<=hi (深于等于 hi), 0 当 d>hi (浅于 hi)
        w_hi = 0.5 * (1.0 + np.tanh((hi - d) / trans))
        return w_lo * w_hi

    # ---- 1) 深海丘陵 (abyssal hills): 深海平原叠加低幅 fBm ----
    # v8.0: 软深度掩膜 (-6000 ~ -3500m, 200m 过渡) 替代硬阈值
    deep_w = _soft_band(dem_m, -6000.0, -3500.0, dt) * sea_f
    if deep_w.sum() > 1.0:
        deep = deep_w > 0.5  # 用于 fBm 归一化统计
        hills = value_fbm(S, base_freq=12.0, octaves=4, gain=0.5, rng=rng)
        hills = (hills - hills[deep].mean()) / (hills[deep].std() + 1e-8)
        out += (abyssal_hill_amp * hills * deep_w).astype(np.float32)

    # ---- 2) 洋中脊增强: 活跃扩张区叠加脊状隆起 ----
    if envS is not None:
        env_arr = np.asarray(envS)
        env_mask = (env_arr > 0.55) & sea if env_arr.ndim > 0 \
            else (float(envS) > 0.55) & sea
        if env_mask.any():
            ridge = ridged_fbm(S, direction=35.0, base_freq=12.0, octaves=4,
                               gain=0.55, rng=rng)
            out[env_mask] += (ridge_amp * ridge)[env_mask].astype(np.float32)

    # ---- 3) 海山链: 热点省内的定向海山群 ----
    # v8.0: 软深度掩膜 (-7000 ~ -2500m, 200m 过渡) 替代硬阈值
    # v44: 旧实现把 >1.5σ 铺满全洋面且走向固定 45°, 视觉上是"全球下斜雨"
    #      (均匀、等长、等间距的短划). 现拆成三层物理含义:
    #      ① 单体海山 seam —— 近各向同性带通噪声, 圆形宽缓隆起 (2~4px 直径);
    #      ② 热点轨迹走廊 corridor —— 低频窄带噪声, 海山沿走廊成链而非均匀散布;
    #      ③ 热点省包络 prov_w —— 只有 ~45% 洋面存在海山省.
    sm_w = _soft_band(dem_m, -7000.0, -2500.0, dt) * sea_f
    if sm_w.sum() > 1.0:
        sm = sm_w > 0.5  # 用于统计归一化
        prov = value_fbm(S, base_freq=3.0, octaves=3, rng=rng)
        prov_w = np.clip((prov - np.quantile(prov[sm], 0.45)) * 2.2, 0.0, 1.0) ** 1.5

        def _std3d(a):
            """按掩膜标准化并截断到 ±3σ (供阈值化使用)."""
            return np.clip((a - a[sm].mean()) / (a[sm].std() + 1e-8), -3.0, 3.0)

        # ① 单体海山: 各向异性 0.9 ≈ 圆形
        # v83: 旧版固定 direction_deg=45°，三条不同随机种子的窄带噪声一起把整个
        #   洋底铺成**同一朝向的斜向短划**（用户 v82「深海里隐约的十字格网 /
        #   斜向下雨」）。现改为按调用随机取主方向，并叠 3 个互成 60° 的方向
        #   做平均 → 各向同性斑点，海山只成"群"、不成"斜雨"。
        _dir0 = float(rng.uniform(0.0, 180.0))
        # v85（用户 v84「深海成片的密集细等深线小圈，像龟裂纹/碎渣」）：
        #   根因是 freq_center=110 —— directional_band_noise 的 freq 单位是
        #   cycles/image，波长 ≈ S/freq = 2048/110 ≈ 18.6px（≈360km）。阈值 1.8
        #   落在 3 方向平均场（std≈0.577）的 3.1σ 上，看似"很稀有"，但噪声是
        #   **空间相关的**：一片 18px 的鼓包整体越过阈值，全洋约 2500 个 10~20px、
        #   最高 ~1.8km 的小丘 —— 每个小丘在等高线图上就是一个闭合小圈。
        #   现改为：波长 2048/55 ≈ 37px（≈730km，真实海山基底宽度），阈值提到
        #   2.35（≈4.1σ）并把超出量乘 1.7 倍，于是海山从"满洋碎渣"变成"少量孤立
        #   的大锥体"；再 1.6px 高斯让阈值截断的边缘变成圆滑的山肩而不是硬台阶。
        seam = np.mean([_std3d(directional_band_noise(
            S, direction_deg=_dir0 + 60.0 * _k, freq_center=55.0,
            anisotropy=0.9, rng=rng)) for _k in range(3)], axis=0)
        seam = gaussian_filter(seam.astype(np.float32), 1.6)
        seam = np.where(seam > 2.35, (seam - 2.35) * (1.7 * seamount_amp), 0.0)
        # ② 热点轨迹: 低频 + 强各向异性 → 长条走廊（走向同样随机化）
        corridor = _std3d(directional_band_noise(S, direction_deg=_dir0, freq_center=8.0,
                                                 anisotropy=0.25, rng=rng))
        corridor = 1.0 / (1.0 + np.exp(-(corridor - 0.6) * 3.0))
        out += (seam * corridor * prov_w * sm_w).astype(np.float32)

    # ---- 4) 大陆隆收窄: 过渡带 (0~-2000m) 向深海侧压缩 ----
    rise = (dem_m < 0.0) & (dem_m > -2000.0) & sea
    if rise.any() and rise_compress_px > 0:
        land_dist = distance_transform_edt(~sea)
        rise_edge = (rise & (land_dist < rise_compress_px))
        if rise_edge.any():
            # 边缘向深海方向轻度加深, 收窄陆隆视觉宽度
            out[rise_edge] -= 50.0

    return out


# ==================== 9f. M8 陆地地貌分类 ====================
def classify_landforms(dem_m, lats, land=None,
                       win_relief=7, slope_thresh_mtn=None,
                       relief_thresh_mtn=None, river_thr=0.012,
                       envS=None, beltS=None, cf=None, tect_T=None,
                       margin_up=None, riftS=None):
    """v7.8 陆地地貌分类 — 基于高程+坡度+局部起伏+水文+构造场.

    五大基本陆地地形: 山地/高原/平原/丘陵/盆地 (int8 位打包).
    低 4 位 = 粗类 (向后兼容旧 0-5), 高 4 位 = 子类:
      0 山地: bit4-5 高度(00高山/01中山/10低山) + bit6-7 成因(00默认/01褶皱/10断层/11火山)
      1 平原: 子类=0 (独立型; 从属型由盆地标签隐含)
      2 高原: 0x10堆积 / 0x20切割 / 0x30侵蚀
      3 河谷: 子类=0
      4 沙丘: 子类=0
      5 丘陵: 0x10高丘陵 / 0x20低丘陵
      6 盆地: 0x10沉积 / 0x20克拉通 / 0x30构造-前陆 / 0x40内流 / 0x50外流
    海洋区为 -1. 全 None 构造场时退化为旧 6 类行为 (子类全 0).

    Parameters
    ----------
    dem_m, lats, land, win_relief, slope_thresh_mtn, relief_thresh_mtn, river_thr
        见 v7.7 文档.
    envS, beltS, cf, tect_T, margin_up, riftS : ndarray, optional
        板块构造场 (generate_world 已计算). 全 None 时跳过成因/盆地子类.

    Returns
    -------
    ndarray (S, S) int16
        地貌标签图 (位打包). 海洋 -1.
    """
    S = dem_m.shape[0]
    if land is None:
        land = dem_m >= 0.0
    lf = np.full(dem_m.shape, -1, np.int16)

    # ---- 坡度 (梯度幅值, m/pixel-index) ----
    gy, gx = np.gradient(dem_m.astype(np.float64))
    slope = np.hypot(gx, gy)

    # ---- 局部起伏 (窗口内 max-min) ----
    dmx = maximum_filter(dem_m, size=win_relief)
    dmn = minimum_filter(dem_m, size=win_relief)
    relief = dmx - dmn

    # ---- 局部均值 (用于河谷/高原/盆地判定) ----
    local_mean = uniform_filter(dem_m.astype(np.float64), size=win_relief)

    L = land
    elev = dem_m

    # ---- 自适应阈值: 用陆地分位数标定 (分辨率无关) ----
    land_slope = slope[L]
    land_relief = relief[L]
    s_thr = slope_thresh_mtn if slope_thresh_mtn is not None else \
        float(np.percentile(land_slope, 75)) if land_slope.size > 100 else 500.0
    r_thr = relief_thresh_mtn if relief_thresh_mtn is not None else \
        max(1000.0, float(np.percentile(land_relief, 75))) if land_relief.size > 100 else 1000.0
    r_med = float(np.median(land_relief)) if land_relief.size > 0 else 300.0
    r_low = float(np.percentile(land_relief, 40)) if land_relief.size > 100 else 200.0

    # 0) 山地: 高坡度或高起伏 (绝对>1000m 或 分位数)
    mtn = L & ((slope > s_thr) | (relief > r_thr))
    lf[mtn] = 0

    # 2) 高原: 高程>1200m 且 顶面平坦 (起伏<40 分位)
    plateau = L & (elev > 1200.0) & (relief < r_low) & (lf == -1)
    lf[plateau] = 2

    # 6) 盆地 (v7.8 新增): 低高程+低起伏+局部低洼, 优先于河谷/平原
    #    核心: dem < local_mean-80m (比河谷50m更深) 且 relief<r_low*0.7 (更平坦)
    has_tect = any(x is not None for x in (envS, beltS, cf, tect_T, margin_up, riftS))
    basin_cand = L & (elev < 1500.0) & (relief < r_med) & (lf == -1)
    basin_core = basin_cand & (dem_m < local_mean - 80.0) & (relief < r_low * 0.7)
    if basin_core.any() and has_tect:
        # 距海距离 (用于内流判定)
        d_in_sea = distance_transform_edt(~L)
        # 子类按优先级赋值 (后赋覆盖前, 故按反优先级顺序)
        # 外流(兜底) 0x50
        lf[basin_core] = 6 | 0x50
        # 沉积 0x10: 极低起伏
        sed = basin_core & (relief < r_low * 0.4)
        lf[sed] = 6 | 0x10
        # 克拉通 0x20: 稳定陆内 (cf>0.62 且 beltS<0.25)
        if cf is not None:
            craton = basin_core & (cf > 0.62) & \
                ((beltS < 0.25) if beltS is not None else True)
            lf[craton] = 6 | 0x20
        # 内流 0x40: 距海远 (d_in_sea>130px)
        endo = basin_core & (d_in_sea > 130.0)
        lf[endo] = 6 | 0x40
        # 构造-前陆 0x30: 造山带前缘 (beltS>0.30 或 tect_T[0]>0.45)
        foreland = None
        if beltS is not None:
            foreland = basin_core & (beltS > 0.30)
        elif tect_T is not None:
            foreland = basin_core & (tect_T[0] > 0.45)
        if foreland is not None and foreland.any():
            lf[foreland] = 6 | 0x30
    elif basin_core.any():
        # 无构造场: 只标粗类 6, 不分子类
        lf[basin_core] = 6

    # 3) 河谷: 低于邻域均值>50m
    low = L & (dem_m < local_mean - 50.0) & (relief < r_med) & (lf == -1)
    lf[low] = 3

    # 4) 沙丘: 低纬 (|lat|<35°) 低起伏干旱地
    abs_lat = np.abs(np.asarray(lats, np.float64))[:, None] \
        if lats is not None else np.zeros((S, 1))
    lat2d = np.broadcast_to(abs_lat, dem_m.shape)
    dune = L & (lat2d < 35.0) & (relief < r_med) & (relief > r_low * 0.5) & (lf == -1)
    lf[dune] = 4

    # 1) 平原: 低高程 低起伏 低坡度
    plain = L & (elev < 800.0) & (relief < r_med) & (slope < s_thr * 0.5) & (lf == -1)
    lf[plain] = 1

    # 5) 丘陵: 其余陆地
    lf[L & (lf == -1)] = 5

    # ====== v7.8 子类精细化 (粗类分配完成后, 全 has_tect 内保向后兼容) ======
    if has_tect:
        # ---- 山地高度子类 (bit4-5): 高山>3500m / 中山1000-3500m / 低山<1000m ----
        mtn_mask = (lf & 0x0F) == 0
        if mtn_mask.any():
            mid = mtn_mask & (elev > 1000.0) & (elev <= 3500.0)
            low_mtn = mtn_mask & (elev <= 1000.0)
            lf[mid] |= 0x10       # bit4-5 = 01 (中山)
            lf[low_mtn] |= 0x20   # bit4-5 = 10 (低山); 高山保持 00

        # ---- 山地成因子类 (bit6-7, OR叠加于高度位) ----
        # 11火山(0xC0) > 01褶皱(0x40) > 10断层(0x80) > 00侵蚀(默认)
        # 按反优先级赋值 (后赋覆盖)
        if envS is not None:
            volc = mtn_mask & (envS > 0.40)
            lf[volc] = (lf[volc] & 0x0F) | 0xC0  # 清除高4位后设成因
        elif tect_T is not None and tect_T.shape[0] > 1:
            volc = mtn_mask & (tect_T[1] > 0.55)
            lf[volc] = (lf[volc] & 0x0F) | 0xC0
        if beltS is not None:
            fold = mtn_mask & (beltS > 0.45) & \
                (((lf >> 6) & 0x03) == 0)  # 未被火山覆盖
            lf[fold] = (lf[fold] & 0x0F) | 0x40
        fault = None
        if riftS is not None:
            fault = mtn_mask & (riftS > 0.30)
        elif margin_up is not None:
            fault = mtn_mask & (margin_up == 1)
        if fault is not None and fault.any():
            unfault = fault & (((lf >> 6) & 0x03) == 0)
            lf[unfault] = (lf[unfault] & 0x0F) | 0x80

        # ---- 高原子类 (0x10堆积 / 0x20切割 / 0x30侵蚀) ----
        plat_mask = (lf & 0x0F) == 2
        if plat_mask.any():
            accum = plat_mask & (relief < r_low * 0.5)
            dissect = plat_mask & (relief > r_low * 0.8)
            lf[accum] |= 0x10
            lf[dissect] |= 0x20
            if cf is not None:
                eros = plat_mask & (cf > 0.62) & \
                    ((beltS < 0.25) if beltS is not None else True) & \
                    ((lf & 0xF0) == 0)  # 未被堆积/切割覆盖
                lf[eros] |= 0x30

        # ---- 丘陵子类 (0x10高丘陵 / 0x20低丘陵) ----
        hill_mask = (lf & 0x0F) == 5
        if hill_mask.any() and land_relief.size > 100:
            r_h60 = float(np.percentile(land_relief, 60))
            hi_h = hill_mask & (relief > r_h60)
            lf[hi_h] |= 0x10
            lo_h = hill_mask & (relief <= r_h60)
            lf[lo_h] |= 0x20

    return lf


# ==================== 9d. M5 克拉通大流域盆地 ====================
def add_craton_basins(dem, conv, rng, belt=None, lats=None,
                      max_exo=4, max_endo=1, endorheic=True,
                      precip=None, endo_p_max=800.0,
                      target_lo=0.020, target_hi=0.065,
                      min_share=0.015,
                      dyke_coast_excl=40.0, dyke_height=250.0,
                      basin_max_share=0.10):
    """M5 克拉通大流域盆地 (亚马逊/刚果/西西伯利亚型):
    稳定陆内低地(低起伏·低收敛·|lat|<72°)重塑为向单一出口的线性倾斜面:
      z = z_out + (z_far−z_out)·(d/d_max) + fBm微起伏(±18m, 远端减半)
    线性坡 (~4m/px) > σ6平滑后噪声 → D8 全流域汇入单一出口
      → top1 流域 3-4.5% 陆地, 干流走廊自动成河。
    超大低地连通域 → 以最低单元为种子盘状裁切到目标份额。
    出口 = 盆地 1000km 内最低海岸点; 无出海路径 → 内陆海 (里海型,
    终端洼地 −28m)。颈道保通: 出口-盆地间 min(原地形, 线性坡道) 下切。
    返回 (dem, n_basins, info)。
    ---- v7.4: 用户报告"内陆湖/内流盆地数量明显不足, 湖泊巨型化"
      max_exo: 2→4 (允许最多4个外流大盆地, 对应地球 亚马逊/刚果/密西西比/拉普拉塔 4大流域)
      target_lo: 0.032 → 0.020 (最小目标份额从3.2%降到2.0%, 允许更多中小型盆地生成,
      不再强求单一大盆地挤到 3.2% 陆地面积, 更像地球 里海/咸海/乍得湖 多种尺度并存)
    ---- v34: 内流盆地干旱门控 (用户报告"巨大内流区违背水文学常识")
      地球内流区 (~18% 陆地) 几乎全部位于干旱区 (里海/乍得/塔里木), 湿润区
      封闭盆地必然溢流切出外流河道. 新增:
        max_endo : 内流盆地数量上限 (默认1, 里海型)
        precip   : 降水场 (mm/yr, 与 dem 同尺寸); 提供了时内流盆地拒建
                   在平均降水 > endo_p_max (默认 800mm) 的湿润封闭盆地,
                   改为维持自然排水 (由后续河道自然外流), 消除"绿水巨湖".
      max_exo=3 让多个大流域各自拥有外流出口, 而非仅最大一个.
    """
    S, W = dem.shape
    land = dem > 0
    n_land = int(land.sum())
    if n_land < 10000:
        return dem, 0, ""
    f = max(1, S // 128)                                   # 粗化因子 (2048→16)
    Hc, Wc = S // f, W // f
    sub = dem[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f)
    cm = sub.mean((1, 3)); cs = sub.std((1, 3))
    cc = conv[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
    cb = (belt[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).max((1, 3))
          if belt is not None else np.zeros_like(cm))
    from scipy.ndimage import label as _lbl, binary_closing

    def _edt_to(m):                                        # 周期 EDT: 到 m 的距离 (m内=0)
        """计算到掩膜 m 的周期欧氏距离场（低分辨率近似，m 内为 0）。"""
        # v9.x: 低分辨率周期 EDT 替代 3x 铺贴全分辨率 EDT (37x 加速)
        return _edt_lowres_periodic(m)

    ocean = ~land
    coastal = land & (np.roll(ocean, 1, 0) | np.roll(ocean, -1, 0)
                      | np.roll(ocean, 1, 1) | np.roll(ocean, -1, 1))
    d_sea_full = _edt_to(ocean)                            # 距海 (全分辨率, px)
    # 粗块级内陆判据: 块均值距海 ≥60px → 排除沿海平原/海湾两岸碎片
    # (水系不连通的伪盆地; 真正克拉通盆地应在大陆内部)
    d_sea_c = d_sea_full[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
    lat_c = (np.abs(np.asarray(lats, np.float64))[:, None]
             if lats is not None else np.zeros((S, 1)))
    info, n_endo = [], 0
    existing_outlets = []  # v26: 出口间距 >=500px, 防坡度覆盖冲突
    land_c = cm > 0.0
    if land_c.sum() < 16:
        return dem, 0, ""
    for relax in (False, True):                            # 二轮: 严格→放宽
        # 分位数自适应阈值: 起伏/收敛取陆块相对低值 (适配任意管线阶段)
        # M6: 陆块占比 ≥0.75 — 群岛混块(陆海各半的粗块)均值可伪过判据,
        # 倾斜面盖在群岛上各岛独立排水 → 流域不收敛
        lf_c = land[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean((1, 3))
        cs_thr = np.quantile(cs[land_c], 0.60 if not relax else 0.78)
        cc_thr = np.quantile(cc[land_c], 0.72 if not relax else 0.88)
        elig = ((cm > 50) & (cm < (800 if not relax else 950))
                & (cs < cs_thr) & (cc < cc_thr) & (cb < 0.45)
                & (lf_c >= 0.75)
                & (d_sea_c >= (45.0 if not relax else 30.0)))
        if lats is not None:                               # 极地冰原不建盆地
            elig &= np.repeat(np.abs(lat_c[f // 2::f, 0])[:, None] < 72.0, Wc, 1)
        el_c = binary_closing(elig, iterations=1)
        lab_c, n_c = _lbl(el_c)
        if n_c == 0:
            continue
        sizes = np.array([(lab_c == i).sum() * f * f for i in range(1, n_c + 1)])
        n_exo = 0
        for oi in np.argsort(sizes)[::-1][:4]:
            if n_exo >= max_exo:
                break
            base = lab_c == (oi + 1)
            a_base = base.sum() * f * f
            if a_base < min_share * n_land:
                continue
            if a_base > target_hi * n_land:                # 超大连通域: 最低单元种子裁切
                ys, xs = np.nonzero(base)
                k = int(np.argmin(cm[ys, xs]))
                seed = np.zeros_like(base); seed[ys[k], xs[k]] = True
                d_seed = distance_transform_edt(~seed)
                grown = None
                for r in range(2, 40):
                    g = base & (d_seed <= r)
                    if g.sum() * f * f >= target_lo * n_land:
                        grown = g
                        break
                if grown is None:
                    continue
            elif a_base < target_lo * n_land:              # 小域: 向外生长
                d_c = distance_transform_edt(~base)
                grown = base
                for r in range(1, 8):
                    g = elig & (d_c <= r)
                    if g.sum() * f * f > target_hi * n_land:
                        break
                    grown = g
                    if g.sum() * f * f >= target_lo * n_land:
                        break
            else:
                grown = base
            # v26: 不再按 dem/belt 过滤精细 mask — enhance_terrain_texture 的 ±200m 噪声
            # 和山地带使 mask 严重碎片化 (1m 低地旁邻 1910m 高峰), 倾斜面无法连续覆盖.
            # 直接对整个粗粒度连通域 (grown) 应用径向倾斜面, 内部孤峰由倾斜面统一拉低.
            mask = np.repeat(np.repeat(grown, f, 0), f, 1)[:S, :W] & land
            if mask.sum() < min_share * n_land:
                continue
            d_to = _edt_to(mask)                           # 到掩模距离 (内=0)
            near = coastal & (d_to <= 50.0)                # ~1000km 内海岸
            if near.any():
                ys, xs = np.nonzero(near)
                k = int(np.argmin(dem[ys, xs]))
                oy, ox = int(ys[k]), int(xs[k])
                mode = 'exo' if n_exo < max_exo else None
                # 外流名额用尽 → 若无缘出口则放弃该候选 (不占用内流名额)
                if mode is None:
                    continue
            elif endorheic and n_endo < max_endo:
                # ---- v34: 内流盆地干旱门控 ----
                # 湿润封闭盆地 (平均降水 > endo_p_max) 不建内流巨型湖 —
                # 现实中湿润盆地必溢流切出外流河道 (河流改道/被袭夺)
                if precip is not None and float(np.mean(precip[mask])) > endo_p_max:
                    continue
                ys, xs = np.nonzero(mask)
                k = int(np.argmin(dem[ys, xs]))
                oy, ox = int(ys[k]), int(xs[k])
                mode = 'endo'
            else:
                continue
            # v26: 出口间距 2200px — 配合 1000px mask 防盆地重叠
            if existing_outlets and min(np.hypot(oy - ey, ox - ex)
                                        for ey, ex in existing_outlets) < 2200.0:
                continue
            existing_outlets.append((oy, ox))
            pt = np.zeros((S, W), bool); pt[oy, ox] = True
            # v26: 用精确欧氏距离替代低分辨率 EDT — 低分辨率 EDT (4px 量化)
            # 导致坡度呈 4px 阶梯, 阶内平坦区被 D8 视为洼地 → 水流停滞
            _yy, _xx = np.ogrid[:S, :W]
            d_o = np.sqrt((_yy - oy) ** 2 + (_xx - ox) ** 2).astype(np.float32)
            # 海岸分水岭后撤 + 出口扇区豁免: 距海 <25px 不重塑 (保分水岭),
            # 但出口 60px 扇区豁免 → 倾斜面直达海岸 (防 25px 原始带挡水,
            # 仅 6px 河道口过流 → 大量分流)
            if mode == 'exo':
                # v27: 1200px mask — 更大流域面积, 提升最大流域占比.
                # mask 仅含陆地, 不产生圆形海洋高原.
                # ===== v37: 面积上限 basin_max_share (默认 10% 陆地) =====
                # 诊断 (3D 模型反馈 + 定量): 原 1200px 膨胀让单一外流盆地覆盖
                # 60% 陆地 (几乎整块大陆), 半径 1000px+ 的环形堤坝因围绕整条
                # 海岸线而成为"全球边缘山脉环"→ 大陆成巨碗(内陆积水成海).
                # 地球最大流域 (亚马逊) 仅约 3.9% 陆地: 按目标份额限制 mask
                # 半径——保留距出口最近的 basin_max_share(=10%) 陆地, 使多个
                # 中型盆地并立、各自独立排海, 而非一个吞并全大陆的超级碗.
                mask = (d_o <= 1200.0) & land
                if mask.sum() > basin_max_share * n_land:
                    _o_srt = d_o[land]
                    _need = basin_max_share * n_land
                    _rmax = float(np.partition(_o_srt, int(_need))[int(_need) - 1])
                    mask = mask & (d_o <= _rmax)
                if mask.sum() < min_share * n_land:
                    continue
            # d_in: 掩模内距边界 (0 外, 正内) — distance_transform_edt(mask) 计算每个 mask 内像素到最近 mask 外像素的距离
            d_in = distance_transform_edt(mask)
            # d_to: 距掩模 (0 内, 正外) — mask 修改后需重新计算 (10px ring/dyke 精度需求)
            d_to = distance_transform_edt(~mask)
            # 分水岭保底: 倾斜面远端 ≥ 盆缘外围地形 75 分位
            # (若 z_far < 外围, 远端水向盆外低地泄漏 → 流域破碎)
            ring = (d_to > 0) & (d_to <= 10.0) & land & ~mask
            z_ring = float(np.quantile(dem[ring], 0.75)) if ring.sum() > 50 \
                else float(np.quantile(dem[mask], 0.85))
            # v26: dmax = mask 实际半径 — 坡度覆盖整个 mask
            dmax = float(d_o[mask].max())
            if dmax < 20:
                continue
            # v27: z_far 上限 1500m — mask 仅含陆地, 抬升陆地不会产生圆形海洋高原.
            # (v38 曾试 950/1.0/t^0.3 缓坡盆地 → mean_land_slope 无变化,
            #  河流收敛反弱, 已回退)
            z_far = min(max(float(np.quantile(dem[mask], 0.60)), z_ring), 1500.0)
            z_out = float(dem[oy, ox]) if mode == 'exo' else -28.0
            # v27: 坡度保底 2.0 m/px — 强坡度确保水流直达出口
            z_far = max(z_far, z_out + 2.0 * dmax)
            # v27: z_far 上限 1500m
            z_far = min(z_far, 1500.0)
            # 坡度覆盖: 仅限 mask 内部 (避免相邻盆地出口互相覆盖抬升)
            # 海岸带由深宽河道走廊引导水流至出口
            w = mask.astype(np.float32)
            # v37: 海岸锥减 — 距海 40px 内坡面权重归零, 90px 外全量.
            # 修复倾斜面远端点 (t→1, z_b≈z_far 最高 1500m) 落在海岸线时,
            # 海岸带被抬成数百米台地 (用户所见"边缘山脉环"的另一来源).
            # 地球实测 (ETOPO1 2048 口径): 0-5px 84m / 5-15px 266m —— 海岸带
            # 应基本保持低地. 出口走廊 (d_o≤60) 豁免.
            _wc = np.clip((d_sea_full - 40.0) / 50.0, 0.0, 1.0).astype(np.float32)
            w = w * np.maximum(_wc, ((d_o <= 60.0) & land).astype(np.float32))
            # v26: 保存海洋像素原始高程 — 坡度不抬升海底, 确保出口连通海洋
            dem_ocean_save = dem.copy()
            rel = 0.2 * _lowres_fbm(S, 60.0, 2, rng=rng)     # 极弱长波起伏 (±0.2m)
            rel = rel * (1.0 - 0.5 * np.clip(d_o / dmax, 0, 1))
            # v26: 逐像素微扰动 (±0.05m) 打破 D8 平坦区域平局
            # 纯线性坡度会产生大片同高程像素 → D8 视为洼地 → 水流停滞
            rel += rng.uniform(-0.05, 0.05, size=(S, W)).astype(np.float32)
            # v27: 激进凹形坡度 t^0.2 — 出口附近极陡, 抗 sigma=6 高斯平滑
            t = np.clip(d_o / max(dmax, 1.0), 0, 1)
            z_b = z_out + (z_far - z_out) * np.power(t, 0.2) + rel
            # v26: 地板仅在出口走廊外 (d_o>60) 生效 — 走廊内保持自然坡度,
            # 避免 8m 平坦区导致 D8 无法定流向、水无法汇入出口
            floor = np.where(d_o > 60.0, 8.0 if mode == 'exo' else -30.0,
                             z_out - 10.0)
            z_b = np.maximum(z_b, floor)
            dem = (dem * (1 - w) + z_b * w).astype(np.float32)   # 倾斜面装配
            # v27: 移除海洋抬升逻辑 — mask 仅含陆地, 海洋保持原始高程.
            # 水流通过自然海岸出口排入海洋, 无需人工阻挡海洋侧漏.
            # v27: 分水岭筑堤 — mask 外缘 30px 环带 (含海洋) 抬升至 z_far+350m,
            # 阻断 D8 边界漏水. 用 fBm 扰动环带距离使堤坝边界自然不规则.
            # ===== v37 重构: 海岸不筑堤 + 堤高降为 z_far+80 =====
            # 诊断 (3D 模型用户反馈"陆地边缘围绕一条山脉 + 陆地中间全是水体",
            # 定量: 海岸 0-5px 均高 46m → 5-15px 陡升 1328m (边缘山环); 内陆
            # 巨型闭合洼地占陆地 47.7%, 水位 1301m vs 内陆地形中位 370m → fill
            # 后 76% 陆地低于湖泊水位; lake_share 0.75 vs 地球 0.33)。根因:
            # 原堤坝把 mask 外缘(海岸线与近海) 整圈抬到最高 1850m → 每块大陆
            # 天然成"巨碗"(边缘山脉环 + 内陆洼地), 洼地无低垭口排海 → 内陆海.
            # 修复: ① 距海 40px 内不筑堤 (海岸/近海无需堤坝, 海面即排水道;
            # 内陆相邻盆地分水岭仍保留堤坝隔离以防流域互漏);
            # ② 堤高 z_far+350 → z_far+dyke_height (默认 250m): 仅内陆分水岭
            # 略高于两侧地形即可引导 D8, 不需要 1800m 人造山脉; ③ 海岸 15-45px
            # 坡面权重锥减, 远端点 (z_b≈z_far) 不再把海岸抬成台地.
            _dyke_noise = 10.0 * (_lowres_fbm(S, 30.0, 3, rng=rng) - 0.5)
            dyke = (d_to > 0) & (d_to + _dyke_noise <= 30.0)
            if mode == 'exo':
                dyke &= (d_o > 60.0)                       # 出口走廊豁免
            dyke &= land & (d_sea_full > dyke_coast_excl)  # v37: 仅内陆分水岭筑堤
            low = dyke & (dem < z_far + dyke_height)
            if low.any():
                ramp2 = (z_far + dyke_height) * np.exp(-(d_to - 0.5) / 10.0)
                dem = np.maximum(dem, np.where(low, ramp2, -np.inf)).astype(np.float32)
            # v27: 不使用 fill_depressions — 它会把盆地填到堤坝高度, 破坏径向坡度.
            # v26: 已移除河道雕刻 — 雕刻产生的封闭洼地需 fill_depressions 修复,
            # 而 fill_depressions 会把整个盆地填到堤坝高度 (1473m) 破坏坡度.
            # 改用 5m/px 纯径向坡度 + ±0.2m 噪声, 25x 优势确保水流直达出口.
            if mode == 'exo':
                n_exo += 1
            else:
                n_endo += 1
            info.append(f"{mode}{mask.sum() / n_land * 100:.1f}%@({oy // f},{ox // f})")
            if n_exo >= max_exo and n_endo >= max_endo:
                break
        if n_exo + n_endo >= max_exo + max_endo:
            break
    return dem, len(info), "; ".join(info)


# ==================== 10. 验证指标套件 (A7) ====================
def _psd_slope(x):
    """估计功率谱密度径向斜率（越负表示地形越粗糙），bin 不足时返回 nan。"""
    x = x - x.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    f1 = np.fft.fftshift(np.fft.fftfreq(x.shape[0]))
    R = np.sqrt(f1[:, None] ** 2 + f1[None, :] ** 2)
    bins = np.geomspace(0.02, 0.45, 13)
    idx = np.digitize(R.ravel(), bins)
    rp, pp = [], []
    for k in range(1, len(bins)):
        sel = idx == k
        if sel.sum() > 8:
            val = p.ravel()[sel].mean()
            if val > 0:                        # 过滤零功率 bin, 避免 log(0)=-inf
                rp.append(bins[k]); pp.append(val)
    if len(rp) < 2:                             # 有效 bin 不足无法拟合
        return float('nan')
    return float(np.polyfit(np.log(rp), np.log(pp), 1)[0])


def _fractal_dim(v01):
    """用盒计数法估计海岸线分形维 D，退化时返回 nan。"""
    land = (v01 > 0.5).astype(np.float32)
    if land.max() == 0 or land.min() == 1:
        return float('nan')
    sizes = np.array([2, 4, 8, 16, 32], float)
    counts = [float(((block_reduce(land, int(b), "max") > 0.5)
                     & (block_reduce(land, int(b), "min") < 0.5)).sum()) for b in sizes]
    if min(counts) <= 2:
        return float('nan')
    return float(-np.polyfit(np.log(sizes), np.log(counts), 1)[0])


def _slope_stats(v01):
    """返回坡度均值与 P95（×100），衡量地形陡峭程度。"""
    gy, gx = np.gradient(v01)
    g = np.hypot(gx, gy)
    return float(g.mean() * 100), float(np.quantile(g, 0.95) * 100)


def _coherence(v01):
    """用结构张量估计山脊方向相干性（0~1），无显著结构时返回 nan。"""
    gy, gx = np.gradient(gaussian_filter(v01, 1.0))
    Jxx = gaussian_filter(gx * gx, 3); Jyy = gaussian_filter(gy * gy, 3)
    Jxy = gaussian_filter(gx * gy, 3)
    tr = Jxx + Jyy
    disc = np.sqrt(0.25 * (Jxx - Jyy) ** 2 + Jxy ** 2)
    lam1, lam2 = 0.5 * tr + disc, 0.5 * tr - disc
    c = (lam1 - lam2) / (lam1 + lam2 + 1e-9)
    w = np.sqrt(tr) > np.quantile(np.sqrt(tr), 0.8)
    return float(c[w].mean()) if w.any() else float('nan')


def snow_area_by_band(v01, lats):
    """统计 0-30/30-50/50-66.5/极地 四个纬度带的**陆地**雪盖占比 (v42 口径统一)。

    与 ``render_snow_field`` / ``polar_ice_mask`` 同一套判据:
      - 基础雪: v01 >= 纬度雪线 ``snowline_m``
      - 66.5~72° 过渡: 雪线下 400m 的极地平顶软冰 (66.5° 起闸门, 69.25° 后全强制)
      - >= 72°: 陆地全强制冰盖
    分母为带内**陆地**像素 (旧口径分母含海洋, 极地带被海水稀释到 10%,
    与出图极地整片白完全不符)。海冰不在此统计内 (由渲染单独处理)。
    """
    sl = m2v(snowline_m(lats))[:, None]
    a = np.abs(lats)[:, None].repeat(v01.shape[1], 1)
    land = v01 >= 0.5
    snow = (v01 >= sl) & land
    gate_66 = np.clip((a - 66.5) / 5.5, 0.0, 1.0)          # 66.5→0, 72→1
    soft_ice = land & (v01 >= (sl - m2v(400.0)))
    snow |= land & ((gate_66 >= 0.5) | (soft_ice & (gate_66 > 0)))
    snow |= land & (a >= 72.0)
    out = []
    for lo, hi in [(0, 30), (30, 50), (50, 66.5), (66.5, 90.01)]:
        m = (a >= lo) & (a < hi) & land
        n = int(m.sum())
        out.append(float((snow & m).sum() / n) if n else 0.0)
    return out


def terrain_metrics(tgt, out, lats, bm, block=8, tag=""):
    """打印目标/输出的一组地形指标对比（PSD/分形维/坡度/相干性/雪盖）。"""
    def f(x):
        """格式化指标数值，非有限值显示 nan。"""
        return f"{x:8.3f}" if np.isfinite(x) else "     nan"
    tm, to_ = _slope_stats(tgt)
    om, oo = _slope_stats(out)
    rmse = float(np.sqrt(np.mean((block_reduce(out, block, "mean") - bm) ** 2)))
    print(f"\n--- 指标 [{tag}] (目标 | 输出) ---")
    for name, a, b in [("PSD 斜率(越负越粗)", _psd_slope(tgt), _psd_slope(out)),
                       ("海岸分形维 D", _fractal_dim(tgt), _fractal_dim(out)),
                       ("坡度均值 x1e2", tm, om), ("坡度 P95 x1e2", to_, oo),
                       ("山脊方向相干性", _coherence(tgt), _coherence(out)),
                       ("块均值RMSE(仅输出)", 0.0, rmse)]:
        print(f"  {name:20s} {f(a)} | {f(b)}")
    print("  雪盖-纬度带 [0-30/30-50/50-66.5/极地]")
    print("    目标:", " ".join(f"{x:.2f}" for x in snow_area_by_band(tgt, lats)))
    print("    输出:", " ".join(f"{x:.2f}" for x in snow_area_by_band(out, lats)))


# ==================== 11. 接缝羽化拼接器 (A8) ====================
@torch.no_grad()
def generate_large_map(gen, dem, ch_stats, y0, x0, out_size=1024, tile=256,
                       overlap=32, block=8, seed=0, tect_provider=None):
    """重叠瓦片推理 + 余弦羽化。每瓦片对同一份全球粗底图做块均值一致性校正
    → 低频跨瓦片天然无缝(山脉带不会在瓦片边界断头), 高频羽化过渡。

    tect_provider: 可选 ``(y0, x0, size) -> (4, size, size)`` 构造场切片提供器。
        条件张量含 4 个构造通道, 缺省时按全 0 处理(与 build_sample 默认一致)。
    """
    gen.eval()
    rng = np.random.default_rng(seed)
    step = tile - overlap
    ramp = np.ones(tile, np.float32)
    ramp[:overlap] = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
    ramp[-overlap:] = ramp[:overlap][::-1]
    win = np.outer(ramp, ramp)
    acc = np.zeros((out_size, out_size), np.float32)
    wgt = np.zeros((out_size, out_size), np.float32)
    for ty in range(0, out_size, step):
        for tx in range(0, out_size, step):
            m, lats = dem.patch(y0 + ty, x0 + tx, tile)
            tect = None if tect_provider is None else tect_provider(
                y0 + ty, x0 + tx, tile)
            ch, _ = build_sample(m, lats, block, rng, ch_stats, tect)
            # 条件张量是 block 分辨率的 (12, tile//block, tile//block),
            # GAN 需要全分辨率输入, 必须按 block 最近邻放大后才能前向。
            xin = np.ascontiguousarray(np.repeat(np.repeat(ch, block, 1), block, 2))
            raw = gen(torch.from_numpy(xin[None]).float().to(device)).cpu().squeeze().numpy()
            vc = m2v(m)
            out = postprocess_pipeline(raw.astype(np.float32),
                                       block_reduce(vc, block, "mean"),
                                       block_reduce(vc, block, "std"),
                                       block_reduce((m > 0).astype(np.float32), block, "mean"),
                                       lats, rng, block)
            h, w = min(tile, out_size - ty), min(tile, out_size - tx)
            acc[ty:ty + h, tx:tx + w] += out[:h, :w] * win[:h, :w]
            wgt[ty:ty + h, tx:tx + w] += win[:h, :w]
    return np.clip(acc / np.maximum(wgt, 1e-8), 0, 1)


def make_terrain_cmap():
    """构建地形色卡：**海洋按米制深度定点布色**，陆地按 v 均匀布色。

    为什么海洋要按米制定点
    ----------------------
    ``m2v`` 把 -11000~0m 压进 v∈[0, 0.5]，其中整段大陆架（0~-200m）只占
    v∈[0.47, 0.50] 这 3% 的色带宽度。若海洋段按 v 均匀布色，-170m 的陆架坡折
    与 -1400m 的陆坡拿到几乎相同的颜色，图上完全看不出"浅海台地"。

    改为在**真实深度锚点**上定点放色标（经 ``m2v`` 换算到 v 位置），色带仍以 v
    索引，因此不影响任何按 v 取色的调用方；但 -170m 处会有一段明确色阶，
    浅海（0~-170m）呈略浅的蓝色台地（v49: 与深海的亮度比 3.4:1，色相偏青），
    陆坡（-170~-2600m）快速转深蓝。
    """
    # v48: 浅水"荧光描边"彻底重写. 旧版有三处致命缺陷:
    #   ① -400m 色 (0.200,0.400,0.560) 的亮度反而低于 -1000m 的 (0.175,0.440,0.670)
    #      → 亮度非单调, 在 -400m 附近形成第二道"亮蓝环";
    #   ② -170→-160m 只有 10m 深差却把亮度从 0.465 跳到 0.533 (+0.068) —— 2048 图上
    #      陆架 6px 都落在这一段, 整条海岸被描成一条等亮度亮环, 即用户看到的"霓虹描边";
    #   ③ 浅水端亮度 (≈0.57) 比 -2500m 深海 (0.27) 高 2.1 倍, 反差过大.
    # v49: 二次修正. v48 虽然把色标改成单调, 但把深海压到亮度 0.068 —— 深蓝接近黑,
    #   而陆架仍有 0.43, 6.4 倍的反差在 3~6px 的坡折上依然被读成"一圈发光的描边".
    #   实测真彩色海底地形图 (GEBCO / ETOPO 渲染) 的深海亮度约 0.12~0.16, 陆架约
    #   0.36~0.40, 反差只有 2~3 倍 —— 深度是靠**色相** (深海偏靛、浅海偏青) 和
    #   缓变梯度表达的, 不是靠明暗反差.
    # v50: 第三次修正 —— 真正的病灶不是"太亮", 而是**平顶 + 陡坎**两个特征叠加:
    #   实测近岸 3px 内海面亮度 p50=0.37、而 8px 外就掉到 0.25, 于是整条海岸被
    #   读成"一条等亮度的浅色带"再"啪"地断开, 这正是用户说的"等高线/发光描边感".
    # v51: 第四次修正 —— v50 消掉了"断崖", 却把 0~-400m 压成了**同色平顶**, 平顶本身
    #   在深海底色上依然是一条等亮度的贴边色带 (实测 v50 剖面: 0px 0.363 / 4px 0.356 /
    #   6px 0.345 / 8px 0.282 —— 前 6px 几乎不动, 然后 2px 内掉 0.06, 仍是一道"边").
    #   地形的正确表达是**全程单调的连续渐变**, 没有平台、没有折点:
    #     0 ~ -400m  0.307 → 0.273  (陆架: 近岸略浅, 全程可见地变深)
    #     -400 ~ -2500m 0.273 → 0.208 (陆坡)
    #     -2500 ~ -11000m 0.208 → 0.109 (深海/海沟)
    #   同时把近岸亮度从 0.367 降到 0.307 —— 与陆地绿/黄 (~0.45~0.55) 形成 0.17 的
    #   阶跃, 海岸线因此**锐利**, 而海底一侧是缓坡而不是描边.
    ocean_stops = [(-11000.0, (0.060, 0.115, 0.200)),   # 亮度 0.109
                   (-6000.0,  (0.075, 0.145, 0.235)),   # 0.137
                   (-4000.0,  (0.100, 0.185, 0.285)),   # 0.174
                   (-2500.0,  (0.125, 0.220, 0.330)),   # 0.208
                   (-1000.0,  (0.150, 0.258, 0.385)),   # 0.244
                   (-400.0,   (0.172, 0.288, 0.422)),   # 0.273
                   (-200.0,   (0.185, 0.310, 0.448)),   # 0.293
                   (-100.0,   (0.190, 0.318, 0.458)),   # 0.301
                   (-40.0,    (0.192, 0.322, 0.463)),   # 0.304
                   (0.0,      (0.194, 0.325, 0.466))]   # 0.307
    land_colors = [(0.35, 0.55, 0.28), (0.55, 0.62, 0.30), (0.68, 0.52, 0.30),
                   (0.55, 0.38, 0.28), (0.80, 0.76, 0.72), (1.00, 1.00, 1.00)]
    stops = [(float(m2v(m)), c) for m, c in ocean_stops]
    stops += [(0.5 + 0.1 * i, c) for i, c in enumerate(land_colors)]
    return LinearSegmentedColormap.from_list('dem', stops, N=256)


# ==================== 12. 训练 (A5: 一致性损失已集成) ====================
def train(dem):
    """训练 Terrain GAN v5：构建数据/网络/对抗+一致性损失，保存最佳权重并出结果图。"""
    # —— 标准配置 (MPS 约 2.5-4h) ——
    # block=4: 条件分辨率 64² (原 8 → 32²)。评测显示 64:1 压缩是"山被摊平"
    #   的主因: 真值陆地 ≥5° 坡面占 13.5%, block=8 权重仅 3.0%。
    size, block = 256, 4
    n_train, n_val, batch, epochs, warmup = 1200, 120, 8, 30, 4
    # —— 快速预览配置 (约 20-30min): 取消下行注释 ——
    # size, block, n_train, n_val, batch, epochs, warmup = 128, 8, 500, 60, 8, 20, 3
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints", "best_terrain_gan_v5.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    print("估计输入通道统计量...")
    tect_provider = make_tect_provider(dem)
    ch_stats = estimate_channel_stats(dem, size, block, tect_provider=tect_provider)
    train_set = TerrainDataset(dem, size, block, n_train, ch_stats, is_train=True,
                               tect_provider=tect_provider)
    val_set = TerrainDataset(dem, size, block, n_val, ch_stats, is_train=False,
                             tect_provider=tect_provider)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=0)
    print(f"训练: 12通道(含构造场) | 在线无限裁剪 | 验证: 太平洋带 held-out")

    gen = UNet(in_ch=12, base=32).to(device)
    ema = EMA(gen, decay=0.999)
    disc_fine = PatchDiscriminator(13).to(device)
    disc_mid = PatchDiscriminator(13).to(device)
    disc_coarse = PatchDiscriminator(13).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_df = optim.Adam(disc_fine.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_dm = optim.Adam(disc_mid.parameters(), lr=1.5e-4, betas=(0.5, 0.999))
    opt_dc = optim.Adam(disc_coarse.parameters(), lr=1e-4, betas=(0.5, 0.999))
    scheds = [optim.lr_scheduler.CosineAnnealingLR(o, T_max=epochs, eta_min=5e-6)
              for o in (opt_g, opt_df, opt_dm, opt_dc)]

    crit_gan, crit_l1 = nn.MSELoss(), nn.L1Loss()
    crit_edge = EdgeLoss().to(device)
    crit_fft = FFTLoss().to(device)
    crit_lap = LaplacianLoss().to(device)
    crit_dist = DistributionLoss()
    crit_slope = SlopeLoss().to(device)
    discs = ((disc_fine, opt_df, 1), (disc_mid, opt_dm, 2), (disc_coarse, opt_dc, 4))

    # ---- A5: 粗图一致性损失所需张量 ----
    ch_mean_t = torch.tensor(ch_stats["mean"], device=device).view(1, -1, 1, 1)
    ch_std_t = torch.tensor(ch_stats["std"], device=device).view(1, -1, 1, 1)
    w_cons = 4.0

    best_score, patience, patience_counter = float('inf'), 10, 0
    print("-" * 60)

    for epoch in range(epochs):
        prog = epoch / max(1, epochs - 1)
        warm = 0.0 if epoch < warmup else min(1.0, (epoch - warmup + 1) / 10.0)
        w_gan, w_fm = 3.5 * warm, 10.0 * warm
        w_l1 = 12.0 - 9.0 * prog          # 12 → 3
        w_dist = 6.0 + 4.0 * prog         # 6 → 10
        w_edge, w_fft, w_lap = 4.0, 2.0, 1.5
        # L_slope: 全程生效并随 prog 加码 (1.0 → 3.0), 只对齐 2 个分布统计量,
        #   不会像逐点梯度损失那样把纹理拉向条件块均值。
        w_slope = 1.0 + 2.0 * prog
        # L_tile: 预热后再上 (与 GAN 同步), 避免早期随机权重下的接缝噪声干扰重建。
        w_tile = 1.5 * warm

        gen.train()
        for D, _, _ in discs:
            D.train()
        ep_d, ep_g, t_ep = 0.0, 0.0, time.time()

        for inp, target in train_loader:
            inp, target = inp.to(device), target.to(device)

            # A5: 反归一化通道0 → 粗底图块均值 (32x32)
            # inp 已是全分辨率(通道被repeat过) → 先池化回块级 32x32,
            # 与 F.avg_pool2d(fake, block) 形状对齐
            bm_t = F.avg_pool2d(inp[:, 0:1] * ch_std_t[:, 0:1] + ch_mean_t[:, 0:1], block)

            if w_gan > 0:   # ---- 判别器 (预热期跳过) ----
                fake = gen(inp).detach()
                for D, optD, sc in discs:
                    rp = torch.cat([downsample(inp, sc), downsample(target, sc)], 1)
                    fp = torch.cat([downsample(inp, sc), downsample(fake, sc)], 1)
                    out_r, out_f = D(rp), D(fp)
                    loss_d = 0.5 * (crit_gan(out_r, torch.ones_like(out_r))
                                    + crit_gan(out_f, torch.zeros_like(out_f)))
                    optD.zero_grad(); loss_d.backward(); optD.step()
                    ep_d += loss_d.item()

            # ---- 生成器 ----
            fake = gen(inp)
            loss_gan = loss_fm = fake.new_tensor(0.0)
            if w_gan > 0:
                for D, _, sc in discs[:2]:
                    rp = torch.cat([downsample(inp, sc), downsample(target, sc)], 1)
                    fp = torch.cat([downsample(inp, sc), downsample(fake, sc)], 1)
                    pred_f, feat_f = D(fp, return_feat=True)
                    _, feat_r = D(rp, return_feat=True)
                    loss_gan = loss_gan + crit_gan(pred_f, torch.ones_like(pred_f))
                    loss_fm = loss_fm + F.l1_loss(feat_f, feat_r)

            bm_fake = F.avg_pool2d(fake, block)
            assert bm_fake.shape == bm_t.shape, \
                f"一致性损失形状不匹配: {bm_fake.shape} vs {bm_t.shape}"
            # 接缝一致性需要第二次前向(左裁 64 列), 预热期不做以省算力。
            loss_tile = (tile_boundary_loss(gen, inp, fake) if w_tile > 0
                         else fake.new_tensor(0.0))
            loss = (w_l1 * crit_l1(fake, target) + w_dist * crit_dist(fake, target)
                    + w_edge * crit_edge(fake, target) + w_fft * crit_fft(fake, target)
                    + w_lap * crit_lap(fake, target) + w_gan * loss_gan + w_fm * loss_fm
                    + w_cons * F.l1_loss(bm_fake, bm_t)
                    + w_slope * crit_slope(fake, target)
                    + w_tile * loss_tile)

            opt_g.zero_grad(); loss.backward(); opt_g.step()
            ema.update(gen)
            ep_g += loss.item()

        if w_gan > 0:
            for s in scheds[1:]:
                s.step()
        scheds[0].step()

        n = len(train_loader)
        avg_d = ep_d / (3 * n) if w_gan > 0 else 0.0
        avg_g = ep_g / n

        # ---- 验证 ----
        gen.eval()
        v_l1 = v_edge = v_dist = v_slope = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                fake = gen(inp)
                v_l1 += crit_l1(fake, target).item()
                v_edge += crit_edge(fake, target).item()
                v_dist += crit_dist(fake, target).item()
                v_slope += crit_slope(fake, target).item()
        m = len(val_loader)
        v_l1, v_edge, v_dist = v_l1 / m, v_edge / m, v_dist / m
        v_slope = v_slope / m
        # val_score 保持 v5 原口径, 便于与 block=8 权重直接对比
        val_score = v_l1 + 2.0 * v_edge + 5.0 * v_dist

        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch:02d}/{epochs}] | D: {avg_d:.4f} | G: {avg_g:.3f} | "
                  f"Val L1: {v_l1:.4f} | Edge: {v_edge:.4f} | Dist: {v_dist:.4f} | "
                  f"Slope: {v_slope:.4f} | "
                  f"w_GAN:{w_gan:.1f} w_L1:{w_l1:.0f} | {time.time()-t_ep:.0f}s/ep")

        if val_score < best_score:
            best_score, patience_counter = val_score, 0
            torch.save({'epoch': epoch, 'gen_state_dict': gen.state_dict(),
                        'gen_ema': ema.shadow,
                        'disc_fine': disc_fine.state_dict(), 'disc_mid': disc_mid.state_dict(),
                        'disc_coarse': disc_coarse.state_dict(),
                        'ch_stats_mean': ch_stats["mean"].tolist(),
                        'ch_stats_std': ch_stats["std"].tolist(),
                        'cfg': {'size': size, 'block': block, 'in_ch': 12, 'base': 32},
                        'val_score': best_score}, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发! 连续 {patience} 个 epoch 未改善。")
                break

    print(f"\n训练完成。最佳复合验证分: {best_score:.4f}")

    """
    # ==================== 可视化 (A9) ====================
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    gen.load_state_dict(ckpt['gen_ema'])
    gen.eval()
    cmap = make_terrain_cmap()
    post_rng = np.random.default_rng(7)

    idxs = [n_val // 6, n_val // 2, 5 * n_val // 6]
    fig, axes = plt.subplots(3, 5, figsize=(23, 13))
    col_titles = ["输入(块均值通道)", "目标 DEM (纬度雪线渲染)", "G 输出 raw",
                  "后处理成品 (雪线渲染)", "海拔分布对比"]
    for row, idx in enumerate(idxs):
        y0, x0, _ = val_set.centers[idx]
        m_tgt, lats = dem.patch(y0, x0, size)
        tgt_v01 = m2v(m_tgt)
        bm = block_reduce(tgt_v01, block, "mean")
        sd = block_reduce(tgt_v01, block, "std")
        lf = block_reduce((m_tgt > 0).astype(np.float32), block, "mean")

        ch_in, _ = val_set[idx]
        with torch.no_grad():
            g_raw = gen(ch_in.unsqueeze(0).to(device)).cpu().squeeze().numpy().astype(np.float32)
        g_pp = postprocess_pipeline(g_raw, bm, sd, lf, lats, post_rng, block)

        lat_c = float(lats[size // 2])
        lon360 = (x0 + size / 2) * 360.0 / dem.nx
        lon_c = lon360 - 360.0 if lon360 >= 180.0 else lon360
                # 原来: reshape(size, block, size, block)  ← 错, 需要 4M 元素
        Bs = size // block
        inp_vis = np.clip(ch_in[0].numpy().reshape(Bs, block, Bs, block).mean(axis=(1, 3))
                          * ch_stats["std"][0] + ch_stats["mean"][0], 0, 1)

        axes[row, 0].imshow(inp_vis, cmap='gray')
        axes[row, 1].imshow(render_snow_field(tgt_v01, lats, cmap))
        axes[row, 2].imshow(g_raw, cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].imshow(render_snow_field(g_pp, lats, cmap))
        axes[row, 4].hist(tgt_v01.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='目标', color='#2c7fb8')
        axes[row, 4].hist(g_pp.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='输出', color='#d95f0e')
        axes[row, 4].axvline(float(m2v(snowline_m(lat_c))), color='gray', ls='--', lw=0.8)
        t_max, o_max = float(v2m(tgt_v01.max())), float(v2m(g_pp.max()))
        axes[row, 4].text(0.03, 0.95, f"最高海拔 目标{t_max:.0f}m / 输出{o_max:.0f}m",
                          transform=axes[row, 4].transAxes, fontsize=8, va='top')
        if row == 0:
            for c, t in enumerate(col_titles):
                axes[row, c].set_title(t, fontsize=10)
            axes[row, 4].legend(fontsize=8)
        axes[row, 0].set_ylabel(f"({lat_c:.0f}°, {lon_c:.0f}°)", fontsize=9)
        for c in range(5):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([]); axes[row, c].tick_params(length=0)
        terrain_metrics(tgt_v01, g_pp, lats, bm, block, tag=f"({lat_c:.0f}°,{lon_c:.0f}°)")
    """
    run_viz(gen, dem, val_set, ch_stats, size, block, n_val)

    plt.suptitle("Terrain GAN v5.1: 包络通道 + 一致性校正 + 分位匹配 + 纬度雪线场", fontsize=14)
    plt.tight_layout()
    _out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(_out_dir, exist_ok=True)
    plt.savefig(os.path.join(_out_dir, "terrain_v5_result.png"),
                dpi=150, bbox_inches='tight')
    if os.isatty(1):          # 后台/无终端运行时 GUI 后端会阻塞, 跳过弹窗
        plt.show()
    plt.close('all')
    print("结果图已保存至 output/terrain_v5_result.png")


# ==================== 13. 冒烟测试 (A10) ====================
def smoke_test(dem, tect_provider=None):
    """冒烟测试：单步前向验证数据形状与 G/D 网络输出尺寸。"""
    stats = {"mean": np.zeros(12, np.float32), "std": np.ones(12, np.float32)}
    ds = TerrainDataset(dem, 256, 8, 2, stats, is_train=True, tect_provider=tect_provider)
    x, y = ds[0]
    g = UNet(in_ch=12).to(device)
    d = PatchDiscriminator(13).to(device)
    out = g(x[None].to(device))
    dout = d(torch.cat([x[None].to(device), y[None].to(device)], 1))
    print(f"[冒烟] 输入 {tuple(x.shape)} | 目标 {tuple(y.shape)} | "
          f"G输出 {tuple(out.shape)} | D输出 {tuple(dout.shape)}")
    assert x.shape == (12, 256, 256) and out.shape == (1, 1, 256, 256)
    print("[冒烟] 通过 ✓\n" + "-" * 60)


# ==================== 可视化循环 ====================
def run_viz(gen, dem, val_set, ch_stats, size, block, n_val,
            outfile=None):
    """生成 3 行×5 列对比可视化图（输入/目标/raw/成品/分布）并保存到 outfile。"""
    if outfile is None:
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "terrain_v5_result.png")
    cmap = make_terrain_cmap()
    post_rng = np.random.default_rng(7)
    idxs = [n_val // 6, n_val // 2, 5 * n_val // 6]
    fig, axes = plt.subplots(3, 5, figsize=(23, 13))
    col_titles = ["输入(块均值通道)", "目标 DEM (纬度雪线渲染)", "G 输出 raw",
                  "后处理成品 (雪线渲染)", "海拔分布对比"]
    for row, idx in enumerate(idxs):
        y0, x0, _ = val_set.centers[idx]
        m_tgt, lats = dem.patch(y0, x0, size)
        tgt_v01 = m2v(m_tgt)
        bm = block_reduce(tgt_v01, block, "mean")
        sd = block_reduce(tgt_v01, block, "std")
        lf = block_reduce((m_tgt > 0).astype(np.float32), block, "mean")

        ch_in, _ = val_set[idx]
        with torch.no_grad():
            g_raw = gen(ch_in.unsqueeze(0).to(device)).cpu().squeeze().numpy().astype(np.float32)
        g_pp = postprocess_pipeline(g_raw, bm, sd, lf, lats, post_rng, block)

        lat_c = float(lats[size // 2])
        lon360 = (x0 + size / 2) * 360.0 / dem.nx
        lon_c = lon360 - 360.0 if lon360 >= 180.0 else lon360
        Bs = size // block   # ← 修复点
        inp_vis = np.clip(ch_in[0].numpy().reshape(Bs, block, Bs, block).mean(axis=(1, 3))
                          * ch_stats["std"][0] + ch_stats["mean"][0], 0, 1)

        axes[row, 0].imshow(inp_vis, cmap='gray')
        axes[row, 1].imshow(render_snow_field(tgt_v01, lats, cmap))
        axes[row, 2].imshow(g_raw, cmap=cmap, vmin=0, vmax=1)
        axes[row, 3].imshow(render_snow_field(g_pp, lats, cmap))
        axes[row, 4].hist(tgt_v01.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='目标', color='#2c7fb8')
        axes[row, 4].hist(g_pp.ravel(), bins=60, range=(0, 1), density=True,
                          alpha=0.55, label='输出', color='#d95f0e')
        axes[row, 4].axvline(float(m2v(snowline_m(lat_c))), color='gray', ls='--', lw=0.8)
        t_max, o_max = float(v2m(tgt_v01.max())), float(v2m(g_pp.max()))
        axes[row, 4].text(0.03, 0.95, f"最高海拔 目标{t_max:.0f}m / 输出{o_max:.0f}m",
                          transform=axes[row, 4].transAxes, fontsize=8, va='top')
        if row == 0:
            for c, t in enumerate(col_titles):
                axes[row, c].set_title(t, fontsize=10)
            axes[row, 4].legend(fontsize=8)
        axes[row, 0].set_ylabel(f"({lat_c:.0f}°, {lon_c:.0f}°)", fontsize=9)
        for c in range(5):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([]); axes[row, c].tick_params(length=0)
        terrain_metrics(tgt_v01, g_pp, lats, bm, block, tag=f"({lat_c:.0f}°,{lon_c:.0f}°)")

    plt.suptitle("Terrain GAN v5.1: 包络通道 + 一致性校正 + 分位匹配 + 纬度雪线场", fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"结果图已保存至 {outfile}")


def visualize_only(dem, ckpt_path=None):
    """加载 checkpoint 后仅对验证集做可视化对比，不进行训练。"""
    ckpt_path = ckpt_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints", "best_terrain_gan_v5.pth")
    size, block, n_val = 256, 8, 120
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert int(ckpt.get("cfg", {}).get("in_ch", 0)) == 12, \
        "旧版 8 通道 checkpoint 与 v6.3 不兼容, 请删除后重训"
    tect_provider = make_tect_provider(dem)
    ch_stats = {"mean": np.array(ckpt["ch_stats_mean"], np.float32),
                "std": np.array(ckpt["ch_stats_std"], np.float32)}
    val_set = TerrainDataset(dem, size, block, n_val, ch_stats, is_train=False,
                             tect_provider=tect_provider)
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ckpt['gen_ema'])
    gen.eval()
    run_viz(gen, dem, val_set, ch_stats, size, block, n_val)

# ==================== 主入口 ====================
def main():
    """入口：加载/下载 DEM（失败用合成地形兜底），有 checkpoint 则可视化否则训练。"""
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = (os.path.dirname(_file_dir)
            if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)
    dem_path = os.environ.get("DEM_PATH",
                              os.path.join(_dir, "data", "ETOPO1_Ice_g_gmt4.grd"))
    dem = None
    if os.path.exists(dem_path):
        dem = GlobalDEM(dem_path)
    elif not dem_path.endswith((".npy", ".npz")):
        try:
            os.makedirs(os.path.dirname(dem_path), exist_ok=True)
            download_dem(dem_path)
            dem = GlobalDEM(dem_path)
        except Exception as e:
            print(f"自动下载失败: {e}")
    if dem is None:
        dem = SyntheticDEMSource()

    ckpt = os.path.join(_dir, "checkpoints", "best_terrain_gan_v5.pth")
    if os.path.exists(ckpt):
        print("发现已有 checkpoint → 仅可视化 (如需重训请删除 .pth)")
        visualize_only(dem)
    else:
        smoke_test(dem, make_tect_provider(dem))
        train(dem)


# ===========================================================================
# 模块化重构 (terrain.modules):
# 气温/气候/水系的物理实现已搬迁到 terrain/modules/*.py，以下为向后兼容再导出。
# 新代码请直接 from terrain.modules.xxx_module import ...
# ===========================================================================
from terrain.modules.temperature_module import (  # noqa: E402
    continentality_index as continentality_index,
    ocean_current_effects as ocean_current_effects,
    temperature_field as temperature_field,
)
from terrain.modules.climate_module import (  # noqa: E402
    _P_LAT as _P_LAT, _P_MM as _P_MM,
    precipitation_field as precipitation_field,
    BIOME_NAMES as BIOME_NAMES, BIOME_COLORS as BIOME_COLORS,
    whittaker_biome as whittaker_biome,
    smooth_biome_labels as smooth_biome_labels,
)
from terrain.modules.hydro_module import (  # noqa: E402
    _D8 as _D8,
    d8_flow_accumulation as d8_flow_accumulation,
    climate_flow_accumulation as climate_flow_accumulation,
    warp_river_corridor as warp_river_corridor,
    extract_river_centerlines as extract_river_centerlines,
    meander_centerlines as meander_centerlines,
    carve_along_centerlines as carve_along_centerlines,
    paint_rivers_rgb as paint_rivers_rgb,
    fill_depressions as fill_depressions,
    droplet_erosion as droplet_erosion,
    carve_trunk_rivers as carve_trunk_rivers,
)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时: {time.time() - t0:.1f}s")