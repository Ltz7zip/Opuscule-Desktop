"""地形生成质量评估指标（阶段 1）。

所有指标均接受 numpy float32 数组（单位米），返回 dict 或标量。

指标清单
--------
1. ``elev_stats``          : 基本高程分布统计
2. ``stripe_index``        : 方向性伪影（斑马条纹）指数
3. ``hypsometric_curve``   : 高程面积累积曲线（与地球参考对比）
4. ``power_spectrum``      : 径向功率谱，判断 1/f 自然性
5. ``slope_distribution``  : 坡度分布（与地球参考对比）
6. ``fid_terrain``         : 简化 FID（基于高阶统计，不依赖 Inception）
7. ``earth_plausibility``  : 综合地球合理性检查
8. ``compute_all_metrics`` : 一键计算全部指标

参考数据
--------
``REF_STATS`` 内置了地球真实 DEM 的典型分布参数（来自 SRTM/Copernicus 统计），
无需联网即可做合理性判断。如需更精确对比，可用 ``compute_reference_stats`` 从
本地真实 DEM 切片目录重新统计并缓存。
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np

# ==================== 地球参考统计（来自 SRTM/Copernicus 全球统计） ====================
REF_STATS = {
    "elev_min_m": -11000.0,        # 马里亚纳海沟附近
    "elev_max_m": 8800.0,          # 珠峰
    "elev_mean_m": -2300.0,        # 海洋占多数
    "elev_std_m": 3800.0,
    "land_ratio": 0.292,           # 地球陆地比例 ~29.2%
    "mean_slope_deg": 6.5,         # 全球平均坡度（含海洋较平）
    "psd_slope": -1.8,             # 自然地形 1/f^α，α ∈ [1.5, 2.2]
    "stripe_index": 1.0,           # 完全各向同性
}


# ==================== 1. 基本高程统计 ====================
def elev_stats(dem: np.ndarray) -> dict:
    """高程分布基本统计。

    Parameters
    ----------
    dem : ndarray (H, W)
        高程，单位米。

    Returns
    -------
    dict
        含 min/max/mean/std/land_ratio/range。
    """
    dem = np.asarray(dem, dtype=np.float32)
    return {
        "min": float(dem.min()),
        "max": float(dem.max()),
        "mean": float(dem.mean()),
        "std": float(dem.std()),
        "range": float(dem.max() - dem.min()),
        "land_ratio": float((dem > 0).mean()),
    }


# ==================== 2. 条纹指数（方向性伪影） ====================
def stripe_index(dem: np.ndarray, window: int | None = None) -> float:
    """各向同性指数 = 水平变率方差 / 垂直变率方差。

    1.0 = 完全各向同性（自然地形）
    >> 1 = 横向条纹（左右一致、上下突变，如 GAN 瓦片接缝）
    << 1 = 纵向条纹

    Parameters
    ----------
    dem : ndarray (H, W)
    window : int, optional
        若给定，计算局部窗口内的条纹指数再取均值，捕捉局部方向性。
    """
    dem = np.asarray(dem, dtype=np.float32)
    if window is None:
        return _stripe_idx_block(dem)

    H, W = dem.shape
    s = window
    if H < s or W < s:
        return _stripe_idx_block(dem)
    vals = []
    for i in range(0, H - s + 1, s):
        for j in range(0, W - s + 1, s):
            vals.append(_stripe_idx_block(dem[i:i + s, j:j + s]))
    return float(np.mean(vals))


def _stripe_idx_block(block: np.ndarray) -> float:
    """单块条纹指数。"""
    gy = np.diff(block, axis=0)
    gx = np.diff(block, axis=1)
    var_y = float(gy.var())
    var_x = float(gx.var())
    if var_y < 1e-12:
        return float("inf") if var_x > 1e-12 else 1.0
    return var_x / var_y


# ==================== 3. 高程曲线（Hypsometric Curve） ====================
def hypsometric_curve(dem: np.ndarray, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """高程-面积累积曲线。

    返回 (elevations, cumulative_area_fraction)，与地球参考对比可判断
    海陆比例与高程分布是否合理。

    Returns
    -------
    (elevs, cdf) : (n_bins,) ndarray
        elevs 为高程分位（米），cdf 为 <= 该高程的面积占比 [0,1]。
    """
    dem = np.asarray(dem, dtype=np.float32).ravel()
    elevs = np.linspace(dem.min(), dem.max(), n_bins)
    cdf = np.searchsorted(np.sort(dem), elevs) / dem.size
    return elevs, cdf


def hypsometric_error(dem: np.ndarray, ref_dem: np.ndarray | None = None,
                      n_bins: int = 100) -> float:
    """与参考 DEM 的高程曲线 L1 误差（面积占比差均值）。

    ref_dem 为 None 时与 REF_STATS 合成参考曲线对比。
    """
    e1, c1 = hypsometric_curve(dem, n_bins)
    if ref_dem is None:
        # 用 REF_STATS 构造简化参考曲线：双峰（海洋 + 陆地）
        ref_dem = _synth_ref_dem(n_bins * 10, REF_STATS)
    e2, c2 = hypsometric_curve(np.asarray(ref_dem, dtype=np.float32), n_bins)
    # 重采样到统一高程网格
    c2_interp = np.interp(e1, e2, c2)
    return float(np.abs(c1 - c2_interp).mean())


def _synth_ref_dem(n: int, ref: dict) -> np.ndarray:
    """根据 REF_STATS 合成参考高程序列（用于无真实 DEM 时对比）。"""
    rng = np.random.default_rng(0)
    out = np.concatenate([
        rng.normal(-4500, 1800, int(n * 0.65)),     # 深海
        rng.normal(-200, 300, int(n * 0.06)),        # 浅海/大陆架
        rng.normal(400, 600, int(n * 0.20)),         # 低地
        rng.normal(2000, 1200, int(n * 0.07)),       # 山地
        rng.normal(5000, 1500, int(n * 0.02)),       # 高山
    ])
    return np.clip(out, ref["elev_min_m"], ref["elev_max_m"])


# ==================== 4. 径向功率谱（1/f 自然性） ====================
def power_spectrum(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各向同性行政功率谱密度。

    返回 (frequencies, psd)。自然地形满足 PSD ∝ 1/f^α，α ∈ [1.5, 2.2]。
    α 过小 → 高频过多（噪声/棋盘格）；α 过大 → 过度平滑。

    Notes
    -----
    使用 Hann 窗减少频谱泄漏；返回径向平均（各方向平均）的 PSD。
    """
    dem = np.asarray(dem, dtype=np.float32)
    H, W = dem.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    f = np.fft.fftshift(np.fft.fft2(dem * win))
    psd = np.abs(f) ** 2

    # 径向平均
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = min(cy, cx)
    psd_radial = np.bincount(r.ravel(), psd.ravel(), minlength=rmax + 1)[:rmax + 1]
    cnt = np.bincount(r.ravel(), minlength=rmax + 1)[:rmax + 1]
    psd_radial = psd_radial / np.maximum(cnt, 1)

    freqs = np.arange(1, rmax + 1) / max(H, W)  # 归一化频率
    return freqs, psd_radial[1:]


def psd_slope(dem: np.ndarray) -> float:
    """功率谱对数斜率 α（PSD ∝ 1/f^α）。

    Returns
    -------
    float
        α，自然地形约 1.5~2.2；GAN 接缝常表现为 α < 1 或高频尖峰。
    """
    freqs, psd = power_spectrum(dem)
    mask = (freqs > 0) & (psd > 0)
    if mask.sum() < 8:
        return 0.0
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    # 线性拟合 log_p = -α * log_f + c
    slope, _ = np.polyfit(log_f, log_p, 1)
    return float(-slope)


# ==================== 5. 坡度分布 ====================
def slope_distribution(dem: np.ndarray, deg: bool = True) -> np.ndarray:
    """坡度分布（直方图 bin 平均，单位度）。

    用中心差分计算梯度，再 arctan 转角度。
    """
    dem = np.asarray(dem, dtype=np.float32)
    gy, gx = np.gradient(dem)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    if deg:
        return np.degrees(np.arctan(grad))
    return grad


def mean_slope(dem: np.ndarray) -> float:
    """平均坡度（度）。"""
    return float(slope_distribution(dem).mean())


# ==================== 6. 简化 FID（不依赖 Inception） ====================
def _feature_stats(dem: np.ndarray) -> dict:
    """从单张 DEM 提取特征统计向量。

    使用多尺度高斯滤波响应的均值/方差作为"特征"，避免依赖 Inception 网络。
    尺度 σ ∈ {1, 2, 4, 8, 16} 像素，捕获从细节到结构的多尺度信息。
    """
    from scipy.ndimage import gaussian_filter
    dem = np.asarray(dem, dtype=np.float32)
    feats = []
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0):
        blurred = gaussian_filter(dem, sigma)
        residual = dem - blurred
        feats.extend([blurred.mean(), blurred.std(),
                      residual.mean(), residual.std()])
    # 加入全局统计
    feats.extend([dem.mean(), dem.std(), dem.min(), dem.max()])
    return {"feats": np.array(feats, dtype=np.float32)}


def _pool_stats(dems: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """从一批 DEM 计算特征均值 μ 与协方差 Σ。"""
    feats = np.stack([_feature_stats(d)["feats"] for d in dems])
    mu = feats.mean(axis=0)
    if feats.shape[0] < 2:
        sigma = np.eye(feats.shape[1]) * 1e-3
    else:
        sigma = np.cov(feats, rowvar=False)
        sigma += np.eye(sigma.shape[0]) * 1e-3   # 正则化
    return mu, sigma


def fid_terrain(samples: Iterable[np.ndarray],
                references: Iterable[np.ndarray]) -> float:
    """简化版 FID，基于多尺度高斯特征而非 Inception。

    FID = ||μ_s - μ_r||² + Tr(Σ_s + Σ_r - 2·(Σ_s·Σ_r)^0.5)

    Parameters
    ----------
    samples : 生成样本列表（每个 (H,W) ndarray，米）
    references : 真实 DEM 参考样本列表

    Returns
    -------
    float
        FID，越小越接近真实分布。注意：与标准 Inception FID 数值不可比，
        仅用于相对比较（同一参考集下对比不同模型）。
    """
    mu_s, sigma_s = _pool_stats(samples)
    mu_r, sigma_r = _pool_stats(references)

    diff = mu_s - mu_r
    # 矩阵平方根（用 SVD 近似，避免 scipy.sqrtm 数值问题）
    sqrt_product = _sqrtm_psd(sigma_s @ sigma_r)
    fid = float(diff @ diff + np.trace(sigma_s + sigma_r - 2 * sqrt_product))
    return max(0.0, fid)


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """对称半正定矩阵平方根（SVD 近似）。"""
    mat = (mat + mat.T) / 2
    u, s, _ = np.linalg.svd(mat)
    s = np.clip(s, 0, None)
    return (u * np.sqrt(s)) @ u.T


# ==================== 7. 综合地球合理性 ====================
def earth_plausibility(dem: np.ndarray, ref: dict = REF_STATS) -> dict:
    """综合地球合理性检查。

    返回每个检查项的通过状态与原始值，便于打印/记录。
    """
    s = elev_stats(dem)
    si = stripe_index(dem)
    alpha = psd_slope(dem)
    ms = mean_slope(dem)
    return {
        # 原始指标
        **s,
        "stripe_index": si,
        "psd_slope": alpha,
        "mean_slope_deg": ms,
        # 合理性布尔
        "elev_in_range": ref["elev_min_m"] <= s["min"] and s["max"] <= ref["elev_max_m"],
        "land_ratio_ok": 0.15 < s["land_ratio"] < 0.45,
        "stripe_ok": 0.7 < si < 1.4,
        "psd_natural": 1.2 < alpha < 2.5,
        "slope_reasonable": 2.0 < ms < 15.0,
    }


# ==================== 8. 一键计算全部指标 ====================
def compute_all_metrics(dem: np.ndarray,
                        ref_dems: list[np.ndarray] | None = None) -> dict:
    """计算全部可用指标，返回扁平 dict（适合 CSV 记录）。

    Parameters
    ----------
    dem : ndarray (H, W)
        待评估高程（米）。
    ref_dems : list[ndarray], optional
        真实参考 DEM 列表；给定则计算 FID，否则跳过。
    """
    out = elev_stats(dem)
    out["stripe_index"] = stripe_index(dem)
    out["psd_slope"] = psd_slope(dem)
    out["mean_slope_deg"] = mean_slope(dem)
    out["hypsometric_err"] = hypsometric_error(dem)
    if ref_dems:
        out["fid"] = fid_terrain([dem], ref_dems)
    return out


# ==================== 参考统计缓存 ====================
def compute_reference_stats(dem_dir: Path, max_n: int = 200,
                            cache_path: Path | None = None) -> tuple[np.ndarray, dict]:
    """从本地真实 DEM 切片目录计算参考统计并缓存。

    Returns
    -------
    (ref_dems, stats) : (list[ndarray], dict)
        ref_dems 为原始切片（供 FID 用），stats 为聚合统计。
    """
    from .config import CACHE_DIR
    if cache_path is None:
        cache_path = CACHE_DIR / "ref_stats.npz"

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return list(z["ref_dems"]), dict(z["stats"].item())

    files = sorted(Path(dem_dir).glob("*.npy"))[:max_n]
    if not files:
        raise RuntimeError(f"{dem_dir} 下无 .npy 切片")
    ref_dems = [np.load(f).astype(np.float32) for f in files]

    all_flat = np.concatenate([d.ravel() for d in ref_dems])
    slopes = np.concatenate([slope_distribution(d).ravel() for d in ref_dems])
    psd_alphas = [psd_slope(d) for d in ref_dems]
    stats = {
        "elev_min_m": float(all_flat.min()),
        "elev_max_m": float(all_flat.max()),
        "elev_mean_m": float(all_flat.mean()),
        "elev_std_m": float(all_flat.std()),
        "land_ratio": float((all_flat > 0).mean()),
        "mean_slope_deg": float(slopes.mean()),
        "psd_slope": float(np.mean(psd_alphas)),
        "stripe_index": 1.0,
    }
    np.savez_compressed(cache_path, ref_dems=np.array(ref_dems, dtype=object),
                        stats=np.array(stats, dtype=object))
    return ref_dems, stats
