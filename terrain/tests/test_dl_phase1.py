"""阶段 1 深度学习模块测试。

覆盖：
- 高程归一化可逆
- VAE / U-Net 前向形状
- DDPM 前向加噪
- 全部评估指标在合成数据上的行为
- Evaluator 周期评估流程

不依赖训练好的权重或真实数据集。
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from terrain.dl.config import (elev_to_norm, norm_to_elev, TILE_SIZE,
                               IN_CHANNELS, LATENT_CHANNELS, LATENT_SIZE)
from terrain.dl.vae import TerrainVAE, vae_loss
from terrain.dl.unet import DiffusionUNet
from terrain.dl.diffusion import DDPM
from terrain.dl.metrics import (
    elev_stats, stripe_index, hypsometric_curve, hypsometric_error,
    power_spectrum, psd_slope, slope_distribution, mean_slope,
    fid_terrain, earth_plausibility, compute_all_metrics, REF_STATS,
)
from terrain.dl.evaluator import Evaluator


# ==================== 归一化 ====================

def test_elev_normalization_roundtrip():
    """归一化可逆（float32 精度）。"""
    dem = np.array([[-1000.0, 0.0, 4000.0, -8000.0]], dtype=np.float32)
    rt = norm_to_elev(elev_to_norm(dem))
    assert np.allclose(dem, rt, atol=1e-2)


def test_norm_range():
    """[-12000, 9000] 应映射到 [-1, 1]。"""
    assert abs(elev_to_norm(-12000.0) + 1.0) < 1e-6
    assert abs(elev_to_norm(9000.0) - 1.0) < 1e-6
    # -1500 恰好是 [-12000, 9000] 的中点 → 映射到 0；改用 -3000 测非零
    assert abs(elev_to_norm(-3000.0)) > 1e-6


# ==================== 模型形状 ====================

def test_vae_shapes():
    """VAE 编码/解码形状正确。"""
    vae = TerrainVAE(in_ch=IN_CHANNELS, latent_ch=LATENT_CHANNELS)
    x = torch.randn(2, IN_CHANNELS, TILE_SIZE, TILE_SIZE)
    mu, logvar = vae.encode(x)
    assert mu.shape == (2, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
    out = vae.decode(mu)
    assert out.shape == x.shape


def test_vae_loss_runs():
    """VAE 损失返回标量与 info dict。"""
    vae = TerrainVAE()
    x = torch.randn(2, 1, TILE_SIZE, TILE_SIZE)
    recon, mu, logvar = vae(x)
    loss, info = vae_loss(recon, x, mu, logvar)
    assert loss.dim() == 0
    assert "rec" in info and "kld" in info


def test_unet_shapes():
    """U-Net 接受时间步嵌入，输出形状不变。"""
    model = DiffusionUNet(in_ch=IN_CHANNELS)
    x = torch.randn(2, IN_CHANNELS, TILE_SIZE, TILE_SIZE)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == x.shape


# ==================== DDPM 调度 ====================

def test_ddpm_q_sample_endpoints():
    """t=0 接近原图，t 大时接近噪声。"""
    ddpm = DDPM(n_steps=1000, device="cpu")
    x0 = torch.ones(1, 1, 8, 8)
    xt0, _ = ddpm.q_sample(x0, torch.tensor([0]))
    assert torch.allclose(xt0, x0, atol=0.05)
    xt999, _ = ddpm.q_sample(x0, torch.tensor([999]))
    assert xt999.std() > 0.5


def test_ddpm_p_loss_finite():
    """训练损失有限。"""
    ddpm = DDPM(n_steps=100, device="cpu")
    model = DiffusionUNet(in_ch=1)
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 100, (2,))
    loss = ddpm.p_loss(model, x, t)
    assert torch.isfinite(loss)


# ==================== 指标 ====================

def test_elev_stats_basic():
    dem = np.array([[0.0, 100.0], [200.0, -50.0]], dtype=np.float32)
    s = elev_stats(dem)
    assert s["min"] == -50.0
    assert s["max"] == 200.0
    # >0 的元素：100, 200；0 不计入 >0；故 2/4 = 0.5
    assert s["land_ratio"] == 0.5


def test_stripe_index_isotropic():
    """各向同性高斯噪声的条纹指数接近 1。"""
    rng = np.random.default_rng(0)
    dem = rng.normal(0, 100, (128, 128)).astype(np.float32)
    si = stripe_index(dem)
    assert 0.6 < si < 1.4


def test_stripe_index_detects_vertical_stripes():
    """竖直条纹（列间突变、列内一致）应使指数 >> 1。

    注：stripe_index = var(水平差分) / var(垂直差分)
        - 竖直条纹 → 水平差分大、垂直差分小 → ratio >> 1
        - 水平条纹 → 水平差分小、垂直差分大 → ratio << 1
    """
    base = np.zeros((64, 64), dtype=np.float32)
    base[:, ::4] += 1000.0   # 每 4 列加高程 → 竖直条纹
    si = stripe_index(base)
    assert si > 5.0


def test_hypsometric_curve_monotonic():
    """高程曲线 CDF 单调不减。"""
    dem = np.random.default_rng(0).normal(500, 200, (64, 64)).astype(np.float32)
    elevs, cdf = hypsometric_curve(dem, n_bins=50)
    assert np.all(np.diff(cdf) >= -1e-9)
    assert cdf[0] >= 0 and cdf[-1] <= 1.0 + 1e-6


def test_hypsometric_error_self_zero():
    """同一 DEM 与自身的高程曲线误差接近 0。"""
    dem = np.random.default_rng(0).normal(500, 200, (64, 64)).astype(np.float32)
    err = hypsometric_error(dem, ref_dem=dem)
    assert err < 1e-3


def test_power_spectrum_shape():
    dem = np.random.default_rng(0).normal(0, 100, (64, 64)).astype(np.float32)
    freqs, psd = power_spectrum(dem)
    assert freqs.shape == psd.shape
    assert (psd > 0).all()


def test_psd_slope_detects_decay():
    """低通滤波图像的 PSD 斜率应为正（高频衰减）。

    注：合成 1/f^α 地形对数值实现细节敏感，此处仅验证 psd_slope 能检测到
    衰减（α > 0），而非精确数值。
    """
    rng = np.random.default_rng(0)
    # 强低通滤波：构造明显高频衰减的图像
    noise = rng.normal(0, 1, (128, 128))
    from scipy.ndimage import gaussian_filter
    dem = gaussian_filter(noise, sigma=4.0).astype(np.float32) * 1000
    alpha = psd_slope(dem)
    assert alpha > 0.5  # 平滑图像应有明显高频衰减


def test_slope_distribution_range():
    dem = np.zeros((32, 32), dtype=np.float32)
    dem[16:, :] = 1000.0   # 一半高一半低 → 边界坡度大
    slopes = slope_distribution(dem)
    assert slopes.min() >= 0
    assert slopes.max() <= 90.0


def test_mean_slope_flat_is_zero():
    dem = np.full((32, 32), 100.0, dtype=np.float32)
    assert abs(mean_slope(dem)) < 1e-3


# ==================== FID ====================

def test_fid_self_zero():
    """同分布与自身的 FID 接近 0。"""
    rng = np.random.default_rng(0)
    samples = [rng.normal(500, 200, (32, 32)).astype(np.float32) for _ in range(8)]
    fid = fid_terrain(samples, samples)
    assert fid < 1.0


def test_fid_different_distributions_positive():
    """不同分布 FID 应为正。"""
    rng = np.random.default_rng(0)
    a = [rng.normal(0, 100, (32, 32)).astype(np.float32) for _ in range(8)]
    b = [rng.normal(5000, 100, (32, 32)).astype(np.float32) for _ in range(8)]
    fid = fid_terrain(a, b)
    assert fid > 1.0


# ==================== 综合指标 ====================

def test_earth_plausibility_returns_all_keys():
    dem = np.random.default_rng(0).normal(500, 300, (64, 64)).astype(np.float32)
    info = earth_plausibility(dem)
    for k in ("min", "max", "land_ratio", "stripe_index", "psd_slope",
              "elev_in_range", "land_ratio_ok", "stripe_ok", "psd_natural"):
        assert k in info


def test_compute_all_metrics_runs():
    dem = np.random.default_rng(0).normal(500, 300, (64, 64)).astype(np.float32)
    out = compute_all_metrics(dem)
    assert "stripe_index" in out
    assert "psd_slope" in out
    assert "hypsometric_err" in out


def test_compute_all_metrics_with_fid():
    dem = np.random.default_rng(0).normal(500, 300, (32, 32)).astype(np.float32)
    refs = [np.random.default_rng(i).normal(500, 300, (32, 32)).astype(np.float32)
            for i in range(6)]
    out = compute_all_metrics(dem, ref_dems=refs)
    assert "fid" in out


# ==================== Evaluator ====================

def test_evaluator_step_runs(tmp_path):
    """Evaluator 周期评估流程跑通。"""
    from terrain.dl.logger import TrainLogger
    logger = TrainLogger("test_run", log_dir=tmp_path, use_tb=False)
    evaluator = Evaluator("test_run", sample_dir=tmp_path / "samples",
                          logger=logger)

    def sample_fn(n):
        rng = np.random.default_rng(0)
        return rng.normal(500, 300, (n, 32, 32)).astype(np.float32)

    result = evaluator.step(epoch=0, step=10, sample_fn=sample_fn, n=4)
    assert "quality_score" in result
    assert "stripe_index" in result
    assert result["epoch"] == 0
    logger.close()
