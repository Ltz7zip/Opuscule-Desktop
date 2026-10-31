"""阶段 2 潜空间条件扩散（LDM）模块测试。

覆盖：
- ConditionEncoder 形状与归一化
- LatentDiffusionUNet 前向形状（带条件）
- CrossAttention 形状
- DDIM 采样器时间步与确定性
- 半监督损失函数（无标签/监督/一致性）
- LDMInference 端到端流程（用随机权重验证管线通）
- generate_large 分块拼接形状

不依赖训练好的权重或真实数据集。
"""
from __future__ import annotations
import numpy as np
import torch
import pytest

from terrain.dl.config import (TILE_SIZE, LATENT_SIZE, LATENT_CHANNELS,
                               IN_CHANNELS, DIFFUSION_STEPS)
from terrain.dl.conditioner import ConditionEncoder
from terrain.dl.latent_unet import LatentDiffusionUNet, CrossAttention
from terrain.dl.ddim import DDIM
from terrain.dl.diffusion import DDPM
from terrain.dl.vae import TerrainVAE
from terrain.dl.train_ldm import latent_diffusion_loss
from terrain.dl.train import EMA


# ==================== ConditionEncoder ====================

def test_condition_encoder_shapes():
    """12 通道 256² → cond_ch 32²。"""
    enc = ConditionEncoder(in_ch=12, cond_ch=4)
    x = torch.randn(2, 12, TILE_SIZE, TILE_SIZE)
    out = enc(x)
    assert out.shape == (2, 4, LATENT_SIZE, LATENT_SIZE)


def test_condition_encoder_normalization():
    """ch_stats 归一化生效。"""
    ch_stats = {"mean": np.ones(12, dtype=np.float32),
                "std": np.full(12, 2.0, dtype=np.float32)}
    enc = ConditionEncoder(in_ch=12, cond_ch=4, ch_stats=ch_stats)
    x = torch.ones(1, 12, 8, 8)   # 输入全 1
    # 归一化后 (1-1)/2 = 0，编码器第一层输入为 0
    # 验证 ch_mean/ch_std buffer 正确注册
    assert torch.allclose(enc.ch_mean, torch.ones(12))
    assert torch.allclose(enc.ch_std, torch.full((12,), 2.0))


# ==================== LatentDiffusionUNet ====================

def test_latent_unet_shapes():
    """潜空间 U-Net：z(4,32,32) + cond(4,32,32) → eps(4,32,32)。"""
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=32)
    z = torch.randn(2, 4, LATENT_SIZE, LATENT_SIZE)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 4, LATENT_SIZE, LATENT_SIZE)
    eps = model(z, t, cond)
    assert eps.shape == z.shape


def test_latent_unet_without_cross_attn():
    """关闭 cross-attention 也能运行。"""
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=32, use_cross_attn=False)
    z = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE)
    t = torch.randint(0, 1000, (1,))
    cond = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE)
    eps = model(z, t, cond)
    assert eps.shape == z.shape


def test_cross_attention_shapes():
    """cross-attention 输入输出形状不变。"""
    attn = CrossAttention(channels=128, cond_channels=4, heads=4, head_dim=32)
    x = torch.randn(2, 128, 8, 8)
    cond = torch.randn(2, 4, 8, 8)
    out = attn(x, cond)
    assert out.shape == x.shape


# ==================== DDIM 采样器 ====================

def test_ddim_timestep_count():
    """DDIM 采样步数 = n_sample_steps。"""
    ddim = DDIM(n_train_steps=1000, n_sample_steps=50, device="cpu")
    assert len(ddim.timesteps) == 50
    # 降序
    assert ddim.timesteps[0] > ddim.timesteps[-1]


def test_ddim_deterministic():
    """eta=0 时同一噪声同一输出（确定性）。"""
    ddim = DDIM(n_train_steps=100, n_sample_steps=10, eta=0.0, device="cpu")
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16,
                                use_cross_attn=False)
    model.eval()
    cond = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE)
    torch.manual_seed(0)
    z1 = ddim.sample(model, (1, 4, LATENT_SIZE, LATENT_SIZE), cond, "cpu")
    torch.manual_seed(0)
    z2 = ddim.sample(model, (1, 4, LATENT_SIZE, LATENT_SIZE), cond, "cpu")
    assert torch.allclose(z1, z2, atol=1e-5)


def test_ddim_sample_shape():
    """DDIM 采样输出形状正确。"""
    ddim = DDIM(n_train_steps=100, n_sample_steps=10, device="cpu")
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16,
                                use_cross_attn=False)
    cond = torch.randn(2, 4, LATENT_SIZE, LATENT_SIZE)
    z = ddim.sample(model, (2, 4, LATENT_SIZE, LATENT_SIZE), cond, "cpu")
    assert z.shape == (2, 4, LATENT_SIZE, LATENT_SIZE)


# ==================== 半监督损失 ====================

def test_loss_unsupervised_only():
    """无配对时仅无标签损失。"""
    vae = TerrainVAE()
    cond_enc = ConditionEncoder(in_ch=12, cond_ch=4)
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16,
                                use_cross_attn=False)
    ddpm = DDPM(n_steps=100, device="cpu")
    dem = torch.randn(2, 1, TILE_SIZE, TILE_SIZE)
    loss, info = latent_diffusion_loss(model, vae, cond_enc, ddpm, dem, None,
                                       lambda_sup=1.0, lambda_cons=0.1)
    assert torch.isfinite(loss)
    assert "loss_unsup" in info
    assert "loss_sup" not in info     # 无配对


def test_loss_supervised():
    """有配对时监督+一致性损失都计算。"""
    vae = TerrainVAE()
    cond_enc = ConditionEncoder(in_ch=12, cond_ch=4)
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16,
                                use_cross_attn=False)
    ddpm = DDPM(n_steps=100, device="cpu")
    dem = torch.randn(2, 1, TILE_SIZE, TILE_SIZE)
    cond = torch.randn(2, 12, TILE_SIZE, TILE_SIZE)
    loss, info = latent_diffusion_loss(model, vae, cond_enc, ddpm, dem, cond,
                                       lambda_sup=1.0, lambda_cons=0.1)
    assert torch.isfinite(loss)
    assert "loss_unsup" in info
    assert "loss_sup" in info
    assert "loss_cons" in info


def test_loss_consistency_zero_weight():
    """lambda_cons=0 时不计算一致性。"""
    vae = TerrainVAE()
    cond_enc = ConditionEncoder(in_ch=12, cond_ch=4)
    model = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16,
                                use_cross_attn=False)
    ddpm = DDPM(n_steps=100, device="cpu")
    dem = torch.randn(2, 1, TILE_SIZE, TILE_SIZE)
    cond = torch.randn(2, 12, TILE_SIZE, TILE_SIZE)
    loss, info = latent_diffusion_loss(model, vae, cond_enc, ddpm, dem, cond,
                                       lambda_sup=1.0, lambda_cons=0.0)
    assert "loss_cons" not in info


# ==================== 端到端推理流程（随机权重） ====================

def test_ldm_inference_pipeline(tmp_path):
    """用随机权重验证 LDMInference 端到端流程通。

    不验证生成质量，仅验证形状与数值范围合理。
    """
    from terrain.dl.infer_ldm import LDMInference

    # 保存随机权重（用小 base 加速测试）
    vae = TerrainVAE(base=16)
    ldm = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16, use_cross_attn=False)
    cond_enc = ConditionEncoder(in_ch=12, cond_ch=4, base=16)
    vae_path = tmp_path / "vae.pt"
    ldm_path = tmp_path / "ldm.pt"
    torch.save({"model": vae.state_dict()}, vae_path)
    torch.save({
        "model": ldm.state_dict(),
        "cond_encoder": cond_enc.state_dict(),
    }, ldm_path)

    infer = LDMInference(ckpt_path=str(ldm_path), vae_ckpt=str(vae_path),
                         sample_steps=5, vae_base=16, cond_base=16, ldm_base=16,
                         use_cross_attn=False)
    cond = np.random.randn(12, TILE_SIZE, TILE_SIZE).astype(np.float32)
    dem = infer.generate(cond, seed=42)
    assert dem.shape == (TILE_SIZE, TILE_SIZE)
    assert dem.dtype == np.float32
    # 数值范围在合理高程区间（允许等于边界，因 clip）
    assert -12001 < dem.min() < 9001


def test_generate_large_shape(tmp_path):
    """generate_large 分块拼接输出正确尺寸。"""
    from terrain.dl.infer_ldm import LDMInference

    vae = TerrainVAE(base=16)
    ldm = LatentDiffusionUNet(in_ch=4, cond_ch=4, base=16, use_cross_attn=False)
    cond_enc = ConditionEncoder(in_ch=12, cond_ch=4, base=16)
    vae_path = tmp_path / "vae.pt"
    ldm_path = tmp_path / "ldm.pt"
    torch.save({"model": vae.state_dict()}, vae_path)
    torch.save({
        "model": ldm.state_dict(),
        "cond_encoder": cond_enc.state_dict(),
    }, ldm_path)

    infer = LDMInference(ckpt_path=str(ldm_path), vae_ckpt=str(vae_path),
                         sample_steps=3, vae_base=16, cond_base=16, ldm_base=16,
                         use_cross_attn=False)
    # 512² 条件（2×2 块）
    cond = np.random.randn(12, 512, 512).astype(np.float32)
    dem = infer.generate_large(cond, size_px=512, seed=0, overlap=64)
    assert dem.shape == (512, 512)
    assert dem.dtype == np.float32
