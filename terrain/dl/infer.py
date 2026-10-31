"""阶段 1 推理接口：用训练好的 DDPM/VAE 生成地形切片。

阶段 1 仅无条件生成（验证训练分布学习）；阶段 2 将扩展为条件生成。
"""
from __future__ import annotations
import numpy as np
import torch

from .config import (DEVICE, CKPT_DIR, DIFFUSION_STEPS, BETA_START, BETA_END,
                     IN_CHANNELS, TILE_SIZE, norm_to_elev, ELEV_MIN, ELEV_MAX)
from .unet import DiffusionUNet
from .diffusion import DDPM
from .vae import TerrainVAE
from .train import EMA


def generate_tile_ddpm(batch: int = 4, seed: int | None = None,
                       ckpt: str | None = None,
                       size: int = TILE_SIZE) -> np.ndarray:
    """用 DDPM 生成 (batch, size, size) 高程切片（米）。

    Parameters
    ----------
    batch : int
        生成数量。
    seed : int, optional
        随机种子；None 不固定。
    ckpt : str, optional
        权重路径，默认 ``checkpoints/ddpm_phase1.pt``。
    size : int
        输出尺寸。
    """
    if seed is not None:
        torch.manual_seed(seed)
    ckpt_path = ckpt or str(CKPT_DIR / "ddpm_phase1.pt")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

    model = DiffusionUNet(in_ch=IN_CHANNELS).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    ema = EMA(model)
    if "ema" in state:
        ema.shadow = state["ema"]

    ddpm = DDPM(n_steps=DIFFUSION_STEPS, beta_start=BETA_START,
                beta_end=BETA_END, device=DEVICE)

    ema.apply(model)
    try:
        x = ddpm.sample(model, (batch, IN_CHANNELS, size, size), DEVICE,
                        progress=True)
    finally:
        ema.restore(model)

    x = x.squeeze(1).cpu().numpy()        # (B, H, W) ∈ [-1,1]
    return np.clip(norm_to_elev(x), ELEV_MIN, ELEV_MAX)   # 米，约束到物理高程范围


def generate_tile_vae(batch: int = 4, seed: int | None = None,
                      ckpt: str | None = None) -> np.ndarray:
    """用 VAE 先验采样生成 (batch, 256, 256) 高程切片（米）。"""
    from .config import LATENT_CHANNELS, LATENT_SIZE
    if seed is not None:
        torch.manual_seed(seed)
    ckpt_path = ckpt or str(CKPT_DIR / "vae_phase1.pt")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

    vae = TerrainVAE(in_ch=IN_CHANNELS, latent_ch=LATENT_CHANNELS).to(DEVICE)
    vae.load_state_dict(state["model"])
    vae.eval()
    ema = EMA(vae)
    if "ema" in state:
        ema.shadow = state["ema"]

    ema.apply(vae)
    try:
        with torch.no_grad():
            z = torch.randn(batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                            device=DEVICE)
            x = vae.decode(z)
    finally:
        ema.restore(vae)

    x = x.squeeze(1).cpu().numpy()
    return np.clip(norm_to_elev(x), ELEV_MIN, ELEV_MAX)


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "ddpm"
    fn = generate_tile_ddpm if kind == "ddpm" else generate_tile_vae
    dems = fn(batch=4, seed=42)
    print(f"shape={dems.shape} range=[{dems.min():.0f}, {dems.max():.0f}]m")
    from . import metrics as M
    for i, d in enumerate(dems):
        info = M.earth_plausibility(d)
        print(f"sample {i}: stripe={info['stripe_index']:.3f} "
              f"land={info['land_ratio']:.3f} psd={info['psd_slope']:.3f}")
