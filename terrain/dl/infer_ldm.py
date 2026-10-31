"""阶段 2 LDM 推理接口：用训练好的潜空间条件扩散生成地形。

核心能力
--------
1. **整图生成**：直接生成 2048² DEM，无需逐瓦片拼贴（消除接缝）
2. **条件控制**：输入 12 通道粗底图（板块骨架），控制生成走向
3. **DDIM 加速**：50 步采样代替 1000 步，推理 < 10 秒（MPS）
4. **与现有引擎集成**：``generate_world(engine="diffusion")`` 调用

工作流
------
    plate_sim → world_to_channels (12ch 粗底图) → ConditionEncoder
              → DDIM 采样 (潜空间) → VAE 解码 → 米制 DEM

若 2048² 单次生成显存不足，自动降级为 4 块 1024² 带重叠拼接
（重叠区用潜空间加权混合，比 GAN 余弦羽化更平滑）。
"""
from __future__ import annotations
import numpy as np
import torch

from .config import (DEVICE, CKPT_DIR, DIFFUSION_STEPS, BETA_START, BETA_END,
                     IN_CHANNELS, LATENT_CHANNELS, LATENT_SIZE, TILE_SIZE,
                     norm_to_elev, ELEV_MIN, ELEV_MAX)
from .vae import TerrainVAE
from .conditioner import ConditionEncoder
from .latent_unet import LatentDiffusionUNet
from .ddim import DDIM
from .train import EMA


class LDMInference:
    """封装 LDM 推理的完整流程（VAE + 条件编码器 + 去噪 U-Net + DDIM）。

    一次加载，多次调用 ``generate()`` 复用模型。

    Parameters
    ----------
    ckpt_path : str, optional
        LDM 权重路径，默认 checkpoints/ldm_phase2.pt。
    vae_ckpt : str, optional
        VAE 权重路径，默认 checkpoints/vae_phase1.pt。
    sample_steps : int
        DDIM 采样步数，默认 50。
    """

    def __init__(self, ckpt_path: str | None = None,
                 vae_ckpt: str | None = None,
                 sample_steps: int = 50,
                 vae_base: int = 64,
                 cond_base: int = 64,
                 ldm_base: int = 128,
                 use_cross_attn: bool = True):
        """加载 VAE、条件编码器、去噪 U-Net 权重（含 EMA 影子）并构建 DDIM 采样器。"""
        # ---- 加载 VAE ----
        vae_path = vae_ckpt or str(CKPT_DIR / "vae_phase1.pt")
        self.vae = TerrainVAE(in_ch=IN_CHANNELS, latent_ch=LATENT_CHANNELS,
                              base=vae_base).to(DEVICE)
        self.vae.load_state_dict(
            torch.load(vae_path, map_location=DEVICE, weights_only=True)["model"])
        self.vae.eval()

        # ---- 加载 LDM ----
        ckpt = ckpt_path or str(CKPT_DIR / "ldm_phase2.pt")
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)

        self.cond_encoder = ConditionEncoder(in_ch=12,
                                             cond_ch=LATENT_CHANNELS,
                                             base=cond_base).to(DEVICE)
        self.cond_encoder.load_state_dict(state["cond_encoder"])
        self.cond_encoder.eval()

        self.model = LatentDiffusionUNet(in_ch=LATENT_CHANNELS,
                                         cond_ch=LATENT_CHANNELS,
                                         base=ldm_base,
                                         use_cross_attn=use_cross_attn).to(DEVICE)
        self.model.load_state_dict(state["model"])
        self.model.eval()

        # EMA 权重（若有）
        self.ema_model = EMA(self.model)
        self.ema_cond = EMA(self.cond_encoder)
        if "ema_model" in state:
            self.ema_model.shadow = state["ema_model"]
        if "ema_cond" in state:
            self.ema_cond.shadow = state["ema_cond"]

        # DDIM 采样器
        self.ddim = DDIM(n_train_steps=DIFFUSION_STEPS,
                         n_sample_steps=sample_steps,
                         beta_start=BETA_START, beta_end=BETA_END,
                         eta=0.0, device=DEVICE)

    @torch.no_grad()
    def generate(self, cond_12ch: np.ndarray | torch.Tensor,
                 seed: int | None = None,
                 tile_size: int = 256) -> np.ndarray:
        """条件生成 DEM。

        Parameters
        ----------
        cond_12ch : ndarray or Tensor (12, H_cond, W_cond)
            12 通道条件（来自 world_to_channels，H_cond 通常 = 256）。
            自动上采样到 256² 供 ConditionEncoder。
        seed : int, optional
            随机种子。
        tile_size : int
            输出 DEM 尺寸（默认 256²，对应单块）。

        Returns
        -------
        ndarray (tile_size, tile_size) float32
            米制高程，范围 [ELEV_MIN, ELEV_MAX]。
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 准备条件张量 (1, 12, 256, 256)
        if isinstance(cond_12ch, np.ndarray):
            cond = torch.from_numpy(cond_12ch).float()
        else:
            cond = cond_12ch.float()
        if cond.dim() == 3:
            cond = cond.unsqueeze(0)   # (1, 12, H, W)
        # 上采样到 256²（ConditionEncoder 输入要求）
        if cond.shape[-1] != TILE_SIZE:
            cond = torch.nn.functional.interpolate(
                cond, size=(TILE_SIZE, TILE_SIZE), mode="bilinear",
                align_corners=False)
        cond = cond.to(DEVICE)

        # 用 EMA 权重
        self.ema_model.apply(self.model)
        self.ema_cond.apply(self.cond_encoder)
        try:
            # 编码条件
            cond_feat = self.cond_encoder(cond)   # (1, 4, 32, 32)

            # DDIM 采样潜变量
            z = self.ddim.sample(
                self.model,
                (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                cond_feat, DEVICE, progress=True
            )

            # VAE 解码
            x = self.vae.decode(z)   # (1, 1, 256, 256)
        finally:
            self.ema_model.restore(self.model)
            self.ema_cond.restore(self.cond_encoder)

        x = x.squeeze().cpu().numpy()    # (256, 256) [-1,1]
        dem = norm_to_elev(x)             # 米
        return np.clip(dem, ELEV_MIN, ELEV_MAX).astype(np.float32)

    @torch.no_grad()
    def generate_large(self, cond_12ch: np.ndarray, size_px: int = 2048,
                       seed: int | None = None,
                       overlap: int = 64) -> np.ndarray:
        """大尺寸 DEM 生成：分块生成 + 潜空间重叠混合。

        替代 GAN 的逐瓦片余弦羽化拼贴，用潜空间加权混合实现无缝拼接。

        Parameters
        ----------
        cond_12ch : ndarray (12, H_cond, W_cond)
            12 通道全球条件（H_cond = size_px // block）。
        size_px : int
            输出 DEM 尺寸（默认 2048）。
        seed : int, optional
        overlap : int
            块间重叠像素数（默认 64）。

        Returns
        -------
        ndarray (size_px, size_px) float32
            米制 DEM。
        """
        if seed is not None:
            torch.manual_seed(seed)

        stride = TILE_SIZE - overlap
        n_tiles = (size_px + stride - 1) // stride

        # 输出累积器（加权平均）
        out_acc = np.zeros((size_px, size_px), dtype=np.float32)
        weight = np.zeros((size_px, size_px), dtype=np.float32)

        # 重叠区加权窗（二维余弦窗）
        win = np.ones((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        if overlap > 0:
            ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, overlap)))
            win[:overlap] *= ramp[:, None]
            win[-overlap:] *= ramp[::-1][:, None]
            win[:, :overlap] *= ramp[None, :]
            win[:, -overlap:] *= ramp[None, ::-1]

        for i in range(n_tiles):
            for j in range(n_tiles):
                h0 = min(i * stride, size_px - TILE_SIZE)
                w0 = min(j * stride, size_px - TILE_SIZE)
                # 切对应条件块（cond_12ch 可能是 3D (12,H,W) 或 4D (1,12,H,W)）
                if cond_12ch.ndim == 3:
                    cond_tile = cond_12ch[:, h0:h0 + TILE_SIZE, w0:w0 + TILE_SIZE]
                else:
                    cond_tile = cond_12ch[0, :, h0:h0 + TILE_SIZE, w0:w0 + TILE_SIZE]
                if cond_tile.shape[-1] < TILE_SIZE:
                    # 边缘补齐
                    pad_h = TILE_SIZE - cond_tile.shape[-2]
                    pad_w = TILE_SIZE - cond_tile.shape[-1]
                    cond_tile = np.pad(cond_tile,
                                       ((0, 0), (0, pad_h), (0, pad_w)),
                                       mode="edge")
                # 生成单块
                dem_tile = self.generate(cond_tile, seed=None)
                # 加权累积
                h_end = min(h0 + TILE_SIZE, size_px)
                w_end = min(w0 + TILE_SIZE, size_px)
                th = h_end - h0
                tw = w_end - w0
                out_acc[h0:h_end, w0:w_end] += dem_tile[:th, :tw] * win[:th, :tw]
                weight[h0:h_end, w0:w_end] += win[:th, :tw]

        dem = out_acc / np.maximum(weight, 1e-8)
        return np.clip(dem, ELEV_MIN, ELEV_MAX).astype(np.float32)


# ==================== 便捷函数 ====================

def generate_world_diffusion(seed: int = 42, size_px: int = 2048,
                             tect_res: int = 512,
                             ckpt_path: str | None = None,
                             vae_ckpt: str | None = None) -> np.ndarray:
    """用 LDM 引擎生成完整世界 DEM（与 generate_world 接口兼容）。

    流程：
        1. plate_sim 生成板块骨架
        2. world_to_channels 编码 12 通道条件
        3. LDM 条件生成 DEM
        4. （可选）接现有后处理管线

    Parameters
    ----------
    seed : int
    size_px : int
    tect_res : int
    ckpt_path, vae_ckpt : str, optional

    Returns
    -------
    ndarray (size_px, size_px) float32
        米制 DEM。
    """
    from terrain.core.plate_sim import run_plate_sim
    from terrain.core.generate_world import world_to_channels
    from scipy.ndimage import zoom as ndi_zoom

    # 1. 板块模拟
    print(f"[LDM] 1/3 板块模拟 (seed={seed}, T={tect_res})...")
    elev_T, tect_T, *_ = run_plate_sim(T=tect_res, seed=seed, verbose=True)

    # 2. 上采样 + 12 通道条件
    print(f"[LDM] 2/3 构造 12 通道条件...")
    up = size_px // tect_res
    elev_m = ndi_zoom(elev_T.astype(np.float32), up, order=1)[:size_px, :size_px]
    lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)
    block = 8
    ch_raw, *_ = world_to_channels(elev_m, lats, tect_T, block, seed)

    # 3. LDM 生成
    print(f"[LDM] 3/3 扩散生成 DEM ({size_px}²)...")
    infer = LDMInference(ckpt_path, vae_ckpt)
    dem = infer.generate_large(ch_raw, size_px=size_px, seed=seed)
    print(f"[LDM] 完成: 陆地 {(dem > 0).mean() * 100:.1f}% | "
          f"峰 {dem.max():.0f}m | 沟 {dem.min():.0f}m")
    return dem


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--tect-res", type=int, default=512)
    p.add_argument("--out", type=str, default="terrain/output/world_dem_ldm.npy")
    a = p.parse_args()
    dem = generate_world_diffusion(a.seed, a.size, a.tect_res)
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.save(a.out, dem)
    print(f"保存到 {a.out}")
