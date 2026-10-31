"""DDIM 采样器：确定性加速采样，将 1000 步 DDPM 压缩到 50 步。

DDIM (Denoising Diffusion Implicit Models) 与 DDPM 用同一训练目标（预测 ε），
但采样时跳过中间步，50 步即可达到 1000 步 DDPM 的质量。

关键性质
--------
- **确定性**：``eta=0`` 时同一噪声 → 同一输出，便于可复现生成
- **加速**：50 步代替 1000 步，推理快 20×
- **兼容**：直接复用 DDPM 训练的 ᾱ 调度，无需重训

公式（eta=0 硯定性版本）：
    x_{t-1} = sqrt(ᾱ_{t-1}) · x0_pred + sqrt(1 - ᾱ_{t-1}) · ε_pred
其中 x0_pred = (x_t - sqrt(1-ᾱ_t)·ε_pred) / sqrt(ᾱ_t)
"""
from __future__ import annotations
import numpy as np
import torch


class DDIM:
    """DDIM 采样器。

    Parameters
    ----------
    n_train_steps : int
        训练时的扩散步数（与 DDPM 一致，默认 1000）。
    n_sample_steps : int
        采样步数（默认 50）。
    beta_start, beta_end : float
        线性 β 调度（与 DDPM 一致）。
    eta : float
        随机性参数，0 = 确定性，1 = DDPM 等价。默认 0。
    device : torch.device | str
    """

    def __init__(self, n_train_steps: int = 1000, n_sample_steps: int = 50,
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 eta: float = 0.0, device: torch.device | str = "cpu"):
        """初始化 DDIM 采样器：预计算 ᾱ 调度并生成降序采样子序列。"""
        self.n_train = n_train_steps
        self.n_sample = n_sample_steps
        self.eta = eta

        betas = torch.linspace(beta_start, beta_end, n_train_steps, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)          # (n_train,)

        # 采样子序列：从 [0, n_train) 中均匀取 n_sample 个步
        step_ratio = n_train_steps // n_sample_steps
        self.timesteps = list(range(0, n_train_steps, step_ratio))[::-1]  # 降序

    @torch.no_grad()
    def sample(self, model, shape, cond, device,
               progress: bool = False) -> torch.Tensor:
        """条件采样。

        Parameters
        ----------
        model : nn.Module
            去噪网络，签名 ``model(z, t, cond) -> eps``。
        shape : tuple
            输出潜变量形状 (B, C, H, W)。
        cond : torch.Tensor
            条件特征图 (B, cond_ch, H, W)，与 z 同空间分辨率。
        device : torch.device
        progress : bool
            是否显示进度条。

        Returns
        -------
        torch.Tensor (B, C, H, W)
            去噪后的潜变量（仍需 VAE 解码回像素空间）。
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        z = torch.randn(shape, device=device)
        iters = self.timesteps
        if progress and tqdm is not None:
            iters = tqdm(iters, desc="ddim sampling")

        for i, t in enumerate(iters):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model(z, t_batch, cond)

            ab_t = self.alpha_bar[t]
            # 预测 x0
            x0_pred = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()

            # 下一个时间步（i+1 对应更小的 t）
            if i + 1 < len(iters):
                ab_prev = self.alpha_bar[self.timesteps[i + 1]]
            else:
                ab_prev = torch.tensor(1.0, device=device)  # t=0

            # 确定性方向（eta=0）
            sigma = self.eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            z = ab_prev.sqrt() * x0_pred + dir_xt
            if self.eta > 0:
                z = z + sigma * torch.randn_like(z)

        return z
