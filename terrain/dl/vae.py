"""地形专用 VAE：256²×1 <-> 32²×4 潜空间。

设计要点
--------
- 单通道输入（高程），不直接复用预训练图像 VAE
- 下采样 8×（3 次 stride=2 卷积），潜空间 32²×4
- KL 权重 β=1e-4，优先重建精度（地形数值准确性比编码紧凑更重要）
- 输出 tanh 限制到 [-1, 1]，与高程归一化域一致
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_groups(channels: int, max_groups: int = 8) -> int:
    """选择能整除 channels 的最大分组数（避免 GroupNorm 整除错误）。"""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class ResBlock(nn.Module):
    """GroupNorm + SiLU + Conv ×2 + 残差。"""

    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        """初始化残差块，构建两层 GroupNorm+SiLU+Conv 与跳跃连接。"""
        super().__init__()
        g1 = _pick_groups(c_in, groups)
        g2 = _pick_groups(c_out, groups)
        self.norm1 = nn.GroupNorm(g1, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        """前向：x 经两层卷积后加残差返回。"""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class TerrainVAE(nn.Module):
    """256²×1 <-> 32²×4 潜空间 VAE。

    Parameters
    ----------
    in_ch : int
        输入通道（高程=1）。
    latent_ch : int
        潜空间通道，默认 4。
    base : int
        基础通道，默认 64。
    """

    def __init__(self, in_ch: int = 1, latent_ch: int = 4, base: int = 64):
        """初始化 VAE：构建编码器、潜变量头与解码器。"""
        super().__init__()
        # Encoder: 256 -> 128 -> 64 -> 32
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlock(base, base),                       # 256
            nn.Conv2d(base, base, 2, stride=2),         # 128
            ResBlock(base, base * 2),
            nn.Conv2d(base * 2, base * 2, 2, stride=2), # 64
            ResBlock(base * 2, base * 2),
            nn.Conv2d(base * 2, base * 2, 2, stride=2), # 32
            ResBlock(base * 2, base * 2),
        )
        self.conv_mu = nn.Conv2d(base * 2, latent_ch, 1)
        self.conv_logvar = nn.Conv2d(base * 2, latent_ch, 1)

        # Decoder
        self.dec_in = nn.Conv2d(latent_ch, base * 2, 1)
        self.dec = nn.Sequential(
            ResBlock(base * 2, base * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),  # 64
            ResBlock(base * 2, base * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base * 2, base, 3, padding=1),      # 128
            ResBlock(base, base),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base, base, 3, padding=1),          # 256
            ResBlock(base, base),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def encode(self, x):
        """编码：返回 (均值, log方差) 潜变量图。"""
        h = self.enc(x)
        return self.conv_mu(h), self.conv_logvar(h)

    def reparam(self, mu, logvar):
        """重参数化采样：mu + std·ε。"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """解码：潜变量 z 重建为像素空间。"""
        return self.dec(self.dec_in(z))

    def forward(self, x):
        """前向：编码→重参数化→解码，返回 (重建, mu, logvar)。"""
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1e-4):
    """VAE 损失 = L1 重建 + β·KL。

    Returns
    -------
    (total, info_dict)
    """
    rec = F.l1_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]))
    return rec + beta * kld, {"rec": float(rec.detach()), "kld": float(kld.detach())}
