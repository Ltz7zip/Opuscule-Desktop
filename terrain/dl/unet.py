"""DDPM 去噪 U-Net：输入 (B,1,256,256) + 时间步，预测噪声 ε。

结构风格与 ``terrain/core/terrain_engine.py:UNet`` 一致（GroupNorm + SiLU + 残差），
但加入正弦时间嵌入。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbed(nn.Module):
    """正弦时间嵌入 + 2 层 MLP。"""

    def __init__(self, dim: int = 128):
        """初始化时间嵌入：保存维度并构建 2 层 MLP。"""
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """前向：对时间步 t 生成正弦嵌入并经 MLP 映射。"""
        half = self.dim // 2
        freq = torch.exp(
            -torch.arange(half, device=t.device).float() / half * math.pi)
        args = t[:, None].float() * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


def _pick_groups(channels: int, max_groups: int = 8) -> int:
    """选择能整除 channels 的最大分组数（避免 GroupNorm 整除错误）。"""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


class Block(nn.Module):
    """Conv 块：GroupNorm + SiLU + Conv + 时间调制 + Conv + 残差。"""

    def __init__(self, c_in: int, c_out: int, time_dim: int, groups: int = 8):
        """初始化卷积块，构建两层 GroupNorm+SiLU+Conv、时间调制 MLP 与残差跳跃连接。"""
        super().__init__()
        g1 = _pick_groups(c_in, groups)
        g2 = _pick_groups(c_out, groups)
        self.norm1 = nn.GroupNorm(g1, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb):
        """前向：x 经归一化卷积与时间调制后加残差返回。"""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """DDPM 去噪 U-Net：256² → 32² 瓶颈 → 256²，4 次下/上采样。

    Parameters
    ----------
    in_ch : int
        输入通道（高程=1）。
    base : int
        基础通道，默认 64。
    time_dim : int
        时间嵌入维度，默认 128。
    """

    def __init__(self, in_ch: int = 1, base: int = 64, time_dim: int = 128):
        """初始化 DDPM 去噪 U-Net，构建编码器/瓶颈/解码器与输出头。"""
        super().__init__()
        self.time = TimeEmbed(time_dim)
        # down: 256 -> 128 -> 64 -> 32（3 次 pool，d4 为瓶颈）
        self.d1 = Block(in_ch, base, time_dim)           # 256, base
        self.d2 = Block(base, base * 2, time_dim)        # 128, base*2
        self.d3 = Block(base * 2, base * 4, time_dim)    # 64,  base*4
        self.d4 = Block(base * 4, base * 8, time_dim)    # 32,  base*8
        self.pool = nn.MaxPool2d(2)
        # mid（不再 pool，在 32 处）
        self.mid = Block(base * 8, base * 8, time_dim)
        # up：3 次上采样回 256；c_in = up 输出 + 对应跳跃
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)  # 32->64
        self.u3 = Block(base * 4 + base * 4, base * 4, time_dim)        # cat(up3, h3)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 64->128
        self.u2 = Block(base * 2 + base * 2, base * 2, time_dim)        # cat(up2, h2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 128->256
        self.u1 = Block(base + base, base, time_dim)                   # cat(up1, h1)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x, t):
        """前向：x 与时间嵌入经 U-Net 编解码，返回预测噪声。"""
        te = self.time(t)
        h1 = self.d1(x, te)                      # 256, base
        h2 = self.d2(self.pool(h1), te)          # 128, base*2
        h3 = self.d3(self.pool(h2), te)          # 64,  base*4
        h4 = self.d4(self.pool(h3), te)          # 32,  base*8
        h = self.mid(h4, te)                     # 32,  base*8（不额外 pool）
        h = self.u3(torch.cat([self.up3(h), h3], dim=1), te)   # 64, base*4
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), te)   # 128, base*2
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), te)   # 256, base
        return self.out(h)
