"""潜空间条件 U-Net：在 VAE 潜空间（32²×4）运行的条件 DDPM 去噪网络。

与阶段 1 的像素空间 U-Net 区别：
- 输入尺寸 32²×4（潜空间），而非 256²×1（像素空间）→ 计算量大幅降低
- 条件通过 **concat** 注入（条件特征图与噪声潜变量在通道维拼接）
- 加入 **cross-attention**（条件特征作 K/V），增强条件遵循度
- 时间嵌入与阶段 1 一致

前向：``forward(z_noisy, t, cond) -> eps_pred``
    z_noisy : (B, 4, 32, 32)  加噪潜变量
    t       : (B,)             时间步
    cond    : (B, cond_ch, 32, 32)  条件特征图（来自 ConditionEncoder）
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import TimeEmbed, _pick_groups


class LatentBlock(nn.Module):
    """潜空间卷积块：GroupNorm + SiLU + Conv + 时间调制 + Conv + 残差。"""

    def __init__(self, c_in: int, c_out: int, time_dim: int, groups: int = 8):
        """初始化潜空间卷积块，构建两层卷积、时间调制 MLP 与残差跳跃连接。"""
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


class CrossAttention(nn.Module):
    """标准 cross-attention：query 来自 x，key/value 来自 cond。

    用于让去噪网络在瓶颈层"关注"条件特征的关键位置（如俯冲带、海岸线）。
    """

    def __init__(self, channels: int, cond_channels: int, heads: int = 4,
                 head_dim: int = 32):
        """初始化 cross-attention 层，构建 Q/K/V 投影与输出投影。"""
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.norm = nn.GroupNorm(_pick_groups(channels, 8), channels)
        self.to_q = nn.Conv2d(channels, inner, 1)
        self.to_k = nn.Conv2d(cond_channels, inner, 1)
        self.to_v = nn.Conv2d(cond_channels, inner, 1)
        self.to_out = nn.Conv2d(inner, channels, 1)

    def forward(self, x, cond):
        """x: (B, C, H, W), cond: (B, Cc, H, W)（同分辨率）。"""
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.to_q(h).reshape(B, self.heads, -1, H * W).permute(0, 1, 3, 2)
        k = self.to_k(cond).reshape(B, self.heads, -1, H * W).permute(0, 1, 3, 2)
        v = self.to_v(cond).reshape(B, self.heads, -1, H * W).permute(0, 1, 3, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, -1, H, W)
        return x + self.to_out(out)


class LatentDiffusionUNet(nn.Module):
    """潜空间条件去噪 U-Net：32² → 16² → 8² 瓶颈 → 32²。

    Parameters
    ----------
    in_ch : int
        潜变量通道（VAE latent_ch = 4）。
    cond_ch : int
        条件特征通道（ConditionEncoder 输出，默认 4）。
    base : int
        基础通道，默认 128（潜空间小，可用更大通道）。
    time_dim : int
        时间嵌入维度，默认 128。
    use_cross_attn : bool
        是否在瓶颈层用 cross-attention；默认 True。
    """

    def __init__(self, in_ch: int = 4, cond_ch: int = 4, base: int = 128,
                 time_dim: int = 128, use_cross_attn: bool = True):
        """初始化潜空间条件去噪 U-Net：构建编码器/瓶颈/解码器及可选 cross-attention。"""
        super().__init__()
        self.time = TimeEmbed(time_dim)
        # 输入 = 噪声潜变量 + 条件特征 concat
        in_with_cond = in_ch + cond_ch
        # down: 32 -> 16 -> 8
        self.d1 = LatentBlock(in_with_cond, base, time_dim)        # 32
        self.d2 = LatentBlock(base, base * 2, time_dim)            # 16
        self.d3 = LatentBlock(base * 2, base * 4, time_dim)        # 8
        self.pool = nn.MaxPool2d(2)
        # mid
        self.mid = LatentBlock(base * 4, base * 4, time_dim)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            # 瓶颈层条件也需下采样到 8²
            self.cond_down = nn.Sequential(
                nn.Conv2d(cond_ch, cond_ch, 2, stride=2),  # 16
                nn.SiLU(),
                nn.Conv2d(cond_ch, cond_ch, 2, stride=2),  # 8
            )
            self.cross = CrossAttention(base * 4, cond_ch)
        # up
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)   # 16
        self.u2 = LatentBlock(base * 2 + base * 2, base * 2, time_dim)   # cat(up2, h2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)       # 32
        self.u1 = LatentBlock(base + base, base, time_dim)              # cat(up1, h1)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, z, t, cond):
        """
        z    : (B, in_ch, 32, 32)  加噪潜变量
        t    : (B,)                时间步
        cond : (B, cond_ch, 32, 32) 条件特征图
        """
        te = self.time(t)
        x = torch.cat([z, cond], dim=1)          # concat 条件
        h1 = self.d1(x, te)                       # 32, base
        h2 = self.d2(self.pool(h1), te)           # 16, base*2
        h3 = self.d3(self.pool(h2), te)           # 8,  base*4
        h = self.mid(h3, te)                      # 8,  base*4
        if self.use_cross_attn:
            cond_8 = self.cond_down(cond)         # 8²
            h = self.cross(h, cond_8)
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), te)   # 16, base*2
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), te)   # 32, base
        return self.out(h)                        # 预测噪声 ε
