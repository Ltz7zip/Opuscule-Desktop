"""条件编码器：把 12 通道粗底图（256²×12）编码为潜空间条件特征（32²×cond_ch）。

与现有 GAN 的 12 通道定义严格一致（见 generate_world.py:world_to_channels）：
    [0] 块均值 bm        [1] 块标准差 bstd
    [2] 块最小值 bmin      [3] 块最大值 bmax
    [4] 海陆比例 lf       [5] 河网
    [6] 冰川覆盖率         [7] 绝对纬度/90°
    [8] 俯冲/碰撞构造场    [9] 离散构造场
    [10] 转换断层构造场   [11] 余裕构造场

设计
----
- 输入：(B, 12, 256, 256)，与 GAN 完全相同的条件张量
- 下采样 8× 到 (B, cond_ch, 32, 32)，与 VAE 潜空间同尺寸
- 输出条件特征图供 LDM 通过 concat 或 cross-attention 注入
- 归一化：沿用 GAN 的 ch_stats（mean/std），可在构造时传入
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TILE_SIZE, LATENT_SIZE
from .vae import ResBlock  # 复用阶段 1 的 ResBlock


# 默认 12 通道统计（与 GAN checkpoint 的 ch_stats 一致；实际从 ckpt 加载）
DEFAULT_CH_STATS = {
    "mean": np.zeros(12, dtype=np.float32),
    "std": np.ones(12, dtype=np.float32),
}


class ConditionEncoder(nn.Module):
    """12 通道粗底图 → 32²×cond_ch 条件特征图。

    Parameters
    ----------
    in_ch : int
        输入条件通道数（与 GAN 一致 = 12）。
    cond_ch : int
        输出条件通道数，默认 4（与 VAE 潜空间通道一致，便于 concat）。
    base : int
        基础通道，默认 64。
    ch_stats : dict, optional
        含 "mean"/"std" 键的 ndarray(12,)，用于输入归一化；None 不归一化。
    """

    def __init__(self, in_ch: int = 12, cond_ch: int = 4, base: int = 64,
                 ch_stats: dict | None = None):
        """初始化条件编码器：注册输入归一化统计并构建下采样编码主干。"""
        super().__init__()
        self.in_ch = in_ch
        self.cond_ch = cond_ch

        # 输入归一化（与 GAN 一致：(x - mean) / std）
        if ch_stats is not None:
            self.register_buffer("ch_mean",
                                 torch.as_tensor(ch_stats["mean"], dtype=torch.float32))
            self.register_buffer("ch_std",
                                 torch.as_tensor(ch_stats["std"], dtype=torch.float32))
        else:
            self.register_buffer("ch_mean", torch.zeros(in_ch))
            self.register_buffer("ch_std", torch.ones(in_ch))

        # 256 -> 128 -> 64 -> 32
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            ResBlock(base, base),                       # 256
            nn.Conv2d(base, base, 2, stride=2),         # 128
            ResBlock(base, base * 2),
            nn.Conv2d(base * 2, base * 2, 2, stride=2), # 64
            ResBlock(base * 2, base * 2),
            nn.Conv2d(base * 2, base * 2, 2, stride=2), # 32
            ResBlock(base * 2, base * 2),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, cond_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 12, 256, 256) -> (B, cond_ch, 32, 32)。"""
        # 归一化（按通道）
        x = (x - self.ch_mean[None, :, None, None]) / self.ch_std[None, :, None, None]
        return self.enc(x)


def encode_world_to_condition(elev_m: np.ndarray, lats: np.ndarray,
                              tect_T: np.ndarray, block: int = 8,
                              seed: int = 0,
                              ch_stats: dict | None = None) -> torch.Tensor:
    """从米制 DEM + 构造场构造 12 通道条件张量。

    这是 generate_world.py:world_to_channels 的薄封装，输出 torch 张量
    供 ConditionEncoder 直接消费。

    Parameters
    ----------
    elev_m : ndarray (H, H), float32
        米制高程骨架（像素级细化后的结果）。
    lats : ndarray (H,)
        像素中心纬度。
    tect_T : ndarray (4, T, T)
        板块模拟 4 构造场。
    block : int
        块大小，默认 8（输出 256² 条件 → 32² 潜空间）。
    seed : int
        河网计算种子。
    ch_stats : dict, optional
        通道统计，传给 ConditionEncoder。

    Returns
    -------
    torch.Tensor (12, H//block, H//block)
        未归一化的 12 通道条件（归一化在 ConditionEncoder 内做）。
    """
    # 复用现有 world_to_channels（避免重复实现）
    from terrain.core.generate_world import world_to_channels  # type: ignore

    ch_raw, *_ = world_to_channels(elev_m, lats, tect_T, block, seed)
    return torch.from_numpy(ch_raw).float()   # (12, B, B)
