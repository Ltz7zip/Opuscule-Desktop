"""阶段 3 物理一致性监督子模块：三个轻量监督网络。

替换现有手写物理后处理：
1. **RiverNet**：DEM → 河网概率图（替换 d8_flow_accumulation + 阈值）
2. **SnowlineNet**：纬度+高程+大陆度 → 雪线高程（替换 snowline_m 查表）
3. **CoastNet**：DEM → 海岸线/陆架边缘锐化掩码（替换 ocean_margin_fix 硬钳制）

设计原则
--------
- **轻量**：每个网络 < 1M 参数，单 GPU 训练 < 1 天
- **可解释先验注入**：网络输入包含物理量（纬度、距海距离），非纯端到端
- **与现有函数对齐**：输出格式与 d8_flow_accumulation / snowline_m / ocean_margin_fix 兼容，
  可作为 drop-in 替换或对比基线
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import ResBlock
from .unet import _pick_groups


# ==================== 1. RiverNet：DEM → 河网概率 ====================

class RiverNet(nn.Module):
    """DEM → 河网概率图（二值分割）。

    输入：(B, 1, H, W) 归一化高程 [-1, 1]
    输出：(B, 1, H, W) 河网概率 [0, 1]

    架构：轻量 U-Net（4 层下采样），带坡度通道注入。
    监督标签：从真实 DEM 用 D8 流量累积 + 分位阈值提取河网二值掩码。

    物理先验注入：除高程外，额外计算坡度图作为第 2 通道输入，
    让网络无需从零学习"水往低处流"。
    """

    def __init__(self, in_ch: int = 2, base: int = 16):  # ch0=高程, ch1=坡度
        """初始化河网分割 U-Net 的编码器/瓶颈/解码器与输出头。"""
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), ResBlock(base, base))
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base, base * 2, 3, padding=1),
            ResBlock(base * 2, base * 2))
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base * 2, base * 4, 3, padding=1),
            ResBlock(base * 4, base * 4))
        # 瓶颈
        self.bot = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base * 4, base * 8, 3, padding=1),
            ResBlock(base * 8, base * 8))
        # 解码器
        self.up3 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8 + base * 4, base * 4, 3, padding=1),
            ResBlock(base * 4, base * 4))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1),
            ResBlock(base * 2, base * 2))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2 + base, base, 3, padding=1),
            ResBlock(base, base))
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def compute_slope_channel(dem: torch.Tensor) -> torch.Tensor:
        """从 DEM 计算坡度通道（中心差分）。"""
        gy = torch.gradient(dem, dim=2)[0]
        gx = torch.gradient(dem, dim=3)[0]
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        # 归一化到 [0, 1]（经验阈值）
        return torch.clamp(grad / 0.5, 0, 1)

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        """dem: (B, 1, H, W) [-1,1] → river_prob: (B, 1, H, W) [0,1]。"""
        slope = self.compute_slope_channel(dem)
        x = torch.cat([dem, slope], dim=1)   # (B, 2, H, W)
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.bot(h3)
        d3 = self.dec3(torch.cat([self.up3(h), h3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), h2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1))
        return torch.sigmoid(self.out(d1))


def river_loss(pred: torch.Tensor, target: torch.Tensor,
               connectivity_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    """河网损失 = BCE + 连通性正则。

    连通性正则：对预测的河网图做形态学膨胀，惩罚"孤点"（膨胀后与原图差异大）。
    鼓励连续河网而非碎片。

    Parameters
    ----------
    pred : (B, 1, H, W) 概率 [0,1]
    target : (B, 1, H, W) 二值标签 {0,1}
    connectivity_weight : float
        连通性正则权重。
    """
    bce = F.binary_cross_entropy(pred, target, reduction="mean")
    # 连通性：用 max_pool(3x3) 近似膨胀，膨胀后概率应 ≥ 原图（连续河网）
    dilated = F.max_pool2d(pred, 3, stride=1, padding=1)
    connectivity = F.mse_loss(dilated, pred)   # 越小越连续
    total = bce + connectivity_weight * connectivity
    return total, {"bce": float(bce.detach()),
                   "conn": float(connectivity.detach()),
                   "loss": float(total.detach())}


# ==================== 2. SnowlineNet：气候 → 雪线高程 ====================

class SnowlineNet(nn.Module):
    """纬度 + 高程 + 大陆度 → 雪线高程（逐像素回归）。

    输入通道：
        ch0 = |lat|/90  归一化纬度
        ch1 = elev_to_norm(elev)  归一化高程
        ch2 = continentality_index  [0,1]
    输出：雪线高程（米），范围 [0, 6500]

    物理先验：网络学习对 snowline_m 查表的平滑修正（考虑大陆度、坡向）。
    标签来源：真实地球雪线数据（如 MODIS 积雪产品）或现有 snowline_m 输出作预训练标签。

    架构：轻量 3 层 CNN（像素级回归，无需大感受野）。
    """

    def __init__(self, in_ch: int = 3, base: int = 16):
        """初始化雪线回归网络：三层轻量 CNN 堆叠。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, 1, 1),
        )

    def forward(self, lat_norm: torch.Tensor, elev_norm: torch.Tensor,
                ci: torch.Tensor) -> torch.Tensor:
        """三个输入均为 (B, 1, H, W) → 雪线高程 (B, 1, H, W) 米。"""
        x = torch.cat([lat_norm, elev_norm, ci], dim=1)
        # sigmoid → [0,1] → 缩放到 [0, 6500] 米
        return torch.sigmoid(self.net(x)) * 6500.0


def snowline_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """雪线 L1 回归损失。"""
    loss = F.l1_loss(pred, target, reduction="mean")
    return loss, {"loss": float(loss.detach())}


# ==================== 3. CoastNet：DEM → 海岸线锐化掩码 ====================

class CoastNet(nn.Module):
    """DEM → 海岸线/陆架边缘锐化掩码（三分类语义分割）。

    输入：(B, 1, H, W) 归一化高程
    输出：(B, 3, H, W) 三类概率：
        class 0 = 陆地（elev > 0）
        class 1 = 陆架/陆坡过渡带（0 ~ -2000m，需锐化）
        class 2 = 深海（< -2000m）

    用途：替代 ocean_margin_fix 的硬钳制，网络学习更自然的边缘过渡。
    监督标签：从真实 DEM 按高程分三类。

    架构：U-Net（与 RiverNet 同结构，输出改为 3 通道）。
    """

    def __init__(self, in_ch: int = 1, base: int = 16):
        """初始化海岸线三分类 U-Net（与 RiverNet 同结构，输出 3 通道）。"""
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), ResBlock(base, base))
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base, base * 2, 3, padding=1),
            ResBlock(base * 2, base * 2))
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base * 2, base * 4, 3, padding=1),
            ResBlock(base * 4, base * 4))
        self.bot = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(base * 4, base * 8, 3, padding=1),
            ResBlock(base * 8, base * 8))
        self.up3 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base * 8 + base * 4, base * 4, 3, padding=1),
            ResBlock(base * 4, base * 4))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base * 4 + base * 2, base * 2, 3, padding=1),
            ResBlock(base * 2, base * 2))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base * 2 + base, base, 3, padding=1),
            ResBlock(base, base))
        self.out = nn.Conv2d(base, 3, 1)   # 3 类

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        """dem: (B, 1, H, W) → logits: (B, 3, H, W)。"""
        h1 = self.enc1(dem)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.bot(h3)
        d3 = self.dec3(torch.cat([self.up3(h), h3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), h2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1))
        return self.out(d1)


def coast_loss(logits: torch.Tensor, target: torch.Tensor,
               class_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """海岸线交叉熵损失（带类别权重）。

    Parameters
    ----------
    logits : (B, 3, H, W)
    target : (B, H, W) long，值 ∈ {0, 1, 2}
    class_weights : (3,) optional
        类别权重；陆架过渡带（class 1）通常需加权以对抗类别不平衡。
    """
    if class_weights is None:
        # 默认权重：陆架带（class 1）加权 3×，因像素少但重要
        class_weights = torch.tensor([1.0, 3.0, 1.0], device=logits.device)
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction="mean")
    return ce, {"loss": float(ce.detach())}


def coast_apply(dem: np.ndarray, coast_logits: np.ndarray,
                shelf_target_depth: float = -2000.0) -> np.ndarray:
    """用 CoastNet 输出对 DEM 做海岸线锐化后处理。

    在陆架过渡带（class 1 概率高）区域，把高程推向陆架目标深度，
    实现比 ocean_margin_fix 硬钳制更平滑的边缘。

    Parameters
    ----------
    dem : (H, W) float32 米制高程
    coast_logits : (3, H, W) 网络输出 logits
    shelf_target_depth : float
        陆架目标深度（米），默认 -2000。

    Returns
    -------
    (H, W) float32 锐化后高程
    """
    probs = torch.softmax(torch.from_numpy(coast_logits), dim=0).numpy()
    shelf_prob = probs[1]   # 陆架过渡带概率
    # 在陆架概率高的区域，把高程推向目标深度（加权混合）
    target = np.full_like(dem, shelf_target_depth, dtype=np.float32)
    alpha = shelf_prob[:, :, None] if dem.ndim == 3 else shelf_prob
    return (1 - alpha) * dem + alpha * target
