"""阶段 5A：DEM 超分辨率网络（ESRGAN/Real-ESRGAN 架构，优化版）。

将低分辨率 DEM 提升到高分辨率（2×/4×/8×），补充细节高频纹理。

优化项（相比初版）
------------------
1. **像素注意力（CCA）**：通道+空间注意力，增强纹理特征选择
2. **灵活缩放**：支持 2×/4×/8×（可配置 PixelShuffle 层数）
3. **多尺度输出**：中间上采样层输出辅助监督
4. **SSIM 损失**：结构相似度，保持地形结构一致性
5. **Charbonnier 损失**：平滑 L1 变体，对异常值更鲁棒
6. **梯度损失**：空间梯度 L1，强化边缘锐度
7. **TTA 推理**：8 方向翻转/旋转增强 + 平均
8. **FP16 推理**：混合精度加速（CUDA/MPS）
9. **批量分块**：多块并行推理，提升 GPU 利用率
10. **SRDataset**：训练数据集（切片 + 增强）

用法
----
    >>> from terrain.dl.sr_net import SRNet, sr_upscale
    >>> dem_lr = np.load("dem_1024.npy")   # (1024, 1024) float32
    >>> dem_hr = sr_upscale(dem_lr, scale=4, tta=True)  # → (4096, 4096)
    >>> # 带注意力 + 8× 超分
    >>> net = SRNet(scale=8, use_attention=True, base=32)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import DEVICE, ELEV_MIN, ELEV_MAX, elev_to_norm, norm_to_elev


# ==================== 1. 注意力模块 ====================

class ChannelAttention(nn.Module):
    """通道注意力（SE 变体）：压缩-激励式特征重标定。

    全局平均池化 → FC 压缩 → FC 扩展 → Sigmoid → 通道加权。
    """

    def __init__(self, channels: int, reduction: int = 8):
        """初始化通道注意力。

        Parameters
        ----------
        channels : int  输入通道数
        reduction : int  压缩比
        """
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：x * channel_weights。"""
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class SpatialAttention(nn.Module):
    """空间注意力：聚焦纹理丰富的区域。

    通道维平均+最大 → 7×7 卷积 → Sigmoid → 空间加权。
    """

    def __init__(self, kernel_size: int = 7):
        """初始化空间注意力。"""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：x * spatial_weights。"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        w = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * w


class CCABlock(nn.Module):
    """CCA（Channel-Spatial Attention）：通道 + 空间双重注意力。

    先通道注意力重标定，再空间注意力聚焦，
    增强纹理特征选择，抑制平坦区域噪声。
    """

    def __init__(self, channels: int, reduction: int = 8):
        """初始化 CCA。"""
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：空间 → 通道 → 输出。"""
        x = self.ca(x)
        x = self.sa(x)
        return x


# ==================== 2. 核心模块 ====================

class DenseBlock(nn.Module):
    """密集连接块：5 层卷积，每层输入拼接所有前层输出。

    特征复用 + 梯度流通畅，适合提取细粒度纹理。
    """

    def __init__(self, in_ch: int = 64, growth: int = 32):
        """初始化密集块。

        Parameters
        ----------
        in_ch : int  输入通道数
        growth : int  每层增长通道数（输出 = in_ch + 5*growth）
        """
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(5):
            self.layers.append(nn.Sequential(
                nn.Conv2d(ch, growth, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch += growth
        # 1×1 卷积压缩回 in_ch
        self.compress = nn.Conv2d(ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：逐层拼接前层输出提取特征，最后 1×1 压缩回 in_ch。"""
        feats = [x]
        for layer in self.layers:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        return self.compress(torch.cat(feats, dim=1))


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block：Real-ESRGAN 核心模块。

    3 个 DenseBlock 嵌套 + 残差缩放（β=0.2），
    深层纹理特征提取，残差连接防止梯度消失。
    可选 CCA 注意力增强。
    """

    def __init__(self, in_ch: int = 64, growth: int = 32,
                 beta: float = 0.2, use_attention: bool = False):
        """初始化 RRDB。

        Parameters
        ----------
        in_ch : int  输入/输出通道
        growth : int  DenseBlock 每层增长通道
        beta : float  残差缩放因子
        use_attention : bool  是否在残差后加 CCA 注意力
        """
        super().__init__()
        self.beta = beta
        self.db1 = DenseBlock(in_ch, growth)
        self.db2 = DenseBlock(in_ch, growth)
        self.db3 = DenseBlock(in_ch, growth)
        self.attention = CCABlock(in_ch) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：3 个 DenseBlock 残差嵌套，可选加 CCA 注意力。"""
        out = x
        out = out + self.beta * self.db1(out)
        out = out + self.beta * self.db2(out)
        out = out + self.beta * self.db3(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


# ==================== 3. 主网络 ====================

class SRNet(nn.Module):
    """DEM 超分辨率网络（Real-ESRGAN + CCA 注意力，优化版）。

    输入：低分辨率 DEM（单通道，归一化到 [-1, 1]）
    输出：高分辨率 DEM（单通道，归一化到 [-1, 1]）

    支持 2×/4×/8× 超分，可选 CCA 注意力与多尺度输出。
    """

    def __init__(self, in_ch: int = 1, base: int = 64,
                 n_rrdb: int = 5, growth: int = 32,
                 scale: int = 4,
                 use_attention: bool = False,
                 multi_scale: bool = False):
        """初始化超分网络。

        Parameters
        ----------
        in_ch : int  输入通道（DEM 单通道 = 1）
        base : int  基础特征通道（默认 64）
        n_rrdb : int  RRDB 块数量（默认 5，更深更慢）
        growth : int  DenseBlock 增长率
        scale : int  超分倍数（2/4/8）
        use_attention : bool  是否启用 CCA 注意力
        multi_scale : bool  是否输出多尺度中间结果（训练辅助监督）
        """
        super().__init__()
        self.scale = scale
        self.multi_scale = multi_scale

        # 计算上采样层数（每层 2×）
        self.n_upsample = {2: 1, 4: 2, 8: 3}.get(scale, 2)
        if scale not in (2, 4, 8):
            raise ValueError(f"不支持的 scale={scale}，仅支持 2/4/8")

        # 浅层特征提取
        self.conv_head = nn.Conv2d(in_ch, base, 3, padding=1)

        # 深层特征提取：n 个 RRDB 块
        self.rrdbs = nn.Sequential(
            *[RRDB(base, growth, use_attention=use_attention)
              for _ in range(n_rrdb)]
        )
        # 残差连接
        self.conv_body = nn.Conv2d(base, base, 3, padding=1)

        # 上采样层：每层 2× PixelShuffle
        self.upsamples = nn.ModuleList()
        for _ in range(self.n_upsample):
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(base, base * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # 多尺度中间输出头
        if multi_scale:
            self.mid_heads = nn.ModuleList([
                nn.Conv2d(base, in_ch, 3, padding=1)
                for _ in range(self.n_upsample - 1)
            ])
        else:
            self.mid_heads = None

        # 重建
        self.conv_tail = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        """前向传播：LR DEM → HR DEM。

        Parameters
        ----------
        x : Tensor (B, 1, H, W)  低分辨率 DEM（归一化 [-1, 1]）

        Returns
        -------
        Tensor (B, 1, H*scale, W*scale)
            或 (out, [mid1, mid2, ...]) 多尺度模式
        """
        # 浅层
        feat = self.conv_head(x)
        # 深层 + 残差
        deep = self.rrdbs(feat)
        feat = feat + self.conv_body(deep)

        # 上采样（含多尺度中间输出）
        mid_outputs = []
        for i, up in enumerate(self.upsamples):
            feat = up(feat)
            if self.multi_scale and i < self.n_upsample - 1:
                mid_outputs.append(self.mid_heads[i](feat))

        # 重建
        out = self.conv_tail(feat)

        if self.multi_scale and mid_outputs:
            return out, mid_outputs
        return out

    @property
    def param_count(self) -> int:
        """总参数量。"""
        return sum(p.numel() for p in self.parameters())


# ==================== 4. 判别器 ====================

class SRDiscriminator(nn.Module):
    """超分判别器：区分真实 HR DEM 与超分结果。

    VGG-style 判别器，输出真/假概率。
    可选组件，仅对抗训练时使用。
    """

    def __init__(self, in_ch: int = 1, base: int = 64):
        """初始化判别器。"""
        super().__init__()
        def block(c_in, c_out, stride=1, norm=True):
            """构造单层判别器卷积块：Conv + 可选 BN + LeakyReLU。"""
            layers = [nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(in_ch, base, norm=False),  # 256
            block(base, base * 2, 2),        # 128
            block(base * 2, base * 4, 2),    # 64
            block(base * 4, base * 8, 2),    # 32
            block(base * 8, base * 8, 2),    # 16
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：返回 logits（正 = 真）。"""
        return self.net(x).squeeze(1)


# ==================== 5. 损失函数（优化版） ====================

def sr_pixel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """像素 L1 损失：保证低频结构一致。"""
    return F.l1_loss(pred, target)


def sr_charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier 损失：平滑 L1 变体，对异常值更鲁棒。

    L = sqrt((pred - target)² + eps²)

    比 L1 在零点附近更平滑，训练更稳定。
    """
    diff = pred - target
    return torch.mean(torch.sqrt(diff ** 2 + eps ** 2))


def sr_freq_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """频率域 L1 损失：强化高频纹理。

    对 pred 和 target 做 2D FFT，在频域计算 L1，
    促使网络学习高频细节。
    """
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    return F.l1_loss(pred_mag, target_mag)


def _gaussian_window(window_size: int, sigma: float = 1.5
                     ) -> torch.Tensor:
    """生成 2D 高斯窗口（SSIM 用）。"""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d


def sr_ssim_loss(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11) -> torch.Tensor:
    """SSIM 损失：结构相似度，保持地形结构一致性。

    SSIM ∈ [-1, 1]，越接近 1 越相似。
    损失 = 1 - SSIM。

    Parameters
    ----------
    pred : Tensor (B, 1, H, W)
    target : Tensor (B, 1, H, W)
    window_size : int  高斯窗口大小
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 高斯窗口
    window = _gaussian_window(window_size).to(pred.device)
    window = window.expand(pred.shape[1], 1, window_size, window_size)

    # 均值
    mu_p = F.conv2d(pred, window, padding=window_size // 2,
                    groups=pred.shape[1])
    mu_t = F.conv2d(target, window, padding=window_size // 2,
                    groups=target.shape[1])
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t

    # 方差 + 协方差
    sigma_p2 = F.conv2d(pred * pred, window, padding=window_size // 2,
                        groups=pred.shape[1]) - mu_p2
    sigma_t2 = F.conv2d(target * target, window,
                        padding=window_size // 2,
                        groups=target.shape[1]) - mu_t2
    sigma_pt = F.conv2d(pred * target, window,
                        padding=window_size // 2,
                        groups=pred.shape[1]) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    return 1.0 - ssim_map.mean()


def sr_gradient_loss(pred: torch.Tensor, target: torch.Tensor
                     ) -> torch.Tensor:
    """梯度损失：空间梯度 L1，强化边缘锐度。

    计算 x/y 方向 Sobel 梯度，L1 差异。
    防止超分结果过度平滑。
    """
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32,
                           device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32,
                           device=pred.device).view(1, 1, 3, 3)

    def grad(img):
        """计算图像 x/y 方向 Sobel 梯度。"""
        gx = F.conv2d(img, sobel_x, padding=1)
        gy = F.conv2d(img, sobel_y, padding=1)
        return gx, gy

    gx_p, gy_p = grad(pred)
    gx_t, gy_t = grad(target)
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def sr_loss(pred: torch.Tensor, target: torch.Tensor,
            lambda_freq: float = 0.1,
            lambda_ssim: float = 0.1,
            lambda_grad: float = 0.1,
            use_charbonnier: bool = True,
            disc_logits: torch.Tensor | None = None,
            lambda_adv: float = 0.005,
            ) -> tuple[torch.Tensor, dict]:
    """超分总损失（优化版）：像素 + SSIM + 频率 + 梯度 + 对抗。

    Parameters
    ----------
    pred : Tensor (B, 1, 4H, 4W)  超分结果
    target : Tensor (B, 1, 4H, 4W)  真实 HR
    lambda_freq : float  频率损失权重
    lambda_ssim : float  SSIM 损失权重
    lambda_grad : float  梯度损失权重
    use_charbonnier : bool  用 Charbonnier 替代 L1
    disc_logits : Tensor, optional  判别器对 pred 的输出
    lambda_adv : float  对抗损失权重

    Returns
    -------
    total_loss : Tensor
    parts : dict  各损失分量
    """
    # 像素损失
    if use_charbonnier:
        l_pix = sr_charbonnier_loss(pred, target)
    else:
        l_pix = sr_pixel_loss(pred, target)

    # 频率损失
    l_freq = sr_freq_loss(pred, target)

    # SSIM 损失
    l_ssim = sr_ssim_loss(pred, target)

    # 梯度损失
    l_grad = sr_gradient_loss(pred, target)

    total = (l_pix
             + lambda_freq * l_freq
             + lambda_ssim * l_ssim
             + lambda_grad * l_grad)
    parts = {
        "pixel": l_pix.item(),
        "freq": l_freq.item(),
        "ssim": l_ssim.item(),
        "grad": l_grad.item(),
    }

    if disc_logits is not None:
        l_adv = F.binary_cross_entropy_with_logits(
            disc_logits, torch.ones_like(disc_logits))
        total = total + lambda_adv * l_adv
        parts["adv"] = l_adv.item()

    parts["total"] = total.item()
    return total, parts


# ==================== 6. TTA 推理 ====================

def _apply_tta(model: SRNet, x: torch.Tensor) -> torch.Tensor:
    """8 方向 TTA（Test-Time Augmentation）。

    对输入做 8 种翻转/旋转变换，分别推理后取平均。
    消除方向偏置，提升纹理一致性。

    8 种变换：原图、水平翻转、垂直翻转、180°旋转、
            90°/270°旋转、转置、反对角转置。

    Parameters
    ----------
    model : SRNet  超分模型
    x : Tensor (B, 1, H, W)  输入

    Returns
    -------
    Tensor (B, 1, H*scale, W*scale)  TTA 平均结果
    """
    transforms = [
        lambda t: t,                          # 原图
        lambda t: torch.flip(t, [-1]),        # 水平翻转
        lambda t: torch.flip(t, [-2]),        # 垂直翻转
        lambda t: torch.flip(t, [-1, -2]),    # 180°
        lambda t: torch.rot90(t, 1, [-2, -1]),  # 90°
        lambda t: torch.rot90(t, 3, [-2, -1]),  # 270°
        lambda t: t.transpose(-2, -1),        # 转置
        lambda t: torch.flip(t.transpose(-2, -1), [-1]),  # 反对角
    ]
    # 对应逆变换
    inv_transforms = [
        lambda t: t,
        lambda t: torch.flip(t, [-1]),
        lambda t: torch.flip(t, [-2]),
        lambda t: torch.flip(t, [-1, -2]),
        lambda t: torch.rot90(t, 3, [-2, -1]),   # 逆 90°
        lambda t: torch.rot90(t, 1, [-2, -1]),   # 逆 270°
        lambda t: t.transpose(-2, -1),
        lambda t: torch.flip(t, [-1]).transpose(-2, -1),
    ]

    outputs = []
    for fwd, inv in zip(transforms, inv_transforms):
        x_aug = fwd(x)
        with torch.no_grad():
            y_aug = model(x_aug)
        y = inv(y_aug)
        outputs.append(y)

    return torch.stack(outputs).mean(dim=0)


# ==================== 7. 推理接口（优化版） ====================

def sr_upscale(dem_lr: np.ndarray, scale: int = 4,
               model: SRNet | None = None,
               ckpt_path: str | None = None,
               tile_size: int = 256,
               device: torch.device = DEVICE,
               tta: bool = False,
               fp16: bool = False,
               batch_tiles: int = 4,
               ) -> np.ndarray:
    """超分辨率放大 DEM（优化版）。

    支持分块推理、TTA 增强、FP16 混合精度、批量分块。

    Parameters
    ----------
    dem_lr : ndarray (H, W) float32  低分辨率 DEM（米制）
    scale : int  放大倍数（2/4/8）
    model : SRNet, optional  已加载权重的模型
    ckpt_path : str, optional  权重文件路径
    tile_size : int  分块大小（LR 域像素）
    device : torch.device  推理设备
    tta : bool  是否启用 8 方向 TTA
    fp16 : bool  是否启用 FP16 混合精度
    batch_tiles : int  批量分块推理数（提升 GPU 利用率）

    Returns
    -------
    ndarray (H*scale, W*scale) float32  高分辨率 DEM（米制）
    """
    if scale not in (2, 4, 8):
        raise ValueError(f"不支持的 scale={scale}，仅支持 2/4/8")

    # 加载模型
    if model is None:
        model = SRNet(scale=scale)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=True)
            model.load_state_dict(ckpt)
    model = model.to(device).eval()

    # FP16 混合精度
    use_fp16 = fp16 and device.type in ("cuda", "mps")
    if use_fp16:
        model = model.half()

    H, W = dem_lr.shape
    if H <= tile_size and W <= tile_size:
        # 小图直接推理
        return _sr_infer_tile(dem_lr, model, device, scale,
                              tta=tta, use_fp16=use_fp16)

    # 大图分块推理
    overlap = tile_size // 4   # 25% 重叠
    stride = tile_size - overlap
    out_h, out_w = H * scale, W * scale
    out_acc = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    # 收集所有瓦片坐标
    tiles = []
    for h0 in range(0, H, stride):
        for w0 in range(0, W, stride):
            h_end = min(h0 + tile_size, H)
            w_end = min(w0 + tile_size, W)
            h0 = max(0, h_end - tile_size)
            w0 = max(0, w_end - tile_size)
            tiles.append((h0, h_end, w0, w_end))

    # 批量推理
    for batch_start in range(0, len(tiles), batch_tiles):
        batch = tiles[batch_start:batch_start + batch_tiles]
        # 统一填充到 tile_size
        batch_tensors = []
        for h0, h_end, w0, w_end in batch:
            tile_lr = dem_lr[h0:h_end, w0:w_end]
            x = elev_to_norm(tile_lr)
            x = torch.from_numpy(x).float()
            if use_fp16:
                x = x.half()
            # 填充到 tile_size × tile_size
            pad_h = tile_size - x.shape[0]
            pad_w = tile_size - x.shape[1]
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x.unsqueeze(0).unsqueeze(0),
                          (0, pad_w, 0, pad_h), mode="reflect")
            else:
                x = x.unsqueeze(0).unsqueeze(0)
            batch_tensors.append(x)

        batch_x = torch.cat(batch_tensors, dim=0).to(device)

        # 推理
        if tta:
            batch_y = _apply_tta(model, batch_x)
        else:
            with torch.no_grad():
                batch_y = model(batch_x)

        batch_y = batch_y.float().cpu().numpy()

        # 写回
        for i, (h0, h_end, w0, w_end) in enumerate(batch):
            th = h_end - h0
            tw = w_end - w0
            tile_hr = batch_y[i, 0, :th * scale, :tw * scale]
            tile_hr = norm_to_elev(tile_hr).astype(np.float32)

            oh0, ow0 = h0 * scale, w0 * scale
            oh_end, ow_end = oh0 + th * scale, ow0 + tw * scale

            win = _make_sr_window(th * scale, tw * scale,
                                  overlap * scale,
                                  is_top=h0 == 0, is_bottom=h_end == H,
                                  is_left=w0 == 0, is_right=w_end == W)
            out_acc[oh0:oh_end, ow0:ow_end] += tile_hr * win
            weight[oh0:oh_end, ow0:ow_end] += win

    return out_acc / np.maximum(weight, 1e-8)


def _sr_infer_tile(dem_lr: np.ndarray, model: SRNet,
                   device: torch.device, scale: int = 4,
                   tta: bool = False, use_fp16: bool = False
                   ) -> np.ndarray:
    """单块超分推理。"""
    x = elev_to_norm(dem_lr)
    x = torch.from_numpy(x).float()
    if use_fp16:
        x = x.half()
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    if tta:
        y = _apply_tta(model, x)
    else:
        with torch.no_grad():
            y = model(x)

    y = y.float().squeeze().cpu().numpy()
    return norm_to_elev(y).astype(np.float32)


def _make_sr_window(h: int, w: int, overlap: int,
                    is_top=False, is_bottom=False,
                    is_left=False, is_right=False) -> np.ndarray:
    """超分混合窗口（余弦渐变 + 边界标志）。"""
    win = np.ones((h, w), dtype=np.float32)
    if overlap <= 0:
        return win
    overlap = min(overlap, h // 2, w // 2)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, overlap)))
    if h > overlap:
        if not is_top:
            win[:overlap, :] *= ramp[:, None]
        if not is_bottom:
            win[-overlap:, :] *= ramp[::-1][:, None]
    if w > overlap:
        if not is_left:
            win[:, :overlap] *= ramp[None, :]
        if not is_right:
            win[:, -overlap:] *= ramp[None, ::-1]
    return win


# ==================== 8. 训练数据集 ====================

class SRDataset(Dataset):
    """DEM 超分辨率训练数据集。

    从高分辨率 DEM 切片生成 (LR, HR) 训练对。
    LR 通过下采样 HR 获得（bicubic）。

    用法
    ----
        >>> dataset = SRDataset("terrain/data/dem_hr.npy",
        ...                     patch_size=128, scale=4)
        >>> loader = DataLoader(dataset, batch_size=8, shuffle=True)
    """

    def __init__(self, dem_hr: np.ndarray | str,
                 patch_size: int = 128,
                 scale: int = 4,
                 n_patches: int = 1000,
                 augment: bool = True,
                 seed: int = 42):
        """初始化数据集。

        Parameters
        ----------
        dem_hr : ndarray or str  HR DEM（米制）或文件路径
        patch_size : int  HR 切片大小
        scale : int  超分倍数
        n_patches : int  总切片数（随机采样）
        augment : bool  数据增强（翻转/旋转）
        seed : int  随机种子
        """
        if isinstance(dem_hr, str):
            dem_hr = np.load(dem_hr)
        self.dem_hr = dem_hr.astype(np.float32)
        self.patch_size = patch_size
        self.scale = scale
        self.n_patches = n_patches
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        # LR patch 大小
        self.lr_patch = patch_size // scale

    def __len__(self) -> int:
        """返回预设切片数 n_patches。"""
        return self.n_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (lr, hr) 训练对，均归一化到 [-1, 1]。"""
        H, W = self.dem_hr.shape
        ps = self.patch_size

        # 随机裁剪 HR patch
        h0 = self.rng.randint(0, H - ps)
        w0 = self.rng.randint(0, W - ps)
        hr_patch = self.dem_hr[h0:h0 + ps, w0:w0 + ps]

        # 下采样生成 LR
        from scipy.ndimage import zoom as ndi_zoom
        lr_patch = ndi_zoom(hr_patch, 1.0 / self.scale, order=1)

        # 归一化 + 裁剪到 [-1, 1]
        hr_norm = np.clip(elev_to_norm(hr_patch), -1, 1)
        lr_norm = np.clip(elev_to_norm(lr_patch), -1, 1)

        # 数据增强
        if self.augment:
            hr_norm, lr_norm = self._augment(hr_norm, lr_norm, idx)

        # 转 Tensor
        hr_t = torch.from_numpy(hr_norm).float().unsqueeze(0)
        lr_t = torch.from_numpy(lr_norm).float().unsqueeze(0)
        return lr_t, hr_t

    def _augment(self, hr: np.ndarray, lr: np.ndarray,
                 idx: int) -> tuple[np.ndarray, np.ndarray]:
        """数据增强：翻转/旋转（hr 和 lr 同步变换）。"""
        mode = idx % 8
        if mode == 0:
            pass
        elif mode == 1:
            hr = hr[:, ::-1].copy(); lr = lr[:, ::-1].copy()
        elif mode == 2:
            hr = hr[::-1].copy(); lr = lr[::-1].copy()
        elif mode == 3:
            hr = hr[::-1, ::-1].copy(); lr = lr[::-1, ::-1].copy()
        elif mode == 4:
            hr = hr.T.copy(); lr = lr.T.copy()
        elif mode == 5:
            hr = hr[:, ::-1].T.copy(); lr = lr[:, ::-1].T.copy()
        elif mode == 6:
            hr = hr[::-1].T.copy(); lr = lr[::-1].T.copy()
        elif mode == 7:
            hr = hr[::-1, ::-1].T.copy(); lr = lr[::-1, ::-1].T.copy()
        return hr, lr


# ==================== 9. 训练入口 ====================

def train_sr(dem_hr: np.ndarray | str,
             scale: int = 4,
             epochs: int = 50,
             batch_size: int = 8,
             lr: float = 2e-4,
             patch_size: int = 128,
             use_attention: bool = True,
             use_charbonnier: bool = True,
             lambda_freq: float = 0.1,
             lambda_ssim: float = 0.1,
             lambda_grad: float = 0.1,
             device: torch.device = DEVICE,
             save_path: str = "terrain/checkpoints/sr_net.pt",
             n_patches: int | None = None,
             ) -> dict:
    """训练 DEM 超分辨率网络。

    Parameters
    ----------
    dem_hr : ndarray or str  训练用 HR DEM
    scale : int  超分倍数
    epochs : int  训练轮数
    batch_size : int  批量大小
    lr : float  学习率
    patch_size : int  HR 切片大小
    use_attention : bool  启用 CCA 注意力
    use_charbonnier : bool  用 Charbonnier 损失
    lambda_freq/ssim/grad : float  各损失权重
    device : torch.device  训练设备
    save_path : str  权重保存路径

    Returns
    -------
    dict  训练历史（每轮 loss）
    """
    from pathlib import Path
    from torch.optim import Adam

    # 数据集
    if n_patches is None:
        n_patches = epochs * batch_size * 10
    dataset = SRDataset(dem_hr, patch_size=patch_size, scale=scale,
                        n_patches=n_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0)

    # 模型
    model = SRNet(scale=scale, use_attention=use_attention,
                  base=32, n_rrdb=3).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {"epoch": [], "pixel": [], "ssim": [], "grad": [],
               "freq": [], "total": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"pixel": 0, "ssim": 0, "grad": 0,
                        "freq": 0, "total": 0}
        n_batches = 0

        for lr_patch, hr_patch in loader:
            lr_patch = lr_patch.to(device)
            hr_patch = hr_patch.to(device)

            pred = model(lr_patch)
            # 确保尺寸匹配
            if pred.shape[-2:] != hr_patch.shape[-2:]:
                pred = F.interpolate(pred, size=hr_patch.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False)

            loss, parts = sr_loss(
                pred, hr_patch,
                lambda_freq=lambda_freq,
                lambda_ssim=lambda_ssim,
                lambda_grad=lambda_grad,
                use_charbonnier=use_charbonnier)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += parts.get(k, 0)
            n_batches += 1

        # 记录
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        history["epoch"].append(epoch)
        for k in ["pixel", "ssim", "grad", "freq", "total"]:
            history[k].append(epoch_losses[k])

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [SR] epoch {epoch+1}/{epochs}: "
                  f"total={epoch_losses['total']:.4f} "
                  f"pix={epoch_losses['pixel']:.4f}")

    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    history["save_path"] = save_path
    history["param_count"] = model.param_count
    return history
