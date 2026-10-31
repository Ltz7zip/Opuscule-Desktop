"""阶段 5C：河网生成网络（U-Net 架构）。

从 DEM 高程生成真实河网（掩码 + 归一化流量），替代/增强物理规则河网。

架构特点
--------
1. **U-Net 编码器-解码器**：4 层下采样 + 4 层上采样，跳跃连接
2. **深度可分离卷积**：减少参数量，加速训练
3. **通道注意力**：在瓶颈层增强特征选择
4. **双输出头**：掩码（Sigmoid）+ 流量（Sigmoid，0~1）

损失设计
--------
- 掩码：BCE + Dice Loss（处理河流/陆地类别不平衡）
- 流量：L1 Loss（回归归一化流量）

用法
----
    >>> from terrain.dl.river_net import RiverNet, river_inference
    >>> dem = np.load("dem_1024.npy")  # (1024, 1024) float32, 米
    >>> mask, flow = river_inference(dem)  # mask: bool, flow: float32 [0,1]
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import (
    DEVICE, ELEV_MIN, ELEV_MAX, elev_to_norm, norm_to_elev,
    CKPT_DIR, BATCH_SIZE, LEARNING_RATE,
)


# ==================== 1. 模型组件 ====================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：depthwise + pointwise。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        """初始化深度可分离卷积：depthwise + pointwise + BN + SiLU。"""
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        """前向：depthwise→pointwise→BN→激活。"""
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class ChannelAttention(nn.Module):
    """通道注意力（SE 变体）。"""

    def __init__(self, channels, reduction=8):
        """初始化通道注意力：池化 + 两层 1×1 卷积压缩扩展。"""
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.SiLU(),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """前向：x 按通道注意力权重重标定。"""
        return x * self.fc(self.avg_pool(x))


class DoubleConv(nn.Module):
    """两层卷积块（U-Net 标准组件）。"""

    def __init__(self, in_ch, out_ch):
        """初始化双层卷积块。"""
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x):
        """前向：x 经两层深度可分离卷积。"""
        return self.conv(x)


class Down(nn.Module):
    """下采样块：MaxPool + DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        """初始化下采样块：MaxPool + DoubleConv。"""
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        """前向：先池化再卷积。"""
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样块：双线性插值 + 跳跃连接 + DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        """初始化上采样块：双线性上采样 + 跳跃连接 + DoubleConv。"""
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        """前向：上采样后与跳跃连接拼接再卷积。"""
        x = self.up(x)
        # 尺寸对齐
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ==================== 2. U-Net 模型 ====================

class RiverNet(nn.Module):
    """河网生成 U-Net：DEM → (河流掩码, 归一化流量)。

    Parameters
    ----------
    in_ch : int   输入通道数（DEM=1）
    base : int    基础通道数
    out_ch : int  输出通道数（mask + flow = 2）
    """

    def __init__(self, in_ch: int = 1, base: int = 32, out_ch: int = 2):
        """初始化河网 U-Net 的编码器/瓶颈/解码器与双输出头。"""
        super().__init__()
        b = base
        # 编码器
        self.enc1 = DoubleConv(in_ch, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)
        # 瓶颈
        self.bottleneck = nn.Sequential(
            Down(b * 8, b * 16),
            ChannelAttention(b * 16),
        )
        # 解码器
        self.dec4 = Up(b * 16 + b * 8, b * 8)
        self.dec3 = Up(b * 8 + b * 4, b * 4)
        self.dec2 = Up(b * 4 + b * 2, b * 2)
        self.dec1 = Up(b * 2 + b, b)
        # 输出头
        self.out_conv = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        """前向：x 经 U-Net 输出未激活的 (B, 2, H, W) 张量。"""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out = self.out_conv(d1)
        return out  # (B, 2, H, W) —— 未激活，训练时用 BCEWithLogits

    def predict(self, x):
        """推理：返回 (mask_prob, flow)。"""
        out = self.forward(x)
        mask = torch.sigmoid(out[:, 0:1])
        flow = torch.sigmoid(out[:, 1:2])
        return mask, flow


# ==================== 3. 损失函数 ====================

def dice_loss(pred_logits, target, eps=1e-6):
    """Dice 损失（处理类别不平衡）。"""
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def river_loss(pred_logits, target_mask, target_flow,
               lambda_dice=1.0, lambda_flow=0.5, pos_weight=7.0):
    """河网总损失：加权 BCE + Dice（掩码）+ L1（流量）。

    Parameters
    ----------
    pred_logits : (B, 2, H, W)  模型输出（未激活）
    target_mask : (B, 1, H, W)   河流掩码 0/1
    target_flow : (B, 1, H, W)   归一化流量 [0,1]
    pos_weight : float           正样本权重（河流占~12%, 取 7.0 平衡类别）
    """
    pred_mask_logits = pred_logits[:, 0:1]
    pred_flow_logits = pred_logits[:, 1:2]

    pw = torch.tensor([pos_weight], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        pred_mask_logits, target_mask, pos_weight=pw)
    dice = dice_loss(pred_mask_logits, target_mask)
    flow_loss = F.l1_loss(torch.sigmoid(pred_flow_logits), target_flow)

    total = bce + lambda_dice * dice + lambda_flow * flow_loss
    parts = {"bce": bce.item(), "dice": dice.item(),
             "flow": flow_loss.item(), "total": total.item()}
    return total, parts


# ==================== 4. 数据集 ====================

class RiverDataset(Dataset):
    """河网训练数据集：从 npy 文件加载 (DEM, mask, flow)。"""

    def __init__(self, data_dir: str, augment: bool = True):
        """初始化河网数据集：扫描目录下全部 .npy 样本。"""
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".npy")
        ])
        self.augment = augment
        print(f"  RiverDataset: {len(self.files)} 样本 from {data_dir}")

    def __len__(self):
        """返回样本数量。"""
        return len(self.files)

    def __getitem__(self, idx):
        """加载第 idx 个样本，统一尺寸并增强，返回 (dem, mask, flow) 张量。"""
        sample = np.load(self.files[idx]).astype(np.float32)
        dem = sample[..., 0:1]      # (H, W, 1) 米
        mask = sample[..., 1:2]     # (H, W, 1) 0/1
        flow = sample[..., 2:3]     # (H, W, 1) 0~1

        # 统一尺寸到 256×256（部分增强样本尺寸不一致）
        from scipy.ndimage import zoom
        if dem.shape[0] != 256 or dem.shape[1] != 256:
            h, w = dem.shape[:2]
            dem = zoom(dem, (256/h, 256/w, 1), order=1)
            mask = zoom(mask, (256/h, 256/w, 1), order=0)  # 最近邻保持二值
            flow = zoom(flow, (256/h, 256/w, 1), order=1)

        # DEM 归一化到 [-1, 1]
        dem = elev_to_norm(dem).astype(np.float32)

        if self.augment:
            # 随机水平/垂直翻转
            if np.random.rand() < 0.5:
                dem = dem[:, ::-1, :].copy()
                mask = mask[:, ::-1, :].copy()
                flow = flow[:, ::-1, :].copy()
            if np.random.rand() < 0.5:
                dem = dem[::-1, :, :].copy()
                mask = mask[::-1, :, :].copy()
                flow = flow[::-1, :, :].copy()

        # (H, W, C) -> (C, H, W)
        dem = torch.from_numpy(dem).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        return dem, mask, flow


# ==================== 5. 训练函数 ====================

def train_river_net(data_dir: str = "terrain/data/river_dataset",
                    epochs: int = 50,
                    batch_size: int = BATCH_SIZE,
                    lr: float = LEARNING_RATE,
                    base: int = 32,
                    lambda_dice: float = 1.0,
                    lambda_flow: float = 0.5,
                    steps_per_epoch: int = 500,
                    val_steps: int = 100,
                    device: torch.device = DEVICE,
                    save_path: str = None,
                    ) -> dict:
    """训练河网生成 U-Net。

    Parameters
    ----------
    data_dir : str   数据集根目录（含 train/val/test）
    epochs : int     训练轮数
    batch_size : int 批量大小
    lr : float       学习率
    base : int       基础通道数
    save_path : str  权重保存路径

    Returns
    -------
    dict  训练历史
    """
    if save_path is None:
        save_path = str(CKPT_DIR / "river_net.pt")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = RiverDataset(train_dir, augment=True)
    val_ds = RiverDataset(val_dir, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    model = RiverNet(in_ch=1, base=base, out_ch=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  RiverNet 参数: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_iou": [], "val_iou": []}
    best_val_iou = 0.0

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        train_losses = {"bce": 0, "dice": 0, "flow": 0, "total": 0}
        n_batches = 0
        train_iter = iter(train_loader)
        for _ in range(steps_per_epoch):
            try:
                dem, mask, flow = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                dem, mask, flow = next(train_iter)
            dem = dem.to(device)
            mask = mask.to(device)
            flow = flow.to(device)

            pred = model(dem)
            loss, parts = river_loss(pred, mask, flow,
                                     lambda_dice=lambda_dice,
                                     lambda_flow=lambda_flow)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += parts.get(k, 0)
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # ---- 验证 ----
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_n = 0
        val_iter = iter(val_loader)
        with torch.no_grad():
            for _ in range(val_steps):
                try:
                    dem, mask, flow = next(val_iter)
                except StopIteration:
                    break
                dem = dem.to(device)
                mask = mask.to(device)
                flow = flow.to(device)
                pred = model(dem)
                loss, _ = river_loss(pred, mask, flow,
                                     lambda_dice=lambda_dice,
                                     lambda_flow=lambda_flow)
                val_loss_sum += loss.item()
                # IoU
                pred_mask = (torch.sigmoid(pred[:, 0:1]) > 0.5).float()
                inter = (pred_mask * mask).sum()
                union = (pred_mask + mask).clamp(0, 1).sum()
                val_iou_sum += (inter / (union + 1e-6)).item()
                val_n += 1

        val_loss = val_loss_sum / max(val_n, 1)
        val_iou = val_iou_sum / max(val_n, 1)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [River] epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_losses['total']:.4f} "
                  f"(bce={train_losses['bce']:.4f} dice={train_losses['dice']:.4f} "
                  f"flow={train_losses['flow']:.4f}) "
                  f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # 保存最佳
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_iou": val_iou,
                "config": {"base": base, "lambda_dice": lambda_dice,
                           "lambda_flow": lambda_flow},
            }, save_path)

    print(f"  最佳 val_iou={best_val_iou:.4f}，权重已保存至 {save_path}")
    return history


# ==================== 6. 推理 ====================

def river_inference(dem: np.ndarray,
                    model_path: str = None,
                    device: torch.device = DEVICE,
                    mask_threshold: float = 0.5,
                    ) -> tuple:
    """对 DEM 做河网推理。

    Parameters
    ----------
    dem : (H, W) float32  高程（米）
    model_path : str       权重路径
    mask_threshold : float 掩码二值化阈值

    Returns
    -------
    (mask, flow) : (H,W) bool, (H,W) float32
    """
    if model_path is None:
        model_path = str(CKPT_DIR / "river_net.pt")

    h, w = dem.shape
    #  padding 到 16 的倍数
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    dem_pad = np.pad(dem, ((0, pad_h), (0, pad_w)), mode="edge")

    dem_t = torch.from_numpy(
        elev_to_norm(dem_pad.astype(np.float32))[None, None]
    ).to(device)

    model = RiverNet(in_ch=1, base=32, out_ch=2).to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        mask_logits = model(dem_t)
        mask_prob = torch.sigmoid(mask_logits[:, 0:1])
        flow = torch.sigmoid(mask_logits[:, 1:2])

    mask = (mask_prob[0, 0].cpu().numpy() > mask_threshold)[:h, :w]
    flow = flow[0, 0].cpu().numpy()[:h, :w]
    return mask, flow


# ==================== 7. 混合策略：流量分级连通 + 骨架化 + DL 增强 ====================

def hybrid_river_extraction(dem: np.ndarray,
                            model_path: str = None,
                            device: torch.device = DEVICE,
                            q_main: float = 0.96,
                            q_trib: float = 0.86,
                            q_source: float = 0.72,
                            min_component: int = 30,
                            dl_enhance: bool = True,
                            dl_thresh: float = 0.35,
                            outlet_pct: float = 0.85,
                            min_source_acc: int = 10,
                            max_rivers: int = 500,
                            min_river_length: int = 30,
                            peak_size: int = 5,
                            peak_tol: float = 20.0,
                            slope_thresh: float = 60.0,
                            slope_acc_thresh: float = 50.0,
                            dem_smooth_sigma: float = 0.3,
                            fix_acc_thresh: float = 100.0,
                            main_acc_pct: float = 0.98,
                            ) -> tuple:
    """混合河网提取 v2：流域树状追溯 + 源头延伸到山脊 + 坡度一致性过滤。

    策略
    ----
    1. **流域树状追溯**：填充洼地 → D8 流向+流量累积 → 找海洋出口 → 追溯上游
    2. **源头延伸到山脊**：将不在山脊线上的源头沿上游方向延伸到山脊线
    3. **坡度一致性过滤**：移除流向与坡度偏差 > slope_thresh 且 acc < slope_acc_thresh 的逆坡段
    4. **过滤**：移除 < min_component 的孤立小分量（8 邻域）
    5. **DL 增强**(可选)：用 RiverNet 补充物理规则遗漏的支流

    Parameters
    ----------
    dem : (H, W) float32  高程（米）
    outlet_pct : float   出口流量分位数阈值
    min_source_acc : int 源头最小流量累积（控制支流密度）
    max_rivers : int    最大追溯河流数
    min_river_length : int  最小河流长度
    peak_size : int      山脊线检测窗口大小
    peak_tol : float     山脊线高程阈值（米）
    slope_thresh : float  坡度一致性角度阈值（度），超过则视为逆坡
    slope_acc_thresh : float  逆坡过滤的 acc 上限（只过滤低流量支流末端）
    dem_smooth_sigma : float  DEM 高斯平滑 sigma（消除 D8 阶梯状锯齿，0=关闭）
    min_component : int  最小连通域像素数
    dl_enhance : bool  是否启用 DL 支流增强
    dl_thresh : float  DL 概率阈值

    Returns
    -------
    (mask, flow) : (H,W) bool, (H,W) float32
    """
    from scipy import ndimage as ndi
    from scipy.ndimage import gaussian_filter
    from terrain.modules.hydro_module import fill_depressions
    from terrain.core.terrain_engine import m2v, v2m

    land = dem > 0
    h, w = dem.shape

    # ---- 1. 物理流量累积 + 流向（同一随机序列，保证一致）----
    rng = np.random.RandomState(42)
    v01 = m2v(dem)
    filled = v2m(fill_depressions(v01))

    # ---- 山脊线检测（源头约束：河流源头应在局部高地）----
    ridge_avg = ndi.uniform_filter(filled, size=peak_size)
    is_ridge = land & (filled >= ridge_avg + peak_tol)

    # ---- 轻度 DEM 平滑（消除 D8 阶梯状锯齿）----
    # 只对流向计算用的高程做平滑，山脊检测保持原始分辨率
    _D8 = [(-1, -1, 1.41421356), (-1, 0, 1.0), (-1, 1, 1.41421356), (0, -1, 1.0),
           (0, 1, 1.0), (1, -1, 1.41421356), (1, 0, 1.0), (1, 1, 1.41421356)]
    if dem_smooth_sigma > 0:
        filled_s = gaussian_filter(filled, sigma=dem_smooth_sigma)
        filled_s[~land] = filled[~land]  # 海洋保持不变
    else:
        filled_s = filled

    # 计算原始 DEM 的 acc（用于主干识别，不受平滑影响）
    e_orig = filled.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)
    best_s_o = np.full((h, w), 1e-9, np.float32)
    recv_flat_o = np.full((h, w), -1, np.int32)
    for di, (dy, dx, dd) in enumerate(_D8):
        shift = np.full((h, w), np.inf, np.float32)
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
        yt, xt = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
        shift[yt, xt] = e_orig[ys, xs]
        s = (e_orig - shift) / dd
        m = s > best_s_o
        if m.any():
            ys_m, xs_m = np.nonzero(m)
            recv_flat_o[m] = ((ys_m + dy) * w + (xs_m + dx)).astype(np.int32)
            best_s_o[m] = s[m]
    flat_o = recv_flat_o.ravel()
    order_o = np.argsort(-e_orig, axis=None)
    acc_orig = np.ones(h * w, np.float32)
    for i in order_o:
        r = flat_o[i]
        if r >= 0:
            acc_orig[r] += acc_orig[i]
    acc_orig = acc_orig.reshape(h, w)

    e = filled_s.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)
    dyx = [(d[0], d[1]) for d in _D8]
    rev_dir = {0:7, 1:6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:0}
    best_s = np.full((h, w), 1e-9, np.float32)
    recv_idx = np.full((h, w), -1, np.int32)
    recv_flat = np.full((h, w), -1, np.int32)
    for di, (dy, dx, dd) in enumerate(_D8):
        shift = np.full((h, w), np.inf, np.float32)
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
        yt, xt = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
        shift[yt, xt] = e[ys, xs]
        s = (e - shift) / dd
        m = s > best_s
        if m.any():
            yy, xx = np.nonzero(m)
            recv_idx[yy, xx] = di
            recv_flat[yy, xx] = ((yy + dy) * w + (xx + dx)).astype(np.int32)
            best_s[m] = s[m]
    flat = recv_flat.ravel()
    order = np.argsort(-e, axis=None)
    acc = np.ones(h * w, np.float32)
    for i in order:
        r = flat[i]
        if r >= 0:
            acc[r] += acc[i]
    acc = acc.reshape(h, w)

    aL = acc[land]
    acc_thr = float(np.quantile(aL, outlet_pct))

    # ---- 2. 找海洋出口 ----
    ocean = ~land
    dilated_ocean = ndi.binary_dilation(ocean, iterations=1)
    outlets = land & dilated_ocean & (acc > acc_thr)
    if outlets.sum() == 0:
        edge = np.zeros_like(land)
        edge[0, :] = land[0, :]; edge[-1, :] = land[-1, :]
        edge[:, 0] = land[:, 0]; edge[:, -1] = land[:, -1]
        outlets = edge & (acc > acc_thr)
    oy, ox = np.where(outlets)
    if len(oy) > 0:
        outlet_acc = acc[oy, ox]
        order2 = np.argsort(-outlet_acc)
        oy, ox = oy[order2], ox[order2]

    # ---- 3. 从出口递归追溯所有上游分支（树状）+ 源头延伸到山脊 ----
    graded = np.zeros_like(land, dtype=bool)
    for idx in range(min(len(oy), max_rivers)):
        sy, sx = int(oy[idx]), int(ox[idx])
        if graded[sy, sx]:
            continue
        # 追溯
        stack = [(sy, sx)]
        visited = set()
        while stack:
            cy, cx = stack.pop()
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            for d in range(8):
                dy, dx = dyx[d]
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and land[ny, nx]:
                    if recv_idx[ny, nx] == rev_dir[d] and (ny, nx) not in visited:
                        if acc[ny, nx] >= min_source_acc:
                            stack.append((ny, nx))
        if len(visited) < min_river_length:
            continue

        # 源头延伸到山脊线
        temp = np.zeros_like(land, dtype=bool)
        for py, px in visited:
            temp[py, px] = True
        for _ in range(30):
            n4 = ndi.convolve(temp.astype(int), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
            endpoints = temp & (n4 == 1) & (~is_ridge)
            if not endpoints.any():
                break
            ey, ex = np.where(endpoints)
            extended = False
            for py, px in zip(ey, ex):
                for d in range(8):
                    dy, dx = dyx[d]
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < h and 0 <= nx < w and land[ny, nx]:
                        if recv_idx[ny, nx] == rev_dir[d] and not temp[ny, nx]:
                            temp[ny, nx] = True
                            extended = True
                            break
            if not extended:
                break
        graded |= temp

    # ---- 3.5 坡度一致性过滤：移除逆坡的低流量支流末端 ----
    # D8 流向角度
    river_angle = np.full((h, w), np.nan, np.float32)
    for d, (dy, dx) in enumerate(dyx):
        river_angle[recv_idx == d] = np.arctan2(dy, dx)
    # 坡度方向（最陡下坡，基于平滑后的 DEM）
    gy, gx = np.gradient(filled_s)
    slope_angle = np.arctan2(gy, gx)
    angle_diff = np.abs(river_angle - slope_angle)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff) * 180.0 / np.pi
    # 移除：角度差 > 阈值 且 acc < 阈值（只过滤支流末端，不动主干）
    bad_slope = graded & (angle_diff > slope_thresh) & (acc < slope_acc_thresh)
    graded[bad_slope] = False

    struct8 = ndi.generate_binary_structure(2, 2)

    # ---- 4. 过滤孤立小分量（8 邻域）----
    # 树状追溯天然无辫状交叉，无需骨架化（骨架化会切碎长河流）
    graded = graded & land
    labeled, n = ndi.label(graded, structure=np.ones((3, 3), dtype=int))
    sizes = ndi.sum(graded, labeled, range(1, n + 1))
    keep = np.zeros_like(graded, bool)
    for i, sz in enumerate(sizes, 1):
        if sz >= min_component:
            keep |= (labeled == i)
    final_mask = keep & land

    # ---- 4.5 精准断裂修复：连接 DEM 平滑导致的高 acc 主干断裂 ----
    # 只修复 acc > fix_acc_thresh 的主干端点，避免连接不同河流
    if dem_smooth_sigma > 0:
        k4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
        for _ in range(3):  # 最多修复 3 轮
            n4 = ndi.convolve(final_mask.astype(int), k4)
            endpoints = final_mask & (n4 == 1) & (acc > fix_acc_thresh)
            ey, ex = np.where(endpoints)
            fixed = False
            for y, x in zip(ey, ex):
                for dist in [2, 3]:  # 检查 2 格和 3 格距离
                    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                   (0, 1), (1, -1), (1, 0), (1, 1)]:
                        ny2, nx2 = y + dy * dist, x + dx * dist
                        if (0 <= ny2 < h and 0 <= nx2 < w and final_mask[ny2, nx2]
                                and acc[ny2, nx2] > fix_acc_thresh):
                            # 连接中间所有格
                            for d in range(1, dist):
                                my, mx = y + dy * d, x + dx * d
                                if land[my, mx]:
                                    final_mask[my, mx] = True
                            fixed = True
                            break
                    if fixed:
                        break
            if not fixed:
                break

    # ---- 4.6 主干补充：用原始 DEM 的高 acc 像素修复平滑后丢失的主干 ----
    # 平滑会改变部分主干流向，用原始 acc 前 main_acc_pct 分位的像素补充
    if dem_smooth_sigma > 0 and main_acc_pct > 0:
        main_thr = float(np.quantile(acc_orig[land], main_acc_pct))
        main_pixels = (acc_orig > main_thr) & land & ~final_mask
        final_mask = final_mask | main_pixels
        # 清理补充后产生的孤立小分量
        labeled, n = ndi.label(final_mask, structure=np.ones((3, 3), dtype=int))
        sizes = ndi.sum(final_mask, labeled, range(1, n + 1))
        keep = np.zeros_like(final_mask, bool)
        for i, sz in enumerate(sizes, 1):
            if sz >= min_component:
                keep |= (labeled == i)
        final_mask = keep & land

    # ---- 5. 可选 DL 增强：补充物理遗漏的支流 ----
    dl_flow_full = None
    if dl_enhance:
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        dem_pad = np.pad(dem, ((0, pad_h), (0, pad_w)), mode="edge")
        dem_t = torch.from_numpy(
            elev_to_norm(dem_pad.astype(np.float32))[None, None]
        ).to(device)
        model = RiverNet(in_ch=1, base=32, out_ch=2).to(device)
        if model_path is None:
            model_path = str(CKPT_DIR / "river_net.pt")
        ck = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            out = model(dem_t)
            dl_prob = torch.sigmoid(out[:, 0:1])[0, 0].cpu().numpy()[:h, :w]
            dl_flow_full = torch.sigmoid(out[:, 1:2])[0, 0].cpu().numpy()[:h, :w]

        dl_candidates = (dl_prob > dl_thresh) & land & (~final_mask)
        dilated_final = ndi.binary_dilation(final_mask, structure=struct8, iterations=2)
        dl_add = dl_candidates & dilated_final
        if dl_add.any():
            combined = final_mask | dl_add
            combined = ndi.binary_closing(combined, structure=struct8, iterations=1)
            combined = combined & land
            labeled2, n2 = ndi.label(combined, structure=np.ones((3, 3), dtype=int))
            sizes2 = ndi.sum(combined, labeled2, range(1, n2 + 1))
            keep2 = np.zeros_like(combined, bool)
            for i, sz in enumerate(sizes2, 1):
                if sz >= min_component:
                    keep2 |= (labeled2 == i)
                elif sz >= 8:
                    comp = (labeled2 == i)
                    dilated = ndi.binary_dilation(comp, structure=struct8, iterations=1)
                    if (dilated & final_mask).any():
                        keep2 |= comp
            final_mask = keep2 & land

    # ---- 6. 流量估计 ----
    phys_flow = np.zeros_like(acc, np.float32)
    if graded.any():
        log_acc = np.log10(np.maximum(acc[graded], 1))
        lo, hi = log_acc.min(), log_acc.max()
        if hi > lo:
            phys_flow[graded] = (log_acc - lo) / (hi - lo)
    if dl_flow_full is not None:
        final_flow = np.where(graded, phys_flow, dl_flow_full)
    else:
        final_flow = phys_flow
    final_flow = final_flow * final_mask

    return final_mask, final_flow
