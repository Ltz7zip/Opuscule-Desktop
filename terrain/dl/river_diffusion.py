"""条件 DDPM 河流精修模型：DEM + D8 mask → 自然河网 mask。

架构
----
- **条件编码器**：将 DEM + D8 物理河流 mask 编码为多尺度条件特征
- **去噪 U-Net**：在噪声 mask 的去噪过程中，通过加性注入条件特征
- **条件注入**：编码器和解码器的每个 Block 都接收对应尺度的条件特征

设计理由
--------
1. D8 物理河流提供了合理的拓扑先验（树状结构、出口位置）
2. Diffusion 只需修正局部不自然模式（阶梯锯齿、辫状交叉）
3. 加性条件注入比拼接更轻量，不改变通道数，兼容现有 Block 设计
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .unet import TimeEmbed, _pick_groups


# ==================== 1. 基础 Block ====================

class ConvBlock(nn.Module):
    """简单 Conv 块（无时间/条件）：GroupNorm + SiLU + Conv × 2 + 残差。

    用于条件编码器，不需要时间嵌入或条件注入。
    """

    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        """初始化简单卷积块，构建两层 GroupNorm+SiLU+Conv 与残差跳跃连接。"""
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


class CondBlock(nn.Module):
    """带时间 + 条件注入的 Conv 块。

    结构：GN → SiLU → Conv → +时间调制 → +条件调制 → GN → SiLU → Conv → +残差

    条件注入：cond 特征经 1×1 Conv 投影到 c_out 通道后**加到**中间特征。
    若 cond=None，退化为标准时间调制 Block。
    """

    def __init__(self, c_in: int, c_out: int, time_dim: int,
                 cond_ch: int = 0, groups: int = 8):
        """初始化带时间与条件注入的卷积块，构建时间 MLP 与条件投影。"""
        super().__init__()
        g1 = _pick_groups(c_in, groups)
        g2 = _pick_groups(c_out, groups)
        self.norm1 = nn.GroupNorm(g1, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(g2, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, c_out) if time_dim > 0 else None
        self.cond_proj = (
            nn.Conv2d(cond_ch, c_out, 1) if cond_ch > 0 else None
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        """前向：x 经卷积与时间/条件调制后加残差返回。"""
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if self.time_mlp is not None:
            h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        if cond is not None and self.cond_proj is not None:
            h = h + self.cond_proj(cond)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ==================== 2. 条件编码器 ====================

class ConditionEncoder(nn.Module):
    """条件编码器：DEM + D8 mask → 多尺度条件特征。

    输入：(B, cond_ch, 256, 256) —— [DEM 归一化, D8 mask 0/1]
    输出：4 个尺度的条件特征

    [cf1] (B, base, 256, 256)
    [cf2] (B, base*2, 128, 128)
    [cf3] (B, base*4, 64, 64)
    [cf4] (B, base*8, 32, 32)
    """

    def __init__(self, cond_ch: int = 2, base: int = 64, groups: int = 8):
        """初始化条件编码器，构建四层卷积块与池化下采样。"""
        super().__init__()
        self.d1 = ConvBlock(cond_ch, base, groups)          # 256
        self.d2 = ConvBlock(base, base * 2, groups)          # 128
        self.d3 = ConvBlock(base * 2, base * 4, groups)     # 64
        self.d4 = ConvBlock(base * 4, base * 8, groups)     # 32
        self.pool = nn.MaxPool2d(2)

    def forward(self, cond):
        """前向：输出四个尺度的条件特征 (cf1~cf4)。"""
        cf1 = self.d1(cond)                          # 256, base
        cf2 = self.d2(self.pool(cf1))                # 128, base*2
        cf3 = self.d3(self.pool(cf2))                # 64, base*4
        cf4 = self.d4(self.pool(cf3))                # 32, base*8
        return cf1, cf2, cf3, cf4


# ==================== 3. 条件 Diffusion U-Net ====================

class ConditionalRiverDiffusion(nn.Module):
    """条件 DDPM 河流精修 U-Net。

    256² → 32² 瓶颈 → 256²，3 次下/上采样。
    每层接收对应尺度的条件特征（加性注入）。

    Parameters
    ----------
    cond_ch : int   条件输入通道数（DEM + D8 mask = 2）
    base : int      基础通道数，默认 64
    time_dim : int  时间嵌入维度，默认 128

    forward 参数
    -------------
    x_t : (B, 1, 256, 256)  噪声河流 mask
    t : (B,)                时间步
    cond : (B, cond_ch, 256, 256)  条件 [DEM_norm, D8_mask]

    Returns
    -------
    eps : (B, 1, 256, 256)  预测噪声
    """

    def __init__(self, cond_ch: int = 2, base: int = 64, time_dim: int = 128):
        """初始化条件河流扩散 U-Net，构建条件编码器与编解码器各层。"""
        super().__init__()
        self.time = TimeEmbed(time_dim)

        # 条件编码器
        self.cond_enc = ConditionEncoder(cond_ch, base)

        # ---- 编码器 ----
        self.e1 = CondBlock(1, base, time_dim, cond_ch=base)           # 256
        self.e2 = CondBlock(base, base * 2, time_dim, cond_ch=base * 2)  # 128
        self.e3 = CondBlock(base * 2, base * 4, time_dim, cond_ch=base * 4)  # 64
        self.e4 = CondBlock(base * 4, base * 8, time_dim, cond_ch=base * 8)  # 32
        self.pool = nn.MaxPool2d(2)

        # ---- 瓶颈 ----
        self.mid = CondBlock(base * 8, base * 8, time_dim, cond_ch=base * 8)

        # ---- 解码器 ----
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)   # 32→64
        self.u3 = CondBlock(base * 4 + base * 4, base * 4, time_dim,
                            cond_ch=base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)   # 64→128
        self.u2 = CondBlock(base * 2 + base * 2, base * 2, time_dim,
                            cond_ch=base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)       # 128→256
        self.u1 = CondBlock(base + base, base, time_dim, cond_ch=base)

        # ---- 输出头 ----
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x_t, t, cond):
        """前向：条件编码并逐层去噪，返回预测噪声 ε。"""
        te = self.time(t)
        cf1, cf2, cf3, cf4 = self.cond_enc(cond)

        # 编码器（带条件注入）
        h1 = self.e1(x_t, te, cf1)                    # 256, base
        h2 = self.e2(self.pool(h1), te, cf2)          # 128, base*2
        h3 = self.e3(self.pool(h2), te, cf3)          # 64, base*4
        h4 = self.e4(self.pool(h3), te, cf4)          # 32, base*8

        # 瓶颈
        h = self.mid(h4, te, cf4)                     # 32, base*8

        # 解码器（跳跃连接 + 条件注入）
        h = self.u3(torch.cat([self.up3(h), h3], dim=1), te, cf3)   # 64
        h = self.u2(torch.cat([self.up2(h), h2], dim=1), te, cf2)   # 128
        h = self.u1(torch.cat([self.up1(h), h1], dim=1), te, cf1)   # 256

        return self.out_conv(h)                        # (B, 1, 256, 256)


# ==================== 4. 损失函数 ====================

def _neighbor_count(mask, kernel):
    """用卷积统计邻居数。"""
    k = torch.tensor(kernel, dtype=mask.dtype, device=mask.device)
    k = k.view(1, 1, *k.shape)
    return F.conv2d(mask, k, padding=1)


def anti_braid_loss(mask_prob):
    """抗辫状正则：惩罚 4 邻域 ≥ 4 的 + 型交叉。

    Parameters
    ----------
    mask_prob : (B, 1, H, W)  河流概率 [0, 1]
    """
    k4 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    n4 = _neighbor_count(mask_prob, k4)
    # 只惩罚 n4 > 3 的像素（辫状交叉点）
    excess = F.relu(n4 - 3.0)
    return (excess * mask_prob).mean()


def connectivity_loss(mask_prob):
    """连通性正则：惩罚孤立河流像素（8 邻域 = 0）。

    Parameters
    ----------
    mask_prob : (B, 1, H, W)  河流概率 [0, 1]
    """
    k8 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    n8 = _neighbor_count(mask_prob, k8)
    # 孤立像素：mask > 0.5 但 8 邻域 = 0
    isolated = (mask_prob > 0.5).float() * (n8 < 0.5).float()
    return isolated.mean()


def river_diffusion_loss(model, x0, t, cond, ddpm, dem=None,
                         lambda_braid=0.1, lambda_conn=0.05,
                         lambda_slope=0.0):
    """河流条件 Diffusion 总损失。

    L = L_diffusion + λ_braid · L_anti_braid + λ_conn · L_connectivity

    结构正则只在低噪声时（t < T/2）施加，避免早期训练噪声干扰。

    Parameters
    ----------
    model : ConditionalRiverDiffusion
    x0 : (B, 1, H, W)       干净河流 mask [0, 1]
    t : (B,)                 时间步
    cond : (B, 2, H, W)      条件 [DEM, D8 mask]
    ddpm : DDPM              调度器
    dem : (B, 1, H, W)       原始高程（可选，用于坡度一致性损失）
    lambda_braid, lambda_conn, lambda_slope : float  正则权重

    Returns
    -------
    loss : scalar tensor
    parts : dict  各项损失值
    """
    # 1. DDPM 去噪损失
    x_t, noise = ddpm.q_sample(x0, t)
    pred_eps = model(x_t, t, cond)
    loss_diff = F.mse_loss(pred_eps, noise)
    loss = loss_diff
    parts = {"diff": loss_diff.item()}

    # 2. 结构正则（仅在低噪声时施加）
    low_noise = t < ddpm.n // 2
    if low_noise.any() and (lambda_braid > 0 or lambda_conn > 0):
        # 从预测噪声反推 x0
        ab = ddpm.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        x0_pred = (x_t - (1 - ab).sqrt() * pred_eps) / ab.sqrt().clamp(min=1e-8)
        x0_prob = torch.sigmoid(x0_pred.clamp(-5, 5))

        if lambda_braid > 0:
            lb = anti_braid_loss(x0_prob)
            loss = loss + lambda_braid * lb
            parts["braid"] = lb.item()

        if lambda_conn > 0:
            lc = connectivity_loss(x0_prob)
            loss = loss + lambda_conn * lc
            parts["conn"] = lc.item()

    parts["total"] = loss.item()
    return loss, parts


# ==================== 5. 数据集 ====================

def _morphological_smooth(mask_np, land_np):
    """对 D8 mask 做形态学平滑，生成更自然的河流目标。

    仅做闭运算连接 1 格断裂，保持原始线宽和拓扑。
    形态学滤波（中值/高斯）会破坏 1 像素宽的河流线，不适用。
    """
    from scipy import ndimage as ndi

    struct8 = np.ones((3, 3), dtype=int)
    m = ndi.binary_closing(mask_np, structure=struct8, iterations=1)
    return m & land_np


class RiverDiffusionDataset(Dataset):
    """条件 DDPM 河流数据集。

    返回 (condition, target) 对：
    - condition : (2, H, W) = [DEM_norm, D8_mask]
    - target    : (1, H, W) = 河流 mask [0,1]（DDPM 前向加噪目标）

    数据来源
    --------
    - ``mode='river_dataset'``：从 river_dataset/ 加载预计算 (DEM, mask, flow) 样本
    - ``mode='dem_tiles'``     ：从 dem_tiles/ 加载 DEM，在线计算 D8 mask

    目标策略
    --------
    - ``target='mask'``  ：直接用 D8 mask 作为目标（模型学习分布，推理时采样产生自然变体）
    - ``target='smooth'``：对 mask 做形态学平滑后作为目标（提供更自然的监督信号）

    设计理由
    --------
    1. 条件含 D8 mask 提供拓扑先验，但 DDPM 框架天然防止退化为拷贝：
       模型预测噪声 ε 而非 mask 本身，且 x_t 在大 t 时被严重噪声化
    2. target='smooth' 通过形态学平滑消除 D8 阶梯锯齿，作为更自然的训练目标
    3. 推理时从 D8 mask 加噪 → 条件去噪 → 产生平滑自然的河流变体
    """

    def __init__(self, mode: str = "river_dataset",
                 data_dir: str | None = None,
                 size: int = 256,
                 augment: bool = True,
                 target: str = "mask",
                 threshold_quantile: float = 0.995):
        """
        Parameters
        ----------
        mode : 'river_dataset' | 'dem_tiles'
        data_dir : str | None
            数据目录。river_dataset 模式默认 terrain/data/river_dataset/{split}，
            dem_tiles 模式默认 terrain/data/dem_tiles/。
        size : int  切片尺寸
        augment : bool  是否数据增强
        target : 'mask' | 'smooth'  目标 mask 生成策略
        threshold_quantile : float  D8 流量累积分位阈值（仅 dem_tiles 模式）
        """
        from .config import DEM_TILES_DIR, DATA_DIR

        self.mode = mode
        self.size = size
        self.augment = augment
        self.target_strategy = target
        self.threshold_quantile = threshold_quantile

        if mode == "river_dataset":
            split_dir = Path(data_dir) if data_dir else DATA_DIR / "river_dataset" / "train"
            self.files = sorted([
                str(f) for f in split_dir.iterdir() if f.suffix == ".npy"
            ])
            if not self.files:
                raise RuntimeError(f"river_dataset 模式：{split_dir} 下无 .npy 文件")
        elif mode == "dem_tiles":
            tile_dir = Path(data_dir) if data_dir else DEM_TILES_DIR
            self.files = sorted([
                str(f) for f in tile_dir.iterdir()
                if f.suffix in (".npy", ".npz")
            ])
            if not self.files:
                raise RuntimeError(f"dem_tiles 模式：{tile_dir} 下无切片文件")
        else:
            raise ValueError(f"未知 mode: {mode}，支持 'river_dataset' / 'dem_tiles'")

        print(f"  RiverDiffusionDataset: {len(self.files)} 样本 "
              f"({mode}, target={target})")

    def __len__(self):
        """返回样本数量。"""
        return len(self.files)

    def _load_river_dataset_sample(self, path: str):
        """从 river_dataset 加载 (DEM, mask) 对。"""
        from .config import elev_to_norm
        sample = np.load(path).astype(np.float32)
        dem = sample[..., 0]      # (H, W) 米
        mask = sample[..., 1]     # (H, W) 0/1
        land = dem > 0

        # 统一到 size×size
        if dem.shape[0] != self.size or dem.shape[1] != self.size:
            from scipy.ndimage import zoom
            h, w = dem.shape
            dem = zoom(dem, (self.size / h, self.size / w), order=1)
            mask = zoom(mask, (self.size / h, self.size / w), order=0)
            land = dem > 0

        dem_norm = elev_to_norm(dem).astype(np.float32)
        return dem_norm, mask.astype(np.float32), land

    def _load_dem_tile_sample(self, path: str):
        """从 dem_tiles 加载 DEM 并在线计算 D8 mask。"""
        from .config import elev_to_norm

        if path.endswith(".npz"):
            data = np.load(path)
            dem = data["dem"].astype(np.float32)
        else:
            dem = np.load(path).astype(np.float32)

        land = dem > 0

        # 裁剪到 size×size
        H, W = dem.shape
        if H >= self.size and W >= self.size:
            h0 = np.random.randint(0, H - self.size + 1)
            w0 = np.random.randint(0, W - self.size + 1)
            dem = dem[h0:h0 + self.size, w0:w0 + self.size]
        else:
            from scipy.ndimage import zoom
            dem = zoom(dem, (self.size / H, self.size / W), order=1)

        land = dem > 0

        # 在线计算 D8 mask
        from .physics_dataset import extract_river_label
        mask = extract_river_label(dem, self.threshold_quantile)

        dem_norm = elev_to_norm(dem).astype(np.float32)
        return dem_norm, mask, land

    def __getitem__(self, idx: int):
        """加载第 idx 个样本，生成目标 mask 并同步增强，返回 (条件, 目标) 张量对。"""
        if self.mode == "river_dataset":
            dem_norm, mask, land = self._load_river_dataset_sample(self.files[idx])
        else:
            dem_norm, mask, land = self._load_dem_tile_sample(self.files[idx])

        # 生成目标 mask
        if self.target_strategy == "smooth" and mask.any():
            target_mask = _morphological_smooth(mask > 0.5, land).astype(np.float32)
        else:
            target_mask = mask

        # 数据增强（DEM / mask / target 同步）
        if self.augment:
            if np.random.rand() < 0.5:
                dem_norm = dem_norm[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
                target_mask = target_mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                dem_norm = dem_norm[::-1, :].copy()
                mask = mask[::-1, :].copy()
                target_mask = target_mask[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                dem_norm = np.rot90(dem_norm, k).copy()
                mask = np.rot90(mask, k).copy()
                target_mask = np.rot90(target_mask, k).copy()

        # 组装 condition: [DEM_norm, D8_mask] → (2, H, W)
        cond = np.stack([dem_norm, mask], axis=0)     # (2, H, W)
        target = target_mask[np.newaxis, ...]          # (1, H, W)

        cond = torch.from_numpy(cond).float()
        target = torch.from_numpy(target).float()
        return cond, target


# ==================== 6. 全图推理 ====================

@torch.no_grad()
def river_diffusion_inference(
    dem: np.ndarray,
    d8_mask: np.ndarray,
    model: ConditionalRiverDiffusion | None = None,
    model_path: str | None = None,
    tile_size: int = 256,
    overlap: int = 32,
    n_infer: int = 20,
    t_start: int = 100,
    eta: float = 0.0,
    threshold: float = 0.5,
    use_ema: bool = True,
    device: torch.device | None = None,
    progress: bool = True,
) -> np.ndarray:
    """全图河流 Diffusion 精修推理（SDEdit 策略）。

    将大图（如 2048×2048）切分为重叠 tile，逐 tile 对 D8 mask 加噪后
    做 SDEdit 去噪，保留 D8 拓扑结构同时平滑不自然模式，
    最后用重叠区加权平均拼接。

    Parameters
    ----------
    dem : (H, W) float32  高程（米）
    d8_mask : (H, W) bool  D8 物理河流 mask
    model : ConditionalRiverDiffusion | None
        已加载的模型。若 None 则从 model_path 加载。
    model_path : str | None
        权重路径，默认 CKPT_DIR / "river_diffusion.pt"
    tile_size : int
        tile 尺寸（必须 256，模型训练分辨率）
    overlap : int
        tile 重叠像素（拼接时加权平均）
    n_infer : int
        SDEdit 去噪步数
    t_start : int
        SDEdit 噪声起始步（越小越接近 D8 原图，推荐 50-200）
    eta : float
        DDIM 随机性（0=确定性）
    threshold : float
        河流 mask 二值化阈值
    use_ema : bool
        是否使用 EMA 权重
    device : torch.device | None
    progress : bool

    Returns
    -------
    mask : (H, W) bool  精修后的河流 mask
    """
    from .config import DEVICE as _DEVICE, CKPT_DIR, elev_to_norm
    from .diffusion import DDPM

    if device is None:
        device = _DEVICE

    # ---- 加载模型 ----
    if model is None:
        model = ConditionalRiverDiffusion(cond_ch=2, base=64, time_dim=128).to(device)
        if model_path is None:
            model_path = str(CKPT_DIR / "river_diffusion.pt")
        ck = torch.load(model_path, map_location=device, weights_only=True)
        if use_ema and "ema" in ck:
            model.load_state_dict(ck["ema"])
        else:
            model.load_state_dict(ck["model"])
    model.eval()

    ddpm = DDPM(n_steps=1000, device=device)

    h, w = dem.shape
    land = dem > 0

    # ---- 准备归一化 DEM 和 D8 mask ----
    dem_norm = elev_to_norm(dem.astype(np.float32))
    d8_float = d8_mask.astype(np.float32)

    # ---- tile 坐标列表 ----
    stride = tile_size - overlap
    y_starts = list(range(0, max(h - tile_size + 1, 1), stride))
    if y_starts[-1] + tile_size < h:
        y_starts.append(max(0, h - tile_size))
    x_starts = list(range(0, max(w - tile_size + 1, 1), stride))
    if x_starts[-1] + tile_size < w:
        x_starts.append(max(0, w - tile_size))

    # ---- 累加器和权重图 ----
    prob_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    # 中心加权：tile 中心权重高，边缘低
    tile_weight = _make_tile_weight(tile_size, overlap)

    n_tiles = len(y_starts) * len(x_starts)
    pbar = range(n_tiles)
    if progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(pbar, desc="river_diffusion")
        except ImportError:
            pass

    idx = 0
    for ys in y_starts:
        for xs in x_starts:
            # ---- 提取 tile ----
            ye, xe = min(ys + tile_size, h), min(xs + tile_size, w)
            ph, pw = ye - ys, xe - xs

            dem_tile = dem_norm[ys:ye, xs:xe]
            d8_tile = d8_float[ys:ye, xs:xe]

            # pad 到 tile_size×tile_size
            if ph < tile_size or pw < tile_size:
                dem_tile = np.pad(dem_tile,
                                  ((0, tile_size - ph), (0, tile_size - pw)),
                                  mode="edge")
                d8_tile = np.pad(d8_tile,
                                 ((0, tile_size - ph), (0, tile_size - pw)),
                                 mode="edge")

            # ---- 条件 + SDEdit 精修采样 ----
            cond = torch.from_numpy(
                np.stack([dem_tile, d8_tile], axis=0)[None]
            ).float().to(device)
            # SDEdit: 从 D8 mask 加噪 → 去噪（保留 D8 结构，平滑锯齿）
            x0_tile = torch.from_numpy(d8_tile[None, None]).float().to(device)
            x = ddpm.sdedit_conditional(
                model, x0_tile, cond, device,
                t_start=t_start, n_infer=n_infer, eta=eta, progress=False)
            prob_tile = torch.sigmoid(x)[0, 0].cpu().numpy()

            # ---- 加权拼接 ----
            tw = tile_weight[:ph, :pw]
            prob_sum[ys:ye, xs:xe] += prob_tile[:ph, :pw] * tw
            weight_sum[ys:ye, xs:xe] += tw

            idx += 1
            if progress and hasattr(pbar, 'update'):
                pbar.update(1)

    # ---- 归一化 + 二值化 ----
    prob_avg = prob_sum / np.maximum(weight_sum, 1e-8)
    mask = (prob_avg > threshold) & land

    return mask


def _make_tile_weight(size: int, overlap: int) -> np.ndarray:
    """构造 tile 权重图：中心 1.0，边缘 0.0，overlap 区线性渐变。"""
    w = np.ones((size, size), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        # 四边渐变
        for i in range(overlap):
            v = ramp[i]
            w[i, :] = np.minimum(w[i, :], v)
            w[size - 1 - i, :] = np.minimum(w[size - 1 - i, :], v)
            w[:, i] = np.minimum(w[:, i], v)
            w[:, size - 1 - i] = np.minimum(w[:, size - 1 - i], v)
    return w


def run_river_diffusion_eval(
    dem_path: str = "terrain/output/world_dem.npy",
    model_path: str | None = None,
    output_dir: str = "terrain/output",
    tile_size: int = 256,
    overlap: int = 32,
    n_infer: int = 50,
    threshold: float = 0.5,
):
    """完整评估流程：D8 物理 → Diffusion 精修 → 形态学对比。

    1. 加载 DEM
    2. 用 hybrid_river_extraction 提取 D8 mask（条件）
    3. 用 river_diffusion_inference 做条件 Diffusion 精修
    4. 用 eval_morphology 对比 D8 vs Diffusion
    5. 保存结果和对比图

    Parameters
    ----------
    dem_path : str       DEM 文件路径
    model_path : str    Diffusion 权重路径
    output_dir : str    输出目录
    tile_size : int     tile 尺寸
    overlap : int       tile 重叠
    n_infer : int       DDIM 采样步数
    threshold : float  二值化阈值
    """
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .river_net import hybrid_river_extraction

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载 DEM ----
    dem = np.load(dem_path).astype(np.float32)
    print(f"DEM: {dem.shape} 范围 [{dem.min():.0f}, {dem.max():.0f}] m")

    # ---- 2. D8 物理提取 ----
    print("\n[1/3] D8 物理河网提取...")
    d8_mask, d8_flow = hybrid_river_extraction(
        dem, dl_enhance=False, dem_smooth_sigma=0.3)
    print(f"  D8 mask: {d8_mask.sum()} 像素 ({d8_mask.mean()*100:.2f}%)")

    # ---- 3. Diffusion 精修 ----
    print("\n[2/3] Diffusion 条件精修...")
    diff_mask = river_diffusion_inference(
        dem, d8_mask, model_path=model_path,
        tile_size=tile_size, overlap=overlap,
        n_infer=n_infer, threshold=threshold)
    print(f"  Diffusion mask: {diff_mask.sum()} 像素 ({diff_mask.mean()*100:.2f}%)")

    # ---- 4. 形态学评估 ----
    print("\n[3/3] 形态学评估...")
    from scipy import ndimage as ndi

    def _eval(mask, dem, name):
        """对单个 mask 计算形态学指标并打印，返回指标字典。"""
        land = dem > 0
        river = mask & land
        if not river.any():
            print(f"  {name}: 无河流")
            return {}
        n4 = ndi.convolve(river.astype(np.int32),
                          np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='constant')
        n8 = ndi.convolve(river.astype(np.int32),
                          np.array([[1,1,1],[1,0,1],[1,1,1]]), mode='constant')
        junctions = river & (n4 >= 3)
        braids = river & (n4 >= 4)
        diag = river & (n8 >= 3) & (n4 < 3)
        ry, rx = np.gradient(river.astype(float))
        angle = np.arctan2(ry, rx) * 180 / np.pi
        bins = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        hist, _ = np.histogram(angle[river], bins=bins)
        p = hist / max(hist.sum(), 1)
        entropy = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        labeled, n = ndi.label(river, structure=np.ones((3,3), dtype=int))
        sizes = ndi.sum(river, labeled, range(1, n+1))
        big = (sizes >= 1000).sum()
        result = {
            "density": river.mean() * 100,
            "junction": junctions.sum() / max(river.sum(), 1) * 100,
            "braid": braids.sum() / max(river.sum(), 1) * 100,
            "diag": diag.sum() / max(river.sum(), 1) * 100,
            "entropy": entropy,
            "components": n,
            "max_size": int(sizes.max()),
            "big_rivers": big,
        }
        print(f"  {name:<20} 密度={result['density']:5.2f}%  "
              f"汇合={result['junction']:5.1f}%  辫状={result['braid']:4.2f}%  "
              f"对角={result['diag']:4.1f}%  熵={result['entropy']:.2f}  "
              f"分量={n:>5,}  最大={int(sizes.max()):>7,}px  ≥1K={big}")
        return result

    print("\n--- 形态学对比 ---")
    r_d8 = _eval(d8_mask, dem, "D8 物理")
    r_diff = _eval(diff_mask, dem, "Diffusion 精修")

    # ---- 5. 保存结果 ----
    np.save(output / "river_mask_diffusion.npy", diff_mask)
    print(f"\n保存: {output / 'river_mask_diffusion.npy'}")

    # ---- 6. 对比图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM")
    axes[0].axis("off")

    axes[1].imshow(dem, cmap="terrain", alpha=0.7)
    d8_overlay = np.ma.masked_where(~d8_mask, np.ones_like(dem))
    axes[1].imshow(d8_overlay, cmap="Blues", alpha=0.8)
    axes[1].set_title(f"D8 物理 (辫状={r_d8.get('braid',0):.2f}%, 熵={r_d8.get('entropy',0):.2f})")
    axes[1].axis("off")

    axes[2].imshow(dem, cmap="terrain", alpha=0.7)
    diff_overlay = np.ma.masked_where(~diff_mask, np.ones_like(dem))
    axes[2].imshow(diff_overlay, cmap="Blues", alpha=0.8)
    axes[2].set_title(f"Diffusion (辫状={r_diff.get('braid',0):.2f}%, 熵={r_diff.get('entropy',0):.2f})")
    axes[2].axis("off")

    plt.tight_layout()
    fig_path = output / "river_diffusion_comparison.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"保存: {fig_path}")

    return {"d8": r_d8, "diffusion": r_diff}


# ==================== 7. 测试 ====================

if __name__ == "__main__":
    from .config import DEVICE
    from .diffusion import DDPM

    print(f"Device: {DEVICE}")

    # ---- 模型测试 ----
    model = ConditionalRiverDiffusion(cond_ch=2, base=64, time_dim=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.1f}M")

    B, H, W = 2, 256, 256
    x_t = torch.randn(B, 1, H, W, device=DEVICE)
    t = torch.randint(0, 1000, (B,), device=DEVICE)
    cond = torch.randn(B, 2, H, W, device=DEVICE)

    out = model(x_t, t, cond)
    print(f"模型前向: x_t={x_t.shape} → eps={out.shape}")
    assert out.shape == (B, 1, H, W)

    # ---- 损失测试 ----
    ddpm = DDPM(n_steps=1000, device=DEVICE)
    x0 = torch.rand(B, 1, H, W, device=DEVICE)
    t = torch.randint(0, 1000, (B,), device=DEVICE)
    loss, parts = river_diffusion_loss(model, x0, t, cond, ddpm)
    print(f"损失: {parts}")
    loss.backward()
    grad_ok = all(p.grad is not None and not torch.isnan(p.grad).any()
                 for p in model.parameters() if p.requires_grad)
    print(f"梯度检查: {'OK' if grad_ok else 'FAIL'}")

    # ---- 数据集测试 ----
    print("\n---- 数据集测试 ----")
    from torch.utils.data import DataLoader

    # river_dataset 模式（smooth 目标）
    ds_smooth = RiverDiffusionDataset(mode="river_dataset", target="smooth")
    cond_s, target_s = ds_smooth[0]
    print(f"river_dataset smooth: cond={cond_s.shape} target={target_s.shape}")
    print(f"  cond 范围: [{cond_s.min():.2f}, {cond_s.max():.2f}]")
    print(f"  target 河流像素: {target_s.sum().item():.0f} / {target_s.numel()}")

    # river_dataset 模式（mask 目标）
    ds_mask = RiverDiffusionDataset(mode="river_dataset", target="mask")
    cond_m, target_m = ds_mask[0]
    print(f"river_dataset mask: cond={cond_m.shape} target={target_m.shape}")
    print(f"  target 河流像素: {target_m.sum().item():.0f} / {target_m.numel()}")

    # smooth vs mask 差异
    diff = (target_s != target_m).sum().item()
    print(f"  smooth vs mask 差异像素: {diff}")

    # DataLoader 批量加载测试
    loader = DataLoader(ds_smooth, batch_size=4, shuffle=True, num_workers=0)
    batch_cond, batch_target = next(iter(loader))
    print(f"DataLoader: cond={batch_cond.shape} target={batch_target.shape}")

    # 端到端：数据 → 模型 → 损失
    batch_cond = batch_cond.to(DEVICE)
    batch_target = batch_target.to(DEVICE)
    t_batch = torch.randint(0, 1000, (4,), device=DEVICE)
    loss, parts = river_diffusion_loss(
        model, batch_target, t_batch, batch_cond, ddpm,
        lambda_braid=0.1, lambda_conn=0.05)
    print(f"端到端损失: {parts}")
    loss.backward()
    print("端到端测试通过")

    # ---- 全图推理测试 ----
    print("\n---- 全图推理测试 ----")
    # 用小尺寸 DEM 模拟（512×512）+ 随机 D8 mask
    dem_small = np.random.uniform(0, 1000, (512, 512)).astype(np.float32)
    dem_small[100:400, 100:400] += 500  # 模拟高地
    d8_small = (dem_small > 200) & (np.random.rand(512, 512) < 0.03)

    # 用未训练的模型做推理（验证 pipeline 不崩溃）
    mask = river_diffusion_inference(
        dem_small, d8_small, model=model,
        tile_size=256, overlap=32, n_infer=10, threshold=0.5,
        progress=False)
    print(f"推理结果: shape={mask.shape}  河流像素={mask.sum()}")
    assert mask.shape == dem_small.shape
    assert mask.dtype == bool

    # tile 权重图测试
    tw = _make_tile_weight(256, 32)
    print(f"tile 权重: shape={tw.shape}  中心={tw[128,128]:.2f}  边角={tw[0,0]:.2f}")

    print("所有测试通过")
