"""DEM 数据集：无标签切片加载 + 数据增强。

支持
----
- 从 ``terrain/data/dem_tiles/`` 加载 .npy 切片
- 随机翻转 / 90° 旋转（地形近似各向同性，安全增强）
- 高程归一化到 [-1, 1]（见 ``config.elev_to_norm``）
- 可选裁剪 / 放大到目标尺寸
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEM_TILES_DIR, TILE_SIZE, elev_to_norm


class DEMTileDataset(Dataset):
    """无标签 DEM 切片数据集。

    Parameters
    ----------
    root : Path
        .npy 切片根目录，默认 ``terrain/data/dem_tiles/``。
    size : int
        输出尺寸，默认 256。
    augment : bool
        是否启用翻转/旋转增强。
    """

    def __init__(self, root: Path = DEM_TILES_DIR, size: int = TILE_SIZE,
                 augment: bool = True):
        """初始化数据集：扫描根目录下全部 .npy 切片。"""
        self.root = Path(root)
        self.files = sorted(glob.glob(str(self.root / "*.npy")))
        if not self.files:
            raise RuntimeError(
                f"未在 {self.root} 找到 .npy 切片，请先运行 scripts/download_dem.py")
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        """返回数据集样本数。"""
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """加载第 idx 个切片：裁剪/放大、增强、归一化后返回 (1, H, W) 张量。"""
        dem = np.load(self.files[idx]).astype(np.float32)  # (H, W) 米

        # 裁剪/放大到目标尺寸
        H, W = dem.shape
        if H >= self.size and W >= self.size:
            h0 = np.random.randint(0, H - self.size + 1)
            w0 = np.random.randint(0, W - self.size + 1)
            dem = dem[h0:h0 + self.size, w0:w0 + self.size]
        else:
            from scipy.ndimage import zoom as ndi_zoom
            dem = ndi_zoom(dem, (self.size / H, self.size / W), order=1)

        # 增强
        if self.augment:
            if np.random.rand() < 0.5:
                dem = dem[:, ::-1].copy()       # 水平翻转
            if np.random.rand() < 0.5:
                dem = dem[::-1, :].copy()       # 垂直翻转
            k = np.random.randint(0, 4)
            if k:
                dem = np.rot90(dem, k).copy()

        x = elev_to_norm(dem)                    # [-1, 1]
        return torch.from_numpy(x).unsqueeze(0).float()   # (1, H, W)


def make_loaders(batch_size: int = 8, val_ratio: float = 0.05,
                 num_workers: int = 2, seed: int = 42):
    """划分训练/验证 DataLoader。

    Returns
    -------
    (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split
    ds = DEMTileDataset(augment=True)
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader
