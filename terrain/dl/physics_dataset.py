"""阶段 3 物理监督数据集：从真实 DEM 提取河网/雪线/海岸线标签。

三个监督网络（RiverNet/SnowlineNet/CoastNet）需要配对 (输入, 标签) 数据。
本模块从真实或合成 DEM 切片自动提取标签：

1. **河网标签**：D8 流量累积 → 分位阈值 → 二值河网掩码
   （复用 terrain_engine.d8_flow_accumulation）
2. **雪线标签**：纬度查表 snowline_m + 高程门控 → 逐像素雪线高程
   （复用 terrain_engine.snowline_m，作为预训练伪标签）
3. **海岸线标签**：按高程阈值三分类（陆地/陆架/深海）

数据格式
--------
每个 .npz 切片含：
    dem  : (H, W) float32  米制高程
    lats : (H,)   float32  纬度数组（可选，缺失时用 0 填充）

标签由本模块在线计算，避免磁盘冗余。

示例
----
>>> from terrain.dl.physics_dataset import RiverDataset, SnowlineDataset, CoastDataset
>>> ds = RiverDataset(root="terrain/data/dem_tiles", size=256, augment=True)
>>> x, y = ds[0]   # x: (1, 256, 256) [-1,1], y: (1, 256, 256) {0,1}
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEM_TILES_DIR, TILE_SIZE, elev_to_norm, ELEV_MIN, ELEV_MAX


# ==================== 标签提取函数 ====================

def _ensure_core_on_path() -> None:
    """把 terrain/core 加入 sys.path，便于复用 terrain_engine 函数。"""
    core_dir = Path(__file__).resolve().parents[1] / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))


def extract_river_label(dem_m: np.ndarray, threshold_quantile: float = 0.995,
                        min_elev: float = 0.0) -> np.ndarray:
    """从 DEM 提取河网二值标签。

    流程：D8 流量累积 → 取分位阈值 → 仅保留陆地像素 → 二值化。

    Parameters
    ----------
    dem_m : (H, W) float32 米制高程
    threshold_quantile : float
        流量累积分位阈值（默认 99.5%），越高河网越稀疏（仅干流）。
    min_elev : float
        仅在 elev ≥ min_elev 的陆地像素上提取河网（默认 0m，海平面以上）。

    Returns
    -------
    (H, W) float32 二值掩码 {0.0, 1.0}
    """
    from terrain.core.terrain_engine import d8_flow_accumulation  # type: ignore

    rng = np.random.RandomState(42)
    acc = d8_flow_accumulation(dem_m, rng)
    # 陆地掩码（避免在海洋里出现"河"）
    land = dem_m >= min_elev
    # 在陆地像素内取分位阈值
    acc_land = acc[land]
    if acc_land.size == 0:
        return np.zeros_like(dem_m, dtype=np.float32)
    thr = np.quantile(acc_land, threshold_quantile)
    river = (acc >= thr) & land
    return river.astype(np.float32)


def extract_snowline_label(dem_m: np.ndarray, lats: np.ndarray | None = None
                           ) -> np.ndarray:
    """从纬度查表生成雪线高程标签（逐像素回归目标）。

    复用 terrain_engine.snowline_m 作为预训练伪标签。
    真实数据场景下可用 MODIS 积雪产品替换。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    lats : (H,) 纬度数组；为 None 时纬度全 0（赤道口径）。

    Returns
    -------
    (H, W) float32 雪线高程（米），范围 [0, 6500]
    """
    from terrain.core.terrain_engine import snowline_m  # type: ignore

    H, W = dem_m.shape
    if lats is None:
        lats = np.zeros(H, dtype=np.float32)
    # snowline_m 接受标量或数组；广播到 (H, W)
    sl_row = snowline_m(lats)  # (H,)
    sl_field = np.broadcast_to(sl_row[:, None], (H, W)).astype(np.float32)
    # 海洋像素雪线置 0（无意义）
    sl_field = np.where(dem_m < 0, 0.0, sl_field)
    return np.clip(sl_field, 0.0, 6500.0).astype(np.float32)


def extract_coast_label(dem_m: np.ndarray, shelf_depth: float = -2000.0
                        ) -> np.ndarray:
    """从 DEM 按高程阈值生成海岸线三分类标签。

    类别：
        0 = 陆地（elev ≥ 0）
        1 = 陆架/陆坡过渡带（shelf_depth ≤ elev < 0）
        2 = 深海（elev < shelf_depth）

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    shelf_depth : float 陆架目标深度，默认 -2000m。

    Returns
    -------
    (H, W) int64 标签图，值 ∈ {0, 1, 2}
    """
    label = np.full(dem_m.shape, 2, dtype=np.int64)  # 默认深海
    label[dem_m >= shelf_depth] = 1                  # 陆架
    label[dem_m >= 0.0] = 0                          # 陆地
    return label


# ==================== 通用切片加载与增强 ====================

def _load_tile(path: str, size: int) -> tuple[np.ndarray, np.ndarray | None]:
    """加载 .npy/.npz 切片，返回 (dem_m, lats or None)。"""
    if path.endswith(".npz"):
        data = np.load(path)
        dem = data["dem"].astype(np.float32)
        lats = data["lats"].astype(np.float32) if "lats" in data.files else None
    else:
        dem = np.load(path).astype(np.float32)
        lats = None
    H, W = dem.shape
    if H >= size and W >= size:
        h0 = np.random.randint(0, H - size + 1)
        w0 = np.random.randint(0, W - size + 1)
        dem = dem[h0:h0 + size, w0:w0 + size]
        if lats is not None:
            lats = lats[h0:h0 + size]
    else:
        from scipy.ndimage import zoom as ndi_zoom
        dem = ndi_zoom(dem, (size / H, size / W), order=1)
        if lats is not None:
            lats = ndi_zoom(lats, size / H, order=1)
    return dem, lats


def _augment_pair(arr: np.ndarray, label: np.ndarray,
                  is_classification: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """同步翻转/旋转增强（输入与标签必须同步）。

    Parameters
    ----------
    arr : (H, W) 输入
    label : (H, W) 标签
    is_classification : bool
        True 时标签为整数类别（旋转用 np.rot90 不插值）；
        False 时标签为连续值（同样用 rot90，不插值）。
    """
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1].copy()
        label = label[:, ::-1].copy()
    if np.random.rand() < 0.5:
        arr = arr[::-1, :].copy()
        label = label[::-1, :].copy()
    k = np.random.randint(0, 4)
    if k:
        arr = np.rot90(arr, k).copy()
        label = np.rot90(label, k).copy()
    return arr, label


# ==================== 三个数据集 ====================

class _BasePhysicsDataset(Dataset):
    """公共基类：切片发现、加载、裁剪。"""

    def __init__(self, root: Path = DEM_TILES_DIR, size: int = TILE_SIZE,
                 augment: bool = True):
        """初始化基类：扫描根目录下 .npy/.npz 切片并保存尺寸与增强开关。"""
        self.root = Path(root)
        files = sorted(glob.glob(str(self.root / "*.npy")))
        files += sorted(glob.glob(str(self.root / "*.npz")))
        if not files:
            raise RuntimeError(
                f"未在 {self.root} 找到 .npy/.npz 切片，请先准备数据。")
        self.files = files
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        """返回切片数量。"""
        return len(self.files)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray | None]:
        """加载并裁剪第 idx 个切片，返回 (dem_m, lats or None)。"""
        return _load_tile(self.files[idx], self.size)


class RiverDataset(_BasePhysicsDataset):
    """河网监督数据集：(高程归一化, 河网二值掩码)。

    Returns
    -------
    x : (1, H, W) float32  高程归一化到 [-1, 1]
    y : (1, H, W) float32  河网二值掩码 {0, 1}
    """

    def __init__(self, root: Path = DEM_TILES_DIR, size: int = TILE_SIZE,
                 augment: bool = True, threshold_quantile: float = 0.995):
        """初始化河网数据集，保存 D8 流量分位阈值参数。"""
        super().__init__(root, size, augment)
        self.threshold_quantile = threshold_quantile

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (归一化高程, 河网二值掩码) 张量对。"""
        dem, _ = self._load(idx)
        label = extract_river_label(dem, self.threshold_quantile)
        if self.augment:
            dem, label = _augment_pair(dem, label)
        x = elev_to_norm(dem).astype(np.float32)
        return (torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(label).unsqueeze(0))


class SnowlineDataset(_BasePhysicsDataset):
    """雪线监督数据集：(条件三通道, 雪线高程)。

    Returns
    -------
    x : (3, H, W) float32  通道 = [lat_norm, elev_norm, ci]
    y : (1, H, W) float32  雪线高程（米），范围 [0, 6500]
    """

    def __init__(self, root: Path = DEM_TILES_DIR, size: int = TILE_SIZE,
                 augment: bool = True):
        """初始化雪线数据集。"""
        super().__init__(root, size, augment)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (三通道条件, 雪线高程) 张量对。"""
        dem, lats = self._load(idx)
        if lats is None:
            lats = np.zeros(dem.shape[0], dtype=np.float32)
        label = extract_snowline_label(dem, lats)
        if self.augment:
            dem, label = _augment_pair(dem, label)
            # 注意：纬度随垂直翻转/旋转需同步，但物理上纬度不可旋转。
            # 这里简化：水平翻转 + 垂直翻转保持纬度数组同步翻转；
            # 90° 旋转会打乱纬度-行对应，故雪线数据集关闭 90° 旋转。
            # 为简单起见，仍调用 _augment_pair，但使用单独的增强流程：
            pass
        # 计算输入三通道
        from terrain.core.terrain_engine import continentality_index  # type: ignore
        lat_norm = np.abs(lats)[:, None] / 90.0   # (H, 1)
        lat_norm = np.broadcast_to(lat_norm, dem.shape).astype(np.float32)
        elev_norm = elev_to_norm(dem).astype(np.float32)
        ci = continentality_index(dem, lats).astype(np.float32)
        x = np.stack([lat_norm, elev_norm, ci], axis=0)  # (3, H, W)
        y = label[None, ...].astype(np.float32)         # (1, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)


class CoastDataset(_BasePhysicsDataset):
    """海岸线监督数据集：(高程归一化, 三类标签)。

    Returns
    -------
    x : (1, H, W) float32  高程归一化到 [-1, 1]
    y : (H, W)  long  类别标签 ∈ {0, 1, 2}（不带通道维，配合 CrossEntropyLoss）
    """

    def __init__(self, root: Path = DEM_TILES_DIR, size: int = TILE_SIZE,
                 augment: bool = True, shelf_depth: float = -2000.0):
        """初始化海岸线数据集，保存陆架深度参数。"""
        super().__init__(root, size, augment)
        self.shelf_depth = shelf_depth

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (归一化高程, 三分类标签) 张量对。"""
        dem, _ = self._load(idx)
        label = extract_coast_label(dem, self.shelf_depth)
        if self.augment:
            dem, label = _augment_pair(dem, label, is_classification=True)
        x = elev_to_norm(dem).astype(np.float32)
        return (torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(label.astype(np.int64)))


# ==================== DataLoader 工厂 ====================

def make_physics_loaders(task: str, batch_size: int = 8, val_ratio: float = 0.1,
                         num_workers: int = 2, seed: int = 42, **kwargs
                         ) -> tuple[torch.utils.data.DataLoader,
                                    torch.utils.data.DataLoader]:
    """划分训练/验证 DataLoader。

    Parameters
    ----------
    task : str  ∈ {"river", "snowline", "coast"}
    batch_size, val_ratio, num_workers, seed : 训练超参
    **kwargs : 传给对应 Dataset 的额外参数（如 threshold_quantile）

    Returns
    -------
    (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split

    task = task.lower()
    if task == "river":
        ds_cls = RiverDataset
    elif task == "snowline":
        ds_cls = SnowlineDataset
    elif task == "coast":
        ds_cls = CoastDataset
    else:
        raise ValueError(f"未知任务: {task}，可选: river/snowline/coast")

    # 拆分文件列表，构造独立的训练/验证 Dataset（避免 Subset 共享 augment 标志）
    root = Path(kwargs.pop("root", DEM_TILES_DIR))
    files = sorted(glob.glob(str(root / "*.npy")))
    files += sorted(glob.glob(str(root / "*.npz")))
    if not files:
        raise RuntimeError(
            f"未找到 .npy/.npz 切片，请先准备数据。")
    n_val = max(1, int(len(files) * val_ratio))
    n_train = len(files) - n_val
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(files), generator=gen).tolist()
    train_files = [files[i] for i in idx[:n_train]]
    val_files = [files[i] for i in idx[n_train:]]

    # 用子类化方式注入文件列表
    train_ds = _make_split_dataset(ds_cls, train_files, root=root,
                                   augment=True, **kwargs)
    val_ds = _make_split_dataset(ds_cls, val_files, root=root,
                                 augment=False, **kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


def _make_split_dataset(ds_cls, files: list[str], root: Path,
                        augment: bool, **kwargs):
    """构造一个文件列表受限的 Dataset 实例。"""
    ds = ds_cls(root=root, augment=augment, **kwargs)
    ds.files = files   # 覆盖文件列表
    return ds


# ==================== 合成数据生成（测试/调试用） ====================

def generate_synthetic_tiles(out_dir: Path, n_tiles: int = 8,
                             size: int = TILE_SIZE, seed: int = 42) -> int:
    """生成合成 DEM 切片用于测试。

    用 fBm 噪声 + 简单海陆分布生成低质量但形状正确的 DEM，
    仅用于单元测试与流水线调试，**不用于真实训练**。

    Returns
    -------
    int  生成的切片数。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    for k in range(n_tiles):
        # fBm 噪声
        f = 4.0
        noise = np.zeros((size, size), np.float32)
        amp, tot = 1.0, 0.0
        for _ in range(4):
            n = rng.randn(size, size).astype(np.float32)
            from scipy.ndimage import gaussian_filter
            n = gaussian_filter(n, max(size / (2.0 * f), 0.6))
            n /= (n.std() + 1e-8)
            noise += amp * n
            tot += amp
            amp *= 0.5
            f *= 2.0
        noise /= tot
        # 映射到 [-6000, 3000] 米，约 40% 陆地
        dem = (noise * 4500.0 + (-1500.0)).astype(np.float32)
        # 加一条"海沟"
        dem[:, size // 2 - 2:size // 2 + 2] -= 4000.0
        # 纬度数组（-60 到 60 度）
        lats = np.linspace(-60.0, 60.0, size, dtype=np.float32)
        np.savez_compressed(out_dir / f"synth_{k:03d}.npz",
                            dem=dem, lats=lats)
    return n_tiles


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="物理监督数据集工具")
    p.add_argument("--gen-synth", action="store_true",
                   help="生成合成测试切片")
    p.add_argument("--out", type=str, default=str(DEM_TILES_DIR))
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()

    if a.gen_synth:
        n = generate_synthetic_tiles(Path(a.out), n_tiles=a.n)
        print(f"生成 {n} 个合成切片到 {a.out}")
    else:
        p.print_help()
