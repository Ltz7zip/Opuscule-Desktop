"""配对数据集：板块骨架条件 → 目标 DEM 切片。

半监督学习需要两种数据：
1. **配对样本**（少量）：(12 通道条件, 目标 DEM 切片)，来自现有引擎生成或地球真实 DEM 反推
2. **无标签样本**（大量）：仅目标 DEM 切片，来自 terrain/data/dem_tiles/

本模块管理配对样本。配对样本来源策略：
- **策略 A（推荐）**：用现有 GAN 引擎批量生成 (条件, DEM) 对作为伪标签预训练
- **策略 B**：从地球真实 DEM 反推板块骨架（用 plate_sim 反向拟合）
- **策略 C**：人工/半自动标注少量切片

存储格式：每个样本一个 .npz，含 ``cond`` (12, 256, 256) 和 ``dem`` (256, 256)。
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import PAIRED_DIR, TILE_SIZE, elev_to_norm
from scipy.ndimage import zoom as ndi_zoom


class PairedDEMDataset(Dataset):
    """配对数据集：(12 通道条件, 目标 DEM)。

    数据格式
    --------
    每个 .npz 文件含：
        cond : (12, 256, 256) float32  — 12 通道条件（米制统计，未归一化）
        dem  : (256, 256) float32       — 目标高程（米）

    Parameters
    ----------
    root : Path
        配对样本目录，默认 ``terrain/data/paired/``。
    augment : bool
        是否启用同步翻转/旋转（条件与 DEM 必须同步变换）。
    """

    def __init__(self, root: Path = PAIRED_DIR, augment: bool = True):
        """初始化配对数据集：扫描根目录下全部 .npz 配对样本。"""
        self.root = Path(root)
        self.files = sorted(glob.glob(str(self.root / "*.npz")))
        if not self.files:
            raise RuntimeError(
                f"未在 {self.root} 找到 .npz 配对样本。"
                f"请先用 scripts/generate_paired_data.py 生成。")
        self.augment = augment

    def __len__(self) -> int:
        """返回配对样本数。"""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """加载第 idx 个配对样本，同步增强后返回 (条件, DEM) 张量对。"""
        data = np.load(self.files[idx])
        cond = data["cond"].astype(np.float32)    # (12, H, W)
        dem = data["dem"].astype(np.float32)      # (H, W)

        # ---- 兼容：若条件空间尺寸 != TILE_SIZE，按通道双线性缩放到 TILE_SIZE ----
        # 旧版 generate_paired_from_engine 曾把条件保存为 (12, 32, 32)，
        # 而 ConditionEncoder 期望 (12, TILE_SIZE, TILE_SIZE)。
        if cond.shape[1] != TILE_SIZE or cond.shape[2] != TILE_SIZE:
            scale_h = TILE_SIZE / cond.shape[1]
            scale_w = TILE_SIZE / cond.shape[2]
            # 对每个空间维度放大（通道轴不动 -> zoom 因子 [1, scale_h, scale_w]）
            cond = ndi_zoom(cond, (1, scale_h, scale_w), order=1).astype(np.float32)

        if self.augment:
            # 同步增强：条件与 DEM 必须做相同变换
            if np.random.rand() < 0.5:
                cond = cond[:, :, ::-1].copy()
                dem = dem[:, ::-1].copy()
            if np.random.rand() < 0.5:
                cond = cond[:, ::-1, :].copy()
                dem = dem[::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                cond = np.rot90(cond, k, axes=(1, 2)).copy()
                dem = np.rot90(dem, k).copy()

        cond_t = torch.from_numpy(cond).float()                  # (12, H, W)
        dem_t = torch.from_numpy(elev_to_norm(dem)).unsqueeze(0).float()  # (1, H, W) [-1,1]
        return cond_t, dem_t


# ==================== 配对数据生成脚本辅助 ====================

def generate_paired_from_engine(seeds: list[int], out_dir: Path = PAIRED_DIR,
                                tect_res: int = 512, size_px: int = 2048,
                                block: int = 8) -> int:
    """用现有 GAN 引擎批量生成 (条件, DEM) 配对样本。

    流程：对每个 seed 调用现有 generate_world 管线，在 GAN 推理阶段
    保存 (12 通道条件, 256² DEM 切片) 对。

    Parameters
    ----------
    seeds : list[int]
        世界种子列表。
    out_dir : Path
        输出目录。
    tect_res, size_px, block : int
        与 generate_world 一致。

    Returns
    -------
    int
        生成的配对样本数。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for seed in seeds:
        # 复用现有管线，但需要 hook 到 GAN 推理阶段保存条件+输出
        # 这里给出框架，实际需在 generate_world 内加 callback 或拆分调用
        # 简化版：直接调 plate_sim + world_to_channels + GAN 切片
        try:
            from terrain.core.plate_sim import run_plate_sim
            from terrain.core.generate_world import world_to_channels
            from terrain.core.terrain_engine import UNet, device
            from scipy.ndimage import zoom as ndi_zoom
            import torch

            # 1. 板块模拟
            elev_T, tect_T, *_ = run_plate_sim(T=tect_res, seed=seed, verbose=False)
            up = size_px // tect_res
            elev_m = ndi_zoom(elev_T.astype(np.float32), up, order=1)[:size_px, :size_px]
            lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)

            # 2. 12 通道条件
            ch_raw, *_ = world_to_channels(elev_m, lats, tect_T, block, seed)

            # 3. 切成 256² 块（这里简化：随机切 4 块）
            for k in range(4):
                h0 = np.random.randint(0, size_px - TILE_SIZE + 1)
                w0 = np.random.randint(0, size_px - TILE_SIZE + 1)
                # 条件在 block 分辨率，先切 block_tile × block_tile
                block_tile = TILE_SIZE // block
                bh0, bw0 = h0 // block, w0 // block
                cond_tile = ch_raw[:, bh0:bh0 + block_tile, bw0:bw0 + block_tile]
                # 再放大到 TILE_SIZE（保持与 ConditionEncoder 输入 (12,256,256) 一致）
                from scipy.ndimage import zoom as _zoom
                cond_tile = _zoom(cond_tile, (1, block, block), order=1)
                # 注：实际 dem_tile 应来自 GAN 输出；这里用 elev_m 切片作占位
                dem_tile = elev_m[h0:h0 + TILE_SIZE, w0:w0 + TILE_SIZE]
                np.savez_compressed(
                    out_dir / f"paired_seed{seed}_tile{k}.npz",
                    cond=cond_tile.astype(np.float32),
                    dem=dem_tile.astype(np.float32),
                )
                count += 1
        except Exception as e:
            print(f"  seed={seed} 失败: {e}")
            continue

    return count


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="生成配对样本")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200, 300, 500])
    p.add_argument("--out", type=str, default=str(PAIRED_DIR))
    a = p.parse_args()
    n = generate_paired_from_engine(a.seeds, Path(a.out))
    print(f"生成 {n} 个配对样本到 {a.out}")
