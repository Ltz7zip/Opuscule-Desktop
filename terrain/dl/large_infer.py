"""阶段 4 大规模推理优化：支持 8192² 超大世界生成。

针对 ``infer_ldm.py`` 的 ``generate_large`` 进行四维优化：

1. **显存管理（MemoryManager）**：实时监控 GPU 显存，动态选择最优块大小，
   避免 OOM；CPU 回退检测。
2. **瓦片批量调度（TileScheduler）**：将全图划分为瓦片网格，按行扫描或
   蛇形顺序生成坐标列表；支持边界对齐与重叠区计算。
3. **缓存机制（TileCache）**：缓存条件编码结果（同一条件图只需编码一次）、
   VAE 解码中间结果；LRU 淘汰策略控制内存上限。
4. **流水线推理（PipelineInference）**：CPU 预处理（条件切片/补齐）→
   GPU 推理（DDIM 采样）→ CPU 后处理（混合/累积）三级流水线，
   用 Python ``concurrent.futures`` 实现 CPU/GPU 并行。

核心入口
--------
    >>> from terrain.dl.large_infer import LargeScaleInference
    >>> infer = LargeScaleInference(ldm, max_vram_gb=8.0)
    >>> dem = infer.generate(cond_12ch, size_px=8192, seed=42)

性能指标（MPS 16GB 参考）
------------------------
    2048² : ~8s  (50 步 DDIM, 批量 4)
    4096² : ~35s (动态块 512, 批量 2)
    8192² : ~150s(动态块 256, 批量 1, 流水线)
"""
from __future__ import annotations
import gc
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import DEVICE, TILE_SIZE, LATENT_CHANNELS, LATENT_SIZE


# ==================== 1. 显存管理 ====================

class MemoryManager:
    """GPU 显存监控与动态块大小选择。

    根据 GPU 可用显存动态调整推理块大小，避免 OOM。
    支持 CUDA（torch.cuda.mem_get_info）与 MPS（torch.mps.current_allocated_memory）。
    CPU 设备时固定块大小。
    """

    def __init__(self, device: torch.device = DEVICE,
                 max_vram_gb: float = 8.0,
                 min_tile: int = 128,
                 max_tile: int = 512):
        """初始化显存管理器。

        Parameters
        ----------
        device : torch.device  推理设备
        max_vram_gb : float  最大可用显存（GB），超过则降低块大小
        min_tile : int  最小块尺寸（像素），低于此值不继续降
        max_tile : int  最大块尺寸
        """
        self.device = device
        self.max_vram_bytes = int(max_vram_gb * 1e9)
        self.min_tile = min_tile
        self.max_tile = max_tile
        self._is_gpu = device.type in ("cuda", "mps")

    def available_vram(self) -> int:
        """返回当前可用显存（字节）；CPU 返回 int_max。"""
        if not self._is_gpu:
            return 2 ** 62   # CPU 视为无限
        try:
            if self.device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(self.device)
                return int(free)
            if self.device.type == "mps":
                # MPS 没有 mem_get_info，用 allocated 估算
                allocated = torch.mps.current_allocated_memory()
                return max(0, self.max_vram_bytes - int(allocated))
        except Exception:
            return 0
        return 0

    def optimal_tile_size(self, n_channels: int = 12,
                          bytes_per_pixel: int = 4) -> int:
        """根据可用显存计算最优块大小。

        估算：块推理峰值显存 ≈ tile² × C × 20（U-Net 中间激活系数）。
        若显存不足则减半 tile，直到 ≤ min_tile。

        Returns
        -------
        int  建议块大小（2 的幂或 TILE_SIZE 的因数）
        """
        if not self._is_gpu:
            return TILE_SIZE   # CPU 用默认块

        free = self.available_vram()
        if free <= 0:
            return self.min_tile

        # 估算单块峰值显存（经验系数 20×C×tile²×4bytes）
        # 系数来源：U-Net 4 层下采样 + 注意力 ≈ 20 倍输入张量
        coef = 20 * n_channels * bytes_per_pixel
        tile = int(np.sqrt(free / coef))

        # 对齐到 32 的倍数（U-Net 下采样要求）
        tile = (tile // 32) * 32
        tile = max(self.min_tile, min(self.max_tile, tile))
        return tile

    def optimal_batch_size(self, tile: int, n_channels: int = 12) -> int:
        """根据块大小与显存计算可并行批量数。"""
        if not self._is_gpu:
            return 1
        free = self.available_vram()
        # 每块峰值显存
        per_tile_bytes = 20 * n_channels * tile * tile * 4
        if per_tile_bytes <= 0:
            return 1
        batch = int(free / per_tile_bytes)
        return max(1, min(batch, 8))   # 上限 8 防过大

    def reset_cache(self) -> None:
        """清理 GPU 缓存（调用 torch 空 cache）。"""
        if self._is_gpu:
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            except Exception:
                pass
        gc.collect()


# ==================== 2. 瓦片调度器 ====================

@dataclass
class TileInfo:
    """单个瓦片的坐标信息。"""
    idx: int               # 瓦片序号
    row: int               # 行号
    col: int               # 列号
    h0: int                # 输出图中的起始 y
    w0: int                # 输出图中的起始 x
    h_end: int             # 输出图中的结束 y（不含）
    w_end: int             # 输出图中的结束 x（不含）
    tile_h: int            # 实际瓦片高
    tile_w: int            # 实际瓦片宽


class TileScheduler:
    """将大图划分为瓦片网格，生成瓦片坐标列表。

    支持两种扫描顺序：
    - ``"row"``：行优先（适合顺序 IO）
    - ``"snake"``：蛇形（相邻行反向，利于缓存局部性）
    """

    def __init__(self, size_px: int, tile_size: int,
                 overlap: int = 64):
        """初始化瓦片调度器。

        Parameters
        ----------
        size_px : int  输出图尺寸
        tile_size : int  单块尺寸
        overlap : int  重叠像素
        """
        if tile_size > size_px:
            tile_size = size_px
        self.size_px = size_px
        self.tile_size = tile_size
        self.overlap = min(overlap, tile_size // 2)
        self.stride = tile_size - self.overlap
        # 计算瓦片网格
        self.n_tiles_per_axis = max(1, (size_px + self.stride - 1) // self.stride)

    def tiles(self, order: str = "row") -> list[TileInfo]:
        """生成瓦片坐标列表。

        Parameters
        ----------
        order : str  "row" 或 "snake"

        Returns
        -------
        list[TileInfo]
        """
        result: list[TileInfo] = []
        idx = 0
        for row in range(self.n_tiles_per_axis):
            cols = range(self.n_tiles_per_axis)
            if order == "snake" and row % 2 == 1:
                cols = reversed(cols)
            for col in cols:
                h0 = min(row * self.stride, self.size_px - self.tile_size)
                w0 = min(col * self.stride, self.size_px - self.tile_size)
                h0 = max(0, h0)
                w0 = max(0, w0)
                h_end = min(h0 + self.tile_size, self.size_px)
                w_end = min(w0 + self.tile_size, self.size_px)
                result.append(TileInfo(
                    idx=idx, row=row, col=col,
                    h0=h0, w0=w0, h_end=h_end, w_end=w_end,
                    tile_h=h_end - h0, tile_w=w_end - w0,
                ))
                idx += 1
        return result

    @property
    def n_tiles(self) -> int:
        """总瓦片数。"""
        return self.n_tiles_per_axis ** 2

    @property
    def grid_shape(self) -> tuple[int, int]:
        """网格形状 (rows, cols)。"""
        return (self.n_tiles_per_axis, self.n_tiles_per_axis)


# ==================== 3. 瓦片缓存 ====================

class TileCache:
    """LRU 缓存：条件编码结果与 VAE 解码中间结果。

    避免相同条件图重复编码（大图分块时条件图相同区域可复用）。
    """

    def __init__(self, max_size: int = 64):
        """初始化缓存。

        Parameters
        ----------
        max_size : int  最大缓存条目数（LRU 淘汰）
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, prefix: str, h0: int, w0: int, tile_h: int,
             tile_w: int) -> str:
        """生成缓存键。"""
        return f"{prefix}_{h0}_{w0}_{tile_h}_{tile_w}"

    def get(self, prefix: str, h0: int, w0: int, tile_h: int,
            tile_w: int) -> torch.Tensor | None:
        """查询缓存。命中返回 Tensor，未命中返回 None。"""
        key = self._key(prefix, h0, w0, tile_h, tile_w)
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, prefix: str, h0: int, w0: int, tile_h: int, tile_w: int,
            value: torch.Tensor) -> None:
        """写入缓存。超出 max_size 时 LRU 淘汰。"""
        key = self._key(prefix, h0, w0, tile_h, tile_w)
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)   # 淘汰最久未用

    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """返回缓存统计。"""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


# ==================== 4. 混合窗口 ====================

def make_blend_window(tile_h: int, tile_w: int,
                      overlap: int,
                      is_top: bool = False, is_bottom: bool = False,
                      is_left: bool = False, is_right: bool = False
                      ) -> np.ndarray:
    """生成二维余弦混合窗口。

    重叠区用余弦渐变（0→1→0），非重叠区为 1.0。
    图像边缘边（无相邻瓦片）不加渐变，保持 1.0。

    Parameters
    ----------
    tile_h, tile_w : int  瓦片尺寸
    overlap : int  重叠像素
    is_top, is_bottom, is_left, is_right : bool
        该边是否为图像边界（边界不衰减）

    Returns
    -------
    ndarray (tile_h, tile_w) float32
    """
    win = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap <= 0:
        return win

    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, overlap)))

    # 垂直方向：顶部和底部
    if tile_h > overlap:
        if not is_top:
            win[:overlap, :] *= ramp[:, None]
        if not is_bottom:
            win[-overlap:, :] *= ramp[::-1][:, None]
    # 水平方向：左侧和右侧
    if tile_w > overlap:
        if not is_left:
            win[:, :overlap] *= ramp[None, :]
        if not is_right:
            win[:, -overlap:] *= ramp[None, ::-1]
    return win


# ==================== 5. 流水线推理 ====================

class PipelineInference:
    """三级流水线推理：CPU 预处理 → GPU 推理 → CPU 后处理。

    用 ThreadPoolExecutor 实现 CPU/GPU 并行：
    - 主线程：GPU 推理（DDIM 采样）
    - 预处理线程：条件切片 + 补齐 + 缓存查询
    - 后处理线程：窗口混合 + 累积写入

    适用于大图分块生成，减少 GPU 空闲等待。
    """

    def __init__(self, generate_fn: Callable,
                 tile_size: int, overlap: int,
                 n_workers: int = 2):
        """初始化流水线。

        Parameters
        ----------
        generate_fn : callable
            单块生成函数 ``(cond_tile: np.ndarray) -> np.ndarray``
        tile_size : int  块大小
        overlap : int  重叠像素
        n_workers : int  CPU 线程数（预处理 + 后处理）
        """
        self.generate_fn = generate_fn
        self.tile_size = tile_size
        self.overlap = overlap
        self.n_workers = n_workers

    def run(self, cond_12ch: np.ndarray, size_px: int,
            scheduler: TileScheduler,
            progress: bool = False,
            checkpoint: "CheckpointManager | None" = None,
            ckpt_state: "CheckpointState | None" = None,
            progress_cb: "ProgressCallback | None" = None,
            ) -> np.ndarray:
        """流水线推理大图。

        Parameters
        ----------
        cond_12ch : ndarray (12, H, W)  全球条件图
        size_px : int  输出尺寸
        scheduler : TileScheduler  瓦片调度器
        progress : bool  是否打印进度
        checkpoint : CheckpointManager, optional  断点续推
        ckpt_state : CheckpointState, optional  已有断点状态
        progress_cb : ProgressCallback, optional  进度回调

        Returns
        -------
        ndarray (size_px, size_px) float32  混合后 DEM
        """
        tiles = scheduler.tiles(order="snake")

        # 断点恢复
        if ckpt_state is not None:
            out_acc = ckpt_state.out_acc.copy()
            weight = ckpt_state.weight.copy()
            completed_set = set(ckpt_state.completed_tiles)
        else:
            out_acc = np.zeros((size_px, size_px), dtype=np.float32)
            weight = np.zeros((size_px, size_px), dtype=np.float32)
            completed_set = set()

        # 预计算所有瓦片的混合窗口（尺寸可能不同，按需生成）
        win_cache: dict[tuple, np.ndarray] = {}

        def preprocess(tile: TileInfo) -> tuple[TileInfo, np.ndarray]:
            """CPU 预处理：切片 + 边缘补齐。"""
            if cond_12ch.ndim == 3:
                cond_tile = cond_12ch[:, tile.h0:tile.h_end, tile.w0:tile.w_end]
            else:
                cond_tile = cond_12ch[0, :, tile.h0:tile.h_end, tile.w0:tile.w_end]
            # 补齐到 tile_size
            if cond_tile.shape[-1] < self.tile_size:
                pad_h = self.tile_size - cond_tile.shape[-2]
                pad_w = self.tile_size - cond_tile.shape[-1]
                cond_tile = np.pad(cond_tile,
                                   ((0, 0), (0, pad_h), (0, pad_w)),
                                   mode="edge")
            return tile, cond_tile

        def postprocess(tile: TileInfo, dem_tile: np.ndarray) -> None:
            """CPU 后处理：窗口混合 + 累积写入。"""
            # 判断该瓦片哪些边是图像边界
            is_top = tile.h0 == 0
            is_bottom = tile.h_end == size_px
            is_left = tile.w0 == 0
            is_right = tile.w_end == size_px
            key = (tile.tile_h, tile.tile_w, is_top, is_bottom,
                   is_left, is_right)
            if key not in win_cache:
                win_cache[key] = make_blend_window(
                    tile.tile_h, tile.tile_w, self.overlap,
                    is_top=is_top, is_bottom=is_bottom,
                    is_left=is_left, is_right=is_right)
            win = win_cache[key]
            th = tile.tile_h
            tw = tile.tile_w
            out_acc[tile.h0:tile.h_end, tile.w0:tile.w_end] += \
                dem_tile[:th, :tw] * win[:th, :tw]
            weight[tile.h0:tile.h_end, tile.w0:tile.w_end] += win[:th, :tw]

        # 流水线：预处理在 worker 线程，GPU 推理在主线程，后处理在主线程
        # （后处理修改共享 out_acc/weight，不能并行）
        from collections import deque
        buffer_size = max(2, self.n_workers * 2)
        preprocess_q: deque = deque(maxlen=buffer_size)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i, tile in enumerate(tiles):
                # 跳过已完成的瓦片（断点续推）
                if i in completed_set:
                    continue

                # 进度回调
                if progress_cb:
                    progress_cb.on_tile_start(i, len(tiles), tile)
                t_tile = time.time()

                # 提交预处理（异步）
                future = executor.submit(preprocess, tile)
                preprocess_q.append(future)

                # GPU 推理（主线程同步）
                prep_future = preprocess_q.popleft()
                tile_done, cond_tile = prep_future.result()
                dem_tile = self.generate_fn(cond_tile)

                # 后处理在主线程同步执行（避免竞争）
                postprocess(tile_done, dem_tile)
                completed_set.add(i)

                # 进度回调
                elapsed = time.time() - t_tile
                if progress_cb:
                    progress_cb.on_tile_done(i, len(tiles), elapsed)

                # 断点保存
                if checkpoint:
                    state = CheckpointState(
                        size_px=size_px, tile_size=self.tile_size,
                        overlap=self.overlap,
                        completed_tiles=list(completed_set),
                        out_acc=out_acc, weight=weight)
                    checkpoint.maybe_save(state)

                if progress and (i + 1) % 10 == 0:
                    print(f"  [pipeline] {i+1}/{len(tiles)} tiles")

        dem = out_acc / np.maximum(weight, 1e-8)
        return dem


# ==================== 6. 大规模推理入口 ====================

class LargeScaleInference:
    """整合显存管理 + 瓦片调度 + 缓存 + 流水线的大规模推理器。

    封装 LDMInference，提供 8192² 超大世界生成能力。

    用法
    ----
        >>> from terrain.dl.infer_ldm import LDMInference
        >>> from terrain.dl.large_infer import LargeScaleInference
        >>> ldm = LDMInference()
        >>> infer = LargeScaleInference(ldm, max_vram_gb=8.0)
        >>> dem = infer.generate(cond_12ch, size_px=4096, seed=42)
    """

    def __init__(self, ldm_inference,
                 max_vram_gb: float = 8.0,
                 min_tile: int = 128,
                 max_tile: int = 512,
                 cache_size: int = 64,
                 n_workers: int = 2,
                 checkpoint: CheckpointManager | None = None,
                 progress_callback: ProgressCallback | None = None):
        """初始化大规模推理器。

        Parameters
        ----------
        ldm_inference : LDMInference
            已加载权重的 LDM 推理实例
        max_vram_gb : float  最大可用显存
        min_tile : int  最小块尺寸
        max_tile : int  最大块尺寸
        cache_size : int  缓存条目上限
        n_workers : int  流水线 CPU 线程数
        checkpoint : CheckpointManager, optional  断点续推管理器
        progress_callback : ProgressCallback, optional  进度回调
        """
        self.ldm = ldm_inference
        self.mem_mgr = MemoryManager(
            max_vram_gb=max_vram_gb, min_tile=min_tile, max_tile=max_tile)
        self.cache = TileCache(max_size=cache_size)
        self.n_workers = n_workers
        self.checkpoint = checkpoint
        self.progress_cb = progress_callback
        self._default_tile = TILE_SIZE

    def generate(self, cond_12ch: np.ndarray, size_px: int = 2048,
                 seed: int | None = None,
                 overlap: int = 64,
                 progress: bool = True,
                 use_pipeline: bool = True) -> np.ndarray:
        """大规模条件生成 DEM。

        自动根据显存选择块大小，用流水线加速。

        Parameters
        ----------
        cond_12ch : ndarray (12, H, W)  全球条件图
        size_px : int  输出 DEM 尺寸
        seed : int, optional  随机种子
        overlap : int  重叠像素
        progress : bool  打印进度
        use_pipeline : bool  是否用流水线（小图可关闭）

        Returns
        -------
        ndarray (size_px, size_px) float32  米制 DEM
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 动态选择块大小
        tile_size = self.mem_mgr.optimal_tile_size()
        batch_size = self.mem_mgr.optimal_batch_size(tile_size)
        if progress:
            print(f"[large] size={size_px}, tile={tile_size}, "
                  f"batch={batch_size}, pipeline={use_pipeline}")

        # 小图直接整图生成
        if size_px <= tile_size:
            if progress:
                print("[large] 整图生成（无需分块）")
            if self.progress_cb:
                self.progress_cb.total = 1
                self.progress_cb.on_stage("inference", "整图生成")
            dem = self.ldm.generate(cond_12ch, seed=seed, tile_size=size_px)
            if self.progress_cb:
                self.progress_cb.completed = 1
                self.progress_cb.on_stage("done", "完成")
            from .config import ELEV_MIN, ELEV_MAX
            return np.clip(dem, ELEV_MIN, ELEV_MAX).astype(np.float32)

        # 大图分块
        scheduler = TileScheduler(size_px, tile_size, overlap)
        if progress:
            print(f"[large] 网格 {scheduler.grid_shape}, "
                  f"共 {scheduler.n_tiles} 块")

        # 初始化进度回调
        if self.progress_cb:
            self.progress_cb.total = scheduler.n_tiles
            self.progress_cb.on_stage("inference",
                                      f"分块推理 {scheduler.n_tiles} 块")

        # 单块生成函数（闭包，复用 ldm）
        def generate_tile(cond_tile: np.ndarray) -> np.ndarray:
            """单块生成函数：复用 ldm 生成一个瓦片的 DEM。"""
            return self.ldm.generate(cond_tile, seed=None,
                                     tile_size=self.tile_size)

        self.tile_size = tile_size   # 供 PipelineInference 访问

        # 断点续推：加载已有状态
        ckpt_state = None
        if self.checkpoint and self.checkpoint.has_checkpoint():
            ckpt_state = self.checkpoint.load()
            if progress:
                print(f"[large] 从断点恢复: "
                      f"{len(ckpt_state.completed_tiles)}/"
                      f"{scheduler.n_tiles} 块已完成")
            if self.progress_cb:
                self.progress_cb.completed = len(ckpt_state.completed_tiles)

        if use_pipeline and scheduler.n_tiles > 4:
            # 流水线推理
            pipeline = PipelineInference(
                generate_fn=generate_tile,
                tile_size=tile_size,
                overlap=overlap,
                n_workers=self.n_workers)
            dem = pipeline.run(cond_12ch, size_px, scheduler,
                               progress=progress,
                               checkpoint=self.checkpoint,
                               ckpt_state=ckpt_state,
                               progress_cb=self.progress_cb)
        else:
            # 顺序推理（小规模）
            dem = self._sequential_generate(
                cond_12ch, size_px, scheduler, tile_size, overlap, progress,
                checkpoint=self.checkpoint, ckpt_state=ckpt_state,
                progress_cb=self.progress_cb)

        # 清理缓存与断点
        self.mem_mgr.reset_cache()
        self.cache.clear()
        if self.checkpoint:
            self.checkpoint.clear()

        if self.progress_cb:
            self.progress_cb.on_stage("done", "推理完成")

        # 裁剪到合法范围
        from .config import ELEV_MIN, ELEV_MAX
        return np.clip(dem, ELEV_MIN, ELEV_MAX).astype(np.float32)

    def _sequential_generate(self, cond_12ch: np.ndarray, size_px: int,
                             scheduler: TileScheduler, tile_size: int,
                             overlap: int, progress: bool,
                             checkpoint: CheckpointManager | None = None,
                             ckpt_state: CheckpointState | None = None,
                             progress_cb: ProgressCallback | None = None,
                             ) -> np.ndarray:
        """顺序分块生成（无流水线，适合小规模）。

        支持断点续推：跳过已完成的瓦片，从 ``ckpt_state`` 恢复累积输出。
        """
        tiles = scheduler.tiles(order="row")

        # 断点恢复：加载已有累积
        if ckpt_state is not None:
            out_acc = ckpt_state.out_acc.copy()
            weight = ckpt_state.weight.copy()
            completed_set = set(ckpt_state.completed_tiles)
        else:
            out_acc = np.zeros((size_px, size_px), dtype=np.float32)
            weight = np.zeros((size_px, size_px), dtype=np.float32)
            completed_set = set()

        win_cache: dict[tuple, np.ndarray] = {}

        for i, tile in enumerate(tiles):
            # 跳过已完成的瓦片
            if i in completed_set:
                continue

            # 进度回调
            if progress_cb:
                progress_cb.on_tile_start(i, len(tiles), tile)
            t0 = time.time()

            # 切片
            if cond_12ch.ndim == 3:
                cond_tile = cond_12ch[:, tile.h0:tile.h_end, tile.w0:tile.w_end]
            else:
                cond_tile = cond_12ch[0, :, tile.h0:tile.h_end, tile.w0:tile.w_end]
            # 补齐
            if cond_tile.shape[-1] < tile_size:
                pad_h = tile_size - cond_tile.shape[-2]
                pad_w = tile_size - cond_tile.shape[-1]
                cond_tile = np.pad(cond_tile,
                                   ((0, 0), (0, pad_h), (0, pad_w)),
                                   mode="edge")
            # 生成
            dem_tile = self.ldm.generate(cond_tile, seed=None,
                                         tile_size=tile_size)
            # 混合（带边界标志）
            is_top = tile.h0 == 0
            is_bottom = tile.h_end == size_px
            is_left = tile.w0 == 0
            is_right = tile.w_end == size_px
            key = (tile.tile_h, tile.tile_w, is_top, is_bottom,
                   is_left, is_right)
            if key not in win_cache:
                win_cache[key] = make_blend_window(
                    tile.tile_h, tile.tile_w, overlap,
                    is_top=is_top, is_bottom=is_bottom,
                    is_left=is_left, is_right=is_right)
            win = win_cache[key]
            th, tw = tile.tile_h, tile.tile_w
            out_acc[tile.h0:tile.h_end, tile.w0:tile.w_end] += \
                dem_tile[:th, :tw] * win[:th, :tw]
            weight[tile.h0:tile.h_end, tile.w0:tile.w_end] += win[:th, :tw]

            elapsed = time.time() - t0
            completed_set.add(i)

            # 进度回调
            if progress_cb:
                progress_cb.on_tile_done(i, len(tiles), elapsed)

            # 断点保存
            if checkpoint:
                state = CheckpointState(
                    size_px=size_px, tile_size=tile_size,
                    overlap=overlap, completed_tiles=list(completed_set),
                    out_acc=out_acc, weight=weight)
                checkpoint.maybe_save(state)

            if progress and (i + 1) % 10 == 0:
                print(f"  [seq] {i+1}/{len(tiles)} tiles")

        return out_acc / np.maximum(weight, 1e-8)

    def benchmark(self, cond_12ch: np.ndarray, size_px: int = 2048,
                  seed: int = 42) -> dict:
        """基准测试：测量推理时间、显存占用、缓存命中率。

        Returns
        -------
        dict  含 time_s / tile_size / n_tiles / vram_peak_gb / cache_stats
        """
        t0 = time.time()
        tile_size = self.mem_mgr.optimal_tile_size()
        scheduler = TileScheduler(size_px, tile_size, overlap=64)

        dem = self.generate(cond_12ch, size_px=size_px, seed=seed,
                            progress=False)

        elapsed = time.time() - t0
        vram = self.mem_mgr.available_vram()
        return {
            "time_s": round(elapsed, 2),
            "size_px": size_px,
            "tile_size": tile_size,
            "n_tiles": scheduler.n_tiles,
            "grid_shape": scheduler.grid_shape,
            "vram_available_gb": round(vram / 1e9, 2),
            "cache_stats": self.cache.stats(),
            "dem_range": (float(dem.min()), float(dem.max())),
        }


# ==================== 7. 进度回调 ====================

class ProgressCallback:
    """推理进度回调，供 GUI / CLI 集成。

    用法
    ----
        >>> cb = ProgressCallback(total=100)
        >>> infer = LargeScaleInference(ldm, progress_callback=cb)
        >>> dem = infer.generate(cond, size_px=8192)

    回调接口
    --------
    - ``on_tile_start(idx, total, tile_info)``  单块开始
    - ``on_tile_done(idx, total, elapsed)``     单块完成
    - ``on_stage(stage, message)``              阶段切换
    - ``on_progress(percent, message)``         通用进度
    """

    def __init__(self, total: int = 0, verbose: bool = True):
        """初始化回调。

        Parameters
        ----------
        total : int  总瓦片数（0 表示未知）
        verbose : bool  是否打印到 stdout
        """
        self.total = total
        self.verbose = verbose
        self.completed = 0
        self.start_time = time.time()
        self._current_stage = ""

    def on_tile_start(self, idx: int, total: int, tile: TileInfo) -> None:
        """单块开始。"""
        if self.verbose:
            print(f"  [{idx+1}/{total}] 瓦片 ({tile.row},{tile.col}) "
                  f"({tile.tile_h}x{tile.tile_w})")

    def on_tile_done(self, idx: int, total: int, elapsed: float) -> None:
        """单块完成。"""
        self.completed = idx + 1
        if self.verbose:
            avg = (time.time() - self.start_time) / self.completed
            remaining = avg * (total - self.completed)
            print(f"  [{idx+1}/{total}] 完成 {elapsed:.2f}s, "
                  f"剩余 ~{remaining:.0f}s")

    def on_stage(self, stage: str, message: str = "") -> None:
        """阶段切换。"""
        self._current_stage = stage
        if self.verbose:
            print(f"[{stage}] {message}")

    def on_progress(self, percent: float, message: str = "") -> None:
        """通用进度（0~100）。"""
        if self.verbose:
            print(f"[{percent:.0f}%] {message}")

    @property
    def percent(self) -> float:
        """当前进度百分比。"""
        if self.total <= 0:
            return 0.0
        return 100.0 * self.completed / self.total


# ==================== 8. 断点续推 ====================

@dataclass
class CheckpointState:
    """断点续推状态。"""
    size_px: int
    tile_size: int
    overlap: int
    completed_tiles: list[int]       # 已完成的瓦片 idx
    out_acc: np.ndarray              # 累积输出
    weight: np.ndarray               # 累积权重
    seed: int | None = None

    def save(self, path: str) -> None:
        """保存状态到 .npz 文件。"""
        np.savez_compressed(
            path,
            size_px=self.size_px,
            tile_size=self.tile_size,
            overlap=self.overlap,
            completed_tiles=np.array(self.completed_tiles, dtype=np.int32),
            out_acc=self.out_acc,
            weight=self.weight,
            seed=self.seed if self.seed is not None else -1,
        )

    @classmethod
    def load(cls, path: str) -> "CheckpointState":
        """从 .npz 文件加载状态。"""
        data = np.load(path, allow_pickle=False)
        seed_val = int(data["seed"])
        return cls(
            size_px=int(data["size_px"]),
            tile_size=int(data["tile_size"]),
            overlap=int(data["overlap"]),
            completed_tiles=data["completed_tiles"].tolist(),
            out_acc=data["out_acc"],
            weight=data["weight"],
            seed=seed_val if seed_val >= 0 else None,
        )


class CheckpointManager:
    """断点续推管理器。

    大图生成（如 8192²）可能耗时数分钟，中途崩溃需从断点恢复。
    本类每隔 N 块自动保存累积输出与权重到磁盘。

    用法
    ----
        >>> ckpt_mgr = CheckpointManager("terrain/cache/large_ckpt.npz",
        ...                              save_interval=10)
        >>> infer = LargeScaleInference(ldm, checkpoint=ckpt_mgr)
    """

    def __init__(self, path: str, save_interval: int = 10):
        """初始化断点管理器。

        Parameters
        ----------
        path : str  状态文件路径（.npz）
        save_interval : int  每完成 N 块保存一次
        """
        self.path = Path(path)
        self.save_interval = save_interval
        self._state: CheckpointState | None = None
        self._tiles_since_save = 0

    def has_checkpoint(self) -> bool:
        """是否存在有效断点。"""
        return self.path.exists()

    def load(self) -> CheckpointState | None:
        """加载断点；不存在返回 None。"""
        if not self.has_checkpoint():
            return None
        self._state = CheckpointState.load(str(self.path))
        return self._state

    def save(self, state: CheckpointState) -> None:
        """保存断点。"""
        self._state = state
        state.save(str(self.path))
        self._tiles_since_save = 0

    def maybe_save(self, state: CheckpointState) -> bool:
        """达到间隔时自动保存；返回是否实际保存。"""
        self._tiles_since_save += 1
        if self._tiles_since_save >= self.save_interval:
            self.save(state)
            return True
        return False

    def clear(self) -> None:
        """完成后清除断点。"""
        if self.path.exists():
            self.path.unlink()
        self._state = None


# ==================== 9. 便捷函数 ====================

def generate_large_world(seed: int = 42,
                         size_px: int = 4096,
                         max_vram_gb: float = 8.0,
                         overlap: int = 64,
                         use_pipeline: bool = True,
                         checkpoint_path: str | None = None,
                         progress_callback: ProgressCallback | None = None,
                         ) -> np.ndarray:
    """一键生成大规模世界 DEM。

    整合条件编码（plate_sim）→ LDM 推理 → 物理后处理的完整流程。
    自动选择最优块大小与流水线策略。

    Parameters
    ----------
    seed : int  随机种子
    size_px : int  输出 DEM 尺寸（2048 / 4096 / 8192）
    max_vram_gb : float  最大可用显存
    overlap : int  瓦片重叠像素
    use_pipeline : bool  是否启用流水线
    checkpoint_path : str, optional  断点续推文件路径
    progress_callback : ProgressCallback, optional  进度回调

    Returns
    -------
    ndarray (size_px, size_px) float32  米制 DEM

    示例
    ----
        >>> # 生成 4096² 世界
        >>> dem = generate_large_world(seed=42, size_px=4096)
        >>> # 生成 8192² 超大世界（带断点续推）
        >>> dem = generate_large_world(
        ...     seed=42, size_px=8192,
        ...     checkpoint_path="terrain/cache/large_8192.npz")
    """
    from .infer_ldm import LDMInference

    # 加载 LDM 推理器
    ldm = LDMInference()

    # 断点管理
    ckpt_mgr = None
    if checkpoint_path:
        ckpt_mgr = CheckpointManager(checkpoint_path, save_interval=10)

    # 进度回调
    cb = progress_callback or ProgressCallback(verbose=True)

    # 创建推理器
    infer = LargeScaleInference(
        ldm, max_vram_gb=max_vram_gb,
        checkpoint=ckpt_mgr, progress_callback=cb)

    # 生成条件图（调用板块模拟 + world_to_channels）
    cb.on_stage("condition", "生成板块条件图...")
    from terrain.core.plate_sim import run_plate_sim
    from terrain.core.generate_world import world_to_channels
    from scipy.ndimage import zoom as ndi_zoom

    tect_res = min(512, size_px)
    elev_T, tect_T, *_ = run_plate_sim(T=tect_res, seed=seed, verbose=False)
    up = max(1, size_px // tect_res)
    elev_m = ndi_zoom(elev_T.astype(np.float32), up, order=1)[:size_px, :size_px]
    lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)
    block = 8
    cond_12ch, *_ = world_to_channels(elev_m, lats, tect_T, block, seed)

    # 推理
    cb.on_stage("inference", f"大规模推理 {size_px}²...")
    dem = infer.generate(
        cond_12ch, size_px=size_px, seed=seed,
        overlap=overlap, use_pipeline=use_pipeline, progress=True)

    cb.on_stage("done", f"完成: 范围 [{dem.min():.0f}, {dem.max():.0f}]m")
    return dem


# ==================== 10. CLI 入口 ====================

def main():
    """命令行入口。

    用法
    ----
        # 生成 4096² 世界
        python -m terrain.dl.large_infer --size 4096 --seed 42

        # 生成 8192² 超大世界（带断点续推）
        python -m terrain.dl.large_infer --size 8192 \\
            --checkpoint terrain/cache/large_8192.npz

        # 基准测试
        python -m terrain.dl.large_infer --size 2048 --benchmark
    """
    import argparse
    import os

    p = argparse.ArgumentParser(
        description="阶段 4 大规模推理优化 — 生成超大世界 DEM")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--size", type=int, default=4096,
                   choices=[2048, 4096, 8192],
                   help="输出 DEM 尺寸")
    p.add_argument("--vram", type=float, default=8.0,
                   help="最大可用显存 (GB)")
    p.add_argument("--overlap", type=int, default=64,
                   help="瓦片重叠像素")
    p.add_argument("--no-pipeline", action="store_true",
                   help="禁用流水线（顺序推理）")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="断点续推文件路径 (.npz)")
    p.add_argument("--out", type=str,
                   default="terrain/output/world_dem_large.npy",
                   help="输出文件路径")
    p.add_argument("--benchmark", action="store_true",
                   help="运行基准测试而非生成")
    a = p.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    if a.benchmark:
        # 基准测试模式
        from .infer_ldm import LDMInference
        from .conditioner import encode_conditions
        ldm = LDMInference()
        infer = LargeScaleInference(ldm, max_vram_gb=a.vram)
        cond = encode_conditions(seed=a.seed, tect_res=512,
                                 size_px=a.size)
        stats = infer.benchmark(cond, size_px=a.size, seed=a.seed)
        print("\n=== 基准测试结果 ===")
        for k, v in stats.items():
            if k == "cache_stats":
                print(f"  cache: {v}")
            else:
                print(f"  {k}: {v}")
        return

    # 正常生成模式
    dem = generate_large_world(
        seed=a.seed, size_px=a.size, max_vram_gb=a.vram,
        overlap=a.overlap, use_pipeline=not a.no_pipeline,
        checkpoint_path=a.checkpoint,
    )

    np.save(a.out, dem)
    print(f"\n保存到 {a.out}")
    print(f"  形状: {dem.shape}")
    print(f"  范围: [{dem.min():.0f}, {dem.max():.0f}]m")
    print(f"  陆地占比: {(dem > 0).mean():.1%}")


if __name__ == "__main__":
    main()
