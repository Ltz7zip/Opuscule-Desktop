"""阶段 4 大规模推理优化测试。

覆盖：
- MemoryManager：显存查询、动态块大小、批量大小、CPU 回退
- TileScheduler：网格划分、行/蛇形顺序、边界对齐
- TileCache：LRU 淘汰、命中率统计、清空
- make_blend_window：余弦渐变、重叠区、尺寸正确
- PipelineInference：流水线推理、混合正确性
- LargeScaleInference：整合入口、小图直通、基准测试

不依赖训练好的权重，用 mock 生成函数模拟 LDM。
"""
from __future__ import annotations
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from terrain.dl.large_infer import (
    MemoryManager, TileScheduler, TileInfo, TileCache,
    make_blend_window, PipelineInference, LargeScaleInference,
    ProgressCallback, CheckpointState, CheckpointManager,
)
from terrain.dl.config import DEVICE, ELEV_MIN, ELEV_MAX


# ==================== MemoryManager 测试 ====================

class TestMemoryManager:

    def test_cpu_returns_default_tile(self):
        """CPU 设备返回默认 TILE_SIZE。"""
        mgr = MemoryManager(device=torch.device("cpu"))
        assert mgr.optimal_tile_size() == 256   # TILE_SIZE

    def test_cpu_batch_size_is_1(self):
        """CPU 设备批量固定为 1。"""
        mgr = MemoryManager(device=torch.device("cpu"))
        assert mgr.optimal_batch_size(tile=256) == 1

    def test_cpu_available_vram_large(self):
        """CPU 可用显存视为无限大。"""
        mgr = MemoryManager(device=torch.device("cpu"))
        assert mgr.available_vram() > 1e18

    def test_min_tile_constraint(self):
        """动态块大小不低于 min_tile。"""
        mgr = MemoryManager(device=torch.device("cpu"),
                            min_tile=128, max_tile=512)
        tile = mgr.optimal_tile_size()
        assert tile >= 128

    def test_max_tile_constraint(self):
        """动态块大小不超过 max_tile。"""
        mgr = MemoryManager(device=torch.device("cpu"),
                            min_tile=128, max_tile=512)
        tile = mgr.optimal_tile_size()
        assert tile <= 512

    def test_reset_cache_no_crash(self):
        """reset_cache 在 CPU 上不崩溃。"""
        mgr = MemoryManager(device=torch.device("cpu"))
        mgr.reset_cache()   # 不应抛异常

    def test_gpu_detection(self):
        """_is_gpu 标志与设备类型一致。"""
        mgr_cpu = MemoryManager(device=torch.device("cpu"))
        assert mgr_cpu._is_gpu is False

        mgr_mps = MemoryManager(device=torch.device("mps"))
        assert mgr_mps._is_gpu is True


# ==================== TileScheduler 测试 ====================

class TestTileScheduler:

    def test_single_tile(self):
        """图尺寸 = 块尺寸 → 单块。"""
        sched = TileScheduler(size_px=256, tile_size=256, overlap=0)
        assert sched.n_tiles == 1
        assert sched.grid_shape == (1, 1)

    def test_grid_2x2(self):
        """256 图 / 128 块 / 0 重叠 → 2×2 网格。"""
        sched = TileScheduler(size_px=256, tile_size=128, overlap=0)
        assert sched.grid_shape == (2, 2)
        assert sched.n_tiles == 4

    def test_grid_with_overlap(self):
        """重叠区增加瓦片数（stride 减小）。"""
        sched_no = TileScheduler(size_px=512, tile_size=256, overlap=0)
        sched_ov = TileScheduler(size_px=512, tile_size=256, overlap=64)
        # overlap=64 → stride=192 → 3 块/轴 → 9 块
        # overlap=0  → stride=256 → 2 块/轴 → 4 块
        assert sched_ov.n_tiles >= sched_no.n_tiles

    def test_row_order(self):
        """行优先顺序：按行扫描。"""
        sched = TileScheduler(size_px=256, tile_size=128, overlap=0)
        tiles = sched.tiles(order="row")
        assert len(tiles) == 4
        # 第一块在左上角
        assert tiles[0].h0 == 0 and tiles[0].w0 == 0
        # 第二块在右上角
        assert tiles[1].h0 == 0 and tiles[1].w0 == 128

    def test_snake_order(self):
        """蛇形顺序：第二行反向。"""
        sched = TileScheduler(size_px=256, tile_size=128, overlap=0)
        tiles = sched.tiles(order="snake")
        assert len(tiles) == 4
        # 第二行第一块应在右下角（col 反向）
        assert tiles[2].col == 1   # 第二行第一个 = col 1

    def test_tile_indices_sequential(self):
        """瓦片序号 idx 从 0 递增。"""
        sched = TileScheduler(size_px=512, tile_size=128, overlap=0)
        tiles = sched.tiles()
        for i, tile in enumerate(tiles):
            assert tile.idx == i

    def test_boundary_alignment(self):
        """边界瓦片不超出 size_px。"""
        sched = TileScheduler(size_px=300, tile_size=128, overlap=32)
        tiles = sched.tiles()
        for tile in tiles:
            assert tile.h_end <= 300
            assert tile.w_end <= 300
            assert tile.h0 >= 0
            assert tile.w0 >= 0

    def test_tile_info_fields(self):
        """TileInfo 字段完整。"""
        sched = TileScheduler(size_px=256, tile_size=128, overlap=0)
        tiles = sched.tiles()
        t = tiles[0]
        assert hasattr(t, "idx")
        assert hasattr(t, "row")
        assert hasattr(t, "col")
        assert hasattr(t, "h0")
        assert hasattr(t, "w0")
        assert hasattr(t, "h_end")
        assert hasattr(t, "w_end")
        assert hasattr(t, "tile_h")
        assert hasattr(t, "tile_w")

    def test_overlap_clamped(self):
        """overlap > tile_size/2 时被截断。"""
        sched = TileScheduler(size_px=256, tile_size=128, overlap=128)
        assert sched.overlap <= 64   # tile_size // 2


# ==================== TileCache 测试 ====================

class TestTileCache:

    def test_put_and_get(self):
        """写入后能读取。"""
        cache = TileCache(max_size=4)
        val = torch.randn(1, 4, 32, 32)
        cache.put("cond", 0, 0, 128, 128, val)
        result = cache.get("cond", 0, 0, 128, 128)
        assert result is not None
        assert torch.equal(result, val)

    def test_miss(self):
        """未写入的键返回 None。"""
        cache = TileCache()
        assert cache.get("cond", 0, 0, 128, 128) is None

    def test_lru_eviction(self):
        """超出 max_size 时淘汰最久未用。"""
        cache = TileCache(max_size=2)
        cache.put("a", 0, 0, 64, 64, torch.zeros(1))
        cache.put("b", 0, 0, 64, 64, torch.zeros(1))
        # 访问 a 使其变为最近使用
        cache.get("a", 0, 0, 64, 64)
        # 写入 c → 淘汰 b（最久未用）
        cache.put("c", 0, 0, 64, 64, torch.zeros(1))
        assert cache.get("a", 0, 0, 64, 64) is not None
        assert cache.get("b", 0, 0, 64, 64) is None
        assert cache.get("c", 0, 0, 64, 64) is not None

    def test_stats(self):
        """统计含 hits/misses/hit_rate。"""
        cache = TileCache(max_size=4)
        cache.put("a", 0, 0, 64, 64, torch.zeros(1))
        cache.get("a", 0, 0, 64, 64)   # hit
        cache.get("b", 0, 0, 64, 64)   # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 1e-6

    def test_clear(self):
        """clear 清空缓存与统计。"""
        cache = TileCache(max_size=4)
        cache.put("a", 0, 0, 64, 64, torch.zeros(1))
        cache.clear()
        assert cache.get("a", 0, 0, 64, 64) is None
        stats = cache.stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0

    def test_different_keys(self):
        """不同坐标的键独立。"""
        cache = TileCache()
        cache.put("cond", 0, 0, 64, 64, torch.zeros(1))
        cache.put("cond", 64, 0, 64, 64, torch.ones(1))
        a = cache.get("cond", 0, 0, 64, 64)
        b = cache.get("cond", 64, 0, 64, 64)
        assert a is not None and b is not None
        assert not torch.equal(a, b)

    def test_move_to_end_on_get(self):
        """get 后该条目变为最近使用。"""
        cache = TileCache(max_size=2)
        cache.put("a", 0, 0, 64, 64, torch.zeros(1))
        cache.put("b", 0, 0, 64, 64, torch.zeros(1))
        cache.get("a", 0, 0, 64, 64)   # a 变最近
        cache.put("c", 0, 0, 64, 64, torch.zeros(1))  # 淘汰 b
        assert cache.get("a", 0, 0, 64, 64) is not None
        assert cache.get("b", 0, 0, 64, 64) is None


# ==================== make_blend_window 测试 ====================

class TestBlendWindow:

    def test_shape(self):
        """窗口形状与输入一致。"""
        win = make_blend_window(128, 128, overlap=32)
        assert win.shape == (128, 128)

    def test_no_overlap_all_ones(self):
        """overlap=0 → 全 1。"""
        win = make_blend_window(64, 64, overlap=0)
        assert np.allclose(win, 1.0)

    def test_center_is_one(self):
        """重叠区中心（非边缘）为 1.0。"""
        win = make_blend_window(128, 128, overlap=32)
        assert abs(win[64, 64] - 1.0) < 1e-6

    def test_corner_is_zero(self):
        """四角为 0（两个重叠区交叉，非边界瓦片）。"""
        win = make_blend_window(128, 128, overlap=32,
                                is_top=False, is_bottom=False,
                                is_left=False, is_right=False)
        assert win[0, 0] < 0.1
        assert win[0, -1] < 0.1
        assert win[-1, 0] < 0.1
        assert win[-1, -1] < 0.1

    def test_edge_gradient(self):
        """边缘有渐变（非恒定），角落在两个方向交叉处接近 0。"""
        win = make_blend_window(128, 128, overlap=32,
                                is_top=False, is_left=False)
        # 顶行被垂直 ramp 归零，用第 5 行（垂直 ramp 中段）检查水平渐变
        row = win[5, :32]   # 水平重叠区
        assert not np.allclose(row, row[0])

    def test_symmetry(self):
        """窗口左右对称。"""
        win = make_blend_window(128, 128, overlap=32)
        assert np.allclose(win, win[:, ::-1], atol=1e-6)
        assert np.allclose(win, win[::-1, :], atol=1e-6)

    def test_dtype_float32(self):
        """窗口类型为 float32。"""
        win = make_blend_window(64, 64, overlap=16)
        assert win.dtype == np.float32

    def test_boundary_edge_no_fade(self):
        """边界边不衰减（保持 1.0）。"""
        win = make_blend_window(128, 128, overlap=32,
                                is_top=True, is_bottom=True,
                                is_left=True, is_right=True)
        # 全边界 → 全 1
        assert np.allclose(win, 1.0)

    def test_partial_boundary(self):
        """仅顶部边界 → 顶行不衰减，底行衰减。"""
        win = make_blend_window(128, 128, overlap=32,
                                is_top=True, is_bottom=False)
        # 顶行中心为 1
        assert abs(win[0, 64] - 1.0) < 1e-6
        # 底行中心 < 1（有衰减）
        assert win[-1, 64] < 1.0


# ==================== PipelineInference 测试 ====================

class TestPipelineInference:

    def _mock_generate(self, cond_tile: np.ndarray) -> np.ndarray:
        """Mock 生成函数：返回与输入同尺寸的常数图。"""
        h, w = cond_tile.shape[-2:]
        return np.full((h, w), 1000.0, dtype=np.float32)

    def test_pipeline_basic(self):
        """流水线基础推理：输出形状正确。"""
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        sched = TileScheduler(256, 128, overlap=32)
        pipe = PipelineInference(
            generate_fn=self._mock_generate,
            tile_size=128, overlap=32, n_workers=2)
        dem = pipe.run(cond, 256, sched, progress=False)
        assert dem.shape == (256, 256)

    def test_pipeline_constant_output(self):
        """全块输出相同常数 → 混合后仍为该常数。"""
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        sched = TileScheduler(256, 128, overlap=32)
        pipe = PipelineInference(
            generate_fn=self._mock_generate,
            tile_size=128, overlap=32)
        dem = pipe.run(cond, 256, sched, progress=False)
        # 因所有块输出 1000，混合后应 ≈ 1000
        assert np.allclose(dem, 1000.0, atol=1e-3)

    def test_pipeline_large(self):
        """大图 512² 分块推理。"""
        cond = np.random.randn(12, 512, 512).astype(np.float32)
        sched = TileScheduler(512, 128, overlap=32)
        pipe = PipelineInference(
            generate_fn=self._mock_generate,
            tile_size=128, overlap=32, n_workers=2)
        dem = pipe.run(cond, 512, sched, progress=False)
        assert dem.shape == (512, 512)
        assert np.allclose(dem, 1000.0, atol=1e-3)

    def test_pipeline_no_overlap(self):
        """overlap=0 仍能拼接（硬接缝）。"""
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        sched = TileScheduler(256, 128, overlap=0)
        pipe = PipelineInference(
            generate_fn=self._mock_generate,
            tile_size=128, overlap=0)
        dem = pipe.run(cond, 256, sched, progress=False)
        assert dem.shape == (256, 256)


# ==================== LargeScaleInference 测试 ====================

class TestLargeScaleInference:

    @pytest.fixture
    def mock_ldm(self):
        """Mock LDMInference：generate 返回常数图。"""
        ldm = MagicMock()
        def mock_generate(cond, seed=None, tile_size=256):
            h, w = cond.shape[-2:] if cond.ndim == 3 else cond.shape[-2:]
            return np.full((h, w), 500.0, dtype=np.float32)
        ldm.generate = mock_generate
        return ldm

    def test_small_image_direct(self, mock_ldm):
        """小图（≤tile_size）直接整图生成。"""
        infer = LargeScaleInference(mock_ldm, max_vram_gb=4.0)
        cond = np.random.randn(12, 128, 128).astype(np.float32)
        dem = infer.generate(cond, size_px=128, progress=False)
        assert dem.shape == (128, 128)

    def test_large_image_tiled(self, mock_ldm):
        """大图分块生成。"""
        infer = LargeScaleInference(mock_ldm, max_vram_gb=4.0)
        cond = np.random.randn(12, 512, 512).astype(np.float32)
        dem = infer.generate(cond, size_px=512, progress=False,
                             use_pipeline=False)
        assert dem.shape == (512, 512)

    def test_large_image_pipeline(self, mock_ldm):
        """大图流水线生成。"""
        infer = LargeScaleInference(mock_ldm, max_vram_gb=4.0)
        cond = np.random.randn(12, 512, 512).astype(np.float32)
        dem = infer.generate(cond, size_px=512, progress=False,
                             use_pipeline=True)
        assert dem.shape == (512, 512)

    def test_output_range(self, mock_ldm):
        """输出在合法高程范围内。"""
        infer = LargeScaleInference(mock_ldm)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        dem = infer.generate(cond, size_px=256, progress=False)
        assert dem.min() >= ELEV_MIN
        assert dem.max() <= ELEV_MAX

    def test_seed_reproducible(self, mock_ldm):
        """相同种子 → 相同结果（mock 下输出相同）。"""
        infer = LargeScaleInference(mock_ldm)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        dem1 = infer.generate(cond, size_px=256, seed=42, progress=False)
        dem2 = infer.generate(cond, size_px=256, seed=42, progress=False)
        np.testing.assert_array_equal(dem1, dem2)

    def test_benchmark(self, mock_ldm):
        """benchmark 返回完整指标。"""
        infer = LargeScaleInference(mock_ldm)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        stats = infer.benchmark(cond, size_px=256, seed=42)
        assert "time_s" in stats
        assert "tile_size" in stats
        assert "n_tiles" in stats
        assert "cache_stats" in stats
        assert stats["time_s"] >= 0

    def test_cache_cleared_after_generate(self, mock_ldm):
        """生成后缓存被清空。"""
        infer = LargeScaleInference(mock_ldm)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        infer.generate(cond, size_px=256, progress=False)
        assert infer.cache.stats()["size"] == 0


# ==================== 集成测试 ====================

class TestIntegration:

    def test_8192_grid(self):
        """8192² 图的瓦片网格正确划分。"""
        sched = TileScheduler(size_px=8192, tile_size=256, overlap=64)
        # stride=192, 8192/192 ≈ 42.7 → 43 块/轴
        assert sched.n_tiles_per_axis >= 42
        assert sched.n_tiles >= 42 ** 2

    def test_4096_grid(self):
        """4096² 图的瓦片网格。"""
        sched = TileScheduler(size_px=4096, tile_size=512, overlap=64)
        # stride=448, 4096/448 ≈ 9.1 → 10 块/轴
        assert sched.n_tiles_per_axis >= 9
        assert sched.n_tiles >= 81

    def test_tiles_cover_full_image(self):
        """所有瓦片覆盖完整图像（无遗漏）。"""
        size = 512
        sched = TileScheduler(size, 128, overlap=32)
        tiles = sched.tiles()
        coverage = np.zeros((size, size), dtype=bool)
        for t in tiles:
            coverage[t.h0:t.h_end, t.w0:t.w_end] = True
        assert coverage.all(), "瓦片未覆盖完整图像"

    def test_blend_window_sums_to_one(self):
        """相邻两块的重叠区窗口和为 1（无权重丢失）。"""
        overlap = 32
        tile_size = 128
        # 左块（右边界非图像边界）的右边重叠区
        win_left = make_blend_window(tile_size, tile_size, overlap,
                                     is_left=True, is_right=False)
        # 右块（左边界非图像边界）的左边重叠区
        win_right = make_blend_window(tile_size, tile_size, overlap,
                                      is_left=False, is_right=True)
        # 取中间行（避免垂直方向交叉）
        mid = tile_size // 2
        right_ramp = win_left[mid, -overlap:]
        left_ramp = win_right[mid, :overlap]
        combined = right_ramp + left_ramp
        assert np.allclose(combined, 1.0, atol=1e-5)

    def test_pipeline_vs_sequential_constant(self, ):
        """流水线与顺序推理在常数输出时结果一致。"""
        def gen(cond):
            h, w = cond.shape[-2:]
            return np.full((h, w), 777.0, dtype=np.float32)

        size_px = 256
        cond = np.random.randn(12, size_px, size_px).astype(np.float32)
        sched = TileScheduler(size_px, 128, overlap=32)

        pipe = PipelineInference(gen, 128, 32, n_workers=2)
        dem_pipe = pipe.run(cond, size_px, sched, progress=False)

        # 顺序手动计算（带边界标志）
        out = np.zeros((size_px, size_px), dtype=np.float32)
        wgt = np.zeros((size_px, size_px), dtype=np.float32)
        for tile in sched.tiles(order="row"):
            dem_tile = gen(cond[:, tile.h0:tile.h_end, tile.w0:tile.w_end])
            is_top = tile.h0 == 0
            is_bottom = tile.h_end == size_px
            is_left = tile.w0 == 0
            is_right = tile.w_end == size_px
            win = make_blend_window(tile.tile_h, tile.tile_w, 32,
                                    is_top=is_top, is_bottom=is_bottom,
                                    is_left=is_left, is_right=is_right)
            out[tile.h0:tile.h_end, tile.w0:tile.w_end] += dem_tile[:tile.tile_h, :tile.tile_w] * win
            wgt[tile.h0:tile.h_end, tile.w0:tile.w_end] += win
        dem_seq = out / np.maximum(wgt, 1e-8)

        np.testing.assert_allclose(dem_pipe, dem_seq, atol=1e-3)


# ==================== ProgressCallback 测试 ====================

class TestProgressCallback:

    def test_init_defaults(self):
        """初始化默认值。"""
        cb = ProgressCallback()
        assert cb.total == 0
        assert cb.completed == 0
        assert cb.percent == 0.0

    def test_on_tile_done_increments(self):
        """on_tile_done 递增 completed。"""
        cb = ProgressCallback(total=10, verbose=False)
        cb.on_tile_done(0, 10, 0.5)
        assert cb.completed == 1
        cb.on_tile_done(1, 10, 0.3)
        assert cb.completed == 2

    def test_percent_calculation(self):
        """percent = completed / total * 100。"""
        cb = ProgressCallback(total=4, verbose=False)
        cb.on_tile_done(0, 4, 0.1)
        assert cb.percent == 25.0
        cb.on_tile_done(1, 4, 0.1)
        cb.on_tile_done(2, 4, 0.1)
        cb.on_tile_done(3, 4, 0.1)
        assert cb.percent == 100.0

    def test_percent_zero_total(self):
        """total=0 → percent=0（防除零）。"""
        cb = ProgressCallback(total=0)
        assert cb.percent == 0.0

    def test_on_stage(self):
        """on_stage 更新当前阶段。"""
        cb = ProgressCallback(verbose=False)
        cb.on_stage("inference", "开始推理")
        assert cb._current_stage == "inference"

    def test_on_progress(self):
        """on_progress 不崩溃。"""
        cb = ProgressCallback(verbose=False)
        cb.on_progress(50.0, "半完成")

    def test_on_tile_start(self):
        """on_tile_start 不崩溃。"""
        cb = ProgressCallback(verbose=False)
        tile = TileInfo(idx=0, row=0, col=0, h0=0, w0=0,
                        h_end=128, w_end=128, tile_h=128, tile_w=128)
        cb.on_tile_start(0, 10, tile)

    def test_verbose_print(self, capsys):
        """verbose=True 时打印到 stdout。"""
        cb = ProgressCallback(verbose=True)
        cb.on_stage("test", "message")
        captured = capsys.readouterr()
        assert "test" in captured.out


# ==================== CheckpointState 测试 ====================

class TestCheckpointState:

    def test_save_and_load(self, tmp_path):
        """保存后加载，数据一致。"""
        state = CheckpointState(
            size_px=256, tile_size=128, overlap=32,
            completed_tiles=[0, 1, 2],
            out_acc=np.ones((256, 256), dtype=np.float32),
            weight=np.full((256, 256), 0.5, dtype=np.float32),
            seed=42,
        )
        path = str(tmp_path / "ckpt.npz")
        state.save(path)

        loaded = CheckpointState.load(path)
        assert loaded.size_px == 256
        assert loaded.tile_size == 128
        assert loaded.overlap == 32
        assert loaded.completed_tiles == [0, 1, 2]
        assert loaded.seed == 42
        np.testing.assert_array_equal(loaded.out_acc, state.out_acc)
        np.testing.assert_array_equal(loaded.weight, state.weight)

    def test_save_load_no_seed(self, tmp_path):
        """seed=None 时保存/加载正确（用 -1 占位）。"""
        state = CheckpointState(
            size_px=128, tile_size=64, overlap=16,
            completed_tiles=[],
            out_acc=np.zeros((128, 128), dtype=np.float32),
            weight=np.zeros((128, 128), dtype=np.float32),
            seed=None,
        )
        path = str(tmp_path / "ckpt_no_seed.npz")
        state.save(path)
        loaded = CheckpointState.load(path)
        assert loaded.seed is None

    def test_save_creates_file(self, tmp_path):
        """save 创建文件。"""
        state = CheckpointState(
            size_px=64, tile_size=32, overlap=0,
            completed_tiles=[],
            out_acc=np.zeros((64, 64), dtype=np.float32),
            weight=np.zeros((64, 64), dtype=np.float32),
        )
        path = tmp_path / "ckpt.npz"
        state.save(str(path))
        assert path.exists()


# ==================== CheckpointManager 测试 ====================

class TestCheckpointManager:

    def test_has_checkpoint_false(self, tmp_path):
        """不存在断点 → False。"""
        mgr = CheckpointManager(str(tmp_path / "nonexistent.npz"))
        assert not mgr.has_checkpoint()

    def test_has_checkpoint_true(self, tmp_path):
        """存在断点 → True。"""
        path = tmp_path / "ckpt.npz"
        state = CheckpointState(
            size_px=64, tile_size=32, overlap=0,
            completed_tiles=[],
            out_acc=np.zeros((64, 64), dtype=np.float32),
            weight=np.zeros((64, 64), dtype=np.float32),
        )
        state.save(str(path))
        mgr = CheckpointManager(str(path))
        assert mgr.has_checkpoint()

    def test_load_returns_state(self, tmp_path):
        """load 返回 CheckpointState。"""
        path = tmp_path / "ckpt.npz"
        state = CheckpointState(
            size_px=128, tile_size=64, overlap=16,
            completed_tiles=[0, 1],
            out_acc=np.ones((128, 128), dtype=np.float32),
            weight=np.ones((128, 128), dtype=np.float32),
            seed=7,
        )
        state.save(str(path))
        mgr = CheckpointManager(str(path))
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.size_px == 128
        assert loaded.seed == 7

    def test_load_nonexistent_returns_none(self, tmp_path):
        """load 不存在文件 → None。"""
        mgr = CheckpointManager(str(tmp_path / "no.npz"))
        assert mgr.load() is None

    def test_save_and_load(self, tmp_path):
        """save 后 load 数据一致。"""
        mgr = CheckpointManager(str(tmp_path / "ckpt.npz"))
        state = CheckpointState(
            size_px=64, tile_size=32, overlap=8,
            completed_tiles=[0],
            out_acc=np.full((64, 64), 3.0, dtype=np.float32),
            weight=np.full((64, 64), 1.0, dtype=np.float32),
        )
        mgr.save(state)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.completed_tiles == [0]

    def test_maybe_save_interval(self, tmp_path):
        """maybe_save 达到间隔时保存。"""
        mgr = CheckpointManager(str(tmp_path / "ckpt.npz"),
                                save_interval=3)
        state = CheckpointState(
            size_px=64, tile_size=32, overlap=0,
            completed_tiles=[],
            out_acc=np.zeros((64, 64), dtype=np.float32),
            weight=np.zeros((64, 64), dtype=np.float32),
        )
        # 前 2 次不保存
        assert not mgr.maybe_save(state)
        assert not mgr.maybe_save(state)
        # 第 3 次保存
        assert mgr.maybe_save(state)
        assert mgr.has_checkpoint()

    def test_maybe_save_resets_counter(self, tmp_path):
        """保存后计数器重置。"""
        mgr = CheckpointManager(str(tmp_path / "ckpt.npz"),
                                save_interval=2)
        state = CheckpointState(
            size_px=32, tile_size=16, overlap=0,
            completed_tiles=[],
            out_acc=np.zeros((32, 32), dtype=np.float32),
            weight=np.zeros((32, 32), dtype=np.float32),
        )
        mgr.maybe_save(state)
        mgr.maybe_save(state)   # 保存
        # 再调 1 次不应保存
        assert not mgr.maybe_save(state)

    def test_clear(self, tmp_path):
        """clear 删除断点文件。"""
        path = tmp_path / "ckpt.npz"
        state = CheckpointState(
            size_px=32, tile_size=16, overlap=0,
            completed_tiles=[],
            out_acc=np.zeros((32, 32), dtype=np.float32),
            weight=np.zeros((32, 32), dtype=np.float32),
        )
        state.save(str(path))
        mgr = CheckpointManager(str(path))
        assert mgr.has_checkpoint()
        mgr.clear()
        assert not mgr.has_checkpoint()

    def test_clear_nonexistent(self, tmp_path):
        """clear 不存在文件时不崩溃。"""
        mgr = CheckpointManager(str(tmp_path / "no.npz"))
        mgr.clear()   # 不应抛异常


# ==================== LargeScaleInference + Checkpoint 集成 ====================

class TestLargeScaleWithCheckpoint:

    @pytest.fixture
    def mock_ldm(self):
        """Mock LDMInference。"""
        ldm = MagicMock()
        def mock_generate(cond, seed=None, tile_size=256):
            h, w = cond.shape[-2:] if cond.ndim == 3 else cond.shape[-2:]
            return np.full((h, w), 500.0, dtype=np.float32)
        ldm.generate = mock_generate
        return ldm

    def test_generate_with_checkpoint(self, mock_ldm, tmp_path):
        """带断点续推的推理。"""
        ckpt_path = str(tmp_path / "ckpt.npz")
        ckpt_mgr = CheckpointManager(ckpt_path, save_interval=1)
        infer = LargeScaleInference(mock_ldm, checkpoint=ckpt_mgr)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        dem = infer.generate(cond, size_px=256, progress=False,
                             use_pipeline=False)
        assert dem.shape == (256, 256)
        # 完成后断点应被清除
        assert not ckpt_mgr.has_checkpoint()

    def test_generate_with_progress_callback(self, mock_ldm):
        """带进度回调的推理。"""
        cb = ProgressCallback(verbose=False)
        infer = LargeScaleInference(mock_ldm, progress_callback=cb)
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        dem = infer.generate(cond, size_px=256, progress=False,
                             use_pipeline=False)
        assert dem.shape == (256, 256)
        # 完成后进度应为 100%
        assert cb.percent == 100.0
        assert cb.completed > 0

    def test_checkpoint_resume(self, mock_ldm, tmp_path):
        """断点续推：先跑一半，再从断点恢复。"""
        ckpt_path = str(tmp_path / "ckpt.npz")

        # 第一次推理：手动保存部分结果
        ckpt_mgr = CheckpointManager(ckpt_path, save_interval=1)
        infer = LargeScaleInference(mock_ldm, checkpoint=ckpt_mgr)

        # 模拟部分完成的断点
        state = CheckpointState(
            size_px=256, tile_size=256, overlap=64,
            completed_tiles=[0, 1, 2, 3],   # 假设前 4 块已完成
            out_acc=np.full((256, 256), 500.0, dtype=np.float32),
            weight=np.ones((256, 256), dtype=np.float32),
        )
        ckpt_mgr.save(state)

        # 第二次推理：应从断点恢复
        assert ckpt_mgr.has_checkpoint()
        cond = np.random.randn(12, 256, 256).astype(np.float32)
        dem = infer.generate(cond, size_px=256, progress=False,
                             use_pipeline=False)
        assert dem.shape == (256, 256)
        # 输出应接近 500（mock 常数输出）
        assert np.allclose(dem, 500.0, atol=1e-3)

    def test_pipeline_with_progress_callback(self, mock_ldm):
        """流水线推理带进度回调。"""
        cb = ProgressCallback(verbose=False)
        infer = LargeScaleInference(mock_ldm, progress_callback=cb)
        cond = np.random.randn(12, 512, 512).astype(np.float32)
        dem = infer.generate(cond, size_px=512, progress=False,
                             use_pipeline=True)
        assert dem.shape == (512, 512)
        # 完成后阶段应为 done
        assert cb._current_stage == "done"

    def test_pipeline_with_checkpoint(self, mock_ldm, tmp_path):
        """流水线推理带断点续推。"""
        ckpt_path = str(tmp_path / "pipe_ckpt.npz")
        ckpt_mgr = CheckpointManager(ckpt_path, save_interval=1)
        infer = LargeScaleInference(mock_ldm, checkpoint=ckpt_mgr)
        cond = np.random.randn(12, 512, 512).astype(np.float32)
        dem = infer.generate(cond, size_px=512, progress=False,
                             use_pipeline=True)
        assert dem.shape == (512, 512)
        assert not ckpt_mgr.has_checkpoint()   # 完成后清除

    def test_small_image_with_callback(self, mock_ldm):
        """小图（整图生成）带进度回调。"""
        cb = ProgressCallback(verbose=False)
        infer = LargeScaleInference(mock_ldm, progress_callback=cb)
        cond = np.random.randn(12, 128, 128).astype(np.float32)
        dem = infer.generate(cond, size_px=128, progress=False)
        assert dem.shape == (128, 128)
        # 小图直接整图生成，应触发 done 阶段
        assert cb._current_stage == "done"


# ==================== generate_large_world 便捷函数测试 ====================

class TestGenerateLargeWorld:

    def test_generate_large_world_mock(self, monkeypatch):
        """mock 依赖后测试便捷函数。"""
        from terrain.dl import large_infer as li

        mock_ldm = MagicMock()
        def mock_generate(cond, seed=None, tile_size=256):
            h, w = cond.shape[-2:] if cond.ndim == 3 else cond.shape[-2:]
            return np.full((h, w), 100.0, dtype=np.float32)
        mock_ldm.generate = mock_generate

        # mock LDMInference 在源模块上
        import terrain.dl.infer_ldm as infer_ldm_mod
        monkeypatch.setattr(infer_ldm_mod, "LDMInference",
                            lambda: mock_ldm)

        # mock plate_sim.run_plate_sim
        def mock_run_plate(T, seed, verbose=False):
            elev = np.random.randn(T, T).astype(np.float32) * 1000
            tect = np.zeros((T, T), dtype=np.float32)
            return elev, tect, None, None

        # mock world_to_channels
        def mock_world_to_channels(elev_m, lats, tect_T, block, seed):
            ch = np.random.randn(12, elev_m.shape[0],
                                 elev_m.shape[1]).astype(np.float32)
            return ch, None

        # mock plate_sim.run_plate_sim 和 generate_world.world_to_channels
        import terrain.core.plate_sim as _plate_mod
        import terrain.core.generate_world as _gen_mod
        monkeypatch.setattr(_plate_mod, "run_plate_sim", mock_run_plate)
        monkeypatch.setattr(_gen_mod, "world_to_channels", mock_world_to_channels)

        # 调用
        dem = li.generate_large_world(seed=42, size_px=256)
        assert dem.shape == (256, 256)


# ==================== CLI 入口测试 ====================

class TestCLI:

    def test_main_help(self, monkeypatch, capsys):
        """CLI --help 不崩溃。"""
        from terrain.dl.large_infer import main
        import sys
        monkeypatch.setattr(sys, "argv",
                            ["large_infer", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        # argparse --help 退出码 0
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "大规模推理" in captured.out
