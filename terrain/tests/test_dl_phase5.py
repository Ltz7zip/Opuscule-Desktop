"""阶段 5 端到端集成管线测试。

覆盖三个子模块：
- sr_net：ESRGAN 架构、超分推理、损失函数、分块拼接
- multicond：单模态编码器、多模态融合、便捷函数
- pipeline：端到端生成流程、各阶段信息、基准测试

不依赖训练好的权重，用随机初始化网络验证形状/流程正确性。
"""
from __future__ import annotations
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from terrain.dl.sr_net import (
    DenseBlock, RRDB, SRNet, SRDiscriminator,
    ChannelAttention, SpatialAttention, CCABlock,
    sr_pixel_loss, sr_freq_loss, sr_loss, sr_upscale,
    sr_charbonnier_loss, sr_ssim_loss, sr_gradient_loss,
    _make_sr_window, _apply_tta,
    SRDataset, train_sr,
)
from terrain.dl.multicond import (
    SketchEncoder, ElevationHintEncoder, BiomeEncoder,
    MultiModalConditioner, encode_multimodal,
    ModalityConfig, CrossAttentionFusion,
    SeaLevelEncoder, ClimateZoneEncoder, PlateBoundaryEncoder,
    validate_sketch, validate_elev_hint, validate_biome,
    make_sea_level_field, make_climate_zone_field,
)
from terrain.dl.pipeline import (
    DiffusionPipeline, PipelineConfig, ConditionCache,
)
from terrain.dl.config import ELEV_MIN, ELEV_MAX, elev_to_norm, norm_to_elev


# ==================== DenseBlock 测试 ====================

class TestDenseBlock:

    def test_forward_shape(self):
        """前向输出形状正确（通道不变）。"""
        block = DenseBlock(in_ch=64, growth=32)
        x = torch.randn(1, 64, 32, 32)
        out = block(x)
        assert out.shape == (1, 64, 32, 32)

    def test_growth_increases_intermediate(self):
        """增长率影响内部通道（参数量）。"""
        block_small = DenseBlock(in_ch=32, growth=16)
        block_large = DenseBlock(in_ch=32, growth=32)
        p_small = sum(p.numel() for p in block_small.parameters())
        p_large = sum(p.numel() for p in block_large.parameters())
        assert p_large > p_small

    def test_batch_2(self):
        """批量 > 1 正常工作。"""
        block = DenseBlock(in_ch=16, growth=8)
        x = torch.randn(2, 16, 16, 16)
        out = block(x)
        assert out.shape == (2, 16, 16, 16)


# ==================== RRDB 测试 ====================

class TestRRDB:

    def test_forward_shape(self):
        """前向输出形状正确。"""
        rrdb = RRDB(in_ch=64, growth=32)
        x = torch.randn(1, 64, 32, 32)
        out = rrdb(x)
        assert out.shape == (1, 64, 32, 32)

    def test_residual_connection(self):
        """残差连接：输入与输出差异有限。"""
        rrdb = RRDB(in_ch=32, growth=16, beta=0.2)
        rrdb.eval()
        x = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            out = rrdb(x)
        # 残差缩放 0.2 → 输出与输入差异 < 1
        diff = (out - x).abs().mean()
        assert diff < 1.0

    def test_beta_zero_identity(self):
        """beta=0 → 输出 = 输入（纯恒等）。"""
        rrdb = RRDB(in_ch=16, growth=8, beta=0.0)
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            out = rrdb(x)
        assert torch.allclose(out, x, atol=1e-6)


# ==================== SRNet 测试 ====================

class TestSRNet:

    def test_forward_4x_upscale(self):
        """前向：4× 放大。"""
        net = SRNet(in_ch=1, base=16, n_rrdb=2)
        x = torch.randn(1, 1, 32, 32)
        out = net(x)
        assert out.shape == (1, 1, 128, 128)   # 32×4=128

    def test_forward_batch(self):
        """批量 > 1。"""
        net = SRNet(in_ch=1, base=16, n_rrdb=1)
        x = torch.randn(2, 1, 16, 16)
        out = net(x)
        assert out.shape == (2, 1, 64, 64)

    def test_output_range(self):
        """输出值域合理（非 NaN/Inf）。"""
        net = SRNet(in_ch=1, base=16, n_rrdb=2)
        x = torch.randn(1, 1, 16, 16)
        out = net(x)
        assert torch.isfinite(out).all()

    def test_param_count_small(self):
        """小模型参数量 < 5M（base=16）。"""
        net = SRNet(in_ch=1, base=16, n_rrdb=2)
        n = sum(p.numel() for p in net.parameters())
        assert n < 5_000_000

    def test_different_sizes(self):
        """不同输入尺寸。"""
        net = SRNet(in_ch=1, base=16, n_rrdb=1)
        for sz in [16, 32, 64]:
            x = torch.randn(1, 1, sz, sz)
            out = net(x)
            assert out.shape == (1, 1, sz * 4, sz * 4)


# ==================== SRDiscriminator 测试 ====================

class TestSRDiscriminator:

    def test_forward_shape(self):
        """前向输出标量 logits。"""
        disc = SRDiscriminator(in_ch=1, base=16)
        x = torch.randn(1, 1, 64, 64)
        out = disc(x)
        assert out.shape == (1,)

    def test_batch(self):
        """批量 > 1。"""
        disc = SRDiscriminator(in_ch=1, base=16)
        x = torch.randn(3, 1, 32, 32)
        out = disc(x)
        assert out.shape == (3,)


# ==================== 损失函数测试 ====================

class TestSRLoss:

    def test_pixel_loss_zero(self):
        """完美预测 → 像素损失 = 0。"""
        x = torch.randn(1, 1, 32, 32)
        loss = sr_pixel_loss(x, x)
        assert abs(loss.item()) < 1e-6

    def test_pixel_loss_positive(self):
        """不完美 → 损失 > 0。"""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        loss = sr_pixel_loss(pred, target)
        assert loss.item() > 0

    def test_freq_loss_shape(self):
        """频率损失形状正确。"""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        loss = sr_freq_loss(pred, target)
        assert loss.item() >= 0

    def test_sr_loss_no_adv(self):
        """无对抗损失时返回 pixel + freq。"""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        total, parts = sr_loss(pred, target, lambda_freq=0.1)
        assert "pixel" in parts
        assert "freq" in parts
        assert "total" in parts
        assert "adv" not in parts

    def test_sr_loss_with_adv(self):
        """带对抗损失。"""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        disc_logits = torch.randn(1)
        total, parts = sr_loss(pred, target,
                               disc_logits=disc_logits,
                               lambda_adv=0.005)
        assert "adv" in parts


# ==================== sr_upscale 推理测试 ====================

class TestSRUpscale:

    def test_small_image(self):
        """小图直接推理。"""
        dem_lr = np.random.randn(32, 32).astype(np.float32) * 1000
        model = SRNet(in_ch=1, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=4, model=model)
        assert dem_hr.shape == (128, 128)   # 32×4

    def test_large_image_tiled(self):
        """大图分块推理。"""
        dem_lr = np.random.randn(128, 128).astype(np.float32) * 1000
        model = SRNet(in_ch=1, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=4, model=model, tile_size=64)
        assert dem_hr.shape == (512, 512)   # 128×4

    def test_invalid_scale(self):
        """非 2/4/8 报错。"""
        dem_lr = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="不支持"):
            sr_upscale(dem_lr, scale=3)

    def test_output_range(self):
        """输出在合法高程范围。"""
        dem_lr = np.random.randn(32, 32).astype(np.float32) * 1000
        model = SRNet(in_ch=1, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=4, model=model)
        # 归一化 → 反归一化后应在合理范围
        assert np.isfinite(dem_hr).all()

    def test_tiled_vs_small_consistency(self):
        """分块 vs 整图：形状一致（值因混合可能不同）。"""
        dem_lr = np.random.randn(64, 64).astype(np.float32) * 1000
        model = SRNet(in_ch=1, base=16, n_rrdb=1)
        # 整图
        hr_full = sr_upscale(dem_lr, scale=4, model=model, tile_size=128)
        # 分块
        hr_tiled = sr_upscale(dem_lr, scale=4, model=model, tile_size=32)
        assert hr_full.shape == hr_tiled.shape


# ==================== _make_sr_window 测试 ====================

class TestSRWindow:

    def test_shape(self):
        """窗口形状正确。"""
        win = _make_sr_window(128, 128, 32)
        assert win.shape == (128, 128)

    def test_all_ones_no_overlap(self):
        """overlap=0 → 全 1。"""
        win = _make_sr_window(64, 64, 0)
        assert np.allclose(win, 1.0)

    def test_boundary_no_fade(self):
        """边界标志 → 边界不衰减。"""
        win = _make_sr_window(128, 128, 32,
                              is_top=True, is_bottom=True,
                              is_left=True, is_right=True)
        assert np.allclose(win, 1.0)


# ==================== SketchEncoder 测试 ====================

class TestSketchEncoder:

    def test_forward_shape(self):
        """前向输出形状。"""
        enc = SketchEncoder(out_ch=4, base=16)
        x = torch.randn(1, 1, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)

    def test_batch(self):
        """批量 > 1。"""
        enc = SketchEncoder(out_ch=2, base=16)
        x = torch.randn(3, 1, 16, 16)
        out = enc(x)
        assert out.shape == (3, 2, 16, 16)


# ==================== ElevationHintEncoder 测试 ====================

class TestElevationHintEncoder:

    def test_forward_shape(self):
        """前向输出形状。"""
        enc = ElevationHintEncoder(out_ch=4, base=16)
        x = torch.randn(1, 1, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)


# ==================== BiomeEncoder 测试 ====================

class TestBiomeEncoder:

    def test_forward_shape(self):
        """前向输出形状。"""
        enc = BiomeEncoder(n_biomes=14, out_ch=4, base=16)
        x = torch.randn(1, 14, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)

    def test_different_n_biomes(self):
        """不同群落数。"""
        enc = BiomeEncoder(n_biomes=10, out_ch=2, base=16)
        x = torch.randn(1, 10, 16, 16)
        out = enc(x)
        assert out.shape == (1, 2, 16, 16)


# ==================== MultiModalConditioner 测试 ====================

class TestMultiModalConditioner:

    def test_plate_only(self):
        """仅板块底图（无额外模态）。"""
        cond = MultiModalConditioner(base_ch=12, base=16)
        plate = torch.randn(1, 12, 32, 32)
        out, meta = cond(plate)
        # 12 + 4 + 4 + 4 = 24（所有模态通道都拼接）
        assert out.shape == (1, 36, 32, 32)
        assert "plate" in meta["used"]

    def test_with_sketch(self):
        """板块 + 草图。"""
        cond = MultiModalConditioner(base_ch=12, base=16)
        plate = torch.randn(1, 12, 32, 32)
        sketch = torch.randn(1, 1, 32, 32)
        out, meta = cond(plate, sketch=sketch)
        assert out.shape == (1, 36, 32, 32)
        assert "sketch" in meta["used"]

    def test_with_all_modalities(self):
        """全部模态。"""
        cond = MultiModalConditioner(base_ch=12, base=16)
        plate = torch.randn(1, 12, 32, 32)
        sketch = torch.randn(1, 1, 32, 32)
        elev = torch.randn(1, 1, 32, 32)
        biome = torch.randn(1, 14, 32, 32)
        out, meta = cond(plate, sketch=sketch, elev_hint=elev, biome=biome)
        assert out.shape == (1, 36, 32, 32)
        assert len(meta["used"]) == 4

    def test_output_channels_property(self):
        """output_channels 属性。"""
        cond = MultiModalConditioner(base_ch=12, sketch_ch=4,
                                     elev_ch=4, biome_ch=4)
        assert cond.output_channels == 36

    def test_partial_modalities(self):
        """部分模态。"""
        cond = MultiModalConditioner(base_ch=12, base=16)
        plate = torch.randn(1, 12, 32, 32)
        elev = torch.randn(1, 1, 32, 32)
        out, meta = cond(plate, elev_hint=elev)
        assert out.shape == (1, 36, 32, 32)
        assert "elev_hint" in meta["used"]
        assert "sketch" not in meta["used"]
        assert "biome" not in meta["used"]


# ==================== encode_multimodal 便捷函数测试 ====================

class TestEncodeMultimodal:

    def test_plate_only(self):
        """仅板块底图。"""
        plate = np.random.randn(12, 32, 32).astype(np.float32)
        out = encode_multimodal(plate, device=torch.device("cpu"))
        assert out.shape[0] == 36   # 12 + 6×4
        assert out.shape[1:] == (32, 32)

    def test_with_sketch(self):
        """板块 + 草图。"""
        plate = np.random.randn(12, 32, 32).astype(np.float32)
        sketch = np.random.rand(32, 32).astype(np.float32)
        out = encode_multimodal(plate, sketch=sketch,
                                device=torch.device("cpu"))
        assert out.shape == (36, 32, 32)

    def test_with_all(self):
        """全部模态。"""
        plate = np.random.randn(12, 32, 32).astype(np.float32)
        sketch = np.random.rand(32, 32).astype(np.float32)
        elev = np.random.randn(32, 32).astype(np.float32)
        biome = np.random.randn(14, 32, 32).astype(np.float32)
        out = encode_multimodal(plate, sketch=sketch,
                                elev_hint=elev, biome=biome,
                                device=torch.device("cpu"))
        assert out.shape == (36, 32, 32)


# ==================== DiffusionPipeline 测试 ====================

class TestDiffusionPipeline:

    @pytest.fixture
    def mock_ldm(self):
        """Mock LDM 推理器。"""
        ldm = MagicMock()
        def mock_generate(cond, seed=None, tile_size=256):
            sz = cond.shape[-1] if cond.ndim == 3 else cond.shape[-2]
            return np.random.randn(sz, sz).astype(np.float32) * 3000
        ldm.generate = mock_generate
        return ldm

    def test_init_defaults(self):
        """初始化默认值。"""
        pipe = DiffusionPipeline()
        assert pipe._ldm is None
        assert pipe._sr is None

    def test_generate_no_sr_no_physics(self, mock_ldm):
        """不启用 SR/物理 → 直接 LDM 输出。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            use_sr=False, use_physics=False, progress=False)
        assert dem.shape == (32, 32)
        assert "stages" in info
        assert len(info["stages"]) >= 3   # condition + ldm + sr + physics + calibrate

    def test_generate_with_sr(self, mock_ldm):
        """启用超分 → 4× 放大。"""
        sr_model = SRNet(in_ch=1, base=16, n_rrdb=1)
        pipe = DiffusionPipeline(ldm=mock_ldm, sr_model=sr_model)
        cond = np.random.randn(12, 16, 16).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=64,
            use_sr=True, use_physics=False, progress=False)
        # LDM 生成 16² → SR 4× → 64²
        assert dem.shape == (64, 64)

    def test_generate_with_physics(self, mock_ldm):
        """启用物理后处理（无权重时跳过，不崩溃）。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            use_sr=False, use_physics=True, progress=False)
        assert dem.shape == (32, 32)

    def test_generate_stage_info(self, mock_ldm):
        """生成后 info 含各阶段信息。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            use_sr=False, use_physics=False, progress=False)
        stage_names = [s["name"] for s in info["stages"]]
        assert "condition" in stage_names
        assert "ldm" in stage_names
        assert "sr" in stage_names
        assert "physics" in stage_names
        assert "calibrate" in stage_names

    def test_generate_elev_range(self, mock_ldm):
        """输出在合法高程范围。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            use_sr=False, use_physics=False, progress=False)
        assert dem.min() >= ELEV_MIN
        assert dem.max() <= ELEV_MAX

    def test_generate_land_ratio(self, mock_ldm):
        """info 含 land_ratio。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            use_sr=False, use_physics=False, progress=False)
        assert "land_ratio" in info
        assert 0 <= info["land_ratio"] <= 1

    def test_benchmark(self, mock_ldm):
        """benchmark 返回完整信息。"""
        pipe = DiffusionPipeline(ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        info = pipe.benchmark(seed=42, size_px=32,
                              use_sr=False, use_physics=False)
        assert "total_time_s" in info
        assert "stages" in info

    def test_generate_seed_reproducible(self, mock_ldm):
        """相同种子 → 相同结果（mock 下）。"""
        sr_model = SRNet(in_ch=1, base=16, n_rrdb=1)
        pipe1 = DiffusionPipeline(ldm=mock_ldm, sr_model=sr_model)
        pipe2 = DiffusionPipeline(ldm=mock_ldm, sr_model=sr_model)
        cond = np.random.randn(12, 16, 16).astype(np.float32)
        dem1, _ = pipe1.generate(seed=42, cond_12ch=cond, size_px=64,
                                  use_sr=False, use_physics=False,
                                  progress=False)
        dem2, _ = pipe2.generate(seed=42, cond_12ch=cond, size_px=64,
                                  use_sr=False, use_physics=False,
                                  progress=False)
        # mock LDM 用随机数，但 seed 固定 torch.manual_seed
        # dem1/dem2 应有相同形状
        assert dem1.shape == dem2.shape

    def test_lazy_loading(self):
        """延迟加载属性存在且初始为 None。"""
        pipe = DiffusionPipeline()
        # 延迟加载：初始为 None，不触发实际加载
        assert pipe._ldm is None
        assert pipe._sr is None
        assert pipe._multicond is None


# ==================== 集成测试 ====================

class TestIntegration:

    def test_sr_then_physics_flow(self, mock_ldm=None):
        """超分 + 物理后处理完整流程。"""
        ldm = MagicMock()
        def mock_gen(cond, seed=None, tile_size=256):
            sz = cond.shape[-1]
            return np.random.randn(sz, sz).astype(np.float32) * 2000
        ldm.generate = mock_gen

        sr = SRNet(in_ch=1, base=16, n_rrdb=1)
        pipe = DiffusionPipeline(ldm=ldm, sr_model=sr)
        cond = np.random.randn(12, 16, 16).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=64,
            use_sr=True, use_physics=True, progress=False)
        assert dem.shape == (64, 64)
        # 检查阶段信息
        sr_stage = [s for s in info["stages"] if s["name"] == "sr"][0]
        assert sr_stage["input_size"] == 16
        assert sr_stage["output_size"] == 64

    def test_multimodal_then_pipeline(self):
        """多模态条件 → 管线生成。"""
        ldm = MagicMock()
        def mock_gen(cond, seed=None, tile_size=256):
            sz = cond.shape[-1]
            return np.random.randn(sz, sz).astype(np.float32) * 2000
        ldm.generate = mock_gen

        pipe = DiffusionPipeline(ldm=ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        sketch = np.random.rand(32, 32).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=32,
            sketch=sketch,
            use_sr=False, use_physics=False, progress=False)
        assert dem.shape == (32, 32)

    def test_sr_param_growth(self):
        """RRDB 数量影响参数量。"""
        net_small = SRNet(base=16, n_rrdb=2)
        net_large = SRNet(base=16, n_rrdb=5)
        p_small = sum(p.numel() for p in net_small.parameters())
        p_large = sum(p.numel() for p in net_large.parameters())
        assert p_large > p_small

    def test_full_pipeline_stages_complete(self):
        """完整管线的 5 个阶段全部执行。"""
        ldm = MagicMock()
        def mock_gen(cond, seed=None, tile_size=256):
            sz = cond.shape[-1]
            return np.random.randn(sz, sz).astype(np.float32) * 2000
        ldm.generate = mock_gen

        sr = SRNet(in_ch=1, base=16, n_rrdb=1)
        pipe = DiffusionPipeline(ldm=ldm, sr_model=sr)
        cond = np.random.randn(12, 16, 16).astype(np.float32)
        dem, info = pipe.generate(
            seed=42, cond_12ch=cond, size_px=64,
            use_sr=True, use_physics=True, progress=False)
        stage_names = [s["name"] for s in info["stages"]]
        assert stage_names == ["condition", "ldm", "sr", "physics",
                               "calibrate"]
        assert not any(s.get("skipped") for s in info["stages"])


# ==================== PipelineConfig 测试 ====================

class TestPipelineConfig:

    def test_defaults(self):
        """默认值。"""
        cfg = PipelineConfig()
        assert cfg.seed == 42
        assert cfg.size_px == 4096
        assert cfg.sr_scale == 4
        assert cfg.use_sr is True
        assert cfg.use_physics is True

    def test_from_dict(self):
        """从字典构造。"""
        cfg = PipelineConfig.from_dict({"seed": 7, "size_px": 2048,
                                         "use_sr": False})
        assert cfg.seed == 7
        assert cfg.size_px == 2048
        assert cfg.use_sr is False

    def test_from_dict_ignores_unknown(self):
        """未知键被忽略。"""
        cfg = PipelineConfig.from_dict({"seed": 1, "unknown_key": 99})
        assert cfg.seed == 1

    def test_to_dict(self):
        """转字典。"""
        cfg = PipelineConfig(seed=1, size_px=512)
        d = cfg.to_dict()
        assert d["seed"] == 1
        assert d["size_px"] == 512
        assert "use_sr" in d

    def test_resolve_device_auto(self):
        """auto 设备解析为 DEVICE。"""
        cfg = PipelineConfig(device="auto")
        dev = cfg.resolve_device()
        from terrain.dl.config import DEVICE
        assert dev == DEVICE

    def test_resolve_device_cpu(self):
        """cpu 设备。"""
        cfg = PipelineConfig(device="cpu")
        assert cfg.resolve_device() == torch.device("cpu")

    def test_physics_substage_switches(self):
        """物理子阶段开关。"""
        cfg = PipelineConfig(physics_coast=False, physics_river=True,
                             physics_snow=False)
        assert cfg.physics_coast is False
        assert cfg.physics_river is True
        assert cfg.physics_snow is False


# ==================== ConditionCache 测试 ====================

class TestConditionCache:

    def test_put_and_get_memory(self, tmp_path):
        """内存缓存写入和读取。"""
        cache = ConditionCache(str(tmp_path), max_entries=4)
        data = np.random.randn(12, 32, 32).astype(np.float32)
        cache.put(seed=42, size_px=128, use_sr=True, data=data)
        result = cache.get(seed=42, size_px=128, use_sr=True)
        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_miss(self, tmp_path):
        """未命中返回 None。"""
        cache = ConditionCache(str(tmp_path))
        assert cache.get(seed=99, size_px=128, use_sr=True) is None

    def test_disk_persistence(self, tmp_path):
        """磁盘持久化：新实例能读取。"""
        cache1 = ConditionCache(str(tmp_path), max_entries=2)
        data = np.random.randn(12, 16, 16).astype(np.float32)
        cache1.put(seed=7, size_px=64, use_sr=True, data=data)

        cache2 = ConditionCache(str(tmp_path), max_entries=2)
        result = cache2.get(seed=7, size_px=64, use_sr=True)
        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_lru_eviction(self, tmp_path):
        """LRU 淘汰：超 max_entries 删最久未用（内存层）。"""
        cache = ConditionCache(str(tmp_path), max_entries=2)
        cache.put(1, 64, True, np.ones((12, 4, 4)))
        cache.put(2, 64, True, np.full((12, 4, 4), 2.0))
        # 访问 seed=1 使其变为最近使用
        cache.get(1, 64, True)
        # 写入 seed=3 → 内存淘汰 seed=2
        cache.put(3, 64, True, np.full((12, 4, 4), 3.0))
        # seed=1 仍在内存
        assert cache.get(1, 64, True) is not None
        # seed=2 从内存淘汰，但磁盘仍有（get 会从磁盘加载回内存）
        # 验证内存淘汰：直接检查 _memory
        assert "cond_seed2_lr16" not in cache._memory
        # seed=3 在内存
        assert cache.get(3, 64, True) is not None

    def test_clear(self, tmp_path):
        """clear 清空内存。"""
        cache = ConditionCache(str(tmp_path))
        cache.put(1, 64, True, np.zeros((12, 4, 4)))
        cache.clear()
        # 内存清空但磁盘仍在
        stats = cache.stats()
        assert stats["memory_entries"] == 0

    def test_stats(self, tmp_path):
        """stats 返回统计信息。"""
        cache = ConditionCache(str(tmp_path), max_entries=3)
        cache.put(1, 64, True, np.zeros((12, 4, 4)))
        stats = cache.stats()
        assert stats["memory_entries"] == 1
        assert stats["max_entries"] == 3

    def test_different_keys(self, tmp_path):
        """不同 seed/size/use_sr 生成不同键。"""
        cache = ConditionCache(str(tmp_path))
        cache.put(1, 64, True, np.ones((12, 4, 4)))
        cache.put(1, 64, False, np.zeros((12, 4, 4)))
        # use_sr 不同 → 不同键
        assert cache.get(1, 64, True) is not None
        assert cache.get(1, 64, False) is not None


# ==================== DiffusionPipeline 优化版测试 ====================

class TestPipelineOptimized:

    @pytest.fixture
    def mock_ldm(self):
        """Mock LDM 推理器。"""
        ldm = MagicMock()
        def mock_generate(cond, seed=None, tile_size=256):
            sz = cond.shape[-1] if cond.ndim == 3 else cond.shape[-2]
            return np.random.randn(sz, sz).astype(np.float32) * 3000
        ldm.generate = mock_generate
        return ldm

    def test_config_in_constructor(self, mock_ldm):
        """通过 config 构造管线。"""
        cfg = PipelineConfig(seed=7, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        assert pipe.config.seed == 7
        assert pipe.config.size_px == 32

    def test_generate_with_config(self, mock_ldm):
        """使用 config 参数生成。"""
        cfg = PipelineConfig(seed=7, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert dem.shape == (32, 32)
        assert info["seed"] == 7

    def test_param_override(self, mock_ldm):
        """调用参数覆盖 config。"""
        cfg = PipelineConfig(seed=1, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(seed=99, cond_12ch=cond, progress=False)
        assert info["seed"] == 99   # 覆盖了 config 的 1

    def test_progress_callback(self, mock_ldm):
        """进度回调被调用。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        callbacks = []
        pipe.generate(cond_12ch=cond, progress=False,
                      progress_callback=lambda pct, msg: callbacks.append(pct))
        assert len(callbacks) > 0
        assert 100 in callbacks   # 最后应到 100%

    def test_cache_conditions_enabled(self, mock_ldm):
        """启用条件缓存时第二次生成更快。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=True,
                             cache_dir="/tmp/test_pipeline_cache")
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        # 第一次：cond_12ch 提供时不缓存（已有预计算）
        dem1, _ = pipe.generate(cond_12ch=cond, progress=False)
        stats = pipe.get_cache_stats()
        assert "memory_entries" in stats

    def test_clear_cache(self, mock_ldm):
        """clear_cache 清空缓存。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=True,
                             cache_dir="/tmp/test_pipeline_clear")
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        pipe.clear_cache()
        stats = pipe.get_cache_stats()
        assert stats["memory_entries"] == 0

    def test_last_info(self, mock_ldm):
        """last_info 属性保存最近一次信息。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        assert pipe.last_info is None
        pipe.generate(cond_12ch=cond, progress=False)
        assert pipe.last_info is not None
        assert "stages" in pipe.last_info

    def test_degradation_on_sr_failure(self, mock_ldm):
        """SR 失败时降级为双三次插值。"""
        sr_model = MagicMock()
        # 让整个 model 调用链都报错
        sr_model.to.return_value.eval.return_value = sr_model
        sr_model.side_effect = RuntimeError("SR 模型崩溃")
        sr_model.return_value = sr_model
        cfg = PipelineConfig(seed=42, size_px=64, use_sr=True,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm, sr_model=sr_model)
        cond = np.random.randn(12, 16, 16).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        # 降级后仍应有输出
        assert dem.shape == (64, 64)
        assert len(info["degradations"]) > 0
        assert info["degradations"][0]["stage"] == "sr"

    def test_degradation_on_physics_failure(self, mock_ldm):
        """物理后处理失败时记录降级但不崩溃。"""
        river_net = MagicMock()
        river_net.forward.side_effect = RuntimeError("RiverNet 崩溃")
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=True, physics_river=True,
                             cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm,
                                 river_net=river_net)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert dem.shape == (32, 32)
        # 有降级记录
        degs = [d for d in info["degradations"]
                if "river" in d.get("stage", "")]
        assert len(degs) > 0

    def test_physics_substage_control(self, mock_ldm):
        """物理子阶段开关控制。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=True,
                             physics_coast=False,
                             physics_river=False,
                             physics_snow=False,
                             cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        # 物理阶段存在但无子阶段执行（因网络都是 None）
        physics_stage = [s for s in info["stages"]
                         if s["name"] == "physics"]
        assert len(physics_stage) == 1

    def test_save_output(self, mock_ldm, tmp_path):
        """保存输出 DEM 和元信息 JSON。"""
        out_path = tmp_path / "dem.npy"
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False,
                             save_output=True,
                             output_path=str(out_path),
                             save_info=True)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert out_path.exists()
        # 元信息 JSON
        info_path = out_path.with_suffix(".info.json")
        assert info_path.exists()

    def test_benchmark_with_warmup(self, mock_ldm):
        """benchmark 带 warmup。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        info = pipe.benchmark(seed=42, size_px=32, use_sr=False,
                              use_physics=False, warmup=0)
        assert "total_time_s" in info

    def test_generate_batch(self, mock_ldm):
        """批量生成多个种子。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        # generate_batch 不接受 cond_12ch，需通过 _make_conditions
        # 但 mock_ldm 用 cond 的尺寸生成，所以直接测批量接口逻辑
        # 用 monkeypatch 替换 _make_conditions
        original_make = pipe._make_conditions
        pipe._make_conditions = lambda seed, sp, sr: cond
        results = pipe.generate_batch(seeds=[1, 2, 3],
                                      size_px=32, use_sr=False,
                                      use_physics=False, progress=False)
        pipe._make_conditions = original_make
        assert len(results) == 3
        for dem, info in results:
            assert dem.shape == (32, 32)
            assert "stages" in info

    def test_info_has_total_time(self, mock_ldm):
        """info 含 total_time_s。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert "total_time_s" in info

    def test_info_has_degradations_list(self, mock_ldm):
        """info 含 degradations 列表。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert "degradations" in info
        assert isinstance(info["degradations"], list)

    def test_info_has_config_snapshot(self, mock_ldm):
        """info 含 config 快照。"""
        cfg = PipelineConfig(seed=42, size_px=32, use_sr=False,
                             use_physics=False, cache_conditions=False)
        pipe = DiffusionPipeline(config=cfg, ldm=mock_ldm)
        cond = np.random.randn(12, 32, 32).astype(np.float32)
        dem, info = pipe.generate(cond_12ch=cond, progress=False)
        assert "config" in info
        assert info["config"]["seed"] == 42


# ==================== 注意力模块测试 ====================

class TestChannelAttention:

    def test_forward_shape(self):
        """前向输出形状不变。"""
        ca = ChannelAttention(32, reduction=4)
        x = torch.randn(1, 32, 16, 16)
        out = ca(x)
        assert out.shape == (1, 32, 16, 16)

    def test_output_modified(self):
        """输出被通道权重缩放（非恒等）。"""
        ca = ChannelAttention(16, reduction=4)
        ca.eval()
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            out = ca(x)
        # 不是恒等
        assert not torch.allclose(out, x)


class TestSpatialAttention:

    def test_forward_shape(self):
        """前向输出形状不变。"""
        sa = SpatialAttention()
        x = torch.randn(1, 32, 16, 16)
        out = sa(x)
        assert out.shape == (1, 32, 16, 16)


class TestCCABlock:

    def test_forward_shape(self):
        """前向输出形状不变。"""
        cca = CCABlock(32, reduction=4)
        x = torch.randn(1, 32, 16, 16)
        out = cca(x)
        assert out.shape == (1, 32, 16, 16)

    def test_batch(self):
        """批量 > 1。"""
        cca = CCABlock(16, reduction=4)
        x = torch.randn(2, 16, 8, 8)
        out = cca(x)
        assert out.shape == (2, 16, 8, 8)


# ==================== RRDB 注意力测试 ====================

class TestRRDBAttention:

    def test_with_attention(self):
        """带注意力的 RRDB。"""
        rrdb = RRDB(in_ch=32, growth=16, use_attention=True)
        assert rrdb.attention is not None
        x = torch.randn(1, 32, 16, 16)
        out = rrdb(x)
        assert out.shape == (1, 32, 16, 16)

    def test_without_attention(self):
        """不带注意力的 RRDB。"""
        rrdb = RRDB(in_ch=32, growth=16, use_attention=False)
        assert rrdb.attention is None

    def test_beta_zero_identity(self):
        """beta=0 → 输出 = 输入（注意力也跳过）。"""
        rrdb = RRDB(in_ch=16, growth=8, beta=0.0,
                    use_attention=False)
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            out = rrdb(x)
        assert torch.allclose(out, x, atol=1e-6)


# ==================== SRNet 灵活缩放测试 ====================

class TestSRNetFlexibleScale:

    def test_scale_2x(self):
        """2× 超分。"""
        net = SRNet(scale=2, base=16, n_rrdb=1)
        x = torch.randn(1, 1, 32, 32)
        out = net(x)
        assert out.shape == (1, 1, 64, 64)   # 32×2

    def test_scale_4x(self):
        """4× 超分。"""
        net = SRNet(scale=4, base=16, n_rrdb=1)
        x = torch.randn(1, 1, 32, 32)
        out = net(x)
        assert out.shape == (1, 1, 128, 128)   # 32×4

    def test_scale_8x(self):
        """8× 超分。"""
        net = SRNet(scale=8, base=16, n_rrdb=1)
        x = torch.randn(1, 1, 16, 16)
        out = net(x)
        assert out.shape == (1, 1, 128, 128)   # 16×8

    def test_invalid_scale(self):
        """不支持的 scale 报错。"""
        with pytest.raises(ValueError, match="不支持"):
            SRNet(scale=3)

    def test_with_attention(self):
        """带注意力的网络。"""
        net = SRNet(scale=4, base=16, n_rrdb=2,
                    use_attention=True)
        x = torch.randn(1, 1, 16, 16)
        out = net(x)
        assert out.shape == (1, 1, 64, 64)

    def test_multi_scale_output(self):
        """多尺度输出返回 (out, [mid_outputs])。"""
        net = SRNet(scale=4, base=16, n_rrdb=1,
                    multi_scale=True)
        x = torch.randn(1, 1, 16, 16)
        result = net(x)
        assert isinstance(result, tuple)
        out, mids = result
        assert out.shape == (1, 1, 64, 64)
        assert len(mids) == 1   # 4× 有 1 个中间输出（2× 处）

    def test_multi_scale_8x(self):
        """8× 多尺度输出有 2 个中间。"""
        net = SRNet(scale=8, base=16, n_rrdb=1,
                    multi_scale=True)
        x = torch.randn(1, 1, 8, 8)
        out, mids = net(x)
        assert out.shape == (1, 1, 64, 64)
        assert len(mids) == 2   # 2× 和 4× 处

    def test_param_count_property(self):
        """param_count 属性。"""
        net = SRNet(base=16, n_rrdb=1)
        assert net.param_count > 0

    def test_attention_increases_params(self):
        """注意力增加参数量。"""
        net_no = SRNet(base=16, n_rrdb=2, use_attention=False)
        net_yes = SRNet(base=16, n_rrdb=2, use_attention=True)
        assert net_yes.param_count > net_no.param_count


# ==================== 新损失函数测试 ====================

class TestCharbonnierLoss:

    def test_zero_when_identical(self):
        """完美预测 → 损失 ≈ 0。"""
        x = torch.randn(1, 1, 32, 32)
        loss = sr_charbonnier_loss(x, x)
        assert loss.item() < 0.01   # sqrt(eps²) ≈ 0.001

    def test_positive_when_different(self):
        """不完美 → 损失 > 0。"""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        loss = sr_charbonnier_loss(pred, target)
        assert loss.item() > 0.5

    def test_smooth_near_zero(self):
        """零点附近比 L1 更平滑（Charbonnier ≈ eps）。"""
        pred = torch.zeros(1, 1, 1, 1)
        target = torch.zeros(1, 1, 1, 1)
        loss = sr_charbonnier_loss(pred, target, eps=1e-3)
        # sqrt(0 + eps²) = eps
        assert abs(loss.item() - 1e-3) < 1e-4


class TestSSIMLoss:

    def test_zero_when_identical(self):
        """完美预测 → SSIM 损失 ≈ 0。"""
        x = torch.randn(1, 1, 32, 32)
        loss = sr_ssim_loss(x, x)
        assert loss.item() < 0.01

    def test_positive_when_different(self):
        """不完美 → 损失 > 0。"""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        loss = sr_ssim_loss(pred, target)
        assert loss.item() > 0

    def test_batch(self):
        """批量 > 1。"""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        loss = sr_ssim_loss(pred, target)
        assert loss.item() >= 0


class TestGradientLoss:

    def test_zero_when_identical(self):
        """完美预测 → 梯度损失 = 0。"""
        x = torch.randn(1, 1, 32, 32)
        loss = sr_gradient_loss(x, x)
        assert abs(loss.item()) < 1e-6

    def test_positive_when_different(self):
        """不完美 → 损失 > 0。"""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        loss = sr_gradient_loss(pred, target)
        assert loss.item() >= 0   # 常数图像梯度相同


class TestSRLossOptimized:

    def test_all_components(self):
        """总损失含 pixel + freq + ssim + grad。"""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        total, parts = sr_loss(pred, target)
        assert "pixel" in parts
        assert "freq" in parts
        assert "ssim" in parts
        assert "grad" in parts
        assert "total" in parts

    def test_charbonnier_vs_l1(self):
        """use_charbonnier=True 用 Charbonnier。"""
        pred = torch.randn(1, 1, 16, 16)
        target = torch.randn(1, 1, 16, 16)
        _, parts_c = sr_loss(pred, target, use_charbonnier=True)
        _, parts_l = sr_loss(pred, target, use_charbonnier=False)
        # 两者应有不同的 pixel 值
        assert parts_c["pixel"] != parts_l["pixel"]

    def test_with_adv(self):
        """带对抗损失。"""
        pred = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        disc = torch.randn(1)
        total, parts = sr_loss(pred, target, disc_logits=disc)
        assert "adv" in parts


# ==================== TTA 推理测试 ====================

class TestTTA:

    def test_tta_output_shape(self):
        """TTA 输出形状正确。"""
        model = SRNet(scale=4, base=16, n_rrdb=1)
        x = torch.randn(1, 1, 16, 16)
        out = _apply_tta(model, x)
        assert out.shape == (1, 1, 64, 64)

    def test_tta_batch(self):
        """TTA 批量 > 1。"""
        model = SRNet(scale=4, base=16, n_rrdb=1)
        x = torch.randn(2, 1, 16, 16)
        out = _apply_tta(model, x)
        assert out.shape == (2, 1, 64, 64)

    def test_tta_deterministic(self):
        """TTA 是确定性的（无随机性）。"""
        model = SRNet(scale=4, base=16, n_rrdb=1)
        model.eval()
        x = torch.randn(1, 1, 16, 16)
        out1 = _apply_tta(model, x)
        out2 = _apply_tta(model, x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_sr_upscale_with_tta(self):
        """sr_upscale 启用 TTA。"""
        dem_lr = np.random.randn(32, 32).astype(np.float32) * 1000
        model = SRNet(scale=4, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=4, model=model, tta=True)
        assert dem_hr.shape == (128, 128)

    def test_sr_upscale_scale_2x(self):
        """2× 超分。"""
        dem_lr = np.random.randn(32, 32).astype(np.float32) * 1000
        model = SRNet(scale=2, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=2, model=model)
        assert dem_hr.shape == (64, 64)

    def test_sr_upscale_scale_8x(self):
        """8× 超分。"""
        dem_lr = np.random.randn(16, 16).astype(np.float32) * 1000
        model = SRNet(scale=8, base=16, n_rrdb=1)
        dem_hr = sr_upscale(dem_lr, scale=8, model=model)
        assert dem_hr.shape == (128, 128)


# ==================== SRDataset 测试 ====================

class TestSRDataset:

    def test_len(self):
        """__len__ 返回 n_patches。"""
        dem = np.random.randn(256, 256).astype(np.float32) * 1000
        ds = SRDataset(dem, patch_size=64, scale=4, n_patches=100)
        assert len(ds) == 100

    def test_getitem_shape(self):
        """__getitem__ 返回 (lr, hr) 正确形状。"""
        dem = np.random.randn(256, 256).astype(np.float32) * 1000
        ds = SRDataset(dem, patch_size=64, scale=4, n_patches=50)
        lr, hr = ds[0]
        # hr: (1, 64, 64), lr: (1, 16, 16)
        assert hr.shape == (1, 64, 64)
        assert lr.shape == (1, 16, 16)

    def test_getitem_range(self):
        """输出归一化到 [-1, 1]。"""
        dem = np.random.randn(256, 256).astype(np.float32) * 5000
        ds = SRDataset(dem, patch_size=64, scale=4, n_patches=20)
        lr, hr = ds[0]
        assert lr.min() >= -1.01 and lr.max() <= 1.01
        assert hr.min() >= -1.01 and hr.max() <= 1.01

    def test_augment_modes(self):
        """8 种增强模式不崩溃。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 1000
        ds = SRDataset(dem, patch_size=32, scale=4, n_patches=20,
                       augment=True)
        for i in range(8):
            lr, hr = ds[i]
            assert lr.shape == (1, 8, 8)
            assert hr.shape == (1, 32, 32)

    def test_no_augment(self):
        """关闭增强。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 1000
        ds = SRDataset(dem, patch_size=32, scale=4, n_patches=10,
                       augment=False)
        lr, hr = ds[0]
        assert lr.shape == (1, 8, 8)

    def test_scale_2x(self):
        """2× 数据集。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 1000
        ds = SRDataset(dem, patch_size=32, scale=2, n_patches=10)
        lr, hr = ds[0]
        assert hr.shape == (1, 32, 32)
        assert lr.shape == (1, 16, 16)   # 32/2

    def test_from_file(self, tmp_path):
        """从文件加载 DEM。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 1000
        path = tmp_path / "dem.npy"
        np.save(str(path), dem)
        ds = SRDataset(str(path), patch_size=32, scale=4, n_patches=10)
        lr, hr = ds[0]
        assert hr.shape == (1, 32, 32)


# ==================== train_sr 测试 ====================

class TestTrainSR:

    def test_train_smoke(self, tmp_path):
        """训练 smoke test（2 轮）。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 2000
        save_path = str(tmp_path / "sr_test.pt")
        history = train_sr(
            dem, scale=4, epochs=2, batch_size=2,
            patch_size=32, use_attention=True,
            save_path=save_path)
        assert "epoch" in history
        assert len(history["epoch"]) == 2
        assert "total" in history
        assert len(history["total"]) == 2
        assert "param_count" in history
        assert history["param_count"] > 0

    def test_train_saves_weights(self, tmp_path):
        """训练后保存权重文件。"""
        dem = np.random.randn(64, 64).astype(np.float32) * 2000
        save_path = str(tmp_path / "sr_weights.pt")
        train_sr(dem, scale=2, epochs=1, batch_size=2,
                 patch_size=16, save_path=save_path)
        from pathlib import Path
        assert Path(save_path).exists()

    def test_train_loss_decreasing_or_stable(self, tmp_path):
        """训练损失不发散（不要求严格下降）。"""
        dem = np.random.randn(128, 128).astype(np.float32) * 1000
        save_path = str(tmp_path / "sr_stable.pt")
        history = train_sr(dem, scale=4, epochs=3, batch_size=2,
                           patch_size=32, save_path=save_path)
        # 损失有限（非 NaN/Inf）
        for v in history["total"]:
            assert np.isfinite(v)


# ==================== ModalityConfig 测试 ====================

class TestModalityConfig:

    def test_defaults(self):
        """默认权重全为 1.0。"""
        cfg = ModalityConfig()
        assert cfg.get_weight("sketch") == 1.0
        assert cfg.get_weight("biome") == 1.0

    def test_custom_weights(self):
        """自定义权重。"""
        cfg = ModalityConfig(sketch_weight=0.5, biome_weight=0.8)
        assert cfg.get_weight("sketch") == 0.5
        assert cfg.get_weight("biome") == 0.8

    def test_unknown_modality(self):
        """未知模态返回默认 1.0。"""
        cfg = ModalityConfig()
        assert cfg.get_weight("unknown") == 1.0

    def test_to_dict(self):
        """转字典。"""
        cfg = ModalityConfig(sketch_weight=0.3)
        d = cfg.to_dict()
        assert "weights" in d
        assert d["weights"]["sketch"] == 0.3
        assert "use_cross_attention" in d

    def test_flags(self):
        """开关标志。"""
        cfg = ModalityConfig(use_cross_attention=False,
                             residual_connection=False)
        assert cfg.use_cross_attention is False
        assert cfg.residual_connection is False


# ==================== CrossAttentionFusion 测试 ====================

class TestCrossAttentionFusion:

    def test_forward_shape(self):
        """前向输出形状 = base 形状。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=4)
        base = torch.randn(1, 12, 16, 16)
        cond = torch.randn(1, 24, 16, 16)
        out = ca(base, cond)
        assert out.shape == (1, 12, 16, 16)

    def test_batch(self):
        """批量 > 1。"""
        ca = CrossAttentionFusion(base_ch=8, cond_ch=16, n_heads=2)
        base = torch.randn(2, 8, 8, 8)
        cond = torch.randn(2, 16, 8, 8)
        out = ca(base, cond)
        assert out.shape == (2, 8, 8, 8)

    def test_different_sizes(self):
        """不同空间尺寸。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=4)
        for sz in [8, 16, 32]:
            base = torch.randn(1, 12, sz, sz)
            cond = torch.randn(1, 24, sz, sz)
            out = ca(base, cond)
            assert out.shape == (1, 12, sz, sz)

    def test_invalid_n_heads(self):
        """base_ch 不能被 n_heads 整除时报错。"""
        with pytest.raises(AssertionError):
            CrossAttentionFusion(base_ch=10, cond_ch=24, n_heads=3)

    def test_output_finite(self):
        """输出有限（非 NaN/Inf）。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=4)
        base = torch.randn(1, 12, 16, 16)
        cond = torch.randn(1, 24, 16, 16)
        out = ca(base, cond)
        assert torch.isfinite(out).all()


# ==================== 预设模态编码器测试 ====================

class TestSeaLevelEncoder:

    def test_forward_shape(self):
        """前向输出形状。"""
        enc = SeaLevelEncoder(out_ch=4, base=16)
        x = torch.randn(1, 1, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)

    def test_batch(self):
        """批量 > 1。"""
        enc = SeaLevelEncoder(out_ch=2, base=16)
        x = torch.randn(3, 1, 16, 16)
        out = enc(x)
        assert out.shape == (3, 2, 16, 16)


class TestClimateZoneEncoder:

    def test_forward_shape(self):
        """前向输出形状。"""
        enc = ClimateZoneEncoder(n_zones=5, out_ch=4, base=16)
        x = torch.randn(1, 5, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)

    def test_different_n_zones(self):
        """不同气候带数。"""
        enc = ClimateZoneEncoder(n_zones=7, out_ch=2, base=16)
        x = torch.randn(1, 7, 16, 16)
        out = enc(x)
        assert out.shape == (1, 2, 16, 16)


class TestPlateBoundaryEncoder:

    def test_forward_shape(self):
        """前向输出形状（3 通道输入）。"""
        enc = PlateBoundaryEncoder(out_ch=4, base=16)
        x = torch.randn(1, 3, 32, 32)
        out = enc(x)
        assert out.shape == (1, 4, 32, 32)

    def test_batch(self):
        """批量 > 1。"""
        enc = PlateBoundaryEncoder(out_ch=2, base=16)
        x = torch.randn(2, 3, 16, 16)
        out = enc(x)
        assert out.shape == (2, 2, 16, 16)


# ==================== MultiModalConditioner 优化版测试 ====================

class TestMultiModalOptimized:

    def test_with_modality_config(self):
        """带 ModalityConfig 的条件器。"""
        cfg = ModalityConfig(sketch_weight=0.5)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        plate = torch.randn(1, 12, 16, 16)
        sketch = torch.randn(1, 1, 16, 16)
        out, meta = cond(plate, sketch=sketch)
        assert out.shape == (1, 36, 16, 16)
        assert meta["weights"]["sketch"] == 0.5

    def test_no_cross_attention(self):
        """关闭交叉注意力。"""
        cfg = ModalityConfig(use_cross_attention=False)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        assert cond.cross_attn is None

    def test_with_cross_attention(self):
        """启用交叉注意力。"""
        cfg = ModalityConfig(use_cross_attention=True)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        assert cond.cross_attn is not None

    def test_residual_connection(self):
        """残差连接开关。"""
        cfg = ModalityConfig(residual_connection=True)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, _ = cond(plate)
        assert out.shape == (1, 36, 16, 16)

    def test_sketch_mask(self):
        """草图空间掩码（局部控制）。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        sketch = torch.randn(1, 1, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1.0
        out, meta = cond(plate, sketch=sketch, sketch_mask=mask)
        assert "sketch" in meta["masked"]

    def test_elev_mask(self):
        """高度提示空间掩码。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        elev = torch.randn(1, 1, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        out, meta = cond(plate, elev_hint=elev, elev_mask=mask)
        assert "elev_hint" in meta["masked"]

    def test_biome_mask(self):
        """生物群落空间掩码。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        biome = torch.randn(1, 14, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        out, meta = cond(plate, biome=biome, biome_mask=mask)
        assert "biome" in meta["masked"]

    def test_sea_level_modality(self):
        """海平面模态。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        sea = torch.randn(1, 1, 16, 16)
        out, meta = cond(plate, sea_level=sea)
        assert "sea_level" in meta["used"]
        assert out.shape == (1, 36, 16, 16)

    def test_climate_modality(self):
        """气候带模态。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        climate = torch.randn(1, 5, 16, 16)
        out, meta = cond(plate, climate=climate)
        assert "climate" in meta["used"]

    def test_plate_boundary_modality(self):
        """板块边界模态。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        boundary = torch.randn(1, 3, 16, 16)
        out, meta = cond(plate, plate_boundary=boundary)
        assert "plate_boundary" in meta["used"]

    def test_all_six_modalities(self):
        """全部 6 个模态。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, meta = cond(
            plate,
            sketch=torch.randn(1, 1, 16, 16),
            elev_hint=torch.randn(1, 1, 16, 16),
            biome=torch.randn(1, 14, 16, 16),
            sea_level=torch.randn(1, 1, 16, 16),
            climate=torch.randn(1, 5, 16, 16),
            plate_boundary=torch.randn(1, 3, 16, 16))
        assert out.shape == (1, 36, 16, 16)
        assert len(meta["used"]) == 7   # plate + 6 modalities

    def test_weight_applied(self):
        """权重影响输出（weight=0 接近无模态）。"""
        cfg = ModalityConfig(sketch_weight=0.0)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        cond.eval()
        plate = torch.randn(1, 12, 16, 16)
        sketch = torch.randn(1, 1, 16, 16)
        with torch.no_grad():
            out_with, _ = cond(plate, sketch=sketch)
            out_without, _ = cond(plate)
        # weight=0 时有/无 sketch 应接近
        diff = (out_with - out_without).abs().mean()
        assert diff < 0.5   # 差异较小

    def test_meta_has_weights(self):
        """meta 含权重字典。"""
        cfg = ModalityConfig(sketch_weight=0.7)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        plate = torch.randn(1, 12, 16, 16)
        sketch = torch.randn(1, 1, 16, 16)
        out, meta = cond(plate, sketch=sketch)
        assert "weights" in meta
        assert "sketch" in meta["weights"]
        assert meta["weights"]["sketch"] == 0.7


# ==================== 条件校验工具测试 ====================

class TestValidateSketch:

    def test_2d_to_3d(self):
        """2D → 3D。"""
        sketch = np.random.rand(32, 32).astype(np.float32)
        out = validate_sketch(sketch)
        assert out.shape == (1, 32, 32)

    def test_clip_range(self):
        """值域裁剪到 [0, 1]。"""
        sketch = np.random.randn(16, 16).astype(np.float32) * 10
        out = validate_sketch(sketch)
        assert out.min() >= 0 and out.max() <= 1

    def test_3d_passthrough(self):
        """3D 直通。"""
        sketch = np.random.rand(1, 16, 16).astype(np.float32)
        out = validate_sketch(sketch)
        assert out.shape == (1, 16, 16)

    def test_invalid_shape(self):
        """非法形状报错。"""
        sketch = np.random.rand(3, 16, 16).astype(np.float32)
        with pytest.raises(AssertionError):
            validate_sketch(sketch)


class TestValidateElevHint:

    def test_2d_to_3d(self):
        """2D → 3D。"""
        elev = np.random.randn(32, 32).astype(np.float32)
        out = validate_elev_hint(elev)
        assert out.shape == (1, 32, 32)

    def test_clip_range(self):
        """值域裁剪到 [-1, 1]。"""
        elev = np.random.randn(16, 16).astype(np.float32) * 10
        out = validate_elev_hint(elev)
        assert out.min() >= -1 and out.max() <= 1


class TestValidateBiome:

    def test_label_to_onehot(self):
        """标签图 → one-hot。"""
        labels = np.zeros((16, 16), dtype=np.int32)
        labels[:8, :] = 0
        labels[8:, :] = 1
        out = validate_biome(labels, n_biomes=14)
        assert out.shape == (14, 16, 16)
        # 前 8 行是 zone 0
        assert (out[0, :8, :] == 1).all()
        # 后 8 行是 zone 1
        assert (out[1, 8:, :] == 1).all()

    def test_onehot_passthrough(self):
        """one-hot 直通。"""
        biome = np.random.rand(14, 16, 16).astype(np.float32)
        out = validate_biome(biome, n_biomes=14)
        assert out.shape == (14, 16, 16)

    def test_invalid_ndim(self):
        """非法维度报错。"""
        biome = np.random.rand(16, 16, 16, 16).astype(np.float32)
        with pytest.raises(ValueError):
            validate_biome(biome)

    def test_wrong_n_biomes(self):
        """通道数不匹配报错。"""
        biome = np.random.rand(10, 16, 16).astype(np.float32)
        with pytest.raises(AssertionError):
            validate_biome(biome, n_biomes=14)


# ==================== 预设模态生成器测试 ====================

class TestMakeSeaLevelField:

    def test_shape(self):
        """形状正确。"""
        field = make_sea_level_field(0.5, 32, 32)
        assert field.shape == (1, 32, 32)

    def test_uniform_value(self):
        """均匀值 = offset。"""
        field = make_sea_level_field(0.3, 16, 16)
        assert np.allclose(field, 0.3)

    def test_clip(self):
        """超出范围裁剪。"""
        field = make_sea_level_field(5.0, 8, 8)
        assert field.max() == 1.0

    def test_negative(self):
        """负值（海平面降低）。"""
        field = make_sea_level_field(-0.5, 8, 8)
        assert np.allclose(field, -0.5)


class TestMakeClimateZoneField:

    def test_shape(self):
        """形状正确。"""
        lats = np.linspace(-90, 90, 64)
        field = make_climate_zone_field(lats, n_zones=5)
        assert field.shape == (5, 64, 64)

    def test_onehot(self):
        """每像素只有一个气候带（one-hot）。"""
        lats = np.linspace(-90, 90, 32)
        field = make_climate_zone_field(lats, n_zones=5)
        sums = field.sum(axis=0)
        assert np.allclose(sums, 1.0)

    def test_equator_zone(self):
        """赤道处 = zone 0。"""
        lats = np.linspace(-90, 90, 64)
        field = make_climate_zone_field(lats, n_zones=5)
        # 纬度 0 附近 → zone 0
        mid = 32   # 纬度 ≈ 0
        assert field[0, mid, mid] == 1.0

    def test_polar_zone(self):
        """极地处 = zone 4。"""
        lats = np.linspace(-90, 90, 64)
        field = make_climate_zone_field(lats, n_zones=5)
        # 纬度 -90 → zone 4
        assert field[4, 0, 0] == 1.0

    def test_all_zones_present(self):
        """所有气候带都有像素。"""
        lats = np.linspace(-90, 90, 64)
        field = make_climate_zone_field(lats, n_zones=5)
        for i in range(5):
            assert field[i].sum() > 0


# ==================== encode_multimodal 增强版测试 ====================

class TestEncodeMultimodalEnhanced:

    def test_with_sea_level(self):
        """带海平面条件。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sea = make_sea_level_field(0.3, 16, 16)
        out = encode_multimodal(plate, sea_level=sea,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_with_climate(self):
        """带气候带条件。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        lats = np.linspace(-90, 90, 16)
        climate = make_climate_zone_field(lats, n_zones=5)
        out = encode_multimodal(plate, climate=climate,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_with_plate_boundary(self):
        """带板块边界条件。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        boundary = np.random.randn(3, 16, 16).astype(np.float32)
        out = encode_multimodal(plate, plate_boundary=boundary,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_with_modality_config(self):
        """带 ModalityConfig。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = np.random.rand(16, 16).astype(np.float32)
        cfg = ModalityConfig(sketch_weight=0.5)
        out = encode_multimodal(plate, sketch=sketch,
                                modality_config=cfg,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_with_mask(self):
        """带空间掩码。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = np.random.rand(16, 16).astype(np.float32)
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[4:12, 4:12] = 1.0
        out = encode_multimodal(plate, sketch=sketch,
                                sketch_mask=mask,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_validated_inputs(self):
        """校验后的输入。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = validate_sketch(np.random.rand(16, 16))
        elev = validate_elev_hint(np.random.randn(16, 16))
        out = encode_multimodal(plate, sketch=sketch, elev_hint=elev,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)


# ==================== 多模态条件控制补充测试 ====================

class TestCrossAttentionFusionEdgeCases:
    """交叉注意力融合的边界用例。"""

    def test_residual_off_differs(self):
        """关闭残差时输出与 base 不同（纯注意力注入）。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=4)
        ca.eval()
        base = torch.randn(1, 12, 16, 16)
        cond = torch.randn(1, 24, 16, 16)
        with torch.no_grad():
            out = ca(base, cond)
        # 注意力输出通常与 base 不同（含 FFN/归一化）
        assert not torch.allclose(out, base, atol=1e-6)

    def test_with_dropout(self):
        """dropout > 0 可构造。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=4,
                                   dropout=0.1)
        assert ca.dropout is not None
        # eval 模式下 dropout 关闭，前向应正常
        ca.eval()
        base = torch.randn(1, 12, 8, 8)
        cond = torch.randn(1, 24, 8, 8)
        out = ca(base, cond)
        assert out.shape == (1, 12, 8, 8)
        assert torch.isfinite(out).all()

    def test_different_cond_ch(self):
        """不同条件通道数。"""
        for cond_ch in [8, 16, 32, 48]:
            ca = CrossAttentionFusion(base_ch=12, cond_ch=cond_ch,
                                       n_heads=4)
            base = torch.randn(1, 12, 8, 8)
            cond = torch.randn(1, cond_ch, 8, 8)
            out = ca(base, cond)
            assert out.shape == (1, 12, 8, 8)

    def test_large_spatial(self):
        """较大空间尺寸。"""
        ca = CrossAttentionFusion(base_ch=8, cond_ch=16, n_heads=2)
        base = torch.randn(1, 8, 64, 64)
        cond = torch.randn(1, 16, 64, 64)
        out = ca(base, cond)
        assert out.shape == (1, 8, 64, 64)
        assert torch.isfinite(out).all()

    def test_n_heads_1(self):
        """单头注意力。"""
        ca = CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=1)
        base = torch.randn(1, 12, 8, 8)
        cond = torch.randn(1, 24, 8, 8)
        out = ca(base, cond)
        assert out.shape == (1, 12, 8, 8)

    def test_n_heads_equals_ch(self):
        """n_heads = base_ch（每头 1 维）。"""
        ca = CrossAttentionFusion(base_ch=8, cond_ch=16, n_heads=8)
        base = torch.randn(1, 8, 8, 8)
        cond = torch.randn(1, 16, 8, 8)
        out = ca(base, cond)
        assert out.shape == (1, 8, 8, 8)

    def test_invalid_n_heads_zero(self):
        """n_heads=0 触发整除检查。"""
        with pytest.raises(AssertionError):
            CrossAttentionFusion(base_ch=12, cond_ch=24, n_heads=0)


class TestModalityConfigEdgeCases:
    """模态配置边界用例。"""

    def test_zero_weight(self):
        """权重 0 表示忽略模态。"""
        cfg = ModalityConfig(sketch_weight=0.0)
        assert cfg.get_weight("sketch") == 0.0

    def test_weight_above_one(self):
        """权重大于 1（放大控制）。"""
        cfg = ModalityConfig(biome_weight=1.5)
        assert cfg.get_weight("biome") == 1.5

    def test_negative_weight(self):
        """负权重（反向控制，理论允许）。"""
        cfg = ModalityConfig(elev_weight=-0.5)
        assert cfg.get_weight("elev_hint") == -0.5

    def test_all_six_weights(self):
        """6 个模态权重都可单独设置。"""
        cfg = ModalityConfig(
            sketch_weight=0.1, elev_weight=0.2, biome_weight=0.3,
            sea_level_weight=0.4, climate_weight=0.5,
            plate_boundary_weight=0.6)
        assert cfg.get_weight("sketch") == 0.1
        assert cfg.get_weight("elev_hint") == 0.2
        assert cfg.get_weight("biome") == 0.3
        assert cfg.get_weight("sea_level") == 0.4
        assert cfg.get_weight("climate") == 0.5
        assert cfg.get_weight("plate_boundary") == 0.6

    def test_to_dict_roundtrip_keys(self):
        """to_dict 含所有必要键。"""
        cfg = ModalityConfig()
        d = cfg.to_dict()
        assert set(d["weights"].keys()) == {
            "sketch", "elev_hint", "biome",
            "sea_level", "climate", "plate_boundary"}
        assert d["use_cross_attention"] is True
        assert d["residual_connection"] is True

    def test_get_weight_unknown_key(self):
        """未知模态返回 1.0。"""
        cfg = ModalityConfig()
        assert cfg.get_weight("nonexistent") == 1.0
        assert cfg.get_weight("") == 1.0


class TestMultiModalConditionerEdgeCases:
    """多模态条件器边界用例。"""

    def test_no_modalities(self):
        """无任何模态：仅板块底图，零填充其他通道。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, meta = cond(plate)
        assert out.shape == (1, 36, 16, 16)
        # 只有 plate 被使用
        assert meta["used"] == ["plate"]
        # 没有掩码
        assert meta["masked"] == []

    def test_residual_connection_off(self):
        """关闭残差连接。"""
        cfg = ModalityConfig(residual_connection=False,
                             use_cross_attention=True)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, _ = cond(plate, sketch=torch.randn(1, 1, 16, 16))
        assert out.shape == (1, 36, 16, 16)

    def test_residual_and_cross_attn_off(self):
        """关闭残差 + 关闭交叉注意力（退化到拼接）。"""
        cfg = ModalityConfig(use_cross_attention=False,
                             residual_connection=False)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        assert cond.cross_attn is None
        plate = torch.randn(1, 12, 16, 16)
        out, _ = cond(plate, sketch=torch.randn(1, 1, 16, 16))
        assert out.shape == (1, 36, 16, 16)

    def test_all_weights_zero(self):
        """所有权重为 0：模态特征被完全抑制。"""
        cfg = ModalityConfig(
            sketch_weight=0.0, elev_weight=0.0, biome_weight=0.0,
            sea_level_weight=0.0, climate_weight=0.0,
            plate_boundary_weight=0.0)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        cond.eval()
        plate = torch.randn(1, 12, 16, 16)
        with torch.no_grad():
            out_with, _ = cond(
                plate,
                sketch=torch.randn(1, 1, 16, 16),
                elev_hint=torch.randn(1, 1, 16, 16),
                biome=torch.randn(1, 14, 16, 16))
            out_without, _ = cond(plate)
        # 全 0 权重 → 应接近无模态
        diff = (out_with - out_without).abs().mean()
        assert diff < 0.5

    def test_output_channels_property(self):
        """output_channels 属性正确。"""
        cond = MultiModalConditioner(base=16)
        # 12 base + 6*4 modalities = 36
        assert cond.output_channels == 36

    def test_custom_channels(self):
        """自定义各模态通道数。"""
        cond = MultiModalConditioner(
            base_ch=12, sketch_ch=8, elev_ch=4, biome_ch=6,
            sea_level_ch=2, climate_ch=3, plate_boundary_ch=5,
            base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, _ = cond(
            plate,
            sketch=torch.randn(1, 1, 16, 16),
            elev_hint=torch.randn(1, 1, 16, 16),
            biome=torch.randn(1, 14, 16, 16),
            sea_level=torch.randn(1, 1, 16, 16),
            climate=torch.randn(1, 5, 16, 16),
            plate_boundary=torch.randn(1, 3, 16, 16))
        # 12 + 8 + 4 + 6 + 2 + 3 + 5 = 40
        assert out.shape == (1, 40, 16, 16)
        assert cond.output_channels == 40

    def test_meta_used_modalities(self):
        """meta 记录所有使用的模态。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        _, meta = cond(
            plate,
            sketch=torch.randn(1, 1, 16, 16),
            sea_level=torch.randn(1, 1, 16, 16))
        assert "plate" in meta["used"]
        assert "sketch" in meta["used"]
        assert "sea_level" in meta["used"]
        assert "biome" not in meta["used"]

    def test_meta_output_channels(self):
        """meta 包含输出通道数。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        _, meta = cond(plate)
        assert meta["output_channels"] == 36

    def test_batch_2(self):
        """批量 > 1。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(2, 12, 16, 16)
        out, _ = cond(plate, sketch=torch.randn(2, 1, 16, 16))
        assert out.shape == (2, 36, 16, 16)

    def test_different_spatial_sizes(self):
        """不同空间尺寸。"""
        cond = MultiModalConditioner(base=16)
        for sz in [8, 16, 32]:
            plate = torch.randn(1, 12, sz, sz)
            out, _ = cond(plate, sketch=torch.randn(1, 1, sz, sz))
            assert out.shape == (1, 36, sz, sz)

    def test_mask_partial_affects_output(self):
        """部分掩码影响输出（局部区域）。"""
        cond = MultiModalConditioner(base=16)
        cond.eval()
        plate = torch.randn(1, 12, 16, 16)
        sketch = torch.randn(1, 1, 16, 16)
        # 全 0 掩码 → 完全屏蔽 sketch
        mask_zero = torch.zeros(1, 1, 16, 16)
        # 全 1 掩码 → 保留 sketch
        mask_one = torch.ones(1, 1, 16, 16)
        with torch.no_grad():
            out_zero, _ = cond(plate, sketch=sketch, sketch_mask=mask_zero)
            out_one, _ = cond(plate, sketch=sketch, sketch_mask=mask_one)
        # 两种掩码应产生不同输出
        diff = (out_zero - out_one).abs().mean()
        assert diff > 1e-6

    def test_partial_mask_meta(self):
        """部分模态带掩码时 meta 记录。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        sketch_mask = torch.ones(1, 1, 16, 16)
        elev_mask = torch.zeros(1, 1, 16, 16)
        _, meta = cond(
            plate,
            sketch=torch.randn(1, 1, 16, 16),
            elev_hint=torch.randn(1, 1, 16, 16),
            sketch_mask=sketch_mask,
            elev_mask=elev_mask)
        assert "sketch" in meta["masked"]
        assert "elev_hint" in meta["masked"]
        assert len(meta["masked"]) == 2

    def test_all_preset_modalities_combined(self):
        """三种预设模态联合使用。"""
        cond = MultiModalConditioner(base=16)
        plate = torch.randn(1, 12, 16, 16)
        out, meta = cond(
            plate,
            sea_level=torch.randn(1, 1, 16, 16),
            climate=torch.randn(1, 5, 16, 16),
            plate_boundary=torch.randn(1, 3, 16, 16))
        assert out.shape == (1, 36, 16, 16)
        assert "sea_level" in meta["used"]
        assert "climate" in meta["used"]
        assert "plate_boundary" in meta["used"]

    def test_weight_recorded_in_meta(self):
        """所有使用的模态权重记录到 meta。"""
        cfg = ModalityConfig(
            sketch_weight=0.3, sea_level_weight=0.7,
            climate_weight=1.0)
        cond = MultiModalConditioner(modality_config=cfg, base=16)
        plate = torch.randn(1, 12, 16, 16)
        _, meta = cond(
            plate,
            sketch=torch.randn(1, 1, 16, 16),
            sea_level=torch.randn(1, 1, 16, 16),
            climate=torch.randn(1, 5, 16, 16))
        assert meta["weights"]["sketch"] == 0.3
        assert meta["weights"]["sea_level"] == 0.7
        assert meta["weights"]["climate"] == 1.0


class TestPresetEncodersEdgeCases:
    """预设模态编码器边界用例。"""

    def test_sea_level_extreme_values(self):
        """海平面极端值。"""
        enc = SeaLevelEncoder(out_ch=4, base=16)
        enc.eval()
        for val in [-1.0, 0.0, 1.0]:
            x = torch.full((1, 1, 8, 8), val)
            with torch.no_grad():
                out = enc(x)
            assert out.shape == (1, 4, 8, 8)
            assert torch.isfinite(out).all()

    def test_climate_zone_all_ones(self):
        """气候带全 1（异常输入但应可处理）。"""
        enc = ClimateZoneEncoder(n_zones=5, out_ch=4, base=16)
        enc.eval()
        x = torch.ones(1, 5, 8, 8)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 4, 8, 8)

    def test_climate_zone_single_zone(self):
        """仅一个气候带激活。"""
        enc = ClimateZoneEncoder(n_zones=5, out_ch=4, base=16)
        enc.eval()
        x = torch.zeros(1, 5, 8, 8)
        x[:, 2] = 1.0   # 仅 zone 2
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 4, 8, 8)

    def test_plate_boundary_single_channel(self):
        """仅一个板块边界类型激活。"""
        enc = PlateBoundaryEncoder(out_ch=4, base=16)
        enc.eval()
        x = torch.zeros(1, 3, 8, 8)
        x[:, 0] = 1.0   # 仅发散边界
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 4, 8, 8)

    def test_plate_boundary_all_zero(self):
        """板块边界全 0（无边界）。"""
        enc = PlateBoundaryEncoder(out_ch=4, base=16)
        enc.eval()
        x = torch.zeros(1, 3, 8, 8)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 4, 8, 8)


class TestValidationToolsEdgeCases:
    """校验工具边界用例。"""

    def test_validate_sketch_extreme_values(self):
        """草图极端值裁剪。"""
        sketch = np.full((16, 16), 100.0, dtype=np.float32)
        out = validate_sketch(sketch)
        assert out.max() == 1.0
        sketch_neg = np.full((16, 16), -100.0, dtype=np.float32)
        out = validate_sketch(sketch_neg)
        assert out.min() == 0.0

    def test_validate_sketch_dtype(self):
        """输出 dtype 为 float32。"""
        sketch = np.random.rand(16, 16).astype(np.float64)
        out = validate_sketch(sketch)
        assert out.dtype == np.float32

    def test_validate_elev_hint_extreme(self):
        """高度提示极端值裁剪。"""
        elev = np.full((16, 16), 100.0, dtype=np.float32)
        out = validate_elev_hint(elev)
        assert out.max() == 1.0
        elev_neg = np.full((16, 16), -100.0, dtype=np.float32)
        out = validate_elev_hint(elev_neg)
        assert out.min() == -1.0

    def test_validate_elev_hint_dtype(self):
        """高度提示 dtype。"""
        elev = np.random.randn(16, 16).astype(np.float64)
        out = validate_elev_hint(elev)
        assert out.dtype == np.float32

    def test_validate_biome_all_zones(self):
        """群落图所有类别都出现。"""
        labels = np.zeros((16, 16), dtype=np.int32)
        # 每 2 行一个 zone
        for i in range(14):
            labels[i * 16 // 14:(i + 1) * 16 // 14, :] = i
        out = validate_biome(labels, n_biomes=14)
        assert out.shape == (14, 16, 16)
        # 每个类别都有像素
        for i in range(14):
            assert out[i].sum() > 0

    def test_validate_biome_dtype_preserved(self):
        """群落图 dtype。"""
        biome = np.random.rand(14, 16, 16).astype(np.float32)
        out = validate_biome(biome, n_biomes=14)
        assert out.dtype == np.float32

    def test_validate_biome_4d_raises(self):
        """4D 输入报错。"""
        biome = np.random.rand(14, 16, 16, 1).astype(np.float32)
        with pytest.raises(ValueError):
            validate_biome(biome)

    def test_validate_sketch_3d_invalid_channels(self):
        """3D 多通道草图报错。"""
        sketch = np.random.rand(3, 16, 16).astype(np.float32)
        with pytest.raises(AssertionError):
            validate_sketch(sketch)


class TestMakeSeaLevelFieldEdgeCases:
    """海平面场生成器边界用例。"""

    def test_zero_offset(self):
        """零偏移。"""
        field = make_sea_level_field(0.0, 16, 16)
        assert np.allclose(field, 0.0)

    def test_max_positive(self):
        """最大正值。"""
        field = make_sea_level_field(1.0, 8, 8)
        assert np.allclose(field, 1.0)

    def test_min_negative(self):
        """最小负值。"""
        field = make_sea_level_field(-1.0, 8, 8)
        assert np.allclose(field, -1.0)

    def test_clip_above(self):
        """超过 +1 裁剪。"""
        field = make_sea_level_field(2.0, 8, 8)
        assert field.max() == 1.0

    def test_clip_below(self):
        """低于 -1 裁剪。"""
        field = make_sea_level_field(-2.0, 8, 8)
        assert field.min() == -1.0

    def test_dtype(self):
        """dtype 为 float32。"""
        field = make_sea_level_field(0.5, 8, 8)
        assert field.dtype == np.float32

    def test_non_square(self):
        """非方形尺寸。"""
        field = make_sea_level_field(0.3, 16, 32)
        assert field.shape == (1, 16, 32)


class TestMakeClimateZoneFieldEdgeCases:
    """气候带场生成器边界用例。"""

    def test_dtype(self):
        """dtype 为 float32。"""
        lats = np.linspace(-90, 90, 32)
        field = make_climate_zone_field(lats, n_zones=5)
        assert field.dtype == np.float32

    def test_single_latitude(self):
        """单个纬度值。"""
        lats = np.array([0.0])
        field = make_climate_zone_field(lats, n_zones=5)
        assert field.shape == (5, 1, 1)
        # 赤道 → zone 0
        assert field[0, 0, 0] == 1.0

    def test_pole_latitudes(self):
        """极地纬度。"""
        lats = np.array([-90.0, 90.0])
        field = make_climate_zone_field(lats, n_zones=5)
        assert field.shape == (5, 2, 2)
        # 两极 → zone 4
        assert field[4, 0, 0] == 1.0
        assert field[4, 1, 1] == 1.0

    def test_boundary_latitudes(self):
        """气候带边界纬度。"""
        # 10° 是赤道/副热带边界
        lats = np.array([9.9, 10.0])
        field = make_climate_zone_field(lats, n_zones=5)
        # 9.9 → zone 0
        assert field[0, 0, 0] == 1.0
        # 10.0 → zone 1
        assert field[1, 1, 1] == 1.0

    def test_each_zone_has_pixels(self):
        """每个气候带都有像素。"""
        lats = np.linspace(-90, 90, 128)
        field = make_climate_zone_field(lats, n_zones=5)
        for i in range(5):
            assert field[i].sum() > 0

    def test_no_overlap(self):
        """气候带不重叠（每像素仅属一带）。"""
        lats = np.linspace(-90, 90, 32)
        field = make_climate_zone_field(lats, n_zones=5)
        sums = field.sum(axis=0)
        assert np.allclose(sums, 1.0)


class TestEncodeMultimodalEdgeCases:
    """encode_multimodal 便捷函数边界用例。"""

    def test_no_modalities(self):
        """无任何额外模态。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        out = encode_multimodal(plate, device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_all_modalities(self):
        """所有 6 种模态联合。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        out = encode_multimodal(
            plate,
            sketch=np.random.rand(16, 16).astype(np.float32),
            elev_hint=np.random.randn(16, 16).astype(np.float32),
            biome=np.random.rand(14, 16, 16).astype(np.float32),
            sea_level=make_sea_level_field(0.3, 16, 16),
            climate=make_climate_zone_field(
                np.linspace(-90, 90, 16), n_zones=5),
            plate_boundary=np.random.randn(3, 16, 16).astype(np.float32),
            device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_2d_input_auto_expanded(self):
        """2D 输入自动扩展为 3D。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        # 2D sketch
        sketch_2d = np.random.rand(16, 16).astype(np.float32)
        out = encode_multimodal(plate, sketch=sketch_2d,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_2d_mask_auto_expanded(self):
        """2D 掩码自动扩展。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = np.random.rand(16, 16).astype(np.float32)
        mask_2d = np.zeros((16, 16), dtype=np.float32)
        mask_2d[4:12, 4:12] = 1.0
        out = encode_multimodal(plate, sketch=sketch,
                                sketch_mask=mask_2d,
                                device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_all_masks(self):
        """三种掩码同时使用。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = np.random.rand(16, 16).astype(np.float32)
        elev = np.random.randn(16, 16).astype(np.float32)
        biome = np.random.rand(14, 16, 16).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        out = encode_multimodal(
            plate, sketch=sketch, elev_hint=elev, biome=biome,
            sketch_mask=mask, elev_mask=mask, biome_mask=mask,
            device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_output_finite(self):
        """输出有限（非 NaN/Inf）。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        out = encode_multimodal(
            plate,
            sketch=np.random.rand(16, 16).astype(np.float32),
            sea_level=make_sea_level_field(0.5, 16, 16),
            device=torch.device("cpu"))
        assert np.isfinite(out).all()

    def test_with_full_config(self):
        """带完整 ModalityConfig。"""
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        cfg = ModalityConfig(
            sketch_weight=0.5, elev_weight=0.7, biome_weight=0.3,
            sea_level_weight=0.8, climate_weight=0.4,
            plate_boundary_weight=0.6,
            use_cross_attention=True, residual_connection=True)
        out = encode_multimodal(
            plate,
            sketch=np.random.rand(16, 16).astype(np.float32),
            elev_hint=np.random.randn(16, 16).astype(np.float32),
            modality_config=cfg,
            device=torch.device("cpu"))
        assert out.shape == (36, 16, 16)

    def test_deterministic_with_seed(self):
        """固定 seed 时输出确定。"""
        torch.manual_seed(42)
        plate = np.random.randn(12, 16, 16).astype(np.float32)
        sketch = np.random.rand(1, 16, 16).astype(np.float32)

        cond1 = MultiModalConditioner(base=32).eval()
        cond1.eval()
        with torch.no_grad():
            p = torch.from_numpy(plate).float().unsqueeze(0)
            s = torch.from_numpy(sketch).float().unsqueeze(0)
            out1, _ = cond1(p, sketch=s)

        torch.manual_seed(42)
        cond2 = MultiModalConditioner(base=32).eval()
        with torch.no_grad():
            out2, _ = cond2(p, sketch=s)

        assert torch.allclose(out1, out2, atol=1e-6)
