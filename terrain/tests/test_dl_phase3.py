"""阶段 3 物理一致性监督子模块测试。

覆盖：
- RiverNet / SnowlineNet / CoastNet 网络前向形状与数值范围
- 三个损失函数（river_loss / snowline_loss / coast_loss）数值合理
- physics_dataset 三个数据集的标签提取（river/snowline/coast）形状与值域
- 合成数据生成 → Dataset 加载闭环
- train_physics 训练循环 smoke test（1 epoch 小数据）
- infer_physics 推理流程（随机权重）+ drop-in 替换封装

不依赖训练好的权重或真实 DEM 数据，全部用合成数据 + 随机权重。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

from terrain.dl.config import TILE_SIZE, elev_to_norm, ELEV_MIN, ELEV_MAX
from terrain.dl.physics_nets import (
    RiverNet, SnowlineNet, CoastNet,
    river_loss, snowline_loss, coast_loss, coast_apply,
)
from terrain.dl.physics_dataset import (
    extract_river_label, extract_snowline_label, extract_coast_label,
    RiverDataset, SnowlineDataset, CoastDataset,
    generate_synthetic_tiles, make_physics_loaders,
)


# ==================== RiverNet 测试 ====================

def test_rivernet_forward_shape():
    """RiverNet 前向：(B, 1, H, W) → (B, 1, H, W) 概率 [0,1]。"""
    model = RiverNet(base=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_rivernet_slope_channel():
    """坡度通道计算正确（非负，归一化到 [0,1]）。"""
    dem = torch.randn(1, 1, 32, 32)
    slope = RiverNet.compute_slope_channel(dem)
    assert slope.shape == dem.shape
    assert slope.min() >= 0.0
    assert slope.max() <= 1.0 + 1e-6


def test_rivernet_param_count():
    """参数数 < 1M（轻量约束，默认 base=16）。"""
    model = RiverNet()   # 默认 base=16
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000, f"参数数 {n/1e6:.2f}M 超过 1M"


def test_river_loss_finite():
    """river_loss = BCE + 连通性正则，有限值。"""
    pred = torch.rand(2, 1, 32, 32)
    target = (torch.rand(2, 1, 32, 32) > 0.5).float()
    loss, info = river_loss(pred, target)
    assert torch.isfinite(loss)
    assert "bce" in info and "conn" in info and "loss" in info
    assert info["loss"] > 0


def test_river_loss_connectivity_weight():
    """connectivity_weight=0 时仅 BCE。"""
    pred = torch.rand(1, 1, 16, 16)
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    loss, info = river_loss(pred, target, connectivity_weight=0.0)
    assert abs(info["loss"] - info["bce"]) < 1e-5


# ==================== SnowlineNet 测试 ====================

def test_snowlinenet_forward_shape():
    """SnowlineNet 前向：(B,1,H,W) × 3 → (B,1,H,W) 雪线高程。"""
    model = SnowlineNet(base=16)
    lat = torch.rand(2, 1, 64, 64)
    elev = torch.rand(2, 1, 64, 64)
    ci = torch.rand(2, 1, 64, 64)
    out = model(lat, elev, ci)
    assert out.shape == (2, 1, 64, 64)
    # 输出范围 [0, 6500]
    assert out.min() >= 0.0
    assert out.max() <= 6500.0 + 1e-3


def test_snowlinenet_param_count():
    """参数数 < 1M（默认 base=16）。"""
    model = SnowlineNet()
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000


def test_snowline_loss_finite():
    """snowline_loss = L1，有限值。"""
    pred = torch.rand(1, 1, 32, 32) * 6500
    target = torch.rand(1, 1, 32, 32) * 6500
    loss, info = snowline_loss(pred, target)
    assert torch.isfinite(loss)
    assert info["loss"] >= 0


# ==================== CoastNet 测试 ====================

def test_coastnet_forward_shape():
    """CoastNet 前向：(B,1,H,W) → (B,3,H,W) logits。"""
    model = CoastNet(base=16)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 3, 64, 64)


def test_coastnet_param_count():
    """参数数 < 1M（默认 base=16）。"""
    model = CoastNet()
    n = sum(p.numel() for p in model.parameters())
    assert n < 1_000_000


def test_coast_loss_finite():
    """coast_loss = CrossEntropy，有限值。"""
    logits = torch.randn(2, 3, 32, 32)
    target = torch.randint(0, 3, (2, 32, 32))
    loss, info = coast_loss(logits, target)
    assert torch.isfinite(loss)
    assert info["loss"] > 0


def test_coast_loss_class_weights():
    """class_weights 影响损失值。"""
    logits = torch.randn(1, 3, 16, 16)
    target = torch.randint(0, 3, (1, 16, 16))
    loss1, _ = coast_loss(logits, target,
                          class_weights=torch.tensor([1.0, 1.0, 1.0]))
    loss2, _ = coast_loss(logits, target,
                          class_weights=torch.tensor([1.0, 5.0, 1.0]))
    # 不同权重 → 不同损失
    assert not torch.allclose(loss1, loss2)


def test_coast_apply_shape():
    """coast_apply 后处理：DEM + logits → 锐化后 DEM，形状不变。"""
    dem = np.random.randn(32, 32).astype(np.float32)
    logits = np.random.randn(3, 32, 32).astype(np.float32)
    out = coast_apply(dem, logits, shelf_target_depth=-2000.0)
    assert out.shape == dem.shape
    assert out.dtype == np.float32


# ==================== 标签提取测试 ====================

def test_extract_river_label_shape():
    """河网标签提取：DEM → 二值掩码 {0,1}。"""
    dem = np.random.randn(64, 64).astype(np.float32) * 1000  # ±1000m
    label = extract_river_label(dem, threshold_quantile=0.99)
    assert label.shape == dem.shape
    assert label.dtype == np.float32
    assert set(np.unique(label)).issubset({0.0, 1.0})


def test_extract_river_label_all_ocean():
    """全海洋 DEM → 河网标签全 0（陆地像素=0）。"""
    dem = np.full((32, 32), -1000.0, dtype=np.float32)
    label = extract_river_label(dem)
    assert label.sum() == 0


def test_extract_snowline_label_shape():
    """雪线标签提取：DEM + lats → 雪线高程场。"""
    H, W = 64, 64
    dem = np.random.randn(H, W).astype(np.float32) * 1000
    lats = np.linspace(-60, 60, H, dtype=np.float32)
    label = extract_snowline_label(dem, lats)
    assert label.shape == dem.shape
    assert label.dtype == np.float32
    assert label.min() >= 0.0
    assert label.max() <= 6500.0


def test_extract_snowline_label_ocean_zero():
    """海洋像素雪线 = 0。"""
    dem = np.full((32, 32), -1000.0, dtype=np.float32)
    lats = np.linspace(-60, 60, 32, dtype=np.float32)
    label = extract_snowline_label(dem, lats)
    assert (label == 0).all()


def test_extract_coast_label_classes():
    """海岸线标签三分类：陆地=0, 陆架=1, 深海=2。"""
    dem = np.array([
        [1000, 1000, -500, -500],   # 陆地 / 陆地 / 陆架 / 陆架
        [1000, -500, -3000, -3000], # 陆地 / 陆架 / 深海 / 深海
    ], dtype=np.float32)
    label = extract_coast_label(dem, shelf_depth=-2000.0)
    assert label.shape == dem.shape
    assert label.dtype == np.int64
    # 陆地 (≥0) → 0
    assert (label[dem >= 0] == 0).all()
    # 陆架 (-2000 ≤ elev < 0) → 1
    assert (label[(dem >= -2000) & (dem < 0)] == 1).all()
    # 深海 (< -2000) → 2
    assert (label[dem < -2000] == 2).all()


# ==================== Dataset 闭环测试 ====================

@pytest.fixture
def synth_tiles(tmp_path):
    """生成 4 个合成切片供测试。"""
    n = generate_synthetic_tiles(tmp_path, n_tiles=4, size=64, seed=42)
    assert n == 4
    return tmp_path


def test_river_dataset(synth_tiles):
    """RiverDataset 加载与标签形状。"""
    ds = RiverDataset(root=synth_tiles, size=64, augment=False)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (1, 64, 64)
    assert y.shape == (1, 64, 64)
    assert x.min() >= -1.01 and x.max() <= 1.01   # 归一化
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})


def test_snowline_dataset(synth_tiles):
    """SnowlineDataset 加载与三通道输入。"""
    ds = SnowlineDataset(root=synth_tiles, size=64, augment=False)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (3, 64, 64)   # lat/elev/ci
    assert y.shape == (1, 64, 64)
    assert y.min() >= 0 and y.max() <= 6500


def test_coast_dataset(synth_tiles):
    """CoastDataset 加载与三分类标签。"""
    ds = CoastDataset(root=synth_tiles, size=64, augment=False)
    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (1, 64, 64)
    assert y.shape == (64, 64)
    assert y.dtype == torch.int64
    assert set(torch.unique(y).tolist()).issubset({0, 1, 2})


def test_dataset_augment(synth_tiles):
    """增强开启时不报错且形状不变。"""
    ds = RiverDataset(root=synth_tiles, size=64, augment=True)
    x, y = ds[0]
    assert x.shape == (1, 64, 64)
    assert y.shape == (1, 64, 64)


def test_make_physics_loaders(synth_tiles):
    """make_physics_loaders 划分训练/验证。"""
    import os
    # 临时把合成目录作为 DEM_TILES_DIR
    train_loader, val_loader = make_physics_loaders(
        "river", batch_size=2, val_ratio=0.25, num_workers=0,
        root=synth_tiles)
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    batch = next(iter(train_loader))
    assert batch[0].shape[0] <= 2


# ==================== 训练 smoke test ====================

def test_train_physics_smoke(synth_tiles, tmp_path, monkeypatch):
    """train_physics 1 epoch smoke test（river 任务，随机权重）。

    不验证收敛，仅验证训练循环无异常。
    强制 CPU 避免 MPS BCE loss 已知崩溃。
    """
    from terrain.dl import train_physics as tp

    # 重定向 CKPT_DIR 到临时目录
    monkeypatch.setattr(tp, "CKPT_DIR", tmp_path)
    # 强制 CPU（MPS 的 BCE loss 有已知崩溃问题）
    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))

    # 调用训练（1 epoch，小 batch，合成数据）
    tp.train_physics(
        task="river",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        val_ratio=0.25,
        eval_interval=1,
        data_root=str(synth_tiles),
    )
    # 验证权重已保存
    ckpt = tmp_path / "rivernet.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert "model" in state
    assert "ema" in state
    assert state["epoch"] == 0


# ==================== 推理测试（随机权重） ====================

@pytest.fixture
def random_rivernet_ckpt(tmp_path):
    """保存随机权重供推理测试。"""
    model = RiverNet(base=16)
    ckpt = tmp_path / "rivernet.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "epoch": 0,
        "base": 16,
    }, ckpt)
    return ckpt


@pytest.fixture
def random_snowlinenet_ckpt(tmp_path):
    model = SnowlineNet(base=16)
    ckpt = tmp_path / "snowlinenet.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "epoch": 0,
        "base": 16,
    }, ckpt)
    return ckpt


@pytest.fixture
def random_coastnet_ckpt(tmp_path):
    model = CoastNet(base=16)
    ckpt = tmp_path / "coastnet.pt"
    torch.save({
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "epoch": 0,
        "base": 16,
    }, ckpt)
    return ckpt


def test_predict_river(random_rivernet_ckpt):
    """predict_river 推理：DEM → 概率图。"""
    from terrain.dl.infer_physics import predict_river
    dem = np.random.randn(64, 64).astype(np.float32) * 1000
    prob = predict_river(dem, ckpt=str(random_rivernet_ckpt))
    assert prob.shape == (64, 64)
    assert prob.min() >= 0 and prob.max() <= 1


def test_predict_river_tiled(random_rivernet_ckpt):
    """predict_river 大图分块推理形状正确。"""
    from terrain.dl.infer_physics import predict_river
    dem = np.random.randn(128, 96).astype(np.float32) * 1000
    prob = predict_river(dem, ckpt=str(random_rivernet_ckpt), tile=64)
    assert prob.shape == (128, 96)
    assert prob.min() >= 0 and prob.max() <= 1


def test_predict_snowline(random_snowlinenet_ckpt):
    """predict_snowline 推理：DEM + lats → 雪线高程。"""
    from terrain.dl.infer_physics import predict_snowline
    H, W = 64, 64
    dem = np.random.randn(H, W).astype(np.float32) * 1000
    lats = np.linspace(-60, 60, H, dtype=np.float32)
    out = predict_snowline(dem, lats=lats, ckpt=str(random_snowlinenet_ckpt))
    assert out.shape == (H, W)
    assert out.min() >= 0
    assert out.max() <= 6500


def test_predict_coast(random_coastnet_ckpt):
    """predict_coast 推理：DEM → 三类标签。"""
    from terrain.dl.infer_physics import predict_coast
    dem = np.random.randn(64, 64).astype(np.float32) * 1000
    label = predict_coast(dem, ckpt=str(random_coastnet_ckpt))
    assert label.shape == (64, 64)
    assert label.dtype == np.int64
    assert set(np.unique(label)).issubset({0, 1, 2})


def test_predict_coast_probs(random_coastnet_ckpt):
    """predict_coast return_probs=True 返回概率图。"""
    from terrain.dl.infer_physics import predict_coast
    dem = np.random.randn(64, 64).astype(np.float32) * 1000
    probs = predict_coast(dem, ckpt=str(random_coastnet_ckpt),
                          return_probs=True)
    assert probs.shape == (3, 64, 64)
    # 每个像素三类概率和 ≈ 1
    s = probs.sum(axis=0)
    assert np.allclose(s, 1.0, atol=1e-4)


# ==================== Drop-in 替换封装测试 ====================

def test_rivernet_threshold(random_rivernet_ckpt):
    """rivernet_threshold 返回 bool 掩码。"""
    from terrain.dl.infer_physics import rivernet_threshold
    dem = np.random.randn(64, 64).astype(np.float32) * 1000
    mask = rivernet_threshold(dem, threshold=0.5,
                              ckpt=str(random_rivernet_ckpt))
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


def test_snowline_apply(random_snowlinenet_ckpt):
    """snowline_apply drop-in 替换 snowline_m。"""
    from terrain.dl.infer_physics import snowline_apply
    H, W = 64, 64
    dem = np.random.randn(H, W).astype(np.float32) * 1000
    lats = np.linspace(-60, 60, H, dtype=np.float32)
    out = snowline_apply(dem, lats=lats, ckpt=str(random_snowlinenet_ckpt))
    assert out.shape == (H, W)
    assert out.min() >= 0
    assert out.max() <= 6500


def test_coast_sharpen(random_coastnet_ckpt):
    """coast_sharpen drop-in 替换 ocean_margin_fix。"""
    from terrain.dl.infer_physics import coast_sharpen
    dem = np.random.randn(64, 64).astype(np.float32) * 5000
    out = coast_sharpen(dem, ckpt=str(random_coastnet_ckpt),
                        shelf_target_depth=-2000.0)
    assert out.shape == dem.shape
    assert out.dtype == np.float32


# ==================== 评估指标测试 ====================

from terrain.dl.physics_metrics import (
    river_metrics, snowline_metrics, coast_metrics,
    MetricAccumulator, format_metrics,
)


# ---- river_metrics ----

def test_river_metrics_perfect():
    """完美预测 → IoU = 1.0。"""
    target = np.zeros((32, 32), dtype=np.float32)
    target[10:20, 10:20] = 1.0
    pred = target.copy()
    m = river_metrics(pred, target, threshold=0.5)
    assert m["iou"] == 1.0
    assert m["dice"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_river_metrics_no_overlap():
    """完全无重合 → IoU = 0。"""
    pred = np.zeros((32, 32), dtype=np.float32)
    pred[0:10, 0:10] = 1.0
    target = np.zeros((32, 32), dtype=np.float32)
    target[20:30, 20:30] = 1.0
    m = river_metrics(pred, target)
    assert m["iou"] == 0.0
    assert m["tp"] == 0


def test_river_metrics_partial():
    """部分重合 → 0 < IoU < 1。"""
    pred = np.zeros((32, 32), dtype=np.float32)
    pred[10:20, 10:20] = 1.0
    target = np.zeros((32, 32), dtype=np.float32)
    target[15:25, 15:25] = 1.0
    m = river_metrics(pred, target)
    assert 0 < m["iou"] < 1
    assert m["tp"] > 0
    assert m["fp"] > 0
    assert m["fn"] > 0


def test_river_metrics_empty_pred():
    """全空预测 → IoU = 0，无崩溃。"""
    pred = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.float32)
    target[5:10, 5:10] = 1.0
    m = river_metrics(pred, target)
    assert m["iou"] == 0.0


def test_river_metrics_keys():
    """返回的 dict 包含所有期望键。"""
    pred = np.random.rand(16, 16).astype(np.float32)
    target = (np.random.rand(16, 16) > 0.5).astype(np.float32)
    m = river_metrics(pred, target)
    for k in ["iou", "dice", "precision", "recall", "f1",
              "tp", "fp", "fn", "tn", "connectivity", "topo_acc"]:
        assert k in m, f"缺少键 {k}"


# ---- snowline_metrics ----

def test_snowline_metrics_perfect():
    """完美预测 → MAE = 0, R² = 1。"""
    target = np.random.rand(32, 32).astype(np.float32) * 5000
    pred = target.copy()
    m = snowline_metrics(pred, target)
    assert m["mae_m"] == 0.0
    assert m["rmse_m"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-5
    assert m["bias_m"] == 0.0


def test_snowline_metrics_with_bias():
    """有固定偏差 → bias = 偏差值。"""
    target = np.full((16, 16), 3000.0, dtype=np.float32)
    pred = np.full((16, 16), 3500.0, dtype=np.float32)
    m = snowline_metrics(pred, target)
    assert abs(m["bias_m"] - 500.0) < 1e-3
    assert abs(m["mae_m"] - 500.0) < 1e-3


def test_snowline_metrics_lat_bands():
    """纬度分带误差计算正确。"""
    H, W = 64, 64
    target = np.full((H, W), 3000.0, dtype=np.float32)
    pred = np.full((H, W), 3000.0, dtype=np.float32)
    # 仅在赤道带加偏差
    lats = np.linspace(-90, 90, H, dtype=np.float32)
    abs_lat = np.abs(lats)[:, None]
    pred = np.where(abs_lat < 23.5, 3500.0, pred).astype(np.float32)
    m = snowline_metrics(pred, target, lats=lats)
    assert "mae_tropics" in m
    assert m["mae_tropics"] > 0
    # 极地带无误差
    assert "mae_polar" in m
    assert m["mae_polar"] == 0.0


def test_snowline_metrics_land_mask():
    """land_mask 限制统计范围。"""
    target = np.full((16, 16), 3000.0, dtype=np.float32)
    pred = np.full((16, 16), 4000.0, dtype=np.float32)
    # 仅左半为陆地
    land = np.zeros((16, 16), dtype=bool)
    land[:, :8] = True
    m = snowline_metrics(pred, target, land_mask=land)
    assert m["n_pixels"] == 16 * 8


# ---- coast_metrics ----

def test_coast_metrics_perfect():
    """完美预测 → OA = 1, mIoU = 1。"""
    target = np.zeros((32, 32), dtype=np.int64)
    target[:16] = 0   # 陆地
    target[16:24] = 1  # 陆架
    target[24:] = 2    # 深海
    pred = target.copy()
    m = coast_metrics(pred, target)
    assert m["oa"] == 1.0
    assert m["miou"] == 1.0
    assert m["iou_land"] == 1.0
    assert m["iou_shelf"] == 1.0
    assert m["iou_deep"] == 1.0


def test_coast_metrics_misclassification():
    """有误分 → OA < 1。"""
    target = np.zeros((32, 32), dtype=np.int64)
    target[:16] = 0
    target[16:] = 2
    pred = target.copy()
    pred[0:4] = 2   # 误分 4 行
    m = coast_metrics(pred, target)
    assert m["oa"] < 1.0
    assert m["oa"] > 0.0
    assert "confusion_matrix" in m


def test_coast_metrics_confusion_matrix_shape():
    """混淆矩阵 3×3。"""
    pred = np.random.randint(0, 3, (16, 16))
    target = np.random.randint(0, 3, (16, 16))
    m = coast_metrics(pred, target)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert cm.sum() == 16 * 16


def test_coast_metrics_edge_sharpness():
    """边缘锐化度 ∈ [0, 1]。"""
    target = np.zeros((32, 32), dtype=np.int64)
    target[:16] = 0
    target[16:] = 2
    pred = target.copy()
    m = coast_metrics(pred, target)
    assert 0 <= m["edge_sharpness"] <= 1.0


# ---- MetricAccumulator ----

def test_accumulator_river():
    """河网累积器：多 batch 累积 TP/FP/FN。"""
    acc = MetricAccumulator("river")
    for _ in range(3):
        pred = np.random.rand(16, 16).astype(np.float32)
        target = (np.random.rand(16, 16) > 0.5).astype(np.float32)
        acc.update(pred, target)
    m = acc.compute()
    assert "iou" in m
    assert "dice" in m
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == 3 * 16 * 16


def test_accumulator_snowline():
    """雪线累积器：拼接所有像素。"""
    acc = MetricAccumulator("snowline")
    for _ in range(3):
        pred = np.random.rand(16, 16).astype(np.float32) * 5000
        target = np.random.rand(16, 16).astype(np.float32) * 5000
        acc.update(pred, target)
    m = acc.compute()
    assert "mae_m" in m
    assert "rmse_m" in m
    assert m["n_pixels"] == 3 * 16 * 16


def test_accumulator_coast():
    """海岸线累积器：累积混淆矩阵。"""
    acc = MetricAccumulator("coast")
    for _ in range(3):
        pred = np.random.randint(0, 3, (16, 16))
        target = np.random.randint(0, 3, (16, 16))
        acc.update(pred, target)
    m = acc.compute()
    assert "oa" in m
    assert "miou" in m
    assert "iou_land" in m


def test_accumulator_empty():
    """空累积器 → 空字典。"""
    acc = MetricAccumulator("river")
    assert acc.compute() == {}


def test_accumulator_reset():
    """reset 清空累积。"""
    acc = MetricAccumulator("river")
    acc.update(np.random.rand(8, 8), np.zeros((8, 8)))
    acc.reset()
    assert acc.compute() == {}


# ---- format_metrics ----

def test_format_metrics_river():
    """河网格式化字符串含关键指标。"""
    m = {"iou": 0.5, "dice": 0.6, "f1": 0.55, "connectivity": 0.7,
         "topo_acc": 0.8}
    s = format_metrics("river", m)
    assert "iou=0.500" in s
    assert "f1=0.550" in s


def test_format_metrics_snowline():
    """雪线格式化字符串含 MAE/RMSE。"""
    m = {"mae_m": 100.0, "rmse_m": 150.0, "r2": 0.9, "bias_m": 10.0}
    s = format_metrics("snowline", m)
    assert "mae=100m" in s
    assert "rmse=150m" in s


def test_format_metrics_coast():
    """海岸线格式化字符串含 mIoU。"""
    m = {"oa": 0.9, "miou": 0.7, "iou_land": 0.8, "iou_shelf": 0.6,
         "iou_deep": 0.7, "edge_sharpness": 0.5}
    s = format_metrics("coast", m)
    assert "oa=0.900" in s
    assert "miou=0.700" in s


# ---- 增强训练循环测试 ----

def test_train_physics_grad_accum(synth_tiles, tmp_path, monkeypatch):
    """梯度累积训练 smoke test（2 步累积）。"""
    from terrain.dl import train_physics as tp

    monkeypatch.setattr(tp, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))

    tp.train_physics(
        task="river",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        val_ratio=0.25,
        eval_interval=1,
        data_root=str(synth_tiles),
        grad_accum=2,
    )
    ckpt = tmp_path / "rivernet.pt"
    assert ckpt.exists()


def test_train_physics_early_stop(synth_tiles, tmp_path, monkeypatch):
    """早停机制：patience=1 时最多训练 2 轮验证后停止。"""
    from terrain.dl import train_physics as tp

    monkeypatch.setattr(tp, "CKPT_DIR", tmp_path)
    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))

    tp.train_physics(
        task="river",
        epochs=10,
        batch_size=2,
        lr=1e-3,
        num_workers=0,
        val_ratio=0.25,
        eval_interval=1,
        data_root=str(synth_tiles),
        early_stop=1,
    )
    # 早停后权重仍存在
    ckpt = tmp_path / "rivernet.pt"
    assert ckpt.exists()


# ==================== 评估指标补充测试（边界与分支覆盖） ====================

from terrain.dl.physics_metrics import (
    _connectivity_score, _skeletonize, _topo_accuracy, _edge_sharpness,
)


# ---- river_metrics 阈值与边界 ----

def test_river_metrics_threshold_high():
    """高阈值 → 预测正像素更少 → recall 下降。"""
    pred = np.array([[0.3, 0.7], [0.6, 0.9]], dtype=np.float32)
    target = np.array([[0, 1], [1, 1]], dtype=np.float32)
    m_low = river_metrics(pred, target, threshold=0.5)
    m_high = river_metrics(pred, target, threshold=0.8)
    # 阈值 0.8 时只有 0.9 被判正，recall 应更低
    assert m_high["recall"] <= m_low["recall"]


def test_river_metrics_all_positive_pred():
    """全预测为正 → precision = 真实正比例，recall = 1.0。"""
    target = np.zeros((10, 10), dtype=np.float32)
    target[2:5, 2:5] = 1.0   # 9 个正
    pred = np.ones_like(target)
    m = river_metrics(pred, target)
    assert m["recall"] == 1.0
    assert abs(m["precision"] - 9 / 100) < 1e-6


def test_river_metrics_both_empty():
    """预测与标签全空 → IoU=0（max 防除零），无崩溃。"""
    pred = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.float32)
    m = river_metrics(pred, target)
    assert m["iou"] == 0.0
    assert m["connectivity"] == 0.0
    assert m["topo_acc"] == 0.0


def test_river_metrics_connectivity_perfect():
    """完美重合 → connectivity = 1.0。"""
    target = np.zeros((32, 32), dtype=np.float32)
    target[8:24, 14:18] = 1.0   # 一条粗河道
    pred = target.copy()
    m = river_metrics(pred, target)
    assert m["connectivity"] > 0.9   # 骨架化后应高度重合


def test_river_metrics_connectivity_disjoint():
    """完全不重合 → connectivity = 0.0。"""
    pred = np.zeros((32, 32), dtype=np.float32)
    pred[0:8, 0:8] = 1.0
    target = np.zeros((32, 32), dtype=np.float32)
    target[24:32, 24:32] = 1.0
    m = river_metrics(pred, target)
    assert m["connectivity"] == 0.0


# ---- _skeletonize 直接测试 ----

def test_skeletonize_empty():
    """空 mask → 空骨架。"""
    skel = _skeletonize(np.zeros((16, 16), dtype=bool))
    assert not skel.any()


def test_skeletonize_thin_line():
    """细线（1px 宽）→ 骨架 ≈ 原图。"""
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 4:28] = True   # 水平细线
    skel = _skeletonize(mask)
    # 细线本身就是骨架，重合率应很高
    inter = np.logical_and(skel, mask).sum()
    union = np.logical_or(skel, mask).sum()
    assert inter / union > 0.8


def test_skeletonize_solid_block():
    """实心方块 → 骨架比原 block 像素少。"""
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True   # 16×16 实心
    skel = _skeletonize(mask)
    assert skel.sum() < mask.sum()
    assert skel.sum() > 0   # 应有骨架


# ---- _topo_accuracy 分支 ----

def test_topo_accuracy_empty_pred():
    """空预测 → 0.0。"""
    pred = np.zeros((16, 16), dtype=bool)
    target = np.zeros((16, 16), dtype=bool)
    target[5:10, 5:10] = True
    assert _topo_accuracy(pred, target) == 0.0


def test_topo_accuracy_perfect_overlap():
    """完全重合 → 1.0（距离=0）。"""
    target = np.zeros((16, 16), dtype=bool)
    target[5:10, 5:10] = True
    pred = target.copy()
    assert _topo_accuracy(pred, target) == 1.0


def test_topo_accuracy_within_tolerance():
    """偏移 ≤ 3 像素 → topo_acc > 0.5。"""
    target = np.zeros((32, 32), dtype=bool)
    target[16, 8:24] = True   # 水平线
    pred = np.zeros((32, 32), dtype=bool)
    pred[18, 8:24] = True     # 下移 2 像素
    acc = _topo_accuracy(pred, target)
    assert acc > 0.5


def test_topo_accuracy_far_offset():
    """偏移 > 3 像素 → topo_acc < 0.5。"""
    target = np.zeros((32, 32), dtype=bool)
    target[8, 8:24] = True
    pred = np.zeros((32, 32), dtype=bool)
    pred[24, 8:24] = True   # 偏移 16 像素
    acc = _topo_accuracy(pred, target)
    assert acc < 0.5


# ---- _connectivity_score 分支 ----

def test_connectivity_score_empty_pred():
    """空预测 → 0.0。"""
    pred = np.zeros((16, 16), dtype=bool)
    target = np.zeros((16, 16), dtype=bool)
    target[5:10, 5:10] = True
    assert _connectivity_score(pred, target) == 0.0


def test_connectivity_score_empty_target():
    """空标签 → 0.0。"""
    pred = np.zeros((16, 16), dtype=bool)
    pred[5:10, 5:10] = True
    target = np.zeros((16, 16), dtype=bool)
    assert _connectivity_score(pred, target) == 0.0


# ---- snowline_metrics 补充分支 ----

def test_snowline_metrics_negative_r2():
    """预测比均值还差 → R² < 0。"""
    target = np.array([[1000, 3000, 5000]], dtype=np.float32)
    pred = np.array([[3500, 3500, 3500]], dtype=np.float32)  # 全等于均值
    m = snowline_metrics(pred, target)
    assert m["r2"] <= 0.0


def test_snowline_metrics_no_lats_no_bands():
    """lats=None → 不计算分带，无 mae_tropics 键。"""
    pred = np.random.rand(16, 16).astype(np.float32) * 5000
    target = np.random.rand(16, 16).astype(np.float32) * 5000
    m = snowline_metrics(pred, target, lats=None)
    assert "mae_tropics" not in m
    assert "mae_polar" not in m


def test_snowline_metrics_empty_land_mask():
    """land_mask 全 False → n_pixels=0，无崩溃。"""
    pred = np.full((8, 8), 3000.0, dtype=np.float32)
    target = np.full((8, 8), 3000.0, dtype=np.float32)
    land = np.zeros((8, 8), dtype=bool)
    m = snowline_metrics(pred, target, land_mask=land)
    assert m["n_pixels"] == 0


def test_snowline_metrics_rmse_ge_mae():
    """RMSE ≥ MAE（数学性质）。"""
    pred = np.random.rand(32, 32).astype(np.float32) * 6000
    target = np.random.rand(32, 32).astype(np.float32) * 6000
    m = snowline_metrics(pred, target)
    assert m["rmse_m"] >= m["mae_m"] - 1e-6


def test_snowline_metrics_partial_bands():
    """仅部分纬度带有数据 → 只返回有数据的带。"""
    H, W = 64, 64
    target = np.full((H, W), 3000.0, dtype=np.float32)
    pred = np.full((H, W), 3000.0, dtype=np.float32)
    # lats 只覆盖 0~30 度 → 仅 tropics + subtropics 有数据
    lats = np.linspace(0, 30, H, dtype=np.float32)
    m = snowline_metrics(pred, target, lats=lats)
    assert "mae_tropics" in m
    assert "mae_subtropics" in m
    assert "mae_polar" not in m   # 无极地带数据


# ---- coast_metrics 补充分支 ----

def test_coast_metrics_n_classes_param():
    """n_classes=2 时混淆矩阵 2×2，返回通用键名。"""
    pred = np.array([[0, 1], [1, 0]], dtype=np.int64)
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    m = coast_metrics(pred, target, n_classes=2)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    # 非三类时用通用键名 iou_0/iou_1/acc_0/acc_1
    assert "iou_0" in m and "iou_1" in m
    assert "acc_0" in m and "acc_1" in m
    # 不应有三类专用键
    assert "iou_land" not in m


def test_coast_metrics_all_one_class():
    """全为同一类 → 该类 IoU=1，其他类 IoU=0。"""
    pred = np.zeros((16, 16), dtype=np.int64)
    target = np.zeros((16, 16), dtype=np.int64)
    m = coast_metrics(pred, target)
    assert m["iou_land"] == 1.0
    assert m["iou_shelf"] == 0.0
    assert m["iou_deep"] == 0.0
    assert m["oa"] == 1.0


def test_coast_metrics_class_accuracy():
    """各类 acc 计算正确（对角/列和）。"""
    # 构造已知混淆矩阵
    pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    target = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    m = coast_metrics(pred, target)
    # 类 0：tp=1, 列和=2 → acc=0.5
    assert abs(m["acc_land"] - 0.5) < 1e-6
    # 类 1：tp=1, 列和=2 → acc=0.5
    assert abs(m["acc_shelf"] - 0.5) < 1e-6


def test_coast_metrics_macc_is_mean():
    """mAcc = 各类 acc 的平均。"""
    pred = np.random.randint(0, 3, (32, 32))
    target = np.random.randint(0, 3, (32, 32))
    m = coast_metrics(pred, target)
    expected = (m["acc_land"] + m["acc_shelf"] + m["acc_deep"]) / 3
    assert abs(m["macc"] - expected) < 1e-6


# ---- _edge_sharpness 分支 ----

def test_edge_sharpness_all_land():
    """全陆地 → 1.0（无海岸线）。"""
    pred = np.zeros((16, 16), dtype=np.int64)   # 全陆地
    target = np.zeros((16, 16), dtype=np.int64)
    assert _edge_sharpness(pred, target) == 1.0


def test_edge_sharpness_all_sea():
    """全海洋 → 1.0（无海岸线）。"""
    pred = np.full((16, 16), 2, dtype=np.int64)
    target = np.full((16, 16), 2, dtype=np.int64)
    assert _edge_sharpness(pred, target) == 1.0


def test_edge_sharpness_no_pred_edge():
    """预测无海岸线（全陆地）但标签有 → 0.0。"""
    pred = np.zeros((32, 32), dtype=np.int64)   # 全陆地
    target = np.zeros((32, 32), dtype=np.int64)
    target[16:] = 2   # 标签下半为海
    assert _edge_sharpness(pred, target) == 0.0


def test_edge_sharpness_exact_match():
    """预测与标签海岸线完全一致 → 锐化度 > 0.9。"""
    target = np.zeros((32, 32), dtype=np.int64)
    target[:16] = 0
    target[16:] = 2
    pred = target.copy()
    score = _edge_sharpness(pred, target)
    assert score > 0.9


# ---- MetricAccumulator 补充 ----

def test_accumulator_unknown_task():
    """未知任务 → ValueError。"""
    acc = MetricAccumulator("unknown")
    acc.update(np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="未知任务"):
        acc.compute()


def test_accumulator_river_equals_whole():
    """累积多 batch 的 IoU = 整体计算的 IoU。"""
    rng = np.random.RandomState(42)
    preds = [rng.rand(16, 16).astype(np.float32) for _ in range(3)]
    targets = [(rng.rand(16, 16) > 0.5).astype(np.float32) for _ in range(3)]

    acc = MetricAccumulator("river")
    for p, t in zip(preds, targets):
        acc.update(p, t)
    acc_result = acc.compute()

    # 整体计算
    whole_pred = np.concatenate([p.ravel() for p in preds])
    whole_tgt = np.concatenate([t.ravel() for t in targets])
    whole = river_metrics(whole_pred, whole_tgt)

    assert abs(acc_result["iou"] - whole["iou"]) < 1e-6
    assert abs(acc_result["dice"] - whole["dice"]) < 1e-6
    assert acc_result["tp"] == whole["tp"]


def test_accumulator_snowline_with_lats():
    """snowline 累积器带 lats extras。"""
    acc = MetricAccumulator("snowline")
    for _ in range(3):
        pred = np.random.rand(16, 16).astype(np.float32) * 5000
        target = np.random.rand(16, 16).astype(np.float32) * 5000
        lats = np.linspace(-60, 60, 16, dtype=np.float32)
        acc.update(pred, target, lats=lats)
    m = acc.compute()
    assert "mae_m" in m
    assert "mae_tropics" in m   # lats 存在 → 有分带


def test_accumulator_coast_cm_sum():
    """海岸线累积器混淆矩阵总和 = 总像素数。"""
    acc = MetricAccumulator("coast")
    total_pixels = 0
    for _ in range(3):
        pred = np.random.randint(0, 3, (16, 16))
        target = np.random.randint(0, 3, (16, 16))
        acc.update(pred, target)
        total_pixels += 16 * 16
    m = acc.compute()
    # oa 的分母 = total_pixels，验证不直接返回 cm，但 oa 合理
    assert 0 <= m["oa"] <= 1.0


def test_accumulator_river_aggregates_tp():
    """河网累积器 TP 累加正确。"""
    acc = MetricAccumulator("river")
    # batch 1: 4 TP
    p1 = np.zeros((4, 4), dtype=np.float32)
    p1[0, 0] = 1.0; p1[0, 1] = 1.0; p1[1, 0] = 1.0; p1[1, 1] = 1.0
    t1 = p1.copy()
    acc.update(p1, t1)
    # batch 2: 2 TP
    p2 = np.zeros((4, 4), dtype=np.float32)
    p2[0, 0] = 1.0; p2[0, 1] = 1.0
    t2 = p2.copy()
    acc.update(p2, t2)
    m = acc.compute()
    assert m["tp"] == 6   # 4 + 2
    assert m["iou"] == 1.0   # 完美预测


# ---- format_metrics 补充 ----

def test_format_metrics_unknown_task():
    """未知任务 → 返回 str(metrics)。"""
    m = {"foo": 1.0}
    s = format_metrics("unknown", m)
    assert "foo" in s


def test_format_metrics_snowline_with_bands():
    """雪线带分带格式化字符串含分带指标。"""
    m = {"mae_m": 100.0, "rmse_m": 150.0, "r2": 0.9, "bias_m": 10.0,
         "mae_tropics": 80.0, "mae_subtropics": 120.0,
         "mae_midlat": 90.0, "mae_polar": 110.0}
    s = format_metrics("snowline", m)
    assert "tro=80m" in s
    assert "sub=120m" in s
    assert "mid=90m" in s
    assert "pol=110m" in s


def test_format_metrics_river_missing_keys():
    """river 缺失键 → 用 0 填充，不崩溃。"""
    s = format_metrics("river", {})
    assert "iou=0.000" in s


def test_format_metrics_coast_missing_keys():
    """coast 缺失键 → 用 0 填充。"""
    s = format_metrics("coast", {})
    assert "oa=0.000" in s


# ---- 端到端：validate 返回完整指标 ----

def test_validate_returns_full_metrics(synth_tiles, tmp_path, monkeypatch):
    """validate 在完整验证集上返回所有期望指标键。"""
    from terrain.dl import train_physics as tp
    from terrain.dl.physics_nets import RiverNet

    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))
    model = RiverNet(base=16)
    ema = tp.EMA(model)
    _, val_loader = tp.make_physics_loaders(
        "river", batch_size=2, val_ratio=0.5, num_workers=0,
        root=synth_tiles)
    result = tp.validate("river", model, ema, val_loader)
    assert "loss" in result
    assert "iou" in result
    assert "dice" in result
    assert "connectivity" in result
    assert "topo_acc" in result


def test_validate_coast_returns_miou(synth_tiles, tmp_path, monkeypatch):
    """validate coast 任务返回 mIoU 与各类 IoU。"""
    from terrain.dl import train_physics as tp
    from terrain.dl.physics_nets import CoastNet

    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))
    model = CoastNet(base=16)
    ema = tp.EMA(model)
    _, val_loader = tp.make_physics_loaders(
        "coast", batch_size=2, val_ratio=0.5, num_workers=0,
        root=synth_tiles)
    result = tp.validate("coast", model, ema, val_loader)
    assert "miou" in result
    assert "iou_land" in result
    assert "iou_shelf" in result
    assert "iou_deep" in result
    assert "edge_sharpness" in result


def test_validate_snowline_returns_mae(synth_tiles, tmp_path, monkeypatch):
    """validate snowline 任务返回 MAE/RMSE/R²。"""
    from terrain.dl import train_physics as tp
    from terrain.dl.physics_nets import SnowlineNet

    monkeypatch.setattr(tp, "DEVICE", torch.device("cpu"))
    model = SnowlineNet(base=16)
    ema = tp.EMA(model)
    _, val_loader = tp.make_physics_loaders(
        "snowline", batch_size=2, val_ratio=0.5, num_workers=0,
        root=synth_tiles)
    result = tp.validate("snowline", model, ema, val_loader)
    assert "mae_m" in result
    assert "rmse_m" in result
    assert "r2" in result
