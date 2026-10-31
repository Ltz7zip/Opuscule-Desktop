"""阶段 3 物理监督网络推理接口：加载训练好的权重做 drop-in 推理。

提供三个网络的推理函数，并给出与现有 terrain_engine 函数对齐的
drop-in 替换封装：

- ``predict_river``       : DEM → 河网概率图
- ``predict_snowline``    : 纬度+高程+大陆度 → 雪线高程
- ``predict_coast``       : DEM → 三类海岸掩码

以及集成辅助：
- ``rivernet_threshold``  : 概率图 → 二值河网（对应 d8_flow_accumulation 阈值输出）
- ``snowline_apply``      : 用预测雪线高程给 DEM 上色（对应 snowline_m 输出）
- ``coast_apply``         : 用三类概率锐化海岸线（对应 ocean_margin_fix 输出）

用法
----
    from terrain.dl.infer_physics import predict_river
    river_prob = predict_river(dem_m)         # (H, W) [0,1]
    river_mask = river_prob > 0.5
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

from .config import DEVICE, CKPT_DIR, elev_to_norm, norm_to_elev, TILE_SIZE
from .physics_nets import (
    RiverNet, SnowlineNet, CoastNet, coast_apply as _coast_apply,
)


# ==================== 模型加载 ====================

def _load_model(task: str, ckpt: str | None = None) -> torch.nn.Module:
    """加载指定任务的 EMA 权重。

    Parameters
    ----------
    task : str  ∈ {"river", "snowline", "coast"}
    ckpt : str, optional  权重路径，默认 CKPT_DIR 下对应文件。
    """
    task = task.lower()
    default_names = {
        "river": "rivernet.pt",
        "snowline": "snowlinenet.pt",
        "coast": "coastnet.pt",
    }
    if task not in default_names:
        raise ValueError(f"未知任务: {task}，可选: {list(default_names)}")

    ckpt_path = Path(ckpt) if ckpt else CKPT_DIR / default_names[task]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"权重未找到: {ckpt_path}。请先运行 train_physics.py 训练。")

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    # 从 checkpoint 读取 base（若无则用默认 16）
    base = state.get("base", 16)

    def _make_model():
        """按任务构造对应的物理网络模型实例。"""
        if task == "river":
            return RiverNet(base=base).to(DEVICE)
        if task == "snowline":
            return SnowlineNet(base=base).to(DEVICE)
        return CoastNet(base=base).to(DEVICE)

    # 优先加载 EMA 权重
    if "ema" in state:
        model = _make_model()
        model.load_state_dict(state["ema"], strict=False)
    else:
        model = _make_model()
        model.load_state_dict(state["model"])
    model.eval()
    return model


# ==================== 分块推理（大图防 OOM） ====================

def _tiled_inference(forward_fn, dem_m: np.ndarray,
                     tile: int = TILE_SIZE, overlap: int = 32) -> np.ndarray:
    """对大图分块推理，重叠区取平均。

    Parameters
    ----------
    forward_fn : callable  (dem_tile_np (H,W) → output_np (C,H,W) or (H,W))
    dem_m : (H, W) 米制高程
    tile : int  块大小
    overlap : int  重叠像素

    Returns
    -------
    ndarray  拼接后的输出，形状与 forward_fn 单块输出一致（C, H, W）或 (H, W)。
    """
    H, W = dem_m.shape
    if H <= tile and W <= tile:
        return forward_fn(dem_m)

    # 计算分块起点
    stride = tile - overlap
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)

    # 探测输出形状
    sample = forward_fn(dem_m[:tile, :tile])
    if sample.ndim == 2:
        out_shape = (H, W)
    else:
        out_shape = (sample.shape[0], H, W)
    accum = np.zeros(out_shape, dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            block = dem_m[y0:y0 + tile, x0:x0 + tile]
            out = forward_fn(block)
            # 加权混合（中心权重高，边缘低，用余弦窗）
            wy = np.hanning(tile + 2)[1:-1]
            wx = np.hanning(tile + 2)[1:-1]
            win = np.outer(wy, wx).astype(np.float32)
            if out.ndim == 3:
                win_b = win[None, ...]
            else:
                win_b = win
            accum[..., y0:y0 + tile, x0:x0 + tile] += out * win_b
            weight[y0:y0 + tile, x0:x0 + tile] += win

    weight = np.maximum(weight, 1e-8)
    if out_shape == (H, W):
        return accum / weight
    return accum / weight[None, ...]


# ==================== 三个推理接口 ====================

@torch.no_grad()
def predict_river(dem_m: np.ndarray, ckpt: str | None = None,
                  tile: int = TILE_SIZE) -> np.ndarray:
    """DEM → 河网概率图。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    ckpt : str, optional  权重路径
    tile : int  分块大小（大图防 OOM）

    Returns
    -------
    (H, W) float32  河网概率 [0, 1]
    """
    model = _load_model("river", ckpt)

    def fwd(block: np.ndarray) -> np.ndarray:
        """单块前向：归一化 DEM 后输入 RiverNet 输出河网概率图。"""
        x = elev_to_norm(block).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
        prob = model(x_t)   # (1,1,H,W)
        return prob.squeeze().cpu().numpy()   # (H, W)

    return _tiled_inference(fwd, dem_m, tile=tile)


@torch.no_grad()
def predict_snowline(dem_m: np.ndarray, lats: np.ndarray | None = None,
                     ckpt: str | None = None,
                     tile: int = TILE_SIZE) -> np.ndarray:
    """纬度+高程+大陆度 → 雪线高程。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    lats : (H,) 纬度数组；None 时用 0 填充
    ckpt : str, optional
    tile : int

    Returns
    -------
    (H, W) float32  雪线高程（米），范围 [0, 6500]
    """
    model = _load_model("snowline", ckpt)
    H, W = dem_m.shape
    if lats is None:
        lats = np.zeros(H, dtype=np.float32)

    # 预计算全局 CI（不参与分块窗口，避免边界跳变）
    from terrain.core.terrain_engine import continentality_index  # type: ignore
    ci_full = continentality_index(dem_m, lats).astype(np.float32)
    lat_full = (np.abs(lats)[:, None] / 90.0).astype(np.float32)
    lat_full = np.broadcast_to(lat_full, (H, W)).astype(np.float32)
    elev_full = elev_to_norm(dem_m).astype(np.float32)

    # 雪线网络感受野小（3 层 CNN），直接整图推理更准
    # 若图太大则按 tile 推理，但需同步切三个通道
    if H <= tile and W <= tile:
        x = np.stack([lat_full, elev_full, ci_full], axis=0)[None]  # (1,3,H,W)
        x_t = torch.from_numpy(x).to(DEVICE)
        pred = model(x_t[:, 0:1], x_t[:, 1:2], x_t[:, 2:3])
        return pred.squeeze().cpu().numpy()

    # 分块推理（同步切三通道）
    stride = tile - 32
    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(W - tile)
    wy = np.hanning(tile + 2)[1:-1]
    wx = np.hanning(tile + 2)[1:-1]
    win = np.outer(wy, wx).astype(np.float32)

    for y0 in ys:
        for x0 in xs:
            lat_b = lat_full[y0:y0 + tile, x0:x0 + tile][None, None]
            el_b = elev_full[y0:y0 + tile, x0:x0 + tile][None, None]
            ci_b = ci_full[y0:y0 + tile, x0:x0 + tile][None, None]
            pred = model(
                torch.from_numpy(lat_b).to(DEVICE),
                torch.from_numpy(el_b).to(DEVICE),
                torch.from_numpy(ci_b).to(DEVICE),
            ).squeeze().cpu().numpy()
            accum[y0:y0 + tile, x0:x0 + tile] += pred * win
            weight[y0:y0 + tile, x0:x0 + tile] += win

    return accum / np.maximum(weight, 1e-8)


@torch.no_grad()
def predict_coast(dem_m: np.ndarray, ckpt: str | None = None,
                  tile: int = TILE_SIZE, return_probs: bool = False
                  ) -> np.ndarray:
    """DEM → 海岸线三类掩码。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    ckpt : str, optional
    tile : int
    return_probs : bool
        True 返回 (3, H, W) 概率图；False 返回 (H, W) argmax 类别图。

    Returns
    -------
    (H, W) int64 类别 ∈ {0,1,2}  或  (3, H, W) float32 概率
    """
    model = _load_model("coast", ckpt)

    def fwd(block: np.ndarray) -> np.ndarray:
        """单块前向：归一化 DEM 后输入 CoastNet 输出三类概率图。"""
        x = elev_to_norm(block).astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logits = model(x_t)   # (1,3,H,W)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3,H,W)
        return probs

    probs = _tiled_inference(fwd, dem_m, tile=tile)   # (3, H, W)
    if return_probs:
        return probs
    return probs.argmax(axis=0).astype(np.int64)


# ==================== Drop-in 替换封装 ====================

def rivernet_threshold(dem_m: np.ndarray, threshold: float = 0.5,
                       ckpt: str | None = None) -> np.ndarray:
    """概率图 → 二值河网掩码（对应 d8_flow_accumulation + 阈值）。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    threshold : float  概率阈值，默认 0.5
    ckpt : str, optional

    Returns
    -------
    (H, W) bool  河网掩码
    """
    prob = predict_river(dem_m, ckpt=ckpt)
    return prob > threshold


def snowline_apply(dem_m: np.ndarray, lats: np.ndarray | None = None,
                   ckpt: str | None = None) -> np.ndarray:
    """用预测雪线高程生成雪线场（drop-in 替换 snowline_m）。

    Returns
    -------
    (H, W) float32  雪线高程（米）
    """
    return predict_snowline(dem_m, lats=lats, ckpt=ckpt)


def coast_sharpen(dem_m: np.ndarray, ckpt: str | None = None,
                  shelf_target_depth: float = -2000.0) -> np.ndarray:
    """用 CoastNet 输出锐化海岸线（drop-in 替换 ocean_margin_fix）。

    在陆架过渡带（class 1 概率高）区域，把高程推向陆架目标深度。

    Parameters
    ----------
    dem_m : (H, W) 米制高程
    ckpt : str, optional
    shelf_target_depth : float  陆架目标深度（米）

    Returns
    -------
    (H, W) float32  锐化后高程
    """
    probs = predict_coast(dem_m, ckpt=ckpt, return_probs=True)  # (3,H,W)
    shelf_prob = probs[1]   # 陆架过渡带概率
    target = np.full_like(dem_m, shelf_target_depth, dtype=np.float32)
    alpha = shelf_prob.astype(np.float32)
    return (1 - alpha) * dem_m.astype(np.float32) + alpha * target


def _ensure_core_on_path() -> None:
    """把 terrain/core 加入 sys.path，便于复用 terrain_engine 函数。"""
    core_dir = Path(__file__).resolve().parents[1] / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))


# ==================== CLI ====================

def main():
    """CLI 入口：按任务调用对应推理接口并保存/打印结果。"""
    import argparse
    p = argparse.ArgumentParser(description="阶段 3 物理网络推理")
    p.add_argument("--task", type=str, required=True,
                   choices=["river", "snowline", "coast"])
    p.add_argument("--dem", type=str, required=True,
                   help="DEM .npy 文件路径（米制高程）")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out", type=str, default=None,
                   help="输出 .npy 路径；省略则打印统计")
    a = p.parse_args()

    dem = np.load(a.dem).astype(np.float32)
    if a.task == "river":
        out = predict_river(dem, ckpt=a.ckpt)
        print(f"river_prob shape={out.shape} range=[{out.min():.3f}, {out.max():.3f}]")
    elif a.task == "snowline":
        out = predict_snowline(dem, ckpt=a.ckpt)
        print(f"snowline shape={out.shape} range=[{out.min():.0f}, {out.max():.0f}]m")
    else:
        out = predict_coast(dem, ckpt=a.ckpt, return_probs=True)
        print(f"coast_probs shape={out.shape} "
              f"land={out[0].mean():.3f} shelf={out[1].mean():.3f} "
              f"deep={out[2].mean():.3f}")

    if a.out:
        np.save(a.out, out)
        print(f"已保存到 {a.out}")


if __name__ == "__main__":
    main()
