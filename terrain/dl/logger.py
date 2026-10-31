"""训练日志工具：CSV 指标记录 + 可选 TensorBoard。

设计
----
- 始终写 CSV（无额外依赖，便于 ``cat`` / ``pandas`` 事后分析）
- 若环境装了 ``tensorboard``，自动同步写 Scalar / Histogram / Image
- 提供 ``log_image`` 用于记录周期性采样可视化
- 线程安全：单进程训练足够，多进程需自行加锁

使用
----
>>> logger = TrainLogger("ddpm_phase1", log_dir=LOG_DIR)
>>> logger.log_scalars({"loss": 0.42, "lr": 1e-4}, step=1000)
>>> logger.log_image("sample", img_tensor, step=1000)
>>> logger.close()
"""
from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import LOG_DIR

# TensorBoard 可选（无则降级为只写 CSV）
try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:                       # tensorboard 未装 / 缺少(tensorflow 依赖
    SummaryWriter = None                # type: ignore
    _HAS_TB = False


class TrainLogger:
    """训练日志记录器，统一管理 CSV 与（可选）TensorBoard。

    Parameters
    ----------
    run_name : str
        运行名，用作 CSV 文件名前缀和 TensorBoard 子目录。
    log_dir : Path
        日志根目录，默认 ``terrain/cache/logs``。
    use_tb : bool
        是否尝试启用 TensorBoard；环境不支持时自动降级。
    """

    def __init__(self, run_name: str, log_dir: Path = LOG_DIR,
                 use_tb: bool = True):
        """初始化日志记录器：创建日志目录，按环境可用性启用 TensorBoard。"""
        self.run_name = run_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{run_name}.csv"
        self._csv_fields: list[str] = []
        self._csv_started = False

        self.use_tb = use_tb and _HAS_TB
        if self.use_tb:
            self.tb = SummaryWriter(log_dir=str(self.log_dir / f"tb_{run_name}"))
        else:
            self.tb = None

    # ---------- 标量 ----------
    def log_scalars(self, scalars: dict[str, float], step: int,
                    prefix: str = "") -> None:
        """记录一组标量。

        首次调用时按 dict 顺序固定 CSV 字段；后续字段必须一致。
        ``step`` 与 ``time`` 始终作为前两列。
        """
        row: dict[str, Any] = {"step": step, "time": time.time()}
        for k, v in scalars.items():
            row[f"{prefix}{k}" if prefix else k] = float(v)

        # 写 CSV
        if not self._csv_started:
            self._csv_fields = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._csv_fields)
            self._csv_started = True
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._csv_fields])

        # 写 TB
        if self.tb is not None:
            for k, v in scalars.items():
                self.tb.add_scalar(f"{prefix}{k}" if prefix else k, float(v), step)

    # ---------- 图像 ----------
    def log_image(self, tag: str, img, step: int) -> None:
        """记录单张图像。支持 numpy (H,W) / (H,W,C) 或 torch (C,H,W)。"""
        if self.tb is None:
            return
        arr = _to_numpy_img(img)
        self.tb.add_image(tag, arr, step, dataformats="HWC" if arr.ndim == 3 else "HW")

    # ---------- 直方图 ----------
    def log_histogram(self, tag: str, values, step: int) -> None:
        """记录直方图到 TensorBoard（环境不支持时静默跳过）。"""
        if self.tb is None:
            return
        import torch
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(np.asarray(values))
        self.tb.add_histogram(tag, values, step)

    # ---------- 文本 ----------
    def log_text(self, tag: str, text: str, step: int) -> None:
        """记录文本：写 TensorBoard 并追加到同名 .log 文件。"""
        if self.tb is not None:
            self.tb.add_text(tag, text, step)
        # 同时追加到 CSV 旁边的 .log 文本文件
        with open(self.log_dir / f"{self.run_name}.log", "a") as f:
            f.write(f"[step={step}] {text}\n")

    def close(self) -> None:
        """关闭 TensorBoard writer，刷新未写入的事件。"""
        if self.tb is not None:
            self.tb.close()


def _to_numpy_img(img) -> np.ndarray:
    """把 numpy / torch 图像统一成 uint8 [0,255] numpy。"""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):    # (C,H,W) -> (H,W,C)
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = img[..., None]
    # 归一化到 [0,255]
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(1e-8, hi - lo)
    return (img * 255).astype(np.uint8)
