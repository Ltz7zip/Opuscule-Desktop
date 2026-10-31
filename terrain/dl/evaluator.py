"""训练中周期性评估器：封装采样 + 指标计算 + 日志记录。

设计
----
- 接收一个 ``sample_fn(n) -> ndarray`` 回调（由训练脚本提供，返回 n 个生成样本）
- 接收可选的真实参考 DEM 列表（用于 FID 与高程曲线对比）
- 周期性调用：``evaluator.step(epoch, sample_fn)`` → 自动采样、算指标、写日志、保存样本图
- 维护历史最佳：基于综合 ``quality_score``（条纹/PSD/FID 加权和），自动保存 best_*.pt

综合质量分定义（越大越好）：
    quality = w1 * stripe_ok + w2 * psd_natural + w3 * (1 - hyps_err) + w4 * (-fid)
其中各分量经 min-max 归一化到 [0,1]。
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import CKPT_DIR, LOG_DIR, SAMPLE_NUM, SAMPLE_SIZE
from .logger import TrainLogger
from . import metrics as M


class Evaluator:
    """周期性评估器。

    Parameters
    ----------
    run_name : str
        与 TrainLogger 同名，便于关联日志。
    ref_dems : list[ndarray], optional
        真实参考 DEM 切片；给定则计算 FID 与高程曲线误差。
    sample_dir : Path
        采样图像保存目录。
    weights : dict
        综合质量分权重，默认 stripe 0.25 / psd 0.25 / hyps 0.25 / fid 0.25。
    """

    def __init__(self, run_name: str,
                 ref_dems: Optional[list[np.ndarray]] = None,
                 sample_dir: Optional[Path] = None,
                 weights: Optional[dict] = None,
                 logger: Optional[TrainLogger] = None):
        """初始化评估器：保存运行名、参考 DEM 与质量权重，创建样本输出目录。"""
        self.run_name = run_name
        self.ref_dems = ref_dems or []
        self.sample_dir = Path(sample_dir or (LOG_DIR / f"samples_{run_name}"))
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.weights = weights or {"stripe": 0.25, "psd": 0.25,
                                   "hyps": 0.25, "fid": 0.25}
        self.logger = logger
        self.best_score = -float("inf")
        self.history: list[dict] = []

    # ---------------- 主入口 ----------------
    def step(self, epoch: int, step: int,
             sample_fn: Callable[[int], np.ndarray],
             n: int = SAMPLE_NUM) -> dict:
        """采样 n 个样本、计算指标、记录日志、返回本轮结果。

        Parameters
        ----------
        epoch, step : int
            当前 epoch 与全局 step，用于日志与文件命名。
        sample_fn : callable
            ``sample_fn(n) -> ndarray (n, H, W)``，返回米单位的生成样本。
        n : int
            采样数量。
        """
        samples = sample_fn(n)
        if samples.ndim != 3:
            samples = samples.squeeze()

        # 逐样本算指标后取均值
        per_sample = [M.compute_all_metrics(s, self.ref_dems or None)
                      for s in samples]
        agg = _aggregate(per_sample)

        # 综合质量分
        score = self._quality_score(agg)
        agg["quality_score"] = score
        agg["epoch"] = epoch
        agg["step"] = step

        # 保存样本可视化
        self._save_samples(samples, step)

        # 日志
        if self.logger is not None:
            self.logger.log_scalars(
                {k: v for k, v in agg.items()
                 if isinstance(v, (int, float)) and k not in ("epoch", "step")},
                step=step, prefix="eval/"
            )
            self.logger.log_text("eval_report",
                                 self._format_report(agg), step=step)

        # 维护最佳
        if score > self.best_score:
            self.best_score = score
            agg["is_best"] = True
        self.history.append(agg)
        return agg

    # ---------------- 综合质量分 ----------------
    def _quality_score(self, agg: dict) -> float:
        """综合质量分 ∈ [0,1]，越大越好。

        各分量归一化：
        - stripe : |log(stripe)| 越小越好，1.0 时最优
        - psd    : α 越接近 1.8 越好
        - hyps   : 1 / (1 + hyps_err)
        - fid    : 1 / (1 + fid)
        """
        w = self.weights
        s_stripe = max(0.0, 1.0 - abs(np.log(max(1e-6, agg["stripe_index"]))))
        s_psd = max(0.0, 1.0 - abs(agg["psd_slope"] - 1.8) / 1.5)
        s_hyps = 1.0 / (1.0 + agg.get("hypsometric_err", 1.0))
        if "fid" in agg:
            s_fid = 1.0 / (1.0 + agg["fid"] * 1e-3)   # FID 数值较大，缩放
        else:
            s_fid = 0.5   # 无参考时中性
        return float(w["stripe"] * s_stripe + w["psd"] * s_psd
                     + w["hyps"] * s_hyps + w["fid"] * s_fid)

    # ---------------- 样本可视化 ----------------
    def _save_samples(self, samples: np.ndarray, step: int) -> None:
        """把一批样本拼成网格图保存为 PNG。"""
        try:
            from PIL import Image
        except ImportError:
            return
        n = len(samples)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        h, w = samples.shape[1:]
        canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
        for i, s in enumerate(samples):
            r, c = i // cols, i % cols
            lo, hi = float(s.min()), float(s.max())
            img = ((s - lo) / max(1e-8, hi - lo) * 255).astype(np.uint8)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
        Image.fromarray(canvas).save(self.sample_dir / f"samples_{step:07d}.png")

    # ---------------- 报告格式化 ----------------
    @staticmethod
    def _format_report(agg: dict) -> str:
        """把评估结果字典格式化为多行文本报告。"""
        lines = [f"Eval @ epoch={agg['epoch']} step={agg['step']}"]
        for k in ("min", "max", "mean", "std", "land_ratio",
                  "stripe_index", "psd_slope", "mean_slope_deg",
                  "hypsometric_err", "fid", "quality_score"):
            if k in agg:
                lines.append(f"  {k:18s}: {agg[k]:.4f}")
        return "\n".join(lines)


def _aggregate(per_sample: list[dict]) -> dict:
    """对一批样本的逐样本指标取均值。"""
    keys = per_sample[0].keys()
    agg = {}
    for k in keys:
        vals = [s[k] for s in per_sample if isinstance(s[k], (int, float))]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg
