# -*- coding: utf-8 -*-
"""terrain.supervisor — 地形监督模块 (审计 + 反思 + 学习日志).

公开入口:
    audit_world(dem_m, lats, seed, size_px, tect_ctx=None) -> AuditReport
        审计一整个世界: 计算 metrics → 生成 reflections。
    reflect_and_learn(report, apply_suggestions=True) -> List[Reflection]
        反思 + 写学习日志 + (可选) 写 overrides.json + 终端简报。
    get_overrides() -> Dict[str, float]
        供 generate_world 注入: 读取 overrides.json。
    reset_overrides() -> None
        清空 overrides.json。
    query_recent(n) / query_by_metric(metric) / aggregate_stats()
        学习日志查询 (供 CLI 用)。

CLI:
    python -m terrain.supervisor --seed 42 --size-px 2048 [--audit-only]
    python -m terrain.supervisor --history 5
    python -m terrain.supervisor --reset-overrides

工作流 (生成 → 审计 → 反思 → 注入 → 再生成):
    1. generate_world 末尾调用 audit_world + reflect_and_learn
    2. reflect_and_learn 写 log.jsonl + summary.md + (可选) overrides.json
    3. 下次 generate_world 在骨架细化前读 get_overrides() 注入低风险调整
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

from .metrics import SupervisorMetrics, compute_metrics
from .reflector import (
    Reflection, reflect, write_overrides, read_overrides,
    reset_overrides as _reset_overrides,
)
from .learning_log import (
    append_log, query_recent, query_by_metric, aggregate_stats,
)

__version__ = "1.0.0"
__all__ = [
    "AuditReport", "SupervisorMetrics", "Reflection",
    "audit_world", "reflect_and_learn", "get_overrides", "reset_overrides",
    "query_recent", "query_by_metric", "aggregate_stats",
    "compute_metrics", "reflect", "__version__",
]


@dataclass
class AuditReport:
    """审计报告 (metrics + reflections + overrides_applied)."""
    metrics: SupervisorMetrics
    reflections: List[Reflection]
    overrides_applied: Dict[str, float] = field(default_factory=dict)


def audit_world(dem_m: np.ndarray, lats: np.ndarray, seed: int, size_px: int,
                tect_ctx: Optional[dict] = None) -> AuditReport:
    """审计一整个世界: 计算 metrics → 生成 reflections.

    复用 render_world 的气候因果链 (generate_world.py L200-216), 幂等重算
    T/P/biome, 与渲染输出 bitwise 一致。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        最终米制高程 (generate_world 输出, 正值陆地 / 负值海洋)。
    lats : ndarray (S,) float32
        纬度数组 (像素中心对齐)。
    seed : int
        世界种子 (供 precipitation_field 的 RNG 重建)。
    size_px : int
        DEM 单边像素数 (>2048 时内部下采样到 2048 审计, 省 30-60s)。
    tect_ctx : dict, optional
        构造场上下文 (与 render_world 同口径; None 时退化, 不影响气候计算)。

    Returns
    -------
    AuditReport
        含 metrics (SupervisorMetrics) + reflections (List[Reflection])。
    """
    metrics = compute_metrics(dem_m, lats, seed, size_px, tect_ctx=tect_ctx)
    reflections = reflect(metrics)
    return AuditReport(metrics=metrics, reflections=reflections,
                       overrides_applied={})


def reflect_and_learn(report: AuditReport,
                      apply_suggestions: bool = True) -> List[Reflection]:
    """反思 + 写学习日志 + (可选) 写 overrides.json + 终端简报.

    副作用:
        1. 写 overrides.json (apply_suggestions=True 时, 低风险项自动注入)
        2. 追加 log.jsonl
        3. 重建 summary.md
        4. 终端打印审计简报

    Parameters
    ----------
    report : AuditReport
        audit_world 的输出。
    apply_suggestions : bool
        True (默认): 写 overrides.json (用户已确认自动注入)。
        False: 仅返回反思, 不写 overrides。

    Returns
    -------
    List[Reflection]
        反思列表 (与 report.reflections 相同)。
    """
    overrides = write_overrides(report.reflections,
                                apply_suggestions=apply_suggestions)
    report.overrides_applied = overrides
    append_log(report.metrics, report.reflections,
               overrides_applied=overrides, version=__version__)
    _print_brief(report)
    return report.reflections


def get_overrides() -> Dict[str, float]:
    """供 generate_world 注入: 读取 overrides.json.

    Returns
    -------
    Dict[str, float]
        overrides 键值对。文件不存在返回空 dict。
    """
    return read_overrides()


def reset_overrides() -> None:
    """清空 overrides.json (--reset-overrides CLI 用)."""
    _reset_overrides()


def _print_brief(report: AuditReport) -> None:
    """终端打印审计简报 (关键指标 + 偏差 + severity 统计).

    格式:
        ============================================================
        [Supervisor] 审计简报 | seed=42 size=2048
          陆地 27.1% (地球 29%) | 峰 8123m (8848) | 沟 -9876m (-10994) | 山地 7.2% (8%)
          反思统计: critical=1 warn=3 info=5 ok=12
          关键反思 (前 5):
            [CRITICAL] land_ratio               陆地比 27.1% vs 地球 29.0% ...
            ...
          已注入 overrides: {'sea_level_shift': 0.02, ...}
        ============================================================
    """
    m = report.metrics
    s = {"critical": 0, "warn": 0, "info": 0, "ok": 0}
    for r in report.reflections:
        s[r.severity] = s.get(r.severity, 0) + 1
    ds = " (下采样到2048审计)" if m.downsampled else ""
    print("\n" + "=" * 60)
    print(f"[Supervisor] 审计简报 | seed={m.seed} size={m.size_px}{ds}")
    print(f"  陆地 {m.land_ratio*100:.1f}% (地球 29%) | "
          f"峰 {m.peak_elevation:.0f}m (8848) | "
          f"沟 {m.ocean_trench:.0f}m (-10994) | "
          f"山地 {m.mountain_ratio*100:.1f}% (8%)")
    print(f"  反思统计: critical={s['critical']} warn={s['warn']} "
          f"info={s['info']} ok={s['ok']}")
    if report.reflections:
        print("  关键反思 (前 5):")
        for r in report.reflections[:5]:
            print(f"    [{r.severity.upper():8s}] {r.metric:25s} {r.suggestion}")
    if report.overrides_applied:
        print(f"  已注入 overrides: {report.overrides_applied}")
    print("=" * 60)
