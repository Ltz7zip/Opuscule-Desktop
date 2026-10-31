# -*- coding: utf-8 -*-
"""LearningLog — 反思学习日志 (log.jsonl 追加 + summary.md 累计报告).

设计：
    - log.jsonl: 每次审计追加一条 JSON 记录 (含 timestamp/seed/size_px/
      全指标/全反思/severity 统计), 便于程序化查询历史趋势。
    - summary.md: 人类可读累计报告, 每次审计后重建 (最近 5 次简报表 +
      land_ratio 趋势 + 当前 overrides.json 内容)。

查询接口：
    - read_all(): 读取全部历史记录
    - query_recent(n): 最近 N 次记录
    - query_by_metric(metric): 按指标名查询历史趋势
    - aggregate_stats(): 聚合统计 (各 severity 计数 + 各指标平均偏差)
"""
from __future__ import annotations

import json
import os
import datetime as _dt
from typing import List, Dict, Any

from .metrics import SupervisorMetrics
from .reflector import Reflection, read_overrides

_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH = os.path.join(_LOG_DIR, "log.jsonl")
_SUMMARY_PATH = os.path.join(_LOG_DIR, "summary.md")


def _now() -> str:
    """当前本地时间字符串 (YYYY-MM-DD HH:MM:SS)."""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _metrics_to_dict(m: SupervisorMetrics) -> Dict[str, Any]:
    """把 SupervisorMetrics dataclass 转可 JSON 序列化 dict.

    保留全部字段 (含嵌套 dict/list), 便于历史回溯。
    """
    return dict(
        land_ratio=m.land_ratio,
        peak_elevation=m.peak_elevation,
        ocean_trench=m.ocean_trench,
        mean_land_elevation=m.mean_land_elevation,
        mountain_ratio=m.mountain_ratio,
        temp_by_band=m.temp_by_band,
        temp_ref_by_band=m.temp_ref_by_band,
        precip_by_band=m.precip_by_band,
        precip_ref_by_band=m.precip_ref_by_band,
        biome_ratio=m.biome_ratio,
        biome_ocean_ratio=m.biome_ocean_ratio,
        snow_area_by_band=m.snow_area_by_band,
        psd_slope=m.psd_slope,
        fractal_dim=m.fractal_dim,
        slope_mean=m.slope_mean,
        slope_p95=m.slope_p95,
        coherence=m.coherence,
        seed=m.seed,
        size_px=m.size_px,
        downsampled=m.downsampled,
    )


def _reflection_to_dict(r: Reflection) -> Dict[str, Any]:
    """把 Reflection dataclass 转可 JSON 序列化 dict."""
    return dict(
        metric=r.metric,
        actual=r.actual,
        reference=r.reference,
        deviation=r.deviation,
        rel_deviation=r.rel_deviation,
        severity=r.severity,
        suggestion=r.suggestion,
        override_key=r.override_key,
        override_value=r.override_value,
    )


def _summarize(reflections: List[Reflection]) -> Dict[str, int]:
    """统计各 severity 数量."""
    s = {"critical": 0, "warn": 0, "info": 0, "ok": 0}
    for r in reflections:
        s[r.severity] = s.get(r.severity, 0) + 1
    return s


def append_log(metrics: SupervisorMetrics,
               reflections: List[Reflection],
               overrides_applied: Dict[str, float] = None,
               version: str = "1.0.0") -> None:
    """追加一条记录到 log.jsonl, 并重建 summary.md.

    每条记录结构:
        {
          "timestamp": "2026-09-03 16:00:00",
          "seed": 42,
          "size_px": 2048,
          "version": "1.0.0",
          "metrics": { ...SupervisorMetrics 序列化... },
          "reflections": [ { ...Reflection 序列化... }, ... ],
          "summary": { "critical": 1, "warn": 3, "info": 5, "ok": 12 },
          "overrides_applied": { "sea_level_shift": 0.02, ... }
        }

    Parameters
    ----------
    metrics : SupervisorMetrics
        compute_metrics 的输出。
    reflections : List[Reflection]
        reflect() 的输出。
    overrides_applied : Dict[str, float], optional
        本次写入 overrides.json 的键值对。
    version : str
        supervisor 版本号。
    """
    record = dict(
        timestamp=_now(),
        seed=metrics.seed,
        size_px=metrics.size_px,
        version=version,
        downsampled=metrics.downsampled,
        metrics=_metrics_to_dict(metrics),
        reflections=[_reflection_to_dict(r) for r in reflections],
        summary=_summarize(reflections),
        overrides_applied=overrides_applied or {},
    )
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _update_summary()


def _update_summary() -> None:
    """重建 summary.md (人类可读累计报告).

    结构:
        # Terrain Supervisor 学习日志
        > 累计运行 N 次 | 最近: 2026-09-03 16:00 | 最新 seed: 42

        ## 最近 5 次运行简报
        | 时间 | seed | critical | warn | info | land_ratio | peak |
        |------|------|----------|------|------|------------|------|
        ...

        ## 指标趋势 (land_ratio)
        seed42: 0.271 | seed7: 0.285 | ...

        ## 当前 overrides.json
        ```json
        { "sea_level_shift": 0.02, "peak_boost": 200 }
        ```
    """
    history = read_all()
    lines = ["# Terrain Supervisor 学习日志", ""]
    if not history:
        lines.append("> (尚无记录)")
        _write_summary(lines)
        return
    latest = history[-1]
    lines.append(f"> 累计运行 {len(history)} 次 | 最近: {latest['timestamp']} | "
                  f"最新 seed: {latest['seed']}")
    lines.append("")
    # 最近 5 次简报
    lines.append("## 最近 5 次运行简报")
    lines.append("| 时间 | seed | critical | warn | info | land_ratio | peak |")
    lines.append("|------|------|----------|------|------|------------|------|")
    for r in history[-5:]:
        s = r.get("summary", {})
        m = r.get("metrics", {})
        lr = m.get("land_ratio", 0)
        pk = m.get("peak_elevation", 0)
        lines.append(f"| {r['timestamp']} | {r.get('seed', '?')} | "
                     f"{s.get('critical', 0)} | {s.get('warn', 0)} | "
                     f"{s.get('info', 0)} | {lr*100:.1f}% | {pk:.0f}m |")
    # land_ratio 趋势 (最近 10 次)
    lines.append("")
    lines.append("## 指标趋势 (land_ratio, 最近 10 次)")
    seeds = [f"seed{r.get('seed', '?')}: {r.get('metrics', {}).get('land_ratio', 0):.3f}"
              for r in history[-10:]]
    lines.append(" | ".join(seeds))
    # 当前 overrides
    lines.append("")
    lines.append("## 当前 overrides.json")
    lines.append("```json")
    lines.append(json.dumps(read_overrides(), ensure_ascii=False, indent=2))
    lines.append("```")
    _write_summary(lines)


def _write_summary(lines: List[str]) -> None:
    """写入 summary.md."""
    with open(_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_all() -> List[Dict]:
    """读取全部历史记录 (每行一条 JSON).

    Returns
    -------
    List[Dict]
        历史记录列表, 按时间升序。损坏行跳过。
    """
    if not os.path.exists(_LOG_PATH):
        return []
    out = []
    with open(_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def query_recent(n: int = 5) -> List[Dict]:
    """最近 N 次记录.

    Parameters
    ----------
    n : int
        返回条数 (默认 5)。

    Returns
    -------
    List[Dict]
        最近 N 条记录。
    """
    return read_all()[-n:] if n > 0 else []


def query_by_metric(metric: str) -> List[Dict]:
    """按指标名查询历史趋势 (返回含该指标反思的记录).

    Parameters
    ----------
    metric : str
        指标名 (如 "land_ratio" / "peak_elevation")。

    Returns
    -------
    List[Dict]
        含该指标反思的记录列表。
    """
    return [r for r in read_all()
            if any(x.get("metric") == metric for x in r.get("reflections", []))]


def aggregate_stats() -> Dict[str, Any]:
    """聚合统计: 各 severity 计数 + 各指标平均相对偏差.

    Returns
    -------
    Dict[str, Any]
        total_runs: 总运行次数
        severity_counts: {critical, warn, info, ok} 累计计数
        avg_deviation: {metric: 平均相对偏差}
    """
    hist = read_all()
    if not hist:
        return {}
    sev = {"critical": 0, "warn": 0, "info": 0, "ok": 0}
    metric_dev: Dict[str, List[float]] = {}
    for r in hist:
        for k, v in r.get("summary", {}).items():
            sev[k] = sev.get(k, 0) + v
        for ref in r.get("reflections", []):
            metric_dev.setdefault(ref["metric"], []).append(ref["rel_deviation"])
    return dict(
        total_runs=len(hist),
        severity_counts=sev,
        avg_deviation={k: sum(v) / len(v) for k, v in metric_dev.items()},
    )


def set_log_dir(path: str) -> None:
    """重定向日志目录 (供测试用 tmp_path fixture).

    Parameters
    ----------
    path : str
        新的日志目录绝对路径。
    """
    global _LOG_DIR, _LOG_PATH, _SUMMARY_PATH
    _LOG_DIR = path
    _LOG_PATH = os.path.join(path, "log.jsonl")
    _SUMMARY_PATH = os.path.join(path, "summary.md")
