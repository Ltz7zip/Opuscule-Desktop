# -*- coding: utf-8 -*-
"""TerrainSupervisor — 对超容差指标生成反思建议 + 低风险 overrides 注入.

设计原则：
    - 不修改 terrain_engine.py 源码。
    - 低/中风险调整（海平面/峰值/海沟/山地/温度带/降水带）自动写入 overrides.json，
      供 generate_world 在下次运行时注入。
    - 高风险调整（群系阈值 whittaker_biome 门限）仅打印建议，不自动注入，
      因为群系阈值改动会影响整个气候链稳定性。

严重度分级 (基于相对偏差 rel = (actual - ref) / |ref|):
    info     |rel| >= 0.05   轻微偏差, 关注
    warn     |rel| >= 0.15   明显偏差, 建议调整
    critical |rel| >= 0.30   严重偏差, 必须调整

注入策略 (用户已确认: 低风险自动注入):
    land_ratio 偏低 → sea_level_shift += 0.02  (低风险)
    peak 偏低       → peak_boost += 200        (低风险)
    trench 偏浅     → trench_deepen += 200     (低风险)
    mountain 偏低   → mountain_boost += 0.01   (低风险)
    temp_band 偏差  → temp_offset_{band} += 反向 30%  (中风险, 仅 warn+)
    precip_band 偏差→ precip_offset_{band} += 反向 30% (中风险)
    biome 偏差      → (无) 仅打印建议          (高风险, 不注入)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import os

from .metrics import SupervisorMetrics, BAND_NAMES
from .references import (
    GLOBAL, TOLERANCE, SEVERITY,
    BIOME_RATIO_BY_IDX, BIOME_NAMES,
    SNOW_BANDS, SNOW_RATIO_REF,
)


@dataclass
class Reflection:
    """单条反思记录 (可 JSON 序列化)."""
    metric: str               # 指标名 (如 "land_ratio")
    actual: float             # 实际值
    reference: float          # 地球参考值
    deviation: float          # 绝对偏差 (actual - reference)
    rel_deviation: float      # 相对偏差 (actual - reference) / |reference|
    severity: str             # "info" | "warn" | "critical"
    suggestion: str           # 调整建议 (人类可读)
    override_key: Optional[str] = None    # 若可注入, 对应 overrides.json 的键
    override_value: Optional[float] = None  # 建议注入值


def _severity(rel: float) -> str:
    """根据相对偏差判定严重度.

    Parameters
    ----------
    rel : float
        相对偏差 (actual - ref) / |ref|。

    Returns
    -------
    str
        "info" / "warn" / "critical"。
    """
    ar = abs(rel)
    if ar >= SEVERITY["critical"]:
        return "critical"
    if ar >= SEVERITY["warn"]:
        return "warn"
    if ar >= SEVERITY["info"]:
        return "info"
    return "ok"


def reflect(metrics: SupervisorMetrics) -> List[Reflection]:
    """对每个指标对比地球参考, 输出超容差 Reflection 列表.

    遍历五维指标 (地形/温度/降水/群系/雪盖), 对每个超容差项生成 Reflection。
    在容差内的指标不生成 Reflection (静默)。

    Parameters
    ----------
    metrics : SupervisorMetrics
        compute_metrics 的输出。

    Returns
    -------
    List[Reflection]
        反思列表, 按 severity 降序 (critical → warn → info)。
    """
    out: List[Reflection] = []

    # ─── 地形维度 (相对容差, mountain_ratio 用绝对容差) ───
    _check_global(out, "land_ratio", metrics.land_ratio,
                  GLOBAL["land_ratio"], TOLERANCE["land_ratio"],
                  is_relative=False,
                  suggestion_fmt=("陆地比 {actual} vs 地球 {ref} "
                                  "(偏低→sea_level_shift抬升 / 偏高→降低)"),
                  override_key="sea_level_shift",
                  override_value=0.02 if metrics.land_ratio < GLOBAL["land_ratio"] else -0.02)
    _check_global(out, "peak_elevation", metrics.peak_elevation,
                  GLOBAL["peak_elevation"], TOLERANCE["peak_elevation"],
                  is_relative=True,
                  suggestion_fmt="最高峰 {actual}m vs 地球 {ref}m",
                  override_key="peak_boost",
                  override_value=200.0 if metrics.peak_elevation < GLOBAL["peak_elevation"] else -200.0)
    _check_global(out, "ocean_trench", metrics.ocean_trench,
                  GLOBAL["ocean_trench"], TOLERANCE["ocean_trench"],
                  is_relative=True,
                  suggestion_fmt="海沟 {actual}m vs 地球 {ref}m",
                  override_key="trench_deepen",
                  override_value=-200.0 if metrics.ocean_trench > GLOBAL["ocean_trench"] else 200.0)
    _check_global(out, "mean_land_elevation", metrics.mean_land_elevation,
                  GLOBAL["mean_land_elevation"], TOLERANCE["mean_land_elevation"],
                  is_relative=True,
                  suggestion_fmt="陆地均高 {actual}m vs 地球 {ref}m")
    _check_global(out, "mountain_ratio", metrics.mountain_ratio,
                  GLOBAL["mountain_ratio"], TOLERANCE["mountain_ratio"],
                  is_relative=False,
                  suggestion_fmt="山地比 {actual} vs 地球 {ref}",
                  override_key="mountain_boost",
                  override_value=0.01 if metrics.mountain_ratio < GLOBAL["mountain_ratio"] else -0.01)

    # ─── 温度维度 (按 5 纬度带, 绝对容差 3°C) ───
    for name in BAND_NAMES:
        a = metrics.temp_by_band.get(name, float('nan'))
        r = metrics.temp_ref_by_band.get(name, float('nan'))
        if not (np_finite(a) and np_finite(r)):
            continue
        dev = a - r
        if abs(dev) > TOLERANCE["temp_band"]:
            out.append(Reflection(
                metric=f"temp_{name}", actual=float(a), reference=float(r),
                deviation=float(dev), rel_deviation=float(dev / (abs(r) + 1e-6)),
                severity="warn" if abs(dev) > 6 else "info",
                suggestion=f"温度带 {name}: {a:.1f}°C vs 地球 {r:.1f}°C (偏差 {dev:+.1f}°C)",
                override_key=f"temp_offset_{name}",
                override_value=round(-dev * 0.3, 2),  # 反向修正 30%
            ))

    # ─── 降水维度 (按 5 纬度带, 相对容差 30%) ───
    for name in BAND_NAMES:
        a = metrics.precip_by_band.get(name, float('nan'))
        r = metrics.precip_ref_by_band.get(name, float('nan'))
        if not (np_finite(a) and np_finite(r)):
            continue
        rel = (a - r) / (abs(r) + 1e-6)
        if abs(rel) > TOLERANCE["precip_band"]:
            out.append(Reflection(
                metric=f"precip_{name}", actual=float(a), reference=float(r),
                deviation=float(a - r), rel_deviation=float(rel),
                severity=_severity(rel),
                suggestion=f"降水带 {name}: {a:.0f}mm vs 地球 {r:.0f}mm (相对 {rel*100:+.0f}%)",
                override_key=f"precip_offset_{name}",
                override_value=round((r - a) * 0.3, 1),  # 反向 30%
            ))

    # ─── 群系维度 (9 类, 绝对容差 10%, 高风险不注入) ───
    for i, name in enumerate(BIOME_NAMES):
        a = metrics.biome_ratio.get(str(name), 0.0)
        r = float(BIOME_RATIO_BY_IDX[i])
        dev = a - r
        if abs(dev) > TOLERANCE["biome_ratio"]:
            out.append(Reflection(
                metric=f"biome_{name}", actual=float(a), reference=r,
                deviation=float(dev), rel_deviation=float(dev / (r + 1e-6)),
                severity="critical" if abs(dev) > 0.20 else "warn",
                suggestion=(f"群系 {name}: {a*100:.1f}% vs 地球 {r*100:.1f}% "
                            f"(需调 whittaker_biome 阈值, 高风险, 不自动注入)"),
                override_key=None,   # 高风险, 不写入 overrides
                override_value=None,
            ))

    # ─── 雪盖维度 (按 4 纬度带, 绝对容差 15%) ───
    for i, (lo, hi) in enumerate(SNOW_BANDS):
        a = float(metrics.snow_area_by_band[i]) if i < len(metrics.snow_area_by_band) else 0.0
        r = float(SNOW_RATIO_REF[i])
        dev = a - r
        if abs(dev) > TOLERANCE["snow_band"]:
            out.append(Reflection(
                metric=f"snow_{lo}-{hi}", actual=a, reference=r,
                deviation=float(dev), rel_deviation=float(dev / (r + 1e-6)),
                severity="warn" if abs(dev) > 0.25 else "info",
                suggestion=f"雪盖带 {lo}-{hi}°: {a*100:.1f}% vs 地球 {r*100:.1f}%",
            ))

    # 按 severity 降序: critical > warn > info > ok
    sev_rank = {"critical": 0, "warn": 1, "info": 2, "ok": 3}
    out.sort(key=lambda x: sev_rank.get(x.severity, 9))
    return out


def _check_global(out: List[Reflection], metric: str, actual: float,
                  reference: float, tolerance: float,
                  is_relative: bool, suggestion_fmt: str,
                  override_key: Optional[str] = None,
                  override_value: Optional[float] = None) -> None:
    """通用地球参考检查 (地形维度复用).

    Parameters
    ----------
    metric : str
        指标名 (如 "land_ratio")。
    actual : float
        实际值。
    reference : float
        地球参考值。
    tolerance : float
        容差 (绝对值或相对值, 由 is_relative 决定)。
    is_relative : bool
        True: 容差为相对值 (|dev/ref| > tol 才反思)。
        False: 容差为绝对值 (|dev| > tol 才反思)。
    suggestion_fmt : str
        建议格式串 (用 {actual} {ref} 占位符, 不带格式说明符;
        百分比显示由本函数根据 metric 名自动处理)。
    override_key : str, optional
        若可注入, overrides.json 的键名。
    override_value : float, optional
        建议注入值。
    """
    dev = actual - reference
    ref_abs = abs(reference) if reference != 0 else 1.0
    rel = dev / ref_abs
    if is_relative:
        exceed = abs(rel) > tolerance
    else:
        exceed = abs(dev) > tolerance
    if not exceed:
        return
    # 百分比类指标 (ratio) 显示 ×100; 其他用原值
    is_ratio = "ratio" in metric
    if is_ratio:
        actual_str = f"{actual*100:.1f}%"
        ref_str = f"{reference*100:.1f}%"
    else:
        actual_str = f"{actual:.1f}"
        ref_str = f"{reference:.1f}"
    suggestion = suggestion_fmt.format(actual=actual_str, ref=ref_str)
    out.append(Reflection(
        metric=metric, actual=float(actual), reference=float(reference),
        deviation=float(dev), rel_deviation=float(rel),
        severity=_severity(rel),
        suggestion=suggestion,
        override_key=override_key, override_value=override_value,
    ))


def np_finite(x) -> bool:
    """安全判断 x 是否有限 (兼容 nan)."""
    try:
        import math
        return not (math.isnan(x) or math.isinf(x))
    except (TypeError, ValueError):
        return False


# ─── overrides.json 注入 (低风险 Reflection → JSON) ───
_OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "overrides.json")

# 安全注入白名单 (仅低/中风险键允许写入 overrides)
# 群系类 (biome_*) 永不在此白名单, 因为 reflect() 对群系 override_key=None
_SAFE_OVERRIDE_KEYS = {
    "sea_level_shift", "peak_boost", "trench_deepen", "mountain_boost",
    "temp_offset_0-10", "temp_offset_10-30", "temp_offset_30-50",
    "temp_offset_50-66.5", "temp_offset_66.5-90",
    "precip_offset_0-10", "precip_offset_10-30", "precip_offset_30-50",
    "precip_offset_50-66.5", "precip_offset_66.5-90",
}


def write_overrides(reflections: List[Reflection],
                    apply_suggestions: bool = True) -> Dict[str, float]:
    """把低风险 Reflection 写入 overrides.json.

    安全策略:
        - 仅 override_key 在 _SAFE_OVERRIDE_KEYS 白名单内的才写入
        - severity == "critical" 的群系类调整不写入 (高风险)
        - apply_suggestions=False 时仅返回建议 dict, 不写文件

    Parameters
    ----------
    reflections : List[Reflection]
        reflect() 的输出。
    apply_suggestions : bool
        True: 写入 overrides.json (默认, 用户已确认自动注入)。
        False: 仅返回建议 dict, 不写文件。

    Returns
    -------
    Dict[str, float]
        本次写入的 overrides (键值对)。
    """
    overrides = {}
    for r in reflections:
        if r.override_key is None or r.override_value is None:
            continue
        if r.override_key not in _SAFE_OVERRIDE_KEYS:
            continue
        if r.severity == "critical" and r.metric.startswith("biome_"):
            continue  # 群系 critical 不自动注入
        overrides[r.override_key] = r.override_value
    if apply_suggestions and overrides:
        # 追加式合并 (保留历史手动调整, 不被覆盖)
        existing = read_overrides()
        existing.update(overrides)
        with open(_OVERRIDES_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    return overrides


def read_overrides() -> Dict[str, float]:
    """读取 overrides.json (供 generate_world 注入). 不存在返回 {}.

    Returns
    -------
    Dict[str, float]
        overrides 键值对。文件不存在或解析失败返回空 dict。
    """
    if not os.path.exists(_OVERRIDES_PATH):
        return {}
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 类型安全: 确保值是 float
            return {str(k): float(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {}


def reset_overrides() -> None:
    """清空 overrides.json (--reset-overrides CLI 用)."""
    with open(_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)
