# -*- coding: utf-8 -*-
"""test_supervisor.py — terrain/supervisor/ 单元测试.

覆盖范围:
  A. references 加载 (地球参考值集合)
  B. metrics 指标计算 (合成 DEM, 不依赖真实生成)
  C. reflector 反思逻辑 (低/中/高风险分级 + overrides 注入)
  D. learning_log 读写 (log.jsonl + summary.md)
  E. overrides 注入 (白名单 + 追加合并)
  F. 集成 audit_world + reflect_and_learn 全流程

运行方式:
    python -m pytest terrain/tests/test_supervisor.py -v
    python -m pytest terrain/tests/test_supervisor.py -v -k "references"
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile

import numpy as np
import pytest

# ─── sys.path 注入 (沿用 test_visualize_context.py 套路) ───
_HERE = os.path.dirname(os.path.abspath(__file__))
_TERRAIN_DIR = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_TERRAIN_DIR)
for _p in (_PROJECT_ROOT, _TERRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 重定向 reflector/learning_log 的文件路径到临时目录, 避免污染真实文件
import terrain.supervisor.reflector as _R
import terrain.supervisor.learning_log as _LL

_ORIG_OVERRIDES_PATH = _R._OVERRIDES_PATH
_ORIG_LOG_DIR = _LL._LOG_DIR
_ORIG_LOG_PATH = _LL._LOG_PATH
_ORIG_SUMMARY_PATH = _LL._SUMMARY_PATH


@pytest.fixture(autouse=True)
def _tmp_log_dir(tmp_path, monkeypatch):
    """每个测试前重定向 overrides/log/summary 到 tmp_path, 测试后还原.

    保证测试不污染真实 terrain/supervisor/overrides.json 等文件。
    """
    tmp_str = str(tmp_path)
    # 重定向 reflector overrides 路径
    monkeypatch.setattr(_R, "_OVERRIDES_PATH",
                        os.path.join(tmp_str, "overrides.json"))
    # 重定向 learning_log 日志目录
    monkeypatch.setattr(_LL, "_LOG_DIR", tmp_str)
    monkeypatch.setattr(_LL, "_LOG_PATH",
                        os.path.join(tmp_str, "log.jsonl"))
    monkeypatch.setattr(_LL, "_SUMMARY_PATH",
                        os.path.join(tmp_str, "summary.md"))
    yield
    # monkeypatch 自动还原


# ===========================================================================
# A. references 加载测试
# ===========================================================================
def test_earth_references_load():
    """验证 GLOBAL/TEMP_LAT/TEMP_C/BIOME_RATIO_BY_IDX 加载正确."""
    from terrain.supervisor import references as R
    assert R.GLOBAL["land_ratio"] == 0.29
    assert R.GLOBAL["peak_elevation"] == 8848.0
    assert R.GLOBAL["ocean_trench"] == -10994.0
    assert R.GLOBAL["mountain_ratio"] == 0.08
    assert len(R.TEMP_LAT) == len(R.TEMP_C) == 11
    assert R.TEMP_C[0] == 27.0      # 赤道
    assert R.TEMP_C[-1] == -45.0    # 极地
    assert len(R.BIOME_RATIO_BY_IDX) == 9
    # 9 类群系粗略地球占比 (口径有重叠: 苔原含极地荒漠, 沙漠含极地荒漠)
    # sum ≈ 0.865 (陆地总占比, 含口径重叠), 合理范围 [0.5, 1.0]
    assert 0.5 < R.BIOME_RATIO_BY_IDX.sum() < 1.0


def test_precip_ref_is_earth_table():
    """v43: PRECIP_LAT/PRECIP_MM 是地球实测纬向降水, 不再复用模型 base 曲线."""
    from terrain.supervisor.references import PRECIP_LAT, PRECIP_MM
    from terrain.modules.climate_module import _P_LAT, _P_MM
    assert PRECIP_LAT is not _P_LAT and PRECIP_MM is not _P_MM
    assert len(PRECIP_LAT) == len(PRECIP_MM)
    assert PRECIP_LAT[0] == 0.0 and PRECIP_LAT[-1] == 90.0
    # 形态校验: 赤道峰 > 副热带谷, 极地最干
    assert PRECIP_MM[np.argmin(np.abs(PRECIP_LAT - 5))] > 2000
    assert PRECIP_MM[np.argmin(np.abs(PRECIP_LAT - 25))] < 900
    assert PRECIP_MM[-1] == PRECIP_MM.min()


def test_area_weighted_ref_vs_midpoint():
    """高纬带面积加权参考值 > 中点点值 (v43 口径修复的回归保护)."""
    from terrain.supervisor.metrics import _interp_ref, _area_weighted_ref
    from terrain.supervisor.references import PRECIP_LAT, PRECIP_MM
    band = (66.5, 90)
    aw = _area_weighted_ref(band, PRECIP_LAT, PRECIP_MM)
    mid = _interp_ref(band, PRECIP_LAT, PRECIP_MM)
    assert aw > mid
    # 低纬带两者差距远小于高纬带 (cos 权重在 0-10° 变化小)
    band_lo = (0, 10)
    aw_lo = _area_weighted_ref(band_lo, PRECIP_LAT, PRECIP_MM)
    mid_lo = _interp_ref(band_lo, PRECIP_LAT, PRECIP_MM)
    assert abs(aw_lo - mid_lo) < 0.10 * mid_lo


def test_tolerance_bounds():
    """验证 TOLERANCE 各值在合理区间."""
    from terrain.supervisor.references import TOLERANCE, SEVERITY
    assert 0 < TOLERANCE["land_ratio"] < 0.1
    assert 0 < TOLERANCE["peak_elevation"] < 0.3
    assert 0 < TOLERANCE["temp_band"] < 10
    assert 0 < TOLERANCE["precip_band"] < 0.5
    assert 0 < TOLERANCE["biome_ratio"] < 0.2
    assert SEVERITY["info"] < SEVERITY["warn"] < SEVERITY["critical"]


def test_snow_bands_alignment():
    """验证 SNOW_BANDS 与 terrain_engine.snow_area_by_band 一致."""
    from terrain.supervisor.references import SNOW_BANDS
    # terrain_engine.snow_area_by_band 用 [(0,30),(30,50),(50,66.5),(66.5,90.01)]
    assert SNOW_BANDS == [(0, 30), (30, 50), (50, 66.5), (66.5, 90.01)]


def test_land_hist_tail_reaches_everest():
    """陆地高程分位表尾端必须能到达珠峰高度 (v44 尾部口径修复的回归保护).

    旧表 q=1.0 的结点是最高的 2.5% 桶的**均值** 6350.8m, 分位重排后全世界
    最高像素被钉死在 6351m, 珠峰永远不可能出现。现在尾部分 4 段加密, 末端
    等于 references.GLOBAL["peak_elevation"], 且 0.975 分位仍是高原面量级。
    """
    from terrain.core.terrain_engine import _LAND_Q, _LAND_V, _LAND_W_V
    from terrain.supervisor.references import GLOBAL
    assert len(_LAND_Q) == len(_LAND_V) == len(_LAND_W_V)
    assert np.all(np.diff(_LAND_Q) > 0)          # 分位轴严格递增
    assert np.all(np.diff(_LAND_V) >= 0)         # 单调映射 (排名保持)
    assert np.all(np.diff(_LAND_W_V) >= 0)
    assert abs(_LAND_V[-1] - GLOBAL["peak_elevation"]) < 1e-6
    assert abs(_LAND_W_V[-1] - GLOBAL["peak_elevation"]) < 1e-6
    # 尾部陡峭度: 0.975 分位(高原面)必须远低于峰顶, 否则等于把整片陆地拉成 6000m+
    assert _LAND_W_V[-5] < 4000.0 and _LAND_V[-5] < 4000.0


# ===========================================================================
# B. metrics 指标计算测试 (合成 DEM)
# ===========================================================================
def test_compute_global_stats_synthetic():
    """全 0m DEM → land_ratio=0."""
    from terrain.supervisor.metrics import compute_global_stats
    dem = np.zeros((100, 100), dtype=np.float32)
    gs = compute_global_stats(dem)
    assert gs["land_ratio"] == 0.0
    assert gs["peak"] == 0.0
    assert gs["trough"] == 0.0
    assert gs["pixel_count"] == 10000


def test_compute_global_stats_half_land():
    """上半陆地 1000m / 下半海洋 -3000m → land_ratio=0.5."""
    from terrain.supervisor.metrics import compute_global_stats
    dem = np.zeros((100, 100), dtype=np.float32)
    dem[:50] = 1000.0
    dem[50:] = -3000.0
    gs = compute_global_stats(dem)
    assert abs(gs["land_ratio"] - 0.5) < 0.01
    assert gs["peak"] == 1000.0
    assert gs["trough"] == -3000.0
    assert abs(gs["mean_land_elevation"] - 1000.0) < 0.01


def test_compute_global_stats_mountain():
    """山地(>=1500m)占比统计."""
    from terrain.supervisor.metrics import compute_global_stats
    dem = np.zeros((100, 100), dtype=np.float32)
    dem[:10] = 2000.0    # 10% 山地
    dem[10:50] = 500.0   # 40% 低地
    dem[50:] = -1000.0   # 50% 海洋
    gs = compute_global_stats(dem)
    assert abs(gs["land_ratio"] - 0.5) < 0.01
    assert abs(gs["mountain_ratio"] - 0.1) < 0.01


def test_metrics_deterministic():
    """同输入调两次 compute_metrics, 结果 bitwise 一致 (幂等)."""
    from terrain.supervisor.metrics import compute_metrics
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    m1 = compute_metrics(dem, lats, seed=42, size_px=S)
    m2 = compute_metrics(dem, lats, seed=42, size_px=S)
    assert m1.land_ratio == m2.land_ratio
    assert m1.peak_elevation == m2.peak_elevation
    assert m1.ocean_trench == m2.ocean_trench
    assert m1.temp_by_band == m2.temp_by_band
    assert m1.biome_ratio == m2.biome_ratio


def test_metrics_biome_sums_with_ocean():
    """群系占比以陆地像素为分母 (v41 口径): ∑biome_ratio ≈ 1.0, 海洋占比独立 ∈[0,1]."""
    from terrain.supervisor.metrics import compute_metrics
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    m = compute_metrics(dem, lats, seed=42, size_px=S)
    assert abs(sum(m.biome_ratio.values()) - 1.0) < 0.01
    assert 0.0 <= m.biome_ocean_ratio <= 1.0
    # 一半陆地 → 海洋约占全图一半
    assert abs(m.biome_ocean_ratio - 0.5) < 0.02


def test_metrics_downsample_4096():
    """size_px=4096 时内部下采样到 2048, downsampled=True."""
    from terrain.supervisor.metrics import compute_metrics
    # 构造 512² (避免耗时), 但传 size_px=4096 触发下采样逻辑
    S = 512
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    # size_px=4096 > 2048 阈值, 应触发下采样
    m = compute_metrics(dem, lats, seed=42, size_px=4096)
    assert m.downsampled is True
    assert m.size_px == 4096
    # 地形维度仍用原尺寸 dem_m, 不受下采样影响
    assert m.peak_elevation == 500.0
    assert m.ocean_trench == -2000.0


def test_metrics_no_downsample_2048():
    """size_px=2048 不触发下采样."""
    from terrain.supervisor.metrics import compute_metrics
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    m = compute_metrics(dem, lats, seed=42, size_px=2048)
    assert m.downsampled is False


# ===========================================================================
# C. reflector 反思逻辑测试
# ===========================================================================
def test_severity_thresholds():
    """rel=0.04→ok / 0.06→info / 0.16→warn / 0.31→critical."""
    from terrain.supervisor.reflector import _severity
    assert _severity(0.04) == "ok"
    assert _severity(0.06) == "info"
    assert _severity(0.16) == "warn"
    assert _severity(0.31) == "critical"
    assert _severity(-0.31) == "critical"


def _make_test_metrics(land_ratio=0.29, peak=8848.0, trench=-10994.0,
                       mean_land=840.0, mountain=0.08):
    """构造一个可定制的 SupervisorMetrics (供 reflect 测试)."""
    from terrain.supervisor.metrics import SupervisorMetrics
    # 全部设为地球参考值 (容差内), 调用方可只改需要的字段
    return SupervisorMetrics(
        land_ratio=land_ratio, peak_elevation=peak, ocean_trench=trench,
        mean_land_elevation=mean_land, mountain_ratio=mountain,
        temp_by_band={"0-10": 26.0, "10-30": 22.0, "30-50": 14.0,
                       "50-66.5": 3.1, "66.5-90": -29.3},
        temp_ref_by_band={"0-10": 26.0, "10-30": 22.0, "30-50": 14.0,
                          "50-66.5": 3.1, "66.5-90": -29.3},
        precip_by_band={"0-10": 2300.0, "10-30": 1150.0, "30-50": 850.0,
                        "50-66.5": 750.0, "66.5-90": 230.0},
        precip_ref_by_band={"0-10": 2300.0, "10-30": 1150.0, "30-50": 850.0,
                            "50-66.5": 750.0, "66.5-90": 230.0},
        biome_ratio={"苔原": 0.10, "北方针叶林": 0.05, "温带草原": 0.08,
                     "温带落叶林": 0.08, "温带雨林": 0.005,
                     "亚热带荒漠": 0.25, "稀树草原": 0.12,
                     "热带季节林": 0.05, "热带雨林": 0.13},
        biome_ocean_ratio=0.135,
        snow_area_by_band=[0.0, 0.02, 0.45, 0.90],
        psd_slope=-2.0, fractal_dim=1.3, slope_mean=5.0,
        slope_p95=15.0, coherence=0.8,
        seed=42, size_px=2048, downsampled=False,
    )


def test_reflect_in_tolerance_no_reflection():
    """全地球完美值 → reflections 为空."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics()   # 全部等于参考值
    refs = reflect(m)
    # 容差内, 应无反思 (或只有 ok 级, 但 _check_global 在容差内直接 return)
    critical_refs = [r for r in refs if r.severity in ("critical", "warn", "info")]
    assert len(critical_refs) == 0, f"容差内不应有反思, got {[(r.metric, r.severity) for r in refs]}"


def test_reflect_land_ratio_low():
    """land_ratio=0.20 (偏低) → 有 land_ratio 反思 + override_key=sea_level_shift."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics(land_ratio=0.20)
    refs = reflect(m)
    lr = [r for r in refs if r.metric == "land_ratio"]
    assert len(lr) == 1
    assert lr[0].override_key == "sea_level_shift"
    assert lr[0].override_value == 0.02   # 偏低 → 抬升
    assert lr[0].severity in ("critical", "warn")


def test_reflect_land_ratio_high():
    """land_ratio=0.50 (偏高) → override_value=-0.02 (降低)."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics(land_ratio=0.50)
    refs = reflect(m)
    lr = [r for r in refs if r.metric == "land_ratio"]
    assert len(lr) == 1
    assert lr[0].override_key == "sea_level_shift"
    assert lr[0].override_value == -0.02   # 偏高 → 降低


def test_reflect_peak_low():
    """peak=5000 → 有 peak_elevation 反思 + override_key=peak_boost."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics(peak=5000.0)
    refs = reflect(m)
    pk = [r for r in refs if r.metric == "peak_elevation"]
    assert len(pk) == 1
    assert pk[0].override_key == "peak_boost"
    assert pk[0].override_value == 200.0


def test_reflect_biome_critical_not_injected():
    """群系严重偏差 → override_key is None (高风险不注入)."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics()
    # 篡改群系: 苔原 0.10 → 0.40 (严重偏高)
    m.biome_ratio = dict(m.biome_ratio)
    m.biome_ratio["苔原"] = 0.40
    m.biome_ratio["热带雨林"] = 0.01   # 相应降低
    refs = reflect(m)
    biome_refs = [r for r in refs if r.metric.startswith("biome_")]
    assert len(biome_refs) >= 1
    for r in biome_refs:
        assert r.override_key is None, f"群系 {r.metric} 不应有 override_key"
        assert r.override_value is None


def test_reflect_temp_band_deviation():
    """温度带偏差>3°C → 有 temp_band 反思."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics()
    m.temp_by_band = dict(m.temp_by_band)
    m.temp_by_band["0-10"] = 20.0   # 偏差 6°C (26→20)
    refs = reflect(m)
    tb = [r for r in refs if r.metric == "temp_0-10"]
    assert len(tb) == 1
    assert tb[0].override_key == "temp_offset_0-10"


def test_reflect_severity_ordering():
    """反思列表按 severity 降序 (critical → warn → info)."""
    from terrain.supervisor.reflector import reflect
    m = _make_test_metrics(land_ratio=0.20, peak=5000.0)
    refs = reflect(m)
    sev_rank = {"critical": 0, "warn": 1, "info": 2, "ok": 3}
    for i in range(len(refs) - 1):
        assert sev_rank[refs[i].severity] <= sev_rank[refs[i + 1].severity]


# ===========================================================================
# D. learning_log 读写测试
# ===========================================================================
def test_append_log_and_read():
    """写 3 条 → read_all 返回 3 条."""
    from terrain.supervisor.metrics import SupervisorMetrics
    from terrain.supervisor.reflector import Reflection
    from terrain.supervisor.learning_log import append_log, read_all

    m = _make_test_metrics(land_ratio=0.20)
    refs = [Reflection(metric="land_ratio", actual=0.20, reference=0.29,
                       deviation=-0.09, rel_deviation=-0.31,
                       severity="critical", suggestion="test")]
    for i in range(3):
        m.seed = i
        append_log(m, refs, overrides_applied={"k": i})
    hist = read_all()
    assert len(hist) == 3
    assert hist[0]["seed"] == 0
    assert hist[2]["seed"] == 2
    assert hist[2]["overrides_applied"] == {"k": 2}


def test_query_recent():
    """写 5 条 → query_recent(2) 返回最后 2 条."""
    from terrain.supervisor.learning_log import append_log, query_recent
    m = _make_test_metrics()
    for i in range(5):
        m.seed = i
        append_log(m, [], overrides_applied={})
    recent = query_recent(2)
    assert len(recent) == 2
    assert recent[0]["seed"] == 3
    assert recent[1]["seed"] == 4


def test_query_by_metric():
    """按指标名查询."""
    from terrain.supervisor.learning_log import append_log, query_by_metric
    from terrain.supervisor.reflector import Reflection
    m = _make_test_metrics(land_ratio=0.20)
    refs = [Reflection(metric="land_ratio", actual=0.20, reference=0.29,
                       deviation=-0.09, rel_deviation=-0.31,
                       severity="critical", suggestion="t")]
    append_log(m, refs, overrides_applied={})
    m.seed = 99
    append_log(m, [], overrides_applied={})   # 这条无 land_ratio 反思
    res = query_by_metric("land_ratio")
    assert len(res) == 1
    assert res[0]["seed"] == 42


def test_aggregate_stats():
    """聚合统计."""
    from terrain.supervisor.learning_log import append_log, aggregate_stats
    from terrain.supervisor.reflector import Reflection
    m = _make_test_metrics(land_ratio=0.20)
    refs = [Reflection(metric="land_ratio", actual=0.20, reference=0.29,
                       deviation=-0.09, rel_deviation=-0.31,
                       severity="critical", suggestion="t")]
    append_log(m, refs, overrides_applied={})
    m.seed = 7
    append_log(m, refs, overrides_applied={})
    stats = aggregate_stats()
    assert stats["total_runs"] == 2
    assert stats["severity_counts"]["critical"] == 2


def test_summary_md_rebuilt():
    """写日志后 summary.md 含标题."""
    from terrain.supervisor.learning_log import append_log
    m = _make_test_metrics()
    append_log(m, [], overrides_applied={})
    # summary.md 路径已被 fixture 重定向
    from terrain.supervisor.learning_log import _SUMMARY_PATH
    assert os.path.exists(_SUMMARY_PATH)
    with open(_SUMMARY_PATH, encoding="utf-8") as f:
        content = f.read()
    assert "Terrain Supervisor 学习日志" in content


# ===========================================================================
# E. overrides 注入测试
# ===========================================================================
def test_write_overrides_no_apply():
    """apply=False → 不写文件."""
    from terrain.supervisor.reflector import (
        reflect, write_overrides, read_overrides)
    m = _make_test_metrics(land_ratio=0.20)
    refs = reflect(m)
    ov = write_overrides(refs, apply_suggestions=False)
    assert len(ov) > 0
    assert read_overrides() == {}   # 不写文件


def test_write_overrides_low_risk():
    """低风险写文件 → read_overrides 非空且键在白名单."""
    from terrain.supervisor.reflector import (
        reflect, write_overrides, read_overrides, _SAFE_OVERRIDE_KEYS)
    m = _make_test_metrics(land_ratio=0.20, peak=5000.0)
    refs = reflect(m)
    ov = write_overrides(refs, apply_suggestions=True)
    assert "sea_level_shift" in ov
    assert "peak_boost" in ov
    ro = read_overrides()
    for k in ro:
        assert k in _SAFE_OVERRIDE_KEYS


def test_reset_overrides():
    """reset_overrides → read_overrides == {}."""
    from terrain.supervisor.reflector import (
        write_overrides, read_overrides, reset_overrides, reflect)
    m = _make_test_metrics(land_ratio=0.20)
    refs = reflect(m)
    write_overrides(refs, apply_suggestions=True)
    assert read_overrides() != {}
    reset_overrides()
    assert read_overrides() == {}


def test_read_overrides_missing_file():
    """文件不存在 → read_overrides == {}."""
    from terrain.supervisor.reflector import read_overrides
    # fixture 已重定向到 tmp, 文件不存在
    assert read_overrides() == {}


def test_overrides_merge():
    """追加合并: 已有 {a:1}, 写 {b:2} → {a:1, b:2}."""
    from terrain.supervisor.reflector import (
        write_overrides, read_overrides, reflect, _OVERRIDES_PATH)
    # 手动写一个已有 overrides
    os.makedirs(os.path.dirname(_OVERRIDES_PATH), exist_ok=True)
    with open(_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump({"manual_tweak": 0.05}, f)
    # 写入新的低风险
    m = _make_test_metrics(land_ratio=0.20)
    refs = reflect(m)
    write_overrides(refs, apply_suggestions=True)
    ro = read_overrides()
    assert "manual_tweak" in ro   # 保留历史手动调整
    assert "sea_level_shift" in ro   # 新增


# ===========================================================================
# F. 集成 audit_world + reflect_and_learn 全流程
# ===========================================================================
def test_audit_world_synthetic():
    """合成 DEM 跑 audit_world → AuditReport."""
    from terrain.supervisor import audit_world
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    report = audit_world(dem, lats, seed=42, size_px=S)
    assert report.metrics.land_ratio == 0.5
    assert report.metrics.peak_elevation == 500.0
    assert report.metrics.ocean_trench == -2000.0
    assert len(report.reflections) > 0


def test_reflect_and_learn_full_flow():
    """audit_world + reflect_and_learn 全流程 (合成 DEM)."""
    from terrain.supervisor import (
        audit_world, reflect_and_learn, get_overrides, query_recent)
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    report = audit_world(dem, lats, seed=42, size_px=S)
    refs = reflect_and_learn(report, apply_suggestions=True)
    # 验证 overrides 写入
    ov = get_overrides()
    assert len(ov) > 0
    # 验证日志写入
    hist = query_recent(1)
    assert len(hist) == 1
    assert hist[0]["seed"] == 42


def test_reflect_and_learn_no_apply():
    """apply_suggestions=False → 不写 overrides 但写日志."""
    from terrain.supervisor import (
        audit_world, reflect_and_learn, get_overrides, query_recent)
    S = 256
    dem = np.zeros((S, S), dtype=np.float32)
    dem[:S // 2] = 500.0
    dem[S // 2:] = -2000.0
    lats = np.linspace(-89.5, 89.5, S)
    report = audit_world(dem, lats, seed=42, size_px=S)
    reflect_and_learn(report, apply_suggestions=False)
    # 不写 overrides
    assert get_overrides() == {}
    # 但写日志
    hist = query_recent(1)
    assert len(hist) == 1
