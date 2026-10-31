# -*- coding: utf-8 -*-
"""P4 压测编排测试：ledger / resume / 超时 / 降级阶梯 / 报告字段。

全部用 ``synthetic`` 负载（不依赖任何业务模块），配置压到最小，
整个文件在本机 <90s 跑完。
"""
from __future__ import annotations

import json
from pathlib import Path

from terrain.bench import StressConfig, read_ledger, run_stress, write_reports


def _cfg(tmp_path, **kw):
    base = dict(tasks=8, concurrency=2, mem_budget_mb=900.0,
                task_timeout_s=20.0, kind="synthetic", out_dir=str(tmp_path),
                size=64, iters=2, sample_interval_s=0.1)
    base.update(kw)
    return StressConfig(**base)


def _lines(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


# ==================== 1. ledger + 报告字段 ====================
def test_ledger_rows_and_report_fields(tmp_path):
    res = run_stress(_cfg(tmp_path))
    s = res["summary"]
    assert s["n_tasks"] == 8 and s["n_ok"] == 8
    assert s["completion_rate"] == 1.0 and s["n_timeout"] == 0 and s["n_oom"] == 0
    assert s["p50_s"] > 0 and s["p95_s"] >= s["p50_s"]

    rows = _lines(tmp_path / "ledger.jsonl")
    assert len(rows) == 16                        # 8 pending + 8 result
    assert sum(1 for r in rows if r["event"] == "pending") == 8
    finals = [r for r in rows if r["event"] == "result"]
    assert len(finals) == 8
    need = {"task_id", "kind", "payload", "status", "attempts", "seconds",
            "rss_mb", "error", "degraded_from", "degraded_to"}
    for r in finals:
        assert need <= set(r)
        assert r["status"] == "ok" and r["attempts"] >= 1
        assert r["kind"] == "synthetic"
    assert {r["task_id"] for r in finals} == {f"t{i + 1:04d}" for i in range(8)}

    reports = list(tmp_path.glob("stress_*.json"))
    assert len(reports) == 1 and list(tmp_path.glob("stress_*.md"))
    data = json.loads(reports[0].read_text("utf-8"))
    assert {"utc", "config", "environment", "rounds", "tasks", "summary"} <= set(data)
    rd = data["rounds"][0]
    assert {"concurrency", "config", "tasks", "rss_series", "summary"} <= set(rd)
    assert len(rd["tasks"]) == 8
    for key in ("n_ok", "n_timeout", "n_oom", "n_degraded", "n_over_budget",
                "n_skipped", "completion_rate", "p50_s", "p95_s",
                "throughput_tps", "wall_s", "peak_total_mb", "peak_parent_mb",
                "peak_children_mb", "rss_source", "rlimit_modes"):
        assert key in rd["summary"], key
    assert rd["rss_series"], "RSS 采样应有时间序列"
    assert rd["summary"]["rss_source"] in ("ps", "libproc")


# ==================== 2. resume 不重复执行 ====================
def test_resume_does_not_rerun_completed(tmp_path):
    first = run_stress(_cfg(tmp_path, tasks=4), write_report=False)
    assert first["summary"]["n_ok"] == 4
    before = _lines(tmp_path / "ledger.jsonl")
    assert read_ledger(tmp_path / "ledger.jsonl")["t0001"]["status"] == "ok"

    again = run_stress(_cfg(tmp_path, tasks=4, resume=True), write_report=False)
    after = _lines(tmp_path / "ledger.jsonl")

    assert again["summary"]["n_skipped"] == 4
    assert again["summary"]["n_tasks"] == 0
    assert again["summary"]["completion_rate"] == 1.0
    assert len(after) == len(before)               # 没有新增任何行


# ==================== 3. 超时被记录 ====================
def test_timeout_is_recorded(tmp_path):
    cfg = _cfg(tmp_path, tasks=1, concurrency=1, size=32, iters=1,
               sleep_s=60.0, task_timeout_s=2.0)
    res = run_stress(cfg, write_report=False)
    task = res["tasks"][0]
    assert task["status"] == "timeout"
    assert "未返回" in (task["error"] or "")
    assert res["summary"]["n_timeout"] == 1
    assert res["summary"]["completion_rate"] == 0.0
    rows = _lines(tmp_path / "ledger.jsonl")
    assert any(r.get("status") == "timeout" for r in rows)


# ==================== 4. 降级阶梯被触发 ====================
def test_degrade_ladder_triggers(tmp_path):
    cfg = _cfg(tmp_path, tasks=2, concurrency=1, size=2048, iters=1,
               alloc_mb=1500.0, mem_budget_mb=900.0, task_timeout_s=60.0)
    res = run_stress(cfg, write_report=False)
    for task in res["tasks"]:
        assert task["status"] == "ok", task
        assert task["attempts"] == 2
        assert task["degraded_from"] == 2048 and task["degraded_to"] == 1024
        assert task["size"] == 1024
        assert len(task["history"]) == 2
        assert task["history"][0]["status"] == "oom"
        assert task["history"][1]["status"] == "ok"
    assert res["summary"]["n_degraded"] == 2
    assert res["summary"]["n_ok"] == 2


# ==================== 5. 多轮汇总成一份报告 ====================
def test_write_reports_merges_rounds(tmp_path):
    r1 = run_stress(_cfg(tmp_path, tasks=2, concurrency=1, ledger_name="l1.jsonl"),
                    write_report=False)
    r2 = run_stress(_cfg(tmp_path, tasks=2, concurrency=2, ledger_name="l2.jsonl"),
                    write_report=False)
    jpath, mpath = write_reports([r1, r2], tmp_path, _cfg(tmp_path))
    data = json.loads(Path(jpath).read_text("utf-8"))
    assert data["summary"]["n_rounds"] == 2
    assert data["summary"]["concurrency_levels"] == [1, 2]
    assert len(data["tasks"]) == 4
    md = Path(mpath).read_text("utf-8")
    assert "P4 压测报告" in md and "| 并发 |" in md and "结论" in md
