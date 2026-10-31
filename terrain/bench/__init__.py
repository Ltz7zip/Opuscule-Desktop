# -*- coding: utf-8 -*-
"""terrain.bench — P4 压测（并发 / 内存 / 降级 / 任务不丢）。

对外 API：
    run_stress(StressConfig) -> dict     跑一轮，返回本轮结果并可写报告
    StressConfig                          配置
    write_reports(rounds, out_dir, cfg)   把多轮结果汇总成一份 json + md
"""
from __future__ import annotations

from .stress import (
    StressConfig as StressConfig,
    read_ledger as read_ledger,
    run_stress as run_stress,
    write_reports as write_reports,
)
from .workloads import run_task as run_task

__all__ = ["run_stress", "StressConfig", "write_reports", "read_ledger",
           "run_task"]
