# -*- coding: utf-8 -*-
"""``python -m terrain.bench`` —— P4 压测 CLI。

默认跑 3 档并发（1/2/4）各一轮，汇总到同一份 json + md 报告。
"""
from __future__ import annotations

import argparse
import sys

from .stress import StressConfig, run_stress, write_reports


def _levels(text):
    vals = []
    for part in str(text).replace(" ", "").split(","):
        if part:
            vals.append(max(1, int(part)))
    return vals or [1]


def _ladder(text):
    return [int(x) for x in str(text).replace(" ", "").split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m terrain.bench",
                                 description="Terrain Engine P4 压测")
    ap.add_argument("--tasks", type=int, default=8, help="每轮任务数（默认 8）")
    ap.add_argument("--concurrency", default="1,2,4",
                    help="并发档位，逗号分隔（默认 1,2,4 各一轮）")
    ap.add_argument("--mem-budget-mb", type=float, default=480.0,
                    help="单 worker 内存预算（任务增量口径，默认 480）")
    ap.add_argument("--kind", default="pack-build",
                    choices=["pack-build", "synthetic", "lite-render"])
    ap.add_argument("--timeout", type=float, default=120.0, help="单任务超时秒")
    ap.add_argument("--out-dir", default="terrain/bench/reports")
    ap.add_argument("--resume", action="store_true", help="只跑 ledger 里非 ok 的任务")
    ap.add_argument("--kill-after-s", type=float, default=None,
                    help="到点 os._exit(1)，模拟崩溃以验证 resume 续跑")
    ap.add_argument("--degrade", default="2048,1024,512", help="降级阶梯（分辨率）")
    ap.add_argument("--size", type=int, default=512, help="每任务分辨率 / 产物 size")
    ap.add_argument("--crop-size", type=int, default=512, help="pack-build 的裁剪边长")
    ap.add_argument("--alloc-mb", type=float, default=0.0, help="synthetic 额外内存压力")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="synthetic 额外休眠")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--retry", type=int, default=0, help="非内存类失败的额外重试次数")
    args = ap.parse_args(argv)

    ladder = _ladder(args.degrade)
    rounds, base_cfg = [], None
    for conc in _levels(args.concurrency):
        cfg = StressConfig(
            tasks=args.tasks, concurrency=conc, mem_budget_mb=args.mem_budget_mb,
            task_timeout_s=args.timeout, kind=args.kind, retry=args.retry,
            degrade_ladder=ladder, out_dir=args.out_dir, seed=args.seed,
            size=args.size, crop_size=args.crop_size, alloc_mb=args.alloc_mb,
            sleep_s=args.sleep_s, resume=args.resume,
            kill_after_s=args.kill_after_s,
            ledger_name=f"ledger_c{conc}.jsonl",
        )
        print(f"[bench] 并发 {conc}：{cfg.tasks} 任务 · {cfg.kind} · "
              f"预算 {cfg.mem_budget_mb}MB · 超时 {cfg.task_timeout_s}s",
              file=sys.stderr, flush=True)
        result = run_stress(cfg, write_report=False)
        rounds.append(result)
        base_cfg = base_cfg or cfg
        s = result["summary"]
        print(f"[bench] 并发 {conc} 完成：ok={s['n_ok']}/{s['n_tasks']} "
              f"p50={s['p50_s']}s p95={s['p95_s']}s "
              f"吞吐={s['throughput_tps']}/s 峰值RSS={s['peak_total_mb']}MB "
              f"降级={s['n_degraded']} 超时={s['n_timeout']} OOM={s['n_oom']}",
              file=sys.stderr, flush=True)

    jpath, mpath = write_reports(rounds, args.out_dir, base_cfg)
    print(f"report_json={jpath}")
    print(f"report_md={mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
