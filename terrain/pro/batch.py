# -*- coding: utf-8 -*-
"""batch — 多种子批量生产（串行 / 进程池），带 ledger 断点续跑。

产物（都在 ``out_dir`` 下）：

    ledger.jsonl         每 seed 一行：seed / status / seconds / pack /
                         lod_levels / error（追加写，便于中断后续跑）
    batch_summary.json   本轮汇总：总数 / 成功 / 失败 / 耗时 / 内存峰值
    seed{S}.seedpack|.mappack  以及 timing.json（由 produce 写）

进程池在 macOS 上使用 ``spawn`` 启动方式，因此 worker 必须是模块级函数
（可 pickle）；``Tier`` 是 frozen dataclass，同样可 pickle。
单个 seed 失败只记 ledger，不中断全局；失败数超过 ``max_failures`` 才收手。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time

from .produce import DEFAULT_OUT, _peak_rss_mb, produce
from .tiers import resolve_tier


def _worker(payload):
    """进程池 / 串行共用的单 seed 任务（模块级，可 pickle）。"""
    seed, tier, out_dir, reuse_dem = payload
    t0 = time.time()
    try:
        r = produce(seed, tier=tier, out_dir=out_dir, reuse_dem=reuse_dem,
                    verbose=False)
        return {
            "seed": int(seed),
            "status": "ok",
            "seconds": round(time.time() - t0, 3),
            "pack": r["mappack"],
            "lod_levels": r["lod_levels"],
            "error": None,
            "reused": bool(r.get("reused")),
        }
    except BaseException as exc:             # noqa: BLE001 —— 单 seed 失败不扩散
        return {
            "seed": int(seed),
            "status": "error",
            "seconds": round(time.time() - t0, 3),
            "pack": None,
            "lod_levels": [],
            "error": f"{type(exc).__name__}: {exc}",
            "reused": False,
        }


def _read_ledger(path):
    """读 ledger.jsonl → {seed: status}（后写的覆盖先写的）。"""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "seed" in rec and "status" in rec:
                done[int(rec["seed"])] = rec["status"]
    return done


def _append_ledger(path, rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def batch_produce(seeds, tier="standard", out_dir=DEFAULT_OUT, *, jobs=1,
                  resume=True, max_failures=3, reuse_dem=None, verbose=True):
    """批量生产多个种子，返回汇总 dict。

    Parameters
    ----------
    seeds : Iterable[int]
        种子列表（重复项自动去重并保持顺序）。
    tier : str | Tier
        档位名或 ``Tier``。
    out_dir : str
        输出目录（相对路径按项目根解析）。
    jobs : int
        ``1`` 串行；``>1`` 用 ``ProcessPoolExecutor``（macOS spawn）。
    resume : bool
        为 True 时跳过 ledger 中已 ``ok`` 的 seed。
    max_failures : int
        累计失败数达到该值即停止后续 seed。
    reuse_dem : str or None
        复用既有 DEM（所有 seed 共用，跳过 ``generate_world``）。
    verbose : bool
        打印每 seed 结果。

    Returns
    -------
    dict
        ``{total, success, failed, skipped, seconds, peak_rss_mb,
        results, ledger, summary}``
    """
    t = resolve_tier(tier)
    out = out_dir if os.path.isabs(str(out_dir)) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), str(out_dir))
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    ledger_path = os.path.join(out, "ledger.jsonl")
    summary_path = os.path.join(out, "batch_summary.json")

    ordered = []
    for s in seeds:
        s = int(s)
        if s not in ordered:
            ordered.append(s)
    done = _read_ledger(ledger_path) if resume else {}
    skipped = [s for s in ordered if done.get(s) == "ok"]
    pending = [s for s in ordered if s not in skipped]

    if verbose:
        print(f"[batch] tier={t.name} 总计 {len(ordered)}，"
              f"跳过 {len(skipped)}，待跑 {len(pending)}，jobs={int(jobs)}",
              flush=True)

    t_all = time.time()
    results = []
    failures = 0
    payloads = [(s, t, out, reuse_dem) for s in pending]

    if jobs and int(jobs) > 1 and len(payloads) > 1:
        ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=int(jobs),
                                    mp_context=ctx) as ex:
            futs = {ex.submit(_worker, p): p[0] for p in payloads}
            for fut in cf.as_completed(futs):
                rec = fut.result()
                results.append(rec)
                _append_ledger(ledger_path, rec)
                if rec["status"] != "ok":
                    failures += 1
                if verbose:
                    print(f"[batch] seed {rec['seed']}: {rec['status']} "
                          f"{rec['seconds']:.1f}s"
                          + (f" — {rec['error']}" if rec["error"] else ""),
                          flush=True)
                if failures >= int(max_failures):
                    for f in futs:
                        f.cancel()
                    if verbose:
                        print(f"[batch] 失败 {failures} 次，停止后续任务",
                              flush=True)
                    break
    else:
        for p in payloads:
            rec = _worker(p)
            results.append(rec)
            _append_ledger(ledger_path, rec)
            if rec["status"] != "ok":
                failures += 1
            if verbose:
                print(f"[batch] seed {rec['seed']}: {rec['status']} "
                      f"{rec['seconds']:.1f}s"
                      + (f" — {rec['error']}" if rec["error"] else ""),
                      flush=True)
            if failures >= int(max_failures):
                if verbose:
                    print(f"[batch] 失败 {failures} 次，停止后续任务", flush=True)
                break

    results.sort(key=lambda r: r["seed"])
    success = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] != "ok")
    summary = {
        "tier": t.name,
        "total": len(ordered),
        "ran": len(results),
        "success": success,
        "failed": failed,
        "skipped": len(skipped),
        "skipped_seeds": skipped,
        "jobs": int(jobs),
        "max_failures": int(max_failures),
        "seconds": round(time.time() - t_all, 3),
        "peak_rss_mb": _peak_rss_mb(),
        "out_dir": out,
        "results": results,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)

    if verbose:
        print(f"[batch] 完成：成功 {success} / 失败 {failed} / 跳过 "
              f"{len(skipped)}，{summary['seconds']:.1f}s，"
              f"峰值 {summary['peak_rss_mb']:.0f} MB", flush=True)
    summary["ledger"] = ledger_path
    summary["summary"] = summary_path
    return summary


__all__ = ["batch_produce"]
