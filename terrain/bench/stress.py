# -*- coding: utf-8 -*-
"""terrain.bench.stress — P4 压测调度器：并发 / 内存预算 / 降级 / 任务不丢。

设计要点
--------
* 并发：``ProcessPoolExecutor(mp_context=spawn)``；worker 是模块级 ``run_task``。
* 超时：每个 future 包一层 ``--timeout`` 秒；超时 → terminate 该池 + 记 timeout。
* 内存：worker 启动时尽力 ``setrlimit(RLIMIT_AS|DATA)``；macOS 上 `setrlimit`
  返回 EINVAL，于是退回「进程内 guard + 父进程 RSS 采样校验」。
* 降级：OOM / 超预算 → 按 ``degrade_ladder`` 降一档分辨率重试，记录
  ``degraded_from → degraded_to``。
* 任务不丢：执行前先把每个任务以 ``status=pending`` 追加进 ``ledger.jsonl``
  （append + flush + fsync），执行后再追加终态行；``--resume`` 只跑非 ok 任务。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .workloads import DEFAULT_DEM, proc_rss_mb, run_task, self_rss_mb

__all__ = ["StressConfig", "run_stress", "write_reports", "read_ledger",
           "sample_processes"]

_ENV_CACHE = None


# ==================== 配置 ====================
@dataclass
class StressConfig:
    tasks: int = 8
    concurrency: int = 2
    mem_budget_mb: float = 900.0
    task_timeout_s: float = 20.0
    kind: str = "synthetic"
    retry: int = 0
    degrade_ladder: list = field(default_factory=lambda: [2048, 1024, 512])
    out_dir: str = "terrain/bench/reports"
    seed: int = 7

    # —— 负载参数 ——
    size: int = 512                 # 每任务的分辨率（pack-build 也是产物 size）
    crop_size: int = 512            # pack-build 从 world_dem 裁多大
    alloc_mb: float = 0.0           # synthetic 的额外内存压力（按 (size/ref)² 缩放）
    sleep_s: float = 0.0            # synthetic 的额外休眠（仅用于超时用例）
    iters: int = 24

    # —— 运行控制 ——
    resume: bool = False
    warmup: bool = True             # 先预热进程池，避免 spawn + 导入算进时延/超时
    kill_after_s: float | None = None
    ledger_name: str = "ledger.jsonl"
    sample_interval_s: float = 0.2


# ==================== 小工具 ====================
def _utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utcstamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 6)


def _pool_pids(ex):
    procs = getattr(ex, "_processes", None) or {}
    vals = list(procs.values()) if isinstance(procs, dict) else list(procs)
    return [getattr(p, "pid", p) for p in vals]


def sample_processes(pids):
    """批量取 RSS（MB）：先 ``ps -o pid=,rss=``，全 0/失败时回退 libproc。

    本机 macOS 的 ``ps`` 对任何 pid 都返回 0，所以 source 会被记成 libproc。
    """
    out, want = {}, [int(p) for p in pids if p]
    if want:
        try:
            r = subprocess.run(["ps", "-o", "pid=,rss=", "-p",
                                ",".join(map(str, want))],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    kb = int(parts[1])
                    if kb > 0:
                        out[int(parts[0])] = kb / 1024.0
        except (OSError, subprocess.SubprocessError):
            pass
    source = "ps" if out else "libproc"
    for pid in want:
        if pid not in out:
            v = proc_rss_mb(pid)
            if v is not None:
                out[pid] = v
    return out, source


def _env_probe():
    """环境画像：解释器基线（不含 torch）与主机信息。"""
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    bare = 0.0
    try:
        p = subprocess.Popen([sys.executable, "-c",
                              "import numpy, time; time.sleep(4)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 5
        while time.time() < deadline:
            v = proc_rss_mb(p.pid)
            if v:
                bare = round(v, 1)
                break
            if p.poll() is not None:
                break
            time.sleep(0.1)
        p.kill()
        p.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass
    _ENV_CACHE = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "control_plane_rss_mb": bare,
        "note": "本机不是 2C2G；预算用 RLIMIT_AS + RSS 采样模拟，口径为近似",
    }
    return _ENV_CACHE


# ==================== Ledger（任务不丢） ====================
class _Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, rec):
        self._fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self):
        try:
            self._fh.close()
        except Exception:                                     # noqa: BLE001
            pass


def read_ledger(path):
    """读 ledger，返回 {task_id: 最后一条记录}；文件不存在返回 {}。"""
    latest = {}
    p = Path(path)
    if not p.exists():
        return latest
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            if tid:
                latest[tid] = rec
    return latest


# ==================== 任务规划 ====================
def _payload(cfg, i):
    if cfg.kind == "pack-build":
        crop = max(64, int(cfg.crop_size))
        span = max(1, 2048 - crop)
        y0 = (i * 137) % span
        x0 = (i * 211) % span
        return {"dem": DEFAULT_DEM, "crop": [y0, x0, crop, crop],
                "size": int(cfg.size), "seed": int(cfg.seed) + i}
    if cfg.kind == "lite-render":
        return {"size": int(cfg.size), "pack": None, "seed": int(cfg.seed) + i}
    p = {"size": int(cfg.size), "ref_size": int(cfg.size),
         "seed": int(cfg.seed) + i, "iters": int(cfg.iters)}
    if cfg.alloc_mb:
        p["alloc_mb"] = float(cfg.alloc_mb)
    if cfg.sleep_s:
        p["sleep_s"] = float(cfg.sleep_s)
    return p


def _plan(cfg):
    return [{"task_id": f"t{i + 1:04d}", "payload": _payload(cfg, i)}
            for i in range(max(0, int(cfg.tasks)))]


def _attempt_sizes(size, ladder):
    """降级阶梯：从当前 size 起，只保留 ≤ size 的档位。"""
    rungs, seen = [], set()
    for s in [int(size)] + [int(x) for x in ladder or []]:
        if s <= int(size) and s not in seen and s > 0:
            seen.add(s)
            rungs.append(s)
    return rungs or [int(size)]


# ==================== RSS 采样线程 ====================
class _RssSampler(threading.Thread):
    def __init__(self, pids_provider, interval=0.2):
        super().__init__(daemon=True, name="bench-rss")
        self._provider = pids_provider
        self._interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self.samples = []
        self._t0 = time.monotonic()

    def run(self):
        self._sample()
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self):
        pids = [p for p in (self._provider() or []) if p]
        child, source = sample_processes(pids)
        parent = self_rss_mb()
        self.samples.append({
            "t": round(time.monotonic() - self._t0, 2),
            "parent_mb": round(parent, 1),
            "children_mb": round(sum(child.values()), 1),
            "total_mb": round(parent + sum(child.values()), 1),
            "n_children": len(pids),
            "source": source,
        })

    def stop(self):
        self._stop.set()
        self.join(timeout=2)
        self._sample()                     # 收尾补一次，避免短任务 0 采样
        return self.samples


# ==================== 进程池 ====================
def _new_pool(ctx, cfg):
    return cf.ProcessPoolExecutor(max_workers=max(1, int(cfg.concurrency)),
                                  mp_context=ctx)


def _warm_pool(ex, n, timeout=180.0):
    """把 worker 拉起来（spawn + import terrain 约 4s），别算进任务时延。"""
    futs = [ex.submit(run_task, "synthetic",
                      {"size": 8, "iters": 1, "quiet": True, "mem_budget_mb": 0})
            for _ in range(max(1, int(n)))]
    for f in futs:
        try:
            f.result(timeout=timeout)
        except Exception:                                     # noqa: BLE001
            pass


def _kill_pool(ex):
    """超时兜底：kill 掉池内所有 worker（已被占用，无法单独取消）。"""
    procs = getattr(ex, "_processes", None) or {}
    vals = list(procs.values()) if isinstance(procs, dict) else list(procs)
    for p in vals:
        pid = getattr(p, "pid", p)
        try:
            if pid:
                os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:                                         # noqa: BLE001
        pass


# ==================== 主执行 ====================
def run_stress(cfg, *, write_report=True):
    """跑一轮压测，返回本轮结果 dict；默认同时写出 json + md 报告。"""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / cfg.ledger_name
    done = read_ledger(ledger_path) if cfg.resume else {}
    specs = _plan(cfg)
    todo = [s for s in specs
            if not (cfg.resume and done.get(s["task_id"], {}).get("status") == "ok")]
    skipped = len(specs) - len(todo)

    ledger = _Ledger(ledger_path)
    try:
        for spec in todo:                       # 先落 ledger：进程挂了也不丢任务
            ledger.write({"ts": _utcnow(), "event": "pending",
                          "task_id": spec["task_id"], "kind": cfg.kind,
                          "payload": spec["payload"], "status": "pending",
                          "attempts": 0, "concurrency": cfg.concurrency})
        records, series, wall = _execute(cfg, todo, ledger)
    finally:
        ledger.close()

    summary = _summarize(cfg, records, series, wall, skipped, len(specs))
    result = {"utc": _utcnow(), "concurrency": int(cfg.concurrency),
              "config": asdict(cfg), "ledger": str(ledger_path),
              "tasks": records, "rss_series": series, "summary": summary}
    if write_report:
        write_reports([result], cfg.out_dir, cfg)
    return result


def _execute(cfg, todo, ledger):
    states, order = {}, []
    for spec in todo:
        size = int(spec["payload"].get("size") or 0)
        states[spec["task_id"]] = {
            "task_id": spec["task_id"], "kind": cfg.kind,
            "payload": dict(spec["payload"]), "status": "running",
            "attempts": 0, "seconds": 0.0, "rss_mb": 0.0, "task_mb": 0.0,
            "base_rss_mb": 0.0, "error": None, "degraded_from": None,
            "degraded_to": None, "size": size, "bytes": 0,
            "rlimit": None, "skipped": None,
            "concurrency": int(cfg.concurrency), "history": [],
            "_sizes": _attempt_sizes(size, cfg.degrade_ladder),
            "_idx": 0, "_retry_left": int(cfg.retry),
        }
        order.append(spec["task_id"])

    if not states:
        return [], [], 0.0

    if cfg.kill_after_s:
        def _suicide():
            time.sleep(float(cfg.kill_after_s))
            print(f"[bench] --kill-after-s={cfg.kill_after_s} 触发，模拟崩溃退出",
                  file=sys.stderr, flush=True)
            os._exit(1)
        threading.Thread(target=_suicide, daemon=True).start()

    queue = deque(order)
    inflight = {}
    ctx = mp.get_context("spawn")
    holder = {"ex": _new_pool(ctx, cfg)}
    if cfg.warmup:
        _warm_pool(holder["ex"], min(max(1, len(order)), max(1, int(cfg.concurrency))))
    sampler = _RssSampler(lambda: _pool_pids(holder["ex"]), cfg.sample_interval_s)
    sampler.start()
    t_start = time.monotonic()

    def finalize(tid):
        st = states[tid]
        ledger.write({
            "ts": _utcnow(), "event": "result", "task_id": tid,
            "kind": st["kind"], "payload": st["payload"], "status": st["status"],
            "attempts": st["attempts"], "seconds": round(st["seconds"], 3),
            "rss_mb": round(st["rss_mb"], 1), "task_mb": round(st["task_mb"], 1),
            "base_rss_mb": round(st["base_rss_mb"], 1), "error": st["error"],
            "degraded_from": st["degraded_from"], "degraded_to": st["degraded_to"],
            "size": st["size"], "bytes": st["bytes"],
            "concurrency": st["concurrency"], "history": st["history"],
        })

    try:
        while queue or inflight:
            while queue and len(inflight) < max(1, int(cfg.concurrency)):
                tid = queue.popleft()
                st = states[tid]
                payload = dict(st["payload"])
                payload["size"] = st["_sizes"][st["_idx"]]
                payload["mem_budget_mb"] = float(cfg.mem_budget_mb or 0)
                try:
                    fut = holder["ex"].submit(run_task, cfg.kind, payload)
                except (cf.process.BrokenProcessPool, RuntimeError):
                    holder["ex"] = _new_pool(ctx, cfg)
                    if cfg.warmup:
                        _warm_pool(holder["ex"], max(1, int(cfg.concurrency)))
                    queue.appendleft(tid)
                    break
                inflight[fut] = (tid, time.monotonic())
            if not inflight:
                break

            done, _ = cf.wait(list(inflight), timeout=0.1,
                              return_when=cf.FIRST_COMPLETED)
            for fut in done:
                tid, t0 = inflight.pop(fut)
                try:
                    res = fut.result()
                except cf.process.BrokenProcessPool as exc:
                    res = {"status": "error", "seconds": time.monotonic() - t0,
                           "error": f"worker 进程池崩溃: {exc}"}
                except Exception as exc:                      # noqa: BLE001
                    res = {"status": "error", "seconds": time.monotonic() - t0,
                           "error": f"{type(exc).__name__}: {exc}"}
                action = _apply(states[tid], res, cfg)
                if action == "queued":
                    queue.appendleft(tid)
                else:
                    finalize(tid)

            now = time.monotonic()
            expired = [(f, v) for f, v in inflight.items()
                       if now - v[1] > float(cfg.task_timeout_s)]
            if expired:
                for fut, (tid, t0) in expired:
                    st = states[tid]
                    st["attempts"] += 1
                    st["seconds"] += now - t0
                    st["status"] = "timeout"
                    st["error"] = (f"超过 {cfg.task_timeout_s}s 未返回"
                                   f"（size={st['_sizes'][st['_idx']]}，worker 被 kill）")
                    st["history"].append({"status": "timeout",
                                          "seconds": round(now - t0, 3),
                                          "size": st["_sizes"][st["_idx"]]})
                    inflight.pop(fut, None)
                    finalize(tid)
                for fut, (tid, t0) in list(inflight.items()):   # 池被连带 kill
                    st = states[tid]
                    st["attempts"] += 1
                    st["seconds"] += now - t0
                    st["status"] = "crashed"
                    st["error"] = "同一 worker 池因超时被 kill，任务改由 resume 续跑"
                    inflight.pop(fut, None)
                    finalize(tid)
                _kill_pool(holder["ex"])
                holder["ex"] = _new_pool(ctx, cfg)
                if cfg.warmup and (queue or inflight):
                    _warm_pool(holder["ex"], max(1, int(cfg.concurrency)))
    finally:
        series = sampler.stop()
        try:
            holder["ex"].shutdown(wait=False, cancel_futures=True)
        except Exception:                                     # noqa: BLE001
            pass

    wall = time.monotonic() - t_start
    return [{k: v for k, v in states[t].items() if not k.startswith("_")}
            for t in order], series, wall


def _apply(st, res, cfg):
    """吸收一次尝试结果；返回 "queued"（降级/重试）或 "done"。"""
    st["attempts"] += 1
    st["seconds"] += float(res.get("seconds") or 0.0)
    st["rss_mb"] = max(st["rss_mb"], float(res.get("rss_mb") or 0.0))
    st["task_mb"] = max(st["task_mb"], float(res.get("task_mb") or 0.0))
    if res.get("base_rss_mb"):
        st["base_rss_mb"] = float(res["base_rss_mb"])
    if res.get("bytes"):
        st["bytes"] = int(res["bytes"])
    if res.get("rlimit"):
        st["rlimit"] = res["rlimit"]
    if res.get("skipped"):
        st["skipped"] = res["skipped"]
    st["size"] = int(res.get("size") or st["_sizes"][st["_idx"]])
    st["history"].append({k: res.get(k) for k in
                          ("status", "seconds", "rss_mb", "task_mb", "size",
                           "error", "rlimit", "skipped")})

    budget = float(cfg.mem_budget_mb or 0.0)
    over = bool(budget) and float(res.get("task_mb") or 0.0) > budget
    status = res.get("status") or "error"
    err = res.get("error")

    if status == "ok" and not over:
        st["status"] = "ok"
        st["error"] = None
        return "done"

    if status == "oom" or over:
        if st["_idx"] + 1 < len(st["_sizes"]):
            st["_idx"] += 1
            st["degraded_from"] = st["degraded_from"] or st["_sizes"][0]
            st["degraded_to"] = st["_sizes"][st["_idx"]]
            st["status"] = "degraded"
            print(f"[bench] {st['task_id']} 触发降级 "
                  f"{st['degraded_from']}→{st['degraded_to']}"
                  f"（{err or 'RSS 采样超预算'}）", file=sys.stderr, flush=True)
            return "queued"
        st["status"] = "oom"
        st["error"] = err or f"任务增量 {st['task_mb']:.0f}MB 超预算 {budget:.0f}MB，无降级档位"
        return "done"

    if status == "error" and st["_retry_left"] > 0:
        st["_retry_left"] -= 1
        return "queued"

    st["status"] = "error"
    st["error"] = err or "未知失败"
    return "done"


# ==================== 汇总 ====================
def _summarize(cfg, records, series, wall, skipped, planned):
    ok = [r for r in records if r["status"] == "ok"]
    times = [r["seconds"] for r in ok]
    parents = [s["parent_mb"] for s in series] or [0.0]
    children = [s["children_mb"] for s in series] or [0.0]
    totals = [s["total_mb"] for s in series] or [0.0]
    sources = [s["source"] for s in series]
    bases = [r["base_rss_mb"] for r in records if r.get("base_rss_mb")]

    def count(status):
        return sum(1 for r in records if r["status"] == status)

    n_ok = len(ok)
    return {
        "concurrency": int(cfg.concurrency),
        "budget_mb": float(cfg.mem_budget_mb or 0),
        "n_planned": planned,
        "n_tasks": len(records),
        "n_ok": n_ok,
        "n_timeout": count("timeout"),
        "n_oom": count("oom"),
        "n_error": count("error"),
        "n_crashed": count("crashed"),
        "n_failed": len(records) - n_ok,
        "n_degraded": sum(1 for r in records if r.get("degraded_to")),
        "n_over_budget": sum(1 for r in records
                             if r.get("task_mb", 0) > float(cfg.mem_budget_mb or 0) > 0),
        "n_skipped": skipped,
        "completion_rate": round((n_ok + skipped) / planned, 4) if planned else 0.0,
        "p50_s": _pct(times, 50),
        "p95_s": _pct(times, 95),
        "mean_s": round(sum(times) / len(times), 6) if times else 0.0,
        "throughput_tps": round(n_ok / wall, 3) if wall > 0 else 0.0,
        "wall_s": round(wall, 2),
        "peak_total_mb": round(max(totals), 1),
        "peak_parent_mb": round(max(parents), 1),
        "peak_children_mb": round(max(children), 1),
        "worker_base_mb": round(sorted(bases)[len(bases) // 2], 1) if bases else 0.0,
        "rss_samples": len(series),
        "rss_source": max(set(sources), key=sources.count) if sources else "n/a",
        "rlimit_modes": _count_modes(records),
    }


def _count_modes(records):
    modes = {}
    for r in records:
        m = r.get("rlimit")
        if m:
            modes[m] = modes.get(m, 0) + 1
    return modes


# ==================== 报告 ====================
def write_reports(rounds, out_dir, base_cfg):
    """写 ``stress_<utc>.json`` + 同名 ``.md``；返回 (json, md)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = _utcstamp()
    base = out / f"stress_{stamp}"
    n = 1
    while (base.with_suffix(".json")).exists():
        n += 1
        base = out / f"stress_{stamp}_{n}"
    env = _env_probe()

    tasks = []
    for rd in rounds:
        for rec in rd["tasks"]:
            tasks.append(rec)
    total_ok = sum(r["summary"]["n_ok"] for r in rounds)
    total_planned = sum(r["summary"]["n_planned"] for r in rounds)
    overall = {
        "n_rounds": len(rounds),
        "concurrency_levels": [r["concurrency"] for r in rounds],
        "n_planned": total_planned,
        "n_ok": total_ok,
        "n_timeout": sum(r["summary"]["n_timeout"] for r in rounds),
        "n_oom": sum(r["summary"]["n_oom"] for r in rounds),
        "n_error": sum(r["summary"]["n_error"] for r in rounds),
        "n_crashed": sum(r["summary"]["n_crashed"] for r in rounds),
        "n_degraded": sum(r["summary"]["n_degraded"] for r in rounds),
        "completion_rate": round(total_ok / total_planned, 4) if total_planned else 0.0,
        "peak_total_mb": round(max([r["summary"]["peak_total_mb"] for r in rounds] or [0]), 1),
        "best_throughput_tps": max([r["summary"]["throughput_tps"] for r in rounds] or [0]),
    }
    payload = {
        "utc": _utcnow(), "config": asdict(base_cfg) if base_cfg else {},
        "environment": env, "rounds": rounds, "tasks": tasks,
        "summary": overall,
    }
    jpath = base.with_suffix(".json")
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")

    mpath = base.with_suffix(".md")
    mpath.write_text(_render_md(rounds, base_cfg, env, overall), "utf-8")
    print(f"[bench] json → {jpath}")
    print(f"[bench] md   → {mpath}")
    return str(jpath), str(mpath)


def _render_md(rounds, cfg, env, overall):
    L = []
    add = L.append
    add("# Terrain Engine v12 · P4 压测报告")
    add("")
    add(f"- 时间（UTC）：{_utcnow()}")
    add(f"- 主机：{env['platform']} · {env['cpu_count']} 核 · "
        f"解释器基线（不含 torch）≈ {env['control_plane_rss_mb']} MB "
        f"· 含 terrain 的 worker 基线 ≈ "
        f"{rounds[0]['summary']['worker_base_mb'] if rounds else 0} MB")
    add(f"- 负载：{cfg.kind} · 每轮 {cfg.tasks} 任务 · size={cfg.size} · seed={cfg.seed}")
    add(f"- 预算：{cfg.mem_budget_mb} MB/worker（任务增量口径）· 超时 {cfg.task_timeout_s}s "
        f"· 重试 {cfg.retry} · 降级阶梯 {'→'.join(str(x) for x in cfg.degrade_ladder)}")
    add(f"- 内存采集：{rounds[0]['summary']['rss_source'] if rounds else 'n/a'} 采样 200ms；"
        f"RLIMIT 模式 {rounds[0]['summary']['rlimit_modes'] if rounds else {}}")
    add("- 口径：本机 18GB，压不了真 2C2G；预算通过 RLIMIT_AS + RSS 采样模拟，结论是近似")
    add("")
    add("## 1. 汇总")
    add("")
    add("| 并发 | 任务 | 成功 | 超时 | OOM | 降级 | 完成率 | 吞吐(任务/s) | p50(s) | p95(s) | 墙钟(s) | 峰值父 RMB | 峰值子 RMB | 峰值合计 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rounds:
        s = r["summary"]
        add(f"| {r['concurrency']} | {s['n_tasks']} | {s['n_ok']} | {s['n_timeout']} | "
            f"{s['n_oom']} | {s['n_degraded']} | {s['completion_rate']:.0%} | "
            f"{s['throughput_tps']} | {s['p50_s']} | {s['p95_s']} | {s['wall_s']} | "
            f"{s['peak_parent_mb']} | {s['peak_children_mb']} | {s['peak_total_mb']} |")
    add("")
    add(f"合计：{overall['n_planned']} 任务，成功 {overall['n_ok']}，"
        f"超时 {overall['n_timeout']}，OOM {overall['n_oom']}，"
        f"错误 {overall['n_error']}，降级 {overall['n_degraded']}，"
        f"完成率 {overall['completion_rate']:.1%}，"
        f"最高吞吐 {overall['best_throughput_tps']} 任务/s，"
        f"全局峰值 RSS {overall['peak_total_mb']} MB。")
    add("")
    add("## 2. 每任务明细")
    add("")
    add("| 并发 | 任务 | 状态 | 尝试 | 秒 | 增量MB | 绝对MB | 大小 | 降级 | 错误 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in rounds:
        for t in r["tasks"]:
            deg = ""
            if t.get("degraded_to"):
                deg = f"{t.get('degraded_from')}→{t.get('degraded_to')}"
            add(f"| {r['concurrency']} | {t['task_id']} | {t['status']} | {t['attempts']} | "
                f"{t['seconds']} | {t.get('task_mb', 0)} | {t.get('rss_mb', 0)} | "
                f"{t.get('size', '')} | {deg} | {(t.get('error') or '').replace('|', '/')} |")
    add("")
    add("## 3. 结论")
    add("")
    for line in _conclusions(rounds, cfg, env, overall):
        add(line)
    add("")
    return "\n".join(L)


def _conclusions(rounds, cfg, env, overall):
    out = []
    base = rounds[0]["summary"]["worker_base_mb"] if rounds else 0
    best = max(rounds, key=lambda r: r["summary"]["throughput_tps"]) if rounds else None
    if best:
        s = best["summary"]
        out.append(f"1. **吞吐与时延**：{cfg.kind} 在 {s['concurrency']} 并发下吞吐最高 "
                   f"{s['throughput_tps']} 任务/s，p50 {s['p50_s']}s、p95 {s['p95_s']}s，"
                   f"单轮墙钟 {s['wall_s']}s。")
    out.append(f"2. **任务不丢**：所有任务执行前已 append+fsync 到 ledger；"
               f"本轮完成率 {overall['completion_rate']:.1%}，"
               f"超时 {overall['n_timeout']}、OOM {overall['n_oom']}、"
               f"崩溃 {overall['n_crashed']}，非 ok 任务均可由 --resume 续跑。")
    out.append(f"3. **内存**：worker 基线 {base} MB（含 terrain 包导入的 torch/MPS），"
               f"任务增量峰值 {max([t.get('task_mb', 0) for r in rounds for t in r['tasks']] or [0])} MB，"
               f"预算 {cfg.mem_budget_mb} MB；父+子总 RSS 峰值 {overall['peak_total_mb']} MB。")
    out.append(f"4. **降级**：本轮降级 {overall['n_degraded']} 次，"
               f"阶梯 {'→'.join(str(x) for x in cfg.degrade_ladder)}；"
               f"OOM 时降分辨率而非硬扛，符合 P4 要求。")
    out.append("5. **口径诚实说明**：本机 18GB（M3 Pro），不是 2C2G；预算用 "
               "RLIMIT_AS（macOS 返回 EINVAL 时退回进程内 guard）+ 200ms RSS 采样模拟，"
               "结论是近似。真实 2C2G 上控制面不应导入 torch，"
               f"实测无 torch 的解释器基线仅 {env['control_plane_rss_mb']} MB，"
               "这才是 512MB 预算里该有的水位。")
    return out
