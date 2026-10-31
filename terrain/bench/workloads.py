# -*- coding: utf-8 -*-
"""terrain.bench.workloads — P4 压测的三个负载（统一在**子进程**里执行）。

统一协议::

    run_task(kind, payload) -> dict

``kind`` 取值：

    pack-build   读 world_dem.npy 的裁剪块 → build_seedpack_from_world → 临时 .seedpack
    lite-render  惰性 ``from terrain.lite import ...`` 渲染（terrain.lite 缺失时优雅跳过）
    synthetic    纯 numpy 的 fBm + 侵蚀式迭代（不依赖业务模块，作为稳定基线）

返回字段（按负载可选）::

    kind / status("ok"|"oom"|"error") / seconds / rss_mb / base_rss_mb / task_mb
    size / bytes / rlimit / checksum / skipped / error / _keep

payload 公共键：

    mem_budget_mb  任务可用的内存增量预算（RLIMIT_AS|DATA + 进程内 guard）
    ``base_rss_mb`` 由 worker 自己测量（解释器 + 依赖库基线），预算按
    ``当前 RSS - 基线`` 判定 —— 2C2G 上控制面本不该导入 torch，
    而本机 ``import terrain`` 会拉起 torch/MPS，基线约 900MB，必须单列。
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import sys
import tempfile
import time

import numpy as np

MB = 1024.0 * 1024.0
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEM = os.path.join(os.path.dirname(_HERE), "output", "world_dem.npy")


# ==================== 1. 进程内存读数 ====================
# 本机 macOS 的 ``ps -o rss=`` 对任何 pid 都返回 0（内核/沙箱不暴露），
# 因此除 ps 外再提供 libproc 这条路径：proc_pid_rusage(RUSAGE_INFO_V2)
# 里的 ri_resident_size 就是常驻集。
_LIBPROC = None
_LIBPROC_TRIED = False
_RI_RESIDENT_OFF = 16 + 8 * 6          # uuid[16] + 6×u64 之后是 ri_resident_size


def _libproc():
    global _LIBPROC, _LIBPROC_TRIED
    if _LIBPROC_TRIED:
        return _LIBPROC
    _LIBPROC_TRIED = True
    if sys.platform != "darwin":
        return None
    try:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _LIBPROC.proc_pid_rusage.restype = ctypes.c_int
    except OSError:
        _LIBPROC = None
    return _LIBPROC


def proc_rss_mb(pid):
    """单进程常驻集（MB）；拿不到返回 None。macOS 走 libproc。"""
    lib = _libproc()
    if lib is None or not pid:
        return None
    buf = ctypes.create_string_buffer(1024)
    if lib.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
        return None
    return ctypes.c_uint64.from_buffer(buf, _RI_RESIDENT_OFF).value / MB


def self_rss_mb():
    """当前进程常驻集（MB）。"""
    v = proc_rss_mb(os.getpid())
    if v is not None:
        return float(v)
    try:                                    # 非 macOS 兜底
        with open("/proc/self/statm", "r") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MB
    except Exception:
        return 0.0


def self_peak_rss_mb():
    """进程生命周期内的 RSS 峰值（MB）；macOS 的 ru_maxrss 单位是字节。"""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru) / MB if sys.platform == "darwin" else float(ru) / 1024.0


# ==================== 2. 内存预算 ====================
def apply_mem_budget(limit_mb):
    """尽力把软上限压到 ``limit_mb``；返回实际生效的方式。

    macOS 对 ``RLIMIT_AS`` / ``RLIMIT_DATA`` 的 ``setrlimit`` 返回 EINVAL
    （Python 抛 ``ValueError: current limit exceeds maximum limit``），
    此时返回 ``{"rlimit": "unsupported"}``，由进程内 guard + 父进程 RSS 采样兜底。
    """
    info = {"budget_mb": float(limit_mb), "rlimit": "off", "note": ""}
    if not limit_mb or limit_mb <= 0:
        info["note"] = "未设置预算"
        return info
    want = int(float(limit_mb) * MB)
    last = ""
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        res = getattr(resource, name, None)
        if res is None:
            continue
        try:
            resource.setrlimit(res, (want, resource.RLIM_INFINITY))
            info.update(rlimit=name, note=f"setrlimit({name}, {want // 2**20}MB)")
            return info
        except (ValueError, OSError) as exc:      # EINVAL → ValueError
            last = f"{name}: {exc}"
    info.update(rlimit="unsupported", note=last or "本平台不支持")
    return info


def _guard(budget_mb, base_mb, extra_mb=0.0):
    """超预算就抛 MemoryError —— 让上层走降级，而不是硬扛到 OOM。"""
    if not budget_mb or budget_mb <= 0:
        return
    live = self_rss_mb() - float(base_mb) + float(extra_mb)
    if live > float(budget_mb):
        raise MemoryError(
            f"内存预算 guard 触发：任务增量 {live:.1f}MB + 待分配 {extra_mb:.1f}MB "
            f"> 预算 {budget_mb:.0f}MB（基线 {base_mb:.0f}MB）")


# ==================== 3. 负载：pack-build ====================
def _pack_build(payload, budget_mb, base_mb):
    """裁一块 DEM → 构建 SeedPack。裁剪用 mmap，避免每任务重复读 16MB。"""
    from terrain.pack.build import build_seedpack_from_world

    rl = apply_mem_budget(budget_mb)
    dem_path = payload.get("dem") or DEFAULT_DEM
    size = int(payload.get("size") or 512)
    crop = payload.get("crop")
    seed = payload.get("seed")

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="terrain-bench-") as td:
        src = os.path.join(td, "crop.npy")
        mm = np.load(dem_path, mmap_mode="r")
        if crop:
            y0, x0, h, w = (int(v) for v in crop)
            arr = np.array(mm[y0:y0 + h, x0:x0 + w], np.float32)
        else:
            arr = np.array(mm, np.float32)
        del mm
        np.save(src, arr)
        _guard(budget_mb, base_mb, arr.nbytes / MB)      # 预留一次打包的余量
        out = os.path.join(td, f"bench_{size}.seedpack")
        build_seedpack_from_world(src, out, size=size,
                                  seed=None if seed is None else int(seed))
        seconds = time.perf_counter() - t0
        nbytes = os.path.getsize(out) if os.path.exists(out) else 0
    return {"status": "ok", "seconds": seconds, "bytes": int(nbytes),
            "size": size, "rlimit": rl["rlimit"], "crop": payload.get("crop")}


# ==================== 4. 负载：lite-render（惰性） ====================
def _lite_render(payload, budget_mb, base_mb):
    """``terrain.lite`` 由另一条线并行实现：缺失时必须优雅跳过。"""
    try:
        from terrain.lite import LiteEngine, LiteParams, load_pack
    except ImportError as exc:
        return {"status": "ok", "skipped": "terrain.lite 未就绪",
                "seconds": 0.0, "size": int(payload.get("size") or 0),
                "note": str(exc)}

    rl = apply_mem_budget(budget_mb)
    size = int(payload.get("size") or 512)
    t0 = time.perf_counter()
    try:
        pack = load_pack(payload["pack"]) if payload.get("pack") else None
        engine = LiteEngine(pack) if pack is not None else LiteEngine()
        img = engine.render(LiteParams(size=size))
    except (TypeError, AttributeError, RuntimeError) as exc:   # API 尚未定型
        return {"status": "error", "seconds": time.perf_counter() - t0,
                "size": size, "rlimit": rl["rlimit"],
                "error": f"terrain.lite API 不匹配：{type(exc).__name__}: {exc}"}
    return {"status": "ok", "seconds": time.perf_counter() - t0, "size": size,
            "rlimit": rl["rlimit"], "bytes": int(getattr(img, "nbytes", 0) or 0)}


# ==================== 5. 负载：synthetic（稳定基线） ====================
def _fbm(size, octaves, seed):
    """多倍频分形噪声（纯 numpy，kron 上采样 + 简单盒式平滑）。"""
    rs = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    z = np.zeros((size, size), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        n = max(2, 2 ** (o + 1))
        coarse = rs.standard_normal((n, n)).astype(np.float32)
        k = max(1, size // n)
        up = np.kron(coarse, np.ones((k, k), np.float32))
        if up.shape[0] < size:
            pad = size - up.shape[0]
            up = np.pad(up, ((0, pad), (0, pad)), mode="edge")
        up = up[:size, :size]
        for _ in range(2):
            up = (up + np.roll(up, 1, 0) + np.roll(up, -1, 0)
                  + np.roll(up, 1, 1) + np.roll(up, -1, 1)) * 0.2
        z += amp * up
        norm += amp
        amp *= 0.5
    z /= norm
    return z


def _erode_step(z, k=0.35, rate=0.02):
    """一次「侵蚀式」迭代：拉普拉斯热侵蚀 + 轻微回中（模拟抬升—削平）。"""
    lap = (np.roll(z, 1, 0) + np.roll(z, -1, 0)
           + np.roll(z, 1, 1) + np.roll(z, -1, 1)) * 0.25 - z
    z = z + k * lap
    return z - rate * (z - z.mean())


def _synthetic(payload, budget_mb, base_mb):
    """纯 numpy 负载；``alloc_mb`` 按 (size/ref_size)² 缩放，用来制造内存压力。"""
    rl = apply_mem_budget(budget_mb)
    size = int(payload.get("size") or 512)
    iters = int(payload.get("iters") or 24)
    octaves = int(payload.get("octaves") or 6)
    ref = int(payload.get("ref_size") or size) or size
    alloc_mb = float(payload.get("alloc_mb") or 0.0) * (size / float(ref)) ** 2
    sleep_s = float(payload.get("sleep_s") or 0.0)

    t0 = time.perf_counter()
    if alloc_mb > 0:
        _guard(budget_mb, base_mb, alloc_mb)         # 先判预算，再真分配
    z = _fbm(size, octaves, payload.get("seed") or 0)
    for _ in range(iters):
        z = _erode_step(z)

    scratch = None
    if alloc_mb > 0:
        n = max(1, int(alloc_mb * MB / 4))           # float32 元素数
        scratch = np.ones(n, np.float32)             # ones 会真实触页
        z += np.float32(0.0) * scratch[0]            # 保持依赖，避免被优化掉
    if sleep_s > 0:
        time.sleep(sleep_s)                          # 仅用于超时用例

    seconds = time.perf_counter() - t0
    out = {"status": "ok", "seconds": seconds, "size": size,
           "rlimit": rl["rlimit"], "checksum": float(z.mean()),
           "alloc_mb": round(alloc_mb, 1), "iters": iters}
    if scratch is not None:
        out["_keep"] = scratch                       # 活到 run_task 测量之后
    return out


# ==================== 6. 统一入口 ====================
_KINDS = {"pack-build": _pack_build, "lite-render": _lite_render,
          "synthetic": _synthetic}


def run_task(kind, payload):
    """在 worker 子进程里执行一个任务（模块级函数，可被 spawn pickle）。"""
    payload = dict(payload or {})
    budget = float(payload.get("mem_budget_mb") or 0.0)
    base = self_rss_mb()
    t0 = time.perf_counter()
    info, status = {}, "ok"
    try:
        fn = _KINDS.get(kind)
        if fn is None:
            raise ValueError(f"未知负载 kind={kind!r}（可选 {sorted(_KINDS)}）")
        info = fn(payload, budget, base) or {}
        status = info.pop("status", "ok")
    except MemoryError as exc:
        status, info = "oom", {"error": f"MemoryError: {exc}"}
    except Exception as exc:                                  # noqa: BLE001
        status, info = "error", {"error": f"{type(exc).__name__}: {exc}"}

    keep = info.pop("_keep", None)
    seconds = info.pop("seconds", None)
    if seconds is None:
        seconds = time.perf_counter() - t0
    cur, peak = self_rss_mb(), self_peak_rss_mb()

    rec = {"kind": kind, "status": status, "seconds": round(float(seconds), 6),
           "rss_mb": round(max(cur, peak), 1), "base_rss_mb": round(base, 1),
           "task_mb": round(max(0.0, cur - base), 1),
           "size": payload.get("size")}
    rec.update(info)
    if not payload.get("quiet"):
        print(f"[bench] {kind} size={rec.get('size')} {status} "
              f"{rec['seconds']}s rss={rec['rss_mb']}MB(+{rec['task_mb']}MB)",
              file=sys.stderr, flush=True)
    if keep is not None:
        del keep
    return rec
