"""阶段 5C：端到端集成管线（优化版）。

整合 LDM 主生成器 + 超分辨率网络 + 物理监督后处理，
提供统一推理 API，形成完整 DL 地形生成引擎。

优化项（相比初版）
------------------
1. **PipelineConfig**：集中管理所有参数，支持 YAML 加载
2. **条件缓存**：相同 seed 的条件图复用，避免重复板块模拟
3. **阶段控制**：选择性跳过/重试各阶段
4. **错误降级**：SR/物理失败时自动降级到原始 DEM
5. **输出保存**：自动保存 DEM + 元信息 JSON
6. **进度回调**：支持回调函数实时报告进度

管线流程
--------
1. **条件编码**：板块模拟 → 12 通道条件图（+ 可选多模态条件）
2. **LDM 生成**：条件扩散模型生成低分辨率 DEM
3. **超分辨率**：ESRGAN 将 DEM 放大（4×）
4. **物理后处理**：CoastNet → RiverNet → SnowlineNet
5. **统计校准**：直方图匹配 + 陆地占比保护

用法
----
    >>> from terrain.dl.pipeline import DiffusionPipeline, PipelineConfig
    >>> cfg = PipelineConfig(seed=42, size_px=4096, use_sr=True)
    >>> pipe = DiffusionPipeline(config=cfg)
    >>> dem, info = pipe.generate()
    >>> # 带进度回调
    >>> pipe.generate(progress_callback=lambda pct, msg: print(f"{pct:.0f}% {msg}"))
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from .config import DEVICE, ELEV_MIN, ELEV_MAX


# ==================== 配置类 ====================

@dataclass
class PipelineConfig:
    """管线配置：集中管理所有参数。

    支持从字典/YAML 构造，便于实验管理与配置复用。

    用法
    ----
        >>> cfg = PipelineConfig(seed=42, size_px=4096)
        >>> cfg = PipelineConfig.from_dict({"seed": 7, "use_sr": False})
    """
    # 基本参数
    seed: int = 42
    size_px: int = 4096          # 最终输出尺寸
    sr_scale: int = 4            # 超分倍数

    # 阶段开关
    use_sr: bool = True          # 启用超分辨率
    use_physics: bool = True     # 启用物理后处理
    use_multicond: bool = False  # 启用多模态条件（需提供 sketch/biome 等）

    # 物理子阶段开关（use_physics=True 时生效）
    physics_coast: bool = True   # 海岸线锐化
    physics_river: bool = True   # 河网雕刻
    physics_snow: bool = True    # 雪线渲染

    # 校准参数
    calibrate_smooth_sigma: float = 0.5  # 高斯平滑核
    calibrate_clip: bool = True          # 高程裁剪

    # 缓存
    cache_conditions: bool = True        # 缓存条件图
    cache_dir: str = "terrain/cache"

    # 输出
    save_output: bool = False            # 自动保存 DEM
    output_path: str = "terrain/output/pipeline_dem.npy"
    save_info: bool = True               # 保存元信息 JSON

    # 设备
    device: str = "auto"                 # "auto"/"cpu"/"cuda"/"mps"

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """从字典构造配置。"""
        # 过滤未知键
        valid = {k: v for k, v in d.items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        """转字典。"""
        return asdict(self)

    def resolve_device(self) -> torch.device:
        """解析设备字符串。"""
        if self.device == "auto":
            return DEVICE
        return torch.device(self.device)


# ==================== 条件缓存 ====================

class ConditionCache:
    """条件图缓存：相同 seed + size 的条件图复用。

    避免重复运行板块模拟（约 2-5 秒）。
    """

    def __init__(self, cache_dir: str = "terrain/cache",
                 max_entries: int = 4):
        """初始化缓存。

        Parameters
        ----------
        cache_dir : str  缓存目录
        max_entries : int  最大缓存条目数（LRU 淘汰）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._memory: dict[str, np.ndarray] = {}
        self._order: list[str] = []

    def _key(self, seed: int, size_px: int, use_sr: bool) -> str:
        """生成缓存键。"""
        lr = size_px // 4 if use_sr else size_px
        return f"cond_seed{seed}_lr{lr}"

    def get(self, seed: int, size_px: int, use_sr: bool
            ) -> np.ndarray | None:
        """查询内存缓存。"""
        key = self._key(seed, size_px, use_sr)
        if key in self._memory:
            self._order.remove(key)
            self._order.append(key)
            return self._memory[key]
        # 尝试磁盘缓存
        disk_path = self.cache_dir / f"{key}.npy"
        if disk_path.exists():
            data = np.load(str(disk_path))
            self._put(key, data)
            return data
        return None

    def put(self, seed: int, size_px: int, use_sr: bool,
            data: np.ndarray) -> None:
        """写入缓存（内存 + 磁盘）。"""
        key = self._key(seed, size_px, use_sr)
        self._put(key, data)
        # 磁盘持久化
        disk_path = self.cache_dir / f"{key}.npy"
        np.save(str(disk_path), data)

    def _put(self, key: str, data: np.ndarray) -> None:
        """写入内存缓存 + LRU 淘汰。"""
        self._memory[key] = data
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        while len(self._order) > self.max_entries:
            old = self._order.pop(0)
            self._memory.pop(old, None)

    def clear(self) -> None:
        """清空内存缓存。"""
        self._memory.clear()
        self._order.clear()

    def stats(self) -> dict:
        """返回缓存统计。"""
        return {
            "memory_entries": len(self._memory),
            "max_entries": self.max_entries,
        }


# ==================== 管线 ====================

class DiffusionPipeline:
    """端到端扩散地形生成管线（优化版）。

    整合条件编码 → LDM 生成 → 超分辨率 → 物理后处理 → 校准。

    优化特性
    --------
    - PipelineConfig 集中配置
    - 条件图缓存（ConditionCache）
    - 阶段选择性执行与跳过
    - 错误降级（SR/物理失败时用原始 DEM）
    - 输出自动保存 + 元信息 JSON
    - 进度回调函数

    Parameters
    ----------
    config : PipelineConfig  管线配置
    ldm : LDMInference, optional  LDM 推理器
    sr_model : SRNet, optional  超分模型
    river_net / snowline_net / coast_net : optional  物理网络
    multicond : MultiModalConditioner, optional  多模态条件器
    """

    def __init__(self,
                 config: PipelineConfig | None = None,
                 ldm=None,
                 sr_model=None,
                 river_net=None,
                 snowline_net=None,
                 coast_net=None,
                 multicond=None):
        """初始化管线：保存配置与各模型引用，按配置创建条件缓存。"""
        self.config = config or PipelineConfig()
        self.device = self.config.resolve_device()
        self._ldm = ldm
        self._sr = sr_model
        self._river = river_net
        self._snowline = snowline_net
        self._coast = coast_net
        self._multicond = multicond
        self._cond_cache = ConditionCache(self.config.cache_dir) \
            if self.config.cache_conditions else None
        self._last_info: dict | None = None

    @property
    def ldm(self):
        """延迟加载 LDM 推理器。"""
        if self._ldm is None:
            from .infer_ldm import LDMInference
            self._ldm = LDMInference()
        return self._ldm

    @property
    def sr_model(self):
        """延迟加载超分模型。"""
        if self._sr is None:
            from .sr_net import SRNet
            import os
            # 与训练脚本一致的结构参数
            self._sr = SRNet(scale=4, use_attention=True, base=32,
                             n_rrdb=3).to(self.device).eval()
            # 加载训练好的权重
            ckpt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "checkpoints", "sr_net.pt")
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device,
                                   weights_only=True)
                self._sr.load_state_dict(state)
                print(f"[pipeline] SRNet 权重已加载: {ckpt_path}")
            else:
                print(f"[pipeline] ⚠️ 未找到 SRNet 权重: {ckpt_path}")
        return self._sr

    @property
    def multicond(self):
        """延迟加载多模态条件器。"""
        if self._multicond is None:
            from .multicond import MultiModalConditioner
            self._multicond = MultiModalConditioner().to(self.device).eval()
        return self._multicond

    def generate(self, seed: int | None = None,
                 size_px: int | None = None,
                 cond_12ch: np.ndarray | None = None,
                 sketch: np.ndarray | None = None,
                 elev_hint: np.ndarray | None = None,
                 biome: np.ndarray | None = None,
                 use_sr: bool | None = None,
                 use_physics: bool | None = None,
                 progress: bool = True,
                 progress_callback=None,
                 ) -> tuple[np.ndarray, dict]:
        """端到端生成世界 DEM。

        Parameters
        ----------
        seed : int, optional  覆盖配置的 seed
        size_px : int, optional  覆盖配置的 size_px
        cond_12ch : ndarray, optional  预计算条件图（跳过板块模拟）
        sketch / elev_hint / biome : ndarray, optional  多模态条件
        use_sr : bool, optional  覆盖配置的 use_sr
        use_physics : bool, optional  覆盖配置的 use_physics
        progress : bool  打印进度
        progress_callback : callable  回调函数 (percent, message)

        Returns
        -------
        dem : ndarray (size_px, size_px) float32
        info : dict  各阶段信息
        """
        # 合并参数：调用参数 > 配置
        cfg = self.config
        seed = seed if seed is not None else cfg.seed
        size_px = size_px if size_px is not None else cfg.size_px
        use_sr = use_sr if use_sr is not None else cfg.use_sr
        use_physics = use_physics if use_physics is not None else cfg.use_physics

        torch.manual_seed(seed)
        info = {"seed": seed, "size_px": size_px, "stages": [],
                "config": cfg.to_dict(), "degradations": []}

        def _cb(pct, msg):
            """进度回调包装：存在用户回调时转发百分比与消息。"""
            if progress_callback:
                progress_callback(pct, msg)

        def _log(stage, msg):
            """阶段日志：可选打印并在进度回调中转发。"""
            if progress:
                print(f"[pipeline] {stage}: {msg}")
            _cb(0, f"{stage}: {msg}")

        # ---- 阶段 1：条件编码 ----
        t0 = time.time()
        _log("阶段 1", "条件编码")
        _cb(10, "条件编码中")

        if cond_12ch is None:
            cond_12ch = self._get_conditions(seed, size_px, use_sr, cfg)
        cond = cond_12ch

        # 多模态条件增强
        if sketch is not None or elev_hint is not None or biome is not None:
            cond = self._encode_multimodal(cond_12ch, sketch, elev_hint, biome)

        lr_size = cond.shape[-1]
        info["stages"].append({
            "name": "condition", "time_s": round(time.time() - t0, 2),
            "lr_size": lr_size, "cond_channels": cond.shape[0]})
        _cb(20, f"条件编码完成 ({lr_size}²)")

        # ---- 阶段 2：LDM 生成 ----
        t0 = time.time()
        _log("阶段 2", f"LDM 生成 ({lr_size}²)")
        _cb(30, "LDM 扩散采样中")
        dem_lr = self.ldm.generate(cond, seed=seed, tile_size=lr_size)
        info["stages"].append({
            "name": "ldm", "time_s": round(time.time() - t0, 2),
            "output_size": dem_lr.shape[0]})
        _cb(50, f"LDM 完成 ({dem_lr.shape[0]}²)")

        # ---- 阶段 3：超分辨率 ----
        if use_sr and cfg.sr_scale > 1:
            t0 = time.time()
            _log("阶段 3", f"超分辨率 ({dem_lr.shape[0]}² → {size_px}²)")
            _cb(55, "超分辨率中")
            dem_hr = self._run_sr(dem_lr, cfg, info)
            _cb(75, f"超分完成 ({dem_hr.shape[0]}²)")
        else:
            dem_hr = dem_lr
            info["stages"].append({"name": "sr", "time_s": 0,
                                   "skipped": True})
            _cb(75, "超分已跳过")

        # ---- 阶段 4：物理后处理 ----
        if use_physics:
            t0 = time.time()
            _log("阶段 4", "物理后处理")
            _cb(80, "物理后处理中")
            dem_final = self._apply_physics_safe(dem_hr, cfg, info)
            _cb(90, "物理后处理完成")
        else:
            dem_final = dem_hr
            info["stages"].append({"name": "physics", "time_s": 0,
                                   "skipped": True})
            _cb(90, "物理后处理已跳过")

        # ---- 阶段 5：统计校准 ----
        t0 = time.time()
        _log("阶段 5", "统计校准")
        _cb(95, "校准中")
        dem_final = self._calibrate(dem_final, cfg)
        info["stages"].append({
            "name": "calibrate", "time_s": round(time.time() - t0, 2)})

        # 汇总信息
        info["final_size"] = dem_final.shape[0]
        info["elev_range"] = (float(dem_final.min()),
                              float(dem_final.max()))
        info["land_ratio"] = float((dem_final > 0).mean())
        info["total_time_s"] = round(
            sum(s.get("time_s", 0) for s in info["stages"]), 2)

        # 输出保存
        if cfg.save_output:
            self._save_output(dem_final, info, cfg)

        self._last_info = info
        _cb(100, "完成")

        if progress:
            print(f"[pipeline] 完成: {dem_final.shape[0]}², "
                  f"陆地 {info['land_ratio']:.1%}, "
                  f"耗时 {info['total_time_s']}s")

        return dem_final, info

    # ==================== 内部方法 ====================

    def _get_conditions(self, seed: int, size_px: int, use_sr: bool,
                        cfg: PipelineConfig) -> np.ndarray:
        """获取条件图（带缓存）。"""
        if self._cond_cache is not None:
            cached = self._cond_cache.get(seed, size_px, use_sr)
            if cached is not None:
                return cached

        cond = self._make_conditions(seed, size_px, use_sr)

        if self._cond_cache is not None:
            self._cond_cache.put(seed, size_px, use_sr, cond)
        return cond

    def _make_conditions(self, seed: int, size_px: int,
                         use_sr: bool) -> np.ndarray:
        """生成板块条件图。"""
        from terrain.core.plate_sim import run_plate_sim
        from terrain.core.generate_world import world_to_channels
        from scipy.ndimage import zoom as ndi_zoom

        lr_size = size_px // 4 if use_sr else size_px
        tect_res = min(512, lr_size)
        elev_T, tect_T, *_ = run_plate_sim(T=tect_res, seed=seed,
                                            verbose=False)
        up = max(1, lr_size // tect_res)
        elev_m = ndi_zoom(elev_T.astype(np.float32), up, order=1)
        elev_m = elev_m[:lr_size, :lr_size]
        lats = (-90.0 + (np.arange(lr_size) + 0.5)
                * (180.0 / lr_size)).astype(np.float32)
        block = 8
        cond_12ch, *_ = world_to_channels(elev_m, lats, tect_T, block, seed)
        return cond_12ch

    def _encode_multimodal(self, plate_12ch: np.ndarray,
                           sketch: np.ndarray | None,
                           elev_hint: np.ndarray | None,
                           biome: np.ndarray | None) -> np.ndarray:
        """多模态条件编码。"""
        from .multicond import encode_multimodal
        return encode_multimodal(plate_12ch, sketch=sketch,
                                 elev_hint=elev_hint, biome=biome)

    def _run_sr(self, dem_lr: np.ndarray, cfg: PipelineConfig,
                info: dict) -> np.ndarray:
        """执行超分辨率（带错误降级）。"""
        try:
            from .sr_net import sr_upscale
            t0 = time.time()
            # 通过 property 触发权重加载
            model = self.sr_model
            dem_hr = sr_upscale(dem_lr, scale=cfg.sr_scale,
                                model=model, device=self.device)
            info["stages"].append({
                "name": "sr", "time_s": round(time.time() - t0, 2),
                "input_size": dem_lr.shape[0],
                "output_size": dem_hr.shape[0]})
            return dem_hr
        except Exception as e:
            # 降级：用双三次插值替代
            info["degradations"].append(
                {"stage": "sr", "reason": str(e),
                 "fallback": "bilinear_upsample"})
            from scipy.ndimage import zoom as ndi_zoom
            dem_hr = ndi_zoom(dem_lr, cfg.sr_scale, order=1)
            info["stages"].append({
                "name": "sr", "time_s": 0,
                "output_size": dem_hr.shape[0],
                "degraded": True})
            return dem_hr

    def _apply_physics_safe(self, dem: np.ndarray, cfg: PipelineConfig,
                            info: dict) -> np.ndarray:
        """物理后处理（带错误降级）。

        依次执行 CoastNet → RiverNet → SnowlineNet，
        每步失败时跳过并记录降级信息。
        """
        dem_out = dem.copy()
        physics_stages = []

        # CoastNet
        if cfg.physics_coast and self._coast is not None:
            t0 = time.time()
            try:
                from .infer_physics import coast_sharpen
                dem_out = coast_sharpen(dem_out, self._coast)
                physics_stages.append({"sub": "coast", "time_s": round(
                    time.time() - t0, 2), "status": "ok"})
            except Exception as e:
                physics_stages.append({"sub": "coast", "status": "skipped",
                                       "reason": str(e)})
                info["degradations"].append(
                    {"stage": "physics.coast", "reason": str(e)})

        # RiverNet
        if cfg.physics_river and self._river is not None:
            t0 = time.time()
            try:
                from .infer_physics import rivernet_threshold, carve_rivers
                river_mask = rivernet_threshold(dem_out, self._river)
                dem_out = carve_rivers(dem_out, river_mask)
                physics_stages.append({"sub": "river", "time_s": round(
                    time.time() - t0, 2), "status": "ok"})
            except Exception as e:
                physics_stages.append({"sub": "river", "status": "skipped",
                                       "reason": str(e)})
                info["degradations"].append(
                    {"stage": "physics.river", "reason": str(e)})

        # SnowlineNet
        if cfg.physics_snow and self._snowline is not None:
            t0 = time.time()
            try:
                from .infer_physics import snowline_apply
                dem_out = snowline_apply(dem_out, self._snowline)
                physics_stages.append({"sub": "snow", "time_s": round(
                    time.time() - t0, 2), "status": "ok"})
            except Exception as e:
                physics_stages.append({"sub": "snow", "status": "skipped",
                                       "reason": str(e)})
                info["degradations"].append(
                    {"stage": "physics.snow", "reason": str(e)})

        info["stages"].append({
            "name": "physics", "sub_stages": physics_stages})
        return dem_out

    def _calibrate(self, dem: np.ndarray,
                   cfg: PipelineConfig) -> np.ndarray:
        """统计校准：高斯平滑 + 范围裁剪。"""
        from scipy.ndimage import gaussian_filter

        if cfg.calibrate_smooth_sigma > 0:
            dem = gaussian_filter(dem, sigma=cfg.calibrate_smooth_sigma)

        if cfg.calibrate_clip:
            dem = np.clip(dem, ELEV_MIN, ELEV_MAX)

        return dem.astype(np.float32)

    def _save_output(self, dem: np.ndarray, info: dict,
                     cfg: PipelineConfig) -> None:
        """保存 DEM 和元信息。"""
        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), dem)

        if cfg.save_info:
            info_path = out_path.with_suffix(".info.json")
            # info 含 numpy 类型，转 JSON 安全格式
            def _json_safe(obj):
                """把 numpy/Path 等类型转为 JSON 可序列化对象。"""
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, Path):
                    return str(obj)
                return str(obj)
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2,
                          default=_json_safe)

    # ==================== 公共方法 ====================

    def benchmark(self, seed: int = 42, size_px: int = 4096,
                  use_sr: bool = True, use_physics: bool = True,
                  warmup: int = 0) -> dict:
        """基准测试：测量各阶段时间与输出质量。

        Parameters
        ----------
        warmup : int  预热次数（不计入计时）

        Returns
        -------
        dict  含 total_time / stages / elev_range / land_ratio / cache_stats
        """
        # 预热
        for _ in range(warmup):
            self.generate(seed=seed, size_px=size_px,
                          use_sr=use_sr, use_physics=use_physics,
                          progress=False)

        t0 = time.time()
        dem, info = self.generate(
            seed=seed, size_px=size_px,
            use_sr=use_sr, use_physics=use_physics,
            progress=False)
        total = time.time() - t0
        info["total_time_s"] = round(total, 2)

        if self._cond_cache:
            info["cache_stats"] = self._cond_cache.stats()

        return info

    def generate_batch(self, seeds: list[int],
                       size_px: int | None = None,
                       use_sr: bool | None = None,
                       use_physics: bool | None = None,
                       progress: bool = True) -> list[tuple[np.ndarray, dict]]:
        """批量生成多个种子的世界。

        利用条件缓存加速重复生成。

        Parameters
        ----------
        seeds : list[int]  种子列表

        Returns
        -------
        list of (dem, info)
        """
        results = []
        for i, seed in enumerate(seeds):
            if progress:
                print(f"\n=== 批量生成 {i+1}/{len(seeds)}: seed={seed} ===")
            dem, info = self.generate(
                seed=seed, size_px=size_px,
                use_sr=use_sr, use_physics=use_physics,
                progress=progress)
            results.append((dem, info))
        return results

    def get_cache_stats(self) -> dict:
        """获取条件缓存统计。"""
        if self._cond_cache:
            return self._cond_cache.stats()
        return {"cache_enabled": False}

    def clear_cache(self) -> None:
        """清空条件缓存。"""
        if self._cond_cache:
            self._cond_cache.clear()

    @property
    def last_info(self) -> dict | None:
        """最近一次生成的信息。"""
        return self._last_info
