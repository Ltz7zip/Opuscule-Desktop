# -*- coding: utf-8 -*-
"""运行 DiffusionPipeline（启用 SRNet 超分）。"""
import sys
import os
# 项目根目录（terrain/scripts/ -> terrain/ -> 项目根）
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import json
from terrain.dl.pipeline import DiffusionPipeline, PipelineConfig


def main():
    """入口: 运行 DiffusionPipeline(启用 SRNet 超分) 生成 2048px DEM 并打印各阶段耗时与统计。"""
    print("=" * 60)

    # 配置：seed=0, size=2048, 启用 SR, 暂不启用物理后处理（物理网络未集成）
    cfg = PipelineConfig(
        seed=0,
        size_px=2048,
        use_sr=True,
        use_physics=False,       # 物理网络未集成，跳过
        sr_scale=4,
        save_output=True,
        output_path="terrain/output/pipeline_dem_sr.npy",
        save_info=True,
    )

    pipe = DiffusionPipeline(config=cfg)

    print(f"  seed={cfg.seed}, size={cfg.size_px}px, SR={cfg.use_sr}")
    print(f"  输出: {cfg.output_path}")
    print()

    # 执行生成
    dem, info = pipe.generate(progress=True)

    # 打印结果
    print()
    print("=" * 60)
    print("【生成结果】")
    print(f"  最终尺寸: {info['final_size']}x{info['final_size']}")
    print(f"  高程范围: {info['elev_range'][0]:.0f} ~ {info['elev_range'][1]:.0f} m")
    print(f"  陆地占比: {info['land_ratio']*100:.1f}%")
    print(f"  总耗时: {info['total_time_s']}s")
    print()
    print("【各阶段耗时】")
    for s in info["stages"]:
        name = s.get("name", "?")
        t = s.get("time_s", 0)
        extra = ""
        if "input_size" in s:
            extra = f" ({s['input_size']}² → {s['output_size']}²)"
        elif "output_size" in s:
            extra = f" ({s['output_size']}²)"
        elif "lr_size" in s:
            extra = f" ({s['lr_size']}²)"
        print(f"  {name}: {t}s{extra}")

    if info.get("degradations"):
        print()
        print("【降级信息】")
        for d in info["degradations"]:
            print(f"  ⚠️ {d['stage']}: {d['reason']}")

    # 保存 DEM 信息
    info_path = cfg.output_path.replace(".npy", ".info.json")
    print(f"\nDEM 已保存: {cfg.output_path}")
    print(f"信息已保存: {info_path}")

    # 简单统计
    land = dem > 0
    print(f"\n【DEM 统计】")
    print(f"  陆地面积: {land.sum():,} px ({land.mean()*100:.1f}%)")
    print(f"  最高峰: {dem.max():.0f} m")
    print(f"  最深处: {dem.min():.0f} m")
    print(f"  陆地平均高程: {dem[land].mean():.0f} m")
    print(f"  海洋平均深度: {dem[~land].mean():.0f} m")


if __name__ == "__main__":
    main()
