# -*- coding: utf-8 -*-
# analyze_compare.py —— 对指定种子区间与真实地球做七大维度对比 + 汇总表
# 运行: python terrain/scripts/analyze_compare.py --seeds 11 12 13 --tag v30
# 复用 analyze_worlds.analyze 的统一口径, 便于评估修改前后多种子指标。
import os
import sys
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DIR = os.path.join(_PROJECT_ROOT, "terrain")

from terrain.scripts.analyze_worlds import (load_earth, load_earth_conv,
                                            sim_conv, analyze)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 21)))
    ap.add_argument("--tag", type=str, default="v30")
    ap.add_argument("--dir", type=str, default=None)   # 覆盖 DEM 目录
    a = ap.parse_args()

    datadir = a.dir if a.dir else os.path.join(_DIR, "output")
    results = {}
    print("加载真实地球 ETOPO1 ...")
    earth = load_earth()
    conv_e = load_earth_conv()
    results["earth"] = analyze("earth", earth, conv_e)

    for seed in a.seeds:
        p = os.path.join(datadir, f"seed{seed}_dem.npy")
        if not os.path.exists(p):
            print(f"seed{seed} 缺失, 跳过")
            continue
        dem = np.load(p).astype(np.float32)
        # 降采样到 2048 (与地球参考/分析管线同分辨率, 消除 4096 网格差异)
        if dem.shape[0] > 2048:
            k = dem.shape[0] // 2048
            dem = dem[:2048 * k, :2048 * k].reshape(2048, k, 2048, k).mean((1, 3))
        try:
            conv = sim_conv(seed)
        except Exception:
            conv = None
        results[f"seed{seed}"] = analyze(f"seed{seed}", dem, conv)
        print(f"seed{seed} 完成")

    out = os.path.join(_DIR, "output", f"analysis_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    keys = [k for k in results["earth"] if k != "name"]
    print(f"\n{'指标':<28}{'真实地球':>14}{'生成均值':>16}{'±std':>12}  评价")
    for k in keys:
        e = results["earth"][k]
        vs = [results[s][k] for s in results if s != "earth" and k in results[s]]
        if not vs:
            continue
        m, sd = float(np.mean(vs)), float(np.std(vs))
        rel = abs(m - e) / (abs(e) + 1e-9)
        flag = "≈" if rel < 0.25 else ("偏差" if rel < 0.7 else "显著偏差")
        print(f"{k:<30}{e:>14.4f}{m:>16.4f}±{sd:<11.4f} {flag}")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()