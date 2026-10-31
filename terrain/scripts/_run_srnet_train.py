"""阶段 5A: SRNet 超分辨率网络正式训练脚本。"""
import sys, os, time, json, warnings
# 允许直接 `python terrain/scripts/_run_srnet_train.py` 运行：把项目根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
warnings.filterwarnings("ignore")
import numpy as np

from terrain.dl.sr_net import train_sr


def main():
    """入口: 用 sr_hr_mosaic_large.npy 数据正式训练 SRNet 超分网络并保存 history。"""
    print("=" * 60, flush=True)
    print("【SRNet 正式训练 v3】epochs=300, scale=4, batch=8, n_patches=2000", flush=True)
    print("  数据: terrain/data/sr_hr_mosaic_large.npy (6144x10240, 60 mosaics)", flush=True)
    print(f"  开始: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    hist = train_sr(
        dem_hr="terrain/data/sr_hr_mosaic_large.npy",
        scale=4,
        epochs=300,
        batch_size=8,
        patch_size=128,
        use_attention=True,
        use_charbonnier=True,
        lambda_freq=0.1,
        lambda_ssim=0.1,
        lambda_grad=0.1,
        lr=2e-4,
        save_path="terrain/checkpoints/sr_net.pt",
        n_patches=2000,
    )
    dt = time.time() - t0

    print("\n" + "=" * 60, flush=True)
    print("SRNet 训练完成", flush=True)
    print(f"  总耗时: {dt:.1f}s ({dt/60:.1f}min)", flush=True)
    print(f"  每ep: {dt/300:.1f}s", flush=True)
    print(f"  参数: {hist['param_count']:,}", flush=True)
    print(f"  total: {hist['total'][0]:.4f} -> {hist['total'][-1]:.4f}", flush=True)
    print(f"  pixel: {hist['pixel'][0]:.4f} -> {hist['pixel'][-1]:.4f}", flush=True)
    print(f"  ssim:  {hist['ssim'][0]:.4f} -> {hist['ssim'][-1]:.4f}", flush=True)
    print(f"  grad:  {hist['grad'][0]:.4f} -> {hist['grad'][-1]:.4f}", flush=True)
    print(f"  freq:  {hist['freq'][0]:.4f} -> {hist['freq'][-1]:.4f}", flush=True)

    with open("terrain/cache/logs/srnet_history.json", "w") as f:
        json.dump({k: (v if isinstance(v, list) else v) for k, v in hist.items()}, f, indent=2)
    print(f"  history saved", flush=True)


if __name__ == "__main__":
    main()
