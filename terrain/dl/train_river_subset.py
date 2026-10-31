"""河流 Diffusion 模型训练（DataLoader 方式）。

使用 DataLoader + num_workers=0（已验证可工作），
支持断点续训和中间 checkpoint 保存。
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import (
    DEVICE, CKPT_DIR, LEARNING_RATE, WEIGHT_DECAY,
    DIFFUSION_STEPS, BETA_START, BETA_END,
)
from .river_diffusion import (
    ConditionalRiverDiffusion, RiverDiffusionDataset, river_diffusion_loss,
)
from .diffusion import DDPM
from .train import EMA, cosine_warmup_lr, amp_enabled


def train_river_subset(
    n_samples: int = 10000,
    epochs: int = 10,
    batch_size: int = 8,
    target: str = "mask",
    lambda_braid: float = 0.1,
    lambda_conn: float = 0.05,
    resume: bool = False,
    warmup_epochs: int = 2,
):
    """DataLoader 方式训练，支持断点续训。"""
    print(f"Device: {DEVICE}")
    print(f"训练: {n_samples} 样本, {epochs} epochs, batch={batch_size}, resume={resume}")

    # ---- 数据 ----
    ds = RiverDiffusionDataset(mode="river_dataset", target=target, augment=True)
    indices = np.arange(min(n_samples, len(ds)))
    ds_sub = Subset(ds, indices)
    loader = DataLoader(ds_sub, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    # ---- 模型 ----
    model = ConditionalRiverDiffusion(cond_ch=2, base=64, time_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    ema = EMA(model)
    ddpm = DDPM(n_steps=DIFFUSION_STEPS, beta_start=BETA_START,
                beta_end=BETA_END, device=DEVICE)

    start_epoch = 0
    prev_total_steps = 0
    ckpt_path = CKPT_DIR / "river_diffusion.pt"

    if resume and ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.shadow = ck["ema"]
        if "epoch" in ck:
            start_epoch = ck["epoch"]
        if "step" in ck:
            prev_total_steps = ck["step"]
        print(f"  续训: 从 epoch {start_epoch} 开始 (prev_steps={prev_total_steps})")

    n_batches = len(loader)
    n_steps_total = epochs * n_batches
    warmup_steps = warmup_epochs * n_batches
    scaler = torch.amp.GradScaler(DEVICE.type) if amp_enabled() else None

    print(f"  {n_batches} 批/epoch, {n_steps_total} 总步数")
    print(f"  amp={amp_enabled()}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        nb = 0
        loss_parts_acc = {"diff": 0, "braid": 0, "conn": 0}

        for cond, target_batch in loader:
            cond, target_batch = cond.to(DEVICE), target_batch.to(DEVICE)

            lr = cosine_warmup_lr(
                epoch * n_batches + nb, n_steps_total,
                warmup_steps, LEARNING_RATE)
            for g in optimizer.param_groups:
                g["lr"] = lr

            t = torch.randint(0, DIFFUSION_STEPS, (target_batch.size(0),), device=DEVICE)
            optimizer.zero_grad()

            if amp_enabled():
                with torch.amp.autocast(DEVICE.type):
                    loss, parts = river_diffusion_loss(
                        model, target_batch, t, cond, ddpm,
                        lambda_braid=lambda_braid, lambda_conn=lambda_conn)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, parts = river_diffusion_loss(
                    model, target_batch, t, cond, ddpm,
                    lambda_braid=lambda_braid, lambda_conn=lambda_conn)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ema.update(model)
            epoch_loss += parts["total"]
            for k in loss_parts_acc:
                if k in parts:
                    loss_parts_acc[k] += parts[k]
            nb += 1

            # 每 100 批打印进度
            if nb % 100 == 0:
                print(f"  ep{start_epoch+epoch:03d} batch {nb}/{n_batches} "
                      f"loss={epoch_loss/nb:.4f}", flush=True)

        dt = time.time() - t0
        avg = epoch_loss / max(1, nb)
        parts_avg = {k: v / max(1, nb) for k, v in loss_parts_acc.items()}
        ep_label = start_epoch + epoch
        print(f"ep{ep_label:03d} loss={avg:.4f} "
              f"diff={parts_avg['diff']:.4f} "
              f"braid={parts_avg.get('braid', 0):.4f} "
              f"conn={parts_avg.get('conn', 0):.4f} "
              f"lr={lr:.2e} {dt:.1f}s", flush=True)

        # 每 2 epoch 保存 checkpoint
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            torch.save({
                "model": model.state_dict(),
                "ema": ema.shadow,
                "epoch": start_epoch + epoch + 1,
                "step": prev_total_steps + (epoch + 1) * n_batches,
            }, ckpt_path)
            print(f"  checkpoint saved: epoch {start_epoch + epoch + 1}", flush=True)

    print(f"\n模型保存: {ckpt_path} (epoch={start_epoch + epochs})")
    return model, ema


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="河流 Diffusion 训练")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--target", choices=["mask", "smooth"], default="mask")
    p.add_argument("--lambda-braid", type=float, default=0.1)
    p.add_argument("--lambda-conn", type=float, default=0.05)
    args = p.parse_args()

    train_river_subset(
        n_samples=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target=args.target,
        lambda_braid=args.lambda_braid,
        lambda_conn=args.lambda_conn,
        resume=args.resume,
    )
