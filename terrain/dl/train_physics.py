"""阶段 3 物理监督网络训练入口。

为 RiverNet / SnowlineNet / CoastNet 三个轻量网络提供统一训练循环：

- EMA 权重滑动平均（采样用 EMA，更稳定）
- warmup + 余弦学习率调度
- 混合精度（CUDA/MPS 启用，CPU 退化 fp32）
- 梯度累积（小 batch 模拟大 batch）
- 梯度裁剪 + 断点续训
- 周期验证集完整评估（IoU/Dice/F1/RMSE/R²/mIoU/连通性/拓扑）
- 早停（patience 轮无提升则停止）
- CSV 训练日志

用法
----
    python -m terrain.dl.train_physics --task river  --epochs 50
    python -m terrain.dl.train_physics --task snowline --epochs 30
    python -m terrain.dl.train_physics --task coast  --epochs 50 --resume
    python -m terrain.dl.train_physics --task river --grad-accum 4 --early-stop 10
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    DEVICE, CKPT_DIR, LOG_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, GRAD_CLIP, EMA_DECAY, USE_AMP, TILE_SIZE,
)
from .logger import TrainLogger
from .physics_nets import (
    RiverNet, SnowlineNet, CoastNet,
    river_loss, snowline_loss, coast_loss,
)
from .physics_dataset import make_physics_loaders
from .physics_metrics import (
    MetricAccumulator, format_metrics,
    river_metrics, snowline_metrics, coast_metrics,
)


# ==================== EMA ====================

class EMA:
    """模型参数指数滑动平均（与 train.py 中一致，独立保留以避免循环依赖）。"""

    def __init__(self, model, decay: float = EMA_DECAY):
        """初始化 EMA：克隆模型可训练参数作为影子权重。"""
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model):
        """用当前模型参数按指数衰减更新影子权重。"""
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        """把模型参数临时替换为 EMA 影子权重（备份原参数）。"""
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        """恢复 apply 前备份的原始模型参数。"""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup.clear()


# ==================== 学习率调度 ====================

def cosine_warmup_lr(step: int, max_steps: int, warmup: int,
                     base_lr: float) -> float:
    """warmup + 余弦衰减。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def amp_enabled() -> bool:
    """AMP 仅在 CUDA/MPS 启用；CPU 无效。"""
    return USE_AMP and DEVICE.type in ("cuda", "mps")


# ==================== 任务配置 ====================

TASK_CONFIG = {
    "river": {
        "model_cls": RiverNet,
        "loss_fn": river_loss,
        "ckpt_name": "rivernet.pt",
        "in_ch": 2,
    },
    "snowline": {
        "model_cls": SnowlineNet,
        "loss_fn": snowline_loss,
        "ckpt_name": "snowlinenet.pt",
        "in_ch": 3,
    },
    "coast": {
        "model_cls": CoastNet,
        "loss_fn": coast_loss,
        "ckpt_name": "coastnet.pt",
        "in_ch": 1,
    },
}


# ==================== 前向 + 损失封装 ====================

def forward_loss(task: str, model: torch.nn.Module,
                 batch) -> tuple[torch.Tensor, dict]:
    """根据任务调用对应前向 + 损失。

    Returns
    -------
    (loss, info_dict)
    """
    if task == "river":
        x, y = batch   # x: (B,1,H,W) [-1,1], y: (B,1,H,W) {0,1}
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        return river_loss(pred, y)

    if task == "snowline":
        x, y = batch   # x: (B,3,H,W), y: (B,1,H,W) 米
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # SnowlineNet 接受三个独立参数
        lat_norm = x[:, 0:1]
        elev_norm = x[:, 1:2]
        ci = x[:, 2:3]
        pred = model(lat_norm, elev_norm, ci)
        return snowline_loss(pred, y)

    if task == "coast":
        x, y = batch   # x: (B,1,H,W) [-1,1], y: (B,H,W) long
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        return coast_loss(logits, y)

    raise ValueError(f"未知任务: {task}")


# ==================== 验证评估（完整指标） ====================

@torch.no_grad()
def validate(task: str, model, ema: EMA, val_loader: DataLoader) -> dict:
    """在验证集上用 EMA 权重计算完整指标（在整个验证集累积）。

    Returns
    -------
    dict  含 "loss" 及任务特定完整指标（见 physics_metrics.py）。
    """
    model.eval()
    ema.apply(model)
    try:
        total_loss = 0.0
        n = 0
        acc = MetricAccumulator(task)

        for batch in val_loader:
            loss, info = forward_loss(task, model, batch)
            total_loss += float(loss.item())
            n += 1

            # 累积预测与标签
            if task == "river":
                x, y = batch
                pred = model(x.to(DEVICE))
                acc.update(pred.squeeze(1).cpu().numpy(),
                           y.squeeze(1).cpu().numpy())
            elif task == "snowline":
                x, y = batch
                lat, el, ci = x[:, 0:1], x[:, 1:2], x[:, 2:3]
                pred = model(lat.to(DEVICE), el.to(DEVICE), ci.to(DEVICE))
                acc.update(pred.squeeze(1).cpu().numpy(),
                           y.squeeze(1).cpu().numpy())
            elif task == "coast":
                x, y = batch
                logits = model(x.to(DEVICE))
                pred = logits.argmax(dim=1)
                acc.update(pred.cpu().numpy(),
                           y.cpu().numpy())
    finally:
        ema.restore(model)

    result = {"loss": total_loss / max(1, n)}
    metrics = acc.compute()
    result.update(metrics)
    return result


# ==================== 训练循环 ====================

def train_physics(task: str, epochs: int = 50, batch_size: int = BATCH_SIZE,
                  lr: float = LEARNING_RATE, resume: bool = False,
                  num_workers: int = 2, val_ratio: float = 0.1,
                  eval_interval: int = 5, data_root: str | None = None,
                  grad_accum: int = 1, early_stop: int = 0) -> None:
    """统一训练入口。

    Parameters
    ----------
    task : str  ∈ {"river", "snowline", "coast"}
    epochs : int
    batch_size : int
    lr : float
    resume : bool  从 last.pt 续训
    num_workers : int  DataLoader 工作进程数
    val_ratio : float  验证集占比
    eval_interval : int  每多少 epoch 跑验证
    data_root : str, optional  数据目录路径；None 用默认 DEM_TILES_DIR
    grad_accum : int  梯度累积步数（有效 batch = batch_size × grad_accum）
    early_stop : int  早停耐心（轮）；0 表示不启用早停
    """
    task = task.lower()
    if task not in TASK_CONFIG:
        raise ValueError(f"未知任务: {task}，可选: {list(TASK_CONFIG)}")

    cfg = TASK_CONFIG[task]
    model = cfg["model_cls"]().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{task}] 模型参数数: {n_params/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    ema = EMA(model)

    # 数据
    load_kwargs = {"batch_size": batch_size, "val_ratio": val_ratio,
                   "num_workers": num_workers}
    if data_root is not None:
        load_kwargs["root"] = data_root
    train_loader, val_loader = make_physics_loaders(task, **load_kwargs)
    steps_per_epoch = math.ceil(len(train_loader) / max(grad_accum, 1))
    n_steps_total = epochs * steps_per_epoch

    # 日志
    run_name = f"physics_{task}"
    logger = TrainLogger(run_name, LOG_DIR)
    ckpt_path = CKPT_DIR / cfg["ckpt_name"]

    # 续训
    start_epoch = 0
    global_step = 0
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            ema.shadow = ckpt["ema"]
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        print(f"[resume] {task} 从 epoch {start_epoch} 续训 (step={global_step})")

    scaler = torch.amp.GradScaler(DEVICE.type) if amp_enabled() else None
    print(f"[{task}] device={DEVICE}, amp={amp_enabled()}, "
          f"epochs={epochs}, batch={batch_size}, "
          f"grad_accum={grad_accum} (有效batch={batch_size*grad_accum}), "
          f"early_stop={early_stop or 'off'}")

    best_score = -float("inf")
    no_improve = 0   # 早停计数器

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        accum_count = 0   # 梯度累积计数

        for batch in train_loader:
            lr_now = cosine_warmup_lr(global_step, n_steps_total,
                                      WARMUP_STEPS, lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            # ---- 梯度累积 ----
            if amp_enabled():
                with torch.amp.autocast(DEVICE.type):
                    loss, info = forward_loss(task, model, batch)
                # 归一化损失以匹配累积
                scaler.scale(loss / grad_accum).backward()
            else:
                loss, info = forward_loss(task, model, batch)
                (loss / grad_accum).backward()

            accum_count += 1
            epoch_loss += info.get("loss", float(loss.item()))
            n_batches += 1

            # 累积到 grad_accum 步才更新
            if accum_count >= grad_accum:
                if amp_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
                accum_count = 0
                global_step += 1

                if global_step % 50 == 0:
                    logger.log_scalars({
                        "loss": float(loss.item()),
                        "lr": lr_now,
                        **{k: v for k, v in info.items() if k != "loss"},
                    }, step=global_step)

        # 处理尾部未满 grad_accum 的剩余梯度
        if accum_count > 0:
            if amp_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad()
            ema.update(model)
            global_step += 1

        avg_loss = epoch_loss / max(1, n_batches)
        dt = time.time() - t0
        print(f"[{task}] ep{epoch:03d} avg_loss={avg_loss:.4f} "
              f"lr={lr_now:.2e} {dt:.1f}s")

        # 保存 last
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.shadow,
            "epoch": epoch,
            "step": global_step,
            "task": task,
            "base": 16,
        }, ckpt_path)

        # ---- 周期验证 ----
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            val_result = validate(task, model, ema, val_loader)
            # 详细打印
            print(f"  [val] loss={val_result['loss']:.4f}  "
                  f"{format_metrics(task, val_result)}")
            # 记录到日志
            for k, v in val_result.items():
                if k == "confusion_matrix":
                    continue
                logger.log_scalars({f"val_{k}": v}, step=global_step)

            # 保存最佳（loss 越小越好）
            score = -val_result["loss"]
            improved = score > best_score
            if improved:
                best_score = score
                no_improve = 0
                best_path = ckpt_path.parent / f"best_{cfg['ckpt_name']}"
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "epoch": epoch,
                    "step": global_step,
                    "task": task,
                    "base": 16,
                    "val_loss": val_result["loss"],
                    "val_metrics": {k: v for k, v in val_result.items()
                                    if k != "confusion_matrix"},
                }, best_path)
                print(f"  [best] 新最佳 val_loss={val_result['loss']:.4f} → {best_path}")
            else:
                no_improve += 1
                if early_stop > 0:
                    print(f"  [early_stop] 无提升 {no_improve}/{early_stop}")

            # 早停检查
            if early_stop > 0 and no_improve >= early_stop:
                print(f"[{task}] 早停：连续 {early_stop} 轮无提升，停止训练")
                break

    logger.close()
    print(f"[{task}] 训练完成. best_val_loss={-best_score:.4f}")


# ==================== CLI ====================

def main():
    """CLI 入口：解析参数并分发到 train_physics。"""
    p = argparse.ArgumentParser(description="阶段 3 物理监督网络训练")
    p.add_argument("--task", type=str, required=True,
                   choices=["river", "snowline", "coast"],
                   help="训练任务")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="梯度累积步数（有效 batch = batch_size × grad_accum）")
    p.add_argument("--early-stop", type=int, default=0,
                   help="早停耐心（轮）；0 表示不启用")
    p.add_argument("--data-root", type=str, default=None,
                   help="数据目录路径；默认 terrain/data/dem_tiles/")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    train_physics(
        task=a.task,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        resume=a.resume,
        num_workers=a.num_workers,
        val_ratio=a.val_ratio,
        eval_interval=a.eval_interval,
        data_root=a.data_root,
        grad_accum=a.grad_accum,
        early_stop=a.early_stop,
    )


if __name__ == "__main__":
    main()
