"""阶段 2 训练入口：半监督潜空间条件扩散（LDM）。

半监督策略
----------
1. **无标签损失**（主）：大量无标签 DEM → VAE 编码到潜空间 → 标准扩散损失
   - 学习地形流形分布，不依赖配对
2. **监督损失**（辅）：少量配对 (条件, DEM) → 条件扩散损失
   - 学习条件遵循，确保板块骨架能控制生成
3. **一致性正则**：对同一条件加不同噪声扰动，模型输出应一致
   - 提升条件鲁棒性，减少伪标签噪声影响

训练目标
--------
    L = λ_unsup · E[||ε - ε_pred(z_t, t, ∅)||²]                    # 无标签
      + λ_sup   · E[||ε - ε_pred(z_t, t, cond)||²]                  # 配对
      + λ_cons  · E[||ε_pred(z_t, t, cond) - ε_pred(z_t', t, cond')||²]  # 一致性

其中 cond' = cond + δ·N(0,1)（条件扰动），z_t' = q_sample(z0, t)（重采样）。

用法
----
    # 先确保阶段 1 VAE 已训练
    python -m terrain.dl.train_ldm --epochs 200 --batch-size 8
    python -m terrain.dl.train_ldm --resume
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .config import (
    DEVICE, CKPT_DIR, LOG_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_STEPS, GRAD_CLIP, EMA_DECAY, USE_AMP, DIFFUSION_STEPS,
    BETA_START, BETA_END, IN_CHANNELS, LATENT_CHANNELS, LATENT_SIZE,
    TILE_SIZE, EVAL_INTERVAL_EPOCHS, SAMPLE_NUM, PAIRED_DIR, DEM_TILES_DIR,
)
from .dataset import DEMTileDataset
from .paired_dataset import PairedDEMDataset
from .vae import TerrainVAE
from .conditioner import ConditionEncoder
from .latent_unet import LatentDiffusionUNet
from .diffusion import DDPM
from .ddim import DDIM
from .logger import TrainLogger
from .evaluator import Evaluator
from .train import EMA, cosine_warmup_lr, amp_enabled, _train_one_stage


# ==================== 半监督损失 ====================

def latent_diffusion_loss(model: LatentDiffusionUNet, vae: TerrainVAE,
                          cond_encoder: ConditionEncoder, ddpm: DDPM,
                          dem_batch: torch.Tensor,
                          cond_batch: torch.Tensor | None,
                          lambda_sup: float = 1.0,
                          lambda_cons: float = 0.1,
                          cons_delta: float = 0.1) -> tuple[torch.Tensor, dict]:
    """半监督潜空间扩散损失。

    Parameters
    ----------
    model : LatentDiffusionUNet
    vae : TerrainVAE（冻结，仅编码用）
    cond_encoder : ConditionEncoder
    ddpm : DDPM 调度器
    dem_batch : (B, 1, 256, 256) [-1,1]
        DEM 切片（无标签或有标签的 dem 部分）。
    cond_batch : (B, 12, 256, 256) or None
        配对样本的条件；None 时仅做无标签损失。
    lambda_sup : float
        监督损失权重。
    lambda_cons : float
        一致性正则权重。
    cons_delta : float
        条件扰动幅度。

    Returns
    -------
    (total_loss, info_dict)
    """
    B = dem_batch.size(0)
    info = {}

    # ---- VAE 编码到潜空间（冻结）----
    with torch.no_grad():
        mu, logvar = vae.encode(dem_batch)
        z0 = vae.reparam(mu, logvar)   # (B, 4, 32, 32)

    # ---- 采样时间步 ----
    t = torch.randint(0, ddpm.n, (B,), device=dem_batch.device)

    # ---- 无标签损失：条件为零向量 ----
    z_t, noise = ddpm.q_sample(z0, t)
    null_cond = torch.zeros(B, cond_encoder.cond_ch, LATENT_SIZE, LATENT_SIZE,
                            device=dem_batch.device)
    eps_pred_unsup = model(z_t, t, null_cond)
    loss_unsup = F.mse_loss(eps_pred_unsup, noise)
    info["loss_unsup"] = float(loss_unsup.detach())
    total = loss_unsup

    # ---- 监督损失：条件来自真实配对 ----
    if cond_batch is not None and lambda_sup > 0:
        cond_feat = cond_encoder(cond_batch)   # (B, cond_ch, 32, 32)
        eps_pred_sup = model(z_t, t, cond_feat)
        loss_sup = F.mse_loss(eps_pred_sup, noise)
        info["loss_sup"] = float(loss_sup.detach())
        total = total + lambda_sup * loss_sup

        # ---- 一致性正则：条件扰动后输出应一致 ----
        if lambda_cons > 0:
            cond_noisy = cond_batch + cons_delta * torch.randn_like(cond_batch)
            cond_feat_noisy = cond_encoder(cond_noisy)
            with torch.no_grad():
                eps_ref = model(z_t, t, cond_feat_noisy)
            loss_cons = F.mse_loss(eps_pred_sup, eps_ref.detach())
            info["loss_cons"] = float(loss_cons.detach())
            total = total + lambda_cons * loss_cons

    info["loss"] = float(total.detach())
    return total, info


# ==================== 采样回调 ====================

def make_sample_fn_ldm(vae: TerrainVAE, cond_encoder: ConditionEncoder,
                       model: LatentDiffusionUNet, ema_model: EMA,
                       ema_cond: EMA, ddim: DDIM,
                       cond_batch: torch.Tensor | None = None):
    """为 LDM 构造采样回调。

    cond_batch 为 None 时做无条件采样（验证分布学习）；
    给定时做条件采样（验证条件遵循）。
    """
    @torch.no_grad()
    def sample(n: int) -> np.ndarray:
        """用 EMA 权重做条件/无条件 DDIM 采样并 VAE 解码。"""
        # 用 EMA 权重
        ema_model.apply(model)
        ema_cond.apply(cond_encoder)
        try:
            if cond_batch is not None:
                # 条件采样
                cond = cond_encoder(cond_batch[:n].to(DEVICE))
            else:
                cond = torch.zeros(n, cond_encoder.cond_ch, LATENT_SIZE, LATENT_SIZE,
                                   device=DEVICE)
            z = ddim.sample(model, (n, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                            cond, DEVICE, progress=False)
            x = vae.decode(z)
        finally:
            ema_model.restore(model)
            ema_cond.restore(cond_encoder)
        x = x.squeeze(1).cpu().numpy()
        from .config import norm_to_elev
        return norm_to_elev(x)
    return sample


# ==================== 主训练循环 ====================

def train_ldm(epochs: int = 200, batch_size: int = BATCH_SIZE,
              lambda_sup: float = 1.0, lambda_cons: float = 0.1,
              resume: bool = False, vae_ckpt: str | None = None):
    """半监督训练潜空间条件扩散模型。

    Parameters
    ----------
    epochs : int
    batch_size : int
    lambda_sup : float
        监督损失权重。
    lambda_cons : float
        一致性正则权重。
    resume : bool
        断点续训。
    vae_ckpt : str, optional
        阶段 1 VAE 权重路径；默认 checkpoints/vae_phase1.pt。
    """
    # ---- 加载冻结的 VAE ----
    vae_path = vae_ckpt or str(CKPT_DIR / "vae_phase1.pt")
    if not Path(vae_path).exists():
        raise RuntimeError(f"VAE 权重不存在: {vae_path}，请先跑阶段 1 VAE 训练")
    vae = TerrainVAE(in_ch=IN_CHANNELS, latent_ch=LATENT_CHANNELS).to(DEVICE)
    vae.load_state_dict(torch.load(vae_path, map_location=DEVICE, weights_only=True)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ---- 数据 ----
    unsup_ds = DEMTileDataset(DEM_TILES_DIR, augment=True)
    unsup_loader = DataLoader(unsup_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    paired_loader = None
    try:
        paired_ds = PairedDEMDataset(PAIRED_DIR, augment=True)
        paired_loader = DataLoader(paired_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=2, drop_last=True)
        print(f"[LDM] 配对样本 {len(paired_ds)} 个，启用半监督")
    except RuntimeError as e:
        print(f"[LDM] 无配对样本，仅无监督训练（{e}）")

    n_steps_total = epochs * len(unsup_loader)

    # ---- 模型 ----
    cond_encoder = ConditionEncoder(in_ch=12, cond_ch=LATENT_CHANNELS).to(DEVICE)
    model = LatentDiffusionUNet(in_ch=LATENT_CHANNELS, cond_ch=LATENT_CHANNELS).to(DEVICE)
    ddpm = DDPM(n_steps=DIFFUSION_STEPS, beta_start=BETA_START, beta_end=BETA_END,
                device=DEVICE)
    ddim = DDIM(n_train_steps=DIFFUSION_STEPS, n_sample_steps=50,
                beta_start=BETA_START, beta_end=BETA_END, eta=0.0, device=DEVICE)

    # 优化器：只优化 LDM 和 cond_encoder，VAE 冻结
    params = list(model.parameters()) + list(cond_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ema_model = EMA(model)
    ema_cond = EMA(cond_encoder)
    ckpt_path = CKPT_DIR / "ldm_phase2.pt"

    logger = TrainLogger("ldm_phase2", LOG_DIR)
    evaluator = Evaluator("ldm_phase2", sample_dir=LOG_DIR / "samples_ldm",
                          logger=logger)

    # ---- 续训 ----
    start_epoch = 0
    global_step = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ck["model"])
        cond_encoder.load_state_dict(ck["cond_encoder"])
        optimizer.load_state_dict(ck["optimizer"])
        ema_model.shadow = ck["ema_model"]
        ema_cond.shadow = ck["ema_cond"]
        start_epoch = ck["epoch"] + 1
        global_step = ck["step"]
        print(f"[resume] LDM 从 epoch {start_epoch} 续训")

    scaler = torch.amp.GradScaler(DEVICE.type) if amp_enabled() else None
    print(f"[LDM] device={DEVICE}, amp={amp_enabled()}, epochs={epochs}")
    print(f"      λ_sup={lambda_sup}, λ_cons={lambda_cons}")

    # 配对迭代器
    paired_iter = None
    if paired_loader is not None:
        paired_iter = iter(paired_loader)

    best_score = -float("inf")

    for epoch in range(start_epoch, epochs):
        model.train()
        cond_encoder.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for dem_batch in unsup_loader:
            dem_batch = dem_batch.to(DEVICE)

            # 获取配对 batch（若有）
            cond_batch = None
            if paired_iter is not None:
                try:
                    cond_batch, paired_dem = next(paired_iter)
                    # 用配对的 dem 而非无标签的，监督更准确
                    dem_batch = paired_dem.to(DEVICE)
                    cond_batch = cond_batch.to(DEVICE)
                except StopIteration:
                    paired_iter = iter(paired_loader)
                    cond_batch, dem_batch = next(paired_iter)
                    dem_batch = dem_batch.to(DEVICE)
                    cond_batch = cond_batch.to(DEVICE)

            # 学习率
            lr = cosine_warmup_lr(global_step, n_steps_total,
                                  WARMUP_STEPS, LEARNING_RATE)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad()
            if amp_enabled():
                with torch.amp.autocast(DEVICE.type):
                    loss, info = latent_diffusion_loss(
                        model, vae, cond_encoder, ddpm, dem_batch, cond_batch,
                        lambda_sup, lambda_cons)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, info = latent_diffusion_loss(
                    model, vae, cond_encoder, ddpm, dem_batch, cond_batch,
                    lambda_sup, lambda_cons)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                optimizer.step()

            ema_model.update(model)
            ema_cond.update(cond_encoder)
            epoch_loss += info["loss"]
            n_batches += 1
            global_step += 1

            if global_step % 50 == 0:
                logger.log_scalars(
                    {k: v for k, v in info.items() if isinstance(v, (int, float))
                     if k != "loss"} | {"loss": info["loss"], "lr": lr},
                    step=global_step)

        avg = epoch_loss / max(1, n_batches)
        dt = time.time() - t0
        print(f"[LDM] ep{epoch:03d} avg={avg:.4f} lr={lr:.2e} {dt:.1f}s")

        # 保存 last
        torch.save({
            "model": model.state_dict(),
            "cond_encoder": cond_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema_model": ema_model.shadow,
            "ema_cond": ema_cond.shadow,
            "epoch": epoch, "step": global_step,
        }, ckpt_path)

        # 周期评估
        if (epoch + 1) % EVAL_INTERVAL_EPOCHS == 0 or epoch == epochs - 1:
            model.eval()
            cond_encoder.eval()
            sample_fn = make_sample_fn_ldm(vae, cond_encoder, model,
                                            ema_model, ema_cond, ddim)
            result = evaluator.step(epoch, global_step, sample_fn, n=SAMPLE_NUM)
            print(f"  [eval] score={result['quality_score']:.3f} "
                  f"stripe={result['stripe_index']:.3f} "
                  f"psd={result['psd_slope']:.3f}")
            if result["quality_score"] > best_score:
                best_score = result["quality_score"]
                best_path = CKPT_DIR / "best_ldm_phase2.pt"
                torch.save({
                    "model": model.state_dict(),
                    "cond_encoder": cond_encoder.state_dict(),
                    "ema_model": ema_model.shadow,
                    "ema_cond": ema_cond.shadow,
                    "epoch": epoch, "step": global_step,
                    "score": best_score,
                }, best_path)
                print(f"  [best] 新最佳 → {best_path}")

    logger.close()
    print(f"[LDM] 训练完成. best_score={best_score:.3f}")


# ==================== CLI ====================

def main():
    """CLI 入口：解析参数并启动 LDM 训练。"""
    p = argparse.ArgumentParser(description="阶段 2 LDM 训练")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lambda-sup", type=float, default=1.0,
                   help="监督损失权重")
    p.add_argument("--lambda-cons", type=float, default=0.1,
                   help="一致性正则权重")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vae-ckpt", type=str, default=None,
                   help="阶段 1 VAE 权重路径")
    a = p.parse_args()
    train_ldm(epochs=a.epochs, batch_size=a.batch_size,
              lambda_sup=a.lambda_sup, lambda_cons=a.lambda_cons,
              resume=a.resume, vae_ckpt=a.vae_ckpt)


if __name__ == "__main__":
    main()
