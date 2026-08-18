#!/usr/bin/env python
"""
train.py — Training script for DA-JRN
======================================

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume weights/checkpoint_epoch_050.pth

Features:
  - Composite loss: Charbonnier + Sobel + FFT + SSIM (weights in config.yaml)
  - Cosine annealing LR schedule with warmup
  - Mixed precision (AMP) training for speed
  - Best model saved by validation PSNR (configurable to SSIM)
  - Incremental checkpoints every N epochs
  - Tensorboard-compatible CSV log
  - Reproducible via seed in config
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))

from src.model   import build_model
from src.losses  import build_loss
from src.utils   import (
    AverageMeter, compute_psnr, compute_ssim,
    get_device, get_logger, load_config, save_checkpoint,
    load_checkpoint, set_seed,
)
from src.data.dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DA-JRN on semiconductor imagery.")
    p.add_argument("--config",  "-c", default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--resume",  "-r", default=None,
                   help="Resume from checkpoint path (overrides config)")
    p.add_argument("--device",  "-d", default=None,
                   help="Device override: 'cuda' | 'cpu'")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LR warmup + cosine annealing scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """
    Warmup for first 5% of epochs, then cosine anneal to lr_min.
    Uses PyTorch's SequentialLR.
    """
    t_cfg       = cfg.get("training", {})
    epochs      = t_cfg.get("epochs",   100)
    lr_min      = t_cfg.get("lr_min",   1e-6)
    warmup_ep   = max(1, int(epochs * 0.05))   # 5% warmup

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_ep,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = epochs - warmup_ep,
        eta_min= lr_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup, cosine],
        milestones  = [warmup_ep],
    )


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler,
    device, cfg, logger, epoch: int,
) -> dict:
    model.train()
    t_cfg        = cfg.get("training", {})
    grad_clip    = t_cfg.get("grad_clip",    1.0)
    log_interval = t_cfg.get("log_interval", 50)
    use_amp      = t_cfg.get("amp", True) and device.type == "cuda"

    meters = {
        "total": AverageMeter("loss"),
        "charb": AverageMeter("charb"),
        "sobel": AverageMeter("sobel"),
        "fft":   AverageMeter("fft"),
        "ssim":  AverageMeter("ssim"),
    }
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        degraded = batch["degraded"].to(device, non_blocking=True)  # (B, 1, H_lr, W_lr)
        clean    = batch["clean"].to(device, non_blocking=True)      # (B, 1, H_gt, W_gt)

        optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------ #
        # Defensive wrapper on the very first batch so we always get a
        # full traceback in train.log if something explodes on GPU.
        # After batch 0 completes cleanly we drop the try/except overhead.
        # ------------------------------------------------------------------ #
        def _run_step():
            if use_amp:
                with autocast(device_type="cuda"):
                    restored = model(degraded)
                    total_loss, comps = loss_fn(restored, clean, return_components=True)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                restored = model(degraded)
                total_loss, comps = loss_fn(restored, clean, return_components=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            return total_loss, comps

        if batch_idx == 0:
            try:
                total_loss, comps = _run_step()
            except Exception:
                tb_str = traceback.format_exc()
                err_msg = (
                    f"\n{'='*70}\n"
                    f"TRAINING CRASHED on epoch {epoch}, batch 0\n"
                    f"{'='*70}\n"
                    f"{tb_str}"
                    f"{'='*70}\n"
                )
                # Write to both stderr and the log file immediately
                print(err_msg, file=sys.stderr, flush=True)
                logger.error(err_msg)
                raise  # re-raise so the caller also sees it
        else:
            total_loss, comps = _run_step()

        B = degraded.size(0)
        for k, v in comps.items():
            if k in meters:
                meters[k].update(v, B)

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch}] step [{batch_idx+1}/{len(loader)}] "
                f"loss={meters['total'].avg:.4f} "
                f"charb={meters['charb'].avg:.4f} "
                f"sobel={meters['sobel'].avg:.4f} "
                f"fft={meters['fft'].avg:.4f} "
                f"ssim={meters['ssim'].avg:.4f} "
                f"time={elapsed:.1f}s"
            )

    return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg) -> dict:
    model.eval()
    use_amp = cfg.get("training", {}).get("amp", True) and device.type == "cuda"

    loss_meter = AverageMeter("val_loss")
    psnr_meter = AverageMeter("psnr")
    ssim_meter = AverageMeter("ssim")

    for batch in loader:
        degraded = batch["degraded"].to(device, non_blocking=True)
        clean    = batch["clean"].to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda"):
                restored = model(degraded)
                loss, _  = loss_fn(restored, clean, return_components=True)
        else:
            restored = model(degraded)
            loss, _  = loss_fn(restored, clean, return_components=True)

        B = degraded.size(0)
        loss_meter.update(loss.item(), B)

        # Compute per-image metrics
        for b in range(B):
            p = restored[b:b+1]
            t = clean[b:b+1]
            # Resize pred to GT size if needed (SR head may differ slightly)
            if p.shape != t.shape:
                import torch.nn.functional as F
                p = F.interpolate(p, size=t.shape[-2:], mode="bilinear", align_corners=False)
            psnr_meter.update(compute_psnr(p, t), 1)
            ssim_meter.update(compute_ssim(p, t), 1)

    return {
        "val_loss": loss_meter.avg,
        "psnr":     psnr_meter.avg,
        "ssim":     ssim_meter.avg,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    t_cfg = cfg.get("training", {})
    c_cfg = cfg.get("checkpoint", {})

    seed = t_cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(args.device) if args.device else get_device()

    # Logging
    ckpt_dir = c_cfg.get("dir", "weights")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(ckpt_dir, "train.log")
    logger   = get_logger("dajrn", log_file)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")

    # Model
    model   = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)   # move Sobel/SSIM kernel buffers to GPU

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = t_cfg.get("lr",           2e-4),
        weight_decay = t_cfg.get("weight_decay", 1e-4),
    )
    scaler = GradScaler("cuda", enabled=(t_cfg.get("amp", True) and device.type == "cuda"))

    # DataLoaders
    train_loader, val_loader = build_dataloaders(cfg)
    epochs = t_cfg.get("epochs", 100)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Resume
    start_epoch = 1
    best_metric = -float("inf")
    best_metric_name = c_cfg.get("best_metric", "psnr")

    resume_path = args.resume or c_cfg.get("resume", "")
    if resume_path and os.path.isfile(resume_path):
        start_epoch, prev_metrics = load_checkpoint(
            resume_path, model, optimizer, device=str(device)
        )
        start_epoch += 1
        best_metric = prev_metrics.get(best_metric_name, -float("inf"))
        logger.info(f"Resumed from {resume_path} (epoch {start_epoch-1})")

    # CSV log
    csv_path = os.path.join(ckpt_dir, "metrics.csv")
    csv_exists = os.path.isfile(csv_path)
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "lr", "train_loss", "val_loss", "psnr", "ssim",
    ])
    if not csv_exists:
        csv_writer.writeheader()

    val_interval  = t_cfg.get("val_interval",  1)
    save_interval = t_cfg.get("save_interval", 5)

    logger.info(f"Starting training: epochs {start_epoch}-{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler,
            device, cfg, logger, epoch,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Validate
        val_metrics = {}
        if epoch % val_interval == 0:
            val_metrics = validate(model, val_loader, loss_fn, device, cfg)

            metric_val = val_metrics.get(best_metric_name, -float("inf"))
            is_best    = metric_val > best_metric
            if is_best:
                best_metric = metric_val

            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pth")
            if epoch % save_interval == 0 or is_best:
                save_checkpoint(
                    model, optimizer, epoch,
                    {**train_metrics, **val_metrics},
                    save_path=ckpt_path,
                    is_best=is_best,
                )
                if is_best:
                    logger.info(f"  ** New best {best_metric_name}: {best_metric:.4f}")

            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"lr={current_lr:.6f} "
                f"train_loss={train_metrics['total']:.4f} "
                f"val_loss={val_metrics.get('val_loss', 0):.4f} "
                f"PSNR={val_metrics.get('psnr', 0):.2f} dB "
                f"SSIM={val_metrics.get('ssim', 0):.4f} "
                f"time={time.time()-epoch_t0:.1f}s"
            )

            csv_writer.writerow({
                "epoch":      epoch,
                "lr":         current_lr,
                "train_loss": train_metrics["total"],
                "val_loss":   val_metrics.get("val_loss", ""),
                "psnr":       val_metrics.get("psnr", ""),
                "ssim":       val_metrics.get("ssim", ""),
            })
            csv_file.flush()
        else:
            logger.info(
                f"Epoch [{epoch}/{epochs}] "
                f"lr={current_lr:.6f} "
                f"train_loss={train_metrics['total']:.4f} "
                f"time={time.time()-epoch_t0:.1f}s"
            )

    csv_file.close()
    logger.info("Training complete.")
    logger.info(f"Best {best_metric_name}: {best_metric:.4f}")
    logger.info(f"Best model saved to: {os.path.join(ckpt_dir, 'best_model.pth')}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb_str = traceback.format_exc()
        # Best-effort: write to stderr so Colab always shows it even if the
        # logger hasn't been initialised yet.
        print("\nFATAL: training aborted with unhandled exception:\n",
              tb_str, file=sys.stderr, flush=True)
        # Also try to append to train.log if it exists already
        _log = Path("weights") / "train.log"
        try:
            _log.parent.mkdir(parents=True, exist_ok=True)
            with _log.open("a") as _f:
                _f.write(f"\nFATAL EXCEPTION:\n{tb_str}\n")
        except Exception:
            pass  # nothing more we can do
        sys.exit(1)
