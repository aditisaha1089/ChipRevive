"""
Utility functions for DA-JRN training and evaluation.
"""

import os
import math
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to console (and optionally a file)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        pred   : Predicted image tensor, any shape, values in [0, max_val]
        target : Ground truth tensor, same shape
        max_val: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        PSNR in dB (higher is better). Returns inf if MSE = 0.
    """
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Structural Similarity Index (SSIM).

    Returns mean SSIM across the batch/image (higher is better, max=1.0).
    Computes using a Gaussian window as per the original Wang et al. paper.
    """
    pred   = pred.float()
    target = target.float()

    # Ensure same size
    if pred.shape != target.shape:
        pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

    # Build Gaussian kernel
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    B, C, H, W = pred.shape
    pad = window_size // 2

    mu_p  = F.conv2d(pred,   kernel, padding=pad)
    mu_t  = F.conv2d(target, kernel, padding=pad)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t

    sig_p2 = F.conv2d(pred   ** 2, kernel, padding=pad) - mu_p2
    sig_t2 = F.conv2d(target ** 2, kernel, padding=pad) - mu_t2
    sig_pt = F.conv2d(pred * target, kernel, padding=pad) - mu_pt

    num = (2 * mu_pt  + C1) * (2 * sig_pt  + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p2 + sig_t2 + C2)

    ssim_map = num / den.clamp(min=1e-8)
    return ssim_map.mean().item()


def compute_metrics(
    pred: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Compute PSNR and SSIM in one call."""
    return {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
    }


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[int, Dict[str, float]]:
    """
    Load checkpoint into model (and optionally optimizer).

    Returns:
        (start_epoch, metrics_dict)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch   = checkpoint.get("epoch",   0)
    metrics = checkpoint.get("metrics", {})
    return epoch, metrics


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """
    Convert a (1, H, W) or (H, W) float tensor in [0,1] to a PIL Image.
    """
    if t.dim() == 3:
        t = t.squeeze(0)
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image (any mode) to a (1, H, W) float tensor in [0, 1].
    """
    img_gray = img.convert("L")
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def pad_to_multiple(
    img: torch.Tensor, multiple: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    Pad image spatial dims to the next multiple of `multiple`.
    Returns padded image and padding tuple (left, right, top, bottom).
    Used so any image size can pass through the network without shape errors.
    """
    _, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    # Pad: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    padded = F.pad(img, padding, mode="reflect")
    return padded, padding


def unpad(
    img: torch.Tensor, padding: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Remove padding added by pad_to_multiple."""
    _, _, H, W = img.shape
    left, right, top, bottom = padding
    H_orig = H - bottom
    W_orig = W - right
    return img[:, :, top:H_orig, left:W_orig]


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the running average of a metric."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Auto-select GPU if available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

class InferenceTimer:
    """Context manager for measuring GPU-accurate inference time."""
    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event   = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device.type == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
