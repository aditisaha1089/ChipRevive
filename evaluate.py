#!/usr/bin/env python
"""
evaluate.py — Standalone inference script for DA-JRN
=====================================================

Usage (copy-paste ready):
    python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_outputs

Requirements:
  - Trained weights at: weights/best_model.pth   (set in config.yaml or via --weights)
  - Python environment set up per README.md

The script:
  1. Loads the trained model from weights/
  2. Iterates every .npy file in --input_dir
  3. Runs inference (with defensive padding to handle arbitrary spatial sizes)
  4. Writes the restored array as .npy to --output_dir (matching filename)
  5. Prints per-image inference time and a summary table at the end

Handles:
  - .npy inputs (real dataset format)
  - .png / .tiff inputs (converts to float32 grayscale [0,1])
  - Arbitrary image sizes (pad-to-multiple-of-16 before inference, unpad after)
  - CPU and CUDA (auto-selected)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.model import build_model
from src.utils import get_device, load_config, tensor_to_image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="DA-JRN inference: restore degraded semiconductor inspection images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_dir", "-i", required=True,
        help="Directory containing degraded input files (.npy, .png, or .tiff).",
    )
    p.add_argument(
        "--output_dir", "-o", required=True,
        help="Directory to write restored outputs (same filenames as input).",
    )
    p.add_argument(
        "--weights", "-w", default=None,
        help="Path to model weights .pth file. "
             "If not set, reads from config.yaml inference.weights_path "
             "(default: weights/best_model.pth).",
    )
    p.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config.yaml.",
    )
    p.add_argument(
        "--device", "-d", default=None,
        help="Device override: 'cuda' | 'cpu'. Auto-detects if not set.",
    )
    p.add_argument(
        "--output_format", default="npy",
        choices=["npy", "png"],
        help="Output file format. 'npy' preserves float32 precision; "
             "'png' converts to uint8 for visual inspection.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Input loading (handles .npy and image formats)
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def load_input(path: str) -> torch.Tensor:
    """
    Load a degraded input file as a float32 tensor (1, 1, H, W).

    Supports:
      - .npy  : loaded directly as float32 array
      - image : converted to float32 grayscale [0, 1]

    Returns tensor on CPU ready for model inference.
    """
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]          # (1, H, W)
        elif arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[[0], ...]                 # take first channel
        t = torch.from_numpy(arr)               # (1, H, W)

    elif ext in SUPPORTED_IMAGE_EXTS:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return t.unsqueeze(0)  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# Defensive padding (handles arbitrary spatial sizes)
# ---------------------------------------------------------------------------

def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    """
    Reflect-pad x to the next spatial multiple of `multiple`.
    Returns (padded_x, (pad_left, pad_right, pad_top, pad_bottom)).
    """
    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padding = (0, pad_w, 0, pad_h)          # F.pad order: left, right, top, bottom
    return F.pad(x, padding, mode="reflect"), padding


def unpad(x: torch.Tensor, padding, scale: int = 1):
    """
    Remove padding added by pad_to_multiple, accounting for SR upscaling.

    scale=2 means output is 2× larger than input → multiply pad extents by 2.
    """
    left, right, top, bottom = [p * scale for p in padding]
    _, _, H, W = x.shape
    H_orig = H - bottom
    W_orig = W - right
    return x[:, :, top:H_orig, left:W_orig]


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_output(tensor: torch.Tensor, out_path: str, fmt: str = "npy") -> None:
    """
    Save restored tensor to disk.

    Args:
        tensor   : (1, 1, H, W) or (1, H, W) float32 tensor in [0, 1]
        out_path : Destination path (without extension override)
        fmt      : 'npy' or 'png'
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t = tensor.squeeze()                     # (H, W)
    arr = t.clamp(0.0, 1.0).cpu().numpy()   # float32, [0,1]

    if fmt == "npy":
        np.save(out_path, arr)
    elif fmt == "png":
        from PIL import Image
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img.save(out_path)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    if not os.path.isfile(args.config):
        print(f"[WARNING] config.yaml not found at {args.config}; using defaults.")
        cfg = {}
    else:
        cfg = load_config(args.config)

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"[evaluate] Device: {device}")

    # ── Weights path ────────────────────────────────────────────────────────
    weights_path = (
        args.weights
        or cfg.get("inference", {}).get("weights_path", "weights/best_model.pth")
    )
    if not os.path.isfile(weights_path):
        print(f"[ERROR] Model weights not found at: {weights_path}")
        print("  Train first:  python train.py --config config.yaml")
        print("  Or pass:      --weights <path_to_checkpoint.pth>")
        sys.exit(1)

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(cfg)
    checkpoint = torch.load(weights_path, map_location=device)
    # Support both raw state_dict and wrapped checkpoint dicts
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    pad_multiple = cfg.get("inference", {}).get("pad_multiple", 16)
    sr_scale     = cfg.get("model",     {}).get("sr_scale",     2)

    print(f"[evaluate] Loaded weights: {weights_path}")
    print(f"[evaluate] Input dir:  {args.input_dir}")
    print(f"[evaluate] Output dir: {args.output_dir}")

    # ── Gather input files ──────────────────────────────────────────────────
    all_exts = {".npy"} | SUPPORTED_IMAGE_EXTS
    input_files = sorted(
        str(p) for p in Path(args.input_dir).rglob("*")
        if p.suffix.lower() in all_exts
    )
    if not input_files:
        print(f"[ERROR] No supported files found in {args.input_dir}")
        sys.exit(1)

    print(f"[evaluate] Found {len(input_files)} files to process.\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────────
    times_ms = []
    results   = []

    for i, in_path in enumerate(input_files):
        fname    = Path(in_path).name
        stem     = Path(in_path).stem
        out_fname = f"{stem}.{args.output_format}"
        out_path  = os.path.join(args.output_dir, out_fname)

        # Load
        try:
            x = load_input(in_path).to(device)         # (1, 1, H, W)
        except Exception as e:
            print(f"  [{i+1}/{len(input_files)}] SKIP {fname}: {e}")
            continue

        # Pad to multiple
        x_padded, padding = pad_to_multiple(x, multiple=pad_multiple)

        # Timed inference
        if device.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_ev.record()
            with torch.no_grad():
                out_padded = model(x_padded)
            end_ev.record()
            torch.cuda.synchronize()
            elapsed_ms = start_ev.elapsed_time(end_ev)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                out_padded = model(x_padded)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Remove padding (output is sr_scale× larger)
        out = unpad(out_padded, padding, scale=sr_scale)  # (1,1, H_gt, W_gt)

        # Save
        save_output(out, out_path, fmt=args.output_format)

        times_ms.append(elapsed_ms)
        results.append((fname, elapsed_ms, out.shape))

        print(f"  [{i+1:>4}/{len(input_files)}] {fname} -> {out_fname}  "
              f"| in={tuple(x.shape[-2:])} out={tuple(out.shape[-2:])} "
              f"| {elapsed_ms:.1f} ms")

    # ── Summary ─────────────────────────────────────────────────────────────
    if times_ms:
        avg_ms  = sum(times_ms) / len(times_ms)
        min_ms  = min(times_ms)
        max_ms  = max(times_ms)
        fps     = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
        print()
        print("="*60)
        print("INFERENCE SUMMARY")
        print("="*60)
        print(f"  Files processed : {len(times_ms)}/{len(input_files)}")
        print(f"  Avg time/image  : {avg_ms:.2f} ms  ({fps:.1f} img/s)")
        print(f"  Min time        : {min_ms:.2f} ms")
        print(f"  Max time        : {max_ms:.2f} ms")
        print(f"  Total time      : {sum(times_ms)/1000:.2f} s")
        print(f"  Output dir      : {args.output_dir}")
        print("="*60)


if __name__ == "__main__":
    main()
