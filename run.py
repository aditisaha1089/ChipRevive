#!/usr/bin/env python
"""
run.py -- DA-JRN Inference Entry Point
Team: ChipRevive | KLA Semicon Hackathon

Usage:
    python run.py INPUT_DIR OUTPUT_DIR

- Reads all .npy files from INPUT_DIR
- Writes restored .npy files (same filenames) to OUTPUT_DIR
- Auto-detects CUDA GPU; falls back to CPU
- No internet / API keys / user interaction required
- Outputs: float32 (H, W) arrays in [0, 1], no NaN/Inf
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow src/ imports when run from the submission folder
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.model import build_model

# ---------------------------------------------------------------------------
# Hard-coded model config (matches training; no external config file needed)
# ---------------------------------------------------------------------------
MODEL_CFG = {
    "model": {
        "in_ch": 1,
        "base_ch": 32,
        "embed_dim": 64,
        "num_blocks": 8,
        "sr_scale": 2,
        "do_sr": True,
    }
}

WEIGHTS_PATH = ROOT / "models" / "best_model.pth"
PAD_MULTIPLE = 16   # pad spatial dims to next multiple before inference
SR_SCALE = 2        # LR -> HR upscale factor  (128 -> 256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[run] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[run] No GPU found -- using CPU.")
    return dev


def load_model(device: torch.device) -> torch.nn.Module:
    if not WEIGHTS_PATH.is_file():
        sys.exit(f"[run] ERROR: model weights not found at {WEIGHTS_PATH}\n"
                 f"       Place best_model.pth inside the models/ folder.")
    model = build_model(MODEL_CFG)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device)
    # Support both raw state_dict and wrapped checkpoint dicts
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run] Loaded: {WEIGHTS_PATH.name}  ({n_params:,} parameters)")
    return model


def load_npy(path: str) -> torch.Tensor:
    """Load .npy file -> float32 tensor (1, 1, H, W) in [0, 1]."""
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :]            # (1, H, W)
    elif arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr[[0], :]                   # multi-channel: keep first
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr[:, :, 0][np.newaxis, :]   # (H,W,C) layout

    # Normalise to [0,1] if values indicate uint8 range
    if arr.max() > 1.5:
        arr = arr / 255.0

    return torch.from_numpy(arr).unsqueeze(0)   # (1, 1, H, W)


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    _, _, H, W = x.shape
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple
    x_padded = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x_padded, (pw, ph)


def unpad(x: torch.Tensor, pw: int, ph: int, scale: int = 1) -> torch.Tensor:
    _, _, H, W = x.shape
    H_out = H - ph * scale
    W_out = W - pw * scale
    return x[:, :, :H_out, :W_out]


def save_npy(tensor: torch.Tensor, path: str) -> None:
    """Save (1,1,H,W) tensor as (H,W) float32 .npy clamped to [0,1]."""
    arr = tensor.squeeze().clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    np.save(path, arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    input_dir  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        sys.exit(f"[run] ERROR: input directory not found: {input_dir}")

    # Create output directory (requirement)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather .npy files
    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        sys.exit(f"[run] ERROR: no .npy files found in {input_dir}")

    print(f"[run] Input dir : {input_dir}  ({len(npy_files)} files)")
    print(f"[run] Output dir: {output_dir}")

    device = get_device()
    model  = load_model(device)

    print(f"[run] Processing {len(npy_files)} images ...")
    times = []
    errors = 0

    for idx, in_path in enumerate(npy_files, 1):
        try:
            x = load_npy(str(in_path)).to(device)       # (1,1,H,W)
            x_pad, (pw, ph) = pad_to_multiple(x, PAD_MULTIPLE)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_pad = model(x_pad)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            out = unpad(out_pad, pw, ph, scale=SR_SCALE) # (1,1,H_out,W_out)

            # Output filename matches input filename (requirement)
            out_path = output_dir / in_path.name
            save_npy(out, str(out_path))

            times.append(elapsed_ms)
            shape_in  = tuple(x.shape[-2:])
            shape_out = tuple(out.shape[-2:])
            print(f"  [{idx:>4}/{len(npy_files)}] {in_path.name}"
                  f"  {shape_in} -> {shape_out}  {elapsed_ms:.1f} ms")

        except Exception as exc:
            errors += 1
            print(f"  [{idx:>4}/{len(npy_files)}] ERROR {in_path.name}: {exc}")

    # Summary
    print()
    if times:
        avg_ms = sum(times) / len(times)
        total_s = sum(times) / 1000.0
        print(f"[run] Finished: {len(times)} OK / {errors} errors")
        print(f"[run] Avg {avg_ms:.1f} ms/image  |  Total {total_s:.1f} s")
        print(f"[run] Outputs: {output_dir}")
    else:
        print("[run] No files were processed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
