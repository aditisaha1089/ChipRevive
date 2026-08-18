# Semiconductor Image Restoration — DA-JRN

**Degradation-Aware Joint Restoration Network** for restoring degraded grayscale
semiconductor inspection images (SEM/optical inspection).  
Built for the KLA Semicon Hackathon.

---

## Problem Summary

Semiconductor inspection images suffer from two types of degradation that can
occur independently or simultaneously:

1. **Speckle noise** — multiplicative coherent-imaging noise (`I_noisy ≈ I_clean × η`)
   that amplifies bright pixels and obscures fine structural detail.
2. **Spatial resolution reduction** — 2× downsampling (512→256 or 256→128) that
   permanently discards high-frequency content, which cannot be recovered by
   naive interpolation.

A single model must handle all three scenarios (noise only, SR only, both combined)
**without being told at inference time which degradation(s) are present**,
and must generalize to out-of-distribution semiconductor structure types not seen during training.

---

## Architecture — DA-JRN

```
Input NoisyLR (128×128)
        │
  ┌─────▼─────────────────────────────────────┐
  │  Degradation Estimator (4-conv CNN + GAP) │  ← learns noise/SR severity
  │  → compact embedding z ∈ ℝ⁶⁴             │     end-to-end, no labels needed
  └─────────────────┬─────────────────────────┘
                    │ z (FiLM conditioning)
        │
  log(1+x) transform     ← converts multiplicative speckle → additive in log-space
        │
  ┌─────▼────────────────────────────────────┐
  │  NAFNet Restoration Backbone (8 blocks)  │
  │  Each block:                              │
  │    DW-sep conv → SimpleGate → SE-attn    │
  │    FiLM(z):  feat = γ(z)·feat + β(z)    │  ← per-image adaptive behavior
  │    Residual + FFN                         │
  └─────────────────┬────────────────────────┘
        │
  exp(x)-1 transform     ← invert log transform
        │
  ┌─────▼──────────────────────────────────────┐
  │  PixelShuffle SR Head (×2, gated by z)     │  ← gate≈0: identity (noise-only)
  │  conv(C→C·r²) → PixelShuffle(r=2)         │  ← gate≈1: learned SR upsampling
  └─────────────────┬──────────────────────────┘
        │
  Output restored (256×256), clipped [0,1]
```

### Why this is NOT a generic U-Net

| Design choice | Reason |
|---|---|
| **Log-domain transform** | Speckle is *multiplicative*. `log(I_noisy) ≈ log(I_clean) + log(η)` — additive in log-space, structurally easier for a CNN to subtract via residual learning. Standard U-Nets process linear-domain signals and implicitly treat speckle as additive, degrading performance. |
| **Degradation Estimator + FiLM** | One model must handle 3 degradation scenarios without being told which. FiLM injects a per-image learned embedding as affine modulation (γ·feat + β) at every backbone block — the network *adapts its restoration behavior* per image rather than applying one fixed transformation. This is fundamentally different from generic encoder-decoders. |
| **NAFNet blocks (no norm)** | Transformers (Restormer, SwinIR) are 3–5× slower at inference for marginal PSNR gains. NAFNet's SimpleGate + depthwise conv gives near-Transformer quality at near-U-Net speed. Removing BatchNorm/LayerNorm prevents distribution-shift failures on out-of-distribution structure types. |
| **PixelShuffle SR** | Bicubic/bilinear interpolation cannot recover detail — it only smooths. PixelShuffle *learns* to synthesize high-frequency content from the depth dimension. The embedding-gated path lets it turn itself off for noise-only inputs. |
| **Composite loss (4 terms)** | MSE alone over-smooths and misses high-frequency detail. Sobel loss directly penalizes edge error (critical since defects are edge-like). FFT loss penalizes missing frequency content even when pixel-wise loss is low. Charbonnier is robust to speckle outliers where MSE would over-weight extreme pixels. |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/aditisaha1089/ChipRevive.git
cd ChipRevive

# 2. Create environment (Python 3.9+)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
# Download from:
#   https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6
# Place at:
#   data/raw/NoisyLR/   ← 3200 .npy files (128×128)
#   data/raw/GT/        ← 3200 .npy files (256×256)
#   data/test/NoisyLR/  ← 400  .npy files (128×128, no GT)
```

---

## Training

```bash
python train.py --config config.yaml
```

To resume from a checkpoint:
```bash
python train.py --config config.yaml --resume weights/checkpoint_epoch_050.pth
```

Training produces:
- `weights/best_model.pth` — best checkpoint by validation PSNR
- `weights/checkpoint_epoch_NNN.pth` — periodic checkpoints
- `weights/metrics.csv` — epoch-by-epoch metrics log
- `weights/train.log` — full training log

---

## Inference (copy-paste ready)

```bash
python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_outputs
```

This writes one restored `.npy` per input file to `restored_outputs/`.

Additional options:
```bash
# Save as PNG for visual inspection
python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_outputs --output_format png

# Custom weights path
python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_outputs --weights weights/checkpoint_epoch_100.pth

# Force CPU
python evaluate.py --input_dir data/test/NoisyLR --output_dir restored_outputs --device cpu
```

The script requires **no manual edits** — it reads weights path from `config.yaml`
(`inference.weights_path`, default: `weights/best_model.pth`).

---

## Loss Function

| Term | Formula | Weight | Purpose |
|---|---|---|---|
| **Charbonnier** | `√((ŷ-y)² + ε²)` | 1.0 | Robust L1; handles speckle outliers better than MSE |
| **Sobel/Gradient** | `L1(∇ŷ, ∇y)` (Sobel-x + Sobel-y) | 0.5 | Defect edge preservation |
| **FFT/Frequency** | `L1(log\|FFT(ŷ)\|, log\|FFT(y)\|)` | 0.1 | Penalizes missing high-frequency content |
| **SSIM** | `1 − SSIM(ŷ, y)` (Gaussian window) | 0.3 | Structural fidelity |

All weights configurable in `config.yaml` under `loss:`.

---

## Reported Metrics

> **Training status:** Running on CPU. Numbers below are from **Epoch 1 / 100** of an ongoing
> training run. Final numbers will improve significantly by epoch 100 as the model converges.
> Re-run `python evaluate.py --input_dir data/raw/NoisyLR --output_dir val_out` on paired data
> after training finishes to get final PSNR/SSIM.

| Metric | Epoch 1 (CPU, untrained baseline) | Notes |
|---|---|---|
| **Validation PSNR** | **22.48 dB** | Epoch 1/100; expect 28-32 dB at convergence |
| **Validation SSIM** | **0.581** | Epoch 1/100; expect 0.85+ at convergence |
| **Val loss** | 0.276 | Composite (Charb + Sobel + FFT + SSIM) |
| **Train loss (ep 1)** | 0.450 | Dropped to 0.286 by step 50 of epoch 2 |
| **Inference time (CPU, i7/Ryzen)** | **~125 ms/image** | Measured on 400 real test files |
| **Inference time (H100 GPU)** | Expected ~2-5 ms/image | NAFNet chosen specifically for H100 speed |
| **Model parameters** | **164,276** | Lightweight; fits comfortably in GPU memory |
| **Test files processed** | **400/400** | Zero crashes, all shapes correct (256x256) |

To monitor training live:
```bash
# See real-time epoch metrics:
Get-Content weights\train.log -Wait       # Windows
tail -f weights/train.log                 # Linux/macOS

# CSV of all epochs so far:
Get-Content weights\metrics.csv
```

---

## Running Tests

```bash
python -m pytest tests/ -v
# or
python tests/test_evaluate.py
```

Tests verify:
- `evaluate.py` produces one output file per input file
- Output filenames match input filenames
- Output arrays are shape `(256, 256)` and values in `[0, 1]`
- Missing weights cause a non-zero exit code (not a silent crash)
- PNG output format works

---

## Model Weights

Weights are excluded from git history (see `.gitignore`).

After training completes, `weights/best_model.pth` will be created automatically.
Expected size: **~2-4 MB** (164k parameters × 4 bytes × overhead) — small enough to commit directly
without Git LFS.

**To commit weights directly** (recommended for this model size):
```bash
# Remove the *.pth line from .gitignore first:
git add weights/best_model.pth
git commit -m "add trained model weights (best val PSNR)"
git push
```

**If weights somehow exceed 100 MB** (e.g. after architecture scaling):
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes weights/best_model.pth
git commit -m "add model weights via LFS"
```

Trained weights download: `[Upload to Google Drive or HuggingFace Hub after training]`

---

## Known Limitations

- **No test GT**: The held-out test set (`data/test/NoisyLR/`) has no local GT —
  KLA holds ground truth for scoring. Validation PSNR is therefore measured only
  on the 15% train-split validation set, which may not perfectly predict test score.
- **Fixed SR scale**: The SR head assumes exactly 2× upsampling (128→256).
  A scale-adaptive variant (predicting scale from embedding) would generalize further.
- **Single-channel only**: Extending to multi-channel inspection modes would require
  adding cross-channel attention in the backbone.
- **No test-time augmentation**: Averaging predictions over flipped copies typically
  adds +0.2–0.5 dB PSNR at 4–8× inference cost — not implemented given speed scoring.
- **Training on synthetic-only data risk**: All training pairs use the real NoisyLR/GT
  files. If the hackathon test set contains degradation types not present in training,
  the model may underperform. The synthetic augmentation pipeline (`src/data/degradation.py`)
  can generate additional variety if needed.

## What We'd Improve with More Time

1. **Test-time ensembling** (flip + rotate averaging for +0.3 dB)
2. **Self-supervised pretraining** on unlabeled test images via blind-spot networks
3. **Curriculum training** — start with easy (noise-only) samples, progressively add SR
4. **Attention U-Net skip connections** for better preservation of fine structural edges
5. **Quantization/TorchScript export** for fastest possible H100 throughput
