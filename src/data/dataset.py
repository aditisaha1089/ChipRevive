"""
PyTorch Dataset and DataLoader for Semiconductor Image Restoration
==================================================================

Real dataset layout (data/):
    data/
    ├── raw/
    │   ├── NoisyLR/    ← 3200 .npy files, shape (128,128), float32, values in ~[-0.05, 1.7]
    │   └── GT/         ← 3200 .npy files, shape (256,256), float32, values in [0, 1]
    └── test/
        └── NoisyLR/    ← 400 .npy files, same format as raw/NoisyLR/ (no GT available)

Pairing: identical filename (e.g. NoisyLR/000040.npy ↔ GT/000040.npy)

Notes on data characteristics observed from real files:
  - NoisyLR values can be slightly negative (sensor noise artifacts) and can exceed 1.0
    (speckle amplification). These are physically valid — the model must handle them.
  - GT is clean, normalized [0, 1].
  - Resolution ratio is exactly 2× (128 → 256), confirming SR scale = 2.
"""

import os
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.degradation import RandomDegradation


# ---------------------------------------------------------------------------
# .npy loading (real dataset format)
# ---------------------------------------------------------------------------

def load_npy_as_tensor(path: str) -> torch.Tensor:
    """
    Load a .npy grayscale array as a float32 tensor (1, H, W).

    The real dataset uses float32 .npy files:
      - NoisyLR: shape (128,128), values in approximately [-0.05, 1.7]
      - GT:      shape (256,256), values in [0, 1]

    We do NOT clamp or normalize here — the model's log_transform handles
    the small negatives by clamping before log1p.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]          # (1, H, W)
    elif arr.ndim == 3 and arr.shape[0] > 1:
        arr = arr[[0], ...]                 # take first channel if multi-channel
    return torch.from_numpy(arr)            # (1, H, W)


def collect_npy_paths(directory: str) -> List[str]:
    """Collect all .npy paths under directory, sorted for reproducibility."""
    if not os.path.isdir(directory):
        return []
    paths = sorted(
        str(p) for p in Path(directory).rglob("*.npy")
    )
    return paths


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------

def stratified_split(
    paths: List[str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split paths into train/val sets.

    Groups by immediate parent directory (= structure type if sub-foldered).
    For a flat directory this degenerates to a simple random split —
    still correct and reproducible.
    """
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = {}
    for p in paths:
        group = str(Path(p).parent)
        groups.setdefault(group, []).append(p)

    train_paths, val_paths = [], []
    for group_paths in groups.values():
        rng.shuffle(group_paths)
        n_val = max(1, math.floor(len(group_paths) * val_ratio))
        val_paths.extend(group_paths[:n_val])
        train_paths.extend(group_paths[n_val:])

    return train_paths, val_paths


# ---------------------------------------------------------------------------
# Paired NoisyLR / GT Dataset  (primary dataset for real data)
# ---------------------------------------------------------------------------

class PairedNpyDataset(Dataset):
    """
    Paired dataset using the real .npy files.

    Each item: (noisy_lr_tensor, gt_tensor)
      - noisy_lr: (1, 128, 128) float32, values ~[-0.05, 1.7]
      - gt:       (1, 256, 256) float32, values [0, 1]

    Training: applies random patch crop at GT resolution (crop_size×crop_size),
              then takes the proportional crop from LR.
    Validation: uses full images (no crop).

    Args:
        lr_paths  : List of NoisyLR .npy paths
        gt_paths  : List of GT .npy paths (same order/length as lr_paths)
        crop_size : Patch size at GT resolution (default 128 → LR patch = 64)
        augment   : Random flip augmentation (train only)
        is_val    : If True, return full images without crop/augment
    """
    def __init__(
        self,
        lr_paths:  List[str],
        gt_paths:  List[str],
        crop_size: int = 128,
        augment:   bool = True,
        is_val:    bool = False,
    ):
        assert len(lr_paths) == len(gt_paths), (
            f"LR/GT count mismatch: {len(lr_paths)} vs {len(gt_paths)}"
        )
        self.lr_paths  = lr_paths
        self.gt_paths  = gt_paths
        self.crop_size = crop_size
        self.augment   = augment and (not is_val)
        self.is_val    = is_val

    def __len__(self) -> int:
        return len(self.lr_paths)

    def _crop_pair(
        self, lr: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random paired crop. GT crop_size×crop_size, LR crop_size/2×crop_size/2.

        Uses the scale ratio between GT and LR to compute matching crop windows.
        """
        _, H_gt, W_gt = gt.shape
        _, H_lr, W_lr = lr.shape

        cs_gt = self.crop_size
        cs_lr = cs_gt // 2                     # LR is half-resolution

        # If image is smaller than crop, use the full image
        if H_gt < cs_gt or W_gt < cs_gt:
            return lr, gt

        top_gt  = random.randint(0, H_gt - cs_gt)
        left_gt = random.randint(0, W_gt - cs_gt)

        # Matching LR window (integer division for exact alignment)
        top_lr  = top_gt  // 2
        left_lr = left_gt // 2

        gt_crop = gt[:, top_gt:top_gt + cs_gt,   left_gt:left_gt + cs_gt]
        lr_crop = lr[:, top_lr:top_lr + cs_lr,   left_lr:left_lr + cs_lr]

        return lr_crop, gt_crop

    def _augment_pair(
        self, lr: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consistent random flips applied to both LR and GT."""
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            gt = torch.flip(gt, dims=[2])
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            gt = torch.flip(gt, dims=[1])
        return lr, gt

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lr = load_npy_as_tensor(self.lr_paths[idx])   # (1, 128, 128)
        gt = load_npy_as_tensor(self.gt_paths[idx])   # (1, 256, 256)

        if not self.is_val:
            lr, gt = self._crop_pair(lr, gt)
            if self.augment:
                lr, gt = self._augment_pair(lr, gt)

        return {
            "degraded": lr,
            "clean":    gt,
            "path":     self.lr_paths[idx],
        }


# ---------------------------------------------------------------------------
# Test Dataset (inference-only, no GT)
# ---------------------------------------------------------------------------

class TestNpyDataset(Dataset):
    """
    Dataset for the held-out test split (data/test/NoisyLR/).
    No GT available — returns only the degraded input.
    """
    def __init__(self, lr_dir: str):
        self.lr_paths = collect_npy_paths(lr_dir)
        if not self.lr_paths:
            raise FileNotFoundError(f"No .npy files found in {lr_dir}")
        print(f"[TestDataset] {len(self.lr_paths)} test files from {lr_dir}")

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lr = load_npy_as_tensor(self.lr_paths[idx])
        return {
            "degraded": lr,
            "path":     self.lr_paths[idx],
        }


# ---------------------------------------------------------------------------
# Build intersected paired paths from LR + GT directories
# ---------------------------------------------------------------------------

def build_paired_paths(
    lr_dir: str,
    gt_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Build matched (LR, GT) path lists by intersecting filenames.

    Returns only files present in BOTH directories (inner join on filename).
    Sorted for reproducibility.
    """
    lr_map = {Path(p).name: p for p in collect_npy_paths(lr_dir)}
    gt_map = {Path(p).name: p for p in collect_npy_paths(gt_dir)}

    common = sorted(lr_map.keys() & gt_map.keys())
    if not common:
        raise FileNotFoundError(
            f"No matching filenames between {lr_dir} and {gt_dir}"
        )

    lr_paths = [lr_map[n] for n in common]
    gt_paths = [gt_map[n] for n in common]

    print(f"[build_paired_paths] {len(common)} matched pairs "
          f"(LR-only: {len(lr_map)-len(common)}, "
          f"GT-only: {len(gt_map)-len(common)})")
    return lr_paths, gt_paths


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config dict.

    Reads data layout from config['data'] and applies stratified train/val
    split on the paired dataset.

    Returns:
        (train_loader, val_loader)
    """
    d_cfg = config.get("data",     {})
    t_cfg = config.get("training", {})

    lr_dir     = d_cfg.get("lr_dir",      "data/raw/NoisyLR")
    gt_dir     = d_cfg.get("gt_dir",      "data/raw/GT")
    crop_size  = d_cfg.get("crop_size",   128)
    val_ratio  = d_cfg.get("val_ratio",   0.15)
    num_workers= d_cfg.get("num_workers", 4)
    batch_size = t_cfg.get("batch_size",  8)

    # Build intersection-matched pairs
    lr_paths, gt_paths = build_paired_paths(lr_dir, gt_dir)

    # Stratified split (by parent dir → filename index range)
    rng = random.Random(42)
    combined = list(zip(lr_paths, gt_paths))
    rng.shuffle(combined)
    n_val = max(1, int(len(combined) * val_ratio))
    val_pairs   = combined[:n_val]
    train_pairs = combined[n_val:]

    val_lr,   val_gt   = zip(*val_pairs)
    train_lr, train_gt = zip(*train_pairs)

    print(f"[DataLoader] train={len(train_lr)}, val={len(val_lr)}")

    train_ds = PairedNpyDataset(
        list(train_lr), list(train_gt),
        crop_size=crop_size, augment=True, is_val=False,
    )
    val_ds = PairedNpyDataset(
        list(val_lr), list(val_gt),
        crop_size=crop_size, augment=False, is_val=True,
    )

    use_cuda   = torch.cuda.is_available()
    pin_mem    = use_cuda
    # On CPU, multiprocessing workers can cause overhead; default to 0
    eff_workers = num_workers if use_cuda else 0

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = eff_workers,
        pin_memory  = pin_mem,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 1,
        shuffle     = False,
        num_workers = eff_workers,
        pin_memory  = pin_mem,
        drop_last   = False,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lr_paths, gt_paths = build_paired_paths("data/raw/NoisyLR", "data/raw/GT")
    print(f"Total pairs: {len(lr_paths)}")

    ds = PairedNpyDataset(lr_paths, gt_paths, crop_size=128, augment=True)
    sample = ds[0]
    lr_t = sample["degraded"]
    gt_t = sample["clean"]

    print(f"LR tensor: shape={lr_t.shape}, dtype={lr_t.dtype}, "
          f"min={lr_t.min():.4f}, max={lr_t.max():.4f}")
    print(f"GT tensor: shape={gt_t.shape}, dtype={gt_t.dtype}, "
          f"min={gt_t.min():.4f}, max={gt_t.max():.4f}")
    print("Dataset smoke test PASSED")
