"""
Synthetic Degradation Pipeline for Semiconductor Inspection Images
==================================================================

Produces physically-motivated degraded images from clean ground-truth inputs.
Used to augment the provided dataset and enable training on clean-only collections.

Degradation modes (applied randomly per sample):
  1. Speckle noise  — multiplicative, Gamma-distributed (NOT naive additive Gaussian)
  2. Resolution loss — Gaussian anti-aliasing + integer downsampling (not just resize)
  3. Both combined  — either order (noise→downsample OR downsample→noise)

Physical Motivation for Speckle Model:
  Real scanning electron microscope (SEM) speckle is coherent-imaging noise.
  Its amplitude is modeled as Rayleigh-distributed (intensity = Gamma-distributed).
  For a Gamma(shape=L, scale=1/L) distribution:
    - mean = 1 (no DC offset)
    - variance = 1/L (lower L → higher noise)
  This is equivalent to averaging L independent speckle realizations.
  We implement this as: I_noisy = I_clean * Gamma(L, 1/L)
  where L is randomly sampled per image to cover mild to severe speckle.

DO NOT use: I_noisy = I_clean + sigma * randn()  ← additive Gaussian, physically wrong.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Core Speckle Noise (Gamma multiplicative)
# ---------------------------------------------------------------------------

def add_speckle_noise(
    img: torch.Tensor,
    L: Optional[float] = None,
    L_range: Tuple[float, float] = (1.0, 10.0),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Add physically-motivated multiplicative speckle noise.

    Model: I_noisy = I_clean × η,  where η ~ Gamma(L, 1/L)
    This gives E[η] = 1 (no DC bias) and Var[η] = 1/L.

    Lower L → heavier speckle (SEM-like):
      L=1  → fully developed speckle (most severe, Rayleigh envelope)
      L=4  → moderate (4-look averaging)
      L=10 → mild residual speckle

    Args:
        img     : Tensor (C, H, W) or (B, C, H, W), values in [0, 1]
        L       : Exact number of looks. If None, sampled from L_range.
        L_range : (min_L, max_L) for random sampling.
        seed    : RNG seed for reproducibility.

    Returns:
        Noisy tensor, same shape as input.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    if L is None:
        L = rng.uniform(*L_range)

    # Gamma distribution: shape=L, scale=1/L → E[η]=1, Var[η]=1/L
    noise = rng.gamma(shape=L, scale=1.0 / L, size=img.shape)
    noise_t = torch.from_numpy(noise).float().to(img.device)

    # Multiplicative application — speckle modulates the SIGNAL, not adds to it
    noisy = img * noise_t

    # Clip: speckle can push values above 1 in bright regions
    # We clip to [0, 1] to simulate sensor saturation / quantization
    return noisy.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Spatial Resolution Reduction
# ---------------------------------------------------------------------------

def reduce_resolution(
    img: torch.Tensor,
    scale_factor: int = 2,
    anti_alias: bool = True,
) -> torch.Tensor:
    """
    Downsample image by scale_factor with proper anti-aliasing.

    Uses Gaussian blur before downsampling to prevent aliasing artifacts
    from naive subsampling. This matches the physics of actual SEM resolution
    reduction (optical system MTF roll-off before sampling).

    Args:
        img         : Tensor (C, H, W) or (B, C, H, W), values in [0, 1]
        scale_factor: Downsampling factor (2 → 512→256 or 256→128)
        anti_alias  : Apply Gaussian pre-blur (default True)

    Returns:
        Downsampled tensor with spatial dims divided by scale_factor.
    """
    # Ensure 4D input for F operations
    squeeze = (img.dim() == 3)
    if squeeze:
        img = img.unsqueeze(0)  # (1, C, H, W)

    if anti_alias:
        # Gaussian kernel for anti-aliasing
        # Sigma chosen based on Nyquist: sigma ≈ scale_factor / 2
        sigma = scale_factor / 2.0
        k_size = int(6 * sigma + 1) | 1  # make odd
        img = _gaussian_blur(img, k_size, sigma)

    # Downsample using area averaging (best for natural images; respects local means)
    downsampled = F.interpolate(
        img,
        scale_factor=1.0 / scale_factor,
        mode='area',
    )

    if squeeze:
        downsampled = downsampled.squeeze(0)

    return downsampled.clamp(0.0, 1.0)


def _gaussian_blur(x: torch.Tensor, k_size: int, sigma: float) -> torch.Tensor:
    """Apply 2D Gaussian blur to a (B, C, H, W) tensor."""
    # Build 1D Gaussian kernel
    coords = torch.arange(k_size, dtype=torch.float32, device=x.device) - k_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    # Outer product → 2D kernel, shape (1, 1, k, k)
    kernel_2d = g.outer(g).unsqueeze(0).unsqueeze(0)

    C = x.shape[1]
    # Expand for all channels (depthwise)
    kernel_2d = kernel_2d.expand(C, 1, k_size, k_size)

    pad = k_size // 2
    return F.conv2d(x, kernel_2d, padding=pad, groups=C)


# ---------------------------------------------------------------------------
# Combined Degradation Pipeline (randomized order)
# ---------------------------------------------------------------------------

def degrade_image(
    img: torch.Tensor,
    apply_noise: bool = True,
    apply_downsample: bool = True,
    scale_factor: int = 2,
    L_range: Tuple[float, float] = (1.0, 10.0),
    randomize_order: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply combined degradation with randomized application order.

    The ground-truth order of degradations in real data is unknown.
    Randomizing order during training makes the model robust to both:
      - noise → downsample  (noise in high-res, then compressed)
      - downsample → noise  (low-res sensor, then noisy readout)

    Args:
        img             : Clean image tensor (C, H, W), values in [0, 1]
        apply_noise     : Whether to add speckle
        apply_downsample: Whether to reduce resolution
        scale_factor    : Downsampling factor (default 2)
        L_range         : (min, max) for random speckle L parameter
        randomize_order : If True, randomly choose application order
        seed            : RNG seed

    Returns:
        Degraded tensor.
    """
    rng = random.Random(seed)

    ops = []
    if apply_noise:
        ops.append("noise")
    if apply_downsample:
        ops.append("downsample")

    # Randomize application order
    if randomize_order and len(ops) > 1:
        rng.shuffle(ops)

    result = img.clone()
    for op in ops:
        if op == "noise":
            result = add_speckle_noise(result, L_range=L_range, seed=seed)
        elif op == "downsample":
            result = reduce_resolution(result, scale_factor=scale_factor)

    return result


# ---------------------------------------------------------------------------
# Randomized Augmentation Factory (used by Dataset)
# ---------------------------------------------------------------------------

class RandomDegradation:
    """
    Callable that applies random degradation to a clean image tensor.

    Randomly selects one of three scenarios per call:
      - Noise only
      - Downsample only
      - Noise + Downsample (random order)

    Args:
        scale_factor     : Resolution reduction factor
        L_range          : Speckle severity range (lower L → heavier)
        noise_prob       : Probability of applying noise
        downsample_prob  : Probability of applying downsampling
        randomize_order  : Randomize noise/downsample order

    Usage:
        deg = RandomDegradation()
        noisy_img = deg(clean_img)
    """
    def __init__(
        self,
        scale_factor: int = 2,
        L_range: Tuple[float, float] = (1.0, 10.0),
        noise_prob: float = 0.7,
        downsample_prob: float = 0.7,
        randomize_order: bool = True,
    ):
        self.scale_factor    = scale_factor
        self.L_range         = L_range
        self.noise_prob      = noise_prob
        self.downsample_prob = downsample_prob
        self.randomize_order = randomize_order

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply random degradation.

        Returns:
            (degraded_img, degradation_info_dict)
            where info contains: {apply_noise, apply_downsample, order}
        """
        apply_noise      = random.random() < self.noise_prob
        apply_downsample = random.random() < self.downsample_prob

        # At least one degradation must be applied (otherwise it's a trivial identity)
        if not apply_noise and not apply_downsample:
            apply_noise = True  # default to noise if both were rejected

        degraded = degrade_image(
            img,
            apply_noise      = apply_noise,
            apply_downsample = apply_downsample,
            scale_factor     = self.scale_factor,
            L_range          = self.L_range,
            randomize_order  = self.randomize_order,
        )

        info = {
            "apply_noise":      apply_noise,
            "apply_downsample": apply_downsample,
        }
        return degraded, info


# ---------------------------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torchvision.transforms.functional as TF
    from PIL import Image, ImageDraw

    # Create a synthetic "clean" test image (gradient + edge)
    img_np = np.zeros((256, 256), dtype=np.float32)
    img_np[:, 128:] = 0.7  # step edge (like a semiconductor line)
    img_np = img_np + 0.05 * np.random.randn(256, 256).clip(-1, 1)
    img_np = img_np.clip(0, 1)

    img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1, 256, 256)

    # Test noise only
    noisy = add_speckle_noise(img_t, L=2.0)
    print(f"Speckle (L=2):  min={noisy.min():.3f}, max={noisy.max():.3f}, "
          f"std={noisy.std():.3f}")

    # Test downsample only
    downsampled = reduce_resolution(img_t, scale_factor=2)
    print(f"Downsampled:    shape={downsampled.shape}")  # (1, 128, 128)

    # Test combined (random order)
    both = degrade_image(img_t, apply_noise=True, apply_downsample=True)
    print(f"Combined:       shape={both.shape}, min={both.min():.3f}, max={both.max():.3f}")

    # Test random augmentation factory
    deg_fn = RandomDegradation()
    for i in range(5):
        d_img, info = deg_fn(img_t)
        print(f"  Sample {i}: noise={info['apply_noise']}, "
              f"downsample={info['apply_downsample']}, shape={d_img.shape}")
