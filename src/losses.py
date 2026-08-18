"""
Composite Loss Function for DA-JRN
====================================

Loss = w_charb * L_charbonnier
     + w_sobel * L_sobel
     + w_fft   * L_fft
     + w_ssim  * L_ssim

Each term serves a specific purpose:

1. Charbonnier Loss (robust L1):
   sqrt((y_pred - y_true)^2 + eps^2)
   - More robust than MSE to speckle outliers (heavy tails in speckle distributions)
   - Less smoothing than L2, preserves sharper edges
   - epsilon=1e-3 gives smooth gradient near zero

2. Sobel/Gradient Loss:
   L1 between Sobel-filtered prediction and ground truth
   - Semiconductor defects ARE edges — preserving them is critical
   - Forces the network to recover sharp boundaries, not just pixel averages
   - Sobel in both x and y directions, combined as gradient magnitude

3. FFT Frequency-Domain Loss:
   L1 on the magnitude spectrum: |FFT(pred)| vs |FFT(gt)|
   - Directly penalizes missing high-frequency content lost to downsampling
   - Complementary to spatial loss — you can have good pixel values but wrong
     texture frequency (over-smoothed). FFT loss explicitly punishes that.
   - Applied to log-magnitude spectrum for numerical stability

4. SSIM Loss (Structural Similarity):
   1 - SSIM(pred, gt)
   - Measures luminance, contrast, and structure simultaneously
   - More perceptually aligned than pixel-wise metrics
   - Complements pixel and frequency losses with structural fidelity term

All weights are configurable in config.yaml under the `loss` section.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Charbonnier Loss
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """
    Charbonnier (pseudo-Huber) loss: sqrt((x-y)^2 + eps^2).

    More robust than MSE to outliers (e.g., extreme speckle noise pixels).
    Behaves like L1 far from zero and like L2 near zero (smooth gradient).
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()


# ---------------------------------------------------------------------------
# Sobel Gradient Loss
# ---------------------------------------------------------------------------

class SobelGradientLoss(nn.Module):
    """
    Gradient magnitude loss using Sobel filters.

    Computes Sobel-x and Sobel-y for both pred and target,
    then applies Charbonnier loss on the combined gradient magnitude.

    Critical for semiconductor inspection: defects are edge-like features.
    Preserving edge sharpness is as important as overall pixel accuracy.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

        # Sobel kernels (3×3) — registered as buffers so they move with the model
        sobel_x = torch.tensor([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1],
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude via Sobel filtering."""
        # x: (B, 1, H, W) — single channel grayscale
        # AMP FIX: Cast x to match the Sobel buffer's device+dtype in one shot.
        # Without loss_fn.to(device), buffers stay on CPU; with it they're on CUDA.
        # Using .to(self.sobel_x) is device+dtype-safe in all cases.
        x = x.to(self.sobel_x)
        gx = F.conv2d(x, self.sobel_x, padding=1)  # (B, 1, H, W)
        gy = F.conv2d(x, self.sobel_y, padding=1)  # (B, 1, H, W)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + self.eps ** 2)
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure same spatial size (SR head may upsample pred)
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)

        grad_pred   = self.gradient_magnitude(pred)
        grad_target = self.gradient_magnitude(target)

        diff = grad_pred - grad_target
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()


# ---------------------------------------------------------------------------
# FFT Frequency-Domain Loss
# ---------------------------------------------------------------------------

class FFTLoss(nn.Module):
    """
    Frequency-domain loss on the log-magnitude spectrum.

    Motivation: Spatial loss (L1/L2/SSIM) can be low even when the model
    produces over-smoothed outputs that lack high-frequency texture.
    FFT loss directly penalizes mismatches in the frequency spectrum,
    forcing recovery of fine-grained detail lost to downsampling.

    Uses log(1 + |FFT|) for numerical stability (dynamic range compression).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # CRITICAL AMP FIX: torch.fft.fft2 does NOT support float16 on CUDA.
        # Under torch.cuda.amp.autocast the inputs may be float16, causing:
        #   RuntimeError: "fft_cuda" not implemented for 'Half'
        # Cast to float32 before FFT (float32 is always safe for fft2).
        pred   = pred.float()
        target = target.float()

        # Compute 2D FFT for both images
        # torch.fft.fft2 returns complex tensor; we use magnitude spectrum
        fft_pred   = torch.fft.fft2(pred,   norm="ortho")
        fft_target = torch.fft.fft2(target, norm="ortho")

        # Log-magnitude spectra (compress dynamic range)
        mag_pred   = torch.log1p(torch.abs(fft_pred))
        mag_target = torch.log1p(torch.abs(fft_target))

        return F.l1_loss(mag_pred, mag_target)


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss: 1 - SSIM(pred, target).

    SSIM measures:
      - Luminance: mean intensity match
      - Contrast:  variance match
      - Structure: covariance / normalized cross-correlation

    Using a Gaussian window for local statistics (window_size=11 is standard).
    This gives a perceptually meaningful loss beyond pixel-wise comparison.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2

        # Create Gaussian kernel
        kernel = self._gaussian_kernel(window_size, sigma)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel_2d = g.outer(g)           # (size, size)
        return kernel_2d.view(1, 1, size, size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)

        # AMP FIX: Cast pred/target to match the Gaussian kernel buffer's device+dtype.
        # (Same pattern as SobelGradientLoss — buffers stay float32, input may be half.)
        pred   = pred.to(self.kernel)
        target = target.to(self.kernel)

        pad = self.window_size // 2
        kernel = self.kernel  # already a buffer

        # Local means
        mu_p = F.conv2d(pred,   kernel, padding=pad, groups=1)
        mu_t = F.conv2d(target, kernel, padding=pad, groups=1)

        mu_p2  = mu_p ** 2
        mu_t2  = mu_t ** 2
        mu_pt  = mu_p * mu_t

        # Local variances and covariance
        sigma_p2  = F.conv2d(pred   ** 2, kernel, padding=pad) - mu_p2
        sigma_t2  = F.conv2d(target ** 2, kernel, padding=pad) - mu_t2
        sigma_pt  = F.conv2d(pred * target, kernel, padding=pad) - mu_pt

        # SSIM formula
        numerator   = (2 * mu_pt  + self.C1) * (2 * sigma_pt  + self.C2)
        denominator = (mu_p2 + mu_t2 + self.C1) * (sigma_p2 + sigma_t2 + self.C2)

        ssim_map = numerator / denominator.clamp(min=1e-8)
        ssim_val = ssim_map.mean()

        return 1.0 - ssim_val  # Loss = 1 - SSIM ∈ [0, 2]


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """
    Composite loss combining all four terms with configurable weights.

    Default weights (empirically tuned for semiconductor imagery):
      w_charb = 1.0   — primary reconstruction fidelity
      w_sobel = 0.5   — edge preservation (high weight since defects are edges)
      w_fft   = 0.1   — frequency fidelity (scaled down; magnitude spectrum large)
      w_ssim  = 0.3   — structural fidelity

    All weights are configurable via config.yaml under `loss:` section.
    """
    def __init__(
        self,
        w_charb: float = 1.0,
        w_sobel: float = 0.5,
        w_fft:   float = 0.1,
        w_ssim:  float = 0.3,
        charb_eps: float = 1e-3,
    ):
        super().__init__()
        self.w_charb = w_charb
        self.w_sobel = w_sobel
        self.w_fft   = w_fft
        self.w_ssim  = w_ssim

        self.charb_loss = CharbonnierLoss(eps=charb_eps)
        self.sobel_loss = SobelGradientLoss(eps=charb_eps)
        self.fft_loss   = FFTLoss()
        self.ssim_loss  = SSIMLoss()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Compute composite loss.

        Args:
            pred:              Model output, (B, 1, H_out, W_out)
            target:            Ground truth clean image, (B, 1, H_gt, W_gt)
            return_components: If True, return dict with individual losses

        Returns:
            total_loss (scalar tensor), or (total_loss, components_dict) if
            return_components=True
        """
        l_charb = self.charb_loss(pred, target)
        l_sobel = self.sobel_loss(pred, target)
        l_fft   = self.fft_loss(pred, target)
        l_ssim  = self.ssim_loss(pred, target)

        total = (
            self.w_charb * l_charb +
            self.w_sobel * l_sobel +
            self.w_fft   * l_fft   +
            self.w_ssim  * l_ssim
        )

        if return_components:
            return total, {
                "charb": l_charb.item(),
                "sobel": l_sobel.item(),
                "fft":   l_fft.item(),
                "ssim":  l_ssim.item(),
                "total": total.item(),
            }
        return total


def build_loss(config: dict) -> CompositeLoss:
    """
    Instantiate CompositeLoss from config dict.

    Expected config structure:
        loss:
          w_charb: 1.0
          w_sobel: 0.5
          w_fft: 0.1
          w_ssim: 0.3
          charb_eps: 0.001
    """
    l_cfg = config.get("loss", {})
    return CompositeLoss(
        w_charb   = l_cfg.get("w_charb",   1.0),
        w_sobel   = l_cfg.get("w_sobel",   0.5),
        w_fft     = l_cfg.get("w_fft",     0.1),
        w_ssim    = l_cfg.get("w_ssim",    0.3),
        charb_eps = l_cfg.get("charb_eps", 1e-3),
    )


if __name__ == "__main__":
    # Smoke test
    loss_fn = CompositeLoss()
    pred   = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    total, components = loss_fn(pred, target, return_components=True)
    print(f"Total loss: {total.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
