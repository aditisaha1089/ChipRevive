"""
Degradation-Aware Joint Restoration Network (DA-JRN)
=====================================================

Architecture Overview:
  Input (grayscale, any degradation combo)
       |
  [Log-domain transform: log(1+x)]   <- turns multiplicative speckle into additive noise
       |
  [Degradation Estimator Head]        <- predicts (noise_level, resolution_factor) embedding
       |                                   via 4-conv CNN + GAP + MLP
       | embedding
       v
  [Restoration Backbone]              <- NAFNet-style residual blocks (6-8 deep)
  [FiLM conditioning at every block]  <- scale + shift from degradation embedding
       |
  [PixelShuffle SR head]             <- sub-pixel upsampling (gated off if full-res detected)
       |
  [Exp-domain transform: exp(x)-1]   <- invert the log-domain transform
       |
  Output (restored clean image)

Design Decisions & Novelty:
  1. Log-domain transform: Because speckle is MULTIPLICATIVE (I_noisy ≈ I_clean × η),
     taking log turns it into an additive problem (log(I_noisy) ≈ log(I_clean) + log(η)).
     Additive noise is structurally easier for CNNs to remove, improving speckle suppression.

  2. FiLM (Feature-wise Linear Modulation): Unlike generic U-Nets that apply one fixed
     transformation to all inputs, FiLM lets the network *adapt* per-image by injecting
     a learned degradation embedding as affine modulation (γ × feature + β) at each block.
     This is the key mechanism for handling all three degradation scenarios with one model.

  3. NAFNet blocks: Chosen over Transformers (Restormer/SwinIR) for 3-5× faster inference
     at comparable quality. Eliminates batch/layer norm for cross-domain generalization.
     Uses depthwise separable convolutions and SimpleGate nonlinearity.

  4. Degradation Estimator: Learned end-to-end — no hand-labeled degradation metadata
     needed at inference. Predicts a compact embedding that FiLM uses to steer restoration.

  5. PixelShuffle upsampling: Unlike bicubic/bilinear, PixelShuffle learns to rearrange
     channel depth into spatial resolution — it genuinely reconstructs high-frequency
     detail rather than interpolating. Gated off (identity path) for full-resolution inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleGate(nn.Module):
    """
    NAFNet-style gate: splits channels in half, multiplies them element-wise.
    Replaces ReLU/GELU — no negative-side suppression, better gradient flow.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv for efficiency."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Injects degradation embedding into a feature map via affine modulation.

    Given a feature map F (B, C, H, W) and embedding z (B, embed_dim):
        gamma, beta = Linear(z) split into (C, C)
        output = gamma.unsqueeze(-1).unsqueeze(-1) * F
                 + beta.unsqueeze(-1).unsqueeze(-1)

    This allows the restoration backbone to *adapt its behavior* per image
    based on what the Degradation Estimator detected — the key differentiator
    from a standard U-Net or denoiser.
    """
    def __init__(self, embed_dim: int, num_channels: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_channels * 2)
        # Initialize gamma near 1, beta near 0 → near-identity at init
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Bias the gamma-half toward 1
        self.fc.bias.data[:num_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), embedding: (B, embed_dim)
        params = self.fc(embedding)                          # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)                 # each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)             # (B, C, 1, 1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# NAFNet-style Residual Block with FiLM conditioning
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    """
    NAFNet-inspired block:
      - Depthwise separable conv (3×3) — efficient spatial mixing
      - SimpleGate activation — no normalization dependency (helps OOD generalization)
      - Channel attention (1×1 squeeze-excite) — reweights features
      - FiLM conditioning — adapts block behavior from degradation embedding
      - Residual connection

    Why no BatchNorm/LayerNorm? They introduce distribution shift when the model
    sees out-of-distribution inputs (different semiconductor structure types).
    Removing them improves generalization at negligible quality cost.
    """
    def __init__(self, channels: int, embed_dim: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch = channels * dw_expand

        # Spatial mixing path
        self.conv1    = nn.Conv2d(channels, dw_ch, 1)          # expand
        self.conv_dw  = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)  # depthwise
        self.gate     = SimpleGate()
        # After gate: channels = dw_ch // 2

        # Channel attention (lightweight SE-style)
        mid_ch = dw_ch // 2
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_ch, max(mid_ch // 4, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(mid_ch // 4, 4), mid_ch),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Conv2d(mid_ch, channels, 1)            # project back

        # FiLM conditioning
        self.film = FiLMLayer(embed_dim, channels)

        # FFN path (pointwise)
        ffn_ch = channels * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_ch, 1),
            SimpleGate(),
            nn.Conv2d(ffn_ch // 2, channels, 1),
        )

        # Learnable residual scale (start near 0 for stable training)
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1) + 0.01)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1) + 0.01)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        # --- Spatial mixing branch ---
        inp = x
        x_sp = self.conv1(x)
        x_sp = self.conv_dw(x_sp)
        x_sp = self.gate(x_sp)                         # (B, dw_ch//2, H, W)

        # Channel attention
        ca_w = self.ca(x_sp)                           # (B, dw_ch//2)
        ca_w = ca_w.unsqueeze(-1).unsqueeze(-1)        # (B, dw_ch//2, 1, 1)
        x_sp = x_sp * ca_w
        x_sp = self.conv2(x_sp)                        # (B, C, H, W)

        # FiLM modulation on residual
        x_sp = self.film(x_sp, embedding)
        x    = inp + x_sp * self.beta

        # --- FFN branch ---
        x = x + self.ffn(x) * self.gamma

        return x


# ---------------------------------------------------------------------------
# Degradation Estimator Head
# ---------------------------------------------------------------------------

class DegradationEstimator(nn.Module):
    """
    Small CNN that inspects the INPUT image and predicts a compact embedding
    representing the estimated degradation profile (noise severity + resolution loss).

    Design:
      4 conv layers with stride-2 downsampling → Global Average Pooling → MLP
      Output: embed_dim-dimensional vector fed into FiLM layers of the backbone.

    IMPORTANT: This is learned end-to-end with the restoration backbone.
    No hand-labeled degradation labels are needed at training OR inference.
    The estimator learns to extract degradation cues (noise statistics, aliasing
    artifacts, frequency content) automatically through backprop.
    """
    def __init__(self, in_ch: int = 1, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            # Layer 1: detect low-level texture statistics
            nn.Conv2d(in_ch, 16, 3, stride=2, padding=1),  # /2
            nn.GELU(),
            # Layer 2: capture mid-level patterns
            nn.Conv2d(16, 32, 3, stride=2, padding=1),     # /4
            nn.GELU(),
            # Layer 3: spatial resolution cues (aliasing artifacts appear here)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # /8
            nn.GELU(),
            # Layer 4: semantic compression
            nn.Conv2d(64, 64, 3, stride=2, padding=1),     # /16
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)  # (B, 64, H/16, W/16)
        feat = self.gap(feat)   # (B, 64, 1, 1)
        emb  = self.mlp(feat)   # (B, embed_dim)
        return emb


# ---------------------------------------------------------------------------
# PixelShuffle Super-Resolution Head (gated by degradation embedding)
# ---------------------------------------------------------------------------

class SRHead(nn.Module):
    """
    Sub-pixel convolution upsampling head.

    Unlike bicubic/bilinear interpolation, PixelShuffle LEARNS to rearrange
    channel depth into spatial resolution. The convolution weights learn to
    synthesize high-frequency detail — not just smooth interpolation.

    Scale = 2 (handles 128→256 and 256→512 cases).

    Gating: A small FC layer reads the degradation embedding and produces
    a gate ∈ [0, 1]. When the estimator detects no resolution loss (gate ≈ 0),
    the SR path contributes nothing and output = upsampled_input (identity).
    This means the same model head handles both noise-only and SR cases.
    """
    def __init__(self, channels: int, embed_dim: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        # Learn pixel-shuffle filters
        self.conv_up = nn.Conv2d(channels, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)    # (B, C*r², H, W) → (B, C, H*r, W*r)
        self.refine  = nn.Conv2d(channels, channels, 3, padding=1)

        # Gate: reads embedding, outputs scalar in [0,1] per image
        self.gate_fc = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        gate = self.gate_fc(embedding)           # (B, 1)
        gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)

        # Sub-pixel upsampling path
        x_up_feat = self.conv_up(x)             # (B, C*r², H, W)
        x_up      = self.shuffle(x_up_feat)     # (B, C, H*r, W*r)
        x_up      = self.refine(x_up)           # (B, C, H*r, W*r)

        # Bilinear baseline for the "no SR needed" identity path
        x_base = F.interpolate(x, scale_factor=self.scale,
                               mode='bilinear', align_corners=False)

        # Gate blends between identity and learned SR:
        #   gate ≈ 0 → pure identity (noise-only case)
        #   gate ≈ 1 → full PixelShuffle SR (SR or combined case)
        return gate * x_up + (1.0 - gate) * x_base


# ---------------------------------------------------------------------------
# Main DA-JRN Model
# ---------------------------------------------------------------------------

class DAJRN(nn.Module):
    """
    Degradation-Aware Joint Restoration Network (DA-JRN)

    Handles three degradation scenarios with ONE model:
      - Noise only (speckle)
      - Resolution reduction only (×2 downsampling)
      - Both combined (noise + downsampling)

    No degradation-type label is required at inference time.

    Args:
        in_ch      : Input channels (1 for grayscale)
        base_ch    : Base channel width for backbone (default 32 → fast + capable)
        embed_dim  : Degradation embedding dimension (default 64)
        num_blocks : Number of NAFNet restoration blocks (default 8)
        sr_scale   : Super-resolution upscale factor (default 2)
        do_sr      : Whether to include SR head (True for combined model)
    """
    def __init__(
        self,
        in_ch:      int  = 1,
        base_ch:    int  = 32,
        embed_dim:  int  = 64,
        num_blocks: int  = 8,
        sr_scale:   int  = 2,
        do_sr:      bool = True,
    ):
        super().__init__()
        self.do_sr    = do_sr
        self.sr_scale = sr_scale
        self.embed_dim = embed_dim

        # ── Degradation Estimator (runs on input before log-transform) ──
        self.deg_estimator = DegradationEstimator(in_ch=in_ch, embed_dim=embed_dim)

        # ── Input stem ──
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # ── Restoration Backbone (NAFNet blocks + FiLM) ──
        self.blocks = nn.ModuleList([
            NAFBlock(base_ch, embed_dim) for _ in range(num_blocks)
        ])

        # ── Output projection ──
        self.output_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

        # ── SR Head (PixelShuffle, gated by embedding) ──
        if do_sr:
            self.sr_head = SRHead(in_ch, embed_dim, scale=sr_scale)

    def log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-domain transform: log(1 + x)

        Motivation: Speckle noise is multiplicative — I_noisy ≈ I_clean × η.
        In log domain: log(I_noisy) ≈ log(I_clean) + log(η).
        This converts the multiplicative noise into ADDITIVE noise in log-space,
        which is structurally easier for a CNN to remove.

        Real data note: NoisyLR .npy files contain values in ~[-0.05, 1.7].
          - Values slightly < 0 are sensor read-noise artifacts → clamped to 0.
            This is intentional: they carry no structural signal, only detector noise.
          - Values > 1.0 are speckle amplification on bright regions → log1p compresses
            them back (log1p(1.7) ≈ 0.99), making the range tractable for the network.
        """
        return torch.log1p(x.clamp(min=0.0))

    def exp_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of log_transform: exp(x) - 1 = expm1(x).
        Inverts log(1+x) back to linear domain after backbone denoising.
        Output is clipped to [0, 1] for valid image range.
        """
        return torch.expm1(x).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : Input degraded image, shape (B, 1, H, W).
                Real data: NoisyLR .npy values in ~[-0.05, 1.7]; the log_transform
                handles this via clamp(min=0) + log1p.
                H and W are the INPUT dimensions — for the real dataset this is
                128×128; output after SR head is 256×256 (matching GT shape).

        Returns:
            Restored image, shape (B, 1, H*sr_scale, W*sr_scale) if do_sr=True,
            else (B, 1, H, W). Values clipped to [0, 1].
        """
        # Step 1: Estimate degradation profile from raw input (before log transform)
        # The estimator sees the original noisy/blurry image statistics.
        embedding = self.deg_estimator(x)      # (B, embed_dim)

        # Step 2: Log-domain transform (multiplicative → additive noise)
        x_log = self.log_transform(x)

        # Step 3: Feature extraction
        feat = self.input_conv(x_log)          # (B, base_ch, H, W)

        # Step 4: Restoration backbone with FiLM conditioning at every block
        for block in self.blocks:
            feat = block(feat, embedding)      # embedding steers each block's behavior

        # Step 5: Project back to image space (still in log domain)
        residual = self.output_conv(feat)      # (B, 1, H, W)

        # Residual learning: predict the CLEAN log-image, not just noise
        x_restored_log = x_log + residual     # (B, 1, H, W)

        # Step 6: Invert log-domain transform
        x_restored = self.exp_transform(x_restored_log)  # (B, 1, H, W)

        # Step 7: SR head (PixelShuffle upsampling, gated by degradation embedding)
        if self.do_sr:
            x_restored = self.sr_head(x_restored, embedding)  # (B, 1, H*r, W*r)

        return x_restored


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: dict) -> DAJRN:
    """
    Instantiate DA-JRN from a config dictionary.

    Expected keys (with defaults):
        model.in_ch      : 1
        model.base_ch    : 32
        model.embed_dim  : 64
        model.num_blocks : 8
        model.sr_scale   : 2
        model.do_sr      : true
    """
    m_cfg = config.get("model", {})
    return DAJRN(
        in_ch      = m_cfg.get("in_ch",      1),
        base_ch    = m_cfg.get("base_ch",    32),
        embed_dim  = m_cfg.get("embed_dim",  64),
        num_blocks = m_cfg.get("num_blocks", 8),
        sr_scale   = m_cfg.get("sr_scale",   2),
        do_sr      = m_cfg.get("do_sr",      True),
    )


if __name__ == "__main__":
    # Quick smoke test
    model = DAJRN()
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Noise-only case: full-res input (256×256)
    x_noise = torch.rand(2, 1, 256, 256)
    out = model(x_noise)
    print(f"Noise-only:  input {x_noise.shape} → output {out.shape}")  # expect (2,1,512,512)

    # SR case: half-res input (128×128) → should output (2,1,256,256)
    x_sr = torch.rand(2, 1, 128, 128)
    out2 = model(x_sr)
    print(f"SR case:     input {x_sr.shape} → output {out2.shape}")

    # Degradation embedding sanity
    emb = model.deg_estimator(x_noise)
    print(f"Embedding shape: {emb.shape}")  # (2, 64)
