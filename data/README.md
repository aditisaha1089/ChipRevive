# Dataset Directory

## Actual Dataset Layout (as delivered)

```
data/
├── raw/
│   ├── NoisyLR/   ← 3200 × .npy, shape (128,128), float32, range ~[-0.05, 1.7]
│   └── GT/        ← 3200 × .npy, shape (256,256), float32, range [0.0, 1.0]
└── test/
    └── NoisyLR/   ← 400  × .npy, same format as raw/NoisyLR/ — NO GT
```

## Pairing Convention

Files are paired **by identical filename**:

| NoisyLR file | GT file | Relationship |
|---|---|---|
| `data/raw/NoisyLR/000040.npy` | `data/raw/GT/000040.npy` | exact match |

The dataset loader (`src/data/dataset.py`) builds the intersection of filenames
from both directories — no manual index file needed.

## Array Characteristics

| Split | Shape | dtype | Value range | Notes |
|---|---|---|---|---|
| NoisyLR (train) | (128, 128) | float32 | [-0.05, ~1.7] | Speckle noise; values outside [0,1] are physically valid |
| GT (train) | (256, 256) | float32 | [0.0, 1.0] | Clean, normalized |
| NoisyLR (test) | (128, 128) | float32 | similar to train | KLA holds GT for scoring |

## Why values exceed [0, 1] in NoisyLR?

Speckle noise is **multiplicative**: `I_noisy ≈ I_clean × η` where η is
Gamma-distributed. For bright regions, η > 1 pushes pixel values above 1.0.
Slight negatives (~−0.05) are sensor read-noise artifacts.

The model's `log_transform` handles both: `log(1 + clamp(x, min=0))`.

## Source

Dataset provided by KLA Corporation for the Semicon Hackathon.
Download link: https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6

Place the downloaded folders at exactly the paths shown above before training.
