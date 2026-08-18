"""
tests/test_evaluate.py — End-to-end test for evaluate.py

Generates dummy degraded .npy files in a temp directory, runs evaluate.py
via subprocess, and asserts the output directory is populated correctly.

Run with:
    python -m pytest tests/test_evaluate.py -v
    # or
    python tests/test_evaluate.py
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Make sure project root is on path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_npy_dir(directory: str, n: int = 5, shape=(128, 128)) -> list:
    """Create n dummy .npy files mimicking NoisyLR format."""
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(n):
        arr = np.random.rand(*shape).astype(np.float32)
        # Simulate real NoisyLR: values slightly outside [0,1]
        arr = arr * 1.4 - 0.05
        fname = os.path.join(directory, f"{i:06d}.npy")
        np.save(fname, arr)
        paths.append(fname)
    return paths


def make_dummy_weights(weights_path: str, cfg: dict) -> None:
    """Save an untrained (random-init) model checkpoint for testing."""
    import torch
    from src.model import build_model
    model = build_model(cfg)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "metrics": {}}, weights_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEvaluateScript:
    """Tests that evaluate.py runs end-to-end without manual intervention."""

    def setup_method(self):
        """Set up temp directories and dummy weights before each test."""
        self.tmp      = tempfile.mkdtemp()
        self.input_dir  = os.path.join(self.tmp, "input")
        self.output_dir = os.path.join(self.tmp, "output")
        self.weights_dir= os.path.join(self.tmp, "weights")
        self.weights_path = os.path.join(self.weights_dir, "best_model.pth")

        # Write dummy config
        import yaml
        self.cfg = {
            "model": {
                "in_ch": 1, "base_ch": 16, "embed_dim": 32,
                "num_blocks": 2, "sr_scale": 2, "do_sr": True,
            },
            "inference": {
                "weights_path": self.weights_path,
                "pad_multiple": 16,
            },
        }
        self.config_path = os.path.join(self.tmp, "config.yaml")
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.cfg, f)

        # Dummy inputs and weights
        self.input_files = make_dummy_npy_dir(self.input_dir, n=5)
        make_dummy_weights(self.weights_path, self.cfg)

    def test_output_dir_populated(self):
        """evaluate.py must write one .npy per input file."""
        result = subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--input_dir",  self.input_dir,
                "--output_dir", self.output_dir,
                "--weights",    self.weights_path,
                "--config",     self.config_path,
                "--device",     "cpu",
            ],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            f"evaluate.py exited with code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        output_files = list(Path(self.output_dir).glob("*.npy"))
        assert len(output_files) == 5, (
            f"Expected 5 output files, got {len(output_files)}\n"
            f"STDOUT:\n{result.stdout}"
        )

    def test_output_filenames_match_input(self):
        """Output filenames must match input filenames (same stem)."""
        subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--input_dir",  self.input_dir,
                "--output_dir", self.output_dir,
                "--weights",    self.weights_path,
                "--config",     self.config_path,
                "--device",     "cpu",
            ],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
            check=True,
        )
        input_stems  = {Path(p).stem for p in self.input_files}
        output_stems = {f.stem for f in Path(self.output_dir).glob("*.npy")}
        assert input_stems == output_stems, (
            f"Filename mismatch.\nInput:  {sorted(input_stems)}\n"
            f"Output: {sorted(output_stems)}"
        )

    def test_output_shape_and_range(self):
        """Restored .npy must be 2× spatially larger than input and in [0,1]."""
        subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--input_dir",  self.input_dir,
                "--output_dir", self.output_dir,
                "--weights",    self.weights_path,
                "--config",     self.config_path,
                "--device",     "cpu",
            ],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
            check=True,
        )
        for out_f in Path(self.output_dir).glob("*.npy"):
            out_arr = np.load(str(out_f))
            assert out_arr.ndim == 2, f"Expected 2D array, got shape {out_arr.shape}"
            assert out_arr.shape == (256, 256), (
                f"Expected (256,256), got {out_arr.shape} in {out_f.name}"
            )
            assert float(out_arr.min()) >= 0.0, f"Output has negatives in {out_f.name}"
            assert float(out_arr.max()) <= 1.0, f"Output exceeds 1.0 in {out_f.name}"

    def test_missing_weights_exits_nonzero(self):
        """evaluate.py must exit with code 1 if weights file not found."""
        result = subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--input_dir",  self.input_dir,
                "--output_dir", self.output_dir,
                "--weights",    "/nonexistent/path/model.pth",
                "--config",     self.config_path,
                "--device",     "cpu",
            ],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode != 0, (
            "Expected non-zero exit when weights are missing"
        )

    def test_png_output_format(self):
        """evaluate.py must produce .png files when --output_format png."""
        result = subprocess.run(
            [
                sys.executable, "evaluate.py",
                "--input_dir",    self.input_dir,
                "--output_dir",   self.output_dir,
                "--weights",      self.weights_path,
                "--config",       self.config_path,
                "--device",       "cpu",
                "--output_format","png",
            ],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        output_files = list(Path(self.output_dir).glob("*.png"))
        assert len(output_files) == 5, f"Expected 5 .png files, got {len(output_files)}"


# ---------------------------------------------------------------------------
# Direct run (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = TestEvaluateScript()
    t.setup_method()
    print("Running: test_output_dir_populated ...")
    t.test_output_dir_populated()
    print("  PASS")

    t.setup_method()
    print("Running: test_output_filenames_match_input ...")
    t.test_output_filenames_match_input()
    print("  PASS")

    t.setup_method()
    print("Running: test_output_shape_and_range ...")
    t.test_output_shape_and_range()
    print("  PASS")

    t.setup_method()
    print("Running: test_missing_weights_exits_nonzero ...")
    t.test_missing_weights_exits_nonzero()
    print("  PASS")

    t.setup_method()
    print("Running: test_png_output_format ...")
    t.test_png_output_format()
    print("  PASS")

    print("\nAll tests passed.")
