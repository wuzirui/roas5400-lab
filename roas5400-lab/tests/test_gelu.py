"""GELU test with fixed input/output snapshot."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_gelu


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "gelu_expected.npz"


def test_gelu_matches_expected():
    expected = np.load(EXPECTED_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]

    actual_output = run_gelu(in_features)

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)
