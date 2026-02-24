"""Softmax test with deterministic input/output snapshot."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_softmax


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "softmax_expected.npz"


def test_softmax_matches_expected():
    expected = np.load(EXPECTED_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    dim = int(expected["dim"].item())
    expected_output = expected["output"]

    actual_output = run_softmax(in_features, dim=dim)
    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)


def test_softmax_numerical_stability_hard_case():
    # High dynamic range values that will overflow with naive exp(x)/sum(exp(x)).
    x = torch.tensor(
        [
            [[10000.0, 10001.0, 9999.0], [-10000.0, -10001.0, -9999.0]],
            [[12000.0, -12000.0, 0.0], [5000.0, 5000.0, 5000.0]],
        ],
        dtype=torch.float32,
    )

    actual = run_softmax(x, dim=-1)
    expected = torch.softmax(x, dim=-1)

    # Must be finite: naive unstable implementations often produce inf/nan.
    assert torch.isfinite(actual).all()

    # Probabilities should sum to 1 on the softmax dimension.
    sums = actual.sum(dim=-1)
    np.testing.assert_allclose(sums.detach().cpu().numpy(), np.ones_like(sums.detach().cpu().numpy()), rtol=1e-6, atol=1e-6)

    # Match PyTorch reference on a numerically challenging input.
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_softmax_shift_invariance():
    x = torch.tensor([[1000.0, 999.0, 998.0], [-1000.0, -1001.0, -1002.0]], dtype=torch.float32)
    shift = 12345.0
    out1 = run_softmax(x, dim=-1)
    out2 = run_softmax(x + shift, dim=-1)
    np.testing.assert_allclose(out1.detach().cpu().numpy(), out2.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
