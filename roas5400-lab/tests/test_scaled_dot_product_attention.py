"""Scaled dot-product attention tests (non-causal by default)."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_scaled_dot_product_attention

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "scaled_dot_product_attention_expected.npz"


def test_scaled_dot_product_attention_matches_expected():
    expected = np.load(EXPECTED_PATH)
    q = torch.from_numpy(expected["q"])
    k = torch.from_numpy(expected["k"])
    v = torch.from_numpy(expected["v"])
    expected_output = expected["output"]

    actual_output = run_scaled_dot_product_attention(q=q, k=k, v=v)
    np.testing.assert_allclose(
        actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6
    )


def test_scaled_dot_product_attention_is_non_causal_by_default():
    # q_len == k_len == 3, d_k == d_v == 2
    q = torch.tensor([[[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.0, -1.0], [0.0, -1.0], [0.0, 10.0]]]], dtype=torch.float32)
    v1 = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)
    v2 = torch.tensor([[[[0.0, 0.0], [0.0, 0.0], [9.0, 9.0]]]], dtype=torch.float32)

    out1 = run_scaled_dot_product_attention(q=q, k=k, v=v1)
    out2 = run_scaled_dot_product_attention(q=q, k=k, v=v2)

    # If attention is non-causal, query index 0 can attend to key index 2, so output[0] must change.
    delta = (out2 - out1).abs()[0, 0, 0].max().item()
    assert delta > 0.1
