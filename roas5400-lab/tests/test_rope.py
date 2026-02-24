"""RoPE2D test with deterministic input and expected output snapshot."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_rope2d


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "rope2d_expected.npz"


def test_rope2d_matches_expected():
    expected = np.load(EXPECTED_PATH)
    in_query_or_key = torch.from_numpy(expected["in_query_or_key"])
    token_positions = torch.from_numpy(expected["token_positions"]).to(torch.long)
    expected_output = expected["output"]
    freq = float(expected["freq"].item())
    f0 = float(expected["f0"].item())

    actual_output = run_rope2d(
        in_query_or_key=in_query_or_key,
        token_positions=token_positions,
        freq=freq,
        f0=f0,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)
