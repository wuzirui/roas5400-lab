"""LayerNorm module test using one real checkpoint norm from tests/assets/model.pt."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_layernorm


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "layernorm_dec_blocks_0_norm1_expected.npz"
LAYERNORM_WEIGHT_KEY = "dec_blocks.0.norm1.weight"
LAYERNORM_BIAS_KEY = "dec_blocks.0.norm1.bias"
EPS = 1e-6


def test_layernorm_from_model_checkpoint_with_bias():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    weight = state_dict[LAYERNORM_WEIGHT_KEY]
    bias = state_dict[LAYERNORM_BIAS_KEY]

    expected = np.load(EXPECTED_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]

    actual_output = run_layernorm(
        d_model=weight.shape[0],
        eps=EPS,
        weights=weight,
        in_features=in_features,
        bias=bias,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
