"""MLP module test using one real checkpoint block from tests/assets/model.pt."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_mlp


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "mlp_dec_blocks_0_expected.npz"
W1_WEIGHT_KEY = "dec_blocks.0.mlp.fc1.weight"
W1_BIAS_KEY = "dec_blocks.0.mlp.fc1.bias"
W2_WEIGHT_KEY = "dec_blocks.0.mlp.fc2.weight"
W2_BIAS_KEY = "dec_blocks.0.mlp.fc2.bias"


def test_mlp_from_model_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    w1_weight = state_dict[W1_WEIGHT_KEY]
    w1_bias = state_dict[W1_BIAS_KEY]
    w2_weight = state_dict[W2_WEIGHT_KEY]
    w2_bias = state_dict[W2_BIAS_KEY]

    expected = np.load(EXPECTED_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]

    actual_output = run_mlp(
        d_model=w1_weight.shape[1],
        d_ff=w1_weight.shape[0],
        w1_weight=w1_weight,
        w1_bias=w1_bias,
        w2_weight=w2_weight,
        w2_bias=w2_bias,
        in_features=in_features,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
