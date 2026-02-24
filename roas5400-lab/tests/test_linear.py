"""Linear module test using one real checkpoint weight from tests/assets/model.pt."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_linear


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_WITH_BIAS_PATH = ASSETS_DIR / "linear_dec_blocks_0_mlp_fc1_expected.npz"
EXPECTED_NO_BIAS_PATH = ASSETS_DIR / "linear_dec_blocks_0_mlp_fc1_expected_no_bias.npz"
LINEAR_WEIGHT_KEY = "dec_blocks.0.mlp.fc1.weight"
LINEAR_BIAS_KEY = "dec_blocks.0.mlp.fc1.bias"


def _load_linear_params() -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    weight = state_dict[LINEAR_WEIGHT_KEY]
    bias = state_dict[LINEAR_BIAS_KEY]
    return weight, bias


def test_linear_from_model_checkpoint_with_bias():
    weight, bias = _load_linear_params()

    expected = np.load(EXPECTED_WITH_BIAS_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]

    actual_output = run_linear(
        d_in=weight.shape[1],
        d_out=weight.shape[0],
        weights=weight,
        bias=bias,
        in_features=in_features,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)


def test_linear_from_model_checkpoint_without_bias():
    weight, _ = _load_linear_params()

    expected = np.load(EXPECTED_NO_BIAS_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]

    actual_output = run_linear(
        d_in=weight.shape[1],
        d_out=weight.shape[0],
        weights=weight,
        bias=None,
        in_features=in_features,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-6, atol=1e-6)
