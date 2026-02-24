"""Non-causal multi-head cross-attention test from checkpoint weights."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_multihead_cross_attention


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "cross_attention_dec_blocks_0_expected.npz"

PROJQ_WEIGHT_KEY = "dec_blocks.0.cross_attn.projq.weight"
PROJQ_BIAS_KEY = "dec_blocks.0.cross_attn.projq.bias"
PROJK_WEIGHT_KEY = "dec_blocks.0.cross_attn.projk.weight"
PROJK_BIAS_KEY = "dec_blocks.0.cross_attn.projk.bias"
PROJV_WEIGHT_KEY = "dec_blocks.0.cross_attn.projv.weight"
PROJV_BIAS_KEY = "dec_blocks.0.cross_attn.projv.bias"
OUT_WEIGHT_KEY = "dec_blocks.0.cross_attn.proj.weight"
OUT_BIAS_KEY = "dec_blocks.0.cross_attn.proj.bias"


def test_multihead_cross_attention_from_model_checkpoint_non_causal():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    projq_weight = state_dict[PROJQ_WEIGHT_KEY]
    projq_bias = state_dict[PROJQ_BIAS_KEY]
    projk_weight = state_dict[PROJK_WEIGHT_KEY]
    projk_bias = state_dict[PROJK_BIAS_KEY]
    projv_weight = state_dict[PROJV_WEIGHT_KEY]
    projv_bias = state_dict[PROJV_BIAS_KEY]
    out_weight = state_dict[OUT_WEIGHT_KEY]
    out_bias = state_dict[OUT_BIAS_KEY]

    expected = np.load(EXPECTED_PATH)
    query = torch.from_numpy(expected["query"])
    key = torch.from_numpy(expected["key"])
    value = torch.from_numpy(expected["value"])
    expected_output = expected["output"]
    d_model = int(expected["d_model"].item())
    num_heads = int(expected["num_heads"].item())

    actual_output = run_multihead_cross_attention(
        d_model=d_model,
        num_heads=num_heads,
        projq_weight=projq_weight,
        projq_bias=projq_bias,
        projk_weight=projk_weight,
        projk_bias=projk_bias,
        projv_weight=projv_weight,
        projv_bias=projv_bias,
        out_proj_weight=out_weight,
        out_proj_bias=out_bias,
        query=query,
        key=key,
        value=value,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
