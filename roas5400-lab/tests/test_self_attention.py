"""Non-causal multi-head self-attention test from checkpoint weights."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_multihead_self_attention


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "self_attention_dec_blocks_0_expected.npz"

QKV_WEIGHT_KEY = "dec_blocks.0.attn.qkv.weight"
QKV_BIAS_KEY = "dec_blocks.0.attn.qkv.bias"
OUT_WEIGHT_KEY = "dec_blocks.0.attn.proj.weight"
OUT_BIAS_KEY = "dec_blocks.0.attn.proj.bias"


def test_multihead_self_attention_from_model_checkpoint_non_causal():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    qkv_weight = state_dict[QKV_WEIGHT_KEY]
    qkv_bias = state_dict[QKV_BIAS_KEY]
    out_weight = state_dict[OUT_WEIGHT_KEY]
    out_bias = state_dict[OUT_BIAS_KEY]

    expected = np.load(EXPECTED_PATH)
    in_features = torch.from_numpy(expected["in_features"])
    expected_output = expected["output"]
    d_model = int(expected["d_model"].item())
    num_heads = int(expected["num_heads"].item())

    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        qkv_proj_weight=qkv_weight,
        qkv_proj_bias=qkv_bias,
        out_proj_weight=out_weight,
        out_proj_bias=out_bias,
        in_features=in_features,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
