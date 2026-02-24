"""Encoder block tests (self-attention + MLP)."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_encoder_block

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "encoder_block_enc_blocks_0_expected.npz"

NORM1_WEIGHT_KEY = "enc_blocks.0.norm1.weight"
NORM1_BIAS_KEY = "enc_blocks.0.norm1.bias"
ATTN_QKV_WEIGHT_KEY = "enc_blocks.0.attn.qkv.weight"
ATTN_QKV_BIAS_KEY = "enc_blocks.0.attn.qkv.bias"
ATTN_OUT_WEIGHT_KEY = "enc_blocks.0.attn.proj.weight"
ATTN_OUT_BIAS_KEY = "enc_blocks.0.attn.proj.bias"
NORM2_WEIGHT_KEY = "enc_blocks.0.norm2.weight"
NORM2_BIAS_KEY = "enc_blocks.0.norm2.bias"
FC1_WEIGHT_KEY = "enc_blocks.0.mlp.fc1.weight"
FC1_BIAS_KEY = "enc_blocks.0.mlp.fc1.bias"
FC2_WEIGHT_KEY = "enc_blocks.0.mlp.fc2.weight"
FC2_BIAS_KEY = "enc_blocks.0.mlp.fc2.bias"


def test_encoder_block_from_model_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    expected = np.load(EXPECTED_PATH)
    x = torch.from_numpy(expected["x"])
    token_positions = torch.from_numpy(expected["token_positions"])
    expected_output = expected["output"]

    actual_output = run_encoder_block(
        d_model=int(expected["d_model"].item()),
        num_heads=int(expected["num_heads"].item()),
        norm1_weight=state_dict[NORM1_WEIGHT_KEY],
        norm1_bias=state_dict[NORM1_BIAS_KEY],
        attn_qkv_weight=state_dict[ATTN_QKV_WEIGHT_KEY],
        attn_qkv_bias=state_dict[ATTN_QKV_BIAS_KEY],
        attn_out_weight=state_dict[ATTN_OUT_WEIGHT_KEY],
        attn_out_bias=state_dict[ATTN_OUT_BIAS_KEY],
        norm2_weight=state_dict[NORM2_WEIGHT_KEY],
        norm2_bias=state_dict[NORM2_BIAS_KEY],
        mlp_fc1_weight=state_dict[FC1_WEIGHT_KEY],
        mlp_fc1_bias=state_dict[FC1_BIAS_KEY],
        mlp_fc2_weight=state_dict[FC2_WEIGHT_KEY],
        mlp_fc2_bias=state_dict[FC2_BIAS_KEY],
        x=x,
        token_positions=token_positions,
        eps=float(expected["eps"].item()),
    )

    np.testing.assert_allclose(
        actual_output.detach().cpu().numpy(), expected_output, rtol=1e-4, atol=1e-4
    )


def test_encoder_block_does_not_skip_attention_sublayer():
    d_model = 8
    num_heads = 2
    d_ff = 16
    bsz, seq_len = 1, 4
    eps = 1e-6

    torch.manual_seed(7)
    x = torch.randn(bsz, seq_len, d_model, dtype=torch.float32)
    token_positions = torch.stack(
        [
            torch.zeros((bsz, seq_len), dtype=torch.int64),
            torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).expand(bsz, -1),
        ],
        dim=-1,
    )

    norm1_weight = torch.ones(d_model, dtype=torch.float32)
    norm1_bias = torch.zeros(d_model, dtype=torch.float32)
    norm2_weight = torch.ones(d_model, dtype=torch.float32)
    norm2_bias = torch.zeros(d_model, dtype=torch.float32)

    mlp_fc1_weight = torch.zeros(d_ff, d_model, dtype=torch.float32)
    mlp_fc1_bias = torch.zeros(d_ff, dtype=torch.float32)
    mlp_fc2_weight = torch.zeros(d_model, d_ff, dtype=torch.float32)
    mlp_fc2_bias = torch.zeros(d_model, dtype=torch.float32)

    attn_qkv_weight = torch.zeros(3 * d_model, d_model, dtype=torch.float32)
    for i in range(d_model):
        attn_qkv_weight[i, i] = 1.0
        attn_qkv_weight[d_model + i, i] = 1.0
        attn_qkv_weight[2 * d_model + i, i] = 1.0
    attn_qkv_bias = torch.zeros(3 * d_model, dtype=torch.float32)
    attn_out_weight = torch.eye(d_model, dtype=torch.float32)
    attn_out_bias = torch.zeros(d_model, dtype=torch.float32)

    output_with_attention = run_encoder_block(
        d_model=d_model,
        num_heads=num_heads,
        norm1_weight=norm1_weight,
        norm1_bias=norm1_bias,
        attn_qkv_weight=attn_qkv_weight,
        attn_qkv_bias=attn_qkv_bias,
        attn_out_weight=attn_out_weight,
        attn_out_bias=attn_out_bias,
        norm2_weight=norm2_weight,
        norm2_bias=norm2_bias,
        mlp_fc1_weight=mlp_fc1_weight,
        mlp_fc1_bias=mlp_fc1_bias,
        mlp_fc2_weight=mlp_fc2_weight,
        mlp_fc2_bias=mlp_fc2_bias,
        x=x,
        token_positions=token_positions,
        eps=eps,
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        output_with_attention.detach().cpu().numpy(),
        x.detach().cpu().numpy(),
        rtol=0.0,
        atol=1e-7,
    )
