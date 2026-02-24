"""Cross decoder block test from checkpoint weights (non-causal)."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_cross_block

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "cross_block_dec_blocks_0_expected.npz"

NORM1_WEIGHT_KEY = "dec_blocks.0.norm1.weight"
NORM1_BIAS_KEY = "dec_blocks.0.norm1.bias"
SELF_QKV_WEIGHT_KEY = "dec_blocks.0.attn.qkv.weight"
SELF_QKV_BIAS_KEY = "dec_blocks.0.attn.qkv.bias"
SELF_OUT_WEIGHT_KEY = "dec_blocks.0.attn.proj.weight"
SELF_OUT_BIAS_KEY = "dec_blocks.0.attn.proj.bias"
NORM2_WEIGHT_KEY = "dec_blocks.0.norm2.weight"
NORM2_BIAS_KEY = "dec_blocks.0.norm2.bias"
CQ_WEIGHT_KEY = "dec_blocks.0.cross_attn.projq.weight"
CQ_BIAS_KEY = "dec_blocks.0.cross_attn.projq.bias"
CK_WEIGHT_KEY = "dec_blocks.0.cross_attn.projk.weight"
CK_BIAS_KEY = "dec_blocks.0.cross_attn.projk.bias"
CV_WEIGHT_KEY = "dec_blocks.0.cross_attn.projv.weight"
CV_BIAS_KEY = "dec_blocks.0.cross_attn.projv.bias"
CO_WEIGHT_KEY = "dec_blocks.0.cross_attn.proj.weight"
CO_BIAS_KEY = "dec_blocks.0.cross_attn.proj.bias"
NORM3_WEIGHT_KEY = "dec_blocks.0.norm3.weight"
NORM3_BIAS_KEY = "dec_blocks.0.norm3.bias"
FC1_WEIGHT_KEY = "dec_blocks.0.mlp.fc1.weight"
FC1_BIAS_KEY = "dec_blocks.0.mlp.fc1.bias"
FC2_WEIGHT_KEY = "dec_blocks.0.mlp.fc2.weight"
FC2_BIAS_KEY = "dec_blocks.0.mlp.fc2.bias"
NORMY_WEIGHT_KEY = "dec_blocks.0.norm_y.weight"
NORMY_BIAS_KEY = "dec_blocks.0.norm_y.bias"


def test_cross_block_from_model_checkpoint():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]

    expected = np.load(EXPECTED_PATH)
    x = torch.from_numpy(expected["x"])
    mem = torch.from_numpy(expected["mem"])
    pos_x = torch.from_numpy(expected["pos_x"])
    pos_mem = torch.from_numpy(expected["pos_mem"])
    expected_output_x = expected["output_x"]
    expected_output_mem = expected["output_mem"]

    actual_output_x, actual_output_mem = run_cross_block(
        d_model=int(expected["d_model"].item()),
        num_heads=int(expected["num_heads"].item()),
        norm1_weight=state_dict[NORM1_WEIGHT_KEY],
        norm1_bias=state_dict[NORM1_BIAS_KEY],
        self_attn_qkv_weight=state_dict[SELF_QKV_WEIGHT_KEY],
        self_attn_qkv_bias=state_dict[SELF_QKV_BIAS_KEY],
        self_attn_out_weight=state_dict[SELF_OUT_WEIGHT_KEY],
        self_attn_out_bias=state_dict[SELF_OUT_BIAS_KEY],
        norm2_weight=state_dict[NORM2_WEIGHT_KEY],
        norm2_bias=state_dict[NORM2_BIAS_KEY],
        cross_attn_q_weight=state_dict[CQ_WEIGHT_KEY],
        cross_attn_q_bias=state_dict[CQ_BIAS_KEY],
        cross_attn_k_weight=state_dict[CK_WEIGHT_KEY],
        cross_attn_k_bias=state_dict[CK_BIAS_KEY],
        cross_attn_v_weight=state_dict[CV_WEIGHT_KEY],
        cross_attn_v_bias=state_dict[CV_BIAS_KEY],
        cross_attn_out_weight=state_dict[CO_WEIGHT_KEY],
        cross_attn_out_bias=state_dict[CO_BIAS_KEY],
        norm3_weight=state_dict[NORM3_WEIGHT_KEY],
        norm3_bias=state_dict[NORM3_BIAS_KEY],
        mlp_fc1_weight=state_dict[FC1_WEIGHT_KEY],
        mlp_fc1_bias=state_dict[FC1_BIAS_KEY],
        mlp_fc2_weight=state_dict[FC2_WEIGHT_KEY],
        mlp_fc2_bias=state_dict[FC2_BIAS_KEY],
        norm_y_weight=state_dict[NORMY_WEIGHT_KEY],
        norm_y_bias=state_dict[NORMY_BIAS_KEY],
        x=x,
        mem=mem,
        pos_x=pos_x,
        pos_mem=pos_mem,
        eps=float(expected["eps"].item()),
    )

    np.testing.assert_allclose(
        actual_output_x.detach().cpu().numpy(),
        expected_output_x,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_output_mem.detach().cpu().numpy(),
        expected_output_mem,
        rtol=1e-6,
        atol=1e-6,
    )
