"""Cat decoder block test with fixed synthetic weights."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_cat_block


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "cat_block_expected.npz"


def test_cat_block_matches_expected():
    expected = np.load(EXPECTED_PATH)

    actual_output_x, actual_output_mem = run_cat_block(
        d_model=int(expected["d_model"].item()),
        num_heads=int(expected["num_heads"].item()),
        norm_mem_weight=torch.from_numpy(expected["norm_mem_weight"]),
        norm_mem_bias=torch.from_numpy(expected["norm_mem_bias"]),
        norm1_weight=torch.from_numpy(expected["norm1_weight"]),
        norm1_bias=torch.from_numpy(expected["norm1_bias"]),
        attn_qkv_weight=torch.from_numpy(expected["attn_qkv_weight"]),
        attn_qkv_bias=torch.from_numpy(expected["attn_qkv_bias"]),
        attn_out_weight=torch.from_numpy(expected["attn_out_weight"]),
        attn_out_bias=torch.from_numpy(expected["attn_out_bias"]),
        norm2_weight=torch.from_numpy(expected["norm2_weight"]),
        norm2_bias=torch.from_numpy(expected["norm2_bias"]),
        mlp_fc1_weight=torch.from_numpy(expected["mlp_fc1_weight"]),
        mlp_fc1_bias=torch.from_numpy(expected["mlp_fc1_bias"]),
        mlp_fc2_weight=torch.from_numpy(expected["mlp_fc2_weight"]),
        mlp_fc2_bias=torch.from_numpy(expected["mlp_fc2_bias"]),
        x=torch.from_numpy(expected["x"]),
        mem=torch.from_numpy(expected["mem"]),
        pos_x=torch.from_numpy(expected["pos_x"]),
        pos_mem=torch.from_numpy(expected["pos_mem"]),
        eps=float(expected["eps"].item()),
    )

    np.testing.assert_allclose(
        actual_output_x.detach().cpu().numpy(),
        expected["output_x"],
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual_output_mem.detach().cpu().numpy(),
        expected["output_mem"],
        rtol=1e-5,
        atol=1e-5,
    )
