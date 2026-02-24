"""PatchEmbed module tests using checkpoint projection weights."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_patch_embed


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "patch_embed_expected.npz"
PATCH_EMBED_WEIGHT_KEY = "patch_embed.proj.weight"
PATCH_EMBED_BIAS_KEY = "patch_embed.proj.bias"


def test_patch_embed_from_model_checkpoint_projection():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    proj_weight = state_dict[PATCH_EMBED_WEIGHT_KEY]
    proj_bias = state_dict[PATCH_EMBED_BIAS_KEY]

    expected = np.load(EXPECTED_PATH)
    images = torch.from_numpy(expected["images"])
    expected_output = expected["output"]
    expected_positions = expected["positions"]

    actual_output, actual_positions = run_patch_embed(
        img_size=int(expected["img_size"].item()),
        patch_size=int(expected["patch_size"].item()),
        in_chans=int(expected["in_chans"].item()),
        embed_dim=int(expected["embed_dim"].item()),
        proj_weight=proj_weight,
        proj_bias=proj_bias,
        images=images,
    )

    np.testing.assert_allclose(actual_output.detach().cpu().numpy(), expected_output, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(actual_positions.detach().cpu().numpy(), expected_positions)


def test_patch_embed_positions_are_row_major_yx():
    img_size = 16
    patch_size = 8
    in_chans = 3
    embed_dim = 4
    batch_size = 2

    proj_weight = torch.zeros((embed_dim, in_chans, patch_size, patch_size), dtype=torch.float32)
    proj_bias = torch.zeros((embed_dim,), dtype=torch.float32)
    images = torch.zeros((batch_size, in_chans, img_size, img_size), dtype=torch.float32)

    actual_output, actual_positions = run_patch_embed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        proj_weight=proj_weight,
        proj_bias=proj_bias,
        images=images,
    )

    assert tuple(actual_output.shape) == (batch_size, 4, embed_dim)
    np.testing.assert_array_equal(actual_output.detach().cpu().numpy(), np.zeros((batch_size, 4, embed_dim)))

    expected_positions = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=np.int64,
    )
    expected_positions = np.repeat(expected_positions[None, ...], batch_size, axis=0)
    np.testing.assert_array_equal(actual_positions.detach().cpu().numpy(), expected_positions)
