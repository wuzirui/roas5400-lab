"""CroCo checkpoint-alignment tests with deterministic manual masks."""

from pathlib import Path

import numpy as np
import torch

from .adapters import build_croco_model, run_croco_forward

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "croco_expected.pt"


def _build_and_load_croco(state_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    model = build_croco_model(
        img_size=128,
        patch_size=8,
        mask_ratio=0.5,
        enc_embed_dim=192,
        enc_depth=3,
        enc_num_heads=6,
        dec_embed_dim=128,
        dec_depth=3,
        dec_num_heads=4,
        mlp_ratio=4.0,
        decoder_type="cross",
        use_rope=True,
        rope_freq=100.0,
        rope_f0=1.0,
        device="cpu",
        dtype=torch.float32,
        state_dict=state_dict,
    )
    model.eval()
    return model


def test_croco_matches_pretrained_checkpoint_with_manual_mask():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    expected = torch.load(EXPECTED_PATH, map_location="cpu")

    model = _build_and_load_croco(state_dict)

    target = expected["target"].to(torch.float32)
    reference = expected["reference"].to(torch.float32)
    manual_mask = expected["mask"].to(torch.bool)

    pred, used_mask, target_patches = run_croco_forward(
        model,
        target,
        reference,
        mask=manual_mask,
    )

    np.testing.assert_array_equal(used_mask.detach().cpu().numpy(), manual_mask.numpy())

    np.testing.assert_allclose(
        pred.detach().cpu().numpy(),
        expected["output"].detach().cpu().numpy(),
        rtol=float(expected["rtol"]),
        atol=float(expected["atol"]),
    )
    np.testing.assert_allclose(
        target_patches.detach().cpu().numpy(),
        expected["target_patches"].detach().cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_croco_manual_mask_override_changes_output():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    expected = torch.load(EXPECTED_PATH, map_location="cpu")

    model = _build_and_load_croco(state_dict)

    target = expected["target"].to(torch.float32)
    reference = expected["reference"].to(torch.float32)
    mask_a = expected["mask"].to(torch.bool)
    mask_b = torch.zeros_like(mask_a)
    mask_b[:, ::3] = True

    pred_a, used_a, _ = run_croco_forward(model, target, reference, mask=mask_a)
    pred_b, used_b, _ = run_croco_forward(model, target, reference, mask=mask_b)

    np.testing.assert_array_equal(used_a.detach().cpu().numpy(), mask_a.numpy())
    np.testing.assert_array_equal(used_b.detach().cpu().numpy(), mask_b.numpy())

    assert not np.array_equal(
        pred_a.detach().cpu().numpy(),
        pred_b.detach().cpu().numpy(),
    )
