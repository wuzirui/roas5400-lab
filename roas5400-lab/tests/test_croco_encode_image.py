"""CroCo encoder-path checkpoint alignment tests."""

from pathlib import Path

import numpy as np
import torch

from .adapters import build_croco_model, croco_encode_image

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS_DIR / "model.pt"
EXPECTED_PATH = ASSETS_DIR / "croco_encode_expected.pt"


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
        state_dict=None,
    )

    encoder_state = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("patch_embed.")
        or k.startswith("enc_blocks.")
        or k.startswith("enc_norm.")
    }
    load_result = model.load_state_dict(encoder_state, strict=False)
    assert len(load_result.unexpected_keys) == 0, (
        f"Unexpected keys while loading encoder weights: {load_result.unexpected_keys}"
    )
    model.eval()
    return model


def test_croco_encode_image_masked_matches_expected():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    expected = torch.load(EXPECTED_PATH, map_location="cpu")

    model = _build_and_load_croco(state_dict)

    image = expected["image"].to(torch.float32)
    manual_mask = expected["mask"].to(torch.bool)

    encoded, positions, used_mask = croco_encode_image(
        model=model,
        image=image,
        apply_mask=True,
        mask=manual_mask,
    )

    np.testing.assert_array_equal(used_mask.detach().cpu().numpy(), manual_mask.numpy())
    np.testing.assert_array_equal(
        positions.detach().cpu().numpy(), expected["positions"].detach().cpu().numpy()
    )
    np.testing.assert_allclose(
        encoded.detach().cpu().numpy(),
        expected["encoded_masked"].detach().cpu().numpy(),
        rtol=float(expected["rtol"]),
        atol=float(expected["atol"]),
    )


def test_croco_encode_image_unmasked_matches_expected():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    expected = torch.load(EXPECTED_PATH, map_location="cpu")

    model = _build_and_load_croco(state_dict)

    image = expected["image"].to(torch.float32)
    encoded, positions, used_mask = croco_encode_image(
        model=model,
        image=image,
        apply_mask=False,
        mask=None,
    )

    np.testing.assert_array_equal(
        positions.detach().cpu().numpy(), expected["positions"].detach().cpu().numpy()
    )
    np.testing.assert_array_equal(
        used_mask.detach().cpu().numpy(),
        expected["used_unmasked"].detach().cpu().numpy(),
    )
    np.testing.assert_allclose(
        encoded.detach().cpu().numpy(),
        expected["encoded_unmasked"].detach().cpu().numpy(),
        rtol=float(expected["rtol"]),
        atol=float(expected["atol"]),
    )


def test_croco_encode_image_manual_mask_override_changes_output():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model"]
    expected = torch.load(EXPECTED_PATH, map_location="cpu")

    model = _build_and_load_croco(state_dict)
    image = expected["image"].to(torch.float32)

    mask_a = expected["mask"].to(torch.bool)
    mask_b = torch.zeros_like(mask_a)
    mask_b[:, ::3] = True

    encoded_a, _pos_a, used_a = croco_encode_image(
        model=model,
        image=image,
        apply_mask=True,
        mask=mask_a,
    )
    encoded_b, _pos_b, used_b = croco_encode_image(
        model=model,
        image=image,
        apply_mask=True,
        mask=mask_b,
    )

    np.testing.assert_array_equal(used_a.detach().cpu().numpy(), mask_a.numpy())
    np.testing.assert_array_equal(used_b.detach().cpu().numpy(), mask_b.numpy())
    assert not np.array_equal(
        encoded_a.detach().cpu().numpy(),
        encoded_b.detach().cpu().numpy(),
    )
