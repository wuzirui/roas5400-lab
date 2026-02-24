"""Patchify and unpatchify tests using a resized duck image."""

from pathlib import Path

import numpy as np
import torch

from .adapters import run_patchify, run_unpatchify


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
EXPECTED_PATH = ASSETS_DIR / "patchify_unpatchify_duck_expected.npz"


def _load_expected():
    expected = np.load(EXPECTED_PATH)
    image = torch.from_numpy(expected["image"])
    patches = expected["patches"]
    reconstructed = expected["reconstructed"]
    patch_size = int(expected["patch_size"].item())
    return image, patches, reconstructed, patch_size


def test_patchify_on_resized_duck_image():
    image, expected_patches, _, patch_size = _load_expected()
    actual_patches = run_patchify(image, patch_size=patch_size)
    np.testing.assert_allclose(actual_patches.detach().cpu().numpy(), expected_patches, rtol=1e-6, atol=1e-6)


def test_patchify_layout_is_patch_h_patch_w_channel():
    image = torch.arange(1 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    patch_size = 2
    actual_patches = run_patchify(image, patch_size=patch_size).detach().cpu().numpy()

    # Expected flatten order inside each patch: (patch_h, patch_w, channel).
    expected_patch_list = []
    for gh in range(0, 4, patch_size):
        for gw in range(0, 4, patch_size):
            patch_values = []
            for ph in range(patch_size):
                for pw in range(patch_size):
                    for c in range(3):
                        patch_values.append(float(image[0, c, gh + ph, gw + pw]))
            expected_patch_list.append(patch_values)
    expected = np.asarray(expected_patch_list, dtype=np.float32)[None, ...]

    np.testing.assert_allclose(actual_patches, expected, rtol=0.0, atol=0.0)


def test_unpatchify_on_duck_patches():
    _, patches, expected_reconstructed, patch_size = _load_expected()
    # Unpatchify assumes a square patch grid (hh == ww); here 784 = 28 * 28.
    assert int(np.sqrt(patches.shape[1])) ** 2 == patches.shape[1]
    patches_tensor = torch.from_numpy(patches)
    actual_reconstructed = run_unpatchify(patches_tensor, patch_size=patch_size, channels=3)
    np.testing.assert_allclose(
        actual_reconstructed.detach().cpu().numpy(),
        expected_reconstructed,
        rtol=1e-6,
        atol=1e-6,
    )
