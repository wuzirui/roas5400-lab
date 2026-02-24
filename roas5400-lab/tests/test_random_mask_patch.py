"""Random patch masking tests."""

import numpy as np
import torch

from .adapters import run_random_mask_patch


def test_random_mask_patch_shape_dtype_and_ratio_range():
    b, n, d = 8, 196, 24
    mask_ratio = 0.60
    patch_features = torch.linspace(-1.0, 1.0, steps=b * n * d, dtype=torch.float32).reshape(b, n, d)

    torch.manual_seed(2026)
    mask = run_random_mask_patch(patch_features, mask_ratio=mask_ratio)

    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (b, n)

    observed = mask.float().mean(dim=1).cpu().numpy()
    lower = max(0.0, mask_ratio - 0.10)
    upper = min(1.0, mask_ratio + 0.10)
    assert np.all((observed >= lower) & (observed <= upper))


def test_random_mask_patch_ratio_range_multiple_settings():
    settings = [(0.25, 64), (0.50, 100), (0.75, 196)]
    b, d = 6, 16

    for idx, (mask_ratio, n) in enumerate(settings):
        patch_features = torch.linspace(
            -1.0, 1.0, steps=b * n * d, dtype=torch.float32
        ).reshape(b, n, d)
        torch.manual_seed(3000 + idx)
        mask = run_random_mask_patch(patch_features, mask_ratio=mask_ratio)

        assert mask.dtype == torch.bool
        assert tuple(mask.shape) == (b, n)

        observed = mask.float().mean(dim=1).cpu().numpy()
        lower = max(0.0, mask_ratio - 0.12)
        upper = min(1.0, mask_ratio + 0.12)
        assert np.all((observed >= lower) & (observed <= upper))


def test_random_mask_patch_changes_across_sampling_runs():
    b, n, d = 4, 196, 24
    mask_ratio = 0.60
    patch_features = torch.linspace(-1.0, 1.0, steps=b * n * d, dtype=torch.float32).reshape(b, n, d)

    masks = []
    for seed in (11, 22, 33, 44, 55):
        torch.manual_seed(seed)
        mask = run_random_mask_patch(patch_features, mask_ratio=mask_ratio)
        assert mask.dtype == torch.bool
        masks.append(mask.cpu())

    # Multiple sampling runs should not collapse to one fixed mask.
    unique_masks = {m.numpy().tobytes() for m in masks}
    assert len(unique_masks) >= 2
