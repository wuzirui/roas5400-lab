"""Demo: apply random patch mask using student APIs and visualize the result."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import run_patchify, run_random_mask_patch, run_unpatchify


def _find_duck_image() -> Path:
    duck_train = PROJECT_ROOT.parent / "duck-v2" / "train"
    if not duck_train.exists():
        raise FileNotFoundError(f"Duck dataset not found: {duck_train}")
    image_paths = sorted(
        p
        for p in duck_train.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {duck_train}")
    return image_paths[0]


def _load_image_224(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _mask_to_pixel_map(mask: torch.Tensor, patch_size: int) -> np.ndarray:
    # mask: [1, num_patches], bool
    n = int(mask.shape[1])
    grid = int(math.isqrt(n))
    if grid * grid != n:
        raise ValueError(f"num_patches must be square, got {n}")
    m = mask[0].detach().cpu().numpy().reshape(grid, grid).astype(np.float32)
    return np.kron(m, np.ones((patch_size, patch_size), dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random patch masking using student APIs.")
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "demo" / "random_mask_patch_demo.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is not installed. Install it first: `python -m pip install matplotlib`.") from exc

    image_path = _find_duck_image()
    image_tensor = _load_image_224(image_path)

    try:
        patches = run_patchify(image_tensor, patch_size=args.patch_size)
    except NotImplementedError as exc:
        raise RuntimeError("run_patchify is not implemented yet in tests/adapters.py.") from exc

    torch.manual_seed(args.seed)
    try:
        mask = run_random_mask_patch(patches, mask_ratio=args.mask_ratio)
    except NotImplementedError as exc:
        raise RuntimeError("run_random_mask_patch is not implemented yet in tests/adapters.py.") from exc

    if mask.dtype != torch.bool:
        raise TypeError(f"run_random_mask_patch must return bool tensor, got {mask.dtype}")

    masked_patches = patches.clone()
    masked_patches[mask] = 0.5

    try:
        masked_image = run_unpatchify(masked_patches, patch_size=args.patch_size, channels=3)
    except NotImplementedError as exc:
        raise RuntimeError("run_unpatchify is not implemented yet in tests/adapters.py.") from exc

    original_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    masked_np = masked_image[0].permute(1, 2, 0).detach().cpu().numpy()
    mask_map = _mask_to_pixel_map(mask, patch_size=args.patch_size)

    masked_count = int(mask.sum().item())
    total = int(mask.numel())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    axes[0].imshow(np.clip(original_np, 0.0, 1.0))
    axes[0].set_title("Original (224x224)")
    axes[0].axis("off")

    im = axes[1].imshow(mask_map, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Patch Mask (white=masked)\n{masked_count}/{total}")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(np.clip(masked_np, 0.0, 1.0))
    axes[2].set_title(f"Masked Reconstruction\npatch={args.patch_size}, ratio={args.mask_ratio:.2f}")
    axes[2].axis("off")

    fig.suptitle(f"Random Mask Patch Demo ({image_path.name})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved demo image to: {args.output}")


if __name__ == "__main__":
    main()
