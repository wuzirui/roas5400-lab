"""Patchify/unpatchify flow demo on one duck image."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import run_patchify, run_unpatchify


def _find_duck_image() -> Path:
    duck_train = PROJECT_ROOT.parent / "duck-v2" / "train"
    if not duck_train.exists():
        raise FileNotFoundError(f"Duck dataset not found: {duck_train}")
    image_paths = sorted(
        p
        for p in duck_train.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {duck_train}")
    return image_paths[0]


def _to_tensor_224(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    # Keep aspect ratio and center-crop to square to avoid geometric distortion.
    image = ImageOps.fit(
        image,
        (224, 224),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _build_patch_board(
    patches: torch.Tensor, patch_size: int, channels: int = 3, gap: int = 3
) -> np.ndarray:
    p = patch_size
    b, n, d = patches.shape
    if b != 1:
        raise ValueError(f"Demo expects batch=1, got batch={b}")
    if d != p * p * channels:
        raise ValueError(f"patch dim mismatch: got {d}, expected {p*p*channels}")
    grid = int(math.isqrt(n))
    if grid * grid != n:
        raise ValueError(f"Number of patches must be square, got {n}")

    board_h = grid * p + (grid + 1) * gap
    board_w = grid * p + (grid + 1) * gap
    board = np.ones((board_h, board_w, channels), dtype=np.float32) * 0.95

    patches_np = patches[0].detach().cpu().numpy().reshape(grid, grid, p, p, channels)
    border = 1 if p >= 4 else 0
    for i in range(grid):
        for j in range(grid):
            y0 = gap + i * (p + gap)
            x0 = gap + j * (p + gap)
            bg = 0.88 if (i + j) % 2 == 0 else 0.78
            board[y0 : y0 + p, x0 : x0 + p, :] = bg
            if border > 0:
                board[
                    y0 + border : y0 + p - border, x0 + border : x0 + p - border, :
                ] = np.clip(
                    patches_np[i, j, border : p - border, border : p - border],
                    0.0,
                    1.0,
                )
            else:
                board[y0 : y0 + p, x0 : x0 + p, :] = np.clip(patches_np[i, j], 0.0, 1.0)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show patchify -> unpatchify flow on one duck image."
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "demo" / "patchify_unpatchify_flow.png",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it first, e.g. `python -m pip install matplotlib`."
        ) from exc

    image_path = _find_duck_image()
    image_tensor = _to_tensor_224(image_path)

    try:
        patches = run_patchify(image_tensor, patch_size=args.patch_size)
    except NotImplementedError as exc:
        raise RuntimeError(
            "run_patchify is not implemented yet in tests/adapters.py."
        ) from exc

    try:
        reconstructed = run_unpatchify(patches, patch_size=args.patch_size, channels=3)
    except NotImplementedError as exc:
        raise RuntimeError(
            "run_unpatchify is not implemented yet in tests/adapters.py."
        ) from exc

    original = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    patches_board = _build_patch_board(
        patches, patch_size=args.patch_size, channels=3, gap=3
    )
    rebuilt = reconstructed[0].permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=140)
    axes[0].imshow(np.clip(original, 0.0, 1.0))
    axes[0].set_title("1) Fit to 224x224 (keep aspect)")
    axes[0].axis("off")

    axes[1].imshow(np.clip(patches_board, 0.0, 1.0))
    axes[1].set_title(f"2) patchify (p={args.patch_size})")
    axes[1].axis("off")

    axes[2].imshow(np.clip(rebuilt, 0.0, 1.0))
    axes[2].set_title("3) unpatchify")
    axes[2].axis("off")

    fig.suptitle(f"Patchify/Unpatchify Flow ({image_path.name})", fontsize=14)

    fig.text(0.335, 0.5, "→", fontsize=28, ha="center", va="center")
    fig.text(0.665, 0.5, "→", fontsize=28, ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved flow figure to: {args.output}")


if __name__ == "__main__":
    main()
