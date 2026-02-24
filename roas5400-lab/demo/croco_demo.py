"""CroCo demo: visualize masked reconstruction with student implementation."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import (
    build_croco_model,
    run_random_mask_patch,
    run_croco_forward,
    run_patchify,
    run_unpatchify,
)


def _find_duck_images(num_images: int = 2) -> list[Path]:
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
    if len(image_paths) < num_images:
        return [image_paths[0]] * num_images
    return image_paths[:num_images]


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize((image_size, image_size), Image.Resampling.BICUBIC)
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    return np.clip(x[0].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize student CroCo masked reconstruction."
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "tests" / "assets" / "model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "demo" / "croco_demo.png",
    )
    parser.add_argument(
        "--timing-runs",
        type=int,
        default=5,
        help="Number of repeated forward passes for runtime measurement.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install it first, e.g. `python -m pip install matplotlib`."
        ) from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"]
    try:
        model = build_croco_model(
            img_size=args.image_size,
            patch_size=args.patch_size,
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
    except NotImplementedError as exc:
        raise RuntimeError("CroCo is not implemented yet in src/croco.py") from exc

    img_a, img_b = _find_duck_images(num_images=2)
    target = _load_image(img_a, args.image_size)
    reference = _load_image(img_b, args.image_size)

    target_patches = run_patchify(target, patch_size=args.patch_size)
    mask = run_random_mask_patch(target_patches, mask_ratio=args.mask_ratio)

    with torch.no_grad():
        pred_patches, used_mask, gt_patches = run_croco_forward(
            model,
            target,
            reference,
            mask=mask,
        )

    if not torch.equal(used_mask, mask):
        raise RuntimeError("CroCo did not use the manual mask provided by the demo.")

    masked_patches = gt_patches.clone()
    masked_patches[used_mask] = 0.5
    masked_input = run_unpatchify(
        masked_patches, patch_size=args.patch_size, channels=3
    )

    raw_prediction = run_unpatchify(
        pred_patches, patch_size=args.patch_size, channels=3
    )

    blended_patches = gt_patches.clone()
    blended_patches[used_mask] = pred_patches[used_mask]
    blended_reconstruction = run_unpatchify(
        blended_patches, patch_size=args.patch_size, channels=3
    )

    mse = torch.mean((blended_reconstruction - target) ** 2).item()
    psnr_db = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)

    runtimes_ms: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, args.timing_runs)):
            t0 = time.perf_counter()
            run_croco_forward(model, target, reference, mask=mask)
            t1 = time.perf_counter()
            runtimes_ms.append((t1 - t0) * 1000.0)
    runtime_ms = float(np.mean(runtimes_ms))

    num_patches = int(used_mask.shape[1])
    grid = int(round(np.sqrt(num_patches)))
    if grid * grid != num_patches:
        raise ValueError(f"num_patches must be a square number, got {num_patches}")
    mask_map = (
        used_mask[0].reshape(grid, grid).detach().cpu().numpy().astype(np.float32)
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)

    axes[0, 0].imshow(_to_numpy_img(target))
    axes[0, 0].set_title("Target image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(_to_numpy_img(reference))
    axes[0, 1].set_title("Reference image")
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(mask_map, cmap="gray_r", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(
        f"Manual mask (True=masked)\nratio={float(mask.float().mean()):.3f}"
    )
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(_to_numpy_img(masked_input))
    axes[1, 0].set_title("Masked target input")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(_to_numpy_img(raw_prediction))
    axes[1, 1].set_title("Raw model prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(_to_numpy_img(blended_reconstruction))
    axes[1, 2].set_title("Blended reconstruction\n(unmasked=GT, masked=prediction)")
    axes[1, 2].axis("off")

    fig.suptitle("CroCo Demo (student implementation via adapters)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Saved demo image to: {args.output}")
    print(
        f"Metrics ({args.image_size}x{args.image_size}): "
        f"PSNR={psnr_db:.3f} dB, Runtime/Image={runtime_ms:.2f} ms "
        f"(avg over {max(1, args.timing_runs)} runs)"
    )


if __name__ == "__main__":
    main()
