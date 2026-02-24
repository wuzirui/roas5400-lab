"""Predict full images from masked test inputs without using GT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.adapters import (
    build_croco_model,
    run_croco_forward,
    run_patchify,
    run_unpatchify,
)


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize((image_size, image_size), Image.Resampling.BICUBIC)
    )
    arr = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    return arr.permute(2, 0, 1).unsqueeze(0).contiguous()


def _save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chw_uint8 = (
        image[0].detach().cpu().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous()
    )
    h, w = int(chw_uint8.shape[1]), int(chw_uint8.shape[2])
    hwc_uint8 = chw_uint8.permute(1, 2, 0).contiguous()
    Image.frombytes("RGB", (w, h), bytes(hwc_uint8.view(-1).tolist())).save(path)


def _infer_patch_mask_from_fill(
    masked_image: torch.Tensor,
    patch_size: int,
    fill_value: float,
    fill_tolerance: float,
    min_patch_fill_ratio: float,
) -> torch.Tensor:
    if masked_image.ndim != 4 or masked_image.shape[1] != 3:
        raise ValueError("masked_image must have shape (batch, 3, H, W)")

    batch, _, height, width = masked_image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image size must be divisible by patch_size")

    pixel_is_fill = (masked_image - fill_value).abs().amax(
        dim=1
    ) <= fill_tolerance  # (B, H, W)

    patch_fill_ratio = (
        pixel_is_fill.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .float()
        .mean(dim=(-1, -2))
    )
    patch_mask = (patch_fill_ratio >= min_patch_fill_ratio).reshape(batch, -1)
    return patch_mask


def _load_patch_mask_file(mask_path: Path, expected_num_patches: int) -> torch.Tensor:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    arr = np.load(mask_path)
    if arr.ndim == 2:
        flat = arr.reshape(-1)
    elif arr.ndim == 1:
        flat = arr
    else:
        raise ValueError(
            f"Mask file must be 1D or 2D, got shape {arr.shape} from {mask_path}"
        )

    mask = torch.from_numpy(flat.astype(np.bool_)).view(1, -1)
    if int(mask.shape[1]) != int(expected_num_patches):
        raise ValueError(
            "Mask patch count mismatch: "
            f"expected {expected_num_patches}, got {int(mask.shape[1])} from {mask_path}"
        )
    return mask


def _build_model_from_checkpoint(
    checkpoint_path: Path,
    image_size: int,
    patch_size: int,
    mask_ratio: float,
    enc_embed_dim: int,
    enc_depth: int,
    enc_num_heads: int,
    dec_embed_dim: int,
    dec_depth: int,
    dec_num_heads: int,
    mlp_ratio: float,
    decoder_type: str,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )

    inferred_patch_size = None
    inferred_enc_embed_dim = None
    inferred_dec_embed_dim = None

    patch_embed_weight = state_dict.get("patch_embed.proj.weight")
    if isinstance(patch_embed_weight, torch.Tensor) and patch_embed_weight.ndim == 4:
        if patch_embed_weight.shape[2] != patch_embed_weight.shape[3]:
            raise ValueError(
                "Checkpoint patch_embed.proj.weight is not square in spatial dims: "
                f"{tuple(patch_embed_weight.shape)}"
            )
        inferred_patch_size = int(patch_embed_weight.shape[2])
        inferred_enc_embed_dim = int(patch_embed_weight.shape[0])

    pred_head_weight = state_dict.get("prediction_head.weight")
    if isinstance(pred_head_weight, torch.Tensor) and pred_head_weight.ndim == 2:
        inferred_dec_embed_dim = int(pred_head_weight.shape[1])

    resolved_patch_size = patch_size
    resolved_enc_embed_dim = enc_embed_dim
    resolved_dec_embed_dim = dec_embed_dim

    if inferred_patch_size is not None and resolved_patch_size != inferred_patch_size:
        print(
            "[predict_masked_test] Overriding --patch-size "
            f"{resolved_patch_size} -> {inferred_patch_size} from checkpoint."
        )
        resolved_patch_size = inferred_patch_size
    if (
        inferred_enc_embed_dim is not None
        and resolved_enc_embed_dim != inferred_enc_embed_dim
    ):
        print(
            "[predict_masked_test] Overriding --enc-embed-dim "
            f"{resolved_enc_embed_dim} -> {inferred_enc_embed_dim} from checkpoint."
        )
        resolved_enc_embed_dim = inferred_enc_embed_dim
    if (
        inferred_dec_embed_dim is not None
        and resolved_dec_embed_dim != inferred_dec_embed_dim
    ):
        print(
            "[predict_masked_test] Overriding --dec-embed-dim "
            f"{resolved_dec_embed_dim} -> {inferred_dec_embed_dim} from checkpoint."
        )
        resolved_dec_embed_dim = inferred_dec_embed_dim

    expected_patch_dim = 3 * resolved_patch_size * resolved_patch_size
    if (
        isinstance(pred_head_weight, torch.Tensor)
        and int(pred_head_weight.shape[0]) != expected_patch_dim
    ):
        raise ValueError(
            "Checkpoint prediction_head output dim does not match resolved patch size. "
            f"prediction_head.out={int(pred_head_weight.shape[0])}, "
            f"expected={expected_patch_dim} (3 * patch_size^2)."
        )

    model = build_croco_model(
        img_size=image_size,
        patch_size=resolved_patch_size,
        mask_ratio=mask_ratio,
        enc_embed_dim=resolved_enc_embed_dim,
        enc_depth=enc_depth,
        enc_num_heads=enc_num_heads,
        dec_embed_dim=resolved_dec_embed_dim,
        dec_depth=dec_depth,
        dec_num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        decoder_type=decoder_type,
        use_rope=True,
        rope_freq=100.0,
        rope_f0=1.0,
        device="cpu",
        dtype=torch.float32,
        state_dict=state_dict,
    )
    model.eval()
    return model, resolved_patch_size


def _reference_image_candidates(target_path: Path) -> list[Path]:
    train_dir = PROJECT_ROOT.parent / "duck-v2" / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Duck train directory not found: {train_dir}")

    candidates = sorted(
        p
        for p in train_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not candidates:
        raise RuntimeError(f"No reference images found in {train_dir}")

    filtered = [c for c in candidates if c.resolve() != target_path.resolve()]
    return filtered if filtered else candidates


def _select_reference_by_self_consistency(
    model: torch.nn.Module,
    masked_target: torch.Tensor,
    patch_mask: torch.Tensor,
    candidate_paths: list[Path],
    image_size: int,
    patch_size: int,
) -> tuple[Path, float]:
    if not candidate_paths:
        raise ValueError("No candidate reference images provided.")

    unmasked = ~patch_mask
    if not bool(unmasked.any().item()):
        return candidate_paths[0], float("inf")

    target_patches = run_patchify(masked_target, patch_size=patch_size)
    best_path = candidate_paths[0]
    best_score = float("inf")

    with torch.no_grad():
        for path in candidate_paths:
            ref = _load_image(path, image_size=image_size)
            pred_patches, _, _ = run_croco_forward(
                model,
                masked_target,
                ref,
                mask=patch_mask,
            )
            score = torch.mean(
                (pred_patches[unmasked] - target_patches[unmasked]) ** 2
            ).item()
            if score < best_score:
                best_score = score
                best_path = path

    return best_path, best_score


def _save_patch_mask(path: Path, patch_mask: torch.Tensor) -> None:
    num_patches = int(patch_mask.shape[1])
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches must be square, got {num_patches}")

    mask_grid = (
        patch_mask[0].reshape(grid_size, grid_size).detach().cpu().to(torch.uint8) * 255
    )
    Image.frombytes(
        "L",
        (grid_size, grid_size),
        bytes(mask_grid.contiguous().view(-1).tolist()),
    ).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a student CroCo checkpoint, inpaint a masked test image, and save the predicted full image. "
            "This script never loads test GT."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--target", type=Path, required=True, help="Masked test image path"
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        default=None,
        help=(
            "Patch mask file (.npy). If not provided, the script uses "
            "../duck-v2/test_masks/<target_stem>_mask_p<patch_size>_fine.npy"
        ),
    )
    parser.add_argument(
        "--reference", type=Path, default=None, help="Optional reference image path"
    )
    parser.add_argument(
        "--reference-selection",
        type=str,
        default="self_consistency",
        choices=["self_consistency", "first"],
        help="How to pick reference image when --reference is not provided.",
    )
    parser.add_argument(
        "--max-reference-candidates",
        type=int,
        default=0,
        help="Limit candidate references when auto-selecting (0 means use all).",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output reconstructed image path"
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=None,
        help="Optional raw prediction image path",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="Optional path to save inferred patch mask",
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.90)
    parser.add_argument("--enc-embed-dim", type=int, default=192)
    parser.add_argument("--enc-depth", type=int, default=3)
    parser.add_argument("--enc-num-heads", type=int, default=6)
    parser.add_argument("--dec-embed-dim", type=int, default=128)
    parser.add_argument("--dec-depth", type=int, default=3)
    parser.add_argument("--dec-num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument(
        "--decoder-type", type=str, default="cross", choices=["cross", "cat"]
    )

    args = parser.parse_args()

    model, resolved_patch_size = _build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        enc_embed_dim=args.enc_embed_dim,
        enc_depth=args.enc_depth,
        enc_num_heads=args.enc_num_heads,
        dec_embed_dim=args.dec_embed_dim,
        dec_depth=args.dec_depth,
        dec_num_heads=args.dec_num_heads,
        mlp_ratio=args.mlp_ratio,
        decoder_type=args.decoder_type,
    )

    masked_target = _load_image(args.target, args.image_size)

    if args.image_size % resolved_patch_size != 0:
        raise ValueError(
            f"image_size {args.image_size} is not divisible by patch_size {resolved_patch_size}"
        )
    expected_num_patches = (args.image_size // resolved_patch_size) ** 2
    default_mask_path = (
        PROJECT_ROOT.parent
        / "duck-v2"
        / "test_masks"
        / f"{args.target.stem}_mask_p{resolved_patch_size}_fine.npy"
    )
    mask_path = args.mask_file or default_mask_path

    patch_mask = _load_patch_mask_file(
        mask_path=mask_path,
        expected_num_patches=expected_num_patches,
    )
    detected_masked = int(patch_mask.sum().item())
    total_patches = int(patch_mask.numel())
    if detected_masked <= 0 or detected_masked >= total_patches:
        raise ValueError(
            "Loaded mask must have both masked and unmasked patches. "
            f"Got {detected_masked}/{total_patches} from {mask_path}"
        )

    if args.reference is not None:
        reference_path = args.reference
    else:
        candidates = _reference_image_candidates(args.target)
        if args.max_reference_candidates > 0:
            candidates = candidates[: args.max_reference_candidates]
        if args.reference_selection == "first":
            reference_path = candidates[0]
        else:
            reference_path, consistency_score = _select_reference_by_self_consistency(
                model=model,
                masked_target=masked_target,
                patch_mask=patch_mask,
                candidate_paths=candidates,
                image_size=args.image_size,
                patch_size=resolved_patch_size,
            )
            print(
                "[predict_masked_test] Auto-selected reference by self-consistency: "
                f"{reference_path.name} (score={consistency_score:.6f})"
            )

    reference_image = _load_image(reference_path, args.image_size)

    with torch.no_grad():
        pred_patches, used_mask, _ = run_croco_forward(
            model,
            masked_target,
            reference_image,
            mask=patch_mask,
        )

    masked_input_patches = run_patchify(masked_target, patch_size=resolved_patch_size)
    blended_patches = masked_input_patches.clone()
    blended_patches[used_mask] = pred_patches[used_mask]

    blended_image = run_unpatchify(
        blended_patches,
        patch_size=resolved_patch_size,
        channels=3,
    )
    _save_image(args.output, blended_image)

    if args.raw_output is not None:
        raw_prediction = run_unpatchify(
            pred_patches,
            patch_size=resolved_patch_size,
            channels=3,
        )
        _save_image(args.raw_output, raw_prediction)

    if args.save_mask is not None:
        args.save_mask.parent.mkdir(parents=True, exist_ok=True)
        _save_patch_mask(args.save_mask, used_mask)

    print(f"Target image: {args.target}")
    print(f"Mask file: {mask_path}")
    print(f"Reference image: {reference_path}")
    print(f"Detected masked patches: {detected_masked}/{total_patches}")
    print(f"Saved reconstructed image to: {args.output}")
    if args.raw_output is not None:
        print(f"Saved raw model prediction to: {args.raw_output}")
    if args.save_mask is not None:
        print(f"Saved inferred mask to: {args.save_mask}")


if __name__ == "__main__":
    main()
