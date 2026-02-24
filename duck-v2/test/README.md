This test split contains masked images only.
Ground-truth full images are hidden and not included in the student bundle.

Mask generation for this bundle:
- image_size=224
- patch_size=8
- mask_ratio=0.5 (exact-k sampling per image)
- fill_value=0

Local mask files are saved in `duck-v2/test_masks/`:
- `*_mask_p8_fine.npy` (28x28 patch mask)
- `*_mask_p8_coarse.npy` (same content, kept for naming consistency)
- corresponding `.png` visualizations and `manifest.json`
