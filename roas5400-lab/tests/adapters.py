from __future__ import annotations

from functools import partial

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor



def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
    bias: Float[Tensor, " d_out"] | None = None,
) -> Float[Tensor, " ... d_out"]:
    """
    Load `weights`/`bias` into your own Linear module and return its forward output.
    """
    raise NotImplementedError

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Load embedding weights into your Embedding module and return lookup results.
    """
    raise NotImplementedError


def run_layernorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
    bias: Float[Tensor, " d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Load LayerNorm parameters into your own LayerNorm module and return its forward output.
    """
    raise NotImplementedError


def run_gelu(
    in_features: Float[Tensor, " ..."],
) -> Float[Tensor, " ..."]:
    """
    Run your GELU module/function on the input tensor.
    """
    raise NotImplementedError


def run_mlp(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w1_bias: Float[Tensor, " d_ff"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w2_bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Load weights into your MLP module and return its forward output.
    """
    raise NotImplementedError


def run_patchify(
    images: Float[Tensor, " batch channels height width"],
    patch_size: int,
) -> Float[Tensor, " batch num_patches (patch_h patch_w channels)"]:
    """
    Run your patchify implementation on image tensors.

    Required output layout:
    - Shape: (batch, num_patches, patch_dim), where patch_dim = patch_size * patch_size * channels.
    - Patch order: row-major over the patch grid (top-left -> top-right -> next row).
    - Within each patch vector: flatten in (patch_h, patch_w, channel) order.
      This means channel is the fastest-changing index.
    """
    raise NotImplementedError


def run_unpatchify(
    patches: Float[Tensor, " batch num_patches (patch_h patch_w channels)"],
    patch_size: int,
    channels: int = 3,
) -> Float[Tensor, " batch channels height width"]:
    """
    Run your unpatchify implementation on patch tensors.

    Input `patches` must follow the same layout as `run_patchify`:
    - num_patches uses a square patch grid: `num_patches = hh * ww` with `hh == ww`.
    - patch_dim = patch_size * patch_size * channels.
    - each patch vector is flattened in (patch_h, patch_w, channel) order.
    """
    raise NotImplementedError


def run_patch_embed(
    img_size: int,
    patch_size: int,
    in_chans: int,
    embed_dim: int,
    proj_weight: Float[Tensor, " embed_dim in_chans patch_size patch_size"],
    proj_bias: Float[Tensor, " embed_dim"],
    images: Float[Tensor, " batch in_chans img_size img_size"],
) -> tuple[
    Float[Tensor, " batch num_patches embed_dim"], Int[Tensor, " batch num_patches 2"]
]:
    """
    Run your PatchEmbed implementation.

    Expected PatchEmbed API:
    - constructor: PatchEmbed(img_size, patch_size, in_chans, embed_dim)
    - module has conv projection params named `proj.weight` and `proj.bias`
    - forward(images) returns:
      1) patch embeddings of shape (batch, num_patches, embed_dim)
      2) patch positions of shape (batch, num_patches, 2), ordered as (y, x)
    """
    raise NotImplementedError


def run_random_mask_patch(
    patch_features: Float[Tensor, " batch num_patches d_model"],
    mask_ratio: float,
) -> Bool[Tensor, " batch num_patches"]:
    """
    Run your random patch masking implementation.

    Contract:
    - Return boolean mask with shape (batch, num_patches).
    - True means the patch is masked out.
    - Number of True values per sample should be `int(mask_ratio * num_patches)`.
    """
    raise NotImplementedError


def run_rope1d(
    in_query_or_key: Float[Tensor, " batch heads seq_len d_head"],
    token_positions: torch.Tensor,
    freq: float = 100.0,
    f0: float = 1.0,
) -> Float[Tensor, " batch heads seq_len d_head"]:
    """
    Run your 1D RoPE implementation.

    Contract:
    - `in_query_or_key` shape is (B, H, N, D), with even D.
    - `token_positions` shape is (B, N), where each value is a token index.
    - RoPE is applied on 2D pairs inside the full head dimension.
    """
    raise NotImplementedError


def run_rope2d(
    in_query_or_key: Float[Tensor, " batch heads seq_len d_head"],
    token_positions: torch.Tensor,
    freq: float = 100.0,
    f0: float = 1.0,
) -> Float[Tensor, " batch heads seq_len d_head"]:
    """
    Run your 2D RoPE implementation.

    Contract:
    - `in_query_or_key` shape is (B, H, N, D), with even D.
    - `token_positions` shape is (B, N, 2), where the last dim is (y, x).
    - RoPE is applied separately on y-half and x-half of the head dimension.
    """
    raise NotImplementedError


def run_softmax(
    in_features: Float[Tensor, " ..."],
    dim: int,
) -> Float[Tensor, " ..."]:
    """
    Run your numerically stable softmax implementation along `dim`.
    """
    raise NotImplementedError


def run_scaled_dot_product_attention(
    q: Float[Tensor, " ... q_len d_k"],
    k: Float[Tensor, " ... k_len d_k"],
    v: Float[Tensor, " ... k_len d_v"],
) -> Float[Tensor, " ... q_len d_v"]:
    """
    Run your scaled dot-product attention implementation.
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    qkv_proj_weight: Float[Tensor, " three_d_model d_model"],
    qkv_proj_bias: Float[Tensor, " three_d_model"],
    out_proj_weight: Float[Tensor, " d_model d_model"],
    out_proj_bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " batch seq_len d_model"],
) -> Float[Tensor, " batch seq_len d_model"]:
    """
    Run your multi-head self-attention implementation.
    """
    raise NotImplementedError


def run_multihead_cross_attention(
    d_model: int,
    num_heads: int,
    projq_weight: Float[Tensor, " d_model d_model"],
    projq_bias: Float[Tensor, " d_model"],
    projk_weight: Float[Tensor, " d_model d_model"],
    projk_bias: Float[Tensor, " d_model"],
    projv_weight: Float[Tensor, " d_model d_model"],
    projv_bias: Float[Tensor, " d_model"],
    out_proj_weight: Float[Tensor, " d_model d_model"],
    out_proj_bias: Float[Tensor, " d_model"],
    query: Float[Tensor, " batch q_len d_model"],
    key: Float[Tensor, " batch k_len d_model"],
    value: Float[Tensor, " batch k_len d_model"],
) -> Float[Tensor, " batch q_len d_model"]:
    """
    Run your multi-head cross-attention implementation.
    """
    raise NotImplementedError


def run_encoder_block(
    d_model: int,
    num_heads: int,
    norm1_weight: Float[Tensor, " d_model"],
    norm1_bias: Float[Tensor, " d_model"],
    attn_qkv_weight: Float[Tensor, " three_d_model d_model"],
    attn_qkv_bias: Float[Tensor, " three_d_model"],
    attn_out_weight: Float[Tensor, " d_model d_model"],
    attn_out_bias: Float[Tensor, " d_model"],
    norm2_weight: Float[Tensor, " d_model"],
    norm2_bias: Float[Tensor, " d_model"],
    mlp_fc1_weight: Float[Tensor, " d_ff d_model"],
    mlp_fc1_bias: Float[Tensor, " d_ff"],
    mlp_fc2_weight: Float[Tensor, " d_model d_ff"],
    mlp_fc2_bias: Float[Tensor, " d_model"],
    x: Float[Tensor, " batch seq_len d_model"],
    token_positions: Int[Tensor, " batch seq_len 2"],
    eps: float = 1e-6,
) -> Float[Tensor, " batch seq_len d_model"]:
    """
    Run your encoder Block implementation (self-attention + MLP with residuals).
    """
    raise NotImplementedError


def run_cat_block(
    d_model: int,
    num_heads: int,
    norm_mem_weight: Float[Tensor, " d_model"],
    norm_mem_bias: Float[Tensor, " d_model"],
    norm1_weight: Float[Tensor, " d_model"],
    norm1_bias: Float[Tensor, " d_model"],
    attn_qkv_weight: Float[Tensor, " three_d_model d_model"],
    attn_qkv_bias: Float[Tensor, " three_d_model"],
    attn_out_weight: Float[Tensor, " d_model d_model"],
    attn_out_bias: Float[Tensor, " d_model"],
    norm2_weight: Float[Tensor, " d_model"],
    norm2_bias: Float[Tensor, " d_model"],
    mlp_fc1_weight: Float[Tensor, " d_ff d_model"],
    mlp_fc1_bias: Float[Tensor, " d_ff"],
    mlp_fc2_weight: Float[Tensor, " d_model d_ff"],
    mlp_fc2_bias: Float[Tensor, " d_model"],
    x: Float[Tensor, " batch n1 d_model"],
    mem: Float[Tensor, " batch n2 d_model"],
    pos_x: Int[Tensor, " batch n1 2"],
    pos_mem: Int[Tensor, " batch n2 2"],
    eps: float = 1e-6,
) -> tuple[
    Float[Tensor, " batch n1 d_model"],
    Float[Tensor, " batch n2 d_model"],
]:
    """
    Run your CatDecoderBlock implementation.
    """
    raise NotImplementedError


def run_cross_block(
    d_model: int,
    num_heads: int,
    norm1_weight: Float[Tensor, " d_model"],
    norm1_bias: Float[Tensor, " d_model"],
    self_attn_qkv_weight: Float[Tensor, " three_d_model d_model"],
    self_attn_qkv_bias: Float[Tensor, " three_d_model"],
    self_attn_out_weight: Float[Tensor, " d_model d_model"],
    self_attn_out_bias: Float[Tensor, " d_model"],
    norm2_weight: Float[Tensor, " d_model"],
    norm2_bias: Float[Tensor, " d_model"],
    cross_attn_q_weight: Float[Tensor, " d_model d_model"],
    cross_attn_q_bias: Float[Tensor, " d_model"],
    cross_attn_k_weight: Float[Tensor, " d_model d_model"],
    cross_attn_k_bias: Float[Tensor, " d_model"],
    cross_attn_v_weight: Float[Tensor, " d_model d_model"],
    cross_attn_v_bias: Float[Tensor, " d_model"],
    cross_attn_out_weight: Float[Tensor, " d_model d_model"],
    cross_attn_out_bias: Float[Tensor, " d_model"],
    norm3_weight: Float[Tensor, " d_model"],
    norm3_bias: Float[Tensor, " d_model"],
    mlp_fc1_weight: Float[Tensor, " d_ff d_model"],
    mlp_fc1_bias: Float[Tensor, " d_ff"],
    mlp_fc2_weight: Float[Tensor, " d_model d_ff"],
    mlp_fc2_bias: Float[Tensor, " d_model"],
    norm_y_weight: Float[Tensor, " d_model"],
    norm_y_bias: Float[Tensor, " d_model"],
    x: Float[Tensor, " batch n1 d_model"],
    mem: Float[Tensor, " batch n2 d_model"],
    pos_x: Int[Tensor, " batch n1 2"],
    pos_mem: Int[Tensor, " batch n2 2"],
    eps: float = 1e-6,
) -> tuple[
    Float[Tensor, " batch n1 d_model"],
    Float[Tensor, " batch n2 d_model"],
]:
    """
    Run your CrossDecoderBlock implementation.
    """
    raise NotImplementedError


def build_croco_model(
    img_size: int,
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
    use_rope: bool = True,
    rope_freq: float = 100.0,
    rope_f0: float = 1.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    """
    Build your CroCo model instance.
    """
    raise NotImplementedError


def run_croco_forward(
    model: torch.nn.Module,
    target_image: Float[Tensor, " batch channels height width"],
    reference_image: Float[Tensor, " batch channels height width"],
    mask: Bool[Tensor, " batch num_patches"] | None = None,
) -> tuple[
    Float[Tensor, " batch num_patches patch_dim"],
    Bool[Tensor, " batch num_patches"],
    Float[Tensor, " batch num_patches patch_dim"],
]:
    """
    Run a CroCo model forward pass.
    """
    raise NotImplementedError


def croco_encode_image(
    model: torch.nn.Module,
    image: Float[Tensor, " batch channels height width"],
    apply_mask: bool,
    mask: Bool[Tensor, " batch num_patches"] | None = None,
) -> tuple[
    Float[Tensor, " batch visible_or_all d_model"],
    Tensor,
    Bool[Tensor, " batch num_patches"],
]:
    """
    Run CroCo's image encoder path and return `(encoded, positions, used_mask)`.
    """
    raise NotImplementedError


def build_adamw_optimizer(params, **kwargs):
    """
    Build student AdamW optimizer instance.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
) -> float:
    """
    Run student cosine warmup schedule function and return a float multiplier.
    """
    raise NotImplementedError
