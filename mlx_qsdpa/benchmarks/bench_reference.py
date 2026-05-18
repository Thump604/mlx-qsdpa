"""Reference attention paths used by mlx-qsdpa benchmarks."""

import mlx.core as mx


def mlx_lm_quantized_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    scale: float,
    bits: int = 4,
    group_size: int = 32,
    mask=None,
) -> mx.array:
    """Reimplementation of mlx_lm.models.base.quantized_scaled_dot_product_attention.

    Source: mlx-lm v0.31.1, mlx_lm/models/base.py:64-105.

    Two mx.quantized_matmul calls (Q*K^T, scores*V) + softmax.
    This is the path Package 1's fused kernel aims to beat.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tuple(mx.expand_dims(x, axis=-3) for x in q_keys)
        q_values = tuple(mx.expand_dims(x, axis=-3) for x in q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out
