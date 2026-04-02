from mlx_qsdpa.dispatch import quantized_sdpa
from mlx_qsdpa.cache import QuantizedSDPACache

import mlx.core as mx


def cache_sdpa(
    q,
    k_quant,
    v_quant,
    cache,
    scale=None,
    mask=None,
    crossover=16384,
):
    """Attention with dynamic dispatch: FP16 below crossover, fused above.

    Below the crossover sequence length, dequantizes K/V and uses
    mx.fast.scaled_dot_product_attention (faster at short context).
    At or above crossover, uses the fused quantized_sdpa kernel
    (1.7x faster than mx.quantized_matmul at 128K with GQA).

    For prefill (qL > 1), always dequantizes regardless of N
    (the fused kernel is decode-only).

    Args:
        q:        (B, H_q, qL, D) float16 query
        k_quant:  (packed_uint32, scales_fp16, biases_fp16) from cache
        v_quant:  (packed_uint32, scales_fp16, biases_fp16) from cache
        cache:    QuantizedSDPACache instance (provides bits, group_size)
        scale:    attention scale (default: 1/sqrt(D))
        mask:     attention mask (default: None)
        crossover: sequence length at which to switch from FP16 to fused.
                   Default 16384 (empirically determined on M2 Ultra).
    """
    D = q.shape[-1]
    qL = q.shape[2]
    N = k_quant[0].shape[2]

    if scale is None:
        scale = D ** -0.5

    # Prefill or short context: dequantize + FP16 SDPA
    if qL > 1 or N < crossover:
        k_fp16 = mx.dequantize(
            k_quant[0], k_quant[1], k_quant[2],
            group_size=cache.group_size, bits=cache.bits,
        )
        v_fp16 = mx.dequantize(
            v_quant[0], v_quant[1], v_quant[2],
            group_size=cache.group_size, bits=cache.bits,
        )
        return mx.fast.scaled_dot_product_attention(
            q, k_fp16, v_fp16, scale=scale, mask=mask
        )

    # Long context decode: fused kernel
    return quantized_sdpa(
        q, k_quant[0], v_quant[0],
        k_quant[1], v_quant[1], k_quant[2], v_quant[2],
        scale=scale, mask=mask,
        bits=cache.bits, group_size=cache.group_size,
    )


__all__ = ["quantized_sdpa", "QuantizedSDPACache", "cache_sdpa"]
