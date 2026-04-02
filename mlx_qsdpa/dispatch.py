"""Quantized SDPA dispatch: routes to single-pass or two-pass kernel."""

import mlx.core as mx

from mlx_qsdpa.kernels import vector_kernel, pass1_kernel, pass2_kernel


def quantized_sdpa(
    q,
    packed_k,
    packed_v,
    k_scales,
    v_scales,
    k_biases=None,
    v_biases=None,
    scale=None,
    mask=None,
    bits=4,
    group_size=32,
    threshold=4096,
):
    """Quantized scaled dot-product attention for decode (qL=1).

    Args:
        q:         (B, H_q, 1, D) float16/bfloat16 query tensor
        packed_k:  (B, H_kv, N, pack_D) uint32 quantized keys
        packed_v:  (B, H_kv, N, pack_D) uint32 quantized values
        k_scales:  (B, H_kv, N, num_groups) float16 key scales
        v_scales:  (B, H_kv, N, num_groups) float16 value scales
        k_biases:  (B, H_kv, N, num_groups) float16 key biases, or None for symmetric
        v_biases:  (B, H_kv, N, num_groups) float16 value biases, or None for symmetric
        scale:     float, attention scale (default: 1/sqrt(D))
        mask:      not yet supported (placeholder)
        bits:      quantization bits (4 or 8)
        group_size: elements per quantization group (32, 64, 128)
        threshold: N above which to use two-pass kernel (not yet implemented)

    Returns:
        (B, H_q, 1, D) float16/bfloat16 attention output
    """
    # ---- validate ----
    if q.ndim != 4:
        raise ValueError(f"q must be 4D (B, H_q, qL, D), got ndim={q.ndim}")
    B, H_q, qL, D = q.shape

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if D % 32 != 0:
        raise ValueError(f"D must be multiple of 32, got {D}")

    elems_per_int = 32 // bits
    pack_D = D // elems_per_int
    num_groups = D // group_size

    if packed_k.ndim != 4 or packed_v.ndim != 4:
        raise ValueError("packed_k and packed_v must be 4D (B, H_kv, N, pack_D)")

    _, H_kv, N, actual_pack_D = packed_k.shape
    if actual_pack_D != pack_D:
        raise ValueError(
            f"packed_k last dim {actual_pack_D} != expected pack_D={pack_D} "
            f"for bits={bits}, D={D}"
        )
    if H_q % H_kv != 0:
        raise ValueError(f"H_q ({H_q}) must be divisible by H_kv ({H_kv})")

    # ---- prefill fallback: dequantize + standard SDPA ----
    if qL > 1:
        k_ref = mx.dequantize(
            packed_k, k_scales,
            k_biases if k_biases is not None else mx.zeros_like(k_scales),
            group_size=group_size, bits=bits,
        )
        v_ref = mx.dequantize(
            packed_v, v_scales,
            v_biases if v_biases is not None else mx.zeros_like(v_scales),
            group_size=group_size, bits=bits,
        )
        if scale is None:
            scale = D ** -0.5
        return mx.fast.scaled_dot_product_attention(q, k_ref, v_ref, scale=scale)

    # ---- decode path (qL == 1) ----
    if scale is None:
        scale = D ** -0.5

    has_biases = k_biases is not None and v_biases is not None

    # Create zero biases for symmetric quantization
    if not has_biases:
        k_biases = mx.zeros_like(k_scales)
        v_biases = mx.zeros_like(v_scales)

    gqa_factor = H_q // H_kv

    if N <= threshold:
        return _dispatch_vector(
            q, packed_k, packed_v,
            k_scales, v_scales, k_biases, v_biases,
            B, H_q, H_kv, N, D, pack_D, num_groups,
            gqa_factor, scale, bits, group_size, has_biases,
        )
    else:
        return _dispatch_2pass(
            q, packed_k, packed_v,
            k_scales, v_scales, k_biases, v_biases,
            B, H_q, H_kv, N, D, pack_D, num_groups,
            gqa_factor, scale, bits, group_size, has_biases,
        )


def _dispatch_vector(
    q, packed_k, packed_v,
    k_scales, v_scales, k_biases, v_biases,
    B, H_q, H_kv, N, D, pack_D, num_groups,
    gqa_factor, scale, bits, group_size, has_biases,
):
    """Dispatch single-pass vector kernel."""
    dtype = q.dtype

    # Reshape to flat layout for the kernel
    # queries: (B, H_q, 1, D) -> (B*H_q, D)
    queries_flat = q.reshape(B * H_q, D)

    # packed_keys: (B, H_kv, N, pack_D) -> (B*H_kv, N, pack_D)
    pk_flat = packed_k.reshape(B * H_kv, N, pack_D)
    pv_flat = packed_v.reshape(B * H_kv, N, pack_D)

    # scales/biases: (B, H_kv, N, num_groups) -> (B*H_kv, N, num_groups)
    ks_flat = k_scales.reshape(B * H_kv, N, num_groups)
    vs_flat = v_scales.reshape(B * H_kv, N, num_groups)
    kb_flat = k_biases.reshape(B * H_kv, N, num_groups)
    vb_flat = v_biases.reshape(B * H_kv, N, num_groups)

    # Scalar inputs
    gqa_scalar = mx.array(gqa_factor, dtype=mx.int32)
    n_scalar = mx.array(N, dtype=mx.int32)
    scale_scalar = mx.array(scale, dtype=mx.float32)

    # Grid: one threadgroup per (batch, q_head).
    # Each threadgroup = 32 simdgroups * 32 threads = 1024 threads
    n_threadgroups = B * H_q
    threads_per_tg = 1024  # 32 * 32
    grid = (n_threadgroups * threads_per_tg, 1, 1)
    threadgroup = (threads_per_tg, 1, 1)

    # Template params
    template = [
        ("T", dtype),
        ("D", D),
        ("bits", bits),
        ("group_size", group_size),
        ("has_bias", has_biases),
    ]

    outputs = vector_kernel(
        inputs=[
            queries_flat,
            pk_flat, pv_flat,
            ks_flat, vs_flat,
            kb_flat, vb_flat,
            gqa_scalar, n_scalar, scale_scalar,
        ],
        output_shapes=[(B * H_q, D)],
        output_dtypes=[dtype],
        grid=grid,
        threadgroup=threadgroup,
        template=template,
    )

    # Reshape output: (B*H_q, D) -> (B, H_q, 1, D)
    return outputs[0].reshape(B, H_q, 1, D)


def _dispatch_2pass(
    q, packed_k, packed_v,
    k_scales, v_scales, k_biases, v_biases,
    B, H_q, H_kv, N, D, pack_D, num_groups,
    gqa_factor, scale, bits, group_size, has_biases,
):
    """Dispatch two-pass split-K kernel for long context decode."""
    dtype = q.dtype

    # ---- compute num_blocks ----
    num_blocks = min(1024, max(32, (N + 31) // 32))
    num_blocks = ((num_blocks + 31) // 32) * 32  # round to multiple of 32

    # ---- flatten inputs ----
    queries_flat = q.reshape(B * H_q, D)
    pk_flat = packed_k.reshape(B * H_kv, N, pack_D)
    pv_flat = packed_v.reshape(B * H_kv, N, pack_D)
    ks_flat = k_scales.reshape(B * H_kv, N, num_groups)
    vs_flat = v_scales.reshape(B * H_kv, N, num_groups)
    kb_flat = k_biases.reshape(B * H_kv, N, num_groups)
    vb_flat = v_biases.reshape(B * H_kv, N, num_groups)

    # ---- scalar inputs ----
    gqa_scalar = mx.array(gqa_factor, dtype=mx.int32)
    n_scalar = mx.array(N, dtype=mx.int32)
    scale_scalar = mx.array(scale, dtype=mx.float32)
    hq_scalar = mx.array(H_q, dtype=mx.int32)
    hkv_scalar = mx.array(H_kv, dtype=mx.int32)
    nb_scalar = mx.array(num_blocks, dtype=mx.int32)

    # ---- pass 1: per-block partial attention ----
    # Grid: one threadgroup per (kv_head, batch, block)
    # Each threadgroup has gqa_factor simdgroups * 32 threads
    # Grid is specified in total threads per dimension, so multiply x by threads_per_tg
    threads_per_tg_p1 = 32 * gqa_factor
    grid_p1 = (H_kv * threads_per_tg_p1, B, num_blocks)
    threadgroup_p1 = (threads_per_tg_p1, 1, 1)

    template_p1 = [
        ("T", dtype),
        ("D", D),
        ("bits", bits),
        ("group_size", group_size),
        ("has_bias", has_biases),
    ]

    total_blocks = B * H_q * num_blocks

    p1_outputs = pass1_kernel(
        inputs=[
            queries_flat,
            pk_flat, pv_flat,
            ks_flat, vs_flat,
            kb_flat, vb_flat,
            gqa_scalar, n_scalar, scale_scalar,
            hq_scalar, hkv_scalar, nb_scalar,
        ],
        output_shapes=[
            (total_blocks, D),   # partials
            (total_blocks,),     # sums
            (total_blocks,),     # maxs
        ],
        output_dtypes=[dtype, mx.float32, mx.float32],
        grid=grid_p1,
        threadgroup=threadgroup_p1,
        template=template_p1,
    )

    partials, sums, maxs = p1_outputs

    # ---- pass 2: reduce blocks ----
    n_tg_p2 = B * H_q
    threads_per_tg_p2 = 1024
    grid_p2 = (n_tg_p2 * threads_per_tg_p2, 1, 1)
    threadgroup_p2 = (threads_per_tg_p2, 1, 1)

    template_p2 = [
        ("T", dtype),
        ("D", D),
    ]

    p2_outputs = pass2_kernel(
        inputs=[partials, sums, maxs, nb_scalar],
        output_shapes=[(B * H_q, D)],
        output_dtypes=[dtype],
        grid=grid_p2,
        threadgroup=threadgroup_p2,
        template=template_p2,
    )

    return p2_outputs[0].reshape(B, H_q, 1, D)
