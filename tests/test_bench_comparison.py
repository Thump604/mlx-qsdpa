"""Tests for bench_comparison: verify inline mlx-lm reimplementation correctness."""

import mlx.core as mx
import pytest


TOLERANCE = 1e-5  # should be exact match (same ops)


class TestInlineQuantizedSDPA:
    def test_matches_raw_ops_non_gqa(self):
        """Inline reimplementation matches manual quantized_matmul x2 + softmax."""
        from mlx_qsdpa.bench_comparison import mlx_lm_quantized_sdpa

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        bits, gs = 4, 32
        scale = D ** -0.5

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        pk, ks, kb = mx.quantize(k_fp16, group_size=gs, bits=bits)
        pv, vs, vb = mx.quantize(v_fp16, group_size=gs, bits=bits)
        mx.eval(q, pk, pv, ks, vs, kb, vb)

        out = mlx_lm_quantized_sdpa(q, (pk, ks, kb), (pv, vs, vb),
                                     scale=scale, bits=bits, group_size=gs)

        # Manual reference: same ops explicitly
        scores = mx.quantized_matmul(q * scale, pk, ks, kb,
                                     transpose=True, group_size=gs, bits=bits)
        scores = mx.softmax(scores, axis=-1, precise=True)
        ref = mx.quantized_matmul(scores, pv, vs, vb,
                                  transpose=False, group_size=gs, bits=bits)

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"non-GQA diff {diff}"

    def test_matches_raw_ops_gqa(self):
        """Inline reimplementation handles GQA repeat correctly."""
        from mlx_qsdpa.bench_comparison import mlx_lm_quantized_sdpa

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 32, 2, 64, 256
        bits, gs = 4, 32
        scale = D ** -0.5

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        pk, ks, kb = mx.quantize(k_fp16, group_size=gs, bits=bits)
        pv, vs, vb = mx.quantize(v_fp16, group_size=gs, bits=bits)
        mx.eval(q, pk, pv, ks, vs, kb, vb)

        out = mlx_lm_quantized_sdpa(q, (pk, ks, kb), (pv, vs, vb),
                                     scale=scale, bits=bits, group_size=gs)

        # Reference: dequant + FP16 SDPA (the ground truth, different code path)
        k_deq = mx.dequantize(pk, ks, kb, group_size=gs, bits=bits)
        v_deq = mx.dequantize(pv, vs, vb, group_size=gs, bits=bits)
        ref = mx.fast.scaled_dot_product_attention(q, k_deq, v_deq, scale=scale)

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        # Looser tolerance: quantized_matmul vs dequant+sdpa may differ slightly
        assert diff < 0.01, f"GQA diff {diff}"

    def test_all_three_paths_agree(self):
        """FP16, quantized_matmul, and fused kernel all produce similar results."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.bench_comparison import mlx_lm_quantized_sdpa

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        bits, gs = 4, 32
        scale = D ** -0.5

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        pk, ks, kb = mx.quantize(k_fp16, group_size=gs, bits=bits)
        pv, vs, vb = mx.quantize(v_fp16, group_size=gs, bits=bits)
        k_deq = mx.dequantize(pk, ks, kb, group_size=gs, bits=bits)
        v_deq = mx.dequantize(pv, vs, vb, group_size=gs, bits=bits)
        mx.eval(q, pk, pv, ks, vs, kb, vb, k_deq, v_deq)

        # Path 1: FP16 SDPA on dequantized K/V
        out_fp16 = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=scale
        )
        # Path 2: mlx-lm quantized_matmul path
        out_qmm = mlx_lm_quantized_sdpa(
            q, (pk, ks, kb), (pv, vs, vb),
            scale=scale, bits=bits, group_size=gs,
        )
        # Path 3: fused kernel
        out_fused = quantized_sdpa(
            q, pk, pv, ks, vs, kb, vb,
            bits=bits, group_size=gs,
        )

        mx.eval(out_fp16, out_qmm, out_fused)

        diff_qmm = mx.max(mx.abs(out_fp16 - out_qmm)).item()
        diff_fused = mx.max(mx.abs(out_fp16 - out_fused)).item()
        assert diff_qmm < 0.01, f"qmm vs fp16: {diff_qmm}"
        assert diff_fused < 0.01, f"fused vs fp16: {diff_fused}"


class TestBandwidthCalculation:
    def test_decode_bytes_fp16(self):
        """FP16 decode byte count: Q + K + V read, output written."""
        from mlx_qsdpa.bench_comparison import compute_bytes

        B, H_q, H_kv, N, D, qL = 1, 2, 2, 4096, 256, 1
        read_b, write_b = compute_bytes(
            B=B, H_q=H_q, H_kv=H_kv, N=N, D=D, qL=qL,
            path="fp16", bits=4, group_size=32,
        )
        # Q: B*H_q*qL*D*2 = 1*2*1*256*2 = 1024
        # K: B*H_kv*N*D*2 = 1*2*4096*256*2 = 4194304
        # V: same = 4194304
        assert read_b == 1024 + 4194304 + 4194304
        # Output: B*H_q*qL*D*2 = 1024
        assert write_b == 1024

    def test_decode_bytes_quantized(self):
        """Quantized decode byte count: Q + packed K/V + scales + biases."""
        from mlx_qsdpa.bench_comparison import compute_bytes

        B, H_q, H_kv, N, D, qL = 1, 2, 2, 4096, 256, 1
        bits, gs = 4, 32
        read_b, write_b = compute_bytes(
            B=B, H_q=H_q, H_kv=H_kv, N=N, D=D, qL=qL,
            path="quantized", bits=bits, group_size=gs,
        )
        elems_per_int = 32 // bits  # 8
        pack_D = D // elems_per_int  # 32
        num_groups = D // gs  # 8
        # Q: 1*2*1*256*2 = 1024
        q_bytes = B * H_q * qL * D * 2
        # packed K: B*H_kv*N*pack_D*4 = 1*2*4096*32*4 = 1048576
        # scales K: B*H_kv*N*num_groups*2 = 1*2*4096*8*2 = 131072
        # biases K: same = 131072
        # V: same as K
        kv_per = B * H_kv * N * (pack_D * 4 + num_groups * 2 + num_groups * 2)
        assert read_b == q_bytes + kv_per * 2
        assert write_b == B * H_q * qL * D * 2

    def test_bandwidth_from_known_values(self):
        """Known bytes + known time = known GB/s."""
        from mlx_qsdpa.bench_comparison import compute_bandwidth_gbps

        total_bytes = 800_000_000  # 800 MB
        median_us = 1000.0  # 1 ms
        # 800 MB / 1 ms = 800 GB/s
        gbps = compute_bandwidth_gbps(total_bytes, median_us)
        assert abs(gbps - 800.0) < 0.01
