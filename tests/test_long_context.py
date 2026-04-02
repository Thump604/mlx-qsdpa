"""Long context tests: 4K+, threshold boundary."""

import mlx.core as mx
import pytest
from conftest import make_qkv, reference_sdpa
from mlx_qsdpa import quantized_sdpa

TOLERANCE = 0.02  # Slightly relaxed for long context (more accumulation error)


class TestLongContext:

    @pytest.mark.parametrize("N", [4096, 8192])
    def test_context_lengths(self, seed, N):
        """Qwen3.5 dims at 4K and 8K context.

        N=4096 hits the default threshold exactly (single-pass).
        N=8192 exceeds the default threshold (two-pass).
        """
        B, H_q, H_kv, D = 1, 16, 2, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4
        )
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"N={N}: max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_threshold_boundary(self, seed):
        """N=4096 at threshold boundary: compare single-pass, two-pass, and reference.

        threshold=N+1 forces single-pass; threshold=N forces two-pass (N > threshold
        is the two-pass condition in dispatch).
        """
        B, H_q, H_kv, N, D = 1, 16, 2, 4096, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4
        )

        # threshold=N+1 -> N <= threshold -> single-pass
        out_single = quantized_sdpa(
            q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32, threshold=N + 1
        )
        # threshold=N-1 -> N > threshold -> two-pass
        out_two = quantized_sdpa(
            q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32, threshold=N - 1
        )
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out_single, out_two, ref)

        diff_single_ref = mx.max(mx.abs(out_single - ref)).item()
        diff_two_ref = mx.max(mx.abs(out_two - ref)).item()
        diff_single_two = mx.max(mx.abs(out_single - out_two)).item()

        assert diff_single_ref < TOLERANCE, (
            f"single-pass vs reference: max diff {diff_single_ref} "
            f"exceeds tolerance {TOLERANCE}"
        )
        assert diff_two_ref < TOLERANCE, (
            f"two-pass vs reference: max diff {diff_two_ref} "
            f"exceeds tolerance {TOLERANCE}"
        )
        assert diff_single_two < TOLERANCE, (
            f"single-pass vs two-pass: max diff {diff_single_two} "
            f"exceeds tolerance {TOLERANCE}"
        )

    @pytest.mark.slow
    def test_32k_context(self, seed):
        """32K context with forced two-pass (threshold=4096)."""
        B, H_q, H_kv, N, D = 1, 16, 2, 32768, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4
        )
        out = quantized_sdpa(
            q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32, threshold=4096
        )
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, (
            f"32K context: max diff {diff} exceeds tolerance {TOLERANCE}"
        )
