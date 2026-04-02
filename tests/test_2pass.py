"""Correctness tests for two-pass quantized SDPA kernel."""

import mlx.core as mx
import pytest
from conftest import make_qkv, reference_sdpa
from mlx_qsdpa import quantized_sdpa

TOLERANCE = 0.01


class TestTwoPassCorrectness:

    def test_forced_2pass_short(self, seed):
        """Force two-pass on short context by setting threshold=0."""
        B, H_q, H_kv, N, D = 1, 4, 1, 64, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32, threshold=0)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_2pass_qwen_dims(self, seed):
        """Qwen3.5 dimensions with forced two-pass."""
        B, H_q, H_kv, N, D = 1, 16, 2, 512, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32, threshold=0)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_2pass_matches_1pass(self, seed):
        """Verify two-pass and single-pass produce the same result."""
        B, H_q, H_kv, N, D = 1, 16, 2, 256, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out_1pass = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, threshold=999999)
        out_2pass = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, threshold=0)
        mx.eval(out_1pass, out_2pass)
        diff = mx.max(mx.abs(out_1pass - out_2pass)).item()
        assert diff < TOLERANCE, f"1pass vs 2pass diff {diff} exceeds tolerance {TOLERANCE}"
