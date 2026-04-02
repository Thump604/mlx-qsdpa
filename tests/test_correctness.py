"""Correctness tests for single-pass quantized SDPA kernel."""

import mlx.core as mx
import pytest
from conftest import make_qkv, reference_sdpa
from mlx_qsdpa import quantized_sdpa

TOLERANCE = 0.01


class TestSinglePassCorrectness:

    def test_basic_4bit_d128(self, seed):
        B, H_q, H_kv, N, D = 1, 1, 1, 32, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_basic_4bit_d256(self, seed):
        B, H_q, H_kv, N, D = 1, 16, 2, 64, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_medium_context_1024(self, seed):
        B, H_q, H_kv, N, D = 1, 16, 2, 1024, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_single_key(self, seed):
        B, H_q, H_kv, N, D = 1, 4, 1, 1, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"
