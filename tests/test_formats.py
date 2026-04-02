"""Test all supported quantization formats."""

import mlx.core as mx
import pytest
from conftest import make_qkv, reference_sdpa
from mlx_qsdpa import quantized_sdpa

TOLERANCE = 0.01
DIMS = (1, 16, 2, 128, 256)  # B, H_q, H_kv, N, D


class TestBitWidths:

    def test_8bit_asymmetric(self, seed):
        B, H_q, H_kv, N, D = DIMS
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=8, group_size=32
        )
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=8, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"8-bit asymmetric max diff {diff} exceeds {TOLERANCE}"

    def test_4bit_asymmetric(self, seed):
        B, H_q, H_kv, N, D = DIMS
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4, group_size=32
        )
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"4-bit asymmetric max diff {diff} exceeds {TOLERANCE}"


class TestSymmetric:

    def test_4bit_symmetric(self, seed):
        B, H_q, H_kv, N, D = DIMS
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4, group_size=32, symmetric=True
        )
        # kb and vb are None when symmetric=True
        assert kb is None and vb is None
        out = quantized_sdpa(
            q, pk, pv, ks, vs, k_biases=None, v_biases=None, bits=4, group_size=32
        )
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"4-bit symmetric max diff {diff} exceeds {TOLERANCE}"

    def test_8bit_symmetric(self, seed):
        B, H_q, H_kv, N, D = DIMS
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=8, group_size=32, symmetric=True
        )
        assert kb is None and vb is None
        out = quantized_sdpa(
            q, pk, pv, ks, vs, k_biases=None, v_biases=None, bits=8, group_size=32
        )
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"8-bit symmetric max diff {diff} exceeds {TOLERANCE}"


class TestGroupSizes:

    @pytest.mark.parametrize("gs", [32, 64, 128])
    def test_group_size_4bit(self, seed, gs):
        B, H_q, H_kv, N, D = DIMS
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(
            B, H_q, H_kv, N, D, bits=4, group_size=gs
        )
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=gs)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, (
            f"4-bit group_size={gs} max diff {diff} exceeds {TOLERANCE}"
        )
