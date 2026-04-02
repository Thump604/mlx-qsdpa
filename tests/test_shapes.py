"""Test shape variants: GQA, head_dim, edge cases."""

import mlx.core as mx
import pytest
from conftest import make_qkv, reference_sdpa
from mlx_qsdpa import quantized_sdpa

TOLERANCE = 0.01


class TestGQA:

    def test_no_gqa(self, seed):
        """H_q == H_kv: no grouping, 1:1 head mapping."""
        B, H_q, H_kv, N, D = 1, 4, 4, 64, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_gqa_8x(self, seed):
        """H_q=16, H_kv=2: Qwen3.5-style 8x GQA grouping."""
        B, H_q, H_kv, N, D = 1, 16, 2, 64, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_gqa_4x(self, seed):
        """H_q=4, H_kv=1: all query heads share a single KV head."""
        B, H_q, H_kv, N, D = 1, 4, 1, 64, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"


class TestHeadDim:

    def test_d128(self, seed):
        """Standard 128-dim heads."""
        B, H_q, H_kv, N, D = 1, 4, 2, 64, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_d256(self, seed):
        """256-dim heads (Qwen3.5-122B full-attention layers)."""
        B, H_q, H_kv, N, D = 1, 4, 2, 64, 256
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"


class TestEdgeCases:

    def test_single_key_n1(self, seed):
        """N=1: degenerate sequence of a single key/value."""
        B, H_q, H_kv, N, D = 1, 4, 2, 1, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_n32_exact_simdgroup(self, seed):
        """N=32: exactly one key per simdgroup lane, boundary condition."""
        B, H_q, H_kv, N, D = 1, 4, 2, 32, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"

    def test_degenerate_scales_no_nan(self, seed):
        """Zero scales and biases: dequant collapses to zero, result must not be NaN."""
        B, H_q, H_kv, N, D = 1, 4, 2, 32, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        # Overwrite scales and biases with zeros so all dequantized values are 0
        ks_zero = mx.zeros_like(ks)
        vs_zero = mx.zeros_like(vs)
        kb_zero = mx.zeros_like(kb)
        vb_zero = mx.zeros_like(vb)
        out = quantized_sdpa(q, pk, pv, ks_zero, vs_zero, kb_zero, vb_zero,
                             bits=4, group_size=32)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item(), "NaN found in output with zero scales"

    def test_alignment_4bit_d128(self, seed):
        """Packed key last-dim must equal D // (32 // bits) = 16 for 4-bit D=128."""
        B, H_q, H_kv, N, D = 1, 4, 2, 32, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D, bits=4)
        expected_pack_d = D // (32 // 4)  # 128 // 8 = 16
        assert pk.shape[-1] == expected_pack_d, (
            f"packed_k last dim {pk.shape[-1]} != expected {expected_pack_d}"
        )

    def test_batch_size_2(self, seed):
        """B=2: verify batch dimension is handled correctly."""
        B, H_q, H_kv, N, D = 2, 4, 2, 64, 128
        q, pk, pv, ks, vs, kb, vb, k_ref, v_ref = make_qkv(B, H_q, H_kv, N, D)
        out = quantized_sdpa(q, pk, pv, ks, vs, kb, vb, bits=4, group_size=32)
        ref = reference_sdpa(q, k_ref, v_ref)
        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < TOLERANCE, f"max diff {diff} exceeds tolerance {TOLERANCE}"
