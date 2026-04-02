"""Integration tests: QuantizedSDPACache output feeds into quantized_sdpa kernel."""

import mlx.core as mx
import pytest

TOLERANCE = 0.02  # slightly looser for full pipeline


class TestDynamicDispatch:
    """Test cache_sdpa: dynamic FP16/fused crossover dispatch."""

    def test_short_context_uses_fp16_path(self):
        """Below crossover, dequantizes and uses FP16 SDPA."""
        from mlx_qsdpa import cache_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 32, 2, 1024, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

        # With crossover=16384, N=1024 should use FP16 path
        out = cache_sdpa(q, k_quant, v_quant, cache, crossover=16384)

        # Reference: dequant + FP16 SDPA
        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        ref = mx.fast.scaled_dot_product_attention(q, k_deq, v_deq, scale=D**-0.5)

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < 1e-5, f"short context diff {diff} (should match FP16 exactly)"

    def test_long_context_uses_fused_path(self):
        """Above crossover, uses fused quantized kernel."""
        from mlx_qsdpa import cache_sdpa, quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 32768, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

        # With crossover=16384, N=32768 should use fused path
        out = cache_sdpa(q, k_quant, v_quant, cache, crossover=16384)

        # Reference: direct fused kernel call
        ref = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < 1e-5, f"long context diff {diff} (should match fused exactly)"

    def test_crossover_boundary(self):
        """At exactly the crossover, uses fused path."""
        from mlx_qsdpa import cache_sdpa, quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, D = 1, 2, 2, 256
        crossover = 16384
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, crossover, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, crossover, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        out = cache_sdpa(q, k_quant, v_quant, cache, crossover=crossover)

        ref = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < 1e-5, f"boundary diff {diff}"

    def test_prefill_always_dequantizes(self):
        """Prefill (qL>1) always dequantizes regardless of N."""
        from mlx_qsdpa import cache_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 32768, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 8, D)).astype(mx.float16)

        # qL=8, should use dequant+SDPA even though N > crossover
        out = cache_sdpa(q, k_quant, v_quant, cache, crossover=16384)

        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        ref = mx.fast.scaled_dot_product_attention(q, k_deq, v_deq, scale=D**-0.5)

        mx.eval(out, ref)
        diff = mx.max(mx.abs(out - ref)).item()
        assert diff < 1e-5, f"prefill diff {diff}"

    def test_default_crossover(self):
        """Default crossover=16384 is used when not specified."""
        from mlx_qsdpa import cache_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

        # Should not raise; N=64 < 16384, uses FP16 path
        out = cache_sdpa(q, k_quant, v_quant, cache)
        mx.eval(out)
        assert out.shape == (B, H_q, 1, D)


class TestCacheKernelIntegration:
    def test_decode_non_gqa(self):
        """Cache -> quantized_sdpa for decode, no GQA."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        # Fill cache with N tokens
        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Query
        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

        # Fused path
        out_fused = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )

        # Reference: dequantize and use FP16 SDPA
        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=D**-0.5
        )

        mx.eval(out_fused, out_ref)
        diff = mx.max(mx.abs(out_fused - out_ref)).item()
        assert diff < TOLERANCE, f"decode non-GQA diff {diff} exceeds {TOLERANCE}"

    def test_decode_gqa(self):
        """Cache -> quantized_sdpa for decode, GQA 16x (Qwen3.5-122B dims)."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 32, 2, 64, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

        out_fused = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )

        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=D**-0.5
        )

        mx.eval(out_fused, out_ref)
        diff = mx.max(mx.abs(out_fused - out_ref)).item()
        assert diff < TOLERANCE, f"decode GQA diff {diff} exceeds {TOLERANCE}"

    def test_prefill_fallback(self):
        """Prefill (qL>1) uses dequant+SDPA fallback inside quantized_sdpa."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        qL = 8
        cache = QuantizedSDPACache(bits=4, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, qL, D)).astype(mx.float16)

        out_fused = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )

        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=D**-0.5
        )

        mx.eval(out_fused, out_ref)
        diff = mx.max(mx.abs(out_fused - out_ref)).item()
        assert diff < TOLERANCE, f"prefill diff {diff} exceeds {TOLERANCE}"

    def test_incremental_cache_then_query(self):
        """Prefill + decode steps, then query -- full pipeline."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, D = 1, 2, 2, 256
        cache = QuantizedSDPACache(bits=4, group_size=32)

        # Prefill 32 tokens
        keys = mx.random.normal((B, H_kv, 32, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 32, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)

        # Decode 8 more tokens one at a time
        for _ in range(8):
            k1 = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v1 = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            k_quant, v_quant = cache.update_and_fetch(k1, v1)

        assert cache.offset == 40

        # Query against full cache
        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        out = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=4, group_size=32,
        )
        mx.eval(out)
        assert out.shape == (B, H_q, 1, D)

    def test_8bit(self):
        """8-bit quantization through cache + kernel."""
        from mlx_qsdpa import quantized_sdpa
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 64, 256
        cache = QuantizedSDPACache(bits=8, group_size=32)

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        out_fused = quantized_sdpa(
            q, k_quant[0], v_quant[0],
            k_quant[1], v_quant[1], k_quant[2], v_quant[2],
            bits=8, group_size=32,
        )

        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=8)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=8)
        out_ref = mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=D**-0.5
        )

        mx.eval(out_fused, out_ref)
        diff = mx.max(mx.abs(out_fused - out_ref)).item()
        # 8-bit should be tighter
        assert diff < 0.005, f"8-bit diff {diff}"
