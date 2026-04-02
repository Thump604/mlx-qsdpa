"""Tests for QuantizedSDPACache."""

import mlx.core as mx
import pytest


def test_top_level_import():
    from mlx_qsdpa import QuantizedSDPACache
    cache = QuantizedSDPACache()
    assert cache.bits == 4


class TestCacheConstruction:
    def test_defaults(self):
        from mlx_qsdpa.cache import QuantizedSDPACache

        cache = QuantizedSDPACache()
        assert cache.bits == 4
        assert cache.group_size == 32
        assert cache.offset == 0
        assert cache.empty()
        assert cache.nbytes == 0
        assert cache.is_trimmable()

    def test_custom_params(self):
        from mlx_qsdpa.cache import QuantizedSDPACache

        cache = QuantizedSDPACache(bits=8, group_size=64, step=512)
        assert cache.bits == 8
        assert cache.group_size == 64


class TestUpdateAndFetch:
    def test_single_step_decode(self):
        """Single token decode: insert 1 token, get back quantized tuples."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        keys = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)

        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Each is a tuple of (packed, scales, biases)
        assert len(k_quant) == 3
        assert len(v_quant) == 3
        assert cache.offset == 1
        assert not cache.empty()

        # packed shape: (B, H_kv, 1, D // elems_per_int)
        elems_per_int = 32 // 4
        assert k_quant[0].shape == (B, H_kv, 1, D // elems_per_int)
        assert k_quant[0].dtype == mx.uint32
        # scales shape: (B, H_kv, 1, D // group_size)
        assert k_quant[1].shape == (B, H_kv, 1, D // 32)
        assert k_quant[1].dtype == mx.float16

    def test_multi_step_accumulation(self):
        """Multiple decode steps accumulate in cache."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        for i in range(5):
            keys = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            values = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            k_quant, v_quant = cache.update_and_fetch(keys, values)

        assert cache.offset == 5
        # Returned tensors span all 5 tokens
        assert k_quant[0].shape[2] == 5

    def test_prefill_multi_token(self):
        """Prefill with multiple tokens at once."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, N, D = 1, 2, 128, 256

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        assert cache.offset == 128
        assert k_quant[0].shape[2] == 128

    def test_prefill_then_decode(self):
        """Prefill followed by decode steps."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        # Prefill 64 tokens
        keys = mx.random.normal((B, H_kv, 64, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 64, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 64

        # Decode 3 more
        for _ in range(3):
            keys = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            values = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            k_quant, v_quant = cache.update_and_fetch(keys, values)

        assert cache.offset == 67
        assert k_quant[0].shape[2] == 67

    def test_quantization_fidelity(self):
        """Quantize via cache, dequantize, compare to original."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, N, D = 1, 2, 64, 256

        keys = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Dequantize and compare
        k_deq = mx.dequantize(k_quant[0], k_quant[1], k_quant[2],
                              group_size=32, bits=4)
        v_deq = mx.dequantize(v_quant[0], v_quant[1], v_quant[2],
                              group_size=32, bits=4)
        mx.eval(k_deq, v_deq)

        k_diff = mx.max(mx.abs(k_deq - keys)).item()
        v_diff = mx.max(mx.abs(v_deq - values)).item()
        # 4-bit quantization error is bounded
        assert k_diff < 1.0, f"key dequant error {k_diff}"
        assert v_diff < 1.0, f"value dequant error {v_diff}"

    def test_step_boundary_growth(self):
        """Cache grows at step boundaries."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32, step=64)
        B, H_kv, D = 1, 2, 256

        # Fill 60 tokens (under step=64)
        keys = mx.random.normal((B, H_kv, 60, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 60, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 60

        # Add 10 more (crosses 64 boundary, triggers growth)
        keys = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)
        assert cache.offset == 70
        assert k_quant[0].shape[2] == 70


class TestCacheProtocol:
    def test_trim(self):
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        keys = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 10

        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 7

    def test_trim_more_than_offset(self):
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        keys = mx.random.normal((B, H_kv, 5, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 5, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)

        trimmed = cache.trim(100)
        assert trimmed == 5
        assert cache.offset == 0

    def test_make_mask_decode(self):
        """Decode (N=1) with offset > 0 returns None (no mask needed)."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        keys = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)

        mask = cache.make_mask(N=1, return_array=False, window_size=None)
        assert mask is None

    def test_make_mask_prefill(self):
        """Prefill (N>1) returns causal mask."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        cache = QuantizedSDPACache(bits=4, group_size=32)
        mask = cache.make_mask(N=8, return_array=False, window_size=None)
        # With N>1 and return_array=False, should return "causal"
        assert mask == "causal"

    def test_state_round_trip(self):
        """Serialize and reconstruct cache."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        keys = mx.random.normal((B, H_kv, 20, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 20, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Save state
        state = cache.state
        meta = cache.meta_state

        # Reconstruct
        cache2 = QuantizedSDPACache.from_state(state, meta)
        assert cache2.offset == 20
        assert cache2.bits == 4
        assert cache2.group_size == 32

        # Verify data matches
        k2, v2 = cache2.state
        mx.eval(k_quant[0], k2[0])
        diff = mx.max(mx.abs(
            k_quant[0].astype(mx.float32) - k2[0].astype(mx.float32)
        )).item()
        assert diff == 0.0, f"state round-trip mismatch: {diff}"

    def test_nbytes_grows(self):
        """nbytes increases as cache fills."""
        from mlx_qsdpa.cache import QuantizedSDPACache

        mx.random.seed(42)
        cache = QuantizedSDPACache(bits=4, group_size=32)
        B, H_kv, D = 1, 2, 256

        assert cache.nbytes == 0

        keys = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.nbytes > 0
