"""Tests for QuantizedRotatingSDPACache."""

import mlx.core as mx
import pytest


class TestConstruction:
    def test_defaults(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(max_size=4096)
        assert cache.max_size == 4096
        assert cache.keep == 0
        assert cache.bits == 4
        assert cache.group_size == 32
        assert cache.offset == 0
        assert cache._idx == 0
        assert cache.empty()
        assert cache.nbytes == 0
        assert cache._use_fused_sdpa is True

    def test_custom_params(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(
            max_size=2048, bits=8, group_size=64
        )
        assert cache.max_size == 2048
        assert cache.bits == 8
        assert cache.group_size == 64

    def test_keep_nonzero_raises(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        with pytest.raises(AssertionError):
            QuantizedRotatingSDPACache(max_size=4096, keep=4)

    def test_size_empty(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(max_size=4096)
        assert cache.size() == 0

    def test_is_trimmable_when_empty(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(max_size=4096)
        assert cache.is_trimmable() is True


class TestDecodeUpdateAndFetch:
    def test_single_decode_step(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(max_size=4096, bits=4, group_size=32)
        B, H_kv, S, D = 1, 2, 1, 256
        k = mx.random.normal(shape=(B, H_kv, S, D)).astype(mx.float16)
        v = mx.random.normal(shape=(B, H_kv, S, D)).astype(mx.float16)

        k_out, v_out = cache.update_and_fetch(k, v)

        assert isinstance(k_out, tuple) and len(k_out) == 3
        assert isinstance(v_out, tuple) and len(v_out) == 3
        assert cache.offset == 1
        assert cache._idx == 1
        assert k_out[0].shape[2] == 1

    def test_fill_to_max_size(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        max_size = 8
        cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)
        B, H_kv, D = 1, 1, 64

        k_out = v_out = None
        for _ in range(max_size):
            k = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            k_out, v_out = cache.update_and_fetch(k, v)

        assert cache.offset == max_size
        assert cache._idx == max_size
        assert cache.size() == max_size
        assert k_out[0].shape[2] == max_size

    def test_rotation_wraps_idx(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        max_size = 8
        extra = 4
        cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)
        B, H_kv, D = 1, 1, 64

        k_out = v_out = None
        for _ in range(max_size + extra):
            k = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            k_out, v_out = cache.update_and_fetch(k, v)

        assert cache.offset == max_size + extra
        assert cache._idx == extra  # wrapped: (max_size + extra) % max_size
        assert cache.size() == max_size
        assert k_out[0].shape[2] == max_size

    def test_nbytes_fixed_after_allocation(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        max_size = 8
        cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)
        B, H_kv, D = 1, 1, 64

        k = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
        cache.update_and_fetch(k, v)
        nbytes_after_first = cache.nbytes

        for _ in range(max_size + 10 - 1):
            k = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal(shape=(B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)

        assert cache.nbytes == nbytes_after_first

    def test_quantization_accuracy(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache

        cache = QuantizedRotatingSDPACache(max_size=4096, bits=4, group_size=32)
        B, H_kv, S, D = 1, 1, 1, 256
        k = mx.random.normal(shape=(B, H_kv, S, D)).astype(mx.float16)
        v = mx.random.normal(shape=(B, H_kv, S, D)).astype(mx.float16)

        k_out, v_out = cache.update_and_fetch(k, v)

        k_dequant = mx.dequantize(
            k_out[0], k_out[1], k_out[2],
            group_size=cache.group_size, bits=cache.bits
        )
        max_err = mx.abs(k_dequant - k).max().item()
        assert max_err < 0.5, f"Max quantization error {max_err:.4f} exceeds 0.5"


class TestMaskAndTrim:
    def test_mask_before_rotation(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        mask = cache.make_mask(1, return_array=True)
        assert mask is None
        mask = cache.make_mask(3, return_array=True)
        assert mask.shape[-1] == 8
        assert mask.shape[-2] == 3

    def test_mask_returns_none_for_decode(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        result = cache.make_mask(1)
        assert result is None

    def test_trim_pre_rotation(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        assert cache.is_trimmable()
        trimmed = cache.trim(2)
        assert trimmed == 2
        assert cache.offset == 3
        assert cache._idx == 3

    def test_not_trimmable_after_rotation(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        cache = QuantizedRotatingSDPACache(max_size=8, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        for _ in range(10):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        assert not cache.is_trimmable()


class TestState:
    def test_state_roundtrip(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        mx.random.seed(42)
        cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        saved_state = cache.state
        saved_offset = cache.offset
        saved_idx = cache._idx
        cache2 = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        cache2.state = saved_state
        cache2.offset = saved_offset
        cache2._idx = saved_idx
        k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        k1, v1 = cache.update_and_fetch(k, v)
        k2, v2 = cache2.update_and_fetch(k, v)
        assert mx.array_equal(k1[0], k2[0])
        assert mx.array_equal(v1[0], v2[0])


class TestPrefill:
    def test_prefill_multi_token(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        mx.random.seed(42)
        cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        keys = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 10, D)).astype(mx.float16)
        k_q, v_q = cache.update_and_fetch(keys, values)
        assert cache.offset == 10
        assert k_q[0].shape[2] == 10

    def test_prefill_exceeds_max_size(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        mx.random.seed(42)
        max_size = 8
        cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        keys = mx.random.normal((B, H_kv, 12, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 12, D)).astype(mx.float16)
        k_q, v_q = cache.update_and_fetch(keys, values)
        assert cache.offset == 12
        assert cache.size() == max_size
        assert k_q[0].shape[2] == max_size

    def test_prefill_then_decode(self):
        from mlx_qsdpa.cache import QuantizedRotatingSDPACache
        mx.random.seed(42)
        max_size = 8
        cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)
        B, H_kv, D = 1, 2, 64
        keys = mx.random.normal((B, H_kv, 6, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, 6, D)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        assert cache.offset == 6
        assert cache._idx == 6
        for i in range(4):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            k_q, v_q = cache.update_and_fetch(k, v)
        assert cache.offset == 10
        assert cache._idx == 2
        assert cache.size() == max_size
        assert k_q[0].shape[2] == max_size
