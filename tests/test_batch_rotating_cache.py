"""Tests for BatchQuantizedRotatingSDPACache."""

import mlx.core as mx
import pytest


class TestBatchConstruction:
    def test_create(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache
        cache = BatchQuantizedRotatingSDPACache(
            left_padding=[0, 0], max_size=4096, bits=4, group_size=32
        )
        assert cache.max_size == 4096
        assert cache.bits == 4
        assert cache.empty()


class TestBatchDecode:
    def test_batch_decode(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache
        mx.random.seed(42)
        max_size = 8
        cache = BatchQuantizedRotatingSDPACache(
            [0, 0], max_size=max_size, bits=4, group_size=32
        )
        B, H_kv, D = 2, 2, 64
        for i in range(max_size + 4):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            k_q, v_q = cache.update_and_fetch(k, v)
        assert cache.size() == max_size
        assert k_q[0].shape[2] == max_size
        assert k_q[0].shape[0] == B

    def test_multi_token_update_across_boundary(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache

        mx.random.seed(42)
        max_size = 8
        cache = BatchQuantizedRotatingSDPACache(
            [0, 0], max_size=max_size, bits=4, group_size=32
        )
        B, H_kv, D = 2, 2, 64

        k = mx.random.normal((B, H_kv, 6, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 6, D)).astype(mx.float16)
        cache.update_and_fetch(k, v)

        k = mx.random.normal((B, H_kv, 4, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 4, D)).astype(mx.float16)
        k_q, v_q = cache.update_and_fetch(k, v)

        assert cache.size() == max_size
        assert k_q[0].shape == (B, H_kv, max_size, D // (32 // 4))


class TestBatchFilter:
    def test_filter_keeps_subset(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache
        mx.random.seed(42)
        cache = BatchQuantizedRotatingSDPACache(
            [0, 0], max_size=16, bits=4, group_size=32
        )
        B, H_kv, D = 2, 2, 64
        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        cache.filter(mx.array([0]))
        assert cache._keys[0].shape[0] == 1
        assert cache.offset.shape[0] == 1

    def test_make_mask_respects_left_padding(self):
        from mlx_qsdpa.cache import (
            BatchQuantizedRotatingSDPACache,
            QuantizedRotatingSDPACache,
        )

        mx.random.seed(42)
        B, H_kv, D = 1, 2, 64

        cache_a = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
        cache_b = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)

        for _ in range(4):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache_a.update_and_fetch(k, v)
        for _ in range(2):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache_b.update_and_fetch(k, v)

        batch = BatchQuantizedRotatingSDPACache.merge([cache_a, cache_b])
        mask = batch.make_mask(2)

        expected = mx.array(
            [
                [
                    [
                        [True, True, True, True, True, False],
                        [True, True, True, True, True, True],
                    ]
                ],
                [
                    [
                        [False, False, True, True, True, False],
                        [False, False, True, True, True, True],
                    ]
                ],
            ],
            dtype=mx.bool_,
        )

        mx.eval(mask, expected)
        assert mask.shape == (2, 1, 2, 6)
        assert mx.array_equal(mask, expected).item()

    def test_make_mask_uses_visible_buffer_not_logical_offset(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache

        cache = BatchQuantizedRotatingSDPACache(
            [0], max_size=65536, bits=8, group_size=64
        )
        cache._idx = 89
        cache.offset = mx.array([819])

        mask = cache.make_mask(1)

        mx.eval(mask)
        assert mask.shape == (1, 1, 1, 90)

    def test_make_mask_decode_matches_next_visible_token(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache

        mx.random.seed(42)
        cache = BatchQuantizedRotatingSDPACache(
            [0], max_size=16, bits=4, group_size=32
        )
        B, H_kv, D = 1, 2, 64

        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)

        mask = cache.make_mask(1)
        k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        keys, _ = cache.update_and_fetch(k, v)

        mx.eval(mask, keys[0])
        assert mask.shape[-1] == keys[0].shape[2]

    def test_decode_grows_compact_merged_buffer(self):
        from mlx_qsdpa.cache import (
            BatchQuantizedRotatingSDPACache,
            QuantizedRotatingSDPACache,
        )

        mx.random.seed(42)
        B, H_kv, D = 1, 2, 64
        single = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)

        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            single.update_and_fetch(k, v)

        batch = BatchQuantizedRotatingSDPACache.merge([single])
        assert batch._keys[0].shape[2] == 5

        mask = batch.make_mask(1)
        k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        keys, _ = batch.update_and_fetch(k, v)

        mx.eval(mask, keys[0])
        assert keys[0].shape[2] == 6
        assert mask.shape[-1] == 6


class TestBatchExtractAndMerge:
    def test_extract_single(self):
        from mlx_qsdpa.cache import (
            BatchQuantizedRotatingSDPACache,
            QuantizedRotatingSDPACache,
        )
        mx.random.seed(42)
        cache = BatchQuantizedRotatingSDPACache(
            [0, 0], max_size=16, bits=4, group_size=32
        )
        B, H_kv, D = 2, 2, 64
        for _ in range(5):
            k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
            cache.update_and_fetch(k, v)
        extracted = cache.extract(0)
        assert isinstance(extracted, QuantizedRotatingSDPACache)
        assert extracted.offset == 5
        assert extracted.max_size == 16

    def test_merge_singles(self):
        from mlx_qsdpa.cache import (
            BatchQuantizedRotatingSDPACache,
            QuantizedRotatingSDPACache,
        )
        mx.random.seed(42)
        B, H_kv, D = 1, 2, 64
        caches = []
        for length in [3, 5]:
            c = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
            for _ in range(length):
                k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
                v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
                c.update_and_fetch(k, v)
            caches.append(c)
        batch = BatchQuantizedRotatingSDPACache.merge(caches)
        assert isinstance(batch, BatchQuantizedRotatingSDPACache)
        assert batch._keys[0].shape[0] == 2
        assert batch.max_size == 16

    def test_single_cache_merge_delegates_to_batch_merge(self):
        from mlx_qsdpa.cache import (
            BatchQuantizedRotatingSDPACache,
            QuantizedRotatingSDPACache,
        )

        mx.random.seed(42)
        B, H_kv, D = 1, 2, 64
        caches = []
        for length in [2, 4]:
            cache = QuantizedRotatingSDPACache(max_size=16, bits=4, group_size=32)
            for _ in range(length):
                k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
                v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
                cache.update_and_fetch(k, v)
            caches.append(cache)

        batch = caches[0].merge(caches)
        assert isinstance(batch, BatchQuantizedRotatingSDPACache)
        assert batch._keys[0].shape[0] == 2
        assert batch.max_size == 16

    def test_extract_preserves_temporal_order_after_wrap(self):
        from mlx_qsdpa.cache import BatchQuantizedRotatingSDPACache

        mx.random.seed(42)
        cache = BatchQuantizedRotatingSDPACache(
            [0], max_size=4, bits=4, group_size=32
        )
        all_keys = []

        for _ in range(6):
            k = mx.random.normal((1, 1, 1, 64)).astype(mx.float16)
            v = mx.random.normal((1, 1, 1, 64)).astype(mx.float16)
            all_keys.append(k)
            cache.update_and_fetch(k, v)

        extracted = cache.extract(0)
        expected_keys = mx.concatenate(all_keys[-4:], axis=2)
        expected_quant = mx.quantize(expected_keys, group_size=32, bits=4)

        mx.eval(extracted.state[0][0], expected_quant[0])
        assert extracted.offset == 4
        assert mx.array_equal(extracted.state[0][0], expected_quant[0]).item()
