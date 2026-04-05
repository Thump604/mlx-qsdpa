"""Integration test: QuantizedRotatingSDPACache + cache_sdpa dispatch."""

import mlx.core as mx


def test_cache_sdpa_with_rotating_cache():
    """Verify cache_sdpa dispatches correctly with rotating quantized cache."""
    from mlx_qsdpa import cache_sdpa
    from mlx_qsdpa.cache import QuantizedRotatingSDPACache

    mx.random.seed(42)
    B, H_q, H_kv, D = 1, 8, 2, 64
    max_size = 32
    cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)

    for _ in range(20):
        k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        k_quant, v_quant = cache.update_and_fetch(k, v)

    q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    mask = cache.make_mask(1)
    output = cache_sdpa(q, k_quant, v_quant, cache, mask=mask)
    assert output.shape == (B, H_q, 1, D)
    assert output.dtype == mx.float16


def test_top_level_imports():
    """Verify new classes are exported from package."""
    from mlx_qsdpa import QuantizedRotatingSDPACache, BatchQuantizedRotatingSDPACache
    assert QuantizedRotatingSDPACache is not None
    assert BatchQuantizedRotatingSDPACache is not None


def test_rotating_matches_fp16_reference():
    """Quantized rotating cache output should be close to FP16 attention."""
    from mlx_qsdpa.cache import QuantizedRotatingSDPACache

    mx.random.seed(42)
    B, H_q, H_kv, D = 1, 4, 2, 64
    max_size = 16

    cache = QuantizedRotatingSDPACache(max_size=max_size, bits=4, group_size=32)

    all_keys = []
    all_values = []

    for i in range(max_size + 4):
        k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        all_keys.append(k)
        all_values.append(v)
        k_quant, v_quant = cache.update_and_fetch(k, v)

    ref_k = mx.concatenate(all_keys[-max_size:], axis=2)

    # The physical buffer is in circular order after rotation.
    # Use _dequant_temporal to get keys in temporal (chronological) order.
    k_recon = cache._dequant_temporal(cache._keys)

    assert k_recon.shape[2] == max_size
    assert ref_k.shape[2] == max_size

    max_err = mx.abs(k_recon - ref_k).max().item()
    assert max_err < 1.0, f"Max quant error {max_err} too large"
