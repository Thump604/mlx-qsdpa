# mlx-qsdpa

Quantized SDPA decode kernel for MLX on Apple Silicon.

## What it does

`mlx-qsdpa` provides a fused quantized attention kernel that reads 4-bit or 8-bit KV cache tensors
and dequantizes inline in a single Metal dispatch. No intermediate FP16 buffer is ever materialized
during decode. This is 1.7x faster than mlx-lm's two-call quantized path (`mx.quantized_matmul` x2
\+ softmax) at 128K context with GQA, and reduces KV cache memory by 4x (6 GB vs 24 GB at 1M
context on Qwen3.5-122B).

The package includes:

- **`quantized_sdpa`** -- fused decode kernel (single-pass and split-K two-pass)
- **`QuantizedSDPACache`** -- quantized KV cache following mlx-lm's `_BaseCache` protocol
- **`BatchQuantizedSDPACache`** -- batch-aware quantized cache for continuous batching / prefix reuse
- **`QuantizedRotatingSDPACache`** -- quantized circular-buffer KV cache for sliding-window layers
- **`BatchQuantizedRotatingSDPACache`** -- batch-aware rotating cache for continuous batching
- **`cache_sdpa`** -- dynamic dispatch: FP16 SDPA below 16K, fused kernel above

## Install

```bash
pip install mlx-qsdpa
```

Or from source:

```bash
git clone https://github.com/Thump604/mlx-qsdpa
cd mlx-qsdpa
pip install -e .
```

Requires Python >= 3.10 and mlx >= 0.21.0. No other dependencies.

## Quick Start

### Low-level kernel

```python
import mlx.core as mx
from mlx_qsdpa import quantized_sdpa

# Quantize your K/V tensors
packed_k, k_scales, k_biases = mx.quantize(k_fp16, group_size=32, bits=4)
packed_v, v_scales, v_biases = mx.quantize(v_fp16, group_size=32, bits=4)

# Decode attention -- no FP16 K/V materialized
q = mx.random.normal((1, 32, 1, 256)).astype(mx.float16)
out = quantized_sdpa(
    q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases,
    bits=4, group_size=32,
)
```

### Cache + dynamic dispatch (recommended)

```python
from mlx_qsdpa import QuantizedSDPACache, cache_sdpa

cache = QuantizedSDPACache(bits=4, group_size=32)

# Prefill
k_quant, v_quant = cache.update_and_fetch(keys, values)

# Decode with dynamic dispatch:
#   N < 16K  -> dequant + FP16 SDPA (faster at short context)
#   N >= 16K -> fused quantized kernel (1.7x faster at long context)
out = cache_sdpa(q, k_quant, v_quant, cache)
```

## Performance

### Fused kernel vs mlx-lm's quantized path

Measured on M2 Ultra 128 GB. B=1, D=256, 4-bit gs=32. Median of 100 iterations after warmup.

**GQA-16x (production config: H_q=32, H_kv=2, Qwen3.5-122B):**

| Context | FP16 (us) | quantized_matmul (us) | fused (us) | fused vs qmm | fused vs FP16 |
|---------|-----------|----------------------|------------|--------------|---------------|
| 1K      | 234       | 312                  | 301        | 1.04x        | 0.78x         |
| 4K      | 264       | 392                  | 512        | 0.77x        | 0.52x         |
| 16K     | 427       | 902                  | 588        | **1.53x**    | 0.73x         |
| 32K     | 568       | 1303                 | 882        | **1.48x**    | 0.64x         |
| 64K     | 920       | 2244                 | 1438       | **1.56x**    | 0.64x         |
| 128K    | 1478      | 4161                 | 2431       | **1.71x**    | 0.61x         |

The fused kernel eliminates the tensor reshapes and intermediate materialization that the two-call
path requires for GQA. The advantage grows with sequence length: 1.71x at 128K.

**Non-GQA (H_q=2, H_kv=2, isolates kernel fusion from GQA handling):**

| Context | FP16 (us) | quantized_matmul (us) | fused (us) | fused vs qmm | fused vs FP16 |
|---------|-----------|----------------------|------------|--------------|---------------|
| 1K      | 221       | 241                  | 294        | 0.82x        | 0.75x         |
| 16K     | 249       | 277                  | 271        | **1.02x**    | 0.92x         |
| 64K     | 397       | 377                  | 340        | **1.11x**    | 1.17x         |
| 128K    | 587       | 542                  | 458        | **1.18x**    | **1.28x**     |

Without GQA, the fused kernel also beats FP16 at 64K+ due to 4x less memory bandwidth. The
crossover happens at ~16K where the split-K two-pass kernel activates.

### Why the 4K dip?

The fused kernel transitions from single-pass (one threadgroup per query head) to split-K two-pass
at N=4096. At exactly 4096, the single-pass kernel has only B\*H_q threadgroups, which cannot
saturate the GPU. Above 4096, the split-K kernel distributes work across hundreds of blocks. This
is a tiling discontinuity, not a fundamental limitation.

### Dynamic dispatch crossover

`cache_sdpa` uses FP16 SDPA below 16K (where it is faster) and the fused kernel at 16K+ (where
memory bandwidth savings dominate). This gives best performance at all context lengths.

### Memory savings

The primary value at shorter contexts is not speed but memory reduction:

| Format | KV size at 1M (122B, 12 attn layers) | Headroom on 128 GB |
|--------|--------------------------------------|--------------------|
| FP16   | ~24 GB                               | ~22 GB after weights |
| 4-bit  | ~6 GB                                | ~40 GB after weights |

This makes 1M+ context viable on 128 GB hardware.

## API Reference

### `quantized_sdpa`

```python
mlx_qsdpa.quantized_sdpa(
    q, packed_k, packed_v, k_scales, v_scales,
    k_biases=None, v_biases=None,
    scale=None, mask=None, bits=4, group_size=32, threshold=4096,
)
```

Low-level fused attention kernel. For decode (qL=1), runs the Metal kernel directly. For prefill
(qL>1), dequantizes and delegates to `mx.fast.scaled_dot_product_attention`.

**Parameters:**

- `q` -- `(B, H_q, qL, D)` float16/bfloat16
- `packed_k`, `packed_v` -- `(B, H_kv, N, D // (32 // bits))` uint32 from `mx.quantize`
- `k_scales`, `v_scales` -- `(B, H_kv, N, D // group_size)` float16
- `k_biases`, `v_biases` -- same shape as scales, or None for symmetric
- `scale` -- float, default `D ** -0.5`
- `bits` -- 4 or 8
- `group_size` -- 32, 64, or 128
- `threshold` -- split-K activation threshold (default 4096)

**Returns:** `(B, H_q, qL, D)` in same dtype as `q`.

### `QuantizedSDPACache`

```python
mlx_qsdpa.QuantizedSDPACache(bits=4, group_size=32, step=256)
```

Quantized KV cache following mlx-lm's `_BaseCache` protocol. Stores K/V in `mx.quantize` format
with pre-allocated buffers.

**`update_and_fetch(keys, values)`** -- Quantizes and appends K/V to the cache. Returns
`(keys_quant, values_quant)` where each is a tuple of `(packed_uint32, scales_fp16, biases_fp16)`.
This differs from mlx-lm's KVCache which returns plain float16 tensors.

Protocol methods: `offset`, `bits`, `group_size`, `empty()`, `nbytes`, `is_trimmable()`, `trim()`,
`rewind()`, `make_mask()`, `state`/`meta_state`, `from_state()`, `merge()`.

`merge([cache_a, cache_b, ...])` returns a `BatchQuantizedSDPACache` suitable for continuous
batching or admitting cached history into a batched prefill path.

### `BatchQuantizedSDPACache`

```python
mlx_qsdpa.BatchQuantizedSDPACache(left_padding=[0, 0], bits=4, group_size=32)
```

Batch-aware quantized KV cache. Mirrors mlx-lm's batch-cache protocol while storing K/V in
quantized `(packed, scales, biases)` form. Supports `prepare()`, `finalize()`, `filter()`,
`extend()`, `extract()`, left-padding-aware `make_mask()`, and `state` round-trips.

### `QuantizedRotatingSDPACache`

```python
mlx_qsdpa.QuantizedRotatingSDPACache(max_size=4096, keep=0, bits=4, group_size=32)
```

Quantized circular-buffer KV cache for sliding-window attention layers such as Gemma 4. Decode
writes tokens in place; multi-token updates rebuild the visible window in temporal order and
re-quantize it back into the bounded buffer.

Supports `make_mask()`, `trim()`, `rewind()`, and temporal-order `state` snapshots. Current
limitation: `keep > 0` is not implemented yet.

### `BatchQuantizedRotatingSDPACache`

```python
mlx_qsdpa.BatchQuantizedRotatingSDPACache(
    left_padding=[0, 0], max_size=4096, keep=0, bits=4, group_size=32
)
```

Batch-aware rotating cache for continuous batching. Supports filtering, extraction back to a
single-request rotating cache, merge from single rotating caches, and left-padding-aware masks for
batched sliding-window attention.

### `cache_sdpa`

```python
mlx_qsdpa.cache_sdpa(
    q, k_quant, v_quant, cache, scale=None, mask=None, crossover=16384,
)
```

Dynamic dispatch attention. Below `crossover`, dequantizes and uses FP16 SDPA. At or above
`crossover`, uses the fused quantized kernel. Prefill (qL>1) always dequantizes.

**Parameters:**

- `q` -- `(B, H_q, qL, D)` float16
- `k_quant`, `v_quant` -- quantized tuples from `QuantizedSDPACache.update_and_fetch`
- `cache` -- any quantized cache carrying `bits` and `group_size`
  (`QuantizedSDPACache`, `BatchQuantizedSDPACache`, `QuantizedRotatingSDPACache`,
  `BatchQuantizedRotatingSDPACache`)
- `crossover` -- sequence length threshold (default 16384)

## Supported Formats

Matches `mx.quantize` / `mx.dequantize` exactly. No conversion needed.

| Format | bits | biases | group_size | Notes |
|--------|------|--------|------------|-------|
| 4-bit asymmetric | 4 | yes | 32 / 64 / 128 | Default. Matches mlx-lm QuantizedLinear. |
| 4-bit symmetric | 4 | no | 32 / 64 / 128 | Faster (no bias read). |
| 8-bit asymmetric | 8 | yes | 32 / 64 / 128 | |
| 8-bit symmetric | 8 | no | 32 / 64 / 128 | |

## How It Works

Two Metal kernels compiled via `mx.fast.metal_kernel()`:

**Single-pass** (N <= threshold): One threadgroup per (batch, query head). 32 SIMDgroups stride
through keys. Each loads a uint32-packed key, extracts elements by bit shift, applies scale/bias,
computes a dot product via `simd_sum`, and updates online softmax state. Cross-SIMDgroup reduction
merges 32 partial outputs.

**Two-pass split-K** (N > threshold): Pass 1 distributes the key sequence across up to 1024 blocks.
Each block computes partial attention with logsumexp stats. Pass 2 merges blocks using
logsumexp-weighted averaging. Keeps latency linear in N at 128K+.

## Benchmarks

```bash
# Three-way comparison: FP16 vs quantized_matmul vs fused
python -m mlx_qsdpa.bench_comparison --headline-only
python -m mlx_qsdpa.bench_comparison --decode-only --seq-len 16384,65536 --heads 32,2
python -m mlx_qsdpa.bench_comparison --json --output results.jsonl

# Single kernel benchmark (original)
python -m mlx_qsdpa.bench --sweep
```

Tests (87 tests):

```bash
pip install mlx-qsdpa[dev]
pytest tests/
```

## Downstream Integrations

`mlx-qsdpa` is designed to be embedded by other runtimes and applications, but those integrations
are downstream-specific and are not part of the standalone `pip install mlx-qsdpa` API described
in this repository.

## Limitations

**GQA compute overhead.** With GQA, the fused kernel is 0.52-0.78x FP16 speed (but 1.5-1.7x faster
than the existing quantized path). The overhead is compute-bound: multiple query heads each do
independent dequant+FMA against shared K/V. Without GQA, fused matches or beats FP16 at 64K+.

**mx.fast.metal_kernel ceiling.** Kernels are written via `mx.fast.metal_kernel()`. This allows
rapid iteration but cannot beat a native C++ kernel. A future upstream C++ kernel (mlx PR #3026
path) would eliminate that ceiling.

**Prefill is not accelerated.** Prefill is compute-bound. For qL > 1, the function dequantizes
and delegates to `mx.fast.scaled_dot_product_attention`.

**No 2-bit support.** 4-bit and 8-bit affine only.

## License

Apache 2.0. See LICENSE.
