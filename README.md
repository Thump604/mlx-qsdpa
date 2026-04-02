# mlx-qsdpa

Quantized SDPA decode kernel for MLX on Apple Silicon.

## What it does

`mlx-qsdpa` provides a drop-in replacement for `mx.fast.scaled_dot_product_attention` that reads
4-bit or 8-bit quantized K/V cache tensors and dequantizes inline in the Metal kernel. No FP16
K/V buffer is ever materialized during decode. At 1M context, 4-bit KV reads are 6 GB vs 24 GB for
FP16, reducing decode memory bandwidth by 4x and making long-context generation practical on
128 GB Apple Silicon hardware.

The package is a pure function with no monkey-patching. It accepts packed integers and per-group
scales/biases from `mx.quantize` directly and returns a standard float16 attention output tensor.

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

```python
import mlx.core as mx
from mlx_qsdpa import quantized_sdpa

# Assume k_fp16, v_fp16 are your key/value tensors before quantization
# Shape: (B, H_kv, N, D) float16
packed_k, k_scales, k_biases = mx.quantize(k_fp16, group_size=32, bits=4)
packed_v, v_scales, v_biases = mx.quantize(v_fp16, group_size=32, bits=4)

# Query tensor for decode step (qL=1)
# Shape: (B, H_q, 1, D) float16
q = mx.random.normal((1, 16, 1, 256)).astype(mx.float16)

# Quantized attention -- no FP16 K/V materialized
out = quantized_sdpa(
    q,
    packed_k, packed_v,
    k_scales, v_scales,
    k_biases, v_biases,
    bits=4,
    group_size=32,
)
# out: (1, 16, 1, 256) float16

mx.eval(out)
```

For symmetric quantization (no biases), omit `k_biases` and `v_biases`:

```python
out = quantized_sdpa(q, packed_k, packed_v, k_scales, v_scales, bits=4)
```

For prefill (qL > 1), the function automatically dequantizes and delegates to
`mx.fast.scaled_dot_product_attention`. No special handling required.

## Performance

Measured on M2 Ultra 128 GB with Qwen3.5-122B dimensions: B=1, H_q=16, H_kv=2, D=256 (GQA factor
8). Kernel-only latency, 100 iterations after warmup.

| Context | FP16 (ms) | 4-bit qSDPA (ms) | 4-bit vs FP16 |
|---------|-----------|-------------------|---------------|
| 1K      | 0.21      | 0.25              | 0.83x         |
| 4K      | 0.24      | 0.29              | 0.81x         |
| 8K      | 0.27      | 0.36              | 0.76x         |
| 32K     | 0.43      | 0.56              | 0.72x         |
| 131K    | 0.99      | 1.41              | 0.70x         |

With GQA 8x (Qwen3.5), the quantized kernel is 0.70-0.83x FP16. The overhead is compute-bound:
8 query heads each do independent dequant+FMA against shared K/V data. L1 cache already handles
the memory sharing efficiently.

**Without GQA (H_q == H_kv), quantized SDPA matches FP16 at 0.98x.** The GQA overhead is the
dominant factor, not the dequant itself.

The primary value is not raw token/s on current hardware but **memory reduction**. At 1M context:

| Format | KV size (122B, 12 attn layers) | Headroom on 128 GB |
|--------|---------------------------------|---------------------|
| FP16   | ~24 GB                          | ~22 GB after weights |
| 4-bit  | ~6 GB                           | ~40 GB after weights |
| 8-bit  | ~12 GB                          | ~34 GB after weights |

This makes 1M+ context viable on 128 GB hardware. The 122B model uses ~82 GB for weights, leaving
46 GB. 4-bit KV at 1M consumes 6 GB; FP16 would consume 24 GB and exhaust the budget.

Without GQA (H_q == H_kv), the quantized kernel matches FP16 at 0.98x due to more balanced work
distribution across SIMDgroups.

## API Reference

```python
mlx_qsdpa.quantized_sdpa(
    q,
    packed_k,
    packed_v,
    k_scales,
    v_scales,
    k_biases=None,
    v_biases=None,
    scale=None,
    mask=None,
    bits=4,
    group_size=32,
    threshold=4096,
)
```

**Parameters:**

- `q` -- `(B, H_q, qL, D)` float16 or bfloat16. Query tensor. For decode, qL must be 1.
  For prefill (qL > 1), the function dequantizes K/V and delegates to standard SDPA.

- `packed_k` -- `(B, H_kv, N, D // (32 // bits))` uint32. Quantized keys from `mx.quantize`.

- `packed_v` -- `(B, H_kv, N, D // (32 // bits))` uint32. Quantized values from `mx.quantize`.

- `k_scales` -- `(B, H_kv, N, D // group_size)` float16. Per-group scale factors for keys.

- `v_scales` -- `(B, H_kv, N, D // group_size)` float16. Per-group scale factors for values.

- `k_biases` -- `(B, H_kv, N, D // group_size)` float16, optional. Per-group biases for keys.
  Pass `None` for symmetric quantization (faster; no bias buffer read).

- `v_biases` -- `(B, H_kv, N, D // group_size)` float16, optional. Per-group biases for values.

- `scale` -- float, optional. Attention scale. Default: `D ** -0.5`.

- `mask` -- not yet supported. Reserved for future use.

- `bits` -- int. Quantization bits. Must be 4 or 8.

- `group_size` -- int. Elements per quantization group. Must be 32, 64, or 128. Default: 32.

- `threshold` -- int. Sequence length above which the two-pass split-K kernel is used instead of
  the single-pass kernel. Default: 4096. Tunable for your hardware.

**Returns:** `(B, H_q, 1, D)` tensor in the same dtype as `q`.

**Raises:**
- `ValueError` if `q.ndim != 4`
- `ValueError` if `bits` is not 4 or 8
- `ValueError` if `D % 32 != 0`
- `ValueError` if `H_q % H_kv != 0`
- `ValueError` if `packed_k` last dim does not match `D // (32 // bits)`

## Supported Formats

The quantization format matches `mx.quantize` / `mx.dequantize` exactly. No conversion needed.

| Format | bits | has_zeros | group_size | Notes |
|--------|------|-----------|------------|-------|
| 4-bit symmetric | 4 | no | 32 / 64 / 128 | Fastest. No bias buffer read. |
| 4-bit asymmetric | 4 | yes | 32 / 64 / 128 | Matches mlx-lm QuantizedLinear default. |
| 8-bit symmetric | 8 | no | 32 / 64 / 128 | |
| 8-bit asymmetric | 8 | yes | 32 / 64 / 128 | |

Dequantization formula (matches `mx.dequantize`):

```
value = scale * packed_int + bias   # asymmetric
value = scale * packed_int          # symmetric (bias == 0)
```

Bit unpacking from uint32:

```
element_i = (packed >> (i * bits)) & ((1 << bits) - 1)
```

Group size 32 is recommended. One SIMDgroup (32 threads) processes exactly one quantization group
in a single coalesced read, minimizing register pressure.

## How It Works

Two Metal kernels share one inner loop, compiled via `mx.fast.metal_kernel()`.

**Single-pass kernel** (`N <= threshold`): One threadgroup per (batch, query head). 32 SIMDgroups
stride through the key sequence. Each SIMDgroup loads a uint32-packed key, extracts elements by
bit shift, applies scale and bias, computes a dot product with the pre-loaded query via `simd_sum`,
and updates online softmax state (max + sum). Value accumulation follows the same dequant path.
A cross-SIMDgroup reduction via threadgroup memory merges the 32 partial outputs into the final
result.

**Two-pass split-K kernel** (`N > threshold`): Pass 1 distributes the key sequence across up to
1024 blocks in parallel. Each block computes partial attention (partial sum, logsumexp stats) over
its slice. Pass 2 merges the blocks using logsumexp-weighted averaging with a second reduction pass.
This avoids GPU watchdog timeouts at 128K+ context and keeps latency linear in N.

The dequant inner loop is fully inlined (not a helper function call). This allows the Metal
compiler to hoist scale/bias loads, unroll the bit-extract loop, and issue half-precision FMA
instructions at full throughput.

## Benchmarks

Run the included benchmark script to measure kernel latency and effective bandwidth on your
hardware:

```bash
# Single config
python -m mlx_qsdpa.bench --bits 4 --context 131072 --q-heads 16 --kv-heads 2 --head-dim 256

# Full sweep across context lengths and bit widths
python -m mlx_qsdpa.bench --sweep

# All options
python -m mlx_qsdpa.bench --help
```

Output includes FP16 baseline latency, quantized latency, speedup ratio, and effective memory
bandwidth for both paths.

Run the test suite (28 tests):

```bash
pip install mlx-qsdpa[dev]
pytest tests/
pytest tests/ -m "not slow"   # skip long-context tests
```

## Limitations

**GQA compute overhead.** When H_q >> H_kv (e.g., 8x GQA), KV data volume is small relative to
query computation. The quantized kernel does not recover the overhead of the two-pass split-K
reduction at high context lengths when KV is narrow. At 131K context with 8x GQA, throughput is
0.70x FP16. At no-GQA ratios the overhead disappears (0.98x). This is a known architectural
limitation of the current kernel design.

**mx.fast.metal_kernel ceiling.** The kernels are written in MSL via `mx.fast.metal_kernel()`.
This is the correct path for rapid iteration but cannot beat a native C++ kernel registered in the
MLX operation registry. Maximum possible throughput is bounded by what the Python-level dispatch
overhead allows. A future upstream C++ kernel (mlx PR #3307 path) would eliminate that ceiling.

**Upstream C++ path not yet merged.** CC-Yeh's quantized SDPA work in mlx (PR #3026) would provide
a native quantized path with lower dispatch overhead. This package is the interim solution while
that PR is in review.

**Contiguous memory required.** `packed_k` and `packed_v` must be contiguous in memory.
`ensure_row_contiguous=True` is set on the kernel, so non-contiguous inputs trigger an implicit
copy. The cache layer is responsible for producing contiguous tensors (or accepting the copy cost).

**No 2-bit support.** 4-bit and 8-bit affine quantization only. 2-bit affine has poor quality for
KV cache. Non-uniform codebooks (e.g., RaBitQ) are a different kernel architecture and are not in
scope for this version.

**Prefill is not accelerated.** Prefill is compute-bound, not memory-bandwidth-bound. For qL > 1,
the function dequantizes K/V and calls the standard `mx.fast.scaled_dot_product_attention`. There
is no benefit to running a quantized kernel during prefill.

## License

Apache 2.0. See LICENSE.
