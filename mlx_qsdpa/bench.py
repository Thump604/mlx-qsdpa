"""Standalone benchmark for mlx-qsdpa kernel performance.

Usage:
    python -m mlx_qsdpa.bench
    python -m mlx_qsdpa.bench --bits 8 --context 32768
    python -m mlx_qsdpa.bench --sweep
"""

import argparse
import time
import mlx.core as mx


def benchmark_kernel(B, H_q, H_kv, N, D, bits, group_size, num_iters=100, warmup=10):
    """Benchmark quantized SDPA kernel vs FP16 baseline."""
    from mlx_qsdpa import quantized_sdpa

    elems_per_int = 32 // bits
    num_groups = D // group_size

    # Create random inputs
    q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

    packed_k, k_scales, k_biases = mx.quantize(k_fp16, group_size=group_size, bits=bits)
    packed_v, v_scales, v_biases = mx.quantize(v_fp16, group_size=group_size, bits=bits)
    mx.eval(q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases, k_fp16, v_fp16)

    scale = D ** -0.5

    # Baseline: FP16 SDPA
    for _ in range(warmup):
        out = mx.fast.scaled_dot_product_attention(q, k_fp16, v_fp16, scale=scale)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(num_iters):
        out = mx.fast.scaled_dot_product_attention(q, k_fp16, v_fp16, scale=scale)
        mx.eval(out)
    fp16_time = (time.perf_counter() - start) / num_iters

    # Quantized SDPA
    for _ in range(warmup):
        out = quantized_sdpa(q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases,
                             bits=bits, group_size=group_size)
        mx.eval(out)

    start = time.perf_counter()
    for _ in range(num_iters):
        out = quantized_sdpa(q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases,
                             bits=bits, group_size=group_size)
        mx.eval(out)
    quant_time = (time.perf_counter() - start) / num_iters

    # Bandwidth calculations
    fp16_bytes = B * H_kv * N * D * 2 * 2  # K+V, 2 bytes each (float16)
    quant_bytes = B * H_kv * N * (D // elems_per_int * 4 + num_groups * 2 * 2) * 2  # packed(4B)+scales+biases(2B each), K+V

    return {
        "fp16_ms": fp16_time * 1000,
        "quant_ms": quant_time * 1000,
        "speedup": fp16_time / quant_time if quant_time > 0 else 0,
        "fp16_bw_gbs": fp16_bytes / fp16_time / 1e9 if fp16_time > 0 else 0,
        "quant_bw_gbs": quant_bytes / quant_time / 1e9 if quant_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="mlx-qsdpa benchmark")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Run full sweep")
    args = parser.parse_args()

    mx.random.seed(42)

    if args.sweep:
        print(f"{'Context':>8} {'Bits':>4} {'FP16 ms':>10} {'Quant ms':>10} "
              f"{'Speedup':>8} {'FP16 BW':>10} {'Quant BW':>10}")
        print("-" * 72)
        for N in [1024, 4096, 8192, 32768, 65536, 131072]:
            for bits in [4, 8]:
                result = benchmark_kernel(
                    args.batch, args.q_heads, args.kv_heads, N,
                    args.head_dim, bits, args.group_size, num_iters=50,
                )
                print(f"{N:>8} {bits:>4} {result['fp16_ms']:>10.3f} "
                      f"{result['quant_ms']:>10.3f} {result['speedup']:>8.2f}x "
                      f"{result['fp16_bw_gbs']:>9.1f}G "
                      f"{result['quant_bw_gbs']:>9.1f}G")
    else:
        result = benchmark_kernel(
            args.batch, args.q_heads, args.kv_heads, args.context,
            args.head_dim, args.bits, args.group_size, num_iters=args.iters,
        )
        print(f"Config: B={args.batch} H_q={args.q_heads} H_kv={args.kv_heads} "
              f"N={args.context} D={args.head_dim} bits={args.bits}")
        print(f"FP16:  {result['fp16_ms']:.3f} ms  ({result['fp16_bw_gbs']:.1f} GB/s)")
        print(f"Quant: {result['quant_ms']:.3f} ms  ({result['quant_bw_gbs']:.1f} GB/s)")
        print(f"Speedup: {result['speedup']:.2f}x")


if __name__ == "__main__":
    main()
