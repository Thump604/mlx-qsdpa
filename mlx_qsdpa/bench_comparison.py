"""Three-way attention benchmark: FP16 vs mlx-lm quantized_matmul vs fused kernel.

Proves that fused quantized SDPA (single dispatch) beats mlx-lm's two-call path
(quantized_matmul x2 + softmax) on Apple Silicon unified memory.

Usage:
    python -m mlx_qsdpa.bench_comparison
    python -m mlx_qsdpa.bench_comparison --decode-only --seq-len 16384,65536
    python -m mlx_qsdpa.bench_comparison --heads 2,2 --json

The mlx-lm quantized attention path is reimplemented inline (~20 lines) to
avoid an external dependency. The reimplementation matches
mlx_lm.models.base.quantized_scaled_dot_product_attention as of mlx-lm v0.31.1
(commit 4be9c29, branch feat/probabilistic-mtp). See mlx_lm_quantized_sdpa().
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from mlx_qsdpa import quantized_sdpa

M2_ULTRA_PEAK_GBPS = 800.0


def mlx_lm_quantized_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    scale: float,
    bits: int = 4,
    group_size: int = 32,
    mask=None,
) -> mx.array:
    """Reimplementation of mlx_lm.models.base.quantized_scaled_dot_product_attention.

    Source: mlx-lm v0.31.1, mlx_lm/models/base.py:64-105.

    Two mx.quantized_matmul calls (Q*K^T, scores*V) + softmax.
    This is the path Package 1's fused kernel aims to beat.
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tuple(mx.expand_dims(x, axis=-3) for x in q_keys)
        q_values = tuple(mx.expand_dims(x, axis=-3) for x in q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def compute_bytes(B, H_q, H_kv, N, D, qL, path, bits=4, group_size=32):
    """Compute bytes read and written for a single attention call.

    Args:
        path: "fp16", "quantized", or "fused" (quantized and fused have same byte count)

    Returns:
        (bytes_read, bytes_written)
    """
    # Output: always FP16
    bytes_written = B * H_q * qL * D * 2

    # Query: always FP16
    q_bytes = B * H_q * qL * D * 2

    if path == "fp16":
        # K and V: FP16
        kv_bytes = B * H_kv * N * D * 2 * 2  # K + V
        bytes_read = q_bytes + kv_bytes
    else:
        # Quantized: packed + scales + biases for K and V
        elems_per_int = 32 // bits
        pack_D = D // elems_per_int
        num_groups = D // group_size
        per_kv = B * H_kv * N * (pack_D * 4 + num_groups * 2 + num_groups * 2)
        bytes_read = q_bytes + per_kv * 2  # K + V

    return bytes_read, bytes_written


def compute_bandwidth_gbps(total_bytes, median_us):
    """Compute achieved bandwidth in GB/s.

    Args:
        total_bytes: bytes_read + bytes_written
        median_us: median latency in microseconds
    """
    return total_bytes / (median_us * 1e-6) / 1e9


def run_measurement(B, H_q, H_kv, N, D, qL, bits, group_size,
                    num_iters=100, warmup=10):
    """Run all three paths and return measurement results.

    Returns list of dicts, one per path.
    """
    scale = D ** -0.5

    # Generate inputs
    q = mx.random.normal((B, H_q, qL, D)).astype(mx.float16)
    k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

    # Quantize once (not timed)
    pk, ks, kb = mx.quantize(k_fp16, group_size=group_size, bits=bits)
    pv, vs, vb = mx.quantize(v_fp16, group_size=group_size, bits=bits)

    # Dequantize for FP16 baseline (use dequantized, not original, for fair comparison)
    k_deq = mx.dequantize(pk, ks, kb, group_size=group_size, bits=bits)
    v_deq = mx.dequantize(pv, vs, vb, group_size=group_size, bits=bits)

    mx.eval(q, pk, pv, ks, vs, kb, vb, k_deq, v_deq, k_fp16, v_fp16)

    results = []

    # Define paths
    paths = {
        "fp16": lambda: mx.fast.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=scale
        ),
        "quantized_mm": lambda: mlx_lm_quantized_sdpa(
            q, (pk, ks, kb), (pv, vs, vb),
            scale=scale, bits=bits, group_size=group_size,
        ),
        "fused": lambda: quantized_sdpa(
            q, pk, pv, ks, vs, kb, vb,
            bits=bits, group_size=group_size,
        ),
    }

    for path_name, fn in paths.items():
        # Warmup
        for _ in range(warmup):
            out = fn()
            mx.eval(out)

        # Timed iterations
        timings = []
        for _ in range(num_iters):
            start = time.perf_counter()
            out = fn()
            mx.eval(out)
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1e6)  # to microseconds

        timings.sort()
        median_us = timings[len(timings) // 2]
        p5_us = timings[int(len(timings) * 0.05)]
        p95_us = timings[int(len(timings) * 0.95)]

        byte_path = "fp16" if path_name == "fp16" else "quantized"
        bytes_read, bytes_written = compute_bytes(
            B, H_q, H_kv, N, D, qL, byte_path, bits, group_size
        )
        total_bytes = bytes_read + bytes_written
        gbps = compute_bandwidth_gbps(total_bytes, median_us)

        gqa = "non-gqa" if H_q == H_kv else f"gqa-{H_q // H_kv}x"
        results.append({
            "config": gqa,
            "path": path_name,
            "B": B, "H_q": H_q, "H_kv": H_kv, "D": D,
            "N": N, "qL": qL, "bits": bits, "group_size": group_size,
            "median_us": round(median_us, 2),
            "p5_us": round(p5_us, 2),
            "p95_us": round(p95_us, 2),
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "achieved_gbps": round(gbps, 2),
            "pct_peak": round(gbps / M2_ULTRA_PEAK_GBPS * 100, 1),
        })

    return results


# Sweep defaults
DECODE_SEQ_LENS = [1024, 4096, 16384, 32768, 65536, 131072]
PREFILL_SEQ_LENS = [4096, 16384, 65536]
PREFILL_QL = [128, 512]
HEAD_CONFIGS = [(2, 2), (32, 2)]  # (H_q, H_kv): non-GQA, GQA-16x
GROUP_SIZES_HEADLINE = [32]
GROUP_SIZES_ALL = [32, 64, 128]
D = 256
B = 1


def build_configs(args):
    """Build list of (B, H_q, H_kv, N, D, qL, bits, gs) from CLI args."""
    configs = []

    # Parse filters
    seq_filter = None
    if args.seq_len:
        seq_filter = set(int(x) for x in args.seq_len.split(","))

    head_filter = None
    if args.heads:
        parts = args.heads.split(",")
        head_filter = (int(parts[0]), int(parts[1]))

    head_cfgs = [head_filter] if head_filter else HEAD_CONFIGS

    if not args.prefill_only:
        # Decode sweep
        for hq, hkv in head_cfgs:
            gs_list = GROUP_SIZES_ALL if not args.headline_only else GROUP_SIZES_HEADLINE
            for gs in gs_list:
                for N in DECODE_SEQ_LENS:
                    if seq_filter and N not in seq_filter:
                        continue
                    configs.append((B, hq, hkv, N, D, 1, 4, gs))

    if not args.decode_only:
        # Prefill sweep
        for hq, hkv in head_cfgs:
            for qL in PREFILL_QL:
                for N in PREFILL_SEQ_LENS:
                    if seq_filter and N not in seq_filter:
                        continue
                    configs.append((B, hq, hkv, N, D, qL, 4, 32))

    return configs


def format_config(H_q, H_kv, bits, gs):
    gqa = "non-gqa" if H_q == H_kv else f"gqa-{H_q // H_kv}x"
    return f"{gqa} {bits}b gs{gs}"


def print_header():
    print(f"{'config':<20} {'path':<14} {'N':>7} {'qL':>4} "
          f"{'median':>9} {'p5':>9} {'p95':>9} {'GB/s':>8} {'%peak':>6}")
    print("-" * 96)


def print_row(r):
    cfg = format_config(r["H_q"], r["H_kv"], r["bits"], r["group_size"])
    print(f"{cfg:<20} {r['path']:<14} {r['N']:>7} {r['qL']:>4} "
          f"{r['median_us']:>8.1f}u {r['p5_us']:>8.1f}u {r['p95_us']:>8.1f}u "
          f"{r['achieved_gbps']:>7.1f} {r['pct_peak']:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Three-way attention benchmark: FP16 vs quantized_matmul vs fused"
    )
    parser.add_argument("--decode-only", action="store_true",
                        help="Skip prefill sweep")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Skip decode sweep")
    parser.add_argument("--headline-only", action="store_true",
                        help="Only gs=32 (skip gs=64,128 appendix)")
    parser.add_argument("--seq-len", type=str, default=None,
                        help="Filter sequence lengths (comma-separated)")
    parser.add_argument("--heads", type=str, default=None,
                        help="Filter head config as H_q,H_kv (e.g., 2,2 or 32,2)")
    parser.add_argument("--iters", type=int, default=100,
                        help="Timed iterations per measurement (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON lines to stdout")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON lines to file")
    args = parser.parse_args()

    mx.random.seed(42)

    configs = build_configs(args)
    if not configs:
        print("No configurations match filters.", file=sys.stderr)
        sys.exit(1)

    all_results = []

    if not args.json:
        print_header()

    for cfg_b, hq, hkv, N, d, qL, bits, gs in configs:
        measurements = run_measurement(
            cfg_b, hq, hkv, N, d, qL, bits, gs,
            num_iters=args.iters, warmup=args.warmup,
        )
        for r in measurements:
            all_results.append(r)
            if args.json:
                print(json.dumps(r))
            else:
                print_row(r)

    # Write to file if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        if not args.json:
            print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
