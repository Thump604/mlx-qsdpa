"""Tests for the mlx-qsdpa benchmark entrypoints."""

from __future__ import annotations

import math


def test_compat_bench_entrypoint_reexports_package_implementation():
    from mlx_qsdpa import bench as compat_bench
    from mlx_qsdpa.benchmarks import bench as package_bench

    assert compat_bench.main is package_bench.main
    assert compat_bench.benchmark_kernel is package_bench.benchmark_kernel


def test_benchmark_kernel_reports_required_metrics(seed):
    from mlx_qsdpa.benchmarks.bench import benchmark_kernel

    result = benchmark_kernel(
        B=1,
        H_q=1,
        H_kv=1,
        N=8,
        D=32,
        bits=4,
        group_size=32,
        num_iters=1,
        warmup=0,
    )

    assert set(result) == {
        "fp16_ms",
        "quant_ms",
        "speedup",
        "fp16_bw_gbs",
        "quant_bw_gbs",
    }
    assert result["fp16_ms"] > 0
    assert result["quant_ms"] > 0
    assert math.isfinite(result["speedup"])
    assert result["fp16_bw_gbs"] > 0
    assert result["quant_bw_gbs"] > 0


def test_main_runs_single_small_config(seed, capsys):
    from mlx_qsdpa.benchmarks.bench import main

    main(
        [
            "--bits",
            "4",
            "--group-size",
            "32",
            "--context",
            "8",
            "--head-dim",
            "32",
            "--q-heads",
            "1",
            "--kv-heads",
            "1",
            "--batch",
            "1",
            "--iters",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "Config: B=1 H_q=1 H_kv=1 N=8 D=32 bits=4" in output
    assert "FP16:" in output
    assert "Quant:" in output
    assert "Speedup:" in output
