"""Compatibility entrypoint for ``python -m mlx_qsdpa.bench_comparison``."""

from mlx_qsdpa.benchmarks.bench_comparison import *  # noqa: F401,F403
from mlx_qsdpa.benchmarks.bench_comparison import main


if __name__ == "__main__":
    main()
