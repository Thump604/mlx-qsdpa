"""Compatibility entrypoint for ``python -m mlx_qsdpa.bench``."""

from mlx_qsdpa.benchmarks.bench import *  # noqa: F401,F403
from mlx_qsdpa.benchmarks.bench import main


if __name__ == "__main__":
    main()
