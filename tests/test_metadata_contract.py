from __future__ import annotations

import importlib.metadata
import re
import tomllib
from pathlib import Path

import mlx.core as mx


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PACKAGE_ROOT.parents[1]
MLX_FLOOR = "0.31.2"


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3])


def test_mlx_qsdpa_declares_runtime_verified_mlx_floor():
    pyproject = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert f"mlx>={MLX_FLOOR}" in dependencies
    assert _version_tuple(importlib.metadata.version("mlx")) >= _version_tuple(
        MLX_FLOOR
    )
    assert hasattr(mx.fast, "metal_kernel")
    assert hasattr(mx.fast, "scaled_dot_product_attention")
    assert hasattr(mx, "quantize")
    assert hasattr(mx, "dequantize")


def test_mlx_floor_is_documented_consistently():
    readme = (PACKAGE_ROOT / "README.md").read_text()
    vendor_readme = (RUNTIME_ROOT / "patches/vendors/README.md").read_text()

    assert f"mlx >= {MLX_FLOOR}" in readme
    assert f"`mlx>={MLX_FLOOR}`" in vendor_readme

    stale_floor_pattern = re.compile(r"mlx\s*>?=\s*0\.21\.0")
    assert not stale_floor_pattern.search(readme)
    assert not stale_floor_pattern.search(vendor_readme)
