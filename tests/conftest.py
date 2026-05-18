"""Shared fixtures for mlx-qsdpa tests."""

from pathlib import Path
import sys

import mlx.core as mx
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _ensure_source_package() -> None:
    package_root_str = str(PACKAGE_ROOT)
    if package_root_str in sys.path:
        sys.path.remove(package_root_str)
    sys.path.insert(0, package_root_str)

    loaded_package = sys.modules.get("mlx_qsdpa")
    if loaded_package is None:
        return

    loaded_path = Path(getattr(loaded_package, "__file__", "")).resolve()
    if PACKAGE_ROOT in loaded_path.parents:
        return

    for module_name in list(sys.modules):
        if module_name == "mlx_qsdpa" or module_name.startswith("mlx_qsdpa."):
            sys.modules.pop(module_name, None)


_ensure_source_package()


def pytest_runtest_setup(item):
    _ensure_source_package()


@pytest.fixture
def seed():
    mx.random.seed(42)


def make_qkv(B, H_q, H_kv, N, D, bits=4, group_size=32, symmetric=False):
    """Create random query + quantized KV for testing.

    Returns (q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases,
             k_ref, v_ref) where k_ref/v_ref are the dequantized references.
    """
    q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

    # Generate random FP16 K/V, then quantize
    k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

    packed_k, k_scales, k_biases = mx.quantize(k_fp16, group_size=group_size, bits=bits)
    packed_v, v_scales, v_biases = mx.quantize(v_fp16, group_size=group_size, bits=bits)

    # Dequantize to get the true reference (quantization is lossy)
    k_ref = mx.dequantize(packed_k, k_scales, k_biases, group_size=group_size, bits=bits)
    v_ref = mx.dequantize(packed_v, v_scales, v_biases, group_size=group_size, bits=bits)

    if symmetric:
        # Re-quantize, then dequant with zero biases for symmetric reference
        packed_k_s, k_scales_s, _ = mx.quantize(k_fp16, group_size=group_size, bits=bits)
        packed_v_s, v_scales_s, _ = mx.quantize(v_fp16, group_size=group_size, bits=bits)
        k_ref = mx.dequantize(
            packed_k_s, k_scales_s,
            mx.zeros_like(k_scales_s),
            group_size=group_size, bits=bits,
        )
        v_ref = mx.dequantize(
            packed_v_s, v_scales_s,
            mx.zeros_like(v_scales_s),
            group_size=group_size, bits=bits,
        )
        packed_k, k_scales = packed_k_s, k_scales_s
        packed_v, v_scales = packed_v_s, v_scales_s
        k_biases = None
        v_biases = None

    mx.eval(q, packed_k, packed_v, k_scales, v_scales, k_ref, v_ref)
    if k_biases is not None:
        mx.eval(k_biases, v_biases)

    return q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases, k_ref, v_ref


def make_qkv_prefill(B, H_q, H_kv, N, D, qL, bits=4, group_size=32):
    """Create random query (qL>1) + quantized KV for prefill testing.

    Returns (q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases,
             k_ref, v_ref) where k_ref/v_ref are the dequantized references.
    """
    q = mx.random.normal((B, H_q, qL, D)).astype(mx.float16)

    k_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v_fp16 = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)

    packed_k, k_scales, k_biases = mx.quantize(k_fp16, group_size=group_size, bits=bits)
    packed_v, v_scales, v_biases = mx.quantize(v_fp16, group_size=group_size, bits=bits)

    k_ref = mx.dequantize(packed_k, k_scales, k_biases, group_size=group_size, bits=bits)
    v_ref = mx.dequantize(packed_v, v_scales, v_biases, group_size=group_size, bits=bits)

    mx.eval(q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases, k_ref, v_ref)

    return q, packed_k, packed_v, k_scales, v_scales, k_biases, v_biases, k_ref, v_ref


def reference_sdpa(q, k_ref, v_ref, scale=None):
    """Gold standard: FP16 SDPA via MLX's built-in kernel."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return mx.fast.scaled_dot_product_attention(q, k_ref, v_ref, scale=scale)
