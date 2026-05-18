"""Shared helpers for mlx-qsdpa cache implementations."""

from __future__ import annotations

import mlx.core as mx

def _create_causal_mask(N, offset, window_size=None, left_padding=None):
    """Create a causal attention mask.

    Args:
        N: query sequence length
        offset: scalar or per-batch array of offsets
        window_size: sliding window size (optional)
        left_padding: per-batch left padding array (optional, for batch masks)
    """
    if isinstance(offset, mx.array) and offset.ndim > 0:
        # Batched: offset is (B,), produce (B, 1, N, offset.max()+N)
        max_off = offset.max().item()
        total = max_off + N
        q_pos = mx.arange(N)[None, :] + offset[:, None]  # (B, N)
        k_pos = mx.arange(total)[None, :]  # (1, total)
        mask = q_pos[:, :, None] >= k_pos[:, None, :]  # (B, N, total)
        if left_padding is not None:
            mask = mask & (k_pos[:, None, :] >= left_padding[:, None, None])
        if window_size is not None:
            mask = mask & (q_pos[:, :, None] - k_pos[:, None, :] < window_size)
        return mask[:, None, :, :]  # (B, 1, N, total)
    # Scalar path
    q_indices = mx.arange(offset, offset + N)
    k_indices = mx.arange(offset + N)
    mask = q_indices[:, None] >= k_indices[None]
    if window_size is not None:
        window_mask = q_indices[:, None] - k_indices[None] < window_size
        mask = mask & window_mask
    return mask


def _dynamic_roll(x, shifts, axis):
    n = x.shape[axis]
    expand_shifts = (...,) + (None,) * (x.ndim - axis)
    expand_indices = expand_shifts[:-1]
    idx = (mx.arange(n)[expand_indices] - shifts[expand_shifts]) % n
    return mx.take_along_axis(x, idx, axis=axis)


