"""Batch-aware quantized SDPA cache."""

from __future__ import annotations

from typing import List

import mlx.core as mx

from mlx_qsdpa.cache_common import _create_causal_mask, _dynamic_roll
from mlx_qsdpa.single_cache import QuantizedSDPACache

class BatchQuantizedSDPACache:
    """Batch-aware quantized KV cache for BatchedEngine / continuous batching.

    Mirrors mlx-lm's BatchKVCache but stores K/V in quantized format.
    Handles left-padding, per-sequence offsets, and batch management
    (filter, extend, extract).

    Created by runtime_patches when ``to_batch_cache`` encounters a
    ``QuantizedSDPACache``. Not instantiated directly.
    """

    _use_fused_sdpa = True
    step = 256

    def __init__(self, left_padding: List[int], bits: int = 4, group_size: int = 32):
        self.bits = bits
        self.group_size = group_size
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self._keys = None    # (packed, scales, biases) or None
        self._values = None
        self._right_padding = None

    def empty(self):
        return self._keys is None

    @property
    def nbytes(self):
        if self._keys is None:
            return 0
        return sum(t.nbytes for t in self._keys) + sum(t.nbytes for t in self._values)

    def is_trimmable(self):
        return True

    def size(self):
        return self._idx

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def update_and_fetch(self, keys, values):
        """Quantize and append. Same contract as QuantizedSDPACache."""
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self._keys is None or (prev + num_steps) > self._keys[0].shape[2]:
            el_per_int = 32 // self.bits
            new_alloc = ((self.step + num_steps - 1) // self.step) * self.step
            shape = (B, n_kv_heads, new_alloc)

            def init_q(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            if self._keys is not None:
                if prev % self.step != 0:
                    self._keys = tuple(t[..., :prev, :] for t in self._keys)
                    self._values = tuple(t[..., :prev, :] for t in self._values)
                new_k = init_q(k_head_dim)
                new_v = init_q(v_head_dim)
                self._keys = tuple(mx.concatenate([o, n], axis=2)
                                   for o, n in zip(self._keys, new_k))
                self._values = tuple(mx.concatenate([o, n], axis=2)
                                     for o, n in zip(self._values, new_v))
            else:
                self._keys = init_q(k_head_dim)
                self._values = init_q(v_head_dim)

        self.offset += num_steps
        self._idx += num_steps

        new_k = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        new_v = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(3):
            self._keys[i][..., prev:self._idx, :] = new_k[i]
            self._values[i][..., prev:self._idx, :] = new_v[i]

        k_out = tuple(t[..., :self._idx, :] for t in self._keys)
        v_out = tuple(t[..., :self._idx, :] for t in self._values)
        return k_out, v_out

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self._keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedSDPACache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self._keys = tuple(
                _dynamic_roll(t, padding[:, None], axis=2) for t in self._keys
            )
            self._values = tuple(
                _dynamic_roll(t, padding[:, None], axis=2) for t in self._values
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        if self._keys is None:
            return None, None, self.offset, self.left_padding

        keys, values = self._keys, self._values
        if self._idx < keys[0].shape[2]:
            keys = tuple(t[..., : self._idx, :] for t in keys)
            values = tuple(t[..., : self._idx, :] for t in values)
        return keys, values, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self._keys, self._values, self.offset, self.left_padding = v
        if self._keys is None:
            self._idx = 0
        else:
            self._idx = self._keys[0].shape[2]

    def make_mask(self, N, return_array=False, **kwargs):
        """Causal mask with left-padding awareness."""
        return _create_causal_mask(
            N,
            offset=self.offset,
            left_padding=self.left_padding,
            window_size=kwargs.get("window_size"),
        )

    def filter(self, batch_indices):
        """Keep only the given batch indices."""
        if self._keys is not None:
            self._keys = tuple(t[batch_indices] for t in self._keys)
            self._values = tuple(t[batch_indices] for t in self._values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_pad = self.left_padding.min().item()
        if min_pad > 0:
            self._keys = tuple(t[..., min_pad:, :] for t in self._keys)
            self._values = tuple(t[..., min_pad:, :] for t in self._values)
            self._idx -= min_pad
            self.left_padding -= min_pad

    def extend(self, other):
        """Merge another batch cache into this one."""
        max_idx = max(self._idx, other._idx)
        max_size = max(self._keys[0].shape[2], other._keys[0].shape[2])

        def pad_quant(c, c_idx, tuples):
            left = max_idx - c_idx
            right = max_size - tuples[0].shape[2] - left
            result = []
            for t in tuples:
                tr = t
                if right < 0:
                    tr = tr[..., :right, :]
                    right_actual = 0
                else:
                    right_actual = right
                if left != 0 or right_actual != 0:
                    pad_spec = [(0, 0)] * (tr.ndim - 2) + [(left, right_actual), (0, 0)]
                    tr = mx.pad(tr, pad_spec)
                result.append(tr)
            return tuple(result), left

        sk, s_left = pad_quant(self, self._idx, self._keys)
        sv, _ = pad_quant(self, self._idx, self._values)
        ok, o_left = pad_quant(other, other._idx, other._keys)
        ov, _ = pad_quant(other, other._idx, other._values)

        self._keys = tuple(mx.concatenate([s, o]) for s, o in zip(sk, ok))
        self._values = tuple(mx.concatenate([s, o]) for s, o in zip(sv, ov))
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate([
            self.left_padding + s_left,
            other.left_padding + o_left,
        ])
        self._idx = max_idx

    def extract(self, idx):
        """Extract a single request as a QuantizedSDPACache."""
        cache = QuantizedSDPACache(bits=self.bits, group_size=self.group_size)
        if self._keys is None:
            return cache
        padding = self.left_padding[idx].item()
        cache._keys = tuple(
            mx.contiguous(t[idx:idx + 1, :, padding:self._idx])
            for t in self._keys
        )
        cache._values = tuple(
            mx.contiguous(t[idx:idx + 1, :, padding:self._idx])
            for t in self._values
        )
        cache.offset = self._idx - padding
        return cache

    @classmethod
    def merge(cls, caches):
        """Merge multiple QuantizedSDPACache instances into a batch."""
        if not caches:
            return cls([0])
        bits = caches[0].bits
        gs = caches[0].group_size
        lengths = [c.offset for c in caches]
        max_len = max(lengths)
        padding = [max_len - l for l in lengths]

        batch = cls(padding, bits=bits, group_size=gs)
        if all(c.empty() for c in caches):
            return batch

        B = len(caches)
        ref = next(c for c in caches if not c.empty())
        H = ref._keys[0].shape[1]
        el_per_int = 32 // bits

        # Infer dims from first non-empty cache
        k_pack_d = ref._keys[0].shape[3]
        k_gs_d = ref._keys[1].shape[3]
        v_pack_d = ref._values[0].shape[3]
        v_gs_d = ref._values[1].shape[3]
        dtype = ref._keys[1].dtype

        def alloc(pack_d, gs_d):
            return (
                mx.zeros((B, H, max_len, pack_d), dtype=mx.uint32),
                mx.zeros((B, H, max_len, gs_d), dtype=dtype),
                mx.zeros((B, H, max_len, gs_d), dtype=dtype),
            )

        batch._keys = alloc(k_pack_d, k_gs_d)
        batch._values = alloc(v_pack_d, v_gs_d)

        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.empty():
                continue
            for j in range(3):
                batch._keys[j][i:i + 1, :, p:p + c.offset] = c._keys[j][..., :c.offset, :]
                batch._values[j][i:i + 1, :, p:p + c.offset] = c._values[j][..., :c.offset, :]

        batch.offset += max_len
        batch._idx = max_len
        return batch


