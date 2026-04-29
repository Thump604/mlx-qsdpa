"""Quantized KV cache that stores keys/values in mx.quantize format.

Compatible with mlx-qsdpa's quantized_sdpa kernel. The consumer is
responsible for routing attention calls based on cache type.

Provides both single-request (QuantizedSDPACache) and batch-aware
(BatchQuantizedSDPACache) implementations.
"""

from typing import List

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


class QuantizedSDPACache:
    """KV cache storing quantized keys and values.

    Follows mlx-lm's _BaseCache protocol. ``update_and_fetch`` returns
    quantized tuples ``(packed_uint32, scales_fp16, biases_fp16)`` for
    both keys and values -- NOT plain float16 tensors. The consumer
    must check ``hasattr(cache, 'bits')`` and dispatch to
    ``quantized_sdpa()`` accordingly.

    Args:
        bits: Quantization bit width (4 or 8).
        group_size: Elements per quantization group (32, 64, or 128).
        step: Buffer pre-allocation step size.
    """

    # Marker for runtime_patches SDPA router. Distinguishes this cache
    # from mlx-lm's built-in QuantizedKVCache.
    _use_fused_sdpa = True

    def __init__(self, bits: int = 4, group_size: int = 32, step: int = 256):
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.step = step
        self.offset = 0
        self._keys = None   # (packed, scales, biases) or None
        self._values = None  # (packed, scales, biases) or None

    def empty(self):
        """Return True if the cache has no stored tokens."""
        return self._keys is None

    @property
    def nbytes(self):
        """Return total bytes used by cached tensors."""
        if self._keys is None:
            return 0
        total = 0
        for tensors in (self._keys, self._values):
            for t in tensors:
                total += t.nbytes
        return total

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        """Trim n tokens from the end of the cache."""
        n = min(self.offset, n)
        self.offset -= n
        return n

    def rewind(self, num_to_trim):
        """Rewind the cache by removing the last num_to_trim tokens."""
        num_to_trim = min(self.offset, num_to_trim)
        self.offset -= num_to_trim
        return num_to_trim > 0

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask compatible with mlx-lm's protocol."""
        if window_size is not None:
            return _create_causal_mask(N, self.offset, window_size=window_size)
        elif N == 1:
            return None
        elif return_array:
            return _create_causal_mask(N, self.offset)
        else:
            return "causal"

    @property
    def state(self):
        """Return cached tensors for serialization."""
        if self._keys is None:
            return None, None
        k = tuple(t[..., :self.offset, :] for t in self._keys)
        v = tuple(t[..., :self.offset, :] for t in self._values)
        return k, v

    @state.setter
    def state(self, v):
        keys, values = v
        self._keys = keys
        self._values = values

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.bits, self.group_size)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset, self.bits, self.group_size = map(int, v)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj.step = 256
        obj.meta_state = meta_state
        obj.state = state
        return obj

    @classmethod
    def merge(cls, caches):
        """Merge single-request quantized caches into a batch-aware cache."""
        return BatchQuantizedSDPACache.merge(caches)

    def update_and_fetch(self, keys, values):
        """Quantize and append keys/values to cache.

        Args:
            keys:   (B, H_kv, num_steps, D) float16/bfloat16
            values: (B, H_kv, num_steps, D) float16/bfloat16

        Returns:
            (keys_quant, values_quant) where each is a tuple of
            (packed_uint32, scales_fp16, biases_fp16) covering
            all cached tokens [0 : self.offset].
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self._keys is None or (prev + num_steps) > self._keys[0].shape[2]:
            self._grow(B, n_kv_heads, num_steps, k_head_dim, v_head_dim,
                       keys.dtype, prev)

        self.offset += num_steps

        # Quantize new tokens
        new_k = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        new_v = mx.quantize(values, group_size=self.group_size, bits=self.bits)

        # Write into pre-allocated buffers
        for i in range(3):
            self._keys[i][..., prev:self.offset, :] = new_k[i]
            self._values[i][..., prev:self.offset, :] = new_v[i]

        # Return sliced view up to current offset
        k_out = tuple(t[..., :self.offset, :] for t in self._keys)
        v_out = tuple(t[..., :self.offset, :] for t in self._values)
        return k_out, v_out

    def _grow(self, B, n_kv_heads, num_steps, k_head_dim, v_head_dim,
              dtype, prev):
        """Grow pre-allocated buffers to fit new tokens."""
        el_per_int = 32 // self.bits
        new_alloc = ((self.step + num_steps - 1) // self.step) * self.step
        shape = (B, n_kv_heads, new_alloc)

        def init_quant(dim):
            return (
                mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            )

        if self._keys is not None:
            # Trim excess allocation before concatenating
            if prev % self.step != 0:
                self._keys = tuple(t[..., :prev, :] for t in self._keys)
                self._values = tuple(t[..., :prev, :] for t in self._values)

            new_k_bufs = init_quant(k_head_dim)
            new_v_bufs = init_quant(v_head_dim)
            self._keys = tuple(
                mx.concatenate([old, new], axis=2)
                for old, new in zip(self._keys, new_k_bufs)
            )
            self._values = tuple(
                mx.concatenate([old, new], axis=2)
                for old, new in zip(self._values, new_v_bufs)
            )
        else:
            self._keys = init_quant(k_head_dim)
            self._values = init_quant(v_head_dim)


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


class QuantizedRotatingSDPACache:
    """Quantized KV cache with fixed-size circular buffer for sliding-window attention.

    Pre-allocates a buffer of max_size tokens in quantized format.
    Decode (S=1) writes quantized data at the write pointer and wraps.
    Prefill (S>1) dequantizes, concatenates, trims, re-quantizes.

    Args:
        max_size: Maximum number of tokens in the sliding window.
        keep: Reserved prefix tokens (must be 0 for now).
        bits: Quantization bit width (4 or 8).
        group_size: Elements per quantization group (32, 64, or 128).
    """

    _use_fused_sdpa = True

    def __init__(self, max_size: int, keep: int = 0, bits: int = 4, group_size: int = 32):
        assert keep == 0, "keep > 0 not yet supported for quantized rotating cache"
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        self.max_size = max_size
        self.keep = keep
        self.bits = bits
        self.group_size = group_size
        self.offset = 0
        self._idx = 0
        self._keys = None    # (packed_uint32, scales, biases) or None
        self._values = None

    def empty(self):
        return self._keys is None

    @property
    def nbytes(self):
        if self._keys is None:
            return 0
        return sum(t.nbytes for t in self._keys) + sum(t.nbytes for t in self._values)

    def size(self):
        return min(self.offset, self.max_size)

    def is_trimmable(self):
        return self.offset < self.max_size

    def _alloc(self, B, n_kv_heads, k_head_dim, v_head_dim, dtype):
        """Pre-allocate fixed-size quantized buffers."""
        el_per_int = 32 // self.bits
        shape = (B, n_kv_heads, self.max_size)

        def init_quant(dim):
            return (
                mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=dtype),
            )

        self._keys = init_quant(k_head_dim)
        self._values = init_quant(v_head_dim)

    def update_and_fetch(self, keys, values):
        """Quantize and store keys/values in the circular buffer.

        Args:
            keys:   (B, H_kv, S, D) float16/bfloat16
            values: (B, H_kv, S, D) float16/bfloat16

        Returns:
            (keys_quant, values_quant) where each is a 3-tuple of
            (packed_uint32, scales, biases) covering n_visible tokens.
        """
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        if self._keys is None:
            self._alloc(B, n_kv_heads, k_head_dim, v_head_dim, keys.dtype)
        if S == 1:
            return self._update_decode(keys, values)
        else:
            return self._update_prefill(keys, values)

    def _update_decode(self, keys, values):
        """Decode path (S=1): quantize and write at _idx, wrap if needed."""
        new_k = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        new_v = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        write_pos = self._idx % self.max_size
        for i in range(3):
            self._keys[i][..., write_pos:write_pos + 1, :] = new_k[i]
            self._values[i][..., write_pos:write_pos + 1, :] = new_v[i]
        self._idx += 1
        self.offset += 1
        # After the buffer is full and writes continue past max_size, wrap _idx.
        # At exactly max_size, _idx stays as max_size (boundary sentinel).
        if self._idx > self.max_size:
            self._idx = self._idx % self.max_size
        n_visible = min(self.offset, self.max_size)
        k_out = tuple(t[..., :n_visible, :] for t in self._keys)
        v_out = tuple(t[..., :n_visible, :] for t in self._values)
        return k_out, v_out

    def _update_prefill(self, keys, values):
        """Prefill path (S>1): dequant existing + concat + trim + requant."""
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        if self.offset > 0:
            existing_k = self._dequant_temporal(self._keys)
            existing_v = self._dequant_temporal(self._values)
            all_k = mx.concatenate([existing_k, keys], axis=2)
            all_v = mx.concatenate([existing_v, values], axis=2)
        else:
            all_k = keys
            all_v = values
        total = all_k.shape[2]
        if total > self.max_size:
            all_k = all_k[..., total - self.max_size:, :]
            all_v = all_v[..., total - self.max_size:, :]
            total = self.max_size
        q_k = mx.quantize(all_k, group_size=self.group_size, bits=self.bits)
        q_v = mx.quantize(all_v, group_size=self.group_size, bits=self.bits)
        for i in range(3):
            self._keys[i][..., :total, :] = q_k[i]
            self._values[i][..., :total, :] = q_v[i]
        self.offset += S
        self._idx = total % self.max_size
        n_visible = min(self.offset, self.max_size)
        k_out = tuple(t[..., :n_visible, :] for t in self._keys)
        v_out = tuple(t[..., :n_visible, :] for t in self._values)
        return k_out, v_out

    def _dequant_temporal(self, quant_tuple):
        """Dequantize buffer and reorder to temporal order if rotated."""
        n_visible = min(self.offset, self.max_size)
        packed = quant_tuple[0][..., :n_visible, :]
        scales = quant_tuple[1][..., :n_visible, :]
        biases = quant_tuple[2][..., :n_visible, :]
        fp = mx.dequantize(packed, scales, biases,
                           group_size=self.group_size, bits=self.bits)
        if self.offset > self.max_size and self._idx > 0:
            fp = mx.concatenate(
                [fp[..., self._idx:, :], fp[..., :self._idx, :]], axis=2
            )
        return fp

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask compatible with mlx-lm's protocol."""
        effective = min(self.offset, self.max_size)
        if N == 1:
            return None
        elif return_array or window_size is not None:
            return _create_causal_mask(N, effective, window_size=window_size)
        else:
            return "causal"

    def trim(self, n):
        """Trim n tokens from the end of the cache (pre-rotation only)."""
        n = min(self.offset, n)
        self.offset -= n
        self._idx -= n
        return n

    def rewind(self, num_to_trim):
        """Rewind the cache by removing the last num_to_trim tokens (pre-rotation only)."""
        if self.offset >= self.max_size:
            return False
        num_to_trim = min(self.offset, num_to_trim)
        self.offset -= num_to_trim
        self._idx -= num_to_trim
        return num_to_trim > 0

    @property
    def state(self):
        """Return cached tensors in temporal order for serialization."""
        if self._keys is None:
            return None, None
        n_visible = min(self.offset, self.max_size)
        if self.offset > self.max_size and self._idx > 0:
            def reorder(tuples):
                return tuple(
                    mx.concatenate([t[..., self._idx:n_visible, :],
                                    t[..., :self._idx, :]], axis=2)
                    for t in tuples
                )
            return reorder(self._keys), reorder(self._values)
        k = tuple(t[..., :n_visible, :] for t in self._keys)
        v = tuple(t[..., :n_visible, :] for t in self._values)
        return k, v

    @state.setter
    def state(self, v):
        """Restore cached tensors from a temporal-order snapshot.

        Expands compact state (n_visible tokens) back into max_size buffers
        so the decode path can write at _idx without bounds errors.
        """
        keys, values = v
        if keys is None:
            self._keys = None
            self._values = None
            return
        # keys/values are compact tuples of shape (B, H_kv, n_visible, dim_quant).
        # Re-expand to max_size by padding with zeros on the right.
        n_visible = keys[0].shape[2]
        pad_len = self.max_size - n_visible
        if pad_len <= 0:
            self._keys = keys
            self._values = values
            return

        def expand(tuples):
            result = []
            for t in tuples:
                if pad_len > 0:
                    pad_spec = [(0, 0)] * (t.ndim - 2) + [(0, pad_len), (0, 0)]
                    t = mx.pad(t, pad_spec)
                result.append(t)
            return tuple(result)

        self._keys = expand(keys)
        self._values = expand(values)

    @classmethod
    def merge(cls, caches):
        """Merge single-request rotating caches into a batch-aware cache."""
        return BatchQuantizedRotatingSDPACache.merge(caches)


class BatchQuantizedRotatingSDPACache:
    """Batch-aware quantized rotating KV cache for sliding-window attention.

    Fixed-size circular buffer with per-sequence offsets for batch management.
    Mirrors BatchQuantizedSDPACache but uses a pre-allocated circular buffer
    rather than growing storage, enabling bounded memory at long contexts.

    Args:
        left_padding: List of per-sequence left-padding token counts.
        max_size: Fixed circular buffer size (maximum tokens retained).
        keep: Reserved prefix tokens (must be 0).
        bits: Quantization bit width (4 or 8).
        group_size: Elements per quantization group.
    """

    _use_fused_sdpa = True
    step = 256

    def __init__(self, left_padding, max_size, keep=0, bits=4, group_size=32):
        assert keep == 0, "keep > 0 not yet supported"
        self.max_size = max_size
        self.keep = keep
        self.bits = bits
        self.group_size = group_size
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self._keys = None    # (packed_uint32, scales, biases) or None
        self._values = None

    def empty(self):
        return self._keys is None

    @property
    def nbytes(self):
        if self._keys is None:
            return 0
        return sum(t.nbytes for t in self._keys) + sum(t.nbytes for t in self._values)

    def size(self):
        return min(self._idx, self.max_size)

    def is_trimmable(self):
        return self._idx < self.max_size

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def update_and_fetch(self, keys, values):
        """Quantize and store keys/values in the circular buffer.

        Args:
            keys:   (B, H_kv, S, D) float16/bfloat16
            values: (B, H_kv, S, D) float16/bfloat16

        Returns:
            (keys_quant, values_quant) where each is a 3-tuple covering
            n_visible tokens from the circular buffer.
        """
        B, n_kv_heads, S, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        if self._keys is None:
            el_per_int = 32 // self.bits
            shape = (B, n_kv_heads, self.max_size)

            def init_q(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            self._keys = init_q(k_head_dim)
            self._values = init_q(v_head_dim)

        if S == 1:
            return self._update_decode(keys, values)
        return self._update_prefill(keys, values)

    def _update_decode(self, keys, values):
        """Decode path (S=1): quantize and write at the ring pointer."""
        self._ensure_decode_capacity(keys, values)
        new_k = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        new_v = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        write_pos = self._idx % self.max_size

        for i in range(3):
            self._keys[i][..., write_pos:write_pos + 1, :] = new_k[i]
            self._values[i][..., write_pos:write_pos + 1, :] = new_v[i]

        self.offset += 1
        self._idx += 1
        n_visible = min(self._idx, self.max_size)
        k_out = tuple(t[..., :n_visible, :] for t in self._keys)
        v_out = tuple(t[..., :n_visible, :] for t in self._values)
        return k_out, v_out

    def _ensure_decode_capacity(self, keys, values):
        """Grow compact merged buffers before a decode write.

        ``merge()`` intentionally stores only the visible tokens.  Decode then
        appends one token at a time, so the compact buffer must grow up to
        ``max_size`` before writing at ``_idx``.
        """
        current = self._keys[0].shape[2]
        if self._idx < current or current >= self.max_size:
            return

        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        grow_by = min(self.step, self.max_size - current)
        el_per_int = 32 // self.bits
        shape = (B, n_kv_heads, grow_by)

        def init_q(dim):
            return (
                mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
            )

        extra_k = init_q(k_head_dim)
        extra_v = init_q(v_head_dim)
        self._keys = tuple(
            mx.concatenate([old, extra], axis=2)
            for old, extra in zip(self._keys, extra_k)
        )
        self._values = tuple(
            mx.concatenate([old, extra], axis=2)
            for old, extra in zip(self._values, extra_v)
        )

    def _update_prefill(self, keys, values):
        """Prefill path (S>1): reorder visible tokens, append, trim, re-quantize."""
        if self._idx > 0:
            existing_k = self._dequant_temporal(self._keys)
            existing_v = self._dequant_temporal(self._values)
            all_k = mx.concatenate([existing_k, keys], axis=2)
            all_v = mx.concatenate([existing_v, values], axis=2)
        else:
            all_k = keys
            all_v = values

        total = all_k.shape[2]
        if total > self.max_size:
            all_k = all_k[..., total - self.max_size:, :]
            all_v = all_v[..., total - self.max_size:, :]
            total = self.max_size

        q_k = mx.quantize(all_k, group_size=self.group_size, bits=self.bits)
        q_v = mx.quantize(all_v, group_size=self.group_size, bits=self.bits)
        for i in range(3):
            self._keys[i][..., :total, :] = q_k[i]
            self._values[i][..., :total, :] = q_v[i]

        self.offset += keys.shape[2]
        self._idx = total
        k_out = tuple(t[..., :total, :] for t in self._keys)
        v_out = tuple(t[..., :total, :] for t in self._values)
        return k_out, v_out

    def _dequant_temporal(self, quant_tuple):
        """Dequantize visible tokens in chronological order."""
        n_visible = min(self._idx, self.max_size)
        packed = quant_tuple[0][..., :n_visible, :]
        scales = quant_tuple[1][..., :n_visible, :]
        biases = quant_tuple[2][..., :n_visible, :]
        fp = mx.dequantize(
            packed, scales, biases, group_size=self.group_size, bits=self.bits
        )
        if self._idx > self.max_size:
            write_pos = self._idx % self.max_size
            if write_pos > 0:
                fp = mx.concatenate(
                    [fp[..., write_pos:, :], fp[..., :write_pos, :]], axis=2
                )
        return fp

    def make_mask(self, N, return_array=False, **kwargs):
        """Causal mask with left-padding awareness.

        ``offset`` can track logical multimodal position progress, which may be
        larger than the visible K/V buffer length.  Attention mask width must
        match the cache tensors returned by ``update_and_fetch``.
        """
        window_size = kwargs.get("window_size") or self.max_size
        offset = min(self.max_size - 1, self._idx)
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]

        mask = linds >= rinds
        mask &= linds < rinds + window_size

        left_padding = self.left_padding
        if (trim_size := self._idx - self.max_size + int(N > 1)) > 0:
            left_padding = left_padding - trim_size

        rotated = N == 1 and self._idx >= self.max_size
        if rotated:
            left_padding = left_padding - 1

        mask = mask & (rinds >= mx.expand_dims(left_padding, (1, 2, 3)))

        if rotated:
            idx = self._idx % self.max_size
            mask = mx.roll(mask, shift=idx + 1, axis=-1)

        return mask

    def filter(self, batch_indices):
        """Keep only the given batch indices."""
        if self._keys is not None:
            self._keys = tuple(t[batch_indices] for t in self._keys)
            self._values = tuple(t[batch_indices] for t in self._values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        """Merge another batch rotating cache into this one (along the batch dimension)."""
        if self._keys is None and other._keys is None:
            self.offset = mx.concatenate([self.offset, other.offset])
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            return
        max_idx = max(self._idx, other._idx)

        def pad_quant(tuples, c_idx):
            left = max_idx - c_idx
            if left == 0:
                return tuples
            return tuple(
                mx.pad(t, [(0, 0)] * (t.ndim - 2) + [(left, 0), (0, 0)])
                for t in tuples
            )

        sk = pad_quant(self._keys, self._idx) if self._keys is not None else None
        sv = pad_quant(self._values, self._idx) if self._values is not None else None
        ok = pad_quant(other._keys, other._idx) if other._keys is not None else None
        ov = pad_quant(other._values, other._idx) if other._values is not None else None

        if sk is not None and ok is not None:
            self._keys = tuple(mx.concatenate([s, o]) for s, o in zip(sk, ok))
            self._values = tuple(mx.concatenate([s, o]) for s, o in zip(sv, ov))
        elif ok is not None:
            self._keys = ok
            self._values = ov

        s_left = max_idx - self._idx
        o_left = max_idx - other._idx
        self.offset = mx.concatenate([self.offset, other.offset])
        self.left_padding = mx.concatenate([
            self.left_padding + s_left,
            other.left_padding + o_left,
        ])
        self._idx = max_idx

    def extract(self, idx):
        """Extract a single request as a QuantizedRotatingSDPACache."""
        cache = QuantizedRotatingSDPACache(
            max_size=self.max_size, bits=self.bits, group_size=self.group_size,
        )
        if self._keys is None:
            return cache
        visible = min(self._idx, self.max_size)
        n_tokens = max(0, min(self.offset[idx].item(), self.max_size))
        if n_tokens == 0:
            return cache

        def extract_temporal(tuples):
            result = tuple(
                mx.contiguous(t[idx:idx + 1, :, :visible, :]) for t in tuples
            )
            if self._idx > self.max_size:
                write_pos = self._idx % self.max_size
                if write_pos > 0:
                    result = tuple(
                        mx.concatenate(
                            [t[..., write_pos:, :], t[..., :write_pos, :]], axis=2
                        )
                        for t in result
                    )
            padding = visible - n_tokens
            if padding > 0:
                result = tuple(mx.contiguous(t[..., padding:visible, :]) for t in result)
            return result

        cache._keys = extract_temporal(self._keys)
        cache._values = extract_temporal(self._values)
        cache.offset = n_tokens
        cache._idx = n_tokens
        # Pad to max_size so subsequent decode writes stay in bounds
        if n_tokens < self.max_size:
            pad_len = self.max_size - n_tokens

            def expand(tuples):
                result = []
                for t in tuples:
                    pad_spec = [(0, 0)] * (t.ndim - 2) + [(0, pad_len), (0, 0)]
                    result.append(mx.pad(t, pad_spec))
                return tuple(result)

            cache._keys = expand(cache._keys)
            cache._values = expand(cache._values)
        return cache

    @classmethod
    def merge(cls, caches):
        """Merge multiple QuantizedRotatingSDPACache instances into a batch."""
        if not caches:
            return cls([0], max_size=4096)
        max_size = caches[0].max_size
        bits = caches[0].bits
        gs = caches[0].group_size
        lengths = [c.size() for c in caches]
        max_len = max(lengths)
        padding = [max_len - l for l in lengths]

        batch = cls(padding, max_size=max_size, bits=bits, group_size=gs)
        if all(c.empty() for c in caches):
            return batch

        B = len(caches)
        ref = next(c for c in caches if not c.empty())
        H = ref._keys[0].shape[1]
        # Infer quantized dims from reference cache state
        k_state_ref, v_state_ref = ref.state
        k_dims = [t.shape[3] for t in k_state_ref]
        v_dims = [t.shape[3] for t in v_state_ref]
        dtype = k_state_ref[1].dtype

        batch._keys = (
            mx.zeros((B, H, max_len, k_dims[0]), dtype=mx.uint32),
            mx.zeros((B, H, max_len, k_dims[1]), dtype=dtype),
            mx.zeros((B, H, max_len, k_dims[2]), dtype=dtype),
        )
        batch._values = (
            mx.zeros((B, H, max_len, v_dims[0]), dtype=mx.uint32),
            mx.zeros((B, H, max_len, v_dims[1]), dtype=dtype),
            mx.zeros((B, H, max_len, v_dims[2]), dtype=dtype),
        )

        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.empty():
                continue
            k_state, v_state = c.state
            n = k_state[0].shape[2]
            for j in range(3):
                batch._keys[j][i:i + 1, :, p:p + n] = k_state[j]
                batch._values[j][i:i + 1, :, p:p + n] = v_state[j]

        batch.offset += max_len
        batch._idx = max_len
        return batch
