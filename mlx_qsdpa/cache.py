"""Quantized KV cache that stores keys/values in mx.quantize format.

Compatible with mlx-qsdpa's quantized_sdpa kernel. The consumer is
responsible for routing attention calls based on cache type.
"""

import mlx.core as mx


def _create_causal_mask(N, offset, window_size=None):
    """Create a causal attention mask."""
    q_indices = mx.arange(offset, offset + N)
    k_indices = mx.arange(offset + N)
    mask = q_indices[:, None] >= k_indices[None]
    if window_size is not None:
        window_mask = q_indices[:, None] - k_indices[None] < window_size
        mask = mask & window_mask
    return mask


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
