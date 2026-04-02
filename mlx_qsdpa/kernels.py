"""Metal kernel source and kernel objects for quantized SDPA."""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Header: empty (dequant is inlined in kernel body for maximum optimization)
# ---------------------------------------------------------------------------
# V2: All dequant logic is inlined. This enables:
# - Hoisted scale/bias loads (1 per key per thread, not 1 per element)
# - Pre-loaded uint32 (1-2 reads per key per thread)
# - Half-precision dequant+FMA (2x throughput on M2 GPU)
# - Compiler can fully optimize the unrolled inner loop
HEADER = """
// No helper functions needed. Dequant is fully inlined in kernel body.
"""

# ---------------------------------------------------------------------------
# Single-pass vector kernel body
# ---------------------------------------------------------------------------
# Template parameters (set at dispatch time):
#   T          - float16 / bfloat16
#   D          - head dimension (128, 192, 256)
#   bits       - quantisation bits (4 or 8)
#   group_size - elements per quant group (32, 64, 128)
#   has_bias   - whether bias buffers are non-zero
#
# Inputs  (see input_names below):
#   queries      - (B*H_q, D) T
#   packed_keys  - (B*H_kv, N, pack_D) uint32
#   packed_vals  - (B*H_kv, N, pack_D) uint32
#   k_scales     - (B*H_kv, N, num_groups) T
#   v_scales     - (B*H_kv, N, num_groups) T
#   k_biases     - (B*H_kv, N, num_groups) T
#   v_biases     - (B*H_kv, N, num_groups) T
#   gqa_factor   - int scalar (H_q / H_kv)
#   seq_len      - int scalar (N, number of key/value tokens)
#   alpha        - float scalar (scale = 1/sqrt(D))
#
# Output:
#   out          - (B*H_q, D) T

QSDPA_VECTOR_SOURCE = """
    // ----- compile-time constants -----
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int elems_per_int = 32 / bits;
    constexpr int pack_D = D / elems_per_int;
    constexpr int num_groups = D / group_size;
    constexpr uint bit_mask = (1u << bits) - 1u;
    constexpr int packs_per_thread = (qk_per_thread + elems_per_int - 1) / elems_per_int;

    // ----- thread indices -----
    uint tg_idx   = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int q_batch_head = int(tg_idx);
    int kv_head      = q_batch_head / gqa_factor;
    int q_base  = q_batch_head * D;
    int kv_base = kv_head * seq_len;

    // ----- query in half registers (pre-scaled) -----
    int q_elem_start = int(simd_lid) * qk_per_thread;
    T q_h[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
        q_h[j] = T(float(queries[q_base + q_elem_start + j]) * alpha);
    }

    // ----- output accumulators (float for stability) -----
    float o_reg[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) o_reg[j] = 0.0f;

    float max_score = -1e38f;
    float sum_exp   = 0.0f;

    // ----- pre-computed per-thread constants -----
    int pack_start = q_elem_start / elems_per_int;
    int grp = q_elem_start / group_size;

    // ----- threadgroup scratch -----
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    // ===== main loop: stride through keys =====
    for (int i = int(simd_gid); i < seq_len; i += BN) {
        int rp = (kv_base + i) * pack_D;
        int rg = (kv_base + i) * num_groups;

        // -- K: hoisted scale/bias + pre-loaded uint32s --
        T sk = k_scales[rg + grp];
        T bk = has_bias ? k_biases[rg + grp] : T(0);

        uint32_t pk[packs_per_thread];
        for (int p = 0; p < packs_per_thread; p++)
            pk[p] = packed_keys[rp + pack_start + p];

        // -- dot(q, k) in half precision, accumulate in float --
        float score = 0.0f;
        for (int j = 0; j < qk_per_thread; j++) {
            int ge = q_elem_start + j;
            int lp = ge / elems_per_int - pack_start;
            int bp = (ge % elems_per_int) * bits;
            T kv = sk * T((pk[lp] >> bp) & bit_mask) + bk;
            score += float(q_h[j] * kv);
        }
        score = simd_sum(score);

        // -- online softmax --
        float new_max   = max(max_score, score);
        float factor    = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        // -- V: hoisted scale/bias + pre-loaded uint32s --
        T sv = v_scales[rg + grp];
        T bv = has_bias ? v_biases[rg + grp] : T(0);

        uint32_t pv[packs_per_thread];
        for (int p = 0; p < packs_per_thread; p++)
            pv[p] = packed_vals[rp + pack_start + p];

        for (int j = 0; j < qk_per_thread; j++) {
            int ge = q_elem_start + j;
            int lp = ge / elems_per_int - pack_start;
            int bp = (ge % elems_per_int) * bits;
            T vv = sv * T((pv[lp] >> bp) & bit_mask) + bv;
            o_reg[j] = o_reg[j] * factor + exp_score * float(vv);
        }
    }

    // ===== cross-simdgroup reduction (unchanged) =====
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lane_max = tg_max[simd_lid];
    float global_max = simd_max(lane_max);
    float factor = fast::exp(lane_max - global_max);
    float global_sum = simd_sum(tg_sum[simd_lid] * factor);
    float my_factor = fast::exp(max_score - global_max);

    for (int j = 0; j < qk_per_thread; j++) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j] * my_factor;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float val = tg_outputs[simd_gid * BD + simd_lid];
        o_reg[j] = simd_sum(val);
        o_reg[j] = (global_sum == 0.0f) ? o_reg[j] : (o_reg[j] / global_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        int out_base = q_batch_head * D + int(simd_gid) * qk_per_thread;
        for (int j = 0; j < qk_per_thread; j++)
            out[out_base + j] = T(o_reg[j]);
    }
"""

# ---------------------------------------------------------------------------
# Input / output names for the vector kernel
# ---------------------------------------------------------------------------
_VECTOR_INPUT_NAMES = [
    "queries",
    "packed_keys",
    "packed_vals",
    "k_scales",
    "v_scales",
    "k_biases",
    "v_biases",
    "gqa_factor",
    "seq_len",
    "alpha",
]

_VECTOR_OUTPUT_NAMES = ["out"]


def _make_vector_kernel():
    """Build the mx.fast.metal_kernel object (compiled once, reused)."""
    return mx.fast.metal_kernel(
        name="qsdpa_vector",
        input_names=_VECTOR_INPUT_NAMES,
        output_names=_VECTOR_OUTPUT_NAMES,
        source=QSDPA_VECTOR_SOURCE,
        header=HEADER,
        ensure_row_contiguous=True,
    )


# ---------------------------------------------------------------------------
# Two-pass kernel: pass 1 (per-block partial attention)
# ---------------------------------------------------------------------------
# Each threadgroup handles one (kv_head, batch, block) combo, with
# gqa_factor simdgroups inside -- one per query head mapping to that KV head.
# Block `block_idx` processes keys at positions block_idx, block_idx+num_blocks,
# block_idx+2*num_blocks, ... accumulating online softmax partials.
#
# Template parameters: T, D, bits, group_size, has_bias (same as vector)
#
# Inputs:
#   queries      - (B*H_q, D) T
#   packed_keys  - (B*H_kv, N, pack_D) uint32
#   packed_vals  - (B*H_kv, N, pack_D) uint32
#   k_scales     - (B*H_kv, N, num_groups) T
#   v_scales     - (B*H_kv, N, num_groups) T
#   k_biases     - (B*H_kv, N, num_groups) T
#   v_biases     - (B*H_kv, N, num_groups) T
#   gqa_factor   - int scalar
#   seq_len      - int scalar
#   alpha        - float scalar
#   num_q_heads  - int scalar (H_q)
#   num_kv_heads - int scalar (H_kv)
#   num_blocks   - int scalar
#
# Outputs:
#   partials     - (B*H_q * num_blocks, D) T
#   sums         - (B*H_q * num_blocks,) float32
#   maxs         - (B*H_q * num_blocks,) float32

QSDPA_2PASS_1_SOURCE = """
    // ----- compile-time constants -----
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int elems_per_int = 32 / bits;
    constexpr int pack_D = D / elems_per_int;
    constexpr int num_groups = D / group_size;
    constexpr uint bit_mask = (1u << bits) - 1u;
    constexpr int packs_per_thread = (qk_per_thread + elems_per_int - 1) / elems_per_int;

    // ----- thread indices -----
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    uint kv_head_idx = threadgroup_position_in_grid.x;
    uint batch_idx   = threadgroup_position_in_grid.y;
    uint block_idx   = threadgroup_position_in_grid.z;

    uint q_head_idx  = kv_head_idx * gqa_factor + simd_gid;
    uint q_bh        = batch_idx * num_q_heads + q_head_idx;

    int q_base  = int(q_bh) * D;
    int kv_head = int(batch_idx) * num_kv_heads + int(kv_head_idx);
    int kv_base = kv_head * seq_len;

    // ----- query in half registers (pre-scaled) -----
    int q_elem_start = int(simd_lid) * qk_per_thread;
    T q_h[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
        q_h[j] = T(float(queries[q_base + q_elem_start + j]) * alpha);
    }

    float o_reg[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) o_reg[j] = 0.0f;

    float max_score = -1e38f;
    float sum_exp   = 0.0f;

    int pack_start = q_elem_start / elems_per_int;
    int grp = q_elem_start / group_size;

    // ===== block-stride through keys =====
    for (int i = int(block_idx); i < seq_len; i += num_blocks) {
        int rp = (kv_base + i) * pack_D;
        int rg = (kv_base + i) * num_groups;

        T sk = k_scales[rg + grp];
        T bk = has_bias ? k_biases[rg + grp] : T(0);

        uint32_t pk[packs_per_thread];
        for (int p = 0; p < packs_per_thread; p++)
            pk[p] = packed_keys[rp + pack_start + p];

        float score = 0.0f;
        for (int j = 0; j < qk_per_thread; j++) {
            int ge = q_elem_start + j;
            int lp = ge / elems_per_int - pack_start;
            int bp = (ge % elems_per_int) * bits;
            T kv = sk * T((pk[lp] >> bp) & bit_mask) + bk;
            score += float(q_h[j] * kv);
        }
        score = simd_sum(score);

        float new_max   = max(max_score, score);
        float factor    = fast::exp(max_score - new_max);
        float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        T sv = v_scales[rg + grp];
        T bv = has_bias ? v_biases[rg + grp] : T(0);

        uint32_t pv_arr[packs_per_thread];
        for (int p = 0; p < packs_per_thread; p++)
            pv_arr[p] = packed_vals[rp + pack_start + p];

        for (int j = 0; j < qk_per_thread; j++) {
            int ge = q_elem_start + j;
            int lp = ge / elems_per_int - pack_start;
            int bp = (ge % elems_per_int) * bits;
            T vv = sv * T((pv_arr[lp] >> bp) & bit_mask) + bv;
            o_reg[j] = o_reg[j] * factor + exp_score * float(vv);
        }
    }

    // ===== write per-block partials =====
    int out_idx = int(q_bh) * num_blocks + int(block_idx);

    if (simd_lid == 0) {
        sums[out_idx] = sum_exp;
        maxs[out_idx] = max_score;
    }

    for (int j = 0; j < qk_per_thread; j++) {
        partials[out_idx * D + q_elem_start + j] = T(o_reg[j]);
    }
"""

# ---------------------------------------------------------------------------
# Two-pass kernel: pass 2 (reduce blocks)
# ---------------------------------------------------------------------------
# Merges per-block partials into final output using logsumexp-weighted
# averaging. Does NOT use DEQUANT macro. Follows MLX sdpa_vector_2pass_2.
#
# Grid: (B*H_q * 1024, 1, 1) with threadgroup (1024, 1, 1)
# Each threadgroup = 32 simdgroups, handles one q_bh.
# Simdgroups distribute across blocks.
#
# Template parameters: T, D
#
# Inputs:
#   partials     - (B*H_q * num_blocks, D) T
#   sums         - (B*H_q * num_blocks,) float32
#   maxs         - (B*H_q * num_blocks,) float32
#   num_blocks   - int scalar
#
# Output:
#   out          - (B*H_q, D) T

QSDPA_2PASS_2_SOURCE = """
    constexpr int BN = 32;                // simdgroups per threadgroup
    constexpr int BD = 32;                // threads per simdgroup
    constexpr int qk_per_thread = D / BD;

    uint tg_idx   = threadgroup_position_in_grid.x;    // which q_bh
    uint simd_gid = simdgroup_index_in_threadgroup;     // 0..31
    uint simd_lid = thread_index_in_simdgroup;          // 0..31

    int q_bh = int(tg_idx);
    int base = q_bh * num_blocks;  // start of this q_bh's blocks

    // ----- threadgroup scratch -----
    threadgroup float tg_outputs[BN * BD];
    threadgroup float tg_max[BN];
    threadgroup float tg_sum[BN];

    // ----- accumulate across blocks assigned to this simdgroup -----
    float o_reg[qk_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
        o_reg[j] = 0.0f;
    }
    float max_score = -1e38f;
    float sum_exp   = 0.0f;

    int elem_start = int(simd_lid) * qk_per_thread;

    for (int b = int(simd_gid); b < num_blocks; b += BN) {
        int idx = base + b;
        float b_sum = sums[idx];
        float b_max = maxs[idx];

        // online merge
        float new_max   = max(max_score, b_max);
        float factor    = fast::exp(max_score - new_max);
        float b_factor  = fast::exp(b_max - new_max);
        max_score = new_max;
        sum_exp   = sum_exp * factor + b_sum * b_factor;

        // accumulate weighted partials
        for (int j = 0; j < qk_per_thread; j++) {
            float p = float(partials[idx * D + elem_start + j]);
            o_reg[j] = o_reg[j] * factor + p * b_factor;
        }
    }

    // ===== cross-simdgroup reduction =====
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lane_max = tg_max[simd_lid];
    float global_max = simd_max(lane_max);
    float factor = fast::exp(lane_max - global_max);
    float global_sum = simd_sum(tg_sum[simd_lid] * factor);

    float my_factor = fast::exp(max_score - global_max);

    for (int j = 0; j < qk_per_thread; j++) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j] * my_factor;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float val = tg_outputs[simd_gid * BD + simd_lid];
        o_reg[j] = simd_sum(val);
        o_reg[j] = (global_sum == 0.0f) ? o_reg[j] : (o_reg[j] / global_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ===== write output =====
    if (simd_lid == 0) {
        int out_base = q_bh * D + int(simd_gid) * qk_per_thread;
        for (int j = 0; j < qk_per_thread; j++) {
            out[out_base + j] = T(o_reg[j]);
        }
    }
"""

# ---------------------------------------------------------------------------
# Input / output names for two-pass kernels
# ---------------------------------------------------------------------------
_2PASS_1_INPUT_NAMES = [
    "queries",
    "packed_keys",
    "packed_vals",
    "k_scales",
    "v_scales",
    "k_biases",
    "v_biases",
    "gqa_factor",
    "seq_len",
    "alpha",
    "num_q_heads",
    "num_kv_heads",
    "num_blocks",
]

_2PASS_1_OUTPUT_NAMES = ["partials", "sums", "maxs"]

_2PASS_2_INPUT_NAMES = [
    "partials",
    "sums",
    "maxs",
    "num_blocks",
]

_2PASS_2_OUTPUT_NAMES = ["out"]


def _make_2pass_1_kernel():
    """Build pass-1 kernel (per-block partial attention)."""
    return mx.fast.metal_kernel(
        name="qsdpa_2pass_1",
        input_names=_2PASS_1_INPUT_NAMES,
        output_names=_2PASS_1_OUTPUT_NAMES,
        source=QSDPA_2PASS_1_SOURCE,
        header=HEADER,
        ensure_row_contiguous=True,
    )


def _make_2pass_2_kernel():
    """Build pass-2 kernel (reduce blocks)."""
    return mx.fast.metal_kernel(
        name="qsdpa_2pass_2",
        input_names=_2PASS_2_INPUT_NAMES,
        output_names=_2PASS_2_OUTPUT_NAMES,
        source=QSDPA_2PASS_2_SOURCE,
        header="",  # No DEQUANT needed for pass 2
        ensure_row_contiguous=True,
    )


# Module-level singleton -- created once on first import
vector_kernel = _make_vector_kernel()
pass1_kernel = _make_2pass_1_kernel()
pass2_kernel = _make_2pass_2_kernel()
