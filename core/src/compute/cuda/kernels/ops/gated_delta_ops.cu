extern "C" __global__ void talu_gated_delta_ssm_f32(
    float* out,
    float* state,
    const float* qkv,
    const float* beta_raw,
    const float* a_raw,
    const float* a_log,
    const float* dt_bias,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias
) {
    const unsigned int head_idx = blockIdx.x;
    if (head_idx >= n_v_heads || d_head == 0 || n_qk_heads == 0 || (n_v_heads % n_qk_heads) != 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int qk_repeat = n_v_heads / n_qk_heads;
    const unsigned int qk_head_idx = head_idx / qk_repeat;
    const unsigned int qk_inner = n_qk_heads * d_head;
    const unsigned int value_base = 2 * qk_inner + head_idx * d_head;
    const unsigned int query_base = qk_head_idx * d_head;
    const unsigned int key_base = qk_inner + qk_head_idx * d_head;
    const unsigned int state_base = head_idx * d_head * d_head;
    const float q_scale = rsqrtf((float)d_head);

    extern __shared__ float shmem[];
    float* query_norm = shmem;
    float* key_norm = shmem + d_head;
    float* reduce = key_norm + d_head;
    __shared__ float q_inv;
    __shared__ float k_inv;
    __shared__ float qk_dot_shared;
    __shared__ float beta;
    __shared__ float g_exp;

    float q_norm_sum = 0.0f;
    float k_norm_sum = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx];
        const float k = qkv[key_base + idx];
        q_norm_sum += q * q;
        k_norm_sum += k * k;
    }

    reduce[tid] = q_norm_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) q_inv = rsqrtf(reduce[0] + 1.0e-6f) * q_scale;

    reduce[tid] = k_norm_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        k_inv = rsqrtf(reduce[0] + 1.0e-6f);
        beta = 1.0f / (1.0f + talu_fast_exp_scalar(-beta_raw[head_idx]));
        const float dt_term = has_dt_bias != 0u ? dt_bias[head_idx] : 0.0f;
        const float softplus_in = a_raw[head_idx] + dt_term;
        const float softplus = softplus_in > 20.0f ? softplus_in : logf(1.0f + talu_fast_exp_scalar(softplus_in));
        const float g = -talu_fast_exp_scalar(a_log[head_idx]) * softplus;
        g_exp = talu_fast_exp_scalar(g);
    }
    __syncthreads();

    float qk_dot = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx] * q_inv;
        const float k = qkv[key_base + idx] * k_inv;
        query_norm[idx] = q;
        key_norm[idx] = k;
        qk_dot += q * k;
    }

    reduce[tid] = qk_dot;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) qk_dot_shared = reduce[0];
    __syncthreads();

    for (unsigned int out_idx = tid; out_idx < d_head; out_idx += blockDim.x) {
        float kv = 0.0f;
        float acc = 0.0f;
        for (unsigned int k_idx = 0; k_idx < d_head; ++k_idx) {
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            const float scaled = state[state_idx] * g_exp;
            kv += scaled * key_norm[k_idx];
            acc += scaled * query_norm[k_idx];
        }

        kv = (qkv[value_base + out_idx] - kv) * beta;
        acc += kv * qk_dot_shared;

        for (unsigned int k_idx = 0; k_idx < d_head; ++k_idx) {
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            const float scaled = state[state_idx] * g_exp;
            state[state_idx] = scaled + kv * key_norm[k_idx];
        }

        out[head_idx * d_head + out_idx] = acc;
    }
}

static __device__ __forceinline__ float talu_gd_warp_reduce_sum(float value, unsigned int mask) {
#pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

// K-parallel SSM helper: processes one row of the gated delta SSM.
// Splits the k_idx loop across n_warps warps for higher occupancy.
// State tile (16KB for d_head=128) fits in L1 — loop 2 re-reads hit cache.
//
// Dynamic shared memory layout:
//   [query_norm: d_head floats]
//   [key_norm:   d_head floats]
static __device__ __forceinline__ void talu_gd_ssm_row_k_parallel(
    float* out,
    float* state,
    const float* qkv,
    const float* beta_raw,
    const float* a_raw,
    const float* a_log,
    const float* dt_bias,
    unsigned int head_idx,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int out_start,
    unsigned int out_end,
    unsigned int out_row_base,
    float* shmem,
    unsigned int k_per_warp
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    static constexpr unsigned int TALU_GD_SSM_MAX_WARPS = 4u;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);

    const unsigned int k_start = min(warp_id * k_per_warp, d_head);
    const unsigned int k_end = min(k_start + k_per_warp, d_head);
    const unsigned int my_k_count = k_end - k_start;
    const unsigned int out_idx = out_start + lane_id;

    const unsigned int qk_repeat = n_v_heads / n_qk_heads;
    const unsigned int qk_head_idx = head_idx / qk_repeat;
    const unsigned int qk_inner = n_qk_heads * d_head;
    const unsigned int value_base = 2 * qk_inner + head_idx * d_head;
    const unsigned int query_base = qk_head_idx * d_head;
    const unsigned int key_base = qk_inner + qk_head_idx * d_head;
    const unsigned int state_base = head_idx * d_head * d_head;
    const float q_scale = rsqrtf((float)d_head);

    float* query_norm = shmem;
    float* key_norm = shmem + d_head;

    __shared__ float warp_reduce[TALU_GD_SSM_MAX_WARPS];
    __shared__ float kv_partial[TALU_GD_SSM_MAX_WARPS][TALU_GD_SSM_OUT_TILE];
    __shared__ float acc_partial[TALU_GD_SSM_MAX_WARPS][TALU_GD_SSM_OUT_TILE];
    __shared__ float q_inv;
    __shared__ float k_inv;
    __shared__ float qk_dot_shared;
    __shared__ float beta_shared;
    __shared__ float g_exp_shared;

    // --- Q/K norm with cross-warp reduction ---
    float q_norm_sum = 0.0f;
    float k_norm_sum = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx];
        const float k = qkv[key_base + idx];
        q_norm_sum += q * q;
        k_norm_sum += k * k;
    }

    const unsigned int mask = __activemask();
    q_norm_sum = talu_gd_warp_reduce_sum(q_norm_sum, mask);
    if (lane_id == 0) warp_reduce[warp_id] = q_norm_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        q_inv = rsqrtf(total + 1.0e-6f) * q_scale;
    }
    __syncthreads();

    k_norm_sum = talu_gd_warp_reduce_sum(k_norm_sum, mask);
    if (lane_id == 0) warp_reduce[warp_id] = k_norm_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        k_inv = rsqrtf(total + 1.0e-6f);
        beta_shared = 1.0f / (1.0f + talu_fast_exp_scalar(-beta_raw[head_idx]));
        const float dt_term = has_dt_bias != 0u ? dt_bias[head_idx] : 0.0f;
        const float softplus_in = a_raw[head_idx] + dt_term;
        const float softplus = softplus_in > 20.0f ? softplus_in : logf(1.0f + talu_fast_exp_scalar(softplus_in));
        const float g = -talu_fast_exp_scalar(a_log[head_idx]) * softplus;
        g_exp_shared = talu_fast_exp_scalar(g);
    }
    __syncthreads();

    // --- QK dot product with cross-warp reduction ---
    float qk_dot = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx] * q_inv;
        const float k = qkv[key_base + idx] * k_inv;
        query_norm[idx] = q;
        key_norm[idx] = k;
        qk_dot += q * k;
    }

    qk_dot = talu_gd_warp_reduce_sum(qk_dot, mask);
    if (lane_id == 0) warp_reduce[warp_id] = qk_dot;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        qk_dot_shared = total;
    }
    __syncthreads();

    // --- Loop 1: partial kv/acc over this warp's k-range ---
    const float g_exp_local = g_exp_shared;
    float kv = 0.0f;
    float acc_val = 0.0f;
    if (out_idx < out_end) {
        for (unsigned int ki = 0; ki < my_k_count; ++ki) {
            const unsigned int k_idx = k_start + ki;
            const float scaled = state[state_base + k_idx * d_head + out_idx] * g_exp_local;
            kv += scaled * key_norm[k_idx];
            acc_val += scaled * query_norm[k_idx];
        }
    }

    // --- Cross-warp reduce kv/acc ---
    kv_partial[warp_id][lane_id] = kv;
    acc_partial[warp_id][lane_id] = acc_val;
    __syncthreads();

    if (warp_id == 0 && out_idx < out_end) {
        float kv_sum = kv_partial[0][lane_id];
        float acc_sum = acc_partial[0][lane_id];
        for (unsigned int w = 1; w < n_warps; ++w) {
            kv_sum += kv_partial[w][lane_id];
            acc_sum += acc_partial[w][lane_id];
        }
        kv_sum = (qkv[value_base + out_idx] - kv_sum) * beta_shared;
        acc_sum += kv_sum * qk_dot_shared;
        kv_partial[0][lane_id] = kv_sum;
        out[out_row_base + head_idx * d_head + out_idx] = acc_sum;
    }
    __syncthreads();

    // --- Loop 2: update state (re-read from L1, split across warps) ---
    if (out_idx < out_end) {
        kv = kv_partial[0][lane_id];
        for (unsigned int ki = 0; ki < my_k_count; ++ki) {
            const unsigned int k_idx = k_start + ki;
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            state[state_idx] = state[state_idx] * g_exp_local + kv * key_norm[k_idx];
        }
    }
}

extern "C" __global__ void talu_gated_delta_ssm_rows_f32(
    float* out,
    float* state,
    const float* qkv_rows,
    const float* a_log,
    const float* dt_bias,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int rows,
    unsigned int row_stride,
    unsigned int beta_offset,
    unsigned int a_offset,
    unsigned int out_row_stride
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);
    const unsigned int k_per_warp = (d_head + n_warps - 1u) / n_warps;
    const unsigned int tiles_per_head = (d_head + TALU_GD_SSM_OUT_TILE - 1u) / TALU_GD_SSM_OUT_TILE;
    if (tiles_per_head == 0u) return;
    const unsigned int head_tile_idx = blockIdx.x;
    const unsigned int head_idx = head_tile_idx / tiles_per_head;
    if (head_idx >= n_v_heads || d_head == 0 || n_qk_heads == 0 || (n_v_heads % n_qk_heads) != 0) return;
    if (rows == 0 || row_stride == 0 || out_row_stride == 0) return;
    const unsigned int tile_idx = head_tile_idx - head_idx * tiles_per_head;
    const unsigned int out_start = tile_idx * TALU_GD_SSM_OUT_TILE;
    if (out_start >= d_head) return;
    const unsigned int out_end = min(out_start + TALU_GD_SSM_OUT_TILE, d_head);

    extern __shared__ float shmem[];

    for (unsigned int row = 0; row < rows; ++row) {
        const unsigned int row_base = row * row_stride;
        const unsigned int out_row_base = row * out_row_stride;
        const float* qkv = qkv_rows + row_base;
        const float* beta_raw = qkv_rows + row_base + beta_offset;
        const float* a_raw = qkv_rows + row_base + a_offset;

        talu_gd_ssm_row_k_parallel(
            out, state, qkv, beta_raw, a_raw, a_log, dt_bias,
            head_idx, n_qk_heads, n_v_heads, d_head, has_dt_bias,
            out_start, out_end, out_row_base, shmem, k_per_warp);
        __syncthreads();
    }
}

// Batched decode SSM kernel. 4 tile-blocks per head (OUT_TILE=32, d_head=128).
// Near DRAM bandwidth saturation (~94% peak at n=2). A head-wide variant
// (1 block/head) was benchmarked and lost by 2-3x due to insufficient
// occupancy. The tile decomposition is the right trade-off for this problem.
extern "C" __global__ void talu_gated_delta_ssm_rows_ptrs_f32(
    float* out,
    const unsigned long long* state_ptrs,
    const float* qkv_rows,
    const float* a_log,
    const float* dt_bias,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int rows,
    unsigned int row_stride,
    unsigned int beta_offset,
    unsigned int a_offset,
    unsigned int out_row_stride
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;

    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);
    const unsigned int k_per_warp = (d_head + n_warps - 1u) / n_warps;
    const unsigned int tiles_per_head = (d_head + TALU_GD_SSM_OUT_TILE - 1u) / TALU_GD_SSM_OUT_TILE;
    if (tiles_per_head == 0u) return;
    const unsigned int head_tile_idx = blockIdx.x;
    const unsigned int head_idx = head_tile_idx / tiles_per_head;
    if (head_idx >= n_v_heads || d_head == 0 || n_qk_heads == 0 || (n_v_heads % n_qk_heads) != 0) return;
    if (row_stride == 0 || out_row_stride == 0) return;
    const unsigned int tile_idx = head_tile_idx - head_idx * tiles_per_head;
    const unsigned int out_start = tile_idx * TALU_GD_SSM_OUT_TILE;
    if (out_start >= d_head) return;
    const unsigned int out_end = min(out_start + TALU_GD_SSM_OUT_TILE, d_head);

    float* state = reinterpret_cast<float*>(state_ptrs[row]);
    if (state == nullptr) return;

    const unsigned int row_base = row * row_stride;
    const unsigned int out_row_base = row * out_row_stride;
    const float* qkv = qkv_rows + row_base;
    const float* beta_raw = qkv_rows + row_base + beta_offset;
    const float* a_raw = qkv_rows + row_base + a_offset;

    extern __shared__ float shmem[];

    talu_gd_ssm_row_k_parallel(
        out, state, qkv, beta_raw, a_raw, a_log, dt_bias,
        head_idx, n_qk_heads, n_v_heads, d_head, has_dt_bias,
        out_start, out_end, out_row_base, shmem, k_per_warp);
}

static __device__ __forceinline__ void talu_gd_ssm_row_k_parallel_i8(
    float* out,
    int8_t* state_i8,
    const float* qkv,
    const float* beta_raw,
    const float* a_raw,
    const float* a_log,
    const float* dt_bias,
    float* state_scales,
    unsigned int head_idx,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int out_start,
    unsigned int out_end,
    unsigned int out_row_base,
    float* shmem,
    unsigned int k_per_warp
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    static constexpr unsigned int TALU_GD_SSM_MAX_WARPS = 4u;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32u;
    const unsigned int lane_id = tid % 32u;
    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);

    const unsigned int k_start = min(warp_id * k_per_warp, d_head);
    const unsigned int k_end = min(k_start + k_per_warp, d_head);
    const unsigned int my_k_count = k_end - k_start;
    const unsigned int out_idx = out_start + lane_id;

    const unsigned int qk_repeat = n_v_heads / n_qk_heads;
    const unsigned int qk_head_idx = head_idx / qk_repeat;
    const unsigned int qk_inner = n_qk_heads * d_head;
    const unsigned int value_base = 2 * qk_inner + head_idx * d_head;
    const unsigned int query_base = qk_head_idx * d_head;
    const unsigned int key_base = qk_inner + qk_head_idx * d_head;
    const unsigned int state_base = head_idx * d_head * d_head;
    const unsigned int scale_base = head_idx * d_head;
    const float q_scale = rsqrtf((float)d_head);

    float* query_norm = shmem;
    float* key_norm = shmem + d_head;

    __shared__ float warp_reduce[TALU_GD_SSM_MAX_WARPS];
    __shared__ float kv_partial[TALU_GD_SSM_MAX_WARPS][TALU_GD_SSM_OUT_TILE];
    __shared__ float acc_partial[TALU_GD_SSM_MAX_WARPS][TALU_GD_SSM_OUT_TILE];
    __shared__ float max_partial[TALU_GD_SSM_MAX_WARPS][TALU_GD_SSM_OUT_TILE];
    __shared__ float scale_shared[TALU_GD_SSM_OUT_TILE];
    __shared__ float q_inv;
    __shared__ float k_inv;
    __shared__ float qk_dot_shared;
    __shared__ float beta_shared;
    __shared__ float g_exp_shared;

    float q_norm_sum = 0.0f;
    float k_norm_sum = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx];
        const float k = qkv[key_base + idx];
        q_norm_sum += q * q;
        k_norm_sum += k * k;
    }

    const unsigned int mask = __activemask();
    q_norm_sum = talu_gd_warp_reduce_sum(q_norm_sum, mask);
    if (lane_id == 0) warp_reduce[warp_id] = q_norm_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        q_inv = rsqrtf(total + 1.0e-6f) * q_scale;
    }
    __syncthreads();

    k_norm_sum = talu_gd_warp_reduce_sum(k_norm_sum, mask);
    if (lane_id == 0) warp_reduce[warp_id] = k_norm_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        k_inv = rsqrtf(total + 1.0e-6f);
        beta_shared = 1.0f / (1.0f + talu_fast_exp_scalar(-beta_raw[head_idx]));
        const float dt_term = has_dt_bias != 0u ? dt_bias[head_idx] : 0.0f;
        const float softplus_in = a_raw[head_idx] + dt_term;
        const float softplus = softplus_in > 20.0f ? softplus_in : logf(1.0f + talu_fast_exp_scalar(softplus_in));
        const float g = -talu_fast_exp_scalar(a_log[head_idx]) * softplus;
        g_exp_shared = talu_fast_exp_scalar(g);
    }
    __syncthreads();

    float qk_dot = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx] * q_inv;
        const float k = qkv[key_base + idx] * k_inv;
        query_norm[idx] = q;
        key_norm[idx] = k;
        qk_dot += q * k;
    }

    qk_dot = talu_gd_warp_reduce_sum(qk_dot, mask);
    if (lane_id == 0) warp_reduce[warp_id] = qk_dot;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (unsigned int w = 0; w < n_warps; ++w) total += warp_reduce[w];
        qk_dot_shared = total;
    }
    __syncthreads();

    const float g_exp_local = g_exp_shared;
    float kv = 0.0f;
    float acc_val = 0.0f;
    float max_abs = 0.0f;
    float state_scale = 0.0f;
    if (out_idx < out_end) {
        state_scale = state_scales[scale_base + out_idx];
        for (unsigned int ki = 0; ki < my_k_count; ++ki) {
            const unsigned int k_idx = k_start + ki;
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            const float state_val = (float)state_i8[state_idx] * state_scale;
            const float scaled = state_val * g_exp_local;
            kv += scaled * key_norm[k_idx];
            acc_val += scaled * query_norm[k_idx];
        }
    }

    kv_partial[warp_id][lane_id] = kv;
    acc_partial[warp_id][lane_id] = acc_val;
    __syncthreads();

    if (warp_id == 0 && out_idx < out_end) {
        float kv_sum = kv_partial[0][lane_id];
        float acc_sum = acc_partial[0][lane_id];
        for (unsigned int w = 1; w < n_warps; ++w) {
            kv_sum += kv_partial[w][lane_id];
            acc_sum += acc_partial[w][lane_id];
        }
        kv_sum = (qkv[value_base + out_idx] - kv_sum) * beta_shared;
        acc_sum += kv_sum * qk_dot_shared;
        kv_partial[0][lane_id] = kv_sum;
        out[out_row_base + head_idx * d_head + out_idx] = acc_sum;
    }
    __syncthreads();

    kv = kv_partial[0][lane_id];
    if (out_idx < out_end) {
        for (unsigned int ki = 0; ki < my_k_count; ++ki) {
            const unsigned int k_idx = k_start + ki;
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            const float state_val = (float)state_i8[state_idx] * state_scale;
            const float updated = state_val * g_exp_local + kv * key_norm[k_idx];
            max_abs = fmaxf(max_abs, fabsf(updated));
        }
    }

    max_partial[warp_id][lane_id] = max_abs;
    __syncthreads();

    if (warp_id == 0 && out_idx < out_end) {
        float max_abs_all = max_partial[0][lane_id];
        for (unsigned int w = 1; w < n_warps; ++w) {
            max_abs_all = fmaxf(max_abs_all, max_partial[w][lane_id]);
        }
        const float new_scale = max_abs_all > 0.0f ? (max_abs_all * (1.0f / 127.0f)) : 0.0f;
        scale_shared[lane_id] = new_scale;
        state_scales[scale_base + out_idx] = new_scale;
    }
    __syncthreads();

    if (out_idx < out_end) {
        const float new_scale = scale_shared[lane_id];
        for (unsigned int ki = 0; ki < my_k_count; ++ki) {
            const unsigned int k_idx = k_start + ki;
            const unsigned int state_idx = state_base + k_idx * d_head + out_idx;
            const float state_val = (float)state_i8[state_idx] * state_scale;
            const float updated = state_val * g_exp_local + kv * key_norm[k_idx];
            int q = 0;
            if (new_scale > 0.0f) {
                q = __float2int_rn(updated / new_scale);
                q = max(-127, min(127, q));
            }
            state_i8[state_idx] = (int8_t)q;
        }
    }
}

extern "C" __global__ void talu_gated_delta_ssm_rows_i8_f32(
    float* out,
    int8_t* state,
    const float* qkv_rows,
    const float* a_log,
    const float* dt_bias,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int rows,
    unsigned int row_stride,
    unsigned int beta_offset,
    unsigned int a_offset,
    unsigned int out_row_stride,
    unsigned int state_scales_offset
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    if ((state_scales_offset & 0x3u) != 0u) return;
    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);
    const unsigned int k_per_warp = (d_head + n_warps - 1u) / n_warps;
    const unsigned int tiles_per_head = (d_head + TALU_GD_SSM_OUT_TILE - 1u) / TALU_GD_SSM_OUT_TILE;
    if (tiles_per_head == 0u) return;
    const unsigned int head_tile_idx = blockIdx.x;
    const unsigned int head_idx = head_tile_idx / tiles_per_head;
    if (head_idx >= n_v_heads || d_head == 0 || n_qk_heads == 0 || (n_v_heads % n_qk_heads) != 0) return;
    if (rows == 0 || row_stride == 0 || out_row_stride == 0) return;
    const unsigned int tile_idx = head_tile_idx - head_idx * tiles_per_head;
    const unsigned int out_start = tile_idx * TALU_GD_SSM_OUT_TILE;
    if (out_start >= d_head) return;
    const unsigned int out_end = min(out_start + TALU_GD_SSM_OUT_TILE, d_head);

    float* state_scales = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(state) + state_scales_offset);

    extern __shared__ float shmem[];

    for (unsigned int row = 0; row < rows; ++row) {
        const unsigned int row_base = row * row_stride;
        const unsigned int out_row_base = row * out_row_stride;
        const float* qkv = qkv_rows + row_base;
        const float* beta_raw = qkv_rows + row_base + beta_offset;
        const float* a_raw = qkv_rows + row_base + a_offset;

        talu_gd_ssm_row_k_parallel_i8(
            out,
            state,
            qkv,
            beta_raw,
            a_raw,
            a_log,
            dt_bias,
            state_scales,
            head_idx,
            n_qk_heads,
            n_v_heads,
            d_head,
            has_dt_bias,
            out_start,
            out_end,
            out_row_base,
            shmem,
            k_per_warp);
        __syncthreads();
    }
}

extern "C" __global__ void talu_gated_delta_ssm_rows_ptrs_i8_f32(
    float* out,
    const unsigned long long* state_ptrs,
    const float* qkv_rows,
    const float* a_log,
    const float* dt_bias,
    unsigned int n_qk_heads,
    unsigned int n_v_heads,
    unsigned int d_head,
    unsigned int has_dt_bias,
    unsigned int rows,
    unsigned int row_stride,
    unsigned int beta_offset,
    unsigned int a_offset,
    unsigned int out_row_stride,
    unsigned int state_scales_offset
) {
    static constexpr unsigned int TALU_GD_SSM_OUT_TILE = 32u;
    const unsigned int row = blockIdx.y;
    if (row >= rows) return;
    if ((state_scales_offset & 0x3u) != 0u) return;

    const unsigned int n_warps = max(1u, (blockDim.x + 31u) / 32u);
    const unsigned int k_per_warp = (d_head + n_warps - 1u) / n_warps;
    const unsigned int tiles_per_head = (d_head + TALU_GD_SSM_OUT_TILE - 1u) / TALU_GD_SSM_OUT_TILE;
    if (tiles_per_head == 0u) return;
    const unsigned int head_tile_idx = blockIdx.x;
    const unsigned int head_idx = head_tile_idx / tiles_per_head;
    if (head_idx >= n_v_heads || d_head == 0 || n_qk_heads == 0 || (n_v_heads % n_qk_heads) != 0) return;
    if (row_stride == 0 || out_row_stride == 0) return;
    const unsigned int tile_idx = head_tile_idx - head_idx * tiles_per_head;
    const unsigned int out_start = tile_idx * TALU_GD_SSM_OUT_TILE;
    if (out_start >= d_head) return;
    const unsigned int out_end = min(out_start + TALU_GD_SSM_OUT_TILE, d_head);

    int8_t* state = reinterpret_cast<int8_t*>(state_ptrs[row]);
    if (state == nullptr) return;
    float* state_scales = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(state) + state_scales_offset);

    const unsigned int row_base = row * row_stride;
    const unsigned int out_row_base = row * out_row_stride;
    const float* qkv = qkv_rows + row_base;
    const float* beta_raw = qkv_rows + row_base + beta_offset;
    const float* a_raw = qkv_rows + row_base + a_offset;

    extern __shared__ float shmem[];

    talu_gd_ssm_row_k_parallel_i8(
        out,
        state,
        qkv,
        beta_raw,
        a_raw,
        a_log,
        dt_bias,
        state_scales,
        head_idx,
        n_qk_heads,
        n_v_heads,
        d_head,
        has_dt_bias,
        out_start,
        out_end,
        out_row_base,
        shmem,
        k_per_warp);
}
