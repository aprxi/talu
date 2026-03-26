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
    __shared__ float q_inv;
    __shared__ float k_inv;
    __shared__ float qk_dot_shared;
    __shared__ float beta;
    __shared__ float g_exp;

    for (unsigned int row = 0; row < rows; ++row) {
        const unsigned int row_base = row * row_stride;
        const unsigned int out_row_base = row * out_row_stride;
        const float* qkv = qkv_rows + row_base;
        const float* beta_raw = qkv_rows + row_base + beta_offset;
        const float* a_raw = qkv_rows + row_base + a_offset;

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
        k_norm_sum = talu_gd_warp_reduce_sum(k_norm_sum, mask);
        if (tid == 0) q_inv = rsqrtf(q_norm_sum + 1.0e-6f) * q_scale;

        if (tid == 0) {
            k_inv = rsqrtf(k_norm_sum + 1.0e-6f);
            beta = 1.0f / (1.0f + talu_fast_exp_scalar(-beta_raw[head_idx]));
            const float dt_term = has_dt_bias != 0u ? dt_bias[head_idx] : 0.0f;
            const float softplus_in = a_raw[head_idx] + dt_term;
            const float softplus = softplus_in > 20.0f ? softplus_in : logf(1.0f + talu_fast_exp_scalar(softplus_in));
            const float g = -talu_fast_exp_scalar(a_log[head_idx]) * softplus;
            g_exp = talu_fast_exp_scalar(g);
        }
        __syncwarp(mask);

        float qk_dot = 0.0f;
        for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
            const float q = qkv[query_base + idx] * q_inv;
            const float k = qkv[key_base + idx] * k_inv;
            query_norm[idx] = q;
            key_norm[idx] = k;
            qk_dot += q * k;
        }

        qk_dot = talu_gd_warp_reduce_sum(qk_dot, mask);
        if (tid == 0) qk_dot_shared = qk_dot;
        __syncwarp(mask);

        for (unsigned int out_idx = out_start + tid; out_idx < out_end; out_idx += blockDim.x) {
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

            out[out_row_base + head_idx * d_head + out_idx] = acc;
        }
    }
}

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

    const unsigned int tid = threadIdx.x;
    const unsigned int qk_repeat = n_v_heads / n_qk_heads;
    const unsigned int qk_head_idx = head_idx / qk_repeat;
    const unsigned int qk_inner = n_qk_heads * d_head;
    const unsigned int value_base = 2 * qk_inner + head_idx * d_head;
    const unsigned int query_base = qk_head_idx * d_head;
    const unsigned int key_base = qk_inner + qk_head_idx * d_head;
    const unsigned int state_base = head_idx * d_head * d_head;
    const float q_scale = rsqrtf((float)d_head);

    const unsigned int row_base = row * row_stride;
    const unsigned int out_row_base = row * out_row_stride;
    const float* qkv = qkv_rows + row_base;
    const float* beta_raw = qkv_rows + row_base + beta_offset;
    const float* a_raw = qkv_rows + row_base + a_offset;

    extern __shared__ float shmem[];
    float* query_norm = shmem;
    float* key_norm = shmem + d_head;
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

    const unsigned int mask = __activemask();
    q_norm_sum = talu_gd_warp_reduce_sum(q_norm_sum, mask);
    k_norm_sum = talu_gd_warp_reduce_sum(k_norm_sum, mask);
    if (tid == 0) q_inv = rsqrtf(q_norm_sum + 1.0e-6f) * q_scale;

    if (tid == 0) {
        k_inv = rsqrtf(k_norm_sum + 1.0e-6f);
        beta = 1.0f / (1.0f + talu_fast_exp_scalar(-beta_raw[head_idx]));
        const float dt_term = has_dt_bias != 0u ? dt_bias[head_idx] : 0.0f;
        const float softplus_in = a_raw[head_idx] + dt_term;
        const float softplus = softplus_in > 20.0f ? softplus_in : logf(1.0f + talu_fast_exp_scalar(softplus_in));
        const float g = -talu_fast_exp_scalar(a_log[head_idx]) * softplus;
        g_exp = talu_fast_exp_scalar(g);
    }
    __syncwarp(mask);

    float qk_dot = 0.0f;
    for (unsigned int idx = tid; idx < d_head; idx += blockDim.x) {
        const float q = qkv[query_base + idx] * q_inv;
        const float k = qkv[key_base + idx] * k_inv;
        query_norm[idx] = q;
        key_norm[idx] = k;
        qk_dot += q * k;
    }

    qk_dot = talu_gd_warp_reduce_sum(qk_dot, mask);
    if (tid == 0) qk_dot_shared = qk_dot;
    __syncwarp(mask);

    for (unsigned int out_idx = out_start + tid; out_idx < out_end; out_idx += blockDim.x) {
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

        out[out_row_base + head_idx * d_head + out_idx] = acc;
    }
}
