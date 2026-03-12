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
