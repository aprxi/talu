extern "C" __global__ void talu_gated_delta_conv_values_f32(
    float* out,
    float* state,
    const float* values,
    const float* weight_time_major,
    const float* bias,
    unsigned int conv_dim,
    unsigned int d_conv,
    unsigned int has_bias,
    unsigned int ring_head
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    if (d_conv == 0 || ring_head >= d_conv) return;

    state[ring_head * conv_dim + ch] = values[ch];

    float acc = 0.0f;
    unsigned int state_row = ring_head + 1;
    if (state_row >= d_conv) state_row = 0;
    for (unsigned int k = 0; k < d_conv; ++k) {
        acc += state[state_row * conv_dim + ch] * weight_time_major[k * conv_dim + ch];
        state_row += 1;
        if (state_row >= d_conv) state_row = 0;
    }
    if (has_bias != 0u) {
        acc += bias[ch];
    }

    out[ch] = acc;
}

extern "C" __global__ void talu_gated_delta_conv_silu_values_f32(
    float* out,
    float* state,
    const float* values,
    const float* weight_time_major,
    const float* bias,
    unsigned int conv_dim,
    unsigned int d_conv,
    unsigned int has_bias,
    unsigned int ring_head
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    if (d_conv == 0 || ring_head >= d_conv) return;

    state[ring_head * conv_dim + ch] = values[ch];

    float acc = 0.0f;
    unsigned int state_row = ring_head + 1;
    if (state_row >= d_conv) state_row = 0;
    for (unsigned int k = 0; k < d_conv; ++k) {
        acc += state[state_row * conv_dim + ch] * weight_time_major[k * conv_dim + ch];
        state_row += 1;
        if (state_row >= d_conv) state_row = 0;
    }
    if (has_bias != 0u) {
        acc += bias[ch];
    }

    // Apply SiLU in the same launch to avoid a separate elementwise kernel.
    const float sigma = 1.0f / (1.0f + expf(-acc));
    out[ch] = acc * sigma;
}

extern "C" __global__ void talu_gated_delta_conv_silu_values_rows_f32(
    float* out,
    float* state,
    const float* values,
    const float* weight_time_major,
    const float* bias,
    unsigned int conv_dim,
    unsigned int d_conv,
    unsigned int has_bias,
    unsigned int ring_head,
    unsigned int rows,
    unsigned int row_stride
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;
    if (d_conv == 0 || ring_head >= d_conv || rows == 0 || row_stride < conv_dim) return;

    unsigned int ring = ring_head;
    for (unsigned int row = 0; row < rows; ++row) {
        const unsigned int row_base = row * row_stride;
        const float v = values[row_base + ch];
        state[ring * conv_dim + ch] = v;

        float acc = 0.0f;
        unsigned int state_row = ring + 1;
        if (state_row >= d_conv) state_row = 0;
        for (unsigned int k = 0; k < d_conv; ++k) {
            acc += state[state_row * conv_dim + ch] * weight_time_major[k * conv_dim + ch];
            state_row += 1;
            if (state_row >= d_conv) state_row = 0;
        }
        if (has_bias != 0u) {
            acc += bias[ch];
        }
        const float sigma = 1.0f / (1.0f + expf(-acc));
        out[row_base + ch] = acc * sigma;

        ring += 1;
        if (ring >= d_conv) ring = 0;
    }
}

extern "C" __global__ void talu_gated_delta_conv_silu_values_rows_ptrs_f32(
    float* out,
    const unsigned long long* state_ptrs,
    const unsigned int* positions,
    const float* values,
    const float* weight_time_major,
    const float* bias,
    unsigned int conv_dim,
    unsigned int d_conv,
    unsigned int has_bias,
    unsigned int rows,
    unsigned int row_stride
) {
    const unsigned int row = blockIdx.y;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || ch >= conv_dim) return;
    if (d_conv == 0 || row_stride < conv_dim) return;

    float* state = reinterpret_cast<float*>(state_ptrs[row]);
    if (state == nullptr) return;

    const unsigned int ring_head = positions[row];
    if (ring_head >= d_conv) return;

    const unsigned int row_base = row * row_stride;
    const float v = values[row_base + ch];
    state[ring_head * conv_dim + ch] = v;

    float acc = 0.0f;
    unsigned int state_row = ring_head + 1;
    if (state_row >= d_conv) state_row = 0;
    for (unsigned int k = 0; k < d_conv; ++k) {
        acc += state[state_row * conv_dim + ch] * weight_time_major[k * conv_dim + ch];
        state_row += 1;
        if (state_row >= d_conv) state_row = 0;
    }
    if (has_bias != 0u) {
        acc += bias[ch];
    }

    const float sigma = 1.0f / (1.0f + expf(-acc));
    out[row_base + ch] = acc * sigma;
}

extern "C" __global__ void talu_gated_delta_advance_ring_heads_f32(
    unsigned int* positions,
    unsigned int d_conv,
    unsigned int rows
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int next = positions[row] + 1;
    if (next >= d_conv) next = 0;
    positions[row] = next;
}
