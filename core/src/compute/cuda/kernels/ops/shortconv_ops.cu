extern "C" __global__ void talu_shortconv_step_f32(
    float* out,
    float* state,
    const float* b_gate,
    const float* c_gate,
    const float* x_proj,
    const float* weight_time_major,
    const float* bias,
    unsigned int conv_dim,
    unsigned int d_conv,
    unsigned int has_bias
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= conv_dim) return;

    if (d_conv > 1) {
        for (unsigned int k = 0; k + 1 < d_conv; ++k) {
            state[k * conv_dim + ch] = state[(k + 1) * conv_dim + ch];
        }
    }

    const float newest = b_gate[ch] * x_proj[ch];
    state[(d_conv - 1) * conv_dim + ch] = newest;

    float acc = 0.0f;
    for (unsigned int k = 0; k < d_conv; ++k) {
        acc += state[k * conv_dim + ch] * weight_time_major[k * conv_dim + ch];
    }
    if (has_bias != 0u) {
        acc += bias[ch];
    }

    out[ch] = c_gate[ch] * acc;
}
