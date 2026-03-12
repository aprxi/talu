extern "C" __global__ void talu_gated_attention_compact_q_f32(
    float* compact_query,
    const float* packed_query,
    unsigned int seq_len,
    unsigned int query_dim,
    unsigned int query_projection_dim,
    unsigned int head_count,
    unsigned int head_dim
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = seq_len * query_dim;
    if (index >= total) return;

    const unsigned int row = index / query_dim;
    const unsigned int row_offset = index - row * query_dim;
    const unsigned int head = row_offset / head_dim;
    const unsigned int dim = row_offset - head * head_dim;
    if (head >= head_count) return;

    const unsigned long long src_base = (unsigned long long)row * query_projection_dim + (unsigned long long)head * head_dim * 2u;
    compact_query[index] = packed_query[src_base + dim];
}

extern "C" __global__ void talu_gated_attention_output_gate_f32(
    float* context,
    const float* packed_query,
    unsigned int seq_len,
    unsigned int query_dim,
    unsigned int query_projection_dim,
    unsigned int head_count,
    unsigned int head_dim
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = seq_len * query_dim;
    if (index >= total) return;

    const unsigned int row = index / query_dim;
    const unsigned int row_offset = index - row * query_dim;
    const unsigned int head = row_offset / head_dim;
    const unsigned int dim = row_offset - head * head_dim;
    if (head >= head_count) return;

    const unsigned long long gate_base = (unsigned long long)row * query_projection_dim + (unsigned long long)head * head_dim * 2u + head_dim;
    const float gate = packed_query[gate_base + dim];
    const float sig = 1.0f / (1.0f + expf(-gate));
    context[index] *= sig;
}
