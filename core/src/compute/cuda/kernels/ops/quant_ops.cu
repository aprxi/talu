static constexpr unsigned int TALU_QUANT_WARP_SIZE = 32;

static __device__ __forceinline__ float talu_quant_warp_sum_f32(float value) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, 16);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 8);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 4);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 2);
    value += __shfl_down_sync(0xFFFFFFFFu, value, 1);
    return value;
}

static constexpr unsigned int U8_BLOCK_THREADS = 128;
static constexpr unsigned int U8_ELEMS_PER_THREAD = 16;
static constexpr unsigned int U8_ELEMS_PER_STEP = U8_BLOCK_THREADS * U8_ELEMS_PER_THREAD;

static __device__ __forceinline__ float talu_block_reduce_sum_128(float val, float* smem) {
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    val = talu_quant_warp_sum_f32(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();
    if (threadIdx.x < 4) {
        val = smem[threadIdx.x];
        val += __shfl_down_sync(0xFu, val, 2);
        val += __shfl_down_sync(0xFu, val, 1);
    }
    return val;
}

static __device__ __forceinline__ float talu_gaffine_u8_partial_dot_block(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int words_per_row = in_dim / 4;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)out_idx * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)out_idx * groups_per_row;

    float acc = 0.0f;
    for (unsigned int i = tid * U8_ELEMS_PER_THREAD; i < in_dim; i += U8_ELEMS_PER_STEP) {
        uint4 w4;
        const unsigned int* waddr = weight_row + (i >> 2);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w4.x), "=r"(w4.y), "=r"(w4.z), "=r"(w4.w)
            : "l"(waddr));

        const float4 in0 = *reinterpret_cast<const float4*>(&input[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input[i + 12]);

        const unsigned int group_idx = i / group_size;
        const float scale = talu_decode_scale_bias_u16(scale_row[group_idx], scales_dtype_tag);
        const float bias = talu_decode_scale_bias_u16(bias_row[group_idx], scales_dtype_tag);

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int q32 = (j == 0) ? w4.x : (j == 1) ? w4.y : (j == 2) ? w4.z : w4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;
            acc = fmaf(inv.x, static_cast<float>(q32 & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.y, static_cast<float>((q32 >> 8) & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.z, static_cast<float>((q32 >> 16) & 0xFFu) * scale + bias, acc);
            acc = fmaf(inv.w, static_cast<float>(q32 >> 24) * scale + bias, acc);
        }
    }
    return acc;
}

extern "C" __global__ void talu_embedding_lookup_gaffine_u4_f32(
    float* out,
    const unsigned int* packed_vals,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int vocab_size,
    unsigned int hidden_dim,
    unsigned int token,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    float multiplier
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_dim) return;
    if (token >= vocab_size) return;
    if (group_size == 0 || (hidden_dim % group_size) != 0 || (group_size % 8) != 0) return;

    const unsigned int groups_per_row = hidden_dim / group_size;
    const unsigned int words_per_group = group_size / 8;
    const unsigned int words_per_row = hidden_dim / 8;
    const unsigned int group_idx = index / group_size;
    const unsigned int in_group = index - group_idx * group_size;
    const unsigned int word_in_group = in_group / 8;
    const unsigned int nibble_in_word = in_group & 7;

    const unsigned int row_word_base = token * words_per_row;
    const unsigned int packed_word = packed_vals[row_word_base + group_idx * words_per_group + word_in_group];
    const unsigned int quant = (packed_word >> (nibble_in_word * 4)) & 0xF;
    const unsigned int sb_index = token * groups_per_row + group_idx;
    const float scale = talu_decode_scale_bias_u16(scales[sb_index], scales_dtype_tag);
    const float bias = talu_decode_scale_bias_u16(biases[sb_index], scales_dtype_tag);
    out[index] = (static_cast<float>(quant) * scale + bias) * multiplier;
}

// Optimized U4 GEMV: warp-per-row design with 128-bit streaming weight loads.
// Each warp (32 threads) processes one output row. Each thread loads 4 packed
// words (32 U4 nibbles) per iteration via ld.global.cs.v4.u32.
// Factored accumulation: accumulate raw nibble*input products, apply scale/bias
// per word to halve the dependency chain depth vs naive fmaf(in, fmaf(n,s,b), acc).
// Grid: (ceil(out_dim/4), batch_rows), Block: (128 = 4 warps)
static constexpr unsigned int U4_WARPS_PER_BLOCK = 4;
static constexpr unsigned int U4_WARP_ELEMS_PER_THREAD = 32;
static constexpr unsigned int U4_WARP_STEP = TALU_QUANT_WARP_SIZE * U4_WARP_ELEMS_PER_THREAD; // 1024

// Process one word (8 U4 nibbles) against 8 F32 input values using factored
// accumulation: accumulate raw nibble*input products, then apply scale/bias
// once per word.  Uses 2-way ILP accumulators (r0, r1).
#define U4_PROCESS_WORD(word, in_lo, in_hi, s, b, acc) do { \
    float r0 = 0.0f, r1 = 0.0f; \
    r0 = fmaf((in_lo).x, static_cast<float>(((word)      ) & 0xFu), r0); \
    r1 = fmaf((in_lo).y, static_cast<float>(((word) >>  4) & 0xFu), r1); \
    r0 = fmaf((in_lo).z, static_cast<float>(((word) >>  8) & 0xFu), r0); \
    r1 = fmaf((in_lo).w, static_cast<float>(((word) >> 12) & 0xFu), r1); \
    r0 = fmaf((in_hi).x, static_cast<float>(((word) >> 16) & 0xFu), r0); \
    r1 = fmaf((in_hi).y, static_cast<float>(((word) >> 20) & 0xFu), r1); \
    r0 = fmaf((in_hi).z, static_cast<float>(((word) >> 24) & 0xFu), r0); \
    r1 = fmaf((in_hi).w, static_cast<float>(((word) >> 28) & 0xFu), r1); \
    const float isum = ((in_lo).x + (in_lo).y) + ((in_lo).z + (in_lo).w) \
                     + ((in_hi).x + (in_hi).y) + ((in_hi).z + (in_hi).w); \
    (acc) = fmaf(r0 + r1, (s), fmaf(isum, (b), (acc))); \
} while (0)

static __device__ __forceinline__ float talu_gaffine_u4_dot_row_warp_v(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    unsigned int in_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int lane
) {
    const unsigned int words_per_row = in_dim >> 3;
    const unsigned int words_per_group = group_size >> 3;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)out_idx * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)out_idx * groups_per_row;

    // 64-bit weight loads: 2 words per iteration → 5 iterations per row for
    // in_dim=2560 (matching I8's DRAM request rate).  Each warp issues more
    // DRAM requests than with 128-bit loads, improving memory pipeline utilization.
    // Hardware coalesces 32 consecutive 64-bit loads into full cache line fills.
    float acc = 0.0f;
    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        unsigned int w0, w1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(w0), "=r"(w1)
            : "l"(&weight_row[w]));

        const unsigned int base_elem = w << 3;
        const unsigned int g0 = w / words_per_group;
        const float s0 = talu_decode_scale_bias_u16(scale_row[g0], scales_dtype_tag);
        const float b0 = talu_decode_scale_bias_u16(bias_row[g0], scales_dtype_tag);
        const float4 i0 = *reinterpret_cast<const float4*>(&input[base_elem]);
        const float4 i1 = *reinterpret_cast<const float4*>(&input[base_elem + 4]);
        U4_PROCESS_WORD(w0, i0, i1, s0, b0, acc);

        const unsigned int g1 = (w + 1) / words_per_group;
        const float s1 = talu_decode_scale_bias_u16(scale_row[g1], scales_dtype_tag);
        const float b1 = talu_decode_scale_bias_u16(bias_row[g1], scales_dtype_tag);
        const float4 i2 = *reinterpret_cast<const float4*>(&input[base_elem + 8]);
        const float4 i3 = *reinterpret_cast<const float4*>(&input[base_elem + 12]);
        U4_PROCESS_WORD(w1, i2, i3, s1, b1, acc);
    }
    return talu_quant_warp_sum_f32(acc);
}

extern "C" __global__ __launch_bounds__(128) void talu_gaffine_u4_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) return;

    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    const float acc = talu_gaffine_u4_dot_row_warp_v(
        input_row, packed_weight, scales, biases,
        in_dim, out_idx, group_size, scales_dtype_tag, lane);

    if (lane == 0) out_row[out_idx] = acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    const unsigned int out_idx = blockIdx.x;
    if (batch >= batch_rows) return;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 ||
        (in_dim % 16) != 0 || (group_size % 16) != 0) {
        return;
    }
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    __shared__ float smem[4];
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, packed_weight, scales, biases,
            in_dim, out_idx, group_size, scales_dtype_tag),
        smem);

    if (threadIdx.x == 0) out_row[out_idx] = acc;
}

// Symmetric I8 GEMV: warp-per-row design with 4 output rows per block.
// Each warp (32 threads) computes one output row's dot product independently.
// No __syncthreads needed — only warp-level shuffles for reduction.
// For in_dim=2560: 32 threads × 16 elems = 512 per step → 5 exact steps (no tail waste).
// Grid: (ceil(out_dim/4), batch_rows), Block: (128 = 4 warps)
static constexpr unsigned int I8_WARPS_PER_BLOCK = 4;
static constexpr unsigned int I8_WARP_ELEMS_PER_THREAD = 16;
static constexpr unsigned int I8_WARP_STEP = TALU_QUANT_WARP_SIZE * I8_WARP_ELEMS_PER_THREAD; // 512

extern "C" __global__ __launch_bounds__(128) void talu_i8_matvec_f32(
    const float* input,
    const signed char* weight,
    const float* weight_scales,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * I8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;
    const signed char* weight_row = weight + (unsigned long long)out_idx * in_dim;

    float acc = 0.0f;
    for (unsigned int i = lane * I8_WARP_ELEMS_PER_THREAD; i < in_dim; i += I8_WARP_STEP) {
        uint4 w4;
        const unsigned int* waddr = reinterpret_cast<const unsigned int*>(weight_row + i);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w4.x), "=r"(w4.y), "=r"(w4.z), "=r"(w4.w)
            : "l"(waddr));

        const float4 in0 = *reinterpret_cast<const float4*>(&input[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input[i + 12]);

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int q32 = (j == 0) ? w4.x : (j == 1) ? w4.y : (j == 2) ? w4.z : w4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;
            acc = fmaf(inv.x, static_cast<float>(static_cast<signed char>(q32 & 0xFFu)), acc);
            acc = fmaf(inv.y, static_cast<float>(static_cast<signed char>((q32 >> 8) & 0xFFu)), acc);
            acc = fmaf(inv.z, static_cast<float>(static_cast<signed char>((q32 >> 16) & 0xFFu)), acc);
            acc = fmaf(inv.w, static_cast<float>(static_cast<signed char>(q32 >> 24)), acc);
        }
    }

    // Warp-level reduction (no shared memory, no __syncthreads).
    acc = talu_quant_warp_sum_f32(acc);
    if (lane == 0) out_row[out_idx] = acc * weight_scales[out_idx];
}

// Fused I8 QKV matvec: single kernel launch for Q, K, V projections.
// Warp-per-row design: 4 warps per block, each computes one output row from
// the combined Q+K+V output space using pre-cached I8 weights.
// Grid: (ceil(total_out/4), batch_rows), Block: 128
extern "C" __global__ __launch_bounds__(128) void talu_i8_matvec_qkv_f32(
    const float* input,
    const signed char* q_weight,
    const float* q_scales,
    float* q_out,
    unsigned int q_out_dim,
    const signed char* k_weight,
    const float* k_scales,
    float* k_out,
    unsigned int k_out_dim,
    const signed char* v_weight,
    const float* v_scales,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * I8_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_idx >= total_dim) return;

    const float* input_row = input + (unsigned long long)batch * in_dim;

    // Select the right projection's weight/scale/output based on combined index.
    const signed char* weight_row;
    const float* scales;
    float* out_row;
    unsigned int row_idx;
    if (out_idx < q_out_dim) {
        weight_row = q_weight + (unsigned long long)out_idx * in_dim;
        scales = q_scales;
        out_row = q_out + (unsigned long long)batch * q_out_dim;
        row_idx = out_idx;
    } else if (out_idx < qk_dim) {
        row_idx = out_idx - q_out_dim;
        weight_row = k_weight + (unsigned long long)row_idx * in_dim;
        scales = k_scales;
        out_row = k_out + (unsigned long long)batch * k_out_dim;
    } else {
        row_idx = out_idx - qk_dim;
        weight_row = v_weight + (unsigned long long)row_idx * in_dim;
        scales = v_scales;
        out_row = v_out + (unsigned long long)batch * v_out_dim;
    }

    float acc = 0.0f;
    for (unsigned int i = lane * I8_WARP_ELEMS_PER_THREAD; i < in_dim; i += I8_WARP_STEP) {
        uint4 w4;
        const unsigned int* waddr = reinterpret_cast<const unsigned int*>(weight_row + i);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(w4.x), "=r"(w4.y), "=r"(w4.z), "=r"(w4.w)
            : "l"(waddr));

        const float4 in0 = *reinterpret_cast<const float4*>(&input_row[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input_row[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input_row[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input_row[i + 12]);

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int q32 = (j == 0) ? w4.x : (j == 1) ? w4.y : (j == 2) ? w4.z : w4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;
            acc = fmaf(inv.x, static_cast<float>(static_cast<signed char>(q32 & 0xFFu)), acc);
            acc = fmaf(inv.y, static_cast<float>(static_cast<signed char>((q32 >> 8) & 0xFFu)), acc);
            acc = fmaf(inv.z, static_cast<float>(static_cast<signed char>((q32 >> 16) & 0xFFu)), acc);
            acc = fmaf(inv.w, static_cast<float>(static_cast<signed char>(q32 >> 24)), acc);
        }
    }

    acc = talu_quant_warp_sum_f32(acc);
    if (lane == 0) out_row[row_idx] = acc * scales[row_idx];
}

// Fused I8 gate/up + SiLU matvec: single kernel computes gate and up dot products,
// then applies SiLU fusion: out[j] = gate[j] * sigmoid(gate[j]) * up[j].
// Interleaved design: each warp reads both gate and up weights in the same loop
// iteration, sharing the input load. No shared memory or __syncthreads needed.
// Grid: (ceil(out_dim/4), batch_rows), Block: 128
extern "C" __global__ __launch_bounds__(128) void talu_i8_matvec_gate_up_silu_f32(
    const float* input,
    const signed char* gate_weight,
    const float* gate_scales,
    const signed char* up_weight,
    const float* up_scales,
    float* out,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * I8_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;

    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;
    const signed char* gate_row = gate_weight + (unsigned long long)out_idx * in_dim;
    const signed char* up_row = up_weight + (unsigned long long)out_idx * in_dim;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int i = lane * I8_WARP_ELEMS_PER_THREAD; i < in_dim; i += I8_WARP_STEP) {
        // Load input (shared between gate and up).
        const float4 in0 = *reinterpret_cast<const float4*>(&input_row[i]);
        const float4 in1 = *reinterpret_cast<const float4*>(&input_row[i + 4]);
        const float4 in2 = *reinterpret_cast<const float4*>(&input_row[i + 8]);
        const float4 in3 = *reinterpret_cast<const float4*>(&input_row[i + 12]);

        // Load gate weights.
        uint4 gw4;
        const unsigned int* gaddr = reinterpret_cast<const unsigned int*>(gate_row + i);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(gw4.x), "=r"(gw4.y), "=r"(gw4.z), "=r"(gw4.w)
            : "l"(gaddr));

        // Load up weights.
        uint4 uw4;
        const unsigned int* uaddr = reinterpret_cast<const unsigned int*>(up_row + i);
        asm volatile(
            "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(uw4.x), "=r"(uw4.y), "=r"(uw4.z), "=r"(uw4.w)
            : "l"(uaddr));

        #pragma unroll
        for (unsigned int j = 0; j < 4; ++j) {
            const unsigned int gq = (j == 0) ? gw4.x : (j == 1) ? gw4.y : (j == 2) ? gw4.z : gw4.w;
            const unsigned int uq = (j == 0) ? uw4.x : (j == 1) ? uw4.y : (j == 2) ? uw4.z : uw4.w;
            const float4 inv = (j == 0) ? in0 : (j == 1) ? in1 : (j == 2) ? in2 : in3;

            gate_acc = fmaf(inv.x, static_cast<float>(static_cast<signed char>(gq & 0xFFu)), gate_acc);
            gate_acc = fmaf(inv.y, static_cast<float>(static_cast<signed char>((gq >> 8) & 0xFFu)), gate_acc);
            gate_acc = fmaf(inv.z, static_cast<float>(static_cast<signed char>((gq >> 16) & 0xFFu)), gate_acc);
            gate_acc = fmaf(inv.w, static_cast<float>(static_cast<signed char>(gq >> 24)), gate_acc);

            up_acc = fmaf(inv.x, static_cast<float>(static_cast<signed char>(uq & 0xFFu)), up_acc);
            up_acc = fmaf(inv.y, static_cast<float>(static_cast<signed char>((uq >> 8) & 0xFFu)), up_acc);
            up_acc = fmaf(inv.z, static_cast<float>(static_cast<signed char>((uq >> 16) & 0xFFu)), up_acc);
            up_acc = fmaf(inv.w, static_cast<float>(static_cast<signed char>(uq >> 24)), up_acc);
        }
    }

    gate_acc = talu_quant_warp_sum_f32(gate_acc);
    up_acc = talu_quant_warp_sum_f32(up_acc);

    if (lane == 0) {
        const float g = gate_acc * gate_scales[out_idx];
        const float u = up_acc * up_scales[out_idx];
        const float sigma = 1.0f / (1.0f + expf(-g));
        out_row[out_idx] = g * sigma * u;
    }
}

extern "C" __global__ __launch_bounds__(128) void talu_gaffine_u4_matvec_qkv_f32(
    const float* input,
    const unsigned int* q_packed_weight,
    const unsigned short* q_scales,
    const unsigned short* q_biases,
    float* q_out,
    unsigned int q_out_dim,
    unsigned int q_group_size,
    unsigned int q_scales_dtype_tag,
    const unsigned int* k_packed_weight,
    const unsigned short* k_scales,
    const unsigned short* k_biases,
    float* k_out,
    unsigned int k_out_dim,
    unsigned int k_group_size,
    unsigned int k_scales_dtype_tag,
    const unsigned int* v_packed_weight,
    const unsigned short* v_scales,
    const unsigned short* v_biases,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int v_group_size,
    unsigned int v_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 32) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* q_out_row = q_out + (unsigned long long)batch * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)batch * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)batch * v_out_dim;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 8) != 0) return;
        const float acc = talu_gaffine_u4_dot_row_warp_v(
            input_row, q_packed_weight, q_scales, q_biases,
            in_dim, out_index, q_group_size, q_scales_dtype_tag, lane);
        if (lane == 0) q_out_row[out_index] = acc;
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 8) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        const float acc = talu_gaffine_u4_dot_row_warp_v(
            input_row, k_packed_weight, k_scales, k_biases,
            in_dim, k_row, k_group_size, k_scales_dtype_tag, lane);
        if (lane == 0) k_out_row[k_row] = acc;
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 8) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    const float acc = talu_gaffine_u4_dot_row_warp_v(
        input_row, v_packed_weight, v_scales, v_biases,
        in_dim, v_row, v_group_size, v_scales_dtype_tag, lane);
    if (lane == 0) v_out_row[v_row] = acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_qkv_f32(
    const float* input,
    const unsigned int* q_packed_weight,
    const unsigned short* q_scales,
    const unsigned short* q_biases,
    float* q_out,
    unsigned int q_out_dim,
    unsigned int q_group_size,
    unsigned int q_scales_dtype_tag,
    const unsigned int* k_packed_weight,
    const unsigned short* k_scales,
    const unsigned short* k_biases,
    float* k_out,
    unsigned int k_out_dim,
    unsigned int k_group_size,
    unsigned int k_scales_dtype_tag,
    const unsigned int* v_packed_weight,
    const unsigned short* v_scales,
    const unsigned short* v_biases,
    float* v_out,
    unsigned int v_out_dim,
    unsigned int v_group_size,
    unsigned int v_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* q_out_row = q_out + (unsigned long long)batch * q_out_dim;
    float* k_out_row = k_out + (unsigned long long)batch * k_out_dim;
    float* v_out_row = v_out + (unsigned long long)batch * v_out_dim;

    const unsigned int out_index = blockIdx.x;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    __shared__ float smem[4];

    if (out_index < q_out_dim) {
        if (q_group_size == 0 || (in_dim % q_group_size) != 0 || (q_group_size % 16) != 0) return;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, q_packed_weight, q_scales, q_biases,
                in_dim, out_index, q_group_size, q_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) q_out_row[out_index] = acc;
        return;
    }

    if (out_index < qk_dim) {
        if (k_group_size == 0 || (in_dim % k_group_size) != 0 || (k_group_size % 16) != 0) return;
        const unsigned int k_row = out_index - q_out_dim;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, k_packed_weight, k_scales, k_biases,
                in_dim, k_row, k_group_size, k_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) k_out_row[k_row] = acc;
        return;
    }

    if (v_group_size == 0 || (in_dim % v_group_size) != 0 || (v_group_size % 16) != 0) return;
    const unsigned int v_row = out_index - qk_dim;
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, v_packed_weight, v_scales, v_biases,
            in_dim, v_row, v_group_size, v_scales_dtype_tag),
        smem);
    if (threadIdx.x == 0) v_out_row[v_row] = acc;
}

extern "C" __global__ __launch_bounds__(128) void talu_gaffine_u4_matvec_gate_up_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    float* gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 32) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* gate_out_row = gate_out + (unsigned long long)batch * gate_out_dim;
    float* up_out_row = up_out + (unsigned long long)batch * up_out_dim;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
        const float acc = talu_gaffine_u4_dot_row_warp_v(
            input_row, gate_packed_weight, gate_scales, gate_biases,
            in_dim, out_index, gate_group_size, gate_scales_dtype_tag, lane);
        if (lane == 0) gate_out_row[out_index] = acc;
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    const float acc = talu_gaffine_u4_dot_row_warp_v(
        input_row, up_packed_weight, up_scales, up_biases,
        in_dim, up_row, up_group_size, up_scales_dtype_tag, lane);
    if (lane == 0) up_out_row[up_row] = acc;
}

// Fused U4 gate/up + SiLU matvec: interleaved design where each warp reads
// both gate and up weights in the same loop iteration, sharing the input load.
// Grid: (ceil(out_dim/4), batch_rows), Block: (128 = 4 warps)
extern "C" __global__ __launch_bounds__(128) void talu_gaffine_u4_matvec_gate_up_silu_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* out,
    unsigned int out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 32) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_index >= out_dim) return;
    if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;

    const unsigned int words_per_row = in_dim >> 3;
    const unsigned int gate_wpg = gate_group_size >> 3;
    const unsigned int up_wpg = up_group_size >> 3;
    const unsigned int gate_gpr = in_dim / gate_group_size;
    const unsigned int up_gpr = in_dim / up_group_size;
    const unsigned int* g_row = gate_packed_weight + (unsigned long long)out_index * words_per_row;
    const unsigned int* u_row = up_packed_weight + (unsigned long long)out_index * words_per_row;
    const unsigned short* gs_row = gate_scales + (unsigned long long)out_index * gate_gpr;
    const unsigned short* gb_row = gate_biases + (unsigned long long)out_index * gate_gpr;
    const unsigned short* us_row = up_scales + (unsigned long long)out_index * up_gpr;
    const unsigned short* ub_row = up_biases + (unsigned long long)out_index * up_gpr;

    // 64-bit weight loads for both gate and up: 2 words per load → 5 iterations
    // per row for in_dim=2560.  4 DRAM loads per iteration (2 gate + 2 up words
    // via two v2.u32 loads) doubles the DRAM request rate vs 128-bit loads.
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        unsigned int gw0, gw1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(gw0), "=r"(gw1)
            : "l"(&g_row[w]));

        unsigned int uw0, uw1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(uw0), "=r"(uw1)
            : "l"(&u_row[w]));

        const unsigned int base_elem = w << 3;

        // Word 0: shared input, sequential gate then up.
        {
            const float4 in_lo = *reinterpret_cast<const float4*>(&input_row[base_elem]);
            const float4 in_hi = *reinterpret_cast<const float4*>(&input_row[base_elem + 4]);
            const unsigned int gg = w / gate_wpg;
            const float gs = talu_decode_scale_bias_u16(gs_row[gg], gate_scales_dtype_tag);
            const float gb = talu_decode_scale_bias_u16(gb_row[gg], gate_scales_dtype_tag);
            U4_PROCESS_WORD(gw0, in_lo, in_hi, gs, gb, gate_acc);
            const unsigned int ug = w / up_wpg;
            const float us = talu_decode_scale_bias_u16(us_row[ug], up_scales_dtype_tag);
            const float ub = talu_decode_scale_bias_u16(ub_row[ug], up_scales_dtype_tag);
            U4_PROCESS_WORD(uw0, in_lo, in_hi, us, ub, up_acc);
        }
        // Word 1: shared input, sequential gate then up.
        {
            const float4 in_lo = *reinterpret_cast<const float4*>(&input_row[base_elem + 8]);
            const float4 in_hi = *reinterpret_cast<const float4*>(&input_row[base_elem + 12]);
            const unsigned int gg = (w + 1) / gate_wpg;
            const float gs = talu_decode_scale_bias_u16(gs_row[gg], gate_scales_dtype_tag);
            const float gb = talu_decode_scale_bias_u16(gb_row[gg], gate_scales_dtype_tag);
            U4_PROCESS_WORD(gw1, in_lo, in_hi, gs, gb, gate_acc);
            const unsigned int ug = (w + 1) / up_wpg;
            const float us = talu_decode_scale_bias_u16(us_row[ug], up_scales_dtype_tag);
            const float ub = talu_decode_scale_bias_u16(ub_row[ug], up_scales_dtype_tag);
            U4_PROCESS_WORD(uw1, in_lo, in_hi, us, ub, up_acc);
        }
    }

    gate_acc = talu_quant_warp_sum_f32(gate_acc);
    up_acc = talu_quant_warp_sum_f32(up_acc);

    if (lane == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_acc));
        out_row[out_index] = gate_acc * sigma * up_acc;
    }
}

extern "C" __global__ void talu_gaffine_u8_matvec_gate_up_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    float* gate_out,
    unsigned int gate_out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* up_out,
    unsigned int up_out_dim,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* gate_out_row = gate_out + (unsigned long long)batch * gate_out_dim;
    float* up_out_row = up_out + (unsigned long long)batch * up_out_dim;

    const unsigned int out_index = blockIdx.x;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    __shared__ float smem[4];

    if (out_index < gate_out_dim) {
        if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 16) != 0) return;
        const float acc = talu_block_reduce_sum_128(
            talu_gaffine_u8_partial_dot_block(
                input_row, gate_packed_weight, gate_scales, gate_biases,
                in_dim, out_index, gate_group_size, gate_scales_dtype_tag),
            smem);
        if (threadIdx.x == 0) gate_out_row[out_index] = acc;
        return;
    }

    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 16) != 0) return;
    const unsigned int up_row = out_index - gate_out_dim;
    const float acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, up_packed_weight, up_scales, up_biases,
            in_dim, up_row, up_group_size, up_scales_dtype_tag),
        smem);
    if (threadIdx.x == 0) up_out_row[up_row] = acc;
}

extern "C" __global__ void talu_gaffine_u8_matvec_gate_up_silu_f32(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* out,
    unsigned int out_dim,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int batch_rows
) {
    const unsigned int batch = blockIdx.y;
    if (batch >= batch_rows) return;
    if (in_dim == 0 || (in_dim % 16) != 0) return;
    const float* input_row = input + (unsigned long long)batch * in_dim;
    float* out_row = out + (unsigned long long)batch * out_dim;

    const unsigned int out_index = blockIdx.x;
    if (out_index >= out_dim) return;
    if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 16) != 0) return;
    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 16) != 0) return;

    __shared__ float smem[4];

    const float gate_acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, gate_packed_weight, gate_scales, gate_biases,
            in_dim, out_index, gate_group_size, gate_scales_dtype_tag),
        smem);

    float gate_result = 0.0f;
    if (threadIdx.x == 0) gate_result = gate_acc;

    const float up_acc = talu_block_reduce_sum_128(
        talu_gaffine_u8_partial_dot_block(
            input_row, up_packed_weight, up_scales, up_biases,
            in_dim, out_index, up_group_size, up_scales_dtype_tag),
        smem);

    if (threadIdx.x == 0) {
        const float sigma = 1.0f / (1.0f + expf(-gate_result));
        out_row[out_index] = gate_result * sigma * up_acc;
    }
}

extern "C" __global__ void talu_gaffine_u8_dequantize_to_f16(
    const unsigned int* __restrict__ packed_weight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    unsigned short* __restrict__ out_f16,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int words_per_row = in_dim >> 2;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)row * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)row * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)row * groups_per_row;
    unsigned short* out_row = out_f16 + (unsigned long long)row * in_dim;

    // Process 16 U8 values per thread via 128-bit loads and stores.
    for (unsigned int i = threadIdx.x * 16; i < in_dim; i += blockDim.x * 16) {
        // 128-bit streaming load: 4 uint32 words = 16 packed U8 values.
        uint4 p;
        asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(p.x), "=r"(p.y), "=r"(p.z), "=r"(p.w)
            : "l"(&weight_row[i >> 2]));

        float4 store0, store1;
        unsigned int g;
        float s, b;

        // Elements i..i+3 (word 0)
        g = i / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store0)[0] = __floats2half2_rn(
            static_cast<float>(p.x & 0xFFu) * s + b,
            static_cast<float>((p.x >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[1] = __floats2half2_rn(
            static_cast<float>((p.x >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.x >> 24) * s + b);

        // Elements i+4..i+7 (word 1)
        g = (i + 4) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store0)[2] = __floats2half2_rn(
            static_cast<float>(p.y & 0xFFu) * s + b,
            static_cast<float>((p.y >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[3] = __floats2half2_rn(
            static_cast<float>((p.y >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.y >> 24) * s + b);

        // 128-bit store for first 8 F16 values.
        *reinterpret_cast<float4*>(&out_row[i]) = store0;

        // Elements i+8..i+11 (word 2)
        g = (i + 8) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store1)[0] = __floats2half2_rn(
            static_cast<float>(p.z & 0xFFu) * s + b,
            static_cast<float>((p.z >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[1] = __floats2half2_rn(
            static_cast<float>((p.z >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.z >> 24) * s + b);

        // Elements i+12..i+15 (word 3)
        g = (i + 12) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store1)[2] = __floats2half2_rn(
            static_cast<float>(p.w & 0xFFu) * s + b,
            static_cast<float>((p.w >> 8) & 0xFFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[3] = __floats2half2_rn(
            static_cast<float>((p.w >> 16) & 0xFFu) * s + b,
            static_cast<float>(p.w >> 24) * s + b);

        // 128-bit store for next 8 F16 values.
        *reinterpret_cast<float4*>(&out_row[i + 8]) = store1;
    }
}

extern "C" __global__ void talu_gaffine_u4_dequantize_to_f16(
    const unsigned int* __restrict__ packed_weight,
    const unsigned short* __restrict__ scales,
    const unsigned short* __restrict__ biases,
    unsigned short* __restrict__ out_f16,
    unsigned int out_dim,
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    if (row >= out_dim) return;

    const unsigned int words_per_row = in_dim / 8;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)row * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)row * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)row * groups_per_row;
    unsigned short* out_row = out_f16 + (unsigned long long)row * in_dim;

    // Process 16 U4 values per thread per iteration (2 packed words, 8 nibbles each).
    for (unsigned int i = threadIdx.x * 16; i < in_dim; i += blockDim.x * 16) {
        const unsigned int w0 = weight_row[i / 8];
        const unsigned int w1 = weight_row[i / 8 + 1];

        float4 store0, store1;
        unsigned int g;
        float s, b;

        // Elements i..i+7 (word 0: 8 nibbles).
        g = i / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store0)[0] = __floats2half2_rn(
            static_cast<float>(w0 & 0xFu) * s + b,
            static_cast<float>((w0 >> 4) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[1] = __floats2half2_rn(
            static_cast<float>((w0 >> 8) & 0xFu) * s + b,
            static_cast<float>((w0 >> 12) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[2] = __floats2half2_rn(
            static_cast<float>((w0 >> 16) & 0xFu) * s + b,
            static_cast<float>((w0 >> 20) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store0)[3] = __floats2half2_rn(
            static_cast<float>((w0 >> 24) & 0xFu) * s + b,
            static_cast<float>((w0 >> 28) & 0xFu) * s + b);

        *reinterpret_cast<float4*>(&out_row[i]) = store0;

        // Elements i+8..i+15 (word 1: 8 nibbles).
        g = (i + 8) / group_size;
        s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        reinterpret_cast<__half2*>(&store1)[0] = __floats2half2_rn(
            static_cast<float>(w1 & 0xFu) * s + b,
            static_cast<float>((w1 >> 4) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[1] = __floats2half2_rn(
            static_cast<float>((w1 >> 8) & 0xFu) * s + b,
            static_cast<float>((w1 >> 12) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[2] = __floats2half2_rn(
            static_cast<float>((w1 >> 16) & 0xFu) * s + b,
            static_cast<float>((w1 >> 20) & 0xFu) * s + b);
        reinterpret_cast<__half2*>(&store1)[3] = __floats2half2_rn(
            static_cast<float>((w1 >> 24) & 0xFu) * s + b,
            static_cast<float>((w1 >> 28) & 0xFu) * s + b);

        *reinterpret_cast<float4*>(&out_row[i + 8]) = store1;
    }
}

// ─── INT8 GEMM support for grouped-affine U8 prefill ───

// Quantize F32 input to I8 with per-row absmax scaling.
// Also computes per-group input sums (F32) for bias correction.
// Launch: grid=(rows), block=(256)
// out_i8:          [rows × in_dim] signed int8
// out_row_scales:  [rows] float — scale per row (absmax / 127)
// out_group_sums:  [rows × num_groups] float — per-group sums of the original F32 input
extern "C" __global__ void talu_quantize_f32_to_i8(
    const float* __restrict__ input,
    signed char* __restrict__ out_i8,
    float* __restrict__ out_row_scales,
    float* __restrict__ out_group_sums,
    unsigned int in_dim,
    unsigned int group_size
) {
    const unsigned int row = blockIdx.x;
    const float* row_in = input + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;
    const unsigned int num_groups = in_dim / group_size;

    // Phase 1: find per-row absmax via block reduction.
    __shared__ float smem[8]; // 256/32 = 8 warps
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_in[i]));
    }
    // Warp reduce.
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Phase 2: quantize to I8 and compute per-group sums.
    // Each thread handles a strided subset of elements.
    // Use shared memory for group sum accumulation.
    extern __shared__ float group_smem[]; // num_groups floats per warp → too complex
    // Simpler: each thread accumulates its own group sums, then warp/block reduce.
    // But num_groups can be up to ~80. Use a different approach:
    // Two-pass or combined pass.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = row_in[i];
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }

    // Per-group sums: each thread accumulates sums for each group.
    // With group_size typically 128 and blockDim=256, 2 threads per group element.
    float* row_gsums = out_group_sums + (unsigned long long)row * num_groups;
    for (unsigned int g = 0; g < num_groups; g++) {
        float gsum = 0.0f;
        const unsigned int base = g * group_size;
        for (unsigned int j = threadIdx.x; j < group_size; j += blockDim.x) {
            gsum += row_in[base + j];
        }
        // Block reduce this group sum.
        gsum = talu_block_reduce_sum_128(gsum, smem);
        if (threadIdx.x == 0) row_gsums[g] = gsum;
        __syncthreads();
    }
}

// Dequantize I32 GEMM output to F32 with grouped-affine correction.
// gemm_i32[m][n] = sum_k(input_i8[m][k] * weight_i8[n][k])
//   where weight_i8 = weight_u8 - 128 (XOR 0x80).
//
// Reconstruct: y[m][n] = sum_g(scale[n][g] * group_dot_u8x[m][n][g]) + bias_term[m][n]
//
// Since we only have the TOTAL dot product (not per-group), we use:
//   y[m][n] ≈ mean_scale[n] * (gemm_i32[m][n] + 128 * i8_rowsum[m]) * row_scale[m]
//            + bias_term[m][n]
//
// Where bias_term[m][n] = sum_g(bias[n][g] * group_input_sum[m][g])
//
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_dequant_i32_gaffine(
    const int* __restrict__ gemm_i32,       // [rows × out_dim]
    const float* __restrict__ row_scales,    // [rows]
    const int* __restrict__ i8_rowsums,      // [rows]
    const float* __restrict__ group_sums,    // [rows × num_groups]
    const unsigned short* __restrict__ scales, // [out_dim × num_groups]
    const unsigned short* __restrict__ biases, // [out_dim × num_groups]
    float* __restrict__ output,              // [rows × out_dim]
    unsigned int rows,
    unsigned int out_dim,
    unsigned int num_groups,
    unsigned int scales_dtype_tag
) {
    const unsigned int n = blockIdx.x; // output dimension index
    if (n >= out_dim) return;
    const unsigned short* scale_row = scales + (unsigned long long)n * num_groups;
    const unsigned short* bias_row = biases + (unsigned long long)n * num_groups;

    // Compute mean scale for this output row.
    float mean_scale = 0.0f;
    for (unsigned int g = 0; g < num_groups; g++) {
        mean_scale += talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
    }
    mean_scale /= static_cast<float>(num_groups);

    for (unsigned int m = threadIdx.x; m < rows; m += blockDim.x) {
        const float rs = row_scales[m];
        const int raw = gemm_i32[(unsigned long long)m * out_dim + n];
        const int i8sum = i8_rowsums[m];

        // Main term: mean_scale * (raw + 128 * i8_rowsum) * input_scale
        float result = mean_scale * static_cast<float>(raw + 128 * i8sum) * rs;

        // Bias correction: sum_g(bias[n][g] * group_input_sum[m][g])
        const float* m_gsums = group_sums + (unsigned long long)m * num_groups;
        for (unsigned int g = 0; g < num_groups; g++) {
            result += talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag) * m_gsums[g];
        }

        output[(unsigned long long)m * out_dim + n] = result;
    }
}

// Compute per-row I8 sums (for U8→I8 offset correction).
// Launch: grid=(rows), block=(256)
extern "C" __global__ void talu_i8_rowsum(
    const signed char* __restrict__ input_i8,
    int* __restrict__ out_rowsums,
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const signed char* row_in = input_i8 + (unsigned long long)row * in_dim;
    __shared__ float smem[8];
    float sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += static_cast<float>(row_in[i]);
    }
    sum = talu_block_reduce_sum_128(sum, smem);
    if (threadIdx.x == 0) out_rowsums[row] = __float2int_rn(sum);
}

// Convert U8 weights to I8 by XOR'ing each byte with 0x80 (subtract 128).
// Launch: grid=ceil(num_words/256), block=(256)
extern "C" __global__ void talu_u8_xor_to_i8(
    const unsigned int* __restrict__ in,
    unsigned int* __restrict__ out,
    unsigned int num_words
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_words) out[idx] = in[idx] ^ 0x80808080u;
}

// ─── Symmetric INT8 GEMM support ───

// Fused grouped-affine U8 dequant → I8 requant with per-row absmax scaling.
// Reads packed U8 weights + per-group scales/biases, dequantizes in registers,
// finds per-row absmax, and requantizes to I8. No F16 intermediate needed.
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_gaffine_u8_to_i8(
    const unsigned char* __restrict__ packed_u8,    // [out_dim × in_dim]
    const unsigned short* __restrict__ scales,      // [out_dim × num_groups]
    const unsigned short* __restrict__ biases,      // [out_dim × num_groups]
    signed char* __restrict__ out_i8,               // [out_dim × in_dim]
    float* __restrict__ out_row_scales,             // [out_dim]
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    const unsigned char* row_u8 = packed_u8 + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;
    const unsigned int num_groups = in_dim / group_size;
    const unsigned short* scale_row = scales + (unsigned long long)row * num_groups;
    const unsigned short* bias_row = biases + (unsigned long long)row * num_groups;

    // Pass 1: dequant to F32 in registers and find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const float val = s * static_cast<float>(row_u8[i]) + b;
        local_max = fmaxf(local_max, fabsf(val));
    }
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Pass 2: dequant + quantize to I8.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const float val = s * static_cast<float>(row_u8[i]) + b;
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Fused U4→I8 conversion with per-row absmax scaling.
// Same 2-pass structure as talu_gaffine_u8_to_i8 but with 4-bit nibble unpacking.
// Pass 1: dequant U4→F32, find per-row absmax.
// Pass 2: dequant + requant to I8.
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_gaffine_u4_to_i8(
    const unsigned int* __restrict__ packed_u4,     // [out_dim × words_per_row]
    const unsigned short* __restrict__ scales,      // [out_dim × num_groups]
    const unsigned short* __restrict__ biases,      // [out_dim × num_groups]
    signed char* __restrict__ out_i8,               // [out_dim × in_dim]
    float* __restrict__ out_row_scales,             // [out_dim]
    unsigned int in_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag
) {
    const unsigned int row = blockIdx.x;
    const unsigned int words_per_row = in_dim / 8;
    const unsigned int* row_packed = packed_u4 + (unsigned long long)row * words_per_row;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;
    const unsigned int num_groups = in_dim / group_size;
    const unsigned short* scale_row = scales + (unsigned long long)row * num_groups;
    const unsigned short* bias_row = biases + (unsigned long long)row * num_groups;

    // Pass 1: dequant to F32 in registers and find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const unsigned int word = row_packed[i / 8];
        const unsigned int nibble = (word >> ((i & 7u) * 4)) & 0xFu;
        const float val = s * static_cast<float>(nibble) + b;
        local_max = fmaxf(local_max, fabsf(val));
    }
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Pass 2: dequant + quantize to I8.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const unsigned int g = i / group_size;
        const float s = talu_decode_scale_bias_u16(scale_row[g], scales_dtype_tag);
        const float b = talu_decode_scale_bias_u16(bias_row[g], scales_dtype_tag);
        const unsigned int word = row_packed[i / 8];
        const unsigned int nibble = (word >> ((i & 7u) * 4)) & 0xFu;
        const float val = s * static_cast<float>(nibble) + b;
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// ─── F16→I8 and F32→I8 quantization kernels ───

// Quantize F16 weight rows to I8 with per-row absmax scaling.
// Each block handles one output row of the weight matrix.
// Launch: grid=(num_rows), block=(256)
extern "C" __global__ void talu_quantize_f16_to_i8(
    const unsigned short* __restrict__ weight_f16,  // [num_rows × in_dim] as raw __half bits
    signed char* __restrict__ out_i8,               // [num_rows × in_dim]
    float* __restrict__ out_row_scales,             // [num_rows]
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const unsigned short* row_in = weight_f16 + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;

    // Phase 1: find per-row absmax.
    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(__half2float(*reinterpret_cast<const __half*>(&row_in[i]))));
    }
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    // Phase 2: quantize to I8.
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = __half2float(*reinterpret_cast<const __half*>(&row_in[i]));
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Quantize F32 input rows to I8 with per-row absmax scaling (no group sums).
// Launch: grid=(rows), block=(256)
extern "C" __global__ void talu_quantize_f32_to_i8_simple(
    const float* __restrict__ input,        // [rows × in_dim]
    signed char* __restrict__ out_i8,       // [rows × in_dim]
    float* __restrict__ out_row_scales,     // [rows]
    unsigned int in_dim
) {
    const unsigned int row = blockIdx.x;
    const float* row_in = input + (unsigned long long)row * in_dim;
    signed char* row_out = out_i8 + (unsigned long long)row * in_dim;

    __shared__ float smem[8];
    float local_max = 0.0f;
    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_in[i]));
    }
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFu, local_max, offset));
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 8) {
        local_max = smem[threadIdx.x];
        for (int offset = 4; offset >= 1; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFu, local_max, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) smem[0] = local_max;
    __syncthreads();
    const float absmax = smem[0];
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) out_row_scales[row] = scale;

    for (unsigned int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float val = row_in[i];
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        row_out[i] = static_cast<signed char>(q);
    }
}

// Dequantize I32 GEMM output to F32 using per-row input and weight scales.
// output[m][n] = gemm_i32[m][n] * input_scale[m] * weight_scale[n]
// Launch: grid=(out_dim), block=(256)
extern "C" __global__ void talu_dequant_i32_scales(
    const int* __restrict__ gemm_i32,           // [rows × out_dim]
    const float* __restrict__ input_scales,     // [rows]
    const float* __restrict__ weight_scales,    // [out_dim]
    float* __restrict__ output,                 // [rows × out_dim]
    unsigned int rows,
    unsigned int out_dim
) {
    const unsigned int n = blockIdx.x;
    if (n >= out_dim) return;
    const float ws = weight_scales[n];
    for (unsigned int m = threadIdx.x; m < rows; m += blockDim.x) {
        const float is = input_scales[m];
        output[(unsigned long long)m * out_dim + n] =
            static_cast<float>(gemm_i32[(unsigned long long)m * out_dim + n]) * is * ws;
    }
}
