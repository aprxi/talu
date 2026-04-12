// Grouped-affine U4 quantized GEMV, dequantization, and converter kernels.

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
static constexpr unsigned int U4_BATCH_TILE = 4;
static constexpr unsigned int U4_BATCH_TILE_X8 = 8;

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

// Inner-batch U4 GEMV: weight loaded once from DRAM, reused across BATCH input
// rows from registers. BATCH=1 compiles to identical code as the original
// single-row kernel. BATCH=2..8 adds one accumulator per extra row.
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_gaffine_u4_matvec_batched(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int lane,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int words_per_row = in_dim >> 3;
    const unsigned int words_per_group = group_size >> 3;
    const unsigned int groups_per_row = in_dim / group_size;
    const unsigned int* weight_row = packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned short* scale_row = scales + (unsigned long long)out_idx * groups_per_row;
    const unsigned short* bias_row = biases + (unsigned long long)out_idx * groups_per_row;

    float acc[BATCH] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        // Weight + scale/bias: loaded ONCE, shared across all batch rows.
        unsigned int w0, w1;
        asm volatile(
            "ld.global.cs.v2.u32 {%0, %1}, [%2];"
            : "=r"(w0), "=r"(w1)
            : "l"(&weight_row[w]));

        const unsigned int base_elem = w << 3;
        const unsigned int g0 = w / words_per_group;
        const float s0 = talu_decode_scale_bias_u16(scale_row[g0], scales_dtype_tag);
        const float b0 = talu_decode_scale_bias_u16(bias_row[g0], scales_dtype_tag);
        const unsigned int g1 = (w + 1) / words_per_group;
        const float s1 = talu_decode_scale_bias_u16(scale_row[g1], scales_dtype_tag);
        const float b1 = talu_decode_scale_bias_u16(bias_row[g1], scales_dtype_tag);

        // Word 0: per-batch input load + accumulate.
        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const float4 i0 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem]);
            const float4 i1 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 4]);
            U4_PROCESS_WORD(w0, i0, i1, s0, b0, acc[b]);
        }
        // Word 1: per-batch input load + accumulate.
        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const float4 i2 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 8]);
            const float4 i3 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 12]);
            U4_PROCESS_WORD(w1, i2, i3, s1, b1, acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float result = talu_quant_warp_sum_f32(acc[b]);
        if (lane == 0) {
            out[(unsigned long long)b * out_dim + out_idx] =
                residual
                    ? result + residual[(unsigned long long)b * out_dim + out_idx]
                    : result;
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_gaffine_u4_matvec_f32(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE
        ? batch_rows - batch_base : U4_BATCH_TILE;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const float* res_tile = residual
        ? residual + (unsigned long long)batch_base * out_dim : nullptr;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_matvec_batched<1>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 1u, res_tile); break;
        case 2u: talu_gaffine_u4_matvec_batched<2>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 2u, res_tile); break;
        case 3u: talu_gaffine_u4_matvec_batched<3>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 3u, res_tile); break;
        case 4u: talu_gaffine_u4_matvec_batched<4>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 4u, res_tile); break;
        case 5u: talu_gaffine_u4_matvec_batched<5>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 5u, res_tile); break;
        case 6u: talu_gaffine_u4_matvec_batched<6>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 6u, res_tile); break;
        case 7u: talu_gaffine_u4_matvec_batched<7>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 7u, res_tile); break;
        case 8u: talu_gaffine_u4_matvec_batched<8>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 8u, res_tile); break;
        default: return;
    }
}

// Tile-8 decode-specialized variant. Keeps the default tile-4 kernel hot path
// unchanged for n<=4 while enabling single-pass weight reuse at n>4.
extern "C" __global__ __launch_bounds__(128, 4) void talu_gaffine_u4_matvec_f32_tile8(
    const float* input,
    const unsigned int* packed_weight,
    const unsigned short* scales,
    const unsigned short* biases,
    float* out,
    unsigned int in_dim,
    unsigned int out_dim,
    unsigned int group_size,
    unsigned int scales_dtype_tag,
    unsigned int batch_rows,
    const float* residual
) {
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE_X8
        ? batch_rows - batch_base : U4_BATCH_TILE_X8;

    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int out_idx = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_idx >= out_dim) return;
    if (group_size == 0 || (in_dim % group_size) != 0 || (group_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;
    const float* res_tile = residual
        ? residual + (unsigned long long)batch_base * out_dim : nullptr;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_matvec_batched<1>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 1u, res_tile); break;
        case 2u: talu_gaffine_u4_matvec_batched<2>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 2u, res_tile); break;
        case 3u: talu_gaffine_u4_matvec_batched<3>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 3u, res_tile); break;
        case 4u: talu_gaffine_u4_matvec_batched<4>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 4u, res_tile); break;
        case 5u: talu_gaffine_u4_matvec_batched<5>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 5u, res_tile); break;
        case 6u: talu_gaffine_u4_matvec_batched<6>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 6u, res_tile); break;
        case 7u: talu_gaffine_u4_matvec_batched<7>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 7u, res_tile); break;
        case 8u: talu_gaffine_u4_matvec_batched<8>(input_tile, packed_weight, scales, biases, out_tile, in_dim, out_dim, out_idx, group_size, scales_dtype_tag, lane, 8u, res_tile); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_gaffine_u4_matvec_qkv_f32(
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
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE
        ? batch_rows - batch_base : U4_BATCH_TILE;
    if (in_dim == 0 || (in_dim % 32) != 0) return;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    // Determine which projection this warp handles.
    const unsigned int* pw;
    const unsigned short* sc;
    const unsigned short* bi;
    float* out_ptr;
    unsigned int out_dim_proj, row_idx, grp_size, dtype_tag;

    if (out_index < q_out_dim) {
        pw = q_packed_weight; sc = q_scales; bi = q_biases;
        out_ptr = q_out; out_dim_proj = q_out_dim;
        row_idx = out_index; grp_size = q_group_size; dtype_tag = q_scales_dtype_tag;
    } else if (out_index < qk_dim) {
        pw = k_packed_weight; sc = k_scales; bi = k_biases;
        out_ptr = k_out; out_dim_proj = k_out_dim;
        row_idx = out_index - q_out_dim; grp_size = k_group_size; dtype_tag = k_scales_dtype_tag;
    } else {
        pw = v_packed_weight; sc = v_scales; bi = v_biases;
        out_ptr = v_out; out_dim_proj = v_out_dim;
        row_idx = out_index - qk_dim; grp_size = v_group_size; dtype_tag = v_scales_dtype_tag;
    }
    if (grp_size == 0 || (in_dim % grp_size) != 0 || (grp_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_matvec_batched<1>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 1u, nullptr); break;
        case 2u: talu_gaffine_u4_matvec_batched<2>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 2u, nullptr); break;
        case 3u: talu_gaffine_u4_matvec_batched<3>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 3u, nullptr); break;
        case 4u: talu_gaffine_u4_matvec_batched<4>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 4u, nullptr); break;
        case 5u: talu_gaffine_u4_matvec_batched<5>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 5u, nullptr); break;
        case 6u: talu_gaffine_u4_matvec_batched<6>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 6u, nullptr); break;
        case 7u: talu_gaffine_u4_matvec_batched<7>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 7u, nullptr); break;
        case 8u: talu_gaffine_u4_matvec_batched<8>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 8u, nullptr); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 4) void talu_gaffine_u4_matvec_qkv_f32_tile8(
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
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE_X8
        ? batch_rows - batch_base : U4_BATCH_TILE_X8;
    if (in_dim == 0 || (in_dim % 32) != 0) return;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    const unsigned int qk_dim = q_out_dim + k_out_dim;
    const unsigned int total_dim = qk_dim + v_out_dim;
    if (out_index >= total_dim) return;

    // Determine which projection this warp handles.
    const unsigned int* pw;
    const unsigned short* sc;
    const unsigned short* bi;
    float* out_ptr;
    unsigned int out_dim_proj, row_idx, grp_size, dtype_tag;

    if (out_index < q_out_dim) {
        pw = q_packed_weight; sc = q_scales; bi = q_biases;
        out_ptr = q_out; out_dim_proj = q_out_dim;
        row_idx = out_index; grp_size = q_group_size; dtype_tag = q_scales_dtype_tag;
    } else if (out_index < qk_dim) {
        pw = k_packed_weight; sc = k_scales; bi = k_biases;
        out_ptr = k_out; out_dim_proj = k_out_dim;
        row_idx = out_index - q_out_dim; grp_size = k_group_size; dtype_tag = k_scales_dtype_tag;
    } else {
        pw = v_packed_weight; sc = v_scales; bi = v_biases;
        out_ptr = v_out; out_dim_proj = v_out_dim;
        row_idx = out_index - qk_dim; grp_size = v_group_size; dtype_tag = v_scales_dtype_tag;
    }
    if (grp_size == 0 || (in_dim % grp_size) != 0 || (grp_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_matvec_batched<1>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 1u, nullptr); break;
        case 2u: talu_gaffine_u4_matvec_batched<2>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 2u, nullptr); break;
        case 3u: talu_gaffine_u4_matvec_batched<3>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 3u, nullptr); break;
        case 4u: talu_gaffine_u4_matvec_batched<4>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 4u, nullptr); break;
        case 5u: talu_gaffine_u4_matvec_batched<5>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 5u, nullptr); break;
        case 6u: talu_gaffine_u4_matvec_batched<6>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 6u, nullptr); break;
        case 7u: talu_gaffine_u4_matvec_batched<7>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 7u, nullptr); break;
        case 8u: talu_gaffine_u4_matvec_batched<8>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 8u, nullptr); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 2) void talu_gaffine_u4_matvec_gate_up_f32(
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
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE
        ? batch_rows - batch_base : U4_BATCH_TILE;
    if (in_dim == 0 || (in_dim % 32) != 0) return;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    const unsigned int total_dim = gate_out_dim + up_out_dim;
    if (out_index >= total_dim) return;

    const unsigned int* pw;
    const unsigned short* sc;
    const unsigned short* bi;
    float* out_ptr;
    unsigned int out_dim_proj, row_idx, grp_size, dtype_tag;

    if (out_index < gate_out_dim) {
        pw = gate_packed_weight; sc = gate_scales; bi = gate_biases;
        out_ptr = gate_out; out_dim_proj = gate_out_dim;
        row_idx = out_index; grp_size = gate_group_size; dtype_tag = gate_scales_dtype_tag;
    } else {
        pw = up_packed_weight; sc = up_scales; bi = up_biases;
        out_ptr = up_out; out_dim_proj = up_out_dim;
        row_idx = out_index - gate_out_dim; grp_size = up_group_size; dtype_tag = up_scales_dtype_tag;
    }
    if (grp_size == 0 || (in_dim % grp_size) != 0 || (grp_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out_ptr + (unsigned long long)batch_base * out_dim_proj;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_matvec_batched<1>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 1u, nullptr); break;
        case 2u: talu_gaffine_u4_matvec_batched<2>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 2u, nullptr); break;
        case 3u: talu_gaffine_u4_matvec_batched<3>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 3u, nullptr); break;
        case 4u: talu_gaffine_u4_matvec_batched<4>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 4u, nullptr); break;
        case 5u: talu_gaffine_u4_matvec_batched<5>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 5u, nullptr); break;
        case 6u: talu_gaffine_u4_matvec_batched<6>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 6u, nullptr); break;
        case 7u: talu_gaffine_u4_matvec_batched<7>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 7u, nullptr); break;
        case 8u: talu_gaffine_u4_matvec_batched<8>(input_tile, pw, sc, bi, out_tile, in_dim, out_dim_proj, row_idx, grp_size, dtype_tag, lane, 8u, nullptr); break;
        default: return;
    }
}

// Fused U4 gate/up + SiLU batched template: loads both gate and up weights once
// from DRAM per iteration, reuses across BATCH input rows. SiLU fusion applied
// after warp reduction.
template <unsigned int BATCH>
static __device__ __forceinline__ void talu_gaffine_u4_gate_up_silu_batched(
    const float* input,
    const unsigned int* gate_packed_weight,
    const unsigned short* gate_scales,
    const unsigned short* gate_biases,
    const unsigned int* up_packed_weight,
    const unsigned short* up_scales,
    const unsigned short* up_biases,
    float* out,
    unsigned int out_dim,
    unsigned int out_idx,
    unsigned int gate_group_size,
    unsigned int gate_scales_dtype_tag,
    unsigned int up_group_size,
    unsigned int up_scales_dtype_tag,
    unsigned int in_dim,
    unsigned int lane,
    unsigned int batch_rows
) {
    const unsigned int words_per_row = in_dim >> 3;
    const unsigned int gate_wpg = gate_group_size >> 3;
    const unsigned int up_wpg = up_group_size >> 3;
    const unsigned int gate_gpr = in_dim / gate_group_size;
    const unsigned int up_gpr = in_dim / up_group_size;
    const unsigned int* g_row = gate_packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned int* u_row = up_packed_weight + (unsigned long long)out_idx * words_per_row;
    const unsigned short* gs_row = gate_scales + (unsigned long long)out_idx * gate_gpr;
    const unsigned short* gb_row = gate_biases + (unsigned long long)out_idx * gate_gpr;
    const unsigned short* us_row = up_scales + (unsigned long long)out_idx * up_gpr;
    const unsigned short* ub_row = up_biases + (unsigned long long)out_idx * up_gpr;

    float gate_acc[BATCH] = {};
    float up_acc[BATCH] = {};

    for (unsigned int w = lane << 1; w < words_per_row; w += TALU_QUANT_WARP_SIZE << 1) {
        // Gate + up weights: loaded ONCE, shared across all batch rows.
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
        const unsigned int gg0 = w / gate_wpg;
        const float gs0 = talu_decode_scale_bias_u16(gs_row[gg0], gate_scales_dtype_tag);
        const float gb0 = talu_decode_scale_bias_u16(gb_row[gg0], gate_scales_dtype_tag);
        const unsigned int ug0 = w / up_wpg;
        const float us0 = talu_decode_scale_bias_u16(us_row[ug0], up_scales_dtype_tag);
        const float ub0 = talu_decode_scale_bias_u16(ub_row[ug0], up_scales_dtype_tag);
        const unsigned int gg1 = (w + 1) / gate_wpg;
        const float gs1 = talu_decode_scale_bias_u16(gs_row[gg1], gate_scales_dtype_tag);
        const float gb1 = talu_decode_scale_bias_u16(gb_row[gg1], gate_scales_dtype_tag);
        const unsigned int ug1 = (w + 1) / up_wpg;
        const float us1 = talu_decode_scale_bias_u16(us_row[ug1], up_scales_dtype_tag);
        const float ub1 = talu_decode_scale_bias_u16(ub_row[ug1], up_scales_dtype_tag);

        // Word 0: per-batch input load + accumulate gate and up.
        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const float4 i0 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem]);
            const float4 i1 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 4]);
            U4_PROCESS_WORD(gw0, i0, i1, gs0, gb0, gate_acc[b]);
            U4_PROCESS_WORD(uw0, i0, i1, us0, ub0, up_acc[b]);
        }
        // Word 1: per-batch input load + accumulate gate and up.
        #pragma unroll
        for (unsigned int b = 0; b < BATCH; b++) {
            if (b >= batch_rows) break;
            const float4 i2 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 8]);
            const float4 i3 = *reinterpret_cast<const float4*>(
                &input[(unsigned long long)b * in_dim + base_elem + 12]);
            U4_PROCESS_WORD(gw1, i2, i3, gs1, gb1, gate_acc[b]);
            U4_PROCESS_WORD(uw1, i2, i3, us1, ub1, up_acc[b]);
        }
    }

    #pragma unroll
    for (unsigned int b = 0; b < BATCH; b++) {
        if (b >= batch_rows) break;
        float g = talu_quant_warp_sum_f32(gate_acc[b]);
        float u = talu_quant_warp_sum_f32(up_acc[b]);
        if (lane == 0) {
            const float sigma = 1.0f / (1.0f + expf(-g));
            out[(unsigned long long)b * out_dim + out_idx] = g * sigma * u;
        }
    }
}

// Grid: (ceil(out_dim/4), ceil(batch_rows/U4_BATCH_TILE)), Block: (128 = 4 warps)
extern "C" __global__ __launch_bounds__(128, 2) void talu_gaffine_u4_matvec_gate_up_silu_f32(
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
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE
        ? batch_rows - batch_base : U4_BATCH_TILE;
    if (in_dim == 0 || (in_dim % 32) != 0) return;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_index >= out_dim) return;
    if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_gate_up_silu_batched<1>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 1u); break;
        case 2u: talu_gaffine_u4_gate_up_silu_batched<2>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 2u); break;
        case 3u: talu_gaffine_u4_gate_up_silu_batched<3>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 3u); break;
        case 4u: talu_gaffine_u4_gate_up_silu_batched<4>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 4u); break;
        case 5u: talu_gaffine_u4_gate_up_silu_batched<5>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 5u); break;
        case 6u: talu_gaffine_u4_gate_up_silu_batched<6>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 6u); break;
        case 7u: talu_gaffine_u4_gate_up_silu_batched<7>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 7u); break;
        case 8u: talu_gaffine_u4_gate_up_silu_batched<8>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 8u); break;
        default: return;
    }
}

extern "C" __global__ __launch_bounds__(128, 4) void talu_gaffine_u4_matvec_gate_up_silu_f32_tile8(
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
    const unsigned int batch_base = blockIdx.y * U4_BATCH_TILE_X8;
    if (batch_base >= batch_rows) return;
    const unsigned int tile_rows = batch_rows - batch_base < U4_BATCH_TILE_X8
        ? batch_rows - batch_base : U4_BATCH_TILE_X8;
    if (in_dim == 0 || (in_dim % 32) != 0) return;

    const unsigned int lane = threadIdx.x & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = threadIdx.x / TALU_QUANT_WARP_SIZE;
    const unsigned int out_index = blockIdx.x * U4_WARPS_PER_BLOCK + warp_id;
    if (out_index >= out_dim) return;
    if (gate_group_size == 0 || (in_dim % gate_group_size) != 0 || (gate_group_size % 8) != 0) return;
    if (up_group_size == 0 || (in_dim % up_group_size) != 0 || (up_group_size % 8) != 0) return;

    const float* input_tile = input + (unsigned long long)batch_base * in_dim;
    float* out_tile = out + (unsigned long long)batch_base * out_dim;

    switch (tile_rows) {
        case 1u: talu_gaffine_u4_gate_up_silu_batched<1>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 1u); break;
        case 2u: talu_gaffine_u4_gate_up_silu_batched<2>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 2u); break;
        case 3u: talu_gaffine_u4_gate_up_silu_batched<3>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 3u); break;
        case 4u: talu_gaffine_u4_gate_up_silu_batched<4>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 4u); break;
        case 5u: talu_gaffine_u4_gate_up_silu_batched<5>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 5u); break;
        case 6u: talu_gaffine_u4_gate_up_silu_batched<6>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 6u); break;
        case 7u: talu_gaffine_u4_gate_up_silu_batched<7>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 7u); break;
        case 8u: talu_gaffine_u4_gate_up_silu_batched<8>(input_tile, gate_packed_weight, gate_scales, gate_biases, up_packed_weight, up_scales, up_biases, out_tile, out_dim, out_index, gate_group_size, gate_scales_dtype_tag, up_group_size, up_scales_dtype_tag, in_dim, lane, 8u); break;
        default: return;
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

// Grid: (group_count, row_count), Block: (128)
extern "C" __global__ void talu_gaffine_quantize_u4_f32(
    const float* __restrict__ input,                // [row_count × col_count]
    const float* __restrict__ group_scale_factors,  // [group_count]
    const float* __restrict__ group_bias_shifts,    // [group_count]
    const float* __restrict__ group_round_shifts,   // [group_count]
    unsigned int* __restrict__ packed_out,          // [row_count × packed_col_count]
    unsigned short* __restrict__ scales_out,        // [row_count × group_count] BF16
    unsigned short* __restrict__ biases_out,        // [row_count × group_count] BF16
    unsigned int row_count,
    unsigned int col_count,
    unsigned int group_size,
    unsigned int packed_col_count
) {
    if (row_count == 0u || col_count == 0u || group_size == 0u) return;
    if ((col_count % group_size) != 0u || (group_size % 8u) != 0u) return;

    const unsigned int row = blockIdx.y;
    const unsigned int group_idx = blockIdx.x;
    const unsigned int group_count = col_count / group_size;
    if (row >= row_count || group_idx >= group_count) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & (TALU_QUANT_WARP_SIZE - 1u);
    const unsigned int warp_id = tid / TALU_QUANT_WARP_SIZE;
    const unsigned int warp_count = (blockDim.x + TALU_QUANT_WARP_SIZE - 1u) / TALU_QUANT_WARP_SIZE;

    const float* row_in = input + (unsigned long long)row * col_count;
    const unsigned int group_start = group_idx * group_size;
    const unsigned int words_per_group = group_size / 8u;

    float local_min = 3.402823466e+38F;
    float local_max = -3.402823466e+38F;
    for (unsigned int i = tid; i < group_size; i += blockDim.x) {
        const float value = row_in[group_start + i];
        local_min = fminf(local_min, value);
        local_max = fmaxf(local_max, value);
    }

    local_min = talu_quant_warp_min_f32(local_min);
    local_max = talu_quant_warp_max_f32(local_max);

    __shared__ float warp_min[8];
    __shared__ float warp_max[8];
    __shared__ float shared_scale;
    __shared__ float shared_bias;
    __shared__ float shared_round_shift;
    if (lane == 0u) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0u) {
        float block_min = (lane < warp_count) ? warp_min[lane] : 3.402823466e+38F;
        float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
        block_min = talu_quant_warp_min_f32(block_min);
        block_max = talu_quant_warp_max_f32(block_max);
        if (lane == 0u) {
            const float base_scale = (block_max > block_min) ? ((block_max - block_min) / 15.0f) : 0.0f;
            const float scale_factor = group_scale_factors[group_idx];
            const float group_scale = base_scale * scale_factor;
            const float group_bias_shift = group_bias_shifts[group_idx];
            const float group_bias = (group_scale > 0.0f)
                ? (block_min + group_bias_shift * group_scale)
                : block_min;
            const float group_round_shift = group_round_shifts[group_idx];

            shared_scale = group_scale;
            shared_bias = group_bias;
            shared_round_shift = group_round_shift;

            const unsigned long long sb_index = (unsigned long long)row * group_count + group_idx;
            scales_out[sb_index] = talu_quant_f32_to_bf16_rne(group_scale);
            biases_out[sb_index] = talu_quant_f32_to_bf16_rne(group_bias);
        }
    }
    __syncthreads();

    const float group_scale = shared_scale;
    const float group_bias = shared_bias;
    const float group_round_shift = shared_round_shift;

    const unsigned int group_word_offset = group_start / 8u;
    const unsigned long long row_word_base = (unsigned long long)row * packed_col_count + group_word_offset;
    for (unsigned int word_idx = tid; word_idx < words_per_group; word_idx += blockDim.x) {
        const unsigned int value_base = group_start + word_idx * 8u;
        unsigned int packed_word = 0u;

        #pragma unroll
        for (unsigned int value_idx = 0; value_idx < 8u; ++value_idx) {
            const float value = row_in[value_base + value_idx];
            unsigned int quantized = 0u;
            if (group_scale > 0.0f) {
                const float normalized = (value - group_bias) / group_scale + group_round_shift;
                float rounded = roundf(normalized);
                rounded = fminf(15.0f, fmaxf(0.0f, rounded));
                quantized = static_cast<unsigned int>(rounded);
            }
            packed_word |= (quantized << (value_idx * 4u));
        }

        packed_out[row_word_base + word_idx] = packed_word;
    }
}

// Quantize F32 weights to grouped-affine U8 layout.
// Grid: (group_count, row_count), Block: (128)
