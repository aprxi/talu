// Converter and calibration utility kernels (dequant cache, MSE reduction, I32 dequant).

extern "C" __global__ void talu_gaffine_build_dq_weights_f32(
    const float* __restrict__ sampled_source_rows,   // [sample_rows × col_count] row-major
    const float* __restrict__ row_group_min,         // [sample_rows × group_count]
    const float* __restrict__ row_group_base_scale,  // [sample_rows × group_count]
    const float* __restrict__ group_scale_factors,   // [group_count]
    const float* __restrict__ group_bias_shifts,     // [group_count]
    const float* __restrict__ group_round_shifts,    // [group_count]
    float* __restrict__ dq_weights_out,              // [col_count × sample_rows] col-major
    unsigned int sample_rows,
    unsigned int col_count,
    unsigned int group_size,
    float max_quant_value
) {
    if (sample_rows == 0u || col_count == 0u || group_size == 0u) return;
    if ((col_count % group_size) != 0u) return;

    const unsigned int row = blockIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= sample_rows || col >= col_count) return;

    const unsigned int group_count = col_count / group_size;
    const unsigned int group_idx = col / group_size;
    const unsigned long long row_group_idx = (unsigned long long)row * group_count + group_idx;

    const float value = sampled_source_rows[(unsigned long long)row * col_count + col];
    const float min_val = row_group_min[row_group_idx];
    const float base_scale = row_group_base_scale[row_group_idx];
    const float group_scale = base_scale * group_scale_factors[group_idx];
    const float group_bias_shift = group_bias_shifts[group_idx];
    const float group_bias = (group_scale > 0.0f)
        ? (min_val + group_bias_shift * group_scale)
        : min_val;
    const float group_round_shift = group_round_shifts[group_idx];

    float quantized = 0.0f;
    if (group_scale > 0.0f) {
        const float normalized = (value - group_bias) / group_scale + group_round_shift;
        float rounded = roundf(normalized);
        rounded = fminf(max_quant_value, fmaxf(0.0f, rounded));
        quantized = rounded;
    }
    const float dequant = quantized * group_scale + group_bias;
    dq_weights_out[(unsigned long long)col * sample_rows + row] = dequant;
}

// Sum of squared error between two F32 vectors.
// Writes sum((a[i]-b[i])^2) into out_sum[0].
// Caller must zero out_sum before launch.
// Launch: grid=(ceil(n/256)), block=(256)
extern "C" __global__ void talu_reduce_mse_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out_sum,
    unsigned int n
) {
    if (n == 0u) return;

    const unsigned int tid = threadIdx.x;
    const unsigned long long global_tid = (unsigned long long)blockIdx.x * blockDim.x + tid;
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (unsigned long long i = global_tid; i < n; i += stride) {
        const float diff = a[i] - b[i];
        local_sum += diff * diff;
    }

    __shared__ float block_sum[256];
    block_sum[tid] = local_sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 0u; offset >>= 1) {
        if (tid < offset) {
            block_sum[tid] += block_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        atomicAdd(out_sum, block_sum[0]);
    }
}

// Dequantize I32 GEMM output to F32 using per-row input and weight scales.
// output[m][n] = gemm_i32[m][n] * input_scale[m] * weight_scale[n]
// Launch: grid=(ceil(out_dim/256), rows), block=(256)
extern "C" __global__ void talu_dequant_i32_scales(
    const int* __restrict__ gemm_i32,           // [rows × out_dim]
    const float* __restrict__ input_scales,     // [rows]
    const float* __restrict__ weight_scales,    // [out_dim]
    float* __restrict__ output,                 // [rows × out_dim]
    unsigned int rows,
    unsigned int out_dim
) {
    const unsigned int m = blockIdx.y;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= rows || n >= out_dim) return;
    const unsigned long long idx = (unsigned long long)m * out_dim + n;
    const float is = input_scales[m];
    const float ws = weight_scales[n];
    output[idx] = static_cast<float>(gemm_i32[idx]) * is * ws;
}

// Dequantize I32 GEMM concat output and split into separate contiguous F32 destinations.
// gemm_i32 layout: [rows × total_dim] where total_dim = dim_a + dim_b + dim_c.
// Columns [0, dim_a) → out_a[rows × dim_a], [dim_a, dim_a+dim_b) → out_b, remainder → out_c.
// Grid: (ceil(total_dim/256), rows), Block: (256). Threads map contiguous columns.
extern "C" __global__ void talu_dequant_i32_scales_split3(
    const int* __restrict__ gemm_i32,           // [rows × total_dim]
    const float* __restrict__ input_scales,     // [rows]
    const float* __restrict__ weight_scales,    // [total_dim]
    float* __restrict__ out_a,                  // [rows × dim_a]
    float* __restrict__ out_b,                  // [rows × dim_b]
    float* __restrict__ out_c,                  // [rows × dim_c]
    unsigned int rows,
    unsigned int dim_a,
    unsigned int dim_b,
    unsigned int dim_c
) {
    const unsigned int total_dim = dim_a + dim_b + dim_c;
    const unsigned int m = blockIdx.y;
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= rows || n >= total_dim) return;
    const float is = input_scales[m];
    const float ws = weight_scales[n];

    float* out;
    unsigned int out_stride, local_n;
    if (n < dim_a) {
        out = out_a; out_stride = dim_a; local_n = n;
    } else if (n < dim_a + dim_b) {
        out = out_b; out_stride = dim_b; local_n = n - dim_a;
    } else {
        out = out_c; out_stride = dim_c; local_n = n - dim_a - dim_b;
    }

    out[(unsigned long long)m * out_stride + local_n] =
        static_cast<float>(gemm_i32[(unsigned long long)m * total_dim + n]) * is * ws;
}
