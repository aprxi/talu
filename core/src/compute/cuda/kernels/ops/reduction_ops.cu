static __device__ __forceinline__ float talu_fast_exp_scalar(float x) {
    const float log2e = 1.4426950408889634f;
    const float ln2_hi = 0.693359375f;
    const float ln2_lo = -2.12194440e-4f;
    const float exp_hi = 88.3762626647949f;
    const float exp_lo = -88.3762626647949f;
    const float p0 = 1.9875691500e-4f;
    const float p1 = 1.3981999507e-3f;
    const float p2 = 8.3334519073e-3f;
    const float p3 = 4.1665795894e-2f;
    const float p4 = 1.6666665459e-1f;
    const float p5 = 5.0000001201e-1f;

    float x_clamped = fminf(fmaxf(x, exp_lo), exp_hi);
    const float fx = floorf(x_clamped * log2e + 0.5f);
    const int fxi = (int)fx;
    x_clamped = x_clamped - fx * ln2_hi - fx * ln2_lo;

    float y = p0;
    y = y * x_clamped + p1;
    y = y * x_clamped + p2;
    y = y * x_clamped + p3;
    y = y * x_clamped + p4;
    y = y * x_clamped + p5;
    y = y * x_clamped * x_clamped + x_clamped + 1.0f;

    union {
        unsigned int u;
        float f;
    } pow2n;
    pow2n.u = (unsigned int)((fxi + 127) << 23);
    return y * pow2n.f;
}

extern "C" __global__ void talu_silu_f32(
    float* out,
    const float* input,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    out[index] = x / (1.0f + talu_fast_exp_scalar(-x));
}

extern "C" __global__ void talu_silu_mul_f32(
    float* out,
    const float* gate,
    const float* up,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    const float silu_x = x / (1.0f + talu_fast_exp_scalar(-x));
    out[index] = silu_x * up[index];
}

extern "C" __global__ void talu_gelu_mul_f32(
    float* out,
    const float* gate,
    const float* up,
    unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    const float x3 = x * x * x;
    const float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    const float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
    out[index] = gelu_x * up[index];
}

extern "C" __global__ void talu_argmax_f32(
    const float* input,
    unsigned int count,
    unsigned int* out_index
) {
    const unsigned int tid = threadIdx.x;
    float best_val = -3.402823466e+38f;
    unsigned int best_idx = 0;

    for (unsigned int idx = tid; idx < count; idx += blockDim.x) {
        const float v = input[idx];
        if (v > best_val || (v == best_val && idx < best_idx)) {
            best_val = v;
            best_idx = idx;
        }
    }

    __shared__ float shared_val[256];
    __shared__ unsigned int shared_idx[256];
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = shared_val[tid + stride];
            const unsigned int other_idx = shared_idx[tid + stride];
            const float cur_val = shared_val[tid];
            const unsigned int cur_idx = shared_idx[tid];
            if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                shared_val[tid] = other_val;
                shared_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_index[0] = shared_idx[0];
    }
}

// Two-phase parallel top-k.  Phase 1 splits each row across CHUNKS
// blocks so that all SMs participate.  Phase 2 merges the per-chunk
// candidates into the final top-k result per row.
//
// Phase 1 grid: (rows * chunks, 1, 1)   block: (256, 1, 1)
// Phase 2 grid: (rows, 1, 1)            block: (256, 1, 1)

extern "C" __global__ void talu_topk_rows_phase1(
    float* chunk_values,       // [rows, chunks, k]
    unsigned int* chunk_ids,   // [rows, chunks, k]
    const float* logits,       // [rows, vocab]
    unsigned int rows,
    unsigned int vocab,
    unsigned int chunks,
    unsigned int k
) {
    const unsigned int global_block = blockIdx.x;
    const unsigned int row = global_block / chunks;
    const unsigned int chunk = global_block % chunks;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int chunk_size = (vocab + chunks - 1) / chunks;
    const unsigned int chunk_start = chunk * chunk_size;
    const unsigned int chunk_end = min(chunk_start + chunk_size, vocab);
    const float* row_logits = logits + row * vocab;
    const float mask_value = -3.402823466e+38f;

    __shared__ float shared_val[256];
    __shared__ unsigned int shared_idx[256];

    // Find top-k within this chunk using repeated scans.
    // chunk_size / 256 is small (~19 elements/thread for 32 chunks
    // over 152K vocab), so k scans are cheap.
    const unsigned int out_base = (row * chunks + chunk) * k;

    // Copy chunk to shared-memory scan range for masking.
    // We mask in the output buffer rather than the input logits
    // (which are read-only to allow graph capture).
    for (unsigned int pick = 0; pick < k; ++pick) {
        float local_best_val = mask_value;
        unsigned int local_best_idx = 0;

        for (unsigned int col = chunk_start + tid; col < chunk_end; col += blockDim.x) {
            float value = row_logits[col];
            // Check if this index was already picked in an earlier round.
            // For small k this is cheaper than maintaining a separate mask.
            bool already_picked = false;
            for (unsigned int p = 0; p < pick; ++p) {
                if (chunk_ids[out_base + p] == col) { already_picked = true; break; }
            }
            if (already_picked) value = mask_value;
            if (value > local_best_val || (value == local_best_val && col < local_best_idx)) {
                local_best_val = value;
                local_best_idx = col;
            }
        }

        shared_val[tid] = local_best_val;
        shared_idx[tid] = local_best_idx;
        __syncthreads();

        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_val = shared_val[tid + stride];
                const unsigned int other_idx = shared_idx[tid + stride];
                const float cur_val = shared_val[tid];
                const unsigned int cur_idx = shared_idx[tid];
                if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                    shared_val[tid] = other_val;
                    shared_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            chunk_values[out_base + pick] = shared_val[0];
            chunk_ids[out_base + pick] = shared_idx[0];
        }
        __syncthreads();
    }
}

extern "C" __global__ void talu_topk_rows_phase2(
    float* out_values,
    unsigned int* out_ids,
    const float* chunk_values,     // [rows, chunks, k]
    const unsigned int* chunk_ids, // [rows, chunks, k]
    unsigned int rows,
    unsigned int chunks,
    unsigned int k,
    unsigned int row_stride
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int total_candidates = chunks * k;
    const float mask_value = -3.402823466e+38f;
    const unsigned int chunk_base = row * chunks * k;

    __shared__ float shared_val[256];
    __shared__ unsigned int shared_idx[256];

    for (unsigned int pick = 0; pick < k; ++pick) {
        float local_best_val = mask_value;
        unsigned int local_best_idx = 0;

        for (unsigned int i = tid; i < total_candidates; i += blockDim.x) {
            const float value = chunk_values[chunk_base + i];
            const unsigned int orig_id = chunk_ids[chunk_base + i];
            // Skip already-picked entries by checking output so far.
            bool already_picked = false;
            for (unsigned int p = 0; p < pick; ++p) {
                if (out_ids[row * row_stride + p] == orig_id) { already_picked = true; break; }
            }
            if (already_picked) continue;
            if (value > local_best_val || (value == local_best_val && orig_id < local_best_idx)) {
                local_best_val = value;
                local_best_idx = orig_id;
            }
        }

        shared_val[tid] = local_best_val;
        shared_idx[tid] = local_best_idx;
        __syncthreads();

        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_val = shared_val[tid + stride];
                const unsigned int other_idx = shared_idx[tid + stride];
                const float cur_val = shared_val[tid];
                const unsigned int cur_idx = shared_idx[tid];
                if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                    shared_val[tid] = other_val;
                    shared_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            out_values[row * row_stride + pick] = shared_val[0];
            out_ids[row * row_stride + pick] = shared_idx[0];
        }
        __syncthreads();
    }
}
