// MLX Bridge - Basic Lazy Operations
//
// Provides C-callable wrappers for MLX array operations.
// All operations are "lazy" - they build a computation graph without executing.
// Call mlx_eval() to execute the graph.

#include "compute_common.h"

extern "C" {

// ============================================================================
// Arithmetic Operations
// ============================================================================

void* mlx_lazy_add(const void* a, const void* b) {
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(lhs + rhs);
}

void* mlx_lazy_multiply(const void* a, const void* b) {
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(lhs * rhs);
}

void* mlx_lazy_multiply_scalar(const void* a, float scalar) {
    const auto& input = *static_cast<const array*>(a);
    return pool_array(input * scalar);
}

// ============================================================================
// Matrix Operations
// ============================================================================

void* mlx_lazy_matmul(const void* a, const void* b) {
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(matmul(lhs, rhs));
}

void* mlx_lazy_quantized_matmul(
    const void* input, const void* weights, const void* scales, const void* biases,
    size_t group_size, size_t bits, bool transpose_weights
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& weights_arr = *static_cast<const array*>(weights);
    const auto& scales_arr = *static_cast<const array*>(scales);
    const auto& biases_arr = *static_cast<const array*>(biases);
    return pool_array(quantized_matmul(
        input_arr,
        weights_arr,
        scales_arr,
        biases_arr,
        transpose_weights,
        static_cast<int>(group_size),
        static_cast<int>(bits),
        "affine"
    ));
}

// ============================================================================
// Shape Operations
// ============================================================================

void* mlx_lazy_reshape(const void* input, const size_t* shape, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    // Optimized paths for common dimensions (avoid heap allocation)
    if (ndim == 2) {
        return pool_array(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1])}));
    } else if (ndim == 3) {
        return pool_array(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2])}));
    } else if (ndim == 4) {
        return pool_array(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2]), static_cast<int>(shape[3])}));
    }
    Shape shape_dims;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        shape_dims.push_back(static_cast<int>(shape[dim_idx]));
    }
    return pool_array(reshape(input_arr, shape_dims));
}

// Persistent reshape - heap-allocated, survives pool resets
void* mlx_persistent_reshape(const void* input, const size_t* shape, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    if (ndim == 2) {
        return new array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1])})));
    } else if (ndim == 3) {
        return new array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2])})));
    } else if (ndim == 4) {
        return new array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2]), static_cast<int>(shape[3])})));
    }
    Shape shape_dims;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        shape_dims.push_back(static_cast<int>(shape[dim_idx]));
    }
    return new array(contiguous(reshape(input_arr, shape_dims)));
}

void* mlx_lazy_transpose(const void* input, const size_t* axes, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    // Optimized paths for common dimensions
    if (ndim == 2) {
        return pool_array(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1])}));
    } else if (ndim == 3) {
        return pool_array(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1]),
             static_cast<int>(axes[2])}));
    } else if (ndim == 4) {
        return pool_array(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1]),
             static_cast<int>(axes[2]), static_cast<int>(axes[3])}));
    }
    std::vector<int> axes_vec;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        axes_vec.push_back(static_cast<int>(axes[dim_idx]));
    }
    return pool_array(transpose(input_arr, axes_vec));
}

void* mlx_lazy_concatenate(const void* a, const void* b, size_t axis) {
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(concatenate({
        lhs,
        rhs
    }, static_cast<int>(axis)));
}

void* mlx_lazy_repeat(const void* input, size_t repeats, size_t axis) {
    const auto& input_arr = *static_cast<const array*>(input);
    return pool_array(repeat(
        input_arr,
        static_cast<int>(repeats),
        static_cast<int>(axis)
    ));
}

// ============================================================================
// Slicing Operations
// ============================================================================

void* mlx_lazy_slice(const void* input, const int* starts, const int* ends, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    Shape start(starts, starts + ndim);
    Shape stop(ends, ends + ndim);
    return pool_array(slice(input_arr, start, stop));
}

// Persistent slice - heap-allocated, survives pool resets
// Use for weight slices that need to persist across forward passes
void* mlx_persistent_slice(const void* input, const int* starts, const int* ends, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    Shape start(starts, starts + ndim);
    Shape stop(ends, ends + ndim);
    // Use contiguous() to force a copy - ensures the slice is independent of the source
    return new array(contiguous(slice(input_arr, start, stop)));
}

void* mlx_lazy_slice_last(const void* input) {
    // Extract last position from [B, L, V] -> [V]
    const auto& input_arr = *static_cast<const array*>(input);
    int seq_len = input_arr.shape(1);
    int vocab_size = input_arr.shape(2);
    Shape start = {0, seq_len - 1, 0};
    Shape stop = {1, seq_len, vocab_size};
    auto sliced = slice(input_arr, start, stop);
    return pool_array(reshape(sliced, {vocab_size}));
}

void* mlx_lazy_slice_update(
    const void* input, const void* update,
    const int* starts, const int* ends, size_t ndim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& update_arr = *static_cast<const array*>(update);
    Shape start(starts, starts + ndim);
    Shape stop(ends, ends + ndim);
    return pool_array(slice_update(input_arr, update_arr, start, stop));
}

// ============================================================================
// Activation Functions
// ============================================================================

void* mlx_lazy_softmax(const void* input, int axis) {
    const auto& input_arr = *static_cast<const array*>(input);
    return pool_array(softmax(input_arr, axis));
}

void* mlx_lazy_silu(const void* input) {
    const auto& input_arr = *static_cast<const array*>(input);
    return pool_array(input_arr * sigmoid(input_arr));
}

void* mlx_lazy_rms_norm(const void* input, const void* weight, float eps) {
    return pool_array(fast::rms_norm(
        *static_cast<const array*>(input),
        *static_cast<const array*>(weight),
        eps
    ));
}

void* mlx_add_one(const void* arr) {
    return pool_array(1.0f + *static_cast<const array*>(arr));
}

void* mlx_scale_by_sqrt(const void* arr, size_t d_model) {
    float scale = std::sqrt(static_cast<float>(d_model));
    return pool_array(*static_cast<const array*>(arr) * scale);
}

void* mlx_lazy_rope(const void* input, size_t head_dim, size_t offset, float rope_base) {
    return pool_array(fast::rope(
        *static_cast<const array*>(input),
        static_cast<int>(head_dim),
        false,
        rope_base,
        1.0f,
        static_cast<int>(offset)
    ));
}

void* mlx_lazy_attention(const void* q, const void* k, const void* v, float scale, bool causal) {
    return pool_array(fast::scaled_dot_product_attention(
        *static_cast<const array*>(q),
        *static_cast<const array*>(k),
        *static_cast<const array*>(v),
        scale,
        causal ? "causal" : ""
    ));
}

// ============================================================================
// Reduction Operations
// ============================================================================

void* mlx_lazy_argmax(const void* handle, int axis) {
    const auto& input_arr = *static_cast<const array*>(handle);
    return pool_array(argmax(input_arr, axis));
}

// ============================================================================
// Creation Operations
// ============================================================================

void* mlx_lazy_full(const size_t* shape, size_t ndim, float value) {
    Shape shape_dims;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        shape_dims.push_back(static_cast<int>(shape[dim_idx]));
    }
    return pool_array(full(shape_dims, value));
}

void* mlx_lazy_triu(const void* input, int k) {
    const auto& input_arr = *static_cast<const array*>(input);
    return pool_array(triu(input_arr, k));
}

// ============================================================================
// Embedding Operations
// ============================================================================

void* mlx_lazy_embedding(const void* weights, const uint32_t* indices, size_t n_indices) {
    const auto& weights_arr = *static_cast<const array*>(weights);
    array idx_arr(reinterpret_cast<const int32_t*>(indices),
                  {1, static_cast<int>(n_indices)}, int32);
    return pool_array(take(weights_arr, idx_arr, 0));
}

void* mlx_lazy_embedding_from_array(const void* weights, const void* indices_handle) {
    const auto& weights_arr = *static_cast<const array*>(weights);
    const auto& indices = *static_cast<const array*>(indices_handle);
    auto idx_arr = astype(reshape(indices, {1, -1}), int32);
    return pool_array(take(weights_arr, idx_arr, 0));
}

// ============================================================================
// Quantization Operations
// ============================================================================

void* mlx_lazy_dequantize(
    const void* weights, const void* scales, const void* biases,
    size_t group_size, size_t bits
) {
    const auto& weights_arr = *static_cast<const array*>(weights);
    const auto& scales_arr = *static_cast<const array*>(scales);
    const auto& biases_arr = *static_cast<const array*>(biases);
    return pool_array(dequantize(
        weights_arr,
        scales_arr,
        biases_arr,
        static_cast<int>(group_size),
        static_cast<int>(bits),
        "affine"
    ));
}

// ============================================================================
// Compound Operations (fused for convenience)
// ============================================================================

void* mlx_lazy_reshape_transpose(
    const void* input,
    const size_t* reshape_dims, size_t reshape_ndim,
    const size_t* transpose_axes, size_t transpose_ndim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    Shape reshape_shape;
    for (size_t dim_idx = 0; dim_idx < reshape_ndim; dim_idx++) {
        reshape_shape.push_back(static_cast<int>(reshape_dims[dim_idx]));
    }
    std::vector<int> transpose_axes_vec;
    for (size_t dim_idx = 0; dim_idx < transpose_ndim; dim_idx++) {
        transpose_axes_vec.push_back(static_cast<int>(transpose_axes[dim_idx]));
    }
    auto reshaped = reshape(input_arr, reshape_shape);
    return pool_array(transpose(reshaped, transpose_axes_vec));
}

void* mlx_lazy_transpose_reshape(
    const void* input,
    const size_t* transpose_axes, size_t transpose_ndim,
    const size_t* reshape_dims, size_t reshape_ndim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    std::vector<int> transpose_axes_vec;
    for (size_t dim_idx = 0; dim_idx < transpose_ndim; dim_idx++) {
        transpose_axes_vec.push_back(static_cast<int>(transpose_axes[dim_idx]));
    }
    Shape reshape_shape;
    for (size_t dim_idx = 0; dim_idx < reshape_ndim; dim_idx++) {
        reshape_shape.push_back(static_cast<int>(reshape_dims[dim_idx]));
    }
    auto transposed = transpose(input_arr, transpose_axes_vec);
    return pool_array(reshape(transposed, reshape_shape));
}

} // extern "C"
