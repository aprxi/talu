// MLX Bridge - Basic Lazy Operations
//
// Provides C-callable wrappers for MLX array operations.
// All operations are "lazy" - they build a computation graph without executing.
// Call mlx_eval() to execute the graph.

#include "compute_common.h"
#include "model_state.h"

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
// ShortConv Operations
// ============================================================================

void* mlx_lazy_shortconv_mixer_bf16(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_proj,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_arr = *static_cast<const array*>(in_proj);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto& out_proj_arr = *static_cast<const array*>(out_proj);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);

    // in_proj is stored as [3*conv_dim, d_model], so transpose for [d_model, 3*conv_dim].
    array bcx = matmul(input_arr, transpose(in_proj_arr));
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    if (conv_kernel.ndim() == 3) {
        // [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
        conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
    }
    // [conv_dim, d_conv] -> [d_conv, conv_dim] for contiguous time-major access.
    array conv_kernel_t = astype(transpose(conv_kernel), float32);
    const array conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        conv_bias_row = reshape(astype(*conv_bias_arr, float32), {1, conv_dim_i});
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != conv_dim_i;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
    }

    const array out_proj_t = transpose(out_proj_arr);
    std::vector<array> token_outputs;
    token_outputs.reserve(seq_len);

    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        const array bx_t = slice(bx, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i});

        if (d_conv_i > 1) {
            const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
            conv_state = concatenate({state_tail, bx_t}, 1);
        } else {
            conv_state = bx_t;
        }

        // Vectorized depthwise convolution over history window:
        // [1, d_conv, conv_dim] * [1, d_conv, conv_dim] -> sum(axis=1) => [1, conv_dim].
        array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
        if (conv_bias_row) {
            conv_t = conv_t + *conv_bias_row;
        }

        const array c_t = reshape(slice(c_gate, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i}), {1, conv_dim_i});
        const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
        token_outputs.push_back(matmul(gated, out_proj_t));
    }

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    if (token_outputs.size() == 1) {
        return pool_array(token_outputs[0]);
    }
    return pool_array(concatenate(token_outputs, 1));
}

void* mlx_lazy_shortconv_mixer_quantized(
    const void* input,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    size_t group_size,
    size_t bits,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_w_arr = *static_cast<const array*>(in_w);
    const auto& in_s_arr = *static_cast<const array*>(in_s);
    const auto& in_b_arr = *static_cast<const array*>(in_b);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);
    const auto& out_w_arr = *static_cast<const array*>(out_w);
    const auto& out_s_arr = *static_cast<const array*>(out_s);
    const auto& out_b_arr = *static_cast<const array*>(out_b);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    array bcx = quantized_matmul(
        input_arr,
        in_w_arr,
        in_s_arr,
        in_b_arr,
        true,
        group_size_i,
        bits_i,
        "affine"
    );
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    if (conv_kernel.ndim() == 3) {
        conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
    }
    array conv_kernel_t = astype(transpose(conv_kernel), float32);
    const array conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        conv_bias_row = reshape(astype(*conv_bias_arr, float32), {1, conv_dim_i});
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != conv_dim_i;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
    }

    std::vector<array> token_outputs;
    token_outputs.reserve(seq_len);

    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        const array bx_t = slice(bx, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i});

        if (d_conv_i > 1) {
            const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
            conv_state = concatenate({state_tail, bx_t}, 1);
        } else {
            conv_state = bx_t;
        }

        // Vectorized depthwise convolution over history window:
        // [1, d_conv, conv_dim] * [1, d_conv, conv_dim] -> sum(axis=1) => [1, conv_dim].
        array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
        if (conv_bias_row) {
            conv_t = conv_t + *conv_bias_row;
        }

        const array c_t = reshape(slice(c_gate, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i}), {1, conv_dim_i});
        const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
        token_outputs.push_back(quantized_matmul(
            gated,
            out_w_arr,
            out_s_arr,
            out_b_arr,
            true,
            group_size_i,
            bits_i,
            "affine"
        ));
    }

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    if (token_outputs.size() == 1) {
        return pool_array(token_outputs[0]);
    }
    return pool_array(concatenate(token_outputs, 1));
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
