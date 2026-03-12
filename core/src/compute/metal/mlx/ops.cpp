// MLX Bridge - Basic Lazy Operations
//
// Provides C-callable wrappers for MLX array operations.
// All operations are "lazy" - they build a computation graph without executing.
// Call mlx_eval() to execute the graph.

#include "compute_common.h"
#include "attention_utils.h"

thread_local bool g_count_ops_enabled = false;
thread_local size_t g_count_ops_value = 0;

static constexpr size_t kQuantParamCastCacheMaxEntries = 4096;

struct QuantParamCastCacheKey {
    const void* handle = nullptr;
    int target_dtype = 0;

    bool operator==(const QuantParamCastCacheKey& other) const {
        return handle == other.handle &&
               target_dtype == other.target_dtype;
    }
};

struct QuantParamCastCacheKeyHash {
    size_t operator()(const QuantParamCastCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.handle);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash>& quant_param_cast_cache_store() {
    return tls_never_destroyed<std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash>>();
}

static inline void quant_param_cast_cache_clear() {
    quant_param_cast_cache_store().clear();
}

static array make_owned_embedding_index_array(const uint32_t* indices, size_t n_indices) {
    static constexpr size_t kIndexAlignment = 16 * 1024;
    const size_t byte_count = std::max(n_indices * sizeof(int32_t), sizeof(int32_t));
    void* aligned_ptr = nullptr;
    if (posix_memalign(&aligned_ptr, kIndexAlignment, byte_count) != 0 || aligned_ptr == nullptr) {
        throw std::bad_alloc();
    }

    auto* copied = static_cast<int32_t*>(aligned_ptr);
    for (size_t i = 0; i < n_indices; ++i) {
        copied[i] = static_cast<int32_t>(indices[i]);
    }

    auto owner = std::shared_ptr<void>(aligned_ptr, [](void* ptr) {
        std::free(ptr);
    });
    auto deleter = [owner](void*) {
        // ownership retained by closure capture
    };
    return array(copied, {1, static_cast<int>(n_indices)}, int32, deleter);
}

static array cast_quant_param_cached(
    const void* handle,
    const array& param,
    Dtype target_dtype
) {
    if (param.dtype() == target_dtype) {
        return param;
    }
    if (handle == nullptr) {
        return astype(param, target_dtype);
    }

    auto& cache = quant_param_cast_cache_store();
    if (cache.size() >= kQuantParamCastCacheMaxEntries) {
        cache.clear();
    }

    QuantParamCastCacheKey key;
    key.handle = handle;
    key.target_dtype = static_cast<int>(target_dtype.val());
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    array casted = astype(param, target_dtype);
    // Materialize once so repeated decode/prefill steps reuse concrete casts.
    eval(casted);
    auto [inserted, _] = cache.emplace(key, std::move(casted));
    return inserted->second;
}

extern "C" {

void mlx_start_counting() {
    g_count_ops_value = 0;
    g_count_ops_enabled = true;
}

size_t mlx_stop_counting() {
    g_count_ops_enabled = false;
    return g_count_ops_value;
}

void mlx_gqa_index_cache_clear() {
    ScopedAutoreleasePool pool;
    gqa_index_cache_clear();
}

size_t mlx_gqa_index_cache_size() {
    return gqa_index_cache_size();
}

size_t mlx_gqa_index_cache_max_entries() {
    return gqa_index_cache_max_entries();
}

void mlx_gqa_index_cache_touch(size_t q_heads, size_t kv_heads) {
    ScopedAutoreleasePool pool;
    (void)gqa_cached_gather_indices(static_cast<int>(q_heads), static_cast<int>(kv_heads));
}

void mlx_quant_param_cast_cache_clear() {
    ScopedAutoreleasePool pool;
    quant_param_cast_cache_clear();
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

void* mlx_lazy_add(const void* a, const void* b) {
    mlx_count_op();
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(lhs + rhs);
}

void* mlx_lazy_multiply(const void* a, const void* b) {
    mlx_count_op();
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    return pool_array(lhs * rhs);
}

void* mlx_lazy_multiply_scalar(const void* a, float scalar) {
    mlx_count_op();
    const auto& input = *static_cast<const array*>(a);
    return pool_array(input * scalar);
}

// ============================================================================
// Matrix Operations
// ============================================================================

static array matmul_lastdim_impl(const array& input, const array& rhs) {
    if (rhs.ndim() != 2) {
        throw std::invalid_argument("[matmul_lastdim] rhs must be rank-2");
    }
    const array input_for_matmul = (input.dtype() == rhs.dtype()) ? input : astype(input, rhs.dtype());

    if (input_for_matmul.ndim() < 2) {
        throw std::invalid_argument("[matmul_lastdim] input rank must be >=2");
    }
    const int in_features = input_for_matmul.shape(input_for_matmul.ndim() - 1);
    if (rhs.shape(0) != in_features) {
        throw std::invalid_argument("[matmul_lastdim] rhs/input feature mismatch");
    }
    if (input_for_matmul.ndim() == 2) {
        return matmul(input_for_matmul, rhs);
    }

    Shape out_shape = input_for_matmul.shape();
    out_shape.back() = rhs.shape(1);
    int rows = 1;
    for (int axis = 0; axis < input_for_matmul.ndim() - 1; ++axis) {
        rows *= input_for_matmul.shape(axis);
    }
    // Decode hot path: [1, 1, D] @ [D, O]. Prefer matvec-style dispatch
    // over generic GEMM flattening to reduce single-token launch overhead.
    if (rows == 1) {
        auto input_vec = reshape(input_for_matmul, {in_features});
        auto out_vec = matmul(input_vec, rhs);
        return reshape(out_vec, out_shape);
    }
    auto input_2d = reshape(input_for_matmul, {rows, in_features});
    auto out_2d = matmul(input_2d, rhs);
    return reshape(out_2d, out_shape);
}

static array quantized_matmul_lastdim_impl(
    const array& input,
    const array& weights,
    const array& scales,
    const array& biases,
    bool transpose_weights,
    int group_size,
    int bits
) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("[quantized_matmul_lastdim] input rank must be >=2");
    }

    if (input.ndim() == 2) {
        return quantized_matmul(
            input,
            weights,
            scales,
            biases,
            transpose_weights,
            group_size,
            bits,
            "affine"
        );
    }

    const int in_features = input.shape(input.ndim() - 1);
    int rows = 1;
    for (int axis = 0; axis < input.ndim() - 1; ++axis) {
        rows *= input.shape(axis);
    }

    Shape out_shape = input.shape();
    // For talu quantized weights, transpose_weights is always true and N is dim 0.
    out_shape.back() = transpose_weights ? weights.shape(0) : weights.shape(1);

    auto input_2d = reshape(input, {rows, in_features});
    auto out_2d = quantized_matmul(
        input_2d,
        weights,
        scales,
        biases,
        transpose_weights,
        group_size,
        bits,
        "affine"
    );
    return reshape(out_2d, out_shape);
}

void* mlx_lazy_matmul(const void* a, const void* b) {
    mlx_count_op();
    const auto& lhs = *static_cast<const array*>(a);
    const auto& rhs = *static_cast<const array*>(b);
    if (rhs.ndim() == 2 && lhs.ndim() >= 2) {
        return pool_array(matmul_lastdim_impl(lhs, rhs));
    }
    return pool_array(matmul(lhs, rhs));
}

void* mlx_lazy_quantized_matmul(
    const void* input, const void* weights, const void* scales, const void* biases,
    size_t group_size, size_t bits, bool transpose_weights
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& weights_arr = *static_cast<const array*>(weights);
    const auto& scales_arr = *static_cast<const array*>(scales);
    const auto& biases_arr = *static_cast<const array*>(biases);
    // Keep quant params in their stored dtype and cast activations instead.
    // This avoids large first-use scale/bias casts on prompt prefill.
    const Dtype param_dtype = scales_arr.dtype();
    if (input_arr.dtype() == param_dtype && biases_arr.dtype() == param_dtype) {
        return pool_array(quantized_matmul_lastdim_impl(
            input_arr,
            weights_arr,
            scales_arr,
            biases_arr,
            transpose_weights,
            static_cast<int>(group_size),
            static_cast<int>(bits)
        ));
    }
    const array input_cast = (input_arr.dtype() == param_dtype) ? input_arr : astype(input_arr, param_dtype);
    const array scales_cast = cast_quant_param_cached(scales, scales_arr, param_dtype);
    const array biases_cast = cast_quant_param_cached(biases, biases_arr, param_dtype);
    return pool_array(quantized_matmul_lastdim_impl(
        input_cast,
        weights_arr,
        scales_cast,
        biases_cast,
        transpose_weights,
        static_cast<int>(group_size),
        static_cast<int>(bits)
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
        return make_owned_array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1])})));
    } else if (ndim == 3) {
        return make_owned_array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2])})));
    } else if (ndim == 4) {
        return make_owned_array(contiguous(reshape(input_arr,
            {static_cast<int>(shape[0]), static_cast<int>(shape[1]),
             static_cast<int>(shape[2]), static_cast<int>(shape[3])})));
    }
    Shape shape_dims;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        shape_dims.push_back(static_cast<int>(shape[dim_idx]));
    }
    return make_owned_array(contiguous(reshape(input_arr, shape_dims)));
}

// Persistent transpose - heap-allocated, survives pool resets
void* mlx_persistent_transpose(const void* input, const size_t* axes, size_t ndim) {
    const auto& input_arr = *static_cast<const array*>(input);
    if (ndim == 2) {
        return make_owned_array(contiguous(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1])})));
    } else if (ndim == 3) {
        return make_owned_array(contiguous(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1]),
             static_cast<int>(axes[2])})));
    } else if (ndim == 4) {
        return make_owned_array(contiguous(transpose(input_arr,
            {static_cast<int>(axes[0]), static_cast<int>(axes[1]),
             static_cast<int>(axes[2]), static_cast<int>(axes[3])})));
    }
    std::vector<int> axes_vec;
    for (size_t dim_idx = 0; dim_idx < ndim; dim_idx++) {
        axes_vec.push_back(static_cast<int>(axes[dim_idx]));
    }
    return make_owned_array(contiguous(transpose(input_arr, axes_vec)));
}

void* mlx_persistent_cast_f16(const void* input) {
    const auto& input_arr = *static_cast<const array*>(input);
    auto casted = astype(input_arr, float16);
    // Materialize once so decode does not re-lower weight casts per token.
    eval(casted);
    return make_owned_array(std::move(casted));
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
    return make_owned_array(contiguous(slice(input_arr, start, stop)));
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

void* mlx_lazy_rms_norm(const void* input, const void* weight, float eps) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& weight_arr = *static_cast<const array*>(weight);
    const int width = input_arr.shape(input_arr.ndim() - 1);
    const array norm_weight = canonicalize_rms_norm_weight(weight_arr, width, "rms_norm");
    return pool_array(fast::rms_norm(
        input_arr,
        norm_weight,
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
    const auto& q_arr = *static_cast<const array*>(q);
    // Decode hot path: when query length is 1 there are no future tokens to mask,
    // so causal masking adds overhead without changing semantics.
    bool use_causal = causal;
    if (causal && q_arr.ndim() >= 2) {
        const int q_seq_axis = q_arr.ndim() - 2;
        if (q_arr.shape(q_seq_axis) <= 1) {
            use_causal = false;
        }
    }
    return pool_array(fast::scaled_dot_product_attention(
        q_arr,
        *static_cast<const array*>(k),
        *static_cast<const array*>(v),
        scale,
        use_causal ? "causal" : ""
    ));
}

// ============================================================================
// Reduction Operations
// ============================================================================

void* mlx_lazy_argmax(const void* handle, int axis) {
    const auto& input_arr = *static_cast<const array*>(handle);
    return pool_array(argmax(input_arr, axis));
}

void* mlx_lazy_argpartition(const void* handle, int kth, int axis) {
    const auto& input_arr = *static_cast<const array*>(handle);
    return pool_array(argpartition(input_arr, kth, axis));
}

void* mlx_lazy_take_along_axis(const void* input, const void* indices, int axis) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& indices_arr = *static_cast<const array*>(indices);
    return pool_array(take_along_axis(input_arr, indices_arr, axis));
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
    array idx_arr = make_owned_embedding_index_array(indices, n_indices);
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
