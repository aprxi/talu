static array apply_runtime_rope_to_tensor(
    const array& x, // [B, H, L, D]
    const array& cos_rows, // [L, rope_dim]
    const array& sin_rows, // [L, rope_dim]
    int seq_len,
    int head_dim,
    int rope_dim
) {
    const int batch = x.shape(0);
    const int heads = x.shape(1);
    const int half = rope_dim / 2;

    auto x_rot = slice(x, {0, 0, 0, 0}, {batch, heads, seq_len, rope_dim});
    auto cos_view = reshape(cos_rows, {1, 1, seq_len, rope_dim});
    auto sin_view = reshape(sin_rows, {1, 1, seq_len, rope_dim});

    auto x_lo = slice(x_rot, {0, 0, 0, 0}, {batch, heads, seq_len, half});
    auto x_hi = slice(x_rot, {0, 0, 0, half}, {batch, heads, seq_len, rope_dim});
    auto cos_lo = slice(cos_view, {0, 0, 0, 0}, {1, 1, seq_len, half});
    auto cos_hi = slice(cos_view, {0, 0, 0, half}, {1, 1, seq_len, rope_dim});
    auto sin_lo = slice(sin_view, {0, 0, 0, 0}, {1, 1, seq_len, half});
    auto sin_hi = slice(sin_view, {0, 0, 0, half}, {1, 1, seq_len, rope_dim});

    auto rotated_lo = x_lo * cos_lo - x_hi * sin_lo;
    auto rotated_hi = x_hi * cos_hi + x_lo * sin_hi;
    auto rotated = concatenate({rotated_lo, rotated_hi}, 3);

    if (rope_dim == head_dim) {
        return rotated;
    }
    auto tail = slice(x, {0, 0, 0, rope_dim}, {batch, heads, seq_len, head_dim});
    return concatenate({rotated, tail}, 3);
}

// Infer matmul RHS orientation from shape so fused dense ops remain robust across
// model families and loader conventions.
static array orient_matmul_rhs(
    const array& weight,
    int in_features,
    std::optional<int> out_features = std::nullopt,
    bool transpose_when_ambiguous = true
) {
    if (weight.ndim() != 2) {
        throw std::invalid_argument("[dense] expected 2D weight for matmul rhs");
    }
    const int rows = weight.shape(0);
    const int cols = weight.shape(1);

    const bool direct = rows == in_features && (!out_features.has_value() || cols == out_features.value());
    const bool transposed = cols == in_features && (!out_features.has_value() || rows == out_features.value());

    if (direct && !transposed) {
        return weight;
    }
    if (transposed && !direct) {
        return transpose(weight);
    }
    if (direct && transposed) {
        return transpose_when_ambiguous ? transpose(weight) : weight;
    }
    throw std::invalid_argument("[dense] weight orientation incompatible with matmul shape");
}

static constexpr size_t kOrientedRhsCacheMaxEntries = 2048;
static constexpr size_t kStateSpaceConvKernelCacheMaxEntries = 512;
static constexpr size_t kCausalConvKernelCacheMaxEntries = 512;
static constexpr size_t kCausalConvBiasCacheMaxEntries = 512;
static constexpr size_t kBiasCastCacheMaxEntries = 1024;

struct OrientedRhsCacheKey {
    const void* weight_handle = nullptr;
    int in_features = 0;
    int out_features = -1;
    bool transpose_when_ambiguous = true;

    bool operator==(const OrientedRhsCacheKey& other) const {
        return weight_handle == other.weight_handle &&
               in_features == other.in_features &&
               out_features == other.out_features &&
               transpose_when_ambiguous == other.transpose_when_ambiguous;
    }
};

struct OrientedRhsCacheKeyHash {
    size_t operator()(const OrientedRhsCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_handle);
        h ^= std::hash<int>{}(key.in_features) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.out_features) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(key.transpose_when_ambiguous) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<OrientedRhsCacheKey, array, OrientedRhsCacheKeyHash>& oriented_rhs_cache_store() {
    static thread_local std::unordered_map<OrientedRhsCacheKey, array, OrientedRhsCacheKeyHash> cache;
    return cache;
}

static inline void oriented_rhs_cache_clear() {
    oriented_rhs_cache_store().clear();
}

// Defined in fused_ops_attention.inc and fused_ops_ffn_expert_mix.inc.
static inline void attention_fused_qkv_cache_clear();
static inline void ffn_fused_gate_up_cache_clear();

struct StateSpaceConvKernelCacheKey {
    const void* weight_handle = nullptr;
    int d_conv = 0;
    int xbc_len = 0;

    bool operator==(const StateSpaceConvKernelCacheKey& other) const {
        return weight_handle == other.weight_handle &&
               d_conv == other.d_conv &&
               xbc_len == other.xbc_len;
    }
};

struct StateSpaceConvKernelCacheKeyHash {
    size_t operator()(const StateSpaceConvKernelCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_handle);
        h ^= std::hash<int>{}(key.d_conv) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.xbc_len) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<StateSpaceConvKernelCacheKey, array, StateSpaceConvKernelCacheKeyHash>& state_space_conv_kernel_cache_store() {
    static thread_local std::unordered_map<StateSpaceConvKernelCacheKey, array, StateSpaceConvKernelCacheKeyHash> cache;
    return cache;
}

static inline void state_space_conv_kernel_cache_clear() {
    state_space_conv_kernel_cache_store().clear();
}

struct CausalConvKernelCacheKey {
    const void* weight_handle = nullptr;
    int d_conv = 0;
    int conv_dim = 0;

    bool operator==(const CausalConvKernelCacheKey& other) const {
        return weight_handle == other.weight_handle &&
               d_conv == other.d_conv &&
               conv_dim == other.conv_dim;
    }
};

struct CausalConvKernelCacheKeyHash {
    size_t operator()(const CausalConvKernelCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_handle);
        h ^= std::hash<int>{}(key.d_conv) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.conv_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<CausalConvKernelCacheKey, array, CausalConvKernelCacheKeyHash>& causal_conv_kernel_cache_store() {
    static thread_local std::unordered_map<CausalConvKernelCacheKey, array, CausalConvKernelCacheKeyHash> cache;
    return cache;
}

static inline void causal_conv_kernel_cache_clear() {
    causal_conv_kernel_cache_store().clear();
}

struct CausalConvBiasCacheKey {
    const void* bias_handle = nullptr;
    int conv_dim = 0;

    bool operator==(const CausalConvBiasCacheKey& other) const {
        return bias_handle == other.bias_handle &&
               conv_dim == other.conv_dim;
    }
};

struct CausalConvBiasCacheKeyHash {
    size_t operator()(const CausalConvBiasCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.bias_handle);
        h ^= std::hash<int>{}(key.conv_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<CausalConvBiasCacheKey, array, CausalConvBiasCacheKeyHash>& causal_conv_bias_cache_store() {
    static thread_local std::unordered_map<CausalConvBiasCacheKey, array, CausalConvBiasCacheKeyHash> cache;
    return cache;
}

static inline void causal_conv_bias_cache_clear() {
    causal_conv_bias_cache_store().clear();
}

struct BiasCastCacheKey {
    const void* bias_handle = nullptr;
    int target_dtype = 0;

    bool operator==(const BiasCastCacheKey& other) const {
        return bias_handle == other.bias_handle &&
               target_dtype == other.target_dtype;
    }
};

struct BiasCastCacheKeyHash {
    size_t operator()(const BiasCastCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.bias_handle);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<BiasCastCacheKey, array, BiasCastCacheKeyHash>& bias_cast_cache_store() {
    static thread_local std::unordered_map<BiasCastCacheKey, array, BiasCastCacheKeyHash> cache;
    return cache;
}

static inline void bias_cast_cache_clear() {
    bias_cast_cache_store().clear();
}

struct QuantParamCastCacheKey {
    const void* param_handle = nullptr;
    int target_dtype = 0;

    bool operator==(const QuantParamCastCacheKey& other) const {
        return param_handle == other.param_handle &&
               target_dtype == other.target_dtype;
    }
};

struct QuantParamCastCacheKeyHash {
    size_t operator()(const QuantParamCastCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.param_handle);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static constexpr size_t kQuantParamCastCacheMaxEntries = 4096;

static inline std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash>& quant_param_cast_cache_store() {
    static thread_local std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash> cache;
    return cache;
}

static inline void quant_param_cast_cache_clear() {
    quant_param_cast_cache_store().clear();
}

static inline array orient_matmul_rhs_cached(
    const void* weight_handle,
    const array& weight,
    int in_features,
    std::optional<int> out_features = std::nullopt,
    bool transpose_when_ambiguous = true
) {
    if (weight_handle == nullptr) {
        return orient_matmul_rhs(weight, in_features, out_features, transpose_when_ambiguous);
    }

    auto& cache = oriented_rhs_cache_store();
    if (cache.size() >= kOrientedRhsCacheMaxEntries) {
        cache.clear();
    }

    OrientedRhsCacheKey key;
    key.weight_handle = weight_handle;
    key.in_features = in_features;
    key.out_features = out_features.has_value() ? out_features.value() : -1;
    key.transpose_when_ambiguous = transpose_when_ambiguous;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    if (weight.ndim() != 2) {
        throw std::invalid_argument("[dense] expected 2D weight for matmul rhs");
    }
    const int rows = weight.shape(0);
    const int cols = weight.shape(1);
    const bool direct = rows == in_features && (!out_features.has_value() || cols == out_features.value());
    const bool transposed = cols == in_features && (!out_features.has_value() || rows == out_features.value());

    array rhs(0.0f, weight.dtype());
    if (direct && !transposed) {
        rhs = weight;
    } else if (transposed && !direct) {
        rhs = contiguous(transpose(weight));
    } else if (direct && transposed) {
        rhs = transpose_when_ambiguous ? contiguous(transpose(weight)) : weight;
    } else {
        throw std::invalid_argument("[dense] weight orientation incompatible with matmul shape");
    }

    // Materialize once so cached transforms do not get re-lowered per decode token.
    eval(rhs);
    auto [inserted_it, _] = cache.emplace(key, rhs);
    return inserted_it->second;
}

void mlx_weight_transform_cache_clear() {
    oriented_rhs_cache_clear();
    state_space_conv_kernel_cache_clear();
    causal_conv_kernel_cache_clear();
    causal_conv_bias_cache_clear();
    bias_cast_cache_clear();
    quant_param_cast_cache_clear();
    attention_fused_qkv_cache_clear();
    ffn_fused_gate_up_cache_clear();
}

static array gelu_approx(const array& x) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    const auto x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + kCoeff * x3)));
}

static inline array ensure_float32(const array& arr) {
    // FP32 allowlist: recurrent state-space SSM/conv state updates are numerically
    // sensitive across long contexts; keep state math in FP32.
    return arr.dtype() == float32 ? arr : astype(arr, float32);
}

static array layer_norm_last_dim(
    const array& x,
    const array& weight,
    const array& bias,
    float eps
) {
    const auto x_mean = mean(x, -1, true);
    const auto centered = x - x_mean;
    const auto variance = mean(centered * centered, -1, true);
    const auto inv_std = rsqrt(variance + eps);
    const auto normed = centered * inv_std;
    return normed * weight + bias;
}

static array matmul_lastdim(
    const array& input,
    const array& rhs
) {
    if (rhs.ndim() != 2) {
        throw std::invalid_argument("[dense] matmul rhs must be rank-2");
    }
    const array input_for_matmul = (input.dtype() == rhs.dtype()) ? input : astype(input, rhs.dtype());

    if (input_for_matmul.ndim() < 2) {
        throw std::invalid_argument("[dense] matmul input rank must be >=2");
    }

    const int in_features = input_for_matmul.shape(input_for_matmul.ndim() - 1);
    if (rhs.shape(0) != in_features) {
        throw std::invalid_argument("[dense] matmul rhs/input feature mismatch");
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

    // Decode hot path: [1, 1, D] @ [D, O]. Prefer matvec dispatch when the
    // flattened row count is one to reduce per-token launch overhead.
    if (rows == 1) {
        auto input_vec = reshape(input_for_matmul, {in_features});
        auto out_vec = matmul(input_vec, rhs);
        return reshape(out_vec, out_shape);
    }

    auto input_2d = reshape(input_for_matmul, {rows, in_features});
    auto out_2d = matmul(input_2d, rhs);
    return reshape(out_2d, out_shape);
}

static array quantized_matmul_lastdim(
    const array& input,
    const array& weights,
    const array& scales,
    const array& biases,
    int group_size,
    int bits
) {
    if (input.ndim() < 2) {
        throw std::invalid_argument("[quantized] matmul input rank must be >=2");
    }
    if (input.ndim() == 2) {
        return quantized_matmul(
            input,
            weights,
            scales,
            biases,
            true,
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
    out_shape.back() = weights.shape(0);

    auto input_2d = reshape(input, {rows, in_features});
    auto out_2d = quantized_matmul(
        input_2d,
        weights,
        scales,
        biases,
        true,
        group_size,
        bits,
        "affine"
    );
    return reshape(out_2d, out_shape);
}

static array linear_bf16_with_bias(
    const array& input,
    const array& weight,
    const array& bias,
    std::optional<int> out_features = std::nullopt,
    const void* weight_handle = nullptr
) {
    const int in_features = input.shape(input.ndim() - 1);
    const auto rhs = orient_matmul_rhs_cached(weight_handle, weight, in_features, out_features, true);
    return matmul_lastdim(input, rhs) + bias;
}

static bool can_use_runtime_rope(
    const array* runtime_rope_cos,
    const array* runtime_rope_sin,
    int seq_start,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (runtime_rope_cos == nullptr || runtime_rope_sin == nullptr) return false;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return false;
    const int seq_stop = seq_start + seq_len;
    return runtime_rope_cos->ndim() == 2 and runtime_rope_sin->ndim() == 2 and
        runtime_rope_cos->shape(0) >= seq_stop and runtime_rope_sin->shape(0) >= seq_stop and
        runtime_rope_cos->shape(1) >= rope_dim and runtime_rope_sin->shape(1) >= rope_dim;
}

static array apply_optional_runtime_rope(
    const array& x, // [B, H, L, D]
    const array* runtime_rope_cos,
    const array* runtime_rope_sin,
    int seq_start,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (!can_use_runtime_rope(runtime_rope_cos, runtime_rope_sin, seq_start, seq_len, head_dim, rope_dim)) {
        return x;
    }
    auto cos_rows = slice(*runtime_rope_cos, {seq_start, 0}, {seq_start + seq_len, rope_dim});
    auto sin_rows = slice(*runtime_rope_sin, {seq_start, 0}, {seq_start + seq_len, rope_dim});
    return apply_runtime_rope_to_tensor(x, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
}

static array dense_linear_no_bias(
    const array& input,
    const array& weight,
    std::optional<int> out_features = std::nullopt,
    const void* weight_handle = nullptr
) {
    const int in_features = input.shape(input.ndim() - 1);
    const auto rhs = orient_matmul_rhs_cached(weight_handle, weight, in_features, out_features, true);
    return matmul_lastdim(input, rhs);
}

static array quantized_linear_no_bias(
    const array& input,
    const array& weights,
    const array& scales,
    const array& biases,
    int group_size,
    int bits,
    const void* scales_handle = nullptr,
    const void* biases_handle = nullptr
) {
    const Dtype param_dtype = scales.dtype();
    const array input_cast = (input.dtype() == param_dtype) ? input : astype(input, param_dtype);
    const auto cast_quant_param = [](const void* handle, const array& param, Dtype target_dtype) -> array {
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
        key.param_handle = handle;
        key.target_dtype = static_cast<int>(target_dtype.val());
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }

        array casted = astype(param, target_dtype);
        // Materialize once so repeated decode/prefill steps reuse casted params.
        eval(casted);
        auto [inserted, _] = cache.emplace(key, std::move(casted));
        return inserted->second;
    };
    const array scales_cast = cast_quant_param(scales_handle, scales, param_dtype);
    const array biases_cast = cast_quant_param(biases_handle, biases, param_dtype);
    return quantized_matmul_lastdim(
        input_cast,
        weights,
        scales_cast,
        biases_cast,
        group_size,
        bits
    );
}

static array resolve_state_space_conv_kernel_uncached(
    const array& conv_weight,
    int d_conv,
    int xbc_len
) {
    array kernel = conv_weight;
    if (kernel.ndim() == 3 &&
        kernel.shape(0) == 1 &&
        kernel.shape(1) == d_conv &&
        kernel.shape(2) == xbc_len) {
        // Precomputed setup-time layout [1, d_conv, xbc_len].
        return ensure_float32(kernel);
    }
    if (kernel.ndim() == 3) {
        // [xbc_len, 1, d_conv] -> [xbc_len, d_conv]
        kernel = reshape(kernel, {kernel.shape(0), kernel.shape(2)});
    }
    if (kernel.ndim() != 2) {
        throw std::invalid_argument("[state_space] conv1d_weight must be rank-2 or rank-3");
    }

    if (kernel.shape(0) == xbc_len and kernel.shape(1) == d_conv) {
        kernel = contiguous(transpose(kernel));
    } else if (!(kernel.shape(0) == d_conv and kernel.shape(1) == xbc_len)) {
        throw std::invalid_argument("[state_space] conv1d_weight shape incompatible with xbc_len/d_conv");
    }

    kernel = ensure_float32(kernel);
    return reshape(kernel, {1, d_conv, xbc_len});
}

static array resolve_state_space_conv_kernel(
    const void* conv_weight_handle,
    const array& conv_weight,
    int d_conv,
    int xbc_len
) {
    if (conv_weight_handle == nullptr) {
        return resolve_state_space_conv_kernel_uncached(conv_weight, d_conv, xbc_len);
    }

    auto& cache = state_space_conv_kernel_cache_store();
    if (cache.size() >= kStateSpaceConvKernelCacheMaxEntries) {
        cache.clear();
    }

    StateSpaceConvKernelCacheKey key;
    key.weight_handle = conv_weight_handle;
    key.d_conv = d_conv;
    key.xbc_len = xbc_len;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    array kernel = resolve_state_space_conv_kernel_uncached(conv_weight, d_conv, xbc_len);
    eval(kernel);
    auto [inserted_it, _] = cache.emplace(key, kernel);
    return inserted_it->second;
}

static array resolve_causal_conv_kernel(
    const void* conv_weight_handle,
    const array& conv_weight,
    int d_conv,
    int conv_dim
) {
    const auto uncached = [&]() -> array {
        array kernel = conv_weight;
        if (kernel.ndim() == 3 &&
            kernel.shape(0) == 1 &&
            kernel.shape(1) == d_conv &&
            kernel.shape(2) == conv_dim) {
            return ensure_float32(kernel);
        }
        if (kernel.ndim() == 3) {
            // [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
            kernel = reshape(kernel, {kernel.shape(0), kernel.shape(2)});
        }
        if (kernel.ndim() != 2) {
            throw std::invalid_argument("[causal_conv] conv weight must be rank-2 or rank-3");
        }
        if (kernel.shape(0) == conv_dim and kernel.shape(1) == d_conv) {
            kernel = contiguous(transpose(kernel));
        } else if (!(kernel.shape(0) == d_conv and kernel.shape(1) == conv_dim)) {
            throw std::invalid_argument("[causal_conv] conv weight shape incompatible with d_conv/conv_dim");
        }
        kernel = ensure_float32(kernel);
        return reshape(kernel, {1, d_conv, conv_dim});
    };

    if (conv_weight_handle == nullptr) {
        return uncached();
    }

    auto& cache = causal_conv_kernel_cache_store();
    if (cache.size() >= kCausalConvKernelCacheMaxEntries) {
        cache.clear();
    }

    CausalConvKernelCacheKey key;
    key.weight_handle = conv_weight_handle;
    key.d_conv = d_conv;
    key.conv_dim = conv_dim;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    array kernel = uncached();
    eval(kernel);
    auto [inserted_it, _] = cache.emplace(key, kernel);
    return inserted_it->second;
}

static std::optional<array> resolve_causal_conv_bias_row(
    const void* conv_bias_handle,
    const array* conv_bias_arr,
    int conv_dim
) {
    if (conv_bias_arr == nullptr) {
        return std::nullopt;
    }

    const auto uncached = [&]() -> array {
        const array conv_bias_f32 = ensure_float32(*conv_bias_arr);
        if (conv_bias_f32.ndim() == 2 && conv_bias_f32.shape(0) == 1 && conv_bias_f32.shape(1) == conv_dim) {
            return conv_bias_f32;
        }
        return reshape(conv_bias_f32, {1, conv_dim});
    };

    if (conv_bias_handle == nullptr) {
        return std::optional<array>(uncached());
    }

    auto& cache = causal_conv_bias_cache_store();
    if (cache.size() >= kCausalConvBiasCacheMaxEntries) {
        cache.clear();
    }

    CausalConvBiasCacheKey key;
    key.bias_handle = conv_bias_handle;
    key.conv_dim = conv_dim;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return std::optional<array>(it->second);
    }

    array bias_row = uncached();
    eval(bias_row);
    auto [inserted_it, _] = cache.emplace(key, bias_row);
    return std::optional<array>(inserted_it->second);
}

static array cast_bias_cached(
    const void* bias_handle,
    const array& bias,
    Dtype target_dtype
) {
    if (bias.dtype() == target_dtype) {
        return bias;
    }
    if (bias_handle == nullptr) {
        return astype(bias, target_dtype);
    }

    auto& cache = bias_cast_cache_store();
    if (cache.size() >= kBiasCastCacheMaxEntries) {
        cache.clear();
    }

    BiasCastCacheKey key;
    key.bias_handle = bias_handle;
    key.target_dtype = static_cast<int>(target_dtype.val());
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    array casted = astype(bias, target_dtype);
    eval(casted);
    auto [inserted_it, _] = cache.emplace(key, casted);
    return inserted_it->second;
}

static array state_space_forward_core(
    const array& input, // [1, L, d_model]
    const array& ln1_weight,
    const std::function<array(const array&)>& in_proj_fn,
    const array& conv_kernel_broadcast, // [1, d_conv, xbc_len]
    const std::optional<array>& conv_bias_row, // [1, xbc_len]
    const array& a_log, // [n_heads]
    const array& d_skip, // [n_heads]
    const std::optional<array>& dt_bias_row, // [1, n_heads]
    const std::optional<array>& norm_weight, // [d_inner]
    const std::function<array(const array&)>& out_proj_fn,
    const std::optional<array>& ln2_weight,
    const std::function<array(const array&)>& gate_up_fn,
    const std::function<array(const array&)>& down_proj_fn,
    bool has_ffn,
    bool use_gelu,
    float residual_multiplier,
    float norm_eps,
    int d_state,
    int d_conv,
    int n_heads,
    int d_head,
    int n_groups,
    uint8_t gate_up_layout,
    array& conv_state,
    array& ssm_state
) {
    const int seq_len = input.shape(1);
    const int d_inner = n_heads * d_head;
    const int bc_len = n_groups * d_state;
    const int xbc_len = d_inner + 2 * bc_len;
    if (n_heads <= 0 or d_head <= 0 or d_state <= 0 or d_conv <= 0 or n_groups <= 0) {
        throw std::invalid_argument("[state_space] invalid config dims");
    }
    if ((n_heads % n_groups) != 0) {
        throw std::invalid_argument("[state_space] n_heads must be divisible by n_groups");
    }

    const array a_log_row = reshape(a_log, {1, n_heads, 1, 1});
    const array d_skip_row = reshape(d_skip, {1, n_heads, 1});
    const int heads_per_group = n_heads / n_groups;
    const int min_proj_dim = (2 * d_inner + 2 * bc_len + n_heads);

    // Hoist token-independent preprocessing out of the recurrent loop.
    const array norm_all = fast::rms_norm(input, ln1_weight, norm_eps);
    const array proj_all = in_proj_fn(norm_all);
    const int proj_dim = proj_all.shape(2);
    if (proj_dim < min_proj_dim) {
        throw std::invalid_argument("[state_space] in_proj output too small");
    }

    const array z_all = slice(proj_all, {0, 0, 0}, {1, seq_len, d_inner});
    const array xbc_all = ensure_float32(slice(proj_all, {0, 0, d_inner}, {1, seq_len, d_inner + xbc_len}));
    array dt_all = ensure_float32(slice(
        proj_all,
        {0, 0, d_inner + xbc_len},
        {1, seq_len, d_inner + xbc_len + n_heads}
    ));
    if (dt_bias_row) {
        dt_all = dt_all + reshape(*dt_bias_row, {1, 1, n_heads});
    }
    dt_all = log(1.0f + exp(dt_all));

    // Vectorized depthwise conv over [state, sequence] for xbc stream.
    const array conv_history = concatenate({conv_state, xbc_all}, 1);
    const array conv_weight = transpose(conv_kernel_broadcast, {2, 1, 0}); // [xbc_len, d_conv, 1]
    array conv_seq = conv1d(conv_history, conv_weight, 1, 0, 1, xbc_len);
    conv_seq = slice(conv_seq, {0, 1, 0}, {1, seq_len + 1, xbc_len});
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, xbc_len});
    }
    const array xbc_conv_all = conv_seq * sigmoid(conv_seq);
    conv_state = slice(conv_history, {0, seq_len, 0}, {1, seq_len + d_conv, xbc_len});

    const array x_conv_all = slice(xbc_conv_all, {0, 0, 0}, {1, seq_len, d_inner});
    array b_heads_all = reshape(
        slice(xbc_conv_all, {0, 0, d_inner}, {1, seq_len, d_inner + bc_len}),
        {seq_len, n_groups, d_state}
    );
    array c_heads_all = reshape(
        slice(xbc_conv_all, {0, 0, d_inner + bc_len}, {1, seq_len, d_inner + 2 * bc_len}),
        {seq_len, n_groups, d_state}
    );
    if (heads_per_group > 1) {
        b_heads_all = repeat(b_heads_all, heads_per_group, 1);
        c_heads_all = repeat(c_heads_all, heads_per_group, 1);
    }

    const array x_heads_all = reshape(x_conv_all, {seq_len, n_heads, d_head});
    const array dt4_all = reshape(dt_all, {seq_len, n_heads, 1, 1});
    const array dA_all = exp((-exp(a_log_row)) * dt4_all);
    const array x_term_all = dt4_all *
        reshape(x_heads_all, {seq_len, n_heads, d_head, 1}) *
        reshape(b_heads_all, {seq_len, n_heads, 1, d_state});

    // Solve s_t = dA_t * s_{t-1} + x_t with parallel prefix products/sums.
    // dA_t is strictly non-negative (exp of real), so clipping protects only
    // against denorm/underflow during long prefills.
    const array prefix = cumprod(dA_all, 0);
    const array prefix_safe = clip(prefix, array(1.0e-20f), std::nullopt);
    const array ssm_terms = x_term_all / prefix_safe;
    const array ssm_accum = cumsum(ssm_terms, 0);
    const array ssm_all = prefix * (ssm_state + ssm_accum);
    ssm_state = slice(ssm_all, {seq_len - 1, 0, 0, 0}, {seq_len, n_heads, d_head, d_state});

    array y_all = sum(ssm_all * reshape(c_heads_all, {seq_len, n_heads, 1, d_state}), 3);
    y_all = y_all + d_skip_row * x_heads_all;
    array ssm_out = reshape(y_all, {1, seq_len, d_inner});
    ssm_out = ssm_out * (z_all * sigmoid(z_all));
    if (norm_weight) {
        ssm_out = fast::rms_norm(ssm_out, *norm_weight, 1e-5f);
    }

    const Dtype model_dtype = input.dtype();
    // FP32 allowlist ends at recurrent-state math; projections stay in model
    // dtype to use bf16/f16 tensor-core kernels.
    const array ssm_out_proj_in = (ssm_out.dtype() == model_dtype) ? ssm_out : astype(ssm_out, model_dtype);
    const array mixer_out = out_proj_fn(ssm_out_proj_in);
    array hidden_1 = input + mixer_out * residual_multiplier;

    if (!has_ffn) {
        return hidden_1;
    }
    if (!ln2_weight.has_value()) {
        throw std::invalid_argument("[state_space] missing ln2 weight for FFN path");
    }

    const array norm2 = fast::rms_norm(hidden_1, *ln2_weight, norm_eps);
    const array norm2_proj_in = (norm2.dtype() == model_dtype) ? norm2 : astype(norm2, model_dtype);
    const array gate_up = gate_up_fn(norm2_proj_in);
    const int gate_up_dim = gate_up.shape(2);
    if ((gate_up_dim % 2) != 0) {
        throw std::invalid_argument("[state_space] fused gate_up must have even width");
    }
    const int d_ff = gate_up_dim / 2;

    const array mid = [&]() -> array {
        if (gate_up_layout == 1) {
            const array reshaped = reshape(gate_up, {1, seq_len, d_ff, 2});
            const array gate = reshape(
                slice(reshaped, {0, 0, 0, 0}, {1, seq_len, d_ff, 1}),
                {1, seq_len, d_ff}
            );
            const array up = reshape(
                slice(reshaped, {0, 0, 0, 1}, {1, seq_len, d_ff, 2}),
                {1, seq_len, d_ff}
            );
            const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
            return activated * up;
        }

        const array gate = slice(gate_up, {0, 0, 0}, {1, seq_len, d_ff});
        const array up = slice(gate_up, {0, 0, d_ff}, {1, seq_len, 2 * d_ff});
        const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
        return activated * up;
    }();

    const array mid_proj_in = (mid.dtype() == model_dtype) ? mid : astype(mid, model_dtype);
    const array ffn_out = down_proj_fn(mid_proj_in);
    return hidden_1 + ffn_out * residual_multiplier;
}

void* mlx_lazy_state_space_block_bf16(
    const void* input,
    const void* ln1_weight,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_proj,
    const void* ln2_weight,
    const void* gate_up,
    const void* down_proj,
    bool use_gelu,
    float residual_multiplier,
    float norm_eps,
    void* state_space_cache_ptr,
    size_t layer_idx,
    size_t d_state,
    size_t d_conv,
    size_t n_heads,
    size_t d_head,
    size_t n_groups,
    uint8_t gate_up_layout
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_proj);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = ensure_float32(*static_cast<const array*>(a_log));
    const auto d_skip_arr = ensure_float32(*static_cast<const array*>(d_skip));
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_proj);
    const auto* ln2_w = static_cast<const array*>(ln2_weight);
    const auto* gate_up_w = static_cast<const array*>(gate_up);
    const auto* down_proj_w = static_cast<const array*>(down_proj);

    const int d_state_i = static_cast<int>(d_state);
    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int n_groups_i = static_cast<int>(n_groups);
    const int d_inner = n_heads_i * d_head_i;
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;

    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(
            (conv_b->ndim() == 2 && conv_b->shape(0) == 1 && conv_b->shape(1) == xbc_len)
                ? ensure_float32(*conv_b)
                : reshape(ensure_float32(*conv_b), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(ensure_float32(*dt_b), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(*ln2_w)
        : std::nullopt;

    auto* cache = static_cast<MLXStateSpaceCache*>(state_space_cache_ptr);
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_state_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_state_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_state_i}, float32);
    }

    const auto in_proj_fn = [&in_proj_w, in_proj](const array& x) -> array {
        return dense_linear_no_bias(x, in_proj_w, std::nullopt, in_proj);
    };
    const auto out_proj_fn = [&out_proj_w, out_proj](const array& x) -> array {
        return dense_linear_no_bias(x, out_proj_w, std::nullopt, out_proj);
    };
    const bool has_ffn = gate_up_w != nullptr && down_proj_w != nullptr && ln2_w != nullptr;
    const auto gate_up_fn = [&gate_up_w, gate_up](const array& x) -> array {
        return dense_linear_no_bias(x, *gate_up_w, std::nullopt, gate_up);
    };
    const auto down_proj_fn = [&down_proj_w, down_proj](const array& x) -> array {
        return dense_linear_no_bias(x, *down_proj_w, std::nullopt, down_proj);
    };

    array out = state_space_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        a_log_arr,
        d_skip_arr,
        dt_bias_row,
        norm_weight_arr,
        out_proj_fn,
        ln2_weight_arr,
        gate_up_fn,
        down_proj_fn,
        has_ffn,
        use_gelu,
        residual_multiplier,
        norm_eps,
        d_state_i,
        d_conv_i,
        n_heads_i,
        d_head_i,
        n_groups_i,
        gate_up_layout,
        conv_state,
        ssm_state
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}

void* mlx_lazy_state_space_block_quantized(
    const void* input,
    const void* ln1_weight,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    const void* ln2_weight,
    const void* gate_up_w,
    const void* gate_up_s,
    const void* gate_up_b,
    const void* down_w,
    const void* down_s,
    const void* down_b,
    size_t group_size,
    size_t bits,
    bool use_gelu,
    float residual_multiplier,
    float norm_eps,
    void* state_space_cache_ptr,
    size_t layer_idx,
    size_t d_state,
    size_t d_conv,
    size_t n_heads,
    size_t d_head,
    size_t n_groups,
    uint8_t gate_up_layout
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_w);
    const auto& in_proj_s = *static_cast<const array*>(in_s);
    const auto& in_proj_b = *static_cast<const array*>(in_b);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = ensure_float32(*static_cast<const array*>(a_log));
    const auto d_skip_arr = ensure_float32(*static_cast<const array*>(d_skip));
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_w);
    const auto& out_proj_s = *static_cast<const array*>(out_s);
    const auto& out_proj_b = *static_cast<const array*>(out_b);
    const auto* ln2_w = static_cast<const array*>(ln2_weight);
    const auto* gate_up_w_arr = static_cast<const array*>(gate_up_w);
    const auto* gate_up_s_arr = static_cast<const array*>(gate_up_s);
    const auto* gate_up_b_arr = static_cast<const array*>(gate_up_b);
    const auto* down_w_arr = static_cast<const array*>(down_w);
    const auto* down_s_arr = static_cast<const array*>(down_s);
    const auto* down_b_arr = static_cast<const array*>(down_b);

    const int d_state_i = static_cast<int>(d_state);
    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int n_groups_i = static_cast<int>(n_groups);
    const int d_inner = n_heads_i * d_head_i;
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(
            (conv_b->ndim() == 2 && conv_b->shape(0) == 1 && conv_b->shape(1) == xbc_len)
                ? ensure_float32(*conv_b)
                : reshape(ensure_float32(*conv_b), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(ensure_float32(*dt_b), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(*ln2_w)
        : std::nullopt;

    auto* cache = static_cast<MLXStateSpaceCache*>(state_space_cache_ptr);
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_state_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_state_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_state_i}, float32);
    }

    const auto in_proj_fn = [&in_proj_w, &in_proj_s, &in_proj_b, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, in_proj_w, in_proj_s, in_proj_b, group_size_i, bits_i);
    };
    const auto out_proj_fn = [&out_proj_w, &out_proj_s, &out_proj_b, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, out_proj_w, out_proj_s, out_proj_b, group_size_i, bits_i);
    };
    const bool has_ffn =
        gate_up_w_arr != nullptr && gate_up_s_arr != nullptr && gate_up_b_arr != nullptr &&
        down_w_arr != nullptr && down_s_arr != nullptr && down_b_arr != nullptr &&
        ln2_w != nullptr;
    const auto gate_up_fn = [&gate_up_w_arr, &gate_up_s_arr, &gate_up_b_arr, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, *gate_up_w_arr, *gate_up_s_arr, *gate_up_b_arr, group_size_i, bits_i);
    };
    const auto down_proj_fn = [&down_w_arr, &down_s_arr, &down_b_arr, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, *down_w_arr, *down_s_arr, *down_b_arr, group_size_i, bits_i);
    };

    array out = state_space_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        a_log_arr,
        d_skip_arr,
        dt_bias_row,
        norm_weight_arr,
        out_proj_fn,
        ln2_weight_arr,
        gate_up_fn,
        down_proj_fn,
        has_ffn,
        use_gelu,
        residual_multiplier,
        norm_eps,
        d_state_i,
        d_conv_i,
        n_heads_i,
        d_head_i,
        n_groups_i,
        gate_up_layout,
        conv_state,
        ssm_state
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}
