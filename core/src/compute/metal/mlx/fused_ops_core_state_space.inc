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
static constexpr size_t kOrientedRhsHotSlots = 512;
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
    return tls_never_destroyed<std::unordered_map<OrientedRhsCacheKey, array, OrientedRhsCacheKeyHash>>();
}

struct OrientedRhsHotSlot {
    bool valid = false;
    OrientedRhsCacheKey key;
    array rhs = array(0.0f);
};

static inline OrientedRhsHotSlot* oriented_rhs_hot_slots_store() {
    return tls_never_destroyed_array<OrientedRhsHotSlot, kOrientedRhsHotSlots>();
}

struct OrientedRhsLastHit {
    bool valid = false;
    OrientedRhsCacheKey key;
    array rhs = array(0.0f);
};

static inline OrientedRhsLastHit& oriented_rhs_last_hit_store() {
    return tls_never_destroyed<OrientedRhsLastHit>();
}

static inline void oriented_rhs_cache_clear() {
    oriented_rhs_cache_store().clear();
    auto* slots = oriented_rhs_hot_slots_store();
    for (size_t i = 0; i < kOrientedRhsHotSlots; ++i) {
        slots[i].valid = false;
    }
    oriented_rhs_last_hit_store().valid = false;
}

// Defined in fused_ops_attention.inc and fused_ops_ffn_expert_mix.inc.
static inline void attention_fused_qkv_cache_clear();
static inline void attention_fused_qkv_bias_cache_clear();
static inline void ffn_quantized_gate_up_cache_clear();
static inline void ffn_fused_gate_up_cache_clear();
static inline void ffn_fused_dense_rhs_cache_clear();

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
    return tls_never_destroyed<std::unordered_map<StateSpaceConvKernelCacheKey, array, StateSpaceConvKernelCacheKeyHash>>();
}

static inline void state_space_conv_kernel_cache_clear() {
    state_space_conv_kernel_cache_store().clear();
}

struct CausalConvKernelCacheKey {
    const void* weight_handle = nullptr;
    int d_conv = 0;
    int conv_dim = 0;
    int target_dtype = 0;

    bool operator==(const CausalConvKernelCacheKey& other) const {
        return weight_handle == other.weight_handle &&
               d_conv == other.d_conv &&
               conv_dim == other.conv_dim &&
               target_dtype == other.target_dtype;
    }
};

struct CausalConvKernelCacheKeyHash {
    size_t operator()(const CausalConvKernelCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.weight_handle);
        h ^= std::hash<int>{}(key.d_conv) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.conv_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<CausalConvKernelCacheKey, array, CausalConvKernelCacheKeyHash>& causal_conv_kernel_cache_store() {
    return tls_never_destroyed<std::unordered_map<CausalConvKernelCacheKey, array, CausalConvKernelCacheKeyHash>>();
}

static inline void causal_conv_kernel_cache_clear() {
    causal_conv_kernel_cache_store().clear();
}

struct CausalConvBiasCacheKey {
    const void* bias_handle = nullptr;
    int conv_dim = 0;
    int target_dtype = 0;

    bool operator==(const CausalConvBiasCacheKey& other) const {
        return bias_handle == other.bias_handle &&
               conv_dim == other.conv_dim &&
               target_dtype == other.target_dtype;
    }
};

struct CausalConvBiasCacheKeyHash {
    size_t operator()(const CausalConvBiasCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.bias_handle);
        h ^= std::hash<int>{}(key.conv_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<CausalConvBiasCacheKey, array, CausalConvBiasCacheKeyHash>& causal_conv_bias_cache_store() {
    return tls_never_destroyed<std::unordered_map<CausalConvBiasCacheKey, array, CausalConvBiasCacheKeyHash>>();
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
    return tls_never_destroyed<std::unordered_map<BiasCastCacheKey, array, BiasCastCacheKeyHash>>();
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
static constexpr size_t kStateSpaceStaticCacheMaxEntries = 1024;
static constexpr size_t kStateSpaceRowCacheMaxEntries = 1024;
static constexpr size_t kGatedDeltaStaticCacheMaxEntries = 1024;

static inline std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash>& quant_param_cast_cache_store() {
    return tls_never_destroyed<std::unordered_map<QuantParamCastCacheKey, array, QuantParamCastCacheKeyHash>>();
}

static inline void quant_param_cast_cache_clear() {
    quant_param_cast_cache_store().clear();
}

struct StateSpaceStaticCacheKey {
    const void* a_log_handle = nullptr;
    const void* d_skip_handle = nullptr;
    int n_groups = 0;
    int heads_per_group = 0;

    bool operator==(const StateSpaceStaticCacheKey& other) const {
        return a_log_handle == other.a_log_handle &&
               d_skip_handle == other.d_skip_handle &&
               n_groups == other.n_groups &&
               heads_per_group == other.heads_per_group;
    }
};

struct StateSpaceStaticCacheKeyHash {
    size_t operator()(const StateSpaceStaticCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.a_log_handle);
        h ^= std::hash<const void*>{}(key.d_skip_handle) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n_groups) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.heads_per_group) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct StateSpaceStaticTerms {
    array a_log_exp_group;
    array d_skip_group;
};

static inline std::unordered_map<StateSpaceStaticCacheKey, StateSpaceStaticTerms, StateSpaceStaticCacheKeyHash>& state_space_static_cache_store() {
    return tls_never_destroyed<std::unordered_map<StateSpaceStaticCacheKey, StateSpaceStaticTerms, StateSpaceStaticCacheKeyHash>>();
}

static inline void state_space_static_cache_clear() {
    state_space_static_cache_store().clear();
}

struct StateSpaceRowCacheKey {
    const void* handle = nullptr;
    int length = 0;
    int target_dtype = 0;
    uint8_t kind = 0; // 0 = conv_bias_row, 1 = dt_bias_row

    bool operator==(const StateSpaceRowCacheKey& other) const {
        return handle == other.handle &&
               length == other.length &&
               target_dtype == other.target_dtype &&
               kind == other.kind;
    }
};

struct StateSpaceRowCacheKeyHash {
    size_t operator()(const StateSpaceRowCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.handle);
        h ^= std::hash<int>{}(key.length) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.target_dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint8_t>{}(key.kind) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<StateSpaceRowCacheKey, array, StateSpaceRowCacheKeyHash>& state_space_row_cache_store() {
    return tls_never_destroyed<std::unordered_map<StateSpaceRowCacheKey, array, StateSpaceRowCacheKeyHash>>();
}

static inline void state_space_row_cache_clear() {
    state_space_row_cache_store().clear();
}

struct GatedDeltaStaticCacheKey {
    const void* a_log_handle = nullptr;
    const void* norm_weight_handle = nullptr;
    int n_heads = 0;
    int d_head = 0;

    bool operator==(const GatedDeltaStaticCacheKey& other) const {
        return a_log_handle == other.a_log_handle &&
               norm_weight_handle == other.norm_weight_handle &&
               n_heads == other.n_heads &&
               d_head == other.d_head;
    }
};

struct GatedDeltaStaticCacheKeyHash {
    size_t operator()(const GatedDeltaStaticCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.a_log_handle);
        h ^= std::hash<const void*>{}(key.norm_weight_handle) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.n_heads) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.d_head) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct GatedDeltaStaticTerms {
    array a_log_exp_row;
    std::optional<array> norm_weight_heads = std::nullopt;
};

static inline array ensure_float32(const array& arr);

static inline std::unordered_map<GatedDeltaStaticCacheKey, GatedDeltaStaticTerms, GatedDeltaStaticCacheKeyHash>& gated_delta_static_cache_store() {
    return tls_never_destroyed<std::unordered_map<GatedDeltaStaticCacheKey, GatedDeltaStaticTerms, GatedDeltaStaticCacheKeyHash>>();
}

static inline void gated_delta_static_cache_clear() {
    gated_delta_static_cache_store().clear();
}

static inline array cast_tensor_cached(
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
}

static inline StateSpaceStaticTerms resolve_state_space_static_terms(
    const void* a_log_handle,
    const void* d_skip_handle,
    const array& a_log_arr,
    const array& d_skip_arr,
    int n_groups,
    int heads_per_group
) {
    const auto uncached = [&]() -> StateSpaceStaticTerms {
        return StateSpaceStaticTerms{
            exp(reshape(a_log_arr, {1, n_groups, heads_per_group, 1, 1})),
            reshape(d_skip_arr, {1, n_groups, heads_per_group, 1}),
        };
    };

    if (a_log_handle == nullptr || d_skip_handle == nullptr) {
        return uncached();
    }

    auto& cache = state_space_static_cache_store();
    if (cache.size() >= kStateSpaceStaticCacheMaxEntries) {
        cache.clear();
    }

    StateSpaceStaticCacheKey key;
    key.a_log_handle = a_log_handle;
    key.d_skip_handle = d_skip_handle;
    key.n_groups = n_groups;
    key.heads_per_group = heads_per_group;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    StateSpaceStaticTerms terms = uncached();
    // Materialize once so repeated decode steps reuse static terms.
    eval({terms.a_log_exp_group, terms.d_skip_group});
    auto [inserted, _] = cache.emplace(key, std::move(terms));
    return inserted->second;
}

static inline array resolve_state_space_row_cached(
    const void* handle,
    const array& param,
    int length,
    Dtype target_dtype,
    uint8_t kind
) {
    const auto uncached = [&]() -> array {
        const array casted = (param.dtype() == target_dtype) ? param : astype(param, target_dtype);
        if (casted.ndim() == 2 && casted.shape(0) == 1 && casted.shape(1) == length) {
            return casted;
        }
        return reshape(casted, {1, length});
    };

    if (handle == nullptr) {
        return uncached();
    }

    auto& cache = state_space_row_cache_store();
    if (cache.size() >= kStateSpaceRowCacheMaxEntries) {
        cache.clear();
    }

    StateSpaceRowCacheKey key;
    key.handle = handle;
    key.length = length;
    key.target_dtype = static_cast<int>(target_dtype.val());
    key.kind = kind;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    array row = uncached();
    eval(row);
    auto [inserted, _] = cache.emplace(key, std::move(row));
    return inserted->second;
}

static inline GatedDeltaStaticTerms resolve_gated_delta_static_terms(
    const void* a_log_handle,
    const void* norm_weight_handle,
    const array& a_log_arr,
    const std::optional<array>& norm_weight_arr,
    int n_heads,
    int d_head
) {
    const auto uncached = [&]() -> GatedDeltaStaticTerms {
        GatedDeltaStaticTerms terms{
            // Keep raw a_log values here. The recurrence kernel applies the
            // CPU-equivalent fast-exp approximation per head.
            reshape(a_log_arr, {n_heads}),
            std::nullopt,
        };
        if (!norm_weight_arr.has_value()) {
            return terms;
        }

        const int d_inner = n_heads * d_head;
        array weights = ensure_float32(*norm_weight_arr);
        if (weights.ndim() == 2 && weights.shape(0) == 1) {
            weights = reshape(weights, {weights.shape(1)});
        }
        if (weights.ndim() != 1) {
            throw std::invalid_argument("[gated_delta] norm_weight must be rank-1");
        }
        if (weights.shape(0) == d_head) {
            terms.norm_weight_heads = contiguous(broadcast_to(reshape(weights, {1, d_head}), {n_heads, d_head}));
        } else if (weights.shape(0) == d_inner) {
            terms.norm_weight_heads = contiguous(reshape(weights, {n_heads, d_head}));
        } else {
            throw std::invalid_argument("[gated_delta] norm_weight shape incompatible with head dims");
        }
        return terms;
    };

    if (a_log_handle == nullptr && norm_weight_handle == nullptr) {
        return uncached();
    }

    auto& cache = gated_delta_static_cache_store();
    if (cache.size() >= kGatedDeltaStaticCacheMaxEntries) {
        cache.clear();
    }

    GatedDeltaStaticCacheKey key;
    key.a_log_handle = a_log_handle;
    key.norm_weight_handle = norm_weight_handle;
    key.n_heads = n_heads;
    key.d_head = d_head;
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    GatedDeltaStaticTerms terms = uncached();
    eval(terms.a_log_exp_row);
    if (terms.norm_weight_heads.has_value()) {
        eval(*terms.norm_weight_heads);
    }
    auto [inserted, _] = cache.emplace(key, std::move(terms));
    return inserted->second;
}

static inline fast::CustomKernelFunction& gated_delta_recurrence_step_kernel() {
    static thread_local fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_gated_delta_recurrence_step",
        {
            "state",
            "query",
            "key",
            "value",
            "z",
            "beta_raw",
            "a_raw",
            "a_log_exp",
            "dt_bias",
            "norm_weight",
        },
        { "out", "next_state" },
        R"(
    constexpr float eps = 1.0e-6f;
    auto fast_exp = [](float x) -> float {
      constexpr float log2e = 1.4426950408889634f;
      constexpr float ln2_hi = 0.693359375f;
      constexpr float ln2_lo = -2.12194440e-4f;
      constexpr float exp_hi = 88.3762626647949f;
      constexpr float exp_lo = -88.3762626647949f;
      x = clamp(x, exp_lo, exp_hi);
      const float fx = floor(x * log2e + 0.5f);
      const float xr = (x - fx * ln2_hi) - fx * ln2_lo;
      float y = 1.9875691500e-4f;
      y = y * xr + 1.3981999507e-3f;
      y = y * xr + 8.3334519073e-3f;
      y = y * xr + 4.1665795894e-2f;
      y = y * xr + 1.6666665459e-1f;
      y = y * xr + 5.0000001201e-1f;
      y = y * xr * xr + xr + 1.0f;
      const int exponent_i = int(fx) + 127;
      const uint exponent_bits = uint(exponent_i) << 23;
      const float pow2n = as_type<float>(exponent_bits);
      return y * pow2n;
    };
    auto fast_sigmoid = [&](float x) -> float {
      return 1.0f / (1.0f + fast_exp(-x));
    };
    auto fast_softplus = [&](float x) -> float {
      if (x > 20.0f) return x;
      return log(1.0f + fast_exp(x));
    };
    const uint lane = thread_position_in_grid.x;
    const uint head = thread_position_in_grid.y;
    const bool active = lane < D_HEAD;
    threadgroup float q_shared[D_HEAD];
    threadgroup float k_shared[D_HEAD];
    threadgroup float out_shared[D_HEAD];
    threadgroup float scalar_shared[8];

    const uint vec_base = head * D_HEAD;
    const uint state_base = head * D_HEAD * D_HEAD;

    float q = 0.0f;
    float k = 0.0f;
    if (active) {
      q = query[vec_base + lane];
      k = key[vec_base + lane];
      // Store q*q, k*k for sequential reduction (matches CPU precision)
      q_shared[lane] = q * q;
      k_shared[lane] = k * k;
      out_shared[lane] = q * k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential reduction by lane 0 for exact CPU match
    // (tiny cost: D_HEAD iterations vs D_HEAD^2 state operations)
    if (lane == 0) {
      float q_sq = 0.0f, k_sq = 0.0f, qk = 0.0f;
      for (uint i = 0; i < D_HEAD; ++i) {
        q_sq += q_shared[i];
        k_sq += k_shared[i];
        qk += out_shared[i];
      }
      const float q_scale = rsqrt(q_sq + eps) * rsqrt((float)D_HEAD);
      const float k_scale = rsqrt(k_sq + eps);
      scalar_shared[0] = qk * q_scale * k_scale;  // qk_dot
      scalar_shared[1] = k_scale;
      scalar_shared[4] = q_scale;
      const float a = a_raw[head] + dt_bias[head];
      const float a_log_exp_value = fast_exp(a_log_exp[head]);
      scalar_shared[2] = fast_sigmoid(beta_raw[head]);
      scalar_shared[3] = fast_exp((-a_log_exp_value) * fast_softplus(a));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      q *= scalar_shared[4];
      k *= scalar_shared[1];
      q_shared[lane] = q;
      k_shared[lane] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_value = 0.0f;
    if (active) {
      const float beta = scalar_shared[2];
      const float decay = scalar_shared[3];
      float kv_mem = 0.0f;
      // Use fma for higher precision accumulation (matches CPU @mulAdd)
      for (uint i = 0; i < D_HEAD; ++i) {
        const uint state_index = state_base + (i * D_HEAD) + lane;
        const float scaled = state[state_index] * decay;
        kv_mem = fma(scaled, k_shared[i], kv_mem);
        out_value = fma(scaled, q_shared[i], out_value);
      }

      kv_mem = (value[vec_base + lane] - kv_mem) * beta;
      out_value = fma(kv_mem, scalar_shared[0], out_value);

      for (uint i = 0; i < D_HEAD; ++i) {
        const uint state_index = state_base + (i * D_HEAD) + lane;
        const float scaled = state[state_index] * decay;
        next_state[state_index] = fma(k_shared[i], kv_mem, scaled);
      }

      // RMS norm: sequential reduction for exact CPU match
      out_shared[lane] = out_value * out_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      float sum_sq = 0.0f;
      for (uint i = 0; i < D_HEAD; ++i) {
        sum_sq += out_shared[i];
      }
      scalar_shared[5] = rsqrt((sum_sq / (float)D_HEAD) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      out_value *= scalar_shared[5];
      out_value *= norm_weight[vec_base + lane];
      const float z_value = z[vec_base + lane];
      out[vec_base + lane] = out_value * (z_value * fast_sigmoid(z_value));
    }
        )",
        "",
        true,
        false
    );
    return kernel;
}

static inline fast::CustomKernelFunction& gated_delta_recurrence_step_debug_kernel() {
    static thread_local fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_gated_delta_recurrence_step_debug",
        {
            "state",
            "query",
            "key",
            "value",
            "z",
            "beta_raw",
            "a_raw",
            "a_log_exp",
            "dt_bias",
            "norm_weight",
        },
        { "out_ssm", "out_norm", "next_state" },
        R"(
    constexpr float eps = 1.0e-6f;
    auto fast_exp = [](float x) -> float {
      constexpr float log2e = 1.4426950408889634f;
      constexpr float ln2_hi = 0.693359375f;
      constexpr float ln2_lo = -2.12194440e-4f;
      constexpr float exp_hi = 88.3762626647949f;
      constexpr float exp_lo = -88.3762626647949f;
      x = clamp(x, exp_lo, exp_hi);
      const float fx = floor(x * log2e + 0.5f);
      const float xr = (x - fx * ln2_hi) - fx * ln2_lo;
      float y = 1.9875691500e-4f;
      y = y * xr + 1.3981999507e-3f;
      y = y * xr + 8.3334519073e-3f;
      y = y * xr + 4.1665795894e-2f;
      y = y * xr + 1.6666665459e-1f;
      y = y * xr + 5.0000001201e-1f;
      y = y * xr * xr + xr + 1.0f;
      const int exponent_i = int(fx) + 127;
      const uint exponent_bits = uint(exponent_i) << 23;
      const float pow2n = as_type<float>(exponent_bits);
      return y * pow2n;
    };
    auto fast_sigmoid = [&](float x) -> float {
      return 1.0f / (1.0f + fast_exp(-x));
    };
    auto fast_softplus = [&](float x) -> float {
      if (x > 20.0f) return x;
      return log(1.0f + fast_exp(x));
    };
    const uint lane = thread_position_in_grid.x;
    const uint head = thread_position_in_grid.y;
    const bool active = lane < D_HEAD;
    threadgroup float q_shared[D_HEAD];
    threadgroup float k_shared[D_HEAD];
    threadgroup float out_shared[D_HEAD];
    threadgroup float scalar_shared[8];

    const uint vec_base = head * D_HEAD;
    const uint state_base = head * D_HEAD * D_HEAD;

    float q = 0.0f;
    float k = 0.0f;
    if (active) {
      q = query[vec_base + lane];
      k = key[vec_base + lane];
      // Store q*q, k*k for sequential reduction (matches CPU precision)
      q_shared[lane] = q * q;
      k_shared[lane] = k * k;
      out_shared[lane] = q * k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential reduction by lane 0 for exact CPU match
    if (lane == 0) {
      float q_sq = 0.0f, k_sq = 0.0f, qk = 0.0f;
      for (uint i = 0; i < D_HEAD; ++i) {
        q_sq += q_shared[i];
        k_sq += k_shared[i];
        qk += out_shared[i];
      }
      const float q_scale = rsqrt(q_sq + eps) * rsqrt((float)D_HEAD);
      const float k_scale = rsqrt(k_sq + eps);
      scalar_shared[0] = qk * q_scale * k_scale;
      scalar_shared[1] = k_scale;
      scalar_shared[4] = q_scale;
      const float a = a_raw[head] + dt_bias[head];
      const float a_log_exp_value = fast_exp(a_log_exp[head]);
      scalar_shared[2] = fast_sigmoid(beta_raw[head]);
      scalar_shared[3] = fast_exp((-a_log_exp_value) * fast_softplus(a));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      q *= scalar_shared[4];
      k *= scalar_shared[1];
      q_shared[lane] = q;
      k_shared[lane] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float out_value = 0.0f;
    if (active) {
      const float beta = scalar_shared[2];
      const float decay = scalar_shared[3];
      float kv_mem = 0.0f;
      // Use fma for higher precision accumulation (matches CPU @mulAdd)
      for (uint i = 0; i < D_HEAD; ++i) {
        const uint state_index = state_base + (i * D_HEAD) + lane;
        const float scaled = state[state_index] * decay;
        kv_mem = fma(scaled, k_shared[i], kv_mem);
        out_value = fma(scaled, q_shared[i], out_value);
      }

      kv_mem = (value[vec_base + lane] - kv_mem) * beta;
      out_value = fma(kv_mem, scalar_shared[0], out_value);
      out_ssm[vec_base + lane] = out_value;

      for (uint i = 0; i < D_HEAD; ++i) {
        const uint state_index = state_base + (i * D_HEAD) + lane;
        const float scaled = state[state_index] * decay;
        next_state[state_index] = fma(k_shared[i], kv_mem, scaled);
      }

      // RMS norm: sequential reduction for exact CPU match
      out_shared[lane] = out_value * out_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      float sum_sq = 0.0f;
      for (uint i = 0; i < D_HEAD; ++i) {
        sum_sq += out_shared[i];
      }
      scalar_shared[5] = rsqrt((sum_sq / (float)D_HEAD) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      out_value *= scalar_shared[5];
      out_value *= norm_weight[vec_base + lane];
      const float z_value = z[vec_base + lane];
      out_norm[vec_base + lane] = out_value * (z_value * fast_sigmoid(z_value));
    }
        )",
        "",
        true,
        false
    );
    return kernel;
}

static inline fast::CustomKernelFunction& gated_delta_fast_silu_kernel() {
    static thread_local fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_gated_delta_fast_silu",
        { "x" },
        { "y" },
        R"(
    const uint col = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    if (col >= WIDTH || row >= SEQ) {
      return;
    }
    auto fast_exp = [](float x) -> float {
      constexpr float log2e = 1.4426950408889634f;
      constexpr float ln2_hi = 0.693359375f;
      constexpr float ln2_lo = -2.12194440e-4f;
      constexpr float exp_hi = 88.3762626647949f;
      constexpr float exp_lo = -88.3762626647949f;
      x = clamp(x, exp_lo, exp_hi);
      const float fx = floor(x * log2e + 0.5f);
      const float xr = (x - fx * ln2_hi) - fx * ln2_lo;
      float yv = 1.9875691500e-4f;
      yv = yv * xr + 1.3981999507e-3f;
      yv = yv * xr + 8.3334519073e-3f;
      yv = yv * xr + 4.1665795894e-2f;
      yv = yv * xr + 1.6666665459e-1f;
      yv = yv * xr + 5.0000001201e-1f;
      yv = yv * xr * xr + xr + 1.0f;
      const int exponent_i = int(fx) + 127;
      const uint exponent_bits = uint(exponent_i) << 23;
      const float pow2n = as_type<float>(exponent_bits);
      return yv * pow2n;
    };
    const uint idx = row * WIDTH + col;
    const float v = x[idx];
    const float sig = 1.0f / (1.0f + fast_exp(-v));
    y[idx] = v * sig;
        )",
        "",
        true,
        false
    );
    return kernel;
}

static inline array gated_delta_fast_silu_3d(const array& input, int seq_len, int width) {
    const int tg_x = (width < 256) ? width : 256;
    auto outputs = gated_delta_fast_silu_kernel()(
        { input },
        { Shape{1, seq_len, width} },
        { float32 },
        { width, seq_len, 1 },
        { tg_x, 1, 1 },
        { { "WIDTH", width }, { "SEQ", seq_len } },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 1) {
        throw std::runtime_error("[gated_delta] fast silu kernel returned unexpected output count");
    }
    return outputs[0];
}

static inline std::pair<array, array> gated_delta_recurrence_step(
    const array& state,
    const array& query,
    const array& key,
    const array& value,
    const array& z,
    const array& beta_raw,
    const array& a_raw,
    const array& a_log_exp_row,
    const array& dt_bias_heads,
    const array& norm_weight_heads,
    int n_heads,
    int d_head
) {
    auto outputs = gated_delta_recurrence_step_kernel()(
        {
            state,
            query,
            key,
            value,
            z,
            beta_raw,
            a_raw,
            a_log_exp_row,
            dt_bias_heads,
            norm_weight_heads,
        },
        {
            Shape{n_heads, d_head},
            Shape{n_heads, d_head, d_head},
        },
        { float32, float32 },
        { d_head, n_heads, 1 },
        { d_head, 1, 1 },
        { { "D_HEAD", d_head } },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 2) {
        throw std::runtime_error("[gated_delta] recurrence kernel returned unexpected output count");
    }
    return { outputs[0], outputs[1] };
}

static inline fast::CustomKernelFunction& gated_delta_recurrence_ssm_probe_kernel() {
    static thread_local fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_gated_delta_recurrence_ssm_probe",
        {
            "state",
            "query",
            "key",
            "value",
            "beta_raw",
            "a_raw",
            "a_log_exp",
            "dt_bias",
        },
        { "out_ssm" },
        R"(
    constexpr float eps = 1.0e-6f;
    auto fast_exp = [](float x) -> float {
      constexpr float log2e = 1.4426950408889634f;
      constexpr float ln2_hi = 0.693359375f;
      constexpr float ln2_lo = -2.12194440e-4f;
      constexpr float exp_hi = 88.3762626647949f;
      constexpr float exp_lo = -88.3762626647949f;
      x = clamp(x, exp_lo, exp_hi);
      const float fx = floor(x * log2e + 0.5f);
      const float xr = (x - fx * ln2_hi) - fx * ln2_lo;
      float y = 1.9875691500e-4f;
      y = y * xr + 1.3981999507e-3f;
      y = y * xr + 8.3334519073e-3f;
      y = y * xr + 4.1665795894e-2f;
      y = y * xr + 1.6666665459e-1f;
      y = y * xr + 5.0000001201e-1f;
      y = y * xr * xr + xr + 1.0f;
      const int exponent_i = int(fx) + 127;
      const uint exponent_bits = uint(exponent_i) << 23;
      const float pow2n = as_type<float>(exponent_bits);
      return y * pow2n;
    };
    auto fast_sigmoid = [&](float x) -> float {
      return 1.0f / (1.0f + fast_exp(-x));
    };
    auto fast_softplus = [&](float x) -> float {
      if (x > 20.0f) return x;
      return log(1.0f + fast_exp(x));
    };
    const uint lane = thread_position_in_grid.x;
    const uint head = thread_position_in_grid.y;
    const bool active = lane < D_HEAD;
    threadgroup float q_shared[D_HEAD];
    threadgroup float k_shared[D_HEAD];
    threadgroup float out_shared[D_HEAD];
    threadgroup float scalar_shared[8];

    const uint vec_base = head * D_HEAD;
    const uint state_base = head * D_HEAD * D_HEAD;

    float q = 0.0f;
    float k = 0.0f;
    if (active) {
      q = query[vec_base + lane];
      k = key[vec_base + lane];
      // Store q*q, k*k for sequential reduction (matches CPU precision)
      q_shared[lane] = q * q;
      k_shared[lane] = k * k;
      out_shared[lane] = q * k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sequential reduction by lane 0 for exact CPU match
    if (lane == 0) {
      float q_sq = 0.0f, k_sq = 0.0f, qk = 0.0f;
      for (uint i = 0; i < D_HEAD; ++i) {
        q_sq += q_shared[i];
        k_sq += k_shared[i];
        qk += out_shared[i];
      }
      const float q_scale = rsqrt(q_sq + eps) * rsqrt((float)D_HEAD);
      const float k_scale = rsqrt(k_sq + eps);
      scalar_shared[0] = qk * q_scale * k_scale;
      scalar_shared[1] = k_scale;
      scalar_shared[4] = q_scale;
      const float a = a_raw[head] + dt_bias[head];
      const float a_log_exp_value = fast_exp(a_log_exp[head]);
      scalar_shared[2] = fast_sigmoid(beta_raw[head]);
      scalar_shared[3] = fast_exp((-a_log_exp_value) * fast_softplus(a));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      q *= scalar_shared[4];
      k *= scalar_shared[1];
      q_shared[lane] = q;
      k_shared[lane] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
      const float beta = scalar_shared[2];
      const float decay = scalar_shared[3];
      float kv_mem = 0.0f;
      float out_value = 0.0f;
      // Use fma for higher precision accumulation (matches CPU @mulAdd)
      for (uint i = 0; i < D_HEAD; ++i) {
        const uint state_index = state_base + (i * D_HEAD) + lane;
        const float scaled = state[state_index] * decay;
        kv_mem = fma(scaled, k_shared[i], kv_mem);
        out_value = fma(scaled, q_shared[i], out_value);
      }

      kv_mem = (value[vec_base + lane] - kv_mem) * beta;
      out_value = fma(kv_mem, scalar_shared[0], out_value);
      out_ssm[vec_base + lane] = out_value;
    }
        )",
        "",
        true,
        false
    );
    return kernel;
}

static inline array gated_delta_recurrence_ssm_probe(
    const array& state,
    const array& query,
    const array& key,
    const array& value,
    const array& beta_raw,
    const array& a_raw,
    const array& a_log_exp_row,
    const array& dt_bias_heads,
    int n_heads,
    int d_head
) {
    auto outputs = gated_delta_recurrence_ssm_probe_kernel()(
        {
            state,
            query,
            key,
            value,
            beta_raw,
            a_raw,
            a_log_exp_row,
            dt_bias_heads,
        },
        { Shape{n_heads, d_head} },
        { float32 },
        { d_head, n_heads, 1 },
        { d_head, 1, 1 },
        { { "D_HEAD", d_head } },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 1) {
        throw std::runtime_error("[gated_delta] ssm probe kernel returned unexpected output count");
    }
    return outputs[0];
}

struct GatedDeltaRecurrenceDebugStep {
    array ssm;
    array norm;
    array next_state;
};

static inline GatedDeltaRecurrenceDebugStep gated_delta_recurrence_step_debug(
    const array& state,
    const array& query,
    const array& key,
    const array& value,
    const array& z,
    const array& beta_raw,
    const array& a_raw,
    const array& a_log_exp_row,
    const array& dt_bias_heads,
    const array& norm_weight_heads,
    int n_heads,
    int d_head
) {
    auto outputs = gated_delta_recurrence_step_debug_kernel()(
        {
            state,
            query,
            key,
            value,
            z,
            beta_raw,
            a_raw,
            a_log_exp_row,
            dt_bias_heads,
            norm_weight_heads,
        },
        {
            Shape{n_heads, d_head},
            Shape{n_heads, d_head},
            Shape{n_heads, d_head, d_head},
        },
        { float32, float32, float32 },
        { d_head, n_heads, 1 },
        { d_head, 1, 1 },
        { { "D_HEAD", d_head } },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 3) {
        throw std::runtime_error("[gated_delta] debug recurrence kernel returned unexpected output count");
    }
    return GatedDeltaRecurrenceDebugStep{
        .ssm = outputs[0],
        .norm = outputs[1],
        .next_state = outputs[2],
    };
}

static inline int infer_state_space_conv_dim_from_weight(
    const array& conv_weight,
    int d_conv,
    const char* scope
) {
    if (d_conv <= 0) {
        throw std::invalid_argument(std::string("[") + scope + "] invalid d_conv");
    }

    array kernel = conv_weight;
    if (kernel.ndim() == 3 &&
        kernel.shape(0) == 1 &&
        kernel.shape(1) == d_conv) {
        return kernel.shape(2);
    }
    if (kernel.ndim() == 3) {
        // [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
        kernel = reshape(kernel, {kernel.shape(0), kernel.shape(2)});
    }
    if (kernel.ndim() != 2) {
        throw std::invalid_argument(std::string("[") + scope + "] conv1d_weight must be rank-2 or rank-3");
    }

    if (kernel.shape(0) == d_conv) {
        return kernel.shape(1);
    }
    if (kernel.shape(1) == d_conv) {
        return kernel.shape(0);
    }
    throw std::invalid_argument(std::string("[") + scope + "] conv1d_weight shape incompatible with d_conv");
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

    OrientedRhsCacheKey key;
    key.weight_handle = weight_handle;
    key.in_features = in_features;
    key.out_features = out_features.has_value() ? out_features.value() : -1;
    key.transpose_when_ambiguous = transpose_when_ambiguous;

    auto* hot_slots = oriented_rhs_hot_slots_store();
    const size_t hot_index = OrientedRhsCacheKeyHash{}(key) & (kOrientedRhsHotSlots - 1);
    auto& hot_slot = hot_slots[hot_index];
    if (hot_slot.valid && hot_slot.key == key) {
        return hot_slot.rhs;
    }

    auto& last_hit = oriented_rhs_last_hit_store();
    if (last_hit.valid && last_hit.key == key) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.rhs = last_hit.rhs;
        return last_hit.rhs;
    }

    auto& cache = oriented_rhs_cache_store();
    if (cache.size() >= kOrientedRhsCacheMaxEntries) {
        cache.clear();
        for (size_t i = 0; i < kOrientedRhsHotSlots; ++i) {
            hot_slots[i].valid = false;
        }
        last_hit.valid = false;
    }

    auto it = cache.find(key);
    if (it != cache.end()) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.rhs = it->second;
        last_hit.valid = true;
        last_hit.key = key;
        last_hit.rhs = it->second;
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
    hot_slot.valid = true;
    hot_slot.key = key;
    hot_slot.rhs = inserted_it->second;
    last_hit.valid = true;
    last_hit.key = key;
    last_hit.rhs = inserted_it->second;
    return inserted_it->second;
}

void mlx_weight_transform_cache_clear() {
    oriented_rhs_cache_clear();
    state_space_conv_kernel_cache_clear();
    causal_conv_kernel_cache_clear();
    causal_conv_bias_cache_clear();
    bias_cast_cache_clear();
    quant_param_cast_cache_clear();
    state_space_static_cache_clear();
    state_space_row_cache_clear();
    gated_delta_static_cache_clear();
    attention_fused_qkv_cache_clear();
    attention_fused_qkv_bias_cache_clear();
    ffn_quantized_gate_up_cache_clear();
    ffn_fused_gate_up_cache_clear();
    ffn_fused_dense_rhs_cache_clear();
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

    // Decode hot path: [1, 1, D] @ [D, O]. Prefer vector matmul dispatch to
    // reduce launch overhead versus generic flattened GEMM.
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
    if (input.dtype() == param_dtype && biases.dtype() == param_dtype) {
        return quantized_matmul_lastdim(
            input,
            weights,
            scales,
            biases,
            group_size,
            bits
        );
    }
    const array input_cast = (input.dtype() == param_dtype) ? input : astype(input, param_dtype);
    const array scales_cast = cast_tensor_cached(scales_handle, scales, param_dtype);
    const array biases_cast = cast_tensor_cached(biases_handle, biases, param_dtype);
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
        return kernel;
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
    // Keep conv kernel in model dtype; FP32 is reintroduced only at recurrent
    // accumulation boundaries.
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
    int conv_dim,
    Dtype target_dtype
) {
    const auto uncached = [&]() -> array {
        array kernel = conv_weight;
        if (kernel.ndim() == 3 &&
            kernel.shape(0) == 1 &&
            kernel.shape(1) == d_conv &&
            kernel.shape(2) == conv_dim) {
            const array casted = (kernel.dtype() == target_dtype) ? kernel : astype(kernel, target_dtype);
            return contiguous(casted);
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
        kernel = (kernel.dtype() == target_dtype) ? kernel : astype(kernel, target_dtype);
        return contiguous(reshape(kernel, {1, d_conv, conv_dim}));
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
    key.target_dtype = static_cast<int>(target_dtype.val());
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
    int conv_dim,
    Dtype target_dtype
) {
    if (conv_bias_arr == nullptr) {
        return std::nullopt;
    }

    const auto uncached = [&]() -> array {
        const array conv_bias_cast =
            (conv_bias_arr->dtype() == target_dtype) ? *conv_bias_arr : astype(*conv_bias_arr, target_dtype);
        if (conv_bias_cast.ndim() == 2 && conv_bias_cast.shape(0) == 1 && conv_bias_cast.shape(1) == conv_dim) {
            return contiguous(conv_bias_cast);
        }
        return contiguous(reshape(conv_bias_cast, {1, conv_dim}));
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
    key.target_dtype = static_cast<int>(target_dtype.val());
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

struct StateSpaceCallable {
    array (*fn)(const array&, const void*) = nullptr;
    const void* ctx = nullptr;
};

static inline array state_space_call(const StateSpaceCallable& callable, const array& x) {
    if (callable.fn == nullptr) {
        throw std::invalid_argument("[state_space] missing callable");
    }
    return callable.fn(x, callable.ctx);
}

static array state_space_apply_matmul_rhs(const array& x, const void* ctx) {
    const auto* rhs = static_cast<const array*>(ctx);
    if (rhs == nullptr) {
        throw std::invalid_argument("[state_space] missing matmul rhs");
    }
    return matmul_lastdim(x, *rhs);
}

static array state_space_apply_matmul_rhs_reference(const array& x, const void* ctx) {
    const auto* rhs = static_cast<const array*>(ctx);
    if (rhs == nullptr) {
        throw std::invalid_argument("[state_space] missing matmul rhs");
    }
    return state_space_host_matmul_lastdim_reference(x, *rhs);
}

static array state_space_apply_linear_reference(const array& x, const void* ctx) {
    const auto* weight = static_cast<const array*>(ctx);
    if (weight == nullptr) {
        throw std::invalid_argument("[state_space] missing linear weight");
    }
    return state_space_host_linear_reference(x, *weight);
}

struct StateSpaceQuantLinearCtx {
    const array* w = nullptr;
    const array* s = nullptr;
    const array* b = nullptr;
    const void* s_handle = nullptr;
    const void* b_handle = nullptr;
    int group_size = 0;
    int bits = 0;
};

static array state_space_apply_quantized_linear(const array& x, const void* ctx_ptr) {
    const auto* ctx = static_cast<const StateSpaceQuantLinearCtx*>(ctx_ptr);
    if (ctx == nullptr || ctx->w == nullptr || ctx->s == nullptr || ctx->b == nullptr) {
        throw std::invalid_argument("[state_space] missing quantized linear params");
    }
    return quantized_linear_no_bias(
        x,
        *ctx->w,
        *ctx->s,
        *ctx->b,
        ctx->group_size,
        ctx->bits,
        ctx->s_handle,
        ctx->b_handle
    );
}

struct GatedDeltaCaptureOutputs {
    std::optional<array> in_proj;
    std::optional<array> conv;
    std::optional<array> ssm;
    std::optional<array> norm;
    std::optional<array> state_conv;
    std::optional<array> state_ssm;
};

static inline void materialize_gated_delta_capture_outputs(GatedDeltaCaptureOutputs* capture_out) {
    if (capture_out == nullptr) {
        return;
    }
    // Batch eval all present captures for efficiency.
    std::vector<array> to_eval;
    to_eval.reserve(4);
    if (capture_out->in_proj.has_value()) to_eval.push_back(*capture_out->in_proj);
    if (capture_out->conv.has_value()) to_eval.push_back(*capture_out->conv);
    if (capture_out->ssm.has_value()) to_eval.push_back(*capture_out->ssm);
    if (capture_out->norm.has_value()) to_eval.push_back(*capture_out->norm);
    if (capture_out->state_conv.has_value()) to_eval.push_back(*capture_out->state_conv);
    if (capture_out->state_ssm.has_value()) to_eval.push_back(*capture_out->state_ssm);
    if (!to_eval.empty()) eval(to_eval);
}

static inline array gated_delta_mixer_forward_core(
    const array& input,
    const StateSpaceCallable& in_proj_fn,
    const array& conv_kernel_broadcast,
    const std::optional<array>& conv_bias_row,
    const array& a_log_exp_row,
    const std::optional<array>& dt_bias_row,
    const std::optional<array>& norm_weight_heads,
    const StateSpaceCallable& out_proj_fn,
    int d_conv,
    int n_heads,
    int n_key_heads,
    int d_head,
    array& conv_state,
    array& ssm_state,
    GatedDeltaCaptureOutputs* capture_out
);

static array state_space_forward_core(
    const array& input, // [1, L, d_model]
    const array& ln1_weight,
    const StateSpaceCallable& in_proj_fn,
    const array& conv_kernel_broadcast, // [1, d_conv, xbc_len]
    const std::optional<array>& conv_bias_row, // [1, xbc_len]
    const array& a_log_exp_group, // [1, n_groups, heads_per_group, 1, 1]
    const array& d_skip_group, // [1, n_groups, heads_per_group, 1]
    const std::optional<array>& dt_bias_row, // [1, n_heads]
    const std::optional<array>& norm_weight, // [d_inner]
    const StateSpaceCallable& out_proj_fn,
    const std::optional<array>& ln2_weight,
    const StateSpaceCallable& gate_up_fn,
    const StateSpaceCallable& down_proj_fn,
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
    const Dtype model_dtype = input.dtype();
    const int d_inner = n_heads * d_head;
    const int bc_len = n_groups * d_state;
    const int xbc_len = d_inner + 2 * bc_len;
    if (n_heads <= 0 or d_head <= 0 or d_state <= 0 or d_conv <= 0 or n_groups <= 0) {
        throw std::invalid_argument("[state_space] invalid config dims");
    }
    if ((n_heads % n_groups) != 0) {
        throw std::invalid_argument("[state_space] n_heads must be divisible by n_groups");
    }

    const int heads_per_group = n_heads / n_groups;
    const int min_proj_dim = (2 * d_inner + 2 * bc_len + n_heads);

    // Hoist token-independent preprocessing out of the recurrent loop.
    const array ln1_norm_weight = canonicalize_rms_norm_weight(ln1_weight, input.shape(2), "state_space.ln1_weight");
    const array norm_all = fast::rms_norm(input, ln1_norm_weight, norm_eps);
    const array proj_all = state_space_call(in_proj_fn, norm_all);
    const int proj_dim = proj_all.shape(2);
    if (proj_dim < min_proj_dim) {
        throw std::invalid_argument("[state_space] in_proj output too small");
    }

    const array z_all = slice(proj_all, {0, 0, 0}, {1, seq_len, d_inner});
    const array xbc_all = slice(proj_all, {0, 0, d_inner}, {1, seq_len, d_inner + xbc_len});
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
    array conv_seq = causal_conv_run_sequence(
        &conv_state,
        xbc_all,
        conv_kernel_broadcast,
        seq_len,
        d_conv,
        xbc_len
    );
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, xbc_len});
    }
    const array xbc_conv_all = conv_seq * sigmoid(conv_seq);

    // FP32 allowlist starts here for recurrent-state-sensitive math.
    // Keep x in model dtype; upcast only recurrent B/C streams.
    const array x_conv_all = slice(xbc_conv_all, {0, 0, 0}, {1, seq_len, d_inner});
    const array b_groups_all = reshape(
        ensure_float32(slice(xbc_conv_all, {0, 0, d_inner}, {1, seq_len, d_inner + bc_len})),
        {seq_len, n_groups, d_state}
    );
    const array c_groups_all = reshape(
        ensure_float32(slice(xbc_conv_all, {0, 0, d_inner + bc_len}, {1, seq_len, d_inner + 2 * bc_len})),
        {seq_len, n_groups, d_state}
    );

    const array x_heads_group_all = reshape(x_conv_all, {seq_len, n_groups, heads_per_group, d_head});
    const array dt5_all = reshape(dt_all, {seq_len, n_groups, heads_per_group, 1, 1});
    array ssm_state_group = reshape(ssm_state, {1, n_groups, heads_per_group, d_head, d_state});

    const array dA_all = exp((-a_log_exp_group) * dt5_all);
    const array x_term_all = dt5_all *
        reshape(x_heads_group_all, {seq_len, n_groups, heads_per_group, d_head, 1}) *
        reshape(b_groups_all, {seq_len, n_groups, 1, 1, d_state});

    // Solve s_t = dA_t * s_{t-1} + x_t with one recurrence path for all
    // sequence lengths, including decode where seq_len == 1.
    const array prefix = cumprod(dA_all, 0);
    const array prefix_safe = clip(prefix, array(1.0e-20f), std::nullopt);
    const array ssm_terms = x_term_all / prefix_safe;
    const array ssm_accum = cumsum(ssm_terms, 0);
    array ssm_all_group = prefix * (ssm_state_group + ssm_accum);
    ssm_state_group = slice(
        ssm_all_group,
        {seq_len - 1, 0, 0, 0, 0},
        {seq_len, n_groups, heads_per_group, d_head, d_state}
    );

    ssm_state = reshape(ssm_state_group, {1, n_heads, d_head, d_state});

    array y_group_all = sum(
        ssm_all_group * reshape(c_groups_all, {seq_len, n_groups, 1, 1, d_state}),
        4
    );
    y_group_all = y_group_all + d_skip_group * x_heads_group_all;
    array y_all = reshape(y_group_all, {seq_len, n_heads, d_head});
    array ssm_out = reshape(y_all, {1, seq_len, d_inner});
    ssm_out = ssm_out * (z_all * sigmoid(z_all));
    if (norm_weight) {
        const array ssm_norm_weight = canonicalize_rms_norm_weight(*norm_weight, d_inner, "state_space.norm_weight");
        ssm_out = fast::rms_norm(ssm_out, ssm_norm_weight, 1e-5f);
    }

    // FP32 allowlist ends at recurrent-state math; projections stay in model
    // dtype to use bf16/f16 tensor-core kernels.
    const array ssm_out_proj_in = (ssm_out.dtype() == model_dtype) ? ssm_out : astype(ssm_out, model_dtype);
    const array mixer_out = state_space_call(out_proj_fn, ssm_out_proj_in);
    array hidden_1 = input + mixer_out * residual_multiplier;

    if (!has_ffn) {
        return hidden_1;
    }
    if (!ln2_weight.has_value()) {
        throw std::invalid_argument("[state_space] missing ln2 weight for FFN path");
    }

    const array ln2_norm_weight = canonicalize_rms_norm_weight(*ln2_weight, hidden_1.shape(2), "state_space.ln2_weight");
    const array norm2 = fast::rms_norm(hidden_1, ln2_norm_weight, norm_eps);
    const array norm2_proj_in = (norm2.dtype() == model_dtype) ? norm2 : astype(norm2, model_dtype);
    const array gate_up = state_space_call(gate_up_fn, norm2_proj_in);
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
    const array ffn_out = state_space_call(down_proj_fn, mid_proj_in);
    return hidden_1 + ffn_out * residual_multiplier;
}

static inline array gated_delta_mixer_forward_core(
    const array& input,
    const StateSpaceCallable& in_proj_fn,
    const array& conv_kernel_broadcast,
    const std::optional<array>& conv_bias_row,
    const array& a_log_exp_row,
    const std::optional<array>& dt_bias_row,
    const std::optional<array>& norm_weight_heads,
    const StateSpaceCallable& out_proj_fn,
    int d_conv,
    int n_heads,
    int n_key_heads,
    int d_head,
    array& conv_state,
    array& ssm_state,
    GatedDeltaCaptureOutputs* capture_out
) {
    const int seq_len = input.shape(1);
    const Dtype model_dtype = input.dtype();
    const int d_inner = n_heads * d_head;
    const int qkv_len = conv_kernel_broadcast.shape(2);
    if (qkv_len <= d_inner) {
        throw std::invalid_argument("[gated_delta] conv-derived qkv width must exceed value width");
    }
    const int qk_total = qkv_len - d_inner;
    if ((qk_total % 2) != 0) {
        throw std::invalid_argument("[gated_delta] qkv width must split evenly into q/k/value");
    }
    const int qk_len = qk_total / 2;
    if ((qk_len % d_head) != 0) {
        throw std::invalid_argument("[gated_delta] q width must be divisible by d_head");
    }
    const int n_kh = qk_len / d_head;
    if (n_kh <= 0 || (n_heads % n_kh) != 0) {
        throw std::invalid_argument("[gated_delta] conv-derived q heads incompatible with value heads");
    }
    // CPU treats configured n_key_heads as part of the contract.
    // Fail loudly if conv-derived projection shape disagrees.
    if (n_key_heads > 0 && n_kh != n_key_heads) {
        throw std::invalid_argument("[gated_delta] conv-derived q heads mismatch configured n_key_heads");
    }
    const int proj_len = qkv_len + d_inner + (2 * n_heads);
    if (n_heads <= 0 || d_head <= 0 || d_conv <= 0) {
        throw std::invalid_argument("[gated_delta] invalid config dims");
    }

    // Projection matmuls stay in model dtype for bf16/f16 tensor-core throughput.
    // FP32 upcasting happens after projection, before recurrent state math.
    const array proj_all = state_space_call(in_proj_fn, input);
    if (proj_all.shape(2) != proj_len) {
        throw std::invalid_argument("[gated_delta] in_proj output width mismatch");
    }

    // Keep qkv in model dtype through causal-conv for throughput.
    array qkv_all = slice(proj_all, {0, 0, 0}, {1, seq_len, qkv_len});
    const array z_all = ensure_float32(slice(proj_all, {0, 0, qkv_len}, {1, seq_len, qkv_len + d_inner}));
    const array beta_raw_all = ensure_float32(slice(
        proj_all,
        {0, 0, qkv_len + d_inner},
        {1, seq_len, qkv_len + d_inner + n_heads}
    ));
    const array a_raw_all = ensure_float32(slice(
        proj_all,
        {0, 0, qkv_len + d_inner + n_heads},
        {1, seq_len, proj_len}
    ));

    // One causal-conv path for any sequence length (no seq_len-specialized
    // dispatch), so prefill/decode share identical implementation structure.
    array conv_seq = causal_conv_run_sequence(
        &conv_state,
        qkv_all,
        conv_kernel_broadcast,
        seq_len,
        d_conv,
        qkv_len
    );
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, qkv_len});
    }
    if (capture_out != nullptr) {
        capture_out->in_proj = proj_all;
        capture_out->conv = conv_seq;
    }
    const bool use_reference_recurrence = gated_delta_reference_enabled();
    // SiLU then upcast to FP32 once for the entire sequence. This avoids
    // per-token ensure_float32 calls inside the recurrence loop.
    const array qkv_conv_all = use_reference_recurrence
        ? gated_delta_apply_fast_silu_reference(conv_seq)
        : ensure_float32(conv_seq * sigmoid(conv_seq));

    const std::optional<array> dt_bias_heads = dt_bias_row
        ? std::optional<array>(contiguous(reshape(*dt_bias_row, {n_heads})))
        : std::nullopt;
    const array dt_bias_heads_value = dt_bias_heads.has_value() ? *dt_bias_heads : zeros({n_heads}, float32);
    const array norm_weight_heads_value = norm_weight_heads.has_value() ? *norm_weight_heads : ones({n_heads, d_head}, float32);

    array state = reshape(ssm_state, {n_heads, d_head, d_head});

    // Slice Q and K (n_kh heads each) and V (n_heads value heads).
    array query_kh = reshape(slice(qkv_conv_all, {0, 0, 0}, {1, seq_len, qk_len}), {seq_len, n_kh, d_head});
    array key_kh   = reshape(slice(qkv_conv_all, {0, 0, qk_len}, {1, seq_len, 2 * qk_len}), {seq_len, n_kh, d_head});
    const array value_all = reshape(slice(qkv_conv_all, {0, 0, 2 * qk_len}, {1, seq_len, qkv_len}), {seq_len, n_heads, d_head});
    const array z_heads_all = reshape(z_all, {seq_len, n_heads, d_head});

    // Expand Q and K from n_kh to n_heads when using GQA (groups_per_key > 1).
    // Each KV head is shared by groups_per_key value heads, so we need interleaved
    // repeat: [k0,k0,k1,k1,...] not block repeat [k0,k1,...,k0,k1,...].
    const int groups_per_key = n_heads / n_kh;
    array query_all = (groups_per_key > 1)
        ? reshape(tile(reshape(query_kh, {seq_len, n_kh, 1, d_head}), {1, 1, groups_per_key, 1}), {seq_len, n_heads, d_head})
        : query_kh;
    array key_all = (groups_per_key > 1)
        ? reshape(tile(reshape(key_kh, {seq_len, n_kh, 1, d_head}), {1, 1, groups_per_key, 1}), {seq_len, n_heads, d_head})
        : key_kh;

    std::vector<array> outputs;
    outputs.reserve(static_cast<size_t>(seq_len));
    std::vector<array> ssm_outputs;
    std::vector<array> norm_outputs;
    if (capture_out != nullptr) {
        ssm_outputs.reserve(static_cast<size_t>(seq_len));
        norm_outputs.reserve(static_cast<size_t>(seq_len));
    }
    for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
        array query = reshape(slice(query_all, {token_idx, 0, 0}, {token_idx + 1, n_heads, d_head}), {n_heads, d_head});
        array key = reshape(slice(key_all, {token_idx, 0, 0}, {token_idx + 1, n_heads, d_head}), {n_heads, d_head});
        const array value = reshape(slice(value_all, {token_idx, 0, 0}, {token_idx + 1, n_heads, d_head}), {n_heads, d_head});
        const array z = reshape(slice(z_heads_all, {token_idx, 0, 0}, {token_idx + 1, n_heads, d_head}), {n_heads, d_head});
        const array beta_raw = reshape(slice(beta_raw_all, {0, token_idx, 0}, {1, token_idx + 1, n_heads}), {n_heads});
        const array a_raw = reshape(slice(a_raw_all, {0, token_idx, 0}, {1, token_idx + 1, n_heads}), {n_heads});
        array out(0.0f, float32);
        if (use_reference_recurrence) {
            const auto ref_step = gated_delta_recurrence_step_reference(
                state,
                query,
                key,
                value,
                z,
                beta_raw,
                a_raw,
                a_log_exp_row,
                dt_bias_heads_value,
                norm_weight_heads_value,
                n_heads,
                d_head
            );
            state = ref_step.next_state;
            out = ref_step.norm;
            if (capture_out != nullptr) {
                ssm_outputs.emplace_back(reshape(ref_step.ssm, {1, 1, d_inner}));
                norm_outputs.emplace_back(reshape(ref_step.norm, {1, 1, d_inner}));
            }
        } else if (capture_out != nullptr) {
            const array ssm_probe = gated_delta_recurrence_ssm_probe(
                state,
                query,
                key,
                value,
                beta_raw,
                a_raw,
                a_log_exp_row,
                dt_bias_heads_value,
                n_heads,
                d_head
            );
            const auto [normed_out, next_state] = gated_delta_recurrence_step(
                state,
                query,
                key,
                value,
                z,
                beta_raw,
                a_raw,
                a_log_exp_row,
                dt_bias_heads_value,
                norm_weight_heads_value,
                n_heads,
                d_head
            );
            state = next_state;
            out = normed_out;
            ssm_outputs.emplace_back(reshape(ssm_probe, {1, 1, d_inner}));
            norm_outputs.emplace_back(reshape(normed_out, {1, 1, d_inner}));
        } else {
            const auto [normed_out, next_state] = gated_delta_recurrence_step(
                state,
                query,
                key,
                value,
                z,
                beta_raw,
                a_raw,
                a_log_exp_row,
                dt_bias_heads_value,
                norm_weight_heads_value,
                n_heads,
                d_head
            );
            state = next_state;
            out = normed_out;
        }

        // Keep recurrence output in fp32 for out_proj to avoid precision loss
        // accumulation across long decode runs.
        const array out_proj_in = reshape(out, {1, 1, d_inner});
        outputs.emplace_back(state_space_call(out_proj_fn, out_proj_in));
    }

    ssm_state = reshape(state, {1, n_heads, d_head, d_head});
    if (outputs.empty()) {
        throw std::invalid_argument("[gated_delta] empty sequence");
    }
    if (capture_out != nullptr) {
        if (ssm_outputs.empty() || norm_outputs.empty()) {
            throw std::invalid_argument("[gated_delta] empty debug outputs");
        }
        capture_out->ssm = (ssm_outputs.size() == 1)
            ? ssm_outputs.front()
            : concatenate(ssm_outputs, 1);
        capture_out->norm = (norm_outputs.size() == 1)
            ? norm_outputs.front()
            : concatenate(norm_outputs, 1);
        capture_out->state_conv = conv_state;
        capture_out->state_ssm = ssm_state;
    }
    if (outputs.size() == 1) {
        return outputs.front();
    }
    return concatenate(outputs, 1);
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
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto d_skip_arr = cast_tensor_cached(d_skip, *static_cast<const array*>(d_skip), float32);
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
    const int heads_per_group = n_heads_i / n_groups_i;
    const int d_inner = n_heads_i * d_head_i;
    const bool use_reference_recurrence = gated_delta_reference_enabled();
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;
    const auto static_terms = resolve_state_space_static_terms(
        a_log,
        d_skip,
        a_log_arr,
        d_skip_arr,
        n_groups_i,
        heads_per_group
    );

    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, xbc_len, input_arr.dtype(), 0))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(*ln2_w)
        : std::nullopt;

    auto* cache = static_cast<MLXStateSpaceCache*>(state_space_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[state_space] cache layer index out of range");
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len ||
            layer_state->conv_state->dtype() != input_arr.dtype();
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, input_arr.dtype()));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, input_arr.dtype());
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

    const int hidden_dim = input_arr.shape(2);
    const int d_inner_i = n_heads_i * d_head_i;

    array in_proj_rhs(0.0f, input_arr.dtype());
    array out_proj_rhs(0.0f, input_arr.dtype());

    const bool has_ffn = gate_up_w != nullptr && down_proj_w != nullptr && ln2_w != nullptr;
    std::optional<array> gate_up_rhs = std::nullopt;
    std::optional<array> down_proj_rhs = std::nullopt;

    if (layer_state != nullptr) {
        const bool need_proj_rhs_refresh =
            layer_state->in_proj_rhs == nullptr ||
            layer_state->out_proj_rhs == nullptr ||
            layer_state->in_proj_handle != in_proj ||
            layer_state->out_proj_handle != out_proj ||
            layer_state->hidden_dim != hidden_dim ||
            layer_state->d_inner != d_inner_i;
        if (need_proj_rhs_refresh) {
            in_proj_rhs = orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
            out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner_i, hidden_dim, true);
            delete layer_state->in_proj_rhs;
            delete layer_state->out_proj_rhs;
            layer_state->in_proj_rhs = new array(in_proj_rhs);
            layer_state->out_proj_rhs = new array(out_proj_rhs);
            layer_state->in_proj_handle = in_proj;
            layer_state->out_proj_handle = out_proj;
            layer_state->hidden_dim = hidden_dim;
            layer_state->d_inner = d_inner_i;
        } else {
            in_proj_rhs = *layer_state->in_proj_rhs;
            out_proj_rhs = *layer_state->out_proj_rhs;
        }
    } else {
        in_proj_rhs = orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
        out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner_i, hidden_dim, true);
    }

    if (has_ffn) {
        array gate_rhs(0.0f, input_arr.dtype());
        if (layer_state != nullptr) {
            const bool need_ffn_rhs_refresh =
                layer_state->gate_up_rhs == nullptr ||
                layer_state->down_proj_rhs == nullptr ||
                layer_state->gate_up_handle != gate_up ||
                layer_state->down_proj_handle != down_proj ||
                layer_state->hidden_dim != hidden_dim;
            if (!need_ffn_rhs_refresh) {
                gate_rhs = *layer_state->gate_up_rhs;
            } else {
                gate_rhs = orient_matmul_rhs_cached(gate_up, *gate_up_w, hidden_dim, std::nullopt, true);
                const int gate_width = gate_rhs.shape(1);
                if ((gate_width % 2) != 0) {
                    throw std::invalid_argument("[state_space] gate_up output width must be even");
                }
                const int d_ff = gate_width / 2;
                const array down_rhs = orient_matmul_rhs_cached(down_proj, *down_proj_w, d_ff, hidden_dim, true);
                delete layer_state->gate_up_rhs;
                delete layer_state->down_proj_rhs;
                layer_state->gate_up_rhs = new array(gate_rhs);
                layer_state->down_proj_rhs = new array(down_rhs);
                layer_state->gate_up_handle = gate_up;
                layer_state->down_proj_handle = down_proj;
                layer_state->d_ff = d_ff;
            }
            if (layer_state->gate_up_rhs != nullptr) {
                gate_up_rhs = *layer_state->gate_up_rhs;
            }
            if (layer_state->down_proj_rhs != nullptr) {
                down_proj_rhs = *layer_state->down_proj_rhs;
            }
        } else {
            gate_rhs = orient_matmul_rhs_cached(gate_up, *gate_up_w, hidden_dim, std::nullopt, true);
            const int gate_width = gate_rhs.shape(1);
            if ((gate_width % 2) != 0) {
                throw std::invalid_argument("[state_space] gate_up output width must be even");
            }
            const int d_ff = gate_width / 2;
            const array down_rhs = orient_matmul_rhs_cached(down_proj, *down_proj_w, d_ff, hidden_dim, true);
            gate_up_rhs = gate_rhs;
            down_proj_rhs = down_rhs;
        }
    } else if (layer_state != nullptr) {
        delete layer_state->gate_up_rhs;
        delete layer_state->down_proj_rhs;
        layer_state->gate_up_rhs = nullptr;
        layer_state->down_proj_rhs = nullptr;
        layer_state->gate_up_handle = nullptr;
        layer_state->down_proj_handle = nullptr;
        layer_state->d_ff = 0;
    }

    StateSpaceCallable in_proj_fn{
        .fn = state_space_apply_matmul_rhs,
        .ctx = &in_proj_rhs,
    };
    StateSpaceCallable out_proj_fn{
        .fn = state_space_apply_matmul_rhs,
        .ctx = &out_proj_rhs,
    };
    StateSpaceCallable gate_up_fn{};
    StateSpaceCallable down_proj_fn{};
    if (has_ffn) {
        if (!gate_up_rhs.has_value()) {
            throw std::invalid_argument("[state_space] missing gate_up rhs");
        }
        if (!down_proj_rhs.has_value()) {
            throw std::invalid_argument("[state_space] missing down_proj rhs");
        }
        gate_up_fn = StateSpaceCallable{
            .fn = state_space_apply_matmul_rhs,
            .ctx = &gate_up_rhs.value(),
        };
        down_proj_fn = StateSpaceCallable{
            .fn = state_space_apply_matmul_rhs,
            .ctx = &down_proj_rhs.value(),
        };
    }

    array out = state_space_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms.a_log_exp_group,
        static_terms.d_skip_group,
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
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}

void* mlx_lazy_gated_delta_mixer_bf16(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_proj,
    void* gated_delta_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t n_heads,
    size_t n_key_heads,
    size_t d_head
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_w = *static_cast<const array*>(in_proj);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_proj);

    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int n_key_heads_i = static_cast<int>(n_key_heads > 0 ? n_key_heads : n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int hidden_dim = input_arr.shape(2);
    const int d_inner = n_heads_i * d_head_i;
    const bool use_reference_recurrence = gated_delta_reference_enabled();
    // Derive qkv width from conv weight shape (matches CPU behavior).
    const int qkv_len = infer_state_space_conv_dim_from_weight(conv_w, d_conv_i, "gated_delta");

    auto* cache = static_cast<MLXStateSpaceCache*>(gated_delta_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[gated_delta] cache layer index out of range");
    }
    array conv_kernel_broadcast(0.0f, input_arr.dtype());
    std::optional<array> conv_bias_row = std::nullopt;
    std::optional<array> dt_bias_row = std::nullopt;
    std::optional<GatedDeltaStaticTerms> static_terms = std::nullopt;
    if (layer_state != nullptr) {
        const bool need_static_refresh =
            layer_state->gdelta_conv_kernel == nullptr ||
            layer_state->gdelta_a_log_exp_row == nullptr ||
            layer_state->gdelta_conv_weight_handle != conv_weight ||
            layer_state->gdelta_conv_bias_handle != conv_bias ||
            layer_state->gdelta_dt_bias_handle != dt_bias ||
            layer_state->gdelta_a_log_handle != a_log ||
            layer_state->gdelta_norm_weight_handle != norm_weight ||
            layer_state->gdelta_qkv_len != qkv_len ||
            layer_state->gdelta_n_heads != n_heads_i ||
            layer_state->gdelta_d_head != d_head_i;
        if (need_static_refresh) {
            // Invalidate compiled decode graph — weight handles or shapes changed.
            delete layer_state->gdelta_compiled_fn;
            layer_state->gdelta_compiled_fn = nullptr;
            const std::optional<array> norm_weight_arr = (norm_w != nullptr)
                ? std::optional<array>(*norm_w)
                : std::nullopt;
            conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, qkv_len);
            conv_bias_row = (conv_b != nullptr)
                ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, qkv_len, input_arr.dtype(), 0))
                : std::nullopt;
            dt_bias_row = (dt_b != nullptr)
                ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
                : std::nullopt;
            static_terms.emplace(resolve_gated_delta_static_terms(
                a_log,
                norm_weight,
                a_log_arr,
                norm_weight_arr,
                n_heads_i,
                d_head_i
            ));
            delete layer_state->gdelta_conv_kernel;
            delete layer_state->gdelta_conv_bias_row;
            delete layer_state->gdelta_dt_bias_row;
            delete layer_state->gdelta_a_log_exp_row;
            delete layer_state->gdelta_norm_weight_heads;
            layer_state->gdelta_conv_kernel = new array(conv_kernel_broadcast);
            layer_state->gdelta_conv_bias_row = conv_bias_row.has_value() ? new array(*conv_bias_row) : nullptr;
            layer_state->gdelta_dt_bias_row = dt_bias_row.has_value() ? new array(*dt_bias_row) : nullptr;
            layer_state->gdelta_a_log_exp_row = new array(static_terms->a_log_exp_row);
            layer_state->gdelta_norm_weight_heads = static_terms->norm_weight_heads.has_value()
                ? new array(*static_terms->norm_weight_heads)
                : nullptr;
            layer_state->gdelta_conv_weight_handle = conv_weight;
            layer_state->gdelta_conv_bias_handle = conv_bias;
            layer_state->gdelta_dt_bias_handle = dt_bias;
            layer_state->gdelta_a_log_handle = a_log;
            layer_state->gdelta_norm_weight_handle = norm_weight;
            layer_state->gdelta_qkv_len = qkv_len;
            layer_state->gdelta_n_heads = n_heads_i;
            layer_state->gdelta_d_head = d_head_i;
        } else {
            conv_kernel_broadcast = *layer_state->gdelta_conv_kernel;
            conv_bias_row = (layer_state->gdelta_conv_bias_row != nullptr)
                ? std::optional<array>(*layer_state->gdelta_conv_bias_row)
                : std::nullopt;
            dt_bias_row = (layer_state->gdelta_dt_bias_row != nullptr)
                ? std::optional<array>(*layer_state->gdelta_dt_bias_row)
                : std::nullopt;
            static_terms = GatedDeltaStaticTerms{
                *layer_state->gdelta_a_log_exp_row,
                (layer_state->gdelta_norm_weight_heads != nullptr)
                    ? std::optional<array>(*layer_state->gdelta_norm_weight_heads)
                    : std::nullopt,
            };
        }
    } else {
        const std::optional<array> norm_weight_arr = (norm_w != nullptr)
            ? std::optional<array>(*norm_w)
            : std::nullopt;
        conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, qkv_len);
        conv_bias_row = (conv_b != nullptr)
            ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, qkv_len, input_arr.dtype(), 0))
            : std::nullopt;
        dt_bias_row = (dt_b != nullptr)
            ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
            : std::nullopt;
        static_terms.emplace(resolve_gated_delta_static_terms(
            a_log,
            norm_weight,
            a_log_arr,
            norm_weight_arr,
            n_heads_i,
            d_head_i
        ));
    }

    const Dtype model_dtype = input_arr.dtype();
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != qkv_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, qkv_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, qkv_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_head_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_head_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_head_i}, float32);
    }

    array in_proj_rhs(0.0f, model_dtype);
    array out_proj_rhs(0.0f, model_dtype);
    if (layer_state != nullptr) {
        const bool need_proj_rhs_refresh =
            layer_state->in_proj_rhs == nullptr ||
            layer_state->out_proj_rhs == nullptr ||
            layer_state->in_proj_handle != in_proj ||
            layer_state->out_proj_handle != out_proj ||
            layer_state->hidden_dim != hidden_dim ||
            layer_state->d_inner != d_inner;
        if (need_proj_rhs_refresh) {
            // Projection RHS changed; invalidate compiled decode graph.
            delete layer_state->gdelta_compiled_fn;
            layer_state->gdelta_compiled_fn = nullptr;
            in_proj_rhs = use_reference_recurrence
                ? orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true)
                : orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
            out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner, hidden_dim, true);
            delete layer_state->in_proj_rhs;
            delete layer_state->out_proj_rhs;
            layer_state->in_proj_rhs = new array(in_proj_rhs);
            layer_state->out_proj_rhs = new array(out_proj_rhs);
            layer_state->in_proj_handle = in_proj;
            layer_state->out_proj_handle = out_proj;
            layer_state->hidden_dim = hidden_dim;
            layer_state->d_inner = d_inner;
        } else {
            in_proj_rhs = *layer_state->in_proj_rhs;
            out_proj_rhs = *layer_state->out_proj_rhs;
        }
    } else {
        in_proj_rhs = use_reference_recurrence
            ? orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true)
            : orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
        out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner, hidden_dim, true);
    }

    // Compile-once: on first single-token decode after any weight refresh, build a
    // compiled MLX function that caches the gated-delta forward graph topology.
    // Restricted to seq_len==1 (decode) to avoid tracing the full prefill loop.
    const bool is_single_token_decode = (input_arr.shape(1) == 1);
    if (!use_reference_recurrence and is_single_token_decode
            and layer_state != nullptr and layer_state->gdelta_compiled_fn == nullptr) {
        const array ck = *layer_state->gdelta_conv_kernel;
        const std::optional<array> cb_cap = layer_state->gdelta_conv_bias_row
            ? std::optional<array>(*layer_state->gdelta_conv_bias_row) : std::nullopt;
        const std::optional<array> db_cap = layer_state->gdelta_dt_bias_row
            ? std::optional<array>(*layer_state->gdelta_dt_bias_row) : std::nullopt;
        const array al = *layer_state->gdelta_a_log_exp_row;
        const std::optional<array> nw_cap = layer_state->gdelta_norm_weight_heads
            ? std::optional<array>(*layer_state->gdelta_norm_weight_heads) : std::nullopt;
        const array ip = *layer_state->in_proj_rhs;
        const array op_cap = *layer_state->out_proj_rhs;
        const int dc = d_conv_i, nh = n_heads_i, nkh = n_key_heads_i, dh = d_head_i;
        auto decode_fn = [ck, cb_cap, db_cap, al, nw_cap, ip, op_cap, dc, nh, nkh, dh](
            const std::vector<array>& inputs
        ) -> std::vector<array> {
            array x = inputs[0];
            array cs = inputs[1];
            array ss = inputs[2];
            array in_r = ip;
            array out_r = op_cap;
            StateSpaceCallable in_fn_c{.fn = state_space_apply_matmul_rhs, .ctx = &in_r};
            StateSpaceCallable out_fn_c{.fn = state_space_apply_matmul_rhs, .ctx = &out_r};
            const array result = gated_delta_mixer_forward_core(
                x, in_fn_c, ck, cb_cap, al, db_cap, nw_cap, out_fn_c,
                dc, nh, nkh, dh, cs, ss, nullptr);
            return {result, cs, ss};
        };
        layer_state->gdelta_compiled_fn = new std::function<std::vector<array>(
            const std::vector<array>&)>(mlx::core::compile(decode_fn));
    }
    // Fast path: use compiled graph to avoid rebuilding decode topology.
    if (!use_reference_recurrence and is_single_token_decode
            and layer_state != nullptr and layer_state->gdelta_compiled_fn != nullptr) {
        const std::vector<array> compiled_out = (*layer_state->gdelta_compiled_fn)(
            std::vector<array>{input_arr, conv_state, ssm_state});
        const array out = compiled_out[0];
        conv_state = compiled_out[1];
        ssm_state = compiled_out[2];
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
        return pool_array(out);
    }

    StateSpaceCallable in_proj_fn{
        .fn = state_space_apply_matmul_rhs,
        .ctx = &in_proj_rhs,
    };
    StateSpaceCallable out_proj_fn{
        .fn = state_space_apply_matmul_rhs,
        .ctx = &out_proj_rhs,
    };
    const array out = gated_delta_mixer_forward_core(
        input_arr,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms->a_log_exp_row,
        dt_bias_row,
        static_terms->norm_weight_heads,
        out_proj_fn,
        d_conv_i,
        n_heads_i,
        n_key_heads_i,
        d_head_i,
        conv_state,
        ssm_state,
        nullptr
    );

    if (layer_state != nullptr) {
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }
    return pool_array(out);
}


void* mlx_lazy_gated_delta_mixer_quantized(
    const void* input,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    size_t group_size,
    size_t bits,
    void* gated_delta_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t n_heads,
    size_t n_key_heads,
    size_t d_head
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_w = *static_cast<const array*>(in_w);
    const auto& in_proj_s = *static_cast<const array*>(in_s);
    const auto& in_proj_b = *static_cast<const array*>(in_b);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_w);
    const auto& out_proj_s = *static_cast<const array*>(out_s);
    const auto& out_proj_b = *static_cast<const array*>(out_b);

    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int n_key_heads_i = static_cast<int>(n_key_heads > 0 ? n_key_heads : n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int d_inner = n_heads_i * d_head_i;
    const bool use_reference_recurrence = gated_delta_reference_enabled();
    // Derive qkv width from conv weight shape (matches CPU behavior).
    const int qkv_len = infer_state_space_conv_dim_from_weight(conv_w, d_conv_i, "gated_delta");
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    auto* cache = static_cast<MLXStateSpaceCache*>(gated_delta_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[gated_delta] cache layer index out of range");
    }
    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, qkv_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, qkv_len, input_arr.dtype(), 0))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const auto static_terms = resolve_gated_delta_static_terms(
        a_log,
        norm_weight,
        a_log_arr,
        norm_weight_arr,
        n_heads_i,
        d_head_i
    );

    const Dtype model_dtype = input_arr.dtype();
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != qkv_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, qkv_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, qkv_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_head_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_head_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_head_i}, float32);
    }

    StateSpaceQuantLinearCtx in_proj_ctx{
        .w = &in_proj_w,
        .s = &in_proj_s,
        .b = &in_proj_b,
        .s_handle = in_s,
        .b_handle = in_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceQuantLinearCtx out_proj_ctx{
        .w = &out_proj_w,
        .s = &out_proj_s,
        .b = &out_proj_b,
        .s_handle = out_s,
        .b_handle = out_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceCallable in_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &in_proj_ctx,
    };
    StateSpaceCallable out_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &out_proj_ctx,
    };
    const array out = gated_delta_mixer_forward_core(
        input_arr,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms.a_log_exp_row,
        dt_bias_row,
        static_terms.norm_weight_heads,
        out_proj_fn,
        d_conv_i,
        n_heads_i,
        n_key_heads_i,
        d_head_i,
        conv_state,
        ssm_state,
        nullptr
    );

    if (layer_state != nullptr) {
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }
    return pool_array(out);
}

void* mlx_lazy_gated_delta_mixer_bf16_capture(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_proj,
    void* gated_delta_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t n_heads,
    size_t n_key_heads,
    size_t d_head,
    void** out_in_proj,
    void** out_conv,
    void** out_ssm,
    void** out_norm,
    void** out_state_conv,
    void** out_state_ssm
) {
    mlx_count_op();
    if (out_in_proj != nullptr) *out_in_proj = nullptr;
    if (out_conv != nullptr) *out_conv = nullptr;
    if (out_ssm != nullptr) *out_ssm = nullptr;
    if (out_norm != nullptr) *out_norm = nullptr;
    if (out_state_conv != nullptr) *out_state_conv = nullptr;
    if (out_state_ssm != nullptr) *out_state_ssm = nullptr;

    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_w = *static_cast<const array*>(in_proj);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_proj);

    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int n_key_heads_i = static_cast<int>(n_key_heads > 0 ? n_key_heads : n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int hidden_dim = input_arr.shape(2);
    const int d_inner = n_heads_i * d_head_i;
    const bool use_reference_recurrence = gated_delta_reference_enabled();
    const int qkv_len = infer_state_space_conv_dim_from_weight(conv_w, d_conv_i, "gated_delta");

    auto* cache = static_cast<MLXStateSpaceCache*>(gated_delta_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[gated_delta] cache layer index out of range");
    }
    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, qkv_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, qkv_len, input_arr.dtype(), 0))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const auto static_terms = resolve_gated_delta_static_terms(
        a_log,
        norm_weight,
        a_log_arr,
        norm_weight_arr,
        n_heads_i,
        d_head_i
    );

    const Dtype model_dtype = input_arr.dtype();
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != qkv_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, qkv_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, qkv_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_head_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_head_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_head_i}, float32);
    }

    array in_proj_rhs(0.0f, model_dtype);
    array out_proj_rhs(0.0f, model_dtype);
    if (layer_state != nullptr) {
        const bool need_proj_rhs_refresh =
            layer_state->in_proj_rhs == nullptr ||
            layer_state->out_proj_rhs == nullptr ||
            layer_state->in_proj_handle != in_proj ||
            layer_state->out_proj_handle != out_proj ||
            layer_state->hidden_dim != hidden_dim ||
            layer_state->d_inner != d_inner;
        if (need_proj_rhs_refresh) {
            in_proj_rhs = use_reference_recurrence
                ? orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true)
                : orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
            out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner, hidden_dim, true);
            delete layer_state->in_proj_rhs;
            delete layer_state->out_proj_rhs;
            layer_state->in_proj_rhs = new array(in_proj_rhs);
            layer_state->out_proj_rhs = new array(out_proj_rhs);
            layer_state->in_proj_handle = in_proj;
            layer_state->out_proj_handle = out_proj;
            layer_state->hidden_dim = hidden_dim;
            layer_state->d_inner = d_inner;
        } else {
            in_proj_rhs = *layer_state->in_proj_rhs;
            out_proj_rhs = *layer_state->out_proj_rhs;
        }
    } else {
        in_proj_rhs = use_reference_recurrence
            ? orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true)
            : orient_matmul_rhs_cached(in_proj, in_proj_w, hidden_dim, std::nullopt, true);
        out_proj_rhs = orient_matmul_rhs_cached(out_proj, out_proj_w, d_inner, hidden_dim, true);
    }
    StateSpaceCallable in_proj_fn{
        .fn = use_reference_recurrence ? state_space_apply_linear_reference : state_space_apply_matmul_rhs,
        .ctx = use_reference_recurrence ? static_cast<const void*>(&in_proj_w) : static_cast<const void*>(&in_proj_rhs),
    };
    StateSpaceCallable out_proj_fn{
        .fn = use_reference_recurrence ? state_space_apply_linear_reference : state_space_apply_matmul_rhs,
        .ctx = use_reference_recurrence ? static_cast<const void*>(&out_proj_w) : static_cast<const void*>(&out_proj_rhs),
    };
    GatedDeltaCaptureOutputs capture_outputs{};
    const array out = gated_delta_mixer_forward_core(
        input_arr,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms.a_log_exp_row,
        dt_bias_row,
        static_terms.norm_weight_heads,
        out_proj_fn,
        d_conv_i,
        n_heads_i,
        n_key_heads_i,
        d_head_i,
        conv_state,
        ssm_state,
        &capture_outputs
    );
    materialize_gated_delta_capture_outputs(&capture_outputs);

    if (layer_state != nullptr) {
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }
    if (out_in_proj != nullptr) *out_in_proj = capture_outputs.in_proj.has_value() ? pool_array(*capture_outputs.in_proj) : nullptr;
    if (out_conv != nullptr) *out_conv = capture_outputs.conv.has_value() ? pool_array(*capture_outputs.conv) : nullptr;
    if (out_ssm != nullptr) *out_ssm = capture_outputs.ssm.has_value() ? pool_array(*capture_outputs.ssm) : nullptr;
    if (out_norm != nullptr) *out_norm = capture_outputs.norm.has_value() ? pool_array(*capture_outputs.norm) : nullptr;
    if (out_state_conv != nullptr) *out_state_conv = capture_outputs.state_conv.has_value() ? pool_array(*capture_outputs.state_conv) : nullptr;
    if (out_state_ssm != nullptr) *out_state_ssm = capture_outputs.state_ssm.has_value() ? pool_array(*capture_outputs.state_ssm) : nullptr;
    return pool_array(out);
}

void* mlx_lazy_gated_delta_mixer_quantized_capture(
    const void* input,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    size_t group_size,
    size_t bits,
    void* gated_delta_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t n_heads,
    size_t n_key_heads,
    size_t d_head,
    void** out_in_proj,
    void** out_conv,
    void** out_ssm,
    void** out_norm,
    void** out_state_conv,
    void** out_state_ssm
) {
    mlx_count_op();
    if (out_in_proj != nullptr) *out_in_proj = nullptr;
    if (out_conv != nullptr) *out_conv = nullptr;
    if (out_ssm != nullptr) *out_ssm = nullptr;
    if (out_norm != nullptr) *out_norm = nullptr;
    if (out_state_conv != nullptr) *out_state_conv = nullptr;
    if (out_state_ssm != nullptr) *out_state_ssm = nullptr;

    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_w = *static_cast<const array*>(in_w);
    const auto& in_proj_s = *static_cast<const array*>(in_s);
    const auto& in_proj_b = *static_cast<const array*>(in_b);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_w);
    const auto& out_proj_s = *static_cast<const array*>(out_s);
    const auto& out_proj_b = *static_cast<const array*>(out_b);

    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int n_key_heads_i = static_cast<int>(n_key_heads > 0 ? n_key_heads : n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int d_inner = n_heads_i * d_head_i;
    const int qkv_len = infer_state_space_conv_dim_from_weight(conv_w, d_conv_i, "gated_delta");
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    auto* cache = static_cast<MLXStateSpaceCache*>(gated_delta_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[gated_delta] cache layer index out of range");
    }
    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, qkv_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, qkv_len, input_arr.dtype(), 0))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const auto static_terms = resolve_gated_delta_static_terms(
        a_log,
        norm_weight,
        a_log_arr,
        norm_weight_arr,
        n_heads_i,
        d_head_i
    );

    const Dtype model_dtype = input_arr.dtype();
    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != qkv_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, qkv_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, qkv_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_head_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_head_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_head_i}, float32);
    }

    StateSpaceQuantLinearCtx in_proj_ctx{
        .w = &in_proj_w,
        .s = &in_proj_s,
        .b = &in_proj_b,
        .s_handle = in_s,
        .b_handle = in_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceQuantLinearCtx out_proj_ctx{
        .w = &out_proj_w,
        .s = &out_proj_s,
        .b = &out_proj_b,
        .s_handle = out_s,
        .b_handle = out_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceCallable in_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &in_proj_ctx,
    };
    StateSpaceCallable out_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &out_proj_ctx,
    };
    GatedDeltaCaptureOutputs capture_outputs{};
    const array out = gated_delta_mixer_forward_core(
        input_arr,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms.a_log_exp_row,
        dt_bias_row,
        static_terms.norm_weight_heads,
        out_proj_fn,
        d_conv_i,
        n_heads_i,
        n_key_heads_i,
        d_head_i,
        conv_state,
        ssm_state,
        &capture_outputs
    );
    materialize_gated_delta_capture_outputs(&capture_outputs);

    if (layer_state != nullptr) {
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }
    if (out_in_proj != nullptr) *out_in_proj = capture_outputs.in_proj.has_value() ? pool_array(*capture_outputs.in_proj) : nullptr;
    if (out_conv != nullptr) *out_conv = capture_outputs.conv.has_value() ? pool_array(*capture_outputs.conv) : nullptr;
    if (out_ssm != nullptr) *out_ssm = capture_outputs.ssm.has_value() ? pool_array(*capture_outputs.ssm) : nullptr;
    if (out_norm != nullptr) *out_norm = capture_outputs.norm.has_value() ? pool_array(*capture_outputs.norm) : nullptr;
    if (out_state_conv != nullptr) *out_state_conv = capture_outputs.state_conv.has_value() ? pool_array(*capture_outputs.state_conv) : nullptr;
    if (out_state_ssm != nullptr) *out_state_ssm = capture_outputs.state_ssm.has_value() ? pool_array(*capture_outputs.state_ssm) : nullptr;
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
    const auto a_log_arr = cast_tensor_cached(a_log, *static_cast<const array*>(a_log), float32);
    const auto d_skip_arr = cast_tensor_cached(d_skip, *static_cast<const array*>(d_skip), float32);
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
    const int heads_per_group = n_heads_i / n_groups_i;
    const int d_inner = n_heads_i * d_head_i;
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;
    const auto static_terms = resolve_state_space_static_terms(
        a_log,
        d_skip,
        a_log_arr,
        d_skip_arr,
        n_groups_i,
        heads_per_group
    );
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    const array conv_kernel_broadcast = resolve_state_space_conv_kernel(conv_weight, conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(conv_bias, *conv_b, xbc_len, input_arr.dtype(), 0))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(resolve_state_space_row_cached(dt_bias, *dt_b, n_heads_i, float32, 1))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(*norm_w)
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(*ln2_w)
        : std::nullopt;

    auto* cache = static_cast<MLXStateSpaceCache*>(state_space_cache_ptr);
    std::unique_lock<std::mutex> cache_lock;
    if (cache != nullptr) {
        cache_lock = std::unique_lock<std::mutex>(cache->mu);
    }
    StateSpaceLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }
    if (cache != nullptr && layer_state == nullptr) {
        throw std::invalid_argument("[state_space] cache layer index out of range");
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len ||
            layer_state->conv_state->dtype() != input_arr.dtype();
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, input_arr.dtype()));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, input_arr.dtype());
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

    StateSpaceQuantLinearCtx in_proj_ctx{
        .w = &in_proj_w,
        .s = &in_proj_s,
        .b = &in_proj_b,
        .s_handle = in_s,
        .b_handle = in_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceQuantLinearCtx out_proj_ctx{
        .w = &out_proj_w,
        .s = &out_proj_s,
        .b = &out_proj_b,
        .s_handle = out_s,
        .b_handle = out_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceCallable in_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &in_proj_ctx,
    };
    StateSpaceCallable out_proj_fn{
        .fn = state_space_apply_quantized_linear,
        .ctx = &out_proj_ctx,
    };
    const bool has_ffn =
        gate_up_w_arr != nullptr && gate_up_s_arr != nullptr && gate_up_b_arr != nullptr &&
        down_w_arr != nullptr && down_s_arr != nullptr && down_b_arr != nullptr &&
        ln2_w != nullptr;
    StateSpaceCallable gate_up_fn{};
    StateSpaceCallable down_proj_fn{};
    StateSpaceQuantLinearCtx gate_up_ctx{
        .w = gate_up_w_arr,
        .s = gate_up_s_arr,
        .b = gate_up_b_arr,
        .s_handle = gate_up_s,
        .b_handle = gate_up_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    StateSpaceQuantLinearCtx down_proj_ctx{
        .w = down_w_arr,
        .s = down_s_arr,
        .b = down_b_arr,
        .s_handle = down_s,
        .b_handle = down_b,
        .group_size = group_size_i,
        .bits = bits_i,
    };
    if (has_ffn) {
        gate_up_fn = StateSpaceCallable{
            .fn = state_space_apply_quantized_linear,
            .ctx = &gate_up_ctx,
        };
        down_proj_fn = StateSpaceCallable{
            .fn = state_space_apply_quantized_linear,
            .ctx = &down_proj_ctx,
        };
    }

    array out = state_space_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        static_terms.a_log_exp_group,
        static_terms.d_skip_group,
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
        async_eval(std::vector<array>{conv_state, ssm_state});
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}
