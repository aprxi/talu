// ============================================================================
// Fused Attention Block (Quantized)
// ============================================================================
// Combines: QKV projection -> reshape -> transpose -> QK norm -> RoPE ->
//           cache update -> attention -> reshape -> output projection
//
// This reduces ~15 FFI round-trips to 1.

struct ProjectedQkv {
    array q;
    array k;
    array v;
};

static constexpr size_t kAttentionFusedQkvCacheMaxEntries = 2048;
static constexpr size_t kAttentionFusedQkvHotSlots = 128;
static constexpr size_t kFfnQuantizedGateUpHotSlots = 64;

struct AttentionFusedQkvCacheKey {
    const void* q_weight = nullptr;
    const void* k_weight = nullptr;
    const void* v_weight = nullptr;
    int in_features = 0;
    int q_out = 0;
    int kv_out = 0;

    bool operator==(const AttentionFusedQkvCacheKey& other) const {
        return q_weight == other.q_weight &&
               k_weight == other.k_weight &&
               v_weight == other.v_weight &&
               in_features == other.in_features &&
               q_out == other.q_out &&
               kv_out == other.kv_out;
    }
};

struct AttentionFusedQkvCacheKeyHash {
    size_t operator()(const AttentionFusedQkvCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.q_weight);
        h ^= std::hash<const void*>{}(key.k_weight) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.v_weight) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.in_features) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.q_out) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.kv_out) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<AttentionFusedQkvCacheKey, array, AttentionFusedQkvCacheKeyHash>& attention_fused_qkv_cache_store() {
    static thread_local std::unordered_map<AttentionFusedQkvCacheKey, array, AttentionFusedQkvCacheKeyHash> cache;
    return cache;
}

struct AttentionFusedQkvHotSlot {
    bool valid = false;
    AttentionFusedQkvCacheKey key;
    array value = array(0.0f);
};

static inline AttentionFusedQkvHotSlot* attention_fused_qkv_hot_slots_store() {
    static thread_local AttentionFusedQkvHotSlot slots[kAttentionFusedQkvHotSlots];
    return slots;
}

static inline void attention_fused_qkv_cache_clear() {
    attention_fused_qkv_cache_store().clear();
    auto* hot_slots = attention_fused_qkv_hot_slots_store();
    for (size_t i = 0; i < kAttentionFusedQkvHotSlots; ++i) {
        hot_slots[i].valid = false;
    }
}

struct FfnQuantizedGateUpCacheValue {
    array weights;
    array scales;
    array biases;
    int d_ff = 0;
};

static constexpr size_t kFfnQuantizedGateUpCacheMaxEntries = 2048;

struct FfnQuantizedGateUpCacheKey {
    const void* gate_w = nullptr;
    const void* gate_s = nullptr;
    const void* gate_b = nullptr;
    const void* up_w = nullptr;
    const void* up_s = nullptr;
    const void* up_b = nullptr;

    bool operator==(const FfnQuantizedGateUpCacheKey& other) const {
        return gate_w == other.gate_w &&
               gate_s == other.gate_s &&
               gate_b == other.gate_b &&
               up_w == other.up_w &&
               up_s == other.up_s &&
               up_b == other.up_b;
    }
};

struct FfnQuantizedGateUpCacheKeyHash {
    size_t operator()(const FfnQuantizedGateUpCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.gate_w);
        h ^= std::hash<const void*>{}(key.gate_s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.gate_b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.up_w) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.up_s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.up_b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<FfnQuantizedGateUpCacheKey, FfnQuantizedGateUpCacheValue, FfnQuantizedGateUpCacheKeyHash>& ffn_quantized_gate_up_cache_store() {
    static thread_local std::unordered_map<FfnQuantizedGateUpCacheKey, FfnQuantizedGateUpCacheValue, FfnQuantizedGateUpCacheKeyHash> cache;
    return cache;
}

struct FfnQuantizedGateUpHotSlot {
    bool valid = false;
    FfnQuantizedGateUpCacheKey key;
    FfnQuantizedGateUpCacheValue value{
        array(0.0f),
        array(0.0f),
        array(0.0f),
        0,
    };
};

static inline FfnQuantizedGateUpHotSlot* ffn_quantized_gate_up_hot_slots_store() {
    static thread_local FfnQuantizedGateUpHotSlot slots[kFfnQuantizedGateUpHotSlots];
    return slots;
}

static inline void ffn_quantized_gate_up_cache_clear() {
    ffn_quantized_gate_up_cache_store().clear();
    auto* hot_slots = ffn_quantized_gate_up_hot_slots_store();
    for (size_t i = 0; i < kFfnQuantizedGateUpHotSlots; ++i) {
        hot_slots[i].valid = false;
    }
}

static inline const FfnQuantizedGateUpCacheValue& ffn_quantized_gate_up_cached(
    const void* gate_w,
    const void* gate_s,
    const void* gate_b,
    const void* up_w,
    const void* up_s,
    const void* up_b,
    const array& gate_w_arr,
    const array& gate_s_arr,
    const array& gate_b_arr,
    const array& up_w_arr,
    const array& up_s_arr,
    const array& up_b_arr
) {
    if (gate_w_arr.ndim() != 2 || up_w_arr.ndim() != 2) {
        throw std::invalid_argument("[ffn_quantized] gate/up weights must be rank-2");
    }
    if (gate_s_arr.ndim() != 2 || up_s_arr.ndim() != 2 || gate_b_arr.ndim() != 2 || up_b_arr.ndim() != 2) {
        throw std::invalid_argument("[ffn_quantized] gate/up scales and biases must be rank-2");
    }
    if (gate_w_arr.shape(1) != up_w_arr.shape(1)) {
        throw std::invalid_argument("[ffn_quantized] gate/up packed-k mismatch");
    }
    if (gate_s_arr.shape(1) != up_s_arr.shape(1) || gate_b_arr.shape(1) != up_b_arr.shape(1)) {
        throw std::invalid_argument("[ffn_quantized] gate/up group mismatch");
    }
    const int d_ff = gate_w_arr.shape(0);
    if (up_w_arr.shape(0) != d_ff || up_s_arr.shape(0) != d_ff || up_b_arr.shape(0) != d_ff) {
        throw std::invalid_argument("[ffn_quantized] gate/up out features mismatch");
    }

    if (gate_w == nullptr || gate_s == nullptr || gate_b == nullptr || up_w == nullptr || up_s == nullptr || up_b == nullptr) {
        static thread_local std::optional<FfnQuantizedGateUpCacheValue> scratch;
        scratch = FfnQuantizedGateUpCacheValue{
            contiguous(concatenate({gate_w_arr, up_w_arr}, 0)),
            contiguous(concatenate({gate_s_arr, up_s_arr}, 0)),
            contiguous(concatenate({gate_b_arr, up_b_arr}, 0)),
            d_ff,
        };
        return *scratch;
    }

    FfnQuantizedGateUpCacheKey key{
        gate_w,
        gate_s,
        gate_b,
        up_w,
        up_s,
        up_b,
    };
    auto* hot_slots = ffn_quantized_gate_up_hot_slots_store();
    const size_t hot_index = FfnQuantizedGateUpCacheKeyHash{}(key) & (kFfnQuantizedGateUpHotSlots - 1);
    auto& hot_slot = hot_slots[hot_index];
    if (hot_slot.valid && hot_slot.key == key) {
        return hot_slot.value;
    }

    auto& cache = ffn_quantized_gate_up_cache_store();
    if (cache.size() >= kFfnQuantizedGateUpCacheMaxEntries) {
        cache.clear();
        for (size_t i = 0; i < kFfnQuantizedGateUpHotSlots; ++i) {
            hot_slots[i].valid = false;
        }
    }

    auto it = cache.find(key);
    if (it != cache.end()) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.value = it->second;
        return it->second;
    }

    FfnQuantizedGateUpCacheValue fused{
        contiguous(concatenate({gate_w_arr, up_w_arr}, 0)),
        contiguous(concatenate({gate_s_arr, up_s_arr}, 0)),
        contiguous(concatenate({gate_b_arr, up_b_arr}, 0)),
        d_ff,
    };
    // Materialize once so repeated decode steps reuse concrete fused tensors.
    eval(fused.weights);
    eval(fused.scales);
    eval(fused.biases);
    auto [inserted, _] = cache.emplace(key, std::move(fused));
    hot_slot.valid = true;
    hot_slot.key = key;
    hot_slot.value = inserted->second;
    return inserted->second;
}

static inline array attention_fused_qkv_rhs_cached(
    const void* q_w,
    const void* k_w,
    const void* v_w,
    const array& q_weight,
    const array& k_weight,
    const array& v_weight,
    int in_features,
    int q_out,
    int kv_out
) {
    if (q_w == nullptr || k_w == nullptr || v_w == nullptr) {
        auto q_wt = orient_matmul_rhs_cached(q_w, q_weight, in_features, q_out, true);
        auto k_wt = orient_matmul_rhs_cached(k_w, k_weight, in_features, kv_out, true);
        auto v_wt = orient_matmul_rhs_cached(v_w, v_weight, in_features, kv_out, true);
        return contiguous(concatenate({q_wt, k_wt, v_wt}, 1));
    }

    AttentionFusedQkvCacheKey key{
        q_w,
        k_w,
        v_w,
        in_features,
        q_out,
        kv_out,
    };
    auto* hot_slots = attention_fused_qkv_hot_slots_store();
    const size_t hot_index = AttentionFusedQkvCacheKeyHash{}(key) & (kAttentionFusedQkvHotSlots - 1);
    auto& hot_slot = hot_slots[hot_index];
    if (hot_slot.valid && hot_slot.key == key) {
        return hot_slot.value;
    }

    auto& cache = attention_fused_qkv_cache_store();
    if (cache.size() >= kAttentionFusedQkvCacheMaxEntries) {
        cache.clear();
        for (size_t i = 0; i < kAttentionFusedQkvHotSlots; ++i) {
            hot_slots[i].valid = false;
        }
    }

    auto it = cache.find(key);
    if (it != cache.end()) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.value = it->second;
        return it->second;
    }

    auto q_wt = orient_matmul_rhs_cached(q_w, q_weight, in_features, q_out, true);
    auto k_wt = orient_matmul_rhs_cached(k_w, k_weight, in_features, kv_out, true);
    auto v_wt = orient_matmul_rhs_cached(v_w, v_weight, in_features, kv_out, true);
    auto fused = contiguous(concatenate({q_wt, k_wt, v_wt}, 1));
    // Materialize once so cached fused projection weights remain concrete.
    eval(fused);
    auto [inserted, _] = cache.emplace(key, std::move(fused));
    hot_slot.valid = true;
    hot_slot.key = key;
    hot_slot.value = inserted->second;
    return inserted->second;
}

static float resolve_attention_scale(
    float query_pre_attn_scalar,
    float attention_multiplier,
    int head_dim
) {
    if (attention_multiplier > 0.0f) {
        return attention_multiplier;
    }
    if (query_pre_attn_scalar > 0.0f) {
        return 1.0f / std::sqrt(query_pre_attn_scalar);
    }
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

static void maybe_add_linear_bias(array* values, const void* bias_ptr) {
    if (values == nullptr) {
        throw std::invalid_argument("[attention] null tensor for bias add");
    }
    if (bias_ptr != nullptr) {
        const auto& bias = *static_cast<const array*>(bias_ptr);
        *values = *values + cast_bias_cached(bias_ptr, bias, values->dtype());
    }
}

static ProjectedQkv project_quantized_qkv(
    const array& input,
    const void* q_w,
    const void* q_s,
    const void* q_b,
    const void* k_w,
    const void* k_s,
    const void* k_b,
    const void* v_w,
    const void* v_s,
    const void* v_b,
    int group_size,
    int bits,
    const void* q_bias,
    const void* k_bias,
    const void* v_bias
) {
    ProjectedQkv projected{
        quantized_linear_no_bias(
            input,
            *static_cast<const array*>(q_w),
            *static_cast<const array*>(q_s),
            *static_cast<const array*>(q_b),
            group_size,
            bits,
            q_s,
            q_b
        ),
        quantized_linear_no_bias(
            input,
            *static_cast<const array*>(k_w),
            *static_cast<const array*>(k_s),
            *static_cast<const array*>(k_b),
            group_size,
            bits,
            k_s,
            k_b
        ),
        quantized_linear_no_bias(
            input,
            *static_cast<const array*>(v_w),
            *static_cast<const array*>(v_s),
            *static_cast<const array*>(v_b),
            group_size,
            bits,
            v_s,
            v_b
        ),
    };
    maybe_add_linear_bias(&projected.q, q_bias);
    maybe_add_linear_bias(&projected.k, k_bias);
    maybe_add_linear_bias(&projected.v, v_bias);
    return projected;
}

static void reshape_qkv_to_attention_layout(
    array* q,
    array* k,
    array* v,
    int batch_size,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim
) {
    if (q == nullptr || k == nullptr || v == nullptr) {
        throw std::invalid_argument("[attention] null qkv tensor");
    }
    if (batch_size == 1 && seq_len == 1) {
        // Decode fast path: reshape directly to attention layout and avoid
        // no-op transpose nodes on singleton sequence dimensions.
        *q = reshape(*q, {1, n_heads, 1, head_dim});
        *k = reshape(*k, {1, n_kv_heads, 1, head_dim});
        *v = reshape(*v, {1, n_kv_heads, 1, head_dim});
        return;
    }
    *q = reshape(*q, {batch_size, seq_len, n_heads, head_dim});
    *k = reshape(*k, {batch_size, seq_len, n_kv_heads, head_dim});
    *v = reshape(*v, {batch_size, seq_len, n_kv_heads, head_dim});
    *q = transpose(*q, {0, 2, 1, 3});
    *k = transpose(*k, {0, 2, 1, 3});
    *v = transpose(*v, {0, 2, 1, 3});
}

static void maybe_apply_qk_norm(
    array* q,
    array* k,
    const void* q_norm_w,
    const void* k_norm_w,
    float rms_eps
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor");
    }
    if (q_norm_w != nullptr) {
        *q = fast::rms_norm(*q, *static_cast<const array*>(q_norm_w), rms_eps);
    }
    if (k_norm_w != nullptr) {
        *k = fast::rms_norm(*k, *static_cast<const array*>(k_norm_w), rms_eps);
    }
}

static void apply_attention_rope(
    array* q,
    array* k,
    int seq_len,
    int head_dim,
    size_t pos_offset,
    float rope_theta,
    const void* runtime_rope_cos,
    const void* runtime_rope_sin,
    size_t runtime_rope_dim
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for rope");
    }

    bool use_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr;
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    if (use_runtime_rope && (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0)) {
        use_runtime_rope = false;
    }
    // Decode fast path: for single-token steps, dispatch the dedicated rope
    // kernel instead of building a runtime-table slice/mul graph.
    if (seq_len == 1) {
        use_runtime_rope = false;
    }
    if (use_runtime_rope) {
        const auto& cos_table = *static_cast<const array*>(runtime_rope_cos);
        const auto& sin_table = *static_cast<const array*>(runtime_rope_sin);
        const int seq_start = static_cast<int>(pos_offset);
        const int seq_stop = seq_start + seq_len;
        const bool valid_tables =
            cos_table.ndim() == 2 && sin_table.ndim() == 2 &&
            cos_table.shape(1) >= rope_dim && sin_table.shape(1) >= rope_dim &&
            cos_table.shape(0) >= seq_stop && sin_table.shape(0) >= seq_stop;
        if (valid_tables) {
            auto cos_rows = slice(cos_table, {seq_start, 0}, {seq_stop, rope_dim});
            auto sin_rows = slice(sin_table, {seq_start, 0}, {seq_stop, rope_dim});
            *q = apply_runtime_rope_to_tensor(*q, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
            *k = apply_runtime_rope_to_tensor(*k, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
        } else {
            use_runtime_rope = false;
        }
    }
    if (!use_runtime_rope) {
        *q = fast::rope(*q, head_dim, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
        *k = fast::rope(*k, head_dim, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    }
}

static void resolve_attention_cache(
    void* cache_ptr,
    size_t layer_idx,
    array* k,
    array* v,
    array* k_for_attn,
    array* v_for_attn,
    bool* is_prefill
) {
    if (k == nullptr || v == nullptr || k_for_attn == nullptr || v_for_attn == nullptr || is_prefill == nullptr) {
        throw std::invalid_argument("[attention] null cache tensor");
    }

    *k_for_attn = *k;
    *v_for_attn = *v;
    *is_prefill = true;

    if (cache_ptr == nullptr) {
        return;
    }

    void* k_cached = nullptr;
    void* v_cached = nullptr;
    mlx_cache_update_and_fetch_bfloat16(
        cache_ptr,
        layer_idx,
        k,
        v,
        &k_cached,
        &v_cached,
        is_prefill
    );
    if (k_cached != nullptr && v_cached != nullptr) {
        *k_for_attn = *static_cast<array*>(k_cached);
        *v_for_attn = *static_cast<array*>(v_cached);
    }
}

static array run_attention_kernel(
    const array& q,
    const array& k_for_attn,
    const array& v_for_attn,
    float scale_value,
    bool is_prefill,
    const void* attn_sinks
) {
    std::optional<array> sinks_opt = std::nullopt;
    if (attn_sinks != nullptr) {
        sinks_opt = *static_cast<const array*>(attn_sinks);
    }
    return fast::scaled_dot_product_attention(
        q,
        k_for_attn,
        v_for_attn,
        scale_value,
        is_prefill ? "causal" : "",
        std::nullopt,
        sinks_opt
    );
}

static array reshape_attention_output(
    const array& attention_out,
    int batch_size,
    int seq_len,
    int output_dim
) {
    auto flattened = transpose(attention_out, {0, 2, 1, 3});
    return reshape(flattened, {batch_size, seq_len, output_dim});
}

void* mlx_lazy_fused_attention(
    const void* input,
    const void* q_w, const void* q_s, const void* q_b,
    const void* k_w, const void* k_s, const void* k_b,
    const void* v_w, const void* v_s, const void* v_b,
    const void* o_w, const void* o_s, const void* o_b,
    const void* q_norm_w,  // can be null
    const void* k_norm_w,  // can be null
    // Linear biases (optional, can be null)
    const void* q_bias,    // [n_heads * head_dim]
    const void* k_bias,    // [n_kv_heads * head_dim]
    const void* v_bias,    // [n_kv_heads * head_dim]
    const void* o_bias,    // [hidden_dim]
    // Attention sinks (optional, can be null)
    const void* attn_sinks,  // [n_heads]
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const int group_size_int = static_cast<int>(group_size);
    const int bits_int = static_cast<int>(bits);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const float scale_value = resolve_attention_scale(query_pre_attn_scalar, attention_multiplier, head_dim_int);

    ProjectedQkv projected = project_quantized_qkv(
        x,
        q_w, q_s, q_b,
        k_w, k_s, k_b,
        v_w, v_s, v_b,
        group_size_int,
        bits_int,
        q_bias,
        k_bias,
        v_bias
    );

    reshape_qkv_to_attention_layout(
        &projected.q,
        &projected.k,
        &projected.v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int
    );
    maybe_apply_qk_norm(&projected.q, &projected.k, q_norm_w, k_norm_w, rms_eps);
    apply_attention_rope(
        &projected.q,
        &projected.k,
        seq_len,
        head_dim_int,
        pos_offset,
        rope_theta,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );

    array k_for_attn = projected.k;
    array v_for_attn = projected.v;
    bool is_prefill = true;
    resolve_attention_cache(
        cache_ptr,
        layer_idx,
        &projected.k,
        &projected.v,
        &k_for_attn,
        &v_for_attn,
        &is_prefill
    );

    auto attn_out = run_attention_kernel(
        projected.q,
        k_for_attn,
        v_for_attn,
        scale_value,
        is_prefill,
        attn_sinks
    );
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, n_heads_int * head_dim_int);

    // Output projection
    auto out = quantized_linear_no_bias(
        attn_out,
        *static_cast<const array*>(o_w),
        *static_cast<const array*>(o_s),
        *static_cast<const array*>(o_b),
        group_size_int,
        bits_int,
        o_s,
        o_b
    );

    // Add output bias if present
    if (o_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(o_bias);
        out = out + cast_bias_cached(o_bias, bias, out.dtype());
    }

    return pool_array(std::move(out));
}

// ============================================================================
// Fused Attention Block (Mixed: quantized QKV, dense output projection)
// ============================================================================

void* mlx_lazy_fused_attention_qkv_quantized_o_dense(
    const void* input,
    const void* q_w, const void* q_s, const void* q_b,
    const void* k_w, const void* k_s, const void* k_b,
    const void* v_w, const void* v_s, const void* v_b,
    const void* o_w,
    const void* q_norm_w,  // can be null
    const void* k_norm_w,  // can be null
    // Linear biases (optional, can be null)
    const void* q_bias,    // [n_heads * head_dim]
    const void* k_bias,    // [n_kv_heads * head_dim]
    const void* v_bias,    // [n_kv_heads * head_dim]
    const void* o_bias,    // [hidden_dim]
    // Attention sinks (optional, can be null)
    const void* attn_sinks,  // [n_heads]
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const int group_size_int = static_cast<int>(group_size);
    const int bits_int = static_cast<int>(bits);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const float scale_value = resolve_attention_scale(query_pre_attn_scalar, attention_multiplier, head_dim_int);

    ProjectedQkv projected = project_quantized_qkv(
        x,
        q_w, q_s, q_b,
        k_w, k_s, k_b,
        v_w, v_s, v_b,
        group_size_int,
        bits_int,
        q_bias,
        k_bias,
        v_bias
    );

    reshape_qkv_to_attention_layout(
        &projected.q,
        &projected.k,
        &projected.v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int
    );
    maybe_apply_qk_norm(&projected.q, &projected.k, q_norm_w, k_norm_w, rms_eps);
    apply_attention_rope(
        &projected.q,
        &projected.k,
        seq_len,
        head_dim_int,
        pos_offset,
        rope_theta,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );

    array k_for_attn = projected.k;
    array v_for_attn = projected.v;
    bool is_prefill = true;
    resolve_attention_cache(
        cache_ptr,
        layer_idx,
        &projected.k,
        &projected.v,
        &k_for_attn,
        &v_for_attn,
        &is_prefill
    );

    auto attn_out = run_attention_kernel(
        projected.q,
        k_for_attn,
        v_for_attn,
        scale_value,
        is_prefill,
        attn_sinks
    );
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, n_heads_int * head_dim_int);

    // Dense output projection (o_w stored as [out, in]).
    auto o_wt = orient_matmul_rhs_cached(
        o_w,
        *static_cast<const array*>(o_w),
        n_heads_int * head_dim_int,
        x.shape(2),
        true
    );
    auto out = matmul_lastdim(attn_out, o_wt);

    // Add output bias if present
    if (o_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(o_bias);
        out = out + cast_bias_cached(o_bias, bias, out.dtype());
    }

    return pool_array(std::move(out));
}

// ============================================================================
// Fused FFN Block (Quantized)
// ============================================================================
// Combines: gate_proj -> SiLU -> multiply(up_proj) -> down_proj

void* mlx_lazy_fused_ffn(
    const void* input,
    const void* gate_w, const void* gate_s, const void* gate_b,
    const void* up_w, const void* up_s, const void* up_b,
    const void* down_w, const void* down_s, const void* down_b,
    size_t group_size, size_t bits,
    bool use_gelu  // true for GELU activation, false for SwiGLU
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    int gs = static_cast<int>(group_size);
    int b = static_cast<int>(bits);
    const auto& gate_w_arr = *static_cast<const array*>(gate_w);
    const auto& gate_s_arr = *static_cast<const array*>(gate_s);
    const auto& gate_b_arr = *static_cast<const array*>(gate_b);
    const auto& up_w_arr = *static_cast<const array*>(up_w);
    const auto& up_s_arr = *static_cast<const array*>(up_s);
    const auto& up_b_arr = *static_cast<const array*>(up_b);

    const auto& fused = ffn_quantized_gate_up_cached(
        gate_w,
        gate_s,
        gate_b,
        up_w,
        up_s,
        up_b,
        gate_w_arr,
        gate_s_arr,
        gate_b_arr,
        up_w_arr,
        up_s_arr,
        up_b_arr
    );
    const auto gate_up = quantized_linear_no_bias(
        x,
        fused.weights,
        fused.scales,
        fused.biases,
        gs,
        b
    );
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    auto gate = slice(gate_up, {0, 0, 0}, {batch_size, seq_len, fused.d_ff});
    auto up = slice(gate_up, {0, 0, fused.d_ff}, {batch_size, seq_len, 2 * fused.d_ff});

    // Activation: GELU or SwiGLU based on model config
    auto mid = [&]() -> array {
        if (use_gelu) {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_over_pi = 0.7978845608f;
            auto x3 = gate * gate * gate;
            auto inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
            return 0.5f * gate * (1.0f + tanh(inner)) * up;
        } else {
            // SwiGLU: gate * sigmoid(gate) * up
            return (gate * sigmoid(gate)) * up;
        }
    }();

    auto out = quantized_linear_no_bias(
        mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        *static_cast<const array*>(down_b),
        gs,
        b,
        down_s,
        down_b
    );

    return pool_array(std::move(out));
}

// ============================================================================
// Fused Attention Block (BFloat16 - non-quantized)
// ============================================================================

void* mlx_lazy_fused_attention_bf16(
    const void* input,
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_norm_w,
    const void* k_norm_w,
    // Linear biases (optional, can be null)
    const void* q_bias,
    const void* k_bias,
    const void* v_bias,
    const void* o_bias,
    // Attention sinks (optional, can be null)
    const void* attn_sinks,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int hidden_dim = x.shape(2);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const int q_out = n_heads_int * head_dim_int;
    const int kv_out = n_kv_heads_int * head_dim_int;
    const float scale_value = resolve_attention_scale(query_pre_attn_scalar, attention_multiplier, head_dim_int);

    const auto& q_weight = *static_cast<const array*>(q_w);
    const auto& k_weight = *static_cast<const array*>(k_w);
    const auto& v_weight = *static_cast<const array*>(v_w);
    auto qkv_wt = attention_fused_qkv_rhs_cached(
        q_w,
        k_w,
        v_w,
        q_weight,
        k_weight,
        v_weight,
        hidden_dim,
        q_out,
        kv_out
    );
    auto o_wt = orient_matmul_rhs_cached(o_w, *static_cast<const array*>(o_w), q_out, hidden_dim, true);

    auto qkv = matmul_lastdim(x, qkv_wt);
    array q = slice(qkv, {0, 0, 0}, {batch_size, seq_len, q_out});
    array k = slice(qkv, {0, 0, q_out}, {batch_size, seq_len, q_out + kv_out});
    array v = slice(qkv, {0, 0, q_out + kv_out}, {batch_size, seq_len, q_out + 2 * kv_out});

    maybe_add_linear_bias(&q, q_bias);
    maybe_add_linear_bias(&k, k_bias);
    maybe_add_linear_bias(&v, v_bias);

    reshape_qkv_to_attention_layout(
        &q,
        &k,
        &v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int
    );
    maybe_apply_qk_norm(&q, &k, q_norm_w, k_norm_w, rms_eps);
    // Keep dense attention compute on the model projection dtype after optional
    // QK-norm to avoid unintended FP32 SDPA execution.
    const Dtype proj_dtype = qkv.dtype();
    if (q.dtype() != proj_dtype) q = astype(q, proj_dtype);
    if (k.dtype() != proj_dtype) k = astype(k, proj_dtype);
    if (v.dtype() != proj_dtype) v = astype(v, proj_dtype);
    apply_attention_rope(
        &q,
        &k,
        seq_len,
        head_dim_int,
        pos_offset,
        rope_theta,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );

    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;
    resolve_attention_cache(
        cache_ptr,
        layer_idx,
        &k,
        &v,
        &k_for_attn,
        &v_for_attn,
        &is_prefill
    );

    auto attn_out = run_attention_kernel(q, k_for_attn, v_for_attn, scale_value, is_prefill, attn_sinks);
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, q_out);

    // Output projection
    auto out = matmul_lastdim(attn_out, o_wt);

    // Add output bias if present
    if (o_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(o_bias);
        out = out + cast_bias_cached(o_bias, bias, out.dtype());
    }

    return pool_array(std::move(out));
}
