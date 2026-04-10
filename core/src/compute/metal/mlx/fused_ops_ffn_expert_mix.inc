// ============================================================================
// Fused FFN Block (BFloat16 - non-quantized)
// ============================================================================

static constexpr size_t kFfnFusedGateUpCacheMaxEntries = 2048;
static constexpr size_t kFfnFusedDenseRhsCacheMaxEntries = 2048;
static constexpr size_t kFfnFusedDenseRhsHotSlots = 64;

struct FfnFusedGateUpCacheKey {
    const void* gate_weight = nullptr;
    const void* up_weight = nullptr;
    int in_features = 0;
    int d_ff = 0;

    bool operator==(const FfnFusedGateUpCacheKey& other) const {
        return gate_weight == other.gate_weight &&
               up_weight == other.up_weight &&
               in_features == other.in_features &&
               d_ff == other.d_ff;
    }
};

struct FfnFusedGateUpCacheKeyHash {
    size_t operator()(const FfnFusedGateUpCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.gate_weight);
        h ^= std::hash<const void*>{}(key.up_weight) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.in_features) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.d_ff) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static inline std::unordered_map<FfnFusedGateUpCacheKey, array, FfnFusedGateUpCacheKeyHash>& ffn_fused_gate_up_cache_store() {
    return tls_never_destroyed<std::unordered_map<FfnFusedGateUpCacheKey, array, FfnFusedGateUpCacheKeyHash>>();
}

static inline void ffn_fused_gate_up_cache_clear() {
    ffn_fused_gate_up_cache_store().clear();
}

struct FfnFusedDenseRhsCacheKey {
    const void* gate_weight = nullptr;
    const void* up_weight = nullptr;
    const void* down_weight = nullptr;
    int in_features = 0;

    bool operator==(const FfnFusedDenseRhsCacheKey& other) const {
        return gate_weight == other.gate_weight &&
               up_weight == other.up_weight &&
               down_weight == other.down_weight &&
               in_features == other.in_features;
    }
};

struct FfnFusedDenseRhsCacheKeyHash {
    size_t operator()(const FfnFusedDenseRhsCacheKey& key) const {
        size_t h = std::hash<const void*>{}(key.gate_weight);
        h ^= std::hash<const void*>{}(key.up_weight) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<const void*>{}(key.down_weight) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.in_features) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct FfnFusedDenseRhsCacheValue {
    array gate_up_wt;
    array down_wt;
    int d_ff = 0;
};

static inline std::unordered_map<FfnFusedDenseRhsCacheKey, FfnFusedDenseRhsCacheValue, FfnFusedDenseRhsCacheKeyHash>& ffn_fused_dense_rhs_cache_store() {
    return tls_never_destroyed<std::unordered_map<FfnFusedDenseRhsCacheKey, FfnFusedDenseRhsCacheValue, FfnFusedDenseRhsCacheKeyHash>>();
}

struct FfnFusedDenseRhsHotSlot {
    bool valid = false;
    FfnFusedDenseRhsCacheKey key;
    FfnFusedDenseRhsCacheValue value{
        array(0.0f),
        array(0.0f),
        0,
    };
};

static inline FfnFusedDenseRhsHotSlot* ffn_fused_dense_rhs_hot_slots_store() {
    return tls_never_destroyed_array<FfnFusedDenseRhsHotSlot, kFfnFusedDenseRhsHotSlots>();
}

struct FfnFusedDenseRhsLastHit {
    bool valid = false;
    FfnFusedDenseRhsCacheKey key;
    FfnFusedDenseRhsCacheValue value{
        array(0.0f),
        array(0.0f),
        0,
    };
};

static inline FfnFusedDenseRhsLastHit& ffn_fused_dense_rhs_last_hit_store() {
    return tls_never_destroyed<FfnFusedDenseRhsLastHit>();
}

static inline std::optional<FfnFusedDenseRhsCacheValue>& ffn_fused_dense_rhs_scratch_store() {
    return tls_never_destroyed<std::optional<FfnFusedDenseRhsCacheValue>>();
}

static inline void ffn_fused_dense_rhs_cache_clear() {
    ffn_fused_dense_rhs_cache_store().clear();
    auto* hot_slots = ffn_fused_dense_rhs_hot_slots_store();
    for (size_t i = 0; i < kFfnFusedDenseRhsHotSlots; ++i) {
        hot_slots[i].valid = false;
    }
    ffn_fused_dense_rhs_last_hit_store().valid = false;
    ffn_fused_dense_rhs_scratch_store().reset();
}

static inline FfnFusedDenseRhsCacheValue ffn_fused_dense_rhs_uncached(
    const void* gate_w,
    const void* up_w,
    const void* down_w,
    const array& gate_weight,
    const array& up_weight,
    const array& down_weight,
    int in_features
) {
    auto gate_wt = orient_matmul_rhs_cached(gate_w, gate_weight, in_features, std::nullopt, true);
    const int d_ff = gate_wt.shape(1);
    auto up_wt = orient_matmul_rhs_cached(up_w, up_weight, in_features, d_ff, true);
    auto down_wt = orient_matmul_rhs_cached(down_w, down_weight, d_ff, in_features, true);
    return FfnFusedDenseRhsCacheValue{
        contiguous(concatenate({gate_wt, up_wt}, 1)),
        down_wt,
        d_ff,
    };
}

static inline const FfnFusedDenseRhsCacheValue& ffn_fused_dense_rhs_cached(
    const void* gate_w,
    const void* up_w,
    const void* down_w,
    const array& gate_weight,
    const array& up_weight,
    const array& down_weight,
    int in_features
) {
    if (gate_w == nullptr || up_w == nullptr || down_w == nullptr) {
        auto& scratch = ffn_fused_dense_rhs_scratch_store();
        scratch = ffn_fused_dense_rhs_uncached(
            gate_w,
            up_w,
            down_w,
            gate_weight,
            up_weight,
            down_weight,
            in_features
        );
        return *scratch;
    }

    FfnFusedDenseRhsCacheKey key{
        gate_w,
        up_w,
        down_w,
        in_features,
    };
    auto* hot_slots = ffn_fused_dense_rhs_hot_slots_store();
    const size_t hot_index = FfnFusedDenseRhsCacheKeyHash{}(key) & (kFfnFusedDenseRhsHotSlots - 1);
    auto& hot_slot = hot_slots[hot_index];
    if (hot_slot.valid && hot_slot.key == key) {
        return hot_slot.value;
    }

    auto& last_hit = ffn_fused_dense_rhs_last_hit_store();
    if (last_hit.valid && last_hit.key == key) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.value = last_hit.value;
        return last_hit.value;
    }

    auto& cache = ffn_fused_dense_rhs_cache_store();
    if (cache.size() >= kFfnFusedDenseRhsCacheMaxEntries) {
        cache.clear();
        for (size_t i = 0; i < kFfnFusedDenseRhsHotSlots; ++i) {
            hot_slots[i].valid = false;
        }
        last_hit.valid = false;
    }

    auto it = cache.find(key);
    if (it != cache.end()) {
        hot_slot.valid = true;
        hot_slot.key = key;
        hot_slot.value = it->second;
        last_hit.valid = true;
        last_hit.key = key;
        last_hit.value = it->second;
        return it->second;
    }

    FfnFusedDenseRhsCacheValue rhs = ffn_fused_dense_rhs_uncached(
        gate_w,
        up_w,
        down_w,
        gate_weight,
        up_weight,
        down_weight,
        in_features
    );
    // Materialize once so dense decode reuses concrete fused rhs tensors.
    eval({rhs.gate_up_wt, rhs.down_wt});
    auto [inserted, _] = cache.emplace(key, std::move(rhs));
    hot_slot.valid = true;
    hot_slot.key = key;
    hot_slot.value = inserted->second;
    last_hit.valid = true;
    last_hit.key = key;
    last_hit.value = inserted->second;
    return inserted->second;
}

static inline array ffn_fused_gate_up_rhs_cached(
    const void* gate_w,
    const void* up_w,
    const array& gate_wt,
    const array& up_weight,
    int in_features,
    int d_ff
) {
    auto up_wt = orient_matmul_rhs_cached(up_w, up_weight, in_features, d_ff, true);

    if (gate_w == nullptr || up_w == nullptr) {
        return contiguous(concatenate({gate_wt, up_wt}, 1));
    }

    auto& cache = ffn_fused_gate_up_cache_store();
    if (cache.size() >= kFfnFusedGateUpCacheMaxEntries) {
        cache.clear();
    }

    FfnFusedGateUpCacheKey key{
        gate_w,
        up_w,
        in_features,
        d_ff,
    };
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    auto fused = contiguous(concatenate({gate_wt, up_wt}, 1));
    // Materialize once so cached fused FFN projection weights remain concrete.
    eval(fused);
    auto [inserted, _] = cache.emplace(key, std::move(fused));
    return inserted->second;
}

static inline fast::CustomKernelFunction& ffn_swiglu_from_fused_gate_up_kernel() {
    static thread_local fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_ffn_swiglu_from_fused_gate_up",
        { "gate_up" },
        { "mid" },
        R"(
    const uint idx = thread_position_in_grid.x;
    if (idx >= ELEMENTS) {
      return;
    }
    const uint d = idx % D_FF;
    const uint token = idx / D_FF;
    const uint base = token * (2 * D_FF);
    const float gate = gate_up[base + d];
    const float up = gate_up[base + D_FF + d];
    mid[idx] = (gate * (1.0f / (1.0f + exp(-gate)))) * up;
        )",
        "",
        true,
        false
    );
    return kernel;
}

static inline array ffn_swiglu_from_fused_gate_up(const array& gate_up, int batch_size, int seq_len, int d_ff) {
    if (batch_size <= 0 || seq_len <= 0 || d_ff <= 0) {
        throw std::invalid_argument("[fused_ffn] invalid swiglu dims");
    }
    const int elements = batch_size * seq_len * d_ff;
    auto outputs = ffn_swiglu_from_fused_gate_up_kernel()(
        { gate_up },
        { Shape{batch_size, seq_len, d_ff} },
        { gate_up.dtype() },
        { elements, 1, 1 },
        { 256, 1, 1 },
        {
            { "D_FF", d_ff },
            { "ELEMENTS", elements },
        },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 1) {
        throw std::runtime_error("[fused_ffn] swiglu kernel returned unexpected output count");
    }
    return outputs[0];
}

void* mlx_lazy_fused_ffn_bf16(
    const void* input,
    const void* gate_w, const void* up_w, const void* down_w
) {
    const auto& x = *static_cast<const array*>(input);
    const int in_features = x.shape(2);

    const auto& gate_weight = *static_cast<const array*>(gate_w);
    const auto& up_weight = *static_cast<const array*>(up_w);
    const auto& down_weight = *static_cast<const array*>(down_w);
    const auto& rhs = ffn_fused_dense_rhs_cached(
        gate_w,
        up_w,
        down_w,
        gate_weight,
        up_weight,
        down_weight,
        in_features
    );

    auto gate_up = matmul_lastdim(x, rhs.gate_up_wt);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    auto mid = ffn_swiglu_from_fused_gate_up(gate_up, batch_size, seq_len, rhs.d_ff);
    auto out = matmul_lastdim(mid, rhs.down_wt);

    return pool_array(std::move(out));
}

void* mlx_lazy_rms_norm_fused_ffn_bf16(
    const void* input,
    const void* norm_w,
    const void* gate_w,
    const void* up_w,
    const void* down_w,
    float eps
) {
    const auto& x = *static_cast<const array*>(input);
    const int in_features = x.shape(2);
    const auto& norm_weight_raw = *static_cast<const array*>(norm_w);
    const array norm_weight = canonicalize_rms_norm_weight(norm_weight_raw, in_features, "rms_norm_fused_ffn_bf16");

    const auto& gate_weight = *static_cast<const array*>(gate_w);
    const auto& up_weight = *static_cast<const array*>(up_w);
    const auto& down_weight = *static_cast<const array*>(down_w);
    const auto& rhs = ffn_fused_dense_rhs_cached(
        gate_w,
        up_w,
        down_w,
        gate_weight,
        up_weight,
        down_weight,
        in_features
    );

    auto normed = fast::rms_norm(x, norm_weight, eps);
    auto gate_up = matmul_lastdim(normed, rhs.gate_up_wt);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    auto mid = ffn_swiglu_from_fused_gate_up(gate_up, batch_size, seq_len, rhs.d_ff);
    auto out = matmul_lastdim(mid, rhs.down_wt);

    return pool_array(std::move(out));
}


// ============================================================================
// Fused ExpertMix FFN Block (MXFP4 quantized experts)
// ============================================================================
// Implements: router -> topk -> gather_qmm (gate/up/down) -> weighted sum
//
// This matches mlx_lm's SwitchGLU layer:
// - Router projects to expert logits
// - TopK selects active experts per token
// - gather_qmm computes expert FFN outputs (only for selected experts)
// - Results are weighted by softmax and summed

// SwiGLU variant with alpha=1.702, clipping, and (up+1) formulation:
// swiglu(x_linear, x_glu) = (glu_scaled * sigmoid(glu_scaled)) * (x_linear + 1)
//   where glu_scaled = 1.702 * clip(x_glu, max=7)
static array swiglu_variant(const array& x_linear, const array& x_glu) {
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    // Clamp values
    auto x_glu_clipped = clip(x_glu, std::nullopt, array(limit));
    auto x_linear_clipped = clip(x_linear, array(-limit), array(limit));

    // Compute activation
    auto glu_scaled = alpha * x_glu_clipped;
    auto sig = sigmoid(glu_scaled);
    auto out_glu = x_glu_clipped * sig;

    // Note: x_linear + 1 bias
    return out_glu * (x_linear_clipped + 1.0f);
}

static array standard_moe_activation(
    const array& gate_out,
    const array& up_out,
    bool use_gelu
) {
    return use_gelu ? gelu_approx(gate_out) * up_out : (gate_out * sigmoid(gate_out)) * up_out;
}

static std::pair<array, array> split_fused_moe_gate_up_output(
    const array& gate_up_out
) {
    const int last_dim = gate_up_out.ndim() - 1;
    const int fused_width = gate_up_out.shape(last_dim);
    if (fused_width % 2 != 0) {
        throw std::runtime_error("MoE fused gate_up output width must be even");
    }

    const int half_width = fused_width / 2;
    Shape gate_start(gate_up_out.ndim(), 0);
    Shape gate_stop = gate_up_out.shape();
    gate_stop[last_dim] = half_width;

    Shape up_start(gate_up_out.ndim(), 0);
    Shape up_stop = gate_up_out.shape();
    up_start[last_dim] = half_width;

    return {
        slice(gate_up_out, gate_start, gate_stop),
        slice(gate_up_out, up_start, up_stop),
    };
}

void* mlx_lazy_fused_expert_mix_ffn_mxfp4(
    const void* input,
    // Router weights (8-bit affine quantized)
    const void* router_w, const void* router_s, const void* router_b,
    const void* router_bias,  // can be null
    // Expert weights [num_experts, d_ff, packed_dim] - MXFP4 quantized
    // Separate gate/up/down projections (not fused)
    const void* gate_w, const void* gate_s,
    const void* up_w, const void* up_s,
    const void* down_w, const void* down_s,
    // Expert biases (optional) - [num_experts, d_ff] or [num_experts, d_model]
    const void* gate_bias,  // can be null
    const void* up_bias,    // can be null
    const void* down_bias,  // can be null
    // Config
    size_t num_experts,
    size_t experts_per_token,
    size_t router_group_size,   // 64 for router (8-bit)
    size_t expert_group_size    // 32 for MXFP4
) {
    const auto& x = *static_cast<const array*>(input);

    // Router: compute expert logits [B, L, num_experts]
    // Supports two formats:
    // 1. MLX community: 8-bit affine quantized router (router_s/router_b present)
    // 2. Hub/OpenAI: BF16 unquantized router (router_s/router_b null)
    array router_logits = (router_s != nullptr && router_b != nullptr)
        ? quantized_linear_no_bias(
            x,
            *static_cast<const array*>(router_w),
            *static_cast<const array*>(router_s),
            *static_cast<const array*>(router_b),
            static_cast<int>(router_group_size),
            8,
            router_s,
            router_b
        )
        : matmul(
            x,
            orient_matmul_rhs_cached(
                router_w,
                *static_cast<const array*>(router_w),
                x.shape(x.ndim() - 1),
                static_cast<int>(num_experts),
                true
            )
        );

    // Add router bias if present
    if (router_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(router_bias);
        router_logits = router_logits + cast_bias_cached(router_bias, bias, router_logits.dtype());
    }

    // TopK: select top experts
    // argpartition returns indices of top K elements (unsorted)
    int k = static_cast<int>(experts_per_token);
    auto partitioned_indices = argpartition(router_logits, -k, -1);

    // Extract top-k indices (last k elements along axis -1)
    int last_dim = router_logits.ndim() - 1;
    int total_experts = static_cast<int>(num_experts);

    // Slice to get top-k indices: [..., -k:]
    Shape start(router_logits.ndim(), 0);
    Shape stop = router_logits.shape();
    start[last_dim] = total_experts - k;
    auto top_k_indices = slice(partitioned_indices, start, stop);

    // Get corresponding logits for softmax weighting
    auto top_k_logits = take_along_axis(router_logits, top_k_indices, last_dim);

    // Softmax to get expert weights [B, L, K]
    auto expert_weights = softmax(top_k_logits, -1, true);  // precise=true

    // Expand input for gather_qmm: [B, L, 1, 1, hidden_dim]
    auto x_expanded = expand_dims(x, {-2, -3});

    // Gather-QMM for gate projection
    // gather_qmm with rhs_indices selects which experts to use per token
    auto gate_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(gate_w),
        *static_cast<const array*>(gate_s),
        std::nullopt,  // MXFP4 has no biases in quantization
        std::nullopt,  // lhs_indices
        top_k_indices, // rhs_indices - selects which experts
        true,          // transpose
        static_cast<int>(expert_group_size),
        4,             // bits for MXFP4
        "mxfp4",       // mode
        false          // sorted_indices
    );

    // Up projection
    auto up_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(up_w),
        *static_cast<const array*>(up_s),
        std::nullopt,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        4,
        "mxfp4",
        false
    );

    // Add expert biases if present
    if (gate_bias != nullptr) {
        // Gather biases for selected experts: [num_experts, d_ff] -> [B*L*K, d_ff]
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto gate_b = take(*static_cast<const array*>(gate_bias), indices_flat, 0);
        // Reshape to match gate_out: [B, L, K, 1, d_ff]
        auto shape = gate_out.shape();
        gate_b = reshape(gate_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        const array gate_b_cast = (gate_b.dtype() == gate_out.dtype()) ? gate_b : astype(gate_b, gate_out.dtype());
        gate_out = gate_out + gate_b_cast;
    }
    if (up_bias != nullptr) {
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto up_b = take(*static_cast<const array*>(up_bias), indices_flat, 0);
        auto shape = up_out.shape();
        up_b = reshape(up_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        const array up_b_cast = (up_b.dtype() == up_out.dtype()) ? up_b : astype(up_b, up_out.dtype());
        up_out = up_out + up_b_cast;
    }

    // SwiGLU variant activation
    auto mid = swiglu_variant(up_out, gate_out);

    // Down projection
    auto down_out = gather_qmm(
        mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        std::nullopt,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        4,
        "mxfp4",
        false
    );

    if (down_bias != nullptr) {
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto down_b = take(*static_cast<const array*>(down_bias), indices_flat, 0);
        auto shape = down_out.shape();
        down_b = reshape(down_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        const array down_b_cast = (down_b.dtype() == down_out.dtype()) ? down_b : astype(down_b, down_out.dtype());
        down_out = down_out + down_b_cast;
    }

    // Squeeze out singleton dimensions: [B, L, K, 1, hidden_dim] -> [B, L, K, hidden_dim]
    down_out = squeeze(down_out, -2);

    // Weight by expert weights and sum
    // expert_weights: [B, L, K] -> [B, L, K, 1]
    auto weights_expanded = expand_dims(expert_weights, -1);
    auto weighted = down_out * weights_expanded;

    // Sum over experts: [B, L, K, hidden_dim] -> [B, L, hidden_dim]
    auto out = sum(weighted, -2);

    return pool_array(std::move(out));
}

void* mlx_lazy_fused_expert_mix_ffn_affine(
    const void* input,
    const void* router_w,
    const void* router_bias,
    const void* gate_w,
    const void* gate_s,
    const void* gate_bias,
    const void* up_w,
    const void* up_s,
    const void* up_bias,
    const void* down_w,
    const void* down_s,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t expert_group_size,
    int expert_bits,
    bool use_gelu
) {
    const auto& x = *static_cast<const array*>(input);

    array router_logits = matmul(
        x,
        orient_matmul_rhs_cached(
            router_w,
            *static_cast<const array*>(router_w),
            x.shape(x.ndim() - 1),
            static_cast<int>(num_experts),
            true
        )
    );
    if (router_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(router_bias);
        router_logits = router_logits + cast_bias_cached(router_bias, bias, router_logits.dtype());
    }

    const int k = static_cast<int>(experts_per_token);
    auto partitioned_indices = argpartition(router_logits, -k, -1);
    const int last_dim = router_logits.ndim() - 1;
    const int total_experts = static_cast<int>(num_experts);
    Shape start(router_logits.ndim(), 0);
    Shape stop = router_logits.shape();
    start[last_dim] = total_experts - k;
    auto top_k_indices = slice(partitioned_indices, start, stop);
    auto top_k_logits = take_along_axis(router_logits, top_k_indices, last_dim);
    auto expert_weights = softmax(top_k_logits, -1, true);

    auto x_expanded = expand_dims(x, {-2, -3});

    const auto gate_biases = gate_bias != nullptr
        ? std::optional<array>(*static_cast<const array*>(gate_bias))
        : std::nullopt;
    const auto up_biases = up_bias != nullptr
        ? std::optional<array>(*static_cast<const array*>(up_bias))
        : std::nullopt;
    const auto down_biases = down_bias != nullptr
        ? std::optional<array>(*static_cast<const array*>(down_bias))
        : std::nullopt;

    auto gate_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(gate_w),
        *static_cast<const array*>(gate_s),
        gate_biases,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        expert_bits,
        "affine",
        false
    );
    auto up_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(up_w),
        *static_cast<const array*>(up_s),
        up_biases,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        expert_bits,
        "affine",
        false
    );

    auto mid = standard_moe_activation(gate_out, up_out, use_gelu);
    auto down_out = gather_qmm(
        mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        down_biases,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        expert_bits,
        "affine",
        false
    );

    down_out = squeeze(down_out, -2);
    auto weights_expanded = expand_dims(expert_weights, -1);
    auto weighted = down_out * weights_expanded;
    auto out = sum(weighted, -2);
    return pool_array(std::move(out));
}

void* mlx_lazy_fused_expert_mix_ffn_affine_fused_gate_up(
    const void* input,
    const void* router_w,
    const void* router_bias,
    const void* gate_up_w,
    const void* gate_up_s,
    const void* gate_up_bias,
    const void* down_w,
    const void* down_s,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t expert_group_size,
    int expert_bits,
    bool use_gelu
) {
    const auto& x = *static_cast<const array*>(input);

    array router_logits = matmul(
        x,
        orient_matmul_rhs_cached(
            router_w,
            *static_cast<const array*>(router_w),
            x.shape(x.ndim() - 1),
            static_cast<int>(num_experts),
            true
        )
    );
    if (router_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(router_bias);
        router_logits = router_logits + cast_bias_cached(router_bias, bias, router_logits.dtype());
    }

    const int k = static_cast<int>(experts_per_token);
    auto partitioned_indices = argpartition(router_logits, -k, -1);
    const int last_dim = router_logits.ndim() - 1;
    const int total_experts = static_cast<int>(num_experts);
    Shape start(router_logits.ndim(), 0);
    Shape stop = router_logits.shape();
    start[last_dim] = total_experts - k;
    auto top_k_indices = slice(partitioned_indices, start, stop);
    auto top_k_logits = take_along_axis(router_logits, top_k_indices, last_dim);
    auto expert_weights = softmax(top_k_logits, -1, true);

    auto x_expanded = expand_dims(x, {-2, -3});

    const auto gate_up_biases = gate_up_bias != nullptr
        ? std::optional<array>(*static_cast<const array*>(gate_up_bias))
        : std::nullopt;
    const auto down_biases = down_bias != nullptr
        ? std::optional<array>(*static_cast<const array*>(down_bias))
        : std::nullopt;

    auto gate_up_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(gate_up_w),
        *static_cast<const array*>(gate_up_s),
        gate_up_biases,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        expert_bits,
        "affine",
        false
    );
    const auto [gate_out, up_out] = split_fused_moe_gate_up_output(gate_up_out);
    auto mid = standard_moe_activation(gate_out, up_out, use_gelu);
    auto down_out = gather_qmm(
        mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        down_biases,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        expert_bits,
        "affine",
        false
    );

    down_out = squeeze(down_out, -2);
    auto weights_expanded = expand_dims(expert_weights, -1);
    auto weighted = down_out * weights_expanded;
    auto out = sum(weighted, -2);
    return pool_array(std::move(out));
}
