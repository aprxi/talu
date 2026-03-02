// ============================================================================
// Fused FFN Block (BFloat16 - non-quantized)
// ============================================================================

static constexpr size_t kFfnFusedGateUpCacheMaxEntries = 2048;

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
    static thread_local std::unordered_map<FfnFusedGateUpCacheKey, array, FfnFusedGateUpCacheKeyHash> cache;
    return cache;
}

static inline void ffn_fused_gate_up_cache_clear() {
    ffn_fused_gate_up_cache_store().clear();
}

static inline array ffn_fused_gate_up_rhs_cached(
    const void* gate_w,
    const void* up_w,
    const array& gate_weight,
    const array& up_weight,
    int in_features
) {
    auto gate_wt = orient_matmul_rhs_cached(gate_w, gate_weight, in_features, std::nullopt, true);
    const int d_ff = gate_wt.shape(1);
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

void* mlx_lazy_fused_ffn_bf16(
    const void* input,
    const void* gate_w, const void* up_w, const void* down_w
) {
    const auto& x = *static_cast<const array*>(input);
    const int in_features = x.shape(2);

    const auto& gate_weight = *static_cast<const array*>(gate_w);
    const auto& up_weight = *static_cast<const array*>(up_w);
    auto gate_wt = orient_matmul_rhs_cached(gate_w, gate_weight, in_features, std::nullopt, true);
    const int d_ff = gate_wt.shape(1);
    auto gate_up_wt = ffn_fused_gate_up_rhs_cached(gate_w, up_w, gate_weight, up_weight, in_features);
    auto down_wt = orient_matmul_rhs_cached(down_w, *static_cast<const array*>(down_w), gate_wt.shape(1), in_features, true);

    auto gate_up = matmul_lastdim(x, gate_up_wt);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    auto gate = slice(gate_up, {0, 0, 0}, {batch_size, seq_len, d_ff});
    auto up = slice(gate_up, {0, 0, d_ff}, {batch_size, seq_len, 2 * d_ff});
    auto mid = (gate * sigmoid(gate)) * up;
    auto out = matmul_lastdim(mid, down_wt);

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
        ? quantized_matmul(x,
            *static_cast<const array*>(router_w),
            *static_cast<const array*>(router_s),
            *static_cast<const array*>(router_b),
            true, static_cast<int>(router_group_size), 8, "affine")
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
