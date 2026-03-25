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

static inline bool attention_stage_trace_enabled() {
    const char* raw = std::getenv("TALU_DBG_ATTN_STAGES");
    if (raw == nullptr) return false;
    if (raw[0] == '\0') return false;
    if ((raw[0] == '0' || raw[0] == 'n' || raw[0] == 'N' || raw[0] == 'f' || raw[0] == 'F') && raw[1] == '\0') {
        return false;
    }
    return true;
}

static inline int attention_stage_trace_layer() {
    const char* raw = std::getenv("TALU_DBG_ATTN_LAYER");
    if (raw == nullptr || raw[0] == '\0') return -1;
    return std::atoi(raw);
}

static inline bool should_trace_attention_stage(size_t layer_idx, size_t pos_offset, bool is_prefill, int seq_len) {
    if (!attention_stage_trace_enabled()) return false;
    (void)is_prefill;
    (void)seq_len;
    if (pos_offset == 0) return false;
    const int wanted = attention_stage_trace_layer();
    if (wanted >= 0 && static_cast<int>(layer_idx) != wanted) return false;
    return true;
}

static inline uint64_t attention_stage_checksum_sampled(const array& values, float* min_out, float* max_out) {
    array fp32 = (values.dtype() == float32) ? values : astype(values, float32);
    int count = 1;
    for (int axis = 0; axis < fp32.ndim(); ++axis) {
        count *= fp32.shape(axis);
    }
    if (count <= 0) {
        *min_out = 0.0f;
        *max_out = 0.0f;
        return 0xcbf29ce484222325ULL;
    }
    fp32 = reshape(fp32, {count});
    eval(fp32);
    const float* data = fp32.data<float>();
    const int sample = std::min(count, 256);
    uint64_t h = 0xcbf29ce484222325ULL;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < sample; ++i) {
        const int idx = (sample <= 1) ? 0 : static_cast<int>((static_cast<long long>(i) * static_cast<long long>(count - 1)) / static_cast<long long>(sample - 1));
        const float v = data[idx];
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(uint32_t));
        h ^= (static_cast<uint64_t>(idx) << 32) ^ static_cast<uint64_t>(bits);
        h *= 0x100000001b3ULL;
    }
    *min_out = mn;
    *max_out = mx;
    return h;
}

static inline void trace_attention_stage(
    const char* stage,
    const array& values,
    size_t layer_idx,
    size_t pos_offset,
    bool is_prefill,
    int seq_len
) {
    if (!should_trace_attention_stage(layer_idx, pos_offset, is_prefill, seq_len)) return;
    float mn = 0.0f;
    float mx = 0.0f;
    const uint64_t checksum = attention_stage_checksum_sampled(values, &mn, &mx);
    std::fprintf(
        stderr,
        "PARITY_ATTN stage=%s layer=%zu pos=%zu prefill=%d seq=%d checksum=0x%016llx min=%.6f max=%.6f\n",
        stage,
        layer_idx,
        pos_offset,
        is_prefill ? 1 : 0,
        seq_len,
        static_cast<unsigned long long>(checksum),
        static_cast<double>(mn),
        static_cast<double>(mx)
    );
}

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
    return tls_never_destroyed<std::unordered_map<AttentionFusedQkvCacheKey, array, AttentionFusedQkvCacheKeyHash>>();
}

struct AttentionFusedQkvHotSlot {
    bool valid = false;
    AttentionFusedQkvCacheKey key;
    array value = array(0.0f);
};

static inline AttentionFusedQkvHotSlot* attention_fused_qkv_hot_slots_store() {
    return tls_never_destroyed_array<AttentionFusedQkvHotSlot, kAttentionFusedQkvHotSlots>();
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
    return tls_never_destroyed<std::unordered_map<FfnQuantizedGateUpCacheKey, FfnQuantizedGateUpCacheValue, FfnQuantizedGateUpCacheKeyHash>>();
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
    return tls_never_destroyed_array<FfnQuantizedGateUpHotSlot, kFfnQuantizedGateUpHotSlots>();
}

static inline std::optional<FfnQuantizedGateUpCacheValue>& ffn_quantized_gate_up_scratch_store() {
    return tls_never_destroyed<std::optional<FfnQuantizedGateUpCacheValue>>();
}

static inline void ffn_quantized_gate_up_cache_clear() {
    ffn_quantized_gate_up_cache_store().clear();
    auto* hot_slots = ffn_quantized_gate_up_hot_slots_store();
    for (size_t i = 0; i < kFfnQuantizedGateUpHotSlots; ++i) {
        hot_slots[i].valid = false;
    }
    ffn_quantized_gate_up_scratch_store().reset();
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
        auto& scratch = ffn_quantized_gate_up_scratch_store();
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
    eval({fused.weights, fused.scales, fused.biases});
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

static inline bool use_attention_reference_mode() {
    const char* raw = std::getenv("TALU_METAL_ATTN_REFERENCE");
    if (raw == nullptr || raw[0] == '\0') return false;
    if (std::strcmp(raw, "0") == 0) return false;
    if (std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0) return false;
    if (std::strcmp(raw, "off") == 0 || std::strcmp(raw, "OFF") == 0) return false;
    if (std::strcmp(raw, "no") == 0 || std::strcmp(raw, "NO") == 0) return false;
    return true;
}

static inline array attention_owned_f32_array(const std::vector<float>& values, const Shape& shape) {
    static constexpr size_t kAlignment = 16 * 1024;
    const size_t element_count = std::max<size_t>(values.size(), 1);
    const size_t byte_count = element_count * sizeof(float);
    void* aligned_ptr = nullptr;
    if (posix_memalign(&aligned_ptr, kAlignment, byte_count) != 0 || aligned_ptr == nullptr) {
        throw std::bad_alloc();
    }
    auto* copied = static_cast<float*>(aligned_ptr);
    if (!values.empty()) {
        std::memcpy(copied, values.data(), values.size() * sizeof(float));
    } else {
        copied[0] = 0.0f;
    }
    auto owner = std::shared_ptr<void>(aligned_ptr, [](void* ptr) {
        std::free(ptr);
    });
    auto deleter = [owner](void*) {
        // Ownership retained by closure capture.
    };
    return array(copied, shape, float32, deleter);
}

static inline array cpu_like_rmsnorm_rows(
    const array& values,
    const array& weight,
    float eps
) {
    if (values.ndim() != 4) {
        return fast::rms_norm(values, weight, eps);
    }
    const int bsz = values.shape(0);
    const int heads = values.shape(1);
    const int seq_len = values.shape(2);
    const int dim = values.shape(3);
    array values_f32 = (values.dtype() == float32) ? values : astype(values, float32);
    array weight_f32 = (weight.dtype() == float32) ? weight : astype(weight, float32);
    array flat_values = reshape(values_f32, {bsz * heads * seq_len * dim});
    array flat_weight = reshape(weight_f32, {dim});
    eval(flat_values, flat_weight);
    const float* values_ptr = flat_values.data<float>();
    const float* weight_ptr = flat_weight.data<float>();
    std::vector<float> out(static_cast<size_t>(bsz) * static_cast<size_t>(heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(dim));
    const size_t row_count = static_cast<size_t>(bsz) * static_cast<size_t>(heads) * static_cast<size_t>(seq_len);
    const float dim_f = static_cast<float>(dim);
    for (size_t row = 0; row < row_count; ++row) {
        const size_t row_base = row * static_cast<size_t>(dim);
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float x = values_ptr[row_base + static_cast<size_t>(d)];
            sum_sq += x * x;
        }
        const float inv_rms = 1.0f / std::sqrt((sum_sq / dim_f) + eps);
        for (int d = 0; d < dim; ++d) {
            out[row_base + static_cast<size_t>(d)] =
                values_ptr[row_base + static_cast<size_t>(d)] * inv_rms * weight_ptr[static_cast<size_t>(d)];
        }
    }
    return attention_owned_f32_array(out, Shape{bsz, heads, seq_len, dim});
}

static inline void cpu_like_apply_static_rope(
    array* q,
    array* k,
    int seq_len,
    int head_dim,
    int rope_dim,
    size_t pos_offset,
    float rope_theta
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for cpu_like rope");
    }
    if (q->ndim() != 4 || k->ndim() != 4) {
        throw std::invalid_argument("[attention] cpu_like rope expects rank-4 q/k");
    }
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
        throw std::invalid_argument("[attention] cpu_like rope invalid rope_dim");
    }
    const int bsz = q->shape(0);
    const int q_heads = q->shape(1);
    const int q_seq = q->shape(2);
    const int q_dim = q->shape(3);
    const int k_heads = k->shape(1);
    const int k_seq = k->shape(2);
    const int k_dim = k->shape(3);
    if (q_seq != seq_len || k_seq != seq_len || q_dim != head_dim || k_dim != head_dim) {
        throw std::invalid_argument("[attention] cpu_like rope q/k shape mismatch");
    }
    array q_f32 = (q->dtype() == float32) ? *q : astype(*q, float32);
    array k_f32 = (k->dtype() == float32) ? *k : astype(*k, float32);
    array q_flat = reshape(q_f32, {bsz * q_heads * seq_len * head_dim});
    array k_flat = reshape(k_f32, {bsz * k_heads * seq_len * head_dim});
    eval(q_flat, k_flat);
    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();
    std::vector<float> q_out(static_cast<size_t>(bsz) * static_cast<size_t>(q_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::vector<float> k_out(static_cast<size_t>(bsz) * static_cast<size_t>(k_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::memcpy(q_out.data(), q_ptr, q_out.size() * sizeof(float));
    std::memcpy(k_out.data(), k_ptr, k_out.size() * sizeof(float));

    const int half_dim = rope_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int i = 0; i < half_dim; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(rope_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta, exponent);
    }

    for (int s = 0; s < seq_len; ++s) {
        const size_t pos = pos_offset + static_cast<size_t>(s);
        for (int i = 0; i < half_dim; ++i) {
            const float angle = static_cast<float>(pos) * inv_freq[static_cast<size_t>(i)];
            const float c = std::cos(angle);
            const float sn = std::sin(angle);
            for (int b = 0; b < bsz; ++b) {
                for (int h = 0; h < q_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(q_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t a = base + static_cast<size_t>(i);
                    const size_t bidx = base + static_cast<size_t>(half_dim + i);
                    const float x0 = q_out[a];
                    const float x1 = q_out[bidx];
                    q_out[a] = x0 * c - x1 * sn;
                    q_out[bidx] = x1 * c + x0 * sn;
                }
                for (int h = 0; h < k_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(k_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t a = base + static_cast<size_t>(i);
                    const size_t bidx = base + static_cast<size_t>(half_dim + i);
                    const float x0 = k_out[a];
                    const float x1 = k_out[bidx];
                    k_out[a] = x0 * c - x1 * sn;
                    k_out[bidx] = x1 * c + x0 * sn;
                }
            }
        }
    }

    *q = attention_owned_f32_array(q_out, Shape{bsz, q_heads, seq_len, head_dim});
    *k = attention_owned_f32_array(k_out, Shape{bsz, k_heads, seq_len, head_dim});
}

static inline void cpu_like_apply_static_rope_interleaved(
    array* q,
    array* k,
    int seq_len,
    int head_dim,
    int rope_dim,
    size_t pos_offset,
    float rope_theta
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for cpu_like interleaved rope");
    }
    if (q->ndim() != 4 || k->ndim() != 4) {
        throw std::invalid_argument("[attention] cpu_like interleaved rope expects rank-4 q/k");
    }
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
        throw std::invalid_argument("[attention] cpu_like interleaved rope invalid rope_dim");
    }
    const int bsz = q->shape(0);
    const int q_heads = q->shape(1);
    const int q_seq = q->shape(2);
    const int q_dim = q->shape(3);
    const int k_heads = k->shape(1);
    const int k_seq = k->shape(2);
    const int k_dim = k->shape(3);
    if (q_seq != seq_len || k_seq != seq_len || q_dim != head_dim || k_dim != head_dim) {
        throw std::invalid_argument("[attention] cpu_like interleaved rope q/k shape mismatch");
    }
    array q_f32 = (q->dtype() == float32) ? *q : astype(*q, float32);
    array k_f32 = (k->dtype() == float32) ? *k : astype(*k, float32);
    array q_flat = reshape(q_f32, {bsz * q_heads * seq_len * head_dim});
    array k_flat = reshape(k_f32, {bsz * k_heads * seq_len * head_dim});
    eval(q_flat, k_flat);
    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();
    std::vector<float> q_out(static_cast<size_t>(bsz) * static_cast<size_t>(q_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::vector<float> k_out(static_cast<size_t>(bsz) * static_cast<size_t>(k_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::memcpy(q_out.data(), q_ptr, q_out.size() * sizeof(float));
    std::memcpy(k_out.data(), k_ptr, k_out.size() * sizeof(float));

    const int half_dim = rope_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(half_dim));
    for (int i = 0; i < half_dim; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(rope_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(rope_theta, exponent);
    }

    for (int s = 0; s < seq_len; ++s) {
        const size_t pos = pos_offset + static_cast<size_t>(s);
        for (int i = 0; i < half_dim; ++i) {
            const float angle = static_cast<float>(pos) * inv_freq[static_cast<size_t>(i)];
            const float c = std::cos(angle);
            const float sn = std::sin(angle);
            const size_t d0 = static_cast<size_t>(i) * 2;
            const size_t d1 = d0 + 1;
            for (int b = 0; b < bsz; ++b) {
                for (int h = 0; h < q_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(q_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t i0 = base + d0;
                    const size_t i1 = base + d1;
                    const float x0 = q_out[i0];
                    const float x1 = q_out[i1];
                    q_out[i0] = x0 * c - x1 * sn;
                    q_out[i1] = x1 * c + x0 * sn;
                }
                for (int h = 0; h < k_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(k_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t i0 = base + d0;
                    const size_t i1 = base + d1;
                    const float x0 = k_out[i0];
                    const float x1 = k_out[i1];
                    k_out[i0] = x0 * c - x1 * sn;
                    k_out[i1] = x1 * c + x0 * sn;
                }
            }
        }
    }

    *q = attention_owned_f32_array(q_out, Shape{bsz, q_heads, seq_len, head_dim});
    *k = attention_owned_f32_array(k_out, Shape{bsz, k_heads, seq_len, head_dim});
}

static inline void cpu_like_apply_runtime_rope(
    array* q,
    array* k,
    const array& cos_rows,
    const array& sin_rows,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for cpu_like runtime rope");
    }
    if (q->ndim() != 4 || k->ndim() != 4) {
        throw std::invalid_argument("[attention] cpu_like runtime rope expects rank-4 q/k");
    }
    if (cos_rows.ndim() != 2 || sin_rows.ndim() != 2) {
        throw std::invalid_argument("[attention] cpu_like runtime rope expects rank-2 cos/sin tables");
    }
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
        throw std::invalid_argument("[attention] cpu_like runtime rope invalid rope_dim");
    }
    if (cos_rows.shape(0) != seq_len || sin_rows.shape(0) != seq_len ||
        cos_rows.shape(1) < rope_dim || sin_rows.shape(1) < rope_dim) {
        throw std::invalid_argument("[attention] cpu_like runtime rope table shape mismatch");
    }
    const int bsz = q->shape(0);
    const int q_heads = q->shape(1);
    const int q_seq = q->shape(2);
    const int q_dim = q->shape(3);
    const int k_heads = k->shape(1);
    const int k_seq = k->shape(2);
    const int k_dim = k->shape(3);
    if (q_seq != seq_len || k_seq != seq_len || q_dim != head_dim || k_dim != head_dim) {
        throw std::invalid_argument("[attention] cpu_like runtime rope q/k shape mismatch");
    }

    array q_f32 = (q->dtype() == float32) ? *q : astype(*q, float32);
    array k_f32 = (k->dtype() == float32) ? *k : astype(*k, float32);
    array cos_f32 = (cos_rows.dtype() == float32) ? cos_rows : astype(cos_rows, float32);
    array sin_f32 = (sin_rows.dtype() == float32) ? sin_rows : astype(sin_rows, float32);

    array q_flat = reshape(q_f32, {bsz * q_heads * seq_len * head_dim});
    array k_flat = reshape(k_f32, {bsz * k_heads * seq_len * head_dim});
    array cos_flat = reshape(cos_f32, {seq_len * rope_dim});
    array sin_flat = reshape(sin_f32, {seq_len * rope_dim});
    eval(q_flat, k_flat, cos_flat, sin_flat);

    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();
    const float* cos_ptr = cos_flat.data<float>();
    const float* sin_ptr = sin_flat.data<float>();

    std::vector<float> q_out(static_cast<size_t>(bsz) * static_cast<size_t>(q_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::vector<float> k_out(static_cast<size_t>(bsz) * static_cast<size_t>(k_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::memcpy(q_out.data(), q_ptr, q_out.size() * sizeof(float));
    std::memcpy(k_out.data(), k_ptr, k_out.size() * sizeof(float));

    const int half = rope_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        const size_t table_base = static_cast<size_t>(s) * static_cast<size_t>(rope_dim);
        for (int idx = 0; idx < half; ++idx) {
            const float c0 = cos_ptr[table_base + static_cast<size_t>(idx)];
            const float s0 = sin_ptr[table_base + static_cast<size_t>(idx)];
            const float c1 = cos_ptr[table_base + static_cast<size_t>(half + idx)];
            const float s1 = sin_ptr[table_base + static_cast<size_t>(half + idx)];
            for (int b = 0; b < bsz; ++b) {
                for (int h = 0; h < q_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(q_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t i0 = base + static_cast<size_t>(idx);
                    const size_t i1 = base + static_cast<size_t>(half + idx);
                    const float x0 = q_out[i0];
                    const float x1 = q_out[i1];
                    q_out[i0] = x0 * c0 - x1 * s0;
                    q_out[i1] = x1 * c1 + x0 * s1;
                }
                for (int h = 0; h < k_heads; ++h) {
                    const size_t base = (((static_cast<size_t>(b) * static_cast<size_t>(k_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) * static_cast<size_t>(head_dim));
                    const size_t i0 = base + static_cast<size_t>(idx);
                    const size_t i1 = base + static_cast<size_t>(half + idx);
                    const float x0 = k_out[i0];
                    const float x1 = k_out[i1];
                    k_out[i0] = x0 * c0 - x1 * s0;
                    k_out[i1] = x1 * c1 + x0 * s1;
                }
            }
        }
    }

    *q = attention_owned_f32_array(q_out, Shape{bsz, q_heads, seq_len, head_dim});
    *k = attention_owned_f32_array(k_out, Shape{bsz, k_heads, seq_len, head_dim});
}

static inline void cpu_like_apply_runtime_rope_interleaved(
    array* q,
    array* k,
    const array& cos_rows,
    const array& sin_rows,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for cpu_like runtime rope");
    }
    if (q->ndim() != 4 || k->ndim() != 4) {
        throw std::invalid_argument("[attention] cpu_like runtime rope expects rank-4 q/k");
    }
    if (cos_rows.ndim() != 2 || sin_rows.ndim() != 2) {
        throw std::invalid_argument("[attention] cpu_like runtime rope expects rank-2 cos/sin tables");
    }
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
        throw std::invalid_argument("[attention] cpu_like runtime rope invalid rope_dim");
    }
    if (cos_rows.shape(0) != seq_len || sin_rows.shape(0) != seq_len ||
        cos_rows.shape(1) < rope_dim || sin_rows.shape(1) < rope_dim) {
        throw std::invalid_argument("[attention] cpu_like runtime rope table shape mismatch");
    }
    const int bsz = q->shape(0);
    const int q_heads = q->shape(1);
    const int q_seq = q->shape(2);
    const int q_dim = q->shape(3);
    const int k_heads = k->shape(1);
    const int k_seq = k->shape(2);
    const int k_dim = k->shape(3);
    if (q_seq != seq_len || k_seq != seq_len || q_dim != head_dim || k_dim != head_dim) {
        throw std::invalid_argument("[attention] cpu_like runtime rope q/k shape mismatch");
    }

    array q_f32 = (q->dtype() == float32) ? *q : astype(*q, float32);
    array k_f32 = (k->dtype() == float32) ? *k : astype(*k, float32);
    array cos_f32 = (cos_rows.dtype() == float32) ? cos_rows : astype(cos_rows, float32);
    array sin_f32 = (sin_rows.dtype() == float32) ? sin_rows : astype(sin_rows, float32);

    array q_flat = reshape(q_f32, {bsz * q_heads * seq_len * head_dim});
    array k_flat = reshape(k_f32, {bsz * k_heads * seq_len * head_dim});
    array cos_flat = reshape(cos_f32, {seq_len * rope_dim});
    array sin_flat = reshape(sin_f32, {seq_len * rope_dim});
    eval(q_flat, k_flat, cos_flat, sin_flat);

    const float* q_ptr = q_flat.data<float>();
    const float* k_ptr = k_flat.data<float>();
    const float* cos_ptr = cos_flat.data<float>();
    const float* sin_ptr = sin_flat.data<float>();

    std::vector<float> q_out(static_cast<size_t>(bsz) * static_cast<size_t>(q_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::vector<float> k_out(static_cast<size_t>(bsz) * static_cast<size_t>(k_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(head_dim));
    std::memcpy(q_out.data(), q_ptr, q_out.size() * sizeof(float));
    std::memcpy(k_out.data(), k_ptr, k_out.size() * sizeof(float));

    const int half = rope_dim / 2;
    for (int s = 0; s < seq_len; ++s) {
        const size_t table_base = static_cast<size_t>(s) * static_cast<size_t>(rope_dim);
        for (int pair_idx = 0; pair_idx < half; ++pair_idx) {
            const float c = cos_ptr[table_base + static_cast<size_t>(pair_idx)];
            const float sn = sin_ptr[table_base + static_cast<size_t>(pair_idx)];
            const size_t d0 = static_cast<size_t>(pair_idx) * 2;
            const size_t d1 = d0 + 1;
            for (int b = 0; b < bsz; ++b) {
                for (int h = 0; h < q_heads; ++h) {
                    const size_t base =
                        (((static_cast<size_t>(b) * static_cast<size_t>(q_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) *
                         static_cast<size_t>(head_dim));
                    const size_t i0 = base + d0;
                    const size_t i1 = base + d1;
                    const float x0 = q_out[i0];
                    const float x1 = q_out[i1];
                    q_out[i0] = x0 * c - x1 * sn;
                    q_out[i1] = x1 * c + x0 * sn;
                }
                for (int h = 0; h < k_heads; ++h) {
                    const size_t base =
                        (((static_cast<size_t>(b) * static_cast<size_t>(k_heads) + static_cast<size_t>(h)) * static_cast<size_t>(seq_len) + static_cast<size_t>(s)) *
                         static_cast<size_t>(head_dim));
                    const size_t i0 = base + d0;
                    const size_t i1 = base + d1;
                    const float x0 = k_out[i0];
                    const float x1 = k_out[i1];
                    k_out[i0] = x0 * c - x1 * sn;
                    k_out[i1] = x1 * c + x0 * sn;
                }
            }
        }
    }

    *q = attention_owned_f32_array(q_out, Shape{bsz, q_heads, seq_len, head_dim});
    *k = attention_owned_f32_array(k_out, Shape{bsz, k_heads, seq_len, head_dim});
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

static std::optional<array> reshape_qkv_to_attention_layout(
    array* q,
    array* k,
    array* v,
    int batch_size,
    int seq_len,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    bool query_gate
) {
    if (q == nullptr || k == nullptr || v == nullptr) {
        throw std::invalid_argument("[attention] null qkv tensor");
    }

    auto canonicalize_projection = [&](array* values, const char* name) -> int {
        if (values == nullptr) {
            throw std::invalid_argument(std::string("[attention] null ") + name + " tensor");
        }
        if (values->ndim() == 0) {
            throw std::invalid_argument(std::string("[attention] scalar ") + name + " tensor is invalid");
        }

        if (values->ndim() == 3 && values->shape(0) == batch_size && values->shape(1) == seq_len) {
            return values->shape(2);
        }

        if (values->ndim() == 4 && values->shape(0) == batch_size && values->shape(1) == seq_len) {
            const int merged_dim = values->shape(2) * values->shape(3);
            *values = reshape(*values, {batch_size, seq_len, merged_dim});
            return merged_dim;
        }

        if (values->ndim() == 4 && values->shape(0) == batch_size && values->shape(2) == seq_len) {
            const int merged_dim = values->shape(1) * values->shape(3);
            auto transposed = transpose(*values, {0, 2, 1, 3});
            *values = reshape(transposed, {batch_size, seq_len, merged_dim});
            return merged_dim;
        }

        if (values->ndim() == 3 && values->shape(0) == batch_size && values->shape(2) == seq_len) {
            *values = transpose(*values, {0, 2, 1});
            return values->shape(2);
        }

        throw std::invalid_argument(std::string("[attention] unsupported ") + name + " projection layout");
    };

    const int q_dim = canonicalize_projection(q, "query");
    const int k_dim = canonicalize_projection(k, "key");
    const int v_dim = canonicalize_projection(v, "value");

    const int q_out = n_heads * head_dim;
    const int q_packed_out = q_out * 2;
    const int kv_out = n_kv_heads * head_dim;
    const bool has_query_gate = query_gate || q_dim == q_packed_out;
    if (has_query_gate) {
        if (q_dim != q_packed_out) {
            throw std::invalid_argument("[attention] gated query projection width mismatch");
        }
    } else if (q_dim != q_out) {
        throw std::invalid_argument("[attention] query projection width mismatch");
    }
    if (k_dim != kv_out || v_dim != kv_out) {
        throw std::invalid_argument("[attention] key/value projection width mismatch");
    }

    std::optional<array> query_gates = std::nullopt;
    if (has_query_gate) {
        // Keep parity with CPU compactQueryProjection/applyOutputGateInPlace:
        // query lanes are packed per head as [q_head, gate_head].
        auto q_packed = reshape(*q, {batch_size, seq_len, n_heads, 2, head_dim});
        auto q_values = slice(q_packed, {0, 0, 0, 0, 0}, {batch_size, seq_len, n_heads, 1, head_dim});
        auto q_gate = slice(q_packed, {0, 0, 0, 1, 0}, {batch_size, seq_len, n_heads, 2, head_dim});
        *q = reshape(q_values, {batch_size, seq_len, n_heads, head_dim});
        query_gates = reshape(q_gate, {batch_size, seq_len, n_heads, head_dim});
    }

    if (batch_size == 1 && seq_len == 1) {
        *q = reshape(*q, {1, n_heads, 1, head_dim});
        *k = reshape(*k, {1, n_kv_heads, 1, head_dim});
        *v = reshape(*v, {1, n_kv_heads, 1, head_dim});
        if (query_gates.has_value()) {
            query_gates = reshape(query_gates.value(), {1, n_heads, 1, head_dim});
        }
        return query_gates;
    }
    if (!has_query_gate) {
        *q = reshape(*q, {batch_size, seq_len, n_heads, head_dim});
    }
    *k = reshape(*k, {batch_size, seq_len, n_kv_heads, head_dim});
    *v = reshape(*v, {batch_size, seq_len, n_kv_heads, head_dim});
    *q = transpose(*q, {0, 2, 1, 3});
    *k = transpose(*k, {0, 2, 1, 3});
    *v = transpose(*v, {0, 2, 1, 3});
    if (query_gates.has_value()) {
        query_gates = transpose(query_gates.value(), {0, 2, 1, 3});
    }
    return query_gates;
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
        const auto& q_weight = *static_cast<const array*>(q_norm_w);
        const int q_width = q->shape(q->ndim() - 1);
        const array q_norm = canonicalize_rms_norm_weight(q_weight, q_width, "attention.q_norm");
        *q = use_attention_reference_mode() ? cpu_like_rmsnorm_rows(*q, q_norm, rms_eps) : fast::rms_norm(*q, q_norm, rms_eps);
    }
    if (k_norm_w != nullptr) {
        const auto& k_weight = *static_cast<const array*>(k_norm_w);
        const int k_width = k->shape(k->ndim() - 1);
        const array k_norm = canonicalize_rms_norm_weight(k_weight, k_width, "attention.k_norm");
        *k = use_attention_reference_mode() ? cpu_like_rmsnorm_rows(*k, k_norm, rms_eps) : fast::rms_norm(*k, k_norm, rms_eps);
    }
}

static void apply_attention_rope(
    array* q,
    array* k,
    int seq_len,
    int head_dim,
    int rope_width,
    size_t pos_offset,
    float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos,
    const void* runtime_rope_sin,
    size_t runtime_rope_dim
) {
    if (q == nullptr || k == nullptr) {
        throw std::invalid_argument("[attention] null qk tensor for rope");
    }

    if (rope_width <= 0 || rope_width > head_dim) {
        rope_width = head_dim;
    }

    bool use_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr;
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    if (use_runtime_rope && (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0)) {
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
            if (use_attention_reference_mode()) {
                if (rope_interleaved) {
                    cpu_like_apply_runtime_rope_interleaved(
                        q,
                        k,
                        cos_rows,
                        sin_rows,
                        seq_len,
                        head_dim,
                        rope_dim
                    );
                } else {
                    cpu_like_apply_runtime_rope(
                        q,
                        k,
                        cos_rows,
                        sin_rows,
                        seq_len,
                        head_dim,
                        rope_dim
                    );
                }
            } else {
                if (rope_interleaved) {
                    cpu_like_apply_runtime_rope_interleaved(
                        q,
                        k,
                        cos_rows,
                        sin_rows,
                        seq_len,
                        head_dim,
                        rope_dim
                    );
                } else {
                    *q = apply_runtime_rope_to_tensor(*q, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
                    *k = apply_runtime_rope_to_tensor(*k, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
                }
            }
        } else {
            use_runtime_rope = false;
        }
    }
    if (!use_runtime_rope) {
        if (use_attention_reference_mode()) {
            if (rope_interleaved) {
                cpu_like_apply_static_rope_interleaved(q, k, seq_len, head_dim, rope_width, pos_offset, rope_theta);
            } else {
                cpu_like_apply_static_rope(q, k, seq_len, head_dim, rope_width, pos_offset, rope_theta);
            }
        } else {
            if (rope_interleaved) {
                cpu_like_apply_static_rope_interleaved(q, k, seq_len, head_dim, rope_width, pos_offset, rope_theta);
            } else {
                *q = fast::rope(*q, rope_width, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
                *k = fast::rope(*k, rope_width, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
            }
        }
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
    const int seq_len = (k->ndim() >= 3) ? k->shape(2) : 0;

    const char* cache_bypass = std::getenv("TALU_METAL_ATTN_BYPASS_CACHE");
    if (cache_bypass != nullptr && cache_bypass[0] != '\0' &&
        std::strcmp(cache_bypass, "0") != 0 &&
        std::strcmp(cache_bypass, "false") != 0 &&
        std::strcmp(cache_bypass, "FALSE") != 0 &&
        std::strcmp(cache_bypass, "off") != 0 &&
        std::strcmp(cache_bypass, "OFF") != 0 &&
        std::strcmp(cache_bypass, "no") != 0 &&
        std::strcmp(cache_bypass, "NO") != 0) {
        return;
    }

    if (cache_ptr == nullptr) {
        return;
    }

    const bool use_reference_cache_io = []() -> bool {
        const char* explicit_ref = std::getenv("TALU_METAL_ATTN_REFERENCE_CACHEIO");
        if (explicit_ref != nullptr && explicit_ref[0] != '\0' &&
            std::strcmp(explicit_ref, "0") != 0 &&
            std::strcmp(explicit_ref, "false") != 0 &&
            std::strcmp(explicit_ref, "FALSE") != 0 &&
            std::strcmp(explicit_ref, "off") != 0 &&
            std::strcmp(explicit_ref, "OFF") != 0 &&
            std::strcmp(explicit_ref, "no") != 0 &&
            std::strcmp(explicit_ref, "NO") != 0) {
            return true;
        }
        const char* ref = std::getenv("TALU_METAL_ATTN_REFERENCE");
        return ref != nullptr && ref[0] != '\0' &&
            std::strcmp(ref, "0") != 0 &&
            std::strcmp(ref, "false") != 0 &&
            std::strcmp(ref, "FALSE") != 0 &&
            std::strcmp(ref, "off") != 0 &&
            std::strcmp(ref, "OFF") != 0 &&
            std::strcmp(ref, "no") != 0 &&
            std::strcmp(ref, "NO") != 0;
    }();

    if (use_reference_cache_io) {
        // Keep prefill attention numerics aligned with CPU: prefill computes
        // attention directly on the active K/V block (while still updating cache).
        if (seq_len > 1) {
            void* k_full_ptr_prefill = nullptr;
            void* v_full_ptr_prefill = nullptr;
            mlx_cache_get_bfloat16(cache_ptr, layer_idx, &k_full_ptr_prefill, &v_full_ptr_prefill);
            if (k_full_ptr_prefill == nullptr || v_full_ptr_prefill == nullptr) {
                mlx_cache_set_full_bfloat16(cache_ptr, layer_idx, k, v);
            } else {
                const auto& k_prev = *static_cast<const array*>(k_full_ptr_prefill);
                const auto& v_prev = *static_cast<const array*>(v_full_ptr_prefill);
                array k_next = concatenate(std::vector<array>{k_prev, *k}, 2);
                array v_next = concatenate(std::vector<array>{v_prev, *v}, 2);
                mlx_cache_set_full_bfloat16(cache_ptr, layer_idx, &k_next, &v_next);
            }
            *is_prefill = true;
            *k_for_attn = *k;
            *v_for_attn = *v;
            return;
        }
        void* k_full_ptr = nullptr;
        void* v_full_ptr = nullptr;
        mlx_cache_get_bfloat16(cache_ptr, layer_idx, &k_full_ptr, &v_full_ptr);
        if (k_full_ptr == nullptr || v_full_ptr == nullptr) {
            mlx_cache_set_full_bfloat16(cache_ptr, layer_idx, k, v);
            mlx_cache_get_bfloat16(cache_ptr, layer_idx, &k_full_ptr, &v_full_ptr);
            *is_prefill = true;
        } else {
            const auto& k_prev = *static_cast<const array*>(k_full_ptr);
            const auto& v_prev = *static_cast<const array*>(v_full_ptr);
            array k_next = concatenate(std::vector<array>{k_prev, *k}, 2);
            array v_next = concatenate(std::vector<array>{v_prev, *v}, 2);
            mlx_cache_set_full_bfloat16(cache_ptr, layer_idx, &k_next, &v_next);
            mlx_cache_get_bfloat16(cache_ptr, layer_idx, &k_full_ptr, &v_full_ptr);
            *is_prefill = (k_prev.shape(2) == 0);
        }
        if (k_full_ptr != nullptr && v_full_ptr != nullptr) {
            *k_for_attn = *static_cast<array*>(k_full_ptr);
            *v_for_attn = *static_cast<array*>(v_full_ptr);
        }
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

static array apply_query_output_gate(
    const array& attention_out,
    const std::optional<array>& query_gate_values
) {
    if (!query_gate_values.has_value()) {
        return attention_out;
    }
    auto gate_values = query_gate_values.value();
    if (gate_values.dtype() != float32) {
        gate_values = astype(gate_values, float32);
    }
    auto gate = sigmoid(gate_values);
    // Keep gate application in f32 to match CPU math, then cast back.
    auto attention_f32 = attention_out;
    if (attention_f32.dtype() != float32) {
        attention_f32 = astype(attention_f32, float32);
    }
    auto gated = attention_f32 * gate;
    if (gated.dtype() != attention_out.dtype()) {
        gated = astype(gated, attention_out.dtype());
    }
    return gated;
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
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier,   // 0 for default, >0 uses this directly as scale
    size_t sliding_window,
    bool query_gate
) {
    mlx_count_op();
    (void)sliding_window;
    const auto& x = *static_cast<const array*>(input);
    const int group_size_int = static_cast<int>(group_size);
    const int bits_int = static_cast<int>(bits);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const int rope_dim_int = (rope_dim > 0) ? static_cast<int>(rope_dim) : head_dim_int;
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
    trace_attention_stage("qkv_proj.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.k", projected.k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.v", projected.v, layer_idx, pos_offset, false, seq_len);

    const auto query_gate_values = reshape_qkv_to_attention_layout(
        &projected.q,
        &projected.k,
        &projected.v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int,
        query_gate
    );
    trace_attention_stage("qkv_layout.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.k", projected.k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.v", projected.v, layer_idx, pos_offset, false, seq_len);
    maybe_apply_qk_norm(&projected.q, &projected.k, q_norm_w, k_norm_w, rms_eps);
    trace_attention_stage("qk_norm.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qk_norm.k", projected.k, layer_idx, pos_offset, false, seq_len);
    apply_attention_rope(
        &projected.q,
        &projected.k,
        seq_len,
        head_dim_int,
        rope_dim_int,
        pos_offset,
        rope_theta,
        rope_interleaved,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );
    trace_attention_stage("rope.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("rope.k", projected.k, layer_idx, pos_offset, false, seq_len);

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
    trace_attention_stage("cache.k", k_for_attn, layer_idx, pos_offset, is_prefill, seq_len);
    trace_attention_stage("cache.v", v_for_attn, layer_idx, pos_offset, is_prefill, seq_len);

    auto attn_out = run_attention_kernel(
        projected.q,
        k_for_attn,
        v_for_attn,
        scale_value,
        is_prefill,
        attn_sinks
    );
    trace_attention_stage("sdpa.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = apply_query_output_gate(attn_out, query_gate_values);
    trace_attention_stage("gate.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, n_heads_int * head_dim_int);
    trace_attention_stage("attn.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);

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
    trace_attention_stage("o_proj.out", out, layer_idx, pos_offset, is_prefill, seq_len);

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
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier,   // 0 for default, >0 uses this directly as scale
    size_t sliding_window,
    bool query_gate
) {
    mlx_count_op();
    (void)sliding_window;
    const auto& x = *static_cast<const array*>(input);
    const int group_size_int = static_cast<int>(group_size);
    const int bits_int = static_cast<int>(bits);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const int rope_dim_int = (rope_dim > 0) ? static_cast<int>(rope_dim) : head_dim_int;
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
    trace_attention_stage("qkv_proj.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.k", projected.k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.v", projected.v, layer_idx, pos_offset, false, seq_len);

    const auto query_gate_values = reshape_qkv_to_attention_layout(
        &projected.q,
        &projected.k,
        &projected.v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int,
        query_gate
    );
    trace_attention_stage("qkv_layout.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.k", projected.k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.v", projected.v, layer_idx, pos_offset, false, seq_len);
    maybe_apply_qk_norm(&projected.q, &projected.k, q_norm_w, k_norm_w, rms_eps);
    trace_attention_stage("qk_norm.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qk_norm.k", projected.k, layer_idx, pos_offset, false, seq_len);
    apply_attention_rope(
        &projected.q,
        &projected.k,
        seq_len,
        head_dim_int,
        rope_dim_int,
        pos_offset,
        rope_theta,
        rope_interleaved,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );
    trace_attention_stage("rope.q", projected.q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("rope.k", projected.k, layer_idx, pos_offset, false, seq_len);

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
    trace_attention_stage("cache.k", k_for_attn, layer_idx, pos_offset, is_prefill, seq_len);
    trace_attention_stage("cache.v", v_for_attn, layer_idx, pos_offset, is_prefill, seq_len);

    auto attn_out = run_attention_kernel(
        projected.q,
        k_for_attn,
        v_for_attn,
        scale_value,
        is_prefill,
        attn_sinks
    );
    trace_attention_stage("sdpa.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = apply_query_output_gate(attn_out, query_gate_values);
    trace_attention_stage("gate.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, n_heads_int * head_dim_int);
    trace_attention_stage("attn.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);

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
    trace_attention_stage("o_proj.out", out, layer_idx, pos_offset, is_prefill, seq_len);

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

void* mlx_lazy_rms_norm_fused_ffn(
    const void* input,
    const void* norm_w,
    const void* gate_w, const void* gate_s, const void* gate_b,
    const void* up_w, const void* up_s, const void* up_b,
    const void* down_w, const void* down_s, const void* down_b,
    size_t group_size, size_t bits,
    bool use_gelu,
    float eps
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const auto& norm_weight_raw = *static_cast<const array*>(norm_w);
    const int in_features = x.shape(2);
    const array norm_weight = canonicalize_rms_norm_weight(norm_weight_raw, in_features, "rms_norm_fused_ffn");
    int gs = static_cast<int>(group_size);
    int b = static_cast<int>(bits);
    const auto& gate_w_arr = *static_cast<const array*>(gate_w);
    const auto& gate_s_arr = *static_cast<const array*>(gate_s);
    const auto& gate_b_arr = *static_cast<const array*>(gate_b);
    const auto& up_w_arr = *static_cast<const array*>(up_w);
    const auto& up_s_arr = *static_cast<const array*>(up_s);
    const auto& up_b_arr = *static_cast<const array*>(up_b);

    auto normed = fast::rms_norm(x, norm_weight, eps);
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
        normed,
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

    auto mid = [&]() -> array {
        if (use_gelu) {
            const float sqrt_2_over_pi = 0.7978845608f;
            auto x3 = gate * gate * gate;
            auto inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
            return 0.5f * gate * (1.0f + tanh(inner)) * up;
        } else {
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
    size_t rope_dim,
    size_t pos_offset, float rope_theta,
    bool rope_interleaved,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier,   // 0 for default, >0 uses this directly as scale
    size_t sliding_window,
    bool query_gate
) {
    mlx_count_op();
    (void)sliding_window;
    const auto& x = *static_cast<const array*>(input);
    const int batch_size = x.shape(0);
    const int seq_len = x.shape(1);
    const int hidden_dim = x.shape(2);
    const int n_heads_int = static_cast<int>(n_heads);
    const int n_kv_heads_int = static_cast<int>(n_kv_heads);
    const int head_dim_int = static_cast<int>(head_dim);
    const int rope_dim_int = (rope_dim > 0) ? static_cast<int>(rope_dim) : head_dim_int;
    const int q_out = n_heads_int * head_dim_int;
    const int q_proj_out = query_gate ? (q_out * 2) : q_out;
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
        q_proj_out,
        kv_out
    );
    auto o_wt = orient_matmul_rhs_cached(o_w, *static_cast<const array*>(o_w), q_out, hidden_dim, true);

    auto qkv = matmul_lastdim(x, qkv_wt);
    array q = slice(qkv, {0, 0, 0}, {batch_size, seq_len, q_proj_out});
    array k = slice(qkv, {0, 0, q_proj_out}, {batch_size, seq_len, q_proj_out + kv_out});
    array v = slice(qkv, {0, 0, q_proj_out + kv_out}, {batch_size, seq_len, q_proj_out + 2 * kv_out});

    maybe_add_linear_bias(&q, q_bias);
    maybe_add_linear_bias(&k, k_bias);
    maybe_add_linear_bias(&v, v_bias);
    trace_attention_stage("qkv_proj.q", q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.k", k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_proj.v", v, layer_idx, pos_offset, false, seq_len);

    const auto query_gate_values = reshape_qkv_to_attention_layout(
        &q,
        &k,
        &v,
        batch_size,
        seq_len,
        n_heads_int,
        n_kv_heads_int,
        head_dim_int,
        query_gate
    );
    trace_attention_stage("qkv_layout.q", q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.k", k, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qkv_layout.v", v, layer_idx, pos_offset, false, seq_len);
    maybe_apply_qk_norm(&q, &k, q_norm_w, k_norm_w, rms_eps);
    trace_attention_stage("qk_norm.q", q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("qk_norm.k", k, layer_idx, pos_offset, false, seq_len);
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
        rope_dim_int,
        pos_offset,
        rope_theta,
        rope_interleaved,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim
    );
    trace_attention_stage("rope.q", q, layer_idx, pos_offset, false, seq_len);
    trace_attention_stage("rope.k", k, layer_idx, pos_offset, false, seq_len);

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
    trace_attention_stage("cache.k", k_for_attn, layer_idx, pos_offset, is_prefill, seq_len);
    trace_attention_stage("cache.v", v_for_attn, layer_idx, pos_offset, is_prefill, seq_len);

    auto attn_out = run_attention_kernel(q, k_for_attn, v_for_attn, scale_value, is_prefill, attn_sinks);
    trace_attention_stage("sdpa.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = apply_query_output_gate(attn_out, query_gate_values);
    trace_attention_stage("gate.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);
    attn_out = reshape_attention_output(attn_out, batch_size, seq_len, q_out);
    trace_attention_stage("attn.out", attn_out, layer_idx, pos_offset, is_prefill, seq_len);

    // Output projection
    auto out = matmul_lastdim(attn_out, o_wt);

    // Add output bias if present
    if (o_bias != nullptr) {
        const auto& bias = *static_cast<const array*>(o_bias);
        out = out + cast_bias_cached(o_bias, bias, out.dtype());
    }
    trace_attention_stage("o_proj.out", out, layer_idx, pos_offset, is_prefill, seq_len);

    return pool_array(std::move(out));
}
