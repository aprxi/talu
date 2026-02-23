// MLX Bridge - Quantized (4-bit) Model
//
// Full transformer implementation using 4-bit quantized weights.
// Uses MLX's quantized_matmul for 8x memory bandwidth savings.

#include "compute_common.h"
#include "model_state.h"
#include <mutex>
#include <unordered_map>

// ============================================================================
// Quantized Model Structure
// ============================================================================

struct FusedModelWeights {
    struct Layer {
        enum class LayerKind : int {
            attention_mlp = 0,
            shortconv = 1,
            mamba = 2,
        };
        LayerKind kind = LayerKind::attention_mlp;

        array ln1_w = array(0.0f, float32);  // attention norm
        array q_w = array(0.0f, float32), q_s = array(0.0f, float32), q_b = array(0.0f, float32);
        array k_w = array(0.0f, float32), k_s = array(0.0f, float32), k_b = array(0.0f, float32);
        array v_w = array(0.0f, float32), v_s = array(0.0f, float32), v_b = array(0.0f, float32);
        array o_w = array(0.0f, float32), o_s = array(0.0f, float32), o_b = array(0.0f, float32);
        // MLA attention (quantized)
        bool use_mla = false;
        int mla_n_heads = 0;
        int mla_q_lora_rank = 0;
        int mla_kv_lora_rank = 0;
        int mla_qk_head_dim = 0;
        int mla_qk_rope_head_dim = 0;
        int mla_qk_nope_head_dim = 0;
        int mla_v_head_dim = 0;
        array mla_q_a_w = array(0.0f, float32), mla_q_a_s = array(0.0f, float32), mla_q_a_b = array(0.0f, float32);
        array mla_q_b_w = array(0.0f, float32), mla_q_b_s = array(0.0f, float32), mla_q_b_b = array(0.0f, float32);
        array mla_kv_a_w = array(0.0f, float32), mla_kv_a_s = array(0.0f, float32), mla_kv_a_b = array(0.0f, float32);
        array mla_kv_b_w = array(0.0f, float32), mla_kv_b_s = array(0.0f, float32), mla_kv_b_b = array(0.0f, float32);
        array mla_q_a_norm = array(0.0f, float32), mla_kv_a_norm = array(0.0f, float32);
        array mla_o_w = array(0.0f, float32), mla_o_s = array(0.0f, float32), mla_o_b = array(0.0f, float32);
        // ShortConv mixer projections (quantized) and convolution weights.
        int shortconv_d_conv = 0;
        int shortconv_conv_dim = 0;
        array shortconv_in_w = array(0.0f, float32), shortconv_in_s = array(0.0f, float32), shortconv_in_b = array(0.0f, float32);
        array shortconv_out_w = array(0.0f, float32), shortconv_out_s = array(0.0f, float32), shortconv_out_b = array(0.0f, float32);
        array shortconv_conv_w = array(0.0f, float32);
        std::optional<array> shortconv_conv_b;
        std::optional<array> shortconv_kernel_broadcast;
        std::optional<array> shortconv_bias_row;
        // Mamba mixer (quantized core; optional quantized FFN)
        int mamba_d_state = 0;
        int mamba_d_conv = 0;
        int mamba_n_heads = 0;
        int mamba_d_head = 0;
        int mamba_n_groups = 1;
        uint8_t mamba_gate_up_layout = 0;
        array mamba_conv_w = array(0.0f, float32);
        std::optional<array> mamba_conv_b;
        array mamba_a_log = array(0.0f, float32);
        array mamba_d_skip = array(0.0f, float32);
        std::optional<array> mamba_dt_bias;
        std::optional<array> mamba_norm_weight;
        array mamba_in_w = array(0.0f, float32), mamba_in_s = array(0.0f, float32), mamba_in_b = array(0.0f, float32);
        array mamba_out_w = array(0.0f, float32), mamba_out_s = array(0.0f, float32), mamba_out_b = array(0.0f, float32);
        std::optional<array> mamba_gate_up_w, mamba_gate_up_s, mamba_gate_up_b;
        std::optional<array> mamba_down_w, mamba_down_s, mamba_down_b;
        array ln2_w = array(0.0f, float32);  // ffn norm (or post_attention_layernorm)
        array gate_w = array(0.0f, float32), gate_s = array(0.0f, float32), gate_b = array(0.0f, float32);
        array up_w = array(0.0f, float32), up_s = array(0.0f, float32), up_b = array(0.0f, float32);
        array down_w = array(0.0f, float32), down_s = array(0.0f, float32), down_b = array(0.0f, float32);
        bool use_moe = false;
        array moe_router_w = array(0.0f, float32);
        std::optional<array> moe_router_s;
        std::optional<array> moe_router_b;
        std::optional<array> moe_router_bias;
        array moe_gate_w = array(0.0f, float32), moe_gate_s = array(0.0f, float32);
        array moe_up_w = array(0.0f, float32), moe_up_s = array(0.0f, float32);
        array moe_down_w = array(0.0f, float32), moe_down_s = array(0.0f, float32);
        std::optional<array> moe_gate_bias;
        std::optional<array> moe_up_bias;
        std::optional<array> moe_down_bias;
        int moe_num_experts = 0;
        int moe_experts_per_token = 0;
        int moe_router_group_size = 0;
        int moe_expert_group_size = 0;
        std::optional<array> q_norm;
        std::optional<array> k_norm;
        // Optional extra FFN norms (4 norms per block)
        std::optional<array> pre_ffn_norm;
        std::optional<array> post_ffn_norm;
        // Fused weights (created at model setup time for efficiency)
        // QKV fused: [hidden_dim, (n_heads + 2*n_kv_heads) * head_dim]
        std::optional<array> qkv_w, qkv_s, qkv_b;
        // Gate+Up fused: [hidden_dim, 2 * d_ff]
        std::optional<array> gate_up_w, gate_up_s, gate_up_b;
        bool use_fused_qkv = false;
        bool use_fused_gate_up = false;
    };
    std::vector<Layer> layers;

    array ln_final = array(0.0f, float32);
    array lm_head_w = array(0.0f, float32), lm_head_s = array(0.0f, float32), lm_head_b = array(0.0f, float32);
    array embed_w = array(0.0f, float32), embed_s = array(0.0f, float32), embed_b = array(0.0f, float32);

    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    int group_size = 0;
    int bits = 4;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;

    // Custom RoPE frequencies for scaled positional encoding
    std::optional<array> rope_freqs;

    // Architecture-specific config
    bool has_norm_weight_offset = false;  // (1+w) RMSNorm formulation
    bool use_gelu = false;  // GELU instead of SiLU
    float query_pre_attn_scalar = 0.0f;  // Custom attention scale

    // Custom scaling multipliers (data-driven from config.json)
    float embedding_multiplier = 1.0f;  // Scales embedding output
    float attention_multiplier = 0.0f;  // Custom attention scale (0 = use 1/sqrt(head_dim))
    float residual_multiplier = 1.0f;   // Scales residual connections
    float logits_scaling = 1.0f;        // Scales output logits

    bool topology_initialized = false;
};

static FusedModelWeights* g_fused_weights = nullptr;

static constexpr uint8_t kLayerKindAttentionMlp = 0;
static constexpr uint8_t kLayerKindShortConv = 1;
static constexpr uint8_t kLayerKindMamba = 2;

static FusedModelWeights::Layer::LayerKind decode_layer_kind(uint8_t kind_id) {
    switch (kind_id) {
        case kLayerKindAttentionMlp:
            return FusedModelWeights::Layer::LayerKind::attention_mlp;
        case kLayerKindShortConv:
            return FusedModelWeights::Layer::LayerKind::shortconv;
        case kLayerKindMamba:
            return FusedModelWeights::Layer::LayerKind::mamba;
        default:
            throw std::invalid_argument("Unsupported fused layer kind id");
    }
}

static inline void gqa_expand_attention_kv(
    const array& q,
    const array& k_in,
    const array& v_in,
    array* k_out,
    array* v_out
) {
    if (k_out == nullptr || v_out == nullptr) {
        throw std::invalid_argument("null gqa output");
    }
    const int q_heads = q.shape(1);
    const int kv_heads = k_in.shape(1);
    if (q_heads == kv_heads) {
        *k_out = k_in;
        *v_out = v_in;
        return;
    }
    if (kv_heads <= 0 || q_heads <= 0 || (q_heads % kv_heads) != 0 || v_in.shape(1) != kv_heads) {
        throw std::invalid_argument("invalid GQA head layout");
    }

    const int heads_per_kv = q_heads / kv_heads;
    std::vector<int32_t> gather_idx(static_cast<size_t>(q_heads));
    for (int head_idx = 0; head_idx < q_heads; head_idx++) {
        gather_idx[static_cast<size_t>(head_idx)] = static_cast<int32_t>(head_idx / heads_per_kv);
    }
    array idx = array(gather_idx.data(), {q_heads}, int32);
    *k_out = take(k_in, idx, 1);
    *v_out = take(v_in, idx, 1);
}

struct QuantizedPipelineState {
    std::optional<array> current_token;
};

static std::mutex g_quant_pipeline_mu;
static std::unordered_map<void*, QuantizedPipelineState> g_quant_pipeline_states;

static inline void* quant_decode_state_key(void* cache_ptr, void* model_ptr) {
    return cache_ptr != nullptr ? cache_ptr : model_ptr;
}

static int round_up_step(size_t value, int step) {
    const size_t s = static_cast<size_t>(step);
    return static_cast<int>(((value + s - 1) / s) * s);
}

static int next_cache_capacity(const CacheLayer& layer, size_t required, int current_capacity) {
    if (required == 0) return current_capacity;

    const size_t current = current_capacity > 0 ? static_cast<size_t>(current_capacity) : 0;

    if (layer.max_seq_len > 0) {
        const size_t max_cap = layer.max_seq_len;
        if (required > max_cap) {
            throw std::invalid_argument("[fused] kv cache capacity exceeded");
        }
        if (current == 0) return static_cast<int>(max_cap);
        if (required > current) return static_cast<int>(max_cap);
        return static_cast<int>(current);
    }

    const size_t step = static_cast<size_t>(layer.step);
    size_t capacity = current;
    if (capacity == 0) {
        const size_t base = std::max(required, step);
        capacity = static_cast<size_t>(round_up_step(base, layer.step));
    }
    while (capacity < required) {
        const size_t doubled = capacity * 2;
        if (doubled <= capacity) {
            throw std::invalid_argument("[fused] kv cache capacity exceeded");
        }
        capacity = doubled;
    }
    return static_cast<int>(capacity);
}

// Async decode API keeps a single thread-local pending token. This path is not
// used by the scheduler pipeline; scheduler state is keyed by cache/model above.
static thread_local std::optional<array> g_pending_token;

extern "C" void* mlx_lazy_fused_moe_ffn_mxfp4(
    const void* input,
    const void* router_w,
    const void* router_s,
    const void* router_b,
    const void* router_bias,
    const void* gate_w,
    const void* gate_s,
    const void* up_w,
    const void* up_s,
    const void* down_w,
    const void* down_s,
    const void* gate_bias,
    const void* up_bias,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t router_group_size,
    size_t expert_group_size);

extern "C" void* mlx_lazy_mla_attention_quantized(
    const void* input,
    const void* q_a_w, const void* q_a_s, const void* q_a_b, const void* q_a_norm_w,
    const void* q_b_w, const void* q_b_s, const void* q_b_b,
    const void* kv_a_w, const void* kv_a_s, const void* kv_a_b, const void* kv_a_norm_w,
    const void* kv_b_w, const void* kv_b_s, const void* kv_b_b,
    const void* o_w, const void* o_s, const void* o_b,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads,
    size_t q_lora_rank, size_t kv_lora_rank,
    size_t qk_head_dim, size_t qk_rope_head_dim, size_t qk_nope_head_dim, size_t v_head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits
);

extern "C" void* mlx_lazy_mamba_block_quantized(
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
    void* mamba_cache_ptr,
    size_t layer_idx,
    size_t d_state,
    size_t d_conv,
    size_t n_heads,
    size_t d_head,
    size_t n_groups,
    uint8_t gate_up_layout
);

// ============================================================================
// Forward Pass Implementation
// ============================================================================

static array fused_forward_logits_from_token(
    FusedModelWeights* fused_weights,
    MLXCache* cache_state,
    MLXShortConvCache* shortconv_cache_state,
    MLXMambaCache* mamba_cache_state,
    const array& token_idx,
    size_t pos_offset
) {
    const int group_size = fused_weights->group_size;
    const int bit_width = fused_weights->bits;
    const int n_layers = static_cast<int>(fused_weights->layers.size());

    // Attention scale:
    // - Custom attention_multiplier: use directly (e.g., 0.015625 = 1/64)
    // - query_pre_attn_scalar: use 1/sqrt(val) for custom scaling
    // - Default: use 1/sqrt(head_dim)
    const float attn_scale = (fused_weights->attention_multiplier > 0.0f)
        ? fused_weights->attention_multiplier
        : (fused_weights->query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(fused_weights->query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(fused_weights->head_dim)));
    const Shape q_shape = {1, 1, fused_weights->n_heads, fused_weights->head_dim};
    const Shape kv_shape = {1, 1, fused_weights->n_kv_heads, fused_weights->head_dim};
    const Shape attn_out_shape = {1, 1, fused_weights->n_heads * fused_weights->head_dim};

    // Embedding lookup (quantized)
    array idx = reshape(astype(token_idx, int32), {1, 1});
    array embed_rows = take(fused_weights->embed_w, idx, 0);
    array scale_rows = take(fused_weights->embed_s, idx, 0);
    array bias_rows = take(fused_weights->embed_b, idx, 0);
    array hidden = dequantize(embed_rows, scale_rows, bias_rows, group_size, bit_width, "affine");

    // Embedding scaling (architecture-specific)
    if (fused_weights->embedding_multiplier != 1.0f) {
        hidden = hidden * fused_weights->embedding_multiplier;
    } else if (fused_weights->has_norm_weight_offset) {
        // Architectures with (1+w) norms also use sqrt(hidden_dim) embedding scaling
        hidden = hidden * std::sqrt(static_cast<float>(fused_weights->hidden_dim));
    }

    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
        const auto& l = fused_weights->layers[layer_idx];
        if (l.kind == FusedModelWeights::Layer::LayerKind::mamba) {
            auto* mamba_out = static_cast<const array*>(mlx_lazy_mamba_block_quantized(
                &hidden,
                &l.ln1_w,
                &l.mamba_in_w,
                &l.mamba_in_s,
                &l.mamba_in_b,
                &l.mamba_conv_w,
                l.mamba_conv_b ? &*l.mamba_conv_b : nullptr,
                &l.mamba_a_log,
                &l.mamba_d_skip,
                l.mamba_dt_bias ? &*l.mamba_dt_bias : nullptr,
                l.mamba_norm_weight ? &*l.mamba_norm_weight : nullptr,
                &l.mamba_out_w,
                &l.mamba_out_s,
                &l.mamba_out_b,
                &l.ln2_w,
                l.mamba_gate_up_w ? &*l.mamba_gate_up_w : nullptr,
                l.mamba_gate_up_s ? &*l.mamba_gate_up_s : nullptr,
                l.mamba_gate_up_b ? &*l.mamba_gate_up_b : nullptr,
                l.mamba_down_w ? &*l.mamba_down_w : nullptr,
                l.mamba_down_s ? &*l.mamba_down_s : nullptr,
                l.mamba_down_b ? &*l.mamba_down_b : nullptr,
                static_cast<size_t>(group_size),
                static_cast<size_t>(bit_width),
                fused_weights->use_gelu,
                fused_weights->residual_multiplier,
                fused_weights->rms_eps,
                mamba_cache_state,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mamba_d_state),
                static_cast<size_t>(l.mamba_d_conv),
                static_cast<size_t>(l.mamba_n_heads),
                static_cast<size_t>(l.mamba_d_head),
                static_cast<size_t>(l.mamba_n_groups),
                l.mamba_gate_up_layout
            ));
            hidden = *mamba_out;
            continue;
        }
        auto& cl = cache_state->layers[layer_idx];

        array normed = fast::rms_norm(hidden, l.ln1_w, fused_weights->rms_eps);
        array attn_proj(0.0f, float32);
        if (l.kind == FusedModelWeights::Layer::LayerKind::shortconv) {
            const int d_conv_i = l.shortconv_d_conv;
            const int conv_dim_i = l.shortconv_conv_dim;

            array bcx = quantized_matmul(
                normed,
                l.shortconv_in_w,
                l.shortconv_in_s,
                l.shortconv_in_b,
                true,
                group_size,
                bit_width,
                "affine"
            );
            bcx = astype(bcx, float32);

            array b_gate = slice(bcx, {0, 0, 0}, {1, 1, conv_dim_i});
            array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, 1, 2 * conv_dim_i});
            array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, 1, 3 * conv_dim_i});
            array bx = b_gate * x_proj;

            array conv_state(0.0f, float32);
            if (shortconv_cache_state != nullptr &&
                layer_idx < static_cast<int>(shortconv_cache_state->layers.size())) {
                auto& scl = shortconv_cache_state->layers[layer_idx];
                const bool need_init =
                    scl.conv_state == nullptr ||
                    scl.conv_state->shape(1) != d_conv_i ||
                    scl.conv_state->shape(2) != conv_dim_i;
                if (need_init) {
                    delete scl.conv_state;
                    scl.conv_state = new array(zeros({1, d_conv_i, conv_dim_i}, float32));
                }
                conv_state = *scl.conv_state;
            } else {
                conv_state = zeros({1, d_conv_i, conv_dim_i}, float32);
            }

            const array bx_t = reshape(bx, {1, 1, conv_dim_i});
            if (d_conv_i > 1) {
                const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                conv_state = concatenate({state_tail, bx_t}, 1);
            } else {
                conv_state = bx_t;
            }

            array conv_kernel_broadcast = l.shortconv_kernel_broadcast
                ? *l.shortconv_kernel_broadcast
                : reshape(astype(transpose(l.shortconv_conv_w), float32), {1, d_conv_i, conv_dim_i});
            array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
            if (l.shortconv_bias_row) {
                conv_t = conv_t + *l.shortconv_bias_row;
            }

            const array c_t = reshape(c_gate, {1, conv_dim_i});
            const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
            attn_proj = quantized_matmul(
                gated,
                l.shortconv_out_w,
                l.shortconv_out_s,
                l.shortconv_out_b,
                true,
                group_size,
                bit_width,
                "affine"
            );

            if (shortconv_cache_state != nullptr &&
                layer_idx < static_cast<int>(shortconv_cache_state->layers.size())) {
                *shortconv_cache_state->layers[layer_idx].conv_state = conv_state;
            }
        } else if (l.use_mla) {
            auto* mla_out = static_cast<const array*>(mlx_lazy_mla_attention_quantized(
                &normed,
                &l.mla_q_a_w,
                &l.mla_q_a_s,
                &l.mla_q_a_b,
                &l.mla_q_a_norm,
                &l.mla_q_b_w,
                &l.mla_q_b_s,
                &l.mla_q_b_b,
                &l.mla_kv_a_w,
                &l.mla_kv_a_s,
                &l.mla_kv_a_b,
                &l.mla_kv_a_norm,
                &l.mla_kv_b_w,
                &l.mla_kv_b_s,
                &l.mla_kv_b_b,
                &l.mla_o_w,
                &l.mla_o_s,
                &l.mla_o_b,
                cache_state,
                static_cast<size_t>(layer_idx),
                static_cast<size_t>(l.mla_n_heads),
                static_cast<size_t>(l.mla_q_lora_rank),
                static_cast<size_t>(l.mla_kv_lora_rank),
                static_cast<size_t>(l.mla_qk_head_dim),
                static_cast<size_t>(l.mla_qk_rope_head_dim),
                static_cast<size_t>(l.mla_qk_nope_head_dim),
                static_cast<size_t>(l.mla_v_head_dim),
                pos_offset,
                fused_weights->rope_theta,
                nullptr,
                nullptr,
                0,
                fused_weights->rms_eps,
                static_cast<size_t>(group_size),
                static_cast<size_t>(bit_width)
            ));
            attn_proj = *mla_out;
        } else {
            array q(0.0f), k(0.0f), v(0.0f);
            if (l.use_fused_qkv && l.qkv_w) {
                // Fused QKV: single matmul then split
                array qkv = quantized_matmul(normed, *l.qkv_w, *l.qkv_s, *l.qkv_b, true, group_size, bit_width, "affine");
                // Split: q=[n_heads*head_dim], k=[n_kv_heads*head_dim], v=[n_kv_heads*head_dim]
                int q_dim = fused_weights->n_heads * fused_weights->head_dim;
                int kv_dim = fused_weights->n_kv_heads * fused_weights->head_dim;
                q = slice(qkv, {0, 0, 0}, {1, 1, q_dim});
                k = slice(qkv, {0, 0, q_dim}, {1, 1, q_dim + kv_dim});
                v = slice(qkv, {0, 0, q_dim + kv_dim}, {1, 1, q_dim + 2 * kv_dim});
            } else {
                q = quantized_matmul(normed, l.q_w, l.q_s, l.q_b, true, group_size, bit_width, "affine");
                k = quantized_matmul(normed, l.k_w, l.k_s, l.k_b, true, group_size, bit_width, "affine");
                v = quantized_matmul(normed, l.v_w, l.v_s, l.v_b, true, group_size, bit_width, "affine");
            }

            q = reshape(q, q_shape);
            k = reshape(k, kv_shape);
            v = reshape(v, kv_shape);

            if (l.q_norm) q = fast::rms_norm(q, *l.q_norm, fused_weights->rms_eps);
            if (l.k_norm) k = fast::rms_norm(k, *l.k_norm, fused_weights->rms_eps);

            q = transpose(q, g_transpose_perm);
            k = transpose(k, g_transpose_perm);
            v = transpose(v, g_transpose_perm);

            // Apply RoPE with custom frequencies (Llama3) or standard base
            if (fused_weights->rope_freqs) {
                q = fast::rope(q, fused_weights->head_dim, false, std::nullopt, 1.0f, static_cast<int>(pos_offset), fused_weights->rope_freqs);
                k = fast::rope(k, fused_weights->head_dim, false, std::nullopt, 1.0f, static_cast<int>(pos_offset), fused_weights->rope_freqs);
            } else {
                q = fast::rope(q, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, static_cast<int>(pos_offset));
                k = fast::rope(k, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, static_cast<int>(pos_offset));
            }

            // Cache update
            size_t prev = cl.offset;
            int offset = static_cast<int>(prev + 1);

            if (cl.k_bfloat16 == nullptr) {
                const int capacity = next_cache_capacity(cl, static_cast<size_t>(offset), 0);
                Shape shape = {1, fused_weights->n_kv_heads, capacity, fused_weights->head_dim};
                cl.k_bfloat16 = new array(zeros(shape, bfloat16));
                cl.v_bfloat16 = new array(zeros(shape, bfloat16));
            } else if (offset > cl.k_bfloat16->shape(2)) {
                const int current_capacity = cl.k_bfloat16->shape(2);
                const int new_capacity = next_cache_capacity(cl, static_cast<size_t>(offset), current_capacity);
                Shape shape = {1, fused_weights->n_kv_heads, new_capacity, fused_weights->head_dim};
                array new_k = zeros(shape, bfloat16);
                array new_v = zeros(shape, bfloat16);
                if (prev > 0) {
                    Shape copy_stop = {1, fused_weights->n_kv_heads, static_cast<int>(prev), fused_weights->head_dim};
                    new_k = slice_update(new_k, slice(*cl.k_bfloat16, g_slice_start, copy_stop), g_slice_start, copy_stop);
                    new_v = slice_update(new_v, slice(*cl.v_bfloat16, g_slice_start, copy_stop), g_slice_start, copy_stop);
                }
                *cl.k_bfloat16 = new_k;
                *cl.v_bfloat16 = new_v;
            }

            Shape update_start = {0, 0, static_cast<int>(prev), 0};
            Shape update_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            *cl.k_bfloat16 = slice_update(*cl.k_bfloat16, k, update_start, update_stop);
            *cl.v_bfloat16 = slice_update(*cl.v_bfloat16, v, update_start, update_stop);
            cl.offset = offset;

            const Shape slice_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            array k_full = slice(*cl.k_bfloat16, g_slice_start, slice_stop);
            array v_full = slice(*cl.v_bfloat16, g_slice_start, slice_stop);

            array attn_k(0.0f, float32);
            array attn_v(0.0f, float32);
            gqa_expand_attention_kv(q, k_full, v_full, &attn_k, &attn_v);
            array attn_out = fast::scaled_dot_product_attention(q, attn_k, attn_v, attn_scale, "");

            attn_out = transpose(attn_out, g_transpose_perm);
            attn_out = reshape(attn_out, attn_out_shape);

            attn_proj = quantized_matmul(attn_out, l.o_w, l.o_s, l.o_b, true, group_size, bit_width, "affine");
        }

        // Apply post-attention norm to attn output before residual (if architecture uses it)
        if (fused_weights->has_norm_weight_offset) {
            attn_proj = fast::rms_norm(attn_proj, l.ln2_w, fused_weights->rms_eps);
        }
        // Scale layer output by residual_multiplier (NOT the residual input)
        array hidden_1 = (fused_weights->residual_multiplier != 1.0f)
            ? hidden + attn_proj * fused_weights->residual_multiplier
            : hidden + attn_proj;

        // FFN normalization: use pre_ffn_norm if present, otherwise use ln2
        array normed_2 = (fused_weights->has_norm_weight_offset && l.pre_ffn_norm)
            ? fast::rms_norm(hidden_1, *l.pre_ffn_norm, fused_weights->rms_eps)
            : fast::rms_norm(hidden_1, l.ln2_w, fused_weights->rms_eps);

        array down(0.0f, float32);
        if (l.use_moe) {
            auto* moe_out = static_cast<const array*>(mlx_lazy_fused_moe_ffn_mxfp4(
                &normed_2,
                &l.moe_router_w,
                l.moe_router_s ? &*l.moe_router_s : nullptr,
                l.moe_router_b ? &*l.moe_router_b : nullptr,
                l.moe_router_bias ? &*l.moe_router_bias : nullptr,
                &l.moe_gate_w,
                &l.moe_gate_s,
                &l.moe_up_w,
                &l.moe_up_s,
                &l.moe_down_w,
                &l.moe_down_s,
                l.moe_gate_bias ? &*l.moe_gate_bias : nullptr,
                l.moe_up_bias ? &*l.moe_up_bias : nullptr,
                l.moe_down_bias ? &*l.moe_down_bias : nullptr,
                static_cast<size_t>(l.moe_num_experts),
                static_cast<size_t>(l.moe_experts_per_token),
                static_cast<size_t>(l.moe_router_group_size),
                static_cast<size_t>(l.moe_expert_group_size)));
            down = *moe_out;
        } else {
            array gate(0.0f), up(0.0f);
            if (l.use_fused_gate_up && l.gate_up_w) {
                // Fused gate/up: single matmul then split
                array gate_up = quantized_matmul(normed_2, *l.gate_up_w, *l.gate_up_s, *l.gate_up_b, true, group_size, bit_width, "affine");
                int d_ff = l.gate_w.shape(0);  // FFN intermediate size
                gate = slice(gate_up, {0, 0, 0}, {1, 1, d_ff});
                up = slice(gate_up, {0, 0, d_ff}, {1, 1, 2 * d_ff});
            } else {
                gate = quantized_matmul(normed_2, l.gate_w, l.gate_s, l.gate_b, true, group_size, bit_width, "affine");
                up = quantized_matmul(normed_2, l.up_w, l.up_s, l.up_b, true, group_size, bit_width, "affine");
            }

            // Activation: GELU or SiLU based on model config
            array mid = [&]() -> array {
                if (fused_weights->use_gelu) {
                    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    const float sqrt_2_over_pi = 0.7978845608f;
                    array x3 = gate * gate * gate;
                    array inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
                    return gate * 0.5f * (1.0f + tanh(inner)) * up;
                } else {
                    return (gate * sigmoid(gate)) * up;
                }
            }();
            down = quantized_matmul(mid, l.down_w, l.down_s, l.down_b, true, group_size, bit_width, "affine");
        }

        // Apply post-FFN norm if present
        if (l.post_ffn_norm) {
            down = fast::rms_norm(down, *l.post_ffn_norm, fused_weights->rms_eps);
        }

        // Scale layer output by residual_multiplier (NOT the residual input)
        hidden = (fused_weights->residual_multiplier != 1.0f)
            ? hidden_1 + down * fused_weights->residual_multiplier
            : hidden_1 + down;
    }

    array final_normed = fast::rms_norm(hidden, fused_weights->ln_final, fused_weights->rms_eps);
    array logits = quantized_matmul(final_normed, fused_weights->lm_head_w, fused_weights->lm_head_s, fused_weights->lm_head_b, true, group_size, bit_width, "affine");

    // Scale logits by logits_scaling (divide, not multiply!)
    if (fused_weights->logits_scaling != 1.0f) {
        logits = logits / fused_weights->logits_scaling;
    }

    return reshape(logits, {-1});
}

static array fused_forward_from_token(
    FusedModelWeights* fused_weights,
    MLXCache* cache_state,
    MLXShortConvCache* shortconv_cache_state,
    MLXMambaCache* mamba_cache_state,
    const array& token_idx,
    size_t pos_offset
) {
    array logits = fused_forward_logits_from_token(
        fused_weights,
        cache_state,
        shortconv_cache_state,
        mamba_cache_state,
        token_idx,
        pos_offset
    );
    return argmax(logits, 0);
}

// ============================================================================
// C API - Model Lifecycle
// ============================================================================

extern "C" {

void* mlx_fused_model_create(
    size_t n_layers,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t hidden_dim,
    size_t group_size, size_t bits, float rope_theta, float rms_eps
) {
    auto* fused_weights = new FusedModelWeights();
    fused_weights->layers.resize(n_layers);
    fused_weights->n_heads = static_cast<int>(n_heads);
    fused_weights->n_kv_heads = static_cast<int>(n_kv_heads);
    fused_weights->head_dim = static_cast<int>(head_dim);
    fused_weights->hidden_dim = static_cast<int>(hidden_dim);
    fused_weights->group_size = static_cast<int>(group_size);
    fused_weights->bits = static_cast<int>(bits);
    fused_weights->rope_theta = rope_theta;
    fused_weights->rms_eps = rms_eps;
    g_fused_weights = fused_weights;
    return fused_weights;
}

void mlx_fused_model_set_embeddings(void* model, const void* w, const void* s, const void* b) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    fused_weights->embed_w = *static_cast<const array*>(w);
    fused_weights->embed_s = *static_cast<const array*>(s);
    fused_weights->embed_b = *static_cast<const array*>(b);
}

void mlx_fused_model_set_final(void* model, const void* ln_w, const void* lm_w, const void* lm_s, const void* lm_b) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    fused_weights->ln_final = *static_cast<const array*>(ln_w);
    fused_weights->lm_head_w = *static_cast<const array*>(lm_w);
    fused_weights->lm_head_s = *static_cast<const array*>(lm_s);
    fused_weights->lm_head_b = *static_cast<const array*>(lm_b);
}

void mlx_fused_model_set_rope_freqs(void* model, const void* freqs) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    fused_weights->rope_freqs = *static_cast<const array*>(freqs);
}

void mlx_fused_model_set_arch_config(void* model, bool has_norm_weight_offset, bool use_gelu, float query_pre_attn_scalar) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    fused_weights->has_norm_weight_offset = has_norm_weight_offset;
    fused_weights->use_gelu = use_gelu;
    fused_weights->query_pre_attn_scalar = query_pre_attn_scalar;
}

void mlx_fused_model_set_scaling_config(
    void* model,
    float embedding_multiplier,
    float attention_multiplier,
    float residual_multiplier,
    float logits_scaling
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    fused_weights->embedding_multiplier = embedding_multiplier;
    fused_weights->attention_multiplier = attention_multiplier;
    fused_weights->residual_multiplier = residual_multiplier;
    fused_weights->logits_scaling = logits_scaling;
}

void mlx_fused_model_set_topology(
    void* model,
    const uint8_t* layer_kinds,
    size_t n_layer_kinds
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    if (n_layer_kinds != fused_weights->layers.size()) {
        throw std::invalid_argument("fused topology length does not match layer count");
    }
    for (size_t idx = 0; idx < n_layer_kinds; idx++) {
        fused_weights->layers[idx].kind = decode_layer_kind(layer_kinds[idx]);
    }
    fused_weights->topology_initialized = true;
}

void mlx_fused_model_set_layer(
    void* model, size_t layer_idx,
    const void* ln1_w,
    const void* q_w, const void* q_s, const void* q_b,
    const void* k_w, const void* k_s, const void* k_b,
    const void* v_w, const void* v_s, const void* v_b,
    const void* o_w, const void* o_s, const void* o_b,
    const void* ln2_w,
    const void* gate_w, const void* gate_s, const void* gate_b,
    const void* up_w, const void* up_s, const void* up_b,
    const void* down_w, const void* down_s, const void* down_b,
    const void* q_norm, const void* k_norm,
    const void* pre_ffn_norm, const void* post_ffn_norm,
    size_t shortconv_d_conv,
    size_t shortconv_conv_dim,
    const void* shortconv_in_w, const void* shortconv_in_s, const void* shortconv_in_b,
    const void* shortconv_out_w, const void* shortconv_out_s, const void* shortconv_out_b,
    const void* shortconv_conv_w, const void* shortconv_conv_b,
    const void* moe_router_w, const void* moe_router_s, const void* moe_router_b, const void* moe_router_bias,
    const void* moe_gate_w, const void* moe_gate_s,
    const void* moe_up_w, const void* moe_up_s,
    const void* moe_down_w, const void* moe_down_s,
    const void* moe_gate_bias, const void* moe_up_bias, const void* moe_down_bias,
    size_t moe_num_experts, size_t moe_experts_per_token,
    size_t moe_router_group_size, size_t moe_expert_group_size
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    if (!fused_weights->topology_initialized) {
        throw std::invalid_argument("mlx_fused_model_set_topology must be called before mlx_fused_model_set_layer");
    }
    auto& layer = fused_weights->layers[layer_idx];
    layer.ln1_w = *static_cast<const array*>(ln1_w);
    if (q_w) layer.q_w = *static_cast<const array*>(q_w);
    if (q_s) layer.q_s = *static_cast<const array*>(q_s);
    if (q_b) layer.q_b = *static_cast<const array*>(q_b);
    if (k_w) layer.k_w = *static_cast<const array*>(k_w);
    if (k_s) layer.k_s = *static_cast<const array*>(k_s);
    if (k_b) layer.k_b = *static_cast<const array*>(k_b);
    if (v_w) layer.v_w = *static_cast<const array*>(v_w);
    if (v_s) layer.v_s = *static_cast<const array*>(v_s);
    if (v_b) layer.v_b = *static_cast<const array*>(v_b);
    if (o_w) layer.o_w = *static_cast<const array*>(o_w);
    if (o_s) layer.o_s = *static_cast<const array*>(o_s);
    if (o_b) layer.o_b = *static_cast<const array*>(o_b);
    layer.ln2_w = *static_cast<const array*>(ln2_w);
    if (gate_w) layer.gate_w = *static_cast<const array*>(gate_w);
    if (gate_s) layer.gate_s = *static_cast<const array*>(gate_s);
    if (gate_b) layer.gate_b = *static_cast<const array*>(gate_b);
    if (up_w) layer.up_w = *static_cast<const array*>(up_w);
    if (up_s) layer.up_s = *static_cast<const array*>(up_s);
    if (up_b) layer.up_b = *static_cast<const array*>(up_b);
    if (down_w) layer.down_w = *static_cast<const array*>(down_w);
    if (down_s) layer.down_s = *static_cast<const array*>(down_s);
    if (down_b) layer.down_b = *static_cast<const array*>(down_b);
    if (q_norm) layer.q_norm = *static_cast<const array*>(q_norm);
    if (k_norm) layer.k_norm = *static_cast<const array*>(k_norm);
    if (pre_ffn_norm) layer.pre_ffn_norm = *static_cast<const array*>(pre_ffn_norm);
    if (post_ffn_norm) layer.post_ffn_norm = *static_cast<const array*>(post_ffn_norm);

    if (moe_router_w) {
        layer.use_moe = true;
        layer.moe_router_w = *static_cast<const array*>(moe_router_w);
        layer.moe_router_s = moe_router_s ? std::make_optional(*static_cast<const array*>(moe_router_s)) : std::nullopt;
        layer.moe_router_b = moe_router_b ? std::make_optional(*static_cast<const array*>(moe_router_b)) : std::nullopt;
        layer.moe_router_bias = moe_router_bias ? std::make_optional(*static_cast<const array*>(moe_router_bias)) : std::nullopt;
        layer.moe_gate_w = *static_cast<const array*>(moe_gate_w);
        layer.moe_gate_s = *static_cast<const array*>(moe_gate_s);
        layer.moe_up_w = *static_cast<const array*>(moe_up_w);
        layer.moe_up_s = *static_cast<const array*>(moe_up_s);
        layer.moe_down_w = *static_cast<const array*>(moe_down_w);
        layer.moe_down_s = *static_cast<const array*>(moe_down_s);
        layer.moe_gate_bias = moe_gate_bias ? std::make_optional(*static_cast<const array*>(moe_gate_bias)) : std::nullopt;
        layer.moe_up_bias = moe_up_bias ? std::make_optional(*static_cast<const array*>(moe_up_bias)) : std::nullopt;
        layer.moe_down_bias = moe_down_bias ? std::make_optional(*static_cast<const array*>(moe_down_bias)) : std::nullopt;
        layer.moe_num_experts = static_cast<int>(moe_num_experts);
        layer.moe_experts_per_token = static_cast<int>(moe_experts_per_token);
        layer.moe_router_group_size = static_cast<int>(moe_router_group_size);
        layer.moe_expert_group_size = static_cast<int>(moe_expert_group_size);
    }

    if (layer.kind == FusedModelWeights::Layer::LayerKind::shortconv) {
        layer.shortconv_d_conv = static_cast<int>(shortconv_d_conv);
        layer.shortconv_conv_dim = static_cast<int>(shortconv_conv_dim);
        layer.shortconv_in_w = *static_cast<const array*>(shortconv_in_w);
        layer.shortconv_in_s = *static_cast<const array*>(shortconv_in_s);
        layer.shortconv_in_b = *static_cast<const array*>(shortconv_in_b);
        layer.shortconv_out_w = *static_cast<const array*>(shortconv_out_w);
        layer.shortconv_out_s = *static_cast<const array*>(shortconv_out_s);
        layer.shortconv_out_b = *static_cast<const array*>(shortconv_out_b);
        layer.shortconv_conv_w = *static_cast<const array*>(shortconv_conv_w);
        if (shortconv_conv_b) {
            layer.shortconv_conv_b = *static_cast<const array*>(shortconv_conv_b);
        } else {
            layer.shortconv_conv_b = std::nullopt;
        }

        array conv_kernel = layer.shortconv_conv_w;
        if (conv_kernel.ndim() == 3) {
            conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
        }
        array conv_kernel_time_major(0.0f, float32);
        if (conv_kernel.ndim() == 2 &&
            conv_kernel.shape(0) == layer.shortconv_d_conv &&
            conv_kernel.shape(1) == layer.shortconv_conv_dim) {
            conv_kernel_time_major = astype(conv_kernel, float32);
        } else {
            conv_kernel_time_major = astype(transpose(conv_kernel), float32);
        }
        layer.shortconv_kernel_broadcast = reshape(
            conv_kernel_time_major,
            {1, layer.shortconv_d_conv, layer.shortconv_conv_dim}
        );
        if (layer.shortconv_conv_b) {
            layer.shortconv_bias_row = reshape(
                astype(*layer.shortconv_conv_b, float32),
                {1, layer.shortconv_conv_dim}
            );
        } else {
            layer.shortconv_bias_row = std::nullopt;
        }
    }
}

void mlx_fused_model_set_layer_mla_quantized(
    void* model,
    size_t layer_idx,
    size_t n_heads,
    size_t q_lora_rank,
    size_t kv_lora_rank,
    size_t qk_head_dim,
    size_t qk_rope_head_dim,
    size_t qk_nope_head_dim,
    size_t v_head_dim,
    const void* q_a_w,
    const void* q_a_s,
    const void* q_a_b,
    const void* q_b_w,
    const void* q_b_s,
    const void* q_b_b,
    const void* kv_a_w,
    const void* kv_a_s,
    const void* kv_a_b,
    const void* kv_b_w,
    const void* kv_b_s,
    const void* kv_b_b,
    const void* q_a_norm,
    const void* kv_a_norm,
    const void* o_w,
    const void* o_s,
    const void* o_b
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    if (!fused_weights->topology_initialized) {
        throw std::invalid_argument("mlx_fused_model_set_topology must be called before mlx_fused_model_set_layer_mla_quantized");
    }
    if (q_a_w == nullptr || q_a_s == nullptr || q_a_b == nullptr ||
        q_b_w == nullptr || q_b_s == nullptr || q_b_b == nullptr ||
        kv_a_w == nullptr || kv_a_s == nullptr || kv_a_b == nullptr ||
        kv_b_w == nullptr || kv_b_s == nullptr || kv_b_b == nullptr ||
        q_a_norm == nullptr || kv_a_norm == nullptr ||
        o_w == nullptr || o_s == nullptr || o_b == nullptr) {
        throw std::invalid_argument("mlx_fused_model_set_layer_mla_quantized requires non-null MLA tensors");
    }
    auto& layer = fused_weights->layers[layer_idx];
    if (layer.kind != FusedModelWeights::Layer::LayerKind::attention_mlp) {
        throw std::invalid_argument("mlx_fused_model_set_layer_mla_quantized requires attention_mlp layer kind");
    }

    layer.use_mla = true;
    layer.mla_n_heads = static_cast<int>(n_heads);
    layer.mla_q_lora_rank = static_cast<int>(q_lora_rank);
    layer.mla_kv_lora_rank = static_cast<int>(kv_lora_rank);
    layer.mla_qk_head_dim = static_cast<int>(qk_head_dim);
    layer.mla_qk_rope_head_dim = static_cast<int>(qk_rope_head_dim);
    layer.mla_qk_nope_head_dim = static_cast<int>(qk_nope_head_dim);
    layer.mla_v_head_dim = static_cast<int>(v_head_dim);

    layer.mla_q_a_w = *static_cast<const array*>(q_a_w);
    layer.mla_q_a_s = *static_cast<const array*>(q_a_s);
    layer.mla_q_a_b = *static_cast<const array*>(q_a_b);
    layer.mla_q_b_w = *static_cast<const array*>(q_b_w);
    layer.mla_q_b_s = *static_cast<const array*>(q_b_s);
    layer.mla_q_b_b = *static_cast<const array*>(q_b_b);
    layer.mla_kv_a_w = *static_cast<const array*>(kv_a_w);
    layer.mla_kv_a_s = *static_cast<const array*>(kv_a_s);
    layer.mla_kv_a_b = *static_cast<const array*>(kv_a_b);
    layer.mla_kv_b_w = *static_cast<const array*>(kv_b_w);
    layer.mla_kv_b_s = *static_cast<const array*>(kv_b_s);
    layer.mla_kv_b_b = *static_cast<const array*>(kv_b_b);
    layer.mla_q_a_norm = *static_cast<const array*>(q_a_norm);
    layer.mla_kv_a_norm = *static_cast<const array*>(kv_a_norm);
    layer.mla_o_w = *static_cast<const array*>(o_w);
    layer.mla_o_s = *static_cast<const array*>(o_s);
    layer.mla_o_b = *static_cast<const array*>(o_b);
}

void mlx_fused_model_set_layer_mamba_quantized(
    void* model,
    size_t layer_idx,
    size_t d_state,
    size_t d_conv,
    size_t n_heads,
    size_t d_head,
    size_t n_groups,
    uint8_t gate_up_layout,
    const void* ln1_w,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    const void* ln2_w,
    const void* gate_up_w,
    const void* gate_up_s,
    const void* gate_up_b,
    const void* down_w,
    const void* down_s,
    const void* down_b
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    if (!fused_weights->topology_initialized) {
        throw std::invalid_argument("mlx_fused_model_set_topology must be called before mlx_fused_model_set_layer_mamba_quantized");
    }
    if (ln1_w == nullptr || conv_weight == nullptr || a_log == nullptr || d_skip == nullptr ||
        in_w == nullptr || in_s == nullptr || in_b == nullptr ||
        out_w == nullptr || out_s == nullptr || out_b == nullptr ||
        ln2_w == nullptr) {
        throw std::invalid_argument("mlx_fused_model_set_layer_mamba_quantized requires non-null Mamba core tensors");
    }

    auto& layer = fused_weights->layers[layer_idx];
    if (layer.kind != FusedModelWeights::Layer::LayerKind::mamba) {
        throw std::invalid_argument("mlx_fused_model_set_layer_mamba_quantized requires mamba layer kind");
    }

    layer.ln1_w = *static_cast<const array*>(ln1_w);
    layer.mamba_d_state = static_cast<int>(d_state);
    layer.mamba_d_conv = static_cast<int>(d_conv);
    layer.mamba_n_heads = static_cast<int>(n_heads);
    layer.mamba_d_head = static_cast<int>(d_head);
    layer.mamba_n_groups = static_cast<int>(n_groups);
    layer.mamba_gate_up_layout = gate_up_layout;
    layer.mamba_conv_w = *static_cast<const array*>(conv_weight);
    layer.mamba_conv_b = conv_bias ? std::make_optional(*static_cast<const array*>(conv_bias)) : std::nullopt;
    layer.mamba_a_log = *static_cast<const array*>(a_log);
    layer.mamba_d_skip = *static_cast<const array*>(d_skip);
    layer.mamba_dt_bias = dt_bias ? std::make_optional(*static_cast<const array*>(dt_bias)) : std::nullopt;
    layer.mamba_norm_weight = norm_weight ? std::make_optional(*static_cast<const array*>(norm_weight)) : std::nullopt;
    layer.mamba_in_w = *static_cast<const array*>(in_w);
    layer.mamba_in_s = *static_cast<const array*>(in_s);
    layer.mamba_in_b = *static_cast<const array*>(in_b);
    layer.mamba_out_w = *static_cast<const array*>(out_w);
    layer.mamba_out_s = *static_cast<const array*>(out_s);
    layer.mamba_out_b = *static_cast<const array*>(out_b);
    layer.ln2_w = *static_cast<const array*>(ln2_w);

    const bool has_quantized_ffn = gate_up_w != nullptr || gate_up_s != nullptr || gate_up_b != nullptr ||
        down_w != nullptr || down_s != nullptr || down_b != nullptr;
    if (has_quantized_ffn) {
        if (gate_up_w == nullptr || gate_up_s == nullptr || gate_up_b == nullptr ||
            down_w == nullptr || down_s == nullptr || down_b == nullptr) {
            throw std::invalid_argument("mlx_fused_model_set_layer_mamba_quantized requires complete quantized FFN tensors");
        }
        layer.mamba_gate_up_w = *static_cast<const array*>(gate_up_w);
        layer.mamba_gate_up_s = *static_cast<const array*>(gate_up_s);
        layer.mamba_gate_up_b = *static_cast<const array*>(gate_up_b);
        layer.mamba_down_w = *static_cast<const array*>(down_w);
        layer.mamba_down_s = *static_cast<const array*>(down_s);
        layer.mamba_down_b = *static_cast<const array*>(down_b);
    } else {
        layer.mamba_gate_up_w = std::nullopt;
        layer.mamba_gate_up_s = std::nullopt;
        layer.mamba_gate_up_b = std::nullopt;
        layer.mamba_down_w = std::nullopt;
        layer.mamba_down_s = std::nullopt;
        layer.mamba_down_b = std::nullopt;
    }
}

void mlx_fused_model_free(void* model) {
    delete static_cast<FusedModelWeights*>(model);
    if (g_fused_weights == model) g_fused_weights = nullptr;
}

// Fuse weights for faster inference - call after all layers are set
// NOTE: Disabled - weight fusion slows down single-token decode
void mlx_fused_model_optimize(void* model) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);

    // Pre-evaluate all weights to ensure they're transferred to GPU
    // This eliminates lazy transfer overhead during first inference
    std::vector<array> to_eval;
    to_eval.push_back(fused_weights->embed_w);
    to_eval.push_back(fused_weights->embed_s);
    to_eval.push_back(fused_weights->embed_b);
    to_eval.push_back(fused_weights->ln_final);
    to_eval.push_back(fused_weights->lm_head_w);
    to_eval.push_back(fused_weights->lm_head_s);
    to_eval.push_back(fused_weights->lm_head_b);

    for (auto& layer : fused_weights->layers) {
        to_eval.push_back(layer.ln1_w);
        if (layer.kind == FusedModelWeights::Layer::LayerKind::shortconv) {
            to_eval.push_back(layer.shortconv_in_w);
            to_eval.push_back(layer.shortconv_in_s);
            to_eval.push_back(layer.shortconv_in_b);
            to_eval.push_back(layer.shortconv_out_w);
            to_eval.push_back(layer.shortconv_out_s);
            to_eval.push_back(layer.shortconv_out_b);
            to_eval.push_back(layer.shortconv_conv_w);
            if (layer.shortconv_conv_b) to_eval.push_back(*layer.shortconv_conv_b);
            if (layer.shortconv_kernel_broadcast) to_eval.push_back(*layer.shortconv_kernel_broadcast);
            if (layer.shortconv_bias_row) to_eval.push_back(*layer.shortconv_bias_row);
        } else if (layer.kind == FusedModelWeights::Layer::LayerKind::mamba) {
            to_eval.push_back(layer.mamba_conv_w);
            if (layer.mamba_conv_b) to_eval.push_back(*layer.mamba_conv_b);
            to_eval.push_back(layer.mamba_a_log);
            to_eval.push_back(layer.mamba_d_skip);
            if (layer.mamba_dt_bias) to_eval.push_back(*layer.mamba_dt_bias);
            if (layer.mamba_norm_weight) to_eval.push_back(*layer.mamba_norm_weight);
            to_eval.push_back(layer.mamba_in_w);
            to_eval.push_back(layer.mamba_in_s);
            to_eval.push_back(layer.mamba_in_b);
            to_eval.push_back(layer.mamba_out_w);
            to_eval.push_back(layer.mamba_out_s);
            to_eval.push_back(layer.mamba_out_b);
            if (layer.mamba_gate_up_w) to_eval.push_back(*layer.mamba_gate_up_w);
            if (layer.mamba_gate_up_s) to_eval.push_back(*layer.mamba_gate_up_s);
            if (layer.mamba_gate_up_b) to_eval.push_back(*layer.mamba_gate_up_b);
            if (layer.mamba_down_w) to_eval.push_back(*layer.mamba_down_w);
            if (layer.mamba_down_s) to_eval.push_back(*layer.mamba_down_s);
            if (layer.mamba_down_b) to_eval.push_back(*layer.mamba_down_b);
        } else if (layer.use_mla) {
            to_eval.push_back(layer.mla_q_a_w);
            to_eval.push_back(layer.mla_q_a_s);
            to_eval.push_back(layer.mla_q_a_b);
            to_eval.push_back(layer.mla_q_b_w);
            to_eval.push_back(layer.mla_q_b_s);
            to_eval.push_back(layer.mla_q_b_b);
            to_eval.push_back(layer.mla_kv_a_w);
            to_eval.push_back(layer.mla_kv_a_s);
            to_eval.push_back(layer.mla_kv_a_b);
            to_eval.push_back(layer.mla_kv_b_w);
            to_eval.push_back(layer.mla_kv_b_s);
            to_eval.push_back(layer.mla_kv_b_b);
            to_eval.push_back(layer.mla_q_a_norm);
            to_eval.push_back(layer.mla_kv_a_norm);
            to_eval.push_back(layer.mla_o_w);
            to_eval.push_back(layer.mla_o_s);
            to_eval.push_back(layer.mla_o_b);
        } else {
            to_eval.push_back(layer.q_w);
            to_eval.push_back(layer.q_s);
            to_eval.push_back(layer.q_b);
            to_eval.push_back(layer.k_w);
            to_eval.push_back(layer.k_s);
            to_eval.push_back(layer.k_b);
            to_eval.push_back(layer.v_w);
            to_eval.push_back(layer.v_s);
            to_eval.push_back(layer.v_b);
            to_eval.push_back(layer.o_w);
            to_eval.push_back(layer.o_s);
            to_eval.push_back(layer.o_b);
        }
        to_eval.push_back(layer.ln2_w);
        if (layer.use_moe) {
            to_eval.push_back(layer.moe_router_w);
            if (layer.moe_router_s) to_eval.push_back(*layer.moe_router_s);
            if (layer.moe_router_b) to_eval.push_back(*layer.moe_router_b);
            if (layer.moe_router_bias) to_eval.push_back(*layer.moe_router_bias);
            to_eval.push_back(layer.moe_gate_w);
            to_eval.push_back(layer.moe_gate_s);
            to_eval.push_back(layer.moe_up_w);
            to_eval.push_back(layer.moe_up_s);
            to_eval.push_back(layer.moe_down_w);
            to_eval.push_back(layer.moe_down_s);
            if (layer.moe_gate_bias) to_eval.push_back(*layer.moe_gate_bias);
            if (layer.moe_up_bias) to_eval.push_back(*layer.moe_up_bias);
            if (layer.moe_down_bias) to_eval.push_back(*layer.moe_down_bias);
        } else {
            to_eval.push_back(layer.gate_w);
            to_eval.push_back(layer.gate_s);
            to_eval.push_back(layer.gate_b);
            to_eval.push_back(layer.up_w);
            to_eval.push_back(layer.up_s);
            to_eval.push_back(layer.up_b);
            to_eval.push_back(layer.down_w);
            to_eval.push_back(layer.down_s);
            to_eval.push_back(layer.down_b);
        }
    }
    eval(to_eval);
}

// Compile hook retained for ABI compatibility.
// Decode executes through fused_forward_from_token() / fused_forward_logits_from_token().
// Do not keep dead alternative execution paths here.
void mlx_fused_model_compile(void* model) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    (void)fused_weights;
}

// ============================================================================
// C API - Synchronous Decode
// ============================================================================

uint32_t mlx_fused_decode_step(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, mamba_cache_state, token, pos_offset);
    eval(next);
    return static_cast<uint32_t>(next.item<int32_t>());
}

void* mlx_fused_decode_step_logits(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array logits = fused_forward_logits_from_token(
        fused_weights,
        cache_state,
        shortconv_cache_state,
        mamba_cache_state,
        token,
        pos_offset
    );
    return pool_array(logits);
}

// ============================================================================
// C API - Pipelined Decode
// ============================================================================

void mlx_pipeline_prime(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    uint32_t first_token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(first_token_id);
    array first_token = array(&token_id_i32, {1}, int32);

    array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, mamba_cache_state, first_token, pos_offset);
    async_eval(next);

    void* key = quant_decode_state_key(cache_ptr, model);
    std::lock_guard<std::mutex> lock(g_quant_pipeline_mu);
    auto& state = g_quant_pipeline_states[key];
    state.current_token = std::move(next);
}

uint32_t mlx_pipeline_step(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    array current(0.0f, float32);
    {
        void* key = quant_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_quant_pipeline_mu);
        auto it = g_quant_pipeline_states.find(key);
        if (it == g_quant_pipeline_states.end() || !it->second.current_token) return 0;
        current = *it->second.current_token;
    }

    // Build next graph using current (lazy) token.
    array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, mamba_cache_state, current, pos_offset);

    // Queue next.
    async_eval(next);

    // Materialize current.
    eval(current);
    const uint32_t result = static_cast<uint32_t>(current.item<int32_t>());

    // Rotate cached state.
    {
        void* key = quant_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_quant_pipeline_mu);
        auto& state = g_quant_pipeline_states[key];
        state.current_token = std::move(next);
    }

    return result;
}

uint32_t mlx_pipeline_flush(void* model, void* cache_ptr, void* shortconv_cache_ptr, void* mamba_cache_ptr) {
    (void)shortconv_cache_ptr;
    (void)mamba_cache_ptr;
    std::optional<array> current_token;
    {
        void* key = quant_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_quant_pipeline_mu);
        auto it = g_quant_pipeline_states.find(key);
        if (it == g_quant_pipeline_states.end() || !it->second.current_token) return 0;
        current_token = std::move(it->second.current_token);
        g_quant_pipeline_states.erase(it);
    }
    eval(*current_token);
    return static_cast<uint32_t>((*current_token).item<int32_t>());
}

// ============================================================================
// C API - Async Decode (async path)
// ============================================================================

void* mlx_fused_decode_async_start(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    g_pending_token = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, mamba_cache_state, token, pos_offset);
    async_eval(*g_pending_token);

    return nullptr;
}

uint32_t mlx_fused_decode_async_get() {
    if (!g_pending_token) return 0;
    return static_cast<uint32_t>(g_pending_token->item<int32_t>());
}

// ============================================================================
// C API - Batch Decode (runs entire generation loop in C++)
// ============================================================================

uint32_t mlx_fused_decode_batch(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    void* mamba_cache_ptr,
    uint32_t first_token,
    size_t start_pos,
    uint32_t* out_tokens,
    size_t max_tokens,
    const uint32_t* eos_ids,
    size_t n_eos_ids
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    auto is_eos = [&](uint32_t tok) {
        for (size_t i = 0; i < n_eos_ids; i++) {
            if (tok == eos_ids[i]) return true;
        }
        return false;
    };

    uint32_t current_token = first_token;
    size_t pos = start_pos;
    size_t gen_count = 0;

    while (gen_count < max_tokens) {
        int32_t token_id_i32 = static_cast<int32_t>(current_token);
        array token = array(&token_id_i32, {1}, int32);
        array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, mamba_cache_state, token, pos);
        eval(next);
        current_token = static_cast<uint32_t>(next.item<int32_t>());
        out_tokens[gen_count++] = current_token;
        pos++;
        if (is_eos(current_token)) break;
    }

    return gen_count;
}

} // extern "C"
