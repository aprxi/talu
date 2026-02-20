// MLX Bridge - Quantized (4-bit) Model
//
// Full transformer implementation using 4-bit quantized weights.
// Uses MLX's quantized_matmul for 8x memory bandwidth savings.

#include "compute_common.h"
#include "model_state.h"

// ============================================================================
// Quantized Model Structure
// ============================================================================

struct FusedModelWeights {
    struct Layer {
        enum class LayerKind : int {
            attention_mlp = 0,
            shortconv = 1,
        };
        LayerKind kind = LayerKind::attention_mlp;

        array ln1_w = array(0.0f, float32);  // attention norm
        array q_w = array(0.0f, float32), q_s = array(0.0f, float32), q_b = array(0.0f, float32);
        array k_w = array(0.0f, float32), k_s = array(0.0f, float32), k_b = array(0.0f, float32);
        array v_w = array(0.0f, float32), v_s = array(0.0f, float32), v_b = array(0.0f, float32);
        array o_w = array(0.0f, float32), o_s = array(0.0f, float32), o_b = array(0.0f, float32);
        // ShortConv mixer projections (quantized) and convolution weights.
        int shortconv_d_conv = 0;
        int shortconv_conv_dim = 0;
        array shortconv_in_w = array(0.0f, float32), shortconv_in_s = array(0.0f, float32), shortconv_in_b = array(0.0f, float32);
        array shortconv_out_w = array(0.0f, float32), shortconv_out_s = array(0.0f, float32), shortconv_out_b = array(0.0f, float32);
        array shortconv_conv_w = array(0.0f, float32);
        std::optional<array> shortconv_conv_b;
        std::optional<array> shortconv_kernel_broadcast;
        std::optional<array> shortconv_bias_row;
        array ln2_w = array(0.0f, float32);  // ffn norm (or post_attention_layernorm)
        array gate_w = array(0.0f, float32), gate_s = array(0.0f, float32), gate_b = array(0.0f, float32);
        array up_w = array(0.0f, float32), up_s = array(0.0f, float32), up_b = array(0.0f, float32);
        array down_w = array(0.0f, float32), down_s = array(0.0f, float32), down_b = array(0.0f, float32);
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

    // Compiled forward function for decode (eliminates graph rebuild overhead)
    std::function<std::vector<array>(const std::vector<array>&)> compiled_decode;
    bool is_compiled = false;
    bool topology_initialized = false;
};

static FusedModelWeights* g_fused_weights = nullptr;

static constexpr uint8_t kLayerKindAttentionMlp = 0;
static constexpr uint8_t kLayerKindShortConv = 1;

static FusedModelWeights::Layer::LayerKind decode_layer_kind(uint8_t kind_id) {
    switch (kind_id) {
        case kLayerKindAttentionMlp:
            return FusedModelWeights::Layer::LayerKind::attention_mlp;
        case kLayerKindShortConv:
            return FusedModelWeights::Layer::LayerKind::shortconv;
        default:
            throw std::invalid_argument("Unsupported fused layer kind id");
    }
}

// Pipeline state
static thread_local std::optional<array> g_current_token;
static thread_local std::optional<array> g_pending_token;

// ============================================================================
// Forward Pass Implementation
// ============================================================================

static array fused_forward_logits_from_token(
    FusedModelWeights* fused_weights,
    MLXCache* cache_state,
    MLXShortConvCache* shortconv_cache_state,
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

            if (cl.k_bfloat16 == nullptr || offset > cl.k_bfloat16->shape(2)) {
                int new_size = ((offset + cl.step - 1) / cl.step) * cl.step;
                Shape shape = {1, fused_weights->n_kv_heads, new_size, fused_weights->head_dim};
                if (cl.k_bfloat16) {
                    array new_k = zeros(shape, bfloat16);
                    array new_v = zeros(shape, bfloat16);
                    Shape stop = {1, fused_weights->n_kv_heads, static_cast<int>(prev), fused_weights->head_dim};
                    new_k = slice_update(new_k, slice(*cl.k_bfloat16, g_slice_start, stop), g_slice_start, stop);
                    new_v = slice_update(new_v, slice(*cl.v_bfloat16, g_slice_start, stop), g_slice_start, stop);
                    *cl.k_bfloat16 = new_k;
                    *cl.v_bfloat16 = new_v;
                } else {
                    cl.k_bfloat16 = new array(zeros(shape, bfloat16));
                    cl.v_bfloat16 = new array(zeros(shape, bfloat16));
                }
            }

            Shape update_start = {0, 0, static_cast<int>(prev), 0};
            Shape update_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            *cl.k_bfloat16 = slice_update(*cl.k_bfloat16, k, update_start, update_stop);
            *cl.v_bfloat16 = slice_update(*cl.v_bfloat16, v, update_start, update_stop);
            cl.offset = offset;

            const Shape slice_stop = {1, fused_weights->n_kv_heads, offset, fused_weights->head_dim};
            array k_full = slice(*cl.k_bfloat16, g_slice_start, slice_stop);
            array v_full = slice(*cl.v_bfloat16, g_slice_start, slice_stop);

            array attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, attn_scale, "");

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
        array down = quantized_matmul(mid, l.down_w, l.down_s, l.down_b, true, group_size, bit_width, "affine");

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
    const array& token_idx,
    size_t pos_offset
) {
    array logits = fused_forward_logits_from_token(
        fused_weights,
        cache_state,
        shortconv_cache_state,
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
    const void* shortconv_conv_w, const void* shortconv_conv_b
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
    layer.gate_w = *static_cast<const array*>(gate_w);
    layer.gate_s = *static_cast<const array*>(gate_s);
    layer.gate_b = *static_cast<const array*>(gate_b);
    layer.up_w = *static_cast<const array*>(up_w);
    layer.up_s = *static_cast<const array*>(up_s);
    layer.up_b = *static_cast<const array*>(up_b);
    layer.down_w = *static_cast<const array*>(down_w);
    layer.down_s = *static_cast<const array*>(down_s);
    layer.down_b = *static_cast<const array*>(down_b);
    if (q_norm) layer.q_norm = *static_cast<const array*>(q_norm);
    if (k_norm) layer.k_norm = *static_cast<const array*>(k_norm);
    if (pre_ffn_norm) layer.pre_ffn_norm = *static_cast<const array*>(pre_ffn_norm);
    if (post_ffn_norm) layer.post_ffn_norm = *static_cast<const array*>(post_ffn_norm);

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
    eval(to_eval);
}

// Global compiled step function (set once, reused for all decode steps)
static std::function<std::vector<array>(const std::vector<array>&)> g_compiled_step;

// Compile the decode step for maximum performance
// This traces the forward pass once and reuses the compiled graph
void mlx_fused_model_compile(void* model) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    if (fused_weights->is_compiled) return;

    // Compiled-step path only models attention cache updates today.
    // Keep shortconv models on the direct fused_forward_from_token path, which
    // correctly updates both KV cache and shortconv conv-state cache.
    for (const auto& layer : fused_weights->layers) {
        if (layer.kind == FusedModelWeights::Layer::LayerKind::shortconv) {
            return;
        }
    }

    const int group_size = fused_weights->group_size;
    const int bits = fused_weights->bits;
    const int n_layers = static_cast<int>(fused_weights->layers.size());

    // Create step function that takes: [hidden, k_caches..., v_caches..., pos]
    // Returns: [next_token, new_k_caches..., new_v_caches...]
    auto step_fn = [fused_weights, group_size, bits, n_layers](const std::vector<array>& inputs) -> std::vector<array> {
        // inputs[0] = hidden state [1, 1, hidden_dim]
        // inputs[1..n_layers] = k_cache for each layer
        // inputs[n_layers+1..2*n_layers] = v_cache for each layer
        // inputs[2*n_layers+1] = position offset scalar

        array hidden = inputs[0];
        int pos_idx = static_cast<int>(inputs[2 * n_layers + 1].item<int32_t>());

        const float attn_scale = (fused_weights->attention_multiplier > 0.0f)
            ? fused_weights->attention_multiplier
            : (fused_weights->query_pre_attn_scalar > 0.0f)
                ? 1.0f / std::sqrt(fused_weights->query_pre_attn_scalar)
                : 1.0f / std::sqrt(static_cast<float>(fused_weights->head_dim));

        std::vector<array> new_k_caches, new_v_caches;

        for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
            const auto& l = fused_weights->layers[layer_idx];
            array k_cache = inputs[1 + layer_idx];
            array v_cache = inputs[1 + n_layers + layer_idx];

            array normed = fast::rms_norm(hidden, l.ln1_w, fused_weights->rms_eps);

            array q = quantized_matmul(normed, l.q_w, l.q_s, l.q_b, true, group_size, bits, "affine");
            array k = quantized_matmul(normed, l.k_w, l.k_s, l.k_b, true, group_size, bits, "affine");
            array v = quantized_matmul(normed, l.v_w, l.v_s, l.v_b, true, group_size, bits, "affine");

            // Reshape for attention
            q = reshape(q, {1, 1, fused_weights->n_heads, fused_weights->head_dim});
            k = reshape(k, {1, 1, fused_weights->n_kv_heads, fused_weights->head_dim});
            v = reshape(v, {1, 1, fused_weights->n_kv_heads, fused_weights->head_dim});

            if (l.q_norm) q = fast::rms_norm(q, *l.q_norm, fused_weights->rms_eps);
            if (l.k_norm) k = fast::rms_norm(k, *l.k_norm, fused_weights->rms_eps);

            q = transpose(q, {0, 2, 1, 3});
            k = transpose(k, {0, 2, 1, 3});
            v = transpose(v, {0, 2, 1, 3});

            // RoPE
            if (fused_weights->rope_freqs) {
                q = fast::rope(q, fused_weights->head_dim, false, std::nullopt, 1.0f, pos_idx, fused_weights->rope_freqs);
                k = fast::rope(k, fused_weights->head_dim, false, std::nullopt, 1.0f, pos_idx, fused_weights->rope_freqs);
            } else {
                q = fast::rope(q, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, pos_idx);
                k = fast::rope(k, fused_weights->head_dim, false, fused_weights->rope_theta, 1.0f, pos_idx);
            }

            // Update cache (concatenate)
            array new_k = concatenate({k_cache, k}, 2);
            array new_v = concatenate({v_cache, v}, 2);
            new_k_caches.push_back(new_k);
            new_v_caches.push_back(new_v);

            // Attention
            array attn_out = fast::scaled_dot_product_attention(q, new_k, new_v, attn_scale, "");
            attn_out = transpose(attn_out, {0, 2, 1, 3});
            attn_out = reshape(attn_out, {1, 1, fused_weights->n_heads * fused_weights->head_dim});

            array attn_proj = quantized_matmul(attn_out, l.o_w, l.o_s, l.o_b, true, group_size, bits, "affine");

            // Apply post-attention norm to attn output before residual (if architecture uses it)
            if (fused_weights->has_norm_weight_offset) {
                attn_proj = fast::rms_norm(attn_proj, l.ln2_w, fused_weights->rms_eps);
            }

            array hidden_1 = hidden + attn_proj;

            // FFN normalization: use pre_ffn_norm if present, otherwise use ln2
            array normed_2 = (fused_weights->has_norm_weight_offset && l.pre_ffn_norm)
                ? fast::rms_norm(hidden_1, *l.pre_ffn_norm, fused_weights->rms_eps)
                : fast::rms_norm(hidden_1, l.ln2_w, fused_weights->rms_eps);

            array gate = quantized_matmul(normed_2, l.gate_w, l.gate_s, l.gate_b, true, group_size, bits, "affine");
            array up = quantized_matmul(normed_2, l.up_w, l.up_s, l.up_b, true, group_size, bits, "affine");
            array mid = (gate * sigmoid(gate)) * up;
            array down = quantized_matmul(mid, l.down_w, l.down_s, l.down_b, true, group_size, bits, "affine");

            hidden = hidden_1 + down;
        }

        // Final norm + LM head
        array final_normed = fast::rms_norm(hidden, fused_weights->ln_final, fused_weights->rms_eps);
        array logits = quantized_matmul(final_normed, fused_weights->lm_head_w, fused_weights->lm_head_s, fused_weights->lm_head_b, true, group_size, bits, "affine");

        // Argmax
        array next_token = argmax(logits, -1);
        next_token = reshape(next_token, {1});

        // Build output: [next_token, k_caches..., v_caches...]
        std::vector<array> outputs;
        outputs.push_back(next_token);
        for (auto& k : new_k_caches) outputs.push_back(k);
        for (auto& v : new_v_caches) outputs.push_back(v);

        return outputs;
    };

    // Compile with shapeless=true to handle variable cache sizes
    g_compiled_step = compile(step_fn, /* shapeless= */ true);
    fused_weights->is_compiled = true;
}

// ============================================================================
// C API - Synchronous Decode
// ============================================================================

uint32_t mlx_fused_decode_step(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, token, pos_offset);
    eval(next);
    return static_cast<uint32_t>(next.item<int32_t>());
}

void* mlx_fused_decode_step_logits(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array logits = fused_forward_logits_from_token(
        fused_weights,
        cache_state,
        shortconv_cache_state,
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
    uint32_t first_token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(first_token_id);
    array first_token = array(&token_id_i32, {1}, int32);

    g_current_token = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, first_token, pos_offset);
    async_eval(*g_current_token);
}

uint32_t mlx_pipeline_step(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    size_t pos_offset
) {
    if (!g_current_token) return 0;

    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    // Build next graph using current (lazy) token
    array& current = *g_current_token;
    array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, current, pos_offset);

    // Queue next
    async_eval(next);

    // Materialize current
    eval(current);

    uint32_t result = static_cast<uint32_t>(*current.data<int32_t>());

    // Rotate
    g_current_token = next;

    return result;
}

uint32_t mlx_pipeline_flush() {
    if (!g_current_token) return 0;
    uint32_t result = static_cast<uint32_t>(g_current_token->item<int32_t>());
    g_current_token.reset();
    return result;
}

// ============================================================================
// C API - Async Decode (async path)
// ============================================================================

void* mlx_fused_decode_async_start(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_weights = static_cast<FusedModelWeights*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    g_pending_token = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, token, pos_offset);
    async_eval(*g_pending_token);

    return nullptr;
}

uint32_t mlx_fused_decode_async_get() {
    if (!g_pending_token) return 0;
    return static_cast<uint32_t>(g_pending_token->item<int32_t>());
}

// ============================================================================
// C API - Compiled Layer (uses MLX compile() for fusion)
// ============================================================================
// These functions compile entire transformer layers for better GPU fusion.

struct CompiledLayer {
    std::function<std::vector<array>(const std::vector<array>&)> fn;
};
static std::vector<CompiledLayer*> g_compiled_layers;

void* mlx_compile_layer(
    const void* q_weight, const void* q_scales, const void* q_biases,
    const void* k_weight, const void* k_scales, const void* k_biases,
    const void* v_weight, const void* v_scales, const void* v_biases,
    const void* o_weight, const void* o_scales, const void* o_biases,
    const void* gate_weight, const void* gate_scales, const void* gate_biases,
    const void* up_weight, const void* up_scales, const void* up_biases,
    const void* down_weight, const void* down_scales, const void* down_biases,
    const void* attn_norm, const void* ffn_norm,
    const void* q_norm, const void* k_norm,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t hidden_dim,
    size_t group_size, size_t bits, float rope_theta, float rms_eps
) {
    // Capture weights by value
    const array w_q = *static_cast<const array*>(q_weight);
    const array s_q = *static_cast<const array*>(q_scales);
    const array b_q = *static_cast<const array*>(q_biases);
    const array w_k = *static_cast<const array*>(k_weight);
    const array s_k = *static_cast<const array*>(k_scales);
    const array b_k = *static_cast<const array*>(k_biases);
    const array w_v = *static_cast<const array*>(v_weight);
    const array s_v = *static_cast<const array*>(v_scales);
    const array b_v = *static_cast<const array*>(v_biases);
    const array w_o = *static_cast<const array*>(o_weight);
    const array s_o = *static_cast<const array*>(o_scales);
    const array b_o = *static_cast<const array*>(o_biases);
    const array w_gate = *static_cast<const array*>(gate_weight);
    const array s_gate = *static_cast<const array*>(gate_scales);
    const array b_gate = *static_cast<const array*>(gate_biases);
    const array w_up = *static_cast<const array*>(up_weight);
    const array s_up = *static_cast<const array*>(up_scales);
    const array b_up = *static_cast<const array*>(up_biases);
    const array w_down = *static_cast<const array*>(down_weight);
    const array s_down = *static_cast<const array*>(down_scales);
    const array b_down = *static_cast<const array*>(down_biases);
    const array norm_attn = *static_cast<const array*>(attn_norm);
    const array norm_ffn = *static_cast<const array*>(ffn_norm);
    const std::optional<array> q_norm_arr = q_norm ? std::optional<array>(*static_cast<const array*>(q_norm)) : std::nullopt;
    const std::optional<array> k_norm_arr = k_norm ? std::optional<array>(*static_cast<const array*>(k_norm)) : std::nullopt;

    auto layer_fn = [=](const std::vector<array>& inputs) -> std::vector<array> {
        const auto& hidden = inputs[0];
        const auto& k_cache_in = inputs[1];
        const auto& v_cache_in = inputs[2];
        int pos_offset = inputs[3].shape(0) - 1;

        int batch_size = hidden.shape(0);
        int seq_len = hidden.shape(1);
        int group_size_int = static_cast<int>(group_size);
        int bits_int = static_cast<int>(bits);

        auto normed = fast::rms_norm(hidden, norm_attn, rms_eps);

        auto q_proj = quantized_matmul(normed, w_q, s_q, b_q, true, group_size_int, bits_int, "affine");
        auto k_proj = quantized_matmul(normed, w_k, s_k, b_k, true, group_size_int, bits_int, "affine");
        auto v_proj = quantized_matmul(normed, w_v, s_v, b_v, true, group_size_int, bits_int, "affine");

        auto q = reshape(q_proj, {batch_size, seq_len, static_cast<int>(n_heads), static_cast<int>(head_dim)});
        auto k = reshape(k_proj, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});
        auto v = reshape(v_proj, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});

        if (q_norm_arr) q = fast::rms_norm(q, *q_norm_arr, rms_eps);
        if (k_norm_arr) k = fast::rms_norm(k, *k_norm_arr, rms_eps);

        q = transpose(q, {0, 2, 1, 3});
        k = transpose(k, {0, 2, 1, 3});
        v = transpose(v, {0, 2, 1, 3});

        q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, pos_offset);
        k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, pos_offset);

        bool is_prefill = (k_cache_in.ndim() == 0 || k_cache_in.size() == 0);
        array k_full = is_prefill ? k : concatenate({k_cache_in, k}, 2);
        array v_full = is_prefill ? v : concatenate({v_cache_in, v}, 2);

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        auto attn_out = fast::scaled_dot_product_attention(q, k_full, v_full, scale, is_prefill ? "causal" : "");

        attn_out = transpose(attn_out, {0, 2, 1, 3});
        attn_out = reshape(attn_out, {batch_size, seq_len, static_cast<int>(n_heads * head_dim)});

        auto attn_proj = quantized_matmul(attn_out, w_o, s_o, b_o, true, group_size_int, bits_int, "affine");
        auto hidden_1 = hidden + attn_proj;

        auto normed_2 = fast::rms_norm(hidden_1, norm_ffn, rms_eps);
        auto gate = quantized_matmul(normed_2, w_gate, s_gate, b_gate, true, group_size_int, bits_int, "affine");
        auto up = quantized_matmul(normed_2, w_up, s_up, b_up, true, group_size_int, bits_int, "affine");
        auto ffn_hidden = (gate * sigmoid(gate)) * up;
        auto down = quantized_matmul(ffn_hidden, w_down, s_down, b_down, true, group_size_int, bits_int, "affine");

        auto output = hidden_1 + down;
        return {output, k, v};
    };

    auto* compiled = new CompiledLayer();
    compiled->fn = compile(layer_fn, /* shapeless= */ true);
    g_compiled_layers.push_back(compiled);
    return reinterpret_cast<void*>(g_compiled_layers.size() - 1);
}

void* mlx_layer_forward(
    void* compiled_handle,
    const void* hidden, void* cache_ptr, size_t layer_idx, size_t pos_offset
) {
    size_t compiled_idx = reinterpret_cast<size_t>(compiled_handle);
    auto& compiled_layer = g_compiled_layers[compiled_idx];
    const auto& h = *static_cast<const array*>(hidden);

    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto& layer = cache_state->layers[layer_idx];

    size_t prev_offset = layer.offset;
    bool is_prefill = (prev_offset == 0);

    array kc = array(0.0f, float32);
    array vc = array(0.0f, float32);

    if (!is_prefill && layer.k_bfloat16) {
        const auto& k_full = *layer.k_bfloat16;
        const auto& v_full = *layer.v_bfloat16;
        Shape start = {0, 0, 0, 0};
        Shape stop = {k_full.shape(0), k_full.shape(1), static_cast<int>(prev_offset), k_full.shape(3)};
        kc = slice(k_full, start, stop);
        vc = slice(v_full, start, stop);
    }

    array pos_arr = zeros({static_cast<int>(pos_offset + 1)}, float32);
    auto results = compiled_layer->fn({h, kc, vc, pos_arr});

    const auto& hidden_out = results[0];
    const auto& k_new = results[1];
    const auto& v_new = results[2];

    const int batch_size = k_new.shape(0);
    const int kv_heads = k_new.shape(1);
    const int head_dim = k_new.shape(3);
    const int step_count = k_new.shape(2);
    const int new_offset = prev_offset + step_count;

    const bool need_expand = !layer.k_bfloat16 ||
                             (prev_offset + step_count) > static_cast<size_t>(layer.k_bfloat16->shape(2));

    if (need_expand) {
        const int step = 256;
        const int n_steps = ((prev_offset + step_count + step - 1) / step) * step;
        Shape new_shape = {batch_size, kv_heads, n_steps, head_dim};

        if (layer.k_bfloat16) {
            array new_k = zeros(new_shape, bfloat16);
            array new_v = zeros(new_shape, bfloat16);
            Shape copy_start = {0, 0, 0, 0};
            Shape copy_stop = {batch_size, kv_heads, static_cast<int>(prev_offset), head_dim};
            array k_existing = slice(*layer.k_bfloat16, copy_start, copy_stop);
            array v_existing = slice(*layer.v_bfloat16, copy_start, copy_stop);
            new_k = slice_update(new_k, k_existing, copy_start, copy_stop);
            new_v = slice_update(new_v, v_existing, copy_start, copy_stop);
            *layer.k_bfloat16 = new_k;
            *layer.v_bfloat16 = new_v;
        } else {
            layer.k_bfloat16 = new array(zeros(new_shape, bfloat16));
            layer.v_bfloat16 = new array(zeros(new_shape, bfloat16));
        }
    }

    Shape start = {0, 0, static_cast<int>(prev_offset), 0};
    Shape stop = {batch_size, kv_heads, static_cast<int>(new_offset), head_dim};
    array k_old = *layer.k_bfloat16;
    array v_old = *layer.v_bfloat16;
    *layer.k_bfloat16 = slice_update(k_old, k_new, start, stop);
    *layer.v_bfloat16 = slice_update(v_old, v_new, start, stop);
    layer.offset = new_offset;

    return pool_array(array(hidden_out));
}

// ============================================================================
// C API - Batch Decode (runs entire generation loop in C++)
// ============================================================================

uint32_t mlx_fused_decode_batch(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
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
        array next = fused_forward_from_token(fused_weights, cache_state, shortconv_cache_state, token, pos);
        eval(next);
        current_token = static_cast<uint32_t>(next.item<int32_t>());
        out_tokens[gen_count++] = current_token;
        pos++;
        if (is_eos(current_token)) break;
    }

    return gen_count;
}

} // extern "C"
