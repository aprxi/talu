// MLX Bridge - Dense Model
//
// Full transformer implementation using dense weights.
// Optimizations:
//   - Pre-concatenated QKV and gate_up weights (single matmul instead of 3/2)
//   - Pre-transposed weights (avoid per-call transpose)
//   - Pipelined decode (async_eval overlaps with graph building)

#include "compute_common.h"
#include "model_state.h"
#include <mutex>
#include <unordered_map>

// ============================================================================
// Dense Model Structure
// ============================================================================

struct FusedDenseModel {
    struct Layer {
        enum class LayerKind : int {
            attention_mlp = 0,
            shortconv = 1,
        };
        LayerKind kind = LayerKind::attention_mlp;

        array ln1_w = array(0.0f, float32);      // attention norm
        array qkv_proj = array(0.0f, bfloat16);  // Pre-concatenated Q+K+V
        array o_proj = array(0.0f, bfloat16);
        // ShortConv mixer (dense): in-proj/out-proj are pre-transposed.
        int shortconv_d_conv = 0;
        int shortconv_conv_dim = 0;
        array shortconv_in_proj_t = array(0.0f, bfloat16);
        array shortconv_out_proj_t = array(0.0f, bfloat16);
        std::optional<array> shortconv_kernel_broadcast;
        std::optional<array> shortconv_bias_row;
        array ln2_w = array(0.0f, float32);      // ffn norm
        array gate_up_proj = array(0.0f, bfloat16);  // Pre-concatenated gate+up
        array down_proj = array(0.0f, bfloat16);
        std::optional<array> q_norm;
        std::optional<array> k_norm;
        int q_size = 0;    // For splitting QKV result
        int kv_size = 0;
    };
    std::vector<Layer> layers;

    array ln_final = array(0.0f, float32);
    array lm_head = array(0.0f, bfloat16);  // Pre-transposed
    array embed_tokens = array(0.0f, bfloat16);

    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden_dim = 0;
    float rope_theta = 0.0f;
    float rms_eps = 0.0f;
    bool topology_initialized = false;

    // Per-model decode constants (avoid thread-local cross-thread/model drift).
    bool decode_shapes_initialized = false;
    Shape q_shape = {1, 1, 1, 1};
    Shape kv_shape = {1, 1, 1, 1};
    Shape attn_out_shape = {1, 1, 1};
    Shape token_shape = {1, 1};
    float attn_scale = 1.0f;
};

// Global model pointer (for pipeline access)
static FusedDenseModel* g_fused_dense = nullptr;

struct DensePipelineState {
    std::optional<array> current_token;
};

static std::mutex g_dense_pipeline_mu;
static std::unordered_map<void*, DensePipelineState> g_dense_pipeline_states;

static inline void* dense_decode_state_key(void* cache_ptr, void* model_ptr) {
    return cache_ptr != nullptr ? cache_ptr : model_ptr;
}

static int round_up_step(size_t value, int step) {
    const size_t s = static_cast<size_t>(step);
    return static_cast<int>(((value + s - 1) / s) * s);
}

static int next_cache_capacity(const CacheLayer& layer, size_t required, int current_capacity) {
    if (required == 0) return current_capacity;

    const size_t current = current_capacity > 0 ? static_cast<size_t>(current_capacity) : 0;
    const size_t step = static_cast<size_t>(layer.step);

    if (layer.max_seq_len > 0) {
        const size_t max_cap = layer.max_seq_len;
        if (required > max_cap) {
            throw std::invalid_argument("[dense] kv cache capacity exceeded");
        }

        size_t capacity = current;
        if (capacity == 0) {
            const size_t base = std::max(required, step);
            capacity = std::min(max_cap, static_cast<size_t>(round_up_step(base, layer.step)));
        }
        while (capacity < required) {
            const size_t doubled = capacity * 2;
            if (doubled <= capacity) {
                capacity = max_cap;
                break;
            }
            capacity = std::min(max_cap, doubled);
        }
        if (capacity < required) {
            throw std::invalid_argument("[dense] kv cache capacity exceeded");
        }
        return static_cast<int>(capacity);
    }

    size_t capacity = current;
    if (capacity == 0) {
        const size_t base = std::max(required, step);
        capacity = static_cast<size_t>(round_up_step(base, layer.step));
    }
    while (capacity < required) {
        const size_t doubled = capacity * 2;
        if (doubled <= capacity) {
            throw std::invalid_argument("[dense] kv cache capacity exceeded");
        }
        capacity = doubled;
    }
    return static_cast<int>(capacity);
}

static inline void gqa_expand_attention_kv(const array& q, const array& k_in, const array& v_in, array* k_out, array* v_out) {
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

static constexpr uint8_t kLayerKindAttentionMlp = 0;
static constexpr uint8_t kLayerKindShortConv = 1;

static FusedDenseModel::Layer::LayerKind decode_layer_kind(uint8_t kind_id) {
    switch (kind_id) {
        case kLayerKindAttentionMlp:
            return FusedDenseModel::Layer::LayerKind::attention_mlp;
        case kLayerKindShortConv:
            return FusedDenseModel::Layer::LayerKind::shortconv;
        default:
            throw std::invalid_argument("Unsupported dense layer kind id");
    }
}

// Keep fused dense kernels architecture-agnostic: weight orientation is inferred
// from tensor shapes, never from parameter names or model-family heuristics.
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
        // Square matrix is ambiguous. Match the non-fused path convention
        // (incoming linear weights are treated as [out, in]) unless caller opts out.
        return transpose_when_ambiguous ? transpose(weight) : weight;
    }
    throw std::invalid_argument("[dense] weight orientation incompatible with matmul shape");
}

static array to_fast_metal_dtype(const array& arr) {
    // Keep dense decode matmuls on float16 to stay on the fast Metal kernels.
    // Reintroducing bf16 activations/cache here forces mixed-precision casts in
    // the hot path and substantially reduces per-token throughput.
    return astype(arr, float16);
}

static inline array ensure_float16(const array& arr) {
    return arr.dtype() == float16 ? arr : astype(arr, float16);
}

static inline array ensure_float32(const array& arr) {
    return arr.dtype() == float32 ? arr : astype(arr, float32);
}

// ============================================================================ 
// Forward Pass Implementation
// ============================================================================

static inline void ensure_dense_decode_shapes(FusedDenseModel* m) {
    if (m->decode_shapes_initialized) return;
    m->attn_scale = 1.0f / std::sqrt(static_cast<float>(m->head_dim));
    m->q_shape = {1, 1, m->n_heads, m->head_dim};
    m->kv_shape = {1, 1, m->n_kv_heads, m->head_dim};
    m->attn_out_shape = {1, 1, m->n_heads * m->head_dim};
    m->token_shape = {1, 1};
    m->decode_shapes_initialized = true;
}

static array dense_forward_logits_from_token(
    FusedDenseModel* m,
    MLXCache* cache,
    MLXShortConvCache* shortconv_cache,
    const array& token_idx,
    size_t pos_offset
) {
    const int n_layers = static_cast<int>(m->layers.size());
    ensure_dense_decode_shapes(m);

    // Embedding lookup
    array hidden = take(m->embed_tokens, reshape(token_idx, m->token_shape), 0);

    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
        const auto& l = m->layers[layer_idx];

        // RMS norm
        array normed = ensure_float16(fast::rms_norm(hidden, l.ln1_w, m->rms_eps));
        array mixer_out(0.0f, float32);

        if (l.kind == FusedDenseModel::Layer::LayerKind::shortconv) {
            const int seq_len = normed.shape(1);
            const int d_conv_i = l.shortconv_d_conv;
            const int conv_dim_i = l.shortconv_conv_dim;

            array bcx = matmul(normed, l.shortconv_in_proj_t);
            bcx = astype(bcx, float32);

            array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
            array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
            array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
            array bx = b_gate * x_proj;

            ShortConvLayer* layer_state = nullptr;
            if (shortconv_cache != nullptr && layer_idx < static_cast<int>(shortconv_cache->layers.size())) {
                layer_state = &shortconv_cache->layers[layer_idx];
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

            if (seq_len == 1) {
                const array bx_t = reshape(bx, {1, 1, conv_dim_i});
                if (d_conv_i > 1) {
                    const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                    conv_state = concatenate({state_tail, bx_t}, 1);
                } else {
                    conv_state = bx_t;
                }

                array conv_t = sum(conv_state * *l.shortconv_kernel_broadcast, 1);
                if (l.shortconv_bias_row) {
                    conv_t = conv_t + *l.shortconv_bias_row;
                }

                array c_t = reshape(c_gate, {1, conv_dim_i});
                const array gated = ensure_float16(reshape(conv_t * c_t, {1, 1, conv_dim_i}));
                mixer_out = matmul(gated, l.shortconv_out_proj_t);
            } else {
                std::vector<array> token_outputs;
                token_outputs.reserve(seq_len);
                for (int token_idx_i = 0; token_idx_i < seq_len; token_idx_i++) {
                    const array bx_t = slice(bx, {0, token_idx_i, 0}, {1, token_idx_i + 1, conv_dim_i});
                    if (d_conv_i > 1) {
                        const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
                        conv_state = concatenate({state_tail, bx_t}, 1);
                    } else {
                        conv_state = bx_t;
                    }

                    array conv_t = sum(conv_state * *l.shortconv_kernel_broadcast, 1);
                    if (l.shortconv_bias_row) {
                        conv_t = conv_t + *l.shortconv_bias_row;
                    }

                    array c_t = slice(c_gate, {0, token_idx_i, 0}, {1, token_idx_i + 1, conv_dim_i});
                    c_t = reshape(c_t, {1, conv_dim_i});
                    const array gated = ensure_float16(reshape(conv_t * c_t, {1, 1, conv_dim_i}));
                    token_outputs.push_back(matmul(gated, l.shortconv_out_proj_t));
                }

                mixer_out = (token_outputs.size() == 1) ? token_outputs[0] : concatenate(token_outputs, 1);
            }

            if (layer_state != nullptr) {
                *layer_state->conv_state = conv_state;
            }
        } else {
            auto& cl = cache->layers[layer_idx];

            // Single matmul for Q+K+V (weights pre-concatenated)
            array qkv = matmul(normed, l.qkv_proj);
            auto qkv_parts = split(qkv, {l.q_size, l.q_size + l.kv_size}, -1);
            array q = reshape(qkv_parts[0], m->q_shape);
            array k = reshape(qkv_parts[1], m->kv_shape);
            array v = reshape(qkv_parts[2], m->kv_shape);

            if (l.q_norm) q = fast::rms_norm(q, *l.q_norm, m->rms_eps);
            if (l.k_norm) k = fast::rms_norm(k, *l.k_norm, m->rms_eps);

            q = transpose(q, g_transpose_perm);
            k = transpose(k, g_transpose_perm);
            v = transpose(v, g_transpose_perm);

            q = fast::rope(q, m->head_dim, false, m->rope_theta, 1.0f, static_cast<int>(pos_offset));
            k = fast::rope(k, m->head_dim, false, m->rope_theta, 1.0f, static_cast<int>(pos_offset));

            // Cache update
            size_t prev = cl.offset;
            int offset = static_cast<int>(prev + 1);

            // Dense decode keeps KV cache in float16 so q/k/v updates do not
            // cast on every token before attention.
            if (cl.k_bfloat16 == nullptr) {
                const int capacity = next_cache_capacity(cl, static_cast<size_t>(offset), 0);
                Shape shape = {1, m->n_kv_heads, capacity, m->head_dim};
                cl.k_bfloat16 = new array(zeros(shape, float16));
                cl.v_bfloat16 = new array(zeros(shape, float16));
            } else if (offset > cl.k_bfloat16->shape(2)) {
                const int current_capacity = cl.k_bfloat16->shape(2);
                const int new_capacity = next_cache_capacity(cl, static_cast<size_t>(offset), current_capacity);
                Shape new_shape = {1, m->n_kv_heads, new_capacity, m->head_dim};
                array new_k = zeros(new_shape, cl.k_bfloat16->dtype());
                array new_v = zeros(new_shape, cl.v_bfloat16->dtype());
                const int prev_capacity = current_capacity;
                if (prev_capacity > 0) {
                    const Shape copy_start = {0, 0, 0, 0};
                    const Shape copy_stop = {1, m->n_kv_heads, prev_capacity, m->head_dim};
                    new_k = slice_update(new_k, slice(*cl.k_bfloat16, copy_start, copy_stop), copy_start, copy_stop);
                    new_v = slice_update(new_v, slice(*cl.v_bfloat16, copy_start, copy_stop), copy_start, copy_stop);
                }
                *cl.k_bfloat16 = new_k;
                *cl.v_bfloat16 = new_v;
            }

            Shape update_start = {0, 0, static_cast<int>(prev), 0};
            Shape update_stop = {1, m->n_kv_heads, offset, m->head_dim};
            *cl.k_bfloat16 = slice_update(*cl.k_bfloat16, k, update_start, update_stop);
            *cl.v_bfloat16 = slice_update(*cl.v_bfloat16, v, update_start, update_stop);
            cl.offset = offset;

            const Shape slice_stop = {1, m->n_kv_heads, offset, m->head_dim};
            array k_full = slice(*cl.k_bfloat16, g_slice_start, slice_stop);
            array v_full = slice(*cl.v_bfloat16, g_slice_start, slice_stop);
            array attn_k(0.0f, float32);
            array attn_v(0.0f, float32);
            gqa_expand_attention_kv(q, k_full, v_full, &attn_k, &attn_v);
            array attn_out = fast::scaled_dot_product_attention(q, attn_k, attn_v, m->attn_scale, "");

            attn_out = transpose(attn_out, g_transpose_perm);
            attn_out = reshape(attn_out, m->attn_out_shape);

            mixer_out = matmul(attn_out, l.o_proj);
        }
        array hidden_1 = ensure_float16(hidden + ensure_float16(mixer_out));

        // FFN - single matmul for gate+up (weights pre-concatenated)
        array normed_2 = ensure_float16(fast::rms_norm(hidden_1, l.ln2_w, m->rms_eps));
        array gate_up = matmul(normed_2, l.gate_up_proj);
        auto parts = split(gate_up, 2, -1);
        array& gate = parts[0];
        array& up = parts[1];
        array down = matmul(ensure_float16(gate * sigmoid(gate) * up), l.down_proj);

        hidden = ensure_float16(hidden_1 + ensure_float16(down));
    }

    // Final norm + LM head
    array final_normed = ensure_float16(fast::rms_norm(hidden, m->ln_final, m->rms_eps));
    array logits = matmul(final_normed, m->lm_head);
    return reshape(logits, {-1});
}

static array dense_forward_from_token(
    FusedDenseModel* m,
    MLXCache* cache,
    MLXShortConvCache* shortconv_cache,
    const array& token_idx,
    size_t pos_offset
) {
    array logits = dense_forward_logits_from_token(m, cache, shortconv_cache, token_idx, pos_offset);
    return argmax(logits, 0);
}

// ============================================================================
// C API - Model Lifecycle
// ============================================================================

extern "C" {

void* mlx_dense_model_create(
    size_t n_layers,
    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t hidden_dim,
    float rope_theta, float rms_eps
) {
    auto* model = new FusedDenseModel();
    model->layers.resize(n_layers);
    model->n_heads = static_cast<int>(n_heads);
    model->n_kv_heads = static_cast<int>(n_kv_heads);
    model->head_dim = static_cast<int>(head_dim);
    model->hidden_dim = static_cast<int>(hidden_dim);
    model->rope_theta = rope_theta;
    model->rms_eps = rms_eps;
    g_fused_dense = model;
    return model;
}

void mlx_dense_model_set_embeddings(void* model, const void* embed) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    fused_model->embed_tokens = to_fast_metal_dtype(*static_cast<const array*>(embed));
}

void mlx_dense_model_set_final(void* model, const void* ln_w, const void* lm_head) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    fused_model->ln_final = *static_cast<const array*>(ln_w);
    fused_model->lm_head = orient_matmul_rhs(
        *static_cast<const array*>(lm_head),
        fused_model->hidden_dim,
        std::nullopt,
        false
    );
    fused_model->lm_head = to_fast_metal_dtype(fused_model->lm_head);
    eval(fused_model->lm_head);
}

void mlx_dense_model_set_topology(
    void* model,
    const uint8_t* layer_kinds,
    size_t n_layer_kinds
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    if (n_layer_kinds != fused_model->layers.size()) {
        throw std::invalid_argument("dense topology length does not match layer count");
    }
    for (size_t idx = 0; idx < n_layer_kinds; idx++) {
        fused_model->layers[idx].kind = decode_layer_kind(layer_kinds[idx]);
    }
    fused_model->topology_initialized = true;
}

void mlx_dense_model_set_layer(
    void* model, size_t layer_idx,
    const void* ln1_w,
    const void* q_proj, const void* k_proj, const void* v_proj, const void* o_proj,
    const void* ln2_w,
    const void* gate_proj, const void* up_proj, const void* down_proj,
    const void* q_norm, const void* k_norm,
    size_t shortconv_d_conv,
    size_t shortconv_conv_dim,
    const void* shortconv_in_proj,
    const void* shortconv_conv_weight,
    const void* shortconv_conv_bias,
    const void* shortconv_out_proj
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    if (!fused_model->topology_initialized) {
        throw std::invalid_argument("mlx_dense_model_set_topology must be called before mlx_dense_model_set_layer");
    }
    auto& layer = fused_model->layers[layer_idx];

    layer.ln1_w = *static_cast<const array*>(ln1_w);
    layer.ln2_w = *static_cast<const array*>(ln2_w);
    std::vector<array> to_eval;

    if (layer.kind == FusedDenseModel::Layer::LayerKind::shortconv) {
        layer.shortconv_d_conv = static_cast<int>(shortconv_d_conv);
        layer.shortconv_conv_dim = static_cast<int>(shortconv_conv_dim);
        layer.shortconv_in_proj_t = orient_matmul_rhs(
            *static_cast<const array*>(shortconv_in_proj),
            fused_model->hidden_dim,
            layer.shortconv_conv_dim * 3,
            true
        );
        layer.shortconv_in_proj_t = to_fast_metal_dtype(layer.shortconv_in_proj_t);
        layer.shortconv_out_proj_t = orient_matmul_rhs(
            *static_cast<const array*>(shortconv_out_proj),
            layer.shortconv_conv_dim,
            fused_model->hidden_dim,
            true
        );
        layer.shortconv_out_proj_t = to_fast_metal_dtype(layer.shortconv_out_proj_t);

        array conv_kernel = *static_cast<const array*>(shortconv_conv_weight);
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
        if (shortconv_conv_bias != nullptr) {
            layer.shortconv_bias_row = reshape(
                astype(*static_cast<const array*>(shortconv_conv_bias), float32),
                {1, layer.shortconv_conv_dim}
            );
        } else {
            layer.shortconv_bias_row = std::nullopt;
        }

        to_eval = {
            layer.shortconv_in_proj_t,
            layer.shortconv_out_proj_t,
            *layer.shortconv_kernel_broadcast,
        };
        if (layer.shortconv_bias_row) {
            to_eval.push_back(*layer.shortconv_bias_row);
        }
    } else {
        auto q_t = orient_matmul_rhs(*static_cast<const array*>(q_proj), fused_model->hidden_dim, std::nullopt, true);
        auto k_t = orient_matmul_rhs(*static_cast<const array*>(k_proj), fused_model->hidden_dim, std::nullopt, true);
        auto v_t = orient_matmul_rhs(*static_cast<const array*>(v_proj), fused_model->hidden_dim, std::nullopt, true);
        // GQA models can have attention output width (n_heads * head_dim)
        // different from hidden_dim. Orient o_proj using the actual Q width.
        layer.o_proj = orient_matmul_rhs(
            *static_cast<const array*>(o_proj),
            q_t.shape(1),
            fused_model->hidden_dim,
            true
        );
        q_t = to_fast_metal_dtype(q_t);
        k_t = to_fast_metal_dtype(k_t);
        v_t = to_fast_metal_dtype(v_t);
        layer.o_proj = to_fast_metal_dtype(layer.o_proj);

        // Pre-concatenate QKV for single matmul
        layer.qkv_proj = concatenate({q_t, k_t, v_t}, 1);
        layer.q_size = q_t.shape(1);
        layer.kv_size = k_t.shape(1);

        if (q_norm) layer.q_norm = *static_cast<const array*>(q_norm);
        if (k_norm) layer.k_norm = *static_cast<const array*>(k_norm);

        to_eval = {layer.qkv_proj, layer.o_proj};
    }

    auto gate_t = orient_matmul_rhs(*static_cast<const array*>(gate_proj), fused_model->hidden_dim, std::nullopt, true);
    auto up_t = orient_matmul_rhs(
        *static_cast<const array*>(up_proj),
        fused_model->hidden_dim,
        gate_t.shape(1),
        true
    );
    layer.down_proj = orient_matmul_rhs(
        *static_cast<const array*>(down_proj),
        gate_t.shape(1),
        fused_model->hidden_dim,
        true
    );
    gate_t = to_fast_metal_dtype(gate_t);
    up_t = to_fast_metal_dtype(up_t);
    layer.down_proj = to_fast_metal_dtype(layer.down_proj);
    // Pre-concatenate gate+up for single matmul
    layer.gate_up_proj = concatenate({gate_t, up_t}, 1);
    to_eval.push_back(layer.gate_up_proj);
    to_eval.push_back(layer.down_proj);

    // Evaluate all to materialize
    eval(to_eval);
}

void mlx_dense_model_free(void* model) {
    delete static_cast<FusedDenseModel*>(model);
    if (g_fused_dense == model) g_fused_dense = nullptr;
}

void* mlx_dense_decode_step_logits(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array logits = dense_forward_logits_from_token(
        fused_model,
        cache_state,
        shortconv_cache_state,
        token,
        pos_offset
    );
    return pool_array(ensure_float32(logits));
}

// ============================================================================
// C API - Pipelined Decode
// ============================================================================
// Implements async pipelining: while GPU runs token N, CPU builds graph for N+1

void mlx_dense_pipeline_prime(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    uint32_t first_token_id,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(first_token_id);
    array first_token = array(&token_id_i32, {1}, int32);

    array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, first_token, pos_offset);
    async_eval(next);

    void* key = dense_decode_state_key(cache_ptr, model);
    std::lock_guard<std::mutex> lock(g_dense_pipeline_mu);
    auto& state = g_dense_pipeline_states[key];
    state.current_token = std::move(next);
}

uint32_t mlx_dense_pipeline_step(
    void* model,
    void* cache_ptr,
    void* shortconv_cache_ptr,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    array current(0.0f, float32);
    {
        void* key = dense_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_dense_pipeline_mu);
        auto it = g_dense_pipeline_states.find(key);
        if (it == g_dense_pipeline_states.end() || !it->second.current_token) return 0;
        current = *it->second.current_token;
    }

    // Build graph for NEXT token using current (lazy) token
    array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, current, pos_offset);

    // Queue next token computation
    async_eval(next);

    // NOW materialize current token
    eval(current);
    uint32_t result = static_cast<uint32_t>(current.item<int32_t>());

    // Rotate buffers
    {
        void* key = dense_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_dense_pipeline_mu);
        auto& state = g_dense_pipeline_states[key];
        state.current_token = std::move(next);
    }

    return result;
}

uint32_t mlx_dense_pipeline_flush(void* model, void* cache_ptr) {
    std::optional<array> current_token;
    {
        void* key = dense_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_dense_pipeline_mu);
        auto it = g_dense_pipeline_states.find(key);
        if (it == g_dense_pipeline_states.end() || !it->second.current_token) return 0;
        current_token = std::move(it->second.current_token);
        g_dense_pipeline_states.erase(it);
    }
    eval(*current_token);
    return static_cast<uint32_t>((*current_token).item<int32_t>());
}

uint32_t mlx_dense_decode_batch(
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
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);

    auto is_eos = [&](uint32_t token) {
        for (size_t idx = 0; idx < n_eos_ids; idx++) {
            if (token == eos_ids[idx]) return true;
        }
        return false;
    };

    uint32_t current_token = first_token;
    size_t pos = start_pos;
    size_t generated_count = 0;

    while (generated_count < max_tokens) {
        int32_t token_id_i32 = static_cast<int32_t>(current_token);
        array token = array(&token_id_i32, {1}, int32);
        array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, token, pos);
        eval(next);
        current_token = static_cast<uint32_t>(next.item<int32_t>());
        out_tokens[generated_count++] = current_token;
        pos++;
        if (is_eos(current_token)) break;
    }

    return static_cast<uint32_t>(generated_count);
}

} // extern "C"
