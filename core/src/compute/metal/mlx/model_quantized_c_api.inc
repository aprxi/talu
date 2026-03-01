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

    const int d_inner = layer.mamba_n_heads * layer.mamba_d_head;
    const int xbc_len = d_inner + 2 * layer.mamba_n_groups * layer.mamba_d_state;
    const auto ensure_f32 = [](const array& arr) -> array {
        return arr.dtype() == float32 ? arr : astype(arr, float32);
    };
    array conv_kernel = *static_cast<const array*>(conv_weight);
    if (conv_kernel.ndim() == 3 &&
        conv_kernel.shape(0) == 1 &&
        conv_kernel.shape(1) == layer.mamba_d_conv &&
        conv_kernel.shape(2) == xbc_len) {
        layer.mamba_conv_w = ensure_f32(conv_kernel);
    } else {
        if (conv_kernel.ndim() == 3) {
            conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
        }
        if (conv_kernel.ndim() != 2) {
            throw std::invalid_argument("mlx_fused_model_set_layer_mamba_quantized expects conv_weight rank-2 or rank-3");
        }
        if (conv_kernel.shape(0) == xbc_len && conv_kernel.shape(1) == layer.mamba_d_conv) {
            conv_kernel = transpose(conv_kernel);
        } else if (!(conv_kernel.shape(0) == layer.mamba_d_conv && conv_kernel.shape(1) == xbc_len)) {
            throw std::invalid_argument("mlx_fused_model_set_layer_mamba_quantized conv_weight shape incompatible with mamba dims");
        }
        layer.mamba_conv_w = reshape(ensure_f32(conv_kernel), {1, layer.mamba_d_conv, xbc_len});
    }
    if (conv_bias != nullptr) {
        const array conv_bias_arr = ensure_f32(*static_cast<const array*>(conv_bias));
        if (conv_bias_arr.ndim() == 2 && conv_bias_arr.shape(0) == 1 && conv_bias_arr.shape(1) == xbc_len) {
            layer.mamba_conv_b = conv_bias_arr;
        } else {
            layer.mamba_conv_b = reshape(conv_bias_arr, {1, xbc_len});
        }
    } else {
        layer.mamba_conv_b = std::nullopt;
    }

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
