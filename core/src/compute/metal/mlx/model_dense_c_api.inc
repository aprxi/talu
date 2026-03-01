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

void mlx_dense_model_set_arch_config(
    void* model,
    bool has_norm_weight_offset,
    bool use_gelu,
    float query_pre_attn_scalar
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    fused_model->has_norm_weight_offset = has_norm_weight_offset;
    fused_model->use_gelu = use_gelu;
    fused_model->query_pre_attn_scalar = query_pre_attn_scalar;
    fused_model->decode_shapes_initialized = false;
}

void mlx_dense_model_set_scaling_config(
    void* model,
    float embedding_multiplier,
    float attention_multiplier,
    float residual_multiplier,
    float logits_scaling
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    fused_model->embedding_multiplier = embedding_multiplier;
    fused_model->attention_multiplier = attention_multiplier;
    fused_model->residual_multiplier = residual_multiplier;
    fused_model->logits_scaling = logits_scaling;
    fused_model->decode_shapes_initialized = false;
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
        const bool has_standard_attention = q_proj != nullptr &&
            k_proj != nullptr &&
            v_proj != nullptr &&
            o_proj != nullptr;
        const bool has_partial_standard_attention = q_proj != nullptr ||
            k_proj != nullptr ||
            v_proj != nullptr ||
            o_proj != nullptr;
        if (has_partial_standard_attention && !has_standard_attention) {
            throw std::invalid_argument("[dense] incomplete standard attention tensors");
        }

        if (has_standard_attention) {
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

void mlx_dense_model_set_layer_mla_bf16(
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
    const void* q_b_w,
    const void* kv_a_w,
    const void* kv_b_w,
    const void* q_a_norm,
    const void* kv_a_norm,
    const void* o_w
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    if (!fused_model->topology_initialized) {
        throw std::invalid_argument("mlx_dense_model_set_topology must be called before mlx_dense_model_set_layer_mla_bf16");
    }
    if (q_a_w == nullptr || q_b_w == nullptr || kv_a_w == nullptr || kv_b_w == nullptr ||
        q_a_norm == nullptr || kv_a_norm == nullptr || o_w == nullptr) {
        throw std::invalid_argument("mlx_dense_model_set_layer_mla_bf16 requires non-null MLA tensors");
    }
    auto& layer = fused_model->layers[layer_idx];
    if (layer.kind != FusedDenseModel::Layer::LayerKind::attention_mlp) {
        throw std::invalid_argument("mlx_dense_model_set_layer_mla_bf16 requires attention_mlp layer kind");
    }

    const int n_heads_i = static_cast<int>(n_heads);
    const int q_lora_rank_i = static_cast<int>(q_lora_rank);
    const int kv_lora_rank_i = static_cast<int>(kv_lora_rank);
    const int qk_head_dim_i = static_cast<int>(qk_head_dim);
    const int qk_rope_head_dim_i = static_cast<int>(qk_rope_head_dim);
    const int qk_nope_head_dim_i = static_cast<int>(qk_nope_head_dim);
    const int v_head_dim_i = static_cast<int>(v_head_dim);

    layer.use_mla = true;
    layer.mla_n_heads = n_heads_i;
    layer.mla_q_lora_rank = q_lora_rank_i;
    layer.mla_kv_lora_rank = kv_lora_rank_i;
    layer.mla_qk_head_dim = qk_head_dim_i;
    layer.mla_qk_rope_head_dim = qk_rope_head_dim_i;
    layer.mla_qk_nope_head_dim = qk_nope_head_dim_i;
    layer.mla_v_head_dim = v_head_dim_i;

    layer.mla_q_a_w = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(q_a_w),
        fused_model->hidden_dim,
        q_lora_rank_i,
        true
    ));
    layer.mla_q_b_w = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(q_b_w),
        q_lora_rank_i,
        n_heads_i * qk_head_dim_i,
        true
    ));
    layer.mla_kv_a_w = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(kv_a_w),
        fused_model->hidden_dim,
        kv_lora_rank_i + qk_rope_head_dim_i,
        true
    ));
    layer.mla_kv_b_w = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(kv_b_w),
        kv_lora_rank_i,
        n_heads_i * (qk_nope_head_dim_i + v_head_dim_i),
        true
    ));
    layer.mla_q_a_norm = *static_cast<const array*>(q_a_norm);
    layer.mla_kv_a_norm = *static_cast<const array*>(kv_a_norm);
    layer.mla_o_w = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(o_w),
        n_heads_i * v_head_dim_i,
        fused_model->hidden_dim,
        true
    ));

    eval({
        layer.mla_q_a_w,
        layer.mla_q_b_w,
        layer.mla_kv_a_w,
        layer.mla_kv_b_w,
        layer.mla_q_a_norm,
        layer.mla_kv_a_norm,
        layer.mla_o_w,
    });
}

void mlx_dense_model_set_layer_mamba_bf16(
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
    const void* in_proj,
    const void* out_proj,
    const void* ln2_w,
    const void* gate_up,
    const void* down_proj
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    if (!fused_model->topology_initialized) {
        throw std::invalid_argument("mlx_dense_model_set_topology must be called before mlx_dense_model_set_layer_mamba_bf16");
    }
    if (ln1_w == nullptr || conv_weight == nullptr || a_log == nullptr || d_skip == nullptr ||
        in_proj == nullptr || out_proj == nullptr || ln2_w == nullptr) {
        throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 requires non-null Mamba core tensors");
    }

    auto& layer = fused_model->layers[layer_idx];
    if (layer.kind != FusedDenseModel::Layer::LayerKind::mamba) {
        throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 requires mamba layer kind");
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
    array conv_kernel = *static_cast<const array*>(conv_weight);
    if (conv_kernel.ndim() == 3 &&
        conv_kernel.shape(0) == 1 &&
        conv_kernel.shape(1) == layer.mamba_d_conv &&
        conv_kernel.shape(2) == xbc_len) {
        layer.mamba_conv_weight = ensure_float32(conv_kernel);
    } else {
        if (conv_kernel.ndim() == 3) {
            conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
        }
        if (conv_kernel.ndim() != 2) {
            throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 expects conv_weight rank-2 or rank-3");
        }
        if (conv_kernel.shape(0) == xbc_len && conv_kernel.shape(1) == layer.mamba_d_conv) {
            conv_kernel = transpose(conv_kernel);
        } else if (!(conv_kernel.shape(0) == layer.mamba_d_conv && conv_kernel.shape(1) == xbc_len)) {
            throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 conv_weight shape incompatible with mamba dims");
        }
        layer.mamba_conv_weight = reshape(ensure_float32(conv_kernel), {1, layer.mamba_d_conv, xbc_len});
    }
    if (conv_bias != nullptr) {
        const array conv_bias_arr = ensure_float32(*static_cast<const array*>(conv_bias));
        if (conv_bias_arr.ndim() == 2 && conv_bias_arr.shape(0) == 1 && conv_bias_arr.shape(1) == xbc_len) {
            layer.mamba_conv_bias = conv_bias_arr;
        } else {
            layer.mamba_conv_bias = reshape(conv_bias_arr, {1, xbc_len});
        }
    } else {
        layer.mamba_conv_bias = std::nullopt;
    }

    layer.mamba_a_log = *static_cast<const array*>(a_log);
    layer.mamba_d_skip = *static_cast<const array*>(d_skip);
    layer.mamba_dt_bias = dt_bias ? std::make_optional(*static_cast<const array*>(dt_bias)) : std::nullopt;
    layer.mamba_norm_weight = norm_weight ? std::make_optional(*static_cast<const array*>(norm_weight)) : std::nullopt;
    layer.mamba_in_proj = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(in_proj),
        fused_model->hidden_dim,
        std::nullopt,
        true
    ));
    layer.mamba_out_proj = to_fast_metal_dtype(orient_matmul_rhs(
        *static_cast<const array*>(out_proj),
        layer.mamba_n_heads * layer.mamba_d_head,
        fused_model->hidden_dim,
        true
    ));
    layer.ln2_w = *static_cast<const array*>(ln2_w);

    const bool has_dense_ffn = gate_up != nullptr || down_proj != nullptr;
    if (has_dense_ffn) {
        if (gate_up == nullptr || down_proj == nullptr) {
            throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 requires complete dense FFN tensors");
        }
        layer.mamba_gate_up = to_fast_metal_dtype(orient_matmul_rhs(
            *static_cast<const array*>(gate_up),
            fused_model->hidden_dim,
            std::nullopt,
            true
        ));
        const int gate_up_width = layer.mamba_gate_up->shape(1);
        if ((gate_up_width % 2) != 0) {
            throw std::invalid_argument("mlx_dense_model_set_layer_mamba_bf16 expects gate_up width divisible by 2");
        }
        const int ffn_width = gate_up_width / 2;
        layer.mamba_down_proj = to_fast_metal_dtype(orient_matmul_rhs(
            *static_cast<const array*>(down_proj),
            ffn_width,
            fused_model->hidden_dim,
            true
        ));
    } else {
        layer.mamba_gate_up = std::nullopt;
        layer.mamba_down_proj = std::nullopt;
    }

    std::vector<array> to_eval = {
        layer.mamba_conv_weight,
        layer.mamba_a_log,
        layer.mamba_d_skip,
        layer.mamba_in_proj,
        layer.mamba_out_proj,
        layer.ln2_w,
    };
    if (layer.mamba_conv_bias) to_eval.push_back(*layer.mamba_conv_bias);
    if (layer.mamba_dt_bias) to_eval.push_back(*layer.mamba_dt_bias);
    if (layer.mamba_norm_weight) to_eval.push_back(*layer.mamba_norm_weight);
    if (layer.mamba_gate_up) to_eval.push_back(*layer.mamba_gate_up);
    if (layer.mamba_down_proj) to_eval.push_back(*layer.mamba_down_proj);
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
    void* mamba_cache_ptr,
    uint32_t token_id,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(token_id);
    array token = array(&token_id_i32, {1}, int32);

    array logits = dense_forward_logits_from_token(
        fused_model,
        cache_state,
        shortconv_cache_state,
        mamba_cache_state,
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
    void* mamba_cache_ptr,
    uint32_t first_token_id,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    int32_t token_id_i32 = static_cast<int32_t>(first_token_id);
    array first_token = array(&token_id_i32, {1}, int32);

    array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, mamba_cache_state, first_token, pos_offset);
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
    void* mamba_cache_ptr,
    size_t pos_offset
) {
    auto* fused_model = static_cast<FusedDenseModel*>(model);
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    auto* shortconv_cache_state = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

    array current(0.0f, float32);
    {
        void* key = dense_decode_state_key(cache_ptr, model);
        std::lock_guard<std::mutex> lock(g_dense_pipeline_mu);
        auto it = g_dense_pipeline_states.find(key);
        if (it == g_dense_pipeline_states.end() || !it->second.current_token) return 0;
        current = *it->second.current_token;
    }

    // Build graph for NEXT token using current (lazy) token
    array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, mamba_cache_state, current, pos_offset);

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

uint32_t mlx_dense_pipeline_flush(void* model, void* cache_ptr, void* shortconv_cache_ptr, void* mamba_cache_ptr) {
    (void)shortconv_cache_ptr;
    (void)mamba_cache_ptr;
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
    void* mamba_cache_ptr,
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
    auto* mamba_cache_state = static_cast<MLXMambaCache*>(mamba_cache_ptr);

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
        array next = dense_forward_from_token(fused_model, cache_state, shortconv_cache_state, mamba_cache_state, token, pos);
        eval(next);
        current_token = static_cast<uint32_t>(next.item<int32_t>());
        out_tokens[generated_count++] = current_token;
        pos++;
        if (is_eos(current_token)) break;
    }

    return static_cast<uint32_t>(generated_count);
}
