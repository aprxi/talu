// bridge_api_init.inc — Model lifecycle: mlx_is_available, mlx_validate_config,
// mlx_create, mlx_clone, mlx_destroy, mlx_reset.

int32_t mlx_is_available(void) {
    return metal::is_available() ? 1 : 0;
}

int32_t mlx_validate_config(const char* model_path) {
    if (!model_path) {
        g_last_error = "mlx_validate_config: model_path is null";
        return 0;
    }
    try {
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(std::string("resolved model_path does not exist: ") + model_path);
        }
        const ParsedModelConfig parsed = parse_model_config(model_path);
        if (!validate_memory_budget_or_set_error(model_path, parsed.cfg)) {
            return 0;
        }
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_validate_config";
        return 0;
    }
}

static mlx_ctx* mlx_create_impl(const char* model_id, const char* model_path, int32_t seed, const mlx_model_flags* flags) {
    bool wired_limit_acquired = false;
    bool rng_entered = false;
    const bool qmm_debug = []() {
        if (const char* value = std::getenv("TALU_METAL_QMM_DEBUG")) {
            return std::string(value) == "1";
        }
        return false;
    }();
    const bool nvfp4_debug = []() {
        if (const char* value = std::getenv("TALU_METAL_NVFP4_DEBUG")) {
            return std::string(value) == "1";
        }
        return false;
    }();
    const size_t qmm_attempts_before = g_qmm_attempts;
    const size_t qmm_success_before = g_qmm_success;
    try {
        if (!metal::is_available()) {
            g_last_error = "Metal backend is not available";
            return nullptr;
        }
        if (!model_id || !model_path) {
            g_last_error = "mlx_create: model_id/model_path is null";
            return nullptr;
        }

        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error(std::string("resolved model_path does not exist: ") + model_path);
        }

        set_default_device(Device::gpu);
        enter_rng_epoch(seed);
        rng_entered = true;

        auto ctx = std::make_unique<mlx_ctx>();
        ctx->model_id = std::string(model_id);
        ctx->model_path = std::string(model_path);
        if (const char* env = std::getenv("TALU_METAL_PROFILE_LAYERS")) {
            ctx->profile_layers = std::string(env) == "1";
        }

        // Cap Metal wired memory only when explicitly requested. Large local
        // models can exceed the recommended working-set ceiling during init.
        const bool enable_wired_limit = env_truthy("TALU_METAL_WIRED_LIMIT", false);
        if (enable_wired_limit) {
            wired_limit_acquired = acquire_wired_limit();
        }

        const ParsedModelConfig parsed_cfg = parse_model_config(ctx->model_path, flags);
        ctx->cfg = parsed_cfg.cfg;
        if (!validate_memory_budget_or_set_error(ctx->model_path, ctx->cfg)) {
            return nullptr;
        }

        auto tensors = load_weight_tensors(ctx->model_path);
        sanitize_model_tensors(tensors, ctx->cfg.tie_word_embeddings, parsed_cfg.allow_norm_shift);
        const bool has_fp8_meta = has_fp8_quant_metadata(tensors);
        ctx->has_fp8_meta = has_fp8_meta;
        const bool has_mxfp8_meta = has_mxfp8_quant_metadata(tensors);
        ctx->has_mxfp8_meta = has_mxfp8_meta;
        bool has_grouped_affine_meta = has_grouped_affine_quant_metadata(tensors);
        const bool has_nvfp4_meta = has_nvfp4_quant_metadata(tensors);
        if (has_grouped_affine_meta && has_nvfp4_meta) {
            // Hybrid artifacts preserve grouped-affine execution tensors.
            ctx->cfg.quant_bits = 4;
            ctx->cfg.quant_group_size = 32;
        } else if (has_nvfp4_meta && !has_grouped_affine_meta) {
            // Pure NVFP4 must stay on the native MLX NVFP4 path.
            ctx->cfg.quant_bits = 4;
            ctx->cfg.quant_group_size = 16;
        }
        ctx->has_grouped_affine_meta = has_grouped_affine_meta;
        // Decode-time QMM defaults on for quantized checkpoints:
        // - native MXFP8 metadata
        // - grouped-affine metadata
        // - nvfp4 packed metadata
        // Callers can still force on/off via TALU_METAL_FP8_DECODE_QMM.
        const bool default_decode_qmm = has_mxfp8_meta || has_grouped_affine_meta || has_nvfp4_meta;
        ctx->fp8_decode_qmm_enabled = env_truthy("TALU_METAL_FP8_DECODE_QMM", default_decode_qmm);
        const bool enable_mlp_qmm = ctx->fp8_decode_qmm_enabled;
        // v3 cache/runtime policy: quantized path is allowed for prefill too,
        // so dense RHS is not retained by default.
        // Note: per-layer input projection weights (per_layer_input_gate_rhs etc.) are
        // small and always loaded as dense unconditionally; they do not depend on this flag.
        const bool default_keep_dense_rhs_for_prefill = false;
        const bool keep_dense_rhs_for_prefill = env_truthy(
            "TALU_METAL_KEEP_DENSE_RHS_FOR_PREFILL",
            default_keep_dense_rhs_for_prefill
        );
        size_t prefill_cache_budget_bytes = 0;
        if (!keep_dense_rhs_for_prefill) {
            const int prefill_cache_mb = env_positive_int("TALU_METAL_PREFILL_DENSE_CACHE_MB", 0);
            if (prefill_cache_mb > 0) {
                prefill_cache_budget_bytes =
                    static_cast<size_t>(prefill_cache_mb) * 1024ULL * 1024ULL;
            }
        }
        const bool enable_linear_prefill_dense_cache =
            prefill_cache_budget_bytes > 0 &&
            env_truthy("TALU_METAL_PREFILL_DENSE_CACHE_LINEAR_ATTN", true);
        const bool enable_full_attn_prefill_dense_cache =
            prefill_cache_budget_bytes > 0 &&
            env_truthy("TALU_METAL_PREFILL_DENSE_CACHE_FULL_ATTN", false);
        const auto maybe_build_prefill_rhs_cache = [&](
            const std::string& weight_key,
            const array& q_w,
            const array& q_scales,
            const array& q_biases
        ) -> array {
            if (prefill_cache_budget_bytes == 0) return array(0.0f);
            if (q_w.ndim() != 2 || q_scales.ndim() != 2 || q_biases.ndim() != 2) {
                return array(0.0f);
            }
            const size_t estimated_bytes =
                grouped_affine_dequantized_weight_bytes(q_w, ctx->cfg.quant_bits, bfloat16);
            if (estimated_bytes == 0 || estimated_bytes > prefill_cache_budget_bytes) {
                return array(0.0f);
            }
            array rhs = build_grouped_affine_prefill_rhs(
                weight_key,
                q_w,
                q_scales,
                q_biases,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                bfloat16
            );
            if (rhs.size() <= 1) return rhs;
            eval(rhs);
            prefill_cache_budget_bytes -= estimated_bytes;
            return rhs;
        };
        auto load_embed_weight = [&](const std::string& weight_key) -> array {
            return load_linear_weight(tensors, ctx->cfg, weight_key);
        };

        auto load_rhs_weight = [&](const std::string& weight_key) -> array {
            const array dense = load_linear_weight(tensors, ctx->cfg, weight_key);
            return to_rhs(dense, weight_key);
        };

        auto maybe_quantize_mxfp8_matrix = [&](
            const std::unordered_map<std::string, array>& tensors_ref,
            const BridgeModelConfig& cfg_ref,
            const std::string& weight_key,
            const array& lhs_weight,
            bool enabled,
            array* out_q_w,
            array* out_q_scales,
            array* out_q_biases,
            bool* out_has_q,
            int* out_detected_bits = nullptr
        ) {
            ::maybe_quantize_mxfp8_matrix(
                tensors_ref,
                cfg_ref,
                weight_key,
                lhs_weight,
                enabled,
                out_q_w,
                out_q_scales,
                out_q_biases,
                out_has_q,
                out_detected_bits
            );
        };

        const std::string text_prefix = detect_text_prefix(tensors);
        const std::string layer_prefix = text_prefix + "layers.";
        const std::string embed_weight_key = has_tensor(tensors, text_prefix + "embed_tokens.weight")
            ? text_prefix + "embed_tokens.weight"
            : "model.embed_tokens.weight";
        const bool enable_quantized_embed_lookup = env_truthy(
            "TALU_METAL_GROUPED_AFFINE_EMBED_LOOKUP",
            has_grouped_affine_meta
        );
        if (enable_quantized_embed_lookup && has_grouped_affine_metadata_for_key(tensors, embed_weight_key)) {
            int embed_bits = 0;
            const array packed_embed = require_tensor(tensors, embed_weight_key);
            ctx->embed_tokens_has_q = maybe_capture_grouped_affine_quant(
                tensors,
                ctx->cfg,
                embed_weight_key,
                packed_embed,
                &ctx->embed_tokens_q_w,
                &ctx->embed_tokens_q_scales,
                &ctx->embed_tokens_q_biases,
                &embed_bits
            );
            if (ctx->embed_tokens_has_q) {
                ctx->embed_tokens_q_bits = embed_bits > 0 ? embed_bits : ctx->cfg.quant_bits;
            }
        }
        if (!ctx->embed_tokens_has_q) {
            ctx->embed_tokens = load_embed_weight(embed_weight_key);
        }
        ctx->final_norm_w = require_any_tensor(
            tensors,
            {
                text_prefix + "norm.weight",
                text_prefix + "embedding_norm.weight",
                "model.norm.weight",
                "model.embedding_norm.weight",
            },
            "final_norm"
        );

        array lm_head_weight = array(0.0f); // [vocab, hidden] dense or packed source
        std::string lm_head_key;
        if (ctx->cfg.tie_word_embeddings) {
            lm_head_key = embed_weight_key;
            if (ctx->embed_tokens_has_q) {
                lm_head_weight = require_tensor(tensors, lm_head_key);
            } else {
                lm_head_weight = ctx->embed_tokens;
            }
        } else {
            if (has_tensor(tensors, text_prefix + "lm_head.weight")) {
                lm_head_key = text_prefix + "lm_head.weight";
            } else if (has_tensor(tensors, "lm_head.weight")) {
                lm_head_key = "lm_head.weight";
            } else {
                throw std::runtime_error("missing lm_head weight (tie_word_embeddings=false)");
            }
            // Defer loading the dense weight: for large quantized checkpoints (TQ4, FP8, NVFP4)
            // the QMM path below avoids the full dequantize. Load dense only as a fallback.
        }

        const bool enable_lm_head_qmm = env_truthy("TALU_METAL_FP8_LM_HEAD_QMM", has_mxfp8_meta || has_grouped_affine_meta || has_nvfp4_meta);
        if (enable_lm_head_qmm) {
            if (ctx->cfg.tie_word_embeddings && ctx->embed_tokens_has_q) {
                // Reuse the captured quantized embedding payload for tied lm_head.
                ctx->lm_head_q_w = ctx->embed_tokens_q_w;
                ctx->lm_head_q_scales = ctx->embed_tokens_q_scales;
                ctx->lm_head_q_biases = ctx->embed_tokens_q_biases;
                ctx->lm_head_q_decode_enabled = true;
                ctx->lm_head_q_bits = ctx->embed_tokens_q_bits;
            } else {
                if (lm_head_weight.size() == 1) {
                    // For grouped-affine: pass raw packed tensor directly — QMM capture reads
                    // scales/biases from the tensors map and only needs the shape from lhs_weight.
                    // For other formats (FP8 etc.) that need a float input, dequantize now.
                    if (has_grouped_affine_metadata_for_key(tensors, lm_head_key)) {
                        lm_head_weight = require_tensor(tensors, lm_head_key);
                    } else {
                        lm_head_weight = load_linear_weight(tensors, ctx->cfg, lm_head_key);
                    }
                }
                bool lm_head_has_q = false;
                int lm_head_detected_bits = 0;
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    lm_head_key,
                    lm_head_weight,
                    true,
                    &ctx->lm_head_q_w,
                    &ctx->lm_head_q_scales,
                    &ctx->lm_head_q_biases,
                    &lm_head_has_q,
                    &lm_head_detected_bits
                );
                ctx->lm_head_q_decode_enabled = lm_head_has_q;
                ctx->lm_head_q_bits = lm_head_detected_bits > 0 ? lm_head_detected_bits : ctx->cfg.quant_bits;
            }
        }

        // Keep dense lm_head RHS only when no quantized matmul format is
        // available.  Grouped-affine and NVFP4 checkpoints support QMM for both
        // prefill and decode, so the dense transpose is unnecessary.
        if (keep_dense_rhs_for_prefill || !ctx->lm_head_q_decode_enabled ||
            (!ctx->has_fp8_meta && !ctx->has_grouped_affine_meta && !has_nvfp4_meta)) {
            // Need dense weight: dequantize now if not already in float form.
            const bool weight_is_float = lm_head_weight.dtype() == bfloat16 ||
                lm_head_weight.dtype() == float16 || lm_head_weight.dtype() == float32;
            if (!weight_is_float) {
                lm_head_weight = load_linear_weight(tensors, ctx->cfg, lm_head_key);
            }
            {
                array rhs = to_rhs(lm_head_weight, lm_head_key);
                if (ctx->cfg.hidden_size_per_layer_input > 0 && !ctx->lm_head_q_decode_enabled) {
                    rhs = stop_gradient(copy(astype(rhs, float32)));
                }
                ctx->lm_head_rhs = rhs;
            }
        }

        array per_layer_model_projection_weight = array(0.0f);
        if (ctx->cfg.hidden_size_per_layer_input > 0) {
            const int hpl = ctx->cfg.hidden_size_per_layer_input;
            const std::string per_layer_embed_key = has_tensor(tensors, text_prefix + "embed_tokens_per_layer.weight")
                ? text_prefix + "embed_tokens_per_layer.weight"
                : "model.embed_tokens_per_layer.weight";
            const std::string per_layer_model_proj_key = has_tensor(tensors, text_prefix + "per_layer_model_projection.weight")
                ? text_prefix + "per_layer_model_projection.weight"
                : "model.per_layer_model_projection.weight";
            const std::string per_layer_proj_norm_key = has_tensor(tensors, text_prefix + "per_layer_projection_norm.weight")
                ? text_prefix + "per_layer_projection_norm.weight"
                : "model.per_layer_projection_norm.weight";

            if (has_tensor(tensors, per_layer_embed_key) &&
                has_tensor(tensors, per_layer_model_proj_key) &&
                has_tensor(tensors, per_layer_proj_norm_key))
            {
                const bool disable_per_layer_input =
                    env_truthy("TALU_METAL_DISABLE_PER_LAYER_INPUT", false);
                if (!disable_per_layer_input) {
                    ctx->per_layer_embedding = load_linear_weight(tensors, ctx->cfg, per_layer_embed_key);
                    per_layer_model_projection_weight = load_linear_weight(tensors, ctx->cfg, per_layer_model_proj_key);
                    ctx->per_layer_projection_norm_w = require_tensor(tensors, per_layer_proj_norm_key);
                    ctx->per_layer_hidden_size = hpl;
                    ctx->per_layer_embed_scale = std::sqrt(static_cast<float>(hpl));
                    ctx->per_layer_model_projection_scale = 1.0f / std::sqrt(static_cast<float>(ctx->cfg.hidden_size));
                    ctx->per_layer_input_enabled = true;
                }
            }
        }

        ctx->layers.reserve(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->kv_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        ctx->linear_cache.resize(static_cast<size_t>(ctx->cfg.num_hidden_layers));
        const std::vector<int> shared_kv_sources = resolve_shared_kv_source_layers(ctx->cfg);

        for (int i = 0; i < ctx->cfg.num_hidden_layers; ++i) {
            LayerWeights lw;
            if (i >= 0 && i < static_cast<int>(shared_kv_sources.size())) {
                lw.kv_shared_source_layer = shared_kv_sources[static_cast<size_t>(i)];
            }
            const std::string p = layer_prefix + std::to_string(i) + ".";

            lw.input_layernorm_w = require_any_tensor(
                tensors,
                {
                    p + "input_layernorm.weight",
                    p + "operator_norm.weight",
                },
                p + "input_layernorm/operator_norm"
            );
            lw.post_attention_layernorm_w = require_any_tensor(
                tensors,
                {
                    p + "post_attention_layernorm.weight",
                    p + "ffn_norm.weight",
                },
                p + "post_attention_layernorm/ffn_norm"
            );
            if (has_tensor(tensors, p + "pre_feedforward_layernorm.weight")) {
                lw.pre_feedforward_layernorm_w = require_tensor(tensors, p + "pre_feedforward_layernorm.weight");
            }
            if (has_tensor(tensors, p + "post_feedforward_layernorm.weight")) {
                lw.post_feedforward_layernorm_w = require_tensor(tensors, p + "post_feedforward_layernorm.weight");
            }
            if (ctx->per_layer_input_enabled) {
                const int hpl = ctx->per_layer_hidden_size;
                const int row_start = i * hpl;
                const int row_end = row_start + hpl;
                if (per_layer_model_projection_weight.ndim() == 2 &&
                    per_layer_model_projection_weight.shape(0) >= row_end &&
                    per_layer_model_projection_weight.shape(1) == ctx->cfg.hidden_size &&
                    has_tensor(tensors, p + "per_layer_input_gate.weight") &&
                    has_tensor(tensors, p + "per_layer_projection.weight") &&
                    has_tensor(tensors, p + "post_per_layer_input_norm.weight") &&
                    has_tensor(tensors, p + "layer_scalar"))
                {
                    const array per_layer_model_projection_slice = slice(
                        per_layer_model_projection_weight,
                        {row_start, 0},
                        {row_end, per_layer_model_projection_weight.shape(1)}
                    );
                    lw.per_layer_model_projection_rhs = to_rhs(
                        per_layer_model_projection_slice,
                        p + "per_layer_model_projection.weight[slice]"
                    );
                    lw.per_layer_input_gate_rhs = load_rhs_weight(p + "per_layer_input_gate.weight");
                    lw.per_layer_projection_rhs = load_rhs_weight(p + "per_layer_projection.weight");
                    lw.post_per_layer_input_norm_w = require_tensor(tensors, p + "post_per_layer_input_norm.weight");
                    lw.layer_scalar = scalar_to_f32(require_tensor(tensors, p + "layer_scalar"), p + "layer_scalar");
                    lw.has_per_layer_input = true;
                }
            }

            const bool has_moe_experts =
                has_tensor(tensors, p + "mlp.experts.down_proj") &&
                (has_tensor(tensors, p + "mlp.experts.gate_up_proj") ||
                 (has_tensor(tensors, p + "mlp.experts.gate_proj") && has_tensor(tensors, p + "mlp.experts.up_proj")));

            if (has_moe_experts) {
                if (ctx->cfg.num_experts <= 0 || ctx->cfg.num_experts_per_tok <= 0) {
                    throw std::runtime_error("missing MoE routing config fields for layer: " + std::to_string(i));
                }
                if (ctx->cfg.quant_group_size <= 0 || (ctx->cfg.quant_bits != 4 && ctx->cfg.quant_bits != 8)) {
                    throw std::runtime_error("Metal MLX MoE bridge requires grouped-affine expert quantization metadata");
                }

                lw.mlp_is_moe = true;
                ctx->has_moe_layers = true;
                lw.moe_num_experts = ctx->cfg.num_experts;
                lw.moe_experts_per_token = ctx->cfg.num_experts_per_tok;
                lw.moe_router_group_size = ctx->cfg.quant_group_size;
                lw.moe_expert_group_size = ctx->cfg.quant_group_size;
                lw.moe_router_w = load_linear_weight(tensors, ctx->cfg, p + "mlp.gate.weight");

                if (has_tensor(tensors, p + "mlp.experts.gate_up_proj")) {
                    lw.moe_gate_up_w = require_tensor(tensors, p + "mlp.experts.gate_up_proj");
                    lw.moe_gate_up_scales = require_tensor(tensors, p + "mlp.experts.gate_up_proj.scales");
                    if (has_tensor(tensors, p + "mlp.experts.gate_up_proj.biases")) {
                        lw.moe_gate_up_biases = require_tensor(tensors, p + "mlp.experts.gate_up_proj.biases");
                    }
                    const auto gate_up_split = split_fused_moe_gate_up_projection(
                        lw.moe_gate_up_w,
                        p + "mlp.experts.gate_up_proj"
                    );
                    const auto gate_up_scales_split = split_fused_moe_gate_up_projection(
                        lw.moe_gate_up_scales,
                        p + "mlp.experts.gate_up_proj.scales"
                    );
                    lw.moe_gate_w = gate_up_split.first;
                    lw.moe_up_w = gate_up_split.second;
                    lw.moe_gate_scales = gate_up_scales_split.first;
                    lw.moe_up_scales = gate_up_scales_split.second;
                    if (lw.moe_gate_up_biases.size() > 1) {
                        const auto gate_up_biases_split = split_fused_moe_gate_up_projection(
                            lw.moe_gate_up_biases,
                            p + "mlp.experts.gate_up_proj.biases"
                        );
                        lw.moe_gate_biases = gate_up_biases_split.first;
                        lw.moe_up_biases = gate_up_biases_split.second;
                    }
                } else {
                    lw.moe_gate_w = require_tensor(tensors, p + "mlp.experts.gate_proj");
                    lw.moe_up_w = require_tensor(tensors, p + "mlp.experts.up_proj");
                    lw.moe_gate_scales = require_tensor(tensors, p + "mlp.experts.gate_proj.scales");
                    lw.moe_up_scales = require_tensor(tensors, p + "mlp.experts.up_proj.scales");
                    if (has_tensor(tensors, p + "mlp.experts.gate_proj.biases")) {
                        lw.moe_gate_biases = require_tensor(tensors, p + "mlp.experts.gate_proj.biases");
                    }
                    if (has_tensor(tensors, p + "mlp.experts.up_proj.biases")) {
                        lw.moe_up_biases = require_tensor(tensors, p + "mlp.experts.up_proj.biases");
                    }
                }

                lw.moe_down_w = require_tensor(tensors, p + "mlp.experts.down_proj");
                lw.moe_down_scales = require_tensor(tensors, p + "mlp.experts.down_proj.scales");
                if (has_tensor(tensors, p + "mlp.experts.down_proj.biases")) {
                    lw.moe_down_biases = require_tensor(tensors, p + "mlp.experts.down_proj.biases");
                }

                if (has_tensor(tensors, p + "mlp.shared_expert.gate_proj.weight") &&
                    has_tensor(tensors, p + "mlp.shared_expert.up_proj.weight") &&
                    has_tensor(tensors, p + "mlp.shared_expert.down_proj.weight"))
                {
                    const std::string shared_gate_key = p + "mlp.shared_expert.gate_proj.weight";
                    const std::string shared_up_key = p + "mlp.shared_expert.up_proj.weight";
                    const std::string shared_down_key = p + "mlp.shared_expert.down_proj.weight";

                    const array& shared_gate_weight = require_tensor(tensors, shared_gate_key);
                    const array& shared_up_weight = require_tensor(tensors, shared_up_key);
                    const array& shared_down_weight = require_tensor(tensors, shared_down_key);

                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        ctx->cfg,
                        shared_gate_key,
                        shared_gate_weight,
                        enable_mlp_qmm,
                        &lw.moe_shared_gate_q_w,
                        &lw.moe_shared_gate_q_scales,
                        &lw.moe_shared_gate_q_biases,
                        &lw.moe_shared_gate_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        ctx->cfg,
                        shared_up_key,
                        shared_up_weight,
                        enable_mlp_qmm,
                        &lw.moe_shared_up_q_w,
                        &lw.moe_shared_up_q_scales,
                        &lw.moe_shared_up_q_biases,
                        &lw.moe_shared_up_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors,
                        ctx->cfg,
                        shared_down_key,
                        shared_down_weight,
                        enable_mlp_qmm,
                        &lw.moe_shared_down_q_w,
                        &lw.moe_shared_down_q_scales,
                        &lw.moe_shared_down_q_biases,
                        &lw.moe_shared_down_has_q
                    );
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.moe_shared_gate_has_q) {
                        lw.moe_shared_gate_rhs = load_rhs_weight(shared_gate_key);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.moe_shared_up_has_q) {
                        lw.moe_shared_up_rhs = load_rhs_weight(shared_up_key);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.moe_shared_down_has_q) {
                        lw.moe_shared_down_rhs = load_rhs_weight(shared_down_key);
                    }
                    combine_shared_expert_gate_up_weights(&lw);
                }

                if (has_tensor(tensors, p + "mlp.shared_expert_gate.weight")) {
                    array shared_gate_weight = require_tensor(tensors, p + "mlp.shared_expert_gate.weight");
                    if (shared_gate_weight.ndim() == 2 && shared_gate_weight.shape(0) == 1) {
                        shared_gate_weight = reshape(shared_gate_weight, {shared_gate_weight.shape(1)});
                    }
                    if (shared_gate_weight.ndim() != 1) {
                        throw std::runtime_error("shared expert gate must be rank-1 after reshape for layer: " + std::to_string(i));
                    }
                    lw.moe_shared_expert_gate_w = shared_gate_weight;
                }
            } else {
                const std::string mlp_down_key = has_tensor(tensors, p + "mlp.down_proj.weight")
                    ? p + "mlp.down_proj.weight"
                    : p + "feed_forward.w2.weight";
                const array& mlp_down_weight = require_tensor(tensors, mlp_down_key);

                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    mlp_down_key,
                    mlp_down_weight,
                    enable_mlp_qmm,
                    &lw.mlp_down_q_w,
                    &lw.mlp_down_q_scales,
                    &lw.mlp_down_q_biases,
                    &lw.mlp_down_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_down_has_q) {
                    lw.mlp_down_rhs = load_rhs_weight(mlp_down_key);
                }

                // Prefer fused gate_up_proj if the checkpoint has it (produced by
                // talu convert). This avoids a concatenate+copy at runtime that
                // duplicates weight data already in the mmap.
                const std::string gate_up_key = p + "mlp.gate_up_proj.weight";
                if (has_tensor(tensors, gate_up_key) &&
                    has_grouped_affine_metadata_for_key(tensors, gate_up_key))
                {
                    const array& gate_up_weight = require_tensor(tensors, gate_up_key);
                    if (gate_up_weight.ndim() == 2 && gate_up_weight.shape(0) > 0) {
                        lw.mlp_gate_width = gate_up_weight.shape(0) / 2;
                        maybe_capture_grouped_affine_quant(
                            tensors, ctx->cfg, gate_up_key, gate_up_weight,
                            &lw.mlp_gate_up_q_w,
                            &lw.mlp_gate_up_q_scales,
                            &lw.mlp_gate_up_q_biases
                        );
                        lw.mlp_gate_up_has_q = lw.mlp_gate_up_q_w.ndim() == 2;
                        if (lw.mlp_gate_up_has_q) {
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_gate_up_has_q) {
                            lw.mlp_gate_up_rhs = load_rhs_weight(gate_up_key);
                        }
                    }
                }

                // Fallback: separate gate + up tensors, combined at runtime.
                if (!lw.mlp_gate_up_has_q && lw.mlp_gate_up_rhs.size() <= 1) {
                    const std::string mlp_gate_key = has_tensor(tensors, p + "mlp.gate_proj.weight")
                        ? p + "mlp.gate_proj.weight"
                        : p + "feed_forward.w1.weight";
                    const std::string mlp_up_key = has_tensor(tensors, p + "mlp.up_proj.weight")
                        ? p + "mlp.up_proj.weight"
                        : p + "feed_forward.w3.weight";
                    const array& mlp_gate_weight = require_tensor(tensors, mlp_gate_key);
                    const array& mlp_up_weight = require_tensor(tensors, mlp_up_key);
                    maybe_quantize_mxfp8_matrix(
                        tensors, ctx->cfg, mlp_gate_key, mlp_gate_weight, enable_mlp_qmm,
                        &lw.mlp_gate_q_w, &lw.mlp_gate_q_scales, &lw.mlp_gate_q_biases, &lw.mlp_gate_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors, ctx->cfg, mlp_up_key, mlp_up_weight, enable_mlp_qmm,
                        &lw.mlp_up_q_w, &lw.mlp_up_q_scales, &lw.mlp_up_q_biases, &lw.mlp_up_has_q
                    );
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_gate_has_q) {
                        lw.mlp_gate_rhs = load_rhs_weight(mlp_gate_key);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.mlp_up_has_q) {
                        lw.mlp_up_rhs = load_rhs_weight(mlp_up_key);
                    }
                    combine_dense_mlp_gate_up_weights(&lw);
                    compact_dense_mlp_gate_up_weights(&lw);
                }
            }

            const bool has_linear_split = has_tensor(tensors, p + "linear_attn.in_proj_qkv.weight");
            const bool has_linear_fused = has_tensor(tensors, p + "mixer.in_proj.weight");
            const bool has_linear = has_linear_split || has_linear_fused;
            const bool has_shortconv = has_tensor(tensors, p + "conv.in_proj.weight");
            const bool has_attn = has_tensor(tensors, p + "self_attn.q_proj.weight")
                || has_tensor(tensors, p + "self_attn.qkv_proj.weight");
            const int branch_count = (has_linear ? 1 : 0) + (has_shortconv ? 1 : 0) + (has_attn ? 1 : 0);
            if (branch_count == 0) {
                std::string present;
                for (const auto& kv : tensors) {
                    if (kv.first.rfind(p, 0) != 0) continue;
                    if (kv.first.find("linear_attn") == std::string::npos &&
                        kv.first.find("mixer") == std::string::npos &&
                        kv.first.find("conv") == std::string::npos &&
                        kv.first.find("self_attn") == std::string::npos) {
                        continue;
                    }
                    if (!present.empty()) present += ", ";
                    present += kv.first;
                    if (present.size() > 1200) {
                        present += ", ...";
                        break;
                    }
                }
                throw std::runtime_error(
                    "layer has neither linear_attn, conv, nor self_attn: " + std::to_string(i) +
                    " (prefix=" + p +
                    ", has_linear_split=" + std::to_string(has_linear_split) +
                    ", has_linear_fused=" + std::to_string(has_linear_fused) +
                    ", has_shortconv=" + std::to_string(has_shortconv) +
                    ", has_attn=" + std::to_string(has_attn) +
                    ", present=[" + present + "])"
                );
            }
            if (branch_count > 1) {
                throw std::runtime_error("layer has multiple mixer branches unexpectedly: " + std::to_string(i));
            }

            lw.is_linear = has_linear;
            lw.is_shortconv = has_shortconv;

            if (lw.is_linear) {
                if (ctx->cfg.linear_num_value_heads <= 0 ||
                    ctx->cfg.linear_num_key_heads <= 0 ||
                    ctx->cfg.linear_key_head_dim <= 0 ||
                    ctx->cfg.linear_value_head_dim <= 0 ||
                    ctx->cfg.linear_conv_kernel_dim <= 0) {
                    throw std::runtime_error("missing linear_attn config fields for hybrid layer");
                }
                lw.lin_num_v_heads = ctx->cfg.linear_num_value_heads;
                lw.lin_num_k_heads = ctx->cfg.linear_num_key_heads;
                lw.lin_head_k_dim = ctx->cfg.linear_key_head_dim;
                lw.lin_head_v_dim = ctx->cfg.linear_value_head_dim;
                lw.lin_key_dim = lw.lin_num_k_heads * lw.lin_head_k_dim;
                lw.lin_value_dim = lw.lin_num_v_heads * lw.lin_head_v_dim;
                lw.lin_conv_dim = lw.lin_key_dim * 2 + lw.lin_value_dim;
                lw.lin_conv_kernel = ctx->cfg.linear_conv_kernel_dim;

                lw.lin_conv1d_w = require_tensor(tensors, p + "linear_attn.conv1d.weight");
                lw.lin_conv1d_decode_w = prepare_depthwise_conv1d_decode_weight(
                    lw.lin_conv1d_w,
                    lw.lin_conv_dim,
                    lw.lin_conv_kernel,
                    p + "linear_attn.conv1d.weight"
                );
                if (has_linear_split) {
                    const std::string qkv_key = p + "linear_attn.in_proj_qkv.weight";
                    const std::string z_key = p + "linear_attn.in_proj_z.weight";
                    const std::string a_key = p + "linear_attn.in_proj_a.weight";
                    const std::string b_key = p + "linear_attn.in_proj_b.weight";
                    const array qkv_weight_src = require_tensor(tensors, qkv_key);
                    const array z_weight_src = require_tensor(tensors, z_key);
                    const array a_weight_src = require_tensor(tensors, a_key);
                    const array b_weight_src = require_tensor(tensors, b_key);
                    bool lin_qkvz_q_captured = false;
                    bool lin_ba_q_captured = false;
                    if (enable_mlp_qmm && env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true)) {
                        // Split linear-attn checkpoints store grouped-affine metadata
                        // per source tensor. Capture each source tensor losslessly
                        // and concatenate packed payload + calibration metadata.
                        array qkv_q_w = array(0.0f);
                        array qkv_q_scales = array(0.0f);
                        array qkv_q_biases = array(0.0f);
                        array z_q_w = array(0.0f);
                        array z_q_scales = array(0.0f);
                        array z_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                qkv_key,
                                qkv_weight_src,
                                &qkv_q_w,
                                &qkv_q_scales,
                                &qkv_q_biases) &&
                            maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                z_key,
                                z_weight_src,
                                &z_q_w,
                                &z_q_scales,
                                &z_q_biases) &&
                            qkv_q_w.ndim() == 2 && z_q_w.ndim() == 2 &&
                            qkv_q_w.shape(1) == z_q_w.shape(1) &&
                            qkv_q_scales.ndim() == 2 && z_q_scales.ndim() == 2 &&
                            qkv_q_scales.shape(1) == z_q_scales.shape(1) &&
                            qkv_q_biases.ndim() == 2 && z_q_biases.ndim() == 2 &&
                            qkv_q_biases.shape(1) == z_q_biases.shape(1))
                        {
                            lw.lin_in_proj_qkvz_q_w = concatenate({qkv_q_w, z_q_w}, 0);
                            lw.lin_in_proj_qkvz_q_scales = concatenate({qkv_q_scales, z_q_scales}, 0);
                            lw.lin_in_proj_qkvz_q_biases = concatenate({qkv_q_biases, z_q_biases}, 0);
                            lw.lin_in_proj_qkvz_has_q = true;
                            lin_qkvz_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }

                        array b_q_w = array(0.0f);
                        array b_q_scales = array(0.0f);
                        array b_q_biases = array(0.0f);
                        array a_q_w = array(0.0f);
                        array a_q_scales = array(0.0f);
                        array a_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                b_key,
                                b_weight_src,
                                &b_q_w,
                                &b_q_scales,
                                &b_q_biases) &&
                            maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                a_key,
                                a_weight_src,
                                &a_q_w,
                                &a_q_scales,
                                &a_q_biases) &&
                            b_q_w.ndim() == 2 && a_q_w.ndim() == 2 &&
                            b_q_w.shape(1) == a_q_w.shape(1) &&
                            b_q_scales.ndim() == 2 && a_q_scales.ndim() == 2 &&
                            b_q_scales.shape(1) == a_q_scales.shape(1) &&
                            b_q_biases.ndim() == 2 && a_q_biases.ndim() == 2 &&
                            b_q_biases.shape(1) == a_q_biases.shape(1))
                        {
                            lw.lin_in_proj_ba_q_w = concatenate({b_q_w, a_q_w}, 0);
                            lw.lin_in_proj_ba_q_scales = concatenate({b_q_scales, a_q_scales}, 0);
                            lw.lin_in_proj_ba_q_biases = concatenate({b_q_biases, a_q_biases}, 0);
                            lw.lin_in_proj_ba_has_q = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    if (!lin_qkvz_q_captured && env_truthy("TALU_METAL_NVFP4_AFFINE_TRANSCODE", false)) {
                        array qkv_q_w = array(0.0f);
                        array qkv_q_scales = array(0.0f);
                        array qkv_q_biases = array(0.0f);
                        array z_q_w = array(0.0f);
                        array z_q_scales = array(0.0f);
                        array z_q_biases = array(0.0f);
                        if (transcode_nvfp4_to_affine_qmm(
                                tensors,
                                qkv_key,
                                qkv_weight_src,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &qkv_q_w,
                                &qkv_q_scales,
                                &qkv_q_biases) &&
                            transcode_nvfp4_to_affine_qmm(
                                tensors,
                                z_key,
                                z_weight_src,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &z_q_w,
                                &z_q_scales,
                                &z_q_biases) &&
                            qkv_q_w.ndim() == 2 && z_q_w.ndim() == 2 &&
                            qkv_q_w.shape(1) == z_q_w.shape(1) &&
                            qkv_q_scales.ndim() == 2 && z_q_scales.ndim() == 2 &&
                            qkv_q_scales.shape(1) == z_q_scales.shape(1) &&
                            qkv_q_biases.ndim() == 2 && z_q_biases.ndim() == 2 &&
                            qkv_q_biases.shape(1) == z_q_biases.shape(1))
                        {
                            lw.lin_in_proj_qkvz_q_w = concatenate({qkv_q_w, z_q_w}, 0);
                            lw.lin_in_proj_qkvz_q_scales = concatenate({qkv_q_scales, z_q_scales}, 0);
                            lw.lin_in_proj_qkvz_q_biases = concatenate({qkv_q_biases, z_q_biases}, 0);
                            lw.lin_in_proj_qkvz_has_q = true;
                            lin_qkvz_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    if (!lin_ba_q_captured && env_truthy("TALU_METAL_NVFP4_AFFINE_TRANSCODE", false)) {
                        array b_q_w = array(0.0f);
                        array b_q_scales = array(0.0f);
                        array b_q_biases = array(0.0f);
                        array a_q_w = array(0.0f);
                        array a_q_scales = array(0.0f);
                        array a_q_biases = array(0.0f);
                        if (transcode_nvfp4_to_affine_qmm(
                                tensors,
                                b_key,
                                b_weight_src,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &b_q_w,
                                &b_q_scales,
                                &b_q_biases) &&
                            transcode_nvfp4_to_affine_qmm(
                                tensors,
                                a_key,
                                a_weight_src,
                                ctx->cfg.quant_group_size > 0 ? ctx->cfg.quant_group_size : 32,
                                &a_q_w,
                                &a_q_scales,
                                &a_q_biases) &&
                            b_q_w.ndim() == 2 && a_q_w.ndim() == 2 &&
                            b_q_w.shape(1) == a_q_w.shape(1) &&
                            b_q_scales.ndim() == 2 && a_q_scales.ndim() == 2 &&
                            b_q_scales.shape(1) == a_q_scales.shape(1) &&
                            b_q_biases.ndim() == 2 && a_q_biases.ndim() == 2 &&
                            b_q_biases.shape(1) == a_q_biases.shape(1))
                        {
                            lw.lin_in_proj_ba_q_w = concatenate({b_q_w, a_q_w}, 0);
                            lw.lin_in_proj_ba_q_scales = concatenate({b_q_scales, a_q_scales}, 0);
                            lw.lin_in_proj_ba_q_biases = concatenate({b_q_biases, a_q_biases}, 0);
                            lw.lin_in_proj_ba_has_q = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                    }
                    std::optional<array> qkv_weight_dense;
                    std::optional<array> z_weight_dense;
                    std::optional<array> a_weight_dense;
                    std::optional<array> b_weight_dense;
                    const auto ensure_qkv_weight_dense = [&]() -> const array& {
                        if (!qkv_weight_dense.has_value()) qkv_weight_dense = load_linear_weight(tensors, ctx->cfg, qkv_key);
                        return *qkv_weight_dense;
                    };
                    const auto ensure_z_weight_dense = [&]() -> const array& {
                        if (!z_weight_dense.has_value()) z_weight_dense = load_linear_weight(tensors, ctx->cfg, z_key);
                        return *z_weight_dense;
                    };
                    const auto ensure_a_weight_dense = [&]() -> const array& {
                        if (!a_weight_dense.has_value()) a_weight_dense = load_linear_weight(tensors, ctx->cfg, a_key);
                        return *a_weight_dense;
                    };
                    const auto ensure_b_weight_dense = [&]() -> const array& {
                        if (!b_weight_dense.has_value()) b_weight_dense = load_linear_weight(tensors, ctx->cfg, b_key);
                        return *b_weight_dense;
                    };
                    if (!lin_qkvz_q_captured) {
                        const array qkvz_weight = concatenate({ensure_qkv_weight_dense(), ensure_z_weight_dense()}, 0);
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "linear_attn.in_proj_qkvz",
                            qkvz_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_qkvz_q_w,
                            &lw.lin_in_proj_qkvz_q_scales,
                            &lw.lin_in_proj_qkvz_q_biases,
                            &lw.lin_in_proj_qkvz_has_q
                        );
                    }
                    if (!lin_ba_q_captured) {
                        const array ba_weight = concatenate({ensure_b_weight_dense(), ensure_a_weight_dense()}, 0);
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "linear_attn.in_proj_ba",
                            ba_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_ba_q_w,
                            &lw.lin_in_proj_ba_q_scales,
                            &lw.lin_in_proj_ba_q_biases,
                            &lw.lin_in_proj_ba_has_q
                        );
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) {
                        const array qkv_rhs = to_rhs(ensure_qkv_weight_dense(), qkv_key);
                        const array z_rhs = to_rhs(ensure_z_weight_dense(), z_key);
                        lw.lin_in_proj_qkvz_rhs = concatenate({ qkv_rhs, z_rhs }, 1);
                    } else if (enable_linear_prefill_dense_cache) {
                        lw.lin_in_proj_qkvz_prefill_rhs = maybe_build_prefill_rhs_cache(
                            p + "linear_attn.in_proj_qkvz",
                            lw.lin_in_proj_qkvz_q_w,
                            lw.lin_in_proj_qkvz_q_scales,
                            lw.lin_in_proj_qkvz_q_biases
                        );
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q) {
                        const array b_rhs = to_rhs(ensure_b_weight_dense(), b_key);
                        const array a_rhs = to_rhs(ensure_a_weight_dense(), a_key);
                        lw.lin_in_proj_ba_rhs = concatenate({ b_rhs, a_rhs }, 1);
                    }
                    // Always combine qkvz + ba into a single projection
                    // tensor so the reference path can use one matmul.
                    combine_linear_in_proj_weights(&lw);
                    compact_linear_in_proj_weights(&lw);
                } else {
                    const std::string fused_key = p + "mixer.in_proj.weight";
                    const array fused_in_proj_weight_src = require_tensor(tensors, fused_key);
                    if (fused_in_proj_weight_src.ndim() != 2) {
                        throw std::runtime_error("linear fused in_proj must be rank-2");
                    }
                    const int qkvz_cols = lw.lin_conv_dim + lw.lin_value_dim;
                    const int ba_cols = 2 * lw.lin_num_v_heads;
                    const int expected_cols = qkvz_cols + ba_cols;
                    const int fused_cols = fused_in_proj_weight_src.shape(0);
                    if (fused_cols != expected_cols) {
                        throw std::runtime_error(
                            "linear fused in_proj shape mismatch: expected cols=" +
                            std::to_string(expected_cols) + " got cols=" + std::to_string(fused_cols)
                        );
                    }
                    bool lin_qkvz_q_captured = false;
                    bool lin_ba_q_captured = false;
                    if (enable_mlp_qmm && env_truthy("TALU_METAL_GAFFINE_LOSSLESS", true)) {
                        // Fused linear-attn stores grouped-affine metadata for
                        // mixer.in_proj as one tensor. Slice packed payload +
                        // metadata rows losslessly into qkvz and ba branches.
                        array fused_q_w = array(0.0f);
                        array fused_q_scales = array(0.0f);
                        array fused_q_biases = array(0.0f);
                        if (maybe_capture_grouped_affine_quant(
                                tensors,
                                ctx->cfg,
                                fused_key,
                                fused_in_proj_weight_src,
                                &fused_q_w,
                                &fused_q_scales,
                                &fused_q_biases) &&
                            fused_q_w.ndim() == 2 &&
                            fused_q_scales.ndim() == 2 &&
                            fused_q_biases.ndim() == 2 &&
                            fused_q_w.shape(0) == expected_cols &&
                            fused_q_scales.shape(0) == expected_cols &&
                            fused_q_biases.shape(0) == expected_cols)
                        {
                            const int packed_cols = fused_q_w.shape(1);
                            const int group_cols = fused_q_scales.shape(1);
                            lw.lin_in_proj_qkvz_q_w = slice(fused_q_w, {0, 0}, {qkvz_cols, packed_cols});
                            lw.lin_in_proj_ba_q_w = slice(fused_q_w, {qkvz_cols, 0}, {expected_cols, packed_cols});
                            lw.lin_in_proj_qkvz_q_scales = slice(fused_q_scales, {0, 0}, {qkvz_cols, group_cols});
                            lw.lin_in_proj_ba_q_scales = slice(fused_q_scales, {qkvz_cols, 0}, {expected_cols, group_cols});
                            lw.lin_in_proj_qkvz_q_biases = slice(fused_q_biases, {0, 0}, {qkvz_cols, group_cols});
                            lw.lin_in_proj_ba_q_biases = slice(fused_q_biases, {qkvz_cols, 0}, {expected_cols, group_cols});
                            lw.lin_in_proj_qkvz_has_q = true;
                            lw.lin_in_proj_ba_has_q = true;
                            lw.lin_in_proj_all_q_w = fused_q_w;
                            lw.lin_in_proj_all_q_scales = fused_q_scales;
                            lw.lin_in_proj_all_q_biases = fused_q_biases;
                            lw.lin_in_proj_all_has_q = true;
                            lin_qkvz_q_captured = true;
                            lin_ba_q_captured = true;
                            g_qmm_attempts += 2;
                            g_qmm_success += 2;
                        }
                    }
                    std::optional<array> fused_in_proj_weight_dense;
                    const auto ensure_fused_in_proj_weight_dense = [&]() -> const array& {
                        if (!fused_in_proj_weight_dense.has_value()) {
                            fused_in_proj_weight_dense = load_linear_weight(tensors, ctx->cfg, fused_key);
                        }
                        return *fused_in_proj_weight_dense;
                    };
                    if (!lin_qkvz_q_captured) {
                        const array& fused_in_proj_weight = ensure_fused_in_proj_weight_dense();
                        const array qkvz_weight = slice(
                            fused_in_proj_weight,
                            {0, 0},
                            {qkvz_cols, fused_in_proj_weight.shape(1)}
                        );
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "mixer.in_proj.qkvz_slice",
                            qkvz_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_qkvz_q_w,
                            &lw.lin_in_proj_qkvz_q_scales,
                            &lw.lin_in_proj_qkvz_q_biases,
                            &lw.lin_in_proj_qkvz_has_q
                        );
                    }
                    if (!lin_ba_q_captured) {
                        const array& fused_in_proj_weight = ensure_fused_in_proj_weight_dense();
                        const array ba_weight = slice(
                            fused_in_proj_weight,
                            {qkvz_cols, 0},
                            {expected_cols, fused_in_proj_weight.shape(1)}
                        );
                        maybe_quantize_mxfp8_matrix(
                            tensors,
                            ctx->cfg,
                            p + "mixer.in_proj.ba_slice",
                            ba_weight,
                            enable_mlp_qmm,
                            &lw.lin_in_proj_ba_q_w,
                            &lw.lin_in_proj_ba_q_scales,
                            &lw.lin_in_proj_ba_q_biases,
                            &lw.lin_in_proj_ba_has_q
                        );
                    }
                    if ((keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) ||
                        (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q)) {
                        const array& fused_in_proj_weight = ensure_fused_in_proj_weight_dense();
                        const array fused_in_proj_rhs = to_rhs(
                            fused_in_proj_weight,
                            fused_key
                        );
                        const int fused_rows = fused_in_proj_rhs.shape(0);
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_qkvz_has_q) {
                            lw.lin_in_proj_qkvz_rhs = slice(
                                fused_in_proj_rhs,
                                {0, 0},
                                {fused_rows, qkvz_cols}
                            );
                        } else if (enable_linear_prefill_dense_cache) {
                            lw.lin_in_proj_qkvz_prefill_rhs = maybe_build_prefill_rhs_cache(
                                p + "mixer.in_proj.qkvz_slice",
                                lw.lin_in_proj_qkvz_q_w,
                                lw.lin_in_proj_qkvz_q_scales,
                                lw.lin_in_proj_qkvz_q_biases
                            );
                        }
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_in_proj_ba_has_q) {
                            lw.lin_in_proj_ba_rhs = slice(
                                fused_in_proj_rhs,
                                {0, qkvz_cols},
                                {fused_rows, expected_cols}
                            );
                        }
                    }
                    combine_linear_in_proj_weights(&lw);
                    compact_linear_in_proj_weights(&lw);
                }
                lw.lin_A_log = require_tensor(tensors, p + "linear_attn.A_log");
                lw.lin_A_exp_f32 = exp(astype(lw.lin_A_log, float32));
                lw.lin_dt_bias = require_tensor(tensors, p + "linear_attn.dt_bias");
                lw.lin_norm_w = require_tensor(tensors, p + "linear_attn.norm.weight");
                const std::string out_proj_key = p + "linear_attn.out_proj.weight";
                const array& out_proj_weight = require_tensor(tensors, out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    out_proj_key,
                    out_proj_weight,
                    enable_mlp_qmm,
                    &lw.lin_out_proj_q_w,
                    &lw.lin_out_proj_q_scales,
                    &lw.lin_out_proj_q_biases,
                    &lw.lin_out_proj_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.lin_out_proj_has_q) {
                    lw.lin_out_proj_rhs = load_rhs_weight(out_proj_key);
                }
            } else if (lw.is_shortconv) {
                const std::string sc_in_proj_key = p + "conv.in_proj.weight";
                const array& sc_in_proj_weight = require_tensor(tensors, sc_in_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    sc_in_proj_key,
                    sc_in_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_in_proj_q_w,
                    &lw.sc_in_proj_q_scales,
                    &lw.sc_in_proj_q_biases,
                    &lw.sc_in_proj_has_q
                );
                lw.sc_conv1d_w = require_tensor(tensors, p + "conv.conv.weight");
                const std::string sc_out_proj_key = p + "conv.out_proj.weight";
                const array& sc_out_proj_weight = require_tensor(tensors, sc_out_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors,
                    ctx->cfg,
                    sc_out_proj_key,
                    sc_out_proj_weight,
                    enable_mlp_qmm,
                    &lw.sc_out_proj_q_w,
                    &lw.sc_out_proj_q_scales,
                    &lw.sc_out_proj_q_biases,
                    &lw.sc_out_proj_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.sc_in_proj_has_q) {
                    lw.sc_in_proj_rhs = load_rhs_weight(sc_in_proj_key);
                }
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.sc_out_proj_has_q) {
                    lw.sc_out_proj_rhs = load_rhs_weight(sc_out_proj_key);
                }
                if (has_tensor(tensors, p + "conv.conv.bias")) {
                    lw.sc_conv_bias = require_tensor(tensors, p + "conv.conv.bias");
                    lw.sc_has_bias = true;
                }
                const int proj_dim = sc_in_proj_weight.shape(0);
                if (proj_dim % 3 != 0) {
                    throw std::runtime_error("shortconv in_proj output dim is not divisible by 3");
                }
                lw.sc_conv_dim = proj_dim / 3;
                lw.sc_conv_kernel = lw.sc_conv1d_w.shape(1);
            } else {
                const std::string o_proj_key = has_tensor(tensors, p + "self_attn.o_proj.weight")
                    ? p + "self_attn.o_proj.weight"
                    : p + "self_attn.out_proj.weight";
                const array& o_proj_weight = require_tensor(tensors, o_proj_key);
                maybe_quantize_mxfp8_matrix(
                    tensors, ctx->cfg, o_proj_key, o_proj_weight, enable_mlp_qmm,
                    &lw.attn_o_q_w, &lw.attn_o_q_scales, &lw.attn_o_q_biases, &lw.attn_o_has_q
                );
                if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_o_has_q) {
                    lw.attn_o_rhs = load_rhs_weight(o_proj_key);
                }

                // Set attention dimensions early so fused QKV path can compute proj widths.
                lw.attn_num_heads = ctx->cfg.num_attention_heads;
                lw.attn_num_kv_heads = ctx->cfg.num_key_value_heads > 0
                    ? ctx->cfg.num_key_value_heads
                    : ctx->cfg.num_attention_heads;
                lw.attn_head_dim = ctx->cfg.head_dim;

                // Prefer fused qkv_proj if the checkpoint has it (produced by talu convert).
                const std::string qkv_proj_key = p + "self_attn.qkv_proj.weight";
                if (has_tensor(tensors, qkv_proj_key) &&
                    has_grouped_affine_metadata_for_key(tensors, qkv_proj_key))
                {
                    const array& qkv_weight = require_tensor(tensors, qkv_proj_key);
                    if (qkv_weight.ndim() == 2) {
                        // Detect Q/K/V widths from config.
                        // Q proj width is the total output rows of the Q projection.
                        // For gated attention, Q carries both q and gate,
                        // so width = 2 * num_heads * head_dim. Detect from tensor shape.
                        lw.attn_q_proj_width = qkv_weight.shape(0)
                            - lw.attn_num_kv_heads * lw.attn_head_dim   // K
                            - lw.attn_num_kv_heads * lw.attn_head_dim;  // V
                        lw.attn_k_proj_width = lw.attn_num_kv_heads * lw.attn_head_dim;
                        lw.attn_v_proj_width = lw.attn_num_kv_heads * lw.attn_head_dim;
                        maybe_capture_grouped_affine_quant(
                            tensors, ctx->cfg, qkv_proj_key, qkv_weight,
                            &lw.attn_qkv_q_w, &lw.attn_qkv_q_scales, &lw.attn_qkv_q_biases
                        );
                        lw.attn_qkv_has_q = lw.attn_qkv_q_w.ndim() == 2;
                        if (lw.attn_qkv_has_q) {
                            g_qmm_attempts += 1;
                            g_qmm_success += 1;
                        }
                        if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_qkv_has_q) {
                            lw.attn_qkv_rhs = load_rhs_weight(qkv_proj_key);
                        }
                    }
                }

                // Fallback: separate Q/K/V tensors, combined at runtime.
                if (!lw.attn_qkv_has_q && lw.attn_qkv_rhs.size() <= 1) {
                    const std::string q_proj_key = p + "self_attn.q_proj.weight";
                    const std::string k_proj_key = p + "self_attn.k_proj.weight";
                    const std::string v_proj_key = p + "self_attn.v_proj.weight";
                    const array& q_proj_weight = require_tensor(tensors, q_proj_key);
                    const array& k_proj_weight = require_tensor(tensors, k_proj_key);
                    const array& v_proj_weight = require_tensor(tensors, v_proj_key);
                    maybe_quantize_mxfp8_matrix(
                        tensors, ctx->cfg, q_proj_key, q_proj_weight, enable_mlp_qmm,
                        &lw.attn_q_q_w, &lw.attn_q_q_scales, &lw.attn_q_q_biases, &lw.attn_q_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors, ctx->cfg, k_proj_key, k_proj_weight, enable_mlp_qmm,
                        &lw.attn_k_q_w, &lw.attn_k_q_scales, &lw.attn_k_q_biases, &lw.attn_k_has_q
                    );
                    maybe_quantize_mxfp8_matrix(
                        tensors, ctx->cfg, v_proj_key, v_proj_weight, enable_mlp_qmm,
                        &lw.attn_v_q_w, &lw.attn_v_q_scales, &lw.attn_v_q_biases, &lw.attn_v_has_q
                    );
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_q_has_q) {
                        lw.attn_q_rhs = load_rhs_weight(q_proj_key);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_k_has_q) {
                        lw.attn_k_rhs = load_rhs_weight(k_proj_key);
                    }
                    if (keep_dense_rhs_for_prefill || !enable_mlp_qmm || !lw.attn_v_has_q) {
                        lw.attn_v_rhs = load_rhs_weight(v_proj_key);
                    }
                    combine_full_attention_qkv_weights(&lw);
                    compact_full_attention_qkv_weights(&lw);
                }
                if (!keep_dense_rhs_for_prefill &&
                    enable_mlp_qmm &&
                    enable_full_attn_prefill_dense_cache &&
                    lw.attn_qkv_has_q)
                {
                    lw.attn_qkv_prefill_rhs = maybe_build_prefill_rhs_cache(
                        p + "self_attn.qkv",
                        lw.attn_qkv_q_w,
                        lw.attn_qkv_q_scales,
                        lw.attn_qkv_q_biases
                    );
                }
                lw.attn_q_norm_w = require_any_tensor(
                    tensors,
                    { p + "self_attn.q_norm.weight", p + "self_attn.q_layernorm.weight" },
                    p + "self_attn.q_norm/q_layernorm"
                );
                lw.attn_k_norm_w = require_any_tensor(
                    tensors,
                    { p + "self_attn.k_norm.weight", p + "self_attn.k_layernorm.weight" },
                    p + "self_attn.k_norm/k_layernorm"
                );

                lw.attn_num_heads = ctx->cfg.num_attention_heads;
                if (lw.attn_num_heads <= 0) {
                    throw std::runtime_error("invalid num_attention_heads in config");
                }

                lw.attn_head_dim = ctx->cfg.head_dim;
                const int q_proj_out_dim = lw.attn_q_proj_width > 0
                    ? lw.attn_q_proj_width
                    : (lw.attn_q_has_q ? lw.attn_q_q_w.shape(0) : (lw.attn_q_rhs.ndim() == 2 ? lw.attn_q_rhs.shape(1) : 0));
                if (q_proj_out_dim <= 0 || (q_proj_out_dim % lw.attn_num_heads) != 0) {
                    throw std::runtime_error(
                        "q_proj output dim is not divisible by num_attention_heads for layer " +
                        std::to_string(i)
                    );
                }

                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    if (lw.attn_q_norm_w.ndim() != 1) {
                        throw std::runtime_error("q_norm must be rank-1 for layer " + std::to_string(i));
                    }
                    lw.attn_head_dim = lw.attn_q_norm_w.shape(0);
                    if (lw.attn_head_dim <= 0) {
                        throw std::runtime_error("derived attn head dim is invalid for layer " + std::to_string(i));
                    }
                }

                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    const int k_proj_out_dim = lw.attn_k_proj_width > 0
                        ? lw.attn_k_proj_width
                        : (lw.attn_k_has_q ? lw.attn_k_q_w.shape(0) : (lw.attn_k_rhs.ndim() == 2 ? lw.attn_k_rhs.shape(1) : 0));
                    if (k_proj_out_dim <= 0 || (k_proj_out_dim % lw.attn_head_dim) != 0) {
                        throw std::runtime_error(
                            "k_proj output dim is not divisible by derived head_dim for layer " +
                            std::to_string(i)
                        );
                    }
                    lw.attn_num_kv_heads = k_proj_out_dim / lw.attn_head_dim;
                    if (lw.attn_num_kv_heads <= 0) {
                        throw std::runtime_error("derived num_key_value_heads is invalid for layer " + std::to_string(i));
                    }
                } else {
                    lw.attn_num_kv_heads = ctx->cfg.num_key_value_heads;
                }
                lw.attn_scale = (ctx->cfg.attention_multiplier > 0.0f)
                    ? ctx->cfg.attention_multiplier
                    : (1.0f / std::sqrt(static_cast<float>(lw.attn_head_dim)));
                lw.attn_rope_theta = ctx->cfg.rope_theta;
                lw.attn_partial_rotary_factor = ctx->cfg.partial_rotary_factor;
                if (ctx->cfg.use_layer_q_norm_head_dim) {
                    bool is_full_attention_layer = lw.attn_head_dim > ctx->cfg.head_dim;
                    if (i >= 0 && i < static_cast<int>(ctx->cfg.layer_types.size())) {
                        is_full_attention_layer = (ctx->cfg.layer_types[static_cast<size_t>(i)] == "full_attention");
                    }
                    if (is_full_attention_layer) {
                        lw.attn_partial_rotary_factor = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.partial_rotary_full
                            : 0.25f;
                        lw.attn_rope_theta = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.rope_theta_full
                            : 1000000.0f;
                    } else {
                        lw.attn_partial_rotary_factor = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.partial_rotary_local
                            : 1.0f;
                        lw.attn_rope_theta = ctx->cfg.has_nested_rope_parameters
                            ? ctx->cfg.rope_theta_local
                            : 10000.0f;
                    }
                }
                const bool layer_type_marks_full =
                    i >= 0 &&
                    i < static_cast<int>(ctx->cfg.layer_types.size()) &&
                    ctx->cfg.layer_types[static_cast<size_t>(i)] == "full_attention";
                lw.attn_is_sliding =
                    i >= 0 &&
                    i < static_cast<int>(ctx->cfg.layer_types.size()) &&
                    ctx->cfg.layer_types[static_cast<size_t>(i)] == "sliding_attention";
                const bool inferred_full_from_head_dim = lw.attn_head_dim > ctx->cfg.head_dim;
                const bool is_full_attention_layer =
                    layer_type_marks_full || inferred_full_from_head_dim;
                const bool disable_global_rope_adjust =
                    env_truthy("TALU_METAL_DISABLE_GLOBAL_ROPE_ADJUST", false);
                if (!disable_global_rope_adjust &&
                    is_full_attention_layer &&
                    ctx->cfg.global_head_dim > 0)
                {
                    const int rope_dim = std::max(
                        0,
                        static_cast<int>(std::round(
                            lw.attn_head_dim * lw.attn_partial_rotary_factor
                        ))
                    );
                    lw.attn_rope_theta = adjust_proportional_global_rope_theta(
                        lw.attn_rope_theta,
                        rope_dim,
                        ctx->cfg.global_head_dim
                    );
                }
            }

            ctx->layers.push_back(std::move(lw));
        }

        // Pre-compute RoPE frequency tables so decode can use array-based
        // rotary embeddings (compilable, no position-dependent int params).
        for (LayerWeights& layer : ctx->layers) {
            if (layer.is_linear || layer.is_shortconv) continue;
            const int rope_dims = static_cast<int>(
                std::round(layer.attn_head_dim * layer.attn_partial_rotary_factor));
            layer.attn_rope_dims = rope_dims;
            if (rope_dims <= 0) continue;
            const int half = rope_dims / 2;
            std::vector<float> freq_values(static_cast<size_t>(half));
            for (int d = 0; d < half; ++d) {
                freq_values[static_cast<size_t>(d)] = 1.0f / std::pow(
                    layer.attn_rope_theta,
                    2.0f * static_cast<float>(d) / static_cast<float>(rope_dims)
                );
            }
            layer.rope_freq = stop_gradient(copy(
                array(freq_values.data(), {half}, float32)
            ));
            eval(layer.rope_freq);
        }

        const bool enable_init_warmup = env_truthy("TALU_METAL_INIT_WARMUP", true);
        if (enable_init_warmup) {
            // Pre-warm compiled compute_g path so WARMUP=0 benchmarks do not
            // include first-trace compilation overhead.
            for (const auto& lw : ctx->layers) {
                if (!lw.is_linear) continue;
                const array a_dummy = zeros({1, 1, lw.lin_num_v_heads}, lw.lin_dt_bias.dtype());
                const array g_dummy = compute_g(lw.lin_A_exp_f32, a_dummy, lw.lin_dt_bias);
                eval(g_dummy);
                synchronize();
                break;
            }

            // Pre-warm one tiny prefill+decode pass so WARMUP=0 timings don't
            // include first-use kernel/JIT materialization costs.
            reset_runtime_state(ctx.get());
            const std::vector<int32_t> warm_ids = { 0, 1 };
            const array warm_prompt(warm_ids.begin(), Shape{1, 2}, int32);
            const array warm_prefill_logits = forward_logits(ctx.get(), warm_prompt);
            const array warm_seed = next_token_greedy(last_token_logits(warm_prefill_logits));
            const array warm_decode_logits = forward_logits(ctx.get(), warm_seed);
            eval(warm_decode_logits);
            synchronize();
            reset_runtime_state(ctx.get());
        }

        {
            size_t layer_q_count = 0;
            size_t layer_q_total = 0;
            size_t layer_per_layer_count = 0;
            // Branch-specific QMM counters.
            size_t ssm_qkvz_q = 0, ssm_qkvz_total = 0;
            size_t ssm_ba_q = 0, ssm_ba_total = 0;
            size_t ssm_out_q = 0, ssm_out_total = 0;
            size_t attn_qkv_q = 0, attn_qkv_total = 0;
            size_t attn_o_q = 0, attn_o_total = 0;
            size_t mlp_q = 0, mlp_total = 0;
            size_t dense_fallback_bytes = 0;
            const bool quantized_checkpoint =
                has_mxfp8_meta || ctx->has_grouped_affine_meta || has_nvfp4_meta;
            const bool strict_quantized_decode_requested =
                env_truthy("TALU_METAL_STRICT_QUANTIZED_DECODE", false);
            const bool strict_quantized_decode_active =
                strict_quantized_decode_requested && quantized_checkpoint;
            if (strict_quantized_decode_active && !ctx->fp8_decode_qmm_enabled) {
                throw std::runtime_error(
                    "TALU_METAL_STRICT_QUANTIZED_DECODE requires TALU_METAL_FP8_DECODE_QMM=1"
                );
            }
            size_t strict_required = 0;
            size_t strict_quantized = 0;
            std::vector<std::string> strict_missing;
            strict_missing.reserve(64);
            auto count_dense = [&](const array& rhs) {
                if (rhs.size() > 1 && rhs.ndim() == 2) {
                    dense_fallback_bytes += static_cast<size_t>(rhs.size()) * rhs.itemsize();
                }
            };
            auto record_strict = [&](bool required, bool is_quantized, const std::string& name) {
                if (!strict_quantized_decode_active || !required) return;
                strict_required += 1;
                if (is_quantized) {
                    strict_quantized += 1;
                } else {
                    strict_missing.push_back(name);
                }
            };
            for (size_t layer_idx = 0; layer_idx < ctx->layers.size(); ++layer_idx) {
                const LayerWeights& lw = ctx->layers[layer_idx];
                const bool flags[] = {
                    lw.mlp_gate_has_q,
                    lw.mlp_up_has_q,
                    lw.mlp_down_has_q,
                    lw.lin_in_proj_qkvz_has_q,
                    lw.lin_in_proj_ba_has_q,
                    lw.lin_out_proj_has_q,
                    lw.sc_in_proj_has_q,
                    lw.sc_out_proj_has_q,
                    lw.attn_q_has_q,
                    lw.attn_k_has_q,
                    lw.attn_v_has_q,
                    lw.attn_o_has_q,
                };
                for (bool flag : flags) {
                    layer_q_total += 1;
                    if (flag) layer_q_count += 1;
                }
                if (lw.has_per_layer_input) layer_per_layer_count += 1;
                // SSM / linear-attention branch.
                // After compact_linear_in_proj_weights, individual qkvz/ba flags
                // are cleared — check the combined all_has_q instead.
                if (lw.is_linear) {
                    ssm_qkvz_total += 1;
                    ssm_ba_total += 1;
                    ssm_out_total += 1;
                    const bool ssm_in_proj_q = lw.lin_in_proj_all_has_q || lw.lin_in_proj_qkvz_has_q;
                    if (ssm_in_proj_q) { ssm_qkvz_q += 1; ssm_ba_q += 1; }
                    else { count_dense(lw.lin_in_proj_qkvz_rhs); count_dense(lw.lin_in_proj_ba_rhs); }
                    if (lw.lin_out_proj_has_q) ssm_out_q += 1;
                    else count_dense(lw.lin_out_proj_rhs);
                    record_strict(true, ssm_in_proj_q, "layer" + std::to_string(layer_idx) + ".lin_in_proj");
                    record_strict(true, lw.lin_out_proj_has_q, "layer" + std::to_string(layer_idx) + ".lin_out_proj");
                }
                // ShortConv branch.
                if (lw.is_shortconv) {
                    record_strict(true, lw.sc_in_proj_has_q, "layer" + std::to_string(layer_idx) + ".sc_in_proj");
                    record_strict(true, lw.sc_out_proj_has_q, "layer" + std::to_string(layer_idx) + ".sc_out_proj");
                }
                // Full-attention branch (count fused QKV + O).
                if (!lw.is_linear && !lw.is_shortconv) {
                    attn_qkv_total += 1;
                    attn_o_total += 1;
                    if (lw.attn_qkv_has_q || lw.attn_q_has_q) attn_qkv_q += 1;
                    else { count_dense(lw.attn_qkv_rhs); count_dense(lw.attn_q_rhs); }
                    if (lw.attn_o_has_q) attn_o_q += 1;
                    else count_dense(lw.attn_o_rhs);
                    const bool attn_qkv_quantized = lw.attn_qkv_has_q || (lw.attn_q_has_q && lw.attn_k_has_q && lw.attn_v_has_q);
                    record_strict(true, attn_qkv_quantized, "layer" + std::to_string(layer_idx) + ".attn_qkv");
                    record_strict(true, lw.attn_o_has_q, "layer" + std::to_string(layer_idx) + ".attn_o");
                }
                // MLP branch.
                mlp_total += 1;
                const bool mlp_q_ok = lw.mlp_gate_up_has_q || lw.mlp_gate_has_q || lw.mlp_is_moe;
                if (mlp_q_ok && lw.mlp_down_has_q) mlp_q += 1;
                else {
                    count_dense(lw.mlp_gate_up_rhs);
                    count_dense(lw.mlp_gate_rhs);
                    count_dense(lw.mlp_up_rhs);
                    if (!lw.mlp_down_has_q) count_dense(lw.mlp_down_rhs);
                }
                if (lw.mlp_is_moe) {
                    const bool moe_quantized =
                        lw.moe_gate_scales.size() > 1 &&
                        lw.moe_up_scales.size() > 1 &&
                        lw.moe_down_scales.size() > 1;
                    record_strict(true, moe_quantized, "layer" + std::to_string(layer_idx) + ".moe_experts");
                } else {
                    const bool mlp_in_quantized = lw.mlp_gate_up_has_q || (lw.mlp_gate_has_q && lw.mlp_up_has_q);
                    record_strict(true, mlp_in_quantized, "layer" + std::to_string(layer_idx) + ".mlp_in");
                    record_strict(true, lw.mlp_down_has_q, "layer" + std::to_string(layer_idx) + ".mlp_down");
                }
            }
            std::fprintf(
                stderr,
                "[mlx][nvfp4] model=%s nvfp4=%d gaffine=%d decode_qmm=%d lm_head_q=%d layers_q=%zu/%zu group=%d bits=%d per_layer=%d/%zu hpl=%d gelu=%d\n",
                ctx->model_id.c_str(),
                has_nvfp4_meta ? 1 : 0,
                ctx->has_grouped_affine_meta ? 1 : 0,
                ctx->fp8_decode_qmm_enabled ? 1 : 0,
                ctx->lm_head_q_decode_enabled ? 1 : 0,
                layer_q_count,
                layer_q_total,
                ctx->cfg.quant_group_size,
                ctx->cfg.quant_bits,
                ctx->per_layer_input_enabled ? 1 : 0,
                layer_per_layer_count,
                ctx->per_layer_hidden_size,
                ctx->cfg.use_gelu ? 1 : 0
            );
            if (ssm_qkvz_total > 0 || attn_qkv_total > 0) {
                std::fprintf(
                    stderr,
                    "[mlx][qmm] ssm_qkvz=%zu/%zu ssm_ba=%zu/%zu ssm_out=%zu/%zu attn_qkv=%zu/%zu attn_o=%zu/%zu mlp=%zu/%zu dense_fallback=%.1fMB\n",
                    ssm_qkvz_q, ssm_qkvz_total,
                    ssm_ba_q, ssm_ba_total,
                    ssm_out_q, ssm_out_total,
                    attn_qkv_q, attn_qkv_total,
                    attn_o_q, attn_o_total,
                    mlp_q, mlp_total,
                    static_cast<double>(dense_fallback_bytes) / (1024.0 * 1024.0)
                );
            }
            std::fprintf(
                stderr,
                "[mlx][qmm_strict] requested=%d active=%d required=%zu quantized=%zu missing=%zu\n",
                strict_quantized_decode_requested ? 1 : 0,
                strict_quantized_decode_active ? 1 : 0,
                strict_required,
                strict_quantized,
                strict_missing.size()
            );
            if (strict_quantized_decode_active && !strict_missing.empty()) {
                std::string preview;
                const size_t max_items = std::min<size_t>(strict_missing.size(), 8);
                for (size_t idx = 0; idx < max_items; ++idx) {
                    if (!preview.empty()) preview += ", ";
                    preview += strict_missing[idx];
                }
                if (strict_missing.size() > max_items) {
                    preview += ", ...";
                }
                throw std::runtime_error(
                    "TALU_METAL_STRICT_QUANTIZED_DECODE: missing quantized decode projections: " + preview
                );
            }
            // Strict quantized decode mode: treat dense fallback as fatal.
            // Prevents silent memory ballooning and bandwidth loss from bf16
            // dequantized weights on memory-constrained systems (24GB).
            if (dense_fallback_bytes > 0) {
                if (strict_quantized_decode_active) {
                    char msg[512];
                    std::snprintf(
                        msg, sizeof(msg),
                        "TALU_METAL_STRICT_QUANTIZED_DECODE: %.1fMB dense fallback detected "
                        "(ssm_qkvz=%zu/%zu ssm_ba=%zu/%zu ssm_out=%zu/%zu "
                        "attn_qkv=%zu/%zu attn_o=%zu/%zu mlp=%zu/%zu). "
                        "All decode projections must use quantized matmul in strict mode.",
                        static_cast<double>(dense_fallback_bytes) / (1024.0 * 1024.0),
                        ssm_qkvz_q, ssm_qkvz_total,
                        ssm_ba_q, ssm_ba_total,
                        ssm_out_q, ssm_out_total,
                        attn_qkv_q, attn_qkv_total,
                        attn_o_q, attn_o_total,
                        mlp_q, mlp_total
                    );
                    throw std::runtime_error(msg);
                }
                if (ctx->has_grouped_affine_meta) {
                    std::fprintf(
                        stderr,
                        "[mlx][WARNING] %.1fMB of weights fell back to dense (QMM capture miss). "
                        "This increases memory usage and slows decode. "
                        "Set TALU_METAL_STRICT_QUANTIZED_DECODE=1 to fail-fast.\n",
                        static_cast<double>(dense_fallback_bytes) / (1024.0 * 1024.0)
                    );
                }
            }

            // Verify lm_head is quantized in strict mode.
            if (strict_quantized_decode_active && !ctx->lm_head_q_decode_enabled)
            {
                throw std::runtime_error(
                    "TALU_METAL_STRICT_QUANTIZED_DECODE: lm_head not captured for quantized decode. "
                    "Dense lm_head causes memory ballooning on constrained systems."
                );
            }

            // Verify embeddings are quantized in strict mode.
            if (strict_quantized_decode_active && !ctx->embed_tokens_has_q)
            {
                throw std::runtime_error(
                    "TALU_METAL_STRICT_QUANTIZED_DECODE: embed_tokens not captured for quantized lookup. "
                    "Dense embedding causes memory ballooning on constrained systems."
                );
            }
            std::fflush(stderr);
        }

        // Memory residency summary: total bytes retained in ctx arrays.
        {
            auto array_bytes = [](const array& a) -> size_t {
                return (a.size() > 1) ? static_cast<size_t>(a.size()) * a.itemsize() : 0;
            };
            size_t ctx_bytes = 0;
            ctx_bytes += array_bytes(ctx->embed_tokens);
            ctx_bytes += array_bytes(ctx->embed_tokens_q_w);
            ctx_bytes += array_bytes(ctx->embed_tokens_q_scales);
            ctx_bytes += array_bytes(ctx->embed_tokens_q_biases);
            ctx_bytes += array_bytes(ctx->lm_head_rhs);
            ctx_bytes += array_bytes(ctx->lm_head_q_w);
            ctx_bytes += array_bytes(ctx->lm_head_q_scales);
            ctx_bytes += array_bytes(ctx->lm_head_q_biases);
            ctx_bytes += array_bytes(ctx->final_norm_w);
            ctx_bytes += array_bytes(ctx->per_layer_embedding);
            size_t layer_q_bytes = 0;
            size_t layer_dense_bytes = 0;
            size_t layer_other_bytes = 0;
            for (const LayerWeights& lw : ctx->layers) {
                // Quantized weight storage (packed + scales + biases).
                for (const auto* a : {
                    &lw.mlp_gate_q_w, &lw.mlp_gate_q_scales, &lw.mlp_gate_q_biases,
                    &lw.mlp_up_q_w, &lw.mlp_up_q_scales, &lw.mlp_up_q_biases,
                    &lw.mlp_down_q_w, &lw.mlp_down_q_scales, &lw.mlp_down_q_biases,
                    &lw.mlp_gate_up_q_w, &lw.mlp_gate_up_q_scales, &lw.mlp_gate_up_q_biases,
                    &lw.attn_q_q_w, &lw.attn_q_q_scales, &lw.attn_q_q_biases,
                    &lw.attn_k_q_w, &lw.attn_k_q_scales, &lw.attn_k_q_biases,
                    &lw.attn_v_q_w, &lw.attn_v_q_scales, &lw.attn_v_q_biases,
                    &lw.attn_qkv_q_w, &lw.attn_qkv_q_scales, &lw.attn_qkv_q_biases,
                    &lw.attn_o_q_w, &lw.attn_o_q_scales, &lw.attn_o_q_biases,
                    &lw.lin_in_proj_qkvz_q_w, &lw.lin_in_proj_qkvz_q_scales, &lw.lin_in_proj_qkvz_q_biases,
                    &lw.lin_in_proj_ba_q_w, &lw.lin_in_proj_ba_q_scales, &lw.lin_in_proj_ba_q_biases,
                    &lw.lin_in_proj_all_q_w, &lw.lin_in_proj_all_q_scales, &lw.lin_in_proj_all_q_biases,
                    &lw.lin_out_proj_q_w, &lw.lin_out_proj_q_scales, &lw.lin_out_proj_q_biases,
                    &lw.sc_in_proj_q_w, &lw.sc_in_proj_q_scales, &lw.sc_in_proj_q_biases,
                    &lw.sc_out_proj_q_w, &lw.sc_out_proj_q_scales, &lw.sc_out_proj_q_biases,
                }) {
                    layer_q_bytes += array_bytes(*a);
                }
                // Dense RHS fallbacks.
                for (const auto* a : {
                    &lw.mlp_gate_rhs, &lw.mlp_up_rhs, &lw.mlp_down_rhs, &lw.mlp_gate_up_rhs,
                    &lw.attn_q_rhs, &lw.attn_k_rhs, &lw.attn_v_rhs, &lw.attn_qkv_rhs, &lw.attn_o_rhs,
                    &lw.attn_qkv_prefill_rhs,
                    &lw.lin_in_proj_qkvz_rhs, &lw.lin_in_proj_qkvz_prefill_rhs,
                    &lw.lin_in_proj_ba_rhs, &lw.lin_in_proj_all_rhs,
                    &lw.lin_out_proj_rhs,
                    &lw.sc_in_proj_rhs, &lw.sc_out_proj_rhs,
                    &lw.moe_shared_gate_rhs, &lw.moe_shared_up_rhs, &lw.moe_shared_down_rhs, &lw.moe_shared_gate_up_rhs,
                }) {
                    layer_dense_bytes += array_bytes(*a);
                }
                // Norms, conv, A_log, dt_bias, etc.
                for (const auto* a : {
                    &lw.input_layernorm_w, &lw.post_attention_layernorm_w,
                    &lw.lin_conv1d_w, &lw.lin_conv1d_decode_w,
                    &lw.lin_A_log, &lw.lin_A_exp_f32, &lw.lin_dt_bias, &lw.lin_norm_w,
                }) {
                    layer_other_bytes += array_bytes(*a);
                }
            }
            std::fprintf(
                stderr,
                "[mlx][mem] ctx=%.1fMB layer_q=%.1fMB layer_dense=%.1fMB layer_other=%.1fMB total=%.1fMB\n",
                static_cast<double>(ctx_bytes) / (1024.0 * 1024.0),
                static_cast<double>(layer_q_bytes) / (1024.0 * 1024.0),
                static_cast<double>(layer_dense_bytes) / (1024.0 * 1024.0),
                static_cast<double>(layer_other_bytes) / (1024.0 * 1024.0),
                static_cast<double>(ctx_bytes + layer_q_bytes + layer_dense_bytes + layer_other_bytes) / (1024.0 * 1024.0)
            );
            std::fflush(stderr);
        }

        if (qmm_debug) {
            const size_t attempts_delta = g_qmm_attempts - qmm_attempts_before;
            const size_t success_delta = g_qmm_success - qmm_success_before;
            std::fprintf(
                stderr,
                "[mlx][qmm] model=%s decode_qmm=%d attempts=%zu success=%zu\n",
                ctx->model_id.c_str(),
                ctx->fp8_decode_qmm_enabled ? 1 : 0,
                attempts_delta,
                success_delta
            );
        }

        // Release the safetensors tensor map. Per-layer weights already hold
        // their own references to the arrays they need; clearing the map drops
        // reference counts on any arrays only referenced here (unused metadata
        // tensors, __metadata__ entries, redundant aliases).
        tensors.clear();

        return ctx.release();
    } catch (const std::exception& e) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = "unknown error in mlx_create";
        return nullptr;
    }
}

mlx_ctx* mlx_create(const char* model_id, const char* model_path, int32_t seed) {
    return mlx_create_impl(model_id, model_path, seed, nullptr);
}

mlx_ctx* mlx_create_with_flags(const char* model_id, const char* model_path, int32_t seed, const mlx_model_flags* flags) {
    return mlx_create_impl(model_id, model_path, seed, flags);
}

mlx_ctx* mlx_clone(mlx_ctx* source_ctx, int32_t seed) {
    bool wired_limit_acquired = false;
    bool rng_entered = false;
    try {
        if (!source_ctx) {
            g_last_error = "mlx_clone: source ctx is null";
            return nullptr;
        }

        const bool enable_wired_limit = env_truthy("TALU_METAL_WIRED_LIMIT", false);
        if (enable_wired_limit) {
            wired_limit_acquired = acquire_wired_limit();
        }

        enter_rng_epoch(seed);
        rng_entered = true;

        auto ctx = std::make_unique<mlx_ctx>();
        ctx->model_id = source_ctx->model_id;
        ctx->model_path = source_ctx->model_path;
        ctx->cfg = source_ctx->cfg;

        // Arrays are ref-counted in MLX. Copying these fields keeps one shared
        // immutable weight set while each context owns independent runtime state.
        ctx->embed_tokens = source_ctx->embed_tokens;
        ctx->embed_tokens_q_w = source_ctx->embed_tokens_q_w;
        ctx->embed_tokens_q_scales = source_ctx->embed_tokens_q_scales;
        ctx->embed_tokens_q_biases = source_ctx->embed_tokens_q_biases;
        ctx->embed_tokens_q_bits = source_ctx->embed_tokens_q_bits;
        ctx->embed_tokens_has_q = source_ctx->embed_tokens_has_q;
        ctx->lm_head_rhs = source_ctx->lm_head_rhs;
        ctx->lm_head_q_w = source_ctx->lm_head_q_w;
        ctx->lm_head_q_scales = source_ctx->lm_head_q_scales;
        ctx->lm_head_q_biases = source_ctx->lm_head_q_biases;
        ctx->lm_head_q_bits = source_ctx->lm_head_q_bits;
        ctx->final_norm_w = source_ctx->final_norm_w;
        ctx->has_fp8_meta = source_ctx->has_fp8_meta;
        ctx->has_mxfp8_meta = source_ctx->has_mxfp8_meta;
        ctx->has_grouped_affine_meta = source_ctx->has_grouped_affine_meta;
        ctx->lm_head_q_decode_enabled = source_ctx->lm_head_q_decode_enabled;
        ctx->fp8_decode_qmm_enabled = source_ctx->fp8_decode_qmm_enabled;
        ctx->per_layer_input_enabled = source_ctx->per_layer_input_enabled;
        ctx->per_layer_hidden_size = source_ctx->per_layer_hidden_size;
        ctx->per_layer_input_scale = source_ctx->per_layer_input_scale;
        ctx->per_layer_embed_scale = source_ctx->per_layer_embed_scale;
        ctx->per_layer_model_projection_scale = source_ctx->per_layer_model_projection_scale;
        ctx->per_layer_embedding = source_ctx->per_layer_embedding;
        ctx->per_layer_projection_norm_w = source_ctx->per_layer_projection_norm_w;
        ctx->layers = source_ctx->layers;
        ctx->profile_layers = source_ctx->profile_layers;

        // Runtime state starts clean for the cloned context.
        ctx->kv_cache.resize(source_ctx->kv_cache.size());
        ctx->linear_cache.resize(source_ctx->linear_cache.size());
        ctx->stream_ready = false;
        ctx->trace_decode_token = 1;
        ctx->xray_enabled = false;
        ctx->sampling_context_counts.clear();
        ctx->sampling_context_len = 0;
        ctx->sampling_unique_ids.clear();
        ctx->sampling_repetition_scales.clear();
        ctx->sampling_additive_penalties.clear();

        return ctx.release();
    } catch (const std::exception& e) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = e.what();
        return nullptr;
    } catch (...) {
        if (wired_limit_acquired) release_wired_limit();
        if (rng_entered) leave_rng_epoch();
        g_last_error = "unknown error in mlx_clone";
        return nullptr;
    }
}

void mlx_destroy(mlx_ctx* ctx) {
    if (ctx) {
        release_wired_limit();
        leave_rng_epoch();
    }
    delete ctx;
}

int32_t mlx_reset(mlx_ctx* ctx) {
    if (!ctx) {
        g_last_error = "mlx_reset: invalid arguments";
        return 0;
    }
    try {
        reset_runtime_state(ctx);
        return 1;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return 0;
    } catch (...) {
        g_last_error = "unknown error in mlx_reset";
        return 0;
    }
}
