static array mla_attention_core(
    const array& input, // [B, L, d_model]
    const std::function<array(const array&)>& q_a_proj_fn,
    const array& q_a_norm_w,
    const std::function<array(const array&)>& q_b_proj_fn,
    const std::function<array(const array&)>& kv_a_proj_fn,
    const array& kv_a_norm_w,
    const std::function<array(const array&)>& kv_b_proj_fn,
    const std::function<array(const array&)>& out_proj_fn,
    void* cache_ptr,
    size_t layer_idx,
    int n_heads,
    int q_lora_rank,
    int kv_lora_rank,
    int qk_head_dim,
    int qk_rope_head_dim,
    int qk_nope_head_dim,
    int v_head_dim,
    size_t pos_offset,
    float rope_theta,
    const void* runtime_rope_cos,
    const void* runtime_rope_sin,
    size_t runtime_rope_dim,
    float rms_eps
) {
    const int batch_size = input.shape(0);
    const int seq_len = input.shape(1);
    const int hidden_dim = input.shape(2);
    if (qk_head_dim != (qk_nope_head_dim + qk_rope_head_dim)) {
        throw std::invalid_argument("[mla] qk_head_dim must equal qk_nope_head_dim + qk_rope_head_dim");
    }
    if (q_lora_rank <= 0 || kv_lora_rank <= 0 || qk_head_dim <= 0 || qk_rope_head_dim <= 0 || qk_nope_head_dim <= 0 || v_head_dim <= 0) {
        throw std::invalid_argument("[mla] invalid projection dimensions");
    }

    // FP32 allowlist: stabilize LoRA-rank reduction before RMS norm.
    array q_comp = astype(q_a_proj_fn(input), float32);
    if (q_comp.shape(2) != q_lora_rank) {
        throw std::invalid_argument("[mla] q_a_proj output does not match q_lora_rank");
    }
    q_comp = fast::rms_norm(q_comp, q_a_norm_w, rms_eps);
    // FP32 allowlist: keep Q projection accumulation in FP32 for attention logits.
    array q_proj = astype(q_b_proj_fn(q_comp), float32);
    if (q_proj.shape(2) != n_heads * qk_head_dim) {
        throw std::invalid_argument("[mla] q_b_proj output shape mismatch");
    }

    // FP32 allowlist: stabilize KV compressed projection before split + RMS norm.
    array kv_comp = astype(kv_a_proj_fn(input), float32);
    if (kv_comp.shape(2) != kv_lora_rank + qk_rope_head_dim) {
        throw std::invalid_argument("[mla] kv_a_proj output shape mismatch");
    }
    array kv_nope = slice(kv_comp, {0, 0, 0}, {batch_size, seq_len, kv_lora_rank});
    array k_rope_shared = slice(kv_comp, {0, 0, kv_lora_rank}, {batch_size, seq_len, kv_lora_rank + qk_rope_head_dim});
    kv_nope = fast::rms_norm(kv_nope, kv_a_norm_w, rms_eps);
    // FP32 allowlist: keep KV expansion accumulation in FP32 for attention logits.
    array kv_proj = astype(kv_b_proj_fn(kv_nope), float32);
    if (kv_proj.shape(2) != n_heads * (qk_nope_head_dim + v_head_dim)) {
        throw std::invalid_argument("[mla] kv_b_proj output shape mismatch");
    }

    array q_all = reshape(q_proj, {batch_size, seq_len, n_heads, qk_head_dim});
    array q_nope = transpose(
        slice(q_all, {0, 0, 0, 0}, {batch_size, seq_len, n_heads, qk_nope_head_dim}),
        {0, 2, 1, 3}
    );
    array q_rope = transpose(
        slice(q_all, {0, 0, 0, qk_nope_head_dim}, {batch_size, seq_len, n_heads, qk_head_dim}),
        {0, 2, 1, 3}
    );

    array kv_all = reshape(kv_proj, {batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim});
    array k_nope = transpose(
        slice(kv_all, {0, 0, 0, 0}, {batch_size, seq_len, n_heads, qk_nope_head_dim}),
        {0, 2, 1, 3}
    );
    array v_heads = transpose(
        slice(kv_all, {0, 0, 0, qk_nope_head_dim}, {batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim}),
        {0, 2, 1, 3}
    );

    k_rope_shared = reshape(k_rope_shared, {batch_size, 1, seq_len, qk_rope_head_dim});
    const bool has_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr && runtime_rope_dim > 0;
    if (has_runtime_rope) {
        const auto* cos_table = static_cast<const array*>(runtime_rope_cos);
        const auto* sin_table = static_cast<const array*>(runtime_rope_sin);
        q_rope = apply_optional_runtime_rope(
            q_rope,
            cos_table,
            sin_table,
            static_cast<int>(pos_offset),
            seq_len,
            qk_rope_head_dim,
            qk_rope_head_dim
        );
        k_rope_shared = apply_optional_runtime_rope(
            k_rope_shared,
            cos_table,
            sin_table,
            static_cast<int>(pos_offset),
            seq_len,
            qk_rope_head_dim,
            qk_rope_head_dim
        );
    } else {
        q_rope = fast::rope(q_rope, qk_rope_head_dim, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
        k_rope_shared = fast::rope(k_rope_shared, qk_rope_head_dim, false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    }

    array k_rope = (n_heads == 1) ? k_rope_shared : repeat(k_rope_shared, n_heads, 1);
    array q = concatenate({q_nope, q_rope}, 3);
    array k = concatenate({k_nope, k_rope}, 3);

    array k_for_attn = k;
    array v_for_attn = v_heads;
    bool is_prefill = true;
    if (cache_ptr != nullptr) {
        void* k_cached = nullptr;
        void* v_cached = nullptr;
        mlx_cache_update_and_fetch_bfloat16(
            cache_ptr,
            layer_idx,
            &k,
            &v_heads,
            &k_cached,
            &v_cached,
            &is_prefill
        );
        if (k_cached != nullptr && v_cached != nullptr) {
            k_for_attn = *static_cast<array*>(k_cached);
            v_for_attn = *static_cast<array*>(v_cached);
        }
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(qk_head_dim));
    array attn_out = fast::scaled_dot_product_attention(
        q,
        k_for_attn,
        v_for_attn,
        scale,
        is_prefill ? "causal" : ""
    );
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch_size, seq_len, n_heads * v_head_dim});

    array out = out_proj_fn(attn_out);
    if (out.ndim() != 3 || out.shape(0) != batch_size || out.shape(1) != seq_len || out.shape(2) != hidden_dim) {
        throw std::invalid_argument("[mla] output projection shape mismatch");
    }
    return out;
}

void* mlx_lazy_mla_attention_quantized(
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
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const auto& q_a_norm = *static_cast<const array*>(q_a_norm_w);
    const auto& kv_a_norm = *static_cast<const array*>(kv_a_norm_w);
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    const auto q_a_fn = [q_a_w, q_a_s, q_a_b, group_size_i, bits_i](const array& in) -> array {
        return quantized_linear_no_bias(
            in,
            *static_cast<const array*>(q_a_w),
            *static_cast<const array*>(q_a_s),
            *static_cast<const array*>(q_a_b),
            group_size_i,
            bits_i
        );
    };
    const auto q_b_fn = [q_b_w, q_b_s, q_b_b, group_size_i, bits_i](const array& in) -> array {
        return quantized_linear_no_bias(
            in,
            *static_cast<const array*>(q_b_w),
            *static_cast<const array*>(q_b_s),
            *static_cast<const array*>(q_b_b),
            group_size_i,
            bits_i
        );
    };
    const auto kv_a_fn = [kv_a_w, kv_a_s, kv_a_b, group_size_i, bits_i](const array& in) -> array {
        return quantized_linear_no_bias(
            in,
            *static_cast<const array*>(kv_a_w),
            *static_cast<const array*>(kv_a_s),
            *static_cast<const array*>(kv_a_b),
            group_size_i,
            bits_i
        );
    };
    const auto kv_b_fn = [kv_b_w, kv_b_s, kv_b_b, group_size_i, bits_i](const array& in) -> array {
        return quantized_linear_no_bias(
            in,
            *static_cast<const array*>(kv_b_w),
            *static_cast<const array*>(kv_b_s),
            *static_cast<const array*>(kv_b_b),
            group_size_i,
            bits_i
        );
    };
    const auto out_fn = [o_w, o_s, o_b, group_size_i, bits_i](const array& in) -> array {
        return quantized_linear_no_bias(
            in,
            *static_cast<const array*>(o_w),
            *static_cast<const array*>(o_s),
            *static_cast<const array*>(o_b),
            group_size_i,
            bits_i
        );
    };

    array out = mla_attention_core(
        x,
        q_a_fn,
        q_a_norm,
        q_b_fn,
        kv_a_fn,
        kv_a_norm,
        kv_b_fn,
        out_fn,
        cache_ptr,
        layer_idx,
        static_cast<int>(n_heads),
        static_cast<int>(q_lora_rank),
        static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_head_dim),
        static_cast<int>(qk_rope_head_dim),
        static_cast<int>(qk_nope_head_dim),
        static_cast<int>(v_head_dim),
        pos_offset,
        rope_theta,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim,
        rms_eps
    );
    return pool_array(std::move(out));
}

void* mlx_lazy_mla_attention_bf16(
    const void* input,
    const void* q_a_w, const void* q_a_norm_w, const void* q_b_w,
    const void* kv_a_w, const void* kv_a_norm_w, const void* kv_b_w,
    const void* o_w,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads,
    size_t q_lora_rank, size_t kv_lora_rank,
    size_t qk_head_dim, size_t qk_rope_head_dim, size_t qk_nope_head_dim, size_t v_head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps
) {
    mlx_count_op();
    const auto& x = *static_cast<const array*>(input);
    const auto& q_a_norm = *static_cast<const array*>(q_a_norm_w);
    const auto& kv_a_norm = *static_cast<const array*>(kv_a_norm_w);

    const int n_heads_i = static_cast<int>(n_heads);
    const int q_lora_rank_i = static_cast<int>(q_lora_rank);
    const int kv_lora_rank_i = static_cast<int>(kv_lora_rank);
    const int qk_head_dim_i = static_cast<int>(qk_head_dim);
    const int qk_rope_head_dim_i = static_cast<int>(qk_rope_head_dim);
    const int qk_nope_head_dim_i = static_cast<int>(qk_nope_head_dim);
    const int v_head_dim_i = static_cast<int>(v_head_dim);
    const int hidden_dim = x.shape(2);

    const auto q_a_fn = [q_a_w, q_lora_rank_i](const array& in) -> array {
        return dense_linear_no_bias(in, *static_cast<const array*>(q_a_w), q_lora_rank_i);
    };
    const auto q_b_fn = [q_b_w, n_heads_i, qk_head_dim_i](const array& in) -> array {
        return dense_linear_no_bias(in, *static_cast<const array*>(q_b_w), n_heads_i * qk_head_dim_i);
    };
    const auto kv_a_fn = [kv_a_w, kv_lora_rank_i, qk_rope_head_dim_i](const array& in) -> array {
        return dense_linear_no_bias(in, *static_cast<const array*>(kv_a_w), kv_lora_rank_i + qk_rope_head_dim_i);
    };
    const auto kv_b_fn = [kv_b_w, n_heads_i, qk_nope_head_dim_i, v_head_dim_i](const array& in) -> array {
        return dense_linear_no_bias(in, *static_cast<const array*>(kv_b_w), n_heads_i * (qk_nope_head_dim_i + v_head_dim_i));
    };
    const auto out_fn = [o_w, hidden_dim](const array& in) -> array {
        return dense_linear_no_bias(in, *static_cast<const array*>(o_w), hidden_dim);
    };

    array out = mla_attention_core(
        x,
        q_a_fn,
        q_a_norm,
        q_b_fn,
        kv_a_fn,
        kv_a_norm,
        kv_b_fn,
        out_fn,
        cache_ptr,
        layer_idx,
        n_heads_i,
        q_lora_rank_i,
        kv_lora_rank_i,
        qk_head_dim_i,
        qk_rope_head_dim_i,
        qk_nope_head_dim_i,
        v_head_dim_i,
        pos_offset,
        rope_theta,
        runtime_rope_cos,
        runtime_rope_sin,
        runtime_rope_dim,
        rms_eps
    );
    return pool_array(std::move(out));
}
