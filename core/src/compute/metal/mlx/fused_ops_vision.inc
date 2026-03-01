void* mlx_lazy_vision_block_fused_qkv_bf16(
    const void* input,
    const void* ln1_weight,
    const void* ln1_bias,
    const void* ln2_weight,
    const void* ln2_bias,
    const void* qkv_weight,
    const void* qkv_bias,
    const void* o_weight,
    const void* o_bias,
    const void* fc1_weight,
    const void* fc1_bias,
    const void* fc2_weight,
    const void* fc2_bias,
    const void* runtime_rope_cos,
    const void* runtime_rope_sin,
    size_t runtime_rope_dim,
    size_t n_heads,
    size_t head_dim,
    float attn_scale,
    float norm_eps
) {
    const auto& x = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& ln1_b = *static_cast<const array*>(ln1_bias);
    const auto& ln2_w = *static_cast<const array*>(ln2_weight);
    const auto& ln2_b = *static_cast<const array*>(ln2_bias);
    const auto& qkv_w = *static_cast<const array*>(qkv_weight);
    const auto& qkv_b = *static_cast<const array*>(qkv_bias);
    const auto& o_w = *static_cast<const array*>(o_weight);
    const auto& o_b = *static_cast<const array*>(o_bias);
    const auto& fc1_w = *static_cast<const array*>(fc1_weight);
    const auto& fc1_b = *static_cast<const array*>(fc1_bias);
    const auto& fc2_w = *static_cast<const array*>(fc2_weight);
    const auto& fc2_b = *static_cast<const array*>(fc2_bias);
    const auto* rope_cos = static_cast<const array*>(runtime_rope_cos);
    const auto* rope_sin = static_cast<const array*>(runtime_rope_sin);

    const int batch = x.shape(0);
    const int seq_len = x.shape(1);
    const int hidden_dim = x.shape(2);
    const int heads = static_cast<int>(n_heads);
    const int hd = static_cast<int>(head_dim);
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    const int expected_qkv_dim = 3 * hidden_dim;

    auto ln1 = layer_norm_last_dim(x, ln1_w, ln1_b, norm_eps);
    auto qkv = linear_bf16_with_bias(ln1, qkv_w, qkv_b, expected_qkv_dim);
    auto q = slice(qkv, {0, 0, 0}, {batch, seq_len, hidden_dim});
    auto k = slice(qkv, {0, 0, hidden_dim}, {batch, seq_len, 2 * hidden_dim});
    auto v = slice(qkv, {0, 0, 2 * hidden_dim}, {batch, seq_len, 3 * hidden_dim});

    q = reshape(q, {batch, seq_len, heads, hd});
    k = reshape(k, {batch, seq_len, heads, hd});
    v = reshape(v, {batch, seq_len, heads, hd});
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = apply_optional_runtime_rope(q, rope_cos, rope_sin, 0, seq_len, hd, rope_dim);
    k = apply_optional_runtime_rope(k, rope_cos, rope_sin, 0, seq_len, hd, rope_dim);

    auto attn = fast::scaled_dot_product_attention(q, k, v, attn_scale, "");
    attn = transpose(attn, {0, 2, 1, 3});
    attn = reshape(attn, {batch, seq_len, hidden_dim});
    const auto attn_out = linear_bf16_with_bias(attn, o_w, o_b, hidden_dim);
    const auto hidden_1 = x + attn_out;

    auto ln2 = layer_norm_last_dim(hidden_1, ln2_w, ln2_b, norm_eps);
    auto ffn = linear_bf16_with_bias(ln2, fc1_w, fc1_b, std::nullopt);
    ffn = gelu_approx(ffn);
    const auto ffn_out = linear_bf16_with_bias(ffn, fc2_w, fc2_b, hidden_dim);
    return pool_array(hidden_1 + ffn_out);
}

void* mlx_lazy_vision_block_split_qkv_bf16(
    const void* input,
    const void* ln1_weight,
    const void* ln1_bias,
    const void* ln2_weight,
    const void* ln2_bias,
    const void* q_weight,
    const void* q_bias,
    const void* k_weight,
    const void* k_bias,
    const void* v_weight,
    const void* v_bias,
    const void* o_weight,
    const void* o_bias,
    const void* fc1_weight,
    const void* fc1_bias,
    const void* fc2_weight,
    const void* fc2_bias,
    const void* runtime_rope_cos,
    const void* runtime_rope_sin,
    size_t runtime_rope_dim,
    size_t n_heads,
    size_t head_dim,
    float attn_scale,
    float norm_eps
) {
    const auto& x = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& ln1_b = *static_cast<const array*>(ln1_bias);
    const auto& ln2_w = *static_cast<const array*>(ln2_weight);
    const auto& ln2_b = *static_cast<const array*>(ln2_bias);
    const auto& q_w = *static_cast<const array*>(q_weight);
    const auto& q_b = *static_cast<const array*>(q_bias);
    const auto& k_w = *static_cast<const array*>(k_weight);
    const auto& k_b = *static_cast<const array*>(k_bias);
    const auto& v_w = *static_cast<const array*>(v_weight);
    const auto& v_b = *static_cast<const array*>(v_bias);
    const auto& o_w = *static_cast<const array*>(o_weight);
    const auto& o_b = *static_cast<const array*>(o_bias);
    const auto& fc1_w = *static_cast<const array*>(fc1_weight);
    const auto& fc1_b = *static_cast<const array*>(fc1_bias);
    const auto& fc2_w = *static_cast<const array*>(fc2_weight);
    const auto& fc2_b = *static_cast<const array*>(fc2_bias);
    const auto* rope_cos = static_cast<const array*>(runtime_rope_cos);
    const auto* rope_sin = static_cast<const array*>(runtime_rope_sin);

    const int batch = x.shape(0);
    const int seq_len = x.shape(1);
    const int hidden_dim = x.shape(2);
    const int heads = static_cast<int>(n_heads);
    const int hd = static_cast<int>(head_dim);
    const int rope_dim = static_cast<int>(runtime_rope_dim);

    auto ln1 = layer_norm_last_dim(x, ln1_w, ln1_b, norm_eps);
    auto q = linear_bf16_with_bias(ln1, q_w, q_b, hidden_dim);
    auto k = linear_bf16_with_bias(ln1, k_w, k_b, hidden_dim);
    auto v = linear_bf16_with_bias(ln1, v_w, v_b, hidden_dim);

    q = reshape(q, {batch, seq_len, heads, hd});
    k = reshape(k, {batch, seq_len, heads, hd});
    v = reshape(v, {batch, seq_len, heads, hd});
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    q = apply_optional_runtime_rope(q, rope_cos, rope_sin, 0, seq_len, hd, rope_dim);
    k = apply_optional_runtime_rope(k, rope_cos, rope_sin, 0, seq_len, hd, rope_dim);

    auto attn = fast::scaled_dot_product_attention(q, k, v, attn_scale, "");
    attn = transpose(attn, {0, 2, 1, 3});
    attn = reshape(attn, {batch, seq_len, hidden_dim});
    const auto attn_out = linear_bf16_with_bias(attn, o_w, o_b, hidden_dim);
    const auto hidden_1 = x + attn_out;

    auto ln2 = layer_norm_last_dim(hidden_1, ln2_w, ln2_b, norm_eps);
    auto ffn = linear_bf16_with_bias(ln2, fc1_w, fc1_b, std::nullopt);
    ffn = gelu_approx(ffn);
    const auto ffn_out = linear_bf16_with_bias(ffn, fc2_w, fc2_b, hidden_dim);
    return pool_array(hidden_1 + ffn_out);
}
