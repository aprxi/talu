static array apply_runtime_rope_to_tensor(
    const array& x, // [B, H, L, D]
    const array& cos_rows, // [L, rope_dim]
    const array& sin_rows, // [L, rope_dim]
    int seq_len,
    int head_dim,
    int rope_dim
) {
    const int batch = x.shape(0);
    const int heads = x.shape(1);
    const int half = rope_dim / 2;

    auto x_rot = slice(x, {0, 0, 0, 0}, {batch, heads, seq_len, rope_dim});
    auto cos_view = reshape(cos_rows, {1, 1, seq_len, rope_dim});
    auto sin_view = reshape(sin_rows, {1, 1, seq_len, rope_dim});

    auto x_lo = slice(x_rot, {0, 0, 0, 0}, {batch, heads, seq_len, half});
    auto x_hi = slice(x_rot, {0, 0, 0, half}, {batch, heads, seq_len, rope_dim});
    auto cos_lo = slice(cos_view, {0, 0, 0, 0}, {1, 1, seq_len, half});
    auto cos_hi = slice(cos_view, {0, 0, 0, half}, {1, 1, seq_len, rope_dim});
    auto sin_lo = slice(sin_view, {0, 0, 0, 0}, {1, 1, seq_len, half});
    auto sin_hi = slice(sin_view, {0, 0, 0, half}, {1, 1, seq_len, rope_dim});

    auto rotated_lo = x_lo * cos_lo - x_hi * sin_lo;
    auto rotated_hi = x_hi * cos_hi + x_lo * sin_hi;
    auto rotated = concatenate({rotated_lo, rotated_hi}, 3);

    if (rope_dim == head_dim) {
        return rotated;
    }
    auto tail = slice(x, {0, 0, 0, rope_dim}, {batch, heads, seq_len, head_dim});
    return concatenate({rotated, tail}, 3);
}

// Infer matmul RHS orientation from shape so fused dense ops remain robust across
// model families and loader conventions.
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
        return transpose_when_ambiguous ? transpose(weight) : weight;
    }
    throw std::invalid_argument("[dense] weight orientation incompatible with matmul shape");
}

static array gelu_approx(const array& x) {
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;
    const auto x3 = x * x * x;
    return 0.5f * x * (1.0f + tanh(kSqrt2OverPi * (x + kCoeff * x3)));
}

static inline array ensure_float32(const array& arr) {
    // FP32 allowlist: recurrent Mamba SSM/conv state updates are numerically
    // sensitive across long contexts; keep state math in FP32.
    return arr.dtype() == float32 ? arr : astype(arr, float32);
}

static array layer_norm_last_dim(
    const array& x,
    const array& weight,
    const array& bias,
    float eps
) {
    const auto x_mean = mean(x, -1, true);
    const auto centered = x - x_mean;
    const auto variance = mean(centered * centered, -1, true);
    const auto inv_std = rsqrt(variance + eps);
    const auto normed = centered * inv_std;
    return normed * weight + bias;
}

static array linear_bf16_with_bias(
    const array& input,
    const array& weight,
    const array& bias,
    std::optional<int> out_features = std::nullopt
) {
    const int in_features = input.shape(input.ndim() - 1);
    const auto rhs = orient_matmul_rhs(weight, in_features, out_features, true);
    return matmul(input, rhs) + bias;
}

static bool can_use_runtime_rope(
    const array* runtime_rope_cos,
    const array* runtime_rope_sin,
    int seq_start,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (runtime_rope_cos == nullptr || runtime_rope_sin == nullptr) return false;
    if (rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return false;
    const int seq_stop = seq_start + seq_len;
    return runtime_rope_cos->ndim() == 2 and runtime_rope_sin->ndim() == 2 and
        runtime_rope_cos->shape(0) >= seq_stop and runtime_rope_sin->shape(0) >= seq_stop and
        runtime_rope_cos->shape(1) >= rope_dim and runtime_rope_sin->shape(1) >= rope_dim;
}

static array apply_optional_runtime_rope(
    const array& x, // [B, H, L, D]
    const array* runtime_rope_cos,
    const array* runtime_rope_sin,
    int seq_start,
    int seq_len,
    int head_dim,
    int rope_dim
) {
    if (!can_use_runtime_rope(runtime_rope_cos, runtime_rope_sin, seq_start, seq_len, head_dim, rope_dim)) {
        return x;
    }
    auto cos_rows = slice(*runtime_rope_cos, {seq_start, 0}, {seq_start + seq_len, rope_dim});
    auto sin_rows = slice(*runtime_rope_sin, {seq_start, 0}, {seq_start + seq_len, rope_dim});
    return apply_runtime_rope_to_tensor(x, cos_rows, sin_rows, seq_len, head_dim, rope_dim);
}

static array dense_linear_no_bias(
    const array& input,
    const array& weight,
    std::optional<int> out_features = std::nullopt
) {
    const int in_features = input.shape(input.ndim() - 1);
    const auto rhs = orient_matmul_rhs(weight, in_features, out_features, true);
    return matmul(input, rhs);
}

static array quantized_linear_no_bias(
    const array& input,
    const array& weights,
    const array& scales,
    const array& biases,
    int group_size,
    int bits
) {
    return quantized_matmul(
        input,
        weights,
        scales,
        biases,
        true,
        group_size,
        bits,
        "affine"
    );
}

static array resolve_mamba_conv_kernel(
    const array& conv_weight,
    int d_conv,
    int xbc_len
) {
    array kernel = conv_weight;
    if (kernel.ndim() == 3 &&
        kernel.shape(0) == 1 &&
        kernel.shape(1) == d_conv &&
        kernel.shape(2) == xbc_len) {
        // Precomputed setup-time layout [1, d_conv, xbc_len].
        return ensure_float32(kernel);
    }
    if (kernel.ndim() == 3) {
        // [xbc_len, 1, d_conv] -> [xbc_len, d_conv]
        kernel = reshape(kernel, {kernel.shape(0), kernel.shape(2)});
    }
    if (kernel.ndim() != 2) {
        throw std::invalid_argument("[mamba] conv1d_weight must be rank-2 or rank-3");
    }

    if (kernel.shape(0) == xbc_len and kernel.shape(1) == d_conv) {
        kernel = transpose(kernel);
    } else if (!(kernel.shape(0) == d_conv and kernel.shape(1) == xbc_len)) {
        throw std::invalid_argument("[mamba] conv1d_weight shape incompatible with xbc_len/d_conv");
    }

    kernel = ensure_float32(kernel);
    return reshape(kernel, {1, d_conv, xbc_len});
}

static array mamba_forward_core(
    const array& input, // [1, L, d_model]
    const array& ln1_weight,
    const std::function<array(const array&)>& in_proj_fn,
    const array& conv_kernel_broadcast, // [1, d_conv, xbc_len]
    const std::optional<array>& conv_bias_row, // [1, xbc_len]
    const array& a_log, // [n_heads]
    const array& d_skip, // [n_heads]
    const std::optional<array>& dt_bias_row, // [1, n_heads]
    const std::optional<array>& norm_weight, // [d_inner]
    const std::function<array(const array&)>& out_proj_fn,
    const std::optional<array>& ln2_weight,
    const std::function<array(const array&)>& gate_up_fn,
    const std::function<array(const array&)>& down_proj_fn,
    bool has_ffn,
    bool use_gelu,
    float residual_multiplier,
    float norm_eps,
    int d_state,
    int d_conv,
    int n_heads,
    int d_head,
    int n_groups,
    uint8_t gate_up_layout,
    array& conv_state,
    array& ssm_state
) {
    const int seq_len = input.shape(1);
    const int d_inner = n_heads * d_head;
    const int bc_len = n_groups * d_state;
    const int xbc_len = d_inner + 2 * bc_len;
    if (n_heads <= 0 or d_head <= 0 or d_state <= 0 or d_conv <= 0 or n_groups <= 0) {
        throw std::invalid_argument("[mamba] invalid config dims");
    }
    if ((n_heads % n_groups) != 0) {
        throw std::invalid_argument("[mamba] n_heads must be divisible by n_groups");
    }

    const array a_log_row = reshape(a_log, {1, n_heads, 1, 1});
    const array d_skip_row = reshape(d_skip, {1, n_heads, 1});
    const int heads_per_group = n_heads / n_groups;
    const int min_proj_dim = (2 * d_inner + 2 * bc_len + n_heads);

    // Hoist token-independent preprocessing out of the recurrent loop.
    const array norm_all = fast::rms_norm(input, ln1_weight, norm_eps);
    const array proj_all = in_proj_fn(norm_all);
    const int proj_dim = proj_all.shape(2);
    if (proj_dim < min_proj_dim) {
        throw std::invalid_argument("[mamba] in_proj output too small");
    }

    const array z_all = slice(proj_all, {0, 0, 0}, {1, seq_len, d_inner});
    const array xbc_all = ensure_float32(slice(proj_all, {0, 0, d_inner}, {1, seq_len, d_inner + xbc_len}));
    array dt_all = ensure_float32(slice(
        proj_all,
        {0, 0, d_inner + xbc_len},
        {1, seq_len, d_inner + xbc_len + n_heads}
    ));
    if (dt_bias_row) {
        dt_all = dt_all + reshape(*dt_bias_row, {1, 1, n_heads});
    }
    dt_all = log(1.0f + exp(dt_all));

    // Vectorized depthwise conv over [state, sequence] for xbc stream.
    const array conv_history = concatenate({conv_state, xbc_all}, 1);
    const array conv_weight = transpose(conv_kernel_broadcast, {2, 1, 0}); // [xbc_len, d_conv, 1]
    array conv_seq = conv1d(conv_history, conv_weight, 1, 0, 1, xbc_len);
    conv_seq = slice(conv_seq, {0, 1, 0}, {1, seq_len + 1, xbc_len});
    if (conv_bias_row) {
        conv_seq = conv_seq + reshape(*conv_bias_row, {1, 1, xbc_len});
    }
    const array xbc_conv_all = conv_seq * sigmoid(conv_seq);
    conv_state = slice(conv_history, {0, seq_len, 0}, {1, seq_len + d_conv, xbc_len});

    const array x_conv_all = slice(xbc_conv_all, {0, 0, 0}, {1, seq_len, d_inner});
    array b_heads_all = reshape(
        slice(xbc_conv_all, {0, 0, d_inner}, {1, seq_len, d_inner + bc_len}),
        {seq_len, n_groups, d_state}
    );
    array c_heads_all = reshape(
        slice(xbc_conv_all, {0, 0, d_inner + bc_len}, {1, seq_len, d_inner + 2 * bc_len}),
        {seq_len, n_groups, d_state}
    );
    if (heads_per_group > 1) {
        b_heads_all = repeat(b_heads_all, heads_per_group, 1);
        c_heads_all = repeat(c_heads_all, heads_per_group, 1);
    }

    const array x_heads_all = reshape(x_conv_all, {seq_len, n_heads, d_head});
    const array dt4_all = reshape(dt_all, {seq_len, n_heads, 1, 1});
    const array dA_all = exp((-exp(a_log_row)) * dt4_all);
    const array x_term_all = dt4_all *
        reshape(x_heads_all, {seq_len, n_heads, d_head, 1}) *
        reshape(b_heads_all, {seq_len, n_heads, 1, d_state});

    // Solve s_t = dA_t * s_{t-1} + x_t with parallel prefix products/sums.
    // dA_t is strictly non-negative (exp of real), so clipping protects only
    // against denorm/underflow during long prefills.
    const array prefix = cumprod(dA_all, 0);
    const array prefix_safe = clip(prefix, array(1.0e-20f), std::nullopt);
    const array ssm_terms = x_term_all / prefix_safe;
    const array ssm_accum = cumsum(ssm_terms, 0);
    const array ssm_all = prefix * (ssm_state + ssm_accum);
    ssm_state = slice(ssm_all, {seq_len - 1, 0, 0, 0}, {seq_len, n_heads, d_head, d_state});

    array y_all = sum(ssm_all * reshape(c_heads_all, {seq_len, n_heads, 1, d_state}), 3);
    y_all = y_all + d_skip_row * x_heads_all;
    array ssm_out = reshape(y_all, {1, seq_len, d_inner});
    ssm_out = ssm_out * (z_all * sigmoid(z_all));
    if (norm_weight) {
        ssm_out = fast::rms_norm(ssm_out, *norm_weight, 1e-5f);
    }

    array mixer_out = ensure_float32(out_proj_fn(ssm_out));
    array hidden_1 = input + mixer_out * residual_multiplier;

    if (!has_ffn) {
        return hidden_1;
    }
    if (!ln2_weight.has_value()) {
        throw std::invalid_argument("[mamba] missing ln2 weight for FFN path");
    }

    const array norm2 = fast::rms_norm(hidden_1, *ln2_weight, norm_eps);
    const array gate_up = ensure_float32(gate_up_fn(norm2));
    const int gate_up_dim = gate_up.shape(2);
    if ((gate_up_dim % 2) != 0) {
        throw std::invalid_argument("[mamba] fused gate_up must have even width");
    }
    const int d_ff = gate_up_dim / 2;

    const array mid = [&]() -> array {
        if (gate_up_layout == 1) {
            const array reshaped = reshape(gate_up, {1, seq_len, d_ff, 2});
            const array gate = reshape(
                slice(reshaped, {0, 0, 0, 0}, {1, seq_len, d_ff, 1}),
                {1, seq_len, d_ff}
            );
            const array up = reshape(
                slice(reshaped, {0, 0, 0, 1}, {1, seq_len, d_ff, 2}),
                {1, seq_len, d_ff}
            );
            const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
            return activated * up;
        }

        const array gate = slice(gate_up, {0, 0, 0}, {1, seq_len, d_ff});
        const array up = slice(gate_up, {0, 0, d_ff}, {1, seq_len, 2 * d_ff});
        const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
        return activated * up;
    }();

    const array ffn_out = ensure_float32(down_proj_fn(mid));
    return hidden_1 + ffn_out * residual_multiplier;
}

void* mlx_lazy_mamba_block_bf16(
    const void* input,
    const void* ln1_weight,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* a_log,
    const void* d_skip,
    const void* dt_bias,
    const void* norm_weight,
    const void* out_proj,
    const void* ln2_weight,
    const void* gate_up,
    const void* down_proj,
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
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_proj);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = ensure_float32(*static_cast<const array*>(a_log));
    const auto d_skip_arr = ensure_float32(*static_cast<const array*>(d_skip));
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_proj);
    const auto* ln2_w = static_cast<const array*>(ln2_weight);
    const auto* gate_up_w = static_cast<const array*>(gate_up);
    const auto* down_proj_w = static_cast<const array*>(down_proj);

    const int d_state_i = static_cast<int>(d_state);
    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int n_groups_i = static_cast<int>(n_groups);
    const int d_inner = n_heads_i * d_head_i;
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;

    const array conv_kernel_broadcast = resolve_mamba_conv_kernel(conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(
            (conv_b->ndim() == 2 && conv_b->shape(0) == 1 && conv_b->shape(1) == xbc_len)
                ? ensure_float32(*conv_b)
                : reshape(ensure_float32(*conv_b), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(ensure_float32(*dt_b), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(ensure_float32(*norm_w))
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(ensure_float32(*ln2_w))
        : std::nullopt;

    auto* cache = static_cast<MLXMambaCache*>(mamba_cache_ptr);
    MambaLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_state_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_state_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_state_i}, float32);
    }

    const auto in_proj_fn = [&in_proj_w](const array& x) -> array {
        return dense_linear_no_bias(x, in_proj_w, std::nullopt);
    };
    const auto out_proj_fn = [&out_proj_w](const array& x) -> array {
        return dense_linear_no_bias(x, out_proj_w, std::nullopt);
    };
    const bool has_ffn = gate_up_w != nullptr && down_proj_w != nullptr && ln2_w != nullptr;
    const auto gate_up_fn = [&gate_up_w](const array& x) -> array {
        return dense_linear_no_bias(x, *gate_up_w, std::nullopt);
    };
    const auto down_proj_fn = [&down_proj_w](const array& x) -> array {
        return dense_linear_no_bias(x, *down_proj_w, std::nullopt);
    };

    array out = mamba_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        a_log_arr,
        d_skip_arr,
        dt_bias_row,
        norm_weight_arr,
        out_proj_fn,
        ln2_weight_arr,
        gate_up_fn,
        down_proj_fn,
        has_ffn,
        use_gelu,
        residual_multiplier,
        norm_eps,
        d_state_i,
        d_conv_i,
        n_heads_i,
        d_head_i,
        n_groups_i,
        gate_up_layout,
        conv_state,
        ssm_state
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}

void* mlx_lazy_mamba_block_quantized(
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
) {
    mlx_count_op();
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_w);
    const auto& in_proj_s = *static_cast<const array*>(in_s);
    const auto& in_proj_b = *static_cast<const array*>(in_b);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto a_log_arr = ensure_float32(*static_cast<const array*>(a_log));
    const auto d_skip_arr = ensure_float32(*static_cast<const array*>(d_skip));
    const auto* dt_b = static_cast<const array*>(dt_bias);
    const auto* norm_w = static_cast<const array*>(norm_weight);
    const auto& out_proj_w = *static_cast<const array*>(out_w);
    const auto& out_proj_s = *static_cast<const array*>(out_s);
    const auto& out_proj_b = *static_cast<const array*>(out_b);
    const auto* ln2_w = static_cast<const array*>(ln2_weight);
    const auto* gate_up_w_arr = static_cast<const array*>(gate_up_w);
    const auto* gate_up_s_arr = static_cast<const array*>(gate_up_s);
    const auto* gate_up_b_arr = static_cast<const array*>(gate_up_b);
    const auto* down_w_arr = static_cast<const array*>(down_w);
    const auto* down_s_arr = static_cast<const array*>(down_s);
    const auto* down_b_arr = static_cast<const array*>(down_b);

    const int d_state_i = static_cast<int>(d_state);
    const int d_conv_i = static_cast<int>(d_conv);
    const int n_heads_i = static_cast<int>(n_heads);
    const int d_head_i = static_cast<int>(d_head);
    const int n_groups_i = static_cast<int>(n_groups);
    const int d_inner = n_heads_i * d_head_i;
    const int xbc_len = d_inner + 2 * n_groups_i * d_state_i;
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    const array conv_kernel_broadcast = resolve_mamba_conv_kernel(conv_w, d_conv_i, xbc_len);
    const std::optional<array> conv_bias_row = (conv_b != nullptr)
        ? std::optional<array>(
            (conv_b->ndim() == 2 && conv_b->shape(0) == 1 && conv_b->shape(1) == xbc_len)
                ? ensure_float32(*conv_b)
                : reshape(ensure_float32(*conv_b), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(ensure_float32(*dt_b), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(ensure_float32(*norm_w))
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(ensure_float32(*ln2_w))
        : std::nullopt;

    auto* cache = static_cast<MLXMambaCache*>(mamba_cache_ptr);
    MambaLayer* layer_state = nullptr;
    if (cache != nullptr && layer_idx < cache->layers.size()) {
        layer_state = &cache->layers[layer_idx];
    }

    array conv_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->conv_state == nullptr ||
            layer_state->conv_state->shape(1) != d_conv_i ||
            layer_state->conv_state->shape(2) != xbc_len;
        if (need_init) {
            delete layer_state->conv_state;
            layer_state->conv_state = new array(zeros({1, d_conv_i, xbc_len}, float32));
        }
        conv_state = *layer_state->conv_state;
    } else {
        conv_state = zeros({1, d_conv_i, xbc_len}, float32);
    }

    array ssm_state(0.0f, float32);
    if (layer_state != nullptr) {
        const bool need_init =
            layer_state->ssm_state == nullptr ||
            layer_state->ssm_state->shape(1) != n_heads_i ||
            layer_state->ssm_state->shape(2) != d_head_i ||
            layer_state->ssm_state->shape(3) != d_state_i;
        if (need_init) {
            delete layer_state->ssm_state;
            layer_state->ssm_state = new array(zeros({1, n_heads_i, d_head_i, d_state_i}, float32));
        }
        ssm_state = *layer_state->ssm_state;
    } else {
        ssm_state = zeros({1, n_heads_i, d_head_i, d_state_i}, float32);
    }

    const auto in_proj_fn = [&in_proj_w, &in_proj_s, &in_proj_b, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, in_proj_w, in_proj_s, in_proj_b, group_size_i, bits_i);
    };
    const auto out_proj_fn = [&out_proj_w, &out_proj_s, &out_proj_b, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, out_proj_w, out_proj_s, out_proj_b, group_size_i, bits_i);
    };
    const bool has_ffn =
        gate_up_w_arr != nullptr && gate_up_s_arr != nullptr && gate_up_b_arr != nullptr &&
        down_w_arr != nullptr && down_s_arr != nullptr && down_b_arr != nullptr &&
        ln2_w != nullptr;
    const auto gate_up_fn = [&gate_up_w_arr, &gate_up_s_arr, &gate_up_b_arr, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, *gate_up_w_arr, *gate_up_s_arr, *gate_up_b_arr, group_size_i, bits_i);
    };
    const auto down_proj_fn = [&down_w_arr, &down_s_arr, &down_b_arr, group_size_i, bits_i](const array& x) -> array {
        return quantized_linear_no_bias(x, *down_w_arr, *down_s_arr, *down_b_arr, group_size_i, bits_i);
    };

    array out = mamba_forward_core(
        input_arr,
        ln1_w,
        in_proj_fn,
        conv_kernel_broadcast,
        conv_bias_row,
        a_log_arr,
        d_skip_arr,
        dt_bias_row,
        norm_weight_arr,
        out_proj_fn,
        ln2_weight_arr,
        gate_up_fn,
        down_proj_fn,
        has_ffn,
        use_gelu,
        residual_multiplier,
        norm_eps,
        d_state_i,
        d_conv_i,
        n_heads_i,
        d_head_i,
        n_groups_i,
        gate_up_layout,
        conv_state,
        ssm_state
    );

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
        *layer_state->ssm_state = ssm_state;
    }

    return pool_array(out);
}
