// bridge_ops.inc — Quantized linear dispatch, elementary ops (linear, silu, gelu_approx),
// custom Metal kernels (rmsnorm_gated, conv1d_decode_silu, gated_delta), rope helpers.

array linear_decode_maybe_quantized(
    bool decode_qmm_enabled,
    int quant_group_size,
    int quant_bits,
    const array& x,
    const array& rhs,
    bool has_q,
    const array& q_w,
    const array& q_scales,
    const array& q_biases,
    bool prefill_only_qmm = false,
    int layer_idx = -1,
    const char* op_tag = nullptr,
    const array* prefill_rhs = nullptr
) {
    const auto quant_compare_enabled = []() -> bool {
        static const bool cached = env_truthy("TALU_METAL_QMM_COMPARE", false);
        return cached;
    };
    const auto quant_compare_layer = []() -> int {
        if (const char* raw = std::getenv("TALU_METAL_QMM_COMPARE_LAYER")) {
            const int parsed = std::atoi(raw);
            return parsed > 0 ? parsed - 1 : parsed;
        }
        return 0;
    };
    const auto quant_compare_op = []() -> std::string {
        if (const char* raw = std::getenv("TALU_METAL_QMM_COMPARE_OP")) return std::string(raw);
        return std::string();
    };
    const auto should_compare_quant = [&](bool q_path_active) -> bool {
        if (!q_path_active || !quant_compare_enabled()) return false;
        if (rhs.size() <= 1 || rhs.ndim() < 2) return false;
        const int target_layer = quant_compare_layer();
        if (layer_idx >= 0 && target_layer >= 0 && layer_idx != target_layer) return false;
        const std::string target_op = quant_compare_op();
        if (target_op.empty()) return true;
        if (op_tag == nullptr) return false;
        return target_op == op_tag;
    };
    const auto emit_quant_compare = [&](const array& quant_out) {
        const array quant_f32 = astype(quant_out, float32);
        const array dense_f32 = astype(matmul(x, rhs), float32);
        const array diff_f32 = abs(quant_f32 - dense_f32);
        eval({quant_f32, dense_f32, diff_f32});
        synchronize();
        const float* diff_ptr = diff_f32.data<float>();
        const size_t n = static_cast<size_t>(diff_f32.size());
        if (diff_ptr == nullptr || n == 0) return;
        double sum = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float v = diff_ptr[i];
            sum += static_cast<double>(v);
            if (v > max_abs) max_abs = v;
        }
        const double mean_abs = sum / static_cast<double>(n);
        std::fprintf(
            stderr,
            "[mlx][qmm_compare] layer=%d op=%s elems=%zu max_abs=%.6f mean_abs=%.6f q_dtype=%d rhs_dtype=%d x_shape=[%d,%d,%d]\n",
            layer_idx,
            op_tag ? op_tag : "(null)",
            n,
            static_cast<double>(max_abs),
            mean_abs,
            static_cast<int>(static_cast<Dtype::Val>(quant_out.dtype())),
            static_cast<int>(static_cast<Dtype::Val>(rhs.dtype())),
            x.ndim() > 0 ? x.shape(0) : 0,
            x.ndim() > 1 ? x.shape(1) : 0,
            x.ndim() > 2 ? x.shape(2) : 0
        );
    };

    const bool is_prefill_shape = x.ndim() >= 3 && x.shape(1) > 1;
    if (is_prefill_shape && prefill_rhs != nullptr && prefill_rhs->size() > 1 && prefill_rhs->ndim() == 2) {
        return matmul(x, *prefill_rhs);
    }
    const bool allow_qmm = !prefill_only_qmm || is_prefill_shape;
    const bool q_ready = has_q && decode_qmm_enabled && allow_qmm && q_w.size() > 0 && q_scales.size() > 0;
    if (q_ready) {
        const bool affine_ready = q_biases.ndim() == 2 && q_biases.size() > 1 &&
            quant_group_size > 0 && (quant_bits == 4 || quant_bits == 8);
        const bool nvfp4_ready =
            !affine_ready &&
            q_w.dtype() == uint32 &&
            q_scales.dtype() == uint8 &&
            q_biases.size() == 1 &&
            q_biases.dtype() == float32 &&
            quant_group_size == 16 &&
            quant_bits == 4;
        if (affine_ready) {
            const array out = quantized_matmul(
                x,
                q_w,
                q_scales,
                std::optional<array>(q_biases),
                true,
                quant_group_size,
                quant_bits,
                "affine"
            );
            if (should_compare_quant(true)) emit_quant_compare(out);
            return out;
        }
        if (nvfp4_ready) {
            const bool reshape_back = x.ndim() == 3;
            const int batch = x.shape(0);
            const int seq = reshape_back ? x.shape(1) : 1;
            const array x2d = reshape_back ? reshape(x, {batch * seq, x.shape(2)}) : x;
            const array out2d = qqmm(
                x2d,
                q_w,
                std::optional<array>(q_scales),
                quant_group_size,
                quant_bits,
                "nvfp4",
                std::optional<array>(nvfp4_unit_global_scale()),
                std::optional<array>(q_biases)
            );
            const array out = reshape_back ? reshape(out2d, {batch, seq, out2d.shape(1)}) : out2d;
            if (should_compare_quant(true)) emit_quant_compare(out);
            return out;
        }
        const array out = quantized_matmul(
            x,
            q_w,
            q_scales,
            std::nullopt,
            true,
            std::nullopt,
            std::nullopt,
            "mxfp8"
        );
        if (should_compare_quant(true)) emit_quant_compare(out);
        return out;
    }
    if (rhs.size() <= 1 || rhs.ndim() < 2) {
        throw std::runtime_error("linear_decode_maybe_quantized: dense rhs is missing while quantized path is unavailable");
    }
    return matmul(x, rhs);
}

array linear(const array& x, const array& rhs) {
    return matmul(x, rhs);
}

array silu(const array& x) {
    return x * sigmoid(x);
}

array rmsnorm_gated_fallback(const array& hidden_states, const array& gate, const array& weight, float eps) {
    const array normed = fast::rms_norm(hidden_states, std::optional<array>(weight), eps);
    const array mixed_f32 = silu(astype(gate, float32)) * astype(normed, float32);
    return astype(mixed_f32, hidden_states.dtype());
}

static inline fast::CustomKernelFunction& rmsnorm_gated_kernel() {
    static fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_rmsnorm_gated",
        { "hidden", "gate", "norm_weight", "eps_in" },
        { "out" },
        R"(
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.z;
    if (row >= ROWS) return;

    const uint row_base = row * DV;
    const uint head_idx = row % N_HEADS;
    const uint weight_base = (WEIGHT_PER_HEAD == 1) ? (head_idx * DV) : 0;

    float sum_sq_local = 0.0f;
    for (uint i = lane; i < DV; i += 32) {
      const float x = static_cast<float>(hidden[row_base + i]);
      sum_sq_local += x * x;
    }
    const float sum_sq = simd_sum(sum_sq_local);
    const float rms_scale = rsqrt((sum_sq / static_cast<float>(DV)) + eps_in);

    for (uint i = lane; i < DV; i += 32) {
      const float x = static_cast<float>(hidden[row_base + i]);
      const float z = static_cast<float>(gate[row_base + i]);
      const float w = static_cast<float>(norm_weight[weight_base + i]);
      const float sig = 1.0f / (1.0f + exp(-z));
      const float mixed = (x * rms_scale) * w * (z * sig);
      using OutT = remove_reference_t<decltype(out[0])>;
      out[row_base + i] = static_cast<OutT>(mixed);
    }
        )",
        "",
        true,
        false
    );
    return kernel;
}

array gelu_approx(const array& x) {
    const array coeff = array(0.044715f, x.dtype());
    const array sqrt_two_over_pi = array(0.7978845608f, x.dtype());
    const array half = array(0.5f, x.dtype());
    const array one = array(1.0f, x.dtype());
    const array x3 = x * x * x;
    const array inner = sqrt_two_over_pi * (x + coeff * x3);
    return half * x * (one + tanh(inner));
}

array prepare_depthwise_conv1d_decode_weight(
    const array& weight,
    int conv_dim,
    int kernel_size,
    const std::string& name
) {
    if (weight.ndim() != 3 ||
        weight.shape(0) != conv_dim ||
        weight.shape(1) != kernel_size ||
        weight.shape(2) != 1)
    {
        throw std::runtime_error("invalid depthwise conv1d weight shape: " + name);
    }
    const array decode_weight = contiguous(reshape(transpose(weight, {2, 1, 0}), {1, kernel_size, conv_dim}));
    eval(decode_weight);
    return decode_weight;
}

// Fused Metal kernel for depthwise conv1d decode + SiLU.
// Replaces 6 separate kernels (mul, sum, mul, add, concat, silu) with 1.
// Writes both the conv+silu output and the shifted cache state.
static inline fast::CustomKernelFunction& conv1d_decode_silu_kernel() {
    static fast::CustomKernelFunction kernel = fast::metal_kernel(
        "talu_conv1d_decode_silu",
        { "cache_state", "token_input", "weight" },
        { "conv_out", "next_state" },
        R"(
    const uint c = thread_position_in_grid.x;
    if (c >= CONV_DIM) return;

    // Depthwise conv: dot product across kernel positions.
    float acc = 0.0f;
    for (uint k = 0; k < KERNEL_SIZE - 1; ++k) {
        acc += float(cache_state[k * CONV_DIM + c]) *
               float(weight[k * CONV_DIM + c]);
    }
    acc += float(token_input[c]) *
           float(weight[(KERNEL_SIZE - 1) * CONV_DIM + c]);

    // SiLU activation: x * sigmoid(x)
    const float sig = 1.0f / (1.0f + exp(-acc));
    // Use the same type as next_state to match output dtype.
    using OutT = remove_reference_t<decltype(next_state[0])>;
    conv_out[c] = static_cast<OutT>(acc * sig);

    // Shift cache: drop oldest, append new token.
    for (uint k = 0; k < KERNEL_SIZE - 2; ++k) {
        next_state[k * CONV_DIM + c] = cache_state[(k + 1) * CONV_DIM + c];
    }
    next_state[(KERNEL_SIZE - 2) * CONV_DIM + c] = token_input[c];
        )",
        "",
        true,
        false
    );
    return kernel;
}

std::pair<array, std::optional<array>> depthwise_conv1d_decode_step(
    const std::optional<array>& cache_state_opt,
    const array& token_input,
    const array& decode_weight,
    int kernel_size,
    int conv_dim,
    const std::string& name
) {
    if (token_input.ndim() != 3 || token_input.shape(1) != 1 || token_input.shape(2) != conv_dim) {
        throw std::runtime_error("depthwise conv decode step expects token input [B,1,C]: " + name);
    }
    if (decode_weight.ndim() != 3 ||
        decode_weight.shape(0) != 1 ||
        decode_weight.shape(1) != kernel_size ||
        decode_weight.shape(2) != conv_dim)
    {
        throw std::runtime_error("depthwise conv decode weight mismatch: " + name);
    }

    const int batch = token_input.shape(0);
    if (kernel_size <= 0) {
        throw std::runtime_error("depthwise conv decode kernel must be positive: " + name);
    }

    array conv_out = array(0.0f);
    std::optional<array> next_state = std::nullopt;
    if (kernel_size == 1) {
        conv_out = token_input * decode_weight;
        return {conv_out, next_state};
    }

    if (!cache_state_opt.has_value()) {
        throw std::runtime_error("depthwise conv decode step requires cache state: " + name);
    }
    const array& cache_state = *cache_state_opt;
    if (cache_state.ndim() != 3 ||
        cache_state.shape(0) != batch ||
        cache_state.shape(1) != kernel_size - 1 ||
        cache_state.shape(2) != conv_dim)
    {
        throw std::runtime_error("depthwise conv decode cache shape mismatch: " + name);
    }

    // Use the fused Metal kernel for batch=1 decode with kernel_size > 1.
    if (batch == 1 && kernel_size > 1) {
        const array flat_cache = reshape(cache_state, {kernel_size - 1, conv_dim});
        const array flat_input = reshape(token_input, {conv_dim});
        const array flat_weight = reshape(decode_weight, {kernel_size, conv_dim});

        auto outputs = conv1d_decode_silu_kernel()(
            { flat_cache, flat_input, flat_weight },
            { Shape{conv_dim}, Shape{kernel_size - 1, conv_dim} },
            { token_input.dtype(), cache_state.dtype() },
            { conv_dim, 1, 1 },
            { std::min(conv_dim, 256), 1, 1 },
            {
                { "CONV_DIM", conv_dim },
                { "KERNEL_SIZE", kernel_size },
            },
            std::nullopt,
            false,
            {}
        );
        conv_out = reshape(outputs[0], {1, 1, conv_dim});
        next_state = reshape(outputs[1], {1, kernel_size - 1, conv_dim});
        return {conv_out, next_state};
    }

    // Fallback for batch > 1 or kernel_size == 1.
    const array prev_weights = slice(decode_weight, {0, 0, 0}, {1, kernel_size - 1, conv_dim});
    const array curr_weight = slice(decode_weight, {0, kernel_size - 1, 0}, {1, kernel_size, conv_dim});
    const array prev_term = sum(cache_state * prev_weights, 1);
    const array curr_term = reshape(token_input * curr_weight, {batch, conv_dim});
    conv_out = reshape(prev_term + curr_term, {batch, 1, conv_dim});

    const array shifted = slice(cache_state, {0, 1, 0}, {batch, kernel_size - 1, conv_dim});
    next_state = concatenate({shifted, token_input}, 1);
    return {conv_out, next_state};
}

// Array-based RoPE that uses pre-computed frequency tables.
// Unlike fast::rope (which takes int offset), the offset is an array input,
// making this compilable — the compiled function can be reused across decode
// steps without retracing.
//
// x: [B, n_heads, S, head_dim] (already transposed)
// freq: [rope_dims/2] pre-computed 1/theta^(2d/rope_dims)
// offset_arr: scalar int32 array (current position offset)
// rope_dims: number of dimensions to rotate
array apply_rope_array(
    const array& x,
    const array& freq,
    const array& offset_arr,
    int rope_dims
) {
    const int B = x.shape(0);
    const int nh = x.shape(1);
    const int S = x.shape(2);
    const int hd = x.shape(3);
    const int half = rope_dims / 2;

    // Compute positions: [1, 1, S, 1]
    // For S=1 (decode): just the offset
    // For S>1 (prefill): offset + arange(S)
    array positions = reshape(astype(offset_arr, float32), {1, 1, 1, 1});
    if (S > 1) {
        positions = positions + reshape(astype(arange(0, S, 1, int32), float32), {1, 1, S, 1});
    }

    // angles: [1, 1, S, half]
    const array freq_bcast = reshape(freq, {1, 1, 1, half});
    const array angle = positions * freq_bcast;
    const array cos_a = cos(angle);
    const array sin_a = sin(angle);

    // Split x into first half and second half of rotary dims
    const array x_first = slice(x, {0, 0, 0, 0}, {B, nh, S, half});
    const array x_second = slice(x, {0, 0, 0, half}, {B, nh, S, rope_dims});

    const array y_first = x_first * cos_a - x_second * sin_a;
    const array y_second = x_first * sin_a + x_second * cos_a;

    if (rope_dims == hd) {
        return concatenate({y_first, y_second}, 3);
    }
    // Partial rotation: pass through the unrotated tail
    const array pass = slice(x, {0, 0, 0, rope_dims}, {B, nh, S, hd});
    return concatenate({y_first, y_second, pass}, 3);
}

// Proportional full-head RoPE: rotates only the leading `rope_dims/2`
// pairs inside a full `head_dim`-wide pairing domain.  Uses array-based
// RoPE internally so the offset flows through an array input.
array apply_rope_proportional_array(
    const array& t,
    const array& freq,
    const array& offset_arr,
    int rope_dims,
    int head_dim
) {
    const int B = t.shape(0);
    const int nh = t.shape(1);
    const int S = t.shape(2);
    const int half_rope = rope_dims / 2;
    const int half_hd = head_dim / 2;
    // Gather the active first-half and matching second-half slices.
    array t_first = slice(t, {0,0,0,0},                    {B,nh,S,half_rope});
    array t_pass1 = slice(t, {0,0,0,half_rope},             {B,nh,S,half_hd});
    array t_third = slice(t, {0,0,0,half_hd},               {B,nh,S,half_hd+half_rope});
    array t_pass2 = slice(t, {0,0,0,half_hd+half_rope},     {B,nh,S,head_dim});
    array rope_in = concatenate({t_first, t_third}, 3);
    rope_in = apply_rope_array(rope_in, freq, offset_arr, rope_dims);
    array first_rot = slice(rope_in, {0,0,0,0},        {B,nh,S,half_rope});
    array third_rot = slice(rope_in, {0,0,0,half_rope}, {B,nh,S,rope_dims});
    return concatenate({first_rot, t_pass1, third_rot, t_pass2}, 3);
}

array single_query_grouped_attention(
    const array& queries,
    const array& keys,
    const array& values,
    int num_heads,
    int num_kv_heads,
    float scale,
    bool force_f32
) {
    if (queries.ndim() != 4 || keys.ndim() != 4 || values.ndim() != 4) {
        throw std::runtime_error("single-query attention expects rank-4 tensors");
    }
    if (queries.shape(2) != 1) {
        throw std::runtime_error("single-query attention expects query length of 1");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || (num_heads % num_kv_heads) != 0) {
        throw std::runtime_error("single-query attention invalid grouped-query head config");
    }

    array q_use = queries;
    array k_use = keys;
    array v_use = values;
    const int repeat_factor = num_heads / num_kv_heads;
    if (repeat_factor > 1) {
        k_use = repeat(k_use, repeat_factor, 1);
        v_use = repeat(v_use, repeat_factor, 1);
    }

    if (force_f32) {
        q_use = astype(q_use, float32);
        k_use = astype(k_use, float32);
        v_use = astype(v_use, float32);
    }

    const array scores = matmul(q_use, transpose(k_use, {0, 1, 3, 2})) * array(scale, q_use.dtype());
    const array weights = softmax(scores, -1, true);
    const array out = matmul(weights, v_use);
    return force_f32 ? astype(out, queries.dtype()) : out;
}

array softplus(const array& x) {
    const array ax = abs(x);
    const array maxv = maximum(x, array(0.0f));
    return log1p(exp(-ax)) + maxv;
}

array compute_g(const array& A_exp_f32, const array& a, const array& dt_bias) {
    static const auto compiled_compute_g = compile(
        [](const std::vector<array>& inputs) {
            const array& a_exp = inputs[0];
            const array& a_in = inputs[1];
            const array& dt_in = inputs[2];
            const array a_plus = astype(a_in + dt_in, float32);
            const array decay = exp(-(a_exp * softplus(a_plus)));
            return std::vector<array>{astype(decay, a_in.dtype())};
        },
        true
    );
    return compiled_compute_g({A_exp_f32, a, dt_bias}).at(0);
}

float scalar_to_f32(const array& value, const std::string& name) {
    if (value.size() <= 0) {
        throw std::runtime_error("invalid scalar tensor: " + name);
    }
    array flat = reshape(astype(value, float32), {1});
    eval(flat);
    const float* ptr = flat.data<float>();
    if (ptr == nullptr) {
        throw std::runtime_error("failed to read scalar tensor: " + name);
    }
    return ptr[0];
}

array last_token_logits(const array& logits) {
    if (logits.ndim() != 3) throw std::runtime_error("expected logits rank-3 [B,T,V]");
    const int b = logits.shape(0);
    const int t = logits.shape(1);
    const int v = logits.shape(2);
    if (t <= 0) throw std::runtime_error("logits have zero sequence length");
    return slice(logits, {0, t - 1, 0}, {b, t, v});
}

array take_time_4d(const array& a, int t) {
    const int b = a.shape(0);
    const int h = a.shape(2);
    const int d = a.shape(3);
    return reshape(slice(a, {0, t, 0, 0}, {b, t + 1, h, d}), {b, h, d});
}

array take_time_3d(const array& a, int t) {
    const int b = a.shape(0);
    const int h = a.shape(2);
    return reshape(slice(a, {0, t, 0}, {b, t + 1, h}), {b, h});
}

std::pair<array, array> gated_delta_update_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    const int B = q.shape(0);
    const int T = q.shape(1);
    const int Hk = q.shape(2);
    const int Dk = q.shape(3);
    const int Hv = v.shape(2);
    const int Dv = v.shape(3);

    array state = state_opt.has_value() ? *state_opt : zeros({B, Hv, Dv, Dk}, q.dtype());

    array q_use = q;
    array k_use = k;
    if (Hv % Hk != 0) {
        throw std::runtime_error("gated-delta head mismatch: Hv must be divisible by Hk");
    }
    const int repeat_factor = Hv / Hk;
    if (repeat_factor > 1) {
        q_use = repeat(q_use, repeat_factor, 2);
        k_use = repeat(k_use, repeat_factor, 2);
    }

    const array beta = sigmoid(b);
    const array g = compute_g(A_exp_f32, a, dt_bias);

    std::vector<array> ys;
    ys.reserve(static_cast<size_t>(T));

    for (int t = 0; t < T; ++t) {
        const array q_t = take_time_4d(q_use, t);     // [B, Hv, Dk]
        const array k_t = take_time_4d(k_use, t);     // [B, Hv, Dk]
        const array v_t = take_time_4d(v, t);         // [B, Hv, Dv]
        const array beta_t = take_time_3d(beta, t);   // [B, Hv]
        const array g_t = take_time_3d(g, t);         // [B, Hv]

        const array k_exp = reshape(k_t, {B, Hv, 1, Dk});
        const array q_exp = reshape(q_t, {B, Hv, 1, Dk});

        state = state * reshape(g_t, {B, Hv, 1, 1});
        const array kv_mem = sum(state * k_exp, std::vector<int>{-1}, false); // [B, Hv, Dv]
        const array delta = (v_t - kv_mem) * reshape(beta_t, {B, Hv, 1});
        state = state + k_exp * reshape(delta, {B, Hv, Dv, 1});

        const array y_t = sum(state * q_exp, std::vector<int>{-1}, false); // [B, Hv, Dv]
        ys.push_back(y_t);
    }

    const array y = stack(ys, 1); // [B, T, Hv, Dv]
    return {y, state};
}

// Extended decode-only SSM kernel: folds sigmoid(b) + compute_g +
// Q/K RMS-norm+scale + recurrence into one Metal dispatch.
// Replaces ~4 kernel dispatches per SSM layer for S=1 decode.
//
// Threading: grid (32, Dv, B*Hv), threadgroup (32, 4, 1).
// The 32 threads cover Dk via simd_sum — same as the base kernel.
// Q/K norms reduce across Dk which matches this layout.
// Output RMS norm needs cross-Dv reduction and stays external.
const std::string& gated_delta_fused_kernel_source() {
    static const std::string source = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        constexpr int n_per_t = Dk / 32;
        constexpr float eps = 1.0e-6f;

        // q_raw, k_raw: [B, T, Hqk, Dk] — NOT yet normalized
        auto q_ = q_raw + b_idx * T * Hqk * Dk + hv_idx * Dk;
        auto k_ = k_raw + b_idx * T * Hqk * Dk + hv_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        }

        auto b_raw_ = b_raw + b_idx * T * Hv;
        auto a_raw_ = a_raw + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          // Compute decay and beta from raw inputs (replaces 2 dispatches).
          const float a_val = static_cast<float>(a_raw_[hv_idx]) + dt_bias[hv_idx];
          const float a_log_exp_val = a_log_exp[hv_idx];
          const float softplus_a = a_val > 20.0f ? a_val : log(1.0f + exp(a_val));
          const float decay = exp(-a_log_exp_val * softplus_a);
          const float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_raw_[hv_idx])));

          // Load raw Q/K values and compute RMS norm + scale inline
          // (replaces 2 compiled norm+scale dispatches).
          float q_vals[n_per_t];
          float k_vals[n_per_t];
          float q_sq_local = 0.0f;
          float k_sq_local = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            const float qv = static_cast<float>(q_[n_per_t * dk_idx + i]);
            const float kv = static_cast<float>(k_[n_per_t * dk_idx + i]);
            q_vals[i] = qv;
            k_vals[i] = kv;
            q_sq_local += qv * qv;
            k_sq_local += kv * kv;
          }
          const float q_sq = simd_sum(q_sq_local);
          const float k_sq = simd_sum(k_sq_local);
          const float q_scale = rsqrt(q_sq + eps) * rsqrt((float)Dk);
          const float k_scale = rsqrt(k_sq + eps);
          for (int i = 0; i < n_per_t; ++i) {
            q_vals[i] *= q_scale;
            k_vals[i] *= k_scale;
          }

          // SSM recurrence (same as base kernel).
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] *= decay;
            kv_mem += state[i] * k_vals[i];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_val;

          float ssm_out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] += k_vals[i] * delta;
            ssm_out += state[i] * q_vals[i];
          }
          ssm_out = simd_sum(ssm_out);

          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(ssm_out);
          }

          q_ += Hqk * Dk;
          k_ += Hqk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          b_raw_ += Hv;
          a_raw_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        }
    )";
    return source;
}

const std::string& gated_delta_kernel_source() {
    static const std::string source = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hqk, Dk]
        auto q_ = q + b_idx * T * Hqk * Dk + hv_idx * Dk;
        auto k_ = k + b_idx * T * Hqk * Dk + hv_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float q_vals[n_per_t];
          float k_vals[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            q_vals[i] = static_cast<float>(q_[s_idx]);
            k_vals[i] = static_cast<float>(k_[s_idx]);
          }

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_vals[i];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] + k_vals[i] * delta;
            out += state[i] * q_vals[i];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          // increment pointers to next timestep
          q_ += Hqk * Dk;
          k_ += Hqk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    )";
    return source;
}

std::pair<array, array> gated_delta_update_kernel(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    const int B = q.shape(0);
    const int T = q.shape(1);
    const int Hk = q.shape(2);
    const int Dk = q.shape(3);
    const int Hv = v.shape(2);
    const int Dv = v.shape(3);

    if (Dk % 32 != 0) {
        throw std::runtime_error("gated-delta kernel requires Dk divisible by 32");
    }

    array state = state_opt.has_value() ? *state_opt : zeros({B, Hv, Dv, Dk}, q.dtype());
    array q_use = q;
    array k_use = k;

    if (Hv % Hk != 0) {
        throw std::runtime_error("gated-delta head mismatch: Hv must be divisible by Hk");
    }
    const int repeat_factor = Hv / Hk;
    if (repeat_factor > 1) {
        q_use = repeat(q_use, repeat_factor, 2);
        k_use = repeat(k_use, repeat_factor, 2);
    }

    const array t_scalar(static_cast<int32_t>(T), int32);

    // For single-token decode, use the fused kernel that computes
    // sigmoid(b) and decay internally — saves 2 kernel dispatches.
    // For prefill (T > 1), use the original kernel with pre-computed
    // beta/g to avoid redundant per-position transcendentals.
    if (T == 1) {
        static const fast::CustomKernelFunction fused_kernel = fast::metal_kernel(
            "gated_delta_step_fused",
            {"q_raw", "k_raw", "v", "b_raw", "a_raw", "a_log_exp", "dt_bias", "state_in", "T"},
            {"y", "state_out"},
            gated_delta_fused_kernel_source()
        );
        const array a_log_exp_flat = reshape(A_exp_f32, {Hv});
        const array dt_bias_flat = reshape(dt_bias, {Hv});
        const std::vector<array> outputs = fused_kernel(
            {q_use, k_use, v,
             reshape(b, {B, T, Hv}), reshape(a, {B, T, Hv}),
             a_log_exp_flat, dt_bias_flat, state, t_scalar},
            {Shape{B, T, Hv, Dv}, state.shape()},
            {q.dtype(), q.dtype()},
            std::make_tuple(32, Dv, B * Hv),
            std::make_tuple(32, 4, 1),
            {
                {"InT", q.dtype()},
                {"Dk", Dk},
                {"Dv", Dv},
                {"Hqk", q_use.shape(2)},
                {"Hv", Hv},
            },
            std::nullopt,
            false,
            {}
        );
        return {outputs.at(0), outputs.at(1)};
    }

    const array beta = sigmoid(b);
    const array g = compute_g(A_exp_f32, a, dt_bias);

    static const fast::CustomKernelFunction kernel = fast::metal_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        gated_delta_kernel_source()
    );

    const std::vector<array> outputs = kernel(
        {q_use, k_use, v, g, beta, state, t_scalar},
        {Shape{B, T, Hv, Dv}, state.shape()},
        {q.dtype(), q.dtype()},
        std::make_tuple(32, Dv, B * Hv),
        std::make_tuple(32, 4, 1),
        {
            {"InT", q.dtype()},
            {"Dk", Dk},
            {"Dv", Dv},
            {"Hqk", q_use.shape(2)},
            {"Hv", Hv},
        },
        std::nullopt,
        false,
        {}
    );
    return {outputs.at(0), outputs.at(1)};
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& a,
    const array& b,
    const array& A_exp_f32,
    const array& dt_bias,
    const std::optional<array>& state_opt
) {
    if (metal::is_available() && default_device() == Device::gpu && q.shape(3) % 32 == 0) {
        return gated_delta_update_kernel(q, k, v, a, b, A_exp_f32, dt_bias, state_opt);
    }
    return gated_delta_update_fallback(q, k, v, a, b, A_exp_f32, dt_bias, state_opt);
}

array rmsnorm_gated(const array& hidden_states, const array& gate, const array& weight, float eps) {
    const bool can_use_kernel =
        metal::is_available() &&
        default_device() == Device::gpu &&
        hidden_states.ndim() == 4 &&
        gate.ndim() == 4 &&
        hidden_states.shape(0) == gate.shape(0) &&
        hidden_states.shape(1) == gate.shape(1) &&
        hidden_states.shape(2) == gate.shape(2) &&
        hidden_states.shape(3) == gate.shape(3);
    if (!can_use_kernel) {
        return rmsnorm_gated_fallback(hidden_states, gate, weight, eps);
    }

    const int B = hidden_states.shape(0);
    const int S = hidden_states.shape(1);
    const int Hv = hidden_states.shape(2);
    const int Dv = hidden_states.shape(3);
    if (B <= 0 || S <= 0 || Hv <= 0 || Dv <= 0) {
        return rmsnorm_gated_fallback(hidden_states, gate, weight, eps);
    }

    bool per_head_weight = false;
    array weight_use = weight;
    if (weight.ndim() == 1) {
        if (weight.shape(0) == Dv) {
            per_head_weight = false;
        } else if (weight.shape(0) == (Hv * Dv)) {
            per_head_weight = true;
            weight_use = reshape(weight, {Hv, Dv});
        } else {
            return rmsnorm_gated_fallback(hidden_states, gate, weight, eps);
        }
    } else if (weight.ndim() == 2 && weight.shape(0) == Hv && weight.shape(1) == Dv) {
        per_head_weight = true;
    } else {
        return rmsnorm_gated_fallback(hidden_states, gate, weight, eps);
    }

    const int rows = B * S * Hv;
    const array hidden_2d = reshape(hidden_states, {rows, Dv});
    const array gate_2d = reshape(gate, {rows, Dv});
    const array weight_flat = per_head_weight
        ? reshape(weight_use, {Hv * Dv})
        : reshape(weight_use, {Dv});
    const array eps_arr(eps, float32);

    const std::vector<array> outputs = rmsnorm_gated_kernel()(
        { hidden_2d, gate_2d, weight_flat, eps_arr },
        { Shape{rows, Dv} },
        { hidden_states.dtype() },
        std::make_tuple(32, 1, rows),
        std::make_tuple(32, 1, 1),
        {
            { "DV", Dv },
            { "ROWS", rows },
            { "N_HEADS", Hv },
            { "WEIGHT_PER_HEAD", per_head_weight ? 1 : 0 },
        },
        std::nullopt,
        false,
        {}
    );
    if (outputs.size() != 1) {
        throw std::runtime_error("rmsnorm_gated kernel returned unexpected output count");
    }
    return reshape(outputs[0], {B, S, Hv, Dv});
}
