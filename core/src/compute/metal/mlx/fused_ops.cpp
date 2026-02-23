// MLX Bridge - Fused Neural Network Operations
//
// High-level operations that combine multiple MLX calls for efficiency.
// Uses MLX fast:: kernels which are highly optimized Metal implementations.

#include "compute_common.h"
#include "model_state.h"

extern "C" {

void mlx_cache_update_and_fetch_bfloat16(
    void* cache_ptr,
    size_t layer_idx,
    const void* k_new,
    const void* v_new,
    void** k_out,
    void** v_out,
    bool* is_prefill_out
);

// ============================================================================
// ShortConv macro operations (inference-owned)
// ============================================================================

void* mlx_lazy_shortconv_mixer_bf16(
    const void* input,
    const void* in_proj,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_proj,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_proj_arr = *static_cast<const array*>(in_proj);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto& out_proj_arr = *static_cast<const array*>(out_proj);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);

    // in_proj is stored as [3*conv_dim, d_model], so transpose for [d_model, 3*conv_dim].
    array bcx = matmul(input_arr, transpose(in_proj_arr));
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    if (conv_kernel.ndim() == 3) {
        // [conv_dim, 1, d_conv] -> [conv_dim, d_conv]
        conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
    }
    // [conv_dim, d_conv] -> [d_conv, conv_dim] for contiguous time-major access.
    array conv_kernel_t = astype(transpose(conv_kernel), float32);
    const array conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        conv_bias_row = reshape(astype(*conv_bias_arr, float32), {1, conv_dim_i});
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
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

    const array out_proj_t = transpose(out_proj_arr);
    std::vector<array> token_outputs;
    token_outputs.reserve(seq_len);

    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        const array bx_t = slice(bx, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i});

        if (d_conv_i > 1) {
            const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
            conv_state = concatenate({state_tail, bx_t}, 1);
        } else {
            conv_state = bx_t;
        }

        // Vectorized depthwise convolution over history window:
        // [1, d_conv, conv_dim] * [1, d_conv, conv_dim] -> sum(axis=1) => [1, conv_dim].
        array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
        if (conv_bias_row) {
            conv_t = conv_t + *conv_bias_row;
        }

        const array c_t = reshape(slice(c_gate, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i}), {1, conv_dim_i});
        const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
        token_outputs.push_back(matmul(gated, out_proj_t));
    }

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    if (token_outputs.size() == 1) {
        return pool_array(token_outputs[0]);
    }
    return pool_array(concatenate(token_outputs, 1));
}

void* mlx_lazy_shortconv_mixer_quantized(
    const void* input,
    const void* in_w,
    const void* in_s,
    const void* in_b,
    const void* conv_weight,
    const void* conv_bias,
    const void* out_w,
    const void* out_s,
    const void* out_b,
    size_t group_size,
    size_t bits,
    void* shortconv_cache_ptr,
    size_t layer_idx,
    size_t d_conv,
    size_t conv_dim
) {
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& in_w_arr = *static_cast<const array*>(in_w);
    const auto& in_s_arr = *static_cast<const array*>(in_s);
    const auto& in_b_arr = *static_cast<const array*>(in_b);
    const auto& conv_weight_arr = *static_cast<const array*>(conv_weight);
    const auto* conv_bias_arr = static_cast<const array*>(conv_bias);
    const auto& out_w_arr = *static_cast<const array*>(out_w);
    const auto& out_s_arr = *static_cast<const array*>(out_s);
    const auto& out_b_arr = *static_cast<const array*>(out_b);

    const int seq_len = input_arr.shape(1);
    const int d_conv_i = static_cast<int>(d_conv);
    const int conv_dim_i = static_cast<int>(conv_dim);
    const int group_size_i = static_cast<int>(group_size);
    const int bits_i = static_cast<int>(bits);

    array bcx = quantized_matmul(
        input_arr,
        in_w_arr,
        in_s_arr,
        in_b_arr,
        true,
        group_size_i,
        bits_i,
        "affine"
    );
    bcx = astype(bcx, float32);

    array b_gate = slice(bcx, {0, 0, 0}, {1, seq_len, conv_dim_i});
    array c_gate = slice(bcx, {0, 0, conv_dim_i}, {1, seq_len, 2 * conv_dim_i});
    array x_proj = slice(bcx, {0, 0, 2 * conv_dim_i}, {1, seq_len, 3 * conv_dim_i});
    array bx = b_gate * x_proj;

    array conv_kernel = conv_weight_arr;
    if (conv_kernel.ndim() == 3) {
        conv_kernel = reshape(conv_kernel, {conv_kernel.shape(0), conv_kernel.shape(2)});
    }
    array conv_kernel_t = astype(transpose(conv_kernel), float32);
    const array conv_kernel_broadcast = reshape(conv_kernel_t, {1, d_conv_i, conv_dim_i});

    std::optional<array> conv_bias_row;
    if (conv_bias_arr != nullptr) {
        conv_bias_row = reshape(astype(*conv_bias_arr, float32), {1, conv_dim_i});
    }

    auto* state_cache = static_cast<MLXShortConvCache*>(shortconv_cache_ptr);
    ShortConvLayer* layer_state = nullptr;
    if (state_cache != nullptr && layer_idx < state_cache->layers.size()) {
        layer_state = &state_cache->layers[layer_idx];
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

    std::vector<array> token_outputs;
    token_outputs.reserve(seq_len);

    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        const array bx_t = slice(bx, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i});

        if (d_conv_i > 1) {
            const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv_i, conv_dim_i});
            conv_state = concatenate({state_tail, bx_t}, 1);
        } else {
            conv_state = bx_t;
        }

        // Vectorized depthwise convolution over history window:
        // [1, d_conv, conv_dim] * [1, d_conv, conv_dim] -> sum(axis=1) => [1, conv_dim].
        array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
        if (conv_bias_row) {
            conv_t = conv_t + *conv_bias_row;
        }

        const array c_t = reshape(slice(c_gate, {0, token_idx, 0}, {1, token_idx + 1, conv_dim_i}), {1, conv_dim_i});
        const array gated = reshape(conv_t * c_t, {1, 1, conv_dim_i});
        token_outputs.push_back(quantized_matmul(
            gated,
            out_w_arr,
            out_s_arr,
            out_b_arr,
            true,
            group_size_i,
            bits_i,
            "affine"
        ));
    }

    if (layer_state != nullptr) {
        *layer_state->conv_state = conv_state;
    }

    if (token_outputs.size() == 1) {
        return pool_array(token_outputs[0]);
    }
    return pool_array(concatenate(token_outputs, 1));
}

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

    kernel = astype(kernel, float32);
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
    const int d_model = input.shape(2);
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

    std::vector<array> token_outputs;
    token_outputs.reserve(static_cast<size_t>(seq_len));

    for (int token_idx = 0; token_idx < seq_len; token_idx++) {
        const array token = slice(input, {0, token_idx, 0}, {1, token_idx + 1, d_model});
        const array norm1 = fast::rms_norm(token, ln1_weight, norm_eps);

        array proj = astype(in_proj_fn(norm1), float32);
        const int proj_dim = proj.shape(2);
        if (proj_dim < (2 * d_inner + 2 * bc_len + n_heads)) {
            throw std::invalid_argument("[mamba] in_proj output too small");
        }
        const array z = slice(proj, {0, 0, 0}, {1, 1, d_inner});
        const array xbc = slice(proj, {0, 0, d_inner}, {1, 1, d_inner + xbc_len});
        const array dt_raw = slice(proj, {0, 0, d_inner + xbc_len}, {1, 1, d_inner + xbc_len + n_heads});

        const array xbc_row = reshape(xbc, {1, xbc_len});
        const array xbc_new = reshape(xbc_row, {1, 1, xbc_len});
        if (d_conv > 1) {
            const array state_tail = slice(conv_state, {0, 1, 0}, {1, d_conv, xbc_len});
            conv_state = concatenate({state_tail, xbc_new}, 1);
        } else {
            conv_state = xbc_new;
        }

        array conv_t = sum(conv_state * conv_kernel_broadcast, 1);
        if (conv_bias_row) {
            conv_t = conv_t + *conv_bias_row;
        }
        const array xbc_conv = conv_t * sigmoid(conv_t);

        const array x_conv = slice(xbc_conv, {0, 0}, {1, d_inner});
        const array b_raw = slice(xbc_conv, {0, d_inner}, {1, d_inner + bc_len});
        const array c_raw = slice(xbc_conv, {0, d_inner + bc_len}, {1, d_inner + 2 * bc_len});

        array dt = reshape(dt_raw, {1, n_heads});
        if (dt_bias_row) {
            dt = dt + *dt_bias_row;
        }
        dt = log(1.0f + exp(dt));

        const array x_heads = reshape(x_conv, {1, n_heads, d_head});
        array b_heads = reshape(b_raw, {1, n_groups, d_state});
        array c_heads = reshape(c_raw, {1, n_groups, d_state});
        if (heads_per_group > 1) {
            b_heads = repeat(b_heads, heads_per_group, 1);
            c_heads = repeat(c_heads, heads_per_group, 1);
        }

        const array dt4 = reshape(dt, {1, n_heads, 1, 1});
        const array dA = exp((-exp(a_log_row)) * dt4);

        const array x_term = dt4 *
            reshape(x_heads, {1, n_heads, d_head, 1}) *
            reshape(b_heads, {1, n_heads, 1, d_state});

        ssm_state = dA * ssm_state + x_term;

        array y = sum(ssm_state * reshape(c_heads, {1, n_heads, 1, d_state}), 3);
        y = y + d_skip_row * x_heads;

        array ssm_out = reshape(y, {1, 1, d_inner});
        ssm_out = ssm_out * (z * sigmoid(z));
        if (norm_weight) {
            ssm_out = fast::rms_norm(ssm_out, *norm_weight, 1e-5f);
        }

        array mixer_out = astype(out_proj_fn(ssm_out), float32);
        array hidden_1 = token + mixer_out * residual_multiplier;

        array token_out = hidden_1;
        if (has_ffn) {
            if (!ln2_weight.has_value()) {
                throw std::invalid_argument("[mamba] missing ln2 weight for FFN path");
            }
            const array norm2 = fast::rms_norm(hidden_1, *ln2_weight, norm_eps);
            const array gate_up = astype(gate_up_fn(norm2), float32);
            const int gate_up_dim = gate_up.shape(2);
            if ((gate_up_dim % 2) != 0) {
                throw std::invalid_argument("[mamba] fused gate_up must have even width");
            }
            const int d_ff = gate_up_dim / 2;

            const array mid = [&]() -> array {
                if (gate_up_layout == 1) {
                    const array reshaped = reshape(gate_up, {1, 1, d_ff, 2});
                    const array gate = reshape(
                        slice(reshaped, {0, 0, 0, 0}, {1, 1, d_ff, 1}),
                        {1, 1, d_ff}
                    );
                    const array up = reshape(
                        slice(reshaped, {0, 0, 0, 1}, {1, 1, d_ff, 2}),
                        {1, 1, d_ff}
                    );
                    const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
                    return activated * up;
                }

                const array gate = slice(gate_up, {0, 0, 0}, {1, 1, d_ff});
                const array up = slice(gate_up, {0, 0, d_ff}, {1, 1, 2 * d_ff});
                const array activated = use_gelu ? gelu_approx(gate) : (gate * sigmoid(gate));
                return activated * up;
            }();

            const array ffn_out = astype(down_proj_fn(mid), float32);
            token_out = hidden_1 + ffn_out * residual_multiplier;
        }

        token_outputs.push_back(token_out);
    }

    if (token_outputs.size() == 1) {
        return token_outputs[0];
    }
    return concatenate(token_outputs, 1);
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
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_proj);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto& a_log_arr = astype(*static_cast<const array*>(a_log), float32);
    const auto& d_skip_arr = astype(*static_cast<const array*>(d_skip), float32);
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
        ? std::optional<array>(reshape(astype(*conv_b, float32), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(astype(*dt_b, float32), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(astype(*norm_w, float32))
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(astype(*ln2_w, float32))
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
    const auto& input_arr = *static_cast<const array*>(input);
    const auto& ln1_w = *static_cast<const array*>(ln1_weight);
    const auto& in_proj_w = *static_cast<const array*>(in_w);
    const auto& in_proj_s = *static_cast<const array*>(in_s);
    const auto& in_proj_b = *static_cast<const array*>(in_b);
    const auto& conv_w = *static_cast<const array*>(conv_weight);
    const auto* conv_b = static_cast<const array*>(conv_bias);
    const auto& a_log_arr = astype(*static_cast<const array*>(a_log), float32);
    const auto& d_skip_arr = astype(*static_cast<const array*>(d_skip), float32);
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
        ? std::optional<array>(reshape(astype(*conv_b, float32), {1, xbc_len}))
        : std::nullopt;
    const std::optional<array> dt_bias_row = (dt_b != nullptr)
        ? std::optional<array>(reshape(astype(*dt_b, float32), {1, n_heads_i}))
        : std::nullopt;
    const std::optional<array> norm_weight_arr = (norm_w != nullptr)
        ? std::optional<array>(astype(*norm_w, float32))
        : std::nullopt;
    const std::optional<array> ln2_weight_arr = (ln2_w != nullptr)
        ? std::optional<array>(astype(*ln2_w, float32))
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

// ============================================================================
// Fused Attention Block (Quantized)
// ============================================================================
// Combines: QKV projection -> reshape -> transpose -> QK norm -> RoPE ->
//           cache update -> attention -> reshape -> output projection
//
// This reduces ~15 FFI round-trips to 1.

void* mlx_lazy_fused_attention(
    const void* input,
    const void* q_w, const void* q_s, const void* q_b,
    const void* k_w, const void* k_s, const void* k_b,
    const void* v_w, const void* v_s, const void* v_b,
    const void* o_w, const void* o_s, const void* o_b,
    const void* q_norm_w,  // can be null
    const void* k_norm_w,  // can be null
    // Linear biases (optional, can be null)
    const void* q_bias,    // [n_heads * head_dim]
    const void* k_bias,    // [n_kv_heads * head_dim]
    const void* v_bias,    // [n_kv_heads * head_dim]
    const void* o_bias,    // [hidden_dim]
    // Attention sinks (optional, can be null)
    const void* attn_sinks,  // [n_heads]
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    const auto& x = *static_cast<const array*>(input);
    int group_size_int = static_cast<int>(group_size);
    int bits_int = static_cast<int>(bits);
    int batch_size = x.shape(0);
    int seq_len = x.shape(1);

    // Attention scale:
    // - Custom attention_multiplier: use directly (e.g., 0.015625)
    // - query_pre_attn_scalar: use 1/sqrt(val) for custom scaling
    // - Default: use 1/sqrt(head_dim)
    float scale_value = (attention_multiplier > 0.0f)
        ? attention_multiplier
        : (query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // QKV projections
    auto q = quantized_matmul(x,
        *static_cast<const array*>(q_w),
        *static_cast<const array*>(q_s),
        *static_cast<const array*>(q_b),
        true, group_size_int, bits_int, "affine");
    auto k = quantized_matmul(x,
        *static_cast<const array*>(k_w),
        *static_cast<const array*>(k_s),
        *static_cast<const array*>(k_b),
        true, group_size_int, bits_int, "affine");
    auto v = quantized_matmul(x,
        *static_cast<const array*>(v_w),
        *static_cast<const array*>(v_s),
        *static_cast<const array*>(v_b),
        true, group_size_int, bits_int, "affine");

    // Add linear biases if present
    if (q_bias != nullptr) {
        q = q + *static_cast<const array*>(q_bias);
    }
    if (k_bias != nullptr) {
        k = k + *static_cast<const array*>(k_bias);
    }
    if (v_bias != nullptr) {
        v = v + *static_cast<const array*>(v_bias);
    }

    // Reshape to [B, L, n_heads, head_dim]
    q = reshape(q, {batch_size, seq_len, static_cast<int>(n_heads), static_cast<int>(head_dim)});
    k = reshape(k, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});
    v = reshape(v, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});

    // QK normalization (optional)
    if (q_norm_w != nullptr) {
        q = fast::rms_norm(q, *static_cast<const array*>(q_norm_w), rms_eps);
    }
    if (k_norm_w != nullptr) {
        k = fast::rms_norm(k, *static_cast<const array*>(k_norm_w), rms_eps);
    }

    // Transpose to [B, n_heads, L, head_dim]
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    // RoPE (runtime table for multimodal prefill, otherwise standard RoPE)
    bool use_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr;
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    if (use_runtime_rope && (rope_dim <= 0 || rope_dim > static_cast<int>(head_dim) || (rope_dim % 2) != 0)) {
        use_runtime_rope = false;
    }
    if (use_runtime_rope) {
        const auto& cos_table = *static_cast<const array*>(runtime_rope_cos);
        const auto& sin_table = *static_cast<const array*>(runtime_rope_sin);
        const int seq_start = static_cast<int>(pos_offset);
        const int seq_stop = seq_start + seq_len;
        const bool valid_tables =
            cos_table.ndim() == 2 && sin_table.ndim() == 2 &&
            cos_table.shape(1) >= rope_dim && sin_table.shape(1) >= rope_dim &&
            cos_table.shape(0) >= seq_stop && sin_table.shape(0) >= seq_stop;
        if (valid_tables) {
            auto cos_rows = slice(cos_table, {seq_start, 0}, {seq_stop, rope_dim});
            auto sin_rows = slice(sin_table, {seq_start, 0}, {seq_stop, rope_dim});
            q = apply_runtime_rope_to_tensor(q, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
            k = apply_runtime_rope_to_tensor(k, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
        } else {
            use_runtime_rope = false;
        }
    }
    if (!use_runtime_rope) {
        q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
        k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    }

    // Cache update
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        void* k_cached = nullptr;
        void* v_cached = nullptr;
        mlx_cache_update_and_fetch_bfloat16(
            cache_ptr,
            layer_idx,
            &k,
            &v,
            &k_cached,
            &v_cached,
            &is_prefill
        );
        if (k_cached != nullptr && v_cached != nullptr) {
            k_for_attn = *static_cast<array*>(k_cached);
            v_for_attn = *static_cast<array*>(v_cached);
        }
    }

    // Attention (with optional sinks)
    std::optional<array> sinks_opt = std::nullopt;
    if (attn_sinks != nullptr) {
        sinks_opt = *static_cast<const array*>(attn_sinks);
    }
    auto attn_out = fast::scaled_dot_product_attention(
        q, k_for_attn, v_for_attn, scale_value, is_prefill ? "causal" : "",
        std::nullopt,  // mask_arr
        sinks_opt      // sinks
    );

    // Reshape back
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch_size, seq_len, static_cast<int>(n_heads * head_dim)});

    // Output projection
    auto out = quantized_matmul(attn_out,
        *static_cast<const array*>(o_w),
        *static_cast<const array*>(o_s),
        *static_cast<const array*>(o_b),
        true, group_size_int, bits_int, "affine");

    // Add output bias if present
    if (o_bias != nullptr) {
        out = out + *static_cast<const array*>(o_bias);
    }

    return pool_array(std::move(out));
}

// ============================================================================
// Fused Attention Block (Mixed: quantized QKV, dense output projection)
// ============================================================================

void* mlx_lazy_fused_attention_qkv_quantized_o_dense(
    const void* input,
    const void* q_w, const void* q_s, const void* q_b,
    const void* k_w, const void* k_s, const void* k_b,
    const void* v_w, const void* v_s, const void* v_b,
    const void* o_w,
    const void* q_norm_w,  // can be null
    const void* k_norm_w,  // can be null
    // Linear biases (optional, can be null)
    const void* q_bias,    // [n_heads * head_dim]
    const void* k_bias,    // [n_kv_heads * head_dim]
    const void* v_bias,    // [n_kv_heads * head_dim]
    const void* o_bias,    // [hidden_dim]
    // Attention sinks (optional, can be null)
    const void* attn_sinks,  // [n_heads]
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    size_t group_size, size_t bits,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    const auto& x = *static_cast<const array*>(input);
    int group_size_int = static_cast<int>(group_size);
    int bits_int = static_cast<int>(bits);
    int batch_size = x.shape(0);
    int seq_len = x.shape(1);

    // Attention scale:
    // - Custom attention_multiplier: use directly (e.g., 0.015625)
    // - query_pre_attn_scalar: use 1/sqrt(val) for custom scaling
    // - Default: use 1/sqrt(head_dim)
    float scale_value = (attention_multiplier > 0.0f)
        ? attention_multiplier
        : (query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // QKV projections (quantized)
    auto q = quantized_matmul(x,
        *static_cast<const array*>(q_w),
        *static_cast<const array*>(q_s),
        *static_cast<const array*>(q_b),
        true, group_size_int, bits_int, "affine");
    auto k = quantized_matmul(x,
        *static_cast<const array*>(k_w),
        *static_cast<const array*>(k_s),
        *static_cast<const array*>(k_b),
        true, group_size_int, bits_int, "affine");
    auto v = quantized_matmul(x,
        *static_cast<const array*>(v_w),
        *static_cast<const array*>(v_s),
        *static_cast<const array*>(v_b),
        true, group_size_int, bits_int, "affine");

    // Add linear biases if present
    if (q_bias != nullptr) {
        q = q + *static_cast<const array*>(q_bias);
    }
    if (k_bias != nullptr) {
        k = k + *static_cast<const array*>(k_bias);
    }
    if (v_bias != nullptr) {
        v = v + *static_cast<const array*>(v_bias);
    }

    // Reshape to [B, L, n_heads, head_dim]
    q = reshape(q, {batch_size, seq_len, static_cast<int>(n_heads), static_cast<int>(head_dim)});
    k = reshape(k, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});
    v = reshape(v, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});

    // QK normalization (optional)
    if (q_norm_w != nullptr) {
        q = fast::rms_norm(q, *static_cast<const array*>(q_norm_w), rms_eps);
    }
    if (k_norm_w != nullptr) {
        k = fast::rms_norm(k, *static_cast<const array*>(k_norm_w), rms_eps);
    }

    // Transpose to [B, n_heads, L, head_dim]
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    // RoPE (runtime table for multimodal prefill, otherwise standard RoPE)
    bool use_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr;
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    if (use_runtime_rope && (rope_dim <= 0 || rope_dim > static_cast<int>(head_dim) || (rope_dim % 2) != 0)) {
        use_runtime_rope = false;
    }
    if (use_runtime_rope) {
        const auto& cos_table = *static_cast<const array*>(runtime_rope_cos);
        const auto& sin_table = *static_cast<const array*>(runtime_rope_sin);
        const int seq_start = static_cast<int>(pos_offset);
        const int seq_stop = seq_start + seq_len;
        const bool valid_tables =
            cos_table.ndim() == 2 && sin_table.ndim() == 2 &&
            cos_table.shape(1) >= rope_dim && sin_table.shape(1) >= rope_dim &&
            cos_table.shape(0) >= seq_stop && sin_table.shape(0) >= seq_stop;
        if (valid_tables) {
            auto cos_rows = slice(cos_table, {seq_start, 0}, {seq_stop, rope_dim});
            auto sin_rows = slice(sin_table, {seq_start, 0}, {seq_stop, rope_dim});
            q = apply_runtime_rope_to_tensor(q, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
            k = apply_runtime_rope_to_tensor(k, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
        } else {
            use_runtime_rope = false;
        }
    }
    if (!use_runtime_rope) {
        q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
        k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    }

    // Cache update
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        void* k_cached = nullptr;
        void* v_cached = nullptr;
        mlx_cache_update_and_fetch_bfloat16(
            cache_ptr,
            layer_idx,
            &k,
            &v,
            &k_cached,
            &v_cached,
            &is_prefill
        );
        if (k_cached != nullptr && v_cached != nullptr) {
            k_for_attn = *static_cast<array*>(k_cached);
            v_for_attn = *static_cast<array*>(v_cached);
        }
    }

    // Attention (with optional sinks)
    std::optional<array> sinks_opt = std::nullopt;
    if (attn_sinks != nullptr) {
        sinks_opt = *static_cast<const array*>(attn_sinks);
    }
    auto attn_out = fast::scaled_dot_product_attention(
        q, k_for_attn, v_for_attn, scale_value, is_prefill ? "causal" : "",
        std::nullopt,  // mask_arr
        sinks_opt      // sinks
    );

    // Reshape back
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch_size, seq_len, static_cast<int>(n_heads * head_dim)});

    // Dense output projection (o_w stored as [out, in])
    auto o_wt = transpose(*static_cast<const array*>(o_w), {1, 0});
    auto out = matmul(attn_out, o_wt);

    // Add output bias if present
    if (o_bias != nullptr) {
        out = out + *static_cast<const array*>(o_bias);
    }

    return pool_array(std::move(out));
}

// ============================================================================
// Fused FFN Block (Quantized)
// ============================================================================
// Combines: gate_proj -> SiLU -> multiply(up_proj) -> down_proj

void* mlx_lazy_fused_ffn(
    const void* input,
    const void* gate_w, const void* gate_s, const void* gate_b,
    const void* up_w, const void* up_s, const void* up_b,
    const void* down_w, const void* down_s, const void* down_b,
    size_t group_size, size_t bits,
    bool use_gelu  // true for GELU activation, false for SwiGLU
) {
    const auto& x = *static_cast<const array*>(input);
    int gs = static_cast<int>(group_size);
    int b = static_cast<int>(bits);

    auto gate = quantized_matmul(x,
        *static_cast<const array*>(gate_w),
        *static_cast<const array*>(gate_s),
        *static_cast<const array*>(gate_b),
        true, gs, b, "affine");

    auto up = quantized_matmul(x,
        *static_cast<const array*>(up_w),
        *static_cast<const array*>(up_s),
        *static_cast<const array*>(up_b),
        true, gs, b, "affine");

    // Activation: GELU or SwiGLU based on model config
    auto mid = [&]() -> array {
        if (use_gelu) {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_over_pi = 0.7978845608f;
            auto x3 = gate * gate * gate;
            auto inner = sqrt_2_over_pi * (gate + 0.044715f * x3);
            return 0.5f * gate * (1.0f + tanh(inner)) * up;
        } else {
            // SwiGLU: silu(gate) * up
            return (gate * sigmoid(gate)) * up;
        }
    }();

    auto out = quantized_matmul(mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        *static_cast<const array*>(down_b),
        true, gs, b, "affine");

    return pool_array(std::move(out));
}

// ============================================================================
// Fused Attention Block (BFloat16 - non-quantized)
// ============================================================================

void* mlx_lazy_fused_attention_bf16(
    const void* input,
    const void* q_w, const void* k_w, const void* v_w, const void* o_w,
    const void* q_norm_w,
    const void* k_norm_w,
    // Linear biases (optional, can be null)
    const void* q_bias,
    const void* k_bias,
    const void* v_bias,
    const void* o_bias,
    // Attention sinks (optional, can be null)
    const void* attn_sinks,
    void* cache_ptr, size_t layer_idx,
    size_t n_heads, size_t n_kv_heads, size_t head_dim,
    size_t pos_offset, float rope_theta,
    const void* runtime_rope_cos, const void* runtime_rope_sin, size_t runtime_rope_dim,
    float rms_eps,
    float query_pre_attn_scalar,  // 0 for default (head_dim), >0 for custom scaling
    float attention_multiplier    // 0 for default, >0 uses this directly as scale
) {
    const auto& x = *static_cast<const array*>(input);
    int batch_size = x.shape(0);
    int seq_len = x.shape(1);

    // Attention scale:
    // - Custom attention_multiplier: use directly
    // - query_pre_attn_scalar: use 1/sqrt(val) for custom scaling
    // - Default: use 1/sqrt(head_dim)
    float scale_value = (attention_multiplier > 0.0f)
        ? attention_multiplier
        : (query_pre_attn_scalar > 0.0f)
            ? (1.0f / std::sqrt(query_pre_attn_scalar))
            : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    const int hidden_dim = x.shape(2);
    const int q_out = static_cast<int>(n_heads * head_dim);
    const int kv_out = static_cast<int>(n_kv_heads * head_dim);

    auto q_wt = orient_matmul_rhs(*static_cast<const array*>(q_w), hidden_dim, q_out, true);
    auto k_wt = orient_matmul_rhs(*static_cast<const array*>(k_w), hidden_dim, kv_out, true);
    auto v_wt = orient_matmul_rhs(*static_cast<const array*>(v_w), hidden_dim, kv_out, true);
    auto o_wt = orient_matmul_rhs(*static_cast<const array*>(o_w), q_out, hidden_dim, true);

    // QKV projections
    auto q = matmul(x, q_wt);
    auto k = matmul(x, k_wt);
    auto v = matmul(x, v_wt);

    // Add linear biases if present
    if (q_bias != nullptr) {
        q = q + *static_cast<const array*>(q_bias);
    }
    if (k_bias != nullptr) {
        k = k + *static_cast<const array*>(k_bias);
    }
    if (v_bias != nullptr) {
        v = v + *static_cast<const array*>(v_bias);
    }

    // Reshape
    q = reshape(q, {batch_size, seq_len, static_cast<int>(n_heads), static_cast<int>(head_dim)});
    k = reshape(k, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});
    v = reshape(v, {batch_size, seq_len, static_cast<int>(n_kv_heads), static_cast<int>(head_dim)});

    // QK normalization
    if (q_norm_w != nullptr) {
        q = fast::rms_norm(q, *static_cast<const array*>(q_norm_w), rms_eps);
    }
    if (k_norm_w != nullptr) {
        k = fast::rms_norm(k, *static_cast<const array*>(k_norm_w), rms_eps);
    }

    // Transpose to [B, n_heads, L, head_dim]
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});

    // RoPE (runtime table for multimodal prefill, otherwise standard RoPE)
    bool use_runtime_rope = runtime_rope_cos != nullptr && runtime_rope_sin != nullptr;
    const int rope_dim = static_cast<int>(runtime_rope_dim);
    if (use_runtime_rope && (rope_dim <= 0 || rope_dim > static_cast<int>(head_dim) || (rope_dim % 2) != 0)) {
        use_runtime_rope = false;
    }
    if (use_runtime_rope) {
        const auto& cos_table = *static_cast<const array*>(runtime_rope_cos);
        const auto& sin_table = *static_cast<const array*>(runtime_rope_sin);
        const int seq_start = static_cast<int>(pos_offset);
        const int seq_stop = seq_start + seq_len;
        const bool valid_tables =
            cos_table.ndim() == 2 && sin_table.ndim() == 2 &&
            cos_table.shape(1) >= rope_dim && sin_table.shape(1) >= rope_dim &&
            cos_table.shape(0) >= seq_stop && sin_table.shape(0) >= seq_stop;
        if (valid_tables) {
            auto cos_rows = slice(cos_table, {seq_start, 0}, {seq_stop, rope_dim});
            auto sin_rows = slice(sin_table, {seq_start, 0}, {seq_stop, rope_dim});
            q = apply_runtime_rope_to_tensor(q, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
            k = apply_runtime_rope_to_tensor(k, cos_rows, sin_rows, seq_len, static_cast<int>(head_dim), rope_dim);
        } else {
            use_runtime_rope = false;
        }
    }
    if (!use_runtime_rope) {
        q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
        k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    }

    // Cache update (same as quantized version)
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        void* k_cached = nullptr;
        void* v_cached = nullptr;
        mlx_cache_update_and_fetch_bfloat16(
            cache_ptr,
            layer_idx,
            &k,
            &v,
            &k_cached,
            &v_cached,
            &is_prefill
        );
        if (k_cached != nullptr && v_cached != nullptr) {
            k_for_attn = *static_cast<array*>(k_cached);
            v_for_attn = *static_cast<array*>(v_cached);
        }
    }

    // Attention (with optional sinks)
    std::optional<array> sinks_opt = std::nullopt;
    if (attn_sinks != nullptr) {
        sinks_opt = *static_cast<const array*>(attn_sinks);
    }
    auto attn_out = fast::scaled_dot_product_attention(
        q, k_for_attn, v_for_attn, scale_value, is_prefill ? "causal" : "",
        std::nullopt,  // mask_arr
        sinks_opt      // sinks
    );

    // Reshape back
    attn_out = transpose(attn_out, {0, 2, 1, 3});
    attn_out = reshape(attn_out, {batch_size, seq_len, static_cast<int>(n_heads * head_dim)});

    // Output projection
    auto out = matmul(attn_out, o_wt);

    // Add output bias if present
    if (o_bias != nullptr) {
        out = out + *static_cast<const array*>(o_bias);
    }

    return pool_array(std::move(out));
}

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

    array q_comp = astype(q_a_proj_fn(input), float32);
    if (q_comp.shape(2) != q_lora_rank) {
        throw std::invalid_argument("[mla] q_a_proj output does not match q_lora_rank");
    }
    q_comp = fast::rms_norm(q_comp, q_a_norm_w, rms_eps);
    array q_proj = astype(q_b_proj_fn(q_comp), float32);
    if (q_proj.shape(2) != n_heads * qk_head_dim) {
        throw std::invalid_argument("[mla] q_b_proj output shape mismatch");
    }

    array kv_comp = astype(kv_a_proj_fn(input), float32);
    if (kv_comp.shape(2) != kv_lora_rank + qk_rope_head_dim) {
        throw std::invalid_argument("[mla] kv_a_proj output shape mismatch");
    }
    array kv_nope = slice(kv_comp, {0, 0, 0}, {batch_size, seq_len, kv_lora_rank});
    array k_rope_shared = slice(kv_comp, {0, 0, kv_lora_rank}, {batch_size, seq_len, kv_lora_rank + qk_rope_head_dim});
    kv_nope = fast::rms_norm(kv_nope, kv_a_norm_w, rms_eps);
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

// ============================================================================
// Fused FFN Block (BFloat16 - non-quantized)
// ============================================================================

void* mlx_lazy_fused_ffn_bf16(
    const void* input,
    const void* gate_w, const void* up_w, const void* down_w
) {
    const auto& x = *static_cast<const array*>(input);
    const int in_features = x.shape(2);

    auto gate_wt = orient_matmul_rhs(*static_cast<const array*>(gate_w), in_features, std::nullopt, true);
    auto up_wt = orient_matmul_rhs(*static_cast<const array*>(up_w), in_features, gate_wt.shape(1), true);
    auto down_wt = orient_matmul_rhs(*static_cast<const array*>(down_w), gate_wt.shape(1), in_features, true);

    auto gate = matmul(x, gate_wt);
    auto up = matmul(x, up_wt);
    auto mid = (gate * sigmoid(gate)) * up;
    auto out = matmul(mid, down_wt);

    return pool_array(std::move(out));
}

// ============================================================================
// Fused MoE FFN Block (MXFP4 quantized experts)
// ============================================================================
// Implements: router -> topk -> gather_qmm (gate/up/down) -> weighted sum
//
// This matches mlx_lm's SwitchGLU layer:
// - Router projects to expert logits
// - TopK selects active experts per token
// - gather_qmm computes expert FFN outputs (only for selected experts)
// - Results are weighted by softmax and summed

// SwiGLU variant with alpha=1.702, clipping, and (up+1) formulation:
// swiglu(x_linear, x_glu) = (glu_scaled * sigmoid(glu_scaled)) * (x_linear + 1)
//   where glu_scaled = 1.702 * clip(x_glu, max=7)
static array swiglu_variant(const array& x_linear, const array& x_glu) {
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    // Clamp values
    auto x_glu_clipped = clip(x_glu, std::nullopt, array(limit));
    auto x_linear_clipped = clip(x_linear, array(-limit), array(limit));

    // Compute activation
    auto glu_scaled = alpha * x_glu_clipped;
    auto sig = sigmoid(glu_scaled);
    auto out_glu = x_glu_clipped * sig;

    // Note: x_linear + 1 bias
    return out_glu * (x_linear_clipped + 1.0f);
}

void* mlx_lazy_fused_moe_ffn_mxfp4(
    const void* input,
    // Router weights (8-bit affine quantized)
    const void* router_w, const void* router_s, const void* router_b,
    const void* router_bias,  // can be null
    // Expert weights [num_experts, d_ff, packed_dim] - MXFP4 quantized
    // Separate gate/up/down projections (not fused)
    const void* gate_w, const void* gate_s,
    const void* up_w, const void* up_s,
    const void* down_w, const void* down_s,
    // Expert biases (optional) - [num_experts, d_ff] or [num_experts, d_model]
    const void* gate_bias,  // can be null
    const void* up_bias,    // can be null
    const void* down_bias,  // can be null
    // Config
    size_t num_experts,
    size_t experts_per_token,
    size_t router_group_size,   // 64 for router (8-bit)
    size_t expert_group_size    // 32 for MXFP4
) {
    const auto& x = *static_cast<const array*>(input);

    // Router: compute expert logits [B, L, num_experts]
    // Supports two formats:
    // 1. MLX community: 8-bit affine quantized router (router_s/router_b present)
    // 2. Hub/OpenAI: BF16 unquantized router (router_s/router_b null)
    array router_logits = (router_s != nullptr && router_b != nullptr)
        ? quantized_matmul(x,
            *static_cast<const array*>(router_w),
            *static_cast<const array*>(router_s),
            *static_cast<const array*>(router_b),
            true, static_cast<int>(router_group_size), 8, "affine")
        : matmul(x, transpose(*static_cast<const array*>(router_w)));

    // Add router bias if present
    if (router_bias != nullptr) {
        router_logits = router_logits + *static_cast<const array*>(router_bias);
    }

    // TopK: select top experts
    // argpartition returns indices of top K elements (unsorted)
    int k = static_cast<int>(experts_per_token);
    auto partitioned_indices = argpartition(router_logits, -k, -1);

    // Extract top-k indices (last k elements along axis -1)
    int last_dim = router_logits.ndim() - 1;
    int total_experts = static_cast<int>(num_experts);

    // Slice to get top-k indices: [..., -k:]
    Shape start(router_logits.ndim(), 0);
    Shape stop = router_logits.shape();
    start[last_dim] = total_experts - k;
    auto top_k_indices = slice(partitioned_indices, start, stop);

    // Get corresponding logits for softmax weighting
    auto top_k_logits = take_along_axis(router_logits, top_k_indices, last_dim);

    // Softmax to get expert weights [B, L, K]
    auto expert_weights = softmax(top_k_logits, -1, true);  // precise=true

    // Expand input for gather_qmm: [B, L, 1, 1, hidden_dim]
    auto x_expanded = expand_dims(x, {-2, -3});

    // Gather-QMM for gate projection
    // gather_qmm with rhs_indices selects which experts to use per token
    auto gate_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(gate_w),
        *static_cast<const array*>(gate_s),
        std::nullopt,  // MXFP4 has no biases in quantization
        std::nullopt,  // lhs_indices
        top_k_indices, // rhs_indices - selects which experts
        true,          // transpose
        static_cast<int>(expert_group_size),
        4,             // bits for MXFP4
        "mxfp4",       // mode
        false          // sorted_indices
    );

    // Up projection
    auto up_out = gather_qmm(
        x_expanded,
        *static_cast<const array*>(up_w),
        *static_cast<const array*>(up_s),
        std::nullopt,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        4,
        "mxfp4",
        false
    );

    // Add expert biases if present
    if (gate_bias != nullptr) {
        // Gather biases for selected experts: [num_experts, d_ff] -> [B*L*K, d_ff]
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto gate_b = take(*static_cast<const array*>(gate_bias), indices_flat, 0);
        // Reshape to match gate_out: [B, L, K, 1, d_ff]
        auto shape = gate_out.shape();
        gate_b = reshape(gate_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        gate_out = gate_out + gate_b;
    }
    if (up_bias != nullptr) {
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto up_b = take(*static_cast<const array*>(up_bias), indices_flat, 0);
        auto shape = up_out.shape();
        up_b = reshape(up_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        up_out = up_out + up_b;
    }

    // SwiGLU variant activation
    auto mid = swiglu_variant(up_out, gate_out);

    // Down projection
    auto down_out = gather_qmm(
        mid,
        *static_cast<const array*>(down_w),
        *static_cast<const array*>(down_s),
        std::nullopt,
        std::nullopt,
        top_k_indices,
        true,
        static_cast<int>(expert_group_size),
        4,
        "mxfp4",
        false
    );

    if (down_bias != nullptr) {
        auto indices_flat = flatten(top_k_indices, 0, -1);
        auto down_b = take(*static_cast<const array*>(down_bias), indices_flat, 0);
        auto shape = down_out.shape();
        down_b = reshape(down_b, {shape[0], shape[1], shape[2], 1, shape[4]});
        down_out = down_out + down_b;
    }

    // Squeeze out singleton dimensions: [B, L, K, 1, hidden_dim] -> [B, L, K, hidden_dim]
    down_out = squeeze(down_out, -2);

    // Weight by expert weights and sum
    // expert_weights: [B, L, K] -> [B, L, K, 1]
    auto weights_expanded = expand_dims(expert_weights, -1);
    auto weighted = down_out * weights_expanded;

    // Sum over experts: [B, L, K, hidden_dim] -> [B, L, hidden_dim]
    auto out = sum(weighted, -2);

    return pool_array(std::move(out));
}

} // extern "C"
