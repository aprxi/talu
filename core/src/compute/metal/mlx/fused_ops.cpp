// MLX Bridge - Fused Neural Network Operations
//
// High-level operations that combine multiple MLX calls for efficiency.
// Uses MLX fast:: kernels which are highly optimized Metal implementations.

#include "common.h"

extern "C" {

// ============================================================================
// MLX Fast Kernels
// ============================================================================
// These wrap MLX's optimized Metal implementations.

void* mlx_lazy_rms_norm(const void* input, const void* weight, float eps) {
    return pool_array(fast::rms_norm(
        *static_cast<const array*>(input),
        *static_cast<const array*>(weight),
        eps
    ));
}

// Add 1 to weight for (1 + weight) style RMSNorm: (1 + weight) * normalized
void* mlx_add_one(const void* arr) {
    return pool_array(1.0f + *static_cast<const array*>(arr));
}

// Scale array by sqrt(d_model) for embedding scaling
void* mlx_scale_by_sqrt(const void* arr, size_t d_model) {
    float scale = std::sqrt(static_cast<float>(d_model));
    return pool_array(*static_cast<const array*>(arr) * scale);
}

void* mlx_lazy_rope(const void* input, size_t head_dim, size_t offset, float rope_base) {
    return pool_array(fast::rope(
        *static_cast<const array*>(input),
        static_cast<int>(head_dim),
        false,  // traditional = false
        rope_base,
        1.0f,   // scale = 1.0
        static_cast<int>(offset)
    ));
}

void* mlx_lazy_attention(const void* q, const void* k, const void* v, float scale, bool causal) {
    return pool_array(fast::scaled_dot_product_attention(
        *static_cast<const array*>(q),
        *static_cast<const array*>(k),
        *static_cast<const array*>(v),
        scale,
        causal ? "causal" : ""
    ));
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
    size_t pos_offset, float rope_theta, float rms_eps,
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

    // RoPE
    q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));

    // Cache update
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        auto cache = static_cast<MLXCache*>(cache_ptr);
        auto& layer = cache->layers[layer_idx];

        int num_steps = k.shape(2);
        int k_head_dim_i = k.shape(3);
        int v_head_dim_i = v.shape(3);
        size_t prev = layer.offset;
        is_prefill = (prev == 0);

        // Pre-allocate or expand buffer
        if (layer.k_bfloat16 == nullptr ||
            (prev + num_steps) > static_cast<size_t>(layer.k_bfloat16->shape(2))) {
            int n_steps_alloc = (layer.step + num_steps - 1) / layer.step;
            Shape k_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, k_head_dim_i};
            Shape v_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, v_head_dim_i};
            auto new_k = zeros(k_shape, k.dtype());
            auto new_v = zeros(v_shape, v.dtype());

            if (layer.k_bfloat16 != nullptr) {
                if (prev % layer.step != 0) {
                    Shape start = {0, 0, 0, 0};
                    Shape stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), k_head_dim_i};
                    Shape stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), v_head_dim_i};
                    *layer.k_bfloat16 = slice(*layer.k_bfloat16, start, stop_k);
                    *layer.v_bfloat16 = slice(*layer.v_bfloat16, start, stop_v);
                }
                *layer.k_bfloat16 = concatenate({*layer.k_bfloat16, new_k}, 2);
                *layer.v_bfloat16 = concatenate({*layer.v_bfloat16, new_v}, 2);
            } else {
                layer.k_bfloat16 = new array(new_k);
                layer.v_bfloat16 = new array(new_v);
            }
        }

        // Update cache with slice_update (matches Python's indexed assignment)
        size_t offset = prev + num_steps;
        Shape update_start = {0, 0, static_cast<int>(prev), 0};
        Shape update_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape update_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};

        *layer.k_bfloat16 = slice_update(*layer.k_bfloat16, k, update_start, update_stop_k);
        *layer.v_bfloat16 = slice_update(*layer.v_bfloat16, v, update_start, update_stop_v);
        layer.offset = offset;

        // Get slice for attention
        Shape slice_start = {0, 0, 0, 0};
        Shape slice_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape slice_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};
        k_for_attn = slice(*layer.k_bfloat16, slice_start, slice_stop_k);
        v_for_attn = slice(*layer.v_bfloat16, slice_start, slice_stop_v);
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
    size_t pos_offset, float rope_theta, float rms_eps,
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

    // RoPE
    q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));

    // Cache update
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        auto cache = static_cast<MLXCache*>(cache_ptr);
        auto& layer = cache->layers[layer_idx];

        int num_steps = k.shape(2);
        int k_head_dim_i = k.shape(3);
        int v_head_dim_i = v.shape(3);
        size_t prev = layer.offset;
        is_prefill = (prev == 0);

        if (layer.k_bfloat16 == nullptr ||
            (prev + num_steps) > static_cast<size_t>(layer.k_bfloat16->shape(2))) {
            int n_steps_alloc = (layer.step + num_steps - 1) / layer.step;
            Shape k_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, k_head_dim_i};
            Shape v_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, v_head_dim_i};
            auto new_k = zeros(k_shape, k.dtype());
            auto new_v = zeros(v_shape, v.dtype());

            if (layer.k_bfloat16 != nullptr) {
                if (prev % layer.step != 0) {
                    Shape start = {0, 0, 0, 0};
                    Shape stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), k_head_dim_i};
                    Shape stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), v_head_dim_i};
                    *layer.k_bfloat16 = slice(*layer.k_bfloat16, start, stop_k);
                    *layer.v_bfloat16 = slice(*layer.v_bfloat16, start, stop_v);
                }
                *layer.k_bfloat16 = concatenate({*layer.k_bfloat16, new_k}, 2);
                *layer.v_bfloat16 = concatenate({*layer.v_bfloat16, new_v}, 2);
            } else {
                layer.k_bfloat16 = new array(new_k);
                layer.v_bfloat16 = new array(new_v);
            }
        }

        size_t offset = prev + num_steps;
        Shape update_start = {0, 0, static_cast<int>(prev), 0};
        Shape update_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape update_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};

        *layer.k_bfloat16 = slice_update(*layer.k_bfloat16, k, update_start, update_stop_k);
        *layer.v_bfloat16 = slice_update(*layer.v_bfloat16, v, update_start, update_stop_v);
        layer.offset = offset;

        Shape slice_start = {0, 0, 0, 0};
        Shape slice_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape slice_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};
        k_for_attn = slice(*layer.k_bfloat16, slice_start, slice_stop_k);
        v_for_attn = slice(*layer.v_bfloat16, slice_start, slice_stop_v);
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
    size_t pos_offset, float rope_theta, float rms_eps,
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

    // Transpose weights for matmul: [out, in] -> [in, out]
    auto q_wt = transpose(*static_cast<const array*>(q_w), {1, 0});
    auto k_wt = transpose(*static_cast<const array*>(k_w), {1, 0});
    auto v_wt = transpose(*static_cast<const array*>(v_w), {1, 0});
    auto o_wt = transpose(*static_cast<const array*>(o_w), {1, 0});

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

    // RoPE
    q = fast::rope(q, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));
    k = fast::rope(k, static_cast<int>(head_dim), false, rope_theta, 1.0f, static_cast<int>(pos_offset));

    // Cache update (same as quantized version)
    array k_for_attn = k;
    array v_for_attn = v;
    bool is_prefill = true;

    if (cache_ptr != nullptr) {
        auto cache = static_cast<MLXCache*>(cache_ptr);
        auto& layer = cache->layers[layer_idx];

        int num_steps = k.shape(2);
        int k_head_dim_i = k.shape(3);
        int v_head_dim_i = v.shape(3);
        size_t prev = layer.offset;
        is_prefill = (prev == 0);

        if (layer.k_bfloat16 == nullptr ||
            (prev + num_steps) > static_cast<size_t>(layer.k_bfloat16->shape(2))) {
            int n_steps_alloc = (layer.step + num_steps - 1) / layer.step;
            Shape k_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, k_head_dim_i};
            Shape v_shape = {batch_size, static_cast<int>(n_kv_heads), n_steps_alloc * layer.step, v_head_dim_i};
            auto new_k = zeros(k_shape, k.dtype());
            auto new_v = zeros(v_shape, v.dtype());

            if (layer.k_bfloat16 != nullptr) {
                if (prev % layer.step != 0) {
                    Shape start = {0, 0, 0, 0};
                    Shape stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), k_head_dim_i};
                    Shape stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(prev), v_head_dim_i};
                    *layer.k_bfloat16 = slice(*layer.k_bfloat16, start, stop_k);
                    *layer.v_bfloat16 = slice(*layer.v_bfloat16, start, stop_v);
                }
                *layer.k_bfloat16 = concatenate({*layer.k_bfloat16, new_k}, 2);
                *layer.v_bfloat16 = concatenate({*layer.v_bfloat16, new_v}, 2);
            } else {
                layer.k_bfloat16 = new array(new_k);
                layer.v_bfloat16 = new array(new_v);
            }
        }

        size_t offset = prev + num_steps;
        Shape update_start = {0, 0, static_cast<int>(prev), 0};
        Shape update_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape update_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};

        *layer.k_bfloat16 = slice_update(*layer.k_bfloat16, k, update_start, update_stop_k);
        *layer.v_bfloat16 = slice_update(*layer.v_bfloat16, v, update_start, update_stop_v);
        layer.offset = offset;

        Shape slice_start = {0, 0, 0, 0};
        Shape slice_stop_k = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), k_head_dim_i};
        Shape slice_stop_v = {batch_size, static_cast<int>(n_kv_heads), static_cast<int>(offset), v_head_dim_i};
        k_for_attn = slice(*layer.k_bfloat16, slice_start, slice_stop_k);
        v_for_attn = slice(*layer.v_bfloat16, slice_start, slice_stop_v);
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

// ============================================================================
// Fused FFN Block (BFloat16 - non-quantized)
// ============================================================================

void* mlx_lazy_fused_ffn_bf16(
    const void* input,
    const void* gate_w, const void* up_w, const void* down_w
) {
    const auto& x = *static_cast<const array*>(input);

    auto gate_wt = transpose(*static_cast<const array*>(gate_w), {1, 0});
    auto up_wt = transpose(*static_cast<const array*>(up_w), {1, 0});
    auto down_wt = transpose(*static_cast<const array*>(down_w), {1, 0});

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
