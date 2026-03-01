// MLX Bridge - KV Cache Management
//
// Implements KV cache for transformer inference, matching Python mlx-lm behavior.
// Uses pre-allocated buffers with slice_update for efficiency.

#include "model_state.h"
#include "cache_utils.h"

extern "C" {

// ============================================================================
// Cache Lifecycle
// ============================================================================

void* mlx_cache_create(size_t n_layers, size_t max_seq_len) {
    auto cache_state = new MLXCache();
    cache_state->layers.resize(n_layers);
    for (auto& layer : cache_state->layers) {
        layer.max_seq_len = max_seq_len;
    }
    return cache_state;
}

void* mlx_cache_create_bfloat16(size_t n_layers, size_t max_seq_len) {
    // Same as mlx_cache_create - cache type determined by usage
    return mlx_cache_create(n_layers, max_seq_len);
}

void mlx_cache_free(void* cache_ptr) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    for (auto& layer : cache_state->layers) {
        delete layer.k_bfloat16;
        delete layer.v_bfloat16;
        delete layer.k_view;
        delete layer.v_view;
    }
    delete cache_state;
}

// ============================================================================
// BFloat16 Cache Operations
// ============================================================================
// Cache policy:
//   1. Allocate backing buffers once (prefer max_seq_len when provided).
//   2. Use slice_update for in-place writes.
//   3. Never grow buffers after allocation (fail fast on capacity overflow).
//   4. Return slice [0:offset].

void mlx_cache_update_and_fetch_bfloat16(
    void* cache_ptr, size_t layer_idx,
    const void* k_new, const void* v_new,
    void** k_out, void** v_out, bool* is_prefill_out
) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    auto& layer = cache_state->layers[layer_idx];

    const auto& k_new_arr = *static_cast<const array*>(k_new);
    const auto& v_new_arr = *static_cast<const array*>(v_new);

    int batch = k_new_arr.shape(0);
    int n_kv_heads = k_new_arr.shape(1);
    int num_steps = k_new_arr.shape(2);
    int k_head_dim = k_new_arr.shape(3);
    int v_head_dim = v_new_arr.shape(3);

    size_t prev = layer.offset;
    *is_prefill_out = (prev == 0);

    const size_t required = prev + static_cast<size_t>(num_steps);

    // Initialize backing storage or grow geometrically as needed.
    if (layer.k_bfloat16 == nullptr) {
        const int capacity = mlx_next_cache_capacity(layer, required, 0, "[cache]");
        Shape k_shape = {batch, n_kv_heads, capacity, k_head_dim};
        Shape v_shape = {batch, n_kv_heads, capacity, v_head_dim};
        layer.k_bfloat16 = new array(zeros(k_shape, k_new_arr.dtype()));
        layer.v_bfloat16 = new array(zeros(v_shape, v_new_arr.dtype()));
    } else {
        const int current_capacity = layer.k_bfloat16->shape(2);
        if (required > static_cast<size_t>(current_capacity)) {
            const int new_capacity = mlx_next_cache_capacity(layer, required, current_capacity, "[cache]");
            Shape new_k_shape = {batch, n_kv_heads, new_capacity, k_head_dim};
            Shape new_v_shape = {batch, n_kv_heads, new_capacity, v_head_dim};
            array new_k = zeros(new_k_shape, layer.k_bfloat16->dtype());
            array new_v = zeros(new_v_shape, layer.v_bfloat16->dtype());
            if (prev > 0) {
                Shape copy_start = {0, 0, 0, 0};
                Shape copy_stop_k = {batch, n_kv_heads, static_cast<int>(prev), k_head_dim};
                Shape copy_stop_v = {batch, n_kv_heads, static_cast<int>(prev), v_head_dim};
                array k_existing = slice(*layer.k_bfloat16, copy_start, copy_stop_k);
                array v_existing = slice(*layer.v_bfloat16, copy_start, copy_stop_v);
                new_k = slice_update(new_k, k_existing, copy_start, copy_stop_k);
                new_v = slice_update(new_v, v_existing, copy_start, copy_stop_v);
            }
            *layer.k_bfloat16 = new_k;
            *layer.v_bfloat16 = new_v;
        }
    }

    // Slice update: cache[..., prev:offset, :] = new_data
    size_t offset = prev + num_steps;
    Shape start = {0, 0, static_cast<int>(prev), 0};
    Shape stop_k = {batch, n_kv_heads, static_cast<int>(offset), k_head_dim};
    Shape stop_v = {batch, n_kv_heads, static_cast<int>(offset), v_head_dim};
    *layer.k_bfloat16 = slice_update(*layer.k_bfloat16, k_new_arr, start, stop_k);
    *layer.v_bfloat16 = slice_update(*layer.v_bfloat16, v_new_arr, start, stop_v);

    layer.offset = offset;

    // Return slice [0:offset]
    Shape view_start = {0, 0, 0, 0};
    Shape view_stop_k = {batch, n_kv_heads, static_cast<int>(offset), k_head_dim};
    Shape view_stop_v = {batch, n_kv_heads, static_cast<int>(offset), v_head_dim};

    // Reuse view arrays to avoid allocation
    if (layer.k_view == nullptr) {
        layer.k_view = new array(slice(*layer.k_bfloat16, view_start, view_stop_k));
        layer.v_view = new array(slice(*layer.v_bfloat16, view_start, view_stop_v));
    } else {
        *layer.k_view = slice(*layer.k_bfloat16, view_start, view_stop_k);
        *layer.v_view = slice(*layer.v_bfloat16, view_start, view_stop_v);
    }

    *k_out = layer.k_view;
    *v_out = layer.v_view;
}

void mlx_cache_get_bfloat16(
    void* cache_ptr, size_t layer_idx,
    void** k_out, void** v_out
) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    auto& layer = cache_state->layers[layer_idx];
    *k_out = layer.k_bfloat16;
    *v_out = layer.v_bfloat16;
}

void mlx_cache_set_full_bfloat16(
    void* cache_ptr, size_t layer_idx,
    const void* k_full, const void* v_full
) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    auto& layer = cache_state->layers[layer_idx];

    const auto& k_arr = *static_cast<const array*>(k_full);
    const auto& v_arr = *static_cast<const array*>(v_full);

    if (layer.k_bfloat16 == nullptr) {
        layer.k_bfloat16 = new array(k_arr);
        layer.v_bfloat16 = new array(v_arr);
    } else {
        *layer.k_bfloat16 = k_arr;
        *layer.v_bfloat16 = v_arr;
    }

    layer.offset = k_arr.shape(2);
}

void mlx_cache_eval_all(void* cache_ptr, size_t n_layers) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    std::vector<array> to_eval;

    for (size_t i = 0; i < n_layers; i++) {
        if (cache_state->layers[i].k_bfloat16) {
            to_eval.push_back(*cache_state->layers[i].k_bfloat16);
            to_eval.push_back(*cache_state->layers[i].v_bfloat16);
        }
    }

    if (!to_eval.empty()) {
        eval(to_eval);
    }
}

// ============================================================================
// Non-BFloat16 Cache Operations (for quantized path compatibility)
// ============================================================================
// These use the same storage as bfloat16 but are provided for API completeness.

void mlx_cache_update_and_fetch(
    void* cache_ptr, size_t layer_idx,
    const void* k_new, const void* v_new,
    void** k_out, void** v_out, bool* is_prefill_out
) {
    // Delegate to bfloat16 implementation - storage is the same
    mlx_cache_update_and_fetch_bfloat16(
        cache_ptr, layer_idx, k_new, v_new, k_out, v_out, is_prefill_out
    );
}

void mlx_cache_get_quantized(
    void* cache_ptr, size_t layer_idx,
    void** k_weights_out, void** k_scales_out, void** k_biases_out,
    void** v_weights_out, void** v_scales_out, void** v_biases_out
) {
    // Quantized cache not implemented - return nulls
    // Real quantization would require storing scales/biases separately
    *k_weights_out = nullptr;
    *k_scales_out = nullptr;
    *k_biases_out = nullptr;
    *v_weights_out = nullptr;
    *v_scales_out = nullptr;
    *v_biases_out = nullptr;
}

} // extern "C"

extern "C" {

// ============================================================================
// ShortConv Cache Lifecycle
// ============================================================================

void* mlx_shortconv_cache_create(size_t n_layers) {
    auto* cache_state = new MLXShortConvCache();
    cache_state->layers.resize(n_layers);
    return cache_state;
}

void mlx_shortconv_cache_reset(void* cache_ptr) {
    auto* cache_state = static_cast<MLXShortConvCache*>(cache_ptr);
    if (cache_state == nullptr) return;
    for (auto& layer : cache_state->layers) {
        if (layer.conv_state != nullptr) {
            const auto shape = layer.conv_state->shape();
            *layer.conv_state = zeros(shape, float32);
        }
    }
}

void mlx_shortconv_cache_free(void* cache_ptr) {
    auto* cache_state = static_cast<MLXShortConvCache*>(cache_ptr);
    if (cache_state == nullptr) return;
    for (auto& layer : cache_state->layers) {
        delete layer.conv_state;
    }
    delete cache_state;
}

} // extern "C"

extern "C" {

// ============================================================================
// Mamba Cache Lifecycle
// ============================================================================

void* mlx_mamba_cache_create(size_t n_layers) {
    auto* cache_state = new MLXMambaCache();
    cache_state->layers.resize(n_layers);
    return cache_state;
}

void mlx_mamba_cache_reset(void* cache_ptr) {
    auto* cache_state = static_cast<MLXMambaCache*>(cache_ptr);
    if (cache_state == nullptr) return;
    for (auto& layer : cache_state->layers) {
        if (layer.conv_state != nullptr) {
            const auto shape = layer.conv_state->shape();
            *layer.conv_state = zeros(shape, float32);
        }
        if (layer.ssm_state != nullptr) {
            const auto shape = layer.ssm_state->shape();
            *layer.ssm_state = zeros(shape, float32);
        }
    }
}

void mlx_mamba_cache_free(void* cache_ptr) {
    auto* cache_state = static_cast<MLXMambaCache*>(cache_ptr);
    if (cache_state == nullptr) return;
    for (auto& layer : cache_state->layers) {
        delete layer.conv_state;
        delete layer.ssm_state;
    }
    delete cache_state;
}

} // extern "C"
