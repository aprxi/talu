// MLX Bridge - KV Cache Management
//
// Implements KV cache for transformer inference, matching Python mlx-lm behavior.
// Uses pre-allocated buffers with slice_update for efficiency.

#include "model_state.h"
#include "cache_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace {
constexpr uint64_t kMagicCache = 0x54414C554B564331ULL;      // TALUKVC1
constexpr uint64_t kMagicCausalConv = 0x54414C5543435631ULL; // TALUCCV1
constexpr uint64_t kMagicState = 0x54414C5553534D31ULL;      // TALUSSM1

std::mutex g_kv_cache_mu;
std::unordered_set<void*> g_kv_cache_live;

std::mutex g_causal_conv_cache_mu;
std::unordered_set<void*> g_causal_conv_cache_live;

std::mutex g_state_space_cache_mu;
std::unordered_set<void*> g_state_space_cache_live;

bool is_live_cache(std::mutex& mu, std::unordered_set<void*>& live, void* ptr) {
    if (ptr == nullptr) return false;
    std::lock_guard<std::mutex> lock(mu);
    return live.find(ptr) != live.end();
}

void register_cache(std::mutex& mu, std::unordered_set<void*>& live, void* ptr) {
    if (ptr == nullptr) return;
    std::lock_guard<std::mutex> lock(mu);
    live.insert(ptr);
}

bool unregister_cache(std::mutex& mu, std::unordered_set<void*>& live, void* ptr) {
    if (ptr == nullptr) return false;
    std::lock_guard<std::mutex> lock(mu);
    return live.erase(ptr) > 0;
}

[[noreturn]] void fail_cache_contract(const char* fn, const char* cache_kind, void* ptr, const char* reason) {
    std::fprintf(
        stderr,
        "talu metal cache contract failure: fn=%s cache=%s ptr=%p reason=%s\n",
        fn,
        cache_kind,
        ptr,
        reason
    );
    std::fflush(stderr);
    std::abort();
}
} // namespace

extern "C" {

// ============================================================================
// Cache Lifecycle
// ============================================================================

void* mlx_cache_create(size_t n_layers, size_t max_seq_len) {
    auto cache_state = new MLXCache();
    cache_state->magic = kMagicCache;
    cache_state->layers.resize(n_layers);
    for (auto& layer : cache_state->layers) {
        layer.max_seq_len = max_seq_len;
    }
    register_cache(g_kv_cache_mu, g_kv_cache_live, cache_state);
    return cache_state;
}

void* mlx_cache_create_bfloat16(size_t n_layers, size_t max_seq_len) {
    // Same as mlx_cache_create - cache type determined by usage
    return mlx_cache_create(n_layers, max_seq_len);
}

void mlx_cache_free(void* cache_ptr) {
    if (cache_ptr == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null handle");
    if (!unregister_cache(g_kv_cache_mu, g_kv_cache_live, cache_ptr)) fail_cache_contract(__func__, "kv", cache_ptr, "handle not live");
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    cache_state->magic = 0;
    for (auto& layer : cache_state->layers) {
        delete layer.k_bfloat16;
        delete layer.v_bfloat16;
        delete layer.k_view;
        delete layer.v_view;
    }
    delete cache_state;
}

void mlx_cache_reset(void* cache_ptr) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    if (!is_live_cache(g_kv_cache_mu, g_kv_cache_live, cache_ptr)) fail_cache_contract(__func__, "kv", cache_ptr, "handle not live");
    if (cache_state == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    for (auto& layer : cache_state->layers) {
        layer.offset = 0;
        layer.shapes_initialized = false;
    }
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
    if (cache_state == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    if (layer_idx >= cache_state->layers.size()) fail_cache_contract(__func__, "kv", cache_ptr, "layer index out of range");
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
    // Use cached shapes to avoid heap allocations on decode hot path.
    const size_t offset = prev + num_steps;
    const int offset_int = static_cast<int>(offset);

    if (!layer.shapes_initialized) {
        // First call: initialize cached shapes with constant dimensions.
        layer.update_start = {0, 0, static_cast<int>(prev), 0};
        layer.update_stop_k = {batch, n_kv_heads, offset_int, k_head_dim};
        layer.update_stop_v = {batch, n_kv_heads, offset_int, v_head_dim};
        layer.view_stop_k = {batch, n_kv_heads, offset_int, k_head_dim};
        layer.view_stop_v = {batch, n_kv_heads, offset_int, v_head_dim};
        layer.shapes_initialized = true;
    } else if (
        layer.update_stop_k[0] != batch || layer.update_stop_k[1] != n_kv_heads || layer.update_stop_k[3] != k_head_dim ||
        layer.update_stop_v[0] != batch || layer.update_stop_v[1] != n_kv_heads || layer.update_stop_v[3] != v_head_dim
    ) {
        // Shape changed unexpectedly (mixed-model/interleaved workload): refresh
        // full bounds to preserve correctness.
        layer.update_start = {0, 0, static_cast<int>(prev), 0};
        layer.update_stop_k = {batch, n_kv_heads, offset_int, k_head_dim};
        layer.update_stop_v = {batch, n_kv_heads, offset_int, v_head_dim};
        layer.view_stop_k = {batch, n_kv_heads, offset_int, k_head_dim};
        layer.view_stop_v = {batch, n_kv_heads, offset_int, v_head_dim};
    } else {
        // Decode hot path: only update sequence index (position 2).
        layer.update_start[2] = static_cast<int>(prev);
        layer.update_stop_k[2] = offset_int;
        layer.update_stop_v[2] = offset_int;
        layer.view_stop_k[2] = offset_int;
        layer.view_stop_v[2] = offset_int;
    }

    *layer.k_bfloat16 = slice_update(*layer.k_bfloat16, k_new_arr, layer.update_start, layer.update_stop_k);
    *layer.v_bfloat16 = slice_update(*layer.v_bfloat16, v_new_arr, layer.update_start, layer.update_stop_v);

    layer.offset = offset;

    // Return slice [0:offset] as fresh array objects.
    //
    // CRITICAL: We must create NEW array objects each call, not reuse via
    // assignment. The previous code did `*layer.k_view = slice(...)` which
    // modified the array object's internal state. This caused lazy graphs
    // that referenced the old array to see the new slice bounds, leading to:
    // - Non-deterministic behavior (same seed, different output)
    // - Rapid token repetition (attention reading wrong cache regions)
    //
    // Solution: Use placement new to reconstruct arrays in existing memory.
    // This creates genuinely new array objects (with new shared state) while
    // reusing the allocated memory. Callers who copied the previous arrays
    // retain valid references via their own shared state.
    if (layer.k_view != nullptr) {
        // Destroy old arrays and reconstruct in same memory (zero allocation).
        layer.k_view->~array();
        layer.v_view->~array();
        new (layer.k_view) array(slice(*layer.k_bfloat16, g_slice_start, layer.view_stop_k));
        new (layer.v_view) array(slice(*layer.v_bfloat16, g_slice_start, layer.view_stop_v));
    } else {
        // First call: allocate memory.
        layer.k_view = new array(slice(*layer.k_bfloat16, g_slice_start, layer.view_stop_k));
        layer.v_view = new array(slice(*layer.v_bfloat16, g_slice_start, layer.view_stop_v));
    }

    *k_out = layer.k_view;
    *v_out = layer.v_view;
}

void mlx_cache_get_bfloat16(
    void* cache_ptr, size_t layer_idx,
    void** k_out, void** v_out
) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    if (cache_state == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    if (layer_idx >= cache_state->layers.size()) fail_cache_contract(__func__, "kv", cache_ptr, "layer index out of range");
    auto& layer = cache_state->layers[layer_idx];
    *k_out = layer.k_bfloat16;
    *v_out = layer.v_bfloat16;
}

void mlx_cache_set_full_bfloat16(
    void* cache_ptr, size_t layer_idx,
    const void* k_full, const void* v_full
) {
    auto cache_state = static_cast<MLXCache*>(cache_ptr);
    if (cache_state == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    if (layer_idx >= cache_state->layers.size()) fail_cache_contract(__func__, "kv", cache_ptr, "layer index out of range");
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
    if (cache_state == nullptr) fail_cache_contract(__func__, "kv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCache) fail_cache_contract(__func__, "kv", cache_ptr, "magic mismatch");
    std::vector<array> to_eval;

    const size_t count = std::min(n_layers, cache_state->layers.size());
    for (size_t i = 0; i < count; i++) {
        if (cache_state->layers[i].k_bfloat16) {
            to_eval.push_back(*cache_state->layers[i].k_bfloat16);
            to_eval.push_back(*cache_state->layers[i].v_bfloat16);
        }
    }

    if (!to_eval.empty()) {
        eval(to_eval);
    }
}

bool mlx_cache_is_valid(void* cache_ptr) {
    if (!is_live_cache(g_kv_cache_mu, g_kv_cache_live, cache_ptr)) return false;
    auto* cache_state = static_cast<MLXCache*>(cache_ptr);
    return cache_state != nullptr && cache_state->magic == kMagicCache;
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
// CausalConv Cache Lifecycle
// ============================================================================

void* mlx_causal_conv_cache_create(size_t n_layers) {
    auto* cache_state = new MLXCausalConvCache();
    cache_state->magic = kMagicCausalConv;
    cache_state->layers.resize(n_layers);
    register_cache(g_causal_conv_cache_mu, g_causal_conv_cache_live, cache_state);
    return cache_state;
}

void mlx_causal_conv_cache_reset(void* cache_ptr) {
    auto* cache_state = static_cast<MLXCausalConvCache*>(cache_ptr);
    if (!is_live_cache(g_causal_conv_cache_mu, g_causal_conv_cache_live, cache_ptr)) fail_cache_contract(__func__, "causal_conv", cache_ptr, "handle not live");
    if (cache_state == nullptr) fail_cache_contract(__func__, "causal_conv", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicCausalConv) fail_cache_contract(__func__, "causal_conv", cache_ptr, "magic mismatch");
    for (auto& layer : cache_state->layers) {
        if (layer.conv_state != nullptr) {
            const auto shape = layer.conv_state->shape();
            *layer.conv_state = zeros(shape, layer.conv_state->dtype());
        }
    }
}

void mlx_causal_conv_cache_free(void* cache_ptr) {
    if (cache_ptr == nullptr) fail_cache_contract(__func__, "causal_conv", cache_ptr, "null handle");
    if (!unregister_cache(g_causal_conv_cache_mu, g_causal_conv_cache_live, cache_ptr)) fail_cache_contract(__func__, "causal_conv", cache_ptr, "handle not live");
    auto* cache_state = static_cast<MLXCausalConvCache*>(cache_ptr);
    if (cache_state->magic != kMagicCausalConv) fail_cache_contract(__func__, "causal_conv", cache_ptr, "magic mismatch");
    cache_state->magic = 0;
    for (auto& layer : cache_state->layers) {
        delete layer.conv_state;
    }
    delete cache_state;
}

bool mlx_causal_conv_cache_is_valid(void* cache_ptr) {
    if (!is_live_cache(g_causal_conv_cache_mu, g_causal_conv_cache_live, cache_ptr)) return false;
    auto* cache_state = static_cast<MLXCausalConvCache*>(cache_ptr);
    return cache_state != nullptr && cache_state->magic == kMagicCausalConv;
}

} // extern "C"

extern "C" {

// ============================================================================
// StateSpace Cache Lifecycle
// ============================================================================

void* mlx_state_space_cache_create(size_t n_layers) {
    auto* cache_state = new MLXStateSpaceCache();
    cache_state->magic = kMagicState;
    cache_state->layers.resize(n_layers);
    register_cache(g_state_space_cache_mu, g_state_space_cache_live, cache_state);
    return cache_state;
}

void mlx_state_space_cache_reset(void* cache_ptr) {
    auto* cache_state = static_cast<MLXStateSpaceCache*>(cache_ptr);
    if (!is_live_cache(g_state_space_cache_mu, g_state_space_cache_live, cache_ptr)) fail_cache_contract(__func__, "state_space", cache_ptr, "handle not live");
    if (cache_state == nullptr) fail_cache_contract(__func__, "state_space", cache_ptr, "null cache state");
    if (cache_state->magic != kMagicState) fail_cache_contract(__func__, "state_space", cache_ptr, "magic mismatch");
    std::lock_guard<std::mutex> lock(cache_state->mu);
    for (auto& layer : cache_state->layers) {
        if (layer.conv_state != nullptr) {
            const auto shape = layer.conv_state->shape();
            *layer.conv_state = zeros(shape, layer.conv_state->dtype());
        }
        if (layer.ssm_state != nullptr) {
            const auto shape = layer.ssm_state->shape();
            *layer.ssm_state = zeros(shape, float32);
        }
    }
}

void mlx_state_space_cache_free(void* cache_ptr) {
    if (cache_ptr == nullptr) fail_cache_contract(__func__, "state_space", cache_ptr, "null handle");
    if (!unregister_cache(g_state_space_cache_mu, g_state_space_cache_live, cache_ptr)) fail_cache_contract(__func__, "state_space", cache_ptr, "handle not live");
    auto* cache_state = static_cast<MLXStateSpaceCache*>(cache_ptr);
    if (cache_state->magic != kMagicState) fail_cache_contract(__func__, "state_space", cache_ptr, "magic mismatch");
    std::lock_guard<std::mutex> lock(cache_state->mu);
    cache_state->magic = 0;
    for (auto& layer : cache_state->layers) {
        delete layer.conv_state;
        delete layer.ssm_state;
        delete layer.in_proj_rhs;
        delete layer.out_proj_rhs;
        delete layer.gate_up_rhs;
        delete layer.down_proj_rhs;
        delete layer.gdelta_conv_kernel;
        delete layer.gdelta_conv_bias_row;
        delete layer.gdelta_dt_bias_row;
        delete layer.gdelta_a_log_exp_row;
        delete layer.gdelta_norm_weight_heads;
        delete layer.gdelta_compiled_fn;
    }
    delete cache_state;
}

bool mlx_state_space_cache_is_valid(void* cache_ptr) {
    if (!is_live_cache(g_state_space_cache_mu, g_state_space_cache_live, cache_ptr)) return false;
    auto* cache_state = static_cast<MLXStateSpaceCache*>(cache_ptr);
    return cache_state != nullptr && cache_state->magic == kMagicState;
}

} // extern "C"
