// MLX Runtime State - compute-owned model/cache containers.
//
// These structs represent runtime state used by the Metal MLX compute bridge.
// Keep this header local to core/src/compute/metal/mlx so compute does not rely
// on inference include paths.

#pragma once

#include "compute_common.h"

// ============================================================================
// KV Cache Layer - stores K/V tensors for one transformer layer.
// ============================================================================
struct CacheLayer {
    // BFloat16 cache (matches mlx_lm default).
    array* k_bfloat16 = nullptr;
    array* v_bfloat16 = nullptr;

    // View arrays for returning slices (avoid allocation per call).
    array* k_view = nullptr;
    array* v_view = nullptr;

    size_t offset = 0; // Current position in cache.
    size_t max_seq_len = 0; // Optional preallocated cache capacity.
    static constexpr int step = 256; // Pre-allocation chunk size.
};

// ============================================================================
// KV Cache - per-model cache containing all layers.
// ============================================================================
struct MLXCache {
    std::vector<CacheLayer> layers;
    // Explicit decode pipeline state owned by caller-visible cache handle.
    // This avoids hidden global request maps in compute runtime.
    std::optional<array> pipeline_current_token;
};

// ============================================================================
// ShortConv Cache - per-layer recurrent convolution state.
// ============================================================================
struct ShortConvLayer {
    // State layout: [1, d_conv, conv_dim]
    array* conv_state = nullptr;
};

struct MLXShortConvCache {
    std::vector<ShortConvLayer> layers;
};

// ============================================================================
// Mamba Cache - per-layer recurrent state (conv + SSM state tensors).
// ============================================================================
struct MambaLayer {
    // Convolution state layout: [1, d_conv, xbc_len]
    array* conv_state = nullptr;
    // SSM state layout: [1, n_heads, d_head, d_state]
    array* ssm_state = nullptr;
};

struct MLXMambaCache {
    std::vector<MambaLayer> layers;
};
