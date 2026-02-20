// MLX Runtime State - inference-owned model/cache containers.
//
// These structs represent model/runtime state used by inference execution.
// They are not compute primitives and should remain owned by inference.

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
    static constexpr int step = 256; // Pre-allocation chunk size.
};

// ============================================================================
// KV Cache - per-model cache containing all layers.
// ============================================================================
struct MLXCache {
    std::vector<CacheLayer> layers;
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
