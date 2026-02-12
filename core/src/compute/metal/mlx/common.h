// MLX Bridge - Common types and utilities.
// This header is shared across all MLX bridge components.
//
// Architecture:
//   Zig --[C ABI]--> extern "C" functions --[C++ internally]--> MLX C++ API
//
// The extern "C" functions provide a C-compatible interface for Zig.
// Internally we use C++ because MLX is a C++ library.

#pragma once

#include "mlx/mlx.h"
#include "mlx/compile.h"
#include "mlx/memory.h"
#include "mlx/backend/metal/metal.h"
#include <optional>
#include <vector>
#include <deque>
#include <chrono>
#include <algorithm>

using namespace mlx::core;

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

    size_t offset = 0;               // Current position in cache.
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

// ============================================================================
// Array Pool - reuses array objects to avoid heap allocations.
// ============================================================================
// MLX arrays are lightweight (~16 bytes, shared_ptr to data).
// But allocating 400+ arrays per token via new/delete adds overhead.
// This pool pre-allocates and reuses array objects.
//
// IMPORTANT: Only call mlx_pool_reset() at the START of a full generation,
// NOT between tokens. Arrays from decode step N must stay valid for step N+1.
// ============================================================================

extern thread_local std::deque<std::optional<array>> g_array_pool;
extern thread_local size_t g_pool_index;

// Pool an array and return pointer (for returning to Zig).
void* pool_array(array&& result);
void* pool_array(const array& result);

// ============================================================================
// Pre-computed constants (avoid per-call allocations).
// ============================================================================
extern const std::vector<int> g_transpose_perm;  // {0, 2, 1, 3}
extern const Shape g_slice_start;                 // {0, 0, 0, 0}
