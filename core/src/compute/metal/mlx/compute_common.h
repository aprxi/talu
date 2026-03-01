// MLX Bridge - Compute-level common types and utilities.
// This header is intentionally model-agnostic.
//
// It only exposes:
// - MLX includes and namespace usage
// - array pool declarations
// - shared compile-time constants
//
// Runtime state containers for the MLX backend live in model_state.h alongside
// this compute bridge code to keep includes local to core/src/compute/metal/mlx.

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
#include <functional>
#include <mutex>
#include <unordered_set>

using namespace mlx::core;

// ============================================================================
// Array Pool - reuses array objects to avoid heap allocations.
// ============================================================================
// MLX arrays are lightweight (~16 bytes, shared_ptr to data).
// But allocating 400+ arrays per step via new/delete adds overhead.
// This pool pre-allocates and reuses array objects.
//
// IMPORTANT: Only call mlx_pool_reset() at the START of a full run,
// NOT between iterative steps. Arrays from step N must stay valid for step N+1.
// ============================================================================

extern thread_local std::deque<std::optional<array>> g_array_pool;
extern thread_local size_t g_pool_index;

// Pool an array and return pointer (for returning to Zig).
void* pool_array(array&& result);
void* pool_array(const array& result);

// Allocate a heap-owned array handle tracked for mlx_array_free().
void* make_owned_array(array&& result);

// ============================================================================
// Pre-computed constants (avoid per-call allocations).
// ============================================================================
extern const std::vector<int> g_transpose_perm; // {0, 2, 1, 3}
extern const Shape g_slice_start; // {0, 0, 0, 0}

// ============================================================================
// Lightweight op counting (test instrumentation).
// ============================================================================
extern thread_local bool g_count_ops_enabled;
extern thread_local size_t g_count_ops_value;

inline void mlx_count_op(size_t n = 1) {
    if (g_count_ops_enabled) {
        g_count_ops_value += n;
    }
}
