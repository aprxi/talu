// MLX Bridge - Compute-level common types and utilities.
// This header is intentionally model-agnostic.
//
// It only exposes:
// - MLX includes and namespace usage
// - array pool declarations
// - shared compile-time constants
//
// Model/runtime state containers (for example KV cache structs) must live
// outside compute and be owned by inference runtime code.

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
extern const std::vector<int> g_transpose_perm; // {0, 2, 1, 3}
extern const Shape g_slice_start; // {0, 0, 0, 0}
