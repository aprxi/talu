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
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// Clears bounded thread-local caches for static weight transforms (e.g.
// oriented/transpose matmul RHS views) used by fused kernels.
extern "C" void mlx_weight_transform_cache_clear();
// Clears cached quantization parameter casts keyed by static handle addresses.
extern "C" void mlx_quant_param_cast_cache_clear();

// Canonicalize RMSNorm weight tensors to rank-1.
// Accepts already-1D tensors, and singleton-expanded forms like [1, 1, D].
// Rejects non-singleton layouts that do not match expected_width.
inline array canonicalize_rms_norm_weight(
    const array& weight,
    int expected_width,
    const char* context
) {
    auto shape_to_string = [&weight]() -> std::string {
        std::string out = "[";
        for (int axis = 0; axis < weight.ndim(); ++axis) {
            if (axis > 0) out += ",";
            out += std::to_string(weight.shape(axis));
        }
        out += "]";
        return out;
    };
    if (weight.ndim() == 1) {
        if (expected_width > 0 && weight.shape(0) != expected_width) {
            throw std::invalid_argument(
                "[" + std::string(context) + "] weight width mismatch (expected=" +
                std::to_string(expected_width) + ", got=" +
                std::to_string(weight.shape(0)) + ", shape=" + shape_to_string() + ")"
            );
        }
        return weight;
    }

    size_t flat_count = 1;
    for (int axis = 0; axis < weight.ndim(); ++axis) {
        const int dim = weight.shape(axis);
        if (dim <= 0) {
            throw std::invalid_argument(
                "[" + std::string(context) + "] invalid non-positive weight dimension"
            );
        }
        flat_count *= static_cast<size_t>(dim);
    }
    if (flat_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "[" + std::string(context) + "] weight element count exceeds reshape limit"
        );
    }
    if (expected_width > 0 && flat_count != static_cast<size_t>(expected_width)) {
        throw std::invalid_argument(
            "[" + std::string(context) + "] weight shape incompatible with input width (expected=" +
            std::to_string(expected_width) + ", flat=" +
            std::to_string(flat_count) + ", shape=" + shape_to_string() + ")"
        );
    }

    return reshape(weight, {static_cast<int>(flat_count)});
}
