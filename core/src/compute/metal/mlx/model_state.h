// MLX Runtime State - compute-owned model/cache containers.
//
// These structs represent runtime state used by the Metal MLX compute bridge.
// Keep this header local to core/src/compute/metal/mlx so compute does not rely
// on inference include paths.

#pragma once

#include "compute_common.h"
#include <cstdint>
#include <mutex>

// ============================================================================
// KV Cache Layer - stores K/V tensors for one transformer layer.
// ============================================================================
struct CacheLayer {
    // BFloat16 cache (matches mlx_lm default).
    array* k_bfloat16 = nullptr;
    array* v_bfloat16 = nullptr;

    // View arrays returned to callers. These are recreated (via placement new)
    // on each updateAndFetch call to avoid lazy graph reference invalidation.
    // Memory is reused but array objects are reconstructed to ensure old
    // references remain valid via their shared state.
    array* k_view = nullptr;
    array* v_view = nullptr;

    size_t offset = 0; // Current position in cache.
    size_t max_seq_len = 0; // Optional preallocated cache capacity.
    static constexpr int step = 256; // Pre-allocation chunk size.

    // Cached shapes for decode hot path. Initialized on first use, then only
    // the sequence index (position 2) is updated on subsequent calls. This
    // eliminates 5 std::vector heap allocations per layer per decode step.
    bool shapes_initialized = false;
    Shape update_start = {0, 0, 0, 0};   // [0, 0, prev, 0]
    Shape update_stop_k = {0, 0, 0, 0};  // [batch, n_kv_heads, offset, k_head_dim]
    Shape update_stop_v = {0, 0, 0, 0};  // [batch, n_kv_heads, offset, v_head_dim]
    Shape view_stop_k = {0, 0, 0, 0};    // [batch, n_kv_heads, offset, k_head_dim]
    Shape view_stop_v = {0, 0, 0, 0};    // [batch, n_kv_heads, offset, v_head_dim]
};

// ============================================================================
// KV Cache - per-model cache containing all layers.
// ============================================================================
struct MLXCache {
    uint64_t magic = 0;
    std::vector<CacheLayer> layers;
};

// ============================================================================
// CausalConv Cache - per-layer recurrent convolution state.
// ============================================================================
struct CausalConvLayer {
    // State layout: [1, d_conv, conv_dim]
    array* conv_state = nullptr;
};

struct MLXCausalConvCache {
    uint64_t magic = 0;
    std::vector<CausalConvLayer> layers;
};

// ============================================================================
// StateSpace Cache - per-layer recurrent state (conv + SSM state tensors).
// ============================================================================
struct StateSpaceLayer {
    // Convolution state layout: [1, d_conv, xbc_len]
    array* conv_state = nullptr;
    // SSM state layout: [1, n_heads, d_head, d_state]
    array* ssm_state = nullptr;

    // Decode hot-path cached dense RHS tensors for this layer.
    // These are derived from immutable model weights and reused across tokens.
    array* in_proj_rhs = nullptr;
    array* out_proj_rhs = nullptr;
    array* gate_up_rhs = nullptr;
    array* down_proj_rhs = nullptr;
    const void* in_proj_handle = nullptr;
    const void* out_proj_handle = nullptr;
    const void* gate_up_handle = nullptr;
    const void* down_proj_handle = nullptr;
    int hidden_dim = 0;
    int d_inner = 0;
    int d_ff = 0;

    // Gated-delta decode hot-path cached resolved tensors.
    array* gdelta_conv_kernel = nullptr;
    array* gdelta_conv_bias_row = nullptr;
    array* gdelta_dt_bias_row = nullptr;
    array* gdelta_a_log_exp_row = nullptr;
    array* gdelta_norm_weight_heads = nullptr;
    const void* gdelta_conv_weight_handle = nullptr;
    const void* gdelta_conv_bias_handle = nullptr;
    const void* gdelta_dt_bias_handle = nullptr;
    const void* gdelta_a_log_handle = nullptr;
    const void* gdelta_norm_weight_handle = nullptr;
    int gdelta_qkv_len = 0;
    int gdelta_n_heads = 0;
    int gdelta_d_head = 0;

    // Compiled MLX decode function for the fused gated-delta single-token path.
    // Signature: inputs = {x, conv_state, ssm_state},
    //            outputs = {out, new_conv_state, new_ssm_state}.
    // Captures static weight arrays from the most recent refresh cycle.
    // nullptr until first decode after any weight-handle change. Invalidated
    // when need_static_refresh or need_proj_rhs_refresh triggers.
    std::function<std::vector<array>(const std::vector<array>&)>* gdelta_compiled_fn = nullptr;

};

struct MLXStateSpaceCache {
    uint64_t magic = 0;
    std::mutex mu;
    std::vector<StateSpaceLayer> layers;
};
