// MLX Bridge - Fused Neural Network Operations
//
// This file is intentionally a single translation unit to preserve current
// build wiring and symbol/linkage behavior, while implementation is split into
// domain-focused include fragments for maintainability.

#include "compute_common.h"
#include "model_state.h"

extern "C" {

void mlx_cache_update_and_fetch_bfloat16(
    void* cache_ptr,
    size_t layer_idx,
    const void* k_new,
    const void* v_new,
    void** k_out,
    void** v_out,
    bool* is_prefill_out
);

#include "fused_ops_shortconv.inc"
#include "fused_ops_core_mamba.inc"
#include "fused_ops_vision.inc"
#include "fused_ops_attention.inc"
#include "fused_ops_mla.inc"
#include "fused_ops_ffn_moe.inc"

} // extern "C"
