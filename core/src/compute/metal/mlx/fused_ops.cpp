// MLX Bridge - Fused Neural Network Operations
//
// This file is intentionally a single translation unit to preserve current
// build wiring and symbol/linkage behavior, while implementation is split into
// domain-focused include fragments for maintainability.

#include "compute_common.h"
#include "model_state.h"
#include "causal_conv_utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
void mlx_cache_get_bfloat16(
    void* cache_ptr,
    size_t layer_idx,
    void** k_out,
    void** v_out
);
void mlx_cache_set_full_bfloat16(
    void* cache_ptr,
    size_t layer_idx,
    const void* k_full,
    const void* v_full
);

#include "fused_ops_core_state_space_reference.inc"
#include "fused_ops_core_state_space.inc"
#include "fused_ops_causal_conv.inc"
#include "fused_ops_vision.inc"
#include "fused_ops_attention.inc"
#include "fused_ops_attention_reference.inc"
#include "fused_ops_mla.inc"
#include "fused_ops_ffn_expert_mix.inc"

} // extern "C"
