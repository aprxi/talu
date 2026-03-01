// MLX Bridge - Dense Model
//
// Full transformer implementation using dense weights.
// This file is a single translation unit wrapper that composes domain-focused
// implementation fragments for maintainability.

#include "compute_common.h"
#include "model_state.h"
#include "attention_utils.h"
#include "cache_utils.h"
#include <mutex>
#include <unordered_map>

#include "model_dense_shared.inc"
#include "model_dense_forward.inc"

extern "C" {
#include "model_dense_c_api.inc"
} // extern "C"
