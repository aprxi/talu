// MLX Bridge - Quantized (4-bit) Model
//
// Full transformer implementation using 4-bit quantized weights.
// This file is a single translation unit wrapper that composes domain-focused
// implementation fragments for maintainability.

#include "compute_common.h"
#include "model_state.h"
#include "attention_utils.h"
#include "cache_utils.h"
#include <mutex>
#include <unordered_map>

#include "model_quantized_shared.inc"
#include "model_quantized_forward.inc"

extern "C" {
#include "model_quantized_c_api.inc"
} // extern "C"
