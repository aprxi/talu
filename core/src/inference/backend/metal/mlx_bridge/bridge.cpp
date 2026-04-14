#include "bridge.h"
#include "config_parse.h"

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/mlx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <tuple>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlx::core;

extern "C" void* mlx_lazy_fused_expert_mix_ffn_mxfp4(
    const void* input,
    const void* router_w,
    const void* router_s,
    const void* router_b,
    const void* router_bias,
    const void* gate_w,
    const void* gate_s,
    const void* up_w,
    const void* up_s,
    const void* down_w,
    const void* down_s,
    const void* gate_bias,
    const void* up_bias,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t router_group_size,
    size_t expert_group_size
);

extern "C" void* mlx_lazy_fused_expert_mix_ffn_affine(
    const void* input,
    const void* router_w,
    const void* router_bias,
    const void* gate_w,
    const void* gate_s,
    const void* gate_bias,
    const void* up_w,
    const void* up_s,
    const void* up_bias,
    const void* down_w,
    const void* down_s,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t expert_group_size,
    int expert_bits,
    bool use_gelu
);

extern "C" void* mlx_lazy_fused_expert_mix_ffn_affine_fused_gate_up(
    const void* input,
    const void* router_w,
    const void* router_bias,
    const void* gate_up_w,
    const void* gate_up_s,
    const void* gate_up_bias,
    const void* down_w,
    const void* down_s,
    const void* down_bias,
    size_t num_experts,
    size_t experts_per_token,
    size_t expert_group_size,
    int expert_bits,
    bool use_gelu
);


// ---------------------------------------------------------------------------
// Anonymous namespace: internal state, sampling, types, weights, ops
// ---------------------------------------------------------------------------
namespace {

thread_local std::string g_last_error;
std::mutex g_wired_limit_mutex;
size_t g_wired_limit_refcount = 0;
size_t g_wired_limit_previous = 0;
bool g_wired_limit_active = false;
std::mutex g_rng_mutex;
size_t g_active_ctx_count = 0;
thread_local size_t g_fused_lm_head_argmax_hits = 0;
thread_local std::vector<array> g_decode_batch_logits_flat_scratch;
thread_local std::vector<array> g_prefill_batch_logits_flat_scratch;
thread_local std::vector<array> g_decode_topk_batch_logits_flat_scratch;
thread_local std::vector<array> g_decode_topk_batch_ids_flat_scratch;
thread_local bool g_nvfp4_mmap_required = false;
size_t g_qmm_attempts = 0;
size_t g_qmm_success = 0;

void enter_rng_epoch(int32_t seed) {
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    if (g_active_ctx_count == 0) {
        random::seed(static_cast<uint64_t>(seed));
    }
    g_active_ctx_count += 1;
}

void leave_rng_epoch() {
    std::lock_guard<std::mutex> lock(g_rng_mutex);
    if (g_active_ctx_count == 0) return;
    g_active_ctx_count -= 1;
}

bool acquire_wired_limit() {
    const auto& info = device_info(Device::gpu);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end() || !std::holds_alternative<size_t>(it->second)) {
        return false;
    }
    const size_t target_wired = std::get<size_t>(it->second);

    std::lock_guard<std::mutex> lock(g_wired_limit_mutex);
    if (g_wired_limit_refcount == 0) {
        g_wired_limit_previous = set_wired_limit(target_wired);
        g_wired_limit_active = true;
    }
    g_wired_limit_refcount += 1;
    return true;
}

void release_wired_limit() {
    std::lock_guard<std::mutex> lock(g_wired_limit_mutex);
    if (g_wired_limit_refcount == 0) return;
    g_wired_limit_refcount -= 1;
    if (g_wired_limit_refcount == 0 && g_wired_limit_active) {
        set_wired_limit(g_wired_limit_previous);
        g_wired_limit_active = false;
        g_wired_limit_previous = 0;
    }
}

#include "bridge_sampling.inc"
#include "bridge_types.inc"
#include "bridge_weight_utils.inc"
#include "bridge_quantization.inc"
#include "bridge_ops.inc"

} // namespace

// ---------------------------------------------------------------------------
// mlx_ctx — central inference context (visible to all sections below)
// ---------------------------------------------------------------------------
struct mlx_ctx {
    std::string model_id;
    std::string model_path;

    BridgeModelConfig cfg;

    array embed_tokens = array(0.0f); // [vocab, hidden]
    array embed_tokens_q_w = array(0.0f); // [vocab, hidden * bits / 32] packed
    array embed_tokens_q_scales = array(0.0f); // [vocab, hidden / group_size]
    array embed_tokens_q_biases = array(0.0f); // [vocab, hidden / group_size]
    int embed_tokens_q_bits = 0; // detected quantization bits for embeddings
    bool embed_tokens_has_q = false;
    array lm_head_rhs = array(0.0f);  // [hidden, vocab]
    array lm_head_q_w = array(0.0f);  // [vocab, hidden * bits / 32] packed
    array lm_head_q_scales = array(0.0f); // [vocab, hidden / group_size]
    array lm_head_q_biases = array(0.0f); // [vocab, hidden / group_size] affine-only
    int lm_head_q_bits = 0; // detected quantization bits for lm_head (may differ from cfg)
    array final_norm_w = array(0.0f); // [hidden]
    bool has_fp8_meta = false;
    bool has_mxfp8_meta = false;
    bool has_grouped_affine_meta = false;
    bool lm_head_q_decode_enabled = false;
    bool fp8_decode_qmm_enabled = false;
    bool per_layer_input_enabled = false;
    int per_layer_hidden_size = 0;
    float per_layer_input_scale = 0.70710677f; // 2^-0.5
    float per_layer_embed_scale = 1.0f;
    float per_layer_model_projection_scale = 1.0f;
    array per_layer_embedding = array(0.0f); // [vocab, layers*hpl]
    array per_layer_projection_norm_w = array(0.0f); // [hpl]
    // Retained only in mmap-strict mode to keep zero-copy views alive.
    std::unordered_map<std::string, array> retained_weight_tensors;

    std::vector<LayerWeights> layers;
    std::vector<KVCacheState> kv_cache;
    std::vector<LinearCacheState> linear_cache;
    int kv_reserve_tokens = 0;
    bool has_moe_layers = false;
    bool profile_layers = false;
    bool stream_ready = false;
    uint32_t trace_decode_token = 1;
    bool xray_enabled = false;
    std::unordered_map<int32_t, int32_t> sampling_context_counts;
    int32_t sampling_context_len = 0;
    std::vector<int32_t> sampling_unique_ids;
    std::vector<float> sampling_repetition_scales;
    std::vector<float> sampling_additive_penalties;
};

// ---------------------------------------------------------------------------
// Anonymous namespace: diagnostics, layers, forward pass, batching
// ---------------------------------------------------------------------------
namespace {

#include "bridge_diagnostics.inc"
#include "bridge_layers.inc"
#include "bridge_moe.inc"
#include "bridge_forward.inc"
#include "bridge_batch.inc"

} // namespace

// ---------------------------------------------------------------------------
// Public C API + tests
// ---------------------------------------------------------------------------
extern "C" {

#include "bridge_api_init.inc"
#include "bridge_api_prefill.inc"
#include "bridge_api_decode.inc"
#include "bridge_api_stream.inc"
#include "bridge_api_run.inc"
#include "bridge_tests_a.inc"
#include "bridge_tests_b.inc"

void mlx_tokens_free(int32_t* ids) {
    std::free(ids);
}

const char* mlx_last_error(void) {
    return g_last_error.c_str();
}

const char* mlx_runtime_binary_dir(void) {
    thread_local std::string dir = current_binary_dir().string();
    return dir.c_str();
}

} // extern "C"
