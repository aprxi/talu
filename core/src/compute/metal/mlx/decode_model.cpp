// Unified decode-model bridge over dense and quantized MLX model runtimes.

#include "compute_common.h"

#include <cstddef>
#include <cstdint>
#include <new>

extern "C" {

void mlx_fused_model_free(void* model);
void mlx_dense_model_free(void* model);

void* mlx_fused_decode_step_logits(void* model, void* cache, void* shortconv_cache, void* mamba_cache, uint32_t token_id, size_t pos_offset);
void* mlx_dense_decode_step_logits(void* model, void* cache, void* shortconv_cache, void* mamba_cache, uint32_t token_id, size_t pos_offset);

uint32_t mlx_fused_decode_batch(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    uint32_t first_token,
    size_t start_pos,
    uint32_t* out_tokens,
    size_t max_tokens,
    const uint32_t* eos_ids,
    size_t n_eos_ids);

uint32_t mlx_dense_decode_batch(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    uint32_t first_token,
    size_t start_pos,
    uint32_t* out_tokens,
    size_t max_tokens,
    const uint32_t* eos_ids,
    size_t n_eos_ids);

void mlx_pipeline_prime(void* model, void* cache, void* shortconv_cache, void* mamba_cache, uint32_t first_token_id, size_t pos_offset);
void mlx_dense_pipeline_prime(void* model, void* cache, void* shortconv_cache, void* mamba_cache, uint32_t first_token_id, size_t pos_offset);

uint32_t mlx_pipeline_step(void* model, void* cache, void* shortconv_cache, void* mamba_cache, size_t pos_offset);
uint32_t mlx_dense_pipeline_step(void* model, void* cache, void* shortconv_cache, void* mamba_cache, size_t pos_offset);

uint32_t mlx_pipeline_flush(void* model, void* cache, void* shortconv_cache, void* mamba_cache);
uint32_t mlx_dense_pipeline_flush(void* model, void* cache, void* shortconv_cache, void* mamba_cache);

} // extern "C"

namespace {

enum class DecodeModelFlavor : uint8_t {
    quantized = 0,
    dense = 1,
};

struct DecodeModelWrapper {
    DecodeModelFlavor flavor;
    void* impl;
};

inline DecodeModelWrapper* as_wrapper(void* model) {
    return static_cast<DecodeModelWrapper*>(model);
}

} // namespace

extern "C" {

void* mlx_decode_model_wrap_fused(void* model) {
    if (model == nullptr) return nullptr;
    auto* wrapper = new (std::nothrow) DecodeModelWrapper{DecodeModelFlavor::quantized, model};
    return static_cast<void*>(wrapper);
}

void* mlx_decode_model_wrap_dense(void* model) {
    if (model == nullptr) return nullptr;
    auto* wrapper = new (std::nothrow) DecodeModelWrapper{DecodeModelFlavor::dense, model};
    return static_cast<void*>(wrapper);
}

void mlx_decode_model_free(void* model) {
    if (model == nullptr) return;
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        mlx_fused_model_free(wrapper->impl);
    } else {
        mlx_dense_model_free(wrapper->impl);
    }
    delete wrapper;
}

void* mlx_decode_model_step_logits(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    uint32_t token_id,
    size_t pos_offset) {
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        return mlx_fused_decode_step_logits(wrapper->impl, cache, shortconv_cache, mamba_cache, token_id, pos_offset);
    }
    return mlx_dense_decode_step_logits(wrapper->impl, cache, shortconv_cache, mamba_cache, token_id, pos_offset);
}

void mlx_decode_model_step_logits_batch(
    void* model,
    void* const* caches,
    void* const* shortconv_caches,
    void* const* mamba_caches,
    const uint32_t* token_ids,
    const size_t* pos_offsets,
    void** out_logits,
    size_t count) {
    auto* wrapper = as_wrapper(model);
    std::vector<array> pending_eval;
    pending_eval.reserve(count);

    for (size_t idx = 0; idx < count; idx++) {
        const uint32_t token_id = token_ids[idx];
        const size_t pos_offset = pos_offsets[idx];
        void* logits = (wrapper->flavor == DecodeModelFlavor::quantized)
            ? mlx_fused_decode_step_logits(wrapper->impl, caches[idx], shortconv_caches[idx], mamba_caches[idx], token_id, pos_offset)
            : mlx_dense_decode_step_logits(wrapper->impl, caches[idx], shortconv_caches[idx], mamba_caches[idx], token_id, pos_offset);
        out_logits[idx] = logits;
        pending_eval.push_back(*static_cast<array*>(logits));
    }

    if (!pending_eval.empty()) {
        eval(pending_eval);
    }
}

uint32_t mlx_decode_model_decode_batch(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    uint32_t first_token,
    size_t start_pos,
    uint32_t* out_tokens,
    size_t max_tokens,
    const uint32_t* eos_ids,
    size_t n_eos_ids) {
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        return mlx_fused_decode_batch(
            wrapper->impl,
            cache,
            shortconv_cache,
            mamba_cache,
            first_token,
            start_pos,
            out_tokens,
            max_tokens,
            eos_ids,
            n_eos_ids);
    }
    return mlx_dense_decode_batch(
        wrapper->impl,
        cache,
        shortconv_cache,
        mamba_cache,
        first_token,
        start_pos,
        out_tokens,
        max_tokens,
        eos_ids,
        n_eos_ids);
}

void mlx_decode_model_pipeline_prime(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    uint32_t first_token_id,
    size_t pos_offset) {
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        mlx_pipeline_prime(wrapper->impl, cache, shortconv_cache, mamba_cache, first_token_id, pos_offset);
    } else {
        mlx_dense_pipeline_prime(wrapper->impl, cache, shortconv_cache, mamba_cache, first_token_id, pos_offset);
    }
}

uint32_t mlx_decode_model_pipeline_step(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache,
    size_t pos_offset) {
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        return mlx_pipeline_step(wrapper->impl, cache, shortconv_cache, mamba_cache, pos_offset);
    }
    return mlx_dense_pipeline_step(wrapper->impl, cache, shortconv_cache, mamba_cache, pos_offset);
}

uint32_t mlx_decode_model_pipeline_flush(
    void* model,
    void* cache,
    void* shortconv_cache,
    void* mamba_cache) {
    auto* wrapper = as_wrapper(model);
    if (wrapper->flavor == DecodeModelFlavor::quantized) {
        return mlx_pipeline_flush(wrapper->impl, cache, shortconv_cache, mamba_cache);
    }
    return mlx_dense_pipeline_flush(wrapper->impl, cache, shortconv_cache, mamba_cache);
}

} // extern "C"
