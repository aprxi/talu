#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_ctx mlx_ctx;

typedef struct mlx_result {
    int32_t prompt_tokens;
    int32_t decode_tokens;
    double prefill_ms;
    double prefill_tps;
    double decode_tps;
} mlx_result;

int32_t mlx_is_available(void);
int32_t mlx_validate_config(const char* model_path);

mlx_ctx* mlx_create(const char* model_id, const char* model_path, int32_t seed);
void mlx_destroy(mlx_ctx* ctx);
int32_t mlx_reset(mlx_ctx* ctx);

int32_t mlx_run(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t decode_tokens,
    int32_t warmup,
    mlx_result* out_result,
    int32_t capture_generated_tokens,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
);

int32_t mlx_prefill_first(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    int32_t* out_first_token
);

int32_t mlx_prefill_logits(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    float* out_logits,
    int32_t logits_len
);

int32_t mlx_embed(
    mlx_ctx* ctx,
    const int32_t* token_ids,
    int32_t token_len,
    int32_t pooling,
    int32_t normalize,
    float* out_embedding,
    int32_t embedding_len
);

int32_t mlx_decode_logits(
    mlx_ctx* ctx,
    int32_t token,
    float* out_logits,
    int32_t logits_len
);

int32_t mlx_decode_logits_batch(
    mlx_ctx* const* ctxs,
    const int32_t* tokens,
    float* const* out_logits_ptrs,
    int32_t batch_size,
    int32_t logits_len
);

int32_t mlx_decode_stream(
    mlx_ctx* ctx,
    int32_t first_token,
    int32_t decode_tokens,
    const int32_t* eos_ids,
    int32_t eos_len,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
);

int32_t mlx_decode_topk_candidates(
    mlx_ctx* ctx,
    int32_t token,
    int32_t top_k,
    float* out_candidate_logits,
    int32_t* out_candidate_ids,
    int32_t* out_candidate_count
);

int32_t mlx_decode_topk_candidates_with_sampling(
    mlx_ctx* ctx,
    int32_t token,
    int32_t top_k,
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    const int32_t* context_ids,
    int32_t context_len,
    float* out_candidate_logits,
    int32_t* out_candidate_ids,
    int32_t* out_candidate_count
);

int32_t mlx_decode_topk_stream(
    mlx_ctx* ctx,
    int32_t first_token,
    int32_t decode_tokens,
    const int32_t* eos_ids,
    int32_t eos_len,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
);

void mlx_tokens_free(int32_t* ids);

const char* mlx_last_error(void);

int32_t talu_metal_xray_should_emit(uint8_t point_id, uint16_t layer, uint32_t position);
int32_t talu_metal_xray_is_enabled(void);
void talu_metal_xray_emit_f32(
    uint8_t point_id,
    uint16_t layer,
    uint32_t token,
    uint32_t position,
    const float* ptr,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    uint32_t dim3,
    uint8_t ndim,
    const char* kernel_name
);

#ifdef __cplusplus
}
#endif
