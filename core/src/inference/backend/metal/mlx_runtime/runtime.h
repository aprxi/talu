#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlx_ctx mlx_ctx;

typedef struct mlx_model_flags {
    uint8_t use_layer_q_norm_head_dim;
    uint8_t allow_norm_shift;
    uint8_t use_v_norm;
    uint8_t _pad;
    float embedding_multiplier;
    float attention_multiplier;
} mlx_model_flags;

typedef struct mlx_result {
    int32_t prompt_tokens;
    int32_t decode_tokens;
    double prefill_ms;
    double prefill_tps;
    double decode_tps;
} mlx_result;

typedef struct mlx_vision_grid {
    uint32_t temporal;
    uint32_t height;
    uint32_t width;
} mlx_vision_grid;

typedef struct mlx_prefill_vision_input {
    const float* merged_embeddings;
    uint32_t merged_value_count;
    uint32_t image_token_id;
} mlx_prefill_vision_input;

typedef struct mlx_init_diagnostics {
    uint8_t has_nvfp4_meta;
    uint8_t has_grouped_affine_meta;
    uint8_t decode_qmm_enabled;
    uint8_t lm_head_quantized;
    uint8_t per_layer_input_enabled;
    uint8_t use_gelu;
    uint8_t converted_nvfp4_model;
    uint8_t nvfp4_mmap_required;
    uint32_t quant_group_size;
    uint32_t quant_bits;
    uint32_t per_layer_hidden_size;
    uint64_t layer_quantized_count;
    uint64_t layer_total_count;
    uint64_t ssm_qkvz_quantized;
    uint64_t ssm_qkvz_total;
    uint64_t ssm_ba_quantized;
    uint64_t ssm_ba_total;
    uint64_t ssm_out_quantized;
    uint64_t ssm_out_total;
    uint64_t attn_qkv_quantized;
    uint64_t attn_qkv_total;
    uint64_t attn_o_quantized;
    uint64_t attn_o_total;
    uint64_t mlp_quantized;
    uint64_t mlp_total;
    uint64_t strict_requested;
    uint64_t strict_active;
    uint64_t strict_required;
    uint64_t strict_quantized;
    uint64_t strict_missing;
    uint64_t dense_decode_bytes;
    uint64_t ctx_bytes;
    uint64_t layer_quantized_bytes;
    uint64_t layer_dense_bytes;
    uint64_t layer_other_bytes;
} mlx_init_diagnostics;

int32_t mlx_is_available(void);
int32_t mlx_validate_config(const char* model_path);

mlx_ctx* mlx_create(const char* model_id, const char* model_path, int32_t seed);
mlx_ctx* mlx_create_with_flags(const char* model_id, const char* model_path, int32_t seed, const mlx_model_flags* flags);
mlx_ctx* mlx_clone(mlx_ctx* source_ctx, int32_t seed);
void mlx_destroy(mlx_ctx* ctx);
int32_t mlx_reset(mlx_ctx* ctx);
int32_t mlx_get_init_diagnostics(const mlx_ctx* ctx, mlx_init_diagnostics* out_diagnostics);

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

int32_t mlx_prefill_logits_with_vision(
    mlx_ctx* ctx,
    const int32_t* prompt_ids,
    int32_t prompt_len,
    const mlx_prefill_vision_input* vision_input,
    float* out_logits,
    int32_t logits_len
);

int32_t mlx_prefill_logits_batch(
    mlx_ctx* const* ctxs,
    const int32_t* const* prompt_ids_ptrs,
    const int32_t* prompt_lens,
    float* const* out_logits_ptrs,
    int32_t batch_size,
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

int32_t mlx_decode_topk_candidates_batch(
    mlx_ctx* const* ctxs,
    const int32_t* tokens,
    int32_t top_k,
    float* const* out_candidate_logits_ptrs,
    int32_t* const* out_candidate_ids_ptrs,
    int32_t* out_candidate_counts,
    int32_t batch_size
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
    float repetition_penalty,
    float presence_penalty,
    float frequency_penalty,
    int32_t** out_generated_ids,
    int32_t* out_generated_len
);

void mlx_tokens_free(int32_t* ids);

const char* mlx_last_error(void);
const char* mlx_runtime_binary_dir(void);

int32_t mlx_test_grouped_affine_moe_gpu_path(void);
int32_t mlx_test_nvfp4_moe_gather_path(void);
int32_t mlx_test_depthwise_conv_decode_step(void);
int32_t mlx_test_single_query_attention_matches_sdpa(void);
int32_t mlx_test_kv_cache_reserve_preserves_prefix(void);
int32_t mlx_test_shared_expert_gate_up_fusion(void);
int32_t mlx_test_dense_mlp_gate_up_fusion(void);
int32_t mlx_test_full_attention_qkv_fusion(void);
int32_t mlx_test_grouped_affine_prefill_cache_helper(void);
int32_t mlx_test_gated_delta_no_double_qk_norm(void);
int32_t mlx_test_rmsnorm_gated_kernel_matches_reference(void);
int32_t mlx_test_chunked_prefill_tail_matches_full_prompt(void);
int32_t mlx_test_linear_attention_fused_quant_inproj_reuse(void);
int32_t mlx_test_nvfp4_rowwise_post_scale_linear_decode(void);
int32_t mlx_test_talu_meta_nvfp4_detection(void);
int32_t mlx_test_nvfp4_mmap_strict_policy(void);
int32_t mlx_test_dense_lm_head_lhs_primary(void);
int32_t mlx_test_grouped_affine_embedding_lookup_matches_reference(void);
int32_t mlx_test_linear_attention_fused_mixer_matches_reference(void);
int32_t mlx_test_topk_candidate_extraction_multi(void);

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
