//! CUDA runtime scratch, logits, and projection buffers.

const std = @import("std");
const models = @import("models_pkg");
const compute = @import("compute_pkg");
const log = @import("log_pkg");
const attention_policy = @import("../attention_policy.zig");

const LoadedModel = models.LoadedModel;
const config = @import("config.zig");
const weights = @import("weights.zig");

const attention_policy_config = config.attention_policy_config;
const default_prefill_chunk_rows_cap = config.default_prefill_chunk_rows_cap;
const enable_device_embedding_lookup = config.enable_device_embedding_lookup;
const EmbeddingLookup = weights.EmbeddingLookup;
const LinearWeight = weights.LinearWeight;
const missing_device_tensor = weights.missing_device_tensor;

const engine_weights = @import("../weights/root.zig");
const uploadLinearWeight = engine_weights.uploadLinearWeight;
const tryPopulateFinalNormWeight = engine_weights.tryPopulateFinalNormWeight;
const canUseModelEmbeddings = engine_weights.canUseModelEmbeddings;
const tryUploadEmbeddingLookup = engine_weights.tryUploadEmbeddingLookup;
const resizeScratchBuffer = engine_weights.resizeScratchBuffer;

pub const RuntimeBuffers = struct {
    projected_vocab: usize,
    max_dff: usize,
    max_attn: usize,
    max_kv: usize,
    max_gdelta_proj: usize,
    max_seq_len: usize,
    head_dim: usize,
    shortconv_dim: usize,
    row_capacity: usize,
    using_model_norm: bool,
    using_model_projection: bool,
    projection_from_lm_head: bool,
    using_model_embeddings: bool,
    embedding_lookup: ?EmbeddingLookup,
    hidden_host: []f32,
    projected_logits_host: []f32,
    projected_logits_batch_host: []f32,
    prefill_tokens_dev: compute.cuda.Buffer,
    input_dev: compute.cuda.Buffer,
    norm_weight_dev: compute.cuda.Buffer,
    norm_out_dev: compute.cuda.Buffer,
    activation_u16_dev: compute.cuda.Buffer,
    attn_q_dev: compute.cuda.Buffer,
    query_gate_proj_dev: compute.cuda.Buffer,
    attn_k_dev: compute.cuda.Buffer,
    attn_v_dev: compute.cuda.Buffer,
    attn_context_dev: compute.cuda.Buffer,
    decode_key_cache_ptrs_host: []u64,
    decode_value_cache_ptrs_host: []u64,
    decode_attn_key_cache_ptrs_table_host: []u64,
    decode_attn_value_cache_ptrs_table_host: []u64,
    decode_attn_k_scale_ptrs_table_host: []u64,
    decode_attn_v_scale_ptrs_table_host: []u64,
    decode_seq_lens_host: []u32,
    decode_positions_host: []u32,
    decode_gd_conv_state_ptrs_table_host: []u64,
    decode_gd_ssm_state_ptrs_table_host: []u64,
    decode_gd_conv_ring_heads_table_host: []u32,
    decode_key_cache_ptrs_dev: compute.cuda.Buffer,
    decode_value_cache_ptrs_dev: compute.cuda.Buffer,
    decode_attn_key_cache_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_value_cache_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_k_scale_ptrs_table_dev: compute.cuda.Buffer,
    decode_attn_v_scale_ptrs_table_dev: compute.cuda.Buffer,
    decode_seq_lens_dev: compute.cuda.Buffer,
    decode_positions_dev: compute.cuda.Buffer,
    decode_gd_conv_state_ptrs_table_dev: compute.cuda.Buffer,
    decode_gd_ssm_state_ptrs_table_dev: compute.cuda.Buffer,
    decode_gd_conv_ring_heads_table_dev: compute.cuda.Buffer,
    attn_scores_dev: ?compute.cuda.Buffer,
    attn_probs_dev: ?compute.cuda.Buffer,
    attn_out_dev: compute.cuda.Buffer,
    ffn_gate_dev: compute.cuda.Buffer,
    ffn_up_dev: compute.cuda.Buffer,
    ffn_mul_dev: compute.cuda.Buffer,
    ffn_down_dev: compute.cuda.Buffer,
    deepstack_add_dev: compute.cuda.Buffer,
    shortconv_proj_dev: compute.cuda.Buffer,
    shortconv_conv_dev: compute.cuda.Buffer,
    gdelta_proj_dev: compute.cuda.Buffer,
    gdelta_ssm_dev: compute.cuda.Buffer,
    projection_weight: LinearWeight,
    logits_dev: compute.cuda.Buffer,
    topk_values_dev: compute.cuda.Buffer,
    topk_ids_dev: compute.cuda.Buffer,
    batched_attn_scores_dev: compute.cuda.Buffer,
    batched_attn_probs_dev: compute.cuda.Buffer,
    batched_attn_max_seq_len: u32,
    dequant_f16_dev: compute.cuda.Buffer,

    pub fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_dff: usize,
        max_attn: usize,
        max_kv: usize,
        max_gdelta_proj: usize,
        max_shortconv_dim: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        max_batch_size: usize,
        max_attn_layers: usize,
        max_gd_layers: usize,
        skip_embedding: bool,
        skip_projection: bool,
    ) !RuntimeBuffers {
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        if (d_model == 0 or vocab_size == 0) return error.InvalidArgument;
        if (max_dff == 0) return error.InvalidArgument;
        if (max_attn == 0) return error.InvalidArgument;
        if (max_kv == 0 or max_gdelta_proj == 0 or max_seq_len == 0 or head_dim == 0) return error.InvalidArgument;
        if (max_batch_size == 0) return error.InvalidArgument;
        if (max_attn_layers == 0) return error.InvalidArgument;
        if (max_gd_layers == 0) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_gdelta_proj_bytes = std.math.mul(usize, max_gdelta_proj, @sizeOf(f32)) catch return error.InvalidArgument;
        const max_linear_in_dim = @max(@max(d_model, max_dff), @max(max_attn, max_gdelta_proj));
        const activation_u16_bytes = std.math.mul(usize, max_linear_in_dim, @sizeOf(u16)) catch return error.InvalidArgument;
        const shortconv_dim = if (max_shortconv_dim > 0) max_shortconv_dim else 1;
        const shortconv_proj_bytes = std.math.mul(usize, shortconv_dim * 3, @sizeOf(f32)) catch return error.InvalidArgument;
        const shortconv_conv_bytes = std.math.mul(usize, shortconv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        const need_attention_score_buffers = attention_policy.needAttentionScoreBuffers(
            attention_policy_config,
            max_seq_len,
            head_dim,
        );
        const attn_rows = std.math.mul(usize, max_seq_len, n_heads) catch return error.InvalidArgument;
        const attn_rows_bytes = std.math.mul(usize, attn_rows, @sizeOf(f32)) catch return error.InvalidArgument;
        const hidden_host = try allocator.alloc(f32, d_model);
        errdefer allocator.free(hidden_host);

        const norm_weight_host = try allocator.alloc(f32, d_model);
        defer allocator.free(norm_weight_host);
        const using_model_norm = tryPopulateFinalNormWeight(loaded, norm_weight_host);
        if (!using_model_norm) {
            log.warn("inference", "CUDA final norm weight unsupported", .{
                .has_ln_final = @as(u8, @intFromBool(loaded.ln_final != null)),
                .dtype = if (loaded.ln_final) |ln_final| @tagName(ln_final.dtype) else "none",
            });
            return error.UnsupportedModel;
        }
        var projection_from_lm_head = false;
        var projection_weight_opt: ?LinearWeight = null;
        errdefer if (projection_weight_opt) |*w| w.deinit(device);

        if (skip_projection) {
            // Intermediate local stage: never computes logits.
            // Skip uploading projection weight to avoid init-time peak memory.
            projection_weight_opt = .{ .dense_f32 = missing_device_tensor };
        } else {
            if (loaded.lm_head) |lm_head| {
                projection_weight_opt = uploadLinearWeight(device, allocator, &lm_head, d_model) catch |err| switch (err) {
                    error.UnsupportedModel, error.InvalidArgument => null,
                    else => return err,
                };
                projection_from_lm_head = projection_weight_opt != null;
            }
            if (projection_weight_opt == null) {
                projection_weight_opt = uploadLinearWeight(device, allocator, &loaded.token_embeddings, d_model) catch |err| switch (err) {
                    error.UnsupportedModel, error.InvalidArgument => null,
                    else => return err,
                };
            }

            if (projection_weight_opt == null) {
                log.warn("inference", "CUDA projection weight unsupported", .{
                    .d_model = d_model,
                    .vocab_size = vocab_size,
                    .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
                    .embed_dtype = @tagName(loaded.token_embeddings.dtype),
                    .embed_ndim = loaded.token_embeddings.n_dims,
                });
                return error.UnsupportedModel;
            }
        }
        const using_model_projection = !skip_projection;
        const projection_weight = projection_weight_opt.?;
        if (@import("env_pkg").getenv("TALU_CUDA_PROJECTION_DEBUG") != null) {
            const projection_kind = switch (projection_weight) {
                .dense_f32 => "dense_f32",
                .dense_u16 => "dense_u16",
                .gaffine_u4 => "gaffine_u4",
                .gaffine_u8 => "gaffine_u8",
                .fp8 => "fp8",
                .mxfp8 => "mxfp8",
                .nvfp4 => "nvfp4",
            };
            log.warn("inference", "CUDA projection selection", .{
                .from_lm_head = @as(u8, @intFromBool(projection_from_lm_head)),
                .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
                .kind = projection_kind,
            });
        }
        const projected_vocab = if (skip_projection) vocab_size else projection_weight.cols();
        const projected_logits_host = try allocator.alloc(f32, projected_vocab);
        errdefer allocator.free(projected_logits_host);
        const projected_logits_batch_count = std.math.mul(usize, projected_vocab, max_batch_size) catch return error.InvalidArgument;
        const projected_logits_batch_host = try allocator.alloc(f32, projected_logits_batch_count);
        errdefer allocator.free(projected_logits_batch_host);
        const decode_key_cache_ptrs_host = try allocator.alloc(u64, max_batch_size);
        errdefer allocator.free(decode_key_cache_ptrs_host);
        const decode_value_cache_ptrs_host = try allocator.alloc(u64, max_batch_size);
        errdefer allocator.free(decode_value_cache_ptrs_host);
        const decode_attn_table_count = std.math.mul(usize, max_batch_size, max_attn_layers) catch return error.InvalidArgument;
        const decode_gd_table_count = std.math.mul(usize, max_batch_size, max_gd_layers) catch return error.InvalidArgument;
        const decode_attn_key_cache_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_key_cache_ptrs_table_host);
        const decode_attn_value_cache_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_value_cache_ptrs_table_host);
        const decode_attn_k_scale_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_k_scale_ptrs_table_host);
        const decode_attn_v_scale_ptrs_table_host = try allocator.alloc(u64, decode_attn_table_count);
        errdefer allocator.free(decode_attn_v_scale_ptrs_table_host);
        const decode_seq_lens_host = try allocator.alloc(u32, max_batch_size);
        errdefer allocator.free(decode_seq_lens_host);
        const decode_positions_host = try allocator.alloc(u32, max_batch_size);
        errdefer allocator.free(decode_positions_host);
        const decode_gd_conv_state_ptrs_table_host = try allocator.alloc(u64, decode_gd_table_count);
        errdefer allocator.free(decode_gd_conv_state_ptrs_table_host);
        const decode_gd_ssm_state_ptrs_table_host = try allocator.alloc(u64, decode_gd_table_count);
        errdefer allocator.free(decode_gd_ssm_state_ptrs_table_host);
        const decode_gd_conv_ring_heads_table_host = try allocator.alloc(u32, decode_gd_table_count);
        errdefer allocator.free(decode_gd_conv_ring_heads_table_host);
        const logits_bytes = std.math.mul(usize, projected_logits_batch_count, @sizeOf(f32)) catch return error.InvalidArgument;
        const topk_buffer_count = std.math.mul(usize, max_batch_size, 256) catch return error.InvalidArgument;
        const topk_values_bytes = std.math.mul(usize, topk_buffer_count, @sizeOf(f32)) catch return error.InvalidArgument;
        const topk_ids_bytes = std.math.mul(usize, topk_buffer_count, @sizeOf(u32)) catch return error.InvalidArgument;
        const decode_ptrs_bytes = std.math.mul(usize, max_batch_size, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_attn_table_ptrs_bytes = std.math.mul(usize, decode_attn_table_count, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_gd_table_ptrs_bytes = std.math.mul(usize, decode_gd_table_count, @sizeOf(u64)) catch return error.InvalidArgument;
        const decode_gd_table_idx_bytes = std.math.mul(usize, decode_gd_table_count, @sizeOf(u32)) catch return error.InvalidArgument;
        const decode_idx_bytes = std.math.mul(usize, max_batch_size, @sizeOf(u32)) catch return error.InvalidArgument;
        var embedding_lookup: ?EmbeddingLookup = null;
        errdefer if (embedding_lookup) |*lookup| lookup.deinit(device);
        const using_model_embeddings = if (skip_embedding) true else canUseModelEmbeddings(loaded, d_model);
        if (!using_model_embeddings) {
            log.warn("inference", "CUDA token embeddings unsupported", .{
                .d_model = d_model,
                .embed_dtype = @tagName(loaded.token_embeddings.dtype),
                .embed_ndim = loaded.token_embeddings.n_dims,
                .embed_shape_0 = loaded.token_embeddings.shape[0],
                .embed_shape_1 = loaded.token_embeddings.shape[1],
            });
            return error.UnsupportedModel;
        }
        if (!skip_embedding) {
            embedding_lookup = try tryUploadEmbeddingLookup(device, loaded, d_model);
        }

        var input_dev = try device.allocBuffer(d_model_bytes);
        errdefer input_dev.deinit(device);
        const prefill_tokens_bytes = std.math.mul(usize, max_seq_len, @sizeOf(u32)) catch return error.InvalidArgument;
        var prefill_tokens_dev = try device.allocBuffer(prefill_tokens_bytes);
        errdefer prefill_tokens_dev.deinit(device);
        var norm_weight_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_weight_dev.deinit(device);
        var norm_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_out_dev.deinit(device);
        var activation_u16_dev = try device.allocBuffer(activation_u16_bytes);
        errdefer activation_u16_dev.deinit(device);
        var attn_q_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_q_dev.deinit(device);
        var query_gate_proj_dev = try device.allocBuffer(d_attn_bytes);
        errdefer query_gate_proj_dev.deinit(device);
        var attn_k_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_k_dev.deinit(device);
        var attn_v_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_v_dev.deinit(device);
        var attn_context_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_context_dev.deinit(device);
        var decode_key_cache_ptrs_dev = try device.allocBuffer(decode_ptrs_bytes);
        errdefer decode_key_cache_ptrs_dev.deinit(device);
        var decode_value_cache_ptrs_dev = try device.allocBuffer(decode_ptrs_bytes);
        errdefer decode_value_cache_ptrs_dev.deinit(device);
        var decode_attn_key_cache_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_key_cache_ptrs_table_dev.deinit(device);
        var decode_attn_value_cache_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_value_cache_ptrs_table_dev.deinit(device);
        var decode_attn_k_scale_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_k_scale_ptrs_table_dev.deinit(device);
        var decode_attn_v_scale_ptrs_table_dev = try device.allocBuffer(decode_attn_table_ptrs_bytes);
        errdefer decode_attn_v_scale_ptrs_table_dev.deinit(device);
        var decode_seq_lens_dev = try device.allocBuffer(decode_idx_bytes);
        errdefer decode_seq_lens_dev.deinit(device);
        var decode_positions_dev = try device.allocBuffer(decode_idx_bytes);
        errdefer decode_positions_dev.deinit(device);
        var decode_gd_conv_state_ptrs_table_dev = try device.allocBuffer(decode_gd_table_ptrs_bytes);
        errdefer decode_gd_conv_state_ptrs_table_dev.deinit(device);
        var decode_gd_ssm_state_ptrs_table_dev = try device.allocBuffer(decode_gd_table_ptrs_bytes);
        errdefer decode_gd_ssm_state_ptrs_table_dev.deinit(device);
        var decode_gd_conv_ring_heads_table_dev = try device.allocBuffer(decode_gd_table_idx_bytes);
        errdefer decode_gd_conv_ring_heads_table_dev.deinit(device);
        var attn_scores_dev: ?compute.cuda.Buffer = null;
        errdefer if (attn_scores_dev) |*buf| buf.deinit(device);
        var attn_probs_dev: ?compute.cuda.Buffer = null;
        errdefer if (attn_probs_dev) |*buf| buf.deinit(device);
        if (need_attention_score_buffers) {
            attn_scores_dev = try device.allocBuffer(attn_rows_bytes);
            attn_probs_dev = try device.allocBuffer(attn_rows_bytes);
        }
        var attn_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer attn_out_dev.deinit(device);
        var ffn_gate_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_gate_dev.deinit(device);
        var ffn_up_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_up_dev.deinit(device);
        var ffn_mul_dev = try device.allocBuffer(d_ff_bytes);
        errdefer ffn_mul_dev.deinit(device);
        var ffn_down_dev = try device.allocBuffer(d_model_bytes);
        errdefer ffn_down_dev.deinit(device);
        var deepstack_add_dev = try device.allocBuffer(d_model_bytes);
        errdefer deepstack_add_dev.deinit(device);
        var shortconv_proj_dev = try device.allocBuffer(shortconv_proj_bytes);
        errdefer shortconv_proj_dev.deinit(device);
        var shortconv_conv_dev = try device.allocBuffer(shortconv_conv_bytes);
        errdefer shortconv_conv_dev.deinit(device);
        var gdelta_proj_dev = try device.allocBuffer(d_gdelta_proj_bytes);
        errdefer gdelta_proj_dev.deinit(device);
        var gdelta_ssm_dev = try device.allocBuffer(d_gdelta_proj_bytes);
        errdefer gdelta_ssm_dev.deinit(device);
        var logits_dev = try device.allocBuffer(logits_bytes);
        errdefer logits_dev.deinit(device);
        var topk_values_dev = try device.allocBuffer(topk_values_bytes);
        errdefer topk_values_dev.deinit(device);
        var topk_ids_dev = try device.allocBuffer(topk_ids_bytes);
        errdefer topk_ids_dev.deinit(device);
        // Batched attention scores/probs buffers: [max_batch_size * n_heads * max_seq_len].
        const batched_attn_max_seq_len: u32 = @intCast(max_seq_len);
        const batched_attn_elems = std.math.mul(usize, max_batch_size * n_heads, max_seq_len) catch return error.InvalidArgument;
        const batched_attn_bytes = std.math.mul(usize, batched_attn_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        var batched_attn_scores_dev = try device.allocBuffer(batched_attn_bytes);
        errdefer batched_attn_scores_dev.deinit(device);
        var batched_attn_probs_dev = try device.allocBuffer(batched_attn_bytes);
        errdefer batched_attn_probs_dev.deinit(device);
        const max_dequant_dim = @max(@max(max_dff, max_attn), @max(max_kv, max_gdelta_proj));
        const dequant_f16_bytes = std.math.mul(usize, std.math.mul(usize, d_model, max_dequant_dim) catch return error.InvalidArgument, @sizeOf(u16)) catch return error.InvalidArgument;
        var dequant_f16_dev = try device.allocBuffer(dequant_f16_bytes);
        errdefer dequant_f16_dev.deinit(device);

        try norm_weight_dev.upload(device, std.mem.sliceAsBytes(norm_weight_host));

        return .{
            .projected_vocab = projected_vocab,
            .max_dff = max_dff,
            .max_attn = max_attn,
            .max_kv = max_kv,
            .max_gdelta_proj = max_gdelta_proj,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .shortconv_dim = shortconv_dim,
            .row_capacity = 1,
            .using_model_norm = using_model_norm,
            .using_model_projection = using_model_projection,
            .projection_from_lm_head = projection_from_lm_head,
            .using_model_embeddings = using_model_embeddings,
            .embedding_lookup = embedding_lookup,
            .hidden_host = hidden_host,
            .projected_logits_host = projected_logits_host,
            .projected_logits_batch_host = projected_logits_batch_host,
            .prefill_tokens_dev = prefill_tokens_dev,
            .input_dev = input_dev,
            .norm_weight_dev = norm_weight_dev,
            .norm_out_dev = norm_out_dev,
            .activation_u16_dev = activation_u16_dev,
            .attn_q_dev = attn_q_dev,
            .query_gate_proj_dev = query_gate_proj_dev,
            .attn_k_dev = attn_k_dev,
            .attn_v_dev = attn_v_dev,
            .attn_context_dev = attn_context_dev,
            .decode_key_cache_ptrs_host = decode_key_cache_ptrs_host,
            .decode_value_cache_ptrs_host = decode_value_cache_ptrs_host,
            .decode_attn_key_cache_ptrs_table_host = decode_attn_key_cache_ptrs_table_host,
            .decode_attn_value_cache_ptrs_table_host = decode_attn_value_cache_ptrs_table_host,
            .decode_attn_k_scale_ptrs_table_host = decode_attn_k_scale_ptrs_table_host,
            .decode_attn_v_scale_ptrs_table_host = decode_attn_v_scale_ptrs_table_host,
            .decode_seq_lens_host = decode_seq_lens_host,
            .decode_positions_host = decode_positions_host,
            .decode_gd_conv_state_ptrs_table_host = decode_gd_conv_state_ptrs_table_host,
            .decode_gd_ssm_state_ptrs_table_host = decode_gd_ssm_state_ptrs_table_host,
            .decode_gd_conv_ring_heads_table_host = decode_gd_conv_ring_heads_table_host,
            .decode_key_cache_ptrs_dev = decode_key_cache_ptrs_dev,
            .decode_value_cache_ptrs_dev = decode_value_cache_ptrs_dev,
            .decode_attn_key_cache_ptrs_table_dev = decode_attn_key_cache_ptrs_table_dev,
            .decode_attn_value_cache_ptrs_table_dev = decode_attn_value_cache_ptrs_table_dev,
            .decode_attn_k_scale_ptrs_table_dev = decode_attn_k_scale_ptrs_table_dev,
            .decode_attn_v_scale_ptrs_table_dev = decode_attn_v_scale_ptrs_table_dev,
            .decode_seq_lens_dev = decode_seq_lens_dev,
            .decode_positions_dev = decode_positions_dev,
            .decode_gd_conv_state_ptrs_table_dev = decode_gd_conv_state_ptrs_table_dev,
            .decode_gd_ssm_state_ptrs_table_dev = decode_gd_ssm_state_ptrs_table_dev,
            .decode_gd_conv_ring_heads_table_dev = decode_gd_conv_ring_heads_table_dev,
            .attn_scores_dev = attn_scores_dev,
            .attn_probs_dev = attn_probs_dev,
            .attn_out_dev = attn_out_dev,
            .ffn_gate_dev = ffn_gate_dev,
            .ffn_up_dev = ffn_up_dev,
            .ffn_mul_dev = ffn_mul_dev,
            .ffn_down_dev = ffn_down_dev,
            .deepstack_add_dev = deepstack_add_dev,
            .shortconv_proj_dev = shortconv_proj_dev,
            .shortconv_conv_dev = shortconv_conv_dev,
            .gdelta_proj_dev = gdelta_proj_dev,
            .gdelta_ssm_dev = gdelta_ssm_dev,
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
            .topk_values_dev = topk_values_dev,
            .topk_ids_dev = topk_ids_dev,
            .batched_attn_scores_dev = batched_attn_scores_dev,
            .batched_attn_probs_dev = batched_attn_probs_dev,
            .batched_attn_max_seq_len = batched_attn_max_seq_len,
            .dequant_f16_dev = dequant_f16_dev,
        };
    }

    pub fn deinit(self: *RuntimeBuffers, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.dequant_f16_dev.deinit(device);
        self.batched_attn_probs_dev.deinit(device);
        self.batched_attn_scores_dev.deinit(device);
        self.topk_ids_dev.deinit(device);
        self.topk_values_dev.deinit(device);
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
        if (self.embedding_lookup) |*lookup| lookup.deinit(device);
        self.shortconv_conv_dev.deinit(device);
        self.shortconv_proj_dev.deinit(device);
        self.gdelta_proj_dev.deinit(device);
        self.gdelta_ssm_dev.deinit(device);
        self.ffn_down_dev.deinit(device);
        self.ffn_mul_dev.deinit(device);
        self.ffn_up_dev.deinit(device);
        self.ffn_gate_dev.deinit(device);
        self.attn_out_dev.deinit(device);
        if (self.attn_probs_dev) |*buf| buf.deinit(device);
        if (self.attn_scores_dev) |*buf| buf.deinit(device);
        self.attn_context_dev.deinit(device);
        self.decode_positions_dev.deinit(device);
        self.decode_seq_lens_dev.deinit(device);
        self.decode_gd_conv_ring_heads_table_dev.deinit(device);
        self.decode_gd_ssm_state_ptrs_table_dev.deinit(device);
        self.decode_gd_conv_state_ptrs_table_dev.deinit(device);
        self.decode_attn_v_scale_ptrs_table_dev.deinit(device);
        self.decode_attn_k_scale_ptrs_table_dev.deinit(device);
        self.decode_attn_value_cache_ptrs_table_dev.deinit(device);
        self.decode_attn_key_cache_ptrs_table_dev.deinit(device);
        self.decode_value_cache_ptrs_dev.deinit(device);
        self.decode_key_cache_ptrs_dev.deinit(device);
        self.attn_v_dev.deinit(device);
        self.attn_k_dev.deinit(device);
        self.query_gate_proj_dev.deinit(device);
        self.attn_q_dev.deinit(device);
        self.norm_out_dev.deinit(device);
        self.norm_weight_dev.deinit(device);
        self.activation_u16_dev.deinit(device);
        self.input_dev.deinit(device);
        self.deepstack_add_dev.deinit(device);
        self.prefill_tokens_dev.deinit(device);
        allocator.free(self.projected_logits_batch_host);
        allocator.free(self.projected_logits_host);
        allocator.free(self.hidden_host);
        allocator.free(self.decode_gd_conv_ring_heads_table_host);
        allocator.free(self.decode_gd_ssm_state_ptrs_table_host);
        allocator.free(self.decode_gd_conv_state_ptrs_table_host);
        allocator.free(self.decode_positions_host);
        allocator.free(self.decode_seq_lens_host);
        allocator.free(self.decode_attn_v_scale_ptrs_table_host);
        allocator.free(self.decode_attn_k_scale_ptrs_table_host);
        allocator.free(self.decode_attn_value_cache_ptrs_table_host);
        allocator.free(self.decode_attn_key_cache_ptrs_table_host);
        allocator.free(self.decode_value_cache_ptrs_host);
        allocator.free(self.decode_key_cache_ptrs_host);
    }

    pub fn deviceByteSize(self: *const RuntimeBuffers) usize {
        return self.input_dev.size +
            self.prefill_tokens_dev.size +
            self.norm_weight_dev.size +
            self.norm_out_dev.size +
            self.activation_u16_dev.size +
            self.attn_q_dev.size +
            self.query_gate_proj_dev.size +
            self.attn_k_dev.size +
            self.attn_v_dev.size +
            self.attn_context_dev.size +
            self.decode_key_cache_ptrs_dev.size +
            self.decode_value_cache_ptrs_dev.size +
            self.decode_attn_key_cache_ptrs_table_dev.size +
            self.decode_attn_value_cache_ptrs_table_dev.size +
            self.decode_attn_k_scale_ptrs_table_dev.size +
            self.decode_attn_v_scale_ptrs_table_dev.size +
            self.decode_seq_lens_dev.size +
            self.decode_positions_dev.size +
            self.decode_gd_conv_state_ptrs_table_dev.size +
            self.decode_gd_ssm_state_ptrs_table_dev.size +
            self.decode_gd_conv_ring_heads_table_dev.size +
            (if (self.attn_scores_dev) |buf| buf.size else 0) +
            (if (self.attn_probs_dev) |buf| buf.size else 0) +
            self.attn_out_dev.size +
            self.ffn_gate_dev.size +
            self.ffn_up_dev.size +
            self.ffn_mul_dev.size +
            self.ffn_down_dev.size +
            self.deepstack_add_dev.size +
            self.shortconv_proj_dev.size +
            self.shortconv_conv_dev.size +
            self.gdelta_proj_dev.size +
            self.gdelta_ssm_dev.size +
            self.topk_values_dev.size +
            self.topk_ids_dev.size +
            self.batched_attn_scores_dev.size +
            self.batched_attn_probs_dev.size +
            self.dequant_f16_dev.size +
            self.logits_dev.size +
            (if (self.embedding_lookup) |lookup| lookup.byteSize() else 0) +
            self.projection_weight.byteSize();
    }

    pub fn requireAttentionScoresDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_scores_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    pub fn requireAttentionProbsDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_probs_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    pub fn ensureRowCapacity(
        self: *RuntimeBuffers,
        device: *compute.cuda.Device,
        required_rows: usize,
        fixed_alloc_mode: bool,
    ) !void {
        if (required_rows == 0) return error.InvalidArgument;
        if (required_rows <= self.row_capacity) return;
        if (required_rows > self.max_seq_len) return error.InvalidArgument;
        if (fixed_alloc_mode) return error.OutOfMemory;

        var new_capacity = self.row_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_rows) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_rows) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, self.hidden_host.len, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, self.max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, self.max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, self.max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_gdelta_proj_bytes = std.math.mul(usize, self.max_gdelta_proj, @sizeOf(f32)) catch return error.InvalidArgument;
        const max_linear_in_dim = @max(@max(self.hidden_host.len, self.max_dff), @max(self.max_attn, self.max_gdelta_proj));
        const activation_u16_bytes = std.math.mul(usize, max_linear_in_dim, @sizeOf(u16)) catch return error.InvalidArgument;
        const shortconv_proj_bytes = std.math.mul(usize, self.shortconv_dim * 3, @sizeOf(f32)) catch return error.InvalidArgument;
        const shortconv_conv_bytes = std.math.mul(usize, self.shortconv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        try resizeScratchBuffer(device, &self.input_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.norm_out_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.activation_u16_dev, std.math.mul(usize, activation_u16_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_q_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.query_gate_proj_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_k_dev, std.math.mul(usize, d_kv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_v_dev, std.math.mul(usize, d_kv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_context_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_out_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_gate_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_up_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_mul_dev, std.math.mul(usize, d_ff_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.ffn_down_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.deepstack_add_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.shortconv_proj_dev, std.math.mul(usize, shortconv_proj_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.shortconv_conv_dev, std.math.mul(usize, shortconv_conv_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.gdelta_proj_dev, std.math.mul(usize, d_gdelta_proj_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.gdelta_ssm_dev, std.math.mul(usize, d_gdelta_proj_bytes, new_capacity) catch return error.InvalidArgument);

        self.row_capacity = new_capacity;
    }
};
