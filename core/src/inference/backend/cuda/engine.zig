//! CUDA backend engine (Phase 1 stub).
//!
//! This implements the backend contract while returning explicit typed errors
//! for unimplemented execution methods.

const std = @import("std");
const build_options = @import("build_options");
const models = @import("../../../models/root.zig");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const runtime_contract = @import("../../runtime_contract/root.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const load_transforms = @import("../../../models/load/transforms.zig");
const vision_types = @import("../../vision_types.zig");
const smoke_checks = @import("smoke_checks.zig");
const attention_policy = @import("attention_policy.zig");
const attention_mod = @import("attention.zig");
const decode_mod = @import("decode.zig");
const prefill_mod = @import("prefill.zig");
const vision_runtime_mod = @import("vision/root.zig");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const prototype_eps: f32 = 1e-5;
const initial_kv_cache_tokens: usize = 256;
const kv_cache_dtype_fp16: bool = true;
const enable_fused_attention_f16_kv: bool = true;
const max_fused_attention_f16_kv_seq_len: u32 = 384;
const enable_device_embedding_lookup: bool = true;
const max_supported_fused_f16_kv_head_dim = 512;
const attention_policy_config = attention_policy.Config{
    .kv_cache_dtype_fp16 = kv_cache_dtype_fp16,
    .enable_fused_attention_f16_kv = enable_fused_attention_f16_kv,
    .max_fused_attention_f16_kv_seq_len = max_fused_attention_f16_kv_seq_len,
    .max_supported_fused_f16_kv_head_dim = max_supported_fused_f16_kv_head_dim,
};
const run_startup_selftests = build_options.cuda_startup_selftests;
const gaffine_scales_dtype_f16 = compute.cuda.gaffine_u4_matvec.scales_dtype_f16;
const gaffine_scales_dtype_bf16 = compute.cuda.gaffine_u4_matvec.scales_dtype_bf16;
const DenseU16Dtype = enum(u8) {
    f16,
    bf16,
};

const EmbeddingLookupKind = enum(u8) {
    f32,
    f16,
    bf16,
    gaffine_u4,
};

const KernelSlot = enum {
    vector_add,
    vector_add_scaled,
    mul,
    copy,
    copy_u16,
    cast_f32_to_f16,
    embedding_lookup_f32,
    embedding_lookup_u16,
    embedding_lookup_gaffine_u4,
    kv_write_f16,
    rmsnorm,
    rope,
    rope_store_f16,
    attn_scores_heads_f32,
    attn_scores_heads_f16_kv,
    attn_fused_heads_f16_kv,
    softmax_rows,
    attn_weighted_sum_heads_f32,
    attn_weighted_sum_heads_f16_kv,
    silu,
    silu_mul,
    gelu_mul,
    shortconv_step,
    argmax,
    matmul_f16,
    matmul_bf16,
    matvec_f16,
    matvec_bf16,
    matvec_gate_up_f16,
    matvec_gate_up_bf16,
    matvec_qkv_f16,
    matvec_qkv_bf16,
    gaffine_u4_matvec,
    gaffine_u4_matvec_gate_up,
    gaffine_u4_matvec_qkv,
};

const RequiredKernel = struct {
    slot: KernelSlot,
    op_name: []const u8,
    embedded_symbol: [:0]const u8,
};

const ProjectionPath = enum {
    fused,
    unfused,
};

const AttentionPath = enum {
    fused_heads_f16_kv,
    heads_f16_kv,
    heads_f32_kv,
};

const AttentionKernelSet = struct {
    attn_scores_heads_f32_function: ?compute.cuda.Function,
    attn_weighted_sum_heads_f32_function: ?compute.cuda.Function,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function,
    softmax_rows_function: ?compute.cuda.Function,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function,
};

const required_kernels = [_]RequiredKernel{
    .{ .slot = .vector_add, .op_name = compute.cuda.vector_add.op_name, .embedded_symbol = compute.cuda.vector_add.embedded_symbol },
    .{ .slot = .vector_add_scaled, .op_name = compute.cuda.vector_add_scaled.op_name, .embedded_symbol = compute.cuda.vector_add_scaled.embedded_symbol },
    .{ .slot = .mul, .op_name = compute.cuda.mul.op_name, .embedded_symbol = compute.cuda.mul.embedded_symbol },
    .{ .slot = .copy, .op_name = compute.cuda.copy.op_name, .embedded_symbol = compute.cuda.copy.embedded_symbol },
    .{ .slot = .copy_u16, .op_name = compute.cuda.copy_u16.op_name, .embedded_symbol = compute.cuda.copy_u16.embedded_symbol },
    .{ .slot = .cast_f32_to_f16, .op_name = compute.cuda.cast_f32_to_f16.op_name, .embedded_symbol = compute.cuda.cast_f32_to_f16.embedded_symbol },
    .{ .slot = .embedding_lookup_f32, .op_name = compute.cuda.embedding_lookup_f32.op_name, .embedded_symbol = compute.cuda.embedding_lookup_f32.embedded_symbol },
    .{ .slot = .embedding_lookup_u16, .op_name = compute.cuda.embedding_lookup_u16.op_name, .embedded_symbol = compute.cuda.embedding_lookup_u16.embedded_symbol },
    .{ .slot = .embedding_lookup_gaffine_u4, .op_name = compute.cuda.embedding_lookup_gaffine_u4.op_name, .embedded_symbol = compute.cuda.embedding_lookup_gaffine_u4.embedded_symbol },
    .{ .slot = .kv_write_f16, .op_name = compute.cuda.kv_write_f16.op_name, .embedded_symbol = compute.cuda.kv_write_f16.embedded_symbol },
    .{ .slot = .rmsnorm, .op_name = compute.cuda.rmsnorm.op_name, .embedded_symbol = compute.cuda.rmsnorm.embedded_symbol },
    .{ .slot = .rope, .op_name = compute.cuda.rope.op_name, .embedded_symbol = compute.cuda.rope.embedded_symbol },
    .{ .slot = .rope_store_f16, .op_name = compute.cuda.rope_store_f16.op_name, .embedded_symbol = compute.cuda.rope_store_f16.embedded_symbol },
    .{ .slot = .attn_scores_heads_f32, .op_name = compute.cuda.attn_scores_heads_f32.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f32.embedded_symbol },
    .{ .slot = .attn_scores_heads_f16_kv, .op_name = compute.cuda.attn_scores_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f16_kv.embedded_symbol },
    .{ .slot = .attn_fused_heads_f16_kv, .op_name = compute.cuda.attn_fused_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_heads_f16_kv.embedded_symbol },
    .{ .slot = .softmax_rows, .op_name = compute.cuda.softmax_rows.op_name, .embedded_symbol = compute.cuda.softmax_rows.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f32, .op_name = compute.cuda.attn_weighted_sum_heads_f32.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f32.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f16_kv, .op_name = compute.cuda.attn_weighted_sum_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f16_kv.embedded_symbol },
    .{ .slot = .silu, .op_name = compute.cuda.silu.op_name, .embedded_symbol = compute.cuda.silu.embedded_symbol },
    .{ .slot = .silu_mul, .op_name = compute.cuda.silu_mul.op_name, .embedded_symbol = compute.cuda.silu_mul.embedded_symbol },
    .{ .slot = .gelu_mul, .op_name = compute.cuda.gelu_mul.op_name, .embedded_symbol = compute.cuda.gelu_mul.embedded_symbol },
    .{ .slot = .shortconv_step, .op_name = compute.cuda.shortconv_step.op_name, .embedded_symbol = compute.cuda.shortconv_step.embedded_symbol },
    .{ .slot = .argmax, .op_name = compute.cuda.argmax.op_name, .embedded_symbol = compute.cuda.argmax.embedded_symbol },
    .{ .slot = .matmul_f16, .op_name = compute.cuda.matmul_u16.op_name_f16, .embedded_symbol = compute.cuda.matmul_u16.embedded_symbol_f16 },
    .{ .slot = .matmul_bf16, .op_name = compute.cuda.matmul_u16.op_name_bf16, .embedded_symbol = compute.cuda.matmul_u16.embedded_symbol_bf16 },
    .{ .slot = .matvec_f16, .op_name = compute.cuda.matvec_u16.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16.embedded_symbol_f16 },
    .{ .slot = .matvec_bf16, .op_name = compute.cuda.matvec_u16.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16.embedded_symbol_bf16 },
    .{ .slot = .matvec_gate_up_f16, .op_name = compute.cuda.matvec_u16_gate_up.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16_gate_up.embedded_symbol_f16 },
    .{ .slot = .matvec_gate_up_bf16, .op_name = compute.cuda.matvec_u16_gate_up.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16_gate_up.embedded_symbol_bf16 },
    .{ .slot = .matvec_qkv_f16, .op_name = compute.cuda.matvec_u16_qkv.op_name_f16, .embedded_symbol = compute.cuda.matvec_u16_qkv.embedded_symbol_f16 },
    .{ .slot = .matvec_qkv_bf16, .op_name = compute.cuda.matvec_u16_qkv.op_name_bf16, .embedded_symbol = compute.cuda.matvec_u16_qkv.embedded_symbol_bf16 },
    .{ .slot = .gaffine_u4_matvec, .op_name = compute.cuda.gaffine_u4_matvec.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_gate_up, .op_name = compute.cuda.gaffine_u4_matvec_gate_up.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec_gate_up.embedded_symbol },
    .{ .slot = .gaffine_u4_matvec_qkv, .op_name = compute.cuda.gaffine_u4_matvec_qkv.op_name, .embedded_symbol = compute.cuda.gaffine_u4_matvec_qkv.embedded_symbol },
};

const DeviceTensor = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,

    fn deinit(self: *DeviceTensor, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    fn byteSize(self: *const DeviceTensor) usize {
        return self.buffer.size;
    }
};

const EmbeddingLookup = struct {
    kind: EmbeddingLookupKind,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    group_size: u32 = 0,
    scales_dtype_tag: u32 = 0,
    scales: ?compute.cuda.Buffer = null,
    biases: ?compute.cuda.Buffer = null,
    multiplier: f32,
    buffer: compute.cuda.Buffer,

    fn deinit(self: *EmbeddingLookup, device: *compute.cuda.Device) void {
        if (self.biases) |*buf| buf.deinit(device);
        if (self.scales) |*buf| buf.deinit(device);
        self.buffer.deinit(device);
    }

    fn byteSize(self: *const EmbeddingLookup) usize {
        return self.buffer.size +
            (if (self.scales) |buf| buf.size else 0) +
            (if (self.biases) |buf| buf.size else 0);
    }
};

const GaffineU4LinearWeight = struct {
    rows: usize,
    cols: usize,
    packed_data: compute.cuda.Buffer,
    scales: compute.cuda.Buffer,
    biases: compute.cuda.Buffer,
    group_size: u32,
    scales_dtype_tag: u32,

    fn deinit(self: *GaffineU4LinearWeight, device: *compute.cuda.Device) void {
        self.biases.deinit(device);
        self.scales.deinit(device);
        self.packed_data.deinit(device);
    }

    fn byteSize(self: *const GaffineU4LinearWeight) usize {
        return self.packed_data.size + self.scales.size + self.biases.size;
    }
};

const U16LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: compute.cuda.Buffer,
    dtype: DenseU16Dtype,

    fn deinit(self: *U16LinearWeight, device: *compute.cuda.Device) void {
        self.buffer.deinit(device);
    }

    fn byteSize(self: *const U16LinearWeight) usize {
        return self.buffer.size;
    }
};

const LinearWeight = union(enum) {
    dense_f32: DeviceTensor,
    dense_u16: U16LinearWeight,
    gaffine_u4: GaffineU4LinearWeight,

    fn deinit(self: *LinearWeight, device: *compute.cuda.Device) void {
        switch (self.*) {
            .dense_f32 => |*w| w.deinit(device),
            .dense_u16 => |*w| w.deinit(device),
            .gaffine_u4 => |*w| w.deinit(device),
        }
    }

    fn rows(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.rows,
            .dense_u16 => |w| w.rows,
            .gaffine_u4 => |w| w.rows,
        };
    }

    fn cols(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.cols,
            .dense_u16 => |w| w.cols,
            .gaffine_u4 => |w| w.cols,
        };
    }

    fn byteSize(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.buffer.size,
            .dense_u16 => |w| w.byteSize(),
            .gaffine_u4 => |w| w.byteSize(),
        };
    }
};

const PrototypeRuntime = struct {
    projected_vocab: usize,
    max_dff: usize,
    max_attn: usize,
    max_kv: usize,
    max_seq_len: usize,
    head_dim: usize,
    using_model_norm: bool,
    using_model_projection: bool,
    projection_from_lm_head: bool,
    using_model_embeddings: bool,
    embedding_lookup: ?EmbeddingLookup,
    hidden_host: []f32,
    projected_logits_host: []f32,
    input_dev: compute.cuda.Buffer,
    norm_weight_dev: compute.cuda.Buffer,
    norm_out_dev: compute.cuda.Buffer,
    attn_q_dev: compute.cuda.Buffer,
    attn_k_dev: compute.cuda.Buffer,
    attn_v_dev: compute.cuda.Buffer,
    attn_context_dev: compute.cuda.Buffer,
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
    projection_weight: LinearWeight,
    logits_dev: compute.cuda.Buffer,

    fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_dff: usize,
        max_attn: usize,
        max_kv: usize,
        max_shortconv_dim: usize,
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) !PrototypeRuntime {
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        if (d_model == 0 or vocab_size == 0) return error.InvalidArgument;
        if (max_dff == 0) return error.InvalidArgument;
        if (max_attn == 0) return error.InvalidArgument;
        if (max_kv == 0 or max_seq_len == 0 or head_dim == 0) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
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
        const using_model_projection = true;
        const projection_weight = projection_weight_opt.?;
        const projected_vocab = projection_weight.cols();
        const projected_logits_host = try allocator.alloc(f32, projected_vocab);
        errdefer allocator.free(projected_logits_host);
        const logits_bytes = std.math.mul(usize, projected_vocab, @sizeOf(f32)) catch return error.InvalidArgument;
        const using_model_embeddings = canUseModelEmbeddings(loaded, d_model);
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
        var embedding_lookup = try tryUploadEmbeddingLookup(device, loaded, d_model);
        errdefer if (embedding_lookup) |*lookup| lookup.deinit(device);

        var input_dev = try device.allocBuffer(d_model_bytes);
        errdefer input_dev.deinit(device);
        var norm_weight_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_weight_dev.deinit(device);
        var norm_out_dev = try device.allocBuffer(d_model_bytes);
        errdefer norm_out_dev.deinit(device);
        var attn_q_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_q_dev.deinit(device);
        var attn_k_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_k_dev.deinit(device);
        var attn_v_dev = try device.allocBuffer(d_kv_bytes);
        errdefer attn_v_dev.deinit(device);
        var attn_context_dev = try device.allocBuffer(d_attn_bytes);
        errdefer attn_context_dev.deinit(device);
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
        var logits_dev = try device.allocBuffer(logits_bytes);
        errdefer logits_dev.deinit(device);

        try norm_weight_dev.upload(device, std.mem.sliceAsBytes(norm_weight_host));

        return .{
            .projected_vocab = projected_vocab,
            .max_dff = max_dff,
            .max_attn = max_attn,
            .max_kv = max_kv,
            .max_seq_len = max_seq_len,
            .head_dim = head_dim,
            .using_model_norm = using_model_norm,
            .using_model_projection = using_model_projection,
            .projection_from_lm_head = projection_from_lm_head,
            .using_model_embeddings = using_model_embeddings,
            .embedding_lookup = embedding_lookup,
            .hidden_host = hidden_host,
            .projected_logits_host = projected_logits_host,
            .input_dev = input_dev,
            .norm_weight_dev = norm_weight_dev,
            .norm_out_dev = norm_out_dev,
            .attn_q_dev = attn_q_dev,
            .attn_k_dev = attn_k_dev,
            .attn_v_dev = attn_v_dev,
            .attn_context_dev = attn_context_dev,
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
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
        };
    }

    fn deinit(self: *PrototypeRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
        if (self.embedding_lookup) |*lookup| lookup.deinit(device);
        self.shortconv_conv_dev.deinit(device);
        self.shortconv_proj_dev.deinit(device);
        self.ffn_down_dev.deinit(device);
        self.ffn_mul_dev.deinit(device);
        self.ffn_up_dev.deinit(device);
        self.ffn_gate_dev.deinit(device);
        self.attn_out_dev.deinit(device);
        if (self.attn_probs_dev) |*buf| buf.deinit(device);
        if (self.attn_scores_dev) |*buf| buf.deinit(device);
        self.attn_context_dev.deinit(device);
        self.attn_v_dev.deinit(device);
        self.attn_k_dev.deinit(device);
        self.attn_q_dev.deinit(device);
        self.norm_out_dev.deinit(device);
        self.norm_weight_dev.deinit(device);
        self.input_dev.deinit(device);
        self.deepstack_add_dev.deinit(device);
        allocator.free(self.projected_logits_host);
        allocator.free(self.hidden_host);
    }

    fn deviceByteSize(self: *const PrototypeRuntime) usize {
        return self.input_dev.size +
            self.norm_weight_dev.size +
            self.norm_out_dev.size +
            self.attn_q_dev.size +
            self.attn_k_dev.size +
            self.attn_v_dev.size +
            self.attn_context_dev.size +
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
            self.logits_dev.size +
            (if (self.embedding_lookup) |lookup| lookup.byteSize() else 0) +
            self.projection_weight.byteSize();
    }

    fn requireAttentionScoresDev(self: *PrototypeRuntime) !*compute.cuda.Buffer {
        if (self.attn_scores_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    fn requireAttentionProbsDev(self: *PrototypeRuntime) !*compute.cuda.Buffer {
        if (self.attn_probs_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }
};

const AttentionMlpBlockRuntime = struct {
    q_dim: usize,
    kv_dim: usize,
    d_ff: usize,
    sliding_window: usize,
    is_causal: bool,
    ln1_weight: DeviceTensor,
    ln2_weight: DeviceTensor,
    pre_ffn_norm_weight: ?DeviceTensor = null,
    post_ffn_norm_weight: ?DeviceTensor = null,
    q_norm_weight: ?DeviceTensor = null,
    k_norm_weight: ?DeviceTensor = null,
    q_proj: LinearWeight,
    k_proj: LinearWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    w1: LinearWeight,
    w2: LinearWeight,
    w3: LinearWeight,
    k_cache: compute.cuda.Buffer,
    v_cache: compute.cuda.Buffer,
    kv_capacity: usize,

    fn deinit(self: *AttentionMlpBlockRuntime, device: *compute.cuda.Device) void {
        self.v_cache.deinit(device);
        self.k_cache.deinit(device);
        if (self.post_ffn_norm_weight) |*w| w.deinit(device);
        if (self.pre_ffn_norm_weight) |*w| w.deinit(device);
        if (self.k_norm_weight) |*w| w.deinit(device);
        if (self.q_norm_weight) |*w| w.deinit(device);
        self.w3.deinit(device);
        self.w2.deinit(device);
        self.w1.deinit(device);
        self.o_proj.deinit(device);
        self.v_proj.deinit(device);
        self.k_proj.deinit(device);
        self.q_proj.deinit(device);
        self.ln2_weight.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

const ShortConvBlockRuntime = struct {
    conv_dim: usize,
    d_conv: usize,
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    in_proj: LinearWeight,
    out_proj: LinearWeight,
    conv_weight_time_major: DeviceTensor,
    conv_bias: ?DeviceTensor = null,
    conv_state: compute.cuda.Buffer,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,

    fn deinit(self: *ShortConvBlockRuntime, device: *compute.cuda.Device) void {
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.conv_state.deinit(device);
        if (self.conv_bias) |*w| w.deinit(device);
        self.conv_weight_time_major.deinit(device);
        self.out_proj.deinit(device);
        self.in_proj.deinit(device);
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

const BlockRuntimeLayer = struct {
    program: ?[]const layer_ops.LayerOp = null,
    attention_mlp: ?AttentionMlpBlockRuntime = null,
    shortconv: ?ShortConvBlockRuntime = null,

    fn deinit(self: *BlockRuntimeLayer, device: *compute.cuda.Device) void {
        if (self.attention_mlp) |*block| block.deinit(device);
        if (self.shortconv) |*block| block.deinit(device);
        self.* = .{};
    }
};

fn finalProgramOutputBuffer(program: []const layer_ops.LayerOp) layer_ops.BufferId {
    return layer_ops.finalOutputBuffer(program);
}

fn validateLayerProgramForCuda(program: []const layer_ops.LayerOp, layer_idx: usize, kind: op_types.BlockKind) !void {
    if (runtime_contract.firstLayerProgramCompatibilityIssue(
        program,
        kind,
        CudaBackend.layer_program_adapter_table,
    )) |issue| {
        switch (issue) {
            .unsupported_opcode => |unsupported| {
                log.warn("inference", "CUDA LayerOp program contains unsupported opcode", .{
                    .layer = layer_idx,
                    .op_index = unsupported.op_index,
                    .kind = @intFromEnum(kind),
                    .op = @tagName(program[unsupported.op_index]),
                    .opcode = @intFromEnum(unsupported.opcode),
                });
            },
            .state_mismatch => |mismatch| {
                log.warn("inference", "CUDA LayerOp program state binding mismatches block kind", .{
                    .layer = layer_idx,
                    .op_index = mismatch.op_index,
                    .kind = @intFromEnum(kind),
                    .op = @tagName(program[mismatch.op_index]),
                    .opcode = @intFromEnum(mismatch.opcode),
                    .state_id = mismatch.state_id,
                });
            },
            .buffer_violation => |violation| switch (violation) {
                .op_index => |bad_op_idx| {
                    log.warn("inference", "CUDA LayerOp program uses unsupported buffer id", .{
                        .layer = layer_idx,
                        .op_index = bad_op_idx,
                        .kind = @intFromEnum(kind),
                        .op = @tagName(program[bad_op_idx]),
                    });
                },
                .final_output => |out| {
                    log.warn("inference", "CUDA LayerOp program final buffer is unsupported", .{
                        .layer = layer_idx,
                        .kind = @intFromEnum(kind),
                        .out = @intFromEnum(out),
                    });
                },
            },
        }
        return error.UnsupportedModel;
    }
}

const BlockRuntime = struct {
    blocks: []BlockRuntimeLayer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    attention_block_count: usize,
    shortconv_block_count: usize,
    q_norm_blocks: usize,
    k_norm_blocks: usize,
    linear_weight_bytes: usize,
    norm_weight_bytes: usize,
    kv_cache_bytes: usize,
    shortconv_state_bytes: usize,
    max_shortconv_dim: usize,

    fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
    ) !BlockRuntime {
        const d_model: usize = @intCast(loaded.config.d_model);
        const n_heads: usize = @intCast(loaded.config.n_heads);
        const n_kv_heads: usize = @intCast(loaded.config.n_kv_groups);
        const head_dim: usize = @intCast(loaded.config.head_dim);
        const max_seq_len: usize = @intCast(loaded.config.max_seq_len);
        if (n_heads == 0 or n_kv_heads == 0 or head_dim == 0 or max_seq_len == 0) return error.InvalidArgument;
        if (n_heads % n_kv_heads != 0) return error.UnsupportedModel;
        const arena_allocator = @constCast(&loaded.arena).allocator();
        const static_entry = if (loaded.runtime.architecture_id) |arch_id|
            models.registry.detectByArchitectureId(arch_id)
        else
            null;
        const layer_count = loaded.blocks.len;
        var attention_block_count: usize = 0;
        var shortconv_block_count: usize = 0;
        var q_norm_blocks: usize = 0;
        var k_norm_blocks: usize = 0;
        var linear_weight_bytes: usize = 0;
        var norm_weight_bytes: usize = 0;
        var kv_cache_bytes: usize = 0;
        var shortconv_state_bytes: usize = 0;
        var max_shortconv_dim: usize = 0;

        var blocks = try allocator.alloc(BlockRuntimeLayer, layer_count);
        errdefer allocator.free(blocks);
        for (blocks) |*layer| layer.* = .{};

        var initialized: usize = 0;
        errdefer {
            while (initialized > 0) {
                initialized -= 1;
                blocks[initialized].deinit(device);
            }
        }

        for (loaded.blocks, 0..) |*layer_weights, layer_idx| {
            const block_weights = try models.runtime_blocks.layerToBlockWeights(arena_allocator, layer_weights);
            switch (block_weights) {
                .attention_mlp => |attn| {
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA block runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    blocks[layer_idx].program = models.registry.blockProgramFor(entry, .attention_mlp) orelse {
                        log.warn("inference", "CUDA block runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.attention_mlp),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    try validateLayerProgramForCuda(
                        blocks[layer_idx].program.?,
                        layer_idx,
                        .attention_mlp,
                    );
                    if (attn.mla_config != null) {
                        log.warn("inference", "CUDA block runtime MLA not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }
                    if (attn.moe_weights != null) {
                        log.warn("inference", "CUDA block runtime MoE not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }
                    const w2 = attn.w2 orelse return error.MissingWeight;
                    const q_out = n_heads * head_dim;
                    const kv_out = n_kv_heads * head_dim;
                    if (attention_block_count == 0) {
                        if (attn.fused.qkv_proj != null or attn.fused.gate_up != null) {
                            log.info("inference", "CUDA block0 fused weight mode", .{
                                .qkv_fused = @as(u8, @intFromBool(attn.fused.qkv_proj != null)),
                                .gate_up_fused = @as(u8, @intFromBool(attn.fused.gate_up != null)),
                                .gate_up_layout = @tagName(attn.fused.gate_up_layout),
                                .qkv_dtype = if (attn.fused.qkv_proj) |qkv| @tagName(qkv.dtype) else "none",
                                .gate_up_dtype = if (attn.fused.gate_up) |gate_up| @tagName(gate_up.dtype) else "none",
                                .w2_dtype = @tagName(w2.dtype),
                                .q_out = q_out,
                                .kv_out = kv_out,
                            });
                        } else {
                            const q_proj = attn.q_proj orelse return error.MissingWeight;
                            const k_proj = attn.k_proj orelse return error.MissingWeight;
                            const v_proj = attn.v_proj orelse return error.MissingWeight;
                            const w1 = attn.w1 orelse return error.MissingWeight;
                            const w3 = attn.w3 orelse return error.MissingWeight;
                            log.info("inference", "CUDA block0 weight dtypes", .{
                                .q_proj = @tagName(q_proj.dtype),
                                .k_proj = @tagName(k_proj.dtype),
                                .v_proj = @tagName(v_proj.dtype),
                                .o_proj = @tagName(attn.o_proj.dtype),
                                .w1 = @tagName(w1.dtype),
                                .w2 = @tagName(w2.dtype),
                                .w3 = @tagName(w3.dtype),
                            });
                            log.info("inference", "CUDA block0 weight shapes", .{
                                .q0 = q_proj.shape[0],
                                .q1 = q_proj.shape[1],
                                .k0 = k_proj.shape[0],
                                .k1 = k_proj.shape[1],
                                .v0 = v_proj.shape[0],
                                .v1 = v_proj.shape[1],
                                .o0 = attn.o_proj.shape[0],
                                .o1 = attn.o_proj.shape[1],
                                .w10 = w1.shape[0],
                                .w11 = w1.shape[1],
                                .w20 = w2.shape[0],
                                .w21 = w2.shape[1],
                                .w30 = w3.shape[0],
                                .w31 = w3.shape[1],
                            });
                        }
                    }

                    var ln1_weight = try uploadTensor(device, allocator, attn.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    var ln2_weight = try uploadTensor(device, allocator, attn.ln2_weight);
                    errdefer ln2_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA block runtime ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (!(ln2_weight.rows == d_model and ln2_weight.cols == 1)) {
                        log.warn("inference", "CUDA block runtime ln2 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln2_weight.rows,
                            .cols = ln2_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var pre_ffn_norm_weight: ?DeviceTensor = null;
                    if (attn.pre_ffn_norm) |pre_ffn_norm| {
                        var pre_ffn = try uploadTensor(device, allocator, pre_ffn_norm);
                        errdefer pre_ffn.deinit(device);
                        if (!(pre_ffn.rows == d_model and pre_ffn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime pre_ffn_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = pre_ffn.rows,
                                .cols = pre_ffn.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        pre_ffn_norm_weight = pre_ffn;
                    }
                    errdefer if (pre_ffn_norm_weight) |*w| w.deinit(device);

                    var post_ffn_norm_weight: ?DeviceTensor = null;
                    if (attn.post_ffn_norm) |post_ffn_norm| {
                        var post_ffn = try uploadTensor(device, allocator, post_ffn_norm);
                        errdefer post_ffn.deinit(device);
                        if (!(post_ffn.rows == d_model and post_ffn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime post_ffn_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = post_ffn.rows,
                                .cols = post_ffn.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        post_ffn_norm_weight = post_ffn;
                    }
                    errdefer if (post_ffn_norm_weight) |*w| w.deinit(device);

                    var q_norm_weight: ?DeviceTensor = null;
                    if (attn.q_norm) |q_norm| {
                        var qn = try uploadTensor(device, allocator, q_norm);
                        errdefer qn.deinit(device);
                        if (!(qn.rows == head_dim and qn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime q_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = qn.rows,
                                .cols = qn.cols,
                                .head_dim = head_dim,
                            });
                            return error.UnsupportedModel;
                        }
                        q_norm_weight = qn;
                        q_norm_blocks += 1;
                    }
                    errdefer if (q_norm_weight) |*w| w.deinit(device);

                    var k_norm_weight: ?DeviceTensor = null;
                    if (attn.k_norm) |k_norm| {
                        var kn = try uploadTensor(device, allocator, k_norm);
                        errdefer kn.deinit(device);
                        if (!(kn.rows == head_dim and kn.cols == 1)) {
                            log.warn("inference", "CUDA block runtime k_norm shape unsupported", .{
                                .layer = layer_idx,
                                .rows = kn.rows,
                                .cols = kn.cols,
                                .head_dim = head_dim,
                            });
                            return error.UnsupportedModel;
                        }
                        k_norm_weight = kn;
                        k_norm_blocks += 1;
                    }
                    errdefer if (k_norm_weight) |*w| w.deinit(device);

                    var q_proj_dev: LinearWeight = undefined;
                    var k_proj_dev: LinearWeight = undefined;
                    var v_proj_dev: LinearWeight = undefined;
                    if (attn.fused.qkv_proj) |qkv_proj| {
                        const fused_qkv = try uploadFusedQkvWeights(
                            device,
                            allocator,
                            &qkv_proj,
                            d_model,
                            q_out,
                            kv_out,
                        );
                        q_proj_dev = fused_qkv.q;
                        k_proj_dev = fused_qkv.k;
                        v_proj_dev = fused_qkv.v;
                    } else {
                        const q_proj = attn.q_proj orelse return error.MissingWeight;
                        const k_proj = attn.k_proj orelse return error.MissingWeight;
                        const v_proj = attn.v_proj orelse return error.MissingWeight;
                        q_proj_dev = try uploadLinearWeight(device, allocator, q_proj, d_model);
                        k_proj_dev = try uploadLinearWeight(device, allocator, k_proj, d_model);
                        v_proj_dev = try uploadLinearWeight(device, allocator, v_proj, d_model);
                    }
                    errdefer q_proj_dev.deinit(device);
                    errdefer k_proj_dev.deinit(device);
                    errdefer v_proj_dev.deinit(device);

                    var o_proj_dev = try uploadLinearWeight(device, allocator, attn.o_proj, q_proj_dev.cols());
                    errdefer o_proj_dev.deinit(device);

                    var w1_dev: LinearWeight = undefined;
                    var w3_dev: LinearWeight = undefined;
                    if (attn.fused.gate_up) |gate_up| {
                        const fused_gate_up = try uploadFusedGateUpWeights(
                            device,
                            allocator,
                            &gate_up,
                            d_model,
                            attn.fused.gate_up_layout,
                        );
                        w1_dev = fused_gate_up.gate;
                        w3_dev = fused_gate_up.up;
                    } else {
                        const w1 = attn.w1 orelse return error.MissingWeight;
                        const w3 = attn.w3 orelse return error.MissingWeight;
                        w1_dev = try uploadLinearWeight(device, allocator, w1, d_model);
                        w3_dev = try uploadLinearWeight(device, allocator, w3, d_model);
                    }
                    errdefer w1_dev.deinit(device);
                    errdefer w3_dev.deinit(device);

                    if (w1_dev.cols() != w3_dev.cols()) {
                        log.warn("inference", "CUDA block runtime gate/up dim mismatch", .{
                            .layer = layer_idx,
                            .w1_cols = w1_dev.cols(),
                            .w3_cols = w3_dev.cols(),
                        });
                        return error.UnsupportedModel;
                    }
                    const d_ff = w1_dev.cols();
                    var w2_dev = try uploadLinearWeight(device, allocator, w2, d_ff);
                    errdefer w2_dev.deinit(device);
                    if (k_proj_dev.cols() != v_proj_dev.cols()) {
                        log.warn("inference", "CUDA block runtime k/v dim mismatch", .{
                            .layer = layer_idx,
                            .k_cols = k_proj_dev.cols(),
                            .v_cols = v_proj_dev.cols(),
                        });
                        return error.UnsupportedModel;
                    }
                    if (o_proj_dev.cols() != d_model) {
                        log.warn("inference", "CUDA block runtime o_proj out dim unsupported", .{
                            .layer = layer_idx,
                            .o_proj_cols = o_proj_dev.cols(),
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (w2_dev.cols() != d_model) {
                        log.warn("inference", "CUDA block runtime down_proj out dim unsupported", .{
                            .layer = layer_idx,
                            .w2_cols = w2_dev.cols(),
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    if (q_proj_dev.cols() != n_heads * head_dim) {
                        log.warn("inference", "CUDA block runtime q_proj dim unsupported", .{
                            .layer = layer_idx,
                            .q_cols = q_proj_dev.cols(),
                            .expected = n_heads * head_dim,
                        });
                        return error.UnsupportedModel;
                    }
                    if (k_proj_dev.cols() != n_kv_heads * head_dim) {
                        log.warn("inference", "CUDA block runtime kv dim unsupported", .{
                            .layer = layer_idx,
                            .kv_cols = k_proj_dev.cols(),
                            .expected = n_kv_heads * head_dim,
                        });
                        return error.UnsupportedModel;
                    }

                    const kv_capacity = @min(max_seq_len, initial_kv_cache_tokens);
                    if (kv_capacity == 0) return error.InvalidArgument;
                    const kv_cache_elems = std.math.mul(usize, kv_capacity, k_proj_dev.cols()) catch return error.InvalidArgument;
                    const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
                    const kv_cache_bytes_per_buffer = std.math.mul(usize, kv_cache_elems, kv_elem_bytes) catch return error.InvalidArgument;
                    var k_cache = try device.allocBuffer(kv_cache_bytes_per_buffer);
                    errdefer k_cache.deinit(device);
                    var v_cache = try device.allocBuffer(kv_cache_bytes_per_buffer);
                    errdefer v_cache.deinit(device);

                    const layer_norm_bytes = ln1_weight.byteSize() +
                        ln2_weight.byteSize() +
                        (if (pre_ffn_norm_weight) |w| w.byteSize() else 0) +
                        (if (post_ffn_norm_weight) |w| w.byteSize() else 0) +
                        (if (q_norm_weight) |w| w.byteSize() else 0) +
                        (if (k_norm_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    const layer_linear_bytes = q_proj_dev.byteSize() +
                        k_proj_dev.byteSize() +
                        v_proj_dev.byteSize() +
                        o_proj_dev.byteSize() +
                        w1_dev.byteSize() +
                        w2_dev.byteSize() +
                        w3_dev.byteSize();
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                    const layer_kv_bytes = std.math.mul(usize, kv_cache_bytes_per_buffer, 2) catch return error.InvalidArgument;
                    kv_cache_bytes = std.math.add(usize, kv_cache_bytes, layer_kv_bytes) catch return error.InvalidArgument;

                    blocks[layer_idx].attention_mlp = .{
                        .q_dim = q_proj_dev.cols(),
                        .kv_dim = k_proj_dev.cols(),
                        .d_ff = d_ff,
                        .sliding_window = attn.sliding_window,
                        .is_causal = attn.is_causal,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .pre_ffn_norm_weight = pre_ffn_norm_weight,
                        .post_ffn_norm_weight = post_ffn_norm_weight,
                        .q_norm_weight = q_norm_weight,
                        .k_norm_weight = k_norm_weight,
                        .q_proj = q_proj_dev,
                        .k_proj = k_proj_dev,
                        .v_proj = v_proj_dev,
                        .o_proj = o_proj_dev,
                        .w1 = w1_dev,
                        .w2 = w2_dev,
                        .w3 = w3_dev,
                        .k_cache = k_cache,
                        .v_cache = v_cache,
                        .kv_capacity = kv_capacity,
                    };
                    attention_block_count += 1;
                },
                .shortconv => |shortconv| {
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA shortconv runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    blocks[layer_idx].program = models.registry.blockProgramFor(entry, .shortconv) orelse {
                        log.warn("inference", "CUDA shortconv runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.shortconv),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    try validateLayerProgramForCuda(
                        blocks[layer_idx].program.?,
                        layer_idx,
                        .shortconv,
                    );
                    if (shortconv.fused_gate_up != null) {
                        log.warn("inference", "CUDA block runtime fused shortconv gate_up not supported yet", .{
                            .layer = layer_idx,
                        });
                        return error.UnsupportedModel;
                    }
                    const conv_dim: usize = @intCast(shortconv.config.conv_dim);
                    const d_conv: usize = @intCast(shortconv.config.d_conv);
                    if (shortconv_block_count == 0) {
                        log.info("inference", "CUDA shortconv block0 config", .{
                            .layer = layer_idx,
                            .d_model = shortconv.config.d_model,
                            .conv_dim = shortconv.config.conv_dim,
                            .conv_dim_out = shortconv.config.conv_dim_out,
                            .d_conv = shortconv.config.d_conv,
                            .has_bias = @as(u8, @intFromBool(shortconv.config.has_bias)),
                            .in_proj_dtype = @tagName(shortconv.weights.in_proj.dtype),
                            .in_proj_0 = shortconv.weights.in_proj.shape[0],
                            .in_proj_1 = shortconv.weights.in_proj.shape[1],
                            .conv_weight_dtype = @tagName(shortconv.weights.conv1d_weight.dtype),
                            .conv_weight_n_dims = shortconv.weights.conv1d_weight.n_dims,
                            .conv_weight_0 = shortconv.weights.conv1d_weight.shape[0],
                            .conv_weight_1 = shortconv.weights.conv1d_weight.shape[1],
                            .conv_weight_2 = shortconv.weights.conv1d_weight.shape[2],
                            .out_proj_dtype = @tagName(shortconv.weights.out_proj.dtype),
                            .out_proj_0 = shortconv.weights.out_proj.shape[0],
                            .out_proj_1 = shortconv.weights.out_proj.shape[1],
                        });
                    }
                    if (conv_dim == 0 or d_conv == 0) return error.UnsupportedModel;
                    if (shortconv.config.d_model != d_model) {
                        log.warn("inference", "CUDA shortconv d_model mismatch", .{
                            .layer = layer_idx,
                            .config_d_model = shortconv.config.d_model,
                            .model_d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln1_weight = try uploadTensor(device, allocator, shortconv.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA shortconv ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln2_weight: ?DeviceTensor = null;
                    if (shortconv.ln2_weight) |ln2| {
                        var ln2_dev = try uploadTensor(device, allocator, ln2);
                        errdefer ln2_dev.deinit(device);
                        if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                            log.warn("inference", "CUDA shortconv ln2 shape unsupported", .{
                                .layer = layer_idx,
                                .rows = ln2_dev.rows,
                                .cols = ln2_dev.cols,
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        ln2_weight = ln2_dev;
                    }
                    errdefer if (ln2_weight) |*w| w.deinit(device);

                    var in_proj_dev = try uploadLinearWeight(device, allocator, shortconv.weights.in_proj, d_model);
                    errdefer in_proj_dev.deinit(device);
                    if (in_proj_dev.cols() != 3 * conv_dim) {
                        log.warn("inference", "CUDA shortconv in_proj dim unsupported", .{
                            .layer = layer_idx,
                            .cols = in_proj_dev.cols(),
                            .expected = 3 * conv_dim,
                        });
                        return error.UnsupportedModel;
                    }

                    var out_proj_dev = try uploadLinearWeight(device, allocator, shortconv.weights.out_proj, conv_dim);
                    errdefer out_proj_dev.deinit(device);
                    if (out_proj_dev.cols() != d_model) {
                        log.warn("inference", "CUDA shortconv out_proj dim unsupported", .{
                            .layer = layer_idx,
                            .cols = out_proj_dev.cols(),
                            .expected = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var conv_weight_time_major = try uploadShortConvWeightTimeMajor(
                        device,
                        allocator,
                        shortconv.weights.conv1d_weight,
                        conv_dim,
                        d_conv,
                    );
                    errdefer conv_weight_time_major.deinit(device);

                    var conv_bias: ?DeviceTensor = null;
                    if (shortconv.weights.conv1d_bias) |bias| {
                        var bias_dev = try uploadVectorTensor(device, allocator, bias, conv_dim);
                        errdefer bias_dev.deinit(device);
                        conv_bias = bias_dev;
                    }
                    errdefer if (conv_bias) |*w| w.deinit(device);

                    const conv_state_count = std.math.mul(usize, conv_dim, d_conv) catch return error.InvalidArgument;
                    var conv_state = try allocZeroedF32Buffer(device, allocator, conv_state_count);
                    errdefer conv_state.deinit(device);

                    var ffn_w1: ?LinearWeight = null;
                    var ffn_w2: ?LinearWeight = null;
                    var ffn_w3: ?LinearWeight = null;
                    var d_ff: usize = 0;
                    if (shortconv.w1 != null or shortconv.w2 != null or shortconv.w3 != null) {
                        const w1 = shortconv.w1 orelse return error.MissingWeight;
                        const w2 = shortconv.w2 orelse return error.MissingWeight;
                        const w3 = shortconv.w3 orelse return error.MissingWeight;
                        if (ln2_weight == null) {
                            log.warn("inference", "CUDA shortconv ffn requires ln2", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        }

                        var w1_dev = try uploadLinearWeight(device, allocator, w1, d_model);
                        errdefer w1_dev.deinit(device);
                        var w3_dev = try uploadLinearWeight(device, allocator, w3, d_model);
                        errdefer w3_dev.deinit(device);
                        if (w1_dev.cols() != w3_dev.cols()) {
                            log.warn("inference", "CUDA shortconv gate/up dim mismatch", .{
                                .layer = layer_idx,
                                .w1_cols = w1_dev.cols(),
                                .w3_cols = w3_dev.cols(),
                            });
                            return error.UnsupportedModel;
                        }
                        d_ff = w1_dev.cols();
                        var w2_dev = try uploadLinearWeight(device, allocator, w2, d_ff);
                        errdefer w2_dev.deinit(device);
                        if (w2_dev.cols() != d_model) {
                            log.warn("inference", "CUDA shortconv down_proj out dim unsupported", .{
                                .layer = layer_idx,
                                .w2_cols = w2_dev.cols(),
                                .d_model = d_model,
                            });
                            return error.UnsupportedModel;
                        }
                        ffn_w1 = w1_dev;
                        ffn_w2 = w2_dev;
                        ffn_w3 = w3_dev;
                    }
                    errdefer if (ffn_w1) |*w| w.deinit(device);
                    errdefer if (ffn_w2) |*w| w.deinit(device);
                    errdefer if (ffn_w3) |*w| w.deinit(device);

                    const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    var layer_linear_bytes = in_proj_dev.byteSize() +
                        out_proj_dev.byteSize() +
                        conv_weight_time_major.byteSize() +
                        (if (conv_bias) |w| w.byteSize() else 0);
                    if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;
                    shortconv_state_bytes = std.math.add(usize, shortconv_state_bytes, conv_state.size) catch return error.InvalidArgument;
                    max_shortconv_dim = @max(max_shortconv_dim, conv_dim);

                    blocks[layer_idx].shortconv = .{
                        .conv_dim = conv_dim,
                        .d_conv = d_conv,
                        .d_ff = d_ff,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .in_proj = in_proj_dev,
                        .out_proj = out_proj_dev,
                        .conv_weight_time_major = conv_weight_time_major,
                        .conv_bias = conv_bias,
                        .conv_state = conv_state,
                        .ffn_w1 = ffn_w1,
                        .ffn_w2 = ffn_w2,
                        .ffn_w3 = ffn_w3,
                    };
                    shortconv_block_count += 1;
                },
                else => {
                    log.warn("inference", "CUDA block runtime unsupported block kind", .{
                        .layer = layer_idx,
                    });
                    return error.UnsupportedModel;
                },
            }
            initialized += 1;
        }

        return .{
            .blocks = blocks,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .attention_block_count = attention_block_count,
            .shortconv_block_count = shortconv_block_count,
            .q_norm_blocks = q_norm_blocks,
            .k_norm_blocks = k_norm_blocks,
            .linear_weight_bytes = linear_weight_bytes,
            .norm_weight_bytes = norm_weight_bytes,
            .kv_cache_bytes = kv_cache_bytes,
            .shortconv_state_bytes = shortconv_state_bytes,
            .max_shortconv_dim = max_shortconv_dim,
        };
    }

    fn deinit(self: *BlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.blocks) |*block| block.deinit(device);
        allocator.free(self.blocks);
    }

    fn maxDff(self: *const BlockRuntime) usize {
        var max_dff: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_mlp) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.shortconv) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
        }
        return max_dff;
    }

    fn maxAttn(self: *const BlockRuntime) usize {
        var max_attn: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_mlp) |block| {
                if (block.q_dim > max_attn) max_attn = block.q_dim;
            }
        }
        return if (max_attn > 0) max_attn else self.n_heads * self.head_dim;
    }

    fn maxKv(self: *const BlockRuntime) usize {
        var max_kv: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_mlp) |block| {
                if (block.kv_dim > max_kv) max_kv = block.kv_dim;
            }
        }
        return if (max_kv > 0) max_kv else self.n_kv_heads * self.head_dim;
    }

    fn maxShortConvDim(self: *const BlockRuntime) usize {
        return self.max_shortconv_dim;
    }
};

const SequencePrefillWorkspace = struct {
    token_count: usize,
    hidden_dev: compute.cuda.Buffer,
    norm_dev: compute.cuda.Buffer,
    q_dev: compute.cuda.Buffer,
    k_dev: compute.cuda.Buffer,
    v_dev: compute.cuda.Buffer,
    context_dev: compute.cuda.Buffer,
    attn_out_dev: compute.cuda.Buffer,
    ffn_gate_dev: compute.cuda.Buffer,
    ffn_up_dev: compute.cuda.Buffer,
    ffn_mul_dev: compute.cuda.Buffer,
    ffn_down_dev: compute.cuda.Buffer,
    attn_scores_dev: compute.cuda.Buffer,
    attn_probs_dev: compute.cuda.Buffer,

    fn init(
        device: *compute.cuda.Device,
        token_count: usize,
        d_model: usize,
        max_attn: usize,
        max_kv: usize,
        max_dff: usize,
        n_heads: usize,
    ) !SequencePrefillWorkspace {
        if (token_count == 0 or d_model == 0 or max_attn == 0 or max_kv == 0 or max_dff == 0 or n_heads == 0) {
            return error.InvalidArgument;
        }

        const hidden_elems = std.math.mul(usize, token_count, d_model) catch return error.InvalidArgument;
        const attn_elems = std.math.mul(usize, token_count, max_attn) catch return error.InvalidArgument;
        const kv_elems = std.math.mul(usize, token_count, max_kv) catch return error.InvalidArgument;
        const dff_elems = std.math.mul(usize, token_count, max_dff) catch return error.InvalidArgument;
        const score_elems = std.math.mul(usize, token_count, n_heads) catch return error.InvalidArgument;

        const hidden_bytes = std.math.mul(usize, hidden_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        const attn_bytes = std.math.mul(usize, attn_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        const kv_bytes = std.math.mul(usize, kv_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        const dff_bytes = std.math.mul(usize, dff_elems, @sizeOf(f32)) catch return error.InvalidArgument;
        const score_bytes = std.math.mul(usize, score_elems, @sizeOf(f32)) catch return error.InvalidArgument;

        var hidden_dev = try device.allocBuffer(hidden_bytes);
        errdefer hidden_dev.deinit(device);
        var norm_dev = try device.allocBuffer(hidden_bytes);
        errdefer norm_dev.deinit(device);
        var q_dev = try device.allocBuffer(attn_bytes);
        errdefer q_dev.deinit(device);
        var k_dev = try device.allocBuffer(kv_bytes);
        errdefer k_dev.deinit(device);
        var v_dev = try device.allocBuffer(kv_bytes);
        errdefer v_dev.deinit(device);
        var context_dev = try device.allocBuffer(attn_bytes);
        errdefer context_dev.deinit(device);
        var attn_out_dev = try device.allocBuffer(hidden_bytes);
        errdefer attn_out_dev.deinit(device);
        var ffn_gate_dev = try device.allocBuffer(dff_bytes);
        errdefer ffn_gate_dev.deinit(device);
        var ffn_up_dev = try device.allocBuffer(dff_bytes);
        errdefer ffn_up_dev.deinit(device);
        var ffn_mul_dev = try device.allocBuffer(dff_bytes);
        errdefer ffn_mul_dev.deinit(device);
        var ffn_down_dev = try device.allocBuffer(hidden_bytes);
        errdefer ffn_down_dev.deinit(device);
        var attn_scores_dev = try device.allocBuffer(score_bytes);
        errdefer attn_scores_dev.deinit(device);
        var attn_probs_dev = try device.allocBuffer(score_bytes);
        errdefer attn_probs_dev.deinit(device);

        return .{
            .token_count = token_count,
            .hidden_dev = hidden_dev,
            .norm_dev = norm_dev,
            .q_dev = q_dev,
            .k_dev = k_dev,
            .v_dev = v_dev,
            .context_dev = context_dev,
            .attn_out_dev = attn_out_dev,
            .ffn_gate_dev = ffn_gate_dev,
            .ffn_up_dev = ffn_up_dev,
            .ffn_mul_dev = ffn_mul_dev,
            .ffn_down_dev = ffn_down_dev,
            .attn_scores_dev = attn_scores_dev,
            .attn_probs_dev = attn_probs_dev,
        };
    }

    fn deinit(self: *SequencePrefillWorkspace, device: *compute.cuda.Device) void {
        self.attn_probs_dev.deinit(device);
        self.attn_scores_dev.deinit(device);
        self.ffn_down_dev.deinit(device);
        self.ffn_mul_dev.deinit(device);
        self.ffn_up_dev.deinit(device);
        self.ffn_gate_dev.deinit(device);
        self.attn_out_dev.deinit(device);
        self.context_dev.deinit(device);
        self.v_dev.deinit(device);
        self.k_dev.deinit(device);
        self.q_dev.deinit(device);
        self.norm_dev.deinit(device);
        self.hidden_dev.deinit(device);
        self.* = undefined;
    }
};

pub const CudaBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = true,
        .embedding = false,
        .warmup = false,
    };

    pub const PrefillVisionInput = vision_types.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,
    device: compute.cuda.Device,
    compute_stream: ?compute.cuda.StreamHandle = null,
    kernel_registry: compute.cuda.Registry,
    vector_add_function: ?compute.cuda.Function = null,
    vector_add_source: ?compute.cuda.registry.KernelSource = null,
    vector_add_scaled_function: ?compute.cuda.Function = null,
    vector_add_scaled_source: ?compute.cuda.registry.KernelSource = null,
    mul_function: ?compute.cuda.Function = null,
    mul_source: ?compute.cuda.registry.KernelSource = null,
    copy_function: ?compute.cuda.Function = null,
    copy_source: ?compute.cuda.registry.KernelSource = null,
    copy_u16_function: ?compute.cuda.Function = null,
    copy_u16_source: ?compute.cuda.registry.KernelSource = null,
    cast_f32_to_f16_function: ?compute.cuda.Function = null,
    cast_f32_to_f16_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_f32_function: ?compute.cuda.Function = null,
    embedding_lookup_f32_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_u16_function: ?compute.cuda.Function = null,
    embedding_lookup_u16_source: ?compute.cuda.registry.KernelSource = null,
    embedding_lookup_gaffine_u4_function: ?compute.cuda.Function = null,
    embedding_lookup_gaffine_u4_source: ?compute.cuda.registry.KernelSource = null,
    kv_write_f16_function: ?compute.cuda.Function = null,
    kv_write_f16_source: ?compute.cuda.registry.KernelSource = null,
    rmsnorm_function: ?compute.cuda.Function = null,
    rmsnorm_source: ?compute.cuda.registry.KernelSource = null,
    rope_function: ?compute.cuda.Function = null,
    rope_source: ?compute.cuda.registry.KernelSource = null,
    rope_store_f16_function: ?compute.cuda.Function = null,
    rope_store_f16_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f32_function: ?compute.cuda.Function = null,
    attn_scores_heads_f32_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_scores_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    softmax_rows_function: ?compute.cuda.Function = null,
    softmax_rows_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f32_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    silu_function: ?compute.cuda.Function = null,
    silu_source: ?compute.cuda.registry.KernelSource = null,
    silu_mul_function: ?compute.cuda.Function = null,
    silu_mul_source: ?compute.cuda.registry.KernelSource = null,
    gelu_mul_function: ?compute.cuda.Function = null,
    gelu_mul_source: ?compute.cuda.registry.KernelSource = null,
    shortconv_step_function: ?compute.cuda.Function = null,
    shortconv_step_source: ?compute.cuda.registry.KernelSource = null,
    argmax_function: ?compute.cuda.Function = null,
    argmax_source: ?compute.cuda.registry.KernelSource = null,
    matmul_f16_function: ?compute.cuda.Function = null,
    matmul_f16_source: ?compute.cuda.registry.KernelSource = null,
    matmul_bf16_function: ?compute.cuda.Function = null,
    matmul_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_f16_function: ?compute.cuda.Function = null,
    matvec_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_bf16_function: ?compute.cuda.Function = null,
    matvec_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_f16_function: ?compute.cuda.Function = null,
    matvec_gate_up_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_gate_up_bf16_function: ?compute.cuda.Function = null,
    matvec_gate_up_bf16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_qkv_f16_function: ?compute.cuda.Function = null,
    matvec_qkv_f16_source: ?compute.cuda.registry.KernelSource = null,
    matvec_qkv_bf16_function: ?compute.cuda.Function = null,
    matvec_qkv_bf16_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_gate_up_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_gate_up_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_u4_matvec_qkv_function: ?compute.cuda.Function = null,
    gaffine_u4_matvec_qkv_source: ?compute.cuda.registry.KernelSource = null,
    gaffine_sequence_rows_supported: bool = false,
    gaffine_sequence_fused_qkv_supported: bool = false,
    gaffine_sequence_fused_gate_up_supported: bool = false,
    kernel_arg_pack: compute.cuda.ArgPack,
    blas: compute.cuda.Blas,
    prototype: PrototypeRuntime,
    block_runtime: BlockRuntime,
    d_model: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    rope_dim: usize,
    attention_scale: f32,
    norm_eps: f32,
    max_batch_size: usize = 1,
    slot_in_use: bool = false,
    slot_position: usize = 0,
    slot_rope_position_delta: isize = 0,
    slot_logits: []f32,
    layer_program_dispatch_total: [256]u64 = [_]u64{0} ** 256,
    prefill_dispatch_window_start: [256]u64 = [_]u64{0} ** 256,
    sequence_prefill_hidden_host: []f32,
    sequence_prefill_workspace: ?SequencePrefillWorkspace = null,
    argmax_index_dev: compute.cuda.Buffer,

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !CudaBackend {
        var device = try compute.cuda.Device.init();
        errdefer device.deinit();

        log.info("inference", "CUDA device ready", .{ .name = device.name() });
        var backend = CudaBackend{
            .allocator = allocator,
            .loaded = loaded,
            .vision_runtime = null,
            .device = device,
            .compute_stream = null,
            .kernel_registry = undefined,
            .kernel_arg_pack = compute.cuda.ArgPack.init(allocator),
            .blas = undefined,
            .prototype = undefined,
            .block_runtime = undefined,
            .d_model = @intCast(loaded.config.d_model),
            .vocab_size = @intCast(loaded.config.vocab_size),
            .n_heads = @intCast(loaded.config.n_heads),
            .n_kv_heads = @intCast(loaded.config.n_kv_groups),
            .head_dim = @intCast(loaded.config.head_dim),
            .max_seq_len = @intCast(loaded.config.max_seq_len),
            .rope_dim = 0,
            .attention_scale = 0.0,
            .norm_eps = prototype_eps,
            .slot_logits = undefined,
            .layer_program_dispatch_total = [_]u64{0} ** 256,
            .prefill_dispatch_window_start = [_]u64{0} ** 256,
            .sequence_prefill_hidden_host = &.{},
            .sequence_prefill_workspace = null,
            .argmax_index_dev = undefined,
        };
        if (backend.n_heads == 0 or backend.n_kv_heads == 0 or backend.head_dim == 0 or backend.max_seq_len == 0) {
            return error.InvalidArgument;
        }
        if (backend.n_heads % backend.n_kv_heads != 0) return error.UnsupportedModel;
        backend.rope_dim = if (loaded.config.rope_dim > 0)
            @intCast(loaded.config.rope_dim)
        else
            backend.head_dim;
        if (backend.rope_dim == 0 or backend.rope_dim > backend.head_dim or (backend.rope_dim & 1) != 0) {
            return error.UnsupportedModel;
        }
        backend.attention_scale = if (loaded.config.attention_multiplier > 0.0)
            loaded.config.attention_multiplier
        else if (loaded.config.query_pre_attn_scalar > 0.0)
            1.0 / std.math.sqrt(loaded.config.query_pre_attn_scalar)
        else
            1.0 / std.math.sqrt(@as(f32, @floatFromInt(backend.head_dim)));
        backend.norm_eps = if (loaded.config.norm_eps > 0.0) loaded.config.norm_eps else prototype_eps;
        if (backend.device.supportsStreams()) {
            const stream = try backend.device.createStream();
            backend.compute_stream = stream;
            backend.device.setLaunchStream(stream);
        }
        errdefer {
            backend.device.setLaunchStream(null);
            if (backend.compute_stream) |stream| backend.device.destroyStream(stream);
            backend.compute_stream = null;
        }
        backend.kernel_registry = compute.cuda.Registry.init(allocator, &backend.device);
        errdefer backend.kernel_registry.deinit();
        backend.slot_logits = try allocator.alloc(f32, backend.vocab_size);
        errdefer allocator.free(backend.slot_logits);
        backend.argmax_index_dev = try backend.device.allocBuffer(@sizeOf(u32));
        errdefer backend.argmax_index_dev.deinit(&backend.device);
        backend.block_runtime = try BlockRuntime.init(allocator, &backend.device, loaded);
        errdefer backend.block_runtime.deinit(allocator, &backend.device);
        backend.vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (backend.vision_runtime) |*rt| rt.deinit();
        if (loaded.config.use_qk_norm and
            (backend.block_runtime.q_norm_blocks != backend.block_runtime.attention_block_count or
                backend.block_runtime.k_norm_blocks != backend.block_runtime.attention_block_count))
        {
            log.warn("inference", "CUDA backend requires explicit q/k norm weights when qk_norm is enabled", .{
                .q_norm_blocks = backend.block_runtime.q_norm_blocks,
                .k_norm_blocks = backend.block_runtime.k_norm_blocks,
                .layers = backend.block_runtime.attention_block_count,
            });
            return error.UnsupportedModel;
        }
        const max_dff = backend.block_runtime.maxDff();
        const max_attn = backend.block_runtime.maxAttn();
        const max_kv = backend.block_runtime.maxKv();
        const max_shortconv_dim = backend.block_runtime.maxShortConvDim();
        backend.blas = try compute.cuda.Blas.init(&backend.device);
        errdefer backend.blas.deinit(&backend.device);
        backend.prototype = try PrototypeRuntime.init(
            allocator,
            &backend.device,
            loaded,
            max_dff,
            max_attn,
            max_kv,
            max_shortconv_dim,
            backend.max_seq_len,
            backend.n_heads,
            backend.head_dim,
        );
        errdefer backend.prototype.deinit(allocator, &backend.device);
        try backend.initKernelFunctions();

        if (loaded.original_weight_dtype == .grouped_affine_u4) {
            backend.gaffine_sequence_rows_supported = smoke_checks.probeGaffineU4SequenceRowsSupport(&backend) catch false;
            if (!backend.gaffine_sequence_rows_supported) {
                log.warn("inference", "CUDA gaffine sequence prefill using per-row linear fallback (multi-row parity probe failed)", .{
                    .reason = "gaffine_batch_rows_probe_failed",
                });
            } else {
                backend.gaffine_sequence_fused_qkv_supported = smoke_checks.probeGaffineU4SequenceFusedQkvSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_qkv_supported) {
                    log.warn("inference", "CUDA gaffine sequence prefill using unfused QKV fallback (multi-row fused parity probe failed)", .{
                        .reason = "gaffine_batch_rows_fused_qkv_probe_failed",
                    });
                }

                backend.gaffine_sequence_fused_gate_up_supported = smoke_checks.probeGaffineU4SequenceFusedGateUpSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_gate_up_supported) {
                    log.warn("inference", "CUDA gaffine sequence prefill using unfused gate/up fallback (multi-row fused parity probe failed)", .{
                        .reason = "gaffine_batch_rows_fused_gate_up_probe_failed",
                    });
                }
            }
        }

        if (run_startup_selftests) {
            try smoke_checks.runMatmulSmoke(&backend);
            try smoke_checks.runKernelSmoke(&backend);
        }
        log.info("inference", "CUDA layered decode path ready", .{
            .d_model = backend.d_model,
            .projected_vocab = backend.prototype.projected_vocab,
            .max_dff = backend.prototype.max_dff,
            .max_attn = backend.prototype.max_attn,
            .max_kv = backend.prototype.max_kv,
            .max_seq = backend.max_seq_len,
            .kv_capacity_init = backend.initialKvCapacity(),
            .n_heads = backend.n_heads,
            .n_kv = backend.n_kv_heads,
            .head_dim = backend.head_dim,
            .use_qk_norm = @as(u8, @intFromBool(loaded.config.use_qk_norm)),
            .attention_bias = @as(u8, @intFromBool(loaded.config.attention_bias)),
            .norm_weight_offset = loaded.runtime.weight_offset,
            .qk_norm_weight_offset = loaded.runtime.qk_norm_weight_offset,
            .q_norm_blocks = backend.block_runtime.q_norm_blocks,
            .k_norm_blocks = backend.block_runtime.k_norm_blocks,
            .vector_add_kernel = @as(u8, @intFromBool(backend.vector_add_function != null)),
            .vector_add_scaled_kernel = @as(u8, @intFromBool(backend.vector_add_scaled_function != null)),
            .rmsnorm_kernel = @as(u8, @intFromBool(backend.rmsnorm_function != null)),
            .mul_kernel = @as(u8, @intFromBool(backend.mul_function != null)),
            .copy_kernel = @as(u8, @intFromBool(backend.copy_function != null)),
            .copy_u16_kernel = @as(u8, @intFromBool(backend.copy_u16_function != null)),
            .cast_f32_to_f16_kernel = @as(u8, @intFromBool(backend.cast_f32_to_f16_function != null)),
            .embedding_lookup_f32_kernel = @as(u8, @intFromBool(backend.embedding_lookup_f32_function != null)),
            .embedding_lookup_u16_kernel = @as(u8, @intFromBool(backend.embedding_lookup_u16_function != null)),
            .embedding_lookup_gaffine_u4_kernel = @as(u8, @intFromBool(backend.embedding_lookup_gaffine_u4_function != null)),
            .kv_write_f16_kernel = @as(u8, @intFromBool(backend.kv_write_f16_function != null)),
            .rope_kernel = @as(u8, @intFromBool(backend.rope_function != null)),
            .rope_store_f16_kernel = @as(u8, @intFromBool(backend.rope_store_f16_function != null)),
            .attn_scores_heads_f32_kernel = @as(u8, @intFromBool(backend.attn_scores_heads_f32_function != null)),
            .attn_scores_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_scores_heads_f16_kv_function != null)),
            .attn_fused_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_fused_heads_f16_kv_function != null)),
            .attn_fused_heads_f16_kv_enabled = @as(u8, @intFromBool(enable_fused_attention_f16_kv)),
            .attn_fused_heads_f16_kv_max_seq = max_fused_attention_f16_kv_seq_len,
            .softmax_rows_kernel = @as(u8, @intFromBool(backend.softmax_rows_function != null)),
            .attn_weighted_sum_heads_f32_kernel = @as(u8, @intFromBool(backend.attn_weighted_sum_heads_f32_function != null)),
            .attn_weighted_sum_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_weighted_sum_heads_f16_kv_function != null)),
            .attn_score_buffers = @as(u8, @intFromBool(backend.prototype.attn_scores_dev != null and backend.prototype.attn_probs_dev != null)),
            .silu_kernel = @as(u8, @intFromBool(backend.silu_function != null)),
            .silu_mul_kernel = @as(u8, @intFromBool(backend.silu_mul_function != null)),
            .gelu_mul_kernel = @as(u8, @intFromBool(backend.gelu_mul_function != null)),
            .shortconv_step_kernel = @as(u8, @intFromBool(backend.shortconv_step_function != null)),
            .argmax_kernel = @as(u8, @intFromBool(backend.argmax_function != null)),
            .matmul_f16_kernel = @as(u8, @intFromBool(backend.matmul_f16_function != null)),
            .matmul_bf16_kernel = @as(u8, @intFromBool(backend.matmul_bf16_function != null)),
            .matvec_f16_kernel = @as(u8, @intFromBool(backend.matvec_f16_function != null)),
            .matvec_bf16_kernel = @as(u8, @intFromBool(backend.matvec_bf16_function != null)),
            .matvec_gate_up_f16_kernel = @as(u8, @intFromBool(backend.matvec_gate_up_f16_function != null)),
            .matvec_gate_up_bf16_kernel = @as(u8, @intFromBool(backend.matvec_gate_up_bf16_function != null)),
            .matvec_qkv_f16_kernel = @as(u8, @intFromBool(backend.matvec_qkv_f16_function != null)),
            .matvec_qkv_bf16_kernel = @as(u8, @intFromBool(backend.matvec_qkv_bf16_function != null)),
            .gaffine_u4_matvec_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_function != null)),
            .gaffine_u4_matvec_gate_up_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_gate_up_function != null)),
            .gaffine_u4_matvec_qkv_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_qkv_function != null)),
            .gaffine_sequence_rows_supported = @as(u8, @intFromBool(backend.gaffine_sequence_rows_supported)),
            .gaffine_sequence_fused_qkv_supported = @as(u8, @intFromBool(backend.gaffine_sequence_fused_qkv_supported)),
            .gaffine_sequence_fused_gate_up_supported = @as(u8, @intFromBool(backend.gaffine_sequence_fused_gate_up_supported)),
            .kv_dtype = if (kv_cache_dtype_fp16) "f16" else "f32",
            .linear_weight_mib = bytesToMiB(backend.block_runtime.linear_weight_bytes),
            .norm_weight_mib = bytesToMiB(backend.block_runtime.norm_weight_bytes),
            .kv_cache_mib = bytesToMiB(backend.block_runtime.kv_cache_bytes),
            .shortconv_state_mib = bytesToMiB(backend.block_runtime.shortconv_state_bytes),
            .prototype_mib = bytesToMiB(backend.prototype.deviceByteSize()),
            .slot_logits_mib = bytesToMiB(std.math.mul(usize, backend.slot_logits.len, @sizeOf(f32)) catch 0),
            .stream_token_select = "gpu_argmax",
            .stream_enabled = @as(u8, @intFromBool(backend.compute_stream != null)),
            .device_blocks = backend.block_runtime.blocks.len,
            .attention_blocks = backend.block_runtime.attention_block_count,
            .shortconv_blocks = backend.block_runtime.shortconv_block_count,
            .model_norm = @as(u8, @intFromBool(backend.prototype.using_model_norm)),
            .model_projection = @as(u8, @intFromBool(backend.prototype.using_model_projection)),
            .projection_lm_head = @as(u8, @intFromBool(backend.prototype.projection_from_lm_head)),
            .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
            .model_embeddings = @as(u8, @intFromBool(backend.prototype.using_model_embeddings)),
            .embedding_lookup_device = @as(u8, @intFromBool(backend.prototype.embedding_lookup != null)),
            .embed_dtype = @tagName(loaded.token_embeddings.dtype),
            .embed_shape_0 = loaded.token_embeddings.shape[0],
            .embed_shape_1 = loaded.token_embeddings.shape[1],
        });
        return backend;
    }

    pub fn deinit(self: *CudaBackend) void {
        if (self.vision_runtime) |*rt| rt.deinit();
        self.device.setLaunchStream(null);
        if (self.compute_stream) |stream| {
            _ = self.device.synchronizeStream(stream) catch {};
            self.device.destroyStream(stream);
            self.compute_stream = null;
        }
        if (self.sequence_prefill_workspace) |*ws| {
            ws.deinit(&self.device);
            self.sequence_prefill_workspace = null;
        }
        if (self.sequence_prefill_hidden_host.len > 0) {
            self.allocator.free(self.sequence_prefill_hidden_host);
            self.sequence_prefill_hidden_host = &.{};
        }
        self.argmax_index_dev.deinit(&self.device);
        self.allocator.free(self.slot_logits);
        self.block_runtime.deinit(self.allocator, &self.device);
        self.prototype.deinit(self.allocator, &self.device);
        self.blas.deinit(&self.device);
        self.kernel_arg_pack.deinit();
        self.kernel_registry.deinit();
        self.device.deinit();
        self.* = undefined;
    }

    pub fn maxBatchSize(self: *const CudaBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const CudaBackend) usize {
        return self.vocab_size;
    }

    fn initialKvCapacity(self: *const CudaBackend) usize {
        for (self.block_runtime.blocks) |layer| {
            if (layer.attention_mlp) |block| return block.kv_capacity;
        }
        return 0;
    }

    fn ensureSequencePrefillHiddenHost(self: *CudaBackend, token_count: usize) ![]f32 {
        if (token_count == 0) return error.InvalidArgument;
        const required = std.math.mul(usize, token_count, self.d_model) catch return error.InvalidArgument;
        if (self.sequence_prefill_hidden_host.len < required) {
            const current = self.sequence_prefill_hidden_host.len;
            const grown = if (current == 0)
                required
            else
                std.math.mul(usize, current, 2) catch required;
            const target = @max(required, grown);
            const new_buffer = try self.allocator.alloc(f32, target);
            if (current > 0) self.allocator.free(self.sequence_prefill_hidden_host);
            self.sequence_prefill_hidden_host = new_buffer;
        }
        return self.sequence_prefill_hidden_host[0..required];
    }

    fn ensureSequencePrefillWorkspace(self: *CudaBackend, token_count: usize) !*SequencePrefillWorkspace {
        if (token_count == 0) return error.InvalidArgument;
        if (self.sequence_prefill_workspace) |*ws| {
            if (ws.token_count >= token_count) return ws;
            ws.deinit(&self.device);
            self.sequence_prefill_workspace = null;
        }

        const grown_tokens = if (token_count < 256)
            token_count
        else
            std.math.mul(usize, token_count, 2) catch token_count;
        const capacity_tokens = @min(grown_tokens, self.max_seq_len);
        self.sequence_prefill_workspace = try SequencePrefillWorkspace.init(
            &self.device,
            capacity_tokens,
            self.d_model,
            self.prototype.max_attn,
            self.prototype.max_kv,
            self.prototype.max_dff,
            self.n_heads,
        );
        return &self.sequence_prefill_workspace.?;
    }

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        return prefill_mod.prefill(self, tokens, logits_out);
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        return decode_mod.decode(self, token, position, logits_out);
    }

    pub fn decodeStreaming(
        self: *CudaBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        return decode_mod.decodeStreaming(
            self,
            first_token,
            start_position,
            max_tokens,
            eos_token_ids,
            output_tokens,
            callback,
            callback_data,
        );
    }

    pub fn allocSlot(self: *CudaBackend) ?usize {
        return decode_mod.allocSlot(self);
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.freeSlot(self, slot_index);
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.resetSlot(self, slot_index);
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        return decode_mod.getPosition(self, slot_index);
    }

    pub fn prefillSlot(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        return prefill_mod.prefillSlot(self, slot_index, tokens, logits_out);
    }

    pub fn prefillSlotWithVision(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        if (vision_input == null) return self.prefillSlot(slot_index, tokens, logits_out);

        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "empty_tokens",
                .slot_index = slot_index,
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "logits_len_mismatch",
                .slot_index = slot_index,
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        if (!self.slot_in_use or slot_index != 0) {
            log.warn("inference", "CUDA prefillSlotWithVision invalid args", .{
                .reason = "slot_state",
                .slot_index = slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
            });
            return error.InvalidArgument;
        }
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        const vision = if (self.vision_runtime) |*rt|
            rt
        else {
            log.warn("inference", "CUDA vision prefill requested but vision runtime is unavailable", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
            });
            return error.UnsupportedContentType;
        };

        const vi = vision_input.?;
        var encoded_vision_output = try vision.encodeImages(vi.images);
        defer encoded_vision_output.deinit(self.allocator);

        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        if (encoded_vision_output.deepstack_layer_embeddings.len > 0) {
            image_token_positions = try collectTokenPositions(self.allocator, tokens, vi.image_token_id);
            if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;

            const expected_values = std.math.mul(usize, image_token_positions.len, self.d_model) catch return error.InvalidArgument;
            for (encoded_vision_output.deepstack_layer_embeddings, 0..) |layer_features, layer_idx| {
                if (layer_features.len != expected_values) {
                    log.warn("inference", "CUDA vision deepstack shape mismatch", .{
                        .slot_index = slot_index,
                        .layer_index = layer_idx,
                        .expected_values = expected_values,
                        .actual_values = layer_features.len,
                        .image_positions = image_token_positions.len,
                        .d_model = self.d_model,
                    });
                    return error.InvalidArgument;
                }
            }
        }

        self.slot_rope_position_delta = 0;
        self.beginPrefillDispatchWindow();
        const prefill_start_ns: i128 = std.time.nanoTimestamp();
        try self.ensureKvCapacity(tokens.len);

        const hidden_count = std.math.mul(usize, tokens.len, self.d_model) catch return error.InvalidArgument;
        const hidden_host = try self.allocator.alloc(f32, hidden_count);
        defer self.allocator.free(hidden_host);

        try populatePrefillHiddenFromTokens(self.loaded, tokens, self.d_model, hidden_host);
        try vision_runtime_mod.scatterVisionEmbeddings(
            hidden_host,
            tokens.len,
            self.d_model,
            tokens,
            vi.image_token_id,
            encoded_vision_output.merged_embeddings,
        );

        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const row_start = std.math.mul(usize, i, self.d_model) catch return error.InvalidArgument;
            const hidden_override = hidden_host[row_start .. row_start + self.d_model];
            const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
            const deepstack_feature_index = if (image_token_positions.len > 0) findPositionIndex(image_token_positions, i) else null;
            try self.computeGpuPrototypeLogitsWithLayerLimit(
                tokens[i],
                i,
                if (download_logits) self.slot_logits else null,
                self.block_runtime.blocks.len,
                download_logits,
                download_logits,
                false,
                hidden_override,
                if (encoded_vision_output.deepstack_layer_embeddings.len > 0) encoded_vision_output.deepstack_layer_embeddings else null,
                deepstack_feature_index,
            );
        }

        const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
        self.logPrefillTimingImpl("prefill_slot_vision", tokens.len, prefill_elapsed_ns);
        @memcpy(logits_out, self.slot_logits);
        self.slot_position = tokens.len;
    }

    /// Optional multi-token prefill accelerator.
    ///
    /// This path is intentionally conservative: it only activates for attention-only
    /// layer programs that preserve residual output semantics. Unsupported
    /// topologies always fall back to the canonical LayerOp prefill route.
    pub fn trySequencePrefill(
        self: *CudaBackend,
        tokens: []const u32,
        logits_out: []f32,
        mode: []const u8,
    ) !bool {
        self.beginPrefillDispatchWindow();
        const gate = self.sequencePrefillGate(tokens.len);
        if (gate != .ok) {
            log.debug("inference", "CUDA sequence prefill path inactive", .{
                .mode = mode,
                .tokens = tokens.len,
                .reason = @tagName(gate),
            }, @src());
            return false;
        }
        log.debug("inference", "CUDA sequence prefill path active", .{
            .mode = mode,
            .tokens = tokens.len,
        }, @src());

        const rmsnorm_function = self.rmsnorm_function orelse return error.CudaKernelUnavailable;
        const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
        const softmax_rows_function = self.softmax_rows_function orelse return error.CudaKernelUnavailable;
        const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
        const cast_f32_to_f16_function = self.cast_f32_to_f16_function;
        const attn_scores_heads_f32_function = self.attn_scores_heads_f32_function;
        const attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function;
        const attn_weighted_sum_heads_f32_function = self.attn_weighted_sum_heads_f32_function;
        const attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function;
        const gelu_mul_function = self.gelu_mul_function;
        const silu_mul_function = self.silu_mul_function;

        const prefill_start_ns: i128 = std.time.nanoTimestamp();
        try self.ensureKvCapacity(tokens.len);

        const hidden_count = std.math.mul(usize, tokens.len, self.d_model) catch return error.InvalidArgument;
        const hidden_host = try self.ensureSequencePrefillHiddenHost(tokens.len);
        try populatePrefillHiddenFromTokens(self.loaded, tokens, self.d_model, hidden_host);

        const ws = try self.ensureSequencePrefillWorkspace(tokens.len);

        try ws.hidden_dev.upload(&self.device, std.mem.sliceAsBytes(hidden_host));

        const token_count_u32: u32 = @intCast(tokens.len);
        const d_model_u32: u32 = @intCast(self.d_model);
        const n_heads_u32: u32 = @intCast(self.n_heads);
        const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
        const head_dim_u32: u32 = @intCast(self.head_dim);
        const rope_dim_u32: u32 = @intCast(self.rope_dim);
        const kv_groups_u32: u32 = @intCast(self.n_heads / self.n_kv_heads);
        const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
        const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
            self.loaded.config.rope_local_theta
        else
            global_rope_theta;

        for (self.block_runtime.blocks) |*layer| {
            const block = if (layer.attention_mlp) |*b| b else return false;

            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_function,
                &ws.hidden_dev,
                &block.ln1_weight.buffer,
                &ws.norm_dev,
                token_count_u32,
                d_model_u32,
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );

            if (!try self.trySequenceFusedQkvRows(&ws.norm_dev, tokens.len, block, &ws.q_dev, &ws.k_dev, &ws.v_dev)) {
                try self.linearForwardRows(&ws.norm_dev, tokens.len, &block.q_proj, &ws.q_dev);
                try self.linearForwardRows(&ws.norm_dev, tokens.len, &block.k_proj, &ws.k_dev);
                try self.linearForwardRows(&ws.norm_dev, tokens.len, &block.v_proj, &ws.v_dev);
            }

            const q_rows = std.math.mul(usize, tokens.len, self.n_heads) catch return error.InvalidArgument;
            const kv_rows = std.math.mul(usize, tokens.len, self.n_kv_heads) catch return error.InvalidArgument;
            if (block.q_norm_weight) |*q_norm| {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.q_dev,
                    &q_norm.buffer,
                    &ws.q_dev,
                    @intCast(q_rows),
                    head_dim_u32,
                    self.norm_eps,
                    self.loaded.runtime.qk_norm_weight_offset,
                );
            }
            if (block.k_norm_weight) |*k_norm| {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.k_dev,
                    &k_norm.buffer,
                    &ws.k_dev,
                    @intCast(kv_rows),
                    head_dim_u32,
                    self.norm_eps,
                    self.loaded.runtime.qk_norm_weight_offset,
                );
            }

            const q_row_bytes = std.math.mul(usize, block.q_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            const kv_row_bytes = std.math.mul(usize, block.kv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            const layer_rope_theta = if (block.sliding_window > 0) local_rope_theta else global_rope_theta;
            for (tokens, 0..) |_, token_index| {
                const q_offset = std.math.mul(usize, token_index, q_row_bytes) catch return error.InvalidArgument;
                const kv_offset = std.math.mul(usize, token_index, kv_row_bytes) catch return error.InvalidArgument;
                var q_row = try bufferSlice(&ws.q_dev, q_offset, q_row_bytes);
                var k_row = try bufferSlice(&ws.k_dev, kv_offset, kv_row_bytes);

                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_function,
                    &q_row,
                    n_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(token_index),
                    layer_rope_theta,
                );
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_function,
                    &k_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(token_index),
                    layer_rope_theta,
                );
            }

            const kv_values = std.math.mul(usize, tokens.len, block.kv_dim) catch return error.InvalidArgument;
            if (kv_values > std.math.maxInt(u32)) return error.InvalidArgument;
            if (kv_cache_dtype_fp16) {
                const cast_kernel = cast_f32_to_f16_function orelse return error.CudaKernelUnavailable;
                const kv_bytes = std.math.mul(usize, kv_values, @sizeOf(u16)) catch return error.InvalidArgument;
                var k_cache_all = try bufferSlice(&block.k_cache, 0, kv_bytes);
                var v_cache_all = try bufferSlice(&block.v_cache, 0, kv_bytes);
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_kernel,
                    &ws.k_dev,
                    &k_cache_all,
                    @intCast(kv_values),
                );
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_kernel,
                    &ws.v_dev,
                    &v_cache_all,
                    @intCast(kv_values),
                );
            } else {
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function,
                    &ws.k_dev,
                    &block.k_cache,
                    @intCast(kv_values),
                );
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function,
                    &ws.v_dev,
                    &block.v_cache,
                    @intCast(kv_values),
                );
            }

            const q_row_bytes_ctx = std.math.mul(usize, block.q_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
            const cache_row_bytes = std.math.mul(usize, block.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;

            for (tokens, 0..) |_, token_index| {
                const token_u32: u32 = @intCast(token_index);
                var seq_len_u32: u32 = token_u32 + 1;
                var start_row: usize = 0;
                if (block.sliding_window > 0 and block.is_causal) {
                    const window_u32 = std.math.cast(u32, block.sliding_window) orelse std.math.maxInt(u32);
                    if (seq_len_u32 > window_u32) {
                        start_row = seq_len_u32 - window_u32;
                        seq_len_u32 = window_u32;
                    }
                }

                const q_offset = std.math.mul(usize, token_index, q_row_bytes_ctx) catch return error.InvalidArgument;
                var q_row = try bufferSlice(&ws.q_dev, q_offset, q_row_bytes_ctx);
                var ctx_row = try bufferSlice(&ws.context_dev, q_offset, q_row_bytes_ctx);

                const cache_offset = std.math.mul(usize, start_row, cache_row_bytes) catch return error.InvalidArgument;
                const cache_rows = std.math.mul(usize, @as(usize, seq_len_u32), cache_row_bytes) catch return error.InvalidArgument;
                var k_cache_view = try bufferSlice(&block.k_cache, cache_offset, cache_rows);
                var v_cache_view = try bufferSlice(&block.v_cache, cache_offset, cache_rows);

                const score_count = std.math.mul(usize, self.n_heads, @as(usize, seq_len_u32)) catch return error.InvalidArgument;
                const score_bytes = std.math.mul(usize, score_count, @sizeOf(f32)) catch return error.InvalidArgument;
                var scores_view = try bufferSlice(&ws.attn_scores_dev, 0, score_bytes);
                var probs_view = try bufferSlice(&ws.attn_probs_dev, 0, score_bytes);

                if (kv_cache_dtype_fp16) {
                    try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                        &q_row,
                        &k_cache_view,
                        &scores_view,
                        n_heads_u32,
                        seq_len_u32,
                        @intCast(block.kv_dim),
                        kv_groups_u32,
                        head_dim_u32,
                        self.attention_scale,
                    );
                    try compute.cuda.softmax_rows.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        softmax_rows_function,
                        &scores_view,
                        &probs_view,
                        n_heads_u32,
                        seq_len_u32,
                    );
                    try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                        &probs_view,
                        &v_cache_view,
                        &ctx_row,
                        n_heads_u32,
                        seq_len_u32,
                        @intCast(block.kv_dim),
                        kv_groups_u32,
                        head_dim_u32,
                    );
                } else {
                    try compute.cuda.attn_scores_heads_f32.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable,
                        &q_row,
                        &k_cache_view,
                        &scores_view,
                        n_heads_u32,
                        seq_len_u32,
                        @intCast(block.kv_dim),
                        kv_groups_u32,
                        head_dim_u32,
                        self.attention_scale,
                    );
                    try compute.cuda.softmax_rows.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        softmax_rows_function,
                        &scores_view,
                        &probs_view,
                        n_heads_u32,
                        seq_len_u32,
                    );
                    try compute.cuda.attn_weighted_sum_heads_f32.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable,
                        &probs_view,
                        &v_cache_view,
                        &ctx_row,
                        n_heads_u32,
                        seq_len_u32,
                        @intCast(block.kv_dim),
                        kv_groups_u32,
                        head_dim_u32,
                    );
                }
            }

            try self.linearForwardRows(&ws.context_dev, tokens.len, &block.o_proj, &ws.attn_out_dev);
            const hidden_count_u32: u32 = @intCast(hidden_count);
            if (block.pre_ffn_norm_weight != null or block.post_ffn_norm_weight != null) {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.attn_out_dev,
                    &block.ln2_weight.buffer,
                    &ws.attn_out_dev,
                    token_count_u32,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );
                try self.addResidualWithModelScale(&ws.hidden_dev, &ws.attn_out_dev, hidden_count_u32);

                const pre_ffn_norm = if (block.pre_ffn_norm_weight) |*w| &w.buffer else &block.ln2_weight.buffer;
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.hidden_dev,
                    pre_ffn_norm,
                    &ws.norm_dev,
                    token_count_u32,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );
            } else {
                try self.addResidualWithModelScale(&ws.hidden_dev, &ws.attn_out_dev, hidden_count_u32);

                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.hidden_dev,
                    &block.ln2_weight.buffer,
                    &ws.norm_dev,
                    token_count_u32,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );
            }

            if (!try self.trySequenceFusedGateUpRows(&ws.norm_dev, tokens.len, &block.w1, &block.w3, &ws.ffn_gate_dev, &ws.ffn_up_dev)) {
                try self.linearForwardRows(&ws.norm_dev, tokens.len, &block.w1, &ws.ffn_gate_dev);
                try self.linearForwardRows(&ws.norm_dev, tokens.len, &block.w3, &ws.ffn_up_dev);
            }
            const ffn_count = std.math.mul(usize, tokens.len, block.d_ff) catch return error.InvalidArgument;
            if (ffn_count > std.math.maxInt(u32)) return error.InvalidArgument;
            if (self.loaded.config.use_gelu) {
                const gelu_kernel = gelu_mul_function orelse return error.CudaKernelUnavailable;
                try compute.cuda.gelu_mul.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    gelu_kernel,
                    &ws.ffn_gate_dev,
                    &ws.ffn_up_dev,
                    &ws.ffn_mul_dev,
                    @intCast(ffn_count),
                );
            } else {
                const silu_kernel = silu_mul_function orelse return error.CudaKernelUnavailable;
                try compute.cuda.silu_mul.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    silu_kernel,
                    &ws.ffn_gate_dev,
                    &ws.ffn_up_dev,
                    &ws.ffn_mul_dev,
                    @intCast(ffn_count),
                );
            }
            try self.linearForwardRows(&ws.ffn_mul_dev, tokens.len, &block.w2, &ws.ffn_down_dev);
            if (block.post_ffn_norm_weight) |*post_ffn_norm| {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &ws.ffn_down_dev,
                    &post_ffn_norm.buffer,
                    &ws.ffn_down_dev,
                    token_count_u32,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );
            }
            try self.addResidualWithModelScale(&ws.hidden_dev, &ws.ffn_down_dev, hidden_count_u32);
        }

        const last_row_offset = std.math.mul(usize, tokens.len - 1, self.d_model * @sizeOf(f32)) catch return error.InvalidArgument;
        var last_row = try bufferSlice(&ws.hidden_dev, last_row_offset, self.d_model * @sizeOf(f32));
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            rmsnorm_function,
            &last_row,
            &self.prototype.norm_weight_dev,
            &self.prototype.norm_out_dev,
            1,
            d_model_u32,
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
        try self.linearForward(&self.prototype.norm_out_dev, &self.prototype.projection_weight, &self.prototype.logits_dev);
        try self.prototype.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.prototype.projected_logits_host));

        if (self.prototype.projected_vocab == logits_out.len) {
            @memcpy(logits_out, self.prototype.projected_logits_host);
        } else {
            @memset(logits_out, -1.0e9);
            @memcpy(logits_out[0..self.prototype.projected_vocab], self.prototype.projected_logits_host);
        }
        if (self.loaded.config.logits_scaling != 1.0) {
            for (logits_out) |*v| {
                v.* /= self.loaded.config.logits_scaling;
            }
        }

        const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
        logPrefillTiming(self, mode, tokens.len, prefill_elapsed_ns);
        return true;
    }

    const SequencePrefillGate = enum {
        ok,
        token_count_le_one,
        token_count_exceeds_max_seq,
        shortconv_present,
        non_attention_blocks_present,
        missing_core_kernels,
        missing_activation_kernel,
        missing_fp16_kv_kernels,
        missing_f32_kv_kernels,
        unsupported_linear_projection,
        unsupported_layer_program,
    };

    fn sequencePrefillProgramCompatible(program: []const layer_ops.LayerOp) bool {
        for (program) |op| switch (op) {
            .kernel => |kernel_op| switch (kernel_op.debug_type) {
                .norm, .multihead_attention, .mlp, .moe => {},
                else => return false,
            },
            .add => {},
            else => return false,
        };
        return finalProgramOutputBuffer(program) == .residual;
    }

    fn sequencePrefillGate(self: *const CudaBackend, token_count: usize) SequencePrefillGate {
        if (token_count <= 1) return .token_count_le_one;
        if (token_count > self.max_seq_len) return .token_count_exceeds_max_seq;
        if (self.block_runtime.shortconv_block_count != 0) return .shortconv_present;
        if (self.block_runtime.attention_block_count != self.block_runtime.blocks.len) return .non_attention_blocks_present;
        if (self.rmsnorm_function == null or
            self.rope_function == null or
            self.vector_add_function == null or
            self.softmax_rows_function == null)
        {
            return .missing_core_kernels;
        }
        if (self.loaded.config.residual_multiplier != 1.0 and self.vector_add_scaled_function == null) {
            return .missing_core_kernels;
        }
        if (self.loaded.config.use_gelu) {
            if (self.gelu_mul_function == null) return .missing_activation_kernel;
        } else if (self.silu_mul_function == null) {
            return .missing_activation_kernel;
        }

        if (kv_cache_dtype_fp16) {
            if (self.cast_f32_to_f16_function == null or
                self.attn_scores_heads_f16_kv_function == null or
                self.attn_weighted_sum_heads_f16_kv_function == null)
            {
                return .missing_fp16_kv_kernels;
            }
        } else {
            if (self.copy_function == null or
                self.attn_scores_heads_f32_function == null or
                self.attn_weighted_sum_heads_f32_function == null)
            {
                return .missing_f32_kv_kernels;
            }
        }

        for (self.block_runtime.blocks) |layer| {
            const block = layer.attention_mlp orelse return .non_attention_blocks_present;
            const program = layer.program orelse return .unsupported_layer_program;
            if (!sequencePrefillProgramCompatible(program)) return .unsupported_layer_program;
            if (!self.linearWeightSupportsSequenceRows(&block.q_proj)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.k_proj)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.v_proj)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.o_proj)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.w1)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.w2)) return .unsupported_linear_projection;
            if (!self.linearWeightSupportsSequenceRows(&block.w3)) return .unsupported_linear_projection;
        }
        return .ok;
    }

    fn linearWeightSupportsSequenceRows(self: *const CudaBackend, weight: *const LinearWeight) bool {
        return linearWeightSupportsSequenceRowsForKernels(
            weight,
            self.matmul_f16_function != null,
            self.matmul_bf16_function != null,
            self.gaffine_u4_matvec_function != null,
        );
    }

    fn linearWeightSupportsSequenceRowsForKernels(
        weight: *const LinearWeight,
        matmul_f16_available: bool,
        matmul_bf16_available: bool,
        gaffine_matvec_available: bool,
    ) bool {
        return switch (weight.*) {
            .dense_f32 => true,
            .dense_u16 => |w| switch (w.dtype) {
                .f16 => matmul_f16_available,
                .bf16 => matmul_bf16_available,
            },
            .gaffine_u4 => gaffine_matvec_available,
        };
    }

    fn trySequenceFusedQkvRows(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        rows: usize,
        block: *const AttentionMlpBlockRuntime,
        q_out: *compute.cuda.Buffer,
        k_out: *compute.cuda.Buffer,
        v_out: *compute.cuda.Buffer,
    ) !bool {
        if (rows == 0) return error.InvalidArgument;
        if (rows != 1 and !self.gaffine_sequence_fused_qkv_supported) return false;

        const fused_kernel = self.gaffine_u4_matvec_qkv_function orelse return false;
        const q = switch (block.q_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const k = switch (block.k_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const v = switch (block.v_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        if (q.rows != self.d_model or k.rows != self.d_model or v.rows != self.d_model) return false;
        if (q.scales_dtype_tag != k.scales_dtype_tag or q.scales_dtype_tag != v.scales_dtype_tag) return false;
        if (q.cols > std.math.maxInt(u32) or
            k.cols > std.math.maxInt(u32) or
            v.cols > std.math.maxInt(u32) or
            q.rows > std.math.maxInt(u32))
        {
            return false;
        }

        try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &q.packed_data,
            &q.scales,
            &q.biases,
            q_out,
            @intCast(q.cols),
            q.group_size,
            q.scales_dtype_tag,
            &k.packed_data,
            &k.scales,
            &k.biases,
            k_out,
            @intCast(k.cols),
            k.group_size,
            k.scales_dtype_tag,
            &v.packed_data,
            &v.scales,
            &v.biases,
            v_out,
            @intCast(v.cols),
            v.group_size,
            v.scales_dtype_tag,
            @intCast(q.rows),
            @intCast(rows),
        );
        return true;
    }

    fn trySequenceFusedGateUpRows(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        rows: usize,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
        gate_out: *compute.cuda.Buffer,
        up_out: *compute.cuda.Buffer,
    ) !bool {
        if (rows == 0) return error.InvalidArgument;
        if (rows != 1 and !self.gaffine_sequence_fused_gate_up_supported) return false;

        const fused_kernel = self.gaffine_u4_matvec_gate_up_function orelse return false;
        const gate = switch (gate_weight.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const up = switch (up_weight.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        if (gate.rows != self.d_model or up.rows != self.d_model) return false;
        if (gate.scales_dtype_tag != up.scales_dtype_tag) return false;
        if (gate.cols > std.math.maxInt(u32) or
            up.cols > std.math.maxInt(u32) or
            gate.rows > std.math.maxInt(u32))
        {
            return false;
        }

        try compute.cuda.gaffine_u4_matvec_gate_up.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &gate.packed_data,
            &gate.scales,
            &gate.biases,
            gate_out,
            @intCast(gate.cols),
            gate.group_size,
            gate.scales_dtype_tag,
            &up.packed_data,
            &up.scales,
            &up.biases,
            up_out,
            @intCast(up.cols),
            up.group_size,
            up.scales_dtype_tag,
            @intCast(gate.rows),
            @intCast(rows),
        );
        return true;
    }

    pub fn decodeBatch(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        results: []contract.DecodeResult,
    ) !void {
        return decode_mod.decodeBatch(self, requests, results);
    }

    pub fn selectNextTokenFromDeviceLogitsImpl(self: *CudaBackend) !u32 {
        return selectNextTokenFromDeviceLogits(self);
    }

    pub fn shouldDownloadPrefillLogitsImpl(self: *const CudaBackend, token_index: usize, token_count: usize) bool {
        _ = self;
        return shouldDownloadPrefillLogits(token_index, token_count);
    }

    pub fn beginPrefillDispatchWindow(self: *CudaBackend) void {
        @memcpy(self.prefill_dispatch_window_start[0..], self.layer_program_dispatch_total[0..]);
    }

    pub fn logPrefillTimingImpl(self: *const CudaBackend, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
        logPrefillTiming(self, mode, token_count, elapsed_ns);
    }

    pub fn computeGpuPrototypeLogits(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        return self.computeGpuPrototypeLogitsWithLayerLimit(
            token,
            position,
            logits_out,
            self.block_runtime.blocks.len,
            true,
            true,
            true,
            null,
            null,
            null,
        );
    }

    pub fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *CudaBackend,
        token: u32,
        position: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        hidden_override: ?[]const f32,
        deepstack_layer_features_opt: ?[]const []const f32,
        deepstack_feature_index_opt: ?usize,
    ) !void {
        if (!compute_logits and download_logits) return error.InvalidArgument;
        if (deepstack_feature_index_opt != null and deepstack_layer_features_opt == null) return error.InvalidArgument;
        if (download_logits) {
            const logits_out = logits_out_opt orelse return error.InvalidArgument;
            if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        }
        if (position >= self.max_seq_len) return error.InvalidArgument;
        if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;
        if (position == 0 and self.block_runtime.shortconv_block_count > 0) {
            try self.resetShortConvStates();
        }
        if (ensure_kv_capacity) {
            try self.ensureKvCapacity(position + 1);
        }

        const rmsnorm_function = self.rmsnorm_function orelse return error.CudaKernelUnavailable;
        const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
        if (self.loaded.config.residual_multiplier != 1.0 and self.vector_add_scaled_function == null) {
            return error.CudaKernelUnavailable;
        }
        if (self.loaded.config.use_gelu) {
            if (self.gelu_mul_function == null) return error.CudaKernelUnavailable;
        } else {
            if (self.silu_mul_function == null) return error.CudaKernelUnavailable;
        }
        const shortconv_step_function = self.shortconv_step_function orelse return error.CudaKernelUnavailable;
        const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
        const embedding_lookup_f32_function = self.embedding_lookup_f32_function;
        const embedding_lookup_u16_function = self.embedding_lookup_u16_function;
        const embedding_lookup_gaffine_u4_function = self.embedding_lookup_gaffine_u4_function;
        const cast_f32_to_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
        else
            null;
        const kv_write_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            self.kv_write_f16_function
        else
            null;
        const rope_store_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            self.rope_store_f16_function
        else
            null;
        const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
        const attn_scores_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable);
        const attn_scores_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null;
        const attn_fused_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            self.attn_fused_heads_f16_kv_function
        else
            null;
        const softmax_rows_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.softmax_rows_function orelse return error.CudaKernelUnavailable)
        else
            self.softmax_rows_function;
        const attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable);
        const attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null;
        const d_model_u32: u32 = @intCast(self.d_model);
        const head_dim_u32: u32 = @intCast(self.head_dim);
        const rope_dim_u32: u32 = @intCast(self.rope_dim);
        const n_heads_u32: u32 = @intCast(self.n_heads);
        const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
        const seq_len_u32: u32 = @intCast(position + 1);
        const position_u32: u32 = @intCast(position);
        const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
        const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
            self.loaded.config.rope_local_theta
        else
            global_rope_theta;

        if (hidden_override) |hidden| {
            if (hidden.len != self.d_model) return error.InvalidArgument;
            @memcpy(self.prototype.hidden_host, hidden);
            try self.prototype.input_dev.upload(&self.device, std.mem.sliceAsBytes(self.prototype.hidden_host));
        } else {
            var used_device_lookup = false;
            if (enable_device_embedding_lookup and self.prototype.embedding_lookup != null) {
                const lookup = &self.prototype.embedding_lookup.?;
                switch (lookup.kind) {
                    .f32 => {
                        if (embedding_lookup_f32_function) |kernel| {
                            try compute.cuda.embedding_lookup_f32.runWithFunction(
                                &self.kernel_arg_pack,
                                &self.device,
                                kernel,
                                &self.prototype.input_dev,
                                &lookup.buffer,
                                lookup.dim0,
                                lookup.dim1,
                                lookup.hidden_dim,
                                token,
                                lookup.layout_tag,
                                lookup.multiplier,
                            );
                            used_device_lookup = true;
                        }
                    },
                    .f16 => {
                        if (embedding_lookup_u16_function) |kernel| {
                            try compute.cuda.embedding_lookup_u16.runWithFunction(
                                &self.kernel_arg_pack,
                                &self.device,
                                kernel,
                                &self.prototype.input_dev,
                                &lookup.buffer,
                                lookup.dim0,
                                lookup.dim1,
                                lookup.hidden_dim,
                                token,
                                lookup.layout_tag,
                                compute.cuda.embedding_lookup_u16.dtype_f16,
                                lookup.multiplier,
                            );
                            used_device_lookup = true;
                        }
                    },
                    .bf16 => {
                        if (embedding_lookup_u16_function) |kernel| {
                            try compute.cuda.embedding_lookup_u16.runWithFunction(
                                &self.kernel_arg_pack,
                                &self.device,
                                kernel,
                                &self.prototype.input_dev,
                                &lookup.buffer,
                                lookup.dim0,
                                lookup.dim1,
                                lookup.hidden_dim,
                                token,
                                lookup.layout_tag,
                                compute.cuda.embedding_lookup_u16.dtype_bf16,
                                lookup.multiplier,
                            );
                            used_device_lookup = true;
                        }
                    },
                    .gaffine_u4 => {
                        if (embedding_lookup_gaffine_u4_function) |kernel| {
                            if (lookup.scales) |*scales_buf| {
                                if (lookup.biases) |*biases_buf| {
                                    try compute.cuda.embedding_lookup_gaffine_u4.runWithFunction(
                                        &self.kernel_arg_pack,
                                        &self.device,
                                        kernel,
                                        &self.prototype.input_dev,
                                        &lookup.buffer,
                                        scales_buf,
                                        biases_buf,
                                        lookup.dim0,
                                        lookup.hidden_dim,
                                        token,
                                        lookup.group_size,
                                        lookup.scales_dtype_tag,
                                        lookup.multiplier,
                                    );
                                    used_device_lookup = true;
                                }
                            }
                        }
                    },
                }
            }
            if (!used_device_lookup) {
                const used_model_embeddings = tryPopulateHiddenFromToken(self.loaded, token, self.prototype.hidden_host) catch |err| switch (err) {
                    error.InvalidArgument => return error.InvalidArgument,
                    else => return err,
                };
                if (!used_model_embeddings) {
                    log.warn("inference", "CUDA embedding layout unsupported", .{
                        .token = token,
                        .embed_shape_0 = self.loaded.token_embeddings.shape[0],
                        .embed_shape_1 = self.loaded.token_embeddings.shape[1],
                        .embed_dtype = @tagName(self.loaded.token_embeddings.dtype),
                        .embed_ndim = self.loaded.token_embeddings.n_dims,
                    });
                    return error.UnsupportedModel;
                }
                if (self.loaded.config.embedding_multiplier != 1.0) {
                    for (self.prototype.hidden_host) |*v| {
                        v.* *= self.loaded.config.embedding_multiplier;
                    }
                }
                try self.prototype.input_dev.upload(&self.device, std.mem.sliceAsBytes(self.prototype.hidden_host));
            }
        }

        var layer_idx: usize = 0;
        while (layer_idx < layer_limit) : (layer_idx += 1) {
            const layer = &self.block_runtime.blocks[layer_idx];
            const attention_kernels = AttentionKernelSet{
                .attn_scores_heads_f32_function = attn_scores_heads_f32_function,
                .attn_weighted_sum_heads_f32_function = attn_weighted_sum_heads_f32_function,
                .attn_scores_heads_f16_kv_function = attn_scores_heads_f16_kv_function,
                .softmax_rows_function = softmax_rows_function,
                .attn_weighted_sum_heads_f16_kv_function = attn_weighted_sum_heads_f16_kv_function,
                .attn_fused_heads_f16_kv_function = attn_fused_heads_f16_kv_function,
            };
            if (layer.program) |program| {
                try self.tryExecuteLayerProgram(
                    layer,
                    program,
                    d_model_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    n_heads_u32,
                    n_kv_heads_u32,
                    seq_len_u32,
                    position,
                    position_u32,
                    global_rope_theta,
                    local_rope_theta,
                    rope_function,
                    copy_function,
                    cast_f32_to_f16_function,
                    kv_write_f16_function,
                    rope_store_f16_function,
                    shortconv_step_function,
                    attention_kernels,
                );
            } else {
                log.warn("inference", "CUDA layer missing LayerOp program", .{ .layer = layer_idx });
                return error.UnsupportedModel;
            }
            if (deepstack_layer_features_opt) |deepstack_layer_features| {
                if (deepstack_feature_index_opt) |deepstack_feature_index| {
                    if (layer_idx < deepstack_layer_features.len) {
                        const layer_features = deepstack_layer_features[layer_idx];
                        const feature_rows = std.math.divExact(usize, layer_features.len, self.d_model) catch return error.InvalidArgument;
                        if (deepstack_feature_index >= feature_rows) return error.InvalidArgument;
                        const row_start = std.math.mul(usize, deepstack_feature_index, self.d_model) catch return error.InvalidArgument;
                        const feature_row = layer_features[row_start .. row_start + self.d_model];
                        try self.prototype.deepstack_add_dev.upload(&self.device, std.mem.sliceAsBytes(feature_row));
                        try compute.cuda.vector_add.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            vector_add_function,
                            &self.prototype.input_dev,
                            &self.prototype.deepstack_add_dev,
                            &self.prototype.input_dev,
                            d_model_u32,
                        );
                    }
                }
            }
        }

        if (!compute_logits) return;

        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            rmsnorm_function,
            &self.prototype.input_dev,
            &self.prototype.norm_weight_dev,
            &self.prototype.norm_out_dev,
            1,
            d_model_u32,
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );

        try self.linearForward(&self.prototype.norm_out_dev, &self.prototype.projection_weight, &self.prototype.logits_dev);
        if (!download_logits) return;

        const logits_out = logits_out_opt.?;
        try self.prototype.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.prototype.projected_logits_host));

        if (self.prototype.projected_vocab == logits_out.len) {
            @memcpy(logits_out, self.prototype.projected_logits_host);
        } else {
            @memset(logits_out, -1.0e9);
            @memcpy(logits_out[0..self.prototype.projected_vocab], self.prototype.projected_logits_host);
        }
        if (self.loaded.config.logits_scaling != 1.0) {
            for (logits_out) |*v| {
                v.* /= self.loaded.config.logits_scaling;
            }
        }
    }

    pub fn ensureKvCapacity(self: *CudaBackend, required_tokens: usize) !void {
        if (required_tokens == 0) return;
        if (required_tokens > self.max_seq_len) return error.InvalidArgument;
        const copy_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            null
        else
            (self.copy_function orelse return error.CudaKernelUnavailable);
        const copy_u16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.copy_u16_function orelse return error.CudaKernelUnavailable)
        else
            null;

        for (self.block_runtime.blocks) |*layer| {
            const block = if (layer.attention_mlp) |*b| b else continue;
            if (required_tokens <= block.kv_capacity) continue;

            var new_capacity = block.kv_capacity;
            if (new_capacity == 0) new_capacity = 1;
            while (new_capacity < required_tokens) {
                const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
                const next = if (doubled > new_capacity) doubled else self.max_seq_len;
                new_capacity = @min(self.max_seq_len, next);
                if (new_capacity == self.max_seq_len) break;
            }
            if (new_capacity < required_tokens) return error.InvalidArgument;

            const new_elems = std.math.mul(usize, new_capacity, block.kv_dim) catch return error.InvalidArgument;
            const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
            const new_bytes = std.math.mul(usize, new_elems, kv_elem_bytes) catch return error.InvalidArgument;
            var new_k_cache = try self.device.allocBuffer(new_bytes);
            errdefer new_k_cache.deinit(&self.device);
            var new_v_cache = try self.device.allocBuffer(new_bytes);
            errdefer new_v_cache.deinit(&self.device);

            if (block.kv_capacity > 0) {
                const old_elems = std.math.mul(usize, block.kv_capacity, block.kv_dim) catch return error.InvalidArgument;
                const old_count_u32: u32 = @intCast(old_elems);
                if (kv_cache_dtype_fp16) {
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function.?,
                        &block.k_cache,
                        &new_k_cache,
                        old_count_u32,
                    );
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function.?,
                        &block.v_cache,
                        &new_v_cache,
                        old_count_u32,
                    );
                } else {
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function.?,
                        &block.k_cache,
                        &new_k_cache,
                        old_count_u32,
                    );
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function.?,
                        &block.v_cache,
                        &new_v_cache,
                        old_count_u32,
                    );
                }
            }

            block.k_cache.deinit(&self.device);
            block.v_cache.deinit(&self.device);
            block.k_cache = new_k_cache;
            block.v_cache = new_v_cache;
            block.kv_capacity = new_capacity;
        }
    }

    fn resetShortConvStates(self: *CudaBackend) !void {
        for (self.block_runtime.blocks) |*layer| {
            const block = if (layer.shortconv) |*b| b else continue;
            const elems = std.math.divExact(usize, block.conv_state.size, @sizeOf(f32)) catch return error.InvalidArgument;
            const zeros = try self.allocator.alloc(f32, elems);
            defer self.allocator.free(zeros);
            @memset(zeros, 0.0);
            try block.conv_state.upload(&self.device, std.mem.sliceAsBytes(zeros));
        }
    }

    fn linearForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        weight: *const LinearWeight,
        out: *compute.cuda.Buffer,
    ) !void {
        return self.linearForwardRows(input, 1, weight, out);
    }

    fn linearForwardRows(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        rows: usize,
        weight: *const LinearWeight,
        out: *compute.cuda.Buffer,
    ) !void {
        if (rows == 0) return error.InvalidArgument;
        switch (weight.*) {
            .dense_f32 => |w| {
                try self.blas.matmulF32(
                    &self.device,
                    input,
                    rows,
                    w.rows,
                    &w.buffer,
                    w.cols,
                    out,
                );
            },
            .dense_u16 => |w| {
                if (rows == 1) {
                    const kernel = switch (w.dtype) {
                        .f16 => self.matvec_f16_function orelse return error.CudaKernelUnavailable,
                        .bf16 => self.matvec_bf16_function orelse return error.CudaKernelUnavailable,
                    };
                    try compute.cuda.matvec_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        input,
                        &w.buffer,
                        out,
                        @intCast(w.rows),
                        @intCast(w.cols),
                    );
                } else {
                    const kernel = switch (w.dtype) {
                        .f16 => self.matmul_f16_function orelse return error.CudaKernelUnavailable,
                        .bf16 => self.matmul_bf16_function orelse return error.CudaKernelUnavailable,
                    };
                    try compute.cuda.matmul_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        input,
                        &w.buffer,
                        out,
                        @intCast(rows),
                        @intCast(w.rows),
                        @intCast(w.cols),
                    );
                }
            },
            .gaffine_u4 => |w| {
                const kernel = self.gaffine_u4_matvec_function orelse return error.CudaKernelUnavailable;
                if (rows == 1 or self.gaffine_sequence_rows_supported) {
                    try compute.cuda.gaffine_u4_matvec.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        input,
                        &w.packed_data,
                        &w.scales,
                        &w.biases,
                        out,
                        @intCast(w.rows),
                        @intCast(w.cols),
                        w.group_size,
                        w.scales_dtype_tag,
                        @intCast(rows),
                    );
                    return;
                }

                const in_row_bytes = std.math.mul(usize, w.rows, @sizeOf(f32)) catch return error.InvalidArgument;
                const out_row_bytes = std.math.mul(usize, w.cols, @sizeOf(f32)) catch return error.InvalidArgument;
                var row_index: usize = 0;
                while (row_index < rows) : (row_index += 1) {
                    const input_offset = std.math.mul(usize, row_index, in_row_bytes) catch return error.InvalidArgument;
                    const out_offset = std.math.mul(usize, row_index, out_row_bytes) catch return error.InvalidArgument;
                    var input_row = try bufferSlice(input, input_offset, in_row_bytes);
                    var out_row = try bufferSlice(out, out_offset, out_row_bytes);
                    try compute.cuda.gaffine_u4_matvec.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        &input_row,
                        &w.packed_data,
                        &w.scales,
                        &w.biases,
                        &out_row,
                        @intCast(w.rows),
                        @intCast(w.cols),
                        w.group_size,
                        w.scales_dtype_tag,
                        1,
                    );
                }
            },
        }
    }

    fn runQkvProjection(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        block: *const AttentionMlpBlockRuntime,
    ) !ProjectionPath {
        if (try self.tryFusedQkvForward(input, block)) return .fused;

        try self.linearForward(input, &block.q_proj, &self.prototype.attn_q_dev);
        try self.linearForward(input, &block.k_proj, &self.prototype.attn_k_dev);
        try self.linearForward(input, &block.v_proj, &self.prototype.attn_v_dev);
        return .unfused;
    }

    fn runGateUpProjection(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        block: *const AttentionMlpBlockRuntime,
    ) !ProjectionPath {
        return self.runGateUpProjectionWithWeights(input, &block.w1, &block.w3);
    }

    fn runGateUpProjectionWithWeights(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
    ) !ProjectionPath {
        if (try self.tryFusedGateUpForward(input, gate_weight, up_weight)) return .fused;

        try self.linearForward(input, gate_weight, &self.prototype.ffn_gate_dev);
        try self.linearForward(input, up_weight, &self.prototype.ffn_up_dev);
        return .unfused;
    }

    fn runFfnActivationMul(self: *CudaBackend, count: u32) !void {
        if (self.loaded.config.use_gelu) {
            const gelu_mul_function = self.gelu_mul_function orelse return error.CudaKernelUnavailable;
            try compute.cuda.gelu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                gelu_mul_function,
                &self.prototype.ffn_gate_dev,
                &self.prototype.ffn_up_dev,
                &self.prototype.ffn_mul_dev,
                count,
            );
            return;
        }

        const silu_mul_function = self.silu_mul_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.silu_mul.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            silu_mul_function,
            &self.prototype.ffn_gate_dev,
            &self.prototype.ffn_up_dev,
            &self.prototype.ffn_mul_dev,
            count,
        );
    }

    fn addResidualWithModelScale(
        self: *CudaBackend,
        residual: *compute.cuda.Buffer,
        branch: *compute.cuda.Buffer,
        count: u32,
    ) !void {
        if (self.loaded.config.residual_multiplier == 1.0) {
            const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
            try compute.cuda.vector_add.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_function,
                residual,
                branch,
                residual,
                count,
            );
            return;
        }

        const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.vector_add_scaled.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            vector_add_scaled_function,
            residual,
            branch,
            residual,
            self.loaded.config.residual_multiplier,
            count,
        );
    }

    fn addResidualWithScale(
        self: *CudaBackend,
        residual: *compute.cuda.Buffer,
        branch: *compute.cuda.Buffer,
        count: u32,
        scale: layer_ops.ResidualScale,
    ) !void {
        switch (scale) {
            .residual_multiplier => return self.addResidualWithModelScale(residual, branch, count),
            .one => {
                const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
                return compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    residual,
                    branch,
                    residual,
                    count,
                );
            },
            .literal => |literal| {
                if (literal == 1.0) {
                    const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
                    return compute.cuda.vector_add.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        vector_add_function,
                        residual,
                        branch,
                        residual,
                        count,
                    );
                }
                const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
                return compute.cuda.vector_add_scaled.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_scaled_function,
                    residual,
                    branch,
                    residual,
                    literal,
                    count,
                );
            },
        }
    }

    fn programBuffer(
        self: *CudaBackend,
        id: layer_ops.BufferId,
        norm_buf: *compute.cuda.Buffer,
        branch_buf: *compute.cuda.Buffer,
    ) ?*compute.cuda.Buffer {
        return switch (id) {
            .residual => &self.prototype.input_dev,
            .norm_out => norm_buf,
            .branch_out => branch_buf,
            else => null,
        };
    }

    fn finalOutputBuffer(program: []const layer_ops.LayerOp) layer_ops.BufferId {
        return finalProgramOutputBuffer(program);
    }

    fn nextAttentionNormWeight(
        block: *const AttentionMlpBlockRuntime,
        norm_index: *usize,
    ) ?*const DeviceTensor {
        defer norm_index.* += 1;
        return switch (norm_index.*) {
            0 => &block.ln1_weight,
            1 => &block.ln2_weight,
            2 => if (block.pre_ffn_norm_weight != null)
                &block.pre_ffn_norm_weight.?
            else if (block.post_ffn_norm_weight != null)
                &block.post_ffn_norm_weight.?
            else
                null,
            3 => if (block.post_ffn_norm_weight != null) &block.post_ffn_norm_weight.? else null,
            else => null,
        };
    }

    fn nextShortConvNormWeight(
        block: *const ShortConvBlockRuntime,
        norm_index: *usize,
    ) ?*const DeviceTensor {
        defer norm_index.* += 1;
        return switch (norm_index.*) {
            0 => &block.ln1_weight,
            1 => if (block.ln2_weight != null) &block.ln2_weight.? else null,
            else => null,
        };
    }

    fn runAttentionMixerStep(
        self: *CudaBackend,
        block: *const AttentionMlpBlockRuntime,
        input: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        seq_len_u32: u32,
        position: usize,
        position_u32: u32,
        global_rope_theta: f32,
        local_rope_theta: f32,
        rope_function: compute.cuda.Function,
        copy_function: compute.cuda.Function,
        cast_f32_to_f16_function: ?compute.cuda.Function,
        kv_write_f16_function: ?compute.cuda.Function,
        rope_store_f16_function: ?compute.cuda.Function,
        attention_kernels: AttentionKernelSet,
    ) !void {
        const layer_rope_theta = if (block.sliding_window > 0) local_rope_theta else global_rope_theta;
        _ = try self.runQkvProjection(input, block);

        if (block.q_norm_weight) |*q_norm| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                &self.prototype.attn_q_dev,
                &q_norm.buffer,
                &self.prototype.attn_q_dev,
                n_heads_u32,
                head_dim_u32,
                self.norm_eps,
                self.loaded.runtime.qk_norm_weight_offset,
            );
        }
        if (block.k_norm_weight) |*k_norm| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                &self.prototype.attn_k_dev,
                &k_norm.buffer,
                &self.prototype.attn_k_dev,
                n_kv_heads_u32,
                head_dim_u32,
                self.norm_eps,
                self.loaded.runtime.qk_norm_weight_offset,
            );
        }

        const use_fused_attention_heads_f16_kv = attention_mod.useFusedHeadsF16Kv(
            attention_policy_config,
            seq_len_u32,
            block.sliding_window,
            block.is_causal,
            head_dim_u32,
            attention_kernels.attn_fused_heads_f16_kv_function != null,
        );
        if (!use_fused_attention_heads_f16_kv) {
            try compute.cuda.rope.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rope_function,
                &self.prototype.attn_q_dev,
                n_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            );
        }
        const use_k_write_fused = kv_cache_dtype_fp16 and (kv_write_f16_function != null or rope_store_f16_function != null);
        if (!use_k_write_fused) {
            try compute.cuda.rope.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rope_function,
                &self.prototype.attn_k_dev,
                n_kv_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            );
        }

        const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
        const kv_row_bytes = std.math.mul(usize, block.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;
        const kv_row_offset = std.math.mul(usize, position, kv_row_bytes) catch return error.InvalidArgument;
        var k_row = try bufferSlice(&block.k_cache, kv_row_offset, kv_row_bytes);
        var v_row = try bufferSlice(&block.v_cache, kv_row_offset, kv_row_bytes);
        if (kv_cache_dtype_fp16) {
            if (kv_write_f16_function) |kv_write_f16| {
                try compute.cuda.kv_write_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kv_write_f16,
                    &self.prototype.attn_k_dev,
                    &self.prototype.attn_v_dev,
                    &k_row,
                    &v_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            } else if (rope_store_f16_function) |rope_store_f16| {
                try compute.cuda.rope_store_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_store_f16,
                    &self.prototype.attn_k_dev,
                    &k_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &self.prototype.attn_v_dev,
                    &v_row,
                    @intCast(block.kv_dim),
                );
            } else {
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &self.prototype.attn_k_dev,
                    &k_row,
                    @intCast(block.kv_dim),
                );
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &self.prototype.attn_v_dev,
                    &v_row,
                    @intCast(block.kv_dim),
                );
            }
        } else {
            try compute.cuda.copy.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                copy_function,
                &self.prototype.attn_k_dev,
                &k_row,
                @intCast(block.kv_dim),
            );
            try compute.cuda.copy.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                copy_function,
                &self.prototype.attn_v_dev,
                &v_row,
                @intCast(block.kv_dim),
            );
        }

        const kv_groups = self.n_heads / self.n_kv_heads;
        const kv_groups_u32: u32 = @intCast(kv_groups);
        const kv_dim_u32: u32 = @intCast(block.kv_dim);
        _ = try self.runAttentionContext(
            block,
            attention_kernels,
            seq_len_u32,
            head_dim_u32,
            kv_dim_u32,
            kv_groups_u32,
            rope_dim_u32,
            position_u32,
            layer_rope_theta,
        );
        try self.linearForward(&self.prototype.attn_context_dev, &block.o_proj, output);
        _ = d_model_u32;
    }

    fn runShortConvMixerStep(
        self: *CudaBackend,
        block: *ShortConvBlockRuntime,
        input: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        shortconv_step_function: compute.cuda.Function,
    ) !void {
        try self.linearForward(input, &block.in_proj, &self.prototype.shortconv_proj_dev);
        const conv_bytes = std.math.mul(usize, block.conv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        var b_gate = try bufferSlice(&self.prototype.shortconv_proj_dev, 0, conv_bytes);
        var c_gate = try bufferSlice(&self.prototype.shortconv_proj_dev, conv_bytes, conv_bytes);
        var x_proj = try bufferSlice(&self.prototype.shortconv_proj_dev, conv_bytes * 2, conv_bytes);

        try compute.cuda.shortconv_step.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            shortconv_step_function,
            &b_gate,
            &c_gate,
            &x_proj,
            &block.conv_state,
            &block.conv_weight_time_major.buffer,
            if (block.conv_bias) |*w| &w.buffer else null,
            &self.prototype.shortconv_conv_dev,
            @intCast(block.conv_dim),
            @intCast(block.d_conv),
        );
        try self.linearForward(&self.prototype.shortconv_conv_dev, &block.out_proj, output);
    }

    fn runFfnStep(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
        down_weight: *const LinearWeight,
        d_ff: u32,
        output: *compute.cuda.Buffer,
    ) !void {
        _ = try self.runGateUpProjectionWithWeights(input, gate_weight, up_weight);
        try self.runFfnActivationMul(d_ff);
        try self.linearForward(&self.prototype.ffn_mul_dev, down_weight, output);
    }

    const LayerProgramExecutionContext = struct {
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        seq_len_u32: u32,
        position: usize,
        position_u32: u32,
        global_rope_theta: f32,
        local_rope_theta: f32,
        rope_function: compute.cuda.Function,
        copy_function: compute.cuda.Function,
        cast_f32_to_f16_function: ?compute.cuda.Function,
        kv_write_f16_function: ?compute.cuda.Function,
        rope_store_f16_function: ?compute.cuda.Function,
        shortconv_step_function: compute.cuda.Function,
        attention_kernels: AttentionKernelSet,
        norm_buf: *compute.cuda.Buffer,
        branch_buf: *compute.cuda.Buffer,
        norm_index: usize,
    };

    const LayerProgramAdapterFn = *const fn (
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) anyerror!void;

    const layer_program_required_opcodes = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .shortconv,
        .swiglu,
        .moe,
        .mamba_mixer,
        .residual_add,
    };

    const layer_program_adapter_table: [256]?LayerProgramAdapterFn = blk: {
        var table: [256]?LayerProgramAdapterFn = [_]?LayerProgramAdapterFn{null} ** 256;
        table[@intFromEnum(opcode_map.Opcode.rmsnorm)] = layerProgramNormAdapter;
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramAttentionAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramShortConvAdapter;
        table[@intFromEnum(opcode_map.Opcode.swiglu)] = layerProgramFfnAdapter;
        table[@intFromEnum(opcode_map.Opcode.moe)] = layerProgramFfnAdapter;
        table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] = layerProgramMambaAdapter;
        table[@intFromEnum(opcode_map.Opcode.residual_add)] = layerProgramResidualAddAdapter;
        break :blk table;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            layer_program_adapter_table,
            layer_program_required_opcodes,
            "cuda.engine.layer_program_adapter_table",
        );
    }

    fn layerProgramAdapterForOpcode(opcode: opcode_map.Opcode) ?LayerProgramAdapterFn {
        return layer_program_adapter_table[@intFromEnum(opcode)];
    }

    fn recordLayerProgramDispatch(self: *CudaBackend, opcode: opcode_map.Opcode) void {
        const opcode_idx = @intFromEnum(opcode);
        self.layer_program_dispatch_total[opcode_idx] +%= 1;
    }

    fn prefillDispatchDelta(self: *const CudaBackend, opcode: opcode_map.Opcode) u64 {
        const opcode_idx = @intFromEnum(opcode);
        return self.layer_program_dispatch_total[opcode_idx] - self.prefill_dispatch_window_start[opcode_idx];
    }

    fn prefillDispatchTotal(self: *const CudaBackend) u64 {
        var total: u64 = 0;
        for (0..self.layer_program_dispatch_total.len) |idx| {
            total += self.layer_program_dispatch_total[idx] - self.prefill_dispatch_window_start[idx];
        }
        return total;
    }

    fn selectCoreKernelOutput(self: *CudaBackend, out_id: layer_ops.BufferId) !*compute.cuda.Buffer {
        return switch (out_id) {
            .norm_out => &self.prototype.norm_out_dev,
            .branch_out => &self.prototype.attn_out_dev,
            else => error.UnsupportedModel,
        };
    }

    fn selectFfnKernelOutput(self: *CudaBackend, out_id: layer_ops.BufferId) !*compute.cuda.Buffer {
        return switch (out_id) {
            .norm_out => &self.prototype.norm_out_dev,
            .branch_out => &self.prototype.ffn_down_dev,
            else => error.UnsupportedModel,
        };
    }

    fn updateProgramOutputBinding(
        ctx: *LayerProgramExecutionContext,
        out_id: layer_ops.BufferId,
        output: *compute.cuda.Buffer,
    ) !void {
        switch (out_id) {
            .norm_out => ctx.norm_buf = output,
            .branch_out => ctx.branch_buf = output,
            else => return error.UnsupportedModel,
        }
    }

    fn layerProgramNormAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidArgument,
        };
        if (kernel_op.debug_type != .norm) return error.InvalidArgument;

        const input = self.programBuffer(kernel_op.in, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        const output = try self.selectCoreKernelOutput(kernel_op.out);

        if (layer.attention_mlp) |*block| {
            const weight = nextAttentionNormWeight(block, &ctx.norm_index) orelse return error.UnsupportedModel;
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                input,
                &weight.buffer,
                output,
                1,
                ctx.d_model_u32,
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
        } else if (layer.shortconv) |*block| {
            const weight = nextShortConvNormWeight(block, &ctx.norm_index) orelse return error.UnsupportedModel;
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                input,
                &weight.buffer,
                output,
                1,
                ctx.d_model_u32,
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
        } else return error.UnsupportedModel;

        try updateProgramOutputBinding(ctx, kernel_op.out, output);
    }

    fn layerProgramAttentionAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidArgument,
        };
        if (kernel_op.debug_type != .multihead_attention) return error.InvalidArgument;

        const input = self.programBuffer(kernel_op.in, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        const output = try self.selectCoreKernelOutput(kernel_op.out);

        if (layer.attention_mlp) |*block| {
            try self.runAttentionMixerStep(
                block,
                input,
                output,
                ctx.d_model_u32,
                ctx.head_dim_u32,
                ctx.rope_dim_u32,
                ctx.n_heads_u32,
                ctx.n_kv_heads_u32,
                ctx.seq_len_u32,
                ctx.position,
                ctx.position_u32,
                ctx.global_rope_theta,
                ctx.local_rope_theta,
                ctx.rope_function,
                ctx.copy_function,
                ctx.cast_f32_to_f16_function,
                ctx.kv_write_f16_function,
                ctx.rope_store_f16_function,
                ctx.attention_kernels,
            );
        } else return error.UnsupportedModel;

        try updateProgramOutputBinding(ctx, kernel_op.out, output);
    }

    fn layerProgramShortConvAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidArgument,
        };
        if (kernel_op.debug_type != .shortconv) return error.InvalidArgument;

        const input = self.programBuffer(kernel_op.in, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        const output = try self.selectCoreKernelOutput(kernel_op.out);

        if (layer.shortconv) |*block| {
            try self.runShortConvMixerStep(block, input, output, ctx.shortconv_step_function);
        } else return error.UnsupportedModel;

        try updateProgramOutputBinding(ctx, kernel_op.out, output);
    }

    fn layerProgramFfnAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidArgument,
        };
        if (kernel_op.debug_type != .mlp and kernel_op.debug_type != .moe) return error.InvalidArgument;

        const input = self.programBuffer(kernel_op.in, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        const output = try self.selectFfnKernelOutput(kernel_op.out);

        if (layer.attention_mlp) |*block| {
            try self.runFfnStep(
                input,
                &block.w1,
                &block.w3,
                &block.w2,
                @intCast(block.d_ff),
                output,
            );
        } else if (layer.shortconv) |*block| {
            const w1 = block.ffn_w1 orelse return error.UnsupportedModel;
            const w2 = block.ffn_w2 orelse return error.UnsupportedModel;
            const w3 = block.ffn_w3 orelse return error.UnsupportedModel;
            try self.runFfnStep(
                input,
                &w1,
                &w3,
                &w2,
                @intCast(block.d_ff),
                output,
            );
        } else return error.UnsupportedModel;

        try updateProgramOutputBinding(ctx, kernel_op.out, output);
    }

    fn layerProgramMambaAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.InvalidArgument,
        };
        if (kernel_op.debug_type != .mamba_mixer) return error.InvalidArgument;

        const input = self.programBuffer(kernel_op.in, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        const output = try self.selectCoreKernelOutput(kernel_op.out);

        if (layer.shortconv) |*block| {
            try self.runShortConvMixerStep(block, input, output, ctx.shortconv_step_function);
        } else return error.UnsupportedModel;

        try updateProgramOutputBinding(ctx, kernel_op.out, output);
    }

    fn layerProgramResidualAddAdapter(
        self: *CudaBackend,
        _: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const add_op = switch (op) {
            .add => |add| add,
            else => return error.InvalidArgument,
        };
        const branch = self.programBuffer(add_op.branch, ctx.norm_buf, ctx.branch_buf) orelse return error.UnsupportedModel;
        try self.addResidualWithScale(&self.prototype.input_dev, branch, ctx.d_model_u32, add_op.scale);
    }

    fn dispatchLayerProgramOp(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        op: layer_ops.LayerOp,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const opcode = opcode_map.opcodeForLayerOp(op);
        const adapter = layerProgramAdapterForOpcode(opcode) orelse return error.UnsupportedModel;
        self.recordLayerProgramDispatch(opcode);
        try adapter(self, layer, op, ctx);
    }

    fn tryExecuteLayerProgram(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        program: []const layer_ops.LayerOp,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        seq_len_u32: u32,
        position: usize,
        position_u32: u32,
        global_rope_theta: f32,
        local_rope_theta: f32,
        rope_function: compute.cuda.Function,
        copy_function: compute.cuda.Function,
        cast_f32_to_f16_function: ?compute.cuda.Function,
        kv_write_f16_function: ?compute.cuda.Function,
        rope_store_f16_function: ?compute.cuda.Function,
        shortconv_step_function: compute.cuda.Function,
        attention_kernels: AttentionKernelSet,
    ) !void {
        var exec_ctx = LayerProgramExecutionContext{
            .d_model_u32 = d_model_u32,
            .head_dim_u32 = head_dim_u32,
            .rope_dim_u32 = rope_dim_u32,
            .n_heads_u32 = n_heads_u32,
            .n_kv_heads_u32 = n_kv_heads_u32,
            .seq_len_u32 = seq_len_u32,
            .position = position,
            .position_u32 = position_u32,
            .global_rope_theta = global_rope_theta,
            .local_rope_theta = local_rope_theta,
            .rope_function = rope_function,
            .copy_function = copy_function,
            .cast_f32_to_f16_function = cast_f32_to_f16_function,
            .kv_write_f16_function = kv_write_f16_function,
            .rope_store_f16_function = rope_store_f16_function,
            .shortconv_step_function = shortconv_step_function,
            .attention_kernels = attention_kernels,
            .norm_buf = &self.prototype.norm_out_dev,
            .branch_buf = &self.prototype.attn_out_dev,
            .norm_index = 0,
        };

        for (program) |op| {
            try self.dispatchLayerProgramOp(layer, op, &exec_ctx);
        }

        const final_buf_id = finalOutputBuffer(program);
        switch (final_buf_id) {
            .residual => {},
            .norm_out, .branch_out => {
                const final_buf = self.programBuffer(final_buf_id, exec_ctx.norm_buf, exec_ctx.branch_buf) orelse return error.UnsupportedModel;
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function,
                    final_buf,
                    &self.prototype.input_dev,
                    d_model_u32,
                );
            },
            else => return error.UnsupportedModel,
        }
    }

    fn runAttentionContext(
        self: *CudaBackend,
        block: *const AttentionMlpBlockRuntime,
        kernels: AttentionKernelSet,
        seq_len_u32: u32,
        head_dim_u32: u32,
        kv_dim_u32: u32,
        kv_groups_u32: u32,
        rope_dim_u32: u32,
        position_u32: u32,
        theta: f32,
    ) !AttentionPath {
        var effective_seq_len_u32 = seq_len_u32;
        var k_cache_view = block.k_cache;
        var v_cache_view = block.v_cache;

        if (block.sliding_window > 0 and block.is_causal) {
            const window_u32 = std.math.cast(u32, block.sliding_window) orelse std.math.maxInt(u32);
            if (effective_seq_len_u32 > window_u32) {
                const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
                const row_bytes = std.math.mul(usize, @as(usize, kv_dim_u32), kv_elem_bytes) catch return error.InvalidArgument;
                const start_row = effective_seq_len_u32 - window_u32;
                const start_offset = std.math.mul(usize, @as(usize, start_row), row_bytes) catch return error.InvalidArgument;
                k_cache_view = try bufferSlice(&block.k_cache, start_offset, block.k_cache.size - start_offset);
                v_cache_view = try bufferSlice(&block.v_cache, start_offset, block.v_cache.size - start_offset);
                effective_seq_len_u32 = window_u32;
            }
        }

        if (kv_cache_dtype_fp16) {
            if (attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                block.sliding_window,
                block.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_f16_kv_function != null,
            )) {
                try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_f16_kv_function.?,
                    &self.prototype.attn_q_dev,
                    &k_cache_view,
                    &v_cache_view,
                    &self.prototype.attn_context_dev,
                    @intCast(self.n_heads),
                    effective_seq_len_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    head_dim_u32,
                    self.attention_scale,
                    rope_dim_u32,
                    position_u32,
                    theta,
                );
                return .fused_heads_f16_kv;
            }

            const attn_scores_dev = try self.prototype.requireAttentionScoresDev();
            const attn_probs_dev = try self.prototype.requireAttentionProbsDev();
            try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                &self.prototype.attn_q_dev,
                &k_cache_view,
                attn_scores_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                self.attention_scale,
            );
            try compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                &self.prototype.attn_context_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            );
            return .heads_f16_kv;
        }

        const attn_scores_heads_f32_function = kernels.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable;
        const attn_weighted_sum_heads_f32_function = kernels.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable;
        const attn_scores_dev = try self.prototype.requireAttentionScoresDev();
        const attn_probs_dev = try self.prototype.requireAttentionProbsDev();

        try compute.cuda.attn_scores_heads_f32.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            attn_scores_heads_f32_function,
            &self.prototype.attn_q_dev,
            &k_cache_view,
            attn_scores_dev,
            @intCast(self.n_heads),
            effective_seq_len_u32,
            kv_dim_u32,
            kv_groups_u32,
            head_dim_u32,
            self.attention_scale,
        );
        try compute.cuda.softmax_rows.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
            attn_scores_dev,
            attn_probs_dev,
            @intCast(self.n_heads),
            effective_seq_len_u32,
        );
        try compute.cuda.attn_weighted_sum_heads_f32.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            attn_weighted_sum_heads_f32_function,
            attn_probs_dev,
            &v_cache_view,
            &self.prototype.attn_context_dev,
            @intCast(self.n_heads),
            effective_seq_len_u32,
            kv_dim_u32,
            kv_groups_u32,
            head_dim_u32,
        );
        return .heads_f32_kv;
    }

    fn tryFusedQkvForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        block: *const AttentionMlpBlockRuntime,
    ) !bool {
        if (try self.tryFusedDenseU16QkvForward(input, block)) return true;

        const fused_kernel = self.gaffine_u4_matvec_qkv_function orelse return false;
        const q = switch (block.q_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const k = switch (block.k_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const v = switch (block.v_proj) {
            .gaffine_u4 => |w| w,
            else => return false,
        };

        if (q.rows != self.d_model or k.rows != self.d_model or v.rows != self.d_model) return false;
        if (q.scales_dtype_tag != k.scales_dtype_tag or q.scales_dtype_tag != v.scales_dtype_tag) return false;
        if (q.cols > std.math.maxInt(u32) or
            k.cols > std.math.maxInt(u32) or
            v.cols > std.math.maxInt(u32) or
            q.rows > std.math.maxInt(u32))
        {
            return false;
        }

        try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &q.packed_data,
            &q.scales,
            &q.biases,
            &self.prototype.attn_q_dev,
            @intCast(q.cols),
            q.group_size,
            q.scales_dtype_tag,
            &k.packed_data,
            &k.scales,
            &k.biases,
            &self.prototype.attn_k_dev,
            @intCast(k.cols),
            k.group_size,
            k.scales_dtype_tag,
            &v.packed_data,
            &v.scales,
            &v.biases,
            &self.prototype.attn_v_dev,
            @intCast(v.cols),
            v.group_size,
            v.scales_dtype_tag,
            @intCast(q.rows),
            1,
        );
        return true;
    }

    fn tryFusedDenseU16QkvForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        block: *const AttentionMlpBlockRuntime,
    ) !bool {
        const q = switch (block.q_proj) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const k = switch (block.k_proj) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const v = switch (block.v_proj) {
            .dense_u16 => |w| w,
            else => return false,
        };
        if (!canFuseDenseU16QkvWeights(self.d_model, q, k, v)) return false;

        const fused_kernel = switch (q.dtype) {
            .f16 => self.matvec_qkv_f16_function orelse return false,
            .bf16 => self.matvec_qkv_bf16_function orelse return false,
        };
        try compute.cuda.matvec_u16_qkv.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &q.buffer,
            &self.prototype.attn_q_dev,
            @intCast(q.cols),
            &k.buffer,
            &self.prototype.attn_k_dev,
            @intCast(k.cols),
            &v.buffer,
            &self.prototype.attn_v_dev,
            @intCast(v.cols),
            @intCast(q.rows),
        );
        return true;
    }

    fn canFuseDenseU16QkvWeights(d_model: usize, q: U16LinearWeight, k: U16LinearWeight, v: U16LinearWeight) bool {
        if (q.rows != d_model or k.rows != d_model or v.rows != d_model) return false;
        if (q.dtype != k.dtype or q.dtype != v.dtype) return false;
        if (q.cols > std.math.maxInt(u32) or
            k.cols > std.math.maxInt(u32) or
            v.cols > std.math.maxInt(u32) or
            q.rows > std.math.maxInt(u32))
        {
            return false;
        }
        return true;
    }

    fn tryFusedGateUpForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
    ) !bool {
        if (try self.tryFusedDenseU16GateUpForward(input, gate_weight, up_weight)) return true;

        const fused_kernel = self.gaffine_u4_matvec_gate_up_function orelse return false;
        const gate = switch (gate_weight.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const up = switch (up_weight.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };

        if (gate.rows != self.d_model or up.rows != self.d_model) return false;
        if (gate.scales_dtype_tag != up.scales_dtype_tag) return false;
        if (gate.cols > std.math.maxInt(u32) or
            up.cols > std.math.maxInt(u32) or
            gate.rows > std.math.maxInt(u32))
        {
            return false;
        }

        try compute.cuda.gaffine_u4_matvec_gate_up.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &gate.packed_data,
            &gate.scales,
            &gate.biases,
            &self.prototype.ffn_gate_dev,
            @intCast(gate.cols),
            gate.group_size,
            gate.scales_dtype_tag,
            &up.packed_data,
            &up.scales,
            &up.biases,
            &self.prototype.ffn_up_dev,
            @intCast(up.cols),
            up.group_size,
            up.scales_dtype_tag,
            @intCast(gate.rows),
            1,
        );
        return true;
    }

    fn tryFusedDenseU16GateUpForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
    ) !bool {
        const gate = switch (gate_weight.*) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const up = switch (up_weight.*) {
            .dense_u16 => |w| w,
            else => return false,
        };
        if (gate.rows != self.d_model or up.rows != self.d_model) return false;
        if (gate.dtype != up.dtype) return false;
        if (gate.cols > std.math.maxInt(u32) or
            up.cols > std.math.maxInt(u32) or
            gate.rows > std.math.maxInt(u32))
        {
            return false;
        }

        const fused_kernel = switch (gate.dtype) {
            .f16 => self.matvec_gate_up_f16_function orelse return false,
            .bf16 => self.matvec_gate_up_bf16_function orelse return false,
        };
        try compute.cuda.matvec_u16_gate_up.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &gate.buffer,
            &self.prototype.ffn_gate_dev,
            @intCast(gate.cols),
            &up.buffer,
            &self.prototype.ffn_up_dev,
            @intCast(up.cols),
            @intCast(gate.rows),
        );
        return true;
    }

    fn initKernelFunctions(self: *CudaBackend) !void {
        if (!self.device.supportsModuleLaunch()) return;

        try self.kernel_registry.loadEmbeddedModule(compute.cuda.vector_add.embedded_module);
        const sideload_loaded = self.tryLoadSideloadModule() catch |err| blk: {
            log.warn("inference", "CUDA sideload unavailable; using embedded PTX", .{
                .reason = @errorName(err),
            });
            break :blk false;
        };
        if (sideload_loaded) {
            log.info("inference", "CUDA sideload kernel module active", .{});
        }

        try self.resolveRequiredKernels();
    }

    fn resolveRequiredKernels(self: *CudaBackend) !void {
        for (required_kernels) |kernel| {
            const resolved = self.kernel_registry.resolveFunction(
                kernel.op_name,
                kernel.embedded_symbol,
            ) catch |err| {
                log.warn("inference", "CUDA kernel resolve failed", .{
                    .op = kernel.op_name,
                    .symbol = kernel.embedded_symbol,
                    .reason = @errorName(err),
                });
                return err;
            };
            self.assignResolvedKernel(kernel.slot, resolved);
        }
    }

    fn assignResolvedKernel(
        self: *CudaBackend,
        slot: KernelSlot,
        resolved: compute.cuda.registry.ResolvedFunction,
    ) void {
        switch (slot) {
            .vector_add => {
                self.vector_add_function = resolved.function;
                self.vector_add_source = resolved.source;
            },
            .vector_add_scaled => {
                self.vector_add_scaled_function = resolved.function;
                self.vector_add_scaled_source = resolved.source;
            },
            .mul => {
                self.mul_function = resolved.function;
                self.mul_source = resolved.source;
            },
            .copy => {
                self.copy_function = resolved.function;
                self.copy_source = resolved.source;
            },
            .copy_u16 => {
                self.copy_u16_function = resolved.function;
                self.copy_u16_source = resolved.source;
            },
            .cast_f32_to_f16 => {
                self.cast_f32_to_f16_function = resolved.function;
                self.cast_f32_to_f16_source = resolved.source;
            },
            .embedding_lookup_f32 => {
                self.embedding_lookup_f32_function = resolved.function;
                self.embedding_lookup_f32_source = resolved.source;
            },
            .embedding_lookup_u16 => {
                self.embedding_lookup_u16_function = resolved.function;
                self.embedding_lookup_u16_source = resolved.source;
            },
            .embedding_lookup_gaffine_u4 => {
                self.embedding_lookup_gaffine_u4_function = resolved.function;
                self.embedding_lookup_gaffine_u4_source = resolved.source;
            },
            .kv_write_f16 => {
                self.kv_write_f16_function = resolved.function;
                self.kv_write_f16_source = resolved.source;
            },
            .rmsnorm => {
                self.rmsnorm_function = resolved.function;
                self.rmsnorm_source = resolved.source;
            },
            .rope => {
                self.rope_function = resolved.function;
                self.rope_source = resolved.source;
            },
            .rope_store_f16 => {
                self.rope_store_f16_function = resolved.function;
                self.rope_store_f16_source = resolved.source;
            },
            .attn_scores_heads_f32 => {
                self.attn_scores_heads_f32_function = resolved.function;
                self.attn_scores_heads_f32_source = resolved.source;
            },
            .attn_scores_heads_f16_kv => {
                self.attn_scores_heads_f16_kv_function = resolved.function;
                self.attn_scores_heads_f16_kv_source = resolved.source;
            },
            .attn_fused_heads_f16_kv => {
                self.attn_fused_heads_f16_kv_function = resolved.function;
                self.attn_fused_heads_f16_kv_source = resolved.source;
            },
            .softmax_rows => {
                self.softmax_rows_function = resolved.function;
                self.softmax_rows_source = resolved.source;
            },
            .attn_weighted_sum_heads_f32 => {
                self.attn_weighted_sum_heads_f32_function = resolved.function;
                self.attn_weighted_sum_heads_f32_source = resolved.source;
            },
            .attn_weighted_sum_heads_f16_kv => {
                self.attn_weighted_sum_heads_f16_kv_function = resolved.function;
                self.attn_weighted_sum_heads_f16_kv_source = resolved.source;
            },
            .silu => {
                self.silu_function = resolved.function;
                self.silu_source = resolved.source;
            },
            .silu_mul => {
                self.silu_mul_function = resolved.function;
                self.silu_mul_source = resolved.source;
            },
            .gelu_mul => {
                self.gelu_mul_function = resolved.function;
                self.gelu_mul_source = resolved.source;
            },
            .shortconv_step => {
                self.shortconv_step_function = resolved.function;
                self.shortconv_step_source = resolved.source;
            },
            .argmax => {
                self.argmax_function = resolved.function;
                self.argmax_source = resolved.source;
            },
            .matmul_f16 => {
                self.matmul_f16_function = resolved.function;
                self.matmul_f16_source = resolved.source;
            },
            .matmul_bf16 => {
                self.matmul_bf16_function = resolved.function;
                self.matmul_bf16_source = resolved.source;
            },
            .matvec_f16 => {
                self.matvec_f16_function = resolved.function;
                self.matvec_f16_source = resolved.source;
            },
            .matvec_bf16 => {
                self.matvec_bf16_function = resolved.function;
                self.matvec_bf16_source = resolved.source;
            },
            .matvec_gate_up_f16 => {
                self.matvec_gate_up_f16_function = resolved.function;
                self.matvec_gate_up_f16_source = resolved.source;
            },
            .matvec_gate_up_bf16 => {
                self.matvec_gate_up_bf16_function = resolved.function;
                self.matvec_gate_up_bf16_source = resolved.source;
            },
            .matvec_qkv_f16 => {
                self.matvec_qkv_f16_function = resolved.function;
                self.matvec_qkv_f16_source = resolved.source;
            },
            .matvec_qkv_bf16 => {
                self.matvec_qkv_bf16_function = resolved.function;
                self.matvec_qkv_bf16_source = resolved.source;
            },
            .gaffine_u4_matvec => {
                self.gaffine_u4_matvec_function = resolved.function;
                self.gaffine_u4_matvec_source = resolved.source;
            },
            .gaffine_u4_matvec_gate_up => {
                self.gaffine_u4_matvec_gate_up_function = resolved.function;
                self.gaffine_u4_matvec_gate_up_source = resolved.source;
            },
            .gaffine_u4_matvec_qkv => {
                self.gaffine_u4_matvec_qkv_function = resolved.function;
                self.gaffine_u4_matvec_qkv_source = resolved.source;
            },
        }
    }

    fn tryLoadSideloadModule(self: *CudaBackend) !bool {
        const base_url_raw = std.process.getEnvVarOwned(self.allocator, compute.cuda.sideload.kernel_base_url_env) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => return false,
            else => return err,
        };
        defer self.allocator.free(base_url_raw);
        const base_url = std.mem.trim(u8, base_url_raw, " \t\r\n");
        if (base_url.len == 0) return false;

        const capability = self.device.computeCapability() catch |err| switch (err) {
            error.CudaQueryUnavailable => return false,
            else => return err,
        };
        const arch = try compute.cuda.sideload.archTag(self.allocator, capability.major, capability.minor);
        defer self.allocator.free(arch);

        const cache_dir = try compute.cuda.sideload.resolveCacheDir(self.allocator);
        defer self.allocator.free(cache_dir);
        try compute.cuda.sideload.ensureCacheDir(cache_dir);

        const manifest_bytes = try compute.cuda.sideload.loadOrFetchManifest(
            self.allocator,
            cache_dir,
            arch,
            base_url,
        );
        defer self.allocator.free(manifest_bytes);
        var parsed_manifest = try compute.cuda.manifest.parse(self.allocator, manifest_bytes);
        defer parsed_manifest.deinit();
        try compute.cuda.manifest.ensureCompatible(
            parsed_manifest.manifest,
            arch,
            compute.cuda.manifest.kernel_abi_version,
        );

        const artifact_bytes = try compute.cuda.sideload.loadOrFetchArtifact(
            self.allocator,
            cache_dir,
            arch,
            base_url,
            parsed_manifest.manifest.sha256,
        );
        defer self.allocator.free(artifact_bytes);

        try self.kernel_registry.loadSideloadModule(
            manifest_bytes,
            artifact_bytes,
            arch,
            compute.cuda.manifest.kernel_abi_version,
        );
        log.info("inference", "CUDA sideload payload loaded", .{
            .arch = arch,
            .cache_dir = cache_dir,
        });
        return true;
    }
};

fn argmaxHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v > best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

fn argminHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v < best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

fn bytesToMiB(bytes: usize) f32 {
    return @as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn populatePrefillHiddenFromTokens(
    loaded: *const LoadedModel,
    tokens: []const u32,
    d_model: usize,
    out: []f32,
) !void {
    if (d_model == 0) return error.InvalidArgument;
    const expected = std.math.mul(usize, tokens.len, d_model) catch return error.InvalidArgument;
    if (out.len != expected) return error.InvalidArgument;

    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const row_start = std.math.mul(usize, idx, d_model) catch return error.InvalidArgument;
        const row = out[row_start .. row_start + d_model];
        const used_model_embeddings = tryPopulateHiddenFromToken(loaded, tokens[idx], row) catch |err| switch (err) {
            error.InvalidArgument => return error.InvalidArgument,
            else => return err,
        };
        if (!used_model_embeddings) return error.UnsupportedModel;
        if (loaded.config.embedding_multiplier != 1.0) {
            for (row) |*value| value.* *= loaded.config.embedding_multiplier;
        }
    }
}

fn selectNextTokenFromLogits(self: *CudaBackend, logits: []const f32) !u32 {
    if (logits.len != self.vocab_size) return error.InvalidArgument;
    return if (self.loaded.config.logits_scaling < 0.0) argminHost(logits) else argmaxHost(logits);
}

fn selectNextTokenFromDeviceLogits(self: *CudaBackend) !u32 {
    if (self.prototype.projected_vocab == 0) return error.InvalidArgument;
    if (self.prototype.projected_vocab > std.math.maxInt(u32)) return error.InvalidArgument;
    if (self.loaded.config.logits_scaling < 0.0) {
        try self.prototype.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.prototype.projected_logits_host));
        return argminHost(self.prototype.projected_logits_host);
    }

    const argmax_function = self.argmax_function orelse return error.CudaKernelUnavailable;
    const count_u32: u32 = @intCast(self.prototype.projected_vocab);
    try compute.cuda.argmax.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        argmax_function,
        &self.prototype.logits_dev,
        &self.argmax_index_dev,
        count_u32,
    );
    var token: u32 = 0;
    try self.argmax_index_dev.download(&self.device, std.mem.asBytes(&token));
    return token;
}

fn bufferSlice(buffer: *const compute.cuda.Buffer, byte_offset: usize, byte_len: usize) !compute.cuda.Buffer {
    if (byte_offset > buffer.size) return error.InvalidArgument;
    const end = std.math.add(usize, byte_offset, byte_len) catch return error.InvalidArgument;
    if (end > buffer.size) return error.InvalidArgument;
    const ptr = std.math.add(u64, buffer.pointer, @intCast(byte_offset)) catch return error.InvalidArgument;
    return .{
        .pointer = ptr,
        .size = byte_len,
    };
}

fn freeOwnedTensorView(allocator: std.mem.Allocator, t: Tensor) void {
    if (t.data_ptr) |ptr| {
        const aligned_ptr: [*]align(32) u8 = @alignCast(ptr);
        allocator.free(aligned_ptr[0..t.data_size]);
    }
}

fn shouldDownloadPrefillLogits(token_index: usize, token_count: usize) bool {
    std.debug.assert(token_count > 0);
    return token_index + 1 == token_count;
}

fn logPrefillTiming(self: *const CudaBackend, mode: []const u8, token_count: usize, elapsed_ns: u64) void {
    const elapsed_ms: f64 = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const tok_per_s: f64 = if (elapsed_ns == 0)
        0.0
    else
        (@as(f64, @floatFromInt(token_count)) * 1_000_000_000.0) / @as(f64, @floatFromInt(elapsed_ns));
    const dispatches = self.prefillDispatchTotal();
    const dispatches_per_token: f64 = if (token_count == 0)
        0.0
    else
        @as(f64, @floatFromInt(dispatches)) / @as(f64, @floatFromInt(token_count));
    log.info("inference", "CUDA prefill timing", .{
        .mode = mode,
        .tokens = token_count,
        .elapsed_ms = elapsed_ms,
        .tok_per_s = tok_per_s,
        .layer_program_dispatches = dispatches,
        .layer_program_dispatches_per_token = dispatches_per_token,
        .layer_program_rmsnorm = self.prefillDispatchDelta(.rmsnorm),
        .layer_program_attention = self.prefillDispatchDelta(.multihead_attention),
        .layer_program_shortconv = self.prefillDispatchDelta(.shortconv),
        .layer_program_ffn = self.prefillDispatchDelta(.swiglu) + self.prefillDispatchDelta(.moe),
        .layer_program_mamba = self.prefillDispatchDelta(.mamba_mixer),
        .layer_program_residual_add = self.prefillDispatchDelta(.residual_add),
        .layers = self.block_runtime.blocks.len,
        .attention_blocks = self.block_runtime.attention_block_count,
        .shortconv_blocks = self.block_runtime.shortconv_block_count,
    });
}

fn collectTokenPositions(
    allocator: std.mem.Allocator,
    token_ids: []const u32,
    needle: u32,
) ![]usize {
    var count: usize = 0;
    for (token_ids) |token| {
        if (token == needle) count += 1;
    }
    if (count == 0) return &.{};

    const positions = try allocator.alloc(usize, count);
    errdefer allocator.free(positions);

    var write_idx: usize = 0;
    for (token_ids, 0..) |token, idx| {
        if (token != needle) continue;
        positions[write_idx] = idx;
        write_idx += 1;
    }
    std.debug.assert(write_idx == count);
    return positions;
}

fn findPositionIndex(positions: []const usize, position: usize) ?usize {
    for (positions, 0..) |value, idx| {
        if (value == position) return idx;
    }
    return null;
}

const DenseLinearLayout = struct {
    in_dim: usize,
    out_dim: usize,
    needs_transpose: bool,
};

fn resolveDenseInOutLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Canonical checkpoint layout is [out_dim, in_dim].
    // Prefer this branch first so square matrices (rows == cols == input_dim)
    // are treated as [out,in] and transposed to the kernel layout [in,out].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = true,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = false,
        };
    }
    return error.UnsupportedModel;
}

fn resolveDenseOutInLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Typed (f16/bf16) path follows loader policy: canonical [out_dim, in_dim].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = false,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = true,
        };
    }
    return error.UnsupportedModel;
}

fn transposeRowMajor(
    comptime T: type,
    allocator: std.mem.Allocator,
    src: []align(1) const T,
    rows: usize,
    cols: usize,
) ![]T {
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    if (src.len < logical_count) return error.InvalidArgument;
    const out = try allocator.alloc(T, logical_count);
    errdefer allocator.free(out);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    return out;
}

fn uploadTensor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
) !DeviceTensor {
    if (src.n_dims < 1 or src.n_dims > 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = if (src.n_dims == 1) 1 else blk: {
        if (src.shape[1] <= 0) return error.InvalidArgument;
        break :blk @intCast(src.shape[1]);
    };

    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);
    var buffer = try device.allocBuffer(host_f32.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(host_f32));

    return .{
        .rows = rows,
        .cols = cols,
        .buffer = buffer,
    };
}

fn uploadLinearWeight(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.dtype == .grouped_affine_u4) {
        return uploadLinearWeightGroupedAffineU4(device, src, input_dim);
    }
    return uploadLinearWeightDense(device, allocator, src, input_dim);
}

const DenseOutInU16 = struct {
    values: []align(1) const u16,
    owned: ?[]u16 = null,

    fn deinit(self: *DenseOutInU16, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

const DenseOutInF32 = struct {
    values: []const f32,
    owned: ?[]f32 = null,

    fn deinit(self: *DenseOutInF32, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

const FusedQkvUpload = struct {
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
};

const FusedGateUpUpload = struct {
    gate: LinearWeight,
    up: LinearWeight,
};

fn materializeDenseOutInU16(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInU16 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSliceUnaligned(u16);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(u16, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
}

fn materializeDenseOutInF32(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInF32 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSlice(f32);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(f32, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
}

fn uploadFusedQkvWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    fused_qkv: *const Tensor,
    input_dim: usize,
    q_out: usize,
    kv_out: usize,
) !FusedQkvUpload {
    const total_out = std.math.add(usize, q_out, std.math.mul(usize, kv_out, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
    if (fused_qkv.dtype == .f16 or fused_qkv.dtype == .bf16) {
        var out_in = try materializeDenseOutInU16(allocator, fused_qkv, input_dim, total_out);
        defer out_in.deinit(allocator);

        const q_count = std.math.mul(usize, q_out, input_dim) catch return error.InvalidArgument;
        const kv_count = std.math.mul(usize, kv_out, input_dim) catch return error.InvalidArgument;
        const expected = std.math.add(usize, q_count, std.math.mul(usize, kv_count, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
        if (out_in.values.len != expected) return error.InvalidArgument;

        const q_vals = out_in.values[0..q_count];
        const k_vals = out_in.values[q_count .. q_count + kv_count];
        const v_vals = out_in.values[q_count + kv_count .. q_count + kv_count + kv_count];
        const q_bytes = std.mem.sliceAsBytes(q_vals);
        const k_bytes = std.mem.sliceAsBytes(k_vals);
        const v_bytes = std.mem.sliceAsBytes(v_vals);
        var q_tensor = Tensor.view(@constCast(q_bytes.ptr), &.{ q_out, input_dim }, fused_qkv.dtype, q_bytes.len);
        var k_tensor = Tensor.view(@constCast(k_bytes.ptr), &.{ kv_out, input_dim }, fused_qkv.dtype, k_bytes.len);
        var v_tensor = Tensor.view(@constCast(v_bytes.ptr), &.{ kv_out, input_dim }, fused_qkv.dtype, v_bytes.len);
        const q = try uploadLinearWeight(device, allocator, &q_tensor, input_dim);
        errdefer {
            var q_mut = q;
            q_mut.deinit(device);
        }
        const k = try uploadLinearWeight(device, allocator, &k_tensor, input_dim);
        errdefer {
            var k_mut = k;
            k_mut.deinit(device);
        }
        const v = try uploadLinearWeight(device, allocator, &v_tensor, input_dim);
        return .{ .q = q, .k = k, .v = v };
    }
    if (fused_qkv.dtype == .f32) {
        var out_in = try materializeDenseOutInF32(allocator, fused_qkv, input_dim, total_out);
        defer out_in.deinit(allocator);

        const q_count = std.math.mul(usize, q_out, input_dim) catch return error.InvalidArgument;
        const kv_count = std.math.mul(usize, kv_out, input_dim) catch return error.InvalidArgument;
        const expected = std.math.add(usize, q_count, std.math.mul(usize, kv_count, 2) catch return error.InvalidArgument) catch return error.InvalidArgument;
        if (out_in.values.len != expected) return error.InvalidArgument;

        const q_vals = out_in.values[0..q_count];
        const k_vals = out_in.values[q_count .. q_count + kv_count];
        const v_vals = out_in.values[q_count + kv_count .. q_count + kv_count + kv_count];
        const q_bytes = std.mem.sliceAsBytes(q_vals);
        const k_bytes = std.mem.sliceAsBytes(k_vals);
        const v_bytes = std.mem.sliceAsBytes(v_vals);
        var q_tensor = Tensor.view(@constCast(q_bytes.ptr), &.{ q_out, input_dim }, .f32, q_bytes.len);
        var k_tensor = Tensor.view(@constCast(k_bytes.ptr), &.{ kv_out, input_dim }, .f32, k_bytes.len);
        var v_tensor = Tensor.view(@constCast(v_bytes.ptr), &.{ kv_out, input_dim }, .f32, v_bytes.len);
        const q = try uploadLinearWeight(device, allocator, &q_tensor, input_dim);
        errdefer {
            var q_mut = q;
            q_mut.deinit(device);
        }
        const k = try uploadLinearWeight(device, allocator, &k_tensor, input_dim);
        errdefer {
            var k_mut = k;
            k_mut.deinit(device);
        }
        const v = try uploadLinearWeight(device, allocator, &v_tensor, input_dim);
        return .{ .q = q, .k = k, .v = v };
    }
    return error.UnsupportedModel;
}

fn uploadFusedGateUpWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    fused_gate_up: *const Tensor,
    input_dim: usize,
    layout: GateUpLayout,
) !FusedGateUpUpload {
    if (fused_gate_up.n_dims != 2) return error.UnsupportedModel;
    if (fused_gate_up.shape[0] <= 0 or fused_gate_up.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(fused_gate_up.shape[0]);
    const cols: usize = @intCast(fused_gate_up.shape[1]);
    const out_dim = if (rows == input_dim) cols else if (cols == input_dim) rows else return error.UnsupportedModel;
    if ((out_dim % 2) != 0) return error.InvalidArgument;
    const d_ff = out_dim / 2;

    if (fused_gate_up.dtype == .f16 or fused_gate_up.dtype == .bf16) {
        var out_in = try materializeDenseOutInU16(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []align(1) const u16 = undefined;
        var up_vals: []align(1) const u16 = undefined;
        var gate_owned: ?[]u16 = null;
        var up_owned: ?[]u16 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    if (fused_gate_up.dtype == .f32) {
        var out_in = try materializeDenseOutInF32(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []const f32 = undefined;
        var up_vals: []const f32 = undefined;
        var gate_owned: ?[]f32 = null;
        var up_owned: ?[]f32 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, .f32, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, .f32, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    return error.UnsupportedModel;
}

fn uploadLinearWeightDense(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.dtype == .f16 or src.dtype == .bf16) {
        return uploadLinearWeightDenseU16(device, allocator, src, input_dim);
    }
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);

    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);

    const layout = resolveDenseInOutLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA dense linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []f32 = host_f32;
    var transposed: ?[]f32 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(f32, allocator, host_f32, rows, cols);
        oriented = transposed.?;
    }

    var buffer = try device.allocBuffer(oriented.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    return .{
        .dense_f32 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
        },
    };
}

fn uploadLinearWeightDenseU16(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;

    const host_u16 = src.asSliceUnaligned(u16);
    if (host_u16.len < logical_count) return error.InvalidArgument;
    const view = host_u16[0..logical_count];

    const layout = resolveDenseOutInLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA u16 linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []align(1) const u16 = view;
    var transposed: ?[]u16 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(u16, allocator, view, rows, cols);
        oriented = transposed.?;
    }

    const bytes = std.math.mul(usize, oriented.len, @sizeOf(u16)) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    const dense_dtype: DenseU16Dtype = switch (src.dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => return error.UnsupportedModel,
    };

    return .{
        .dense_u16 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
            .dtype = dense_dtype,
        },
    };
}

fn uploadLinearWeightGroupedAffineU4(
    device: *compute.cuda.Device,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    if (src.data_ptr == null) return error.InvalidArgument;
    const gaffine = src.gaffine orelse return error.UnsupportedModel;
    const out_dim: usize = @intCast(src.shape[0]);
    const in_dim: usize = @intCast(src.shape[1]);

    if (in_dim != input_dim) {
        log.warn("inference", "CUDA grouped-affine U4 orientation unsupported", .{
            .rows = out_dim,
            .cols = in_dim,
            .input_dim = input_dim,
        });
        return error.UnsupportedModel;
    }
    if (in_dim == 0 or out_dim == 0) return error.InvalidArgument;
    if ((in_dim % 8) != 0) return error.UnsupportedModel;
    if (gaffine.group_size == 0 or (in_dim % gaffine.group_size) != 0 or (gaffine.group_size % 8) != 0) {
        return error.UnsupportedModel;
    }

    const packed_words_per_row = in_dim / 8;
    const groups_per_row = in_dim / gaffine.group_size;
    const packed_words = std.math.mul(usize, out_dim, packed_words_per_row) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, packed_words, @sizeOf(u32)) catch return error.InvalidArgument;
    const sb_count = std.math.mul(usize, out_dim, groups_per_row) catch return error.InvalidArgument;
    const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
    if (src.data_size < packed_bytes) return error.InvalidArgument;
    if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;

    const scales_dtype_tag: u32 = switch (gaffine.scales_dtype) {
        .f16 => gaffine_scales_dtype_f16,
        .bf16 => gaffine_scales_dtype_bf16,
        else => return error.UnsupportedModel,
    };

    var packed_dev = try device.allocBuffer(packed_bytes);
    errdefer packed_dev.deinit(device);
    var scales_dev = try device.allocBuffer(sb_bytes);
    errdefer scales_dev.deinit(device);
    var biases_dev = try device.allocBuffer(sb_bytes);
    errdefer biases_dev.deinit(device);

    const packed_host = src.data()[0..packed_bytes];
    try packed_dev.upload(device, packed_host);
    try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
    try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

    return .{
        .gaffine_u4 = .{
            .rows = in_dim,
            .cols = out_dim,
            .packed_data = packed_dev,
            .scales = scales_dev,
            .biases = biases_dev,
            .group_size = @intCast(gaffine.group_size),
            .scales_dtype_tag = scales_dtype_tag,
        },
    };
}

fn uploadVectorTensor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    expected_len: usize,
) !DeviceTensor {
    if (expected_len == 0) return error.InvalidArgument;
    const values = try materializeTensorF32(allocator, src);
    defer allocator.free(values);
    if (values.len != expected_len) return error.UnsupportedModel;

    var buffer = try device.allocBuffer(values.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(values));
    return .{
        .rows = expected_len,
        .cols = 1,
        .buffer = buffer,
    };
}

fn allocZeroedF32Buffer(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    count: usize,
) !compute.cuda.Buffer {
    if (count == 0) return error.InvalidArgument;
    const zeros = try allocator.alloc(f32, count);
    defer allocator.free(zeros);
    @memset(zeros, 0.0);
    var buffer = try device.allocBuffer(count * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(zeros));
    return buffer;
}

fn tryUploadEmbeddingLookup(
    device: *compute.cuda.Device,
    loaded: *const LoadedModel,
    d_model: usize,
) !?EmbeddingLookup {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return null;
    if (embeddings.n_dims != 2) return null;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return null;
    const kind: EmbeddingLookupKind = switch (embeddings.dtype) {
        .f32 => .f32,
        .f16 => .f16,
        .bf16 => .bf16,
        .grouped_affine_u4 => .gaffine_u4,
        else => return null,
    };

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    var layout_tag: u32 = undefined;
    var hidden_dim: usize = undefined;
    if (dim1 == d_model) {
        layout_tag = compute.cuda.embedding_lookup_f32.layout_vocab_hidden;
        hidden_dim = dim1;
    } else if (dim0 == d_model) {
        if (kind == .gaffine_u4) return null;
        layout_tag = compute.cuda.embedding_lookup_f32.layout_hidden_vocab;
        hidden_dim = dim0;
    } else {
        return null;
    }
    const dim0_u32 = std.math.cast(u32, dim0) orelse return error.InvalidArgument;
    const dim1_u32 = std.math.cast(u32, dim1) orelse return error.InvalidArgument;
    const hidden_dim_u32 = std.math.cast(u32, hidden_dim) orelse return error.InvalidArgument;

    if (kind == .gaffine_u4) {
        const gaffine = embeddings.gaffine orelse return null;
        if (gaffine.group_size == 0) return null;
        const group_size: usize = gaffine.group_size;
        if ((hidden_dim % group_size) != 0 or (group_size % 8) != 0) return null;
        const groups_per_row = hidden_dim / group_size;
        const packed_words_per_row = hidden_dim / 8;
        const packed_words_total = std.math.mul(usize, dim0, packed_words_per_row) catch return error.InvalidArgument;
        const sb_count = std.math.mul(usize, dim0, groups_per_row) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, packed_words_total, @sizeOf(u32)) catch return error.InvalidArgument;
        const sb_bytes = std.math.mul(usize, sb_count, @sizeOf(u16)) catch return error.InvalidArgument;
        const packed_vals = embeddings.asSliceUnaligned(u32);
        if (packed_vals.len < packed_words_total) return error.InvalidArgument;
        if (gaffine.scales.len < sb_bytes or gaffine.biases.len < sb_bytes) return error.InvalidArgument;
        const scales_dtype_tag = switch (gaffine.scales_dtype) {
            .f16 => gaffine_scales_dtype_f16,
            .bf16 => gaffine_scales_dtype_bf16,
            else => return error.UnsupportedModel,
        };

        var packed_dev = try device.allocBuffer(packed_bytes);
        errdefer packed_dev.deinit(device);
        var scales_dev = try device.allocBuffer(sb_bytes);
        errdefer scales_dev.deinit(device);
        var biases_dev = try device.allocBuffer(sb_bytes);
        errdefer biases_dev.deinit(device);
        try packed_dev.upload(device, std.mem.sliceAsBytes(packed_vals[0..packed_words_total]));
        try scales_dev.upload(device, gaffine.scales[0..sb_bytes]);
        try biases_dev.upload(device, gaffine.biases[0..sb_bytes]);

        return .{
            .kind = .gaffine_u4,
            .dim0 = dim0_u32,
            .dim1 = dim1_u32,
            .hidden_dim = hidden_dim_u32,
            .layout_tag = layout_tag,
            .group_size = std.math.cast(u32, group_size) orelse return error.InvalidArgument,
            .scales_dtype_tag = scales_dtype_tag,
            .scales = scales_dev,
            .biases = biases_dev,
            .multiplier = loaded.config.embedding_multiplier,
            .buffer = packed_dev,
        };
    }

    const elem_count = std.math.mul(usize, dim0, dim1) catch return error.InvalidArgument;
    const elem_bytes: usize = switch (kind) {
        .f32 => @sizeOf(f32),
        .f16, .bf16 => @sizeOf(u16),
        .gaffine_u4 => unreachable,
    };
    const bytes = std.math.mul(usize, elem_count, elem_bytes) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    switch (kind) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            if (src.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src[0..elem_count]));
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            if (src_u16.len < elem_count) return error.InvalidArgument;
            try buffer.upload(device, std.mem.sliceAsBytes(src_u16[0..elem_count]));
        },
        .gaffine_u4 => unreachable,
    }

    return .{
        .kind = kind,
        .dim0 = dim0_u32,
        .dim1 = dim1_u32,
        .hidden_dim = hidden_dim_u32,
        .layout_tag = layout_tag,
        .multiplier = loaded.config.embedding_multiplier,
        .buffer = buffer,
    };
}

fn uploadShortConvWeightTimeMajor(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    conv_dim: usize,
    d_conv: usize,
) !DeviceTensor {
    if (src.n_dims < 2 or src.n_dims > 3) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = blk: {
        if (src.n_dims == 2) break :blk @intCast(src.shape[1]);
        if (src.shape[2] <= 0) return error.InvalidArgument;
        const dim1: usize = @intCast(src.shape[1]);
        const dim2: usize = @intCast(src.shape[2]);
        if (dim1 == 1) break :blk dim2;
        if (dim2 == 1) break :blk dim1;
        log.warn("inference", "CUDA shortconv conv1d 3D layout unsupported", .{
            .shape0 = src.shape[0],
            .shape1 = src.shape[1],
            .shape2 = src.shape[2],
        });
        return error.UnsupportedModel;
    };
    const expected = std.math.mul(usize, conv_dim, d_conv) catch return error.InvalidArgument;
    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);
    if (host_f32.len != expected) return error.InvalidArgument;

    var oriented: []f32 = host_f32;
    var transposed: ?[]f32 = null;
    defer if (transposed) |t| allocator.free(t);

    if (rows == conv_dim and cols == d_conv) {
        // Convert channel-major [conv_dim, d_conv] -> time-major [d_conv, conv_dim].
        transposed = try transposeRowMajor(f32, allocator, host_f32, rows, cols);
        oriented = transposed.?;
    } else if (!(rows == d_conv and cols == conv_dim)) {
        log.warn("inference", "CUDA shortconv conv1d weight shape unsupported", .{
            .rows = rows,
            .cols = cols,
            .conv_dim = conv_dim,
            .d_conv = d_conv,
        });
        return error.UnsupportedModel;
    }

    var buffer = try device.allocBuffer(oriented.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));
    return .{
        .rows = d_conv,
        .cols = conv_dim,
        .buffer = buffer,
    };
}

fn materializeTensorF32(allocator: std.mem.Allocator, src: *const Tensor) ![]f32 {
    if (src.data_ptr == null) return error.InvalidArgument;
    if (src.n_dims < 1 or src.n_dims > 3) return error.UnsupportedModel;
    if (src.shape[0] <= 0) return error.InvalidArgument;
    var logical_count: usize = @intCast(src.shape[0]);
    if (src.n_dims >= 2) {
        if (src.shape[1] <= 0) return error.InvalidArgument;
        logical_count = std.math.mul(usize, logical_count, @as(usize, @intCast(src.shape[1]))) catch return error.InvalidArgument;
    }
    if (src.n_dims >= 3) {
        if (src.shape[2] <= 0) return error.InvalidArgument;
        logical_count = std.math.mul(usize, logical_count, @as(usize, @intCast(src.shape[2]))) catch return error.InvalidArgument;
    }

    const out = try allocator.alloc(f32, logical_count);
    errdefer allocator.free(out);

    switch (src.dtype) {
        .f32 => {
            const values = src.asSlice(f32);
            if (values.len < logical_count) return error.InvalidArgument;
            @memcpy(out, values[0..logical_count]);
        },
        .f16, .bf16 => {
            const values = src.asSliceUnaligned(u16);
            if (values.len < logical_count) return error.InvalidArgument;
            for (out, 0..) |*dst, i| {
                dst.* = if (src.dtype == .f16) dtype.fp16ToF32(values[i]) else dtype.bf16ToF32(values[i]);
            }
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            const dequantized = try load_transforms.convertToF32(allocator, src.*);
            defer freeOwnedTensorView(allocator, dequantized);
            const src_f32 = dequantized.asSlice(f32);
            if (src_f32.len < logical_count) return error.InvalidArgument;
            @memcpy(out, src_f32[0..logical_count]);
        },
        else => return error.UnsupportedModel,
    }

    return out;
}

fn canUseModelEmbeddings(loaded: *const LoadedModel, d_model: usize) bool {
    if (d_model == 0) return false;
    const embeddings = loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;
    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    if (dim0 != d_model and dim1 != d_model) return false;
    return switch (embeddings.dtype) {
        .f32, .f16, .bf16, .grouped_affine_u4, .grouped_affine_u8 => true,
        else => false,
    };
}

fn tryPopulateHiddenFromToken(
    loaded: *const LoadedModel,
    token: u32,
    out: []f32,
) !bool {
    const embeddings = &loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    if (embeddings.n_dims != 2) return false;
    if (embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0) return false;

    const dim0: usize = @intCast(embeddings.shape[0]);
    const dim1: usize = @intCast(embeddings.shape[1]);
    const token_idx: usize = @intCast(token);
    const hidden_dim = out.len;

    switch (embeddings.dtype) {
        .f32 => {
            const src = embeddings.asSlice(f32);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                const row_start = token_idx * dim1;
                @memcpy(out, src[row_start .. row_start + hidden_dim]);
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = src[i * dim1 + token_idx];
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const src_u16 = embeddings.asSliceUnaligned(u16);
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[token_idx * dim1 + i];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    const raw = src_u16[i * dim1 + token_idx];
                    out[i] = if (embeddings.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            // Common layout: [vocab, d_model].
            if (dim1 == hidden_dim) {
                if (token_idx >= dim0) return error.InvalidArgument;
                try decodeGaffineRow(embeddings, token_idx, out);
                return true;
            }

            // Transposed layout: [d_model, vocab].
            if (dim0 == hidden_dim) {
                if (token_idx >= dim1) return error.InvalidArgument;
                var i: usize = 0;
                while (i < hidden_dim) : (i += 1) {
                    out[i] = try gaffineValueAt(embeddings, i, token_idx);
                }
                return true;
            }

            return false;
        },
        else => return false,
    }
}

fn decodeGaffineRow(weight: *const Tensor, row: usize, out: []f32) !void {
    if (weight.dtype != .grouped_affine_u4 and weight.dtype != .grouped_affine_u8) return error.InvalidArgument;
    const gaffine = weight.gaffine orelse return error.InvalidArgument;
    if (weight.n_dims != 2) return error.InvalidArgument;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (row >= rows) return error.InvalidArgument;
    if (out.len != cols) return error.InvalidArgument;

    const values_per_word: usize = if (weight.dtype == .grouped_affine_u4) 8 else 4;
    const bits: u5 = if (weight.dtype == .grouped_affine_u4) 4 else 8;
    const mask: u32 = if (weight.dtype == .grouped_affine_u4) 0xF else 0xFF;
    if (values_per_word == 0 or cols % values_per_word != 0) return error.InvalidArgument;
    if (gaffine.group_size == 0 or cols % gaffine.group_size != 0) return error.InvalidArgument;

    const packed_stride = cols / values_per_word;
    const group_stride = cols / gaffine.group_size;
    if (group_stride == 0) return error.InvalidArgument;
    const packed_words = weight.asSliceUnaligned(u32);
    const required_words = std.math.mul(usize, rows, packed_stride) catch return error.InvalidArgument;
    if (packed_words.len < required_words) return error.InvalidArgument;

    var current_group_idx: usize = std.math.maxInt(usize);
    var current_scale: f32 = 0.0;
    var current_bias: f32 = 0.0;
    var col: usize = 0;
    while (col < cols) : (col += 1) {
        const group_idx = col / gaffine.group_size;
        if (group_idx != current_group_idx) {
            current_group_idx = group_idx;
            const sb_idx = row * group_stride + group_idx;
            current_scale = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.scales, sb_idx);
            current_bias = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.biases, sb_idx);
        }

        const pack_idx = row * packed_stride + (col / values_per_word);
        const packed_word = packed_words[pack_idx];
        const shift: u5 = @intCast((col % values_per_word) * bits);
        const quant = (packed_word >> shift) & mask;
        out[col] = @as(f32, @floatFromInt(quant)) * current_scale + current_bias;
    }
}

fn tryPopulateFinalNormWeight(loaded: *const LoadedModel, out: []f32) bool {
    if (loaded.ln_final) |ln_final| {
        if (ln_final.data_ptr == null or ln_final.numel < out.len) return false;
        switch (ln_final.dtype) {
            .f32 => {
                const src = ln_final.asSlice(f32);
                if (src.len < out.len) return false;
                @memcpy(out, src[0..out.len]);
                return true;
            },
            .f16, .bf16 => {
                const src = ln_final.asSliceUnaligned(u16);
                if (src.len < out.len) return false;
                for (out, 0..) |*v, i| {
                    const raw = src[i];
                    v.* = if (ln_final.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                }
                return true;
            },
            else => return false,
        }
    }
    return false;
}

fn tryPopulateProjectionFromWeight(
    allocator: std.mem.Allocator,
    weight: *const Tensor,
    d_model: usize,
    projected_vocab: usize,
    out: []f32,
) bool {
    if (weight.data_ptr == null) return false;
    if (weight.n_dims != 2) return false;

    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return false;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    switch (weight.dtype) {
        .f32 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src = weight.asSlice(f32);
            if (src.len < expected_len) return false;

            // Direct layout: [d_model, vocab] so each row can be copied contiguously.
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }

            // Transposed layout: [vocab, d_model], so gather one column per token row.
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = src[col * dim1 + row];
                    }
                }
                return true;
            }

            return false;
        },
        .f16, .bf16 => {
            const expected_len = std.math.mul(usize, dim0, dim1) catch return false;
            const src_u16 = weight.asSliceUnaligned(u16);
            if (src_u16.len < expected_len) return false;

            // Direct layout: [d_model, vocab]
            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[row * dim1 + col];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            // Transposed layout: [vocab, d_model]
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        const raw = src_u16[col * dim1 + row];
                        out[row * projected_vocab + col] = if (weight.dtype == .f16) dtype.fp16ToF32(raw) else dtype.bf16ToF32(raw);
                    }
                }
                return true;
            }

            return false;
        },
        .grouped_affine_u4, .grouped_affine_u8 => {
            const dequantized = load_transforms.convertToF32(allocator, weight.*) catch return false;
            defer freeOwnedTensorView(allocator, dequantized);
            const src = dequantized.asSlice(f32);

            if (dim0 == d_model and dim1 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    const src_start = row * dim1;
                    const dst_start = row * projected_vocab;
                    @memcpy(out[dst_start .. dst_start + projected_vocab], src[src_start .. src_start + projected_vocab]);
                }
                return true;
            }
            if (dim1 == d_model and dim0 >= projected_vocab) {
                var row: usize = 0;
                while (row < d_model) : (row += 1) {
                    var col: usize = 0;
                    while (col < projected_vocab) : (col += 1) {
                        out[row * projected_vocab + col] = src[col * dim1 + row];
                    }
                }
                return true;
            }
            return false;
        },
        else => return false,
    }
}

fn gaffineScaleBiasToF32(scales_dtype: tensor.DType, bytes: []const u8, idx: usize) !f32 {
    const byte_offset = std.math.mul(usize, idx, @sizeOf(u16)) catch return error.InvalidArgument;
    if (byte_offset + @sizeOf(u16) > bytes.len) return error.InvalidArgument;

    const v = std.mem.readInt(u16, bytes[byte_offset..][0..2], .little);
    return switch (scales_dtype) {
        .f16 => dtype.fp16ToF32(v),
        .bf16 => dtype.bf16ToF32(v),
        else => error.InvalidArgument,
    };
}

fn gaffineValueAt(weight: *const Tensor, row: usize, col: usize) !f32 {
    if (weight.dtype != .grouped_affine_u4 and weight.dtype != .grouped_affine_u8) return error.InvalidArgument;
    const gaffine = weight.gaffine orelse return error.InvalidArgument;
    if (weight.n_dims != 2) return error.InvalidArgument;
    if (weight.shape[0] <= 0 or weight.shape[1] <= 0) return error.InvalidArgument;

    const rows: usize = @intCast(weight.shape[0]);
    const cols: usize = @intCast(weight.shape[1]);
    if (row >= rows or col >= cols) return error.InvalidArgument;

    const values_per_word: usize = if (weight.dtype == .grouped_affine_u4) 8 else 4;
    const bits: u5 = if (weight.dtype == .grouped_affine_u4) 4 else 8;
    const mask: u32 = if (weight.dtype == .grouped_affine_u4) 0xF else 0xFF;
    if (values_per_word == 0 or cols % values_per_word != 0) return error.InvalidArgument;
    if (gaffine.group_size == 0 or cols % gaffine.group_size != 0) return error.InvalidArgument;

    const packed_stride = cols / values_per_word;
    const group_stride = cols / gaffine.group_size;
    if (group_stride == 0) return error.InvalidArgument;

    const pack_idx = row * packed_stride + (col / values_per_word);
    const pack_byte_offset = std.math.mul(usize, pack_idx, @sizeOf(u32)) catch return error.InvalidArgument;
    if (pack_byte_offset + @sizeOf(u32) > weight.data().len) return error.InvalidArgument;
    const packed_word = std.mem.readInt(u32, weight.data()[pack_byte_offset..][0..4], .little);
    const shift: u5 = @intCast((col % values_per_word) * bits);
    const quant = (packed_word >> shift) & mask;

    const group_idx = col / gaffine.group_size;
    const sb_idx = row * group_stride + group_idx;
    const scale = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.scales, sb_idx);
    const bias = try gaffineScaleBiasToF32(gaffine.scales_dtype, gaffine.biases, sb_idx);

    return @as(f32, @floatFromInt(quant)) * scale + bias;
}

test "resolveDenseInOutLayout keeps [in,out] orientation" {
    const layout = try resolveDenseInOutLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "finalOutputBuffer returns residual when program ends with add" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .multihead_attention,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.residual, CudaBackend.finalOutputBuffer(&program));
}

test "finalOutputBuffer returns kernel output buffer for post-norm endings" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, CudaBackend.finalOutputBuffer(&program));
}

test "validateLayerProgramForCuda accepts kernel-add programs" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
        .{ .kernel = .{
            .id = 1,
            .in = .norm_out,
            .out = .branch_out,
            .debug_type = .mlp,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try validateLayerProgramForCuda(&program, 0, .attention_mlp);
}

test "layerProgramAdapterForOpcode covers CUDA LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .residual_add,
    };
    for (supported) |opcode| {
        try std.testing.expect(CudaBackend.layerProgramAdapterForOpcode(opcode) != null);
    }

    try std.testing.expect(CudaBackend.layerProgramAdapterForOpcode(.mul_scalar) == null);
    try std.testing.expect(CudaBackend.layerProgramAdapterForOpcode(.vision_patch_embed) == null);
}

test "validateLayerProgramForCuda rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    try std.testing.expectError(
        error.UnsupportedModel,
        validateLayerProgramForCuda(&program, 0, .attention_mlp),
    );
}

test "validateLayerProgramForCuda rejects stateful opcode bound to wrong block kind" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectError(
        error.UnsupportedModel,
        validateLayerProgramForCuda(&program, 0, .attention_mlp),
    );
}

test "resolveDenseInOutLayout transposes [out,in] orientation" {
    const layout = try resolveDenseInOutLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseInOutLayout rejects incompatible orientation" {
    try std.testing.expectError(error.UnsupportedModel, resolveDenseInOutLayout(96, 64, 128));
}

test "resolveDenseInOutLayout prefers [out,in] for square matrices" {
    const layout = try resolveDenseInOutLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps [out,in] orientation" {
    const layout = try resolveDenseOutInLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "resolveDenseOutInLayout transposes [in,out] orientation" {
    const layout = try resolveDenseOutInLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps square typed layout untransposed" {
    const layout = try resolveDenseOutInLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "transposeRowMajor transposes compact row-major matrix" {
    const src = [_]u16{
        1, 2, 3,
        4, 5, 6,
    };
    const transposed = try transposeRowMajor(u16, std.testing.allocator, src[0..], 2, 3);
    defer std.testing.allocator.free(transposed);

    const expected = [_]u16{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(u16, expected[0..], transposed);
}

test "required_kernels contract has unique slots and operation names" {
    var seen_slots = std.AutoHashMap(KernelSlot, void).init(std.testing.allocator);
    defer seen_slots.deinit();

    var seen_ops = std.StringHashMap(void).init(std.testing.allocator);
    defer seen_ops.deinit();

    for (required_kernels) |entry| {
        const slot_put = try seen_slots.getOrPut(entry.slot);
        try std.testing.expect(!slot_put.found_existing);

        const op_put = try seen_ops.getOrPut(entry.op_name);
        try std.testing.expect(!op_put.found_existing);

        try std.testing.expect(std.mem.startsWith(u8, entry.embedded_symbol, "talu_"));
        try std.testing.expect(!hasVersionSuffixName(entry.op_name));
        try std.testing.expect(!hasVersionSuffixName(entry.embedded_symbol));
    }

    const slot_count = @typeInfo(KernelSlot).@"enum".fields.len;
    try std.testing.expectEqual(slot_count, required_kernels.len);
}

test "required_kernels keeps heads-based attention path canonical" {
    const required_ops = [_][]const u8{
        compute.cuda.attn_scores_heads_f32.op_name,
        compute.cuda.attn_scores_heads_f16_kv.op_name,
        compute.cuda.attn_weighted_sum_heads_f32.op_name,
        compute.cuda.attn_weighted_sum_heads_f16_kv.op_name,
        compute.cuda.softmax_rows.op_name,
    };
    const legacy_ops = [_][]const u8{
        "attn_scores_f32",
        "attn_scores_f16_kv",
        "attn_weighted_sum_f32",
        "attn_weighted_sum_f16_kv",
        "softmax_f32",
    };

    for (required_ops) |op| {
        var found = false;
        for (required_kernels) |entry| {
            if (std.mem.eql(u8, entry.op_name, op)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    for (legacy_ops) |op| {
        for (required_kernels) |entry| {
            try std.testing.expect(!std.mem.eql(u8, entry.op_name, op));
        }
    }
}

fn hasVersionSuffixName(name: []const u8) bool {
    const marker = "_v";
    const at = std.mem.lastIndexOf(u8, name, marker) orelse return false;
    const digits = name[at + marker.len ..];
    if (digits.len == 0) return false;
    for (digits) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

test "tryPopulateProjectionFromWeight supports [d_model, vocab] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ d_model, 4 }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateProjectionFromWeight supports [vocab, d_model] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0,  9.0,
        10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ 4, d_model }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [vocab, d_model] layout" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [d_model, vocab] layout" {
    var embedding_data = [_]f32{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 3, 2 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens applies embedding multiplier" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(tensor.ModelConfig);
    cfg.embedding_multiplier = 2.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 0 };
    var out = [_]f32{0.0} ** 6;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..]);

    const expected = [_]f32{
        8.0, 10.0, 12.0,
        2.0, 4.0,  6.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "gaffineValueAt decodes grouped_affine_u4 values" {
    var packed_words = [_]u32{
        // 8 packed 4-bit values: 0,1,2,3,4,5,6,7
        0x7654_3210,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    const one_bf16 = dtype.f32ToBf16(1.0);
    const zero_bf16 = dtype.f32ToBf16(0.0);
    var scales_u16 = [_]u16{one_bf16};
    var biases_u16 = [_]u16{zero_bf16};
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u4, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 8,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "tryPopulateFinalNormWeight supports bf16 weights" {
    var norm_u16 = [_]u16{
        dtype.f32ToBf16(1.25),
        dtype.f32ToBf16(-0.5),
    };
    const norm_bytes = std.mem.sliceAsBytes(norm_u16[0..]);
    const norm_tensor = Tensor.view(norm_bytes.ptr, &.{2}, .bf16, norm_bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .ln_final = norm_tensor,
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .bf16,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 2;
    try std.testing.expect(tryPopulateFinalNormWeight(&loaded, out[0..]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.01);
}

test "populatePrefillHiddenFromTokens rejects missing embeddings" {
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(tensor.ModelConfig),
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{0};
    var out = [_]f32{0.0} ** 4;
    try std.testing.expectError(
        error.UnsupportedModel,
        populatePrefillHiddenFromTokens(&loaded, tokens[0..], 4, out[0..]),
    );
}

test "shouldDownloadPrefillLogits only on final token" {
    try std.testing.expect(!shouldDownloadPrefillLogits(0, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(1, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(2, 4));
    try std.testing.expect(shouldDownloadPrefillLogits(3, 4));
}

test "shouldDownloadPrefillLogits true for single-token prefill" {
    try std.testing.expect(shouldDownloadPrefillLogits(0, 1));
}

test "linearWeightSupportsSequenceRows allows gaffine when matvec kernel is loaded" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    var weight = LinearWeight{
        .gaffine_u4 = .{
            .rows = 16,
            .cols = 16,
            .packed_data = dummy_buffer,
            .scales = dummy_buffer,
            .biases = dummy_buffer,
            .group_size = 8,
            .scales_dtype_tag = gaffine_scales_dtype_bf16,
        },
    };

    try std.testing.expect(!CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, false));
    try std.testing.expect(CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, true));
}

test "canFuseDenseU16QkvWeights supports GQA-style unequal output dims" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 2048,
        .cols = 2048,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const k = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(CudaBackend.canFuseDenseU16QkvWeights(2048, q, k, v));
}

test "canFuseDenseU16QkvWeights rejects mixed dtypes" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 1024,
        .cols = 1024,
        .buffer = dummy_buffer,
        .dtype = .f16,
    };
    const k = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(!CudaBackend.canFuseDenseU16QkvWeights(1024, q, k, v));
}

test "collectTokenPositions returns all matching positions" {
    const tokens = [_]u32{ 7, 3, 7, 9, 7 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 7);
    defer if (positions.len > 0) std.testing.allocator.free(positions);

    const expected = [_]usize{ 0, 2, 4 };
    try std.testing.expectEqualSlices(usize, expected[0..], positions);
}

test "collectTokenPositions returns empty when token is absent" {
    const tokens = [_]u32{ 1, 2, 3 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 9);
    try std.testing.expectEqual(@as(usize, 0), positions.len);
}

test "findPositionIndex locates mapped image feature index" {
    const positions = [_]usize{ 2, 5, 9 };
    try std.testing.expectEqual(@as(?usize, 0), findPositionIndex(positions[0..], 2));
    try std.testing.expectEqual(@as(?usize, 1), findPositionIndex(positions[0..], 5));
    try std.testing.expectEqual(@as(?usize, 2), findPositionIndex(positions[0..], 9));
    try std.testing.expectEqual(@as(?usize, null), findPositionIndex(positions[0..], 7));
}

test "materializeDenseOutInU16 handles out-in and in-out source layouts" {
    var out_in_data = [_]u16{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .bf16, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInU16(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    for (out_in_data, out_in_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }

    var in_out_data = [_]u16{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .bf16, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInU16(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]u16{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    for (expected, in_out_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }
}

test "materializeDenseOutInF32 handles out-in and in-out source layouts" {
    var out_in_data = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .f32, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInF32(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    try std.testing.expectEqualSlices(f32, out_in_data[0..], out_in_view.values);

    var in_out_data = [_]f32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .f32, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInF32(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]f32{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    try std.testing.expectEqualSlices(f32, expected[0..], in_out_view.values);
}
