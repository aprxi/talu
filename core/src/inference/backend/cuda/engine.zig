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
const plan_compiler = models.plan.compiler;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("../../runtime_contract/root.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const load_transforms = @import("../../../models/load/transforms.zig");
const vision_types = @import("../../vision_types.zig");
const smoke_checks = @import("smoke_checks.zig");
const attention_policy = @import("attention_policy.zig");
const attention_mod = @import("attention.zig");
const decode_mod = @import("decode.zig");
const prefill_mod = @import("prefill.zig");
const vision_runtime_mod = @import("vision/root.zig");
const cpu_kernels = @import("../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const prototype_eps: f32 = 1e-5;
const initial_kv_cache_tokens: usize = 256;
const kv_cache_dtype_fp16: bool = false;
const enable_fused_attention_f16_kv: bool = true;
const max_fused_attention_f16_kv_seq_len: u32 = 384;
const enable_device_embedding_lookup: bool = true;
const max_supported_fused_f16_kv_head_dim = 512;
// Optional dispatch observability. Keep disabled by default so production
// execution adds zero atomic overhead in the token loop.
const enable_dispatch_observability: bool = false;
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

fn saturatingU64FromU128(value: u128) u64 {
    return if (value > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(value);
}

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

const missing_device_tensor: DeviceTensor = std.mem.zeroes(DeviceTensor);
const missing_host_tensor: Tensor = std.mem.zeroes(Tensor);

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

const RuntimeBuffers = struct {
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
    gdelta_proj_dev: compute.cuda.Buffer,
    projection_weight: LinearWeight,
    logits_dev: compute.cuda.Buffer,

    fn init(
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
    ) !RuntimeBuffers {
        const d_model: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        if (d_model == 0 or vocab_size == 0) return error.InvalidArgument;
        if (max_dff == 0) return error.InvalidArgument;
        if (max_attn == 0) return error.InvalidArgument;
        if (max_kv == 0 or max_gdelta_proj == 0 or max_seq_len == 0 or head_dim == 0) return error.InvalidArgument;

        const d_model_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_ff_bytes = std.math.mul(usize, max_dff, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_attn_bytes = std.math.mul(usize, max_attn, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_kv_bytes = std.math.mul(usize, max_kv, @sizeOf(f32)) catch return error.InvalidArgument;
        const d_gdelta_proj_bytes = std.math.mul(usize, max_gdelta_proj, @sizeOf(f32)) catch return error.InvalidArgument;
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
        var gdelta_proj_dev = try device.allocBuffer(d_gdelta_proj_bytes);
        errdefer gdelta_proj_dev.deinit(device);
        var logits_dev = try device.allocBuffer(logits_bytes);
        errdefer logits_dev.deinit(device);

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
            .gdelta_proj_dev = gdelta_proj_dev,
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
        };
    }

    fn deinit(self: *RuntimeBuffers, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
        if (self.embedding_lookup) |*lookup| lookup.deinit(device);
        self.shortconv_conv_dev.deinit(device);
        self.shortconv_proj_dev.deinit(device);
        self.gdelta_proj_dev.deinit(device);
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

    fn deviceByteSize(self: *const RuntimeBuffers) usize {
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
            self.gdelta_proj_dev.size +
            self.logits_dev.size +
            (if (self.embedding_lookup) |lookup| lookup.byteSize() else 0) +
            self.projection_weight.byteSize();
    }

    fn requireAttentionScoresDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_scores_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    fn requireAttentionProbsDev(self: *RuntimeBuffers) !*compute.cuda.Buffer {
        if (self.attn_probs_dev) |*buf| return buf;
        return error.CudaKernelUnavailable;
    }

    fn ensureRowCapacity(self: *RuntimeBuffers, device: *compute.cuda.Device, required_rows: usize) !void {
        if (required_rows == 0) return error.InvalidArgument;
        if (required_rows <= self.row_capacity) return;
        if (required_rows > self.max_seq_len) return error.InvalidArgument;

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
        const shortconv_proj_bytes = std.math.mul(usize, self.shortconv_dim * 3, @sizeOf(f32)) catch return error.InvalidArgument;
        const shortconv_conv_bytes = std.math.mul(usize, self.shortconv_dim, @sizeOf(f32)) catch return error.InvalidArgument;

        try resizeScratchBuffer(device, &self.input_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.norm_out_dev, std.math.mul(usize, d_model_bytes, new_capacity) catch return error.InvalidArgument);
        try resizeScratchBuffer(device, &self.attn_q_dev, std.math.mul(usize, d_attn_bytes, new_capacity) catch return error.InvalidArgument);
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
        self.row_capacity = new_capacity;
    }
};

const LayerAttentionRuntime = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    d_ff: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
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
    cpu_kernel: ?cpu_kernels.MultiHeadAttention = null,
    cpu_cache: ?cpu_kernels.AttnCache = null,
    cpu_scratch: ?cpu_kernels.AttnTemp = null,
    cpu_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null,

    fn deinit(self: *LayerAttentionRuntime, device: *compute.cuda.Device) void {
        if (self.cpu_matmul_scratch) |*scratch| scratch.deinit();
        if (self.cpu_scratch) |*scratch| scratch.deinit(self.cpu_kernel.?.allocator);
        if (self.cpu_cache) |*cache| cache.deinit(self.cpu_kernel.?.allocator);
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

const LayerAttentionExecConfig = struct {
    q_dim: usize,
    q_projection_dim: usize,
    kv_dim: usize,
    sliding_window: usize,
    is_causal: bool,
    query_gate: bool,
};

fn expectedAttentionQProjectionDim(cfg: *const LayerAttentionExecConfig) usize {
    return if (cfg.query_gate) cfg.q_projection_dim else cfg.q_dim;
}

fn tensorProjectionOutputDim(weight: *const Tensor, input_dim: usize) !usize {
    if (weight.n_dims != 2) return error.InvalidShape;
    const dim0: usize = @intCast(weight.shape[0]);
    const dim1: usize = @intCast(weight.shape[1]);
    if (dim0 == 0 or dim1 == 0) return error.InvalidShape;
    if (dim0 == input_dim and dim1 != input_dim) return dim1;
    if (dim1 == input_dim and dim0 != input_dim) return dim0;
    if (dim0 == input_dim and dim1 == input_dim) return input_dim;
    return dim0;
}

fn bufferF32RowCount(buffer: *const compute.cuda.Buffer, width: usize) !usize {
    if (width == 0) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
    if (row_bytes == 0) return error.InvalidArgument;
    const rows = std.math.divExact(usize, buffer.size, row_bytes) catch return error.InvalidArgument;
    if (rows == 0) return error.InvalidArgument;
    return rows;
}

fn logicalF32RowSlice(
    buffer: *const compute.cuda.Buffer,
    rows: usize,
    row_index: usize,
    logical_width: usize,
) !compute.cuda.Buffer {
    if (rows == 0 or logical_width == 0 or row_index >= rows) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, logical_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
    if (buffer.size < packed_bytes) return error.InvalidInstructionBinding;

    const row_stride = if (buffer.size == packed_bytes)
        row_bytes
    else blk: {
        if (buffer.size % rows != 0) return error.InvalidInstructionBinding;
        const stride = buffer.size / rows;
        if (stride < row_bytes) return error.InvalidInstructionBinding;
        break :blk stride;
    };

    const row_offset = std.math.mul(usize, row_index, row_stride) catch return error.InvalidArgument;
    return bufferSlice(buffer, row_offset, row_bytes);
}

const AttentionWeightRefs = struct {
    q_proj: ?*const LinearWeight = null,
    k_proj: ?*const LinearWeight = null,
    v_proj: ?*const LinearWeight = null,
    o_proj: ?*const LinearWeight = null,
    q_norm_weight: ?*const DeviceTensor = null,
    k_norm_weight: ?*const DeviceTensor = null,
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

const GatedDeltaBlockRuntime = struct {
    d_ff: usize,
    ln1_weight: DeviceTensor,
    ln2_weight: ?DeviceTensor = null,
    ffn_w1: ?LinearWeight = null,
    ffn_w2: ?LinearWeight = null,
    ffn_w3: ?LinearWeight = null,
    in_proj: LinearWeight,
    kernel: cpu_kernels.GatedDeltaKernel,
    state: cpu_kernels.GatedDeltaState,
    scratch: cpu_kernels.GatedDeltaScratch,
    matmul_scratch: compute.cpu.linalg.MatmulScratch,

    fn deinit(self: *GatedDeltaBlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.in_proj.deinit(device);
        if (self.ffn_w3) |*w| w.deinit(device);
        if (self.ffn_w2) |*w| w.deinit(device);
        if (self.ffn_w1) |*w| w.deinit(device);
        self.scratch.deinit();
        self.state.deinit();
        self.kernel.deinit();
        _ = allocator;
        self.matmul_scratch.deinit();
        if (self.ln2_weight) |*w| w.deinit(device);
        self.ln1_weight.deinit(device);
    }
};

const ShortConvExecConfig = struct {
    conv_dim: usize,
    d_conv: usize,
};

const ShortConvWeightRefs = struct {
    in_proj: ?*const LinearWeight = null,
    conv_weight: ?*const DeviceTensor = null,
    out_proj: ?*const LinearWeight = null,
    conv_bias: ?*const DeviceTensor = null,
};

const GatedDeltaWeightRefs = struct {
    in_proj: ?*const Tensor = null,
    conv_weight: ?*const Tensor = null,
    a_log: ?*const Tensor = null,
    out_proj: ?*const Tensor = null,
    conv_bias: ?*const Tensor = null,
    dt_bias: ?*const Tensor = null,
    norm_weight: ?*const Tensor = null,
};

const SwiGluWeightRefs = struct {
    w1: ?*const LinearWeight = null,
    w3: ?*const LinearWeight = null,
    w2: ?*const LinearWeight = null,
    w1_bias: ?*const DeviceTensor = null,
    w2_bias: ?*const DeviceTensor = null,
};

const BlockRuntimeLayer = struct {
    const invalid_slot = std.math.maxInt(u8);
    const MaxNormWeights = 4;

    compiled_plan: ?runtime_contract.CompiledPlan = null,
    instruction_norm_weight_slots: []?*const DeviceTensor = &.{},
    instruction_attention_exec_meta: []?LayerAttentionExecConfig = &.{},
    instruction_attention_weight_slots: []?AttentionWeightRefs = &.{},
    instruction_shortconv_exec_meta: []?ShortConvExecConfig = &.{},
    instruction_shortconv_weight_slots: []?ShortConvWeightRefs = &.{},
    instruction_gated_delta_weight_slots: []?GatedDeltaWeightRefs = &.{},
    instruction_swiglu_weight_slots: []?SwiGluWeightRefs = &.{},
    instruction_weight_offsets: []u32 = &.{},
    instruction_weight_ptrs: []?*anyopaque = &.{},
    register_to_slot_map: []const u8 = &.{},
    slot_width_hints: []const usize = &.{},
    attention_runtime: ?LayerAttentionRuntime = null,
    shortconv_runtime: ?ShortConvBlockRuntime = null,
    gated_delta_runtime: ?GatedDeltaBlockRuntime = null,
    attention_binding: ?*LayerAttentionRuntime = null,
    shortconv_binding: ?*ShortConvBlockRuntime = null,
    gated_delta_binding: ?*GatedDeltaBlockRuntime = null,
    norm_weights: [MaxNormWeights]?*const DeviceTensor = [_]?*const DeviceTensor{null} ** MaxNormWeights,
    norm_weight_count: u8 = 0,

    fn instructionKernelIdFromWeightBindings(
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        opcode: runtime_contract.Opcode,
    ) !u32 {
        return runtime_contract.instructionKernelBindingId(compiled, op_index, opcode);
    }

    const InstructionRefBinderFn = *const fn (
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) anyerror!void;

    fn bindInstructionNoop(
        _: *BlockRuntimeLayer,
        _: *const runtime_contract.CompiledPlan,
        _: usize,
        _: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {}

    fn bindInstructionRmsNorm(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        norm_index: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (norm_index.* >= self.norm_weight_count) return error.UnsupportedModel;
        const weight = self.norm_weights[norm_index.*] orelse return error.UnsupportedModel;
        self.instruction_norm_weight_slots[op_index] = weight;
        norm_index.* += 1;
    }

    fn bindInstructionAttention(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.attention_binding orelse return error.UnsupportedModel;
        self.instruction_attention_exec_meta[op_index] = .{
            .q_dim = binding.q_dim,
            .q_projection_dim = binding.q_projection_dim,
            .kv_dim = binding.kv_dim,
            .sliding_window = binding.sliding_window,
            .is_causal = binding.is_causal,
            .query_gate = binding.query_gate,
        };
        self.instruction_attention_weight_slots[op_index] = .{
            .q_proj = &binding.q_proj,
            .k_proj = &binding.k_proj,
            .v_proj = &binding.v_proj,
            .o_proj = &binding.o_proj,
            .q_norm_weight = if (binding.q_norm_weight) |*weight| weight else null,
            .k_norm_weight = if (binding.k_norm_weight) |*weight| weight else null,
        };
    }

    fn bindInstructionShortConv(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.shortconv_binding orelse return error.UnsupportedModel;
        self.instruction_shortconv_exec_meta[op_index] = .{
            .conv_dim = binding.conv_dim,
            .d_conv = binding.d_conv,
        };
        self.instruction_shortconv_weight_slots[op_index] = .{
            .in_proj = &binding.in_proj,
            .conv_weight = &binding.conv_weight_time_major,
            .out_proj = &binding.out_proj,
            .conv_bias = if (binding.conv_bias) |*weight| weight else null,
        };
    }

    fn bindInstructionGatedDelta(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        const binding = self.gated_delta_binding orelse return error.UnsupportedModel;
        self.instruction_gated_delta_weight_slots[op_index] = .{
            .in_proj = binding.kernel.weights.in_proj,
            .conv_weight = binding.kernel.weights.conv1d_weight,
            .a_log = binding.kernel.weights.A_log,
            .out_proj = binding.kernel.weights.out_proj,
            .conv_bias = binding.kernel.weights.conv1d_bias,
            .dt_bias = binding.kernel.weights.dt_bias,
            .norm_weight = binding.kernel.weights.norm_weight,
        };
    }

    fn bindInstructionSwiGlu(
        self: *BlockRuntimeLayer,
        compiled: *const runtime_contract.CompiledPlan,
        op_index: usize,
        insn: *const runtime_contract.Instruction,
        _: *usize,
    ) !void {
        _ = try instructionKernelIdFromWeightBindings(compiled, op_index, insn.opcode);
        if (self.attention_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = &binding.w1,
                .w3 = &binding.w3,
                .w2 = &binding.w2,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.shortconv_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        if (self.gated_delta_binding) |binding| {
            self.instruction_swiglu_weight_slots[op_index] = .{
                .w1 = if (binding.ffn_w1) |*w| w else null,
                .w3 = if (binding.ffn_w3) |*w| w else null,
                .w2 = if (binding.ffn_w2) |*w| w else null,
                .w1_bias = null,
                .w2_bias = null,
            };
            return;
        }
        return error.UnsupportedModel;
    }

    const instruction_rebind_table: [256]?InstructionRefBinderFn = blk: {
        var table: [256]?InstructionRefBinderFn = [_]?InstructionRefBinderFn{bindInstructionNoop} ** 256;
        table[@intFromEnum(runtime_contract.Opcode.rmsnorm)] = bindInstructionRmsNorm;
        table[@intFromEnum(runtime_contract.Opcode.multihead_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.mla_attention)] = bindInstructionAttention;
        table[@intFromEnum(runtime_contract.Opcode.gated_delta_net)] = bindInstructionGatedDelta;
        table[@intFromEnum(runtime_contract.Opcode.shortconv)] = bindInstructionShortConv;
        table[@intFromEnum(runtime_contract.Opcode.swiglu)] = bindInstructionSwiGlu;
        break :blk table;
    };

    fn rebuildInstructionMetadata(self: *BlockRuntimeLayer, allocator: std.mem.Allocator) !void {
        if (self.instruction_norm_weight_slots.len != 0) {
            allocator.free(self.instruction_norm_weight_slots);
            self.instruction_norm_weight_slots = &.{};
        }
        if (self.instruction_attention_exec_meta.len != 0) {
            allocator.free(self.instruction_attention_exec_meta);
            self.instruction_attention_exec_meta = &.{};
        }
        if (self.instruction_attention_weight_slots.len != 0) {
            allocator.free(self.instruction_attention_weight_slots);
            self.instruction_attention_weight_slots = &.{};
        }
        if (self.instruction_shortconv_exec_meta.len != 0) {
            allocator.free(self.instruction_shortconv_exec_meta);
            self.instruction_shortconv_exec_meta = &.{};
        }
        if (self.instruction_shortconv_weight_slots.len != 0) {
            allocator.free(self.instruction_shortconv_weight_slots);
            self.instruction_shortconv_weight_slots = &.{};
        }
        if (self.instruction_gated_delta_weight_slots.len != 0) {
            allocator.free(self.instruction_gated_delta_weight_slots);
            self.instruction_gated_delta_weight_slots = &.{};
        }
        if (self.instruction_swiglu_weight_slots.len != 0) {
            allocator.free(self.instruction_swiglu_weight_slots);
            self.instruction_swiglu_weight_slots = &.{};
        }
        if (self.instruction_weight_offsets.len != 0) {
            allocator.free(self.instruction_weight_offsets);
            self.instruction_weight_offsets = &.{};
        }
        if (self.instruction_weight_ptrs.len != 0) {
            allocator.free(self.instruction_weight_ptrs);
            self.instruction_weight_ptrs = &.{};
        }

        const compiled = self.compiled_plan orelse return;
        const len = compiled.plan.instructions.len;
        self.instruction_norm_weight_slots = try allocator.alloc(?*const DeviceTensor, len);
        self.instruction_attention_exec_meta = try allocator.alloc(?LayerAttentionExecConfig, len);
        self.instruction_attention_weight_slots = try allocator.alloc(?AttentionWeightRefs, len);
        self.instruction_shortconv_exec_meta = try allocator.alloc(?ShortConvExecConfig, len);
        self.instruction_shortconv_weight_slots = try allocator.alloc(?ShortConvWeightRefs, len);
        self.instruction_gated_delta_weight_slots = try allocator.alloc(?GatedDeltaWeightRefs, len);
        self.instruction_swiglu_weight_slots = try allocator.alloc(?SwiGluWeightRefs, len);
        @memset(self.instruction_norm_weight_slots, null);
        @memset(self.instruction_attention_exec_meta, null);
        @memset(self.instruction_attention_weight_slots, null);
        @memset(self.instruction_shortconv_exec_meta, null);
        @memset(self.instruction_shortconv_weight_slots, null);
        @memset(self.instruction_gated_delta_weight_slots, null);
        @memset(self.instruction_swiglu_weight_slots, null);

        var norm_index: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            const binder = instruction_rebind_table[@intFromEnum(insn.opcode)] orelse continue;
            try binder(self, &compiled, op_index, &insn, &norm_index);
        }
        try self.buildInstructionWeightTable(allocator, &compiled);
    }

    fn resolveInstructionWeightPtrForSlot(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_idx: usize,
    ) !*anyopaque {
        switch (opcode) {
            .rmsnorm => {
                return switch (slot_idx) {
                    0 => blk: {
                        const weight = try self.instructionNormWeightRef(op_index);
                        break :blk @ptrCast(@constCast(weight));
                    },
                    1 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .multihead_attention => {
                if (op_index >= self.instruction_attention_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_attention_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.q_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.k_proj orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.v_proj orelse return error.MissingWeight)),
                    3 => @ptrCast(@constCast(binding.o_proj orelse return error.MissingWeight)),
                    4 => if (binding.q_norm_weight) |q_norm|
                        @ptrCast(@constCast(q_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    5 => if (binding.k_norm_weight) |k_norm|
                        @ptrCast(@constCast(k_norm))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    6 => @ptrCast(@constCast(&missing_device_tensor)),
                    7 => @ptrCast(@constCast(&missing_device_tensor)),
                    8 => @ptrCast(@constCast(&missing_device_tensor)),
                    9 => @ptrCast(@constCast(&missing_device_tensor)),
                    10 => @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .mla_attention => {
                return error.UnsupportedModel;
            },
            .mamba_mixer => {
                return error.UnsupportedModel;
            },
            .gated_delta_net => {
                if (op_index >= self.instruction_gated_delta_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_gated_delta_weight_slots[op_index] orelse return error.UnsupportedModel;
                const missing_tensor = &missing_host_tensor;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse missing_tensor)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse missing_tensor)),
                    2 => @ptrCast(@constCast(binding.a_log orelse missing_tensor)),
                    3 => @ptrCast(@constCast(binding.out_proj orelse missing_tensor)),
                    4 => @ptrCast(@constCast(binding.conv_bias orelse missing_tensor)),
                    5 => @ptrCast(@constCast(binding.dt_bias orelse missing_tensor)),
                    6 => @ptrCast(@constCast(binding.norm_weight orelse missing_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .swiglu => {
                if (op_index >= self.instruction_swiglu_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_swiglu_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.w1 orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.w3 orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.w2 orelse return error.MissingWeight)),
                    3 => if (binding.w1_bias) |w1_bias|
                        @ptrCast(@constCast(w1_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    4 => if (binding.w2_bias) |w2_bias|
                        @ptrCast(@constCast(w2_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            .moe => {
                return error.UnsupportedModel;
            },
            .shortconv => {
                if (op_index >= self.instruction_shortconv_weight_slots.len) return error.InvalidInstructionIndex;
                const binding = self.instruction_shortconv_weight_slots[op_index] orelse return error.UnsupportedModel;
                return switch (slot_idx) {
                    0 => @ptrCast(@constCast(binding.in_proj orelse return error.MissingWeight)),
                    1 => @ptrCast(@constCast(binding.conv_weight orelse return error.MissingWeight)),
                    2 => @ptrCast(@constCast(binding.out_proj orelse return error.MissingWeight)),
                    3 => if (binding.conv_bias) |conv_bias|
                        @ptrCast(@constCast(conv_bias))
                    else
                        @ptrCast(@constCast(&missing_device_tensor)),
                    else => error.InvalidWeightRefCount,
                };
            },
            else => return error.InvalidInstructionBinding,
        }
    }

    fn resolveInstructionWeightPtr(
        self: *const BlockRuntimeLayer,
        opcode: runtime_contract.Opcode,
        op_index: usize,
        slot_name: []const u8,
        slot_idx: usize,
    ) !*anyopaque {
        const expected_slots = runtime_contract.expectedKernelWeightSlots(opcode);
        if (slot_idx >= expected_slots.len) return error.InvalidWeightRefCount;
        if (!std.mem.eql(u8, expected_slots[slot_idx], slot_name)) return error.InvalidWeightBindingName;
        return self.resolveInstructionWeightPtrForSlot(opcode, op_index, slot_idx);
    }

    fn buildInstructionWeightTable(
        self: *BlockRuntimeLayer,
        allocator: std.mem.Allocator,
        compiled: *const runtime_contract.CompiledPlan,
    ) !void {
        const insn_len = compiled.plan.instructions.len;
        const offsets = try allocator.alloc(u32, insn_len + 1);
        errdefer allocator.free(offsets);

        var total_slots: usize = 0;
        for (compiled.plan.instructions) |insn| total_slots += insn.weights.len;
        const ptrs = try allocator.alloc(?*anyopaque, total_slots);
        errdefer allocator.free(ptrs);
        @memset(ptrs, null);

        var cursor: usize = 0;
        for (compiled.plan.instructions, 0..) |insn, op_index| {
            offsets[op_index] = @intCast(cursor);
            const expected_slots = runtime_contract.expectedKernelWeightSlots(insn.opcode);
            if (insn.weights.len != expected_slots.len) return error.InvalidWeightRefCount;
            for (insn.weights, 0..) |_, slot_idx| {
                const parsed = try runtime_contract.instructionKernelWeightBinding(
                    compiled,
                    op_index,
                    insn.opcode,
                    slot_idx,
                );
                const weight_ptr = try self.resolveInstructionWeightPtr(insn.opcode, op_index, parsed.slot_name, slot_idx);
                ptrs[cursor] = weight_ptr;
                cursor += 1;
            }
        }
        offsets[insn_len] = @intCast(cursor);
        self.instruction_weight_offsets = offsets;
        self.instruction_weight_ptrs = ptrs;
    }

    fn instructionNormWeightRef(self: *const BlockRuntimeLayer, op_index: usize) !*const DeviceTensor {
        if (op_index >= self.instruction_norm_weight_slots.len) return error.InvalidInstructionIndex;
        return self.instruction_norm_weight_slots[op_index] orelse return error.UnsupportedModel;
    }

    fn instructionAttentionRef(self: *const BlockRuntimeLayer, op_index: usize) !*const LayerAttentionExecConfig {
        if (op_index >= self.instruction_attention_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_attention_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    fn instructionShortConvRef(self: *const BlockRuntimeLayer, op_index: usize) !*const ShortConvExecConfig {
        if (op_index >= self.instruction_shortconv_exec_meta.len) return error.InvalidInstructionIndex;
        if (self.instruction_shortconv_exec_meta[op_index]) |*binding| return binding;
        return error.UnsupportedModel;
    }

    fn deinit(self: *BlockRuntimeLayer, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        if (self.register_to_slot_map.len != 0) allocator.free(self.register_to_slot_map);
        if (self.slot_width_hints.len != 0) allocator.free(self.slot_width_hints);
        if (self.instruction_norm_weight_slots.len != 0) allocator.free(self.instruction_norm_weight_slots);
        if (self.instruction_attention_exec_meta.len != 0) allocator.free(self.instruction_attention_exec_meta);
        if (self.instruction_attention_weight_slots.len != 0) allocator.free(self.instruction_attention_weight_slots);
        if (self.instruction_shortconv_exec_meta.len != 0) allocator.free(self.instruction_shortconv_exec_meta);
        if (self.instruction_shortconv_weight_slots.len != 0) allocator.free(self.instruction_shortconv_weight_slots);
        if (self.instruction_gated_delta_weight_slots.len != 0) allocator.free(self.instruction_gated_delta_weight_slots);
        if (self.instruction_swiglu_weight_slots.len != 0) allocator.free(self.instruction_swiglu_weight_slots);
        if (self.instruction_weight_offsets.len != 0) allocator.free(self.instruction_weight_offsets);
        if (self.instruction_weight_ptrs.len != 0) allocator.free(self.instruction_weight_ptrs);
        if (self.compiled_plan) |*compiled_plan| {
            plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
            self.compiled_plan = null;
        }
        if (self.attention_runtime) |*block| block.deinit(device);
        if (self.shortconv_runtime) |*block| block.deinit(device);
        if (self.gated_delta_runtime) |*block| block.deinit(allocator, device);
        self.* = .{};
    }
};

fn buildCudaLayerProgramRegisterSlotMap(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
) ![]u8 {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count = compiled.plan.register_count;
    const register_to_slot = try allocator.alloc(u8, register_count);
    @memset(register_to_slot, invalid_slot);
    errdefer allocator.free(register_to_slot);
    if (register_count <= 1) return register_to_slot;

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    // Register 0 (residual) uses runtime_buffers.input_dev, not a slot buffer.
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;
    // Plan specs already contain floors applied at compile time.
    // Backends consume specs exactly.
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }

    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);
    if (physical.physical_count == 0) return register_to_slot;

    const physical_to_slot = try allocator.alloc(u8, physical.physical_count);
    defer allocator.free(physical_to_slot);
    @memset(physical_to_slot, invalid_slot);

    var next_slot: u8 = 0;
    const invalid_physical = std.math.maxInt(u16);
    var register_idx: usize = 0;
    while (register_idx < register_count) : (register_idx += 1) {
        const physical_id_u16 = physical.register_to_physical[register_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        if (physical_id >= physical_to_slot.len) return error.UnsupportedModel;
        if (physical_to_slot[physical_id] == invalid_slot) {
            physical_to_slot[physical_id] = next_slot;
            next_slot += 1;
        }
        register_to_slot[register_idx] = physical_to_slot[physical_id];
    }

    return register_to_slot;
}

fn buildCudaLayerProgramSlotWidthHints(
    allocator: std.mem.Allocator,
    compiled: *const runtime_contract.CompiledPlan,
    register_to_slot_map: []const u8,
) ![]usize {
    const invalid_slot = BlockRuntimeLayer.invalid_slot;
    const register_count: usize = compiled.plan.register_count;
    if (register_to_slot_map.len != register_count) return error.InvalidRegisterSpecCount;
    if (compiled.register_buffer_specs.len != register_count) return error.InvalidRegisterSpecCount;

    var required_slots: usize = 0;
    for (register_to_slot_map) |slot_idx| {
        if (slot_idx == invalid_slot) continue;
        const next = @as(usize, slot_idx) + 1;
        if (next > required_slots) required_slots = next;
    }
    if (required_slots == 0) return &.{};

    const slot_width_hints = try allocator.alloc(usize, required_slots);
    @memset(slot_width_hints, 0);
    errdefer allocator.free(slot_width_hints);

    const register_specs = try allocator.alloc(runtime_contract.RegisterBufferSpec, register_count);
    defer allocator.free(register_specs);
    register_specs[0] = .{ .size = 0, .@"align" = 0 };
    for (register_specs[1..], 1..) |*spec, idx| {
        const plan_spec = compiled.register_buffer_specs[idx];
        spec.* = .{
            .size = plan_spec.size,
            .@"align" = plan_spec.@"align",
            .dtype = plan_spec.dtype,
            .layout = plan_spec.layout,
        };
    }
    var physical = try runtime_contract.buildPhysicalMappingLinearScan(allocator, compiled, register_specs);
    defer runtime_contract.deinitPhysicalMapping(allocator, &physical);

    const invalid_physical = std.math.maxInt(u16);
    for (0..register_count) |reg_idx| {
        const slot_idx = register_to_slot_map[reg_idx];
        if (slot_idx == invalid_slot) continue;
        const physical_id_u16 = physical.register_to_physical[reg_idx];
        if (physical_id_u16 == invalid_physical) continue;
        const physical_id: usize = physical_id_u16;
        const width = physical.physical_specs[physical_id].size;
        if (width == 0) return error.InvalidRegisterSpecSize;
        const slot_usize: usize = @intCast(slot_idx);
        if (slot_width_hints[slot_usize] == 0) {
            slot_width_hints[slot_usize] = width;
        } else if (slot_width_hints[slot_usize] != width) {
            return error.InvalidRegisterSpecSize;
        }
    }
    for (slot_width_hints) |width| {
        if (width == 0) return error.InvalidRegisterSpecSize;
    }
    return slot_width_hints;
}

fn validateCompiledLayerPlanForCuda(
    compiled: *const runtime_contract.CompiledPlan,
    layer_idx: usize,
    kind: op_types.BlockKind,
) !void {
    runtime_contract.validateExecutionPlanForBlockKind(&compiled.plan, kind) catch |err| {
        log.warn("inference", "CUDA compiled layer plan fails block-kind validation", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .reason = @errorName(err),
        });
        return error.UnsupportedModel;
    };
    if (runtime_contract.firstUnsupportedInstructionOpcode(&compiled.plan, CudaBackend.layer_program_adapter_table)) |unsupported| {
        log.warn("inference", "CUDA compiled layer plan contains unsupported opcode", .{
            .layer = layer_idx,
            .kind = @intFromEnum(kind),
            .op_index = unsupported.instruction_index,
            .opcode = @intFromEnum(unsupported.opcode),
        });
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
    gated_delta_block_count: usize,
    q_norm_blocks: usize,
    k_norm_blocks: usize,
    linear_weight_bytes: usize,
    norm_weight_bytes: usize,
    kv_cache_bytes: usize,
    shortconv_state_bytes: usize,
    gated_delta_state_bytes: usize,
    max_shortconv_dim: usize,
    max_gdelta_proj: usize,

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
        var gated_delta_block_count: usize = 0;
        var q_norm_blocks: usize = 0;
        var k_norm_blocks: usize = 0;
        var linear_weight_bytes: usize = 0;
        var norm_weight_bytes: usize = 0;
        var kv_cache_bytes: usize = 0;
        var shortconv_state_bytes: usize = 0;
        var gated_delta_state_bytes: usize = 0;
        var max_shortconv_dim: usize = 0;
        var max_gdelta_proj: usize = 0;
        var blocks = try allocator.alloc(BlockRuntimeLayer, layer_count);
        errdefer allocator.free(blocks);
        for (blocks) |*layer| layer.* = .{};

        var initialized: usize = 0;
        errdefer {
            while (initialized > 0) {
                initialized -= 1;
                blocks[initialized].deinit(allocator, device);
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
                    const program = models.registry.blockProgramFor(entry, .attention_mlp) orelse {
                        log.warn("inference", "CUDA block runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.attention_mlp),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[layer_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[layer_idx].compiled_plan.?, layer_idx, .attention_mlp);
                    errdefer if (blocks[layer_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[layer_idx].compiled_plan = null;
                    };
                    blocks[layer_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[layer_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[layer_idx].register_to_slot_map);
                        blocks[layer_idx].register_to_slot_map = &.{};
                    };
                    blocks[layer_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                        blocks[layer_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[layer_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[layer_idx].slot_width_hints);
                        blocks[layer_idx].slot_width_hints = &.{};
                    };
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
                    const q_proj_out = if (attn.attention_config.query_gate) q_out * 2 else q_out;
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
                                .q_proj_out = q_proj_out,
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
                        if (attn.attention_config.query_gate) {
                            log.warn("inference", "CUDA block runtime fused qkv with query_gate unsupported", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        }
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
                        q_proj_dev = try uploadLinearWeightWithContext(device, allocator, q_proj, d_model, layer_idx, "self_attn.q_proj.weight");
                        k_proj_dev = try uploadLinearWeightWithContext(device, allocator, k_proj, d_model, layer_idx, "self_attn.k_proj.weight");
                        v_proj_dev = try uploadLinearWeightWithContext(device, allocator, v_proj, d_model, layer_idx, "self_attn.v_proj.weight");
                    }
                    errdefer q_proj_dev.deinit(device);
                    errdefer k_proj_dev.deinit(device);
                    errdefer v_proj_dev.deinit(device);

                    var o_proj_dev = try uploadLinearWeightWithContext(device, allocator, attn.o_proj, q_out, layer_idx, "self_attn.o_proj.weight");
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
                        w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
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
                    var w2_dev = try uploadLinearWeightWithContext(device, allocator, w2, d_ff, layer_idx, "mlp.down_proj.weight");
                    errdefer w2_dev.deinit(device);
                    var cpu_attention_kernel: ?cpu_kernels.MultiHeadAttention = null;
                    var cpu_attention_cache: ?cpu_kernels.AttnCache = null;
                    var cpu_attention_scratch: ?cpu_kernels.AttnTemp = null;
                    var cpu_attention_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null;
                    if (attn.attention_config.query_gate) {
                        const q_proj_host = attn.q_proj orelse return error.MissingWeight;
                        const k_proj_host = attn.k_proj orelse return error.MissingWeight;
                        const v_proj_host = attn.v_proj orelse return error.MissingWeight;
                        const dk_q = try compute.cpu.linalg.matmulKernel(q_proj_host.dtype);
                        const dk_k = if (k_proj_host.dtype != q_proj_host.dtype) try compute.cpu.linalg.matmulKernel(k_proj_host.dtype) else null;
                        const dk_v = if (v_proj_host.dtype != q_proj_host.dtype) try compute.cpu.linalg.matmulKernel(v_proj_host.dtype) else null;
                        const dk_o = try compute.cpu.linalg.matmulKernel(attn.o_proj.dtype);
                        cpu_attention_kernel = .{
                            .d_model = d_model,
                            .n_heads = n_heads,
                            .n_kv_heads = n_kv_heads,
                            .head_dim = head_dim,
                            .max_seq_len = max_seq_len,
                            .scale = if (loaded.config.attention_multiplier > 0.0)
                                loaded.config.attention_multiplier
                            else if (loaded.config.query_pre_attn_scalar > 0.0)
                                1.0 / std.math.sqrt(loaded.config.query_pre_attn_scalar)
                            else
                                1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim))),
                            .qk_norm_weight_offset = loaded.runtime.qk_norm_weight_offset,
                            .sliding_window = attn.sliding_window,
                            .is_causal = attn.is_causal,
                            .layer_idx = @intCast(layer_idx),
                            .query_gate = attn.attention_config.query_gate,
                            .q_proj = attn.q_proj,
                            .k_proj = attn.k_proj,
                            .v_proj = attn.v_proj,
                            .o_proj = attn.o_proj,
                            .fused_qkv = attn.fused.qkv_proj,
                            .rope = null,
                            .runtime_rope = null,
                            .position_delta = 0,
                            .rope_interleaved = attn.attention_config.rope_interleaved orelse false,
                            .q_norm = attn.q_norm,
                            .k_norm = attn.k_norm,
                            .norm_eps = if (loaded.config.norm_eps > 0.0) loaded.config.norm_eps else prototype_eps,
                            .allocator = allocator,
                            .matmul_qkv = dk_q.func,
                            .matmul_k = if (dk_k) |dk| dk.func else null,
                            .matmul_v = if (dk_v) |dk| dk.func else null,
                            .matmul_qkv_fused = null,
                            .matmul_o = dk_o.func,
                            .kernel_name_qkv = dk_q.name,
                            .kernel_name_k = if (dk_k) |dk| dk.name else null,
                            .kernel_name_v = if (dk_v) |dk| dk.name else null,
                            .kernel_name_qkv_fused = null,
                            .kernel_name_o = dk_o.name,
                            .q_bias = attn.q_bias,
                            .k_bias = attn.k_bias,
                            .v_bias = attn.v_bias,
                            .o_bias = attn.o_bias,
                            .sinks = attn.sinks,
                            .flash_attention_fn = null,
                        };
                        cpu_attention_cache = .{};
                        cpu_attention_scratch = .{};
                        cpu_attention_matmul_scratch = try compute.cpu.linalg.MatmulScratch.init(allocator);
                    }
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
                    if (q_proj_dev.cols() != q_proj_out) {
                        log.warn("inference", "CUDA block runtime q_proj dim unsupported", .{
                            .layer = layer_idx,
                            .q_cols = q_proj_dev.cols(),
                            .expected = q_proj_out,
                            .query_gate = @as(u8, @intFromBool(attn.attention_config.query_gate)),
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

                    blocks[layer_idx].attention_runtime = .{
                        .q_dim = q_out,
                        .q_projection_dim = q_proj_dev.cols(),
                        .kv_dim = k_proj_dev.cols(),
                        .d_ff = d_ff,
                        .sliding_window = attn.sliding_window,
                        .is_causal = attn.is_causal,
                        .query_gate = attn.attention_config.query_gate,
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
                        .cpu_kernel = cpu_attention_kernel,
                        .cpu_cache = cpu_attention_cache,
                        .cpu_scratch = cpu_attention_scratch,
                        .cpu_matmul_scratch = cpu_attention_matmul_scratch,
                    };
                    blocks[layer_idx].attention_binding = &blocks[layer_idx].attention_runtime.?;
                    CudaBackend.bindAttentionNormWeights(&blocks[layer_idx], &blocks[layer_idx].attention_runtime.?);
                    attention_block_count += 1;
                },
                .gated_delta => |gated_delta| {
                    const in_proj_cols = try tensorProjectionOutputDim(gated_delta.weights.in_proj, d_model);
                    max_gdelta_proj = @max(max_gdelta_proj, in_proj_cols);
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA gated-delta runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    const program = models.registry.blockProgramFor(entry, .gated_delta) orelse {
                        log.warn("inference", "CUDA gated-delta runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.gated_delta),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[layer_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[layer_idx].compiled_plan.?, layer_idx, .gated_delta);
                    errdefer if (blocks[layer_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[layer_idx].compiled_plan = null;
                    };
                    blocks[layer_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[layer_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[layer_idx].register_to_slot_map);
                        blocks[layer_idx].register_to_slot_map = &.{};
                    };
                    blocks[layer_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                        blocks[layer_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[layer_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[layer_idx].slot_width_hints);
                        blocks[layer_idx].slot_width_hints = &.{};
                    };
                    if (gated_delta.fused_gate_up != null) {
                        log.warn("inference", "CUDA gated-delta fused gate_up not supported yet", .{
                            .layer = layer_idx,
                        });
                        return error.UnsupportedModel;
                    }
                    if (gated_delta.config.d_model != d_model) {
                        log.warn("inference", "CUDA gated-delta d_model mismatch", .{
                            .layer = layer_idx,
                            .config_d_model = gated_delta.config.d_model,
                            .model_d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln1_weight = try uploadTensor(device, allocator, gated_delta.ln1_weight);
                    errdefer ln1_weight.deinit(device);
                    if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                        log.warn("inference", "CUDA gated-delta ln1 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln1_weight.rows,
                            .cols = ln1_weight.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }

                    var ln2_weight: ?DeviceTensor = null;
                    if (gated_delta.ln2_weight) |ln2| {
                        var ln2_dev = try uploadTensor(device, allocator, ln2);
                        errdefer ln2_dev.deinit(device);
                        if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                            log.warn("inference", "CUDA gated-delta ln2 shape unsupported", .{
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

                    var ffn_w1: ?LinearWeight = null;
                    var ffn_w2: ?LinearWeight = null;
                    var ffn_w3: ?LinearWeight = null;
                    var d_ff: usize = 0;
                    if (gated_delta.w1 != null or gated_delta.w2 != null or gated_delta.w3 != null) {
                        const w1 = gated_delta.w1 orelse return error.MissingWeight;
                        const w2 = gated_delta.w2 orelse return error.MissingWeight;
                        const w3 = gated_delta.w3 orelse return error.MissingWeight;
                        if (ln2_weight == null) {
                            log.warn("inference", "CUDA gated-delta ffn requires ln2", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        }
                        var w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        errdefer w1_dev.deinit(device);
                        var w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                        errdefer w3_dev.deinit(device);
                        if (w1_dev.cols() != w3_dev.cols()) {
                            log.warn("inference", "CUDA gated-delta gate/up dim mismatch", .{
                                .layer = layer_idx,
                                .w1_cols = w1_dev.cols(),
                                .w3_cols = w3_dev.cols(),
                            });
                            return error.UnsupportedModel;
                        }
                        d_ff = w1_dev.cols();
                        var w2_dev = try uploadLinearWeightWithContext(device, allocator, w2, d_ff, layer_idx, "mlp.down_proj.weight");
                        errdefer w2_dev.deinit(device);
                        if (w2_dev.cols() != d_model) {
                            log.warn("inference", "CUDA gated-delta down_proj out dim unsupported", .{
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

                    var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, gated_delta.weights.in_proj, d_model, layer_idx, "gated_delta.in_proj");
                    errdefer in_proj_dev.deinit(device);

                    const in_proj_dispatch = try compute.cpu.linalg.matmulKernel(gated_delta.weights.in_proj.dtype);
                    const out_proj_dispatch = try compute.cpu.linalg.matmulKernel(gated_delta.weights.out_proj.dtype);
                    var gated_delta_kernel = cpu_kernels.GatedDeltaKernel.init(
                        .{
                            .d_model = gated_delta.config.d_model,
                            .d_conv = gated_delta.config.d_conv,
                            .n_heads = gated_delta.config.n_heads,
                            .d_head = gated_delta.config.d_head,
                        },
                        .{
                            .in_proj = gated_delta.weights.in_proj,
                            .conv1d_weight = gated_delta.weights.conv1d_weight,
                            .conv1d_bias = gated_delta.weights.conv1d_bias,
                            .A_log = gated_delta.weights.A_log,
                            .dt_bias = gated_delta.weights.dt_bias,
                            .norm_weight = gated_delta.weights.norm_weight,
                            .out_proj = gated_delta.weights.out_proj,
                        },
                        in_proj_dispatch.func,
                        out_proj_dispatch.func,
                    );
                    gated_delta_kernel.layer_idx = @intCast(layer_idx);
                    try gated_delta_kernel.initTransposedWeights(allocator);
                    errdefer gated_delta_kernel.deinit();

                    var gated_delta_state = try cpu_kernels.GatedDeltaState.init(
                        allocator,
                        1,
                        .{
                            .d_model = gated_delta.config.d_model,
                            .d_conv = gated_delta.config.d_conv,
                            .n_heads = gated_delta.config.n_heads,
                            .d_head = gated_delta.config.d_head,
                        },
                    );
                    errdefer gated_delta_state.deinit();

                    var gated_delta_scratch = try cpu_kernels.GatedDeltaScratch.init(
                        allocator,
                        .{
                            .d_model = gated_delta.config.d_model,
                            .d_conv = gated_delta.config.d_conv,
                            .n_heads = gated_delta.config.n_heads,
                            .d_head = gated_delta.config.d_head,
                        },
                    );
                    errdefer gated_delta_scratch.deinit();

                    var gated_delta_matmul_scratch = try compute.cpu.linalg.MatmulScratch.init(allocator);
                    errdefer gated_delta_matmul_scratch.deinit();

                    const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                    var layer_linear_bytes: usize = 0;
                    if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                    const gated_delta_state_bytes_layer = std.math.add(
                        usize,
                        gated_delta_state.conv_state.len * @sizeOf(f32),
                        gated_delta_state.ssm_state.len * @sizeOf(f32),
                    ) catch return error.InvalidArgument;
                    gated_delta_state_bytes = std.math.add(usize, gated_delta_state_bytes, gated_delta_state_bytes_layer) catch return error.InvalidArgument;

                    blocks[layer_idx].gated_delta_runtime = .{
                        .d_ff = d_ff,
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
                        .ffn_w1 = ffn_w1,
                        .ffn_w2 = ffn_w2,
                        .ffn_w3 = ffn_w3,
                        .in_proj = in_proj_dev,
                        .kernel = gated_delta_kernel,
                        .state = gated_delta_state,
                        .scratch = gated_delta_scratch,
                        .matmul_scratch = gated_delta_matmul_scratch,
                    };
                    blocks[layer_idx].gated_delta_binding = &blocks[layer_idx].gated_delta_runtime.?;
                    CudaBackend.bindGatedDeltaNormWeights(&blocks[layer_idx], &blocks[layer_idx].gated_delta_runtime.?);
                    gated_delta_block_count += 1;
                },
                .shortconv => |shortconv| {
                    const entry = static_entry orelse {
                        log.warn("inference", "CUDA shortconv runtime missing architecture metadata", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    };
                    const program = models.registry.blockProgramFor(entry, .shortconv) orelse {
                        log.warn("inference", "CUDA shortconv runtime missing LayerOp program", .{
                            .layer = layer_idx,
                            .kind = @intFromEnum(op_types.BlockKind.shortconv),
                            .architecture = entry.id,
                        });
                        return error.UnsupportedModel;
                    };
                    blocks[layer_idx].compiled_plan = try plan_compiler.compileLayerProgram(
                        allocator,
                        program,
                        .decode,
                        .{
                            .size_floor = d_model,
                            .state_descriptor_entry = entry,
                        },
                    );
                    try validateCompiledLayerPlanForCuda(&blocks[layer_idx].compiled_plan.?, layer_idx, .shortconv);
                    errdefer if (blocks[layer_idx].compiled_plan) |*compiled_plan| {
                        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
                        blocks[layer_idx].compiled_plan = null;
                    };
                    blocks[layer_idx].register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                    );
                    errdefer if (blocks[layer_idx].register_to_slot_map.len != 0) {
                        allocator.free(blocks[layer_idx].register_to_slot_map);
                        blocks[layer_idx].register_to_slot_map = &.{};
                    };
                    blocks[layer_idx].slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
                        allocator,
                        &blocks[layer_idx].compiled_plan.?,
                        blocks[layer_idx].register_to_slot_map,
                    );
                    errdefer if (blocks[layer_idx].slot_width_hints.len != 0) {
                        allocator.free(blocks[layer_idx].slot_width_hints);
                        blocks[layer_idx].slot_width_hints = &.{};
                    };
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

                    var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.in_proj, d_model, layer_idx, "conv.in_proj.weight");
                    errdefer in_proj_dev.deinit(device);
                    if (in_proj_dev.cols() != 3 * conv_dim) {
                        log.warn("inference", "CUDA shortconv in_proj dim unsupported", .{
                            .layer = layer_idx,
                            .cols = in_proj_dev.cols(),
                            .expected = 3 * conv_dim,
                        });
                        return error.UnsupportedModel;
                    }

                    var out_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.out_proj, conv_dim, layer_idx, "conv.out_proj.weight");
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

                        var w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        errdefer w1_dev.deinit(device);
                        var w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
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
                        var w2_dev = try uploadLinearWeightWithContext(device, allocator, w2, d_ff, layer_idx, "mlp.down_proj.weight");
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

                    blocks[layer_idx].shortconv_runtime = .{
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
                    blocks[layer_idx].shortconv_binding = &blocks[layer_idx].shortconv_runtime.?;
                    CudaBackend.bindShortConvNormWeights(&blocks[layer_idx], &blocks[layer_idx].shortconv_runtime.?);
                    shortconv_block_count += 1;
                },
                else => {
                    log.warn("inference", "CUDA block runtime unsupported block kind", .{
                        .layer = layer_idx,
                    });
                    return error.UnsupportedModel;
                },
            }
            try blocks[layer_idx].rebuildInstructionMetadata(allocator);
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
            .gated_delta_block_count = gated_delta_block_count,
            .q_norm_blocks = q_norm_blocks,
            .k_norm_blocks = k_norm_blocks,
            .linear_weight_bytes = linear_weight_bytes,
            .norm_weight_bytes = norm_weight_bytes,
            .kv_cache_bytes = kv_cache_bytes,
            .shortconv_state_bytes = shortconv_state_bytes,
            .gated_delta_state_bytes = gated_delta_state_bytes,
            .max_shortconv_dim = max_shortconv_dim,
            .max_gdelta_proj = max_gdelta_proj,
        };
    }

    fn deinit(self: *BlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.blocks) |*block| block.deinit(allocator, device);
        allocator.free(self.blocks);
    }

    fn maxDff(self: *const BlockRuntime) usize {
        var max_dff: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.shortconv_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
            if (layer.gated_delta_binding) |block| {
                if (block.d_ff > max_dff) max_dff = block.d_ff;
            }
        }
        return max_dff;
    }

    fn maxAttn(self: *const BlockRuntime) usize {
        var max_attn: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.q_projection_dim > max_attn) max_attn = block.q_projection_dim;
            }
        }
        return if (max_attn > 0) max_attn else self.n_heads * self.head_dim;
    }

    fn maxKv(self: *const BlockRuntime) usize {
        var max_kv: usize = 0;
        for (self.blocks) |layer| {
            if (layer.attention_binding) |block| {
                if (block.kv_dim > max_kv) max_kv = block.kv_dim;
            }
        }
        return if (max_kv > 0) max_kv else self.n_kv_heads * self.head_dim;
    }

    fn maxShortConvDim(self: *const BlockRuntime) usize {
        return self.max_shortconv_dim;
    }

    fn maxGatedDeltaProj(self: *const BlockRuntime) usize {
        return if (self.max_gdelta_proj > 0) self.max_gdelta_proj else 1;
    }
};

const KvRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

const RecurrentRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    block_runtime: *BlockRuntime,
    slot_index: usize,
};

const ShortConvRuntimeState = RecurrentRuntimeState;
const MambaRuntimeState = RecurrentRuntimeState;
const GatedDeltaRuntimeState = RecurrentRuntimeState;

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
    runtime_buffers: RuntimeBuffers,
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
    cpu_rope_global: ?*cpu_kernels.RoPE = null,
    cpu_rope_local: ?*cpu_kernels.RoPE = null,
    max_batch_size: usize = 1,
    slot_in_use: bool = false,
    slot_position: usize = 0,
    slot_rope_position_delta: isize = 0,
    slot_logits: []f32,
    state_descriptors_storage: [runtime_contract.max_state_descriptors]runtime_contract.StateDescriptor = undefined,
    state_descriptor_count: u8 = 0,
    slot_state_bindings: []SlotStateBinding = &.{},
    runtime_dispatch_counters: runtime_contract.DispatchCounters = .{},
    layer_program_dispatch_total: [256]u64 = [_]u64{0} ** 256,
    prefill_dispatch_window_start: [256]u64 = [_]u64{0} ** 256,
    layer_program_slot_buffers: []compute.cuda.Buffer = &.{},
    layer_program_slot_ptrs: []*compute.cuda.Buffer = &.{},
    layer_program_slot_widths: []usize = &.{},
    layer_program_row_capacity: usize = 1,
    argmax_index_dev: compute.cuda.Buffer,
    gated_delta_stage_input_host: []f32,
    gated_delta_stage_output_host: []f32,
    query_gate_projection_host: []f32,
    query_gate_values_host: []f32,
    qk_norm_weight_host: []f32,
    trace_checkpoint_host: []f32,
    parity_prefill_seq_len: usize,
    parity_prefill_token_index: usize,
    parity_prefill_layer_attn_norm_host: []f32,
    parity_prefill_layer_ffn_norm_host: []f32,
    parity_prefill_block_out_host: []f32,
    parity_checkpoint_warned: [256]bool,

    const max_state_bindings_per_slot: usize = runtime_contract.max_state_descriptors;

    const SlotStateBinding = struct {
        handles: [max_state_bindings_per_slot]runtime_contract.StateBlockHandle = undefined,
        count: u8 = 0,
        bound: bool = false,

        fn reset(self: *SlotStateBinding) void {
            self.count = 0;
            self.bound = false;
        }
    };

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
            .runtime_buffers = undefined,
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
            .cpu_rope_global = null,
            .cpu_rope_local = null,
            .slot_logits = undefined,
            .state_descriptors_storage = undefined,
            .state_descriptor_count = 0,
            .slot_state_bindings = &.{},
            .runtime_dispatch_counters = .{},
            .layer_program_dispatch_total = [_]u64{0} ** 256,
            .prefill_dispatch_window_start = [_]u64{0} ** 256,
            .layer_program_slot_buffers = &.{},
            .layer_program_slot_ptrs = &.{},
            .layer_program_slot_widths = &.{},
            .layer_program_row_capacity = 1,
            .argmax_index_dev = undefined,
            .gated_delta_stage_input_host = &.{},
            .gated_delta_stage_output_host = &.{},
            .query_gate_projection_host = &.{},
            .query_gate_values_host = &.{},
            .qk_norm_weight_host = &.{},
            .trace_checkpoint_host = &.{},
            .parity_prefill_seq_len = 0,
            .parity_prefill_token_index = 0,
            .parity_prefill_layer_attn_norm_host = &.{},
            .parity_prefill_layer_ffn_norm_host = &.{},
            .parity_prefill_block_out_host = &.{},
            .parity_checkpoint_warned = [_]bool{false} ** 256,
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
        try backend.initCpuRuntimeRopeHandles();
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
        backend.slot_state_bindings = try allocator.alloc(SlotStateBinding, backend.max_batch_size);
        errdefer allocator.free(backend.slot_state_bindings);
        for (backend.slot_state_bindings) |*binding| binding.* = .{};
        backend.argmax_index_dev = try backend.device.allocBuffer(@sizeOf(u32));
        errdefer backend.argmax_index_dev.deinit(&backend.device);
        backend.block_runtime = try BlockRuntime.init(allocator, &backend.device, loaded);
        errdefer backend.block_runtime.deinit(allocator, &backend.device);
        backend.assignCpuRuntimeRopeToAttentionFallbacks();
        for (backend.block_runtime.blocks) |*layer| {
            if (layer.compiled_plan) |*compiled_plan| {
                try runtime_contract.appendUniquePlanStateDescriptors(
                    backend.state_descriptors_storage[0..],
                    &backend.state_descriptor_count,
                    &compiled_plan.plan,
                );
            }
        }
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
        const max_gdelta_proj = backend.block_runtime.maxGatedDeltaProj();
        const max_shortconv_dim = backend.block_runtime.maxShortConvDim();
        backend.blas = try compute.cuda.Blas.init(&backend.device);
        errdefer backend.blas.deinit(&backend.device);
        backend.runtime_buffers = try RuntimeBuffers.init(
            allocator,
            &backend.device,
            loaded,
            max_dff,
            max_attn,
            max_kv,
            max_gdelta_proj,
            max_shortconv_dim,
            backend.max_seq_len,
            backend.n_heads,
            backend.head_dim,
        );
        errdefer backend.runtime_buffers.deinit(allocator, &backend.device);
        try backend.initLayerProgramSlotBuffers();
        errdefer backend.deinitLayerProgramSlotBuffers();
        try backend.initKernelFunctions();

        if (loaded.original_weight_dtype == .grouped_affine_u4) {
            backend.gaffine_sequence_rows_supported = smoke_checks.probeGaffineU4SequenceRowsSupport(&backend) catch false;
            if (!backend.gaffine_sequence_rows_supported) {
                log.warn("inference", "CUDA gaffine batch-rows linear degraded mode active (multi-row parity probe failed)", .{
                    .reason = "gaffine_batch_rows_probe_failed",
                });
            } else {
                backend.gaffine_sequence_fused_qkv_supported = smoke_checks.probeGaffineU4SequenceFusedQkvSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_qkv_supported) {
                    log.warn("inference", "CUDA gaffine batch-rows unfused QKV degraded mode active (multi-row fused parity probe failed)", .{
                        .reason = "gaffine_batch_rows_fused_qkv_probe_failed",
                    });
                }

                backend.gaffine_sequence_fused_gate_up_supported = smoke_checks.probeGaffineU4SequenceFusedGateUpSupport(&backend) catch false;
                if (!backend.gaffine_sequence_fused_gate_up_supported) {
                    log.warn("inference", "CUDA gaffine batch-rows unfused gate/up degraded mode active (multi-row fused parity probe failed)", .{
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
            .projected_vocab = backend.runtime_buffers.projected_vocab,
            .max_dff = backend.runtime_buffers.max_dff,
            .max_attn = backend.runtime_buffers.max_attn,
            .max_kv = backend.runtime_buffers.max_kv,
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
            .attn_score_buffers = @as(u8, @intFromBool(backend.runtime_buffers.attn_scores_dev != null and backend.runtime_buffers.attn_probs_dev != null)),
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
            .gated_delta_state_mib = bytesToMiB(backend.block_runtime.gated_delta_state_bytes),
            .prototype_mib = bytesToMiB(backend.runtime_buffers.deviceByteSize()),
            .slot_logits_mib = bytesToMiB(std.math.mul(usize, backend.slot_logits.len, @sizeOf(f32)) catch 0),
            .stream_token_select = "gpu_argmax",
            .stream_enabled = @as(u8, @intFromBool(backend.compute_stream != null)),
            .device_blocks = backend.block_runtime.blocks.len,
            .attention_blocks = backend.block_runtime.attention_block_count,
            .shortconv_blocks = backend.block_runtime.shortconv_block_count,
            .gated_delta_blocks = backend.block_runtime.gated_delta_block_count,
            .model_norm = @as(u8, @intFromBool(backend.runtime_buffers.using_model_norm)),
            .model_projection = @as(u8, @intFromBool(backend.runtime_buffers.using_model_projection)),
            .projection_lm_head = @as(u8, @intFromBool(backend.runtime_buffers.projection_from_lm_head)),
            .has_lm_head = @as(u8, @intFromBool(loaded.lm_head != null)),
            .model_embeddings = @as(u8, @intFromBool(backend.runtime_buffers.using_model_embeddings)),
            .embedding_lookup_device = @as(u8, @intFromBool(backend.runtime_buffers.embedding_lookup != null)),
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
        self.argmax_index_dev.deinit(&self.device);
        if (self.slot_state_bindings.len > 0) self.allocator.free(self.slot_state_bindings);
        if (self.query_gate_projection_host.len > 0) self.allocator.free(self.query_gate_projection_host);
        if (self.query_gate_values_host.len > 0) self.allocator.free(self.query_gate_values_host);
        if (self.qk_norm_weight_host.len > 0) self.allocator.free(self.qk_norm_weight_host);
        if (self.trace_checkpoint_host.len > 0) self.allocator.free(self.trace_checkpoint_host);
        if (self.parity_prefill_layer_attn_norm_host.len > 0) self.allocator.free(self.parity_prefill_layer_attn_norm_host);
        if (self.parity_prefill_layer_ffn_norm_host.len > 0) self.allocator.free(self.parity_prefill_layer_ffn_norm_host);
        if (self.parity_prefill_block_out_host.len > 0) self.allocator.free(self.parity_prefill_block_out_host);
        if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
        if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
        if (self.cpu_rope_local) |rope| {
            rope.deinit(self.allocator);
            self.allocator.destroy(rope);
        }
        if (self.cpu_rope_global) |rope| {
            rope.deinit(self.allocator);
            self.allocator.destroy(rope);
        }
        self.allocator.free(self.slot_logits);
        self.deinitLayerProgramSlotBuffers();
        self.block_runtime.deinit(self.allocator, &self.device);
        self.runtime_buffers.deinit(self.allocator, &self.device);
        self.blas.deinit(&self.device);
        self.kernel_arg_pack.deinit();
        self.kernel_registry.deinit();
        self.device.deinit();
        self.* = undefined;
    }

    pub fn maxBatchSize(self: *const CudaBackend) usize {
        return self.max_batch_size;
    }

    fn layerProgramRequiredSlotCount(self: *const CudaBackend) usize {
        var required: usize = 0;
        for (self.block_runtime.blocks) |layer| {
            for (layer.register_to_slot_map) |slot_idx| {
                if (slot_idx == BlockRuntimeLayer.invalid_slot) continue;
                const next = @as(usize, slot_idx) + 1;
                if (next > required) required = next;
            }
        }
        return required;
    }

    fn initLayerProgramSlotBuffers(self: *CudaBackend) !void {
        const required = self.layerProgramRequiredSlotCount();
        if (required == 0) {
            self.layer_program_slot_buffers = &.{};
            self.layer_program_slot_ptrs = &.{};
            self.layer_program_slot_widths = &.{};
            self.layer_program_row_capacity = 1;
            return;
        }

        self.layer_program_slot_widths = try self.allocator.alloc(usize, required);
        errdefer self.allocator.free(self.layer_program_slot_widths);
        @memset(self.layer_program_slot_widths, 0);
        for (self.block_runtime.blocks) |layer| {
            for (layer.slot_width_hints, 0..) |width, slot_idx| {
                if (slot_idx >= self.layer_program_slot_widths.len) continue;
                if (self.layer_program_slot_widths[slot_idx] == 0) {
                    self.layer_program_slot_widths[slot_idx] = width;
                } else if (width != 0 and self.layer_program_slot_widths[slot_idx] != width) {
                    return error.InvalidRegisterSpecSize;
                }
            }
        }
        self.layer_program_slot_buffers = try self.allocator.alloc(compute.cuda.Buffer, required);
        errdefer self.allocator.free(self.layer_program_slot_buffers);

        var initialized: usize = 0;
        errdefer {
            for (self.layer_program_slot_buffers[0..initialized]) |*buf| {
                buf.deinit(&self.device);
            }
        }

        for (0..required) |idx| {
            const width = self.layer_program_slot_widths[idx];
            if (width == 0) return error.InvalidRegisterSpecSize;
            const bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
            self.layer_program_slot_buffers[idx] = try self.device.allocBuffer(bytes);
            initialized += 1;
        }

        // Pre-allocate pointer array for execution dispatch.
        self.layer_program_slot_ptrs = try self.allocator.alloc(*compute.cuda.Buffer, required);
        for (self.layer_program_slot_buffers, 0..) |*buf, idx| {
            self.layer_program_slot_ptrs[idx] = buf;
        }
        self.layer_program_row_capacity = 1;
    }

    fn ensureLayerProgramSlotRowCapacity(self: *CudaBackend, required_rows: usize) !void {
        if (required_rows == 0) return error.InvalidArgument;
        if (required_rows <= self.layer_program_row_capacity) return;
        if (required_rows > self.max_seq_len) return error.InvalidArgument;
        if (self.layer_program_slot_buffers.len == 0) return;

        var new_capacity = self.layer_program_row_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_rows) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_rows) return error.InvalidArgument;

        for (self.layer_program_slot_buffers, 0..) |*buf, idx| {
            const width = self.layer_program_slot_widths[idx];
            const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
            try resizeScratchBuffer(&self.device, buf, std.math.mul(usize, row_bytes, new_capacity) catch return error.InvalidArgument);
        }
        self.layer_program_row_capacity = new_capacity;
    }

    fn deinitLayerProgramSlotBuffers(self: *CudaBackend) void {
        if (self.layer_program_slot_ptrs.len > 0) {
            self.allocator.free(self.layer_program_slot_ptrs);
            self.layer_program_slot_ptrs = &.{};
        }
        if (self.layer_program_slot_buffers.len == 0) return;
        for (self.layer_program_slot_buffers) |*buf| {
            buf.deinit(&self.device);
        }
        self.allocator.free(self.layer_program_slot_buffers);
        self.layer_program_slot_buffers = &.{};
        if (self.layer_program_slot_widths.len > 0) {
            self.allocator.free(self.layer_program_slot_widths);
            self.layer_program_slot_widths = &.{};
        }
        self.layer_program_row_capacity = 1;
    }

    pub fn vocabSize(self: *const CudaBackend) usize {
        return self.vocab_size;
    }

    fn initialKvCapacity(self: *const CudaBackend) usize {
        for (self.block_runtime.blocks) |layer| {
            if (layer.attention_binding) |block| return block.kv_capacity;
        }
        return 0;
    }

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
        return prefill_mod.prefill(self, tokens, logits_out);
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        try self.ensureSlotStateBlocksBoundForScheduler(0);
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
        try self.ensureSlotStateBlocksBoundForScheduler(0);
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
        const slot_index = decode_mod.allocSlot(self) orelse return null;
        self.unbindSlotStateBlocks(slot_index);
        return slot_index;
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.freeSlot(self, slot_index);
        self.unbindSlotStateBlocks(slot_index);
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        decode_mod.resetSlot(self, slot_index);
        if (self.state_descriptor_count == 0) return;
        if (!self.slotIndexSupported(slot_index)) return;
        if (!self.slot_state_bindings[slot_index].bound) return;
        self.resetShortConvStates() catch |err| {
            log.warn("inference", "CUDA resetSlot shortconv reset failed", .{
                .slot_index = slot_index,
                .reason = @errorName(err),
            });
        };
        self.resetAttentionCpuStates();
        self.resetGatedDeltaStates();
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        return decode_mod.getPosition(self, slot_index);
    }

    pub fn stateDescriptors(self: *const CudaBackend) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    fn bindRuntimeState(
        self: *CudaBackend,
        slot_index: usize,
        runtime_kind: u8,
        state_block: *runtime_contract.StateBlockHandle,
    ) !void {
        if (runtime_kind == runtime_contract.state_runtime_kind_none) {
            return;
        }
        if (runtime_kind == runtime_contract.state_runtime_kind_kv_cache) {
            const state_value = runtime_contract.stateValueFromBlock(*KvRuntimeState, state_block) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            state_value.* = .{
                .runtime_kind = runtime_kind,
                .block_runtime = &self.block_runtime,
                .slot_index = slot_index,
            };
            return;
        }
        const state_value = runtime_contract.stateValueFromBlock(*RecurrentRuntimeState, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        state_value.* = .{
            .runtime_kind = runtime_kind,
            .block_runtime = &self.block_runtime,
            .slot_index = slot_index,
        };
    }

    pub fn bindSlotStateBlocks(
        self: *CudaBackend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks) catch |err| {
            log.warn("inference", "CUDA bindSlotStateBlocks descriptor validation failed", .{
                .slot_index = slot_index,
                .state_blocks = state_blocks.len,
                .state_descriptors = self.stateDescriptors().len,
                .reason = @errorName(err),
            });
            return err;
        };
        var binding = &self.slot_state_bindings[slot_index];
        if (state_blocks.len > binding.handles.len) {
            log.warn("inference", "CUDA bindSlotStateBlocks too many state blocks", .{
                .slot_index = slot_index,
                .state_blocks = state_blocks.len,
                .capacity = binding.handles.len,
            });
            return error.InvalidStateDescriptorBinding;
        }
        for (self.stateDescriptors(), 0..) |descriptor, idx| {
            const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
                log.warn("inference", "CUDA bindSlotStateBlocks missing descriptor state id", .{
                    .slot_index = slot_index,
                    .state_id = descriptor.id,
                });
                return error.InvalidStateDescriptorBinding;
            };
            var bound = incoming.*;
            try bindRuntimeState(self, slot_index, descriptor.runtime_kind, &bound);
            binding.handles[idx] = .{
                .id = descriptor.id,
                .ptr = bound.ptr,
                .size = bound.size,
                .align_bytes = bound.align_bytes,
            };
        }
        binding.count = @intCast(state_blocks.len);
        binding.bound = true;
    }

    pub fn unbindSlotStateBlocks(self: *CudaBackend, slot_index: usize) void {
        if (!self.slotIndexSupported(slot_index)) return;
        self.slot_state_bindings[slot_index].reset();
    }

    fn slotStateBlocks(self: *const CudaBackend, slot_index: usize) []const runtime_contract.StateBlockHandle {
        const binding = &self.slot_state_bindings[slot_index];
        return binding.handles[0..binding.count];
    }

    inline fn slotIndexSupported(self: *const CudaBackend, slot_index: usize) bool {
        return slot_index < self.max_batch_size;
    }

    pub fn ensureSlotStateBlocksBoundForScheduler(self: *CudaBackend, slot_index: usize) !void {
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (self.state_descriptor_count == 0) return;
        if (!self.slot_state_bindings[slot_index].bound) return error.InvalidStateDescriptorBinding;
        try runtime_contract.validateStateBlocksForDescriptors(
            self.stateDescriptors(),
            self.slotStateBlocks(slot_index),
        );
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
        try self.ensureSlotStateBlocksBoundForScheduler(slot_index);

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
        if (!self.slot_in_use or !self.slotIndexSupported(slot_index)) {
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
        var encoded_vision_output = vision.encodeImages(vi.images) catch |err| {
            log.warn("inference", "CUDA vision encode failed", .{
                .slot_index = slot_index,
                .reason = @errorName(err),
                .images = vi.images.len,
            });
            return err;
        };
        defer encoded_vision_output.deinit(self.allocator);

        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        var deepstack_layer_features_opt: ?[]const []const f32 = null;
        if (encoded_vision_output.deepstack_layer_embeddings.len > 0) {
            image_token_positions = try collectTokenPositions(self.allocator, tokens, vi.image_token_id);
            if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;
            if (deepstackLayersCompatibleWithPrompt(
                encoded_vision_output.deepstack_layer_embeddings,
                image_token_positions.len,
                self.d_model,
            )) {
                deepstack_layer_features_opt = encoded_vision_output.deepstack_layer_embeddings;
            } else {
                log.warn("inference", "CUDA vision deepstack disabled: invalid layer feature shapes", .{
                    .slot_index = slot_index,
                    .deepstack_layers = encoded_vision_output.deepstack_layer_embeddings.len,
                    .image_positions = image_token_positions.len,
                    .d_model = self.d_model,
                });
            }
        }

        self.slot_rope_position_delta = 0;
        self.beginPrefillDispatchWindow();
        const prefill_start_ns: i128 = std.time.nanoTimestamp();
        try self.ensureKvCapacity(tokens.len);

        const hidden_count = std.math.mul(usize, tokens.len, self.d_model) catch return error.InvalidArgument;
        const hidden_host = try self.allocator.alloc(f32, hidden_count);
        defer self.allocator.free(hidden_host);

        populatePrefillHiddenFromTokens(
            self.loaded,
            tokens,
            self.d_model,
            hidden_host,
            vi.image_token_id,
        ) catch |err| {
            log.warn("inference", "CUDA vision prefill hidden population failed", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
                .reason = @errorName(err),
            });
            return err;
        };
        vision.scatterIntoHidden(
            hidden_host,
            tokens.len,
            self.d_model,
            tokens,
            vi.image_token_id,
            encoded_vision_output.merged_embeddings,
        ) catch |err| {
            log.warn("inference", "CUDA vision scatter into hidden failed", .{
                .slot_index = slot_index,
                .tokens = tokens.len,
                .reason = @errorName(err),
                .merged_embeddings = encoded_vision_output.merged_embeddings.len,
                .d_model = self.d_model,
            });
            return err;
        };

        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const row_start = std.math.mul(usize, i, self.d_model) catch return error.InvalidArgument;
            const hidden_override = hidden_host[row_start .. row_start + self.d_model];
            const download_logits = self.shouldDownloadPrefillLogitsImpl(i, tokens.len);
            const deepstack_feature_index = if (deepstack_layer_features_opt != null and image_token_positions.len > 0)
                findPositionIndex(image_token_positions, i)
            else
                null;
            self.computeGpuPrototypeLogitsWithLayerLimit(
                tokens[i],
                i,
                slot_index,
                if (download_logits) self.slot_logits else null,
                self.block_runtime.blocks.len,
                download_logits,
                download_logits,
                false,
                @intCast(i + 1),
                i,
                hidden_override,
                deepstack_layer_features_opt,
                deepstack_feature_index,
            ) catch |err| {
                log.warn("inference", "CUDA vision token prefill step failed", .{
                    .slot_index = slot_index,
                    .token_index = i,
                    .token_id = tokens[i],
                    .reason = @errorName(err),
                    .has_deepstack = @as(u8, @intFromBool(deepstack_layer_features_opt != null)),
                    .deepstack_feature_index = if (deepstack_feature_index) |idx| idx else std.math.maxInt(usize),
                });
                return err;
            };
        }

        const prefill_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - prefill_start_ns);
        self.logPrefillTimingImpl("prefill_slot_vision", tokens.len, prefill_elapsed_ns);
        @memcpy(logits_out, self.slot_logits);
        self.slot_position = tokens.len;
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
            0,
            logits_out,
            self.block_runtime.blocks.len,
            true,
            true,
            true,
            1,
            position,
            null,
            null,
            null,
        );
    }

    pub fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *CudaBackend,
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
        hidden_override: ?[]const f32,
        deepstack_layer_features_opt: ?[]const []const f32,
        deepstack_feature_index_opt: ?usize,
    ) !void {
        if (!compute_logits and download_logits) return error.InvalidArgument;
        if (deepstack_feature_index_opt != null and deepstack_layer_features_opt == null) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (download_logits) {
            const logits_out = logits_out_opt orelse return error.InvalidArgument;
            if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        }
        if (position >= self.max_seq_len) return error.InvalidArgument;
        if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;
        if (position == 0) {
            try self.resetShortConvStates();
            self.resetAttentionCpuStates();
            self.resetGatedDeltaStates();
        }
        if (ensure_kv_capacity) {
            try self.ensureKvCapacity(position + 1);
        }

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
        const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        var input_row = try bufferSlice(&self.runtime_buffers.input_dev, 0, row_bytes);
        var norm_out_row = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, row_bytes);
        const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
        const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
            self.loaded.config.rope_local_theta
        else
            global_rope_theta;

        if (hidden_override) |hidden| {
            if (hidden.len != self.d_model) return error.InvalidArgument;
            @memcpy(self.runtime_buffers.hidden_host, hidden);
            try input_row.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
        } else {
            var used_device_lookup = false;
            if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
                const lookup = &self.runtime_buffers.embedding_lookup.?;
                switch (lookup.kind) {
                    .f32 => {
                        if (embedding_lookup_f32_function) |kernel| {
                            try compute.cuda.embedding_lookup_f32.runWithFunction(
                                &self.kernel_arg_pack,
                                &self.device,
                                kernel,
                                &input_row,
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
                                &input_row,
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
                                &input_row,
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
                                        &input_row,
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
                const used_model_embeddings = tryPopulateHiddenFromToken(self.loaded, token, self.runtime_buffers.hidden_host) catch |err| switch (err) {
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
                    for (self.runtime_buffers.hidden_host) |*v| {
                        v.* *= self.loaded.config.embedding_multiplier;
                    }
                }
                try input_row.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
            }
        }

        var final_hidden = input_row;
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
            final_hidden = try self.tryExecuteLayerProgram(
                layer,
                slot_index,
                layer_idx,
                d_model_u32,
                head_dim_u32,
                rope_dim_u32,
                n_heads_u32,
                n_kv_heads_u32,
                1,
                seq_len_u32,
                trace_seq_len_u32,
                trace_pos_offset,
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
            // Deepstack: per-request feature vector addition between layer program
            // dispatches. Operates outside the per-instruction adapter table — same
            // pattern as embedding lookup and final logit projection.
            if (deepstack_layer_features_opt) |deepstack_layer_features| {
                if (deepstack_feature_index_opt) |deepstack_feature_index| {
                    if (layer_idx < deepstack_layer_features.len) {
                        const layer_features = deepstack_layer_features[layer_idx];
                        const feature_rows = std.math.divExact(usize, layer_features.len, self.d_model) catch {
                            log.warn("inference", "CUDA deepstack add skipped: invalid layer feature stride", .{
                                .layer_index = layer_idx,
                                .feature_len = layer_features.len,
                                .d_model = self.d_model,
                            });
                            continue;
                        };
                        if (deepstack_feature_index >= feature_rows) {
                            log.warn("inference", "CUDA deepstack add skipped: feature row index out of range", .{
                                .layer_index = layer_idx,
                                .feature_index = deepstack_feature_index,
                                .feature_rows = feature_rows,
                            });
                            continue;
                        }
                        const row_start = std.math.mul(usize, deepstack_feature_index, self.d_model) catch {
                            log.warn("inference", "CUDA deepstack add skipped: row offset overflow", .{
                                .layer_index = layer_idx,
                                .feature_index = deepstack_feature_index,
                                .d_model = self.d_model,
                            });
                            continue;
                        };
                        const feature_row = layer_features[row_start .. row_start + self.d_model];
                        try self.runtime_buffers.deepstack_add_dev.upload(&self.device, std.mem.sliceAsBytes(feature_row));
                        try compute.cuda.vector_add.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            vector_add_function,
                            &self.runtime_buffers.input_dev,
                            &self.runtime_buffers.deepstack_add_dev,
                            &self.runtime_buffers.input_dev,
                            d_model_u32,
                        );
                        final_hidden = self.runtime_buffers.input_dev;
                    }
                }
            }
        }

        if (!compute_logits) return;

        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &final_hidden,
            &self.runtime_buffers.norm_weight_dev,
            &norm_out_row,
            1,
            @intCast(self.d_model),
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
        if (trace.isEnabled()) {
            try norm_out_row.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
            trace.emitFinal(
                .final_norm,
                @intCast(position),
                1,
                @ptrCast(self.runtime_buffers.hidden_host.ptr),
                .f32,
                .{ @intCast(self.d_model), 0, 0, 0 },
                1,
                "cuda_final_norm_host",
            );
        }

        try self.linearForwardRows(&norm_out_row, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
        if (!download_logits) return;

        const logits_out = logits_out_opt.?;
        try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
        if (trace.isEnabled()) {
            const rows128: u128 = 1;
            const d_model128: u128 = @intCast(self.d_model);
            const vocab128: u128 = @intCast(self.runtime_buffers.projected_vocab);
            const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
            const total_bytes = saturatingU64FromU128(
                rows128 * d_model128 * @sizeOf(f32) +
                    @as(u128, self.runtime_buffers.projection_weight.byteSize()) +
                    rows128 * vocab128 * @sizeOf(f32),
            );
            const kernel_name = switch (self.runtime_buffers.projection_weight) {
                .dense_f32 => "matmul_lm_head_f32_host",
                .dense_u16 => |w| switch (w.dtype) {
                    .bf16 => "matmul_lm_head_bf16_host",
                    .f16 => "matmul_lm_head_f16_host",
                },
                .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
            };
            trace.emitFinalWithWork(
                .lm_head,
                @intCast(position),
                0,
                @ptrCast(self.runtime_buffers.projected_logits_host.ptr),
                .f32,
                .{ @intCast(self.runtime_buffers.projected_vocab), 0, 0, 0 },
                1,
                kernel_name,
                .{ .flops = total_flops, .bytes = total_bytes },
            );
        }

        if (self.runtime_buffers.projected_vocab == logits_out.len) {
            @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
        } else {
            @memset(logits_out, -1.0e9);
            @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
        }
        if (self.loaded.config.logits_scaling != 1.0) {
            for (logits_out) |*v| {
                v.* /= self.loaded.config.logits_scaling;
            }
            if (trace.isEnabled()) {
                trace.emitFinal(
                    .logits_scaled,
                    @intCast(position),
                    0,
                    @ptrCast(logits_out.ptr),
                    .f32,
                    .{ @intCast(self.vocab_size), 0, 0, 0 },
                    1,
                    null,
                );
            }
        }
    }

    pub fn computeGpuPrototypePrefillLogitsWithLayerLimit(
        self: *CudaBackend,
        tokens: []const u32,
        slot_index: usize,
        logits_out: []f32,
        layer_limit: usize,
    ) !void {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;
        if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;

        const rows = tokens.len;
        const hidden_count = std.math.mul(usize, rows, self.d_model) catch return error.InvalidArgument;
        const hidden_host = try self.allocator.alloc(f32, hidden_count);
        defer self.allocator.free(hidden_host);

        try populatePrefillHiddenFromTokens(self.loaded, tokens, self.d_model, hidden_host, null);
        try self.runtime_buffers.ensureRowCapacity(&self.device, rows);
        try self.ensureLayerProgramSlotRowCapacity(rows);
        try self.ensureKvCapacity(rows);
        try self.resetShortConvStates();
        self.resetAttentionCpuStates();
        self.resetGatedDeltaStates();
        try self.runtime_buffers.input_dev.upload(&self.device, std.mem.sliceAsBytes(hidden_host));

        const d_model_u32: u32 = @intCast(self.d_model);
        const head_dim_u32: u32 = @intCast(self.head_dim);
        const rope_dim_u32: u32 = @intCast(self.rope_dim);
        const n_heads_u32: u32 = @intCast(self.n_heads);
        const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(rows);
        const last_position = rows - 1;
        const last_position_u32: u32 = @intCast(last_position);
        const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
        const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
            self.loaded.config.rope_local_theta
        else
            global_rope_theta;
        const attention_kernels = AttentionKernelSet{
            .attn_scores_heads_f32_function = if (kv_cache_dtype_fp16)
                null
            else
                (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable),
            .attn_weighted_sum_heads_f32_function = if (kv_cache_dtype_fp16)
                null
            else
                (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable),
            .attn_scores_heads_f16_kv_function = if (kv_cache_dtype_fp16)
                (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
            else
                null,
            .softmax_rows_function = self.softmax_rows_function,
            .attn_weighted_sum_heads_f16_kv_function = if (kv_cache_dtype_fp16)
                (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
            else
                null,
            .attn_fused_heads_f16_kv_function = if (kv_cache_dtype_fp16)
                self.attn_fused_heads_f16_kv_function
            else
                null,
        };

        var final_hidden_rows = self.runtime_buffers.input_dev;
        var layer_idx: usize = 0;
        while (layer_idx < layer_limit) : (layer_idx += 1) {
            const layer = &self.block_runtime.blocks[layer_idx];
            final_hidden_rows = try self.tryExecuteLayerProgram(
                layer,
                slot_index,
                layer_idx,
                d_model_u32,
                head_dim_u32,
                rope_dim_u32,
                n_heads_u32,
                n_kv_heads_u32,
                active_rows_u32,
                seq_len_u32,
                seq_len_u32,
                0,
                last_position,
                last_position_u32,
                global_rope_theta,
                local_rope_theta,
                self.rope_function orelse return error.CudaKernelUnavailable,
                self.copy_function orelse return error.CudaKernelUnavailable,
                if (kv_cache_dtype_fp16)
                    (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
                else
                    null,
                if (kv_cache_dtype_fp16) self.kv_write_f16_function else null,
                if (kv_cache_dtype_fp16) self.rope_store_f16_function else null,
                self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                attention_kernels,
            );
        }

        const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
        const last_offset = std.math.mul(usize, last_position, row_bytes) catch return error.InvalidArgument;
        var last_hidden = try bufferSlice(&final_hidden_rows, last_offset, row_bytes);
        var last_norm = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, row_bytes);
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &last_hidden,
            &self.runtime_buffers.norm_weight_dev,
            &last_norm,
            1,
            @intCast(self.d_model),
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
        if (trace.isEnabled()) {
            try last_norm.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
            trace.emitFinal(
                .final_norm,
                @intCast(last_position),
                1,
                @ptrCast(self.runtime_buffers.hidden_host.ptr),
                .f32,
                .{ @intCast(self.d_model), 0, 0, 0 },
                1,
                "cuda_final_norm_host",
            );
        }

        try self.linearForwardRows(&last_norm, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
        try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
        if (trace.isEnabled()) {
            const rows128: u128 = 1;
            const d_model128: u128 = @intCast(self.d_model);
            const vocab128: u128 = @intCast(self.runtime_buffers.projected_vocab);
            const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
            const total_bytes = saturatingU64FromU128(
                rows128 * d_model128 * @sizeOf(f32) +
                    @as(u128, self.runtime_buffers.projection_weight.byteSize()) +
                    rows128 * vocab128 * @sizeOf(f32),
            );
            const kernel_name = switch (self.runtime_buffers.projection_weight) {
                .dense_f32 => "matmul_lm_head_f32_host",
                .dense_u16 => |w| switch (w.dtype) {
                    .bf16 => "matmul_lm_head_bf16_host",
                    .f16 => "matmul_lm_head_f16_host",
                },
                .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
            };
            trace.emitFinalWithWork(
                .lm_head,
                @intCast(last_position),
                0,
                @ptrCast(self.runtime_buffers.projected_logits_host.ptr),
                .f32,
                .{ @intCast(self.runtime_buffers.projected_vocab), 0, 0, 0 },
                1,
                kernel_name,
                .{ .flops = total_flops, .bytes = total_bytes },
            );
        }

        if (self.runtime_buffers.projected_vocab == logits_out.len) {
            @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
        } else {
            @memset(logits_out, -1.0e9);
            @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
        }
        if (self.loaded.config.logits_scaling != 1.0) {
            for (logits_out) |*v| {
                v.* /= self.loaded.config.logits_scaling;
            }
            if (trace.isEnabled()) {
                trace.emitFinal(
                    .logits_scaled,
                    @intCast(last_position),
                    0,
                    @ptrCast(logits_out.ptr),
                    .f32,
                    .{ @intCast(self.vocab_size), 0, 0, 0 },
                    1,
                    null,
                );
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
            const block = layer.attention_binding orelse continue;
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
            const block = layer.shortconv_binding orelse continue;
            const elems = std.math.divExact(usize, block.conv_state.size, @sizeOf(f32)) catch return error.InvalidArgument;
            const zeros = try self.allocator.alloc(f32, elems);
            defer self.allocator.free(zeros);
            @memset(zeros, 0.0);
            try block.conv_state.upload(&self.device, std.mem.sliceAsBytes(zeros));
        }
    }

    fn resetGatedDeltaStates(self: *CudaBackend) void {
        for (self.block_runtime.blocks) |*layer| {
            const block = layer.gated_delta_binding orelse continue;
            block.state.reset();
        }
    }

    fn resetAttentionCpuStates(self: *CudaBackend) void {
        for (self.block_runtime.blocks) |*layer| {
            const block = layer.attention_binding orelse continue;
            if (block.cpu_cache) |*cache| cache.resetCache();
        }
    }

    fn ensureGatedDeltaHostStageCapacity(self: *CudaBackend, elements: usize) !void {
        if (elements == 0) return error.InvalidArgument;
        if (self.gated_delta_stage_input_host.len < elements) {
            if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
            self.gated_delta_stage_input_host = try self.allocator.alloc(f32, elements);
        }
        if (self.gated_delta_stage_output_host.len < elements) {
            if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
            self.gated_delta_stage_output_host = try self.allocator.alloc(f32, elements);
        }
    }

    fn ensureQueryGateHostStageCapacity(self: *CudaBackend, projection_elements: usize, value_elements: usize) !void {
        if (projection_elements == 0 or value_elements == 0) return error.InvalidArgument;
        if (self.query_gate_projection_host.len < projection_elements) {
            if (self.query_gate_projection_host.len > 0) self.allocator.free(self.query_gate_projection_host);
            self.query_gate_projection_host = try self.allocator.alloc(f32, projection_elements);
        }
        if (self.query_gate_values_host.len < value_elements) {
            if (self.query_gate_values_host.len > 0) self.allocator.free(self.query_gate_values_host);
            self.query_gate_values_host = try self.allocator.alloc(f32, value_elements);
        }
    }

    fn ensureQkNormWeightHostCapacity(self: *CudaBackend, weight_elements: usize) !void {
        if (weight_elements == 0) return error.InvalidArgument;
        if (self.qk_norm_weight_host.len < weight_elements) {
            if (self.qk_norm_weight_host.len > 0) self.allocator.free(self.qk_norm_weight_host);
            self.qk_norm_weight_host = try self.allocator.alloc(f32, weight_elements);
        }
    }

    fn compactQueryGateProjectionInPlace(
        self: *CudaBackend,
        seq_len: usize,
        q_dim: usize,
        q_projection_dim: usize,
    ) !void {
        const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
        const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
        const projection_bytes = std.math.mul(usize, projection_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        const query_bytes = std.math.mul(usize, query_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        try self.ensureQueryGateHostStageCapacity(projection_elements, query_elements);
        var q_projection_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, projection_bytes);
        try q_projection_stage.download(&self.device, std.mem.sliceAsBytes(self.query_gate_projection_host[0..projection_elements]));
        try compute.cpu.gated_attention.compactQueryProjection(
            self.query_gate_projection_host[0..projection_elements],
            self.query_gate_values_host[0..query_elements],
            seq_len,
            q_dim,
            q_projection_dim,
            self.n_heads,
            self.head_dim,
        );
        var q_values_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, query_bytes);
        try q_values_stage.upload(&self.device, std.mem.sliceAsBytes(self.query_gate_values_host[0..query_elements]));
    }

    fn applyQueryGateToContextInPlace(
        self: *CudaBackend,
        seq_len: usize,
        q_dim: usize,
        q_projection_dim: usize,
    ) !void {
        const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
        const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
        const context_bytes = std.math.mul(usize, query_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        try self.ensureQueryGateHostStageCapacity(projection_elements, query_elements);
        var context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_bytes);
        try context_stage.download(&self.device, std.mem.sliceAsBytes(self.query_gate_values_host[0..query_elements]));
        try compute.cpu.gated_attention.applyOutputGateInPlace(
            self.query_gate_values_host[0..query_elements],
            self.query_gate_projection_host[0..projection_elements],
            seq_len,
            q_dim,
            q_projection_dim,
            self.n_heads,
            self.head_dim,
        );
        try context_stage.upload(&self.device, std.mem.sliceAsBytes(self.query_gate_values_host[0..query_elements]));
    }

    fn applyQueryGateQkNormCpuInPlace(
        self: *CudaBackend,
        q_norm_weight: ?*const DeviceTensor,
        k_norm_weight: ?*const DeviceTensor,
        stage_rows: usize,
        q_dim: usize,
        kv_dim: usize,
    ) !void {
        if (stage_rows == 0) return error.InvalidArgument;
        const query_elements = std.math.mul(usize, stage_rows, q_dim) catch return error.InvalidArgument;
        const key_elements = std.math.mul(usize, stage_rows, kv_dim) catch return error.InvalidArgument;
        const query_bytes = std.math.mul(usize, query_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        const key_bytes = std.math.mul(usize, key_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        try self.ensureQueryGateHostStageCapacity(query_elements, query_elements);
        if (self.gated_delta_stage_input_host.len < key_elements) {
            if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
            self.gated_delta_stage_input_host = try self.allocator.alloc(f32, key_elements);
        }

        const query_host = self.query_gate_values_host[0..query_elements];
        const key_host = self.gated_delta_stage_input_host[0..key_elements];
        var q_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, query_bytes);
        var k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, key_bytes);
        try q_stage.download(&self.device, std.mem.sliceAsBytes(query_host));
        try k_stage.download(&self.device, std.mem.sliceAsBytes(key_host));

        if (q_norm_weight) |weight| {
            const head_dim = self.head_dim;
            try self.ensureQkNormWeightHostCapacity(head_dim);
            const weight_host = self.qk_norm_weight_host[0..head_dim];
            try weight.buffer.download(&self.device, std.mem.sliceAsBytes(weight_host));
            if (self.loaded.runtime.qk_norm_weight_offset != 0.0) {
                for (weight_host) |*v| v.* += self.loaded.runtime.qk_norm_weight_offset;
            }
            var row_idx: usize = 0;
            while (row_idx < stage_rows) : (row_idx += 1) {
                const row_base = row_idx * q_dim;
                var head_idx: usize = 0;
                while (head_idx < self.n_heads) : (head_idx += 1) {
                    const base = row_base + head_idx * head_dim;
                    compute.cpu.normalization.rmsnormInPlace(query_host[base .. base + head_dim], weight_host, self.norm_eps);
                }
            }
        }
        if (k_norm_weight) |weight| {
            const head_dim = self.head_dim;
            try self.ensureQkNormWeightHostCapacity(head_dim);
            const weight_host = self.qk_norm_weight_host[0..head_dim];
            try weight.buffer.download(&self.device, std.mem.sliceAsBytes(weight_host));
            if (self.loaded.runtime.qk_norm_weight_offset != 0.0) {
                for (weight_host) |*v| v.* += self.loaded.runtime.qk_norm_weight_offset;
            }
            var row_idx: usize = 0;
            while (row_idx < stage_rows) : (row_idx += 1) {
                const row_base = row_idx * kv_dim;
                var head_idx: usize = 0;
                while (head_idx < self.n_kv_heads) : (head_idx += 1) {
                    const base = row_base + head_idx * head_dim;
                    compute.cpu.normalization.rmsnormInPlace(key_host[base .. base + head_dim], weight_host, self.norm_eps);
                }
            }
        }

        try q_stage.upload(&self.device, std.mem.sliceAsBytes(query_host));
        try k_stage.upload(&self.device, std.mem.sliceAsBytes(key_host));
    }

    fn linearForward(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        weight: *const LinearWeight,
        out: *compute.cuda.Buffer,
    ) !void {
        return self.linearForwardRows(input, try bufferF32RowCount(input, weight.rows()), weight, out);
    }

    fn linearForwardRows(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        rows: usize,
        weight: *const LinearWeight,
        out: *compute.cuda.Buffer,
    ) !void {
        if (rows == 0) return error.InvalidArgument;
        const input_row_width = weight.rows();
        const output_row_width = weight.cols();
        const input_row_bytes = std.math.mul(usize, input_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
        const output_row_bytes = std.math.mul(usize, output_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
        const packed_input_bytes = std.math.mul(usize, rows, input_row_bytes) catch return error.InvalidArgument;
        const packed_output_bytes = std.math.mul(usize, rows, output_row_bytes) catch return error.InvalidArgument;
        const packed_rows = input.size == packed_input_bytes and out.size == packed_output_bytes;
        if (input.size < packed_input_bytes or out.size < packed_output_bytes) {
            return error.InvalidInstructionBinding;
        }

        switch (weight.*) {
            .dense_f32 => |w| {
                if (!packed_rows) {
                    var row_index: usize = 0;
                    while (row_index < rows) : (row_index += 1) {
                        var input_row = try logicalF32RowSlice(input, rows, row_index, w.rows);
                        var out_row = try logicalF32RowSlice(out, rows, row_index, w.cols);
                        try self.blas.matmulF32(
                            &self.device,
                            &input_row,
                            1,
                            w.rows,
                            &w.buffer,
                            w.cols,
                            &out_row,
                        );
                    }
                    return;
                }
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
                const kernel = switch (w.dtype) {
                    .f16 => self.matmul_f16_function orelse return error.CudaKernelUnavailable,
                    .bf16 => self.matmul_bf16_function orelse return error.CudaKernelUnavailable,
                };
                var row_index: usize = 0;
                while (row_index < rows) : (row_index += 1) {
                    var input_row = try logicalF32RowSlice(input, rows, row_index, w.rows);
                    var out_row = try logicalF32RowSlice(out, rows, row_index, w.cols);
                    try compute.cuda.matmul_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        &input_row,
                        &w.buffer,
                        &out_row,
                        1,
                        @intCast(w.rows),
                        @intCast(w.cols),
                    );
                }
                return;
            },
            .gaffine_u4 => |w| {
                const kernel = self.gaffine_u4_matvec_function orelse return error.CudaKernelUnavailable;
                if (packed_rows and (rows == 1 or self.gaffine_sequence_rows_supported)) {
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

                var row_index: usize = 0;
                while (row_index < rows) : (row_index += 1) {
                    var input_row = try logicalF32RowSlice(input, rows, row_index, w.rows);
                    var out_row = try logicalF32RowSlice(out, rows, row_index, w.cols);
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
        q_proj: *const LinearWeight,
        k_proj: *const LinearWeight,
        v_proj: *const LinearWeight,
        rows: usize,
    ) !ProjectionPath {
        if (rows == 1 and try self.tryFusedQkvForward(input, q_proj, k_proj, v_proj)) return .fused;

        const q_bytes = std.math.mul(usize, rows, q_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
        const k_bytes = std.math.mul(usize, rows, k_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
        const v_bytes = std.math.mul(usize, rows, v_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
        var q_out = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_bytes);
        var k_out = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, k_bytes);
        var v_out = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, v_bytes);
        try self.linearForwardRows(input, rows, q_proj, &q_out);
        try self.linearForwardRows(input, rows, k_proj, &k_out);
        try self.linearForwardRows(input, rows, v_proj, &v_out);
        return .unfused;
    }

    fn runGateUpProjection(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        block: *const LayerAttentionRuntime,
        rows: usize,
    ) !ProjectionPath {
        return self.runGateUpProjectionWithWeights(input, &block.w1, &block.w3, rows);
    }

    fn runGateUpProjectionWithWeights(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
        rows: usize,
    ) !ProjectionPath {
        if (rows == 1 and try self.tryFusedGateUpForward(input, gate_weight, up_weight)) return .fused;

        const gate_bytes = std.math.mul(usize, rows, gate_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
        const up_bytes = std.math.mul(usize, rows, up_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
        var gate_out = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, gate_bytes);
        var up_out = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, up_bytes);
        try self.linearForwardRows(input, rows, gate_weight, &gate_out);
        try self.linearForwardRows(input, rows, up_weight, &up_out);
        return .unfused;
    }

    fn runFfnActivationMul(self: *CudaBackend, count: u32) !void {
        if (self.loaded.config.use_gelu) {
            const gelu_mul_function = self.gelu_mul_function orelse return error.CudaKernelUnavailable;
            try compute.cuda.gelu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                gelu_mul_function,
                &self.runtime_buffers.ffn_gate_dev,
                &self.runtime_buffers.ffn_up_dev,
                &self.runtime_buffers.ffn_mul_dev,
                count,
            );
            return;
        }

        const silu_mul_function = self.silu_mul_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.silu_mul.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            silu_mul_function,
            &self.runtime_buffers.ffn_gate_dev,
            &self.runtime_buffers.ffn_up_dev,
            &self.runtime_buffers.ffn_mul_dev,
            count,
        );
    }

    fn addResidualWithModelScale(
        self: *CudaBackend,
        out: *compute.cuda.Buffer,
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
                out,
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
            out,
            self.loaded.config.residual_multiplier,
            count,
        );
    }

    fn addResidualWithScale(
        self: *CudaBackend,
        out: *compute.cuda.Buffer,
        residual: *compute.cuda.Buffer,
        branch: *compute.cuda.Buffer,
        count: u32,
        scale: layer_ops.ResidualScale,
    ) !void {
        switch (scale) {
            .residual_multiplier => return self.addResidualWithModelScale(out, residual, branch, count),
            .one => {
                const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
                return compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    residual,
                    branch,
                    out,
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
                        out,
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
                    out,
                    literal,
                    count,
                );
            },
        }
    }

    fn addResidualWithScaleRowsStrideAware(
        self: *CudaBackend,
        out: *compute.cuda.Buffer,
        residual: *compute.cuda.Buffer,
        branch: *compute.cuda.Buffer,
        rows: u32,
        cols: u32,
        scale: layer_ops.ResidualScale,
    ) !void {
        if (rows == 0 or cols == 0) return error.InvalidArgument;
        const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
        if (out.size == packed_bytes and residual.size == packed_bytes and branch.size == packed_bytes) {
            return self.addResidualWithScale(out, residual, branch, packed_count, scale);
        }
        if (out.size < packed_bytes or residual.size < packed_bytes or branch.size < packed_bytes) {
            return error.InvalidInstructionBinding;
        }

        const row_count: usize = @intCast(rows);
        if (out.size % row_count != 0 or residual.size % row_count != 0 or branch.size % row_count != 0) {
            return error.InvalidInstructionBinding;
        }
        const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
        const out_stride = out.size / row_count;
        const residual_stride = residual.size / row_count;
        const branch_stride = branch.size / row_count;
        if (out_stride < row_bytes or residual_stride < row_bytes or branch_stride < row_bytes) {
            return error.InvalidInstructionBinding;
        }

        var row_idx: usize = 0;
        while (row_idx < row_count) : (row_idx += 1) {
            const out_offset = std.math.mul(usize, row_idx, out_stride) catch return error.InvalidArgument;
            const residual_offset = std.math.mul(usize, row_idx, residual_stride) catch return error.InvalidArgument;
            const branch_offset = std.math.mul(usize, row_idx, branch_stride) catch return error.InvalidArgument;
            var out_row = try bufferSlice(out, out_offset, row_bytes);
            var residual_row = try bufferSlice(residual, residual_offset, row_bytes);
            var branch_row = try bufferSlice(branch, branch_offset, row_bytes);
            try self.addResidualWithScale(&out_row, &residual_row, &branch_row, cols, scale);
        }
    }

    fn runRmsnormRowsStrideAware(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        weight: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        rows: u32,
        cols: u32,
    ) !void {
        if (rows == 0 or cols == 0) return error.InvalidArgument;
        const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
        if (input.size < packed_bytes or output.size < packed_bytes) return error.InvalidInstructionBinding;

        const row_count: usize = @intCast(rows);
        if (input.size % row_count != 0 or output.size % row_count != 0) return error.InvalidInstructionBinding;
        const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
        const input_stride = input.size / row_count;
        const output_stride = output.size / row_count;
        if (input_stride < row_bytes or output_stride < row_bytes) return error.InvalidInstructionBinding;

        var row_idx: usize = 0;
        while (row_idx < row_count) : (row_idx += 1) {
            const input_offset = std.math.mul(usize, row_idx, input_stride) catch return error.InvalidArgument;
            const output_offset = std.math.mul(usize, row_idx, output_stride) catch return error.InvalidArgument;
            var input_row = try bufferSlice(input, input_offset, row_bytes);
            var output_row = try bufferSlice(output, output_offset, row_bytes);
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                &input_row,
                weight,
                &output_row,
                1,
                cols,
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
        }
    }

    fn programBuffer(self: *CudaBackend, reg_idx: usize, ctx: *const LayerProgramExecutionContext) ?*compute.cuda.Buffer {
        _ = self;
        if (reg_idx == 0) return @constCast(&ctx.input_view);
        if (reg_idx >= ctx.register_to_slot_map.len) return null;
        const slot_idx = ctx.register_to_slot_map[reg_idx];
        if (slot_idx == BlockRuntimeLayer.invalid_slot or slot_idx >= ctx.slot_buffers.len) return null;
        return @constCast(&ctx.slot_buffers[slot_idx]);
    }

    fn appendLayerNormWeight(layer: *BlockRuntimeLayer, weight: ?*const DeviceTensor) void {
        const value = weight orelse return;
        if (layer.norm_weight_count >= layer.norm_weights.len) return;
        layer.norm_weights[layer.norm_weight_count] = value;
        layer.norm_weight_count += 1;
    }

    fn bindAttentionNormWeights(layer: *BlockRuntimeLayer, block: *const LayerAttentionRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        appendLayerNormWeight(layer, &block.ln2_weight);
        if (block.pre_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        } else if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
        if (block.post_ffn_norm_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    fn bindShortConvNormWeights(layer: *BlockRuntimeLayer, block: *const ShortConvBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    fn bindGatedDeltaNormWeights(layer: *BlockRuntimeLayer, block: *const GatedDeltaBlockRuntime) void {
        appendLayerNormWeight(layer, &block.ln1_weight);
        if (block.ln2_weight) |*weight| {
            appendLayerNormWeight(layer, weight);
        }
    }

    fn runAttentionMixerStep(
        self: *CudaBackend,
        cfg: *const LayerAttentionExecConfig,
        k_cache: *const compute.cuda.Buffer,
        v_cache: *const compute.cuda.Buffer,
        q_proj: *const LinearWeight,
        k_proj: *const LinearWeight,
        v_proj: *const LinearWeight,
        o_proj: *const LinearWeight,
        q_norm_weight: ?*const DeviceTensor,
        k_norm_weight: ?*const DeviceTensor,
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
        const layer_rope_theta = if (cfg.sliding_window > 0) local_rope_theta else global_rope_theta;
        const stage_rows = bufferF32RowCount(input, @intCast(d_model_u32)) catch |err| {
            log.warn("inference", "CUDA attention staged row count invalid", .{
                .seq_len = seq_len_u32,
                .input_bytes = input.size,
                .d_model = d_model_u32,
                .reason = @errorName(err),
            });
            return err;
        };
        const q_stage_bytes = std.math.mul(usize, stage_rows, cfg.q_projection_dim * @sizeOf(f32)) catch return error.InvalidArgument;
        const kv_stage_bytes = std.math.mul(usize, stage_rows, cfg.kv_dim * @sizeOf(f32)) catch return error.InvalidArgument;
        const context_stage_bytes = std.math.mul(usize, stage_rows, o_proj.rows() * @sizeOf(f32)) catch return error.InvalidArgument;
        var attn_q_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_stage_bytes);
        var attn_k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, kv_stage_bytes);
        var attn_v_stage = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, kv_stage_bytes);
        var attn_context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_stage_bytes);
        _ = self.runQkvProjection(input, q_proj, k_proj, v_proj, stage_rows) catch |err| {
            log.warn("inference", "CUDA attention qkv projection failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .q_dim = cfg.q_dim,
                .q_projection_dim = cfg.q_projection_dim,
                .kv_dim = cfg.kv_dim,
                .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
                .reason = @errorName(err),
            });
            return err;
        };
        if (cfg.query_gate) {
            self.compactQueryGateProjectionInPlace(
                stage_rows,
                cfg.q_dim,
                cfg.q_projection_dim,
            ) catch |err| {
                log.warn("inference", "CUDA attention query-gate compact failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .q_dim = cfg.q_dim,
                    .q_projection_dim = cfg.q_projection_dim,
                    .reason = @errorName(err),
                });
                return err;
            };
        }

        if (cfg.query_gate and (q_norm_weight != null or k_norm_weight != null)) {
            self.applyQueryGateQkNormCpuInPlace(
                q_norm_weight,
                k_norm_weight,
                stage_rows,
                cfg.q_dim,
                cfg.kv_dim,
            ) catch |err| {
                log.warn("inference", "CUDA attention query-gate qk_norm CPU stage failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .q_dim = cfg.q_dim,
                    .kv_dim = cfg.kv_dim,
                    .reason = @errorName(err),
                });
                return err;
            };
        } else {
            if (q_norm_weight) |q_norm_value| {
                const q_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_heads_u32) catch return error.InvalidArgument;
                compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                    &attn_q_stage,
                    &q_norm_value.buffer,
                    &attn_q_stage,
                    q_norm_rows,
                    head_dim_u32,
                    self.norm_eps,
                    self.loaded.runtime.qk_norm_weight_offset,
                ) catch |err| {
                    log.warn("inference", "CUDA attention q_norm failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .n_heads = n_heads_u32,
                        .head_dim = head_dim_u32,
                        .reason = @errorName(err),
                    });
                    return err;
                };
            }
            if (k_norm_weight) |k_norm_value| {
                const k_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_kv_heads_u32) catch return error.InvalidArgument;
                compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &k_norm_value.buffer,
                    &attn_k_stage,
                    k_norm_rows,
                    head_dim_u32,
                    self.norm_eps,
                    self.loaded.runtime.qk_norm_weight_offset,
                ) catch |err| {
                    log.warn("inference", "CUDA attention k_norm failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .n_kv_heads = n_kv_heads_u32,
                        .head_dim = head_dim_u32,
                        .reason = @errorName(err),
                    });
                    return err;
                };
            }
        }

        const use_fused_attention_heads_f16_kv = (!cfg.query_gate) and attention_mod.useFusedHeadsF16Kv(
            attention_policy_config,
            seq_len_u32,
            cfg.sliding_window,
            cfg.is_causal,
            head_dim_u32,
            attention_kernels.attn_fused_heads_f16_kv_function != null,
        );
        if (!use_fused_attention_heads_f16_kv) {
            compute.cuda.rope.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rope_function,
                &attn_q_stage,
                n_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            ) catch |err| {
                log.warn("inference", "CUDA attention q rope failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .position = position_u32,
                    .n_heads = n_heads_u32,
                    .head_dim = head_dim_u32,
                    .rope_dim = rope_dim_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
        }
        const use_k_write_fused = kv_cache_dtype_fp16 and (kv_write_f16_function != null or rope_store_f16_function != null);
        if (!use_k_write_fused) {
            compute.cuda.rope.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rope_function,
                &attn_k_stage,
                n_kv_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            ) catch |err| {
                log.warn("inference", "CUDA attention k rope failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .position = position_u32,
                    .n_kv_heads = n_kv_heads_u32,
                    .head_dim = head_dim_u32,
                    .rope_dim = rope_dim_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
        }

        const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
        const kv_row_bytes = std.math.mul(usize, cfg.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;
        const kv_row_offset = std.math.mul(usize, position, kv_row_bytes) catch return error.InvalidArgument;
        var k_row = bufferSlice(k_cache, kv_row_offset, kv_row_bytes) catch |err| {
            log.warn("inference", "CUDA attention k cache slice failed", .{
                .seq_len = seq_len_u32,
                .position = position_u32,
                .kv_dim = cfg.kv_dim,
                .row_offset = kv_row_offset,
                .row_bytes = kv_row_bytes,
                .cache_bytes = k_cache.size,
                .reason = @errorName(err),
            });
            return err;
        };
        var v_row = bufferSlice(v_cache, kv_row_offset, kv_row_bytes) catch |err| {
            log.warn("inference", "CUDA attention v cache slice failed", .{
                .seq_len = seq_len_u32,
                .position = position_u32,
                .kv_dim = cfg.kv_dim,
                .row_offset = kv_row_offset,
                .row_bytes = kv_row_bytes,
                .cache_bytes = v_cache.size,
                .reason = @errorName(err),
            });
            return err;
        };
        if (kv_cache_dtype_fp16) {
            if (kv_write_f16_function) |kv_write_f16| {
                compute.cuda.kv_write_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kv_write_f16,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_row,
                    &v_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                ) catch |err| {
                    log.warn("inference", "CUDA attention kv_write_f16 failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .position = position_u32,
                        .kv_dim = cfg.kv_dim,
                        .n_kv_heads = n_kv_heads_u32,
                        .head_dim = head_dim_u32,
                        .reason = @errorName(err),
                    });
                    return err;
                };
            } else if (rope_store_f16_function) |rope_store_f16| {
                compute.cuda.rope_store_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_store_f16,
                    &attn_k_stage,
                    &k_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                ) catch |err| {
                    log.warn("inference", "CUDA attention rope_store_f16 failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .position = position_u32,
                        .kv_dim = cfg.kv_dim,
                        .n_kv_heads = n_kv_heads_u32,
                        .head_dim = head_dim_u32,
                        .reason = @errorName(err),
                    });
                    return err;
                };
                compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_v_stage,
                    &v_row,
                    @intCast(cfg.kv_dim),
                ) catch |err| {
                    log.warn("inference", "CUDA attention v cast_f16 failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .position = position_u32,
                        .kv_dim = cfg.kv_dim,
                        .reason = @errorName(err),
                    });
                    return err;
                };
            } else {
                compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &k_row,
                    @intCast(cfg.kv_dim),
                ) catch |err| {
                    log.warn("inference", "CUDA attention k cast_f16 failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .position = position_u32,
                        .kv_dim = cfg.kv_dim,
                        .reason = @errorName(err),
                    });
                    return err;
                };
                compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_v_stage,
                    &v_row,
                    @intCast(cfg.kv_dim),
                ) catch |err| {
                    log.warn("inference", "CUDA attention v cast_f16 failed", .{
                        .seq_len = seq_len_u32,
                        .stage_rows = stage_rows,
                        .position = position_u32,
                        .kv_dim = cfg.kv_dim,
                        .reason = @errorName(err),
                    });
                    return err;
                };
            }
        } else {
            compute.cuda.copy.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                copy_function,
                &attn_k_stage,
                &k_row,
                @intCast(cfg.kv_dim),
            ) catch |err| {
                log.warn("inference", "CUDA attention k copy failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .position = position_u32,
                    .kv_dim = cfg.kv_dim,
                    .reason = @errorName(err),
                });
                return err;
            };
            compute.cuda.copy.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                copy_function,
                &attn_v_stage,
                &v_row,
                @intCast(cfg.kv_dim),
            ) catch |err| {
                log.warn("inference", "CUDA attention v copy failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .position = position_u32,
                    .kv_dim = cfg.kv_dim,
                    .reason = @errorName(err),
                });
                return err;
            };
        }

        const kv_groups = self.n_heads / self.n_kv_heads;
        const kv_groups_u32: u32 = @intCast(kv_groups);
        const kv_dim_u32: u32 = @intCast(cfg.kv_dim);
        _ = self.runAttentionContext(
            cfg,
            &attn_q_stage,
            &attn_context_stage,
            k_cache,
            v_cache,
            attention_kernels,
            seq_len_u32,
            head_dim_u32,
            kv_dim_u32,
            kv_groups_u32,
            rope_dim_u32,
            position_u32,
            layer_rope_theta,
        ) catch |err| {
            log.warn("inference", "CUDA attention context failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .position = position_u32,
                .q_dim = cfg.q_dim,
                .kv_dim = cfg.kv_dim,
                .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
                .reason = @errorName(err),
            });
            return err;
        };
        if (cfg.query_gate) {
            self.applyQueryGateToContextInPlace(
                stage_rows,
                cfg.q_dim,
                cfg.q_projection_dim,
            ) catch |err| {
                log.warn("inference", "CUDA attention query-gate output failed", .{
                    .seq_len = seq_len_u32,
                    .stage_rows = stage_rows,
                    .q_dim = cfg.q_dim,
                    .q_projection_dim = cfg.q_projection_dim,
                    .reason = @errorName(err),
                });
                return err;
            };
        }
        self.linearForwardRows(&attn_context_stage, stage_rows, o_proj, output) catch |err| {
            log.warn("inference", "CUDA attention output projection failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .o_proj_in_dim = o_proj.rows(),
                .o_proj_out_dim = o_proj.cols(),
                .context_bytes = attn_context_stage.size,
                .output_bytes = output.size,
                .reason = @errorName(err),
            });
            return err;
        };
    }

    fn runShortConvMixerStep(
        self: *CudaBackend,
        cfg: *const ShortConvExecConfig,
        conv_state: *compute.cuda.Buffer,
        in_proj: *const LinearWeight,
        out_proj: *const LinearWeight,
        conv_weight_time_major: *const DeviceTensor,
        conv_bias: ?*const DeviceTensor,
        input: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        shortconv_step_function: compute.cuda.Function,
    ) !void {
        const rows = try bufferF32RowCount(input, in_proj.rows());
        try self.linearForward(input, in_proj, &self.runtime_buffers.shortconv_proj_dev);
        const conv_bytes = std.math.mul(usize, cfg.conv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
        var b_gate = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, 0, conv_bytes);
        var c_gate = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, conv_bytes, conv_bytes);
        var x_proj = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, conv_bytes * 2, conv_bytes);

        try compute.cuda.shortconv_step.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            shortconv_step_function,
            &b_gate,
            &c_gate,
            &x_proj,
            conv_state,
            &conv_weight_time_major.buffer,
            if (conv_bias) |w| &w.buffer else null,
            &self.runtime_buffers.shortconv_conv_dev,
            @intCast(cfg.conv_dim),
            @intCast(cfg.d_conv),
        );
        try self.linearForwardRows(&self.runtime_buffers.shortconv_conv_dev, rows, out_proj, output);
    }

    fn downloadRowsF32StrideAware(
        self: *CudaBackend,
        src: *const compute.cuda.Buffer,
        rows: usize,
        row_width: usize,
        dst: []f32,
    ) !void {
        if (rows == 0 or row_width == 0) return error.InvalidArgument;
        const logical_elements = std.math.mul(usize, rows, row_width) catch return error.InvalidArgument;
        if (dst.len < logical_elements) return error.InvalidArgument;
        const row_bytes = std.math.mul(usize, row_width, @sizeOf(f32)) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, logical_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        if (src.size < packed_bytes) return error.InvalidInstructionBinding;

        if (rows == 1 or src.size == packed_bytes) {
            return src.download(&self.device, std.mem.sliceAsBytes(dst[0..logical_elements]));
        }

        if (src.size % rows != 0) return error.InvalidInstructionBinding;
        const src_row_stride = src.size / rows;
        if (src_row_stride < row_bytes) return error.InvalidInstructionBinding;

        var row_idx: usize = 0;
        while (row_idx < rows) : (row_idx += 1) {
            const src_offset = std.math.mul(usize, row_idx, src_row_stride) catch return error.InvalidArgument;
            var src_row = try bufferSlice(src, src_offset, row_bytes);
            const dst_start = std.math.mul(usize, row_idx, row_width) catch return error.InvalidArgument;
            const dst_row = dst[dst_start .. dst_start + row_width];
            try src_row.download(&self.device, std.mem.sliceAsBytes(dst_row));
        }
    }

    fn uploadRowsF32StrideAware(
        self: *CudaBackend,
        src: []const f32,
        rows: usize,
        row_width: usize,
        dst: *compute.cuda.Buffer,
    ) !void {
        if (rows == 0 or row_width == 0) return error.InvalidArgument;
        const logical_elements = std.math.mul(usize, rows, row_width) catch return error.InvalidArgument;
        if (src.len < logical_elements) return error.InvalidArgument;
        const row_bytes = std.math.mul(usize, row_width, @sizeOf(f32)) catch return error.InvalidArgument;
        const packed_bytes = std.math.mul(usize, logical_elements, @sizeOf(f32)) catch return error.InvalidArgument;
        if (dst.size < packed_bytes) return error.InvalidInstructionBinding;

        if (rows == 1 or dst.size == packed_bytes) {
            return dst.upload(&self.device, std.mem.sliceAsBytes(src[0..logical_elements]));
        }

        if (dst.size % rows != 0) return error.InvalidInstructionBinding;
        const dst_row_stride = dst.size / rows;
        if (dst_row_stride < row_bytes) return error.InvalidInstructionBinding;

        var row_idx: usize = 0;
        while (row_idx < rows) : (row_idx += 1) {
            const dst_offset = std.math.mul(usize, row_idx, dst_row_stride) catch return error.InvalidArgument;
            var dst_row = try bufferSlice(dst, dst_offset, row_bytes);
            const src_start = std.math.mul(usize, row_idx, row_width) catch return error.InvalidArgument;
            const src_row = src[src_start .. src_start + row_width];
            try dst_row.upload(&self.device, std.mem.sliceAsBytes(src_row));
        }
    }

    fn runGatedDeltaMixerStep(
        self: *CudaBackend,
        block: *GatedDeltaBlockRuntime,
        input: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        seq_len: usize,
    ) !void {
        const d_model = self.d_model;
        const cfg = block.kernel.config;
        const d_inner: usize = @as(usize, cfg.n_heads) * @as(usize, cfg.d_head);
        const d_conv: usize = cfg.d_conv;
        const n_v_heads: usize = cfg.n_heads;
        const d_head: usize = cfg.d_head;
        const proj_len = block.in_proj.cols();
        const qkv_len = blk: {
            const values = block.kernel.weights.conv1d_weight.asSlice(f32);
            if (values.len == 0 or d_conv == 0 or (values.len % d_conv) != 0) return error.InvalidShape;
            break :blk values.len / d_conv;
        };
        const minimum_proj = d_inner + (2 * n_v_heads);
        if (proj_len <= minimum_proj) return error.InvalidShape;
        if (block.state.conv_state.len < qkv_len * d_conv) return error.InvalidShape;

        const proj_element_count = std.math.mul(usize, seq_len, proj_len) catch return error.InvalidArgument;
        const proj_bytes = std.math.mul(usize, proj_element_count, @sizeOf(f32)) catch return error.InvalidArgument;
        try self.ensureGatedDeltaHostStageCapacity(proj_element_count);
        const proj_host = self.gated_delta_stage_input_host[0..proj_element_count];
        const output_element_count = std.math.mul(usize, seq_len, d_model) catch return error.InvalidArgument;
        if (self.gated_delta_stage_output_host.len < output_element_count) {
            if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
            self.gated_delta_stage_output_host = try self.allocator.alloc(f32, output_element_count);
        }
        const output_host = self.gated_delta_stage_output_host[0..output_element_count];

        var proj_dev = try bufferSlice(&self.runtime_buffers.gdelta_proj_dev, 0, proj_bytes);
        try self.linearForwardRows(input, seq_len, &block.in_proj, &proj_dev);

        try self.device.synchronize();
        self.downloadRowsF32StrideAware(&proj_dev, seq_len, proj_len, proj_host) catch |err| {
            log.warn("inference", "CUDA gated-delta projection download failed", .{
                .seq_len = seq_len,
                .d_model = d_model,
                .proj_len = proj_len,
                .proj_bytes = proj_dev.size,
                .stage_bytes = proj_element_count * @sizeOf(f32),
                .reason = @errorName(err),
            });
            return err;
        };
        const prev_trace_position_offset = block.kernel.trace_position_offset;
        block.kernel.trace_position_offset = if (self.parity_prefill_seq_len > 1 and seq_len == 1)
            self.parity_prefill_token_index
        else
            0;
        defer block.kernel.trace_position_offset = prev_trace_position_offset;

        const trace_pos_offset = block.kernel.trace_position_offset;
        const trace_enabled = trace.isEnabled();
        if (trace_enabled) {
            const prev_backend = trace.setBackendContext(.cuda);
            defer _ = trace.setBackendContext(prev_backend);
            for (0..seq_len) |t| {
                const proj_row = proj_host[t * proj_len ..][0..proj_len];
                trace.emit(
                    .gdelta_in_proj,
                    block.kernel.layer_idx,
                    0,
                    @intCast(trace_pos_offset + t),
                    @ptrCast(proj_row.ptr),
                    .f32,
                    .{ 1, 1, @intCast(proj_len), 0 },
                    3,
                    "cuda_gdelta_in_proj_host",
                );
            }
        }

        const conv_weight_t = block.kernel.conv_weight_transposed orelse return error.InvalidConfiguration;
        const conv_bias = if (block.kernel.weights.conv1d_bias) |bias| bias.asSlice(f32) else null;
        const A_log = block.kernel.weights.A_log.asSlice(f32);
        const dt_bias = if (block.kernel.weights.dt_bias) |bias| bias.asSlice(f32) else null;
        const norm_data = if (block.kernel.weights.norm_weight) |norm_w| norm_w.asSlice(f32) else null;

        const prev_backend = trace.setBackendContext(.cpu);
        defer _ = trace.setBackendContext(prev_backend);
        for (0..seq_len) |t| {
            const proj_row = proj_host[t * proj_len ..][0..proj_len];
            const output_row = output_host[t * d_model ..][0..d_model];
            const temp = block.scratch.getConvOutput(d_inner);
            const ssm_out = block.scratch.getSsmOutput(d_inner);

            const qkv = proj_row[0..qkv_len];
            const z = proj_row[qkv_len .. qkv_len + d_inner];
            const beta_raw = proj_row[qkv_len + d_inner .. qkv_len + d_inner + n_v_heads];
            const a_raw = proj_row[qkv_len + d_inner + n_v_heads .. qkv_len + d_inner + 2 * n_v_heads];

            if (qkv_len <= d_inner) return error.InvalidShape;
            const qk_total = qkv_len - d_inner;
            if ((qk_total % 2) != 0) return error.InvalidShape;
            const qk_inner = qk_total / 2;
            if ((qk_inner % d_head) != 0) return error.InvalidShape;
            const n_qk_heads = qk_inner / d_head;
            if (n_qk_heads == 0 or (n_v_heads % n_qk_heads) != 0) return error.InvalidShape;

            cpu_conv1d.runTimeMajorValues(qkv, block.state.conv_state, conv_weight_t, qkv, conv_bias, qkv_len, d_conv);
            if (trace_enabled) {
                trace.emit(
                    .gdelta_conv,
                    block.kernel.layer_idx,
                    0,
                    @intCast(trace_pos_offset + t),
                    @ptrCast(qkv.ptr),
                    .f32,
                    .{ 1, 1, @intCast(qkv_len), 0 },
                    3,
                    null,
                );
            }
            compute.cpu.gated_delta.applySiluInPlace(qkv);

            const query = qkv[0..qk_inner];
            const key = qkv[qk_inner .. 2 * qk_inner];
            const value = qkv[2 * qk_inner .. 2 * qk_inner + d_inner];
            try compute.cpu.gated_delta.normalizeQueryKeyInPlace(query, key, n_qk_heads, d_head);
            try compute.cpu.gated_delta.runStateSpaceStep(
                temp,
                ssm_out,
                block.state.ssm_state,
                query,
                key,
                value,
                beta_raw,
                a_raw,
                A_log,
                dt_bias,
                n_qk_heads,
                n_v_heads,
                d_head,
            );
            if (trace_enabled) {
                trace.emit(
                    .gdelta_ssm,
                    block.kernel.layer_idx,
                    0,
                    @intCast(trace_pos_offset + t),
                    @ptrCast(ssm_out.ptr),
                    .f32,
                    .{ 1, 1, @intCast(d_inner), 0 },
                    3,
                    null,
                );
            }

            for (0..n_v_heads) |head_idx| {
                const out_head = ssm_out[head_idx * d_head ..][0..d_head];
                const z_head = z[head_idx * d_head ..][0..d_head];
                const norm_head = try compute.cpu.gated_delta.normWeightSlice(norm_data, head_idx, d_head, d_inner);
                try compute.cpu.gated_delta.applyGatedRmsNormInPlace(out_head, z_head, norm_head);
            }
            if (trace_enabled) {
                trace.emit(
                    .gdelta_norm,
                    block.kernel.layer_idx,
                    0,
                    @intCast(trace_pos_offset + t),
                    @ptrCast(ssm_out.ptr),
                    .f32,
                    .{ 1, 1, @intCast(d_inner), 0 },
                    3,
                    null,
                );
            }

            var ssm_view = Tensor.view2DSlice(ssm_out, 1, d_inner);
            var out_view = Tensor.view2DSlice(output_row, 1, d_model);
            block.kernel.matmul_out_proj(&ssm_view, block.kernel.weights.out_proj, &out_view, &block.matmul_scratch);
            if (trace_enabled) {
                trace.emit(
                    .gdelta_out,
                    block.kernel.layer_idx,
                    0,
                    @intCast(trace_pos_offset + t),
                    out_view.data().ptr,
                    .f32,
                    .{ 1, 1, @intCast(d_model), 0 },
                    3,
                    null,
                );
            }
        }

        self.uploadRowsF32StrideAware(output_host, seq_len, self.d_model, output) catch |err| {
            log.warn("inference", "CUDA gated-delta output upload failed", .{
                .seq_len = seq_len,
                .d_model = d_model,
                .output_bytes = output.size,
                .stage_bytes = output_element_count * @sizeOf(f32),
                .reason = @errorName(err),
            });
            return err;
        };
        // Ensure host-produced fallback output is globally visible before
        // subsequent CUDA kernels consume this buffer.
        try self.device.synchronize();
    }

    fn runAttentionMixerStepCpu(
        self: *CudaBackend,
        block: *LayerAttentionRuntime,
        input: *const compute.cuda.Buffer,
        output: *compute.cuda.Buffer,
        seq_len: usize,
    ) !void {
        const kernel = &(block.cpu_kernel orelse return error.InvalidInstructionBinding);
        const cache = &(block.cpu_cache orelse return error.InvalidInstructionBinding);
        const scratch = &(block.cpu_scratch orelse return error.InvalidInstructionBinding);
        const matmul_scratch = &(block.cpu_matmul_scratch orelse return error.InvalidInstructionBinding);
        const element_count = std.math.mul(usize, seq_len, self.d_model) catch return error.InvalidArgument;
        try self.ensureGatedDeltaHostStageCapacity(element_count);
        const input_host = self.gated_delta_stage_input_host[0..element_count];
        const output_host = self.gated_delta_stage_output_host[0..element_count];
        try self.device.synchronize();
        try self.downloadRowsF32StrideAware(input, seq_len, self.d_model, input_host);
        kernel.position_delta = self.slot_rope_position_delta;
        var input_view = Tensor.view3DSlice(input_host, seq_len, self.d_model);
        var output_view = Tensor.view3DSlice(output_host, seq_len, self.d_model);
        const use_cache = attentionFallbackUsesCache(seq_len);
        // Attention fallback executes on CPU tensors; preserve host-readable
        // trace semantics even when wrapped by the CUDA execution route.
        const prev_backend = trace.setBackendContext(.cpu);
        defer _ = trace.setBackendContext(prev_backend);
        try kernel.forward(
            &input_view,
            &output_view,
            cache,
            scratch,
            matmul_scratch,
            use_cache,
        );
        try self.uploadRowsF32StrideAware(output_host, seq_len, self.d_model, output);
        try self.device.synchronize();
    }

    fn attentionFallbackUsesCache(seq_len: usize) bool {
        return seq_len == 1;
    }

    fn applyBiasF32(
        self: *CudaBackend,
        target: *compute.cuda.Buffer,
        bias: *const DeviceTensor,
        count: u32,
    ) !void {
        const element_count = std.math.mul(usize, bias.rows, bias.cols) catch return error.InvalidArgument;
        const count_usize: usize = @intCast(count);
        if (element_count != count_usize) return error.InvalidInstructionBinding;
        const expected_bytes = std.math.mul(usize, @as(usize, count), @sizeOf(f32)) catch return error.InvalidArgument;
        if (bias.buffer.size != expected_bytes) return error.InvalidInstructionBinding;
        const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.vector_add.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            vector_add_function,
            target,
            &bias.buffer,
            target,
            count,
        );
    }

    fn runFfnStep(
        self: *CudaBackend,
        input: *const compute.cuda.Buffer,
        rows: usize,
        gate_weight: *const LinearWeight,
        up_weight: *const LinearWeight,
        down_weight: *const LinearWeight,
        gate_bias: ?*const DeviceTensor,
        down_bias: ?*const DeviceTensor,
        d_ff: u32,
        output: *compute.cuda.Buffer,
    ) !void {
        const activation_count = std.math.mul(u32, @intCast(rows), d_ff) catch return error.InvalidArgument;
        const activation_bytes = std.math.mul(usize, @as(usize, activation_count), @sizeOf(f32)) catch return error.InvalidArgument;
        _ = try self.runGateUpProjectionWithWeights(input, gate_weight, up_weight, rows);
        if (gate_bias) |bias| {
            if (rows != 1) return error.UnsupportedModel;
            try self.applyBiasF32(&self.runtime_buffers.ffn_gate_dev, bias, d_ff);
        }
        try self.runFfnActivationMul(activation_count);
        var mul_in = try bufferSlice(&self.runtime_buffers.ffn_mul_dev, 0, activation_bytes);
        try self.linearForwardRows(&mul_in, rows, down_weight, output);
        if (down_bias) |bias| {
            if (rows != 1) return error.UnsupportedModel;
            const d_model = std.math.cast(u32, down_weight.cols()) orelse return error.InvalidArgument;
            try self.applyBiasF32(output, bias, d_model);
        }
    }

    const LayerProgramExecutionContext = struct {
        backend: *CudaBackend,
        layer: *BlockRuntimeLayer,
        slot_index: usize,
        layer_index: usize,
        op_index: usize,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        active_rows_u32: u32,
        seq_len_u32: u32,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
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
        register_to_slot_map: []const u8,
        input_view: compute.cuda.Buffer,
        slot_buffers: []compute.cuda.Buffer,
        instruction_handles: []runtime_contract.TensorHandle,
        instruction_views: []runtime_contract.TensorViewDesc,
    };

    const BuiltLayerProgramHandles = struct {
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
    };

    const LayerProgramInstructionStateBlocks = struct {
        handles: [1]runtime_contract.StateBlockHandle = undefined,
        len: usize = 0,

        fn slice(self: *LayerProgramInstructionStateBlocks) []runtime_contract.StateBlockHandle {
            return self.handles[0..self.len];
        }
    };

    const layer_program_required_opcodes = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .gated_delta_net,
        .shortconv,
        .swiglu,
        .residual_add,
    };

    const layer_program_adapter_table: runtime_contract.AdapterTable = blk: {
        var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;
        table[@intFromEnum(opcode_map.Opcode.rmsnorm)] = layerProgramNormRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramAttentionRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.gated_delta_net)] = layerProgramGatedDeltaRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramShortConvRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.swiglu)] = layerProgramSwiGluRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.residual_add)] = layerProgramResidualAddRuntimeAdapter;
        break :blk table;
    };

    const layer_program_adapter_capabilities: runtime_contract.AdapterCapabilities = blk: {
        var caps: runtime_contract.AdapterCapabilities = [_]runtime_contract.AdapterCapability{.{
            .supports_batch = false,
            .supports_graph_emit = false,
            .max_batch_size = 1,
        }} ** 256;
        for (layer_program_required_opcodes) |opcode| {
            caps[@intFromEnum(opcode)] = .{
                .supports_batch = false,
                .supports_graph_emit = false,
                .max_batch_size = 1,
            };
        }
        break :blk caps;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            layer_program_adapter_table,
            layer_program_required_opcodes,
            "cuda.engine.layer_program_adapter_table",
        );
    }

    fn layerProgramExecutionState(ctx: *runtime_contract.ExecutionContext) !*LayerProgramExecutionContext {
        const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
        return @ptrCast(@alignCast(raw_state));
    }

    fn traceShapeBsd(seq_len: u32, dim: u32) [4]u32 {
        return .{ 1, seq_len, dim, 0 };
    }

    fn traceTokenIndex(seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return seq_len - 1;
    }

    fn tracePositionForPoint(point: trace.TracePoint, pos_offset: usize, seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return switch (point) {
            .attn_q, .attn_k, .attn_v, .embed_pos => if (seq_len == 1)
                @intCast(@min(pos_offset, std.math.maxInt(u32)))
            else
                seq_len,
            .attn_qk, .attn_weights, .attn_out => if (seq_len == 1)
                @intCast(@min(pos_offset + 1, std.math.maxInt(u32)))
            else
                seq_len,
            else => seq_len,
        };
    }

    fn ensureParityPrefillBufferCapacity(self: *CudaBackend, buffer: *[]f32, elements: usize) !void {
        if (buffer.*.len >= elements) return;
        if (buffer.*.len > 0) self.allocator.free(buffer.*);
        buffer.* = try self.allocator.alloc(f32, elements);
    }

    pub fn beginParityPrefillCapture(self: *CudaBackend, seq_len: usize) !void {
        self.parity_prefill_seq_len = 0;
        self.parity_prefill_token_index = 0;
        @memset(&self.parity_checkpoint_warned, false);
        if (!trace.isEnabled() or seq_len == 0) return;
        const layer_count = self.block_runtime.blocks.len;
        const elements = std.math.mul(usize, layer_count, std.math.mul(usize, seq_len, self.d_model) catch return error.InvalidArgument) catch return error.InvalidArgument;
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_layer_attn_norm_host, elements);
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_layer_ffn_norm_host, elements);
        try self.ensureParityPrefillBufferCapacity(&self.parity_prefill_block_out_host, elements);
        self.parity_prefill_seq_len = seq_len;
    }

    pub fn endParityPrefillCapture(self: *CudaBackend) void {
        self.parity_prefill_seq_len = 0;
        self.parity_prefill_token_index = 0;
        @memset(&self.parity_checkpoint_warned, false);
    }

    fn parityPrefillBufferForPoint(self: *CudaBackend, point: trace.TracePoint) ?[]f32 {
        return switch (point) {
            .layer_attn_norm => self.parity_prefill_layer_attn_norm_host,
            .layer_ffn_norm => self.parity_prefill_layer_ffn_norm_host,
            .block_out => self.parity_prefill_block_out_host,
            else => null,
        };
    }

    fn ensureTraceCheckpointHostCapacity(self: *CudaBackend, elements: usize) !void {
        if (self.trace_checkpoint_host.len >= elements) return;
        if (self.trace_checkpoint_host.len > 0) self.allocator.free(self.trace_checkpoint_host);
        self.trace_checkpoint_host = try self.allocator.alloc(f32, elements);
    }

    fn emitLayerProgramTracePoint(
        ctx: *LayerProgramExecutionContext,
        point: trace.TracePoint,
        shape: [4]u32,
        ndim: u8,
        kernel_name: []const u8,
        output: ?*const compute.cuda.Buffer,
    ) void {
        if (!trace.isEnabled()) return;
        var marker = [_]f32{0.0};
        // CUDA timing emits may exist without a host-materialized tensor. Only mark
        // the emission as f32 after a successful host download so verification never
        // computes stats from synthetic marker storage.
        var emit_dtype: trace.DType = .u8;
        const parity_prefill_seq_len_u32: u32 = if (ctx.backend.parity_prefill_seq_len > 0)
            @intCast(ctx.backend.parity_prefill_seq_len)
        else
            ctx.trace_seq_len_u32;
        const parity_prefill_active = parity_prefill_seq_len_u32 > 1 and ctx.backend.parityPrefillBufferForPoint(point) != null;
        const logical_seq_len: u32 = ctx.trace_seq_len_u32;
        const shape_seq_len: u32 = if (parity_prefill_active)
            parity_prefill_seq_len_u32
        else
            ctx.active_rows_u32;
        const position_seq_len: u32 = if (parity_prefill_active)
            parity_prefill_seq_len_u32
        else
            ctx.seq_len_u32;
        const emission_token: u32 = if (parity_prefill_active) 0 else traceTokenIndex(logical_seq_len);
        var emit_shape = shape;
        if (ndim >= 2) emit_shape[1] = shape_seq_len;
        var data_ptr: [*]const u8 = @ptrCast(marker[0..].ptr);
        if (output) |buffer| {
            const width = @as(usize, emit_shape[ndim - 1]);
            if (parity_prefill_active) {
                if (ctx.backend.parityPrefillBufferForPoint(point)) |prefill_buffer| {
                    const parity_seq_len = ctx.backend.parity_prefill_seq_len;
                    if (parity_seq_len == 0) return;
                    if (width > 0 and ctx.layer_index < ctx.backend.block_runtime.blocks.len) {
                        const per_layer = parity_seq_len * width;
                        ctx.backend.ensureTraceCheckpointHostCapacity(per_layer) catch return;
                        const layer_host = ctx.backend.trace_checkpoint_host[0..per_layer];
                        ctx.backend.downloadRowsF32StrideAware(buffer, parity_seq_len, width, layer_host) catch |err| {
                            const warn_idx = @intFromEnum(point);
                            if (!ctx.backend.parity_checkpoint_warned[warn_idx]) {
                                ctx.backend.parity_checkpoint_warned[warn_idx] = true;
                                log.warn("inference", "CUDA parity checkpoint download failed", .{
                                    .point = point.name(),
                                    .layer_index = ctx.layer_index,
                                    .position = ctx.trace_pos_offset,
                                    .seq_len = parity_seq_len,
                                    .width = width,
                                    .element_count = per_layer,
                                    .buffer_bytes = buffer.size,
                                    .reason = @errorName(err),
                                });
                            }
                            return;
                        };
                        const dst_start = ctx.layer_index * per_layer;
                        @memcpy(prefill_buffer[dst_start .. dst_start + per_layer], layer_host);
                        data_ptr = @ptrCast(prefill_buffer[dst_start .. dst_start + per_layer].ptr);
                        emit_dtype = .f32;
                    }
                } else {
                    return;
                }
            } else {
                var element_count: usize = 1;
                for (0..ndim) |dim_idx| {
                    // Trace downloads must match the emitted checkpoint shape, not
                    // logical sequence length metadata.
                    element_count *= @as(usize, emit_shape[dim_idx]);
                }
                if (element_count > 0) {
                    ctx.backend.ensureTraceCheckpointHostCapacity(element_count) catch return;
                    const host = ctx.backend.trace_checkpoint_host[0..element_count];
                    buffer.download(&ctx.backend.device, std.mem.sliceAsBytes(host)) catch |err| {
                        const warn_idx = @intFromEnum(point);
                        if (!ctx.backend.parity_checkpoint_warned[warn_idx]) {
                            ctx.backend.parity_checkpoint_warned[warn_idx] = true;
                            log.warn("inference", "CUDA checkpoint download failed", .{
                                .point = point.name(),
                                .layer_index = ctx.layer_index,
                                .position = ctx.trace_pos_offset,
                                .element_count = element_count,
                                .buffer_bytes = buffer.size,
                                .reason = @errorName(err),
                            });
                        }
                        return;
                    };
                    data_ptr = @ptrCast(host.ptr);
                    emit_dtype = .f32;
                }
            }
        }
        trace.emit(
            point,
            @intCast(ctx.layer_index),
            emission_token,
            tracePositionForPoint(point, ctx.trace_pos_offset, position_seq_len),
            data_ptr,
            emit_dtype,
            emit_shape,
            ndim,
            kernel_name,
        );
    }

    fn inferNormTracePoint(layer: *const BlockRuntimeLayer, op_index: usize) trace.TracePoint {
        const compiled = layer.compiled_plan orelse return .layer_ffn_norm;
        if (op_index + 1 < compiled.plan.instructions.len) {
            const next_opcode = compiled.plan.instructions[op_index + 1].opcode;
            if (next_opcode == .multihead_attention or
                next_opcode == .mla_attention or
                next_opcode == .shortconv or
                next_opcode == .gated_delta_net)
            {
                return .layer_attn_norm;
            }
        }
        return .layer_ffn_norm;
    }

    fn layerProgramStateBlocksForInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !LayerProgramInstructionStateBlocks {
        var blocks = LayerProgramInstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const descriptor = runtime_contract.findStateDescriptor(&ctx.layer.compiled_plan.?.plan, state_id) orelse {
            return error.UnknownStateDescriptorId;
        };
        const slot_block = runtime_contract.findStateBlock(
            ctx.backend.slotStateBlocks(ctx.slot_index),
            state_id,
        ) orelse return error.InvalidStateDescriptorBinding;
        if (slot_block.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
        if (descriptor.size_bytes > 0 and slot_block.size < descriptor.size_bytes) return error.InvalidStateDescriptorBinding;
        blocks.handles[0] = slot_block.*;
        blocks.len = 1;
        return blocks;
    }

    fn bufferFromTensorHandle(handle: runtime_contract.TensorHandle) *compute.cuda.Buffer {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn instructionIoSlices(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !struct { inputs: []const runtime_contract.TensorHandle, outputs: []const runtime_contract.TensorHandle } {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        return .{
            .inputs = registers[0..insn.inputs.len],
            .outputs = registers[insn.inputs.len..io_count],
        };
    }

    fn instructionWeightSlice(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) ![]const runtime_contract.TensorHandle {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        const weights = registers[io_count..];
        if (weights.len != insn.weights.len) return error.InvalidWeightRefCount;
        return weights;
    }

    fn layerProgramInstructionHandleCapacity(plan: *const runtime_contract.ExecutionPlan) usize {
        var max_handles: usize = 0;
        for (plan.instructions) |insn| {
            const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
            if (handle_count > max_handles) max_handles = handle_count;
        }
        return max_handles;
    }

    fn deviceTensorFromWeightHandle(handle: runtime_contract.TensorHandle) *const DeviceTensor {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn optionalDeviceTensorFromWeightHandle(handle: runtime_contract.TensorHandle) ?*const DeviceTensor {
        const value: *const DeviceTensor = @ptrCast(@alignCast(handle.ptr));
        if (value == &missing_device_tensor) return null;
        return value;
    }

    fn linearWeightFromWeightHandle(handle: runtime_contract.TensorHandle) *const LinearWeight {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn decodeResidualScaleFromParams(params: []const runtime_contract.ParamBlock) !layer_ops.ResidualScale {
        const p = try runtime_contract.paramAs(runtime_contract.ResidualAddParam, params, .residual_add);
        return switch (p.scale_tag) {
            0 => .one,
            1 => .residual_multiplier,
            2 => .{ .literal = @bitCast(p.scale_literal) },
            else => error.InvalidParamBlockABI,
        };
    }

    fn requireLayerProgramRuntimeState(
        ctx: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !*BlockRuntimeLayer {
        const state_id = insn.state_block_id orelse return ctx.layer;
        if (runtime_contract.findStateBlock(state_blocks, state_id) == null) {
            return error.InvalidStateDescriptorBinding;
        }
        return ctx.layer;
    }

    fn requireStateValue(
        comptime T: type,
        state_blocks: []const runtime_contract.StateBlockHandle,
        state_id: u8,
    ) !*T {
        const block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        const value = runtime_contract.stateValueFromBlock(*T, block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        return value;
    }

    fn requireAttentionRuntimeBinding(state: *const KvRuntimeState, layer_index: usize) !*LayerAttentionRuntime {
        if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
        return state.block_runtime.blocks[layer_index].attention_binding orelse error.InvalidStateDescriptorBinding;
    }

    fn requireShortConvRuntimeBinding(state: *const ShortConvRuntimeState, layer_index: usize) !*ShortConvBlockRuntime {
        if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
        return state.block_runtime.blocks[layer_index].shortconv_binding orelse error.InvalidStateDescriptorBinding;
    }

    fn requireGatedDeltaRuntimeBinding(state: *const GatedDeltaRuntimeState, layer_index: usize) !*GatedDeltaBlockRuntime {
        if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
        return state.block_runtime.blocks[layer_index].gated_delta_binding orelse error.InvalidStateDescriptorBinding;
    }

    fn instructionParams(
        insn: *const runtime_contract.Instruction,
        compiled: *const runtime_contract.CompiledPlan,
        storage: *[1]runtime_contract.ParamBlock,
    ) ![]const runtime_contract.ParamBlock {
        const param_id = insn.param_block_id orelse return &.{};
        if (param_id >= compiled.param_blocks.len) return error.MissingParamBlock;
        storage[0] = compiled.param_blocks[param_id];
        return storage[0..1];
    }

    fn tensorViewDescForCudaBuffer() runtime_contract.TensorViewDesc {
        return .{
            .dtype = .f32,
            .rank = 0,
            .shape = .{ 0, 0, 0, 0 },
            .stride_elems = .{ 0, 0, 0, 0 },
            .layout = .backend_native,
        };
    }

    fn layerProgramWeightHandlePtr(ctx: *LayerProgramExecutionContext, slot_idx: usize) !*anyopaque {
        if (ctx.op_index + 1 >= ctx.layer.instruction_weight_offsets.len) return error.InvalidInstructionBinding;
        const start: usize = ctx.layer.instruction_weight_offsets[ctx.op_index];
        const end: usize = ctx.layer.instruction_weight_offsets[ctx.op_index + 1];
        if (end < start) return error.InvalidInstructionBinding;
        const count = end - start;
        if (slot_idx >= count) return error.InvalidWeightRefCount;
        const idx = start + slot_idx;
        if (idx >= ctx.layer.instruction_weight_ptrs.len) return error.InvalidInstructionBinding;
        return ctx.layer.instruction_weight_ptrs[idx] orelse error.MissingWeight;
    }

    fn buildLayerProgramInstructionHandles(
        self: *CudaBackend,
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
        handle_storage: []runtime_contract.TensorHandle,
        view_storage: []runtime_contract.TensorViewDesc,
    ) !BuiltLayerProgramHandles {
        var handle_count: usize = 0;
        var view_count: usize = 0;

        for (insn.inputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const reg_idx = runtime_contract.registerToIndex(reg);
            const input = self.programBuffer(reg_idx, ctx) orelse return error.UnsupportedModel;
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(input),
            };
            view_storage[view_count] = tensorViewDescForCudaBuffer();
            handle_count += 1;
            view_count += 1;
        }
        for (insn.outputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const reg_idx = runtime_contract.registerToIndex(reg);
            const output = self.programBuffer(reg_idx, ctx) orelse return error.UnsupportedModel;
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(output),
            };
            view_storage[view_count] = tensorViewDescForCudaBuffer();
            handle_count += 1;
            view_count += 1;
        }
        for (insn.weights, 0..) |_, slot_idx| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const weight_ptr = try layerProgramWeightHandlePtr(ctx, slot_idx);
            handle_storage[handle_count] = .{
                .register = runtime_contract.registerFromIndex(@intCast(slot_idx)),
                .ptr = weight_ptr,
            };
            view_storage[view_count] = tensorViewDescForCudaBuffer();
            handle_count += 1;
            view_count += 1;
        }

        return .{
            .registers = handle_storage[0..handle_count],
            .views = view_storage[0..view_count],
        };
    }

    fn recordLayerProgramDispatch(self: *CudaBackend, opcode: opcode_map.Opcode) void {
        const opcode_idx = @intFromEnum(opcode);
        self.layer_program_dispatch_total[opcode_idx] +%= 1;
        if (enable_dispatch_observability) {
            self.runtime_dispatch_counters.record(opcode);
        }
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

    fn initCpuRuntimeRopeHandles(self: *CudaBackend) !void {
        if (self.loaded.position_embeddings != null) return;
        if (self.rope_dim == 0) return;

        var global_freqs = try rope_scaling.materializeInverseFrequencies(
            self.allocator,
            self.rope_dim,
            self.loaded.config.rope_theta,
            self.loaded.config.rope_scaling,
        );
        defer global_freqs.deinit(self.allocator);

        const global_rope = try self.allocator.create(cpu_kernels.RoPE);
        errdefer self.allocator.destroy(global_rope);
        global_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
            self.allocator,
            self.rope_dim,
            @intCast(self.loaded.config.max_seq_len),
            global_freqs.inv_freq,
            global_freqs.attention_scaling,
        );
        self.cpu_rope_global = global_rope;

        if (self.loaded.config.rope_local_theta > 0 and self.loaded.config.sliding_window > 0) {
            var local_freqs = try rope_scaling.materializeInverseFrequencies(
                self.allocator,
                self.rope_dim,
                self.loaded.config.rope_local_theta,
                self.loaded.config.rope_scaling,
            );
            defer local_freqs.deinit(self.allocator);

            const local_rope = try self.allocator.create(cpu_kernels.RoPE);
            errdefer self.allocator.destroy(local_rope);
            local_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
                self.allocator,
                self.rope_dim,
                @intCast(self.loaded.config.max_seq_len),
                local_freqs.inv_freq,
                local_freqs.attention_scaling,
            );
            self.cpu_rope_local = local_rope;
        }
    }

    fn assignCpuRuntimeRopeToAttentionFallbacks(self: *CudaBackend) void {
        for (self.block_runtime.blocks) |*layer| {
            const block = layer.attention_binding orelse continue;
            if (block.cpu_kernel) |*kernel| {
                kernel.rope = if (kernel.sliding_window > 0 and self.cpu_rope_local != null)
                    self.cpu_rope_local
                else
                    self.cpu_rope_global;
            }
        }
    }

    fn layerProgramNormAdapter(
        self: *CudaBackend,
        _: *BlockRuntimeLayer,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 2) return error.InvalidWeightRefCount;
        const input = bufferFromTensorHandle(io.inputs[0]);
        const output = bufferFromTensorHandle(io.outputs[0]);
        const weight = deviceTensorFromWeightHandle(weight_handles[0]);
        self.runRmsnormRowsStrideAware(input, &weight.buffer, output, ctx.active_rows_u32, ctx.d_model_u32) catch |err| {
            if (err == error.InvalidArgument) {
                const expected_io_bytes = std.math.mul(usize, @as(usize, ctx.active_rows_u32), @as(usize, ctx.d_model_u32) * @sizeOf(f32)) catch 0;
                const expected_weight_bytes = std.math.mul(usize, @as(usize, ctx.d_model_u32), @sizeOf(f32)) catch 0;
                log.warn("inference", "CUDA rmsnorm adapter invalid args", .{
                    .layer_index = ctx.layer_index,
                    .op_index = ctx.op_index,
                    .input_bytes = input.size,
                    .output_bytes = output.size,
                    .weight_bytes = weight.buffer.size,
                    .expected_io_bytes = expected_io_bytes,
                    .expected_weight_bytes = expected_weight_bytes,
                    .d_model = ctx.d_model_u32,
                    .seq_len = ctx.seq_len_u32,
                });
            }
            return err;
        };
    }

    fn layerProgramAttentionAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        state_blocks: []const runtime_contract.StateBlockHandle,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 11) return error.InvalidWeightRefCount;
        const input = bufferFromTensorHandle(io.inputs[0]);
        const output = bufferFromTensorHandle(io.outputs[0]);
        const cfg = try layer.instructionAttentionRef(ctx.op_index);
        const q_proj = linearWeightFromWeightHandle(weight_handles[0]).*;
        const k_proj = linearWeightFromWeightHandle(weight_handles[1]).*;
        const v_proj = linearWeightFromWeightHandle(weight_handles[2]).*;
        const o_proj = linearWeightFromWeightHandle(weight_handles[3]).*;
        const q_norm_weight = optionalDeviceTensorFromWeightHandle(weight_handles[4]);
        const k_norm_weight = optionalDeviceTensorFromWeightHandle(weight_handles[5]);
        if (q_proj.cols() != expectedAttentionQProjectionDim(cfg)) return error.InvalidInstructionBinding;
        if (k_proj.cols() != cfg.kv_dim or v_proj.cols() != cfg.kv_dim) return error.InvalidInstructionBinding;
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const kv_state = try requireStateValue(KvRuntimeState, state_blocks, state_id);
        if (kv_state.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const attention_binding = try requireAttentionRuntimeBinding(kv_state, ctx.layer_index);
        if (attention_binding.cpu_kernel != null) {
            return self.runAttentionMixerStepCpu(
                attention_binding,
                input,
                output,
                @intCast(ctx.active_rows_u32),
            );
        }
        if (ctx.active_rows_u32 <= 1) {
            try self.runAttentionMixerStep(
                cfg,
                &attention_binding.k_cache,
                &attention_binding.v_cache,
                &q_proj,
                &k_proj,
                &v_proj,
                &o_proj,
                q_norm_weight,
                k_norm_weight,
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
            return;
        }

        var row_idx: usize = 0;
        while (row_idx < ctx.active_rows_u32) : (row_idx += 1) {
            var input_row = try logicalF32RowSlice(
                input,
                @intCast(ctx.active_rows_u32),
                row_idx,
                @intCast(ctx.d_model_u32),
            );
            var output_row = try logicalF32RowSlice(
                output,
                @intCast(ctx.active_rows_u32),
                row_idx,
                @intCast(ctx.d_model_u32),
            );
            try self.runAttentionMixerStep(
                cfg,
                &attention_binding.k_cache,
                &attention_binding.v_cache,
                &q_proj,
                &k_proj,
                &v_proj,
                &o_proj,
                q_norm_weight,
                k_norm_weight,
                &input_row,
                &output_row,
                ctx.d_model_u32,
                ctx.head_dim_u32,
                ctx.rope_dim_u32,
                ctx.n_heads_u32,
                ctx.n_kv_heads_u32,
                @intCast(row_idx + 1),
                row_idx,
                @intCast(row_idx),
                ctx.global_rope_theta,
                ctx.local_rope_theta,
                ctx.rope_function,
                ctx.copy_function,
                ctx.cast_f32_to_f16_function,
                ctx.kv_write_f16_function,
                ctx.rope_store_f16_function,
                ctx.attention_kernels,
            );
        }
    }

    fn layerProgramShortConvAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        state_blocks: []const runtime_contract.StateBlockHandle,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 4) return error.InvalidWeightRefCount;
        const input = bufferFromTensorHandle(io.inputs[0]);
        const output = bufferFromTensorHandle(io.outputs[0]);
        const cfg = try layer.instructionShortConvRef(ctx.op_index);
        const in_proj = linearWeightFromWeightHandle(weight_handles[0]).*;
        const conv_weight = deviceTensorFromWeightHandle(weight_handles[1]).*;
        const out_proj = linearWeightFromWeightHandle(weight_handles[2]).*;
        const conv_bias = optionalDeviceTensorFromWeightHandle(weight_handles[3]);
        if (in_proj.cols() != (3 * cfg.conv_dim)) return error.InvalidInstructionBinding;
        if (out_proj.cols() != self.d_model) return error.InvalidInstructionBinding;
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const shortconv_state = try requireStateValue(ShortConvRuntimeState, state_blocks, state_id);
        if (shortconv_state.runtime_kind != runtime_contract.state_runtime_kind_shortconv_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const shortconv_binding = try requireShortConvRuntimeBinding(shortconv_state, ctx.layer_index);
        if (ctx.active_rows_u32 <= 1) {
            try self.runShortConvMixerStep(
                cfg,
                &shortconv_binding.conv_state,
                &in_proj,
                &out_proj,
                &conv_weight,
                conv_bias,
                input,
                output,
                ctx.shortconv_step_function,
            );
            return;
        }

        var row_idx: usize = 0;
        while (row_idx < ctx.active_rows_u32) : (row_idx += 1) {
            var input_row = try logicalF32RowSlice(
                input,
                @intCast(ctx.active_rows_u32),
                row_idx,
                in_proj.rows(),
            );
            var output_row = try logicalF32RowSlice(
                output,
                @intCast(ctx.active_rows_u32),
                row_idx,
                out_proj.cols(),
            );
            try self.runShortConvMixerStep(
                cfg,
                &shortconv_binding.conv_state,
                &in_proj,
                &out_proj,
                &conv_weight,
                conv_bias,
                &input_row,
                &output_row,
                ctx.shortconv_step_function,
            );
        }
    }

    fn layerProgramGatedDeltaAdapter(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        state_blocks: []const runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        _ = try runtime_contract.paramAs(runtime_contract.GatedDeltaKernelParam, params, .gated_delta_net);
        const input = bufferFromTensorHandle(io.inputs[0]);
        const output = bufferFromTensorHandle(io.outputs[0]);
        const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
        const gated_delta_state = try requireStateValue(GatedDeltaRuntimeState, state_blocks, state_id);
        if (gated_delta_state.runtime_kind != runtime_contract.state_runtime_kind_gated_delta_cache) {
            return error.InvalidStateDescriptorBinding;
        }
        const binding = try requireGatedDeltaRuntimeBinding(gated_delta_state, ctx.layer_index);
        const expected_rows: usize = @intCast(ctx.active_rows_u32);
        const expected_bytes = std.math.mul(usize, expected_rows, self.d_model * @sizeOf(f32)) catch return error.InvalidArgument;
        if (input.size != expected_bytes or output.size != expected_bytes) {
            log.warn("inference", "CUDA gated-delta row count mismatch", .{
                .layer_index = ctx.layer_index,
                .op_index = ctx.op_index,
                .expected_rows = expected_rows,
                .input_bytes = input.size,
                .output_bytes = output.size,
                .expected_bytes = expected_bytes,
                .d_model = self.d_model,
            });
            return error.InvalidInstructionBinding;
        }
        try self.runGatedDeltaMixerStep(
            binding,
            input,
            output,
            expected_rows,
        );
        _ = layer;
    }

    fn layerProgramSwiGluAdapter(
        self: *CudaBackend,
        _: *BlockRuntimeLayer,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = bufferFromTensorHandle(io.inputs[0]);
        const output = bufferFromTensorHandle(io.outputs[0]);
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != 5) return error.InvalidWeightRefCount;
        const gate_weight = linearWeightFromWeightHandle(weight_handles[0]);
        const up_weight = linearWeightFromWeightHandle(weight_handles[1]);
        const down_weight = linearWeightFromWeightHandle(weight_handles[2]);
        const gate_bias = optionalDeviceTensorFromWeightHandle(weight_handles[3]);
        const down_bias = optionalDeviceTensorFromWeightHandle(weight_handles[4]);
        const d_ff = gate_weight.cols();
        if (up_weight.cols() != d_ff) return error.InvalidInstructionBinding;
        if (down_weight.rows() != d_ff) return error.InvalidInstructionBinding;
        const d_ff_u32: u32 = @intCast(d_ff);
        try self.runFfnStep(
            input,
            @intCast(ctx.active_rows_u32),
            gate_weight,
            up_weight,
            down_weight,
            gate_bias,
            down_bias,
            d_ff_u32,
            output,
        );
    }

    fn layerProgramResidualAddAdapter(
        self: *CudaBackend,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        scale: layer_ops.ResidualScale,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len < 2 or io.outputs.len == 0) return error.InvalidInstructionBinding;
        const residual_src = bufferFromTensorHandle(io.inputs[0]);
        const residual = bufferFromTensorHandle(io.outputs[0]);
        const branch = bufferFromTensorHandle(io.inputs[1]);
        try self.addResidualWithScaleRowsStrideAware(
            residual,
            residual_src,
            branch,
            ctx.active_rows_u32,
            ctx.d_model_u32,
            scale,
        );
    }

    fn layerProgramNormRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        try exec_ctx.backend.layerProgramNormAdapter(layer, insn, registers, exec_ctx);
        emitLayerProgramTracePoint(
            exec_ctx,
            inferNormTracePoint(layer, exec_ctx.op_index),
            traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
            3,
            "cuda_rmsnorm",
            bufferFromTensorHandle(io.outputs[0]),
        );
    }

    fn layerProgramAttentionRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const emits_traced_inside_cpu_fallback = if (layer.instructionAttentionRef(exec_ctx.op_index)) |cfg| blk: {
            // Query-gated attention currently runs through the traced CPU fallback inside
            // the CUDA backend. Skipping the wrapper emit here avoids duplicate attn.q/out
            // rows with one synthetic metadata record and one real host-backed record.
            if (!cfg.query_gate) {
                emitLayerProgramTracePoint(
                    exec_ctx,
                    .attn_q,
                    traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(cfg.q_dim)),
                    3,
                    "cuda_attention_q",
                    null,
                );
            }
            break :blk cfg.query_gate;
        } else |_| false;
        try exec_ctx.backend.layerProgramAttentionAdapter(layer, insn, registers, state_blocks, exec_ctx);
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        if (!emits_traced_inside_cpu_fallback) {
            emitLayerProgramTracePoint(
                exec_ctx,
                .attn_out,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
                3,
                "cuda_attention_out",
                bufferFromTensorHandle(io.outputs[0]),
            );
        }
    }

    fn layerProgramShortConvRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        if (layer.instructionShortConvRef(exec_ctx.op_index)) |cfg| {
            emitLayerProgramTracePoint(
                exec_ctx,
                .conv_in_proj,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(cfg.conv_dim * 3)),
                3,
                "cuda_shortconv_in_proj",
                null,
            );
        } else |_| {}
        try exec_ctx.backend.layerProgramShortConvAdapter(layer, insn, registers, state_blocks, exec_ctx);
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        emitLayerProgramTracePoint(
            exec_ctx,
            .conv_out_proj,
            traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
            3,
            "cuda_shortconv_out_proj",
            bufferFromTensorHandle(io.outputs[0]),
        );
    }

    fn layerProgramGatedDeltaRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        try exec_ctx.backend.layerProgramGatedDeltaAdapter(layer, insn, registers, state_blocks, params, exec_ctx);
    }

    fn layerProgramSwiGluRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        const layer = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len >= 2) {
            const gate = linearWeightFromWeightHandle(weight_handles[0]);
            emitLayerProgramTracePoint(
                exec_ctx,
                .ffn_gate,
                traceShapeBsd(exec_ctx.trace_seq_len_u32, @intCast(gate.cols())),
                3,
                "cuda_ffn_gate",
                null,
            );
        }
        try exec_ctx.backend.layerProgramSwiGluAdapter(layer, insn, registers, exec_ctx);
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        emitLayerProgramTracePoint(
            exec_ctx,
            .ffn_down,
            traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
            3,
            "cuda_ffn_down",
            bufferFromTensorHandle(io.outputs[0]),
        );
    }

    fn layerProgramResidualAddRuntimeAdapter(
        rt_ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const exec_ctx = try layerProgramExecutionState(rt_ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &exec_ctx.layer.compiled_plan.?.plan,
            state_blocks,
        );
        _ = try requireLayerProgramRuntimeState(exec_ctx, insn, state_blocks);
        const scale = try decodeResidualScaleFromParams(params);
        try exec_ctx.backend.layerProgramResidualAddAdapter(insn, registers, scale, exec_ctx);
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 2 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        emitLayerProgramTracePoint(
            exec_ctx,
            .block_out,
            traceShapeBsd(exec_ctx.trace_seq_len_u32, exec_ctx.d_model_u32),
            3,
            "cuda_residual_add",
            bufferFromTensorHandle(io.outputs[0]),
        );
    }

    fn dispatchLayerProgramInstruction(
        self: *CudaBackend,
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const adapter = layer_program_adapter_table[@intFromEnum(insn.opcode)].?;
        self.recordLayerProgramDispatch(insn.opcode);

        var active_slots: [1]usize = .{0};
        var seq_lengths: [1]u32 = .{ctx.active_rows_u32};
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = if (ctx.active_rows_u32 > 1) .prefill else .decode,
            .active_slots = active_slots[0..],
            .sequence_lengths = seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = if (enable_dispatch_observability) &self.runtime_dispatch_counters else null,
            .stream_or_queue = null,
            .workspace = .{ .any = @ptrCast(ctx) },
        };
        try runtime_contract.validateBatchCapability(
            layer_program_adapter_capabilities[@intFromEnum(insn.opcode)],
            rt_ctx.batch_size,
        );
        const built_handles = try self.buildLayerProgramInstructionHandles(
            insn,
            ctx,
            ctx.instruction_handles,
            ctx.instruction_views,
        );
        var param_storage: [1]runtime_contract.ParamBlock = undefined;
        const params = try instructionParams(insn, &ctx.layer.compiled_plan.?, &param_storage);
        var state_blocks = try layerProgramStateBlocksForInstruction(insn, ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &ctx.layer.compiled_plan.?.plan,
            state_blocks.slice(),
        );
        try adapter(
            &rt_ctx,
            insn,
            built_handles.registers,
            built_handles.views,
            state_blocks.slice(),
            params,
        );
    }

    fn tryExecuteLayerProgram(
        self: *CudaBackend,
        layer: *BlockRuntimeLayer,
        slot_index: usize,
        layer_index: usize,
        d_model_u32: u32,
        head_dim_u32: u32,
        rope_dim_u32: u32,
        n_heads_u32: u32,
        n_kv_heads_u32: u32,
        active_rows_u32: u32,
        seq_len_u32: u32,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
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
    ) !compute.cuda.Buffer {
        const prev_backend = trace.setBackendContext(.cuda);
        defer _ = trace.setBackendContext(prev_backend);
        const compiled_plan = layer.compiled_plan orelse return error.UnsupportedModel;
        const required_slot_count = blk: {
            var required: usize = 0;
            for (layer.register_to_slot_map) |slot_idx| {
                if (slot_idx == BlockRuntimeLayer.invalid_slot) continue;
                const next = @as(usize, slot_idx) + 1;
                if (next > required) required = next;
            }
            break :blk required;
        };
        if (required_slot_count > self.layer_program_slot_buffers.len) return error.UnsupportedModel;
        const handle_capacity = layerProgramInstructionHandleCapacity(&compiled_plan.plan);
        const instruction_handles = try self.allocator.alloc(runtime_contract.TensorHandle, handle_capacity);
        defer if (instruction_handles.len > 0) self.allocator.free(instruction_handles);
        const instruction_views = try self.allocator.alloc(runtime_contract.TensorViewDesc, handle_capacity);
        defer if (instruction_views.len > 0) self.allocator.free(instruction_views);
        const active_input_bytes = std.math.mul(usize, @as(usize, active_rows_u32), @as(usize, d_model_u32) * @sizeOf(f32)) catch return error.InvalidArgument;
        const input_view = try bufferSlice(&self.runtime_buffers.input_dev, 0, active_input_bytes);
        const slot_buffer_views = try self.allocator.alloc(compute.cuda.Buffer, required_slot_count);
        defer if (slot_buffer_views.len > 0) self.allocator.free(slot_buffer_views);
        for (slot_buffer_views, 0..) |*view, slot_idx| {
            const width = layer.slot_width_hints[slot_idx];
            const bytes = std.math.mul(usize, @as(usize, active_rows_u32), width * @sizeOf(f32)) catch return error.InvalidArgument;
            view.* = try bufferSlice(self.layer_program_slot_ptrs[slot_idx], 0, bytes);
        }
        var exec_ctx = LayerProgramExecutionContext{
            .backend = self,
            .layer = layer,
            .slot_index = slot_index,
            .layer_index = layer_index,
            .op_index = 0,
            .d_model_u32 = d_model_u32,
            .head_dim_u32 = head_dim_u32,
            .rope_dim_u32 = rope_dim_u32,
            .n_heads_u32 = n_heads_u32,
            .n_kv_heads_u32 = n_kv_heads_u32,
            .active_rows_u32 = active_rows_u32,
            .seq_len_u32 = seq_len_u32,
            .trace_seq_len_u32 = trace_seq_len_u32,
            .trace_pos_offset = trace_pos_offset,
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
            .register_to_slot_map = layer.register_to_slot_map,
            .input_view = input_view,
            .slot_buffers = slot_buffer_views,
            .instruction_handles = instruction_handles,
            .instruction_views = instruction_views,
        };

        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            exec_ctx.op_index = op_index;
            self.dispatchLayerProgramInstruction(&insn, &exec_ctx) catch |err| {
                log.warn("inference", "CUDA layer program instruction failed", .{
                    .layer_index = layer_index,
                    .op_index = op_index,
                    .opcode = @tagName(insn.opcode),
                    .seq_len = seq_len_u32,
                    .position = position,
                    .reason = @errorName(err),
                });
                return err;
            };
        }

        const final_register = runtime_contract.planFinalOutputRegister(&compiled_plan.plan);
        const final_register_idx = runtime_contract.registerToIndex(final_register);
        if (final_register_idx != 0) {
            const final_buf = self.programBuffer(final_register_idx, &exec_ctx) orelse return error.UnsupportedModel;
            const row_elems_usize: usize = @intCast(d_model_u32);
            const row_bytes = std.math.mul(usize, row_elems_usize, @sizeOf(f32)) catch return error.InvalidArgument;
            const slot_idx = if (final_register_idx < layer.register_to_slot_map.len)
                layer.register_to_slot_map[final_register_idx]
            else
                BlockRuntimeLayer.invalid_slot;
            const final_row_width = if (slot_idx == BlockRuntimeLayer.invalid_slot or slot_idx >= layer.slot_width_hints.len)
                row_elems_usize
            else
                layer.slot_width_hints[slot_idx];

            // Final output may live in a widened temporary slot. For multi-row prefill,
            // copy row-by-row using slot stride so later rows do not alias padding.
            if (final_row_width == row_elems_usize) {
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function,
                    final_buf,
                    &self.runtime_buffers.input_dev,
                    std.math.mul(u32, active_rows_u32, d_model_u32) catch return error.InvalidArgument,
                );
            } else {
                const src_row_stride = std.math.mul(usize, final_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
                const row_count: usize = @intCast(active_rows_u32);
                var row_idx: usize = 0;
                while (row_idx < row_count) : (row_idx += 1) {
                    const src_offset = std.math.mul(usize, row_idx, src_row_stride) catch return error.InvalidArgument;
                    const dst_offset = std.math.mul(usize, row_idx, row_bytes) catch return error.InvalidArgument;
                    var src_row = try bufferSlice(final_buf, src_offset, row_bytes);
                    var dst_row = try bufferSlice(&self.runtime_buffers.input_dev, dst_offset, row_bytes);
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function,
                        &src_row,
                        &dst_row,
                        d_model_u32,
                    );
                }
            }
            return final_buf.*;
        }
        return exec_ctx.input_view;
    }

    fn runAttentionContext(
        self: *CudaBackend,
        cfg: *const LayerAttentionExecConfig,
        q_stage: *const compute.cuda.Buffer,
        context_stage: *compute.cuda.Buffer,
        k_cache: *const compute.cuda.Buffer,
        v_cache: *const compute.cuda.Buffer,
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
        var k_cache_view = k_cache.*;
        var v_cache_view = v_cache.*;

        if (cfg.sliding_window > 0 and cfg.is_causal) {
            const window_u32 = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
            if (effective_seq_len_u32 > window_u32) {
                const kv_elem_bytes: usize = if (kv_cache_dtype_fp16) @sizeOf(u16) else @sizeOf(f32);
                const row_bytes = std.math.mul(usize, @as(usize, kv_dim_u32), kv_elem_bytes) catch return error.InvalidArgument;
                const start_row = effective_seq_len_u32 - window_u32;
                const start_offset = std.math.mul(usize, @as(usize, start_row), row_bytes) catch return error.InvalidArgument;
                k_cache_view = try bufferSlice(k_cache, start_offset, k_cache.size - start_offset);
                v_cache_view = try bufferSlice(v_cache, start_offset, v_cache.size - start_offset);
                effective_seq_len_u32 = window_u32;
            }
        }

        if (kv_cache_dtype_fp16) {
            if (!cfg.query_gate and attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                cfg.sliding_window,
                cfg.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_f16_kv_function != null,
            )) {
                try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_f16_kv_function.?,
                    q_stage,
                    &k_cache_view,
                    &v_cache_view,
                    context_stage,
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

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
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
                context_stage,
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
        const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
        const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();

        try compute.cuda.attn_scores_heads_f32.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            attn_scores_heads_f32_function,
            q_stage,
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
            context_stage,
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
        q_proj: *const LinearWeight,
        k_proj: *const LinearWeight,
        v_proj: *const LinearWeight,
    ) !bool {
        if (try self.tryFusedDenseU16QkvForward(input, q_proj, k_proj, v_proj)) return true;

        const fused_kernel = self.gaffine_u4_matvec_qkv_function orelse return false;
        const q = switch (q_proj.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const k = switch (k_proj.*) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const v = switch (v_proj.*) {
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
            &self.runtime_buffers.attn_q_dev,
            @intCast(q.cols),
            q.group_size,
            q.scales_dtype_tag,
            &k.packed_data,
            &k.scales,
            &k.biases,
            &self.runtime_buffers.attn_k_dev,
            @intCast(k.cols),
            k.group_size,
            k.scales_dtype_tag,
            &v.packed_data,
            &v.scales,
            &v.biases,
            &self.runtime_buffers.attn_v_dev,
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
        q_proj: *const LinearWeight,
        k_proj: *const LinearWeight,
        v_proj: *const LinearWeight,
    ) !bool {
        const q = switch (q_proj.*) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const k = switch (k_proj.*) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const v = switch (v_proj.*) {
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
            &self.runtime_buffers.attn_q_dev,
            @intCast(q.cols),
            &k.buffer,
            &self.runtime_buffers.attn_k_dev,
            @intCast(k.cols),
            &v.buffer,
            &self.runtime_buffers.attn_v_dev,
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
        // The grouped-affine fused gate/up path is still not numerically aligned with
        // CPU on Qwen3.5 parity verification. Use the shared unfused matvec path until
        // the fused kernel has a model-true regression guard.
        return false;
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
        const rows = bufferF32RowCount(input, gate.rows) catch return false;
        if (rows != 1) return false;

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
            &self.runtime_buffers.ffn_gate_dev,
            @intCast(gate.cols),
            &up.buffer,
            &self.runtime_buffers.ffn_up_dev,
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
    skip_token_id: ?u32,
) !void {
    if (d_model == 0) return error.InvalidArgument;
    const expected = std.math.mul(usize, tokens.len, d_model) catch return error.InvalidArgument;
    if (out.len != expected) return error.InvalidArgument;

    var idx: usize = 0;
    while (idx < tokens.len) : (idx += 1) {
        const row_start = std.math.mul(usize, idx, d_model) catch return error.InvalidArgument;
        const row = out[row_start .. row_start + d_model];
        if (skip_token_id) |skip_id| {
            if (tokens[idx] == skip_id) {
                @memset(row, 0.0);
                continue;
            }
        }
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
    if (self.runtime_buffers.projected_vocab == 0) return error.InvalidArgument;
    if (self.runtime_buffers.projected_vocab > std.math.maxInt(u32)) return error.InvalidArgument;
    if (self.loaded.config.logits_scaling < 0.0) {
        try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
        return argminHost(self.runtime_buffers.projected_logits_host);
    }

    const argmax_function = self.argmax_function orelse return error.CudaKernelUnavailable;
    const count_u32: u32 = @intCast(self.runtime_buffers.projected_vocab);
    try compute.cuda.argmax.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        argmax_function,
        &self.runtime_buffers.logits_dev,
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

fn resizeScratchBuffer(device: *compute.cuda.Device, buffer: *compute.cuda.Buffer, new_size: usize) !void {
    if (new_size == 0) return error.InvalidArgument;
    if (buffer.size == new_size) return;
    var next = try device.allocBuffer(new_size);
    errdefer next.deinit(device);
    buffer.deinit(device);
    buffer.* = next;
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
        .layer_program_gated_delta = self.prefillDispatchDelta(.gated_delta_net),
        .layer_program_ffn = self.prefillDispatchDelta(.swiglu) + self.prefillDispatchDelta(.moe),
        .layer_program_mamba = self.prefillDispatchDelta(.mamba_mixer),
        .layer_program_residual_add = self.prefillDispatchDelta(.residual_add),
        .layers = self.block_runtime.blocks.len,
        .attention_blocks = self.block_runtime.attention_block_count,
        .shortconv_blocks = self.block_runtime.shortconv_block_count,
        .gated_delta_blocks = self.block_runtime.gated_delta_block_count,
    });
}

fn deepstackLayersCompatibleWithPrompt(
    layers: []const []const f32,
    image_positions: usize,
    d_model: usize,
) bool {
    if (d_model == 0) return false;
    for (layers) |layer_features| {
        if (layer_features.len == 0) return false;
        if (layer_features.len % d_model != 0) return false;
        const feature_rows = layer_features.len / d_model;
        if (feature_rows < image_positions) return false;
    }
    return true;
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

fn uploadLinearWeightWithContext(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    layer_idx: usize,
    weight_name: []const u8,
) !LinearWeight {
    return uploadLinearWeight(device, allocator, src, input_dim) catch |err| {
        if (src.n_dims == 2) {
            log.warn("inference", "CUDA linear weight upload failed", .{
                .layer = layer_idx,
                .weight = weight_name,
                .rows = src.shape[0],
                .cols = src.shape[1],
                .input_dim = input_dim,
                .dtype = @tagName(src.dtype),
                .reason = @errorName(err),
            });
        } else {
            log.warn("inference", "CUDA linear weight upload failed", .{
                .layer = layer_idx,
                .weight = weight_name,
                .n_dims = src.n_dims,
                .input_dim = input_dim,
                .dtype = @tagName(src.dtype),
                .reason = @errorName(err),
            });
        }
        return err;
    };
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
            .state_block_id = runtime_contract.kv_cache_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.residual, layer_ops.finalOutputBuffer(&program));
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
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, layer_ops.finalOutputBuffer(&program));
}

test "layer program support envelope accepts kernel-add programs" {
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
    try std.testing.expect(
        runtime_contract.firstLayerProgramCompatibilityIssue(
            &program,
            .attention_mlp,
            CudaBackend.layer_program_adapter_table,
        ) == null,
    );
}

test "layer_program_adapter_table covers CUDA LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .gated_delta_net,
        .swiglu,
        .shortconv,
        .residual_add,
    };
    for (supported) |opcode| {
        try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode)] != null);
    }

    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mla_attention)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.moe)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mul_scalar)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.vision_patch_embed)] == null);
}

test "layer program support envelope rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mul_scalar, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope rejects CUDA-unsupported macro opcodes at load time" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .mamba_mixer,
            .state_block_id = runtime_contract.mamba_state_id,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mamba_mixer, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope is block-kind agnostic for state descriptors" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
            .state_block_id = runtime_contract.shortconv_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expect(
        runtime_contract.firstLayerProgramCompatibilityIssue(
            &program,
            .attention_mlp,
            CudaBackend.layer_program_adapter_table,
        ) == null,
    );
}

test "buildCudaLayerProgramRegisterSlotMap reuses temp slots from liveness" {
    const inputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const outputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const inputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const outputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const inputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const outputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(3)};
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = inputs0[0..],
            .outputs = outputs0[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs1[0..],
            .outputs = outputs1[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs2[0..],
            .outputs = outputs2[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };
    const kill0 = [_]u64{0b0000};
    const kill1 = [_]u64{0b0010};
    const kill2 = [_]u64{0b1100};
    const compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = instructions[0..],
            .register_count = 4,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
        },

        .liveness = .{
            .register_last_read = &.{ 0, 1, 2, 2 },
            .kill_after_instruction = &.{ kill0[0..], kill1[0..], kill2[0..] },
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };

    const map = try buildCudaLayerProgramRegisterSlotMap(std.testing.allocator, &compiled);
    defer std.testing.allocator.free(map);
    try std.testing.expect(map[1] < 2);
    try std.testing.expect(map[2] < 2);
    try std.testing.expect(map[3] < 2);
    try std.testing.expectEqual(map[1], map[3]);
    try std.testing.expect(map[2] != map[1]);
}

test "resolveDenseInOutLayout transposes [out,in] orientation" {
    const layout = try resolveDenseInOutLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseInOutLayout rejects mismatched orientation" {
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
    const removed_ops = [_][]const u8{
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

    for (removed_ops) |op| {
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
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], null);

    const expected = [_]f32{
        8.0, 10.0, 12.0,
        2.0, 4.0,  6.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens zero-fills configured skip token rows" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(tensor.ModelConfig);
    cfg.embedding_multiplier = 1.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 99, 0 };
    var out = [_]f32{0.0} ** 9;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], 99);

    const expected = [_]f32{
        4.0, 5.0, 6.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
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
        populatePrefillHiddenFromTokens(&loaded, tokens[0..], 4, out[0..], null),
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

test "deepstackLayersCompatibleWithPrompt accepts valid layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const layer0 = [_]f32{0} ** (2 * 4);
    const layer1 = [_]f32{0} ** (3 * 4);
    const layers = [_][]const f32{ layer0[0..], layer1[0..] };
    try std.testing.expect(deepstackLayersCompatibleWithPrompt(layers[0..], image_positions, d_model));
}

test "deepstackLayersCompatibleWithPrompt rejects malformed layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const too_few_rows = [_]f32{0} ** (1 * 4);
    const bad_stride = [_]f32{0} ** 7;
    const valid = [_]f32{0} ** (2 * 4);

    const layers_few = [_][]const f32{too_few_rows[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_few[0..], image_positions, d_model));

    const layers_stride = [_][]const f32{bad_stride[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_stride[0..], image_positions, d_model));

    const layers_zero_dim = [_][]const f32{valid[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_zero_dim[0..], image_positions, 0));
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

test "BlockRuntimeLayer.rebuildInstructionMetadata binds per-op runtime metadata" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const zero_buffer = std.mem.zeroes(compute.cuda.Buffer);
    const zero_tensor = DeviceTensor{
        .rows = 0,
        .cols = 0,
        .buffer = zero_buffer,
    };
    const zero_weight = LinearWeight{ .dense_f32 = zero_tensor };
    var norm0: DeviceTensor = zero_tensor;
    var norm1: DeviceTensor = zero_tensor;
    var attention_runtime: LayerAttentionRuntime = .{
        .q_dim = 0,
        .q_projection_dim = 0,
        .kv_dim = 0,
        .d_ff = 0,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
        .ln1_weight = zero_tensor,
        .ln2_weight = zero_tensor,
        .pre_ffn_norm_weight = null,
        .post_ffn_norm_weight = null,
        .q_norm_weight = null,
        .k_norm_weight = null,
        .q_proj = zero_weight,
        .k_proj = zero_weight,
        .v_proj = zero_weight,
        .o_proj = zero_weight,
        .w1 = zero_weight,
        .w2 = zero_weight,
        .w3 = zero_weight,
        .k_cache = zero_buffer,
        .v_cache = zero_buffer,
        .kv_capacity = 0,
    };
    var shortconv_runtime: ShortConvBlockRuntime = .{
        .conv_dim = 0,
        .d_conv = 0,
        .d_ff = 0,
        .ln1_weight = zero_tensor,
        .ln2_weight = null,
        .in_proj = zero_weight,
        .out_proj = zero_weight,
        .conv_weight_time_major = zero_tensor,
        .conv_bias = null,
        .conv_state = zero_buffer,
        .ffn_w1 = null,
        .ffn_w2 = null,
        .ffn_w3 = null,
    };
    const gate_weight: LinearWeight = zero_weight;
    const up_weight: LinearWeight = zero_weight;
    const down_weight: LinearWeight = zero_weight;

    layer.norm_weights[0] = &norm0;
    layer.norm_weights[1] = &norm1;
    layer.norm_weight_count = 2;
    attention_runtime.w1 = gate_weight;
    attention_runtime.w3 = up_weight;
    attention_runtime.w2 = down_weight;
    attention_runtime.d_ff = 32;
    layer.attention_binding = &attention_runtime;
    layer.shortconv_binding = &shortconv_runtime;

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
        .{ .kernel = .{ .id = 2, .in = .branch_out, .out = .tmp3, .debug_type = .shortconv, .state_block_id = runtime_contract.shortconv_state_id } },
        .{ .kernel = .{ .id = 3, .in = .tmp3, .out = .branch_out, .debug_type = .mlp } },
        .{ .kernel = .{ .id = 4, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try layer.rebuildInstructionMetadata(std.testing.allocator);

    try std.testing.expect(layer.instruction_norm_weight_slots[0].? == &norm0);
    try std.testing.expect(layer.instruction_norm_weight_slots[4].? == &norm1);
    try std.testing.expectEqual(@as(usize, ops.len + 1), layer.instruction_weight_offsets.len);
    try std.testing.expect(layer.instruction_weight_ptrs.len != 0);
    // Instruction weight pointers are flattened and directly sourced from layer bindings.
    const attn_start = layer.instruction_weight_offsets[1];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 0].?), @intFromPtr(&attention_runtime.q_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 1].?), @intFromPtr(&attention_runtime.k_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 2].?), @intFromPtr(&attention_runtime.v_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 3].?), @intFromPtr(&attention_runtime.o_proj));
    const shortconv_start = layer.instruction_weight_offsets[2];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 0].?), @intFromPtr(&shortconv_runtime.in_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 1].?), @intFromPtr(&shortconv_runtime.conv_weight_time_major));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 2].?), @intFromPtr(&shortconv_runtime.out_proj));
}

test "BlockRuntimeLayer.rebuildInstructionMetadata rejects norm op without bound norm weights" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try std.testing.expectError(error.UnsupportedModel, layer.rebuildInstructionMetadata(std.testing.allocator));
}

test "bindSlotStateBlocks stores typed runtime states by runtime_kind" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 91,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 92,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 93,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 94,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 91,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 92,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 93,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 94,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(kv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(shortconv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(mamba_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(gated_delta_state.block_runtime));
    try std.testing.expectEqual(@as(usize, 0), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks preserves bound slot index in runtime states" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 2;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 101,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 102,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 103,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 104,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [2]CudaBackend.SlotStateBinding = .{ .{}, .{} };
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 101,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 102,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 103,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 104,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(1, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(1);
    const bound = backend.slotStateBlocks(1);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 1), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks preserves opaque descriptor blocks with runtime_kind none" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 1;
    backend.state_descriptors_storage[0] = .{
        .id = 111,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 111,
            .ptr = state_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    try std.testing.expectEqual(@as(usize, 1), bound.len);
    try std.testing.expectEqual(@intFromPtr(state_blocks[0].ptr), @intFromPtr(bound[0].ptr));
    try std.testing.expectEqual(state_blocks[0].size, bound[0].size);
    try std.testing.expectEqual(state_blocks[0].align_bytes, bound[0].align_bytes);
}

test "expectedAttentionQProjectionDim uses packed query width only for query-gated attention" {
    const plain = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
    };
    try std.testing.expectEqual(@as(usize, 2048), expectedAttentionQProjectionDim(&plain));

    const gated = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = true,
    };
    try std.testing.expectEqual(@as(usize, 4096), expectedAttentionQProjectionDim(&gated));
}

test "bufferF32RowCount derives staged row count from buffer bytes" {
    const bytes = 2 * 1024 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 0,
        .size = bytes,
    };
    try std.testing.expectEqual(@as(usize, 2), try bufferF32RowCount(&buffer, 1024));
}

test "logicalF32RowSlice uses packed row offsets for tightly packed buffers" {
    const row_width = 8;
    const row_bytes = row_width * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 4096,
        .size = 2 * row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, row_width);
    try std.testing.expectEqual(buffer.pointer + row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, row_bytes), row1.size);
}

test "logicalF32RowSlice uses widened row stride for staged slot buffers" {
    const logical_width = 8;
    const logical_row_bytes = logical_width * @sizeOf(f32);
    const widened_row_bytes = 16 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 8192,
        .size = 2 * widened_row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, logical_width);
    try std.testing.expectEqual(buffer.pointer + widened_row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, logical_row_bytes), row1.size);
}

test "attentionFallbackUsesCache uses decode mode only for single-row execution" {
    try std.testing.expect(CudaBackend.attentionFallbackUsesCache(1));
    try std.testing.expect(!CudaBackend.attentionFallbackUsesCache(2));
    try std.testing.expect(!CudaBackend.attentionFallbackUsesCache(15));
}
