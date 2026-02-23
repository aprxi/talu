//! CUDA backend engine (Phase 1 stub).
//!
//! This implements the backend contract while returning explicit typed errors
//! for unimplemented execution methods.

const std = @import("std");
const models = @import("../../../models/root.zig");
const contract = @import("../contract.zig");
const cpu_vision = @import("../cpu/vision/root.zig");
const cpu_engine = @import("../cpu/engine.zig");
const progress = @import("../../../progress.zig");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const load_transforms = @import("../../../models/load/transforms.zig");
const parity_probe = @import("parity_probe.zig");
const smoke_checks = @import("smoke_checks.zig");

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const prototype_eps: f32 = 1e-5;
const parity_probe_max_mean_abs_diff: f32 = 1.0;
const parity_probe_max_abs_diff: f32 = 25.0;
const prototype_low_token_band: usize = 4096;
const initial_kv_cache_tokens: usize = 256;
const kv_cache_dtype_fp16: bool = true;
const enable_fused_attention_f16_kv: bool = true;
const max_supported_fused_f16_kv_head_dim = 512;
const gaffine_scales_dtype_f16 = compute.cuda.gaffine_u4_matvec.scales_dtype_f16;
const gaffine_scales_dtype_bf16 = compute.cuda.gaffine_u4_matvec.scales_dtype_bf16;
const DenseU16Dtype = enum(u8) {
    f16,
    bf16,
};

const KernelSlot = enum {
    vector_add,
    mul,
    copy,
    copy_u16,
    cast_f32_to_f16,
    kv_write_f16,
    rmsnorm,
    rope,
    rope_store_f16,
    attn_scores,
    attn_scores_f16_kv,
    attn_scores_heads_f16_kv,
    attn_fused_heads_f16_kv,
    softmax,
    softmax_rows,
    attn_weighted_sum,
    attn_weighted_sum_f16_kv,
    attn_weighted_sum_heads_f16_kv,
    silu,
    silu_mul,
    shortconv_step,
    argmax,
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
    per_head_f32_kv,
};

const AttentionKernelSet = struct {
    attn_scores_function: ?compute.cuda.Function,
    softmax_function: compute.cuda.Function,
    attn_weighted_sum_function: ?compute.cuda.Function,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function,
    softmax_rows_function: ?compute.cuda.Function,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function,
};

const required_kernels = [_]RequiredKernel{
    .{ .slot = .vector_add, .op_name = compute.cuda.vector_add.op_name, .embedded_symbol = compute.cuda.vector_add.embedded_symbol },
    .{ .slot = .mul, .op_name = compute.cuda.mul.op_name, .embedded_symbol = compute.cuda.mul.embedded_symbol },
    .{ .slot = .copy, .op_name = compute.cuda.copy.op_name, .embedded_symbol = compute.cuda.copy.embedded_symbol },
    .{ .slot = .copy_u16, .op_name = compute.cuda.copy_u16.op_name, .embedded_symbol = compute.cuda.copy_u16.embedded_symbol },
    .{ .slot = .cast_f32_to_f16, .op_name = compute.cuda.cast_f32_to_f16.op_name, .embedded_symbol = compute.cuda.cast_f32_to_f16.embedded_symbol },
    .{ .slot = .kv_write_f16, .op_name = compute.cuda.kv_write_f16.op_name, .embedded_symbol = compute.cuda.kv_write_f16.embedded_symbol },
    .{ .slot = .rmsnorm, .op_name = compute.cuda.rmsnorm.op_name, .embedded_symbol = compute.cuda.rmsnorm.embedded_symbol },
    .{ .slot = .rope, .op_name = compute.cuda.rope.op_name, .embedded_symbol = compute.cuda.rope.embedded_symbol },
    .{ .slot = .rope_store_f16, .op_name = compute.cuda.rope_store_f16.op_name, .embedded_symbol = compute.cuda.rope_store_f16.embedded_symbol },
    .{ .slot = .attn_scores, .op_name = compute.cuda.attn_scores.op_name, .embedded_symbol = compute.cuda.attn_scores.embedded_symbol },
    .{ .slot = .attn_scores_f16_kv, .op_name = compute.cuda.attn_scores_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_f16_kv.embedded_symbol },
    .{ .slot = .attn_scores_heads_f16_kv, .op_name = compute.cuda.attn_scores_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_scores_heads_f16_kv.embedded_symbol },
    .{ .slot = .attn_fused_heads_f16_kv, .op_name = compute.cuda.attn_fused_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_fused_heads_f16_kv.embedded_symbol },
    .{ .slot = .softmax, .op_name = compute.cuda.softmax.op_name, .embedded_symbol = compute.cuda.softmax.embedded_symbol },
    .{ .slot = .softmax_rows, .op_name = compute.cuda.softmax_rows.op_name, .embedded_symbol = compute.cuda.softmax_rows.embedded_symbol },
    .{ .slot = .attn_weighted_sum, .op_name = compute.cuda.attn_weighted_sum.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum.embedded_symbol },
    .{ .slot = .attn_weighted_sum_f16_kv, .op_name = compute.cuda.attn_weighted_sum_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_f16_kv.embedded_symbol },
    .{ .slot = .attn_weighted_sum_heads_f16_kv, .op_name = compute.cuda.attn_weighted_sum_heads_f16_kv.op_name, .embedded_symbol = compute.cuda.attn_weighted_sum_heads_f16_kv.embedded_symbol },
    .{ .slot = .silu, .op_name = compute.cuda.silu.op_name, .embedded_symbol = compute.cuda.silu.embedded_symbol },
    .{ .slot = .silu_mul, .op_name = compute.cuda.silu_mul.op_name, .embedded_symbol = compute.cuda.silu_mul.embedded_symbol },
    .{ .slot = .shortconv_step, .op_name = compute.cuda.shortconv_step.op_name, .embedded_symbol = compute.cuda.shortconv_step.embedded_symbol },
    .{ .slot = .argmax, .op_name = compute.cuda.argmax.op_name, .embedded_symbol = compute.cuda.argmax.embedded_symbol },
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
        const need_attention_score_buffers = !kv_cache_dtype_fp16 or
            !enable_fused_attention_f16_kv or
            head_dim > max_supported_fused_f16_kv_head_dim;
        const attn_rows = std.math.mul(usize, max_seq_len, n_heads) catch return error.InvalidArgument;
        const attn_rows_bytes = std.math.mul(usize, attn_rows, @sizeOf(f32)) catch return error.InvalidArgument;
        const hidden_host = try allocator.alloc(f32, d_model);
        errdefer allocator.free(hidden_host);

        const norm_weight_host = try allocator.alloc(f32, d_model);
        defer allocator.free(norm_weight_host);
        const using_model_norm = tryPopulateFinalNormWeight(loaded, norm_weight_host);
        if (!using_model_norm) fillPrototypeNormWeight(norm_weight_host);
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

        var using_model_projection = projection_weight_opt != null;
        if (projection_weight_opt == null) {
            const projected_vocab_fallback = vocab_size;
            const projection_elements = std.math.mul(usize, d_model, projected_vocab_fallback) catch return error.InvalidArgument;
            const projection_host = try allocator.alloc(f32, projection_elements);
            defer allocator.free(projection_host);
            fillPrototypeProjection(projection_host, d_model, projected_vocab_fallback);

            var projection_buffer = try device.allocBuffer(projection_elements * @sizeOf(f32));
            errdefer projection_buffer.deinit(device);
            try projection_buffer.upload(device, std.mem.sliceAsBytes(projection_host));
            projection_weight_opt = .{
                .dense_f32 = .{
                    .rows = d_model,
                    .cols = projected_vocab_fallback,
                    .buffer = projection_buffer,
                },
            };
            using_model_projection = false;
            projection_from_lm_head = false;
        }
        const projection_weight = projection_weight_opt.?;
        const projected_vocab = projection_weight.cols();
        const projected_logits_host = try allocator.alloc(f32, projected_vocab);
        errdefer allocator.free(projected_logits_host);
        const logits_bytes = std.math.mul(usize, projected_vocab, @sizeOf(f32)) catch return error.InvalidArgument;

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
            .using_model_embeddings = canUseModelEmbeddings(loaded),
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
            .shortconv_proj_dev = shortconv_proj_dev,
            .shortconv_conv_dev = shortconv_conv_dev,
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
        };
    }

    fn deinit(self: *PrototypeRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
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
            self.shortconv_proj_dev.size +
            self.shortconv_conv_dev.size +
            self.logits_dev.size +
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
    ln1_weight: DeviceTensor,
    ln2_weight: DeviceTensor,
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
    attention_mlp: ?AttentionMlpBlockRuntime = null,
    shortconv: ?ShortConvBlockRuntime = null,

    fn deinit(self: *BlockRuntimeLayer, device: *compute.cuda.Device) void {
        if (self.attention_mlp) |*block| block.deinit(device);
        if (self.shortconv) |*block| block.deinit(device);
        self.* = .{};
    }
};

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

        for (loaded.blocks, 0..) |block_weights, layer_idx| {
            switch (block_weights) {
                .attention_mlp => |attn| {
                    if (attn.mla_config != null) {
                        log.warn("inference", "CUDA block runtime MLA not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }
                    if (attn.moe_weights != null) {
                        log.warn("inference", "CUDA block runtime MoE not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }
                    if (attn.fused.qkv_proj != null or attn.fused.gate_up != null) {
                        log.warn("inference", "CUDA block runtime fused attention/ffn weights not supported yet", .{ .layer = layer_idx });
                        return error.UnsupportedModel;
                    }

                    const q_proj = attn.q_proj orelse return error.MissingWeight;
                    const k_proj = attn.k_proj orelse return error.MissingWeight;
                    const v_proj = attn.v_proj orelse return error.MissingWeight;
                    const w1 = attn.w1 orelse return error.MissingWeight;
                    const w2 = attn.w2 orelse return error.MissingWeight;
                    const w3 = attn.w3 orelse return error.MissingWeight;
                    if (attention_block_count == 0) {
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

                    var q_proj_dev = try uploadLinearWeight(device, allocator, q_proj, d_model);
                    errdefer q_proj_dev.deinit(device);
                    var k_proj_dev = try uploadLinearWeight(device, allocator, k_proj, d_model);
                    errdefer k_proj_dev.deinit(device);
                    var v_proj_dev = try uploadLinearWeight(device, allocator, v_proj, d_model);
                    errdefer v_proj_dev.deinit(device);
                    var o_proj_dev = try uploadLinearWeight(device, allocator, attn.o_proj, q_proj_dev.cols());
                    errdefer o_proj_dev.deinit(device);
                    var w1_dev = try uploadLinearWeight(device, allocator, w1, d_model);
                    errdefer w1_dev.deinit(device);
                    var w3_dev = try uploadLinearWeight(device, allocator, w3, d_model);
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
                        .ln1_weight = ln1_weight,
                        .ln2_weight = ln2_weight,
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

pub const CudaBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = true,
        .embedding = false,
        .warmup = false,
    };

    pub const PrefillVisionInput = cpu_vision.PrefillVisionInput;

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    device: compute.cuda.Device,
    compute_stream: ?compute.cuda.StreamHandle = null,
    kernel_registry: compute.cuda.Registry,
    vector_add_function: ?compute.cuda.Function = null,
    vector_add_source: ?compute.cuda.registry.KernelSource = null,
    mul_function: ?compute.cuda.Function = null,
    mul_source: ?compute.cuda.registry.KernelSource = null,
    copy_function: ?compute.cuda.Function = null,
    copy_source: ?compute.cuda.registry.KernelSource = null,
    copy_u16_function: ?compute.cuda.Function = null,
    copy_u16_source: ?compute.cuda.registry.KernelSource = null,
    cast_f32_to_f16_function: ?compute.cuda.Function = null,
    cast_f32_to_f16_source: ?compute.cuda.registry.KernelSource = null,
    kv_write_f16_function: ?compute.cuda.Function = null,
    kv_write_f16_source: ?compute.cuda.registry.KernelSource = null,
    rmsnorm_function: ?compute.cuda.Function = null,
    rmsnorm_source: ?compute.cuda.registry.KernelSource = null,
    rope_function: ?compute.cuda.Function = null,
    rope_source: ?compute.cuda.registry.KernelSource = null,
    rope_store_f16_function: ?compute.cuda.Function = null,
    rope_store_f16_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_function: ?compute.cuda.Function = null,
    attn_scores_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_f16_kv_function: ?compute.cuda.Function = null,
    attn_scores_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_scores_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_scores_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_fused_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_fused_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    softmax_function: ?compute.cuda.Function = null,
    softmax_source: ?compute.cuda.registry.KernelSource = null,
    softmax_rows_function: ?compute.cuda.Function = null,
    softmax_rows_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_function: ?compute.cuda.Function = null,
    attn_weighted_sum_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_f16_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = null,
    attn_weighted_sum_heads_f16_kv_source: ?compute.cuda.registry.KernelSource = null,
    silu_function: ?compute.cuda.Function = null,
    silu_source: ?compute.cuda.registry.KernelSource = null,
    silu_mul_function: ?compute.cuda.Function = null,
    silu_mul_source: ?compute.cuda.registry.KernelSource = null,
    shortconv_step_function: ?compute.cuda.Function = null,
    shortconv_step_source: ?compute.cuda.registry.KernelSource = null,
    argmax_function: ?compute.cuda.Function = null,
    argmax_source: ?compute.cuda.registry.KernelSource = null,
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
    slot_logits: []f32,
    argmax_index_dev: compute.cuda.Buffer,

    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel) !CudaBackend {
        var device = try compute.cuda.Device.init();
        errdefer device.deinit();

        log.info("inference", "CUDA device ready", .{ .name = device.name() });
        var backend = CudaBackend{
            .allocator = allocator,
            .loaded = loaded,
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

        try smoke_checks.runMatmulSmoke(&backend);
        try smoke_checks.runKernelSmoke(&backend);
        runCpuParityProbe(&backend) catch |err| switch (err) {
            error.CudaParityMismatch => return err,
            else => log.warn("inference", "CUDA parity probe unavailable", .{
                .reason = @errorName(err),
            }),
        };
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
            .rmsnorm_kernel = @as(u8, @intFromBool(backend.rmsnorm_function != null)),
            .mul_kernel = @as(u8, @intFromBool(backend.mul_function != null)),
            .copy_kernel = @as(u8, @intFromBool(backend.copy_function != null)),
            .copy_u16_kernel = @as(u8, @intFromBool(backend.copy_u16_function != null)),
            .cast_f32_to_f16_kernel = @as(u8, @intFromBool(backend.cast_f32_to_f16_function != null)),
            .kv_write_f16_kernel = @as(u8, @intFromBool(backend.kv_write_f16_function != null)),
            .rope_kernel = @as(u8, @intFromBool(backend.rope_function != null)),
            .rope_store_f16_kernel = @as(u8, @intFromBool(backend.rope_store_f16_function != null)),
            .attn_scores_kernel = @as(u8, @intFromBool(backend.attn_scores_function != null)),
            .attn_scores_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_scores_f16_kv_function != null)),
            .attn_scores_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_scores_heads_f16_kv_function != null)),
            .attn_fused_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_fused_heads_f16_kv_function != null)),
            .attn_fused_heads_f16_kv_enabled = @as(u8, @intFromBool(enable_fused_attention_f16_kv)),
            .softmax_kernel = @as(u8, @intFromBool(backend.softmax_function != null)),
            .softmax_rows_kernel = @as(u8, @intFromBool(backend.softmax_rows_function != null)),
            .attn_weighted_sum_kernel = @as(u8, @intFromBool(backend.attn_weighted_sum_function != null)),
            .attn_weighted_sum_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_weighted_sum_f16_kv_function != null)),
            .attn_weighted_sum_heads_f16_kv_kernel = @as(u8, @intFromBool(backend.attn_weighted_sum_heads_f16_kv_function != null)),
            .attn_score_buffers = @as(u8, @intFromBool(backend.prototype.attn_scores_dev != null and backend.prototype.attn_probs_dev != null)),
            .silu_kernel = @as(u8, @intFromBool(backend.silu_function != null)),
            .silu_mul_kernel = @as(u8, @intFromBool(backend.silu_mul_function != null)),
            .shortconv_step_kernel = @as(u8, @intFromBool(backend.shortconv_step_function != null)),
            .argmax_kernel = @as(u8, @intFromBool(backend.argmax_function != null)),
            .matvec_f16_kernel = @as(u8, @intFromBool(backend.matvec_f16_function != null)),
            .matvec_bf16_kernel = @as(u8, @intFromBool(backend.matvec_bf16_function != null)),
            .matvec_gate_up_f16_kernel = @as(u8, @intFromBool(backend.matvec_gate_up_f16_function != null)),
            .matvec_gate_up_bf16_kernel = @as(u8, @intFromBool(backend.matvec_gate_up_bf16_function != null)),
            .matvec_qkv_f16_kernel = @as(u8, @intFromBool(backend.matvec_qkv_f16_function != null)),
            .matvec_qkv_bf16_kernel = @as(u8, @intFromBool(backend.matvec_qkv_bf16_function != null)),
            .gaffine_u4_matvec_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_function != null)),
            .gaffine_u4_matvec_gate_up_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_gate_up_function != null)),
            .gaffine_u4_matvec_qkv_kernel = @as(u8, @intFromBool(backend.gaffine_u4_matvec_qkv_function != null)),
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
            .embed_dtype = @tagName(loaded.token_embeddings.dtype),
            .embed_shape_0 = loaded.token_embeddings.shape[0],
            .embed_shape_1 = loaded.token_embeddings.shape[1],
        });
        return backend;
    }

    pub fn deinit(self: *CudaBackend) void {
        self.device.setLaunchStream(null);
        if (self.compute_stream) |stream| {
            _ = self.device.synchronizeStream(stream) catch {};
            self.device.destroyStream(stream);
            self.compute_stream = null;
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

    pub fn prefill(self: *CudaBackend, tokens: []const u32, logits_out: []f32) !void {
        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefill invalid args", .{
                .reason = "empty_tokens",
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefill invalid args", .{
                .reason = "logits_len_mismatch",
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            try self.computeGpuPrototypeLogits(tokens[i], i, self.slot_logits);
        }
        @memcpy(logits_out, self.slot_logits);
        self.slot_position = tokens.len;
    }

    pub fn decode(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA decode invalid args", .{
                .reason = "logits_len_mismatch",
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        try self.computeGpuPrototypeLogits(token, position, logits_out);
        self.slot_position = position + 1;
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
        if (max_tokens == 0 or output_tokens.len == 0) return 0;
        if (!self.slot_in_use) {
            self.slot_in_use = true;
            self.slot_position = start_position;
        }

        var current_token = first_token;
        var generated: usize = 0;
        var position = self.slot_position;
        const budget = @min(max_tokens, output_tokens.len);
        while (generated < budget) : (generated += 1) {
            try self.computeGpuPrototypeLogitsWithLayerLimit(
                current_token,
                position,
                null,
                self.block_runtime.blocks.len,
                false,
            );
            const next_token = try selectNextTokenFromDeviceLogits(self);
            output_tokens[generated] = next_token;
            position += 1;
            self.slot_position = position;
            if (callback) |cb| cb(next_token, callback_data);

            for (eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    return generated + 1;
                }
            }
            current_token = next_token;
        }
        return generated;
    }

    pub fn allocSlot(self: *CudaBackend) ?usize {
        if (self.slot_in_use) return null;
        self.slot_in_use = true;
        self.slot_position = 0;
        return 0;
    }

    pub fn freeSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_in_use = false;
        self.slot_position = 0;
    }

    pub fn resetSlot(self: *CudaBackend, slot_index: usize) void {
        if (slot_index != 0) return;
        self.slot_position = 0;
    }

    pub fn getPosition(self: *const CudaBackend, slot_index: usize) usize {
        if (slot_index != 0) return 0;
        return self.slot_position;
    }

    pub fn prefillSlot(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        if (tokens.len == 0) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "empty_tokens",
                .slot_index = slot_index,
            });
            return error.InvalidArgument;
        }
        if (logits_out.len != self.vocab_size) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "logits_len_mismatch",
                .slot_index = slot_index,
                .logits_len = logits_out.len,
                .vocab_size = self.vocab_size,
            });
            return error.InvalidArgument;
        }
        if (!self.slot_in_use or slot_index != 0) {
            log.warn("inference", "CUDA prefillSlot invalid args", .{
                .reason = "slot_state",
                .slot_index = slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
            });
            return error.InvalidArgument;
        }

        if (tokens.len > self.max_seq_len) return error.InvalidArgument;
        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            try self.computeGpuPrototypeLogits(tokens[i], i, self.slot_logits);
        }
        @memcpy(logits_out, self.slot_logits);
        self.slot_position = tokens.len;
    }

    pub fn prefillSlotWithVision(
        self: *CudaBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        _ = vision_input;
        return self.prefillSlot(slot_index, tokens, logits_out);
    }

    pub fn decodeBatch(
        self: *CudaBackend,
        requests: []const contract.DecodeRequest,
        results: []contract.DecodeResult,
    ) !void {
        if (results.len < requests.len) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "results_short",
                .requests = requests.len,
                .results = results.len,
            });
            return error.InvalidArgument;
        }
        if (requests.len == 0) return;
        if (requests.len > 1) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "batch_gt_one",
                .requests = requests.len,
            });
            return error.InvalidArgument;
        }

        const req = requests[0];
        if (!self.slot_in_use or req.slot_index != 0) {
            log.warn("inference", "CUDA decodeBatch invalid args", .{
                .reason = "slot_state",
                .slot_index = req.slot_index,
                .slot_in_use = @as(u8, @intFromBool(self.slot_in_use)),
            });
            return error.InvalidArgument;
        }

        try self.computeGpuPrototypeLogits(req.token, self.slot_position, self.slot_logits);
        results[0] = .{
            .slot_index = req.slot_index,
            .logits = self.slot_logits,
        };
        self.slot_position += 1;
    }

    fn computeGpuPrototypeLogits(self: *CudaBackend, token: u32, position: usize, logits_out: []f32) !void {
        return self.computeGpuPrototypeLogitsWithLayerLimit(
            token,
            position,
            logits_out,
            self.block_runtime.blocks.len,
            true,
        );
    }

    fn computeGpuPrototypeLogitsWithLayerLimit(
        self: *CudaBackend,
        token: u32,
        position: usize,
        logits_out_opt: ?[]f32,
        layer_limit: usize,
        download_logits: bool,
    ) !void {
        if (download_logits) {
            const logits_out = logits_out_opt orelse return error.InvalidArgument;
            if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        }
        if (position >= self.max_seq_len) return error.InvalidArgument;
        if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;
        if (position == 0 and self.block_runtime.shortconv_block_count > 0) {
            try self.resetShortConvStates();
        }
        try self.ensureKvCapacity(position + 1);

        const rmsnorm_function = self.rmsnorm_function orelse return error.CudaKernelUnavailable;
        const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
        const silu_mul_function = self.silu_mul_function orelse return error.CudaKernelUnavailable;
        const shortconv_step_function = self.shortconv_step_function orelse return error.CudaKernelUnavailable;
        const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
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
        const attn_scores_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_scores_function orelse return error.CudaKernelUnavailable);
        const attn_scores_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null;
        const attn_fused_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            self.attn_fused_heads_f16_kv_function
        else
            null;
        const softmax_function = self.softmax_function orelse return error.CudaKernelUnavailable;
        const softmax_rows_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            (self.softmax_rows_function orelse return error.CudaKernelUnavailable)
        else
            null;
        const attn_weighted_sum_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_weighted_sum_function orelse return error.CudaKernelUnavailable);
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
        const theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;

        const used_model_embeddings = tryPopulateHiddenFromToken(self.loaded, token, self.prototype.hidden_host) catch |err| switch (err) {
            error.InvalidArgument => return error.InvalidArgument,
            else => return err,
        };
        if (!used_model_embeddings) fillPrototypeInput(self.prototype.hidden_host, token);
        if (!used_model_embeddings and position == 0) {
            log.warn("inference", "CUDA using synthetic embedding fallback", .{
                .token = token,
                .embed_shape_0 = self.loaded.token_embeddings.shape[0],
                .embed_shape_1 = self.loaded.token_embeddings.shape[1],
                .embed_dtype = @tagName(self.loaded.token_embeddings.dtype),
                .embed_ndim = self.loaded.token_embeddings.n_dims,
            });
        }
        if (self.loaded.config.embedding_multiplier != 1.0) {
            for (self.prototype.hidden_host) |*v| {
                v.* *= self.loaded.config.embedding_multiplier;
            }
        }
        try self.prototype.input_dev.upload(&self.device, std.mem.sliceAsBytes(self.prototype.hidden_host));

        var layer_idx: usize = 0;
        while (layer_idx < layer_limit) : (layer_idx += 1) {
            const layer = &self.block_runtime.blocks[layer_idx];
            if (layer.attention_mlp) |*block| {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &self.prototype.input_dev,
                    &block.ln1_weight.buffer,
                    &self.prototype.norm_out_dev,
                    1,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );

                _ = try self.runQkvProjection(&self.prototype.norm_out_dev, block);

                if (block.q_norm_weight) |*q_norm| {
                    try compute.cuda.rmsnorm.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        rmsnorm_function,
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
                        rmsnorm_function,
                        &self.prototype.attn_k_dev,
                        &k_norm.buffer,
                        &self.prototype.attn_k_dev,
                        n_kv_heads_u32,
                        head_dim_u32,
                        self.norm_eps,
                        self.loaded.runtime.qk_norm_weight_offset,
                    );
                }

                const use_fused_attention_heads_f16_kv = kv_cache_dtype_fp16 and
                    enable_fused_attention_f16_kv and
                    head_dim_u32 <= max_supported_fused_f16_kv_head_dim and
                    attn_fused_heads_f16_kv_function != null;
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
                        theta,
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
                        theta,
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
                            theta,
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
                            theta,
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function.?,
                            &self.prototype.attn_v_dev,
                            &v_row,
                            @intCast(block.kv_dim),
                        );
                    } else {
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function.?,
                            &self.prototype.attn_k_dev,
                            &k_row,
                            @intCast(block.kv_dim),
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function.?,
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
                const attention_kernels = AttentionKernelSet{
                    .attn_scores_function = attn_scores_function,
                    .softmax_function = softmax_function,
                    .attn_weighted_sum_function = attn_weighted_sum_function,
                    .attn_scores_heads_f16_kv_function = attn_scores_heads_f16_kv_function,
                    .softmax_rows_function = softmax_rows_function,
                    .attn_weighted_sum_heads_f16_kv_function = attn_weighted_sum_heads_f16_kv_function,
                    .attn_fused_heads_f16_kv_function = attn_fused_heads_f16_kv_function,
                };
                _ = try self.runAttentionContext(
                    block,
                    attention_kernels,
                    seq_len_u32,
                    head_dim_u32,
                    kv_dim_u32,
                    kv_groups,
                    kv_groups_u32,
                    rope_dim_u32,
                    position_u32,
                    theta,
                );

                try self.linearForward(&self.prototype.attn_context_dev, &block.o_proj, &self.prototype.attn_out_dev);
                try compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    &self.prototype.input_dev,
                    &self.prototype.attn_out_dev,
                    &self.prototype.input_dev,
                    d_model_u32,
                );

                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &self.prototype.input_dev,
                    &block.ln2_weight.buffer,
                    &self.prototype.norm_out_dev,
                    1,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );

                _ = try self.runGateUpProjectionWithWeights(&self.prototype.norm_out_dev, &block.w1, &block.w3);
                const d_ff_u32: u32 = @intCast(block.d_ff);
                try compute.cuda.silu_mul.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    silu_mul_function,
                    &self.prototype.ffn_gate_dev,
                    &self.prototype.ffn_up_dev,
                    &self.prototype.ffn_mul_dev,
                    d_ff_u32,
                );
                try self.linearForward(&self.prototype.ffn_mul_dev, &block.w2, &self.prototype.ffn_down_dev);
                try compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    &self.prototype.input_dev,
                    &self.prototype.ffn_down_dev,
                    &self.prototype.input_dev,
                    d_model_u32,
                );
            } else if (layer.shortconv) |*block| {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rmsnorm_function,
                    &self.prototype.input_dev,
                    &block.ln1_weight.buffer,
                    &self.prototype.norm_out_dev,
                    1,
                    d_model_u32,
                    self.norm_eps,
                    self.loaded.runtime.weight_offset,
                );

                try self.linearForward(&self.prototype.norm_out_dev, &block.in_proj, &self.prototype.shortconv_proj_dev);
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

                try self.linearForward(&self.prototype.shortconv_conv_dev, &block.out_proj, &self.prototype.attn_out_dev);
                try compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    &self.prototype.input_dev,
                    &self.prototype.attn_out_dev,
                    &self.prototype.input_dev,
                    d_model_u32,
                );

                if (block.ln2_weight != null and block.ffn_w1 != null and block.ffn_w2 != null and block.ffn_w3 != null) {
                    try compute.cuda.rmsnorm.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        rmsnorm_function,
                        &self.prototype.input_dev,
                        &block.ln2_weight.?.buffer,
                        &self.prototype.norm_out_dev,
                        1,
                        d_model_u32,
                        self.norm_eps,
                        self.loaded.runtime.weight_offset,
                    );
                    _ = try self.runGateUpProjectionWithWeights(&self.prototype.norm_out_dev, &block.ffn_w1.?, &block.ffn_w3.?);
                    const d_ff_u32: u32 = @intCast(block.d_ff);
                    try compute.cuda.silu_mul.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        silu_mul_function,
                        &self.prototype.ffn_gate_dev,
                        &self.prototype.ffn_up_dev,
                        &self.prototype.ffn_mul_dev,
                        d_ff_u32,
                    );
                    try self.linearForward(&self.prototype.ffn_mul_dev, &block.ffn_w2.?, &self.prototype.ffn_down_dev);
                    try compute.cuda.vector_add.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        vector_add_function,
                        &self.prototype.input_dev,
                        &self.prototype.ffn_down_dev,
                        &self.prototype.input_dev,
                        d_model_u32,
                    );
                }
            } else {
                return error.InvalidArgument;
            }
        }

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

    fn ensureKvCapacity(self: *CudaBackend, required_tokens: usize) !void {
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
        switch (weight.*) {
            .dense_f32 => |w| {
                try self.blas.matmulF32(
                    &self.device,
                    input,
                    1,
                    w.rows,
                    &w.buffer,
                    w.cols,
                    out,
                );
            },
            .dense_u16 => |w| {
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
            },
            .gaffine_u4 => |w| {
                const kernel = self.gaffine_u4_matvec_function orelse return error.CudaKernelUnavailable;
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
                );
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

    fn runAttentionContext(
        self: *CudaBackend,
        block: *const AttentionMlpBlockRuntime,
        kernels: AttentionKernelSet,
        seq_len_u32: u32,
        head_dim_u32: u32,
        kv_dim_u32: u32,
        kv_groups: usize,
        kv_groups_u32: u32,
        rope_dim_u32: u32,
        position_u32: u32,
        theta: f32,
    ) !AttentionPath {
        if (kv_cache_dtype_fp16) {
            if (enable_fused_attention_f16_kv and head_dim_u32 <= max_supported_fused_f16_kv_head_dim and kernels.attn_fused_heads_f16_kv_function != null) {
                try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_f16_kv_function.?,
                    &self.prototype.attn_q_dev,
                    &block.k_cache,
                    &block.v_cache,
                    &self.prototype.attn_context_dev,
                    @intCast(self.n_heads),
                    seq_len_u32,
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
                &block.k_cache,
                attn_scores_dev,
                @intCast(self.n_heads),
                seq_len_u32,
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
                seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &block.v_cache,
                &self.prototype.attn_context_dev,
                @intCast(self.n_heads),
                seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            );
            return .heads_f16_kv;
        }

        const attn_scores_function = kernels.attn_scores_function orelse return error.CudaKernelUnavailable;
        const attn_weighted_sum_function = kernels.attn_weighted_sum_function orelse return error.CudaKernelUnavailable;
        const attn_scores_dev = try self.prototype.requireAttentionScoresDev();
        const attn_probs_dev = try self.prototype.requireAttentionProbsDev();

        var head_idx: usize = 0;
        while (head_idx < self.n_heads) : (head_idx += 1) {
            const q_offset = std.math.mul(usize, head_idx, self.head_dim) catch return error.InvalidArgument;
            const q_offset_bytes = std.math.mul(usize, q_offset, @sizeOf(f32)) catch return error.InvalidArgument;
            const head_bytes = std.math.mul(usize, self.head_dim, @sizeOf(f32)) catch return error.InvalidArgument;
            var q_head = try bufferSlice(&self.prototype.attn_q_dev, q_offset_bytes, head_bytes);
            var ctx_head = try bufferSlice(&self.prototype.attn_context_dev, q_offset_bytes, head_bytes);

            const kv_head = kvHeadForQueryHead(head_idx, kv_groups);
            const kv_offset = std.math.mul(usize, kv_head, self.head_dim) catch return error.InvalidArgument;
            const kv_offset_u32: u32 = @intCast(kv_offset);

            try compute.cuda.attn_scores.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                attn_scores_function,
                &q_head,
                &block.k_cache,
                attn_scores_dev,
                seq_len_u32,
                kv_dim_u32,
                kv_offset_u32,
                head_dim_u32,
                self.attention_scale,
            );
            try compute.cuda.softmax.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_function,
                attn_scores_dev,
                attn_probs_dev,
                seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                attn_weighted_sum_function,
                attn_probs_dev,
                &block.v_cache,
                &ctx_head,
                seq_len_u32,
                kv_dim_u32,
                kv_offset_u32,
                head_dim_u32,
            );
        }
        return .per_head_f32_kv;
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
        if (q.rows != self.d_model or k.rows != self.d_model or v.rows != self.d_model) return false;
        if (q.dtype != k.dtype or q.dtype != v.dtype) return false;
        if (q.cols > std.math.maxInt(u32) or
            k.cols > std.math.maxInt(u32) or
            v.cols > std.math.maxInt(u32) or
            q.rows > std.math.maxInt(u32))
        {
            return false;
        }

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
            .attn_scores => {
                self.attn_scores_function = resolved.function;
                self.attn_scores_source = resolved.source;
            },
            .attn_scores_f16_kv => {
                self.attn_scores_f16_kv_function = resolved.function;
                self.attn_scores_f16_kv_source = resolved.source;
            },
            .attn_scores_heads_f16_kv => {
                self.attn_scores_heads_f16_kv_function = resolved.function;
                self.attn_scores_heads_f16_kv_source = resolved.source;
            },
            .attn_fused_heads_f16_kv => {
                self.attn_fused_heads_f16_kv_function = resolved.function;
                self.attn_fused_heads_f16_kv_source = resolved.source;
            },
            .softmax => {
                self.softmax_function = resolved.function;
                self.softmax_source = resolved.source;
            },
            .softmax_rows => {
                self.softmax_rows_function = resolved.function;
                self.softmax_rows_source = resolved.source;
            },
            .attn_weighted_sum => {
                self.attn_weighted_sum_function = resolved.function;
                self.attn_weighted_sum_source = resolved.source;
            },
            .attn_weighted_sum_f16_kv => {
                self.attn_weighted_sum_f16_kv_function = resolved.function;
                self.attn_weighted_sum_f16_kv_source = resolved.source;
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
            .shortconv_step => {
                self.shortconv_step_function = resolved.function;
                self.shortconv_step_source = resolved.source;
            },
            .argmax => {
                self.argmax_function = resolved.function;
                self.argmax_source = resolved.source;
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

fn runCpuParityProbe(backend: *CudaBackend) !void {
    const probe_tokens = [_]u32{ 42, 17, 99, 7 };
    var cpu_backend = try cpu_engine.FusedCpuBackend.init(
        backend.allocator,
        backend.loaded,
        1,
        progress.Context.NONE,
    );
    defer cpu_backend.deinit();

    const vocab = backend.vocab_size;
    const cpu_logits = try backend.allocator.alloc(f32, vocab);
    defer backend.allocator.free(cpu_logits);
    const cuda_logits = try backend.allocator.alloc(f32, vocab);
    defer backend.allocator.free(cuda_logits);

    const cpu_slot = cpu_backend.allocSlot() orelse return error.OutOfMemory;
    defer cpu_backend.freeSlot(cpu_slot);
    try cpu_backend.prefillSlot(cpu_slot, probe_tokens[0..1], cpu_logits);
    try backend.computeGpuPrototypeLogits(probe_tokens[0], 0, cuda_logits);

    const prefill = try parity_probe.summarize(cpu_logits, cuda_logits);
    log.info("inference", "CUDA parity probe prefill", .{
        .token = probe_tokens[0],
        .layers = backend.block_runtime.blocks.len,
        .cpu_top = prefill.cpu_top,
        .cuda_top = prefill.cuda_top,
        .same_top = @as(u8, @intFromBool(prefill.cpu_top == prefill.cuda_top)),
        .mean_abs_diff = prefill.mean_abs_diff,
        .max_abs_diff = prefill.max_abs_diff,
        .finite_count = prefill.finite_count,
        .cpu_nan_count = prefill.cpu_nan_count,
        .cuda_nan_count = prefill.cuda_nan_count,
    });
    try parity_probe.enforce(
        prefill,
        parity_probe_max_mean_abs_diff,
        parity_probe_max_abs_diff,
    );

    var requests = [_]contract.DecodeRequest{
        .{ .slot_index = cpu_slot, .token = 0 },
    };
    var results = [_]contract.DecodeResult{
        .{ .slot_index = cpu_slot, .logits = &.{} },
    };
    var step: usize = 1;
    while (step < probe_tokens.len) : (step += 1) {
        requests[0] = .{
            .slot_index = cpu_slot,
            .token = probe_tokens[step],
        };
        results[0] = .{
            .slot_index = cpu_slot,
            .logits = &.{},
        };
        try cpu_backend.decodeBatch(requests[0..], results[0..]);
        if (results[0].logits.len != cpu_logits.len) return error.InvalidShape;
        @memcpy(cpu_logits, results[0].logits);

        try backend.computeGpuPrototypeLogits(probe_tokens[step], step, cuda_logits);
        const decode = try parity_probe.summarize(cpu_logits, cuda_logits);
        log.info("inference", "CUDA parity probe decode", .{
            .step = step,
            .token = probe_tokens[step],
            .cpu_top = decode.cpu_top,
            .cuda_top = decode.cuda_top,
            .same_top = @as(u8, @intFromBool(decode.cpu_top == decode.cuda_top)),
            .mean_abs_diff = decode.mean_abs_diff,
            .max_abs_diff = decode.max_abs_diff,
            .finite_count = decode.finite_count,
            .cpu_nan_count = decode.cpu_nan_count,
            .cuda_nan_count = decode.cuda_nan_count,
        });
        try parity_probe.enforce(
            decode,
            parity_probe_max_mean_abs_diff,
            parity_probe_max_abs_diff,
        );
    }
}

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

fn fillPrototypeInput(out: []f32, token: u32) void {
    for (out, 0..) |*value, i| {
        const i_u32: u32 = @intCast(i);
        const hashed = token +% (i_u32 *% 2654435761);
        const centered: i32 = @intCast(hashed & 0x3ff);
        value.* = (@as(f32, @floatFromInt(centered)) - 512.0) / 512.0;
    }
}

fn suppressPrototypeLowTokenBand(logits_out: []f32, projected_vocab: usize) void {
    if (projected_vocab <= prototype_low_token_band) return;
    const masked = @min(projected_vocab, logits_out.len);
    const band = @min(masked, prototype_low_token_band);
    if (band == 0) return;
    @memset(logits_out[0..band], -1.0e9);
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

fn kvHeadForQueryHead(query_head: usize, kv_group: usize) usize {
    return query_head / kv_group;
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

fn canUseModelEmbeddings(loaded: *const LoadedModel) bool {
    const embeddings = loaded.token_embeddings;
    if (embeddings.data_ptr == null) return false;
    return embeddings.dtype == .f32 or
        embeddings.dtype == .f16 or
        embeddings.dtype == .bf16 or
        embeddings.dtype == .grouped_affine_u4 or
        embeddings.dtype == .grouped_affine_u8;
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
                const gaffine = embeddings.gaffine orelse return false;
                const scales: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.scales.ptr))[0 .. gaffine.scales.len / @sizeOf(u16)];
                const biases: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(gaffine.biases.ptr))[0 .. gaffine.biases.len / @sizeOf(u16)];
                const rows = dim0;
                const row_ids = [_]u32{@intCast(token_idx)};
                if (embeddings.dtype == .grouped_affine_u4) {
                    const packed_vals = embeddings.asSliceUnaligned(u32);
                    try compute.cpu.quant_decode.gatherDecodeGroupedAffineU4Rows(
                        packed_vals,
                        scales,
                        biases,
                        gaffine.scales_dtype,
                        gaffine.group_size,
                        rows,
                        hidden_dim,
                        &row_ids,
                        out,
                    );
                } else {
                    const packed_vals = embeddings.asSliceUnaligned(u32);
                    try compute.cpu.quant_decode.gatherDecodeGroupedAffineU8Rows(
                        packed_vals,
                        scales,
                        biases,
                        gaffine.scales_dtype,
                        gaffine.group_size,
                        rows,
                        hidden_dim,
                        &row_ids,
                        out,
                    );
                }
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

fn fillPrototypeNormWeight(out: []f32) void {
    for (out, 0..) |*value, i| {
        const i_u32: u32 = @intCast(i);
        value.* = 1.0 + @as(f32, @floatFromInt(i_u32 % 7)) * 0.01;
    }
}

fn fillPrototypeProjection(out: []f32, hidden_dim: usize, projected_vocab: usize) void {
    var row: usize = 0;
    while (row < hidden_dim) : (row += 1) {
        const row_u32: u32 = @intCast(row + 1);
        var col: usize = 0;
        while (col < projected_vocab) : (col += 1) {
            const col_u32: u32 = @intCast(col + 1);
            const mixed = (row_u32 *% 1664525) +% (col_u32 *% 1013904223) +% 12345;
            const centered: i32 = @intCast(mixed & 0x1ff);
            out[row * projected_vocab + col] = (@as(f32, @floatFromInt(centered)) - 256.0) * 0.0005;
        }
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

fn tryPopulateProjectionFromLoadedModel(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
    d_model: usize,
    projected_vocab: usize,
    out: []f32,
) bool {
    if (loaded.lm_head) |lm_head| {
        if (tryPopulateProjectionFromWeight(allocator, &lm_head, d_model, projected_vocab, out)) return true;
    }
    return tryPopulateProjectionFromWeight(allocator, &loaded.token_embeddings, d_model, projected_vocab, out);
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

test "fillPrototypeInput is deterministic for token id" {
    var a: [8]f32 = undefined;
    var b: [8]f32 = undefined;
    fillPrototypeInput(a[0..], 42);
    fillPrototypeInput(b[0..], 42);

    for (a, b) |lhs, rhs| {
        try std.testing.expectApproxEqAbs(lhs, rhs, 0.0);
    }
}

test "fillPrototypeProjection writes non-zero coefficients" {
    var coeffs: [4 * 6]f32 = undefined;
    fillPrototypeProjection(coeffs[0..], 4, 6);

    var has_non_zero = false;
    for (coeffs) |value| {
        if (value != 0.0) {
            has_non_zero = true;
            break;
        }
    }
    try std.testing.expect(has_non_zero);
}

test "suppressPrototypeLowTokenBand masks first band when projected vocab is large" {
    var logits = [_]f32{1.0} ** (prototype_low_token_band + 32);
    suppressPrototypeLowTokenBand(logits[0..], logits.len);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0e9), logits[0], 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0e9), logits[prototype_low_token_band - 1], 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), logits[prototype_low_token_band], 0.0);
}

test "suppressPrototypeLowTokenBand keeps logits when projected vocab is small" {
    var logits = [_]f32{2.0} ** 128;
    suppressPrototypeLowTokenBand(logits[0..], 128);
    for (logits) |v| try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 0.0);
}
