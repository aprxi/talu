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
    projection_weight: LinearWeight,
    logits_dev: compute.cuda.Buffer,

    fn init(
        allocator: std.mem.Allocator,
        device: *compute.cuda.Device,
        loaded: *const LoadedModel,
        max_dff: usize,
        max_attn: usize,
        max_kv: usize,
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
            .projection_weight = projection_weight,
            .logits_dev = logits_dev,
        };
    }

    fn deinit(self: *PrototypeRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        self.logits_dev.deinit(device);
        self.projection_weight.deinit(device);
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

const BlockRuntime = struct {
    blocks: []AttentionMlpBlockRuntime,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    q_norm_blocks: usize,
    k_norm_blocks: usize,
    linear_weight_bytes: usize,
    norm_weight_bytes: usize,
    kv_cache_bytes: usize,

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
        var q_norm_blocks: usize = 0;
        var k_norm_blocks: usize = 0;
        var linear_weight_bytes: usize = 0;
        var norm_weight_bytes: usize = 0;
        var kv_cache_bytes: usize = 0;
        var blocks = try allocator.alloc(AttentionMlpBlockRuntime, layer_count);
        errdefer allocator.free(blocks);
        var initialized: usize = 0;
        errdefer {
            while (initialized > 0) {
                initialized -= 1;
                blocks[initialized].deinit(device);
            }
        }

        for (loaded.blocks, 0..) |block_weights, layer_idx| {
            const attn = switch (block_weights) {
                .attention_mlp => |weights| weights,
                else => {
                    log.warn("inference", "CUDA block runtime unsupported block kind", .{
                        .layer = layer_idx,
                    });
                    return error.UnsupportedModel;
                },
            };
            if (attn.mla_config != null) {
                log.warn("inference", "CUDA block runtime MLA not supported yet", .{ .layer = layer_idx });
                return error.UnsupportedModel;
            }
            if (attn.moe_weights != null) {
                log.warn("inference", "CUDA block runtime MoE not supported yet", .{ .layer = layer_idx });
                return error.UnsupportedModel;
            }
            if (attn.fused.qkv_proj != null or attn.fused.gate_up != null) {
                log.warn("inference", "CUDA block runtime fused weights not supported yet", .{ .layer = layer_idx });
                return error.UnsupportedModel;
            }

            const q_proj = attn.q_proj orelse return error.MissingWeight;
            const k_proj = attn.k_proj orelse return error.MissingWeight;
            const v_proj = attn.v_proj orelse return error.MissingWeight;
            const w1 = attn.w1 orelse return error.MissingWeight;
            const w2 = attn.w2 orelse return error.MissingWeight;
            const w3 = attn.w3 orelse return error.MissingWeight;
            if (layer_idx == 0) {
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

            blocks[layer_idx] = .{
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
            initialized += 1;
        }

        return .{
            .blocks = blocks,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .q_norm_blocks = q_norm_blocks,
            .k_norm_blocks = k_norm_blocks,
            .linear_weight_bytes = linear_weight_bytes,
            .norm_weight_bytes = norm_weight_bytes,
            .kv_cache_bytes = kv_cache_bytes,
        };
    }

    fn deinit(self: *BlockRuntime, allocator: std.mem.Allocator, device: *compute.cuda.Device) void {
        for (self.blocks) |*block| block.deinit(device);
        allocator.free(self.blocks);
    }

    fn maxDff(self: *const BlockRuntime) usize {
        var max_dff: usize = 0;
        for (self.blocks) |block| {
            if (block.d_ff > max_dff) max_dff = block.d_ff;
        }
        return max_dff;
    }

    fn maxAttn(self: *const BlockRuntime) usize {
        var max_attn: usize = 0;
        for (self.blocks) |block| {
            if (block.q_dim > max_attn) max_attn = block.q_dim;
        }
        return max_attn;
    }

    fn maxKv(self: *const BlockRuntime) usize {
        var max_kv: usize = 0;
        for (self.blocks) |block| {
            if (block.kv_dim > max_kv) max_kv = block.kv_dim;
        }
        return max_kv;
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
            (backend.block_runtime.q_norm_blocks != backend.block_runtime.blocks.len or backend.block_runtime.k_norm_blocks != backend.block_runtime.blocks.len))
        {
            log.warn("inference", "CUDA backend requires explicit q/k norm weights when qk_norm is enabled", .{
                .q_norm_blocks = backend.block_runtime.q_norm_blocks,
                .k_norm_blocks = backend.block_runtime.k_norm_blocks,
                .layers = backend.block_runtime.blocks.len,
            });
            return error.UnsupportedModel;
        }
        const max_dff = backend.block_runtime.maxDff();
        const max_attn = backend.block_runtime.maxAttn();
        const max_kv = backend.block_runtime.maxKv();
        backend.blas = try compute.cuda.Blas.init(&backend.device);
        errdefer backend.blas.deinit(&backend.device);
        backend.prototype = try PrototypeRuntime.init(
            allocator,
            &backend.device,
            loaded,
            max_dff,
            max_attn,
            max_kv,
            backend.max_seq_len,
            backend.n_heads,
            backend.head_dim,
        );
        errdefer backend.prototype.deinit(allocator, &backend.device);
        try backend.initKernelFunctions();

        try runMatmulSmoke(&backend);
        try runKernelSmoke(&backend);
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
            .kv_capacity_init = if (backend.block_runtime.blocks.len > 0) backend.block_runtime.blocks[0].kv_capacity else 0,
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
            .prototype_mib = bytesToMiB(backend.prototype.deviceByteSize()),
            .slot_logits_mib = bytesToMiB(std.math.mul(usize, backend.slot_logits.len, @sizeOf(f32)) catch 0),
            .stream_token_select = "gpu_argmax",
            .stream_enabled = @as(u8, @intFromBool(backend.compute_stream != null)),
            .device_blocks = backend.block_runtime.blocks.len,
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
        try self.ensureKvCapacity(position + 1);

        const rmsnorm_function = self.rmsnorm_function orelse return error.CudaKernelUnavailable;
        const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
        const silu_mul_function = self.silu_mul_function orelse return error.CudaKernelUnavailable;
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
            const block = &self.block_runtime.blocks[layer_idx];
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

            _ = try self.runGateUpProjection(&self.prototype.norm_out_dev, block);
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

        for (self.block_runtime.blocks) |*block| {
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
        if (try self.tryFusedGateUpForward(input, block)) return .fused;

        try self.linearForward(input, &block.w1, &self.prototype.ffn_gate_dev);
        try self.linearForward(input, &block.w3, &self.prototype.ffn_up_dev);
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
        block: *const AttentionMlpBlockRuntime,
    ) !bool {
        if (try self.tryFusedDenseU16GateUpForward(input, block)) return true;

        const fused_kernel = self.gaffine_u4_matvec_gate_up_function orelse return false;
        const gate = switch (block.w1) {
            .gaffine_u4 => |w| w,
            else => return false,
        };
        const up = switch (block.w3) {
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
        block: *const AttentionMlpBlockRuntime,
    ) !bool {
        const gate = switch (block.w1) {
            .dense_u16 => |w| w,
            else => return false,
        };
        const up = switch (block.w3) {
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
            const resolved = try self.kernel_registry.resolveFunction(
                kernel.op_name,
                kernel.embedded_symbol,
            );
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
    const probe_token: u32 = 42;
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

    const probe_tokens = [_]u32{probe_token};
    const cpu_slot = cpu_backend.allocSlot() orelse return error.OutOfMemory;
    defer cpu_backend.freeSlot(cpu_slot);
    try cpu_backend.prefillSlot(cpu_slot, probe_tokens[0..], cpu_logits);

    try backend.computeGpuPrototypeLogits(probe_token, 0, cuda_logits);

    const cpu_top = argmaxHost(cpu_logits);
    const cuda_top = argmaxHost(cuda_logits);

    var max_abs_diff: f32 = 0.0;
    var sum_abs_diff: f64 = 0.0;
    var finite_count: usize = 0;
    var cpu_nan_count: usize = 0;
    var cuda_nan_count: usize = 0;
    for (cpu_logits, cuda_logits) |cpu_v, cuda_v| {
        if (std.math.isNan(cpu_v)) cpu_nan_count += 1;
        if (std.math.isNan(cuda_v)) cuda_nan_count += 1;
        if (std.math.isFinite(cpu_v) and std.math.isFinite(cuda_v)) {
            const diff = @abs(cpu_v - cuda_v);
            if (diff > max_abs_diff) max_abs_diff = diff;
            sum_abs_diff += diff;
            finite_count += 1;
        }
    }
    const mean_abs_diff: f32 = if (finite_count > 0)
        @floatCast(sum_abs_diff / @as(f64, @floatFromInt(finite_count)))
    else
        0.0;

    log.info("inference", "CUDA parity probe full", .{
        .token = probe_token,
        .layers = backend.block_runtime.blocks.len,
        .cpu_top = cpu_top,
        .cuda_top = cuda_top,
        .same_top = @as(u8, @intFromBool(cpu_top == cuda_top)),
        .mean_abs_diff = mean_abs_diff,
        .max_abs_diff = max_abs_diff,
        .finite_count = finite_count,
        .cpu_nan_count = cpu_nan_count,
        .cuda_nan_count = cuda_nan_count,
    });

    if (cuda_nan_count != 0 or
        cpu_top != cuda_top or
        mean_abs_diff > parity_probe_max_mean_abs_diff or
        max_abs_diff > parity_probe_max_abs_diff)
    {
        return error.CudaParityMismatch;
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

fn runMatmulSmoke(backend: *CudaBackend) !void {
    const device = &backend.device;
    const m: usize = 2;
    const k: usize = 2;
    const n: usize = 2;

    const a = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
    };
    const b = [_]f32{
        5.0, 6.0,
        7.0, 8.0,
    };
    const expected = [_]f32{
        19.0, 22.0,
        43.0, 50.0,
    };
    var actual = [_]f32{0.0} ** (m * n);

    var a_dev = try device.allocBuffer(@sizeOf(f32) * a.len);
    defer a_dev.deinit(device);
    var b_dev = try device.allocBuffer(@sizeOf(f32) * b.len);
    defer b_dev.deinit(device);
    var c_dev = try device.allocBuffer(@sizeOf(f32) * actual.len);
    defer c_dev.deinit(device);

    try a_dev.upload(device, std.mem.sliceAsBytes(a[0..]));
    try b_dev.upload(device, std.mem.sliceAsBytes(b[0..]));
    try backend.blas.matmulF32(device, &a_dev, m, k, &b_dev, n, &c_dev);
    try c_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaSmokeMismatch;
    }

    log.info("inference", "CUDA matmul smoke passed", .{
        .m = m,
        .k = k,
        .n = n,
        .c00 = actual[0],
    });
}

fn runKernelSmoke(
    backend: *CudaBackend,
) !void {
    if (!backend.device.supportsModuleLaunch()) {
        log.info("inference", "CUDA module launch API unavailable; skipping kernel smoke", .{});
        return;
    }
    if (backend.vector_add_function == null or
        backend.mul_function == null or
        backend.copy_function == null or
        backend.copy_u16_function == null or
        backend.cast_f32_to_f16_function == null or
        backend.kv_write_f16_function == null or
        backend.rmsnorm_function == null or
        backend.rope_function == null or
        backend.rope_store_f16_function == null or
        backend.attn_scores_function == null or
        backend.attn_scores_f16_kv_function == null or
        backend.softmax_function == null or
        backend.attn_scores_heads_f16_kv_function == null or
        backend.attn_fused_heads_f16_kv_function == null or
        backend.softmax_rows_function == null or
        backend.attn_weighted_sum_function == null or
        backend.attn_weighted_sum_f16_kv_function == null or
        backend.attn_weighted_sum_heads_f16_kv_function == null or
        backend.silu_function == null or
        backend.silu_mul_function == null or
        backend.argmax_function == null or
        backend.matvec_f16_function == null or
        backend.matvec_bf16_function == null or
        backend.matvec_gate_up_f16_function == null or
        backend.matvec_gate_up_bf16_function == null or
        backend.matvec_qkv_f16_function == null or
        backend.matvec_qkv_bf16_function == null or
        backend.gaffine_u4_matvec_function == null or
        backend.gaffine_u4_matvec_gate_up_function == null or
        backend.gaffine_u4_matvec_qkv_function == null)
    {
        return error.CudaKernelUnavailable;
    }

    try runVectorAddSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.vector_add_function.?,
        backend.vector_add_source orelse .embedded_module,
    );
    try runMulSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.mul_function.?,
        backend.mul_source orelse .embedded_module,
    );
    try runCopySmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.copy_function.?,
        backend.copy_source orelse .embedded_module,
    );
    try runCopyU16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.copy_u16_function.?,
        backend.copy_u16_source orelse .embedded_module,
    );
    try runCastF32ToF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.cast_f32_to_f16_source orelse .embedded_module,
    );
    try runKvWriteF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.kv_write_f16_function.?,
        backend.kv_write_f16_source orelse .embedded_module,
    );
    try runRmsNormSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rmsnorm_function.?,
        backend.rmsnorm_source orelse .embedded_module,
    );
    try runRopeStoreF16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rope_store_f16_function.?,
        backend.rope_store_f16_source orelse .embedded_module,
    );
    try runSiluSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.silu_function.?,
        backend.silu_source orelse .embedded_module,
    );
    try runSiluMulSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.silu_mul_function.?,
        backend.silu_mul_source orelse .embedded_module,
    );
    try runAttentionSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rope_function.?,
        backend.attn_scores_function.?,
        backend.softmax_function.?,
        backend.attn_weighted_sum_function.?,
    );
    try runF16KvScalarKernelsSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.attn_scores_f16_kv_function.?,
        backend.attn_scores_f16_kv_source orelse .embedded_module,
        backend.attn_weighted_sum_f16_kv_function.?,
        backend.attn_weighted_sum_f16_kv_source orelse .embedded_module,
    );
    try runArgmaxSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.argmax_function.?,
        backend.argmax_source orelse .embedded_module,
    );
    try runMatvecU16Smoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_f16_function.?,
        backend.matvec_f16_source orelse .embedded_module,
        backend.matvec_bf16_function.?,
        backend.matvec_bf16_source orelse .embedded_module,
    );
    try runMatvecU16GateUpSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_gate_up_f16_function.?,
        backend.matvec_gate_up_f16_source orelse .embedded_module,
        backend.matvec_gate_up_bf16_function.?,
        backend.matvec_gate_up_bf16_source orelse .embedded_module,
    );
    try runMatvecU16QkvSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.matvec_qkv_f16_function.?,
        backend.matvec_qkv_f16_source orelse .embedded_module,
        backend.matvec_qkv_bf16_function.?,
        backend.matvec_qkv_bf16_source orelse .embedded_module,
    );
    try runGaffineU4MatvecSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_function.?,
        backend.gaffine_u4_matvec_source orelse .embedded_module,
    );
    try runGaffineU4MatvecGateUpSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_gate_up_function.?,
        backend.gaffine_u4_matvec_gate_up_source orelse .embedded_module,
    );
    try runGaffineU4MatvecQkvSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.gaffine_u4_matvec_qkv_function.?,
        backend.gaffine_u4_matvec_qkv_source orelse .embedded_module,
    );
    try runF16KvAttentionSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.attn_scores_heads_f16_kv_function.?,
        backend.softmax_rows_function.?,
        backend.attn_weighted_sum_heads_f16_kv_function.?,
    );
    try runF16KvAttentionFusedSmoke(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.cast_f32_to_f16_function.?,
        backend.attn_fused_heads_f16_kv_function.?,
    );
}

fn runVectorAddSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const expected = [_]f32{ 11.0, 22.0, 33.0, 44.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try lhs_dev.upload(device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(device, std.mem.sliceAsBytes(rhs[0..]));
    try compute.cuda.vector_add.runWithFunction(
        arg_pack,
        device,
        function,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA vector_add smoke passed", .{
        .n = lhs.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runMulSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const lhs = [_]f32{ 1.0, -2.0, 3.0, -4.0 };
    const rhs = [_]f32{ 10.0, 20.0, -30.0, -40.0 };
    const expected = [_]f32{ 10.0, -40.0, -90.0, 160.0 };
    var actual = [_]f32{0.0} ** lhs.len;

    var lhs_dev = try device.allocBuffer(lhs.len * @sizeOf(f32));
    defer lhs_dev.deinit(device);
    var rhs_dev = try device.allocBuffer(rhs.len * @sizeOf(f32));
    defer rhs_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try lhs_dev.upload(device, std.mem.sliceAsBytes(lhs[0..]));
    try rhs_dev.upload(device, std.mem.sliceAsBytes(rhs[0..]));
    try compute.cuda.mul.runWithFunction(
        arg_pack,
        device,
        function,
        &lhs_dev,
        &rhs_dev,
        &out_dev,
        @intCast(lhs.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA mul smoke passed", .{
        .n = lhs.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runCopySmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ 5.5, -2.0, 9.25, 0.125 };
    var actual = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.copy.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_dev,
        @intCast(input.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (input, actual) |want, got| {
        if (@abs(want - got) > 0.0001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA copy smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runCopyU16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]u16{
        @bitCast(@as(f16, 1.0)),
        @bitCast(@as(f16, -2.5)),
        @bitCast(@as(f16, 3.25)),
        @bitCast(@as(f16, 0.125)),
    };
    var actual = [_]u16{0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(u16));
    defer input_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(u16));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.copy_u16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_dev,
        @intCast(input.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));
    try std.testing.expectEqualSlices(u16, input[0..], actual[0..]);

    log.info("inference", "CUDA copy_u16 smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
    });
}

fn runCastF32ToF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ 1.0, -2.5, 3.25, 0.125 };
    var output_bits = [_]u16{0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_bits.len * @sizeOf(u16));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        @intCast(input.len),
    );
    try output_dev.download(device, std.mem.sliceAsBytes(output_bits[0..]));

    for (input, output_bits) |want, got_bits| {
        const got = dtype.fp16ToF32(got_bits);
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA cast_f32_to_f16 smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = dtype.fp16ToF32(output_bits[0]),
    });
}

fn runKvWriteF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const n_heads: u32 = 1;
    const head_dim: u32 = 4;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;

    const k_input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const v_input = [_]f32{ 5.0, -1.0, 0.5, 2.0 };
    var expected_k = k_input;
    const half: usize = @intCast(rope_dim / 2);
    var pair: usize = 0;
    while (pair < half) : (pair += 1) {
        const pair_f: f32 = @floatFromInt(pair);
        const rope_dim_f: f32 = @floatFromInt(rope_dim);
        const inv_freq = std.math.pow(f32, theta, -2.0 * pair_f / rope_dim_f);
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const s = @sin(angle);
        const c = @cos(angle);
        const lo = pair;
        const hi = half + pair;
        const x0 = k_input[lo];
        const x1 = k_input[hi];
        expected_k[lo] = x0 * c - x1 * s;
        expected_k[hi] = x0 * s + x1 * c;
    }

    var out_k_bits = [_]u16{0} ** k_input.len;
    var out_v_bits = [_]u16{0} ** v_input.len;
    var out_k = [_]f32{0.0} ** k_input.len;
    var out_v = [_]f32{0.0} ** v_input.len;

    var input_k_dev = try device.allocBuffer(k_input.len * @sizeOf(f32));
    defer input_k_dev.deinit(device);
    var input_v_dev = try device.allocBuffer(v_input.len * @sizeOf(f32));
    defer input_v_dev.deinit(device);
    var out_k_dev = try device.allocBuffer(out_k_bits.len * @sizeOf(u16));
    defer out_k_dev.deinit(device);
    var out_v_dev = try device.allocBuffer(out_v_bits.len * @sizeOf(u16));
    defer out_v_dev.deinit(device);

    try input_k_dev.upload(device, std.mem.sliceAsBytes(k_input[0..]));
    try input_v_dev.upload(device, std.mem.sliceAsBytes(v_input[0..]));
    try compute.cuda.kv_write_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_k_dev,
        &input_v_dev,
        &out_k_dev,
        &out_v_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try out_k_dev.download(device, std.mem.sliceAsBytes(out_k_bits[0..]));
    try out_v_dev.download(device, std.mem.sliceAsBytes(out_v_bits[0..]));

    for (out_k_bits, 0..) |bits, i| out_k[i] = dtype.fp16ToF32(bits);
    for (out_v_bits, 0..) |bits, i| out_v[i] = dtype.fp16ToF32(bits);

    for (expected_k, out_k) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }
    for (v_input, out_v) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA kv_write_f16 smoke passed", .{
        .source = @tagName(source),
        .k0 = out_k[0],
        .v0 = out_v[0],
    });
}

fn runRopeStoreF16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const n_heads: u32 = 1;
    const head_dim: u32 = 4;
    const rope_dim: u32 = 4;
    const position: u32 = 3;
    const theta: f32 = 10000.0;

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var expected = input;
    const half: usize = @intCast(rope_dim / 2);
    var pair: usize = 0;
    while (pair < half) : (pair += 1) {
        const pair_f: f32 = @floatFromInt(pair);
        const rope_dim_f: f32 = @floatFromInt(rope_dim);
        const inv_freq = std.math.pow(f32, theta, -2.0 * pair_f / rope_dim_f);
        const angle = @as(f32, @floatFromInt(position)) * inv_freq;
        const s = @sin(angle);
        const c = @cos(angle);
        const lo = pair;
        const hi = half + pair;
        const x0 = input[lo];
        const x1 = input[hi];
        expected[lo] = x0 * c - x1 * s;
        expected[hi] = x0 * s + x1 * c;
    }

    var output_bits = [_]u16{0} ** input.len;
    var output = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(output_bits.len * @sizeOf(u16));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.rope_store_f16.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        n_heads,
        head_dim,
        rope_dim,
        position,
        theta,
    );
    try output_dev.download(device, std.mem.sliceAsBytes(output_bits[0..]));

    for (output_bits, 0..) |bits, i| {
        output[i] = dtype.fp16ToF32(bits);
    }
    for (expected, output) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA rope_store_f16 smoke passed", .{
        .source = @tagName(source),
        .out0 = output[0],
    });
}

fn runRmsNormSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const rows: u32 = 2;
    const cols: u32 = 4;
    const eps: f32 = 1e-5;

    const input = [_]f32{
        1.0,  2.0, 3.0, 4.0,
        -1.0, 0.0, 1.0, 2.0,
    };
    const weight = [_]f32{ 1.0, 1.5, 0.5, 2.0 };
    var expected = [_]f32{0.0} ** input.len;
    var actual = [_]f32{0.0} ** input.len;

    computeRmsNormReference(&expected, &input, &weight, rows, cols, eps);

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_dev = try device.allocBuffer(weight.len * @sizeOf(f32));
    defer weight_dev.deinit(device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_dev.upload(device, std.mem.sliceAsBytes(weight[0..]));
    try compute.cuda.rmsnorm.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &weight_dev,
        &output_dev,
        rows,
        cols,
        eps,
        0.0,
    );
    try output_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA rmsnorm smoke passed", .{
        .rows = rows,
        .cols = cols,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runSiluSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const expected = [_]f32{
        -0.26894143,
        0.0,
        0.7310586,
        1.7615942,
    };
    var actual = [_]f32{0.0} ** input.len;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var output_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer output_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.silu.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &output_dev,
        @intCast(input.len),
    );
    try output_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA silu smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runSiluMulSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const gate = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const up = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const expected = [_]f32{
        -0.26894143 * 2.0,
        0.0 * 3.0,
        0.7310586 * 4.0,
        1.7615942 * 5.0,
    };
    var actual = [_]f32{0.0} ** gate.len;

    var gate_dev = try device.allocBuffer(gate.len * @sizeOf(f32));
    defer gate_dev.deinit(device);
    var up_dev = try device.allocBuffer(up.len * @sizeOf(f32));
    defer up_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try gate_dev.upload(device, std.mem.sliceAsBytes(gate[0..]));
    try up_dev.upload(device, std.mem.sliceAsBytes(up[0..]));
    try compute.cuda.silu_mul.runWithFunction(
        arg_pack,
        device,
        function,
        &gate_dev,
        &up_dev,
        &out_dev,
        @intCast(gate.len),
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.001) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA silu_mul smoke passed", .{
        .n = gate.len,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runAttentionSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    rope_function: compute.cuda.Function,
    scores_function: compute.cuda.Function,
    softmax_function: compute.cuda.Function,
    weighted_sum_function: compute.cuda.Function,
) !void {
    const head_dim: u32 = 4;
    const n_heads: u32 = 1;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;

    const query = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const key_cache = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    const scores = [_]f32{0.0} ** seq_len;
    var probs = [_]f32{0.0} ** seq_len;
    var out = [_]f32{0.0} ** head_dim;

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_dev = try device.allocBuffer(key_cache.len * @sizeOf(f32));
    defer key_dev.deinit(device);
    var value_dev = try device.allocBuffer(value_cache.len * @sizeOf(f32));
    defer value_dev.deinit(device);
    var scores_dev = try device.allocBuffer(scores.len * @sizeOf(f32));
    defer scores_dev.deinit(device);
    var probs_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer probs_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_dev.upload(device, std.mem.sliceAsBytes(key_cache[0..]));
    try value_dev.upload(device, std.mem.sliceAsBytes(value_cache[0..]));

    try compute.cuda.rope.runWithFunction(
        arg_pack,
        device,
        rope_function,
        &query_dev,
        n_heads,
        head_dim,
        head_dim,
        0,
        10000.0,
    );
    try compute.cuda.attn_scores.runWithFunction(
        arg_pack,
        device,
        scores_function,
        &query_dev,
        &key_dev,
        &scores_dev,
        seq_len,
        row_stride,
        0,
        head_dim,
        scale,
    );
    try compute.cuda.softmax.runWithFunction(
        arg_pack,
        device,
        softmax_function,
        &scores_dev,
        &probs_dev,
        seq_len,
    );
    try compute.cuda.attn_weighted_sum.runWithFunction(
        arg_pack,
        device,
        weighted_sum_function,
        &probs_dev,
        &value_dev,
        &out_dev,
        seq_len,
        row_stride,
        0,
        head_dim,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));
    try probs_dev.download(device, std.mem.sliceAsBytes(probs[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    if (@abs(probs[0] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[1] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    const expected_out0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    if (@abs(out[0] - expected_out0) > 0.02) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention smoke passed", .{
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runF16KvScalarKernelsSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    scores_f16_kv_function: compute.cuda.Function,
    scores_source: compute.cuda.registry.KernelSource,
    weighted_sum_f16_kv_function: compute.cuda.Function,
    weighted_sum_source: compute.cuda.registry.KernelSource,
) !void {
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const head_offset: u32 = 0;
    const head_dim: u32 = 2;
    const scale: f32 = 0.5;

    const query = [_]f32{ 1.0, 2.0 };
    const key_cache_f16 = [_]u16{
        @bitCast(@as(f16, 1.0)), @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 0.0)),
        @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 1.0)), @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 0.0)),
    };
    const probs = [_]f32{ 0.25, 0.75 };
    const value_cache_f16 = [_]u16{
        @bitCast(@as(f16, 2.0)), @bitCast(@as(f16, 4.0)), @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 0.0)),
        @bitCast(@as(f16, 6.0)), @bitCast(@as(f16, 8.0)), @bitCast(@as(f16, 0.0)), @bitCast(@as(f16, 0.0)),
    };
    const expected_scores = [_]f32{ 0.5, 1.0 };
    const expected_out = [_]f32{ 5.0, 7.0 };
    var scores_actual = [_]f32{0.0} ** expected_scores.len;
    var out_actual = [_]f32{0.0} ** expected_out.len;

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_dev = try device.allocBuffer(key_cache_f16.len * @sizeOf(u16));
    defer key_dev.deinit(device);
    var scores_dev = try device.allocBuffer(scores_actual.len * @sizeOf(f32));
    defer scores_dev.deinit(device);
    var probs_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer probs_dev.deinit(device);
    var value_dev = try device.allocBuffer(value_cache_f16.len * @sizeOf(u16));
    defer value_dev.deinit(device);
    var out_dev = try device.allocBuffer(out_actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_dev.upload(device, std.mem.sliceAsBytes(key_cache_f16[0..]));
    try probs_dev.upload(device, std.mem.sliceAsBytes(probs[0..]));
    try value_dev.upload(device, std.mem.sliceAsBytes(value_cache_f16[0..]));

    try compute.cuda.attn_scores_f16_kv.runWithFunction(
        arg_pack,
        device,
        scores_f16_kv_function,
        &query_dev,
        &key_dev,
        &scores_dev,
        seq_len,
        row_stride,
        head_offset,
        head_dim,
        scale,
    );
    try scores_dev.download(device, std.mem.sliceAsBytes(scores_actual[0..]));

    try compute.cuda.attn_weighted_sum_f16_kv.runWithFunction(
        arg_pack,
        device,
        weighted_sum_f16_kv_function,
        &probs_dev,
        &value_dev,
        &out_dev,
        seq_len,
        row_stride,
        head_offset,
        head_dim,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out_actual[0..]));

    for (expected_scores, scores_actual) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }
    for (expected_out, out_actual) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA f16-kv scalar kernels smoke passed", .{
        .seq = seq_len,
        .head_dim = head_dim,
        .scores_source = @tagName(scores_source),
        .weighted_sum_source = @tagName(weighted_sum_source),
        .out0 = out_actual[0],
    });
}

fn runF16KvAttentionSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    cast_f32_to_f16_function: compute.cuda.Function,
    scores_heads_f16_kv_function: compute.cuda.Function,
    softmax_rows_function: compute.cuda.Function,
    weighted_sum_heads_f16_kv_function: compute.cuda.Function,
) !void {
    const n_heads: u32 = 2;
    const kv_groups: u32 = 2;
    const head_dim: u32 = 4;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;

    const query = [_]f32{
        1.0, 0.0, 0.0, 0.0, // head 0
        0.0, 1.0, 0.0, 0.0, // head 1
    };
    const key_cache_f32 = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache_f32 = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    var probs = [_]f32{0.0} ** (n_heads * seq_len);
    var out = [_]f32{0.0} ** (n_heads * head_dim);

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_f32_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(f32));
    defer key_f32_dev.deinit(device);
    var value_f32_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(f32));
    defer value_f32_dev.deinit(device);
    var key_f16_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(u16));
    defer key_f16_dev.deinit(device);
    var value_f16_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(u16));
    defer value_f16_dev.deinit(device);
    var scores_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer scores_dev.deinit(device);
    var probs_dev = try device.allocBuffer(probs.len * @sizeOf(f32));
    defer probs_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_f32_dev.upload(device, std.mem.sliceAsBytes(key_cache_f32[0..]));
    try value_f32_dev.upload(device, std.mem.sliceAsBytes(value_cache_f32[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &key_f32_dev,
        &key_f16_dev,
        @intCast(key_cache_f32.len),
    );
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &value_f32_dev,
        &value_f16_dev,
        @intCast(value_cache_f32.len),
    );
    try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        scores_heads_f16_kv_function,
        &query_dev,
        &key_f16_dev,
        &scores_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
    );
    try compute.cuda.softmax_rows.runWithFunction(
        arg_pack,
        device,
        softmax_rows_function,
        &scores_dev,
        &probs_dev,
        n_heads,
        seq_len,
    );
    try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        weighted_sum_heads_f16_kv_function,
        &probs_dev,
        &value_f16_dev,
        &out_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));
    try probs_dev.download(device, std.mem.sliceAsBytes(probs[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    if (@abs(probs[0] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[1] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[2] - expected_p1) > 0.01) return error.CudaKernelSmokeMismatch;
    if (@abs(probs[3] - expected_p0) > 0.01) return error.CudaKernelSmokeMismatch;

    const expected_out_h0_d0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    const expected_out_h0_d1 = expected_p0 * 3.0 + expected_p1 * 1.0;
    const expected_out_h1_d0 = expected_p1 * 2.0 + expected_p0 * 4.0;
    const expected_out_h1_d1 = expected_p1 * 3.0 + expected_p0 * 1.0;
    if (@abs(out[0] - expected_out_h0_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[1] - expected_out_h0_d1) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[4] - expected_out_h1_d0) > 0.03) return error.CudaKernelSmokeMismatch;
    if (@abs(out[5] - expected_out_h1_d1) > 0.03) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention f16-kv smoke passed", .{
        .n_heads = n_heads,
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runF16KvAttentionFusedSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    cast_f32_to_f16_function: compute.cuda.Function,
    fused_function: compute.cuda.Function,
) !void {
    const n_heads: u32 = 2;
    const kv_groups: u32 = 2;
    const head_dim: u32 = 4;
    const seq_len: u32 = 2;
    const row_stride: u32 = 4;
    const scale: f32 = 0.5;
    const rope_dim: u32 = head_dim;
    const position: u32 = 0;
    const theta: f32 = 10000.0;

    const query = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const key_cache_f32 = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    };
    const value_cache_f32 = [_]f32{
        2.0, 3.0, 0.0, 0.0,
        4.0, 1.0, 0.0, 0.0,
    };
    var out = [_]f32{0.0} ** (n_heads * head_dim);

    var query_dev = try device.allocBuffer(query.len * @sizeOf(f32));
    defer query_dev.deinit(device);
    var key_f32_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(f32));
    defer key_f32_dev.deinit(device);
    var value_f32_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(f32));
    defer value_f32_dev.deinit(device);
    var key_f16_dev = try device.allocBuffer(key_cache_f32.len * @sizeOf(u16));
    defer key_f16_dev.deinit(device);
    var value_f16_dev = try device.allocBuffer(value_cache_f32.len * @sizeOf(u16));
    defer value_f16_dev.deinit(device);
    var out_dev = try device.allocBuffer(out.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try query_dev.upload(device, std.mem.sliceAsBytes(query[0..]));
    try key_f32_dev.upload(device, std.mem.sliceAsBytes(key_cache_f32[0..]));
    try value_f32_dev.upload(device, std.mem.sliceAsBytes(value_cache_f32[0..]));
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &key_f32_dev,
        &key_f16_dev,
        @intCast(key_cache_f32.len),
    );
    try compute.cuda.cast_f32_to_f16.runWithFunction(
        arg_pack,
        device,
        cast_f32_to_f16_function,
        &value_f32_dev,
        &value_f16_dev,
        @intCast(value_cache_f32.len),
    );

    try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
        arg_pack,
        device,
        fused_function,
        &query_dev,
        &key_f16_dev,
        &value_f16_dev,
        &out_dev,
        n_heads,
        seq_len,
        row_stride,
        kv_groups,
        head_dim,
        scale,
        rope_dim,
        position,
        theta,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(out[0..]));

    const expected_p0 = std.math.exp(0.5) / (std.math.exp(0.5) + std.math.exp(0.0));
    const expected_p1 = 1.0 - expected_p0;
    const expected_out_h0_d0 = expected_p0 * 2.0 + expected_p1 * 4.0;
    const expected_out_h0_d1 = expected_p0 * 3.0 + expected_p1 * 1.0;
    const expected_out_h1_d0 = expected_p1 * 2.0 + expected_p0 * 4.0;
    const expected_out_h1_d1 = expected_p1 * 3.0 + expected_p0 * 1.0;
    if (@abs(out[0] - expected_out_h0_d0) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[1] - expected_out_h0_d1) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[4] - expected_out_h1_d0) > 0.04) return error.CudaKernelSmokeMismatch;
    if (@abs(out[5] - expected_out_h1_d1) > 0.04) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA attention fused f16-kv smoke passed", .{
        .n_heads = n_heads,
        .seq = seq_len,
        .head_dim = head_dim,
        .out0 = out[0],
    });
}

fn runArgmaxSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const input = [_]f32{ -1.0, 4.5, 3.25, 4.5, 0.0 };
    const expected_index: u32 = 1;
    var actual_index: u32 = 0;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var out_index_dev = try device.allocBuffer(@sizeOf(u32));
    defer out_index_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try compute.cuda.argmax.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &out_index_dev,
        @intCast(input.len),
    );
    try out_index_dev.download(device, std.mem.asBytes(&actual_index));

    if (actual_index != expected_index) return error.CudaKernelSmokeMismatch;

    log.info("inference", "CUDA argmax smoke passed", .{
        .n = input.len,
        .source = @tagName(source),
        .idx = actual_index,
    });
}

fn runMatvecU16Smoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const out_dim_usize: usize = 3;
    const in_dim: u32 = in_dim_usize;
    const out_dim: u32 = out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var weights_bf16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var weights_f16: [out_dim_usize * in_dim_usize]u16 = undefined;
    var expected: [out_dim_usize]f32 = [_]f32{0.0} ** out_dim_usize;
    for (0..out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 1));
            const idx = row * in_dim_usize + col;
            weights_bf16[idx] = dtype.f32ToBf16(w);
            weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected[row] = acc;
    }

    var actual_bf16 = [_]f32{0.0} ** out_dim_usize;
    var actual_f16 = [_]f32{0.0} ** out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var weight_bf16_dev = try device.allocBuffer(weights_bf16.len * @sizeOf(u16));
    defer weight_bf16_dev.deinit(device);
    var weight_f16_dev = try device.allocBuffer(weights_f16.len * @sizeOf(u16));
    defer weight_f16_dev.deinit(device);
    var out_bf16_dev = try device.allocBuffer(actual_bf16.len * @sizeOf(f32));
    defer out_bf16_dev.deinit(device);
    var out_f16_dev = try device.allocBuffer(actual_f16.len * @sizeOf(f32));
    defer out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try weight_bf16_dev.upload(device, std.mem.sliceAsBytes(weights_bf16[0..]));
    try weight_f16_dev.upload(device, std.mem.sliceAsBytes(weights_f16[0..]));
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &weight_bf16_dev,
        &out_bf16_dev,
        in_dim,
        out_dim,
    );
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &weight_f16_dev,
        &out_f16_dev,
        in_dim,
        out_dim,
    );
    try out_bf16_dev.download(device, std.mem.sliceAsBytes(actual_bf16[0..]));
    try out_f16_dev.download(device, std.mem.sliceAsBytes(actual_f16[0..]));

    for (expected, actual_bf16) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }
    for (expected, actual_f16) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned input/weight pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const weight_bytes = weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var weight_unaligned_raw_dev = try device.allocBuffer(weight_bytes + weight_pad);
    defer weight_unaligned_raw_dev.deinit(device);
    var out_unaligned_dev = try device.allocBuffer(actual_bf16.len * @sizeOf(f32));
    defer out_unaligned_dev.deinit(device);
    var actual_unaligned = [_]f32{0.0} ** out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var weight_blob: [weight_pad + weight_bytes]u8 = [_]u8{0} ** (weight_pad + weight_bytes);
    @memcpy(weight_blob[weight_pad..], std.mem.sliceAsBytes(weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try weight_unaligned_raw_dev.upload(device, weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var weight_unaligned_dev = try bufferSlice(&weight_unaligned_raw_dev, weight_pad, weight_bytes);
    try compute.cuda.matvec_u16.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &weight_unaligned_dev,
        &out_unaligned_dev,
        in_dim,
        out_dim,
    );
    try out_unaligned_dev.download(device, std.mem.sliceAsBytes(actual_unaligned[0..]));
    for (expected, actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16 smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .out0_f16 = actual_f16[0],
        .out0_bf16 = actual_bf16[0],
    });
}

fn runMatvecU16GateUpSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const gate_out_dim_usize: usize = 3;
    const up_out_dim_usize: usize = 4;
    const in_dim: u32 = in_dim_usize;
    const gate_out_dim: u32 = gate_out_dim_usize;
    const up_out_dim: u32 = up_out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var gate_weights_bf16: [gate_out_dim_usize * in_dim_usize]u16 = undefined;
    var gate_weights_f16: [gate_out_dim_usize * in_dim_usize]u16 = undefined;
    var up_weights_bf16: [up_out_dim_usize * in_dim_usize]u16 = undefined;
    var up_weights_f16: [up_out_dim_usize * in_dim_usize]u16 = undefined;
    var expected_gate: [gate_out_dim_usize]f32 = [_]f32{0.0} ** gate_out_dim_usize;
    var expected_up: [up_out_dim_usize]f32 = [_]f32{0.0} ** up_out_dim_usize;

    for (0..gate_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 2));
            const idx = row * in_dim_usize + col;
            gate_weights_bf16[idx] = dtype.f32ToBf16(w);
            gate_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_gate[row] = acc;
    }
    for (0..up_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 2) * (col + 1));
            const idx = row * in_dim_usize + col;
            up_weights_bf16[idx] = dtype.f32ToBf16(w);
            up_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_up[row] = acc;
    }

    var gate_actual_f16 = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_f16 = [_]f32{0.0} ** up_out_dim_usize;
    var gate_actual_bf16 = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_bf16 = [_]f32{0.0} ** up_out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var gate_weight_bf16_dev = try device.allocBuffer(gate_weights_bf16.len * @sizeOf(u16));
    defer gate_weight_bf16_dev.deinit(device);
    var up_weight_bf16_dev = try device.allocBuffer(up_weights_bf16.len * @sizeOf(u16));
    defer up_weight_bf16_dev.deinit(device);
    var gate_weight_f16_dev = try device.allocBuffer(gate_weights_f16.len * @sizeOf(u16));
    defer gate_weight_f16_dev.deinit(device);
    var up_weight_f16_dev = try device.allocBuffer(up_weights_f16.len * @sizeOf(u16));
    defer up_weight_f16_dev.deinit(device);
    var gate_out_bf16_dev = try device.allocBuffer(gate_actual_bf16.len * @sizeOf(f32));
    defer gate_out_bf16_dev.deinit(device);
    var up_out_bf16_dev = try device.allocBuffer(up_actual_bf16.len * @sizeOf(f32));
    defer up_out_bf16_dev.deinit(device);
    var gate_out_f16_dev = try device.allocBuffer(gate_actual_f16.len * @sizeOf(f32));
    defer gate_out_f16_dev.deinit(device);
    var up_out_f16_dev = try device.allocBuffer(up_actual_f16.len * @sizeOf(f32));
    defer up_out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try gate_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(gate_weights_bf16[0..]));
    try up_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(up_weights_bf16[0..]));
    try gate_weight_f16_dev.upload(device, std.mem.sliceAsBytes(gate_weights_f16[0..]));
    try up_weight_f16_dev.upload(device, std.mem.sliceAsBytes(up_weights_f16[0..]));

    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &gate_weight_bf16_dev,
        &gate_out_bf16_dev,
        gate_out_dim,
        &up_weight_bf16_dev,
        &up_out_bf16_dev,
        up_out_dim,
        in_dim,
    );
    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &gate_weight_f16_dev,
        &gate_out_f16_dev,
        gate_out_dim,
        &up_weight_f16_dev,
        &up_out_f16_dev,
        up_out_dim,
        in_dim,
    );
    try gate_out_bf16_dev.download(device, std.mem.sliceAsBytes(gate_actual_bf16[0..]));
    try up_out_bf16_dev.download(device, std.mem.sliceAsBytes(up_actual_bf16[0..]));
    try gate_out_f16_dev.download(device, std.mem.sliceAsBytes(gate_actual_f16[0..]));
    try up_out_f16_dev.download(device, std.mem.sliceAsBytes(up_actual_f16[0..]));

    for (expected_gate, gate_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_gate, gate_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const gate_weight_bytes = gate_weights_bf16.len * @sizeOf(u16);
    const up_weight_bytes = up_weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var gate_weight_unaligned_raw_dev = try device.allocBuffer(gate_weight_bytes + weight_pad);
    defer gate_weight_unaligned_raw_dev.deinit(device);
    var up_weight_unaligned_raw_dev = try device.allocBuffer(up_weight_bytes + weight_pad);
    defer up_weight_unaligned_raw_dev.deinit(device);
    var gate_out_unaligned_dev = try device.allocBuffer(gate_actual_bf16.len * @sizeOf(f32));
    defer gate_out_unaligned_dev.deinit(device);
    var up_out_unaligned_dev = try device.allocBuffer(up_actual_bf16.len * @sizeOf(f32));
    defer up_out_unaligned_dev.deinit(device);

    var gate_actual_unaligned = [_]f32{0.0} ** gate_out_dim_usize;
    var up_actual_unaligned = [_]f32{0.0} ** up_out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var gate_weight_blob: [weight_pad + gate_weight_bytes]u8 = [_]u8{0} ** (weight_pad + gate_weight_bytes);
    @memcpy(gate_weight_blob[weight_pad..], std.mem.sliceAsBytes(gate_weights_bf16[0..]));
    var up_weight_blob: [weight_pad + up_weight_bytes]u8 = [_]u8{0} ** (weight_pad + up_weight_bytes);
    @memcpy(up_weight_blob[weight_pad..], std.mem.sliceAsBytes(up_weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try gate_weight_unaligned_raw_dev.upload(device, gate_weight_blob[0..]);
    try up_weight_unaligned_raw_dev.upload(device, up_weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var gate_weight_unaligned_dev = try bufferSlice(&gate_weight_unaligned_raw_dev, weight_pad, gate_weight_bytes);
    var up_weight_unaligned_dev = try bufferSlice(&up_weight_unaligned_raw_dev, weight_pad, up_weight_bytes);
    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &gate_weight_unaligned_dev,
        &gate_out_unaligned_dev,
        gate_out_dim,
        &up_weight_unaligned_dev,
        &up_out_unaligned_dev,
        up_out_dim,
        in_dim,
    );
    try gate_out_unaligned_dev.download(device, std.mem.sliceAsBytes(gate_actual_unaligned[0..]));
    try up_out_unaligned_dev.download(device, std.mem.sliceAsBytes(up_actual_unaligned[0..]));

    for (expected_gate, gate_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, up_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16_gate_up smoke passed", .{
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .gate0 = gate_actual_f16[0],
        .up0 = up_actual_f16[0],
    });
}

fn runMatvecU16QkvSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function_f16: compute.cuda.Function,
    source_f16: compute.cuda.registry.KernelSource,
    function_bf16: compute.cuda.Function,
    source_bf16: compute.cuda.registry.KernelSource,
) !void {
    const in_dim_usize: usize = 8;
    const q_out_dim_usize: usize = 3;
    const k_out_dim_usize: usize = 2;
    const v_out_dim_usize: usize = 4;
    const in_dim: u32 = in_dim_usize;
    const q_out_dim: u32 = q_out_dim_usize;
    const k_out_dim: u32 = k_out_dim_usize;
    const v_out_dim: u32 = v_out_dim_usize;

    var input: [in_dim_usize]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @floatFromInt(i + 1);

    var q_weights_bf16: [q_out_dim_usize * in_dim_usize]u16 = undefined;
    var q_weights_f16: [q_out_dim_usize * in_dim_usize]u16 = undefined;
    var k_weights_bf16: [k_out_dim_usize * in_dim_usize]u16 = undefined;
    var k_weights_f16: [k_out_dim_usize * in_dim_usize]u16 = undefined;
    var v_weights_bf16: [v_out_dim_usize * in_dim_usize]u16 = undefined;
    var v_weights_f16: [v_out_dim_usize * in_dim_usize]u16 = undefined;

    var expected_q: [q_out_dim_usize]f32 = [_]f32{0.0} ** q_out_dim_usize;
    var expected_k: [k_out_dim_usize]f32 = [_]f32{0.0} ** k_out_dim_usize;
    var expected_v: [v_out_dim_usize]f32 = [_]f32{0.0} ** v_out_dim_usize;

    for (0..q_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 1) * (col + 1));
            const idx = row * in_dim_usize + col;
            q_weights_bf16[idx] = dtype.f32ToBf16(w);
            q_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_q[row] = acc;
    }
    for (0..k_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 2) * (col + 1));
            const idx = row * in_dim_usize + col;
            k_weights_bf16[idx] = dtype.f32ToBf16(w);
            k_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_k[row] = acc;
    }
    for (0..v_out_dim_usize) |row| {
        var acc: f32 = 0.0;
        for (0..in_dim_usize) |col| {
            const w: f32 = @floatFromInt((row + 3) * (col + 1));
            const idx = row * in_dim_usize + col;
            v_weights_bf16[idx] = dtype.f32ToBf16(w);
            v_weights_f16[idx] = @bitCast(@as(f16, @floatCast(w)));
            acc += input[col] * w;
        }
        expected_v[row] = acc;
    }

    var q_actual_f16 = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_f16 = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_f16 = [_]f32{0.0} ** v_out_dim_usize;
    var q_actual_bf16 = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_bf16 = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_bf16 = [_]f32{0.0} ** v_out_dim_usize;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var q_weight_bf16_dev = try device.allocBuffer(q_weights_bf16.len * @sizeOf(u16));
    defer q_weight_bf16_dev.deinit(device);
    var k_weight_bf16_dev = try device.allocBuffer(k_weights_bf16.len * @sizeOf(u16));
    defer k_weight_bf16_dev.deinit(device);
    var v_weight_bf16_dev = try device.allocBuffer(v_weights_bf16.len * @sizeOf(u16));
    defer v_weight_bf16_dev.deinit(device);
    var q_weight_f16_dev = try device.allocBuffer(q_weights_f16.len * @sizeOf(u16));
    defer q_weight_f16_dev.deinit(device);
    var k_weight_f16_dev = try device.allocBuffer(k_weights_f16.len * @sizeOf(u16));
    defer k_weight_f16_dev.deinit(device);
    var v_weight_f16_dev = try device.allocBuffer(v_weights_f16.len * @sizeOf(u16));
    defer v_weight_f16_dev.deinit(device);
    var q_out_bf16_dev = try device.allocBuffer(q_actual_bf16.len * @sizeOf(f32));
    defer q_out_bf16_dev.deinit(device);
    var k_out_bf16_dev = try device.allocBuffer(k_actual_bf16.len * @sizeOf(f32));
    defer k_out_bf16_dev.deinit(device);
    var v_out_bf16_dev = try device.allocBuffer(v_actual_bf16.len * @sizeOf(f32));
    defer v_out_bf16_dev.deinit(device);
    var q_out_f16_dev = try device.allocBuffer(q_actual_f16.len * @sizeOf(f32));
    defer q_out_f16_dev.deinit(device);
    var k_out_f16_dev = try device.allocBuffer(k_actual_f16.len * @sizeOf(f32));
    defer k_out_f16_dev.deinit(device);
    var v_out_f16_dev = try device.allocBuffer(v_actual_f16.len * @sizeOf(f32));
    defer v_out_f16_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try q_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(q_weights_bf16[0..]));
    try k_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(k_weights_bf16[0..]));
    try v_weight_bf16_dev.upload(device, std.mem.sliceAsBytes(v_weights_bf16[0..]));
    try q_weight_f16_dev.upload(device, std.mem.sliceAsBytes(q_weights_f16[0..]));
    try k_weight_f16_dev.upload(device, std.mem.sliceAsBytes(k_weights_f16[0..]));
    try v_weight_f16_dev.upload(device, std.mem.sliceAsBytes(v_weights_f16[0..]));

    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_dev,
        &q_weight_bf16_dev,
        &q_out_bf16_dev,
        q_out_dim,
        &k_weight_bf16_dev,
        &k_out_bf16_dev,
        k_out_dim,
        &v_weight_bf16_dev,
        &v_out_bf16_dev,
        v_out_dim,
        in_dim,
    );
    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_f16,
        &input_dev,
        &q_weight_f16_dev,
        &q_out_f16_dev,
        q_out_dim,
        &k_weight_f16_dev,
        &k_out_f16_dev,
        k_out_dim,
        &v_weight_f16_dev,
        &v_out_f16_dev,
        v_out_dim,
        in_dim,
    );
    try q_out_bf16_dev.download(device, std.mem.sliceAsBytes(q_actual_bf16[0..]));
    try k_out_bf16_dev.download(device, std.mem.sliceAsBytes(k_actual_bf16[0..]));
    try v_out_bf16_dev.download(device, std.mem.sliceAsBytes(v_actual_bf16[0..]));
    try q_out_f16_dev.download(device, std.mem.sliceAsBytes(q_actual_f16[0..]));
    try k_out_f16_dev.download(device, std.mem.sliceAsBytes(k_actual_f16[0..]));
    try v_out_f16_dev.download(device, std.mem.sliceAsBytes(v_actual_f16[0..]));

    for (expected_q, q_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_q, q_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_bf16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_f16) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    // Exercise non-vectorized fallback path with intentionally unaligned pointers.
    const input_bytes = input.len * @sizeOf(f32);
    const q_weight_bytes = q_weights_bf16.len * @sizeOf(u16);
    const k_weight_bytes = k_weights_bf16.len * @sizeOf(u16);
    const v_weight_bytes = v_weights_bf16.len * @sizeOf(u16);
    const input_pad: usize = 4;
    const weight_pad: usize = 2;

    var input_unaligned_raw_dev = try device.allocBuffer(input_bytes + input_pad);
    defer input_unaligned_raw_dev.deinit(device);
    var q_weight_unaligned_raw_dev = try device.allocBuffer(q_weight_bytes + weight_pad);
    defer q_weight_unaligned_raw_dev.deinit(device);
    var k_weight_unaligned_raw_dev = try device.allocBuffer(k_weight_bytes + weight_pad);
    defer k_weight_unaligned_raw_dev.deinit(device);
    var v_weight_unaligned_raw_dev = try device.allocBuffer(v_weight_bytes + weight_pad);
    defer v_weight_unaligned_raw_dev.deinit(device);
    var q_out_unaligned_dev = try device.allocBuffer(q_actual_bf16.len * @sizeOf(f32));
    defer q_out_unaligned_dev.deinit(device);
    var k_out_unaligned_dev = try device.allocBuffer(k_actual_bf16.len * @sizeOf(f32));
    defer k_out_unaligned_dev.deinit(device);
    var v_out_unaligned_dev = try device.allocBuffer(v_actual_bf16.len * @sizeOf(f32));
    defer v_out_unaligned_dev.deinit(device);

    var q_actual_unaligned = [_]f32{0.0} ** q_out_dim_usize;
    var k_actual_unaligned = [_]f32{0.0} ** k_out_dim_usize;
    var v_actual_unaligned = [_]f32{0.0} ** v_out_dim_usize;

    var input_blob: [input_pad + input_bytes]u8 = [_]u8{0} ** (input_pad + input_bytes);
    @memcpy(input_blob[input_pad..], std.mem.sliceAsBytes(input[0..]));
    var q_weight_blob: [weight_pad + q_weight_bytes]u8 = [_]u8{0} ** (weight_pad + q_weight_bytes);
    @memcpy(q_weight_blob[weight_pad..], std.mem.sliceAsBytes(q_weights_bf16[0..]));
    var k_weight_blob: [weight_pad + k_weight_bytes]u8 = [_]u8{0} ** (weight_pad + k_weight_bytes);
    @memcpy(k_weight_blob[weight_pad..], std.mem.sliceAsBytes(k_weights_bf16[0..]));
    var v_weight_blob: [weight_pad + v_weight_bytes]u8 = [_]u8{0} ** (weight_pad + v_weight_bytes);
    @memcpy(v_weight_blob[weight_pad..], std.mem.sliceAsBytes(v_weights_bf16[0..]));

    try input_unaligned_raw_dev.upload(device, input_blob[0..]);
    try q_weight_unaligned_raw_dev.upload(device, q_weight_blob[0..]);
    try k_weight_unaligned_raw_dev.upload(device, k_weight_blob[0..]);
    try v_weight_unaligned_raw_dev.upload(device, v_weight_blob[0..]);

    var input_unaligned_dev = try bufferSlice(&input_unaligned_raw_dev, input_pad, input_bytes);
    var q_weight_unaligned_dev = try bufferSlice(&q_weight_unaligned_raw_dev, weight_pad, q_weight_bytes);
    var k_weight_unaligned_dev = try bufferSlice(&k_weight_unaligned_raw_dev, weight_pad, k_weight_bytes);
    var v_weight_unaligned_dev = try bufferSlice(&v_weight_unaligned_raw_dev, weight_pad, v_weight_bytes);
    try compute.cuda.matvec_u16_qkv.runWithFunction(
        arg_pack,
        device,
        function_bf16,
        &input_unaligned_dev,
        &q_weight_unaligned_dev,
        &q_out_unaligned_dev,
        q_out_dim,
        &k_weight_unaligned_dev,
        &k_out_unaligned_dev,
        k_out_dim,
        &v_weight_unaligned_dev,
        &v_out_unaligned_dev,
        v_out_dim,
        in_dim,
    );
    try q_out_unaligned_dev.download(device, std.mem.sliceAsBytes(q_actual_unaligned[0..]));
    try k_out_unaligned_dev.download(device, std.mem.sliceAsBytes(k_actual_unaligned[0..]));
    try v_out_unaligned_dev.download(device, std.mem.sliceAsBytes(v_actual_unaligned[0..]));

    for (expected_q, q_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, k_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, v_actual_unaligned) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA matvec_u16_qkv smoke passed", .{
        .source_f16 = @tagName(source_f16),
        .source_bf16 = @tagName(source_bf16),
        .q0 = q_actual_f16[0],
        .k0 = k_actual_f16[0],
        .v0 = v_actual_f16[0],
    });
}

fn runGaffineU4MatvecSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(2.0),
    };
    const biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(1.0),
    };
    const expected = [_]f32{
        28.0,
        64.0,
    };
    var actual = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer packed_dev.deinit(device);
    var scales_dev = try device.allocBuffer(scales.len * @sizeOf(u16));
    defer scales_dev.deinit(device);
    var biases_dev = try device.allocBuffer(biases.len * @sizeOf(u16));
    defer biases_dev.deinit(device);
    var out_dev = try device.allocBuffer(actual.len * @sizeOf(f32));
    defer out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try scales_dev.upload(device, std.mem.sliceAsBytes(scales[0..]));
    try biases_dev.upload(device, std.mem.sliceAsBytes(biases[0..]));

    try compute.cuda.gaffine_u4_matvec.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &packed_dev,
        &scales_dev,
        &biases_dev,
        &out_dev,
        in_dim,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
    );
    try out_dev.download(device, std.mem.sliceAsBytes(actual[0..]));

    for (expected, actual) |want, got| {
        if (@abs(want - got) > 0.01) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .out0 = actual[0],
    });
}

fn runGaffineU4MatvecGateUpSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const packed_words = [_]u32{
        0x7654_3210,
        0x0123_4567,
    };
    const gate_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const gate_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const up_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const up_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_gate = [_]f32{ 168.0, 84.0 };
    const expected_up = [_]f32{ 372.0, 204.0 };
    var out_gate = [_]f32{0.0} ** out_dim;
    var out_up = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var gate_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer gate_packed_dev.deinit(device);
    var gate_scales_dev = try device.allocBuffer(gate_scales.len * @sizeOf(u16));
    defer gate_scales_dev.deinit(device);
    var gate_biases_dev = try device.allocBuffer(gate_biases.len * @sizeOf(u16));
    defer gate_biases_dev.deinit(device);
    var out_gate_dev = try device.allocBuffer(out_gate.len * @sizeOf(f32));
    defer out_gate_dev.deinit(device);

    var up_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer up_packed_dev.deinit(device);
    var up_scales_dev = try device.allocBuffer(up_scales.len * @sizeOf(u16));
    defer up_scales_dev.deinit(device);
    var up_biases_dev = try device.allocBuffer(up_biases.len * @sizeOf(u16));
    defer up_biases_dev.deinit(device);
    var out_up_dev = try device.allocBuffer(out_up.len * @sizeOf(f32));
    defer out_up_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try gate_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try gate_scales_dev.upload(device, std.mem.sliceAsBytes(gate_scales[0..]));
    try gate_biases_dev.upload(device, std.mem.sliceAsBytes(gate_biases[0..]));
    try up_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try up_scales_dev.upload(device, std.mem.sliceAsBytes(up_scales[0..]));
    try up_biases_dev.upload(device, std.mem.sliceAsBytes(up_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_gate_up.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &gate_packed_dev,
        &gate_scales_dev,
        &gate_biases_dev,
        &out_gate_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &up_packed_dev,
        &up_scales_dev,
        &up_biases_dev,
        &out_up_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
    );
    try out_gate_dev.download(device, std.mem.sliceAsBytes(out_gate[0..]));
    try out_up_dev.download(device, std.mem.sliceAsBytes(out_up[0..]));

    for (expected_gate, out_gate) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_up, out_up) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec_gate_up smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .gate0 = out_gate[0],
        .up0 = out_up[0],
    });
}

fn runGaffineU4MatvecQkvSmoke(
    arg_pack: *compute.cuda.ArgPack,
    device: *compute.cuda.Device,
    function: compute.cuda.Function,
    source: compute.cuda.registry.KernelSource,
) !void {
    const in_dim: u32 = 8;
    const out_dim: u32 = 2;
    const group_size: u32 = 8;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const packed_words = [_]u32{
        0x7654_3210, // row 0 => 0..7
        0x0123_4567, // row 1 => 7..0
    };
    const q_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const q_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const k_scales = [_]u16{
        dtype.f32ToBf16(2.0),
        dtype.f32ToBf16(2.0),
    };
    const k_biases = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const v_scales = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const v_biases = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    const expected_q = [_]f32{ 168.0, 84.0 };
    const expected_k = [_]f32{ 336.0, 168.0 };
    const expected_v = [_]f32{ 204.0, 120.0 };
    var out_q = [_]f32{0.0} ** out_dim;
    var out_k = [_]f32{0.0} ** out_dim;
    var out_v = [_]f32{0.0} ** out_dim;

    var input_dev = try device.allocBuffer(input.len * @sizeOf(f32));
    defer input_dev.deinit(device);
    var q_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer q_packed_dev.deinit(device);
    var q_scales_dev = try device.allocBuffer(q_scales.len * @sizeOf(u16));
    defer q_scales_dev.deinit(device);
    var q_biases_dev = try device.allocBuffer(q_biases.len * @sizeOf(u16));
    defer q_biases_dev.deinit(device);
    var q_out_dev = try device.allocBuffer(out_q.len * @sizeOf(f32));
    defer q_out_dev.deinit(device);

    var k_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer k_packed_dev.deinit(device);
    var k_scales_dev = try device.allocBuffer(k_scales.len * @sizeOf(u16));
    defer k_scales_dev.deinit(device);
    var k_biases_dev = try device.allocBuffer(k_biases.len * @sizeOf(u16));
    defer k_biases_dev.deinit(device);
    var k_out_dev = try device.allocBuffer(out_k.len * @sizeOf(f32));
    defer k_out_dev.deinit(device);

    var v_packed_dev = try device.allocBuffer(packed_words.len * @sizeOf(u32));
    defer v_packed_dev.deinit(device);
    var v_scales_dev = try device.allocBuffer(v_scales.len * @sizeOf(u16));
    defer v_scales_dev.deinit(device);
    var v_biases_dev = try device.allocBuffer(v_biases.len * @sizeOf(u16));
    defer v_biases_dev.deinit(device);
    var v_out_dev = try device.allocBuffer(out_v.len * @sizeOf(f32));
    defer v_out_dev.deinit(device);

    try input_dev.upload(device, std.mem.sliceAsBytes(input[0..]));
    try q_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try q_scales_dev.upload(device, std.mem.sliceAsBytes(q_scales[0..]));
    try q_biases_dev.upload(device, std.mem.sliceAsBytes(q_biases[0..]));
    try k_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try k_scales_dev.upload(device, std.mem.sliceAsBytes(k_scales[0..]));
    try k_biases_dev.upload(device, std.mem.sliceAsBytes(k_biases[0..]));
    try v_packed_dev.upload(device, std.mem.sliceAsBytes(packed_words[0..]));
    try v_scales_dev.upload(device, std.mem.sliceAsBytes(v_scales[0..]));
    try v_biases_dev.upload(device, std.mem.sliceAsBytes(v_biases[0..]));

    try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
        arg_pack,
        device,
        function,
        &input_dev,
        &q_packed_dev,
        &q_scales_dev,
        &q_biases_dev,
        &q_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &k_packed_dev,
        &k_scales_dev,
        &k_biases_dev,
        &k_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        &v_packed_dev,
        &v_scales_dev,
        &v_biases_dev,
        &v_out_dev,
        out_dim,
        group_size,
        gaffine_scales_dtype_bf16,
        in_dim,
    );
    try q_out_dev.download(device, std.mem.sliceAsBytes(out_q[0..]));
    try k_out_dev.download(device, std.mem.sliceAsBytes(out_k[0..]));
    try v_out_dev.download(device, std.mem.sliceAsBytes(out_v[0..]));

    for (expected_q, out_q) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_k, out_k) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }
    for (expected_v, out_v) |want, got| {
        if (@abs(want - got) > 0.02) return error.CudaKernelSmokeMismatch;
    }

    log.info("inference", "CUDA gaffine_u4_matvec_qkv smoke passed", .{
        .in_dim = in_dim,
        .out_dim = out_dim,
        .source = @tagName(source),
        .q0 = out_q[0],
        .k0 = out_k[0],
        .v0 = out_v[0],
    });
}

fn computeRmsNormReference(
    out: []f32,
    input: []const f32,
    weight: []const f32,
    rows: u32,
    cols: u32,
    eps: f32,
) void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    var row: usize = 0;
    while (row < rows_usize) : (row += 1) {
        const base = row * cols_usize;
        var sum_sq: f32 = 0.0;
        var col: usize = 0;
        while (col < cols_usize) : (col += 1) {
            const v = input[base + col];
            sum_sq += v * v;
        }
        const mean_sq = sum_sq / @as(f32, @floatFromInt(cols_usize));
        const inv_rms = 1.0 / std.math.sqrt(mean_sq + eps);
        col = 0;
        while (col < cols_usize) : (col += 1) {
            out[base + col] = input[base + col] * inv_rms * weight[col];
        }
    }
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

fn materializeTensorF32(allocator: std.mem.Allocator, src: *const Tensor) ![]f32 {
    if (src.data_ptr == null) return error.InvalidArgument;
    if (src.n_dims < 1 or src.n_dims > 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = if (src.n_dims == 1) 1 else blk: {
        if (src.shape[1] <= 0) return error.InvalidArgument;
        break :blk @intCast(src.shape[1]);
    };
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;

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
