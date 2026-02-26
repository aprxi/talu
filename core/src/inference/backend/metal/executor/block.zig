//! Metal backend block executor.
//!
//! Centralizes single-layer lazy graph assembly so model-level orchestration
//! can delegate layer work through a stable `TransformerBlock.forward` surface.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
const log = @import("../../../../log.zig");
const runtime_graph = @import("../runtime_graph.zig");
const attention_kernel = @import("../kernels/attention.zig");
const ffn_kernel = @import("../kernels/ffn.zig");
const mamba_kernel = @import("../kernels/mamba.zig");
const mla_kernel = @import("../kernels/mla_attention.zig");
const moe_kernel = @import("../kernels/moe.zig");
const norm_kernel = @import("../kernels/norm.zig");
const shortconv_kernel = @import("../kernels/shortconv.zig");
const mlx_graph = compute.metal.graph;

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;

pub const TransformerBlock = struct {
    fn finalOutputBuffer(program: []const layer_ops.LayerOp) layer_ops.BufferId {
        if (program.len == 0) return .residual;
        const last = program[program.len - 1];
        return switch (last) {
            .kernel => |k| k.out,
            .add => .residual,
            .linear => |op| op.out,
            .matmul => |op| op.out,
            .softmax => |op| op.out,
            .silu => |op| op.out,
            .gelu => |op| op.out,
            .mul => |op| op.out,
            .add_tensor => |op| op.out,
            .add_scalar => |op| op.out,
            .mul_scalar => |op| op.out,
            .mean => |op| op.out,
            .pow => |op| op.out,
            .rsqrt => |op| op.out,
            .add_param => |op| op.out,
            .add_param_scalar => |op| op.out,
            .mul_param => |op| op.out,
            .reshape => |op| op.out,
            .transpose => |op| op.out,
            .rope => |op| op.out,
            .triu => |op| op.out,
            .sdpa => |op| op.out,
            .patch_embed => |op| op.out,
            .spatial_merge => |op| op.out,
            .deepstack_extract => |op| op.out,
            .scatter => |op| op.out,
            else => .residual,
        };
    }

    fn isCoreProgramBuffer(id: layer_ops.BufferId) bool {
        return switch (id) {
            .residual, .norm_out, .branch_out => true,
            else => false,
        };
    }

    fn validateLayerProgram(program: []const layer_ops.LayerOp, layer_idx: usize, kind: usize) !void {
        for (program, 0..) |op, op_idx| {
            switch (op) {
                .kernel => |kernel_op| {
                    if (!isCoreProgramBuffer(kernel_op.in) or !isCoreProgramBuffer(kernel_op.out)) {
                        log.warn("inference", "Metal LayerOp program uses unsupported buffer id", .{
                            .layer = layer_idx,
                            .op_index = op_idx,
                            .kind = kind,
                            .debug_type = @tagName(kernel_op.debug_type),
                        });
                        return error.NotImplemented;
                    }
                    switch (kernel_op.debug_type) {
                        .norm, .multihead_attention, .shortconv, .mlp, .moe, .mamba_mixer => {},
                        else => {
                            log.warn("inference", "Metal LayerOp program uses unsupported kernel debug type", .{
                                .layer = layer_idx,
                                .op_index = op_idx,
                                .kind = kind,
                                .debug_type = @tagName(kernel_op.debug_type),
                            });
                            return error.NotImplemented;
                        },
                    }
                },
                .add => |add_op| {
                    if (!isCoreProgramBuffer(add_op.branch)) {
                        log.warn("inference", "Metal LayerOp add uses unsupported buffer id", .{
                            .layer = layer_idx,
                            .op_index = op_idx,
                            .kind = kind,
                        });
                        return error.NotImplemented;
                    }
                },
                else => {
                    log.warn("inference", "Metal LayerOp program contains unsupported op kind", .{
                        .layer = layer_idx,
                        .op_index = op_idx,
                        .kind = kind,
                        .op = @tagName(op),
                    });
                    return error.NotImplemented;
                },
            }
        }

        if (!isCoreProgramBuffer(finalOutputBuffer(program))) {
            log.warn("inference", "Metal LayerOp program final buffer is unsupported", .{
                .layer = layer_idx,
                .kind = kind,
            });
            return error.NotImplemented;
        }
    }

    fn getBuffer(
        buffer_id: layer_ops.BufferId,
        residual: mlx_graph.ArrayHandle,
        norm_out: mlx_graph.ArrayHandle,
        branch_out: mlx_graph.ArrayHandle,
    ) !mlx_graph.ArrayHandle {
        return switch (buffer_id) {
            .residual => residual,
            .norm_out => norm_out,
            .branch_out => branch_out,
            else => error.NotImplemented,
        };
    }

    fn setBuffer(
        buffer_id: layer_ops.BufferId,
        residual: *mlx_graph.ArrayHandle,
        norm_out: *mlx_graph.ArrayHandle,
        branch_out: *mlx_graph.ArrayHandle,
        value: mlx_graph.ArrayHandle,
    ) !void {
        switch (buffer_id) {
            .residual => residual.* = value,
            .norm_out => norm_out.* = value,
            .branch_out => branch_out.* = value,
            else => return error.NotImplemented,
        }
    }

    fn nextNormWeight(lw: anytype, norm_index: *usize) !mlx_graph.ArrayHandle {
        const idx = norm_index.*;
        norm_index.* = idx + 1;
        return switch (idx) {
            0 => lw.getLn1(),
            1 => lw.getLn2(),
            // Gemma-style 4-norm blocks use pre/post-FFN norms.
            2 => lw.getPreFfnNorm() orelse lw.getPostFfnNorm() orelse error.InvalidState,
            3 => lw.getPostFfnNorm() orelse error.InvalidState,
            else => error.InvalidState,
        };
    }

    fn residualScale(
        scale: layer_ops.ResidualScale,
        weight_handles: anytype,
    ) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => weight_handles.residual_multiplier,
            .literal => |value| value,
        };
    }

    fn runMixerKernel(
        input: mlx_graph.ArrayHandle,
        lw: anytype,
        layer_idx: usize,
        config: anytype,
        weight_handles: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const norm_eps = config.norm_eps;
        const head_count: usize = @intCast(config.n_heads);
        const kv_head_count: usize = @intCast(config.n_kv_groups);
        const head_dim: usize = @intCast(config.head_dim);
        const attention_storage = lw.attentionStorageKind();
        const shortconv_storage = lw.shortconvStorageKind();

        return switch (lw.kind) {
            .attention_mlp => blk: {
                if (lw.isMLA()) {
                    const mla_cfg = lw.mla_config orelse return error.MissingField;
                    const mla_attention = mla_kernel.MLAttention{
                        .n_heads = head_count,
                        .rope_theta = config.rope_theta,
                        .norm_eps = norm_eps,
                        .q_lora_rank = mla_cfg.q_lora_rank,
                        .kv_lora_rank = mla_cfg.kv_lora_rank,
                        .qk_head_dim = mla_cfg.qk_head_dim,
                        .qk_rope_head_dim = mla_cfg.qk_rope_head_dim,
                        .qk_nope_head_dim = mla_cfg.qk_nope_head_dim,
                        .v_head_dim = mla_cfg.v_head_dim,
                        .q_a_proj = lw.mla_q_a_proj,
                        .q_b_proj = lw.mla_q_b_proj,
                        .kv_a_proj = lw.mla_kv_a_proj,
                        .kv_b_proj = lw.mla_kv_b_proj,
                        .q_a_proj_bf16 = lw.mla_q_a_proj_bf16,
                        .q_b_proj_bf16 = lw.mla_q_b_proj_bf16,
                        .kv_a_proj_bf16 = lw.mla_kv_a_proj_bf16,
                        .kv_b_proj_bf16 = lw.mla_kv_b_proj_bf16,
                        .q_a_norm = lw.mla_q_a_norm,
                        .kv_a_norm = lw.mla_kv_a_norm,
                        .o_proj = lw.o_proj,
                        .o_proj_bf16 = lw.o_proj_bf16,
                    };
                    var mla_cache = mla_kernel.AttnCache{
                        .cache = cache,
                        .layer_idx = layer_idx,
                        .pos_offset = pos_offset,
                    };
                    var mla_scratch = mla_kernel.AttnTemp{
                        .runtime_rope_cos_handle = runtime_rope_cos_handle,
                        .runtime_rope_sin_handle = runtime_rope_sin_handle,
                        .runtime_rope_dim = runtime_rope_dim,
                    };
                    var mla_matmul_scratch = mla_kernel.MatmulScratch{};
                    var mla_out: mlx_graph.ArrayHandle = undefined;
                    try mla_attention.forward(
                        input,
                        &mla_out,
                        &mla_cache,
                        &mla_scratch,
                        &mla_matmul_scratch,
                        cache != null,
                    );
                    break :blk mla_out;
                }

                if (attention_storage == .invalid) return error.InvalidTensorType;
                if (attention_storage == .missing) return error.MissingField;
                const attention = attention_kernel.MultiHeadAttention{
                    .n_heads = head_count,
                    .n_kv_heads = kv_head_count,
                    .head_dim = head_dim,
                    .rope_theta = config.rope_theta,
                    .norm_eps = norm_eps,
                    .query_pre_attn_scalar = config.query_pre_attn_scalar,
                    .attention_multiplier = weight_handles.attention_multiplier,
                    .q_proj = lw.q_proj,
                    .k_proj = lw.k_proj,
                    .v_proj = lw.v_proj,
                    .o_proj = lw.o_proj,
                    .q_proj_bf16 = lw.q_proj_bf16,
                    .k_proj_bf16 = lw.k_proj_bf16,
                    .v_proj_bf16 = lw.v_proj_bf16,
                    .o_proj_bf16 = lw.o_proj_bf16,
                    .q_norm = lw.q_norm,
                    .k_norm = lw.k_norm,
                    .q_bias = lw.q_bias,
                    .k_bias = lw.k_bias,
                    .v_bias = lw.v_bias,
                    .o_bias = lw.o_bias,
                    .attn_sinks = lw.attn_sinks,
                };
                var attn_cache = attention_kernel.AttnCache{
                    .cache = cache,
                    .layer_idx = layer_idx,
                    .pos_offset = pos_offset,
                };
                var attn_scratch = attention_kernel.AttnTemp{
                    .runtime_rope_cos_handle = runtime_rope_cos_handle,
                    .runtime_rope_sin_handle = runtime_rope_sin_handle,
                    .runtime_rope_dim = runtime_rope_dim,
                };
                var attn_matmul_scratch = attention_kernel.MatmulScratch{};
                var attn_out: mlx_graph.ArrayHandle = undefined;
                try attention.forward(
                    input,
                    &attn_out,
                    &attn_cache,
                    &attn_scratch,
                    &attn_matmul_scratch,
                    cache != null,
                );
                break :blk attn_out;
            },
            .shortconv => blk: {
                if (shortconv_storage == .invalid) return error.InvalidTensorType;
                if (shortconv_storage == .missing) return error.MissingField;
                const conv_weight = lw.shortconv_conv_weight orelse return error.MissingField;
                const shortconv = shortconv_kernel.ShortConvKernel{
                    .in_proj = if (lw.shortconv_in_proj) |w| w else null,
                    .out_proj = if (lw.shortconv_out_proj) |w| w else null,
                    .in_proj_bf16 = if (lw.shortconv_in_proj_bf16) |h| h else null,
                    .out_proj_bf16 = if (lw.shortconv_out_proj_bf16) |h| h else null,
                    .conv_weight = conv_weight,
                    .conv_bias = if (lw.shortconv_conv_bias) |b| b else null,
                    .d_conv = lw.shortconv_d_conv,
                    .conv_dim = lw.shortconv_conv_dim,
                };
                var shortconv_state = shortconv_kernel.ShortConvState{
                    .cache = shortconv_cache,
                    .layer_idx = layer_idx,
                };
                var shortconv_scratch = shortconv_kernel.ShortConvScratch{};
                var shortconv_matmul_scratch = shortconv_kernel.MatmulScratch{};
                var shortconv_out: mlx_graph.ArrayHandle = undefined;
                try shortconv.forward(
                    input,
                    &shortconv_out,
                    &shortconv_state,
                    &shortconv_scratch,
                    &shortconv_matmul_scratch,
                );
                break :blk shortconv_out;
            },
            .mamba => blk: {
                const conv_weight = lw.mamba_conv_weight orelse return error.MissingField;
                const a_log = lw.mamba_a_log orelse return error.MissingField;
                const d_skip = lw.mamba_d_skip orelse return error.MissingField;
                const mamba = mamba_kernel.MambaKernel{
                    .d_state = lw.mamba_d_state,
                    .d_conv = lw.mamba_d_conv,
                    .n_heads = lw.mamba_n_heads,
                    .d_head = lw.mamba_d_head,
                    .n_groups = lw.mamba_n_groups,
                    .use_gelu = weight_handles.use_gelu,
                    .residual_multiplier = weight_handles.residual_multiplier,
                    .norm_eps = norm_eps,
                    .gate_up_layout = @intFromEnum(lw.mamba_gate_up_layout),
                    .ln1_weight = lw.getLn1(),
                    .in_proj = lw.mamba_in_proj,
                    .in_proj_bf16 = lw.mamba_in_proj_bf16,
                    .conv_weight = conv_weight,
                    .conv_bias = lw.mamba_conv_bias,
                    .a_log = a_log,
                    .d_skip = d_skip,
                    .dt_bias = lw.mamba_dt_bias,
                    .norm_weight = lw.mamba_norm_weight,
                    .out_proj = lw.mamba_out_proj,
                    .out_proj_bf16 = lw.mamba_out_proj_bf16,
                    .ln2_weight = lw.getLn2(),
                    .gate_up = lw.mamba_gate_up,
                    .gate_up_bf16 = lw.mamba_gate_up_bf16,
                    .down_proj = lw.mamba_down_proj,
                    .down_proj_bf16 = lw.mamba_down_proj_bf16,
                };
                var m_state = mamba_kernel.MambaState{
                    .cache = mamba_cache,
                    .layer_idx = layer_idx,
                };
                var m_scratch = mamba_kernel.MambaScratch{};
                var m_matmul = mamba_kernel.MatmulScratch{};
                var m_out: mlx_graph.ArrayHandle = undefined;
                try mamba.forward(
                    input,
                    &m_out,
                    &m_state,
                    &m_scratch,
                    &m_matmul,
                );
                break :blk m_out;
            },
        };
    }

    fn runFfnKernel(
        input: mlx_graph.ArrayHandle,
        lw: anytype,
        weight_handles: anytype,
    ) !mlx_graph.ArrayHandle {
        const ffn_storage = lw.ffnStorageKind();
        return switch (ffn_storage) {
            .moe => blk: {
                const moe = lw.moe orelse return error.MissingField;
                const ffn_moe = moe_kernel.MoEFFN{ .weights = moe };
                var moe_scratch = moe_kernel.MoEScratch{};
                var moe_matmul_scratch = moe_kernel.MatmulScratch{};
                var moe_out: mlx_graph.ArrayHandle = undefined;
                try ffn_moe.forward(
                    input,
                    &moe_out,
                    &moe_scratch,
                    &moe_matmul_scratch,
                );
                break :blk moe_out;
            },
            .quantized, .dense => blk: {
                const swiglu = ffn_kernel.SwiGLU{
                    .use_gelu = weight_handles.use_gelu,
                    .w1 = lw.w1,
                    .w2 = lw.w2,
                    .w3 = lw.w3,
                    .w1_bf16 = lw.w1_bf16,
                    .w2_bf16 = lw.w2_bf16,
                    .w3_bf16 = lw.w3_bf16,
                };
                var ffn_scratch = ffn_kernel.FfnScratch{};
                var ffn_matmul_scratch = ffn_kernel.MatmulScratch{};
                var ffn_result: mlx_graph.ArrayHandle = undefined;
                try swiglu.forward(
                    input,
                    &ffn_result,
                    &ffn_scratch,
                    &ffn_matmul_scratch,
                );
                break :blk ffn_result;
            },
            .missing => error.MissingField,
            .invalid => error.InvalidTensorType,
        };
    }

    fn forwardWithProgram(
        hidden: mlx_graph.ArrayHandle,
        program: []const layer_ops.LayerOp,
        lw: anytype,
        layer_idx: usize,
        config: anytype,
        weight_handles: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        var residual = hidden;
        var norm_out: mlx_graph.ArrayHandle = hidden;
        var branch_out: mlx_graph.ArrayHandle = hidden;
        var norm_index: usize = 0;

        for (program) |op| {
            switch (op) {
                .kernel => |kernel_op| {
                    const input = try getBuffer(kernel_op.in, residual, norm_out, branch_out);
                    var output: mlx_graph.ArrayHandle = undefined;
                    switch (kernel_op.debug_type) {
                        .norm => {
                            const norm = norm_kernel.RMSNorm{
                                .weight = try nextNormWeight(lw, &norm_index),
                                .eps = config.norm_eps,
                            };
                            norm.forward(input, &output);
                        },
                        .multihead_attention, .shortconv => {
                            output = try runMixerKernel(
                                input,
                                lw,
                                layer_idx,
                                config,
                                weight_handles,
                                cache,
                                shortconv_cache,
                                mamba_cache,
                                pos_offset,
                                runtime_rope_cos_handle,
                                runtime_rope_sin_handle,
                                runtime_rope_dim,
                            );
                        },
                        .mlp, .moe => {
                            output = try runFfnKernel(input, lw, weight_handles);
                        },
                        .mamba_mixer => {
                            output = try runMixerKernel(
                                input,
                                lw,
                                layer_idx,
                                config,
                                weight_handles,
                                cache,
                                shortconv_cache,
                                mamba_cache,
                                pos_offset,
                                runtime_rope_cos_handle,
                                runtime_rope_sin_handle,
                                runtime_rope_dim,
                            );
                        },
                        else => return error.NotImplemented,
                    }
                    try setBuffer(kernel_op.out, &residual, &norm_out, &branch_out, output);
                },
                .add => |add_op| {
                    const branch = try getBuffer(add_op.branch, residual, norm_out, branch_out);
                    const scale = residualScale(add_op.scale, weight_handles);
                    const scaled_branch = if (scale == 1.0)
                        branch
                    else
                        mlx_graph.mlx_lazy_multiply_scalar(branch, scale);
                    residual = mlx_graph.mlx_lazy_add(residual, scaled_branch);
                },
                else => return error.NotImplemented,
            }
        }
        return getBuffer(finalOutputBuffer(program), residual, norm_out, branch_out);
    }

    pub fn forward(
        hidden: mlx_graph.ArrayHandle,
        layer_weights: anytype,
        layer_idx: usize,
        config: anytype,
        weight_handles: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const lw = layer_weights.*;

        if (lw.program) |program| {
            try validateLayerProgram(program, layer_idx, @intFromEnum(lw.kind));
            return forwardWithProgram(
                hidden,
                program,
                lw,
                layer_idx,
                config,
                weight_handles,
                cache,
                shortconv_cache,
                mamba_cache,
                pos_offset,
                runtime_rope_cos_handle,
                runtime_rope_sin_handle,
                runtime_rope_dim,
            );
        }
        log.warn("inference", "Metal block missing LayerOp program", .{
            .layer = layer_idx,
            .kind = @intFromEnum(lw.kind),
        });
        return error.UnsupportedModel;
    }

    pub fn projectLogits(
        hidden: mlx_graph.ArrayHandle,
        weight_handles: anytype,
        norm_eps: f32,
    ) mlx_graph.ArrayHandle {
        const final_normed = projectHidden(hidden, weight_handles, norm_eps);
        const logits = if (weight_handles.lm_head_quantized) |quantized_lm_head| blk: {
            break :blk mlx_graph.mlx_lazy_quantized_matmul(
                final_normed,
                quantized_lm_head.weights,
                quantized_lm_head.scales,
                quantized_lm_head.biases,
                quantized_lm_head.group_size,
                quantized_lm_head.bits,
                true,
            );
        } else blk: {
            if (weight_handles.lm_head_needs_transpose) {
                const transpose_axes = [_]usize{ 1, 0 };
                const lm_head_t = mlx_graph.mlx_lazy_transpose(weight_handles.lm_head.?, &transpose_axes, 2);
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, lm_head_t);
            } else {
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, weight_handles.lm_head.?);
            }
        };
        return if (weight_handles.logits_scaling != 1.0)
            mlx_graph.mlx_lazy_multiply_scalar(logits, 1.0 / weight_handles.logits_scaling)
        else
            logits;
    }

    pub fn projectHidden(
        hidden: mlx_graph.ArrayHandle,
        weight_handles: anytype,
        norm_eps: f32,
    ) mlx_graph.ArrayHandle {
        const final_norm = norm_kernel.RMSNorm{
            .weight = weight_handles.ln_final,
            .eps = norm_eps,
        };
        var final_normed: mlx_graph.ArrayHandle = undefined;
        final_norm.forward(hidden, &final_normed);
        return final_normed;
    }
};

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
    try std.testing.expectEqual(layer_ops.BufferId.residual, TransformerBlock.finalOutputBuffer(&program));
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
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, TransformerBlock.finalOutputBuffer(&program));
}

test "validateLayerProgram accepts kernel-add programs" {
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
    try TransformerBlock.validateLayerProgram(&program, 0, 0);
}

test "validateLayerProgram rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    try std.testing.expectError(error.NotImplemented, TransformerBlock.validateLayerProgram(&program, 0, 0));
}
