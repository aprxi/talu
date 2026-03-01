//! Metal backend model executor surface.
//!
//! Keeps model-level orchestration discoverable under executor/model while
//! preserving the existing fused MLX execution path.

const std = @import("std");
const builtin = @import("builtin");
const compute = @import("../../../../compute/root.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const runtime_graph = @import("../runtime_graph.zig");
const block_executor = @import("block.zig");

const ArrayHandle = compute.metal.graph.ArrayHandle;
const mlx_graph = compute.metal.graph;

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;

pub const DeepstackAdditions = struct {
    positions: []const usize,
    layer_features: []const []const f32,
};

pub const RuntimeRoPEOverride = struct {
    cos: []const f32,
    sin: []const f32,
    dim: usize,
};

pub fn gatherTokenEmbeddingsLazy(weight_handles: anytype, input_ids: []const u32) !ArrayHandle {
    const sequence_len = input_ids.len;

    var hidden: ArrayHandle = undefined; // Safe: both branches assign before use
    if (weight_handles.embed_tokens_quantized) |quantized_weight| {
        const qw_rows = mlx_graph.mlx_lazy_embedding(quantized_weight.weights, input_ids.ptr, sequence_len);
        const scales_rows = mlx_graph.mlx_lazy_embedding(quantized_weight.scales, input_ids.ptr, sequence_len);
        const biases_rows = mlx_graph.mlx_lazy_embedding(quantized_weight.biases, input_ids.ptr, sequence_len);
        hidden = mlx_graph.mlx_lazy_dequantize(
            qw_rows,
            scales_rows,
            biases_rows,
            quantized_weight.group_size,
            quantized_weight.bits,
        );
    } else {
        hidden = mlx_graph.mlx_lazy_embedding(weight_handles.embed_tokens.?, input_ids.ptr, sequence_len);
    }

    if (weight_handles.embedding_multiplier != 1.0) {
        hidden = mlx_graph.mlx_lazy_multiply_scalar(hidden, weight_handles.embedding_multiplier);
    } else if (weight_handles.use_sqrt_embedding_scale) {
        hidden = mlx_graph.mlx_scale_by_sqrt(hidden, @intCast(weight_handles.d_model));
    }
    return hidden;
}

fn buildDeepstackLayerAdditions(
    allocator: std.mem.Allocator,
    sequence_len: usize,
    hidden_dim: usize,
    positions: []const usize,
    layer_features: []const []const f32,
) ![]ArrayHandle {
    if (positions.len == 0 or layer_features.len == 0) return &.{};

    const handles = try allocator.alloc(ArrayHandle, layer_features.len);
    errdefer allocator.free(handles);

    for (layer_features, 0..) |features, layer_idx| {
        if ((features.len % hidden_dim) != 0) return error.InvalidShape;

        const dense_values = try allocator.alloc(f32, sequence_len * hidden_dim);
        defer allocator.free(dense_values);
        @memset(dense_values, 0);

        const available_rows = features.len / hidden_dim;
        const row_count = @min(positions.len, available_rows);
        for (0..row_count) |row_idx| {
            const token_pos = positions[row_idx];
            if (token_pos >= sequence_len) continue;
            const dst = dense_values[token_pos * hidden_dim ..][0..hidden_dim];
            const src = features[row_idx * hidden_dim ..][0..hidden_dim];
            for (dst, src) |*d, s| d.* += s;
        }

        const add_shape = [_]i64{ 1, @intCast(sequence_len), @intCast(hidden_dim) };
        handles[layer_idx] = mlx_graph.createArrayF32(dense_values, &add_shape);
    }

    return handles;
}

fn traceLastHiddenVector(
    allocator: std.mem.Allocator,
    comptime backend: []const u8,
    phase: []const u8,
    layer: ?usize,
    hidden: ArrayHandle,
    seq_len: usize,
    d_model: usize,
) !void {
    if (seq_len == 0) return;

    var starts: [3]c_int = .{ 0, @intCast(seq_len - 1), 0 };
    var ends: [3]c_int = .{ 1, @intCast(seq_len), @intCast(d_model) };
    const slice = mlx_graph.mlx_lazy_slice(hidden, &starts, &ends, 3);
    defer mlx_graph.freeArray(slice);

    var shape: [1]usize = .{d_model};
    const flat = mlx_graph.mlx_lazy_reshape(slice, &shape, 1);
    defer mlx_graph.freeArray(flat);

    mlx_graph.eval(&.{flat});

    const host_buf = try allocator.alloc(f32, d_model);
    defer allocator.free(host_buf);
    mlx_graph.copyToHost(flat, host_buf);

    var minv: f32 = host_buf[0];
    var maxv: f32 = host_buf[0];
    var sum: f64 = 0;
    var sumsq: f64 = 0;
    for (host_buf) |x| {
        if (x < minv) minv = x;
        if (x > maxv) maxv = x;
        sum += x;
        sumsq += @as(f64, x) * @as(f64, x);
    }
    const mean: f64 = sum / @as(f64, @floatFromInt(host_buf.len));
    const rms: f64 = @sqrt(sumsq / @as(f64, @floatFromInt(host_buf.len)));
    const first0 = host_buf[0];
    const first1 = if (host_buf.len > 1) host_buf[1] else 0;
    const first2 = if (host_buf.len > 2) host_buf[2] else 0;
    const first3 = if (host_buf.len > 3) host_buf[3] else 0;

    if (layer) |layer_idx| {
        std.debug.print("TRACE backend={s} phase={s} layer={} hidden_last min={d:.6} max={d:.6} mean={d:.6} rms={d:.6} first4=[{d:.6},{d:.6},{d:.6},{d:.6}]\n", .{
            backend, phase, layer_idx, minv, maxv, mean, rms, first0, first1, first2, first3,
        });
    } else {
        std.debug.print("TRACE backend={s} phase={s} layer=embed hidden_last min={d:.6} max={d:.6} mean={d:.6} rms={d:.6} first4=[{d:.6},{d:.6},{d:.6},{d:.6}]\n", .{
            backend, phase, minv, maxv, mean, rms, first0, first1, first2, first3,
        });
    }
}

pub const Model = struct {
    pub fn forward(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
    ) !ArrayHandle {
        return forwardWithEmbeddingOverride(
            allocator,
            weight_handles,
            input_ids,
            state_blocks,
            config,
            pos_offset,
            null,
            null,
            null,
        );
    }

    pub fn forwardWithEmbeddingOverride(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
        embedding_override: ?[]const f32,
        deepstack: ?DeepstackAdditions,
        runtime_rope: ?RuntimeRoPEOverride,
    ) !ArrayHandle {
        const hidden = try forwardHiddenCoreWithEmbeddingOverride(
            allocator,
            weight_handles,
            input_ids,
            state_blocks,
            config,
            pos_offset,
            embedding_override,
            deepstack,
            runtime_rope,
        );
        return block_executor.TransformerBlock.projectLogits(hidden, weight_handles, config.norm_eps);
    }

    pub fn forwardHidden(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
    ) !ArrayHandle {
        return forwardHiddenWithEmbeddingOverride(
            allocator,
            weight_handles,
            input_ids,
            state_blocks,
            config,
            pos_offset,
            null,
            null,
            null,
        );
    }

    pub fn forwardHiddenWithEmbeddingOverride(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
        embedding_override: ?[]const f32,
        deepstack: ?DeepstackAdditions,
        runtime_rope: ?RuntimeRoPEOverride,
    ) !ArrayHandle {
        const hidden = try forwardHiddenCoreWithEmbeddingOverride(
            allocator,
            weight_handles,
            input_ids,
            state_blocks,
            config,
            pos_offset,
            embedding_override,
            deepstack,
            runtime_rope,
        );
        return block_executor.TransformerBlock.projectHidden(hidden, weight_handles, config.norm_eps);
    }

    fn forwardHiddenCoreWithEmbeddingOverride(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
        embedding_override: ?[]const f32,
        deepstack: ?DeepstackAdditions,
        runtime_rope: ?RuntimeRoPEOverride,
    ) !ArrayHandle {
        if (builtin.os.tag != .macos) {
            return error.MLXNotAvailable;
        }

        const trace = std.posix.getenv("TALU_TRACE_METAL_UNSAFE") != null;
        const phase: []const u8 = if (input_ids.len == 1) "decode" else "prefill";

        const layer_count: usize = @intCast(config.n_layers);
        const sequence_len = input_ids.len;
        const hidden_dim = @as(usize, @intCast(weight_handles.d_model));

        var runtime_rope_cos_handle: ArrayHandle = null;
        var runtime_rope_sin_handle: ArrayHandle = null;
        var runtime_rope_dim: usize = 0;
        if (runtime_rope) |rr| {
            if (rr.dim == 0 or rr.cos.len != rr.sin.len or (rr.cos.len % rr.dim) != 0) return error.InvalidShape;
            const table_rows = rr.cos.len / rr.dim;
            const rope_shape = [_]i64{ @intCast(table_rows), @intCast(rr.dim) };
            runtime_rope_cos_handle = mlx_graph.createArrayF32(rr.cos, &rope_shape);
            runtime_rope_sin_handle = mlx_graph.createArrayF32(rr.sin, &rope_shape);
            runtime_rope_dim = rr.dim;
        }

        var deepstack_layer_additions: []ArrayHandle = &.{};
        defer if (deepstack_layer_additions.len > 0) allocator.free(deepstack_layer_additions);
        if (deepstack) |ctx| {
            deepstack_layer_additions = try buildDeepstackLayerAdditions(
                allocator,
                sequence_len,
                hidden_dim,
                ctx.positions,
                ctx.layer_features,
            );
        }

        var hidden: ArrayHandle = if (embedding_override) |hidden_values| blk: {
            if (hidden_values.len != sequence_len * weight_handles.d_model) return error.InvalidShape;
            const hidden_shape = [_]i64{ 1, @intCast(sequence_len), @intCast(weight_handles.d_model) };
            break :blk mlx_graph.createArrayF32(hidden_values, &hidden_shape);
        } else try gatherTokenEmbeddingsLazy(weight_handles, input_ids);

        if (trace) {
            try traceLastHiddenVector(allocator, "metal", phase, null, hidden, sequence_len, @intCast(weight_handles.d_model));
        }

        for (0..layer_count) |layer_idx| {
            hidden = try block_executor.TransformerBlock.forward(
                hidden,
                &weight_handles.layers[layer_idx],
                layer_idx,
                config,
                weight_handles,
                state_blocks,
                pos_offset,
                runtime_rope_cos_handle,
                runtime_rope_sin_handle,
                runtime_rope_dim,
            );
            // Deepstack: per-request feature addition between block-level adapter
            // dispatches. Operates outside the per-instruction adapter table — same
            // pattern as embedding lookup and final logit projection.
            if (layer_idx < deepstack_layer_additions.len) {
                hidden = mlx_graph.mlx_lazy_add(hidden, deepstack_layer_additions[layer_idx]);
            }

            if (trace) {
                try traceLastHiddenVector(allocator, "metal", phase, layer_idx, hidden, sequence_len, @intCast(weight_handles.d_model));
            }
        }

        return hidden;
    }

    pub fn forwardFromGPUToken(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        token_handle: ArrayHandle,
        state_blocks: []const runtime_contract.StateBlockHandle,
        config: anytype,
        pos_offset: usize,
    ) !ArrayHandle {
        if (builtin.os.tag != .macos) {
            return error.MLXNotAvailable;
        }

        const trace = std.posix.getenv("TALU_TRACE_METAL_UNSAFE") != null;

        const norm_eps = config.norm_eps;
        const layer_count: usize = @intCast(config.n_layers);

        var hidden: ArrayHandle = undefined; // Safe: both branches assign before use
        if (weight_handles.embed_tokens_quantized) |quantized_weight| {
            const qw_rows = mlx_graph.mlx_lazy_embedding_from_array(quantized_weight.weights, token_handle);
            const scales_rows = mlx_graph.mlx_lazy_embedding_from_array(quantized_weight.scales, token_handle);
            const biases_rows = mlx_graph.mlx_lazy_embedding_from_array(quantized_weight.biases, token_handle);
            hidden = mlx_graph.mlx_lazy_dequantize(qw_rows, scales_rows, biases_rows, quantized_weight.group_size, quantized_weight.bits);
        } else {
            hidden = mlx_graph.mlx_lazy_embedding_from_array(weight_handles.embed_tokens.?, token_handle);
        }

        for (0..layer_count) |layer_idx| {
            hidden = try block_executor.TransformerBlock.forward(
                hidden,
                &weight_handles.layers[layer_idx],
                layer_idx,
                config,
                weight_handles,
                state_blocks,
                pos_offset,
                null,
                null,
                0,
            );

            if (trace) {
                try traceLastHiddenVector(allocator, "metal", "decode", layer_idx, hidden, 1, @intCast(weight_handles.d_model));
            }
        }
        return block_executor.TransformerBlock.projectLogits(hidden, weight_handles, norm_eps);
    }
};

test "buildDeepstackLayerAdditions rejects misaligned feature rows" {
    const allocator = std.testing.allocator;
    const positions = [_]usize{0};
    const layer0 = [_]f32{ 1.0, 2.0, 3.0 };
    const layer_features = [_][]const f32{layer0[0..]};

    try std.testing.expectError(
        error.InvalidShape,
        buildDeepstackLayerAdditions(
            allocator,
            1,
            2,
            positions[0..],
            layer_features[0..],
        ),
    );
}

test "Model.forwardHidden exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(Model.forwardHidden)).@"fn";
    try std.testing.expectEqual(@as(usize, 6), fn_info.params.len);
}

test "Model.forwardHiddenWithEmbeddingOverride exposes stable callable signature" {
    const fn_info = @typeInfo(@TypeOf(Model.forwardHiddenWithEmbeddingOverride)).@"fn";
    try std.testing.expectEqual(@as(usize, 9), fn_info.params.len);
}

test "Model.forward matches forwardFromGPUToken for single-token zero-layer model" {
    if (builtin.os.tag != .macos) return;

    const allocator = std.testing.allocator;
    const d_model: usize = 4;
    const vocab_size: usize = 8;

    const embeddings_data = [_]f32{
        0.11,  -0.05, 0.23,  0.07,
        -0.18, 0.09,  0.04,  -0.12,
        0.31,  0.08,  -0.21, 0.16,
        0.27,  -0.14, 0.19,  0.05,
        -0.07, 0.22,  0.13,  -0.04,
        0.15,  -0.11, 0.06,  0.29,
        -0.03, 0.17,  -0.26, 0.10,
        0.24,  0.02,  -0.09, -0.19,
    };
    const embeddings_shape = [_]i64{ @intCast(vocab_size), @intCast(d_model) };
    const embeddings = mlx_graph.createArrayF32(embeddings_data[0..], &embeddings_shape);
    defer mlx_graph.freeArray(embeddings);

    const ln_final_data = [_]f32{ 1.0, 0.5, 1.5, 2.0 };
    const ln_final_shape = [_]i64{@intCast(d_model)};
    const ln_final = mlx_graph.createArrayF32(ln_final_data[0..], &ln_final_shape);
    defer mlx_graph.freeArray(ln_final);

    const lm_head_data = [_]f32{
        0.05,  -0.02, 0.11,  0.07,  -0.09, 0.03,  0.04,  -0.01,
        0.08,  0.12,  -0.05, 0.06,  0.02,  -0.04, 0.10,  0.09,
        -0.03, 0.01,  0.07,  -0.08, 0.13,  0.05,  -0.06, 0.02,
        0.14,  -0.10, 0.03,  0.12,  -0.07, 0.11,  0.00,  -0.05,
    };
    const lm_head_shape = [_]i64{ @intCast(d_model), @intCast(vocab_size) };
    const lm_head = mlx_graph.createArrayF32(lm_head_data[0..], &lm_head_shape);
    defer mlx_graph.freeArray(lm_head);

    const weights_mod = @import("weights.zig");
    var weight_handles = weights_mod.WeightHandles{
        .embed_tokens = embeddings,
        .embed_tokens_quantized = null,
        .layers = &.{},
        .ln_final = ln_final,
        .lm_head = lm_head,
        .lm_head_quantized = null,
        .lm_head_needs_transpose = false,
        .d_model = d_model,
        .is_quantized = false,
    };

    const cfg = struct {
        n_layers: usize = 0,
        norm_eps: f32 = 1e-5,
    }{};

    const input_ids = [_]u32{3};
    const logits_from_ids = try Model.forward(
        allocator,
        &weight_handles,
        input_ids[0..],
        &.{},
        cfg,
        0,
    );
    defer mlx_graph.freeArray(logits_from_ids);

    const token_shape = [_]usize{input_ids.len};
    const token_handle = mlx_graph.mlx_array_from_uint32(input_ids[0..].ptr, &token_shape, 1);
    defer mlx_graph.freeArray(token_handle);

    const logits_from_gpu_token = try Model.forwardFromGPUToken(
        allocator,
        &weight_handles,
        token_handle,
        &.{},
        cfg,
        0,
    );
    defer mlx_graph.freeArray(logits_from_gpu_token);

    mlx_graph.eval(&.{ logits_from_ids, logits_from_gpu_token });

    var shape_buf: [8]usize = undefined;
    const rank_a = mlx_graph.getShape(logits_from_ids, &shape_buf);
    var total_logits: usize = 1;
    for (shape_buf[0..rank_a]) |dim| total_logits *= dim;
    try std.testing.expectEqual(vocab_size, total_logits);

    var shape_buf_b: [8]usize = undefined;
    const rank_b = mlx_graph.getShape(logits_from_gpu_token, &shape_buf_b);
    var total_logits_b: usize = 1;
    for (shape_buf_b[0..rank_b]) |dim| total_logits_b *= dim;
    try std.testing.expectEqual(vocab_size, total_logits_b);

    const host_a = try allocator.alloc(f32, total_logits);
    defer allocator.free(host_a);
    const host_b = try allocator.alloc(f32, total_logits_b);
    defer allocator.free(host_b);

    mlx_graph.copyToHost(logits_from_ids, host_a);
    mlx_graph.copyToHost(logits_from_gpu_token, host_b);

    for (host_a, host_b) |a, b| {
        try std.testing.expectApproxEqAbs(a, b, 1e-5);
    }
}
