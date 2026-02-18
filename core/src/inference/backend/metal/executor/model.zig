//! Metal backend model executor surface.
//!
//! Keeps model-level orchestration discoverable under executor/model while
//! preserving the existing fused MLX execution path.

const std = @import("std");
const builtin = @import("builtin");
const compute = @import("../../../../compute/root.zig");
const block_executor = @import("block.zig");

const ArrayHandle = compute.metal.graph.ArrayHandle;
const mlx_graph = compute.metal.graph;

pub const Cache = mlx_graph.Cache;
pub const ShortConvCache = mlx_graph.ShortConvCache;

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
        config: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        pos_offset: usize,
        use_compiled: bool,
    ) !ArrayHandle {
        return forwardWithEmbeddingOverride(
            allocator,
            weight_handles,
            input_ids,
            config,
            cache,
            shortconv_cache,
            pos_offset,
            use_compiled,
            null,
            null,
            null,
        );
    }

    pub fn forwardWithEmbeddingOverride(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        input_ids: []const u32,
        config: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        pos_offset: usize,
        use_compiled: bool,
        embedding_override: ?[]const f32,
        deepstack: ?DeepstackAdditions,
        runtime_rope: ?RuntimeRoPEOverride,
    ) !ArrayHandle {
        if (builtin.os.tag != .macos) {
            return error.MLXNotAvailable;
        }

        var trace = std.posix.getenv("TALU_TRACE_METAL_UNSAFE") != null;
        if (trace and cache != null) trace = false;
        const phase: []const u8 = if (input_ids.len == 1) "decode" else "prefill";

        const norm_eps = config.norm_eps;
        const layer_count: usize = @intCast(config.n_layers);
        const sequence_len = input_ids.len;
        const use_compiled_effective = use_compiled and !trace;
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

        var used_fusion: bool = false;
        for (0..layer_count) |layer_idx| {
            if (use_compiled_effective and weight_handles.compiled_layers != null and sequence_len == 1) {
                const compiled = weight_handles.compiled_layers.?;
                used_fusion = true;

                if (cache) |c| {
                    if (c.use_bfloat16) {
                        hidden = compiled[layer_idx].forward(hidden, c.handle, layer_idx, pos_offset);
                    } else {
                        used_fusion = false;
                    }
                } else {
                    hidden = compiled[layer_idx].forward(hidden, null, layer_idx, pos_offset);
                }

                if (used_fusion) {
                    if (trace) {
                        try traceLastHiddenVector(allocator, "metal", phase, layer_idx, hidden, sequence_len, @intCast(weight_handles.d_model));
                    }
                    continue;
                }
            }

            hidden = try block_executor.TransformerBlock.forward(
                hidden,
                &weight_handles.layers[layer_idx],
                layer_idx,
                config,
                weight_handles,
                cache,
                shortconv_cache,
                pos_offset,
                runtime_rope_cos_handle,
                runtime_rope_sin_handle,
                runtime_rope_dim,
            );
            if (layer_idx < deepstack_layer_additions.len) {
                hidden = mlx_graph.mlx_lazy_add(hidden, deepstack_layer_additions[layer_idx]);
            }

            if (trace) {
                try traceLastHiddenVector(allocator, "metal", phase, layer_idx, hidden, sequence_len, @intCast(weight_handles.d_model));
            }
        }

        return block_executor.TransformerBlock.projectLogits(hidden, weight_handles, norm_eps);
    }

    pub fn forwardFromGPUToken(
        allocator: std.mem.Allocator,
        weight_handles: anytype,
        token_handle: ArrayHandle,
        config: anytype,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        pos_offset: usize,
    ) !ArrayHandle {
        if (builtin.os.tag != .macos) {
            return error.MLXNotAvailable;
        }

        var trace = std.posix.getenv("TALU_TRACE_METAL_UNSAFE") != null;
        if (trace and cache != null) trace = false;

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
                cache,
                shortconv_cache,
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
