//! Per-layer branch kernel for CPU backend.
//!
//! Implements a gated embedding branch applied after each transformer layer.
//! The branch projects source embeddings through a narrow bottleneck, gates with
//! the current hidden state, and adds the result back to the residual stream.
//!
//! Used by architectures that define per-layer embedding inputs (e.g. Gemma4 PLE).
//! This kernel is generic — it is parameterized by weights and scalars from the
//! execution plan, not by model family.

const std = @import("std");
const compute = @import("compute_pkg");
const cpu_linalg = compute.cpu.linalg;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_activation = compute.cpu.activation;
const cpu_common = compute.cpu.common;
const tensor_mod = @import("tensor_pkg");
const dtype_mod = @import("dtype_pkg");
const norm_kernel = @import("norm.zig");
const trace = @import("xray_pkg").trace;

const Tensor = tensor_mod.Tensor;
const RMSNorm = norm_kernel.RMSNorm;
const MatmulFn = cpu_linalg.MatmulFn;

/// Linear projection (y = x @ W), constructed from a weight tensor.
/// Project input through a weight matrix: output = input @ W.
fn linearForward(
    input: *const Tensor,
    weight: *const Tensor,
    output: *Tensor,
    in_features: usize,
    out_features: usize,
    matmul_fn: MatmulFn,
    scratch: *cpu_linalg.MatmulScratch,
) void {
    const row_count: usize = if (input.n_dims == 3) @intCast(input.shape[0] * input.shape[1]) else @intCast(input.shape[0]);
    const input_view = Tensor.view2D(input.data(), row_count, in_features);
    var output_view = Tensor.view2DSlice(output.asSlice(f32), row_count, out_features);
    matmul_fn(&input_view, weight, &output_view, scratch);
}

/// Runtime state for a single per-layer-branch invocation.
/// Constructed per-layer from weight handles and params.
pub const PerLayerBranchConfig = struct {
    hidden_size: usize,
    hidden_size_per_layer_input: usize,
    use_gelu: bool,
    per_layer_input_scale: f32,
    per_layer_embed_scale: f32,
    per_layer_model_projection_scale: f32,
    norm_eps: f32,
    norm_weight_offset: f32,
    layer_scalar: f32,
    layer_idx: usize,
};

/// Scratch buffers for intermediate computation.
pub const PerLayerBranchScratch = struct {
    projection: []f32,
    per_layer_input: []f32,
    gated: []f32,
    branch: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        row_count: usize,
        hidden_size: usize,
        hidden_size_per_layer_input: usize,
    ) !PerLayerBranchScratch {
        return .{
            .projection = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .per_layer_input = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .gated = try allocator.alloc(f32, row_count * hidden_size_per_layer_input),
            .branch = try allocator.alloc(f32, row_count * hidden_size),
        };
    }

    pub fn deinit(self: *PerLayerBranchScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.projection);
        allocator.free(self.per_layer_input);
        allocator.free(self.gated);
        allocator.free(self.branch);
        self.* = undefined;
    }
};

/// Canonical kernel-call contract for backend parity checks.
pub const ForwardParams = struct {
    hidden_states: *Tensor,
    source_embeddings: *const Tensor,
    token_ids: []const u32,
    per_layer_embedding: *const Tensor,
    model_projection_weight: *const Tensor,
    projection_norm_weight: *const Tensor,
    input_gate_weight: *const Tensor,
    projection_weight: *const Tensor,
    post_norm_weight: *const Tensor,
    config: PerLayerBranchConfig,
    scratch: *PerLayerBranchScratch,
    matmul_scratch: *cpu_linalg.MatmulScratch,
    /// Cached matmul dispatch function (resolved once from weight dtype at load time).
    matmul_fn: MatmulFn,
};

/// Gather per-layer embedding values for given token IDs at a specific layer.
/// Supports f32, bf16, and f16 embedding dtypes.
pub fn gatherPerLayerEmbedding(
    embedding: *const Tensor,
    token_ids: []const u32,
    layer_idx: usize,
    hidden_size_per_layer_input: usize,
    embed_scale: f32,
    out: []f32,
) !void {
    const total_width: usize = @intCast(embedding.shape[1]);
    const layer_offset = layer_idx * hidden_size_per_layer_input;
    switch (embedding.dtype) {
        .f32 => {
            const values = embedding.asSlice(f32);
            for (token_ids, 0..) |token_id, row_idx| {
                const src_base = @as(usize, token_id) * total_width + layer_offset;
                const dst_base = row_idx * hidden_size_per_layer_input;
                for (0..hidden_size_per_layer_input) |col| {
                    out[dst_base + col] += values[src_base + col] * embed_scale;
                }
            }
        },
        .bf16 => {
            const values = embedding.asSlice(u16);
            for (token_ids, 0..) |token_id, row_idx| {
                const src_base = @as(usize, token_id) * total_width + layer_offset;
                const dst_base = row_idx * hidden_size_per_layer_input;
                for (0..hidden_size_per_layer_input) |col| {
                    out[dst_base + col] += dtype_mod.bf16ToF32(values[src_base + col]) * embed_scale;
                }
            }
        },
        .f16 => {
            const values = embedding.asSlice(u16);
            for (token_ids, 0..) |token_id, row_idx| {
                const src_base = @as(usize, token_id) * total_width + layer_offset;
                const dst_base = row_idx * hidden_size_per_layer_input;
                for (0..hidden_size_per_layer_input) |col| {
                    out[dst_base + col] += dtype_mod.fp16ToF32(values[src_base + col]) * embed_scale;
                }
            }
        },
        else => return error.UnsupportedDType,
    }
}

/// Execute the per-layer branch computation.
///
/// Steps:
///  1. Project source_embeddings → narrow dim via model_projection
///  2. Scale by model_projection_scale
///  3. RMSNorm via projection_norm
///  4. Gather per-layer embeddings, scale, add to normalized projection
///  5. Scale by per_layer_input_scale
///  6. Gate: linear(hidden_states) → activation → element-wise multiply
///  7. Project gated output back to full dim
///  8. RMSNorm via post_norm
///  9. Residual add: hidden_states += branch
/// 10. Scale full result by layer_scalar
pub fn forward(params: *const ForwardParams) !void {
    const cfg = params.config;
    const scr = params.scratch;
    const row_count: usize = @intCast(params.hidden_states.shape[1]);

    var projection_tensor = Tensor.view3DSlice(
        scr.projection,
        row_count,
        cfg.hidden_size_per_layer_input,
    );
    var per_layer_input_tensor = Tensor.view3DSlice(
        scr.per_layer_input,
        row_count,
        cfg.hidden_size_per_layer_input,
    );
    var gated_tensor = Tensor.view3DSlice(
        scr.gated,
        row_count,
        cfg.hidden_size_per_layer_input,
    );
    var branch_tensor = Tensor.view3DSlice(
        scr.branch,
        row_count,
        cfg.hidden_size,
    );

    // 1. Project source embeddings to narrow dimension
    linearForward(
        params.source_embeddings,
        params.model_projection_weight,
        &projection_tensor,
        cfg.hidden_size,
        cfg.hidden_size_per_layer_input,
        params.matmul_fn,
        params.matmul_scratch,
    );

    // 2. Scale projection
    cpu_rowwise.scaleInPlace(scr.projection, cfg.per_layer_model_projection_scale);

    // 3. RMSNorm on projection
    const projection_norm = RMSNorm{
        .weight = params.projection_norm_weight,
        .dim = cfg.hidden_size_per_layer_input,
        .eps = cfg.norm_eps,
        .weight_offset = cfg.norm_weight_offset,
        .trace_point = .layer_ffn_norm,
        .layer_idx = trace.TraceEmission.NO_LAYER,
    };
    projection_norm.forward(&projection_tensor, &per_layer_input_tensor);

    // 4. Gather per-layer embeddings and add (scaled)
    try gatherPerLayerEmbedding(
        params.per_layer_embedding,
        params.token_ids,
        cfg.layer_idx,
        cfg.hidden_size_per_layer_input,
        cfg.per_layer_embed_scale,
        scr.per_layer_input,
    );

    // 5. Scale per-layer input
    cpu_rowwise.scaleInPlace(scr.per_layer_input, cfg.per_layer_input_scale);

    // 6. Gate: project hidden → narrow, apply activation, element-wise multiply
    linearForward(
        params.hidden_states,
        params.input_gate_weight,
        &gated_tensor,
        cfg.hidden_size,
        cfg.hidden_size_per_layer_input,
        params.matmul_fn,
        params.matmul_scratch,
    );
    for (scr.gated) |*value| {
        value.* = if (cfg.use_gelu)
            cpu_activation.geluApprox(value.*)
        else
            cpu_activation.silu(value.*);
    }
    for (scr.gated, 0..) |*value, idx| {
        value.* *= scr.per_layer_input[idx];
    }

    // 7. Project back to full dimension
    linearForward(
        &gated_tensor,
        params.projection_weight,
        &branch_tensor,
        cfg.hidden_size_per_layer_input,
        cfg.hidden_size,
        params.matmul_fn,
        params.matmul_scratch,
    );

    // 8. Post-norm on branch
    const post_norm = RMSNorm{
        .weight = params.post_norm_weight,
        .dim = cfg.hidden_size,
        .eps = cfg.norm_eps,
        .weight_offset = cfg.norm_weight_offset,
        .trace_point = .block_out,
        .layer_idx = @intCast(cfg.layer_idx),
    };
    post_norm.forward(&branch_tensor, &branch_tensor);

    // 9. Residual add
    const hidden_values = params.hidden_states.asSliceMut(f32);
    for (hidden_values, 0..) |*hidden_value, idx| {
        hidden_value.* += scr.branch[idx];
    }

    // 10. Scale full result
    cpu_rowwise.scaleInPlace(hidden_values, cfg.layer_scalar);
}

// ── Tests ──

test "gatherPerLayerEmbedding accumulates scaled embeddings for f32" {
    // 2 tokens, 2 layers * 2 hpl = 4 cols per token
    var embed_data = [_]f32{
        // token 0: [layer0: 1,2] [layer1: 3,4]
        1.0, 2.0, 3.0, 4.0,
        // token 1: [layer0: 5,6] [layer1: 7,8]
        5.0, 6.0, 7.0, 8.0,
    };
    var embed_tensor = Tensor.view2DSlice(embed_data[0..], 2, 4);
    embed_tensor.dtype = .f32;

    // Gather layer 0 for tokens [0, 1], hpl=2, scale=2.0
    var out = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const token_ids = [_]u32{ 0, 1 };
    try gatherPerLayerEmbedding(&embed_tensor, token_ids[0..], 0, 2, 2.0, out[0..]);

    // token 0 layer 0: [1*2, 2*2] = [2, 4]
    // token 1 layer 0: [5*2, 6*2] = [10, 12]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), out[3], 1e-6);
}

test "gatherPerLayerEmbedding rejects unsupported dtype" {
    var embed_data = [_]f32{ 1.0, 2.0 };
    var embed_tensor = Tensor.view2DSlice(embed_data[0..], 1, 2);
    embed_tensor.dtype = .i32;

    var out = [_]f32{ 0.0, 0.0 };
    const token_ids = [_]u32{0};
    const result = gatherPerLayerEmbedding(&embed_tensor, token_ids[0..], 0, 2, 1.0, out[0..]);
    try std.testing.expectError(error.UnsupportedDType, result);
}
