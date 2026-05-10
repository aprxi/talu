//! Query-gate projection helpers for the CUDA inference backend.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const ProjectionPath = engine_types.ProjectionPath;
const Nvfp4RouteKind = engine_types.Nvfp4RouteKind;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

const models = @import("models_pkg");
const layer_ops = models.layer_ops;

// --- Utility functions from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

pub fn compactQueryGateProjection(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
    n_heads_u32: u32,
    head_dim_u32: u32,
    q_projection_stage: *const compute.cuda.Buffer,
    q_values_stage: *compute.cuda.Buffer,
) !void {
    const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
    const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
    _ = projection_elements;
    _ = query_elements;
    try compute.cuda.gated_attention_compact_q.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.gated_attention_compact_q_function orelse return error.CudaKernelUnavailable,
        q_projection_stage,
        q_values_stage,
        @intCast(seq_len),
        @intCast(q_dim),
        @intCast(q_projection_dim),
        n_heads_u32,
        head_dim_u32,
    );
}

pub fn applyQueryGateToContextInPlace(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
    n_heads_u32: u32,
    head_dim_u32: u32,
) !void {
    const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
    const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
    const projection_bytes = std.math.mul(usize, projection_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    const context_bytes = std.math.mul(usize, query_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    var context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_bytes);
    var projection_stage = try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, projection_bytes);
    try compute.cuda.gated_attention_output_gate.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.gated_attention_output_gate_function orelse return error.CudaKernelUnavailable,
        &context_stage,
        &projection_stage,
        @intCast(seq_len),
        @intCast(q_dim),
        @intCast(q_projection_dim),
        n_heads_u32,
        head_dim_u32,
    );
}
