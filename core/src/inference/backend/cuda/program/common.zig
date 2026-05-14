//! Shared CUDA layer-program helpers that do not own dispatch.
//!
//! Dispatch, adapter implementations, handle decoding, and kernel
//! initialization live in their own modules. Keep this file limited to the
//! small helpers shared by those modules.

const std = @import("std");
const compute = @import("compute_pkg");
const models = @import("models_pkg");
const layer_ops = models.layer_ops;
const runtime_contract = @import("runtime_contract_pkg");

const engine = @import("../engine.zig");
const engine_ops = @import("../operators/root.zig");
const engine_types = @import("../runtime/root.zig");
const handles = @import("handles.zig");

const AttentionPath = engine_types.AttentionPath;
const CudaBackend = engine.CudaBackend;
const LayerProgramExecutionContext = CudaBackend.LayerProgramExecutionContext;

pub fn residualScaleFactor(self: anytype, scale: layer_ops.ResidualScale) f32 {
    return switch (scale) {
        .one => 1.0,
        .residual_multiplier => self.loaded.config.residual_multiplier,
        .literal => |literal| literal,
    };
}

pub fn runResidualAddRmsnormRowsStrideAware(
    self: anytype,
    fused_fn: compute.cuda.Function,
    residual_out: *compute.cuda.Buffer,
    norm_out: *compute.cuda.Buffer,
    residual_in: *const compute.cuda.Buffer,
    branch: *const compute.cuda.Buffer,
    weight: *const compute.cuda.Buffer,
    residual_scale: f32,
    rows: u32,
    cols: u32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (residual_out.size < packed_bytes or
        norm_out.size < packed_bytes or
        residual_in.size < packed_bytes or
        branch.size < packed_bytes)
    {
        return error.InvalidInstructionBinding;
    }

    var residual_out_stride_elems: u32 = cols;
    var norm_out_stride_elems: u32 = cols;
    var residual_in_stride_elems: u32 = cols;
    var branch_stride_elems: u32 = cols;

    if (!(residual_out.size == packed_bytes and
        norm_out.size == packed_bytes and
        residual_in.size == packed_bytes and
        branch.size == packed_bytes))
    {
        const row_count: usize = @intCast(rows);
        if (residual_out.size % row_count != 0 or
            norm_out.size % row_count != 0 or
            residual_in.size % row_count != 0 or
            branch.size % row_count != 0)
        {
            return error.InvalidInstructionBinding;
        }
        const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
        const residual_out_stride = residual_out.size / row_count;
        const norm_out_stride = norm_out.size / row_count;
        const residual_in_stride = residual_in.size / row_count;
        const branch_stride = branch.size / row_count;
        if (residual_out_stride < row_bytes or
            norm_out_stride < row_bytes or
            residual_in_stride < row_bytes or
            branch_stride < row_bytes)
        {
            return error.InvalidInstructionBinding;
        }
        if ((residual_out_stride % @sizeOf(f32)) != 0 or
            (norm_out_stride % @sizeOf(f32)) != 0 or
            (residual_in_stride % @sizeOf(f32)) != 0 or
            (branch_stride % @sizeOf(f32)) != 0)
        {
            return error.InvalidInstructionBinding;
        }

        residual_out_stride_elems = std.math.cast(u32, residual_out_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        norm_out_stride_elems = std.math.cast(u32, norm_out_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        residual_in_stride_elems = std.math.cast(u32, residual_in_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        branch_stride_elems = std.math.cast(u32, branch_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    }

    try compute.cuda.residual_scaled_rmsnorm_rows_strided.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_fn,
        residual_out,
        norm_out,
        residual_in,
        branch,
        weight,
        residual_scale,
        rows,
        cols,
        residual_out_stride_elems,
        norm_out_stride_elems,
        residual_in_stride_elems,
        branch_stride_elems,
        self.norm_eps,
        self.loaded.runtime.weight_offset,
    );
}

pub fn tryFuseResidualAddIntoNextRmsnorm(
    self: anytype,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    scale: layer_ops.ResidualScale,
    ctx: *LayerProgramExecutionContext,
) !bool {
    const fused_fn = self.residual_scaled_rmsnorm_rows_strided_function orelse return false;
    const compiled = ctx.layer.compiled_plan orelse return error.UnsupportedModel;
    if (ctx.op_index + 1 >= compiled.plan.instructions.len) return false;

    const next_insn = &compiled.plan.instructions[ctx.op_index + 1];
    if (next_insn.opcode != .rmsnorm) return false;
    if (insn.outputs.len != 1 or next_insn.inputs.len != 1 or next_insn.outputs.len != 1) return false;
    if (insn.outputs[0] != next_insn.inputs[0]) return false;

    const io = try handles.instructionIoSlices(insn, registers);
    if (io.inputs.len < 2 or io.outputs.len == 0) return error.InvalidInstructionBinding;
    const residual_src = handles.bufferFromTensorHandle(io.inputs[0]);
    const residual_dst = handles.bufferFromTensorHandle(io.outputs[0]);
    const branch = handles.bufferFromTensorHandle(io.inputs[1]);

    const norm_out_reg = runtime_contract.registerToIndex(next_insn.outputs[0]);
    const norm_out = engine_ops.programBuffer(self, norm_out_reg, ctx) orelse return error.UnsupportedModel;
    const norm_weight = try ctx.layer.instructionNormWeightRef(ctx.op_index + 1);
    try runResidualAddRmsnormRowsStrideAware(
        self,
        fused_fn,
        residual_dst,
        norm_out,
        residual_src,
        branch,
        &norm_weight.buffer,
        residualScaleFactor(self, scale),
        ctx.active_rows_u32,
        ctx.d_model_u32,
    );
    self.skip_next_rmsnorm = true;
    return true;
}

pub fn standaloneLayerScalarOutputScale(self: anytype, ctx: *LayerProgramExecutionContext) f32 {
    if (comptime @hasField(@TypeOf(self.*), "enable_layer_scalars")) {
        if (!self.enable_layer_scalars) return 1.0;
    } else {
        return 1.0;
    }
    if (!@hasField(@TypeOf(self.*), "standalone_layer_scalars")) return 1.0;
    if (!@hasField(@TypeOf(self.*), "standalone_layer_scalar_fused_layers")) return 1.0;

    const scalars = self.standalone_layer_scalars orelse return 1.0;
    const fused_flags = self.standalone_layer_scalar_fused_layers orelse return 1.0;
    if (ctx.layer_index >= scalars.len or ctx.layer_index >= fused_flags.len) return 1.0;
    if (!fused_flags[ctx.layer_index]) return 1.0;
    const compiled = ctx.layer.compiled_plan orelse return 1.0;
    if (ctx.op_index + 1 != compiled.plan.instructions.len) return 1.0;
    return scalars[ctx.layer_index];
}

pub fn finishAttentionRecord(self: anytype, path: AttentionPath, start_ns: i128, is_causal: bool) AttentionPath {
    const elapsed_i128 = std.time.nanoTimestamp() - start_ns;
    const elapsed_ns: u64 = if (elapsed_i128 > 0) @intCast(elapsed_i128) else 0;
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordAttention(path, elapsed_ns);
        self.nvfp4_phase_counters.recordAttentionCausality(is_causal);
        self.nvfp4_phase_counters.recordAttentionContext();
    }
    return path;
}

test "residualScaleFactor preserves residual scale contract" {
    const Backend = struct {
        loaded: struct {
            config: struct {
                residual_multiplier: f32 = 0.5,
            } = .{},
        } = .{},
    };
    var backend = Backend{};
    try std.testing.expectEqual(@as(f32, 1.0), residualScaleFactor(&backend, .one));
    try std.testing.expectEqual(@as(f32, 0.5), residualScaleFactor(&backend, .residual_multiplier));
    try std.testing.expectEqual(@as(f32, 2.0), residualScaleFactor(&backend, .{ .literal = 2.0 }));
}

test "standaloneLayerScalarOutputScale returns one when scalar state is absent" {
    const Backend = struct {};
    var backend = Backend{};
    var ctx = std.mem.zeroes(LayerProgramExecutionContext);
    try std.testing.expectEqual(@as(f32, 1.0), standaloneLayerScalarOutputScale(&backend, @constCast(&ctx)));
}
