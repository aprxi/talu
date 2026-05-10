//! FFN projection routing for the CUDA inference backend.

const fused_gate_up = @import("fused_gate_up.zig");
const linear = @import("../linear/root.zig");
const tryFusedGateUpForward = fused_gate_up.tryFusedGateUpForward;
const tryFusedNvfp4GateUpLtForward = fused_gate_up.tryFusedNvfp4GateUpLtForward;
const linearForwardRows = linear.linearForwardRows;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/_types_impl.zig");
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
const engine_weights = @import("../../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

fn recordNvfp4Route(self: anytype, comptime kind: Nvfp4RouteKind) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_route_counters")) {
        self.nvfp4_route_counters.record(kind);
    }
}

fn recordPhaseLinearNs(self: anytype, elapsed_ns: u64) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordLinear(elapsed_ns);
    }
}

fn recordPhaseQkvPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordQkv(path);
    }
}

fn recordPhaseGateUpPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordGateUp(path);
    }
}

fn phaseEventTimingEnabled(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "phase_event_timing_enabled")) {
        return self.phase_event_timing_enabled;
    }
    return false;
}

fn tryPrepareNvfp4LtInput(
    self: anytype,
    input: *const compute.cuda.Buffer,
    in_dim: usize,
    rows: usize,
    input_fp4_out: *compute.cuda.Buffer,
    input_scales_out: *compute.cuda.Buffer,
) !bool {
    const quant_fn = self.quantize_f32_to_nvfp4_function orelse return false;
    const packed_in_cols = (in_dim + 1) / 2;
    const input_fp4_bytes = std.math.mul(usize, rows, packed_in_cols) catch return false;
    const input_scale_bytes = engine_types.Nvfp4LinearWeight.cublasLtScaleTensorSize(in_dim, rows);
    if (self.runtime_buffers.activation_u16_dev.size < input_fp4_bytes) return false;
    if (self.runtime_buffers.dequant_f16_dev.size < input_scale_bytes) return false;

    input_fp4_out.* = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_fp4_bytes) catch return false;
    input_scales_out.* = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, input_scale_bytes) catch return false;

    const padded_outer: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(rows, 128));
    const sf_k = (in_dim + 15) / 16;
    const padded_sf_k: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(sf_k, 4));
    const quant_grid_x = std.math.cast(u32, sf_k) orelse return false;

    self.kernel_arg_pack.reset();
    self.kernel_arg_pack.appendBufferPtr(input) catch return false;
    self.kernel_arg_pack.appendBufferPtr(input_fp4_out) catch return false;
    self.kernel_arg_pack.appendBufferPtr(input_scales_out) catch return false;
    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch return false;
    self.kernel_arg_pack.appendScalar(u32, @intCast(rows)) catch return false;
    self.kernel_arg_pack.appendScalar(u32, padded_outer) catch return false;
    self.kernel_arg_pack.appendScalar(u32, padded_sf_k) catch return false;
    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
        .grid_x = quant_grid_x,
        .grid_y = padded_outer,
        .block_x = 32,
    }, &self.kernel_arg_pack, .other) catch return false;

    return true;
}

pub fn runGateUpProjectionWithWeights(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !ProjectionPath {
    const prefer_nvfp4_lt_fused = rows > 1 and switch (gate_weight.*) {
        .nvfp4 => switch (up_weight.*) {
            .nvfp4 => true,
            else => false,
        },
        else => false,
    };
    if (!prefer_nvfp4_lt_fused) {
        if (try tryFusedGateUpForward(self, input, gate_weight, up_weight, rows)) {
            recordPhaseGateUpPath(self, .fused);
            return .fused;
        }
    }

    const gate_bytes = std.math.mul(usize, rows, gate_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    const up_bytes = std.math.mul(usize, rows, up_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    var gate_out = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, gate_bytes);
    var up_out = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, up_bytes);
    if (try tryFusedNvfp4GateUpLtForward(self, input, gate_weight, up_weight, rows, &gate_out, &up_out)) {
        recordPhaseGateUpPath(self, .fused);
        return .fused;
    }
    try linearForwardRows(self, input, rows, gate_weight, &gate_out);
    try linearForwardRows(self, input, rows, up_weight, &up_out);
    recordPhaseGateUpPath(self, .unfused);
    return .unfused;
}

pub fn runFfnActivationMul(self: anytype, count: u32) !void {
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
