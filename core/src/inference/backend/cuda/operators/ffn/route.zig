//! FFN projection routing for the CUDA inference backend.

const fused_gate_up = @import("fused_gate_up.zig");
const linear = @import("../linear/root.zig");
const tryFusedGateUpForward = fused_gate_up.tryFusedGateUpForward;
const tryFusedNvfp4GateUpLtForward = fused_gate_up.tryFusedNvfp4GateUpLtForward;
const linearForwardRows = linear.linearForwardRows;

const std = @import("std");
const compute = @import("compute_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/root.zig");
const LinearWeight = engine_types.LinearWeight;
const ProjectionPath = engine_types.ProjectionPath;

// --- Utility functions from engine_weights.zig ---
const engine_weights = @import("../../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

fn recordPhaseGateUpPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordGateUp(path);
    }
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

    const gate_row_bytes = std.math.mul(usize, gate_weight.cols(), @sizeOf(f32)) catch return error.InvalidArgument;
    const up_row_bytes = std.math.mul(usize, up_weight.cols(), @sizeOf(f32)) catch return error.InvalidArgument;
    const gate_bytes = std.math.mul(usize, rows, gate_row_bytes) catch return error.InvalidArgument;
    const up_bytes = std.math.mul(usize, rows, up_row_bytes) catch return error.InvalidArgument;
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
