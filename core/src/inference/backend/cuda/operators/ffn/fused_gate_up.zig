//! Fused two-output projection wrappers for the CUDA inference backend.

const std = @import("std");
const compute = @import("compute_pkg");

const engine_types = @import("../../runtime/root.zig");
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;

const fused_common = @import("../fused_linear.zig");
const fused_linear = fused_common.linear;
const LinearWeightTag = std.meta.Tag(LinearWeight);

fn pairSplitOutputs(self: anytype) fused_linear.PairOutputs {
    return .{
        .first = &self.runtime_buffers.ffn_gate_dev,
        .second = &self.runtime_buffers.ffn_up_dev,
    };
}

fn pairActivatedOutputs(self: anytype) fused_linear.PairOutputs {
    return .{
        .first = &self.runtime_buffers.ffn_gate_dev,
        .second = &self.runtime_buffers.ffn_up_dev,
        .activated_product = &self.runtime_buffers.ffn_mul_dev,
    };
}

fn pairHasTag(first_weight: *const LinearWeight, second_weight: *const LinearWeight, comptime tag: LinearWeightTag) bool {
    return std.meta.activeTag(first_weight.*) == tag and std.meta.activeTag(second_weight.*) == tag;
}

fn isGaffineU8Pair(first_weight: *const LinearWeight, second_weight: *const LinearWeight) bool {
    return pairHasTag(first_weight, second_weight, .gaffine_u8);
}

fn isDenseU16Pair(first_weight: *const LinearWeight, second_weight: *const LinearWeight) bool {
    return pairHasTag(first_weight, second_weight, .dense_u16);
}

fn runPairSplit(
    self: anytype,
    input: *const compute.cuda.Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: fused_linear.PairOutputs,
) !bool {
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const ok = fused_linear.tryPairSplit(&ctx, input, first_weight, second_weight, rows, input_dim, outputs) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

fn runPairActivated(
    self: anytype,
    input: *const compute.cuda.Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: fused_linear.PairActivation,
) !bool {
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const ok = fused_linear.tryPairActivated(
        &ctx,
        input,
        first_weight,
        second_weight,
        rows,
        input_dim,
        expected_output_dim,
        activation,
        pairActivatedOutputs(self),
    ) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

pub fn canFuseGaffineGateUpWeights(
    d_model: usize,
    gate: anytype,
    up: anytype,
) bool {
    return fused_linear.gaffinePairWeightsCompatible(d_model, gate, up);
}

pub fn tryFusedGaffineU8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!isGaffineU8Pair(gate_weight, up_weight)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn tryFusedGaffineU4GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!pairHasTag(gate_weight, up_weight, .gaffine_u4)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn tryFusedFp8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!pairHasTag(gate_weight, up_weight, .fp8)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn tryFusedFp8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (!pairHasTag(gate_weight, up_weight, .fp8)) return false;
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedMxfp8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!pairHasTag(gate_weight, up_weight, .mxfp8)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn tryFusedNvfp4GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!pairHasTag(gate_weight, up_weight, .nvfp4)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn tryFusedNvfp4GateUpGeluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (!self.loaded.config.use_gelu) return false;
    if (!pairHasTag(gate_weight, up_weight, .nvfp4)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .gelu);
}

pub fn tryFusedMxfp8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (!pairHasTag(gate_weight, up_weight, .mxfp8)) return false;
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedNvfp4GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (!pairHasTag(gate_weight, up_weight, .nvfp4)) return false;
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedNvfp4GateUpLtForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    gate_out: *compute.cuda.Buffer,
    up_out: *compute.cuda.Buffer,
) !bool {
    if (!pairHasTag(gate_weight, up_weight, .nvfp4)) return false;
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const outputs = fused_linear.PairOutputs{ .first = gate_out, .second = up_out };
    const input_dim = gate_weight.rows();
    const ok = fused_linear.tryPairNvfp4Lt(&ctx, input, gate_weight, up_weight, rows, input_dim, outputs) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

pub fn tryFusedGateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedGaffineU8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (!isGaffineU8Pair(gate_weight, up_weight)) return false;
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedDenseU16GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (!isDenseU16Pair(gate_weight, up_weight)) return false;
    return runPairSplit(self, input, gate_weight, up_weight, rows, self.d_model, pairSplitOutputs(self));
}

pub fn tryFusedDenseU16GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (!isDenseU16Pair(gate_weight, up_weight)) return false;
    return runPairActivated(self, input, gate_weight, up_weight, rows, self.d_model, expected_out_dim, .silu);
}

pub fn canFuseDenseU16GateUpWeights(d_model: usize, gate: U16LinearWeight, up: U16LinearWeight) bool {
    return fused_linear.denseU16PairWeightsCompatible(d_model, gate, up);
}

fn zeroDenseU16Weight() LinearWeight {
    return .{ .dense_u16 = std.mem.zeroes(U16LinearWeight) };
}

fn zeroGaffineU8Weight() LinearWeight {
    return .{ .gaffine_u8 = std.mem.zeroes(engine_types.GaffineU8LinearWeight) };
}

pub const testing = struct {
    pub fn expectSplitWrapperTagPolicy() !void {
        const dense = zeroDenseU16Weight();
        const gaffine_u8 = zeroGaffineU8Weight();

        try std.testing.expect(isDenseU16Pair(&dense, &dense));
        try std.testing.expect(!isDenseU16Pair(&gaffine_u8, &gaffine_u8));
        try std.testing.expect(isGaffineU8Pair(&gaffine_u8, &gaffine_u8));
        try std.testing.expect(!isGaffineU8Pair(&dense, &dense));
    }
};

test "inference.backend.cuda ffn fused split wrapper tag policy matches wrapper names" {
    try testing.expectSplitWrapperTagPolicy();
}
