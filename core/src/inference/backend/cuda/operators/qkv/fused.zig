//! Fused three-output projection wrappers for the CUDA inference backend.

const std = @import("std");
const compute = @import("compute_pkg");

const engine_types = @import("../../runtime/root.zig");
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const QkvI8ConcatRef = engine_types.QkvI8ConcatRef;

const fused_common = @import("../fused_linear.zig");
const fused_linear = fused_common.linear;
const LinearWeightTag = std.meta.Tag(LinearWeight);

fn tripleOutputs(
    self: anytype,
    first_out: *compute.cuda.Buffer,
) fused_linear.TripleOutputs {
    return .{
        .first = first_out,
        .second = &self.runtime_buffers.attn_k_dev,
        .third = &self.runtime_buffers.attn_v_dev,
    };
}

fn tripleHasTag(
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    comptime tag: LinearWeightTag,
) bool {
    return std.meta.activeTag(first_weight.*) == tag and
        std.meta.activeTag(second_weight.*) == tag and
        std.meta.activeTag(third_weight.*) == tag;
}

fn isGaffineU4Triple(first_weight: *const LinearWeight, second_weight: *const LinearWeight, third_weight: *const LinearWeight) bool {
    return tripleHasTag(first_weight, second_weight, third_weight, .gaffine_u4);
}

fn isGaffineU8Triple(first_weight: *const LinearWeight, second_weight: *const LinearWeight, third_weight: *const LinearWeight) bool {
    return tripleHasTag(first_weight, second_weight, third_weight, .gaffine_u8);
}

fn isNvfp4Triple(first_weight: *const LinearWeight, second_weight: *const LinearWeight, third_weight: *const LinearWeight) bool {
    return tripleHasTag(first_weight, second_weight, third_weight, .nvfp4);
}

fn isDenseU16Triple(first_weight: *const LinearWeight, second_weight: *const LinearWeight, third_weight: *const LinearWeight) bool {
    return tripleHasTag(first_weight, second_weight, third_weight, .dense_u16);
}

fn runTripleKernel(
    self: anytype,
    input: *const compute.cuda.Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: fused_linear.TripleOutputs,
) !bool {
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const ok = fused_linear.tryTripleKernel(&ctx, input, first_weight, second_weight, third_weight, rows, input_dim, outputs) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

pub fn tryFusedQkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    return runTripleKernel(self, input, q_proj, k_proj, v_proj, rows, self.d_model, tripleOutputs(self, q_out_dest));
}

pub fn tryFusedGaffineU4QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (!isGaffineU4Triple(q_proj, k_proj, v_proj)) return false;
    return runTripleKernel(self, input, q_proj, k_proj, v_proj, rows, self.d_model, tripleOutputs(self, q_out_dest));
}

pub fn tryFusedGaffineU8QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (!isGaffineU8Triple(q_proj, k_proj, v_proj)) return false;
    return runTripleKernel(self, input, q_proj, k_proj, v_proj, rows, self.d_model, tripleOutputs(self, q_out_dest));
}

pub fn tryFusedNvfp4QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (!isNvfp4Triple(q_proj, k_proj, v_proj)) return false;
    return runTripleKernel(self, input, q_proj, k_proj, v_proj, rows, self.d_model, tripleOutputs(self, q_out_dest));
}

pub fn tryFusedNvfp4QkvLtForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out: *compute.cuda.Buffer,
    k_out: *compute.cuda.Buffer,
    v_out: *compute.cuda.Buffer,
) !bool {
    if (!isNvfp4Triple(q_proj, k_proj, v_proj)) return false;
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const outputs = fused_linear.TripleOutputs{ .first = q_out, .second = k_out, .third = v_out };
    const input_dim = q_proj.rows();
    const ok = fused_linear.tryTripleNvfp4Lt(&ctx, input, q_proj, k_proj, v_proj, rows, input_dim, outputs) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

pub fn tryFusedDenseU16QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (!isDenseU16Triple(q_proj, k_proj, v_proj)) return false;
    return runTripleKernel(self, input, q_proj, k_proj, v_proj, rows, self.d_model, tripleOutputs(self, q_out_dest));
}

pub fn tryFusedQkvI8ConcatForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    rows: usize,
    input_dim: usize,
    concat: QkvI8ConcatRef,
    q_out: *compute.cuda.Buffer,
    k_out: *compute.cuda.Buffer,
    v_out: *compute.cuda.Buffer,
) !bool {
    var diagnostics = fused_linear.Diagnostics{};
    var capabilities = fused_common.makeCapabilities(self);
    var ctx = fused_common.makeContext(self, &diagnostics, &capabilities);
    const descriptor = fused_linear.ConcatI8TripleWeight{
        .input_dim = input_dim,
        .output_dims = concat.dims,
        .i8_buffer = concat.i8_buf,
        .scales_buffer = concat.scales_buf,
    };
    const outputs = fused_linear.TripleOutputs{ .first = q_out, .second = k_out, .third = v_out };
    const ok = fused_linear.tryTripleI8Concat(&ctx, input, rows, &descriptor, outputs) catch |err| {
        fused_common.syncCapabilityFlags(self, &capabilities);
        fused_common.emitDiagnostics(self, &diagnostics);
        return err;
    };
    fused_common.syncCapabilityFlags(self, &capabilities);
    fused_common.emitDiagnostics(self, &diagnostics);
    return ok;
}

pub fn canFuseDenseU16QkvWeights(d_model: usize, q: U16LinearWeight, k: U16LinearWeight, v: U16LinearWeight) bool {
    return fused_linear.denseU16TripleWeightsCompatible(d_model, q, k, v);
}

pub fn canFuseGaffineQkvWeights(
    d_model: usize,
    q: anytype,
    k: anytype,
    v: anytype,
) bool {
    return fused_linear.gaffineTripleWeightsCompatible(d_model, q, k, v);
}

fn zeroDenseU16Weight() LinearWeight {
    return .{ .dense_u16 = std.mem.zeroes(U16LinearWeight) };
}

fn zeroGaffineU4Weight() LinearWeight {
    return .{ .gaffine_u4 = std.mem.zeroes(engine_types.GaffineU4LinearWeight) };
}

fn zeroGaffineU8Weight() LinearWeight {
    return .{ .gaffine_u8 = std.mem.zeroes(engine_types.GaffineU8LinearWeight) };
}

fn zeroNvfp4Weight() LinearWeight {
    return .{ .nvfp4 = std.mem.zeroes(engine_types.Nvfp4LinearWeight) };
}

pub const testing = struct {
    pub fn expectWrapperTagPolicy() !void {
        const dense = zeroDenseU16Weight();
        const gaffine_u4 = zeroGaffineU4Weight();
        const gaffine_u8 = zeroGaffineU8Weight();
        const nvfp4 = zeroNvfp4Weight();

        try std.testing.expect(isDenseU16Triple(&dense, &dense, &dense));
        try std.testing.expect(!isDenseU16Triple(&gaffine_u4, &gaffine_u4, &gaffine_u4));
        try std.testing.expect(isGaffineU4Triple(&gaffine_u4, &gaffine_u4, &gaffine_u4));
        try std.testing.expect(!isGaffineU4Triple(&gaffine_u8, &gaffine_u8, &gaffine_u8));
        try std.testing.expect(isGaffineU8Triple(&gaffine_u8, &gaffine_u8, &gaffine_u8));
        try std.testing.expect(!isGaffineU8Triple(&nvfp4, &nvfp4, &nvfp4));
        try std.testing.expect(isNvfp4Triple(&nvfp4, &nvfp4, &nvfp4));
        try std.testing.expect(!isNvfp4Triple(&dense, &dense, &dense));
    }
};

test "inference.backend.cuda qkv fused wrapper tag policy matches wrapper names" {
    try testing.expectWrapperTagPolicy();
}
