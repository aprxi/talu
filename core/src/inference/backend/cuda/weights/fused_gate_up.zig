//! Fused gate-up weight upload helpers for the CUDA inference backend.

const dense = @import("dense.zig");
const upload_dispatch = @import("upload_dispatch.zig");
const denseMaterializeOutInU16 = dense.materializeDenseOutInU16;
const denseMaterializeOutInF32 = dense.materializeDenseOutInF32;
const uploadLinearWeight = upload_dispatch.uploadLinearWeight;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const load_transforms = @import("models_pkg").load.transforms;
const models = @import("models_pkg");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

/// Convert UE8M0 block scale exponent to f32 scale factor.
inline fn ue8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const MoEWeightRefs = engine_types.MoEWeightRefs;
const MoEWeights = models.runtime_blocks.MoEWeights;

pub const FusedGateUpUpload = struct {
    gate: LinearWeight,
    up: LinearWeight,
};

pub fn uploadFusedGateUpWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    fused_gate_up: *const Tensor,
    input_dim: usize,
    layout: GateUpLayout,
) !FusedGateUpUpload {
    if (fused_gate_up.n_dims != 2) return error.UnsupportedModel;
    if (fused_gate_up.shape[0] <= 0 or fused_gate_up.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(fused_gate_up.shape[0]);
    const cols: usize = @intCast(fused_gate_up.shape[1]);
    const out_dim = if (rows == input_dim) cols else if (cols == input_dim) rows else return error.UnsupportedModel;
    if ((out_dim % 2) != 0) return error.InvalidArgument;
    const d_ff = out_dim / 2;

    if (fused_gate_up.dtype == .f16 or fused_gate_up.dtype == .bf16) {
        var out_in = try denseMaterializeOutInU16(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []align(1) const u16 = undefined;
        var up_vals: []align(1) const u16 = undefined;
        var gate_owned: ?[]u16 = null;
        var up_owned: ?[]u16 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(u16, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, fused_gate_up.dtype, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    if (fused_gate_up.dtype == .f32) {
        var out_in = try denseMaterializeOutInF32(allocator, fused_gate_up, input_dim, out_dim);
        defer out_in.deinit(allocator);

        const part_count = std.math.mul(usize, d_ff, input_dim) catch return error.InvalidArgument;
        var gate_vals: []const f32 = undefined;
        var up_vals: []const f32 = undefined;
        var gate_owned: ?[]f32 = null;
        var up_owned: ?[]f32 = null;
        defer if (gate_owned) |buf| allocator.free(buf);
        defer if (up_owned) |buf| allocator.free(buf);
        switch (layout) {
            .concat => {
                gate_vals = out_in.values[0..part_count];
                up_vals = out_in.values[part_count .. part_count * 2];
            },
            .interleaved => {
                const gate_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(gate_tmp);
                const up_tmp = try allocator.alloc(f32, part_count);
                errdefer allocator.free(up_tmp);
                var row: usize = 0;
                while (row < d_ff) : (row += 1) {
                    const gate_src_row = (2 * row) * input_dim;
                    const up_src_row = (2 * row + 1) * input_dim;
                    const dst_row = row * input_dim;
                    @memcpy(gate_tmp[dst_row .. dst_row + input_dim], out_in.values[gate_src_row .. gate_src_row + input_dim]);
                    @memcpy(up_tmp[dst_row .. dst_row + input_dim], out_in.values[up_src_row .. up_src_row + input_dim]);
                }
                gate_vals = gate_tmp;
                up_vals = up_tmp;
                gate_owned = gate_tmp;
                up_owned = up_tmp;
            },
        }

        const gate_bytes = std.mem.sliceAsBytes(gate_vals);
        const up_bytes = std.mem.sliceAsBytes(up_vals);
        var gate_tensor = Tensor.view(@constCast(gate_bytes.ptr), &.{ d_ff, input_dim }, .f32, gate_bytes.len);
        var up_tensor = Tensor.view(@constCast(up_bytes.ptr), &.{ d_ff, input_dim }, .f32, up_bytes.len);
        const gate = try uploadLinearWeight(device, allocator, &gate_tensor, input_dim);
        errdefer {
            var gate_mut = gate;
            gate_mut.deinit(device);
        }
        const up = try uploadLinearWeight(device, allocator, &up_tensor, input_dim);
        return .{ .gate = gate, .up = up };
    }

    return error.UnsupportedModel;
}
