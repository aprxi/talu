//! Dense weight helpers for the CUDA inference backend.

const host = @import("host.zig");
const materializeTensorF32 = host.materializeTensorF32;

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

pub const DenseLinearLayout = struct {
    in_dim: usize,
    out_dim: usize,
    needs_transpose: bool,
};

pub fn resolveDenseInOutLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Canonical checkpoint layout is [out_dim, in_dim].
    // Prefer this branch first so square matrices (rows == cols == input_dim)
    // are treated as [out,in] and transposed to the kernel layout [in,out].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = true,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = false,
        };
    }
    return error.UnsupportedModel;
}

pub fn resolveDenseOutInLayout(rows: usize, cols: usize, input_dim: usize) !DenseLinearLayout {
    if (input_dim == 0 or rows == 0 or cols == 0) return error.InvalidArgument;
    // Typed (f16/bf16) path follows loader policy: canonical [out_dim, in_dim].
    if (cols == input_dim) {
        return .{
            .in_dim = cols,
            .out_dim = rows,
            .needs_transpose = false,
        };
    }
    if (rows == input_dim) {
        return .{
            .in_dim = rows,
            .out_dim = cols,
            .needs_transpose = true,
        };
    }
    return error.UnsupportedModel;
}

pub fn transposeRowMajor(
    comptime T: type,
    allocator: std.mem.Allocator,
    src: []align(1) const T,
    rows: usize,
    cols: usize,
) ![]T {
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    if (src.len < logical_count) return error.InvalidArgument;
    const out = try allocator.alloc(T, logical_count);
    errdefer allocator.free(out);

    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < cols) : (c += 1) {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    return out;
}

pub const DenseOutInU16 = struct {
    values: []align(1) const u16,
    owned: ?[]u16 = null,

    pub fn deinit(self: *DenseOutInU16, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

pub const DenseOutInF32 = struct {
    values: []const f32,
    owned: ?[]f32 = null,

    pub fn deinit(self: *DenseOutInF32, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
        self.* = .{ .values = &.{}, .owned = null };
    }
};

pub const FusedQkvUpload = struct {
    q: LinearWeight,
    k: LinearWeight,
    v: LinearWeight,
};

pub const FusedGateUpUpload = struct {
    gate: LinearWeight,
    up: LinearWeight,
};

pub fn materializeDenseOutInU16(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInU16 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSliceUnaligned(u16);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(u16, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
}

pub fn materializeDenseOutInF32(
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
    out_dim: usize,
) !DenseOutInF32 {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;
    const view = src.asSlice(f32);
    if (view.len < logical_count) return error.InvalidArgument;
    const values = view[0..logical_count];

    if (rows == out_dim and cols == input_dim) {
        return .{ .values = values };
    }
    if (rows == input_dim and cols == out_dim) {
        const transposed = try transposeRowMajor(f32, allocator, values, rows, cols);
        return .{ .values = transposed, .owned = transposed };
    }
    return error.UnsupportedModel;
}

pub fn uploadLinearWeightDense(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.dtype == .f16 or src.dtype == .bf16) {
        return uploadLinearWeightDenseU16(device, allocator, src, input_dim);
    }
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);

    const host_f32 = try materializeTensorF32(allocator, src);
    defer allocator.free(host_f32);

    const layout = resolveDenseInOutLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA dense linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []f32 = host_f32;
    var transposed: ?[]f32 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(f32, allocator, host_f32, rows, cols);
        oriented = transposed.?;
    }

    var buffer = try device.allocBuffer(oriented.len * @sizeOf(f32));
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    return .{
        .dense_f32 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
        },
    };
}

pub fn uploadLinearWeightDenseU16(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    src: *const Tensor,
    input_dim: usize,
) !LinearWeight {
    if (src.n_dims != 2) return error.UnsupportedModel;
    if (src.shape[0] <= 0 or src.shape[1] <= 0) return error.InvalidArgument;
    const rows: usize = @intCast(src.shape[0]);
    const cols: usize = @intCast(src.shape[1]);
    const logical_count = std.math.mul(usize, rows, cols) catch return error.InvalidArgument;

    const host_u16 = src.asSliceUnaligned(u16);
    if (host_u16.len < logical_count) return error.InvalidArgument;
    const view = host_u16[0..logical_count];

    const layout = resolveDenseOutInLayout(rows, cols, input_dim) catch |err| {
        log.warn("inference", "CUDA u16 linear weight orientation unsupported", .{
            .rows = rows,
            .cols = cols,
            .input_dim = input_dim,
            .dtype = @tagName(src.dtype),
        });
        return err;
    };

    var oriented: []align(1) const u16 = view;
    var transposed: ?[]u16 = null;
    defer if (transposed) |t| allocator.free(t);
    if (layout.needs_transpose) {
        transposed = try transposeRowMajor(u16, allocator, view, rows, cols);
        oriented = transposed.?;
    }

    const bytes = std.math.mul(usize, oriented.len, @sizeOf(u16)) catch return error.InvalidArgument;
    var buffer = try device.allocBuffer(bytes);
    errdefer buffer.deinit(device);
    try buffer.upload(device, std.mem.sliceAsBytes(oriented));

    const dense_dtype: DenseU16Dtype = switch (src.dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => return error.UnsupportedModel,
    };

    return .{
        .dense_u16 = .{
            .rows = layout.in_dim,
            .cols = layout.out_dim,
            .buffer = buffer,
            .dtype = dense_dtype,
        },
    };
}
