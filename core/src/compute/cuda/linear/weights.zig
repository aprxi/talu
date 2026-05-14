//! CUDA device tensor and generic linear-weight handle types.

const std = @import("std");
const device_mod = @import("../device.zig");
const tensor = @import("../../tensor.zig");

const Tensor = tensor.Tensor;

pub const DenseU16Dtype = enum(u8) {
    f16,
    bf16,
};

pub const EmbeddingLookupKind = enum(u8) {
    f32,
    f16,
    bf16,
    gaffine_u4,
};

pub const DeviceTensor = struct {
    rows: usize,
    cols: usize,
    buffer: device_mod.Buffer,

    pub fn deinit(self: *DeviceTensor, device: *device_mod.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const DeviceTensor) usize {
        return self.buffer.size;
    }
};

pub const missing_device_tensor: DeviceTensor = std.mem.zeroes(DeviceTensor);
pub const missing_host_tensor: Tensor = std.mem.zeroes(Tensor);

pub const EmbeddingLookup = struct {
    kind: EmbeddingLookupKind,
    dim0: u32,
    dim1: u32,
    hidden_dim: u32,
    layout_tag: u32,
    group_size: u32 = 0,
    scales_dtype_tag: u32 = 0,
    scales: ?device_mod.Buffer = null,
    biases: ?device_mod.Buffer = null,
    multiplier: f32,
    buffer: device_mod.Buffer,

    pub fn deinit(self: *EmbeddingLookup, device: *device_mod.Device) void {
        if (self.biases) |*buf| buf.deinit(device);
        if (self.scales) |*buf| buf.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const EmbeddingLookup) usize {
        return self.buffer.size +
            (if (self.scales) |buf| buf.size else 0) +
            (if (self.biases) |buf| buf.size else 0);
    }
};

pub const GaffineU4LinearWeight = struct {
    rows: usize,
    cols: usize,
    packed_data: device_mod.Buffer,
    scales: device_mod.Buffer,
    biases: device_mod.Buffer,
    group_size: u32,
    scales_dtype_tag: u32,
    dequant_f16_cache: device_mod.Buffer = .{ .pointer = 0, .size = 0 },
    dequant_i8_cache: device_mod.Buffer = .{ .pointer = 0, .size = 0 },
    mean_scale_cache: device_mod.Buffer = .{ .pointer = 0, .size = 0 },

    pub fn deinit(self: *GaffineU4LinearWeight, device: *device_mod.Device) void {
        if (self.mean_scale_cache.pointer != 0) self.mean_scale_cache.deinit(device);
        if (self.dequant_i8_cache.pointer != 0) self.dequant_i8_cache.deinit(device);
        if (self.dequant_f16_cache.pointer != 0) self.dequant_f16_cache.deinit(device);
        self.biases.deinit(device);
        self.scales.deinit(device);
        self.packed_data.deinit(device);
    }

    pub fn byteSize(self: *const GaffineU4LinearWeight) usize {
        return self.packed_data.size + self.scales.size + self.biases.size +
            self.dequant_f16_cache.size + self.dequant_i8_cache.size + self.mean_scale_cache.size;
    }
};

pub const GaffineU8LinearWeight = GaffineU4LinearWeight;

pub const U16LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: device_mod.Buffer,
    dtype: DenseU16Dtype,

    pub fn deinit(self: *U16LinearWeight, device: *device_mod.Device) void {
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const U16LinearWeight) usize {
        return self.buffer.size;
    }
};

pub const Fp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    buffer: device_mod.Buffer,
    /// GPU buffer holding BF16 per-block scales [scale_rows × scale_cols]
    scales_buffer: device_mod.Buffer,
    scale_rows: u32,
    scale_cols: u32,
    block_size: u32,
    /// Per-tensor scale (used only when scale_rows == 1 && scale_cols == 1)
    weight_scale_inv: f32,

    pub fn deinit(self: *Fp8LinearWeight, device: *device_mod.Device) void {
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Fp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size;
    }
};

pub const Mxfp8LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// GPU buffer holding E4M3 weight bytes [rows × cols]
    buffer: device_mod.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_buffer: device_mod.Buffer,
    /// GPU buffer holding UE8M0 block-32 scales in simple row-major layout
    /// [cols × scale_cols] for the GEMV kernel path. Same data, different layout.
    scales_raw_buffer: device_mod.Buffer,
    scale_cols: u32,

    pub fn deinit(self: *Mxfp8LinearWeight, device: *device_mod.Device) void {
        if (self.scales_raw_buffer.size > 0) self.scales_raw_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Mxfp8LinearWeight) usize {
        return self.buffer.size + self.scales_buffer.size + self.scales_raw_buffer.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC32_UE8M0 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns the total number of UE8M0 scale bytes needed (padded to 128-tile boundaries).
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const block_rows: usize = 128; // inner dimension tile
        const block_cols: usize = 128; // outer dimension tile
        const s_rows = roundoff(inner, block_rows) / 32;
        const s_cols = roundoff(outer, block_cols);
        return s_rows * s_cols;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const Nvfp4LinearWeight = struct {
    rows: usize,
    cols: usize,
    /// Packed FP4 bytes: [out_dim × packed_in] (2 FP4 values per byte).
    buffer: device_mod.Buffer,
    /// FP8 E4M3 scales in row-major layout [out_dim × scale_cols].
    scales_buffer: device_mod.Buffer,
    /// FP8 UE4M3 block-16 scales in cuBLASLt interleaved layout.
    /// Padded to 128-tile boundaries: [padded_outer × padded_sf_k] bytes.
    scales_lt_buffer: device_mod.Buffer,
    packed_cols: u32,
    scale_cols: u32,
    group_size: u32,
    weight_global_scale: f32,
    /// Optional persistent INT8 cache for fast small-row kernels.
    /// Built by the caller before execution via NVFP4->I8 conversion.
    dequant_i8_cache: device_mod.Buffer = .{ .pointer = 0, .size = 0 },
    /// Per-output-row scales for dequant_i8_cache.
    mean_scale_cache: device_mod.Buffer = .{ .pointer = 0, .size = 0 },

    pub fn deinit(self: *Nvfp4LinearWeight, device: *device_mod.Device) void {
        if (self.mean_scale_cache.pointer != 0) self.mean_scale_cache.deinit(device);
        if (self.dequant_i8_cache.pointer != 0) self.dequant_i8_cache.deinit(device);
        if (self.scales_lt_buffer.size > 0) self.scales_lt_buffer.deinit(device);
        if (self.scales_buffer.size > 0) self.scales_buffer.deinit(device);
        self.buffer.deinit(device);
    }

    pub fn byteSize(self: *const Nvfp4LinearWeight) usize {
        return self.buffer.size +
            self.scales_buffer.size +
            self.scales_lt_buffer.size +
            self.dequant_i8_cache.size +
            self.mean_scale_cache.size;
    }

    /// Compute cuBLASLt-required scale tensor size for VEC16_UE4M3 block scaling.
    /// inner = contraction dimension (K), outer = non-contraction dimension (M or N).
    /// Returns total UE4M3 scale bytes padded to cuBLASLt tile boundaries.
    pub fn cublasLtScaleTensorSize(inner: usize, outer: usize) usize {
        const sf_k = roundoff((inner + 15) / 16, 4);
        return roundoff(outer, 128) * sf_k;
    }

    pub fn roundoff(x: usize, granul: usize) usize {
        return granul * ((x + (granul - 1)) / granul);
    }
};

pub const LinearWeight = union(enum) {
    dense_f32: DeviceTensor,
    dense_u16: U16LinearWeight,
    gaffine_u4: GaffineU4LinearWeight,
    gaffine_u8: GaffineU8LinearWeight,
    fp8: Fp8LinearWeight,
    mxfp8: Mxfp8LinearWeight,
    nvfp4: Nvfp4LinearWeight,

    pub fn deinit(self: *LinearWeight, device: *device_mod.Device) void {
        switch (self.*) {
            .dense_f32 => |*w| w.deinit(device),
            .dense_u16 => |*w| w.deinit(device),
            .gaffine_u4 => |*w| w.deinit(device),
            .gaffine_u8 => |*w| w.deinit(device),
            .fp8 => |*w| w.deinit(device),
            .mxfp8 => |*w| w.deinit(device),
            .nvfp4 => |*w| w.deinit(device),
        }
    }

    pub fn rows(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.rows,
            .dense_u16 => |w| w.rows,
            .gaffine_u4 => |w| w.rows,
            .gaffine_u8 => |w| w.rows,
            .fp8 => |w| w.rows,
            .mxfp8 => |w| w.rows,
            .nvfp4 => |w| w.rows,
        };
    }

    pub fn cols(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.cols,
            .dense_u16 => |w| w.cols,
            .gaffine_u4 => |w| w.cols,
            .gaffine_u8 => |w| w.cols,
            .fp8 => |w| w.cols,
            .mxfp8 => |w| w.cols,
            .nvfp4 => |w| w.cols,
        };
    }

    pub fn byteSize(self: *const LinearWeight) usize {
        return switch (self.*) {
            .dense_f32 => |w| w.buffer.size,
            .dense_u16 => |w| w.byteSize(),
            .gaffine_u4 => |w| w.byteSize(),
            .gaffine_u8 => |w| w.byteSize(),
            .fp8 => |w| w.byteSize(),
            .mxfp8 => |w| w.byteSize(),
            .nvfp4 => |w| w.byteSize(),
        };
    }
};

pub fn bufferSlice(buffer: *const device_mod.Buffer, byte_offset: usize, byte_len: usize) !device_mod.Buffer {
    if (byte_offset > buffer.size) return error.InvalidArgument;
    const end = std.math.add(usize, byte_offset, byte_len) catch return error.InvalidArgument;
    if (end > buffer.size) return error.InvalidArgument;
    const ptr = std.math.add(u64, buffer.pointer, @intCast(byte_offset)) catch return error.InvalidArgument;
    return .{
        .pointer = ptr,
        .size = byte_len,
    };
}

pub fn bufferF32RowCount(buffer: *const device_mod.Buffer, width: usize) !usize {
    if (width == 0) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return error.InvalidArgument;
    if (row_bytes == 0) return error.InvalidArgument;
    const rows = std.math.divExact(usize, buffer.size, row_bytes) catch return error.InvalidArgument;
    if (rows == 0) return error.InvalidArgument;
    return rows;
}

pub fn logicalF32RowSlice(
    buffer: *const device_mod.Buffer,
    rows: usize,
    row_index: usize,
    logical_width: usize,
) !device_mod.Buffer {
    if (rows == 0 or logical_width == 0 or row_index >= rows) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, logical_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
    if (buffer.size < packed_bytes) return error.InvalidInstructionBinding;

    const row_stride = if (buffer.size == packed_bytes)
        row_bytes
    else blk: {
        if (buffer.size % rows != 0) return error.InvalidInstructionBinding;
        const stride = buffer.size / rows;
        if (stride < row_bytes) return error.InvalidInstructionBinding;
        break :blk stride;
    };

    const row_offset = std.math.mul(usize, row_index, row_stride) catch return error.InvalidArgument;
    return bufferSlice(buffer, row_offset, row_bytes);
}

pub fn linearWeightHasI8Cache(weight: *const LinearWeight) bool {
    return switch (weight.*) {
        .gaffine_u4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .gaffine_u8 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .nvfp4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        else => false,
    };
}

test "bufferSlice returns bounded device-buffer views" {
    const base = device_mod.Buffer{ .pointer = 0x1000, .size = 128 };
    const view = try bufferSlice(&base, 16, 32);
    try std.testing.expectEqual(@as(u64, 0x1010), view.pointer);
    try std.testing.expectEqual(@as(usize, 32), view.size);

    try std.testing.expectError(error.InvalidArgument, bufferSlice(&base, 129, 1));
    try std.testing.expectError(error.InvalidArgument, bufferSlice(&base, 120, 16));
}

test "bufferF32RowCount counts exact packed f32 rows" {
    const buffer = device_mod.Buffer{ .pointer = 0x2000, .size = 3 * 5 * @sizeOf(f32) };
    try std.testing.expectEqual(@as(usize, 3), try bufferF32RowCount(&buffer, 5));

    try std.testing.expectError(error.InvalidArgument, bufferF32RowCount(&buffer, 0));
    const partial = device_mod.Buffer{ .pointer = 0x2000, .size = 19 };
    try std.testing.expectError(error.InvalidArgument, bufferF32RowCount(&partial, 5));
}

test "logicalF32RowSlice handles packed and strided rows" {
    const packed_buffer = device_mod.Buffer{ .pointer = 0x3000, .size = 3 * 4 * @sizeOf(f32) };
    const packed_row = try logicalF32RowSlice(&packed_buffer, 3, 2, 4);
    try std.testing.expectEqual(@as(u64, 0x3000 + 2 * 4 * @sizeOf(f32)), packed_row.pointer);
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), packed_row.size);

    const strided = device_mod.Buffer{ .pointer = 0x4000, .size = 3 * 32 };
    const strided_row = try logicalF32RowSlice(&strided, 3, 1, 4);
    try std.testing.expectEqual(@as(u64, 0x4000 + 32), strided_row.pointer);
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), strided_row.size);
}

test "Mxfp8LinearWeight.cublasLtScaleTensorSize computes padded VEC32 layout bytes" {
    try std.testing.expectEqual(@as(usize, 80 * 128), Mxfp8LinearWeight.cublasLtScaleTensorSize(2560, 8));
    try std.testing.expectEqual(@as(usize, 4 * 128), Mxfp8LinearWeight.cublasLtScaleTensorSize(48, 1));
}

test "Mxfp8LinearWeight.roundoff rounds up to granularity" {
    try std.testing.expectEqual(@as(usize, 128), Mxfp8LinearWeight.roundoff(1, 128));
    try std.testing.expectEqual(@as(usize, 128), Mxfp8LinearWeight.roundoff(128, 128));
    try std.testing.expectEqual(@as(usize, 256), Mxfp8LinearWeight.roundoff(129, 128));
}

test "Nvfp4LinearWeight.cublasLtScaleTensorSize computes padded VEC16 layout bytes" {
    try std.testing.expectEqual(@as(usize, 128 * 160), Nvfp4LinearWeight.cublasLtScaleTensorSize(2560, 8));
    try std.testing.expectEqual(@as(usize, 128 * 4), Nvfp4LinearWeight.cublasLtScaleTensorSize(48, 1));
}

test "Nvfp4LinearWeight.roundoff rounds up to granularity" {
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(1, 128));
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(128, 128));
    try std.testing.expectEqual(@as(usize, 256), Nvfp4LinearWeight.roundoff(129, 128));
}

test "LinearWeight.rows cols and byteSize report variant dimensions and bytes" {
    const buffer = device_mod.Buffer{ .pointer = 0x5000, .size = 96 };
    const scales = device_mod.Buffer{ .pointer = 0x6000, .size = 24 };
    const biases = device_mod.Buffer{ .pointer = 0x7000, .size = 24 };
    const cache = device_mod.Buffer{ .pointer = 0x8000, .size = 48 };
    const mean = device_mod.Buffer{ .pointer = 0x9000, .size = 12 };
    const weights = [_]LinearWeight{
        .{ .dense_f32 = .{ .rows = 2, .cols = 3, .buffer = buffer } },
        .{ .dense_u16 = .{ .rows = 4, .cols = 5, .buffer = buffer, .dtype = .bf16 } },
        .{ .gaffine_u4 = .{
            .rows = 6,
            .cols = 7,
            .packed_data = buffer,
            .scales = scales,
            .biases = biases,
            .group_size = 8,
            .scales_dtype_tag = 1,
            .dequant_i8_cache = cache,
            .mean_scale_cache = mean,
        } },
        .{ .fp8 = .{
            .rows = 8,
            .cols = 9,
            .buffer = buffer,
            .scales_buffer = scales,
            .scale_rows = 1,
            .scale_cols = 1,
            .block_size = 128,
            .weight_scale_inv = 1.0,
        } },
    };

    try std.testing.expectEqual(@as(usize, 2), weights[0].rows());
    try std.testing.expectEqual(@as(usize, 3), weights[0].cols());
    try std.testing.expectEqual(@as(usize, 96), weights[0].byteSize());
    try std.testing.expectEqual(@as(usize, 4), weights[1].rows());
    try std.testing.expectEqual(@as(usize, 5), weights[1].cols());
    try std.testing.expectEqual(@as(usize, 96), weights[1].byteSize());
    try std.testing.expectEqual(@as(usize, 6), weights[2].rows());
    try std.testing.expectEqual(@as(usize, 7), weights[2].cols());
    try std.testing.expectEqual(@as(usize, 204), weights[2].byteSize());
    try std.testing.expectEqual(@as(usize, 8), weights[3].rows());
    try std.testing.expectEqual(@as(usize, 9), weights[3].cols());
    try std.testing.expectEqual(@as(usize, 120), weights[3].byteSize());
}

test "linearWeightHasI8Cache detects complete cached INT8 paths" {
    const empty = device_mod.Buffer{ .pointer = 0, .size = 0 };
    const present = device_mod.Buffer{ .pointer = 0xA000, .size = 16 };
    var cached = LinearWeight{ .nvfp4 = .{
        .rows = 2,
        .cols = 3,
        .buffer = empty,
        .scales_buffer = empty,
        .scales_lt_buffer = empty,
        .packed_cols = 1,
        .scale_cols = 1,
        .group_size = 16,
        .weight_global_scale = 1.0,
        .dequant_i8_cache = present,
        .mean_scale_cache = present,
    } };
    var missing_scale = LinearWeight{ .gaffine_u8 = .{
        .rows = 2,
        .cols = 3,
        .packed_data = empty,
        .scales = empty,
        .biases = empty,
        .group_size = 16,
        .scales_dtype_tag = 1,
        .dequant_i8_cache = present,
    } };
    var dense = LinearWeight{ .dense_f32 = .{ .rows = 2, .cols = 3, .buffer = present } };

    try std.testing.expect(linearWeightHasI8Cache(&cached));
    try std.testing.expect(!linearWeightHasI8Cache(&missing_scale));
    try std.testing.expect(!linearWeightHasI8Cache(&dense));
}
