//! Allocation-free tensor metadata and byte-span validation helpers.
//!
//! `Tensor.data_size` is physical storage bytes. Dense logical byte counts and
//! strided physical byte spans are separate contracts so callers cannot
//! accidentally treat non-contiguous storage as dense.

const std = @import("std");
const device_mod = @import("device.zig");
const dtype_mod = @import("dtype.zig");
const tensor_mod = @import("tensor.zig");

pub const max_rank = tensor_mod.MAX_NDIM;
pub const DType = dtype_mod.DType;
pub const Device = device_mod.Device;
pub const DeviceType = device_mod.DeviceType;

pub const Layout = enum(u8) {
    row_major_contiguous,
    strided,
    opaque_backend,
};

pub const TensorDescriptor = struct {
    dtype: DType,
    device: Device,
    rank: u8,
    shape: [max_rank]i64,
    strides: [max_rank]i64,
    data_size: ?usize = null,
    layout: Layout,
};

pub fn fromTensor(tensor: *const tensor_mod.Tensor) !TensorDescriptor {
    const tensor_rank = std.math.cast(u8, tensor.n_dims) orelse return error.InvalidRank;
    var desc = TensorDescriptor{
        .dtype = tensor.dtype,
        .device = try convertTensorDevice(tensor.device),
        .rank = tensor_rank,
        .shape = tensor.shape,
        .strides = tensor.strides,
        .data_size = tensor.data_size,
        .layout = .strided,
    };
    try validateMetadata(&desc);
    desc.layout = if (try isRowMajorContiguous(&desc)) .row_major_contiguous else .strided;
    return desc;
}

pub fn activeShape(desc: *const TensorDescriptor) []const i64 {
    return desc.shape[0..@as(usize, desc.rank)];
}

pub fn activeStrides(desc: *const TensorDescriptor) []const i64 {
    return desc.strides[0..@as(usize, desc.rank)];
}

pub fn validateMetadata(desc: *const TensorDescriptor) !void {
    if (desc.rank == 0 or desc.rank > max_rank) return error.InvalidRank;

    const active_rank: usize = desc.rank;
    for (0..active_rank) |dim_idx| {
        if (desc.shape[dim_idx] <= 0) return error.InvalidShape;
        if (desc.layout != .opaque_backend and desc.strides[dim_idx] <= 0) return error.InvalidStride;
        if (desc.layout == .opaque_backend and desc.strides[dim_idx] < 0) return error.InvalidStride;
    }

    for (active_rank..max_rank) |dim_idx| {
        if (desc.shape[dim_idx] != 0) return error.InvalidShape;
        if (desc.strides[dim_idx] != 0) return error.InvalidStride;
    }
}

pub fn rowMajorStrides(shape: []const i64, out: *[max_rank]i64) !void {
    if (shape.len == 0 or shape.len > max_rank) return error.InvalidRank;
    out.* = .{0} ** max_rank;

    var expected_stride: i64 = 1;
    var dim_idx = shape.len;
    while (dim_idx > 0) {
        dim_idx -= 1;
        const dim = shape[dim_idx];
        if (dim <= 0) return error.InvalidShape;
        out[dim_idx] = expected_stride;
        expected_stride = std.math.mul(i64, expected_stride, dim) catch return error.ByteCountOverflow;
    }
}

pub fn logicalElementCount(desc: *const TensorDescriptor) !usize {
    try validateMetadata(desc);
    var count: usize = 1;
    for (activeShape(desc)) |dim| {
        const dim_count = std.math.cast(usize, dim) orelse return error.InvalidShape;
        count = std.math.mul(usize, count, dim_count) catch return error.ByteCountOverflow;
    }
    return count;
}

pub fn denseByteCount(desc: *const TensorDescriptor) !usize {
    const elem_size = desc.dtype.checkedDenseElementSize() catch return error.UnsupportedDType;
    const count = try logicalElementCount(desc);
    return std.math.mul(usize, count, elem_size) catch error.ByteCountOverflow;
}

pub fn physicalStorageByteSpan(desc: *const TensorDescriptor) !usize {
    try validateGenericLayout(desc);
    const elem_size = desc.dtype.checkedDenseElementSize() catch return error.UnsupportedDType;

    var max_element_offset: usize = 0;
    const active_rank: usize = desc.rank;
    for (0..active_rank) |dim_idx| {
        const dim_extent = std.math.cast(usize, desc.shape[dim_idx] - 1) orelse return error.InvalidShape;
        const stride = std.math.cast(usize, desc.strides[dim_idx]) orelse return error.InvalidStride;
        const contribution = std.math.mul(usize, dim_extent, stride) catch return error.ByteCountOverflow;
        max_element_offset = std.math.add(usize, max_element_offset, contribution) catch return error.ByteCountOverflow;
    }

    const span_elems = std.math.add(usize, max_element_offset, 1) catch return error.ByteCountOverflow;
    return std.math.mul(usize, span_elems, elem_size) catch error.ByteCountOverflow;
}

pub fn declaredDataSize(desc: *const TensorDescriptor) !usize {
    return desc.data_size orelse error.BufferTooSmall;
}

pub fn isRowMajorContiguous(desc: *const TensorDescriptor) !bool {
    try validateMetadata(desc);
    if (desc.layout == .opaque_backend) return false;

    var expected: [max_rank]i64 = undefined;
    try rowMajorStrides(activeShape(desc), &expected);
    return std.mem.eql(i64, activeStrides(desc), expected[0..@as(usize, desc.rank)]);
}

pub fn validateGenericLayout(desc: *const TensorDescriptor) !void {
    try validateMetadata(desc);
    switch (desc.layout) {
        .opaque_backend => return error.UnsupportedLayout,
        .row_major_contiguous => {
            if (!(try isRowMajorContiguous(desc))) return error.InvalidStride;
        },
        .strided => {},
    }
}

pub fn validateDenseBuffer(desc: *const TensorDescriptor) !usize {
    try validateGenericLayout(desc);
    if (desc.layout != .row_major_contiguous) return error.UnsupportedLayout;
    const required_bytes = try denseByteCount(desc);
    const available_bytes = try declaredDataSize(desc);
    if (available_bytes < required_bytes) return error.BufferTooSmall;
    return required_bytes;
}

pub fn validatePhysicalBuffer(desc: *const TensorDescriptor) !usize {
    try validateGenericLayout(desc);
    const required_bytes = try physicalStorageByteSpan(desc);
    const available_bytes = try declaredDataSize(desc);
    if (available_bytes < required_bytes) return error.BufferTooSmall;
    return required_bytes;
}

pub fn validateDeclaredPhysicalBytes(desc: *const TensorDescriptor) !usize {
    try validateGenericLayout(desc);
    const available_bytes = try declaredDataSize(desc);
    if (available_bytes == 0) return error.BufferTooSmall;
    return available_bytes;
}

fn convertTensorDevice(device: tensor_mod.Device) !Device {
    return switch (device.device_type) {
        .kDLCPU => Device.cpu(),
        .kDLCUDA => Device.cuda(device.device_id),
        .kDLMetal => Device.metal(device.device_id),
        else => error.UnsupportedDevice,
    };
}

fn descriptor(
    dtype: DType,
    layout: Layout,
    rank: u8,
    shape_values: [max_rank]i64,
    stride_values: [max_rank]i64,
    data_size: ?usize,
) TensorDescriptor {
    return .{
        .dtype = dtype,
        .device = Device.cpu(),
        .rank = rank,
        .shape = shape_values,
        .strides = stride_values,
        .data_size = data_size,
        .layout = layout,
    };
}

test "compute rowMajorStrides computes contiguous strides" {
    var strides: [max_rank]i64 = undefined;
    try rowMajorStrides(&.{ 2, 3, 4 }, &strides);
    try std.testing.expectEqualSlices(i64, &.{ 12, 4, 1 }, strides[0..3]);
    try std.testing.expectEqual(@as(i64, 0), strides[3]);
}

test "compute activeShape and activeStrides expose rank metadata" {
    const desc = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, activeShape(&desc));
    try std.testing.expectEqualSlices(i64, &.{ 3, 1 }, activeStrides(&desc));
}

test "compute validateMetadata rejects invalid rank" {
    const desc = descriptor(.f32, .row_major_contiguous, 0, .{0} ** max_rank, .{0} ** max_rank, 0);
    try std.testing.expectError(error.InvalidRank, validateMetadata(&desc));
}

test "compute validateMetadata rejects zero and negative dimensions" {
    const zero_dim = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 0, 0, 0, 0, 0, 0, 0 }, .{ 1, 1, 0, 0, 0, 0, 0, 0 }, 8);
    try std.testing.expectError(error.InvalidShape, validateMetadata(&zero_dim));

    const negative_dim = descriptor(.f32, .row_major_contiguous, 2, .{ 2, -1, 0, 0, 0, 0, 0, 0 }, .{ 1, 1, 0, 0, 0, 0, 0, 0 }, 8);
    try std.testing.expectError(error.InvalidShape, validateMetadata(&negative_dim));
}

test "compute validateMetadata rejects negative stride and inactive metadata" {
    const negative_stride = descriptor(.f32, .strided, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, -1, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectError(error.InvalidStride, validateMetadata(&negative_stride));

    const inactive_shape = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 1, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectError(error.InvalidShape, validateMetadata(&inactive_shape));

    const inactive_stride = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 1, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectError(error.InvalidStride, validateMetadata(&inactive_stride));
}

test "compute logicalElementCount returns checked product" {
    const desc = descriptor(.f32, .row_major_contiguous, 3, .{ 2, 3, 4, 0, 0, 0, 0, 0 }, .{ 12, 4, 1, 0, 0, 0, 0, 0 }, 96);
    try std.testing.expectEqual(@as(usize, 24), try logicalElementCount(&desc));
}

test "compute logicalElementCount rejects multiplication overflow" {
    const dim = std.math.maxInt(i64);
    const desc = descriptor(.u8, .strided, 2, .{ dim, dim, 0, 0, 0, 0, 0, 0 }, .{ dim, 1, 0, 0, 0, 0, 0, 0 }, null);
    try std.testing.expectError(error.ByteCountOverflow, logicalElementCount(&desc));
}

test "compute denseByteCount computes dense logical bytes" {
    const desc = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectEqual(@as(usize, 24), try denseByteCount(&desc));
}

test "compute denseByteCount rejects block quantized dtype" {
    const desc = descriptor(.grouped_affine_u4, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 16);
    try std.testing.expectError(error.UnsupportedDType, denseByteCount(&desc));
}

test "compute isRowMajorContiguous detects non-contiguous stride" {
    const contiguous = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expect(try isRowMajorContiguous(&contiguous));

    const strided = descriptor(.f32, .strided, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 5, 1, 0, 0, 0, 0, 0, 0 }, 32);
    try std.testing.expect(!(try isRowMajorContiguous(&strided)));
}

test "compute physicalStorageByteSpan separates strided span from dense bytes" {
    const desc = descriptor(.f32, .strided, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 5, 1, 0, 0, 0, 0, 0, 0 }, 32);
    try std.testing.expectEqual(@as(usize, 24), try denseByteCount(&desc));
    try std.testing.expectEqual(@as(usize, 32), try physicalStorageByteSpan(&desc));
}

test "compute physicalStorageByteSpan rejects byte-count overflow" {
    const dim = std.math.maxInt(i64);
    const desc = descriptor(.u64, .strided, 1, .{ dim, 0, 0, 0, 0, 0, 0, 0 }, .{ dim, 0, 0, 0, 0, 0, 0, 0 }, null);
    try std.testing.expectError(error.ByteCountOverflow, physicalStorageByteSpan(&desc));
}

test "compute declaredDataSize returns declared physical bytes" {
    const desc = descriptor(.f32, .row_major_contiguous, 1, .{ 4, 0, 0, 0, 0, 0, 0, 0 }, .{ 1, 0, 0, 0, 0, 0, 0, 0 }, 16);
    try std.testing.expectEqual(@as(usize, 16), try declaredDataSize(&desc));
}

test "compute validateGenericLayout rejects opaque layout" {
    const desc = descriptor(.f32, .opaque_backend, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 0, 0, 0, 0, 0, 0, 0, 0 }, 24);
    try std.testing.expectError(error.UnsupportedLayout, validateGenericLayout(&desc));
}

test "compute validateDenseBuffer rejects insufficient data_size and strided layout" {
    const small = descriptor(.f32, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 20);
    try std.testing.expectError(error.BufferTooSmall, validateDenseBuffer(&small));

    const strided = descriptor(.f32, .strided, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 5, 1, 0, 0, 0, 0, 0, 0 }, 32);
    try std.testing.expectError(error.UnsupportedLayout, validateDenseBuffer(&strided));
}

test "compute validatePhysicalBuffer accepts strided physical span" {
    const desc = descriptor(.f32, .strided, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 5, 1, 0, 0, 0, 0, 0, 0 }, 32);
    try std.testing.expectEqual(@as(usize, 32), try validatePhysicalBuffer(&desc));
}

test "compute validateDeclaredPhysicalBytes accepts block quantized declared storage" {
    const desc = descriptor(.grouped_affine_u4, .row_major_contiguous, 2, .{ 2, 3, 0, 0, 0, 0, 0, 0 }, .{ 3, 1, 0, 0, 0, 0, 0, 0 }, 17);
    try std.testing.expectEqual(@as(usize, 17), try validateDeclaredPhysicalBytes(&desc));
}

test "compute fromTensor builds descriptor from Tensor metadata" {
    var storage = [_]u8{0} ** 24;
    const tensor = tensor_mod.Tensor.view(storage[0..].ptr, &.{ 2, 3 }, .f32, null);
    const desc = try fromTensor(&tensor);
    try std.testing.expectEqual(Layout.row_major_contiguous, desc.layout);
    try std.testing.expectEqual(@as(usize, 24), try validateDenseBuffer(&desc));
}
