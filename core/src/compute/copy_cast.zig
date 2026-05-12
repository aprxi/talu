//! Shared copy and cast contract validators for compute backends.
//!
//! These helpers validate sizes, layouts, dtypes, directions, and alignment
//! before callers mutate destination payloads or kernel argument state.

const std = @import("std");
const capability = @import("capability.zig");
const device_mod = @import("device.zig");
const dtype_mod = @import("dtype.zig");
const tensor_desc = @import("tensor_desc.zig");

const cpu_capabilities = @import("cpu/capabilities.zig");
const cuda_capabilities = @import("cuda/capabilities.zig");
const metal_capabilities = @import("metal/capabilities.zig");

pub const Backend = capability.Backend;
pub const CopyDirection = capability.CopyDirection;
pub const DType = dtype_mod.DType;
pub const Device = device_mod.Device;
pub const Layout = tensor_desc.Layout;

pub const CopyBufferSpec = struct {
    backend: Backend,
    direction: CopyDirection,
    dtype: DType,
    layout: Layout,
    element_count: usize,
    src_size: usize,
    dst_size: usize,
    src_address: ?usize = null,
    dst_address: ?usize = null,
};

pub const RawCopyBufferSpec = struct {
    backend: Backend,
    src_device: Device,
    dst_device: Device,
    byte_count: usize,
    src_size: usize,
    dst_size: usize,
    src_address: ?usize = null,
    dst_address: ?usize = null,
};

pub const CastBufferSpec = struct {
    backend: Backend,
    src_dtype: DType,
    dst_dtype: DType,
    layout: Layout,
    element_count: usize,
    src_size: usize,
    dst_size: usize,
    src_address: ?usize = null,
    dst_address: ?usize = null,
};

pub fn inferCopyDirection(src: Device, dst: Device) !CopyDirection {
    return capability.inferCopyDirection(src, dst);
}

pub fn validateRawCopyBuffers(spec: RawCopyBufferSpec) !usize {
    if (spec.byte_count == 0) return error.InvalidShape;

    const alignment_bytes = try minimumAlignment(spec.src_address, spec.dst_address);
    try capability.validateRawCopy(rawCopyCapabilities(spec.backend), .{
        .backend = spec.backend,
        .src_device = spec.src_device,
        .dst_device = spec.dst_device,
        .byte_count = spec.byte_count,
        .alignment_bytes = alignment_bytes,
    });

    if (spec.src_size < spec.byte_count or spec.dst_size < spec.byte_count) return error.BufferTooSmall;
    return spec.byte_count;
}

pub fn validateCopyBuffers(spec: CopyBufferSpec) !usize {
    if (spec.element_count == 0) return error.InvalidShape;
    if (spec.layout == .opaque_backend) return error.UnsupportedLayout;

    const alignment_bytes = try minimumAlignment(spec.src_address, spec.dst_address);
    try capability.validateCopy(copyCapabilities(spec.backend), .{
        .backend = spec.backend,
        .direction = spec.direction,
        .dtype = spec.dtype,
        .layout = spec.layout,
        .alignment_bytes = alignment_bytes,
    });

    const elem_size = spec.dtype.checkedDenseElementSize() catch return error.UnsupportedDType;
    const required_bytes = std.math.mul(usize, spec.element_count, elem_size) catch return error.ByteCountOverflow;
    if (spec.src_size < required_bytes or spec.dst_size < required_bytes) return error.BufferTooSmall;
    return required_bytes;
}

pub fn validateCastBuffers(spec: CastBufferSpec) !struct { src_bytes: usize, dst_bytes: usize } {
    if (spec.element_count == 0) return error.InvalidShape;
    if (spec.layout == .opaque_backend) return error.UnsupportedLayout;

    const alignment_bytes = try minimumAlignment(spec.src_address, spec.dst_address);
    try capability.validateCast(castCapabilities(spec.backend), .{
        .backend = spec.backend,
        .src_dtype = spec.src_dtype,
        .dst_dtype = spec.dst_dtype,
        .layout = spec.layout,
        .alignment_bytes = alignment_bytes,
    });

    const src_elem_size = spec.src_dtype.checkedDenseElementSize() catch return error.UnsupportedDType;
    const dst_elem_size = spec.dst_dtype.checkedDenseElementSize() catch return error.UnsupportedDType;
    const src_bytes = std.math.mul(usize, spec.element_count, src_elem_size) catch return error.ByteCountOverflow;
    const dst_bytes = std.math.mul(usize, spec.element_count, dst_elem_size) catch return error.ByteCountOverflow;
    if (spec.src_size < src_bytes or spec.dst_size < dst_bytes) return error.BufferTooSmall;
    return .{ .src_bytes = src_bytes, .dst_bytes = dst_bytes };
}

fn rawCopyCapabilities(backend: Backend) []const capability.RawCopyCapability {
    return switch (backend) {
        .cpu => &cpu_capabilities.raw_copy_capabilities,
        .cuda => &cuda_capabilities.raw_copy_capabilities,
        .metal => &metal_capabilities.raw_copy_capabilities,
    };
}

fn copyCapabilities(backend: Backend) []const capability.CopyCapability {
    return switch (backend) {
        .cpu => &cpu_capabilities.copy_capabilities,
        .cuda => &cuda_capabilities.copy_capabilities,
        .metal => &metal_capabilities.copy_capabilities,
    };
}

fn castCapabilities(backend: Backend) []const capability.CastCapability {
    return switch (backend) {
        .cpu => &cpu_capabilities.cast_capabilities,
        .cuda => &cuda_capabilities.cast_capabilities,
        .metal => &metal_capabilities.cast_capabilities,
    };
}

fn minimumAlignment(src_address: ?usize, dst_address: ?usize) !?usize {
    if (src_address == null and dst_address == null) return null;
    var min_alignment: usize = std.math.maxInt(usize);
    if (src_address) |address| min_alignment = @min(min_alignment, try addressAlignment(address));
    if (dst_address) |address| min_alignment = @min(min_alignment, try addressAlignment(address));
    return min_alignment;
}

fn addressAlignment(address: usize) !usize {
    if (address == 0) return error.AlignmentMismatch;
    return address & (~address +% 1);
}

test "compute inferCopyDirection classifies supported device pairs" {
    try std.testing.expectEqual(CopyDirection.host_to_host, try inferCopyDirection(Device.cpu(), Device.cpu()));
    try std.testing.expectEqual(CopyDirection.host_to_device, try inferCopyDirection(Device.cpu(), Device.cuda(0)));
    try std.testing.expectEqual(CopyDirection.device_to_host, try inferCopyDirection(Device.cuda(0), Device.cpu()));
    try std.testing.expectEqual(CopyDirection.device_to_device, try inferCopyDirection(Device.cuda(0), Device.cuda(0)));
    try std.testing.expectEqual(CopyDirection.peer_device_to_device, try inferCopyDirection(Device.cuda(0), Device.cuda(1)));
}

test "compute validateCopyBuffers returns required bytes and rejects small buffers" {
    const required = try validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
    });
    try std.testing.expectEqual(@as(usize, 16), required);

    try std.testing.expectError(error.BufferTooSmall, validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 12,
    }));
}

test "compute validateRawCopyBuffers validates byte movement separately from typed copy" {
    const required = try validateRawCopyBuffers(.{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 16,
        .src_size = 16,
        .dst_size = 16,
    });
    try std.testing.expectEqual(@as(usize, 16), required);

    const peer_required = try validateRawCopyBuffers(.{
        .backend = .cuda,
        .src_device = Device.cuda(0),
        .dst_device = Device.cuda(1),
        .byte_count = 16,
        .src_size = 16,
        .dst_size = 16,
    });
    try std.testing.expectEqual(@as(usize, 16), peer_required);

    try std.testing.expectError(error.InvalidShape, validateRawCopyBuffers(.{
        .backend = .cuda,
        .src_device = Device.cuda(0),
        .dst_device = Device.cuda(1),
        .byte_count = 0,
        .src_size = 16,
        .dst_size = 16,
        .src_address = 0,
        .dst_address = 0,
    }));

    try std.testing.expectError(error.BufferTooSmall, validateRawCopyBuffers(.{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 16,
        .src_size = 12,
        .dst_size = 16,
    }));

    try std.testing.expectError(error.UnsupportedCopyDirection, validateRawCopyBuffers(.{
        .backend = .metal,
        .src_device = Device.metal(0),
        .dst_device = Device.metal(0),
        .byte_count = 16,
        .src_size = 16,
        .dst_size = 16,
    }));

    try std.testing.expectError(error.UnsupportedCopyDirection, validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .host_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
    }));
}

test "compute validateCopyBuffers rejects unsupported copy direction layout and alignment" {
    try std.testing.expectError(error.UnsupportedCopyDirection, validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
    }));

    try std.testing.expectError(error.UnsupportedLayout, validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .opaque_backend,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
    }));

    try std.testing.expectError(error.AlignmentMismatch, validateCopyBuffers(.{
        .backend = .cuda,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
        .src_address = 0x1002,
        .dst_address = 0x2000,
    }));
}

test "compute validateCopyBuffers rejects block quantized dense copies" {
    try std.testing.expectError(error.UnsupportedDType, validateCopyBuffers(.{
        .backend = .cpu,
        .direction = .host_to_host,
        .dtype = .grouped_affine_u4,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 16,
    }));
}

test "compute validateCastBuffers returns source and destination bytes" {
    const bytes = try validateCastBuffers(.{
        .backend = .cuda,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 16,
        .dst_size = 8,
    });
    try std.testing.expectEqual(@as(usize, 16), bytes.src_bytes);
    try std.testing.expectEqual(@as(usize, 8), bytes.dst_bytes);
}

test "compute validateCastBuffers rejects unsupported dtype pair before size checks" {
    try std.testing.expectError(error.UnsupportedCast, validateCastBuffers(.{
        .backend = .cuda,
        .src_dtype = .f16,
        .dst_dtype = .bf16,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 8,
        .dst_size = 8,
    }));
}

test "compute validateCastBuffers rejects too-small destination buffer" {
    try std.testing.expectError(error.BufferTooSmall, validateCastBuffers(.{
        .backend = .cuda,
        .src_dtype = .bf16,
        .dst_dtype = .f32,
        .layout = .row_major_contiguous,
        .element_count = 4,
        .src_size = 8,
        .dst_size = 12,
    }));
}
