//! Static backend capability query vocabulary for compute primitives.
//!
//! Capability tables are cold-path facts. Unknown combinations fail closed.

const std = @import("std");
const device_mod = @import("device.zig");
const dtype_mod = @import("dtype.zig");
const tensor_desc = @import("tensor_desc.zig");

pub const DType = dtype_mod.DType;
pub const Layout = tensor_desc.Layout;
pub const Device = device_mod.Device;
pub const DeviceType = device_mod.DeviceType;

pub const Backend = enum(u8) {
    cpu,
    cuda,
    metal,
};

pub const CopyDirection = enum(u8) {
    host_to_host,
    host_to_device,
    device_to_host,
    device_to_device,
    peer_device_to_device,
    backend_native,
};

pub const RankLimits = struct {
    min: u8 = 1,
    max: u8 = tensor_desc.max_rank,
};

pub const PrimitiveCapability = struct {
    backend: Backend,
    name: []const u8,
    dtypes: []const DType,
    layouts: []const Layout,
    rank_limits: RankLimits = .{},
    required_alignment: usize = 1,
};

pub const PrimitiveQuery = struct {
    backend: Backend,
    name: []const u8,
    dtype: DType,
    layout: Layout,
    rank: ?u8 = null,
    alignment_bytes: ?usize = null,
};

pub const CopyCapability = struct {
    backend: Backend,
    direction: CopyDirection,
    dtypes: []const DType,
    layouts: []const Layout,
    required_alignment: usize = 1,
};

pub const CopyQuery = struct {
    backend: Backend,
    direction: CopyDirection,
    dtype: DType,
    layout: Layout,
    alignment_bytes: ?usize = null,
};

pub const CastCapability = struct {
    backend: Backend,
    src_dtype: DType,
    dst_dtype: DType,
    layouts: []const Layout,
    required_alignment: usize = 1,
};

pub const CastQuery = struct {
    backend: Backend,
    src_dtype: DType,
    dst_dtype: DType,
    layout: Layout,
    alignment_bytes: ?usize = null,
};

pub fn backendFromDevice(device: Device) !Backend {
    return switch (device.device_type) {
        .CPU => .cpu,
        .CUDA => .cuda,
        .Metal => .metal,
        else => error.UnsupportedDevice,
    };
}

pub fn hasPrimitiveEntry(entries: []const PrimitiveCapability, backend: Backend, name: []const u8) bool {
    for (entries) |entry| {
        if (entry.backend == backend and std.mem.eql(u8, entry.name, name)) return true;
    }
    return false;
}

pub fn supportsPrimitive(entries: []const PrimitiveCapability, query: PrimitiveQuery) bool {
    validatePrimitive(entries, query) catch return false;
    return true;
}

pub fn validatePrimitive(entries: []const PrimitiveCapability, query: PrimitiveQuery) !void {
    var backend_seen = false;
    var primitive_seen = false;
    var dtype_seen = false;
    var layout_seen = false;

    for (entries) |entry| {
        if (entry.backend != query.backend) continue;
        backend_seen = true;
        if (!std.mem.eql(u8, entry.name, query.name)) continue;
        primitive_seen = true;
        if (!containsDType(entry.dtypes, query.dtype)) continue;
        dtype_seen = true;
        if (!containsLayout(entry.layouts, query.layout)) continue;
        layout_seen = true;

        if (query.rank) |rank| {
            if (rank < entry.rank_limits.min or rank > entry.rank_limits.max) return error.InvalidRank;
        }
        if (query.alignment_bytes) |alignment_bytes| {
            if (alignment_bytes < entry.required_alignment) return error.AlignmentMismatch;
        }
        return;
    }

    if (!backend_seen) return error.UnsupportedDevice;
    if (!primitive_seen) return error.UnsupportedPrimitive;
    if (!dtype_seen) return error.UnsupportedDType;
    if (!layout_seen) return error.UnsupportedLayout;
    return error.UnsupportedPrimitive;
}

pub fn supportsCopy(entries: []const CopyCapability, query: CopyQuery) bool {
    validateCopy(entries, query) catch return false;
    return true;
}

pub fn validateCopy(entries: []const CopyCapability, query: CopyQuery) !void {
    var backend_seen = false;
    var direction_seen = false;
    var dtype_seen = false;
    var layout_seen = false;

    for (entries) |entry| {
        if (entry.backend != query.backend) continue;
        backend_seen = true;
        if (entry.direction != query.direction) continue;
        direction_seen = true;
        if (!containsDType(entry.dtypes, query.dtype)) continue;
        dtype_seen = true;
        if (!containsLayout(entry.layouts, query.layout)) continue;
        layout_seen = true;

        if (query.alignment_bytes) |alignment_bytes| {
            if (alignment_bytes < entry.required_alignment) return error.AlignmentMismatch;
        }
        return;
    }

    if (!backend_seen) return error.UnsupportedDevice;
    if (!direction_seen) return error.UnsupportedCopyDirection;
    if (!dtype_seen) return error.UnsupportedDType;
    if (!layout_seen) return error.UnsupportedLayout;
    return error.UnsupportedCopyDirection;
}

pub fn supportsCast(entries: []const CastCapability, query: CastQuery) bool {
    validateCast(entries, query) catch return false;
    return true;
}

pub fn validateCast(entries: []const CastCapability, query: CastQuery) !void {
    var backend_seen = false;
    var cast_seen = false;
    var layout_seen = false;

    for (entries) |entry| {
        if (entry.backend != query.backend) continue;
        backend_seen = true;
        if (entry.src_dtype != query.src_dtype or entry.dst_dtype != query.dst_dtype) continue;
        cast_seen = true;
        if (!containsLayout(entry.layouts, query.layout)) continue;
        layout_seen = true;

        if (query.alignment_bytes) |alignment_bytes| {
            if (alignment_bytes < entry.required_alignment) return error.AlignmentMismatch;
        }
        return;
    }

    if (!backend_seen) return error.UnsupportedDevice;
    if (!cast_seen) return error.UnsupportedCast;
    if (!layout_seen) return error.UnsupportedLayout;
    return error.UnsupportedCast;
}

fn containsDType(values: []const DType, value: DType) bool {
    for (values) |candidate| {
        if (candidate == value) return true;
    }
    return false;
}

fn containsLayout(values: []const Layout, value: Layout) bool {
    for (values) |candidate| {
        if (candidate == value) return true;
    }
    return false;
}

const test_f32_dtypes = [_]DType{.f32};
const test_f16_dtypes = [_]DType{.f16};
const test_contiguous_layouts = [_]Layout{.row_major_contiguous};
const test_strided_layouts = [_]Layout{ .row_major_contiguous, .strided };

const test_primitives = [_]PrimitiveCapability{
    .{
        .backend = .cpu,
        .name = "add",
        .dtypes = &test_f32_dtypes,
        .layouts = &test_contiguous_layouts,
        .rank_limits = .{ .min = 1, .max = 4 },
        .required_alignment = 16,
    },
};

const test_copies = [_]CopyCapability{
    .{
        .backend = .cpu,
        .direction = .host_to_host,
        .dtypes = &test_f32_dtypes,
        .layouts = &test_contiguous_layouts,
        .required_alignment = 4,
    },
};

const test_casts = [_]CastCapability{
    .{
        .backend = .cuda,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layouts = &test_strided_layouts,
        .required_alignment = 2,
    },
};

test "compute backendFromDevice maps supported devices and rejects unknown devices" {
    try std.testing.expectEqual(Backend.cpu, try backendFromDevice(Device.cpu()));
    try std.testing.expectEqual(Backend.cuda, try backendFromDevice(Device.cuda(0)));
    try std.testing.expectEqual(Backend.metal, try backendFromDevice(Device.metal(0)));
    try std.testing.expectError(error.UnsupportedDevice, backendFromDevice(.{ .device_type = .Vulkan, .device_id = 0 }));
}

test "compute hasPrimitiveEntry returns false for unknown primitive" {
    try std.testing.expect(hasPrimitiveEntry(&test_primitives, .cpu, "add"));
    try std.testing.expect(!hasPrimitiveEntry(&test_primitives, .cpu, "missing"));
}

test "compute supportsPrimitive and validatePrimitive check dtype layout rank alignment" {
    try validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .rank = 2,
        .alignment_bytes = 16,
    });
    try std.testing.expect(supportsPrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedDType, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f16,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedLayout, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f32,
        .layout = .strided,
    }));
    try std.testing.expectError(error.InvalidRank, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .rank = 5,
    }));
    try std.testing.expectError(error.AlignmentMismatch, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .dtype = .f32,
        .layout = .row_major_contiguous,
        .alignment_bytes = 8,
    }));
}

test "compute supportsCopy and validateCopy fail closed for unsupported direction" {
    try validateCopy(&test_copies, .{
        .backend = .cpu,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    });
    try std.testing.expect(supportsCopy(&test_copies, .{
        .backend = .cpu,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedCopyDirection, validateCopy(&test_copies, .{
        .backend = .cpu,
        .direction = .device_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
}

test "compute supportsCast and validateCast fail closed for unsupported pair" {
    try validateCast(&test_casts, .{
        .backend = .cuda,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .strided,
    });
    try std.testing.expect(supportsCast(&test_casts, .{
        .backend = .cuda,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedCast, validateCast(&test_casts, .{
        .backend = .cuda,
        .src_dtype = .f16,
        .dst_dtype = .f32,
        .layout = .row_major_contiguous,
    }));
}
