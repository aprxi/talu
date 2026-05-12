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
    input_dtypes: []const DType,
    output_dtypes: []const DType,
    layouts: []const Layout,
    rank_limits: RankLimits = .{},
    required_alignment: usize = 1,
};

pub const PrimitiveQuery = struct {
    backend: Backend,
    name: []const u8,
    input_dtype: DType,
    output_dtype: DType,
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

pub const RawCopyCapability = struct {
    backend: Backend,
    direction: CopyDirection,
    required_alignment: usize = 1,
};

pub const CopyQuery = struct {
    backend: Backend,
    direction: CopyDirection,
    dtype: DType,
    layout: Layout,
    alignment_bytes: ?usize = null,
};

pub const RawCopyQuery = struct {
    backend: Backend,
    src_device: Device,
    dst_device: Device,
    byte_count: usize,
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

pub fn inferCopyDirection(src: Device, dst: Device) !CopyDirection {
    const src_backend = try backendFromDevice(src);
    const dst_backend = try backendFromDevice(dst);

    if (src_backend == .cpu and dst_backend == .cpu) return .host_to_host;
    if (src_backend == .cpu and dst_backend != .cpu) return .host_to_device;
    if (src_backend != .cpu and dst_backend == .cpu) return .device_to_host;
    if (src_backend == dst_backend and src.device_id == dst.device_id) return .device_to_device;
    return .peer_device_to_device;
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
    var input_dtype_seen = false;
    var output_dtype_seen = false;
    var layout_seen = false;

    for (entries) |entry| {
        if (entry.backend != query.backend) continue;
        backend_seen = true;
        if (!std.mem.eql(u8, entry.name, query.name)) continue;
        primitive_seen = true;
        if (!containsDType(entry.input_dtypes, query.input_dtype)) continue;
        input_dtype_seen = true;
        if (!containsDType(entry.output_dtypes, query.output_dtype)) continue;
        output_dtype_seen = true;
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
    if (!input_dtype_seen or !output_dtype_seen) return error.UnsupportedDType;
    if (!layout_seen) return error.UnsupportedLayout;
    return error.UnsupportedPrimitive;
}

pub fn supportsCopy(entries: []const CopyCapability, query: CopyQuery) bool {
    validateCopy(entries, query) catch return false;
    return true;
}

pub fn supportsRawCopy(entries: []const RawCopyCapability, query: RawCopyQuery) bool {
    validateRawCopy(entries, query) catch return false;
    return true;
}

pub fn validateRawCopy(entries: []const RawCopyCapability, query: RawCopyQuery) !void {
    if (query.byte_count == 0) return error.InvalidShape;

    const direction = try inferCopyDirection(query.src_device, query.dst_device);
    try validateRawCopyDevices(query.backend, query.src_device, query.dst_device, direction);

    var backend_seen = false;
    var direction_seen = false;

    for (entries) |entry| {
        if (entry.backend != query.backend) continue;
        backend_seen = true;
        if (entry.direction != direction) continue;
        direction_seen = true;

        if (query.alignment_bytes) |alignment_bytes| {
            if (alignment_bytes < entry.required_alignment) return error.AlignmentMismatch;
        }
        return;
    }

    if (!backend_seen) return error.UnsupportedDevice;
    if (!direction_seen) return error.UnsupportedCopyDirection;
    return error.UnsupportedCopyDirection;
}

fn validateRawCopyDevices(backend: Backend, src: Device, dst: Device, direction: CopyDirection) !void {
    const src_backend = try backendFromDevice(src);
    const dst_backend = try backendFromDevice(dst);

    switch (backend) {
        .cpu => {
            if (src_backend == .cpu and dst_backend == .cpu and direction == .host_to_host) return;
            return error.UnsupportedDevice;
        },
        .cuda => switch (direction) {
            .host_to_device => if (src_backend == .cpu and dst_backend == .cuda) return else return error.UnsupportedDevice,
            .device_to_host => if (src_backend == .cuda and dst_backend == .cpu) return else return error.UnsupportedDevice,
            .device_to_device => if (src_backend == .cuda and dst_backend == .cuda and src.device_id == dst.device_id) return else return error.UnsupportedDevice,
            .peer_device_to_device => if (src_backend == .cuda and dst_backend == .cuda and src.device_id != dst.device_id) return else return error.UnsupportedDevice,
            else => return error.UnsupportedCopyDirection,
        },
        .metal => switch (direction) {
            .host_to_device => if (src_backend == .cpu and dst_backend == .metal) return else return error.UnsupportedDevice,
            .device_to_host => if (src_backend == .metal and dst_backend == .cpu) return else return error.UnsupportedDevice,
            .device_to_device, .peer_device_to_device => if (src_backend == .metal or dst_backend == .metal) return error.UnsupportedCopyDirection else return error.UnsupportedDevice,
            else => return error.UnsupportedCopyDirection,
        },
    }
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
        .input_dtypes = &test_f32_dtypes,
        .output_dtypes = &test_f32_dtypes,
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

const test_raw_copies = [_]RawCopyCapability{
    .{
        .backend = .cuda,
        .direction = .host_to_device,
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

test "compute inferCopyDirection classifies device pairs" {
    try std.testing.expectEqual(CopyDirection.host_to_host, try inferCopyDirection(Device.cpu(), Device.cpu()));
    try std.testing.expectEqual(CopyDirection.host_to_device, try inferCopyDirection(Device.cpu(), Device.cuda(0)));
    try std.testing.expectEqual(CopyDirection.device_to_host, try inferCopyDirection(Device.cuda(0), Device.cpu()));
    try std.testing.expectEqual(CopyDirection.device_to_device, try inferCopyDirection(Device.cuda(0), Device.cuda(0)));
    try std.testing.expectEqual(CopyDirection.peer_device_to_device, try inferCopyDirection(Device.cuda(0), Device.cuda(1)));
}

test "compute hasPrimitiveEntry returns false for unknown primitive" {
    try std.testing.expect(hasPrimitiveEntry(&test_primitives, .cpu, "add"));
    try std.testing.expect(!hasPrimitiveEntry(&test_primitives, .cpu, "missing"));
}

test "compute supportsPrimitive and validatePrimitive check dtype layout rank alignment" {
    try validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f32,
        .layout = .row_major_contiguous,
        .rank = 2,
        .alignment_bytes = 16,
    });
    try std.testing.expect(supportsPrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedDType, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f16,
        .output_dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedDType, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f16,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedLayout, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f32,
        .layout = .strided,
    }));
    try std.testing.expectError(error.InvalidRank, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f32,
        .layout = .row_major_contiguous,
        .rank = 5,
    }));
    try std.testing.expectError(error.AlignmentMismatch, validatePrimitive(&test_primitives, .{
        .backend = .cpu,
        .name = "add",
        .input_dtype = .f32,
        .output_dtype = .f32,
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

test "compute supportsRawCopy and validateRawCopy are separate from typed copy" {
    try validateRawCopy(&test_raw_copies, .{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 16,
        .alignment_bytes = 4,
    });
    try std.testing.expect(supportsRawCopy(&test_raw_copies, .{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 16,
    }));
    try std.testing.expectError(error.InvalidShape, validateRawCopy(&test_raw_copies, .{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 0,
    }));
    try std.testing.expectError(error.AlignmentMismatch, validateRawCopy(&test_raw_copies, .{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.cuda(0),
        .byte_count = 16,
        .alignment_bytes = 2,
    }));
    try std.testing.expectError(error.UnsupportedDevice, validateRawCopy(&test_raw_copies, .{
        .backend = .cuda,
        .src_device = Device.cpu(),
        .dst_device = Device.metal(0),
        .byte_count = 16,
    }));
    try std.testing.expect(!supportsCopy(&[_]CopyCapability{}, .{
        .backend = .cuda,
        .direction = .host_to_device,
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
