//! Boundary byte-image readiness contract for staged activation payloads.
//!
//! This module validates already-described payload bytes. It never moves,
//! copies, serializes, synchronizes, downloads, uploads, or inspects payload
//! contents.

const std = @import("std");
const tensor_frame = @import("tensor_frame.zig");

pub const BoundaryByteImageContractVersion = u32;
pub const boundary_byte_image_contract_version: BoundaryByteImageContractVersion = 1;

pub const BoundaryByteImageError = tensor_frame.TensorFrameValidationError || error{
    InvalidBoundaryByteImageContractVersion,
    InvalidPayloadByteCount,
    MissingHostReadableBytes,
    HostReadableLengthMismatch,
    InvalidPayloadLocationHint,
    InvalidPayloadOwnership,
    InvalidPayloadLifetime,
    InvalidPayloadReadiness,
    RemoteByteImageUnavailable,
};

pub const BoundaryByteImageReadiness = enum(u8) {
    host_readable_now,
    producer_sync_required,
    device_download_required,
    local_only_opaque,
};

pub const BoundaryByteImageRef = struct {
    version: BoundaryByteImageContractVersion = boundary_byte_image_contract_version,
    metadata: *const tensor_frame.TensorFrameMetadata,
    byte_count: u64,
    host_bytes: ?[]const u8 = null,
    host_segments: ?[]const []const u8 = null,
    location_hint: ?tensor_frame.TensorFramePayloadLocationHint = null,
    readiness: BoundaryByteImageReadiness,
    ownership: tensor_frame.TensorFrameOwnership,
    lifetime: tensor_frame.TensorFrameLifetime,
};

pub const BoundaryByteImageValidationOptions = struct {
    require_host_readable: bool = false,
    allow_opaque_local: bool = true,
};

pub fn validateBoundaryByteImage(
    image: *const BoundaryByteImageRef,
    options: BoundaryByteImageValidationOptions,
) BoundaryByteImageError!void {
    if (image.version != boundary_byte_image_contract_version) {
        return error.InvalidBoundaryByteImageContractVersion;
    }

    try image.metadata.validate();

    if (image.byte_count == 0 or image.byte_count != image.metadata.payload.byte_count) {
        return error.InvalidPayloadByteCount;
    }

    if (!locationHintEql(image.location_hint, image.metadata.payload.location_hint)) {
        return error.InvalidPayloadLocationHint;
    }

    if (image.ownership != image.metadata.payload.ownership) {
        return error.InvalidPayloadOwnership;
    }

    if (image.lifetime != image.metadata.payload.lifetime) {
        return error.InvalidPayloadLifetime;
    }

    try validateHostReadableBytes(image);

    try validateReadinessLocationMatrix(image);

    if (!options.allow_opaque_local and image.readiness == .local_only_opaque) {
        return error.RemoteByteImageUnavailable;
    }

    if (options.require_host_readable and image.readiness != .host_readable_now) {
        return error.RemoteByteImageUnavailable;
    }

    if (options.require_host_readable and image.readiness == .host_readable_now and !hasHostReadableBytes(image)) {
        return error.MissingHostReadableBytes;
    }
}

pub fn boundaryByteImageIsRemoteReadable(
    image: *const BoundaryByteImageRef,
) bool {
    validateBoundaryByteImage(image, .{
        .require_host_readable = true,
        .allow_opaque_local = false,
    }) catch return false;
    return true;
}

fn validateReadinessLocationMatrix(image: *const BoundaryByteImageRef) BoundaryByteImageError!void {
    switch (image.readiness) {
        .host_readable_now => {
            if (!hasHostReadableBytes(image)) return error.MissingHostReadableBytes;
            if (!locationHintIsNullOrCpu(image.location_hint)) return error.InvalidPayloadReadiness;
        },
        .producer_sync_required => {
            if (hasHostReadableBytes(image)) return error.InvalidPayloadReadiness;
            if (!locationHintIsNullOrCpu(image.location_hint)) return error.InvalidPayloadReadiness;
        },
        .device_download_required => {
            if (hasHostReadableBytes(image)) return error.InvalidPayloadReadiness;
            if (!locationHintIsCudaOrMetal(image.location_hint)) return error.InvalidPayloadReadiness;
        },
        .local_only_opaque => {
            if (hasHostReadableBytes(image)) return error.InvalidPayloadReadiness;
            if (!locationHintIsOpaqueLocal(image.location_hint)) return error.InvalidPayloadReadiness;
        },
    }
}

fn validateHostReadableBytes(image: *const BoundaryByteImageRef) BoundaryByteImageError!void {
    if (image.host_bytes != null and image.host_segments != null) {
        return error.InvalidPayloadReadiness;
    }

    if (image.host_bytes) |host_bytes| {
        const expected_len = std.math.cast(usize, image.byte_count) orelse return error.HostReadableLengthMismatch;
        if (host_bytes.len != expected_len) return error.HostReadableLengthMismatch;
    }

    if (image.host_segments) |host_segments| {
        const expected_len = std.math.cast(usize, image.byte_count) orelse return error.HostReadableLengthMismatch;
        if (host_segments.len == 0) return error.HostReadableLengthMismatch;
        var total_len: usize = 0;
        for (host_segments) |segment| {
            if (segment.len == 0) return error.HostReadableLengthMismatch;
            total_len = std.math.add(usize, total_len, segment.len) catch return error.HostReadableLengthMismatch;
        }
        if (total_len != expected_len) return error.HostReadableLengthMismatch;
    }
}

fn hasHostReadableBytes(image: *const BoundaryByteImageRef) bool {
    return image.host_bytes != null or image.host_segments != null;
}

fn locationHintEql(
    lhs: ?tensor_frame.TensorFramePayloadLocationHint,
    rhs: ?tensor_frame.TensorFramePayloadLocationHint,
) bool {
    const lhs_hint = lhs orelse return rhs == null;
    const rhs_hint = rhs orelse return false;
    return switch (lhs_hint) {
        .cpu => switch (rhs_hint) {
            .cpu => true,
            else => false,
        },
        .cuda => |lhs_device_id| switch (rhs_hint) {
            .cuda => |rhs_device_id| lhs_device_id == rhs_device_id,
            else => false,
        },
        .metal => |lhs_device_id| switch (rhs_hint) {
            .metal => |rhs_device_id| lhs_device_id == rhs_device_id,
            else => false,
        },
        .opaque_local => |lhs_token| switch (rhs_hint) {
            .opaque_local => |rhs_token| lhs_token == rhs_token,
            else => false,
        },
    };
}

fn locationHintIsNullOrCpu(hint: ?tensor_frame.TensorFramePayloadLocationHint) bool {
    const value = hint orelse return true;
    return switch (value) {
        .cpu => true,
        else => false,
    };
}

fn locationHintIsCudaOrMetal(hint: ?tensor_frame.TensorFramePayloadLocationHint) bool {
    const value = hint orelse return false;
    return switch (value) {
        .cuda, .metal => true,
        else => false,
    };
}

fn locationHintIsOpaqueLocal(hint: ?tensor_frame.TensorFramePayloadLocationHint) bool {
    const value = hint orelse return false;
    return switch (value) {
        .opaque_local => true,
        else => false,
    };
}

const test_batch = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 3,
    .slot_id = 7,
    .sequence_start = 9,
    .token_count = 1,
}};

fn testMetadata(location_hint: ?tensor_frame.TensorFramePayloadLocationHint) !tensor_frame.TensorFrameMetadata {
    const boundary = tensor_frame.TensorFrameBoundaryRef{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 4,
        .consumer_layer_start = 4,
        .consumer_layer_end = 8,
    };
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, .{ 1, 1, 16, 0 });
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(1),
        .plan = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = .decode,
        .shape_context = .{ .expected_hidden_size = 16, .expected_step_kind = .decode },
        .tensor = tensor,
        .batch = .{ .entries = &test_batch },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = location_hint,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn testImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    readiness: BoundaryByteImageReadiness,
    host_bytes: ?[]const u8,
) BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = readiness,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn testSegmentedImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    readiness: BoundaryByteImageReadiness,
    host_segments: ?[]const []const u8,
) BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_segments = host_segments,
        .location_hint = metadata.payload.location_hint,
        .readiness = readiness,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

test "inference bridge boundary_byte_image validateBoundaryByteImage accepts host readable exact byte image" {
    const metadata = try testMetadata(.cpu);
    const payload = [_]u8{0xaa} ** 64;
    const image = testImage(&metadata, .host_readable_now, &payload);

    try validateBoundaryByteImage(&image, .{});
}

test "inference bridge boundary_byte_image validateBoundaryByteImage accepts valid host_segments for host readable exact byte image" {
    const metadata = try testMetadata(.cpu);
    const first = [_]u8{0xaa} ** 16;
    const second = [_]u8{0xbb} ** 48;
    const segments = [_][]const u8{ &first, &second };
    const image = testSegmentedImage(&metadata, .host_readable_now, &segments);

    try validateBoundaryByteImage(&image, .{});
    try validateBoundaryByteImage(&image, .{ .require_host_readable = true });
}

test "inference bridge boundary_byte_image validateBoundaryByteImage accepts producer sync cpu image without host bytes" {
    const metadata = try testMetadata(.cpu);
    const image = testImage(&metadata, .producer_sync_required, null);

    try validateBoundaryByteImage(&image, .{});
}

test "inference bridge boundary_byte_image validateBoundaryByteImage accepts opaque local image when allowed" {
    const metadata = try testMetadata(.{ .opaque_local = 9 });
    const image = testImage(&metadata, .local_only_opaque, null);

    try validateBoundaryByteImage(&image, .{});
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects invalid contract version before metadata validation" {
    var metadata = try testMetadata(.cpu);
    metadata.frame_id = .{ .value = 0 };
    var image = testImage(&metadata, .host_readable_now, null);
    image.version = 99;

    try std.testing.expectError(error.InvalidBoundaryByteImageContractVersion, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects host readable length mismatch" {
    const metadata = try testMetadata(.cpu);
    const payload = [_]u8{0xaa} ** 63;
    const image = testImage(&metadata, .host_readable_now, &payload);

    try std.testing.expectError(error.HostReadableLengthMismatch, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects invalid host_segments" {
    const metadata = try testMetadata(.cpu);
    const first = [_]u8{0xaa} ** 16;
    const second = [_]u8{0xbb} ** 48;
    const segments = [_][]const u8{ &first, &second };
    const contiguous = [_]u8{0xcc} ** 64;
    var image = testSegmentedImage(&metadata, .host_readable_now, &segments);
    image.host_bytes = &contiguous;
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&image, .{}));

    const empty_segments = [_][]const u8{};
    image = testSegmentedImage(&metadata, .host_readable_now, &empty_segments);
    try std.testing.expectError(error.HostReadableLengthMismatch, validateBoundaryByteImage(&image, .{}));

    const zero = [_]u8{};
    const zero_segments = [_][]const u8{ &first, &zero, &second };
    image = testSegmentedImage(&metadata, .host_readable_now, &zero_segments);
    try std.testing.expectError(error.HostReadableLengthMismatch, validateBoundaryByteImage(&image, .{}));

    const short_segments = [_][]const u8{&first};
    image = testSegmentedImage(&metadata, .host_readable_now, &short_segments);
    try std.testing.expectError(error.HostReadableLengthMismatch, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects invalid payload byte count" {
    const metadata = try testMetadata(.cpu);
    const payload = [_]u8{0xaa} ** 64;
    var image = testImage(&metadata, .host_readable_now, &payload);

    image.byte_count = 0;
    try std.testing.expectError(error.InvalidPayloadByteCount, validateBoundaryByteImage(&image, .{}));

    image.byte_count = metadata.payload.byte_count + 1;
    try std.testing.expectError(error.InvalidPayloadByteCount, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects missing host bytes when required" {
    const metadata = try testMetadata(.cpu);
    const image = testImage(&metadata, .host_readable_now, null);

    try std.testing.expectError(error.MissingHostReadableBytes, validateBoundaryByteImage(&image, .{ .require_host_readable = true }));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage accepts device resident byte image without forcing download" {
    const metadata = try testMetadata(.{ .cuda = 2 });
    const image = testImage(&metadata, .device_download_required, null);

    try validateBoundaryByteImage(&image, .{});
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects invalid readiness location combinations" {
    const cpu_metadata = try testMetadata(.cpu);
    const payload = [_]u8{0xaa} ** 64;
    var image = testImage(&cpu_metadata, .device_download_required, null);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&image, .{}));

    const cuda_metadata = try testMetadata(.{ .cuda = 1 });
    image = testImage(&cuda_metadata, .host_readable_now, &payload);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&image, .{}));

    image = testImage(&cpu_metadata, .producer_sync_required, &payload);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&image, .{}));

    const segments = [_][]const u8{ payload[0..16], payload[16..64] };
    var segmented_image = testSegmentedImage(&cpu_metadata, .producer_sync_required, &segments);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&segmented_image, .{}));

    segmented_image = testSegmentedImage(&cuda_metadata, .device_download_required, &segments);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&segmented_image, .{}));

    image = testImage(&cpu_metadata, .local_only_opaque, null);
    try std.testing.expectError(error.InvalidPayloadReadiness, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects opaque local when remote readable bytes are required" {
    const metadata = try testMetadata(.{ .opaque_local = 9 });
    const image = testImage(&metadata, .local_only_opaque, null);

    try std.testing.expectError(error.RemoteByteImageUnavailable, validateBoundaryByteImage(&image, .{ .allow_opaque_local = false }));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects payload location hint mismatches including absent hints" {
    const metadata = try testMetadata(.{ .cuda = 2 });

    var image = testImage(&metadata, .device_download_required, null);
    image.location_hint = .{ .cuda = 3 };
    try std.testing.expectError(error.InvalidPayloadLocationHint, validateBoundaryByteImage(&image, .{}));

    image = testImage(&metadata, .device_download_required, null);
    image.location_hint = null;
    try std.testing.expectError(error.InvalidPayloadLocationHint, validateBoundaryByteImage(&image, .{}));

    const absent_metadata = try testMetadata(null);
    image = testImage(&absent_metadata, .producer_sync_required, null);
    image.location_hint = .cpu;
    try std.testing.expectError(error.InvalidPayloadLocationHint, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage rejects payload ownership and lifetime mismatches" {
    const metadata = try testMetadata(.cpu);

    var image = testImage(&metadata, .producer_sync_required, null);
    image.ownership = .owned_by_sender_until_completion;
    try std.testing.expectError(error.InvalidPayloadOwnership, validateBoundaryByteImage(&image, .{}));

    image = testImage(&metadata, .producer_sync_required, null);
    image.lifetime = .request_scoped;
    try std.testing.expectError(error.InvalidPayloadLifetime, validateBoundaryByteImage(&image, .{}));
}

test "inference bridge boundary_byte_image validateBoundaryByteImage returns RemoteByteImageUnavailable for valid non remote readable images" {
    const cpu_metadata = try testMetadata(.cpu);
    const producer_sync = testImage(&cpu_metadata, .producer_sync_required, null);
    try std.testing.expectError(error.RemoteByteImageUnavailable, validateBoundaryByteImage(&producer_sync, .{ .require_host_readable = true }));

    const cuda_metadata = try testMetadata(.{ .cuda = 2 });
    const device_resident = testImage(&cuda_metadata, .device_download_required, null);
    try std.testing.expectError(error.RemoteByteImageUnavailable, validateBoundaryByteImage(&device_resident, .{ .require_host_readable = true }));

    const opaque_metadata = try testMetadata(.{ .opaque_local = 9 });
    const opaque_image = testImage(&opaque_metadata, .local_only_opaque, null);
    try std.testing.expectError(error.RemoteByteImageUnavailable, validateBoundaryByteImage(&opaque_image, .{ .require_host_readable = true, .allow_opaque_local = false }));
}

test "inference bridge boundary_byte_image boundaryByteImageIsRemoteReadable returns true only for host readable exact bytes" {
    const cpu_metadata = try testMetadata(.cpu);
    const payload = [_]u8{0xaa} ** 64;
    const remote_readable = testImage(&cpu_metadata, .host_readable_now, &payload);
    try std.testing.expect(boundaryByteImageIsRemoteReadable(&remote_readable));

    const first = [_]u8{0xbb} ** 16;
    const second = [_]u8{0xcc} ** 48;
    const segments = [_][]const u8{ &first, &second };
    const segmented_remote_readable = testSegmentedImage(&cpu_metadata, .host_readable_now, &segments);
    try std.testing.expect(boundaryByteImageIsRemoteReadable(&segmented_remote_readable));

    const producer_sync = testImage(&cpu_metadata, .producer_sync_required, null);
    try std.testing.expect(!boundaryByteImageIsRemoteReadable(&producer_sync));

    const cuda_metadata = try testMetadata(.{ .cuda = 2 });
    const device_resident = testImage(&cuda_metadata, .device_download_required, null);
    try std.testing.expect(!boundaryByteImageIsRemoteReadable(&device_resident));

    const opaque_metadata = try testMetadata(.{ .opaque_local = 9 });
    const opaque_image = testImage(&opaque_metadata, .local_only_opaque, null);
    try std.testing.expect(!boundaryByteImageIsRemoteReadable(&opaque_image));

    const short_payload = [_]u8{0xaa} ** 63;
    const invalid_host_readable = testImage(&cpu_metadata, .host_readable_now, &short_payload);
    try std.testing.expect(!boundaryByteImageIsRemoteReadable(&invalid_host_readable));

    const short_segments = [_][]const u8{&first};
    const invalid_segmented = testSegmentedImage(&cpu_metadata, .host_readable_now, &short_segments);
    try std.testing.expect(!boundaryByteImageIsRemoteReadable(&invalid_segmented));
}
