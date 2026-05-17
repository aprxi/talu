//! Fixed binary headers for staged activation byte images.

const std = @import("std");
const tensor_frame = @import("tensor_frame.zig");
const host_capability = @import("host_capability.zig");

pub const stage_frame_header_magic = [_]u8{ 'T', 'S', 'F', 'H' };
pub const stage_frame_header_contract_version: u16 = 1;
pub const stage_frame_header_encoded_len: usize = 224;

pub const StageFrameHeaderError = tensor_frame.TensorFrameValidationError || error{
    InvalidStageFrameHeaderMagic,
    InvalidStageFrameHeaderVersion,
    InvalidStageFrameHeaderLength,
    InvalidStageFrameHeaderFlags,
    InvalidStageFrameHeaderHostId,
    InvalidStageFrameHeaderPayloadByteCount,
    InvalidStageFrameHeaderTensorFacts,
    UnsupportedStageFrameHeaderBatch,
    StageFrameHeaderMismatch,
};

pub const StageFrameHeaderOptions = struct {
    source_host_id: ?host_capability.HostId = null,
    target_host_id: ?host_capability.HostId = null,
};

pub const StageFrameHeader = struct {
    flags: u32,
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id_digest: [32]u8,
    frame_id: u64,
    boundary_index: u64,
    source_stage_id: u64,
    target_stage_id: u64,
    source_host_id: ?host_capability.HostId = null,
    target_host_id: ?host_capability.HostId = null,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
    batch_index: u32,
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    layout: tensor_frame.TensorFrameLayout,
    rank: u8,
    shape: [4]u64,
    payload_byte_count: u64,
};

const flag_source_host_present: u32 = 1 << 0;
const flag_target_host_present: u32 = 1 << 1;
const valid_flags_mask: u32 = flag_source_host_present | flag_target_host_present;

const offset_magic: usize = 0;
const offset_version: usize = 4;
const offset_length: usize = 6;
const offset_flags: usize = 8;
const offset_graph_digest: usize = 12;
const offset_graph_contract_version: usize = 44;
const offset_stage_plan_contract_version: usize = 48;
const offset_stage_plan_id_digest: usize = 52;
const offset_frame_id: usize = 84;
const offset_boundary_index: usize = 92;
const offset_source_stage_id: usize = 100;
const offset_target_stage_id: usize = 108;
const offset_source_host_id: usize = 116;
const offset_target_host_id: usize = 124;
const offset_request_id: usize = 132;
const offset_slot_id: usize = 140;
const offset_sequence_start: usize = 148;
const offset_token_count: usize = 156;
const offset_batch_index: usize = 164;
const offset_step_kind: usize = 168;
const offset_dtype: usize = 169;
const offset_layout: usize = 170;
const offset_rank: usize = 171;
const offset_shape: usize = 172;
const offset_payload_byte_count: usize = 204;
const offset_reserved: usize = 212;

pub fn stageFrameHeaderFromMetadata(
    metadata: *const tensor_frame.TensorFrameMetadata,
    options: StageFrameHeaderOptions,
) StageFrameHeaderError!StageFrameHeader {
    try metadata.validate();
    if (metadata.batch.entries.len != 1) return error.UnsupportedStageFrameHeaderBatch;
    try validateHostIdOption(options.source_host_id);
    try validateHostIdOption(options.target_host_id);

    const entry = metadata.batch.entries[0];
    const header = StageFrameHeader{
        .flags = derivedFlags(options.source_host_id, options.target_host_id),
        .graph_digest = metadata.plan.graph_digest,
        .graph_contract_version = metadata.plan.graph_contract_version,
        .stage_plan_contract_version = metadata.plan.stage_plan_contract_version,
        .stage_plan_id_digest = metadata.plan.stage_plan_id.digest,
        .frame_id = metadata.frame_id.value,
        .boundary_index = @intCast(metadata.boundary.boundary_index),
        .source_stage_id = @intCast(metadata.boundary.source_stage_id),
        .target_stage_id = @intCast(metadata.boundary.target_stage_id),
        .source_host_id = options.source_host_id,
        .target_host_id = options.target_host_id,
        .request_id = entry.request_id,
        .slot_id = entry.slot_id,
        .sequence_start = entry.sequence_start,
        .token_count = entry.token_count,
        .batch_index = entry.batch_index,
        .step_kind = metadata.step_kind,
        .dtype = metadata.tensor.dtype,
        .layout = metadata.tensor.layout,
        .rank = metadata.tensor.rank,
        .shape = metadata.tensor.shape,
        .payload_byte_count = metadata.payload.byte_count,
    };
    try validateHeaderForEncode(header);
    return header;
}

pub fn encodeStageFrameHeader(
    dest: *[stage_frame_header_encoded_len]u8,
    header: StageFrameHeader,
) StageFrameHeaderError!void {
    try validateHeaderForEncode(header);

    dest.* = [_]u8{0} ** stage_frame_header_encoded_len;
    @memcpy(dest[offset_magic .. offset_magic + stage_frame_header_magic.len], &stage_frame_header_magic);
    writeU16(dest, offset_version, stage_frame_header_contract_version);
    writeU16(dest, offset_length, @intCast(stage_frame_header_encoded_len));
    writeU32(dest, offset_flags, header.flags);
    @memcpy(dest[offset_graph_digest .. offset_graph_digest + 32], &header.graph_digest);
    writeU32(dest, offset_graph_contract_version, header.graph_contract_version);
    writeU32(dest, offset_stage_plan_contract_version, header.stage_plan_contract_version);
    @memcpy(dest[offset_stage_plan_id_digest .. offset_stage_plan_id_digest + 32], &header.stage_plan_id_digest);
    writeU64(dest, offset_frame_id, header.frame_id);
    writeU64(dest, offset_boundary_index, header.boundary_index);
    writeU64(dest, offset_source_stage_id, header.source_stage_id);
    writeU64(dest, offset_target_stage_id, header.target_stage_id);
    writeU64(dest, offset_source_host_id, hostIdWireValue(header.source_host_id));
    writeU64(dest, offset_target_host_id, hostIdWireValue(header.target_host_id));
    writeU64(dest, offset_request_id, header.request_id);
    writeU64(dest, offset_slot_id, header.slot_id);
    writeU64(dest, offset_sequence_start, header.sequence_start);
    writeU64(dest, offset_token_count, header.token_count);
    writeU32(dest, offset_batch_index, header.batch_index);
    dest[offset_step_kind] = stepKindToWire(header.step_kind);
    dest[offset_dtype] = dtypeToWire(header.dtype);
    dest[offset_layout] = layoutToWire(header.layout);
    dest[offset_rank] = header.rank;
    for (header.shape, 0..) |dimension, dimension_idx| {
        writeU64(dest, offset_shape + dimension_idx * @sizeOf(u64), dimension);
    }
    writeU64(dest, offset_payload_byte_count, header.payload_byte_count);
}

pub fn decodeStageFrameHeader(
    src: *const [stage_frame_header_encoded_len]u8,
) StageFrameHeaderError!StageFrameHeader {
    if (!std.mem.eql(u8, src[offset_magic .. offset_magic + stage_frame_header_magic.len], &stage_frame_header_magic)) {
        return error.InvalidStageFrameHeaderMagic;
    }
    if (readU16(src, offset_version) != stage_frame_header_contract_version) {
        return error.InvalidStageFrameHeaderVersion;
    }
    if (readU16(src, offset_length) != stage_frame_header_encoded_len) {
        return error.InvalidStageFrameHeaderLength;
    }
    for (src[offset_reserved..stage_frame_header_encoded_len]) |reserved_byte| {
        if (reserved_byte != 0) return error.InvalidStageFrameHeaderFlags;
    }

    const flags = readU32(src, offset_flags);
    if ((flags & ~valid_flags_mask) != 0) return error.InvalidStageFrameHeaderFlags;

    const source_host_value = readU64(src, offset_source_host_id);
    const target_host_value = readU64(src, offset_target_host_id);
    if ((flags & flag_source_host_present) != 0 and source_host_value == 0) {
        return error.InvalidStageFrameHeaderHostId;
    }
    if ((flags & flag_source_host_present) == 0 and source_host_value != 0) {
        return error.InvalidStageFrameHeaderHostId;
    }
    if ((flags & flag_target_host_present) != 0 and target_host_value == 0) {
        return error.InvalidStageFrameHeaderHostId;
    }
    if ((flags & flag_target_host_present) == 0 and target_host_value != 0) {
        return error.InvalidStageFrameHeaderHostId;
    }

    const step_kind = try stepKindFromWire(src[offset_step_kind]);
    const dtype = try dtypeFromWire(src[offset_dtype]);
    const layout = try layoutFromWire(src[offset_layout]);

    var graph_digest: [32]u8 = undefined;
    @memcpy(&graph_digest, src[offset_graph_digest .. offset_graph_digest + graph_digest.len]);
    var stage_plan_id_digest: [32]u8 = undefined;
    @memcpy(&stage_plan_id_digest, src[offset_stage_plan_id_digest .. offset_stage_plan_id_digest + stage_plan_id_digest.len]);
    var shape: [4]u64 = undefined;
    for (&shape, 0..) |*dimension, dimension_idx| {
        dimension.* = readU64(src, offset_shape + dimension_idx * @sizeOf(u64));
    }

    const header = StageFrameHeader{
        .flags = flags,
        .graph_digest = graph_digest,
        .graph_contract_version = readU32(src, offset_graph_contract_version),
        .stage_plan_contract_version = readU32(src, offset_stage_plan_contract_version),
        .stage_plan_id_digest = stage_plan_id_digest,
        .frame_id = readU64(src, offset_frame_id),
        .boundary_index = readU64(src, offset_boundary_index),
        .source_stage_id = readU64(src, offset_source_stage_id),
        .target_stage_id = readU64(src, offset_target_stage_id),
        .source_host_id = hostIdFromWire(flags, flag_source_host_present, source_host_value),
        .target_host_id = hostIdFromWire(flags, flag_target_host_present, target_host_value),
        .request_id = readU64(src, offset_request_id),
        .slot_id = readU64(src, offset_slot_id),
        .sequence_start = readU64(src, offset_sequence_start),
        .token_count = readU64(src, offset_token_count),
        .batch_index = readU32(src, offset_batch_index),
        .step_kind = step_kind,
        .dtype = dtype,
        .layout = layout,
        .rank = src[offset_rank],
        .shape = shape,
        .payload_byte_count = readU64(src, offset_payload_byte_count),
    };
    try validateStandaloneIdentity(header);
    const expected_byte_count = try validateTensorFacts(header);
    try validatePayloadByteCount(header.payload_byte_count, expected_byte_count);
    if (derivedFlags(header.source_host_id, header.target_host_id) != flags) {
        return error.InvalidStageFrameHeaderFlags;
    }
    return header;
}

pub fn validateStageFrameHeaderForMetadata(
    header: StageFrameHeader,
    metadata: *const tensor_frame.TensorFrameMetadata,
) StageFrameHeaderError!void {
    try metadata.validate();
    try validateHeaderForEncode(header);
    if (metadata.batch.entries.len != 1) return error.UnsupportedStageFrameHeaderBatch;
    const expected = try stageFrameHeaderFromMetadata(metadata, .{
        .source_host_id = header.source_host_id,
        .target_host_id = header.target_host_id,
    });
    if (!headerComparableFieldsEql(header, expected)) {
        return error.StageFrameHeaderMismatch;
    }
}

fn validateHeaderForEncode(header: StageFrameHeader) StageFrameHeaderError!void {
    if (header.flags != derivedFlags(header.source_host_id, header.target_host_id)) {
        return error.InvalidStageFrameHeaderFlags;
    }
    if ((header.flags & ~valid_flags_mask) != 0) return error.InvalidStageFrameHeaderFlags;
    try validateHostIdOption(header.source_host_id);
    try validateHostIdOption(header.target_host_id);
    try validateStandaloneIdentity(header);
    const expected_byte_count = try validateTensorFacts(header);
    try validatePayloadByteCount(header.payload_byte_count, expected_byte_count);
}

fn validateStandaloneIdentity(header: StageFrameHeader) StageFrameHeaderError!void {
    if (header.frame_id == 0) return error.InvalidFrameId;
    if (header.request_id == 0) return error.InvalidRequestId;
    if (header.slot_id == 0) return error.InvalidSlotId;
    if (header.token_count == 0) return error.InvalidSequenceRange;
    _ = std.math.add(u64, header.sequence_start, header.token_count) catch return error.InvalidSequenceRange;
    if (header.step_kind == .decode and header.token_count != 1) return error.InvalidSequenceRange;
    if (header.source_stage_id == header.target_stage_id) return error.InvalidStageFrameHeaderTensorFacts;
}

fn validateTensorFacts(header: StageFrameHeader) StageFrameHeaderError!u64 {
    if (header.rank != 3) return error.InvalidStageFrameHeaderTensorFacts;
    if (header.shape[0] == 0 or header.shape[1] == 0 or header.shape[2] == 0) {
        return error.InvalidStageFrameHeaderTensorFacts;
    }
    if (header.shape[3] != 0) return error.InvalidStageFrameHeaderTensorFacts;
    if (header.layout != .row_major) return error.InvalidStageFrameHeaderTensorFacts;
    _ = stepKindToWire(header.step_kind);
    _ = dtypeToWire(header.dtype);
    _ = layoutToWire(header.layout);
    if (header.step_kind == .decode and header.shape[1] != 1) {
        return error.InvalidStageFrameHeaderTensorFacts;
    }
    return checkedTensorByteCount(header);
}

fn validatePayloadByteCount(payload_byte_count: u64, expected_byte_count: u64) StageFrameHeaderError!void {
    if (payload_byte_count == 0) return error.InvalidStageFrameHeaderPayloadByteCount;
    if (payload_byte_count != expected_byte_count) return error.InvalidStageFrameHeaderPayloadByteCount;
}

fn validateHostIdOption(host_id: ?host_capability.HostId) StageFrameHeaderError!void {
    if (host_id) |id| {
        if (id.value == 0) return error.InvalidStageFrameHeaderHostId;
    }
}

fn hostIdWireValue(host_id: ?host_capability.HostId) u64 {
    return if (host_id) |id| id.value else 0;
}

fn hostIdFromWire(flags: u32, flag: u32, value: u64) ?host_capability.HostId {
    return if ((flags & flag) != 0) .{ .value = value } else null;
}

fn hostIdEql(lhs: ?host_capability.HostId, rhs: ?host_capability.HostId) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return lhs.?.value == rhs.?.value;
}

fn derivedFlags(source_host_id: ?host_capability.HostId, target_host_id: ?host_capability.HostId) u32 {
    var flags: u32 = 0;
    if (source_host_id != null) flags |= flag_source_host_present;
    if (target_host_id != null) flags |= flag_target_host_present;
    return flags;
}

fn stepKindToWire(step_kind: tensor_frame.TensorFrameStepKind) u8 {
    return switch (step_kind) {
        .prefill => 0,
        .decode => 1,
    };
}

fn stepKindFromWire(value: u8) StageFrameHeaderError!tensor_frame.TensorFrameStepKind {
    return switch (value) {
        0 => .prefill,
        1 => .decode,
        else => error.InvalidStageFrameHeaderTensorFacts,
    };
}

fn dtypeToWire(dtype: tensor_frame.TensorFrameDType) u8 {
    return switch (dtype) {
        .bf16 => 0,
        .f16 => 1,
        .f32 => 2,
    };
}

fn dtypeFromWire(value: u8) StageFrameHeaderError!tensor_frame.TensorFrameDType {
    return switch (value) {
        0 => .bf16,
        1 => .f16,
        2 => .f32,
        else => error.InvalidStageFrameHeaderTensorFacts,
    };
}

fn layoutToWire(layout: tensor_frame.TensorFrameLayout) u8 {
    return switch (layout) {
        .row_major => 0,
    };
}

fn layoutFromWire(value: u8) StageFrameHeaderError!tensor_frame.TensorFrameLayout {
    return switch (value) {
        0 => .row_major,
        else => error.InvalidStageFrameHeaderTensorFacts,
    };
}

fn checkedTensorByteCount(header: StageFrameHeader) StageFrameHeaderError!u64 {
    const batch_tokens = std.math.mul(u64, header.shape[0], header.shape[1]) catch return error.ByteCountOverflow;
    const elements = std.math.mul(u64, batch_tokens, header.shape[2]) catch return error.ByteCountOverflow;
    return std.math.mul(u64, elements, tensor_frame.dtypeByteSize(header.dtype)) catch error.ByteCountOverflow;
}

fn headerComparableFieldsEql(lhs: StageFrameHeader, rhs: StageFrameHeader) bool {
    return lhs.flags == rhs.flags and
        std.mem.eql(u8, &lhs.graph_digest, &rhs.graph_digest) and
        lhs.graph_contract_version == rhs.graph_contract_version and
        lhs.stage_plan_contract_version == rhs.stage_plan_contract_version and
        std.mem.eql(u8, &lhs.stage_plan_id_digest, &rhs.stage_plan_id_digest) and
        lhs.frame_id == rhs.frame_id and
        lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        hostIdEql(lhs.source_host_id, rhs.source_host_id) and
        hostIdEql(lhs.target_host_id, rhs.target_host_id) and
        lhs.request_id == rhs.request_id and
        lhs.slot_id == rhs.slot_id and
        lhs.sequence_start == rhs.sequence_start and
        lhs.token_count == rhs.token_count and
        lhs.batch_index == rhs.batch_index and
        lhs.step_kind == rhs.step_kind and
        lhs.dtype == rhs.dtype and
        lhs.layout == rhs.layout and
        lhs.rank == rhs.rank and
        lhs.shape == rhs.shape and
        lhs.payload_byte_count == rhs.payload_byte_count;
}

fn readU16(src: *const [stage_frame_header_encoded_len]u8, offset: usize) u16 {
    return std.mem.readInt(u16, src[offset..][0..2], .little);
}

fn readU32(src: *const [stage_frame_header_encoded_len]u8, offset: usize) u32 {
    return std.mem.readInt(u32, src[offset..][0..4], .little);
}

fn readU64(src: *const [stage_frame_header_encoded_len]u8, offset: usize) u64 {
    return std.mem.readInt(u64, src[offset..][0..8], .little);
}

fn writeU16(dest: *[stage_frame_header_encoded_len]u8, offset: usize, value: u16) void {
    std.mem.writeInt(u16, dest[offset..][0..2], value, .little);
}

fn writeU32(dest: *[stage_frame_header_encoded_len]u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, dest[offset..][0..4], value, .little);
}

fn writeU64(dest: *[stage_frame_header_encoded_len]u8, offset: usize, value: u64) void {
    std.mem.writeInt(u64, dest[offset..][0..8], value, .little);
}

fn testDigest(seed: u8) [32]u8 {
    var digest: [32]u8 = undefined;
    for (&digest, 0..) |*byte, byte_idx| {
        byte.* = seed +% @as(u8, @intCast(byte_idx));
    }
    return digest;
}

fn testBoundary() tensor_frame.TensorFrameBoundaryRef {
    return .{
        .boundary_index = 2,
        .source_stage_id = 10,
        .target_stage_id = 11,
        .producer_layer_start = 4,
        .producer_layer_end = 8,
        .consumer_layer_start = 8,
        .consumer_layer_end = 12,
    };
}

fn testBatchEntry(batch_index: u32, request_id: u64, slot_id: u64, sequence_start: u64, token_count: u64) tensor_frame.TensorFrameBatchEntry {
    return .{
        .batch_index = batch_index,
        .request_id = request_id,
        .slot_id = slot_id,
        .sequence_start = sequence_start,
        .token_count = token_count,
    };
}

const test_prefill_entries = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 3,
    .request_id = 99,
    .slot_id = 77,
    .sequence_start = 12,
    .token_count = 4,
}};

const test_decode_entries = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 101,
    .slot_id = 88,
    .sequence_start = 12,
    .token_count = 1,
}};

fn testMetadata(
    entries: []const tensor_frame.TensorFrameBatchEntry,
    step_kind: tensor_frame.TensorFrameStepKind,
    dtype: tensor_frame.TensorFrameDType,
    shape: [4]u64,
) StageFrameHeaderError!tensor_frame.TensorFrameMetadata {
    const boundary = testBoundary();
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(dtype, shape);
    return .{
        .frame_id = .{ .value = 55 },
        .plan = .{
            .graph_digest = testDigest(0x10),
            .graph_contract_version = 7,
            .stage_plan_contract_version = 9,
            .stage_plan_id = .{ .digest = testDigest(0x40) },
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = dtype,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = step_kind,
        .shape_context = .{
            .expected_hidden_size = shape[2],
            .expected_step_kind = step_kind,
        },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{ .byte_count = tensor.payload_byte_count },
    };
}

fn testPrefillMetadata() StageFrameHeaderError!tensor_frame.TensorFrameMetadata {
    return testMetadata(&test_prefill_entries, .prefill, .f16, .{ 1, 4, 8, 0 });
}

fn testDecodeMetadata() StageFrameHeaderError!tensor_frame.TensorFrameMetadata {
    return testMetadata(&test_decode_entries, .decode, .f32, .{ 1, 1, 8, 0 });
}

fn testHeader() StageFrameHeaderError!StageFrameHeader {
    var metadata = try testPrefillMetadata();
    return stageFrameHeaderFromMetadata(&metadata, .{
        .source_host_id = .{ .value = 0x0102030405060708 },
        .target_host_id = .{ .value = 0x1112131415161718 },
    });
}

fn encodedHeader() StageFrameHeaderError![stage_frame_header_encoded_len]u8 {
    const header = try testHeader();
    var encoded: [stage_frame_header_encoded_len]u8 = undefined;
    try encodeStageFrameHeader(&encoded, header);
    return encoded;
}

fn expectLeU16(bytes: []const u8, value: u16) !void {
    var expected: [2]u8 = undefined;
    std.mem.writeInt(u16, &expected, value, .little);
    try std.testing.expectEqualSlices(u8, &expected, bytes);
}

fn expectLeU32(bytes: []const u8, value: u32) !void {
    var expected: [4]u8 = undefined;
    std.mem.writeInt(u32, &expected, value, .little);
    try std.testing.expectEqualSlices(u8, &expected, bytes);
}

fn expectLeU64(bytes: []const u8, value: u64) !void {
    var expected: [8]u8 = undefined;
    std.mem.writeInt(u64, &expected, value, .little);
    try std.testing.expectEqualSlices(u8, &expected, bytes);
}

fn expectMetadataMismatch(header: StageFrameHeader, metadata: *const tensor_frame.TensorFrameMetadata) !void {
    try std.testing.expectError(error.StageFrameHeaderMismatch, validateStageFrameHeaderForMetadata(header, metadata));
}

fn expectEncodeHeaderError(expected_error: anyerror, header: StageFrameHeader) !void {
    var encoded: [stage_frame_header_encoded_len]u8 = undefined;
    try std.testing.expectError(expected_error, encodeStageFrameHeader(&encoded, header));
}

test "inference pipeline stage_frame_header stageFrameHeaderFromMetadata copies tensor frame identity and payload facts" {
    var metadata = try testPrefillMetadata();
    const header = try stageFrameHeaderFromMetadata(&metadata, .{});

    try std.testing.expectEqual(@as(u32, 0), header.flags);
    try std.testing.expectEqualSlices(u8, &metadata.plan.graph_digest, &header.graph_digest);
    try std.testing.expectEqual(metadata.plan.graph_contract_version, header.graph_contract_version);
    try std.testing.expectEqual(metadata.plan.stage_plan_contract_version, header.stage_plan_contract_version);
    try std.testing.expectEqualSlices(u8, &metadata.plan.stage_plan_id.digest, &header.stage_plan_id_digest);
    try std.testing.expectEqual(metadata.frame_id.value, header.frame_id);
    try std.testing.expectEqual(@as(u64, @intCast(metadata.boundary.boundary_index)), header.boundary_index);
    try std.testing.expectEqual(@as(u64, @intCast(metadata.boundary.source_stage_id)), header.source_stage_id);
    try std.testing.expectEqual(@as(u64, @intCast(metadata.boundary.target_stage_id)), header.target_stage_id);
    try std.testing.expectEqual(metadata.batch.entries[0].request_id, header.request_id);
    try std.testing.expectEqual(metadata.batch.entries[0].slot_id, header.slot_id);
    try std.testing.expectEqual(metadata.batch.entries[0].sequence_start, header.sequence_start);
    try std.testing.expectEqual(metadata.batch.entries[0].token_count, header.token_count);
    try std.testing.expectEqual(metadata.batch.entries[0].batch_index, header.batch_index);
    try std.testing.expectEqual(metadata.step_kind, header.step_kind);
    try std.testing.expectEqual(metadata.tensor.dtype, header.dtype);
    try std.testing.expectEqual(metadata.tensor.layout, header.layout);
    try std.testing.expectEqual(metadata.tensor.rank, header.rank);
    try std.testing.expectEqual(metadata.tensor.shape, header.shape);
    try std.testing.expectEqual(metadata.payload.byte_count, header.payload_byte_count);
}

test "inference pipeline stage_frame_header stageFrameHeaderFromMetadata derives host flags and rejects zero host ids" {
    var metadata = try testPrefillMetadata();
    const source_only = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 12 } });
    try std.testing.expectEqual(flag_source_host_present, source_only.flags);
    try std.testing.expectEqual(@as(u64, 12), source_only.source_host_id.?.value);
    try std.testing.expectEqual(@as(?host_capability.HostId, null), source_only.target_host_id);

    const both = try stageFrameHeaderFromMetadata(&metadata, .{
        .source_host_id = .{ .value = 12 },
        .target_host_id = .{ .value = 13 },
    });
    try std.testing.expectEqual(valid_flags_mask, both.flags);

    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, stageFrameHeaderFromMetadata(&metadata, .{
        .source_host_id = .{ .value = 0 },
    }));
    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, stageFrameHeaderFromMetadata(&metadata, .{
        .target_host_id = .{ .value = 0 },
    }));
}

test "inference pipeline stage_frame_header stageFrameHeaderFromMetadata rejects invalid metadata and multi batch metadata" {
    var invalid_metadata = try testPrefillMetadata();
    invalid_metadata.frame_id.value = 0;
    try std.testing.expectError(error.InvalidFrameId, stageFrameHeaderFromMetadata(&invalid_metadata, .{}));

    const multi_entries = [_]tensor_frame.TensorFrameBatchEntry{
        testBatchEntry(0, 100, 200, 3, 1),
        testBatchEntry(1, 101, 201, 3, 1),
    };
    var multi_batch = try testMetadata(&multi_entries, .decode, .f32, .{ 2, 1, 8, 0 });
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, stageFrameHeaderFromMetadata(&multi_batch, .{}));
}

test "inference pipeline stage_frame_header encodeStageFrameHeader decodeStageFrameHeader round trips fixed little endian header and exact byte offsets" {
    const header = try testHeader();
    var encoded: [stage_frame_header_encoded_len]u8 = undefined;
    try encodeStageFrameHeader(&encoded, header);

    try std.testing.expectEqualSlices(u8, &stage_frame_header_magic, encoded[offset_magic .. offset_magic + 4]);
    try expectLeU16(encoded[offset_version .. offset_version + 2], stage_frame_header_contract_version);
    try expectLeU16(encoded[offset_length .. offset_length + 2], @intCast(stage_frame_header_encoded_len));
    try expectLeU32(encoded[offset_flags .. offset_flags + 4], valid_flags_mask);
    try std.testing.expectEqual(header.graph_digest[7], encoded[offset_graph_digest + 7]);
    try std.testing.expectEqual(header.stage_plan_id_digest[9], encoded[offset_stage_plan_id_digest + 9]);
    try expectLeU64(encoded[offset_frame_id .. offset_frame_id + 8], header.frame_id);
    try expectLeU64(encoded[offset_boundary_index .. offset_boundary_index + 8], header.boundary_index);
    try expectLeU64(encoded[offset_source_stage_id .. offset_source_stage_id + 8], header.source_stage_id);
    try expectLeU64(encoded[offset_target_stage_id .. offset_target_stage_id + 8], header.target_stage_id);
    try expectLeU64(encoded[offset_source_host_id .. offset_source_host_id + 8], header.source_host_id.?.value);
    try expectLeU64(encoded[offset_target_host_id .. offset_target_host_id + 8], header.target_host_id.?.value);
    try expectLeU64(encoded[offset_request_id .. offset_request_id + 8], header.request_id);
    try expectLeU64(encoded[offset_slot_id .. offset_slot_id + 8], header.slot_id);
    try expectLeU64(encoded[offset_sequence_start .. offset_sequence_start + 8], header.sequence_start);
    try expectLeU64(encoded[offset_token_count .. offset_token_count + 8], header.token_count);
    try expectLeU32(encoded[offset_batch_index .. offset_batch_index + 4], header.batch_index);
    try std.testing.expectEqual(@as(u8, 0), encoded[offset_step_kind]);
    try std.testing.expectEqual(@as(u8, 1), encoded[offset_dtype]);
    try std.testing.expectEqual(@as(u8, 0), encoded[offset_layout]);
    try std.testing.expectEqual(@as(u8, 3), encoded[offset_rank]);
    for (header.shape, 0..) |dimension, dimension_idx| {
        const offset = offset_shape + dimension_idx * @sizeOf(u64);
        try expectLeU64(encoded[offset .. offset + 8], dimension);
    }
    try expectLeU64(encoded[offset_payload_byte_count .. offset_payload_byte_count + 8], header.payload_byte_count);
    for (encoded[offset_reserved..stage_frame_header_encoded_len]) |reserved_byte| {
        try std.testing.expectEqual(@as(u8, 0), reserved_byte);
    }

    const decoded = try decodeStageFrameHeader(&encoded);
    try std.testing.expect(headerComparableFieldsEql(header, decoded));
}

test "inference pipeline stage_frame_header encodeStageFrameHeader rejects invalid flags host ids tensor facts and payload byte count" {
    var header = try testHeader();
    header.flags = 0;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderFlags, header);

    header = try testHeader();
    header.flags |= 1 << 4;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderFlags, header);

    header = try testHeader();
    header.source_host_id = .{ .value = 0 };
    header.flags = derivedFlags(header.source_host_id, header.target_host_id);
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderHostId, header);

    header = try testHeader();
    header.frame_id = 0;
    try expectEncodeHeaderError(error.InvalidFrameId, header);

    header = try testHeader();
    header.request_id = 0;
    try expectEncodeHeaderError(error.InvalidRequestId, header);

    header = try testHeader();
    header.slot_id = 0;
    try expectEncodeHeaderError(error.InvalidSlotId, header);

    header = try testHeader();
    header.token_count = 0;
    try expectEncodeHeaderError(error.InvalidSequenceRange, header);

    header = try testHeader();
    header.sequence_start = std.math.maxInt(u64);
    header.token_count = 1;
    try expectEncodeHeaderError(error.InvalidSequenceRange, header);

    header = try testHeader();
    header.step_kind = .decode;
    header.token_count = 2;
    try expectEncodeHeaderError(error.InvalidSequenceRange, header);

    header = try testHeader();
    header.target_stage_id = header.source_stage_id;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderTensorFacts, header);

    header = try testHeader();
    header.rank = 2;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderTensorFacts, header);

    header = try testHeader();
    header.shape[3] = 1;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderTensorFacts, header);

    header = try testHeader();
    header.shape = .{ std.math.maxInt(u64), 2, 2, 0 };
    try expectEncodeHeaderError(error.ByteCountOverflow, header);

    header = try testHeader();
    header.payload_byte_count = 0;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderPayloadByteCount, header);

    header = try testHeader();
    header.payload_byte_count += 2;
    try expectEncodeHeaderError(error.InvalidStageFrameHeaderPayloadByteCount, header);
}

test "inference pipeline stage_frame_header decodeStageFrameHeader rejects wrong magic version length reserved bytes flags host ids identities enums tensor facts and payload byte count" {
    var encoded = try encodedHeader();
    encoded[offset_magic] = 'B';
    try std.testing.expectError(error.InvalidStageFrameHeaderMagic, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU16(&encoded, offset_version, stage_frame_header_contract_version + 1);
    try std.testing.expectError(error.InvalidStageFrameHeaderVersion, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU16(&encoded, offset_length, stage_frame_header_encoded_len - 1);
    try std.testing.expectError(error.InvalidStageFrameHeaderLength, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_reserved] = 1;
    try std.testing.expectError(error.InvalidStageFrameHeaderFlags, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU32(&encoded, offset_flags, valid_flags_mask | (1 << 4));
    try std.testing.expectError(error.InvalidStageFrameHeaderFlags, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_source_host_id, 0);
    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU32(&encoded, offset_flags, flag_target_host_present);
    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_target_host_id, 0);
    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU32(&encoded, offset_flags, flag_source_host_present);
    try std.testing.expectError(error.InvalidStageFrameHeaderHostId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_frame_id, 0);
    try std.testing.expectError(error.InvalidFrameId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_request_id, 0);
    try std.testing.expectError(error.InvalidRequestId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_slot_id, 0);
    try std.testing.expectError(error.InvalidSlotId, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_token_count, 0);
    try std.testing.expectError(error.InvalidSequenceRange, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_sequence_start, std.math.maxInt(u64));
    writeU64(&encoded, offset_token_count, 1);
    try std.testing.expectError(error.InvalidSequenceRange, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_step_kind] = 1;
    writeU64(&encoded, offset_token_count, 2);
    try std.testing.expectError(error.InvalidSequenceRange, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_target_stage_id, readU64(&encoded, offset_source_stage_id));
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_step_kind] = 2;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_dtype] = 3;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_layout] = 1;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_rank] = 2;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_shape, 0);
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_shape + 3 * @sizeOf(u64), 1);
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    encoded[offset_step_kind] = 1;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_shape, std.math.maxInt(u64));
    writeU64(&encoded, offset_shape + @sizeOf(u64), 2);
    writeU64(&encoded, offset_shape + 2 * @sizeOf(u64), 2);
    try std.testing.expectError(error.ByteCountOverflow, decodeStageFrameHeader(&encoded));

    encoded = try encodedHeader();
    writeU64(&encoded, offset_payload_byte_count, 0);
    try std.testing.expectError(error.InvalidStageFrameHeaderPayloadByteCount, decodeStageFrameHeader(&encoded));
}

test "inference pipeline stage_frame_header validateStageFrameHeaderForMetadata rejects invalid metadata multi batch graph stage plan boundary tensor batch and byte count mismatches" {
    var metadata = try testPrefillMetadata();
    var header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    try validateStageFrameHeaderForMetadata(header, &metadata);

    var invalid_metadata = metadata;
    invalid_metadata.frame_id.value = 0;
    try std.testing.expectError(error.InvalidFrameId, validateStageFrameHeaderForMetadata(header, &invalid_metadata));

    const multi_entries = [_]tensor_frame.TensorFrameBatchEntry{
        testBatchEntry(0, 100, 200, 3, 1),
        testBatchEntry(1, 101, 201, 3, 1),
    };
    var multi_batch = try testMetadata(&multi_entries, .decode, .f32, .{ 2, 1, 8, 0 });
    const multi_header = StageFrameHeader{
        .flags = 0,
        .graph_digest = multi_batch.plan.graph_digest,
        .graph_contract_version = multi_batch.plan.graph_contract_version,
        .stage_plan_contract_version = multi_batch.plan.stage_plan_contract_version,
        .stage_plan_id_digest = multi_batch.plan.stage_plan_id.digest,
        .frame_id = multi_batch.frame_id.value,
        .boundary_index = @intCast(multi_batch.boundary.boundary_index),
        .source_stage_id = @intCast(multi_batch.boundary.source_stage_id),
        .target_stage_id = @intCast(multi_batch.boundary.target_stage_id),
        .request_id = multi_batch.batch.entries[0].request_id,
        .slot_id = multi_batch.batch.entries[0].slot_id,
        .sequence_start = multi_batch.batch.entries[0].sequence_start,
        .token_count = multi_batch.batch.entries[0].token_count,
        .batch_index = multi_batch.batch.entries[0].batch_index,
        .step_kind = multi_batch.step_kind,
        .dtype = multi_batch.tensor.dtype,
        .layout = multi_batch.tensor.layout,
        .rank = multi_batch.tensor.rank,
        .shape = multi_batch.tensor.shape,
        .payload_byte_count = multi_batch.payload.byte_count,
    };
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, validateStageFrameHeaderForMetadata(multi_header, &multi_batch));

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.graph_digest[0] ^= 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.stage_plan_contract_version += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.boundary_index += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.request_id += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.slot_id += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.sequence_start += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.token_count += 1;
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.batch_index += 1;
    try expectMetadataMismatch(header, &metadata);

    var decode_metadata = try testDecodeMetadata();
    header = try stageFrameHeaderFromMetadata(&decode_metadata, .{});
    header.step_kind = .prefill;
    try expectMetadataMismatch(header, &decode_metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.dtype = .f32;
    header.payload_byte_count = header.shape[0] * header.shape[1] * header.shape[2] * tensor_frame.dtypeByteSize(header.dtype);
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.shape = .{ 1, 2, 16, 0 };
    header.payload_byte_count = header.shape[0] * header.shape[1] * header.shape[2] * tensor_frame.dtypeByteSize(header.dtype);
    try expectMetadataMismatch(header, &metadata);

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.rank = 4;
    try std.testing.expectError(error.InvalidStageFrameHeaderTensorFacts, validateStageFrameHeaderForMetadata(header, &metadata));

    header = try stageFrameHeaderFromMetadata(&metadata, .{ .source_host_id = .{ .value = 31 } });
    header.payload_byte_count += tensor_frame.dtypeByteSize(header.dtype);
    try std.testing.expectError(error.InvalidStageFrameHeaderPayloadByteCount, validateStageFrameHeaderForMetadata(header, &metadata));
}
