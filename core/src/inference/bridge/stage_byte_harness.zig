//! Local byte reader/writer harness for staged activation frames.
//!
//! This module moves an already-validated fixed header and raw activation
//! payload bytes through caller-provided local byte interfaces. It performs no
//! allocation, transport setup, retry policy, device synchronization, or tensor
//! value inspection.

const std = @import("std");
const tensor_frame = @import("tensor_frame.zig");
const boundary_byte_image = @import("boundary_byte_image.zig");
const stage_frame_header = @import("stage_frame_header.zig");

pub const StageByteHarnessError =
    stage_frame_header.StageFrameHeaderError ||
    boundary_byte_image.BoundaryByteImageError ||
    error{
        StageFrameMetadataMismatch,
        ShortStageHeaderWrite,
        ShortStageHeaderRead,
        ShortStagePayloadWrite,
        ShortStagePayloadRead,
        StagePayloadDestinationTooSmall,
    };

pub const StageFrameReadResult = struct {
    header: stage_frame_header.StageFrameHeader,
    payload: []u8,
};

pub fn writeStageFrameBytes(
    writer: anytype,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    header_options: stage_frame_header.StageFrameHeaderOptions,
) StageByteHarnessError!stage_frame_header.StageFrameHeader {
    try boundary_byte_image.validateBoundaryByteImage(image, .{
        .require_host_readable = true,
        .allow_opaque_local = false,
    });
    if (image.metadata != metadata) return error.StageFrameMetadataMismatch;

    const header = try stage_frame_header.stageFrameHeaderFromMetadata(metadata, header_options);
    var header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
    try stage_frame_header.encodeStageFrameHeader(&header_bytes, header);

    const header_count = writer.write(header_bytes[0..]) catch return error.ShortStageHeaderWrite;
    if (header_count < header_bytes.len) return error.ShortStageHeaderWrite;

    if (image.host_bytes) |payload| {
        const payload_count = writer.write(payload) catch return error.ShortStagePayloadWrite;
        if (payload_count < payload.len) return error.ShortStagePayloadWrite;
    } else if (image.host_segments) |segments| {
        for (segments) |segment| {
            const payload_count = writer.write(segment) catch return error.ShortStagePayloadWrite;
            if (payload_count < segment.len) return error.ShortStagePayloadWrite;
        }
    } else {
        unreachable;
    }

    return header;
}

pub fn readStageFrameBytes(
    reader: anytype,
    expected_metadata: *const tensor_frame.TensorFrameMetadata,
    payload_dest: []u8,
) StageByteHarnessError!StageFrameReadResult {
    var header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
    const header_count = reader.read(header_bytes[0..]) catch return error.ShortStageHeaderRead;
    if (header_count < header_bytes.len) return error.ShortStageHeaderRead;

    const header = try stage_frame_header.decodeStageFrameHeader(&header_bytes);
    try stage_frame_header.validateStageFrameHeaderForMetadata(header, expected_metadata);

    const payload_len = std.math.cast(usize, header.payload_byte_count) orelse return error.StagePayloadDestinationTooSmall;
    if (payload_dest.len < payload_len) return error.StagePayloadDestinationTooSmall;

    const payload = payload_dest[0..payload_len];
    const payload_count = reader.read(payload) catch return error.ShortStagePayloadRead;
    if (payload_count < payload.len) return error.ShortStagePayloadRead;

    return .{
        .header = header,
        .payload = payload,
    };
}

const test_payload_len: usize = 16;
const test_large_payload_len: usize = 32;

const TestWriter = struct {
    dest: []u8,
    len: usize = 0,
    call_count: usize = 0,
    call_lengths: [4]usize = [_]usize{0} ** 4,
    fail_call: ?usize = null,
    short_call: ?usize = null,

    fn write(self: *TestWriter, bytes: []const u8) !usize {
        const call_idx = self.call_count;
        self.call_count += 1;
        if (call_idx < self.call_lengths.len) {
            self.call_lengths[call_idx] = bytes.len;
        }
        if (self.fail_call) |fail_call| {
            if (fail_call == call_idx) return error.InjectedWriteFailure;
        }

        var count = bytes.len;
        if (self.short_call) |short_call| {
            if (short_call == call_idx and count > 0) count -= 1;
        }
        if (count > self.dest.len - self.len) return error.InjectedWriteFailure;
        @memcpy(self.dest[self.len..][0..count], bytes[0..count]);
        self.len += count;
        return count;
    }
};

const TestReader = struct {
    source: []const u8,
    offset: usize = 0,
    call_count: usize = 0,
    call_lengths: [4]usize = [_]usize{0} ** 4,
    fail_call: ?usize = null,
    short_call: ?usize = null,

    fn read(self: *TestReader, dest: []u8) !usize {
        const call_idx = self.call_count;
        self.call_count += 1;
        if (call_idx < self.call_lengths.len) {
            self.call_lengths[call_idx] = dest.len;
        }
        if (self.fail_call) |fail_call| {
            if (fail_call == call_idx) return error.InjectedReadFailure;
        }

        var count = dest.len;
        if (self.short_call) |short_call| {
            if (short_call == call_idx and count > 0) count -= 1;
        }
        if (count > self.source.len - self.offset) {
            count = self.source.len - self.offset;
        }
        @memcpy(dest[0..count], self.source[self.offset..][0..count]);
        self.offset += count;
        return count;
    }
};

fn testDigest(seed: u8) [32]u8 {
    var digest: [32]u8 = undefined;
    for (&digest, 0..) |*byte, byte_idx| {
        byte.* = seed +% @as(u8, @intCast(byte_idx));
    }
    return digest;
}

fn testBoundary(boundary_index: usize) tensor_frame.TensorFrameBoundaryRef {
    return .{
        .boundary_index = boundary_index,
        .source_stage_id = 10,
        .target_stage_id = 11,
        .producer_layer_start = 4,
        .producer_layer_end = 8,
        .consumer_layer_start = 8,
        .consumer_layer_end = 12,
    };
}

fn testBatchEntry(
    batch_index: u32,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
) tensor_frame.TensorFrameBatchEntry {
    return .{
        .batch_index = batch_index,
        .request_id = request_id,
        .slot_id = slot_id,
        .sequence_start = sequence_start,
        .token_count = token_count,
    };
}

const test_decode_entries = [_]tensor_frame.TensorFrameBatchEntry{
    testBatchEntry(0, 101, 88, 12, 1),
};

const test_multi_entries = [_]tensor_frame.TensorFrameBatchEntry{
    testBatchEntry(0, 201, 91, 20, 1),
    testBatchEntry(1, 202, 92, 20, 1),
};

fn testMetadata(
    frame_id: u64,
    boundary: tensor_frame.TensorFrameBoundaryRef,
    entries: []const tensor_frame.TensorFrameBatchEntry,
    shape: [4]u64,
) StageByteHarnessError!tensor_frame.TensorFrameMetadata {
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, shape);
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(frame_id),
        .plan = .{
            .graph_digest = testDigest(0x10),
            .graph_contract_version = 7,
            .stage_plan_contract_version = 9,
            .stage_plan_id = .{ .digest = testDigest(0x40) },
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
        .shape_context = .{
            .expected_hidden_size = shape[2],
            .expected_step_kind = .decode,
        },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = .cpu,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn testDecodeMetadata() StageByteHarnessError!tensor_frame.TensorFrameMetadata {
    return testMetadata(55, testBoundary(2), &test_decode_entries, .{ 1, 1, 4, 0 });
}

fn testLargeDecodeMetadata() StageByteHarnessError!tensor_frame.TensorFrameMetadata {
    return testMetadata(55, testBoundary(2), &test_decode_entries, .{ 1, 1, 8, 0 });
}

fn testMultiBatchMetadata() StageByteHarnessError!tensor_frame.TensorFrameMetadata {
    return testMetadata(57, testBoundary(2), &test_multi_entries, .{ 2, 1, 4, 0 });
}

fn testImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    host_bytes: []const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = .host_readable_now,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn testSegmentedImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    host_segments: []const []const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_segments = host_segments,
        .location_hint = metadata.payload.location_hint,
        .readiness = .host_readable_now,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn encodeFrame(
    metadata: *const tensor_frame.TensorFrameMetadata,
    dest: []u8,
    payload: []const u8,
) StageByteHarnessError!stage_frame_header.StageFrameHeader {
    const header = try stage_frame_header.stageFrameHeaderFromMetadata(metadata, .{});
    var header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
    try stage_frame_header.encodeStageFrameHeader(&header_bytes, header);
    @memcpy(dest[0..stage_frame_header.stage_frame_header_encoded_len], &header_bytes);
    @memcpy(dest[stage_frame_header.stage_frame_header_encoded_len..][0..payload.len], payload);
    return header;
}

fn expectFilled(bytes: []const u8, value: u8) !void {
    for (bytes) |byte| {
        try std.testing.expectEqual(value, byte);
    }
}

test "inference bridge stage_byte_harness writeStageFrameBytes readStageFrameBytes copies header plus raw payload bytes" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{ 0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41, 0x42, 0x43 };
    const image = testImage(&metadata, &payload);
    var written: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);
    var writer = TestWriter{ .dest = written[0..] };

    const written_header = try writeStageFrameBytes(&writer, &metadata, &image, .{
        .source_host_id = .{ .value = 31 },
        .target_host_id = .{ .value = 32 },
    });

    try std.testing.expectEqual(@as(usize, 2), writer.call_count);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len, writer.call_lengths[0]);
    try std.testing.expectEqual(payload.len, writer.call_lengths[1]);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len + payload.len, writer.len);

    var expected_header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
    try stage_frame_header.encodeStageFrameHeader(&expected_header_bytes, written_header);
    try std.testing.expectEqualSlices(u8, &expected_header_bytes, written[0..stage_frame_header.stage_frame_header_encoded_len]);
    try std.testing.expectEqualSlices(u8, &payload, written[stage_frame_header.stage_frame_header_encoded_len..writer.len]);

    var reader = TestReader{ .source = written[0..writer.len] };
    var received: [test_payload_len]u8 = [_]u8{0} ** test_payload_len;
    const result = try readStageFrameBytes(&reader, &metadata, received[0..]);

    try std.testing.expectEqual(@as(usize, 2), reader.call_count);
    try stage_frame_header.validateStageFrameHeaderForMetadata(result.header, &metadata);
    try std.testing.expectEqual(written_header.payload_byte_count, result.header.payload_byte_count);
    try std.testing.expectEqualSlices(u8, &payload, result.payload);
    try std.testing.expectEqualSlices(u8, &payload, &received);
}

test "inference bridge stage_byte_harness writeStageFrameBytes writes segmented host readable bytes in order" {
    var metadata = try testDecodeMetadata();
    const first = [_]u8{ 0x10, 0x11, 0x12, 0x13 };
    const second = [_]u8{ 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33, 0x40, 0x41, 0x42, 0x43 };
    const expected_payload = first ++ second;
    const segments = [_][]const u8{ &first, &second };
    const image = testSegmentedImage(&metadata, &segments);
    var written: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);
    var writer = TestWriter{ .dest = written[0..] };

    const written_header = try writeStageFrameBytes(&writer, &metadata, &image, .{});

    try std.testing.expectEqual(@as(usize, 3), writer.call_count);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len, writer.call_lengths[0]);
    try std.testing.expectEqual(first.len, writer.call_lengths[1]);
    try std.testing.expectEqual(second.len, writer.call_lengths[2]);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len + expected_payload.len, writer.len);

    var expected_header_bytes: [stage_frame_header.stage_frame_header_encoded_len]u8 = undefined;
    try stage_frame_header.encodeStageFrameHeader(&expected_header_bytes, written_header);
    try std.testing.expectEqualSlices(u8, &expected_header_bytes, written[0..stage_frame_header.stage_frame_header_encoded_len]);
    try std.testing.expectEqualSlices(u8, &expected_payload, written[stage_frame_header.stage_frame_header_encoded_len..writer.len]);
}

test "inference bridge stage_byte_harness writeStageFrameBytes rejects mismatched metadata and image before header write" {
    var metadata = try testDecodeMetadata();
    var other_metadata = try testMetadata(66, testBoundary(2), &test_decode_entries, .{ 1, 1, 4, 0 });
    const payload = [_]u8{0xaa} ** test_payload_len;
    const image = testImage(&metadata, &payload);
    var dest: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);
    var writer = TestWriter{ .dest = dest[0..] };

    try std.testing.expectError(error.StageFrameMetadataMismatch, writeStageFrameBytes(&writer, &other_metadata, &image, .{}));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);

    var unavailable_image = image;
    unavailable_image.readiness = .producer_sync_required;
    unavailable_image.host_bytes = null;
    try std.testing.expectError(error.RemoteByteImageUnavailable, writeStageFrameBytes(&writer, &metadata, &unavailable_image, .{}));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);

    var multi_metadata = try testMultiBatchMetadata();
    const multi_payload = [_]u8{0xbb} ** test_large_payload_len;
    const multi_image = testImage(&multi_metadata, &multi_payload);
    try std.testing.expectError(error.UnsupportedStageFrameHeaderBatch, writeStageFrameBytes(&writer, &multi_metadata, &multi_image, .{}));
    try std.testing.expectEqual(@as(usize, 0), writer.call_count);
}

test "inference bridge stage_byte_harness writeStageFrameBytes reports writer errors short header and short payload writes" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xcc} ** test_payload_len;
    const image = testImage(&metadata, &payload);
    var dest: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = [_]u8{0} ** (stage_frame_header.stage_frame_header_encoded_len + test_payload_len);

    var header_error_writer = TestWriter{ .dest = dest[0..], .fail_call = 0 };
    try std.testing.expectError(error.ShortStageHeaderWrite, writeStageFrameBytes(&header_error_writer, &metadata, &image, .{}));
    try std.testing.expectEqual(@as(usize, 1), header_error_writer.call_count);

    var header_short_writer = TestWriter{ .dest = dest[0..], .short_call = 0 };
    try std.testing.expectError(error.ShortStageHeaderWrite, writeStageFrameBytes(&header_short_writer, &metadata, &image, .{}));
    try std.testing.expectEqual(@as(usize, 1), header_short_writer.call_count);

    var payload_error_writer = TestWriter{ .dest = dest[0..], .fail_call = 1 };
    try std.testing.expectError(error.ShortStagePayloadWrite, writeStageFrameBytes(&payload_error_writer, &metadata, &image, .{}));
    try std.testing.expectEqual(@as(usize, 2), payload_error_writer.call_count);
    try std.testing.expectEqual(stage_frame_header.stage_frame_header_encoded_len, payload_error_writer.len);

    var payload_short_writer = TestWriter{ .dest = dest[0..], .short_call = 1 };
    try std.testing.expectError(error.ShortStagePayloadWrite, writeStageFrameBytes(&payload_short_writer, &metadata, &image, .{}));
    try std.testing.expectEqual(@as(usize, 2), payload_short_writer.call_count);

    const first = [_]u8{0xcd} ** 4;
    const second = [_]u8{0xce} ** 12;
    const segments = [_][]const u8{ &first, &second };
    const segmented_image = testSegmentedImage(&metadata, &segments);

    var segment_error_writer = TestWriter{ .dest = dest[0..], .fail_call = 2 };
    try std.testing.expectError(error.ShortStagePayloadWrite, writeStageFrameBytes(&segment_error_writer, &metadata, &segmented_image, .{}));
    try std.testing.expectEqual(@as(usize, 3), segment_error_writer.call_count);

    var segment_short_writer = TestWriter{ .dest = dest[0..], .short_call = 2 };
    try std.testing.expectError(error.ShortStagePayloadWrite, writeStageFrameBytes(&segment_short_writer, &metadata, &segmented_image, .{}));
    try std.testing.expectEqual(@as(usize, 3), segment_short_writer.call_count);
}

test "inference bridge stage_byte_harness readStageFrameBytes rejects reader errors short header and short payload" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xdd} ** test_payload_len;
    var frame: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = undefined;
    _ = try encodeFrame(&metadata, frame[0..], &payload);

    var dest: [test_payload_len]u8 = [_]u8{0x5a} ** test_payload_len;
    var header_error_reader = TestReader{ .source = frame[0..], .fail_call = 0 };
    try std.testing.expectError(error.ShortStageHeaderRead, readStageFrameBytes(&header_error_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), header_error_reader.call_count);
    try expectFilled(&dest, 0x5a);

    var header_short_reader = TestReader{ .source = frame[0..], .short_call = 0 };
    try std.testing.expectError(error.ShortStageHeaderRead, readStageFrameBytes(&header_short_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), header_short_reader.call_count);
    try expectFilled(&dest, 0x5a);

    var payload_error_reader = TestReader{ .source = frame[0..], .fail_call = 1 };
    try std.testing.expectError(error.ShortStagePayloadRead, readStageFrameBytes(&payload_error_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 2), payload_error_reader.call_count);

    var payload_short_reader = TestReader{ .source = frame[0..], .short_call = 1 };
    try std.testing.expectError(error.ShortStagePayloadRead, readStageFrameBytes(&payload_short_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 2), payload_short_reader.call_count);
}

test "inference bridge stage_byte_harness readStageFrameBytes rejects graph stage boundary and byte count mismatch before receiver mutation" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xee} ** test_payload_len;
    var frame: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = undefined;
    _ = try encodeFrame(&metadata, frame[0..], &payload);

    var boundary_mismatch_metadata = try testMetadata(55, testBoundary(3), &test_decode_entries, .{ 1, 1, 4, 0 });
    var dest: [test_large_payload_len]u8 = [_]u8{0x7b} ** test_large_payload_len;
    var boundary_reader = TestReader{ .source = frame[0..] };
    try std.testing.expectError(error.StageFrameHeaderMismatch, readStageFrameBytes(&boundary_reader, &boundary_mismatch_metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), boundary_reader.call_count);
    try expectFilled(&dest, 0x7b);

    var large_metadata = try testLargeDecodeMetadata();
    const large_payload = [_]u8{0xef} ** test_large_payload_len;
    var large_frame: [stage_frame_header.stage_frame_header_encoded_len + test_large_payload_len]u8 = undefined;
    _ = try encodeFrame(&large_metadata, large_frame[0..], &large_payload);
    var byte_count_reader = TestReader{ .source = large_frame[0..] };
    try std.testing.expectError(error.StageFrameHeaderMismatch, readStageFrameBytes(&byte_count_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), byte_count_reader.call_count);
    try expectFilled(&dest, 0x7b);

    var corrupt_frame = frame;
    corrupt_frame[0] = 'B';
    var decode_reader = TestReader{ .source = corrupt_frame[0..] };
    try std.testing.expectError(error.InvalidStageFrameHeaderMagic, readStageFrameBytes(&decode_reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), decode_reader.call_count);
    try expectFilled(&dest, 0x7b);
}

test "inference bridge stage_byte_harness readStageFrameBytes rejects destination buffer too small before receiver mutation" {
    var metadata = try testDecodeMetadata();
    const payload = [_]u8{0xf0} ** test_payload_len;
    var frame: [stage_frame_header.stage_frame_header_encoded_len + test_payload_len]u8 = undefined;
    _ = try encodeFrame(&metadata, frame[0..], &payload);

    var dest: [test_payload_len - 1]u8 = [_]u8{0x9c} ** (test_payload_len - 1);
    var reader = TestReader{ .source = frame[0..] };
    try std.testing.expectError(error.StagePayloadDestinationTooSmall, readStageFrameBytes(&reader, &metadata, dest[0..]));
    try std.testing.expectEqual(@as(usize, 1), reader.call_count);
    try expectFilled(&dest, 0x9c);
}
