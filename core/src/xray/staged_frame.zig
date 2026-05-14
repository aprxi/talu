//! Xray-owned staged-frame metadata capture.
//!
//! This module stores copied boundary metadata facts only. It has no dependency
//! on runtime bridge contracts and does not inspect tensor payload bytes.

const std = @import("std");

pub const StagedFrameStepKind = enum(u8) {
    prefill,
    decode,
};

pub const StagedFrameDType = enum(u8) {
    bf16,
    f16,
    f32,
};

pub const StagedFrameLayout = enum(u8) {
    row_major,
};

pub const StagedFramePayloadLocation = union(enum) {
    none,
    cpu,
    cuda: u16,
    metal: u16,
    opaque_local: u32,
};

pub const StagedFrameRecord = struct {
    graph_digest: [32]u8,
    graph_contract_version: u32,
    stage_plan_contract_version: u32,
    stage_plan_id_digest: [32]u8,
    frame_id: u64,
    boundary_index: usize,
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
    step_kind: StagedFrameStepKind,
    dtype: StagedFrameDType,
    layout: StagedFrameLayout,
    shape: [4]u64,
    rank: u8,
    payload_byte_count: u64,
    batch_index: u32,
    request_id: u64,
    slot_id: u64,
    sequence_start: u64,
    token_count: u64,
    payload_location: StagedFramePayloadLocation,
};

pub const StagedFrameCapture = struct {
    allocator: std.mem.Allocator,
    stored_records: std.ArrayList(StagedFrameRecord),

    pub fn init(allocator: std.mem.Allocator) StagedFrameCapture {
        return .{
            .allocator = allocator,
            .stored_records = .empty,
        };
    }

    pub fn deinit(self: *StagedFrameCapture) void {
        self.stored_records.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn clear(self: *StagedFrameCapture) void {
        self.stored_records.clearRetainingCapacity();
    }

    pub fn append(
        self: *StagedFrameCapture,
        record: StagedFrameRecord,
    ) std.mem.Allocator.Error!void {
        try self.stored_records.append(self.allocator, record);
    }

    pub fn count(self: *const StagedFrameCapture) usize {
        return self.stored_records.items.len;
    }

    pub fn get(self: *const StagedFrameCapture, index: usize) ?*const StagedFrameRecord {
        if (index >= self.stored_records.items.len) return null;
        return &self.stored_records.items[index];
    }

    pub fn records(self: *const StagedFrameCapture) []const StagedFrameRecord {
        return self.stored_records.items;
    }
};

pub const StagedFrameSequenceError = error{
    FrameCountMismatch,
    FrameMismatch,
    InconsistentBoundaryGroup,
    NonAdjacentBoundaryOrder,
    NonMonotonicBoundaryIndex,
};

pub fn stagedFrameRecordEql(lhs: StagedFrameRecord, rhs: StagedFrameRecord) bool {
    return std.mem.eql(u8, &lhs.graph_digest, &rhs.graph_digest) and
        lhs.graph_contract_version == rhs.graph_contract_version and
        lhs.stage_plan_contract_version == rhs.stage_plan_contract_version and
        std.mem.eql(u8, &lhs.stage_plan_id_digest, &rhs.stage_plan_id_digest) and
        lhs.frame_id == rhs.frame_id and
        lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.producer_layer_start == rhs.producer_layer_start and
        lhs.producer_layer_end == rhs.producer_layer_end and
        lhs.consumer_layer_start == rhs.consumer_layer_start and
        lhs.consumer_layer_end == rhs.consumer_layer_end and
        lhs.step_kind == rhs.step_kind and
        lhs.dtype == rhs.dtype and
        lhs.layout == rhs.layout and
        lhs.shape == rhs.shape and
        lhs.rank == rhs.rank and
        lhs.payload_byte_count == rhs.payload_byte_count and
        lhs.batch_index == rhs.batch_index and
        lhs.request_id == rhs.request_id and
        lhs.slot_id == rhs.slot_id and
        lhs.sequence_start == rhs.sequence_start and
        lhs.token_count == rhs.token_count and
        payloadLocationEql(lhs.payload_location, rhs.payload_location);
}

pub fn validateExpectedSequence(
    actual: []const StagedFrameRecord,
    expected: []const StagedFrameRecord,
) StagedFrameSequenceError!void {
    if (actual.len != expected.len) return error.FrameCountMismatch;
    for (actual, expected) |actual_record, expected_record| {
        if (!stagedFrameRecordEql(actual_record, expected_record)) return error.FrameMismatch;
    }
}

pub fn validateAdjacentBoundarySequence(
    records_slice: []const StagedFrameRecord,
) StagedFrameSequenceError!void {
    if (records_slice.len <= 1) return;

    var group_start: usize = 0;
    var previous_group: ?StagedFrameRecord = null;
    while (group_start < records_slice.len) {
        const group_record = records_slice[group_start];
        var group_end = group_start + 1;
        while (group_end < records_slice.len and records_slice[group_end].boundary_index == group_record.boundary_index) {
            if (!boundaryFactsEql(group_record, records_slice[group_end])) return error.InconsistentBoundaryGroup;
            group_end += 1;
        }

        if (previous_group) |previous| {
            if (group_record.boundary_index <= previous.boundary_index) return error.NonMonotonicBoundaryIndex;
            if (group_record.boundary_index != previous.boundary_index + 1) return error.NonAdjacentBoundaryOrder;
            if (group_record.source_stage_id != previous.target_stage_id or
                group_record.producer_layer_start != previous.consumer_layer_start or
                group_record.producer_layer_end != previous.consumer_layer_end)
            {
                return error.NonAdjacentBoundaryOrder;
            }
        }

        previous_group = group_record;
        group_start = group_end;
    }
}

pub fn writeStagedFrameTsv(
    writer: anytype,
    records_slice: []const StagedFrameRecord,
) !void {
    try writer.writeAll("frame_id\tboundary_index\tsource_stage_id\ttarget_stage_id\tproducer_layer_start\tproducer_layer_end\tconsumer_layer_start\tconsumer_layer_end\tstep_kind\tdtype\tlayout\trank\tshape0\tshape1\tshape2\tshape3\tpayload_byte_count\tbatch_index\trequest_id\tslot_id\tsequence_start\ttoken_count\tpayload_location\n");
    for (records_slice) |record| {
        try writer.print(
            "{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{s}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\t",
            .{
                record.frame_id,
                record.boundary_index,
                record.source_stage_id,
                record.target_stage_id,
                record.producer_layer_start,
                record.producer_layer_end,
                record.consumer_layer_start,
                record.consumer_layer_end,
                stepKindName(record.step_kind),
                dtypeName(record.dtype),
                layoutName(record.layout),
                record.rank,
                record.shape[0],
                record.shape[1],
                record.shape[2],
                record.shape[3],
                record.payload_byte_count,
                record.batch_index,
                record.request_id,
                record.slot_id,
                record.sequence_start,
                record.token_count,
            },
        );
        try writePayloadLocation(writer, record.payload_location);
        try writer.writeByte('\n');
    }
}

fn payloadLocationEql(lhs: StagedFramePayloadLocation, rhs: StagedFramePayloadLocation) bool {
    return switch (lhs) {
        .none => rhs == .none,
        .cpu => rhs == .cpu,
        .cuda => |lhs_ordinal| rhs == .cuda and rhs.cuda == lhs_ordinal,
        .metal => |lhs_ordinal| rhs == .metal and rhs.metal == lhs_ordinal,
        .opaque_local => |lhs_value| rhs == .opaque_local and rhs.opaque_local == lhs_value,
    };
}

fn boundaryFactsEql(lhs: StagedFrameRecord, rhs: StagedFrameRecord) bool {
    return lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.producer_layer_start == rhs.producer_layer_start and
        lhs.producer_layer_end == rhs.producer_layer_end and
        lhs.consumer_layer_start == rhs.consumer_layer_start and
        lhs.consumer_layer_end == rhs.consumer_layer_end;
}

fn stepKindName(kind: StagedFrameStepKind) []const u8 {
    return switch (kind) {
        .prefill => "prefill",
        .decode => "decode",
    };
}

fn dtypeName(dtype: StagedFrameDType) []const u8 {
    return switch (dtype) {
        .bf16 => "bf16",
        .f16 => "f16",
        .f32 => "f32",
    };
}

fn layoutName(layout: StagedFrameLayout) []const u8 {
    return switch (layout) {
        .row_major => "row_major",
    };
}

fn writePayloadLocation(writer: anytype, location: StagedFramePayloadLocation) !void {
    switch (location) {
        .none => try writer.writeAll("none"),
        .cpu => try writer.writeAll("cpu"),
        .cuda => |ordinal| try writer.print("cuda:{d}", .{ordinal}),
        .metal => |ordinal| try writer.print("metal:{d}", .{ordinal}),
        .opaque_local => |value| try writer.print("opaque_local:{d}", .{value}),
    }
}

fn testRecord(boundary_index: usize) StagedFrameRecord {
    return .{
        .graph_digest = [_]u8{1} ** 32,
        .graph_contract_version = 2,
        .stage_plan_contract_version = 3,
        .stage_plan_id_digest = [_]u8{4} ** 32,
        .frame_id = 10 + boundary_index,
        .boundary_index = boundary_index,
        .source_stage_id = boundary_index,
        .target_stage_id = boundary_index + 1,
        .producer_layer_start = boundary_index * 4,
        .producer_layer_end = (boundary_index + 1) * 4,
        .consumer_layer_start = (boundary_index + 1) * 4,
        .consumer_layer_end = (boundary_index + 2) * 4,
        .step_kind = .decode,
        .dtype = .f32,
        .layout = .row_major,
        .shape = .{ 1, 1, 8, 0 },
        .rank = 3,
        .payload_byte_count = 32,
        .batch_index = 0,
        .request_id = 100,
        .slot_id = 7,
        .sequence_start = 12,
        .token_count = 1,
        .payload_location = .none,
    };
}

test "xray staged_frame StagedFrameCapture.init append count get records clear deinit stores copied records" {
    var capture = StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    var record = testRecord(0);
    try capture.append(record);
    record.request_id = 999;

    try std.testing.expectEqual(@as(usize, 1), capture.count());
    try std.testing.expect(capture.get(1) == null);
    const stored = capture.get(0).?;
    try std.testing.expectEqual(@as(u64, 100), stored.request_id);
    try std.testing.expectEqual(@as(usize, 1), capture.records().len);

    capture.clear();
    try std.testing.expectEqual(@as(usize, 0), capture.count());
}

test "xray staged_frame stagedFrameRecordEql compares all record facts" {
    const base = testRecord(0);
    try std.testing.expect(stagedFrameRecordEql(base, base));

    var changed = base;
    changed.graph_digest[0] ^= 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.graph_contract_version += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.stage_plan_contract_version += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.stage_plan_id_digest[0] ^= 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.frame_id += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.boundary_index += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.source_stage_id += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.target_stage_id += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.producer_layer_start += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.producer_layer_end += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.consumer_layer_start += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.consumer_layer_end += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.step_kind = .prefill;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.dtype = .f16;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.shape[0] += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.rank += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.payload_byte_count += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.batch_index += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.request_id += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.slot_id += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.sequence_start += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.token_count += 1;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
    changed = base;
    changed.payload_location = .cpu;
    try std.testing.expect(!stagedFrameRecordEql(base, changed));
}

test "xray staged_frame validateExpectedSequence accepts exact records and rejects mismatch" {
    const expected = [_]StagedFrameRecord{ testRecord(0), testRecord(1) };
    const actual = expected;
    try validateExpectedSequence(&actual, &expected);

    try std.testing.expectError(error.FrameCountMismatch, validateExpectedSequence(actual[0..1], &expected));
    var mismatched = actual;
    mismatched[1].slot_id += 1;
    try std.testing.expectError(error.FrameMismatch, validateExpectedSequence(&mismatched, &expected));
}

test "xray staged_frame validateAdjacentBoundarySequence groups multi batch records by boundary" {
    var records_buf = [_]StagedFrameRecord{
        testRecord(0),
        testRecord(0),
        testRecord(1),
        testRecord(1),
    };
    records_buf[1].batch_index = 1;
    records_buf[1].request_id = 101;
    records_buf[2].payload_location = .{ .cuda = 3 };
    records_buf[3].batch_index = 1;
    records_buf[3].request_id = 101;
    try validateAdjacentBoundarySequence(&records_buf);
}

test "xray staged_frame validateAdjacentBoundarySequence rejects non adjacent boundary sequence" {
    const skipped = [_]StagedFrameRecord{ testRecord(0), testRecord(2) };
    try std.testing.expectError(error.NonAdjacentBoundaryOrder, validateAdjacentBoundarySequence(&skipped));

    const reused = [_]StagedFrameRecord{ testRecord(0), testRecord(1), testRecord(0) };
    try std.testing.expectError(error.NonMonotonicBoundaryIndex, validateAdjacentBoundarySequence(&reused));

    var inconsistent = [_]StagedFrameRecord{ testRecord(0), testRecord(0) };
    inconsistent[1].target_stage_id += 1;
    try std.testing.expectError(error.InconsistentBoundaryGroup, validateAdjacentBoundarySequence(&inconsistent));
}

test "xray staged_frame writeStagedFrameTsv writes deterministic tsv report" {
    var records_buf = [_]StagedFrameRecord{ testRecord(0), testRecord(1) };
    records_buf[1].payload_location = .{ .cuda = 3 };

    var buffer: [1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try writeStagedFrameTsv(stream.writer(), &records_buf);

    const expected =
        "frame_id\tboundary_index\tsource_stage_id\ttarget_stage_id\tproducer_layer_start\tproducer_layer_end\tconsumer_layer_start\tconsumer_layer_end\tstep_kind\tdtype\tlayout\trank\tshape0\tshape1\tshape2\tshape3\tpayload_byte_count\tbatch_index\trequest_id\tslot_id\tsequence_start\ttoken_count\tpayload_location\n" ++
        "10\t0\t0\t1\t0\t4\t4\t8\tdecode\tf32\trow_major\t3\t1\t1\t8\t0\t32\t0\t100\t7\t12\t1\tnone\n" ++
        "11\t1\t1\t2\t4\t8\t8\t12\tdecode\tf32\trow_major\t3\t1\t1\t8\t0\t32\t0\t100\t7\t12\t1\tcuda:3\n";
    try std.testing.expectEqualStrings(expected, stream.getWritten());
}
