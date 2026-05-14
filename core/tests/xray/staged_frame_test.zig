//! Integration coverage for xray staged-frame capture exports.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

fn record(boundary_index: usize) xray.staged_frame.StagedFrameRecord {
    return .{
        .graph_digest = [_]u8{7} ** 32,
        .graph_contract_version = 1,
        .stage_plan_contract_version = 2,
        .stage_plan_id_digest = [_]u8{9} ** 32,
        .frame_id = 100 + @as(u64, @intCast(boundary_index)),
        .boundary_index = boundary_index,
        .source_stage_id = boundary_index,
        .target_stage_id = boundary_index + 1,
        .producer_layer_start = boundary_index * 3,
        .producer_layer_end = (boundary_index + 1) * 3,
        .consumer_layer_start = (boundary_index + 1) * 3,
        .consumer_layer_end = (boundary_index + 2) * 3,
        .step_kind = .decode,
        .dtype = .f32,
        .layout = .row_major,
        .shape = .{ 1, 1, 16, 0 },
        .rank = 3,
        .payload_byte_count = 64,
        .batch_index = 0,
        .request_id = 11,
        .slot_id = 12,
        .sequence_start = 13,
        .token_count = 1,
        .payload_location = .none,
    };
}

test "xray staged_frame StagedFrameCapture.init append count get records clear deinit public API stores records through xray root export" {
    var capture = xray.staged_frame.StagedFrameCapture.init(std.testing.allocator);
    defer capture.deinit();

    var source_record = record(0);
    try capture.append(source_record);
    source_record.request_id = 99;

    try std.testing.expectEqual(@as(usize, 1), capture.count());
    try std.testing.expectEqual(@as(usize, 1), capture.records().len);
    try std.testing.expect(capture.get(1) == null);
    try std.testing.expectEqual(@as(u64, 11), capture.get(0).?.request_id);

    capture.clear();
    try std.testing.expectEqual(@as(usize, 0), capture.count());
}

test "xray staged_frame validateAdjacentBoundarySequence public API accepts multi boundary sequence" {
    var records = [_]xray.staged_frame.StagedFrameRecord{
        record(0),
        record(0),
        record(1),
        record(1),
        record(2),
    };
    records[1].batch_index = 1;
    records[1].request_id = 21;
    records[3].batch_index = 1;
    records[3].request_id = 21;
    records[4].payload_location = .{ .opaque_local = 17 };

    try xray.staged_frame.validateAdjacentBoundarySequence(&records);
}

test "xray staged_frame writeStagedFrameTsv public API emits stable report" {
    var records = [_]xray.staged_frame.StagedFrameRecord{
        record(0),
        record(1),
    };
    records[1].payload_location = .{ .metal = 4 };

    var buffer: [1024]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try xray.staged_frame.writeStagedFrameTsv(stream.writer(), &records);

    const expected =
        "frame_id\tboundary_index\tsource_stage_id\ttarget_stage_id\tproducer_layer_start\tproducer_layer_end\tconsumer_layer_start\tconsumer_layer_end\tstep_kind\tdtype\tlayout\trank\tshape0\tshape1\tshape2\tshape3\tpayload_byte_count\tbatch_index\trequest_id\tslot_id\tsequence_start\ttoken_count\tpayload_location\n" ++
        "100\t0\t0\t1\t0\t3\t3\t6\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\n" ++
        "101\t1\t1\t2\t3\t6\t6\t9\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tmetal:4\n";
    try std.testing.expectEqualStrings(expected, stream.getWritten());
}
