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

    var buffer: [2048]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try xray.staged_frame.writeStagedFrameTsv(stream.writer(), &records);

    const expected =
        "frame_id\tboundary_index\tsource_stage_id\ttarget_stage_id\tproducer_layer_start\tproducer_layer_end\tconsumer_layer_start\tconsumer_layer_end\tstep_kind\tdtype\tlayout\trank\tshape0\tshape1\tshape2\tshape3\tpayload_byte_count\tbatch_index\trequest_id\tslot_id\tsequence_start\ttoken_count\tpayload_location\tbyte_image_readiness\ttransfer_mode\thost_readable\tremote_readable\tdevice_download_required\n" ++
        "100\t0\t0\t1\t0\t3\t3\t6\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tunknown\tunknown\tfalse\tfalse\tfalse\n" ++
        "101\t1\t1\t2\t3\t6\t6\t9\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tmetal:4\tunknown\tunknown\tfalse\tfalse\tfalse\n";
    try std.testing.expectEqualStrings(expected, stream.getWritten());
}

test "xray staged_frame writeStagedFrameTsv public API emits byte image readiness and transfer mode columns" {
    var records = [_]xray.StagedFrameRecord{
        record(0),
        record(1),
        record(2),
        record(3),
        record(4),
        record(5),
        record(6),
    };
    records[0].byte_image_readiness = xray.StagedFrameByteImageReadiness.unknown;
    records[0].transfer_mode = xray.StagedFrameTransferMode.unknown;
    records[1].byte_image_readiness = xray.StagedFrameByteImageReadiness.host_readable_now;
    records[1].transfer_mode = xray.StagedFrameTransferMode.borrow_in_process;
    records[1].host_readable = true;
    records[1].remote_readable = true;
    records[2].byte_image_readiness = xray.StagedFrameByteImageReadiness.producer_sync_required;
    records[2].transfer_mode = xray.StagedFrameTransferMode.copy_in_process;
    records[3].byte_image_readiness = xray.StagedFrameByteImageReadiness.device_download_required;
    records[3].transfer_mode = xray.StagedFrameTransferMode.device_download_then_copy;
    records[3].device_download_required = true;
    records[4].byte_image_readiness = xray.StagedFrameByteImageReadiness.device_download_required;
    records[4].transfer_mode = xray.StagedFrameTransferMode.device_peer_copy_in_process;
    records[4].device_download_required = true;
    records[5].byte_image_readiness = xray.StagedFrameByteImageReadiness.local_only_opaque;
    records[5].transfer_mode = xray.StagedFrameTransferMode.remote_stream;
    records[6].byte_image_readiness = xray.StagedFrameByteImageReadiness.host_readable_now;
    records[6].transfer_mode = xray.StagedFrameTransferMode.device_download_then_remote_stream;
    records[6].host_readable = true;
    records[6].remote_readable = true;

    var buffer: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    try xray.writeStagedFrameTsv(stream.writer(), &records);

    const expected =
        "frame_id\tboundary_index\tsource_stage_id\ttarget_stage_id\tproducer_layer_start\tproducer_layer_end\tconsumer_layer_start\tconsumer_layer_end\tstep_kind\tdtype\tlayout\trank\tshape0\tshape1\tshape2\tshape3\tpayload_byte_count\tbatch_index\trequest_id\tslot_id\tsequence_start\ttoken_count\tpayload_location\tbyte_image_readiness\ttransfer_mode\thost_readable\tremote_readable\tdevice_download_required\n" ++
        "100\t0\t0\t1\t0\t3\t3\t6\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tunknown\tunknown\tfalse\tfalse\tfalse\n" ++
        "101\t1\t1\t2\t3\t6\t6\t9\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\thost_readable_now\tborrow_in_process\ttrue\ttrue\tfalse\n" ++
        "102\t2\t2\t3\t6\t9\t9\t12\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tproducer_sync_required\tcopy_in_process\tfalse\tfalse\tfalse\n" ++
        "103\t3\t3\t4\t9\t12\t12\t15\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tdevice_download_required\tdevice_download_then_copy\tfalse\tfalse\ttrue\n" ++
        "104\t4\t4\t5\t12\t15\t15\t18\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tdevice_download_required\tdevice_peer_copy_in_process\tfalse\tfalse\ttrue\n" ++
        "105\t5\t5\t6\t15\t18\t18\t21\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\tlocal_only_opaque\tremote_stream\tfalse\tfalse\tfalse\n" ++
        "106\t6\t6\t7\t18\t21\t21\t24\tdecode\tf32\trow_major\t3\t1\t1\t16\t0\t64\t0\t11\t12\t13\t1\tnone\thost_readable_now\tdevice_download_then_remote_stream\ttrue\ttrue\tfalse\n";
    try std.testing.expectEqualStrings(expected, stream.getWritten());
}
