//! Tests for bridge-owned local pipeline runtime lifecycle fanout.

const std = @import("std");
const main = @import("main");

const bridge = main.inference.bridge;
const transport = main.inference.transport;
const runtime_contract = main.inference.runtime_contract;

const LocalPipelineRuntime = bridge.LocalPipelineRuntime;
const StageDecodeRequest = bridge.local_pipeline_runtime.StageDecodeRequest;
const StageHandle = bridge.LocalPipelineStageHandle;
const StagePrefillRequest = bridge.local_pipeline_runtime.StagePrefillRequest;
const StageVTable = bridge.LocalPipelineStageVTable;

const MockPipelineStage = struct {
    descriptors: []const runtime_contract.StateDescriptor = &.{},
    max_batch_size_value: usize = 1,
    prefill_chunk_rows_cap_value: usize = 1,
    alloc_result: ?usize = 0,
    bind_fails: bool = false,
    alloc_count: usize = 0,
    free_count: usize = 0,
    last_free_slot: ?usize = null,
    reset_count: usize = 0,
    bind_count: usize = 0,
    last_bind_slot: ?usize = null,
    last_bound_count: usize = 0,
    last_bound_ids: [4]u8 = [_]u8{0} ** 4,
    unbind_count: usize = 0,
    last_unbind_slot: ?usize = null,

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        _ = ptr;
        _ = allocator;
    }

    fn maxBatchSize(ptr: *const anyopaque) usize {
        const self: *const MockPipelineStage = @ptrCast(@alignCast(ptr));
        return self.max_batch_size_value;
    }

    fn prefillChunkRowsCap(ptr: *const anyopaque) usize {
        const self: *const MockPipelineStage = @ptrCast(@alignCast(ptr));
        return self.prefill_chunk_rows_cap_value;
    }

    fn stateDescriptors(ptr: *const anyopaque) []const runtime_contract.StateDescriptor {
        const self: *const MockPipelineStage = @ptrCast(@alignCast(ptr));
        return self.descriptors;
    }

    fn allocSlot(ptr: *anyopaque) ?usize {
        const self: *MockPipelineStage = @ptrCast(@alignCast(ptr));
        self.alloc_count += 1;
        return self.alloc_result;
    }

    fn freeSlot(ptr: *anyopaque, slot_index: usize) void {
        const self: *MockPipelineStage = @ptrCast(@alignCast(ptr));
        self.free_count += 1;
        self.last_free_slot = slot_index;
    }

    fn resetSlot(ptr: *anyopaque, slot_index: usize) void {
        const self: *MockPipelineStage = @ptrCast(@alignCast(ptr));
        _ = slot_index;
        self.reset_count += 1;
    }

    fn bindSlotStateBlocks(
        ptr: *anyopaque,
        slot_index: usize,
        blocks: []const runtime_contract.StateBlockHandle,
    ) anyerror!void {
        const self: *MockPipelineStage = @ptrCast(@alignCast(ptr));
        if (blocks.len > self.last_bound_ids.len) return error.InvalidStateDescriptorBinding;
        self.bind_count += 1;
        self.last_bind_slot = slot_index;
        self.last_bound_count = blocks.len;
        for (blocks, 0..) |block, index| self.last_bound_ids[index] = block.id;
        if (self.bind_fails) return error.MockStageBindFailed;
    }

    fn unbindSlotStateBlocks(ptr: *anyopaque, slot_index: usize) void {
        const self: *MockPipelineStage = @ptrCast(@alignCast(ptr));
        self.unbind_count += 1;
        self.last_unbind_slot = slot_index;
    }

    fn executeDecode(ptr: *anyopaque, request: StageDecodeRequest) anyerror!void {
        _ = ptr;
        _ = request;
        return error.UnexpectedMockStageCall;
    }

    fn executePrefill(ptr: *anyopaque, request: StagePrefillRequest) anyerror!void {
        _ = ptr;
        _ = request;
        return error.UnexpectedMockStageCall;
    }

    fn synchronize(ptr: *anyopaque, receipt: transport.StageExecutionReceipt) anyerror!void {
        _ = ptr;
        _ = receipt;
    }

    fn downloadDecodeActivation(ptr: *anyopaque, slot_index: usize, host_buf: []u8, byte_count: usize) anyerror!void {
        _ = ptr;
        _ = slot_index;
        _ = host_buf;
        _ = byte_count;
        return error.UnexpectedMockStageCall;
    }

    fn downloadPrefillActivation(ptr: *anyopaque, host_buf: []u8, byte_count: usize) anyerror!void {
        _ = ptr;
        _ = host_buf;
        _ = byte_count;
        return error.UnexpectedMockStageCall;
    }

    fn uploadDecodeActivation(ptr: *anyopaque, slot_index: usize, host_buf: []const u8, byte_count: usize) anyerror!void {
        _ = ptr;
        _ = slot_index;
        _ = host_buf;
        _ = byte_count;
        return error.UnexpectedMockStageCall;
    }

    fn uploadPrefillActivation(ptr: *anyopaque, host_buf: []const u8, byte_count: usize) anyerror!void {
        _ = ptr;
        _ = host_buf;
        _ = byte_count;
        return error.UnexpectedMockStageCall;
    }

    fn hostDecodeActivation(ptr: *anyopaque, slot_index: usize, byte_count: usize) anyerror![]const u8 {
        _ = ptr;
        _ = slot_index;
        _ = byte_count;
        return &.{};
    }

    fn hostPrefillActivation(ptr: *anyopaque, byte_count: usize) anyerror![]const u8 {
        _ = ptr;
        _ = byte_count;
        return &.{};
    }

    fn deviceLocationHint(ptr: *const anyopaque) ?bridge.TensorFramePayloadLocationHint {
        _ = ptr;
        return null;
    }
};

const mock_pipeline_stage_vtable = StageVTable{
    .deinit = MockPipelineStage.deinit,
    .max_batch_size = MockPipelineStage.maxBatchSize,
    .prefill_chunk_rows_cap = MockPipelineStage.prefillChunkRowsCap,
    .state_descriptors = MockPipelineStage.stateDescriptors,
    .alloc_slot = MockPipelineStage.allocSlot,
    .free_slot = MockPipelineStage.freeSlot,
    .reset_slot = MockPipelineStage.resetSlot,
    .bind_slot_state_blocks = MockPipelineStage.bindSlotStateBlocks,
    .unbind_slot_state_blocks = MockPipelineStage.unbindSlotStateBlocks,
    .execute_decode = MockPipelineStage.executeDecode,
    .execute_prefill = MockPipelineStage.executePrefill,
    .synchronize = MockPipelineStage.synchronize,
    .download_decode_activation = MockPipelineStage.downloadDecodeActivation,
    .download_prefill_activation = MockPipelineStage.downloadPrefillActivation,
    .upload_decode_activation = MockPipelineStage.uploadDecodeActivation,
    .upload_prefill_activation = MockPipelineStage.uploadPrefillActivation,
    .host_decode_activation = MockPipelineStage.hostDecodeActivation,
    .host_prefill_activation = MockPipelineStage.hostPrefillActivation,
    .device_location_hint = MockPipelineStage.deviceLocationHint,
};

fn mockStageHandle(
    stage_id: usize,
    backend_kind: bridge.HostBackendKind,
    layer_start: usize,
    layer_end: usize,
    stage: *MockPipelineStage,
) StageHandle {
    return .{
        .stage_id = stage_id,
        .backend_kind = backend_kind,
        .layer_start = layer_start,
        .layer_end = layer_end,
        .supported_boundary_dtypes = &.{.f32},
        .ptr = stage,
        .vtable = &mock_pipeline_stage_vtable,
    };
}

fn mockRuntime(
    stages: []StageHandle,
    slot_in_use: []bool,
    slot_positions: []usize,
    slot_request_ids: []?u64,
    slot_logits: []f32,
) LocalPipelineRuntime {
    return .{
        .allocator = std.testing.allocator,
        .loaded = @ptrFromInt(0x1000),
        .d_model = 1,
        .vocab_size = 1,
        .max_batch_size = slot_in_use.len,
        .prefill_chunk_rows_cap = 1,
        .stages = stages,
        .runtime_plan = .{ .allocator = std.testing.allocator },
        .contracts = .{},
        .slot_in_use = slot_in_use,
        .slot_positions = slot_positions,
        .slot_request_ids = slot_request_ids,
        .slot_logits = slot_logits,
    };
}

test "LocalPipelineRuntime allocSlot rolls back earlier stages when a later stage fails" {
    var stage0 = MockPipelineStage{ .alloc_result = 0 };
    var stage1 = MockPipelineStage{ .alloc_result = null };
    var stage2 = MockPipelineStage{ .alloc_result = 0 };
    var handles = [_]StageHandle{
        mockStageHandle(0, .cpu, 0, 1, &stage0),
        mockStageHandle(1, .cuda, 1, 2, &stage1),
        mockStageHandle(2, .cpu, 2, 3, &stage2),
    };
    var slot_in_use = [_]bool{false};
    var slot_positions = [_]usize{0};
    var slot_request_ids = [_]?u64{null};
    var slot_logits = [_]f32{0};
    var runtime = mockRuntime(handles[0..], slot_in_use[0..], slot_positions[0..], slot_request_ids[0..], slot_logits[0..]);

    try std.testing.expectEqual(@as(?usize, null), runtime.allocSlot());
    try std.testing.expectEqual(@as(usize, 1), stage0.alloc_count);
    try std.testing.expectEqual(@as(usize, 1), stage0.free_count);
    try std.testing.expectEqual(@as(?usize, 0), stage0.last_free_slot);
    try std.testing.expectEqual(@as(usize, 2), stage0.unbind_count);
    try std.testing.expectEqual(@as(usize, 1), stage1.alloc_count);
    try std.testing.expectEqual(@as(usize, 0), stage1.free_count);
    try std.testing.expectEqual(@as(usize, 0), stage2.alloc_count);
    try std.testing.expect(!slot_in_use[0]);
}

test "LocalPipelineRuntime state binding validates union and forwards only relevant blocks" {
    const descriptor0 = runtime_contract.StateDescriptor{ .id = 1, .size_bytes = 8, .align_bytes = 8, .zero_init = false, .lifecycle = .slot_persistent, .runtime_kind = 0 };
    const descriptor1 = runtime_contract.StateDescriptor{ .id = 2, .size_bytes = 8, .align_bytes = 8, .zero_init = false, .lifecycle = .slot_persistent, .runtime_kind = 0 };
    const descriptors0 = [_]runtime_contract.StateDescriptor{descriptor0};
    const descriptors1 = [_]runtime_contract.StateDescriptor{descriptor1};
    var stage0 = MockPipelineStage{ .descriptors = &descriptors0 };
    var stage1 = MockPipelineStage{ .descriptors = &descriptors1 };
    var handles = [_]StageHandle{
        mockStageHandle(0, .cpu, 0, 1, &stage0),
        mockStageHandle(1, .cuda, 1, 2, &stage1),
    };
    var slot_in_use = [_]bool{true};
    var slot_positions = [_]usize{0};
    var slot_request_ids = [_]?u64{101};
    var slot_logits = [_]f32{0};
    var runtime = mockRuntime(handles[0..], slot_in_use[0..], slot_positions[0..], slot_request_ids[0..], slot_logits[0..]);
    runtime.state_descriptors_storage[0] = descriptor0;
    runtime.state_descriptors_storage[1] = descriptor1;
    runtime.state_descriptor_count = 2;

    var block0_storage: [8]u8 align(64) = undefined;
    var block1_storage: [8]u8 align(64) = undefined;
    const blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 1, .ptr = block0_storage[0..].ptr, .size = block0_storage.len, .align_bytes = 64 },
        .{ .id = 2, .ptr = block1_storage[0..].ptr, .size = block1_storage.len, .align_bytes = 64 },
    };

    try runtime.bindSlotStateBlocks(0, &blocks);

    try std.testing.expectEqual(@as(usize, 2), runtime.stateDescriptors().len);
    try std.testing.expectEqual(@as(usize, 1), stage0.bind_count);
    try std.testing.expectEqual(@as(?usize, 0), stage0.last_bind_slot);
    try std.testing.expectEqual(@as(usize, 1), stage0.last_bound_count);
    try std.testing.expectEqual(@as(u8, 1), stage0.last_bound_ids[0]);
    try std.testing.expectEqual(@as(usize, 1), stage1.bind_count);
    try std.testing.expectEqual(@as(usize, 1), stage1.last_bound_count);
    try std.testing.expectEqual(@as(u8, 2), stage1.last_bound_ids[0]);
}

test "LocalPipelineRuntime bindSlotStateBlocks rolls back earlier stages on bind failure" {
    const descriptor0 = runtime_contract.StateDescriptor{ .id = 1, .size_bytes = 8, .align_bytes = 8, .zero_init = false, .lifecycle = .slot_persistent, .runtime_kind = 0 };
    const descriptor1 = runtime_contract.StateDescriptor{ .id = 2, .size_bytes = 8, .align_bytes = 8, .zero_init = false, .lifecycle = .slot_persistent, .runtime_kind = 0 };
    const descriptors0 = [_]runtime_contract.StateDescriptor{descriptor0};
    const descriptors1 = [_]runtime_contract.StateDescriptor{descriptor1};
    var stage0 = MockPipelineStage{ .descriptors = &descriptors0 };
    var stage1 = MockPipelineStage{ .descriptors = &descriptors1, .bind_fails = true };
    var handles = [_]StageHandle{
        mockStageHandle(0, .cpu, 0, 1, &stage0),
        mockStageHandle(1, .cuda, 1, 2, &stage1),
    };
    var slot_in_use = [_]bool{true};
    var slot_positions = [_]usize{0};
    var slot_request_ids = [_]?u64{101};
    var slot_logits = [_]f32{0};
    var runtime = mockRuntime(handles[0..], slot_in_use[0..], slot_positions[0..], slot_request_ids[0..], slot_logits[0..]);
    runtime.state_descriptors_storage[0] = descriptor0;
    runtime.state_descriptors_storage[1] = descriptor1;
    runtime.state_descriptor_count = 2;

    var block0_storage: [8]u8 align(64) = undefined;
    var block1_storage: [8]u8 align(64) = undefined;
    const blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 1, .ptr = block0_storage[0..].ptr, .size = block0_storage.len, .align_bytes = 64 },
        .{ .id = 2, .ptr = block1_storage[0..].ptr, .size = block1_storage.len, .align_bytes = 64 },
    };

    try std.testing.expectError(error.MockStageBindFailed, runtime.bindSlotStateBlocks(0, &blocks));
    try std.testing.expectEqual(@as(usize, 1), stage0.bind_count);
    try std.testing.expectEqual(@as(usize, 1), stage0.unbind_count);
    try std.testing.expectEqual(@as(?usize, 0), stage0.last_unbind_slot);
    try std.testing.expectEqual(@as(usize, 1), stage1.bind_count);
    try std.testing.expectEqual(@as(usize, 1), stage1.last_bound_count);
    try std.testing.expectEqual(@as(u8, 2), stage1.last_bound_ids[0]);
}
