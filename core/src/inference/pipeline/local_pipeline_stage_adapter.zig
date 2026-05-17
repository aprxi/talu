//! Generic adapter from backend interface modules to local pipeline stage vtables.
//!
//! Concrete backends expose only self-owned layer execution plus external
//! activation input/output surfaces. This pipeline adapter erases those self-only
//! interfaces for `LocalPipelineRuntime`; it does not put source/target pairing
//! logic inside any backend module.

const std = @import("std");

const local_pipeline_runtime = @import("local_pipeline_runtime.zig");
const tensor_frame = @import("tensor_frame.zig");
const transport = @import("../transport/root.zig");
const runtime_contract = @import("runtime_contract_pkg");

const StageDecodeContext = struct {
    token: u32,
    position: usize,
    slot_index: usize,
    ensure_kv_capacity: bool,
};

fn stagePtr(comptime BackendType: type, ptr: *anyopaque) *BackendType {
    return @ptrCast(@alignCast(ptr));
}

fn stagePtrConst(comptime BackendType: type, ptr: *const anyopaque) *const BackendType {
    return @ptrCast(@alignCast(ptr));
}

fn descriptorHasField(comptime Descriptor: type, comptime field_name: []const u8) bool {
    return switch (@typeInfo(Descriptor)) {
        .@"struct" => @hasField(Descriptor, field_name),
        else => false,
    };
}

fn externalSourceFromBackendOutput(descriptor: anytype) !transport.ExternalActivationSource {
    const Descriptor = @TypeOf(descriptor);
    if (comptime descriptorHasField(Descriptor, "bytes")) {
        return .{ .host = descriptor.bytes };
    }
    if (comptime Descriptor == transport.CudaBufferDescriptor) {
        return .{ .cuda = descriptor };
    }
    return error.InvalidTopologyConfig;
}

fn externalTargetFromBackendInput(descriptor: anytype) !transport.ExternalActivationTarget {
    const Descriptor = @TypeOf(descriptor);
    if (comptime descriptorHasField(Descriptor, "bytes")) {
        return .{ .host = descriptor.bytes };
    }
    if (comptime Descriptor == transport.CudaBufferDescriptor) {
        return .{ .cuda = descriptor };
    }
    return error.InvalidTopologyConfig;
}

fn Adapter(comptime BackendModule: type, comptime BackendType: type) type {
    const stage_executor = BackendModule.interface.stage_executor;
    const transport_endpoint = BackendModule.interface.transport_endpoint;

    return struct {
        fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
            const backend = stagePtr(BackendType, ptr);
            backend.deinit();
            allocator.destroy(backend);
        }

        fn maxBatchSize(ptr: *const anyopaque) usize {
            return stage_executor.maxBatchSize(stagePtrConst(BackendType, ptr));
        }

        fn prefillChunkRowsCap(ptr: *const anyopaque) usize {
            return stage_executor.prefillChunkRowsCap(stagePtrConst(BackendType, ptr));
        }

        fn stateDescriptors(ptr: *const anyopaque) []const runtime_contract.StateDescriptor {
            return stagePtrConst(BackendType, ptr).stateDescriptors();
        }

        fn allocSlot(ptr: *anyopaque) ?usize {
            return stagePtr(BackendType, ptr).allocSlot();
        }

        fn freeSlot(ptr: *anyopaque, slot_index: usize) void {
            stagePtr(BackendType, ptr).freeSlot(slot_index);
        }

        fn resetSlot(ptr: *anyopaque, slot_index: usize) void {
            stagePtr(BackendType, ptr).resetSlot(slot_index);
        }

        fn bindSlotStateBlocks(
            ptr: *anyopaque,
            slot_index: usize,
            blocks: []const runtime_contract.StateBlockHandle,
        ) anyerror!void {
            try stagePtr(BackendType, ptr).bindSlotStateBlocks(slot_index, blocks);
        }

        fn unbindSlotStateBlocks(ptr: *anyopaque, slot_index: usize) void {
            stagePtr(BackendType, ptr).unbindSlotStateBlocks(slot_index);
        }

        fn executeDecode(ptr: *anyopaque, request: local_pipeline_runtime.StageDecodeRequest) anyerror!void {
            const context = StageDecodeContext{
                .token = request.token,
                .position = request.position,
                .slot_index = request.slot_index,
                .ensure_kv_capacity = request.ensure_kv_capacity,
            };
            try stage_executor.executeDecodeLayerRange(
                stagePtr(BackendType, ptr),
                context,
                request.layer_start,
                request.layer_end,
                request.logits_out_opt,
                request.compute_logits,
                request.download_logits,
                request.use_preloaded_input,
            );
        }

        fn executePrefill(ptr: *anyopaque, request: local_pipeline_runtime.StagePrefillRequest) anyerror!void {
            try stage_executor.executePrefillLayerRange(
                stagePtr(BackendType, ptr),
                request.slot_index,
                request.tokens,
                request.sequence_start,
                request.layer_start,
                request.layer_end,
                request.use_preloaded_input,
                request.compute_logits,
                request.logits_out_opt,
                request.source_embeddings_out,
            );
        }

        fn synchronize(ptr: *anyopaque, _: transport.StageExecutionReceipt) anyerror!void {
            switch (comptime stage_executor.backendKind()) {
                .cuda => try stagePtr(BackendType, ptr).synchronize(),
                else => {},
            }
        }

        fn decodeExternalOutput(ptr: *anyopaque, slot_index: usize, byte_count: usize) anyerror!transport.ExternalActivationSource {
            return externalSourceFromBackendOutput(
                try transport_endpoint.decodeExternalOutput(stagePtr(BackendType, ptr), slot_index, byte_count),
            );
        }

        fn prefillExternalOutput(ptr: *anyopaque, byte_count: usize) anyerror!transport.ExternalActivationSource {
            return externalSourceFromBackendOutput(
                try transport_endpoint.prefillExternalOutput(stagePtr(BackendType, ptr), byte_count),
            );
        }

        fn decodeExternalInput(ptr: *anyopaque, slot_index: usize, byte_count: usize) anyerror!transport.ExternalActivationTarget {
            return externalTargetFromBackendInput(
                try transport_endpoint.decodeExternalInput(stagePtr(BackendType, ptr), slot_index, byte_count),
            );
        }

        fn prefillExternalInput(ptr: *anyopaque, byte_count: usize) anyerror!transport.ExternalActivationTarget {
            return externalTargetFromBackendInput(
                try transport_endpoint.prefillExternalInput(stagePtr(BackendType, ptr), byte_count),
            );
        }

        fn deviceLocationHint(ptr: *const anyopaque) ?tensor_frame.TensorFramePayloadLocationHint {
            return transport_endpoint.deviceLocationHint(stagePtrConst(BackendType, ptr)) catch null;
        }

        const vtable = local_pipeline_runtime.StageVTable{
            .deinit = deinit,
            .max_batch_size = maxBatchSize,
            .prefill_chunk_rows_cap = prefillChunkRowsCap,
            .state_descriptors = stateDescriptors,
            .alloc_slot = allocSlot,
            .free_slot = freeSlot,
            .reset_slot = resetSlot,
            .bind_slot_state_blocks = bindSlotStateBlocks,
            .unbind_slot_state_blocks = unbindSlotStateBlocks,
            .execute_decode = executeDecode,
            .execute_prefill = executePrefill,
            .synchronize = synchronize,
            .decode_external_output = decodeExternalOutput,
            .prefill_external_output = prefillExternalOutput,
            .decode_external_input = decodeExternalInput,
            .prefill_external_input = prefillExternalInput,
            .device_location_hint = deviceLocationHint,
        };
    };
}

pub fn stageVTable(comptime BackendModule: type, comptime BackendType: type) *const local_pipeline_runtime.StageVTable {
    return &Adapter(BackendModule, BackendType).vtable;
}

pub fn canPeerCopy(
    source: *const local_pipeline_runtime.StageHandle,
    target: *const local_pipeline_runtime.StageHandle,
) bool {
    const source_surface = source.vtable.prefill_external_output(source.ptr, 1) catch return false;
    const target_surface = target.vtable.prefill_external_input(target.ptr, 1) catch return false;
    return transport.canPeerCopyExternalActivation(source_surface, target_surface);
}
