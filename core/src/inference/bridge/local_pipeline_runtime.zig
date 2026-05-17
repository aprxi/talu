//! Bridge-owned local pipeline runtime.
//!
//! Concrete backends provide erased stage handles. This runtime owns the
//! ordered stage list, bridge contracts, slot/request bookkeeping, and
//! activation handoff requests. Backends execute only their configured layer
//! ranges and expose transfer descriptors/callbacks through the stage vtable.

const std = @import("std");

const local_pipeline = @import("local_pipeline.zig");
const local_stage_contract = @import("local_stage_contract.zig");
const local_stage_runner = @import("local_stage_runner.zig");
const local_stage_runtime = @import("local_stage_runtime.zig");
const transport = @import("../transport/root.zig");
const runtime_contract = @import("runtime_contract_pkg");
const models = @import("models_pkg");
const scheduler_contracts = @import("../scheduler/contracts.zig");

const Allocator = std.mem.Allocator;
const max_inline_stage_count: usize = 8;

pub const StageSpec = local_stage_contract.StageSpec;
pub const StageInputSpec = local_stage_runtime.StageInputSpec;
pub const StageRuntimeCapability = local_stage_runtime.StageRuntimeCapability;
pub const BoundaryRuntime = local_stage_contract.BoundaryRuntime;
pub const HostBackendKind = @import("host_capability.zig").HostBackendKind;

pub const StageDecodeRequest = struct {
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    layer_start: usize,
    layer_end: usize,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    use_preloaded_input: bool,
};

pub const StagePrefillRequest = struct {
    slot_index: usize,
    tokens: []const u32,
    sequence_start: usize,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
    compute_logits: bool,
    logits_out_opt: ?[]f32,
    source_embeddings_out: ?[]f32 = null,
};

pub const StageVTable = struct {
    deinit: *const fn (*anyopaque, Allocator) void,
    max_batch_size: *const fn (*const anyopaque) usize,
    prefill_chunk_rows_cap: *const fn (*const anyopaque) usize,
    state_descriptors: *const fn (*const anyopaque) []const runtime_contract.StateDescriptor,
    alloc_slot: *const fn (*anyopaque) ?usize,
    free_slot: *const fn (*anyopaque, usize) void,
    reset_slot: *const fn (*anyopaque, usize) void,
    bind_slot_state_blocks: *const fn (*anyopaque, usize, []const runtime_contract.StateBlockHandle) anyerror!void,
    unbind_slot_state_blocks: *const fn (*anyopaque, usize) void,
    execute_decode: *const fn (*anyopaque, StageDecodeRequest) anyerror!void,
    execute_prefill: *const fn (*anyopaque, StagePrefillRequest) anyerror!void,
    synchronize: *const fn (*anyopaque, transport.StageExecutionReceipt) anyerror!void,
    download_decode_activation: *const fn (*anyopaque, usize, []u8, usize) anyerror!void,
    download_prefill_activation: *const fn (*anyopaque, []u8, usize) anyerror!void,
    upload_decode_activation: *const fn (*anyopaque, usize, []const u8, usize) anyerror!void,
    upload_prefill_activation: *const fn (*anyopaque, []const u8, usize) anyerror!void,
    upload_activation_segments: ?*const fn (*anyopaque, []const []const u8, usize) anyerror!void = null,
    peer_copy_activation_to: ?*const fn (*anyopaque, *anyopaque, usize) anyerror!void = null,
    peer_copy_handles_stage_sync: ?*const fn (*const anyopaque) bool = null,
    host_decode_activation: *const fn (*anyopaque, usize, usize) anyerror![]const u8,
    host_prefill_activation: *const fn (*anyopaque, usize) anyerror![]const u8,
    device_location_hint: *const fn (*const anyopaque) ?@import("tensor_frame.zig").TensorFramePayloadLocationHint,
};

pub const StageHandle = struct {
    stage_id: usize,
    backend_kind: HostBackendKind,
    layer_start: usize,
    layer_end: usize,
    supported_boundary_dtypes: []const @import("pipeline.zig").BoundaryDType,
    ptr: *anyopaque,
    vtable: *const StageVTable,

    pub fn deinit(self: *@This(), allocator: Allocator) void {
        self.vtable.deinit(self.ptr, allocator);
        self.* = undefined;
    }

    fn maxBatchSize(self: *const @This()) usize {
        return self.vtable.max_batch_size(self.ptr);
    }

    fn prefillChunkRowsCap(self: *const @This()) usize {
        return self.vtable.prefill_chunk_rows_cap(self.ptr);
    }

    fn stateDescriptors(self: *const @This()) []const runtime_contract.StateDescriptor {
        return self.vtable.state_descriptors(self.ptr);
    }

    fn deviceLocationHint(self: *const @This()) ?@import("tensor_frame.zig").TensorFramePayloadLocationHint {
        return self.vtable.device_location_hint(self.ptr);
    }
};

pub const InitRequest = struct {
    allocator: Allocator,
    loaded: *models.LoadedModel,
    d_model: usize,
    vocab_size: usize,
    total_layers: usize,
    stages: []StageHandle,
    stage_inputs: []const StageInputSpec,
    stage_capabilities: []const StageRuntimeCapability,
    boundary_peer_copy_available: []const bool,
    load_semantics: models.stage_plan.LoadSemantics,
};

pub const LocalPipelineRuntime = struct {
    allocator: Allocator,
    loaded: *models.LoadedModel,
    d_model: usize,
    vocab_size: usize,
    max_batch_size: usize,
    prefill_chunk_rows_cap: usize,
    stages: []StageHandle,
    runtime_plan: local_stage_runtime.Plan,
    contracts: local_stage_contract.ContractBundle,
    slot_in_use: []bool,
    slot_positions: []usize,
    slot_request_ids: []?u64,
    next_slot_request_id: u64 = 1,
    slot_logits: []f32,
    state_descriptors_storage: [runtime_contract.max_state_descriptors]runtime_contract.StateDescriptor = undefined,
    state_descriptor_count: u8 = 0,

    pub fn init(request: InitRequest) !LocalPipelineRuntime {
        if (request.stages.len < 2 or request.stages.len != request.stage_inputs.len) {
            return error.InvalidTopologyConfig;
        }
        if (request.stages.len > max_inline_stage_count) return error.InvalidTopologyConfig;
        if (request.stage_capabilities.len != request.stages.len) return error.InvalidTopologyConfig;

        var runtime_plan = try local_stage_runtime.buildPlan(.{
            .allocator = request.allocator,
            .d_model = request.d_model,
            .total_layers = request.total_layers,
            .stages = request.stage_inputs,
            .stage_capabilities = request.stage_capabilities,
            .boundary_peer_copy_available = request.boundary_peer_copy_available,
        });
        errdefer runtime_plan.deinit();

        var contracts = try local_stage_contract.buildContractBundle(.{
            .allocator = request.allocator,
            .loaded = request.loaded,
            .d_model = request.d_model,
            .total_layers = request.total_layers,
            .split_points = runtime_plan.split_points,
            .stage_backend_kinds = runtime_plan.stage_backend_kinds,
            .boundary_configs = runtime_plan.boundary_configs,
            .load_semantics = request.load_semantics,
        });
        errdefer contracts.deinit();

        var max_batch_size = request.stages[0].maxBatchSize();
        var prefill_chunk_rows_cap = request.stages[0].prefillChunkRowsCap();
        for (request.stages[1..]) |*stage| {
            max_batch_size = @min(max_batch_size, stage.maxBatchSize());
            prefill_chunk_rows_cap = @min(prefill_chunk_rows_cap, stage.prefillChunkRowsCap());
        }
        if (max_batch_size == 0 or prefill_chunk_rows_cap == 0) return error.InvalidTopologyConfig;

        var runtime = LocalPipelineRuntime{
            .allocator = request.allocator,
            .loaded = request.loaded,
            .d_model = request.d_model,
            .vocab_size = request.vocab_size,
            .max_batch_size = max_batch_size,
            .prefill_chunk_rows_cap = prefill_chunk_rows_cap,
            .stages = request.stages,
            .runtime_plan = runtime_plan,
            .contracts = contracts,
            .slot_in_use = &.{},
            .slot_positions = &.{},
            .slot_request_ids = &.{},
            .slot_logits = &.{},
        };
        runtime_plan = .{ .allocator = request.allocator };
        contracts = .{};
        errdefer runtime.deinit();

        runtime.slot_in_use = try request.allocator.alloc(bool, max_batch_size);
        @memset(runtime.slot_in_use, false);
        runtime.slot_positions = try request.allocator.alloc(usize, max_batch_size);
        @memset(runtime.slot_positions, 0);
        runtime.slot_request_ids = try request.allocator.alloc(?u64, max_batch_size);
        @memset(runtime.slot_request_ids, null);
        runtime.slot_logits = try request.allocator.alloc(f32, max_batch_size * request.vocab_size);

        for (runtime.stages) |*stage| {
            for (stage.stateDescriptors()) |descriptor| {
                try runtime_contract.appendUniqueStateDescriptor(
                    runtime.state_descriptors_storage[0..],
                    &runtime.state_descriptor_count,
                    descriptor,
                );
            }
        }

        return runtime;
    }

    pub fn deinit(self: *@This()) void {
        if (self.slot_logits.len != 0) self.allocator.free(self.slot_logits);
        if (self.slot_request_ids.len != 0) self.allocator.free(self.slot_request_ids);
        if (self.slot_positions.len != 0) self.allocator.free(self.slot_positions);
        if (self.slot_in_use.len != 0) self.allocator.free(self.slot_in_use);
        self.contracts.deinit();
        self.runtime_plan.deinit();
        for (self.stages) |*stage| stage.deinit(self.allocator);
        if (self.stages.len != 0) self.allocator.free(self.stages);
        self.* = undefined;
    }

    pub fn maxBatchSize(self: *const @This()) usize {
        return self.max_batch_size;
    }

    pub fn allocSlot(self: *@This()) ?usize {
        var slot_index: usize = 0;
        while (slot_index < self.slot_in_use.len) : (slot_index += 1) {
            if (!self.slot_in_use[slot_index]) break;
        } else return null;

        var allocated_until: usize = 0;
        for (self.stages, 0..) |*stage, index| {
            const child_slot = stage.vtable.alloc_slot(stage.ptr) orelse {
                self.rollbackAlloc(slot_index, allocated_until);
                return null;
            };
            if (child_slot != slot_index) {
                stage.vtable.free_slot(stage.ptr, child_slot);
                self.rollbackAlloc(slot_index, allocated_until);
                return null;
            }
            stage.vtable.unbind_slot_state_blocks(stage.ptr, slot_index);
            allocated_until = index + 1;
        }

        self.slot_in_use[slot_index] = true;
        self.slot_positions[slot_index] = 0;
        self.slot_request_ids[slot_index] = self.next_slot_request_id;
        self.next_slot_request_id +%= 1;
        if (self.next_slot_request_id == 0) self.next_slot_request_id = 1;
        return slot_index;
    }

    fn rollbackAlloc(self: *@This(), slot_index: usize, allocated_until: usize) void {
        for (self.stages[0..allocated_until]) |*stage| {
            stage.vtable.free_slot(stage.ptr, slot_index);
            stage.vtable.unbind_slot_state_blocks(stage.ptr, slot_index);
        }
    }

    pub fn freeSlot(self: *@This(), slot_index: usize) void {
        if (slot_index >= self.slot_in_use.len) return;
        for (self.stages) |*stage| {
            stage.vtable.free_slot(stage.ptr, slot_index);
            stage.vtable.unbind_slot_state_blocks(stage.ptr, slot_index);
        }
        self.slot_in_use[slot_index] = false;
        self.slot_positions[slot_index] = 0;
        self.slot_request_ids[slot_index] = null;
    }

    pub fn resetSlot(self: *@This(), slot_index: usize) void {
        if (slot_index >= self.slot_in_use.len) return;
        for (self.stages) |*stage| stage.vtable.reset_slot(stage.ptr, slot_index);
        self.slot_positions[slot_index] = 0;
    }

    pub fn getPosition(self: *const @This(), slot_index: usize) usize {
        if (slot_index >= self.slot_positions.len) return 0;
        return self.slot_positions[slot_index];
    }

    pub fn stateDescriptors(self: *const @This()) []const runtime_contract.StateDescriptor {
        return self.state_descriptors_storage[0..self.state_descriptor_count];
    }

    pub fn synchronize(self: *@This()) !void {
        for (self.stages) |*stage| {
            try stage.vtable.synchronize(stage.ptr, transport.StageExecutionReceipt.completed(stage.stage_id));
        }
    }

    pub fn bindSlotStateBlocks(
        self: *@This(),
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        if (slot_index >= self.slot_in_use.len) return error.InvalidArgument;
        try runtime_contract.validateStateBlocksForDescriptors(self.stateDescriptors(), state_blocks);

        var bound_until: usize = 0;
        for (self.stages, 0..) |*stage, index| {
            var filtered: [runtime_contract.max_state_descriptors]runtime_contract.StateBlockHandle = undefined;
            const stage_blocks = try filterStateBlocksForStage(stage.stateDescriptors(), state_blocks, filtered[0..]);
            stage.vtable.bind_slot_state_blocks(stage.ptr, slot_index, stage_blocks) catch |err| {
                for (self.stages[0..bound_until]) |*rollback| {
                    rollback.vtable.unbind_slot_state_blocks(rollback.ptr, slot_index);
                }
                return err;
            };
            bound_until = index + 1;
        }
    }

    pub fn unbindSlotStateBlocks(self: *@This(), slot_index: usize) void {
        if (slot_index >= self.slot_in_use.len) return;
        for (self.stages) |*stage| stage.vtable.unbind_slot_state_blocks(stage.ptr, slot_index);
    }

    pub fn prefill(self: *@This(), tokens: []const u32, logits_out: []f32) !void {
        try self.prefillSlot(0, tokens, logits_out);
    }

    pub fn decode(self: *@This(), token: u32, position: usize, logits_out: []f32) !void {
        try self.decodeSlotAtPosition(0, token, position, logits_out);
    }

    pub fn prefillSlot(self: *@This(), slot_index: usize, tokens: []const u32, logits_out: []f32) !void {
        if (tokens.len == 0 or logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        var sequence_start: usize = 0;
        while (sequence_start < tokens.len) {
            const rows = @min(tokens.len - sequence_start, self.prefill_chunk_rows_cap);
            const chunk = tokens[sequence_start..][0..rows];
            const is_final_chunk = sequence_start + rows == tokens.len;
            try self.executePrefillChunk(slot_index, chunk, sequence_start, is_final_chunk, if (is_final_chunk) logits_out else null);
            sequence_start += rows;
        }
        self.slot_positions[slot_index] = tokens.len;
    }

    pub fn prefillBatch(self: *@This(), requests: []const scheduler_contracts.PrefillBatchRequest) !void {
        for (requests) |request| {
            try self.prefillSlot(request.slot_index, request.prompt_tokens, request.logits_out);
        }
    }

    pub fn prefillSlotWithVision(
        self: *@This(),
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const anyopaque,
        logits_out: []f32,
    ) !void {
        if (vision_input != null) return error.UnsupportedContentType;
        try self.prefillSlot(slot_index, tokens, logits_out);
    }

    pub fn decodeBatch(
        self: *@This(),
        requests: []const scheduler_contracts.DecodeRequest,
        results: []scheduler_contracts.DecodeResult,
    ) !void {
        if (results.len < requests.len) return error.InvalidArgument;
        for (requests, 0..) |request, index| {
            const logits = self.slotLogits(request.slot_index);
            try self.decodeSlotAtPosition(request.slot_index, request.token, self.slot_positions[request.slot_index], logits);
            results[index] = .{ .slot_index = request.slot_index, .logits = logits };
        }
    }

    pub fn decodeTopKCandidates(
        self: *@This(),
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        const logits = self.slotLogits(slot_index);
        try self.decodeSlotAtPosition(slot_index, token, self.slot_positions[slot_index], logits);
        return local_stage_runtime.extractTopKFromHostLogitsRow(logits, top_k, candidate_logits_out, candidate_ids_out);
    }

    pub fn decodeBatchTopKCandidates(
        self: *@This(),
        requests: []const scheduler_contracts.DecodeRequest,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
        candidate_counts_out: []usize,
    ) !void {
        if (candidate_counts_out.len < requests.len) return error.InvalidArgument;
        for (requests, 0..) |request, index| {
            const row_start = std.math.mul(usize, index, top_k) catch return error.InvalidArgument;
            const row_end = std.math.add(usize, row_start, top_k) catch return error.InvalidArgument;
            if (row_end > candidate_logits_out.len or row_end > candidate_ids_out.len) return error.InvalidArgument;
            candidate_counts_out[index] = try self.decodeTopKCandidates(
                request.slot_index,
                request.token,
                top_k,
                candidate_logits_out[row_start..row_end],
                candidate_ids_out[row_start..row_end],
            );
        }
    }

    fn decodeSlotAtPosition(
        self: *@This(),
        slot_index: usize,
        token: u32,
        position: usize,
        logits_out: []f32,
    ) !void {
        if (!self.slotIndexSupported(slot_index) or logits_out.len != self.vocab_size) return error.InvalidArgument;
        const slots = [_]usize{slot_index};
        const positions = [_]usize{position};
        var endpoints: [8]PipelineEndpoint = undefined;
        var stage_refs: [8]local_stage_runner.LocalStageChainStage = undefined;
        var transport_refs: [8]transport.LocalStageTransportEndpoint = undefined;
        const registry = try self.buildDecodeRegistry(
            endpoints[0..],
            stage_refs[0..],
            transport_refs[0..],
            .{
                .token = token,
                .position = position,
                .slot_index = slot_index,
                .logits_out_opt = logits_out,
                .layer_start = 0,
                .layer_end = 0,
                .compute_logits = true,
                .download_logits = true,
                .ensure_kv_capacity = true,
                .use_preloaded_input = false,
            },
        );
        var payloads: [8]local_pipeline.LocalDecodeBoundaryPayloadSpec = undefined;
        const boundary_payloads = try self.decodeBoundaryPayloads(&slots, &positions, payloads[0..]);
        try local_pipeline.executeLocalDecodePipelineStepWithEndpointRegistry(self.context(), registry, .{
            .tensor_frame_plan_ref = &self.contracts.tensor_frame_plan_ref.?,
            .hidden_size = self.d_model,
            .slot_request_ids = self.slot_request_ids,
            .slot_indices = &slots,
            .positions = &positions,
            .boundary_payloads = boundary_payloads,
        }, false);
        self.slot_positions[slot_index] = position + 1;
    }

    fn executePrefillChunk(
        self: *@This(),
        slot_index: usize,
        tokens: []const u32,
        sequence_start: usize,
        is_final_chunk: bool,
        logits_out_opt: ?[]f32,
    ) !void {
        var endpoints: [8]PipelineEndpoint = undefined;
        var stage_refs: [8]local_stage_runner.LocalStageChainStage = undefined;
        var transport_refs: [8]transport.LocalStageTransportEndpoint = undefined;
        const registry = try self.buildPrefillRegistry(
            endpoints[0..],
            stage_refs[0..],
            transport_refs[0..],
            .{
                .slot_index = slot_index,
                .tokens = tokens,
                .sequence_start = sequence_start,
                .layer_start = 0,
                .layer_end = 0,
                .use_preloaded_input = false,
                .compute_logits = is_final_chunk,
                .logits_out_opt = logits_out_opt,
            },
        );
        var payloads: [8]local_pipeline.LocalPrefillBoundaryPayloadSpec = undefined;
        const boundary_payloads = try self.prefillBoundaryPayloads(slot_index, sequence_start, tokens.len, payloads[0..]);
        try local_pipeline.executeLocalPrefillPipelineStepWithEndpointRegistry(self.context(), registry, .{
            .tensor_frame_plan_ref = &self.contracts.tensor_frame_plan_ref.?,
            .hidden_size = self.d_model,
            .slot_request_ids = self.slot_request_ids,
            .boundary_payloads = boundary_payloads,
        }, false);
    }

    fn context(self: *@This()) local_pipeline.LocalPipelineContext {
        return .{
            .allocator = self.allocator,
            .plan_ref = &self.contracts.local_stage_runner_plan_ref.?,
            .placement_plan = &self.contracts.placement_plan.?,
            .state_ownership_plan = if (self.contracts.state_ownership_plan) |*plan| plan else null,
        };
    }

    fn slotIndexSupported(self: *const @This(), slot_index: usize) bool {
        return slot_index < self.max_batch_size;
    }

    fn slotLogits(self: *@This(), slot_index: usize) []f32 {
        const start = slot_index * self.vocab_size;
        return self.slot_logits[start..][0..self.vocab_size];
    }

    fn boundaryFrame(self: *@This(), boundary_index: usize) !local_pipeline.LocalPipelineBoundaryFrameSpec {
        if (boundary_index >= self.runtime_plan.boundary_runtimes.len) return error.InvalidTopologyConfig;
        const boundary = self.runtime_plan.boundary_runtimes[boundary_index];
        if (boundary.boundary_index != boundary_index) return error.InvalidTopologyConfig;
        return .{
            .boundary_index = boundary.boundary_index,
            .dtype = boundary.dtype,
            .layout = boundary.layout,
            .staging = boundary.staging,
            .local_device_peer_copy_available = boundary.local_device_peer_copy_available,
        };
    }

    fn rowByteCount(self: *const @This(), boundary_index: usize) !usize {
        if (boundary_index >= self.runtime_plan.boundary_runtimes.len) return error.InvalidTopologyConfig;
        const dtype_value = self.runtime_plan.boundary_runtimes[boundary_index].dtype;
        const element_bytes: usize = switch (dtype_value) {
            .bf16, .f16 => @sizeOf(u16),
            .f32 => @sizeOf(f32),
        };
        return std.math.mul(usize, self.d_model, element_bytes) catch return error.InvalidArgument;
    }

    fn decodeBoundaryPayloads(
        self: *@This(),
        slot_indices: []const usize,
        positions: []const usize,
        out: []local_pipeline.LocalDecodeBoundaryPayloadSpec,
    ) ![]local_pipeline.LocalDecodeBoundaryPayloadSpec {
        _ = positions;
        const boundary_count = self.stages.len - 1;
        if (out.len < boundary_count) return error.InvalidArgument;
        for (out[0..boundary_count], 0..) |*payload, boundary_index| {
            const source_stage = &self.stages[boundary_index];
            const row_bytes = try self.rowByteCount(boundary_index);
            const transfer_bytes = std.math.mul(usize, slot_indices.len, row_bytes) catch return error.InvalidArgument;
            payload.* = .{
                .frame = try self.boundaryFrame(boundary_index),
                .activation_byte_count = transfer_bytes,
                .location_hint = sourceLocationHint(source_stage),
                .image = if (source_stage.backend_kind == .cpu)
                    .{ .host_bytes = try source_stage.vtable.host_decode_activation(source_stage.ptr, slot_indices[0], row_bytes) }
                else
                    .device,
            };
        }
        return out[0..boundary_count];
    }

    fn prefillBoundaryPayloads(
        self: *@This(),
        slot_index: usize,
        sequence_start: usize,
        token_count: usize,
        out: []local_pipeline.LocalPrefillBoundaryPayloadSpec,
    ) ![]local_pipeline.LocalPrefillBoundaryPayloadSpec {
        const boundary_count = self.stages.len - 1;
        if (out.len < boundary_count) return error.InvalidArgument;
        for (out[0..boundary_count], 0..) |*payload, boundary_index| {
            const source_stage = &self.stages[boundary_index];
            const row_bytes = try self.rowByteCount(boundary_index);
            const transfer_bytes = std.math.mul(usize, token_count, row_bytes) catch return error.InvalidArgument;
            payload.* = .{
                .frame = try self.boundaryFrame(boundary_index),
                .slot_index = slot_index,
                .sequence_start = sequence_start,
                .token_count = token_count,
                .activation_byte_count = transfer_bytes,
                .location_hint = sourceLocationHint(source_stage),
                .image = if (source_stage.backend_kind == .cpu)
                    .{ .host_bytes = try source_stage.vtable.host_prefill_activation(source_stage.ptr, transfer_bytes) }
                else
                    .device,
            };
        }
        return out[0..boundary_count];
    }

    fn sourceLocationHint(stage: *const StageHandle) ?@import("tensor_frame.zig").TensorFramePayloadLocationHint {
        return switch (stage.backend_kind) {
            .cpu => .{ .cpu = {} },
            .cuda => stage.deviceLocationHint(),
            .metal, .mock, .@"opaque" => null,
        };
    }

    fn buildDecodeRegistry(
        self: *@This(),
        endpoint_storage: []PipelineEndpoint,
        stage_storage: []local_stage_runner.LocalStageChainStage,
        transport_storage: []transport.LocalStageTransportEndpoint,
        request: StageDecodeRequest,
    ) !local_stage_runner.LocalStageEndpointRegistry {
        if (endpoint_storage.len < self.stages.len or stage_storage.len < self.stages.len or transport_storage.len < self.stages.len) {
            return error.InvalidArgument;
        }
        for (self.stages, 0..) |*stage, index| {
            endpoint_storage[index] = PipelineEndpoint{
                .runtime = self,
                .stage_index = index,
                .mode = .{ .decode = request },
            };
            stage_storage[index] = endpoint_storage[index].stageEndpoint(stage.stage_id);
            transport_storage[index] = endpoint_storage[index].transportEndpoint(stage.stage_id, .decode);
        }
        return .{
            .endpoints = stage_storage[0..self.stages.len],
            .transport_endpoints = transport_storage[0..self.stages.len],
        };
    }

    fn buildPrefillRegistry(
        self: *@This(),
        endpoint_storage: []PipelineEndpoint,
        stage_storage: []local_stage_runner.LocalStageChainStage,
        transport_storage: []transport.LocalStageTransportEndpoint,
        request: StagePrefillRequest,
    ) !local_stage_runner.LocalStageEndpointRegistry {
        if (endpoint_storage.len < self.stages.len or stage_storage.len < self.stages.len or transport_storage.len < self.stages.len) {
            return error.InvalidArgument;
        }
        for (self.stages, 0..) |*stage, index| {
            endpoint_storage[index] = PipelineEndpoint{
                .runtime = self,
                .stage_index = index,
                .mode = .{ .prefill = request },
            };
            stage_storage[index] = endpoint_storage[index].stageEndpoint(stage.stage_id);
            transport_storage[index] = endpoint_storage[index].transportEndpoint(stage.stage_id, .prefill);
        }
        return .{
            .endpoints = stage_storage[0..self.stages.len],
            .transport_endpoints = transport_storage[0..self.stages.len],
        };
    }
};

const PipelineMode = union(enum) {
    decode: StageDecodeRequest,
    prefill: StagePrefillRequest,
};

const TransportScope = enum {
    decode,
    prefill,
};

const PipelineEndpoint = struct {
    runtime: *LocalPipelineRuntime,
    stage_index: usize,
    mode: PipelineMode,

    fn stage(self: *@This()) *StageHandle {
        return &self.runtime.stages[self.stage_index];
    }

    fn stageEndpoint(self: *@This(), stage_id: usize) local_stage_runner.LocalStageChainStage {
        return .{
            .stage_id = stage_id,
            .ptr = self,
            .vtable = &stage_endpoint_vtable,
        };
    }

    fn transportEndpoint(self: *@This(), stage_id: usize, scope: TransportScope) transport.LocalStageTransportEndpoint {
        _ = scope;
        return .{
            .stage_id = stage_id,
            .ptr = self,
            .vtable = &transport_endpoint_vtable,
        };
    }

    fn executeDecode(ptr: *anyopaque, input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
        if (input.len != 0) return error.InvalidArgument;
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        var request = switch (self.mode) {
            .decode => |value| value,
            .prefill => return error.InvalidStepRequest,
        };
        const final_index = self.runtime.stages.len - 1;
        request.layer_start = layer_start;
        request.layer_end = layer_end;
        request.use_preloaded_input = self.stage_index != 0;
        if (self.stage_index != final_index) {
            request.logits_out_opt = null;
            request.compute_logits = false;
            request.download_logits = false;
        }
        try self.stage().vtable.execute_decode(self.stage().ptr, request);
    }

    fn executePrefill(ptr: *anyopaque, input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
        if (input.len != 0) return error.InvalidArgument;
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        var request = switch (self.mode) {
            .decode => return error.InvalidStepRequest,
            .prefill => |value| value,
        };
        const final_index = self.runtime.stages.len - 1;
        request.layer_start = layer_start;
        request.layer_end = layer_end;
        request.use_preloaded_input = self.stage_index != 0;
        if (self.stage_index != final_index) {
            request.logits_out_opt = null;
            request.compute_logits = false;
        }
        try self.stage().vtable.execute_prefill(self.stage().ptr, request);
    }

    fn synchronize(ptr: *anyopaque, receipt: transport.StageExecutionReceipt) anyerror!void {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        try self.stage().vtable.synchronize(self.stage().ptr, receipt);
    }

    fn downloadActivation(ptr: *anyopaque, host_buf: []u8, byte_count: usize) anyerror!void {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        switch (self.mode) {
            .decode => |request| try self.stage().vtable.download_decode_activation(
                self.stage().ptr,
                request.slot_index,
                host_buf,
                byte_count,
            ),
            .prefill => try self.stage().vtable.download_prefill_activation(self.stage().ptr, host_buf, byte_count),
        }
    }

    fn uploadActivation(ptr: *anyopaque, host_buf: []const u8, byte_count: usize) anyerror!void {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        switch (self.mode) {
            .decode => |request| try self.stage().vtable.upload_decode_activation(
                self.stage().ptr,
                request.slot_index,
                host_buf,
                byte_count,
            ),
            .prefill => try self.stage().vtable.upload_prefill_activation(self.stage().ptr, host_buf, byte_count),
        }
    }

    fn uploadActivationSegments(ptr: *anyopaque, host_segments: []const []const u8, byte_count: usize) anyerror!void {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        const upload = self.stage().vtable.upload_activation_segments orelse return error.LocalStageTransportSegmentedUploadUnsupported;
        try upload(self.stage().ptr, host_segments, byte_count);
    }

    fn peerCopyActivationTo(ptr: *anyopaque, target_ptr: *anyopaque, byte_count: usize) anyerror!void {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        const target: *PipelineEndpoint = @ptrCast(@alignCast(target_ptr));
        const copy = self.stage().vtable.peer_copy_activation_to orelse return error.LocalStageTransportPeerCopyUnsupported;
        try copy(self.stage().ptr, target.stage().ptr, byte_count);
    }

    fn peerCopyHandlesStageSync(ptr: *anyopaque) bool {
        const self: *PipelineEndpoint = @ptrCast(@alignCast(ptr));
        const handles = self.stage().vtable.peer_copy_handles_stage_sync orelse return false;
        return handles(self.stage().ptr);
    }

    const stage_endpoint_vtable = local_stage_runner.LocalStageEndpointVTable{
        .execute_decode_layer_range = executeDecode,
        .execute_prefill_layer_range = executePrefill,
    };

    const transport_endpoint_vtable = transport.LocalStageTransportEndpointVTable{
        .synchronize = synchronize,
        .download_activation = downloadActivation,
        .upload_activation = uploadActivation,
        .upload_activation_segments = uploadActivationSegments,
        .peer_copy_activation_to = peerCopyActivationTo,
        .peer_copy_handles_stage_sync = peerCopyHandlesStageSync,
    };
};

fn filterStateBlocksForStage(
    descriptors: []const runtime_contract.StateDescriptor,
    state_blocks: []const runtime_contract.StateBlockHandle,
    out: []runtime_contract.StateBlockHandle,
) ![]const runtime_contract.StateBlockHandle {
    if (out.len < descriptors.len) return error.InvalidStateDescriptorBinding;
    for (descriptors, 0..) |descriptor, index| {
        const incoming = runtime_contract.findStateBlock(state_blocks, descriptor.id) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        out[index] = incoming.*;
    }
    return out[0..descriptors.len];
}

test "filterStateBlocksForStage forwards only stage descriptors" {
    const descriptors = [_]runtime_contract.StateDescriptor{
        .{ .id = 2, .size_bytes = 8, .align_bytes = 8, .zero_init = false, .lifecycle = .slot_persistent, .runtime_kind = 0 },
    };
    var storage: [8]u8 align(8) = undefined;
    const blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 1, .ptr = storage[0..].ptr, .size = storage.len, .align_bytes = 8 },
        .{ .id = 2, .ptr = storage[0..].ptr, .size = storage.len, .align_bytes = 8 },
    };
    var out: [2]runtime_contract.StateBlockHandle = undefined;

    const filtered = try filterStateBlocksForStage(&descriptors, &blocks, out[0..]);

    try std.testing.expectEqual(@as(usize, 1), filtered.len);
    try std.testing.expectEqual(@as(u8, 2), filtered[0].id);
}
