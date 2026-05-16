//! Local decode stage-chain assembly for process-local pipelines.

const std = @import("std");
const bridge = @import("../bridge/root.zig");
const transport = @import("../transport/root.zig");

const stage_adapters = @import("local_stage_adapters.zig");

pub const DecodeBoundaryStageSide = enum {
    source,
    target,
};

pub const BatchedDecodeExecutionPlan = struct {
    allow_staged_internal_execution: bool = false,
    use_preloaded_input: bool = false,
    compute_logits: bool = true,
    emit_decode_summary: bool = true,
    summary_label_override: ?[]const u8 = null,
};

pub const SingleTokenDecodeRequest = struct {
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
};

pub fn optionalPointerPayload(comptime MaybePointer: type) type {
    return switch (@typeInfo(MaybePointer)) {
        .optional => |optional| optional.child,
        else => MaybePointer,
    };
}

pub fn localCpuStagePointerType(comptime Backend: type) type {
    const return_type = @typeInfo(@TypeOf(Backend.localCpuStage)).@"fn".return_type.?;
    return optionalPointerPayload(return_type);
}

pub fn localCudaStagePointerType(comptime Backend: type) type {
    const return_type = @typeInfo(@TypeOf(Backend.localCudaStage)).@"fn".return_type.?;
    return optionalPointerPayload(return_type);
}

pub fn rootCudaStageId(self: anytype) !usize {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "local_stage_specs")) return error.InvalidTopologyConfig;
    if (comptime @hasField(SelfType, "local_root_stage_id")) return self.local_root_stage_id;
    const specs = self.local_stage_specs;
    if (specs.len < 2) return error.InvalidTopologyConfig;
    if (specs[0].backend_kind == .cuda) return specs[0].stage_id;
    for (specs) |spec| {
        if (spec.backend_kind == .cuda and spec.owns_projection) return spec.stage_id;
    }
    return error.InvalidTopologyConfig;
}

fn deepstackFeaturesForStage(
    deepstack_layer_features_opt: ?[]const []const f32,
    layer_start: usize,
) ?[]const []const f32 {
    const deepstack_layer_features = deepstack_layer_features_opt orelse return null;
    if (layer_start >= deepstack_layer_features.len) return null;
    return deepstack_layer_features[layer_start..];
}

pub fn cudaLocationHintForStageId(
    self: anytype,
    root_stage_id: usize,
    stage_id: usize,
) !bridge.TensorFramePayloadLocationHint {
    const SelfType = @TypeOf(self.*);
    if (stage_id == root_stage_id) return try stage_adapters.cudaPayloadLocationHint(self);
    if (comptime !@hasDecl(SelfType, "localCudaStage")) return error.InvalidTopologyConfig;
    return try stage_adapters.cudaPayloadLocationHint(self.localCudaStage(stage_id) orelse return error.InvalidTopologyConfig);
}

fn mirrorDecodeStageDescriptorsFromRoot(stage_backend: anytype, root_backend: anytype, slot_indices: []const usize) !void {
    const StageType = @TypeOf(stage_backend.*);
    if (comptime @hasField(StageType, "state_descriptor_count") and @hasDecl(StageType, "mirrorSlotStateBlocksFrom")) {
        if (stage_backend.state_descriptor_count > 0) {
            for (slot_indices) |slot_idx| {
                try stage_backend.mirrorSlotStateBlocksFrom(root_backend, slot_idx);
            }
        }
    }
}

const LocalDecodeEndpointKind = enum {
    cpu,
    cuda,
};

const MissingLocalCpuStage = opaque {};

fn SingleDecodeStageEndpoint(
    comptime CudaBackendPtr: type,
    comptime CpuBackendPtr: type,
    comptime has_cpu_stage: bool,
    comptime execute_decode_with_layer_limit: anytype,
) type {
    return struct {
        kind: LocalDecodeEndpointKind,
        cuda_backend: ?CudaBackendPtr = null,
        cpu_backend: ?CpuBackendPtr = null,
        ctx: *const stage_adapters.DecodeContext,
        activation_slot_index: usize,
        is_final: bool = false,
        logits_out_opt: ?[]f32 = null,
        compute_logits: bool = false,
        download_logits: bool = false,
        hidden_override: ?[]const f32 = null,
        deepstack_layer_features_opt: ?[]const []const f32 = null,
        deepstack_feature_index_opt: ?usize = null,
        use_preloaded_input: bool = false,
        peer_copy_synchronization: transport.CudaPeerCopySynchronization = .source_stream,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            switch (stage.kind) {
                .cpu => {
                    if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
                    try stage_adapters.executeCpuDecodeLayerRange(
                        stage.cpu_backend orelse return error.InvalidTopologyConfig,
                        stage.ctx,
                        layer_start,
                        layer_end,
                        if (stage.is_final) stage.logits_out_opt else null,
                        stage.is_final and stage.compute_logits,
                        stage.is_final and stage.download_logits,
                        stage.use_preloaded_input,
                    );
                },
                .cuda => {
                    try stage_adapters.executeCudaDecodeLayerRange(
                        execute_decode_with_layer_limit,
                        stage.cuda_backend orelse return error.InvalidTopologyConfig,
                        stage.ctx,
                        layer_start,
                        layer_end,
                        if (stage.is_final) stage.logits_out_opt else null,
                        stage.is_final and stage.compute_logits,
                        stage.is_final and stage.download_logits,
                        stage.hidden_override,
                        stage.deepstack_layer_features_opt,
                        stage.deepstack_feature_index_opt,
                        stage.use_preloaded_input,
                    );
                },
            }
        }
    };
}

pub fn executeSingleTokenDecodePipeline(
    comptime execute_decode_with_layer_limit: anytype,
    self: anytype,
    request: SingleTokenDecodeRequest,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasDecl(SelfType, "localActivationByteCount")) return error.InvalidTopologyConfig;
    if (comptime !@hasField(SelfType, "local_stage_specs")) return error.InvalidTopologyConfig;

    var ctx = stage_adapters.DecodeContext{
        .token = request.token,
        .position = request.position,
        .slot_index = request.slot_index,
        .logits_out_opt = request.logits_out_opt,
        .compute_logits = request.compute_logits,
        .download_logits = request.download_logits,
        .ensure_kv_capacity = request.ensure_kv_capacity,
        .trace_seq_len_u32 = request.trace_seq_len_u32,
        .trace_pos_offset = request.trace_pos_offset,
    };
    const slot_indices = [_]usize{request.slot_index};
    const positions = [_]usize{request.position};
    const specs = self.local_stage_specs;
    if (specs.len < 2) return error.InvalidTopologyConfig;
    if (specs[0].backend_kind != .cuda) {
        if (request.hidden_override != null or request.deepstack_layer_features_opt != null or request.deepstack_feature_index_opt != null) {
            return error.InvalidTopologyConfig;
        }
    }

    const root_stage_id = try rootCudaStageId(self);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage");
    const has_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage");
    const CpuPtr = if (has_cpu_stage) localCpuStagePointerType(SelfType) else *MissingLocalCpuStage;
    const CudaPtr = if (has_cuda_stage) localCudaStagePointerType(SelfType) else @TypeOf(self);

    for (specs) |spec| {
        if (spec.backend_kind != .cuda) continue;
        if (spec.stage_id == root_stage_id) {
            if (!spec.owns_embedding) self.activateKvSlot(request.slot_index);
            continue;
        }
        if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
        const cuda_backend = self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
        try mirrorDecodeStageDescriptorsFromRoot(cuda_backend, self, &slot_indices);
        if (!spec.owns_embedding) cuda_backend.activateKvSlot(request.slot_index);
    }

    const Endpoint = SingleDecodeStageEndpoint(CudaPtr, CpuPtr, has_cpu_stage, execute_decode_with_layer_limit);
    const allocator = stage_adapters.backendAllocator(self);
    var inline_endpoints: [8]Endpoint = undefined;
    var inline_stages: [8]bridge.LocalStageChainStage = undefined;
    var inline_transport_stages: [8]transport.LocalStageTransportEndpoint = undefined;
    var heap_endpoints: []Endpoint = &.{};
    var heap_stages: []bridge.LocalStageChainStage = &.{};
    var heap_transport_stages: []transport.LocalStageTransportEndpoint = &.{};
    defer {
        if (heap_transport_stages.len != 0) allocator.?.free(heap_transport_stages);
        if (heap_stages.len != 0) allocator.?.free(heap_stages);
        if (heap_endpoints.len != 0) allocator.?.free(heap_endpoints);
    }
    const endpoints = if (specs.len <= inline_endpoints.len)
        inline_endpoints[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_endpoints = try alloc.alloc(Endpoint, specs.len);
        break :blk heap_endpoints;
    };
    const stages = if (specs.len <= inline_stages.len)
        inline_stages[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_stages = try alloc.alloc(bridge.LocalStageChainStage, specs.len);
        break :blk heap_stages;
    };
    const transport_stages = if (specs.len <= inline_transport_stages.len)
        inline_transport_stages[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_transport_stages = try alloc.alloc(transport.LocalStageTransportEndpoint, specs.len);
        break :blk heap_transport_stages;
    };

    for (specs, 0..) |spec, index| {
        const is_final = index + 1 == specs.len;
        const endpoint_kind: LocalDecodeEndpointKind = switch (spec.backend_kind) {
            .cpu => .cpu,
            .cuda => .cuda,
            else => return error.InvalidTopologyConfig,
        };
        const cpu_backend: ?CpuPtr = if (spec.backend_kind == .cpu) blk: {
            if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
            break :blk self.localCpuStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
        } else null;
        const cuda_backend: ?CudaPtr = if (spec.backend_kind == .cuda) blk: {
            if (spec.stage_id == root_stage_id) break :blk self;
            if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
            break :blk self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
        } else null;
        const boundary = if (!is_final) try stage_adapters.localBoundaryRuntime(self, index) else null;
        endpoints[index] = .{
            .kind = endpoint_kind,
            .cuda_backend = cuda_backend,
            .cpu_backend = cpu_backend,
            .ctx = &ctx,
            .activation_slot_index = request.slot_index,
            .is_final = is_final,
            .logits_out_opt = request.logits_out_opt,
            .compute_logits = request.compute_logits,
            .download_logits = request.download_logits,
            .hidden_override = if (spec.owns_embedding and !is_final) request.hidden_override else null,
            .deepstack_layer_features_opt = if (spec.backend_kind == .cuda)
                deepstackFeaturesForStage(request.deepstack_layer_features_opt, spec.layer_start)
            else
                null,
            .deepstack_feature_index_opt = if (spec.backend_kind == .cuda) request.deepstack_feature_index_opt else null,
            .use_preloaded_input = !spec.owns_embedding,
            .peer_copy_synchronization = if (boundary) |value| value.peer_copy_synchronization else .source_stream,
        };
        stages[index] = bridge.localStageAdapter(Endpoint, spec.stage_id, &endpoints[index]);
        transport_stages[index] = transport.localEndpointTransportAdapter(Endpoint, spec.stage_id, &endpoints[index], .{
            .has_cpu_stage = has_cpu_stage,
            .allow_cpu_decode_download = true,
            .prepare_cpu_boundary = true,
        });
    }

    const boundary_count = specs.len - 1;
    var inline_payload_specs: [8]bridge.LocalDecodeBoundaryPayloadSpec = undefined;
    var heap_payload_specs: []bridge.LocalDecodeBoundaryPayloadSpec = &.{};
    defer if (heap_payload_specs.len != 0) allocator.?.free(heap_payload_specs);
    const payload_specs = if (boundary_count <= inline_payload_specs.len)
        inline_payload_specs[0..boundary_count]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_payload_specs = try alloc.alloc(bridge.LocalDecodeBoundaryPayloadSpec, boundary_count);
        break :blk heap_payload_specs;
    };

    for (payload_specs, 0..) |*payload, boundary_index| {
        const source_spec = specs[boundary_index];
        const activation_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
        payload.* = switch (source_spec.backend_kind) {
            .cpu => blk: {
                if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
                const cpu = self.localCpuStage(source_spec.stage_id) orelse return error.InvalidTopologyConfig;
                break :blk .{
                    .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                    .activation_byte_count = activation_bytes,
                    .location_hint = .{ .cpu = {} },
                    .image = .{ .host_bytes = cpu.slotActivationBytes(request.slot_index) },
                    .local_device_peer_copy_available = false,
                };
            },
            .cuda => .{
                .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                .activation_byte_count = activation_bytes,
                .location_hint = try cudaLocationHintForStageId(self, root_stage_id, source_spec.stage_id),
                .image = .device,
            },
            else => return error.InvalidTopologyConfig,
        };
    }

    try bridge.executeLocalDecodePipelineStepWithEndpointRegistry(try stage_adapters.localPipelineContext(self), .{ .endpoints = stages, .transport_endpoints = transport_stages }, .{
        .tensor_frame_plan_ref = try stage_adapters.localStageTensorFramePlanRef(self),
        .hidden_size = self.d_model,
        .slot_request_ids = self.slot_request_ids[0..],
        .slot_indices = &slot_indices,
        .positions = &positions,
        .boundary_payloads = payload_specs,
    }, false);
}

fn BatchedDecodeStageEndpoint(
    comptime CudaBackendPtr: type,
    comptime CpuBackendPtr: type,
    comptime has_cpu_stage: bool,
    comptime OutputMode: type,
    comptime compute_stage: anytype,
) type {
    return struct {
        kind: LocalDecodeEndpointKind,
        root_backend: CudaBackendPtr,
        cuda_backend: ?CudaBackendPtr = null,
        cpu_backend: ?CpuBackendPtr = null,
        boundary: stage_adapters.LocalBoundaryRuntimeView,
        location_hint: ?bridge.TensorFramePayloadLocationHint = null,
        slot_indices: []const usize,
        positions: []const usize,
        tokens: []const u32,
        output_mode: OutputMode,
        plan: BatchedDecodeExecutionPlan,
        active_side: DecodeBoundaryStageSide,
        layer_start: usize = 0,
        layer_end: usize = 0,
        use_preloaded_input: bool = false,
        row_bytes: usize = 0,
        host_segments: [][]const u8 = &.{},
        activation_slot_index: usize = 0,
        peer_copy_synchronization: transport.CudaPeerCopySynchronization = .source_stream,

        pub fn executeLayers(stage: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            switch (stage.kind) {
                .cpu => {
                    if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
                    try prepareCpuBatchedDecodeSegments(
                        stage.root_backend,
                        stage.cpu_backend orelse return error.InvalidTopologyConfig,
                        false,
                        stage.root_backend,
                        stage.boundary,
                        stage.tokens,
                        stage.slot_indices,
                        stage.positions,
                        stage.layer_start,
                        stage.layer_end,
                        stage.use_preloaded_input,
                        stage.plan.compute_logits and stage.output_mode == .host_logits,
                        stage.row_bytes,
                        stage.host_segments,
                    );
                },
                .cuda => try compute_stage(
                    stage.root_backend,
                    stage.cuda_backend orelse return error.InvalidTopologyConfig,
                    stage.boundary,
                    stage.location_hint,
                    stage.slot_indices,
                    stage.positions,
                    stage.active_side,
                    stage.tokens,
                    stage.output_mode,
                    stage.plan,
                ),
            }
        }
    };
}

pub fn executeBatchedDecodePipeline(
    comptime compute_stage: anytype,
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    output_mode: anytype,
    mode_label: []const u8,
) !void {
    const SelfType = @TypeOf(self.*);
    const OutputMode = @TypeOf(output_mode);
    if (comptime !@hasField(SelfType, "local_stage_specs")) return error.InvalidTopologyConfig;
    const specs = self.local_stage_specs;
    if (specs.len < 2) return error.InvalidTopologyConfig;
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const root_stage_id = try rootCudaStageId(self);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage");
    const has_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage");
    const CpuPtr = if (has_cpu_stage) localCpuStagePointerType(SelfType) else *MissingLocalCpuStage;
    const CudaPtr = if (has_cuda_stage) localCudaStagePointerType(SelfType) else @TypeOf(self);

    for (specs) |spec| {
        if (spec.backend_kind != .cuda) continue;
        const cuda_backend = if (spec.stage_id == root_stage_id) self else blk: {
            if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
            const local = self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
            try mirrorDecodeStageDescriptorsFromRoot(local, self, slot_indices);
            break :blk local;
        };
        const StageBackendType = @TypeOf(cuda_backend.*);
        if (comptime @hasField(StageBackendType, "runtime_buffers") and @hasField(StageBackendType, "device")) {
            try cuda_backend.runtime_buffers.ensureRowCapacity(&cuda_backend.device, tokens.len, cuda_backend.fixed_alloc_mode);
        }
    }

    var cpu_stage_count: usize = 0;
    for (specs) |spec| {
        if (spec.backend_kind == .cpu) cpu_stage_count += 1;
    }
    var host_segment_scratch = try stage_adapters.HostSegmentScratch.init(
        stage_adapters.backendAllocator(self),
        if (cpu_stage_count == 0) 0 else tokens.len,
    );
    defer host_segment_scratch.deinit();
    var empty_host_segments: [0][]const u8 = .{};
    const host_segments = if (cpu_stage_count == 0) empty_host_segments[0..] else host_segment_scratch.slice(tokens.len);

    const Endpoint = BatchedDecodeStageEndpoint(CudaPtr, CpuPtr, has_cpu_stage, OutputMode, compute_stage);
    const allocator = stage_adapters.backendAllocator(self);
    var inline_endpoints: [8]Endpoint = undefined;
    var inline_stages: [8]bridge.LocalStageChainStage = undefined;
    var inline_transport_stages: [8]transport.LocalStageTransportEndpoint = undefined;
    var heap_endpoints: []Endpoint = &.{};
    var heap_stages: []bridge.LocalStageChainStage = &.{};
    var heap_transport_stages: []transport.LocalStageTransportEndpoint = &.{};
    defer {
        if (heap_transport_stages.len != 0) allocator.?.free(heap_transport_stages);
        if (heap_stages.len != 0) allocator.?.free(heap_stages);
        if (heap_endpoints.len != 0) allocator.?.free(heap_endpoints);
    }
    const endpoints = if (specs.len <= inline_endpoints.len)
        inline_endpoints[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_endpoints = try alloc.alloc(Endpoint, specs.len);
        break :blk heap_endpoints;
    };
    const stages = if (specs.len <= inline_stages.len)
        inline_stages[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_stages = try alloc.alloc(bridge.LocalStageChainStage, specs.len);
        break :blk heap_stages;
    };
    const transport_stages = if (specs.len <= inline_transport_stages.len)
        inline_transport_stages[0..specs.len]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_transport_stages = try alloc.alloc(transport.LocalStageTransportEndpoint, specs.len);
        break :blk heap_transport_stages;
    };

    for (specs, 0..) |spec, index| {
        const is_final = index + 1 == specs.len;
        const endpoint_kind: LocalDecodeEndpointKind = switch (spec.backend_kind) {
            .cpu => .cpu,
            .cuda => .cuda,
            else => return error.InvalidTopologyConfig,
        };
        const cpu_backend: ?CpuPtr = if (spec.backend_kind == .cpu) blk: {
            if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
            break :blk self.localCpuStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
        } else null;
        const cuda_backend: ?CudaPtr = if (spec.backend_kind == .cuda) blk: {
            if (spec.stage_id == root_stage_id) break :blk self;
            if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
            break :blk self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
        } else null;
        const active_boundary_index = if (is_final) index - 1 else index;
        const boundary = try stage_adapters.localBoundaryRuntime(self, active_boundary_index);
        const active_side: DecodeBoundaryStageSide = if (is_final) .target else .source;
        const source_index = if (is_final) index - 1 else index;
        const source_stage_id = specs[source_index].stage_id;
        const location_hint: ?bridge.TensorFramePayloadLocationHint = if (specs[source_index].backend_kind == .cpu)
            .{ .cpu = {} }
        else
            cudaLocationHintForStageId(self, root_stage_id, source_stage_id) catch null;
        const work_plan = BatchedDecodeExecutionPlan{
            .allow_staged_internal_execution = true,
            .use_preloaded_input = !spec.owns_embedding,
            .compute_logits = is_final,
            .emit_decode_summary = is_final,
            .summary_label_override = if (is_final) mode_label else null,
        };
        endpoints[index] = .{
            .kind = endpoint_kind,
            .root_backend = self,
            .cuda_backend = cuda_backend,
            .cpu_backend = cpu_backend,
            .boundary = boundary,
            .location_hint = location_hint,
            .slot_indices = slot_indices,
            .positions = positions,
            .tokens = tokens,
            .output_mode = if (is_final) output_mode else .device_only,
            .plan = work_plan,
            .active_side = active_side,
            .layer_start = spec.layer_start,
            .layer_end = spec.layer_end,
            .use_preloaded_input = !spec.owns_embedding,
            .row_bytes = row_bytes,
            .host_segments = host_segments,
            .activation_slot_index = if (slot_indices.len > 0) slot_indices[0] else 0,
            .peer_copy_synchronization = boundary.peer_copy_synchronization,
        };
        stages[index] = bridge.localStageAdapter(Endpoint, spec.stage_id, &endpoints[index]);
        transport_stages[index] = transport.localEndpointTransportAdapter(Endpoint, spec.stage_id, &endpoints[index], .{
            .has_cpu_stage = has_cpu_stage,
            .prepare_cpu_boundary = true,
        });
    }

    const boundary_count = specs.len - 1;
    var inline_payload_specs: [8]bridge.LocalDecodeBoundaryPayloadSpec = undefined;
    var heap_payload_specs: []bridge.LocalDecodeBoundaryPayloadSpec = &.{};
    defer if (heap_payload_specs.len != 0) allocator.?.free(heap_payload_specs);
    const payload_specs = if (boundary_count <= inline_payload_specs.len)
        inline_payload_specs[0..boundary_count]
    else blk: {
        const alloc = allocator orelse return error.InvalidTopologyConfig;
        heap_payload_specs = try alloc.alloc(bridge.LocalDecodeBoundaryPayloadSpec, boundary_count);
        break :blk heap_payload_specs;
    };

    for (payload_specs, 0..) |*payload, boundary_index| {
        const source_spec = specs[boundary_index];
        const row_transfer_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
        const transfer_bytes = std.math.mul(usize, tokens.len, row_transfer_bytes) catch return error.InvalidArgument;
        payload.* = switch (source_spec.backend_kind) {
            .cpu => .{ .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index), .activation_byte_count = transfer_bytes, .location_hint = .{ .cpu = {} }, .image = .{ .host_segments = host_segments }, .local_device_peer_copy_available = false },
            .cuda => .{ .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index), .activation_byte_count = transfer_bytes, .location_hint = cudaLocationHintForStageId(self, root_stage_id, source_spec.stage_id) catch null, .image = .device },
            else => return error.InvalidTopologyConfig,
        };
    }

    try bridge.executeLocalDecodePipelineStepWithEndpointRegistry(try stage_adapters.localPipelineContext(self), .{ .endpoints = stages, .transport_endpoints = transport_stages }, .{
        .tensor_frame_plan_ref = try stage_adapters.localStageTensorFramePlanRef(self),
        .hidden_size = self.d_model,
        .slot_request_ids = self.slot_request_ids[0..],
        .slot_indices = slot_indices,
        .positions = positions,
        .boundary_payloads = payload_specs,
    }, false);

    if (output_mode == .host_logits) {
        const final_spec = specs[specs.len - 1];
        switch (final_spec.backend_kind) {
            .cpu => {},
            .cuda => if (final_spec.stage_id != root_stage_id) {
                if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
                try copyBatchedDecodeHostLogitsFromStage(self, self.localCudaStage(final_spec.stage_id) orelse return error.InvalidTopologyConfig, tokens.len);
            },
            else => return error.InvalidTopologyConfig,
        }
    }
}

fn preserveBatchedDecodeStageFailure(
    root_backend: anytype,
    boundary_index: usize,
    boundary_dtype: bridge.BoundaryDType,
    boundary_layout: bridge.BoundaryLayout,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: DecodeBoundaryStageSide,
    source_error: anyerror,
) anyerror {
    const allocator = stage_adapters.backendAllocator(root_backend) orelse return source_error;
    var batch_entry_scratch = stage_adapters.DecodeBatchEntryScratch.init(allocator, slot_indices.len) catch return source_error;
    defer batch_entry_scratch.deinit();
    const metadata = stage_adapters.buildDecodeActivationMetadata(
        root_backend,
        boundary_index,
        boundary_dtype,
        boundary_layout,
        location_hint,
        slot_indices,
        positions,
        batch_entry_scratch.slice(slot_indices.len),
    ) catch return source_error;
    const placement_plan = stage_adapters.localStagePlacementPlan(root_backend) catch return source_error;
    return bridge.preserveLocalStageExecutionError(allocator, .{
        .placement_plan = placement_plan,
        .state_ownership_plan = stage_adapters.localStageStateOwnershipPlan(root_backend),
        .metadata = &metadata,
        .active_stage_id = switch (active_side) {
            .source => metadata.boundary.source_stage_id,
            .target => metadata.boundary.target_stage_id,
        },
        .source_error = source_error,
    });
}

pub fn preserveDecodeBoundaryFailure(
    root_backend: anytype,
    boundary: stage_adapters.LocalBoundaryRuntimeView,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: DecodeBoundaryStageSide,
    source_error: anyerror,
) anyerror {
    return preserveBatchedDecodeStageFailure(
        root_backend,
        boundary.boundary_index,
        boundary.dtype,
        boundary.layout,
        location_hint,
        slot_indices,
        positions,
        active_side,
        source_error,
    );
}

fn copyBatchedDecodeHostLogitsFromStage(root_backend: anytype, source_backend: anytype, rows: usize) !void {
    const src_vocab = source_backend.runtime_buffers.projected_vocab;
    const dst_vocab = root_backend.runtime_buffers.projected_vocab;
    if (src_vocab == 0 or dst_vocab == 0) return;
    for (0..rows) |row| {
        const src_row_start = std.math.mul(usize, row, src_vocab) catch continue;
        const src_row_end = std.math.add(usize, src_row_start, src_vocab) catch continue;
        const dst_row_start = std.math.mul(usize, row, dst_vocab) catch continue;
        const dst_row_end = std.math.add(usize, dst_row_start, dst_vocab) catch continue;
        if (src_row_end > source_backend.runtime_buffers.projected_logits_batch_host.len or
            dst_row_end > root_backend.runtime_buffers.projected_logits_batch_host.len)
        {
            continue;
        }
        const src_row = source_backend.runtime_buffers.projected_logits_batch_host[src_row_start..src_row_end];
        const dst_row = root_backend.runtime_buffers.projected_logits_batch_host[dst_row_start..dst_row_end];
        const copy_len = @min(src_row.len, dst_row.len);
        @memset(dst_row, -1.0e9);
        @memcpy(dst_row[0..copy_len], src_row[0..copy_len]);
    }
}

fn prepareCpuBatchedDecodeSegments(
    root_backend: anytype,
    cpu_stage: anytype,
    activate_intermediate: bool,
    intermediate_backend: anytype,
    boundary0: stage_adapters.LocalBoundaryRuntimeView,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
    compute_logits: bool,
    row_bytes: usize,
    host_segments: [][]const u8,
) !void {
    for (0..tokens.len) |row_i| {
        const token = tokens[row_i];
        const slot_index = slot_indices[row_i];
        const position = positions[row_i];
        if (activate_intermediate) intermediate_backend.activateKvSlot(slot_index);
        root_backend.activateKvSlot(slot_index);
        const logits_out_opt: ?[]f32 = if (compute_logits) blk: {
            const RootType = @TypeOf(root_backend.*);
            if (comptime !@hasField(RootType, "runtime_buffers")) return error.InvalidTopologyConfig;
            const projected_vocab = root_backend.runtime_buffers.projected_vocab;
            const row_start = std.math.mul(usize, row_i, projected_vocab) catch return error.InvalidArgument;
            const row_end = std.math.add(usize, row_start, projected_vocab) catch return error.InvalidArgument;
            if (row_end > root_backend.runtime_buffers.projected_logits_batch_host.len) return error.InvalidArgument;
            break :blk root_backend.runtime_buffers.projected_logits_batch_host[row_start..row_end];
        } else null;
        cpu_stage.executeDecodeLayerRange(
            token,
            position,
            slot_index,
            logits_out_opt,
            layer_start,
            layer_end,
            compute_logits,
            false,
            true,
            use_preloaded_input,
        ) catch |err| {
            const row_slot_indices = [_]usize{slot_index};
            const row_positions = [_]usize{position};
            return preserveDecodeBoundaryFailure(
                root_backend,
                boundary0,
                .{ .cpu = {} },
                &row_slot_indices,
                &row_positions,
                .source,
                err,
            );
        };
        const src_row = cpu_stage.slotActivationBytes(slot_index);
        if (src_row.len < row_bytes) return error.InvalidTopologyConfig;
        host_segments[row_i] = src_row[0..row_bytes];
    }
}
