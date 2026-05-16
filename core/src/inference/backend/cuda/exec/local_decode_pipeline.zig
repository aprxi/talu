//! Local decode stage-chain assembly for CUDA-backed local pipelines.

const std = @import("std");
const bridge = @import("../../../bridge/root.zig");
const transport = @import("../../../transport/root.zig");

const stage_adapters = @import("stage_adapters.zig");

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

fn optionalPointerPayload(comptime MaybePointer: type) type {
    return switch (@typeInfo(MaybePointer)) {
        .optional => |optional| optional.child,
        else => MaybePointer,
    };
}

fn localCpuStagePointerType(comptime Backend: type) type {
    const return_type = @typeInfo(@TypeOf(Backend.localCpuStage0)).@"fn".return_type.?;
    return optionalPointerPayload(return_type);
}

fn localCudaStagePointerType(comptime Backend: type) type {
    const return_type = @typeInfo(@TypeOf(Backend.localCudaStage1)).@"fn".return_type.?;
    return optionalPointerPayload(return_type);
}

fn rootCudaStageId(self: anytype) !usize {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "local_stage_specs")) return error.InvalidTopologyConfig;
    const specs = self.local_stage_specs;
    if (specs.len < 2) return error.InvalidTopologyConfig;
    if (specs[0].backend_kind == .cuda) return specs[0].stage_id;
    for (specs) |spec| {
        if (spec.backend_kind == .cuda and spec.owns_projection) return spec.stage_id;
    }
    return error.InvalidTopologyConfig;
}

fn auxCudaStageId(self: anytype, root_stage_id: usize) !?usize {
    var found: ?usize = null;
    for (self.local_stage_specs) |spec| {
        if (spec.backend_kind != .cuda or spec.stage_id == root_stage_id) continue;
        if (found != null) return error.InvalidTopologyConfig;
        found = spec.stage_id;
    }
    return found;
}

fn deepstackFeaturesForStage(
    deepstack_layer_features_opt: ?[]const []const f32,
    layer_start: usize,
) ?[]const []const f32 {
    const deepstack_layer_features = deepstack_layer_features_opt orelse return null;
    if (layer_start >= deepstack_layer_features.len) return null;
    return deepstack_layer_features[layer_start..];
}

fn cudaLocationHintForStageId(
    self: anytype,
    aux_cuda_backend: anytype,
    root_stage_id: usize,
    stage_id: usize,
) !bridge.TensorFramePayloadLocationHint {
    if (stage_id == root_stage_id) return try stage_adapters.cudaPayloadLocationHint(self);
    return try stage_adapters.cudaPayloadLocationHint(aux_cuda_backend orelse return error.InvalidTopologyConfig);
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

fn cudaDecodeSourceWork(
    comptime Backend: type,
    comptime execute_decode_with_layer_limit: anytype,
    ctx: *const stage_adapters.DecodeContext,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
) stage_adapters.CudaDecodeLayerWork(Backend, execute_decode_with_layer_limit) {
    return .{
        .ctx = ctx,
        .hidden_override = hidden_override,
        .deepstack_layer_features_opt = deepstack_layer_features_opt,
        .deepstack_feature_index_opt = deepstack_feature_index_opt,
    };
}

fn cudaDecodeTargetWork(
    comptime Backend: type,
    comptime execute_decode_with_layer_limit: anytype,
    ctx: *const stage_adapters.DecodeContext,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
) stage_adapters.CudaDecodeLayerWork(Backend, execute_decode_with_layer_limit) {
    return .{
        .ctx = ctx,
        .compute_logits = compute_logits,
        .download_logits = download_logits,
        .logits_out_opt = logits_out_opt,
        .deepstack_layer_features_opt = deepstack_layer_features_opt,
        .deepstack_feature_index_opt = deepstack_feature_index_opt,
        .use_preloaded_input = true,
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
    if (specs.len < 2 or specs.len > 3) return error.InvalidTopologyConfig;
    if (specs[0].backend_kind != .cuda) {
        if (request.hidden_override != null or request.deepstack_layer_features_opt != null or request.deepstack_feature_index_opt != null) {
            return error.InvalidTopologyConfig;
        }
    }

    const root_stage_id = try rootCudaStageId(self);
    const aux_stage_id = try auxCudaStageId(self, root_stage_id);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage0");
    const has_aux_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage1");
    const CpuPtr = if (has_cpu_stage) localCpuStagePointerType(SelfType) else @TypeOf(self);
    const AuxCudaPtr = if (has_aux_cuda_stage) localCudaStagePointerType(SelfType) else @TypeOf(self);
    const cpu_backend: ?CpuPtr = if (has_cpu_stage) self.localCpuStage0() else null;
    const aux_cuda_backend: ?AuxCudaPtr = if (has_aux_cuda_stage) self.localCudaStage1() else null;
    if (aux_stage_id != null and aux_cuda_backend == null) return error.InvalidTopologyConfig;
    if (cpu_backend == null) {
        for (specs) |spec| if (spec.backend_kind == .cpu) return error.InvalidTopologyConfig;
    }

    if (aux_cuda_backend) |stage_backend| try mirrorDecodeStageDescriptorsFromRoot(stage_backend, self, &slot_indices);
    for (specs) |spec| {
        if (spec.backend_kind != .cuda or spec.owns_embedding) continue;
        if (spec.stage_id == root_stage_id) {
            self.activateKvSlot(request.slot_index);
        } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.? and aux_cuda_backend != null) {
            aux_cuda_backend.?.activateKvSlot(request.slot_index);
        }
    }

    const RootWork = stage_adapters.CudaDecodeLayerWork(@TypeOf(self), execute_decode_with_layer_limit);
    const AuxWork = stage_adapters.CudaDecodeLayerWork(AuxCudaPtr, execute_decode_with_layer_limit);
    const RootToRoot = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), @TypeOf(self), RootWork);
    const RootToAux = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), AuxCudaPtr, RootWork);
    const AuxToRoot = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, @TypeOf(self), AuxWork);
    const AuxToAux = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, AuxCudaPtr, AuxWork);
    const CpuToRoot = stage_adapters.CpuDecodeSourceStage(CpuPtr, @TypeOf(self));
    const CpuToAux = stage_adapters.CpuDecodeSourceStage(CpuPtr, AuxCudaPtr);

    var root_to_root: RootToRoot = undefined;
    var root_to_aux: RootToAux = undefined;
    var aux_to_root: AuxToRoot = undefined;
    var aux_to_aux: AuxToAux = undefined;
    var cpu_to_root: CpuToRoot = undefined;
    var cpu_to_aux: CpuToAux = undefined;
    var stages: [3]bridge.LocalStageChainStage = undefined;
    var stage_count: usize = 0;

    for (specs, 0..) |spec, index| {
        const is_final = index + 1 == specs.len;
        const next_stage_id: ?usize = if (is_final) null else specs[index + 1].stage_id;
        switch (spec.backend_kind) {
            .cpu => {
                const cpu = cpu_backend orelse return error.InvalidTopologyConfig;
                if (next_stage_id == root_stage_id) {
                    cpu_to_root = .{ .backend = cpu, .gpu_backend = self, .ctx = &ctx };
                    stages[stage_count] = bridge.localStageAdapter(CpuToRoot, spec.stage_id, &cpu_to_root);
                } else if (aux_stage_id != null and next_stage_id == aux_stage_id.?) {
                    const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                    cpu_to_aux = .{ .backend = cpu, .gpu_backend = aux, .ctx = &ctx };
                    stages[stage_count] = bridge.localStageAdapter(CpuToAux, spec.stage_id, &cpu_to_aux);
                } else {
                    return error.InvalidTopologyConfig;
                }
            },
            .cuda => {
                const boundary = if (!is_final) try stage_adapters.localBoundaryRuntime(self, index) else null;
                if (spec.stage_id == root_stage_id) {
                    const work = if (is_final)
                        cudaDecodeTargetWork(@TypeOf(self), execute_decode_with_layer_limit, &ctx, request.logits_out_opt, request.compute_logits, request.download_logits, deepstackFeaturesForStage(request.deepstack_layer_features_opt, spec.layer_start), request.deepstack_feature_index_opt)
                    else
                        cudaDecodeSourceWork(@TypeOf(self), execute_decode_with_layer_limit, &ctx, request.hidden_override, deepstackFeaturesForStage(request.deepstack_layer_features_opt, spec.layer_start), request.deepstack_feature_index_opt);
                    if (next_stage_id != null and aux_stage_id != null and next_stage_id.? == aux_stage_id.?) {
                        const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                        root_to_aux = .{
                            .backend = self,
                            .target_backend = aux,
                            .activation_slot_index = request.slot_index,
                            .work = work,
                            .peer_copy_synchronization = boundary.?.peer_copy_synchronization,
                        };
                        stages[stage_count] = bridge.localStageAdapter(RootToAux, spec.stage_id, &root_to_aux);
                    } else {
                        root_to_root = .{ .backend = self, .target_backend = self, .activation_slot_index = request.slot_index, .work = work };
                        stages[stage_count] = bridge.localStageAdapter(RootToRoot, spec.stage_id, &root_to_root);
                    }
                } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.?) {
                    const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                    const work = cudaDecodeTargetWork(AuxCudaPtr, execute_decode_with_layer_limit, &ctx, if (is_final) request.logits_out_opt else null, is_final and request.compute_logits, is_final and request.download_logits, deepstackFeaturesForStage(request.deepstack_layer_features_opt, spec.layer_start), request.deepstack_feature_index_opt);
                    if (next_stage_id != null and next_stage_id.? == root_stage_id) {
                        aux_to_root = .{ .backend = aux, .target_backend = self, .activation_slot_index = request.slot_index, .work = work, .peer_copy_synchronization = boundary.?.peer_copy_synchronization };
                        stages[stage_count] = bridge.localStageAdapter(AuxToRoot, spec.stage_id, &aux_to_root);
                    } else {
                        aux_to_aux = .{ .backend = aux, .target_backend = aux, .activation_slot_index = request.slot_index, .work = work };
                        stages[stage_count] = bridge.localStageAdapter(AuxToAux, spec.stage_id, &aux_to_aux);
                    }
                } else {
                    return error.InvalidTopologyConfig;
                }
            },
            else => return error.InvalidTopologyConfig,
        }
        stage_count += 1;
    }

    var payload_specs: [2]bridge.LocalDecodeBoundaryPayloadSpec = undefined;
    for (payload_specs[0 .. specs.len - 1], 0..) |*payload, boundary_index| {
        const source_spec = specs[boundary_index];
        const activation_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
        payload.* = switch (source_spec.backend_kind) {
            .cpu => blk: {
                const cpu = cpu_backend orelse return error.InvalidTopologyConfig;
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
                .location_hint = try cudaLocationHintForStageId(self, aux_cuda_backend, root_stage_id, source_spec.stage_id),
                .image = .device,
            },
            else => return error.InvalidTopologyConfig,
        };
    }

    try bridge.executeLocalDecodePipelineStep(try stage_adapters.localPipelineContext(self), stages[0..stage_count], .{
        .tensor_frame_plan_ref = try stage_adapters.localTopologyTensorFramePlanRef(self),
        .hidden_size = self.d_model,
        .slot_request_ids = self.slot_request_ids[0..],
        .slot_indices = &slot_indices,
        .positions = &positions,
        .boundary_payloads = payload_specs[0 .. specs.len - 1],
    });
}

fn BatchedDecodePipelineWork(
    comptime RootBackend: type,
    comptime Backend: type,
    comptime OutputMode: type,
    comptime compute_stage: anytype,
) type {
    return struct {
        root_backend: RootBackend,
        boundary: stage_adapters.LocalBoundaryRuntimeView,
        location_hint: ?bridge.TensorFramePayloadLocationHint,
        slot_indices: []const usize,
        positions: []const usize,
        tokens: []const u32,
        output_mode: OutputMode,
        plan: BatchedDecodeExecutionPlan,
        active_side: DecodeBoundaryStageSide,

        pub fn execute(work: *@This(), backend: Backend, input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try compute_stage(
                work.root_backend,
                backend,
                work.boundary,
                work.location_hint,
                work.slot_indices,
                work.positions,
                work.active_side,
                work.tokens,
                work.output_mode,
                work.plan,
            );
        }
    };
}

fn CpuBatchedDecodePipelineSourceStage(
    comptime RootBackend: type,
    comptime CpuBackend: type,
    comptime GpuBackend: type,
    comptime IntermediateBackend: type,
    comptime prepare_cpu_segments: anytype,
) type {
    return struct {
        root_backend: RootBackend,
        backend: CpuBackend,
        gpu_backend: GpuBackend,
        intermediate_backend: IntermediateBackend,
        activate_intermediate: bool,
        boundary: stage_adapters.LocalBoundaryRuntimeView,
        tokens: []const u32,
        slot_indices: []const usize,
        positions: []const usize,
        split_layer: usize,
        row_bytes: usize,
        host_segments: [][]const u8,

        pub fn executeLayers(stage: *@This(), input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try prepare_cpu_segments(
                stage.root_backend,
                stage.backend,
                stage.activate_intermediate,
                stage.intermediate_backend,
                stage.boundary,
                stage.tokens,
                stage.slot_indices,
                stage.positions,
                stage.split_layer,
                stage.row_bytes,
                stage.host_segments,
            );
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            for (stage.slot_indices, stage.positions) |slot_index, position| {
                try transport.uploadCpuKvToCudaMirrors(stage.gpu_backend, stage.backend, slot_index, position, 1);
            }
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };
}

pub fn executeBatchedDecodePipeline(
    comptime compute_stage: anytype,
    comptime copy_host_logits: anytype,
    comptime prepare_cpu_segments: anytype,
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
    if (specs.len < 2 or specs.len > 3) return error.InvalidTopologyConfig;
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const root_stage_id = try rootCudaStageId(self);
    const aux_stage_id = try auxCudaStageId(self, root_stage_id);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage0");
    const has_aux_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage1");
    const CpuPtr = if (has_cpu_stage) localCpuStagePointerType(SelfType) else @TypeOf(self);
    const AuxCudaPtr = if (has_aux_cuda_stage) localCudaStagePointerType(SelfType) else @TypeOf(self);
    const cpu_backend: ?CpuPtr = if (has_cpu_stage) self.localCpuStage0() else null;
    const aux_cuda_backend: ?AuxCudaPtr = if (has_aux_cuda_stage) self.localCudaStage1() else null;
    if (aux_stage_id != null and aux_cuda_backend == null) return error.InvalidTopologyConfig;
    if (cpu_backend == null) {
        for (specs) |spec| if (spec.backend_kind == .cpu) return error.InvalidTopologyConfig;
    }

    if (aux_cuda_backend) |stage_backend| {
        try mirrorDecodeStageDescriptorsFromRoot(stage_backend, self, slot_indices);
        const StageBackendType = @TypeOf(stage_backend.*);
        if (comptime @hasField(StageBackendType, "runtime_buffers") and @hasField(StageBackendType, "device")) {
            try stage_backend.runtime_buffers.ensureRowCapacity(&stage_backend.device, tokens.len, stage_backend.fixed_alloc_mode);
        }
    }
    if (comptime @hasField(SelfType, "runtime_buffers") and @hasField(SelfType, "device")) {
        try self.runtime_buffers.ensureRowCapacity(&self.device, tokens.len, self.fixed_alloc_mode);
    }

    var host_segment_scratch = try stage_adapters.HostSegmentScratch.init(
        stage_adapters.backendAllocator(self),
        if (cpu_backend == null) 0 else tokens.len,
    );
    defer host_segment_scratch.deinit();
    var empty_host_segments: [0][]const u8 = .{};
    const host_segments = if (cpu_backend == null) empty_host_segments[0..] else host_segment_scratch.slice(tokens.len);

    const RootWork = BatchedDecodePipelineWork(@TypeOf(self), @TypeOf(self), OutputMode, compute_stage);
    const AuxWork = BatchedDecodePipelineWork(@TypeOf(self), AuxCudaPtr, OutputMode, compute_stage);
    const RootToRoot = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), @TypeOf(self), RootWork);
    const RootToAux = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), AuxCudaPtr, RootWork);
    const AuxToRoot = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, @TypeOf(self), AuxWork);
    const AuxToAux = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, AuxCudaPtr, AuxWork);
    const CpuToRoot = CpuBatchedDecodePipelineSourceStage(@TypeOf(self), CpuPtr, @TypeOf(self), @TypeOf(self), prepare_cpu_segments);
    const CpuToAux = CpuBatchedDecodePipelineSourceStage(@TypeOf(self), CpuPtr, AuxCudaPtr, AuxCudaPtr, prepare_cpu_segments);

    var root_to_root: RootToRoot = undefined;
    var root_to_aux: RootToAux = undefined;
    var aux_to_root: AuxToRoot = undefined;
    var aux_to_aux: AuxToAux = undefined;
    var cpu_to_root: CpuToRoot = undefined;
    var cpu_to_aux: CpuToAux = undefined;
    var stages: [3]bridge.LocalStageChainStage = undefined;
    var stage_count: usize = 0;

    for (specs, 0..) |spec, index| {
        const is_final = index + 1 == specs.len;
        const next_stage_id: ?usize = if (is_final) null else specs[index + 1].stage_id;
        switch (spec.backend_kind) {
            .cpu => {
                const cpu = cpu_backend orelse return error.InvalidTopologyConfig;
                const boundary = try stage_adapters.localBoundaryRuntime(self, index);
                if (next_stage_id == root_stage_id) {
                    cpu_to_root = .{ .root_backend = self, .backend = cpu, .gpu_backend = self, .intermediate_backend = self, .activate_intermediate = false, .boundary = boundary, .tokens = tokens, .slot_indices = slot_indices, .positions = positions, .split_layer = spec.layer_end, .row_bytes = row_bytes, .host_segments = host_segments };
                    stages[stage_count] = bridge.localStageAdapter(CpuToRoot, spec.stage_id, &cpu_to_root);
                } else if (aux_stage_id != null and next_stage_id == aux_stage_id.?) {
                    const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                    cpu_to_aux = .{ .root_backend = self, .backend = cpu, .gpu_backend = aux, .intermediate_backend = aux, .activate_intermediate = true, .boundary = boundary, .tokens = tokens, .slot_indices = slot_indices, .positions = positions, .split_layer = spec.layer_end, .row_bytes = row_bytes, .host_segments = host_segments };
                    stages[stage_count] = bridge.localStageAdapter(CpuToAux, spec.stage_id, &cpu_to_aux);
                } else {
                    return error.InvalidTopologyConfig;
                }
            },
            .cuda => {
                const active_boundary_index = if (is_final) index - 1 else index;
                const boundary = try stage_adapters.localBoundaryRuntime(self, active_boundary_index);
                const active_side: DecodeBoundaryStageSide = if (is_final) .target else .source;
                const source_index = if (is_final) index - 1 else index;
                const source_stage_id = specs[source_index].stage_id;
                const location_hint: ?bridge.TensorFramePayloadLocationHint = if (specs[source_index].backend_kind == .cpu)
                    .{ .cpu = {} }
                else
                    cudaLocationHintForStageId(self, aux_cuda_backend, root_stage_id, source_stage_id) catch null;
                const work_plan = BatchedDecodeExecutionPlan{
                    .allow_staged_internal_execution = true,
                    .use_preloaded_input = !spec.owns_embedding,
                    .compute_logits = is_final,
                    .emit_decode_summary = is_final,
                    .summary_label_override = if (is_final) mode_label else null,
                };
                if (spec.stage_id == root_stage_id) {
                    const work = RootWork{ .root_backend = self, .boundary = boundary, .location_hint = location_hint, .slot_indices = slot_indices, .positions = positions, .tokens = tokens, .output_mode = if (is_final) output_mode else .device_only, .plan = work_plan, .active_side = active_side };
                    if (next_stage_id != null and aux_stage_id != null and next_stage_id.? == aux_stage_id.?) {
                        const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                        root_to_aux = .{ .backend = self, .target_backend = aux, .work = work, .peer_copy_synchronization = boundary.peer_copy_synchronization };
                        stages[stage_count] = bridge.localStageAdapter(RootToAux, spec.stage_id, &root_to_aux);
                    } else {
                        root_to_root = .{ .backend = self, .target_backend = self, .work = work };
                        stages[stage_count] = bridge.localStageAdapter(RootToRoot, spec.stage_id, &root_to_root);
                    }
                } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.?) {
                    const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                    const work = AuxWork{ .root_backend = self, .boundary = boundary, .location_hint = location_hint, .slot_indices = slot_indices, .positions = positions, .tokens = tokens, .output_mode = if (is_final) output_mode else .device_only, .plan = work_plan, .active_side = active_side };
                    if (next_stage_id != null and next_stage_id.? == root_stage_id) {
                        aux_to_root = .{ .backend = aux, .target_backend = self, .work = work, .peer_copy_synchronization = boundary.peer_copy_synchronization };
                        stages[stage_count] = bridge.localStageAdapter(AuxToRoot, spec.stage_id, &aux_to_root);
                    } else {
                        aux_to_aux = .{ .backend = aux, .target_backend = aux, .work = work };
                        stages[stage_count] = bridge.localStageAdapter(AuxToAux, spec.stage_id, &aux_to_aux);
                    }
                } else {
                    return error.InvalidTopologyConfig;
                }
            },
            else => return error.InvalidTopologyConfig,
        }
        stage_count += 1;
    }

    var payload_specs: [2]bridge.LocalDecodeBoundaryPayloadSpec = undefined;
    for (payload_specs[0 .. specs.len - 1], 0..) |*payload, boundary_index| {
        const source_spec = specs[boundary_index];
        const row_transfer_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
        const transfer_bytes = std.math.mul(usize, tokens.len, row_transfer_bytes) catch return error.InvalidArgument;
        payload.* = switch (source_spec.backend_kind) {
            .cpu => .{ .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index), .activation_byte_count = transfer_bytes, .location_hint = .{ .cpu = {} }, .image = .{ .host_segments = host_segments }, .local_device_peer_copy_available = false },
            .cuda => .{ .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index), .activation_byte_count = transfer_bytes, .location_hint = cudaLocationHintForStageId(self, aux_cuda_backend, root_stage_id, source_spec.stage_id) catch null, .image = .device },
            else => return error.InvalidTopologyConfig,
        };
    }

    try bridge.executeLocalDecodePipelineStep(try stage_adapters.localPipelineContext(self), stages[0..stage_count], .{
        .tensor_frame_plan_ref = try stage_adapters.localTopologyTensorFramePlanRef(self),
        .hidden_size = self.d_model,
        .slot_request_ids = self.slot_request_ids[0..],
        .slot_indices = slot_indices,
        .positions = positions,
        .boundary_payloads = payload_specs[0 .. specs.len - 1],
    });

    if (output_mode == .host_logits and aux_stage_id != null and specs[specs.len - 1].stage_id == aux_stage_id.?) {
        try copy_host_logits(self, aux_cuda_backend orelse return error.InvalidTopologyConfig, tokens.len);
    }
}
