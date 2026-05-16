//! Bridge-driven local staged execution for process-local stage chains.

const std = @import("std");
const compute = @import("compute_pkg");
const bridge = @import("../bridge/root.zig");
const transport = @import("../transport/root.zig");
const per_layer_branch_feature = @import("cuda/per_layer_branch.zig");

const engine_weights = @import("cuda/weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;

const common = @import("cuda/exec/common.zig");
const local_decode = @import("local_decode_pipeline.zig");
const stage_adapters = @import("local_stage_adapters.zig");
const resolveStagedPrefillChunkRows = @import("cuda/exec/prefill_route.zig").resolveStagedPrefillChunkRows;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const prefillMath = common.prefillMath;
const prefillChunkContext = common.prefillChunkContext;
const prepareCudaPrefillBackend = common.prepareCudaPrefillBackend;
const populateCudaPrefillInputRows = common.populateCudaPrefillInputRows;
const executeGpuPrefillLayers = common.executeGpuPrefillLayers;
const projectFinalLogitsFromCudaStage = common.projectFinalLogitsFromCudaStage;
const AttentionKernelSet = @import("cuda/runtime/root.zig").AttentionKernelSet;

fn hasActivePerLayerBranch(backend: anytype) bool {
    const BackendType = @TypeOf(backend.*);
    if (!per_layer_branch_feature.hasPerLayerBranchRuntime(backend)) return false;
    if (comptime @hasField(BackendType, "per_layer_branch_runtime")) {
        return backend.per_layer_branch_runtime != null;
    }
    return false;
}

fn allocateSourceEmbeddings(
    allocator: std.mem.Allocator,
    needed: bool,
    total_rows: usize,
    d_model: usize,
) !?[]f32 {
    if (!needed) return null;
    return try allocator.alloc(f32, total_rows * d_model);
}

fn uploadChunkSourceEmbeddings(
    backend: anytype,
    source_embeddings_host: []const f32,
    pos_base: usize,
    rows: usize,
    d_model: usize,
) !compute.cuda.Buffer {
    const se_chunk_offset = pos_base * d_model;
    const se_chunk_f32s = rows * d_model;
    const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
    var se_dst = try bufferSlice(&backend.runtime_buffers.deepstack_add_dev, 0, se_bytes);
    try se_dst.upload(&backend.device, std.mem.sliceAsBytes(source_embeddings_host[se_chunk_offset..][0..se_chunk_f32s]));
    return se_dst;
}

fn ensureCudaChunkBuffers(backend: anytype, rows: usize) !void {
    try backend.runtime_buffers.ensureRowCapacity(&backend.device, rows, backend.fixed_alloc_mode);
    try backend.ensureLayerProgramSlotRowCapacity(rows, backend.fixed_alloc_mode);
}

fn canExecuteConcreteCudaPrefill(comptime BackendType: type) bool {
    return @hasField(BackendType, "loaded") and
        @hasField(BackendType, "runtime_buffers") and
        @hasDecl(BackendType, "tryExecuteLayerProgram");
}

fn activatePrefillKvSlot(backend: anytype, slot_index: usize) !void {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasDecl(BackendType, "activateKvSlot")) return error.InvalidTopologyConfig;
    backend.activateKvSlot(slot_index);
}

fn mirrorPrefillStageDescriptorsFromRoot(stage_backend: anytype, root_backend: anytype, slot_index: usize) !void {
    const StageType = @TypeOf(stage_backend.*);
    if (comptime @hasField(StageType, "state_descriptor_count") and @hasDecl(StageType, "mirrorSlotStateBlocksFrom")) {
        if (stage_backend.state_descriptor_count > 0) try stage_backend.mirrorSlotStateBlocksFrom(root_backend, slot_index);
    }
}

const LocalPrefillEndpointKind = enum {
    cpu,
    cuda,
};

fn PrefillStageEndpoint(
    comptime CudaBackendPtr: type,
    comptime CpuBackendPtr: type,
) type {
    return struct {
        kind: LocalPrefillEndpointKind,
        cuda_backend: ?CudaBackendPtr = null,
        cpu_backend: ?CpuBackendPtr = null,
        slot_index: usize,
        chunk_tokens: []const u32,
        math: common.PrefillMath,
        chunk: common.PrefillChunkContext,
        attention_kernels: AttentionKernelSet,
        per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null,
        source_embeddings_out: ?[]f32 = null,
        branch_enabled: bool = false,
        dump_layer_offset: ?usize = null,
        layer_limit: usize,
        final_hidden_out: *?compute.cuda.Buffer,
        activation_slot_index: usize,
        sequence_start: usize,
        is_final: bool = false,
        compute_logits: bool = false,
        logits_out_opt: ?[]f32 = null,
        peer_copy_synchronization: transport.CudaPeerCopySynchronization = .source_stream,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            switch (stage.kind) {
                .cpu => try stage_adapters.executeCpuPrefillLayerRange(
                    stage.cpu_backend orelse return error.InvalidTopologyConfig,
                    stage.slot_index,
                    stage.chunk_tokens,
                    stage.sequence_start,
                    layer_start,
                    layer_end,
                    layer_start != 0,
                    stage.compute_logits,
                    stage.logits_out_opt,
                    stage.source_embeddings_out,
                ),
                .cuda => {
                    stage.final_hidden_out.* = try executeGpuPrefillLayers(
                        stage.cuda_backend orelse return error.InvalidTopologyConfig,
                        stage.slot_index,
                        stage.chunk_tokens,
                        stage.math,
                        stage.chunk,
                        stage.attention_kernels,
                        stage.per_layer_source_embeddings_opt,
                        stage.branch_enabled,
                        stage.dump_layer_offset,
                        stage.layer_limit,
                    );
                },
            }
        }
    };
}

pub fn executeLocalPrefillPipeline(
    self: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "local_stage_specs")) return error.InvalidTopologyConfig;
    const specs = self.local_stage_specs;
    if (specs.len < 2) return error.InvalidTopologyConfig;
    if (comptime !canExecuteConcreteCudaPrefill(SelfType)) return error.InvalidTopologyConfig;
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;

    const total_rows = tokens.len;
    const math = try prefillMath(self);
    const root_stage_id = try local_decode.rootCudaStageId(self);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage");
    const has_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage");
    const CpuPtr = if (has_cpu_stage) local_decode.localCpuStagePointerType(SelfType) else @TypeOf(self);
    const CudaPtr = if (has_cuda_stage) local_decode.localCudaStagePointerType(SelfType) else @TypeOf(self);

    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);
    var launch_phase_backends: [8]CudaPtr = undefined;
    var launch_phase_values: [8]@TypeOf(previous_launch_phase) = undefined;
    var launch_phase_count: usize = 0;
    defer {
        for (launch_phase_backends[0..launch_phase_count], launch_phase_values[0..launch_phase_count]) |backend, phase| {
            _ = backend.device.setLaunchPhase(phase);
        }
    }

    var chunk_cap_limit = self.prefill_chunk_rows_cap;
    for (specs) |spec| {
        if (spec.backend_kind != .cuda) continue;
        const cuda_backend = if (spec.stage_id == root_stage_id) self else blk: {
            if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
            const local = self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
            try mirrorPrefillStageDescriptorsFromRoot(local, self, slot_index);
            if (launch_phase_count >= launch_phase_backends.len) return error.InvalidTopologyConfig;
            launch_phase_backends[launch_phase_count] = local;
            launch_phase_values[launch_phase_count] = local.device.setLaunchPhase(.prefill);
            launch_phase_count += 1;
            break :blk local;
        };
        const CudaStageType = @TypeOf(cuda_backend.*);
        if (comptime !canExecuteConcreteCudaPrefill(CudaStageType)) return error.InvalidTopologyConfig;
        if (comptime !@hasField(CudaStageType, "device")) return error.InvalidArgument;
        chunk_cap_limit = @min(chunk_cap_limit, cuda_backend.prefill_chunk_rows_cap);
        try prepareCudaPrefillBackend(cuda_backend, total_rows);
        try activatePrefillKvSlot(cuda_backend, slot_index);
    }
    for (specs) |spec| {
        if (spec.backend_kind == .cpu) {
            if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
        }
    }

    const root_attention_kernels = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const chunk_cap = resolveStagedPrefillChunkRows(
        total_rows,
        chunk_cap_limit,
        @import("env_pkg").getenv("TALU_CUDA_PREFILL_CHUNK_ROWS") != null,
    );

    var any_branch_active = hasActivePerLayerBranch(self);
    for (specs) |spec| {
        if (spec.backend_kind != .cuda or spec.stage_id == root_stage_id) continue;
        if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
        if (hasActivePerLayerBranch(self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig)) {
            any_branch_active = true;
        }
    }
    const source_embeddings_host = try allocateSourceEmbeddings(
        self.allocator,
        any_branch_active,
        total_rows,
        math.d_model,
    );
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    if (source_embeddings_host) |se_host| {
        try populatePrefillHiddenFromTokens(self.loaded, tokens, math.d_model, se_host, null);
    }

    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];

        var source_embeddings_by_stage: [16]?compute.cuda.Buffer = .{null} ** 16;
        for (specs) |spec| {
            if (spec.backend_kind != .cuda) continue;
            if (spec.stage_id >= source_embeddings_by_stage.len) return error.InvalidTopologyConfig;
            const cuda_backend = if (spec.stage_id == root_stage_id) self else blk: {
                if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
                break :blk self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
            };
            try ensureCudaChunkBuffers(cuda_backend, rows);
            if (spec.owns_embedding) try populateCudaPrefillInputRows(cuda_backend, chunk_tokens, rows, math.row_bytes);
            if (hasActivePerLayerBranch(cuda_backend)) {
                if (source_embeddings_host) |se_host| {
                    source_embeddings_by_stage[spec.stage_id] = try uploadChunkSourceEmbeddings(cuda_backend, se_host, pos_base, rows, math.d_model);
                } else if (spec.stage_id == root_stage_id and per_layer_branch_feature.hasPerLayerBranchRuntime(cuda_backend)) {
                    source_embeddings_by_stage[spec.stage_id] = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(cuda_backend, rows);
                }
            }
        }

        var final_hidden_by_stage: [16]?compute.cuda.Buffer = .{null} ** 16;
        const Endpoint = PrefillStageEndpoint(CudaPtr, CpuPtr);
        var inline_endpoints: [8]Endpoint = undefined;
        var inline_stages: [8]bridge.LocalStageChainStage = undefined;
        var inline_transport_stages: [8]transport.LocalStageTransportEndpoint = undefined;
        var heap_endpoints: []Endpoint = &.{};
        var heap_stages: []bridge.LocalStageChainStage = &.{};
        var heap_transport_stages: []transport.LocalStageTransportEndpoint = &.{};
        defer {
            if (heap_transport_stages.len != 0) self.allocator.free(heap_transport_stages);
            if (heap_stages.len != 0) self.allocator.free(heap_stages);
            if (heap_endpoints.len != 0) self.allocator.free(heap_endpoints);
        }
        const endpoints = if (specs.len <= inline_endpoints.len)
            inline_endpoints[0..specs.len]
        else blk: {
            heap_endpoints = try self.allocator.alloc(Endpoint, specs.len);
            break :blk heap_endpoints;
        };
        const stages = if (specs.len <= inline_stages.len)
            inline_stages[0..specs.len]
        else blk: {
            heap_stages = try self.allocator.alloc(bridge.LocalStageChainStage, specs.len);
            break :blk heap_stages;
        };
        const transport_stages = if (specs.len <= inline_transport_stages.len)
            inline_transport_stages[0..specs.len]
        else blk: {
            heap_transport_stages = try self.allocator.alloc(transport.LocalStageTransportEndpoint, specs.len);
            break :blk heap_transport_stages;
        };

        for (specs, 0..) |spec, index| {
            const is_final = index + 1 == specs.len;
            const endpoint_kind: LocalPrefillEndpointKind = switch (spec.backend_kind) {
                .cpu => .cpu,
                .cuda => .cuda,
                else => return error.InvalidTopologyConfig,
            };
            const cpu_backend: ?CpuPtr = if (spec.backend_kind == .cpu) blk: {
                if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
                const cpu = self.localCpuStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
                _ = try cpu.ensureLocalPrefillActivationRows(rows);
                break :blk cpu;
            } else null;
            const cuda_backend: ?CudaPtr = if (spec.backend_kind == .cuda) blk: {
                if (spec.stage_id == root_stage_id) break :blk self;
                if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
                break :blk self.localCudaStage(spec.stage_id) orelse return error.InvalidTopologyConfig;
            } else null;
            const boundary = if (!is_final) try stage_adapters.localBoundaryRuntime(self, index) else null;
            if (spec.stage_id >= final_hidden_by_stage.len) return error.InvalidTopologyConfig;
            endpoints[index] = .{
                .kind = endpoint_kind,
                .cuda_backend = cuda_backend,
                .cpu_backend = cpu_backend,
                .slot_index = slot_index,
                .chunk_tokens = chunk_tokens,
                .math = math,
                .chunk = chunk,
                .attention_kernels = if (cuda_backend) |backend|
                    if (spec.stage_id == root_stage_id) root_attention_kernels else buildAttentionKernelSet(backend) catch return error.CudaKernelUnavailable
                else
                    root_attention_kernels,
                .per_layer_source_embeddings_opt = source_embeddings_by_stage[spec.stage_id],
                .source_embeddings_out = if (spec.owns_embedding) blk: {
                    const se_host = source_embeddings_host orelse break :blk null;
                    const offset = pos_base * math.d_model;
                    const len = rows * math.d_model;
                    break :blk se_host[offset..][0..len];
                } else null,
                .branch_enabled = if (cuda_backend) |backend|
                    per_layer_branch_feature.hasPerLayerBranchRuntime(backend)
                else
                    false,
                .dump_layer_offset = if (spec.owns_embedding) null else spec.layer_start,
                .layer_limit = if (cuda_backend) |backend| backend.block_runtime.blocks.len else 0,
                .final_hidden_out = &final_hidden_by_stage[spec.stage_id],
                .activation_slot_index = slot_index,
                .sequence_start = pos_base,
                .is_final = is_final,
                .compute_logits = is_final and pos_base + rows >= total_rows,
                .logits_out_opt = if (is_final and pos_base + rows >= total_rows) logits_out else null,
                .peer_copy_synchronization = if (boundary) |value| value.peer_copy_synchronization else .source_stream,
            };
            stages[index] = bridge.localStageAdapter(Endpoint, spec.stage_id, &endpoints[index]);
            transport_stages[index] = transport.localEndpointTransportAdapter(Endpoint, spec.stage_id, &endpoints[index], .{
                .has_cpu_stage = has_cpu_stage,
                .prepare_cpu_boundary = true,
                .cpu_activation_scope = .prefill_stage,
            });
        }

        const boundary_count = specs.len - 1;
        var inline_payload_specs: [8]bridge.LocalPrefillBoundaryPayloadSpec = undefined;
        var heap_payload_specs: []bridge.LocalPrefillBoundaryPayloadSpec = &.{};
        defer if (heap_payload_specs.len != 0) self.allocator.free(heap_payload_specs);
        const payload_specs = if (boundary_count <= inline_payload_specs.len)
            inline_payload_specs[0..boundary_count]
        else blk: {
            heap_payload_specs = try self.allocator.alloc(bridge.LocalPrefillBoundaryPayloadSpec, boundary_count);
            break :blk heap_payload_specs;
        };

        for (payload_specs, 0..) |*payload, boundary_index| {
            const source_spec = specs[boundary_index];
            const row_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
            const transfer_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
            payload.* = switch (source_spec.backend_kind) {
                .cpu => blk: {
                    if (comptime !has_cpu_stage) return error.InvalidTopologyConfig;
                    const cpu = self.localCpuStage(source_spec.stage_id) orelse return error.InvalidTopologyConfig;
                    const host_bytes = cpu.localPrefillActivationBytes(transfer_bytes);
                    if (host_bytes.len != transfer_bytes) return error.InvalidArgument;
                    break :blk .{
                        .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                        .slot_index = slot_index,
                        .sequence_start = pos_base,
                        .token_count = rows,
                        .activation_byte_count = transfer_bytes,
                        .location_hint = .{ .cpu = {} },
                        .image = .{ .host_bytes = host_bytes },
                        .local_device_peer_copy_available = false,
                    };
                },
                .cuda => .{
                    .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                    .slot_index = slot_index,
                    .sequence_start = pos_base,
                    .token_count = rows,
                    .activation_byte_count = transfer_bytes,
                    .location_hint = try local_decode.cudaLocationHintForStageId(self, root_stage_id, source_spec.stage_id),
                    .image = .device,
                },
                else => return error.InvalidTopologyConfig,
            };
        }

        try bridge.executeLocalPrefillPipelineStepWithEndpointRegistry(try stage_adapters.localPipelineContext(self), .{ .endpoints = stages, .transport_endpoints = transport_stages }, .{
            .tensor_frame_plan_ref = try stage_adapters.localStageTensorFramePlanRef(self),
            .hidden_size = self.d_model,
            .slot_request_ids = self.slot_request_ids[0..],
            .boundary_payloads = payload_specs,
        }, false);

        if (pos_base + rows >= total_rows) {
            const final_spec = specs[specs.len - 1];
            switch (final_spec.backend_kind) {
                .cpu => {},
                .cuda => {
                    if (final_spec.stage_id >= final_hidden_by_stage.len) return error.InvalidTopologyConfig;
                    const final_cuda_backend = if (final_spec.stage_id == root_stage_id) self else blk: {
                        if (comptime !has_cuda_stage) return error.InvalidTopologyConfig;
                        break :blk self.localCudaStage(final_spec.stage_id) orelse return error.InvalidTopologyConfig;
                    };
                    try projectFinalLogitsFromCudaStage(final_cuda_backend, final_hidden_by_stage[final_spec.stage_id] orelse return error.InvalidTopologyConfig, rows, math.row_bytes, chunk.last_position, logits_out, null);
                },
                else => return error.InvalidTopologyConfig,
            }
        }

        pos_base += rows;
    }
}

test "executeLocalPrefillPipeline rejects missing placement adapters" {
    const MockBackend = struct {};
    var backend = MockBackend{};
    const tokens = [_]u32{1};
    var logits: [1]f32 = undefined;

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        executeLocalPrefillPipeline(&backend, tokens[0..], 0, logits[0..]),
    );
}
