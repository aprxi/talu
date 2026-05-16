//! Bridge-driven local staged execution for CUDA-backed local topologies.
//!
//! CPU->CUDA, CUDA->CUDA, and CPU->CUDA->CUDA are placement instances of this local stage-chain route.

const std = @import("std");
const compute = @import("compute_pkg");
const bridge = @import("../../../bridge/root.zig");
const transport = @import("../../../transport/root.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;

const common = @import("common.zig");
const stage_adapters = @import("stage_adapters.zig");
const resolveStagedPrefillChunkRows = @import("prefill_route.zig").resolveStagedPrefillChunkRows;
const uploadCpuKvToMirrors = transport.uploadCpuKvToCudaMirrors;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const prefillMath = common.prefillMath;
const prefillChunkContext = common.prefillChunkContext;
const prepareCudaPrefillBackend = common.prepareCudaPrefillBackend;
const populateCudaPrefillInputRows = common.populateCudaPrefillInputRows;
const executeGpuPrefillLayers = common.executeGpuPrefillLayers;
const projectFinalLogitsFromCudaStage = common.projectFinalLogitsFromCudaStage;
const AttentionKernelSet = @import("../runtime/root.zig").AttentionKernelSet;

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

fn PrefillPipelineWork(comptime Backend: type) type {
    return struct {
        slot_index: usize,
        chunk_tokens: []const u32,
        math: common.PrefillMath,
        chunk: common.PrefillChunkContext,
        attention_kernels: AttentionKernelSet,
        per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null,
        branch_enabled: bool = false,
        dump_layer_offset: ?usize = null,
        layer_limit: usize,
        final_hidden_out: *?compute.cuda.Buffer,

        pub fn execute(work: *@This(), backend: Backend, input: []const u8, _: usize, _: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            work.final_hidden_out.* = try executeGpuPrefillLayers(
                backend,
                work.slot_index,
                work.chunk_tokens,
                work.math,
                work.chunk,
                work.attention_kernels,
                work.per_layer_source_embeddings_opt,
                work.branch_enabled,
                work.dump_layer_offset,
                work.layer_limit,
            );
        }
    };
}

fn prefillPipelineWork(
    comptime Backend: type,
    slot_index: usize,
    chunk_tokens: []const u32,
    math: common.PrefillMath,
    chunk: common.PrefillChunkContext,
    attention_kernels: AttentionKernelSet,
    per_layer_source_embeddings_opt: ?compute.cuda.Buffer,
    branch_enabled: bool,
    dump_layer_offset: ?usize,
    layer_limit: usize,
    final_hidden_out: *?compute.cuda.Buffer,
) PrefillPipelineWork(Backend) {
    return .{
        .slot_index = slot_index,
        .chunk_tokens = chunk_tokens,
        .math = math,
        .chunk = chunk,
        .attention_kernels = attention_kernels,
        .per_layer_source_embeddings_opt = per_layer_source_embeddings_opt,
        .branch_enabled = branch_enabled,
        .dump_layer_offset = dump_layer_offset,
        .layer_limit = layer_limit,
        .final_hidden_out = final_hidden_out,
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
    if (specs.len < 2 or specs.len > 3) return error.InvalidTopologyConfig;
    if (comptime !canExecuteConcreteCudaPrefill(SelfType)) return error.InvalidTopologyConfig;
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;

    const total_rows = tokens.len;
    const math = try prefillMath(self);
    const root_stage_id = try rootCudaStageId(self);
    const aux_stage_id = try auxCudaStageId(self, root_stage_id);
    const has_cpu_stage = comptime @hasDecl(SelfType, "localCpuStage0");
    const has_aux_cuda_stage = comptime @hasDecl(SelfType, "localCudaStage1");
    const CpuPtr = if (has_cpu_stage) localCpuStagePointerType(SelfType) else @TypeOf(self);
    const AuxCudaPtr = if (has_aux_cuda_stage) localCudaStagePointerType(SelfType) else @TypeOf(self);
    const cpu_stage0: ?CpuPtr = if (has_cpu_stage) self.localCpuStage0() else null;
    const aux_cuda_backend: ?AuxCudaPtr = if (has_aux_cuda_stage) self.localCudaStage1() else null;
    if (aux_stage_id != null and aux_cuda_backend == null) return error.InvalidTopologyConfig;
    if (cpu_stage0 == null) {
        for (specs) |spec| if (spec.backend_kind == .cpu) return error.InvalidTopologyConfig;
    }
    if (aux_cuda_backend) |aux| {
        const AuxType = @TypeOf(aux.*);
        if (comptime !canExecuteConcreteCudaPrefill(AuxType)) return error.InvalidTopologyConfig;
        if (comptime !@hasField(AuxType, "device")) return error.InvalidArgument;
    }

    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);
    const previous_aux_launch_phase = if (aux_cuda_backend) |aux| aux.device.setLaunchPhase(.prefill) else null;
    defer if (aux_cuda_backend) |aux| {
        if (previous_aux_launch_phase) |phase| _ = aux.device.setLaunchPhase(phase);
    };

    if (aux_cuda_backend) |aux| try mirrorPrefillStageDescriptorsFromRoot(aux, self, slot_index);
    try prepareCudaPrefillBackend(self, total_rows);
    if (aux_cuda_backend) |aux| try prepareCudaPrefillBackend(aux, total_rows);
    for (specs) |spec| {
        if (spec.backend_kind != .cuda) continue;
        if (spec.stage_id == root_stage_id) {
            try activatePrefillKvSlot(self, slot_index);
        } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.?) {
            try activatePrefillKvSlot(aux_cuda_backend orelse return error.InvalidTopologyConfig, slot_index);
        }
    }

    const root_attention_kernels = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const aux_attention_kernels: ?AttentionKernelSet = if (aux_cuda_backend) |aux|
        buildAttentionKernelSet(aux) catch return error.CudaKernelUnavailable
    else
        null;
    var chunk_cap_limit = self.prefill_chunk_rows_cap;
    if (aux_cuda_backend) |aux| chunk_cap_limit = @min(chunk_cap_limit, aux.prefill_chunk_rows_cap);
    const chunk_cap = resolveStagedPrefillChunkRows(
        total_rows,
        chunk_cap_limit,
        @import("env_pkg").getenv("TALU_CUDA_PREFILL_CHUNK_ROWS") != null,
    );

    const aux_branch_active = if (aux_cuda_backend) |aux| hasActivePerLayerBranch(aux) else false;
    const source_embeddings_host = try allocateSourceEmbeddings(
        self.allocator,
        hasActivePerLayerBranch(self) or aux_branch_active,
        total_rows,
        math.d_model,
    );
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    var prefill_buffer: ?[]f32 = null;
    defer if (prefill_buffer) |buffer| self.allocator.free(buffer);

    if (specs[0].backend_kind == .cpu) {
        const cpu = cpu_stage0 orelse return error.InvalidTopologyConfig;
        prefill_buffer = try self.allocator.alloc(f32, total_rows * math.d_model);
        try cpu.prefillSlotLayerRange(slot_index, tokens, prefill_buffer.?, specs[0].layer_end, source_embeddings_host);
        const first_cuda_stage_id = specs[1].stage_id;
        if (first_cuda_stage_id == root_stage_id) {
            try uploadCpuKvToMirrors(self, cpu, slot_index, 0, total_rows);
        } else if (aux_stage_id != null and first_cuda_stage_id == aux_stage_id.?) {
            try uploadCpuKvToMirrors(aux_cuda_backend orelse return error.InvalidTopologyConfig, cpu, slot_index, 0, total_rows);
        } else {
            return error.InvalidTopologyConfig;
        }
    } else if (source_embeddings_host) |se_host| {
        try populatePrefillHiddenFromTokens(self.loaded, tokens, math.d_model, se_host, null);
    }

    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];
        const chunk_host_bytes = if (prefill_buffer) |buffer| blk: {
            const chunk_offset = pos_base * math.d_model;
            const chunk_f32s = rows * math.d_model;
            break :blk std.mem.sliceAsBytes(buffer[chunk_offset..][0..chunk_f32s]);
        } else &[_]u8{};

        var root_source_embeddings: ?compute.cuda.Buffer = null;
        var aux_source_embeddings: ?compute.cuda.Buffer = null;
        for (specs) |spec| {
            if (spec.backend_kind != .cuda) continue;
            if (spec.stage_id == root_stage_id) {
                try ensureCudaChunkBuffers(self, rows);
                if (spec.owns_embedding) try populateCudaPrefillInputRows(self, chunk_tokens, rows, math.row_bytes);
                if (hasActivePerLayerBranch(self)) {
                    if (source_embeddings_host) |se_host| {
                        root_source_embeddings = try uploadChunkSourceEmbeddings(self, se_host, pos_base, rows, math.d_model);
                    } else if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                        root_source_embeddings = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
                    }
                }
            } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.?) {
                const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                try ensureCudaChunkBuffers(aux, rows);
                if (hasActivePerLayerBranch(aux)) {
                    if (source_embeddings_host) |se_host| {
                        aux_source_embeddings = try uploadChunkSourceEmbeddings(aux, se_host, pos_base, rows, math.d_model);
                    }
                }
            } else {
                return error.InvalidTopologyConfig;
            }
        }

        const RootWork = PrefillPipelineWork(@TypeOf(self));
        const AuxWork = PrefillPipelineWork(AuxCudaPtr);
        const RootToRoot = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), @TypeOf(self), RootWork);
        const RootToAux = stage_adapters.CudaLocalPipelineStage(@TypeOf(self), AuxCudaPtr, RootWork);
        const AuxToRoot = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, @TypeOf(self), AuxWork);
        const AuxToAux = stage_adapters.CudaLocalPipelineStage(AuxCudaPtr, AuxCudaPtr, AuxWork);
        const Noop = transport.NoopActivationStage;

        var root_final_hidden: ?compute.cuda.Buffer = null;
        var aux_final_hidden: ?compute.cuda.Buffer = null;
        var root_to_root: RootToRoot = undefined;
        var root_to_aux: RootToAux = undefined;
        var aux_to_root: AuxToRoot = undefined;
        var aux_to_aux: AuxToAux = undefined;
        var noop = Noop{};
        var stages: [3]bridge.LocalStageChainStage = undefined;
        var stage_count: usize = 0;

        for (specs, 0..) |spec, index| {
            const is_final = index + 1 == specs.len;
            const next_stage_id: ?usize = if (is_final) null else specs[index + 1].stage_id;
            switch (spec.backend_kind) {
                .cpu => {
                    stages[stage_count] = bridge.localStageAdapter(Noop, spec.stage_id, &noop);
                },
                .cuda => {
                    const boundary = if (!is_final) try stage_adapters.localBoundaryRuntime(self, index) else null;
                    if (spec.stage_id == root_stage_id) {
                        const work = prefillPipelineWork(
                            @TypeOf(self),
                            slot_index,
                            chunk_tokens,
                            math,
                            chunk,
                            root_attention_kernels,
                            root_source_embeddings,
                            per_layer_branch_feature.hasPerLayerBranchRuntime(self),
                            if (spec.owns_embedding) null else spec.layer_start,
                            self.block_runtime.blocks.len,
                            &root_final_hidden,
                        );
                        if (next_stage_id != null and aux_stage_id != null and next_stage_id.? == aux_stage_id.?) {
                            root_to_aux = .{
                                .backend = self,
                                .target_backend = aux_cuda_backend orelse return error.InvalidTopologyConfig,
                                .work = work,
                                .activation_slot_index = slot_index,
                                .peer_copy_synchronization = boundary.?.peer_copy_synchronization,
                            };
                            stages[stage_count] = bridge.localStageAdapter(RootToAux, spec.stage_id, &root_to_aux);
                        } else {
                            root_to_root = .{
                                .backend = self,
                                .target_backend = self,
                                .work = work,
                                .activation_slot_index = slot_index,
                            };
                            stages[stage_count] = bridge.localStageAdapter(RootToRoot, spec.stage_id, &root_to_root);
                        }
                    } else if (aux_stage_id != null and spec.stage_id == aux_stage_id.?) {
                        const aux = aux_cuda_backend orelse return error.InvalidTopologyConfig;
                        const work = prefillPipelineWork(
                            AuxCudaPtr,
                            slot_index,
                            chunk_tokens,
                            math,
                            chunk,
                            aux_attention_kernels orelse return error.InvalidTopologyConfig,
                            aux_source_embeddings,
                            per_layer_branch_feature.hasPerLayerBranchRuntime(aux),
                            if (spec.owns_embedding) null else spec.layer_start,
                            aux.block_runtime.blocks.len,
                            &aux_final_hidden,
                        );
                        if (next_stage_id != null and next_stage_id.? == root_stage_id) {
                            aux_to_root = .{
                                .backend = aux,
                                .target_backend = self,
                                .work = work,
                                .activation_slot_index = slot_index,
                                .peer_copy_synchronization = boundary.?.peer_copy_synchronization,
                            };
                            stages[stage_count] = bridge.localStageAdapter(AuxToRoot, spec.stage_id, &aux_to_root);
                        } else {
                            aux_to_aux = .{
                                .backend = aux,
                                .target_backend = aux,
                                .work = work,
                                .activation_slot_index = slot_index,
                            };
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

        var payload_specs: [2]bridge.LocalPrefillBoundaryPayloadSpec = undefined;
        for (payload_specs[0 .. specs.len - 1], 0..) |*payload, boundary_index| {
            const source_spec = specs[boundary_index];
            const row_bytes = try stage_adapters.localBoundaryActivationByteCount(self, boundary_index);
            const transfer_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
            payload.* = switch (source_spec.backend_kind) {
                .cpu => .{
                    .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                    .slot_index = slot_index,
                    .sequence_start = pos_base,
                    .token_count = rows,
                    .activation_byte_count = chunk_host_bytes.len,
                    .location_hint = .{ .cpu = {} },
                    .image = .{ .host_bytes = chunk_host_bytes },
                    .local_device_peer_copy_available = false,
                },
                .cuda => .{
                    .frame = try stage_adapters.localBoundaryFrameSpec(self, boundary_index),
                    .slot_index = slot_index,
                    .sequence_start = pos_base,
                    .token_count = rows,
                    .activation_byte_count = transfer_bytes,
                    .location_hint = try cudaLocationHintForStageId(self, aux_cuda_backend, root_stage_id, source_spec.stage_id),
                    .image = .device,
                },
                else => return error.InvalidTopologyConfig,
            };
        }

        try bridge.executeLocalPrefillPipelineStep(try stage_adapters.localPipelineContext(self), stages[0..stage_count], .{
            .tensor_frame_plan_ref = try stage_adapters.localTopologyTensorFramePlanRef(self),
            .hidden_size = self.d_model,
            .slot_request_ids = self.slot_request_ids[0..],
            .boundary_payloads = payload_specs[0 .. specs.len - 1],
        });

        if (pos_base + rows >= total_rows) {
            const final_stage_id = specs[specs.len - 1].stage_id;
            if (final_stage_id == root_stage_id) {
                try projectFinalLogitsFromCudaStage(self, root_final_hidden orelse return error.InvalidTopologyConfig, rows, math.row_bytes, chunk.last_position, logits_out, null);
            } else if (aux_stage_id != null and final_stage_id == aux_stage_id.?) {
                try projectFinalLogitsFromCudaStage(aux_cuda_backend orelse return error.InvalidTopologyConfig, aux_final_hidden orelse return error.InvalidTopologyConfig, rows, math.row_bytes, chunk.last_position, logits_out, "cuda_pipeline2_final_norm_host");
            } else {
                return error.InvalidTopologyConfig;
            }
        }

        pos_base += rows;
    }
}

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

fn cudaLocationHintForStageId(
    self: anytype,
    aux_cuda_backend: anytype,
    root_stage_id: usize,
    stage_id: usize,
) !bridge.TensorFramePayloadLocationHint {
    if (stage_id == root_stage_id) return try stage_adapters.cudaPayloadLocationHint(self);
    return try stage_adapters.cudaPayloadLocationHint(aux_cuda_backend orelse return error.InvalidTopologyConfig);
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
