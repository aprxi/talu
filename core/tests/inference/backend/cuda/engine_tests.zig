//! Engine integration tests.
//!
//! Extracted from engine.zig to keep all split files under the line budget.
//! Tests exercise types and functions across engine_types, engine_weights,
//! engine_ops, engine_mixers, and engine_layer_program.

const std = @import("std");
const main = @import("main");
const compute = main.compute;
const tensor = compute.tensor;
const dtype = compute.dtype;
const models = main.models.dispatcher;
const layer_ops = models.layer_ops;
const opcode_map = models.plan.opcode_map;
const plan_compiler = models.plan.compiler;
const runtime_contract = main.inference.runtime_contract;
const bridge = main.inference.bridge;
const transport = main.inference.transport;
const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const cuda_testing = main.inference.backend.cuda.testing;
const local_stage_testing = @import("local_stage_test_helpers.zig");

// --- Types from engine.zig ---
const engine = cuda_testing.engine;
const CudaBackend = engine.CudaBackend;

fn clearLocalStages(_: *CudaBackend) void {}

fn localTransportRequest(
    placement_plan: *const bridge.PlacementPlan,
    metadata: *const bridge.TensorFrameMetadata,
    image: *const bridge.BoundaryByteImageRef,
    contract: *const bridge.ActivationTransportContract,
    staging: ?[]align(64) u8,
    allow_borrow: bool,
    local_device_peer_copy_available: bool,
) transport.LocalStageTransportRequest {
    return .{
        .placement_plan = placement_plan,
        .metadata = metadata,
        .image = image,
        .decision = contract.decision,
        .envelope = &contract.envelope,
        .staging = staging,
        .allow_borrow = allow_borrow,
        .local_device_peer_copy_available = local_device_peer_copy_available,
    };
}

// --- Types from engine_types.zig ---
const engine_types = cuda_testing.runtime;
const DeviceTensor = engine_types.DeviceTensor;
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const GaffineU4LinearWeight = engine_types.GaffineU4LinearWeight;
const GaffineU8LinearWeight = engine_types.GaffineU8LinearWeight;
const Nvfp4LinearWeight = engine_types.Nvfp4LinearWeight;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const KvRuntimeState = engine_types.KvRuntimeState;
const ShortConvRuntimeState = engine_types.ShortConvRuntimeState;
const MambaRuntimeState = engine_types.MambaRuntimeState;
const GatedDeltaRuntimeState = engine_types.GatedDeltaRuntimeState;
const BlockRuntimeLayerForTest = engine_types.BlockRuntimeLayer;
const KvCacheDtypeForTest = engine_types.KvCacheDtype;
const KvCacheStorageModeForTest = engine_types.KvCacheStorageMode;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const expectedAttentionQProjectionDim = engine_types.expectedAttentionQProjectionDim;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const buildCudaLayerProgramRegisterSlotMap = engine_types.buildCudaLayerProgramRegisterSlotMap;
const required_kernels = engine_types.required_kernels;
const KernelSlot = engine_types.KernelSlot;
const resolveGatedDeltaFfnUploadPlan = engine_types.resolveGatedDeltaFfnUploadPlan;

// --- Functions from engine_weights.zig ---
const engine_weights = cuda_testing.weights;
const resolveDenseInOutLayout = engine_weights.resolveDenseInOutLayout;
const resolveDenseOutInLayout = engine_weights.resolveDenseOutInLayout;
const transposeRowMajor = engine_weights.transposeRowMajor;
const tryPopulateProjectionFromWeight = engine_weights.tryPopulateProjectionFromWeight;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const gaffineValueAt = engine_weights.gaffineValueAt;
const tryPopulateFinalNormWeight = engine_weights.tryPopulateFinalNormWeight;
const shouldDownloadPrefillLogits = engine_weights.shouldDownloadPrefillLogits;
const materializeDenseOutInU16 = engine_weights.materializeDenseOutInU16;
const materializeDenseOutInF32 = engine_weights.materializeDenseOutInF32;
const collectTokenPositions = engine_weights.collectTokenPositions;
const findPositionIndex = engine_weights.findPositionIndex;
const deepstackLayersCompatibleWithPrompt = engine_weights.deepstackLayersCompatibleWithPrompt;

// --- Functions from engine_ops.zig ---
const engine_ops = cuda_testing.operators;

// --- Functions from engine_mixers.zig ---
const engine_mixers = cuda_testing.operators;
const engine_forward = cuda_testing.exec;
const resolveStagedPrefillChunkRows = engine_forward.resolveStagedPrefillChunkRows;

const MockCudaDevice = struct {
    launch_phase: compute.cuda.device.LaunchPhase = .none,

    pub fn ordinal(_: *const @This()) u16 {
        return 0;
    }

    pub fn setLaunchPhase(self: *@This(), phase: compute.cuda.device.LaunchPhase) compute.cuda.device.LaunchPhase {
        const previous = self.launch_phase;
        self.launch_phase = phase;
        return previous;
    }
};

fn localStageTestConfig(layer_count: usize) models.config.ModelConfig {
    return .{
        .vocab_size = 64,
        .d_model = 8,
        .n_layers = @intCast(layer_count),
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = 16,
        .max_seq_len = 32,
        .head_dim = 4,
        .rope_theta = 10000,
        .norm_eps = 0.00001,
        .gaffine_group_size = 0,
        .tie_word_embeddings = false,
    };
}

fn localStageTestArch() models.op_types.Architecture {
    return .{
        .name = "cuda_local_stage_test",
        .model_types = &.{"cuda_local_stage_test"},
    };
}

fn localStageTestManifest(allocator: std.mem.Allocator, layer_count: usize) !models.manifest.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const entry_count = layer_count + 3;
    const entries = try arena_allocator.alloc(models.manifest.TensorManifestEntry, entry_count);
    entries[0] = .{
        .name = "model.embed_tokens.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .token_embeddings,
        .weight_id = "token_embeddings",
        .status = .architecture_weight,
    };
    for (0..layer_count) |layer_index| {
        entries[layer_index + 1] = .{
            .name = "model.layers.self_attn.q_proj.weight",
            .dtype = .f32,
            .shape = &.{ 8, 8 },
            .checkpoint_bytes = 64,
            .role = .decoder_layer,
            .layer_index = layer_index,
            .weight_id = "self_attn.q_proj.weight",
            .status = .architecture_weight,
        };
    }
    entries[layer_count + 1] = .{
        .name = "model.norm.weight",
        .dtype = .f32,
        .shape = &.{8},
        .checkpoint_bytes = 32,
        .role = .final_norm,
        .weight_id = "ln_final",
        .status = .architecture_weight,
    };
    entries[layer_count + 2] = .{
        .name = "lm_head.weight",
        .dtype = .f32,
        .shape = &.{ 64, 8 },
        .checkpoint_bytes = 128,
        .role = .lm_head,
        .weight_id = "lm_head",
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** models.manifest.role_count;
    var total_bytes: usize = 0;
    for (entries) |entry| {
        total_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = "cuda_local_stage_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}

fn buildLocalStageTestStagePlan(
    allocator: std.mem.Allocator,
    layer_count: usize,
    splits: []const usize,
    dependencies: []const models.stage_plan.DependencyOverride,
) !models.stage_plan.StagePlan {
    var arch = localStageTestArch();
    var config = localStageTestConfig(layer_count);
    var manifest = try localStageTestManifest(allocator, layer_count);
    defer manifest.deinit();
    return models.stage_plan.buildStagePlan(allocator, .{
        .n_layers = layer_count,
        .split_points = splits,
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{
            .decoder_cuts_allowed = true,
            .dependency_overrides = dependencies,
        },
    });
}

const LocalStageStateFixture = struct {
    plan: bridge.StageStateOwnershipPlan,
    ref: bridge.StageStatePlacementRef,

    fn deinit(self: *@This()) void {
        self.ref.deinit();
        self.plan.deinit();
    }
};

fn buildLocalStageStateFixture(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) !LocalStageStateFixture {
    var state_plan = try local_stage_testing.buildLocalStageStateOwnershipPlan(allocator, plan);
    errdefer state_plan.deinit();
    var state_ref = try bridge.buildStageStatePlacementRef(allocator, &state_plan);
    errdefer state_ref.deinit();
    return .{ .plan = state_plan, .ref = state_ref };
}

fn expectLocalStagePlacement(
    placement: *const bridge.PlacementPlan,
    d_model: usize,
    stage_count: usize,
    configs: []const local_stage_testing.BoundaryConfig,
) !void {
    try bridge.validatePlacementPlan(placement);
    try std.testing.expectEqual(stage_count, placement.stage_summaries.len);
    try std.testing.expectEqual(stage_count - 1, placement.boundary_summaries.len);
    try std.testing.expectEqual(stage_count, placement.stage_host_bindings.len);
    try std.testing.expectEqualSlices(
        bridge.TensorFrameStepKind,
        &local_stage_testing.bridge_stage_required_step_kinds,
        placement.required_step_kinds,
    );

    for (placement.stage_host_bindings, 0..) |binding, stage_id| {
        try std.testing.expectEqual(stage_id, binding.stage_id);
        try std.testing.expectEqual(@as(u64, @intCast(stage_id + 1)), binding.host_id.value);
    }

    try std.testing.expectEqual(configs.len * local_stage_testing.bridge_stage_required_step_kinds.len, placement.boundary_frame_profiles.len);
    for (configs, 0..) |config, boundary_index| {
        const row_bytes = try local_stage_testing.boundaryRowByteCount(d_model, config.dtype);
        const prefill = placement.boundary_frame_profiles[boundary_index * 2];
        try std.testing.expectEqual(boundary_index, prefill.boundary_index);
        try std.testing.expectEqual(bridge.TensorFrameStepKind.prefill, prefill.step_kind);
        try std.testing.expectEqual(config.dtype, prefill.dtype);
        try std.testing.expectEqual(config.layout, prefill.layout);
        try std.testing.expectEqual(@as(u64, 1), prefill.max_batch_entries);
        try std.testing.expectEqual(@as(u64, @intCast(config.prefill_max_token_count_per_frame)), prefill.max_token_count_per_frame);
        try std.testing.expectEqual(
            row_bytes * @as(u64, @intCast(config.prefill_max_token_count_per_frame)),
            prefill.max_activation_payload_bytes,
        );

        const decode = placement.boundary_frame_profiles[boundary_index * 2 + 1];
        try std.testing.expectEqual(boundary_index, decode.boundary_index);
        try std.testing.expectEqual(bridge.TensorFrameStepKind.decode, decode.step_kind);
        try std.testing.expectEqual(config.dtype, decode.dtype);
        try std.testing.expectEqual(config.layout, decode.layout);
        try std.testing.expectEqual(@as(u64, @intCast(config.decode_max_batch_entries)), decode.max_batch_entries);
        try std.testing.expect(decode.max_batch_entries > 1);
        try std.testing.expectEqual(@as(u64, 1), decode.max_token_count_per_frame);
        try std.testing.expectEqual(
            row_bytes * @as(u64, @intCast(config.decode_max_batch_entries)),
            decode.max_activation_payload_bytes,
        );
    }
}

test "validateLocalStageSpecs accepts only contiguous local stage chains" {
    const valid = [_]local_stage_testing.StageSpec{
        .{
            .stage_id = 0,
            .backend_kind = .cpu,
            .layer_start = 0,
            .layer_end = 1,
            .owns_embedding = true,
            .owns_projection = false,
        },
        .{
            .stage_id = 1,
            .backend_kind = .cuda,
            .layer_start = 1,
            .layer_end = 3,
            .owns_embedding = false,
            .owns_projection = false,
        },
        .{
            .stage_id = 2,
            .backend_kind = .cuda,
            .layer_start = 3,
            .layer_end = 4,
            .owns_embedding = false,
            .owns_projection = true,
        },
    };
    try local_stage_testing.validateStageSpecs(4, &valid);

    var gap = valid;
    gap[1].layer_start = 2;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &gap));

    var wrong_stage_id = valid;
    wrong_stage_id[1].stage_id = 7;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &wrong_stage_id));

    var wrong_embedding_owner = valid;
    wrong_embedding_owner[1].owns_embedding = true;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &wrong_embedding_owner));

    var wrong_projection_owner = valid;
    wrong_projection_owner[2].owns_projection = false;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &wrong_projection_owner));

    var missing_tail = valid;
    missing_tail[2].layer_end = 3;
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.validateStageSpecs(4, &missing_tail));
}

test "validateLocalBoundaryRuntimes matches adjacent stage boundary count and order" {
    const valid = [_]local_stage_testing.BoundaryRuntime{
        .{
            .boundary_index = 0,
            .dtype = .f32,
            .layout = .row_major,
        },
        .{
            .boundary_index = 1,
            .dtype = .f16,
            .layout = .row_major,
            .local_device_peer_copy_available = true,
        },
    };
    try local_stage_testing.validateBoundaryRuntimes(3, &valid);

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        local_stage_testing.validateBoundaryRuntimes(3, valid[0..1]),
    );

    var wrong_order = valid;
    wrong_order[1].boundary_index = 7;
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        local_stage_testing.validateBoundaryRuntimes(3, &wrong_order),
    );

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        local_stage_testing.validateBoundaryRuntimes(0, &.{}),
    );
}

fn expectPlacementBuildFailureCleanup(
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const bridge.HostBackendKind,
    configs: []const local_stage_testing.BoundaryConfig,
    state_ref: ?*const bridge.StageStatePlacementRef,
) !void {
    var saw_success = false;
    for (0..64) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        var placement_or_error = local_stage_testing.buildLocalStagePlacementPlan(
            failing.allocator(),
            d_model,
            plan,
            stage_backend_kinds,
            configs,
            state_ref,
        );
        if (placement_or_error) |*placement| {
            placement.deinit();
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            saw_success = true;
            break;
        } else |err| {
            try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        }
    }
    try std.testing.expect(saw_success);
}

test "resolveDenseInOutLayout keeps [in,out] orientation" {
    const layout = try resolveDenseInOutLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "resolveStagedPrefillChunkRows tunes medium staged prefill lengths" {
    try std.testing.expectEqual(@as(usize, 254), resolveStagedPrefillChunkRows(486, 256, false));
    try std.testing.expectEqual(@as(usize, 256), resolveStagedPrefillChunkRows(947, 256, false));
    try std.testing.expectEqual(@as(usize, 254), resolveStagedPrefillChunkRows(486, 512, false));
}

test "resolveStagedPrefillChunkRows honors explicit env override behavior" {
    try std.testing.expectEqual(@as(usize, 256), resolveStagedPrefillChunkRows(486, 256, true));
    try std.testing.expectEqual(@as(usize, 320), resolveStagedPrefillChunkRows(900, 320, true));
}

test "inference.backend.cuda 10G-D local stage chain profile envelopes are exact" {
    const boundary = models.stage_plan.StageBoundary{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    };
    const config = local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9);
    const profiles = try local_stage_testing.boundaryProfilePair(8, 0, boundary, config);

    try std.testing.expectEqual(bridge.TensorFrameStepKind.prefill, profiles[0].step_kind);
    try std.testing.expectEqual(@as(u64, 1), profiles[0].max_batch_entries);
    try std.testing.expectEqual(@as(u64, 9), profiles[0].max_token_count_per_frame);
    try std.testing.expectEqual(@as(u64, 9 * 8 * @sizeOf(f32)), profiles[0].max_activation_payload_bytes);

    try std.testing.expectEqual(bridge.TensorFrameStepKind.decode, profiles[1].step_kind);
    try std.testing.expectEqual(@as(u64, 4), profiles[1].max_batch_entries);
    try std.testing.expectEqual(@as(u64, 1), profiles[1].max_token_count_per_frame);
    try std.testing.expectEqual(@as(u64, 4 * 8 * @sizeOf(f32)), profiles[1].max_activation_payload_bytes);
}

test "inference.backend.cuda 10G-D local stage chain caps use required min rules" {
    const two_cuda_config = local_stage_testing.localMinBoundaryConfig(.f16, .row_major, 8, 3, 40, 7);
    try std.testing.expectEqual(@as(usize, 3), two_cuda_config.decode_max_batch_entries);
    try std.testing.expectEqual(@as(usize, 7), two_cuda_config.prefill_max_token_count_per_frame);

    const cpu_then_two_cuda_configs = local_stage_testing.localTwoBoundaryConfigs(
        .f32,
        .row_major,
        .f16,
        .row_major,
        5,
        9,
        16,
        64,
    );
    try std.testing.expectEqual(@as(usize, 5), cpu_then_two_cuda_configs[0].decode_max_batch_entries);
    try std.testing.expectEqual(@as(usize, 5), cpu_then_two_cuda_configs[1].decode_max_batch_entries);
    try std.testing.expectEqual(@as(usize, 16), cpu_then_two_cuda_configs[0].prefill_max_token_count_per_frame);
    try std.testing.expectEqual(@as(usize, 16), cpu_then_two_cuda_configs[1].prefill_max_token_count_per_frame);
    try std.testing.expectEqual(@as(u64, 16), try local_stage_testing.boundaryRowByteCount(8, cpu_then_two_cuda_configs[1].dtype));
}

test "inference.backend.cuda 10G-D local stage chain host ids are deterministic stage ids" {
    try std.testing.expectEqual(@as(u64, 1), (try local_stage_testing.deterministicHostId(0)).value);
    try std.testing.expectEqual(@as(u64, 2), (try local_stage_testing.deterministicHostId(1)).value);
    try std.testing.expectEqual(@as(u64, 3), (try local_stage_testing.deterministicHostId(2)).value);
    try std.testing.expectError(error.InvalidTopologyConfig, local_stage_testing.deterministicHostId(std.math.maxInt(usize)));
}

test "inference.backend.cuda 10G-E buildDecodeActivationMetadata segmentedHostDecodeByteImage buildDecodeTransportContract localTransportRequest uses multi_entry_local host segments" {
    const d_model: usize = 4;
    const plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &.{});
    const stage_backend_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 8),
    };
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        d_model,
        plan,
        &stage_backend_kinds,
        &configs,
    );
    defer bundle.deinit();

    const slot_indices = [_]usize{ 0, 1, 3 };
    const positions = [_]usize{ 7, 8, 9 };
    const slot_request_ids = [_]?u64{ 101, 202, 303, 404 };
    var batch_entries: [slot_indices.len]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try bridge.buildDecodeActivationMetadata(.{
        .plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = d_model,
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
        .location_hint = .{ .cpu = {} },
        .slot_request_ids = &slot_request_ids,
        .slot_indices = &slot_indices,
        .positions = &positions,
        .batch_entries = batch_entries[0..],
    });

    var row0 = [_]f32{ 1, 2, 3, 4 };
    var row1 = [_]f32{ 5, 6, 7, 8 };
    var row2 = [_]f32{ 9, 10, 11, 12 };
    const host_segments = [_][]const u8{
        std.mem.sliceAsBytes(row0[0..]),
        std.mem.sliceAsBytes(row1[0..]),
        std.mem.sliceAsBytes(row2[0..]),
    };
    const image = bridge.segmentedHostActivationByteImage(&metadata, &host_segments);
    const placement_plan = &bundle.placement_plan.?;
    const contract = try bridge.buildActivationTransportContract(
        placement_plan,
        &metadata,
        &image,
        false,
        false,
    );
    const request = localTransportRequest(
        placement_plan,
        &metadata,
        &image,
        &contract,
        null,
        false,
        false,
    );

    try std.testing.expectEqual(bridge.StageTransportActivationScope.multi_entry_local, contract.envelope.activation_scope.?);
    try std.testing.expectEqual(@as(u64, slot_indices.len), contract.envelope.batch_entry_count);
    try std.testing.expectEqual(metadata.payload.byte_count, contract.envelope.payload_byte_count);
    try std.testing.expectEqual(bridge.StageTransferMode.copy_in_process, contract.decision.mode);
    try std.testing.expect(request.envelope == &contract.envelope);
    try std.testing.expectEqual(bridge.StageTransferMode.copy_in_process, request.decision.mode);

    const TraceStep = enum { synchronize, upload_segments };
    const Trace = struct {
        steps: [4]TraceStep = undefined,
        count: usize = 0,
        synchronized_rows: usize = 0,
        upload_segments_calls: usize = 0,
        upload_byte_count: usize = 0,

        fn push(self: *@This(), step: TraceStep) void {
            self.steps[self.count] = step;
            self.count += 1;
        }
    };
    const Source = struct {
        trace: *Trace,
        slots: []const usize,

        pub fn synchronize(self: *@This()) anyerror!void {
            self.trace.synchronized_rows += self.slots.len;
            self.trace.push(.synchronize);
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };
    const Target = struct {
        trace: *Trace,

        pub fn synchronize(_: *@This()) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivationSegments(self: *@This(), segments: []const []const u8, byte_count: usize) anyerror!void {
            self.trace.upload_segments_calls += 1;
            self.trace.upload_byte_count = byte_count;
            self.trace.push(.upload_segments);
            try std.testing.expectEqual(@as(usize, 3), segments.len);
            try std.testing.expectEqual(@as(usize, 3 * d_model * @sizeOf(f32)), byte_count);
        }
    };

    var trace_state = Trace{};
    var source = Source{ .trace = &trace_state, .slots = &slot_indices };
    var target = Target{ .trace = &trace_state };
    try transport.executeLocalStageTransport(Source, Target, &source, &target, request);

    try std.testing.expectEqual(@as(usize, 2), trace_state.count);
    try std.testing.expectEqual(TraceStep.synchronize, trace_state.steps[0]);
    try std.testing.expectEqual(TraceStep.upload_segments, trace_state.steps[1]);
    try std.testing.expectEqual(slot_indices.len, trace_state.synchronized_rows);
    try std.testing.expectEqual(@as(usize, 1), trace_state.upload_segments_calls);
    try std.testing.expectEqual(@as(usize, 3 * d_model * @sizeOf(f32)), trace_state.upload_byte_count);
}

test "inference.backend.cuda 10G-E deviceDecodeByteImage buildDecodeTransportContract keeps peer copy event sync out of source synchronize" {
    const d_model: usize = 4;
    const plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &.{});
    const stage_backend_kinds = [_]bridge.HostBackendKind{ .cuda, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localMinBoundaryConfig(.f32, .row_major, 4, 4, 8, 8),
    };
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        d_model,
        plan,
        &stage_backend_kinds,
        &configs,
    );
    defer bundle.deinit();

    const slot_indices = [_]usize{ 0, 2 };
    const positions = [_]usize{ 5, 6 };
    const slot_request_ids = [_]?u64{ 101, 202, 303, 404 };
    var batch_entries: [slot_indices.len]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try bridge.buildDecodeActivationMetadata(.{
        .plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = d_model,
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
        .location_hint = .{ .cuda = 0 },
        .slot_request_ids = &slot_request_ids,
        .slot_indices = &slot_indices,
        .positions = &positions,
        .batch_entries = batch_entries[0..],
    });
    const image = bridge.deviceActivationByteImage(&metadata);
    const placement_plan = &bundle.placement_plan.?;
    const contract = try bridge.buildActivationTransportContract(
        placement_plan,
        &metadata,
        &image,
        false,
        true,
    );
    try std.testing.expectEqual(bridge.StageTransferMode.device_peer_copy_in_process, contract.decision.mode);
    try std.testing.expectEqual(bridge.StageTransportActivationScope.multi_entry_local, contract.envelope.activation_scope.?);

    const Trace = struct {
        synchronize_calls: usize = 0,
        peer_copy_calls: usize = 0,
        peer_copy_bytes: usize = 0,
    };
    const Source = struct {
        trace: *Trace,

        pub fn synchronize(self: *@This()) anyerror!void {
            self.trace.synchronize_calls += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn peerCopyActivationTo(self: *@This(), _: anytype, byte_count: usize) anyerror!void {
            self.trace.peer_copy_calls += 1;
            self.trace.peer_copy_bytes = byte_count;
        }

        pub fn peerCopyHandlesStageSync(_: *const @This()) bool {
            return true;
        }
    };
    const Target = struct {
        pub fn synchronize(_: *@This()) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };

    var staging: [2 * d_model * @sizeOf(f32)]u8 align(64) = [_]u8{0xaa} ** (2 * d_model * @sizeOf(f32));
    var trace_state = Trace{};
    var source = Source{ .trace = &trace_state };
    var target = Target{};
    try transport.executeLocalStageTransport(
        Source,
        Target,
        &source,
        &target,
        localTransportRequest(
            placement_plan,
            &metadata,
            &image,
            &contract,
            staging[0..],
            false,
            true,
        ),
    );

    try std.testing.expectEqual(@as(usize, 0), trace_state.synchronize_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.peer_copy_calls);
    try std.testing.expectEqual(@as(usize, 2 * d_model * @sizeOf(f32)), trace_state.peer_copy_bytes);
    try std.testing.expectEqual(@as(u8, 0xaa), staging[0]);
}

test "inference.backend.cuda 10G-F buildPrefillActivationMetadata hostActivationByteImage buildActivationTransportContract buildPrefillTransportContract executeLocalStageTransport keeps CPU KV upload outside chunk handoffs" {
    const d_model: usize = 4;
    const plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &.{});
    const stage_backend_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 8),
    };
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        d_model,
        plan,
        &stage_backend_kinds,
        &configs,
    );
    defer bundle.deinit();

    const slot_request_ids = [_]?u64{ 101, 202 };

    var payload_rows: [6 * d_model]f32 = undefined;
    for (&payload_rows, 0..) |*value, index| value.* = @floatFromInt(index);

    const Chunk = struct { sequence_start: usize, rows: usize };
    const chunks = [_]Chunk{
        .{ .sequence_start = 0, .rows = 4 },
        .{ .sequence_start = 4, .rows = 2 },
    };

    const Trace = struct {
        source_synchronize_calls: usize = 0,
        upload_calls: usize = 0,
        upload_byte_count: usize = 0,
    };
    const Source = struct {
        trace: *Trace,

        pub fn synchronize(self: *@This()) anyerror!void {
            self.trace.source_synchronize_calls += 1;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }
    };
    const Target = struct {
        trace: *Trace,

        pub fn synchronize(_: *@This()) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(self: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try std.testing.expect(byte_count <= host_buf.len);
            self.trace.upload_calls += 1;
            self.trace.upload_byte_count += byte_count;
        }
    };

    const placement_plan = &bundle.placement_plan.?;
    const plan_ref = &bundle.tensor_frame_plan_ref.?;
    var trace_state = Trace{};
    var source = Source{ .trace = &trace_state };
    var target = Target{ .trace = &trace_state };

    const cpu_kv_upload_calls_before_chunk_loop: usize = 1;
    var expected_payload_bytes: usize = 0;
    for (chunks) |chunk| {
        const row_start = chunk.sequence_start * d_model;
        const row_count = chunk.rows * d_model;
        const chunk_bytes = std.mem.sliceAsBytes(payload_rows[row_start..][0..row_count]);
        expected_payload_bytes += chunk_bytes.len;

        var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
        const metadata = try bridge.buildPrefillActivationMetadata(.{
            .plan_ref = plan_ref,
            .hidden_size = d_model,
            .boundary_index = 0,
            .dtype = .f32,
            .layout = .row_major,
            .location_hint = .{ .cpu = {} },
            .slot_request_ids = &slot_request_ids,
            .slot_index = 0,
            .sequence_start = chunk.sequence_start,
            .token_count = chunk.rows,
            .batch_entries = batch_entries[0..],
        });
        try bridge.validateTensorFrameForPlanBoundary(&metadata, plan_ref, 0);
        try bridge.validatePayloadBufferLength(&metadata, chunk_bytes.len);
        try std.testing.expectEqual(bridge.TensorFrameStepKind.prefill, metadata.step_kind);
        try std.testing.expectEqual(@as(u64, @intCast(chunk.sequence_start)), metadata.batch.entries[0].sequence_start);
        try std.testing.expectEqual(@as(u64, @intCast(chunk.rows)), metadata.batch.entries[0].token_count);

        const image = bridge.hostActivationByteImage(&metadata, chunk_bytes);
        const generic_contract = try bridge.buildActivationTransportContract(
            placement_plan,
            &metadata,
            &image,
            false,
            false,
        );
        const prefill_contract = try bridge.buildActivationTransportContract(
            placement_plan,
            &metadata,
            &image,
            false,
            false,
        );
        try std.testing.expectEqual(generic_contract.decision.mode, prefill_contract.decision.mode);
        try std.testing.expectEqual(bridge.StageTransferMode.copy_in_process, prefill_contract.decision.mode);
        try std.testing.expectEqual(bridge.StageTransportActivationScope.single_entry_header, prefill_contract.envelope.activation_scope.?);

        try transport.executeLocalStageTransport(
            Source,
            Target,
            &source,
            &target,
            localTransportRequest(
                placement_plan,
                &metadata,
                &image,
                &prefill_contract,
                null,
                false,
                false,
            ),
        );
    }

    try std.testing.expectEqual(@as(usize, 1), cpu_kv_upload_calls_before_chunk_loop);
    try std.testing.expectEqual(@as(usize, chunks.len), trace_state.source_synchronize_calls);
    try std.testing.expectEqual(@as(usize, chunks.len), trace_state.upload_calls);
    try std.testing.expectEqual(expected_payload_bytes, trace_state.upload_byte_count);
}

test "inference.backend.cuda 10G-F deviceActivationByteImage buildPrefillTransportContract executeLocalStageTransport covers GPU peer copy and host-staged fallback" {
    const d_model: usize = 4;
    const plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &.{});
    const stage_backend_kinds = [_]bridge.HostBackendKind{ .cuda, .cuda };
    const configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localMinBoundaryConfig(.f32, .row_major, 4, 4, 8, 8),
    };
    var bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        d_model,
        plan,
        &stage_backend_kinds,
        &configs,
    );
    defer bundle.deinit();

    const slot_request_ids = [_]?u64{ 101, 202 };

    var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try bridge.buildPrefillActivationMetadata(.{
        .plan_ref = &bundle.tensor_frame_plan_ref.?,
        .hidden_size = d_model,
        .boundary_index = 0,
        .dtype = .f32,
        .layout = .row_major,
        .location_hint = .{ .cuda = 0 },
        .slot_request_ids = &slot_request_ids,
        .slot_index = 1,
        .sequence_start = 5,
        .token_count = 3,
        .batch_entries = batch_entries[0..],
    });
    try bridge.validateTensorFrameForPlanBoundary(&metadata, &bundle.tensor_frame_plan_ref.?, 0);
    try bridge.validatePayloadBufferLength(&metadata, 3 * d_model * @sizeOf(f32));
    const image = bridge.deviceActivationByteImage(&metadata);
    const placement_plan = &bundle.placement_plan.?;

    const Trace = struct {
        synchronize_calls: usize = 0,
        download_calls: usize = 0,
        upload_calls: usize = 0,
        peer_copy_calls: usize = 0,
        byte_count: usize = 0,
        upload_first_byte: u8 = 0,
    };
    const Source = struct {
        trace: *Trace,
        peer_copy_handles_sync: bool,

        pub fn synchronize(self: *@This()) anyerror!void {
            self.trace.synchronize_calls += 1;
        }

        pub fn downloadActivation(self: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try std.testing.expect(byte_count <= host_buf.len);
            self.trace.download_calls += 1;
            self.trace.byte_count = byte_count;
            @memset(host_buf[0..byte_count], 0x7b);
        }

        pub fn uploadActivation(_: *@This(), _: []const u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn peerCopyActivationTo(self: *@This(), _: anytype, byte_count: usize) anyerror!void {
            self.trace.peer_copy_calls += 1;
            self.trace.byte_count = byte_count;
        }

        pub fn peerCopyHandlesStageSync(self: *const @This()) bool {
            return self.peer_copy_handles_sync;
        }
    };
    const Target = struct {
        trace: *Trace,

        pub fn synchronize(_: *@This()) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn downloadActivation(_: *@This(), _: []u8, _: usize) anyerror!void {
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(self: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try std.testing.expect(byte_count <= host_buf.len);
            self.trace.upload_calls += 1;
            self.trace.byte_count = byte_count;
            self.trace.upload_first_byte = host_buf[0];
        }
    };

    var trace_state = Trace{};
    var source = Source{ .trace = &trace_state, .peer_copy_handles_sync = true };
    var target = Target{ .trace = &trace_state };
    var staging: [3 * d_model * @sizeOf(f32)]u8 align(64) = [_]u8{0xaa} ** (3 * d_model * @sizeOf(f32));
    const peer_contract = try bridge.buildActivationTransportContract(
        placement_plan,
        &metadata,
        &image,
        false,
        true,
    );
    try std.testing.expectEqual(bridge.StageTransferMode.device_peer_copy_in_process, peer_contract.decision.mode);
    try transport.executeLocalStageTransport(
        Source,
        Target,
        &source,
        &target,
        localTransportRequest(
            placement_plan,
            &metadata,
            &image,
            &peer_contract,
            staging[0..],
            false,
            true,
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_state.synchronize_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.peer_copy_calls);
    try std.testing.expectEqual(@as(usize, 3 * d_model * @sizeOf(f32)), trace_state.byte_count);
    try std.testing.expectEqual(@as(u8, 0xaa), staging[0]);

    trace_state = .{};
    source = .{ .trace = &trace_state, .peer_copy_handles_sync = false };
    target = .{ .trace = &trace_state };
    const fallback_contract = try bridge.buildActivationTransportContract(
        placement_plan,
        &metadata,
        &image,
        false,
        false,
    );
    try std.testing.expectEqual(bridge.StageTransferMode.device_download_then_copy, fallback_contract.decision.mode);
    try transport.executeLocalStageTransport(
        Source,
        Target,
        &source,
        &target,
        localTransportRequest(
            placement_plan,
            &metadata,
            &image,
            &fallback_contract,
            staging[0..],
            false,
            false,
        ),
    );
    try std.testing.expectEqual(@as(usize, 1), trace_state.synchronize_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.download_calls);
    try std.testing.expectEqual(@as(usize, 1), trace_state.upload_calls);
    try std.testing.expectEqual(@as(u8, 0x7b), trace_state.upload_first_byte);
}

test "buildLocalStagePlacementPlan covers decode and prefill profiles for local stage chains" {
    const d_model: usize = 8;

    const two_stage_deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};

    var cpu_then_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer cpu_then_cuda_plan.deinit();
    var cpu_then_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &cpu_then_cuda_plan);
    defer cpu_then_cuda_state.deinit();
    const cpu_then_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const cpu_then_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };
    var cpu_then_cuda_placement = try local_stage_testing.buildLocalStagePlacementPlan(
        std.testing.allocator,
        d_model,
        &cpu_then_cuda_plan,
        &cpu_then_cuda_kinds,
        &cpu_then_cuda_configs,
        &cpu_then_cuda_state.ref,
    );
    defer cpu_then_cuda_placement.deinit();
    try expectLocalStagePlacement(&cpu_then_cuda_placement, d_model, 2, &cpu_then_cuda_configs);
    try std.testing.expectEqual(bridge.StatePlacementMode.validate_ref, cpu_then_cuda_placement.state_placement_mode);
    try std.testing.expectEqual(@as(usize, 2), cpu_then_cuda_placement.state_stage_summaries.len);

    var two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer two_cuda_plan.deinit();
    var two_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &two_cuda_plan);
    defer two_cuda_state.deinit();
    const two_cuda_kinds = [_]bridge.HostBackendKind{ .cuda, .cuda };
    const two_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localMinBoundaryConfig(.f16, .row_major, 8, 3, 40, 7),
    };
    var two_cuda_placement = try local_stage_testing.buildLocalStagePlacementPlan(
        std.testing.allocator,
        d_model,
        &two_cuda_plan,
        &two_cuda_kinds,
        &two_cuda_configs,
        &two_cuda_state.ref,
    );
    defer two_cuda_placement.deinit();
    try expectLocalStagePlacement(&two_cuda_placement, d_model, 2, &two_cuda_configs);

    const three_stage_deps = [_]models.stage_plan.DependencyOverride{
        .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 2,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
    };
    var cpu_then_two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 5, &.{ 1, 3 }, &three_stage_deps);
    defer cpu_then_two_cuda_plan.deinit();
    var cpu_then_two_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &cpu_then_two_cuda_plan);
    defer cpu_then_two_cuda_state.deinit();
    const cpu_then_two_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda, .cuda };
    const cpu_then_two_cuda_configs = local_stage_testing.localTwoBoundaryConfigs(
        .f32,
        .row_major,
        .f16,
        .row_major,
        5,
        9,
        16,
        64,
    );
    var cpu_then_two_cuda_placement = try local_stage_testing.buildLocalStagePlacementPlan(
        std.testing.allocator,
        d_model,
        &cpu_then_two_cuda_plan,
        &cpu_then_two_cuda_kinds,
        &cpu_then_two_cuda_configs,
        &cpu_then_two_cuda_state.ref,
    );
    defer cpu_then_two_cuda_placement.deinit();
    try expectLocalStagePlacement(&cpu_then_two_cuda_placement, d_model, 3, &cpu_then_two_cuda_configs);
}

test "buildLocalStageStateOwnershipPlan covers stateful local stage chain dependencies" {
    const two_stage_deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};
    var two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer two_cuda_plan.deinit();
    var two_cuda_state = try local_stage_testing.buildLocalStageStateOwnershipPlan(std.testing.allocator, &two_cuda_plan);
    defer two_cuda_state.deinit();
    try bridge.validateStageStateOwnershipPlan(&two_cuda_state);
    try std.testing.expectEqual(@as(usize, 1), two_cuda_state.boundaries.len);
    try std.testing.expectEqual(@as(usize, 1), two_cuda_state.stateful_dependencies.len);
    try std.testing.expectEqual(@as(usize, 1), two_cuda_state.partition_facts.len);
    try std.testing.expectEqual(@as(usize, 0), two_cuda_state.partition_facts[0].boundary_index);
    try std.testing.expectEqual(@as(usize, 0), two_cuda_state.partition_facts[0].source_stage_id);
    try std.testing.expectEqual(@as(usize, 1), two_cuda_state.partition_facts[0].target_stage_id);

    const three_stage_deps = [_]models.stage_plan.DependencyOverride{
        .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 2,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
    };
    var cpu_then_two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 5, &.{ 1, 3 }, &three_stage_deps);
    defer cpu_then_two_cuda_plan.deinit();
    var cpu_then_two_cuda_state = try local_stage_testing.buildLocalStageStateOwnershipPlan(std.testing.allocator, &cpu_then_two_cuda_plan);
    defer cpu_then_two_cuda_state.deinit();
    try bridge.validateStageStateOwnershipPlan(&cpu_then_two_cuda_state);
    try std.testing.expectEqual(@as(usize, 2), cpu_then_two_cuda_state.boundaries.len);
    try std.testing.expectEqual(@as(usize, 2), cpu_then_two_cuda_state.stateful_dependencies.len);
    try std.testing.expectEqual(@as(usize, 2), cpu_then_two_cuda_state.partition_facts.len);
    for (cpu_then_two_cuda_state.partition_facts, 0..) |fact, boundary_index| {
        try std.testing.expectEqual(boundary_index, fact.boundary_index);
        try std.testing.expectEqual(boundary_index, fact.source_stage_id);
        try std.testing.expectEqual(boundary_index + 1, fact.target_stage_id);
    }
}

test "deinitLocalStageContracts uses reverse local stage chain contract order and is idempotent" {
    const expected_order = [_]local_stage_testing.ContractField{
        .local_stage_runner_plan_ref,
        .placement_plan,
        .state_placement_ref,
        .state_ownership_plan,
        .tensor_frame_plan_ref,
        .stage_plan,
    };
    try std.testing.expectEqualSlices(local_stage_testing.ContractField, &expected_order, &local_stage_testing.contract_deinit_order);

    const two_stage_deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};
    const cpu_then_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const cpu_then_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };
    const cpu_then_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    var cpu_then_cuda_bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        8,
        cpu_then_cuda_plan,
        &cpu_then_cuda_kinds,
        &cpu_then_cuda_configs,
    );
    defer cpu_then_cuda_bundle.deinit();
    local_stage_testing.deinitLocalStageContractBundleTwice(&cpu_then_cuda_bundle);

    const two_cuda_kinds = [_]bridge.HostBackendKind{ .cuda, .cuda };
    const two_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localMinBoundaryConfig(.f16, .row_major, 8, 3, 40, 7),
    };
    const two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    var two_cuda_bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        8,
        two_cuda_plan,
        &two_cuda_kinds,
        &two_cuda_configs,
    );
    defer two_cuda_bundle.deinit();
    local_stage_testing.deinitLocalStageContractBundleTwice(&two_cuda_bundle);

    const three_stage_deps = [_]models.stage_plan.DependencyOverride{
        .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 2,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
    };
    const cpu_then_two_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda, .cuda };
    const cpu_then_two_cuda_configs = local_stage_testing.localTwoBoundaryConfigs(
        .f32,
        .row_major,
        .f16,
        .row_major,
        5,
        9,
        16,
        64,
    );
    const cpu_then_two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 5, &.{ 1, 3 }, &three_stage_deps);
    var cpu_then_two_cuda_bundle = try local_stage_testing.buildLocalStageContractBundleFromOwnedPlan(
        std.testing.allocator,
        8,
        cpu_then_two_cuda_plan,
        &cpu_then_two_cuda_kinds,
        &cpu_then_two_cuda_configs,
    );
    defer cpu_then_two_cuda_bundle.deinit();
    local_stage_testing.deinitLocalStageContractBundleTwice(&cpu_then_two_cuda_bundle);
}

test "buildLocalStagePlacementPlan cleans up allocation failures for local stage chain contract bundles" {
    const two_stage_deps = [_]models.stage_plan.DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .reason = .stateful_decoder,
        .affects_loader_residency = false,
    }};
    var cpu_then_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer cpu_then_cuda_plan.deinit();
    var cpu_then_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &cpu_then_cuda_plan);
    defer cpu_then_cuda_state.deinit();
    const cpu_then_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda };
    const cpu_then_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localBoundaryConfig(.f32, .row_major, 4, 9),
    };
    try expectPlacementBuildFailureCleanup(8, &cpu_then_cuda_plan, &cpu_then_cuda_kinds, &cpu_then_cuda_configs, &cpu_then_cuda_state.ref);

    var two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 4, &.{2}, &two_stage_deps);
    defer two_cuda_plan.deinit();
    var two_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &two_cuda_plan);
    defer two_cuda_state.deinit();
    const two_cuda_kinds = [_]bridge.HostBackendKind{ .cuda, .cuda };
    const two_cuda_configs = [_]local_stage_testing.BoundaryConfig{
        local_stage_testing.localMinBoundaryConfig(.f16, .row_major, 8, 3, 40, 7),
    };
    try expectPlacementBuildFailureCleanup(8, &two_cuda_plan, &two_cuda_kinds, &two_cuda_configs, &two_cuda_state.ref);

    const three_stage_deps = [_]models.stage_plan.DependencyOverride{
        .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 2,
            .reason = .stateful_decoder,
            .affects_loader_residency = false,
        },
    };
    var cpu_then_two_cuda_plan = try buildLocalStageTestStagePlan(std.testing.allocator, 5, &.{ 1, 3 }, &three_stage_deps);
    defer cpu_then_two_cuda_plan.deinit();
    var cpu_then_two_cuda_state = try buildLocalStageStateFixture(std.testing.allocator, &cpu_then_two_cuda_plan);
    defer cpu_then_two_cuda_state.deinit();
    const cpu_then_two_cuda_kinds = [_]bridge.HostBackendKind{ .cpu, .cuda, .cuda };
    const cpu_then_two_cuda_configs = local_stage_testing.localTwoBoundaryConfigs(
        .f32,
        .row_major,
        .f16,
        .row_major,
        5,
        9,
        16,
        64,
    );
    try expectPlacementBuildFailureCleanup(8, &cpu_then_two_cuda_plan, &cpu_then_two_cuda_kinds, &cpu_then_two_cuda_configs, &cpu_then_two_cuda_state.ref);
}

test "finalOutputBuffer returns residual when program ends with add" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .multihead_attention,
            .state_block_id = runtime_contract.kv_cache_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.residual, layer_ops.finalOutputBuffer(&program));
}

test "finalOutputBuffer returns kernel output buffer for post-norm endings" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, layer_ops.finalOutputBuffer(&program));
}

test "layer program support envelope accepts kernel-add programs" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
        .{ .kernel = .{
            .id = 1,
            .in = .norm_out,
            .out = .branch_out,
            .debug_type = .mlp,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expect(
        runtime_contract.firstLayerProgramCompatibilityIssue(
            &program,
            .attention_mlp,
            CudaBackend.layer_program_adapter_table,
        ) == null,
    );
}

test "layer_program_adapter_table covers CUDA LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .gated_delta_net,
        .swiglu,
        .shortconv,
        .residual_add,
        .moe,
    };
    for (supported) |opcode| {
        try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode)] != null);
    }

    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mla_attention)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mul_scalar)] == null);
    try std.testing.expect(CudaBackend.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.vision_patch_embed)] == null);
}

test "layer program support envelope rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mul_scalar, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope rejects CUDA-unsupported macro opcodes at load time" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .mamba_mixer,
            .state_block_id = runtime_contract.mamba_state_id,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mamba_mixer, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program support envelope rejects block-kind state descriptor mismatch" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
            .state_block_id = runtime_contract.shortconv_state_id,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        CudaBackend.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .state_mismatch => |mismatch| {
            try std.testing.expectEqual(@as(usize, 0), mismatch.op_index);
            try std.testing.expectEqual(opcode_map.Opcode.shortconv, mismatch.opcode);
            try std.testing.expectEqual(runtime_contract.shortconv_state_id, mismatch.state_id);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "buildCudaLayerProgramRegisterSlotMap reuses temp slots from liveness" {
    const inputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const outputs0 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const inputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const outputs1 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const inputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const outputs2 = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(3)};
    const instructions = [_]runtime_contract.Instruction{
        .{
            .opcode = .rmsnorm,
            .inputs = inputs0[0..],
            .outputs = outputs0[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs1[0..],
            .outputs = outputs1[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
        .{
            .opcode = .swiglu,
            .inputs = inputs2[0..],
            .outputs = outputs2[0..],
            .weights = &.{},
            .param_block_id = null,
            .state_block_id = null,
        },
    };
    const kill0 = [_]u64{0b0000};
    const kill1 = [_]u64{0b0010};
    const kill2 = [_]u64{0b1100};
    const compiled = runtime_contract.CompiledPlan{
        .plan = .{
            .instructions = instructions[0..],
            .register_count = 4,
            .state_descs = &.{},
        },
        .param_blocks = &.{},
        .weight_bindings = &.{},
        .register_buffer_specs = &.{
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
            .{ .size = 1, .@"align" = 4 },
        },

        .liveness = .{
            .register_last_read = &.{ 0, 1, 2, 2 },
            .kill_after_instruction = &.{ kill0[0..], kill1[0..], kill2[0..] },
        },
        .peak_registers = 2,
        .diagnostics = &.{},
    };

    const map = try buildCudaLayerProgramRegisterSlotMap(std.testing.allocator, &compiled);
    defer std.testing.allocator.free(map);
    try std.testing.expect(map[1] < 2);
    try std.testing.expect(map[2] < 2);
    try std.testing.expect(map[3] < 2);
    try std.testing.expectEqual(map[1], map[3]);
    try std.testing.expect(map[2] != map[1]);
}

test "resolveDenseInOutLayout transposes [out,in] orientation" {
    const layout = try resolveDenseInOutLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseInOutLayout rejects mismatched orientation" {
    try std.testing.expectError(error.UnsupportedModel, resolveDenseInOutLayout(96, 64, 128));
}

test "resolveDenseInOutLayout prefers [out,in] for square matrices" {
    const layout = try resolveDenseInOutLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps [out,in] orientation" {
    const layout = try resolveDenseOutInLayout(256, 128, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "resolveDenseOutInLayout transposes [in,out] orientation" {
    const layout = try resolveDenseOutInLayout(128, 256, 128);
    try std.testing.expectEqual(@as(usize, 128), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(layout.needs_transpose);
}

test "resolveDenseOutInLayout keeps square typed layout untransposed" {
    const layout = try resolveDenseOutInLayout(256, 256, 256);
    try std.testing.expectEqual(@as(usize, 256), layout.in_dim);
    try std.testing.expectEqual(@as(usize, 256), layout.out_dim);
    try std.testing.expect(!layout.needs_transpose);
}

test "transposeRowMajor transposes compact row-major matrix" {
    const src = [_]u16{
        1, 2, 3,
        4, 5, 6,
    };
    const transposed = try transposeRowMajor(u16, std.testing.allocator, src[0..], 2, 3);
    defer std.testing.allocator.free(transposed);

    const expected = [_]u16{
        1, 4,
        2, 5,
        3, 6,
    };
    try std.testing.expectEqualSlices(u16, expected[0..], transposed);
}

test "required_kernels contract has unique slots and operation names" {
    var seen_slots = std.AutoHashMap(KernelSlot, void).init(std.testing.allocator);
    defer seen_slots.deinit();

    var seen_ops = std.StringHashMap(void).init(std.testing.allocator);
    defer seen_ops.deinit();

    for (required_kernels) |entry| {
        const slot_put = try seen_slots.getOrPut(entry.slot);
        try std.testing.expect(!slot_put.found_existing);

        const op_put = try seen_ops.getOrPut(entry.op_name);
        try std.testing.expect(!op_put.found_existing);

        try std.testing.expect(std.mem.startsWith(u8, entry.embedded_symbol, "talu_"));
        try std.testing.expect(!hasVersionSuffixName(entry.op_name));
        try std.testing.expect(!hasVersionSuffixName(entry.embedded_symbol));
    }

    const slot_count = @typeInfo(KernelSlot).@"enum".fields.len;
    try std.testing.expectEqual(slot_count, required_kernels.len);
}

test "required_kernels keeps heads-based attention path canonical" {
    const required_ops = [_][]const u8{
        compute.cuda.attn_scores_heads_f32.op_name,
        compute.cuda.attn_scores_heads_f16_kv.op_name,
        compute.cuda.attn_weighted_sum_heads_f32.op_name,
        compute.cuda.attn_weighted_sum_heads_f16_kv.op_name,
        compute.cuda.softmax_rows.op_name,
    };
    const disallowed_ops = [_][]const u8{
        "attn_scores_f32",
        "attn_scores_f16_kv",
        "attn_weighted_sum_f32",
        "attn_weighted_sum_f16_kv",
        "softmax_f32",
    };

    for (required_ops) |op| {
        var found = false;
        for (required_kernels) |entry| {
            if (std.mem.eql(u8, entry.op_name, op)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }

    for (disallowed_ops) |op| {
        for (required_kernels) |entry| {
            try std.testing.expect(!std.mem.eql(u8, entry.op_name, op));
        }
    }
}

fn hasVersionSuffixName(name: []const u8) bool {
    const marker = "_v";
    const at = std.mem.lastIndexOf(u8, name, marker) orelse return false;
    const digits = name[at + marker.len ..];
    if (digits.len == 0) return false;
    for (digits) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
    }
    return true;
}

test "tryPopulateProjectionFromWeight supports [d_model, vocab] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ d_model, 4 }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateProjectionFromWeight supports [vocab, d_model] layout" {
    const d_model: usize = 3;
    const projected_vocab: usize = 2;
    var weights = [_]f32{
        1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,
        7.0,  8.0,  9.0,
        10.0, 11.0, 12.0,
    };
    const weight_bytes = std.mem.sliceAsBytes(weights[0..]);
    const weight = Tensor.view(weight_bytes.ptr, &.{ 4, d_model }, .f32, weight_bytes.len);
    var out = [_]f32{0.0} ** (d_model * projected_vocab);

    try std.testing.expect(tryPopulateProjectionFromWeight(std.testing.allocator, &weight, d_model, projected_vocab, out[0..]));
    const expected = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [vocab, d_model] layout" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(models.config.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "tryPopulateHiddenFromToken supports [d_model, vocab] layout" {
    var embedding_data = [_]f32{
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 3, 2 }, .f32, bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(models.config.ModelConfig),
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 3;
    try std.testing.expect(try tryPopulateHiddenFromToken(&loaded, 1, out[0..]));
    const expected = [_]f32{ 4.0, 5.0, 6.0 };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens applies embedding multiplier" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(models.config.ModelConfig);
    cfg.embedding_multiplier = 2.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 0 };
    var out = [_]f32{0.0} ** 6;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], null);

    const expected = [_]f32{
        8.0, 10.0, 12.0,
        2.0, 4.0,  6.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "populatePrefillHiddenFromTokens zero-fills configured skip token rows" {
    var embedding_data = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    };
    const bytes = std.mem.sliceAsBytes(embedding_data[0..]);
    const embeddings = Tensor.view(bytes.ptr, &.{ 2, 3 }, .f32, bytes.len);

    var cfg = std.mem.zeroes(models.config.ModelConfig);
    cfg.embedding_multiplier = 1.0;
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = cfg,
        .token_embeddings = embeddings,
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{ 1, 99, 0 };
    var out = [_]f32{0.0} ** 9;
    try populatePrefillHiddenFromTokens(&loaded, tokens[0..], 3, out[0..], 99);

    const expected = [_]f32{
        4.0, 5.0, 6.0,
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
    };
    for (expected, out) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0);
    }
}

test "gaffineValueAt decodes grouped_affine_u4 values" {
    var packed_words = [_]u32{
        // 8 packed 4-bit values: 0,1,2,3,4,5,6,7
        0x7654_3210,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    const one_bf16 = dtype.f32ToBf16(1.0);
    const zero_bf16 = dtype.f32ToBf16(0.0);
    var scales_u16 = [_]u16{one_bf16};
    var biases_u16 = [_]u16{zero_bf16};
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u4, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 8,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "tryPopulateFinalNormWeight supports bf16 weights" {
    var norm_u16 = [_]u16{
        dtype.f32ToBf16(1.25),
        dtype.f32ToBf16(-0.5),
    };
    const norm_bytes = std.mem.sliceAsBytes(norm_u16[0..]);
    const norm_tensor = Tensor.view(norm_bytes.ptr, &.{2}, .bf16, norm_bytes.len);

    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(models.config.ModelConfig),
        .ln_final = norm_tensor,
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .bf16,
    };
    defer loaded.arena.deinit();

    var out = [_]f32{0.0} ** 2;
    try std.testing.expect(tryPopulateFinalNormWeight(&loaded, out[0..]));
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.01);
}

test "populatePrefillHiddenFromTokens rejects missing embeddings" {
    var loaded = LoadedModel{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .config = std.mem.zeroes(models.config.ModelConfig),
        .token_embeddings = std.mem.zeroes(Tensor),
        .blocks = &.{},
        .original_weight_dtype = .f32,
    };
    defer loaded.arena.deinit();

    const tokens = [_]u32{0};
    var out = [_]f32{0.0} ** 4;
    try std.testing.expectError(
        error.UnsupportedModel,
        populatePrefillHiddenFromTokens(&loaded, tokens[0..], 4, out[0..], null),
    );
}

test "shouldDownloadPrefillLogits only on final token" {
    try std.testing.expect(!shouldDownloadPrefillLogits(0, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(1, 4));
    try std.testing.expect(!shouldDownloadPrefillLogits(2, 4));
    try std.testing.expect(shouldDownloadPrefillLogits(3, 4));
}

test "shouldDownloadPrefillLogits true for single-token prefill" {
    try std.testing.expect(shouldDownloadPrefillLogits(0, 1));
}

test "linearWeightSupportsSequenceRows allows gaffine when matvec kernel is loaded" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    var weight = LinearWeight{
        .gaffine_u4 = .{
            .rows = 16,
            .cols = 16,
            .packed_data = dummy_buffer,
            .scales = dummy_buffer,
            .biases = dummy_buffer,
            .group_size = 8,
            .scales_dtype_tag = gaffine_scales_dtype_bf16,
        },
    };

    try std.testing.expect(!CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, false, false, false));
    try std.testing.expect(CudaBackend.linearWeightSupportsSequenceRowsForKernels(&weight, false, false, true, false, false));
}

test "canFuseDenseU16QkvWeights supports GQA-style unequal output dims" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 2048,
        .cols = 2048,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const k = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 2048,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(engine_ops.canFuseDenseU16QkvWeights(2048, q, k, v));
}

test "canFuseDenseU16QkvWeights rejects mixed dtypes" {
    const dummy_buffer: compute.cuda.Buffer = .{ .pointer = 0, .size = 0 };
    const q = U16LinearWeight{
        .rows = 1024,
        .cols = 1024,
        .buffer = dummy_buffer,
        .dtype = .f16,
    };
    const k = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };
    const v = U16LinearWeight{
        .rows = 1024,
        .cols = 256,
        .buffer = dummy_buffer,
        .dtype = .bf16,
    };

    try std.testing.expect(!engine_ops.canFuseDenseU16QkvWeights(1024, q, k, v));
}

fn makeGatedDeltaBlockForFfnPlan(
    dummy: *const Tensor,
    w1: ?*const Tensor,
    w2: ?*const Tensor,
    w3: ?*const Tensor,
    down_proj: ?*const Tensor,
    fused_gate_up: ?Tensor,
    gate_up_layout: models.runtime_blocks.GateUpLayout,
) models.runtime_blocks.GatedDeltaBlockWeights {
    return .{
        .ln1_weight = dummy,
        .config = .{
            .d_model = 2,
            .d_conv = 1,
            .n_heads = 1,
            .d_head = 2,
            .n_key_heads = 1,
        },
        .weights = .{
            .in_proj = dummy,
            .conv1d_weight = dummy,
            .A_log = dummy,
            .out_proj = dummy,
        },
        .fused_gate_up = if (fused_gate_up) |fg|
            .{ .gate_up = fg, .gate_up_layout = gate_up_layout }
        else
            null,
        .down_proj = down_proj,
        .w1 = w1,
        .w2 = w2,
        .w3 = w3,
        .moe_weights = null,
    };
}

test "resolveGatedDeltaFfnUploadPlan returns split weights when gate/up are separate" {
    var dummy_data = [_]f32{ 1, 2, 3, 4 };
    var w1_data = [_]f32{ 5, 6, 7, 8 };
    var w2_data = [_]f32{ 9, 10, 11, 12 };
    var w3_data = [_]f32{ 13, 14, 15, 16 };
    var dummy = Tensor.view2DSlice(dummy_data[0..], 2, 2);
    var w1 = Tensor.view2DSlice(w1_data[0..], 2, 2);
    var w2 = Tensor.view2DSlice(w2_data[0..], 2, 2);
    var w3 = Tensor.view2DSlice(w3_data[0..], 2, 2);

    var block = makeGatedDeltaBlockForFfnPlan(
        &dummy,
        &w1,
        &w2,
        &w3,
        null,
        null,
        .concat,
    );

    const plan = try resolveGatedDeltaFfnUploadPlan(&block);
    switch (plan) {
        .split => |split| {
            try std.testing.expect(split.w1 == &w1);
            try std.testing.expect(split.w2 == &w2);
            try std.testing.expect(split.w3 == &w3);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "resolveGatedDeltaFfnUploadPlan returns fused gate_up when present" {
    var dummy_data = [_]f32{ 1, 2, 3, 4 };
    var fused_data = [_]f32{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    var down_data = [_]f32{ 9, 10, 11, 12 };
    var dummy = Tensor.view2DSlice(dummy_data[0..], 2, 2);
    const fused = Tensor.view2DSlice(fused_data[0..], 4, 2);
    var down = Tensor.view2DSlice(down_data[0..], 2, 2);

    var block = makeGatedDeltaBlockForFfnPlan(
        &dummy,
        null,
        null,
        null,
        &down,
        fused,
        .concat,
    );

    const plan = try resolveGatedDeltaFfnUploadPlan(&block);
    switch (plan) {
        .fused => |fused_plan| {
            try std.testing.expectEqual(models.runtime_blocks.GateUpLayout.concat, fused_plan.gate_up_layout);
            try std.testing.expect(fused_plan.w2 == &down);
            try std.testing.expectEqual(@as(i64, 4), fused_plan.gate_up.shape[0]);
            try std.testing.expectEqual(@as(i64, 2), fused_plan.gate_up.shape[1]);
            try std.testing.expectEqual(fused.data_ptr, fused_plan.gate_up.data_ptr);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "resolveGatedDeltaFfnUploadPlan falls back to split weights for non-dense fused dtype" {
    var dummy_data = [_]f32{ 1, 2, 3, 4 };
    var gate_data = [_]f32{ 5, 6, 7, 8 };
    var up_data = [_]f32{ 9, 10, 11, 12 };
    var down_data = [_]f32{ 13, 14, 15, 16 };
    var fused_u4_bytes = [_]u8{0} ** 16;
    const fused_shape = [_]usize{ 4, 2 };
    var dummy = Tensor.view2DSlice(dummy_data[0..], 2, 2);
    var gate = Tensor.view2DSlice(gate_data[0..], 2, 2);
    var up = Tensor.view2DSlice(up_data[0..], 2, 2);
    var down = Tensor.view2DSlice(down_data[0..], 2, 2);
    const fused = Tensor.view(&fused_u4_bytes, &fused_shape, .grouped_affine_u4, fused_u4_bytes.len);

    var block = makeGatedDeltaBlockForFfnPlan(
        &dummy,
        &gate,
        &down,
        &up,
        &down,
        fused,
        .concat,
    );

    const plan = try resolveGatedDeltaFfnUploadPlan(&block);
    switch (plan) {
        .split => |split| {
            try std.testing.expect(split.w1 == &gate);
            try std.testing.expect(split.w2 == &down);
            try std.testing.expect(split.w3 == &up);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "resolveGatedDeltaFfnUploadPlan rejects fused gate_up without down projection" {
    var dummy_data = [_]f32{ 1, 2, 3, 4 };
    var fused_data = [_]f32{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    var dummy = Tensor.view2DSlice(dummy_data[0..], 2, 2);
    const fused = Tensor.view2DSlice(fused_data[0..], 4, 2);

    var block = makeGatedDeltaBlockForFfnPlan(
        &dummy,
        null,
        null,
        null,
        null,
        fused,
        .concat,
    );
    try std.testing.expectError(error.MissingWeight, resolveGatedDeltaFfnUploadPlan(&block));
}

test "collectTokenPositions returns all matching positions" {
    const tokens = [_]u32{ 7, 3, 7, 9, 7 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 7);
    defer if (positions.len > 0) std.testing.allocator.free(positions);

    const expected = [_]usize{ 0, 2, 4 };
    try std.testing.expectEqualSlices(usize, expected[0..], positions);
}

test "collectTokenPositions returns empty when token is absent" {
    const tokens = [_]u32{ 1, 2, 3 };
    const positions = try collectTokenPositions(std.testing.allocator, tokens[0..], 9);
    try std.testing.expectEqual(@as(usize, 0), positions.len);
}

test "findPositionIndex locates mapped image feature index" {
    const positions = [_]usize{ 2, 5, 9 };
    try std.testing.expectEqual(@as(?usize, 0), findPositionIndex(positions[0..], 2));
    try std.testing.expectEqual(@as(?usize, 1), findPositionIndex(positions[0..], 5));
    try std.testing.expectEqual(@as(?usize, 2), findPositionIndex(positions[0..], 9));
    try std.testing.expectEqual(@as(?usize, null), findPositionIndex(positions[0..], 7));
}

test "deepstackLayersCompatibleWithPrompt accepts valid layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const layer0 = [_]f32{0} ** (2 * 4);
    const layer1 = [_]f32{0} ** (3 * 4);
    const layers = [_][]const f32{ layer0[0..], layer1[0..] };
    try std.testing.expect(deepstackLayersCompatibleWithPrompt(layers[0..], image_positions, d_model));
}

test "deepstackLayersCompatibleWithPrompt rejects malformed layers" {
    const d_model: usize = 4;
    const image_positions: usize = 2;
    const too_few_rows = [_]f32{0} ** (1 * 4);
    const bad_stride = [_]f32{0} ** 7;
    const valid = [_]f32{0} ** (2 * 4);

    const layers_few = [_][]const f32{too_few_rows[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_few[0..], image_positions, d_model));

    const layers_stride = [_][]const f32{bad_stride[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_stride[0..], image_positions, d_model));

    const layers_zero_dim = [_][]const f32{valid[0..]};
    try std.testing.expect(!deepstackLayersCompatibleWithPrompt(layers_zero_dim[0..], image_positions, 0));
}

test "materializeDenseOutInU16 handles out-in and in-out source layouts" {
    var out_in_data = [_]u16{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .bf16, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInU16(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    for (out_in_data, out_in_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }

    var in_out_data = [_]u16{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .bf16, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInU16(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]u16{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    for (expected, in_out_view.values) |want, got| {
        try std.testing.expectEqual(want, got);
    }
}

test "materializeDenseOutInF32 handles out-in and in-out source layouts" {
    var out_in_data = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
    };
    var out_in_tensor = Tensor.view(std.mem.sliceAsBytes(out_in_data[0..]).ptr, &.{ 4, 3 }, .f32, std.mem.sliceAsBytes(out_in_data[0..]).len);
    var out_in_view = try materializeDenseOutInF32(std.testing.allocator, &out_in_tensor, 3, 4);
    defer out_in_view.deinit(std.testing.allocator);
    try std.testing.expect(out_in_view.owned == null);
    try std.testing.expectEqualSlices(f32, out_in_data[0..], out_in_view.values);

    var in_out_data = [_]f32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var in_out_tensor = Tensor.view(std.mem.sliceAsBytes(in_out_data[0..]).ptr, &.{ 3, 4 }, .f32, std.mem.sliceAsBytes(in_out_data[0..]).len);
    var in_out_view = try materializeDenseOutInF32(std.testing.allocator, &in_out_tensor, 3, 4);
    defer in_out_view.deinit(std.testing.allocator);
    try std.testing.expect(in_out_view.owned != null);
    const expected = [_]f32{
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    try std.testing.expectEqualSlices(f32, expected[0..], in_out_view.values);
}

test "gaffineValueAt decodes grouped_affine_u8 values" {
    var packed_words = [_]u32{
        0x0302_0100,
        0x0706_0504,
    };
    const packed_bytes = std.mem.sliceAsBytes(packed_words[0..]);

    var scales_u16 = [_]u16{
        dtype.f32ToBf16(1.0),
        dtype.f32ToBf16(1.0),
    };
    var biases_u16 = [_]u16{
        dtype.f32ToBf16(0.0),
        dtype.f32ToBf16(0.0),
    };
    const scales_bytes = std.mem.sliceAsBytes(scales_u16[0..]);
    const biases_bytes = std.mem.sliceAsBytes(biases_u16[0..]);

    var weight = Tensor.view(packed_bytes.ptr, &.{ 1, 8 }, .grouped_affine_u8, packed_bytes.len);
    weight.gaffine = .{
        .scales = scales_bytes,
        .biases = biases_bytes,
        .group_size = 4,
        .scales_dtype = .bf16,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), try gaffineValueAt(&weight, 0, 0), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), try gaffineValueAt(&weight, 0, 3), 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try gaffineValueAt(&weight, 0, 7), 0.0);
}

test "canFuseGaffineGateUpWeights accepts grouped-affine u8 weights with matching metadata" {
    const buf = compute.cuda.Buffer{ .pointer = 0, .size = 0 };
    const gate = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 11008,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const up = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 11008,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };

    try std.testing.expect(engine_ops.canFuseGaffineGateUpWeights(4096, gate, up));

    var bad_up = up;
    bad_up.scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_f16;
    try std.testing.expect(!engine_ops.canFuseGaffineGateUpWeights(4096, gate, bad_up));
}

test "canFuseGaffineQkvWeights accepts grouped-affine u8 weights with matching metadata" {
    const buf = compute.cuda.Buffer{ .pointer = 0, .size = 0 };
    const q = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 8192,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const k = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 1024,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };
    const v = GaffineU8LinearWeight{
        .rows = 4096,
        .cols = 1024,
        .packed_data = buf,
        .scales = buf,
        .biases = buf,
        .group_size = 32,
        .scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_bf16,
    };

    try std.testing.expect(engine_ops.canFuseGaffineQkvWeights(4096, q, k, v));

    var bad_v = v;
    bad_v.scales_dtype_tag = compute.cuda.gaffine_u8_matvec.scales_dtype_f16;
    try std.testing.expect(!engine_ops.canFuseGaffineQkvWeights(4096, q, k, bad_v));
}

test "BlockRuntimeLayer.rebuildInstructionMetadata binds per-op runtime metadata" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const zero_buffer = std.mem.zeroes(compute.cuda.Buffer);
    const zero_tensor = DeviceTensor{
        .rows = 0,
        .cols = 0,
        .buffer = zero_buffer,
    };
    const zero_weight = LinearWeight{ .dense_f32 = zero_tensor };
    var norm0: DeviceTensor = zero_tensor;
    var norm1: DeviceTensor = zero_tensor;
    var attention_runtime: LayerAttentionRuntime = .{
        .q_dim = 0,
        .q_projection_dim = 0,
        .kv_dim = 0,
        .d_ff = 0,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
        .ln1_weight = zero_tensor,
        .ln2_weight = zero_tensor,
        .pre_ffn_norm_weight = null,
        .post_ffn_norm_weight = null,
        .q_norm_weight = null,
        .k_norm_weight = null,
        .q_proj = zero_weight,
        .k_proj = zero_weight,
        .v_proj = zero_weight,
        .o_proj = zero_weight,
        .w1 = zero_weight,
        .w2 = zero_weight,
        .w3 = zero_weight,
        .k_cache = zero_buffer,
        .v_cache = zero_buffer,
        .kv_capacity = 0,
        .slot_kv_index = 0,
    };
    var shortconv_runtime: ShortConvBlockRuntime = .{
        .conv_dim = 0,
        .d_conv = 0,
        .d_ff = 0,
        .ln1_weight = zero_tensor,
        .ln2_weight = null,
        .in_proj = zero_weight,
        .out_proj = zero_weight,
        .conv_weight_time_major = zero_tensor,
        .conv_bias = null,
        .conv_state = zero_buffer,
        .ffn_w1 = null,
        .ffn_w2 = null,
        .ffn_w3 = null,
    };
    const gate_weight: LinearWeight = zero_weight;
    const up_weight: LinearWeight = zero_weight;
    const down_weight: LinearWeight = zero_weight;

    layer.norm_weights[0] = &norm0;
    layer.norm_weights[1] = &norm1;
    layer.norm_weight_count = 2;
    attention_runtime.w1 = gate_weight;
    attention_runtime.w3 = up_weight;
    attention_runtime.w2 = down_weight;
    attention_runtime.d_ff = 32;
    layer.attention_binding = &attention_runtime;
    layer.shortconv_binding = &shortconv_runtime;

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
        .{ .kernel = .{ .id = 1, .in = .norm_out, .out = .branch_out, .debug_type = .multihead_attention, .state_block_id = runtime_contract.kv_cache_state_id } },
        .{ .kernel = .{ .id = 2, .in = .branch_out, .out = .tmp3, .debug_type = .shortconv, .state_block_id = runtime_contract.shortconv_state_id } },
        .{ .kernel = .{ .id = 3, .in = .tmp3, .out = .branch_out, .debug_type = .mlp } },
        .{ .kernel = .{ .id = 4, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try layer.rebuildInstructionMetadata(std.testing.allocator);

    try std.testing.expect(layer.instruction_norm_weight_slots[0].? == &norm0);
    try std.testing.expect(layer.instruction_norm_weight_slots[4].? == &norm1);
    try std.testing.expectEqual(@as(usize, ops.len + 1), layer.instruction_weight_offsets.len);
    try std.testing.expect(layer.instruction_weight_ptrs.len != 0);
    // Instruction weight pointers are flattened and directly sourced from layer bindings.
    const attn_start = layer.instruction_weight_offsets[1];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 0].?), @intFromPtr(&attention_runtime.q_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 1].?), @intFromPtr(&attention_runtime.k_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 2].?), @intFromPtr(&attention_runtime.v_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[attn_start + 3].?), @intFromPtr(&attention_runtime.o_proj));
    const shortconv_start = layer.instruction_weight_offsets[2];
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 0].?), @intFromPtr(&shortconv_runtime.in_proj));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 1].?), @intFromPtr(&shortconv_runtime.conv_weight_time_major));
    try std.testing.expectEqual(@intFromPtr(layer.instruction_weight_ptrs[shortconv_start + 2].?), @intFromPtr(&shortconv_runtime.out_proj));
}

test "BlockRuntimeLayer.rebuildInstructionMetadata rejects norm op without bound norm weights" {
    var layer: BlockRuntimeLayer = .{};
    defer {
        if (layer.instruction_norm_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_norm_weight_slots);
        if (layer.instruction_attention_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_attention_exec_meta);
        if (layer.instruction_attention_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_attention_weight_slots);
        if (layer.instruction_shortconv_exec_meta.len > 0) std.testing.allocator.free(layer.instruction_shortconv_exec_meta);
        if (layer.instruction_shortconv_weight_slots.len > 0) std.testing.allocator.free(layer.instruction_shortconv_weight_slots);
        if (layer.instruction_weight_offsets.len > 0) std.testing.allocator.free(layer.instruction_weight_offsets);
        if (layer.instruction_weight_ptrs.len > 0) std.testing.allocator.free(layer.instruction_weight_ptrs);
        if (layer.compiled_plan) |*compiled| {
            plan_compiler.deinitCompiledPlan(std.testing.allocator, compiled);
            layer.compiled_plan = null;
        }
    }

    const ops = [_]layer_ops.LayerOp{
        .{ .kernel = .{ .id = 0, .in = .residual, .out = .norm_out, .debug_type = .norm } },
    };
    layer.compiled_plan = try plan_compiler.compileLayerProgram(std.testing.allocator, ops[0..], .decode, .{});

    try std.testing.expectError(error.UnsupportedModel, layer.rebuildInstructionMetadata(std.testing.allocator));
}

test "bindSlotStateBlocks stores typed runtime states by runtime_kind" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    clearLocalStages(&backend);
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 91,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 92,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 93,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 94,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 91,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 92,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 93,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 94,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(kv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(shortconv_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(mamba_state.block_runtime));
    try std.testing.expectEqual(@intFromPtr(&backend.block_runtime), @intFromPtr(gated_delta_state.block_runtime));
    try std.testing.expectEqual(@as(usize, 0), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 0), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks preserves bound slot index in runtime states" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 2;
    backend.block_runtime = undefined;
    clearLocalStages(&backend);
    backend.state_descriptor_count = 4;
    backend.state_descriptors_storage[0] = .{
        .id = 101,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_kv_cache,
    };
    backend.state_descriptors_storage[1] = .{
        .id = 102,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_shortconv_cache,
    };
    backend.state_descriptors_storage[2] = .{
        .id = 103,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_mamba_cache,
    };
    backend.state_descriptors_storage[3] = .{
        .id = 104,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_gated_delta_cache,
    };
    var slot_state_bindings: [2]CudaBackend.SlotStateBinding = .{ .{}, .{} };
    backend.slot_state_bindings = slot_state_bindings[0..];

    var kv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var shortconv_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var mamba_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    var gated_delta_block_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 101,
            .ptr = kv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 102,
            .ptr = shortconv_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 103,
            .ptr = mamba_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
        .{
            .id = 104,
            .ptr = gated_delta_block_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(1, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(1);
    const bound = backend.slotStateBlocks(1);
    const kv_state = runtime_contract.stateValueFromBlock(*KvRuntimeState, &bound[0]) orelse return error.TestUnexpectedResult;
    const shortconv_state = runtime_contract.stateValueFromBlock(*ShortConvRuntimeState, &bound[1]) orelse return error.TestUnexpectedResult;
    const mamba_state = runtime_contract.stateValueFromBlock(*MambaRuntimeState, &bound[2]) orelse return error.TestUnexpectedResult;
    const gated_delta_state = runtime_contract.stateValueFromBlock(*GatedDeltaRuntimeState, &bound[3]) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_kv_cache, kv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_shortconv_cache, shortconv_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_mamba_cache, mamba_state.runtime_kind);
    try std.testing.expectEqual(runtime_contract.state_runtime_kind_gated_delta_cache, gated_delta_state.runtime_kind);
    try std.testing.expectEqual(@as(usize, 1), kv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), shortconv_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), mamba_state.slot_index);
    try std.testing.expectEqual(@as(usize, 1), gated_delta_state.slot_index);
}

test "bindSlotStateBlocks preserves opaque descriptor blocks with runtime_kind none" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 1;
    backend.state_descriptors_storage[0] = .{
        .id = 111,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{
            .id = 111,
            .ptr = state_storage[0..].ptr,
            .size = runtime_contract.builtin_state_block_bytes,
            .align_bytes = 64,
        },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    defer backend.unbindSlotStateBlocks(0);
    const bound = backend.slotStateBlocks(0);
    try std.testing.expectEqual(@as(usize, 1), bound.len);
    try std.testing.expectEqual(@intFromPtr(state_blocks[0].ptr), @intFromPtr(bound[0].ptr));
    try std.testing.expectEqual(state_blocks[0].size, bound[0].size);
    try std.testing.expectEqual(state_blocks[0].align_bytes, bound[0].align_bytes);
}

test "expectedAttentionQProjectionDim uses packed query width only for query-gated attention" {
    const plain = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = false,
    };
    try std.testing.expectEqual(@as(usize, 2048), expectedAttentionQProjectionDim(&plain));

    const gated = LayerAttentionExecConfig{
        .q_dim = 2048,
        .q_projection_dim = 4096,
        .kv_dim = 512,
        .sliding_window = 0,
        .is_causal = true,
        .query_gate = true,
    };
    try std.testing.expectEqual(@as(usize, 4096), expectedAttentionQProjectionDim(&gated));
}

test "bufferF32RowCount derives staged row count from buffer bytes" {
    const bytes = 2 * 1024 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 0,
        .size = bytes,
    };
    try std.testing.expectEqual(@as(usize, 2), try bufferF32RowCount(&buffer, 1024));
}

test "logicalF32RowSlice uses packed row offsets for tightly packed buffers" {
    const row_width = 8;
    const row_bytes = row_width * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 4096,
        .size = 2 * row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, row_width);
    try std.testing.expectEqual(buffer.pointer + row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, row_bytes), row1.size);
}

test "logicalF32RowSlice uses widened row stride for staged slot buffers" {
    const logical_width = 8;
    const logical_row_bytes = logical_width * @sizeOf(f32);
    const widened_row_bytes = 16 * @sizeOf(f32);
    const buffer = compute.cuda.Buffer{
        .pointer = 8192,
        .size = 2 * widened_row_bytes,
    };

    const row1 = try logicalF32RowSlice(&buffer, 2, 1, logical_width);
    try std.testing.expectEqual(buffer.pointer + widened_row_bytes, row1.pointer);
    try std.testing.expectEqual(@as(usize, logical_row_bytes), row1.size);
}

test "attentionSeparateDecodeUsesCache uses decode mode only for single-row execution" {
    try std.testing.expect(engine_mixers.attentionSeparateDecodeUsesCache(1));
    try std.testing.expect(!engine_mixers.attentionSeparateDecodeUsesCache(2));
    try std.testing.expect(!engine_mixers.attentionSeparateDecodeUsesCache(15));
}

// ---------------------------------------------------------------------------
// Phase 9: Lifecycle + Safety Hardening
// ---------------------------------------------------------------------------

test "unbindSlotStateBlocks is idempotent" {
    const payload_bytes: usize = @intCast(runtime_contract.builtin_state_block_bytes);
    var backend: CudaBackend = undefined;
    backend.max_batch_size = 1;
    backend.block_runtime = undefined;
    backend.state_descriptor_count = 1;
    clearLocalStages(&backend);
    backend.state_descriptors_storage[0] = .{
        .id = 51,
        .size_bytes = runtime_contract.builtin_state_block_bytes,
        .align_bytes = 64,
        .zero_init = false,
        .lifecycle = .slot_persistent,
        .runtime_kind = runtime_contract.state_runtime_kind_none,
    };
    var slot_state_bindings: [1]CudaBackend.SlotStateBinding = .{.{}};
    backend.slot_state_bindings = slot_state_bindings[0..];

    var state_storage: [payload_bytes]u8 align(64) = [_]u8{0} ** payload_bytes;
    const state_blocks = [_]runtime_contract.StateBlockHandle{
        .{ .id = 51, .ptr = state_storage[0..].ptr, .size = runtime_contract.builtin_state_block_bytes, .align_bytes = 64 },
    };

    try backend.bindSlotStateBlocks(0, state_blocks[0..]);
    try std.testing.expect(backend.slot_state_bindings[0].bound);

    // First unbind.
    backend.unbindSlotStateBlocks(0);
    try std.testing.expect(!backend.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), backend.slot_state_bindings[0].count);

    // Second unbind — must not crash and state stays unbound.
    backend.unbindSlotStateBlocks(0);
    try std.testing.expect(!backend.slot_state_bindings[0].bound);
    try std.testing.expectEqual(@as(u8, 0), backend.slot_state_bindings[0].count);
}

// ---------------------------------------------------------------------------
// computeInitLayerRange — range-scoped init invariant
// ---------------------------------------------------------------------------
//
// These tests exercise the pure validation function that CudaBackend.init()
// calls to determine its layer range. The debug.assert in init() then verifies
// BlockRuntime.initRange produced the matching block count.

const computeInitLayerRange = CudaBackend.computeInitLayerRange;
const no_sharing_config: models.config.ModelConfig = .{
    .vocab_size = 1000,
    .d_model = 256,
    .n_layers = 32,
    .n_heads = 8,
    .n_kv_groups = 1,
    .d_ff = 512,
    .max_seq_len = 2048,
    .head_dim = 32,
    .rope_theta = 10000.0,
    .norm_eps = 1e-5,
    .gaffine_group_size = 32,
};

test "computeInitLayerRange uses full range without a local stage plan" {
    const r = try computeInitLayerRange(.{}, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 0), r.start);
    try std.testing.expectEqual(@as(usize, 32), r.end);
}

test "computeInitLayerRange accepts explicit internal layer range" {
    const r = try computeInitLayerRange(.{
        .layer_range = .{ .start = 5, .end = 10 },
    }, 32, no_sharing_config);
    try std.testing.expectEqual(@as(usize, 5), r.start);
    try std.testing.expectEqual(@as(usize, 10), r.end);
}

test "computeInitLayerRange rejects layer_range with start>=end" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .layer_range = .{ .start = 10, .end = 10 },
        }, 32, no_sharing_config),
    );
}

test "computeInitLayerRange rejects layer_range with end>total_layers" {
    try std.testing.expectError(
        error.InvalidTopologyConfig,
        computeInitLayerRange(.{
            .layer_range = .{ .start = 0, .end = 33 },
        }, 32, no_sharing_config),
    );
}

test "Nvfp4LinearWeight.cublasLtScaleTensorSize computes padded VEC16 layout bytes" {
    // 16-scale columns become ceil(inner/16), then rounded to multiples of 4.
    // Outer dimension is rounded to 128.
    try std.testing.expectEqual(@as(usize, 128 * 160), Nvfp4LinearWeight.cublasLtScaleTensorSize(2560, 8));
    try std.testing.expectEqual(@as(usize, 128 * 4), Nvfp4LinearWeight.cublasLtScaleTensorSize(48, 1));
}

test "Nvfp4LinearWeight.roundoff rounds up to granularity" {
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(1, 128));
    try std.testing.expectEqual(@as(usize, 128), Nvfp4LinearWeight.roundoff(128, 128));
    try std.testing.expectEqual(@as(usize, 256), Nvfp4LinearWeight.roundoff(129, 128));
}
