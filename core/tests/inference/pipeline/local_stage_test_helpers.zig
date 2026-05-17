//! Test helpers for pipeline-owned local stage contracts.

const std = @import("std");
const main = @import("main");

const models = main.models.dispatcher;
const pipeline = main.inference.pipeline;
const local_stage = pipeline.local_stage_contract;

pub const runtime_stage_required_step_kinds = local_stage.required_step_kinds;
pub const BoundaryConfig = local_stage.BoundaryConfig;
pub const StageBackendKind = pipeline.HostBackendKind;
pub const StageSpec = local_stage.StageSpec;
pub const BoundaryRuntime = local_stage.BoundaryRuntime;
pub const deterministicHostId = local_stage.deterministicHostId;
pub const localBoundaryConfig = local_stage.boundaryConfig;
pub const localMinBoundaryConfig = local_stage.minBoundaryConfig;
pub const localTwoBoundaryConfigs = local_stage.twoBoundaryConfigs;
pub const validateStageSpecs = local_stage.validateStageSpecs;
pub const validateBoundaryRuntimes = local_stage.validateBoundaryRuntimes;
pub const boundaryRowByteCount = local_stage.boundaryRowByteCount;
pub const boundaryProfilePair = local_stage.boundaryProfilePair;
pub const ContractField = local_stage.ContractField;
pub const contract_deinit_order = local_stage.contract_deinit_order;
pub const LocalStageContractBundle = local_stage.ContractBundle;

pub fn testModelConfig(layer_count: usize) models.config.ModelConfig {
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

pub fn testArchitecture() models.op_types.Architecture {
    return .{
        .name = "runtime_local_stage_test",
        .model_types = &.{"runtime_local_stage_test"},
    };
}

pub fn testManifest(allocator: std.mem.Allocator, layer_count: usize) !models.manifest.ModelManifest {
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
        .architecture_id = "runtime_local_stage_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}

pub fn buildLocalStageTestStagePlan(
    allocator: std.mem.Allocator,
    layer_count: usize,
    splits: []const usize,
    dependencies: []const models.stage_plan.DependencyOverride,
) !models.stage_plan.StagePlan {
    var arch = testArchitecture();
    var config = testModelConfig(layer_count);
    var manifest = try testManifest(allocator, layer_count);
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

pub fn buildLocalStageStateOwnershipPlan(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) !pipeline.StageStateOwnershipPlan {
    return local_stage.buildStateOwnershipPlan(allocator, plan);
}

pub fn buildLocalStagePlacementPlan(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
    state_ref: ?*const pipeline.StageStatePlacementRef,
) !pipeline.PlacementPlan {
    return local_stage.buildPlacementPlan(allocator, d_model, plan, stage_backend_kinds, boundary_configs, state_ref);
}

pub fn buildLocalStageContractBundleFromOwnedPlan(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: models.stage_plan.StagePlan,
    stage_backend_kinds: []const pipeline.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
) !LocalStageContractBundle {
    var bundle = LocalStageContractBundle{ .stage_plan = plan };
    errdefer bundle.deinit();

    const plan_ptr = &bundle.stage_plan.?;
    bundle.tensor_frame_plan_ref = try pipeline.TensorFramePlanRef.fromStagePlan(allocator, plan_ptr);

    if (local_stage.countStatefulDependencies(plan_ptr) > 0) {
        bundle.state_ownership_plan = try local_stage.buildStateOwnershipPlan(allocator, plan_ptr);
        if (bundle.state_ownership_plan) |*state_plan| {
            bundle.state_placement_ref = try pipeline.buildStageStatePlacementRef(allocator, state_plan);
        }
    }

    const state_ref_ptr: ?*const pipeline.StageStatePlacementRef = if (bundle.state_placement_ref) |*state_ref| state_ref else null;
    bundle.placement_plan = try local_stage.buildPlacementPlan(
        allocator,
        d_model,
        plan_ptr,
        stage_backend_kinds,
        boundary_configs,
        state_ref_ptr,
    );
    const state_plan_ptr: ?*const pipeline.StageStateOwnershipPlan = if (bundle.state_ownership_plan) |*state_plan| state_plan else null;
    bundle.local_stage_runner_plan_ref = try pipeline.buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = plan_ptr,
        .tensor_frame_plan_ref = &bundle.tensor_frame_plan_ref.?,
        .placement_plan = &bundle.placement_plan.?,
        .state_ownership_plan = state_plan_ptr,
    });
    return bundle;
}

pub fn deinitLocalStageContractBundleTwice(bundle: *LocalStageContractBundle) void {
    bundle.deinit();
    bundle.deinit();
}
