//! Test helpers for bridge-owned local stage contracts.

const std = @import("std");
const main = @import("main");

const models = main.models.dispatcher;
const bridge = main.inference.bridge;
const local_stage = bridge.local_stage_contract;

pub const bridge_stage_required_step_kinds = local_stage.required_step_kinds;
pub const BoundaryConfig = local_stage.BoundaryConfig;
pub const StageBackendKind = bridge.HostBackendKind;
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

pub fn buildLocalStageStateOwnershipPlan(
    allocator: std.mem.Allocator,
    plan: *const models.stage_plan.StagePlan,
) !bridge.StageStateOwnershipPlan {
    return local_stage.buildStateOwnershipPlan(allocator, plan);
}

pub fn buildLocalStagePlacementPlan(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: *const models.stage_plan.StagePlan,
    stage_backend_kinds: []const bridge.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
    state_ref: ?*const bridge.StageStatePlacementRef,
) !bridge.PlacementPlan {
    return local_stage.buildPlacementPlan(allocator, d_model, plan, stage_backend_kinds, boundary_configs, state_ref);
}

pub fn buildLocalStageContractBundleFromOwnedPlan(
    allocator: std.mem.Allocator,
    d_model: usize,
    plan: models.stage_plan.StagePlan,
    stage_backend_kinds: []const bridge.HostBackendKind,
    boundary_configs: []const BoundaryConfig,
) !LocalStageContractBundle {
    var bundle = LocalStageContractBundle{ .stage_plan = plan };
    errdefer bundle.deinit();

    const plan_ptr = &bundle.stage_plan.?;
    bundle.tensor_frame_plan_ref = try bridge.TensorFramePlanRef.fromStagePlan(allocator, plan_ptr);

    if (local_stage.countStatefulDependencies(plan_ptr) > 0) {
        bundle.state_ownership_plan = try local_stage.buildStateOwnershipPlan(allocator, plan_ptr);
        if (bundle.state_ownership_plan) |*state_plan| {
            bundle.state_placement_ref = try bridge.buildStageStatePlacementRef(allocator, state_plan);
        }
    }

    const state_ref_ptr: ?*const bridge.StageStatePlacementRef = if (bundle.state_placement_ref) |*state_ref| state_ref else null;
    bundle.placement_plan = try local_stage.buildPlacementPlan(
        allocator,
        d_model,
        plan_ptr,
        stage_backend_kinds,
        boundary_configs,
        state_ref_ptr,
    );
    const state_plan_ptr: ?*const bridge.StageStateOwnershipPlan = if (bundle.state_ownership_plan) |*state_plan| state_plan else null;
    bundle.local_stage_runner_plan_ref = try bridge.buildLocalStageRunnerPlanRef(allocator, .{
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
