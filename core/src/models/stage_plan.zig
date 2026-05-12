//! Deterministic model graph identity and logical decoder stage plans.
//!
//! This module is a pure models contract. It validates logical stage ownership
//! and derives loader requests, but it does not initialize backends or schedule
//! runtime execution.

const std = @import("std");
const config_types = @import("config/types.zig");
const loader = @import("loader.zig");
const manifest_mod = @import("manifest.zig");
const op_types = @import("op_types.zig");

const Allocator = std.mem.Allocator;
const Architecture = op_types.Architecture;
const DType = @import("compute_pkg").dtype.DType;
const ModelConfig = config_types.ModelConfig;
const ModelManifest = manifest_mod.ModelManifest;
const RopeScaling = config_types.RopeScaling;
const Sha256 = std.crypto.hash.sha2.Sha256;
const StageLoadRequest = loader.StageLoadRequest;
const StageRoleRequest = loader.StageRoleRequest;
const TensorManifestEntry = manifest_mod.TensorManifestEntry;
const TensorRole = manifest_mod.TensorRole;

pub const graph_identity_contract_version: u32 = 1;
pub const stage_plan_contract_version: u32 = 1;
pub const model_config_identity_field_count: usize = 71;
pub const rope_scaling_identity_field_count: usize = 13;

pub const StagePlanError = error{
    OutOfMemory,
    MissingManifest,
    MissingGraphIdentity,
    InvalidLayerCount,
    InvalidSplitPoint,
    DuplicateSplitPoint,
    ForbiddenSplitPoint,
    MissingPartitionConstraints,
    MissingStageDependency,
    UnknownStageId,
    UnsupportedRoleOwnerOverride,
    DuplicateRoleOwnerOverride,
    MissingRoleOwner,
    UnclassifiedGlobalNotAllowed,
    InvalidDependency,
    InvalidContractVersion,
    GraphIdentityMismatch,
    MissingRoleSemantics,
    DuplicateDependency,
    InvalidStageRange,
    DuplicateStageId,
    NonContiguousStageRange,
    ResidencyMismatch,
    PlanFingerprintMismatch,
};

pub const LoadSemantics = struct {
    preserve_native_norm_dtype: bool = false,
    dequantize_mxfp8_to_bf16: bool = false,
    dequantize_nvfp4_to_bf16: bool = true,

    pub fn fromLoadOptions(options: loader.LoadOptions) LoadSemantics {
        return .{
            .preserve_native_norm_dtype = options.preserve_native_norm_dtype,
            .dequantize_mxfp8_to_bf16 = options.dequantize_mxfp8_to_bf16,
            .dequantize_nvfp4_to_bf16 = options.dequantize_nvfp4_to_bf16,
        };
    }
};

pub const GraphIdentityInputs = struct {
    graph_contract_version: u32 = graph_identity_contract_version,
    stage_contract_version: u32 = stage_plan_contract_version,
    architecture_id: []const u8,
    architecture: *const Architecture,
    config: *const ModelConfig,
    manifest: *const ModelManifest,
    load_semantics: LoadSemantics = .{},
};

/// Graph identity value. `architecture_id` is a borrowed slice unless this
/// value came from `dupeGraphIdentity` or from a `StagePlan`, which owns a copy
/// in its arena.
pub const GraphIdentity = struct {
    graph_contract_version: u32 = graph_identity_contract_version,
    stage_contract_version: u32 = stage_plan_contract_version,
    architecture_id: []const u8 = "",
    digest: [32]u8,
};

pub const StagePlanId = struct {
    digest: [32]u8,
};

pub const PartitionConstraintSource = enum(u8) {
    single_stage,
    explicit,
    derived_plain_decoder,
};

pub const StagePlanValidationOptions = struct {
    expected_graph_identity: ?GraphIdentity = null,
    manifest: ?*const ModelManifest = null,
};

pub const ForbiddenSplitPoint = struct {
    layer_index: usize,
    reason: []const u8 = "",
};

pub const RoleOwnerOverride = struct {
    role: TensorRole,
    stage_id: usize,
};

pub const StageDependencyReason = enum(u8) {
    tied_lm_head,
    vision_side,
    architecture_side,
    unclassified_global,
    explicit,
    stateful_decoder,
};

pub const DependencyOverride = struct {
    source_stage_id: usize,
    target_stage_id: usize,
    role: ?TensorRole = null,
    reason: StageDependencyReason = .explicit,
    affects_loader_residency: bool = false,
};

pub const PartitionConstraints = struct {
    decoder_cuts_allowed: bool = false,
    forbidden_split_points: []const ForbiddenSplitPoint = &.{},
    role_owner_overrides: []const RoleOwnerOverride = &.{},
    dependency_overrides: []const DependencyOverride = &.{},
};

pub const StageRoleSemantics = struct {
    /// Required in precomputed-identity mode when the manifest does not carry
    /// an independent lm_head role and tied-output-head behavior would affect
    /// stage ownership or residency.
    tie_word_embeddings: ?bool = null,
};

pub const StagePlanRequest = struct {
    n_layers: usize,
    split_points: []const usize = &.{},
    architecture: ?*const Architecture = null,
    model_config: ?*const ModelConfig = null,
    manifest: ?*const ModelManifest = null,
    graph_identity: ?GraphIdentity = null,
    load_semantics: LoadSemantics = .{},
    role_semantics: StageRoleSemantics = .{},
    partition_constraints: ?PartitionConstraints = null,
    role_owner_overrides: []const RoleOwnerOverride = &.{},
    dependency_overrides: []const DependencyOverride = &.{},
    allow_unclassified_global: bool = false,
};

pub const StageBoundary = struct {
    source_stage_id: usize,
    target_stage_id: usize,
    producer_layer_start: usize,
    producer_layer_end: usize,
    consumer_layer_start: usize,
    consumer_layer_end: usize,
};

pub const StageDependency = struct {
    source_stage_id: usize,
    target_stage_id: usize,
    role: ?TensorRole = null,
    reason: StageDependencyReason,
    affects_loader_residency: bool,
};

pub const StagePlanDiagnosticKind = enum(u8) {
    unclassified_global_explicit_owner,
};

pub const StagePlanDiagnostic = struct {
    kind: StagePlanDiagnosticKind,
    role: TensorRole,
    stage_id: usize,
};

pub const StagePlanStage = struct {
    id: usize,
    layer_start: usize,
    layer_end: usize,
    owned_roles: [manifest_mod.role_count]bool = [_]bool{false} ** manifest_mod.role_count,
    residency: manifest_mod.StageResidencyReport,
};

pub const StagePlan = struct {
    arena: std.heap.ArenaAllocator,
    stage_contract_version: u32,
    graph_identity: GraphIdentity,
    split_points: []const usize,
    plan_id: StagePlanId,
    partition_constraint_source: PartitionConstraintSource,
    n_layers: usize,
    stages: []const StagePlanStage,
    boundaries: []const StageBoundary,
    dependencies: []const StageDependency,
    diagnostics: []const StagePlanDiagnostic,

    pub fn deinit(self: *StagePlan) void {
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn stage(self: *const StagePlan, stage_id: usize) StagePlanError!*const StagePlanStage {
        if (stage_id >= self.stages.len) return error.UnknownStageId;
        return &self.stages[stage_id];
    }

    pub fn stageLoadRequest(self: *const StagePlan, stage_id: usize) StagePlanError!StageLoadRequest {
        return stageLoadRequestFromParts(stage_id, self.stages, self.dependencies);
    }
};

/// Returns a graph identity whose `architecture_id` borrows
/// `inputs.architecture_id`. Use `dupeGraphIdentity` before the input backing
/// memory is released when the identity must outlive its inputs.
pub fn graphIdentity(
    allocator: Allocator,
    inputs: GraphIdentityInputs,
) StagePlanError!GraphIdentity {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.graph_identity");
    encoder.writeU32(inputs.graph_contract_version);
    encoder.writeU32(inputs.stage_contract_version);
    try writeArchitecture(&encoder, allocator, inputs.architecture_id, inputs.architecture);
    writeModelConfig(&encoder, inputs.config);
    try writeManifest(&encoder, allocator, inputs.manifest);
    writeLoadSemantics(&encoder, inputs.load_semantics);
    return .{
        .graph_contract_version = inputs.graph_contract_version,
        .stage_contract_version = inputs.stage_contract_version,
        .architecture_id = inputs.architecture_id,
        .digest = encoder.finish(),
    };
}

/// Caller owns the returned `architecture_id` and must release it with
/// `deinitGraphIdentity`.
pub fn dupeGraphIdentity(allocator: Allocator, identity: GraphIdentity) StagePlanError!GraphIdentity {
    return .{
        .graph_contract_version = identity.graph_contract_version,
        .stage_contract_version = identity.stage_contract_version,
        .architecture_id = try allocator.dupe(u8, identity.architecture_id),
        .digest = identity.digest,
    };
}

/// Releases a `GraphIdentity` created by `dupeGraphIdentity`.
pub fn deinitGraphIdentity(allocator: Allocator, identity: *GraphIdentity) void {
    allocator.free(identity.architecture_id);
    identity.* = undefined;
}

pub fn graphIdentityEql(lhs: GraphIdentity, rhs: GraphIdentity) bool {
    return lhs.graph_contract_version == rhs.graph_contract_version and
        lhs.stage_contract_version == rhs.stage_contract_version and
        std.mem.eql(u8, lhs.architecture_id, rhs.architecture_id) and
        std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

pub fn validateGraphIdentity(identity: GraphIdentity) StagePlanError!void {
    if (identity.graph_contract_version != graph_identity_contract_version or
        identity.stage_contract_version != stage_plan_contract_version)
    {
        return error.InvalidContractVersion;
    }
    if (identity.architecture_id.len == 0) return error.MissingGraphIdentity;
}

pub fn assertGraphIdentity(plan: *const StagePlan, expected: GraphIdentity) StagePlanError!void {
    if (!graphIdentityEql(plan.graph_identity, expected)) return error.GraphIdentityMismatch;
}

pub fn validateStagePlan(plan: *const StagePlan, options: StagePlanValidationOptions) StagePlanError!void {
    if (plan.stage_contract_version != stage_plan_contract_version) return error.InvalidContractVersion;
    try validateGraphIdentity(plan.graph_identity);
    if (options.expected_graph_identity) |expected| {
        try validateGraphIdentity(expected);
        try assertGraphIdentity(plan, expected);
    }
    try validateStagePlanRanges(plan);
    try validateStagePlanBoundaries(plan);
    try validateStagePlanDependencies(plan);
    try validateStagePlanResidency(plan, options.manifest);

    const expected_plan_id = computeStagePlanId(.{
        .stage_contract_version = plan.stage_contract_version,
        .graph_identity = plan.graph_identity,
        .split_points = plan.split_points,
        .n_layers = plan.n_layers,
        .stages = plan.stages,
        .boundaries = plan.boundaries,
        .dependencies = plan.dependencies,
        .diagnostics = plan.diagnostics,
    });
    if (!stagePlanIdEql(plan.plan_id, expected_plan_id)) return error.PlanFingerprintMismatch;
}

pub fn buildStagePlan(
    allocator: Allocator,
    request: StagePlanRequest,
) StagePlanError!StagePlan {
    const model_manifest = request.manifest orelse return error.MissingManifest;
    try validateLayerCount(request);
    try validateSplitPoints(request.n_layers, request.split_points);
    const stage_count = request.split_points.len + 1;
    try validateRoleSemantics(request);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const resolved_graph_identity = try resolveGraphIdentity(allocator, arena_allocator, request, stage_count);
    const partition_constraint_source = try validatePartitioning(request, stage_count);
    try validateRoleOwnerOverrides(request, stage_count);
    try validateDependencyOverrides(request, stage_count);

    const split_points = try arena_allocator.dupe(usize, request.split_points);
    const stages = try arena_allocator.alloc(StagePlanStage, stage_count);
    fillStages(stages, request.n_layers, split_points);

    const boundary_count = if (stage_count > 1) stage_count - 1 else 0;
    const boundaries = try arena_allocator.alloc(StageBoundary, boundary_count);
    fillBoundaries(boundaries, stages);

    var diagnostics = std.ArrayListUnmanaged(StagePlanDiagnostic){};
    var dependencies = std.ArrayListUnmanaged(StageDependency){};

    try assignRoleOwnership(stages, &diagnostics, arena_allocator, request);
    try validateRequiredSideDomainDependencies(request, stages);
    try appendTiedLmHeadDependency(&dependencies, arena_allocator, request, stages);
    try appendDependencyOverrides(&dependencies, arena_allocator, request, stages);

    const dependency_slice = try dependencies.toOwnedSlice(arena_allocator);
    const diagnostic_slice = try diagnostics.toOwnedSlice(arena_allocator);
    try fillStageResidency(stages, dependency_slice, model_manifest);

    const plan_id = computeStagePlanId(.{
        .stage_contract_version = stage_plan_contract_version,
        .graph_identity = resolved_graph_identity,
        .split_points = split_points,
        .n_layers = request.n_layers,
        .stages = stages,
        .boundaries = boundaries,
        .dependencies = dependency_slice,
        .diagnostics = diagnostic_slice,
    });

    return .{
        .arena = arena,
        .stage_contract_version = stage_plan_contract_version,
        .graph_identity = resolved_graph_identity,
        .split_points = split_points,
        .plan_id = plan_id,
        .partition_constraint_source = partition_constraint_source,
        .n_layers = request.n_layers,
        .stages = stages,
        .boundaries = boundaries,
        .dependencies = dependency_slice,
        .diagnostics = diagnostic_slice,
    };
}

fn resolveGraphIdentity(
    allocator: Allocator,
    arena_allocator: Allocator,
    request: StagePlanRequest,
    stage_count: usize,
) StagePlanError!GraphIdentity {
    if (request.architecture) |architecture| {
        if (request.model_config) |config| {
            const model_manifest = request.manifest orelse return error.MissingManifest;
            const expected = try graphIdentity(allocator, .{
                .architecture_id = model_manifest.architecture_id,
                .architecture = architecture,
                .config = config,
                .manifest = model_manifest,
                .load_semantics = request.load_semantics,
            });
            if (request.graph_identity) |identity| {
                try validateGraphIdentity(identity);
                if (!graphIdentityEql(identity, expected)) return error.GraphIdentityMismatch;
            }
            try validateGraphIdentity(expected);
            return copyGraphIdentity(arena_allocator, expected);
        }
    }

    if (request.graph_identity) |identity| {
        try validateStandaloneGraphIdentityRequest(identity, request, stage_count);
        return copyGraphIdentity(arena_allocator, identity);
    }

    return error.MissingGraphIdentity;
}

fn validateStandaloneGraphIdentityRequest(
    identity: GraphIdentity,
    request: StagePlanRequest,
    stage_count: usize,
) StagePlanError!void {
    try validateGraphIdentity(identity);
    if (request.architecture != null or request.model_config != null) return error.GraphIdentityMismatch;
    if (stage_count > 1 and request.partition_constraints == null) return error.MissingPartitionConstraints;
    try validateStandaloneRoleSemantics(request);
}

fn validateRoleSemantics(request: StagePlanRequest) StagePlanError!void {
    if (request.model_config) |config| {
        if (request.role_semantics.tie_word_embeddings) |explicit| {
            if (explicit != config.tie_word_embeddings) return error.GraphIdentityMismatch;
        }
    }
}

fn validateStandaloneRoleSemantics(request: StagePlanRequest) StagePlanError!void {
    if (request.role_semantics.tie_word_embeddings != null) return;
    const model_manifest = request.manifest orelse return error.MissingManifest;
    if (model_manifest.hasGlobalWeight("token_embeddings") and !hasIndependentLmHead(model_manifest)) {
        return error.MissingRoleSemantics;
    }
}

fn copyGraphIdentity(allocator: Allocator, identity: GraphIdentity) StagePlanError!GraphIdentity {
    return dupeGraphIdentity(allocator, identity);
}

fn validateStagePlanRanges(plan: *const StagePlan) StagePlanError!void {
    if (plan.n_layers == 0 or plan.stages.len == 0) return error.InvalidLayerCount;
    if (plan.split_points.len + 1 != plan.stages.len) return error.InvalidSplitPoint;
    try validateSplitPoints(plan.n_layers, plan.split_points);

    for (plan.stages, 0..) |stage_entry, stage_index| {
        if (stage_entry.id >= plan.stages.len) return error.UnknownStageId;
        for (plan.stages[0..stage_index]) |previous| {
            if (previous.id == stage_entry.id) return error.DuplicateStageId;
        }
    }

    var expected_start: usize = 0;
    for (plan.stages, 0..) |stage_entry, stage_index| {
        if (stage_entry.id != stage_index) return error.NonContiguousStageRange;
        if (stage_entry.layer_start >= stage_entry.layer_end or stage_entry.layer_end > plan.n_layers) {
            return error.InvalidStageRange;
        }
        if (stage_entry.layer_start != expected_start) return error.NonContiguousStageRange;
        if (stage_index < plan.split_points.len and plan.split_points[stage_index] != stage_entry.layer_end) {
            return error.NonContiguousStageRange;
        }
        expected_start = stage_entry.layer_end;
    }
    if (expected_start != plan.n_layers) return error.NonContiguousStageRange;
}

fn validateStagePlanBoundaries(plan: *const StagePlan) StagePlanError!void {
    const expected_boundary_count = if (plan.stages.len > 1) plan.stages.len - 1 else 0;
    if (plan.boundaries.len != expected_boundary_count) return error.NonContiguousStageRange;

    for (plan.boundaries, 0..) |boundary, index| {
        const source = plan.stages[index];
        const target = plan.stages[index + 1];
        if (boundary.source_stage_id != source.id or boundary.target_stage_id != target.id) {
            return error.NonContiguousStageRange;
        }
        if (boundary.producer_layer_start != source.layer_start or
            boundary.producer_layer_end != source.layer_end or
            boundary.consumer_layer_start != target.layer_start or
            boundary.consumer_layer_end != target.layer_end)
        {
            return error.NonContiguousStageRange;
        }
    }
}

fn validateStagePlanDependencies(plan: *const StagePlan) StagePlanError!void {
    for (plan.dependencies, 0..) |dependency, index| {
        if (dependency.source_stage_id >= plan.stages.len or dependency.target_stage_id >= plan.stages.len) {
            return error.UnknownStageId;
        }
        if (dependency.source_stage_id == dependency.target_stage_id) return error.InvalidDependency;
        if (dependency.affects_loader_residency and dependency.role == null) return error.InvalidDependency;
        if (dependency.role) |role| {
            switch (role) {
                .decoder_layer, .quant_companion => return error.InvalidDependency,
                else => {},
            }
            if (!plan.stages[dependency.source_stage_id].owned_roles[@intFromEnum(role)]) {
                return error.InvalidDependency;
            }
        }
        for (plan.dependencies[0..index]) |previous| {
            if (dependencyIdentityEql(previous, dependency)) return error.DuplicateDependency;
        }
    }
}

fn validateStagePlanResidency(
    plan: *const StagePlan,
    model_manifest: ?*const ModelManifest,
) StagePlanError!void {
    for (plan.stages) |stage_entry| {
        if (stage_entry.residency.layer_start != stage_entry.layer_start or
            stage_entry.residency.layer_end != stage_entry.layer_end)
        {
            return error.ResidencyMismatch;
        }
        if (model_manifest) |manifest| {
            const request = try plan.stageLoadRequest(stage_entry.id);
            const expected = try stageResidencyReportForRequest(manifest, request);
            if (!stageResidencyReportEql(stage_entry.residency, expected)) return error.ResidencyMismatch;
        }
    }
}

fn stageResidencyReportEql(
    lhs: manifest_mod.StageResidencyReport,
    rhs: manifest_mod.StageResidencyReport,
) bool {
    return lhs.layer_start == rhs.layer_start and
        lhs.layer_end == rhs.layer_end and
        lhs.total_checkpoint_bytes == rhs.total_checkpoint_bytes and
        std.mem.eql(usize, &lhs.role_bytes, &rhs.role_bytes);
}

fn stagePlanIdEql(lhs: StagePlanId, rhs: StagePlanId) bool {
    return std.mem.eql(u8, &lhs.digest, &rhs.digest);
}

fn validateLayerCount(request: StagePlanRequest) StagePlanError!void {
    if (request.n_layers == 0) return error.InvalidLayerCount;
    if (request.model_config) |config| {
        if (config.n_layers < 0) return error.InvalidLayerCount;
        if (@as(usize, @intCast(config.n_layers)) != request.n_layers) return error.InvalidLayerCount;
    }
    const model_manifest = request.manifest orelse return error.MissingManifest;
    if (model_manifest.layer_count != request.n_layers) return error.InvalidLayerCount;
}

fn validateSplitPoints(n_layers: usize, split_points: []const usize) StagePlanError!void {
    var previous: usize = 0;
    for (split_points, 0..) |split_point, index| {
        if (split_point == 0 or split_point >= n_layers) return error.InvalidSplitPoint;
        if (index > 0 and split_point == previous) return error.DuplicateSplitPoint;
        if (split_point < previous) return error.InvalidSplitPoint;
        previous = split_point;
    }
}

fn validatePartitioning(request: StagePlanRequest, stage_count: usize) StagePlanError!PartitionConstraintSource {
    if (stage_count == 1) return .single_stage;

    if (request.partition_constraints) |constraints| {
        for (constraints.forbidden_split_points) |forbidden| {
            for (request.split_points) |split_point| {
                if (split_point == forbidden.layer_index) return error.ForbiddenSplitPoint;
            }
        }

        if (stage_count > 1 and !constraints.decoder_cuts_allowed) return error.MissingPartitionConstraints;

        if (stage_count > 1 and
            requiresBoundaryDependencies(request) and
            constraints.decoder_cuts_allowed and
            !hasExplicitBoundaryDependencies(request, stage_count))
        {
            return error.MissingStageDependency;
        }

        return .explicit;
    }

    if (!canDerivePlainIndependentDecoderCuts(request)) return error.MissingPartitionConstraints;
    return .derived_plain_decoder;
}

fn canDerivePlainIndependentDecoderCuts(request: StagePlanRequest) bool {
    const architecture = request.architecture orelse return false;
    const config = request.model_config orelse return false;
    if (architecture.block_variants != null) return false;
    if (architecture.layer_map != null) return false;
    if (config.layer_types != null) return false;
    if (requiresBoundaryDependencies(request)) return false;
    if (config.hidden_size_per_layer_input != 0 or config.vocab_size_per_layer_input != 0) return false;
    if (config.vision_depth != 0 or config.vision_hidden_size != 0) return false;
    return true;
}

pub fn requiresBoundaryDependenciesFor(
    architecture_opt: ?*const Architecture,
    config_opt: ?*const ModelConfig,
) bool {
    if (architecture_opt) |architecture| {
        if (architecture.has_mamba or architecture.has_gated_delta or architecture.has_shortconv or architecture.has_mla) {
            return true;
        }
    }
    if (config_opt) |config| {
        if (config.num_kv_shared_layers != 0) return true;
        if (config.mamba_d_state != 0 or config.mamba_d_conv != 0 or config.mamba_n_heads != 0) return true;
        if (config.shortconv_d_conv != 0 or config.shortconv_conv_dim != 0) return true;
        if (config.linear_num_key_heads != 0 or config.linear_num_value_heads != 0) return true;
    }
    return false;
}

fn requiresBoundaryDependencies(request: StagePlanRequest) bool {
    return requiresBoundaryDependenciesFor(request.architecture, request.model_config);
}

fn hasExplicitBoundaryDependencies(request: StagePlanRequest, stage_count: usize) bool {
    for (0..stage_count - 1) |stage_index| {
        if (!hasDependencyOverride(request, stage_index, stage_index + 1, null)) return false;
    }
    return true;
}

fn hasDependencyOverride(
    request: StagePlanRequest,
    source_stage_id: usize,
    target_stage_id: usize,
    role: ?TensorRole,
) bool {
    for (request.dependency_overrides) |dependency| {
        if (dependency.source_stage_id == source_stage_id and
            dependency.target_stage_id == target_stage_id and
            optionalRoleEql(dependency.role, role))
        {
            return true;
        }
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.dependency_overrides) |dependency| {
            if (dependency.source_stage_id == source_stage_id and
                dependency.target_stage_id == target_stage_id and
                optionalRoleEql(dependency.role, role))
            {
                return true;
            }
        }
    }
    return false;
}

fn optionalRoleEql(lhs: ?TensorRole, rhs: ?TensorRole) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return lhs.? == rhs.?;
}

fn validateRoleOwnerOverrides(request: StagePlanRequest, stage_count: usize) StagePlanError!void {
    var seen = [_]bool{false} ** manifest_mod.role_count;
    for (request.role_owner_overrides) |override| {
        try validateRoleOwnerOverride(override, stage_count, &seen);
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.role_owner_overrides) |override| {
            try validateRoleOwnerOverride(override, stage_count, &seen);
        }
    }
}

fn validateRoleOwnerOverride(
    override: RoleOwnerOverride,
    stage_count: usize,
    seen: *[manifest_mod.role_count]bool,
) StagePlanError!void {
    if (!roleOwnerOverrideAllowed(override.role)) return error.UnsupportedRoleOwnerOverride;
    if (override.stage_id >= stage_count) return error.UnknownStageId;
    const role_index = @intFromEnum(override.role);
    if (seen[role_index]) return error.DuplicateRoleOwnerOverride;
    seen[role_index] = true;
}

fn roleOwnerOverrideAllowed(role: TensorRole) bool {
    return switch (role) {
        .vision_side, .architecture_side, .unclassified_global => true,
        .token_embeddings,
        .decoder_layer,
        .final_norm,
        .lm_head,
        .embedding_side,
        .quant_companion,
        => false,
    };
}

fn validateDependencyOverrides(request: StagePlanRequest, stage_count: usize) StagePlanError!void {
    for (request.dependency_overrides) |dependency| {
        try validateDependencyOverride(dependency, stage_count);
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.dependency_overrides) |dependency| {
            try validateDependencyOverride(dependency, stage_count);
        }
    }
}

fn validateDependencyOverride(dependency: DependencyOverride, stage_count: usize) StagePlanError!void {
    if (dependency.source_stage_id >= stage_count or dependency.target_stage_id >= stage_count) {
        return error.UnknownStageId;
    }
    if (dependency.source_stage_id == dependency.target_stage_id) return error.InvalidDependency;
    if (dependency.affects_loader_residency and dependency.role == null) return error.InvalidDependency;
    if (dependency.role) |role| {
        switch (role) {
            .decoder_layer, .quant_companion => return error.InvalidDependency,
            else => {},
        }
    }
}

fn fillStages(stages: []StagePlanStage, n_layers: usize, split_points: []const usize) void {
    var start: usize = 0;
    for (stages, 0..) |*stage_entry, stage_id| {
        const end = if (stage_id < split_points.len) split_points[stage_id] else n_layers;
        stage_entry.* = .{
            .id = stage_id,
            .layer_start = start,
            .layer_end = end,
            .residency = .{
                .layer_start = start,
                .layer_end = end,
            },
        };
        stage_entry.owned_roles[@intFromEnum(TensorRole.decoder_layer)] = true;
        start = end;
    }
}

fn fillBoundaries(boundaries: []StageBoundary, stages: []const StagePlanStage) void {
    for (boundaries, 0..) |*boundary, index| {
        const source = stages[index];
        const target = stages[index + 1];
        boundary.* = .{
            .source_stage_id = source.id,
            .target_stage_id = target.id,
            .producer_layer_start = source.layer_start,
            .producer_layer_end = source.layer_end,
            .consumer_layer_start = target.layer_start,
            .consumer_layer_end = target.layer_end,
        };
    }
}

fn assignRoleOwnership(
    stages: []StagePlanStage,
    diagnostics: *std.ArrayListUnmanaged(StagePlanDiagnostic),
    allocator: Allocator,
    request: StagePlanRequest,
) StagePlanError!void {
    const first_stage_id: usize = 0;
    const final_stage_id = stages.len - 1;

    if (roleAvailable(request, .token_embeddings)) setRoleOwner(stages, .token_embeddings, first_stage_id);
    if (roleAvailable(request, .embedding_side)) setRoleOwner(stages, .embedding_side, first_stage_id);
    if (roleAvailable(request, .final_norm)) setRoleOwner(stages, .final_norm, final_stage_id);
    if (roleAvailable(request, .lm_head) or tieWordEmbeddings(request)) {
        setRoleOwner(stages, .lm_head, final_stage_id);
    }

    if (roleAvailable(request, .vision_side)) {
        const owner = try roleOwnerOverride(request, .vision_side) orelse first_stage_id;
        setRoleOwner(stages, .vision_side, owner);
    }

    if (roleAvailable(request, .architecture_side)) {
        const owner = try roleOwnerOverride(request, .architecture_side) orelse blk: {
            if (stages.len == 1) break :blk first_stage_id;
            return error.MissingRoleOwner;
        };
        setRoleOwner(stages, .architecture_side, owner);
    }

    if (roleAvailable(request, .unclassified_global)) {
        if (!request.allow_unclassified_global) return error.UnclassifiedGlobalNotAllowed;
        const owner = try roleOwnerOverride(request, .unclassified_global) orelse return error.MissingRoleOwner;
        setRoleOwner(stages, .unclassified_global, owner);
        try diagnostics.append(allocator, .{
            .kind = .unclassified_global_explicit_owner,
            .role = .unclassified_global,
            .stage_id = owner,
        });
    }
}

fn validateRequiredSideDomainDependencies(
    request: StagePlanRequest,
    stages: []const StagePlanStage,
) StagePlanError!void {
    if (stages.len == 1) return;
    const model_manifest = request.manifest orelse return error.MissingManifest;
    try validateRequiredRoleDependency(request, stages, model_manifest, .vision_side);
    try validateRequiredRoleDependency(request, stages, model_manifest, .architecture_side);
    try validateRequiredRoleDependency(request, stages, model_manifest, .unclassified_global);
}

fn validateRequiredRoleDependency(
    request: StagePlanRequest,
    stages: []const StagePlanStage,
    model_manifest: *const ModelManifest,
    role: TensorRole,
) StagePlanError!void {
    if (!model_manifest.hasRole(role)) return;
    _ = findRoleOwner(stages, role) orelse return error.MissingRoleOwner;
    if (!hasDependencyOverrideForRole(request, role)) return error.MissingStageDependency;
}

fn hasDependencyOverrideForRole(request: StagePlanRequest, role: TensorRole) bool {
    for (request.dependency_overrides) |dependency| {
        if (dependency.role) |dependency_role| {
            if (dependency_role == role) return true;
        }
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.dependency_overrides) |dependency| {
            if (dependency.role) |dependency_role| {
                if (dependency_role == role) return true;
            }
        }
    }
    return false;
}

fn roleAvailable(request: StagePlanRequest, role: TensorRole) bool {
    const model_manifest = request.manifest orelse return false;
    return model_manifest.hasRole(role);
}

fn tieWordEmbeddings(request: StagePlanRequest) bool {
    if (request.model_config) |config| return config.tie_word_embeddings;
    return request.role_semantics.tie_word_embeddings orelse false;
}

fn hasIndependentLmHead(model_manifest: *const ModelManifest) bool {
    return model_manifest.hasGlobalWeight("lm_head");
}

fn roleOwnerOverride(request: StagePlanRequest, role: TensorRole) StagePlanError!?usize {
    var found: ?usize = null;
    for (request.role_owner_overrides) |override| {
        if (override.role != role) continue;
        found = override.stage_id;
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.role_owner_overrides) |override| {
            if (override.role != role) continue;
            found = override.stage_id;
        }
    }
    return found;
}

fn setRoleOwner(stages: []StagePlanStage, role: TensorRole, stage_id: usize) void {
    stages[stage_id].owned_roles[@intFromEnum(role)] = true;
}

fn stageLoadRequestFromParts(
    stage_id: usize,
    stages: []const StagePlanStage,
    dependencies: []const StageDependency,
) StagePlanError!StageLoadRequest {
    if (stage_id >= stages.len) return error.UnknownStageId;
    const plan_stage = &stages[stage_id];
    var roles = StageRoleRequest{};

    for (0..manifest_mod.role_count) |role_index| {
        if (!plan_stage.owned_roles[role_index]) continue;
        const role: TensorRole = @enumFromInt(@as(u8, @intCast(role_index)));
        if (role == .decoder_layer or role == .quant_companion) continue;
        try setRoleRequest(&roles, role, true);
    }

    for (dependencies) |dependency| {
        if (dependency.target_stage_id != stage_id or !dependency.affects_loader_residency) continue;
        const role = dependency.role orelse return error.InvalidDependency;
        try setRoleRequest(&roles, role, true);
    }

    return .{
        .layer_start = plan_stage.layer_start,
        .layer_end = plan_stage.layer_end,
        .roles = roles,
    };
}

fn fillStageResidency(
    stages: []StagePlanStage,
    dependencies: []const StageDependency,
    model_manifest: *const ModelManifest,
) StagePlanError!void {
    for (stages) |*stage_entry| {
        const request = try stageLoadRequestFromParts(stage_entry.id, stages, dependencies);
        stage_entry.residency = try stageResidencyReportForRequest(model_manifest, request);
    }
}

fn stageResidencyReportForRequest(
    model_manifest: *const ModelManifest,
    request: StageLoadRequest,
) StagePlanError!manifest_mod.StageResidencyReport {
    return model_manifest.stageResidencyReport(request.roles.toResidencyRequest(request.range())) catch |err| switch (err) {
        error.InvalidLayerRange => error.InvalidStageRange,
    };
}

fn appendTiedLmHeadDependency(
    dependencies: *std.ArrayListUnmanaged(StageDependency),
    allocator: Allocator,
    request: StagePlanRequest,
    stages: []const StagePlanStage,
) StagePlanError!void {
    if (!tieWordEmbeddings(request)) return;
    const model_manifest = request.manifest orelse return error.MissingManifest;
    if (hasIndependentLmHead(model_manifest)) return;
    const token_owner = findRoleOwner(stages, .token_embeddings) orelse return;
    const lm_head_owner = findRoleOwner(stages, .lm_head) orelse return;
    if (token_owner == lm_head_owner) return;
    try appendUniqueDependency(dependencies, allocator, .{
        .source_stage_id = token_owner,
        .target_stage_id = lm_head_owner,
        .role = .token_embeddings,
        .reason = .tied_lm_head,
        .affects_loader_residency = true,
    });
}

fn appendDependencyOverrides(
    dependencies: *std.ArrayListUnmanaged(StageDependency),
    allocator: Allocator,
    request: StagePlanRequest,
    stages: []const StagePlanStage,
) StagePlanError!void {
    for (request.dependency_overrides) |dependency| {
        try appendDependencyOverride(dependencies, allocator, dependency, stages);
    }
    if (request.partition_constraints) |constraints| {
        for (constraints.dependency_overrides) |dependency| {
            try appendDependencyOverride(dependencies, allocator, dependency, stages);
        }
    }
}

fn appendDependencyOverride(
    dependencies: *std.ArrayListUnmanaged(StageDependency),
    allocator: Allocator,
    dependency: DependencyOverride,
    stages: []const StagePlanStage,
) StagePlanError!void {
    if (dependency.role) |role| {
        if (!stages[dependency.source_stage_id].owned_roles[@intFromEnum(role)]) return error.InvalidDependency;
    } else if (dependency.affects_loader_residency) {
        return error.InvalidDependency;
    }
    try appendUniqueDependency(dependencies, allocator, .{
        .source_stage_id = dependency.source_stage_id,
        .target_stage_id = dependency.target_stage_id,
        .role = dependency.role,
        .reason = dependency.reason,
        .affects_loader_residency = dependency.affects_loader_residency,
    });
}

fn appendUniqueDependency(
    dependencies: *std.ArrayListUnmanaged(StageDependency),
    allocator: Allocator,
    dependency: StageDependency,
) StagePlanError!void {
    for (dependencies.items) |existing| {
        if (dependencyIdentityEql(existing, dependency)) return error.DuplicateDependency;
    }
    try dependencies.append(allocator, dependency);
}

fn dependencyIdentityEql(lhs: StageDependency, rhs: StageDependency) bool {
    return lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        optionalRoleEql(lhs.role, rhs.role);
}

fn findRoleOwner(stages: []const StagePlanStage, role: TensorRole) ?usize {
    for (stages) |stage_entry| {
        if (stage_entry.owned_roles[@intFromEnum(role)]) return stage_entry.id;
    }
    return null;
}

fn setRoleRequest(roles: *StageRoleRequest, role: TensorRole, value: bool) StagePlanError!void {
    switch (role) {
        .token_embeddings => roles.include_token_embeddings = value,
        .final_norm => roles.include_final_norm = value,
        .lm_head => roles.include_lm_head = value,
        .embedding_side => roles.include_embedding_side = value,
        .vision_side => roles.include_vision_side = value,
        .architecture_side => roles.include_architecture_side = value,
        .unclassified_global => roles.include_unclassified_global = value,
        .decoder_layer, .quant_companion => return error.InvalidDependency,
    }
}

const StagePlanIdInputs = struct {
    stage_contract_version: u32,
    graph_identity: GraphIdentity,
    split_points: []const usize,
    n_layers: usize,
    stages: []const StagePlanStage,
    boundaries: []const StageBoundary,
    dependencies: []const StageDependency,
    diagnostics: []const StagePlanDiagnostic,
};

fn computeStagePlanId(inputs: StagePlanIdInputs) StagePlanId {
    var encoder = HashEncoder.init();
    encoder.writeString("talu.stage_plan");
    encoder.writeU32(inputs.stage_contract_version);
    writeGraphIdentity(&encoder, inputs.graph_identity);
    encoder.writeUsize(inputs.n_layers);
    writeUsizeSlice(&encoder, inputs.split_points);
    writeStagePlanStages(&encoder, inputs.stages);
    writeStageBoundaries(&encoder, inputs.boundaries);
    writeStageDependenciesCanonical(&encoder, inputs.dependencies);
    writeStagePlanDiagnostics(&encoder, inputs.diagnostics);
    return .{ .digest = encoder.finish() };
}

fn writeGraphIdentity(encoder: *HashEncoder, identity: GraphIdentity) void {
    encoder.writeString("GraphIdentity");
    encoder.writeU32(identity.graph_contract_version);
    encoder.writeU32(identity.stage_contract_version);
    encoder.writeString(identity.architecture_id);
    encoder.writeBytes(&identity.digest);
}

fn writeStagePlanStages(encoder: *HashEncoder, stages: []const StagePlanStage) void {
    encoder.writeUsize(stages.len);
    for (stages) |stage_entry| {
        encoder.writeUsize(stage_entry.id);
        encoder.writeUsize(stage_entry.layer_start);
        encoder.writeUsize(stage_entry.layer_end);
        for (stage_entry.owned_roles) |owned| encoder.writeBool(owned);
        writeStageResidencyReport(encoder, stage_entry.residency);
    }
}

fn writeStageResidencyReport(encoder: *HashEncoder, residency: manifest_mod.StageResidencyReport) void {
    encoder.writeUsize(residency.layer_start);
    encoder.writeUsize(residency.layer_end);
    encoder.writeUsize(residency.total_checkpoint_bytes);
    for (residency.role_bytes) |bytes| encoder.writeUsize(bytes);
}

fn writeStageBoundaries(encoder: *HashEncoder, boundaries: []const StageBoundary) void {
    encoder.writeUsize(boundaries.len);
    for (boundaries) |boundary| {
        encoder.writeUsize(boundary.source_stage_id);
        encoder.writeUsize(boundary.target_stage_id);
        encoder.writeUsize(boundary.producer_layer_start);
        encoder.writeUsize(boundary.producer_layer_end);
        encoder.writeUsize(boundary.consumer_layer_start);
        encoder.writeUsize(boundary.consumer_layer_end);
    }
}

fn writeStageDependenciesCanonical(encoder: *HashEncoder, dependencies: []const StageDependency) void {
    encoder.writeUsize(dependencies.len);
    var previous: ?StageDependency = null;
    var written: usize = 0;
    while (written < dependencies.len) : (written += 1) {
        var selected: ?StageDependency = null;
        for (dependencies) |dependency| {
            if (previous) |prev| {
                if (!dependencyLess(prev, dependency)) continue;
            }
            if (selected == null or dependencyLess(dependency, selected.?)) {
                selected = dependency;
            }
        }
        const dependency = selected orelse return;
        writeStageDependency(encoder, dependency);
        previous = dependency;
    }
}

fn writeStageDependency(encoder: *HashEncoder, dependency: StageDependency) void {
    encoder.writeUsize(dependency.source_stage_id);
    encoder.writeUsize(dependency.target_stage_id);
    encoder.writeOptionalRole(dependency.role);
    encoder.writeU8(@intFromEnum(dependency.reason));
    encoder.writeBool(dependency.affects_loader_residency);
}

fn dependencyLess(lhs: StageDependency, rhs: StageDependency) bool {
    if (lhs.source_stage_id != rhs.source_stage_id) return lhs.source_stage_id < rhs.source_stage_id;
    if (lhs.target_stage_id != rhs.target_stage_id) return lhs.target_stage_id < rhs.target_stage_id;
    if (!optionalRoleEql(lhs.role, rhs.role)) return optionalRoleLess(lhs.role, rhs.role);
    if (@intFromEnum(lhs.reason) != @intFromEnum(rhs.reason)) {
        return @intFromEnum(lhs.reason) < @intFromEnum(rhs.reason);
    }
    if (lhs.affects_loader_residency != rhs.affects_loader_residency) {
        return !lhs.affects_loader_residency and rhs.affects_loader_residency;
    }
    return false;
}

fn writeStagePlanDiagnostics(encoder: *HashEncoder, diagnostics: []const StagePlanDiagnostic) void {
    encoder.writeUsize(diagnostics.len);
    for (diagnostics) |diagnostic| {
        encoder.writeU8(@intFromEnum(diagnostic.kind));
        encoder.writeU8(@intFromEnum(diagnostic.role));
        encoder.writeUsize(diagnostic.stage_id);
    }
}

const HashEncoder = struct {
    hasher: Sha256,

    fn init() HashEncoder {
        return .{ .hasher = Sha256.init(.{}) };
    }

    fn finish(self: *HashEncoder) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);
        return digest;
    }

    fn writeBytes(self: *HashEncoder, bytes: []const u8) void {
        self.hasher.update(bytes);
    }

    fn writeString(self: *HashEncoder, value: []const u8) void {
        self.writeU64(value.len);
        self.writeBytes(value);
    }

    fn writeOptionalString(self: *HashEncoder, value: ?[]const u8) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeString(payload);
    }

    fn writeBool(self: *HashEncoder, value: bool) void {
        self.writeU8(@intFromBool(value));
    }

    fn writeU8(self: *HashEncoder, value: u8) void {
        self.writeBytes(&.{value});
    }

    fn writeU16(self: *HashEncoder, value: u16) void {
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU32(self: *HashEncoder, value: u32) void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeI32(self: *HashEncoder, value: i32) void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(i32, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU64(self: *HashEncoder, value: u64) void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeUsize(self: *HashEncoder, value: usize) void {
        self.writeU64(@intCast(value));
    }

    fn writeF32(self: *HashEncoder, value: f32) void {
        self.writeU32(@bitCast(value));
    }

    fn writeOptionalUsize(self: *HashEncoder, value: ?usize) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeUsize(payload);
    }

    fn writeOptionalI32(self: *HashEncoder, value: ?i32) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeI32(payload);
    }

    fn writeOptionalRole(self: *HashEncoder, value: ?TensorRole) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU8(@intFromEnum(payload));
    }

    fn writeEnumTag(self: *HashEncoder, value: anytype) void {
        self.writeString(@tagName(value));
    }
};

fn writeLoadSemantics(encoder: *HashEncoder, semantics: LoadSemantics) void {
    encoder.writeString("LoadSemantics");
    encoder.writeBool(semantics.preserve_native_norm_dtype);
    encoder.writeBool(semantics.dequantize_mxfp8_to_bf16);
    encoder.writeBool(semantics.dequantize_nvfp4_to_bf16);
}

fn writeModelConfig(encoder: *HashEncoder, config: *const ModelConfig) void {
    encoder.writeString("ModelConfig");
    encoder.writeI32(config.vocab_size);
    encoder.writeI32(config.d_model);
    encoder.writeI32(config.n_layers);
    encoder.writeI32(config.n_heads);
    encoder.writeI32(config.n_kv_groups);
    encoder.writeI32(config.d_ff);
    encoder.writeI32(config.max_seq_len);
    encoder.writeI32(config.head_dim);
    encoder.writeI32(config.global_head_dim);
    encoder.writeI32(config.rope_dim);
    encoder.writeF32(config.rope_theta);
    encoder.writeF32(config.norm_eps);
    encoder.writeI32(config.gaffine_group_size);
    encoder.writeI32(config.gaffine_bits);
    encoder.writeBool(config.tie_word_embeddings);
    encoder.writeI32(config.num_experts);
    encoder.writeI32(config.experts_per_token);
    encoder.writeBool(config.attention_bias);
    encoder.writeEnumTag(config.quant_method);
    writeRopeScaling(encoder, config.rope_scaling);
    encoder.writeEnumTag(config.model_arch);
    encoder.writeBool(config.use_gelu);
    encoder.writeBool(config.use_qk_norm);
    encoder.writeF32(config.query_pre_attn_scalar);
    encoder.writeF32(config.rope_local_theta);
    encoder.writeI32(config.sliding_window);
    encoder.writeI32(config.sliding_window_pattern);
    encoder.writeF32(config.embedding_multiplier);
    encoder.writeF32(config.attention_multiplier);
    encoder.writeF32(config.residual_multiplier);
    encoder.writeF32(config.logits_scaling);
    encoder.writeF32(config.final_logit_softcapping);
    encoder.writeI32(config.hidden_size_per_layer_input);
    encoder.writeI32(config.vocab_size_per_layer_input);
    encoder.writeI32(config.num_kv_shared_layers);
    encoder.writeBool(config.attention_k_eq_v);
    encoder.writeBool(config.use_raw_rms_norm);
    encoder.writeBool(config.use_v_norm);
    encoder.writeOptionalI32(config.bos_token_id);
    encoder.writeI32(config.mamba_d_state);
    encoder.writeI32(config.mamba_d_conv);
    encoder.writeI32(config.mamba_n_heads);
    encoder.writeI32(config.mamba_d_head);
    encoder.writeI32(config.mamba_n_groups);
    encoder.writeI32(config.mamba_expand);
    encoder.writeI32(config.shortconv_d_conv);
    encoder.writeI32(config.shortconv_conv_dim);
    encoder.writeI32(config.shortconv_conv_dim_out);
    encoder.writeBool(config.shortconv_has_bias);
    encoder.writeI32(config.linear_num_key_heads);
    encoder.writeI32(config.linear_num_value_heads);
    encoder.writeI32(config.linear_key_head_dim);
    encoder.writeI32(config.linear_value_head_dim);
    encoder.writeI32(config.vision_hidden_size);
    encoder.writeI32(config.vision_depth);
    encoder.writeI32(config.vision_num_heads);
    encoder.writeI32(config.vision_intermediate_size);
    encoder.writeI32(config.projector_hidden_size);
    encoder.writeI32(config.vision_out_hidden_size);
    encoder.writeI32(config.vision_patch_size);
    encoder.writeI32(config.vision_spatial_merge_size);
    encoder.writeI32(config.vision_temporal_patch_size);
    encoder.writeI32(config.vision_num_position_embeddings);
    encoder.writeI32(config.vision_max_num_patches);
    encoder.writeI32(config.image_token_id);
    encoder.writeI32(config.vision_start_token_id);
    encoder.writeI32(config.vision_end_token_id);
    encoder.writeU8(config.vision_probe_layer_count);
    for (config.vision_probe_layers) |layer| encoder.writeU16(layer);
    encoder.writeBool(config.flash_attn_compatible);
    writeOptionalU8Slice(encoder, config.layer_types);
}

fn writeRopeScaling(encoder: *HashEncoder, rope_scaling: RopeScaling) void {
    encoder.writeString("RopeScaling");
    encoder.writeEnumTag(rope_scaling.rope_type);
    encoder.writeF32(rope_scaling.factor);
    encoder.writeF32(rope_scaling.low_freq_factor);
    encoder.writeF32(rope_scaling.high_freq_factor);
    encoder.writeF32(rope_scaling.beta_slow);
    encoder.writeF32(rope_scaling.beta_fast);
    encoder.writeF32(rope_scaling.attention_factor);
    encoder.writeF32(rope_scaling.mscale);
    encoder.writeF32(rope_scaling.mscale_all_dim);
    encoder.writeBool(rope_scaling.truncate);
    encoder.writeI32(rope_scaling.original_max_position_embeddings);
    for (rope_scaling.mrope_section) |section| encoder.writeU32(section);
    encoder.writeBool(rope_scaling.mrope_interleaved);
}

fn writeArchitecture(
    encoder: *HashEncoder,
    allocator: Allocator,
    architecture_id: []const u8,
    architecture: *const Architecture,
) StagePlanError!void {
    encoder.writeString("Architecture");
    encoder.writeString(architecture_id);
    encoder.writeString(architecture.name);
    writeStringSlice(encoder, architecture.model_types);
    writeKernelMeta(encoder, architecture.kernel_meta);
    try writeStateDescriptors(encoder, allocator, architecture.state_descriptors);
    writeStateDescriptor(encoder, architecture.unknown_state_descriptor);
    writeOptionalBlockVariants(encoder, architecture.block_variants);
    writeOptionalU8Slice(encoder, architecture.layer_map);
    try writeVariantAliases(encoder, allocator, architecture.variant_aliases);
    writeWeightSpecs(encoder, architecture.block_weights);
    writeWeightSpecs(encoder, architecture.global_weights);
    writeStringSlice(encoder, architecture.weight_prefixes);
    writeStringSlice(encoder, architecture.d_ff_source_weight_ids);
    encoder.writeBool(architecture.resolve_d_ff_from_weights);
    encoder.writeOptionalString(architecture.shortconv_dims_source_weight_id);
    encoder.writeBool(architecture.resolve_shortconv_dims_from_weights);
    writeStringSlice(encoder, architecture.weight_dtype_source_weight_ids);
    encoder.writeBool(architecture.enable_loader_fusions);
    encoder.writeBool(architecture.has_qk_norm);
    encoder.writeBool(architecture.has_moe);
    encoder.writeBool(architecture.has_mamba);
    encoder.writeBool(architecture.has_gated_delta);
    encoder.writeBool(architecture.has_shortconv);
    encoder.writeBool(architecture.has_mla);
    encoder.writeBool(architecture.has_fused_qkv);
    encoder.writeBool(architecture.has_fused_gate_up);
    encoder.writeU8(architecture.num_norms_per_block);
    encoder.writeBool(architecture.use_gelu);
    encoder.writeBool(architecture.use_swiglu_oss);
    encoder.writeF32(architecture.norm_weight_offset);
    encoder.writeBool(architecture.explicit_qk_norm_ops);
    encoder.writeBool(architecture.norm_weights_pre_shifted);
    encoder.writeF32(architecture.embedding_multiplier);
    writeVisionMetadata(encoder, architecture.vision);
}

fn writeKernelMeta(encoder: *HashEncoder, meta: op_types.KernelMeta) void {
    encoder.writeString("KernelMeta");
    encoder.writeBool(meta.is_causal);
    encoder.writeBool(meta.attention_config.rope_interleaved != null);
    if (meta.attention_config.rope_interleaved) |interleaved| encoder.writeBool(interleaved);
    encoder.writeBool(meta.attention_config.query_gate);
    encoder.writeBool(meta.mla_config != null);
    if (meta.mla_config) |config| {
        encoder.writeU32(config.q_lora_rank);
        encoder.writeU32(config.kv_lora_rank);
        encoder.writeU32(config.qk_head_dim);
        encoder.writeU32(config.qk_rope_head_dim);
        encoder.writeU32(config.qk_nope_head_dim);
        encoder.writeU32(config.v_head_dim);
        encoder.writeBool(config.rope_interleave);
    }
    encoder.writeBool(meta.mamba_config != null);
    if (meta.mamba_config) |config| {
        encoder.writeU32(config.d_state);
        encoder.writeU32(config.d_conv);
        encoder.writeU32(config.n_heads);
        encoder.writeU32(config.d_head);
        encoder.writeU32(config.n_groups);
        encoder.writeU32(config.d_inner);
    }
    encoder.writeBool(meta.gated_delta_config != null);
    if (meta.gated_delta_config) |config| {
        encoder.writeU32(config.d_conv);
        encoder.writeU32(config.n_heads);
        encoder.writeU32(config.d_head);
        encoder.writeU32(config.d_inner);
    }
    encoder.writeBool(meta.shortconv_config != null);
    if (meta.shortconv_config) |config| {
        encoder.writeU32(config.d_conv);
        encoder.writeU32(config.conv_dim);
        encoder.writeU32(config.conv_dim_out);
        encoder.writeBool(config.has_bias);
    }
}

fn writeStateDescriptors(
    encoder: *HashEncoder,
    allocator: Allocator,
    descriptors: []const op_types.StateDescriptorSpec,
) StagePlanError!void {
    const order = try allocator.alloc(usize, descriptors.len);
    defer allocator.free(order);
    for (order, 0..) |*slot, index| slot.* = index;
    std.mem.sort(usize, order, descriptors, struct {
        fn less(specs: []const op_types.StateDescriptorSpec, lhs: usize, rhs: usize) bool {
            return stateDescriptorLess(specs[lhs], specs[rhs]);
        }
    }.less);
    encoder.writeUsize(descriptors.len);
    for (order) |index| writeStateDescriptor(encoder, descriptors[index]);
}

fn stateDescriptorLess(lhs: op_types.StateDescriptorSpec, rhs: op_types.StateDescriptorSpec) bool {
    if (lhs.id != rhs.id) return lhs.id < rhs.id;
    if (lhs.size_bytes != rhs.size_bytes) return lhs.size_bytes < rhs.size_bytes;
    if (lhs.align_bytes != rhs.align_bytes) return lhs.align_bytes < rhs.align_bytes;
    if (lhs.zero_init != rhs.zero_init) return !lhs.zero_init and rhs.zero_init;
    if (@intFromEnum(lhs.lifecycle) != @intFromEnum(rhs.lifecycle)) return @intFromEnum(lhs.lifecycle) < @intFromEnum(rhs.lifecycle);
    if (lhs.runtime_kind != rhs.runtime_kind) return lhs.runtime_kind < rhs.runtime_kind;
    return false;
}

fn writeStateDescriptor(encoder: *HashEncoder, descriptor: op_types.StateDescriptorSpec) void {
    encoder.writeU8(descriptor.id);
    encoder.writeU64(descriptor.size_bytes);
    encoder.writeU16(descriptor.align_bytes);
    encoder.writeBool(descriptor.zero_init);
    encoder.writeEnumTag(descriptor.lifecycle);
    encoder.writeU8(descriptor.runtime_kind);
}

fn writeOptionalBlockVariants(
    encoder: *HashEncoder,
    variants_opt: ?[]op_types.BlockVariant,
) void {
    encoder.writeBool(variants_opt != null);
    if (variants_opt) |variants| {
        encoder.writeUsize(variants.len);
        for (variants) |variant| {
            encoder.writeString(variant.name);
            encoder.writeBool(variant.meta != null);
            if (variant.meta) |meta| writeKernelMeta(encoder, meta);
            writeWeightSpecs(encoder, variant.weights);
        }
    }
}

fn writeVariantAliases(
    encoder: *HashEncoder,
    allocator: Allocator,
    aliases_opt: ?[]const op_types.VariantAlias,
) StagePlanError!void {
    encoder.writeBool(aliases_opt != null);
    const aliases = aliases_opt orelse return;
    const order = try allocator.alloc(usize, aliases.len);
    defer allocator.free(order);
    for (order, 0..) |*slot, index| slot.* = index;
    std.mem.sort(usize, order, aliases, struct {
        fn less(items: []const op_types.VariantAlias, lhs: usize, rhs: usize) bool {
            const alias_order = std.mem.order(u8, items[lhs].alias, items[rhs].alias);
            if (alias_order != .eq) return alias_order == .lt;
            return items[lhs].variant_index < items[rhs].variant_index;
        }
    }.less);
    encoder.writeUsize(aliases.len);
    for (order) |index| {
        encoder.writeString(aliases[index].alias);
        encoder.writeU8(aliases[index].variant_index);
    }
}

fn writeWeightSpecs(encoder: *HashEncoder, specs: []const op_types.WeightSpec) void {
    encoder.writeUsize(specs.len);
    for (specs) |spec| {
        encoder.writeString(spec.id);
        encoder.writeString(spec.suffix);
        writeStringSlice(encoder, spec.aliases);
        encoder.writeString(spec.module_type);
        encoder.writeEnumTag(spec.layout);
        encoder.writeString(spec.dtype);
        encoder.writeBool(spec.required);
        encoder.writeBool(spec.expected_shape != null);
        if (spec.expected_shape) |shape| writeUsizeSlice(encoder, shape);
        encoder.writeUsize(spec.transforms.len);
        for (spec.transforms) |transform| encoder.writeEnumTag(transform);
        encoder.writeBool(spec.force_f32);
    }
}

fn writeVisionMetadata(encoder: *HashEncoder, metadata: op_types.VisionMetadata) void {
    writeStringSlice(encoder, metadata.fused_qkv_probe_candidates);
    writeStringSlice(encoder, metadata.split_qkv_probe_candidates);
    writeStringSlice(encoder, metadata.patch_embed_candidates);
    writeStringSlice(encoder, metadata.patch_embed_bias_candidates);
    writeStringSlice(encoder, metadata.position_embed_candidates);
    writeStringSlice(encoder, metadata.post_norm_weight_candidates);
    writeStringSlice(encoder, metadata.post_norm_bias_candidates);
    writeStringSlice(encoder, metadata.merger_norm_weight_candidates);
    writeStringSlice(encoder, metadata.merger_norm_bias_candidates);
    writeStringSlice(encoder, metadata.merger_fc1_candidates);
    writeStringSlice(encoder, metadata.merger_fc1_bias_candidates);
    writeStringSlice(encoder, metadata.merger_fc2_candidates);
    writeStringSlice(encoder, metadata.merger_fc2_bias_candidates);
    writeStringSlice(encoder, metadata.ln1_weight_templates);
    writeStringSlice(encoder, metadata.ln1_bias_templates);
    writeStringSlice(encoder, metadata.ln2_weight_templates);
    writeStringSlice(encoder, metadata.ln2_bias_templates);
    writeStringSlice(encoder, metadata.fused_qkv_weight_templates);
    writeStringSlice(encoder, metadata.fused_qkv_bias_templates);
    writeStringSlice(encoder, metadata.split_q_weight_templates);
    writeStringSlice(encoder, metadata.split_q_bias_templates);
    writeStringSlice(encoder, metadata.split_k_weight_templates);
    writeStringSlice(encoder, metadata.split_k_bias_templates);
    writeStringSlice(encoder, metadata.split_v_weight_templates);
    writeStringSlice(encoder, metadata.split_v_bias_templates);
    writeStringSlice(encoder, metadata.out_proj_weight_templates);
    writeStringSlice(encoder, metadata.out_proj_bias_templates);
    writeStringSlice(encoder, metadata.fc1_weight_templates);
    writeStringSlice(encoder, metadata.fc1_bias_templates);
    writeStringSlice(encoder, metadata.fc2_weight_templates);
    writeStringSlice(encoder, metadata.fc2_bias_templates);
    writeStringSlice(encoder, metadata.deepstack_norm_weight_templates);
    writeStringSlice(encoder, metadata.deepstack_norm_bias_templates);
    writeStringSlice(encoder, metadata.deepstack_fc1_weight_templates);
    writeStringSlice(encoder, metadata.deepstack_fc1_bias_templates);
    writeStringSlice(encoder, metadata.deepstack_fc2_weight_templates);
    writeStringSlice(encoder, metadata.deepstack_fc2_bias_templates);
    writeStringSlice(encoder, metadata.depth_split_qproj_templates);
    writeStringSlice(encoder, metadata.depth_fused_qkv_templates);
    writeStringSlice(encoder, metadata.intermediate_fc1_templates);
}

fn writeManifest(
    encoder: *HashEncoder,
    allocator: Allocator,
    model_manifest: *const ModelManifest,
) StagePlanError!void {
    encoder.writeString("ModelManifest");
    encoder.writeString(model_manifest.architecture_id);
    encoder.writeUsize(model_manifest.layer_count);

    const order = try allocator.alloc(usize, model_manifest.entries.len);
    defer allocator.free(order);
    for (order, 0..) |*slot, index| slot.* = index;
    std.mem.sort(usize, order, model_manifest.entries, struct {
        fn less(entries: []const TensorManifestEntry, lhs: usize, rhs: usize) bool {
            return manifestEntryLess(entries[lhs], entries[rhs]);
        }
    }.less);

    encoder.writeUsize(model_manifest.entries.len);
    for (order) |entry_index| writeManifestEntry(encoder, model_manifest.entries[entry_index]);
}

fn manifestEntryLess(lhs: TensorManifestEntry, rhs: TensorManifestEntry) bool {
    const name_order = std.mem.order(u8, lhs.name, rhs.name);
    if (name_order != .eq) return name_order == .lt;
    if (@intFromEnum(lhs.dtype) != @intFromEnum(rhs.dtype)) return @intFromEnum(lhs.dtype) < @intFromEnum(rhs.dtype);
    if (!std.mem.eql(usize, lhs.shape, rhs.shape)) return usizeSliceLess(lhs.shape, rhs.shape);
    if (lhs.checkpoint_bytes != rhs.checkpoint_bytes) return lhs.checkpoint_bytes < rhs.checkpoint_bytes;
    if (@intFromEnum(lhs.role) != @intFromEnum(rhs.role)) return @intFromEnum(lhs.role) < @intFromEnum(rhs.role);
    if (lhs.owner_role != rhs.owner_role) return optionalRoleLess(lhs.owner_role, rhs.owner_role);
    if (lhs.layer_index != rhs.layer_index) return optionalUsizeLess(lhs.layer_index, rhs.layer_index);
    if (!optionalStringEql(lhs.weight_id, rhs.weight_id)) return optionalStringLess(lhs.weight_id, rhs.weight_id);
    if (!optionalStringEql(lhs.primary_name, rhs.primary_name)) return optionalStringLess(lhs.primary_name, rhs.primary_name);
    if (@intFromEnum(lhs.status) != @intFromEnum(rhs.status)) return @intFromEnum(lhs.status) < @intFromEnum(rhs.status);
    return false;
}

fn usizeSliceLess(lhs: []const usize, rhs: []const usize) bool {
    const shared_len = @min(lhs.len, rhs.len);
    for (lhs[0..shared_len], rhs[0..shared_len]) |lhs_value, rhs_value| {
        if (lhs_value != rhs_value) return lhs_value < rhs_value;
    }
    return lhs.len < rhs.len;
}

fn optionalUsizeLess(lhs: ?usize, rhs: ?usize) bool {
    if (lhs == null and rhs == null) return false;
    if (lhs == null) return true;
    if (rhs == null) return false;
    return lhs.? < rhs.?;
}

fn optionalRoleLess(lhs: ?TensorRole, rhs: ?TensorRole) bool {
    if (lhs == null and rhs == null) return false;
    if (lhs == null) return true;
    if (rhs == null) return false;
    return @intFromEnum(lhs.?) < @intFromEnum(rhs.?);
}

fn optionalStringEql(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

fn optionalStringLess(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null and rhs == null) return false;
    if (lhs == null) return true;
    if (rhs == null) return false;
    return std.mem.lessThan(u8, lhs.?, rhs.?);
}

fn writeManifestEntry(encoder: *HashEncoder, entry: TensorManifestEntry) void {
    encoder.writeString(entry.name);
    encoder.writeU8(@intFromEnum(entry.dtype));
    writeUsizeSlice(encoder, entry.shape);
    encoder.writeUsize(entry.checkpoint_bytes);
    encoder.writeU8(@intFromEnum(entry.role));
    encoder.writeOptionalRole(entry.owner_role);
    encoder.writeOptionalUsize(entry.layer_index);
    encoder.writeOptionalString(entry.weight_id);
    encoder.writeOptionalString(entry.primary_name);
    encoder.writeU8(@intFromEnum(entry.status));
}

fn writeStringSlice(encoder: *HashEncoder, values: []const []const u8) void {
    encoder.writeUsize(values.len);
    for (values) |value| encoder.writeString(value);
}

fn writeUsizeSlice(encoder: *HashEncoder, values: []const usize) void {
    encoder.writeUsize(values.len);
    for (values) |value| encoder.writeUsize(value);
}

fn writeOptionalU8Slice(encoder: *HashEncoder, values_opt: ?[]const u8) void {
    encoder.writeBool(values_opt != null);
    if (values_opt) |values| {
        encoder.writeUsize(values.len);
        for (values) |value| encoder.writeU8(value);
    }
}

fn minimalConfig(n_layers: usize) ModelConfig {
    return .{
        .vocab_size = 8,
        .d_model = 4,
        .n_layers = @intCast(n_layers),
        .n_heads = 1,
        .n_kv_groups = 1,
        .d_ff = 16,
        .max_seq_len = 32,
        .head_dim = 4,
        .rope_theta = 10000.0,
        .norm_eps = 0.00001,
        .gaffine_group_size = 32,
    };
}

pub const testing = struct {
    pub fn runContractTests(allocator: Allocator) !void {
        const semantics = LoadSemantics.fromLoadOptions(.{
            .preserve_native_norm_dtype = true,
            .dequantize_mxfp8_to_bf16 = true,
            .dequantize_nvfp4_to_bf16 = false,
        });
        try std.testing.expect(semantics.preserve_native_norm_dtype);
        try std.testing.expect(semantics.dequantize_mxfp8_to_bf16);
        try std.testing.expect(!semantics.dequantize_nvfp4_to_bf16);
        try runGraphIdentityContract(allocator);
        try runGraphIdentityChangeContract(allocator);
        try runStagePlanningContract(allocator);
        try runStagePlanIdentityContract(allocator);
        try runStageShapeContract(allocator);
        try runFailureContract(allocator);
        try runValidationFailureContract(allocator);
        try runValidateStagePlanContract(allocator);
        try runRoleOwnershipContract(allocator);
        try runSideDomainDependencyContract(allocator);
        try runResidencyContract(allocator);
    }
};

const TestEntrySpec = struct {
    name: []const u8,
    dtype: DType = .f16,
    shape: []const usize = &.{ 4, 4 },
    checkpoint_bytes: usize = 32,
    role: TensorRole,
    owner_role: ?TensorRole = null,
    layer_index: ?usize = null,
    weight_id: ?[]const u8 = null,
    primary_name: ?[]const u8 = null,
    status: manifest_mod.ClassificationStatus = .architecture_weight,
};

fn testArch() Architecture {
    return .{
        .name = "stage_plan_test",
        .model_types = &.{"stage_plan_test"},
        .block_weights = &.{.{
            .id = "self_attn.q_proj.weight",
            .suffix = "self_attn.q_proj.weight",
            .module_type = "Linear",
            .layout = .linear,
            .dtype = "F16",
            .required = true,
        }},
        .global_weights = &.{
            .{
                .id = "token_embeddings",
                .suffix = "model.embed_tokens.weight",
                .module_type = "Embedding",
                .layout = .embedding,
                .dtype = "F16",
                .required = true,
            },
            .{
                .id = "ln_final",
                .suffix = "model.norm.weight",
                .module_type = "RMSNorm",
                .layout = .none,
                .dtype = "F32",
                .required = false,
            },
            .{
                .id = "lm_head",
                .suffix = "lm_head.weight",
                .module_type = "Linear",
                .layout = .linear,
                .dtype = "F16",
                .required = false,
            },
        },
        .weight_prefixes = &.{"model.layers.{d}."},
        .weight_dtype_source_weight_ids = &.{"self_attn.q_proj.weight"},
    };
}

const default_test_arch = testArch();

fn testConfig(n_layers: usize) ModelConfig {
    var config = minimalConfig(n_layers);
    config.tie_word_embeddings = false;
    return config;
}

fn testManifest(allocator: Allocator, layer_count: usize, specs: []const TestEntrySpec) !ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const entries = try arena_allocator.alloc(TensorManifestEntry, specs.len);
    var role_bytes = [_]usize{0} ** manifest_mod.role_count;
    var total_checkpoint_bytes: usize = 0;
    for (specs, 0..) |spec, index| {
        entries[index] = .{
            .name = try arena_allocator.dupe(u8, spec.name),
            .dtype = spec.dtype,
            .shape = try arena_allocator.dupe(usize, spec.shape),
            .checkpoint_bytes = spec.checkpoint_bytes,
            .role = spec.role,
            .owner_role = spec.owner_role,
            .layer_index = spec.layer_index,
            .weight_id = if (spec.weight_id) |value| try arena_allocator.dupe(u8, value) else null,
            .primary_name = if (spec.primary_name) |value| try arena_allocator.dupe(u8, value) else null,
            .status = spec.status,
        };
        total_checkpoint_bytes += spec.checkpoint_bytes;
        role_bytes[@intFromEnum(spec.role)] += spec.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = try arena_allocator.dupe(u8, "stage_plan_test"),
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_checkpoint_bytes,
        .role_bytes = role_bytes,
    };
}

fn standardManifest(allocator: Allocator, layer_count: usize) !ModelManifest {
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.2.self_attn.q_proj.weight", .checkpoint_bytes = 30, .role = .decoder_layer, .layer_index = 2, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.3.self_attn.q_proj.weight", .checkpoint_bytes = 40, .role = .decoder_layer, .layer_index = 3, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.3.self_attn.q_proj.weight_scale", .checkpoint_bytes = 4, .role = .quant_companion, .owner_role = .decoder_layer, .layer_index = 3, .weight_id = "self_attn.q_proj.weight", .primary_name = "model.layers.3.self_attn.q_proj.weight", .status = .quant_companion },
        .{ .name = "model.norm.weight", .dtype = .f32, .shape = &.{4}, .checkpoint_bytes = 16, .role = .final_norm, .weight_id = "ln_final" },
        .{ .name = "lm_head.weight", .checkpoint_bytes = 100, .role = .lm_head, .weight_id = "lm_head" },
        .{ .name = "model.embed_positions.weight", .checkpoint_bytes = 12, .role = .embedding_side, .weight_id = "position_embeddings" },
    };
    return testManifest(allocator, layer_count, &specs);
}

fn tiedManifestWithoutIndependentLmHead(allocator: Allocator, layer_count: usize) !ModelManifest {
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.2.self_attn.q_proj.weight", .checkpoint_bytes = 30, .role = .decoder_layer, .layer_index = 2, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.3.self_attn.q_proj.weight", .checkpoint_bytes = 40, .role = .decoder_layer, .layer_index = 3, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.norm.weight", .dtype = .f32, .shape = &.{4}, .checkpoint_bytes = 16, .role = .final_norm, .weight_id = "ln_final" },
    };
    return testManifest(allocator, layer_count, &specs);
}

fn planRequest(
    n_layers: usize,
    splits: []const usize,
    config: *const ModelConfig,
    model_manifest: *const ModelManifest,
) StagePlanRequest {
    return .{
        .n_layers = n_layers,
        .split_points = splits,
        .architecture = &default_test_arch,
        .model_config = config,
        .manifest = model_manifest,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
}

fn runGraphIdentityContract(allocator: Allocator) !void {
    var arch = testArch();
    var config = testConfig(2);
    const specs_a = [_]TestEntrySpec{
        .{ .name = "b.weight", .role = .decoder_layer, .layer_index = 1, .weight_id = "b" },
        .{ .name = "a.weight", .role = .token_embeddings, .weight_id = "token_embeddings" },
    };
    const specs_b = [_]TestEntrySpec{
        .{ .name = "a.weight", .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "b.weight", .role = .decoder_layer, .layer_index = 1, .weight_id = "b" },
    };
    var manifest_a = try testManifest(allocator, 2, &specs_a);
    defer manifest_a.deinit();
    var manifest_b = try testManifest(allocator, 2, &specs_b);
    defer manifest_b.deinit();

    const identity_a = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest_a,
    });
    const identity_b = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest_b,
    });
    try std.testing.expectEqualSlices(u8, &identity_a.digest, &identity_b.digest);

    const tie_specs_a = [_]TestEntrySpec{
        .{ .name = "same.weight", .checkpoint_bytes = 32, .role = .decoder_layer, .layer_index = 0, .weight_id = "same", .status = .architecture_weight },
        .{ .name = "same.weight", .checkpoint_bytes = 33, .role = .decoder_layer, .layer_index = 0, .weight_id = "same", .status = .unclassified },
    };
    const tie_specs_b = [_]TestEntrySpec{
        .{ .name = "same.weight", .checkpoint_bytes = 33, .role = .decoder_layer, .layer_index = 0, .weight_id = "same", .status = .unclassified },
        .{ .name = "same.weight", .checkpoint_bytes = 32, .role = .decoder_layer, .layer_index = 0, .weight_id = "same", .status = .architecture_weight },
    };
    var tie_manifest_a = try testManifest(allocator, 2, &tie_specs_a);
    defer tie_manifest_a.deinit();
    var tie_manifest_b = try testManifest(allocator, 2, &tie_specs_b);
    defer tie_manifest_b.deinit();
    const tie_identity_a = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &tie_manifest_a,
    });
    const tie_identity_b = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &tie_manifest_b,
    });
    try std.testing.expectEqualSlices(u8, &tie_identity_a.digest, &tie_identity_b.digest);

    const states_a = [_]op_types.StateDescriptorSpec{
        .{ .id = 7, .size_bytes = 10, .align_bytes = 8, .zero_init = false, .lifecycle = .request_scoped },
        .{ .id = 7, .size_bytes = 11, .align_bytes = 8, .zero_init = false, .lifecycle = .request_scoped },
    };
    const states_b = [_]op_types.StateDescriptorSpec{
        .{ .id = 7, .size_bytes = 11, .align_bytes = 8, .zero_init = false, .lifecycle = .request_scoped },
        .{ .id = 7, .size_bytes = 10, .align_bytes = 8, .zero_init = false, .lifecycle = .request_scoped },
    };
    var state_arch_a = arch;
    state_arch_a.state_descriptors = &states_a;
    var state_arch_b = arch;
    state_arch_b.state_descriptors = &states_b;
    const state_identity_a = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &state_arch_a,
        .config = &config,
        .manifest = &manifest_a,
    });
    const state_identity_b = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &state_arch_b,
        .config = &config,
        .manifest = &manifest_a,
    });
    try std.testing.expectEqualSlices(u8, &state_identity_a.digest, &state_identity_b.digest);

    const aliases_a = [_]op_types.VariantAlias{
        .{ .alias = "same", .variant_index = 1 },
        .{ .alias = "same", .variant_index = 0 },
    };
    const aliases_b = [_]op_types.VariantAlias{
        .{ .alias = "same", .variant_index = 0 },
        .{ .alias = "same", .variant_index = 1 },
    };
    var alias_arch_a = arch;
    alias_arch_a.variant_aliases = &aliases_a;
    var alias_arch_b = arch;
    alias_arch_b.variant_aliases = &aliases_b;
    const alias_identity_a = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &alias_arch_a,
        .config = &config,
        .manifest = &manifest_a,
    });
    const alias_identity_b = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &alias_arch_b,
        .config = &config,
        .manifest = &manifest_a,
    });
    try std.testing.expectEqualSlices(u8, &alias_identity_a.digest, &alias_identity_b.digest);

    var changed_config = config;
    changed_config.d_model += 1;
    const changed_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &changed_config,
        .manifest = &manifest_a,
        .load_semantics = .{ .preserve_native_norm_dtype = true },
    });
    try std.testing.expect(!std.mem.eql(u8, &identity_a.digest, &changed_identity.digest));
    try std.testing.expectEqual(model_config_identity_field_count, std.meta.fields(ModelConfig).len);
    try std.testing.expectEqual(rope_scaling_identity_field_count, std.meta.fields(RopeScaling).len);
}

fn runStagePlanningContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    var one_stage_request = planRequest(4, &.{}, &config, &manifest);
    one_stage_request.partition_constraints = null;
    var one_stage_plan = try buildStagePlan(allocator, one_stage_request);
    defer one_stage_plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), one_stage_plan.stages.len);
    try std.testing.expectEqual(@as(usize, 0), one_stage_plan.boundaries.len);

    var derived_request = planRequest(4, &.{ 1, 3 }, &config, &manifest);
    derived_request.partition_constraints = null;
    var plan = try buildStagePlan(allocator, derived_request);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 3), plan.stages.len);
    try std.testing.expectEqual(@as(usize, 2), plan.boundaries.len);
    const middle_stage = try plan.stage(1);
    try std.testing.expectEqual(@as(usize, 1), middle_stage.layer_start);
    try std.testing.expectEqual(@as(usize, 3), middle_stage.layer_end);

    const first_request = try plan.stageLoadRequest(0);
    try std.testing.expect(first_request.roles.include_token_embeddings);
    try std.testing.expect(first_request.roles.include_embedding_side);
    try std.testing.expect(!first_request.roles.include_lm_head);

    const final_request = try plan.stageLoadRequest(2);
    try std.testing.expect(final_request.roles.include_final_norm);
    try std.testing.expect(final_request.roles.include_lm_head);
    try std.testing.expect(!final_request.roles.include_token_embeddings);

    config.tie_word_embeddings = true;
    var tied_plan = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer tied_plan.deinit();
    const tied_final_request = try tied_plan.stageLoadRequest(1);
    try std.testing.expect(tied_final_request.roles.include_lm_head);
    try std.testing.expect(!tied_final_request.roles.include_token_embeddings);
    try std.testing.expectEqual(@as(usize, 0), tied_plan.dependencies.len);
    try std.testing.expect(!tied_plan.stages[1].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
}

fn runStagePlanIdentityContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    var derived_request = planRequest(4, &.{ 1, 3 }, &config, &manifest);
    derived_request.partition_constraints = null;
    var plan = try buildStagePlan(allocator, derived_request);
    defer plan.deinit();

    const expected_identity = try graphIdentity(allocator, .{
        .architecture_id = manifest.architecture_id,
        .architecture = derived_request.architecture.?,
        .config = derived_request.model_config.?,
        .manifest = &manifest,
    });
    try std.testing.expectEqual(stage_plan_contract_version, plan.stage_contract_version);
    try std.testing.expectEqual(graph_identity_contract_version, plan.graph_identity.graph_contract_version);
    try std.testing.expectEqual(stage_plan_contract_version, plan.graph_identity.stage_contract_version);
    try std.testing.expectEqualStrings(manifest.architecture_id, plan.graph_identity.architecture_id);
    try std.testing.expect(graphIdentityEql(expected_identity, plan.graph_identity));
    try assertGraphIdentity(&plan, expected_identity);
    try validateGraphIdentity(plan.graph_identity);
    try validateStagePlan(&plan, .{ .expected_graph_identity = expected_identity, .manifest = &manifest });
    try std.testing.expectEqualSlices(usize, derived_request.split_points, plan.split_points);
    try std.testing.expect(plan.split_points.ptr != derived_request.split_points.ptr);
    try std.testing.expectEqual(PartitionConstraintSource.derived_plain_decoder, plan.partition_constraint_source);
    const zero_digest = [_]u8{0} ** 32;
    try std.testing.expect(!std.mem.eql(u8, &zero_digest, &plan.plan_id.digest));

    const stage2_request = try plan.stageLoadRequest(2);
    const expected_residency = try manifest.stageResidencyReport(stage2_request.roles.toResidencyRequest(stage2_request.range()));
    try std.testing.expect(stageResidencyReportEql(expected_residency, plan.stages[2].residency));

    var explicit_plan = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer explicit_plan.deinit();
    try std.testing.expectEqual(PartitionConstraintSource.explicit, explicit_plan.partition_constraint_source);

    var wrong_identity = expected_identity;
    wrong_identity.digest[0] ^= 0x80;
    try std.testing.expectError(error.GraphIdentityMismatch, assertGraphIdentity(&plan, wrong_identity));
    var mismatched_request = derived_request;
    mismatched_request.graph_identity = wrong_identity;
    try std.testing.expectError(error.GraphIdentityMismatch, buildStagePlan(allocator, mismatched_request));

    const first_dependency_order = [_]DependencyOverride{
        .{ .source_stage_id = 1, .target_stage_id = 2, .reason = .stateful_decoder },
        .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder },
    };
    const second_dependency_order = [_]DependencyOverride{
        .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder },
        .{ .source_stage_id = 1, .target_stage_id = 2, .reason = .stateful_decoder },
    };
    var order_a_request = planRequest(4, &.{ 1, 3 }, &config, &manifest);
    order_a_request.dependency_overrides = &first_dependency_order;
    var order_b_request = planRequest(4, &.{ 1, 3 }, &config, &manifest);
    order_b_request.dependency_overrides = &second_dependency_order;
    var order_a = try buildStagePlan(allocator, order_a_request);
    defer order_a.deinit();
    var order_b = try buildStagePlan(allocator, order_b_request);
    defer order_b.deinit();
    try std.testing.expect(!dependencyIdentityEql(order_a.dependencies[0], order_b.dependencies[0]));
    try std.testing.expect(stagePlanIdEql(order_a.plan_id, order_b.plan_id));

    var standalone_identity_a = expected_identity;
    standalone_identity_a.architecture_id = "stage_plan_precomputed_a";
    standalone_identity_a.digest[0] ^= 0x11;
    var standalone_identity_b = standalone_identity_a;
    standalone_identity_b.architecture_id = "stage_plan_precomputed_b";
    standalone_identity_b.digest[0] ^= 0x22;
    const standalone_a_request = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{ 1, 3 },
        .manifest = &manifest,
        .graph_identity = standalone_identity_a,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
    const standalone_b_request = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{ 1, 3 },
        .manifest = &manifest,
        .graph_identity = standalone_identity_b,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
    var standalone_a = try buildStagePlan(allocator, standalone_a_request);
    defer standalone_a.deinit();
    var standalone_b = try buildStagePlan(allocator, standalone_b_request);
    defer standalone_b.deinit();
    try std.testing.expect(!stagePlanIdEql(standalone_a.plan_id, standalone_b.plan_id));

    var owned_transient_identity: GraphIdentity = undefined;
    {
        var transient_manifest = try standardManifest(allocator, 4);
        defer transient_manifest.deinit();
        const transient_identity = try graphIdentity(allocator, .{
            .architecture_id = transient_manifest.architecture_id,
            .architecture = &default_test_arch,
            .config = &config,
            .manifest = &transient_manifest,
        });
        owned_transient_identity = try dupeGraphIdentity(allocator, transient_identity);
    }
    defer deinitGraphIdentity(allocator, &owned_transient_identity);
    try validateGraphIdentity(owned_transient_identity);
    const owned_identity_request = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{ 1, 3 },
        .manifest = &manifest,
        .graph_identity = owned_transient_identity,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
    var owned_identity_plan = try buildStagePlan(allocator, owned_identity_request);
    defer owned_identity_plan.deinit();
    try std.testing.expect(graphIdentityEql(owned_transient_identity, owned_identity_plan.graph_identity));

    var partial_config_request = standalone_a_request;
    partial_config_request.model_config = &config;
    try std.testing.expectError(error.GraphIdentityMismatch, buildStagePlan(allocator, partial_config_request));

    var partial_architecture_request = standalone_a_request;
    partial_architecture_request.architecture = &default_test_arch;
    try std.testing.expectError(error.GraphIdentityMismatch, buildStagePlan(allocator, partial_architecture_request));

    var invalid_graph_version_request = standalone_a_request;
    invalid_graph_version_request.graph_identity.?.graph_contract_version += 1;
    try std.testing.expectError(error.InvalidContractVersion, buildStagePlan(allocator, invalid_graph_version_request));

    var invalid_stage_version_request = standalone_a_request;
    invalid_stage_version_request.graph_identity.?.stage_contract_version += 1;
    try std.testing.expectError(error.InvalidContractVersion, buildStagePlan(allocator, invalid_stage_version_request));

    var missing_architecture_id_request = standalone_a_request;
    missing_architecture_id_request.graph_identity.?.architecture_id = "";
    try std.testing.expectError(error.MissingGraphIdentity, buildStagePlan(allocator, missing_architecture_id_request));

    const changed_split_request = planRequest(4, &.{2}, &config, &manifest);
    var changed_split = try buildStagePlan(allocator, changed_split_request);
    defer changed_split.deinit();
    try std.testing.expect(!stagePlanIdEql(plan.plan_id, changed_split.plan_id));

    var changed_dependency_request = planRequest(4, &.{ 1, 3 }, &config, &manifest);
    changed_dependency_request.dependency_overrides = &.{.{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder }};
    var changed_dependency = try buildStagePlan(allocator, changed_dependency_request);
    defer changed_dependency.deinit();
    try std.testing.expect(!stagePlanIdEql(plan.plan_id, changed_dependency.plan_id));

    var tied_config = config;
    tied_config.tie_word_embeddings = true;
    const tied_request = planRequest(4, &.{ 1, 3 }, &tied_config, &manifest);
    var tied_plan = try buildStagePlan(allocator, tied_request);
    defer tied_plan.deinit();
    try std.testing.expect(!stagePlanIdEql(plan.plan_id, tied_plan.plan_id));
}

fn runFailureContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{0}, &config, &manifest)));
    try std.testing.expectError(error.DuplicateSplitPoint, buildStagePlan(allocator, planRequest(4, &.{ 2, 2 }, &config, &manifest)));

    config.mamba_d_state = 16;
    var arch = testArch();
    arch.has_mamba = true;
    const no_constraints = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{2},
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
    };
    try std.testing.expectError(error.MissingPartitionConstraints, buildStagePlan(allocator, no_constraints));

    var missing_dependency = no_constraints;
    missing_dependency.partition_constraints = .{ .decoder_cuts_allowed = true };
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, missing_dependency));

    var single_stage_stateful = no_constraints;
    single_stage_stateful.split_points = &.{};
    var single_stage_plan = try buildStagePlan(allocator, single_stage_stateful);
    defer single_stage_plan.deinit();
    try std.testing.expectEqual(PartitionConstraintSource.single_stage, single_stage_plan.partition_constraint_source);

    var explicit_blocked = planRequest(4, &.{2}, &config, &manifest);
    explicit_blocked.partition_constraints = .{ .decoder_cuts_allowed = false };
    try std.testing.expectError(error.MissingPartitionConstraints, buildStagePlan(allocator, explicit_blocked));

    var plain_derived = planRequest(4, &.{2}, &config, &manifest);
    plain_derived.partition_constraints = null;
    arch.has_mamba = false;
    plain_derived.architecture = &arch;
    config.mamba_d_state = 0;
    var derived_plan = try buildStagePlan(allocator, plain_derived);
    defer derived_plan.deinit();
    try std.testing.expectEqual(PartitionConstraintSource.derived_plain_decoder, derived_plan.partition_constraint_source);

    const duplicate_dependencies = [_]DependencyOverride{
        .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder },
        .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .explicit },
    };
    var duplicate_request = planRequest(4, &.{2}, &config, &manifest);
    duplicate_request.dependency_overrides = &duplicate_dependencies;
    try std.testing.expectError(error.DuplicateDependency, buildStagePlan(allocator, duplicate_request));

    var tied_manifest = try tiedManifestWithoutIndependentLmHead(allocator, 4);
    defer tied_manifest.deinit();
    config.tie_word_embeddings = true;
    const duplicate_implicit = [_]DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .role = .token_embeddings,
        .reason = .explicit,
        .affects_loader_residency = true,
    }};
    var duplicate_implicit_request = planRequest(4, &.{2}, &config, &tied_manifest);
    duplicate_implicit_request.dependency_overrides = &duplicate_implicit;
    try std.testing.expectError(error.DuplicateDependency, buildStagePlan(allocator, duplicate_implicit_request));
}

fn runRoleOwnershipContract(allocator: Allocator) !void {
    var config = testConfig(2);
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "vision.patch_embed.weight", .checkpoint_bytes = 50, .role = .vision_side, .weight_id = "vision.patch_embed" },
        .{ .name = "architecture.routing.weight", .checkpoint_bytes = 33, .role = .architecture_side, .weight_id = "architecture.routing" },
        .{ .name = "unknown.weight", .checkpoint_bytes = 9, .role = .unclassified_global, .status = .unclassified },
    };
    var manifest = try testManifest(allocator, 2, &specs);
    defer manifest.deinit();

    try std.testing.expectError(error.MissingRoleOwner, buildStagePlan(allocator, planRequest(2, &.{1}, &config, &manifest)));

    var request = planRequest(2, &.{1}, &config, &manifest);
    request.allow_unclassified_global = true;
    request.role_owner_overrides = &.{
        .{ .role = .architecture_side, .stage_id = 1 },
        .{ .role = .unclassified_global, .stage_id = 1 },
    };
    request.dependency_overrides = &.{
        .{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .role = .vision_side,
            .reason = .vision_side,
            .affects_loader_residency = true,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 0,
            .role = .architecture_side,
            .reason = .architecture_side,
            .affects_loader_residency = true,
        },
        .{
            .source_stage_id = 1,
            .target_stage_id = 0,
            .role = .unclassified_global,
            .reason = .unclassified_global,
            .affects_loader_residency = true,
        },
    };

    var plan = try buildStagePlan(allocator, request);
    defer plan.deinit();
    try std.testing.expect(plan.stages[0].owned_roles[@intFromEnum(TensorRole.vision_side)]);
    try std.testing.expect(plan.stages[1].owned_roles[@intFromEnum(TensorRole.architecture_side)]);
    try std.testing.expectEqual(@as(usize, 1), plan.diagnostics.len);
    const load_request = try plan.stageLoadRequest(1);
    try std.testing.expect(load_request.roles.include_vision_side);
    try std.testing.expect(load_request.roles.include_architecture_side);
    try std.testing.expect(load_request.roles.include_unclassified_global);
}

fn runGraphIdentityChangeContract(allocator: Allocator) !void {
    var arch = testArch();
    var config = testConfig(2);
    const specs = [_]TestEntrySpec{
        .{ .name = "base.weight", .role = .decoder_layer, .layer_index = 0, .weight_id = "base" },
    };
    var manifest = try testManifest(allocator, 2, &specs);
    defer manifest.deinit();

    const baseline = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });

    try expectDigestChanged(allocator, baseline, .{
        .graph_contract_version = graph_identity_contract_version + 1,
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });
    try expectDigestChanged(allocator, baseline, .{
        .stage_contract_version = stage_plan_contract_version + 1,
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test_2",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });

    var renamed_arch = arch;
    renamed_arch.name = "stage_plan_test_renamed";
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &renamed_arch,
        .config = &config,
        .manifest = &manifest,
    });

    var changed_arch = arch;
    changed_arch.norm_weight_offset = 1.0;
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &changed_arch,
        .config = &config,
        .manifest = &manifest,
    });

    var changed_config = config;
    changed_config.d_model += 1;
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &changed_config,
        .manifest = &manifest,
    });

    const layer_types_a = [_]u8{ 0, 1 };
    const layer_types_b = [_]u8{ 1, 0 };
    var layer_config = config;
    layer_config.layer_types = &layer_types_a;
    var changed_layer_config = config;
    changed_layer_config.layer_types = &layer_types_b;
    const layer_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &layer_config,
        .manifest = &manifest,
    });
    try expectDigestChanged(allocator, layer_identity, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &changed_layer_config,
        .manifest = &manifest,
    });

    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
        .load_semantics = .{ .preserve_native_norm_dtype = true },
    });
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
        .load_semantics = .{ .dequantize_mxfp8_to_bf16 = true },
    });
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
        .load_semantics = .{ .dequantize_nvfp4_to_bf16 = false },
    });

    try expectManifestEntryDigestChanges(allocator, baseline, &arch, &config);
}

fn expectManifestEntryDigestChanges(
    allocator: Allocator,
    baseline: GraphIdentity,
    arch: *const Architecture,
    config: *const ModelConfig,
) !void {
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "renamed.weight",
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .dtype = .f32,
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .shape = &.{ 8, 4 },
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .checkpoint_bytes = 33,
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .lm_head,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .decoder_layer,
        .owner_role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .decoder_layer,
        .layer_index = 1,
        .weight_id = "base",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "other",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
        .primary_name = "primary.weight",
    }});
    try expectManifestDigestChanged(allocator, baseline, arch, config, &.{.{
        .name = "base.weight",
        .role = .decoder_layer,
        .layer_index = 0,
        .weight_id = "base",
        .status = .unclassified,
    }});
}

fn expectManifestDigestChanged(
    allocator: Allocator,
    baseline: GraphIdentity,
    arch: *const Architecture,
    config: *const ModelConfig,
    specs: []const TestEntrySpec,
) !void {
    var changed_manifest = try testManifest(allocator, 2, specs);
    defer changed_manifest.deinit();
    try expectDigestChanged(allocator, baseline, .{
        .architecture_id = "stage_plan_test",
        .architecture = arch,
        .config = config,
        .manifest = &changed_manifest,
    });
}

fn expectDigestChanged(
    allocator: Allocator,
    baseline: GraphIdentity,
    inputs: GraphIdentityInputs,
) !void {
    const changed = try graphIdentity(allocator, inputs);
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &changed.digest));
}

fn runStageShapeContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    var one = try buildStagePlan(allocator, planRequest(4, &.{}, &config, &manifest));
    defer one.deinit();
    try std.testing.expectEqual(@as(usize, 1), one.stages.len);
    try std.testing.expectEqual(@as(usize, 0), one.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), one.stages[0].layer_start);
    try std.testing.expectEqual(@as(usize, 4), one.stages[0].layer_end);

    var two = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer two.deinit();
    try std.testing.expectEqual(@as(usize, 2), two.stages.len);
    try std.testing.expectEqual(@as(usize, 1), two.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), two.boundaries[0].producer_layer_start);
    try std.testing.expectEqual(@as(usize, 2), two.boundaries[0].producer_layer_end);
    try std.testing.expectEqual(@as(usize, 2), two.boundaries[0].consumer_layer_start);
    try std.testing.expectEqual(@as(usize, 4), two.boundaries[0].consumer_layer_end);

    var three = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer three.deinit();
    try std.testing.expectEqual(@as(usize, 3), three.stages.len);
    try std.testing.expectEqual(@as(usize, 1), three.stages[1].layer_start);
    try std.testing.expectEqual(@as(usize, 3), three.stages[1].layer_end);

    var four = try buildStagePlan(allocator, planRequest(4, &.{ 1, 2, 3 }, &config, &manifest));
    defer four.deinit();
    try std.testing.expectEqual(@as(usize, 4), four.stages.len);
    try std.testing.expectEqual(@as(usize, 3), four.boundaries.len);
    try std.testing.expectEqual(@as(usize, 3), four.stages[3].layer_start);
    try std.testing.expectEqual(@as(usize, 4), four.stages[3].layer_end);

    try std.testing.expectError(error.UnknownStageId, four.stage(4));
    try std.testing.expectError(error.UnknownStageId, four.stageLoadRequest(4));
    for (four.stages) |stage_entry| {
        try std.testing.expect(!stage_entry.owned_roles[@intFromEnum(TensorRole.quant_companion)]);
    }
}

fn runValidationFailureContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    try std.testing.expectError(error.MissingManifest, buildStagePlan(allocator, .{ .n_layers = 4 }));
    try std.testing.expectError(error.InvalidLayerCount, buildStagePlan(allocator, .{
        .n_layers = 0,
        .manifest = &manifest,
    }));

    var bad_config = testConfig(3);
    try std.testing.expectError(error.InvalidLayerCount, buildStagePlan(allocator, .{
        .n_layers = 4,
        .model_config = &bad_config,
        .manifest = &manifest,
    }));

    var short_manifest = try standardManifest(allocator, 3);
    defer short_manifest.deinit();
    try std.testing.expectError(error.InvalidLayerCount, buildStagePlan(allocator, .{
        .n_layers = 4,
        .model_config = &config,
        .manifest = &short_manifest,
    }));

    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{4}, &config, &manifest)));
    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{ 3, 2 }, &config, &manifest)));

    var forbidden_request = planRequest(4, &.{2}, &config, &manifest);
    forbidden_request.partition_constraints = .{
        .decoder_cuts_allowed = true,
        .forbidden_split_points = &.{.{ .layer_index = 2, .reason = "state crosses split" }},
    };
    try std.testing.expectError(error.ForbiddenSplitPoint, buildStagePlan(allocator, forbidden_request));

    var no_metadata = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{2},
        .manifest = &manifest,
    };
    try std.testing.expectError(error.MissingGraphIdentity, buildStagePlan(allocator, no_metadata));
    no_metadata.split_points = &.{};
    try std.testing.expectError(error.MissingGraphIdentity, buildStagePlan(allocator, no_metadata));

    const supplied_identity = GraphIdentity{
        .architecture_id = "stage_plan_precomputed",
        .digest = [_]u8{1} ** 32,
    };
    no_metadata.graph_identity = supplied_identity;
    no_metadata.split_points = &.{2};
    try std.testing.expectError(error.MissingPartitionConstraints, buildStagePlan(allocator, no_metadata));
    no_metadata.split_points = &.{};
    var standalone_one_stage = try buildStagePlan(allocator, no_metadata);
    defer standalone_one_stage.deinit();
    try std.testing.expectEqual(PartitionConstraintSource.single_stage, standalone_one_stage.partition_constraint_source);

    const blocked_roles = [_]TensorRole{
        .token_embeddings,
        .embedding_side,
        .final_norm,
        .lm_head,
        .decoder_layer,
        .quant_companion,
    };
    for (blocked_roles) |role| {
        const overrides = [_]RoleOwnerOverride{.{ .role = role, .stage_id = 1 }};
        var request = planRequest(4, &.{2}, &config, &manifest);
        request.role_owner_overrides = &overrides;
        try std.testing.expectError(error.UnsupportedRoleOwnerOverride, buildStagePlan(allocator, request));
    }

    const duplicate_overrides = [_]RoleOwnerOverride{
        .{ .role = .vision_side, .stage_id = 0 },
        .{ .role = .vision_side, .stage_id = 1 },
    };
    var duplicate_request = planRequest(4, &.{2}, &config, &manifest);
    duplicate_request.role_owner_overrides = &duplicate_overrides;
    try std.testing.expectError(error.DuplicateRoleOwnerOverride, buildStagePlan(allocator, duplicate_request));

    const unknown_owner = [_]RoleOwnerOverride{.{ .role = .vision_side, .stage_id = 9 }};
    var unknown_owner_request = planRequest(4, &.{2}, &config, &manifest);
    unknown_owner_request.role_owner_overrides = &unknown_owner;
    try std.testing.expectError(error.UnknownStageId, buildStagePlan(allocator, unknown_owner_request));

    const unknown_dependency = [_]DependencyOverride{.{ .source_stage_id = 0, .target_stage_id = 9 }};
    var unknown_dependency_request = planRequest(4, &.{2}, &config, &manifest);
    unknown_dependency_request.dependency_overrides = &unknown_dependency;
    try std.testing.expectError(error.UnknownStageId, buildStagePlan(allocator, unknown_dependency_request));

    const same_stage_dependency = [_]DependencyOverride{.{ .source_stage_id = 0, .target_stage_id = 0 }};
    var same_stage_request = planRequest(4, &.{2}, &config, &manifest);
    same_stage_request.dependency_overrides = &same_stage_dependency;
    try std.testing.expectError(error.InvalidDependency, buildStagePlan(allocator, same_stage_request));

    const residency_without_role = [_]DependencyOverride{.{ .source_stage_id = 0, .target_stage_id = 1, .affects_loader_residency = true }};
    var residency_without_role_request = planRequest(4, &.{2}, &config, &manifest);
    residency_without_role_request.dependency_overrides = &residency_without_role;
    try std.testing.expectError(error.InvalidDependency, buildStagePlan(allocator, residency_without_role_request));

    const decoder_role_dependency = [_]DependencyOverride{.{ .source_stage_id = 0, .target_stage_id = 1, .role = .decoder_layer }};
    var decoder_role_request = planRequest(4, &.{2}, &config, &manifest);
    decoder_role_request.dependency_overrides = &decoder_role_dependency;
    try std.testing.expectError(error.InvalidDependency, buildStagePlan(allocator, decoder_role_request));
}

fn runValidateStagePlanContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    var plan = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer plan.deinit();
    try validateStagePlan(&plan, .{ .expected_graph_identity = plan.graph_identity, .manifest = &manifest });

    var invalid_version = plan;
    invalid_version.stage_contract_version += 1;
    try std.testing.expectError(error.InvalidContractVersion, validateStagePlan(&invalid_version, .{}));

    var invalid_graph_contract = plan;
    invalid_graph_contract.graph_identity.graph_contract_version += 1;
    try std.testing.expectError(error.InvalidContractVersion, validateStagePlan(&invalid_graph_contract, .{}));

    var invalid_graph_stage_contract = plan;
    invalid_graph_stage_contract.graph_identity.stage_contract_version += 1;
    try std.testing.expectError(error.InvalidContractVersion, validateStagePlan(&invalid_graph_stage_contract, .{}));

    var missing_graph_architecture_id = plan;
    missing_graph_architecture_id.graph_identity.architecture_id = "";
    try std.testing.expectError(error.MissingGraphIdentity, validateStagePlan(&missing_graph_architecture_id, .{}));

    var wrong_identity = plan.graph_identity;
    wrong_identity.digest[0] ^= 0x01;
    try std.testing.expectError(error.GraphIdentityMismatch, validateStagePlan(&plan, .{ .expected_graph_identity = wrong_identity }));

    {
        const stages = try allocator.dupe(StagePlanStage, plan.stages);
        defer allocator.free(stages);
        stages[1].id = stages[0].id;
        var invalid = plan;
        invalid.stages = stages;
        try std.testing.expectError(error.DuplicateStageId, validateStagePlan(&invalid, .{}));
    }

    {
        const stages = try allocator.dupe(StagePlanStage, plan.stages);
        defer allocator.free(stages);
        stages[1].layer_end = stages[1].layer_start;
        var invalid = plan;
        invalid.stages = stages;
        try std.testing.expectError(error.InvalidStageRange, validateStagePlan(&invalid, .{}));
    }

    {
        const stages = try allocator.dupe(StagePlanStage, plan.stages);
        defer allocator.free(stages);
        stages[1].layer_start += 1;
        var invalid = plan;
        invalid.stages = stages;
        try std.testing.expectError(error.NonContiguousStageRange, validateStagePlan(&invalid, .{}));
    }

    {
        const boundaries = try allocator.dupe(StageBoundary, plan.boundaries);
        defer allocator.free(boundaries);
        boundaries[0].producer_layer_end += 1;
        var invalid = plan;
        invalid.boundaries = boundaries;
        try std.testing.expectError(error.NonContiguousStageRange, validateStagePlan(&invalid, .{}));
    }

    {
        const dependencies = try allocator.dupe(StageDependency, &.{
            .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .stateful_decoder, .affects_loader_residency = false },
            .{ .source_stage_id = 0, .target_stage_id = 1, .reason = .explicit, .affects_loader_residency = false },
        });
        defer allocator.free(dependencies);
        var invalid = plan;
        invalid.dependencies = dependencies;
        try std.testing.expectError(error.DuplicateDependency, validateStagePlan(&invalid, .{}));
    }

    {
        const stages = try allocator.dupe(StagePlanStage, plan.stages);
        defer allocator.free(stages);
        stages[0].residency.total_checkpoint_bytes += 1;
        var invalid = plan;
        invalid.stages = stages;
        try std.testing.expectError(error.ResidencyMismatch, validateStagePlan(&invalid, .{ .manifest = &manifest }));
    }

    var invalid_fingerprint = plan;
    invalid_fingerprint.plan_id.digest[0] ^= 0x01;
    try std.testing.expectError(error.PlanFingerprintMismatch, validateStagePlan(&invalid_fingerprint, .{}));
}

fn runSideDomainDependencyContract(allocator: Allocator) !void {
    var config = testConfig(2);

    const vision_extra = [_]TestEntrySpec{
        .{ .name = "vision.patch_embed.weight", .checkpoint_bytes = 50, .role = .vision_side, .weight_id = "vision.patch_embed" },
    };
    var vision_manifest = try twoLayerManifestWith(allocator, &vision_extra);
    defer vision_manifest.deinit();
    var vision_request = planRequest(2, &.{1}, &config, &vision_manifest);
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, vision_request));

    const wrong_vision_dependency = [_]DependencyOverride{.{
        .source_stage_id = 1,
        .target_stage_id = 0,
        .role = .vision_side,
        .reason = .vision_side,
        .affects_loader_residency = true,
    }};
    var wrong_vision_request = vision_request;
    wrong_vision_request.dependency_overrides = &wrong_vision_dependency;
    try std.testing.expectError(error.InvalidDependency, buildStagePlan(allocator, wrong_vision_request));

    const vision_dependency = [_]DependencyOverride{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .role = .vision_side,
        .reason = .vision_side,
        .affects_loader_residency = true,
    }};
    vision_request.dependency_overrides = &vision_dependency;
    var vision_plan = try buildStagePlan(allocator, vision_request);
    defer vision_plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), vision_plan.boundaries.len);
    try std.testing.expectEqual(@as(usize, 1), vision_plan.dependencies.len);
    try std.testing.expect(vision_plan.stages[0].owned_roles[@intFromEnum(TensorRole.vision_side)]);
    try std.testing.expect((try vision_plan.stageLoadRequest(1)).roles.include_vision_side);

    const architecture_extra = [_]TestEntrySpec{
        .{ .name = "architecture.routing.weight", .checkpoint_bytes = 33, .role = .architecture_side, .weight_id = "architecture.routing" },
    };
    var architecture_manifest = try twoLayerManifestWith(allocator, &architecture_extra);
    defer architecture_manifest.deinit();
    try std.testing.expectError(error.MissingRoleOwner, buildStagePlan(allocator, planRequest(2, &.{1}, &config, &architecture_manifest)));

    var architecture_owner_request = planRequest(2, &.{1}, &config, &architecture_manifest);
    architecture_owner_request.role_owner_overrides = &.{.{ .role = .architecture_side, .stage_id = 1 }};
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, architecture_owner_request));

    architecture_owner_request.dependency_overrides = &.{.{
        .source_stage_id = 1,
        .target_stage_id = 0,
        .role = .architecture_side,
        .reason = .architecture_side,
        .affects_loader_residency = true,
    }};
    var architecture_plan = try buildStagePlan(allocator, architecture_owner_request);
    defer architecture_plan.deinit();
    try std.testing.expect(architecture_plan.stages[1].owned_roles[@intFromEnum(TensorRole.architecture_side)]);
    try std.testing.expect((try architecture_plan.stageLoadRequest(0)).roles.include_architecture_side);

    var one_stage_architecture = planRequest(2, &.{}, &config, &architecture_manifest);
    one_stage_architecture.partition_constraints = null;
    var one_stage_architecture_plan = try buildStagePlan(allocator, one_stage_architecture);
    defer one_stage_architecture_plan.deinit();
    try std.testing.expect(one_stage_architecture_plan.stages[0].owned_roles[@intFromEnum(TensorRole.architecture_side)]);

    const unclassified_extra = [_]TestEntrySpec{
        .{ .name = "unknown.weight", .checkpoint_bytes = 9, .role = .unclassified_global, .status = .unclassified },
    };
    var unclassified_manifest = try twoLayerManifestWith(allocator, &unclassified_extra);
    defer unclassified_manifest.deinit();
    try std.testing.expectError(error.UnclassifiedGlobalNotAllowed, buildStagePlan(allocator, planRequest(2, &.{1}, &config, &unclassified_manifest)));

    var unclassified_request = planRequest(2, &.{1}, &config, &unclassified_manifest);
    unclassified_request.allow_unclassified_global = true;
    try std.testing.expectError(error.MissingRoleOwner, buildStagePlan(allocator, unclassified_request));

    unclassified_request.role_owner_overrides = &.{.{ .role = .unclassified_global, .stage_id = 1 }};
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, unclassified_request));

    unclassified_request.dependency_overrides = &.{.{
        .source_stage_id = 1,
        .target_stage_id = 0,
        .role = .unclassified_global,
        .reason = .unclassified_global,
        .affects_loader_residency = true,
    }};
    var unclassified_plan = try buildStagePlan(allocator, unclassified_request);
    defer unclassified_plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), unclassified_plan.diagnostics.len);
    try std.testing.expect((try unclassified_plan.stageLoadRequest(0)).roles.include_unclassified_global);
}

fn runResidencyContract(allocator: Allocator) !void {
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.tie_word_embeddings = true;

    var plan = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer plan.deinit();

    const stage0_report = plan.stages[0].residency;
    try std.testing.expectEqual(@as(usize, 0), stage0_report.layer_start);
    try std.testing.expectEqual(@as(usize, 1), stage0_report.layer_end);
    try std.testing.expectEqual(@as(usize, 122), stage0_report.total_checkpoint_bytes);
    try std.testing.expectEqual(@as(usize, 100), stage0_report.bytesForRole(.token_embeddings));
    try std.testing.expectEqual(@as(usize, 12), stage0_report.bytesForRole(.embedding_side));

    const stage1_report = plan.stages[1].residency;
    try std.testing.expectEqual(@as(usize, 1), stage1_report.layer_start);
    try std.testing.expectEqual(@as(usize, 3), stage1_report.layer_end);
    try std.testing.expectEqual(@as(usize, 50), stage1_report.total_checkpoint_bytes);
    try std.testing.expectEqual(@as(usize, 0), stage1_report.bytesForRole(.token_embeddings));

    const stage2_report = plan.stages[2].residency;
    try std.testing.expectEqual(@as(usize, 3), stage2_report.layer_start);
    try std.testing.expectEqual(@as(usize, 4), stage2_report.layer_end);
    try std.testing.expectEqual(@as(usize, 160), stage2_report.total_checkpoint_bytes);
    try std.testing.expectEqual(@as(usize, 0), stage2_report.bytesForRole(.token_embeddings));
    try std.testing.expectEqual(@as(usize, 100), stage2_report.bytesForRole(.lm_head));
    try std.testing.expectEqual(@as(usize, 4), stage2_report.bytesForRole(.quant_companion));
    try std.testing.expect(!plan.stages[2].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
}

fn twoLayerManifestWith(allocator: Allocator, extra_specs: []const TestEntrySpec) !ModelManifest {
    const base_specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.norm.weight", .dtype = .f32, .shape = &.{4}, .checkpoint_bytes = 16, .role = .final_norm, .weight_id = "ln_final" },
        .{ .name = "lm_head.weight", .checkpoint_bytes = 100, .role = .lm_head, .weight_id = "lm_head" },
    };
    const specs = try allocator.alloc(TestEntrySpec, base_specs.len + extra_specs.len);
    defer allocator.free(specs);
    @memcpy(specs[0..base_specs.len], base_specs[0..]);
    @memcpy(specs[base_specs.len..], extra_specs);
    return testManifest(allocator, 2, specs);
}

test "stage_plan graphIdentity buildStagePlan StagePlan.stageLoadRequest contract tests" {
    try testing.runContractTests(std.testing.allocator);
}

test "stage_plan graphIdentityEql dupeGraphIdentity deinitGraphIdentity validateGraphIdentity assertGraphIdentity validateStagePlan contract tests" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    var plan = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer plan.deinit();

    try std.testing.expect(graphIdentityEql(plan.graph_identity, plan.graph_identity));
    try assertGraphIdentity(&plan, plan.graph_identity);
    try validateStagePlan(&plan, .{ .expected_graph_identity = plan.graph_identity, .manifest = &manifest });
}

test "stage_plan graphIdentity equal inputs produce equal digests and ignore manifest order" {
    const allocator = std.testing.allocator;
    var arch = testArch();
    var config = testConfig(2);
    const specs_a = [_]TestEntrySpec{
        .{ .name = "b.weight", .role = .decoder_layer, .layer_index = 1, .weight_id = "b" },
        .{ .name = "a.weight", .role = .token_embeddings, .weight_id = "token_embeddings" },
    };
    const specs_b = [_]TestEntrySpec{
        .{ .name = "a.weight", .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "b.weight", .role = .decoder_layer, .layer_index = 1, .weight_id = "b" },
    };
    var manifest_a = try testManifest(allocator, 2, &specs_a);
    defer manifest_a.deinit();
    var manifest_b = try testManifest(allocator, 2, &specs_b);
    defer manifest_b.deinit();

    const identity_a = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest_a,
    });
    const identity_b = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest_b,
    });
    try std.testing.expectEqualSlices(u8, &identity_a.digest, &identity_b.digest);
}

test "stage_plan graphIdentity changes on config architecture manifest load semantics and versions" {
    const allocator = std.testing.allocator;
    var arch = testArch();
    var config = testConfig(2);
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();

    const baseline = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });

    var changed_config = config;
    changed_config.d_model += 1;
    const config_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &changed_config,
        .manifest = &manifest,
    });
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &config_identity.digest));

    var changed_arch = arch;
    changed_arch.norm_weight_offset = 1.0;
    const arch_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &changed_arch,
        .config = &config,
        .manifest = &manifest,
    });
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &arch_identity.digest));

    const changed_specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 101, .role = .token_embeddings, .weight_id = "token_embeddings" },
    };
    var changed_manifest = try testManifest(allocator, 2, &changed_specs);
    defer changed_manifest.deinit();
    const manifest_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &changed_manifest,
    });
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &manifest_identity.digest));

    const load_identity = try graphIdentity(allocator, .{
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
        .load_semantics = .{ .preserve_native_norm_dtype = true },
    });
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &load_identity.digest));

    const version_identity = try graphIdentity(allocator, .{
        .graph_contract_version = graph_identity_contract_version + 1,
        .architecture_id = "stage_plan_test",
        .architecture = &arch,
        .config = &config,
        .manifest = &manifest,
    });
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &version_identity.digest));
}

test "stage_plan graphIdentity ModelConfig field coverage guard" {
    try std.testing.expectEqual(model_config_identity_field_count, std.meta.fields(ModelConfig).len);
    try std.testing.expectEqual(rope_scaling_identity_field_count, std.meta.fields(RopeScaling).len);
}

test "stage_plan LoadSemantics.fromLoadOptions copies loader choices" {
    const semantics = LoadSemantics.fromLoadOptions(.{
        .preserve_native_norm_dtype = true,
        .dequantize_mxfp8_to_bf16 = true,
        .dequantize_nvfp4_to_bf16 = false,
    });
    try std.testing.expect(semantics.preserve_native_norm_dtype);
    try std.testing.expect(semantics.dequantize_mxfp8_to_bf16);
    try std.testing.expect(!semantics.dequantize_nvfp4_to_bf16);
}

test "stage_plan buildStagePlan creates one two three and four stage ranges" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    var one = try buildStagePlan(allocator, planRequest(4, &.{}, &config, &manifest));
    defer one.deinit();
    try std.testing.expectEqual(@as(usize, 1), one.stages.len);
    try std.testing.expectEqual(@as(usize, 0), one.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), (try one.stage(0)).layer_start);
    try std.testing.expectEqual(@as(usize, 4), (try one.stage(0)).layer_end);

    var two = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer two.deinit();
    try std.testing.expectEqual(@as(usize, 2), two.stages.len);
    try std.testing.expectEqual(@as(usize, 1), two.boundaries.len);
    try std.testing.expectEqual(@as(usize, 0), two.boundaries[0].producer_layer_start);
    try std.testing.expectEqual(@as(usize, 2), two.boundaries[0].producer_layer_end);
    try std.testing.expectEqual(@as(usize, 2), two.boundaries[0].consumer_layer_start);
    try std.testing.expectEqual(@as(usize, 4), two.boundaries[0].consumer_layer_end);

    var three = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer three.deinit();
    try std.testing.expectEqual(@as(usize, 3), three.stages.len);
    try std.testing.expectEqual(@as(usize, 1), three.stages[1].layer_start);
    try std.testing.expectEqual(@as(usize, 3), three.stages[1].layer_end);

    var four = try buildStagePlan(allocator, planRequest(4, &.{ 1, 2, 3 }, &config, &manifest));
    defer four.deinit();
    try std.testing.expectEqual(@as(usize, 4), four.stages.len);
    try std.testing.expectEqual(@as(usize, 3), four.boundaries.len);
    try std.testing.expectEqual(@as(usize, 3), four.stages[3].layer_start);
    try std.testing.expectEqual(@as(usize, 4), four.stages[3].layer_end);
}

test "stage_plan StagePlan.stage and StagePlan.stageLoadRequest return typed unknown stage errors" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    var plan = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer plan.deinit();

    try std.testing.expectError(error.UnknownStageId, plan.stage(9));
    try std.testing.expectError(error.UnknownStageId, plan.stageLoadRequest(9));
}

test "stage_plan buildStagePlan rejects invalid splits and forbidden split points" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);

    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{0}, &config, &manifest)));
    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{4}, &config, &manifest)));
    try std.testing.expectError(error.InvalidSplitPoint, buildStagePlan(allocator, planRequest(4, &.{ 3, 2 }, &config, &manifest)));
    try std.testing.expectError(error.DuplicateSplitPoint, buildStagePlan(allocator, planRequest(4, &.{ 2, 2 }, &config, &manifest)));

    var request = planRequest(4, &.{2}, &config, &manifest);
    request.partition_constraints = .{
        .decoder_cuts_allowed = true,
        .forbidden_split_points = &.{.{ .layer_index = 2, .reason = "state crosses split" }},
    };
    try std.testing.expectError(error.ForbiddenSplitPoint, buildStagePlan(allocator, request));
}

test "stage_plan buildStagePlan fails closed for missing constraints and missing dependencies" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.mamba_d_state = 16;
    var arch = testArch();
    arch.has_mamba = true;

    const no_constraints = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{2},
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
    };
    try std.testing.expectError(error.MissingPartitionConstraints, buildStagePlan(allocator, no_constraints));

    const no_dependencies = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{2},
        .architecture = &arch,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, no_dependencies));

    var with_dependency = no_dependencies;
    with_dependency.partition_constraints = .{
        .decoder_cuts_allowed = true,
        .dependency_overrides = &.{.{
            .source_stage_id = 0,
            .target_stage_id = 1,
            .reason = .stateful_decoder,
        }},
    };
    var plan = try buildStagePlan(allocator, with_dependency);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.dependencies.len);
}

test "stage_plan requiresBoundaryDependenciesFor centralizes stateful decoder cut semantics" {
    var config = testConfig(4);
    var arch = testArch();
    try std.testing.expect(!requiresBoundaryDependenciesFor(&arch, &config));

    config.num_kv_shared_layers = 1;
    try std.testing.expect(requiresBoundaryDependenciesFor(&arch, &config));

    config = testConfig(4);
    arch.has_gated_delta = true;
    try std.testing.expect(requiresBoundaryDependenciesFor(&arch, &config));
}

test "stage_plan buildStagePlan independent lm_head does not require token embedding residency" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.tie_word_embeddings = true;
    var plan = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer plan.deinit();

    try std.testing.expect(plan.stages[0].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(!plan.stages[1].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(plan.stages[1].owned_roles[@intFromEnum(TensorRole.lm_head)]);
    try std.testing.expectEqual(@as(usize, 0), plan.dependencies.len);

    const final_request = try plan.stageLoadRequest(1);
    try std.testing.expect(final_request.roles.include_lm_head);
    try std.testing.expect(!final_request.roles.include_token_embeddings);
}

test "stage_plan buildStagePlan tied lm_head dependency applies without independent lm_head" {
    const allocator = std.testing.allocator;
    var manifest = try tiedManifestWithoutIndependentLmHead(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.tie_word_embeddings = true;
    var plan = try buildStagePlan(allocator, planRequest(4, &.{2}, &config, &manifest));
    defer plan.deinit();

    try std.testing.expect(plan.stages[0].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(!plan.stages[1].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(plan.stages[1].owned_roles[@intFromEnum(TensorRole.lm_head)]);
    try std.testing.expectEqual(@as(usize, 1), plan.dependencies.len);
    try std.testing.expectEqual(StageDependencyReason.tied_lm_head, plan.dependencies[0].reason);
    try std.testing.expectEqual(TensorRole.token_embeddings, plan.dependencies[0].role.?);

    const final_request = try plan.stageLoadRequest(1);
    try std.testing.expect(final_request.roles.include_lm_head);
    try std.testing.expect(final_request.roles.include_token_embeddings);
}

test "stage_plan standalone graph identity requires explicit tied lm_head semantics" {
    const allocator = std.testing.allocator;
    var manifest = try tiedManifestWithoutIndependentLmHead(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.tie_word_embeddings = true;

    const identity = try graphIdentity(allocator, .{
        .architecture_id = manifest.architecture_id,
        .architecture = &default_test_arch,
        .config = &config,
        .manifest = &manifest,
    });

    const standalone_missing_semantics = StagePlanRequest{
        .n_layers = 4,
        .split_points = &.{2},
        .manifest = &manifest,
        .graph_identity = identity,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    };
    try std.testing.expectError(error.MissingRoleSemantics, buildStagePlan(allocator, standalone_missing_semantics));

    var standalone_tied_semantics = standalone_missing_semantics;
    standalone_tied_semantics.role_semantics = .{ .tie_word_embeddings = true };
    var plan = try buildStagePlan(allocator, standalone_tied_semantics);
    defer plan.deinit();

    try std.testing.expect(plan.stages[0].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(!plan.stages[1].owned_roles[@intFromEnum(TensorRole.token_embeddings)]);
    try std.testing.expect(plan.stages[1].owned_roles[@intFromEnum(TensorRole.lm_head)]);
    try std.testing.expectEqual(@as(usize, 1), plan.dependencies.len);
    try std.testing.expectEqual(StageDependencyReason.tied_lm_head, plan.dependencies[0].reason);
    try std.testing.expectEqual(TensorRole.token_embeddings, plan.dependencies[0].role.?);

    const final_request = try plan.stageLoadRequest(1);
    try std.testing.expect(final_request.roles.include_lm_head);
    try std.testing.expect(final_request.roles.include_token_embeddings);
    try validateStagePlan(&plan, .{ .expected_graph_identity = identity, .manifest = &manifest });
}

test "stage_plan buildStagePlan uses StageDependency for vision side residency" {
    const allocator = std.testing.allocator;
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "vision.patch_embed.weight", .checkpoint_bytes = 50, .role = .vision_side, .weight_id = "vision.patch_embed" },
    };
    var manifest = try testManifest(allocator, 2, &specs);
    defer manifest.deinit();
    var config = testConfig(2);
    var request = planRequest(2, &.{1}, &config, &manifest);
    request.dependency_overrides = &.{.{
        .source_stage_id = 0,
        .target_stage_id = 1,
        .role = .vision_side,
        .reason = .vision_side,
        .affects_loader_residency = true,
    }};

    var plan = try buildStagePlan(allocator, request);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.boundaries.len);
    try std.testing.expectEqual(@as(usize, 1), plan.dependencies.len);
    try std.testing.expect(plan.stages[0].owned_roles[@intFromEnum(TensorRole.vision_side)]);
    try std.testing.expect((try plan.stageLoadRequest(1)).roles.include_vision_side);
}

test "stage_plan buildStagePlan handles architecture side explicit ownership" {
    const allocator = std.testing.allocator;
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "architecture.routing.weight", .checkpoint_bytes = 33, .role = .architecture_side, .weight_id = "architecture.routing" },
    };
    var manifest = try testManifest(allocator, 2, &specs);
    defer manifest.deinit();
    var config = testConfig(2);

    try std.testing.expectError(error.MissingRoleOwner, buildStagePlan(allocator, planRequest(2, &.{1}, &config, &manifest)));

    var request = planRequest(2, &.{1}, &config, &manifest);
    request.role_owner_overrides = &.{.{ .role = .architecture_side, .stage_id = 1 }};
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, request));

    request.dependency_overrides = &.{.{
        .source_stage_id = 1,
        .target_stage_id = 0,
        .role = .architecture_side,
        .reason = .architecture_side,
        .affects_loader_residency = true,
    }};
    var plan = try buildStagePlan(allocator, request);
    defer plan.deinit();
    try std.testing.expect(plan.stages[1].owned_roles[@intFromEnum(TensorRole.architecture_side)]);
    try std.testing.expect((try plan.stageLoadRequest(0)).roles.include_architecture_side);
    try std.testing.expect((try plan.stageLoadRequest(1)).roles.include_architecture_side);
}

test "stage_plan buildStagePlan requires unclassified allow flag owner and emits diagnostic" {
    const allocator = std.testing.allocator;
    const specs = [_]TestEntrySpec{
        .{ .name = "model.embed_tokens.weight", .checkpoint_bytes = 100, .role = .token_embeddings, .weight_id = "token_embeddings" },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .checkpoint_bytes = 10, .role = .decoder_layer, .layer_index = 0, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "model.layers.1.self_attn.q_proj.weight", .checkpoint_bytes = 20, .role = .decoder_layer, .layer_index = 1, .weight_id = "self_attn.q_proj.weight" },
        .{ .name = "unknown.weight", .checkpoint_bytes = 9, .role = .unclassified_global, .status = .unclassified },
    };
    var manifest = try testManifest(allocator, 2, &specs);
    defer manifest.deinit();
    var config = testConfig(2);

    try std.testing.expectError(error.UnclassifiedGlobalNotAllowed, buildStagePlan(allocator, planRequest(2, &.{1}, &config, &manifest)));

    var no_owner = planRequest(2, &.{1}, &config, &manifest);
    no_owner.allow_unclassified_global = true;
    try std.testing.expectError(error.MissingRoleOwner, buildStagePlan(allocator, no_owner));

    var request = no_owner;
    request.role_owner_overrides = &.{.{ .role = .unclassified_global, .stage_id = 1 }};
    try std.testing.expectError(error.MissingStageDependency, buildStagePlan(allocator, request));

    request.dependency_overrides = &.{.{
        .source_stage_id = 1,
        .target_stage_id = 0,
        .role = .unclassified_global,
        .reason = .unclassified_global,
        .affects_loader_residency = true,
    }};
    var plan = try buildStagePlan(allocator, request);
    defer plan.deinit();
    try std.testing.expectEqual(@as(usize, 1), plan.diagnostics.len);
    try std.testing.expect((try plan.stageLoadRequest(0)).roles.include_unclassified_global);
    try std.testing.expect((try plan.stageLoadRequest(1)).roles.include_unclassified_global);
}

test "stage_plan buildStagePlan rejects prohibited direct role owner overrides" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    var request = planRequest(4, &.{2}, &config, &manifest);
    request.role_owner_overrides = &.{.{ .role = .token_embeddings, .stage_id = 1 }};
    try std.testing.expectError(error.UnsupportedRoleOwnerOverride, buildStagePlan(allocator, request));
}

test "stage_plan stageLoadRequest planned residency equals manifest-derived residency" {
    const allocator = std.testing.allocator;
    var manifest = try standardManifest(allocator, 4);
    defer manifest.deinit();
    var config = testConfig(4);
    config.tie_word_embeddings = true;
    var plan = try buildStagePlan(allocator, planRequest(4, &.{ 1, 3 }, &config, &manifest));
    defer plan.deinit();

    for (plan.stages) |stage_entry| {
        const request = try plan.stageLoadRequest(stage_entry.id);
        const report = try manifest.stageResidencyReport(request.roles.toResidencyRequest(request.range()));
        try std.testing.expect(stageResidencyReportEql(report, stage_entry.residency));
        try std.testing.expectEqual(request.layer_start, report.layer_start);
        try std.testing.expectEqual(request.layer_end, report.layer_end);
        if (stage_entry.id == 0) {
            try std.testing.expectEqual(@as(usize, 100), stage_entry.residency.bytesForRole(.token_embeddings));
        }
        if (stage_entry.id == 2) {
            try std.testing.expectEqual(@as(usize, 0), stage_entry.residency.bytesForRole(.token_embeddings));
            try std.testing.expectEqual(@as(usize, 100), stage_entry.residency.bytesForRole(.lm_head));
        }
    }
}
