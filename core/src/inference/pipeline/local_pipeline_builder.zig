//! Pipeline-owned construction of local pipeline runtimes.
//!
//! Concrete backend modules are provided through a small factory callback that
//! creates one requested stage. This builder owns the ordered stage list, stage
//! input/capability descriptors, and transfer of those handles into
//! `LocalPipelineRuntime`.

const std = @import("std");

const host_capability = @import("host_capability.zig");
const local_pipeline_runtime = @import("local_pipeline_runtime.zig");
const topology = @import("local_pipeline_topology.zig");
const models = @import("models_pkg");

const Allocator = std.mem.Allocator;

pub const StageFactoryRequest = struct {
    allocator: Allocator,
    loaded: *models.LoadedModel,
    stage_id: usize,
    stage_count: usize,
    stage: topology.LocalStageSpec,
    host_kind: host_capability.HostBackendKind,
    max_batch_size: usize,
    cpu_max_sequence_len: usize,
    primary_cuda_device: usize,
};

pub const StageFactory = struct {
    context: *anyopaque,
    create: *const fn (*anyopaque, StageFactoryRequest) anyerror!local_pipeline_runtime.StageHandle,
};

pub const InitRequest = struct {
    allocator: Allocator,
    loaded: *models.LoadedModel,
    plan: topology.LocalStagePlan,
    max_batch_size: usize,
    cpu_max_sequence_len: usize,
    load_semantics: models.stage_plan.LoadSemantics,
    factory: StageFactory,
};

pub fn initLocalPipelineRuntime(request: InitRequest) !local_pipeline_runtime.LocalPipelineRuntime {
    if (request.plan.stage_count < 2) return error.InvalidTopologyConfig;

    const stage_count = request.plan.stage_count;
    var handles = try request.allocator.alloc(local_pipeline_runtime.StageHandle, stage_count);
    errdefer request.allocator.free(handles);
    var created_count: usize = 0;
    errdefer {
        for (handles[0..created_count]) |*handle| handle.deinit(request.allocator);
    }

    var stage_inputs = try request.allocator.alloc(local_pipeline_runtime.StageInputSpec, stage_count);
    defer request.allocator.free(stage_inputs);
    var stage_capabilities = try request.allocator.alloc(local_pipeline_runtime.StageRuntimeCapability, stage_count);
    defer request.allocator.free(stage_capabilities);

    const primary_cuda_device = request.plan.primaryDeviceOrdinal();
    for (request.plan.stagesSlice(), 0..) |stage, stage_id| {
        const host_kind = topology.hostBackendKind(stage.backend_kind);
        stage_inputs[stage_id] = .{
            .backend_kind = host_kind,
            .layer_start = stage.layer_start,
            .layer_end = stage.layer_end,
        };

        handles[stage_id] = try request.factory.create(request.factory.context, .{
            .allocator = request.allocator,
            .loaded = request.loaded,
            .stage_id = stage_id,
            .stage_count = stage_count,
            .stage = stage,
            .host_kind = host_kind,
            .max_batch_size = request.max_batch_size,
            .cpu_max_sequence_len = request.cpu_max_sequence_len,
            .primary_cuda_device = primary_cuda_device,
        });
        created_count += 1;
        try validateCreatedStageHandle(&handles[stage_id], stage_id, host_kind, stage);
        stage_capabilities[stage_id] = .{
            .stage_id = stage_id,
            .backend_kind = host_kind,
            .max_batch_size = handles[stage_id].vtable.max_batch_size(handles[stage_id].ptr),
            .prefill_chunk_rows_cap = handles[stage_id].vtable.prefill_chunk_rows_cap(handles[stage_id].ptr),
            .supported_boundary_dtypes = handles[stage_id].supported_boundary_dtypes,
        };
    }

    created_count = 0;
    return local_pipeline_runtime.LocalPipelineRuntime.init(.{
        .allocator = request.allocator,
        .loaded = request.loaded,
        .d_model = @intCast(request.loaded.config.d_model),
        .vocab_size = @intCast(request.loaded.config.vocab_size),
        .total_layers = request.loaded.blocks.len,
        .stages = handles,
        .stage_inputs = stage_inputs,
        .stage_capabilities = stage_capabilities,
        .load_semantics = request.load_semantics,
    });
}

fn validateCreatedStageHandle(
    handle: *const local_pipeline_runtime.StageHandle,
    stage_id: usize,
    host_kind: host_capability.HostBackendKind,
    stage: topology.LocalStageSpec,
) !void {
    if (handle.stage_id != stage_id) return error.InvalidTopologyConfig;
    if (handle.backend_kind != host_kind) return error.InvalidTopologyConfig;
    if (handle.layer_start != stage.layer_start or handle.layer_end != stage.layer_end) {
        return error.InvalidTopologyConfig;
    }
    if (handle.supported_boundary_dtypes.len == 0) return error.InvalidTopologyConfig;
}

test "initLocalPipelineRuntime rejects single-stage plans before backend factory use" {
    const Factory = struct {
        fn create(
            _: *anyopaque,
            _: StageFactoryRequest,
        ) anyerror!local_pipeline_runtime.StageHandle {
            return error.TestUnexpectedResult;
        }
    };

    var context: u8 = 0;
    var loaded: models.LoadedModel = undefined;
    const plan = try topology.defaultCudaLocalStagePlan(1, 0);

    try std.testing.expectError(error.InvalidTopologyConfig, initLocalPipelineRuntime(.{
        .allocator = std.testing.allocator,
        .loaded = &loaded,
        .plan = plan,
        .max_batch_size = 1,
        .cpu_max_sequence_len = 1,
        .load_semantics = .{},
        .factory = .{
            .context = &context,
            .create = Factory.create,
        },
    }));
}
