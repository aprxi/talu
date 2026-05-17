//! Scheduler-owned factory for concrete stages in a local pipeline.

const std = @import("std");

const backend = @import("../backend/root.zig");
const models = @import("models_pkg");
const pipeline_builder = @import("../pipeline/local_pipeline_builder.zig");
const pipeline_runtime = @import("../pipeline/local_pipeline_runtime.zig");
const pipeline_stage_adapter = @import("../pipeline/local_pipeline_stage_adapter.zig");
const topology = @import("../pipeline/local_pipeline_topology.zig");
const progress_mod = @import("progress_pkg");

const cpu = backend.cpu;
const cuda = backend.cuda;
const has_cuda = backend.has_cuda;

pub const FactoryContext = struct {};

pub fn createStage(
    context: *anyopaque,
    request: pipeline_builder.StageFactoryRequest,
) anyerror!pipeline_runtime.StageHandle {
    _ = context;
    return switch (request.stage.backend_kind) {
        .cpu => blk: {
            const ptr = try request.allocator.create(cpu.BackendType);
            var ptr_owned = true;
            errdefer if (ptr_owned) request.allocator.destroy(ptr);
            ptr.* = try cpu.BackendType.init(request.allocator, request.loaded, .{
                .max_batch_size = request.max_batch_size,
                .max_sequence_len = request.cpu_max_sequence_len,
                .layer_range = .{ .start = request.stage.layer_start, .end = request.stage.layer_end },
                .build_logits_head = request.stage.owns_projection,
                .progress = progress_mod.Context.NONE,
            });
            ptr_owned = false;
            break :blk .{
                .stage_id = request.stage_id,
                .backend_kind = request.host_kind,
                .layer_start = request.stage.layer_start,
                .layer_end = request.stage.layer_end,
                .supported_boundary_dtypes = cpu.interface.stage_executor.supportedBoundaryDTypes(),
                .ptr = ptr,
                .vtable = pipeline_stage_adapter.stageVTable(cpu, cpu.BackendType),
            };
        },
        .cuda => if (comptime has_cuda) blk: {
            const ptr = try request.allocator.create(cuda.BackendType);
            var ptr_owned = true;
            errdefer if (ptr_owned) request.allocator.destroy(ptr);
            ptr.* = try cuda.BackendType.init(request.allocator, request.loaded, request.max_batch_size, .{
                .device_ordinal = request.stage.device_ordinal orelse request.primary_cuda_device,
                .layer_range = .{ .start = request.stage.layer_start, .end = request.stage.layer_end },
                .owns_embedding = request.stage.owns_embedding,
                .owns_projection = request.stage.owns_projection,
                .progress = progress_mod.Context.NONE,
            });
            ptr_owned = false;
            break :blk .{
                .stage_id = request.stage_id,
                .backend_kind = request.host_kind,
                .layer_start = request.stage.layer_start,
                .layer_end = request.stage.layer_end,
                .supported_boundary_dtypes = cuda.interface.stage_executor.supportedBoundaryDTypes(),
                .ptr = ptr,
                .vtable = pipeline_stage_adapter.stageVTable(cuda, cuda.BackendType),
            };
        } else return error.CudaNotEnabled,
        .metal => return error.UnsupportedModel,
    };
}

pub fn init(
    allocator: std.mem.Allocator,
    loaded: *models.LoadedModel,
    plan: topology.LocalStagePlan,
    max_batch_size: usize,
    cpu_max_sequence_len: usize,
    load_semantics: models.stage_plan.LoadSemantics,
) !pipeline_runtime.LocalPipelineRuntime {
    var factory_context = FactoryContext{};
    return pipeline_builder.initLocalPipelineRuntime(.{
        .allocator = allocator,
        .loaded = loaded,
        .plan = plan,
        .max_batch_size = max_batch_size,
        .cpu_max_sequence_len = cpu_max_sequence_len,
        .load_semantics = load_semantics,
        .factory = .{
            .context = &factory_context,
            .create = createStage,
        },
    });
}

test "createStage rejects metal local stages" {
    var loaded: models.LoadedModel = undefined;
    var context = FactoryContext{};
    try std.testing.expectError(error.UnsupportedModel, createStage(&context, .{
        .allocator = std.testing.allocator,
        .loaded = &loaded,
        .stage_id = 0,
        .stage_count = 1,
        .stage = .{
            .backend_kind = .metal,
            .device_ordinal = 0,
            .layer_start = 0,
            .layer_end = 1,
            .owns_embedding = true,
            .owns_projection = true,
        },
        .host_kind = .metal,
        .max_batch_size = 1,
        .cpu_max_sequence_len = 1,
        .primary_cuda_device = 0,
    }));
}

test "init rejects single-stage local pipeline before factory use" {
    var loaded: models.LoadedModel = undefined;
    const plan = try topology.defaultCudaLocalStagePlan(1, 0);
    try std.testing.expectError(error.InvalidTopologyConfig, init(
        std.testing.allocator,
        &loaded,
        plan,
        1,
        1,
        .{},
    ));
}
