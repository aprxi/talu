//! CUDA local-stage plan resolution for scheduler execution-target initialization.

const std = @import("std");

const backend = @import("../backend/root.zig");
const compute = @import("compute_pkg");
const device_summary = @import("device_summary.zig");
const log = @import("log_pkg");
const models = @import("models_pkg");
const topology = @import("../pipeline/local_pipeline_topology.zig");

const has_cuda = backend.has_cuda;

pub const ResolvedCudaLocalStagePlan = struct {
    plan: topology.LocalStagePlan,
    summary: device_summary.DeviceLayerSummary,
    device_count: usize,
};

pub fn resolveCudaLocalStagePlan(
    allocator: std.mem.Allocator,
    loaded: *models.LoadedModel,
    local_stage_override: ?[]const topology.LocalStageSpec,
) !ResolvedCudaLocalStagePlan {
    var plan = try topology.resolveLocalStagePlan(local_stage_override, loaded.blocks.len, 0);
    const explicit_stage_list = if (local_stage_override == null)
        std.process.getEnvVarOwned(allocator, "TALU_LOCAL_STAGES") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => {
                log.err("inference", "Failed to read TALU_LOCAL_STAGES", .{ .err = @errorName(err) }, @src());
                return err;
            },
        }
    else
        null;
    defer if (explicit_stage_list) |value| allocator.free(value);
    const has_explicit_stage_list = explicit_stage_list != null;
    if (explicit_stage_list) |raw| {
        var specs = topology.parseLocalStageSpecs(allocator, raw, loaded.blocks.len) catch |err| {
            log.err("inference", "Invalid TALU_LOCAL_STAGES", .{
                .value = std.mem.trim(u8, raw, " \t\r\n"),
                .expected = "backend[@device]:start..end,...",
            }, @src());
            return err;
        };
        defer specs.deinit();
        plan = try topology.localStagePlanFromSpecs(specs.stages);
    }

    const explicit_cpu_layers = if (local_stage_override == null and !has_explicit_stage_list)
        std.process.getEnvVarOwned(allocator, "TALU_CPU_LAYERS") catch |err| switch (err) {
            error.EnvironmentVariableNotFound => null,
            else => {
                log.err("inference", "Failed to read TALU_CPU_LAYERS", .{ .err = @errorName(err) }, @src());
                return err;
            },
        }
    else
        null;
    defer if (explicit_cpu_layers) |value| allocator.free(value);
    const has_explicit_cpu_layers = explicit_cpu_layers != null;
    if (explicit_cpu_layers) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        const cpu_layers = std.fmt.parseUnsigned(usize, trimmed, 10) catch |err| {
            log.err("inference", "Invalid TALU_CPU_LAYERS", .{
                .value = trimmed,
                .err = @errorName(err),
            }, @src());
            return error.InvalidTopologyConfig;
        };
        plan = topology.cpuPrefixCudaLocalStagePlan(loaded.blocks.len, cpu_layers, 0) catch |err| {
            log.err("inference", "Invalid TALU_CPU_LAYERS", .{
                .value = trimmed,
                .total_layers = loaded.blocks.len,
                .err = @errorName(err),
            }, @src());
            return err;
        };
    }

    // Local stage plan resolution priority:
    //   1. programmatic stage list
    //   2. TALU_LOCAL_STAGES stage list
    //   3. TALU_CPU_LAYERS CPU-prefix shorthand
    //   4. Auto-detection (probe GPU memory, estimate model size)
    if (local_stage_override == null and !has_explicit_stage_list and !has_explicit_cpu_layers) {
        plan = topology.autoDetectLocalStagePlanForModel(allocator, loaded) catch |err| {
            log.err("inference", "Auto local stage detection failed", .{
                .err = @errorName(err),
            }, @src());
            return err;
        };
    }

    if (local_stage_override == null and
        !has_explicit_stage_list and
        !has_explicit_cpu_layers and
        plan.stage_count > plan.cudaStageCount() and
        topology.loadedModelHasPackedNvfp4Weights(loaded))
    {
        const primary_device = plan.primaryDeviceOrdinal();
        log.warn("inference", "CUDA auto local stage plan disabled CPU stages for packed NVFP4 model", .{
            .requested_stage_count = plan.stage_count,
            .device = primary_device,
        });
        plan = try topology.defaultCudaLocalStagePlan(loaded.blocks.len, primary_device);
    }

    const device_count = if (has_cuda)
        compute.cuda.Device.deviceCount() catch |err| {
            log.err("inference", "Failed to query CUDA device count for local stage validation", .{
                .err = @errorName(err),
            }, @src());
            return err;
        }
    else
        0;
    topology.validateLocalStagePlan(plan, loaded.blocks.len, device_count) catch |err| switch (err) {
        error.LocalStageDeviceOrdinalOutOfRange => {
            log.err("inference", "Local stage CUDA device ordinal out of range", .{
                .device_count = device_count,
                .stage_count = plan.stage_count,
            }, @src());
            return error.CudaInvalidDevice;
        },
        error.LocalStageUnsupportedBackend => {
            log.err("inference", "Local stage backend is not supported by the local pipeline runtime", .{
                .stage_count = plan.stage_count,
            }, @src());
            return error.UnsupportedModel;
        },
        else => {
            log.err("inference", "Invalid local stage plan", .{
                .err = @errorName(err),
                .total_layers = loaded.blocks.len,
                .stage_count = plan.stage_count,
            }, @src());
            return error.InvalidTopologyConfig;
        },
    };

    return .{
        .plan = plan,
        .summary = device_summary.fromLocalStagePlan(plan),
        .device_count = device_count,
    };
}

test "resolveCudaLocalStagePlan accepts explicit CPU-only plan without CUDA device probing" {
    if (has_cuda) return error.SkipZigTest;

    var blocks: [2]models.runtime_blocks.LayerWeights = undefined;
    var loaded: models.LoadedModel = undefined;
    loaded.blocks = blocks[0..];

    const resolved = try resolveCudaLocalStagePlan(std.testing.allocator, &loaded, &.{.{
        .backend_kind = .cpu,
        .layer_start = 0,
        .layer_end = 2,
    }});

    try std.testing.expectEqual(@as(usize, 1), resolved.plan.stage_count);
    try std.testing.expectEqual(@as(usize, 2), resolved.summary.cpu_layers);
    try std.testing.expectEqual(@as(usize, 0), resolved.device_count);
}
