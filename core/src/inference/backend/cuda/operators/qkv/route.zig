//! QKV projection routing for the CUDA inference backend.

const fused = @import("fused.zig");
const linear = @import("../linear/root.zig");
const tryFusedQkvForward = fused.tryFusedQkvForward;
const tryFusedNvfp4QkvLtForward = fused.tryFusedNvfp4QkvLtForward;
const tryFusedQkvI8ConcatForward = fused.tryFusedQkvI8ConcatForward;
const linearForwardRows = linear.linearForwardRows;

const std = @import("std");
const compute = @import("compute_pkg");
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/root.zig");
const LinearWeight = engine_types.LinearWeight;
const ProjectionPath = engine_types.ProjectionPath;

// --- Utility functions from engine_weights.zig ---
const engine_weights = @import("../../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

fn recordPhaseQkvPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordQkv(path);
    }
}

pub fn runQkvProjection(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !ProjectionPath {
    const q_row_bytes = std.math.mul(usize, q_proj.cols(), @sizeOf(f32)) catch return error.InvalidArgument;
    const k_row_bytes = std.math.mul(usize, k_proj.cols(), @sizeOf(f32)) catch return error.InvalidArgument;
    const v_row_bytes = std.math.mul(usize, v_proj.cols(), @sizeOf(f32)) catch return error.InvalidArgument;
    const q_bytes = std.math.mul(usize, rows, q_row_bytes) catch return error.InvalidArgument;
    const k_bytes = std.math.mul(usize, rows, k_row_bytes) catch return error.InvalidArgument;
    const v_bytes = std.math.mul(usize, rows, v_row_bytes) catch return error.InvalidArgument;
    const prefer_nvfp4_lt_fused = rows > 1 and switch (q_proj.*) {
        .nvfp4 => switch (k_proj.*) {
            .nvfp4 => switch (v_proj.*) {
                .nvfp4 => true,
                else => false,
            },
            else => false,
        },
        else => false,
    };
    // Keep the gate broad: kernel-specific fusion guards live in tryFusedQkvForward.
    // Some fused kernels support asymmetric Q/K/V widths, so requiring equal cols
    // here can incorrectly force decode onto 3x unfused matvec paths.
    const allow_fused_qkv = self.loaded.config.hidden_size_per_layer_input <= 0;
    const prefer_i8_concat = self.active_qkv_concat != null and rows > 1 and self.i8_blas_supported;
    if (!prefer_nvfp4_lt_fused and allow_fused_qkv and !prefer_i8_concat and q_out_dest.size >= q_bytes) {
        const fused_ok = tryFusedQkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest) catch |err| blk: {
            if (err == error.CudaKernelLaunchFailed) {
                log.warn("inference", "CUDA fused QKV launch failed; using unfused projections", .{
                    .rows = rows,
                    .q_dim = q_proj.cols(),
                    .k_dim = k_proj.cols(),
                    .v_dim = v_proj.cols(),
                });
                break :blk false;
            }
            return err;
        };
        if (fused_ok) {
            recordPhaseQkvPath(self, .fused);
            return .fused;
        }
    }
    var q_out = if (q_out_dest.size == q_bytes)
        q_out_dest.*
    else
        try bufferSlice(q_out_dest, 0, q_bytes);
    var k_out = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, k_bytes);
    var v_out = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, v_bytes);

    if (try tryFusedNvfp4QkvLtForward(self, input, q_proj, k_proj, v_proj, rows, &q_out, &k_out, &v_out)) {
        recordPhaseQkvPath(self, .fused);
        return .fused;
    }

    // Fused I8 prefill: single GEMM with concatenated I8 weights.
    if (self.active_qkv_concat) |concat| {
        if (try tryFusedQkvI8ConcatForward(self, input, rows, q_proj.rows(), concat, &q_out, &k_out, &v_out)) {
            recordPhaseQkvPath(self, .unfused);
            return .unfused;
        }
    }

    // If concat-I8 path is unavailable/failed, retry fused QKV kernel before
    // using three separate projections.
    if (allow_fused_qkv and q_out_dest.size >= q_bytes) {
        const fused_ok_retry = tryFusedQkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest) catch |err| blk: {
            if (err == error.CudaKernelLaunchFailed) {
                log.warn("inference", "CUDA fused QKV retry launch failed; using unfused projections", .{
                    .rows = rows,
                    .q_dim = q_proj.cols(),
                    .k_dim = k_proj.cols(),
                    .v_dim = v_proj.cols(),
                });
                break :blk false;
            }
            return err;
        };
        if (fused_ok_retry) {
            recordPhaseQkvPath(self, .fused);
            return .fused;
        }
    }

    linearForwardRows(self, input, rows, q_proj, &q_out) catch |err| {
        log.warn("inference", "CUDA Q projection failed in runQkvProjection", .{
            .rows = rows,
            .in_dim = q_proj.rows(),
            .out_dim = q_proj.cols(),
            .reason = @errorName(err),
        });
        return err;
    };
    linearForwardRows(self, input, rows, k_proj, &k_out) catch |err| {
        log.warn("inference", "CUDA K projection failed in runQkvProjection", .{
            .rows = rows,
            .in_dim = k_proj.rows(),
            .out_dim = k_proj.cols(),
            .reason = @errorName(err),
        });
        return err;
    };
    linearForwardRows(self, input, rows, v_proj, &v_out) catch |err| {
        log.warn("inference", "CUDA V projection failed in runQkvProjection", .{
            .rows = rows,
            .in_dim = v_proj.rows(),
            .out_dim = v_proj.cols(),
            .reason = @errorName(err),
        });
        return err;
    };
    recordPhaseQkvPath(self, .unfused);
    return .unfused;
}
