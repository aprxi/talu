//! QKV projection routing for the CUDA inference backend.

const fused = @import("fused.zig");
const linear = @import("../linear/root.zig");
const tryFusedQkvForward = fused.tryFusedQkvForward;
const tryFusedNvfp4QkvLtForward = fused.tryFusedNvfp4QkvLtForward;
const linearForwardRows = linear.linearForwardRows;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/root.zig");
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const ProjectionPath = engine_types.ProjectionPath;
const Nvfp4RouteKind = engine_types.Nvfp4RouteKind;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

const models = @import("models_pkg");
const layer_ops = models.layer_ops;

// --- Utility functions from engine_weights.zig ---
const engine_weights = @import("../../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

fn recordNvfp4Route(self: anytype, comptime kind: Nvfp4RouteKind) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_route_counters")) {
        self.nvfp4_route_counters.record(kind);
    }
}

fn recordPhaseLinearNs(self: anytype, elapsed_ns: u64) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordLinear(elapsed_ns);
    }
}

fn recordPhaseQkvPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordQkv(path);
    }
}

fn recordPhaseGateUpPath(self: anytype, path: ProjectionPath) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordGateUp(path);
    }
}

fn phaseEventTimingEnabled(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "phase_event_timing_enabled")) {
        return self.phase_event_timing_enabled;
    }
    return false;
}

fn tryPrepareNvfp4LtInput(
    self: anytype,
    input: *const compute.cuda.Buffer,
    in_dim: usize,
    rows: usize,
    input_fp4_out: *compute.cuda.Buffer,
    input_scales_out: *compute.cuda.Buffer,
) !bool {
    const quant_fn = self.quantize_f32_to_nvfp4_function orelse return false;
    const packed_in_cols = (in_dim + 1) / 2;
    const input_fp4_bytes = std.math.mul(usize, rows, packed_in_cols) catch return false;
    const input_scale_bytes = engine_types.Nvfp4LinearWeight.cublasLtScaleTensorSize(in_dim, rows);
    if (self.runtime_buffers.activation_u16_dev.size < input_fp4_bytes) return false;
    if (self.runtime_buffers.dequant_f16_dev.size < input_scale_bytes) return false;

    input_fp4_out.* = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_fp4_bytes) catch return false;
    input_scales_out.* = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, input_scale_bytes) catch return false;

    const padded_outer: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(rows, 128));
    const sf_k = (in_dim + 15) / 16;
    const padded_sf_k: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(sf_k, 4));
    const quant_grid_x = std.math.cast(u32, sf_k) orelse return false;

    self.kernel_arg_pack.reset();
    self.kernel_arg_pack.appendBufferPtr(input) catch return false;
    self.kernel_arg_pack.appendBufferPtr(input_fp4_out) catch return false;
    self.kernel_arg_pack.appendBufferPtr(input_scales_out) catch return false;
    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch return false;
    self.kernel_arg_pack.appendScalar(u32, @intCast(rows)) catch return false;
    self.kernel_arg_pack.appendScalar(u32, padded_outer) catch return false;
    self.kernel_arg_pack.appendScalar(u32, padded_sf_k) catch return false;
    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
        .grid_x = quant_grid_x,
        .grid_y = padded_outer,
        .block_x = 32,
    }, &self.kernel_arg_pack, .other) catch return false;

    return true;
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
    const q_bytes = std.math.mul(usize, rows, q_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    const k_bytes = std.math.mul(usize, rows, k_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    const v_bytes = std.math.mul(usize, rows, v_proj.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
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

    // Fused I8 QKV prefill: single GEMM with concatenated I8 weights.
    // When the full batch doesn't fit in scratch, process in row-chunks.
    if (self.active_qkv_concat) |concat| {
        if (rows > 1 and self.i8_blas_supported) fused_i8_qkv: {
            const quant_fn = self.quantize_f32_to_i8_simple_function orelse break :fused_i8_qkv;
            const split_fn = self.dequant_i32_scales_split3_function orelse break :fused_i8_qkv;

            const in_dim = q_proj.rows();
            const q_dim: usize = concat.dims[0];
            const k_dim: usize = concat.dims[1];
            const v_dim: usize = concat.dims[2];
            const total_out_dim: usize = q_dim + k_dim + v_dim;

            // Max rows per chunk from available scratch buffers.
            const act_per_row = in_dim + @sizeOf(f32);
            const i32_per_row = std.math.mul(usize, total_out_dim, @sizeOf(i32)) catch break :fused_i8_qkv;
            const max_chunk = @min(
                self.runtime_buffers.activation_u16_dev.size / act_per_row,
                self.runtime_buffers.dequant_f16_dev.size / i32_per_row,
            );
            if (max_chunk == 0) break :fused_i8_qkv;

            var done: usize = 0;
            while (done < rows) {
                const chunk = @min(rows - done, max_chunk);
                const i8_input_bytes = chunk * in_dim;
                const row_scales_bytes = chunk * @sizeOf(f32);
                const i32_out_bytes = std.math.mul(usize, chunk * total_out_dim, @sizeOf(i32)) catch break :fused_i8_qkv;

                var i8_input_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, i8_input_bytes) catch break :fused_i8_qkv;
                var row_scales_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, i8_input_bytes, row_scales_bytes) catch break :fused_i8_qkv;
                var i32_out_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, i32_out_bytes) catch break :fused_i8_qkv;

                const in_off = done * in_dim * @sizeOf(f32);
                var chunk_input = bufferSlice(input, in_off, chunk * in_dim * @sizeOf(f32)) catch break :fused_i8_qkv;
                var chunk_q = bufferSlice(&q_out, done * q_dim * @sizeOf(f32), chunk * q_dim * @sizeOf(f32)) catch break :fused_i8_qkv;
                var chunk_k = bufferSlice(&k_out, done * k_dim * @sizeOf(f32), chunk * k_dim * @sizeOf(f32)) catch break :fused_i8_qkv;
                var chunk_v = bufferSlice(&v_out, done * v_dim * @sizeOf(f32), chunk * v_dim * @sizeOf(f32)) catch break :fused_i8_qkv;

                // Step 1: Quantize F32 input → I8 + per-row scales.
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&chunk_input) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&i8_input_buf) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :fused_i8_qkv;
                compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                    .grid_x = @intCast(chunk),
                    .block_x = 256,
                }, &self.kernel_arg_pack, .other) catch break :fused_i8_qkv;

                // Step 2: I8 GEMM — I8 input × concat I8 QKV weights → I32.
                self.blas.matmulI8I8I32(
                    &self.device,
                    &i8_input_buf,
                    chunk,
                    in_dim,
                    &concat.i8_buf,
                    total_out_dim,
                    &i32_out_buf,
                ) catch {
                    self.i8_blas_supported = false;
                    break :fused_i8_qkv;
                };

                // Step 3: Dequant I32 → split F32 into Q/K/V outputs.
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&i32_out_buf) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&concat.scales_buf) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&chunk_q) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&chunk_k) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendBufferPtr(&chunk_v) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendScalar(u32, @intCast(chunk)) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendScalar(u32, concat.dims[0]) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendScalar(u32, concat.dims[1]) catch break :fused_i8_qkv;
                self.kernel_arg_pack.appendScalar(u32, concat.dims[2]) catch break :fused_i8_qkv;
                const split_blocks_x_usize = std.math.divCeil(usize, total_out_dim, 256) catch break :fused_i8_qkv;
                const split_blocks_x = std.math.cast(u32, split_blocks_x_usize) orelse break :fused_i8_qkv;
                const split_rows = std.math.cast(u32, chunk) orelse break :fused_i8_qkv;
                compute.cuda.launch.launchWithFamily(&self.device, split_fn, .{
                    .grid_x = split_blocks_x,
                    .grid_y = split_rows,
                    .block_x = 256,
                }, &self.kernel_arg_pack, .other) catch break :fused_i8_qkv;

                done += chunk;
            }
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
