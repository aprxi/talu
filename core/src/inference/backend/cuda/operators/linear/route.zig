//! Inference wrapper for generic CUDA linear projection execution.

const std = @import("std");
const compute = @import("compute_pkg");
const log = @import("log_pkg");

const engine_types = @import("../../runtime/root.zig");
const LinearWeight = engine_types.LinearWeight;
const Nvfp4RouteKind = engine_types.Nvfp4RouteKind;

const cuda_linear = compute.cuda.linear;

fn recordNvfp4Route(self: anytype, comptime kind: Nvfp4RouteKind) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_route_counters")) {
        self.nvfp4_route_counters.record(kind);
    }
}

fn recordNvfp4LinearRoute(self: anytype, kind: cuda_linear.Nvfp4LinearRouteKind) void {
    switch (kind) {
        .native_cublaslt => recordNvfp4Route(self, .native_cublaslt),
        .small_rows_nvfp4_matvec => recordNvfp4Route(self, .small_rows_nvfp4_matvec),
        .small_rows_i8_matvec => recordNvfp4Route(self, .small_rows_i8_matvec),
    }
}

fn recordPhaseLinearNs(self: anytype, elapsed_ns: u64) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordLinear(elapsed_ns);
    }
}

fn phaseEventTimingEnabled(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "phase_event_timing_enabled")) {
        return self.phase_event_timing_enabled;
    }
    return false;
}

fn makeLinearContext(
    self: anytype,
    diagnostics: *cuda_linear.Diagnostics,
    report_nvfp4_native_single_row_skip: bool,
) cuda_linear.Context {
    const blas_lt = if (self.blas_lt) |*handle| handle else null;
    return .{
        .device = &self.device,
        .kernel_arg_pack = &self.kernel_arg_pack,
        .blas = &self.blas,
        .blas_lt = blas_lt,
        .workspace = .{
            .activation_scratch = self.runtime_buffers.activation_u16_dev,
            .auxiliary_scratch = self.runtime_buffers.dequant_f16_dev,
        },
        .diagnostics = diagnostics,
        .matmul_f16_function = self.matmul_f16_function,
        .matmul_bf16_function = self.matmul_bf16_function,
        .matvec_f16_function = self.matvec_f16_function,
        .matvec_bf16_function = self.matvec_bf16_function,
        .cast_f32_to_f16_function = self.cast_f32_to_f16_function,
        .cast_f32_to_bf16_function = self.cast_f32_to_bf16_function,
        .gaffine_u4_matvec_function = self.gaffine_u4_matvec_function,
        .gaffine_u4_matvec_tile8_function = self.gaffine_u4_matvec_tile8_function,
        .gaffine_u8_matvec_function = self.gaffine_u8_matvec_function,
        .gaffine_u4_dequant_f16_function = self.gaffine_u4_dequant_f16_function,
        .gaffine_u8_dequant_f16_function = self.gaffine_u8_dequant_f16_function,
        .quantize_f32_to_i8_simple_function = self.quantize_f32_to_i8_simple_function,
        .dequant_i32_scales_function = self.dequant_i32_scales_function,
        .gaffine_u4_to_i8_function = self.gaffine_u4_to_i8_function,
        .i8_matvec_function = self.i8_matvec_function,
        .fp8_matvec_function = self.fp8_matvec_function,
        .fp8_matvec_tile8_function = self.fp8_matvec_tile8_function,
        .fp8_dequant_to_bf16_function = self.fp8_dequant_to_bf16_function,
        .quantize_f32_to_fp8_function = self.quantize_f32_to_fp8_function,
        .scale_rows_f32_function = self.scale_rows_f32_function,
        .mxfp8_matvec_function = self.mxfp8_matvec_function,
        .mxfp8_matvec_tile8_function = self.mxfp8_matvec_tile8_function,
        .quantize_f32_to_mxfp8_function = self.quantize_f32_to_mxfp8_function,
        .mxfp8_dequant_to_bf16_function = self.mxfp8_dequant_to_bf16_function,
        .nvfp4_matvec_function = self.nvfp4_matvec_function,
        .nvfp4_matvec_tile8_function = self.nvfp4_matvec_tile8_function,
        .quantize_f32_to_nvfp4_function = self.quantize_f32_to_nvfp4_function,
        .u16_blas_f16_supported = self.u16_blas_f16_supported,
        .u16_blas_bf16_supported = self.u16_blas_bf16_supported,
        .i8_blas_supported = self.i8_blas_supported,
        .fp8_blas_supported = self.fp8_blas_supported,
        .gaffine_multi_row_matvec_supported = self.gaffine_sequence_rows_supported,
        .nvfp4_multi_row_matvec_supported = self.nvfp4_sequence_rows_supported,
        .gaffine_u4_tile8_enabled = self.gaffine_u4_tile8_enabled,
        .gaffine_u4_small_batch_i8_gemm_enabled = self.gaffine_u4_decode_i8_enabled,
        .max_batch_size = self.max_batch_size,
        .report_nvfp4_native_single_row_skip = report_nvfp4_native_single_row_skip,
    };
}

fn syncLinearCapabilityFlags(self: anytype, ctx: *const cuda_linear.Context) void {
    self.u16_blas_f16_supported = ctx.u16_blas_f16_supported;
    self.u16_blas_bf16_supported = ctx.u16_blas_bf16_supported;
    self.i8_blas_supported = ctx.i8_blas_supported;
    self.fp8_blas_supported = ctx.fp8_blas_supported;
}

fn emitLinearDiagnostics(
    self: anytype,
    rows: usize,
    weight: *const LinearWeight,
    diagnostics: *const cuda_linear.Diagnostics,
) void {
    if (diagnostics.mxfp8_lt_error) |err| {
        log.err("inference", "MXFP8 cuBLASLt failed", .{ .err = @errorName(err) }, @src());
    }
    if (diagnostics.nvfp4_native_quant_error) |err| {
        log.warn("inference", "NVFP4 native FAIL: quant launch", .{ .err = @errorName(err) });
    }
    if (diagnostics.nvfp4_native_single_row_skipped) {
        switch (weight.*) {
            .nvfp4 => |w| log.warn("inference", "CUDA NVFP4 native single-row skipped", .{
                .rows = rows,
                .in_dim = w.rows,
                .out_dim = w.cols,
                .has_scales_lt = @as(u8, @intFromBool(w.scales_lt_buffer.size > 0)),
                .has_quant_fn = @as(u8, @intFromBool(self.quantize_f32_to_nvfp4_function != null)),
                .has_blas_lt = @as(u8, @intFromBool(self.blas_lt != null)),
            }),
            else => {},
        }
    }
    if (diagnostics.nvfp4_matvec_unavailable) {
        log.warn("inference", "CUDA NVFP4 matvec kernel unavailable", .{});
    }
    if (diagnostics.nvfp4_route) |route| {
        recordNvfp4LinearRoute(self, route);
    }
}

pub fn linearForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    weight: *const LinearWeight,
    out: *compute.cuda.Buffer,
) !void {
    return linearForwardRows(self, input, try cuda_linear.bufferF32RowCount(input, weight.rows()), weight, out);
}

pub fn linearForwardRows(
    self: anytype,
    input: *const compute.cuda.Buffer,
    rows: usize,
    weight: *const LinearWeight,
    out: *compute.cuda.Buffer,
) !void {
    const SelfType = @TypeOf(self.*);
    const linear_start_ns: i128 = std.time.nanoTimestamp();
    var linear_event_timing_active = false;
    if (comptime @hasField(SelfType, "phase_linear_start_event")) {
        if (phaseEventTimingEnabled(self)) {
            if (self.phase_linear_start_event) |start_evt| {
                if (self.phase_linear_stop_event != null) {
                    self.device.recordEvent(start_evt, self.compute_stream) catch {};
                    linear_event_timing_active = true;
                }
            }
        }
    }
    defer {
        var elapsed_ns: u64 = 0;
        if (comptime @hasField(SelfType, "phase_linear_start_event")) {
            if (linear_event_timing_active) {
                if (self.phase_linear_start_event) |start_evt| {
                    if (self.phase_linear_stop_event) |stop_evt| {
                        self.device.recordEvent(stop_evt, self.compute_stream) catch {};
                        self.device.synchronizeEvent(stop_evt) catch {};
                        elapsed_ns = self.device.elapsedEventNs(start_evt, stop_evt) catch 0;
                    }
                }
            }
        }
        if (elapsed_ns == 0) {
            const elapsed_i128 = std.time.nanoTimestamp() - linear_start_ns;
            elapsed_ns = if (elapsed_i128 > 0) @intCast(elapsed_i128) else 0;
        }
        recordPhaseLinearNs(self, elapsed_ns);
    }

    if (rows == 0) return error.InvalidArgument;
    const input_row_width = weight.rows();
    const output_row_width = weight.cols();
    const input_row_bytes = std.math.mul(usize, input_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const output_row_bytes = std.math.mul(usize, output_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_input_bytes = std.math.mul(usize, rows, input_row_bytes) catch return error.InvalidArgument;
    const packed_output_bytes = std.math.mul(usize, rows, output_row_bytes) catch return error.InvalidArgument;
    if (input.size < packed_input_bytes or out.size < packed_output_bytes) {
        return error.InvalidInstructionBinding;
    }

    const report_nvfp4_native_single_row_skip = switch (weight.*) {
        .nvfp4 => |w| rows == 1 and w.cols >= 65536 and @import("env_pkg").getenv("TALU_CUDA_NVFP4_NATIVE_DEBUG") != null,
        else => false,
    };
    var diagnostics = cuda_linear.Diagnostics{};
    var ctx = makeLinearContext(self, &diagnostics, report_nvfp4_native_single_row_skip);

    // Canonical path: residual add is always handled by the explicit residual op.
    // Do not fuse residual behavior into projection kernels.
    self.pending_residual_add_buf = null;

    cuda_linear.executeRows(&ctx, input, rows, weight, out) catch |err| {
        syncLinearCapabilityFlags(self, &ctx);
        emitLinearDiagnostics(self, rows, weight, &diagnostics);
        return switch (err) {
            error.UnsupportedLinearRowCount => error.UnsupportedModel,
            else => err,
        };
    };
    syncLinearCapabilityFlags(self, &ctx);
    emitLinearDiagnostics(self, rows, weight, &diagnostics);
}

pub const linearWeightHasI8Cache = cuda_linear.linearWeightHasI8Cache;
