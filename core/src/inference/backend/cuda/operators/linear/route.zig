//! Linear projection routing for the CUDA inference backend.

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

pub fn linearForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    weight: *const LinearWeight,
    out: *compute.cuda.Buffer,
) !void {
    return linearForwardRows(self, input, try bufferF32RowCount(input, weight.rows()), weight, out);
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
    // Canonical path: residual add is always handled by the explicit residual op.
    // Do not fuse residual behavior into projection kernels.
    self.pending_residual_add_buf = null;

    var packed_input = if (input.size == packed_input_bytes)
        input.*
    else
        try bufferSlice(input, 0, packed_input_bytes);
    var packed_out = if (out.size == packed_output_bytes)
        out.*
    else
        try bufferSlice(out, 0, packed_output_bytes);

    switch (weight.*) {
        .dense_f32 => |w| {
            try self.blas.matmulF32(
                &self.device,
                &packed_input,
                rows,
                w.rows,
                &w.buffer,
                w.cols,
                &packed_out,
            );
        },
        .dense_u16 => |w| {
            // Canonical batched path: treat rows as matrix M dimension for
            // all row counts (including decode). No batch-specialized kernels.
            const matmul_kernel = switch (w.dtype) {
                .f16 => self.matmul_f16_function orelse return error.CudaKernelUnavailable,
                .bf16 => self.matmul_bf16_function orelse return error.CudaKernelUnavailable,
            };
            const matvec_kernel = switch (w.dtype) {
                .f16 => self.matvec_f16_function orelse return error.CudaKernelUnavailable,
                .bf16 => self.matvec_bf16_function orelse return error.CudaKernelUnavailable,
            };
            const blas_payload: compute.cuda.Blas.U16Payload = switch (w.dtype) {
                .f16 => .f16,
                .bf16 => .bf16,
            };
            const blas_supported = switch (w.dtype) {
                .f16 => self.u16_blas_f16_supported,
                .bf16 => self.u16_blas_bf16_supported,
            };
            // Decode path selection:
            // - f16: keep native batched GEMV for small row counts.
            // - bf16: switch to tensor-core GEMM at n>=8 to avoid leaving
            //   throughput on the table for batched decode.
            const prefer_matvec = switch (w.dtype) {
                .f16 => rows <= 32,
                .bf16 => rows < 8,
            };
            // Keep single-row decode on the custom matvec path. cuBLAS GEMV
            // launch/dispatch overhead can dominate for per-token projections.
            if (prefer_matvec) {
                try compute.cuda.matvec_u16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    matvec_kernel,
                    &packed_input,
                    &w.buffer,
                    &packed_out,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    @intCast(rows),
                    0,
                );
                return;
            }
            if (blas_supported) {
                const input_u16_count = std.math.mul(usize, rows, w.rows) catch return error.InvalidArgument;
                const input_u16_bytes = std.math.mul(usize, input_u16_count, @sizeOf(u16)) catch return error.InvalidArgument;
                if (self.runtime_buffers.activation_u16_dev.size < input_u16_bytes) return error.InvalidInstructionBinding;
                var input_u16_dev = try bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_u16_bytes);
                const cast_kernel = switch (w.dtype) {
                    .f16 => self.cast_f32_to_f16_function orelse null,
                    .bf16 => self.cast_f32_to_bf16_function orelse null,
                };
                if (cast_kernel) |cast_fn| {
                    switch (w.dtype) {
                        .f16 => try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_fn,
                            &packed_input,
                            &input_u16_dev,
                            @intCast(input_u16_count),
                        ),
                        .bf16 => try compute.cuda.cast_f32_to_bf16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_fn,
                            &packed_input,
                            &input_u16_dev,
                            @intCast(input_u16_count),
                        ),
                    }
                } else {
                    switch (w.dtype) {
                        .f16 => self.u16_blas_f16_supported = false,
                        .bf16 => self.u16_blas_bf16_supported = false,
                    }
                    try compute.cuda.matmul_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        matmul_kernel,
                        &packed_input,
                        &w.buffer,
                        &packed_out,
                        @intCast(rows),
                        @intCast(w.rows),
                        @intCast(w.cols),
                    );
                    return;
                }

                self.blas.matmulU16U16F32(
                    &self.device,
                    &input_u16_dev,
                    blas_payload,
                    rows,
                    w.rows,
                    &w.buffer,
                    blas_payload,
                    w.cols,
                    &packed_out,
                ) catch {
                    switch (w.dtype) {
                        .f16 => self.u16_blas_f16_supported = false,
                        .bf16 => self.u16_blas_bf16_supported = false,
                    }
                    try compute.cuda.matmul_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        matmul_kernel,
                        &packed_input,
                        &w.buffer,
                        &packed_out,
                        @intCast(rows),
                        @intCast(w.rows),
                        @intCast(w.cols),
                    );
                };
                return;
            }
            try compute.cuda.matmul_u16.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                matmul_kernel,
                &packed_input,
                &w.buffer,
                &packed_out,
                @intCast(rows),
                @intCast(w.rows),
                @intCast(w.cols),
            );
            return;
        },
        .gaffine_u4 => |w| {
            var kernel = self.gaffine_u4_matvec_function orelse return error.CudaKernelUnavailable;
            var use_tile8 = false;
            if (self.gaffine_u4_tile8_enabled and rows > 4) {
                if (self.gaffine_u4_matvec_tile8_function) |tile8_kernel| {
                    kernel = tile8_kernel;
                    use_tile8 = true;
                }
            }
            // Native U4 GEMV stays default for single-row decode and prefill.
            // Optional TALU_CUDA_GAFFINE_U4_DECODE_I8 reroutes batched decode
            // rows with enough parallelism (4..max_batch_size) to tensor-core
            // I8 GEMM.
            const prefer_i8_decode = self.gaffine_u4_decode_i8_enabled and
                rows >= 4 and rows <= self.max_batch_size;
            if (rows <= 32 and !prefer_i8_decode) {
                if (use_tile8) {
                    try compute.cuda.gaffine_u4_matvec.runWithFunctionTile8(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        &packed_input,
                        &w.packed_data,
                        &w.scales,
                        &w.biases,
                        &packed_out,
                        @intCast(w.rows),
                        @intCast(w.cols),
                        w.group_size,
                        w.scales_dtype_tag,
                        @intCast(rows),
                        0,
                    );
                } else {
                    try compute.cuda.gaffine_u4_matvec.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        &packed_input,
                        &w.packed_data,
                        &w.scales,
                        &w.biases,
                        &packed_out,
                        @intCast(w.rows),
                        @intCast(w.cols),
                        w.group_size,
                        w.scales_dtype_tag,
                        @intCast(rows),
                        0,
                    );
                }
                return;
            }

            // INT8 tensor core path for prefill.
            // When the full batch doesn't fit in scratch, process in
            // row-chunks that do — avoids growing dequant_f16_dev.
            if (w.dequant_i8_cache.pointer != 0 and
                w.mean_scale_cache.pointer != 0 and
                self.i8_blas_supported)
            i8_blas: {
                const quant_fn = self.quantize_f32_to_i8_simple_function orelse break :i8_blas;
                const dequant_fn = self.dequant_i32_scales_function orelse break :i8_blas;

                const in_dim = w.rows;
                const out_dim = w.cols;

                // Max rows per chunk from available scratch buffers.
                const act_per_row = in_dim + @sizeOf(f32); // I8 input + F32 row scale
                const i32_per_row = std.math.mul(usize, out_dim, @sizeOf(i32)) catch break :i8_blas;
                const max_chunk = @min(
                    self.runtime_buffers.activation_u16_dev.size / act_per_row,
                    self.runtime_buffers.dequant_f16_dev.size / i32_per_row,
                );
                if (max_chunk == 0) break :i8_blas;

                var done: usize = 0;
                while (done < rows) {
                    const chunk = @min(rows - done, max_chunk);
                    const i8_input_bytes = chunk * in_dim;
                    const row_scales_bytes = chunk * @sizeOf(f32);
                    const i32_out_bytes = std.math.mul(usize, chunk * out_dim, @sizeOf(i32)) catch break :i8_blas;

                    var i8_input_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, i8_input_bytes) catch break :i8_blas;
                    var row_scales_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, i8_input_bytes, row_scales_bytes) catch break :i8_blas;
                    var i32_out_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, i32_out_bytes) catch break :i8_blas;

                    const in_off = done * in_dim * @sizeOf(f32);
                    const out_off = done * out_dim * @sizeOf(f32);
                    var chunk_input = bufferSlice(&packed_input, in_off, chunk * in_dim * @sizeOf(f32)) catch break :i8_blas;
                    var chunk_output = bufferSlice(&packed_out, out_off, chunk * out_dim * @sizeOf(f32)) catch break :i8_blas;

                    // Step 1: Quantize F32 input → I8 + per-row scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&chunk_input) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&i8_input_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :i8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                        .grid_x = @intCast(chunk),
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :i8_blas;

                    // Step 2: INT8 GEMM — I8 input × I8 weight → I32 output.
                    self.blas.matmulI8I8I32(
                        &self.device,
                        &i8_input_buf,
                        chunk,
                        in_dim,
                        &w.dequant_i8_cache,
                        out_dim,
                        &i32_out_buf,
                    ) catch {
                        self.i8_blas_supported = false;
                        break :i8_blas;
                    };

                    // Step 3: Dequantize I32 → F32 with per-row × per-col scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&i32_out_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&w.mean_scale_cache) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&chunk_output) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(chunk)) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(out_dim)) catch break :i8_blas;
                    const dequant_blocks_x_usize = std.math.divCeil(usize, out_dim, 256) catch break :i8_blas;
                    const dequant_blocks_x = std.math.cast(u32, dequant_blocks_x_usize) orelse break :i8_blas;
                    const dequant_rows = std.math.cast(u32, chunk) orelse break :i8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, dequant_fn, .{
                        .grid_x = dequant_blocks_x,
                        .grid_y = dequant_rows,
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :i8_blas;

                    done += chunk;
                }
                return;
            }

            // On-demand I8 tensor core path: dequant U4→I8 into scratch per
            // linear op, then cuBLAS I8×I8→I32 GEMM. No persistent I8 cache
            // needed — uses existing scratch buffers. ~2× faster than F16 path.
            if (self.gaffine_u4_to_i8_function != null and self.i8_blas_supported) on_demand_i8: {
                const to_i8_fn = self.gaffine_u4_to_i8_function.?;
                const quant_fn = self.quantize_f32_to_i8_simple_function orelse break :on_demand_i8;
                const dequant_fn = self.dequant_i32_scales_function orelse break :on_demand_i8;

                const in_dim = w.rows;
                const out_dim = w.cols;
                const weight_i8_bytes = std.math.mul(usize, in_dim, out_dim) catch break :on_demand_i8;
                const col_scale_bytes = std.math.mul(usize, out_dim, @sizeOf(f32)) catch break :on_demand_i8;

                // Layout scratch: [I8 weights | F32 col scales] in dequant_f16_dev,
                //                 [I8 input | F32 row scales] in activation_u16_dev,
                //                 [I32 output] needs separate space.
                if (self.runtime_buffers.dequant_f16_dev.size < weight_i8_bytes + col_scale_bytes)
                    break :on_demand_i8;

                var i8_weight_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, weight_i8_bytes) catch break :on_demand_i8;
                var col_scale_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, weight_i8_bytes, col_scale_bytes) catch break :on_demand_i8;

                // Step 0: Dequant U4 weights → I8 + per-col F32 scales (on the fly).
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&w.packed_data) catch break :on_demand_i8;
                self.kernel_arg_pack.appendBufferPtr(&w.scales) catch break :on_demand_i8;
                self.kernel_arg_pack.appendBufferPtr(&w.biases) catch break :on_demand_i8;
                self.kernel_arg_pack.appendBufferPtr(&i8_weight_buf) catch break :on_demand_i8;
                self.kernel_arg_pack.appendBufferPtr(&col_scale_buf) catch break :on_demand_i8;
                self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :on_demand_i8;
                self.kernel_arg_pack.appendScalar(u32, w.group_size) catch break :on_demand_i8;
                self.kernel_arg_pack.appendScalar(u32, w.scales_dtype_tag) catch break :on_demand_i8;
                compute.cuda.launch.launchWithFamily(&self.device, to_i8_fn, .{
                    .grid_x = @intCast(out_dim),
                    .block_x = 256,
                }, &self.kernel_arg_pack, .other) catch break :on_demand_i8;

                // Chunked I8 GEMM (same pattern as cached path).
                const act_per_row = in_dim + @sizeOf(f32);
                const i32_per_row = std.math.mul(usize, out_dim, @sizeOf(i32)) catch break :on_demand_i8;
                const max_chunk = @min(
                    self.runtime_buffers.activation_u16_dev.size / act_per_row,
                    // I32 output goes into the tail of dequant_f16_dev after weights+scales.
                    (self.runtime_buffers.dequant_f16_dev.size - weight_i8_bytes - col_scale_bytes) / i32_per_row,
                );
                if (max_chunk == 0) break :on_demand_i8;

                const i32_buf_offset = weight_i8_bytes + col_scale_bytes;
                var done: usize = 0;
                while (done < rows) {
                    const chunk = @min(rows - done, max_chunk);
                    const i8_input_bytes = chunk * in_dim;
                    const row_scales_bytes = chunk * @sizeOf(f32);
                    const i32_out_bytes = std.math.mul(usize, chunk * out_dim, @sizeOf(i32)) catch break :on_demand_i8;

                    var i8_input_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, i8_input_bytes) catch break :on_demand_i8;
                    var row_scales_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, i8_input_bytes, row_scales_bytes) catch break :on_demand_i8;
                    var i32_out_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, i32_buf_offset, i32_out_bytes) catch break :on_demand_i8;

                    const in_off = done * in_dim * @sizeOf(f32);
                    const out_off = done * out_dim * @sizeOf(f32);
                    var chunk_input = bufferSlice(&packed_input, in_off, chunk * in_dim * @sizeOf(f32)) catch break :on_demand_i8;
                    var chunk_output = bufferSlice(&packed_out, out_off, chunk * out_dim * @sizeOf(f32)) catch break :on_demand_i8;

                    // Step 1: Quantize F32 input → I8 + per-row scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&chunk_input) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendBufferPtr(&i8_input_buf) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :on_demand_i8;
                    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                        .grid_x = @intCast(chunk),
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :on_demand_i8;

                    // Step 2: INT8 GEMM — I8 input × I8 weight → I32 output.
                    self.blas.matmulI8I8I32(
                        &self.device,
                        &i8_input_buf,
                        chunk,
                        in_dim,
                        &i8_weight_buf,
                        out_dim,
                        &i32_out_buf,
                    ) catch {
                        self.i8_blas_supported = false;
                        break :on_demand_i8;
                    };

                    // Step 3: Dequantize I32 → F32 with per-row × per-col scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&i32_out_buf) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendBufferPtr(&col_scale_buf) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendBufferPtr(&chunk_output) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(chunk)) catch break :on_demand_i8;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(out_dim)) catch break :on_demand_i8;
                    const dequant_blocks_x = std.math.cast(u32, std.math.divCeil(usize, out_dim, 256) catch break :on_demand_i8) orelse break :on_demand_i8;
                    compute.cuda.launch.launchWithFamily(&self.device, dequant_fn, .{
                        .grid_x = dequant_blocks_x,
                        .grid_y = @intCast(chunk),
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :on_demand_i8;

                    done += chunk;
                }
                if (done == rows) return;
            }

            // F16 dequant + cuBLAS GEMM path for prefill.
            if (self.cast_f32_to_f16_function) |cast_fn| {
                if (self.u16_blas_f16_supported) dequant_blas: {
                    const weight_elems = std.math.mul(usize, w.rows, w.cols) catch break :dequant_blas;
                    const weight_f16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch break :dequant_blas;

                    const input_elems = std.math.mul(usize, rows, w.rows) catch break :dequant_blas;
                    const input_u16_bytes = std.math.mul(usize, input_elems, @sizeOf(u16)) catch break :dequant_blas;
                    if (self.runtime_buffers.activation_u16_dev.size < input_u16_bytes) break :dequant_blas;

                    // Use pre-dequantized F16 cache if available, otherwise dequant on the fly.
                    var dequant_weight: compute.cuda.Buffer = undefined;
                    if (w.dequant_f16_cache.pointer != 0 and w.dequant_f16_cache.size >= weight_f16_bytes) {
                        dequant_weight = bufferSlice(&w.dequant_f16_cache, 0, weight_f16_bytes) catch break :dequant_blas;
                    } else {
                        const dequant_fn = self.gaffine_u4_dequant_f16_function orelse break :dequant_blas;
                        if (self.runtime_buffers.dequant_f16_dev.size < weight_f16_bytes) break :dequant_blas;
                        dequant_weight = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, weight_f16_bytes) catch break :dequant_blas;
                        compute.cuda.gaffine_u4_dequantize_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            dequant_fn,
                            &w.packed_data,
                            &w.scales,
                            &w.biases,
                            &dequant_weight,
                            @intCast(w.cols),
                            @intCast(w.rows),
                            w.group_size,
                            w.scales_dtype_tag,
                        ) catch break :dequant_blas;
                    }

                    var input_u16_dev = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_u16_bytes) catch break :dequant_blas;
                    compute.cuda.cast_f32_to_f16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        cast_fn,
                        &packed_input,
                        &input_u16_dev,
                        @intCast(input_elems),
                    ) catch break :dequant_blas;

                    self.blas.matmulU16U16F32(
                        &self.device,
                        &input_u16_dev,
                        .f16,
                        rows,
                        w.rows,
                        &dequant_weight,
                        .f16,
                        w.cols,
                        &packed_out,
                    ) catch {
                        self.u16_blas_f16_supported = false;
                        break :dequant_blas;
                    };
                    return;
                }
            }

            if (rows > 1 and !self.gaffine_sequence_rows_supported) return error.UnsupportedModel;
            if (use_tile8) {
                try compute.cuda.gaffine_u4_matvec.runWithFunctionTile8(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernel,
                    &packed_input,
                    &w.packed_data,
                    &w.scales,
                    &w.biases,
                    &packed_out,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    w.group_size,
                    w.scales_dtype_tag,
                    @intCast(rows),
                    0,
                );
            } else {
                try compute.cuda.gaffine_u4_matvec.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernel,
                    &packed_input,
                    &w.packed_data,
                    &w.scales,
                    &w.biases,
                    &packed_out,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    w.group_size,
                    w.scales_dtype_tag,
                    @intCast(rows),
                    0,
                );
            }
            return;
        },
        .gaffine_u8 => |w| {
            const kernel = self.gaffine_u8_matvec_function orelse return error.CudaKernelUnavailable;
            if (rows == 1) {
                // Try I8 GEMV: warp-per-row kernel, 4 output rows per block.
                if (w.dequant_i8_cache.pointer != 0 and
                    w.mean_scale_cache.pointer != 0)
                i8_decode: {
                    const i8_fn = self.i8_matvec_function orelse break :i8_decode;
                    const out_cols: u32 = @intCast(w.cols);
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :i8_decode;
                    self.kernel_arg_pack.appendBufferPtr(&w.dequant_i8_cache) catch break :i8_decode;
                    self.kernel_arg_pack.appendBufferPtr(&w.mean_scale_cache) catch break :i8_decode;
                    self.kernel_arg_pack.appendBufferPtr(&packed_out) catch break :i8_decode;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch break :i8_decode;
                    self.kernel_arg_pack.appendScalar(u32, out_cols) catch break :i8_decode;
                    self.kernel_arg_pack.appendScalar(u32, 1) catch break :i8_decode;
                    self.kernel_arg_pack.appendDevicePtr(0) catch break :i8_decode;
                    compute.cuda.launch.launchWithFamily(&self.device, i8_fn, .{
                        .grid_x = (out_cols + 3) / 4,
                        .grid_y = 1,
                        .block_x = 128,
                    }, &self.kernel_arg_pack, .matvec) catch break :i8_decode;
                    return;
                }
                // U8 GEMV with per-group dequant.
                try compute.cuda.gaffine_u8_matvec.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernel,
                    &packed_input,
                    &w.packed_data,
                    &w.scales,
                    &w.biases,
                    &packed_out,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    w.group_size,
                    w.scales_dtype_tag,
                    1,
                    0,
                );
                return;
            }

            // Symmetric INT8 tensor core path for prefill.
            // Weight is pre-quantized F16→I8 with per-output-row scales at warmup.
            // When the full batch doesn't fit in scratch, process in
            // row-chunks that do — avoids growing dequant_f16_dev.
            if (w.dequant_i8_cache.pointer != 0 and
                w.mean_scale_cache.pointer != 0 and
                self.i8_blas_supported)
            i8_blas: {
                const quant_fn = self.quantize_f32_to_i8_simple_function orelse break :i8_blas;
                const dequant_fn = self.dequant_i32_scales_function orelse break :i8_blas;

                const in_dim = w.rows;
                const out_dim = w.cols;

                // Max rows per chunk from available scratch buffers.
                const act_per_row = in_dim + @sizeOf(f32); // I8 input + F32 row scale
                const i32_per_row = std.math.mul(usize, out_dim, @sizeOf(i32)) catch break :i8_blas;
                const max_chunk = @min(
                    self.runtime_buffers.activation_u16_dev.size / act_per_row,
                    self.runtime_buffers.dequant_f16_dev.size / i32_per_row,
                );
                if (max_chunk == 0) break :i8_blas;

                var done: usize = 0;
                while (done < rows) {
                    const chunk = @min(rows - done, max_chunk);
                    const i8_input_bytes = chunk * in_dim;
                    const row_scales_bytes = chunk * @sizeOf(f32);
                    const i32_out_bytes = std.math.mul(usize, chunk * out_dim, @sizeOf(i32)) catch break :i8_blas;

                    var i8_input_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, i8_input_bytes) catch break :i8_blas;
                    var row_scales_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, i8_input_bytes, row_scales_bytes) catch break :i8_blas;
                    var i32_out_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, i32_out_bytes) catch break :i8_blas;

                    const in_off = done * in_dim * @sizeOf(f32);
                    const out_off = done * out_dim * @sizeOf(f32);
                    var chunk_input = bufferSlice(&packed_input, in_off, chunk * in_dim * @sizeOf(f32)) catch break :i8_blas;
                    var chunk_output = bufferSlice(&packed_out, out_off, chunk * out_dim * @sizeOf(f32)) catch break :i8_blas;

                    // Step 1: Quantize F32 input → I8 + per-row scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&chunk_input) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&i8_input_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :i8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                        .grid_x = @intCast(chunk),
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :i8_blas;

                    // Step 2: INT8 GEMM — I8 input × I8 weight → I32 output.
                    self.blas.matmulI8I8I32(
                        &self.device,
                        &i8_input_buf,
                        chunk,
                        in_dim,
                        &w.dequant_i8_cache,
                        out_dim,
                        &i32_out_buf,
                    ) catch {
                        self.i8_blas_supported = false;
                        break :i8_blas;
                    };

                    // Step 3: Dequantize I32 → F32 with per-row × per-col scales.
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&i32_out_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&w.mean_scale_cache) catch break :i8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&chunk_output) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(chunk)) catch break :i8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(out_dim)) catch break :i8_blas;
                    const dequant_blocks_x_usize = std.math.divCeil(usize, out_dim, 256) catch break :i8_blas;
                    const dequant_blocks_x = std.math.cast(u32, dequant_blocks_x_usize) orelse break :i8_blas;
                    const dequant_rows = std.math.cast(u32, chunk) orelse break :i8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, dequant_fn, .{
                        .grid_x = dequant_blocks_x,
                        .grid_y = dequant_rows,
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :i8_blas;

                    done += chunk;
                }
                return;
            }

            // Prefill (rows > 1): use cached F16 weights or dequantize, cast input to F16, cuBLAS GEMM.
            if (self.cast_f32_to_f16_function) |cast_fn| {
                if (self.u16_blas_f16_supported) dequant_blas: {
                    const weight_elems = std.math.mul(usize, w.rows, w.cols) catch break :dequant_blas;
                    const weight_f16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch break :dequant_blas;

                    const input_elems = std.math.mul(usize, rows, w.rows) catch break :dequant_blas;
                    const input_u16_bytes = std.math.mul(usize, input_elems, @sizeOf(u16)) catch break :dequant_blas;
                    if (self.runtime_buffers.activation_u16_dev.size < input_u16_bytes) break :dequant_blas;

                    // Use pre-dequantized F16 cache if available, otherwise dequant on the fly.
                    var dequant_weight: compute.cuda.Buffer = undefined;
                    if (w.dequant_f16_cache.pointer != 0 and w.dequant_f16_cache.size >= weight_f16_bytes) {
                        dequant_weight = bufferSlice(&w.dequant_f16_cache, 0, weight_f16_bytes) catch break :dequant_blas;
                    } else {
                        const dequant_fn = self.gaffine_u8_dequant_f16_function orelse break :dequant_blas;
                        if (self.runtime_buffers.dequant_f16_dev.size < weight_f16_bytes) break :dequant_blas;
                        dequant_weight = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, weight_f16_bytes) catch break :dequant_blas;
                        compute.cuda.gaffine_u8_dequantize_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            dequant_fn,
                            &w.packed_data,
                            &w.scales,
                            &w.biases,
                            &dequant_weight,
                            @intCast(w.cols),
                            @intCast(w.rows),
                            w.group_size,
                            w.scales_dtype_tag,
                        ) catch break :dequant_blas;
                    }

                    var input_u16_dev = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_u16_bytes) catch break :dequant_blas;
                    compute.cuda.cast_f32_to_f16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        cast_fn,
                        &packed_input,
                        &input_u16_dev,
                        @intCast(input_elems),
                    ) catch break :dequant_blas;

                    self.blas.matmulU16U16F32(
                        &self.device,
                        &input_u16_dev,
                        .f16,
                        rows,
                        w.rows,
                        &dequant_weight,
                        .f16,
                        w.cols,
                        &packed_out,
                    ) catch {
                        self.u16_blas_f16_supported = false;
                        break :dequant_blas;
                    };
                    return;
                }
            }

            if (rows > 1 and !self.gaffine_sequence_rows_supported) return error.UnsupportedModel;
            try compute.cuda.gaffine_u8_matvec.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernel,
                &packed_input,
                &w.packed_data,
                &w.scales,
                &w.biases,
                &packed_out,
                @intCast(w.rows),
                @intCast(w.cols),
                w.group_size,
                w.scales_dtype_tag,
                @intCast(rows),
                0,
            );
            return;
        },
        .fp8 => |w| {
            // Decode path: FP8 GEMV with per-block BF16 scales (DRAM-bound, 1 byte/weight).
            // Uses batched template: tile4 for rows<=4, tile8 for rows>4.
            // Weight loaded once from DRAM, reused across batch rows.
            if (rows <= 32) fp8_gemv: {
                if (w.scales_buffer.pointer == 0 or w.scales_buffer.size == 0) break :fp8_gemv;
                var fp8_fn = self.fp8_matvec_function orelse break :fp8_gemv;
                var fp8_batch_tile: u32 = 4;
                if (rows > 4) {
                    if (self.fp8_matvec_tile8_function) |tile8_fn| {
                        fp8_fn = tile8_fn;
                        fp8_batch_tile = 8;
                    }
                }

                const out_cols: u32 = @intCast(w.cols);
                const batch_rows: u32 = @intCast(rows);
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :fp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch break :fp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch break :fp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&packed_out) catch break :fp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch break :fp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, out_cols) catch break :fp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, w.block_size) catch break :fp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, w.scale_cols) catch break :fp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, batch_rows) catch break :fp8_gemv;
                compute.cuda.launch.launchWithFamily(&self.device, fp8_fn, .{
                    .grid_x = (out_cols + 3) / 4,
                    .grid_y = (batch_rows + fp8_batch_tile - 1) / fp8_batch_tile,
                    .block_x = 128,
                }, &self.kernel_arg_pack, .matvec) catch break :fp8_gemv;
                return;
            }

            // Prefill path: dequant FP8→BF16, then cuBLAS BF16 tensor core GEMM.
            if (w.scales_buffer.pointer != 0 and w.scales_buffer.size > 0) fp8_dequant_blas: {
                const dequant_fn = self.fp8_dequant_to_bf16_function orelse break :fp8_dequant_blas;
                const cast_fn = self.cast_f32_to_bf16_function orelse break :fp8_dequant_blas;
                if (!self.u16_blas_bf16_supported) break :fp8_dequant_blas;

                const weight_elems = std.math.mul(usize, w.rows, w.cols) catch break :fp8_dequant_blas;
                const weight_bf16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch break :fp8_dequant_blas;
                if (self.runtime_buffers.dequant_f16_dev.size < weight_bf16_bytes) break :fp8_dequant_blas;

                const input_elems = std.math.mul(usize, rows, w.rows) catch break :fp8_dequant_blas;
                const input_bf16_bytes = std.math.mul(usize, input_elems, @sizeOf(u16)) catch break :fp8_dequant_blas;
                if (self.runtime_buffers.activation_u16_dev.size < input_bf16_bytes) break :fp8_dequant_blas;

                // Step 1: Dequant FP8 weights → BF16 in scratch buffer.
                var dequant_weight = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, weight_bf16_bytes) catch break :fp8_dequant_blas;
                const grid_x = std.math.cast(u32, std.math.divCeil(usize, w.rows, 256) catch break :fp8_dequant_blas) orelse break :fp8_dequant_blas;
                const grid_y = std.math.cast(u32, w.cols) orelse break :fp8_dequant_blas;
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendBufferPtr(&dequant_weight) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendScalar(u32, @intCast(w.cols)) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendScalar(u32, w.block_size) catch break :fp8_dequant_blas;
                self.kernel_arg_pack.appendScalar(u32, w.scale_cols) catch break :fp8_dequant_blas;
                compute.cuda.launch.launchWithFamily(&self.device, dequant_fn, .{
                    .grid_x = grid_x,
                    .grid_y = grid_y,
                    .block_x = 256,
                }, &self.kernel_arg_pack, .other) catch break :fp8_dequant_blas;

                // Step 2: Cast F32 input → BF16.
                var input_bf16_dev = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_bf16_bytes) catch break :fp8_dequant_blas;
                try compute.cuda.cast_f32_to_bf16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_fn,
                    &packed_input,
                    &input_bf16_dev,
                    @intCast(input_elems),
                );

                // Step 3: cuBLAS BF16 × BF16 → F32 GEMM (tensor cores).
                self.blas.matmulU16U16F32(
                    &self.device,
                    &input_bf16_dev,
                    .bf16,
                    rows,
                    w.rows,
                    &dequant_weight,
                    .bf16,
                    w.cols,
                    &packed_out,
                ) catch {
                    self.u16_blas_bf16_supported = false;
                    break :fp8_dequant_blas;
                };
                return;
            }

            // Per-tensor scalar scale: cuBLAS FP8×FP8→F32 W8A8 path.
            if (w.weight_scale_inv != 0 and self.fp8_blas_supported) fp8_blas: {
                const quant_fn = self.quantize_f32_to_fp8_function orelse break :fp8_blas;
                const scale_fn = self.scale_rows_f32_function orelse break :fp8_blas;

                const in_dim = w.rows;
                const out_dim = w.cols;
                const act_per_row = in_dim + @sizeOf(f32);
                const max_chunk = self.runtime_buffers.activation_u16_dev.size / act_per_row;
                if (max_chunk == 0) break :fp8_blas;

                var done: usize = 0;
                while (done < rows) {
                    const chunk = @min(rows - done, max_chunk);
                    const fp8_input_bytes = chunk * in_dim;
                    const row_scales_bytes = chunk * @sizeOf(f32);

                    var fp8_input_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, fp8_input_bytes) catch break :fp8_blas;
                    var row_scales_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, fp8_input_bytes, row_scales_bytes) catch break :fp8_blas;

                    const in_off = done * in_dim * @sizeOf(f32);
                    const out_off = done * out_dim * @sizeOf(f32);
                    var chunk_input = bufferSlice(&packed_input, in_off, chunk * in_dim * @sizeOf(f32)) catch break :fp8_blas;
                    var chunk_output = bufferSlice(&packed_out, out_off, chunk * out_dim * @sizeOf(f32)) catch break :fp8_blas;

                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&chunk_input) catch break :fp8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&fp8_input_buf) catch break :fp8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :fp8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :fp8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                        .grid_x = @intCast(chunk),
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :fp8_blas;

                    self.blas.matmulFp8Fp8F32(
                        &self.device,
                        &fp8_input_buf,
                        chunk,
                        in_dim,
                        &w.buffer,
                        out_dim,
                        &chunk_output,
                        w.weight_scale_inv,
                    ) catch {
                        self.fp8_blas_supported = false;
                        break :fp8_blas;
                    };

                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&chunk_output) catch break :fp8_blas;
                    self.kernel_arg_pack.appendBufferPtr(&row_scales_buf) catch break :fp8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(chunk)) catch break :fp8_blas;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(out_dim)) catch break :fp8_blas;
                    const scale_blocks_x = std.math.cast(u32, std.math.divCeil(usize, out_dim, 256) catch break :fp8_blas) orelse break :fp8_blas;
                    const scale_rows_y = std.math.cast(u32, chunk) orelse break :fp8_blas;
                    compute.cuda.launch.launchWithFamily(&self.device, scale_fn, .{
                        .grid_x = scale_blocks_x,
                        .grid_y = scale_rows_y,
                        .block_x = 256,
                    }, &self.kernel_arg_pack, .other) catch break :fp8_blas;

                    done += chunk;
                }
                return;
            }

            return error.CudaKernelUnavailable;
        },
        .mxfp8 => |w| {
            // MXFP8 GEMV for small batch sizes (≤4 rows).
            // Reads F32 input directly — no activation quantization needed.
            // Uses row-major UE8M0 scales (scales_raw_buffer).
            // At n>4 the tile-8 kernel drops to 50% occupancy; cuBLASLt
            // tensor cores achieve better bandwidth (and its overhead is
            // amortised to zero inside CUDA graph replay).
            if (rows <= 4) mxfp8_gemv: {
                if (w.scales_raw_buffer.pointer == 0 or w.scales_raw_buffer.size == 0) break :mxfp8_gemv;
                var mxfp8_fn = self.mxfp8_matvec_function orelse break :mxfp8_gemv;
                var mxfp8_batch_tile: u32 = 4;
                if (rows > 4) {
                    if (self.mxfp8_matvec_tile8_function) |tile8_fn| {
                        mxfp8_fn = tile8_fn;
                        mxfp8_batch_tile = 8;
                    }
                }

                const out_cols: u32 = @intCast(w.cols);
                const batch_rows: u32 = @intCast(rows);
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&w.scales_raw_buffer) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendBufferPtr(&packed_out) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, out_cols) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, w.scale_cols) catch break :mxfp8_gemv;
                self.kernel_arg_pack.appendScalar(u32, batch_rows) catch break :mxfp8_gemv;
                compute.cuda.launch.launchWithFamily(&self.device, mxfp8_fn, .{
                    .grid_x = (out_cols + 3) / 4,
                    .grid_y = (batch_rows + mxfp8_batch_tile - 1) / mxfp8_batch_tile,
                    .block_x = 128,
                }, &self.kernel_arg_pack, .matvec) catch break :mxfp8_gemv;
                return;
            }

            // cuBLASLt block-scaled FP8 tensor core GEMM for large batch sizes.
            // Requires: (1) cuBLASLt handle, (2) activation quantization kernel F32→E4M3+UE8M0.
            if (self.blas_lt) |*blas_lt| mxfp8_lt: {
                const quant_fn = self.quantize_f32_to_mxfp8_function orelse break :mxfp8_lt;

                const in_dim = w.rows;
                const out_dim = w.cols;
                const scale_cols: usize = (in_dim + 31) / 32;

                // cuBLASLt VEC32_UE8M0 requires scale tensors padded to 128-tile
                // boundaries with interleaved layout. Compute padded dimensions.
                const padded_outer = engine_types.Mxfp8LinearWeight.roundoff(rows, 128);
                const padded_sf_k = engine_types.Mxfp8LinearWeight.roundoff(scale_cols, 4);
                const act_scale_bytes = padded_outer * padded_sf_k;

                // Activation E4M3 bytes from activation_u16_dev, interleaved scales from
                // dequant_f16_dev (large buffer, unused during cuBLASLt path).
                const act_e4m3_bytes = std.math.mul(usize, rows, in_dim) catch break :mxfp8_lt;
                if (self.runtime_buffers.activation_u16_dev.size < act_e4m3_bytes) break :mxfp8_lt;
                if (self.runtime_buffers.dequant_f16_dev.size < act_scale_bytes) break :mxfp8_lt;

                var act_e4m3_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, act_e4m3_bytes) catch break :mxfp8_lt;
                var act_scale_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, act_scale_bytes) catch break :mxfp8_lt;

                // Step 1: Quantize F32 activations → E4M3 + interleaved UE8M0 scales.
                // Kernel launches with padded_outer rows; padded rows write zero scales.
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendBufferPtr(&act_e4m3_buf) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendBufferPtr(&act_scale_buf) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendScalar(u32, @intCast(rows)) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendScalar(u32, @intCast(padded_outer)) catch break :mxfp8_lt;
                self.kernel_arg_pack.appendScalar(u32, @intCast(padded_sf_k)) catch break :mxfp8_lt;
                compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                    .grid_x = @intCast(scale_cols),
                    .grid_y = @intCast(padded_outer),
                    .block_x = 32,
                }, &self.kernel_arg_pack, .other) catch break :mxfp8_lt;

                // Step 2: cuBLASLt block-scaled MXFP8 GEMM
                blas_lt.matmulMxfp8(
                    &self.device,
                    &w.buffer,
                    &w.scales_buffer,
                    &act_e4m3_buf,
                    &act_scale_buf,
                    &packed_out,
                    rows,
                    out_dim,
                    in_dim,
                ) catch |lt_err| {
                    log.err("inference", "MXFP8 cuBLASLt failed", .{ .err = @errorName(lt_err) }, @src());
                    break :mxfp8_lt;
                };
                return;
            }

            // Fallback: dequant MXFP8→BF16, then cuBLAS BF16 GEMM.
            if (self.mxfp8_dequant_to_bf16_function) |dequant_fn| {
                if (self.cast_f32_to_bf16_function) |cast_fn| {
                    if (self.u16_blas_bf16_supported) mxfp8_dequant_blas: {
                        const in_dim = w.rows;
                        const out_dim = w.cols;

                        const weight_elems = std.math.mul(usize, in_dim, out_dim) catch break :mxfp8_dequant_blas;
                        const weight_bf16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch break :mxfp8_dequant_blas;
                        if (self.runtime_buffers.dequant_f16_dev.size < weight_bf16_bytes) break :mxfp8_dequant_blas;

                        const input_elems = std.math.mul(usize, rows, in_dim) catch break :mxfp8_dequant_blas;
                        const input_bf16_bytes = std.math.mul(usize, input_elems, @sizeOf(u16)) catch break :mxfp8_dequant_blas;
                        if (self.runtime_buffers.activation_u16_dev.size < input_bf16_bytes) break :mxfp8_dequant_blas;

                        // Step 1: Dequant MXFP8 weights → BF16.
                        var dequant_weight = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, weight_bf16_bytes) catch break :mxfp8_dequant_blas;
                        const grid_x = std.math.cast(u32, std.math.divCeil(usize, in_dim, 256) catch break :mxfp8_dequant_blas) orelse break :mxfp8_dequant_blas;
                        const grid_y = std.math.cast(u32, out_dim) orelse break :mxfp8_dequant_blas;
                        // Compute interleaved scale dimensions (weight scales are stored interleaved)
                        const dequant_scale_cols: usize = w.scale_cols;
                        const dequant_padded_outer = engine_types.Mxfp8LinearWeight.roundoff(out_dim, 128);
                        const dequant_padded_sf_k = engine_types.Mxfp8LinearWeight.roundoff(dequant_scale_cols, 4);

                        self.kernel_arg_pack.reset();
                        self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendBufferPtr(&dequant_weight) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendScalar(u32, @intCast(out_dim)) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendScalar(u32, @intCast(dequant_padded_outer)) catch break :mxfp8_dequant_blas;
                        self.kernel_arg_pack.appendScalar(u32, @intCast(dequant_padded_sf_k)) catch break :mxfp8_dequant_blas;
                        compute.cuda.launch.launchWithFamily(&self.device, dequant_fn, .{
                            .grid_x = grid_x,
                            .grid_y = grid_y,
                            .block_x = 256,
                        }, &self.kernel_arg_pack, .other) catch break :mxfp8_dequant_blas;

                        // Step 2: Cast F32 input → BF16.
                        var input_bf16_dev = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_bf16_bytes) catch break :mxfp8_dequant_blas;
                        try compute.cuda.cast_f32_to_bf16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_fn,
                            &packed_input,
                            &input_bf16_dev,
                            @intCast(input_elems),
                        );

                        // Step 3: cuBLAS BF16 × BF16 → F32 GEMM.
                        self.blas.matmulU16U16F32(
                            &self.device,
                            &input_bf16_dev,
                            .bf16,
                            rows,
                            in_dim,
                            &dequant_weight,
                            .bf16,
                            out_dim,
                            &packed_out,
                        ) catch {
                            self.u16_blas_bf16_supported = false;
                            break :mxfp8_dequant_blas;
                        };
                        return;
                    }
                }
            }

            return error.CudaKernelUnavailable;
        },
        .nvfp4 => |w| {
            // Native FP4 tensor core prefill (Blackwell SM_100+):
            // Quantize F32 input → FP4, then cuBLASLt FP4×FP4 → F32 GEMM.
            // Native cuBLASLt FP4×FP4 → BF16 GEMM + BF16→F32 cast (Blackwell SM_100+).
            const prefer_native_single_row = rows == 1 and w.cols >= 65536;
            if ((rows > 4 or prefer_native_single_row) and w.scales_lt_buffer.size > 0) nvfp4_native_blas: {
                const quant_fn = self.quantize_f32_to_nvfp4_function orelse break :nvfp4_native_blas;
                var blas_lt = self.blas_lt orelse break :nvfp4_native_blas;

                const in_dim = w.rows;
                const out_dim = w.cols;

                // Scratch layout:
                // - activation_u16_dev: FP4 packed input [rows × packed_in_cols]
                // - dequant_f16_dev:    [0..scale_bytes] interleaved UE4M3 scales
                //                       [scale_bytes..] BF16 GEMM output [rows × out_dim]
                const packed_in_cols = (in_dim + 1) / 2;
                const input_fp4_bytes = std.math.mul(usize, rows, packed_in_cols) catch break :nvfp4_native_blas;
                const input_scale_bytes = engine_types.Nvfp4LinearWeight.cublasLtScaleTensorSize(in_dim, rows);
                if (self.runtime_buffers.activation_u16_dev.size < input_fp4_bytes) break :nvfp4_native_blas;
                if (self.runtime_buffers.dequant_f16_dev.size < input_scale_bytes) break :nvfp4_native_blas;

                var input_fp4_buf = bufferSlice(&self.runtime_buffers.activation_u16_dev, 0, input_fp4_bytes) catch break :nvfp4_native_blas;
                var input_scales_buf = bufferSlice(&self.runtime_buffers.dequant_f16_dev, 0, input_scale_bytes) catch break :nvfp4_native_blas;

                // Step 1: Quantize F32 input → FP4 E2M1 + interleaved UE4M3 scales.
                const padded_outer: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(rows, 128));
                const sf_k = (in_dim + 15) / 16;
                const padded_sf_k: u32 = @intCast(engine_types.Nvfp4LinearWeight.roundoff(sf_k, 4));
                const quant_grid_x = std.math.cast(u32, sf_k) orelse break :nvfp4_native_blas;
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendBufferPtr(&input_fp4_buf) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendBufferPtr(&input_scales_buf) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendScalar(u32, @intCast(in_dim)) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendScalar(u32, @intCast(rows)) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendScalar(u32, padded_outer) catch break :nvfp4_native_blas;
                self.kernel_arg_pack.appendScalar(u32, padded_sf_k) catch break :nvfp4_native_blas;
                compute.cuda.launch.launchWithFamily(&self.device, quant_fn, .{
                    .grid_x = quant_grid_x,
                    .grid_y = padded_outer,
                    .block_x = 32,
                }, &self.kernel_arg_pack, .other) catch |err| {
                    log.warn("inference", "NVFP4 native FAIL: quant launch", .{ .err = @errorName(err) });
                    break :nvfp4_native_blas;
                };

                // Step 2: cuBLASLt FP4×FP4 → F32 GEMM (native FP4 tensor cores).
                const alpha_scale: f32 = 1.0 / w.weight_global_scale;

                blas_lt.matmulNvfp4(
                    &self.device,
                    &w.buffer,
                    &w.scales_lt_buffer,
                    &input_fp4_buf,
                    &input_scales_buf,
                    &packed_out,
                    rows,
                    out_dim,
                    in_dim,
                    alpha_scale,
                ) catch break :nvfp4_native_blas;
                recordNvfp4Route(self, .native_cublaslt);
                return;
            }

            if (prefer_native_single_row and @import("env_pkg").getenv("TALU_CUDA_NVFP4_NATIVE_DEBUG") != null) {
                log.warn("inference", "CUDA NVFP4 native single-row skipped", .{
                    .rows = rows,
                    .in_dim = w.rows,
                    .out_dim = w.cols,
                    .has_scales_lt = @as(u8, @intFromBool(w.scales_lt_buffer.size > 0)),
                    .has_quant_fn = @as(u8, @intFromBool(self.quantize_f32_to_nvfp4_function != null)),
                    .has_blas_lt = @as(u8, @intFromBool(self.blas_lt != null)),
                });
            }

            // Decode / small-batch path: native weight-only GEMV kernels.
            const base_nvfp4_fn = self.nvfp4_matvec_function orelse {
                log.warn("inference", "CUDA NVFP4 matvec kernel unavailable", .{});
                return error.CudaKernelUnavailable;
            };
            const out_cols: u32 = @intCast(w.cols);

            // Decode / small-batch fast path: reuse persistent NVFP4->I8 cache
            // and run i8_matvec_f32 directly. This avoids FP4 nibble decode in
            // token-time projection kernels.
            if (rows <= 4 and
                w.dequant_i8_cache.pointer != 0 and
                w.mean_scale_cache.pointer != 0 and
                (w.cols < 65536 or @import("builtin").os.tag == .windows))
            i8_decode: {
                const i8_fn = self.i8_matvec_function orelse break :i8_decode;
                const batch_rows: u32 = @intCast(rows);
                self.kernel_arg_pack.reset();
                self.kernel_arg_pack.appendBufferPtr(&packed_input) catch break :i8_decode;
                self.kernel_arg_pack.appendBufferPtr(&w.dequant_i8_cache) catch break :i8_decode;
                self.kernel_arg_pack.appendBufferPtr(&w.mean_scale_cache) catch break :i8_decode;
                self.kernel_arg_pack.appendBufferPtr(&packed_out) catch break :i8_decode;
                self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch break :i8_decode;
                self.kernel_arg_pack.appendScalar(u32, out_cols) catch break :i8_decode;
                self.kernel_arg_pack.appendScalar(u32, batch_rows) catch break :i8_decode;
                self.kernel_arg_pack.appendDevicePtr(0) catch break :i8_decode;
                try compute.cuda.launch.launchWithFamily(&self.device, i8_fn, .{
                    .grid_x = (out_cols + 3) / 4,
                    .grid_y = batch_rows,
                    .block_x = 128,
                }, &self.kernel_arg_pack, .matvec);
                recordNvfp4Route(self, .small_rows_i8_matvec);
                return;
            }

            if (rows > 1 and !self.nvfp4_sequence_rows_supported) {
                return error.UnsupportedModel;
            }

            var nvfp4_fn = base_nvfp4_fn;
            var nvfp4_batch_tile: u32 = 4;
            // Decode is overwhelmingly single-row on NVFP4; prefer the wider
            // tile8 kernel for practical decode projection widths.
            const prefer_tile8 = rows > 4 or
                (rows == 1 and w.cols >= 2048 and @import("builtin").os.tag != .windows);
            if (prefer_tile8) {
                if (self.nvfp4_matvec_tile8_function) |tile8_fn| {
                    nvfp4_fn = tile8_fn;
                    nvfp4_batch_tile = 8;
                }
            }

            const batch_rows: u32 = @intCast(rows);
            self.kernel_arg_pack.reset();
            self.kernel_arg_pack.appendBufferPtr(&packed_input) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendBufferPtr(&packed_out) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(u32, out_cols) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(u32, w.scale_cols) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(u32, w.group_size) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(f32, w.weight_global_scale) catch return error.CudaKernelUnavailable;
            self.kernel_arg_pack.appendScalar(u32, batch_rows) catch return error.CudaKernelUnavailable;
            try compute.cuda.launch.launchWithFamily(&self.device, nvfp4_fn, .{
                .grid_x = (out_cols + 3) / 4,
                .grid_y = (batch_rows + nvfp4_batch_tile - 1) / nvfp4_batch_tile,
                .block_x = 128,
            }, &self.kernel_arg_pack, .matvec);
            recordNvfp4Route(self, .small_rows_nvfp4_matvec);
            return;
        },
    }
}

pub fn linearWeightHasI8Cache(weight: *const LinearWeight) bool {
    return switch (weight.*) {
        .gaffine_u4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .gaffine_u8 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .nvfp4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        else => false,
    };
}
