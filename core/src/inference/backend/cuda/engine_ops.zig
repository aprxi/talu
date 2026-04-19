//! Compute operation building blocks and fused operation attempts.
//!
//! Contains linear forward, QKV projection, gate-up projection, FFN activation,
//! residual addition, RMSNorm, and all fused operation variants.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
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
const engine_weights = @import("engine_weights.zig");
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

pub fn compactQueryGateProjection(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
    n_heads_u32: u32,
    head_dim_u32: u32,
    q_projection_stage: *const compute.cuda.Buffer,
    q_values_stage: *compute.cuda.Buffer,
) !void {
    const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
    const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
    _ = projection_elements;
    _ = query_elements;
    try compute.cuda.gated_attention_compact_q.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.gated_attention_compact_q_function orelse return error.CudaKernelUnavailable,
        q_projection_stage,
        q_values_stage,
        @intCast(seq_len),
        @intCast(q_dim),
        @intCast(q_projection_dim),
        n_heads_u32,
        head_dim_u32,
    );
}

pub fn applyQueryGateToContextInPlace(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
    n_heads_u32: u32,
    head_dim_u32: u32,
) !void {
    const projection_elements = std.math.mul(usize, seq_len, q_projection_dim) catch return error.InvalidArgument;
    const query_elements = std.math.mul(usize, seq_len, q_dim) catch return error.InvalidArgument;
    const projection_bytes = std.math.mul(usize, projection_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    const context_bytes = std.math.mul(usize, query_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    var context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_bytes);
    var projection_stage = try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, projection_bytes);
    try compute.cuda.gated_attention_output_gate.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.gated_attention_output_gate_function orelse return error.CudaKernelUnavailable,
        &context_stage,
        &projection_stage,
        @intCast(seq_len),
        @intCast(q_dim),
        @intCast(q_projection_dim),
        n_heads_u32,
        head_dim_u32,
    );
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

            // Fallback: F16 dequant + cuBLAS GEMM path for prefill.
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

            // Fallback: row-by-row U4 GEMV.
            if (self.gaffine_sequence_rows_supported) {
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

            var row_index: usize = 0;
            while (row_index < rows) : (row_index += 1) {
                var input_row = try logicalF32RowSlice(&packed_input, rows, row_index, w.rows);
                var out_row = try logicalF32RowSlice(&packed_out, rows, row_index, w.cols);
                try compute.cuda.gaffine_u4_matvec.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernel,
                    &input_row,
                    &w.packed_data,
                    &w.scales,
                    &w.biases,
                    &out_row,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    w.group_size,
                    w.scales_dtype_tag,
                    1,
                    0,
                );
            }
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
                // Fallback: U8 GEMV with per-group dequant.
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

            // Fallback: row-by-row GEMV.
            if (self.gaffine_sequence_rows_supported) {
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
            }

            var row_index: usize = 0;
            while (row_index < rows) : (row_index += 1) {
                var input_row = try logicalF32RowSlice(&packed_input, rows, row_index, w.rows);
                var out_row = try logicalF32RowSlice(&packed_out, rows, row_index, w.cols);
                try compute.cuda.gaffine_u8_matvec.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernel,
                    &input_row,
                    &w.packed_data,
                    &w.scales,
                    &w.biases,
                    &out_row,
                    @intCast(w.rows),
                    @intCast(w.cols),
                    w.group_size,
                    w.scales_dtype_tag,
                    1,
                    0,
                );
            }
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
                    std.log.err("MXFP8 cuBLASLt failed: {s}", .{@errorName(lt_err)});
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

            // Strict NVFP4 policy for multi-row paths:
            // do not dequantize NVFP4 weights; require native FP4 tensor-core route.
            if (rows > 4) {
                log.warn("inference", "CUDA NVFP4 strict mode rejected multi-row linear op (native FP4 route unavailable)", .{
                    .rows = rows,
                    .in_dim = w.rows,
                    .out_dim = w.cols,
                    .has_scales_lt = @as(u8, @intFromBool(w.scales_lt_buffer.size > 0)),
                });
                return error.CudaKernelUnavailable;
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
                w.mean_scale_cache.pointer != 0)
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
                recordNvfp4Route(self, .small_rows_matvec);
                return;
            }

            // Fail closed when the startup parity probe found multi-row mismatches:
            // execute one row at a time on the known-good single-row kernel.
            if (rows > 1 and !self.nvfp4_sequence_rows_supported) {
                var row_index: usize = 0;
                while (row_index < rows) : (row_index += 1) {
                    var input_row = try logicalF32RowSlice(&packed_input, rows, row_index, w.rows);
                    var out_row = try logicalF32RowSlice(&packed_out, rows, row_index, w.cols);
                    self.kernel_arg_pack.reset();
                    self.kernel_arg_pack.appendBufferPtr(&input_row) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendBufferPtr(&w.buffer) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendBufferPtr(&w.scales_buffer) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendBufferPtr(&out_row) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(u32, out_cols) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(u32, w.scale_cols) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(u32, w.group_size) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(f32, w.weight_global_scale) catch return error.CudaKernelUnavailable;
                    self.kernel_arg_pack.appendScalar(u32, 1) catch return error.CudaKernelUnavailable;
                    try compute.cuda.launch.launchWithFamily(&self.device, base_nvfp4_fn, .{
                        .grid_x = (out_cols + 3) / 4,
                        .grid_y = 1,
                        .block_x = 128,
                    }, &self.kernel_arg_pack, .matvec);
                }
                recordNvfp4Route(self, .small_rows_matvec);
                return;
            }

            var nvfp4_fn = base_nvfp4_fn;
            var nvfp4_batch_tile: u32 = 4;
            // Decode is overwhelmingly single-row on NVFP4; prefer the wider
            // tile8 kernel for practical decode projection widths.
            const prefer_tile8 = rows > 4 or (rows == 1 and w.cols >= 2048);
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
            recordNvfp4Route(self, .small_rows_matvec);
            return;
        },
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
                log.warn("inference", "CUDA fused QKV launch failed; falling back to unfused projections", .{
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
    // falling back to three separate projections.
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

pub fn runGateUpProjection(
    self: anytype,
    input: *const compute.cuda.Buffer,
    block: *const LayerAttentionRuntime,
    rows: usize,
) !ProjectionPath {
    return runGateUpProjectionWithWeights(self, input, &block.w1, &block.w3, rows);
}

pub fn runGateUpProjectionWithWeights(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !ProjectionPath {
    const prefer_nvfp4_lt_fused = rows > 1 and switch (gate_weight.*) {
        .nvfp4 => switch (up_weight.*) {
            .nvfp4 => true,
            else => false,
        },
        else => false,
    };
    if (!prefer_nvfp4_lt_fused) {
        if (try tryFusedGateUpForward(self, input, gate_weight, up_weight, rows)) {
            recordPhaseGateUpPath(self, .fused);
            return .fused;
        }
    }

    const gate_bytes = std.math.mul(usize, rows, gate_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    const up_bytes = std.math.mul(usize, rows, up_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    var gate_out = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, gate_bytes);
    var up_out = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, up_bytes);
    if (try tryFusedNvfp4GateUpLtForward(self, input, gate_weight, up_weight, rows, &gate_out, &up_out)) {
        recordPhaseGateUpPath(self, .fused);
        return .fused;
    }
    try linearForwardRows(self, input, rows, gate_weight, &gate_out);
    try linearForwardRows(self, input, rows, up_weight, &up_out);
    recordPhaseGateUpPath(self, .unfused);
    return .unfused;
}

pub fn runFfnActivationMul(self: anytype, count: u32) !void {
    if (self.loaded.config.use_gelu) {
        const gelu_mul_function = self.gelu_mul_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.gelu_mul.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            gelu_mul_function,
            &self.runtime_buffers.ffn_gate_dev,
            &self.runtime_buffers.ffn_up_dev,
            &self.runtime_buffers.ffn_mul_dev,
            count,
        );
        return;
    }

    const silu_mul_function = self.silu_mul_function orelse return error.CudaKernelUnavailable;
    try compute.cuda.silu_mul.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        silu_mul_function,
        &self.runtime_buffers.ffn_gate_dev,
        &self.runtime_buffers.ffn_up_dev,
        &self.runtime_buffers.ffn_mul_dev,
        count,
    );
}

pub fn addResidualWithModelScale(
    self: anytype,
    out: *compute.cuda.Buffer,
    residual: *compute.cuda.Buffer,
    branch: *compute.cuda.Buffer,
    count: u32,
) !void {
    if (self.loaded.config.residual_multiplier == 1.0) {
        const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.vector_add.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            vector_add_function,
            residual,
            branch,
            out,
            count,
        );
        return;
    }

    const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
    try compute.cuda.vector_add_scaled.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        vector_add_scaled_function,
        residual,
        branch,
        out,
        self.loaded.config.residual_multiplier,
        count,
    );
}

pub fn addResidualWithScale(
    self: anytype,
    out: *compute.cuda.Buffer,
    residual: *compute.cuda.Buffer,
    branch: *compute.cuda.Buffer,
    count: u32,
    scale: layer_ops.ResidualScale,
) !void {
    switch (scale) {
        .residual_multiplier => return addResidualWithModelScale(self, out, residual, branch, count),
        .one => {
            const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
            return compute.cuda.vector_add.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_function,
                residual,
                branch,
                out,
                count,
            );
        },
        .literal => |literal| {
            if (literal == 1.0) {
                const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
                return compute.cuda.vector_add.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_function,
                    residual,
                    branch,
                    out,
                    count,
                );
            }
            const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
            return compute.cuda.vector_add_scaled.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_scaled_function,
                residual,
                branch,
                out,
                literal,
                count,
            );
        },
    }
}

pub fn addResidualWithScaleRowsStrideAware(
    self: anytype,
    out: *compute.cuda.Buffer,
    residual: *compute.cuda.Buffer,
    branch: *compute.cuda.Buffer,
    rows: u32,
    cols: u32,
    scale: layer_ops.ResidualScale,
    output_scale: f32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size < packed_bytes or residual.size < packed_bytes or branch.size < packed_bytes) {
        return error.InvalidInstructionBinding;
    }

    const residual_scale: f32 = switch (scale) {
        .one => 1.0,
        .residual_multiplier => self.loaded.config.residual_multiplier,
        .literal => |literal| literal,
    };
    const has_fused_scalar = output_scale != 1.0 and self.residual_add_scaled_rows_strided_function != null;

    if (out.size == packed_bytes and residual.size == packed_bytes and branch.size == packed_bytes) {
        if (has_fused_scalar) {
            try compute.cuda.residual_add_scaled_rows_strided.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.residual_add_scaled_rows_strided_function.?,
                residual,
                branch,
                out,
                residual_scale,
                output_scale,
                rows,
                cols,
                cols,
                cols,
                cols,
            );
            return;
        }
        try addResidualWithScale(self, out, residual, branch, packed_count, scale);
        if (output_scale != 1.0) {
            const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
            try compute.cuda.vector_add_scaled.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_scaled_function,
                out,
                out,
                out,
                output_scale - 1.0,
                packed_count,
            );
        }
        return;
    }

    const row_count: usize = @intCast(rows);
    if (out.size % row_count != 0 or residual.size % row_count != 0 or branch.size % row_count != 0) {
        return error.InvalidInstructionBinding;
    }
    const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
    const out_stride = out.size / row_count;
    const residual_stride = residual.size / row_count;
    const branch_stride = branch.size / row_count;
    if (out_stride < row_bytes or residual_stride < row_bytes or branch_stride < row_bytes) {
        return error.InvalidInstructionBinding;
    }
    if ((out_stride % @sizeOf(f32)) != 0 or
        (residual_stride % @sizeOf(f32)) != 0 or
        (branch_stride % @sizeOf(f32)) != 0)
    {
        return error.InvalidInstructionBinding;
    }

    const out_stride_elems_u32 = std.math.cast(u32, out_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    const residual_stride_elems_u32 = std.math.cast(u32, residual_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    const branch_stride_elems_u32 = std.math.cast(u32, branch_stride / @sizeOf(f32)) orelse return error.InvalidArgument;

    if (has_fused_scalar) {
        try compute.cuda.residual_add_scaled_rows_strided.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.residual_add_scaled_rows_strided_function.?,
            residual,
            branch,
            out,
            residual_scale,
            output_scale,
            rows,
            cols,
            residual_stride_elems_u32,
            branch_stride_elems_u32,
            out_stride_elems_u32,
        );
        return;
    }

    if (self.vector_add_rows_strided_function != null and self.vector_add_scaled_rows_strided_function != null) {
        switch (scale) {
            .one => {
                try compute.cuda.vector_add_rows_strided.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.vector_add_rows_strided_function.?,
                    residual,
                    branch,
                    out,
                    rows,
                    cols,
                    residual_stride_elems_u32,
                    branch_stride_elems_u32,
                    out_stride_elems_u32,
                );
            },
            .residual_multiplier => {
                try compute.cuda.vector_add_scaled_rows_strided.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.vector_add_scaled_rows_strided_function.?,
                    residual,
                    branch,
                    out,
                    self.loaded.config.residual_multiplier,
                    rows,
                    cols,
                    residual_stride_elems_u32,
                    branch_stride_elems_u32,
                    out_stride_elems_u32,
                );
            },
            .literal => |literal| {
                if (literal == 1.0) {
                    try compute.cuda.vector_add_rows_strided.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.vector_add_rows_strided_function.?,
                        residual,
                        branch,
                        out,
                        rows,
                        cols,
                        residual_stride_elems_u32,
                        branch_stride_elems_u32,
                        out_stride_elems_u32,
                    );
                } else {
                    try compute.cuda.vector_add_scaled_rows_strided.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.vector_add_scaled_rows_strided_function.?,
                        residual,
                        branch,
                        out,
                        literal,
                        rows,
                        cols,
                        residual_stride_elems_u32,
                        branch_stride_elems_u32,
                        out_stride_elems_u32,
                    );
                }
            },
        }
        if (output_scale != 1.0) {
            try compute.cuda.vector_add_scaled_rows_strided.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.vector_add_scaled_rows_strided_function.?,
                out,
                out,
                out,
                output_scale - 1.0,
                rows,
                cols,
                out_stride_elems_u32,
                out_stride_elems_u32,
                out_stride_elems_u32,
            );
        }
        return;
    }

    var row_idx: usize = 0;
    while (row_idx < row_count) : (row_idx += 1) {
        const out_offset = std.math.mul(usize, row_idx, out_stride) catch return error.InvalidArgument;
        const residual_offset = std.math.mul(usize, row_idx, residual_stride) catch return error.InvalidArgument;
        const branch_offset = std.math.mul(usize, row_idx, branch_stride) catch return error.InvalidArgument;
        var out_row = try bufferSlice(out, out_offset, row_bytes);
        var residual_row = try bufferSlice(residual, residual_offset, row_bytes);
        var branch_row = try bufferSlice(branch, branch_offset, row_bytes);
        try addResidualWithScale(self, &out_row, &residual_row, &branch_row, cols, scale);
        if (output_scale != 1.0) {
            const vector_add_scaled_function = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;
            try compute.cuda.vector_add_scaled.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_scaled_function,
                &out_row,
                &out_row,
                &out_row,
                output_scale - 1.0,
                cols,
            );
        }
    }
}

pub fn runRmsnormRowsStrideAware(
    self: anytype,
    input: *const compute.cuda.Buffer,
    weight: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    rows: u32,
    cols: u32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size < packed_bytes or output.size < packed_bytes) return error.InvalidInstructionBinding;
    if (input.size == packed_bytes and output.size == packed_bytes) {
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            input,
            weight,
            output,
            rows,
            cols,
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
        return;
    }

    const row_count: usize = @intCast(rows);
    if (input.size % row_count != 0 or output.size % row_count != 0) return error.InvalidInstructionBinding;
    const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
    const input_stride = input.size / row_count;
    const output_stride = output.size / row_count;
    if (input_stride < row_bytes or output_stride < row_bytes) return error.InvalidInstructionBinding;
    if ((input_stride % @sizeOf(f32)) != 0 or (output_stride % @sizeOf(f32)) != 0) {
        return error.InvalidInstructionBinding;
    }

    const input_stride_elems_u32 = std.math.cast(u32, input_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    const output_stride_elems_u32 = std.math.cast(u32, output_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    if (self.rmsnorm_rows_strided_function) |kernel| {
        try compute.cuda.rmsnorm_rows_strided.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            kernel,
            input,
            weight,
            output,
            rows,
            cols,
            input_stride_elems_u32,
            output_stride_elems_u32,
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
        return;
    }

    var row_idx: usize = 0;
    while (row_idx < row_count) : (row_idx += 1) {
        const input_offset = std.math.mul(usize, row_idx, input_stride) catch return error.InvalidArgument;
        const output_offset = std.math.mul(usize, row_idx, output_stride) catch return error.InvalidArgument;
        var input_row = try bufferSlice(input, input_offset, row_bytes);
        var output_row = try bufferSlice(output, output_offset, row_bytes);
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &input_row,
            weight,
            &output_row,
            1,
            cols,
            self.norm_eps,
            self.loaded.runtime.weight_offset,
        );
    }
}

pub fn programBuffer(self: anytype, reg_idx: usize, ctx: anytype) ?*compute.cuda.Buffer {
    _ = self;
    if (reg_idx == 0) return @constCast(&ctx.input_view);
    if (reg_idx >= ctx.register_to_slot_map.len) return null;
    const slot_idx = ctx.register_to_slot_map[reg_idx];
    if (slot_idx == BlockRuntimeLayer.invalid_slot or slot_idx >= ctx.slot_buffers.len) return null;
    return @constCast(&ctx.slot_buffers[slot_idx]);
}
pub fn tryFusedQkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (try tryFusedDenseU16QkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest)) return true;
    if (try tryFusedGaffineU4QkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest)) return true;
    if (try tryFusedNvfp4QkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest)) return true;
    return tryFusedGaffineU8QkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest);
}

pub fn tryFusedGaffineU4QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    var fused_kernel = self.gaffine_u4_matvec_qkv_function orelse return false;
    var use_tile8 = false;
    if (self.gaffine_u4_tile8_enabled and rows > 4) {
        if (self.gaffine_u4_matvec_qkv_tile8_function) |tile8_kernel| {
            fused_kernel = tile8_kernel;
            use_tile8 = true;
        }
    }
    const q = switch (q_proj.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const k = switch (k_proj.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const v = switch (v_proj.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    if (!canFuseGaffineQkvWeights(self.d_model, q, k, v)) return false;
    const batch_rows: u32 = @intCast(rows);

    if (use_tile8) {
        try compute.cuda.gaffine_u4_matvec_qkv.runWithFunctionTile8(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &q.packed_data,
            &q.scales,
            &q.biases,
            q_out_dest,
            @intCast(q.cols),
            q.group_size,
            q.scales_dtype_tag,
            &k.packed_data,
            &k.scales,
            &k.biases,
            &self.runtime_buffers.attn_k_dev,
            @intCast(k.cols),
            k.group_size,
            k.scales_dtype_tag,
            &v.packed_data,
            &v.scales,
            &v.biases,
            &self.runtime_buffers.attn_v_dev,
            @intCast(v.cols),
            v.group_size,
            v.scales_dtype_tag,
            @intCast(q.rows),
            batch_rows,
        );
    } else {
        try compute.cuda.gaffine_u4_matvec_qkv.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &q.packed_data,
            &q.scales,
            &q.biases,
            q_out_dest,
            @intCast(q.cols),
            q.group_size,
            q.scales_dtype_tag,
            &k.packed_data,
            &k.scales,
            &k.biases,
            &self.runtime_buffers.attn_k_dev,
            @intCast(k.cols),
            k.group_size,
            k.scales_dtype_tag,
            &v.packed_data,
            &v.scales,
            &v.biases,
            &self.runtime_buffers.attn_v_dev,
            @intCast(v.cols),
            v.group_size,
            v.scales_dtype_tag,
            @intCast(q.rows),
            batch_rows,
        );
    }
    return true;
}

pub fn tryFusedGaffineU8QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    const fused_kernel = self.gaffine_u8_matvec_qkv_function orelse return false;
    const q = switch (q_proj.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const k = switch (k_proj.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const v = switch (v_proj.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!canFuseGaffineQkvWeights(self.d_model, q, k, v)) return false;
    const batch_rows: u32 = @intCast(rows);

    try compute.cuda.gaffine_u8_matvec_qkv.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_kernel,
        input,
        &q.packed_data,
        &q.scales,
        &q.biases,
        q_out_dest,
        @intCast(q.cols),
        q.group_size,
        q.scales_dtype_tag,
        &k.packed_data,
        &k.scales,
        &k.biases,
        &self.runtime_buffers.attn_k_dev,
        @intCast(k.cols),
        k.group_size,
        k.scales_dtype_tag,
        &v.packed_data,
        &v.scales,
        &v.biases,
        &self.runtime_buffers.attn_v_dev,
        @intCast(v.cols),
        v.group_size,
        v.scales_dtype_tag,
        @intCast(q.rows),
        batch_rows,
    );
    return true;
}

pub fn tryFusedNvfp4QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !self.nvfp4_sequence_fused_qkv_supported) return false;
    const q = switch (q_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const k = switch (k_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const v = switch (v_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (q.rows != k.rows or q.rows != v.rows) return false;
    if (q.rows != self.d_model) return false;
    if (q.weight_global_scale == 0.0 or k.weight_global_scale == 0.0 or v.weight_global_scale == 0.0) return false;

    const q_out_dim: u32 = @intCast(q.cols);
    const k_out_dim: u32 = @intCast(k.cols);
    const v_out_dim: u32 = @intCast(v.cols);
    const in_dim: u32 = @intCast(q.rows);
    const batch_rows: u32 = @intCast(rows);
    const qk_dim = std.math.add(u32, q_out_dim, k_out_dim) catch return false;
    const total_out = std.math.add(u32, qk_dim, v_out_dim) catch return false;

    var fused_fn = self.nvfp4_matvec_qkv_function orelse return false;
    var nvfp4_batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and total_out >= 3072);
    if (prefer_tile8) {
        if (self.nvfp4_matvec_qkv_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            nvfp4_batch_tile = 8;
        }
    }

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&q.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&q.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(q_out_dest);
    try self.kernel_arg_pack.appendScalar(u32, q_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, q.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, q.group_size);
    try self.kernel_arg_pack.appendScalar(f32, q.weight_global_scale);
    try self.kernel_arg_pack.appendBufferPtr(&k.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&k.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.attn_k_dev);
    try self.kernel_arg_pack.appendScalar(u32, k_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, k.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, k.group_size);
    try self.kernel_arg_pack.appendScalar(f32, k.weight_global_scale);
    try self.kernel_arg_pack.appendBufferPtr(&v.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&v.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.attn_v_dev);
    try self.kernel_arg_pack.appendScalar(u32, v_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, v.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, v.group_size);
    try self.kernel_arg_pack.appendScalar(f32, v.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (total_out + 3) / 4,
        .grid_y = (batch_rows + nvfp4_batch_tile - 1) / nvfp4_batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_qkv);
    recordNvfp4Route(self, .fused_qkv);
    return true;
}

pub fn tryFusedNvfp4QkvLtForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out: *compute.cuda.Buffer,
    k_out: *compute.cuda.Buffer,
    v_out: *compute.cuda.Buffer,
) !bool {
    if (rows <= 32) return false;
    var blas_lt = self.blas_lt orelse return false;
    const q = switch (q_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const k = switch (k_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const v = switch (v_proj.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (q.rows != k.rows or q.rows != v.rows) return false;
    if (q.weight_global_scale == 0.0 or k.weight_global_scale == 0.0 or v.weight_global_scale == 0.0) return false;
    if (q.scales_lt_buffer.size == 0 or k.scales_lt_buffer.size == 0 or v.scales_lt_buffer.size == 0) return false;

    var input_fp4_buf: compute.cuda.Buffer = undefined;
    var input_scales_buf: compute.cuda.Buffer = undefined;
    if (!(try tryPrepareNvfp4LtInput(self, input, q.rows, rows, &input_fp4_buf, &input_scales_buf))) return false;

    blas_lt.matmulNvfp4(
        &self.device,
        &q.buffer,
        &q.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        q_out,
        rows,
        q.cols,
        q.rows,
        1.0 / q.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        &self.device,
        &k.buffer,
        &k.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        k_out,
        rows,
        k.cols,
        k.rows,
        1.0 / k.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        &self.device,
        &v.buffer,
        &v.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        v_out,
        rows,
        v.cols,
        v.rows,
        1.0 / v.weight_global_scale,
    ) catch return false;
    recordNvfp4Route(self, .fused_qkv);
    return true;
}

pub fn tryFusedDenseU16QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    const q = switch (q_proj.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const k = switch (k_proj.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const v = switch (v_proj.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (!canFuseDenseU16QkvWeights(self.d_model, q, k, v)) return false;

    const batch_rows: u32 = @intCast(rows);
    const in_dim: u32 = @intCast(q.rows);
    const q_out_dim: u32 = @intCast(q.cols);
    const k_out_dim: u32 = @intCast(k.cols);
    const v_out_dim: u32 = @intCast(v.cols);

    const fused_batch_kernel = switch (q.dtype) {
        .f16 => self.matvec_qkv_f16_function orelse return false,
        .bf16 => self.matvec_qkv_bf16_function orelse return false,
    };

    try compute.cuda.matvec_u16_qkv.runWithFunctionGridBatch(
        &self.kernel_arg_pack,
        &self.device,
        fused_batch_kernel,
        input,
        &q.buffer,
        q_out_dest,
        q_out_dim,
        &k.buffer,
        &self.runtime_buffers.attn_k_dev,
        k_out_dim,
        &v.buffer,
        &self.runtime_buffers.attn_v_dev,
        v_out_dim,
        in_dim,
        batch_rows,
    );
    return true;
}

pub fn canFuseDenseU16QkvWeights(d_model: usize, q: U16LinearWeight, k: U16LinearWeight, v: U16LinearWeight) bool {
    if (q.rows != d_model or k.rows != d_model or v.rows != d_model) return false;
    if (q.dtype != k.dtype or q.dtype != v.dtype) return false;
    if (q.cols > std.math.maxInt(u32) or
        k.cols > std.math.maxInt(u32) or
        v.cols > std.math.maxInt(u32) or
        q.rows > std.math.maxInt(u32))
    {
        return false;
    }
    return true;
}

pub fn canFuseGaffineQkvWeights(
    d_model: usize,
    q: anytype,
    k: anytype,
    v: anytype,
) bool {
    if (q.rows != d_model or k.rows != d_model or v.rows != d_model) return false;
    if (q.scales_dtype_tag != k.scales_dtype_tag or q.scales_dtype_tag != v.scales_dtype_tag) return false;
    if (q.cols > std.math.maxInt(u32) or
        k.cols > std.math.maxInt(u32) or
        v.cols > std.math.maxInt(u32) or
        q.rows > std.math.maxInt(u32))
    {
        return false;
    }
    return true;
}

pub fn canFuseGaffineGateUpWeights(
    d_model: usize,
    gate: anytype,
    up: anytype,
) bool {
    if (gate.rows != d_model or up.rows != d_model) return false;
    if (gate.scales_dtype_tag != up.scales_dtype_tag) return false;
    if (gate.cols > std.math.maxInt(u32) or
        up.cols > std.math.maxInt(u32) or
        gate.rows > std.math.maxInt(u32))
    {
        return false;
    }
    return true;
}

pub fn linearWeightHasI8Cache(weight: *const LinearWeight) bool {
    return switch (weight.*) {
        .gaffine_u4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .gaffine_u8 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        .nvfp4 => |w| w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0,
        else => false,
    };
}

pub fn tryFusedGaffineU8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;

    const gate = switch (gate_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!canFuseGaffineGateUpWeights(self.d_model, gate, up)) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;

    const row_count = bufferF32RowCount(input, gate.rows) catch return false;
    if (row_count != rows) return false;
    const batch_rows: u32 = @intCast(rows);

    const fused_kernel = self.gaffine_u8_matvec_gate_up_silu_function orelse return false;
    try compute.cuda.gaffine_u8_matvec_gate_up_silu.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_kernel,
        input,
        &gate.packed_data,
        &gate.scales,
        &gate.biases,
        &up.packed_data,
        &up.scales,
        &up.biases,
        &self.runtime_buffers.ffn_mul_dev,
        @intCast(gate.cols),
        gate.group_size,
        gate.scales_dtype_tag,
        up.group_size,
        up.scales_dtype_tag,
        @intCast(gate.rows),
        batch_rows,
    );
    return true;
}

pub fn tryFusedGaffineU4GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > 32) return false;

    const gate = switch (gate_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    if (!canFuseGaffineGateUpWeights(self.d_model, gate, up)) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;

    var fused_kernel = self.gaffine_u4_matvec_gate_up_silu_function orelse return false;
    var use_tile8 = false;
    if (self.gaffine_u4_tile8_enabled and rows > 4) {
        if (self.gaffine_u4_matvec_gate_up_silu_tile8_function) |tile8_kernel| {
            fused_kernel = tile8_kernel;
            use_tile8 = true;
        }
    }
    if (use_tile8) {
        try compute.cuda.gaffine_u4_matvec_gate_up_silu.runWithFunctionTile8(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &gate.packed_data,
            &gate.scales,
            &gate.biases,
            &up.packed_data,
            &up.scales,
            &up.biases,
            &self.runtime_buffers.ffn_mul_dev,
            @intCast(gate.cols),
            gate.group_size,
            gate.scales_dtype_tag,
            up.group_size,
            up.scales_dtype_tag,
            @intCast(gate.rows),
            @intCast(rows),
        );
    } else {
        try compute.cuda.gaffine_u4_matvec_gate_up_silu.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            fused_kernel,
            input,
            &gate.packed_data,
            &gate.scales,
            &gate.biases,
            &up.packed_data,
            &up.scales,
            &up.biases,
            &self.runtime_buffers.ffn_mul_dev,
            @intCast(gate.cols),
            gate.group_size,
            gate.scales_dtype_tag,
            up.group_size,
            up.scales_dtype_tag,
            @intCast(gate.rows),
            @intCast(rows),
        );
    }
    return true;
}

pub fn tryFusedFp8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > 32) return false;

    const gate = switch (gate_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    // Both projections must share in_dim and have matching out_dim.
    if (gate.rows != up.rows) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.scales_buffer.pointer == 0 or up.scales_buffer.pointer == 0) return false;

    var fused_fn = self.fp8_matvec_gate_up_silu_function orelse return false;
    var fp8_batch_tile: u32 = 4;
    if (rows > 4) {
        if (self.fp8_matvec_gate_up_silu_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            fp8_batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(gate.cols);
    const in_dim: u32 = @intCast(gate.rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_mul_dev);
    try self.kernel_arg_pack.appendScalar(u32, out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.block_size);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, @intCast(rows));

    const batch_rows: u32 = @intCast(rows);
    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + fp8_batch_tile - 1) / fp8_batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_gate_up_silu);
    return true;
}

pub fn tryFusedFp8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (rows == 0 or rows > 32) return false;

    const gate = switch (gate_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.scales_buffer.pointer == 0 or up.scales_buffer.pointer == 0) return false;

    var fused_fn = self.fp8_matvec_gate_up_function orelse return false;
    var fp8_batch_tile: u32 = 4;
    if (rows > 4) {
        if (self.fp8_matvec_gate_up_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            fp8_batch_tile = 8;
        }
    }

    const gate_out_dim: u32 = @intCast(gate.cols);
    const up_out_dim: u32 = @intCast(up.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const total_dim = gate_out_dim + up_out_dim;

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_gate_dev);
    try self.kernel_arg_pack.appendScalar(u32, gate_out_dim);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_up_dev);
    try self.kernel_arg_pack.appendScalar(u32, up_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.block_size);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, @intCast(rows));

    const fp8_batch_rows: u32 = @intCast(rows);
    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (fp8_batch_rows + fp8_batch_tile - 1) / fp8_batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec);
    return true;
}

pub fn tryFusedMxfp8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    // At n>4 the tile-8 GEMV drops to 50% occupancy; prefer falling through
    // to two separate cuBLASLt calls which achieve better bandwidth.
    if (rows == 0 or rows > 4) return false;

    const gate = switch (gate_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.scales_raw_buffer.pointer == 0 or up.scales_raw_buffer.pointer == 0) return false;

    var fused_fn = self.mxfp8_matvec_gate_up_silu_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (self.mxfp8_matvec_gate_up_silu_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(gate.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const batch_rows: u32 = @intCast(rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_raw_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_raw_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_mul_dev);
    try self.kernel_arg_pack.appendScalar(u32, out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_gate_up_silu);
    return true;
}

pub fn tryFusedNvfp4GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !self.nvfp4_sequence_fused_gate_up_supported) return false;

    const gate = switch (gate_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.group_size == 0 or up.group_size == 0) return false;
    if ((gate.rows % gate.group_size) != 0 or (up.rows % up.group_size) != 0) return false;
    if (gate.scales_buffer.pointer == 0 or up.scales_buffer.pointer == 0) return false;
    if (gate.weight_global_scale == 0.0 or up.weight_global_scale == 0.0) return false;

    var fused_fn = self.nvfp4_matvec_gate_up_silu_function orelse return false;
    var batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and gate.cols >= 8192);
    if (prefer_tile8) {
        if (self.nvfp4_matvec_gate_up_silu_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(gate.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const batch_rows: u32 = @intCast(rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_mul_dev);
    try self.kernel_arg_pack.appendScalar(u32, out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, gate.group_size);
    try self.kernel_arg_pack.appendScalar(f32, gate.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.group_size);
    try self.kernel_arg_pack.appendScalar(f32, up.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_gate_up_silu);
    recordNvfp4Route(self, .fused_gate_up);
    return true;
}

pub fn tryFusedNvfp4GateUpGeluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (!self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !self.nvfp4_sequence_fused_gate_up_supported) return false;

    const gate = switch (gate_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.group_size == 0 or up.group_size == 0) return false;
    if ((gate.rows % gate.group_size) != 0 or (up.rows % up.group_size) != 0) return false;
    if (gate.scales_buffer.pointer == 0 or up.scales_buffer.pointer == 0) return false;
    if (gate.weight_global_scale == 0.0 or up.weight_global_scale == 0.0) return false;

    var fused_fn = self.nvfp4_matvec_gate_up_gelu_function orelse return false;
    var batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and gate.cols >= 8192);
    if (prefer_tile8) {
        if (self.nvfp4_matvec_gate_up_gelu_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(gate.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const batch_rows: u32 = @intCast(rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_mul_dev);
    try self.kernel_arg_pack.appendScalar(u32, out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, gate.group_size);
    try self.kernel_arg_pack.appendScalar(f32, gate.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.group_size);
    try self.kernel_arg_pack.appendScalar(f32, up.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_gate_up_silu);
    recordNvfp4Route(self, .fused_gate_up);
    return true;
}

pub fn tryFusedMxfp8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (rows == 0 or rows > 4) return false;

    const gate = switch (gate_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.scales_raw_buffer.pointer == 0 or up.scales_raw_buffer.pointer == 0) return false;

    var fused_fn = self.mxfp8_matvec_gate_up_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (self.mxfp8_matvec_gate_up_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            batch_tile = 8;
        }
    }

    const gate_out_dim: u32 = @intCast(gate.cols);
    const up_out_dim: u32 = @intCast(up.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const total_dim = gate_out_dim + up_out_dim;
    const batch_rows: u32 = @intCast(rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_raw_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_gate_dev);
    try self.kernel_arg_pack.appendScalar(u32, gate_out_dim);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_raw_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_up_dev);
    try self.kernel_arg_pack.appendScalar(u32, up_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec);
    return true;
}

pub fn tryFusedNvfp4GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    // Fused GEMV gate+up: efficient for decode (small row counts).
    // For prefill (rows > 32), per-projection cuBLASLt GEMM is faster.
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !self.nvfp4_sequence_fused_gate_up_supported) return false;
    const gate = switch (gate_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.rows != self.d_model) return false;
    if (gate.weight_global_scale == 0.0 or up.weight_global_scale == 0.0) return false;
    var fused_fn = self.nvfp4_matvec_gate_up_function orelse return false;
    var nvfp4_batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and (gate.cols + up.cols) >= 8192);
    if (prefer_tile8) {
        if (self.nvfp4_matvec_gate_up_tile8_function) |tile8_fn| {
            fused_fn = tile8_fn;
            nvfp4_batch_tile = 8;
        }
    }

    const gate_bytes = std.math.mul(usize, rows, gate.cols * @sizeOf(f32)) catch return false;
    const up_bytes = std.math.mul(usize, rows, up.cols * @sizeOf(f32)) catch return false;
    _ = gate_bytes;
    _ = up_bytes;

    const gate_out_dim: u32 = @intCast(gate.cols);
    const up_out_dim: u32 = @intCast(up.cols);
    const in_dim: u32 = @intCast(gate.rows);
    const batch_rows: u32 = @intCast(rows);
    const total_dim = std.math.add(u32, gate_out_dim, up_out_dim) catch return false;

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&gate.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_gate_dev);
    try self.kernel_arg_pack.appendScalar(u32, gate_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, gate.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, gate.group_size);
    try self.kernel_arg_pack.appendScalar(f32, gate.weight_global_scale);
    try self.kernel_arg_pack.appendBufferPtr(&up.buffer);
    try self.kernel_arg_pack.appendBufferPtr(&up.scales_buffer);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_up_dev);
    try self.kernel_arg_pack.appendScalar(u32, up_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, up.scale_cols);
    try self.kernel_arg_pack.appendScalar(u32, up.group_size);
    try self.kernel_arg_pack.appendScalar(f32, up.weight_global_scale);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, fused_fn, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (batch_rows + nvfp4_batch_tile - 1) / nvfp4_batch_tile,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec);
    recordNvfp4Route(self, .fused_gate_up);
    return true;
}

pub fn tryFusedNvfp4GateUpLtForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    gate_out: *compute.cuda.Buffer,
    up_out: *compute.cuda.Buffer,
) !bool {
    if (rows <= 32) return false;
    var blas_lt = self.blas_lt orelse return false;
    const gate = switch (gate_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (gate.rows != up.rows) return false;
    if (gate.weight_global_scale == 0.0 or up.weight_global_scale == 0.0) return false;
    if (gate.scales_lt_buffer.size == 0 or up.scales_lt_buffer.size == 0) return false;

    var input_fp4_buf: compute.cuda.Buffer = undefined;
    var input_scales_buf: compute.cuda.Buffer = undefined;
    if (!(try tryPrepareNvfp4LtInput(self, input, gate.rows, rows, &input_fp4_buf, &input_scales_buf))) return false;

    blas_lt.matmulNvfp4(
        &self.device,
        &gate.buffer,
        &gate.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        gate_out,
        rows,
        gate.cols,
        gate.rows,
        1.0 / gate.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        &self.device,
        &up.buffer,
        &up.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        up_out,
        rows,
        up.cols,
        up.rows,
        1.0 / up.weight_global_scale,
    ) catch return false;
    recordNvfp4Route(self, .fused_gate_up);
    return true;
}

pub fn tryFusedGateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (try tryFusedDenseU16GateUpForward(self, input, gate_weight, up_weight, rows)) return true;
    if (try tryFusedMxfp8GateUpForward(self, input, gate_weight, up_weight, rows)) return true;
    if (try tryFusedFp8GateUpForward(self, input, gate_weight, up_weight, rows)) return true;
    // For decode/small batches, keep NVFP4 gate/up on the canonical gate+up
    // buffer path first.
    if (try tryFusedNvfp4GateUpForward(self, input, gate_weight, up_weight, rows)) return true;
    // The U4 fused gate/up+silu path is currently unstable for quality-sensitive
    // generation; keep the established U8 path as the default correctness path.
    return tryFusedGaffineU8GateUpForward(self, input, gate_weight, up_weight, rows);
}

pub fn tryFusedGaffineU8GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    const gate = switch (gate_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!canFuseGaffineGateUpWeights(self.d_model, gate, up)) return false;
    const row_count = bufferF32RowCount(input, gate.rows) catch return false;
    if (row_count != rows) return false;
    const batch_rows: u32 = @intCast(rows);

    const fused_kernel = self.gaffine_u8_matvec_gate_up_function orelse return false;
    try compute.cuda.gaffine_u8_matvec_gate_up.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_kernel,
        input,
        &gate.packed_data,
        &gate.scales,
        &gate.biases,
        &self.runtime_buffers.ffn_gate_dev,
        @intCast(gate.cols),
        gate.group_size,
        gate.scales_dtype_tag,
        &up.packed_data,
        &up.scales,
        &up.biases,
        &self.runtime_buffers.ffn_up_dev,
        @intCast(up.cols),
        up.group_size,
        up.scales_dtype_tag,
        @intCast(gate.rows),
        batch_rows,
    );
    return true;
}

pub fn tryFusedDenseU16GateUpForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
) !bool {
    const gate = switch (gate_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (gate.rows != self.d_model or up.rows != self.d_model) return false;
    if (gate.dtype != up.dtype) return false;
    if (gate.cols > std.math.maxInt(u32) or
        up.cols > std.math.maxInt(u32) or
        gate.rows > std.math.maxInt(u32))
    {
        return false;
    }
    const row_count = bufferF32RowCount(input, gate.rows) catch return false;
    if (row_count != rows or rows != 1) return false;

    const fused_kernel = switch (gate.dtype) {
        .f16 => self.matvec_gate_up_f16_function orelse return false,
        .bf16 => self.matvec_gate_up_bf16_function orelse return false,
    };
    try compute.cuda.matvec_u16_gate_up.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_kernel,
        input,
        &gate.buffer,
        &self.runtime_buffers.ffn_gate_dev,
        @intCast(gate.cols),
        &up.buffer,
        &self.runtime_buffers.ffn_up_dev,
        @intCast(up.cols),
        @intCast(gate.rows),
    );
    return true;
}

pub fn tryFusedDenseU16GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > 32) return false;

    const gate = switch (gate_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (gate.rows != self.d_model or up.rows != self.d_model) return false;
    if (gate.dtype != up.dtype) return false;
    if (gate.cols != up.cols) return false;
    if (gate.cols != expected_out_dim) return false;
    if (gate.cols > std.math.maxInt(u32) or gate.rows > std.math.maxInt(u32)) return false;

    const fused_kernel = switch (gate.dtype) {
        .f16 => self.matvec_gate_up_silu_f16_function orelse return false,
        .bf16 => self.matvec_gate_up_silu_bf16_function orelse return false,
    };
    try compute.cuda.matvec_u16_gate_up_silu.runWithFunctionGridBatch(
        &self.kernel_arg_pack,
        &self.device,
        fused_kernel,
        input,
        &gate.buffer,
        &up.buffer,
        &self.runtime_buffers.ffn_mul_dev,
        @intCast(gate.cols),
        @intCast(gate.rows),
        @intCast(rows),
    );
    return true;
}
