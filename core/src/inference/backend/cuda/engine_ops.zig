//! Compute operation building blocks and fused operation attempts.
//!
//! Contains linear forward, QKV projection, gate-up projection, FFN activation,
//! residual addition, RMSNorm, and all fused operation variants.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LinearWeight = engine_types.LinearWeight;
const U16LinearWeight = engine_types.U16LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const ProjectionPath = engine_types.ProjectionPath;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

const models = @import("../../../models/root.zig");
const layer_ops = models.layer_ops;

// --- Utility functions from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const bufferSlice = engine_weights.bufferSlice;

pub fn compactQueryGateProjection(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
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
        @intCast(self.n_heads),
        @intCast(self.head_dim),
    );
}

pub fn applyQueryGateToContextInPlace(
    self: anytype,
    seq_len: usize,
    q_dim: usize,
    q_projection_dim: usize,
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
        @intCast(self.n_heads),
        @intCast(self.head_dim),
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
            // Batched decode path: native batched GEMV avoids cast-to-u16 +
            // BLAS dispatch overhead. Kernel tiles in groups of 8 rows via
            // blockIdx.y, so any batch size works.
            if (rows <= 32) {
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
            const kernel = self.gaffine_u4_matvec_function orelse return error.CudaKernelUnavailable;
            // Native U4 GEMV: reads weights in 4-bit format (half the DRAM
            // bandwidth of the I8 BLAS path). The kernel tiles batch rows
            // via grid_y, so any batch_rows value works.
            if (rows <= 32) {
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
    const prefer_i8_concat = self.active_qkv_concat != null and rows > 1 and self.i8_blas_supported;
    if (!prefer_i8_concat and q_out_dest.size >= q_bytes and
        try tryFusedQkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest))
    {
        return .fused;
    }
    var q_out = if (q_out_dest.size == q_bytes)
        q_out_dest.*
    else
        try bufferSlice(q_out_dest, 0, q_bytes);
    var k_out = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, k_bytes);
    var v_out = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, v_bytes);

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
            return .unfused;
        }
    }

    // If concat-I8 path is unavailable/failed, retry fused QKV kernel before
    // falling back to three separate projections.
    if (q_out_dest.size >= q_bytes and
        try tryFusedQkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest))
    {
        return .fused;
    }

    try linearForwardRows(self, input, rows, q_proj, &q_out);
    try linearForwardRows(self, input, rows, k_proj, &k_out);
    try linearForwardRows(self, input, rows, v_proj, &v_out);
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
    if (try tryFusedGateUpForward(self, input, gate_weight, up_weight, rows)) return .fused;

    const gate_bytes = std.math.mul(usize, rows, gate_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    const up_bytes = std.math.mul(usize, rows, up_weight.cols() * @sizeOf(f32)) catch return error.InvalidArgument;
    var gate_out = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, gate_bytes);
    var up_out = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, up_bytes);
    try linearForwardRows(self, input, rows, gate_weight, &gate_out);
    try linearForwardRows(self, input, rows, up_weight, &up_out);
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
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (out.size == packed_bytes and residual.size == packed_bytes and branch.size == packed_bytes) {
        return addResidualWithScale(self, out, residual, branch, packed_count, scale);
    }
    if (out.size < packed_bytes or residual.size < packed_bytes or branch.size < packed_bytes) {
        return error.InvalidInstructionBinding;
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
    if (try tryFusedI8QkvForward(self, input, q_proj, k_proj, v_proj, rows, q_out_dest)) return true;
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
    const fused_kernel = self.gaffine_u4_matvec_qkv_function orelse return false;
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
    return true;
}

pub fn tryFusedI8QkvForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    rows: usize,
    q_out_dest: *compute.cuda.Buffer,
) !bool {
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    const i8_fn = self.i8_matvec_qkv_function orelse return false;
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
    // All three projections must have I8 caches populated.
    if (q.dequant_i8_cache.pointer == 0 or q.mean_scale_cache.pointer == 0) return false;
    if (k.dequant_i8_cache.pointer == 0 or k.mean_scale_cache.pointer == 0) return false;
    if (v.dequant_i8_cache.pointer == 0 or v.mean_scale_cache.pointer == 0) return false;
    if (!canFuseGaffineQkvWeights(self.d_model, q, k, v)) return false;
    const batch_rows: u32 = @intCast(rows);

    const q_out_dim: u32 = @intCast(q.cols);
    const k_out_dim: u32 = @intCast(k.cols);
    const v_out_dim: u32 = @intCast(v.cols);
    const in_dim: u32 = @intCast(q.rows);
    const total_out = std.math.add(u32, q_out_dim, std.math.add(u32, k_out_dim, v_out_dim) catch return error.InvalidArgument) catch return error.InvalidArgument;

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&q.dequant_i8_cache);
    try self.kernel_arg_pack.appendBufferPtr(&q.mean_scale_cache);
    try self.kernel_arg_pack.appendBufferPtr(q_out_dest);
    try self.kernel_arg_pack.appendScalar(u32, q_out_dim);
    try self.kernel_arg_pack.appendBufferPtr(&k.dequant_i8_cache);
    try self.kernel_arg_pack.appendBufferPtr(&k.mean_scale_cache);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.attn_k_dev);
    try self.kernel_arg_pack.appendScalar(u32, k_out_dim);
    try self.kernel_arg_pack.appendBufferPtr(&v.dequant_i8_cache);
    try self.kernel_arg_pack.appendBufferPtr(&v.mean_scale_cache);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.attn_v_dev);
    try self.kernel_arg_pack.appendScalar(u32, v_out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, i8_fn, .{
        .grid_x = (total_out + 3) / 4,
        .grid_y = batch_rows,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_qkv);
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

pub fn tryFusedI8GateUpSiluForward(
    self: anytype,
    input: *const compute.cuda.Buffer,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    rows: usize,
    expected_out_dim: u32,
) !bool {
    if (self.loaded.config.use_gelu) return false;
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    const i8_fn = self.i8_matvec_gate_up_silu_function orelse return false;

    const gate = switch (gate_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const up = switch (up_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    // Both must have I8 caches populated.
    if (gate.dequant_i8_cache.pointer == 0 or gate.mean_scale_cache.pointer == 0) return false;
    if (up.dequant_i8_cache.pointer == 0 or up.mean_scale_cache.pointer == 0) return false;
    if (!canFuseGaffineGateUpWeights(self.d_model, gate, up)) return false;
    if (gate.cols != up.cols or gate.cols != expected_out_dim) return false;

    const row_count = bufferF32RowCount(input, gate.rows) catch return false;
    if (row_count != rows) return false;
    const batch_rows: u32 = @intCast(rows);

    const out_dim: u32 = @intCast(gate.cols);
    const in_dim: u32 = @intCast(gate.rows);

    self.kernel_arg_pack.reset();
    try self.kernel_arg_pack.appendBufferPtr(input);
    try self.kernel_arg_pack.appendBufferPtr(&gate.dequant_i8_cache);
    try self.kernel_arg_pack.appendBufferPtr(&gate.mean_scale_cache);
    try self.kernel_arg_pack.appendBufferPtr(&up.dequant_i8_cache);
    try self.kernel_arg_pack.appendBufferPtr(&up.mean_scale_cache);
    try self.kernel_arg_pack.appendBufferPtr(&self.runtime_buffers.ffn_mul_dev);
    try self.kernel_arg_pack.appendScalar(u32, out_dim);
    try self.kernel_arg_pack.appendScalar(u32, in_dim);
    try self.kernel_arg_pack.appendScalar(u32, batch_rows);

    try compute.cuda.launch.launchWithFamily(&self.device, i8_fn, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = batch_rows,
        .block_x = 128,
    }, &self.kernel_arg_pack, .matvec_gate_up_silu);
    return true;
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

    const fused_kernel = self.gaffine_u4_matvec_gate_up_silu_function orelse return false;
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
