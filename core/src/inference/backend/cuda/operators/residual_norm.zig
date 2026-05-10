//! Residual and normalization helpers for the CUDA inference backend.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
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
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

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
