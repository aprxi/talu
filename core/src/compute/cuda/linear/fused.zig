//! Generic fused CUDA linear projection execution for two- and three-output groups.

const std = @import("std");

const args_mod = @import("../args.zig");
const device_mod = @import("../device.zig");
const gaffine_u4_matvec_gate_up_silu = @import("../gaffine_u4_matvec_gate_up_silu.zig");
const gaffine_u4_matvec_qkv = @import("../gaffine_u4_matvec_qkv.zig");
const gaffine_u8_matvec_gate_up = @import("../gaffine_u8_matvec_gate_up.zig");
const gaffine_u8_matvec_gate_up_silu = @import("../gaffine_u8_matvec_gate_up_silu.zig");
const gaffine_u8_matvec_qkv = @import("../gaffine_u8_matvec_qkv.zig");
const launch = @import("../launch.zig");
const matmul = @import("../matmul.zig");
const matvec_u16_gate_up = @import("../matvec_u16_gate_up.zig");
const matvec_u16_gate_up_silu = @import("../matvec_u16_gate_up_silu.zig");
const matvec_u16_qkv = @import("../matvec_u16_qkv.zig");
const module = @import("../module.zig");
const weights = @import("weights.zig");

const Buffer = device_mod.Buffer;
const LinearWeight = weights.LinearWeight;
const U16LinearWeight = weights.U16LinearWeight;
const GaffineU4LinearWeight = weights.GaffineU4LinearWeight;
const GaffineU8LinearWeight = weights.GaffineU8LinearWeight;
const Nvfp4LinearWeight = weights.Nvfp4LinearWeight;

pub const PairActivation = enum {
    none,
    silu,
    gelu,
};

pub const PairOutputs = struct {
    first: *Buffer,
    second: *Buffer,
    activated_product: ?*Buffer = null,
};

pub const TripleOutputs = struct {
    first: *Buffer,
    second: *Buffer,
    third: *Buffer,
};

pub const ConcatI8TripleWeight = struct {
    input_dim: usize,
    output_dims: [3]u32,
    i8_buffer: Buffer,
    scales_buffer: Buffer,

    pub fn validate(self: *const ConcatI8TripleWeight) !void {
        try validateConcatI8TripleWeight(self);
    }
};

pub const Nvfp4RouteKind = enum {
    pair_custom_kernel,
    pair_native_cublaslt,
    triple_custom_kernel,
    triple_native_cublaslt,
};

pub const Diagnostics = struct {
    nvfp4_route: ?Nvfp4RouteKind = null,

    pub fn reset(self: *Diagnostics) void {
        self.* = .{};
    }
};

pub const CapabilityFlags = struct {
    i8_blas_supported: bool = true,
    gaffine_u4_tile8_enabled: bool = false,
    nvfp4_pair_multi_row_supported: bool = false,
    nvfp4_triple_multi_row_supported: bool = false,
    nvfp4_custom_supported: bool = true,

    pub fn disableI8Blas(self: *CapabilityFlags) void {
        self.i8_blas_supported = false;
    }
};

pub const Workspace = struct {
    activation_scratch: Buffer,
    auxiliary_scratch: Buffer,
};

pub const FusedContext = struct {
    device: *device_mod.Device,
    arg_pack: *args_mod.ArgPack,
    blas: *matmul.Blas,
    blas_lt: ?*matmul.BlasLt,
    workspace: Workspace,
    capabilities: *CapabilityFlags,
    diagnostics: *Diagnostics,

    dense_u16_pair_f16_function: ?module.Function = null,
    dense_u16_pair_bf16_function: ?module.Function = null,
    dense_u16_pair_silu_f16_function: ?module.Function = null,
    dense_u16_pair_silu_bf16_function: ?module.Function = null,
    dense_u16_triple_f16_function: ?module.Function = null,
    dense_u16_triple_bf16_function: ?module.Function = null,

    gaffine_u4_pair_silu_function: ?module.Function = null,
    gaffine_u4_pair_silu_tile8_function: ?module.Function = null,
    gaffine_u4_triple_function: ?module.Function = null,
    gaffine_u4_triple_tile8_function: ?module.Function = null,
    gaffine_u8_pair_function: ?module.Function = null,
    gaffine_u8_pair_silu_function: ?module.Function = null,
    gaffine_u8_triple_function: ?module.Function = null,

    fp8_pair_function: ?module.Function = null,
    fp8_pair_tile8_function: ?module.Function = null,
    fp8_pair_silu_function: ?module.Function = null,
    fp8_pair_silu_tile8_function: ?module.Function = null,
    mxfp8_pair_function: ?module.Function = null,
    mxfp8_pair_tile8_function: ?module.Function = null,
    mxfp8_pair_silu_function: ?module.Function = null,
    mxfp8_pair_silu_tile8_function: ?module.Function = null,

    nvfp4_pair_function: ?module.Function = null,
    nvfp4_pair_tile8_function: ?module.Function = null,
    nvfp4_pair_silu_function: ?module.Function = null,
    nvfp4_pair_silu_tile8_function: ?module.Function = null,
    nvfp4_pair_gelu_function: ?module.Function = null,
    nvfp4_pair_gelu_tile8_function: ?module.Function = null,
    nvfp4_triple_function: ?module.Function = null,
    nvfp4_triple_tile8_function: ?module.Function = null,

    quantize_f32_to_nvfp4_function: ?module.Function = null,
    quantize_f32_to_i8_simple_function: ?module.Function = null,
    dequant_i32_scales_split3_function: ?module.Function = null,
};

pub fn denseU16PairWeightsCompatible(
    input_dim: usize,
    first: U16LinearWeight,
    second: U16LinearWeight,
) bool {
    if (first.rows != input_dim or second.rows != input_dim) return false;
    if (first.dtype != second.dtype) return false;
    if (!u32Fits(first.cols) or !u32Fits(second.cols) or !u32Fits(first.rows)) return false;
    return true;
}

pub fn denseU16TripleWeightsCompatible(
    input_dim: usize,
    first: U16LinearWeight,
    second: U16LinearWeight,
    third: U16LinearWeight,
) bool {
    if (first.rows != input_dim or second.rows != input_dim or third.rows != input_dim) return false;
    if (first.dtype != second.dtype or first.dtype != third.dtype) return false;
    if (!u32Fits(first.cols) or !u32Fits(second.cols) or !u32Fits(third.cols) or !u32Fits(first.rows)) return false;
    return true;
}

pub fn gaffinePairWeightsCompatible(
    input_dim: usize,
    first: anytype,
    second: anytype,
) bool {
    if (first.rows != input_dim or second.rows != input_dim) return false;
    if (first.scales_dtype_tag != second.scales_dtype_tag) return false;
    if (!u32Fits(first.cols) or !u32Fits(second.cols) or !u32Fits(first.rows)) return false;
    return true;
}

pub fn gaffineTripleWeightsCompatible(
    input_dim: usize,
    first: anytype,
    second: anytype,
    third: anytype,
) bool {
    if (first.rows != input_dim or second.rows != input_dim or third.rows != input_dim) return false;
    if (first.scales_dtype_tag != second.scales_dtype_tag or first.scales_dtype_tag != third.scales_dtype_tag) return false;
    if (!u32Fits(first.cols) or !u32Fits(second.cols) or !u32Fits(third.cols) or !u32Fits(first.rows)) return false;
    return true;
}

pub fn tryPairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    ctx.diagnostics.reset();
    if (try tryDenseU16PairSplit(ctx, input, first_weight, second_weight, rows, input_dim, outputs)) return true;
    if (try tryMxfp8PairSplit(ctx, input, first_weight, second_weight, rows, input_dim, outputs)) return true;
    if (try tryFp8PairSplit(ctx, input, first_weight, second_weight, rows, input_dim, outputs)) return true;
    if (try tryNvfp4PairSplit(ctx, input, first_weight, second_weight, rows, input_dim, outputs)) return true;
    return tryGaffineU8PairSplit(ctx, input, first_weight, second_weight, rows, input_dim, outputs);
}

pub fn tryPairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    ctx.diagnostics.reset();
    if (activation == .none) return false;
    if (try tryDenseU16PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs)) return true;
    if (try tryMxfp8PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs)) return true;
    if (try tryFp8PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs)) return true;
    if (try tryNvfp4PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs)) return true;
    if (try tryGaffineU4PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs)) return true;
    return tryGaffineU8PairActivated(ctx, input, first_weight, second_weight, rows, input_dim, expected_output_dim, activation, outputs);
}

pub fn tryPairNvfp4Lt(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    ctx.diagnostics.reset();
    if (rows <= 32) return false;
    var blas_lt = ctx.blas_lt orelse return false;
    const first = switch (first_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (first.weight_global_scale == 0.0 or second.weight_global_scale == 0.0) return false;
    if (first.scales_lt_buffer.size == 0 or second.scales_lt_buffer.size == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    var input_fp4_buf: Buffer = undefined;
    var input_scales_buf: Buffer = undefined;
    if (!(try prepareNvfp4LtInput(ctx, input, input_dim, rows, &input_fp4_buf, &input_scales_buf))) return false;

    blas_lt.matmulNvfp4(
        ctx.device,
        &first.buffer,
        &first.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        outputs.first,
        rows,
        first.cols,
        first.rows,
        1.0 / first.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        ctx.device,
        &second.buffer,
        &second.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        outputs.second,
        rows,
        second.cols,
        second.rows,
        1.0 / second.weight_global_scale,
    ) catch return false;
    ctx.diagnostics.nvfp4_route = .pair_native_cublaslt;
    return true;
}

pub fn tryTripleKernel(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    ctx.diagnostics.reset();
    if (try tryDenseU16TripleKernel(ctx, input, first_weight, second_weight, third_weight, rows, input_dim, outputs)) return true;
    if (try tryGaffineU4TripleKernel(ctx, input, first_weight, second_weight, third_weight, rows, input_dim, outputs)) return true;
    if (try tryNvfp4TripleKernel(ctx, input, first_weight, second_weight, third_weight, rows, input_dim, outputs)) return true;
    return tryGaffineU8TripleKernel(ctx, input, first_weight, second_weight, third_weight, rows, input_dim, outputs);
}

pub fn tryTripleNvfp4Lt(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    ctx.diagnostics.reset();
    if (rows <= 32) return false;
    var blas_lt = ctx.blas_lt orelse return false;
    const first = switch (first_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const third = switch (third_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or third.rows != input_dim) return false;
    if (first.weight_global_scale == 0.0 or second.weight_global_scale == 0.0 or third.weight_global_scale == 0.0) return false;
    if (first.scales_lt_buffer.size == 0 or second.scales_lt_buffer.size == 0 or third.scales_lt_buffer.size == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateTripleOutputs(rows, first.cols, second.cols, third.cols, outputs);

    var input_fp4_buf: Buffer = undefined;
    var input_scales_buf: Buffer = undefined;
    if (!(try prepareNvfp4LtInput(ctx, input, input_dim, rows, &input_fp4_buf, &input_scales_buf))) return false;

    blas_lt.matmulNvfp4(
        ctx.device,
        &first.buffer,
        &first.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        outputs.first,
        rows,
        first.cols,
        first.rows,
        1.0 / first.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        ctx.device,
        &second.buffer,
        &second.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        outputs.second,
        rows,
        second.cols,
        second.rows,
        1.0 / second.weight_global_scale,
    ) catch return false;
    blas_lt.matmulNvfp4(
        ctx.device,
        &third.buffer,
        &third.scales_lt_buffer,
        &input_fp4_buf,
        &input_scales_buf,
        outputs.third,
        rows,
        third.cols,
        third.rows,
        1.0 / third.weight_global_scale,
    ) catch return false;
    ctx.diagnostics.nvfp4_route = .triple_native_cublaslt;
    return true;
}

pub fn tryTripleI8Concat(
    ctx: *FusedContext,
    input: *const Buffer,
    rows: usize,
    descriptor: *const ConcatI8TripleWeight,
    outputs: TripleOutputs,
) !bool {
    ctx.diagnostics.reset();
    descriptor.validate() catch return false;
    if (rows <= 1 or !ctx.capabilities.i8_blas_supported) return false;
    const quant_fn = ctx.quantize_f32_to_i8_simple_function orelse return false;
    const split_fn = ctx.dequant_i32_scales_split3_function orelse return false;

    const input_dim = descriptor.input_dim;
    const first_dim: usize = descriptor.output_dims[0];
    const second_dim: usize = descriptor.output_dims[1];
    const third_dim: usize = descriptor.output_dims[2];
    const total_output_dim = std.math.add(
        usize,
        first_dim,
        std.math.add(usize, second_dim, third_dim) catch return false,
    ) catch return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateTripleOutputs(rows, first_dim, second_dim, third_dim, outputs);

    const act_per_row = std.math.add(usize, input_dim, @sizeOf(f32)) catch return false;
    const i32_per_row = std.math.mul(usize, total_output_dim, @sizeOf(i32)) catch return false;
    const max_chunk = @min(
        ctx.workspace.activation_scratch.size / act_per_row,
        ctx.workspace.auxiliary_scratch.size / i32_per_row,
    );
    if (max_chunk == 0) return false;

    const input_row_bytes = std.math.mul(usize, input_dim, @sizeOf(f32)) catch return false;
    const first_row_bytes = std.math.mul(usize, first_dim, @sizeOf(f32)) catch return false;
    const second_row_bytes = std.math.mul(usize, second_dim, @sizeOf(f32)) catch return false;
    const third_row_bytes = std.math.mul(usize, third_dim, @sizeOf(f32)) catch return false;

    var done: usize = 0;
    while (done < rows) {
        const chunk = @min(rows - done, max_chunk);
        const chunk_u32 = std.math.cast(u32, chunk) orelse return false;
        const i8_input_bytes = std.math.mul(usize, chunk, input_dim) catch return false;
        const row_scales_bytes = std.math.mul(usize, chunk, @sizeOf(f32)) catch return false;
        const chunk_total_output = std.math.mul(usize, chunk, total_output_dim) catch return false;
        const i32_out_bytes = std.math.mul(usize, chunk_total_output, @sizeOf(i32)) catch return false;

        var i8_input_buf = weights.bufferSlice(&ctx.workspace.activation_scratch, 0, i8_input_bytes) catch return false;
        var row_scales_buf = weights.bufferSlice(&ctx.workspace.activation_scratch, i8_input_bytes, row_scales_bytes) catch return false;
        var i32_out_buf = weights.bufferSlice(&ctx.workspace.auxiliary_scratch, 0, i32_out_bytes) catch return false;

        const input_offset = std.math.mul(usize, done, input_row_bytes) catch return false;
        const first_offset = std.math.mul(usize, done, first_row_bytes) catch return false;
        const second_offset = std.math.mul(usize, done, second_row_bytes) catch return false;
        const third_offset = std.math.mul(usize, done, third_row_bytes) catch return false;
        const chunk_input_bytes = std.math.mul(usize, chunk, input_row_bytes) catch return false;
        const chunk_first_bytes = std.math.mul(usize, chunk, first_row_bytes) catch return false;
        const chunk_second_bytes = std.math.mul(usize, chunk, second_row_bytes) catch return false;
        const chunk_third_bytes = std.math.mul(usize, chunk, third_row_bytes) catch return false;
        var chunk_input = weights.bufferSlice(input, input_offset, chunk_input_bytes) catch return false;
        var chunk_first = weights.bufferSlice(outputs.first, first_offset, chunk_first_bytes) catch return false;
        var chunk_second = weights.bufferSlice(outputs.second, second_offset, chunk_second_bytes) catch return false;
        var chunk_third = weights.bufferSlice(outputs.third, third_offset, chunk_third_bytes) catch return false;

        ctx.arg_pack.reset();
        ctx.arg_pack.appendBufferPtr(&chunk_input) catch return false;
        ctx.arg_pack.appendBufferPtr(&i8_input_buf) catch return false;
        ctx.arg_pack.appendBufferPtr(&row_scales_buf) catch return false;
        ctx.arg_pack.appendScalar(u32, @intCast(input_dim)) catch return false;
        launch.launchWithFamily(ctx.device, quant_fn, .{
            .grid_x = chunk_u32,
            .block_x = 256,
        }, ctx.arg_pack, .other) catch return false;

        ctx.blas.matmulI8I8I32(
            ctx.device,
            &i8_input_buf,
            chunk,
            input_dim,
            &descriptor.i8_buffer,
            total_output_dim,
            &i32_out_buf,
        ) catch {
            ctx.capabilities.disableI8Blas();
            return false;
        };

        ctx.arg_pack.reset();
        ctx.arg_pack.appendBufferPtr(&i32_out_buf) catch return false;
        ctx.arg_pack.appendBufferPtr(&row_scales_buf) catch return false;
        ctx.arg_pack.appendBufferPtr(&descriptor.scales_buffer) catch return false;
        ctx.arg_pack.appendBufferPtr(&chunk_first) catch return false;
        ctx.arg_pack.appendBufferPtr(&chunk_second) catch return false;
        ctx.arg_pack.appendBufferPtr(&chunk_third) catch return false;
        ctx.arg_pack.appendScalar(u32, chunk_u32) catch return false;
        ctx.arg_pack.appendScalar(u32, descriptor.output_dims[0]) catch return false;
        ctx.arg_pack.appendScalar(u32, descriptor.output_dims[1]) catch return false;
        ctx.arg_pack.appendScalar(u32, descriptor.output_dims[2]) catch return false;
        const split_blocks_x_usize = std.math.divCeil(usize, total_output_dim, 256) catch return false;
        const split_blocks_x = std.math.cast(u32, split_blocks_x_usize) orelse return false;
        launch.launchWithFamily(ctx.device, split_fn, .{
            .grid_x = split_blocks_x,
            .grid_y = chunk_u32,
            .block_x = 256,
        }, ctx.arg_pack, .other) catch return false;

        done += chunk;
    }
    return true;
}

fn tryDenseU16PairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    const first = switch (first_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (!denseU16PairWeightsCompatible(input_dim, first, second)) return false;
    if (rows != 1) return false;
    if (!rowCountMatches(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    const kernel = switch (first.dtype) {
        .f16 => ctx.dense_u16_pair_f16_function orelse return false,
        .bf16 => ctx.dense_u16_pair_bf16_function orelse return false,
    };
    try matvec_u16_gate_up.runWithFunction(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.buffer,
        outputs.first,
        @intCast(first.cols),
        &second.buffer,
        outputs.second,
        @intCast(second.cols),
        @intCast(input_dim),
    );
    return true;
}

fn tryDenseU16PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (activation != .silu) return false;
    if (rows == 0 or rows > 32) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (!denseU16PairWeightsCompatible(input_dim, first, second)) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    try validateProductOutput(rows, first.cols, product);

    const kernel = switch (first.dtype) {
        .f16 => ctx.dense_u16_pair_silu_f16_function orelse return false,
        .bf16 => ctx.dense_u16_pair_silu_bf16_function orelse return false,
    };
    try matvec_u16_gate_up_silu.runWithFunctionGridBatch(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.buffer,
        &second.buffer,
        product,
        @intCast(first.cols),
        @intCast(input_dim),
        @intCast(rows),
    );
    return true;
}

fn tryMxfp8PairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    if (rows == 0 or rows > 4) return false;
    const first = switch (first_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (!u32Fits(rows) or !u32Fits(input_dim) or !u32Fits(first.cols) or !u32Fits(second.cols)) return false;
    if (first.scales_raw_buffer.pointer == 0 or second.scales_raw_buffer.pointer == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    var kernel = ctx.mxfp8_pair_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (ctx.mxfp8_pair_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const first_out_dim: u32 = @intCast(first.cols);
    const second_out_dim: u32 = @intCast(second.cols);
    const total_dim = std.math.add(u32, first_out_dim, second_out_dim) catch return false;
    const batch_rows: u32 = @intCast(rows);

    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_raw_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.first);
    try ctx.arg_pack.appendScalar(u32, first_out_dim);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_raw_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.second);
    try ctx.arg_pack.appendScalar(u32, second_out_dim);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec);
    return true;
}

fn tryMxfp8PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (activation != .silu) return false;
    if (rows == 0 or rows > 4) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .mxfp8 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    if (!u32Fits(rows) or !u32Fits(input_dim) or !u32Fits(first.cols)) return false;
    if (first.scales_raw_buffer.pointer == 0 or second.scales_raw_buffer.pointer == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateProductOutput(rows, first.cols, product);

    var kernel = ctx.mxfp8_pair_silu_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (ctx.mxfp8_pair_silu_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(first.cols);
    const batch_rows: u32 = @intCast(rows);
    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_raw_buffer);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_raw_buffer);
    try ctx.arg_pack.appendBufferPtr(product);
    try ctx.arg_pack.appendScalar(u32, out_dim);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec_gate_up_silu);
    return true;
}

fn tryFp8PairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    if (rows == 0 or rows > 32) return false;
    const first = switch (first_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (!u32Fits(rows) or !u32Fits(input_dim) or !u32Fits(first.cols) or !u32Fits(second.cols)) return false;
    if (first.scales_buffer.pointer == 0 or second.scales_buffer.pointer == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    var kernel = ctx.fp8_pair_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (ctx.fp8_pair_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const first_out_dim: u32 = @intCast(first.cols);
    const second_out_dim: u32 = @intCast(second.cols);
    const total_dim = std.math.add(u32, first_out_dim, second_out_dim) catch return false;
    const batch_rows: u32 = @intCast(rows);

    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.first);
    try ctx.arg_pack.appendScalar(u32, first_out_dim);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.second);
    try ctx.arg_pack.appendScalar(u32, second_out_dim);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, first.block_size);
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec);
    return true;
}

fn tryFp8PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (activation != .silu) return false;
    if (rows == 0 or rows > 32) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .fp8 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    if (!u32Fits(rows) or !u32Fits(input_dim) or !u32Fits(first.cols)) return false;
    if (first.scales_buffer.pointer == 0 or second.scales_buffer.pointer == 0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateProductOutput(rows, first.cols, product);

    var kernel = ctx.fp8_pair_silu_function orelse return false;
    var batch_tile: u32 = 4;
    if (rows > 4) {
        if (ctx.fp8_pair_silu_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const out_dim: u32 = @intCast(first.cols);
    const batch_rows: u32 = @intCast(rows);
    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(product);
    try ctx.arg_pack.appendScalar(u32, out_dim);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, first.block_size);
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec_gate_up_silu);
    return true;
}

fn tryNvfp4PairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    if (!ctx.capabilities.nvfp4_custom_supported) return false;
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !ctx.capabilities.nvfp4_pair_multi_row_supported) return false;
    const first = switch (first_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (!u32Fits(input_dim) or !u32Fits(first.cols) or !u32Fits(second.cols)) return false;
    if (first.weight_global_scale == 0.0 or second.weight_global_scale == 0.0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    var kernel = ctx.nvfp4_pair_function orelse return false;
    var batch_tile: u32 = 4;
    const pair_cols = std.math.add(usize, first.cols, second.cols) catch return false;
    const prefer_tile8 = rows > 4 or (rows == 1 and pair_cols >= 8192);
    if (prefer_tile8) {
        if (ctx.nvfp4_pair_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const first_out_dim: u32 = @intCast(first.cols);
    const second_out_dim: u32 = @intCast(second.cols);
    const total_dim = std.math.add(u32, first_out_dim, second_out_dim) catch return false;
    const batch_rows: u32 = @intCast(rows);

    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.first);
    try ctx.arg_pack.appendScalar(u32, first_out_dim);
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, first.group_size);
    try ctx.arg_pack.appendScalar(f32, first.weight_global_scale);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.second);
    try ctx.arg_pack.appendScalar(u32, second_out_dim);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.group_size);
    try ctx.arg_pack.appendScalar(f32, second.weight_global_scale);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (total_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec);
    ctx.diagnostics.nvfp4_route = .pair_custom_kernel;
    return true;
}

fn tryNvfp4PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (!ctx.capabilities.nvfp4_custom_supported) return false;
    if (activation == .none) return false;
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !ctx.capabilities.nvfp4_pair_multi_row_supported) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or first.rows != second.rows) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    if (!u32Fits(input_dim) or !u32Fits(first.cols)) return false;
    if (first.group_size == 0 or second.group_size == 0) return false;
    if ((first.rows % first.group_size) != 0 or (second.rows % second.group_size) != 0) return false;
    if (first.scales_buffer.pointer == 0 or second.scales_buffer.pointer == 0) return false;
    if (first.weight_global_scale == 0.0 or second.weight_global_scale == 0.0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateProductOutput(rows, first.cols, product);

    var kernel = switch (activation) {
        .silu => ctx.nvfp4_pair_silu_function orelse return false,
        .gelu => ctx.nvfp4_pair_gelu_function orelse return false,
        .none => unreachable,
    };
    var batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and first.cols >= 8192);
    if (prefer_tile8) {
        switch (activation) {
            .silu => if (ctx.nvfp4_pair_silu_tile8_function) |tile8_fn| {
                kernel = tile8_fn;
                batch_tile = 8;
            },
            .gelu => if (ctx.nvfp4_pair_gelu_tile8_function) |tile8_fn| {
                kernel = tile8_fn;
                batch_tile = 8;
            },
            .none => unreachable,
        }
    }

    const out_dim: u32 = @intCast(first.cols);
    const batch_rows: u32 = @intCast(rows);
    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(product);
    try ctx.arg_pack.appendScalar(u32, out_dim);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, first.group_size);
    try ctx.arg_pack.appendScalar(f32, first.weight_global_scale);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.group_size);
    try ctx.arg_pack.appendScalar(f32, second.weight_global_scale);
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (out_dim + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec_gate_up_silu);
    ctx.diagnostics.nvfp4_route = .pair_custom_kernel;
    return true;
}

fn tryGaffineU8PairSplit(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: PairOutputs,
) !bool {
    if (rows == 0 or !u32Fits(rows)) return false;
    const first = switch (first_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!gaffinePairWeightsCompatible(input_dim, first, second)) return false;
    if (!rowCountMatches(input, rows, input_dim)) return false;
    try validatePairSplitOutputs(rows, first.cols, second.cols, outputs);

    const kernel = ctx.gaffine_u8_pair_function orelse return false;
    try gaffine_u8_matvec_gate_up.runWithFunction(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.packed_data,
        &first.scales,
        &first.biases,
        outputs.first,
        @intCast(first.cols),
        first.group_size,
        first.scales_dtype_tag,
        &second.packed_data,
        &second.scales,
        &second.biases,
        outputs.second,
        @intCast(second.cols),
        second.group_size,
        second.scales_dtype_tag,
        @intCast(input_dim),
        @intCast(rows),
    );
    return true;
}

fn tryGaffineU4PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (activation != .silu) return false;
    if (rows == 0 or rows > 32) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    if (!gaffinePairWeightsCompatible(input_dim, first, second)) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    try validateProductOutput(rows, first.cols, product);

    var kernel = ctx.gaffine_u4_pair_silu_function orelse return false;
    var use_tile8 = false;
    if (ctx.capabilities.gaffine_u4_tile8_enabled and rows > 4) {
        if (ctx.gaffine_u4_pair_silu_tile8_function) |tile8_kernel| {
            kernel = tile8_kernel;
            use_tile8 = true;
        }
    }

    if (use_tile8) {
        try gaffine_u4_matvec_gate_up_silu.runWithFunctionTile8(
            ctx.arg_pack,
            ctx.device,
            kernel,
            input,
            &first.packed_data,
            &first.scales,
            &first.biases,
            &second.packed_data,
            &second.scales,
            &second.biases,
            product,
            @intCast(first.cols),
            first.group_size,
            first.scales_dtype_tag,
            second.group_size,
            second.scales_dtype_tag,
            @intCast(input_dim),
            @intCast(rows),
        );
    } else {
        try gaffine_u4_matvec_gate_up_silu.runWithFunction(
            ctx.arg_pack,
            ctx.device,
            kernel,
            input,
            &first.packed_data,
            &first.scales,
            &first.biases,
            &second.packed_data,
            &second.scales,
            &second.biases,
            product,
            @intCast(first.cols),
            first.group_size,
            first.scales_dtype_tag,
            second.group_size,
            second.scales_dtype_tag,
            @intCast(input_dim),
            @intCast(rows),
        );
    }
    return true;
}

fn tryGaffineU8PairActivated(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    expected_output_dim: usize,
    activation: PairActivation,
    outputs: PairOutputs,
) !bool {
    if (activation != .silu) return false;
    if (rows == 0 or rows > std.math.maxInt(u32)) return false;
    const product = outputs.activated_product orelse return false;
    const first = switch (first_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!gaffinePairWeightsCompatible(input_dim, first, second)) return false;
    if (first.cols != second.cols or first.cols != expected_output_dim) return false;
    if (!rowCountMatches(input, rows, input_dim)) return false;
    try validateProductOutput(rows, first.cols, product);

    const kernel = ctx.gaffine_u8_pair_silu_function orelse return false;
    try gaffine_u8_matvec_gate_up_silu.runWithFunction(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.packed_data,
        &first.scales,
        &first.biases,
        &second.packed_data,
        &second.scales,
        &second.biases,
        product,
        @intCast(first.cols),
        first.group_size,
        first.scales_dtype_tag,
        second.group_size,
        second.scales_dtype_tag,
        @intCast(input_dim),
        @intCast(rows),
    );
    return true;
}

fn tryDenseU16TripleKernel(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    if (rows == 0 or !u32Fits(rows)) return false;
    const first = switch (first_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    const third = switch (third_weight.*) {
        .dense_u16 => |w| w,
        else => return false,
    };
    if (!denseU16TripleWeightsCompatible(input_dim, first, second, third)) return false;
    try validateTripleOutputs(rows, first.cols, second.cols, third.cols, outputs);

    const kernel = switch (first.dtype) {
        .f16 => ctx.dense_u16_triple_f16_function orelse return false,
        .bf16 => ctx.dense_u16_triple_bf16_function orelse return false,
    };
    try matvec_u16_qkv.runWithFunctionGridBatch(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.buffer,
        outputs.first,
        @intCast(first.cols),
        &second.buffer,
        outputs.second,
        @intCast(second.cols),
        &third.buffer,
        outputs.third,
        @intCast(third.cols),
        @intCast(input_dim),
        @intCast(rows),
    );
    return true;
}

fn tryGaffineU4TripleKernel(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    if (rows == 0 or !u32Fits(rows)) return false;
    var kernel = ctx.gaffine_u4_triple_function orelse return false;
    var use_tile8 = false;
    if (ctx.capabilities.gaffine_u4_tile8_enabled and rows > 4) {
        if (ctx.gaffine_u4_triple_tile8_function) |tile8_kernel| {
            kernel = tile8_kernel;
            use_tile8 = true;
        }
    }
    const first = switch (first_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    const third = switch (third_weight.*) {
        .gaffine_u4 => |w| w,
        else => return false,
    };
    if (!gaffineTripleWeightsCompatible(input_dim, first, second, third)) return false;
    try validateTripleOutputs(rows, first.cols, second.cols, third.cols, outputs);

    if (use_tile8) {
        try gaffine_u4_matvec_qkv.runWithFunctionTile8(
            ctx.arg_pack,
            ctx.device,
            kernel,
            input,
            &first.packed_data,
            &first.scales,
            &first.biases,
            outputs.first,
            @intCast(first.cols),
            first.group_size,
            first.scales_dtype_tag,
            &second.packed_data,
            &second.scales,
            &second.biases,
            outputs.second,
            @intCast(second.cols),
            second.group_size,
            second.scales_dtype_tag,
            &third.packed_data,
            &third.scales,
            &third.biases,
            outputs.third,
            @intCast(third.cols),
            third.group_size,
            third.scales_dtype_tag,
            @intCast(input_dim),
            @intCast(rows),
        );
    } else {
        try gaffine_u4_matvec_qkv.runWithFunction(
            ctx.arg_pack,
            ctx.device,
            kernel,
            input,
            &first.packed_data,
            &first.scales,
            &first.biases,
            outputs.first,
            @intCast(first.cols),
            first.group_size,
            first.scales_dtype_tag,
            &second.packed_data,
            &second.scales,
            &second.biases,
            outputs.second,
            @intCast(second.cols),
            second.group_size,
            second.scales_dtype_tag,
            &third.packed_data,
            &third.scales,
            &third.biases,
            outputs.third,
            @intCast(third.cols),
            third.group_size,
            third.scales_dtype_tag,
            @intCast(input_dim),
            @intCast(rows),
        );
    }
    return true;
}

fn tryNvfp4TripleKernel(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    if (!ctx.capabilities.nvfp4_custom_supported) return false;
    if (rows == 0 or rows > 32) return false;
    if (rows > 1 and !ctx.capabilities.nvfp4_triple_multi_row_supported) return false;
    const first = switch (first_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    const third = switch (third_weight.*) {
        .nvfp4 => |w| w,
        else => return false,
    };
    if (first.rows != input_dim or second.rows != input_dim or third.rows != input_dim) return false;
    if (!u32Fits(input_dim) or !u32Fits(first.cols) or !u32Fits(second.cols) or !u32Fits(third.cols)) return false;
    if (first.weight_global_scale == 0.0 or second.weight_global_scale == 0.0 or third.weight_global_scale == 0.0) return false;
    if (!inputCoversRows(input, rows, input_dim)) return false;
    try validateTripleOutputs(rows, first.cols, second.cols, third.cols, outputs);

    const first_out_dim: u32 = @intCast(first.cols);
    const second_out_dim: u32 = @intCast(second.cols);
    const third_out_dim: u32 = @intCast(third.cols);
    const first_second = std.math.add(u32, first_out_dim, second_out_dim) catch return false;
    const total_out = std.math.add(u32, first_second, third_out_dim) catch return false;

    var kernel = ctx.nvfp4_triple_function orelse return false;
    var batch_tile: u32 = 4;
    const prefer_tile8 = rows > 4 or (rows == 1 and total_out >= 3072);
    if (prefer_tile8) {
        if (ctx.nvfp4_triple_tile8_function) |tile8_fn| {
            kernel = tile8_fn;
            batch_tile = 8;
        }
    }

    const batch_rows: u32 = @intCast(rows);
    ctx.arg_pack.reset();
    try ctx.arg_pack.appendBufferPtr(input);
    try ctx.arg_pack.appendBufferPtr(&first.buffer);
    try ctx.arg_pack.appendBufferPtr(&first.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.first);
    try ctx.arg_pack.appendScalar(u32, first_out_dim);
    try ctx.arg_pack.appendScalar(u32, first.scale_cols);
    try ctx.arg_pack.appendScalar(u32, first.group_size);
    try ctx.arg_pack.appendScalar(f32, first.weight_global_scale);
    try ctx.arg_pack.appendBufferPtr(&second.buffer);
    try ctx.arg_pack.appendBufferPtr(&second.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.second);
    try ctx.arg_pack.appendScalar(u32, second_out_dim);
    try ctx.arg_pack.appendScalar(u32, second.scale_cols);
    try ctx.arg_pack.appendScalar(u32, second.group_size);
    try ctx.arg_pack.appendScalar(f32, second.weight_global_scale);
    try ctx.arg_pack.appendBufferPtr(&third.buffer);
    try ctx.arg_pack.appendBufferPtr(&third.scales_buffer);
    try ctx.arg_pack.appendBufferPtr(outputs.third);
    try ctx.arg_pack.appendScalar(u32, third_out_dim);
    try ctx.arg_pack.appendScalar(u32, third.scale_cols);
    try ctx.arg_pack.appendScalar(u32, third.group_size);
    try ctx.arg_pack.appendScalar(f32, third.weight_global_scale);
    try ctx.arg_pack.appendScalar(u32, @intCast(input_dim));
    try ctx.arg_pack.appendScalar(u32, batch_rows);

    try launch.launchWithFamily(ctx.device, kernel, .{
        .grid_x = (total_out + 3) / 4,
        .grid_y = (batch_rows + batch_tile - 1) / batch_tile,
        .block_x = 128,
    }, ctx.arg_pack, .matvec_qkv);
    ctx.diagnostics.nvfp4_route = .triple_custom_kernel;
    return true;
}

fn tryGaffineU8TripleKernel(
    ctx: *FusedContext,
    input: *const Buffer,
    first_weight: *const LinearWeight,
    second_weight: *const LinearWeight,
    third_weight: *const LinearWeight,
    rows: usize,
    input_dim: usize,
    outputs: TripleOutputs,
) !bool {
    if (rows == 0 or !u32Fits(rows)) return false;
    const kernel = ctx.gaffine_u8_triple_function orelse return false;
    const first = switch (first_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const second = switch (second_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    const third = switch (third_weight.*) {
        .gaffine_u8 => |w| w,
        else => return false,
    };
    if (!gaffineTripleWeightsCompatible(input_dim, first, second, third)) return false;
    try validateTripleOutputs(rows, first.cols, second.cols, third.cols, outputs);

    try gaffine_u8_matvec_qkv.runWithFunction(
        ctx.arg_pack,
        ctx.device,
        kernel,
        input,
        &first.packed_data,
        &first.scales,
        &first.biases,
        outputs.first,
        @intCast(first.cols),
        first.group_size,
        first.scales_dtype_tag,
        &second.packed_data,
        &second.scales,
        &second.biases,
        outputs.second,
        @intCast(second.cols),
        second.group_size,
        second.scales_dtype_tag,
        &third.packed_data,
        &third.scales,
        &third.biases,
        outputs.third,
        @intCast(third.cols),
        third.group_size,
        third.scales_dtype_tag,
        @intCast(input_dim),
        @intCast(rows),
    );
    return true;
}

fn prepareNvfp4LtInput(
    ctx: *FusedContext,
    input: *const Buffer,
    input_dim: usize,
    rows: usize,
    input_fp4_out: *Buffer,
    input_scales_out: *Buffer,
) !bool {
    const quant_fn = ctx.quantize_f32_to_nvfp4_function orelse return false;
    if (!u32Fits(input_dim) or !u32Fits(rows)) return false;
    const packed_in_cols = std.math.divCeil(usize, input_dim, 2) catch return false;
    const input_fp4_bytes = std.math.mul(usize, rows, packed_in_cols) catch return false;
    const input_scale_bytes = Nvfp4LinearWeight.cublasLtScaleTensorSize(input_dim, rows);
    if (ctx.workspace.activation_scratch.size < input_fp4_bytes) return false;
    if (ctx.workspace.auxiliary_scratch.size < input_scale_bytes) return false;

    input_fp4_out.* = weights.bufferSlice(&ctx.workspace.activation_scratch, 0, input_fp4_bytes) catch return false;
    input_scales_out.* = weights.bufferSlice(&ctx.workspace.auxiliary_scratch, 0, input_scale_bytes) catch return false;

    const padded_outer: u32 = @intCast(Nvfp4LinearWeight.roundoff(rows, 128));
    const sf_k = std.math.divCeil(usize, input_dim, 16) catch return false;
    const padded_sf_k: u32 = @intCast(Nvfp4LinearWeight.roundoff(sf_k, 4));
    const quant_grid_x = std.math.cast(u32, sf_k) orelse return false;

    ctx.arg_pack.reset();
    ctx.arg_pack.appendBufferPtr(input) catch return false;
    ctx.arg_pack.appendBufferPtr(input_fp4_out) catch return false;
    ctx.arg_pack.appendBufferPtr(input_scales_out) catch return false;
    ctx.arg_pack.appendScalar(u32, @intCast(input_dim)) catch return false;
    ctx.arg_pack.appendScalar(u32, @intCast(rows)) catch return false;
    ctx.arg_pack.appendScalar(u32, padded_outer) catch return false;
    ctx.arg_pack.appendScalar(u32, padded_sf_k) catch return false;
    launch.launchWithFamily(ctx.device, quant_fn, .{
        .grid_x = quant_grid_x,
        .grid_y = padded_outer,
        .block_x = 32,
    }, ctx.arg_pack, .other) catch return false;

    return true;
}

fn validateConcatI8TripleWeight(descriptor: *const ConcatI8TripleWeight) !void {
    if (descriptor.input_dim == 0) return error.InvalidArgument;
    if (!u32Fits(descriptor.input_dim)) return error.InvalidArgument;
    const first_dim: usize = descriptor.output_dims[0];
    const second_dim: usize = descriptor.output_dims[1];
    const third_dim: usize = descriptor.output_dims[2];
    if (first_dim == 0 or second_dim == 0 or third_dim == 0) return error.InvalidArgument;
    const total_output_dim = std.math.add(
        usize,
        first_dim,
        std.math.add(usize, second_dim, third_dim) catch return error.InvalidArgument,
    ) catch return error.InvalidArgument;
    if (!u32Fits(total_output_dim)) return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, descriptor.input_dim, total_output_dim) catch return error.InvalidArgument;
    const scale_bytes = std.math.mul(usize, total_output_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    if (descriptor.i8_buffer.size < weight_bytes or descriptor.scales_buffer.size < scale_bytes) return error.InvalidArgument;
}

fn validatePairSplitOutputs(rows: usize, first_dim: usize, second_dim: usize, outputs: PairOutputs) !void {
    if (!bufferCoversF32Rows(outputs.first, rows, first_dim)) return error.InvalidInstructionBinding;
    if (!bufferCoversF32Rows(outputs.second, rows, second_dim)) return error.InvalidInstructionBinding;
}

fn validateProductOutput(rows: usize, output_dim: usize, product: *Buffer) !void {
    if (!bufferCoversF32Rows(product, rows, output_dim)) return error.InvalidInstructionBinding;
}

fn validateTripleOutputs(rows: usize, first_dim: usize, second_dim: usize, third_dim: usize, outputs: TripleOutputs) !void {
    if (!bufferCoversF32Rows(outputs.first, rows, first_dim)) return error.InvalidInstructionBinding;
    if (!bufferCoversF32Rows(outputs.second, rows, second_dim)) return error.InvalidInstructionBinding;
    if (!bufferCoversF32Rows(outputs.third, rows, third_dim)) return error.InvalidInstructionBinding;
}

fn bufferCoversF32Rows(buffer: *const Buffer, rows: usize, width: usize) bool {
    if (rows == 0 or width == 0) return false;
    const row_bytes = std.math.mul(usize, width, @sizeOf(f32)) catch return false;
    const required = std.math.mul(usize, rows, row_bytes) catch return false;
    return buffer.size >= required;
}

fn inputCoversRows(input: *const Buffer, rows: usize, input_dim: usize) bool {
    return bufferCoversF32Rows(input, rows, input_dim);
}

fn rowCountMatches(input: *const Buffer, rows: usize, input_dim: usize) bool {
    const actual = weights.bufferF32RowCount(input, input_dim) catch return false;
    return actual == rows;
}

fn u32Fits(value: usize) bool {
    return value <= std.math.maxInt(u32);
}

fn validationOnlyContext(
    device: *device_mod.Device,
    blas: *matmul.Blas,
    diagnostics: *Diagnostics,
    capabilities: *CapabilityFlags,
    pack: *args_mod.ArgPack,
) FusedContext {
    return .{
        .device = device,
        .arg_pack = pack,
        .blas = blas,
        .blas_lt = null,
        .workspace = .{
            .activation_scratch = .{ .pointer = 0, .size = 0 },
            .auxiliary_scratch = .{ .pointer = 0, .size = 0 },
        },
        .capabilities = capabilities,
        .diagnostics = diagnostics,
    };
}

fn dummyBuffer(bytes: usize) Buffer {
    return .{ .pointer = 0x1000, .size = bytes };
}

fn denseWeight(rows: usize, cols: usize, dtype_tag: weights.DenseU16Dtype) U16LinearWeight {
    return .{
        .rows = rows,
        .cols = cols,
        .buffer = dummyBuffer(rows * cols * @sizeOf(u16)),
        .dtype = dtype_tag,
    };
}

fn gaffineWeight(rows: usize, cols: usize, scales_dtype_tag: u32) GaffineU8LinearWeight {
    return .{
        .rows = rows,
        .cols = cols,
        .packed_data = dummyBuffer(4096),
        .scales = dummyBuffer(4096),
        .biases = dummyBuffer(4096),
        .group_size = 16,
        .scales_dtype_tag = scales_dtype_tag,
    };
}

fn nvfp4Weight(rows: usize, cols: usize) LinearWeight {
    return .{ .nvfp4 = .{
        .rows = rows,
        .cols = cols,
        .buffer = dummyBuffer(4096),
        .scales_buffer = dummyBuffer(4096),
        .scales_lt_buffer = dummyBuffer(4096),
        .packed_cols = @intCast((rows + 1) / 2),
        .scale_cols = @intCast((rows + 15) / 16),
        .group_size = 16,
        .weight_global_scale = 1.0,
    } };
}

test "compute cuda linear fused denseU16PairWeightsCompatible accepts matching dtype and input rows" {
    const first = denseWeight(8, 4, .bf16);
    const second = denseWeight(8, 6, .bf16);
    try std.testing.expect(denseU16PairWeightsCompatible(8, first, second));

    const mixed = denseWeight(8, 6, .f16);
    try std.testing.expect(!denseU16PairWeightsCompatible(8, first, mixed));
    try std.testing.expect(!denseU16PairWeightsCompatible(7, first, second));
}

test "compute cuda linear fused denseU16TripleWeightsCompatible accepts asymmetric outputs" {
    const first = denseWeight(8, 4, .f16);
    const second = denseWeight(8, 2, .f16);
    const third = denseWeight(8, 3, .f16);
    try std.testing.expect(denseU16TripleWeightsCompatible(8, first, second, third));

    const mixed = denseWeight(8, 3, .bf16);
    try std.testing.expect(!denseU16TripleWeightsCompatible(8, first, second, mixed));
}

test "compute cuda linear fused gaffinePairWeightsCompatible rejects mismatched scale metadata" {
    const first = gaffineWeight(16, 8, 1);
    var second = gaffineWeight(16, 8, 1);
    try std.testing.expect(gaffinePairWeightsCompatible(16, first, second));

    second.scales_dtype_tag = 2;
    try std.testing.expect(!gaffinePairWeightsCompatible(16, first, second));
}

test "compute cuda linear fused gaffineTripleWeightsCompatible rejects input-row drift" {
    const first = gaffineWeight(16, 8, 1);
    const second = gaffineWeight(16, 4, 1);
    var third = gaffineWeight(16, 4, 1);
    try std.testing.expect(gaffineTripleWeightsCompatible(16, first, second, third));

    third.rows = 8;
    try std.testing.expect(!gaffineTripleWeightsCompatible(16, first, second, third));
}

test "compute cuda linear fused PairActivation selection distinguishes none silu and gelu" {
    try std.testing.expectEqual(PairActivation.none, PairActivation.none);
    try std.testing.expect(PairActivation.silu != PairActivation.gelu);
}

test "compute cuda linear fused PairOutputs validation rejects undersized product output" {
    var first = dummyBuffer(2 * 4 * @sizeOf(f32));
    var second = dummyBuffer(2 * 4 * @sizeOf(f32));
    var product = dummyBuffer(2 * 4 * @sizeOf(f32) - 1);
    const outputs = PairOutputs{ .first = &first, .second = &second, .activated_product = &product };

    try validatePairSplitOutputs(2, 4, 4, outputs);
    try std.testing.expectError(error.InvalidInstructionBinding, validateProductOutput(2, 4, &product));
}

test "compute cuda linear fused TripleOutputs validation rejects undersized third output" {
    var first = dummyBuffer(2 * 4 * @sizeOf(f32));
    var second = dummyBuffer(2 * 2 * @sizeOf(f32));
    var third = dummyBuffer(2 * 3 * @sizeOf(f32) - 1);
    const outputs = TripleOutputs{ .first = &first, .second = &second, .third = &third };

    try std.testing.expectError(error.InvalidInstructionBinding, validateTripleOutputs(2, 4, 2, 3, outputs));
}

test "compute cuda linear fused input coverage validates complete f32 rows" {
    const input = dummyBuffer(2 * 8 * @sizeOf(f32));
    const short = dummyBuffer(2 * 8 * @sizeOf(f32) - 1);

    try std.testing.expect(inputCoversRows(&input, 2, 8));
    try std.testing.expect(!inputCoversRows(&short, 2, 8));
    try std.testing.expect(!inputCoversRows(&input, 0, 8));
    try std.testing.expect(!inputCoversRows(&input, 2, 0));
}

test "compute cuda linear fused ConcatI8TripleWeight.validate accepts complete descriptor" {
    const descriptor = ConcatI8TripleWeight{
        .input_dim = 8,
        .output_dims = .{ 4, 2, 2 },
        .i8_buffer = dummyBuffer(8 * 8),
        .scales_buffer = dummyBuffer(8 * @sizeOf(f32)),
    };
    try descriptor.validate();
}

test "compute cuda linear fused ConcatI8TripleWeight.validate rejects zero dimension" {
    const descriptor = ConcatI8TripleWeight{
        .input_dim = 0,
        .output_dims = .{ 4, 2, 2 },
        .i8_buffer = dummyBuffer(8 * 8),
        .scales_buffer = dummyBuffer(8 * @sizeOf(f32)),
    };
    try std.testing.expectError(error.InvalidArgument, descriptor.validate());
}

test "compute cuda linear fused ConcatI8TripleWeight.validate rejects undersized buffers" {
    const descriptor = ConcatI8TripleWeight{
        .input_dim = 8,
        .output_dims = .{ 4, 2, 2 },
        .i8_buffer = dummyBuffer(8 * 8 - 1),
        .scales_buffer = dummyBuffer(8 * @sizeOf(f32)),
    };
    try std.testing.expectError(error.InvalidArgument, descriptor.validate());
}

test "compute cuda linear fused Diagnostics.reset clears fused route" {
    var diagnostics = Diagnostics{ .nvfp4_route = .triple_custom_kernel };
    diagnostics.reset();
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}

test "compute cuda linear fused CapabilityFlags.disableI8Blas clears support flag" {
    var capabilities = CapabilityFlags{ .i8_blas_supported = true };
    capabilities.disableI8Blas();
    try std.testing.expect(!capabilities.i8_blas_supported);
}

test "compute cuda linear fused tryPairSplit resets diagnostics on validation miss" {
    var diagnostics = Diagnostics{ .nvfp4_route = .pair_custom_kernel };
    var capabilities = CapabilityFlags{};
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const first = LinearWeight{ .dense_u16 = denseWeight(8, 4, .f16) };
    const second = LinearWeight{ .dense_u16 = denseWeight(8, 4, .f16) };
    const input = dummyBuffer(8 * @sizeOf(f32));
    var out_first = dummyBuffer(4 * @sizeOf(f32));
    var out_second = dummyBuffer(4 * @sizeOf(f32));
    const outputs = PairOutputs{ .first = &out_first, .second = &out_second };

    try std.testing.expect(!try tryPairSplit(&ctx, &input, &first, &second, 0, 8, outputs));
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}

test "compute cuda linear fused tryPairActivated returns false for none activation" {
    var diagnostics = Diagnostics{};
    var capabilities = CapabilityFlags{};
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const first = LinearWeight{ .dense_u16 = denseWeight(8, 4, .f16) };
    const second = LinearWeight{ .dense_u16 = denseWeight(8, 4, .f16) };
    const input = dummyBuffer(8 * @sizeOf(f32));
    var out_first = dummyBuffer(4 * @sizeOf(f32));
    var out_second = dummyBuffer(4 * @sizeOf(f32));
    var product = dummyBuffer(4 * @sizeOf(f32));
    const outputs = PairOutputs{ .first = &out_first, .second = &out_second, .activated_product = &product };

    try std.testing.expect(!try tryPairActivated(&ctx, &input, &first, &second, 1, 8, 4, .none, outputs));
}

test "compute cuda linear fused tryPairNvfp4Lt returns false without cuBLASLt" {
    var diagnostics = Diagnostics{ .nvfp4_route = .pair_custom_kernel };
    var capabilities = CapabilityFlags{};
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const first = nvfp4Weight(8, 4);
    const second = nvfp4Weight(8, 4);
    const input = dummyBuffer(64 * 8 * @sizeOf(f32));
    var out_first = dummyBuffer(64 * 4 * @sizeOf(f32));
    var out_second = dummyBuffer(64 * 4 * @sizeOf(f32));
    const outputs = PairOutputs{ .first = &out_first, .second = &out_second };

    try std.testing.expect(!try tryPairNvfp4Lt(&ctx, &input, &first, &second, 64, 8, outputs));
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}

test "compute cuda linear fused tryTripleKernel resets diagnostics on zero rows" {
    var diagnostics = Diagnostics{ .nvfp4_route = .triple_custom_kernel };
    var capabilities = CapabilityFlags{};
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const first = LinearWeight{ .dense_u16 = denseWeight(8, 4, .f16) };
    const second = LinearWeight{ .dense_u16 = denseWeight(8, 2, .f16) };
    const third = LinearWeight{ .dense_u16 = denseWeight(8, 2, .f16) };
    const input = dummyBuffer(8 * @sizeOf(f32));
    var out_first = dummyBuffer(4 * @sizeOf(f32));
    var out_second = dummyBuffer(2 * @sizeOf(f32));
    var out_third = dummyBuffer(2 * @sizeOf(f32));
    const outputs = TripleOutputs{ .first = &out_first, .second = &out_second, .third = &out_third };

    try std.testing.expect(!try tryTripleKernel(&ctx, &input, &first, &second, &third, 0, 8, outputs));
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}

test "compute cuda linear fused tryTripleNvfp4Lt returns false without cuBLASLt" {
    var diagnostics = Diagnostics{ .nvfp4_route = .triple_custom_kernel };
    var capabilities = CapabilityFlags{};
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const first = nvfp4Weight(8, 4);
    const second = nvfp4Weight(8, 2);
    const third = nvfp4Weight(8, 2);
    const input = dummyBuffer(64 * 8 * @sizeOf(f32));
    var out_first = dummyBuffer(64 * 4 * @sizeOf(f32));
    var out_second = dummyBuffer(64 * 2 * @sizeOf(f32));
    var out_third = dummyBuffer(64 * 2 * @sizeOf(f32));
    const outputs = TripleOutputs{ .first = &out_first, .second = &out_second, .third = &out_third };

    try std.testing.expect(!try tryTripleNvfp4Lt(&ctx, &input, &first, &second, &third, 64, 8, outputs));
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}

test "compute cuda linear fused tryTripleI8Concat returns false when capability is disabled" {
    var diagnostics = Diagnostics{ .nvfp4_route = .triple_custom_kernel };
    var capabilities = CapabilityFlags{ .i8_blas_supported = false };
    var pack = args_mod.ArgPack.init(std.testing.allocator);
    defer pack.deinit();
    var device: device_mod.Device = undefined;
    var blas: matmul.Blas = undefined;
    var ctx = validationOnlyContext(&device, &blas, &diagnostics, &capabilities, &pack);

    const descriptor = ConcatI8TripleWeight{
        .input_dim = 8,
        .output_dims = .{ 4, 2, 2 },
        .i8_buffer = dummyBuffer(8 * 8),
        .scales_buffer = dummyBuffer(8 * @sizeOf(f32)),
    };
    const input = dummyBuffer(2 * 8 * @sizeOf(f32));
    var out_first = dummyBuffer(2 * 4 * @sizeOf(f32));
    var out_second = dummyBuffer(2 * 2 * @sizeOf(f32));
    var out_third = dummyBuffer(2 * 2 * @sizeOf(f32));
    const outputs = TripleOutputs{ .first = &out_first, .second = &out_second, .third = &out_third };

    try std.testing.expect(!try tryTripleI8Concat(&ctx, &input, 2, &descriptor, outputs));
    try std.testing.expectEqual(@as(?Nvfp4RouteKind, null), diagnostics.nvfp4_route);
}
