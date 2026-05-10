//! Fused QKV helpers for the CUDA inference backend.

const builtin = @import("builtin");
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

fn shouldAvoidWindowsPreSm89Nvfp4Fused(self: anytype) bool {
    if (builtin.os.tag != .windows) return false;
    const capability = self.device.computeCapability() catch return true;
    return capability.major < 8 or (capability.major == 8 and capability.minor < 9);
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
    if (shouldAvoidWindowsPreSm89Nvfp4Fused(self)) return false;
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
    recordNvfp4Route(self, .fused_qkv_custom);
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
    recordNvfp4Route(self, .fused_qkv_native_cublaslt);
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
