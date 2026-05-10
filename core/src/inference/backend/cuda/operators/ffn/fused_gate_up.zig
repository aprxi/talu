//! Fused gate-up helpers for the CUDA inference backend.

const builtin = @import("builtin");
const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/_types_impl.zig");
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
    if (shouldAvoidWindowsPreSm89Nvfp4Fused(self)) return false;
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
    recordNvfp4Route(self, .fused_gate_up_custom);
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
    if (shouldAvoidWindowsPreSm89Nvfp4Fused(self)) return false;
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
    recordNvfp4Route(self, .fused_gate_up_custom);
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
    if (shouldAvoidWindowsPreSm89Nvfp4Fused(self)) return false;
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
    recordNvfp4Route(self, .fused_gate_up_custom);
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
    recordNvfp4Route(self, .fused_gate_up_native_cublaslt);
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
