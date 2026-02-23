//! Stateful shortconv step kernel wrapper.
//!
//! Computes one autoregressive shortconv step:
//! - `state` shift/update
//! - depthwise conv over `d_conv` taps
//! - C-gating into output

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_shortconv_step_f32";
pub const op_name: []const u8 = "shortconv_step_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    b_gate: *const device_mod.Buffer,
    c_gate: *const device_mod.Buffer,
    x_proj: *const device_mod.Buffer,
    state: *device_mod.Buffer,
    weight_time_major: *const device_mod.Buffer,
    bias: ?*const device_mod.Buffer,
    out: *device_mod.Buffer,
    conv_dim: u32,
    d_conv: u32,
) !void {
    try validateArgs(b_gate, c_gate, x_proj, state, weight_time_major, bias, out, conv_dim, d_conv);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(state);
    try arg_pack.appendBufferPtr(b_gate);
    try arg_pack.appendBufferPtr(c_gate);
    try arg_pack.appendBufferPtr(x_proj);
    try arg_pack.appendBufferPtr(weight_time_major);
    try arg_pack.appendDevicePtr(if (bias) |buf| buf.pointer else 0);
    try arg_pack.appendScalar(u32, conv_dim);
    try arg_pack.appendScalar(u32, d_conv);
    try arg_pack.appendScalar(u32, if (bias != null) 1 else 0);

    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(conv_dim, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    b_gate: *const device_mod.Buffer,
    c_gate: *const device_mod.Buffer,
    x_proj: *const device_mod.Buffer,
    state: *const device_mod.Buffer,
    weight_time_major: *const device_mod.Buffer,
    bias: ?*const device_mod.Buffer,
    out: *const device_mod.Buffer,
    conv_dim: u32,
    d_conv: u32,
) !void {
    if (conv_dim == 0 or d_conv == 0) return error.InvalidArgument;
    const vec_bytes = std.math.mul(usize, @as(usize, conv_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    if (b_gate.size < vec_bytes or c_gate.size < vec_bytes or x_proj.size < vec_bytes or out.size < vec_bytes) {
        return error.InvalidArgument;
    }

    const taps = std.math.mul(usize, @as(usize, conv_dim), @as(usize, d_conv)) catch return error.InvalidArgument;
    const taps_bytes = std.math.mul(usize, taps, @sizeOf(f32)) catch return error.InvalidArgument;
    if (state.size < taps_bytes or weight_time_major.size < taps_bytes) return error.InvalidArgument;
    if (bias) |bias_buf| {
        if (bias_buf.size < vec_bytes) return error.InvalidArgument;
    }
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero dimensions" {
    const vec = device_mod.Buffer{ .pointer = 0, .size = 16 };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&vec, &vec, &vec, &vec, &vec, null, &vec, 0, 3),
    );
}

test "validateArgs rejects undersized state buffer" {
    const vec = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    const small_state = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&vec, &vec, &vec, &small_state, &vec, null, &vec, 8, 2),
    );
}

test "validateArgs rejects undersized bias when provided" {
    const vec = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    const taps = device_mod.Buffer{ .pointer = 0, .size = 16 * @sizeOf(f32) };
    const small_bias = device_mod.Buffer{ .pointer = 0, .size = @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&vec, &vec, &vec, &taps, &taps, &small_bias, &vec, 8, 2),
    );
}
