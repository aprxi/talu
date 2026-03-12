//! Gated-delta state-space step kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol: [:0]const u8 = "talu_gated_delta_ssm_f32";
pub const op_name: []const u8 = "gated_delta_ssm_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    qkv: *const device_mod.Buffer,
    beta_raw: *const device_mod.Buffer,
    a_raw: *const device_mod.Buffer,
    a_log: *const device_mod.Buffer,
    dt_bias: ?*const device_mod.Buffer,
    state: *device_mod.Buffer,
    out: *device_mod.Buffer,
    n_qk_heads: u32,
    n_v_heads: u32,
    d_head: u32,
) !void {
    try validateArgs(qkv, beta_raw, a_raw, a_log, dt_bias, state, out, n_qk_heads, n_v_heads, d_head);

    arg_pack.reset();
    try arg_pack.appendBufferPtr(out);
    try arg_pack.appendBufferPtr(state);
    try arg_pack.appendBufferPtr(qkv);
    try arg_pack.appendBufferPtr(beta_raw);
    try arg_pack.appendBufferPtr(a_raw);
    try arg_pack.appendBufferPtr(a_log);
    try arg_pack.appendDevicePtr(if (dt_bias) |buf| buf.pointer else 0);
    try arg_pack.appendScalar(u32, n_qk_heads);
    try arg_pack.appendScalar(u32, n_v_heads);
    try arg_pack.appendScalar(u32, d_head);
    try arg_pack.appendScalar(u32, if (dt_bias != null) 1 else 0);

    // Match thread count to head width to avoid idle warps on d_head=64/128.
    const block_x: u32 = @min(d_head, 256);
    const shared_bytes = std.math.mul(usize, 2 * @as(usize, d_head) + @as(usize, block_x), @sizeOf(f32)) catch return error.InvalidArgument;
    try launch_mod.launch(device, function, .{
        .grid_x = n_v_heads,
        .block_x = block_x,
        .shared_mem_bytes = @intCast(shared_bytes),
    }, arg_pack);
}

fn validateArgs(
    qkv: *const device_mod.Buffer,
    beta_raw: *const device_mod.Buffer,
    a_raw: *const device_mod.Buffer,
    a_log: *const device_mod.Buffer,
    dt_bias: ?*const device_mod.Buffer,
    state: *const device_mod.Buffer,
    out: *const device_mod.Buffer,
    n_qk_heads: u32,
    n_v_heads: u32,
    d_head: u32,
) !void {
    if (n_qk_heads == 0 or n_v_heads == 0 or d_head == 0) return error.InvalidArgument;
    if ((n_v_heads % n_qk_heads) != 0) return error.InvalidArgument;

    const qk_inner = std.math.mul(usize, @as(usize, n_qk_heads), @as(usize, d_head)) catch return error.InvalidArgument;
    const d_inner = std.math.mul(usize, @as(usize, n_v_heads), @as(usize, d_head)) catch return error.InvalidArgument;
    const qkv_len = std.math.add(usize, std.math.mul(usize, 2, qk_inner) catch return error.InvalidArgument, d_inner) catch return error.InvalidArgument;

    const qkv_bytes = std.math.mul(usize, qkv_len, @sizeOf(f32)) catch return error.InvalidArgument;
    const head_bytes = std.math.mul(usize, @as(usize, n_v_heads), @sizeOf(f32)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
    const state_bytes = std.math.mul(usize, std.math.mul(usize, d_inner, @as(usize, d_head)) catch return error.InvalidArgument, @sizeOf(f32)) catch return error.InvalidArgument;

    if (qkv.size < qkv_bytes or beta_raw.size < head_bytes or a_raw.size < head_bytes or a_log.size < head_bytes or out.size < out_bytes or state.size < state_bytes) {
        return error.InvalidArgument;
    }
    if (dt_bias) |bias_buf| {
        if (bias_buf.size < head_bytes) return error.InvalidArgument;
    }
}

test "validateArgs rejects invalid head topology" {
    const vec = device_mod.Buffer{ .pointer = 0, .size = 64 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&vec, &vec, &vec, &vec, null, &vec, &vec, 3, 4, 8),
    );
}

test "validateArgs rejects undersized qkv buffer" {
    const qkv = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    const vec = device_mod.Buffer{ .pointer = 0, .size = 64 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&qkv, &vec, &vec, &vec, null, &vec, &vec, 1, 2, 8),
    );
}

test "validateArgs rejects undersized state buffer" {
    const qkv = device_mod.Buffer{ .pointer = 0, .size = 32 * @sizeOf(f32) };
    const vec = device_mod.Buffer{ .pointer = 0, .size = 8 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&qkv, &vec, &vec, &vec, null, &vec, &vec, 1, 1, 8),
    );
}
