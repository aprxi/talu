//! Fused dense u16-weight (f16/bf16) QKV matvec kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol_f16: [:0]const u8 = "talu_matvec_qkv_f16_f32";
pub const embedded_symbol_bf16: [:0]const u8 = "talu_matvec_qkv_bf16_f32";
pub const op_name_f16: []const u8 = "matvec_qkv_f16_f32";
pub const op_name_bf16: []const u8 = "matvec_qkv_bf16_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    q_weight_u16: *const device_mod.Buffer,
    q_out: *device_mod.Buffer,
    q_out_dim: u32,
    k_weight_u16: *const device_mod.Buffer,
    k_out: *device_mod.Buffer,
    k_out_dim: u32,
    v_weight_u16: *const device_mod.Buffer,
    v_out: *device_mod.Buffer,
    v_out_dim: u32,
    in_dim: u32,
) !void {
    try validateArgs(
        input,
        q_weight_u16,
        q_out,
        q_out_dim,
        k_weight_u16,
        k_out,
        k_out_dim,
        v_weight_u16,
        v_out,
        v_out_dim,
        in_dim,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(q_weight_u16);
    try arg_pack.appendBufferPtr(q_out);
    try arg_pack.appendScalar(u32, q_out_dim);
    try arg_pack.appendBufferPtr(k_weight_u16);
    try arg_pack.appendBufferPtr(k_out);
    try arg_pack.appendScalar(u32, k_out_dim);
    try arg_pack.appendBufferPtr(v_weight_u16);
    try arg_pack.appendBufferPtr(v_out);
    try arg_pack.appendScalar(u32, v_out_dim);
    try arg_pack.appendScalar(u32, in_dim);

    const qk = std.math.add(u32, q_out_dim, k_out_dim) catch return error.InvalidArgument;
    const total = std.math.add(u32, qk, v_out_dim) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(total, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    q_weight_u16: *const device_mod.Buffer,
    q_out: *device_mod.Buffer,
    q_out_dim: u32,
    k_weight_u16: *const device_mod.Buffer,
    k_out: *device_mod.Buffer,
    k_out_dim: u32,
    v_weight_u16: *const device_mod.Buffer,
    v_out: *device_mod.Buffer,
    v_out_dim: u32,
    in_dim: u32,
) !void {
    if (in_dim == 0 or q_out_dim == 0 or k_out_dim == 0 or v_out_dim == 0) return error.InvalidArgument;

    try validateOne(input, q_weight_u16, q_out, in_dim, q_out_dim);
    try validateOne(input, k_weight_u16, k_out, in_dim, k_out_dim);
    try validateOne(input, v_weight_u16, v_out, in_dim, v_out_dim);
}

fn validateOne(
    input: *const device_mod.Buffer,
    weight_u16: *const device_mod.Buffer,
    out: *device_mod.Buffer,
    in_dim: u32,
    out_dim: u32,
) !void {
    const input_bytes = std.math.mul(usize, @as(usize, in_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    const weight_elems = std.math.mul(usize, @as(usize, in_dim), @as(usize, out_dim)) catch return error.InvalidArgument;
    const weight_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const out_bytes = std.math.mul(usize, @as(usize, out_dim), @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size < input_bytes or weight_u16.size < weight_bytes or out.size < out_bytes) return error.InvalidArgument;
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

test "validateArgs rejects zero dimensions" {
    const b = device_mod.Buffer{ .pointer = 0, .size = 4096 };
    var out = b;
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 2, &b, &out, 2, &b, &out, 2, 0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 0, &b, &out, 2, &b, &out, 2, 2));
}

test "validateArgs rejects undersized q weight buffer" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    const small_weight = device_mod.Buffer{ .pointer = 0, .size = 8 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&input, &small_weight, &out, 4, &input, &out, 4, &input, &out, 4, 4),
    );
}
