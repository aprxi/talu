//! Fused dense u16-weight (f16/bf16) gate/up matvec kernel wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const args_mod = @import("args.zig");
const launch_mod = @import("launch.zig");
const module_mod = @import("module.zig");

const cuda_assets = @import("cuda_assets");
pub const embedded_module = cuda_assets.kernels_fatbin;
pub const embedded_symbol_f16: [:0]const u8 = "talu_matvec_gate_up_f16_f32";
pub const embedded_symbol_bf16: [:0]const u8 = "talu_matvec_gate_up_bf16_f32";
pub const op_name_f16: []const u8 = "matvec_gate_up_f16_f32";
pub const op_name_bf16: []const u8 = "matvec_gate_up_bf16_f32";

pub fn runWithFunction(
    arg_pack: *args_mod.ArgPack,
    device: *device_mod.Device,
    function: module_mod.Function,
    input: *const device_mod.Buffer,
    gate_weight_u16: *const device_mod.Buffer,
    gate_out: *device_mod.Buffer,
    gate_out_dim: u32,
    up_weight_u16: *const device_mod.Buffer,
    up_out: *device_mod.Buffer,
    up_out_dim: u32,
    in_dim: u32,
) !void {
    try validateArgs(
        input,
        gate_weight_u16,
        gate_out,
        gate_out_dim,
        up_weight_u16,
        up_out,
        up_out_dim,
        in_dim,
    );

    arg_pack.reset();
    try arg_pack.appendBufferPtr(input);
    try arg_pack.appendBufferPtr(gate_weight_u16);
    try arg_pack.appendBufferPtr(gate_out);
    try arg_pack.appendScalar(u32, gate_out_dim);
    try arg_pack.appendBufferPtr(up_weight_u16);
    try arg_pack.appendBufferPtr(up_out);
    try arg_pack.appendScalar(u32, up_out_dim);
    try arg_pack.appendScalar(u32, in_dim);

    const total = std.math.add(u32, gate_out_dim, up_out_dim) catch return error.InvalidArgument;
    const block_x: u32 = 256;
    try launch_mod.launch(device, function, .{
        .grid_x = ceilDiv(total, block_x),
        .block_x = block_x,
    }, arg_pack);
}

fn validateArgs(
    input: *const device_mod.Buffer,
    gate_weight_u16: *const device_mod.Buffer,
    gate_out: *device_mod.Buffer,
    gate_out_dim: u32,
    up_weight_u16: *const device_mod.Buffer,
    up_out: *device_mod.Buffer,
    up_out_dim: u32,
    in_dim: u32,
) !void {
    if (in_dim == 0 or gate_out_dim == 0 or up_out_dim == 0) return error.InvalidArgument;

    try validateOne(input, gate_weight_u16, gate_out, in_dim, gate_out_dim);
    try validateOne(input, up_weight_u16, up_out, in_dim, up_out_dim);
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
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 2, &b, &out, 2, 0));
    try std.testing.expectError(error.InvalidArgument, validateArgs(&b, &b, &out, 0, &b, &out, 2, 2));
}

test "validateArgs rejects undersized up weight buffer" {
    const input = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    const small_weight = device_mod.Buffer{ .pointer = 0, .size = 8 };
    var out = device_mod.Buffer{ .pointer = 0, .size = 4 * @sizeOf(f32) };
    try std.testing.expectError(
        error.InvalidArgument,
        validateArgs(&input, &input, &out, 4, &small_weight, &out, 4, 4),
    );
}
