//! Formatting helpers for kernel introspection/describe methods.
//! This file contains no imports that would create circular dependencies,
//! so kernel files can import it directly.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const inspect = @import("../../../../xray/root.zig");

pub const Tensor = tensor.Tensor;
pub const KernelOp = inspect.kernel_info.KernelOp;
pub const kernel_info = inspect.kernel_info;

fn writeIndent(writer: anytype, indent_level: usize) !void {
    try writer.writeByteNTimes(' ', indent_level);
}

fn formatLinearLike(
    writer: anytype,
    weight: *const Tensor,
    bias: ?[]const f32,
    in_features: usize,
    out_features: usize,
) !void {
    const weight_dtype = weight.dtype;
    if (weight_dtype == .grouped_affine_u4) {
        const group_size = if (weight.gaffine) |meta| meta.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=4, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else if (weight_dtype == .grouped_affine_u8) {
        const group_size = if (weight.gaffine) |meta| meta.group_size else 64;
        try writer.print("QuantizedLinear(in={}, out={}, bits=8, group_size={})", .{
            in_features,
            out_features,
            group_size,
        });
    } else {
        const dtype_name: []const u8 = switch (weight_dtype) {
            .f32 => "f32",
            .f16 => "f16",
            .bf16 => "bf16",
            .q5_0 => "q5_0",
            else => "unknown",
        };
        try writer.print("Linear(in={}, out={}, bias={}, dtype={s})", .{
            in_features,
            out_features,
            bias != null,
            dtype_name,
        });
    }
}

fn formatRmsNormLike(writer: anytype, dim: usize, eps: f32, weight_offset: f32) !void {
    if (weight_offset != 0.0) {
        try writer.print("RMSNorm(dim={}, eps={e}, weight_offset={d:.1})", .{ dim, eps, weight_offset });
    } else {
        try writer.print("RMSNorm(dim={}, eps={e})", .{ dim, eps });
    }
}

fn formatSeqMatmulOp(
    writer: anytype,
    indent: usize,
    in_features: usize,
    out_features: usize,
    dtype: @import("../../../../dtype.zig").DType,
) !void {
    const matmul_op = KernelOp{ .matmul = .{
        .m = .seq,
        .k = in_features,
        .n = out_features,
        .dtype = dtype,
        .kernel_name = kernel_info.matmulKernelName(dtype),
    } };
    try matmul_op.format(writer, indent);
}

fn describeLinearLine(
    writer: anytype,
    indent: usize,
    label: []const u8,
    weight: *const Tensor,
    bias: ?[]const f32,
    in_features: usize,
    out_features: usize,
) !void {
    try writeIndent(writer, indent);
    try writer.print("({s}): ", .{label});
    try formatLinearLike(writer, weight, bias, in_features, out_features);
    try writer.writeAll("\n");
}

fn describeRmsNormLine(
    writer: anytype,
    indent: usize,
    label: []const u8,
    dim: usize,
    eps: f32,
    weight_offset: f32,
) !void {
    try writeIndent(writer, indent);
    try writer.print("({s}): ", .{label});
    try formatRmsNormLike(writer, dim, eps, weight_offset);
    try writer.writeAll("\n");
}
