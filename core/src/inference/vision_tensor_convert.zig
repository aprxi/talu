//! Shared tensor conversion helpers for inference vision runtimes.

const std = @import("std");
const tensor = @import("../tensor.zig");
const dtype_mod = @import("../dtype.zig");

const Tensor = tensor.Tensor;

pub fn tensorToOwnedF32(allocator: std.mem.Allocator, t: Tensor) ![]f32 {
    const n = t.numel;
    var out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);

    switch (t.dtype) {
        .f32 => {
            const src = t.asSlice(f32);
            @memcpy(out, src[0..n]);
        },
        .bf16 => {
            const src = t.asSliceUnaligned(u16);
            for (0..n) |i| out[i] = dtype_mod.bf16ToF32(src[i]);
        },
        .f16 => {
            const src = t.asSliceUnaligned(u16);
            for (0..n) |i| out[i] = dtype_mod.fp16ToF32(src[i]);
        },
        else => return error.UnsupportedDType,
    }

    return out;
}

test "tensorToOwnedF32 converts f32 and bf16 tensors" {
    const allocator = std.testing.allocator;

    var f32_data = [_]f32{ 1.0, 2.0 };
    const f32_tensor = Tensor.view2DSlice(&f32_data, 1, 2);
    const f32_out = try tensorToOwnedF32(allocator, f32_tensor);
    defer allocator.free(f32_out);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0 }, f32_out);

    var bf16_data = [_]u16{ 0x3F80, 0x4000 }; // 1.0, 2.0
    const bf16_tensor = Tensor{
        .dtype = .bf16,
        .n_dims = 1,
        .shape = .{ 2, 0, 0, 0, 0, 0, 0, 0 },
        .data_ptr = @ptrCast(bf16_data[0..].ptr),
        .data_size = bf16_data.len * @sizeOf(u16),
        .numel = bf16_data.len,
        .strides = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
    };
    const bf16_out = try tensorToOwnedF32(allocator, bf16_tensor);
    defer allocator.free(bf16_out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bf16_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), bf16_out[1], 1e-6);
}
