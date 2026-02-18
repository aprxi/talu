//! Shared tensor conversion helpers for CPU vision runtimes.

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const dtype_mod = @import("../../../../dtype.zig");

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
