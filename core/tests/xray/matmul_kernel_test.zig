//! Integration tests for xray.MatmulKernel
//!
//! MatmulKernel is an enum selecting the matmul implementation based on
//! model quantization. Provides name() and weightDtype() methods.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const MatmulKernel = xray.MatmulKernel;

// ===== name =====

test "MatmulKernel: name returns unique string for each variant" {
    const names = [_][:0]const u8{
        MatmulKernel.matmul_f32.name(),
        MatmulKernel.matmul_f16.name(),
        MatmulKernel.matmul_bf16.name(),
        MatmulKernel.matmul_grouped_affine_u4.name(),
        MatmulKernel.matmul_grouped_affine_u8.name(),
        MatmulKernel.matmul_mxfp4.name(),
    };

    // Verify each name is non-empty and unique
    for (names, 0..) |a, i| {
        try std.testing.expect(a.len > 0);
        for (names[i + 1 ..]) |b| {
            try std.testing.expect(!std.mem.eql(u8, a, b));
        }
    }
}

test "MatmulKernel: name matches expected strings" {
    try std.testing.expectEqualStrings("matmul_f32", MatmulKernel.matmul_f32.name());
    try std.testing.expectEqualStrings("matmul_bf16", MatmulKernel.matmul_bf16.name());
    try std.testing.expectEqualStrings("matmul_grouped_affine_u4", MatmulKernel.matmul_grouped_affine_u4.name());
    try std.testing.expectEqualStrings("matmul_mxfp4", MatmulKernel.matmul_mxfp4.name());
}

// ===== weightDtype =====

test "MatmulKernel: weightDtype returns distinct dtype per variant" {
    // Each kernel maps to a specific weight dtype — verify they're distinct
    // where kernels differ and consistent with the naming.
    const f32_dtype = MatmulKernel.matmul_f32.weightDtype();
    const bf16_dtype = MatmulKernel.matmul_bf16.weightDtype();
    const u4_dtype = MatmulKernel.matmul_grouped_affine_u4.weightDtype();
    const u8_dtype = MatmulKernel.matmul_grouped_affine_u8.weightDtype();
    const mxfp4_dtype = MatmulKernel.matmul_mxfp4.weightDtype();

    // Standard types differ from quantized types
    try std.testing.expect(f32_dtype != bf16_dtype);
    try std.testing.expect(f32_dtype != u4_dtype);
    try std.testing.expect(u4_dtype != u8_dtype);
    try std.testing.expect(u4_dtype != mxfp4_dtype);

    // Quantized types are identified as quantized
    try std.testing.expect(u4_dtype.isQuantized());
    try std.testing.expect(u8_dtype.isQuantized());
    try std.testing.expect(mxfp4_dtype.isQuantized());

    // Standard types are not quantized
    try std.testing.expect(!f32_dtype.isQuantized());
    try std.testing.expect(!bf16_dtype.isQuantized());
}
