//! Static Metal and MLX primitive, copy, and cast capability facts.

const std = @import("std");
const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;

const backend = cap.Backend.metal;

const f32_dtypes = [_]DType{.f32};
const f32_i8_dtypes = [_]DType{ .f32, .i8 };
const f32_gaffine_u4_dtypes = [_]DType{ .f32, .grouped_affine_u4 };
const mlx_dense_dtypes = [_]DType{ .f32, .f16, .bf16, .u8, .u32 };
const mlx_quantized_dtypes = [_]DType{ .f32, .f16, .bf16, .grouped_affine_u4, .grouped_affine_u8, .mxfp4, .f8_e4m3 };
const row_major_layouts = [_]Layout{.row_major_contiguous};
const backend_native_layouts = [_]Layout{.opaque_backend};

fn primitive(name: []const u8, dtypes: []const DType, layouts: []const Layout) cap.PrimitiveCapability {
    return .{
        .backend = backend,
        .name = name,
        .dtypes = dtypes,
        .layouts = layouts,
        .required_alignment = 1,
    };
}

pub const primitive_capabilities = [_]cap.PrimitiveCapability{
    primitive("matmul_f32", &f32_dtypes, &row_major_layouts),
    primitive("matmul_f32_trans_b_scaled", &f32_dtypes, &row_major_layouts),
    primitive("matmul_f32_i8_trans_b_scaled", &f32_i8_dtypes, &row_major_layouts),
    primitive("matmul_gaffine_u4", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("mlx_matmul_gaffine_u4", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("mlx_rms_norm", &f32_dtypes, &row_major_layouts),
    primitive("mlx_rope", &f32_dtypes, &row_major_layouts),
    primitive("mlx_array_create", &mlx_dense_dtypes, &row_major_layouts),
    primitive("mlx_array_copy_to_host", &mlx_dense_dtypes, &row_major_layouts),
    primitive("mlx_lazy_graph", &mlx_quantized_dtypes, &backend_native_layouts),
    primitive("mlx_backend_native_graph", &mlx_quantized_dtypes, &backend_native_layouts),
    primitive("buffer_upload", &mlx_dense_dtypes, &row_major_layouts),
    primitive("buffer_download", &mlx_dense_dtypes, &row_major_layouts),
};

pub const copy_capabilities = [_]cap.CopyCapability{
    .{
        .backend = backend,
        .direction = .host_to_device,
        .dtypes = &mlx_dense_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .device_to_host,
        .dtypes = &mlx_dense_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .backend_native,
        .dtypes = &mlx_quantized_dtypes,
        .layouts = &backend_native_layouts,
        .required_alignment = 1,
    },
};

pub const cast_capabilities = [_]cap.CastCapability{
    .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layouts = &backend_native_layouts,
        .required_alignment = 1,
    },
};

pub const PrimitiveCapabilities = struct {
    linalg: bool = supportsPrimitive("matmul_f32", .f32, .row_major_contiguous),
    normalization: bool = supportsPrimitive("mlx_rms_norm", .f32, .row_major_contiguous),
    activation: bool = supportsPrimitive("mlx_lazy_graph", .f32, .opaque_backend),
    softmax: bool = supportsPrimitive("mlx_lazy_graph", .f32, .opaque_backend),
    layout: bool = supportsPrimitive("mlx_lazy_graph", .f32, .opaque_backend),
    memory: bool = supportsPrimitive("buffer_upload", .f32, .row_major_contiguous),
    indexing: bool = supportsPrimitive("mlx_lazy_graph", .u32, .opaque_backend),
    quant_decode: bool = supportsPrimitive("mlx_lazy_graph", .grouped_affine_u4, .opaque_backend),
    state_space: bool = supportsPrimitive("mlx_backend_native_graph", .bf16, .opaque_backend),
};

pub const support: PrimitiveCapabilities = .{};

pub fn supportsPrimitive(name: []const u8, dtype: DType, layout: Layout) bool {
    return cap.supportsPrimitive(&primitive_capabilities, .{
        .backend = backend,
        .name = name,
        .dtype = dtype,
        .layout = layout,
    });
}

test "compute supportsPrimitive advertises Metal matmul and MLX graph capabilities" {
    try std.testing.expect(supportsPrimitive("matmul_f32", .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("matmul_gaffine_u4", .grouped_affine_u4, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("mlx_rms_norm", .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("mlx_rope", .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("mlx_lazy_graph", .bf16, .opaque_backend));
    try std.testing.expect(supportsPrimitive("buffer_upload", .u8, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("buffer_download", .u8, .row_major_contiguous));
}

test "compute supportsPrimitive rejects generic byte layout for backend-native MLX graph" {
    try std.testing.expect(!supportsPrimitive("mlx_lazy_graph", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("mlx_backend_native_graph", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("buffer_upload", .f32, .opaque_backend));
    try std.testing.expect(!supportsPrimitive("mlx_rms_norm", .f32, .opaque_backend));
}

test "compute metal copy_capabilities and cast_capabilities fail closed" {
    try cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    });
    try std.testing.expectError(error.UnsupportedCopyDirection, cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));

    try cap.validateCast(&cast_capabilities, .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .opaque_backend,
    });
    try std.testing.expectError(error.UnsupportedCast, cap.validateCast(&cast_capabilities, .{
        .backend = backend,
        .src_dtype = .f16,
        .dst_dtype = .f32,
        .layout = .opaque_backend,
    }));
}
