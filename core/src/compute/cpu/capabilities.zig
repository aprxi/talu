//! Static CPU primitive, copy, and cast capability facts.

const std = @import("std");
const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;

const backend = cap.Backend.cpu;

const dense_dtypes = [_]DType{ .f32, .f64, .f16, .bf16, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f8_e4m3 };
const namespace_dtypes = [_]DType{};
const namespace_layouts = [_]Layout{};
const row_major_layouts = [_]Layout{.row_major_contiguous};

fn primitiveNamespace(name: []const u8) cap.PrimitiveCapability {
    return .{
        .backend = backend,
        .name = name,
        .dtypes = &namespace_dtypes,
        .layouts = &namespace_layouts,
        .required_alignment = 1,
    };
}

pub const primitive_capabilities = [_]cap.PrimitiveCapability{
    primitiveNamespace("common"),
    primitiveNamespace("tensor_view"),
    primitiveNamespace("activation_view"),
    primitiveNamespace("mxfp4"),
    primitiveNamespace("math"),
    primitiveNamespace("simd"),
    primitiveNamespace("quant"),
    primitiveNamespace("linalg"),
    primitiveNamespace("layout"),
    primitiveNamespace("memory"),
    primitiveNamespace("recurrence"),
    primitiveNamespace("indexing"),
    primitiveNamespace("activation"),
    primitiveNamespace("gated_attention"),
    primitiveNamespace("gated_delta"),
    primitiveNamespace("elementwise"),
    primitiveNamespace("normalization"),
    primitiveNamespace("rowwise"),
    primitiveNamespace("image_ops"),
    primitiveNamespace("quant_decode"),
    primitiveNamespace("rotary"),
    primitiveNamespace("conv1d_depthwise"),
    primitiveNamespace("topk"),
    primitiveNamespace("reduction"),
    primitiveNamespace("math_fast"),
    primitiveNamespace("softmax"),
    primitiveNamespace("sampling_ops"),
    primitiveNamespace("sdpa_rowwise"),
    primitiveNamespace("linalg_sdpa"),
    primitiveNamespace("state_space"),
    primitiveNamespace("matmul"),
    primitiveNamespace("accelerate"),
    primitiveNamespace("metal_accel"),
};

pub const copy_capabilities = [_]cap.CopyCapability{
    .{
        .backend = backend,
        .direction = .host_to_host,
        .dtypes = &dense_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
};

pub const cast_capabilities = [_]cap.CastCapability{};

pub const PrimitiveCapabilities = struct {
    linalg: bool = hasPrimitiveNamespace("linalg"),
    normalization: bool = hasPrimitiveNamespace("normalization"),
    activation: bool = hasPrimitiveNamespace("activation"),
    softmax: bool = hasPrimitiveNamespace("softmax"),
    layout: bool = hasPrimitiveNamespace("layout"),
    memory: bool = hasPrimitiveNamespace("memory"),
    indexing: bool = hasPrimitiveNamespace("indexing"),
    quant_decode: bool = hasPrimitiveNamespace("quant_decode"),
    state_space: bool = hasPrimitiveNamespace("state_space"),
};

pub const support: PrimitiveCapabilities = .{};

fn hasPrimitiveNamespace(name: []const u8) bool {
    return cap.hasPrimitiveEntry(&primitive_capabilities, backend, name);
}

pub fn supportsPrimitive(name: []const u8, dtype: DType, layout: Layout) bool {
    return cap.supportsPrimitive(&primitive_capabilities, .{
        .backend = backend,
        .name = name,
        .dtype = dtype,
        .layout = layout,
    });
}

test "compute supportsPrimitive accepts public CPU primitive namespaces" {
    const expected = [_][]const u8{
        "common",
        "tensor_view",
        "activation_view",
        "mxfp4",
        "math",
        "simd",
        "quant",
        "linalg",
        "layout",
        "memory",
        "recurrence",
        "indexing",
        "activation",
        "gated_attention",
        "gated_delta",
        "elementwise",
        "normalization",
        "rowwise",
        "image_ops",
        "quant_decode",
        "rotary",
        "conv1d_depthwise",
        "topk",
        "reduction",
        "math_fast",
        "softmax",
        "sampling_ops",
        "sdpa_rowwise",
        "linalg_sdpa",
        "state_space",
        "matmul",
        "accelerate",
        "metal_accel",
    };
    inline for (expected) |name| {
        try std.testing.expect(cap.hasPrimitiveEntry(&primitive_capabilities, backend, name));
    }
}

test "compute supportsPrimitive fails closed for unknown CPU primitive and dtype" {
    try std.testing.expect(!supportsPrimitive("missing", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("linalg", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("linalg", .grouped_affine_u4, .row_major_contiguous));
}

test "compute CPU copy_capabilities support host dense copies only" {
    try cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    });
    try std.testing.expectError(error.UnsupportedCopyDirection, cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .device_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
}
