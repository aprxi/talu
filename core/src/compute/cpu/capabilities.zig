//! Static CPU primitive, copy, and cast capability facts.

const std = @import("std");
const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;

const activation = @import("activation.zig");
const indexing = @import("tensor_gather.zig");
const matmul = @import("matmul_primitives.zig");
const normalization = @import("normalization.zig");
const quant_decode = @import("quant_decode.zig");
const softmax = @import("softmax.zig");
const state_space = @import("state_space.zig");

const backend = cap.Backend.cpu;

const dense_dtypes = [_]DType{ .f32, .f64, .f16, .bf16, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f8_e4m3 };
const f32_dtypes = [_]DType{.f32};
const grouped_affine_u4_dtypes = [_]DType{.grouped_affine_u4};
const u16_dtypes = [_]DType{.u16};
const row_major_layouts = [_]Layout{.row_major_contiguous};

fn primitive(name: []const u8, input_dtypes: []const DType, output_dtypes: []const DType, layouts: []const Layout) cap.PrimitiveCapability {
    return .{
        .backend = backend,
        .name = name,
        .input_dtypes = input_dtypes,
        .output_dtypes = output_dtypes,
        .layouts = layouts,
        .required_alignment = 1,
    };
}

pub const primitive_capabilities = [_]cap.PrimitiveCapability{
    primitive("matmul_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("matmul_gaffine_u4", &grouped_affine_u4_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("rmsnorm_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("silu_mul_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("softmax_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("gather_rows_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("decode_f16_row_to_f32", &u16_dtypes, &f32_dtypes, &row_major_layouts),
    primitive("state_scan_f32", &f32_dtypes, &f32_dtypes, &row_major_layouts),
};

pub const raw_copy_capabilities = [_]cap.RawCopyCapability{
    .{
        .backend = backend,
        .direction = .host_to_host,
        .required_alignment = 1,
    },
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
    linalg: bool = supportsPrimitive("matmul_f32", .f32, .f32, .row_major_contiguous),
    normalization: bool = supportsPrimitive("rmsnorm_f32", .f32, .f32, .row_major_contiguous),
    activation: bool = supportsPrimitive("silu_mul_f32", .f32, .f32, .row_major_contiguous),
    softmax: bool = supportsPrimitive("softmax_f32", .f32, .f32, .row_major_contiguous),
    layout: bool = false,
    memory: bool = false,
    indexing: bool = supportsPrimitive("gather_rows_f32", .f32, .f32, .row_major_contiguous),
    quant_decode: bool = supportsPrimitive("decode_f16_row_to_f32", .u16, .f32, .row_major_contiguous),
    state_space: bool = supportsPrimitive("state_scan_f32", .f32, .f32, .row_major_contiguous),
};

pub const support: PrimitiveCapabilities = .{};

pub fn supportsPrimitive(name: []const u8, input_dtype: DType, output_dtype: DType, layout: Layout) bool {
    return cap.supportsPrimitive(&primitive_capabilities, .{
        .backend = backend,
        .name = name,
        .input_dtype = input_dtype,
        .output_dtype = output_dtype,
        .layout = layout,
    });
}

test "compute CPU primitive_capabilities are backed by local implementation symbols" {
    _ = matmul.matmulF32;
    _ = matmul.matmulGaffineU4;
    _ = normalization.rmsnormInPlace;
    _ = activation.siluMulSplit;
    _ = softmax.stableInPlace;
    _ = indexing.gatherRowsF32;
    _ = quant_decode.decodeF16Row;
    _ = state_space.scanStep;
}

test "compute supportsPrimitive accepts concrete CPU primitive facts" {
    try std.testing.expect(supportsPrimitive("matmul_f32", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("matmul_gaffine_u4", .grouped_affine_u4, .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("rmsnorm_f32", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("softmax_f32", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(support.linalg);
    try std.testing.expect(support.softmax);
}

test "compute supportsPrimitive rejects CPU namespaces and unsupported combinations" {
    try std.testing.expect(!supportsPrimitive("missing", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("linalg", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("matmul_f32", .grouped_affine_u4, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("matmul_gaffine_u4", .grouped_affine_u4, .grouped_affine_u4, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("matmul_f32", .f32, .f32, .strided));
}

test "compute CPU raw_copy_capabilities support host byte copies only" {
    try cap.validateRawCopy(&raw_copy_capabilities, .{
        .backend = backend,
        .src_device = cap.Device.cpu(),
        .dst_device = cap.Device.cpu(),
        .byte_count = 16,
    });
    try std.testing.expectError(error.UnsupportedDevice, cap.validateRawCopy(&raw_copy_capabilities, .{
        .backend = backend,
        .src_device = cap.Device.cuda(0),
        .dst_device = cap.Device.cpu(),
        .byte_count = 16,
    }));
    try cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_host,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    });
    try std.testing.expect(!cap.supportsCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_host,
        .dtype = .grouped_affine_u4,
        .layout = .row_major_contiguous,
    }));
}
