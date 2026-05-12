//! Static CUDA primitive, raw-copy, typed-copy, and cast capability facts.

const std = @import("std");
const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;
const descriptors = @import("descriptors.zig");

const backend = cap.Backend.cuda;

const f32_dtypes = [_]DType{.f32};
const u16_dtypes = [_]DType{.u16};
const row_major_layouts = [_]Layout{.row_major_contiguous};

pub const primitive_capabilities = descriptors.primitiveCapabilities(backend);

pub const copy_capabilities = [_]cap.CopyCapability{
    .{
        .backend = backend,
        .direction = .device_to_device,
        .dtypes = &f32_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 4,
    },
    .{
        .backend = backend,
        .direction = .device_to_device,
        .dtypes = &u16_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 2,
    },
};

pub const raw_copy_capabilities = [_]cap.RawCopyCapability{
    .{
        .backend = backend,
        .direction = .host_to_device,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .device_to_host,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .device_to_device,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .peer_device_to_device,
        .required_alignment = 1,
    },
};

pub const cast_capabilities = [_]cap.CastCapability{
    .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .bf16,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .src_dtype = .bf16,
        .dst_dtype = .f32,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
};

pub const PrimitiveCapabilities = struct {
    linalg: bool = supportsPrimitive("cublas_matmul_f32", .f32, .f32, .row_major_contiguous),
    normalization: bool = supportsPrimitive("rmsnorm_f32", .f32, .f32, .row_major_contiguous),
    activation: bool = supportsPrimitive("silu_f32", .f32, .f32, .row_major_contiguous),
    softmax: bool = supportsPrimitive("softmax_rows_f32", .f32, .f32, .row_major_contiguous),
    layout: bool = supportsPrimitive("vector_add_rows_strided_f32", .f32, .f32, .strided),
    memory: bool = supportsPrimitive("copy_f32", .f32, .f32, .row_major_contiguous),
    indexing: bool = supportsPrimitive("embedding_lookup_f32", .f32, .f32, .row_major_contiguous),
    quant_decode: bool = supportsPrimitive("dequant_kv_i8_to_f16", .i8, .f16, .row_major_contiguous),
    state_space: bool = supportsPrimitive("gated_delta_ssm_f32", .f32, .f32, .row_major_contiguous),
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

fn primitiveEntryCount(name: []const u8) usize {
    var count: usize = 0;
    for (primitive_capabilities) |entry| {
        if (entry.backend == backend and std.mem.eql(u8, entry.name, name)) count += 1;
    }
    return count;
}

test "compute CUDA primitive_capabilities have exactly one entry for every descriptor" {
    inline for (descriptors.primitive_descriptors) |descriptor| {
        try std.testing.expectEqual(@as(usize, 1), primitiveEntryCount(descriptor.name));
    }
}

test "compute CUDA primitive_capabilities resolve to descriptor-backed implementations" {
    for (primitive_capabilities) |entry| {
        try std.testing.expectEqual(@as(usize, 1), descriptors.descriptorCount(entry.name));
    }
}

test "compute CUDA primitive_capabilities declare each primitive name once" {
    for (primitive_capabilities, 0..) |entry, entry_idx| {
        for (primitive_capabilities[entry_idx + 1 ..]) |candidate| {
            try std.testing.expect(!std.mem.eql(u8, entry.name, candidate.name));
        }
    }
}

test "compute supportsPrimitive accepts supported CUDA input output dtype and layout combinations" {
    try std.testing.expect(supportsPrimitive("vector_add_rows_strided_f32", .f32, .f32, .strided));
    try std.testing.expect(supportsPrimitive("matvec_bf16_f32", .bf16, .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("attn_fused_heads_fp8_kv", .f8_e4m3, .f32, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("dequant_kv_i8_to_f16", .i8, .f16, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("cast_f32_to_f16", .f32, .f16, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("cublas_gemm_u16", .u16, .f32, .row_major_contiguous));
}

test "compute supportsPrimitive fails closed for unsupported CUDA primitive combinations" {
    try std.testing.expect(!supportsPrimitive("missing", .f32, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("vector_add_f32", .f16, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("vector_add_f32", .f32, .f32, .strided));
    try std.testing.expect(!supportsPrimitive("dequant_kv_i8_to_f16", .f32, .f16, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("dequant_kv_i8_to_f16", .i8, .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("cublas_matmul_i8_i8_i32", .f32, .i32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("cublas_gemm_u16", .u16, .u16, .row_major_contiguous));
}

test "compute CUDA raw copy typed copy and cast capabilities validate supported paths and fail closed" {
    try cap.validateRawCopy(&raw_copy_capabilities, .{
        .backend = backend,
        .src_device = cap.Device.cpu(),
        .dst_device = cap.Device.cuda(0),
        .byte_count = 16,
    });
    try cap.validateRawCopy(&raw_copy_capabilities, .{
        .backend = backend,
        .src_device = cap.Device.cuda(0),
        .dst_device = cap.Device.cuda(1),
        .byte_count = 16,
    });
    try cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    });
    try cap.validateCast(&cast_capabilities, .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .row_major_contiguous,
    });
    try std.testing.expectError(error.UnsupportedCopyDirection, cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expect(!cap.supportsCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .host_to_device,
        .dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedDType, cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .device_to_device,
        .dtype = .grouped_affine_u4,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedLayout, cap.validateCopy(&copy_capabilities, .{
        .backend = backend,
        .direction = .device_to_device,
        .dtype = .f32,
        .layout = .strided,
    }));
    try std.testing.expectError(error.UnsupportedCast, cap.validateCast(&cast_capabilities, .{
        .backend = backend,
        .src_dtype = .f16,
        .dst_dtype = .f32,
        .layout = .row_major_contiguous,
    }));
    try std.testing.expectError(error.UnsupportedLayout, cap.validateCast(&cast_capabilities, .{
        .backend = backend,
        .src_dtype = .f32,
        .dst_dtype = .f16,
        .layout = .strided,
    }));
}
