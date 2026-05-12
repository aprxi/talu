//! CUDA primitive descriptors owned by implementation modules.
//!
//! Descriptors reference each wrapper's exported op name constants so the
//! capability table cannot drift into a second string registry.

const std = @import("std");

const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;

const matmul = @import("matmul.zig");

const f32_dtypes = [_]DType{.f32};
const f16_dtypes = [_]DType{.f16};
const bf16_dtypes = [_]DType{.bf16};
const i8_dtypes = [_]DType{.i8};
const i32_dtypes = [_]DType{.i32};
const u16_dtypes = [_]DType{.u16};
const u32_dtypes = [_]DType{.u32};
const fp8_dtypes = [_]DType{.f8_e4m3};
const grouped_affine_u4 = [_]DType{.grouped_affine_u4};
const grouped_affine_u8 = [_]DType{.grouped_affine_u8};
const mxfp4 = [_]DType{.mxfp4};
const f32_or_u32_dtypes = [_]DType{ .f32, .u32 };

const row_major = [_]Layout{.row_major_contiguous};
const row_major_and_strided = [_]Layout{ .row_major_contiguous, .strided };

pub const PrimitiveDescriptor = struct {
    name: []const u8,
    input_dtypes: []const DType,
    output_dtypes: []const DType,
    layouts: []const Layout,
    rank_limits: cap.RankLimits = .{},
    required_alignment: usize = 1,
};

fn primitive(name: []const u8, input_dtypes: []const DType, output_dtypes: []const DType, layouts: []const Layout) PrimitiveDescriptor {
    return .{
        .name = name,
        .input_dtypes = input_dtypes,
        .output_dtypes = output_dtypes,
        .layouts = layouts,
    };
}

fn same(name: []const u8, dtypes: []const DType, layouts: []const Layout) PrimitiveDescriptor {
    return primitive(name, dtypes, dtypes, layouts);
}

pub const primitive_descriptors = [_]PrimitiveDescriptor{
    same(matmul.op_name_cublas_matmul_f32, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublas_matmul_u16_f32, &u16_dtypes, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublas_matmul_u16_u16_f32, &u16_dtypes, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublas_matmul_i8_i8_i32, &i8_dtypes, &i32_dtypes, &row_major),
    primitive(matmul.op_name_cublas_matmul_fp8_fp8_f32, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublas_gemm_u16_strided_batched, &u16_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(matmul.op_name_cublas_gemm_u16, &u16_dtypes, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublaslt_matmul_mxfp8, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(matmul.op_name_cublaslt_matmul_nvfp4, &mxfp4, &f32_dtypes, &row_major),

    primitive(@import("argmax.zig").op_name, &f32_dtypes, &u32_dtypes, &row_major),
    primitive(@import("attn_fused_decode_heads_f16_kv_ptrs.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_fused_decode_heads_fp8_kv_ptrs.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_fused_decode_heads_i8_kv_ptrs.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_fused_heads_f16_kv.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_heads_fp8_kv.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_heads_i8_kv.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_f16_kv.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_f16_kv_gqa.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_fp8_kv.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_fp8_kv_gqa.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_i8_kv.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_fused_prefill_heads_i8_kv_gqa.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_scores_heads_f16_kv.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_scores_heads_f16_kv_ptrs.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major_and_strided),
    same(@import("attn_scores_heads_f32.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("attn_scores_heads_fp8_kv.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_scores_heads_fp8_kv_ptrs.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_scores_heads_i8_kv.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_scores_heads_i8_kv_ptrs.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_weighted_sum_heads_f16_kv.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_weighted_sum_heads_f16_kv_ptrs.zig").op_name, &f16_dtypes, &f32_dtypes, &row_major_and_strided),
    same(@import("attn_weighted_sum_heads_f32.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("attn_weighted_sum_heads_fp8_kv.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_weighted_sum_heads_fp8_kv_ptrs.zig").op_name, &fp8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("attn_weighted_sum_heads_i8_kv.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("attn_weighted_sum_heads_i8_kv_ptrs.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major_and_strided),
    primitive(@import("cast_bf16_to_f32.zig").op_name, &bf16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("cast_f32_to_bf16.zig").op_name, &f32_dtypes, &bf16_dtypes, &row_major),
    primitive(@import("cast_f32_to_f16.zig").op_name, &f32_dtypes, &f16_dtypes, &row_major),
    same(@import("causal_attn_softmax_f32.zig").op_name, &f32_dtypes, &row_major),
    same(@import("copy.zig").op_name, &f32_dtypes, &row_major),
    same(@import("copy_u16.zig").op_name, &u16_dtypes, &row_major),
    same(@import("decode_u32_increment.zig").op_name, &u32_dtypes, &row_major),
    primitive(@import("dequant_kv_fp8_to_f16.zig").op_name, &fp8_dtypes, &f16_dtypes, &row_major),
    primitive(@import("dequant_kv_i8_to_f16.zig").op_name, &i8_dtypes, &f16_dtypes, &row_major),
    same(@import("embedding_lookup_f32.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("embedding_lookup_gaffine_u4.zig").op_name, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("embedding_lookup_u16.zig").op_name, &u16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("embedding_lookup_u16_rows.zig").op_name, &u16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("flash_decode.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("flash_decode.zig").op_name_i8, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("flash_decode.zig").op_name_fp8, &fp8_dtypes, &f32_dtypes, &row_major),
    same(@import("flash_decode.zig").op_name_reduce, &f32_dtypes, &row_major),
    primitive(@import("flash_prefill.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("flash_prefill.zig").op_name_i8, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("flash_prefill.zig").op_name_fp8, &fp8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_dequantize_f16.zig").op_name, &grouped_affine_u4, &f16_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec.zig").op_name, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec.zig").op_name_tile8, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec_gate_up.zig").op_name, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec_gate_up_silu.zig").op_name, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec_gate_up_silu.zig").op_name_tile8, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec_qkv.zig").op_name, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u4_matvec_qkv.zig").op_name_tile8, &grouped_affine_u4, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u8_dequantize_f16.zig").op_name, &grouped_affine_u8, &f16_dtypes, &row_major),
    primitive(@import("gaffine_u8_matvec.zig").op_name, &grouped_affine_u8, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u8_matvec_gate_up.zig").op_name, &grouped_affine_u8, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u8_matvec_gate_up_silu.zig").op_name, &grouped_affine_u8, &f32_dtypes, &row_major),
    primitive(@import("gaffine_u8_matvec_qkv.zig").op_name, &grouped_affine_u8, &f32_dtypes, &row_major),
    same(@import("gated_attention_compact_q.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_attention_output_gate.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_conv.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_conv_silu.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_conv_silu_rows.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_conv_silu_rows_ptrs.zig").op_name, &f32_dtypes, &row_major_and_strided),
    same(@import("gated_delta_conv_silu_rows_ptrs.zig").op_name_advance, &f32_dtypes, &row_major_and_strided),
    same(@import("gated_delta_qk_norm.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_rmsnorm_silu_mul.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_rmsnorm_silu_mul_rows.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_ssm.zig").op_name, &f32_dtypes, &row_major),
    same(@import("gated_delta_ssm_rows.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("gated_delta_ssm_rows_i8.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    same(@import("gated_delta_ssm_rows_ptrs.zig").op_name, &f32_dtypes, &row_major_and_strided),
    primitive(@import("gated_delta_ssm_rows_ptrs_i8.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major_and_strided),
    same(@import("gelu_mul.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("i8_matvec_gate_up_silu.zig").op_name, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("i8_matvec_gate_up_silu.zig").op_name_tile8, &i8_dtypes, &f32_dtypes, &row_major),
    primitive(@import("kv_write_f16.zig").op_name, &f32_dtypes, &f16_dtypes, &row_major),
    primitive(@import("kv_write_f16_rows.zig").op_name, &f32_dtypes, &f16_dtypes, &row_major),
    primitive(@import("kv_write_f16_rows_ptrs.zig").op_name, &f32_dtypes, &f16_dtypes, &row_major_and_strided),
    primitive(@import("kv_write_fp8.zig").op_name, &f32_dtypes, &fp8_dtypes, &row_major),
    primitive(@import("kv_write_fp8_rows.zig").op_name, &f32_dtypes, &fp8_dtypes, &row_major),
    primitive(@import("kv_write_fp8_rows_ptrs.zig").op_name, &f32_dtypes, &fp8_dtypes, &row_major_and_strided),
    primitive(@import("kv_write_i8.zig").op_name, &f32_dtypes, &i8_dtypes, &row_major),
    primitive(@import("kv_write_i8_rows.zig").op_name, &f32_dtypes, &i8_dtypes, &row_major),
    primitive(@import("kv_write_i8_rows_ptrs.zig").op_name, &f32_dtypes, &i8_dtypes, &row_major_and_strided),
    primitive(@import("matmul_u16.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matmul_u16.zig").op_name_bf16, &bf16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16.zig").op_name_bf16, &bf16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_gate_up.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_gate_up.zig").op_name_bf16, &bf16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_gate_up_silu.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_gate_up_silu.zig").op_name_bf16, &bf16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_qkv.zig").op_name_f16, &f16_dtypes, &f32_dtypes, &row_major),
    primitive(@import("matvec_u16_qkv.zig").op_name_bf16, &bf16_dtypes, &f32_dtypes, &row_major),
    same(@import("mul.zig").op_name, &f32_dtypes, &row_major),
    primitive(@import("nvfp4_matvec.zig").op_name, &mxfp4, &f32_dtypes, &row_major),
    primitive(@import("nvfp4_matvec.zig").op_name_tile8, &mxfp4, &f32_dtypes, &row_major),
    same(@import("residual_add_scaled_rows_strided.zig").op_name, &f32_dtypes, &row_major_and_strided),
    same(@import("residual_scaled_rmsnorm_rows_strided.zig").op_name, &f32_dtypes, &row_major_and_strided),
    same(@import("rmsnorm.zig").op_name, &f32_dtypes, &row_major),
    same(@import("rmsnorm_rows_strided.zig").op_name, &f32_dtypes, &row_major_and_strided),
    same(@import("rope.zig").op_name, &f32_dtypes, &row_major),
    same(@import("rope_rows_ptrs.zig").op_name, &f32_dtypes, &row_major_and_strided),
    primitive(@import("rope_store_f16.zig").op_name, &f32_dtypes, &f16_dtypes, &row_major),
    primitive(@import("rope_store_fp8.zig").op_name, &f32_dtypes, &fp8_dtypes, &row_major),
    primitive(@import("rope_store_i8.zig").op_name, &f32_dtypes, &i8_dtypes, &row_major),
    same(@import("shortconv_step.zig").op_name, &f32_dtypes, &row_major),
    same(@import("silu.zig").op_name, &f32_dtypes, &row_major),
    same(@import("silu_mul.zig").op_name, &f32_dtypes, &row_major),
    same(@import("softmax_rows.zig").op_name, &f32_dtypes, &row_major),
    same(@import("softmax_rows_dynamic_cols_ptrs.zig").op_name, &f32_dtypes, &row_major_and_strided),
    primitive(@import("topk_rows_f32.zig").phase1_op_name, &f32_dtypes, &f32_or_u32_dtypes, &row_major),
    primitive(@import("topk_rows_f32.zig").phase2_op_name, &f32_dtypes, &f32_or_u32_dtypes, &row_major),
    same(@import("vector_add.zig").op_name, &f32_dtypes, &row_major),
    same(@import("vector_add_rows_strided.zig").op_name, &f32_dtypes, &row_major_and_strided),
    same(@import("vector_add_scaled.zig").op_name, &f32_dtypes, &row_major),
    same(@import("vector_add_scaled_rows_strided.zig").op_name, &f32_dtypes, &row_major_and_strided),
};

pub fn primitiveCapabilities(comptime backend: cap.Backend) [primitive_descriptors.len]cap.PrimitiveCapability {
    var result: [primitive_descriptors.len]cap.PrimitiveCapability = undefined;
    for (primitive_descriptors, 0..) |descriptor, idx| {
        result[idx] = .{
            .backend = backend,
            .name = descriptor.name,
            .input_dtypes = descriptor.input_dtypes,
            .output_dtypes = descriptor.output_dtypes,
            .layouts = descriptor.layouts,
            .rank_limits = descriptor.rank_limits,
            .required_alignment = descriptor.required_alignment,
        };
    }
    return result;
}

pub fn descriptorCount(name: []const u8) usize {
    var count: usize = 0;
    for (primitive_descriptors) |descriptor| {
        if (std.mem.eql(u8, descriptor.name, name)) count += 1;
    }
    return count;
}

test "compute CUDA descriptors declare implementation-backed primitive names once" {
    for (primitive_descriptors, 0..) |descriptor, idx| {
        for (primitive_descriptors[idx + 1 ..]) |candidate| {
            try std.testing.expect(!std.mem.eql(u8, descriptor.name, candidate.name));
        }
    }
}

test "compute CUDA primitiveCapabilities derives capability facts from descriptors" {
    const capabilities = primitiveCapabilities(.cuda);
    try std.testing.expectEqual(primitive_descriptors.len, capabilities.len);
    try std.testing.expectEqual(cap.Backend.cuda, capabilities[0].backend);
    try std.testing.expectEqualSlices(u8, primitive_descriptors[0].name, capabilities[0].name);
    try std.testing.expectEqualSlices(DType, primitive_descriptors[0].input_dtypes, capabilities[0].input_dtypes);
    try std.testing.expectEqualSlices(DType, primitive_descriptors[0].output_dtypes, capabilities[0].output_dtypes);
}

test "compute CUDA descriptorCount counts descriptor names" {
    try std.testing.expectEqual(@as(usize, 1), descriptorCount(primitive_descriptors[0].name));
    try std.testing.expectEqual(@as(usize, 0), descriptorCount("missing"));
}
