//! Static CUDA primitive, copy, and cast capability facts.

const std = @import("std");
const cap = @import("../capability.zig");
const DType = @import("../dtype.zig").DType;
const Layout = @import("../tensor_desc.zig").Layout;

const backend = cap.Backend.cuda;

const dense_dtypes = [_]DType{ .f32, .f64, .f16, .bf16, .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .f8_e4m3 };
const f32_dtypes = [_]DType{.f32};
const f32_f16_dtypes = [_]DType{ .f32, .f16 };
const f32_bf16_dtypes = [_]DType{ .f32, .bf16 };
const f32_i8_dtypes = [_]DType{ .f32, .i8 };
const f32_fp8_dtypes = [_]DType{ .f32, .f8_e4m3 };
const f32_u16_dtypes = [_]DType{ .f32, .u16 };
const f32_u32_dtypes = [_]DType{ .f32, .u32 };
const f32_gaffine_u4_dtypes = [_]DType{ .f32, .grouped_affine_u4 };
const f32_gaffine_u8_dtypes = [_]DType{ .f32, .grouped_affine_u8 };
const f32_mxfp4_dtypes = [_]DType{ .f32, .mxfp4 };
const f16_i8_dtypes = [_]DType{ .f16, .i8 };
const f16_fp8_dtypes = [_]DType{ .f16, .f8_e4m3 };
const f16_gaffine_u4_dtypes = [_]DType{ .f16, .grouped_affine_u4 };
const f16_gaffine_u8_dtypes = [_]DType{ .f16, .grouped_affine_u8 };
const i8_i32_dtypes = [_]DType{ .i8, .i32 };
const u16_dtypes = [_]DType{.u16};
const u32_dtypes = [_]DType{.u32};
const row_major_layouts = [_]Layout{.row_major_contiguous};
const row_major_and_strided_layouts = [_]Layout{ .row_major_contiguous, .strided };

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
    primitive("cublas_matmul_f32", &f32_dtypes, &row_major_layouts),
    primitive("cublas_matmul_u16_f32", &f32_u16_dtypes, &row_major_layouts),
    primitive("cublas_matmul_u16_u16_f32", &f32_u16_dtypes, &row_major_layouts),
    primitive("cublas_matmul_i8_i8_i32", &i8_i32_dtypes, &row_major_layouts),
    primitive("cublas_matmul_fp8_fp8_f32", &f32_fp8_dtypes, &row_major_layouts),
    primitive("cublas_gemm_u16_strided_batched", &f32_u16_dtypes, &row_major_and_strided_layouts),
    primitive("cublas_gemm_u16", &f32_u16_dtypes, &row_major_layouts),
    primitive("cublaslt_matmul_mxfp8", &f32_fp8_dtypes, &row_major_layouts),
    primitive("cublaslt_matmul_nvfp4", &f32_mxfp4_dtypes, &row_major_layouts),

    primitive("argmax_f32", &f32_dtypes, &row_major_layouts),
    primitive("attn_fused_decode_heads_f16_kv_ptrs", &f32_f16_dtypes, &row_major_and_strided_layouts),
    primitive("attn_fused_decode_heads_fp8_kv_ptrs", &f32_fp8_dtypes, &row_major_and_strided_layouts),
    primitive("attn_fused_decode_heads_i8_kv_ptrs", &f32_i8_dtypes, &row_major_and_strided_layouts),
    primitive("attn_fused_heads_f16_kv", &f32_f16_dtypes, &row_major_layouts),
    primitive("attn_fused_heads_fp8_kv", &f32_fp8_dtypes, &row_major_layouts),
    primitive("attn_fused_heads_i8_kv", &f32_i8_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_f16_kv", &f32_f16_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_f16_kv_gqa", &f32_f16_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_fp8_kv", &f32_fp8_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_fp8_kv_gqa", &f32_fp8_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_i8_kv", &f32_i8_dtypes, &row_major_layouts),
    primitive("attn_fused_prefill_heads_i8_kv_gqa", &f32_i8_dtypes, &row_major_layouts),
    primitive("attn_scores_heads_f16_kv", &f32_f16_dtypes, &row_major_layouts),
    primitive("attn_scores_heads_f16_kv_ptrs", &f32_f16_dtypes, &row_major_and_strided_layouts),
    primitive("attn_scores_heads_f32", &f32_dtypes, &row_major_layouts),
    primitive("attn_scores_heads_fp8_kv", &f32_fp8_dtypes, &row_major_layouts),
    primitive("attn_scores_heads_fp8_kv_ptrs", &f32_fp8_dtypes, &row_major_and_strided_layouts),
    primitive("attn_scores_heads_i8_kv", &f32_i8_dtypes, &row_major_layouts),
    primitive("attn_scores_heads_i8_kv_ptrs", &f32_i8_dtypes, &row_major_and_strided_layouts),
    primitive("attn_weighted_sum_heads_f16_kv", &f32_f16_dtypes, &row_major_layouts),
    primitive("attn_weighted_sum_heads_f16_kv_ptrs", &f32_f16_dtypes, &row_major_and_strided_layouts),
    primitive("attn_weighted_sum_heads_f32", &f32_dtypes, &row_major_layouts),
    primitive("attn_weighted_sum_heads_fp8_kv", &f32_fp8_dtypes, &row_major_layouts),
    primitive("attn_weighted_sum_heads_fp8_kv_ptrs", &f32_fp8_dtypes, &row_major_and_strided_layouts),
    primitive("attn_weighted_sum_heads_i8_kv", &f32_i8_dtypes, &row_major_layouts),
    primitive("attn_weighted_sum_heads_i8_kv_ptrs", &f32_i8_dtypes, &row_major_and_strided_layouts),
    primitive("cast_bf16_to_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("cast_f32_to_bf16", &f32_bf16_dtypes, &row_major_layouts),
    primitive("cast_f32_to_f16", &f32_f16_dtypes, &row_major_layouts),
    primitive("causal_attn_softmax_f32", &f32_dtypes, &row_major_layouts),
    primitive("copy_f32", &f32_dtypes, &row_major_layouts),
    primitive("copy_u16", &u16_dtypes, &row_major_layouts),
    primitive("decode_u32_increment", &u32_dtypes, &row_major_layouts),
    primitive("dequant_kv_fp8_to_f16", &f16_fp8_dtypes, &row_major_layouts),
    primitive("dequant_kv_i8_to_f16", &f16_i8_dtypes, &row_major_layouts),
    primitive("embedding_lookup_f32", &f32_dtypes, &row_major_layouts),
    primitive("embedding_lookup_gaffine_u4_f32", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("embedding_lookup_u16_f32", &f32_u16_dtypes, &row_major_layouts),
    primitive("embedding_lookup_u16_rows_f32", &f32_u16_dtypes, &row_major_layouts),
    primitive("flash_decode_f16", &f32_f16_dtypes, &row_major_layouts),
    primitive("flash_decode_i8", &f32_i8_dtypes, &row_major_layouts),
    primitive("flash_decode_fp8", &f32_fp8_dtypes, &row_major_layouts),
    primitive("flash_decode_reduce", &f32_dtypes, &row_major_layouts),
    primitive("flash_prefill_f16", &f32_f16_dtypes, &row_major_layouts),
    primitive("flash_prefill_i8", &f32_i8_dtypes, &row_major_layouts),
    primitive("flash_prefill_fp8", &f32_fp8_dtypes, &row_major_layouts),
    primitive("gaffine_u4_dequantize_to_f16", &f16_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_f32", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_f32_tile8", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_gate_up_f32", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_gate_up_silu_f32", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_gate_up_silu_f32_tile8", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_qkv_f32", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u4_matvec_qkv_f32_tile8", &f32_gaffine_u4_dtypes, &row_major_layouts),
    primitive("gaffine_u8_dequantize_to_f16", &f16_gaffine_u8_dtypes, &row_major_layouts),
    primitive("gaffine_u8_matvec_f32", &f32_gaffine_u8_dtypes, &row_major_layouts),
    primitive("gaffine_u8_matvec_gate_up_f32", &f32_gaffine_u8_dtypes, &row_major_layouts),
    primitive("gaffine_u8_matvec_gate_up_silu_f32", &f32_gaffine_u8_dtypes, &row_major_layouts),
    primitive("gaffine_u8_matvec_qkv_f32", &f32_gaffine_u8_dtypes, &row_major_layouts),
    primitive("gated_attention_compact_q_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_attention_output_gate_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_conv_values_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_conv_silu_values_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_conv_silu_values_rows_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_conv_silu_values_rows_ptrs_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("gated_delta_advance_ring_heads_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("gated_delta_qk_norm_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_rmsnorm_silu_mul_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_rmsnorm_silu_mul_rows_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_ssm_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_ssm_rows_f32", &f32_dtypes, &row_major_layouts),
    primitive("gated_delta_ssm_rows_i8_f32", &f32_i8_dtypes, &row_major_layouts),
    primitive("gated_delta_ssm_rows_ptrs_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("gated_delta_ssm_rows_ptrs_i8_f32", &f32_i8_dtypes, &row_major_and_strided_layouts),
    primitive("gelu_mul_f32", &f32_dtypes, &row_major_layouts),
    primitive("i8_matvec_gate_up_silu_f32", &f32_i8_dtypes, &row_major_layouts),
    primitive("i8_matvec_gate_up_silu_f32_tile8", &f32_i8_dtypes, &row_major_layouts),
    primitive("kv_write_f16", &f32_f16_dtypes, &row_major_layouts),
    primitive("kv_write_f16_rows", &f32_f16_dtypes, &row_major_layouts),
    primitive("kv_write_f16_rows_ptrs", &f32_f16_dtypes, &row_major_and_strided_layouts),
    primitive("kv_write_fp8", &f32_fp8_dtypes, &row_major_layouts),
    primitive("kv_write_fp8_rows", &f32_fp8_dtypes, &row_major_layouts),
    primitive("kv_write_fp8_rows_ptrs", &f32_fp8_dtypes, &row_major_and_strided_layouts),
    primitive("kv_write_i8", &f32_i8_dtypes, &row_major_layouts),
    primitive("kv_write_i8_rows", &f32_i8_dtypes, &row_major_layouts),
    primitive("kv_write_i8_rows_ptrs", &f32_i8_dtypes, &row_major_and_strided_layouts),
    primitive("matmul_f16_f32", &f32_f16_dtypes, &row_major_layouts),
    primitive("matmul_bf16_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("matvec_f16_f32", &f32_f16_dtypes, &row_major_layouts),
    primitive("matvec_bf16_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("matvec_gate_up_f16_f32", &f32_f16_dtypes, &row_major_layouts),
    primitive("matvec_gate_up_bf16_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("matvec_gate_up_silu_f16_f32", &f32_f16_dtypes, &row_major_layouts),
    primitive("matvec_gate_up_silu_bf16_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("matvec_qkv_f16_f32", &f32_f16_dtypes, &row_major_layouts),
    primitive("matvec_qkv_bf16_f32", &f32_bf16_dtypes, &row_major_layouts),
    primitive("mul_f32", &f32_dtypes, &row_major_layouts),
    primitive("nvfp4_matvec_f32", &f32_mxfp4_dtypes, &row_major_layouts),
    primitive("nvfp4_matvec_f32_tile8", &f32_mxfp4_dtypes, &row_major_layouts),
    primitive("residual_add_scaled_rows_strided_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("residual_scaled_rmsnorm_rows_strided_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("rmsnorm_f32", &f32_dtypes, &row_major_layouts),
    primitive("rmsnorm_rows_strided_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("rope_f32", &f32_dtypes, &row_major_layouts),
    primitive("rope_rows_ptrs", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("rope_store_f16", &f32_f16_dtypes, &row_major_layouts),
    primitive("rope_store_fp8", &f32_fp8_dtypes, &row_major_layouts),
    primitive("rope_store_i8", &f32_i8_dtypes, &row_major_layouts),
    primitive("shortconv_step_f32", &f32_dtypes, &row_major_layouts),
    primitive("silu_f32", &f32_dtypes, &row_major_layouts),
    primitive("silu_mul_f32", &f32_dtypes, &row_major_layouts),
    primitive("softmax_rows_f32", &f32_dtypes, &row_major_layouts),
    primitive("softmax_rows_dynamic_cols_ptrs", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("topk_rows_phase1", &f32_u32_dtypes, &row_major_layouts),
    primitive("topk_rows_phase2", &f32_u32_dtypes, &row_major_layouts),
    primitive("vector_add_f32", &f32_dtypes, &row_major_layouts),
    primitive("vector_add_rows_strided_f32", &f32_dtypes, &row_major_and_strided_layouts),
    primitive("vector_add_scaled_f32", &f32_dtypes, &row_major_layouts),
    primitive("vector_add_scaled_rows_strided_f32", &f32_dtypes, &row_major_and_strided_layouts),
};

pub const copy_capabilities = [_]cap.CopyCapability{
    .{
        .backend = backend,
        .direction = .host_to_device,
        .dtypes = &dense_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .device_to_host,
        .dtypes = &dense_dtypes,
        .layouts = &row_major_layouts,
        .required_alignment = 1,
    },
    .{
        .backend = backend,
        .direction = .device_to_device,
        .dtypes = &dense_dtypes,
        .layouts = &row_major_layouts,
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
    linalg: bool = supportsPrimitive("cublas_matmul_f32", .f32, .row_major_contiguous),
    normalization: bool = supportsPrimitive("rmsnorm_f32", .f32, .row_major_contiguous),
    activation: bool = supportsPrimitive("silu_f32", .f32, .row_major_contiguous),
    softmax: bool = supportsPrimitive("softmax_rows_f32", .f32, .row_major_contiguous),
    layout: bool = supportsPrimitive("vector_add_rows_strided_f32", .f32, .strided),
    memory: bool = supportsPrimitive("copy_f32", .f32, .row_major_contiguous),
    indexing: bool = supportsPrimitive("embedding_lookup_f32", .f32, .row_major_contiguous),
    quant_decode: bool = supportsPrimitive("dequant_kv_i8_to_f16", .f16, .row_major_contiguous),
    state_space: bool = supportsPrimitive("gated_delta_ssm_f32", .f32, .row_major_contiguous),
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

fn primitiveEntryCount(name: []const u8) usize {
    var count: usize = 0;
    for (primitive_capabilities) |entry| {
        if (entry.backend == backend and std.mem.eql(u8, entry.name, name)) count += 1;
    }
    return count;
}

const exported_op_names = [_][]const u8{
    "argmax_f32",
    "attn_fused_decode_heads_f16_kv_ptrs",
    "attn_fused_decode_heads_fp8_kv_ptrs",
    "attn_fused_decode_heads_i8_kv_ptrs",
    "attn_fused_heads_f16_kv",
    "attn_fused_heads_fp8_kv",
    "attn_fused_heads_i8_kv",
    "attn_fused_prefill_heads_f16_kv",
    "attn_fused_prefill_heads_f16_kv_gqa",
    "attn_fused_prefill_heads_fp8_kv",
    "attn_fused_prefill_heads_fp8_kv_gqa",
    "attn_fused_prefill_heads_i8_kv",
    "attn_fused_prefill_heads_i8_kv_gqa",
    "attn_scores_heads_f16_kv",
    "attn_scores_heads_f16_kv_ptrs",
    "attn_scores_heads_f32",
    "attn_scores_heads_fp8_kv",
    "attn_scores_heads_fp8_kv_ptrs",
    "attn_scores_heads_i8_kv",
    "attn_scores_heads_i8_kv_ptrs",
    "attn_weighted_sum_heads_f16_kv",
    "attn_weighted_sum_heads_f16_kv_ptrs",
    "attn_weighted_sum_heads_f32",
    "attn_weighted_sum_heads_fp8_kv",
    "attn_weighted_sum_heads_fp8_kv_ptrs",
    "attn_weighted_sum_heads_i8_kv",
    "attn_weighted_sum_heads_i8_kv_ptrs",
    "cast_bf16_to_f32",
    "cast_f32_to_bf16",
    "cast_f32_to_f16",
    "causal_attn_softmax_f32",
    "copy_f32",
    "copy_u16",
    "decode_u32_increment",
    "dequant_kv_fp8_to_f16",
    "dequant_kv_i8_to_f16",
    "embedding_lookup_f32",
    "embedding_lookup_gaffine_u4_f32",
    "embedding_lookup_u16_f32",
    "embedding_lookup_u16_rows_f32",
    "flash_decode_f16",
    "flash_decode_i8",
    "flash_decode_fp8",
    "flash_decode_reduce",
    "flash_prefill_f16",
    "flash_prefill_i8",
    "flash_prefill_fp8",
    "gaffine_u4_dequantize_to_f16",
    "gaffine_u4_matvec_f32",
    "gaffine_u4_matvec_f32_tile8",
    "gaffine_u4_matvec_gate_up_f32",
    "gaffine_u4_matvec_gate_up_silu_f32",
    "gaffine_u4_matvec_gate_up_silu_f32_tile8",
    "gaffine_u4_matvec_qkv_f32",
    "gaffine_u4_matvec_qkv_f32_tile8",
    "gaffine_u8_dequantize_to_f16",
    "gaffine_u8_matvec_f32",
    "gaffine_u8_matvec_gate_up_f32",
    "gaffine_u8_matvec_gate_up_silu_f32",
    "gaffine_u8_matvec_qkv_f32",
    "gated_attention_compact_q_f32",
    "gated_attention_output_gate_f32",
    "gated_delta_conv_values_f32",
    "gated_delta_conv_silu_values_f32",
    "gated_delta_conv_silu_values_rows_f32",
    "gated_delta_conv_silu_values_rows_ptrs_f32",
    "gated_delta_advance_ring_heads_f32",
    "gated_delta_qk_norm_f32",
    "gated_delta_rmsnorm_silu_mul_f32",
    "gated_delta_rmsnorm_silu_mul_rows_f32",
    "gated_delta_ssm_f32",
    "gated_delta_ssm_rows_f32",
    "gated_delta_ssm_rows_i8_f32",
    "gated_delta_ssm_rows_ptrs_f32",
    "gated_delta_ssm_rows_ptrs_i8_f32",
    "gelu_mul_f32",
    "i8_matvec_gate_up_silu_f32",
    "i8_matvec_gate_up_silu_f32_tile8",
    "kv_write_f16",
    "kv_write_f16_rows",
    "kv_write_f16_rows_ptrs",
    "kv_write_fp8",
    "kv_write_fp8_rows",
    "kv_write_fp8_rows_ptrs",
    "kv_write_i8",
    "kv_write_i8_rows",
    "kv_write_i8_rows_ptrs",
    "matmul_f16_f32",
    "matmul_bf16_f32",
    "matvec_f16_f32",
    "matvec_bf16_f32",
    "matvec_gate_up_f16_f32",
    "matvec_gate_up_bf16_f32",
    "matvec_gate_up_silu_f16_f32",
    "matvec_gate_up_silu_bf16_f32",
    "matvec_qkv_f16_f32",
    "matvec_qkv_bf16_f32",
    "mul_f32",
    "nvfp4_matvec_f32",
    "nvfp4_matvec_f32_tile8",
    "residual_add_scaled_rows_strided_f32",
    "residual_scaled_rmsnorm_rows_strided_f32",
    "rmsnorm_f32",
    "rmsnorm_rows_strided_f32",
    "rope_f32",
    "rope_rows_ptrs",
    "rope_store_f16",
    "rope_store_fp8",
    "rope_store_i8",
    "shortconv_step_f32",
    "silu_f32",
    "silu_mul_f32",
    "softmax_rows_f32",
    "softmax_rows_dynamic_cols_ptrs",
    "topk_rows_phase1",
    "topk_rows_phase2",
    "vector_add_f32",
    "vector_add_rows_strided_f32",
    "vector_add_scaled_f32",
    "vector_add_scaled_rows_strided_f32",
};

test "compute supportsPrimitive has exactly one entry for every exported CUDA op_name variant" {
    inline for (exported_op_names) |name| {
        try std.testing.expectEqual(@as(usize, 1), primitiveEntryCount(name));
    }
}

test "compute CUDA primitive_capabilities declare each primitive name once" {
    for (primitive_capabilities, 0..) |entry, entry_idx| {
        for (primitive_capabilities[entry_idx + 1 ..]) |candidate| {
            try std.testing.expect(!std.mem.eql(u8, entry.name, candidate.name));
        }
    }
}

test "compute supportsPrimitive accepts supported CUDA dtype and layout combinations" {
    try std.testing.expect(supportsPrimitive("vector_add_rows_strided_f32", .f32, .strided));
    try std.testing.expect(supportsPrimitive("matvec_bf16_f32", .bf16, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("attn_fused_heads_fp8_kv", .f8_e4m3, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("dequant_kv_i8_to_f16", .i8, .row_major_contiguous));
    try std.testing.expect(supportsPrimitive("dequant_kv_i8_to_f16", .f16, .row_major_contiguous));
}

test "compute supportsPrimitive fails closed for unsupported CUDA primitive combinations" {
    try std.testing.expect(!supportsPrimitive("missing", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("vector_add_f32", .f16, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("vector_add_f32", .f32, .strided));
    try std.testing.expect(!supportsPrimitive("dequant_kv_i8_to_f16", .f32, .row_major_contiguous));
    try std.testing.expect(!supportsPrimitive("cublas_matmul_i8_i8_i32", .f32, .row_major_contiguous));
}

test "compute CUDA copy_capabilities and cast_capabilities validate supported paths and fail closed" {
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
        .direction = .host_to_host,
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
