//! Compile-time/backward-compat checks for backend kernel/executor symmetry.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const backend = main.inference.backend;
const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

fn expectKernelModuleDecls(comptime K: type) !void {
    inline for ([_][]const u8{
        "attention",
        "describe_fmt",
        "embedding",
        "ffn",
        "fused_attention",
        "kv_cache",
        "mamba",
        "mla_attention",
        "moe",
        "norm",
        "rope",
        "shortconv",
        "weights",
    }) |module_name| {
        try std.testing.expect(@hasDecl(K, module_name));
    }
}

test "cpu executor exposes model and block modules" {
    _ = backend.cpu.executor.model;
    _ = backend.cpu.executor.block;
}

test "metal executor exposes model and block modules" {
    if (comptime !has_metal) return;
    _ = backend.metal.executor.model;
    _ = backend.metal.executor.block;
}

test "cuda executor exposes model and block modules" {
    if (comptime !has_cuda) return;
    _ = backend.cuda.executor.model;
    _ = backend.cuda.executor.block;
}

test "kernel module names stay symmetric" {
    try expectKernelModuleDecls(backend.cpu.kernels);
    if (comptime !has_metal) return;
    try expectKernelModuleDecls(backend.metal.kernels);
}

test "kernel module names stay symmetric for cuda" {
    if (comptime !has_cuda) return;
    try expectKernelModuleDecls(backend.cuda.kernels);
}

test "cpu kernel symbols expose symmetric struct names" {
    try std.testing.expect(@hasDecl(backend.cpu.kernels, "TransformerBlock"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.norm, "RMSNorm"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.attention, "MultiHeadAttention"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.ffn, "SwiGLU"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.norm.RMSNorm, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.attention.MultiHeadAttention, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.ffn.SwiGLU, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.norm.RMSNorm, "forward"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.attention.MultiHeadAttention, "forward"));
    try std.testing.expect(@hasDecl(backend.cpu.kernels.ffn.SwiGLU, "forward"));
}

test "kernel ForwardParams expose shared field names" {
    const cpu_norm_params = backend.cpu.kernels.norm.RMSNorm.ForwardParams;
    const cpu_ffn_params = backend.cpu.kernels.ffn.SwiGLU.ForwardParams;
    const cpu_attn_params = backend.cpu.kernels.attention.MultiHeadAttention.ForwardParams;
    try std.testing.expect(@hasField(cpu_norm_params, "input"));
    try std.testing.expect(@hasField(cpu_norm_params, "output"));
    try std.testing.expect(@hasField(cpu_ffn_params, "input_tensor"));
    try std.testing.expect(@hasField(cpu_ffn_params, "output_tensor"));
    try std.testing.expect(@hasField(cpu_ffn_params, "scratch"));
    try std.testing.expect(@hasField(cpu_ffn_params, "matmul_scratch"));
    try std.testing.expect(@hasField(cpu_attn_params, "input_tensor"));
    try std.testing.expect(@hasField(cpu_attn_params, "output_tensor"));
    try std.testing.expect(@hasField(cpu_attn_params, "cache"));
    try std.testing.expect(@hasField(cpu_attn_params, "scratch"));
    try std.testing.expect(@hasField(cpu_attn_params, "matmul_scratch"));
    try std.testing.expect(@hasField(cpu_attn_params, "use_cache"));

    if (comptime !has_metal) return;
    const metal_norm_params = backend.metal.kernels.norm.RMSNorm.ForwardParams;
    const metal_ffn_params = backend.metal.kernels.ffn.SwiGLU.ForwardParams;
    const metal_attn_params = backend.metal.kernels.attention.MultiHeadAttention.ForwardParams;
    try std.testing.expect(@hasField(metal_norm_params, "input"));
    try std.testing.expect(@hasField(metal_norm_params, "output"));
    try std.testing.expect(@hasField(metal_ffn_params, "input_tensor"));
    try std.testing.expect(@hasField(metal_ffn_params, "output_tensor"));
    try std.testing.expect(@hasField(metal_ffn_params, "scratch"));
    try std.testing.expect(@hasField(metal_ffn_params, "matmul_scratch"));
    try std.testing.expect(@hasField(metal_attn_params, "input_tensor"));
    try std.testing.expect(@hasField(metal_attn_params, "output_tensor"));
    try std.testing.expect(@hasField(metal_attn_params, "cache"));
    try std.testing.expect(@hasField(metal_attn_params, "scratch"));
    try std.testing.expect(@hasField(metal_attn_params, "matmul_scratch"));
    try std.testing.expect(@hasField(metal_attn_params, "use_cache"));
}

fn forwardArity(comptime T: type) usize {
    return @typeInfo(@TypeOf(@field(T, "forward"))).@"fn".params.len;
}

test "kernel forward arity matches across cpu and metal" {
    if (comptime !has_metal) return;
    try std.testing.expectEqual(@as(usize, 3), forwardArity(backend.cpu.kernels.norm.RMSNorm));
    try std.testing.expectEqual(@as(usize, 3), forwardArity(backend.metal.kernels.norm.RMSNorm));
    try std.testing.expectEqual(@as(usize, 5), forwardArity(backend.cpu.kernels.ffn.SwiGLU));
    try std.testing.expectEqual(@as(usize, 5), forwardArity(backend.metal.kernels.ffn.SwiGLU));
    try std.testing.expectEqual(@as(usize, 7), forwardArity(backend.cpu.kernels.attention.MultiHeadAttention));
    try std.testing.expectEqual(@as(usize, 7), forwardArity(backend.metal.kernels.attention.MultiHeadAttention));
}

test "kernel forward arity matches across cpu and cuda" {
    if (comptime !has_cuda) return;
    try std.testing.expectEqual(@as(usize, 3), forwardArity(backend.cpu.kernels.norm.RMSNorm));
    try std.testing.expectEqual(@as(usize, 3), forwardArity(backend.cuda.kernels.norm.RMSNorm));
    try std.testing.expectEqual(@as(usize, 5), forwardArity(backend.cpu.kernels.ffn.SwiGLU));
    try std.testing.expectEqual(@as(usize, 5), forwardArity(backend.cuda.kernels.ffn.SwiGLU));
    try std.testing.expectEqual(@as(usize, 7), forwardArity(backend.cpu.kernels.attention.MultiHeadAttention));
    try std.testing.expectEqual(@as(usize, 7), forwardArity(backend.cuda.kernels.attention.MultiHeadAttention));
}

test "metal kernel symbols expose symmetric struct names" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(backend.metal.kernels, "TransformerBlock"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.norm, "RMSNorm"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.attention, "MultiHeadAttention"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.ffn, "SwiGLU"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.norm.RMSNorm, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.attention.MultiHeadAttention, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.ffn.SwiGLU, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.norm.RMSNorm, "forward"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.attention.MultiHeadAttention, "forward"));
    try std.testing.expect(@hasDecl(backend.metal.kernels.ffn.SwiGLU, "forward"));
}

test "metal unsupported kernel declarations stay explicit" {
    if (comptime !has_metal) return;
    try std.testing.expect(backend.metal.kernels.mamba.supported);
    try std.testing.expect(!backend.metal.kernels.mla_attention.supported);
    try std.testing.expectError(error.MLANotSupportedOnMetal, backend.metal.kernels.mla_attention.unsupported());
}

test "cuda kernel symbols expose symmetric struct names" {
    if (comptime !has_cuda) return;
    try std.testing.expect(@hasDecl(backend.cuda.kernels, "TransformerBlock"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.norm, "RMSNorm"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.attention, "MultiHeadAttention"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.ffn, "SwiGLU"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.norm.RMSNorm, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.attention.MultiHeadAttention, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.ffn.SwiGLU, "ForwardParams"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.norm.RMSNorm, "forward"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.attention.MultiHeadAttention, "forward"));
    try std.testing.expect(@hasDecl(backend.cuda.kernels.ffn.SwiGLU, "forward"));
}
