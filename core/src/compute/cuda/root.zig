//! CUDA GPU compute primitives.
//!
//! Current phase includes runtime probing, device/buffer lifecycle,
//! cuBLAS matmul, and modular kernel runtime scaffolding.

pub const capabilities = @import("capabilities.zig");
pub const device = @import("device.zig");
pub const matmul = @import("matmul.zig");
pub const args = @import("args.zig");
pub const module = @import("module.zig");
pub const launch = @import("launch.zig");
pub const manifest = @import("manifest.zig");
pub const sideload = @import("sideload.zig");
pub const registry = @import("registry.zig");
pub const vector_add = @import("vector_add.zig");
pub const mul = @import("mul.zig");
pub const copy = @import("copy.zig");
pub const copy_u16 = @import("copy_u16.zig");
pub const cast_f32_to_f16 = @import("cast_f32_to_f16.zig");
pub const kv_write_f16 = @import("kv_write_f16.zig");
pub const rmsnorm = @import("rmsnorm.zig");
pub const rope = @import("rope.zig");
pub const rope_store_f16 = @import("rope_store_f16.zig");
pub const attn_scores = @import("attn_scores.zig");
pub const attn_scores_f16_kv = @import("attn_scores_f16_kv.zig");
pub const attn_scores_heads_f16_kv = @import("attn_scores_heads_f16_kv.zig");
pub const attn_fused_heads_f16_kv = @import("attn_fused_heads_f16_kv.zig");
pub const softmax = @import("softmax.zig");
pub const softmax_rows = @import("softmax_rows.zig");
pub const attn_weighted_sum = @import("attn_weighted_sum.zig");
pub const attn_weighted_sum_f16_kv = @import("attn_weighted_sum_f16_kv.zig");
pub const attn_weighted_sum_heads_f16_kv = @import("attn_weighted_sum_heads_f16_kv.zig");
pub const silu = @import("silu.zig");
pub const silu_mul = @import("silu_mul.zig");
pub const gelu_mul = @import("gelu_mul.zig");
pub const argmax = @import("argmax.zig");
pub const matvec_u16 = @import("matvec_u16.zig");
pub const matvec_u16_gate_up = @import("matvec_u16_gate_up.zig");
pub const matvec_u16_qkv = @import("matvec_u16_qkv.zig");
pub const gaffine_u4_matvec = @import("gaffine_u4_matvec.zig");
pub const gaffine_u4_matvec_gate_up = @import("gaffine_u4_matvec_gate_up.zig");
pub const gaffine_u4_matvec_qkv = @import("gaffine_u4_matvec_qkv.zig");
pub const shortconv_step = @import("shortconv_step.zig");

pub const Device = device.Device;
pub const Buffer = device.Buffer;
pub const StreamHandle = device.StreamHandle;
pub const Probe = device.Probe;
pub const probeRuntime = device.probeRuntime;
pub const Blas = matmul.Blas;
pub const Module = module.Module;
pub const Function = module.Function;
pub const ArgPack = args.ArgPack;
pub const LaunchConfig = launch.LaunchConfig;
pub const Registry = registry.Registry;

test {
    _ = capabilities;
    _ = device;
    _ = matmul;
    _ = args;
    _ = module;
    _ = launch;
    _ = manifest;
    _ = sideload;
    _ = registry;
    _ = vector_add;
    _ = mul;
    _ = copy;
    _ = copy_u16;
    _ = cast_f32_to_f16;
    _ = kv_write_f16;
    _ = rmsnorm;
    _ = rope;
    _ = rope_store_f16;
    _ = attn_scores;
    _ = attn_scores_f16_kv;
    _ = attn_scores_heads_f16_kv;
    _ = attn_fused_heads_f16_kv;
    _ = softmax;
    _ = softmax_rows;
    _ = attn_weighted_sum;
    _ = attn_weighted_sum_f16_kv;
    _ = attn_weighted_sum_heads_f16_kv;
    _ = silu;
    _ = silu_mul;
    _ = gelu_mul;
    _ = argmax;
    _ = matvec_u16;
    _ = matvec_u16_gate_up;
    _ = matvec_u16_qkv;
    _ = gaffine_u4_matvec;
    _ = gaffine_u4_matvec_gate_up;
    _ = gaffine_u4_matvec_qkv;
    _ = shortconv_step;
}
