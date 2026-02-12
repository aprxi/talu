//! Integration tests for xray.ExecutionPlan
//!
//! ExecutionPlan is produced by xray.execution_plan.analyze() and describes
//! which kernels will be used for a model. It provides format() for display
//! and summary() for compact logging.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const ExecutionPlan = xray.ExecutionPlan;
const ExecutionPlanConfig = xray.ExecutionPlanConfig;
const MatmulKernel = xray.MatmulKernel;
const AttentionType = xray.AttentionType;
const FfnType = xray.FfnType;

const analyze = xray.execution_plan.analyze;

fn buildGqaPlan() ExecutionPlan {
    return analyze(.{
        .model_type = "qwen3",
        .num_layers = 28,
        .hidden_size = 1024,
        .num_heads = 16,
        .num_kv_heads = 4,
        .head_dim = 64,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_method = .gaffine,
    });
}

fn buildMoePlan() ExecutionPlan {
    return analyze(.{
        .model_type = "mixtral",
        .num_layers = 32,
        .hidden_size = 4096,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .num_experts = 8,
        .experts_per_token = 2,
    });
}

// ===== format =====

test "ExecutionPlan: format produces display with model info and kernel names" {
    const plan = buildGqaPlan();

    var buf: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try plan.format(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(output.len > 0);
    // Verify structural sections are present
    try std.testing.expect(std.mem.indexOf(u8, output, "EXECUTION PLAN") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "KERNEL SELECTION") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "QUANTIZATION") != null);
    // Verify actual kernel names appear
    try std.testing.expect(std.mem.indexOf(u8, output, "matmul_grouped_affine_u4") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "GroupedQueryAttention") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "SwiGLU(SiLU)") != null);
    // Verify model type appears
    try std.testing.expect(std.mem.indexOf(u8, output, "qwen3") != null);
}

test "ExecutionPlan: format includes MoE section for expert models" {
    const plan = buildMoePlan();

    var buf: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try plan.format(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "MIXTURE OF EXPERTS") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "MoE(SiLU)") != null);
}

test "ExecutionPlan: format omits MoE and quantization sections when not applicable" {
    const plan = analyze(.{
        .model_type = "bert",
        .num_layers = 12,
        .hidden_size = 768,
        .num_heads = 12,
        .num_kv_heads = 12,
        .head_dim = 64,
    });

    var buf: [4096]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    try plan.format(stream.writer());

    const output = stream.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, output, "MIXTURE OF EXPERTS") == null);
    try std.testing.expect(std.mem.indexOf(u8, output, "QUANTIZATION") == null);
}

// ===== summary =====

test "ExecutionPlan: summary contains kernel names separated by pipes" {
    const plan = buildGqaPlan();
    const s = plan.summary();

    const len = std.mem.indexOfScalar(u8, &s, 0) orelse s.len;
    const text = s[0..len];

    try std.testing.expect(len > 0);
    try std.testing.expect(len < 128);
    try std.testing.expect(std.mem.indexOf(u8, text, "matmul_grouped_affine_u4") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "GroupedQueryAttention") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "SwiGLU(SiLU)") != null);
    // Pipe-separated
    try std.testing.expect(std.mem.indexOf(u8, text, " | ") != null);
}

test "ExecutionPlan: summary is null-terminated" {
    const plan = buildMoePlan();
    const s = plan.summary();

    // Must have at least one null terminator within the 128-byte buffer
    const null_pos = std.mem.indexOfScalar(u8, &s, 0);
    try std.testing.expect(null_pos != null);
    try std.testing.expect(null_pos.? < 128);
}
