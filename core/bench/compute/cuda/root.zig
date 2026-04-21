const std = @import("std");
const harness = @import("harness.zig");
const metrics = @import("metrics.zig");
const scenarios = @import("scenarios.zig");

const CliConfig = struct {
    scenario: scenarios.Scenario = .decode_attn_q,
    quant: scenarios.QuantKind = .nvfp4,
    run: scenarios.RunConfig = .{},
};

fn parseScenario(value: []const u8) !scenarios.Scenario {
    if (std.mem.eql(u8, value, "decode_gdelta_ssm_i8")) return .decode_gdelta_ssm_i8;
    if (std.mem.eql(u8, value, "decode_gdelta_in_proj")) return .decode_gdelta_in_proj;
    if (std.mem.eql(u8, value, "decode_gdelta_out_proj")) return .decode_gdelta_out_proj;
    if (std.mem.eql(u8, value, "decode_gdelta_conv_silu_ptrs")) return .decode_gdelta_conv_silu_ptrs;
    if (std.mem.eql(u8, value, "decode_gdelta_norm_gate_rows")) return .decode_gdelta_norm_gate_rows;
    if (std.mem.eql(u8, value, "decode_attn_rope")) return .decode_attn_rope;
    if (std.mem.eql(u8, value, "decode_attn_scores")) return .decode_attn_scores;
    if (std.mem.eql(u8, value, "decode_attn_softmax")) return .decode_attn_softmax;
    if (std.mem.eql(u8, value, "decode_attn_weighted_sum")) return .decode_attn_weighted_sum;
    if (std.mem.eql(u8, value, "decode_attn_q")) return .decode_attn_q;
    if (std.mem.eql(u8, value, "decode_attn_out")) return .decode_attn_out;
    if (std.mem.eql(u8, value, "decode_attn_i8_kv_ptrs")) return .decode_attn_i8_kv_ptrs;
    if (std.mem.eql(u8, value, "decode_attn_i8_kv_fused")) return .decode_attn_i8_kv_fused;
    if (std.mem.eql(u8, value, "decode_attn_i8_flash")) return .decode_attn_i8_flash;
    if (std.mem.eql(u8, value, "decode_attn_qkv_fused")) return .decode_attn_qkv_fused;
    if (std.mem.eql(u8, value, "decode_token_chain")) return .decode_token_chain;
    if (std.mem.eql(u8, value, "decode_ffn_gate")) return .decode_ffn_gate;
    if (std.mem.eql(u8, value, "decode_ffn_down")) return .decode_ffn_down;
    if (std.mem.eql(u8, value, "decode_ffn_gate_up_fused_silu")) return .decode_ffn_gate_up_fused_silu;
    if (std.mem.eql(u8, value, "decode_lm_head")) return .decode_lm_head;
    if (std.mem.eql(u8, value, "prefill_attn_q")) return .prefill_attn_q;
    if (std.mem.eql(u8, value, "prefill_attn_qkv_fused")) return .prefill_attn_qkv_fused;
    if (std.mem.eql(u8, value, "prefill_ffn_gate")) return .prefill_ffn_gate;
    if (std.mem.eql(u8, value, "prefill_ffn_gate_up_fused_silu")) return .prefill_ffn_gate_up_fused_silu;
    if (std.mem.eql(u8, value, "prefill_ffn_down")) return .prefill_ffn_down;
    return error.InvalidArgument;
}

fn parseQuant(value: []const u8) !scenarios.QuantKind {
    if (std.mem.eql(u8, value, "tq4")) return .tq4;
    if (std.mem.eql(u8, value, "nvfp4")) return .nvfp4;
    if (std.mem.eql(u8, value, "nvfp4_i8cache")) return .nvfp4_i8cache;
    if (std.mem.eql(u8, value, "nvfp4_native")) return .nvfp4_native;
    return error.InvalidArgument;
}

fn parseUsize(value: []const u8) !usize {
    return std.fmt.parseUnsigned(usize, value, 10);
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage:
        \\  zig build bench-cuda-compute -Drelease -- [options]
        \\
        \\Options:
        \\  --scenario <decode_gdelta_ssm_i8|decode_gdelta_in_proj|decode_gdelta_out_proj|decode_gdelta_conv_silu_ptrs|decode_gdelta_norm_gate_rows|decode_attn_rope|decode_attn_scores|decode_attn_softmax|decode_attn_weighted_sum|decode_attn_q|decode_attn_out|decode_attn_i8_kv_ptrs|decode_attn_i8_kv_fused|decode_attn_i8_flash|decode_attn_qkv_fused|decode_token_chain|decode_ffn_gate|decode_ffn_down|decode_ffn_gate_up_fused_silu|decode_lm_head|prefill_attn_q|prefill_attn_qkv_fused|prefill_ffn_gate|prefill_ffn_gate_up_fused_silu|prefill_ffn_down>
        \\  --quant <tq4|nvfp4|nvfp4_i8cache|nvfp4_native>
        \\  --warmup <N>                  default: 10
        \\  --iters <N>                   default: 30
        \\  --model <architecture_id>     default: qwen3_5
        \\  --rows <N>                    override scenario row/token count
        \\  --seq-len <N>                 override decode attention sequence length
        \\  --layers <N>                  override decode token chain layer count
        \\  --help
        \\
    );
}

fn parseArgs(args: [][:0]u8) !CliConfig {
    var cfg: CliConfig = .{};
    var idx: usize = 1;
    while (idx < args.len) : (idx += 1) {
        const arg = args[idx];
        if (std.mem.eql(u8, arg, "--help")) {
            return error.HelpRequested;
        } else if (std.mem.eql(u8, arg, "--scenario")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.scenario = try parseScenario(args[idx]);
        } else if (std.mem.eql(u8, arg, "--quant")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.quant = try parseQuant(args[idx]);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.warmup = try parseUsize(args[idx]);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.iters = try parseUsize(args[idx]);
        } else if (std.mem.eql(u8, arg, "--model")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.model_id = args[idx];
        } else if (std.mem.eql(u8, arg, "--rows")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.rows_override = try parseUsize(args[idx]);
        } else if (std.mem.eql(u8, arg, "--seq-len")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.seq_len_override = try parseUsize(args[idx]);
        } else if (std.mem.eql(u8, arg, "--layers")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.layers_override = try parseUsize(args[idx]);
        } else {
            return error.InvalidArgument;
        }
    }
    if (cfg.run.iters == 0) return error.InvalidArgument;
    return cfg;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const cfg = parseArgs(args) catch |err| switch (err) {
        error.HelpRequested => {
            try printUsage(std.fs.File.stdout().deprecatedWriter());
            return;
        },
        else => {
            try printUsage(std.fs.File.stderr().deprecatedWriter());
            return err;
        },
    };

    var result = try scenarios.runScenario(allocator, cfg.scenario, cfg.quant, cfg.run);
    defer result.deinit(allocator);
    const summary = try harness.summarizeSamples(allocator, result.samples);

    const gbps = metrics.gbps(result.bytes_per_iter, summary.p50_ns);
    const tflops = metrics.tflops(result.flops_per_iter, summary.p50_ns);
    const writer = std.fs.File.stdout().deprecatedWriter();
    try writer.print(
        "scenario={s} quant={s} model={s} rows={} in_dim={} out_dim={} warmup={} iters={} p50_us={d:.3} p90_us={d:.3} gbps={d:.3} tflops={d:.6}\n",
        .{
            result.name,
            @tagName(result.quant),
            cfg.run.model_id,
            result.rows,
            result.in_dim,
            result.out_dim,
            cfg.run.warmup,
            cfg.run.iters,
            @as(f64, @floatFromInt(summary.p50_ns)) / 1000.0,
            @as(f64, @floatFromInt(summary.p90_ns)) / 1000.0,
            gbps,
            tflops,
        },
    );
}

test "parseScenario accepts known scenarios" {
    try std.testing.expectEqual(scenarios.Scenario.decode_gdelta_ssm_i8, try parseScenario("decode_gdelta_ssm_i8"));
    try std.testing.expectEqual(scenarios.Scenario.decode_gdelta_in_proj, try parseScenario("decode_gdelta_in_proj"));
    try std.testing.expectEqual(scenarios.Scenario.decode_gdelta_out_proj, try parseScenario("decode_gdelta_out_proj"));
    try std.testing.expectEqual(scenarios.Scenario.decode_gdelta_conv_silu_ptrs, try parseScenario("decode_gdelta_conv_silu_ptrs"));
    try std.testing.expectEqual(scenarios.Scenario.decode_gdelta_norm_gate_rows, try parseScenario("decode_gdelta_norm_gate_rows"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_rope, try parseScenario("decode_attn_rope"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_scores, try parseScenario("decode_attn_scores"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_softmax, try parseScenario("decode_attn_softmax"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_weighted_sum, try parseScenario("decode_attn_weighted_sum"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_q, try parseScenario("decode_attn_q"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_out, try parseScenario("decode_attn_out"));
    try std.testing.expectEqual(scenarios.Scenario.prefill_ffn_down, try parseScenario("prefill_ffn_down"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_i8_kv_ptrs, try parseScenario("decode_attn_i8_kv_ptrs"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_i8_kv_fused, try parseScenario("decode_attn_i8_kv_fused"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_i8_flash, try parseScenario("decode_attn_i8_flash"));
    try std.testing.expectEqual(scenarios.Scenario.decode_attn_qkv_fused, try parseScenario("decode_attn_qkv_fused"));
    try std.testing.expectEqual(scenarios.Scenario.decode_token_chain, try parseScenario("decode_token_chain"));
    try std.testing.expectEqual(scenarios.Scenario.decode_ffn_down, try parseScenario("decode_ffn_down"));
}
