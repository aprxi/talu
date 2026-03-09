const std = @import("std");
const harness = @import("harness.zig");
const metrics = @import("metrics.zig");
const scenarios = @import("scenarios.zig");
const project = @import("main");
const models = project.models.dispatcher;

const CliConfig = struct {
    scenario: scenarios.Scenario = .all,
    run: scenarios.RunConfig = .{},
    format: OutputFormat = .table,
};

const OutputFormat = enum {
    table,
    csv,
    tsv,
};

const RowMetrics = struct {
    name: []const u8,
    gbps: f64,
    tflops: f64,
};

fn spreadPct(summary: harness.Summary) f64 {
    if (summary.p50_ns == 0) return 0.0;
    if (summary.p90_ns <= summary.p50_ns) return 0.0;
    return (@as(f64, @floatFromInt(summary.p90_ns - summary.p50_ns)) * 100.0) / @as(f64, @floatFromInt(summary.p50_ns));
}

const PeakStats = struct {
    gbps: f64 = 0.0,
    gbps_name: []const u8 = "-",
    tflops: f64 = 0.0,
    tflops_name: []const u8 = "-",
};

const cold_probe_repeats: usize = 5;

fn updatePeaks(peaks: *PeakStats, row: RowMetrics) void {
    if (row.gbps >= peaks.gbps) {
        peaks.gbps = row.gbps;
        peaks.gbps_name = row.name;
    }
    if (row.tflops >= peaks.tflops) {
        peaks.tflops = row.tflops;
        peaks.tflops_name = row.name;
    }
}

fn profileName(profile: scenarios.Profile) []const u8 {
    return switch (profile) {
        .ci => "ci",
        .bw => "bw",
    };
}

fn parseProfile(value: []const u8) !scenarios.Profile {
    if (std.mem.eql(u8, value, "ci")) return .ci;
    if (std.mem.eql(u8, value, "bw")) return .bw;
    return error.InvalidArgument;
}

fn parsePreset(value: []const u8) ![]const u8 {
    if (models.performanceHintsByName(value) != null) return value;
    if (std.mem.eql(u8, value, "qwen35") and models.performanceHintsByName("qwen3_5") != null) return "qwen3_5";
    return error.InvalidArgument;
}

fn parseScenario(value: []const u8) !scenarios.Scenario {
    if (std.mem.eql(u8, value, "all")) return .all;
    if (std.mem.eql(u8, value, "prefill_attn_q") or std.mem.eql(u8, value, "prefill.attn_q")) return .prefill_attn_q_bf16;
    if (std.mem.eql(u8, value, "prefill_attn_k") or std.mem.eql(u8, value, "prefill.attn_k")) return .prefill_attn_k_bf16;
    if (std.mem.eql(u8, value, "prefill_attn_v") or std.mem.eql(u8, value, "prefill.attn_v")) return .prefill_attn_v_bf16;
    if (std.mem.eql(u8, value, "prefill_attn_out") or std.mem.eql(u8, value, "prefill.attn_out")) return .prefill_attn_out_bf16;
    if (std.mem.eql(u8, value, "prefill_ffn_gate") or std.mem.eql(u8, value, "prefill.ffn_gate")) return .prefill_ffn_gate_bf16;
    if (std.mem.eql(u8, value, "prefill_ffn_down") or std.mem.eql(u8, value, "prefill.ffn_down")) return .prefill_ffn_down_bf16;
    if (std.mem.eql(u8, value, "decode_attn_q") or std.mem.eql(u8, value, "decode.attn_q")) return .decode_attn_q_bf16;
    if (std.mem.eql(u8, value, "decode_attn_k") or std.mem.eql(u8, value, "decode.attn_k")) return .decode_attn_k_bf16;
    if (std.mem.eql(u8, value, "decode_attn_v") or std.mem.eql(u8, value, "decode.attn_v")) return .decode_attn_v_bf16;
    if (std.mem.eql(u8, value, "decode_attn_out") or std.mem.eql(u8, value, "decode.attn_out")) return .decode_attn_out_bf16;
    if (std.mem.eql(u8, value, "decode_ffn_gate") or std.mem.eql(u8, value, "decode.ffn_gate")) return .decode_ffn_gate_bf16;
    if (std.mem.eql(u8, value, "decode_ffn_down") or std.mem.eql(u8, value, "decode.ffn_down")) return .decode_ffn_down_bf16;
    if (std.mem.eql(u8, value, "decode_lm_head") or std.mem.eql(u8, value, "decode.lm_head")) return .decode_lm_head_bf16;
    if (std.mem.eql(u8, value, "decode_lm_head_bf16") or std.mem.eql(u8, value, "decode.lm_head_bf16")) return .decode_lm_head_bf16;
    if (std.mem.eql(u8, value, "decode_lm_head_f16") or std.mem.eql(u8, value, "decode.lm_head_f16")) return .decode_lm_head_f16;
    if (std.mem.eql(u8, value, "decode_lm_head_f32") or std.mem.eql(u8, value, "decode.lm_head_f32")) return .decode_lm_head_f32;
    if (std.mem.eql(u8, value, "decode_lm_head_runtime_f32") or std.mem.eql(u8, value, "decode.lm_head_runtime_f32")) return .decode_lm_head_runtime_f32;
    if (std.mem.eql(u8, value, "prefill_layer_attn_norm") or std.mem.eql(u8, value, "prefill.layer_attn_norm")) return .prefill_layer_attn_norm_f32;
    if (std.mem.eql(u8, value, "prefill_layer_ffn_norm") or std.mem.eql(u8, value, "prefill.layer_ffn_norm")) return .prefill_layer_ffn_norm_f32;
    if (std.mem.eql(u8, value, "prefill_final_norm") or std.mem.eql(u8, value, "prefill.final_norm")) return .prefill_final_norm_f32;
    if (std.mem.eql(u8, value, "decode_layer_attn_norm") or std.mem.eql(u8, value, "decode.layer_attn_norm")) return .decode_layer_attn_norm_f32;
    if (std.mem.eql(u8, value, "decode_layer_ffn_norm") or std.mem.eql(u8, value, "decode.layer_ffn_norm")) return .decode_layer_ffn_norm_f32;
    if (std.mem.eql(u8, value, "decode_final_norm") or std.mem.eql(u8, value, "decode.final_norm")) return .decode_final_norm_f32;
    if (std.mem.eql(u8, value, "delta") or std.mem.eql(u8, value, "gdelta")) return .gated_delta_step_f32;
    if (std.mem.eql(u8, value, "gdnorm") or std.mem.eql(u8, value, "gdelta_norm")) return .gated_delta_norm_f32;
    if (std.mem.eql(u8, value, "gate") or std.mem.eql(u8, value, "gattn")) return .gated_attention_gate_f32;
    if (std.mem.eql(u8, value, "gate_long") or std.mem.eql(u8, value, "gattn_long")) return .gated_attention_gate_long_f32;
    if (std.mem.eql(u8, value, "rope_f32") or std.mem.eql(u8, value, "rope")) return .rope_f32;
    if (std.mem.eql(u8, value, "rope_f16")) return .rope_f16;
    if (std.mem.eql(u8, value, "rope_bf16")) return .rope_bf16;
    if (std.mem.eql(u8, value, "sdpa_f32") or std.mem.eql(u8, value, "sdpa")) return .sdpa_f32;
    if (std.mem.eql(u8, value, "sdpa_f16")) return .sdpa_f16;
    if (std.mem.eql(u8, value, "sdpa_bf16")) return .sdpa_bf16;
    if (std.mem.eql(u8, value, "rms") or std.mem.eql(u8, value, "rms_f32")) return .rmsnorm_f32;
    if (std.mem.eql(u8, value, "rms_bf16") or std.mem.eql(u8, value, "rms_wbf16")) return .rmsnorm_bf16_weight;
    if (std.mem.eql(u8, value, "rms_f16") or std.mem.eql(u8, value, "rms_wf16")) return .rmsnorm_f16_weight;
    if (std.mem.eql(u8, value, "smx") or std.mem.eql(u8, value, "softmax_f32")) return .softmax_f32;
    if (std.mem.eql(u8, value, "decode_bf16")) return .decode_bf16;
    if (std.mem.eql(u8, value, "decode_u4")) return .decode_u4;
    if (std.mem.eql(u8, value, "decode_u8")) return .decode_u8;
    if (std.mem.eql(u8, value, "scv") or std.mem.eql(u8, value, "shortconv_f32")) return .shortconv_decode_f32;
    if (std.mem.eql(u8, value, "ssm") or std.mem.eql(u8, value, "mamba_scan_f32")) return .mamba_scan_f32;
    if (std.mem.eql(u8, value, "mmthr") or std.mem.eql(u8, value, "matmul_thr_f32")) return .matmul_throughput_f32;
    if (std.mem.eql(u8, value, "mm_bf16") or std.mem.eql(u8, value, "matmul_thr_bf16")) return .matmul_throughput_bf16;
    if (std.mem.eql(u8, value, "mm_f16") or std.mem.eql(u8, value, "matmul_thr_f16")) return .matmul_throughput_f16;
    if (std.mem.eql(u8, value, "mm_u4") or std.mem.eql(u8, value, "matmul_thr_u4")) return .matmul_throughput_gaffine_u4;
    if (std.mem.eql(u8, value, "mm_u8") or std.mem.eql(u8, value, "matmul_thr_u8")) return .matmul_throughput_gaffine_u8;
    if (std.mem.eql(u8, value, "micro") or std.mem.eql(u8, value, "micro_matmul_f32")) return .micro_matmul_f32;
    if (std.mem.eql(u8, value, "add") or std.mem.eql(u8, value, "add_f32")) return .add_f32;
    if (std.mem.eql(u8, value, "mul") or std.mem.eql(u8, value, "mul_f32")) return .mul_f32;
    return error.InvalidArgument;
}

fn parseFormat(value: []const u8) !OutputFormat {
    if (std.mem.eql(u8, value, "table")) return .table;
    if (std.mem.eql(u8, value, "csv")) return .csv;
    if (std.mem.eql(u8, value, "tsv")) return .tsv;
    return error.InvalidArgument;
}

fn parseUsize(value: []const u8) !usize {
    return std.fmt.parseUnsigned(usize, value, 10);
}

fn printUsage(writer: anytype) !void {
    try writer.writeAll(
        \\Usage:
        \\  zig build bench-cpu-compute -Drelease -- [options]
        \\
        \\Options:
        \\  --scenario <all|prefill_attn_q|prefill_attn_k|prefill_attn_v|prefill_attn_out|prefill_ffn_gate|prefill_ffn_down|decode_attn_q|decode_attn_k|decode_attn_v|decode_attn_out|decode_ffn_gate|decode_ffn_down|decode_lm_head|decode_lm_head_bf16|decode_lm_head_f16|decode_lm_head_f32|decode_lm_head_runtime_f32|prefill_layer_attn_norm|prefill_layer_ffn_norm|prefill_final_norm|decode_layer_attn_norm|decode_layer_ffn_norm|decode_final_norm|delta|gdnorm|gate|gate_long|rope|rope_f16|rope_bf16|sdpa|sdpa_f16|sdpa_bf16|rms|rms_bf16|rms_f16|smx|decode_bf16|decode_u4|decode_u8|scv|ssm|mmthr|mm_bf16|mm_f16|mm_u4|mm_u8|micro|add|mul>
        \\  --preset <architecture_id>
        \\  --profile <ci|bw>               default: bw
        \\  --format <table|csv|tsv>        default: table
        \\  --warmup <N>                    default: 8
        \\  --iters <N>                     default: 24
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
        } else if (std.mem.eql(u8, arg, "--profile")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.profile = try parseProfile(args[idx]);
        } else if (std.mem.eql(u8, arg, "--preset")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.model_id = try parsePreset(args[idx]);
        } else if (std.mem.eql(u8, arg, "--format")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.format = try parseFormat(args[idx]);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.warmup = try parseUsize(args[idx]);
        } else if (std.mem.eql(u8, arg, "--iters")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.iters = try parseUsize(args[idx]);
        } else {
            return error.InvalidArgument;
        }
    }
    if (cfg.run.iters == 0) return error.InvalidArgument;
    return cfg;
}

fn coldGapUs(cold_ns: u64, warm_p50_ns: u64) f64 {
    if (cold_ns <= warm_p50_ns) return 0.0;
    return @as(f64, @floatFromInt(cold_ns - warm_p50_ns)) / 1000.0;
}

fn printResultTable(writer: anytype, cfg: scenarios.RunConfig, result: *scenarios.ScenarioResult, summary: harness.Summary, cold_us: f64) !void {
    const gbps = metrics.gbps(result.bytes_per_iter, summary.p50_ns);
    const tflops = metrics.tflops(result.flops_per_iter, summary.p50_ns);
    const warm_us = @as(f64, @floatFromInt(summary.p50_ns)) / 1000.0;
    const spr_pct = spreadPct(summary);
    const bytes_m = @as(f64, @floatFromInt(result.bytes_per_iter)) / 1_000_000.0;
    const flops_m = @as(f64, @floatFromInt(result.flops_per_iter)) / 1_000_000.0;
    try writer.print(
        "{s: <16} {s: <2} {d: >5} {d: >7.1} {d: >6.1} {d: >7.1} {d: >6.2} {d: >7.4} {d: >6.1} {d: >6.1}\n",
        .{ result.name, profileName(cfg.profile), result.sample_loops, warm_us, spr_pct, cold_us, gbps, tflops, bytes_m, flops_m },
    );
}

fn printTsvHeader(writer: anytype) !void {
    try writer.writeAll("scenario\tprofile\twarmup\titers\tloops\twarm_us\tspread_pct\tcold_us\tgbps\ttflops\tbytes_it\tflops_it\n");
}

fn printResultTsv(writer: anytype, cfg: scenarios.RunConfig, result: *scenarios.ScenarioResult, summary: harness.Summary, cold_us: f64) !void {
    const gbps = metrics.gbps(result.bytes_per_iter, summary.p50_ns);
    const tflops = metrics.tflops(result.flops_per_iter, summary.p50_ns);
    const warm_us = @as(f64, @floatFromInt(summary.p50_ns)) / 1000.0;
    const spr_pct = spreadPct(summary);
    try writer.print(
        "{s}\t{s}\t{}\t{}\t{}\t{d:.3}\t{d:.2}\t{d:.3}\t{d:.3}\t{d:.6}\t{}\t{}\n",
        .{ result.name, profileName(cfg.profile), cfg.warmup, cfg.iters, result.sample_loops, warm_us, spr_pct, cold_us, gbps, tflops, result.bytes_per_iter, result.flops_per_iter },
    );
}

fn printCsvHeader(writer: anytype) !void {
    try writer.writeAll("scenario,profile,warmup,iters,loops,warm_us,spread_pct,cold_us,gbps,tflops,bytes_it,flops_it\n");
}

fn printResultCsv(writer: anytype, cfg: scenarios.RunConfig, result: *scenarios.ScenarioResult, summary: harness.Summary, cold_us: f64) !void {
    const gbps = metrics.gbps(result.bytes_per_iter, summary.p50_ns);
    const tflops = metrics.tflops(result.flops_per_iter, summary.p50_ns);
    const warm_us = @as(f64, @floatFromInt(summary.p50_ns)) / 1000.0;
    const spr_pct = spreadPct(summary);
    try writer.print(
        "{s},{s},{},{},{},{d:.3},{d:.2},{d:.3},{d:.3},{d:.6},{},{}\n",
        .{ result.name, profileName(cfg.profile), cfg.warmup, cfg.iters, result.sample_loops, warm_us, spr_pct, cold_us, gbps, tflops, result.bytes_per_iter, result.flops_per_iter },
    );
}

fn runScenario(allocator: std.mem.Allocator, cfg: scenarios.RunConfig, which: scenarios.Scenario) !scenarios.ScenarioResult {
    return switch (which) {
        .prefill_attn_q_bf16 => try scenarios.runPrefillAttnQBf16(allocator, cfg),
        .prefill_attn_k_bf16 => try scenarios.runPrefillAttnKBf16(allocator, cfg),
        .prefill_attn_v_bf16 => try scenarios.runPrefillAttnVBf16(allocator, cfg),
        .prefill_attn_out_bf16 => try scenarios.runPrefillAttnOutBf16(allocator, cfg),
        .prefill_ffn_gate_bf16 => try scenarios.runPrefillFfnGateBf16(allocator, cfg),
        .prefill_ffn_down_bf16 => try scenarios.runPrefillFfnDownBf16(allocator, cfg),
        .decode_attn_q_bf16 => try scenarios.runDecodeAttnQBf16(allocator, cfg),
        .decode_attn_k_bf16 => try scenarios.runDecodeAttnKBf16(allocator, cfg),
        .decode_attn_v_bf16 => try scenarios.runDecodeAttnVBf16(allocator, cfg),
        .decode_attn_out_bf16 => try scenarios.runDecodeAttnOutBf16(allocator, cfg),
        .decode_ffn_gate_bf16 => try scenarios.runDecodeFfnGateBf16(allocator, cfg),
        .decode_ffn_down_bf16 => try scenarios.runDecodeFfnDownBf16(allocator, cfg),
        .decode_lm_head_bf16 => try scenarios.runDecodeLmHeadBf16(allocator, cfg),
        .decode_lm_head_f16 => try scenarios.runDecodeLmHeadF16(allocator, cfg),
        .decode_lm_head_f32 => try scenarios.runDecodeLmHeadF32(allocator, cfg),
        .decode_lm_head_runtime_f32 => try scenarios.runDecodeLmHeadRuntimeF32(allocator, cfg),
        .prefill_layer_attn_norm_f32 => try scenarios.runPrefillLayerAttnNormF32(allocator, cfg),
        .prefill_layer_ffn_norm_f32 => try scenarios.runPrefillLayerFfnNormF32(allocator, cfg),
        .prefill_final_norm_f32 => try scenarios.runPrefillFinalNormF32(allocator, cfg),
        .decode_layer_attn_norm_f32 => try scenarios.runDecodeLayerAttnNormF32(allocator, cfg),
        .decode_layer_ffn_norm_f32 => try scenarios.runDecodeLayerFfnNormF32(allocator, cfg),
        .decode_final_norm_f32 => try scenarios.runDecodeFinalNormF32(allocator, cfg),
        .add_f32 => try scenarios.runAddF32(allocator, cfg),
        .mul_f32 => try scenarios.runMulF32(allocator, cfg),
        .rmsnorm_f32 => try scenarios.runRmsNormF32(allocator, cfg),
        .rmsnorm_bf16_weight => try scenarios.runRmsNormBf16Weight(allocator, cfg),
        .rmsnorm_f16_weight => try scenarios.runRmsNormF16Weight(allocator, cfg),
        .softmax_f32 => try scenarios.runSoftmaxF32(allocator, cfg),
        .shortconv_decode_f32 => try scenarios.runShortconvDecodeF32(allocator, cfg),
        .mamba_scan_f32 => try scenarios.runMambaScanF32(allocator, cfg),
        .gated_delta_conv_f32 => try scenarios.runGatedDeltaConvF32(allocator, cfg),
        .gated_delta_qk_norm_f32 => try scenarios.runGatedDeltaQkNormF32(allocator, cfg),
        .gated_delta_step_f32 => try scenarios.runGatedDeltaStepF32(allocator, cfg),
        .gated_delta_norm_f32 => try scenarios.runGatedDeltaNormF32(allocator, cfg),
        .gated_attention_gate_f32 => try scenarios.runGatedAttentionGateF32(allocator, cfg),
        .gated_attention_gate_long_f32 => try scenarios.runGatedAttentionGateLongF32(allocator, cfg),
        .rope_f32 => try scenarios.runRopeF32(allocator, cfg),
        .rope_f16 => try scenarios.runRopeF16(allocator, cfg),
        .rope_bf16 => try scenarios.runRopeBf16(allocator, cfg),
        .sdpa_f32 => try scenarios.runSdpaF32(allocator, cfg),
        .sdpa_f16 => try scenarios.runSdpaF16(allocator, cfg),
        .sdpa_bf16 => try scenarios.runSdpaBf16(allocator, cfg),
        .matmul_throughput_f32 => try scenarios.runMatmulThroughputF32(allocator, cfg),
        .matmul_throughput_bf16 => try scenarios.runMatmulThroughputBf16(allocator, cfg),
        .matmul_throughput_f16 => try scenarios.runMatmulThroughputF16(allocator, cfg),
        .matmul_throughput_gaffine_u4 => try scenarios.runMatmulThroughputGroupedAffineU4(allocator, cfg),
        .matmul_throughput_gaffine_u8 => try scenarios.runMatmulThroughputGroupedAffineU8(allocator, cfg),
        .micro_matmul_f32 => try scenarios.runMicroMatmulF32(allocator, cfg),
        .decode_bf16 => try scenarios.runDecodeBf16(allocator, cfg),
        .decode_u4 => try scenarios.runDecodeU4(allocator, cfg),
        .decode_u8 => try scenarios.runDecodeU8(allocator, cfg),
        .all => return error.InvalidArgument,
    };
}

fn probeColdP50Ns(allocator: std.mem.Allocator, cfg: scenarios.RunConfig, which: scenarios.Scenario) !u64 {
    const values = try allocator.alloc(u64, cold_probe_repeats);
    defer allocator.free(values);
    var probe_cfg = cfg;
    probe_cfg.warmup = 1;
    probe_cfg.iters = 1;
    for (values, 0..) |*slot, idx| {
        _ = idx;
        var probe = try runScenario(allocator, probe_cfg, which);
        defer probe.deinit(allocator);
        slot.* = probe.cold_ns;
    }
    const summary = try harness.summarizeValues(allocator, values);
    return summary.p50_ns;
}

fn runOne(writer: anytype, allocator: std.mem.Allocator, cfg: scenarios.RunConfig, format: OutputFormat, which: scenarios.Scenario) !RowMetrics {
    if (cfg.profile == .bw and which == .matmul_throughput_f32) {
        var prime = try runScenario(allocator, cfg, which);
        prime.deinit(allocator);
    }

    var result = try runScenario(allocator, cfg, which);
    defer result.deinit(allocator);
    const summary = try harness.summarizeSamples(allocator, result.samples);
    const cold_p50_ns = try probeColdP50Ns(allocator, cfg, which);
    const cold_us = coldGapUs(cold_p50_ns, summary.p50_ns);
    const row = RowMetrics{
        .name = result.name,
        .gbps = metrics.gbps(result.bytes_per_iter, summary.p50_ns),
        .tflops = metrics.tflops(result.flops_per_iter, summary.p50_ns),
    };
    switch (format) {
        .table => try printResultTable(writer, cfg, &result, summary, cold_us),
        .csv => try printResultCsv(writer, cfg, &result, summary, cold_us),
        .tsv => try printResultTsv(writer, cfg, &result, summary, cold_us),
    }
    return row;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) @panic("benchmark leaked memory");
    }
    const allocator = gpa.allocator();
    const stdout = std.fs.File.stdout().deprecatedWriter();
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const cfg = parseArgs(args) catch |err| switch (err) {
        error.HelpRequested => {
            try printUsage(stdout);
            return;
        },
        else => {
            try printUsage(stdout);
            return err;
        },
    };

    switch (cfg.format) {
        .table => {
            try stdout.writeAll("CPU Compute Overview (warm p50 + cold gap)\n");
            if (cfg.run.model_id) |model_id| {
                try stdout.print("config: profile={s} warmup={} iters={} preset={s}\n", .{ profileName(cfg.run.profile), cfg.run.warmup, cfg.run.iters, model_id });
            } else {
                try stdout.print("config: profile={s} warmup={} iters={}\n", .{ profileName(cfg.run.profile), cfg.run.warmup, cfg.run.iters });
            }
            try stdout.writeAll("scenario         pr loops warm_us  spr% cold_us  GB/s    TF/s  MB_it  MF_it\n");
            try stdout.writeAll("------------------------------------------------------------------------\n");
        },
        .csv => try printCsvHeader(stdout),
        .tsv => try printTsvHeader(stdout),
    }

    var peaks = PeakStats{};
    switch (cfg.scenario) {
        .all => {
            if (cfg.run.model_id) |model_id| {
                try runModelPreset(stdout, allocator, cfg.run, cfg.format, model_id, &peaks);
            } else {
                if (cfg.format == .table) try stdout.writeAll("P1 model-critical\n");
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_conv_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_qk_norm_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_step_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_norm_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_attention_gate_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_attention_gate_long_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_f16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_bf16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_f16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_bf16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .shortconv_decode_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mamba_scan_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_bf16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_gaffine_u4));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_gaffine_u8));
                if (cfg.format == .table) try stdout.writeAll("P2 component methods\n");
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .add_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mul_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_bf16_weight));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_f16_weight));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .softmax_f32));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_bf16));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_u4));
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_u8));
                if (cfg.format == .table) try stdout.writeAll("P3 micro-shape sanity\n");
                updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .micro_matmul_f32));
            }
        },
        .prefill_attn_q_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_attn_q_bf16)),
        .prefill_attn_k_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_attn_k_bf16)),
        .prefill_attn_v_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_attn_v_bf16)),
        .prefill_attn_out_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_attn_out_bf16)),
        .prefill_ffn_gate_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_ffn_gate_bf16)),
        .prefill_ffn_down_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_ffn_down_bf16)),
        .decode_attn_q_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_attn_q_bf16)),
        .decode_attn_k_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_attn_k_bf16)),
        .decode_attn_v_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_attn_v_bf16)),
        .decode_attn_out_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_attn_out_bf16)),
        .decode_ffn_gate_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_ffn_gate_bf16)),
        .decode_ffn_down_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_ffn_down_bf16)),
        .decode_lm_head_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_lm_head_bf16)),
        .decode_lm_head_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_lm_head_f16)),
        .decode_lm_head_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_lm_head_f32)),
        .decode_lm_head_runtime_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_lm_head_runtime_f32)),
        .prefill_layer_attn_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_layer_attn_norm_f32)),
        .prefill_layer_ffn_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_layer_ffn_norm_f32)),
        .prefill_final_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .prefill_final_norm_f32)),
        .decode_layer_attn_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_layer_attn_norm_f32)),
        .decode_layer_ffn_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_layer_ffn_norm_f32)),
        .decode_final_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_final_norm_f32)),
        .add_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .add_f32)),
        .mul_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mul_f32)),
        .rmsnorm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_f32)),
        .rmsnorm_bf16_weight => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_bf16_weight)),
        .rmsnorm_f16_weight => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rmsnorm_f16_weight)),
        .softmax_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .softmax_f32)),
        .shortconv_decode_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .shortconv_decode_f32)),
        .mamba_scan_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mamba_scan_f32)),
        .gated_delta_conv_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_conv_f32)),
        .gated_delta_qk_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_qk_norm_f32)),
        .gated_delta_step_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_step_f32)),
        .gated_delta_norm_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_delta_norm_f32)),
        .gated_attention_gate_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_attention_gate_f32)),
        .gated_attention_gate_long_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .gated_attention_gate_long_f32)),
        .rope_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_f32)),
        .rope_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_f16)),
        .rope_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rope_bf16)),
        .sdpa_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_f32)),
        .sdpa_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_f16)),
        .sdpa_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .sdpa_bf16)),
        .matmul_throughput_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f32)),
        .matmul_throughput_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_bf16)),
        .matmul_throughput_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f16)),
        .matmul_throughput_gaffine_u4 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_gaffine_u4)),
        .matmul_throughput_gaffine_u8 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_gaffine_u8)),
        .micro_matmul_f32 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .micro_matmul_f32)),
        .decode_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_bf16)),
        .decode_u4 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_u4)),
        .decode_u8 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_u8)),
    }

    if (cfg.format == .table) {
        try stdout.writeAll("------------------------------------------------------------------------\n");
        try stdout.print("peak GB/s: {d:.2} ({s})  peak TF/s: {d:.4} ({s})\n", .{ peaks.gbps, peaks.gbps_name, peaks.tflops, peaks.tflops_name });
        try stdout.writeAll("Legend:\n");
        try stdout.writeAll("  loops   : inner kernel repeats per measured sample. Higher means less timer noise.\n");
        try stdout.writeAll("  warm_us : warm-path p50 eval latency (us). Lower is better.\n");
        try stdout.writeAll("  spr%    : (p90 - p50) / p50 * 100. Lower is more trustworthy.\n");
        try stdout.writeAll("  cold_us : one-time cold gap in eval latency (us). Lower is better.\n");
        try stdout.writeAll("  GB/s    : effective bytes_it / eval_p50_ns. Cache-resident methods can exceed DRAM GB/s.\n");
        try stdout.writeAll("  TF/s    : flops_it / eval_p50_ns / 1000. Higher is better for compute-bound.\n");
        try stdout.writeAll("  MB_it   : per-iteration bytes in decimal MB.\n");
        try stdout.writeAll("  MF_it   : per-iteration flops in decimal MF.\n");
    }
}
fn scenarioFromBenchRowName(name: []const u8) ?scenarios.Scenario {
    if (std.mem.eql(u8, name, "prefill.attn_q")) return .prefill_attn_q_bf16;
    if (std.mem.eql(u8, name, "prefill.attn_k")) return .prefill_attn_k_bf16;
    if (std.mem.eql(u8, name, "prefill.attn_v")) return .prefill_attn_v_bf16;
    if (std.mem.eql(u8, name, "prefill.attn_out")) return .prefill_attn_out_bf16;
    if (std.mem.eql(u8, name, "prefill.ffn_gate")) return .prefill_ffn_gate_bf16;
    if (std.mem.eql(u8, name, "prefill.ffn_down")) return .prefill_ffn_down_bf16;
    if (std.mem.eql(u8, name, "decode.attn_q")) return .decode_attn_q_bf16;
    if (std.mem.eql(u8, name, "decode.attn_k")) return .decode_attn_k_bf16;
    if (std.mem.eql(u8, name, "decode.attn_v")) return .decode_attn_v_bf16;
    if (std.mem.eql(u8, name, "decode.attn_out")) return .decode_attn_out_bf16;
    if (std.mem.eql(u8, name, "decode.ffn_gate")) return .decode_ffn_gate_bf16;
    if (std.mem.eql(u8, name, "decode.ffn_down")) return .decode_ffn_down_bf16;
    if (std.mem.eql(u8, name, "decode.lm_head")) return .decode_lm_head_bf16;
    if (std.mem.eql(u8, name, "decode.lm_head_bf16")) return .decode_lm_head_bf16;
    if (std.mem.eql(u8, name, "decode.lm_head_f16")) return .decode_lm_head_f16;
    if (std.mem.eql(u8, name, "decode.lm_head_f32")) return .decode_lm_head_f32;
    if (std.mem.eql(u8, name, "decode.lm_head_runtime_f32")) return .decode_lm_head_runtime_f32;
    if (std.mem.eql(u8, name, "prefill.layer_attn_norm")) return .prefill_layer_attn_norm_f32;
    if (std.mem.eql(u8, name, "prefill.layer_ffn_norm")) return .prefill_layer_ffn_norm_f32;
    if (std.mem.eql(u8, name, "prefill.final_norm")) return .prefill_final_norm_f32;
    if (std.mem.eql(u8, name, "decode.layer_attn_norm")) return .decode_layer_attn_norm_f32;
    if (std.mem.eql(u8, name, "decode.layer_ffn_norm")) return .decode_layer_ffn_norm_f32;
    if (std.mem.eql(u8, name, "decode.final_norm")) return .decode_final_norm_f32;
    if (std.mem.eql(u8, name, "gdelta_conv_f32")) return .gated_delta_conv_f32;
    if (std.mem.eql(u8, name, "gdelta_qk_norm_f32")) return .gated_delta_qk_norm_f32;
    if (std.mem.eql(u8, name, "gdelta_step_f32")) return .gated_delta_step_f32;
    if (std.mem.eql(u8, name, "gdelta_norm_f32")) return .gated_delta_norm_f32;
    if (std.mem.eql(u8, name, "rope_f32")) return .rope_f32;
    if (std.mem.eql(u8, name, "rope_f16")) return .rope_f16;
    if (std.mem.eql(u8, name, "rope_bf16")) return .rope_bf16;
    if (std.mem.eql(u8, name, "rms_f32")) return .rmsnorm_f32;
    if (std.mem.eql(u8, name, "shortconv_f32")) return .shortconv_decode_f32;
    if (std.mem.eql(u8, name, "mamba_scan_f32")) return .mamba_scan_f32;
    return null;
}

fn runModelPreset(writer: anytype, allocator: std.mem.Allocator, cfg: scenarios.RunConfig, format: OutputFormat, model_id: []const u8, peaks: *PeakStats) !void {
    const hints = models.performanceHintsByName(model_id) orelse return error.InvalidArgument;

    var seen = std.StringHashMap(void).init(allocator);
    defer seen.deinit();

    if (format == .table and hints.prefill_point_mappings.len != 0) try writer.writeAll("P0 xray prefill-aligned roles\n");
    for (hints.prefill_point_mappings) |mapping| {
        if (seen.contains(mapping.bench_row)) continue;
        try seen.put(mapping.bench_row, {});
        const scenario = scenarioFromBenchRowName(mapping.bench_row) orelse continue;
        updatePeaks(peaks, try runOne(writer, allocator, cfg, format, scenario));
    }

    if (format == .table and hints.decode_point_mappings.len != 0) try writer.writeAll("P1 xray decode-aligned roles\n");
    for (hints.decode_point_mappings) |mapping| {
        if (seen.contains(mapping.bench_row)) continue;
        try seen.put(mapping.bench_row, {});
        const scenario = scenarioFromBenchRowName(mapping.bench_row) orelse continue;
        updatePeaks(peaks, try runOne(writer, allocator, cfg, format, scenario));
    }

    if (format == .table and hints.prefill_hidden_rows.len != 0) try writer.writeAll("P2 hidden compute behind prefill roles\n");
    for (hints.prefill_hidden_rows) |row_name| {
        const scenario = scenarioFromBenchRowName(row_name) orelse continue;
        updatePeaks(peaks, try runOne(writer, allocator, cfg, format, scenario));
    }

    if (format == .table and hints.decode_hidden_rows.len != 0) try writer.writeAll("P3 hidden compute behind decode roles\n");
    for (hints.decode_hidden_rows) |row_name| {
        const scenario = scenarioFromBenchRowName(row_name) orelse continue;
        updatePeaks(peaks, try runOne(writer, allocator, cfg, format, scenario));
    }
}
