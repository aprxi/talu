const std = @import("std");
const builtin = @import("builtin");
const core_main = @import("main");
const metal = core_main.compute.metal;
const harness = @import("harness.zig");
const metrics = @import("metrics.zig");
const scenarios = @import("scenarios.zig");

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

fn parseScenario(value: []const u8) !scenarios.Scenario {
    if (std.mem.eql(u8, value, "all")) return .all;
    if (std.mem.eql(u8, value, "add") or std.mem.eql(u8, value, "add_f16")) return .add_f16;
    if (std.mem.eql(u8, value, "mul") or std.mem.eql(u8, value, "mul_f16")) return .mul_f16;
    if (std.mem.eql(u8, value, "rms") or std.mem.eql(u8, value, "rms_f16")) return .rms_f16;
    if (std.mem.eql(u8, value, "smx") or std.mem.eql(u8, value, "softmax_f16")) return .softmax_f16;
    if (std.mem.eql(u8, value, "ffnq") or std.mem.eql(u8, value, "ffnq_u4") or std.mem.eql(u8, value, "fused_ffn_quantized_decode_u4")) return .fused_ffn_quantized_decode_u4;
    if (std.mem.eql(u8, value, "ffnq_u8") or std.mem.eql(u8, value, "fused_ffn_quantized_decode_u8")) return .fused_ffn_quantized_decode_u8;
    if (std.mem.eql(u8, value, "qmm_u4") or std.mem.eql(u8, value, "quantized_matmul_u4")) return .quantized_matmul_u4;
    if (std.mem.eql(u8, value, "qmm_u8") or std.mem.eql(u8, value, "quantized_matmul_u8")) return .quantized_matmul_u8;
    if (std.mem.eql(u8, value, "attn") or std.mem.eql(u8, value, "attention_decode_f16")) return .attention_decode_f16;
    if (std.mem.eql(u8, value, "scv") or std.mem.eql(u8, value, "shortconv_decode_bf16")) return .shortconv_decode_bf16;
    if (std.mem.eql(u8, value, "mmthr") or std.mem.eql(u8, value, "matmul_throughput_f16")) return .matmul_throughput_f16;
    if (std.mem.eql(u8, value, "micro") or std.mem.eql(u8, value, "micro_matmul_f16")) return .micro_matmul_f16;
    if (std.mem.eql(u8, value, "decode") or std.mem.eql(u8, value, "decode_synth_f16")) return .decode_synth_f16;
    if (std.mem.eql(u8, value, "decd") or std.mem.eql(u8, value, "decode_dense_f16")) return .decode_dense_f16;
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
        \\  zig build bench-metal-compute -Drelease -- [options]
        \\
        \\Options:
        \\  --scenario <all|add|mul|rms|smx|ffnq_u4|ffnq_u8|qmm_u4|qmm_u8|attn|scv|mmthr|micro|decode|decd> default: all
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

fn coldGapUs(cold_total_ns: u64, warm_total_p50_ns: u64) f64 {
    if (cold_total_ns <= warm_total_p50_ns) return 0.0;
    return @as(f64, @floatFromInt(cold_total_ns - warm_total_p50_ns)) / 1000.0;
}

fn printResultTable(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.ScenarioSummary,
    cold_us: f64,
) !void {
    const gbps_eval_p50 = metrics.gbps(result.bytes_per_iter, summary.eval.p50_ns);
    const tflops_eval_p50 = metrics.tflops(result.flops_per_iter, summary.eval.p50_ns);
    const warm_us_p50 = @as(f64, @floatFromInt(summary.eval.p50_ns)) / 1000.0;
    const bytes_m = @as(f64, @floatFromInt(result.bytes_per_iter)) / 1_000_000.0;
    const flops_m = @as(f64, @floatFromInt(result.flops_per_iter)) / 1_000_000.0;

    try writer.print(
        "{s: <12} {s: <2} {d: >7.1} {d: >7.1} {d: >6.2} {d: >7.4} {d: >6.1} {d: >6.1}\n",
        .{
            result.name,
            profileName(cfg.profile),
            warm_us_p50,
            cold_us,
            gbps_eval_p50,
            tflops_eval_p50,
            bytes_m,
            flops_m,
        },
    );
}

fn printTsvHeader(writer: anytype) !void {
    try writer.writeAll(
        "scenario\tprofile\twarmup\titers\twarm_us\tcold_us\tgbps\ttflops\tbytes_it\tflops_it\n",
    );
}

fn printResultTsv(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.ScenarioSummary,
    cold_us: f64,
) !void {
    const gbps_eval_p50 = metrics.gbps(result.bytes_per_iter, summary.eval.p50_ns);
    const tflops_eval_p50 = metrics.tflops(result.flops_per_iter, summary.eval.p50_ns);
    const warm_us_p50 = @as(f64, @floatFromInt(summary.eval.p50_ns)) / 1000.0;

    try writer.print(
        "{s}\t{s}\t{}\t{}\t{d:.3}\t{d:.3}\t{d:.3}\t{d:.6}\t{}\t{}\n",
        .{
            result.name,
            profileName(cfg.profile),
            cfg.warmup,
            cfg.iters,
            warm_us_p50,
            cold_us,
            gbps_eval_p50,
            tflops_eval_p50,
            result.bytes_per_iter,
            result.flops_per_iter,
        },
    );
}

fn printCsvHeader(writer: anytype) !void {
    try writer.writeAll("scenario,profile,warmup,iters,warm_us,cold_us,gbps,tflops,bytes_it,flops_it\n");
}

fn printResultCsv(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.ScenarioSummary,
    cold_us: f64,
) !void {
    const gbps_eval_p50 = metrics.gbps(result.bytes_per_iter, summary.eval.p50_ns);
    const tflops_eval_p50 = metrics.tflops(result.flops_per_iter, summary.eval.p50_ns);
    const warm_us_p50 = @as(f64, @floatFromInt(summary.eval.p50_ns)) / 1000.0;

    try writer.print(
        "{s},{s},{},{},{d:.3},{d:.3},{d:.3},{d:.6},{},{}\n",
        .{
            result.name,
            profileName(cfg.profile),
            cfg.warmup,
            cfg.iters,
            warm_us_p50,
            cold_us,
            gbps_eval_p50,
            tflops_eval_p50,
            result.bytes_per_iter,
            result.flops_per_iter,
        },
    );
}

fn runScenario(
    allocator: std.mem.Allocator,
    cfg: scenarios.RunConfig,
    which: scenarios.Scenario,
) !scenarios.ScenarioResult {
    return switch (which) {
        .add_f16 => try scenarios.runAddF16(allocator, cfg),
        .mul_f16 => try scenarios.runMultiplyF16(allocator, cfg),
        .rms_f16 => try scenarios.runRmsNormF16(allocator, cfg),
        .softmax_f16 => try scenarios.runSoftmaxF16(allocator, cfg),
        .fused_ffn_quantized_decode_u4 => try scenarios.runFusedFfnQuantizedDecodeU4(allocator, cfg),
        .fused_ffn_quantized_decode_u8 => try scenarios.runFusedFfnQuantizedDecodeU8(allocator, cfg),
        .quantized_matmul_u4 => try scenarios.runQuantizedMatmulU4(allocator, cfg),
        .quantized_matmul_u8 => try scenarios.runQuantizedMatmulU8(allocator, cfg),
        .attention_decode_f16 => try scenarios.runAttentionDecodeF16(allocator, cfg),
        .shortconv_decode_bf16 => try scenarios.runShortconvDecodeBF16(allocator, cfg),
        .matmul_throughput_f16 => try scenarios.runMatmulThroughputF16(allocator, cfg),
        .micro_matmul_f16 => try scenarios.runMicroMatmulF16(allocator, cfg),
        .decode_synth_f16 => try scenarios.runDecodeSynthF16(allocator, cfg),
        .decode_dense_f16 => try scenarios.runDecodeDenseF16(allocator, cfg),
        .all => return error.InvalidArgument,
    };
}

fn probeColdTotalP50Ns(
    allocator: std.mem.Allocator,
    cfg: scenarios.RunConfig,
    which: scenarios.Scenario,
) !u64 {
    var cold_totals = try allocator.alloc(u64, cold_probe_repeats);
    defer allocator.free(cold_totals);

    var probe_cfg = cfg;
    probe_cfg.warmup = 1;
    probe_cfg.iters = 1;

    var idx: usize = 0;
    while (idx < cold_probe_repeats) : (idx += 1) {
        var probe_result = try runScenario(allocator, probe_cfg, which);
        defer probe_result.deinit(allocator);
        cold_totals[idx] = probe_result.cold_first.total_ns;
    }
    const summary = try harness.summarizeValues(allocator, cold_totals);
    return summary.p50_ns;
}

fn runOne(
    writer: anytype,
    allocator: std.mem.Allocator,
    cfg: scenarios.RunConfig,
    format: OutputFormat,
    which: scenarios.Scenario,
) !RowMetrics {
    if (cfg.profile == .bw and which == .matmul_throughput_f16) {
        // Prime one full unmeasured pass so single-scenario throughput runs
        // report sustained steady-state rather than first-invocation ramp.
        var prime = try runScenario(allocator, cfg, which);
        prime.deinit(allocator);
    }

    var result = try runScenario(allocator, cfg, which);
    defer result.deinit(allocator);

    const summary = try harness.summarizeSamples(allocator, result.samples);
    const cold_total_p50_ns = try probeColdTotalP50Ns(allocator, cfg, which);
    const cold_us = coldGapUs(cold_total_p50_ns, summary.total.p50_ns);
    const row = RowMetrics{
        .name = result.name,
        .gbps = metrics.gbps(result.bytes_per_iter, summary.eval.p50_ns),
        .tflops = metrics.tflops(result.flops_per_iter, summary.eval.p50_ns),
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

    if (comptime builtin.os.tag != .macos) {
        try stdout.writeAll("BENCH status=skip reason=non_macos\n");
        return;
    }
    if (!metal.isAvailable()) {
        try stdout.writeAll("BENCH status=skip reason=metal_unavailable\n");
        return;
    }

    switch (cfg.format) {
        .table => {
            try stdout.writeAll("Metal Compute Overview (warm p50 + cold gap)\n");
            try stdout.print("config: profile={s} warmup={} iters={}\n", .{
                profileName(cfg.run.profile),
                cfg.run.warmup,
                cfg.run.iters,
            });
            try stdout.writeAll("scenario     pr warm_us cold_us  GB/s    TF/s  MB_it  MF_it\n");
            try stdout.writeAll("-----------------------------------------------------------\n");
        },
        .csv => try printCsvHeader(stdout),
        .tsv => try printTsvHeader(stdout),
    }

    var peaks = PeakStats{};
    switch (cfg.scenario) {
        .all => {
            if (cfg.format == .table) try stdout.writeAll("P1 model-critical\n");
            // Most predictive rows first for model-level performance readouts.
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .fused_ffn_quantized_decode_u4));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .fused_ffn_quantized_decode_u8));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .quantized_matmul_u4));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .quantized_matmul_u8));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .attention_decode_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .shortconv_decode_bf16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_dense_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_synth_f16));
            if (cfg.format == .table) try stdout.writeAll("P2 component methods\n");
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .add_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mul_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rms_f16));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .softmax_f16));
            if (cfg.format == .table) try stdout.writeAll("P3 micro-shape sanity\n");
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .micro_matmul_f16));
        },
        .add_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .add_f16)),
        .mul_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .mul_f16)),
        .rms_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .rms_f16)),
        .softmax_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .softmax_f16)),
        .fused_ffn_quantized_decode_u4 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .fused_ffn_quantized_decode_u4)),
        .fused_ffn_quantized_decode_u8 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .fused_ffn_quantized_decode_u8)),
        .quantized_matmul_u4 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .quantized_matmul_u4)),
        .quantized_matmul_u8 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .quantized_matmul_u8)),
        .attention_decode_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .attention_decode_f16)),
        .shortconv_decode_bf16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .shortconv_decode_bf16)),
        .matmul_throughput_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .matmul_throughput_f16)),
        .micro_matmul_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .micro_matmul_f16)),
        .decode_synth_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_synth_f16)),
        .decode_dense_f16 => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .decode_dense_f16)),
    }

    if (cfg.format == .table) {
        try stdout.writeAll("-----------------------------------------------------------\n");
        try stdout.print("peak GB/s: {d:.2} ({s})  peak TF/s: {d:.4} ({s})\n", .{
            peaks.gbps,
            peaks.gbps_name,
            peaks.tflops,
            peaks.tflops_name,
        });
        try stdout.writeAll("Legend:\n");
        try stdout.writeAll("  warm_us : warm-path p50 eval latency (us). Lower is better.\n");
        try stdout.writeAll("  cold_us : one-time cold gap in total latency (us):\n");
        try stdout.writeAll("            max(0, cold_total_us - warm_total_us). Lower is better.\n");
        try stdout.writeAll("  GB/s    : bytes_it / eval_p50_ns. Higher is better for memory-bound methods.\n");
        try stdout.writeAll("  TF/s    : flops_it / eval_p50_ns / 1000. Higher is better for compute-bound.\n");
        try stdout.writeAll("  MB_it   : per-iteration bytes in decimal MB.\n");
        try stdout.writeAll("  MF_it   : per-iteration flops in decimal MF.\n");
    }
}

test "parseScenario accepts short and full names" {
    try std.testing.expectEqual(scenarios.Scenario.micro_matmul_f16, try parseScenario("micro"));
    try std.testing.expectEqual(scenarios.Scenario.decode_synth_f16, try parseScenario("decode_synth_f16"));
    try std.testing.expectEqual(scenarios.Scenario.decode_dense_f16, try parseScenario("decd"));
    try std.testing.expectEqual(scenarios.Scenario.shortconv_decode_bf16, try parseScenario("scv"));
}
