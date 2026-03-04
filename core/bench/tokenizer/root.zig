const std = @import("std");
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
    mbps: f64,
    mtok: f64,
};

const PeakStats = struct {
    mbps: f64 = 0.0,
    mbps_name: []const u8 = "-",
    mtok: f64 = 0.0,
    mtok_name: []const u8 = "-",
};

const cold_probe_repeats: usize = 5;

fn updatePeaks(peaks: *PeakStats, row: RowMetrics) void {
    if (row.mbps >= peaks.mbps) {
        peaks.mbps = row.mbps;
        peaks.mbps_name = row.name;
    }
    if (row.mtok >= peaks.mtok) {
        peaks.mtok = row.mtok;
        peaks.mtok_name = row.name;
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
    if (std.mem.eql(u8, value, "encode_1k") or std.mem.eql(u8, value, "1k")) return .encode_1k;
    if (std.mem.eql(u8, value, "encode_100k") or std.mem.eql(u8, value, "100k")) return .encode_100k;
    if (std.mem.eql(u8, value, "encode_1m") or std.mem.eql(u8, value, "1m")) return .encode_1m;
    if (std.mem.eql(u8, value, "load_json") or std.mem.eql(u8, value, "load")) return .load_json;
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
        \\  zig build bench-tokenizer -Drelease -- [options]
        \\
        \\Options:
        \\  --scenario <all|1k|100k|1m|load>     default: all
        \\  --tokenizer <path/to/tokenizer.json>  required
        \\  --profile <ci|bw>                     default: bw
        \\  --format <table|csv|tsv>              default: table
        \\  --warmup <N>                          default: 4
        \\  --iters <N>                           default: 10
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
        } else if (std.mem.eql(u8, arg, "--tokenizer")) {
            idx += 1;
            if (idx >= args.len) return error.InvalidArgument;
            cfg.run.tokenizer_json_path = args[idx];
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
    if (cfg.run.tokenizer_json_path.len == 0) return error.MissingTokenizer;
    return cfg;
}

fn coldGapMs(cold_ns: u64, warm_p50_ns: u64) f64 {
    if (cold_ns <= warm_p50_ns) return 0.0;
    return @as(f64, @floatFromInt(cold_ns - warm_p50_ns)) / 1_000_000.0;
}

fn printResultTable(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.Summary,
    cold_ms: f64,
) !void {
    const warm_ms = @as(f64, @floatFromInt(summary.p50_ns)) / 1_000_000.0;
    const in_kb = @as(f64, @floatFromInt(result.input_bytes)) / 1024.0;

    if (result.output_tokens > 0) {
        const mb_s = metrics.mbps(result.input_bytes, summary.p50_ns);
        const mt_s = metrics.mtok_per_sec(result.output_tokens, summary.p50_ns);
        try writer.print(
            "{s: <12} {s: <2} {d: >8.2} {d: >7.2} {d: >6.2} {d: >6.3} {d: >7.1} {d: >7}\n",
            .{
                result.name,
                profileName(cfg.profile),
                warm_ms,
                cold_ms,
                mb_s,
                mt_s,
                in_kb,
                result.output_tokens,
            },
        );
    } else {
        try writer.print(
            "{s: <12} {s: <2} {d: >8.2} {d: >7.2} {s: >6} {s: >6} {d: >7.1} {s: >7}\n",
            .{
                result.name,
                profileName(cfg.profile),
                warm_ms,
                cold_ms,
                "-",
                "-",
                in_kb,
                "-",
            },
        );
    }
}

fn printCsvHeader(writer: anytype) !void {
    try writer.writeAll("scenario,profile,warmup,iters,warm_ms,cold_ms,mbps,mtok_s,input_bytes,output_tokens\n");
}

fn printResultCsv(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.Summary,
    cold_ms: f64,
) !void {
    const warm_ms = @as(f64, @floatFromInt(summary.p50_ns)) / 1_000_000.0;
    const mb_s = metrics.mbps(result.input_bytes, summary.p50_ns);
    const mt_s = metrics.mtok_per_sec(result.output_tokens, summary.p50_ns);

    try writer.print(
        "{s},{s},{},{},{d:.3},{d:.3},{d:.3},{d:.6},{},{}\n",
        .{
            result.name,
            profileName(cfg.profile),
            cfg.warmup,
            cfg.iters,
            warm_ms,
            cold_ms,
            mb_s,
            mt_s,
            result.input_bytes,
            result.output_tokens,
        },
    );
}

fn printTsvHeader(writer: anytype) !void {
    try writer.writeAll(
        "scenario\tprofile\twarmup\titers\twarm_ms\tcold_ms\tmbps\tmtok_s\tinput_bytes\toutput_tokens\n",
    );
}

fn printResultTsv(
    writer: anytype,
    cfg: scenarios.RunConfig,
    result: *scenarios.ScenarioResult,
    summary: harness.Summary,
    cold_ms: f64,
) !void {
    const warm_ms = @as(f64, @floatFromInt(summary.p50_ns)) / 1_000_000.0;
    const mb_s = metrics.mbps(result.input_bytes, summary.p50_ns);
    const mt_s = metrics.mtok_per_sec(result.output_tokens, summary.p50_ns);

    try writer.print(
        "{s}\t{s}\t{}\t{}\t{d:.3}\t{d:.3}\t{d:.3}\t{d:.6}\t{}\t{}\n",
        .{
            result.name,
            profileName(cfg.profile),
            cfg.warmup,
            cfg.iters,
            warm_ms,
            cold_ms,
            mb_s,
            mt_s,
            result.input_bytes,
            result.output_tokens,
        },
    );
}

fn runScenario(
    allocator: std.mem.Allocator,
    cfg: scenarios.RunConfig,
    which: scenarios.Scenario,
) !scenarios.ScenarioResult {
    return switch (which) {
        .encode_1k => try scenarios.runEncode1k(allocator, cfg),
        .encode_100k => try scenarios.runEncode100k(allocator, cfg),
        .encode_1m => try scenarios.runEncode1m(allocator, cfg),
        .load_json => try scenarios.runLoadJson(allocator, cfg),
        .all => return error.InvalidArgument,
    };
}

fn probeColdP50Ns(
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
        cold_totals[idx] = probe_result.cold_first.encode_ns;
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
    var result = try runScenario(allocator, cfg, which);
    defer result.deinit(allocator);

    const summary = try harness.summarizeSamples(allocator, result.samples);
    const cold_total_p50_ns = try probeColdP50Ns(allocator, cfg, which);
    const cold_ms = coldGapMs(cold_total_p50_ns, summary.p50_ns);
    const row = RowMetrics{
        .name = result.name,
        .mbps = metrics.mbps(result.input_bytes, summary.p50_ns),
        .mtok = metrics.mtok_per_sec(result.output_tokens, summary.p50_ns),
    };
    switch (format) {
        .table => try printResultTable(writer, cfg, &result, summary, cold_ms),
        .csv => try printResultCsv(writer, cfg, &result, summary, cold_ms),
        .tsv => try printResultTsv(writer, cfg, &result, summary, cold_ms),
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
        error.MissingTokenizer => {
            try stdout.writeAll("error: --tokenizer <path> is required\n\n");
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
            try stdout.writeAll("Tokenizer Encode Benchmark (warm p50)\n");
            try stdout.print("config: profile={s} warmup={} iters={}\n", .{
                profileName(cfg.run.profile),
                cfg.run.warmup,
                cfg.run.iters,
            });
            try stdout.print("tokenizer: {s}\n", .{cfg.run.tokenizer_json_path});
            try stdout.writeAll("scenario     pr  warm_ms cold_ms  MB/s  Mtok/s   in_KB out_tok\n");
            try stdout.writeAll("--------------------------------------------------------------\n");
        },
        .csv => try printCsvHeader(stdout),
        .tsv => try printTsvHeader(stdout),
    }

    var peaks = PeakStats{};
    switch (cfg.scenario) {
        .all => {
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_1k));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_100k));
            updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_1m));
            _ = try runOne(stdout, allocator, cfg.run, cfg.format, .load_json);
        },
        .encode_1k => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_1k)),
        .encode_100k => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_100k)),
        .encode_1m => updatePeaks(&peaks, try runOne(stdout, allocator, cfg.run, cfg.format, .encode_1m)),
        .load_json => _ = try runOne(stdout, allocator, cfg.run, cfg.format, .load_json),
    }

    if (cfg.format == .table) {
        try stdout.writeAll("--------------------------------------------------------------\n");
        try stdout.print("peak MB/s: {d:.2} ({s})  peak Mtok/s: {d:.3} ({s})\n", .{
            peaks.mbps,
            peaks.mbps_name,
            peaks.mtok,
            peaks.mtok_name,
        });
    }
}

test "parseScenario accepts short and full names" {
    try std.testing.expectEqual(scenarios.Scenario.encode_1k, try parseScenario("1k"));
    try std.testing.expectEqual(scenarios.Scenario.encode_1k, try parseScenario("encode_1k"));
    try std.testing.expectEqual(scenarios.Scenario.encode_100k, try parseScenario("100k"));
    try std.testing.expectEqual(scenarios.Scenario.encode_1m, try parseScenario("1m"));
    try std.testing.expectEqual(scenarios.Scenario.load_json, try parseScenario("load"));
    try std.testing.expectEqual(scenarios.Scenario.all, try parseScenario("all"));
}
