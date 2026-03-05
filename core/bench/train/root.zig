//! Training benchmark harness CLI.
//!
//! Usage:
//!   zig build bench-train -Drelease -- --scenario forward_linear
//!   zig build bench-train -Drelease -- --scenario all
//!   zig build bench-train -Drelease -- --scenario all --profile ci

const std = @import("std");
const scenarios = @import("scenarios.zig");
const metrics = @import("metrics.zig");

const Scenario = scenarios.Scenario;
const ScenarioResult = scenarios.ScenarioResult;
const RunConfig = scenarios.RunConfig;
const Profile = scenarios.Profile;

const CliConfig = struct {
    scenario: Scenario = .all,
    run_config: RunConfig = .{},
    format: Format = .table,
};

const Format = enum { table, csv };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cli = parseArgs() catch {
        printUsage();
        return;
    };

    const results = try scenarios.run(cli.scenario, cli.run_config, allocator);
    defer allocator.free(results);

    switch (cli.format) {
        .table => printTable(results),
        .csv => printCsv(results),
    }
}

fn printTable(results: []const ScenarioResult) void {
    const stdout = std.fs.File.stdout().deprecatedWriter();

    stdout.print("\n{s:<20} {s:>10} {s:>10} {s:>10} {s:>10} {s:>10} {s}\n", .{
        "scenario", "p50", "p90", "min", "max", "GFLOP/s", "note",
    }) catch {};
    stdout.print("{s:->20} {s:->10} {s:->10} {s:->10} {s:->10} {s:->10} {s:->20}\n", .{
        "", "", "", "", "", "", "",
    }) catch {};

    for (results) |r| {
        const gf = if (r.flops > 0)
            metrics.gflops(r.flops, r.summary.p50_ns)
        else
            0.0;

        stdout.print("{s:<20} {s:>10} {s:>10} {s:>10} {s:>10} {d:>10.2} {s}\n", .{
            r.name,
            fmtNs(r.summary.p50_ns),
            fmtNs(r.summary.p90_ns),
            fmtNs(r.summary.min_ns),
            fmtNs(r.summary.max_ns),
            gf,
            r.note,
        }) catch {};
    }
    stdout.print("\n", .{}) catch {};
}

fn printCsv(results: []const ScenarioResult) void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    stdout.print("scenario,p50_ns,p90_ns,min_ns,max_ns,mean_ns,flops,gflops\n", .{}) catch {};
    for (results) |r| {
        const gf = if (r.flops > 0) metrics.gflops(r.flops, r.summary.p50_ns) else 0.0;
        stdout.print("{s},{d},{d},{d},{d},{d},{d},{d:.2}\n", .{
            r.name,
            r.summary.p50_ns,
            r.summary.p90_ns,
            r.summary.min_ns,
            r.summary.max_ns,
            r.summary.mean_ns,
            r.flops,
            gf,
        }) catch {};
    }
}

fn fmtNs(ns: u64) [10]u8 {
    var buf: [10]u8 = [_]u8{' '} ** 10;
    if (ns == 0) {
        buf[8] = '0';
        buf[9] = ' ';
        return buf;
    }
    if (ns < 1_000) {
        // nanoseconds
        _ = std.fmt.bufPrint(&buf, "{d:>7} ns", .{ns}) catch {};
    } else if (ns < 1_000_000) {
        // microseconds
        const us = @as(f64, @floatFromInt(ns)) / 1_000.0;
        _ = std.fmt.bufPrint(&buf, "{d:>6.1} us", .{us}) catch {};
    } else if (ns < 1_000_000_000) {
        // milliseconds
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        _ = std.fmt.bufPrint(&buf, "{d:>6.1} ms", .{ms}) catch {};
    } else {
        const s = @as(f64, @floatFromInt(ns)) / 1_000_000_000.0;
        _ = std.fmt.bufPrint(&buf, "{d:>6.2}  s", .{s}) catch {};
    }
    return buf;
}

fn parseArgs() !CliConfig {
    var config = CliConfig{};
    var args = std.process.args();
    _ = args.next(); // skip program name

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--scenario")) {
            const val = args.next() orelse return error.MissingArgument;
            config.scenario = Scenario.fromString(val) orelse return error.UnknownScenario;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            const val = args.next() orelse return error.MissingArgument;
            if (std.mem.eql(u8, val, "ci")) {
                config.run_config = .{ .warmup = 2, .iters = 5, .profile = .ci };
            } else {
                config.run_config = .{ .warmup = 4, .iters = 20, .profile = .bw };
            }
        } else if (std.mem.eql(u8, arg, "--format")) {
            const val = args.next() orelse return error.MissingArgument;
            if (std.mem.eql(u8, val, "csv")) {
                config.format = .csv;
            }
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            const val = args.next() orelse return error.MissingArgument;
            config.run_config.warmup = std.fmt.parseInt(usize, val, 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--iters")) {
            const val = args.next() orelse return error.MissingArgument;
            config.run_config.iters = std.fmt.parseInt(usize, val, 10) catch return error.InvalidArgument;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return error.HelpRequested;
        }
    }

    return config;
}

fn printUsage() void {
    const stdout = std.fs.File.stdout().deprecatedWriter();
    stdout.print(
        \\Usage: bench-train [options]
        \\
        \\Options:
        \\  --scenario <name>    Scenario to run (default: all)
        \\  --profile <ci|bw>    Benchmark profile (default: bw)
        \\  --format <table|csv> Output format (default: table)
        \\  --warmup <n>         Warmup iterations (default: 4)
        \\  --iters <n>          Measurement iterations (default: 20)
        \\
        \\Scenarios:
        \\  forward_linear       Single linear projection
        \\  forward_attention    Causal full-seq attention
        \\  forward_norm         RMSNorm with saved inv_rms
        \\  forward_activation   SwiGLU forward
        \\  forward_rope         RoPE forward
        \\  forward_embedding    Token embedding lookup
        \\  forward_loss         Cross-entropy loss
        \\  forward_full         Full forward pass (all layers)
        \\  backward_linear      Linear backward (gradWeight + gradInput)
        \\  backward_full        Full backward pass
        \\  step_full            Forward + backward + clip + optimizer
        \\  optimizer_step       AdamW update only
        \\  all                  Run all scenarios
        \\
    , .{}) catch {};
}
