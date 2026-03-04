const std = @import("std");
const core_main = @import("main");
const tokenizer_mod = core_main.tokenizer;
const harness = @import("harness.zig");

pub const Profile = enum { ci, bw };

pub const RunConfig = struct {
    warmup: usize = 4,
    iters: usize = 10,
    profile: Profile = .bw,
    tokenizer_json_path: []const u8 = "",
};

pub const Scenario = enum {
    encode_1k,
    encode_100k,
    encode_1m,
    load_json,
    all,
};

pub const ScenarioResult = struct {
    name: []const u8,
    profile: Profile,
    samples: []harness.Sample,
    cold_first: harness.Sample,
    input_bytes: u64,
    output_tokens: u64,
    note: []const u8,

    pub fn deinit(self: *ScenarioResult, allocator: std.mem.Allocator) void {
        allocator.free(self.samples);
    }
};

// ---------------------------------------------------------------------------
// Text generation
// ---------------------------------------------------------------------------

/// Deterministic printable-ASCII text via LCG PRNG. Produces space-separated
/// "words" of 3-8 chars so pretokenization has realistic splits.
fn generateText(allocator: std.mem.Allocator, target_bytes: usize) ![]u8 {
    const buf = try allocator.alloc(u8, target_bytes);

    var state: u64 = 0xCAFE_BABE_DEAD_BEEF;
    var pos: usize = 0;
    while (pos < target_bytes) {
        // Word length 3-8
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const word_len: usize = 3 + @as(usize, @intCast((state >> 33) % 6));
        var w: usize = 0;
        while (w < word_len and pos < target_bytes) : (w += 1) {
            state = state *% 6364136223846793005 +% 1442695040888963407;
            // Printable ASCII a-z (26 chars)
            buf[pos] = @as(u8, @intCast((state >> 33) % 26)) + 'a';
            pos += 1;
        }
        // Space separator
        if (pos < target_bytes) {
            buf[pos] = ' ';
            pos += 1;
        }
    }
    return buf;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn readTokenizerJson(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const json = try allocator.alloc(u8, stat.size);
    const bytes_read = try file.readAll(json);
    if (bytes_read != stat.size) {
        allocator.free(json);
        return error.IncompleteRead;
    }
    return json;
}

fn loadTokenizer(allocator: std.mem.Allocator, json: []const u8) !tokenizer_mod.Tokenizer {
    return tokenizer_mod.Tokenizer.initFromJson(allocator, json);
}

fn elapsedNs(start: i128, end: i128) u64 {
    if (end <= start) return 0;
    return @intCast(end - start);
}

// ---------------------------------------------------------------------------
// Encode scenarios
// ---------------------------------------------------------------------------

fn runEncode(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    // Read tokenizer JSON
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);

    // Load tokenizer once
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();

    // Generate deterministic input text
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_token_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        const t0 = std.time.nanoTimestamp();
        const ids = try tok.encodeSlice(text);
        const t1 = std.time.nanoTimestamp();
        last_token_count = ids.len;
        tok.allocator.free(ids);

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = @intCast(text.len),
        .output_tokens = last_token_count,
        .note = "",
    };
}

pub fn runEncode1k(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runEncode(allocator, cfg, 1024, "encode_1k");
}

pub fn runEncode100k(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runEncode(allocator, cfg, 100 * 1024, "encode_100k");
}

pub fn runEncode1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runEncode(allocator, cfg, 1024 * 1024, "encode_1m");
}

// ---------------------------------------------------------------------------
// Load scenario
// ---------------------------------------------------------------------------

pub fn runLoadJson(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    // Read JSON once
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);

    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        const t0 = std.time.nanoTimestamp();
        var tok = try loadTokenizer(allocator, json);
        const t1 = std.time.nanoTimestamp();
        tok.deinit();

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "load_json",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = @intCast(json.len),
        .output_tokens = 0,
        .note = "tokenizer init from JSON",
    };
}
