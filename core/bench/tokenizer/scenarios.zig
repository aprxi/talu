const std = @import("std");
const core_main = @import("main");
const tokenizer_mod = core_main.tokenizer;
const harness = @import("harness.zig");

// Internal tokenizer modules for phase-level benchmarking
const ct = tokenizer_mod.c_types;
const norm_mod = tokenizer_mod.pipeline.normalize;
const pretok_mod = tokenizer_mod.pipeline.pretokenize;
const encode_mod = tokenizer_mod.pipeline.encode;
const bpe_mod = tokenizer_mod.bpe;

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
    normalize_1m,
    pretok_1m,
    bpe_1m,
    bpe_merge_1m,
    bpe_vocab_1m,
    bpe_alloc_1m,
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

// ---------------------------------------------------------------------------
// Phase-level scenarios (isolate individual pipeline stages)
// ---------------------------------------------------------------------------

pub fn runNormalize1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        const t0 = std.time.nanoTimestamp();
        var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
        const t1 = std.time.nanoTimestamp();
        normalized.deinit();

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "normalize_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = 0,
        .note = "normalization only",
    };
}

pub fn runPretok1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    // Normalize once outside the timed loop
    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_word_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        const t0 = std.time.nanoTimestamp();
        var pretokenized = try pretok_mod.pretokenize(
            &tok.tokenizer_handle.pretokenizer,
            normalized.text,
            .{ .start = 0, .end = normalized.text.len },
        );
        const t1 = std.time.nanoTimestamp();
        last_word_count = pretokenized.tokens.items.len;
        pretokenized.deinit();

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "pretok_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_word_count,
        .note = "pretokenization only",
    };
}

pub fn runBpe1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    // Normalize + pretokenize once outside the timed loop
    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    // Collect word slices for stable iteration
    const words = try allocator.alloc([]const u8, pretokenized.tokens.items.len);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_token_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        var token_count: u64 = 0;
        const t0 = std.time.nanoTimestamp();
        for (words) |word| {
            var encoding = std.mem.zeroes(ct.TokenizerEncoding);
            const rc = tok.tokenizer_handle.encodeSlice(word, &encoding);
            if (rc == 0) token_count += encoding.ids_len;
            encode_mod.tokenizer_encoding_free_struct(&encoding);
        }
        const t1 = std.time.nanoTimestamp();
        last_token_count = token_count;

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "bpe_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_token_count,
        .note = "BPE per-word only",
    };
}

// ---------------------------------------------------------------------------
// Sub-BPE scenarios (isolate BPE internals)
// ---------------------------------------------------------------------------

// Symbol for the merge loop benchmark (mirrors bpe.zig's Symbol layout)
const BenchSymbol = struct {
    id: i32,
    start: u32,
    len: u32,
    prev: i16,
    next: i16,
};
const BENCH_MAX_SYMS = 512;

/// Replicates the BPE merge loop from bpe.zig (symbol init + merge).
/// No result collection or allocation — pure computation + hash lookups.
/// Reports total merge operations as output_tokens for cost-per-merge analysis.
pub fn runBpeMerge1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    const words = try allocator.alloc([]const u8, pretokenized.tokens.items.len);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    const model: *bpe_mod.BpeModel = @ptrCast(@alignCast(tok.tokenizer_handle.model.?));

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_merge_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        var total_merges: u64 = 0;
        const t0 = std.time.nanoTimestamp();
        for (words) |word| {
            // 1. Symbol init (bpe.zig lines 618-641)
            var syms: [BENCH_MAX_SYMS]BenchSymbol = undefined;
            var n_syms: usize = 0;
            {
                var byte_idx: usize = 0;
                while (byte_idx < word.len) {
                    if (n_syms >= BENCH_MAX_SYMS) break;
                    const char_len = std.unicode.utf8ByteSequenceLength(word[byte_idx]) catch 1;
                    const end = @min(byte_idx + char_len, word.len);
                    const id = model.vocab_hash.get(word[byte_idx..end]) orelse -1;
                    syms[n_syms] = .{
                        .id = id,
                        .start = @intCast(byte_idx),
                        .len = @intCast(end - byte_idx),
                        .prev = if (n_syms > 0) @as(i16, @intCast(n_syms - 1)) else -1,
                        .next = -1,
                    };
                    if (n_syms > 0) syms[n_syms - 1].next = @intCast(n_syms);
                    byte_idx = end;
                    n_syms += 1;
                }
            }

            // 2. Merge loop (bpe.zig lines 644-684)
            if (n_syms >= 2) {
                while (true) {
                    var best_rank: i32 = std.math.maxInt(i32);
                    var best_pos: i16 = -1;

                    var si: i16 = 0;
                    while (si >= 0) {
                        const sym = &syms[@intCast(si)];
                        if (sym.next >= 0 and sym.id >= 0) {
                            const right_sym = &syms[@intCast(sym.next)];
                            if (right_sym.id >= 0) {
                                if (model.pair_merges.get(.{ .left = sym.id, .right = right_sym.id })) |info| {
                                    if (info.rank < best_rank) {
                                        best_rank = info.rank;
                                        best_pos = si;
                                    }
                                }
                            }
                        }
                        si = sym.next;
                    }

                    if (best_pos < 0) break;

                    const left = &syms[@intCast(best_pos)];
                    const right_idx = left.next;
                    const right = &syms[@intCast(right_idx)];
                    const info = model.pair_merges.get(.{ .left = left.id, .right = right.id }).?;
                    left.id = info.new_id;
                    left.len += right.len;
                    left.next = right.next;
                    if (right.next >= 0) {
                        syms[@intCast(right.next)].prev = best_pos;
                    }
                    right.id = -2;
                    total_merges += 1;
                }
            }
        }
        const t1 = std.time.nanoTimestamp();
        last_merge_count = total_merges;

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "bpe_merge_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_merge_count,
        .note = "init+merge, no alloc",
    };
}

/// Measures vocab_hash.get() throughput: for each word, split into UTF-8
/// chars and look up each in the BPE model's vocabulary hash map.
pub fn runBpeVocab1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    const words = try allocator.alloc([]const u8, pretokenized.tokens.items.len);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    const model: *bpe_mod.BpeModel = @ptrCast(@alignCast(tok.tokenizer_handle.model.?));

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_lookup_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        var lookup_count: u64 = 0;
        const t0 = std.time.nanoTimestamp();
        for (words) |word| {
            var byte_idx: usize = 0;
            while (byte_idx < word.len) {
                const char_len = std.unicode.utf8ByteSequenceLength(word[byte_idx]) catch 1;
                const end = @min(byte_idx + char_len, word.len);
                _ = model.vocab_hash.get(word[byte_idx..end]);
                lookup_count += 1;
                byte_idx = end;
            }
        }
        const t1 = std.time.nanoTimestamp();
        last_lookup_count = lookup_count;

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "bpe_vocab_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_lookup_count,
        .note = "vocab_hash lookups only",
    };
}

/// Measures allocator throughput for the BPE result-collection pattern:
/// per word, dupeZ the word string + alloc/free an i32 slice.
/// Isolates allocation cost from hash lookup and merge computation.
pub fn runBpeAlloc1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    const target_bytes: usize = 1024 * 1024;
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try generateText(allocator, target_bytes);
    defer allocator.free(text);

    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    const words = try allocator.alloc([]const u8, pretokenized.tokens.items.len);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    // Use the model's actual allocator (c_allocator) to match real BPE behavior
    const model: *bpe_mod.BpeModel = @ptrCast(@alignCast(tok.tokenizer_handle.model.?));
    const bpe_allocator = model.allocator;

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_alloc_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        var alloc_count: u64 = 0;
        const t0 = std.time.nanoTimestamp();
        for (words) |word| {
            // Simulate BPE result collection: ids slice + token string dupeZ
            const ids = try bpe_allocator.alloc(i32, 1);
            const dup = try bpe_allocator.dupeZ(u8, word);
            bpe_allocator.free(std.mem.sliceTo(dup, 0));
            bpe_allocator.free(ids);
            alloc_count += 1;
        }
        const t1 = std.time.nanoTimestamp();
        last_alloc_count = alloc_count;

        const sample = harness.Sample{ .encode_ns = elapsedNs(t0, t1) };
        if (iter == 0) cold_first = sample;
        if (iter >= cfg.warmup and sample_idx < samples.len) {
            samples[sample_idx] = sample;
            sample_idx += 1;
        }
    }

    return .{
        .name = "bpe_alloc_1m",
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_alloc_count,
        .note = "alloc pattern only",
    };
}
