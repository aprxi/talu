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
    text_file: []const u8 = "", // optional: read text from file instead of generating
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
    bpe_collect_1m,
    breakdown,
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

/// Load input text: from file if text_file is set, otherwise generate synthetic text.
fn loadInputText(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize) ![]u8 {
    if (cfg.text_file.len > 0) {
        const raw = try readTokenizerJson(allocator, cfg.text_file);
        if (raw.len <= target_bytes) return raw;
        // Truncate to target_bytes
        const result = try allocator.alloc(u8, target_bytes);
        @memcpy(result, raw[0..target_bytes]);
        allocator.free(raw);
        return result;
    }
    return generateText(allocator, target_bytes);
}

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
// BPE merge loop types (mirrors bpe.zig internals)
// ---------------------------------------------------------------------------

const BenchSymbol = struct {
    id: i32,
    start: u32,
    len: u32,
    prev: i16,
    next: i16,
};
const BENCH_MAX_SYMS = 512;

// Surviving symbol after merge (for collect benchmark)
const CollectSym = struct {
    id: i32,
    start: u32,
    len: u32,
};

/// Initialize symbols for a word (UTF-8 split + vocab_hash lookup).
fn initSymbols(model: *bpe_mod.BpeModel, word: []const u8, syms: *[BENCH_MAX_SYMS]BenchSymbol) usize {
    var n_syms: usize = 0;
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
    return n_syms;
}

/// Run the merge loop on initialized symbols (cached-pair approach).
/// Mirrors bpe.zig's encodeWord merge loop.
fn runMergeLoop(model: *bpe_mod.BpeModel, syms: *[BENCH_MAX_SYMS]BenchSymbol, n_syms: usize) u64 {
    var merge_count: u64 = 0;
    if (n_syms < 2) return 0;

    const CachedPair = struct { pos: i16, rank: i32, new_id: i32 };
    var pair_cache: [BENCH_MAX_SYMS]CachedPair = undefined;
    var n_cached: usize = 0;

    // Phase 1: Build pair cache
    {
        var si: i16 = 0;
        while (si >= 0) {
            const sym = &syms[@intCast(si)];
            if (sym.next >= 0 and sym.id >= 0) {
                const right_sym = &syms[@intCast(sym.next)];
                if (right_sym.id >= 0) {
                    if (model.pair_merges.get(.{ .left = sym.id, .right = right_sym.id })) |info| {
                        pair_cache[n_cached] = .{ .pos = si, .rank = info.rank, .new_id = info.new_id };
                        n_cached += 1;
                    }
                }
            }
            si = sym.next;
        }
    }

    // Phase 2: Process merges from cache
    while (n_cached > 0) {
        var best_idx: usize = 0;
        for (1..n_cached) |i| {
            if (pair_cache[i].rank < pair_cache[best_idx].rank or
                (pair_cache[i].rank == pair_cache[best_idx].rank and
                pair_cache[i].pos < pair_cache[best_idx].pos))
            {
                best_idx = i;
            }
        }
        const best = pair_cache[best_idx];

        n_cached -= 1;
        if (best_idx < n_cached) pair_cache[best_idx] = pair_cache[n_cached];

        const left = &syms[@intCast(best.pos)];
        if (left.id < 0 or left.next < 0) continue;
        const right_idx = left.next;
        const right = &syms[@intCast(right_idx)];
        if (right.id < 0) continue;

        left.id = best.new_id;
        left.len += right.len;
        left.next = right.next;
        if (right.next >= 0) {
            syms[@intCast(right.next)].prev = best.pos;
        }
        right.id = -2;
        merge_count += 1;

        // Remove stale entries
        {
            var ci: usize = 0;
            while (ci < n_cached) {
                if (pair_cache[ci].pos == right_idx or pair_cache[ci].pos == left.prev) {
                    n_cached -= 1;
                    if (ci < n_cached) pair_cache[ci] = pair_cache[n_cached];
                } else {
                    ci += 1;
                }
            }
        }

        // Add new pair: (left_neighbor, merged)
        if (left.prev >= 0) {
            const ln = &syms[@intCast(left.prev)];
            if (ln.id >= 0) {
                if (model.pair_merges.get(.{ .left = ln.id, .right = left.id })) |info| {
                    pair_cache[n_cached] = .{ .pos = left.prev, .rank = info.rank, .new_id = info.new_id };
                    n_cached += 1;
                }
            }
        }
        // Add new pair: (merged, right_neighbor)
        if (left.next >= 0) {
            const rn = &syms[@intCast(left.next)];
            if (rn.id >= 0) {
                if (model.pair_merges.get(.{ .left = left.id, .right = rn.id })) |info| {
                    pair_cache[n_cached] = .{ .pos = best.pos, .rank = info.rank, .new_id = info.new_id };
                    n_cached += 1;
                }
            }
        }
    }
    return merge_count;
}

// ---------------------------------------------------------------------------
// Encode scenarios
// ---------------------------------------------------------------------------

pub fn runEncode(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);

    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();

    const text = try loadInputText(allocator, cfg, target_bytes);
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
// Phase-level scenarios (parameterized by target_bytes)
// ---------------------------------------------------------------------------

pub fn runNormalize(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
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
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = 0,
        .note = "normalization only",
    };
}

pub fn runNormalize1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runNormalize(allocator, cfg, 1024 * 1024, "normalize_1m");
}

pub fn runPretok(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
    defer allocator.free(text);

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
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_word_count,
        .note = "pretokenization only",
    };
}

pub fn runPretok1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runPretok(allocator, cfg, 1024 * 1024, "pretok_1m");
}

pub fn runBpe(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
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
    var last_token_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        // Direct BPE path: shared IdList across all words (matches production)
        var shared_ids = bpe_mod.IdList{};
        defer shared_ids.deinit(model.allocator);
        const t0 = std.time.nanoTimestamp();
        for (words) |word| {
            model.encodeWordDirect(tok.tokenizer_handle, word, &shared_ids) catch continue;
        }
        const t1 = std.time.nanoTimestamp();
        last_token_count = shared_ids.items.len;

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
        .input_bytes = target_bytes,
        .output_tokens = last_token_count,
        .note = "BPE per-word (direct path)",
    };
}

pub fn runBpe1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runBpe(allocator, cfg, 1024 * 1024, "bpe_1m");
}

// ---------------------------------------------------------------------------
// Sub-BPE scenarios (isolate individual BPE methods)
// ---------------------------------------------------------------------------

/// vocab_hash.get() per UTF-8 char — measures symbol init cost.
/// Code under test: bpe.zig:618-641 (char split + vocab lookup)
pub fn runBpeVocab(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
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
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_lookup_count,
        .note = "vocab_hash lookups",
    };
}

pub fn runBpeVocab1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runBpeVocab(allocator, cfg, 1024 * 1024, "bpe_vocab_1m");
}

/// Merge loop only — symbols pre-computed outside timed section.
/// Code under test: bpe.zig:644-684 (pair_merges lookup + linked-list merge)
pub fn runBpeMerge(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
    defer allocator.free(text);

    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    const word_count = pretokenized.tokens.items.len;
    const words = try allocator.alloc([]const u8, word_count);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    const model: *bpe_mod.BpeModel = @ptrCast(@alignCast(tok.tokenizer_handle.model.?));

    // Pre-compute initial symbols for all words (outside timed section)
    var all_syms = std.ArrayListUnmanaged(BenchSymbol){};
    defer all_syms.deinit(allocator);
    var word_offsets = try allocator.alloc(usize, word_count + 1);
    defer allocator.free(word_offsets);

    for (words, 0..) |word, wi| {
        word_offsets[wi] = all_syms.items.len;
        var syms: [BENCH_MAX_SYMS]BenchSymbol = undefined;
        const n = initSymbols(model, word, &syms);
        try all_syms.appendSlice(allocator, syms[0..n]);
    }
    word_offsets[word_count] = all_syms.items.len;

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
        for (0..word_count) |wi| {
            const start = word_offsets[wi];
            const end = word_offsets[wi + 1];
            const n_syms = end - start;
            if (n_syms == 0) continue;

            // Copy pre-computed symbols to stack for this iteration
            var syms: [BENCH_MAX_SYMS]BenchSymbol = undefined;
            @memcpy(syms[0..n_syms], all_syms.items[start..end]);

            total_merges += runMergeLoop(model, &syms, n_syms);
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
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_merge_count,
        .note = "merge loop only",
    };
}

pub fn runBpeMerge1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runBpeMerge(allocator, cfg, 1024 * 1024, "bpe_merge_1m");
}

/// Real result collection — ArrayList + dupeZ + toOwnedSlice + free.
/// Merged symbols pre-computed outside timed section.
/// Code under test: bpe.zig:737-766 (result collection + allocation)
pub fn runBpeCollect(allocator: std.mem.Allocator, cfg: RunConfig, target_bytes: usize, name: []const u8) !ScenarioResult {
    const json = try readTokenizerJson(allocator, cfg.tokenizer_json_path);
    defer allocator.free(json);
    var tok = try loadTokenizer(allocator, json);
    defer tok.deinit();
    const text = try loadInputText(allocator, cfg, target_bytes);
    defer allocator.free(text);

    var normalized = try norm_mod.normalize_text(&tok.tokenizer_handle.normalizer, text);
    defer normalized.deinit();
    var pretokenized = try pretok_mod.pretokenize(
        &tok.tokenizer_handle.pretokenizer,
        normalized.text,
        .{ .start = 0, .end = normalized.text.len },
    );
    defer pretokenized.deinit();

    const word_count = pretokenized.tokens.items.len;
    const words = try allocator.alloc([]const u8, word_count);
    defer allocator.free(words);
    for (pretokenized.tokens.items, 0..) |token, i| {
        words[i] = token.sliceConst();
    }

    const model: *bpe_mod.BpeModel = @ptrCast(@alignCast(tok.tokenizer_handle.model.?));
    const bpe_allocator = model.allocator;

    // Pre-compute merged symbols for all words (init + merge, outside timed section)
    var all_collect = std.ArrayListUnmanaged(CollectSym){};
    defer all_collect.deinit(allocator);
    var collect_offsets = try allocator.alloc(usize, word_count + 1);
    defer allocator.free(collect_offsets);

    for (words, 0..) |word, wi| {
        collect_offsets[wi] = all_collect.items.len;

        var syms: [BENCH_MAX_SYMS]BenchSymbol = undefined;
        const n_syms = initSymbols(model, word, &syms);
        _ = runMergeLoop(model, &syms, n_syms);

        // Walk surviving symbols
        var si: i16 = 0;
        while (si >= 0) {
            const sym = &syms[@intCast(si)];
            if (sym.id >= 0) {
                try all_collect.append(allocator, .{
                    .id = sym.id,
                    .start = sym.start,
                    .len = sym.len,
                });
            }
            si = sym.next;
        }
    }
    collect_offsets[word_count] = all_collect.items.len;

    const total_iters = cfg.warmup + cfg.iters;
    const samples = try allocator.alloc(harness.Sample, cfg.iters);
    errdefer allocator.free(samples);
    var cold_first: harness.Sample = .{ .encode_ns = 0 };
    var sample_idx: usize = 0;
    var last_token_count: u64 = 0;

    var iter: usize = 0;
    while (iter < total_iters) : (iter += 1) {
        var token_count: u64 = 0;
        // Shared IdList across all words (matches production direct path)
        var shared_ids = bpe_mod.IdList{};
        defer shared_ids.deinit(bpe_allocator);
        const t0 = std.time.nanoTimestamp();
        for (0..word_count) |wi| {
            const cs = collect_offsets[wi];
            const ce = collect_offsets[wi + 1];

            for (all_collect.items[cs..ce]) |csym| {
                shared_ids.append(bpe_allocator, csym.id) catch continue;
            }

            token_count += ce - cs;
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
        .name = name,
        .profile = cfg.profile,
        .samples = samples,
        .cold_first = cold_first,
        .input_bytes = target_bytes,
        .output_tokens = last_token_count,
        .note = "result collection",
    };
}

pub fn runBpeCollect1m(allocator: std.mem.Allocator, cfg: RunConfig) !ScenarioResult {
    return runBpeCollect(allocator, cfg, 1024 * 1024, "bpe_collect_1m");
}
