//! Tokenizer Loader
//!
//! Loads tokenizer configurations from JSON files or strings.
//! Supports HuggingFace tokenizer.json format with fast vocab/merges parsing.

const std = @import("std");
const schema = @import("schema.zig");
const loader_model_json = @import("loader_model_json.zig");
const loader_pipeline_json = @import("loader_pipeline_json.zig");
const json_utils = @import("json_utils.zig");
const wordpiece_model = @import("wordpiece.zig");
const unigram_model = @import("unigram.zig");
const utils = @import("utils.zig");
const ct = @import("c_types.zig");
const tok_fns = @import("pipeline.zig");
const log = @import("log_pkg");

const ManagedArrayList = std.array_list.Managed;

const parseMetadataSections = loader_pipeline_json.parseMetadataSections;
const ensureJsonDepthWithinLimit = loader_pipeline_json.ensureJsonDepthWithinLimit;
const parseNormalizer = loader_pipeline_json.parseNormalizer;
const parsePreTokenizer = loader_pipeline_json.parsePreTokenizer;
const parsePostProcessor = loader_pipeline_json.parsePostProcessor;
const parseDecoder = loader_pipeline_json.parseDecoder;
const MAX_JSON_PIPELINE_DEPTH = loader_pipeline_json.MAX_JSON_PIPELINE_DEPTH;
const validateModelOptions = loader_model_json.validateModelOptions;
const validateBpeMergeReferences = loader_model_json.validateBpeMergeReferences;
const findQuotedString = loader_model_json.findQuotedString;
const parseMergesFastSection = loader_model_json.parseMergesFastSection;
const parseVocabFastSection = loader_model_json.parseVocabFastSection;
const unescapeJsonStringFast = loader_model_json.unescapeJsonStringFast;
const findSection = utils.findJsonSection;
const findMatchingBrace = utils.findMatchingBrace;
const findJsonFieldValue = json_utils.findJsonFieldValue;
const findJsonFieldString = json_utils.findJsonFieldString;
const findJsonFieldArrayString = json_utils.findJsonFieldArrayString;

// -------------------- Streaming Loader (hot path) --------------------

/// Fast tokenizer loader - uses direct scanning for vocab/merges (the heavy parts)
/// Falls back to std.json only for small metadata sections
pub fn parseTokenizerJson(allocator: std.mem.Allocator, json_content: []const u8) !schema.TokenizerRoot {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const model_section = findSection(json_content, "\"model\"") orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_missing",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    };
    if (model_section.len == 0 or model_section[0] != '{') {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_not_object",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    }
    const model_end = findMatchingBrace(model_section, '{', '}') orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_section_unbalanced",
            .json_bytes = json_content.len,
        });
        return error.InvalidModel;
    };
    const model_json = model_section[0..model_end];
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{ .stage = "model_section", .model_bytes = model_json.len }, @src());

    // Fast path: directly scan for vocab and merges sections
    var vocab_entries = ManagedArrayList(schema.TokenId).init(arena_allocator);
    var merge_entries = ManagedArrayList([]const u8).init(arena_allocator);
    const model_type_name = findJsonFieldString(model_json, "\"type\"") orelse {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_type_missing",
            .model_bytes = model_json.len,
        });
        return error.InvalidModel;
    };
    if (!std.mem.eql(u8, model_type_name, "BPE") and
        !std.mem.eql(u8, model_type_name, "WordPiece") and
        !std.mem.eql(u8, model_type_name, "Unigram"))
    {
        log.warn("tokenizer", "Tokenizer model parse failed", .{
            .stage = "model_type_unsupported",
            .model_type = model_type_name,
        });
        return error.UnsupportedModel;
    }
    var vocab_is_array = false;

    // Find and parse vocab section directly
    if (findSection(model_json, "\"vocab\"")) |vocab_section| {
        if (vocab_section.len > 0 and vocab_section[0] == '{') {
            // Find matching closing brace
            const vocab_end = findMatchingBrace(vocab_section, '{', '}') orelse vocab_section.len;
            try parseVocabFastSection(arena_allocator, vocab_section[0..vocab_end], &vocab_entries);
        } else if (vocab_section.len > 0 and vocab_section[0] == '[') {
            vocab_is_array = true;
            // Unigram array format: [["token", score], ...]
            // Use std.json Scanner for correctness (this path is rare)
            const vocab_end = findMatchingBrace(vocab_section, '[', ']') orelse vocab_section.len;
            const vocab_json = vocab_section[0 .. vocab_end + 1];
            var scanner = std.json.Scanner.initCompleteInput(arena_allocator, vocab_json);
            if ((try scanner.next()) == .array_begin) {
                var next_id: i32 = 0;
                while (true) {
                    const tok = try scanner.next();
                    if (tok == .array_end) break;
                    if (tok != .array_begin) return error.InvalidVocab;
                    // Inner array: ["token", score]
                    const token_tok = try scanner.nextAlloc(arena_allocator, .alloc_if_needed);
                    const token_str: []const u8 = switch (token_tok) {
                        .string => |s| s,
                        .allocated_string => |s| s,
                        else => return error.InvalidVocab,
                    };
                    if (token_str.len == 0 or std.mem.indexOfScalar(u8, token_str, 0) != null) return error.InvalidVocab;
                    const score_tok = try scanner.next();
                    const score: f32 = switch (score_tok) {
                        .number => |n| std.fmt.parseFloat(f32, n) catch return error.InvalidVocab,
                        else => return error.InvalidVocab,
                    };
                    if ((try scanner.next()) != .array_end) return error.InvalidVocab;
                    try vocab_entries.append(.{ .token = token_str, .id = next_id, .score = score });
                    next_id += 1;
                }
            } else {
                return error.InvalidVocab;
            }
        } else {
            return error.InvalidVocab;
        }
    } else {
        return error.InvalidVocab;
    }

    // Find and parse merges section directly
    if (findSection(model_json, "\"merges\"")) |merges_section| {
        if (merges_section.len > 0 and merges_section[0] == '[') {
            const merges_end = findMatchingBrace(merges_section, '[', ']') orelse merges_section.len;
            try parseMergesFastSection(arena_allocator, merges_section[0..merges_end], &merge_entries);
        } else {
            return error.InvalidMerges;
        }
    }
    if (std.mem.eql(u8, model_type_name, "WordPiece") and vocab_is_array) return error.InvalidVocab;
    if (std.mem.eql(u8, model_type_name, "Unigram") and !vocab_is_array) return error.InvalidVocab;
    if (std.mem.eql(u8, model_type_name, "BPE") and vocab_is_array) return error.InvalidVocab;
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{
        .stage = "model_core_parsed",
        .model_type = model_type_name,
        .vocab_entries = vocab_entries.items.len,
        .merge_entries = merge_entries.items.len,
    }, @src());
    try validateModelOptions(model_type_name, model_json);
    try validateBpeMergeReferences(arena_allocator, model_type_name, vocab_entries.items, merge_entries.items);
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{ .stage = "model_validated" }, @src());

    // Extract unk_token from model section
    var unk_token_str: ?[]const u8 = null;
    const search_len = @min(500, model_json.len);
    const search_region = model_json[0..search_len];
    // Try "unk_token" (string value) — used by WordPiece
    if (std.mem.indexOf(u8, search_region, "\"unk_token\"")) |unk_pos| {
        const after_key = model_json[unk_pos + "\"unk_token\"".len ..];
        if (findQuotedString(after_key)) |val| {
            unk_token_str = val;
        }
    }
    // Try "unk_id" (integer index into vocab) — used by Unigram
    if (unk_token_str == null) {
        if (std.mem.indexOf(u8, search_region, "\"unk_id\"")) |unk_pos| {
            const after_key = model_json[unk_pos + "\"unk_id\"".len ..];
            // Skip colon and whitespace to find the integer
            var skip: usize = 0;
            while (skip < after_key.len and (after_key[skip] == ':' or after_key[skip] == ' ' or after_key[skip] == '\t')) : (skip += 1) {}
            if (skip < after_key.len) {
                const num_start = skip;
                while (skip < after_key.len and after_key[skip] >= '0' and after_key[skip] <= '9') : (skip += 1) {}
                if (skip > num_start) {
                    const unk_id = std.fmt.parseInt(usize, after_key[num_start..skip], 10) catch null;
                    if (unk_id) |id| {
                        if (id < vocab_entries.items.len) {
                            unk_token_str = vocab_entries.items[id].token;
                        }
                    }
                }
            }
        }
    }

    // Extract max_input_chars_per_word from model section (used by WordPiece)
    var max_input_chars_per_word: i32 = 200;
    const mic_search_len = @min(2000, model_json.len);
    if (std.mem.indexOf(u8, model_json[0..mic_search_len], "\"max_input_chars_per_word\"")) |mic_pos| {
        const after_key = model_json[mic_pos + "\"max_input_chars_per_word\"".len ..];
        var skip: usize = 0;
        while (skip < after_key.len and (after_key[skip] == ':' or after_key[skip] == ' ' or after_key[skip] == '\t')) : (skip += 1) {}
        if (skip < after_key.len) {
            const num_start = skip;
            while (skip < after_key.len and after_key[skip] >= '0' and after_key[skip] <= '9') : (skip += 1) {}
            if (skip > num_start) {
                max_input_chars_per_word = std.fmt.parseInt(i32, after_key[num_start..skip], 10) catch 200;
            }
        }
    }

    // Parse small sections with std.json (added_tokens, normalizer, etc.)
    var added_token_entries = ManagedArrayList(schema.AddedToken).init(arena_allocator);
    var normalizer_spec: schema.Normalizer = .{};
    var pretokenizer_spec: schema.PreTokenizer = .{};
    var postprocessor_spec: schema.PostProcessor = .{};
    var decoder_spec: schema.Decoder = .{};

    try parseMetadataSections(
        arena_allocator,
        json_content,
        &added_token_entries,
        &normalizer_spec,
        &pretokenizer_spec,
        &postprocessor_spec,
        &decoder_spec,
    );
    log.debug("tokenizer", "Tokenizer fast-parse stage", .{
        .stage = "metadata_parsed",
        .added_tokens = added_token_entries.items.len,
    }, @src());

    var vocab_by_id = std.AutoHashMap(i32, []const u8).init(arena_allocator);
    defer vocab_by_id.deinit();
    for (vocab_entries.items) |entry| {
        try vocab_by_id.put(entry.id, entry.token);
    }
    for (added_token_entries.items) |entry| {
        const vocab_token = vocab_by_id.get(entry.id) orelse continue;
        if (!std.mem.eql(u8, vocab_token, entry.content)) return error.InvalidAdded;
    }

    log.trace("tokenizer", "Parsed vocab and merges", .{
        .vocab_entries = vocab_entries.items.len,
        .merge_entries = merge_entries.items.len,
    }, @src());

    return schema.TokenizerRoot{
        .version = null,
        .model = .{
            .type = model_type_name,
            .vocab = try vocab_entries.toOwnedSlice(),
            .merges = if (merge_entries.items.len > 0) try merge_entries.toOwnedSlice() else null,
            .unk_token = unk_token_str,
            .bos_token = null,
            .eos_token = null,
            .max_input_chars_per_word = max_input_chars_per_word,
        },
        .added_tokens = try added_token_entries.toOwnedSlice(),
        .normalizer = normalizer_spec,
        .pre_tokenizer = pretokenizer_spec,
        .post_processor = postprocessor_spec,
        .decoder = decoder_spec,
    };
}

// -------------------- Exports retained for C API --------------------

const LazyBpeRawPostProcessorMode = enum {
    raw_json,
    root_for_noop_sequence,
};

fn rootPostProcessorSchemaIsNoopSequence(root: schema.TokenizerRoot) bool {
    const postprocessor = root.post_processor;
    return std.mem.eql(u8, postprocessor.type, "Sequence") and
        !postprocessor.add_special and
        !postprocessor.pair and
        postprocessor.cls_token == null and
        postprocessor.sep_token == null;
}

fn findJsonObjectSection(json_bytes: []const u8, key: []const u8) ?[]const u8 {
    const section = findSection(json_bytes, key) orelse return null;
    if (section.len == 0 or section[0] != '{') return null;
    const end = findMatchingBrace(section, '{', '}') orelse section.len;
    return section[0..end];
}

fn applyLazyBpePostProcessorFromRawJson(
    arena: *std.heap.ArenaAllocator,
    tokenizer: *ct.Tokenizer,
    root: schema.TokenizerRoot,
    json_bytes: []const u8,
    mode: LazyBpeRawPostProcessorMode,
) !void {
    if (mode == .root_for_noop_sequence and rootPostProcessorSchemaIsNoopSequence(root)) {
        try applyRootPostProcessorMetadata(arena, tokenizer, root);
        return;
    }

    // schema.PostProcessor stores flattened metadata for direct model builders.
    // Lazy BPE reparses the raw object so nested Sequence processors are not
    // dropped before TemplateProcessing/BertProcessing metadata is applied.
    const postprocessor_json = findJsonObjectSection(json_bytes, "\"post_processor\"") orelse return;
    try applyPostProcessorFromJson(tokenizer, postprocessor_json);
}

fn buildLazyBpeTokenizerFromOwnedJson(
    allocator: std.mem.Allocator,
    arena: *std.heap.ArenaAllocator,
    root: schema.TokenizerRoot,
    json_bytes: []u8,
    postprocessor_mode: LazyBpeRawPostProcessorMode,
) !*ct.Tokenizer {
    const tokenizer = ct.Tokenizer.initBpe(allocator, json_bytes, true) catch |err| {
        allocator.free(json_bytes);
        return err;
    };
    errdefer tok_fns.tokenizer_free(tokenizer);

    try applyRootMetadataWithoutPostProcessor(arena, tokenizer, root);
    try applyLazyBpePostProcessorFromRawJson(arena, tokenizer, root, json_bytes, postprocessor_mode);
    applyRootDecoderMetadata(tokenizer, root);
    return tokenizer;
}

pub fn tokenizer_loader_from_json_string(json_data: ?[*:0]const u8) ?*ct.Tokenizer {
    const json_ptr = json_data orelse return null;
    const json_bytes = std.mem.sliceTo(json_ptr, 0);
    const allocator = std.heap.c_allocator;

    // Validate and parse once through the strict streaming loader first.
    // BPE still uses the lazy runtime model after validation to preserve
    // its encode-time behavior; WordPiece/Unigram build directly from root.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = parseTokenizerJson(arena.allocator(), json_bytes) catch |err| {
        log.warn("tokenizer", "Tokenizer JSON parse failed", .{
            .reason = @errorName(err),
            .json_bytes = json_bytes.len,
        });
        return null;
    };
    const model_type_name = root.model.type;

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader (same path as file-based loading)
        // Need to copy the JSON since lazy loader may keep references to it
        const json_copy = allocator.dupeZ(u8, json_bytes) catch |err| {
            log.warn("tokenizer", "Tokenizer JSON copy failed", .{
                .reason = @errorName(err),
                .json_bytes = json_bytes.len,
                .model_type = model_type_name,
            });
            return null;
        };

        return buildLazyBpeTokenizerFromOwnedJson(
            allocator,
            &arena,
            root,
            json_copy,
            .root_for_noop_sequence,
        ) catch |err| {
            log.warn("tokenizer", "Lazy BPE tokenizer build failed", .{
                .reason = @errorName(err),
                .json_bytes = json_bytes.len,
                .model_type = model_type_name,
            });
            return null;
        };
    }

    return buildTokenizerFromRoot(&arena, root) catch |err| {
        log.warn("tokenizer", "Tokenizer model build failed", .{
            .reason = @errorName(err),
            .model_type = model_type_name,
        });
        return null;
    };
}

/// Find the snapshot directory for cache model layout (models--org--name/snapshots/)
fn findSnapshotDir(ally: std.mem.Allocator, base_path: []const u8) ?[]const u8 {
    const snapshots_path = std.fs.path.join(ally, &.{ base_path, "snapshots" }) catch return null;
    defer ally.free(snapshots_path);

    // First try refs/main to get the canonical revision
    const refs_main_path = std.fs.path.join(ally, &.{ base_path, "refs", "main" }) catch return null;
    defer ally.free(refs_main_path);

    if (std.fs.cwd().openFile(refs_main_path, .{})) |file| {
        defer file.close();
        var rev_buf: [256]u8 = undefined;
        const read_len = file.read(&rev_buf) catch 0;
        if (read_len > 0) {
            // Trim whitespace/newlines
            var end = read_len;
            while (end > 0 and (rev_buf[end - 1] == '\n' or rev_buf[end - 1] == '\r' or rev_buf[end - 1] == ' ')) {
                end -= 1;
            }
            if (end > 0) {
                const rev = rev_buf[0..end];
                const candidate = std.fs.path.join(ally, &.{ snapshots_path, rev }) catch return null;
                // Check if directory exists using access
                if (std.fs.cwd().access(candidate, .{})) |_| {
                    return candidate;
                } else |_| {
                    ally.free(candidate);
                }
            }
        }
    } else |_| {}

    // Not in cache - iterate snapshots directory (not available on WASM/Emscripten)
    const builtin = @import("builtin");
    if (comptime builtin.target.os.tag == .emscripten or builtin.target.os.tag == .wasi) {
        return null;
    }

    var snapshots_dir = std.fs.cwd().openDir(snapshots_path, .{ .iterate = true }) catch return null;
    defer snapshots_dir.close();

    var iter = snapshots_dir.iterate();
    while (iter.next() catch null) |entry| {
        if (entry.kind == .directory) {
            const candidate = std.fs.path.join(ally, &.{ snapshots_path, entry.name }) catch continue;
            return candidate;
        }
    }

    return null;
}

pub fn tokenizer_loader_from_dir(path: ?[*:0]const u8) ?*ct.Tokenizer {
    const path_ptr = path orelse return null;
    const dir_bytes = std.mem.sliceTo(path_ptr, 0);
    const allocator = std.heap.c_allocator;
    var t_start: i128 = std.time.nanoTimestamp();

    var path_buf: [512]u8 = undefined; // filled by bufPrint before use
    const json_path = blk: {
        // If path already ends with tokenizer.json, use it directly
        if (std.mem.endsWith(u8, dir_bytes, "tokenizer.json")) {
            if (std.fs.cwd().access(dir_bytes, .{})) |_| {
                break :blk dir_bytes;
            } else |_| {}
        }

        // Try direct path first
        const direct_len = std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{dir_bytes}) catch return null;
        if (std.fs.cwd().access(path_buf[0..direct_len.len], .{})) |_| {
            break :blk path_buf[0..direct_len.len];
        } else |_| {}

        // Try cache snapshot layout
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        if (findSnapshotDir(arena.allocator(), dir_bytes)) |snapshot_dir| {
            const snap_len = std.fmt.bufPrint(&path_buf, "{s}/tokenizer.json", .{snapshot_dir}) catch return null;
            break :blk path_buf[0..snap_len.len];
        }
        return null;
    };

    // Read JSON file
    var file = std.fs.cwd().openFile(json_path, .{}) catch return null;
    defer file.close();
    const stat = file.stat() catch return null;
    const json_len: usize = @intCast(stat.size);
    const json_bytes = allocator.alloc(u8, json_len) catch return null;
    errdefer allocator.free(json_bytes);
    const bytes_read = file.readAll(json_bytes) catch {
        allocator.free(json_bytes);
        return null;
    };
    if (bytes_read != json_len) {
        allocator.free(json_bytes);
        return null;
    }

    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Read JSON", .{ .size_kb = json_len / 1024, .duration_ms = duration_ms }, @src());
        t_start = now;
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const root = parseTokenizerJson(arena.allocator(), json_bytes) catch |err| {
        log.warn("tokenizer", "Tokenizer JSON parse failed", .{
            .reason = @errorName(err),
            .json_path = json_path,
            .json_bytes = json_bytes.len,
        });
        allocator.free(json_bytes);
        return null;
    };
    const model_type_name = root.model.type;

    if (std.mem.eql(u8, model_type_name, "BPE")) {
        // Use lazy BPE loader - defers vocab/merges parsing until first encode
        const tokenizer = buildLazyBpeTokenizerFromOwnedJson(
            allocator,
            &arena,
            root,
            json_bytes,
            .raw_json,
        ) catch |err| {
            log.warn("tokenizer", "Lazy BPE tokenizer build failed", .{
                .reason = @errorName(err),
                .json_path = json_path,
                .json_bytes = json_bytes.len,
                .model_type = model_type_name,
            });
            return null;
        };
        {
            const now = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
            log.debug("tokenizer", "Build lazy BPE", .{ .duration_ms = duration_ms }, @src());
        }
        return @ptrCast(tokenizer);
    }

    // json_bytes must stay alive until after buildTokenizerFromRoot — the
    // parsed root struct contains slices that reference the original buffer
    // from std.json Scanner's .alloc_if_needed path.
    defer allocator.free(json_bytes);
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Parse JSON", .{ .duration_ms = duration_ms }, @src());
        t_start = now;
    }
    const result = buildTokenizerFromRoot(&arena, root) catch |err| {
        log.warn("tokenizer", "Tokenizer model build failed", .{
            .reason = @errorName(err),
            .json_path = json_path,
            .model_type = model_type_name,
        });
        return null;
    };
    {
        const now = std.time.nanoTimestamp();
        const duration_ms = @as(f64, @floatFromInt(now - t_start)) / 1_000_000.0;
        log.debug("tokenizer", "Build model", .{ .duration_ms = duration_ms }, @src());
    }
    return result;
}

/// Find a field value (number/bool) in JSON object
/// Apply post_processor settings from JSON for the fast-path BPE loader.
/// Parses type, cls/sep tokens (BertProcessing format), and resolves IDs
/// from the added_tokens list.
fn applyPostProcessorFromJson(tokenizer: *ct.Tokenizer, json_bytes: []const u8) !void {
    try ensureJsonDepthWithinLimit(json_bytes, error.InvalidPostProcessor);
    return applyPostProcessorFromJsonDepth(tokenizer, json_bytes, 0);
}

fn applyPostProcessorFromJsonDepth(tokenizer: *ct.Tokenizer, json_bytes: []const u8, depth: usize) !void {
    if (depth > MAX_JSON_PIPELINE_DEPTH) return error.InvalidPostProcessor;
    const type_str = findJsonFieldString(json_bytes, "\"type\"") orelse return;

    // Sequence type: iterate processors array and apply each sub-processor
    if (std.mem.eql(u8, type_str, "Sequence")) {
        if (findSection(json_bytes, "\"processors\"")) |arr_section| {
            if (arr_section.len > 0 and arr_section[0] == '[') {
                const arr_end = findMatchingBrace(arr_section, '[', ']') orelse arr_section.len;
                const arr_content = arr_section[0..arr_end];
                var cursor: usize = 0;
                while (cursor < arr_content.len) {
                    while (cursor < arr_content.len and arr_content[cursor] != '{') : (cursor += 1) {}
                    if (cursor >= arr_content.len) break;
                    const obj_end = if (findMatchingBrace(arr_content[cursor..], '{', '}')) |len| cursor + len else break;
                    try applyPostProcessorFromJsonDepth(tokenizer, arr_content[cursor..obj_end], depth + 1);
                    cursor = obj_end;
                }
            }
        }
        return;
    }

    // Set add_special for known post-processor types
    if (std.mem.eql(u8, type_str, "BertProcessing") or
        std.mem.eql(u8, type_str, "TemplateProcessing"))
    {
        tokenizer.postproc.add_special = 1;
    } else if (std.mem.eql(u8, type_str, "RobertaProcessing")) {
        tokenizer.postproc.add_special = 1;
        tokenizer.postproc.pair = 1;
    } else {
        return;
    }

    // For BertProcessing: parse cls/sep from ["[CLS]", 1] arrays
    // For TemplateProcessing: parse from special_tokens map
    // In both cases, try to extract token strings
    var cls_str: ?[]const u8 = null;
    var sep_str: ?[]const u8 = null;

    if (std.mem.eql(u8, type_str, "BertProcessing")) {
        // BertProcessing stores cls/sep as ["token_string", id] arrays.
        // BertProcessing stores arrays in tokenizer.json; accept the plain
        // string shape as a secondary valid encoding.
        cls_str = findJsonFieldArrayString(json_bytes, "\"cls\"") orelse
            findJsonFieldString(json_bytes, "\"cls\"");
        sep_str = findJsonFieldArrayString(json_bytes, "\"sep\"") orelse
            findJsonFieldString(json_bytes, "\"sep\"");
    }

    // For TemplateProcessing: determine CLS/SEP from the "single" template
    // structure, not from special_tokens map iteration order (which is
    // undefined in JSON). SpecialToken before Sequence(A) → CLS; after → SEP.
    if (std.mem.eql(u8, type_str, "TemplateProcessing")) {
        if (findSection(json_bytes, "\"single\"")) |single_raw| {
            // Scope to just the [...] array (findSection returns from '[' to EOF)
            const single_end = findMatchingBrace(single_raw, '[', ']') orelse single_raw.len;
            const single_section = single_raw[0..single_end];
            const seq_pos = std.mem.indexOf(u8, single_section, "\"Sequence\"");
            var search_pos: usize = 0;
            while (std.mem.indexOfPos(u8, single_section, search_pos, "\"SpecialToken\"")) |spec_pos| {
                // Extract the "id" string from the SpecialToken object.
                // Limit search scope to avoid crossing into the next entry.
                const search_end = @min(spec_pos + 200, single_section.len);
                const spec_slice = single_section[spec_pos..search_end];
                if (findJsonFieldString(spec_slice, "\"id\"")) |id_str| {
                    if (seq_pos) |sp| {
                        if (spec_pos < sp) {
                            if (cls_str == null) cls_str = id_str;
                        } else {
                            if (sep_str == null) sep_str = id_str;
                        }
                    } else {
                        // No Sequence in template — treat first token as CLS
                        if (cls_str == null) cls_str = id_str;
                    }
                }
                search_pos = spec_pos + "\"SpecialToken\"".len;
            }
        }
    }

    // Default token strings (skip for TemplateProcessing — its cls/sep
    // assignment is determined by template ordering, not convention)
    if (cls_str == null and !std.mem.eql(u8, type_str, "TemplateProcessing")) {
        if (std.mem.eql(u8, type_str, "RobertaProcessing")) {
            cls_str = "<s>";
            sep_str = if (sep_str == null) "</s>" else sep_str;
        } else {
            cls_str = "[CLS]";
            sep_str = if (sep_str == null) "[SEP]" else sep_str;
        }
    }
    // For BertProcessing/RobertaProcessing, default sep to cls.
    // For TemplateProcessing, only add SEP if explicitly in special_tokens.
    if (sep_str == null and !std.mem.eql(u8, type_str, "TemplateProcessing")) {
        sep_str = cls_str;
    }

    // Copy token strings into postproc
    if (cls_str) |cls| {
        const copy_len = @min(cls.len, tokenizer.postproc.cls_token.len - 1);
        @memcpy(tokenizer.postproc.cls_token[0..copy_len], cls[0..copy_len]);
        tokenizer.postproc.cls_token[copy_len] = 0;
    }
    if (sep_str) |sep| {
        const copy_len = @min(sep.len, tokenizer.postproc.sep_token.len - 1);
        @memcpy(tokenizer.postproc.sep_token[0..copy_len], sep[0..copy_len]);
        tokenizer.postproc.sep_token[copy_len] = 0;
    }

    resolvePostProcessorAddedTokenIds(tokenizer);
}

fn resolvePostProcessorAddedTokenIds(tokenizer: *ct.Tokenizer) void {
    if (tokenizer.postproc.add_special == 0) return;

    if (tokenizer.postproc.cls_id == -1) {
        const cls_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.cls_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, cls_z)) |added| {
            tokenizer.postproc.cls_id = added.id;
        }
    }
    if (tokenizer.postproc.sep_id == -1) {
        const sep_z: [*:0]const u8 = @ptrCast(&tokenizer.postproc.sep_token);
        if (tok_fns.tokenizer_added_token_find(tokenizer, sep_z)) |added| {
            tokenizer.postproc.sep_id = added.id;
        }
    }
}

fn dupeArenaZ(arena: *std.heap.ArenaAllocator, bytes: []const u8) ![*c]const u8 {
    return (try arena.allocator().dupeZ(u8, bytes)).ptr;
}

fn dupeArenaZIfPresent(arena: *std.heap.ArenaAllocator, bytes: []const u8) ![*c]const u8 {
    return if (bytes.len > 0) try dupeArenaZ(arena, bytes) else null;
}

fn dupeArenaZOptional(arena: *std.heap.ArenaAllocator, maybe_bytes: ?[]const u8) ![*c]const u8 {
    return if (maybe_bytes) |bytes| try dupeArenaZ(arena, bytes) else null;
}

fn dupeRuntimeZOptional(maybe_bytes: ?[]const u8) ![*c]const u8 {
    return if (maybe_bytes) |bytes| (try std.heap.c_allocator.dupeZ(u8, bytes)).ptr else null;
}

fn applyRootMetadataWithoutPostProcessor(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    try applyAddedTokens(tokenizer, root.added_tokens);

    // Apply normalizer settings
    const normalizer_spec = ct.NormalizerSpec{
        .type = try dupeArenaZIfPresent(arena, root.normalizer.type),
        .lowercase = if (root.normalizer.lowercase) 1 else 0,
        .strip_accents = if (root.normalizer.strip_accents) 1 else 0,
        .nfc = if (root.normalizer.nfc) 1 else 0,
        .nfd = if (root.normalizer.nfd) 1 else 0,
        .nfkc = if (root.normalizer.nfkc) 1 else 0,
        .nfkd = if (root.normalizer.nfkd) 1 else 0,
        .clean_text = if (root.normalizer.clean_text) 1 else 0,
        .handle_chinese_chars = if (root.normalizer.handle_chinese_chars) 1 else 0,
        // SentencePiece-style normalizers
        .prepend = try dupeRuntimeZOptional(root.normalizer.prepend),
        .replace_pattern = try dupeRuntimeZOptional(root.normalizer.replace_pattern),
        .replace_content = try dupeRuntimeZOptional(root.normalizer.replace_content),
    };
    tok_fns.tokenizer_apply_normalizer_spec(tokenizer, &normalizer_spec);

    // Apply pre_tokenizer settings
    const pretokenizer_spec = ct.PreTokenizerSpec{
        .type = try dupeArenaZIfPresent(arena, root.pre_tokenizer.type),
        .add_prefix_space = if (root.pre_tokenizer.add_prefix_space) 1 else 0,
        .trim_offsets = if (root.pre_tokenizer.trim_offsets) 1 else 0,
        .use_regex = if (root.pre_tokenizer.use_regex) 1 else 0,
        .byte_level = if (root.pre_tokenizer.byte_level) 1 else 0,
        .whitespace = if (root.pre_tokenizer.whitespace) 1 else 0,
        .punctuation = if (root.pre_tokenizer.punctuation) 1 else 0,
        .pattern = try dupeArenaZOptional(arena, root.pre_tokenizer.pattern),
        .regex_split = if (root.pre_tokenizer.regex_split) 1 else 0,
        .regex_invert = if (root.pre_tokenizer.regex_invert) 1 else 0,
        .metaspace = if (root.pre_tokenizer.metaspace) 1 else 0,
    };
    tok_fns.tokenizer_apply_pretokenizer_spec(tokenizer, &pretokenizer_spec);
}

fn applyRootPostProcessorMetadata(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    // Apply post_processor settings from JSON.
    // When JSON explicitly specifies a post_processor, use those settings.
    // When JSON has no post_processor (null), clear any model default (e.g.,
    // WordPiece init sets add_special=1 for BERT-like behavior, but we must
    // respect the JSON configuration as authoritative).
    if (root.post_processor.type.len > 0 or root.post_processor.add_special or root.post_processor.pair or root.post_processor.cls_token != null or root.post_processor.sep_token != null) {
        const postprocessor_spec = ct.PostProcessorSpec{
            .type = try dupeArenaZIfPresent(arena, root.post_processor.type),
            .add_special = if (root.post_processor.add_special) 1 else 0,
            .pair = if (root.post_processor.pair) 1 else 0,
            .cls_token = try dupeArenaZOptional(arena, root.post_processor.cls_token),
            .sep_token = try dupeArenaZOptional(arena, root.post_processor.sep_token),
        };
        tok_fns.tokenizer_apply_postprocessor_spec(tokenizer, &postprocessor_spec);
        // Reset IDs so they're re-resolved from the configured token strings.
        // WordPiece init sets cls_id/sep_id from default [CLS]/[SEP] vocab lookup,
        // but BertProcessing may configure different tokens (e.g. <s>/<\/s>).
        tokenizer.postproc.cls_id = -1;
        tokenizer.postproc.sep_id = -1;
    }

    // Resolve cls_id/sep_id from added tokens when post_processor is active.
    // Look up the token strings in the added_tokens list to find the correct IDs.
    resolvePostProcessorAddedTokenIds(tokenizer);
}

fn applyRootDecoderMetadata(tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) void {
    // Apply decoder settings (e.g., Strip decoder for SentencePiece)
    tokenizer.decoder.strip_start = @intCast(root.decoder.strip_start);
    tokenizer.decoder.strip_stop = @intCast(root.decoder.strip_stop);
    tokenizer.decoder.wordpiece = if (std.mem.eql(u8, root.decoder.type, "WordPiece")) 1 else 0;
    tokenizer.decoder.cleanup = if (root.decoder.cleanup) 1 else 0;
    tokenizer.decoder.add_prefix_space = if (root.decoder.add_prefix_space) 1 else 0;
    tokenizer.decoder.metaspace = if (root.decoder.metaspace) 1 else 0;
}

fn applyRootMetadata(arena: *std.heap.ArenaAllocator, tokenizer: *ct.Tokenizer, root: schema.TokenizerRoot) !void {
    try applyRootMetadataWithoutPostProcessor(arena, tokenizer, root);
    try applyRootPostProcessorMetadata(arena, tokenizer, root);
    applyRootDecoderMetadata(tokenizer, root);
}

fn buildTokenizerFromRoot(arena: *std.heap.ArenaAllocator, root: schema.TokenizerRoot) !*ct.Tokenizer {
    const model_type_name = root.model.type;
    if (std.mem.eql(u8, model_type_name, "WordPiece")) {
        const tokenizer = try buildWordPiece(root.model);
        errdefer tok_fns.tokenizer_free(tokenizer);
        try applyRootMetadata(arena, tokenizer, root);
        return tokenizer;
    } else if (std.mem.eql(u8, model_type_name, "Unigram")) {
        const tokenizer = try buildUnigram(root.model);
        errdefer tok_fns.tokenizer_free(tokenizer);
        try applyRootMetadata(arena, tokenizer, root);
        return tokenizer;
    } else {
        return error.UnsupportedModel;
    }
}

fn buildWordPiece(model: schema.Model) !*ct.Tokenizer {
    return wordpiece_model.createWordPieceTokenizer(
        model.vocab,
        model.unk_token,
        model.max_input_chars_per_word,
    );
}

fn buildUnigram(model: schema.Model) !*ct.Tokenizer {
    return unigram_model.createUnigramTokenizer(
        model.vocab,
        model.unk_token,
        model.bos_token,
        model.eos_token,
    );
}

fn applyAddedTokens(tokenizer: *ct.Tokenizer, added_tokens: []const schema.AddedToken) !void {
    for (added_tokens) |token_entry| {
        const content_dup = try std.heap.c_allocator.dupeZ(u8, token_entry.content);
        defer std.heap.c_allocator.free(content_dup);
        const added_node = tok_fns.tokenizer_added_token_add(tokenizer, content_dup.ptr, token_entry.id, if (token_entry.special) 1 else 0);
        if (added_node == null) return error.BuildFailed;
        added_node.?.single_word = if (token_entry.single_word) 1 else 0;
        added_node.?.lstrip = if (token_entry.lstrip) 1 else 0;
        added_node.?.rstrip = if (token_entry.rstrip) 1 else 0;
        added_node.?.normalized = if (token_entry.normalized) 1 else 0;
    }
}

// =============================================================================
// Tests
// =============================================================================

test "parseTokenizerJson accepts valid minimal bpe" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;

    const root = try parseTokenizerJson(std.testing.allocator, json);
    try std.testing.expectEqualStrings("BPE", root.model.type);
    try std.testing.expectEqual(@as(usize, 1), root.model.vocab.len);
    try std.testing.expectEqual(@as(usize, 0), root.model.merges.?.len);
}

test "findJsonFieldValue extracts number" {
    const json =
        \\{"id": 123, "name": "test"}
    ;
    const value = findJsonFieldValue(json, "\"id\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("123", value.?);
}

test "validateModelOptions rejects unsupported BPE options" {
    try validateModelOptions("BPE",
        \\{"type":"BPE","dropout":null,"fuse_unk":true,"byte_fallback":true,"ignore_merges":false}
    );
    try validateModelOptions("WordPiece",
        \\{"type":"WordPiece","dropout":0.25}
    );

    try std.testing.expectError(error.InvalidModel, validateModelOptions("BPE",
        \\{"type":"BPE","dropout":0.25}
    ));
    try std.testing.expectError(error.InvalidModel, validateModelOptions("BPE",
        \\{"type":"BPE","fuse_unk":true,"byte_fallback":false}
    ));
    try std.testing.expectError(error.InvalidModel, validateModelOptions("BPE",
        \\{"type":"BPE","continuing_subword_prefix":"##"}
    ));
}

test "parseMetadataSections parses added tokens and pipeline sections" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{
        \\  "added_tokens": [
        \\    {"id": 4, "content": "[X]", "special": true, "single_word": true}
        \\  ],
        \\  "normalizer": {"type": "Lowercase"},
        \\  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
        \\  "post_processor": null,
        \\  "decoder": {"type": "ByteFallback"}
        \\}
    ;

    var added = ManagedArrayList(schema.AddedToken).init(allocator);
    var normalizer = schema.Normalizer{};
    var pretokenizer = schema.PreTokenizer{};
    var postprocessor = schema.PostProcessor{};
    var decoder = schema.Decoder{};

    try parseMetadataSections(allocator, json, &added, &normalizer, &pretokenizer, &postprocessor, &decoder);

    try std.testing.expectEqual(@as(usize, 1), added.items.len);
    try std.testing.expectEqual(@as(i32, 4), added.items[0].id);
    try std.testing.expectEqualStrings("[X]", added.items[0].content);
    try std.testing.expect(added.items[0].special);
    try std.testing.expect(added.items[0].single_word);
    try std.testing.expect(normalizer.lowercase);
    try std.testing.expect(pretokenizer.byte_level);
    try std.testing.expect(pretokenizer.add_prefix_space);
    try std.testing.expectEqualStrings("ByteFallback", decoder.type);
    try std.testing.expectEqualStrings("", postprocessor.type);
}

test "ensureJsonDepthWithinLimit rejects excessive object nesting" {
    var json = std.ArrayListUnmanaged(u8){};
    defer json.deinit(std.testing.allocator);

    for (0..MAX_JSON_PIPELINE_DEPTH + 2) |_| {
        try json.append(std.testing.allocator, '{');
    }
    for (0..MAX_JSON_PIPELINE_DEPTH + 2) |_| {
        try json.append(std.testing.allocator, '}');
    }

    try std.testing.expectError(
        error.InvalidDecoder,
        ensureJsonDepthWithinLimit(json.items, error.InvalidDecoder),
    );
    try ensureJsonDepthWithinLimit("{}", error.InvalidDecoder);
}

test "unescapeJsonStringFast handles escape sequences" {
    const allocator = std.testing.allocator;
    const input = "hello\\nworld";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello\nworld", result);
}

test "unescapeJsonStringFast handles multiple escape types" {
    const allocator = std.testing.allocator;

    // Test \t, \r, \n
    {
        const input = "line1\\tline2\\rline3\\nline4";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("line1\tline2\rline3\nline4", result);
    }

    // Test \\, \", \/
    {
        const input = "path\\\\to\\\"file\\\"/";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("path\\to\"file\"/", result);
    }

    // Test \b (backspace) and \f (form feed)
    {
        const input = "text\\bwith\\fspecial";
        const result = try unescapeJsonStringFast(allocator, input);
        defer allocator.free(result);
        try std.testing.expectEqualStrings("text\x08with\x0Cspecial", result);
    }
}

test "unescapeJsonStringFast handles empty string" {
    const allocator = std.testing.allocator;
    const input = "";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "unescapeJsonStringFast handles string without escapes" {
    const allocator = std.testing.allocator;
    const input = "hello world";
    const result = try unescapeJsonStringFast(allocator, input);
    defer allocator.free(result);
    try std.testing.expectEqualStrings("hello world", result);
}

test "findJsonFieldValue extracts string value" {
    const json =
        \\{"name": "test", "count": 42}
    ;
    const value = findJsonFieldValue(json, "\"name\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("\"test\"", value.?);
}

test "findJsonFieldValue extracts boolean true" {
    const json =
        \\{"enabled": true, "other": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("true", value.?);
}

test "findJsonFieldValue extracts boolean false" {
    const json =
        \\{"enabled": false}
    ;
    const value = findJsonFieldValue(json, "\"enabled\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("false", value.?);
}

test "findJsonFieldValue returns null for missing field" {
    const json =
        \\{"name": "test"}
    ;
    const value = findJsonFieldValue(json, "\"missing\"");
    try std.testing.expect(value == null);
}

test "findJsonFieldValue handles fields with whitespace" {
    const json =
        \\{"name"  :   123  ,  "other": 456}
    ;
    const value = findJsonFieldValue(json, "\"name\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("123", value.?);
}

test "findJsonFieldString extracts quoted string" {
    const json =
        \\{"type": "BPE", "name": "tokenizer"}
    ;
    const value = findJsonFieldString(json, "\"type\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("BPE", value.?);
}

test "findJsonFieldString extracts value without outer quotes" {
    const json =
        \\{"path": "test_value"}
    ;
    const value = findJsonFieldString(json, "\"path\"");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("test_value", value.?);
}

test "findJsonFieldString returns null for non-string value" {
    const json =
        \\{"count": 42}
    ;
    const value = findJsonFieldString(json, "\"count\"");
    try std.testing.expect(value == null);
}

test "findJsonFieldString returns null for missing field" {
    const json =
        \\{"name": "test"}
    ;
    const value = findJsonFieldString(json, "\"missing\"");
    try std.testing.expect(value == null);
}

test "findQuotedString finds first quoted string" {
    const input =
        \\some text "hello world" more text
    ;
    const result = findQuotedString(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("hello world", result.?);
}

test "findQuotedString returns null when no quotes" {
    const input = "no quotes here";
    const result = findQuotedString(input);
    try std.testing.expect(result == null);
}

test "findQuotedString handles unclosed quote" {
    // findQuotedString returns content from opening quote to end of string
    // when there's no closing quote (it doesn't validate)
    const input = "some text \"unclosed";
    const result = findQuotedString(input);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("unclosed", result.?);
}

test "parseVocabFastSection parses simple vocab" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"hello": 0, "world": 1, "test": 2}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 3), vocab_entries.items.len);
    try std.testing.expectEqualStrings("hello", vocab_entries.items[0].token);
    try std.testing.expectEqual(@as(i32, 0), vocab_entries.items[0].id);
    try std.testing.expectEqualStrings("world", vocab_entries.items[1].token);
    try std.testing.expectEqual(@as(i32, 1), vocab_entries.items[1].id);
}

test "parseVocabFastSection handles escaped tokens" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // JSON with escape sequences: backslash followed by 'n' should become newline
    // Use string concatenation to get actual backslash-n in the JSON
    const json = "{\"hello" ++ [_]u8{ '\\', 'n' } ++ "world\": 0}";
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 1), vocab_entries.items.len);
    // The function unescapes \n to newline
    try std.testing.expectEqualStrings("hello\nworld", vocab_entries.items[0].token);
}

test "parseVocabFastSection skips non-numeric values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"token": 0, "config": {"nested": "value"}, "other": 1}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try parseVocabFastSection(allocator, json, &vocab_entries);

    try std.testing.expectEqual(@as(usize, 2), vocab_entries.items.len);
    try std.testing.expectEqualStrings("token", vocab_entries.items[0].token);
    try std.testing.expectEqualStrings("other", vocab_entries.items[1].token);
}

test "parseVocabFastSection rejects sparse ids outside i32 range" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\{"<unk>": 0, "boom": 2147483648}
    ;
    var vocab_entries = ManagedArrayList(schema.TokenId).init(allocator);

    try std.testing.expectError(error.InvalidVocab, parseVocabFastSection(allocator, json, &vocab_entries));
}

test "parseMergesFastSection parses array format" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const json =
        \\[["a", "b"], ["c", "d"], ["e", "f"]]
    ;
    var merge_entries = ManagedArrayList([]const u8).init(allocator);

    try parseMergesFastSection(allocator, json, &merge_entries);

    try std.testing.expectEqual(@as(usize, 3), merge_entries.items.len);
    try std.testing.expectEqualStrings("a b", merge_entries.items[0]);
    try std.testing.expectEqualStrings("c d", merge_entries.items[1]);
    try std.testing.expectEqualStrings("e f", merge_entries.items[2]);
}

test "parseMergesFastSection parses mixed array format" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Real-world format: array of arrays
    const json =
        \\[["a", "b"], ["c", "d"]]
    ;
    var merge_entries = ManagedArrayList([]const u8).init(allocator);

    try parseMergesFastSection(allocator, json, &merge_entries);

    try std.testing.expectEqual(@as(usize, 2), merge_entries.items.len);
    try std.testing.expectEqualStrings("a b", merge_entries.items[0]);
    try std.testing.expectEqualStrings("c d", merge_entries.items[1]);
}

test "parseNormalizer parses NFC normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "NFC"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.nfc);
    try std.testing.expect(!normalizer.nfd);
    try std.testing.expect(!normalizer.nfkc);
}

test "parseNormalizer parses NFKD normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "NFKD"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.nfkd);
    try std.testing.expect(!normalizer.nfc);
    try std.testing.expect(!normalizer.nfd);
    try std.testing.expect(!normalizer.nfkc);
}

test "parseNormalizer parses lowercase normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Lowercase"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.lowercase);
}

test "parseNormalizer handles explicit flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Custom", "lowercase": true, "strip_accents": true, "nfc": true}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(normalizer.lowercase);
    try std.testing.expect(normalizer.strip_accents);
    try std.testing.expect(normalizer.nfc);
}

test "parseNormalizer preserves explicit BertNormalizer false flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "BertNormalizer", "clean_text": false, "handle_chinese_chars": false, "strip_accents": false, "lowercase": false}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(!normalizer.clean_text);
    try std.testing.expect(!normalizer.handle_chinese_chars);
    try std.testing.expect(!normalizer.strip_accents);
    try std.testing.expect(!normalizer.lowercase);
}

test "parseNormalizer rejects unknown type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "DoesNotExist"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer rejects Replace without pattern" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Replace", "content": "x"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer rejects Prepend without prepend text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Prepend"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidNormalizer, parseNormalizer(arena.allocator(), &scanner));
}

test "parseNormalizer handles null normalizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const normalizer = try parseNormalizer(arena.allocator(), &scanner);

    try std.testing.expect(!normalizer.lowercase);
    try std.testing.expect(!normalizer.nfc);
}

test "parsePreTokenizer parses ByteLevel type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "ByteLevel", "add_prefix_space": true}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(pretok.byte_level);
    try std.testing.expect(pretok.add_prefix_space);
}

test "parsePreTokenizer parses Whitespace type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Whitespace"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(pretok.whitespace);
}

test "parsePreTokenizer handles null pretokenizer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const pretok = try parsePreTokenizer(arena.allocator(), &scanner);

    try std.testing.expect(!pretok.byte_level);
    try std.testing.expect(!pretok.whitespace);
}

test "parsePostProcessor parses BertProcessing type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "BertProcessing"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("[CLS]", postproc.cls_token.?);
    try std.testing.expectEqualStrings("[SEP]", postproc.sep_token.?);
}

test "parsePostProcessor parses RobertaProcessing type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "RobertaProcessing"}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expect(postproc.pair);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expectEqualStrings("</s>", postproc.sep_token.?);
}

test "parsePostProcessor rejects TemplateProcessing with undefined special token mapping" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}}
        \\  ],
        \\  "special_tokens": {}
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    try std.testing.expectError(error.InvalidPostProcessor, parsePostProcessor(arena.allocator(), &scanner));
}

test "parsePostProcessor accepts Sequence wrapping ByteLevel and TemplateProcessing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "processors": [
        \\    {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
        \\    {
        \\      "type": "TemplateProcessing",
        \\      "single": [
        \\        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\        {"Sequence": {"id": "A", "type_id": 0}}
        \\      ],
        \\      "pair": [
        \\        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\        {"Sequence": {"id": "A", "type_id": 0}},
        \\        {"Sequence": {"id": "B", "type_id": 1}}
        \\      ],
        \\      "special_tokens": {
        \\        "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        \\      }
        \\    }
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("TemplateProcessing", postproc.type);
    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expect(postproc.sep_token == null);
}

test "parsePostProcessor infers TemplateProcessing BOS and EOS from single template order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"SpecialToken": {"id": "<s>", "type_id": 0}},
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "</s>", "type_id": 0}}
        \\  ],
        \\  "special_tokens": {
        \\    "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
        \\    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
        \\  }
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expectEqualStrings("<s>", postproc.cls_token.?);
    try std.testing.expectEqualStrings("</s>", postproc.sep_token.?);
}

test "parsePostProcessor infers EOS-only TemplateProcessing without default CLS" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "TemplateProcessing",
        \\  "single": [
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
        \\  ],
        \\  "pair": [
        \\    {"Sequence": {"id": "A", "type_id": 0}},
        \\    {"Sequence": {"id": "B", "type_id": 1}},
        \\    {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
        \\  ],
        \\  "special_tokens": {
        \\    "<|endoftext|>": {"id": "<|endoftext|>", "ids": [1], "tokens": ["<|endoftext|>"]}
        \\  }
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(postproc.add_special);
    try std.testing.expect(postproc.cls_token == null);
    try std.testing.expectEqualStrings("<|endoftext|>", postproc.sep_token.?);
}

test "parsePostProcessor handles null postprocessor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const postproc = try parsePostProcessor(arena.allocator(), &scanner);

    try std.testing.expect(!postproc.add_special);
    try std.testing.expect(postproc.cls_token == null);
}

test "parseDecoder parses Strip decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "Strip", "start": 1, "stop": 0}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqual(@as(i32, 1), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder handles null decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json = "null";

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqual(@as(i32, 0), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder preserves WordPiece cleanup=false" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{"type": "WordPiece", "prefix": "##", "cleanup": false}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("WordPiece", decoder.type);
    try std.testing.expect(!decoder.cleanup);
}

test "parseDecoder handles ByteFallback decoder" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "ByteFallback"
        \\}
    ;
    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);
    try std.testing.expectEqualStrings("ByteFallback", decoder.type);
}

test "parseDecoder Sequence maps Replace metaspace and Strip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "decoders": [
        \\    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        \\    {"type": "ByteFallback"},
        \\    {"type": "Fuse"},
        \\    {"type": "Strip", "content": " ", "start": 1, "stop": 0}
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("Sequence", decoder.type);
    try std.testing.expect(decoder.metaspace);
    try std.testing.expectEqual(@as(i32, 1), decoder.strip_start);
    try std.testing.expectEqual(@as(i32, 0), decoder.strip_stop);
}

test "parseDecoder Sequence preserves nested Metaspace add_prefix_space" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "decoders": [
        \\    {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}
        \\  ]
        \\}
    ;

    var scanner = std.json.Scanner.initCompleteInput(arena.allocator(), json);
    const decoder = try parseDecoder(arena.allocator(), &scanner);

    try std.testing.expectEqualStrings("Sequence", decoder.type);
    try std.testing.expect(decoder.metaspace);
    try std.testing.expect(decoder.add_prefix_space);
}

test "validateBpeMergeReferences accepts chained merge intermediates" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const vocab = [_]schema.TokenId{
        .{ .token = "h", .id = 0, .score = -1.0 },
        .{ .token = "e", .id = 1, .score = -1.0 },
        .{ .token = "l", .id = 2, .score = -1.0 },
        .{ .token = "o", .id = 3, .score = -1.0 },
    };
    const merges = [_][]const u8{
        "h e",
        "he l",
        "hel l",
        "hell o",
    };

    try validateBpeMergeReferences(arena.allocator(), "BPE", &vocab, &merges);
}

fn appendNestedSequenceSection(
    allocator: std.mem.Allocator,
    field_name: []const u8,
    entry_name: []const u8,
    depth: usize,
    leaf: []const u8,
) ![]u8 {
    var current = try allocator.dupe(u8, leaf);
    errdefer allocator.free(current);
    for (0..depth) |_| {
        const next = try std.fmt.allocPrint(
            allocator,
            "{{\"type\":\"Sequence\",\"{s}\":[{s}]}}",
            .{ entry_name, current },
        );
        allocator.free(current);
        current = next;
    }
    errdefer allocator.free(current);
    const normalizer = if (std.mem.eql(u8, field_name, "normalizer")) current else "null";
    const pre_tokenizer = if (std.mem.eql(u8, field_name, "pre_tokenizer")) current else "null";
    const post_processor = if (std.mem.eql(u8, field_name, "post_processor")) current else "null";
    const decoder = if (std.mem.eql(u8, field_name, "decoder")) current else "null";
    return std.fmt.allocPrint(
        allocator,
        "{{\n  \"version\": \"1.0\",\n  \"model\": {{\"type\": \"BPE\", \"vocab\": {{\"<unk>\": 0, \"a\": 1}}, \"merges\": []}},\n  \"added_tokens\": [{{\"id\": 0, \"content\": \"<unk>\", \"special\": true}}],\n  \"normalizer\": {s},\n  \"pre_tokenizer\": {s},\n  \"post_processor\": {s},\n  \"decoder\": {s}\n}}",
        .{ normalizer, pre_tokenizer, post_processor, decoder },
    );
}

test "parseTokenizerJson accepts normalizer nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "normalizer",
        "normalizers",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"Lowercase\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try parseTokenizerJson(std.testing.allocator, json);
    try std.testing.expect(root.normalizer.lowercase);
}

test "parseTokenizerJson rejects normalizer nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "normalizer",
        "normalizers",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"Lowercase\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidNormalizer, parseTokenizerJson(std.testing.allocator, json));
}

test "parseTokenizerJson rejects pretokenizer nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "pre_tokenizer",
        "pretokenizers",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"Whitespace\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidPreTokenizer, parseTokenizerJson(std.testing.allocator, json));
}

test "parseTokenizerJson rejects postprocessor nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "post_processor",
        "processors",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidPostProcessor, parseTokenizerJson(std.testing.allocator, json));
}

test "parseTokenizerJson accepts postprocessor nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "post_processor",
        "processors",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try parseTokenizerJson(std.testing.allocator, json);
    try std.testing.expect(root.post_processor.type.len > 0);
}

test "parseTokenizerJson rejects decoder nesting above depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "decoder",
        "decoders",
        MAX_JSON_PIPELINE_DEPTH + 1,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    try std.testing.expectError(error.InvalidDecoder, parseTokenizerJson(std.testing.allocator, json));
}

test "parseTokenizerJson accepts decoder nesting at depth limit" {
    const json = try appendNestedSequenceSection(
        std.testing.allocator,
        "decoder",
        "decoders",
        MAX_JSON_PIPELINE_DEPTH,
        "{\"type\":\"ByteLevel\"}",
    );
    defer std.testing.allocator.free(json);

    const root = try parseTokenizerJson(std.testing.allocator, json);
    try std.testing.expectEqualStrings("Sequence", root.decoder.type);
}

test "parseTokenizerJson accepts direct ByteFallback decoder" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {"type": "BPE", "vocab": {"<unk>": 0}, "merges": []},
        \\  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": {"type": "ByteFallback"}
        \\}
    ;

    const root = try parseTokenizerJson(std.testing.allocator, json);
    try std.testing.expectEqualStrings("ByteFallback", root.decoder.type);
}

test "tokenizer_loader_from_json_string applies whitespace pretokenizer on lazy bpe path" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
        \\      "a": 4, "b": 5
        \\    },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<pad>", "special": true},
        \\    {"id": 1, "content": "<s>", "special": true},
        \\    {"id": 2, "content": "</s>", "special": true},
        \\    {"id": 3, "content": "<unk>", "special": true}
        \\  ],
        \\  "normalizer": null,
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.OutOfMemory;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer @import("encode.zig").tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(@as(c_int, 0), @import("encode.zig").tokenizer_encode_struct_with_options(tokenizer, "a b", &encoding, .{ .add_special_tokens = false }));
    try std.testing.expectEqual(@as(usize, 2), encoding.ids_len);

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(i32, 4), ids[0]);
    try std.testing.expectEqual(@as(i32, 5), ids[1]);
}

test "tokenizer_loader_from_json_string preserves sentencepiece normalizer ownership on lazy bpe path" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "vocab": {
        \\      "<unk>": 0, "<s>": 1, "</s>": 2,
        \\      "▁": 3,
        \\      "h": 4, "e": 5, "l": 6, "o": 7, "w": 8, "r": 9, "d": 10,
        \\      "▁h": 11, "▁he": 12, "▁hel": 13, "▁hell": 14, "▁hello": 15,
        \\      "▁w": 16, "▁wo": 17, "▁wor": 18, "▁worl": 19, "▁world": 20
        \\    },
        \\    "merges": [
        \\      "▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o",
        \\      "▁ w", "▁w o", "▁wo r", "▁wor l", "▁worl d"
        \\    ]
        \\  },
        \\  "added_tokens": [
        \\    {"id": 0, "content": "<unk>", "special": true},
        \\    {"id": 1, "content": "<s>", "special": true},
        \\    {"id": 2, "content": "</s>", "special": true}
        \\  ],
        \\  "normalizer": {
        \\    "type": "Sequence",
        \\    "normalizers": [
        \\      {"type": "Prepend", "prepend": "▁"},
        \\      {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        \\    ]
        \\  },
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.OutOfMemory;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer @import("encode.zig").tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(@as(c_int, 0), @import("encode.zig").tokenizer_encode_struct_with_options(tokenizer, "<s> hello </s> world", &encoding, .{ .add_special_tokens = false }));

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(usize, 7), encoding.ids_len);
    try std.testing.expectEqualSlices(i32, &.{ 1, 3, 15, 3, 2, 3, 20 }, ids[0..encoding.ids_len]);
}

test "tokenizer_loader_from_json_string accepts empty inert BPE affix defaults" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "dropout": null,
        \\    "unk_token": "<unk>",
        \\    "continuing_subword_prefix": "",
        \\    "end_of_word_suffix": "",
        \\    "fuse_unk": false,
        \\    "byte_fallback": false,
        \\    "ignore_merges": false,
        \\    "vocab": { "<unk>": 0, "h": 1, "e": 2, "l": 3, "o": 4 },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.TestUnexpectedResult;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }
}

test "tokenizer_loader_from_json_string accepts Gemma-style BPE options with byte fallback" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "dropout": null,
        \\    "unk_token": "<unk>",
        \\    "continuing_subword_prefix": null,
        \\    "end_of_word_suffix": null,
        \\    "fuse_unk": true,
        \\    "byte_fallback": true,
        \\    "ignore_merges": false,
        \\    "vocab": { "<unk>": 0, "h": 1, "e": 2, "l": 3, "o": 4 },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": {"type":"ByteFallback"}
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.TestUnexpectedResult;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }
}

test "tokenizer_loader_from_json_string applies ignore_merges when enabled" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "ignore_merges": true,
        \\    "vocab": { "<unk>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "abc": 5 },
        \\    "merges": [["a", "b"]]
        \\  },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    const json_z = try allocator.dupeZ(u8, json);
    defer allocator.free(json_z);

    const tokenizer = tokenizer_loader_from_json_string(json_z.ptr) orelse return error.TestUnexpectedResult;
    defer {
        tokenizer.destroy();
        std.heap.c_allocator.destroy(tokenizer);
    }

    var encoding = std.mem.zeroes(ct.TokenizerEncoding);
    defer @import("encode.zig").tokenizer_encoding_free_struct(&encoding);

    try std.testing.expectEqual(
        @as(c_int, 0),
        @import("encode.zig").tokenizer_encode_struct_with_options(
            tokenizer,
            "abc",
            &encoding,
            .{ .add_special_tokens = false },
        ),
    );

    const ids: [*]i32 = @ptrCast(encoding.ids.?);
    try std.testing.expectEqual(@as(usize, 1), encoding.ids_len);
    try std.testing.expectEqual(@as(i32, 5), ids[0]);
}

test "parseTokenizerJson rejects fuse_unk without byte_fallback" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "model": {
        \\    "type": "BPE",
        \\    "dropout": null,
        \\    "unk_token": "<unk>",
        \\    "continuing_subword_prefix": null,
        \\    "end_of_word_suffix": null,
        \\    "fuse_unk": true,
        \\    "byte_fallback": false,
        \\    "ignore_merges": false,
        \\    "vocab": { "<unk>": 0, "h": 1, "e": 2, "l": 3, "o": 4 },
        \\    "merges": []
        \\  },
        \\  "added_tokens": [],
        \\  "normalizer": null,
        \\  "pre_tokenizer": null,
        \\  "post_processor": null,
        \\  "decoder": null
        \\}
    ;
    try std.testing.expectError(error.InvalidModel, parseTokenizerJson(std.testing.allocator, json));
}
