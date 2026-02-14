//! WordPiece Model
//!
//! Implements BERT-style WordPiece tokenization.
//! Uses greedy longest-match-first algorithm with ##-prefixed subwords.

const std = @import("std");
const ct = @import("c_types.zig");
const decoders = @import("decoders.zig");
const utils = @import("utils.zig");

const tok_fns = @import("pipeline.zig");

const WORDPIECE_PATTERN: [:0]const u8 = "[A-Za-z0-9]+|[^A-Za-z0-9\\s]+";
const DEFAULT_CLS: [:0]const u8 = "[CLS]";
const DEFAULT_SEP: [:0]const u8 = "[SEP]";
const DEFAULT_UNK: [:0]const u8 = "[UNK]";
const DEFAULT_CLS_ID: i32 = 101;
const DEFAULT_SEP_ID: i32 = 102;
const DEFAULT_UNK_ID: i32 = 100;

pub const WordPieceModel = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMapUnmanaged(i32),
    id_to_token: []?[*:0]u8,
    vocab_strings: std.ArrayListUnmanaged([:0]u8),
    vocab_size: usize,
    unk_id: i32,
    unk_token: [16]u8,
    owner: ?*ct.Tokenizer,
};

const EncodedWord = struct {
    ids: []i32,
    tokens: [][*:0]u8,
};

fn freeEncodedWordDeep(allocator: std.mem.Allocator, encoded: EncodedWord) void {
    for (encoded.tokens) |tok_ptr| {
        allocator.free(std.mem.sliceTo(tok_ptr, 0));
    }
    allocator.free(encoded.tokens);
    allocator.free(encoded.ids);
}

fn freeEncodedWordBuffers(allocator: std.mem.Allocator, encoded: EncodedWord) void {
    allocator.free(encoded.tokens);
    allocator.free(encoded.ids);
}

fn initModel(allocator: std.mem.Allocator) !*WordPieceModel {
    const model = try allocator.create(WordPieceModel);
    model.* = .{
        .allocator = allocator,
        .vocab = .{},
        .id_to_token = &[_]?[*:0]u8{},
        .vocab_strings = .{},
        .vocab_size = 0,
        .unk_id = DEFAULT_UNK_ID,
        .unk_token = undefined,
        .owner = null,
    };
    setUnkToken(model, DEFAULT_UNK);
    return model;
}

fn setUnkToken(model: *WordPieceModel, token: []const u8) void {
    utils.setUnkToken(&model.unk_token, token);
}

fn unkSlice(model: *const WordPieceModel) []const u8 {
    return utils.unkSlice(&model.unk_token);
}

fn addVocabEntry(model: *WordPieceModel, token_z: [:0]const u8, id: i32) !void {
    if (id < 0) return error.InvalidId;
    const dup = try model.allocator.dupeZ(u8, token_z);
    model.vocab_strings.append(model.allocator, dup) catch |err| {
        model.allocator.free(dup);
        return err;
    };

    model.vocab.put(model.allocator, dup[0..dup.len], id) catch |err| {
        _ = model.vocab_strings.pop();
        model.allocator.free(dup);
        return err;
    };
    if (@as(usize, @intCast(id)) < model.id_to_token.len) {
        model.id_to_token[@as(usize, @intCast(id))] = dup.ptr;
    }
}

fn findId(model: *const WordPieceModel, token: []const u8) ?i32 {
    return model.vocab.get(token);
}

fn encodeWord(model: *WordPieceModel, tokenizer: *ct.Tokenizer, word: []const u8) !EncodedWord {
    const allocator = model.allocator;

    // Added token short-circuit
    const word_z = try allocator.dupeZ(u8, word);
    defer allocator.free(word_z);
    if (tok_fns.tokenizer_added_token_find(tokenizer, word_z.ptr)) |added| {
        const ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(ids);
        const toks = try allocator.alloc([*:0]u8, 1);
        errdefer allocator.free(toks);
        ids[0] = added.*.id;
        const dup_tok = try allocator.dupeZ(u8, word);
        toks[0] = dup_tok.ptr;
        return EncodedWord{ .ids = ids, .tokens = toks };
    }

    var ids = std.ArrayList(i32).empty;
    defer ids.deinit(allocator);
    var tokens = std.ArrayList([*:0]u8).empty;
    defer tokens.deinit(allocator);
    errdefer {
        for (tokens.items) |tok_ptr| allocator.free(std.mem.sliceTo(tok_ptr, 0));
    }
    var scratch = std.ArrayList(u8).empty;
    defer scratch.deinit(allocator);

    var pos: usize = 0;
    while (pos < word.len) {
        var end = word.len;
        var found_id: ?i32 = null;
        while (end > pos) {
            scratch.clearRetainingCapacity();
            if (pos != 0) try scratch.appendSlice(allocator, "##");
            try scratch.appendSlice(allocator, word[pos..end]);
            if (model.vocab.get(scratch.items)) |v| {
                found_id = v;
                break;
            }
            end -= 1;
        }
        if (found_id == null) return error.UnknownWord;

        const dup_tok = try allocator.dupeZ(u8, scratch.items);
        errdefer allocator.free(dup_tok);
        try ids.append(allocator, found_id.?);
        try tokens.append(allocator, dup_tok.ptr);
        pos = end;
    }

    return EncodedWord{
        .ids = try ids.toOwnedSlice(allocator),
        .tokens = try tokens.toOwnedSlice(allocator),
    };
}

fn wordpiece_encode(tokenizer: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    if (tokenizer.model == null) return -1;
    const model = @as(*WordPieceModel, @ptrCast(@alignCast(tokenizer.model.?)));
    const allocator = model.allocator;

    var ids = std.ArrayList(i32).empty;
    defer ids.deinit(allocator);
    var tokens = std.ArrayList([*:0]u8).empty;
    defer tokens.deinit(allocator);
    errdefer {
        for (tokens.items) |tok_ptr| allocator.free(std.mem.sliceTo(tok_ptr, 0));
    }

    var cursor: usize = 0;
    while (cursor < text.len) {
        while (cursor < text.len and text[cursor] == ' ') cursor += 1;
        if (cursor >= text.len) break;
        const start = cursor;
        while (cursor < text.len and text[cursor] != ' ') cursor += 1;
        if (cursor <= start) continue;
        const word = text[start..cursor];

        const encoded = encodeWord(model, tokenizer, word) catch {
            // Unknown word -> push UNK
            ids.append(allocator, model.unk_id) catch return -1;
            const unk = allocator.dupeZ(u8, unkSlice(model)) catch return -1;
            tokens.append(allocator, unk.ptr) catch {
                allocator.free(unk);
                return -1;
            };
            continue;
        };

        ids.ensureUnusedCapacity(allocator, encoded.ids.len) catch {
            freeEncodedWordDeep(allocator, encoded);
            return -1;
        };
        tokens.ensureUnusedCapacity(allocator, encoded.tokens.len) catch {
            freeEncodedWordDeep(allocator, encoded);
            return -1;
        };
        ids.appendSlice(allocator, encoded.ids) catch {
            freeEncodedWordDeep(allocator, encoded);
            return -1;
        };
        tokens.appendSlice(allocator, encoded.tokens) catch {
            freeEncodedWordDeep(allocator, encoded);
            return -1;
        };
        freeEncodedWordBuffers(allocator, encoded);
    }

    const ids_owned = ids.toOwnedSlice(allocator) catch return -1;
    const toks_owned = tokens.toOwnedSlice(allocator) catch {
        allocator.free(ids_owned);
        return -1;
    };
    enc.ids_len = ids_owned.len;
    enc.tokens_len = toks_owned.len;
    enc.ids = @ptrCast(ids_owned.ptr);
    enc.tokens = @ptrCast(toks_owned.ptr);
    return 0;
}

fn wordpiece_decode_impl(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
    if (tokenizer.model == null) return -1;
    const model = @as(*WordPieceModel, @ptrCast(@alignCast(tokenizer.model.?)));
    const allocator = model.allocator;
    const unk_ptr: [*:0]const u8 = @ptrCast(&model.unk_token);

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    var first = true;
    for (0..ids_len) |id_idx| {
        const id = ids[id_idx];

        // Check if this is a special token (skip if requested)
        if (options.skip_special_tokens) {
            if (isSpecialToken(tokenizer, id)) continue;
        }

        const token_slice = blk: {
            if (id >= 0 and @as(usize, @intCast(id)) < model.id_to_token.len) {
                if (model.id_to_token[@as(usize, @intCast(id))]) |ptr| {
                    break :blk std.mem.sliceTo(ptr, 0);
                }
            }
            break :blk std.mem.sliceTo(unk_ptr, 0);
        };

        const is_subword = token_slice.len >= 2 and token_slice[0] == '#' and token_slice[1] == '#';
        // Only strip ## prefix from non-first tokens. HuggingFace's WordPiece
        // decoder checks `i != 0` — the first token always keeps its prefix.
        const content = if (is_subword and !first) token_slice[2..] else token_slice;

        if (!is_subword and !first) {
            result.append(allocator, ' ') catch return -1;
        }
        result.appendSlice(allocator, content) catch return -1;
        first = false;
    }

    // Cleanup: remove space before certain punctuation characters
    var write_pos: usize = 0;
    for (result.items, 0..) |ch, read_pos| {
        if (ch == ' ' and read_pos + 1 < result.items.len and isCleanupPunct(result.items[read_pos + 1])) {
            continue; // skip the space
        }
        result.items[write_pos] = ch;
        write_pos += 1;
    }
    result.items.len = write_pos;

    // Null-terminate and return
    result.append(allocator, 0) catch return -1;
    out_len.* = result.items.len - 1; // exclude null terminator
    const owned = result.toOwnedSlice(allocator) catch return -1;
    out.* = @ptrCast(owned.ptr);
    return 0;
}

fn isCleanupPunct(ch: u8) bool {
    // Match HuggingFace's clean_up_tokenization_spaces behavior
    return switch (ch) {
        '.', '?', '!', ',', '\'', '-' => true,
        else => false,
    };
}

fn isSpecialToken(tokenizer: *ct.Tokenizer, id: i32) bool {
    var cur = tokenizer.added;
    while (cur) |node| {
        if (node.id == id and node.special != 0) return true;
        cur = node.next;
    }
    return false;
}

fn wordpiece_destroy(tokenizer: *ct.Tokenizer) void {
    if (tokenizer.model == null) return;
    const model = @as(*WordPieceModel, @ptrCast(@alignCast(tokenizer.model.?)));
    tokenizer.model = null;

    for (model.vocab_strings.items) |s| model.allocator.free(s);
    model.vocab_strings.deinit(model.allocator);
    model.vocab.deinit(model.allocator);
    if (model.id_to_token.len > 0) model.allocator.free(model.id_to_token);
    model.allocator.destroy(model);
}

fn initTokenizerWithAllocator(allocator: std.mem.Allocator) !*ct.Tokenizer {
    const tokenizer = try allocator.create(ct.Tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.type = ct.ModelType.wordpiece;
    tokenizer.normalizer.lowercase = 1;
    tokenizer.normalizer.nfd = 1;
    tokenizer.postproc.cls_id = -1;
    tokenizer.postproc.sep_id = -1;
    tokenizer.postproc.add_special = 1;
    setFixedString(tokenizer.postproc.cls_token[0..], DEFAULT_CLS);
    setFixedString(tokenizer.postproc.sep_token[0..], DEFAULT_SEP);
    tokenizer.postproc.cls_id = DEFAULT_CLS_ID;
    tokenizer.postproc.sep_id = DEFAULT_SEP_ID;
    return tokenizer;
}

fn initTokenizer() !*ct.Tokenizer {
    return initTokenizerWithAllocator(std.heap.c_allocator);
}

fn attachPretokenizer(tokenizer: *ct.Tokenizer) !void {
    if (tok_fns.tokenizer_pretokenizer_set(&tokenizer.pretokenizer, WORDPIECE_PATTERN.ptr) != 0) {
        tok_fns.tokenizer_set_error(tokenizer, "Failed to compile WordPiece regex");
        return error.PretokenizerInitFailed;
    }
}

fn finalizeSpecialIds(model: *WordPieceModel, tokenizer: *ct.Tokenizer) void {
    if (findId(model, DEFAULT_CLS)) |id| tokenizer.postproc.cls_id = id;
    if (findId(model, DEFAULT_SEP)) |id| tokenizer.postproc.sep_id = id;
    if (findId(model, unkSlice(model))) |id| model.unk_id = id;
}

fn setFixedString(dst: []u8, text: []const u8) void {
    @memset(dst, 0);
    const copy_len = @min(dst.len - 1, text.len);
    std.mem.copyForwards(u8, dst[0..copy_len], text[0..copy_len]);
    dst[copy_len] = 0;
}

fn allocIdToToken(model: *WordPieceModel, size: usize) !void {
    model.id_to_token = try model.allocator.alloc(?[*:0]u8, size);
    for (model.id_to_token) |*slot| slot.* = null;
    model.vocab_size = size;
}

fn buildFromSpec(model: *WordPieceModel, spec: *const ct.WordPieceModelSpec) !void {
    var max_id: usize = 0;
    const vocab_ptr: [*]const ct.TokenIdPair = @ptrCast(spec.vocab.?);
    const vocab = vocab_ptr[0..spec.vocab_len];
    for (vocab) |entry| {
        if (entry.id < 0) continue;
        const next = @as(usize, @intCast(entry.id)) + 1;
        if (next > max_id) max_id = next;
    }
    if (max_id == 0) return error.IncompleteSpec;
    try allocIdToToken(model, max_id);

    for (vocab) |entry| {
        if (entry.token == null or entry.id < 0) continue;
        const token_ptr: [*:0]const u8 = @ptrCast(entry.token.?);
        const token_slice = std.mem.sliceTo(token_ptr, 0);
        addVocabEntry(model, token_slice, entry.id) catch continue;
    }

    if (spec.unk_token) |unk| {
        const unk_ptr: [*:0]const u8 = @ptrCast(unk);
        setUnkToken(model, std.mem.sliceTo(unk_ptr, 0));
    }
}

fn buildFromVocabFile(model: *WordPieceModel, path_z: [*:0]const u8) !void {
    const path = std.mem.sliceTo(path_z, 0);
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(model.allocator, std.math.maxInt(usize));
    defer model.allocator.free(data);

    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line| {
        const trimmed = std.mem.trimRight(u8, line, "\r");
        if (trimmed.len == 0) continue;
        const dup = try model.allocator.dupeZ(u8, trimmed);
        model.vocab_strings.append(model.allocator, dup) catch |err| {
            model.allocator.free(dup);
            return err;
        };
    }
    try allocIdToToken(model, model.vocab_strings.items.len);
    for (model.vocab_strings.items, 0..) |token_z, idx| {
        try model.vocab.put(model.allocator, token_z[0..token_z.len], @intCast(idx));
        model.id_to_token[idx] = token_z.ptr;
    }
}

pub fn tokenizer_wordpiece_create_from_spec(spec: ?*const ct.WordPieceModelSpec) ?*ct.Tokenizer {
    if (spec == null or spec.?.vocab == null or spec.?.vocab_len == 0) return null;
    const allocator = std.heap.c_allocator;
    var tokenizer = initTokenizer() catch return null;
    errdefer allocator.destroy(tokenizer);

    var model = initModel(allocator) catch {
        allocator.destroy(tokenizer);
        return null;
    };
    tokenizer.model = model;
    model.owner = tokenizer;

    attachPretokenizer(tokenizer) catch {
        wordpiece_destroy(tokenizer);
        allocator.destroy(tokenizer);
        return null;
    };

    buildFromSpec(model, spec.?) catch |err| switch (err) {
        error.IncompleteSpec => {
            tok_fns.tokenizer_set_error(tokenizer, "Incomplete WordPiece specification");
            wordpiece_destroy(tokenizer);
            allocator.destroy(tokenizer);
            return null;
        },
        else => {
            wordpiece_destroy(tokenizer);
            allocator.destroy(tokenizer);
            return null;
        },
    };

    finalizeSpecialIds(model, tokenizer);
    return tokenizer;
}

// =============================================================================
// Native Zig Dispatch Entry Points
// =============================================================================

pub fn wordpieceEncode(tokenizer: *ct.Tokenizer, input: []const u8, enc: *ct.TokenizerEncoding) c_int {
    return wordpiece_encode(tokenizer, input, enc);
}

pub fn wordpieceDecode(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
    return wordpiece_decode_impl(tokenizer, ids, ids_len, out, out_len, .{});
}

pub fn wordpieceDecodeWithOptions(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
    return wordpiece_decode_impl(tokenizer, ids, ids_len, out, out_len, options);
}

pub fn wordpieceDestroy(tokenizer: *ct.Tokenizer) void {
    wordpiece_destroy(tokenizer);
}

// =============================================================================
// Tests
// =============================================================================

// Note: Most wordpiece functions (encodeWord, wordpiece_encode, wordpiece_decode, etc.)
// require full model context with vocab and tokenizer state.
// They are tested via integration tests in tests/tokenizer/.

test "setUnkToken copies token to buffer" {
    var model = WordPieceModel{
        .allocator = std.testing.allocator,
        .vocab = .{},
        .id_to_token = &[_]?[*:0]u8{},
        .vocab_strings = .{},
        .vocab_size = 0,
        .unk_id = DEFAULT_UNK_ID,
        .unk_token = undefined,
        .owner = null,
    };

    setUnkToken(&model, DEFAULT_UNK);
    const result = unkSlice(&model);
    try std.testing.expectEqualStrings(DEFAULT_UNK, result);
}

test "initModel creates empty model" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);

    try std.testing.expectEqual(@as(usize, 0), model.vocab.count());
    try std.testing.expectEqualStrings(DEFAULT_UNK, unkSlice(model));
}

test "freeEncodedWordDeep frees all memory" {
    const allocator = std.testing.allocator;

    const ids = try allocator.alloc(i32, 2);
    errdefer allocator.free(ids);
    ids[0] = 1;
    ids[1] = 2;

    const tokens = try allocator.alloc([*:0]u8, 2);
    errdefer allocator.free(tokens);
    const tok1 = try allocator.dupeZ(u8, "hello");
    errdefer allocator.free(tok1);
    const tok2 = try allocator.dupeZ(u8, "world");
    tokens[0] = tok1.ptr;
    tokens[1] = tok2.ptr;

    const encoded = EncodedWord{ .ids = ids, .tokens = tokens };
    freeEncodedWordDeep(allocator, encoded);
    // Should not crash or leak
}

test "setFixedString copies string to buffer" {
    var buffer: [16]u8 = undefined;
    setFixedString(&buffer, "test");

    try std.testing.expectEqual(@as(u8, 't'), buffer[0]);
    try std.testing.expectEqual(@as(u8, 0), buffer[4]);
    try std.testing.expectEqualStrings("test", std.mem.sliceTo(&buffer, 0));
}

test "wordpieceEncode requires integration testing" {
    // This function requires:
    // - Fully initialized WordPiece model with vocab
    // - Greedy longest-match algorithm
    // - ## prefix handling for subword tokens
    // Integration tests: tests/tokenizer/test_*.py
}

test "wordpieceDecode requires integration testing" {
    // This function requires:
    // - Complete WordPiece model with id_to_token mapping
    // - Decoder that removes ## prefixes and joins tokens
    // Integration tests: tests/tokenizer/test_*.py
}

test "wordpieceDestroy requires integration testing" {
    // This function requires:
    // - Fully allocated WordPiece model
    // - Proper cleanup of vocab, strings, and mappings
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_wordpiece_create_from_spec requires integration testing" {
    // This function requires:
    // - Complete WordPieceModelSpec with vocab entries
    // - Model initialization and vocab setup
    // - Pretokenizer attachment with regex pattern
    // - Special token ID finalization (CLS, SEP, UNK)
    // Integration tests: tests/tokenizer/test_*.py
}

test "addVocabEntry adds token to vocab and updates mappings" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    // Setup id_to_token array
    try allocIdToToken(model, 10);

    // Add a vocab entry
    try addVocabEntry(model, "hello", 5);

    // Verify vocab hashmap contains the entry
    const found_id = model.vocab.get("hello");
    try std.testing.expect(found_id != null);
    try std.testing.expectEqual(@as(i32, 5), found_id.?);

    // Verify vocab_strings contains the entry
    try std.testing.expectEqual(@as(usize, 1), model.vocab_strings.items.len);
    try std.testing.expectEqualStrings("hello", model.vocab_strings.items[0]);

    // Verify id_to_token mapping
    try std.testing.expect(model.id_to_token[5] != null);
    const token_slice = std.mem.sliceTo(model.id_to_token[5].?, 0);
    try std.testing.expectEqualStrings("hello", token_slice);
}

test "addVocabEntry handles multiple entries" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    try allocIdToToken(model, 20);

    // Add multiple entries
    try addVocabEntry(model, "hello", 0);
    try addVocabEntry(model, "world", 1);
    try addVocabEntry(model, "##ing", 2);

    // Verify all entries exist
    try std.testing.expectEqual(@as(i32, 0), model.vocab.get("hello").?);
    try std.testing.expectEqual(@as(i32, 1), model.vocab.get("world").?);
    try std.testing.expectEqual(@as(i32, 2), model.vocab.get("##ing").?);
    try std.testing.expectEqual(@as(usize, 3), model.vocab_strings.items.len);
}

test "addVocabEntry rejects negative IDs" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    try allocIdToToken(model, 10);

    // Should return error for negative ID
    const result = addVocabEntry(model, "token", -1);
    try std.testing.expectError(error.InvalidId, result);
}

test "addVocabEntry handles IDs beyond id_to_token bounds" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    try allocIdToToken(model, 5);

    // Add entry with ID beyond current array size
    try addVocabEntry(model, "token", 10);

    // Should still be in vocab hashmap
    try std.testing.expectEqual(@as(i32, 10), model.vocab.get("token").?);
    // But not in id_to_token array
    try std.testing.expect(10 >= model.id_to_token.len);
}

test "findId returns correct ID for existing token" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "test", 7);

    const result = findId(model, "test");
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(i32, 7), result.?);
}

test "findId returns null for non-existent token" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const result = findId(model, "nonexistent");
    try std.testing.expect(result == null);
}

test "findId handles exact string matching" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "hello", 1);
    try addVocabEntry(model, "hell", 2);

    // Should find exact match only
    try std.testing.expectEqual(@as(i32, 1), findId(model, "hello").?);
    try std.testing.expectEqual(@as(i32, 2), findId(model, "hell").?);
    try std.testing.expect(findId(model, "hel") == null);
}

test "encodeWord encodes single-token word" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "hello", 1);

    const result = try encodeWord(model, tokenizer, "hello");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(i32, 1), result.ids[0]);
    try std.testing.expectEqualStrings("hello", std.mem.sliceTo(result.tokens[0], 0));
}

test "encodeWord encodes multi-token word with subword prefix" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "play", 1);
    try addVocabEntry(model, "##ing", 2);

    const result = try encodeWord(model, tokenizer, "playing");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(i32, 1), result.ids[0]);
    try std.testing.expectEqual(@as(i32, 2), result.ids[1]);
    try std.testing.expectEqualStrings("play", std.mem.sliceTo(result.tokens[0], 0));
    try std.testing.expectEqualStrings("##ing", std.mem.sliceTo(result.tokens[1], 0));
}

test "encodeWord uses greedy longest-match algorithm" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "un", 1);
    try addVocabEntry(model, "##want", 2);
    try addVocabEntry(model, "##ed", 3);
    try addVocabEntry(model, "unwanted", 4);

    // Should match "unwanted" greedily rather than "un" + "##want" + "##ed"
    const result = try encodeWord(model, tokenizer, "unwanted");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(i32, 4), result.ids[0]);
    try std.testing.expectEqualStrings("unwanted", std.mem.sliceTo(result.tokens[0], 0));
}

test "encodeWord returns error for unknown word" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "hello", 1);

    // "xyz" is not in vocab
    const result = encodeWord(model, tokenizer, "xyz");
    try std.testing.expectError(error.UnknownWord, result);
}

test "encodeWord handles partial unknown word" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "play", 1);

    // "playing" cannot be fully encoded (missing "##ing")
    const result = encodeWord(model, tokenizer, "playing");
    try std.testing.expectError(error.UnknownWord, result);
}

test "encodeWord handles three-part subword split" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);
    defer {
        for (model.vocab_strings.items) |s| allocator.free(s);
        model.vocab_strings.deinit(allocator);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
    }

    const tokenizer = try initTokenizerWithAllocator(allocator);
    defer allocator.destroy(tokenizer);
    model.owner = tokenizer;

    try allocIdToToken(model, 10);
    try addVocabEntry(model, "test", 1);
    try addVocabEntry(model, "##able", 2);
    try addVocabEntry(model, "##ness", 3);

    const result = try encodeWord(model, tokenizer, "testableness");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 3), result.ids.len);
    try std.testing.expectEqual(@as(i32, 1), result.ids[0]);
    try std.testing.expectEqual(@as(i32, 2), result.ids[1]);
    try std.testing.expectEqual(@as(i32, 3), result.ids[2]);
    try std.testing.expectEqualStrings("test", std.mem.sliceTo(result.tokens[0], 0));
    try std.testing.expectEqualStrings("##able", std.mem.sliceTo(result.tokens[1], 0));
    try std.testing.expectEqualStrings("##ness", std.mem.sliceTo(result.tokens[2], 0));
}
