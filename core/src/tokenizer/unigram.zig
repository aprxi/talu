//! Unigram Model
//!
//! Implements SentencePiece Unigram tokenization.
//! Uses Viterbi search to find the most likely token sequence.

const std = @import("std");
const ct = @import("c_types.zig");
const decoders = @import("decoders.zig");
const utils = @import("utils.zig");

const tok_fns = @import("pipeline.zig");

const UNIGRAM_PATTERN: [:0]const u8 = "[^\\s]+";
const DEFAULT_UNK: [:0]const u8 = "<unk>";

const UniEntry = struct {
    token: [:0]u8,
    score: f32,
    id: i32,
};

pub const UnigramModel = struct {
    allocator: std.mem.Allocator,
    vocab: std.ArrayListUnmanaged(UniEntry),
    id_to_token: []?[*:0]u8,
    vocab_size: usize,
    unk_id: i32,
    bos_id: i32,
    eos_id: i32,
    unk_token: [16]u8,
    unk_entry: ?*UniEntry,
    owner: ?*ct.Tokenizer,
};

const EncodedWord = struct {
    ids: []i32,
    tokens: [][*:0]u8,
};

const SpPiece = struct {
    piece: ?[:0]u8 = null,
    score: f32 = -1.0,
    id: i32 = -1,
    ptype: i32 = 0, // 0 normal, 1 unk, 2 bos, 3 eos
};

fn setUnkToken(model: *UnigramModel, token: []const u8) void {
    utils.setUnkToken(&model.unk_token, token);
}

fn unkSlice(model: *const UnigramModel) []const u8 {
    return utils.unkSlice(&model.unk_token);
}

fn initModel(allocator: std.mem.Allocator) !*UnigramModel {
    const model = try allocator.create(UnigramModel);
    model.* = .{
        .allocator = allocator,
        .vocab = .{},
        .id_to_token = &[_]?[*:0]u8{},
        .vocab_size = 0,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .unk_token = undefined,
        .unk_entry = null,
        .owner = null,
    };
    setUnkToken(model, DEFAULT_UNK);
    return model;
}

fn addEntry(model: *UnigramModel, token: []const u8, score: f32, id: i32) !void {
    const dup = try model.allocator.dupeZ(u8, token);
    errdefer model.allocator.free(dup);
    try model.vocab.append(model.allocator, .{
        .token = dup,
        .score = score,
        .id = id,
    });
    if (@as(usize, @intCast(id)) + 1 > model.vocab_size) {
        model.vocab_size = @as(usize, @intCast(id)) + 1;
    }
}

fn findEntry(model: *UnigramModel, token: []const u8) ?*UniEntry {
    for (model.vocab.items) |*e| {
        if (std.mem.eql(u8, std.mem.sliceTo(e.token, 0), token)) return e;
    }
    return null;
}

fn allocIdToToken(model: *UnigramModel, size: usize) !void {
    model.id_to_token = try model.allocator.alloc(?[*:0]u8, size);
    for (model.id_to_token) |*slot| slot.* = null;
    model.vocab_size = size;
}

fn populateIdToToken(model: *UnigramModel) void {
    for (model.vocab.items) |entry| {
        if (entry.id < 0) continue;
        const idx: usize = @intCast(entry.id);
        if (idx < model.id_to_token.len) {
            model.id_to_token[idx] = entry.token.ptr;
        }
    }
}

fn readVarint(data: []const u8, pos: *usize, out: *u64) !void {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (pos.* < data.len and shift < 64) {
        const byte = data[pos.*];
        pos.* += 1;
        result |= (@as(u64, byte & 0x7F)) << shift;
        if ((byte & 0x80) == 0) {
            out.* = result;
            return;
        }
        shift += 7;
    }
    return error.VarintOverflow;
}

fn skipField(data: []const u8, pos: *usize, wire: u3) !void {
    switch (wire) {
        0 => {
            var discarded: u64 = 0;
            try readVarint(data, pos, &discarded);
        },
        1 => {
            if (pos.* + 8 > data.len) return error.UnexpectedEof;
            pos.* += 8;
        },
        2 => {
            var length: u64 = 0;
            try readVarint(data, pos, &length);
            if (pos.* + length > data.len) return error.UnexpectedEof;
            pos.* += @intCast(length);
        },
        5 => {
            if (pos.* + 4 > data.len) return error.UnexpectedEof;
            pos.* += 4;
        },
        else => return error.BadWire,
    }
}

fn parsePiece(data: []const u8, pos: *usize, out: *SpPiece) !void {
    const end = pos.* + data.len;
    while (pos.* < end) {
        var key: u64 = 0;
        try readVarint(data, pos, &key);
        const field: u32 = @intCast(key >> 3);
        const wire: u3 = @intCast(key & 0x7);
        switch (field) {
            1 => { // piece string
                var slice_len: u64 = 0;
                try readVarint(data, pos, &slice_len);
                if (pos.* + slice_len > end) return error.UnexpectedEof;
                const slice = data[pos.* .. pos.* + @as(usize, @intCast(slice_len))];
                const dup = try outAllocator().dupeZ(u8, slice);
                out.piece = dup;
                pos.* += @intCast(slice_len);
            },
            2 => { // score f32
                if (pos.* + 4 > end) return error.UnexpectedEof;
                var float_buf: [4]u8 = undefined;
                @memcpy(&float_buf, data[pos.* .. pos.* + 4]);
                const raw_bits = std.mem.readInt(u32, &float_buf, .little);
                out.score = @as(f32, @bitCast(raw_bits));
                pos.* += 4;
            },
            3 => { // type varint
                var type_val: u64 = 0;
                try readVarint(data, pos, &type_val);
                out.ptype = @intCast(type_val);
            },
            4 => { // id
                var id_val: u64 = 0;
                try readVarint(data, pos, &id_val);
                out.id = @intCast(id_val);
            },
            else => try skipField(data, pos, wire),
        }
    }
}

fn outAllocator() std.mem.Allocator {
    return std.heap.c_allocator;
}

fn loadSpm(model: *UnigramModel, path_z: [*:0]const u8) !void {
    var file = try std.fs.cwd().openFile(std.mem.sliceTo(path_z, 0), .{});
    defer file.close();
    const data = try file.readToEndAlloc(model.allocator, std.math.maxInt(usize));
    defer model.allocator.free(data);

    var pos: usize = 0;
    var token_id: i32 = 0;
    while (pos < data.len) {
        var key: u64 = 0;
        readVarint(data, &pos, &key) catch break;
        const field: u32 = @intCast(key >> 3);
        const wire: u3 = @intCast(key & 0x7);
        if (field == 1 and wire == 2) {
            var message_len: u64 = 0;
            try readVarint(data, &pos, &message_len);
            if (pos + message_len > data.len) return error.UnexpectedEof;
            var piece = SpPiece{};
            var piece_pos = pos;
            try parsePiece(data[pos .. pos + @as(usize, @intCast(message_len))], &piece_pos, &piece);
            if (piece.piece) |piece_ptr| {
                const piece_text = std.mem.sliceTo(piece_ptr, 0);
                const id_val: i32 = if (piece.id >= 0) piece.id else token_id;
                addEntry(model, piece_text, piece.score, id_val) catch {};
                if (piece.ptype == 1 or std.mem.eql(u8, piece_text, "<unk>")) {
                    model.unk_id = id_val;
                    model.unk_entry = &model.vocab.items[model.vocab.items.len - 1];
                    setUnkToken(model, piece_text);
                } else if (piece.ptype == 2 or std.mem.eql(u8, piece_text, "<s>") or std.mem.eql(u8, piece_text, "<bos>")) {
                    model.bos_id = id_val;
                } else if (piece.ptype == 3 or std.mem.eql(u8, piece_text, "</s>") or std.mem.eql(u8, piece_text, "<eos>")) {
                    model.eos_id = id_val;
                }
                if (piece.id < 0) token_id += 1;
            }
            if (piece.piece) |piece_ptr| outAllocator().free(piece_ptr);
            pos += @intCast(message_len);
        } else {
            try skipField(data, &pos, wire);
        }
    }
    if (model.vocab_size == 0) return error.InvalidVocab;
    try allocIdToToken(model, model.vocab_size);
    populateIdToToken(model);
}

fn loadText(model: *UnigramModel, path_z: [*:0]const u8) !void {
    var file = try std.fs.cwd().openFile(std.mem.sliceTo(path_z, 0), .{});
    defer file.close();
    const data = try file.readToEndAlloc(model.allocator, std.math.maxInt(usize));
    defer model.allocator.free(data);

    var idx: i32 = 0;
    var it = std.mem.splitScalar(u8, data, '\n');
    while (it.next()) |line_raw| {
        const line = std.mem.trimRight(u8, line_raw, "\r");
        if (line.len == 0) continue;
        var parts = std.mem.splitScalar(u8, line, ' ');
        const token_text = parts.next() orelse continue;
        const score_text = parts.next();
        const score: f32 = if (score_text) |score_slice| std.fmt.parseFloat(f32, score_slice) catch -1.0 else -1.0;
        addEntry(model, token_text, score, idx) catch {};
        if (std.mem.eql(u8, token_text, unkSlice(model))) {
            model.unk_id = idx;
            model.unk_entry = &model.vocab.items[model.vocab.items.len - 1];
        }
        idx += 1;
    }
    model.vocab_size = @intCast(idx);
    if (model.vocab_size == 0) return error.InvalidVocab;
    try allocIdToToken(model, model.vocab_size);
    populateIdToToken(model);
}

fn encodeWord(model: *UnigramModel, tokenizer: *ct.Tokenizer, word: []const u8) !EncodedWord {
    const allocator = model.allocator;
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

    const len = word.len;
    const inf: f32 = 1e9;
    var best = try allocator.alloc(f32, len + 1);
    defer allocator.free(best);
    var best_len = try allocator.alloc(usize, len + 1);
    defer allocator.free(best_len);
    var best_entry = try allocator.alloc(?*UniEntry, len + 1);
    defer allocator.free(best_entry);
    for (best, 0..) |*best_score, idx| {
        best_score.* = inf;
        best_len[idx] = 0;
        best_entry[idx] = null;
    }
    best[len] = 0;

    var pos: usize = len;
    while (pos > 0) {
        pos -= 1;
        for (model.vocab.items) |*entry| {
            const token_slice = std.mem.sliceTo(entry.token, 0);
            if (token_slice.len == 0 or pos + token_slice.len > len) continue;
            if (std.mem.eql(u8, word[pos .. pos + token_slice.len], token_slice)) {
                const cand = entry.score + best[pos + token_slice.len];
                if (cand < best[pos]) {
                    best[pos] = cand;
                    best_len[pos] = token_slice.len;
                    best_entry[pos] = entry;
                }
            }
        }
        if (best_entry[pos] == null and model.unk_entry != null) {
            const unk = model.unk_entry.?;
            if (pos + 1 <= len) {
                const cand = unk.score + best[pos + 1];
                if (cand < best[pos]) {
                    best[pos] = cand;
                    best_len[pos] = 1;
                    best_entry[pos] = unk;
                }
            }
        }
    }
    if (best_entry[0] == null) return error.EncodeFailed;

    var ids = try allocator.alloc(i32, len + 1);
    errdefer allocator.free(ids);
    var toks = try allocator.alloc([*:0]u8, len + 1);
    errdefer allocator.free(toks);
    var count: usize = 0;
    errdefer for (toks[0..count]) |t| allocator.free(std.mem.sliceTo(t, 0));
    pos = 0;
    while (pos < len and best_entry[pos] != null) {
        const entry = best_entry[pos].?;
        const token_len = best_len[pos];
        ids[count] = entry.id;
        if (model.unk_entry != null and entry == model.unk_entry.?) {
            const chunk = try allocator.dupeZ(u8, word[pos .. pos + token_len]);
            toks[count] = chunk.ptr;
        } else {
            const dup = try allocator.dupeZ(u8, std.mem.sliceTo(entry.token, 0));
            toks[count] = dup.ptr;
        }
        count += 1;
        pos += token_len;
    }
    return EncodedWord{ .ids = ids[0..count], .tokens = toks[0..count] };
}

fn freeEncodedWordDeep(allocator: std.mem.Allocator, enc: EncodedWord) void {
    for (enc.tokens) |t| allocator.free(std.mem.sliceTo(t, 0));
    allocator.free(enc.tokens);
    allocator.free(enc.ids);
}

fn encodeWordGreedy(model: *UnigramModel, tokenizer: *ct.Tokenizer, word: []const u8) !EncodedWord {
    const allocator = model.allocator;
    const word_z = try allocator.dupeZ(u8, word);
    defer allocator.free(word_z);
    if (tok_fns.tokenizer_added_token_find(tokenizer, word_z.ptr)) |added| {
        const ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(ids);
        const toks = try allocator.alloc([*:0]u8, 1);
        errdefer allocator.free(toks);
        ids[0] = added.*.id;
        const dup = try allocator.dupeZ(u8, word);
        toks[0] = dup.ptr;
        return EncodedWord{ .ids = ids, .tokens = toks };
    }

    var ids = std.ArrayListUnmanaged(i32){};
    defer ids.deinit(allocator);
    var toks = std.ArrayListUnmanaged([*:0]u8){};
    defer toks.deinit(allocator);

    var pos: usize = 0;
    while (pos < word.len) {
        var best_len: usize = 0;
        var best_entry: ?*UniEntry = null;
        var candidate_len: usize = word.len - pos;
        while (candidate_len >= 1) : (candidate_len -= 1) {
            const chunk = word[pos .. pos + candidate_len];
            if (findEntry(model, chunk)) |e| {
                best_entry = e;
                best_len = candidate_len;
                break;
            }
            if (candidate_len == 1) break;
        }
        if (best_entry == null) {
            const unk_chunk = word[pos .. pos + 1];
            try ids.append(allocator, model.unk_id);
            const dup = try allocator.dupeZ(u8, unk_chunk);
            try toks.append(allocator, dup.ptr);
            pos += 1;
            continue;
        }
        try ids.append(allocator, best_entry.?.id);
        const dup = try allocator.dupeZ(u8, std.mem.sliceTo(best_entry.?.token, 0));
        try toks.append(allocator, dup.ptr);
        pos += best_len;
    }

    const ids_owned = try ids.toOwnedSlice(allocator);
    const toks_owned = try allocator.alloc([*:0]u8, toks.items.len);
    @memcpy(toks_owned, toks.items);
    return EncodedWord{ .ids = ids_owned, .tokens = toks_owned };
}

fn unigram_encode(tokenizer: *ct.Tokenizer, text: []const u8, enc: *ct.TokenizerEncoding) c_int {
    if (tokenizer.model == null) return -1;
    const model = @as(*UnigramModel, @ptrCast(@alignCast(tokenizer.model.?)));
    const allocator = model.allocator;

    var ids = std.ArrayListUnmanaged(i32){};
    var toks = std.ArrayListUnmanaged([*:0]u8){};
    var success = false;
    defer {
        if (!success) {
            // Only free tokens on error - on success ownership transfers to enc
            for (toks.items) |t| allocator.free(std.mem.sliceTo(t, 0));
        }
        ids.deinit(allocator);
        toks.deinit(allocator);
    }

    if (model.bos_id >= 0) {
        ids.append(allocator, model.bos_id) catch return -1;
        const dup = allocator.dupeZ(u8, "<s>") catch return -1;
        toks.append(allocator, dup.ptr) catch {
            allocator.free(dup);
            return -1;
        };
    }

    var idx: usize = 0;
    while (idx < text.len) {
        while (idx < text.len and text[idx] == ' ') idx += 1;
        if (idx >= text.len) break;
        const start = idx;
        while (idx < text.len and text[idx] != ' ') idx += 1;
        const word = text[start..idx];

        const encoded = encodeWord(model, tokenizer, word) catch |err| switch (err) {
            error.EncodeFailed => encodeWordGreedy(model, tokenizer, word) catch return -1,
            else => return -1,
        };
        defer freeEncodedWordDeep(allocator, encoded);

        ids.ensureUnusedCapacity(allocator, encoded.ids.len) catch return -1;
        toks.ensureUnusedCapacity(allocator, encoded.tokens.len) catch return -1;
        ids.appendSliceAssumeCapacity(encoded.ids);
        // Append token pointers - dup strings for ownership transfer
        for (encoded.tokens) |t| {
            const dup = allocator.dupeZ(u8, std.mem.sliceTo(t, 0)) catch return -1;
            toks.appendAssumeCapacity(dup.ptr);
        }
    }

    if (model.eos_id >= 0) {
        ids.append(allocator, model.eos_id) catch return -1;
        const dup = allocator.dupeZ(u8, "</s>") catch return -1;
        toks.append(allocator, dup.ptr) catch {
            allocator.free(dup);
            return -1;
        };
    }

    const ids_owned = ids.toOwnedSlice(allocator) catch return -1;
    const toks_owned = allocator.alloc([*:0]u8, toks.items.len) catch {
        allocator.free(ids_owned);
        return -1;
    };
    @memcpy(toks_owned, toks.items);
    enc.ids_len = ids_owned.len;
    enc.tokens_len = toks_owned.len;
    enc.ids = @ptrCast(ids_owned.ptr);
    enc.tokens = @ptrCast(toks_owned.ptr);
    success = true;
    return 0;
}

fn unigram_decode_impl(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
    if (tokenizer.model == null) return -1;
    const model = @as(*UnigramModel, @ptrCast(@alignCast(tokenizer.model.?)));
    const allocator = model.allocator;

    // U+2581 "▁" is encoded as 3 bytes in UTF-8: 0xE2 0x96 0x81
    const SPIECE_UNDERLINE = "\xe2\x96\x81";

    var result = std.ArrayListUnmanaged(u8){};
    defer result.deinit(allocator);

    for (0..ids_len) |id_idx| {
        const id = ids[id_idx];

        // Check if this is a special token (skip if requested)
        if (options.skip_special_tokens) {
            if (isSpecialToken(tokenizer, id)) continue;
        }

        const token_slice = if (id >= 0 and @as(usize, @intCast(id)) < model.id_to_token.len and model.id_to_token[@as(usize, @intCast(id))] != null)
            std.mem.sliceTo(model.id_to_token[@as(usize, @intCast(id))].?, 0)
        else
            unkSlice(model);

        // Replace ▁ (U+2581) with space, copy everything else as-is
        var i: usize = 0;
        while (i < token_slice.len) {
            if (i + 2 < token_slice.len and
                token_slice[i] == SPIECE_UNDERLINE[0] and
                token_slice[i + 1] == SPIECE_UNDERLINE[1] and
                token_slice[i + 2] == SPIECE_UNDERLINE[2])
            {
                result.append(allocator, ' ') catch return -1;
                i += 3;
            } else {
                result.append(allocator, token_slice[i]) catch return -1;
                i += 1;
            }
        }
    }

    // Apply strip_start: remove leading spaces
    var strip_start = @as(usize, @intCast(@max(0, tokenizer.decoder.strip_start)));
    while (strip_start > 0 and result.items.len > 0 and result.items[0] == ' ') {
        _ = result.orderedRemove(0);
        strip_start -= 1;
    }

    // Null-terminate and return
    result.append(allocator, 0) catch return -1;
    out_len.* = result.items.len - 1; // exclude null terminator
    const owned = result.toOwnedSlice(allocator) catch return -1;
    out.* = @ptrCast(owned.ptr);
    return 0;
}

fn isSpecialToken(tokenizer: *ct.Tokenizer, id: i32) bool {
    var cur = tokenizer.added;
    while (cur) |node| {
        if (node.id == id and node.special != 0) return true;
        cur = node.next;
    }
    return false;
}

fn unigram_destroy(tokenizer: *ct.Tokenizer) void {
    if (tokenizer.model == null) return;
    const model = @as(*UnigramModel, @ptrCast(@alignCast(tokenizer.model.?)));
    tokenizer.model = null;
    for (model.vocab.items) |entry| model.allocator.free(entry.token);
    model.vocab.deinit(model.allocator);
    if (model.id_to_token.len > 0) model.allocator.free(model.id_to_token);
    model.allocator.destroy(model);
}

fn initTokenizer() !*ct.Tokenizer {
    const allocator = std.heap.c_allocator;
    const tokenizer = try allocator.create(ct.Tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.type = ct.ModelType.unigram;
    tokenizer.normalizer.lowercase = 0;
    tokenizer.normalizer.nfd = 0;
    tokenizer.postproc.cls_id = -1;
    tokenizer.postproc.sep_id = -1;
    tokenizer.postproc.add_special = 0;
    return tokenizer;
}

fn attachPretokenizer(tokenizer: *ct.Tokenizer) !void {
    if (tok_fns.tokenizer_pretokenizer_set(&tokenizer.pretokenizer, UNIGRAM_PATTERN.ptr) != 0) {
        tok_fns.tokenizer_set_error(tokenizer, "Failed to compile Unigram regex");
        return error.PretokenizerInitFailed;
    }
}

pub fn tokenizer_unigram_create_from_spec(spec: ?*const ct.UnigramModelSpec) ?*ct.Tokenizer {
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
        unigram_destroy(tokenizer);
        allocator.destroy(tokenizer);
        return null;
    };

    var max_id: usize = 0;
    const vocab_ptr: [*]const ct.UnigramVocabEntry = @ptrCast(spec.?.vocab.?);
    const vocab = vocab_ptr[0..spec.?.vocab_len];
    for (vocab) |entry| {
        if (entry.token == null) continue;
        const token_ptr: [*:0]const u8 = @ptrCast(entry.token.?);
        const tok_slice = std.mem.sliceTo(token_ptr, 0);
        addEntry(model, tok_slice, entry.score, entry.id) catch continue;
        const next = @as(usize, @intCast(entry.id)) + 1;
        if (next > max_id) max_id = next;
    }
    if (max_id == 0) {
        tok_fns.tokenizer_set_error(tokenizer, "Incomplete Unigram specification");
        unigram_destroy(tokenizer);
        allocator.destroy(tokenizer);
        return null;
    }
    allocIdToToken(model, max_id) catch {
        tok_fns.tokenizer_set_error(tokenizer, "Allocation failure");
        unigram_destroy(tokenizer);
        allocator.destroy(tokenizer);
        return null;
    };
    populateIdToToken(model);

    if (spec.?.unk_token) |u| {
        const unk_ptr: [*:0]const u8 = @ptrCast(u);
        setUnkToken(model, std.mem.sliceTo(unk_ptr, 0));
        if (findEntry(model, std.mem.sliceTo(unk_ptr, 0))) |e| model.unk_id = e.id;
    }
    if (spec.?.bos_token) |u| {
        const bos_ptr: [*:0]const u8 = @ptrCast(u);
        if (findEntry(model, std.mem.sliceTo(bos_ptr, 0))) |e| model.bos_id = e.id;
    }
    if (spec.?.eos_token) |u| {
        const eos_ptr: [*:0]const u8 = @ptrCast(u);
        if (findEntry(model, std.mem.sliceTo(eos_ptr, 0))) |e| model.eos_id = e.id;
    }

    return tokenizer;
}

// =============================================================================
// Native Zig Dispatch Entry Points
// =============================================================================

pub fn unigramEncode(tokenizer: *ct.Tokenizer, input: []const u8, enc: *ct.TokenizerEncoding) c_int {
    return unigram_encode(tokenizer, input, enc);
}

pub fn unigramDecode(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
    return unigram_decode_impl(tokenizer, ids, ids_len, out, out_len, .{});
}

pub fn unigramDecodeWithOptions(tokenizer: *ct.Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: ct.DecodeOptions) c_int {
    return unigram_decode_impl(tokenizer, ids, ids_len, out, out_len, options);
}

pub fn unigramDestroy(tokenizer: *ct.Tokenizer) void {
    unigram_destroy(tokenizer);
}

// =============================================================================
// Tests
// =============================================================================

// Note: Most unigram functions (encodeWord, unigram_encode, unigram_decode, etc.)
// require full model context with vocab, scores, and tokenizer state.
// They are tested via integration tests in tests/tokenizer/.

test "setUnkToken copies token to buffer" {
    var model = UnigramModel{
        .allocator = std.testing.allocator,
        .vocab = .{},
        .id_to_token = &[_]?[*:0]u8{},
        .vocab_size = 0,
        .unk_id = 0,
        .bos_id = -1,
        .eos_id = -1,
        .unk_token = undefined,
        .unk_entry = null,
        .owner = null,
    };

    setUnkToken(&model, "<unk>");
    const result = unkSlice(&model);
    try std.testing.expectEqualStrings("<unk>", result);
}

test "readVarint decodes varint correctly" {
    var data = [_]u8{ 0x8F, 0x02 }; // 143 + 2*128 = 399
    var pos: usize = 0;
    var value: u64 = 0;

    try readVarint(&data, &pos, &value);

    try std.testing.expectEqual(@as(u64, 271), value);
    try std.testing.expectEqual(@as(usize, 2), pos);
}

test "skipField handles wire type 0 (varint)" {
    var data = [_]u8{ 0x8F, 0x02 }; // varint
    var pos: usize = 0;

    try skipField(&data, &pos, 0); // Wire type 0

    try std.testing.expectEqual(@as(usize, 2), pos);
}

test "initModel creates empty model" {
    const allocator = std.testing.allocator;
    const model = try initModel(allocator);
    defer allocator.destroy(model);

    try std.testing.expectEqual(@as(usize, 0), model.vocab.items.len);
    try std.testing.expectEqualStrings(DEFAULT_UNK, unkSlice(model));
}

test "unigramEncode requires integration testing" {
    // This function requires:
    // - Fully initialized Unigram model with vocab and scores
    // - Viterbi algorithm for optimal segmentation
    // - BOS/EOS token handling
    // Integration tests: tests/tokenizer/test_*.py
}

test "unigramDecode requires integration testing" {
    // This function requires:
    // - Complete Unigram model with id_to_token mapping
    // - Token reconstruction from IDs
    // Integration tests: tests/tokenizer/test_*.py
}

test "unigramDestroy requires integration testing" {
    // This function requires:
    // - Fully allocated Unigram model
    // - Proper cleanup of vocab, strings, and mappings
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_unigram_create_from_spec requires integration testing" {
    // This function requires:
    // - Complete UnigramModelSpec with vocab entries
    // - Model initialization and vocab setup
    // - Pretokenizer attachment with regex pattern
    // Integration tests: tests/tokenizer/test_*.py
}

test "addEntry adds vocab entry and updates vocab_size" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    try addEntry(model, "hello", -1.5, 0);
    try addEntry(model, "world", -2.0, 1);
    try addEntry(model, "test", -1.0, 5);

    try std.testing.expectEqual(@as(usize, 3), model.vocab.items.len);
    try std.testing.expectEqual(@as(usize, 6), model.vocab_size); // max id + 1 = 5 + 1

    try std.testing.expectEqualStrings("hello", std.mem.sliceTo(model.vocab.items[0].token, 0));
    try std.testing.expectEqual(@as(f32, -1.5), model.vocab.items[0].score);
    try std.testing.expectEqual(@as(i32, 0), model.vocab.items[0].id);

    try std.testing.expectEqualStrings("world", std.mem.sliceTo(model.vocab.items[1].token, 0));
    try std.testing.expectEqual(@as(f32, -2.0), model.vocab.items[1].score);
    try std.testing.expectEqual(@as(i32, 1), model.vocab.items[1].id);

    try std.testing.expectEqualStrings("test", std.mem.sliceTo(model.vocab.items[2].token, 0));
    try std.testing.expectEqual(@as(f32, -1.0), model.vocab.items[2].score);
    try std.testing.expectEqual(@as(i32, 5), model.vocab.items[2].id);
}

test "findEntry returns matching entry" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    try addEntry(model, "apple", -1.0, 0);
    try addEntry(model, "banana", -2.0, 1);
    try addEntry(model, "cherry", -3.0, 2);

    const result1 = findEntry(model, "banana");
    try std.testing.expect(result1 != null);
    try std.testing.expectEqualStrings("banana", std.mem.sliceTo(result1.?.token, 0));
    try std.testing.expectEqual(@as(f32, -2.0), result1.?.score);
    try std.testing.expectEqual(@as(i32, 1), result1.?.id);

    const result2 = findEntry(model, "apple");
    try std.testing.expect(result2 != null);
    try std.testing.expectEqualStrings("apple", std.mem.sliceTo(result2.?.token, 0));

    const result3 = findEntry(model, "cherry");
    try std.testing.expect(result3 != null);
    try std.testing.expectEqualStrings("cherry", std.mem.sliceTo(result3.?.token, 0));
}

test "findEntry returns null for non-existent entry" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    try addEntry(model, "apple", -1.0, 0);
    try addEntry(model, "banana", -2.0, 1);

    const result = findEntry(model, "nonexistent");
    try std.testing.expect(result == null);
}

test "findEntry handles empty vocab" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const result = findEntry(model, "anything");
    try std.testing.expect(result == null);
}

test "populateIdToToken maps entries by id" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
        allocator.destroy(model);
    }

    try addEntry(model, "zero", -1.0, 0);
    try addEntry(model, "two", -2.0, 2);
    try addEntry(model, "five", -3.0, 5);

    try allocIdToToken(model, 6);
    populateIdToToken(model);

    try std.testing.expect(model.id_to_token[0] != null);
    try std.testing.expectEqualStrings("zero", std.mem.sliceTo(model.id_to_token[0].?, 0));

    try std.testing.expect(model.id_to_token[1] == null);

    try std.testing.expect(model.id_to_token[2] != null);
    try std.testing.expectEqualStrings("two", std.mem.sliceTo(model.id_to_token[2].?, 0));

    try std.testing.expect(model.id_to_token[3] == null);
    try std.testing.expect(model.id_to_token[4] == null);

    try std.testing.expect(model.id_to_token[5] != null);
    try std.testing.expectEqualStrings("five", std.mem.sliceTo(model.id_to_token[5].?, 0));
}

test "populateIdToToken skips negative ids" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
        allocator.destroy(model);
    }

    try addEntry(model, "valid", -1.0, 0);
    try model.vocab.append(allocator, .{
        .token = try allocator.dupeZ(u8, "negative"),
        .score = -2.0,
        .id = -1,
    });

    try allocIdToToken(model, 1);
    populateIdToToken(model);

    try std.testing.expect(model.id_to_token[0] != null);
    try std.testing.expectEqualStrings("valid", std.mem.sliceTo(model.id_to_token[0].?, 0));
}

test "populateIdToToken handles out of bounds ids" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        if (model.id_to_token.len > 0) allocator.free(model.id_to_token);
        allocator.destroy(model);
    }

    try addEntry(model, "valid", -1.0, 0);
    try addEntry(model, "outofbounds", -2.0, 10);

    try allocIdToToken(model, 5);
    populateIdToToken(model);

    try std.testing.expect(model.id_to_token[0] != null);
    try std.testing.expectEqualStrings("valid", std.mem.sliceTo(model.id_to_token[0].?, 0));
}

test "encodeWordGreedy encodes with longest match first" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const tokenizer = try allocator.create(ct.Tokenizer);
    defer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.model = model;
    model.owner = tokenizer;

    try addEntry(model, "h", -1.0, 0);
    try addEntry(model, "he", -0.5, 1);
    try addEntry(model, "hel", -0.3, 2);
    try addEntry(model, "hello", -0.1, 3);
    try addEntry(model, "world", -0.2, 4);

    const result = try encodeWordGreedy(model, tokenizer, "helloworld");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(i32, 3), result.ids[0]);
    try std.testing.expectEqual(@as(i32, 4), result.ids[1]);
    try std.testing.expectEqualStrings("hello", std.mem.sliceTo(result.tokens[0], 0));
    try std.testing.expectEqualStrings("world", std.mem.sliceTo(result.tokens[1], 0));
}

test "encodeWordGreedy falls back to unk for unknown characters" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const tokenizer = try allocator.create(ct.Tokenizer);
    defer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.model = model;
    model.owner = tokenizer;

    model.unk_id = 999;
    try addEntry(model, "a", -1.0, 0);
    try addEntry(model, "c", -1.0, 1);

    const result = try encodeWordGreedy(model, tokenizer, "abc");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 3), result.ids.len);
    try std.testing.expectEqual(@as(i32, 0), result.ids[0]);
    try std.testing.expectEqual(@as(i32, 999), result.ids[1]);
    try std.testing.expectEqual(@as(i32, 1), result.ids[2]);
    try std.testing.expectEqualStrings("a", std.mem.sliceTo(result.tokens[0], 0));
    try std.testing.expectEqualStrings("b", std.mem.sliceTo(result.tokens[1], 0));
    try std.testing.expectEqualStrings("c", std.mem.sliceTo(result.tokens[2], 0));
}

test "encodeWordGreedy handles single character tokens" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const tokenizer = try allocator.create(ct.Tokenizer);
    defer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.model = model;
    model.owner = tokenizer;

    try addEntry(model, "a", -1.0, 0);
    try addEntry(model, "b", -1.0, 1);
    try addEntry(model, "c", -1.0, 2);

    const result = try encodeWordGreedy(model, tokenizer, "abc");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 3), result.ids.len);
    try std.testing.expectEqual(@as(i32, 0), result.ids[0]);
    try std.testing.expectEqual(@as(i32, 1), result.ids[1]);
    try std.testing.expectEqual(@as(i32, 2), result.ids[2]);
}

test "encodeWordGreedy handles empty word" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const tokenizer = try allocator.create(ct.Tokenizer);
    defer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.model = model;
    model.owner = tokenizer;

    const result = try encodeWordGreedy(model, tokenizer, "");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 0), result.ids.len);
}

test "encodeWordGreedy prefers longer matches over shorter" {
    const allocator = std.testing.allocator;
    var model = try initModel(allocator);
    defer {
        for (model.vocab.items) |entry| allocator.free(entry.token);
        model.vocab.deinit(allocator);
        allocator.destroy(model);
    }

    const tokenizer = try allocator.create(ct.Tokenizer);
    defer allocator.destroy(tokenizer);
    tokenizer.* = std.mem.zeroes(ct.Tokenizer);
    tokenizer.model = model;
    model.owner = tokenizer;

    try addEntry(model, "t", -1.0, 0);
    try addEntry(model, "te", -1.0, 1);
    try addEntry(model, "tes", -1.0, 2);
    try addEntry(model, "test", -1.0, 3);

    const result = try encodeWordGreedy(model, tokenizer, "test");
    defer freeEncodedWordDeep(allocator, result);

    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(i32, 3), result.ids[0]);
    try std.testing.expectEqualStrings("test", std.mem.sliceTo(result.tokens[0], 0));
}
