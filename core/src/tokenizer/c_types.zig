//! Tokenizer C Types
//!
//! C-compatible struct and enum definitions for FFI interop.
//! Used by the tokenizer pipeline for cross-language compatibility.

const std = @import("std");

pub const ModelType = enum(c_int) {
    bpe = 1,
    wordpiece = 2,
    unigram = 3,
};

pub const PaddingSide = enum(c_int) {
    right = 0,
    left = 1,
};

pub const TruncationStrategy = enum(c_int) {
    longest_first = 0,
    only_first = 1,
};

pub const PostProcessorKind = enum(c_int) {
    none = 0,
    bert = 1,
    roberta = 2,
    template = 3,
};

pub const Pcre2Code = opaque {};

pub const Offset = extern struct {
    start: i32,
    end: i32,
};

pub const TokenizerEncoding = extern struct {
    ids: ?*i32,
    ids_len: usize,
    tokens: ?*[*c]u8,
    tokens_len: usize,
    attention_mask: ?*i32,
    type_ids: ?*i32,
    special_tokens_mask: ?*i32,
    offsets: ?*Offset,
    overflows: ?*TokenizerEncoding,
    overflow_count: usize,
};

// =============================================================================
// Model type imports for Tokenizer dispatch
// =============================================================================

const bpe = @import("bpe.zig");
const wordpiece = @import("wordpiece.zig");
const unigram = @import("unigram.zig");

/// Decode options for controlling output.
pub const DecodeOptions = struct {
    /// If true, skip special tokens (BOS/EOS/etc.) in the output.
    skip_special_tokens: bool = false,
};

pub const Normalizer = extern struct {
    lowercase: c_int,
    nfc: c_int,
    nfd: c_int,
    nfkc: c_int,
    nfkd: c_int,
    strip_accents: c_int,
    strip_left: c_int,
    strip_right: c_int,
    clean_text: c_int,
    handle_chinese_chars: c_int,
    // SentencePiece-style normalizers
    prepend: ?[*:0]const u8, // Prepend this string to input (e.g., "▁")
    replace_pattern: ?[*:0]const u8, // Pattern to replace (e.g., " ")
    replace_content: ?[*:0]const u8, // Replacement string (e.g., "▁")
};

pub const PreTokenizer = extern struct {
    pattern: ?*u8,
    re: ?*Pcre2Code,
    regex_split: c_int,
    regex_invert: c_int, // For Split with invert=true: emit matches instead of gaps
    byte_level: c_int,
    whitespace: c_int,
    punctuation: c_int,
    lowercase: c_int,
    add_prefix_space: c_int,
    trim_offsets: c_int,
    is_sequence: c_int,
    seq: ?*PreTokenizer,
    seq_count: usize,
};

pub const PostProcessorEntry = extern struct {
    is_special: c_int,
    id: c_int,
    type_id: c_int,
    sequence: c_int,
    token: [64]u8,
};

pub const PostProcessor = extern struct {
    cls_token: [64]u8,
    sep_token: [64]u8,
    cls_id: c_int,
    sep_id: c_int,
    add_special: c_int,
    pair: c_int,
    kind: PostProcessorKind,
    single: [32]PostProcessorEntry,
    single_len: usize,
    pair_tmpl: [64]PostProcessorEntry,
    pair_len: usize,
};

pub const Padding = extern struct {
    enabled: c_int,
    length: c_int,
    pad_id: c_int,
    pad_type_id: c_int,
    pad_token: [64]u8,
    side: PaddingSide,
};

pub const Truncation = extern struct {
    enabled: c_int,
    max_length: c_int,
    strategy: TruncationStrategy,
    stride: c_int,
    overflow_to_sample: c_int,
};

pub const AddedToken = extern struct {
    content: ?*u8,
    id: c_int,
    special: c_int,
    single_word: c_int,
    lstrip: c_int,
    rstrip: c_int,
    normalized: c_int,
    next: ?*AddedToken,
};

/// Decoder configuration parsed from tokenizer.json "decoder" section
pub const Decoder = extern struct {
    /// Number of leading spaces to strip (from Strip decoder "start" field)
    strip_start: c_int,
    /// Number of trailing spaces to strip (from Strip decoder "stop" field)
    strip_stop: c_int,
};

pub const Tokenizer = extern struct {
    model: ?*anyopaque,
    type: ModelType,
    normalizer: Normalizer,
    pretokenizer: PreTokenizer,
    postproc: PostProcessor,
    decoder: Decoder,
    padding: Padding,
    truncation: Truncation,
    added: ?*AddedToken,
    last_error: ?*u8,

    /// Encode with explicit length (supports text with embedded null bytes)
    pub fn encodeSlice(self: *Tokenizer, input: []const u8, enc: *TokenizerEncoding) c_int {
        return switch (self.type) {
            .bpe => blk: {
                const model: *bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.encodeSlice(self, input, enc);
            },
            .wordpiece => wordpiece.wordpieceEncode(self, @ptrCast(input.ptr), enc),
            .unigram => unigram.unigramEncode(self, @ptrCast(input.ptr), enc),
        };
    }

    pub fn decode(self: *Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize) c_int {
        return switch (self.type) {
            .bpe => blk: {
                const model: *bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.decode(self, ids, ids_len, out, out_len);
            },
            .wordpiece => wordpiece.wordpieceDecode(self, ids, ids_len, out, out_len),
            .unigram => unigram.unigramDecode(self, ids, ids_len, out, out_len),
        };
    }

    pub fn decodeWithOptions(self: *Tokenizer, ids: [*c]const i32, ids_len: usize, out: *[*c]u8, out_len: *usize, options: DecodeOptions) c_int {
        return switch (self.type) {
            .bpe => blk: {
                const model: *bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.decodeWithOptions(self, ids, ids_len, out, out_len, .{ .skip_special_tokens = options.skip_special_tokens });
            },
            .wordpiece => wordpiece.wordpieceDecode(self, ids, ids_len, out, out_len),
            .unigram => unigram.unigramDecode(self, ids, ids_len, out, out_len),
        };
    }

    pub fn destroy(self: *Tokenizer) void {
        switch (self.type) {
            .bpe => {
                const model: *bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                model.destroy(self);
            },
            .wordpiece => wordpiece.wordpieceDestroy(self),
            .unigram => unigram.unigramDestroy(self),
        }
    }

    /// Get the vocabulary size from the underlying model.
    pub fn getVocabSize(self: *const Tokenizer) usize {
        return switch (self.type) {
            .bpe => blk: {
                const model: *const bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.vocab_size;
            },
            .wordpiece => blk: {
                const model: *const wordpiece.WordPieceModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.vocab_size;
            },
            .unigram => blk: {
                const model: *const unigram.UnigramModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.vocab_size;
            },
        };
    }

    /// Get the UNK token ID (-1 if not set).
    pub fn getUnkId(self: *const Tokenizer) i32 {
        return switch (self.type) {
            .bpe => blk: {
                const model: *const bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.unk_id;
            },
            .wordpiece => blk: {
                const model: *const wordpiece.WordPieceModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.unk_id;
            },
            .unigram => blk: {
                const model: *const unigram.UnigramModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.unk_id;
            },
        };
    }

    /// Get the BOS token ID (-1 if not set).
    pub fn getBosId(self: *const Tokenizer) i32 {
        return switch (self.type) {
            .bpe => blk: {
                const model: *const bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.bos_id;
            },
            .wordpiece => -1,
            .unigram => blk: {
                const model: *const unigram.UnigramModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.bos_id;
            },
        };
    }

    /// Convert token ID to token string. Returns null if ID is out of range.
    pub fn idToToken(self: *const Tokenizer, id: i32) ?[]const u8 {
        if (id < 0) return null;
        const token_idx: usize = @intCast(id);

        return switch (self.type) {
            .bpe => blk: {
                const model: *const bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                if (token_idx >= model.id_to_token.len) break :blk null;
                break :blk model.id_to_token[token_idx];
            },
            .wordpiece => blk: {
                const model: *const wordpiece.WordPieceModel = @ptrCast(@alignCast(self.model.?));
                if (token_idx >= model.id_to_token.len) break :blk null;
                const ptr = model.id_to_token[token_idx] orelse break :blk null;
                break :blk std.mem.span(ptr);
            },
            .unigram => blk: {
                const model: *const unigram.UnigramModel = @ptrCast(@alignCast(self.model.?));
                if (token_idx >= model.id_to_token.len) break :blk null;
                const ptr = model.id_to_token[token_idx] orelse break :blk null;
                break :blk std.mem.span(ptr);
            },
        };
    }

    /// Convert token string to ID. Returns null if token not found.
    pub fn tokenToId(self: *const Tokenizer, token: []const u8) ?i32 {
        return switch (self.type) {
            .bpe => blk: {
                const model: *const bpe.BpeModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.vocab_hash.get(token);
            },
            .wordpiece => blk: {
                const model: *const wordpiece.WordPieceModel = @ptrCast(@alignCast(self.model.?));
                break :blk model.vocab.get(token);
            },
            .unigram => blk: {
                const model: *const unigram.UnigramModel = @ptrCast(@alignCast(self.model.?));
                for (model.vocab.items) |entry| {
                    if (std.mem.eql(u8, entry.token, token)) {
                        break :blk entry.id;
                    }
                }
                break :blk null;
            },
        };
    }

    /// Create a BPE tokenizer from JSON buffer.
    pub fn initBpe(allocator: std.mem.Allocator, json_buffer: []const u8, json_owned: bool) !*Tokenizer {
        return bpe.createBpeTokenizer(allocator, json_buffer, json_owned);
    }
};

pub const TokenIdPair = extern struct {
    token: [*c]const u8,
    id: c_int,
};

pub const BpeMergePair = extern struct {
    a: [*c]const u8,
    b: [*c]const u8,
};

pub const BpeModelSpec = extern struct {
    vocab: [*c]const TokenIdPair,
    vocab_len: usize,
    merges: [*c]const BpeMergePair,
    merges_len: usize,
    unk_token: [*c]const u8,
};

pub const WordPieceModelSpec = extern struct {
    vocab: [*c]const TokenIdPair,
    vocab_len: usize,
    unk_token: [*c]const u8,
};

pub const UnigramVocabEntry = extern struct {
    token: [*c]const u8,
    score: f32,
    id: c_int,
};

pub const UnigramModelSpec = extern struct {
    vocab: [*c]const UnigramVocabEntry,
    vocab_len: usize,
    unk_token: [*c]const u8,
    bos_token: [*c]const u8,
    eos_token: [*c]const u8,
};

pub const NormalizerSpec = extern struct {
    type: [*c]const u8,
    lowercase: c_int,
    strip_accents: c_int,
    nfc: c_int,
    nfd: c_int,
    nfkc: c_int,
    clean_text: c_int,
    handle_chinese_chars: c_int,
    // SentencePiece-style normalizers
    prepend: [*c]const u8,
    replace_pattern: [*c]const u8,
    replace_content: [*c]const u8,
};

pub const PreTokenizerSpec = extern struct {
    type: [*c]const u8,
    add_prefix_space: c_int,
    trim_offsets: c_int,
    use_regex: c_int,
    byte_level: c_int,
    whitespace: c_int,
    punctuation: c_int,
    pattern: [*c]const u8,
    regex_split: c_int,
    regex_invert: c_int, // For Split with invert=true
};

pub const PostProcessorSpec = extern struct {
    type: [*c]const u8,
    add_special: c_int,
    pair: c_int, // RoBERTa style double SEP
    cls_token: [*c]const u8,
    sep_token: [*c]const u8,
};

// =============================================================================
// C-Compatible Vocabulary Types (for FFI)
// =============================================================================

/// A vocabulary entry for C interop (token string + id).
pub const VocabEntry = struct {
    token: []const u8,
    id: u32,
};

/// A vocabulary list with parallel C arrays (tokens, lengths, ids).
/// Used for tokenizer vocabulary export via FFI.
pub const VocabListC = struct {
    tokens: [][*:0]u8,
    lengths: []u32,
    ids: []u32,

    const Self = @This();

    /// Convert a slice of VocabEntry to C-compatible parallel arrays.
    /// All token strings are copied with null terminators.
    /// Caller owns the result and must call deinit().
    pub fn fromEntries(allocator: std.mem.Allocator, entries: []const VocabEntry) !Self {
        const num_entries = entries.len;
        if (num_entries == 0) {
            return Self{
                .tokens = &.{},
                .lengths = &.{},
                .ids = &.{},
            };
        }

        const tokens = try allocator.alloc([*:0]u8, num_entries);
        errdefer allocator.free(tokens);

        const lengths = try allocator.alloc(u32, num_entries);
        errdefer allocator.free(lengths);

        const ids = try allocator.alloc(u32, num_entries);
        errdefer allocator.free(ids);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                allocator.free(tokens[j][0 .. lengths[j] + 1]);
            }
        }

        for (entries, 0..) |entry, i| {
            const cstr = try allocator.allocSentinel(u8, entry.token.len, 0);
            @memcpy(cstr, entry.token);
            tokens[i] = cstr.ptr;
            lengths[i] = @intCast(entry.token.len);
            ids[i] = entry.id;
            initialized = i + 1;
        }

        return Self{
            .tokens = tokens,
            .lengths = lengths,
            .ids = ids,
        };
    }

    /// Free all token strings and the arrays.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.tokens, self.lengths) |tok, len| {
            allocator.free(tok[0 .. len + 1]);
        }
        if (self.tokens.len > 0) {
            allocator.free(self.tokens);
            allocator.free(self.lengths);
            allocator.free(self.ids);
        }
        self.tokens = &.{};
        self.lengths = &.{};
        self.ids = &.{};
    }

    /// Get the number of entries.
    pub fn count(self: *const Self) usize {
        return self.tokens.len;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "DecodeOptions defaults" {
    const opts = DecodeOptions{};
    try std.testing.expect(opts.skip_special_tokens == false);
}

test "DecodeOptions skip_special_tokens" {
    const opts = DecodeOptions{ .skip_special_tokens = true };
    try std.testing.expect(opts.skip_special_tokens == true);
}

test "ModelType enum values" {
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(ModelType.bpe));
    try std.testing.expectEqual(@as(c_int, 2), @intFromEnum(ModelType.wordpiece));
    try std.testing.expectEqual(@as(c_int, 3), @intFromEnum(ModelType.unigram));
}

test "PaddingSide enum values" {
    try std.testing.expectEqual(@as(c_int, 0), @intFromEnum(PaddingSide.right));
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(PaddingSide.left));
}

test "TruncationStrategy enum values" {
    try std.testing.expectEqual(@as(c_int, 0), @intFromEnum(TruncationStrategy.longest_first));
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(TruncationStrategy.only_first));
}

test "PostProcessorKind enum values" {
    try std.testing.expectEqual(@as(c_int, 0), @intFromEnum(PostProcessorKind.none));
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(PostProcessorKind.bert));
    try std.testing.expectEqual(@as(c_int, 2), @intFromEnum(PostProcessorKind.roberta));
    try std.testing.expectEqual(@as(c_int, 3), @intFromEnum(PostProcessorKind.template));
}

test "Offset struct layout" {
    var offset = std.mem.zeroes(Offset);
    offset.start = 5;
    offset.end = 10;
    try std.testing.expectEqual(@as(i32, 5), offset.start);
    try std.testing.expectEqual(@as(i32, 10), offset.end);
}

test "Decoder struct default values" {
    var decoder = std.mem.zeroes(Decoder);
    decoder.strip_start = 1;
    try std.testing.expectEqual(@as(c_int, 1), decoder.strip_start);
    try std.testing.expectEqual(@as(c_int, 0), decoder.strip_stop);
}

test "VocabListC.fromEntries creates valid parallel arrays" {
    const allocator = std.testing.allocator;
    const entries = &[_]VocabEntry{
        .{ .token = "hello", .id = 0 },
        .{ .token = "world", .id = 1 },
        .{ .token = "test", .id = 42 },
    };

    var list = try VocabListC.fromEntries(allocator, entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), list.count());
    try std.testing.expectEqual(@as(u32, 5), list.lengths[0]);
    try std.testing.expectEqual(@as(u32, 5), list.lengths[1]);
    try std.testing.expectEqual(@as(u32, 4), list.lengths[2]);
    try std.testing.expectEqual(@as(u32, 0), list.ids[0]);
    try std.testing.expectEqual(@as(u32, 1), list.ids[1]);
    try std.testing.expectEqual(@as(u32, 42), list.ids[2]);
}

test "VocabListC.fromEntries handles empty input" {
    const allocator = std.testing.allocator;
    const entries = &[_]VocabEntry{};

    var list = try VocabListC.fromEntries(allocator, entries);
    defer list.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), list.count());
}
