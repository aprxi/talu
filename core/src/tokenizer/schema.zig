//! Tokenizer Schema Types
//!
//! Zig struct definitions for HuggingFace tokenizer.json schema.
//! Used by the loader to parse tokenizer configurations.

const std = @import("std");

pub const TokenId = struct {
    token: []const u8,
    id: i32,
    score: f32, // use -1.0 for BPE/WordPiece entries
};

pub const AddedToken = struct {
    id: i32,
    content: []const u8,
    single_word: bool = false,
    lstrip: bool = false,
    rstrip: bool = false,
    normalized: bool = true,
    special: bool = false,
};

pub const Model = struct {
    type: []const u8,
    vocab: []TokenId,
    merges: ?[][]const u8 = null,
    unk_token: ?[]const u8 = null,
    bos_token: ?[]const u8 = null,
    eos_token: ?[]const u8 = null,
};

pub const Normalizer = struct {
    type: []const u8 = "",
    lowercase: bool = false,
    strip_accents: bool = false,
    nfc: bool = false,
    nfd: bool = false,
    nfkc: bool = false,
    clean_text: bool = false,
    handle_chinese_chars: bool = false,
    // SentencePiece-style normalizers
    prepend: ?[]const u8 = null, // Prepend this string to input
    replace_pattern: ?[]const u8 = null, // Pattern to replace (e.g., " ")
    replace_content: ?[]const u8 = null, // Replacement string (e.g., "▁")
};

pub const PreTokenizer = struct {
    type: []const u8 = "",
    add_prefix_space: bool = false,
    trim_offsets: bool = true,
    use_regex: bool = false,
    // ByteLevel specific
    byte_level: bool = false,
    // Whitespace/Punctuation
    whitespace: bool = false,
    punctuation: bool = false,
    // Custom regex pattern (for Split type)
    pattern: ?[]const u8 = null,
    regex_split: bool = false,
    regex_invert: bool = false,
    // Metaspace: replace spaces with ▁ (U+2581)
    metaspace: bool = false,
};

pub const PostProcessor = struct {
    type: []const u8 = "",
    add_special: bool = false,
    pair: bool = false, // RoBERTa style double SEP
    cls_token: ?[]const u8 = null,
    sep_token: ?[]const u8 = null,
};

pub const Decoder = struct {
    type: []const u8 = "",
    /// Number of leading spaces to strip (from "Strip" decoder "start" field)
    strip_start: i32 = 0,
    /// Number of trailing spaces to strip (from "Strip" decoder "stop" field)
    strip_stop: i32 = 0,
    /// Metaspace decoder: strip leading space added during encode
    add_prefix_space: bool = false,
};

pub const TokenizerRoot = struct {
    version: ?[]const u8 = null,
    model: Model,
    added_tokens: []AddedToken = &.{},
    normalizer: Normalizer = .{},
    pre_tokenizer: PreTokenizer = .{},
    post_processor: PostProcessor = .{},
    decoder: Decoder = .{},
};
