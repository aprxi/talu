//! Tokenizer C API Implementation
//!
//! Internal C-callable functions for tokenizer operations.
//! Used by the pipeline facade; external access is via src/capi/tokenizer.zig.

const std = @import("std");
const ct = @import("c_types.zig");
const encode = @import("encode.zig");
const errors = @import("errors.zig");
const pretokenize = @import("pretokenize.zig");
const strings = @import("strings.zig");
const types = @import("types.zig");

const Allocator = types.Allocator;

// Native Zig API (used by api.zig and internal code)
const loader_mod = @import("loader.zig");

pub fn tokenizer_from_pretrained(path: [*:0]const u8) ?*ct.Tokenizer {
    return loader_mod.tokenizer_loader_from_dir(path);
}

pub fn tokenizer_from_json_string(data: [*:0]const u8) ?*ct.Tokenizer {
    return loader_mod.tokenizer_loader_from_json_string(data);
}

pub const tokenizer_set_error = errors.tokenizer_set_error;

pub fn tokenizer_added_token_add(tokenizer: *ct.Tokenizer, content: ?[*:0]const u8, id: c_int, special: c_int) ?*ct.AddedToken {
    const content_ptr = content orelse return null;
    const added_node = Allocator.create(ct.AddedToken) catch return null;
    const content_copy = strings.tokenizer_strdup(content_ptr);
    added_node.* = .{
        .content = if (content_copy) |copy| @ptrCast(copy) else null,
        .id = id,
        .special = special,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 0,
        .next = tokenizer.added,
    };
    tokenizer.added = added_node;
    return added_node;
}

pub fn tokenizer_added_token_find(tokenizer: *const ct.Tokenizer, content: ?[*:0]const u8) ?*const ct.AddedToken {
    const content_ptr = content orelse return null;
    var cursor = tokenizer.added;
    while (cursor) |node| {
        if (node.content) |node_content| {
            const node_str = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(node_content)), 0);
            const search_str = std.mem.sliceTo(content_ptr, 0);
            if (std.mem.eql(u8, node_str, search_str)) return node;
        }
        cursor = node.next;
    }
    return null;
}

fn tokenizer_added_tokens_free(tokenizer: *ct.Tokenizer) void {
    var cursor = tokenizer.added;
    while (cursor) |node| {
        const next_node = node.next;
        if (node.content) |cptr| Allocator.free(std.mem.span(@as([*:0]u8, @ptrCast(cptr))));
        Allocator.destroy(node);
        cursor = next_node;
    }
    tokenizer.added = null;
}

fn tokenizer_free_impl(tokenizer_opt: ?*ct.Tokenizer) void {
    if (tokenizer_opt == null) return;
    const tokenizer = tokenizer_opt.?;
    tokenizer.destroy();
    tokenizer_added_tokens_free(tokenizer);
    pretokenize.tokenizer_pretokenizer_free(&tokenizer.pretokenizer);
    errors.freeLastError(tokenizer);
    Allocator.destroy(tokenizer);
}

pub fn tokenizer_free(tokenizer_handle: ?*ct.Tokenizer) void {
    tokenizer_free_impl(tokenizer_handle);
}

fn writeEncodingIds(encoding: *ct.TokenizerEncoding, out_ids: ?*i32, out_len: *usize) void {
    out_len.* = encoding.ids_len;
    if (out_ids) |out_ptr| {
        if (encoding.ids) |ids_ptr| {
            const ids_slice: [*]i32 = @ptrCast(ids_ptr);
            const out_slice: [*]i32 = @ptrCast(out_ptr);
            const copy_len = @min(out_len.*, encoding.ids_len);
            @memcpy(out_slice[0..copy_len], ids_slice[0..copy_len]);
        }
    }
}

pub fn tokenizer_encode_ids(tokenizer_handle: ?*ct.Tokenizer, text: [*:0]const u8, out_ids: ?*i32, out_len: *usize) c_int {
    const encoding = tokenizer_encode(tokenizer_handle, text) orelse return -1;
    defer tokenizer_encoding_free(encoding);
    writeEncodingIds(encoding, out_ids, out_len);
    return 0;
}

fn tokenizer_encode(tokenizer_handle: ?*ct.Tokenizer, input: [*:0]const u8) ?*ct.TokenizerEncoding {
    if (tokenizer_handle == null) return null;
    const input_slice = std.mem.sliceTo(input, 0);
    return tokenizer_encode_slice(tokenizer_handle, input_slice);
}

/// Encode text to tokens using a slice (supports text with null bytes).
fn tokenizer_encode_slice(tokenizer_handle: ?*ct.Tokenizer, input: []const u8) ?*ct.TokenizerEncoding {
    // tokenize() is raw tokenization — never add special tokens.
    // Callers wanting special tokens use tokenizer_encode_slice_with_options.
    return tokenizer_encode_slice_with_options(tokenizer_handle, input, .{ .add_special_tokens = false });
}

/// Encode text to tokens with options (thread-safe).
fn tokenizer_encode_slice_with_options(
    tokenizer_handle: ?*ct.Tokenizer,
    input: []const u8,
    options: encode.EncodeOptions,
) ?*ct.TokenizerEncoding {
    if (tokenizer_handle == null) return null;
    const encoding = Allocator.create(ct.TokenizerEncoding) catch return null;
    encoding.* = std.mem.zeroes(ct.TokenizerEncoding);
    if (encode.tokenizer_encode_struct_with_options(tokenizer_handle.?, input, encoding, options) != 0) {
        encode.tokenizer_encoding_free_struct(encoding);
        Allocator.destroy(encoding);
        return null;
    }
    return encoding;
}

/// Encode text to token IDs using a slice (supports text with null bytes).
pub fn tokenizer_encode_ids_slice(tokenizer_handle: ?*ct.Tokenizer, text: []const u8, out_ids: ?*i32, out_len: *usize) c_int {
    const encoding = tokenizer_encode_slice(tokenizer_handle, text) orelse return -1;
    defer tokenizer_encoding_free(encoding);
    writeEncodingIds(encoding, out_ids, out_len);
    return 0;
}

/// Encode text to token IDs with options (thread-safe).
pub fn tokenizer_encode_ids_slice_with_options(
    tokenizer_handle: ?*ct.Tokenizer,
    text: []const u8,
    out_ids: ?*i32,
    out_len: *usize,
    add_special_tokens: bool,
) c_int {
    const options = encode.EncodeOptions{ .add_special_tokens = add_special_tokens };
    const encoding = tokenizer_encode_slice_with_options(tokenizer_handle, text, options) orelse return -1;
    defer tokenizer_encoding_free(encoding);
    writeEncodingIds(encoding, out_ids, out_len);
    return 0;
}

fn tokenizer_encoding_free(encoding_handle: ?*ct.TokenizerEncoding) void {
    if (encoding_handle == null) return;
    encode.tokenizer_encoding_free_struct(encoding_handle.?);
    Allocator.destroy(encoding_handle.?);
}

pub fn tokenizer_decode(tokenizer_handle: ?*ct.Tokenizer, ids: [*]const i32, id_count: usize, out: *[*c]u8, out_len: *usize) c_int {
    if (tokenizer_handle == null) return -1;
    return tokenizer_handle.?.decode(ids, id_count, out, out_len);
}

/// Decode token IDs to text with options (thread-safe).
pub fn tokenizer_decode_with_options(
    tokenizer_handle: ?*ct.Tokenizer,
    ids: [*]const i32,
    id_count: usize,
    out: *[*c]u8,
    out_len: *usize,
    skip_special_tokens: bool,
) c_int {
    if (tokenizer_handle == null) return -1;
    const options = ct.DecodeOptions{ .skip_special_tokens = skip_special_tokens };
    return tokenizer_handle.?.decodeWithOptions(ids, id_count, out, out_len, options);
}

/// Tokenize text to token strings (without converting to IDs).
/// Returns the token strings that would be produced by encoding.
/// out_tokens: array to receive token string pointers (caller must free each with tokenizer_string_free)
/// out_len: receives number of tokens
pub fn tokenizer_tokenize(tokenizer_handle: ?*ct.Tokenizer, text: []const u8, out_tokens: ?[*][*:0]u8, out_len: *usize) c_int {
    const encoding = tokenizer_encode_slice(tokenizer_handle, text) orelse return -1;
    defer tokenizer_encoding_free(encoding);

    out_len.* = encoding.tokens_len;

    if (out_tokens) |out_ptr| {
        if (encoding.tokens) |tokens_ptr| {
            const token_ptrs: [*][*c]u8 = @ptrCast(tokens_ptr);
            const copy_len = @min(out_len.*, encoding.tokens_len);

            // Duplicate each token string (caller owns these)
            for (0..copy_len) |token_idx| {
                if (token_ptrs[token_idx]) |token_ptr| {
                    const token_bytes = std.mem.span(@as([*:0]u8, @ptrCast(token_ptr)));
                    out_ptr[token_idx] = strings.dupTokenString(token_bytes) orelse return -1;
                } else {
                    // Empty token
                    out_ptr[token_idx] = strings.dupTokenString("") orelse return -1;
                }
            }
        }
    }
    return 0;
}

pub fn tokenizer_string_free(s: ?[*:0]u8) void {
    if (s) |token_ptr| Allocator.free(std.mem.sliceTo(token_ptr, 0));
}

pub fn tokenizer_string_free_with_len(s: ?[*]u8, len: usize) void {
    if (s) |token_ptr| {
        if (len > 0) Allocator.free(token_ptr[0..len]);
    }
}

pub fn tokenizer_get_last_error(tokenizer_handle: ?*ct.Tokenizer) ?[*:0]const u8 {
    if (tokenizer_handle == null) return null;
    return @ptrCast(tokenizer_handle.?.last_error);
}

// =============================================================================
// Tests
// =============================================================================

// Note: All functions in c_api.zig are C API wrappers that require full
// tokenizer initialization with vocab, models, and external dependencies.
// They are comprehensively tested via integration tests in tests/tokenizer/.
// These functions are designed to be called from C/Python, not as standalone
// units, so integration testing provides better coverage.

test "tokenizer_from_pretrained requires integration testing" {
    // This function requires:
    // - File I/O access to model directory
    // - Full tokenizer loading and initialization
    // - Complete vocab, model, normalizer, pretokenizer setup
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_from_json_string requires integration testing" {
    // This function requires:
    // - Complete JSON tokenizer specification
    // - Model initialization (BPE/WordPiece/Unigram)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_added_token_add requires integration testing" {
    // This function requires:
    // - Full tokenizer context
    // - Proper memory management for added tokens
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_added_token_find requires integration testing" {
    // This function requires:
    // - Tokenizer with added tokens
    // - String matching and lookup
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_free requires integration testing" {
    // This function requires:
    // - Fully allocated tokenizer
    // - Proper cleanup of all resources (vocab, model, pretokenizer, etc.)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_encode_ids requires integration testing" {
    // This function requires:
    // - Complete tokenizer with vocab and model
    // - Encoding pipeline (normalize, pretokenize, encode, postprocess)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_encode_ids_slice requires integration testing" {
    // This function requires:
    // - Complete tokenizer context
    // - Support for null bytes in input
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_encode_ids_slice_with_options requires integration testing" {
    // This function requires:
    // - Complete tokenizer context
    // - Thread-safe encoding with options (add_special_tokens)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_decode requires integration testing" {
    // This function requires:
    // - Complete tokenizer with vocab mapping
    // - Model-specific decode implementation
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_decode_with_options requires integration testing" {
    // This function requires:
    // - Complete tokenizer context
    // - Thread-safe decoding with options (skip_special_tokens)
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_tokenize requires integration testing" {
    // This function requires:
    // - Complete tokenizer with encoding pipeline
    // - Token string extraction without ID conversion
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_string_free requires integration testing" {
    // This function requires:
    // - Allocated token strings from tokenizer_tokenize
    // - Proper memory management
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_string_free_with_len requires integration testing" {
    // This function requires:
    // - Allocated strings with known length
    // - Proper memory management
    // Integration tests: tests/tokenizer/test_*.py
}

test "tokenizer_get_last_error requires integration testing" {
    // This function requires:
    // - Tokenizer with error state
    // - Error message retrieval
    // Integration tests: tests/tokenizer/test_*.py
}
