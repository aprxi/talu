//! Tokenizer C API Implementation
//!
//! Internal C-callable functions for tokenizer operations.
//! Used by the pipeline facade; external access is via src/capi/tokenizer.zig.

const std = @import("std");
const ct = @import("c_types.zig");
const encode = @import("encode.zig");
const errors = @import("errors.zig");
const normalize = @import("normalize.zig");
const pretokenize = @import("pretokenize.zig");
const strings = @import("strings.zig");
const token_surface = @import("token_surface.zig");
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
        .normalized = 1,
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
    normalize.tokenizer_normalizer_free(&tokenizer.normalizer);
    pretokenize.tokenizer_pretokenizer_free(&tokenizer.pretokenizer);
    errors.freeLastError(tokenizer);
    Allocator.destroy(tokenizer);
}

pub fn tokenizer_free(tokenizer_handle: ?*ct.Tokenizer) void {
    tokenizer_free_impl(tokenizer_handle);
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

fn tokenizer_encoding_free(encoding_handle: ?*ct.TokenizerEncoding) void {
    if (encoding_handle == null) return;
    encode.tokenizer_encoding_free_struct(encoding_handle.?);
    Allocator.destroy(encoding_handle.?);
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
    const tok = tokenizer_handle orelse return -1;
    const encoding = tokenizer_encode_slice(tok, text) orelse return -1;
    defer tokenizer_encoding_free(encoding);

    out_len.* = encoding.ids_len;

    if (out_tokens) |out_ptr| {
        const ids_ptr: [*]i32 = if (encoding.ids) |ids| @ptrCast(ids) else return 0;
        const token_ptrs: ?[*][*c]u8 = if (encoding.tokens) |t| @ptrCast(t) else null;

        for (0..encoding.ids_len) |token_idx| {
            const token_bytes = token_surface.fromEncoding(tok, ids_ptr, token_ptrs, token_idx);
            out_ptr[token_idx] = strings.dupTokenString(token_bytes) orelse return -1;
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
