const std = @import("std");

const added_tokens = @import("added_tokens.zig");
const ct = @import("c_types.zig");

/// Resolve a token ID to the text surface used by tokenize/offset callers.
/// Added tokens are considered after model vocabulary so normal vocab IDs win.
pub fn byId(tokenizer: *const ct.Tokenizer, token_id: i32) []const u8 {
    return tokenizer.idToToken(token_id) orelse
        added_tokens.findContentById(tokenizer, token_id) orelse "";
}

/// Resolve token text from an encoding entry.
/// Encoding token strings take precedence because they can preserve exact
/// encode-time surfaces for unknowns, added tokens, and model-specific paths.
pub fn fromEncoding(
    tokenizer: *const ct.Tokenizer,
    ids: [*]const i32,
    token_strings: ?[*][*c]u8,
    token_index: usize,
) []const u8 {
    if (token_strings) |strings| {
        if (strings[token_index]) |token_ptr| {
            return std.mem.span(@as([*:0]const u8, @ptrCast(token_ptr)));
        }
    }
    return byId(tokenizer, ids[token_index]);
}

test "fromEncoding prefers explicit token string" {
    const token: [:0]const u8 = "encoded-surface";
    var token_strings = [_][*c]u8{@ptrCast(@constCast(token.ptr))};
    const ids = [_]i32{42};
    var tokenizer = std.mem.zeroes(ct.Tokenizer);

    try std.testing.expectEqualStrings(
        "encoded-surface",
        fromEncoding(&tokenizer, &ids, &token_strings, 0),
    );
}

test "byId falls back to added token content" {
    const allocator = std.testing.allocator;
    const json =
        \\{"vocab": {"<unk>": 0}, "merges": []}
    ;

    const tokenizer = try ct.Tokenizer.initBpe(allocator, json, false);
    defer {
        tokenizer.added = null;
        tokenizer.destroy();
        allocator.destroy(tokenizer);
    }

    const content: [:0]const u8 = "<added>";
    var added = ct.AddedToken{
        .content = @ptrCast(@constCast(content.ptr)),
        .id = 7,
        .special = 1,
        .single_word = 0,
        .lstrip = 0,
        .rstrip = 0,
        .normalized = 1,
        .next = null,
    };
    tokenizer.added = &added;

    try std.testing.expectEqualStrings("<added>", byId(tokenizer, 7));
}
