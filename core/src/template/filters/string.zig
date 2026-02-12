//! String Filters
//!
//! Filters that operate on string values: transformation, formatting, encoding.

const std = @import("std");
const types = @import("types.zig");

const TemplateInput = types.TemplateInput;
const EvalError = types.EvalError;
const Evaluator = types.Evaluator;
const Expr = types.Expr;

// ============================================================================
// UTF-8 Helpers
// ============================================================================

/// Count UTF-8 codepoints in a string
fn countCodepoints(s: []const u8) usize {
    var count: usize = 0;
    var iter = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (iter.nextCodepointSlice()) |_| {
        count += 1;
    }
    return count;
}

// ============================================================================
// Case Transformation
// ============================================================================

pub fn filterLower(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };
    var result = e.ctx.arena.allocator().alloc(u8, input_str.len) catch return EvalError.OutOfMemory;
    for (input_str, 0..) |c, i| result[i] = std.ascii.toLower(c);
    return .{ .string = result };
}

pub fn filterUpper(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };
    var result = e.ctx.arena.allocator().alloc(u8, input_str.len) catch return EvalError.OutOfMemory;
    for (input_str, 0..) |c, i| result[i] = std.ascii.toUpper(c);
    return .{ .string = result };
}

pub fn filterCapitalize(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };
    if (input_str.len == 0) return value;
    var result = e.ctx.arena.allocator().alloc(u8, input_str.len) catch return EvalError.OutOfMemory;
    result[0] = std.ascii.toUpper(input_str[0]);
    for (input_str[1..], 1..) |c, i| result[i] = std.ascii.toLower(c);
    return .{ .string = result };
}

pub fn filterTitle(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    if (input_str.len == 0) return value;

    var result = e.ctx.arena.allocator().alloc(u8, input_str.len) catch return EvalError.OutOfMemory;
    var capitalize_next = true;

    for (input_str, 0..) |c, i| {
        if (std.ascii.isAlphabetic(c)) {
            result[i] = if (capitalize_next) std.ascii.toUpper(c) else std.ascii.toLower(c);
            capitalize_next = false;
        } else {
            result[i] = c;
            capitalize_next = (c == ' ' or c == '\t' or c == '\n' or c == '-' or c == '_');
        }
    }

    return .{ .string = result };
}

// ============================================================================
// Whitespace
// ============================================================================

pub fn filterTrim(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };
    _ = args;
    const trimmed = std.mem.trim(u8, input_str, " \t\n\r");
    const result = e.ctx.arena.allocator().dupe(u8, trimmed) catch return EvalError.OutOfMemory;
    return .{ .string = result };
}

// ============================================================================
// Search and Replace
// ============================================================================

pub fn filterReplace(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    if (args.len < 2) return EvalError.TypeError;

    const old_val = try e.evalExpr(args[0]);
    const new_val = try e.evalExpr(args[1]);

    const old = switch (old_val) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };
    const new = switch (new_val) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    const result = std.mem.replaceOwned(u8, e.ctx.arena.allocator(), input_str, old, new) catch return EvalError.OutOfMemory;
    return .{ .string = result };
}

pub fn filterSplit(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    // Get separator (default: single space)
    var sep: []const u8 = " ";
    if (args.len > 0) {
        const sep_val = try e.evalExpr(args[0]);
        sep = switch (sep_val) {
            .string => |s| s,
            else => return EvalError.TypeError,
        };
    }

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(TemplateInput){};

    if (sep.len == 0) {
        // Empty separator: split into UTF-8 codepoints (not bytes)
        var iter = std.unicode.Utf8View.initUnchecked(input_str).iterator();
        while (iter.nextCodepointSlice()) |cp_slice| {
            const char_str = arena.dupe(u8, cp_slice) catch return EvalError.OutOfMemory;
            result.append(arena, .{ .string = char_str }) catch return EvalError.OutOfMemory;
        }
    } else {
        // Split by separator
        var iter = std.mem.splitSequence(u8, input_str, sep);
        while (iter.next()) |part| {
            const part_copy = arena.dupe(u8, part) catch return EvalError.OutOfMemory;
            result.append(arena, .{ .string = part_copy }) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .array = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// HTML/XML Encoding
// ============================================================================

pub fn filterEscape(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    for (input_str) |c| {
        switch (c) {
            '&' => result.appendSlice(arena, "&amp;") catch return EvalError.OutOfMemory,
            '<' => result.appendSlice(arena, "&lt;") catch return EvalError.OutOfMemory,
            '>' => result.appendSlice(arena, "&gt;") catch return EvalError.OutOfMemory,
            '"' => result.appendSlice(arena, "&quot;") catch return EvalError.OutOfMemory,
            '\'' => result.appendSlice(arena, "&#39;") catch return EvalError.OutOfMemory,
            else => result.append(arena, c) catch return EvalError.OutOfMemory,
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterSafe(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    // safe() marks content as safe - in our context we just pass through
    return value;
}

pub fn filterStriptags(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};
    var in_tag = false;

    for (input_str) |c| {
        if (c == '<') {
            in_tag = true;
        } else if (c == '>') {
            in_tag = false;
        } else if (!in_tag) {
            result.append(arena, c) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// URL Encoding
// ============================================================================

pub fn filterUrlencode(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    for (input_str) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            // Unreserved characters - pass through
            result.append(arena, c) catch return EvalError.OutOfMemory;
        } else if (c == ' ') {
            // Space becomes +
            result.append(arena, '+') catch return EvalError.OutOfMemory;
        } else {
            // Percent-encode everything else
            result.append(arena, '%') catch return EvalError.OutOfMemory;
            const hex = "0123456789ABCDEF";
            result.append(arena, hex[c >> 4]) catch return EvalError.OutOfMemory;
            result.append(arena, hex[c & 0x0F]) catch return EvalError.OutOfMemory;
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterUrlize(e: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |s| s,
        else => return EvalError.TypeError,
    };

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};

    var i: usize = 0;
    while (i < input_str.len) {
        // Check for URL patterns
        const remaining = input_str[i..];
        var url_start: ?usize = null;
        var url_len: usize = 0;

        if (std.mem.startsWith(u8, remaining, "http://") or std.mem.startsWith(u8, remaining, "https://")) {
            url_start = i;
            // Find end of URL (whitespace or end of string)
            var j: usize = 0;
            while (j < remaining.len and !std.ascii.isWhitespace(remaining[j]) and remaining[j] != '<' and remaining[j] != '>') : (j += 1) {}
            url_len = j;
        } else if (std.mem.startsWith(u8, remaining, "www.")) {
            url_start = i;
            var j: usize = 0;
            while (j < remaining.len and !std.ascii.isWhitespace(remaining[j]) and remaining[j] != '<' and remaining[j] != '>') : (j += 1) {}
            url_len = j;
        }

        if (url_start != null and url_len > 0) {
            const url = remaining[0..url_len];
            // Write <a href="url">url</a>
            result.appendSlice(arena, "<a href=\"") catch return EvalError.OutOfMemory;
            if (std.mem.startsWith(u8, url, "www.")) {
                result.appendSlice(arena, "http://") catch return EvalError.OutOfMemory;
            }
            result.appendSlice(arena, url) catch return EvalError.OutOfMemory;
            result.appendSlice(arena, "\">") catch return EvalError.OutOfMemory;
            result.appendSlice(arena, url) catch return EvalError.OutOfMemory;
            result.appendSlice(arena, "</a>") catch return EvalError.OutOfMemory;
            i += url_len;
        } else {
            result.append(arena, input_str[i]) catch return EvalError.OutOfMemory;
            i += 1;
        }
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

// ============================================================================
// Formatting
// ============================================================================

pub fn filterWordwrap(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    var width: usize = 79;
    if (args.len > 0) {
        const width_val = try e.evalExpr(args[0]);
        width = switch (width_val) {
            .integer => |int_val| if (int_val > 0) @intCast(int_val) else 79,
            else => 79,
        };
    }

    const arena = e.ctx.arena.allocator();
    var result = std.ArrayListUnmanaged(u8){};
    var line_len: usize = 0;

    var iter = std.mem.splitScalar(u8, input_str, ' ');
    var first_word = true;
    while (iter.next()) |word| {
        if (!first_word and line_len + word.len + 1 > width) {
            result.append(arena, '\n') catch return EvalError.OutOfMemory;
            line_len = 0;
        } else if (!first_word) {
            result.append(arena, ' ') catch return EvalError.OutOfMemory;
            line_len += 1;
        }
        result.appendSlice(arena, word) catch return EvalError.OutOfMemory;
        line_len += word.len;
        first_word = false;
    }

    return .{ .string = result.toOwnedSlice(arena) catch return EvalError.OutOfMemory };
}

pub fn filterCenter(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    var width: usize = 80;
    if (args.len > 0) {
        const width_val = try e.evalExpr(args[0]);
        width = switch (width_val) {
            .integer => |int_val| if (int_val > 0) @intCast(int_val) else 80,
            else => 80,
        };
    }

    // Count codepoints, not bytes
    const str_codepoints = countCodepoints(input_str);

    if (str_codepoints >= width) return value;

    const arena = e.ctx.arena.allocator();
    const padding = width - str_codepoints;
    const left_pad = padding / 2;
    const right_pad = padding - left_pad;

    // Result size: left_pad spaces + original bytes + right_pad spaces
    var result = arena.alloc(u8, left_pad + input_str.len + right_pad) catch return EvalError.OutOfMemory;
    @memset(result[0..left_pad], ' ');
    @memcpy(result[left_pad .. left_pad + input_str.len], input_str);
    @memset(result[left_pad + input_str.len ..], ' ');

    return .{ .string = result };
}

pub fn filterTruncate(e: *Evaluator, value: TemplateInput, args: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    var length: usize = 255;
    var end: []const u8 = "...";

    if (args.len > 0) {
        const len_val = try e.evalExpr(args[0]);
        length = switch (len_val) {
            .integer => |int_val| if (int_val > 0) @intCast(int_val) else 255,
            else => 255,
        };
    }
    if (args.len > 1) {
        const end_val = try e.evalExpr(args[1]);
        end = switch (end_val) {
            .string => |str| str,
            else => "...",
        };
    }

    // Count codepoints in suffix
    const end_codepoints = countCodepoints(end);

    // Count codepoints in input and find truncation point
    var codepoint_count: usize = 0;
    var byte_pos: usize = 0;
    var iter = std.unicode.Utf8View.initUnchecked(input_str).iterator();
    while (iter.nextCodepointSlice()) |cp_slice| {
        codepoint_count += 1;
        byte_pos += cp_slice.len;
    }

    // If string is short enough (in codepoints), return as-is
    if (codepoint_count <= length) return value;

    // Find byte position for truncation (length - suffix codepoints)
    const trunc_codepoints = if (length > end_codepoints) length - end_codepoints else 0;
    var trunc_byte_pos: usize = 0;
    var cp_count: usize = 0;
    var iter2 = std.unicode.Utf8View.initUnchecked(input_str).iterator();
    while (iter2.nextCodepointSlice()) |cp_slice| {
        if (cp_count >= trunc_codepoints) break;
        trunc_byte_pos += cp_slice.len;
        cp_count += 1;
    }

    const arena = e.ctx.arena.allocator();
    var result = arena.alloc(u8, trunc_byte_pos + end.len) catch return EvalError.OutOfMemory;
    @memcpy(result[0..trunc_byte_pos], input_str[0..trunc_byte_pos]);
    @memcpy(result[trunc_byte_pos..], end);

    return .{ .string = result };
}

pub fn filterWordcount(_: *Evaluator, value: TemplateInput, _: []const *const Expr) EvalError!TemplateInput {
    const input_str = switch (value) {
        .string => |str| str,
        else => return EvalError.TypeError,
    };

    var count: i64 = 0;
    var in_word = false;
    for (input_str) |c| {
        const is_space = (c == ' ' or c == '\t' or c == '\n' or c == '\r');
        if (!is_space and !in_word) {
            count += 1;
            in_word = true;
        } else if (is_space) {
            in_word = false;
        }
    }

    return .{ .integer = count };
}

// ============================================================================
// Unit Tests
// ============================================================================

const TemplateParser = @import("../eval.zig").TemplateParser;

test "filterUpper" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const result = try filterUpper(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("HELLO WORLD", result.string);
}

test "filterLower" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "HELLO WORLD" };
    const result = try filterLower(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello world", result.string);
}

test "filterTrim" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "  hello  " };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterTrim - strip alias" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "\t\nhello\n\t" };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterReplace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const old_expr = Expr{ .string = "world" };
    const new_expr = Expr{ .string = "there" };
    const args = [_]*const Expr{ &old_expr, &new_expr };
    const result = try filterReplace(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("hello there", result.string);
}

test "filterTitle" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const result = try filterTitle(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Hello World", result.string);
}

test "filterCapitalize" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello WORLD" };
    const result = try filterCapitalize(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Hello world", result.string);
}

test "filterEscape - html entities" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<script>alert('xss')</script>" };
    const result = try filterEscape(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;", result.string);
}

test "filterWordcount" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world foo bar" };
    const result = try filterWordcount(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 4), result.integer);
}

test "filterSplit - with comma" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "a,b,c" };
    const sep_expr = Expr{ .string = "," };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterSplit(&eval_ctx, input, &args);
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("a", result.array[0].string);
    try std.testing.expectEqualStrings("b", result.array[1].string);
    try std.testing.expectEqualStrings("c", result.array[2].string);
}

test "filterSplit - default separator" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world foo" };
    const result = try filterSplit(&eval_ctx, input, &.{});
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("hello", result.array[0].string);
    try std.testing.expectEqualStrings("world", result.array[1].string);
    try std.testing.expectEqualStrings("foo", result.array[2].string);
}

test "filterSplit - empty separator" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "abc" };
    const sep_expr = Expr{ .string = "" };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterSplit(&eval_ctx, input, &args);
    try std.testing.expectEqual(@as(usize, 3), result.array.len);
    try std.testing.expectEqualStrings("a", result.array[0].string);
    try std.testing.expectEqualStrings("b", result.array[1].string);
    try std.testing.expectEqualStrings("c", result.array[2].string);
}

test "filterSplit - empty separator UTF-8" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // "你好" is 2 codepoints (6 bytes) - should split into 2 items, not 6
    const input = TemplateInput{ .string = "你好" };
    const sep_expr = Expr{ .string = "" };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterSplit(&eval_ctx, input, &args);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqualStrings("你", result.array[0].string);
    try std.testing.expectEqualStrings("好", result.array[1].string);
}

test "filterUrlize - with http" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Visit http://example.com for more" };
    const result = try filterUrlize(&eval_ctx, input, &.{});
    try std.testing.expect(std.mem.indexOf(u8, result.string, "<a href=\"http://example.com\">http://example.com</a>") != null);
}

test "filterUrlize - with www" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Visit www.example.com today" };
    const result = try filterUrlize(&eval_ctx, input, &.{});
    try std.testing.expect(std.mem.indexOf(u8, result.string, "<a href=\"http://www.example.com\">www.example.com</a>") != null);
}

test "filterLower - mixed case" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "HeLLo WoRLd" };
    const result = try filterLower(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello world", result.string);
}

test "filterLower - already lowercase" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "already lowercase" };
    const result = try filterLower(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("already lowercase", result.string);
}

test "filterLower - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterLower(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterUpper - mixed case" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "HeLLo WoRLd" };
    const result = try filterUpper(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("HELLO WORLD", result.string);
}

test "filterUpper - already uppercase" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "ALREADY UPPERCASE" };
    const result = try filterUpper(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("ALREADY UPPERCASE", result.string);
}

test "filterUpper - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterUpper(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterTrim - leading whitespace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "   hello" };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterTrim - trailing whitespace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello   " };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterTrim - mixed whitespace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = " \t\n\rhello world\r\n\t " };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello world", result.string);
}

test "filterTrim - no whitespace" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterReplace - multiple occurrences" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "foo bar foo baz foo" };
    const old_expr = Expr{ .string = "foo" };
    const new_expr = Expr{ .string = "qux" };
    const args = [_]*const Expr{ &old_expr, &new_expr };
    const result = try filterReplace(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("qux bar qux baz qux", result.string);
}

test "filterReplace - no matches" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const old_expr = Expr{ .string = "xyz" };
    const new_expr = Expr{ .string = "abc" };
    const args = [_]*const Expr{ &old_expr, &new_expr };
    const result = try filterReplace(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("hello world", result.string);
}

test "filterReplace - empty replacement" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const old_expr = Expr{ .string = " world" };
    const new_expr = Expr{ .string = "" };
    const args = [_]*const Expr{ &old_expr, &new_expr };
    const result = try filterReplace(&eval_ctx,input, &args);
    try std.testing.expectEqualStrings("hello", result.string);
}

test "filterSplit - multi-word string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "one two three four" };
    const result = try filterSplit(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(usize, 4), result.array.len);
    try std.testing.expectEqualStrings("one", result.array[0].string);
    try std.testing.expectEqualStrings("two", result.array[1].string);
    try std.testing.expectEqualStrings("three", result.array[2].string);
    try std.testing.expectEqualStrings("four", result.array[3].string);
}

test "filterSplit - consecutive separators" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "a,,b,,c" };
    const sep_expr = Expr{ .string = "," };
    const args = [_]*const Expr{&sep_expr};
    const result = try filterSplit(&eval_ctx,input, &args);
    try std.testing.expectEqual(@as(usize, 5), result.array.len);
    try std.testing.expectEqualStrings("a", result.array[0].string);
    try std.testing.expectEqualStrings("", result.array[1].string);
    try std.testing.expectEqualStrings("b", result.array[2].string);
    try std.testing.expectEqualStrings("", result.array[3].string);
    try std.testing.expectEqualStrings("c", result.array[4].string);
}

test "filterCapitalize - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterCapitalize(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterCapitalize - single character" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "a" };
    const result = try filterCapitalize(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("A", result.string);
}

test "filterTitle - multiple words" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "the quick brown fox" };
    const result = try filterTitle(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("The Quick Brown Fox", result.string);
}

test "filterTitle - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterTitle(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterEscape - quotes" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "She said \"hello\"" };
    const result = try filterEscape(&eval_ctx,input, &.{});
    try std.testing.expect(std.mem.indexOf(u8, result.string, "&#34;") != null or std.mem.indexOf(u8, result.string, "&quot;") != null);
}

test "filterEscape - e alias" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<div>test</div>" };
    const result = try filterEscape(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("&lt;div&gt;test&lt;/div&gt;", result.string);
}

test "filterStrip - comprehensive" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "  \n\r\t  hello world  \t\r\n  " };
    const result = try filterTrim(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello world", result.string);
}

test "filterWordcount - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterWordcount(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 0), result.integer);
}

test "filterWordcount - single word" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello" };
    const result = try filterWordcount(&eval_ctx,input, &.{});
    try std.testing.expectEqual(@as(i64, 1), result.integer);
}

test "filterStriptags - basic HTML" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<p>Hello <b>world</b>!</p>" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Hello world!", result.string);
}

test "filterStriptags - nested tags" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<div><span><a href='test'>Link</a></span></div>" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Link", result.string);
}

test "filterStriptags - no tags" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Plain text without tags" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Plain text without tags", result.string);
}

test "filterStriptags - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterStriptags - self-closing tags" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Line1<br/>Line2<hr/>Line3" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Line1Line2Line3", result.string);
}

test "filterStriptags - tags with attributes" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<div class='test' id='main'>Content</div>" };
    const result = try filterStriptags(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("Content", result.string);
}

test "filterTruncate - default" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const long_str = try std.testing.allocator.alloc(u8, 300);
    defer std.testing.allocator.free(long_str);
    @memset(long_str, 'a');

    const input = TemplateInput{ .string = long_str };
    const result = try filterTruncate(&eval_ctx,input, &.{});

    // Default is 255 chars, with "..." (3 chars), so 252 + 3 = 255
    try std.testing.expectEqual(@as(usize, 255), result.string.len);
    try std.testing.expect(std.mem.endsWith(u8, result.string, "..."));
}

test "filterTruncate - custom length" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "This is a long string that needs truncation" };
    const len_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&len_expr};
    const result = try filterTruncate(&eval_ctx,input, &args);

    // 10 chars total: 7 chars + "..." (3 chars)
    try std.testing.expectEqual(@as(usize, 10), result.string.len);
    try std.testing.expect(std.mem.endsWith(u8, result.string, "..."));
}

test "filterTruncate - custom suffix" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "This is a long string" };
    const len_expr = Expr{ .integer = 15 };
    const suffix_expr = Expr{ .string = " [more]" };
    const args = [_]*const Expr{ &len_expr, &suffix_expr };
    const result = try filterTruncate(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 15), result.string.len);
    try std.testing.expect(std.mem.endsWith(u8, result.string, " [more]"));
}

test "filterTruncate - no truncation needed" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Short" };
    const len_expr = Expr{ .integer = 100 };
    const args = [_]*const Expr{&len_expr};
    const result = try filterTruncate(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("Short", result.string);
}

test "filterTruncate - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const len_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&len_expr};
    const result = try filterTruncate(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("", result.string);
}

test "filterTruncate - exact length" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "12345" };
    const len_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&len_expr};
    const result = try filterTruncate(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("12345", result.string);
}

test "filterTruncate - UTF-8 CJK" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // "你好世界再见" is 6 codepoints (18 bytes)
    // truncate(5) with no suffix should give first 5 codepoints
    const input = TemplateInput{ .string = "你好世界再见" };
    const len_expr = Expr{ .integer = 5 };
    const suffix_expr = Expr{ .string = "" };
    const args = [_]*const Expr{ &len_expr, &suffix_expr };
    const result = try filterTruncate(&eval_ctx, input, &args);

    try std.testing.expectEqualStrings("你好世界再", result.string);
}

test "filterTruncate - UTF-8 with suffix" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // "你好世界" is 4 codepoints, truncate(5) with "..." (3 codepoints) = 2 chars + ...
    const input = TemplateInput{ .string = "你好世界再见" };
    const len_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&len_expr};
    const result = try filterTruncate(&eval_ctx, input, &args);

    try std.testing.expectEqualStrings("你好...", result.string);
}

test "filterUrlencode - basic string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello world" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello+world", result.string);
}

test "filterUrlencode - special characters" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "hello@world.com" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("hello%40world.com", result.string);
}

test "filterUrlencode - unreserved characters" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "abc-123_test.file~name" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("abc-123_test.file~name", result.string);
}

test "filterUrlencode - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("", result.string);
}

test "filterUrlencode - symbols" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "a+b=c&d" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("a%2Bb%3Dc%26d", result.string);
}

test "filterUrlencode - mixed content" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "user name: test@example.com" };
    const result = try filterUrlencode(&eval_ctx,input, &.{});
    try std.testing.expectEqualStrings("user+name%3A+test%40example.com", result.string);
}

test "filterWordwrap - default width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    var long_str = try std.testing.allocator.alloc(u8, 100);
    defer std.testing.allocator.free(long_str);
    @memset(long_str, 'a');
    long_str[20] = ' ';
    long_str[40] = ' ';
    long_str[60] = ' ';
    long_str[80] = ' ';

    const input = TemplateInput{ .string = long_str };
    const result = try filterWordwrap(&eval_ctx,input, &.{});

    // Should contain newlines due to wrapping
    try std.testing.expect(std.mem.indexOf(u8, result.string, "\n") != null);
}

test "filterWordwrap - custom width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "This is a test string that should be wrapped" };
    const width_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterWordwrap(&eval_ctx,input, &args);

    // Should have wrapped at 10 chars
    var lines = std.mem.splitScalar(u8, result.string, '\n');
    var count: usize = 0;
    while (lines.next()) |_| {
        count += 1;
    }
    try std.testing.expect(count > 1);
}

test "filterWordwrap - no wrap needed" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Short text" };
    const width_expr = Expr{ .integer = 79 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterWordwrap(&eval_ctx,input, &args);

    // Should not contain newlines
    try std.testing.expect(std.mem.indexOf(u8, result.string, "\n") == null);
    try std.testing.expectEqualStrings("Short text", result.string);
}

test "filterWordwrap - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const width_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterWordwrap(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("", result.string);
}

test "filterWordwrap - single word" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "Hello" };
    const width_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterWordwrap(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("Hello", result.string);
}

test "filterWordwrap - word boundaries" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "one two three four five" };
    const width_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterWordwrap(&eval_ctx,input, &args);

    // Verify no word is split
    try std.testing.expect(std.mem.indexOf(u8, result.string, "tw\no") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.string, "thre\ne") == null);
}

test "filterCenter - basic centering" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "test" };
    const width_expr = Expr{ .integer = 10 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterCenter(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 10), result.string.len);
    try std.testing.expectEqualStrings("   test   ", result.string);
}

test "filterCenter - odd padding" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "test" };
    const width_expr = Expr{ .integer = 9 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterCenter(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 9), result.string.len);
    // Odd padding: left gets less, right gets more (4 + 2 + 3 = 9)
    try std.testing.expectEqualStrings("  test   ", result.string);
}

test "filterCenter - exact width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "test" };
    const width_expr = Expr{ .integer = 4 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterCenter(&eval_ctx,input, &args);

    try std.testing.expectEqualStrings("test", result.string);
}

test "filterCenter - string too long" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "verylongstring" };
    const width_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterCenter(&eval_ctx,input, &args);

    // Should return original string when longer than width
    try std.testing.expectEqualStrings("verylongstring", result.string);
}

test "filterCenter - empty string" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "" };
    const width_expr = Expr{ .integer = 5 };
    const args = [_]*const Expr{&width_expr};
    const result = try filterCenter(&eval_ctx,input, &args);

    try std.testing.expectEqual(@as(usize, 5), result.string.len);
    try std.testing.expectEqualStrings("     ", result.string);
}

test "filterCenter - default width" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "x" };
    const result = try filterCenter(&eval_ctx,input, &.{});

    // Default width is 80
    try std.testing.expectEqual(@as(usize, 80), result.string.len);
}

test "filterCenter - UTF-8 CJK" {
    // CJK string "你好" (2 codepoints, 6 bytes) should center based on codepoints
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    // Width 6 with 2-codepoint string = 2 spaces each side
    const width_expr = Expr{ .integer = 6 };
    const args = [_]*const Expr{&width_expr};
    const input = TemplateInput{ .string = "你好" };
    const result = try filterCenter(&eval_ctx, input, &args);

    // Result: "  你好  " - 2 spaces + 6 bytes CJK + 2 spaces = 10 bytes
    try std.testing.expectEqualStrings("  你好  ", result.string);
}

test "filterSafe - passthrough" {
    var ctx = TemplateParser.init(std.testing.allocator);
    defer ctx.deinit();
    var eval_ctx = Evaluator.init(std.testing.allocator, &ctx);
    defer eval_ctx.deinit();

    const input = TemplateInput{ .string = "<script>alert('xss')</script>" };
    const result = try filterSafe(&eval_ctx, input, &.{});
    // safe() just passes through the value unchanged
    try std.testing.expectEqualStrings("<script>alert('xss')</script>", result.string);
}
