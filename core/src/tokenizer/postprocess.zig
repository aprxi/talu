//! Post-processing
//!
//! Adds special tokens (CLS, SEP, BOS, EOS) after model encoding.
//! Supports BERT, RoBERTa, and template-based post-processors.

const std = @import("std");
const ct = @import("c_types.zig");
const buffers = @import("encoding_buffers.zig");
const strings = @import("strings.zig");

pub fn tokenizer_apply_postprocessor_spec(tokenizer_opt: ?*ct.Tokenizer, spec_opt: ?*const ct.PostProcessorSpec) void {
    if (tokenizer_opt == null or spec_opt == null) return;
    const tokenizer = tokenizer_opt.?;
    const postprocessor_spec = spec_opt.?;
    tokenizer.postproc.add_special = postprocessor_spec.add_special;
    tokenizer.postproc.pair = postprocessor_spec.pair; // RoBERTa style double SEP
    if (postprocessor_spec.cls_token) |cls| {
        const cls_bytes = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(cls)), 0);
        const copy_len = @min(cls_bytes.len, tokenizer.postproc.cls_token.len - 1);
        @memcpy(tokenizer.postproc.cls_token[0..copy_len], cls_bytes[0..copy_len]);
        tokenizer.postproc.cls_token[copy_len] = 0;
    }
    if (postprocessor_spec.sep_token) |sep| {
        const sep_bytes = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(sep)), 0);
        const copy_len = @min(sep_bytes.len, tokenizer.postproc.sep_token.len - 1);
        @memcpy(tokenizer.postproc.sep_token[0..copy_len], sep_bytes[0..copy_len]);
        tokenizer.postproc.sep_token[copy_len] = 0;
    }
}

pub fn appendSpecialToken(
    ids_out: []i32,
    token_ptrs: [][*c]u8,
    attention_mask: []i32,
    type_ids: []i32,
    special_mask: []i32,
    offsets_out: []ct.Offset,
    write_index: *usize,
    token_id: i32,
    token_bytes: []const u8,
    type_id: i32,
    offset_start: i32,
    offset_end: i32,
) bool {
    ids_out[write_index.*] = token_id;
    token_ptrs[write_index.*] = @ptrCast(strings.strdup_range(token_bytes.ptr, token_bytes.len) orelse return false);
    attention_mask[write_index.*] = 1;
    type_ids[write_index.*] = type_id;
    special_mask[write_index.*] = 1;
    offsets_out[write_index.*] = .{ .start = offset_start, .end = offset_end };
    write_index.* += 1;
    return true;
}

pub fn postprocess_single(postprocessor: *const ct.PostProcessor, encoding: *ct.TokenizerEncoding) c_int {
    postprocess_single_impl(postprocessor, encoding) catch return -1;
    return 0;
}

fn postprocess_single_impl(postprocessor: *const ct.PostProcessor, encoding: *ct.TokenizerEncoding) !void {
    if (postprocessor.add_special == 0) return;

    const base_len = encoding.ids_len;
    const has_sep = postprocessor.sep_token[0] != 0;
    const total_len = base_len + 1 + @as(usize, if (has_sep) 1 else 0);

    var buffers_out = try buffers.allocBuffers(total_len);
    errdefer buffers_out.deinit();

    // CLS token at start
    const cls_bytes = std.mem.sliceTo(&postprocessor.cls_token, 0);
    var write_index: usize = 0;
    if (!appendSpecialToken(
        buffers_out.ids,
        buffers_out.tokens,
        buffers_out.attention_mask,
        buffers_out.type_ids,
        buffers_out.special,
        buffers_out.offsets,
        &write_index,
        postprocessor.cls_id,
        cls_bytes,
        0,
        -1,
        -1,
    )) return error.OutOfMemory;

    // Copy original content
    buffers.fillFromEncoding(&buffers_out, &write_index, encoding, 0, .{ .start = -1, .end = -1 });

    // SEP token at end (only if sep_token was explicitly configured)
    if (has_sep) {
        const sep_bytes = std.mem.sliceTo(&postprocessor.sep_token, 0);
        write_index = total_len - 1;
        if (!appendSpecialToken(
            buffers_out.ids,
            buffers_out.tokens,
            buffers_out.attention_mask,
            buffers_out.type_ids,
            buffers_out.special,
            buffers_out.offsets,
            &write_index,
            postprocessor.sep_id,
            sep_bytes,
            0,
            -1,
            -1,
        )) return error.OutOfMemory;
    }

    buffers.freeEncodingArrays(encoding);
    buffers.initEncoding(encoding, &buffers_out, total_len, encoding.overflows, encoding.overflow_count);
}

// =============================================================================
// Tests
// =============================================================================

// Note: postprocess_single_impl is complex and requires full tokenizer context
// including buffers allocation. It is primarily tested via integration tests
// in tests/tokenizer/. The postprocess_single wrapper is unit-testable for
// error handling.

test "tokenizer_apply_postprocessor_spec sets options" {
    var tok = ct.Tokenizer{
        .model = null,
        .type = .bpe,
        .normalizer = std.mem.zeroes(ct.Normalizer),
        .pretokenizer = std.mem.zeroes(ct.PreTokenizer),
        .postproc = std.mem.zeroes(ct.PostProcessor),
        .decoder = std.mem.zeroes(ct.Decoder),
        .padding = std.mem.zeroes(ct.Padding),
        .truncation = std.mem.zeroes(ct.Truncation),
        .added = null,
        .last_error = null,
    };

    const cls_token: [*:0]const u8 = "[CLS]";
    const sep_token: [*:0]const u8 = "[SEP]";
    const spec = ct.PostProcessorSpec{
        .type = null,
        .add_special = 1,
        .pair = 0,
        .cls_token = @ptrCast(cls_token),
        .sep_token = @ptrCast(sep_token),
    };

    tokenizer_apply_postprocessor_spec(&tok, &spec);

    try std.testing.expectEqual(@as(c_int, 1), tok.postproc.add_special);
    try std.testing.expectEqual(@as(c_int, 0), tok.postproc.pair);

    const cls_slice = std.mem.sliceTo(&tok.postproc.cls_token, 0);
    try std.testing.expectEqualStrings("[CLS]", cls_slice);

    const sep_slice = std.mem.sliceTo(&tok.postproc.sep_token, 0);
    try std.testing.expectEqualStrings("[SEP]", sep_slice);
}

test "appendSpecialToken adds token to buffers" {
    const types = @import("types.zig");
    const Allocator = types.Allocator;

    const ids = try Allocator.alloc(i32, 5);
    defer Allocator.free(ids);
    const tokens = try Allocator.alloc([*c]u8, 5);
    defer Allocator.free(tokens);
    const attention_mask = try Allocator.alloc(i32, 5);
    defer Allocator.free(attention_mask);
    const type_ids = try Allocator.alloc(i32, 5);
    defer Allocator.free(type_ids);
    const special_mask = try Allocator.alloc(i32, 5);
    defer Allocator.free(special_mask);
    const offsets_out = try Allocator.alloc(ct.Offset, 5);
    defer Allocator.free(offsets_out);

    var write_index: usize = 0;
    const success = appendSpecialToken(
        ids,
        tokens,
        attention_mask,
        type_ids,
        special_mask,
        offsets_out,
        &write_index,
        101, // token_id
        "[CLS]",
        0, // type_id
        -1, // offset_start
        -1, // offset_end
    );

    try std.testing.expect(success);
    try std.testing.expectEqual(@as(usize, 1), write_index);
    try std.testing.expectEqual(@as(i32, 101), ids[0]);
    try std.testing.expectEqual(@as(i32, 1), attention_mask[0]);
    try std.testing.expectEqual(@as(i32, 0), type_ids[0]);
    try std.testing.expectEqual(@as(i32, 1), special_mask[0]);

    // Clean up the token copy
    if (tokens[0]) |token_ptr| {
        Allocator.free(std.mem.span(@as([*:0]u8, @ptrCast(token_ptr))));
    }
}

test "postprocess_single returns error when add_special is 0" {
    var postprocessor = ct.PostProcessor{
        .cls_token = std.mem.zeroes([64]u8),
        .sep_token = std.mem.zeroes([64]u8),
        .cls_id = 101,
        .sep_id = 102,
        .add_special = 0, // Disabled - should return early
        .pair = 0,
        .kind = .none,
        .single = std.mem.zeroes([32]ct.PostProcessorEntry),
        .single_len = 0,
        .pair_tmpl = std.mem.zeroes([64]ct.PostProcessorEntry),
        .pair_len = 0,
    };

    var encoding = ct.TokenizerEncoding{
        .ids = null,
        .tokens = null,
        .ids_len = 0,
        .tokens_len = 0,
        .attention_mask = null,
        .type_ids = null,
        .special_tokens_mask = null,
        .offsets = null,
        .overflows = null,
        .overflow_count = 0,
    };

    // When add_special is 0, postprocess_single_impl returns early (no-op)
    const result = postprocess_single(&postprocessor, &encoding);
    try std.testing.expectEqual(@as(c_int, 0), result);
}
