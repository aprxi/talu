//! Tokenizer C API
//!
//! C-callable functions for tokenization (lightweight, no model weights).
//! This is the core text<->token conversion facility used by both standalone
//! tokenizer handles and session-bound operations.

const std = @import("std");
const tokenizer_mod = @import("../tokenizer/root.zig");
pub const TokenizerHandle = tokenizer_mod.TokenizerHandle;
const tok_pipeline = tokenizer_mod.pipeline;
const tok_offsets = tokenizer_mod.offsets;
const tok_batch = tokenizer_mod.batch;
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const SignalGuard = @import("signal_guard.zig").SignalGuard;

const allocator = std.heap.c_allocator;
const MAX_CAPI_BATCH_TEXTS: usize = 1_000_000;
const MAX_CAPI_TEXT_LEN: usize = 1 << 30; // 1 GiB safety ceiling for hostile FFI lengths

var empty_byte_storage: [1]u8 = .{0};
var empty_u32_storage: [1]u32 = .{0};
var empty_usize_storage: [1]usize = .{0};
var empty_offset_storage: [1]TokenOffset = .{.{ .start = 0, .end = 0 }};
var empty_token_string_storage: [1][*:0]u8 = .{@constCast("")};

fn emptyBytePtr() [*c]u8 {
    return empty_byte_storage[0..].ptr;
}

fn emptyU32Ptr() [*c]u32 {
    return empty_u32_storage[0..].ptr;
}

fn emptyOffsetPtr() [*c]TokenOffset {
    return empty_offset_storage[0..].ptr;
}

fn emptyUsizePtr() [*c]usize {
    return empty_usize_storage[0..].ptr;
}

fn emptyTokenStringPtr() [*c][*:0]u8 {
    return empty_token_string_storage[0..].ptr;
}

/// Allocates a NUL-terminated copy of a byte slice.
/// Returns null on allocation failure (OOM).
fn allocZFromSlice(bytes: []const u8) ?[*:0]u8 {
    // allocSentinel allocates len+1 bytes and sets the sentinel (0) at the end
    const buf = allocator.allocSentinel(u8, bytes.len, 0) catch return null;
    @memcpy(buf, bytes);
    return buf.ptr;
}

// =============================================================================
// Types
// =============================================================================

/// Opaque handle for C API. Points to TokenizerHandle from tokenizer module.
///
/// Thread Safety: The tokenizer is read-only after initialization and is
/// thread-safe for concurrent encode/decode operations. Multiple threads
/// can safely call talu_encode/talu_decode on the same handle. However,
/// talu_tokenizer_create and talu_tokenizer_destroy must not be called
/// concurrently with any other operation on the same handle.
pub const OpaqueTokenizerHandle = opaque {};

pub const EncodeResult = extern struct {
    ids: [*c]u32,
    offsets: [*c]TokenOffset,
    attention_mask: [*c]u32,
    special_tokens_mask: [*c]u32,
    num_tokens: usize,
    error_msg: [*c]const u8,
};

pub const DecodeResult = extern struct {
    text: [*c]u8,
    text_len: usize,
    error_msg: [*c]const u8,
};

pub const EncodeOptions = extern struct {
    add_bos: u8 = 1,
    add_eos: u8 = 1,
    truncation: u8 = 0,
    truncation_side: u8 = 0,
    _padding: [4]u8 = .{ 0, 0, 0, 0 },
    max_length: usize = 0,
};

pub const TokenizeResult = extern struct {
    tokens: [*c][*:0]u8,
    num_tokens: usize,
    error_msg: [*c]const u8,
};

pub const TokenizeBytesResult = extern struct {
    data: [*c]u8,
    data_len: usize,
    offsets: [*c]usize,
    num_tokens: usize,
    error_msg: [*c]const u8,
};

pub const TokenOffset = extern struct {
    start: u32,
    end: u32,
};

pub const EosTokenResult = extern struct {
    tokens: [*c]u32,
    num_tokens: usize,
};

pub const BatchEncodeResult = extern struct {
    ids: [*c]u32,
    offsets: [*c]usize,
    total_tokens: usize,
    num_sequences: usize,
    error_msg: [*c]const u8,
};

pub const SpecialTokensResult = extern struct {
    bos_token_id: i32,
    unk_token_id: i32,
    pad_token_id: i32,
};

pub const DecodeOptionsC = extern struct {
    skip_special_tokens: c_int = 1,
};

pub const VocabResult = extern struct {
    tokens: [*c][*:0]u8,
    lengths: [*c]u32,
    num_entries: usize,
    ids: [*c]u32,
    error_msg: [*c]const u8,
};

pub const PaddedTensorOptions = extern struct {
    pad_id: u32 = 0,
    padding_side: u8 = 0,
    max_length: usize = 0,
    truncate: bool = false,
    return_attention_mask: bool = true,
};

pub const PaddedTensorResult = extern struct {
    input_ids: [*c]u32,
    attention_mask: [*c]u32,
    num_sequences: usize,
    padded_length: usize,
    error_msg: [*c]const u8,
};

// =============================================================================
// Tokenizer Creation/Destruction
// =============================================================================

/// Context for signal-guarded tokenizer creation.
const TokenizerCreateContext = struct {
    model_path: []const u8,
    result_handle: ?*TokenizerHandle = null,
    result_error: ?anyerror = null,
};

/// Callback for signal-guarded tokenizer creation.
fn tokenizerCreateCallback(ctx_ptr: *anyopaque) callconv(.c) c_int {
    const ctx: *TokenizerCreateContext = @ptrCast(@alignCast(ctx_ptr));
    ctx.result_handle = TokenizerHandle.init(allocator, ctx.model_path) catch |err| {
        ctx.result_error = err;
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Creates a tokenizer from a model path.
///
/// The path can be a local directory or a HuggingFace model ID (e.g., "org/model-name").
/// Caller must call talu_tokenizer_free() when done.
///
/// This function is protected against fatal signals (SIGBUS) that can occur during
/// resource exhaustion. If a signal is caught, returns error code 905 (resource_exhausted).
pub export fn talu_tokenizer_create(
    model_path: ?[*:0]const u8,
    out_tokenizer: ?*?*OpaqueTokenizerHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const model_path_ptr = model_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out = out_tokenizer orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tokenizer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    // Install signal guard to catch SIGBUS/SIGSEGV during tokenizer loading.
    // This can happen under heavy parallel load due to resource exhaustion.
    const guard = SignalGuard.init();
    defer guard.deinit();

    var ctx = TokenizerCreateContext{
        .model_path = std.mem.span(model_path_ptr),
    };

    const result = guard.call(tokenizerCreateCallback, @ptrCast(&ctx));
    if (result) |code| {
        if (code == 0) {
            // Success
            out.* = @ptrCast(ctx.result_handle);
            return 0;
        } else {
            // Tokenizer init failed with an error
            if (ctx.result_error) |err| {
                capi_error.setError(err, "Failed to create tokenizer", .{});
            }
            return code;
        }
    } else {
        // Signal was caught (SIGBUS/SIGSEGV) - resource exhaustion
        capi_error.setError(error.ResourceExhausted, "Resource exhaustion: fatal signal caught during tokenizer creation. " ++
            "This typically occurs under heavy parallel load. Try reducing concurrency.", .{});
        return @intFromEnum(error_codes.errorToCode(error.ResourceExhausted));
    }
}

/// Context for signal-guarded tokenizer creation from JSON.
const TokenizerCreateJsonContext = struct {
    json_ptr: [*]const u8,
    json_len: usize,
    result_handle: ?*TokenizerHandle = null,
    result_error: ?anyerror = null,
};

/// Callback for signal-guarded tokenizer creation from JSON.
fn tokenizerCreateJsonCallback(ctx_ptr: *anyopaque) callconv(.c) c_int {
    const ctx: *TokenizerCreateJsonContext = @ptrCast(@alignCast(ctx_ptr));
    ctx.result_handle = TokenizerHandle.initFromJson(allocator, ctx.json_ptr[0..ctx.json_len]) catch |err| {
        ctx.result_error = err;
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Creates a tokenizer from JSON content.
///
/// This creates a minimal tokenizer without model directory or generation config.
/// Useful for testing or standalone tokenization without a full model.
/// Caller must call talu_tokenizer_free() when done.
///
/// This function is protected against fatal signals (SIGBUS) that can occur during
/// resource exhaustion. If a signal is caught, returns error code 905 (resource_exhausted).
pub export fn talu_tokenizer_create_from_json(
    json_ptr: [*]const u8,
    json_len: usize,
    out_tokenizer: ?*?*OpaqueTokenizerHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tokenizer orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tokenizer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    if (json_len == 0) {
        capi_error.setError(error.InvalidArgument, "Empty JSON content", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    }

    // Install signal guard to catch SIGBUS/SIGSEGV during tokenizer loading.
    const guard = SignalGuard.init();
    defer guard.deinit();

    var ctx = TokenizerCreateJsonContext{
        .json_ptr = json_ptr,
        .json_len = json_len,
    };

    const code = guard.call(tokenizerCreateJsonCallback, @ptrCast(&ctx)) orelse {
        // Signal was caught (SIGBUS/SIGSEGV) - resource exhaustion
        capi_error.setError(error.ResourceExhausted, "fatal signal during tokenizer creation (try reducing concurrency)", .{});
        return @intFromEnum(error_codes.errorToCode(error.ResourceExhausted));
    };
    if (code == 0) {
        out.* = @ptrCast(ctx.result_handle);
        return 0;
    }
    if (ctx.result_error) |err| capi_error.setError(err, "Failed to create tokenizer from JSON", .{});
    return code;
}

/// Frees a tokenizer created by talu_tokenizer_create() or talu_tokenizer_create_from_json().
///
/// Safe to call with null (no-op).
pub export fn talu_tokenizer_free(handle: ?*OpaqueTokenizerHandle) callconv(.c) void {
    // Cast opaque handle back to TokenizerHandle; no-op if null
    const tokenizer_handle: *TokenizerHandle = @ptrCast(@alignCast(handle orelse return));
    tokenizer_handle.deinit();
}

// =============================================================================
// Encoding
// =============================================================================

/// Truncation parameters for encoding.
const TruncationParams = struct { start: usize, count: usize };

/// Compute truncation start and count based on options.
fn computeTruncation(ids_len: usize, options: ?*const EncodeOptions) TruncationParams {
    var start: usize = 0;
    var count = ids_len;
    if (options) |o| {
        if (o.truncation != 0 and o.max_length > 0 and ids_len > o.max_length) {
            count = o.max_length;
            if (o.truncation_side != 0) start = ids_len - o.max_length;
        }
    }
    return .{ .start = start, .count = count };
}

fn validateEncodeOptions(options: ?*const EncodeOptions) ?[*:0]const u8 {
    const o = options orelse return null;
    if (o.truncation != 0 and o.truncation_side > 1) {
        return "Invalid truncation_side";
    }
    return null;
}

/// Encodes text to token IDs, byte offsets, attention mask, and special tokens mask.
///
/// Runs the full encoding pipeline once and computes all metadata simultaneously.
/// Options control BOS/EOS handling and truncation.
/// Caller must free via talu_encode_result_free().
pub export fn talu_tokenizer_encode(
    handle: ?*OpaqueTokenizerHandle,
    text: [*c]const u8,
    text_len: usize,
    options: ?*const EncodeOptions,
) callconv(.c) EncodeResult {
    capi_error.clearError();
    if (validateEncodeOptions(options)) |msg| {
        capi_error.setError(error.InvalidArgument, "{s}", .{msg});
        var err = std.mem.zeroes(EncodeResult);
        err.error_msg = msg;
        return err;
    }
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        var err = std.mem.zeroes(EncodeResult);
        err.error_msg = "Invalid handle";
        return err;
    }));

    const text_slice = if (text_len == 0)
        ""
    else blk: {
        const text_ptr = text orelse {
            capi_error.setError(error.InvalidArgument, "text is null with non-zero length", .{});
            var err = std.mem.zeroes(EncodeResult);
            err.error_msg = "InvalidArgument";
            return err;
        };
        break :blk text_ptr[0..text_len];
    };

    const add_bos = if (options) |o| o.add_bos != 0 else true;
    const add_eos = if (options) |o| o.add_eos != 0 else true;
    const add_special = add_bos == add_eos and add_bos;
    var rich = tok_offsets.encode(allocator, tok.tok.tokenizer_handle, text_slice, add_special) catch |e| {
        capi_error.setError(e, "Encode failed", .{});
        var err = std.mem.zeroes(EncodeResult);
        err.error_msg = "Encode failed";
        return err;
    };
    if (add_bos != add_eos) {
        rich.applySelectiveSpecialTokens(tok.tok.tokenizer_handle, add_bos, add_eos) catch {
            rich.deinit();
            capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
            var err = std.mem.zeroes(EncodeResult);
            err.error_msg = "OutOfMemory";
            return err;
        };
    }

    const params = computeTruncation(rich.ids.len, options);
    rich.truncate(params.start, params.count) catch {
        rich.deinit();
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        var err = std.mem.zeroes(EncodeResult);
        err.error_msg = "OutOfMemory";
        return err;
    };

    var ret = std.mem.zeroes(EncodeResult);
    ret.ids = if (rich.ids.len > 0) rich.ids.ptr else emptyU32Ptr();
    ret.offsets = if (rich.offsets.len > 0) @ptrCast(rich.offsets.ptr) else emptyOffsetPtr();
    ret.attention_mask = if (rich.attention_mask.len > 0) rich.attention_mask.ptr else emptyU32Ptr();
    ret.special_tokens_mask = if (rich.special_tokens_mask.len > 0) rich.special_tokens_mask.ptr else emptyU32Ptr();
    ret.num_tokens = rich.ids.len;
    return ret;
}

/// Frees all buffers in an EncodeResult.
/// Safe to call with null fields (no-op for each null pointer).
pub export fn talu_encode_result_free(result: EncodeResult) callconv(.c) void {
    const n = result.num_tokens;
    if (n == 0) return;
    if (result.ids) |p| allocator.free(p[0..n]);
    if (result.offsets) |p| allocator.free(p[0..n]);
    if (result.attention_mask) |p| allocator.free(p[0..n]);
    if (result.special_tokens_mask) |p| allocator.free(p[0..n]);
}

// =============================================================================
// Tokenization (String Tokens)
// =============================================================================

/// Get token count from tokenizer (first pass).
fn getTokenCount(tok_handle: anytype, text: []const u8) ?usize {
    var count: usize = 0;
    if (tok_pipeline.tokenizer_tokenize(tok_handle, text, null, &count) != 0) return null;
    return count;
}

/// Fill token strings into pre-allocated buffer.
fn fillTokenStrings(tok_handle: anytype, text: []const u8, out: [][*:0]u8) ?usize {
    var actual = out.len;
    if (tok_pipeline.tokenizer_tokenize(tok_handle, text, out.ptr, &actual) != 0) return null;
    return actual;
}

/// Tokenizes text and returns string representations of each token.
///
/// Unlike encode, this returns the string form of each token, useful for
/// debugging or visualization. Caller must free via talu_tokenize_result_free().
pub export fn talu_tokenizer_tokenize(
    handle: ?*OpaqueTokenizerHandle,
    text: [*c]const u8,
    text_len: usize,
) callconv(.c) TokenizeResult {
    capi_error.clearError();
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        return .{ .tokens = null, .num_tokens = 0, .error_msg = "Invalid handle" };
    }));

    const text_slice = if (text_len == 0)
        ""
    else blk: {
        const text_ptr = text orelse {
            capi_error.setError(error.InvalidArgument, "text is null with non-zero length", .{});
            return .{ .tokens = null, .num_tokens = 0, .error_msg = "InvalidArgument" };
        };
        break :blk text_ptr[0..text_len];
    };

    const count = getTokenCount(tok.tok.tokenizer_handle, text_slice) orelse {
        capi_error.setError(error.InternalError, "Tokenization failed", .{});
        return .{ .tokens = null, .num_tokens = 0, .error_msg = "Tokenization failed" };
    };

    const out = allocator.alloc([*:0]u8, count) catch {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        return .{ .tokens = null, .num_tokens = 0, .error_msg = "OutOfMemory" };
    };

    if (count == 0) {
        allocator.free(out);
        return .{ .tokens = emptyTokenStringPtr(), .num_tokens = 0, .error_msg = null };
    }

    const actual = fillTokenStrings(tok.tok.tokenizer_handle, text_slice, out) orelse {
        capi_error.setError(error.InternalError, "Tokenization failed", .{});
        allocator.free(out);
        return .{ .tokens = null, .num_tokens = 0, .error_msg = "Tokenization failed" };
    };

    return .{ .tokens = out.ptr, .num_tokens = actual, .error_msg = null };
}

/// Frees tokenization result returned by talu_tokenizer_tokenize().
///
/// Safe to call with null (no-op).
pub export fn talu_tokenize_result_free(tokens: ?[*][*:0]u8, num_tokens: usize) callconv(.c) void {
    const t = tokens orelse return;
    if (num_tokens == 0) return;
    // Free each token string, then free the array of pointers
    for (0..num_tokens) |i| tok_pipeline.tokenizer_string_free(t[i]);
    allocator.free(t[0..num_tokens]);
}

/// Tokenizes text and returns raw bytes for each token.
///
/// Returns a contiguous byte buffer with offset indices.
/// Useful for byte-level operations. Caller must free via talu_tokenize_bytes_result_free().
pub export fn talu_tokenizer_tokenize_bytes(
    handle: ?*OpaqueTokenizerHandle,
    text: [*c]const u8,
    text_len: usize,
) callconv(.c) TokenizeBytesResult {
    capi_error.clearError();
    // Cast opaque handle to TokenizerHandle
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        var err_result = std.mem.zeroes(TokenizeBytesResult);
        err_result.error_msg = "Invalid handle";
        return err_result;
    }));

    const text_slice = if (text_len == 0)
        ""
    else blk: {
        const text_ptr = text orelse {
            capi_error.setError(error.InvalidArgument, "text is null with non-zero length", .{});
            var err_result = std.mem.zeroes(TokenizeBytesResult);
            err_result.error_msg = "InvalidArgument";
            return err_result;
        };
        break :blk text_ptr[0..text_len];
    };

    var result = tokenizer_mod.tokenizeToBytes(allocator, tok.tok.tokenizer_handle, text_slice) catch |e| {
        capi_error.setError(e, "Tokenization failed", .{});
        var err_result = std.mem.zeroes(TokenizeBytesResult);
        err_result.error_msg = "Tokenization failed";
        return err_result;
    };

    // Transfer ownership to caller by extracting pointers and clearing result
    // This prevents result.deinit() from freeing the memory we're returning
    const data_ptr = if (result.data.len > 0) result.data.ptr else emptyBytePtr();
    const offsets_ptr = if (result.offsets.len > 0) result.offsets.ptr else emptyUsizePtr();
    const data_len = result.data.len;
    const num_tokens = result.tokenCount();

    // Clear slices so destructor won't free them
    result.data = &.{};
    result.offsets = &.{};

    var ret = std.mem.zeroes(TokenizeBytesResult);
    ret.data = data_ptr;
    ret.data_len = data_len;
    ret.offsets = offsets_ptr;
    ret.num_tokens = num_tokens;
    return ret;
}

/// Frees bytes result returned by talu_tokenizer_tokenize_bytes().
pub export fn talu_tokenize_bytes_result_free(
    data: ?[*]u8,
    data_len: usize,
    offsets: ?[*]usize,
    num_tokens: usize,
) callconv(.c) void {
    if (data) |d| if (data_len > 0) allocator.free(d[0..data_len]);
    if (offsets) |o| if (num_tokens > 0) allocator.free(o[0 .. num_tokens + 1]);
}

// =============================================================================
// Decoding
// =============================================================================

/// Decodes token IDs back to text.
///
/// Options control whether special tokens are included in output.
/// Pass null for options to use defaults.
/// Caller must free via talu_decode_result_free().
pub export fn talu_tokenizer_decode(
    handle: ?*OpaqueTokenizerHandle,
    tokens: [*c]const u32,
    num_tokens: usize,
    options: ?*const DecodeOptionsC,
) callconv(.c) DecodeResult {
    capi_error.clearError();
    // Cast opaque handle to TokenizerHandle
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        var err_result = std.mem.zeroes(DecodeResult);
        err_result.error_msg = "Invalid handle";
        return err_result;
    }));

    if (num_tokens == 0) return std.mem.zeroes(DecodeResult);
    const token_ptr = tokens orelse {
        capi_error.setError(error.InvalidArgument, "tokens is null with non-zero length", .{});
        var err_result = std.mem.zeroes(DecodeResult);
        err_result.error_msg = "InvalidArgument";
        return err_result;
    };

    const opts = tokenizer_mod.Tokenizer.DecodeOptions{
        .skip_special_tokens = if (options) |o| o.skip_special_tokens != 0 else true,
    };

    const decoded = tok.tok.decodeWithOptions(token_ptr[0..num_tokens], opts) catch |e| {
        const msg = switch (e) {
            error.InvalidTokenId => "Invalid token ID: out of range for vocabulary",
            else => "Decode failed",
        };
        capi_error.setError(e, "{s}", .{msg});
        var err_result = std.mem.zeroes(DecodeResult);
        err_result.error_msg = msg;
        return err_result;
    };
    defer tok.tok.allocator.free(decoded);

    const out = allocator.alloc(u8, decoded.len) catch {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        var err_result = std.mem.zeroes(DecodeResult);
        err_result.error_msg = "OutOfMemory";
        return err_result;
    };
    @memcpy(out, decoded);

    var ret = std.mem.zeroes(DecodeResult);
    ret.text = out.ptr;
    ret.text_len = decoded.len;
    return ret;
}

// =============================================================================
// Batch Encoding
// =============================================================================

/// Encodes multiple texts to token IDs in a single call.
///
/// Returns a flat token array with offsets indicating sequence boundaries.
/// Caller must free via talu_batch_encode_result_free().
pub export fn talu_tokenizer_encode_batch(
    handle: ?*OpaqueTokenizerHandle,
    texts: [*c]const [*c]const u8,
    lengths: [*c]const usize,
    num_texts: usize,
    options: ?*const EncodeOptions,
) callconv(.c) BatchEncodeResult {
    capi_error.clearError();
    // Cast opaque handle to TokenizerHandle
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = "Invalid handle";
        return err_result;
    }));

    if (num_texts == 0) return std.mem.zeroes(BatchEncodeResult);
    const text_ptrs = texts orelse {
        capi_error.setError(error.InvalidArgument, "texts is null with non-zero count", .{});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = "InvalidArgument";
        return err_result;
    };
    const length_ptrs = lengths orelse {
        capi_error.setError(error.InvalidArgument, "lengths is null with non-zero count", .{});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = "InvalidArgument";
        return err_result;
    };
    const batch_text_ptrs: [*]const [*]const u8 = @ptrCast(text_ptrs);
    const batch_length_ptrs: [*]const usize = @ptrCast(length_ptrs);
    if (num_texts > MAX_CAPI_BATCH_TEXTS) {
        capi_error.setError(error.InvalidArgument, "num_texts exceeds API safety limit", .{});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = "InvalidArgument";
        return err_result;
    }
    var total_declared_bytes: usize = 0;
    for (0..num_texts) |idx| {
        const ptr_i = batch_text_ptrs[idx];
        const len_i = batch_length_ptrs[idx];
        if (len_i > 0 and @intFromPtr(ptr_i) == 0) {
            capi_error.setError(error.InvalidArgument, "texts contains null pointer with non-zero length", .{});
            var err_result = std.mem.zeroes(BatchEncodeResult);
            err_result.error_msg = "InvalidArgument";
            return err_result;
        }
        if (len_i > MAX_CAPI_TEXT_LEN) {
            capi_error.setError(error.InvalidArgument, "text length exceeds API safety limit", .{});
            var err_result = std.mem.zeroes(BatchEncodeResult);
            err_result.error_msg = "InvalidArgument";
            return err_result;
        }
        const sum, const overflow = @addWithOverflow(total_declared_bytes, len_i);
        if (overflow != 0) {
            capi_error.setError(error.InvalidArgument, "batch lengths overflow", .{});
            var err_result = std.mem.zeroes(BatchEncodeResult);
            err_result.error_msg = "InvalidArgument";
            return err_result;
        }
        total_declared_bytes = sum;
    }
    if (validateEncodeOptions(options)) |msg| {
        capi_error.setError(error.InvalidArgument, "{s}", .{msg});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = msg;
        return err_result;
    }

    const batch_options = tok_batch.BatchEncodeOptions{
        .add_bos = if (options) |o| o.add_bos != 0 else true,
        .add_eos = if (options) |o| o.add_eos != 0 else true,
        .truncation = if (options) |o| o.truncation != 0 else false,
        .truncation_left = if (options) |o| o.truncation_side != 0 else false,
        .max_length = if (options) |o| o.max_length else 0,
    };

    var result = tok_batch.encodeBatch(allocator, &tok.tok, batch_text_ptrs, batch_length_ptrs, num_texts, batch_options) catch |e| {
        capi_error.setError(e, "Batch encoding failed", .{});
        var err_result = std.mem.zeroes(BatchEncodeResult);
        err_result.error_msg = "Batch encoding failed";
        return err_result;
    };

    // Transfer ownership to caller
    const ids_ptr = if (result.ids.len > 0) result.ids.ptr else null;
    const off_ptr = if (result.offsets.len > 0) result.offsets.ptr else null;
    const total = result.total_tokens;
    const num_seq = result.num_sequences;

    // Clear slices so destructor won't free them
    result.ids = &.{};
    result.offsets = &.{};

    var ret = std.mem.zeroes(BatchEncodeResult);
    ret.ids = ids_ptr;
    ret.offsets = off_ptr;
    ret.total_tokens = total;
    ret.num_sequences = num_seq;
    return ret;
}

/// Frees batch encode result returned by talu_tokenizer_encode_batch().
pub export fn talu_batch_encode_result_free(
    ids: ?[*]u32,
    offsets: ?[*]usize,
    total_tokens: usize,
    num_sequences: usize,
) callconv(.c) void {
    if (ids) |i| if (total_tokens > 0) allocator.free(i[0..total_tokens]);
    if (offsets) |o| allocator.free(o[0 .. num_sequences + 1]);
}

// =============================================================================
// Padded Tensor Conversion
// =============================================================================

/// Creates an error result for padded tensor operations.
fn paddedTensorError(msg: [*:0]const u8) PaddedTensorResult {
    var result = std.mem.zeroes(PaddedTensorResult);
    result.error_msg = msg;
    return result;
}

/// Converts C options to internal options format.
fn convertPaddedOptions(options: ?*const PaddedTensorOptions) tok_batch.PaddedTensorOptions {
    const o = options orelse return .{};
    var opt = std.mem.zeroes(tok_batch.PaddedTensorOptions);
    const raw_bytes: [*]const u8 = @ptrCast(o);
    const truncate_raw = raw_bytes[@offsetOf(PaddedTensorOptions, "truncate")];
    const mask_raw = raw_bytes[@offsetOf(PaddedTensorOptions, "return_attention_mask")];
    opt.pad_id = o.pad_id;
    opt.padding_side = if (o.padding_side == 1) .left else .right;
    opt.max_length = o.max_length;
    // FFI callers may pass non-canonical bool bytes; treat any nonzero byte as true.
    opt.truncate = truncate_raw != 0;
    opt.return_attention_mask = mask_raw != 0;
    return opt;
}

/// Converts batch encoding results to padded tensor format.
///
/// Pads all sequences to the same length for efficient batch processing.
/// The consumed ID prefix is defined by `offsets[num_sequences]`; callers must
/// ensure `ids` is valid for at least that many elements.
/// Caller must free via talu_padded_tensor_result_free().
pub export fn talu_batch_to_padded_tensor(
    ids: ?[*]const u32,
    offsets: ?[*]const usize,
    num_sequences: usize,
    options: ?*const PaddedTensorOptions,
) callconv(.c) PaddedTensorResult {
    if (num_sequences == 0) return std.mem.zeroes(PaddedTensorResult);
    const ids_ptr = ids orelse return paddedTensorError("Invalid ids");
    const off_ptr = offsets orelse return paddedTensorError("Invalid offsets");
    if (options) |o| {
        if (o.padding_side > 1) return paddedTensorError("Invalid padding_side");
    }
    if (off_ptr[0] != 0) return paddedTensorError("Invalid offsets");
    for (1..num_sequences + 1) |idx| {
        if (off_ptr[idx] < off_ptr[idx - 1]) return paddedTensorError("Invalid offsets");
    }

    var result = tok_batch.batchToPaddedTensor(
        allocator,
        ids_ptr[0..off_ptr[num_sequences]],
        off_ptr[0 .. num_sequences + 1],
        num_sequences,
        convertPaddedOptions(options),
    ) catch return paddedTensorError("OutOfMemory");

    // Transfer ownership to caller, clear slices so destructor won't free them
    var ret = std.mem.zeroes(PaddedTensorResult);
    ret.input_ids = result.input_ids.ptr;
    ret.attention_mask = if (result.attention_mask) |m| m.ptr else null;
    ret.num_sequences = result.num_sequences;
    ret.padded_length = result.padded_length;
    result.input_ids = &.{};
    result.attention_mask = null;
    return ret;
}

/// Frees padded tensor result returned by talu_batch_to_padded_tensor().
pub export fn talu_padded_tensor_result_free(
    input_ids: ?[*]u32,
    attention_mask: ?[*]u32,
    num_sequences: usize,
    padded_length: usize,
) callconv(.c) void {
    const total = num_sequences * padded_length;
    if (input_ids) |i| if (total > 0) allocator.free(i[0..total]);
    if (attention_mask) |m| if (total > 0) allocator.free(m[0..total]);
}

// =============================================================================
// Vocabulary Access
// =============================================================================

/// Gets the EOS (end-of-sequence) token IDs for this model.
///
/// Returns an array of token IDs that signal generation should stop.
/// Caller must free via talu_tokens_free().
pub export fn talu_tokenizer_get_eos_tokens(handle: ?*OpaqueTokenizerHandle) callconv(.c) EosTokenResult {
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse return std.mem.zeroes(EosTokenResult)));
    if (tok.gen_config.eos_token_ids.len == 0) return std.mem.zeroes(EosTokenResult);

    const ids = allocator.alloc(u32, tok.gen_config.eos_token_ids.len) catch return std.mem.zeroes(EosTokenResult);
    @memcpy(ids, tok.gen_config.eos_token_ids);
    var ret = std.mem.zeroes(EosTokenResult);
    ret.tokens = ids.ptr;
    ret.num_tokens = ids.len;
    return ret;
}

/// Gets the resolved model directory path.
///
/// Returns the filesystem path to the model's directory.
/// Caller must free the returned string via talu_text_free().
pub export fn talu_tokenizer_get_model_dir(handle: ?*OpaqueTokenizerHandle, out_path: ?*?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_path orelse {
        capi_error.setError(error.InvalidArgument, "out_path is null", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    };
    out.* = null;
    // Cast opaque handle to TokenizerHandle
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    }));
    out.* = allocZFromSlice(tok.model_dir) orelse {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    return 0;
}

/// Gets the vocabulary size (number of tokens in the tokenizer).
pub export fn talu_tokenizer_get_vocab_size(handle: ?*OpaqueTokenizerHandle) callconv(.c) usize {
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse return 0));
    return tok.tok.tokenizer_handle.getVocabSize();
}

/// Gets the full vocabulary as token strings, lengths, and IDs.
///
/// Caller must free via talu_vocab_result_free().
pub export fn talu_tokenizer_get_vocab(handle: ?*OpaqueTokenizerHandle) callconv(.c) VocabResult {
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        var err_result = std.mem.zeroes(VocabResult);
        err_result.error_msg = "Invalid handle";
        return err_result;
    }));

    var result = tokenizer_mod.getVocab(allocator, tok.tok.tokenizer_handle) catch {
        var err_result = std.mem.zeroes(VocabResult);
        err_result.error_msg = "Vocab extraction failed";
        return err_result;
    };
    defer result.deinit();

    const count = result.entries.len;
    if (count == 0) return std.mem.zeroes(VocabResult);

    // Use tokenizer module's VocabListC for conversion (entries have compatible layout)
    const vocab_entries: []const tokenizer_mod.VocabEntryC = @ptrCast(result.entries);
    const vocab_list = tokenizer_mod.VocabListC.fromEntries(allocator, vocab_entries) catch {
        var err_result = std.mem.zeroes(VocabResult);
        err_result.error_msg = "OutOfMemory";
        return err_result;
    };
    // Note: we transfer ownership to the caller, so don't deinit vocab_list

    var ret = std.mem.zeroes(VocabResult);
    ret.tokens = vocab_list.tokens.ptr;
    ret.lengths = vocab_list.lengths.ptr;
    ret.num_entries = count;
    ret.ids = vocab_list.ids.ptr;
    return ret;
}

/// Frees vocabulary result returned by talu_tokenizer_get_vocab().
pub export fn talu_vocab_result_free(
    tokens: ?[*][*:0]u8,
    lengths: ?[*]u32,
    ids: ?[*]u32,
    num_entries: usize,
) callconv(.c) void {
    if (num_entries == 0) return;
    const t = tokens orelse return;
    const l = lengths orelse return;
    const id = ids orelse return;

    // Reconstruct VocabListC and use its deinit
    var vocab_list = tokenizer_mod.VocabListC{
        .tokens = t[0..num_entries],
        .lengths = l[0..num_entries],
        .ids = id[0..num_entries],
    };
    vocab_list.deinit(allocator);
}

/// Gets special token IDs (BOS, UNK, PAD).
///
/// Returns -1 for any token that is not defined for this model.
pub export fn talu_tokenizer_get_special_tokens(handle: ?*OpaqueTokenizerHandle) callconv(.c) SpecialTokensResult {
    var ret = std.mem.zeroes(SpecialTokensResult);
    ret.bos_token_id = -1;
    ret.unk_token_id = -1;
    ret.pad_token_id = -1;

    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse return ret));

    const bos: i32 = if (tok.gen_config.bos_explicitly_disabled)
        -1
    else if (tok.resolved_bos_id) |id|
        id
    else if (tok.gen_config.bos_token_id) |id|
        @intCast(id)
    else
        tok.tok.tokenizer_handle.getBosId();

    const pad: i32 = if (tok.gen_config.pad_token_id) |id| @intCast(id) else tok.tok.tokenizer_handle.padding.pad_id;

    ret.bos_token_id = bos;
    ret.unk_token_id = tok.tok.tokenizer_handle.getUnkId();
    ret.pad_token_id = pad;
    return ret;
}

/// Get the model's maximum sequence length from tokenizer_config.json.
/// Returns 0 if not set (caller should handle this as "no limit specified").
pub export fn talu_tokenizer_get_model_max_length(handle: ?*OpaqueTokenizerHandle) callconv(.c) u64 {
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse return 0));
    return tok.gen_config.model_max_length orelse 0;
}

/// Converts a token ID to its string representation.
///
/// Caller must free the returned string via talu_text_free().
pub export fn talu_tokenizer_id_to_token(handle: ?*OpaqueTokenizerHandle, token_id: i32, out_token: ?*?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_token orelse {
        capi_error.setError(error.InvalidArgument, "out_token is null", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    };
    out.* = null;
    // Cast opaque handle to TokenizerHandle
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "Invalid handle", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    }));

    const text = tok.tok.tokenizer_handle.idToToken(token_id) orelse {
        capi_error.setError(error.InvalidArgument, "Token id out of range", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    };

    out.* = allocZFromSlice(text) orelse {
        capi_error.setError(error.OutOfMemory, "Allocation failed", .{});
        return @intFromEnum(error_codes.errorToCode(error.OutOfMemory));
    };
    return 0;
}

/// Converts a token string to its ID.
///
/// Returns -1 if the token is not in the vocabulary.
/// Returns error code (negative) if handle is null.
pub export fn talu_tokenizer_token_to_id(handle: ?*OpaqueTokenizerHandle, token: ?[*]const u8, token_len: usize) callconv(.c) i32 {
    capi_error.clearError();
    const tok: *TokenizerHandle = @ptrCast(@alignCast(handle orelse {
        capi_error.setError(error.InvalidHandle, "tokenizer handle is null", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidHandle));
    }));
    if (token_len > 0 and token == null) {
        capi_error.setError(error.InvalidArgument, "token pointer is null with non-zero length", .{});
        return @intFromEnum(error_codes.errorToCode(error.InvalidArgument));
    }
    const token_slice = if (token_len == 0) "" else token.?[0..token_len];

    // Non-special added tokens preempt base vocab for exact content collisions.
    // Special tokens that only exist in added_tokens are intentionally not
    // surfaced by token_to_id; decode skip_special behavior relies on this.
    var added_iter = tok.tok.tokenizer_handle.added;
    while (added_iter) |added| : (added_iter = added.next) {
        if (added.special == 0) {
            if (added.content) |content_ptr| {
                const content = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(content_ptr)), 0);
                if (std.mem.eql(u8, content, token_slice)) {
                    return added.id;
                }
            }
        }
    }

    // Returns -1 if token not found (not an error, just "not in vocabulary")
    return tok.tok.tokenizer_handle.tokenToId(token_slice) orelse -1;
}

// =============================================================================
// Token Utilities
// =============================================================================

/// Frees a token ID array.
pub export fn talu_tokens_free(tokens: ?[*]u32, num_tokens: usize) callconv(.c) void {
    if (tokens) |t| if (num_tokens > 0) allocator.free(t[0..num_tokens]);
}

/// Concatenates two token arrays into a new array.
///
/// Caller must free the result via talu_tokens_free().
pub export fn talu_tokens_concat(
    tokens_a: ?[*]const u32,
    num_a: usize,
    tokens_b: ?[*]const u32,
    num_b: usize,
) callconv(.c) ?[*]u32 {
    const total, const overflow = @addWithOverflow(num_a, num_b);
    if (overflow != 0) return null;
    if (total == 0) return null;
    if (num_a > 0 and tokens_a == null) return null;
    if (num_b > 0 and tokens_b == null) return null;

    const out = allocator.alloc(u32, total) catch return null;
    if (tokens_a) |a| @memcpy(out[0..num_a], a[0..num_a]);
    if (tokens_b) |b| @memcpy(out[num_a..total], b[0..num_b]);
    return out.ptr;
}

/// Frees a NUL-terminated string.
pub export fn talu_text_free(text: ?[*:0]u8) callconv(.c) void {
    const t = text orelse return;
    var len: usize = 0;
    while (t[len] != 0) : (len += 1) {}
    allocator.free(t[0 .. len + 1]);
}

/// Frees decode result text buffer.
pub export fn talu_decode_result_free(text: ?[*]u8, text_len: usize) callconv(.c) void {
    if (text) |t| if (text_len > 0) allocator.free(t[0..text_len]);
}

// =============================================================================
// Fuzz Tests
// =============================================================================

test "fuzz tokenizer JSON parsing with random input" {
    // Fuzz the tokenizer JSON parser with arbitrary byte sequences.
    // The loader should never crash, only return errors gracefully.
    try std.testing.fuzz({}, struct {
        fn testOne(_: void, input: []const u8) !void {
            // Need null-terminated string for C API
            const json_z = std.testing.allocator.allocSentinel(u8, input.len, 0) catch return;
            defer std.testing.allocator.free(json_z[0 .. input.len + 1]);
            @memcpy(json_z[0..input.len], input);

            // Try to create tokenizer from JSON - should not crash
            const tok = tok_pipeline.c_api.tokenizer_from_json_string(json_z.ptr);
            if (tok) |t| {
                tok_pipeline.c_api.tokenizer_free(t);
            }
            // Any return is fine - we just want no crashes
        }
    }.testOne, .{});
}

const regression_test_tokenizer_json =
    \\{
    \\  "version": "1.0",
    \\  "model": {
    \\    "type": "BPE",
    \\    "vocab": {"<unk>": 0, "a": 1},
    \\    "merges": []
    \\  },
    \\  "added_tokens": [
    \\    {"id": 0, "content": "<unk>", "special": true}
    \\  ],
    \\  "normalizer": null,
    \\  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": true},
    \\  "post_processor": null,
    \\  "decoder": {"type": "ByteLevel"}
    \\}
;

fn createRegressionTestTokenizerHandle() !*OpaqueTokenizerHandle {
    var handle: ?*OpaqueTokenizerHandle = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_tokenizer_create_from_json(
            regression_test_tokenizer_json.ptr,
            regression_test_tokenizer_json.len,
            &handle,
        ),
    );
    return handle.?;
}

fn expectNoLastError() !void {
    try std.testing.expectEqual(@as(i32, 0), capi_error.talu_last_error_code());
    try std.testing.expect(capi_error.talu_last_error() == null);
}

fn expectLastError(expected_code: error_codes.ErrorCode, expected_message: []const u8) !void {
    try std.testing.expectEqual(@as(i32, @intFromEnum(expected_code)), capi_error.talu_last_error_code());
    const message = capi_error.talu_last_error() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(expected_message, std.mem.span(message));
    capi_error.talu_clear_error();
}

test "talu_tokenizer_encode empty input returns sliceable empty buffers" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    capi_error.setError(error.InvalidArgument, "stale error", .{});
    var options = std.mem.zeroes(EncodeOptions);
    const result = talu_tokenizer_encode(handle, "".ptr, 0, &options);
    defer talu_encode_result_free(result);

    try std.testing.expect(result.error_msg == null);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try std.testing.expect(result.ids != null);
    try std.testing.expect(result.offsets != null);
    try std.testing.expect(result.attention_mask != null);
    try std.testing.expect(result.special_tokens_mask != null);
    try expectNoLastError();
}

test "talu_tokenizer_tokenize empty input returns sliceable empty token array" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    capi_error.setError(error.InvalidArgument, "stale error", .{});
    const result = talu_tokenizer_tokenize(handle, "".ptr, 0);
    defer talu_tokenize_result_free(result.tokens, result.num_tokens);

    try std.testing.expect(result.error_msg == null);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try std.testing.expect(result.tokens != null);
    try expectNoLastError();
}

test "talu_tokenizer_tokenize_bytes empty input returns sliceable empty byte buffer" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    capi_error.setError(error.InvalidArgument, "stale error", .{});
    const result = talu_tokenizer_tokenize_bytes(handle, "".ptr, 0);

    try std.testing.expect(result.error_msg == null);
    try std.testing.expectEqual(@as(usize, 0), result.data_len);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try std.testing.expect(result.data != null);
    try std.testing.expect(result.offsets != null);
    try std.testing.expectEqual(@as(usize, 0), result.offsets.?[0]);
    try expectNoLastError();

    // Empty byte tokenization returns sentinel pointers; free must treat them as a no-op.
    talu_tokenize_bytes_result_free(result.data, result.data_len, result.offsets, result.num_tokens);
}

test "talu_tokenizer_decode null tokens with non-zero length returns error" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    const result = talu_tokenizer_decode(handle, null, 3, null);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(result.text == null);
    try std.testing.expectEqual(@as(usize, 0), result.text_len);
    try expectLastError(.invalid_argument, "tokens is null with non-zero length");
}

test "talu_tokenizer_encode null text with non-zero length returns error" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    const result = talu_tokenizer_encode(handle, null, 5, null);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(result.ids == null);
    try std.testing.expect(result.offsets == null);
    try std.testing.expect(result.attention_mask == null);
    try std.testing.expect(result.special_tokens_mask == null);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try expectLastError(.invalid_argument, "text is null with non-zero length");
}

test "talu_tokenizer_tokenize null text with non-zero length returns error" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    const result = talu_tokenizer_tokenize(handle, null, 5);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(result.tokens == null);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try expectLastError(.invalid_argument, "text is null with non-zero length");
}

test "talu_tokenizer_tokenize_bytes null text with non-zero length returns error" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    const result = talu_tokenizer_tokenize_bytes(handle, null, 5);
    try std.testing.expect(result.error_msg != null);
    try std.testing.expect(result.data == null);
    try std.testing.expectEqual(@as(usize, 0), result.data_len);
    try std.testing.expect(result.offsets == null);
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
    try expectLastError(.invalid_argument, "text is null with non-zero length");
}

test "talu_tokenizer_encode_batch null arrays with non-zero count return error" {
    const handle = try createRegressionTestTokenizerHandle();
    defer talu_tokenizer_free(handle);

    const null_texts = talu_tokenizer_encode_batch(handle, null, &[_]usize{5}, 1, null);
    try std.testing.expect(null_texts.error_msg != null);
    try std.testing.expect(null_texts.ids == null);
    try std.testing.expect(null_texts.offsets == null);
    try std.testing.expectEqual(@as(usize, 0), null_texts.total_tokens);
    try std.testing.expectEqual(@as(usize, 0), null_texts.num_sequences);
    try expectLastError(.invalid_argument, "texts is null with non-zero count");

    const null_lengths = talu_tokenizer_encode_batch(
        handle,
        @ptrCast(&[_][*c]const u8{"hello".ptr}),
        null,
        1,
        null,
    );
    try std.testing.expect(null_lengths.error_msg != null);
    try std.testing.expect(null_lengths.ids == null);
    try std.testing.expect(null_lengths.offsets == null);
    try std.testing.expectEqual(@as(usize, 0), null_lengths.total_tokens);
    try std.testing.expectEqual(@as(usize, 0), null_lengths.num_sequences);
    try expectLastError(.invalid_argument, "lengths is null with non-zero count");
}
