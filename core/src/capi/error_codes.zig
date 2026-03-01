//! C API error code definitions.
//!
//! Centralized error code mapping for the FFI boundary.
const std = @import("std");

pub const ErrorCode = enum(i32) {
    // Success
    ok = 0,

    // Model errors (100-199)
    model_not_found = 100,
    model_invalid_format = 101,
    model_unsupported_architecture = 102,
    model_config_missing = 103,
    model_weights_missing = 104,
    model_weights_corrupted = 105,

    // Tokenizer errors (200-299)
    tokenizer_not_found = 200,
    tokenizer_invalid_format = 201,
    tokenizer_encode_failed = 202,
    tokenizer_decode_failed = 203,
    tokenizer_invalid_token_id = 204,

    // Generation errors (300-399)
    generation_failed = 300,
    generation_empty_prompt = 301,
    generation_context_overflow = 302,
    generation_invalid_params = 303,

    // Conversion errors (400-499)
    convert_failed = 400,
    convert_unsupported_format = 401,
    convert_already_quantized = 402,
    convert_output_exists = 403,

    // I/O errors (500-599)
    io_file_not_found = 500,
    io_permission_denied = 501,
    io_read_failed = 502,
    io_write_failed = 503,
    io_network_failed = 504,
    io_path_invalid = 505,
    io_path_outside_workspace = 506,
    io_parent_not_found = 507,
    io_is_directory = 508,
    io_not_directory = 509,
    io_not_empty = 510,
    io_file_too_big = 511,

    // Template errors (600-699)
    template_syntax_error = 600,
    template_undefined_var = 601,
    template_type_error = 602,
    template_render_failed = 603,
    template_not_found = 604,
    template_invalid_json = 605,
    template_raise_exception = 606,

    // Storage errors (700-799)
    storage_error = 700, // Storage backend operation failed (items may be in memory only)
    resource_busy = 701, // Resource locked by another process
    item_not_found = 702, // Requested item_id does not exist
    session_not_found = 703, // Requested session does not exist
    tag_not_found = 704, // Requested tag does not exist

    // Shell errors (800-899)
    shell_command_denied = 800,
    shell_exec_failed = 801,

    // System errors (900-999)
    out_of_memory = 900,
    invalid_argument = 901,
    invalid_handle = 902,
    ambiguous_backend = 903,
    unsupported_abi_version = 904,
    resource_exhausted = 905,
    internal_error = 999,
};

/// Map a Zig error to a stable ErrorCode.
pub fn errorToCode(err: anyerror) ErrorCode {
    return switch (err) {
        error.FileNotFound => .io_file_not_found,
        error.ModelNotCached => .model_not_found,
        error.OutOfMemory => .out_of_memory,
        error.AccessDenied => .io_permission_denied,
        error.InvalidPath => .io_path_invalid,
        error.PathOutsideWorkspace => .io_path_outside_workspace,
        error.ParentNotFound => .io_parent_not_found,
        error.IsDir => .io_is_directory,
        error.NotDir => .io_not_directory,
        error.DirNotEmpty => .io_not_empty,
        error.FileTooBig => .io_file_too_big,
        error.InvalidArgument => .invalid_argument,
        error.AlreadyExists => .invalid_argument,
        error.IdempotencyConflict => .invalid_argument,
        error.ManifestGenerationConflict => .invalid_argument,
        error.InvalidHandle => .invalid_handle,
        error.ResourceExhausted => .resource_exhausted,
        error.ZeroVectorNotAllowed => .invalid_argument,
        error.InvalidColumnData => .invalid_argument,
        error.InvalidJson => .invalid_argument,
        error.InvalidSchema => .invalid_argument,
        error.InputTooLarge => .invalid_argument,
        error.InputTooDeep => .invalid_argument,
        error.StringTooLong => .invalid_argument,
        error.InvalidTemperature, error.InvalidTopP, error.InvalidTopK => .generation_invalid_params,
        // Template high-level errors (from root.zig Error type)
        error.LexError, error.ParseError => .template_syntax_error,
        error.IncludeTypeError => .template_render_failed,
        // Template lexer errors (syntax)
        error.UnterminatedString, error.UnterminatedTag, error.InvalidCharacter, error.UnterminatedComment => .template_syntax_error,
        // Template parser errors (syntax)
        error.UnexpectedToken, error.UnexpectedEof, error.InvalidSyntax, error.UnclosedBlock, error.InvalidSlice => .template_syntax_error,
        // Template eval errors
        error.UndefinedVariable => .template_undefined_var,
        error.TypeError => .template_type_error,
        error.EvalError => .template_render_failed,
        error.RaiseException => .template_raise_exception,
        error.IndexOutOfBounds, error.KeyError, error.DivisionByZero, error.InvalidOperation => .template_render_failed,
        error.UnsupportedFilter, error.UnsupportedTest, error.UnsupportedMethod => .template_render_failed,
        // JSON parse errors
        error.SyntaxError => .template_invalid_json,
        // Spec/config errors
        error.AmbiguousBackend => .ambiguous_backend,
        error.UnsupportedAbiVersion => .unsupported_abi_version,
        error.ModelNotFound => .model_not_found,
        // Model architecture errors
        error.UnsupportedModel => .model_unsupported_architecture,
        // Storage errors
        error.LockUnavailable => .resource_busy,
        error.ItemNotFound => .item_not_found,
        error.SessionNotFound => .session_not_found,
        error.TagNotFound => .tag_not_found,
        error.StorageForkFailed => .storage_error,
        error.InvalidTokenId => .tokenizer_invalid_token_id,
        // Image pipeline errors
        error.UnsupportedImageFormat => .convert_unsupported_format,
        error.ImageInputTooLarge,
        error.ImageDimensionExceeded,
        error.ImagePixelCountExceeded,
        error.ImageOutputTooLarge,
        => .resource_exhausted,
        error.InvalidImageDimensions,
        error.InvalidPixelFormat,
        error.UnsupportedStride,
        error.JpegInitFailed,
        error.JpegHeaderFailed,
        error.JpegDecodeFailed,
        error.JpegEncodeFailed,
        error.PngInitFailed,
        error.PngSetBufferFailed,
        error.PngHeaderFailed,
        error.PngSizeFailed,
        error.PngDecodeFailed,
        error.PngEncodeFailed,
        error.WebpHeaderFailed,
        error.WebpDecodeFailed,
        => .invalid_argument,
        else => .internal_error,
    };
}

test "ErrorCode: error codes are stable" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(ErrorCode.ok));
    try std.testing.expectEqual(@as(i32, 100), @intFromEnum(ErrorCode.model_not_found));
    try std.testing.expectEqual(@as(i32, 301), @intFromEnum(ErrorCode.generation_empty_prompt));
    try std.testing.expectEqual(@as(i32, 900), @intFromEnum(ErrorCode.out_of_memory));
    try std.testing.expectEqual(@as(i32, 903), @intFromEnum(ErrorCode.ambiguous_backend));
    try std.testing.expectEqual(@as(i32, 904), @intFromEnum(ErrorCode.unsupported_abi_version));
}
