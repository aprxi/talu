//! Router C API - Generation, Streaming, Embeddings, and Backend Management
//!
//! C-callable functions for the Router subsystem. Routes generation requests
//! to inference backends (local engines or remote APIs).
//!
//! Architecture:
//!   - Generation: sync and callback-based streaming via InferenceBackend
//!   - Embeddings: text embedding extraction with configurable pooling
//!   - Config: model spec validation, canonicalization, and backend creation
//!   - Backend: lifecycle management for inference backends
//!
//! Maps to Python: talu/chat/_bindings.py (router/backend/config functions)
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const router_mod = @import("../router/root.zig");
const spec_mod = @import("../router/spec.zig");
const capi_types = @import("types.zig");
const progress_mod = @import("progress.zig");
const responses_capi = @import("responses.zig");
const responses_mod = @import("../responses/root.zig");

const allocator = std.heap.c_allocator;

// Error handling
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

// Types from other capi modules
const ChatHandle = responses_capi.ChatHandle;
const Chat = responses_mod.Chat;

const capi_bridge = router_mod.capi_bridge;

// =============================================================================
// Generation Types
// =============================================================================

/// Result from generation.
pub const RouterGenerateResult = capi_bridge.CGenerateResult;

/// Logit bias entry for generation.
pub const CLogitBiasEntry = capi_bridge.CLogitBiasEntry;

/// Generation configuration.
pub const RouterGenerateConfig = capi_bridge.CGenerateConfig;

/// Pooling strategy for embeddings.
pub const CPoolingStrategy = enum(u8) {
    last = 0,
    mean = 1,
    first = 2,
};

/// Content part for generation input.
pub const GenerateContentPart = capi_bridge.GenerateContentPart;

// =============================================================================
// Model Specification Types
// =============================================================================

pub const BackendType = capi_types.BackendType;
pub const BackendUnion = capi_types.BackendUnion;
pub const TaluModelSpec = capi_types.TaluModelSpec;
pub const TaluCapabilities = capi_types.TaluCapabilities;

/// Opaque handle for canonical spec (wraps spec_mod.CanonicalSpec).
pub const TaluCanonicalSpec = opaque {};

/// Opaque handle for inference backend (wraps spec_mod.InferenceBackend).
pub const TaluInferenceBackend = opaque {};

/// Model information from a remote endpoint.
pub const RemoteModelInfo = capi_bridge.CRemoteModelInfo;

/// Result from listing remote models.
pub const RemoteModelListResult = capi_bridge.CRemoteModelListResult;

/// Options for backend creation.
pub const BackendCreateOptions = extern struct {
    progress_callback: ?progress_mod.CProgressCallback = null,
    progress_user_data: ?*anyopaque = null,

    pub fn progressContext(self: BackendCreateOptions) progress_mod.ProgressContext {
        return progress_mod.ProgressContext.init(self.progress_callback, self.progress_user_data);
    }
};

// =============================================================================
// Generation API
// =============================================================================

/// Frees a generation result returned by talu_router_generate_with_backend.
///
/// Passing null is a safe no-op.
pub export fn talu_router_result_free(result: ?*RouterGenerateResult) callconv(.c) void {
    const ptr = result orelse return;
    capi_bridge.freeResult(allocator, ptr);
}

/// Closes all cached inference engines and frees their resources.
///
/// Call this when shutting down or when memory needs to be reclaimed.
pub export fn talu_router_close_all() callconv(.c) void {
    router_mod.closeAllEngines(allocator);
}

// =============================================================================
// Spec-Based Generation API
// =============================================================================

/// Generates a response using a spec-based InferenceBackend.
/// Caller must free the result via talu_router_result_free().
pub export fn talu_router_generate_with_backend(
    chat_handle: ?*ChatHandle,
    parts: ?[*]const GenerateContentPart,
    num_parts: usize,
    backend: ?*TaluInferenceBackend,
    config: ?*const RouterGenerateConfig,
) callconv(.c) RouterGenerateResult {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    }));
    // Allow null parts with num_parts==0 for continuation calls (agent loop).
    // When parts are null/empty, generateWithBackend continues from the
    // existing conversation state without appending a new user message.
    const empty_parts: [0]GenerateContentPart = .{};
    const effective_parts: []const GenerateContentPart = if (parts) |p|
        p[0..num_parts]
    else if (num_parts == 0)
        &empty_parts
    else {
        capi_error.setErrorWithCode(.invalid_argument, "parts is null but num_parts > 0", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    };
    const backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    }));

    const result = capi_bridge.generateWithBackend(allocator, chat, effective_parts, backend_ptr, config);
    return capi_bridge.toCResult(allocator, result);
}

// =============================================================================
// Streaming Callback API
// =============================================================================

/// Streaming callback type.
/// Called per decoded text segment: (text_ptr, text_len, item_type, content_type, is_final, userdata).
/// Return 1 to continue, 0 to stop generation.
pub const StreamCallback = capi_bridge.StreamCallback;

/// Generates a response with per-token streaming via callback.
///
/// The callback fires once per decoded text segment (after UTF-8 assembly and
/// reasoning-tag filtering). Blocks until generation completes or callback
/// returns 0. Returns the same result struct as talu_router_generate_with_backend.
///
/// Only supported for local backends. Returns error for remote/HTTP backends.
pub export fn talu_router_generate_streaming(
    chat_handle: ?*ChatHandle,
    parts: ?[*]const GenerateContentPart,
    num_parts: usize,
    backend: ?*TaluInferenceBackend,
    config: ?*const RouterGenerateConfig,
    stream_cb: ?*anyopaque,
    stream_cb_data: ?*anyopaque,
) callconv(.c) RouterGenerateResult {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    }));
    const empty_parts: [0]GenerateContentPart = .{};
    const effective_parts: []const GenerateContentPart = if (parts) |p|
        p[0..num_parts]
    else if (num_parts == 0)
        &empty_parts
    else {
        capi_error.setErrorWithCode(.invalid_argument, "parts is null but num_parts > 0", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    };
    const backend_ptr: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    }));
    // Cast void pointer back to function pointer (Rust passes fn ptr as void*).
    const cb: StreamCallback = @ptrCast(stream_cb orelse {
        capi_error.setErrorWithCode(.invalid_argument, "stream_cb is null", .{});
        return capi_bridge.toCResult(allocator, .{ .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) });
    });

    const result = capi_bridge.generateStreamingWithBackend(allocator, chat, effective_parts, backend_ptr, config, cb, stream_cb_data);
    return capi_bridge.toCResult(allocator, result);
}

// =============================================================================
// Embedding API
// =============================================================================

/// Gets the embedding dimension for a model.
///
/// Returns 0 if the model cannot be loaded or does not support embeddings.
pub export fn talu_router_embedding_dim(model: ?[*:0]const u8) callconv(.c) usize {
    const model_id = std.mem.sliceTo(model orelse return 0, 0);
    const engine = router_mod.getOrCreateEngine(allocator, model_id) catch return 0;
    return engine.embeddingDim();
}

/// Extracts embeddings from text using the specified model.
///
/// Returns 0 on success, non-zero error code on failure.
/// Caller must free the embedding via talu_router_embedding_free().
pub export fn talu_router_embed(
    model: ?[*:0]const u8,
    text: ?[*:0]const u8,
    pooling: CPoolingStrategy,
    normalize: bool,
    out_embedding: *?[*]f32,
    out_dim: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_embedding.* = null;
    out_dim.* = 0;

    const model_id = std.mem.sliceTo(model orelse {
        capi_error.setErrorWithCode(.invalid_argument, "model is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const text_input = std.mem.sliceTo(text orelse {
        capi_error.setErrorWithCode(.invalid_argument, "text is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    const engine = router_mod.getOrCreateEngine(allocator, model_id) catch |err| {
        capi_error.setError(err, "failed to load model", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const dim = engine.embeddingDim();

    const buffer = allocator.alloc(f32, dim) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate embedding buffer", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.free(buffer);

    engine.embed(text_input, @enumFromInt(@intFromEnum(pooling)), normalize, buffer) catch |err| {
        allocator.free(buffer);
        capi_error.setError(err, "embedding extraction failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_embedding.* = buffer.ptr;
    out_dim.* = dim;
    return 0;
}

/// Frees embedding memory returned by talu_router_embed().
pub export fn talu_router_embedding_free(embedding: ?[*]f32, dim: usize) callconv(.c) void {
    if (embedding) |ptr| {
        if (dim > 0) allocator.free(ptr[0..dim]);
    }
}

// =============================================================================
// Model Specification API
// =============================================================================

pub export fn talu_config_validate(spec: ?*const TaluModelSpec) callconv(.c) c_int {
    capi_error.clearError();
    const spec_ptr = spec orelse {
        capi_error.setErrorWithCode(.invalid_argument, "spec is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const result = spec_mod.validateSpecDetailed(spec_ptr);
    if (!result.valid) {
        const msg = spec_mod.validationIssueMessage(result.issue);
        const code = validationIssueToErrorCode(result.issue);
        capi_error.setErrorWithCode(code, "validation failed: {s}", .{msg});
        return @intFromEnum(code);
    }
    return 0;
}

pub export fn talu_config_canonicalize(
    in_spec: ?*const TaluModelSpec,
    out_handle: ?*?*TaluCanonicalSpec,
) callconv(.c) c_int {
    capi_error.clearError();
    const out_ptr = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_ptr.* = null;

    const spec_ptr = in_spec orelse {
        capi_error.setErrorWithCode(.invalid_argument, "in_spec is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    var canonical = spec_mod.canonicalizeSpec(allocator, spec_ptr) catch |err| {
        capi_error.setError(err, "canonicalize failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const boxed = allocator.create(spec_mod.CanonicalSpec) catch {
        canonical.deinit(allocator);
        capi_error.setErrorWithCode(.out_of_memory, "canonicalize allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    boxed.* = canonical;
    out_ptr.* = @ptrCast(boxed);
    return 0;
}

pub export fn talu_config_get_view(
    handle: ?*const TaluCanonicalSpec,
    out_spec: ?*TaluModelSpec,
) callconv(.c) c_int {
    capi_error.clearError();
    const handle_ptr = handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_ptr = out_spec orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_spec is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const canon: *const spec_mod.CanonicalSpec = @ptrCast(@alignCast(handle_ptr));
    spec_mod.getView(canon, out_ptr);
    return 0;
}

pub export fn talu_config_free(handle: ?*TaluCanonicalSpec) callconv(.c) void {
    const handle_ptr = handle orelse return;
    const canon: *spec_mod.CanonicalSpec = @ptrCast(@alignCast(handle_ptr));
    canon.deinit(allocator);
    allocator.destroy(canon);
}

pub export fn talu_backend_get_capabilities(
    backend_type_raw: c_int,
    backend_config: ?*const BackendUnion,
    out_caps: ?*TaluCapabilities,
) callconv(.c) c_int {
    capi_error.clearError();
    const out_ptr = out_caps orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_caps is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const cfg = backend_config orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend_config is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const backend_type = spec_mod.parseBackendType(backend_type_raw) orelse {
        capi_error.setErrorWithCode(.invalid_argument, "invalid backend_type_raw", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (backend_type == .Unspecified) {
        capi_error.setErrorWithCode(.invalid_argument, "backend_type_raw is unspecified", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const caps = spec_mod.getCapabilities(backend_type, cfg);
    out_ptr.* = caps;
    return 0;
}

pub export fn talu_backend_create_from_canonical(
    canon: ?*const TaluCanonicalSpec,
    options: BackendCreateOptions,
    out_backend: ?*?*TaluInferenceBackend,
) callconv(.c) c_int {
    capi_error.clearError();
    const out_ptr = out_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_ptr.* = null;

    const canon_ptr = canon orelse {
        capi_error.setErrorWithCode(.invalid_argument, "canon is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const canon_spec: *const spec_mod.CanonicalSpec = @ptrCast(@alignCast(canon_ptr));

    const progress = options.progressContext();
    var backend = spec_mod.createInferenceBackend(allocator, canon_spec, progress) catch |err| {
        capi_error.setError(err, "backend create failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const boxed = allocator.create(spec_mod.InferenceBackend) catch {
        backend.deinit(allocator);
        capi_error.setErrorWithCode(.out_of_memory, "backend allocation failed", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    boxed.* = backend;
    out_ptr.* = @ptrCast(boxed);
    return 0;
}

pub export fn talu_backend_free(backend: ?*TaluInferenceBackend) callconv(.c) void {
    const backend_ptr = backend orelse return;
    const boxed: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend_ptr));
    boxed.deinit(allocator);
    allocator.destroy(boxed);
}

pub export fn talu_backend_synchronize(backend: ?*TaluInferenceBackend) callconv(.c) c_int {
    capi_error.clearError();
    const backend_ptr = backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const boxed: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend_ptr));
    boxed.synchronize() catch |err| {
        capi_error.setError(err, "backend synchronize failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

// =============================================================================
// Remote Model Listing
// =============================================================================

/// List models from a remote OpenAI-compatible backend.
pub export fn talu_backend_list_models(backend: ?*TaluInferenceBackend) callconv(.c) RemoteModelListResult {
    capi_error.clearError();

    const backend_ptr = backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend is null", .{});
        return .{ .models = null, .count = 0, .error_code = @intFromEnum(error_codes.ErrorCode.invalid_argument) };
    };

    const boxed: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend_ptr));
    const result = capi_bridge.listModels(allocator, boxed);

    if (result.error_code != 0) {
        capi_error.setErrorWithCode(@enumFromInt(result.error_code), "failed to list models", .{});
    }

    return result;
}

/// Free result from talu_backend_list_models.
///
/// Passing null is a safe no-op.
pub export fn talu_backend_list_models_free(result: ?*RemoteModelListResult) callconv(.c) void {
    const ptr = result orelse return;
    capi_bridge.freeModelListResult(allocator, ptr);
}

// =============================================================================
// Model Info
// =============================================================================

/// Static model metadata returned by talu_backend_model_info.
/// All fields are zero when the backend is null or remote (no local engine).
pub const CModelInfo = extern struct {
    file_size: u64,
    tensor_count: u64,
    vocab_size: i32,
    d_model: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_groups: i32,
    d_ff: i32,
    max_seq_len: i32,
    gaffine_group_size: i32,
    weight_dtype: u8,
    _pad: [7]u8 = .{0} ** 7,
};

/// Return static model metadata from a local backend.
/// Returns a zero struct if the backend is null or remote.
pub export fn talu_backend_model_info(
    backend: ?*TaluInferenceBackend,
) callconv(.c) CModelInfo {
    const backend_ptr = backend orelse return std.mem.zeroes(CModelInfo);
    const boxed: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend_ptr));
    const engine = boxed.getLocalEngine() orelse return std.mem.zeroes(CModelInfo);
    const loaded = engine.loaded;
    const cfg = loaded.config;
    return .{
        .file_size = @intCast(loaded.file_size),
        .tensor_count = @intCast(loaded.tensor_count),
        .vocab_size = cfg.vocab_size,
        .d_model = cfg.d_model,
        .n_layers = cfg.n_layers,
        .n_heads = cfg.n_heads,
        .n_kv_groups = cfg.n_kv_groups,
        .d_ff = cfg.d_ff,
        .max_seq_len = cfg.max_seq_len,
        .gaffine_group_size = cfg.gaffine_group_size,
        .weight_dtype = @intFromEnum(loaded.original_weight_dtype),
        ._pad = .{0} ** 7,
    };
}

// =============================================================================
// Helpers
// =============================================================================

fn validationIssueToErrorCode(issue: spec_mod.ValidationIssue) error_codes.ErrorCode {
    return switch (issue) {
        .bad_abi => .unsupported_abi_version,
        .model_not_found => .model_not_found,
        else => .invalid_argument,
    };
}
