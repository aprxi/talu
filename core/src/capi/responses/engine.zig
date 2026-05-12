//! Backend C API - Embeddings, Backend Management, and Configuration.
//!
//! C-callable functions for backend lifecycle, embeddings, and model
//! configuration. Local generation is served by the batch C API.
//!
//! Thread safety: NOT thread-safe. All access must be from a single thread.

const std = @import("std");
const responses_mod = @import("../../responses/root.zig");
const spec_mod = @import("../../responses/spec.zig");
const capi_types = @import("../types.zig");
const progress_mod = @import("../progress.zig");

const allocator = std.heap.c_allocator;

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");

const capi_bridge = responses_mod.capi_bridge;
const embeddings_mod = responses_mod.embeddings;

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

pub const BackendType = capi_types.BackendType;
pub const BackendUnion = capi_types.BackendUnion;
pub const TaluModelSpec = capi_types.TaluModelSpec;
pub const TaluCapabilities = capi_types.TaluCapabilities;

/// Opaque handle for canonical spec (wraps spec_mod.CanonicalSpec).
pub const TaluCanonicalSpec = opaque {};

/// Opaque handle for inference backend (wraps spec_mod.InferenceBackend).
pub const TaluInferenceBackend = opaque {};

/// Options for backend creation.
pub const BackendCreateOptions = extern struct {
    progress_callback: ?progress_mod.CProgressCallback = null,
    progress_user_data: ?*anyopaque = null,

    pub fn progressContext(self: BackendCreateOptions) progress_mod.ProgressContext {
        return progress_mod.ProgressContext.init(self.progress_callback, self.progress_user_data);
    }
};

/// Closes all cached inference engines and frees their resources.
///
/// Call this when shutting down or when memory needs to be reclaimed.
pub export fn talu_router_close_all() callconv(.c) void {
    responses_mod.closeAllEngines(allocator);
}

// =============================================================================
// Embedding API
// =============================================================================

/// Gets the embedding dimension for a model.
///
/// Returns 0 if the model cannot be loaded or does not support embeddings.
pub export fn talu_router_embedding_dim(model: ?[*:0]const u8) callconv(.c) usize {
    const model_id = std.mem.sliceTo(model orelse return 0, 0);
    const engine = responses_mod.getOrCreateEngine(allocator, model_id) catch return 0;
    return embeddings_mod.dimension(engine);
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

    const engine = responses_mod.getOrCreateEngine(allocator, model_id) catch |err| {
        capi_error.setError(err, "failed to load model", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const result = embeddings_mod.extract(
        allocator,
        engine,
        text_input,
        @enumFromInt(@intFromEnum(pooling)),
        normalize,
    ) catch |err| {
        capi_error.setError(err, "embedding extraction failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_embedding.* = result.values.ptr;
    out_dim.* = result.values.len;
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
// Model Info
// =============================================================================

/// Static model metadata returned by talu_backend_model_info.
/// All fields are zero when the backend is null or non-local.
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
/// Returns a zero struct if the backend is null or non-local.
pub export fn talu_backend_model_info(
    backend: ?*TaluInferenceBackend,
) callconv(.c) CModelInfo {
    const backend_ptr = backend orelse return std.mem.zeroes(CModelInfo);
    const boxed: *spec_mod.InferenceBackend = @ptrCast(@alignCast(backend_ptr));
    const engine = boxed.getLocalEngine() orelse return std.mem.zeroes(CModelInfo);
    const cfg = engine.model_config;
    return .{
        .file_size = @intCast(engine.model_file_size),
        .tensor_count = @intCast(engine.model_tensor_count),
        .vocab_size = cfg.vocab_size,
        .d_model = cfg.d_model,
        .n_layers = cfg.n_layers,
        .n_heads = cfg.n_heads,
        .n_kv_groups = cfg.n_kv_groups,
        .d_ff = cfg.d_ff,
        .max_seq_len = cfg.max_seq_len,
        .gaffine_group_size = cfg.gaffine_group_size,
        .weight_dtype = engine.model_weight_dtype_tag,
        ._pad = .{0} ** 7,
    };
}
