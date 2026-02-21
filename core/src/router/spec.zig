//! Model Specification - Structured model configuration for multi-backend routing.
//!
//! ModelSpec provides a formal way to specify models with backend-specific configuration.
//! This replaces string-based model IDs with structured specs that carry all necessary
//! information for routing and engine creation.
//!
//! ## Supported Backends
//!
//! Current:
//!   - Local: Native inference using talu's engine (gpu_layers, use_mmap, num_threads)
//!   - OpenAICompatible: OpenAI-compatible APIs (base_url, api_key, timeout, retries)
//!
//! Future (Phase 5):
//!   - Anthropic: Claude API
//!   - vLLM: vLLM server
//!   - Ollama: Ollama server
//!   - Bedrock: AWS Bedrock
//!
//! ## Usage
//!
//! ```zig
//! const spec = @import("router/spec.zig");
//!
//! // Validate a spec from C API
//! const result = spec.validateSpecDetailed(&c_spec);
//! if (result.code != .Ok) {
//!     const msg = spec.validationIssueMessage(result.issue);
//!     return error.InvalidSpec;
//! }
//!
//! // Canonicalize (resolve defaults, validate paths)
//! var canonical = try spec.canonicalizeSpec(allocator, &c_spec);
//! defer canonical.deinit(allocator);
//!
//! // Create engine from canonical spec
//! var backend = try spec.createInferenceBackend(allocator, &canonical);
//! defer backend.deinit(allocator);
//! ```

const std = @import("std");
const capi_types = @import("../capi/types.zig");
const local_mod = @import("local.zig");
const http_engine_mod = @import("http_engine.zig");
const repository_scheme = @import("../io/repository/scheme.zig");
const progress_mod = @import("../capi/progress.zig");

pub const HttpEngine = http_engine_mod.HttpEngine;

// =============================================================================
// Type Re-exports (from capi/types.zig for ABI compatibility)
// =============================================================================

pub const BackendType = capi_types.BackendType;
pub const LocalConfig = capi_types.LocalConfig;
pub const OpenAICompatibleConfig = capi_types.OpenAICompatibleConfig;
pub const BackendUnion = capi_types.BackendUnion;
pub const TaluModelSpec = capi_types.TaluModelSpec;
pub const TaluCapabilities = capi_types.TaluCapabilities;

// =============================================================================
// Validation Types
// =============================================================================

pub const ValidationIssue = enum {
    none,
    ref_null,
    ref_empty,
    bad_abi,
    struct_too_small,
    invalid_backend_type,
    local_config_truncated,
    openai_config_truncated,
    timeout_negative,
    retries_negative,
    base_url_invalid,
    model_not_found,
};

pub const ValidationResult = struct {
    valid: bool,
    issue: ValidationIssue,
};

// =============================================================================
// Canonical Specification
// =============================================================================

/// Canonical (validated, resolved) model specification.
/// Created from TaluModelSpec via canonicalizeSpec().
pub const CanonicalSpec = struct {
    backend_type: BackendType,
    ref: [:0]u8,
    backend: CanonicalBackend,

    pub fn deinit(self: *CanonicalSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.ref);
        switch (self.backend) {
            .Local => {},
            .OpenAICompatible => |*cfg| {
                if (cfg.base_url) |url| allocator.free(url);
                if (cfg.api_key) |key| allocator.free(key);
                if (cfg.org_id) |org| allocator.free(org);
                if (cfg.custom_headers_json) |hdrs| allocator.free(hdrs);
            },
            .Unspecified => {},
        }
    }
};

const CanonicalBackend = union(BackendType) {
    Unspecified: void,
    Local: CanonicalLocal,
    OpenAICompatible: CanonicalOpenAI,
};

const CanonicalLocal = struct {
    gpu_layers: c_int,
    use_mmap: u8,
    num_threads: c_int,
};

const CanonicalOpenAI = struct {
    base_url: ?[:0]u8,
    api_key: ?[:0]u8,
    org_id: ?[:0]u8,
    timeout_ms: c_int,
    max_retries: c_int,
    custom_headers_json: ?[:0]u8,
};

// =============================================================================
// Inference Backend
// =============================================================================

/// Backend that performs inference, created from a CanonicalSpec.
/// Wraps LocalEngine for native inference or HttpEngine for remote inference.
pub const InferenceBackend = struct {
    backend_type: BackendType,
    backend: BackendImpl,
    /// Model reference (owned copy for remote backends).
    model_ref: ?[]u8 = null,

    pub fn deinit(self: *InferenceBackend, allocator: std.mem.Allocator) void {
        switch (self.backend) {
            .Local => |engine| {
                engine.deinit();
                allocator.destroy(engine);
            },
            .OpenAICompatible => |engine| {
                engine.deinit();
                allocator.destroy(engine);
            },
            .Unspecified => {},
        }
        if (self.model_ref) |ref| allocator.free(ref);
    }

    /// Get the LocalEngine if this is a local backend.
    pub fn getLocalEngine(self: *InferenceBackend) ?*local_mod.LocalEngine {
        return switch (self.backend) {
            .Local => |engine| engine,
            else => null,
        };
    }

    /// Get the HttpEngine if this is an OpenAI-compatible backend.
    pub fn getHttpEngine(self: *InferenceBackend) ?*http_engine_mod.HttpEngine {
        return switch (self.backend) {
            .OpenAICompatible => |engine| engine,
            else => null,
        };
    }

    /// Check if this is a remote backend (OpenAI-compatible).
    pub fn isRemote(self: *const InferenceBackend) bool {
        return self.backend_type == .OpenAICompatible;
    }
};

const BackendImpl = union(BackendType) {
    Unspecified: void,
    Local: *local_mod.LocalEngine,
    OpenAICompatible: *http_engine_mod.HttpEngine,
};

// =============================================================================
// Constants
// =============================================================================

pub const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";
pub const DEFAULT_OPENAI_TIMEOUT_MS: c_int = 60_000;
pub const DEFAULT_OPENAI_MAX_RETRIES: c_int = 3;

pub const MIN_HEADER_SIZE: u32 = @offsetOf(TaluModelSpec, "backend_type_raw") + @sizeOf(c_int);

// =============================================================================
// Validation
// =============================================================================

/// Validate a model specification.
pub fn validateSpec(spec: *const TaluModelSpec) bool {
    return validateSpecDetailed(spec).valid;
}

/// Validate a model specification with detailed error information.
pub fn validateSpecDetailed(spec: *const TaluModelSpec) ValidationResult {
    if (spec.abi_version != 1) return .{ .valid = false, .issue = .bad_abi };
    if (spec.struct_size < MIN_HEADER_SIZE) return .{ .valid = false, .issue = .struct_too_small };

    const ref_ptr = spec.ref orelse return .{ .valid = false, .issue = .ref_null };
    const ref_slice = std.mem.sliceTo(ref_ptr, 0);
    if (ref_slice.len == 0) return .{ .valid = false, .issue = .ref_empty };

    const backend_type = parseBackendType(spec.backend_type_raw) orelse return .{ .valid = false, .issue = .invalid_backend_type };
    switch (backend_type) {
        .Unspecified => return .{ .valid = true, .issue = .none },
        .Local => {
            if (!localConfigInBounds(spec.struct_size)) return .{ .valid = false, .issue = .local_config_truncated };
            if (!pathExists(ref_slice)) return .{ .valid = false, .issue = .model_not_found };
            return .{ .valid = true, .issue = .none };
        },
        .OpenAICompatible => {
            if (!openAiConfigInBounds(spec.struct_size)) return .{ .valid = false, .issue = .openai_config_truncated };
            const timeout_ms = spec.backend_config.openai_compat.timeout_ms;
            const max_retries = spec.backend_config.openai_compat.max_retries;
            if (timeout_ms < 0) return .{ .valid = false, .issue = .timeout_negative };
            if (max_retries < 0) return .{ .valid = false, .issue = .retries_negative };
            if (spec.backend_config.openai_compat.base_url) |base_url_ptr| {
                const base_url = std.mem.sliceTo(base_url_ptr, 0);
                if (!isValidBaseUrl(base_url)) return .{ .valid = false, .issue = .base_url_invalid };
            }
            return .{ .valid = true, .issue = .none };
        },
    }
}

/// Get a human-readable message for a validation issue.
pub fn validationIssueMessage(issue: ValidationIssue) []const u8 {
    return switch (issue) {
        .none => "ok",
        .ref_null => "ref is null",
        .ref_empty => "ref is empty",
        .bad_abi => "unsupported abi_version",
        .struct_too_small => "struct_size too small",
        .invalid_backend_type => "invalid backend_type_raw",
        .local_config_truncated => "struct_size too small for LocalConfig",
        .openai_config_truncated => "struct_size too small for OpenAICompatibleConfig",
        .timeout_negative => "timeout_ms must be >= 0",
        .retries_negative => "max_retries must be >= 0",
        .base_url_invalid => "base_url must be http(s)://",
        .model_not_found => "model not found",
    };
}

// =============================================================================
// Canonicalization
// =============================================================================

/// Canonicalize a model specification.
/// Resolves defaults, validates paths, and creates an owned copy.
pub fn canonicalizeSpec(
    allocator: std.mem.Allocator,
    spec: *const TaluModelSpec,
) !CanonicalSpec {
    if (spec.abi_version != 1) return error.UnsupportedAbiVersion;
    if (spec.struct_size < MIN_HEADER_SIZE) return error.InvalidArgument;

    const ref_ptr = spec.ref orelse return error.InvalidArgument;
    const ref_slice = std.mem.sliceTo(ref_ptr, 0);
    if (ref_slice.len == 0) return error.InvalidArgument;

    const backend_type = parseBackendType(spec.backend_type_raw) orelse return error.InvalidArgument;
    var resolved_type = backend_type;
    if (backend_type == .Unspecified) {
        resolved_type = try inferBackendType(ref_slice);
    }

    switch (resolved_type) {
        .Local => {
            if (!localConfigInBounds(spec.struct_size)) return error.InvalidArgument;
            // Only check pathExists for refs that look like local paths.
            // HuggingFace Hub model IDs (org/model-name) are resolved later
            // by LocalEngine.init() via io.repository.resolveModelPath().
            const uri = repository_scheme.parse(ref_slice) catch return error.InvalidArgument;
            if (uri.scheme == .local and !pathExists(ref_slice)) return error.ModelNotFound;

            var num_threads = spec.backend_config.local.num_threads;
            if (num_threads == 0) {
                const cpu_count = std.Thread.getCpuCount() catch 0;
                num_threads = if (cpu_count > 0) @intCast(cpu_count) else 1;
            }

            const ref_copy = try allocator.dupeZ(u8, ref_slice);
            return CanonicalSpec{
                .backend_type = .Local,
                .ref = ref_copy,
                .backend = .{
                    .Local = .{
                        .gpu_layers = spec.backend_config.local.gpu_layers,
                        .use_mmap = spec.backend_config.local.use_mmap,
                        .num_threads = num_threads,
                    },
                },
            };
        },
        .OpenAICompatible => {
            if (!openAiConfigInBounds(spec.struct_size)) return error.InvalidArgument;
            const timeout_ms_raw = spec.backend_config.openai_compat.timeout_ms;
            const max_retries_raw = spec.backend_config.openai_compat.max_retries;
            if (timeout_ms_raw < 0 or max_retries_raw < 0) return error.InvalidArgument;
            if (spec.backend_config.openai_compat.base_url) |base_url_ptr| {
                const base_url = std.mem.sliceTo(base_url_ptr, 0);
                if (!isValidBaseUrl(base_url)) return error.InvalidArgument;
            }

            const ref_copy = try allocator.dupeZ(u8, ref_slice);
            errdefer allocator.free(ref_copy);
            const base_url = try resolveBaseUrl(allocator, spec.backend_config.openai_compat.base_url);
            errdefer if (base_url) |url| allocator.free(url);
            const api_key = try resolveApiKey(allocator, spec.backend_config.openai_compat.api_key);
            errdefer if (api_key) |key| allocator.free(key);
            const org_id = try dupeOptionalZ(allocator, spec.backend_config.openai_compat.org_id);
            errdefer if (org_id) |oid| allocator.free(std.mem.sliceTo(oid, 0));
            const custom_headers_json = try dupeOptionalZ(allocator, spec.backend_config.openai_compat.custom_headers_json);
            const timeout_ms = if (timeout_ms_raw == 0) DEFAULT_OPENAI_TIMEOUT_MS else timeout_ms_raw;
            const max_retries = if (max_retries_raw == 0) DEFAULT_OPENAI_MAX_RETRIES else max_retries_raw;

            return CanonicalSpec{
                .backend_type = .OpenAICompatible,
                .ref = ref_copy,
                .backend = .{
                    .OpenAICompatible = .{
                        .base_url = base_url,
                        .api_key = api_key,
                        .org_id = org_id,
                        .timeout_ms = timeout_ms,
                        .max_retries = max_retries,
                        .custom_headers_json = custom_headers_json,
                    },
                },
            };
        },
        .Unspecified => return error.InvalidArgument,
    }
}

/// Get a view of the canonical spec as a TaluModelSpec.
/// The view borrows pointers from the canonical spec.
pub fn getView(
    canon: *const CanonicalSpec,
    out_spec: *TaluModelSpec,
) void {
    const out_bytes = @as([*]u8, @ptrCast(out_spec));
    @memset(out_bytes[0..@sizeOf(TaluModelSpec)], 0);
    out_spec.abi_version = 1;
    out_spec.struct_size = @sizeOf(TaluModelSpec);
    out_spec.ref = canon.ref.ptr;
    out_spec.backend_type_raw = @intFromEnum(canon.backend_type);

    switch (canon.backend) {
        .Local => |cfg| {
            out_spec.backend_config = .{ .local = .{
                .gpu_layers = cfg.gpu_layers,
                .use_mmap = cfg.use_mmap,
                .num_threads = cfg.num_threads,
                ._reserved = .{0} ** 32,
            } };
        },
        .OpenAICompatible => |cfg| {
            out_spec.backend_config = .{ .openai_compat = .{
                .base_url = if (cfg.base_url) |url| url.ptr else null,
                .api_key = if (cfg.api_key) |key| key.ptr else null,
                .org_id = if (cfg.org_id) |org| org.ptr else null,
                .timeout_ms = cfg.timeout_ms,
                .max_retries = cfg.max_retries,
                .custom_headers_json = if (cfg.custom_headers_json) |hdrs| hdrs.ptr else null,
                ._reserved = .{0} ** 24,
            } };
        },
        .Unspecified => {},
    }
}

// =============================================================================
// Capabilities
// =============================================================================

/// Get capabilities for a backend type.
pub fn getCapabilities(
    backend_type: BackendType,
    _: *const BackendUnion,
) TaluCapabilities {
    return switch (backend_type) {
        .Local => .{
            .abi_version = 1,
            .struct_size = @sizeOf(TaluCapabilities),
            .streaming = 1,
            .tool_calling = 1,
            .logprobs = 0,
            .embeddings = 1,
            .json_schema = 1,
            ._reserved = .{0} ** 32,
        },
        .OpenAICompatible => .{
            .abi_version = 1,
            .struct_size = @sizeOf(TaluCapabilities),
            .streaming = 1,
            .tool_calling = 1,
            .logprobs = 1,
            .embeddings = 1,
            .json_schema = 1,
            ._reserved = .{0} ** 32,
        },
        .Unspecified => .{
            .abi_version = 1,
            .struct_size = @sizeOf(TaluCapabilities),
            .streaming = 0,
            .tool_calling = 0,
            .logprobs = 0,
            .embeddings = 0,
            .json_schema = 0,
            ._reserved = .{0} ** 32,
        },
    };
}

// =============================================================================
// Inference Backend Creation
// =============================================================================

/// Create an inference backend from a canonical specification.
pub fn createInferenceBackend(
    allocator: std.mem.Allocator,
    canon: *const CanonicalSpec,
    progress: progress_mod.ProgressContext,
) !InferenceBackend {
    switch (canon.backend_type) {
        .Local => {
            const engine = try allocator.create(local_mod.LocalEngine);
            errdefer allocator.destroy(engine);
            engine.* = try local_mod.LocalEngine.initWithSeedAndResolutionConfig(allocator, canon.ref, 42, .{}, .{}, progress);
            if (std.debug.runtime_safety) {
                // Ensure the engine owns an independent model path copy.
                std.debug.assert(engine.*.model_path.len == canon.ref.len);
                std.debug.assert(std.mem.eql(u8, engine.*.model_path, canon.ref));
                std.debug.assert(engine.*.model_path.ptr != canon.ref.ptr);
            }
            return InferenceBackend{ .backend_type = .Local, .backend = .{ .Local = engine } };
        },
        .OpenAICompatible => {
            const openai_cfg = canon.backend.OpenAICompatible;

            // Create HttpEngine for remote inference
            const http_engine = try allocator.create(http_engine_mod.HttpEngine);
            errdefer allocator.destroy(http_engine);

            // Build HttpEngineConfig from canonical spec
            // base_url must be valid (non-null) for OpenAI backend
            const base_url_slice = openai_cfg.base_url orelse return error.InvalidArgument;

            http_engine.* = try http_engine_mod.HttpEngine.init(allocator, .{
                .base_url = base_url_slice,
                .api_key = openai_cfg.api_key,
                .org_id = openai_cfg.org_id,
                .model = canon.ref,
                .timeout_ms = openai_cfg.timeout_ms,
                .max_retries = openai_cfg.max_retries,
                .custom_headers_json = openai_cfg.custom_headers_json,
            });

            // Copy model ref for the backend
            const model_ref = try allocator.dupe(u8, canon.ref);

            return InferenceBackend{
                .backend_type = .OpenAICompatible,
                .backend = .{ .OpenAICompatible = http_engine },
                .model_ref = model_ref,
            };
        },
        .Unspecified => return error.InvalidArgument,
    }
}

// =============================================================================
// Helpers
// =============================================================================

pub fn parseBackendType(raw: c_int) ?BackendType {
    return switch (raw) {
        -1 => .Unspecified,
        0 => .Local,
        1 => .OpenAICompatible,
        else => null,
    };
}

fn inferBackendType(ref: []const u8) !BackendType {
    if (hasOpenAiScheme(ref)) return .OpenAICompatible;
    if (pathExists(ref)) return .Local;
    // Check if it's a HuggingFace Hub model ID (org/model-name)
    // These resolve to local paths after download/caching
    const uri = repository_scheme.parse(ref) catch return error.AmbiguousBackend;
    if (uri.scheme == .hub or uri.scheme == .local) return .Local;
    return error.AmbiguousBackend;
}

fn hasOpenAiScheme(ref: []const u8) bool {
    return std.mem.startsWith(u8, ref, "openai://") or std.mem.startsWith(u8, ref, "oaic://");
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn isValidBaseUrl(url: []const u8) bool {
    if (url.len == 0) return false;
    if (std.mem.startsWith(u8, url, "http://") or std.mem.startsWith(u8, url, "https://")) {
        return std.mem.indexOf(u8, url, "://") != null;
    }
    return false;
}

fn localConfigInBounds(struct_size: u32) bool {
    const base = @offsetOf(TaluModelSpec, "backend_config");
    return @as(usize, struct_size) >= base + @sizeOf(LocalConfig);
}

fn openAiConfigInBounds(struct_size: u32) bool {
    const base = @offsetOf(TaluModelSpec, "backend_config");
    return @as(usize, struct_size) >= base + @sizeOf(OpenAICompatibleConfig);
}

fn dupeOptionalZ(allocator: std.mem.Allocator, value: ?[*:0]const u8) !?[:0]u8 {
    if (value) |ptr| {
        const slice = std.mem.sliceTo(ptr, 0);
        return try allocator.dupeZ(u8, slice);
    }
    return null;
}

fn dupeOptionalSliceZ(allocator: std.mem.Allocator, value: ?[:0]u8) !?[:0]u8 {
    if (value) |slice| {
        return try allocator.dupeZ(u8, slice[0..slice.len]);
    }
    return null;
}

fn resolveBaseUrl(allocator: std.mem.Allocator, value: ?[*:0]const u8) !?[:0]u8 {
    if (value) |ptr| {
        const slice = std.mem.sliceTo(ptr, 0);
        return try allocator.dupeZ(u8, slice);
    }
    return try allocator.dupeZ(u8, DEFAULT_OPENAI_BASE_URL);
}

fn resolveApiKey(allocator: std.mem.Allocator, value: ?[*:0]const u8) !?[:0]u8 {
    if (value) |ptr| {
        const slice = std.mem.sliceTo(ptr, 0);
        return try allocator.dupeZ(u8, slice);
    }

    const env = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(env);
    return try allocator.dupeZ(u8, env);
}

// =============================================================================
// Tests
// =============================================================================

test "validateSpec rejects truncated struct_size for local backend" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "model.bin", .data = "ok" });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "model.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);

    var spec = std.mem.zeroes(TaluModelSpec);
    spec.abi_version = 1;
    spec.struct_size = MIN_HEADER_SIZE;
    spec.ref = path_z.ptr;
    spec.backend_type_raw = 0;

    try std.testing.expectEqual(false, validateSpec(&spec));
}

test "validateSpec accepts valid local backend" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(.{ .sub_path = "model.bin", .data = "ok" });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "model.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);

    var spec = std.mem.zeroes(TaluModelSpec);
    spec.abi_version = 1;
    spec.struct_size = @sizeOf(TaluModelSpec);
    spec.ref = path_z.ptr;
    spec.backend_type_raw = 0; // Local

    try std.testing.expectEqual(true, validateSpec(&spec));
}

test "validateSpec rejects missing model path" {
    const path_z = try std.testing.allocator.dupeZ(u8, "/nonexistent/path/to/model");
    defer std.testing.allocator.free(path_z);

    var spec = std.mem.zeroes(TaluModelSpec);
    spec.abi_version = 1;
    spec.struct_size = @sizeOf(TaluModelSpec);
    spec.ref = path_z.ptr;
    spec.backend_type_raw = 0; // Local

    const result = validateSpecDetailed(&spec);
    try std.testing.expectEqual(false, result.valid);
    try std.testing.expectEqual(ValidationIssue.model_not_found, result.issue);
}

test "parseBackendType handles all values" {
    try std.testing.expectEqual(BackendType.Unspecified, parseBackendType(-1).?);
    try std.testing.expectEqual(BackendType.Local, parseBackendType(0).?);
    try std.testing.expectEqual(BackendType.OpenAICompatible, parseBackendType(1).?);
    try std.testing.expectEqual(@as(?BackendType, null), parseBackendType(99));
}
