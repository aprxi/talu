//! Model Specification - Local backend configuration and lifecycle.
//!
//! This module canonicalizes and validates model specs for talu's local
//! inference backend only.

const std = @import("std");
const capi_types = @import("../capi/types.zig");
const local_mod = @import("local.zig");
const repository_scheme = @import("io_pkg").repository.scheme;
const progress_mod = @import("progress_pkg");

// =============================================================================
// Type Re-exports (from capi/types.zig for ABI compatibility)
// =============================================================================

pub const BackendType = capi_types.BackendType;
pub const LocalConfig = capi_types.LocalConfig;
pub const BackendUnion = capi_types.BackendUnion;
pub const TaluModelSpec = capi_types.TaluModelSpec;
pub const TaluCapabilities = capi_types.TaluCapabilities;

// =============================================================================
// Canonical Specification
// =============================================================================

/// Canonical (validated, resolved) model specification.
pub const CanonicalSpec = struct {
    backend_type: BackendType,
    ref: [:0]u8,
    backend: CanonicalBackend,

    pub fn deinit(self: *CanonicalSpec, allocator: std.mem.Allocator) void {
        allocator.free(self.ref);
    }
};

const CanonicalBackend = union(BackendType) {
    Unspecified: void,
    Local: CanonicalLocal,
};

const CanonicalLocal = struct {
    gpu_layers: c_int,
    use_mmap: u8,
    num_threads: c_int,
};

// =============================================================================
// Inference Backend
// =============================================================================

/// Backend that performs inference, created from a CanonicalSpec.
pub const InferenceBackend = struct {
    backend_type: BackendType,
    backend: BackendImpl,

    pub fn deinit(self: *InferenceBackend, allocator: std.mem.Allocator) void {
        switch (self.backend) {
            .Local => |engine| {
                engine.deinit();
                allocator.destroy(engine);
            },
            .Unspecified => {},
        }
    }

    pub fn synchronize(self: *InferenceBackend) !void {
        switch (self.backend) {
            .Local => |engine| try engine.synchronize(),
            .Unspecified => {},
        }
    }

    /// Get the LocalEngine if this is a local backend.
    pub fn getLocalEngine(self: *InferenceBackend) ?*local_mod.LocalEngine {
        return switch (self.backend) {
            .Local => |engine| engine,
            else => null,
        };
    }
};

const BackendImpl = union(BackendType) {
    Unspecified: void,
    Local: *local_mod.LocalEngine,
};

// =============================================================================
// Constants
// =============================================================================

pub const MIN_HEADER_SIZE: u32 = @offsetOf(TaluModelSpec, "backend_type_raw") + @sizeOf(c_int);

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
    const ref_raw = std.mem.sliceTo(ref_ptr, 0);
    if (ref_raw.len == 0) return error.InvalidArgument;
    if (hasUnsupportedNamespace(ref_raw)) return error.InvalidArgument;

    const parsed_backend = parseBackendType(spec.backend_type_raw) orelse return error.InvalidArgument;
    const backend_type = if (parsed_backend == .Unspecified) try inferBackendType(ref_raw) else parsed_backend;
    if (backend_type != .Local) return error.InvalidArgument;
    if (!localConfigInBounds(spec.struct_size)) return error.InvalidArgument;

    // Only check pathExists for refs that look like local paths.
    // HuggingFace Hub model IDs (org/model-name) are resolved later by
    // LocalEngine.init() via io.repository.resolveModelPath().
    const uri = repository_scheme.parse(ref_raw) catch return error.InvalidArgument;
    if (uri.scheme == .local and !pathExists(ref_raw)) return error.ModelNotFound;

    var num_threads = spec.backend_config.local.num_threads;
    if (num_threads == 0) {
        const cpu_count = std.Thread.getCpuCount() catch 0;
        num_threads = if (cpu_count > 0) @intCast(cpu_count) else 1;
    }

    const ref_copy = try allocator.dupeZ(u8, ref_raw);
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
    progress: progress_mod.Context,
) !InferenceBackend {
    switch (canon.backend_type) {
        .Local => {
            const engine = try allocator.create(local_mod.LocalEngine);
            errdefer allocator.destroy(engine);
            engine.* = try local_mod.LocalEngine.initWithSeedAndResolutionConfig(allocator, canon.ref, 42, .{}, .{}, progress);
            if (std.debug.runtime_safety) {
                std.debug.assert(engine.*.model_path.len == canon.ref.len);
                std.debug.assert(std.mem.eql(u8, engine.*.model_path, canon.ref));
                std.debug.assert(engine.*.model_path.ptr != canon.ref.ptr);
            }
            return InferenceBackend{ .backend_type = .Local, .backend = .{ .Local = engine } };
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
        else => null,
    };
}

fn inferBackendType(ref: []const u8) !BackendType {
    if (hasUnsupportedNamespace(ref)) return error.AmbiguousBackend;

    if (ref.len == 0) return error.AmbiguousBackend;
    if (pathExists(ref)) return .Local;

    // Check if it's a HuggingFace Hub model ID (org/model-name)
    // These resolve to local paths after download/caching.
    const uri = repository_scheme.parse(ref) catch return error.AmbiguousBackend;
    if (uri.scheme == .hub or uri.scheme == .local) return .Local;
    return error.AmbiguousBackend;
}

fn hasUnsupportedNamespace(ref: []const u8) bool {
    return std.mem.indexOf(u8, ref, "::") != null;
}

fn pathExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn localConfigInBounds(struct_size: u32) bool {
    const base = @offsetOf(TaluModelSpec, "backend_config");
    return @as(usize, struct_size) >= base + @sizeOf(LocalConfig);
}

// =============================================================================
// Tests
// =============================================================================

test "parseBackendType handles all values" {
    try std.testing.expectEqual(BackendType.Unspecified, parseBackendType(-1).?);
    try std.testing.expectEqual(BackendType.Local, parseBackendType(0).?);
    try std.testing.expectEqual(@as(?BackendType, null), parseBackendType(1));
    try std.testing.expectEqual(@as(?BackendType, null), parseBackendType(99));
}

test "inferBackendType rejects unsupported namespaces" {
    try std.testing.expectError(error.AmbiguousBackend, inferBackendType("foo::bar"));
    try std.testing.expectError(error.AmbiguousBackend, inferBackendType("native::org/model-name"));
    try std.testing.expectError(error.AmbiguousBackend, inferBackendType("vendor::model"));
}

test "canonicalizeSpec rejects namespaced model identifiers" {
    const ref_z = try std.testing.allocator.dupeZ(u8, "native::org/model-name");
    defer std.testing.allocator.free(ref_z);

    var spec = std.mem.zeroes(TaluModelSpec);
    spec.abi_version = 1;
    spec.struct_size = @sizeOf(TaluModelSpec);
    spec.ref = ref_z.ptr;
    spec.backend_type_raw = 0; // Local

    try std.testing.expectError(error.InvalidArgument, canonicalizeSpec(std.testing.allocator, &spec));
}
