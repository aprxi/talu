//! Integration tests for InferenceBackend
//!
//! InferenceBackend is the backend that performs inference, created from a CanonicalSpec.
//! It wraps LocalEngine for native inference or API configuration for remote inference.
//!
//! These tests verify the type structure, methods, and creation behavior.

const std = @import("std");
const main = @import("main");
const spec = main.router.spec;
const InferenceBackend = spec.InferenceBackend;
const CanonicalSpec = spec.CanonicalSpec;
const TaluModelSpec = spec.TaluModelSpec;
const BackendType = spec.BackendType;
const LocalEngine = main.router.LocalEngine;

// =============================================================================
// Type Export Tests
// =============================================================================

test "InferenceBackend is exported from router" {
    const type_info = @typeInfo(InferenceBackend);
    try std.testing.expect(type_info == .@"struct");
}

test "InferenceBackend has expected fields" {
    const fields = @typeInfo(InferenceBackend).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 3), fields.len);

    var has_backend_type = false;
    var has_backend = false;
    var has_model_ref = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "backend_type")) has_backend_type = true;
        if (comptime std.mem.eql(u8, field.name, "backend")) has_backend = true;
        if (comptime std.mem.eql(u8, field.name, "model_ref")) has_model_ref = true;
    }

    try std.testing.expect(has_backend_type);
    try std.testing.expect(has_backend);
    try std.testing.expect(has_model_ref);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "InferenceBackend has deinit method" {
    try std.testing.expect(@hasDecl(InferenceBackend, "deinit"));
}

test "InferenceBackend has getLocalEngine method" {
    try std.testing.expect(@hasDecl(InferenceBackend, "getLocalEngine"));
}

test "InferenceBackend.getLocalEngine returns optional LocalEngine pointer" {
    const fn_info = @typeInfo(@TypeOf(InferenceBackend.getLocalEngine)).@"fn";
    const ReturnType = fn_info.return_type.?;

    // Should be ?*LocalEngine
    const optional_info = @typeInfo(ReturnType);
    try std.testing.expect(optional_info == .optional);
}

// =============================================================================
// Creation Function Tests
// =============================================================================

test "createInferenceBackend is exported" {
    try std.testing.expect(@hasDecl(spec, "createInferenceBackend"));
}

test "createInferenceBackend succeeds for valid OpenAI spec" {
    const ref = "openai://gpt-4";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.OpenAICompatible),
        .backend_config = .{
            .openai_compat = .{
                .base_url = null,
                .api_key = "test-key",
                .org_id = null,
                .timeout_ms = 30000,
                .max_retries = 3,
                .custom_headers_json = null,
                ._reserved = [_]u8{0} ** 24,
            },
        },
    };

    // First canonicalize
    var canonical = try spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    defer canonical.deinit(std.testing.allocator);

    // Then create backend
    var backend = try spec.createInferenceBackend(std.testing.allocator, &canonical);
    defer backend.deinit(std.testing.allocator);

    try std.testing.expectEqual(BackendType.OpenAICompatible, backend.backend_type);
}

test "createInferenceBackend OpenAI backend has no LocalEngine" {
    const ref = "openai://gpt-4";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.OpenAICompatible),
        .backend_config = .{
            .openai_compat = .{
                .base_url = null,
                .api_key = "test-key",
                .org_id = null,
                .timeout_ms = 30000,
                .max_retries = 3,
                .custom_headers_json = null,
                ._reserved = [_]u8{0} ** 24,
            },
        },
    };

    var canonical = try spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    defer canonical.deinit(std.testing.allocator);

    var backend = try spec.createInferenceBackend(std.testing.allocator, &canonical);
    defer backend.deinit(std.testing.allocator);

    // OpenAI backend should not have a LocalEngine
    try std.testing.expect(backend.getLocalEngine() == null);
}

test "createInferenceBackend fails for Unspecified backend type" {
    // Create a canonical spec directly with Unspecified type
    // This shouldn't happen in practice but tests error handling
    var canonical = CanonicalSpec{
        .backend_type = .Unspecified,
        .ref = try std.testing.allocator.allocSentinel(u8, 4, 0),
        .backend = .{ .Unspecified = {} },
    };
    @memcpy(canonical.ref[0..4], "test");
    defer std.testing.allocator.free(canonical.ref);

    const result = spec.createInferenceBackend(std.testing.allocator, &canonical);
    try std.testing.expectError(error.InvalidArgument, result);
}

// =============================================================================
// Backend Type Relationship Tests
// =============================================================================

test "InferenceBackend backend_type matches canonical backend_type" {
    const ref = "openai://gpt-4";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.OpenAICompatible),
        .backend_config = .{
            .openai_compat = .{
                .base_url = null,
                .api_key = "test-key",
                .org_id = null,
                .timeout_ms = 30000,
                .max_retries = 3,
                .custom_headers_json = null,
                ._reserved = [_]u8{0} ** 24,
            },
        },
    };

    var canonical = try spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    defer canonical.deinit(std.testing.allocator);

    var backend = try spec.createInferenceBackend(std.testing.allocator, &canonical);
    defer backend.deinit(std.testing.allocator);

    try std.testing.expectEqual(canonical.backend_type, backend.backend_type);
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

test "InferenceBackend deinit is safe to call" {
    const ref = "openai://gpt-4";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.OpenAICompatible),
        .backend_config = .{
            .openai_compat = .{
                .base_url = "https://api.example.com/v1",
                .api_key = "test-key",
                .org_id = "org-123",
                .timeout_ms = 30000,
                .max_retries = 3,
                .custom_headers_json = null,
                ._reserved = [_]u8{0} ** 24,
            },
        },
    };

    var canonical = try spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    defer canonical.deinit(std.testing.allocator);

    var backend = try spec.createInferenceBackend(std.testing.allocator, &canonical);

    // deinit should free all owned strings without crash
    backend.deinit(std.testing.allocator);
}

// =============================================================================
// Local Backend Tests (require model file)
// =============================================================================

test "createInferenceBackend fails for missing local model" {
    const ref = "/nonexistent/path/to/model";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.Local),
        .backend_config = .{
            .local = .{
                .gpu_layers = -1,
                .use_mmap = 1,
                .num_threads = 0,
                ._reserved = [_]u8{0} ** 32,
            },
        },
    };

    // Canonicalization should fail for missing local model
    const result = spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    try std.testing.expectError(error.ModelNotFound, result);
}
