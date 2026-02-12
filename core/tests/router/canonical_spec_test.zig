//! Integration tests for CanonicalSpec
//!
//! CanonicalSpec is a validated, resolved model specification created from TaluModelSpec.
//! It owns all memory and provides a canonical representation for engine creation.
//!
//! These tests verify the type structure, methods, and validation behavior.

const std = @import("std");
const main = @import("main");
const spec = main.router.spec;
const CanonicalSpec = spec.CanonicalSpec;
const TaluModelSpec = spec.TaluModelSpec;
const BackendType = spec.BackendType;
const ValidationIssue = spec.ValidationIssue;

// =============================================================================
// Type Export Tests
// =============================================================================

test "CanonicalSpec is exported from router" {
    const type_info = @typeInfo(CanonicalSpec);
    try std.testing.expect(type_info == .@"struct");
}

test "CanonicalSpec has expected fields" {
    const fields = @typeInfo(CanonicalSpec).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 3), fields.len);

    var has_backend_type = false;
    var has_ref = false;
    var has_backend = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "backend_type")) has_backend_type = true;
        if (comptime std.mem.eql(u8, field.name, "ref")) has_ref = true;
        if (comptime std.mem.eql(u8, field.name, "backend")) has_backend = true;
    }

    try std.testing.expect(has_backend_type);
    try std.testing.expect(has_ref);
    try std.testing.expect(has_backend);
}

// =============================================================================
// Method Existence Tests
// =============================================================================

test "CanonicalSpec has deinit method" {
    try std.testing.expect(@hasDecl(CanonicalSpec, "deinit"));
}

// =============================================================================
// Validation Function Tests
// =============================================================================

test "validateSpec returns false for null ref" {
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = null,
        .backend_type_raw = @intFromEnum(BackendType.Local),
        .backend_config = undefined,
    };

    const result = spec.validateSpec(&c_spec);
    try std.testing.expect(!result);
}

test "validateSpecDetailed identifies null ref" {
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = null,
        .backend_type_raw = @intFromEnum(BackendType.Local),
        .backend_config = undefined,
    };

    const result = spec.validateSpecDetailed(&c_spec);
    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(ValidationIssue.ref_null, result.issue);
}

test "validateSpecDetailed identifies bad ABI version" {
    const ref = "test-model";
    var c_spec = TaluModelSpec{
        .abi_version = 99, // Invalid
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.Local),
        .backend_config = undefined,
    };

    const result = spec.validateSpecDetailed(&c_spec);
    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(ValidationIssue.bad_abi, result.issue);
}

test "validateSpecDetailed identifies struct too small" {
    const ref = "test-model";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = 4, // Too small
        .ref = ref,
        .backend_type_raw = @intFromEnum(BackendType.Local),
        .backend_config = undefined,
    };

    const result = spec.validateSpecDetailed(&c_spec);
    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(ValidationIssue.struct_too_small, result.issue);
}

test "validateSpecDetailed identifies invalid backend type" {
    const ref = "test-model";
    var c_spec = TaluModelSpec{
        .abi_version = 1,
        .struct_size = @sizeOf(TaluModelSpec),
        .ref = ref,
        .backend_type_raw = 999, // Invalid
        .backend_config = undefined,
    };

    const result = spec.validateSpecDetailed(&c_spec);
    try std.testing.expect(!result.valid);
    try std.testing.expectEqual(ValidationIssue.invalid_backend_type, result.issue);
}

test "validateSpecDetailed accepts valid OpenAI spec" {
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

    const result = spec.validateSpecDetailed(&c_spec);
    try std.testing.expect(result.valid);
    try std.testing.expectEqual(ValidationIssue.none, result.issue);
}

// =============================================================================
// Canonicalization Tests
// =============================================================================

test "canonicalizeSpec succeeds for valid OpenAI spec" {
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

    try std.testing.expectEqual(BackendType.OpenAICompatible, canonical.backend_type);
    try std.testing.expectEqualStrings("openai://gpt-4", canonical.ref);
}

test "canonicalizeSpec owns its memory" {
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

    // Verify ref is a copy, not a pointer to original
    try std.testing.expect(canonical.ref.ptr != ref.ptr);
}

test "canonicalizeSpec fails for missing local model" {
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

    const result = spec.canonicalizeSpec(std.testing.allocator, &c_spec);
    try std.testing.expectError(error.ModelNotFound, result);
}

// =============================================================================
// Helper Function Tests
// =============================================================================

test "validationIssueMessage returns descriptive messages" {
    const msg = spec.validationIssueMessage(.ref_null);
    try std.testing.expect(msg.len > 0);
}

test "parseBackendType returns null for invalid values" {
    try std.testing.expect(spec.parseBackendType(999) == null);
    try std.testing.expect(spec.parseBackendType(-99) == null);
}

test "parseBackendType returns correct types" {
    try std.testing.expectEqual(BackendType.Unspecified, spec.parseBackendType(-1).?);
    try std.testing.expectEqual(BackendType.Local, spec.parseBackendType(0).?);
    try std.testing.expectEqual(BackendType.OpenAICompatible, spec.parseBackendType(1).?);
}

// =============================================================================
// BackendType Tests
// =============================================================================

test "BackendType enum has expected values" {
    try std.testing.expectEqual(@as(i32, -1), @intFromEnum(BackendType.Unspecified));
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(BackendType.Local));
    try std.testing.expectEqual(@as(i32, 1), @intFromEnum(BackendType.OpenAICompatible));
}
