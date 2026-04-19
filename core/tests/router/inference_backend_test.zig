//! Integration tests for InferenceBackend
//!
//! InferenceBackend is the backend that performs inference, created from a CanonicalSpec.
//! talu now routes local inference only.
//!
//! These tests verify the type structure, methods, and creation behavior.

const std = @import("std");
const main = @import("main");
const spec = main.router.spec;
const InferenceBackend = spec.InferenceBackend;
const CanonicalSpec = spec.CanonicalSpec;
const TaluModelSpec = spec.TaluModelSpec;
const BackendType = spec.BackendType;

// =============================================================================
// Type Export Tests
// =============================================================================

test "InferenceBackend is exported from router" {
    const type_info = @typeInfo(InferenceBackend);
    try std.testing.expect(type_info == .@"struct");
}

test "InferenceBackend has expected fields" {
    const fields = @typeInfo(InferenceBackend).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 2), fields.len);

    var has_backend_type = false;
    var has_backend = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "backend_type")) has_backend_type = true;
        if (comptime std.mem.eql(u8, field.name, "backend")) has_backend = true;
    }

    try std.testing.expect(has_backend_type);
    try std.testing.expect(has_backend);
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

test "getLocalEngine returns null for Unspecified backend union variant" {
    var backend = InferenceBackend{
        .backend_type = .Unspecified,
        .backend = .{ .Unspecified = {} },
    };
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

    const result = spec.createInferenceBackend(std.testing.allocator, &canonical, main.capi.progress.ProgressContext.NONE);
    try std.testing.expectError(error.InvalidArgument, result);
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
