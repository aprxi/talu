//! Integration tests for OwnedTensor
//!
//! OwnedTensor is a stack-allocated owning tensor with SIMD-aligned memory.
//! Uses Zig allocator (not libc) for memory management.

const std = @import("std");
const main = @import("main");
const OwnedTensor = main.OwnedTensor;
const Tensor = main.Tensor;
const DType = main.DType;

// =============================================================================
// Creation Tests
// =============================================================================

test "OwnedTensor.init creates tensor with correct shape" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(i32, 3), tensor.n_dims);
    try std.testing.expectEqual(@as(usize, 2), tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), tensor.shape[1]);
    try std.testing.expectEqual(@as(usize, 4), tensor.shape[2]);
}

test "OwnedTensor.init allocates SIMD-aligned memory" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{ 10, 10 });
    defer tensor.deinit();

    // Check alignment (32 bytes for AVX2)
    const addr = @intFromPtr(tensor.data.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % 32);
}

test "OwnedTensor.init zeros memory" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{100});
    defer tensor.deinit();

    const slice = tensor.asSlice(f32);
    for (slice) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "OwnedTensor.init with different dtypes" {
    const allocator = std.testing.allocator;

    // f32
    {
        var t = try OwnedTensor.init(allocator, .f32, &[_]usize{10});
        defer t.deinit();
        try std.testing.expectEqual(DType.f32, t.dtype);
        try std.testing.expectEqual(@as(usize, 40), t.data_size);
    }

    // f16
    {
        var t = try OwnedTensor.init(allocator, .f16, &[_]usize{10});
        defer t.deinit();
        try std.testing.expectEqual(DType.f16, t.dtype);
        try std.testing.expectEqual(@as(usize, 20), t.data_size);
    }

    // i32
    {
        var t = try OwnedTensor.init(allocator, .i32, &[_]usize{10});
        defer t.deinit();
        try std.testing.expectEqual(DType.i32, t.dtype);
        try std.testing.expectEqual(@as(usize, 40), t.data_size);
    }
}

// =============================================================================
// Data Access Tests
// =============================================================================

test "OwnedTensor.asSlice returns typed slice" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{4});
    defer tensor.deinit();

    const slice = tensor.asSlice(f32);
    try std.testing.expectEqual(@as(usize, 4), slice.len);

    // Write and read back
    slice[0] = 1.0;
    slice[1] = 2.0;
    slice[2] = 3.0;
    slice[3] = 4.0;

    try std.testing.expectEqual(@as(f32, 1.0), slice[0]);
    try std.testing.expectEqual(@as(f32, 4.0), slice[3]);
}

test "OwnedTensor.numElements returns total element count" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(usize, 24), tensor.numElements());
}

// =============================================================================
// Conversion Tests
// =============================================================================

test "OwnedTensor.toTensor creates view" {
    const allocator = std.testing.allocator;

    var owned = try OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3 });
    defer owned.deinit();

    // Write some data
    const owned_slice = owned.asSlice(f32);
    owned_slice[0] = 1.5;
    owned_slice[5] = 2.5;

    // Convert to Tensor view
    const tensor = owned.toTensor();

    try std.testing.expectEqual(@as(i32, 2), tensor.n_dims);
    try std.testing.expectEqual(@as(i64, 2), tensor.shape[0]);
    try std.testing.expectEqual(@as(i64, 3), tensor.shape[1]);
    try std.testing.expect(!tensor.owns_data); // View doesn't own data

    // Data should be accessible
    const tensor_slice = tensor.asSlice(f32);
    try std.testing.expectEqual(@as(f32, 1.5), tensor_slice[0]);
    try std.testing.expectEqual(@as(f32, 2.5), tensor_slice[5]);
}

test "OwnedTensor.view is alias for toTensor" {
    const allocator = std.testing.allocator;

    var owned = try OwnedTensor.init(allocator, .f32, &[_]usize{10});
    defer owned.deinit();

    const view1 = owned.toTensor();
    const view2 = owned.view();

    try std.testing.expectEqual(view1.data_ptr, view2.data_ptr);
    try std.testing.expectEqual(view1.numel, view2.numel);
}

test "OwnedTensor.toTensor preserves dtype" {
    const allocator = std.testing.allocator;

    var owned = try OwnedTensor.init(allocator, .bf16, &[_]usize{10});
    defer owned.deinit();

    const tensor = owned.toTensor();

    try std.testing.expectEqual(DType.bf16, tensor.dtype);
}

test "OwnedTensor.toTensor creates contiguous strides" {
    const allocator = std.testing.allocator;

    var owned = try OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3, 4 });
    defer owned.deinit();

    const tensor = owned.toTensor();

    // Row-major strides: [12, 4, 1]
    try std.testing.expectEqual(@as(i64, 12), tensor.strides[0]);
    try std.testing.expectEqual(@as(i64, 4), tensor.strides[1]);
    try std.testing.expectEqual(@as(i64, 1), tensor.strides[2]);
    try std.testing.expect(tensor.isContiguous());
}

// =============================================================================
// Shape Limit Tests
// =============================================================================

test "OwnedTensor supports up to 4 dimensions" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3, 4, 5 });
    defer tensor.deinit();

    try std.testing.expectEqual(@as(i32, 4), tensor.n_dims);
    try std.testing.expectEqual(@as(usize, 120), tensor.numElements());
}

test "OwnedTensor.init fails for more than 4 dimensions" {
    const allocator = std.testing.allocator;

    const result = OwnedTensor.init(allocator, .f32, &[_]usize{ 2, 3, 4, 5, 6 });
    try std.testing.expectError(error.ShapeTooLarge, result);
}

// =============================================================================
// Memory Management Tests
// =============================================================================

test "OwnedTensor.deinit frees memory" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{ 100, 100 });
    tensor.deinit();

    // If we get here without leak detection failing, memory was freed properly
}

test "OwnedTensor with 1D shape" {
    const allocator = std.testing.allocator;

    var tensor = try OwnedTensor.init(allocator, .f32, &[_]usize{100});
    defer tensor.deinit();

    try std.testing.expectEqual(@as(i32, 1), tensor.n_dims);
    try std.testing.expectEqual(@as(usize, 100), tensor.shape[0]);
    try std.testing.expectEqual(@as(usize, 0), tensor.shape[1]);
}
