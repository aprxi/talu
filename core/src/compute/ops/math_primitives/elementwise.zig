//! Binary element-wise arithmetic operations: add, sub, mul, div.

const std = @import("std");
const simd = @import("../../simd/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Element-wise addition: out = a + b
pub fn addContiguous(out: []f32, a: []const f32, b: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);

    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < a.len) : (vec_idx += VEC_LEN) {
        const a_vec: F32Vec = a[vec_idx..][0..VEC_LEN].*;
        const b_vec: F32Vec = b[vec_idx..][0..VEC_LEN].*;
        out[vec_idx..][0..VEC_LEN].* = a_vec + b_vec;
    }
    while (vec_idx < a.len) : (vec_idx += 1) {
        out[vec_idx] = a[vec_idx] + b[vec_idx];
    }
}

/// Element-wise subtraction: out = a - b
pub fn subContiguous(out: []f32, a: []const f32, b: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);

    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < a.len) : (vec_idx += VEC_LEN) {
        const a_vec: F32Vec = a[vec_idx..][0..VEC_LEN].*;
        const b_vec: F32Vec = b[vec_idx..][0..VEC_LEN].*;
        out[vec_idx..][0..VEC_LEN].* = a_vec - b_vec;
    }
    while (vec_idx < a.len) : (vec_idx += 1) {
        out[vec_idx] = a[vec_idx] - b[vec_idx];
    }
}

/// Element-wise multiplication: out = a * b
pub fn mulContiguous(out: []f32, a: []const f32, b: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);

    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < a.len) : (vec_idx += VEC_LEN) {
        const a_vec: F32Vec = a[vec_idx..][0..VEC_LEN].*;
        const b_vec: F32Vec = b[vec_idx..][0..VEC_LEN].*;
        out[vec_idx..][0..VEC_LEN].* = a_vec * b_vec;
    }
    while (vec_idx < a.len) : (vec_idx += 1) {
        out[vec_idx] = a[vec_idx] * b[vec_idx];
    }
}

/// Element-wise division: out = a / b
pub fn divContiguous(out: []f32, a: []const f32, b: []const f32) void {
    @setFloatMode(.optimized);
    std.debug.assert(out.len == a.len);
    std.debug.assert(a.len == b.len);

    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < a.len) : (vec_idx += VEC_LEN) {
        const a_vec: F32Vec = a[vec_idx..][0..VEC_LEN].*;
        const b_vec: F32Vec = b[vec_idx..][0..VEC_LEN].*;
        out[vec_idx..][0..VEC_LEN].* = a_vec / b_vec;
    }
    while (vec_idx < a.len) : (vec_idx += 1) {
        out[vec_idx] = a[vec_idx] / b[vec_idx];
    }
}

test "addContiguous - basic correctness" {
    const allocator = std.testing.allocator;

    // Test basic addition with known values
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const b = [_]f32{ 0.5, 1.5, 2.5, 3.5, 4.5 };
    const expected = [_]f32{ 1.5, 3.5, 5.5, 7.5, 9.5 };

    const out = try allocator.alloc(f32, a.len);
    defer allocator.free(out);

    addContiguous(out, &a, &b);

    for (out, expected) |result, exp| {
        try std.testing.expectApproxEqRel(exp, result, 1e-6);
    }
}

test "addContiguous - edge cases" {
    const allocator = std.testing.allocator;

    // Test with zeros
    {
        const a = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        addContiguous(out, &a, &b);

        for (out, b) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with negative values
    {
        const a = [_]f32{ -1.0, -2.0, -3.0, -4.0 };
        const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const expected = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        addContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with large values
    {
        const a = [_]f32{ 1e6, 1e7, 1e8 };
        const b = [_]f32{ 1e6, 1e7, 1e8 };
        const expected = [_]f32{ 2e6, 2e7, 2e8 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        addContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-5);
        }
    }
}

test "addContiguous - SIMD and scalar paths" {
    const allocator = std.testing.allocator;

    // Test size that exercises both SIMD vector path and scalar remainder
    const size = VEC_LEN * 3 + 2; // 3 full vectors + 2 scalar remainder
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);

    // Fill with sequential values
    for (0..size) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @as(f32, @floatFromInt(i)) * 0.5;
    }

    addContiguous(out, a, b);

    // Verify all elements
    for (0..size) |i| {
        const expected = a[i] + b[i];
        try std.testing.expectApproxEqRel(expected, out[i], 1e-6);
    }
}

test "subContiguous - basic correctness" {
    const allocator = std.testing.allocator;

    // Test basic subtraction with known values
    const a = [_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0 };
    const b = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0 };
    const expected = [_]f32{ 4.0, 3.0, 2.0, 1.0, 0.0 };

    const out = try allocator.alloc(f32, a.len);
    defer allocator.free(out);

    subContiguous(out, &a, &b);

    for (out, expected) |result, exp| {
        try std.testing.expectApproxEqRel(exp, result, 1e-6);
    }
}

test "subContiguous - edge cases" {
    const allocator = std.testing.allocator;

    // Test with zeros
    {
        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        subContiguous(out, &a, &b);

        for (out, a) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with negative results
    {
        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
        const expected = [_]f32{ -1.0, -1.0, -1.0, -1.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        subContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test subtraction of same values (should be zero)
    {
        const a = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        subContiguous(out, &a, &a);

        for (out) |result| {
            try std.testing.expectApproxEqRel(0.0, result, 1e-6);
        }
    }
}

test "subContiguous - SIMD and scalar paths" {
    const allocator = std.testing.allocator;

    // Test size that exercises both SIMD vector path and scalar remainder
    const size = VEC_LEN * 4 + 3; // 4 full vectors + 3 scalar remainder
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);

    // Fill with sequential values
    for (0..size) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 2.0;
        b[i] = @floatFromInt(i);
    }

    subContiguous(out, a, b);

    // Verify all elements
    for (0..size) |i| {
        const expected = a[i] - b[i];
        try std.testing.expectApproxEqRel(expected, out[i], 1e-6);
    }
}

test "mulContiguous - basic correctness" {
    const allocator = std.testing.allocator;

    // Test basic multiplication with known values
    const a = [_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const expected = [_]f32{ 2.0, 6.0, 12.0, 20.0, 30.0 };

    const out = try allocator.alloc(f32, a.len);
    defer allocator.free(out);

    mulContiguous(out, &a, &b);

    for (out, expected) |result, exp| {
        try std.testing.expectApproxEqRel(exp, result, 1e-6);
    }
}

test "mulContiguous - edge cases" {
    const allocator = std.testing.allocator;

    // Test with zeros (should produce zeros)
    {
        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const expected = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        mulContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with negative values
    {
        const a = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
        const b = [_]f32{ 2.0, -2.0, 3.0, -4.0 };
        const expected = [_]f32{ -2.0, -4.0, -9.0, -16.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        mulContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with fractional values
    {
        const a = [_]f32{ 0.5, 0.25, 0.1, 0.01 };
        const b = [_]f32{ 2.0, 4.0, 10.0, 100.0 };
        const expected = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        mulContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-5);
        }
    }
}

test "mulContiguous - SIMD and scalar paths" {
    const allocator = std.testing.allocator;

    // Test size that exercises both SIMD vector path and scalar remainder
    const size = VEC_LEN * 2 + 5; // 2 full vectors + 5 scalar remainder
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);

    // Fill with sequential values
    for (0..size) |i| {
        a[i] = @as(f32, @floatFromInt(i + 1)) * 0.5;
        b[i] = @as(f32, @floatFromInt(i + 1)) * 2.0;
    }

    mulContiguous(out, a, b);

    // Verify all elements
    for (0..size) |i| {
        const expected = a[i] * b[i];
        try std.testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "divContiguous - basic correctness" {
    const allocator = std.testing.allocator;

    // Test basic division with known values
    const a = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const b = [_]f32{ 2.0, 4.0, 5.0, 8.0, 10.0 };
    const expected = [_]f32{ 5.0, 5.0, 6.0, 5.0, 5.0 };

    const out = try allocator.alloc(f32, a.len);
    defer allocator.free(out);

    divContiguous(out, &a, &b);

    for (out, expected) |result, exp| {
        try std.testing.expectApproxEqRel(exp, result, 1e-6);
    }
}

test "divContiguous - edge cases" {
    const allocator = std.testing.allocator;

    // Test division by 1 (should be identity)
    {
        const a = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
        const b = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        divContiguous(out, &a, &b);

        for (out, a) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test zero divided by non-zero
    {
        const a = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const expected = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        divContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test with negative values
    {
        const a = [_]f32{ -10.0, 20.0, -30.0, 40.0 };
        const b = [_]f32{ 2.0, -4.0, 5.0, -8.0 };
        const expected = [_]f32{ -5.0, -5.0, -6.0, -5.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        divContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-6);
        }
    }

    // Test division by small numbers (numerical stability)
    {
        const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const b = [_]f32{ 0.001, 0.002, 0.003, 0.004 };
        const expected = [_]f32{ 1000.0, 1000.0, 1000.0, 1000.0 };
        const out = try allocator.alloc(f32, a.len);
        defer allocator.free(out);

        divContiguous(out, &a, &b);

        for (out, expected) |result, exp| {
            try std.testing.expectApproxEqRel(exp, result, 1e-5);
        }
    }
}

test "divContiguous - SIMD and scalar paths" {
    const allocator = std.testing.allocator;

    // Test size that exercises both SIMD vector path and scalar remainder
    const size = VEC_LEN * 5 + 1; // 5 full vectors + 1 scalar remainder
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);

    // Fill with sequential values (avoid division by zero)
    for (0..size) |i| {
        a[i] = @as(f32, @floatFromInt(i + 1)) * 10.0;
        b[i] = @as(f32, @floatFromInt(i + 1)) * 2.0;
    }

    divContiguous(out, a, b);

    // Verify all elements
    for (0..size) |i| {
        const expected = a[i] / b[i];
        try std.testing.expectApproxEqRel(expected, out[i], 1e-5);
    }
}

test "addContiguous mulContiguous commutativity" {
    const allocator = std.testing.allocator;

    const a = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const b = [_]f32{ 0.5, 1.0, 1.5, 2.0 };
    const out1 = try allocator.alloc(f32, a.len);
    defer allocator.free(out1);
    const out2 = try allocator.alloc(f32, a.len);
    defer allocator.free(out2);

    // Test addition commutativity: a + b == b + a
    addContiguous(out1, &a, &b);
    addContiguous(out2, &b, &a);
    for (out1, out2) |v1, v2| {
        try std.testing.expectApproxEqRel(v1, v2, 1e-6);
    }

    // Test multiplication commutativity: a * b == b * a
    mulContiguous(out1, &a, &b);
    mulContiguous(out2, &b, &a);
    for (out1, out2) |v1, v2| {
        try std.testing.expectApproxEqRel(v1, v2, 1e-6);
    }
}

test "addContiguous subContiguous mulContiguous divContiguous small array" {
    const allocator = std.testing.allocator;

    // Use size smaller than VEC_LEN to test scalar-only path
    const size = @min(VEC_LEN - 1, 3);
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const out = try allocator.alloc(f32, size);
    defer allocator.free(out);

    for (0..size) |i| {
        a[i] = @as(f32, @floatFromInt(i + 1)) * 2.0;
        b[i] = @floatFromInt(i + 1);
    }

    // Test all operations work correctly on small arrays
    addContiguous(out, a, b);
    for (0..size) |i| {
        try std.testing.expectApproxEqRel(a[i] + b[i], out[i], 1e-6);
    }

    subContiguous(out, a, b);
    for (0..size) |i| {
        try std.testing.expectApproxEqRel(a[i] - b[i], out[i], 1e-6);
    }

    mulContiguous(out, a, b);
    for (0..size) |i| {
        try std.testing.expectApproxEqRel(a[i] * b[i], out[i], 1e-6);
    }

    divContiguous(out, a, b);
    for (0..size) |i| {
        try std.testing.expectApproxEqRel(a[i] / b[i], out[i], 1e-6);
    }
}
