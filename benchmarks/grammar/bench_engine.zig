//! Isolated grammar engine benchmarks - no model inference required.
//!
//! This benchmark tests the low-level grammar engine operations directly,
//! enabling fast iteration on performance optimization without waiting
//! for model loading.
//!
//! Run with: zig build-exe benchmarks/grammar/bench_engine.zig -lc && ./bench_engine

const std = @import("std");
const capi = @import("../../src/capi/root.zig");

const talu_grammar_engine_create = capi.talu_grammar_engine_create;
const talu_grammar_engine_destroy = capi.talu_grammar_engine_destroy;
const talu_grammar_engine_reset = capi.talu_grammar_engine_reset;
const talu_grammar_engine_is_complete = capi.talu_grammar_engine_is_complete;
const talu_grammar_engine_get_valid_bytes = capi.talu_grammar_engine_get_valid_bytes;
const talu_grammar_engine_count_valid_bytes = capi.talu_grammar_engine_count_valid_bytes;
const talu_grammar_engine_can_accept = capi.talu_grammar_engine_can_accept;
const talu_grammar_engine_advance = capi.talu_grammar_engine_advance;
const talu_grammar_engine_validate = capi.talu_grammar_engine_validate;
const talu_token_mask_create = capi.talu_token_mask_create;
const talu_token_mask_destroy = capi.talu_token_mask_destroy;

// Test schemas
const simple_schema =
    \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
;

const complex_schema =
    \\{"type":"object","properties":{"users":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"}},"required":["name"]}},"metadata":{"type":"object","properties":{"count":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}}}}}
;

fn benchmarkEngineCreation(allocator: std.mem.Allocator) !void {
    _ = allocator;
    const iterations: usize = 1000;

    std.debug.print("\n=== Engine Creation Benchmark ===\n", .{});

    // Simple schema
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const handle = talu_grammar_engine_create(simple_schema) orelse unreachable;
            talu_grammar_engine_destroy(handle);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
        std.debug.print("Simple schema creation: {d:.2} µs/iter\n", .{per_iter_us});
    }

    // Complex schema
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const handle = talu_grammar_engine_create(complex_schema) orelse unreachable;
            talu_grammar_engine_destroy(handle);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
        std.debug.print("Complex schema creation: {d:.2} µs/iter\n", .{per_iter_us});
    }
}

fn benchmarkGetValidBytes() !void {
    const iterations: usize = 100_000;

    std.debug.print("\n=== Get Valid Bytes Benchmark ===\n", .{});

    const handle = talu_grammar_engine_create(simple_schema) orelse return error.CreateFailed;
    defer talu_grammar_engine_destroy(handle);

    var valid: [256]bool = undefined;

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = talu_grammar_engine_get_valid_bytes(handle, &valid);
    }
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const per_iter_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
    std.debug.print("getValidBytes: {d:.0} ns/iter\n", .{per_iter_ns});

    // Also count valid bytes
    const count = talu_grammar_engine_count_valid_bytes(handle);
    std.debug.print("Valid bytes at initial state: {d}\n", .{count});
}

fn benchmarkCanAccept() !void {
    const iterations: usize = 100_000;

    std.debug.print("\n=== Can Accept Benchmark ===\n", .{});

    const handle = talu_grammar_engine_create(simple_schema) orelse return error.CreateFailed;
    defer talu_grammar_engine_destroy(handle);

    // Single byte
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            _ = talu_grammar_engine_can_accept(handle, "{", 1);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        std.debug.print("canAccept (1 byte): {d:.0} ns/iter\n", .{per_iter_ns});
    }

    // Short string (key name)
    {
        const test_str = "\"name\"";
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            // First advance to get to key position
            _ = talu_grammar_engine_reset(handle);
            _ = talu_grammar_engine_advance(handle, "{", 1);
            _ = talu_grammar_engine_can_accept(handle, test_str, test_str.len);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        std.debug.print("canAccept (6 bytes): {d:.0} ns/iter\n", .{per_iter_ns});
    }
}

fn benchmarkAdvanceReset() !void {
    const iterations: usize = 10_000;

    std.debug.print("\n=== Advance + Reset Benchmark ===\n", .{});

    const handle = talu_grammar_engine_create(simple_schema) orelse return error.CreateFailed;
    defer talu_grammar_engine_destroy(handle);

    const test_json = "{\"name\":\"test\",\"age\":25}";

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        // Advance byte by byte
        for (test_json) |byte| {
            _ = talu_grammar_engine_advance(handle, &[_]u8{byte}, 1);
        }
        _ = talu_grammar_engine_reset(handle);
    }
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
    std.debug.print("advance full JSON ({d} bytes) + reset: {d:.2} µs/iter\n", .{ test_json.len, per_iter_us });

    // Also test bulk advance
    {
        const start2 = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            _ = talu_grammar_engine_advance(handle, test_json, test_json.len);
            _ = talu_grammar_engine_reset(handle);
        }
        const elapsed_ns2 = std.time.nanoTimestamp() - start2;
        const per_iter_us2 = @as(f64, @floatFromInt(elapsed_ns2)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
        std.debug.print("advance bulk ({d} bytes) + reset: {d:.2} µs/iter\n", .{ test_json.len, per_iter_us2 });
    }
}

fn benchmarkValidation() !void {
    const iterations: usize = 10_000;

    std.debug.print("\n=== Validation Benchmark ===\n", .{});

    const handle = talu_grammar_engine_create(simple_schema) orelse return error.CreateFailed;
    defer talu_grammar_engine_destroy(handle);

    const valid_json = "{\"name\":\"Alice\",\"age\":30}";
    const invalid_json = "{\"name\":123,\"age\":30}"; // name should be string

    // Valid JSON
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const result = talu_grammar_engine_validate(handle, valid_json, valid_json.len);
            std.debug.assert(result == 1);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
        std.debug.print("validate valid JSON ({d} bytes): {d:.2} µs/iter\n", .{ valid_json.len, per_iter_us });
    }

    // Invalid JSON (should fail fast)
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const result = talu_grammar_engine_validate(handle, invalid_json, invalid_json.len);
            std.debug.assert(result == 0);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
        std.debug.print("validate invalid JSON ({d} bytes): {d:.2} µs/iter\n", .{ invalid_json.len, per_iter_us });
    }
}

fn benchmarkTokenMaskCreation() !void {
    const iterations: usize = 10_000;

    std.debug.print("\n=== Token Mask Creation Benchmark ===\n", .{});

    // Small vocab (1k)
    {
        const vocab_size: usize = 1000;
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const mask = talu_token_mask_create(vocab_size) orelse unreachable;
            talu_token_mask_destroy(mask);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        std.debug.print("mask create/destroy (1k tokens): {d:.0} ns/iter\n", .{per_iter_ns});
    }

    // Large vocab (150k - typical LLM)
    {
        const vocab_size: usize = 150_000;
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const mask = talu_token_mask_create(vocab_size) orelse unreachable;
            talu_token_mask_destroy(mask);
        }
        const elapsed_ns = std.time.nanoTimestamp() - start;
        const per_iter_ns = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations));
        std.debug.print("mask create/destroy (150k tokens): {d:.0} ns/iter\n", .{per_iter_ns});
    }
}

fn benchmarkMockTokenMask() !void {
    const iterations: usize = 1000;

    std.debug.print("\n=== Mock Token Mask Computation Benchmark ===\n", .{});

    const handle = talu_grammar_engine_create(simple_schema) orelse return error.CreateFailed;
    defer talu_grammar_engine_destroy(handle);

    const vocab_size: usize = 1000;
    const mask = talu_token_mask_create(vocab_size) orelse return error.CreateFailed;
    defer talu_token_mask_destroy(mask);

    // Mock token bytes lookup
    const MockTokens = struct {
        const tokens = [_][]const u8{
            "{",
            "}",
            "\"",
            ":",
            ",",
            "\"name\"",
            "\"age\"",
            "\"test\"",
            "123",
            "true",
        };

        fn callback(token_id: u32, out_len: *usize, _: ?*anyopaque) callconv(.c) ?[*]const u8 {
            if (token_id < tokens.len) {
                out_len.* = tokens[token_id].len;
                return tokens[token_id].ptr;
            }
            out_len.* = 0;
            return null;
        }
    };

    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = capi.talu_grammar_engine_get_valid_tokens(handle, vocab_size, mask, MockTokens.callback, null);
        _ = talu_grammar_engine_reset(handle);
    }
    const elapsed_ns = std.time.nanoTimestamp() - start;
    const per_iter_us = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(iterations)) / 1000.0;
    std.debug.print("getValidTokens (mock 1k vocab): {d:.2} µs/iter\n", .{per_iter_us});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Grammar Engine Benchmarks ===\n", .{});
    std.debug.print("(No model loading - isolated grammar operations)\n", .{});

    try benchmarkEngineCreation(allocator);
    try benchmarkGetValidBytes();
    try benchmarkCanAccept();
    try benchmarkAdvanceReset();
    try benchmarkValidation();
    try benchmarkTokenMaskCreation();
    try benchmarkMockTokenMask();

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

test "benchmark can run" {
    // Just verify the benchmarks compile and run without crashing
    const handle = talu_grammar_engine_create(simple_schema);
    try std.testing.expect(handle != null);
    talu_grammar_engine_destroy(handle);
}
