//! Standalone test for SignalGuard SIGBUS recovery.
//!
//! This test cannot run in the Zig test framework because the framework
//! has its own signal handlers. Run via CI script:
//!
//!   ./scripts/check_signal_tests.sh
//!
//! Or manually:
//!   zig build-exe tests/helpers/signal/signal_guard_test.zig src/capi/signal_guard.c -lc
//!   ./signal_guard_test
//!
//! Expected output:
//!   Test 1: Normal operation... PASS
//!   Test 2: SIGBUS recovery... PASS
//!   Test 3: Multiple cycles... PASS
//!   Test 4: Return value propagation... PASS
//!   All tests passed!

const std = @import("std");
const builtin = @import("builtin");

// Import the signal guard C functions (callback-based API)
extern fn talu_signal_guard_install() callconv(.c) void;
extern fn talu_signal_guard_call(
    func: *const fn (?*anyopaque) callconv(.c) c_int,
    ctx: ?*anyopaque,
) callconv(.c) c_int;

const c = @cImport({
    @cInclude("signal.h");
});

// Simple print helper using debug.print (works without buffer setup)
fn print(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

// Test callbacks
fn normalCallback(_: ?*anyopaque) callconv(.c) c_int {
    return 0;
}

fn sigbusCallback(_: ?*anyopaque) callconv(.c) c_int {
    _ = c.raise(c.SIGBUS);
    return 0; // Should not reach here
}

fn returns42(_: ?*anyopaque) callconv(.c) c_int {
    return 42;
}

pub fn main() void {
    if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) {
        print("Signal guard not supported on this platform\n", .{});
        return;
    }

    talu_signal_guard_install();

    // Test 1: Normal operation (no signal)
    print("Test 1: Normal operation... ", .{});
    {
        const result = talu_signal_guard_call(normalCallback, null);
        if (result == 0) {
            print("PASS\n", .{});
        } else {
            print("FAIL (expected 0, got {d})\n", .{result});
            std.process.exit(1);
        }
    }

    // Test 2: SIGBUS recovery
    print("Test 2: SIGBUS recovery... ", .{});
    {
        const result = talu_signal_guard_call(sigbusCallback, null);
        if (result == -1) {
            print("PASS\n", .{});
        } else {
            print("FAIL (expected -1, got {d})\n", .{result});
            std.process.exit(1);
        }
    }

    // Test 3: Multiple cycles
    print("Test 3: Multiple cycles... ", .{});
    {
        // Cycle 1 - normal
        var result = talu_signal_guard_call(normalCallback, null);
        if (result != 0) {
            print("FAIL (cycle 1: expected 0, got {d})\n", .{result});
            std.process.exit(1);
        }

        // Cycle 2 - with signal
        result = talu_signal_guard_call(sigbusCallback, null);
        if (result != -1) {
            print("FAIL (cycle 2: expected -1, got {d})\n", .{result});
            std.process.exit(1);
        }

        // Cycle 3 - normal again
        result = talu_signal_guard_call(normalCallback, null);
        if (result != 0) {
            print("FAIL (cycle 3: expected 0, got {d})\n", .{result});
            std.process.exit(1);
        }

        print("PASS\n", .{});
    }

    // Test 4: Return value propagation
    print("Test 4: Return value propagation... ", .{});
    {
        const result = talu_signal_guard_call(returns42, null);
        if (result == 42) {
            print("PASS\n", .{});
        } else {
            print("FAIL (expected 42, got {d})\n", .{result});
            std.process.exit(1);
        }
    }

    print("\nAll tests passed!\n", .{});
}
