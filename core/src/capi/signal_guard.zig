//! Signal Guard - Graceful handling of fatal signals (SIGBUS, SIGSEGV)
//!
//! Provides a mechanism to catch fatal signals and convert them to error codes
//! instead of crashing. This is used to handle resource exhaustion scenarios
//! where the kernel sends SIGBUS (e.g., mmap page fault failures).
//!
//! Usage:
//!   const guard = SignalGuard.init();
//!   const result = guard.call(myFunction, context);
//!   // result is null if SIGBUS was caught
//!
//! Thread Safety:
//!   Each thread has its own jump buffer via threadlocal storage.
//!   Multiple threads can use SignalGuard simultaneously.
//!
//! Limitations:
//!   - Only works on POSIX systems (Linux, macOS)
//!   - Cannot recover from signals raised outside call() scope
//!   - Uses sigsetjmp/siglongjmp for async-signal-safe recovery

const std = @import("std");
const builtin = @import("builtin");

// C function from signal_guard.c
// Uses callback pattern because sigsetjmp must be called at the same stack
// frame where siglongjmp will return.
extern fn talu_signal_guard_install() callconv(.c) void;
extern fn talu_signal_guard_call(
    func: *const fn (?*anyopaque) callconv(.c) c_int,
    ctx: ?*anyopaque,
) callconv(.c) c_int;

/// Signal guard for catching fatal signals.
pub const SignalGuard = struct {
    /// Initialize signal handlers. Call once per thread that needs protection.
    pub fn init() SignalGuard {
        if (comptime builtin.os.tag == .linux or builtin.os.tag == .macos) {
            talu_signal_guard_install();
        }
        return .{};
    }

    /// Cleanup (currently no-op, handlers persist for thread lifetime).
    pub fn deinit(_: SignalGuard) void {
        // Handlers remain installed - they're safe to leave in place
    }

    /// Call a function with SIGBUS protection.
    /// The callback must have signature: fn(*anyopaque) callconv(.c) c_int
    /// Returns null if SIGBUS was caught, otherwise returns the function's return value.
    pub fn call(
        _: SignalGuard,
        func: *const fn (*anyopaque) callconv(.c) c_int,
        ctx: *anyopaque,
    ) ?c_int {
        if (comptime builtin.os.tag != .linux and builtin.os.tag != .macos) {
            // No protection on unsupported platforms - just call directly
            return func(ctx);
        }

        // The C function expects fn(?*anyopaque) callconv(.c) c_int
        // We need to cast our function pointer to match
        const c_func: *const fn (?*anyopaque) callconv(.c) c_int = @ptrCast(func);
        const result = talu_signal_guard_call(c_func, ctx);
        if (result == -1) {
            return null; // Signal was caught
        }
        return result;
    }
};

// =============================================================================
// Tests
// =============================================================================

fn testCallback(ctx: *anyopaque) callconv(.c) c_int {
    const flag: *bool = @ptrCast(@alignCast(ctx));
    flag.* = true;
    return 0;
}

test "SignalGuard: basic usage without signal" {
    const guard = SignalGuard.init();
    defer guard.deinit();

    var executed = false;
    const result = guard.call(&testCallback, @ptrCast(&executed));

    try std.testing.expect(executed);
    try std.testing.expectEqual(@as(?c_int, 0), result);
}

fn testReturns42(_: *anyopaque) callconv(.c) c_int {
    return 42;
}

test "SignalGuard: return value propagation" {
    const guard = SignalGuard.init();
    defer guard.deinit();

    var dummy: bool = false;
    const result = guard.call(&testReturns42, @ptrCast(&dummy));

    try std.testing.expectEqual(@as(?c_int, 42), result);
}

// Note: SIGBUS recovery tests cannot run in the Zig test framework because
// the framework has its own signal handlers that interfere. Run via CI:
//   ./scripts/check_signal_tests.sh
//
// Or see: tests/helpers/signal/signal_guard_test.zig
