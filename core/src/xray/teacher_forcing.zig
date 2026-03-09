//! Teacher Forcing Hook for Reference Verification
//!
//! Provides a mechanism to override token sampling during verification runs.
//! The router/sampler checks this hook before sampling to allow forcing
//! exact token sequences from a reference file.

const std = @import("std");

/// Teacher forcing mode
pub const TeacherForcingMode = enum {
    /// Normal sampling (default)
    normal,
    /// Force tokens from reference
    forced,
};

/// Teacher forcing hook - provides forced tokens when active
pub const TeacherForcingHook = struct {
    mode: TeacherForcingMode,

    /// Token provider function (when mode == .forced)
    /// Returns null when no more forced tokens available
    get_next_token: ?*const fn (ctx: ?*anyopaque) ?u32,

    /// Context pointer for get_next_token
    context: ?*anyopaque,

    pub fn init() TeacherForcingHook {
        return .{
            .mode = .normal,
            .get_next_token = null,
            .context = null,
        };
    }

    /// Get the next forced token (if any)
    pub fn getNext(self: *const TeacherForcingHook) ?u32 {
        if (self.mode != .forced) return null;
        if (self.get_next_token) |func| {
            return func(self.context);
        }
        return null;
    }
};

/// Global teacher forcing hook (thread-local for safety)
threadlocal var global_hook: TeacherForcingHook = TeacherForcingHook.init();

/// Set teacher forcing mode with a token provider
pub fn enable(get_next_token: *const fn (ctx: ?*anyopaque) ?u32, context: ?*anyopaque) void {
    global_hook = .{
        .mode = .forced,
        .get_next_token = get_next_token,
        .context = context,
    };
}

/// Disable teacher forcing (return to normal sampling)
pub fn disable() void {
    global_hook = TeacherForcingHook.init();
}

/// Check if teacher forcing is active
pub fn isEnabled() bool {
    return global_hook.mode == .forced;
}

/// Get next forced token (for use by router/sampler)
pub fn getNextToken() ?u32 {
    return global_hook.getNext();
}

// ============================================================================
// Integration Example
// ============================================================================

/// Example integration with ReferenceVerifier
pub const VerifierTokenProvider = struct {
    verifier: *anyopaque, // Actually *ReferenceVerifier but kept opaque

    pub fn getNextToken(ctx: ?*anyopaque) ?u32 {
        const self: *VerifierTokenProvider = @ptrCast(@alignCast(ctx orelse return null));
        // In real usage, would call verifier.getNextToken()
        // For now, return null
        _ = self;
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "teacher forcing basic flow" {
    // Test helper that provides tokens
    const TestProvider = struct {
        tokens: []const u32,
        index: usize,

        fn getNext(ctx: ?*anyopaque) ?u32 {
            const self: *@This() = @ptrCast(@alignCast(ctx orelse return null));
            if (self.index >= self.tokens.len) return null;
            const token = self.tokens[self.index];
            self.index += 1;
            return token;
        }
    };

    var provider = TestProvider{
        .tokens = &[_]u32{ 100, 200, 300 },
        .index = 0,
    };

    try std.testing.expect(!isEnabled());

    enable(&TestProvider.getNext, &provider);
    try std.testing.expect(isEnabled());

    try std.testing.expectEqual(@as(?u32, 100), getNextToken());
    try std.testing.expectEqual(@as(?u32, 200), getNextToken());
    try std.testing.expectEqual(@as(?u32, 300), getNextToken());
    try std.testing.expectEqual(@as(?u32, null), getNextToken());

    disable();
    try std.testing.expect(!isEnabled());
}
