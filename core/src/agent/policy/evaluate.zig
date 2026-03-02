//! Policy evaluation.
//!
//! Implements evaluation semantics:
//! 1. Collect all statements whose pattern matches the action.
//! 2. If any matching statement has effect = deny -> DENIED.
//! 3. If any matching statement has effect = allow -> ALLOWED.
//! 4. If no statement matches -> apply default effect.
//!
//! Explicit deny always wins. No allow can override a deny.

const std = @import("std");
const pattern_mod = @import("pattern.zig");

pub const Effect = enum(u8) {
    allow = 0,
    deny = 1,
};

pub const Mode = enum(u8) {
    enforce = 0,
    audit = 1,
};

pub const Statement = struct {
    effect: Effect,
    action_pattern: []const u8,
};

pub const Policy = struct {
    default_effect: Effect,
    mode: Mode,
    statements: []const Statement,
    /// Backing memory for statement patterns (owned, freed on deinit).
    _pattern_buf: []u8,
    allocator: std.mem.Allocator,

    /// Evaluate an action string against this policy.
    ///
    /// Returns the effect (allow or deny) for the given action.
    /// In audit mode, the caller should log denials but allow through.
    pub fn evaluate(self: *const Policy, action: []const u8) Effect {
        var has_allow = false;

        for (self.statements) |stmt| {
            if (matchesStatement(stmt.action_pattern, action)) {
                // Explicit deny wins immediately.
                if (stmt.effect == .deny) return .deny;
                has_allow = true;
            }
        }

        if (has_allow) return .allow;
        return self.default_effect;
    }

    /// Match an action against a statement pattern.
    ///
    /// A pattern ending with ` *` (space-wildcard) means "this command
    /// with any arguments". Since zero arguments is valid, `"ls *"` must
    /// match both `"ls -la"` and bare `"ls"`. We handle this by also
    /// trying an exact match against the prefix when the trailing ` *`
    /// glob doesn't match.
    fn matchesStatement(pat: []const u8, action: []const u8) bool {
        if (pattern_mod.globMatch(pat, action)) return true;

        // If pattern ends with " *", also accept the bare prefix.
        // e.g. pattern "ls *" with action "ls" -> match prefix "ls".
        if (pat.len >= 2 and
            pat[pat.len - 1] == '*' and
            pat[pat.len - 2] == ' ')
        {
            const prefix = pat[0 .. pat.len - 2]; // "ls"
            return pattern_mod.globMatch(prefix, action);
        }
        return false;
    }

    pub fn deinit(self: *Policy) void {
        self.allocator.free(self._pattern_buf);
        self.allocator.free(self.statements);
    }
};

// =============================================================================
// Tests
// =============================================================================

test "evaluate default deny blocks unmatched action" {
    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .allow, .action_pattern = "ls *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
}

test "evaluate trailing space-star matches bare command" {
    // "ls *" should match bare "ls" (zero args) as well as "ls -la" (with args).
    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .allow, .action_pattern = "ls *" },
            .{ .effect = .allow, .action_pattern = "git show *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    // Bare command (no args) — must be allowed.
    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show"));

    // With args — still allowed.
    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show HEAD"));

    // Unrelated commands — denied.
    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("git push origin main"));

    // Prefix substring should NOT match (no false positives).
    try std.testing.expectEqual(Effect.deny, policy.evaluate("lsblk"));
}

test "evaluate explicit deny wins over allow" {
    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .allow, .action_pattern = "find *" },
            .{ .effect = .deny, .action_pattern = "find * -exec *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    try std.testing.expectEqual(Effect.allow, policy.evaluate("find . -name foo"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("find . -exec rm {} ;"));
}

test "evaluate default allow permits unmatched action" {
    var policy = Policy{
        .default_effect = .allow,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .deny, .action_pattern = "rm *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
}

test "evaluate deny wins regardless of order" {
    // deny before allow
    var policy1 = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .deny, .action_pattern = "git push *" },
            .{ .effect = .allow, .action_pattern = "git *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy1;

    // allow before deny
    var policy2 = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .allow, .action_pattern = "git *" },
            .{ .effect = .deny, .action_pattern = "git push *" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy2;

    // Both should deny "git push origin main"
    try std.testing.expectEqual(Effect.deny, policy1.evaluate("git push origin main"));
    try std.testing.expectEqual(Effect.deny, policy2.evaluate("git push origin main"));

    // Both should allow "git show HEAD"
    try std.testing.expectEqual(Effect.allow, policy1.evaluate("git show HEAD"));
    try std.testing.expectEqual(Effect.allow, policy2.evaluate("git show HEAD"));
}

test "evaluate empty statements uses default" {
    var allow_policy = Policy{
        .default_effect = .allow,
        .mode = .enforce,
        .statements = &.{},
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &allow_policy;

    var deny_policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &.{},
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &deny_policy;

    try std.testing.expectEqual(Effect.allow, allow_policy.evaluate("anything"));
    try std.testing.expectEqual(Effect.deny, deny_policy.evaluate("anything"));
}
