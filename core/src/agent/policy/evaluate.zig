//! Policy evaluation.
//!
//! Semantics:
//! 1. Match statements by action + optional command/cwd/resource filters.
//! 2. Explicit deny wins over allow.
//! 3. If no statements match, use default effect.

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

pub const TerminalShellMode = enum(u8) {
    host = 0,
    builtin = 1,
};

pub const Statement = struct {
    effect: Effect,
    action_pattern: []const u8,
    command_pattern: ?[]const u8 = null,
    cwd_pattern: ?[]const u8 = null,
    resource_pattern: ?[]const u8 = null,
};

pub const EvaluateInput = struct {
    action: []const u8,
    command: ?[]const u8 = null,
    cwd: ?[]const u8 = null,
    resource: ?[]const u8 = null,
    resource_is_dir: bool = false,
};

pub const Policy = struct {
    default_effect: Effect,
    mode: Mode,
    statements: []const Statement,
    /// Backing memory for statement patterns (owned, freed on deinit).
    _pattern_buf: []u8,
    allocator: std.mem.Allocator,
    max_timeout_ms: ?u64 = null,
    terminal_shell_mode: TerminalShellMode = .host,

    /// Evaluate a simple action string.
    pub fn evaluate(self: *const Policy, action: []const u8) Effect {
        return self.evaluateDetailed(.{ .action = action });
    }

    /// Evaluate action + optional command/cwd/resource constraints.
    pub fn evaluateDetailed(self: *const Policy, input: EvaluateInput) Effect {
        var has_allow = false;

        for (self.statements) |stmt| {
            if (!actionMatch(stmt.action_pattern, input.action)) continue;
            if (!optionalGlobMatch(stmt.command_pattern, input.command, true)) continue;
            if (!optionalCwdMatch(stmt.cwd_pattern, input.cwd)) continue;
            if (!optionalPathMatch(stmt.resource_pattern, input.resource, input.resource_is_dir)) continue;

            if (stmt.effect == .deny) return .deny;
            has_allow = true;
        }

        if (has_allow) return .allow;
        return self.default_effect;
    }

    /// Clamp requested timeout against policy max timeout, when configured.
    pub fn clampTimeoutMs(self: *const Policy, requested_ms: u64) u64 {
        if (self.max_timeout_ms) |max_timeout| {
            if (requested_ms == 0) return max_timeout;
            return @min(requested_ms, max_timeout);
        }
        return requested_ms;
    }

    pub fn deinit(self: *Policy) void {
        self.allocator.free(self._pattern_buf);
        self.allocator.free(self.statements);
    }
};

fn actionMatch(pattern: []const u8, action: []const u8) bool {
    if (pattern_mod.globMatch(pattern, action)) return true;

    // "ls *" should also match bare "ls" (zero args).
    if (pattern.len >= 2 and
        pattern[pattern.len - 1] == '*' and
        pattern[pattern.len - 2] == ' ')
    {
        const prefix = pattern[0 .. pattern.len - 2];
        return pattern_mod.globMatch(prefix, action);
    }
    return false;
}

fn optionalGlobMatch(pattern: ?[]const u8, value: ?[]const u8, allow_action_fallback: bool) bool {
    if (pattern == null) return true;
    const actual = value orelse return false;
    if (allow_action_fallback) {
        return actionMatch(pattern.?, actual);
    }
    return pattern_mod.globMatch(pattern.?, actual);
}

fn optionalCwdMatch(pattern: ?[]const u8, value: ?[]const u8) bool {
    if (pattern == null) return true;
    const actual = value orelse return false;

    // CWD is a normalized workspace-relative directory path, so use path
    // semantics first (supports `**` across separators).
    if (pattern_mod.pathMatch(pattern.?, actual, true)) return true;

    // Backward-compatible fallback for legacy glob-only patterns.
    return pattern_mod.globMatch(pattern.?, actual);
}

fn optionalPathMatch(pattern: ?[]const u8, value: ?[]const u8, is_dir: bool) bool {
    if (pattern == null) return true;
    const actual = value orelse return false;
    return pattern_mod.pathMatch(pattern.?, actual, is_dir);
}

// =============================================================================
// Tests
// =============================================================================

test "evaluateDetailed default deny blocks unmatched action" {
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

test "evaluateDetailed trailing space-star matches bare command" {
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

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show HEAD"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("lsblk"));
}

test "evaluateDetailed explicit deny wins over allow" {
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

test "evaluateDetailed matches command and cwd constraints" {
    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{
                .effect = .allow,
                .action_pattern = "tool.exec",
                .command_pattern = "rg *",
                .cwd_pattern = "repo/**",
            },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    try std.testing.expectEqual(Effect.allow, policy.evaluateDetailed(.{
        .action = "tool.exec",
        .command = "rg foo src",
        .cwd = "repo/core",
    }));
    try std.testing.expectEqual(Effect.deny, policy.evaluateDetailed(.{
        .action = "tool.exec",
        .command = "rg foo src",
        .cwd = "tmp",
    }));
}

test "evaluateDetailed matches resource constraints" {
    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]Statement{
            .{ .effect = .allow, .action_pattern = "tool.fs.read", .resource_pattern = "src/**" },
            .{ .effect = .deny, .action_pattern = "tool.fs.read", .resource_pattern = "src/secrets/**" },
        },
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
    };
    _ = &policy;

    try std.testing.expectEqual(Effect.allow, policy.evaluateDetailed(.{
        .action = "tool.fs.read",
        .resource = "src/main.zig",
    }));
    try std.testing.expectEqual(Effect.deny, policy.evaluateDetailed(.{
        .action = "tool.fs.read",
        .resource = "src/secrets/token.txt",
    }));
}

test "clampTimeoutMs applies policy max timeout" {
    var policy = Policy{
        .default_effect = .allow,
        .mode = .enforce,
        .statements = &.{},
        ._pattern_buf = &.{},
        .allocator = std.testing.allocator,
        .max_timeout_ms = 30_000,
    };
    _ = &policy;

    try std.testing.expectEqual(@as(u64, 20_000), policy.clampTimeoutMs(20_000));
    try std.testing.expectEqual(@as(u64, 30_000), policy.clampTimeoutMs(50_000));
    try std.testing.expectEqual(@as(u64, 30_000), policy.clampTimeoutMs(0));
}
