//! Policy Module - IAM-style tool call firewall.
//!
//! Provides a policy engine that evaluates actions (tool call commands)
//! against user-defined allow/deny rules following AWS IAM semantics:
//!   - Explicit deny always wins
//!   - Explicit allow grants access when no deny matches
//!   - Default effect applies when no statement matches
//!
//! The policy is backend-agnostic: it evaluates a plain action string
//! and returns allow or deny. Shell-specific concerns (chain splitting,
//! subshell detection) are handled by the binding layer.
//!
//! Thread safety: Policy is immutable after creation. Safe to share
//! across threads via const pointer. To change policy mid-session,
//! create a new Policy and swap the pointer on the Chat.

const std = @import("std");
const json_mod = @import("../io/json/root.zig");
pub const pattern = @import("pattern.zig");
pub const evaluate = @import("evaluate.zig");

// Re-export main types at module level.
pub const Policy = evaluate.Policy;
pub const Statement = evaluate.Statement;
pub const Effect = evaluate.Effect;
pub const Mode = evaluate.Mode;
pub const globMatch = pattern.globMatch;

/// Parse a policy from a JSON string.
///
/// Expected format:
/// ```json
/// {
///   "default": "deny",
///   "mode": "enforce",
///   "statements": [
///     { "effect": "allow", "action": "ls *" },
///     { "effect": "deny",  "action": "rm *" }
///   ]
/// }
/// ```
///
/// `mode` is optional (defaults to `"enforce"`).
/// Caller owns the returned Policy and must call `deinit()`.
pub fn parsePolicy(allocator: std.mem.Allocator, json_bytes: []const u8) !Policy {
    const parsed = json_mod.parseStruct(allocator, PolicyJson, json_bytes, .{
        .max_size_bytes = 1 * 1024 * 1024,
        .ignore_unknown_fields = true,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();

    const pj = parsed.value;

    const default_effect = try parseEffect(pj.default);
    const mode: Mode = if (pj.mode) |m| try parseMode(m) else .enforce;

    // Allocate statements array.
    const stmt_count = pj.statements.len;
    const statements = try allocator.alloc(Statement, stmt_count);
    errdefer allocator.free(statements);

    // Calculate total pattern buffer size.
    var total_len: usize = 0;
    for (pj.statements) |s| {
        total_len += s.action.len;
    }

    // Single allocation for all pattern strings (contiguous).
    const pattern_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(pattern_buf);

    var offset: usize = 0;
    for (pj.statements, 0..) |s, i| {
        const len = s.action.len;
        @memcpy(pattern_buf[offset .. offset + len], s.action);

        statements[i] = .{
            .effect = try parseEffect(s.effect),
            .action_pattern = pattern_buf[offset .. offset + len],
        };
        offset += len;
    }

    return Policy{
        .default_effect = default_effect,
        .mode = mode,
        .statements = statements,
        ._pattern_buf = pattern_buf,
        .allocator = allocator,
    };
}

// =============================================================================
// JSON Schema Types
// =============================================================================

const PolicyJson = struct {
    default: []const u8,
    mode: ?[]const u8 = null,
    statements: []const StatementJson,
};

const StatementJson = struct {
    effect: []const u8,
    action: []const u8,
};

fn parseEffect(s: []const u8) !Effect {
    if (std.mem.eql(u8, s, "allow")) return .allow;
    if (std.mem.eql(u8, s, "deny")) return .deny;
    return error.InvalidEffect;
}

fn parseMode(s: []const u8) !Mode {
    if (std.mem.eql(u8, s, "enforce")) return .enforce;
    if (std.mem.eql(u8, s, "audit")) return .audit;
    return error.InvalidMode;
}

// =============================================================================
// Tests
// =============================================================================

test "parsePolicy basic allowlist" {
    const json =
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"ls *"},
        \\  {"effect":"allow","action":"grep *"},
        \\  {"effect":"deny","action":"rm *"}
        \\]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expectEqual(Effect.deny, policy.default_effect);
    try std.testing.expectEqual(Mode.enforce, policy.mode);
    try std.testing.expectEqual(@as(usize, 3), policy.statements.len);

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("grep foo bar.txt"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("dd if=/dev/zero"));
}

test "parsePolicy audit mode" {
    const json =
        \\{"default":"deny","mode":"audit","statements":[
        \\  {"effect":"allow","action":"ls *"}
        \\]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expectEqual(Mode.audit, policy.mode);
}

test "parsePolicy default allow with denies" {
    const json =
        \\{"default":"allow","statements":[
        \\  {"effect":"deny","action":"rm *"},
        \\  {"effect":"deny","action":"find * -exec *"}
        \\]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("find . -name foo"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("find . -exec rm {} ;"));
}

test "parsePolicy deny wins over allow (IAM invariant)" {
    const json =
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"git *"},
        \\  {"effect":"deny","action":"git push *"},
        \\  {"effect":"deny","action":"git commit *"}
        \\]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show HEAD"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("git log --oneline"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("git push origin main"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("git commit -m test"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("curl http://example.com"));
}

test "parsePolicy invalid effect" {
    const json =
        \\{"default":"maybe","statements":[]}
    ;
    try std.testing.expectError(error.InvalidEffect, parsePolicy(std.testing.allocator, json));
}

test "parsePolicy invalid mode" {
    const json =
        \\{"default":"deny","mode":"yolo","statements":[]}
    ;
    try std.testing.expectError(error.InvalidMode, parsePolicy(std.testing.allocator, json));
}

test "parsePolicy empty statements" {
    const json =
        \\{"default":"deny","statements":[]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expectEqual(Effect.deny, policy.evaluate("anything"));
}

test "parsePolicy explicit deny wins regardless of statement order" {
    const alloc = std.testing.allocator;
    const policy_json_a = try std.fmt.allocPrint(alloc,
        "{{\"default\":\"allow\",\"statements\":[{{\"effect\":\"allow\",\"action\":\"rm *\"}},{{\"effect\":\"deny\",\"action\":\"rm *\"}}]}}",
        .{},
    );
    defer alloc.free(policy_json_a);
    const policy_json_b = try std.fmt.allocPrint(alloc,
        "{{\"default\":\"allow\",\"statements\":[{{\"effect\":\"deny\",\"action\":\"rm *\"}},{{\"effect\":\"allow\",\"action\":\"rm *\"}}]}}",
        .{},
    );
    defer alloc.free(policy_json_b);

    var policy_a = try parsePolicy(alloc, policy_json_a);
    defer policy_a.deinit();
    var policy_b = try parsePolicy(alloc, policy_json_b);
    defer policy_b.deinit();

    try std.testing.expect(policy_a.evaluate("rm -rf /tmp") == .deny);
    try std.testing.expect(policy_b.evaluate("rm -rf /tmp") == .deny);
}

test "parsePolicy default effect applied when no statements match" {
    const alloc = std.testing.allocator;
    const policy_json = "{\"default\":\"deny\",\"statements\":[{\"effect\":\"allow\",\"action\":\"ls *\"}]}";
    var policy = try parsePolicy(alloc, policy_json);
    defer policy.deinit();

    try std.testing.expect(policy.evaluate("whoami") == .deny);
}
