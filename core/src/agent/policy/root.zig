//! Unified IAM-style policy module for agent tool/runtime operations.
//!
//! Explicit deny always wins over allow. If no statement matches, the policy
//! default applies.

const std = @import("std");
const json_mod = @import("../../io/json/root.zig");
pub const pattern = @import("pattern.zig");
pub const evaluate = @import("evaluate.zig");

// Re-export primary types.
pub const Policy = evaluate.Policy;
pub const Statement = evaluate.Statement;
pub const Effect = evaluate.Effect;
pub const Mode = evaluate.Mode;
pub const EvaluateInput = evaluate.EvaluateInput;
pub const globMatch = pattern.globMatch;
pub const pathMatch = pattern.pathMatch;

pub const ProcessDenyReason = enum {
    action,
    cwd,
};

pub const ProcessCheckResult = struct {
    allowed: bool,
    deny_reason: ?ProcessDenyReason = null,
};

/// Return a definitive effect for all descendants of `directory_resource`, when
/// it can be proven from recursive resource rules.
///
/// Returns `null` when per-entry evaluation is required.
pub fn checkFileDescendantSubtree(
    policy: *const Policy,
    action: []const u8,
    directory_resource: []const u8,
) ?Effect {
    var has_action_deny = false;
    var has_universal_allow = false;

    for (policy.statements) |stmt| {
        if (!actionPatternMatches(stmt.action_pattern, action)) continue;

        switch (stmt.effect) {
            .deny => {
                has_action_deny = true;
                if (resourceMatchesAllDescendants(stmt.resource_pattern, directory_resource)) {
                    return .deny;
                }
            },
            .allow => {
                if (resourceMatchesAllDescendants(stmt.resource_pattern, directory_resource)) {
                    has_universal_allow = true;
                }
            },
        }
    }

    // Conservative allow short-circuit: only when this action has no deny
    // statements at all.
    if (has_action_deny) return null;
    if (policy.default_effect == .allow) return .allow;
    if (has_universal_allow) return .allow;
    return null;
}

/// Parse a policy from JSON.
///
/// Accepted schema:
/// ```json
/// {
///   "version": 1,
///   "default": "deny",
///   "mode": "enforce",
///   "max_timeout_ms": 120000,
///   "statements": [
///     {"effect":"allow","action":"tool.exec","command":"rg *","cwd":"repo/**"},
///     {"effect":"allow","action":"tool.fs.read","resource":"src/**"}
///   ]
/// }
/// ```
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
    if (pj.version != null and pj.version.? != 1) return error.InvalidVersion;

    const default_effect = try parseEffect(pj.default);
    const mode: Mode = if (pj.mode) |m| try parseMode(m) else .enforce;

    const stmt_count = pj.statements.len;
    const statements = try allocator.alloc(Statement, stmt_count);
    errdefer allocator.free(statements);

    var total_len: usize = 0;
    for (pj.statements) |s| {
        total_len += s.action.len;
        if (s.command) |value| total_len += value.len;
        if (s.cwd) |value| total_len += value.len;
        if (s.resource) |value| total_len += value.len;
    }

    const pattern_buf = try allocator.alloc(u8, total_len);
    errdefer allocator.free(pattern_buf);

    var offset: usize = 0;
    for (pj.statements, 0..) |s, i| {
        const action = copyInto(pattern_buf, &offset, s.action);
        const command = if (s.command) |v| copyInto(pattern_buf, &offset, v) else null;
        const cwd = if (s.cwd) |v| copyInto(pattern_buf, &offset, v) else null;
        const resource = if (s.resource) |v| copyInto(pattern_buf, &offset, v) else null;

        statements[i] = .{
            .effect = try parseEffect(s.effect),
            .action_pattern = action,
            .command_pattern = command,
            .cwd_pattern = cwd,
            .resource_pattern = resource,
        };
    }

    return Policy{
        .default_effect = default_effect,
        .mode = mode,
        .statements = statements,
        ._pattern_buf = pattern_buf,
        .allocator = allocator,
        .max_timeout_ms = pj.max_timeout_ms,
    };
}

/// Check a file action on a normalized workspace-relative resource path.
pub fn checkFileAction(
    policy: *const Policy,
    action: []const u8,
    resource: []const u8,
    is_dir: bool,
) bool {
    return policy.evaluateDetailed(.{
        .action = action,
        .resource = resource,
        .resource_is_dir = is_dir,
    }) == .allow;
}

/// Check process action constraints and classify cwd-specific denials.
pub fn checkProcessAction(
    policy: *const Policy,
    action: []const u8,
    command: ?[]const u8,
    cwd: ?[]const u8,
) ProcessCheckResult {
    const full_allowed = policy.evaluateDetailed(.{
        .action = action,
        .command = command,
        .cwd = cwd,
    }) == .allow;
    if (full_allowed) return .{ .allowed = true };

    if (cwd != null) {
        const without_cwd = policy.evaluateDetailed(.{
            .action = action,
            .command = command,
            .cwd = null,
        }) == .allow;
        if (without_cwd) {
            return .{ .allowed = false, .deny_reason = .cwd };
        }
    }

    return .{ .allowed = false, .deny_reason = .action };
}

fn actionPatternMatches(pattern_value: []const u8, action: []const u8) bool {
    if (pattern.globMatch(pattern_value, action)) return true;

    // Keep parity with evaluate.actionMatch for trailing " *".
    if (pattern_value.len >= 2 and
        pattern_value[pattern_value.len - 1] == '*' and
        pattern_value[pattern_value.len - 2] == ' ')
    {
        const prefix = pattern_value[0 .. pattern_value.len - 2];
        return pattern.globMatch(prefix, action);
    }
    return false;
}

fn resourceMatchesAllDescendants(resource_pattern: ?[]const u8, directory_resource: []const u8) bool {
    const raw = resource_pattern orelse return true;
    if (raw.len == 0) return false;

    var pattern_value = raw;
    if (pattern_value[0] == '/') {
        pattern_value = pattern_value[1..];
        if (pattern_value.len == 0) return true;
    }

    if (std.mem.eql(u8, pattern_value, "**")) return true;
    if (!std.mem.endsWith(u8, pattern_value, "/**")) return false;

    const base = pattern_value[0 .. pattern_value.len - 3];
    if (base.len == 0) return true;
    if (std.mem.indexOfAny(u8, base, "*?") != null) return false;

    return isAncestorOrSelf(base, directory_resource);
}

fn isAncestorOrSelf(ancestor: []const u8, candidate: []const u8) bool {
    if (ancestor.len == 0) return true;
    if (std.mem.eql(u8, ancestor, candidate)) return true;
    if (!std.mem.startsWith(u8, candidate, ancestor)) return false;
    if (candidate.len <= ancestor.len) return false;

    const sep = candidate[ancestor.len];
    return sep == '/' or sep == '\\';
}

fn copyInto(storage: []u8, offset: *usize, value: []const u8) []const u8 {
    const start = offset.*;
    const end = start + value.len;
    @memcpy(storage[start..end], value);
    offset.* = end;
    return storage[start..end];
}

const PolicyJson = struct {
    version: ?u32 = null,
    default: []const u8,
    mode: ?[]const u8 = null,
    max_timeout_ms: ?u64 = null,
    statements: []const StatementJson,
};

const StatementJson = struct {
    effect: []const u8,
    action: []const u8,
    command: ?[]const u8 = null,
    cwd: ?[]const u8 = null,
    resource: ?[]const u8 = null,
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

test "parsePolicy with command cwd and resource filters" {
    const json =
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"tool.exec","command":"rg *","cwd":"repo/**"},
        \\  {"effect":"allow","action":"tool.fs.read","resource":"src/**"}
        \\]}
    ;
    var policy = try parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    try std.testing.expect(checkProcessAction(&policy, "tool.exec", "rg foo", "repo").allowed);
    try std.testing.expect(!checkProcessAction(&policy, "tool.exec", "rg foo", "tmp").allowed);
    try std.testing.expect(checkFileAction(&policy, "tool.fs.read", "src/main.zig", false));
    try std.testing.expect(!checkFileAction(&policy, "tool.fs.read", "README.md", false));
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

test "parsePolicy invalid version" {
    const json =
        \\{"version":2,"default":"deny","statements":[]}
    ;
    try std.testing.expectError(error.InvalidVersion, parsePolicy(std.testing.allocator, json));
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
    const policy_json_a = try std.fmt.allocPrint(
        alloc,
        "{{\"default\":\"allow\",\"statements\":[{{\"effect\":\"allow\",\"action\":\"rm *\"}},{{\"effect\":\"deny\",\"action\":\"rm *\"}}]}}",
        .{},
    );
    defer alloc.free(policy_json_a);
    const policy_json_b = try std.fmt.allocPrint(
        alloc,
        "{{\"default\":\"allow\",\"statements\":[{{\"effect\":\"deny\",\"action\":\"rm *\"}},{{\"effect\":\"allow\",\"action\":\"rm *\"}}]}}",
        .{},
    );
    defer alloc.free(policy_json_b);

    var policy_a = try parsePolicy(alloc, policy_json_a);
    defer policy_a.deinit();
    var policy_b = try parsePolicy(alloc, policy_json_b);
    defer policy_b.deinit();

    try std.testing.expectEqual(Effect.deny, policy_a.evaluate("rm -rf /tmp"));
    try std.testing.expectEqual(Effect.deny, policy_b.evaluate("rm -rf /tmp"));
}

test "checkFileDescendantSubtree returns deny for recursive deny ancestor" {
    var policy_obj = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[
        \\  {"effect":"deny","action":"tool.fs.read","resource":"src/**"}
        \\]}
    );
    defer policy_obj.deinit();

    try std.testing.expectEqual(
        Effect.deny,
        checkFileDescendantSubtree(&policy_obj, "tool.fs.read", "src").?,
    );
    try std.testing.expectEqual(
        Effect.deny,
        checkFileDescendantSubtree(&policy_obj, "tool.fs.read", "src/deep").?,
    );
}

test "checkFileDescendantSubtree returns allow only when deny-free for action" {
    var policy_obj = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"tool.fs.read","resource":"src/**"},
        \\  {"effect":"deny","action":"tool.exec","command":"rm *"}
        \\]}
    );
    defer policy_obj.deinit();

    try std.testing.expectEqual(
        Effect.allow,
        checkFileDescendantSubtree(&policy_obj, "tool.fs.read", "src").?,
    );
}

test "checkFileDescendantSubtree is conservative when action has deny statements" {
    var policy_obj = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"tool.fs.read","resource":"src/**"},
        \\  {"effect":"deny","action":"tool.fs.read","resource":"src/private/**"}
        \\]}
    );
    defer policy_obj.deinit();

    try std.testing.expect(
        checkFileDescendantSubtree(&policy_obj, "tool.fs.read", "src") == null,
    );
}
