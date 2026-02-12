//! Integration tests for policy.Policy
//!
//! Tests the Policy struct through its public interface: parsePolicy (constructor),
//! evaluate, and deinit. Exercises IAM-style allow/deny semantics end-to-end.

const std = @import("std");
const main = @import("main");

const policy_mod = main.policy;
const Policy = policy_mod.Policy;
const Effect = policy_mod.Effect;
const Mode = policy_mod.Mode;
const parsePolicy = policy_mod.parsePolicy;

// =============================================================================
// evaluate
// =============================================================================

test "Policy.evaluate allows matching action with allow statement" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls /tmp"));
}

test "Policy.evaluate denies matching action with deny statement" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[{"effect":"deny","action":"rm *"}]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Effect.deny, policy.evaluate("rm -rf /"));
}

test "Policy.evaluate applies default effect when no statement matches" {
    var deny_default = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}
    );
    defer deny_default.deinit();

    try std.testing.expectEqual(Effect.deny, deny_default.evaluate("whoami"));

    var allow_default = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[{"effect":"deny","action":"rm *"}]}
    );
    defer allow_default.deinit();

    try std.testing.expectEqual(Effect.allow, allow_default.evaluate("whoami"));
}

test "Policy.evaluate explicit deny wins over allow (IAM invariant)" {
    // Both allow and deny match the same action; deny must win.
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"git *"},
        \\  {"effect":"deny","action":"git push *"}
        \\]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Effect.allow, policy.evaluate("git show HEAD"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("git push origin main"));
}

test "Policy.evaluate deny wins regardless of statement order" {
    // allow before deny
    var policy_a = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[
        \\  {"effect":"allow","action":"rm *"},
        \\  {"effect":"deny","action":"rm *"}
        \\]}
    );
    defer policy_a.deinit();

    // deny before allow
    var policy_b = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[
        \\  {"effect":"deny","action":"rm *"},
        \\  {"effect":"allow","action":"rm *"}
        \\]}
    );
    defer policy_b.deinit();

    try std.testing.expectEqual(Effect.deny, policy_a.evaluate("rm -rf /tmp"));
    try std.testing.expectEqual(Effect.deny, policy_b.evaluate("rm -rf /tmp"));
}

test "Policy.evaluate empty statements uses default" {
    var allow_policy = try parsePolicy(std.testing.allocator,
        \\{"default":"allow","statements":[]}
    );
    defer allow_policy.deinit();

    var deny_policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[]}
    );
    defer deny_policy.deinit();

    try std.testing.expectEqual(Effect.allow, allow_policy.evaluate("anything"));
    try std.testing.expectEqual(Effect.deny, deny_policy.evaluate("anything"));
}

test "Policy.evaluate bare command matches trailing space-star pattern" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[{"effect":"allow","action":"ls *"}]}
    );
    defer policy.deinit();

    // "ls *" should match bare "ls" (zero args) via matchesStatement.
    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls"));
    // Prefix substring must NOT match.
    try std.testing.expectEqual(Effect.deny, policy.evaluate("lsblk"));
}

test "Policy.evaluate middle wildcard pattern" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"find *"},
        \\  {"effect":"deny","action":"find * -exec *"}
        \\]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Effect.allow, policy.evaluate("find . -name foo"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("find . -exec rm {} ;"));
}

// =============================================================================
// deinit (lifecycle)
// =============================================================================

test "Policy.deinit releases all owned memory" {
    // Using testing allocator to verify no leaks on deinit.
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[
        \\  {"effect":"allow","action":"ls *"},
        \\  {"effect":"deny","action":"rm *"}
        \\]}
    );
    policy.deinit();
    // If deinit leaks, testing allocator will report it.
}

// =============================================================================
// parsePolicy (constructor + mode)
// =============================================================================

test "Policy.evaluate respects audit mode field" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","mode":"audit","statements":[{"effect":"allow","action":"ls *"}]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Mode.audit, policy.mode);
    // Evaluate still returns the computed effect; audit vs enforce is caller's concern.
    try std.testing.expectEqual(Effect.allow, policy.evaluate("ls -la"));
    try std.testing.expectEqual(Effect.deny, policy.evaluate("whoami"));
}

test "Policy parsePolicy defaults to enforce mode" {
    var policy = try parsePolicy(std.testing.allocator,
        \\{"default":"deny","statements":[]}
    );
    defer policy.deinit();

    try std.testing.expectEqual(Mode.enforce, policy.mode);
}

test "Policy parsePolicy rejects invalid effect" {
    try std.testing.expectError(error.InvalidEffect, parsePolicy(std.testing.allocator,
        \\{"default":"maybe","statements":[]}
    ));
}

test "Policy parsePolicy rejects invalid mode" {
    try std.testing.expectError(error.InvalidMode, parsePolicy(std.testing.allocator,
        \\{"default":"deny","mode":"yolo","statements":[]}
    ));
}

test "Policy parsePolicy rejects invalid JSON" {
    try std.testing.expectError(error.InvalidJson, parsePolicy(std.testing.allocator, "not json"));
}
