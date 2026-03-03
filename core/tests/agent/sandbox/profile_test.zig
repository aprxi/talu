//! Integration tests for `agent.sandbox.profile`.

const std = @import("std");
const main = @import("main");
const policy_mod = main.agent.policy;
const profile_mod = main.agent.sandbox.profile;

fn hasName(list: []const []const u8, needle: []const u8) bool {
    for (list) |item| {
        if (std.mem.eql(u8, item, needle)) return true;
    }
    return false;
}

test "buildExecProfile honors tool.exec deny rules" {
    const json =
        \\{
        \\  "default":"allow",
        \\  "statements":[
        \\    {"effect":"deny","action":"tool.exec","command":"ls *"}
        \\  ]
        \\}
    ;
    var policy = try policy_mod.parsePolicy(std.testing.allocator, json);
    defer policy.deinit();

    var profile = try profile_mod.buildExecProfile(std.testing.allocator, &policy, .{
        .action = "tool.exec",
        .include_shell_paths = true,
    });
    defer profile.deinit();

    try std.testing.expect(!hasName(profile.names, "ls"));
    try std.testing.expect(hasName(profile.names, "bash"));
}

