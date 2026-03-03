//! Integration tests for `agent.sandbox` runtime validation.

const std = @import("std");
const main = @import("main");
const sandbox = main.agent.sandbox;

test "sandbox.validate accepts host mode for all backends" {
    try sandbox.validate(.host, .linux_local);
    try sandbox.validate(.host, .oci);
    try sandbox.validate(.host, .apple_container);
}

test "sandbox.validate strict rejects unsupported backend" {
    try std.testing.expectError(error.StrictUnavailable, sandbox.validate(.strict, .oci));
    try std.testing.expectError(error.StrictUnavailable, sandbox.validate(.strict, .apple_container));
}

