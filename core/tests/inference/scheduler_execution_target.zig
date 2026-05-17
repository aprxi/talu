//! Integration tests for the scheduler-owned inference execution target.

const std = @import("std");
const main = @import("main");

const scheduler = main.inference.scheduler;

test "scheduler module exposes execution target selection" {
    try std.testing.expect(@hasDecl(scheduler, "ExecutionTarget"));
    try std.testing.expect(@hasDecl(scheduler, "ExecutionTargetInitOptions"));
    try std.testing.expect(@hasDecl(scheduler, "TargetSelection"));
}
