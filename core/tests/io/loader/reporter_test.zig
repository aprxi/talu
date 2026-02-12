//! Integration tests for io.loader.Reporter
//!
//! Reporter is used for validation reporting during model loading.

const std = @import("std");
const main = @import("main");
const Reporter = main.io.loader.Reporter;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Reporter type is accessible" {
    const T = Reporter;
    _ = T;
}

test "Reporter is a struct" {
    const info = @typeInfo(Reporter);
    try std.testing.expect(info == .@"struct");
}

test "Reporter has expected fields" {
    const info = @typeInfo(Reporter);
    const fields = info.@"struct".fields;

    var has_writer = false;
    var has_is_verbose = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "writer")) has_writer = true;
        if (comptime std.mem.eql(u8, field.name, "is_verbose")) has_is_verbose = true;
    }

    try std.testing.expect(has_writer);
    try std.testing.expect(has_is_verbose);
}

// =============================================================================
// Method Tests
// =============================================================================

test "Reporter has init method" {
    try std.testing.expect(@hasDecl(Reporter, "init"));
}

test "Reporter has reportInfo method" {
    try std.testing.expect(@hasDecl(Reporter, "reportInfo"));
}

test "Reporter has reportError method" {
    try std.testing.expect(@hasDecl(Reporter, "reportError"));
}

// =============================================================================
// Factory Method Tests
// =============================================================================

test "Reporter init creates reporter with correct verbosity" {
    var buffer: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    var any_writer = stream.writer().any();

    const reporter = Reporter.init(&any_writer, true);
    try std.testing.expect(reporter.is_verbose);

    const non_verbose = Reporter.init(&any_writer, false);
    try std.testing.expect(!non_verbose.is_verbose);
}
