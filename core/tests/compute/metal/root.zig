//! Metal Integration Tests
//!
//! Tests for Metal GPU compute primitives.
//! All tests are skipped on non-macOS platforms at compile time.

pub const device = @import("device_test.zig");
pub const buffer = @import("buffer_test.zig");
pub const cache = @import("cache_test.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}
