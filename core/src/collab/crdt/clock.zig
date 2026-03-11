//! Lamport clock utility for collaboration envelopes.

const std = @import("std");

pub const LamportClock = struct {
    value: u64 = 0,

    /// Advance local clock for a locally-authored operation.
    pub fn tick(self: *LamportClock) u64 {
        self.value += 1;
        return self.value;
    }

    /// Observe a remote clock value and move local clock forward.
    pub fn observe(self: *LamportClock, remote: u64) u64 {
        self.value = @max(self.value, remote) + 1;
        return self.value;
    }
};

test "LamportClock tick and observe are monotonic" {
    var c = LamportClock{};
    try std.testing.expectEqual(@as(u64, 1), c.tick());
    try std.testing.expectEqual(@as(u64, 2), c.tick());
    try std.testing.expectEqual(@as(u64, 6), c.observe(5));
    try std.testing.expectEqual(@as(u64, 7), c.observe(3));
}
