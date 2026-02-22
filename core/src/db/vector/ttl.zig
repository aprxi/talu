//! TTL helpers for tombstone lifecycle decisions.

pub fn isExpired(event_ts_ms: i64, now_ms: i64, max_age_ms: i64) bool {
    if (max_age_ms < 0) return false;
    return event_ts_ms <= now_ms - max_age_ms;
}

pub fn shouldCompactForTtl(expired_tombstones: usize) bool {
    return expired_tombstones > 0;
}

test "isExpired returns true when tombstone age exceeds max_age" {
    try std.testing.expect(isExpired(1000, 5000, 3000));
    try std.testing.expect(!isExpired(3000, 5000, 3000));
}

test "shouldCompactForTtl is true only when expired tombstones exist" {
    try std.testing.expect(shouldCompactForTtl(1));
    try std.testing.expect(!shouldCompactForTtl(0));
}

const std = @import("std");
