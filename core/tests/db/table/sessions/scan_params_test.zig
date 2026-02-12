//! Integration tests for db.table.sessions.ScanParams
//!
//! ScanParams configures scanSessions pagination, cursor-based iteration,
//! and session filtering. Tests verify fromArgs translates arguments correctly.

const std = @import("std");
const main = @import("main");
const db = main.db;

const ScanParams = db.table.sessions.ScanParams;
const computeSessionHash = db.table.sessions.computeSessionHash;
const computeGroupHash = db.table.sessions.computeGroupHash;

// ===== fromArgs =====

test "ScanParams: fromArgs sets limit" {
    const params = ScanParams.fromArgs(50, 0, null, null);
    try std.testing.expectEqual(@as(u32, 50), params.limit);
    try std.testing.expect(params.before_ts == null);
    try std.testing.expect(params.before_session_hash == null);
    try std.testing.expect(params.target_group_hash == null);
    try std.testing.expect(params.target_group_id == null);
}

test "ScanParams: fromArgs sets cursor from timestamp" {
    const params = ScanParams.fromArgs(20, 1700000000, null, null);
    try std.testing.expectEqual(@as(u32, 20), params.limit);
    try std.testing.expectEqual(@as(i64, 1700000000), params.before_ts.?);
    try std.testing.expect(params.before_session_hash == null);
}

test "ScanParams: fromArgs sets cursor with session hash" {
    const params = ScanParams.fromArgs(10, 1700000000, "session-abc", null);
    try std.testing.expectEqual(@as(i64, 1700000000), params.before_ts.?);
    try std.testing.expectEqual(computeSessionHash("session-abc"), params.before_session_hash.?);
}

test "ScanParams: fromArgs zero timestamp skips cursor" {
    const params = ScanParams.fromArgs(10, 0, "session-abc", null);
    try std.testing.expect(params.before_ts == null);
    // session_hash is also skipped when timestamp is 0
    try std.testing.expect(params.before_session_hash == null);
}

test "ScanParams: fromArgs sets group filter" {
    const params = ScanParams.fromArgs(25, 0, null, "group-xyz");
    try std.testing.expectEqual(computeGroupHash("group-xyz"), params.target_group_hash.?);
    try std.testing.expectEqualStrings("group-xyz", params.target_group_id.?);
}

test "ScanParams: fromArgs sets all parameters" {
    const params = ScanParams.fromArgs(100, 999999, "cursor-sess", "my-group");
    try std.testing.expectEqual(@as(u32, 100), params.limit);
    try std.testing.expectEqual(@as(i64, 999999), params.before_ts.?);
    try std.testing.expectEqual(computeSessionHash("cursor-sess"), params.before_session_hash.?);
    try std.testing.expectEqual(computeGroupHash("my-group"), params.target_group_hash.?);
    try std.testing.expectEqualStrings("my-group", params.target_group_id.?);
}
