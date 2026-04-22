//! Collaboration domain data contracts.

const std = @import("std");
const memory = @import("memory.zig");

/// Session participant category.
pub const ParticipantKind = enum(u8) {
    human = 0,
    agent = 1,
    external = 2,
    system = 3,
};

/// Storage lane for collaboration data.
///
/// - `strong` for durable checkpoints/audit-critical state.
/// - `batched` for high-frequency recoverable metadata.
/// - `ephemeral` for transient presence/liveness state.
pub const StorageLane = enum(u8) {
    strong = 0,
    batched = 1,
    ephemeral = 2,

    pub fn toDurabilityClass(self: StorageLane) memory.DurabilityClass {
        return switch (self) {
            .strong => .strong,
            .batched => .batched,
            .ephemeral => .ephemeral,
        };
    }
};

/// CRDT operation envelope persisted by collaboration sessions.
///
/// The payload is intentionally opaque bytes so resource-specific engines can
/// own their own schema (text/file/structured/etc.).
pub const OperationEnvelope = struct {
    actor_id: []const u8,
    actor_seq: u64,
    op_id: []const u8,
    payload: []const u8,
    issued_at_ms: i64,

    pub fn validate(self: OperationEnvelope) !void {
        if (self.actor_id.len == 0) return error.InvalidArgument;
        if (self.actor_seq == 0) return error.InvalidArgument;
        if (self.op_id.len == 0) return error.InvalidArgument;
        if (self.payload.len == 0) return error.InvalidArgument;
    }

    pub fn key(self: OperationEnvelope, allocator: std.mem.Allocator) ![]u8 {
        try self.validate();
        return std.fmt.allocPrint(allocator, "ops/{s}:{d}:{s}", .{ self.actor_id, self.actor_seq, self.op_id });
    }
};

test "StorageLane maps to KV durability classes" {
    try std.testing.expectEqual(memory.DurabilityClass.strong, StorageLane.strong.toDurabilityClass());
    try std.testing.expectEqual(memory.DurabilityClass.batched, StorageLane.batched.toDurabilityClass());
    try std.testing.expectEqual(memory.DurabilityClass.ephemeral, StorageLane.ephemeral.toDurabilityClass());
}

test "OperationEnvelope validation rejects invalid identities" {
    try std.testing.expectError(error.InvalidArgument, (OperationEnvelope{
        .actor_id = "",
        .actor_seq = 1,
        .op_id = "op1",
        .payload = "{}",
        .issued_at_ms = 1,
    }).validate());
    try std.testing.expectError(error.InvalidArgument, (OperationEnvelope{
        .actor_id = "a",
        .actor_seq = 0,
        .op_id = "op1",
        .payload = "{}",
        .issued_at_ms = 1,
    }).validate());
    try std.testing.expectError(error.InvalidArgument, (OperationEnvelope{
        .actor_id = "a",
        .actor_seq = 1,
        .op_id = "",
        .payload = "{}",
        .issued_at_ms = 1,
    }).validate());
    try std.testing.expectError(error.InvalidArgument, (OperationEnvelope{
        .actor_id = "a",
        .actor_seq = 1,
        .op_id = "op1",
        .payload = "",
        .issued_at_ms = 1,
    }).validate());
}
