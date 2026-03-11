//! Stable identifiers for collaboration CRDT entities.

const std = @import("std");

/// Unique element identifier in the text CRDT.
pub const ElementId = struct {
    actor_id: u64,
    counter: u64,

    pub fn lessThan(a: ElementId, b: ElementId) bool {
        if (a.actor_id != b.actor_id) return a.actor_id < b.actor_id;
        return a.counter < b.counter;
    }

    pub fn eql(a: ElementId, b: ElementId) bool {
        return a.actor_id == b.actor_id and a.counter == b.counter;
    }
};

test "ElementId ordering is deterministic" {
    const a = ElementId{ .actor_id = 1, .counter = 2 };
    const b = ElementId{ .actor_id = 2, .counter = 1 };
    const c = ElementId{ .actor_id = 2, .counter = 3 };
    try std.testing.expect(ElementId.lessThan(a, b));
    try std.testing.expect(ElementId.lessThan(b, c));
    try std.testing.expect(ElementId.eql(c, .{ .actor_id = 2, .counter = 3 }));
}
