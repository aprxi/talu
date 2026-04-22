//! Text CRDT engine (position-based sequence with tombstones).
//!
//! This engine provides deterministic convergence for concurrent inserts and
//! deletes. Insert order is position-first, then element ID as a tie-breaker.
//! Deletes are tombstones and are commutative with inserts via pending tombstone
//! tracking.

const std = @import("std");
const ids = @import("ids.zig");

const Allocator = std.mem.Allocator;
const ElementId = ids.ElementId;

pub const InsertOp = struct {
    id: ElementId,
    position: []u32,
    byte: u8,
};

pub const DeleteOp = struct {
    target: ElementId,
};

const Element = struct {
    id: ElementId,
    position: []u32,
    byte: u8,
    tombstone: bool,

    fn deinit(self: *Element, allocator: Allocator) void {
        allocator.free(self.position);
    }
};

pub const TextCrdt = struct {
    allocator: Allocator,
    elements: std.ArrayListUnmanaged(Element),
    pending_deletes: std.AutoHashMapUnmanaged(ElementId, void),

    pub fn init(allocator: Allocator) TextCrdt {
        return .{
            .allocator = allocator,
            .elements = .{},
            .pending_deletes = .{},
        };
    }

    pub fn deinit(self: *TextCrdt) void {
        for (self.elements.items) |*elem| elem.deinit(self.allocator);
        self.elements.deinit(self.allocator);
        self.pending_deletes.deinit(self.allocator);
    }

    /// Apply a remote insert operation.
    /// Returns false when the operation is a duplicate by element ID.
    pub fn applyInsert(self: *TextCrdt, op: InsertOp) !bool {
        try validatePosition(op.position);
        if (self.findById(op.id) != null) return false;

        const insert_idx = findInsertIndex(self.elements.items, op.position, op.id);
        const pos_copy = try self.allocator.dupe(u32, op.position);
        errdefer self.allocator.free(pos_copy);

        const tombstone = self.pending_deletes.fetchRemove(op.id) != null;
        try self.elements.insert(self.allocator, insert_idx, .{
            .id = op.id,
            .position = pos_copy,
            .byte = op.byte,
            .tombstone = tombstone,
        });
        return true;
    }

    /// Apply a delete operation.
    /// Returns true if a live element was found and tombstoned.
    /// Returns false if the element was already tombstoned.
    /// If the element does not exist yet, tombstone intent is persisted.
    pub fn applyDelete(self: *TextCrdt, op: DeleteOp) !bool {
        if (self.findById(op.target)) |idx| {
            if (self.elements.items[idx].tombstone) return false;
            self.elements.items[idx].tombstone = true;
            return true;
        }
        try self.pending_deletes.put(self.allocator, op.target, {});
        return false;
    }

    /// Allocate and return visible text bytes (tombstones omitted).
    pub fn visibleText(self: *const TextCrdt, allocator: Allocator) ![]u8 {
        const len = self.visibleLen();
        var out = try allocator.alloc(u8, len);
        var idx: usize = 0;
        for (self.elements.items) |elem| {
            if (elem.tombstone) continue;
            out[idx] = elem.byte;
            idx += 1;
        }
        return out;
    }

    pub fn visibleLen(self: *const TextCrdt) usize {
        var count: usize = 0;
        for (self.elements.items) |elem| {
            if (!elem.tombstone) count += 1;
        }
        return count;
    }

    /// Build and apply local insert operations at a visible index.
    /// Caller owns returned ops and must free them with `freeInsertOps`.
    pub fn localInsert(
        self: *TextCrdt,
        allocator: Allocator,
        actor_id: u64,
        next_counter: *u64,
        visible_index: usize,
        bytes: []const u8,
    ) ![]InsertOp {
        if (bytes.len == 0) return try allocator.alloc(InsertOp, 0);

        var ops = std.ArrayListUnmanaged(InsertOp){};
        errdefer freeInsertOps(allocator, ops.items);
        errdefer ops.deinit(allocator);

        var cursor = visible_index;
        for (bytes) |b| {
            const boundary = try self.visibleBoundary(cursor);
            const pos = try allocateBetween(allocator, boundary.prev, boundary.next);
            errdefer allocator.free(pos);

            const op = InsertOp{
                .id = .{ .actor_id = actor_id, .counter = next_counter.* },
                .position = pos,
                .byte = b,
            };
            next_counter.* += 1;

            _ = try self.applyInsert(op);
            try ops.append(allocator, op);
            cursor += 1;
        }
        return try ops.toOwnedSlice(allocator);
    }

    fn findById(self: *const TextCrdt, id: ElementId) ?usize {
        for (self.elements.items, 0..) |elem, idx| {
            if (ElementId.eql(elem.id, id)) return idx;
        }
        return null;
    }

    const Boundary = struct {
        prev: ?[]const u32,
        next: ?[]const u32,
    };

    fn visibleBoundary(self: *const TextCrdt, target_index: usize) !Boundary {
        var seen: usize = 0;
        var prev: ?[]const u32 = null;
        for (self.elements.items) |elem| {
            if (elem.tombstone) continue;
            if (seen == target_index) {
                return .{ .prev = prev, .next = elem.position };
            }
            prev = elem.position;
            seen += 1;
        }
        if (seen != target_index) return error.InvalidArgument;
        return .{ .prev = prev, .next = null };
    }
};

/// Free a slice returned by `TextCrdt.localInsert`.
pub fn freeInsertOps(allocator: Allocator, ops: []InsertOp) void {
    for (ops) |op| allocator.free(op.position);
    allocator.free(ops);
}

fn findInsertIndex(elements: []const Element, position: []const u32, id: ElementId) usize {
    var lo: usize = 0;
    var hi: usize = elements.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (lessThanPosThenId(elements[mid].position, elements[mid].id, position, id)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn lessThanPosThenId(a_pos: []const u32, a_id: ElementId, b_pos: []const u32, b_id: ElementId) bool {
    return switch (cmpPosition(a_pos, b_pos)) {
        .lt => true,
        .gt => false,
        .eq => ElementId.lessThan(a_id, b_id),
    };
}

fn cmpPosition(a: []const u32, b: []const u32) std.math.Order {
    const common = @min(a.len, b.len);
    var i: usize = 0;
    while (i < common) : (i += 1) {
        if (a[i] < b[i]) return .lt;
        if (a[i] > b[i]) return .gt;
    }
    if (a.len < b.len) return .lt;
    if (a.len > b.len) return .gt;
    return .eq;
}

fn validatePosition(pos: []const u32) !void {
    if (pos.len == 0) return error.InvalidArgument;
    for (pos) |digit| {
        if (digit == 0 or digit == std.math.maxInt(u32)) return error.InvalidArgument;
    }
}

fn allocateBetween(allocator: Allocator, prev: ?[]const u32, next: ?[]const u32) ![]u32 {
    const max_digit = std.math.maxInt(u32);
    var out = std.ArrayListUnmanaged(u32){};
    errdefer out.deinit(allocator);

    var depth: usize = 0;
    while (true) : (depth += 1) {
        if (depth > 128) return error.ResourceExhausted;
        const p = if (prev) |v| if (depth < v.len) v[depth] else 0 else 0;
        const n = if (next) |v| if (depth < v.len) v[depth] else max_digit else max_digit;
        if (p >= n) return error.InvalidArgument;
        if (p + 1 < n) {
            const mid = p + (n - p) / 2;
            if (mid == 0 or mid == max_digit) return error.InvalidArgument;
            try out.append(allocator, mid);
            break;
        }
        if (p == 0 or p == max_digit) return error.InvalidArgument;
        try out.append(allocator, p);
    }

    return out.toOwnedSlice(allocator);
}

test "TextCrdt.localInsert builds ordered text" {
    var crdt = TextCrdt.init(std.testing.allocator);
    defer crdt.deinit();

    var counter: u64 = 1;
    const ops = try crdt.localInsert(std.testing.allocator, 7, &counter, 0, "abc");
    defer freeInsertOps(std.testing.allocator, ops);

    try std.testing.expectEqual(@as(usize, 3), ops.len);
    try std.testing.expectEqual(@as(u64, 4), counter);

    const text = try crdt.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("abc", text);
}

test "TextCrdt concurrent inserts converge regardless of arrival order" {
    var a = TextCrdt.init(std.testing.allocator);
    defer a.deinit();
    var b = TextCrdt.init(std.testing.allocator);
    defer b.deinit();

    const op1 = InsertOp{ .id = .{ .actor_id = 1, .counter = 1 }, .position = &.{100}, .byte = 'x' };
    const op2 = InsertOp{ .id = .{ .actor_id = 2, .counter = 1 }, .position = &.{100}, .byte = 'y' };

    _ = try a.applyInsert(op1);
    _ = try a.applyInsert(op2);

    _ = try b.applyInsert(op2);
    _ = try b.applyInsert(op1);

    const text_a = try a.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(text_a);
    const text_b = try b.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(text_b);
    try std.testing.expectEqualStrings(text_a, text_b);
    try std.testing.expectEqualStrings("xy", text_a);
}

test "TextCrdt delete-before-insert leaves element tombstoned" {
    var crdt = TextCrdt.init(std.testing.allocator);
    defer crdt.deinit();

    _ = try crdt.applyDelete(.{ .target = .{ .actor_id = 9, .counter = 42 } });
    _ = try crdt.applyInsert(.{
        .id = .{ .actor_id = 9, .counter = 42 },
        .position = &.{50},
        .byte = 'z',
    });

    try std.testing.expectEqual(@as(usize, 0), crdt.visibleLen());
    const text = try crdt.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("", text);
}

test "TextCrdt applyDelete tombstones a visible element" {
    var crdt = TextCrdt.init(std.testing.allocator);
    defer crdt.deinit();

    var counter: u64 = 1;
    const ops = try crdt.localInsert(std.testing.allocator, 11, &counter, 0, "abc");
    defer freeInsertOps(std.testing.allocator, ops);

    _ = try crdt.applyDelete(.{ .target = ops[1].id });
    const text = try crdt.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("ac", text);
}

test "TextCrdt validatePosition rejects zero and max digits" {
    var crdt = TextCrdt.init(std.testing.allocator);
    defer crdt.deinit();

    try std.testing.expectError(error.InvalidArgument, crdt.applyInsert(.{
        .id = .{ .actor_id = 1, .counter = 1 },
        .position = &.{0},
        .byte = 'a',
    }));
    try std.testing.expectError(error.InvalidArgument, crdt.applyInsert(.{
        .id = .{ .actor_id = 1, .counter = 2 },
        .position = &.{std.math.maxInt(u32)},
        .byte = 'b',
    }));
}
