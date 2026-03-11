const std = @import("std");
const collab = @import("main").collab;

test "TextCrdt local and remote ops converge deterministically" {
    var local = collab.TextCrdt.init(std.testing.allocator);
    defer local.deinit();
    var remote = collab.TextCrdt.init(std.testing.allocator);
    defer remote.deinit();

    var counter: u64 = 1;
    const ops = try local.localInsert(std.testing.allocator, 3, &counter, 0, "ab");
    defer collab.crdt.text_engine.freeInsertOps(std.testing.allocator, ops);

    _ = try remote.applyInsert(ops[1]);
    _ = try remote.applyInsert(ops[0]);

    const left = try local.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(left);
    const right = try remote.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(right);
    try std.testing.expectEqualStrings(left, right);
    try std.testing.expectEqualStrings("ab", left);

    _ = try remote.applyDelete(.{ .target = ops[0].id });
    const after = try remote.visibleText(std.testing.allocator);
    defer std.testing.allocator.free(after);
    try std.testing.expectEqualStrings("b", after);
}

test "LamportClock remains monotonic across observations" {
    var c = collab.LamportClock{};
    _ = c.tick();
    _ = c.tick();
    try std.testing.expectEqual(@as(u64, 8), c.observe(7));
    try std.testing.expectEqual(@as(u64, 9), c.observe(2));
}
