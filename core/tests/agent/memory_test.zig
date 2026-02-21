//! Integration tests for agent memory.

const std = @import("std");
const main = @import("main");

const memory = main.agent.memory;

test "agent.memory MemoryStore persists daily markdown and recalls snippets" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const db_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(db_path);

    var store = try memory.MemoryStore.init(allocator, .{
        .db_path = db_path,
        .namespace = "agent",
    });
    defer store.deinit();

    try store.upsertMarkdown("20260221.md", "First memory item");
    try store.appendDaily(1771632000000, "Second memory item");

    const markdown = try store.readMarkdown("20260221.md");
    try std.testing.expect(markdown != null);
    defer allocator.free(markdown.?);
    try std.testing.expect(std.mem.indexOf(u8, markdown.?, "First memory item") != null);
    try std.testing.expect(std.mem.indexOf(u8, markdown.?, "Second memory item") != null);

    const hits = try store.recall("second", 3);
    defer store.freeRecallHits(hits);

    try std.testing.expectEqual(@as(usize, 1), hits.len);
    try std.testing.expectEqualStrings("20260221.md", hits[0].filename);
    try std.testing.expect(std.mem.indexOf(u8, hits[0].snippet, "Second") != null);
}
