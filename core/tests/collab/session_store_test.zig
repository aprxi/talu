const std = @import("std");
const collab = @import("main").collab;

test "SessionStore persists checkpoint and expires ephemeral presence" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    {
        var store = try collab.SessionStore.init(std.testing.allocator, root, "integration-session");
        defer store.deinit();

        store.setNowMsForTesting(1_000);
        try store.putPresence("user:1", "{\"cursor\":1}", 100);
        try store.putCheckpoint("rev", "{\"revision\":2}");
        try store.flush();

        const cp = (try store.getCheckpointCopy(std.testing.allocator, "rev")).?;
        defer std.testing.allocator.free(cp);
        try std.testing.expectEqualStrings("{\"revision\":2}", cp);

        store.setNowMsForTesting(1_100);
        try std.testing.expect((try store.getPresenceCopy(std.testing.allocator, "user:1")) == null);
    }

    {
        var reopened = try collab.SessionStore.init(std.testing.allocator, root, "integration-session");
        defer reopened.deinit();
        const cp = (try reopened.getCheckpointCopy(std.testing.allocator, "rev")).?;
        defer std.testing.allocator.free(cp);
        try std.testing.expectEqualStrings("{\"revision\":2}", cp);
    }
}
