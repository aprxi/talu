const std = @import("std");
const collab = @import("main").collab;

test "ResourceStore namespace is stable across reopen" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var a = try collab.ResourceStore.init(std.testing.allocator, root, "text_document", "doc-7");
    defer a.deinit();
    var b = try collab.ResourceStore.init(std.testing.allocator, root, "text_document", "doc-7");
    defer b.deinit();

    try std.testing.expectEqualStrings(a.namespaceId(), b.namespaceId());
}
