//! Segment source abstraction for block readers.
//!
//! Provides a narrow seam for swapping local filesystem segment access with
//! alternate backends (for example object-store range reads + cache).

const std = @import("std");

pub const SegmentHandle = struct {
    file: std.fs.File,
    size: u64,
};

pub const SegmentSource = struct {
    context: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        openReadOnly: *const fn (context: *anyopaque, rel_path: []const u8) anyerror!SegmentHandle,
        close: *const fn (context: *anyopaque, handle: *SegmentHandle) void,
    };

    pub fn openReadOnly(self: SegmentSource, rel_path: []const u8) !SegmentHandle {
        return self.vtable.openReadOnly(self.context, rel_path);
    }

    pub fn close(self: SegmentSource, handle: *SegmentHandle) void {
        self.vtable.close(self.context, handle);
    }
};

pub const LocalSegmentSource = struct {
    dir: *std.fs.Dir,

    pub fn init(dir: *std.fs.Dir) LocalSegmentSource {
        return .{ .dir = dir };
    }

    pub fn asSource(self: *LocalSegmentSource) SegmentSource {
        return .{
            .context = self,
            .vtable = &.{
                .openReadOnly = openReadOnly,
                .close = close,
            },
        };
    }

    fn openReadOnly(context: *anyopaque, rel_path: []const u8) !SegmentHandle {
        const self: *LocalSegmentSource = @ptrCast(@alignCast(context));
        const file = try self.dir.openFile(rel_path, .{ .mode = .read_only });
        errdefer file.close();
        const stat = try file.stat();
        return .{
            .file = file,
            .size = stat.size,
        };
    }

    fn close(_: *anyopaque, handle: *SegmentHandle) void {
        handle.file.close();
    }
};

test "LocalSegmentSource opens and closes relative path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.makePath("chat");
    var f = try tmp.dir.createFile("chat/seg-1.talu", .{ .read = true });
    defer f.close();
    try f.writeAll("abc");

    var dir = try tmp.dir.openDir(".", .{});
    defer dir.close();

    var local = LocalSegmentSource.init(&dir);
    const source = local.asSource();
    var handle = try source.openReadOnly("chat/seg-1.talu");
    try std.testing.expectEqual(@as(u64, 3), handle.size);

    var buf: [3]u8 = undefined;
    const n = try handle.file.preadAll(&buf, 0);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(u8, "abc", &buf);

    source.close(&handle);
}
