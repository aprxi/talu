//! Durable append-only job state log.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub const JobState = enum {
    pending,
    running,
    succeeded,
    failed,
};

pub const JobRecord = struct {
    id: u64,
    name: []u8,
    state: JobState,
    updated_ms: i64,
    message: []u8,
};

const JobLine = struct {
    id: u64,
    name: []const u8,
    state: JobState,
    updated_ms: i64,
    message: []const u8,
};

pub const JobStore = struct {
    allocator: Allocator,
    log_path: []u8,
    by_id: std.AutoHashMap(u64, JobRecord),
    next_id: u64,

    pub fn init(allocator: Allocator, log_path: []const u8) !JobStore {
        var store = JobStore{
            .allocator = allocator,
            .log_path = try allocator.dupe(u8, log_path),
            .by_id = std.AutoHashMap(u64, JobRecord).init(allocator),
            .next_id = 1,
        };
        errdefer store.deinit();
        try store.loadFromDisk();
        return store;
    }

    pub fn deinit(self: *JobStore) void {
        var iter = self.by_id.valueIterator();
        while (iter.next()) |record| {
            self.allocator.free(record.name);
            self.allocator.free(record.message);
        }
        self.by_id.deinit();
        self.allocator.free(self.log_path);
        self.log_path = &[_]u8{};
    }

    pub fn startJob(self: *JobStore, name: []const u8) !u64 {
        const id = self.next_id;
        self.next_id += 1;
        try self.updateJob(id, name, .pending, "");
        return id;
    }

    pub fn updateJob(self: *JobStore, id: u64, name: []const u8, state: JobState, message: []const u8) !void {
        const line = JobLine{
            .id = id,
            .name = name,
            .state = state,
            .updated_ms = std.time.milliTimestamp(),
            .message = message,
        };
        try self.appendLine(line);
        try self.applyLine(line);
        if (id >= self.next_id) self.next_id = id + 1;
    }

    pub fn getJob(self: *JobStore, id: u64) ?JobRecord {
        if (self.by_id.get(id)) |record| {
            return .{
                .id = record.id,
                .name = self.allocator.dupe(u8, record.name) catch return null,
                .state = record.state,
                .updated_ms = record.updated_ms,
                .message = self.allocator.dupe(u8, record.message) catch return null,
            };
        }
        return null;
    }

    fn loadFromDisk(self: *JobStore) !void {
        const bytes = std.fs.cwd().readFileAlloc(self.allocator, self.log_path, 64 * 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer self.allocator.free(bytes);

        var iter = std.mem.splitScalar(u8, bytes, '\n');
        while (iter.next()) |line| {
            if (line.len == 0) continue;
            const parsed = try std.json.parseFromSlice(JobLine, self.allocator, line, .{});
            defer parsed.deinit();
            try self.applyLine(parsed.value);
        }
    }

    fn appendLine(self: *JobStore, line: JobLine) !void {
        const parent = std.fs.path.dirname(self.log_path) orelse ".";
        try std.fs.cwd().makePath(parent);

        var file = try std.fs.cwd().createFile(self.log_path, .{
            .truncate = false,
            .read = true,
        });
        defer file.close();
        try file.seekFromEnd(0);

        const encoded = try std.json.Stringify.valueAlloc(self.allocator, line, .{});
        defer self.allocator.free(encoded);
        try file.writeAll(encoded);
        try file.writeAll("\n");
        try file.sync();
    }

    fn applyLine(self: *JobStore, line: JobLine) !void {
        const name = try self.allocator.dupe(u8, line.name);
        errdefer self.allocator.free(name);
        const message = try self.allocator.dupe(u8, line.message);
        errdefer self.allocator.free(message);

        if (try self.by_id.fetchPut(line.id, .{
            .id = line.id,
            .name = name,
            .state = line.state,
            .updated_ms = line.updated_ms,
            .message = message,
        })) |existing| {
            self.allocator.free(existing.value.name);
            self.allocator.free(existing.value.message);
        }
    }
};

test "JobStore.init and JobStore.deinit initialize and release store" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const log_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(log_path);

    var store = try JobStore.init(std.testing.allocator, log_path);
    store.deinit();
}

test "JobStore.startJob, JobStore.updateJob, and JobStore.getJob track latest state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const log_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(log_path);

    var store = try JobStore.init(std.testing.allocator, log_path);
    defer store.deinit();

    const job_id = try store.startJob("compact");
    try store.updateJob(job_id, "compact", .running, "started");
    try store.updateJob(job_id, "compact", .succeeded, "done");

    const record = store.getJob(job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(JobState.succeeded, record.state);
    try std.testing.expectEqualStrings("compact", record.name);
    try std.testing.expectEqualStrings("done", record.message);
}

test "JobStore.init reloads durable state from log" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const log_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(log_path);

    {
        var writer = try JobStore.init(std.testing.allocator, log_path);
        defer writer.deinit();
        const id = try writer.startJob("index-build");
        try writer.updateJob(id, "index-build", .failed, "boom");
    }

    var reader = try JobStore.init(std.testing.allocator, log_path);
    defer reader.deinit();

    const record = reader.getJob(1) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(JobState.failed, record.state);
    try std.testing.expectEqualStrings("index-build", record.name);
}
