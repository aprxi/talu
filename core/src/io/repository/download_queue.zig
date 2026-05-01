//! In-process repository download queue.
//!
//! Core owns download lifecycle state so language bindings can be thin
//! controllers instead of inventing separate queue semantics.

const std = @import("std");
const log = @import("log_pkg");
const cache = @import("cache.zig");
const transport = @import("../transport/root.zig");
const progress_api = @import("progress_pkg");

pub const Status = enum(u8) {
    queued,
    active,
    paused,
    completed,
    failed,
    cancelled,

    pub fn asString(self: Status) []const u8 {
        return switch (self) {
            .queued => "queued",
            .active => "active",
            .paused => "paused",
            .completed => "completed",
            .failed => "failed",
            .cancelled => "cancelled",
        };
    }

    fn terminal(self: Status) bool {
        return self == .completed or self == .failed or self == .cancelled;
    }
};

pub const EnqueueOptions = struct {
    token: ?[]const u8 = null,
    force: bool = false,
    endpoint_url: ?[]const u8 = null,
    skip_weights: bool = false,
};

const Job = struct {
    id: []u8,
    model_id: []u8,
    token: ?[]u8,
    endpoint_url: ?[]u8,
    force: bool,
    skip_weights: bool,
    status: Status = .queued,
    path: ?[]u8 = null,
    err: ?[]u8 = null,
    started_at: i64,
    updated_at: i64,
    completed_at: ?i64 = null,
    current: u64 = 0,
    total: u64 = 0,
    label: []u8,
    message: []u8,
    cancel_flag: bool = false,
    pause_requested: bool = false,
    thread_running: bool = false,
};

const PersistedState = struct {
    version: u32 = 1,
    next_id: u64 = 1,
    jobs: []PersistedJob = &.{},
};

const PersistedJob = struct {
    id: []const u8,
    model_id: []const u8,
    status: []const u8,
    endpoint_url: ?[]const u8 = null,
    force: bool = false,
    skip_weights: bool = false,
    path: ?[]const u8 = null,
    @"error": ?[]const u8 = null,
    started_at: i64 = 0,
    updated_at: i64 = 0,
    completed_at: ?i64 = null,
    current: u64 = 0,
    total: u64 = 0,
    label: []const u8 = "",
    message: []const u8 = "",
};

const Manager = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    jobs: std.ArrayListUnmanaged(*Job) = .{},
    active_count: usize = 0,
    max_active: usize = 2,
    next_id: u64 = 1,
    loaded: bool = false,

    const WorkerProgressCtx = struct {
        manager: *Manager,
        job: *Job,
    };

    fn init(allocator: std.mem.Allocator) Manager {
        return .{ .allocator = allocator };
    }

    fn enqueue(self: *Manager, model_id: []const u8, options: EnqueueOptions) ![]u8 {
        try self.ensureLoaded();

        const job = try self.allocator.create(Job);
        errdefer self.allocator.destroy(job);

        const id_num = @atomicRmw(u64, &self.next_id, .Add, 1, .monotonic);
        const id = try std.fmt.allocPrint(self.allocator, "repo-download-{d}", .{id_num});
        errdefer self.allocator.free(id);

        job.* = .{
            .id = id,
            .model_id = try self.allocator.dupe(u8, model_id),
            .token = if (options.token) |v| try self.allocator.dupe(u8, v) else null,
            .endpoint_url = if (options.endpoint_url) |v| try self.allocator.dupe(u8, v) else null,
            .force = options.force,
            .skip_weights = options.skip_weights,
            .started_at = now(),
            .updated_at = now(),
            .label = try self.allocator.dupe(u8, ""),
            .message = try self.allocator.dupe(u8, ""),
        };
        errdefer self.freeJob(job);

        self.mutex.lock();
        defer self.mutex.unlock();
        try self.jobs.append(self.allocator, job);
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        self.spawnWorkerLocked(job) catch |err| {
            _ = self.jobs.pop();
            self.saveLocked() catch {};
            return err;
        };
        self.cond.broadcast();
        return try self.allocator.dupe(u8, job.id);
    }

    fn pause(self: *Manager, id: []const u8) !void {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();
        const job = self.findLocked(id) orelse return error.NotFound;
        if (job.status.terminal()) return error.InvalidState;
        job.pause_requested = true;
        @atomicStore(bool, &job.cancel_flag, true, .monotonic);
        if (job.status == .queued) job.status = .paused;
        job.updated_at = now();
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        self.cond.broadcast();
    }

    fn resumeJob(self: *Manager, id: []const u8) !void {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();
        const job = self.findLocked(id) orelse return error.NotFound;
        if (job.status != .paused) return error.InvalidState;
        job.status = .queued;
        job.pause_requested = false;
        @atomicStore(bool, &job.cancel_flag, false, .monotonic);
        job.updated_at = now();
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        if (!job.thread_running) try self.spawnWorkerLocked(job);
        self.cond.broadcast();
    }

    fn cancel(self: *Manager, id: []const u8) !void {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();
        const job = self.findLocked(id) orelse return error.NotFound;
        if (job.status.terminal()) return error.InvalidState;
        job.pause_requested = false;
        @atomicStore(bool, &job.cancel_flag, true, .monotonic);
        if (job.status == .queued or job.status == .paused) {
            job.status = .cancelled;
            job.completed_at = now();
            job.updated_at = job.completed_at.?;
        }
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        self.cond.broadcast();
    }

    fn clearFinished(self: *Manager) !usize {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();

        var removed: usize = 0;
        var i: usize = 0;
        while (i < self.jobs.items.len) {
            const job = self.jobs.items[i];
            if (!job.status.terminal()) {
                i += 1;
                continue;
            }
            _ = self.jobs.orderedRemove(i);
            self.freeJob(job);
            removed += 1;
        }

        if (removed > 0) {
            self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
            self.cond.broadcast();
        }
        return removed;
    }

    fn cancelAll(self: *Manager) !usize {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();

        var affected: usize = 0;
        const ts = now();
        for (self.jobs.items) |job| {
            if (job.status.terminal()) continue;
            affected += 1;
            job.pause_requested = false;
            @atomicStore(bool, &job.cancel_flag, true, .monotonic);
            if (job.status == .queued or job.status == .paused) {
                job.status = .cancelled;
                job.completed_at = ts;
                job.updated_at = ts;
            }
        }

        if (affected > 0) {
            self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
            self.cond.broadcast();
        }
        return affected;
    }

    fn snapshotJson(self: *Manager, allocator: std.mem.Allocator) ![]u8 {
        try self.ensureLoaded();

        self.mutex.lock();
        defer self.mutex.unlock();
        var out = std.ArrayListUnmanaged(u8){};
        errdefer out.deinit(allocator);
        const w = out.writer(allocator);

        try w.writeAll("{\"active\":[");
        try self.writeJobs(w, .active);
        try w.writeAll("],\"queued\":[");
        try self.writeJobs(w, .queued);
        try w.writeAll("],\"paused\":[");
        try self.writeJobs(w, .paused);
        try w.writeAll("],\"recent\":[");
        try self.writeRecent(w);
        try w.writeAll("]}");
        return out.toOwnedSlice(allocator);
    }

    fn writeJobs(self: *Manager, writer: anytype, status: Status) !void {
        var first = true;
        for (self.jobs.items) |job| {
            if (job.status != status) continue;
            if (!first) try writer.writeByte(',');
            first = false;
            try writeJobJson(writer, job);
        }
    }

    fn writeRecent(self: *Manager, writer: anytype) !void {
        var first = true;
        var remaining: usize = 50;
        var idx = self.jobs.items.len;
        while (idx > 0 and remaining > 0) {
            idx -= 1;
            const job = self.jobs.items[idx];
            if (!job.status.terminal()) continue;
            if (!first) try writer.writeByte(',');
            first = false;
            remaining -= 1;
            try writeJobJson(writer, job);
        }
    }

    fn ensureLoaded(self: *Manager) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.loaded) return;

        try self.loadLocked();
        self.loaded = true;

        for (self.jobs.items) |job| {
            if (job.status != .queued or job.thread_running) continue;
            self.spawnWorkerLocked(job) catch |err| {
                job.status = .failed;
                job.err = std.fmt.allocPrint(self.allocator, "spawn: {s}", .{@errorName(err)}) catch null;
                job.updated_at = now();
                job.completed_at = job.updated_at;
            };
        }
        self.saveLocked() catch {};
    }

    fn loadLocked(self: *Manager) !void {
        const path = try stateFilePath(self.allocator);
        defer self.allocator.free(path);

        const bytes = std.fs.cwd().readFileAlloc(self.allocator, path, 10 * 1024 * 1024) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer self.allocator.free(bytes);

        const parsed = std.json.parseFromSlice(PersistedState, self.allocator, bytes, .{
            .ignore_unknown_fields = true,
        }) catch return;
        defer parsed.deinit();

        self.next_id = @max(self.next_id, parsed.value.next_id);
        for (parsed.value.jobs) |persisted| {
            const status = parseStatus(persisted.status) orelse continue;
            const job = try self.allocator.create(Job);
            errdefer self.allocator.destroy(job);

            const runtime_status: Status = switch (status) {
                .active, .queued => .queued,
                else => status,
            };
            job.* = .{
                .id = try self.allocator.dupe(u8, persisted.id),
                .model_id = try self.allocator.dupe(u8, persisted.model_id),
                .token = null,
                .endpoint_url = if (persisted.endpoint_url) |v| try self.allocator.dupe(u8, v) else null,
                .force = persisted.force,
                .skip_weights = persisted.skip_weights,
                .status = runtime_status,
                .path = if (persisted.path) |v| try self.allocator.dupe(u8, v) else null,
                .err = if (persisted.@"error") |v| try self.allocator.dupe(u8, v) else null,
                .started_at = persisted.started_at,
                .updated_at = persisted.updated_at,
                .completed_at = persisted.completed_at,
                .current = persisted.current,
                .total = persisted.total,
                .label = try self.allocator.dupe(u8, persisted.label),
                .message = try self.allocator.dupe(u8, persisted.message),
            };
            errdefer self.freeJob(job);
            try self.jobs.append(self.allocator, job);
        }
    }

    fn saveLocked(self: *Manager) !void {
        const path = try stateFilePath(self.allocator);
        defer self.allocator.free(path);

        const dir_path = std.fs.path.dirname(path) orelse return error.InvalidState;
        try std.fs.cwd().makePath(dir_path);

        const temp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{path});
        defer self.allocator.free(temp_path);

        var out = std.ArrayListUnmanaged(u8){};
        defer out.deinit(self.allocator);
        try self.writeStateJson(out.writer(self.allocator));

        var file = try std.fs.cwd().createFile(temp_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(out.items);
        try std.fs.cwd().rename(temp_path, path);
    }

    fn writeStateJson(self: *Manager, writer: anytype) !void {
        try writer.print("{{\"version\":1,\"next_id\":{d},\"jobs\":[", .{self.next_id});
        var first = true;
        for (self.jobs.items) |job| {
            if (!first) try writer.writeByte(',');
            first = false;
            try writePersistedJobJson(writer, job);
        }
        try writer.writeAll("]}");
    }

    fn workerMain(self: *Manager, job: *Job) void {
        self.waitForSlot(job);
        if (@atomicLoad(bool, &job.cancel_flag, .monotonic)) {
            self.finishCancelledBeforeStart(job);
            return;
        }

        transport.globalInit();
        var progress_ctx = WorkerProgressCtx{ .manager = self, .job = job };
        const path = transport.hf.fetchModel(self.allocator, job.model_id, .{
            .token = job.token,
            .force = job.force,
            .endpoint_url = job.endpoint_url,
            .skip_weights = job.skip_weights,
            .progress = progress_api.Context.init(onProgress, @ptrCast(&progress_ctx)),
            .cancel_flag = &job.cancel_flag,
        });
        transport.globalCleanup();

        self.finish(job, path);
    }

    fn onProgress(update: *const progress_api.ProgressUpdate, user_data: ?*anyopaque) callconv(.c) void {
        const ctx: *WorkerProgressCtx = @ptrCast(@alignCast(user_data orelse return));
        ctx.manager.applyProgress(ctx.job, update);
    }

    fn applyProgress(self: *Manager, job: *Job, update: *const progress_api.ProgressUpdate) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (job.status.terminal()) return;

        job.updated_at = now();
        if (update.line_id == 1) {
            job.current = update.current;
            if (update.total > 0) job.total = update.total;
            if (update.action == .complete and job.total > 0) {
                job.current = job.total;
            }
        }
        if (update.label) |label| self.replaceOwnedRequired(&job.label, std.mem.span(label));
        if (update.message) |message| self.replaceOwnedRequired(&job.message, std.mem.span(message));
    }

    fn waitForSlot(self: *Manager, job: *Job) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.active_count >= self.max_active or job.status == .paused) {
            self.cond.wait(&self.mutex);
        }
        if (job.status == .queued) {
            job.status = .active;
            job.updated_at = now();
            self.active_count += 1;
        }
    }

    fn finish(self: *Manager, job: *Job, result: anyerror![]const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.active_count > 0) self.active_count -= 1;
        job.thread_running = false;
        job.updated_at = now();
        job.completed_at = job.updated_at;
        if (result) |path| {
            job.status = .completed;
            self.replaceOwned(&job.path, path);
        } else |err| {
            if (job.pause_requested) {
                job.status = .paused;
                job.completed_at = null;
                job.pause_requested = false;
                @atomicStore(bool, &job.cancel_flag, false, .monotonic);
            } else if (err == error.Cancelled) {
                job.status = .cancelled;
            } else {
                job.status = .failed;
                self.replaceOwnedFmt(&job.err, "{s}", .{@errorName(err)});
            }
        }
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        self.cond.broadcast();
    }

    fn finishCancelledBeforeStart(self: *Manager, job: *Job) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        job.thread_running = false;
        job.status = .cancelled;
        job.updated_at = now();
        job.completed_at = job.updated_at;
        self.saveLocked() catch |err| log.warn("repo", "Failed to persist download queue", .{ .err = @errorName(err) });
        self.cond.broadcast();
    }

    fn spawnWorkerLocked(self: *Manager, job: *Job) !void {
        job.thread_running = true;
        const thread = try std.Thread.spawn(.{}, Manager.workerMain, .{ self, job });
        thread.detach();
    }

    fn findLocked(self: *Manager, id: []const u8) ?*Job {
        for (self.jobs.items) |job| {
            if (std.mem.eql(u8, job.id, id)) return job;
        }
        return null;
    }

    fn replaceOwned(self: *Manager, field: *?[]u8, value: []const u8) void {
        if (field.*) |old| self.allocator.free(old);
        field.* = self.allocator.dupe(u8, value) catch null;
        self.allocator.free(value);
    }

    fn replaceOwnedRequired(self: *Manager, field: *[]u8, value: []const u8) void {
        const next = self.allocator.dupe(u8, value) catch return;
        self.allocator.free(field.*);
        field.* = next;
    }

    fn replaceOwnedFmt(self: *Manager, field: *?[]u8, comptime fmt: []const u8, args: anytype) void {
        if (field.*) |old| self.allocator.free(old);
        field.* = std.fmt.allocPrint(self.allocator, fmt, args) catch null;
    }

    fn freeJob(self: *Manager, job: *Job) void {
        self.allocator.free(job.id);
        self.allocator.free(job.model_id);
        if (job.token) |v| self.allocator.free(v);
        if (job.endpoint_url) |v| self.allocator.free(v);
        if (job.path) |v| self.allocator.free(v);
        if (job.err) |v| self.allocator.free(v);
        self.allocator.free(job.label);
        self.allocator.free(job.message);
        self.allocator.destroy(job);
    }
};

var manager = Manager.init(std.heap.c_allocator);

pub fn enqueue(_: std.mem.Allocator, model_id: []const u8, options: EnqueueOptions) ![]u8 {
    return manager.enqueue(model_id, options);
}

pub fn pause(id: []const u8) !void {
    return manager.pause(id);
}

pub fn resumeJob(id: []const u8) !void {
    return manager.resumeJob(id);
}

pub fn cancel(id: []const u8) !void {
    return manager.cancel(id);
}

pub fn clearFinished() !usize {
    return manager.clearFinished();
}

pub fn cancelAll() !usize {
    return manager.cancelAll();
}

pub fn snapshotJson(allocator: std.mem.Allocator) ![]u8 {
    return manager.snapshotJson(allocator);
}

fn writeJobJson(writer: anytype, job: *const Job) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, job.id);
    try writer.writeAll(",\"model_id\":");
    try writeJsonString(writer, job.model_id);
    try writer.writeAll(",\"status\":");
    try writeJsonString(writer, job.status.asString());
    try writer.print(",\"started_at\":{d},\"updated_at\":{d}", .{ job.started_at, job.updated_at });
    if (job.completed_at) |ts| try writer.print(",\"completed_at\":{d}", .{ts}) else try writer.writeAll(",\"completed_at\":null");
    try writer.print(",\"current\":{d},\"total\":{d}", .{ job.current, job.total });
    try writer.writeAll(",\"label\":");
    try writeJsonString(writer, job.label);
    try writer.writeAll(",\"message\":");
    try writeJsonString(writer, job.message);
    try writer.writeAll(",\"path\":");
    if (job.path) |path| try writeJsonString(writer, path) else try writer.writeAll("null");
    try writer.writeAll(",\"error\":");
    if (job.err) |err| try writeJsonString(writer, err) else try writer.writeAll("null");
    try writer.writeByte('}');
}

fn writePersistedJobJson(writer: anytype, job: *const Job) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, job.id);
    try writer.writeAll(",\"model_id\":");
    try writeJsonString(writer, job.model_id);
    try writer.writeAll(",\"status\":");
    try writeJsonString(writer, job.status.asString());
    try writer.writeAll(",\"endpoint_url\":");
    if (job.endpoint_url) |endpoint_url| try writeJsonString(writer, endpoint_url) else try writer.writeAll("null");
    try writer.print(",\"force\":{},\"skip_weights\":{}", .{ job.force, job.skip_weights });
    try writer.writeAll(",\"path\":");
    if (job.path) |path| try writeJsonString(writer, path) else try writer.writeAll("null");
    try writer.writeAll(",\"error\":");
    if (job.err) |err| try writeJsonString(writer, err) else try writer.writeAll("null");
    try writer.print(",\"started_at\":{d},\"updated_at\":{d}", .{ job.started_at, job.updated_at });
    if (job.completed_at) |ts| try writer.print(",\"completed_at\":{d}", .{ts}) else try writer.writeAll(",\"completed_at\":null");
    try writer.print(",\"current\":{d},\"total\":{d}", .{ job.current, job.total });
    try writer.writeAll(",\"label\":");
    try writeJsonString(writer, job.label);
    try writer.writeAll(",\"message\":");
    try writeJsonString(writer, job.message);
    try writer.writeByte('}');
}

fn parseStatus(value: []const u8) ?Status {
    inline for (std.meta.fields(Status)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @field(Status, field.name);
    }
    return null;
}

fn stateFilePath(allocator: std.mem.Allocator) ![]u8 {
    const state_home = try stateHome(allocator);
    defer allocator.free(state_home);
    return std.fs.path.join(allocator, &.{ state_home, "downloads.json" });
}

fn stateHome(allocator: std.mem.Allocator) ![]u8 {
    if (@import("env_pkg").getenv("TALU_HOME")) |talu_home| {
        return allocator.dupe(u8, std.mem.sliceTo(talu_home, 0));
    }
    const home = try cache.getUserHome(allocator);
    defer allocator.free(home);
    return std.fs.path.join(allocator, &.{ home, ".talu" });
}

fn writeJsonString(writer: anytype, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |ch| {
        switch (ch) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (ch < 0x20) {
                    try writer.print("\\u{x:0>4}", .{ch});
                } else {
                    try writer.writeByte(ch);
                }
            },
        }
    }
    try writer.writeByte('"');
}

fn now() i64 {
    return std.time.timestamp();
}

const EnvFns = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

fn setEnvVar(allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    const key_z = try allocator.dupeZ(u8, key);
    defer allocator.free(key_z);
    const value_z = try allocator.dupeZ(u8, value);
    defer allocator.free(value_z);
    if (EnvFns.setenv(key_z.ptr, value_z.ptr, 1) != 0) return error.Unexpected;
}

fn unsetEnvVar(allocator: std.mem.Allocator, key: []const u8) !void {
    const key_z = try allocator.dupeZ(u8, key);
    defer allocator.free(key_z);
    _ = EnvFns.unsetenv(key_z.ptr);
}

test "download queue state path uses TALU_HOME when set" {
    const allocator = std.testing.allocator;
    const old_talu_home = @import("env_pkg").getenv("TALU_HOME");
    defer {
        if (old_talu_home) |old| {
            setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(old, 0)) catch {};
        } else {
            unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    try setEnvVar(allocator, "TALU_HOME", "/tmp/talu-state-test");
    const path = try stateFilePath(allocator);
    defer allocator.free(path);
    try std.testing.expectEqualStrings("/tmp/talu-state-test/downloads.json", path);
}

test "download queue persists terminal job metadata without token" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const old_talu_home = @import("env_pkg").getenv("TALU_HOME");
    defer {
        if (old_talu_home) |old| {
            setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(old, 0)) catch {};
        } else {
            unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }
    try setEnvVar(allocator, "TALU_HOME", tmp_path);

    var first = Manager.init(allocator);
    const job = try allocator.create(Job);
    job.* = .{
        .id = try allocator.dupe(u8, "repo-download-99"),
        .model_id = try allocator.dupe(u8, "org/model"),
        .token = try allocator.dupe(u8, "hf_secret"),
        .endpoint_url = null,
        .force = false,
        .skip_weights = true,
        .status = .completed,
        .path = try allocator.dupe(u8, "/models/org/model"),
        .err = null,
        .started_at = 10,
        .updated_at = 20,
        .completed_at = 20,
        .label = try allocator.dupe(u8, ""),
        .message = try allocator.dupe(u8, ""),
    };
    try first.jobs.append(allocator, job);
    first.next_id = 100;
    try first.saveLocked();
    first.freeJob(job);
    first.jobs.deinit(allocator);

    const state_path = try stateFilePath(allocator);
    defer allocator.free(state_path);
    const state_bytes = try std.fs.cwd().readFileAlloc(allocator, state_path, 1024 * 1024);
    defer allocator.free(state_bytes);
    try std.testing.expect(std.mem.indexOf(u8, state_bytes, "hf_secret") == null);

    var second = Manager.init(allocator);
    try second.loadLocked();
    defer {
        for (second.jobs.items) |loaded| second.freeJob(loaded);
        second.jobs.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), second.jobs.items.len);
    try std.testing.expectEqual(@as(u64, 100), second.next_id);
    try std.testing.expectEqual(Status.completed, second.jobs.items[0].status);
    try std.testing.expectEqualStrings("org/model", second.jobs.items[0].model_id);
    try std.testing.expect(second.jobs.items[0].token == null);
}

fn testJob(allocator: std.mem.Allocator, id: []const u8, model_id: []const u8, status: Status) !*Job {
    const job = try allocator.create(Job);
    job.* = .{
        .id = try allocator.dupe(u8, id),
        .model_id = try allocator.dupe(u8, model_id),
        .token = null,
        .endpoint_url = null,
        .force = false,
        .skip_weights = false,
        .status = status,
        .started_at = 1,
        .updated_at = 1,
        .completed_at = if (status.terminal()) 1 else null,
        .current = 0,
        .total = 0,
        .label = try allocator.dupe(u8, ""),
        .message = try allocator.dupe(u8, ""),
    };
    return job;
}

test "download queue clearFinished removes only terminal jobs" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const old_talu_home = @import("env_pkg").getenv("TALU_HOME");
    defer {
        if (old_talu_home) |old| {
            setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(old, 0)) catch {};
        } else {
            unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }
    try setEnvVar(allocator, "TALU_HOME", tmp_path);

    var m = Manager.init(allocator);
    m.loaded = true;
    try m.jobs.append(allocator, try testJob(allocator, "j1", "org/model-a", .completed));
    try m.jobs.append(allocator, try testJob(allocator, "j2", "org/model-b", .failed));
    try m.jobs.append(allocator, try testJob(allocator, "j3", "org/model-c", .queued));
    try m.jobs.append(allocator, try testJob(allocator, "j4", "org/model-d", .active));

    const removed = try m.clearFinished();
    defer {
        for (m.jobs.items) |job| m.freeJob(job);
        m.jobs.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), removed);
    try std.testing.expectEqual(@as(usize, 2), m.jobs.items.len);
    try std.testing.expectEqual(Status.queued, m.jobs.items[0].status);
    try std.testing.expectEqual(Status.active, m.jobs.items[1].status);
}

test "download queue cancelAll cancels queued and paused jobs" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(tmp_path);

    const old_talu_home = @import("env_pkg").getenv("TALU_HOME");
    defer {
        if (old_talu_home) |old| {
            setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(old, 0)) catch {};
        } else {
            unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }
    try setEnvVar(allocator, "TALU_HOME", tmp_path);

    var m = Manager.init(allocator);
    m.loaded = true;
    try m.jobs.append(allocator, try testJob(allocator, "j1", "org/model-a", .queued));
    try m.jobs.append(allocator, try testJob(allocator, "j2", "org/model-b", .paused));
    try m.jobs.append(allocator, try testJob(allocator, "j3", "org/model-c", .active));
    try m.jobs.append(allocator, try testJob(allocator, "j4", "org/model-d", .completed));

    const affected = try m.cancelAll();
    defer {
        for (m.jobs.items) |job| m.freeJob(job);
        m.jobs.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), affected);
    try std.testing.expectEqual(Status.cancelled, m.jobs.items[0].status);
    try std.testing.expectEqual(Status.cancelled, m.jobs.items[1].status);
    try std.testing.expectEqual(Status.active, m.jobs.items[2].status);
    try std.testing.expect(@atomicLoad(bool, &m.jobs.items[2].cancel_flag, .monotonic));
}
