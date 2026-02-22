//! Compaction orchestration helpers using durable core job records.

const std = @import("std");
const db_jobs = @import("../jobs.zig");
const vector_store = @import("store.zig");

pub const CompactionJobOptions = struct {
    max_retries: u32 = 3,
    expected_generation: ?u64 = null,
};

pub const AsyncCompactionHandle = struct {
    job_id: u64,
    thread: std.Thread,

    pub fn wait(self: *AsyncCompactionHandle) void {
        self.thread.join();
    }
};

const AsyncCompactionCtx = struct {
    db_root: []u8,
    jobs_path: []u8,
    dims: u32,
    options: CompactionJobOptions,
    job_id: u64,
};

pub fn runCompactionJob(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    dims: u32,
) !u64 {
    return runCompactionJobWithOptions(adapter, jobs, dims, .{});
}

pub fn runCompactionJobWithOptions(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    dims: u32,
    options: CompactionJobOptions,
) !u64 {
    const job_id = try jobs.startJob("vector-compact");
    _ = try executeCompactionJob(adapter, jobs, dims, options, job_id);
    return job_id;
}

/// Start compaction in a background thread with durable job-state updates.
///
/// The worker opens a fresh VectorAdapter and JobStore from paths so the
/// foreground request path remains non-blocking.
pub fn spawnCompactionJob(
    allocator: std.mem.Allocator,
    db_root: []const u8,
    jobs_path: []const u8,
    dims: u32,
    options: CompactionJobOptions,
) !AsyncCompactionHandle {
    var jobs = try db_jobs.JobStore.init(allocator, jobs_path);
    defer jobs.deinit();
    const job_id = try jobs.startJob("vector-compact");
    try jobs.updateJob(job_id, "vector-compact", .pending, "queued");

    const ctx = try allocator.create(AsyncCompactionCtx);
    errdefer allocator.destroy(ctx);
    const db_root_copy = try allocator.dupe(u8, db_root);
    errdefer allocator.free(db_root_copy);
    const jobs_path_copy = try allocator.dupe(u8, jobs_path);
    errdefer allocator.free(jobs_path_copy);
    ctx.* = .{
        .db_root = db_root_copy,
        .jobs_path = jobs_path_copy,
        .dims = dims,
        .options = options,
        .job_id = job_id,
    };
    errdefer {
        allocator.free(ctx.db_root);
        allocator.free(ctx.jobs_path);
    }

    const thread = try std.Thread.spawn(.{}, runAsyncCompactionJob, .{ allocator, ctx });
    return .{
        .job_id = job_id,
        .thread = thread,
    };
}

fn runAsyncCompactionJob(allocator: std.mem.Allocator, ctx: *AsyncCompactionCtx) void {
    defer {
        allocator.free(ctx.db_root);
        allocator.free(ctx.jobs_path);
        allocator.destroy(ctx);
    }

    var adapter = vector_store.VectorAdapter.init(allocator, ctx.db_root) catch return;
    defer adapter.deinit();
    var jobs = db_jobs.JobStore.init(allocator, ctx.jobs_path) catch return;
    defer jobs.deinit();
    _ = executeCompactionJob(&adapter, &jobs, ctx.dims, ctx.options, ctx.job_id) catch {};
}

fn executeCompactionJob(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    dims: u32,
    options: CompactionJobOptions,
    job_id: u64,
) !vector_store.CompactResult {
    try adapter.fs_writer.flushBlock();
    _ = try adapter.fs_reader.refreshIfChanged();
    var expected_generation = options.expected_generation orelse adapter.fs_reader.snapshotGeneration();
    var attempt: u32 = 0;

    try jobs.updateJob(job_id, "vector-compact", .running, "running:attempt=1");

    const result = while (true) {
        const compact_result = adapter.compactWithExpectedGeneration(dims, expected_generation) catch |err| {
            if (err == error.ManifestGenerationConflict and attempt < options.max_retries) {
                attempt += 1;
                _ = try adapter.fs_reader.refreshIfChanged();
                expected_generation = adapter.fs_reader.snapshotGeneration();
                const message = try std.fmt.allocPrint(
                    adapter.allocator,
                    "running:attempt={d}:rebase_generation={d}",
                    .{ attempt + 1, expected_generation },
                );
                defer adapter.allocator.free(message);
                try jobs.updateJob(job_id, "vector-compact", .running, message);
                continue;
            }

            const message = try std.fmt.allocPrint(adapter.allocator, "failed:{s}", .{@errorName(err)});
            defer adapter.allocator.free(message);
            try jobs.updateJob(job_id, "vector-compact", .failed, message);
            return err;
        };
        break compact_result;
    };

    const message = try std.fmt.allocPrint(
        adapter.allocator,
        "succeeded:attempt={d}:kept={d},removed={d}",
        .{ attempt + 1, result.kept_count, result.removed_tombstones },
    );
    defer adapter.allocator.free(message);
    try jobs.updateJob(job_id, "vector-compact", .succeeded, message);
    return result;
}

test "runCompactionJob writes succeeded job state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, 2);
    _ = try adapter.deleteIds(&[_]u64{2});

    var jobs = try db_jobs.JobStore.init(std.testing.allocator, jobs_path);
    defer jobs.deinit();

    const job_id = try runCompactionJob(&adapter, &jobs, 2);
    const record = jobs.getJob(job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }

    try std.testing.expectEqual(db_jobs.JobState.succeeded, record.state);
}

test "runCompactionJobWithOptions retries after generation conflict and succeeds" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    try adapter.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, 2);
    _ = try adapter.deleteIds(&[_]u64{2});

    try adapter.fs_writer.flushBlock();
    _ = try adapter.fs_reader.refreshIfChanged();
    const current_generation = adapter.fs_reader.snapshotGeneration();

    var jobs = try db_jobs.JobStore.init(std.testing.allocator, jobs_path);
    defer jobs.deinit();

    const job_id = try runCompactionJobWithOptions(&adapter, &jobs, 2, .{
        .max_retries = 1,
        .expected_generation = current_generation + 1,
    });
    const record = jobs.getJob(job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(db_jobs.JobState.succeeded, record.state);
    try std.testing.expect(std.mem.indexOf(u8, record.message, "attempt=2") != null);
}

test "spawnCompactionJob runs compaction asynchronously and persists terminal state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
        defer adapter.deinit();
        try adapter.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
            1.0, 0.0,
            0.0, 1.0,
        }, 2);
        _ = try adapter.deleteIds(&[_]u64{2});
    }

    var handle = try spawnCompactionJob(std.testing.allocator, root, jobs_path, 2, .{});
    handle.wait();

    var jobs = try db_jobs.JobStore.init(std.testing.allocator, jobs_path);
    defer jobs.deinit();
    const record = jobs.getJob(handle.job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(db_jobs.JobState.succeeded, record.state);
}
