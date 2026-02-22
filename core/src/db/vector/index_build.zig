//! IVF index build orchestration helpers using durable core job records.

const std = @import("std");
const db_jobs = @import("../jobs.zig");
const vector_store = @import("store.zig");
const manifest = @import("../manifest.zig");

pub const IndexBuildJobOptions = struct {
    max_retries: u32 = 3,
    expected_generation: ?u64 = null,
    max_segments: usize = 32,
};

pub const AsyncIndexBuildHandle = struct {
    job_id: u64,
    thread: std.Thread,

    pub fn wait(self: *AsyncIndexBuildHandle) void {
        self.thread.join();
    }
};

const AsyncIndexBuildCtx = struct {
    db_root: []u8,
    jobs_path: []u8,
    options: IndexBuildJobOptions,
    job_id: u64,
};

pub fn runIndexBuildJob(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    max_segments: usize,
) !u64 {
    return runIndexBuildJobWithOptions(adapter, jobs, .{
        .max_segments = max_segments,
    });
}

pub fn runIndexBuildJobWithOptions(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    options: IndexBuildJobOptions,
) !u64 {
    const job_id = try jobs.startJob("vector-index-build");
    _ = try executeIndexBuildJob(adapter, jobs, options, job_id);
    return job_id;
}

/// Start index build in a background thread with durable job-state updates.
///
/// The worker opens a fresh VectorAdapter and JobStore from paths so the
/// foreground request path remains non-blocking.
pub fn spawnIndexBuildJob(
    allocator: std.mem.Allocator,
    db_root: []const u8,
    jobs_path: []const u8,
    options: IndexBuildJobOptions,
) !AsyncIndexBuildHandle {
    var jobs = try db_jobs.JobStore.init(allocator, jobs_path);
    defer jobs.deinit();
    const job_id = try jobs.startJob("vector-index-build");
    try jobs.updateJob(job_id, "vector-index-build", .pending, "queued");

    const ctx = try allocator.create(AsyncIndexBuildCtx);
    errdefer allocator.destroy(ctx);
    const db_root_copy = try allocator.dupe(u8, db_root);
    errdefer allocator.free(db_root_copy);
    const jobs_path_copy = try allocator.dupe(u8, jobs_path);
    errdefer allocator.free(jobs_path_copy);
    ctx.* = .{
        .db_root = db_root_copy,
        .jobs_path = jobs_path_copy,
        .options = options,
        .job_id = job_id,
    };
    errdefer {
        allocator.free(ctx.db_root);
        allocator.free(ctx.jobs_path);
    }

    const thread = try std.Thread.spawn(.{}, runAsyncIndexBuildJob, .{ allocator, ctx });
    return .{
        .job_id = job_id,
        .thread = thread,
    };
}

fn runAsyncIndexBuildJob(allocator: std.mem.Allocator, ctx: *AsyncIndexBuildCtx) void {
    defer {
        allocator.free(ctx.db_root);
        allocator.free(ctx.jobs_path);
        allocator.destroy(ctx);
    }

    var adapter = vector_store.VectorAdapter.init(allocator, ctx.db_root) catch return;
    defer adapter.deinit();
    var jobs = db_jobs.JobStore.init(allocator, ctx.jobs_path) catch return;
    defer jobs.deinit();
    _ = executeIndexBuildJob(&adapter, &jobs, ctx.options, ctx.job_id) catch {};
}

fn executeIndexBuildJob(
    adapter: *vector_store.VectorAdapter,
    jobs: *db_jobs.JobStore,
    options: IndexBuildJobOptions,
    job_id: u64,
) !vector_store.IndexBuildResult {
    try adapter.fs_writer.flushBlock();
    _ = try adapter.fs_reader.refreshIfChanged();
    var expected_generation = options.expected_generation orelse adapter.fs_reader.snapshotGeneration();
    var attempt: u32 = 0;

    try jobs.updateJob(job_id, "vector-index-build", .running, "running:attempt=1");

    const result = while (true) {
        const build_result = adapter.buildPendingApproximateIndexesWithExpectedGeneration(
            expected_generation,
            options.max_segments,
        ) catch |err| {
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
                try jobs.updateJob(job_id, "vector-index-build", .running, message);
                continue;
            }

            const message = try std.fmt.allocPrint(adapter.allocator, "failed:{s}", .{@errorName(err)});
            defer adapter.allocator.free(message);
            try jobs.updateJob(job_id, "vector-index-build", .failed, message);
            return err;
        };
        break build_result;
    };

    const message = try std.fmt.allocPrint(
        adapter.allocator,
        "succeeded:attempt={d}:built={d},failed={d},pending={d}",
        .{
            attempt + 1,
            result.built_segments,
            result.failed_segments,
            result.pending_segments,
        },
    );
    defer adapter.allocator.free(message);
    try jobs.updateJob(job_id, "vector-index-build", .succeeded, message);
    return result;
}

test "runIndexBuildJob writes succeeded job state and publishes ready index metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    adapter.fs_writer.max_segment_size = 1;
    try adapter.upsertBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, 2);
    try adapter.fs_writer.flushBlock();
    try adapter.upsertBatch(&[_]u64{2}, &[_]f32{ 0.0, 1.0 }, 2);
    try adapter.fs_writer.flushBlock();

    var jobs = try db_jobs.JobStore.init(std.testing.allocator, jobs_path);
    defer jobs.deinit();

    const job_id = try runIndexBuildJob(&adapter, &jobs, 8);
    const record = jobs.getJob(job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(db_jobs.JobState.succeeded, record.state);

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root, "vector", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    var loaded = try manifest.Manifest.load(std.testing.allocator, manifest_path);
    defer loaded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), loaded.segments.len);
    try std.testing.expect(loaded.segments[0].index != null);
    try std.testing.expectEqual(manifest.SegmentIndexState.ready, loaded.segments[0].index.?.state);
    try std.testing.expect(loaded.segments[0].index.?.checksum_crc32c != 0);
}

test "runIndexBuildJobWithOptions retries after generation conflict and succeeds" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
    defer adapter.deinit();
    adapter.fs_writer.max_segment_size = 1;
    try adapter.upsertBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, 2);
    try adapter.fs_writer.flushBlock();
    try adapter.upsertBatch(&[_]u64{2}, &[_]f32{ 0.0, 1.0 }, 2);
    try adapter.fs_writer.flushBlock();

    try adapter.fs_writer.flushBlock();
    _ = try adapter.fs_reader.refreshIfChanged();
    const current_generation = adapter.fs_reader.snapshotGeneration();

    var jobs = try db_jobs.JobStore.init(std.testing.allocator, jobs_path);
    defer jobs.deinit();

    const job_id = try runIndexBuildJobWithOptions(&adapter, &jobs, .{
        .max_retries = 1,
        .expected_generation = current_generation + 1,
        .max_segments = 8,
    });
    const record = jobs.getJob(job_id) orelse return error.TestUnexpectedResult;
    defer {
        std.testing.allocator.free(record.name);
        std.testing.allocator.free(record.message);
    }
    try std.testing.expectEqual(db_jobs.JobState.succeeded, record.state);
    try std.testing.expect(std.mem.indexOf(u8, record.message, "attempt=2") != null);
}

test "spawnIndexBuildJob runs index build asynchronously and persists terminal state" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const jobs_path = try std.fs.path.join(std.testing.allocator, &.{ root, "jobs.log" });
    defer std.testing.allocator.free(jobs_path);

    {
        var adapter = try vector_store.VectorAdapter.init(std.testing.allocator, root);
        defer adapter.deinit();
        adapter.fs_writer.max_segment_size = 1;
        try adapter.upsertBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, 2);
        try adapter.fs_writer.flushBlock();
        try adapter.upsertBatch(&[_]u64{2}, &[_]f32{ 0.0, 1.0 }, 2);
        try adapter.fs_writer.flushBlock();
    }

    var handle = try spawnIndexBuildJob(std.testing.allocator, root, jobs_path, .{
        .max_segments = 8,
    });
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
