//! TaluDB vector adapter for high-throughput embedding storage.
//!
//! Provides batch append and bulk load for schema 10 (embeddings).
//! Thread safety: NOT thread-safe (single-writer semantics via <ns>/talu.lock).

const std = @import("std");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const manifest = @import("../manifest.zig");
const segment_source = @import("../segment_source.zig");
const types = @import("../types.zig");
const vector_filter = @import("filter.zig");
const vector_index = @import("index/root.zig");
const vector_planner = @import("planner.zig");
const vector_cdc = @import("cdc.zig");
const vector_ttl = @import("ttl.zig");
const tensor_mod = @import("../../tensor.zig");
const dot_product = @import("../../compute/cpu/linalg.zig").dot;
const parallel = @import("../../system/parallel.zig");

const Allocator = std.mem.Allocator;

const schema_embeddings: u16 = 10;
const schema_vector_changes: u16 = 16;
const schema_vector_idempotency: u16 = 17;

const col_doc_id: u32 = 1;
const col_ts: u32 = 2;
const col_seq: u32 = 3;
const col_embedding: u32 = 10;
const col_change_op: u32 = 11;
const col_idempotency_key_hash: u32 = 20;
const col_idempotency_request_hash: u32 = 21;
const col_idempotency_op: u32 = 22;
const col_idempotency_status: u32 = 23;
const col_idempotency_result_a: u32 = 24;
const col_idempotency_result_b: u32 = 25;

pub const VectorBatch = struct {
    ids: []u64,
    vectors: []f32,
    dims: u32,

    pub fn deinit(self: *VectorBatch, allocator: Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.vectors);
    }
};

pub const VectorTensorBatch = struct {
    ids: []u64,
    tensor: ?*tensor_mod.Tensor,
    dims: u32,
};

pub const SearchEntry = struct {
    score: f32,
    id: u64,
};

pub const SearchResult = struct {
    ids: []u64,
    scores: []f32,

    pub fn deinit(self: *SearchResult, allocator: Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.scores);
    }
};

pub const SearchBatchResult = struct {
    ids: []u64,
    scores: []f32,
    count_per_query: u32,
    query_count: u32,

    pub fn deinit(self: *SearchBatchResult, allocator: Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.scores);
    }
};

pub const DeleteResult = struct {
    deleted_count: usize,
    not_found_count: usize,
};

pub const FetchResult = struct {
    ids: []u64,
    vectors: []f32,
    missing_ids: []u64,
    dims: u32,
    include_values: bool,

    pub fn deinit(self: *FetchResult, allocator: Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.vectors);
        allocator.free(self.missing_ids);
    }
};

pub const StatsResult = struct {
    visible_count: usize,
    tombstone_count: usize,
    segment_count: usize,
    total_count: usize,
};

pub const CompactResult = struct {
    kept_count: usize,
    removed_tombstones: usize,
};

pub const IdempotencyStatus = enum(u8) {
    pending = 0,
    committed = 1,
};

const IdempotencyRecord = struct {
    seq: u64,
    request_hash: u64,
    op: ChangeOp,
    status: IdempotencyStatus,
    result_a: u64,
    result_b: u64,
};

pub const ChangeOp = enum(u8) {
    append = 1,
    upsert = 2,
    delete = 3,
    compact = 4,
};

pub const ChangeEvent = struct {
    seq: u64,
    op: ChangeOp,
    id: u64,
    timestamp: i64,
};

pub const MutationOptions = struct {
    normalize: bool = false,
    reject_existing: bool = false,
};

pub const SearchOptions = struct {
    normalize_queries: bool = false,
    filter_expr: ?*const vector_filter.FilterExpr = null,
    approximate: bool = false,
};

pub const ChangesResult = struct {
    events: []ChangeEvent,
    has_more: bool,
    next_since: u64,

    pub fn deinit(self: *ChangesResult, allocator: Allocator) void {
        allocator.free(self.events);
    }
};

pub const ScoreCallback = *const fn (ctx: *anyopaque, id: u64, score: f32) void;
pub const ScoreCallbackC = *const fn (ctx: ?*anyopaque, id: u64, score: f32) callconv(.c) void;

const PointVisibility = struct {
    seq: u64,
    deleted: bool,
};

const LatestVectors = struct {
    dims: u32,
    map: std.AutoHashMap(u64, []f32),

    fn deinit(self: *LatestVectors, allocator: Allocator) void {
        var iter = self.map.valueIterator();
        while (iter.next()) |values| {
            allocator.free(values.*);
        }
        self.map.deinit();
    }
};

const ScratchBuffers = struct {
    id_buf: []u8 = &[_]u8{},
    vector_buf: []u8 = &[_]u8{},

    fn deinit(self: *ScratchBuffers, allocator: Allocator) void {
        if (self.id_buf.len > 0) allocator.free(self.id_buf);
        if (self.vector_buf.len > 0) allocator.free(self.vector_buf);
    }

    fn ensureId(self: *ScratchBuffers, allocator: Allocator, len: usize) ![]u8 {
        if (len == 0) return &[_]u8{};
        if (self.id_buf.len >= len) return self.id_buf[0..len];
        if (self.id_buf.len > 0) allocator.free(self.id_buf);
        self.id_buf = try allocator.alloc(u8, len);
        return self.id_buf;
    }

    fn ensureVector(self: *ScratchBuffers, allocator: Allocator, len: usize) ![]u8 {
        if (len == 0) return &[_]u8{};
        if (self.vector_buf.len >= len) return self.vector_buf[0..len];
        if (self.vector_buf.len > 0) allocator.free(self.vector_buf);
        self.vector_buf = try allocator.alloc(u8, len);
        return self.vector_buf;
    }
};

const BlockSnapshots = struct {
    vector_generation: u64,
    change_generation: u64,
    vector_blocks: []db_reader.BlockRef,
    change_blocks: []db_reader.BlockRef,

    fn deinit(self: BlockSnapshots, allocator: Allocator) void {
        allocator.free(self.vector_blocks);
        allocator.free(self.change_blocks);
    }
};

const SearchCandidates = struct {
    ids: []const u64,
    vectors: []const f32,
    owned_ids: ?[]u64 = null,
    owned_vectors: ?[]f32 = null,

    fn deinit(self: *SearchCandidates, allocator: Allocator) void {
        if (self.owned_ids) |owned| allocator.free(owned);
        if (self.owned_vectors) |owned| allocator.free(owned);
        self.* = .{
            .ids = &[_]u64{},
            .vectors = &[_]f32{},
            .owned_ids = null,
            .owned_vectors = null,
        };
    }
};

/// Vector backend API ("vector" namespace).
///
/// Public operations:
/// - `init`: Initialize vector TaluDB backend for the "vector" namespace.
/// - `appendBatch`: Append a batch of embeddings.
/// - `loadVectors`: Load all stored embeddings.
/// - `loadVectorsTensor`: Load all stored embeddings into a Tensor for DLPack export.
/// - `search`: Search stored embeddings with a dot-product scan.
/// - `searchBatch`: Search multiple queries using a dot-product scan.
/// - `searchScores`: Stream scores for every vector to a caller-provided callback.
/// - `searchScoresC`: Stream scores with a C-callable callback.
/// - `scanScoresBatchInto`: Scan scores into caller-provided buffers.
/// - `countEmbeddingRows`: Count stored embedding rows and validate dims.
///
/// Teardown:
/// - `deinit`: Flush pending data, release resources (normal shutdown).
/// - `simulateCrash`: Release resources (fds, flocks) WITHOUT flushing
///   pending data or deleting the WAL file (crash-testing only).
///
/// See also: `create` / `destroy` for heap-allocated lifecycle.
pub const VectorAdapter = struct {
    allocator: Allocator,
    db_root: []u8,
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,
    change_writer: *db_writer.Writer,
    change_reader: *db_reader.Reader,
    idempotency_writer: *db_writer.Writer,
    idempotency_reader: *db_reader.Reader,
    next_seq: u64,
    read_cache_valid: bool,
    cached_vector_block_count: usize,
    cached_change_block_count: usize,
    cached_vector_generation: u64,
    cached_change_generation: u64,
    cached_vector_block_hash: u64,
    cached_change_block_hash: u64,
    cached_visibility: std.AutoHashMap(u64, PointVisibility),
    cached_latest: LatestVectors,
    cached_visible_ids: []u64,
    cached_visible_vectors: []f32,

    /// Initialize vector TaluDB backend for the "vector" namespace.
    pub fn init(allocator: Allocator, db_root: []const u8) !VectorAdapter {
        const db_root_copy = try allocator.dupe(u8, db_root);
        errdefer allocator.free(db_root_copy);

        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "vector");
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "vector");
        errdefer reader_ptr.deinit();

        var change_writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(change_writer_ptr);
        change_writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "vector_changes");
        errdefer change_writer_ptr.deinit();

        var change_reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(change_reader_ptr);
        change_reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "vector_changes");
        errdefer change_reader_ptr.deinit();

        var idempotency_writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(idempotency_writer_ptr);
        idempotency_writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "vector_idempotency");
        errdefer idempotency_writer_ptr.deinit();

        var idempotency_reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(idempotency_reader_ptr);
        idempotency_reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "vector_idempotency");
        errdefer idempotency_reader_ptr.deinit();

        var adapter = VectorAdapter{
            .allocator = allocator,
            .db_root = db_root_copy,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .change_writer = change_writer_ptr,
            .change_reader = change_reader_ptr,
            .idempotency_writer = idempotency_writer_ptr,
            .idempotency_reader = idempotency_reader_ptr,
            .next_seq = 1,
            .read_cache_valid = false,
            .cached_vector_block_count = 0,
            .cached_change_block_count = 0,
            .cached_vector_generation = 0,
            .cached_change_generation = 0,
            .cached_vector_block_hash = 0,
            .cached_change_block_hash = 0,
            .cached_visibility = std.AutoHashMap(u64, PointVisibility).init(allocator),
            .cached_latest = .{
                .dims = 0,
                .map = std.AutoHashMap(u64, []f32).init(allocator),
            },
            .cached_visible_ids = &[_]u64{},
            .cached_visible_vectors = &[_]f32{},
        };
        adapter.next_seq = try adapter.discoverNextSeq();
        return adapter;
    }

    /// Append a batch of embeddings.
    pub fn appendBatch(self: *VectorAdapter, doc_ids: []const u64, vectors: []const f32, dims: u32) !void {
        try self.appendBatchWithOptions(doc_ids, vectors, dims, .{});
    }

    /// Append a batch with explicit mutation options.
    pub fn appendBatchWithOptions(
        self: *VectorAdapter,
        doc_ids: []const u64,
        vectors: []const f32,
        dims: u32,
        options: MutationOptions,
    ) !void {
        try self.appendBatchWithOp(doc_ids, vectors, dims, .append, options);
    }

    /// Append with durable idempotency tracking.
    pub fn appendBatchIdempotentWithOptions(
        self: *VectorAdapter,
        doc_ids: []const u64,
        vectors: []const f32,
        dims: u32,
        options: MutationOptions,
        key_hash: u64,
        request_hash: u64,
    ) !void {
        if (doc_ids.len == 0) return;
        if (vectors.len != doc_ids.len * @as(usize, dims)) return error.InvalidColumnData;

        var vectors_buf = vectors;
        var normalized_vectors: ?[]f32 = null;
        defer if (normalized_vectors) |owned| self.allocator.free(owned);
        if (options.normalize) {
            const owned = try self.allocator.dupe(f32, vectors);
            errdefer self.allocator.free(owned);
            try normalizeRowsInPlace(owned, dims);
            vectors_buf = owned;
            normalized_vectors = owned;
        }

        if (key_hash != 0) {
            if (try self.loadIdempotencyRecord(key_hash, request_hash, .append)) |record| {
                switch (record.status) {
                    .committed => return,
                    .pending => {
                        if (!(try self.allIdsMatchVectors(doc_ids, vectors_buf, dims))) {
                            const pending_options = MutationOptions{
                                .normalize = false,
                                .reject_existing = options.reject_existing,
                            };
                            self.appendBatchWithOp(doc_ids, vectors_buf, dims, .append, pending_options) catch |err| {
                                if (err == error.AlreadyExists and try self.allIdsMatchVectors(doc_ids, vectors_buf, dims)) {
                                    // Prior attempt likely completed before crash.
                                } else {
                                    return err;
                                }
                            };
                        }
                        try self.appendIdempotencyRecord(
                            key_hash,
                            request_hash,
                            .append,
                            .committed,
                            record.result_a,
                            record.result_b,
                        );
                        return;
                    },
                }
            }
        }

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .append,
                .pending,
                @intCast(doc_ids.len),
                0,
            );
        }

        const pending_options = MutationOptions{
            .normalize = false,
            .reject_existing = options.reject_existing,
        };
        self.appendBatchWithOp(doc_ids, vectors_buf, dims, .append, pending_options) catch |err| {
            if (err == error.AlreadyExists and try self.allIdsMatchVectors(doc_ids, vectors_buf, dims)) {
                // Prior attempt likely completed before crash.
            } else {
                return err;
            }
        };

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .append,
                .committed,
                @intCast(doc_ids.len),
                0,
            );
        }
    }

    /// Upsert a batch of embeddings.
    ///
    /// Physical storage remains append-only; upsert visibility is resolved
    /// through the vector change log.
    pub fn upsertBatch(self: *VectorAdapter, doc_ids: []const u64, vectors: []const f32, dims: u32) !void {
        try self.upsertBatchWithOptions(doc_ids, vectors, dims, .{});
    }

    /// Upsert a batch with explicit mutation options.
    pub fn upsertBatchWithOptions(
        self: *VectorAdapter,
        doc_ids: []const u64,
        vectors: []const f32,
        dims: u32,
        options: MutationOptions,
    ) !void {
        try self.appendBatchWithOp(doc_ids, vectors, dims, .upsert, options);
    }

    /// Upsert with durable idempotency tracking.
    pub fn upsertBatchIdempotentWithOptions(
        self: *VectorAdapter,
        doc_ids: []const u64,
        vectors: []const f32,
        dims: u32,
        options: MutationOptions,
        key_hash: u64,
        request_hash: u64,
    ) !void {
        if (doc_ids.len == 0) return;
        if (vectors.len != doc_ids.len * @as(usize, dims)) return error.InvalidColumnData;

        var vectors_buf = vectors;
        var normalized_vectors: ?[]f32 = null;
        defer if (normalized_vectors) |owned| self.allocator.free(owned);
        if (options.normalize) {
            const owned = try self.allocator.dupe(f32, vectors);
            errdefer self.allocator.free(owned);
            try normalizeRowsInPlace(owned, dims);
            vectors_buf = owned;
            normalized_vectors = owned;
        }

        if (key_hash != 0) {
            if (try self.loadIdempotencyRecord(key_hash, request_hash, .upsert)) |record| {
                switch (record.status) {
                    .committed => return,
                    .pending => {
                        if (!(try self.allIdsMatchVectors(doc_ids, vectors_buf, dims))) {
                            try self.appendBatchWithOp(
                                doc_ids,
                                vectors_buf,
                                dims,
                                .upsert,
                                .{ .normalize = false, .reject_existing = false },
                            );
                        }
                        try self.appendIdempotencyRecord(
                            key_hash,
                            request_hash,
                            .upsert,
                            .committed,
                            record.result_a,
                            record.result_b,
                        );
                        return;
                    },
                }
            }
        }

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .upsert,
                .pending,
                @intCast(doc_ids.len),
                0,
            );
        }

        if (!(try self.allIdsMatchVectors(doc_ids, vectors_buf, dims))) {
            try self.appendBatchWithOp(
                doc_ids,
                vectors_buf,
                dims,
                .upsert,
                .{ .normalize = false, .reject_existing = false },
            );
        }

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .upsert,
                .committed,
                @intCast(doc_ids.len),
                0,
            );
        }
    }

    /// Delete vectors by ID using tombstone semantics.
    pub fn deleteIds(self: *VectorAdapter, ids: []const u64) !DeleteResult {
        try self.ensureReadCache();

        var ids_to_delete = std.ArrayList(u64).empty;
        defer ids_to_delete.deinit(self.allocator);
        var seqs_to_delete = std.ArrayList(u64).empty;
        defer seqs_to_delete.deinit(self.allocator);

        var deleted: usize = 0;
        var not_found: usize = 0;
        for (ids) |id| {
            const visible = self.idIsVisibleCached(id);

            if (visible) {
                deleted += 1;
                try ids_to_delete.append(self.allocator, id);
                try seqs_to_delete.append(self.allocator, self.allocateNextSeq());
            } else {
                not_found += 1;
            }
        }

        if (ids_to_delete.items.len > 0) {
            try self.appendChangeRowsForOp(ids_to_delete.items, seqs_to_delete.items, .delete);
            self.invalidateReadCache();
        }

        return .{ .deleted_count = deleted, .not_found_count = not_found };
    }

    /// Delete vectors with durable idempotency tracking.
    pub fn deleteIdsIdempotent(
        self: *VectorAdapter,
        ids: []const u64,
        key_hash: u64,
        request_hash: u64,
    ) !DeleteResult {
        if (key_hash != 0) {
            if (try self.loadIdempotencyRecord(key_hash, request_hash, .delete)) |record| {
                switch (record.status) {
                    .committed => {
                        return .{
                            .deleted_count = @intCast(record.result_a),
                            .not_found_count = @intCast(record.result_b),
                        };
                    },
                    .pending => {
                        _ = try self.deleteIds(ids);
                        try self.appendIdempotencyRecord(
                            key_hash,
                            request_hash,
                            .delete,
                            .committed,
                            record.result_a,
                            record.result_b,
                        );
                        return .{
                            .deleted_count = @intCast(record.result_a),
                            .not_found_count = @intCast(record.result_b),
                        };
                    },
                }
            }
        }

        const preview = try self.previewDeleteCounts(ids);
        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .delete,
                .pending,
                @intCast(preview.deleted_count),
                @intCast(preview.not_found_count),
            );
        }

        _ = try self.deleteIds(ids);

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .delete,
                .committed,
                @intCast(preview.deleted_count),
                @intCast(preview.not_found_count),
            );
        }

        return preview;
    }

    /// Fetch vectors by ID from the current visible state.
    pub fn fetchByIds(self: *VectorAdapter, allocator: Allocator, ids: []const u64, include_values: bool) !FetchResult {
        try self.ensureReadCache();

        var missing_count: usize = 0;
        var found_count: usize = 0;
        for (ids) |id| {
            const visible = self.idIsVisibleCached(id);
            if (visible) {
                found_count += 1;
            } else {
                missing_count += 1;
            }
        }

        const dims_value = self.cached_latest.dims;
        const found_ids = try allocator.alloc(u64, found_count);
        errdefer allocator.free(found_ids);
        const found_vectors = try allocator.alloc(f32, found_count * @as(usize, dims_value));
        errdefer allocator.free(found_vectors);
        const missing_ids = try allocator.alloc(u64, missing_count);
        errdefer allocator.free(missing_ids);

        var found_idx: usize = 0;
        var missing_idx: usize = 0;
        for (ids) |id| {
            const visible = self.idIsVisibleCached(id);
            if (visible) {
                const values = self.cached_latest.map.get(id) orelse {
                    missing_ids[missing_idx] = id;
                    missing_idx += 1;
                    continue;
                };
                found_ids[found_idx] = id;
                if (include_values and dims_value > 0) {
                    const base = found_idx * @as(usize, dims_value);
                    std.mem.copyForwards(f32, found_vectors[base .. base + @as(usize, dims_value)], values);
                }
                found_idx += 1;
            } else {
                missing_ids[missing_idx] = id;
                missing_idx += 1;
            }
        }

        return .{ .ids = found_ids, .vectors = found_vectors, .missing_ids = missing_ids, .dims = dims_value, .include_values = include_values };
    }

    /// Return vector mutation statistics from the change log.
    pub fn stats(self: *VectorAdapter) !StatsResult {
        try self.ensureReadCache();

        var visible: usize = 0;
        var latest_iter = self.cached_latest.map.iterator();
        while (latest_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            if (id == 0) continue;
            if (self.cached_visibility.get(id)) |state| {
                if (!state.deleted) visible += 1;
            } else {
                visible += 1;
            }
        }

        var tombstones: usize = 0;
        var vis_iter = self.cached_visibility.iterator();
        while (vis_iter.next()) |entry| {
            if (entry.key_ptr.* == 0) continue;
            if (entry.value_ptr.deleted) {
                tombstones += 1;
            }
        }

        return .{ .visible_count = visible, .tombstone_count = tombstones, .segment_count = try self.countVectorSegments(), .total_count = visible + tombstones };
    }

    /// Compact vector storage by rebuilding physical segments from visible rows.
    pub fn compact(self: *VectorAdapter, dims: u32) !CompactResult {
        if (dims == 0) return error.InvalidColumnData;
        try self.ensureReadCache();
        if (self.cached_latest.dims != 0 and self.cached_latest.dims != dims) return error.InvalidColumnData;

        var removed_tombstones: usize = 0;
        var vis_iter = self.cached_visibility.iterator();
        while (vis_iter.next()) |entry| {
            if (entry.key_ptr.* == 0) continue;
            if (entry.value_ptr.deleted) removed_tombstones += 1;
        }

        var kept_count: usize = 0;
        var latest_iter = self.cached_latest.map.iterator();
        while (latest_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            if (id == 0) continue;
            if (self.cached_visibility.get(id)) |state| {
                if (!state.deleted) kept_count += 1;
            } else {
                kept_count += 1;
            }
        }

        const ids = try self.allocator.alloc(u64, kept_count);
        defer self.allocator.free(ids);
        const vectors = try self.allocator.alloc(f32, kept_count * @as(usize, dims));
        defer self.allocator.free(vectors);

        var write_idx: usize = 0;
        latest_iter = self.cached_latest.map.iterator();
        while (latest_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            if (id == 0) continue;
            if (self.cached_visibility.get(id)) |state| {
                if (state.deleted) continue;
            }
            ids[write_idx] = id;
            const base = write_idx * @as(usize, dims);
            std.mem.copyForwards(f32, vectors[base .. base + @as(usize, dims)], entry.value_ptr.*);
            write_idx += 1;
        }

        const compact_seq = self.allocateNextSeq();
        const compact_ts = std.time.milliTimestamp();
        const compact_event = ChangeEvent{
            .seq = compact_seq,
            .op = .compact,
            .id = 0,
            .timestamp = compact_ts,
        };

        try self.resetVectorNamespace();
        if (kept_count > 0) {
            try self.appendBatchRaw(ids, vectors, dims);
        }
        try self.appendChangeEvents(&[_]ChangeEvent{compact_event});
        if (kept_count > 0) {
            const compact_seqs = try self.allocator.alloc(u64, kept_count);
            defer self.allocator.free(compact_seqs);
            for (compact_seqs) |*seq| {
                seq.* = self.allocateNextSeq();
            }
            try self.appendChangeRowsForOp(ids, compact_seqs, .upsert);
        }
        self.invalidateReadCache();

        return .{ .kept_count = kept_count, .removed_tombstones = removed_tombstones };
    }

    /// Compact only if the vector manifest generation matches `expected_generation`.
    ///
    /// This protects background jobs from publishing stale compaction output
    /// when another writer already advanced the manifest snapshot.
    pub fn compactWithExpectedGeneration(
        self: *VectorAdapter,
        dims: u32,
        expected_generation: u64,
    ) !CompactResult {
        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const current_generation = self.fs_reader.snapshotGeneration();
        if (current_generation != expected_generation) return error.ManifestGenerationConflict;
        return self.compact(dims);
    }

    /// Compact only when tombstones older than TTL are present.
    pub fn compactExpiredTombstones(
        self: *VectorAdapter,
        dims: u32,
        now_ms: i64,
        max_age_ms: i64,
    ) !CompactResult {
        const changes = try self.loadAllChanges(self.allocator);
        defer self.allocator.free(changes);

        var expired_tombstones: usize = 0;
        for (changes) |change| {
            if (change.op != .delete) continue;
            if (vector_ttl.isExpired(change.timestamp, now_ms, max_age_ms)) {
                expired_tombstones += 1;
            }
        }
        if (!vector_ttl.shouldCompactForTtl(expired_tombstones)) {
            const current = try self.stats();
            return .{
                .kept_count = current.visible_count,
                .removed_tombstones = 0,
            };
        }
        return self.compact(dims);
    }

    /// Compact vectors with durable idempotency tracking.
    pub fn compactIdempotent(
        self: *VectorAdapter,
        dims: u32,
        key_hash: u64,
        request_hash: u64,
    ) !CompactResult {
        if (key_hash != 0) {
            if (try self.loadIdempotencyRecord(key_hash, request_hash, .compact)) |record| {
                switch (record.status) {
                    .committed => {
                        return .{
                            .kept_count = @intCast(record.result_a),
                            .removed_tombstones = @intCast(record.result_b),
                        };
                    },
                    .pending => {
                        _ = try self.compact(dims);
                        try self.appendIdempotencyRecord(
                            key_hash,
                            request_hash,
                            .compact,
                            .committed,
                            record.result_a,
                            record.result_b,
                        );
                        return .{
                            .kept_count = @intCast(record.result_a),
                            .removed_tombstones = @intCast(record.result_b),
                        };
                    },
                }
            }
        }

        const stats_before = try self.stats();
        const expected = CompactResult{
            .kept_count = stats_before.visible_count,
            .removed_tombstones = stats_before.tombstone_count,
        };
        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .compact,
                .pending,
                @intCast(expected.kept_count),
                @intCast(expected.removed_tombstones),
            );
        }

        _ = try self.compact(dims);

        if (key_hash != 0) {
            try self.appendIdempotencyRecord(
                key_hash,
                request_hash,
                .compact,
                .committed,
                @intCast(expected.kept_count),
                @intCast(expected.removed_tombstones),
            );
        }

        return expected;
    }

    /// Read mutation events with cursor pagination.
    pub fn readChanges(self: *VectorAdapter, allocator: Allocator, since: u64, limit: usize) !ChangesResult {
        const all_changes = try self.loadAllChanges(allocator);
        defer allocator.free(all_changes);

        const clamped_limit = @max(@as(usize, 1), @min(limit, 1000));

        var matched_total: usize = 0;
        for (all_changes) |change| {
            if (change.seq > since) {
                matched_total += 1;
            }
        }

        const out_count = @min(matched_total, clamped_limit);
        const out = try allocator.alloc(ChangeEvent, out_count);
        errdefer allocator.free(out);

        var write_idx: usize = 0;
        for (all_changes) |change| {
            if (change.seq <= since) continue;
            if (write_idx >= out_count) break;
            out[write_idx] = change;
            write_idx += 1;
        }

        const has_more = matched_total > out_count;
        const next_since = if (out_count > 0) out[out_count - 1].seq else since;
        return .{ .events = out, .has_more = has_more, .next_since = next_since };
    }

    fn appendBatchWithOp(
        self: *VectorAdapter,
        doc_ids: []const u64,
        vectors: []const f32,
        dims: u32,
        op: ChangeOp,
        options: MutationOptions,
    ) !void {
        if (doc_ids.len == 0) return;
        if (vectors.len != doc_ids.len * @as(usize, dims)) return error.InvalidColumnData;

        var vectors_buf = vectors;
        var normalized_vectors: ?[]f32 = null;
        defer if (normalized_vectors) |owned| self.allocator.free(owned);
        if (options.normalize) {
            const owned = try self.allocator.dupe(f32, vectors);
            errdefer self.allocator.free(owned);
            try normalizeRowsInPlace(owned, dims);
            vectors_buf = owned;
            normalized_vectors = owned;
        }

        if (op == .append and options.reject_existing and try self.anyVisibleIdExists(doc_ids)) {
            return error.AlreadyExists;
        }

        const seqs = try self.allocator.alloc(u64, doc_ids.len);
        defer self.allocator.free(seqs);
        for (seqs) |*seq| {
            seq.* = self.allocateNextSeq();
        }

        try self.appendBatchRaw(doc_ids, vectors_buf, dims);
        try self.appendChangeRowsForOp(doc_ids, seqs, op);
        self.invalidateReadCache();
    }

    fn anyVisibleIdExists(self: *VectorAdapter, ids: []const u64) !bool {
        try self.ensureReadCache();
        for (ids) |id| {
            if (self.idIsVisibleCached(id)) {
                return true;
            }
        }

        return false;
    }

    fn allIdsMatchVectors(self: *VectorAdapter, ids: []const u64, vectors: []const f32, dims: u32) !bool {
        if (ids.len == 0) return true;
        if (vectors.len != ids.len * @as(usize, dims)) return false;

        var unique = std.AutoHashMap(u64, void).init(self.allocator);
        defer unique.deinit();
        for (ids) |id| {
            if (try unique.fetchPut(id, {})) |_| return false;
        }
        try self.ensureReadCache();
        if (self.cached_latest.dims != dims) return false;

        for (ids, 0..) |id, idx| {
            if (!self.idIsVisibleCached(id)) return false;
            const stored = self.cached_latest.map.get(id) orelse return false;
            if (stored.len != @as(usize, dims)) return false;
            const start = idx * @as(usize, dims);
            const expected = vectors[start .. start + @as(usize, dims)];
            if (!std.mem.eql(f32, stored, expected)) return false;
        }

        return true;
    }

    fn previewDeleteCounts(self: *VectorAdapter, ids: []const u64) !DeleteResult {
        try self.ensureReadCache();

        var deleted: usize = 0;
        var not_found: usize = 0;
        for (ids) |id| {
            if (self.idIsVisibleCached(id)) {
                deleted += 1;
            } else {
                not_found += 1;
            }
        }
        return .{ .deleted_count = deleted, .not_found_count = not_found };
    }

    fn idIsVisibleCached(self: *VectorAdapter, id: u64) bool {
        if (self.cached_visibility.get(id)) |entry| {
            if (entry.deleted) return false;
            return self.cached_latest.map.contains(id);
        }
        return self.cached_latest.map.contains(id);
    }

    fn ensureReadCache(self: *VectorAdapter) !void {
        var snapshots = try self.readBlockSnapshots(self.allocator);
        defer snapshots.deinit(self.allocator);

        const vector_count = snapshots.vector_blocks.len;
        const change_count = snapshots.change_blocks.len;
        const vector_generation = snapshots.vector_generation;
        const change_generation = snapshots.change_generation;
        const vector_hash = hashBlockRefs(snapshots.vector_blocks);
        const change_hash = hashBlockRefs(snapshots.change_blocks);

        if (!self.read_cache_valid) {
            try self.refreshReadCacheFromBlocks(
                snapshots.vector_generation,
                snapshots.change_generation,
                snapshots.vector_blocks,
                snapshots.change_blocks,
                vector_hash,
                change_hash,
            );
            return;
        }

        if (vector_count == self.cached_vector_block_count and
            change_count == self.cached_change_block_count and
            vector_generation == self.cached_vector_generation and
            change_generation == self.cached_change_generation and
            vector_hash == self.cached_vector_block_hash and
            change_hash == self.cached_change_block_hash)
        {
            return;
        }

        if (vector_generation != self.cached_vector_generation or
            change_generation != self.cached_change_generation)
        {
            try self.refreshReadCacheFromBlocks(
                snapshots.vector_generation,
                snapshots.change_generation,
                snapshots.vector_blocks,
                snapshots.change_blocks,
                vector_hash,
                change_hash,
            );
            return;
        }

        if (vector_count < self.cached_vector_block_count or
            change_count < self.cached_change_block_count)
        {
            try self.refreshReadCacheFromBlocks(
                snapshots.vector_generation,
                snapshots.change_generation,
                snapshots.vector_blocks,
                snapshots.change_blocks,
                vector_hash,
                change_hash,
            );
            return;
        }

        const vector_is_append_only = blk: {
            if (vector_count == self.cached_vector_block_count) {
                break :blk vector_hash == self.cached_vector_block_hash;
            }
            const prefix_hash = hashBlockRefs(snapshots.vector_blocks[0..self.cached_vector_block_count]);
            break :blk prefix_hash == self.cached_vector_block_hash;
        };
        const change_is_append_only = blk: {
            if (change_count == self.cached_change_block_count) {
                break :blk change_hash == self.cached_change_block_hash;
            }
            const prefix_hash = hashBlockRefs(snapshots.change_blocks[0..self.cached_change_block_count]);
            break :blk prefix_hash == self.cached_change_block_hash;
        };
        if (!vector_is_append_only or !change_is_append_only) {
            try self.refreshReadCacheFromBlocks(
                snapshots.vector_generation,
                snapshots.change_generation,
                snapshots.vector_blocks,
                snapshots.change_blocks,
                vector_hash,
                change_hash,
            );
            return;
        }

        self.read_cache_valid = false;
        const change_needs_full = try self.applyChangeBlocks(snapshots.change_blocks[self.cached_change_block_count..]);
        if (change_needs_full) {
            try self.refreshReadCacheFromBlocks(
                snapshots.vector_generation,
                snapshots.change_generation,
                snapshots.vector_blocks,
                snapshots.change_blocks,
                vector_hash,
                change_hash,
            );
            return;
        }
        try self.applyVectorBlocks(snapshots.vector_blocks[self.cached_vector_block_count..]);
        try self.rebuildVisibleDenseCache();

        self.cached_vector_block_count = vector_count;
        self.cached_change_block_count = change_count;
        self.cached_vector_generation = vector_generation;
        self.cached_change_generation = change_generation;
        self.cached_vector_block_hash = vector_hash;
        self.cached_change_block_hash = change_hash;
        self.read_cache_valid = true;
    }

    fn refreshReadCacheFromBlocks(
        self: *VectorAdapter,
        vector_generation: u64,
        change_generation: u64,
        vector_blocks: []const db_reader.BlockRef,
        change_blocks: []const db_reader.BlockRef,
        vector_hash: u64,
        change_hash: u64,
    ) !void {
        self.read_cache_valid = false;
        var visibility = try self.loadLatestVisibilityFromBlocks(self.allocator, change_blocks);
        errdefer visibility.deinit();
        var latest = try self.loadLatestVectorsFromBlocks(self.allocator, vector_blocks, null);
        errdefer latest.deinit(self.allocator);

        self.cached_visibility.deinit();
        self.cached_latest.deinit(self.allocator);
        self.cached_visibility = visibility;
        self.cached_latest = latest;
        try self.rebuildVisibleDenseCache();
        self.cached_vector_block_count = vector_blocks.len;
        self.cached_change_block_count = change_blocks.len;
        self.cached_vector_generation = vector_generation;
        self.cached_change_generation = change_generation;
        self.cached_vector_block_hash = vector_hash;
        self.cached_change_block_hash = change_hash;
        self.read_cache_valid = true;
    }

    fn invalidateReadCache(self: *VectorAdapter) void {
        self.read_cache_valid = false;
        self.cached_vector_block_count = 0;
        self.cached_change_block_count = 0;
        self.cached_vector_generation = 0;
        self.cached_change_generation = 0;
        self.cached_vector_block_hash = 0;
        self.cached_change_block_hash = 0;
        self.clearVisibleDenseCache();
    }

    fn clearVisibleDenseCache(self: *VectorAdapter) void {
        if (self.cached_visible_ids.len > 0) {
            self.allocator.free(self.cached_visible_ids);
        }
        if (self.cached_visible_vectors.len > 0) {
            self.allocator.free(self.cached_visible_vectors);
        }
        self.cached_visible_ids = &[_]u64{};
        self.cached_visible_vectors = &[_]f32{};
    }

    fn rebuildVisibleDenseCache(self: *VectorAdapter) !void {
        self.clearVisibleDenseCache();

        const dims = self.cached_latest.dims;
        if (dims == 0) return;

        var visible_count: usize = 0;
        var latest_iter = self.cached_latest.map.iterator();
        while (latest_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            if (id == 0) continue;
            if (self.cached_visibility.get(id)) |state| {
                if (state.deleted) continue;
            }
            visible_count += 1;
        }

        if (visible_count == 0) return;

        const ids = try self.allocator.alloc(u64, visible_count);
        errdefer self.allocator.free(ids);
        const vectors = try self.allocator.alloc(f32, visible_count * @as(usize, dims));
        errdefer self.allocator.free(vectors);

        var write_idx: usize = 0;
        latest_iter = self.cached_latest.map.iterator();
        while (latest_iter.next()) |entry| {
            const id = entry.key_ptr.*;
            if (id == 0) continue;
            if (self.cached_visibility.get(id)) |state| {
                if (state.deleted) continue;
            }

            ids[write_idx] = id;
            const base = write_idx * @as(usize, dims);
            std.mem.copyForwards(f32, vectors[base .. base + @as(usize, dims)], entry.value_ptr.*);
            write_idx += 1;
        }

        self.cached_visible_ids = ids;
        self.cached_visible_vectors = vectors;
    }

    fn readBlockSnapshots(self: *VectorAdapter, allocator: Allocator) !BlockSnapshots {
        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const vector_blocks = try self.fs_reader.getBlocks(allocator);
        errdefer allocator.free(vector_blocks);
        const vector_generation = self.fs_reader.snapshotGeneration();

        try self.change_writer.flushBlock();
        _ = try self.change_reader.refreshIfChanged();
        const change_blocks = try self.change_reader.getBlocks(allocator);
        errdefer allocator.free(change_blocks);
        const change_generation = self.change_reader.snapshotGeneration();

        return .{
            .vector_generation = vector_generation,
            .change_generation = change_generation,
            .vector_blocks = vector_blocks,
            .change_blocks = change_blocks,
        };
    }

    fn applyChangeBlocks(self: *VectorAdapter, blocks: []const db_reader.BlockRef) !bool {
        if (blocks.len == 0) return false;

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.change_reader, &current_handle);

        for (blocks) |block| {
            try ensureBlockHandle(self.change_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, self.allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_vector_changes) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer self.allocator.free(descs);

            const seq_desc = findColumn(descs, col_seq) orelse return error.MissingColumn;
            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const op_desc = findColumn(descs, col_change_op) orelse return error.MissingColumn;

            const seq_buf = try self.allocator.alloc(u8, seq_desc.data_len);
            defer self.allocator.free(seq_buf);
            try reader.readColumnDataInto(block.offset, seq_desc, seq_buf);
            _ = try checkedRowCount(header.row_count, seq_buf.len, @sizeOf(u64));

            const id_buf = try self.allocator.alloc(u8, id_desc.data_len);
            defer self.allocator.free(id_buf);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const op_buf = try self.allocator.alloc(u8, op_desc.data_len);
            defer self.allocator.free(op_buf);
            try reader.readColumnDataInto(block.offset, op_desc, op_buf);
            _ = try checkedRowCount(header.row_count, op_buf.len, @sizeOf(u8));

            const seqs = @as([*]const u64, @ptrCast(@alignCast(seq_buf.ptr)))[0..header.row_count];
            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const ops = @as([*]const u8, @ptrCast(@alignCast(op_buf.ptr)))[0..header.row_count];

            for (0..header.row_count) |idx| {
                const op = try parseChangeOpByte(ops[idx]);
                if (op == .compact) return true;

                const id = ids[idx];
                if (id == 0) continue;
                if (self.cached_visibility.get(id)) |existing| {
                    if (existing.seq > seqs[idx]) continue;
                }
                try self.cached_visibility.put(id, .{
                    .seq = seqs[idx],
                    .deleted = (op == .delete),
                });
            }
        }
        return false;
    }

    fn applyVectorBlocks(self: *VectorAdapter, blocks: []const db_reader.BlockRef) !void {
        if (blocks.len == 0) return;

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.fs_reader, &current_handle);
        var scratch = ScratchBuffers{};
        defer scratch.deinit(self.allocator);

        for (blocks) |block| {
            try ensureBlockHandle(self.fs_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, self.allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer self.allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;
            if (embedding_desc.dims == 0) return error.InvalidColumnData;

            if (self.cached_latest.dims == 0) {
                self.cached_latest.dims = embedding_desc.dims;
            } else if (self.cached_latest.dims != embedding_desc.dims) {
                return error.InvalidColumnData;
            }

            const id_buf = try scratch.ensureId(self.allocator, id_desc.data_len);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const vector_buf = try scratch.ensureVector(self.allocator, embedding_desc.data_len);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_buf);
            const vector_len = @as(usize, header.row_count) * @as(usize, embedding_desc.dims);
            if (vector_buf.len != vector_len * @sizeOf(f32)) return error.InvalidColumnData;

            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const vectors = @as([*]const f32, @ptrCast(@alignCast(vector_buf.ptr)))[0..vector_len];

            for (0..header.row_count) |row_idx| {
                const id = ids[row_idx];
                const base = row_idx * @as(usize, embedding_desc.dims);
                const values = try self.allocator.dupe(f32, vectors[base .. base + @as(usize, embedding_desc.dims)]);
                if (try self.cached_latest.map.fetchPut(id, values)) |replaced| {
                    self.allocator.free(replaced.value);
                }
            }
        }
    }

    fn appendBatchRaw(self: *VectorAdapter, doc_ids: []const u64, vectors: []const f32, dims: u32) !void {
        if (dims == 0) return error.InvalidColumnData;
        if (doc_ids.len == 0) return;
        if (vectors.len != doc_ids.len * @as(usize, dims)) return error.InvalidColumnData;
        if (dims > std.math.maxInt(u16)) return error.InvalidColumnData;

        const ts_buf = try self.allocator.alloc(i64, doc_ids.len);
        defer self.allocator.free(ts_buf);
        const ts_value = std.time.milliTimestamp();
        for (ts_buf) |*entry| {
            entry.* = ts_value;
        }

        const columns = [_]db_writer.ColumnBatch{
            .{ .column_id = col_doc_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(doc_ids) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(ts_buf) },
            .{ .column_id = col_embedding, .shape = .VECTOR, .phys_type = .F32, .encoding = .RAW, .dims = @intCast(dims), .data = std.mem.sliceAsBytes(vectors) },
        };

        try self.fs_writer.appendBatch(schema_embeddings, @intCast(doc_ids.len), &columns);
    }

    fn appendChangeRowsForOp(self: *VectorAdapter, ids: []const u64, seqs: []const u64, op: ChangeOp) !void {
        if (ids.len == 0) return;
        if (ids.len != seqs.len) return error.InvalidColumnData;

        const ops = try self.allocator.alloc(u8, ids.len);
        defer self.allocator.free(ops);
        for (ops) |*entry| {
            entry.* = @intFromEnum(op);
        }

        const timestamps = try self.allocator.alloc(i64, ids.len);
        defer self.allocator.free(timestamps);
        const now = std.time.milliTimestamp();
        for (timestamps) |*entry| {
            entry.* = now;
        }

        try self.appendChangeRowsRaw(ids, seqs, ops, timestamps);
    }

    fn appendChangeEvents(self: *VectorAdapter, events: []const ChangeEvent) !void {
        if (events.len == 0) return;

        const ids = try self.allocator.alloc(u64, events.len);
        defer self.allocator.free(ids);
        const seqs = try self.allocator.alloc(u64, events.len);
        defer self.allocator.free(seqs);
        const ops = try self.allocator.alloc(u8, events.len);
        defer self.allocator.free(ops);
        const timestamps = try self.allocator.alloc(i64, events.len);
        defer self.allocator.free(timestamps);

        for (events, 0..) |event, idx| {
            ids[idx] = event.id;
            seqs[idx] = event.seq;
            ops[idx] = @intFromEnum(event.op);
            timestamps[idx] = event.timestamp;
        }

        try self.appendChangeRowsRaw(ids, seqs, ops, timestamps);
    }

    fn appendChangeRowsRaw(self: *VectorAdapter, ids: []const u64, seqs: []const u64, ops: []const u8, timestamps: []const i64) !void {
        if (ids.len == 0) return;
        if (ids.len != seqs.len or ids.len != ops.len or ids.len != timestamps.len) {
            return error.InvalidColumnData;
        }

        const columns = [_]db_writer.ColumnBatch{
            .{ .column_id = col_seq, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(seqs) },
            .{ .column_id = col_doc_id, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(ids) },
            .{ .column_id = col_change_op, .shape = .SCALAR, .phys_type = .U8, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(ops) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(timestamps) },
        };

        try self.change_writer.flushBlock();
        try self.change_writer.appendBatch(schema_vector_changes, @intCast(ids.len), &columns);
    }

    fn loadIdempotencyRecord(
        self: *VectorAdapter,
        key_hash: u64,
        request_hash: u64,
        op: ChangeOp,
    ) !?IdempotencyRecord {
        if (key_hash == 0) return null;
        var records = try self.loadLatestIdempotency(self.allocator);
        defer records.deinit();

        const record = records.get(key_hash) orelse return null;
        if (record.request_hash != request_hash or record.op != op) {
            return error.IdempotencyConflict;
        }
        return record;
    }

    fn appendIdempotencyRecord(
        self: *VectorAdapter,
        key_hash: u64,
        request_hash: u64,
        op: ChangeOp,
        status: IdempotencyStatus,
        result_a: u64,
        result_b: u64,
    ) !void {
        if (key_hash == 0) return;

        const seq = self.allocateNextSeq();
        const timestamp = std.time.milliTimestamp();

        const seqs = [_]u64{seq};
        const keys = [_]u64{key_hash};
        const req_hashes = [_]u64{request_hash};
        const ops = [_]u8{@intFromEnum(op)};
        const statuses = [_]u8{@intFromEnum(status)};
        const results_a = [_]u64{result_a};
        const results_b = [_]u64{result_b};
        const timestamps = [_]i64{timestamp};

        try self.appendIdempotencyRowsRaw(
            &seqs,
            &keys,
            &req_hashes,
            &ops,
            &statuses,
            &results_a,
            &results_b,
            &timestamps,
        );
    }

    fn appendIdempotencyRowsRaw(
        self: *VectorAdapter,
        seqs: []const u64,
        key_hashes: []const u64,
        request_hashes: []const u64,
        ops: []const u8,
        statuses: []const u8,
        result_as: []const u64,
        result_bs: []const u64,
        timestamps: []const i64,
    ) !void {
        if (seqs.len == 0) return;
        if (seqs.len != key_hashes.len or
            seqs.len != request_hashes.len or
            seqs.len != ops.len or
            seqs.len != statuses.len or
            seqs.len != result_as.len or
            seqs.len != result_bs.len or
            seqs.len != timestamps.len)
        {
            return error.InvalidColumnData;
        }

        const columns = [_]db_writer.ColumnBatch{
            .{ .column_id = col_seq, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(seqs) },
            .{ .column_id = col_idempotency_key_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(key_hashes) },
            .{ .column_id = col_idempotency_request_hash, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(request_hashes) },
            .{ .column_id = col_idempotency_op, .shape = .SCALAR, .phys_type = .U8, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(ops) },
            .{ .column_id = col_idempotency_status, .shape = .SCALAR, .phys_type = .U8, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(statuses) },
            .{ .column_id = col_idempotency_result_a, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(result_as) },
            .{ .column_id = col_idempotency_result_b, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(result_bs) },
            .{ .column_id = col_ts, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.sliceAsBytes(timestamps) },
        };

        try self.idempotency_writer.flushBlock();
        try self.idempotency_writer.appendBatch(schema_vector_idempotency, @intCast(seqs.len), &columns);
    }

    fn loadLatestIdempotency(self: *VectorAdapter, allocator: Allocator) !std.AutoHashMap(u64, IdempotencyRecord) {
        try self.idempotency_writer.flushBlock();
        _ = try self.idempotency_reader.refreshIfChanged();
        const blocks = try self.idempotency_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var records = std.AutoHashMap(u64, IdempotencyRecord).init(allocator);
        errdefer records.deinit();

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.idempotency_reader, &current_handle);

        for (blocks) |block| {
            try ensureBlockHandle(self.idempotency_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_vector_idempotency) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const seq_desc = findColumn(descs, col_seq) orelse return error.MissingColumn;
            const key_desc = findColumn(descs, col_idempotency_key_hash) orelse return error.MissingColumn;
            const req_hash_desc = findColumn(descs, col_idempotency_request_hash) orelse return error.MissingColumn;
            const op_desc = findColumn(descs, col_idempotency_op) orelse return error.MissingColumn;
            const status_desc = findColumn(descs, col_idempotency_status) orelse return error.MissingColumn;
            const result_a_desc = findColumn(descs, col_idempotency_result_a) orelse return error.MissingColumn;
            const result_b_desc = findColumn(descs, col_idempotency_result_b) orelse return error.MissingColumn;

            const seq_buf = try allocator.alloc(u8, seq_desc.data_len);
            defer allocator.free(seq_buf);
            try reader.readColumnDataInto(block.offset, seq_desc, seq_buf);
            _ = try checkedRowCount(header.row_count, seq_buf.len, @sizeOf(u64));

            const key_buf = try allocator.alloc(u8, key_desc.data_len);
            defer allocator.free(key_buf);
            try reader.readColumnDataInto(block.offset, key_desc, key_buf);
            _ = try checkedRowCount(header.row_count, key_buf.len, @sizeOf(u64));

            const req_hash_buf = try allocator.alloc(u8, req_hash_desc.data_len);
            defer allocator.free(req_hash_buf);
            try reader.readColumnDataInto(block.offset, req_hash_desc, req_hash_buf);
            _ = try checkedRowCount(header.row_count, req_hash_buf.len, @sizeOf(u64));

            const op_buf = try allocator.alloc(u8, op_desc.data_len);
            defer allocator.free(op_buf);
            try reader.readColumnDataInto(block.offset, op_desc, op_buf);
            _ = try checkedRowCount(header.row_count, op_buf.len, @sizeOf(u8));

            const status_buf = try allocator.alloc(u8, status_desc.data_len);
            defer allocator.free(status_buf);
            try reader.readColumnDataInto(block.offset, status_desc, status_buf);
            _ = try checkedRowCount(header.row_count, status_buf.len, @sizeOf(u8));

            const result_a_buf = try allocator.alloc(u8, result_a_desc.data_len);
            defer allocator.free(result_a_buf);
            try reader.readColumnDataInto(block.offset, result_a_desc, result_a_buf);
            _ = try checkedRowCount(header.row_count, result_a_buf.len, @sizeOf(u64));

            const result_b_buf = try allocator.alloc(u8, result_b_desc.data_len);
            defer allocator.free(result_b_buf);
            try reader.readColumnDataInto(block.offset, result_b_desc, result_b_buf);
            _ = try checkedRowCount(header.row_count, result_b_buf.len, @sizeOf(u64));

            const seqs = @as([*]const u64, @ptrCast(@alignCast(seq_buf.ptr)))[0..header.row_count];
            const keys = @as([*]const u64, @ptrCast(@alignCast(key_buf.ptr)))[0..header.row_count];
            const request_hashes = @as([*]const u64, @ptrCast(@alignCast(req_hash_buf.ptr)))[0..header.row_count];
            const ops = @as([*]const u8, @ptrCast(@alignCast(op_buf.ptr)))[0..header.row_count];
            const statuses = @as([*]const u8, @ptrCast(@alignCast(status_buf.ptr)))[0..header.row_count];
            const results_a = @as([*]const u64, @ptrCast(@alignCast(result_a_buf.ptr)))[0..header.row_count];
            const results_b = @as([*]const u64, @ptrCast(@alignCast(result_b_buf.ptr)))[0..header.row_count];

            for (0..header.row_count) |idx| {
                const key_hash = keys[idx];
                const seq = seqs[idx];
                const record = IdempotencyRecord{
                    .seq = seq,
                    .request_hash = request_hashes[idx],
                    .op = try parseChangeOpByte(ops[idx]),
                    .status = try parseIdempotencyStatusByte(statuses[idx]),
                    .result_a = results_a[idx],
                    .result_b = results_b[idx],
                };

                if (records.get(key_hash)) |existing| {
                    if (existing.seq > seq) continue;
                }
                try records.put(key_hash, record);
            }
        }

        return records;
    }

    fn loadAllChanges(self: *VectorAdapter, allocator: Allocator) ![]ChangeEvent {
        var snapshots = try self.readBlockSnapshots(allocator);
        defer snapshots.deinit(allocator);
        return self.loadAllChangesFromBlocks(allocator, snapshots.change_blocks);
    }

    fn loadAllChangesFromBlocks(
        self: *VectorAdapter,
        allocator: Allocator,
        blocks: []const db_reader.BlockRef,
    ) ![]ChangeEvent {
        var events = std.ArrayList(ChangeEvent).empty;
        errdefer events.deinit(allocator);

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.change_reader, &current_handle);

        for (blocks) |block| {
            try ensureBlockHandle(self.change_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_vector_changes) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const seq_desc = findColumn(descs, col_seq) orelse return error.MissingColumn;
            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const op_desc = findColumn(descs, col_change_op) orelse return error.MissingColumn;
            const ts_desc = findColumn(descs, col_ts) orelse return error.MissingColumn;

            const seq_buf = try allocator.alloc(u8, seq_desc.data_len);
            defer allocator.free(seq_buf);
            try reader.readColumnDataInto(block.offset, seq_desc, seq_buf);
            _ = try checkedRowCount(header.row_count, seq_buf.len, @sizeOf(u64));

            const id_buf = try allocator.alloc(u8, id_desc.data_len);
            defer allocator.free(id_buf);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const op_buf = try allocator.alloc(u8, op_desc.data_len);
            defer allocator.free(op_buf);
            try reader.readColumnDataInto(block.offset, op_desc, op_buf);
            _ = try checkedRowCount(header.row_count, op_buf.len, @sizeOf(u8));

            const ts_buf = try allocator.alloc(u8, ts_desc.data_len);
            defer allocator.free(ts_buf);
            try reader.readColumnDataInto(block.offset, ts_desc, ts_buf);
            _ = try checkedRowCount(header.row_count, ts_buf.len, @sizeOf(i64));

            const seqs = @as([*]const u64, @ptrCast(@alignCast(seq_buf.ptr)))[0..header.row_count];
            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const ops = @as([*]const u8, @ptrCast(@alignCast(op_buf.ptr)))[0..header.row_count];
            const timestamps = @as([*]const i64, @ptrCast(@alignCast(ts_buf.ptr)))[0..header.row_count];

            for (0..header.row_count) |idx| {
                try events.append(allocator, .{
                    .seq = seqs[idx],
                    .op = try parseChangeOpByte(ops[idx]),
                    .id = ids[idx],
                    .timestamp = timestamps[idx],
                });
            }
        }

        std.sort.pdq(ChangeEvent, events.items, {}, struct {
            fn lessThan(_: void, a: ChangeEvent, b: ChangeEvent) bool {
                return a.seq < b.seq;
            }
        }.lessThan);

        return events.toOwnedSlice(allocator);
    }

    fn buildVisibilityMap(self: *VectorAdapter, allocator: Allocator, changes: []const ChangeEvent) !std.AutoHashMap(u64, PointVisibility) {
        _ = self;
        var visibility = std.AutoHashMap(u64, PointVisibility).init(allocator);
        errdefer visibility.deinit();
        for (changes) |change| {
            if (change.op == .compact) {
                visibility.clearRetainingCapacity();
                continue;
            }
            if (change.id == 0) continue;
            try visibility.put(change.id, .{
                .seq = change.seq,
                .deleted = (change.op == .delete),
            });
        }
        return visibility;
    }

    fn loadLatestVisibility(self: *VectorAdapter, allocator: Allocator) !std.AutoHashMap(u64, PointVisibility) {
        var snapshots = try self.readBlockSnapshots(allocator);
        defer snapshots.deinit(allocator);
        return self.loadLatestVisibilityFromBlocks(allocator, snapshots.change_blocks);
    }

    fn loadLatestVisibilityFromBlocks(
        self: *VectorAdapter,
        allocator: Allocator,
        blocks: []const db_reader.BlockRef,
    ) !std.AutoHashMap(u64, PointVisibility) {
        const changes = try self.loadAllChangesFromBlocks(allocator, blocks);
        defer allocator.free(changes);
        return self.buildVisibilityMap(allocator, changes);
    }

    fn loadLatestVectors(self: *VectorAdapter, allocator: Allocator, only_ids: ?*const std.AutoHashMap(u64, void)) !LatestVectors {
        var snapshots = try self.readBlockSnapshots(allocator);
        defer snapshots.deinit(allocator);
        return self.loadLatestVectorsFromBlocks(allocator, snapshots.vector_blocks, only_ids);
    }

    fn loadLatestVectorsFromBlocks(
        self: *VectorAdapter,
        allocator: Allocator,
        blocks: []const db_reader.BlockRef,
        only_ids: ?*const std.AutoHashMap(u64, void),
    ) !LatestVectors {
        var latest = LatestVectors{
            .dims = 0,
            .map = std.AutoHashMap(u64, []f32).init(allocator),
        };
        errdefer latest.deinit(allocator);

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.fs_reader, &current_handle);
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        for (blocks) |block| {
            try ensureBlockHandle(self.fs_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;
            if (embedding_desc.dims == 0) return error.InvalidColumnData;

            if (latest.dims == 0) {
                latest.dims = embedding_desc.dims;
            } else if (latest.dims != embedding_desc.dims) {
                return error.InvalidColumnData;
            }

            const id_buf = try scratch.ensureId(allocator, id_desc.data_len);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const vector_buf = try scratch.ensureVector(allocator, embedding_desc.data_len);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_buf);
            const vector_len = @as(usize, header.row_count) * @as(usize, embedding_desc.dims);
            if (vector_buf.len != vector_len * @sizeOf(f32)) return error.InvalidColumnData;

            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const vectors = @as([*]const f32, @ptrCast(@alignCast(vector_buf.ptr)))[0..vector_len];

            for (0..header.row_count) |row_idx| {
                const id = ids[row_idx];
                if (only_ids) |subset| {
                    if (!subset.contains(id)) continue;
                }

                const base = row_idx * @as(usize, embedding_desc.dims);
                const values = try allocator.dupe(f32, vectors[base .. base + @as(usize, embedding_desc.dims)]);
                if (try latest.map.fetchPut(id, values)) |replaced| {
                    allocator.free(replaced.value);
                }
            }
        }

        return latest;
    }

    fn discoverNextSeq(self: *VectorAdapter) !u64 {
        var max_seq: u64 = 0;
        max_seq = @max(max_seq, try self.maxSeqInChangeLog(self.allocator));
        max_seq = @max(max_seq, try self.maxSeqInIdempotencyLog(self.allocator));

        if (max_seq == 0) return 1;
        if (max_seq == std.math.maxInt(u64)) return std.math.maxInt(u64);
        return max_seq + 1;
    }

    fn maxSeqInChangeLog(self: *VectorAdapter, allocator: Allocator) !u64 {
        try self.change_writer.flushBlock();
        _ = try self.change_reader.refreshIfChanged();
        const blocks = try self.change_reader.getBlocks(allocator);
        defer allocator.free(blocks);
        return self.maxSeqFromBlocks(allocator, self.change_reader, blocks, schema_vector_changes);
    }

    fn maxSeqInIdempotencyLog(self: *VectorAdapter, allocator: Allocator) !u64 {
        try self.idempotency_writer.flushBlock();
        _ = try self.idempotency_reader.refreshIfChanged();
        const blocks = try self.idempotency_reader.getBlocks(allocator);
        defer allocator.free(blocks);
        return self.maxSeqFromBlocks(allocator, self.idempotency_reader, blocks, schema_vector_idempotency);
    }

    fn maxSeqFromBlocks(
        self: *VectorAdapter,
        allocator: Allocator,
        reader_owner: *db_reader.Reader,
        blocks: []const db_reader.BlockRef,
        schema_id: u16,
    ) !u64 {
        _ = self;
        var max_seq: u64 = 0;
        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(reader_owner, &current_handle);

        for (blocks) |block| {
            try ensureBlockHandle(reader_owner, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_id or header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);
            const seq_desc = findColumn(descs, col_seq) orelse return error.MissingColumn;

            const seq_buf = try allocator.alloc(u8, seq_desc.data_len);
            defer allocator.free(seq_buf);
            try reader.readColumnDataInto(block.offset, seq_desc, seq_buf);
            _ = try checkedRowCount(header.row_count, seq_buf.len, @sizeOf(u64));

            const seqs = @as([*]const u64, @ptrCast(@alignCast(seq_buf.ptr)))[0..header.row_count];
            for (seqs) |seq| {
                max_seq = @max(max_seq, seq);
            }
        }

        return max_seq;
    }

    fn allocateNextSeq(self: *VectorAdapter) u64 {
        const seq = self.next_seq;
        self.next_seq = if (seq == std.math.maxInt(u64)) std.math.maxInt(u64) else seq + 1;
        return seq;
    }

    fn resetVectorNamespace(self: *VectorAdapter) !void {
        self.invalidateReadCache();
        self.fs_writer.flushBlock() catch {};
        self.fs_writer.deinit();
        self.allocator.destroy(self.fs_writer);
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);

        const vector_dir = try self.vectorDirPath(self.allocator);
        defer self.allocator.free(vector_dir);
        try std.fs.cwd().deleteTree(vector_dir);

        self.fs_writer = try self.allocator.create(db_writer.Writer);
        errdefer self.allocator.destroy(self.fs_writer);
        self.fs_writer.* = try db_writer.Writer.open(self.allocator, self.db_root, "vector");

        self.fs_reader = try self.allocator.create(db_reader.Reader);
        errdefer self.allocator.destroy(self.fs_reader);
        self.fs_reader.* = try db_reader.Reader.open(self.allocator, self.db_root, "vector");
    }

    fn vectorDirPath(self: *VectorAdapter, allocator: Allocator) ![]u8 {
        return std.fs.path.join(allocator, &.{ self.db_root, "vector" });
    }

    fn countVectorSegments(self: *VectorAdapter) !usize {
        const vector_dir = try self.vectorDirPath(self.allocator);
        defer self.allocator.free(vector_dir);

        var dir = std.fs.cwd().openDir(vector_dir, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return 0,
            else => return err,
        };
        defer dir.close();

        var iter = dir.iterate();
        var count: usize = 0;
        while (try iter.next()) |entry| {
            if (entry.kind != .file) continue;
            if (std.mem.endsWith(u8, entry.name, ".talu")) {
                count += 1;
            }
        }
        return count;
    }

    fn rebuildVectorNamespace(self: *VectorAdapter, ids: []const u64, vectors: []const f32, dims: u32) !void {
        try self.resetVectorNamespace();
        if (ids.len == 0) return;
        try self.appendBatchRaw(ids, vectors, dims);
    }

    fn parseChangeOpByte(op: u8) !ChangeOp {
        return switch (op) {
            1 => .append,
            2 => .upsert,
            3 => .delete,
            4 => .compact,
            else => error.InvalidColumnData,
        };
    }

    fn parseIdempotencyStatusByte(status: u8) !IdempotencyStatus {
        return switch (status) {
            0 => .pending,
            1 => .committed,
            else => error.InvalidColumnData,
        };
    }

    /// Load all stored embeddings.
    pub fn loadVectors(self: *VectorAdapter, allocator: Allocator) !VectorBatch {
        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);
        var total_rows: usize = 0;
        var dims: ?u32 = null;

        for (blocks) |block| {
            var handle = try self.fs_reader.openBlockReadOnly(block.path);
            defer self.fs_reader.closeBlock(&handle);
            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;
            if (embedding_desc.dims == 0) return error.InvalidColumnData;
            if (dims) |existing| {
                if (existing != embedding_desc.dims) return error.InvalidColumnData;
            } else {
                dims = embedding_desc.dims;
            }
            total_rows += header.row_count;
        }

        const dims_value = dims orelse 0;
        const ids = try allocator.alloc(u64, total_rows);
        errdefer allocator.free(ids);
        const vectors = try allocator.alloc(f32, total_rows * @as(usize, dims_value));
        errdefer allocator.free(vectors);

        if (total_rows == 0 or dims_value == 0) {
            return .{ .ids = ids, .vectors = vectors, .dims = 0 };
        }

        var row_offset: usize = 0;
        var vector_offset: usize = 0;

        for (blocks) |block| {
            var handle = try self.fs_reader.openBlockReadOnly(block.path);
            defer self.fs_reader.closeBlock(&handle);
            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;

            if (embedding_desc.dims != dims_value) return error.InvalidColumnData;

            const id_dest = std.mem.sliceAsBytes(ids[row_offset .. row_offset + header.row_count]);
            try reader.readColumnDataInto(block.offset, id_desc, id_dest);
            _ = try checkedRowCount(header.row_count, id_dest.len, @sizeOf(u64));

            const vector_len = @as(usize, header.row_count) * @as(usize, dims_value);
            const vector_dest = std.mem.sliceAsBytes(vectors[vector_offset .. vector_offset + vector_len]);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_dest);

            row_offset += header.row_count;
            vector_offset += vector_len;
        }

        return .{
            .ids = ids,
            .vectors = vectors,
            .dims = dims_value,
        };
    }

    /// Load all stored embeddings into a Tensor for DLPack export.
    pub fn loadVectorsTensor(self: *VectorAdapter, allocator: Allocator) !VectorTensorBatch {
        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var total_rows: usize = 0;
        var dims: ?u32 = null;

        for (blocks) |block| {
            var handle = try self.fs_reader.openBlockReadOnly(block.path);
            defer self.fs_reader.closeBlock(&handle);
            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;
            if (embedding_desc.dims == 0) return error.InvalidColumnData;
            if (dims) |existing| {
                if (existing != embedding_desc.dims) return error.InvalidColumnData;
            } else {
                dims = embedding_desc.dims;
            }
            total_rows += header.row_count;
        }

        const dims_value = dims orelse 0;
        const ids = try allocator.alloc(u64, total_rows);
        errdefer allocator.free(ids);

        if (total_rows == 0 or dims_value == 0) {
            return .{
                .ids = ids,
                .tensor = null,
                .dims = 0,
            };
        }

        const shape = [_]i64{ @intCast(total_rows), @intCast(dims_value) };
        const tensor_handle = try tensor_mod.Tensor.init(allocator, &shape, .f32, tensor_mod.Device.cpu());
        errdefer tensor_handle.deinit(allocator);

        const tensor_slice = tensor_handle.asSlice(f32);
        var row_offset: usize = 0;
        var vector_offset: usize = 0;

        for (blocks) |block| {
            var handle = try self.fs_reader.openBlockReadOnly(block.path);
            defer self.fs_reader.closeBlock(&handle);
            const reader = block_reader.BlockReader.init(handle.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;

            if (embedding_desc.dims != dims_value) return error.InvalidColumnData;

            const id_dest = std.mem.sliceAsBytes(ids[row_offset .. row_offset + header.row_count]);
            try reader.readColumnDataInto(block.offset, id_desc, id_dest);
            _ = try checkedRowCount(header.row_count, id_dest.len, @sizeOf(u64));

            const vector_len = @as(usize, header.row_count) * @as(usize, dims_value);
            const vector_dest = std.mem.sliceAsBytes(tensor_slice[vector_offset .. vector_offset + vector_len]);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_dest);

            row_offset += header.row_count;
            vector_offset += vector_len;
        }

        return .{
            .ids = ids,
            .tensor = tensor_handle,
            .dims = dims_value,
        };
    }

    /// Search stored embeddings with a dot-product scan.
    pub fn search(
        self: *VectorAdapter,
        allocator: Allocator,
        query: []const f32,
        k: u32,
    ) !SearchResult {
        return self.searchWithOptions(allocator, query, k, .{});
    }

    /// Search stored embeddings with a dot-product scan and explicit options.
    pub fn searchWithOptions(
        self: *VectorAdapter,
        allocator: Allocator,
        query: []const f32,
        k: u32,
        options: SearchOptions,
    ) !SearchResult {
        if (query.len == 0) return error.InvalidColumnData;
        if (query.len > std.math.maxInt(u32)) return error.InvalidColumnData;
        const batch = try self.searchBatchWithOptions(allocator, query, @intCast(query.len), 1, k, options);
        return .{
            .ids = batch.ids,
            .scores = batch.scores,
        };
    }

    /// Stream scores for every vector to a caller-provided callback.
    pub fn searchScores(
        self: *VectorAdapter,
        allocator: Allocator,
        query: []const f32,
        ctx: *anyopaque,
        callback: ScoreCallback,
    ) !void {
        if (query.len == 0) return error.InvalidColumnData;

        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.fs_reader, &current_handle);
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        for (blocks) |block| {
            try ensureBlockHandle(self.fs_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;

            if (embedding_desc.dims == 0) return error.InvalidColumnData;
            const dims_u32: u32 = embedding_desc.dims;
            if (@as(usize, dims_u32) != query.len) return error.InvalidColumnData;

            const id_buf = try scratch.ensureId(allocator, id_desc.data_len);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const vector_buf = try scratch.ensureVector(allocator, embedding_desc.data_len);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_buf);

            const vector_len = @as(usize, header.row_count) * @as(usize, dims_u32);
            if (vector_buf.len != vector_len * @sizeOf(f32)) return error.InvalidColumnData;

            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const vectors = @as([*]const f32, @ptrCast(@alignCast(vector_buf.ptr)))[0..vector_len];

            const scores = try allocator.alloc(f32, header.row_count);
            defer allocator.free(scores);
            parallelScoreRows(query, vectors, @as(usize, dims_u32), scores);

            for (0..header.row_count) |row_idx| {
                callback(ctx, ids[row_idx], scores[row_idx]);
            }
        }
    }

    /// Search scores with a C-callable callback.
    pub fn searchScoresC(
        self: *VectorAdapter,
        allocator: Allocator,
        query: []const f32,
        ctx: ?*anyopaque,
        callback: ScoreCallbackC,
    ) !void {
        const wrapped = struct {
            fn run(raw_ctx: *anyopaque, id: u64, score: f32) void {
                const pair: *const struct {
                    ctx: ?*anyopaque,
                    cb: ScoreCallbackC,
                } = @ptrCast(@alignCast(raw_ctx));
                pair.cb(pair.ctx, id, score);
            }
        }.run;

        var pair = .{ .ctx = ctx, .cb = callback };
        try self.searchScores(allocator, query, &pair, wrapped);
    }

    /// Count stored embedding rows and validate dims.
    pub fn countEmbeddingRows(self: *VectorAdapter, allocator: Allocator, dims: u32) !usize {
        if (dims == 0) return error.InvalidColumnData;

        try self.fs_writer.flushBlock();
        _ = try self.fs_reader.refreshIfChanged();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var total_rows: usize = 0;
        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.fs_reader, &current_handle);

        for (blocks) |block| {
            try ensureBlockHandle(self.fs_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;
            if (embedding_desc.dims == 0) return error.InvalidColumnData;
            if (embedding_desc.dims != dims) return error.InvalidColumnData;

            total_rows += header.row_count;
        }

        return total_rows;
    }

    /// Scan scores into caller-provided buffers.
    pub fn scanScoresBatchInto(
        self: *VectorAdapter,
        allocator: Allocator,
        queries: []const f32,
        dims: u32,
        query_count: u32,
        ids_out: []u64,
        scores_out: []f32,
    ) !usize {
        if (dims == 0) return error.InvalidColumnData;
        if (query_count == 0) return 0;
        if (queries.len != @as(usize, dims) * @as(usize, query_count)) {
            return error.InvalidColumnData;
        }

        const total_rows = try self.countEmbeddingRows(allocator, dims);
        if (ids_out.len < total_rows) return error.InvalidColumnData;
        const total_scores = total_rows * @as(usize, query_count);
        if (scores_out.len < total_scores) return error.InvalidColumnData;

        if (total_rows == 0) return 0;

        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var current_path: ?[]const u8 = null;
        var current_handle: ?segment_source.SegmentHandle = null;
        defer closeCurrentBlockHandle(self.fs_reader, &current_handle);
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        var row_offset: usize = 0;
        const dims_usize = @as(usize, dims);

        for (blocks) |block| {
            try ensureBlockHandle(self.fs_reader, &current_path, &current_handle, block.path);
            const reader = block_reader.BlockReader.init(current_handle.?.file, allocator);
            const header = try reader.readHeader(block.offset);
            if (header.schema_id != schema_embeddings) continue;
            if (header.row_count == 0) continue;

            const descs = try reader.readColumnDirectory(header, block.offset);
            defer allocator.free(descs);

            const id_desc = findColumn(descs, col_doc_id) orelse return error.MissingColumn;
            const embedding_desc = findColumn(descs, col_embedding) orelse return error.MissingColumn;

            if (embedding_desc.dims == 0) return error.InvalidColumnData;
            if (embedding_desc.dims != dims) return error.InvalidColumnData;

            const id_buf = try scratch.ensureId(allocator, id_desc.data_len);
            try reader.readColumnDataInto(block.offset, id_desc, id_buf);
            _ = try checkedRowCount(header.row_count, id_buf.len, @sizeOf(u64));

            const vector_buf = try scratch.ensureVector(allocator, embedding_desc.data_len);
            try reader.readColumnDataInto(block.offset, embedding_desc, vector_buf);

            const vector_len = @as(usize, header.row_count) * dims_usize;
            if (vector_buf.len != vector_len * @sizeOf(f32)) return error.InvalidColumnData;

            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const vectors = @as([*]const f32, @ptrCast(@alignCast(vector_buf.ptr)))[0..vector_len];

            // Copy IDs for this block.
            for (0..header.row_count) |row_idx| {
                ids_out[row_offset + row_idx] = ids[row_idx];
            }

            // Parallel multi-query scoring for this block.
            parallelScoreBatchRows(queries, vectors, dims_usize, query_count, header.row_count, total_rows, row_offset, scores_out);

            row_offset += header.row_count;
        }

        return total_rows;
    }

    /// Search multiple queries using a dot-product scan.
    pub fn searchBatch(
        self: *VectorAdapter,
        allocator: Allocator,
        queries: []const f32,
        dims: u32,
        query_count: u32,
        k: u32,
    ) !SearchBatchResult {
        return self.searchBatchWithOptions(allocator, queries, dims, query_count, k, .{});
    }

    /// Search multiple queries using a dot-product scan with explicit options.
    pub fn searchBatchWithOptions(
        self: *VectorAdapter,
        allocator: Allocator,
        queries: []const f32,
        dims: u32,
        query_count: u32,
        k: u32,
        options: SearchOptions,
    ) !SearchBatchResult {
        if (query_count == 0 or k == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }
        if (dims == 0) return error.InvalidColumnData;
        if (queries.len != @as(usize, dims) * @as(usize, query_count)) {
            return error.InvalidColumnData;
        }
        var plan = try vector_planner.buildSearchPlan(k, query_count, options.filter_expr, options.approximate);

        var query_buf = queries;
        var normalized_queries: ?[]f32 = null;
        defer if (normalized_queries) |owned| allocator.free(owned);
        if (options.normalize_queries) {
            const owned = try allocator.dupe(f32, queries);
            errdefer allocator.free(owned);
            try normalizeRowsInPlace(owned, dims);
            query_buf = owned;
            normalized_queries = owned;
        }

        try self.ensureReadCache();

        if (plan.index_kind == .ivf_flat and !self.canUseApproximateIndex()) {
            plan.index_kind = .flat;
        }

        if (self.cached_latest.dims == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }
        if (self.cached_latest.dims != dims) return error.InvalidColumnData;
        var candidates = try self.selectSearchCandidates(allocator, dims, plan, options.filter_expr);
        defer candidates.deinit(allocator);

        const row_count = candidates.ids.len;
        if (row_count == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
                .count_per_query = 0,
                .query_count = query_count,
            };
        }

        const index = switch (plan.index_kind) {
            .flat => vector_index.VectorIndex{ .flat = .{} },
            .ivf_flat => vector_index.VectorIndex{ .ivf_flat = .{} },
        };
        const index_result = try index.searchBatch(
            allocator,
            .{
                .ids = candidates.ids,
                .vectors = candidates.vectors,
                .dims = dims,
            },
            .{
                .queries = query_buf,
                .query_count = query_count,
                .k = k,
                .dims = dims,
            },
        );

        return .{
            .ids = index_result.ids,
            .scores = index_result.scores,
            .count_per_query = index_result.count_per_query,
            .query_count = index_result.query_count,
        };
    }

    fn selectSearchCandidates(
        self: *VectorAdapter,
        allocator: Allocator,
        dims: u32,
        plan: vector_planner.QueryPlan,
        filter_expr: ?*const vector_filter.FilterExpr,
    ) !SearchCandidates {
        _ = dims;
        if (plan.filter_mode == .none) {
            return .{
                .ids = self.cached_visible_ids,
                .vectors = self.cached_visible_vectors,
            };
        }

        var allow = try vector_filter.evaluateAllowList(allocator, filter_expr, self.cached_visible_ids);
        defer vector_filter.deinitAllowList(allocator, &allow);

        const kept = vector_filter.allowListCount(&allow);
        if (kept == 0) {
            return .{
                .ids = &[_]u64{},
                .vectors = &[_]f32{},
            };
        }

        const dims_usize = @as(usize, self.cached_latest.dims);
        const ids = try allocator.alloc(u64, kept);
        errdefer allocator.free(ids);
        const vectors = try allocator.alloc(f32, kept * dims_usize);
        errdefer allocator.free(vectors);

        var write_idx: usize = 0;
        for (self.cached_visible_ids, 0..) |id, idx| {
            if (!vector_filter.allowListContains(&allow, idx)) continue;
            ids[write_idx] = id;
            const src_base = idx * dims_usize;
            const dst_base = write_idx * dims_usize;
            std.mem.copyForwards(
                f32,
                vectors[dst_base .. dst_base + dims_usize],
                self.cached_visible_vectors[src_base .. src_base + dims_usize],
            );
            write_idx += 1;
        }

        return .{
            .ids = ids,
            .vectors = vectors,
            .owned_ids = ids,
            .owned_vectors = vectors,
        };
    }

    fn canUseApproximateIndex(self: *VectorAdapter) bool {
        var snapshot = self.fs_reader.loadManifestSnapshot(self.allocator) catch return false;
        defer snapshot.deinit(self.allocator);
        if (snapshot.segments.len == 0) return false;

        for (snapshot.segments) |segment| {
            const index_meta = segment.index orelse return false;
            if (index_meta.kind != manifest.SegmentIndexKind.ivf_flat) return false;
            if (index_meta.path.len == 0) return false;

            const ns_path = std.fmt.allocPrint(self.allocator, "vector/{s}", .{index_meta.path}) catch return false;
            defer self.allocator.free(ns_path);

            var handle = self.fs_reader.openBlockReadOnly(ns_path) catch return false;
            defer self.fs_reader.closeBlock(&handle);

            const actual = crc32cFile(handle.file, handle.size) catch return false;
            if (actual != index_meta.checksum_crc32c) return false;
        }
        return true;
    }

    /// Release resources.
    pub fn deinit(self: *VectorAdapter) void {
        self.clearVisibleDenseCache();
        self.cached_visibility.deinit();
        self.cached_latest.deinit(self.allocator);
        self.fs_writer.flushBlock() catch {};
        self.fs_writer.deinit();
        self.allocator.destroy(self.fs_writer);
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
        self.change_writer.flushBlock() catch {};
        self.change_writer.deinit();
        self.allocator.destroy(self.change_writer);
        self.change_reader.deinit();
        self.allocator.destroy(self.change_reader);
        self.idempotency_writer.flushBlock() catch {};
        self.idempotency_writer.deinit();
        self.allocator.destroy(self.idempotency_writer);
        self.idempotency_reader.deinit();
        self.allocator.destroy(self.idempotency_reader);
        self.allocator.free(self.db_root);
    }

    /// Simulates a process crash for testing.
    ///
    /// Releases all resources (closing fds, releasing flocks) WITHOUT
    /// flushing pending data or deleting the WAL file.
    pub fn simulateCrash(self: *VectorAdapter) void {
        self.clearVisibleDenseCache();
        self.cached_visibility.deinit();
        self.cached_latest.deinit(self.allocator);
        self.fs_writer.simulateCrash();
        self.allocator.destroy(self.fs_writer);
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
        self.change_writer.simulateCrash();
        self.allocator.destroy(self.change_writer);
        self.change_reader.deinit();
        self.allocator.destroy(self.change_reader);
        self.idempotency_writer.simulateCrash();
        self.allocator.destroy(self.idempotency_writer);
        self.idempotency_reader.deinit();
        self.allocator.destroy(self.idempotency_reader);
        self.allocator.free(self.db_root);
    }
};

/// Allocate and initialize a vector backend.
pub fn create(allocator: Allocator, db_root: []const u8) !*VectorAdapter {
    const backend = try allocator.create(VectorAdapter);
    errdefer allocator.destroy(backend);
    backend.* = try VectorAdapter.init(allocator, db_root);
    return backend;
}

/// Destroy a vector backend allocated with create().
pub fn destroy(allocator: Allocator, backend: *VectorAdapter) void {
    backend.deinit();
    allocator.destroy(backend);
}

fn findColumn(descs: []const types.ColumnDesc, column_id: u32) ?types.ColumnDesc {
    for (descs) |desc| {
        if (desc.column_id == column_id) return desc;
    }
    return null;
}

fn checkedRowCount(row_count: u32, data_len: usize, value_size: usize) !usize {
    const expected = @as(usize, row_count) * value_size;
    if (expected != data_len) return error.InvalidColumnData;
    return @as(usize, row_count);
}

fn hashBlockRefs(blocks: []const db_reader.BlockRef) u64 {
    var hasher = std.hash.Wyhash.init(0);
    for (blocks) |block| {
        hasher.update(block.path);
        hasher.update(&[_]u8{0});
        hasher.update(std.mem.asBytes(&block.offset));
    }
    return hasher.final();
}

fn crc32cFile(file: std.fs.File, len: u64) !u32 {
    var crc = std.hash.crc.Crc32Iscsi.init();
    var buf: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < len) {
        const remaining = len - offset;
        const chunk_len: usize = @intCast(@min(remaining, buf.len));
        const chunk = buf[0..chunk_len];
        const read_len = try file.preadAll(chunk, offset);
        if (read_len != chunk_len) return error.UnexpectedEof;
        crc.update(chunk);
        offset += @as(u64, @intCast(chunk_len));
    }
    return crc.final();
}

fn closeCurrentBlockHandle(reader_owner: *db_reader.Reader, current_handle: *?segment_source.SegmentHandle) void {
    if (current_handle.*) |*handle| {
        reader_owner.closeBlock(handle);
        current_handle.* = null;
    }
}

fn ensureBlockHandle(
    reader_owner: *db_reader.Reader,
    current_path: *?[]const u8,
    current_handle: *?segment_source.SegmentHandle,
    path: []const u8,
) !void {
    if (current_path.* != null and std.mem.eql(u8, current_path.*.?, path)) {
        return;
    }
    closeCurrentBlockHandle(reader_owner, current_handle);
    current_handle.* = try reader_owner.openBlockReadOnly(path);
    current_path.* = path;
}

// =============================================================================
// Parallel vector scoring
// =============================================================================

/// Context for single-query parallel scoring.
const ScoreScanCtx = struct {
    query: []const f32,
    vectors: []const f32,
    dims: usize,
    scores: []f32,
};

/// Worker: compute dot products for rows [start..end) into disjoint score slots.
fn scoreScanWorker(start: usize, end: usize, ctx: *ScoreScanCtx) void {
    for (start..end) |row_idx| {
        const base = row_idx * ctx.dims;
        ctx.scores[row_idx] = dot_product.dotProductF32(
            ctx.query,
            ctx.vectors[base .. base + ctx.dims],
        );
    }
}

/// Parallel dot-product scoring of all rows against a single query.
/// Each thread writes to disjoint positions in `scores` — no synchronization.
fn parallelScoreRows(query: []const f32, vectors: []const f32, dims: usize, scores: []f32) void {
    var ctx = ScoreScanCtx{
        .query = query,
        .vectors = vectors,
        .dims = dims,
        .scores = scores,
    };
    parallel.global().parallelFor(scores.len, scoreScanWorker, &ctx);
}

/// Context for multi-query parallel scoring.
const BatchScoreScanCtx = struct {
    queries: []const f32,
    vectors: []const f32,
    dims: usize,
    query_count: u32,
    row_count: usize,
    total_rows: usize,
    row_offset: usize,
    scores_out: []f32,
};

/// Worker: score rows [start..end) against all queries, writing into scores_out.
/// Layout: scores_out[query_idx * total_rows + row_offset + row_idx].
fn batchScoreScanWorker(start: usize, end: usize, ctx: *BatchScoreScanCtx) void {
    const dims = ctx.dims;
    const query_count = @as(usize, ctx.query_count);
    for (start..end) |row_idx| {
        const base = row_idx * dims;
        const row_slice = ctx.vectors[base .. base + dims];

        var qi: usize = 0;
        while (qi + 4 <= query_count) : (qi += 4) {
            const q0 = ctx.queries[(qi + 0) * dims .. (qi + 1) * dims];
            const q1 = ctx.queries[(qi + 1) * dims .. (qi + 2) * dims];
            const q2 = ctx.queries[(qi + 2) * dims .. (qi + 3) * dims];
            const q3 = ctx.queries[(qi + 3) * dims .. (qi + 4) * dims];
            const scores = dot_product.dotProductF32Batch4(q0, q1, q2, q3, row_slice);
            const global_row = ctx.row_offset + row_idx;
            ctx.scores_out[(qi + 0) * ctx.total_rows + global_row] = scores[0];
            ctx.scores_out[(qi + 1) * ctx.total_rows + global_row] = scores[1];
            ctx.scores_out[(qi + 2) * ctx.total_rows + global_row] = scores[2];
            ctx.scores_out[(qi + 3) * ctx.total_rows + global_row] = scores[3];
        }
        while (qi < query_count) : (qi += 1) {
            const q_base = qi * dims;
            const score = dot_product.dotProductF32(ctx.queries[q_base .. q_base + dims], row_slice);
            ctx.scores_out[qi * ctx.total_rows + ctx.row_offset + row_idx] = score;
        }
    }
}

/// Parallel multi-query scoring of block rows.
/// Distributes rows across threads; each thread scores its rows against all queries.
fn parallelScoreBatchRows(
    queries: []const f32,
    vectors: []const f32,
    dims: usize,
    query_count: u32,
    row_count: usize,
    total_rows: usize,
    row_offset: usize,
    scores_out: []f32,
) void {
    var ctx = BatchScoreScanCtx{
        .queries = queries,
        .vectors = vectors,
        .dims = dims,
        .query_count = query_count,
        .row_count = row_count,
        .total_rows = total_rows,
        .row_offset = row_offset,
        .scores_out = scores_out,
    };
    parallel.global().parallelFor(row_count, batchScoreScanWorker, &ctx);
}

fn computeScores(query: []const f32, vectors: []const f32, dims: u32, out_scores: []f32) !void {
    if (dims == 0) return error.InvalidColumnData;
    const dims_usize = @as(usize, dims);
    if (query.len != dims_usize) return error.InvalidColumnData;
    if (vectors.len % dims_usize != 0) return error.InvalidColumnData;
    const row_count = vectors.len / dims_usize;
    if (out_scores.len != row_count) return error.InvalidColumnData;

    parallelScoreRows(query, vectors, dims_usize, out_scores);
}

fn normalizeRowsInPlace(values: []f32, dims: u32) !void {
    if (dims == 0) return error.InvalidColumnData;
    const dims_usize = @as(usize, dims);
    if (values.len % dims_usize != 0) return error.InvalidColumnData;

    var row_idx: usize = 0;
    const row_count = values.len / dims_usize;
    while (row_idx < row_count) : (row_idx += 1) {
        const base = row_idx * dims_usize;
        const row = values[base .. base + dims_usize];
        var norm_sq: f32 = 0.0;
        for (row) |v| norm_sq += v * v;
        if (norm_sq == 0.0) return error.ZeroVectorNotAllowed;
        const inv_norm = 1.0 / @sqrt(norm_sq);
        for (row) |*v| {
            v.* *= inv_norm;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "VectorAdapter.appendBatch and loadVectors roundtrip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const count: usize = 1000;
    const dims: u32 = 128;
    const ids = try std.testing.allocator.alloc(u64, count);
    defer std.testing.allocator.free(ids);
    const vectors = try std.testing.allocator.alloc(f32, count * @as(usize, dims));
    defer std.testing.allocator.free(vectors);

    for (ids, 0..) |*id, idx| {
        id.* = @intCast(idx);
    }

    for (0..count) |row_idx| {
        for (0..dims) |dim_idx| {
            const offset = row_idx * @as(usize, dims) + dim_idx;
            vectors[offset] = @as(f32, @floatFromInt(row_idx)) + @as(f32, @floatFromInt(dim_idx)) * 0.25;
        }
    }

    try backend.appendBatch(ids, vectors, dims);
    try backend.fs_writer.flushBlock();

    var batch = try backend.loadVectors(std.testing.allocator);
    defer batch.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, count), batch.ids.len);
    try std.testing.expectEqual(@as(usize, count * @as(usize, dims)), batch.vectors.len);
    try std.testing.expectEqual(dims, batch.dims);

    for (0..count) |row_idx| {
        try std.testing.expectEqual(@as(u64, row_idx), batch.ids[row_idx]);
    }

    for (0..count) |row_idx| {
        for (0..dims) |dim_idx| {
            const offset = row_idx * @as(usize, dims) + dim_idx;
            const expected = @as(f32, @floatFromInt(row_idx)) + @as(f32, @floatFromInt(dim_idx)) * 0.25;
            try std.testing.expectApproxEqAbs(expected, batch.vectors[offset], 0.0);
        }
    }
}

test "VectorAdapter.loadVectorsTensor roundtrip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const count: usize = 4;
    const dims: u32 = 3;
    const ids = try std.testing.allocator.alloc(u64, count);
    defer std.testing.allocator.free(ids);
    const vectors = try std.testing.allocator.alloc(f32, count * @as(usize, dims));
    defer std.testing.allocator.free(vectors);

    for (ids, 0..) |*id, idx| {
        id.* = @intCast(100 + idx);
    }

    for (0..count) |row_idx| {
        for (0..dims) |dim_idx| {
            const offset = row_idx * @as(usize, dims) + dim_idx;
            vectors[offset] = @as(f32, @floatFromInt(row_idx)) * 0.5 + @as(f32, @floatFromInt(dim_idx));
        }
    }

    try backend.appendBatch(ids, vectors, dims);

    const batch = try backend.loadVectorsTensor(std.testing.allocator);
    defer std.testing.allocator.free(batch.ids);
    defer if (batch.tensor) |tensor| tensor.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, count), batch.ids.len);
    try std.testing.expectEqual(dims, batch.dims);
    try std.testing.expect(batch.tensor != null);

    for (0..count) |row_idx| {
        try std.testing.expectEqual(@as(u64, 100 + row_idx), batch.ids[row_idx]);
    }

    const tensor = batch.tensor.?;
    const tensor_slice = tensor.asSlice(f32);
    try std.testing.expectEqual(count * @as(usize, dims), tensor_slice.len);

    for (0..count) |row_idx| {
        for (0..dims) |dim_idx| {
            const offset = row_idx * @as(usize, dims) + dim_idx;
            const expected = @as(f32, @floatFromInt(row_idx)) * 0.5 + @as(f32, @floatFromInt(dim_idx));
            try std.testing.expectApproxEqAbs(expected, tensor_slice[offset], 0.0);
        }
    }
}

test "VectorAdapter.loadVectors includes sealed segment rows after rotation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    backend.fs_writer.max_segment_size = 1;

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, dims);
    try backend.fs_writer.flushBlock();

    try backend.appendBatch(&[_]u64{2}, &[_]f32{ 0.0, 1.0 }, dims);
    try backend.fs_writer.flushBlock();

    var batch = try backend.loadVectors(std.testing.allocator);
    defer batch.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), batch.ids.len);
    try std.testing.expectEqual(dims, batch.dims);

    var saw_one = false;
    var saw_two = false;
    for (batch.ids) |id| {
        if (id == 1) saw_one = true;
        if (id == 2) saw_two = true;
    }
    try std.testing.expect(saw_one);
    try std.testing.expect(saw_two);

    const row_count = try backend.countEmbeddingRows(std.testing.allocator, dims);
    try std.testing.expectEqual(@as(usize, 2), row_count);
}

test "VectorAdapter.loadIdempotencyRecord sees sealed segment rows after rotation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    backend.idempotency_writer.max_segment_size = 1;

    const key_a: u64 = 0xa1;
    const request_a: u64 = 0xb1;
    const key_b: u64 = 0xa2;
    const request_b: u64 = 0xb2;

    try backend.appendIdempotencyRecord(key_a, request_a, .append, .pending, 1, 0);
    try backend.appendIdempotencyRecord(key_a, request_a, .append, .committed, 1, 0);
    try backend.appendIdempotencyRecord(key_b, request_b, .append, .pending, 1, 0);
    try backend.appendIdempotencyRecord(key_b, request_b, .append, .committed, 1, 0);

    const record = try backend.loadIdempotencyRecord(key_a, request_a, .append);
    try std.testing.expect(record != null);
    try std.testing.expectEqual(IdempotencyStatus.committed, record.?.status);
    try std.testing.expectEqual(@as(u64, 1), record.?.result_a);
    try std.testing.expectEqual(@as(u64, 0), record.?.result_b);
}

test "VectorAdapter.searchScores streams scores" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const ids = [_]u64{ 10, 11 };
    const vectors = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
    };
    try backend.appendBatch(&ids, &vectors, 2);

    const query = [_]f32{ 1.0, 0.0 };

    var collected = std.ArrayList(SearchEntry).empty;
    defer collected.deinit(std.testing.allocator);

    const ctx = &collected;
    const callback = struct {
        fn onScore(raw_ctx: *anyopaque, id: u64, score: f32) void {
            const list: *std.ArrayList(SearchEntry) = @ptrCast(@alignCast(raw_ctx));
            list.append(std.testing.allocator, .{ .score = score, .id = id }) catch {};
        }
    }.onScore;

    try backend.searchScores(std.testing.allocator, &query, ctx, callback);

    try std.testing.expectEqual(@as(usize, 2), collected.items.len);
    try std.testing.expectEqual(@as(u64, 10), collected.items[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), collected.items[0].score, 0.0);
    try std.testing.expectEqual(@as(u64, 11), collected.items[1].id);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), collected.items[1].score, 0.0);
}

test "VectorAdapter.searchBatch returns visible latest vectors only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});
    try backend.upsertBatch(&[_]u64{1}, &[_]f32{
        0.5, 0.5,
    }, dims);

    var result = try backend.searchBatch(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, dims, 1, 5);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), result.query_count);
    try std.testing.expectEqual(@as(u32, 1), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result.scores[0], 1e-6);
}

test "VectorAdapter.search returns visible latest vectors only" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{7}, &[_]f32{
        1.0, 0.0,
    }, dims);
    try backend.upsertBatch(&[_]u64{7}, &[_]f32{
        0.0, 1.0,
    }, dims);

    var latest = try backend.search(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, 1);
    defer latest.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), latest.ids.len);
    try std.testing.expectEqual(@as(u64, 7), latest.ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), latest.scores[0], 1e-6);

    _ = try backend.deleteIds(&[_]u64{7});
    var deleted = try backend.search(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, 1);
    defer deleted.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), deleted.ids.len);
    try std.testing.expectEqual(@as(usize, 0), deleted.scores.len);
}

test "VectorAdapter.ensureReadCache applies incremental delta blocks" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{1}, &[_]f32{
        1.0, 0.0,
    }, dims);

    var warm = try backend.searchBatch(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, dims, 1, 4);
    defer warm.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), warm.count_per_query);
    try std.testing.expectEqual(@as(usize, 1), warm.ids.len);
    try std.testing.expectEqual(@as(u64, 1), warm.ids[0]);
    try std.testing.expectEqual(@as(usize, 1), backend.cached_visible_ids.len);
    try std.testing.expectEqual(@as(usize, 2), backend.cached_visible_vectors.len);

    const cached_vector_blocks_before = backend.cached_vector_block_count;
    const cached_change_blocks_before = backend.cached_change_block_count;
    try std.testing.expect(backend.read_cache_valid);

    try backend.upsertBatch(&[_]u64{2}, &[_]f32{
        0.0, 1.0,
    }, dims);
    try std.testing.expect(!backend.read_cache_valid);

    // Simulate a stale-but-initialized reader to force incremental refresh.
    backend.read_cache_valid = true;
    backend.cached_vector_block_count = cached_vector_blocks_before;
    backend.cached_change_block_count = cached_change_blocks_before;

    var refreshed = try backend.searchBatch(std.testing.allocator, &[_]f32{ 0.0, 1.0 }, dims, 1, 4);
    defer refreshed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), refreshed.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), refreshed.ids.len);
    try std.testing.expectEqual(@as(u64, 2), refreshed.ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), refreshed.scores[0], 1e-6);
    try std.testing.expectEqual(@as(usize, 2), backend.cached_visible_ids.len);
    try std.testing.expectEqual(@as(usize, 4), backend.cached_visible_vectors.len);

    try std.testing.expect(backend.cached_vector_block_count > cached_vector_blocks_before);
    try std.testing.expect(backend.cached_change_block_count > cached_change_blocks_before);
}

test "VectorAdapter.ensureReadCache refreshes when block hash differs at same count" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{1}, &[_]f32{
        1.0, 0.0,
    }, dims);

    var warm = try backend.searchBatch(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, dims, 1, 2);
    defer warm.deinit(std.testing.allocator);
    try std.testing.expect(backend.read_cache_valid);

    var snapshots = try backend.readBlockSnapshots(std.testing.allocator);
    defer snapshots.deinit(std.testing.allocator);
    const expected_vector_hash = hashBlockRefs(snapshots.vector_blocks);
    const expected_change_hash = hashBlockRefs(snapshots.change_blocks);

    backend.cached_vector_block_hash ^= 0x9e3779b97f4a7c15;
    backend.cached_change_block_hash ^= 0xc2b2ae3d27d4eb4f;

    var refreshed = try backend.searchBatch(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, dims, 1, 2);
    defer refreshed.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), refreshed.count_per_query);
    try std.testing.expectEqual(@as(usize, 1), refreshed.ids.len);
    try std.testing.expectEqual(@as(u64, 1), refreshed.ids[0]);
    try std.testing.expectEqual(expected_vector_hash, backend.cached_vector_block_hash);
    try std.testing.expectEqual(expected_change_hash, backend.cached_change_block_hash);
}

test "VectorAdapter.appendBatchWithOptions rejects existing visible IDs" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, dims);

    try std.testing.expectError(
        error.AlreadyExists,
        backend.appendBatchWithOptions(&[_]u64{1}, &[_]f32{ 0.0, 1.0 }, dims, .{ .reject_existing = true }),
    );
}

test "VectorAdapter.searchBatchWithOptions normalizes queries" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatchWithOptions(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims, .{ .normalize = true });

    var result = try backend.searchBatchWithOptions(
        std.testing.allocator,
        &[_]f32{ 2.0, 0.0 },
        dims,
        1,
        1,
        .{
            .normalize_queries = true,
        },
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.scores[0], 1e-6);
}

test "VectorAdapter.searchBatchWithOptions applies filter pre-allowlist" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{ 1, 2, 3 }, &[_]f32{
        1.0, 0.0,
        0.8, 0.0,
        0.7, 0.0,
    }, dims);

    const filter_expr = vector_filter.FilterExpr{ .id_in = &[_]u64{ 2, 3 } };
    var result = try backend.searchBatchWithOptions(
        std.testing.allocator,
        &[_]f32{ 1.0, 0.0 },
        dims,
        1,
        2,
        .{ .filter_expr = &filter_expr },
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 2), result.ids[0]);
    try std.testing.expectEqual(@as(u64, 3), result.ids[1]);
}

test "VectorAdapter.searchBatchWithOptions supports approximate planner path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{ 1, 2, 3 }, &[_]f32{
        1.0, 0.0,
        0.8, 0.0,
        0.2, 0.0,
    }, dims);

    var result = try backend.searchBatchWithOptions(
        std.testing.allocator,
        &[_]f32{ 1.0, 0.0 },
        dims,
        1,
        2,
        .{ .approximate = true },
    );
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
}

test "VectorAdapter.searchBatchWithOptions falls back to flat when segment index checksum mismatches" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    backend.fs_writer.max_segment_size = 1;
    const dims: u32 = 2;
    try backend.upsertBatch(&[_]u64{1}, &[_]f32{ 1.0, 0.0 }, dims);
    try backend.fs_writer.flushBlock();
    try backend.upsertBatch(&[_]u64{ 2, 3 }, &[_]f32{
        0.9, 0.0,
        0.2, 0.0,
    }, dims);
    try backend.fs_writer.flushBlock();

    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "vector", "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    var loaded_manifest = try manifest.Manifest.load(std.testing.allocator, manifest_path);
    defer loaded_manifest.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), loaded_manifest.segments.len);
    try std.testing.expect(loaded_manifest.segments[0].index != null);
    const index_rel = loaded_manifest.segments[0].index.?.path;

    const index_path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "vector", index_rel });
    defer std.testing.allocator.free(index_path);
    var index_file = try std.fs.cwd().createFile(index_path, .{ .truncate = true });
    defer index_file.close();
    try index_file.writeAll("corrupt-index");

    var result = try backend.searchBatchWithOptions(
        std.testing.allocator,
        &[_]f32{ 1.0, 0.0 },
        dims,
        1,
        2,
        .{ .approximate = true },
    );
    defer result.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 2), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 2), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
    try std.testing.expectEqual(@as(u64, 2), result.ids[1]);
}

test "VectorAdapter.searchBatch uses deterministic id tie-break for equal scores" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    // Equal vectors with equal scores; smaller id should win.
    try backend.upsertBatch(&[_]u64{ 2, 1 }, &[_]f32{
        1.0, 0.0,
        1.0, 0.0,
    }, dims);

    var result = try backend.searchBatch(std.testing.allocator, &[_]f32{ 1.0, 0.0 }, dims, 1, 1);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), result.count_per_query);
    try std.testing.expectEqual(@as(usize, 1), result.ids.len);
    try std.testing.expectEqual(@as(u64, 1), result.ids[0]);
}

test "VectorAdapter.appendBatchIdempotentWithOptions replays and conflicts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    const key_hash: u64 = 0xabc1;
    const request_hash: u64 = 0x1111;
    try backend.appendBatchIdempotentWithOptions(
        &[_]u64{1},
        &[_]f32{ 1.0, 0.0 },
        dims,
        .{ .reject_existing = true },
        key_hash,
        request_hash,
    );
    try backend.appendBatchIdempotentWithOptions(
        &[_]u64{1},
        &[_]f32{ 1.0, 0.0 },
        dims,
        .{ .reject_existing = true },
        key_hash,
        request_hash,
    );

    try std.testing.expectError(
        error.IdempotencyConflict,
        backend.appendBatchIdempotentWithOptions(
            &[_]u64{1},
            &[_]f32{ 1.0, 0.0 },
            dims,
            .{ .reject_existing = true },
            key_hash,
            0x2222,
        ),
    );

    var changes = try backend.readChanges(std.testing.allocator, 0, 20);
    defer changes.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), changes.events.len);
    try std.testing.expectEqual(ChangeOp.append, changes.events[0].op);
}

test "VectorAdapter.upsertBatchIdempotentWithOptions replays committed writes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.upsertBatchIdempotentWithOptions(
        &[_]u64{7},
        &[_]f32{ 0.0, 1.0 },
        dims,
        .{},
        0xabc2,
        0x3333,
    );
    try backend.upsertBatchIdempotentWithOptions(
        &[_]u64{7},
        &[_]f32{ 0.0, 1.0 },
        dims,
        .{},
        0xabc2,
        0x3333,
    );

    var changes = try backend.readChanges(std.testing.allocator, 0, 20);
    defer changes.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), changes.events.len);
    try std.testing.expectEqual(ChangeOp.upsert, changes.events[0].op);
}

test "VectorAdapter.deleteIdsIdempotent replays delete counts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{42}, &[_]f32{ 1.0, 0.0 }, dims);

    const first = try backend.deleteIdsIdempotent(&[_]u64{42}, 0xabc3, 0x4444);
    try std.testing.expectEqual(@as(usize, 1), first.deleted_count);
    try std.testing.expectEqual(@as(usize, 0), first.not_found_count);

    const replay = try backend.deleteIdsIdempotent(&[_]u64{42}, 0xabc3, 0x4444);
    try std.testing.expectEqual(@as(usize, 1), replay.deleted_count);
    try std.testing.expectEqual(@as(usize, 0), replay.not_found_count);
}

test "VectorAdapter.compactIdempotent replays compact counts" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});

    const first = try backend.compactIdempotent(dims, 0xabc4, 0x5555);
    try std.testing.expectEqual(@as(usize, 1), first.kept_count);
    try std.testing.expectEqual(@as(usize, 1), first.removed_tombstones);

    const replay = try backend.compactIdempotent(dims, 0xabc4, 0x5555);
    try std.testing.expectEqual(@as(usize, 1), replay.kept_count);
    try std.testing.expectEqual(@as(usize, 1), replay.removed_tombstones);
}

test "VectorAdapter.compactWithExpectedGeneration succeeds on matching generation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});

    try backend.fs_writer.flushBlock();
    _ = try backend.fs_reader.refreshIfChanged();
    const generation = backend.fs_reader.snapshotGeneration();

    const result = try backend.compactWithExpectedGeneration(dims, generation);
    try std.testing.expectEqual(@as(usize, 1), result.kept_count);
    try std.testing.expectEqual(@as(usize, 1), result.removed_tombstones);
}

test "VectorAdapter.compactWithExpectedGeneration rejects stale generation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);

    try backend.fs_writer.flushBlock();
    _ = try backend.fs_reader.refreshIfChanged();
    const generation = backend.fs_reader.snapshotGeneration();

    try std.testing.expectError(
        error.ManifestGenerationConflict,
        backend.compactWithExpectedGeneration(dims, generation + 1),
    );
}

test "VectorAdapter.compactExpiredTombstones skips compaction when no tombstones are expired" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});

    const result = try backend.compactExpiredTombstones(dims, std.time.milliTimestamp(), std.time.ms_per_hour);
    try std.testing.expectEqual(@as(usize, 1), result.kept_count);
    try std.testing.expectEqual(@as(usize, 0), result.removed_tombstones);
}

test "VectorAdapter.compactExpiredTombstones compacts when tombstones are expired" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});

    const far_future = std.math.maxInt(i64);
    const result = try backend.compactExpiredTombstones(dims, far_future, 1);
    try std.testing.expectEqual(@as(usize, 1), result.kept_count);
    try std.testing.expectEqual(@as(usize, 1), result.removed_tombstones);
}

test "VectorAdapter.readChanges preserves cursor order across compaction" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    var backend = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend.deinit();

    const dims: u32 = 2;
    try backend.appendBatch(&[_]u64{ 1, 2 }, &[_]f32{
        1.0, 0.0,
        0.0, 1.0,
    }, dims);
    _ = try backend.deleteIds(&[_]u64{2});

    var first_page = try backend.readChanges(std.testing.allocator, 0, 2);
    defer first_page.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), first_page.events.len);

    const encoded_cursor = try vector_cdc.encodeCursor(std.testing.allocator, .{
        .since = first_page.next_since,
        .generation = backend.change_reader.snapshotGeneration(),
    });
    defer std.testing.allocator.free(encoded_cursor);
    const cursor = try vector_cdc.decodeCursor(encoded_cursor);

    _ = try backend.compact(dims);

    var second_page = try backend.readChanges(std.testing.allocator, cursor.since, 100);
    defer second_page.deinit(std.testing.allocator);
    try std.testing.expect(second_page.events.len >= 2);

    var last_seq = cursor.since;
    for (second_page.events) |event| {
        try std.testing.expect(event.seq > last_seq);
        last_seq = event.seq;
    }
}

test "VectorAdapter.discoverNextSeq resumes after idempotency records" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const dims: u32 = 2;
    var backend_a = try VectorAdapter.init(std.testing.allocator, root_path);
    try backend_a.appendBatchIdempotentWithOptions(
        &[_]u64{42},
        &[_]f32{ 1.0, 0.0 },
        dims,
        .{},
        0x9001,
        0x7001,
    );

    var records_before = try backend_a.loadLatestIdempotency(std.testing.allocator);
    defer records_before.deinit();
    var max_seq_before: u64 = 0;
    var iter_before = records_before.valueIterator();
    while (iter_before.next()) |record| {
        max_seq_before = @max(max_seq_before, record.seq);
    }
    backend_a.deinit();

    var backend_b = try VectorAdapter.init(std.testing.allocator, root_path);
    defer backend_b.deinit();
    try backend_b.appendIdempotencyRecord(0x9002, 0x7002, .append, .pending, 0, 0);

    var records_after = try backend_b.loadLatestIdempotency(std.testing.allocator);
    defer records_after.deinit();
    const resumed = records_after.get(0x9002) orelse return error.TestUnexpectedResult;
    try std.testing.expect(resumed.seq > max_seq_before);
}
