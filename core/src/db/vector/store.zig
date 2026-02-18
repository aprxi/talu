//! TaluDB vector adapter for high-throughput embedding storage.
//!
//! Provides batch append and bulk load for schema 10 (embeddings).
//! Thread safety: NOT thread-safe (single-writer semantics via <ns>/talu.lock).

const std = @import("std");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const types = @import("../types.zig");
const tensor_mod = @import("../../tensor.zig");
const dot_product = @import("../../compute/cpu/dot_product.zig");
const parallel = @import("../../compute/parallel.zig");

const Allocator = std.mem.Allocator;

const schema_embeddings: u16 = 10;

const col_doc_id: u32 = 1;
const col_ts: u32 = 2;
const col_embedding: u32 = 10;

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

pub const ScoreCallback = *const fn (ctx: *anyopaque, id: u64, score: f32) void;
pub const ScoreCallbackC = *const fn (ctx: ?*anyopaque, id: u64, score: f32) callconv(.c) void;

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
    fs_writer: *db_writer.Writer,
    fs_reader: *db_reader.Reader,

    /// Initialize vector TaluDB backend for the "vector" namespace.
    pub fn init(allocator: Allocator, db_root: []const u8) !VectorAdapter {
        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, "vector");
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, "vector");
        errdefer reader_ptr.deinit();

        return .{
            .allocator = allocator,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
        };
    }

    /// Append a batch of embeddings.
    pub fn appendBatch(self: *VectorAdapter, doc_ids: []const u64, vectors: []const f32, dims: u32) !void {
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
            .{
                .column_id = col_doc_id,
                .shape = .SCALAR,
                .phys_type = .U64,
                .encoding = .RAW,
                .dims = 1,
                .data = std.mem.sliceAsBytes(doc_ids),
            },
            .{
                .column_id = col_ts,
                .shape = .SCALAR,
                .phys_type = .I64,
                .encoding = .RAW,
                .dims = 1,
                .data = std.mem.sliceAsBytes(ts_buf),
            },
            .{
                .column_id = col_embedding,
                .shape = .VECTOR,
                .phys_type = .F32,
                .encoding = .RAW,
                .dims = @intCast(dims),
                .data = std.mem.sliceAsBytes(vectors),
            },
        };

        try self.fs_writer.appendBatch(schema_embeddings, @intCast(doc_ids.len), &columns);
    }

    /// Load all stored embeddings.
    pub fn loadVectors(self: *VectorAdapter, allocator: Allocator) !VectorBatch {
        try self.fs_writer.flushBlock();
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);
        var total_rows: usize = 0;
        var dims: ?u32 = null;

        for (blocks) |block| {
            var file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
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
            var file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
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
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var total_rows: usize = 0;
        var dims: ?u32 = null;

        for (blocks) |block| {
            var file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
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
            var file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
            defer file.close();

            const reader = block_reader.BlockReader.init(file, allocator);
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
        if (k == 0) {
            return .{
                .ids = try allocator.alloc(u64, 0),
                .scores = try allocator.alloc(f32, 0),
            };
        }
        if (query.len == 0) return error.InvalidColumnData;
        if (k > std.math.maxInt(usize)) return error.InvalidColumnData;

        try self.fs_writer.flushBlock();
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var heap = try MinHeap.init(allocator, @intCast(k));
        defer heap.deinit(allocator);
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        var current_path: ?[]const u8 = null;
        var current_file: ?std.fs.File = null;
        defer if (current_file) |file| file.close();

        for (blocks) |block| {
            if (current_path == null or !std.mem.eql(u8, current_path.?, block.path)) {
                if (current_file) |file| file.close();
                current_file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
                current_path = block.path;
            }

            const reader = block_reader.BlockReader.init(current_file.?, allocator);
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

            try scoreAndCollect(allocator, query, ids, vectors, @as(usize, dims_u32), header.row_count, &heap);
        }

        return heap.finalize(allocator);
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
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var current_path: ?[]const u8 = null;
        var current_file: ?std.fs.File = null;
        defer if (current_file) |file| file.close();
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        for (blocks) |block| {
            if (current_path == null or !std.mem.eql(u8, current_path.?, block.path)) {
                if (current_file) |file| file.close();
                current_file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
                current_path = block.path;
            }

            const reader = block_reader.BlockReader.init(current_file.?, allocator);
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
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var total_rows: usize = 0;
        var current_path: ?[]const u8 = null;
        var current_file: ?std.fs.File = null;
        defer if (current_file) |file| file.close();

        for (blocks) |block| {
            if (current_path == null or !std.mem.eql(u8, current_path.?, block.path)) {
                if (current_file) |file| file.close();
                current_file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
                current_path = block.path;
            }

            const reader = block_reader.BlockReader.init(current_file.?, allocator);
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
        var current_file: ?std.fs.File = null;
        defer if (current_file) |file| file.close();
        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        var row_offset: usize = 0;
        const dims_usize = @as(usize, dims);

        for (blocks) |block| {
            if (current_path == null or !std.mem.eql(u8, current_path.?, block.path)) {
                if (current_file) |file| file.close();
                current_file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
                current_path = block.path;
            }

            const reader = block_reader.BlockReader.init(current_file.?, allocator);
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
        if (k == 0 or query_count == 0) {
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

        try self.fs_writer.flushBlock();
        try self.fs_reader.refreshCurrent();
        const blocks = try self.fs_reader.getBlocks(allocator);
        defer allocator.free(blocks);

        var heaps = try allocator.alloc(MinHeap, query_count);
        errdefer {
            for (heaps) |*heap| heap.deinit(allocator);
            allocator.free(heaps);
        }

        for (heaps) |*heap| {
            heap.* = try MinHeap.init(allocator, @intCast(k));
        }

        var current_path: ?[]const u8 = null;
        var current_file: ?std.fs.File = null;
        defer if (current_file) |file| file.close();

        var scratch = ScratchBuffers{};
        defer scratch.deinit(allocator);

        for (blocks) |block| {
            if (current_path == null or !std.mem.eql(u8, current_path.?, block.path)) {
                if (current_file) |file| file.close();
                current_file = try self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only });
                current_path = block.path;
            }

            const reader = block_reader.BlockReader.init(current_file.?, allocator);
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

            const vector_len = @as(usize, header.row_count) * @as(usize, dims);
            if (vector_buf.len != vector_len * @sizeOf(f32)) return error.InvalidColumnData;

            const ids = @as([*]const u64, @ptrCast(@alignCast(id_buf.ptr)))[0..header.row_count];
            const vectors = @as([*]const f32, @ptrCast(@alignCast(vector_buf.ptr)))[0..vector_len];

            // Allocate flat score buffer: query_count scores per row.
            const block_scores = try allocator.alloc(f32, @as(usize, header.row_count) * @as(usize, query_count));
            defer allocator.free(block_scores);

            // Parallel scoring: each thread scores a range of rows against all queries.
            parallelScoreBatchRows(queries, vectors, @as(usize, dims), query_count, header.row_count, header.row_count, 0, block_scores);

            // Serial heap collection from the flat score buffer.
            for (0..header.row_count) |row_idx| {
                for (0..@as(usize, query_count)) |query_idx| {
                    heaps[query_idx].push(block_scores[query_idx * header.row_count + row_idx], ids[row_idx]);
                }
            }
        }

        const count_per_query: u32 = if (query_count == 0) 0 else @intCast(heaps[0].size);
        const total = @as(usize, count_per_query) * @as(usize, query_count);
        const ids_out = try allocator.alloc(u64, total);
        errdefer allocator.free(ids_out);
        const scores_out = try allocator.alloc(f32, total);
        errdefer allocator.free(scores_out);

        var query_idx: usize = 0;
        while (query_idx < query_count) : (query_idx += 1) {
            const offset = query_idx * @as(usize, count_per_query);
            const ids_slice = ids_out[offset .. offset + @as(usize, count_per_query)];
            const scores_slice = scores_out[offset .. offset + @as(usize, count_per_query)];
            try heaps[query_idx].writeSorted(allocator, ids_slice, scores_slice);
        }

        for (heaps) |*heap| heap.deinit(allocator);
        allocator.free(heaps);

        return .{
            .ids = ids_out,
            .scores = scores_out,
            .count_per_query = count_per_query,
            .query_count = query_count,
        };
    }

    /// Release resources.
    pub fn deinit(self: *VectorAdapter) void {
        self.fs_writer.flushBlock() catch {};
        self.fs_writer.deinit();
        self.allocator.destroy(self.fs_writer);
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
    }

    /// Simulates a process crash for testing.
    ///
    /// Releases all resources (closing fds, releasing flocks) WITHOUT
    /// flushing pending data or deleting the WAL file.
    pub fn simulateCrash(self: *VectorAdapter) void {
        self.fs_writer.simulateCrash();
        self.allocator.destroy(self.fs_writer);
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);
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

const MinHeap = struct {
    entries: []SearchEntry,
    size: usize,

    pub fn init(allocator: Allocator, capacity: usize) !MinHeap {
        if (capacity == 0) return error.InvalidColumnData;
        const entries = try allocator.alloc(SearchEntry, capacity);
        return .{ .entries = entries, .size = 0 };
    }

    pub fn deinit(self: *MinHeap, allocator: Allocator) void {
        allocator.free(self.entries);
    }

    pub fn push(self: *MinHeap, score: f32, id: u64) void {
        if (self.size < self.entries.len) {
            self.entries[self.size] = .{ .score = score, .id = id };
            self.siftUp(self.size);
            self.size += 1;
            return;
        }
        if (score <= self.entries[0].score) return;
        self.entries[0] = .{ .score = score, .id = id };
        self.siftDown(0);
    }

    pub fn finalize(self: *MinHeap, allocator: Allocator) !SearchResult {
        const ids = try allocator.alloc(u64, self.size);
        errdefer allocator.free(ids);
        const scores = try allocator.alloc(f32, self.size);
        errdefer allocator.free(scores);
        try self.writeSorted(allocator, ids, scores);

        return .{ .ids = ids, .scores = scores };
    }

    fn siftUp(self: *MinHeap, start_idx: usize) void {
        var idx = start_idx;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (self.entries[parent].score <= self.entries[idx].score) break;
            std.mem.swap(SearchEntry, &self.entries[parent], &self.entries[idx]);
            idx = parent;
        }
    }

    fn siftDown(self: *MinHeap, start_idx: usize) void {
        var idx = start_idx;
        const size = self.size;
        while (true) {
            const left = idx * 2 + 1;
            if (left >= size) break;
            const right = left + 1;
            var smallest = left;
            if (right < size and self.entries[right].score < self.entries[left].score) {
                smallest = right;
            }
            if (self.entries[idx].score <= self.entries[smallest].score) break;
            std.mem.swap(SearchEntry, &self.entries[idx], &self.entries[smallest]);
            idx = smallest;
        }
    }

    fn writeSorted(self: *MinHeap, allocator: Allocator, ids: []u64, scores: []f32) !void {
        if (ids.len != self.size or scores.len != self.size) return error.InvalidColumnData;
        const items = try allocator.alloc(SearchEntry, self.size);
        defer allocator.free(items);
        std.mem.copyForwards(SearchEntry, items, self.entries[0..self.size]);

        std.sort.pdq(SearchEntry, items, {}, struct {
            fn lessThan(_: void, a: SearchEntry, b: SearchEntry) bool {
                return a.score > b.score;
            }
        }.lessThan);

        for (items, 0..) |entry, idx| {
            ids[idx] = entry.id;
            scores[idx] = entry.score;
        }
    }
};

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

/// Score all rows against a single query (parallel), then push into the heap (serial).
fn scoreAndCollect(
    allocator: Allocator,
    query: []const f32,
    ids: []const u64,
    vectors: []const f32,
    dims: usize,
    row_count: usize,
    heap: *MinHeap,
) !void {
    const scores = try allocator.alloc(f32, row_count);
    defer allocator.free(scores);

    parallelScoreRows(query, vectors, dims, scores);

    for (0..row_count) |i| {
        heap.push(scores[i], ids[i]);
    }
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
