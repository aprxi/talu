//! Generic table engine for TaluDB.
//!
//! Provides a domain-agnostic layer over Writer/Reader that handles the common
//! patterns all table adapters need: append rows, scan with scalar column
//! filters, deduplicate by PK hash, handle tombstones, cursor-based pagination,
//! and point lookups with dual-hash support.
//!
//! Domain adapters encode/decode their own payloads and define their own schema
//! IDs. Table treats payloads as opaque bytes.

const std = @import("std");
const db_writer = @import("../writer.zig");
const db_reader = @import("../reader.zig");
const block_reader = @import("../block_reader.zig");
const types = @import("../types.zig");

const Allocator = std.mem.Allocator;
pub const ColumnValue = db_writer.ColumnValue;

// =============================================================================
// Public types
// =============================================================================

/// Compaction policy passed at open time. Tells the engine which schema IDs
/// are live records vs tombstones, and which columns serve as PK, timestamp,
/// and TTL.
pub const CompactionPolicy = struct {
    /// Schema IDs for live (non-tombstone) records.
    active_schema_ids: []const u16,
    /// Schema ID for delete markers. Null if tombstones are not used.
    tombstone_schema_id: ?u16 = null,
    /// Column holding the primary-key hash (used for dedup). Default: 1.
    dedup_column_id: u32 = 1,
    /// Column holding the timestamp (used for ordering/dedup). Default: 2.
    ts_column_id: u32 = 2,
    /// Column holding TTL expiration (optional). Null if no TTL.
    ttl_column_id: ?u32 = null,
};

// =============================================================================
// Policy persistence
// =============================================================================

const policy_filename = "policy.json";

/// JSON-serializable policy format written to disk.
const PersistedPolicy = struct {
    version: u8 = 1,
    active_schema_ids: []const u16,
    tombstone_schema_id: ?u16 = null,
    dedup_column_id: u32 = 1,
    ts_column_id: u32 = 2,
    ttl_column_id: ?u32 = null,
};

/// Owned slices returned by loadPersistedPolicy. Caller must free via freeLoadedPolicy.
pub const LoadedPolicy = struct {
    policy: CompactionPolicy,
    owned_schema_ids: []u16,
};

/// Write the compaction policy to `{db_root}/{namespace}/policy.json` (atomic rename).
/// The namespace directory must already exist (created by Writer.open).
pub fn persistPolicy(allocator: Allocator, db_root: []const u8, namespace: []const u8, policy: CompactionPolicy) !void {
    const doc = PersistedPolicy{
        .active_schema_ids = policy.active_schema_ids,
        .tombstone_schema_id = policy.tombstone_schema_id,
        .dedup_column_id = policy.dedup_column_id,
        .ts_column_id = policy.ts_column_id,
        .ttl_column_id = policy.ttl_column_id,
    };

    const path = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ db_root, namespace, policy_filename });
    defer allocator.free(path);
    const tmp_path = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}.tmp", .{ db_root, namespace, policy_filename });
    defer allocator.free(tmp_path);

    const json_bytes = try std.json.Stringify.valueAlloc(allocator, doc, .{ .whitespace = .indent_2 });
    defer allocator.free(json_bytes);

    var tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
    errdefer std.fs.cwd().deleteFile(tmp_path) catch {};
    try tmp_file.writeAll(json_bytes);
    try tmp_file.sync();
    tmp_file.close();

    try std.fs.cwd().rename(tmp_path, path);
}

/// Load a persisted compaction policy from `{db_root}/{namespace}/policy.json`.
/// Returns null if the file does not exist. Caller owns the returned slices.
pub fn loadPersistedPolicy(allocator: Allocator, db_root: []const u8, namespace: []const u8) !?LoadedPolicy {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ db_root, namespace, policy_filename });
    defer allocator.free(path);

    const file = std.fs.cwd().openFile(path, .{ .mode = .read_only }) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close();

    const bytes = try file.readToEndAlloc(allocator, 64 * 1024);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(PersistedPolicy, allocator, bytes, .{});
    defer parsed.deinit();
    const doc = parsed.value;

    if (doc.version != 1) return error.UnsupportedPolicyVersion;

    // Copy slices to caller-owned memory (parsed arena will be freed).
    const schema_ids = try allocator.alloc(u16, doc.active_schema_ids.len);
    errdefer allocator.free(schema_ids);
    @memcpy(schema_ids, doc.active_schema_ids);

    return .{
        .policy = .{
            .active_schema_ids = schema_ids,
            .tombstone_schema_id = doc.tombstone_schema_id,
            .dedup_column_id = doc.dedup_column_id,
            .ts_column_id = doc.ts_column_id,
            .ttl_column_id = doc.ttl_column_id,
        },
        .owned_schema_ids = schema_ids,
    };
}

/// Free slices from a LoadedPolicy.
pub fn freeLoadedPolicy(allocator: Allocator, loaded: LoadedPolicy) void {
    allocator.free(loaded.owned_schema_ids);
}

/// Equality/range filter on a scalar column.
pub const ColumnFilter = struct {
    column_id: u32,
    op: FilterOp,
    value: u64,
};

pub const FilterOp = enum(u8) {
    eq = 0,
    ne = 1,
    lt = 2,
    le = 3,
    gt = 4,
    ge = 5,
};

/// Parameters for a scan operation.
pub const ScanParams = struct {
    /// Primary schema ID to scan.
    schema_id: u16,
    /// Additional schema IDs to also accept (for backward compat with legacy).
    additional_schema_ids: ?[]const u16 = null,
    /// Scalar column filters (equality/range).
    filters: []const ColumnFilter = &.{},
    /// Column used for deduplication (keep latest per unique value). Default: 1.
    /// Null disables dedup — all matching rows are returned (useful for junction
    /// tables that use positive/negative timestamps for add/remove semantics).
    dedup_column_id: ?u32 = 1,
    /// Schema ID for tombstones. Null to skip tombstone checking.
    delete_schema_id: ?u16 = null,
    /// Column holding timestamp for ordering. Default: 2.
    ts_column_id: u32 = 2,
    /// Column holding TTL expiration. Null to skip TTL checks.
    ttl_column_id: ?u32 = null,
    /// Max results. 0 = unlimited (capped internally).
    limit: u32 = 0,
    /// Cursor: only return rows with ts strictly before this value.
    cursor_ts: ?i64 = null,
    /// Cursor tiebreak: when ts equals cursor_ts, only return rows with
    /// dedup hash strictly less than this value.
    cursor_hash: ?u64 = null,
    /// Column holding the payload (VARBYTES). Default: 20.
    payload_column_id: u32 = 20,
    /// Scan order. True = newest first (reverse chronological).
    reverse: bool = true,
    /// Additional scalar column IDs to include in Row.scalars beyond
    /// dedup, ts, and filter columns. Useful when the caller needs column values
    /// without filtering on them.
    extra_columns: []const u32 = &.{},
};

/// A scalar column value from a scanned row.
pub const ColumnData = struct {
    column_id: u32,
    /// Raw 64-bit value. Caller reinterprets as u64/i64/f64 based on schema.
    value_u64: u64,
};

/// A row returned by scan or get. All memory owned by the caller's allocator.
pub const Row = struct {
    /// Scalar column values from the block.
    scalars: []ColumnData,
    /// Raw VARBYTES payload bytes. Encoding is the caller's concern.
    payload: []const u8,
};

/// Result of a scan operation.
pub const ScanResult = struct {
    /// Matching rows.
    rows: []Row,
    /// True if more rows exist beyond the limit.
    has_more: bool,
};

/// Internal cap for unlimited scans to prevent unbounded memory.
const max_unlimited_rows: u32 = 10_000;

// =============================================================================
// Table
// =============================================================================

pub const Table = struct {
    allocator: Allocator,
    fs_writer: ?*db_writer.Writer,
    /// Exposed for domain adapters that need direct block access (e.g., parallel search).
    fs_reader: *db_reader.Reader,
    policy: CompactionPolicy,
    /// Non-null when the policy was loaded from disk (owned by allocator).
    policy_owned_schema_ids: ?[]u16 = null,

    /// Open a read-write generic table.
    /// On first open, persists the provided policy to disk. Subsequent opens
    /// load the persisted policy (the `policy` parameter is ignored).
    pub fn open(
        allocator: Allocator,
        db_root: []const u8,
        namespace: []const u8,
        policy: CompactionPolicy,
    ) !Table {
        var writer_ptr = try allocator.create(db_writer.Writer);
        errdefer allocator.destroy(writer_ptr);
        writer_ptr.* = try db_writer.Writer.open(allocator, db_root, namespace);
        errdefer writer_ptr.deinit();

        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, namespace);
        errdefer reader_ptr.deinit();

        // Try to load a persisted policy; if none exists, persist the provided one.
        if (try loadPersistedPolicy(allocator, db_root, namespace)) |loaded| {
            return .{
                .allocator = allocator,
                .fs_writer = writer_ptr,
                .fs_reader = reader_ptr,
                .policy = loaded.policy,
                .policy_owned_schema_ids = loaded.owned_schema_ids,
            };
        }

        // First open — persist the provided policy for future opens.
        // Only persist if the policy carries real information (non-empty active_schema_ids).
        // A caller like DELETE may pass a placeholder policy with empty active_schema_ids;
        // persisting that would poison all future opens.
        if (policy.active_schema_ids.len > 0) {
            persistPolicy(allocator, db_root, namespace, policy) catch |err| switch (err) {
                error.FileNotFound => {},
                else => return err,
            };
        }

        return .{
            .allocator = allocator,
            .fs_writer = writer_ptr,
            .fs_reader = reader_ptr,
            .policy = policy,
        };
    }

    /// Open a read-only generic table (no writer allocated).
    /// Loads the persisted policy from disk if available; otherwise uses the
    /// provided `policy` as a fallback.
    pub fn openReadOnly(
        allocator: Allocator,
        db_root: []const u8,
        namespace: []const u8,
        policy: CompactionPolicy,
    ) !Table {
        var reader_ptr = try allocator.create(db_reader.Reader);
        errdefer allocator.destroy(reader_ptr);
        reader_ptr.* = try db_reader.Reader.open(allocator, db_root, namespace);
        errdefer reader_ptr.deinit();

        if (try loadPersistedPolicy(allocator, db_root, namespace)) |loaded| {
            return .{
                .allocator = allocator,
                .fs_writer = null,
                .fs_reader = reader_ptr,
                .policy = loaded.policy,
                .policy_owned_schema_ids = loaded.owned_schema_ids,
            };
        }

        return .{
            .allocator = allocator,
            .fs_writer = null,
            .fs_reader = reader_ptr,
            .policy = policy,
        };
    }

    /// Release resources. Flushes buffered rows for write-mode tables.
    pub fn deinit(self: *Table) void {
        if (self.fs_writer) |writer| {
            writer.flushBlock() catch {};
            writer.deinit();
            self.allocator.destroy(writer);
            self.fs_writer = null;
        }
        self.fs_reader.deinit();
        self.allocator.destroy(self.fs_reader);

        // Free policy slices loaded from disk.
        if (self.policy_owned_schema_ids) |ids| self.allocator.free(ids);
    }

    /// Append a row. Handles schema switching (flushes current block if the
    /// schema changes from the previous append).
    ///
    /// Returns error.MissingDedupColumn if dedup_column_id is absent,
    /// error.MissingTimestampColumn if ts_column_id is absent.
    pub fn appendRow(self: *Table, schema_id: u16, columns: []const ColumnValue) !void {
        const writer = self.fs_writer orelse return error.ReadOnly;

        // Validate that required policy columns are present.
        var has_dedup = false;
        var has_ts = false;
        for (columns) |col| {
            if (col.column_id == self.policy.dedup_column_id) has_dedup = true;
            if (col.column_id == self.policy.ts_column_id) has_ts = true;
        }
        if (!has_dedup) return error.MissingDedupColumn;
        if (!has_ts) return error.MissingTimestampColumn;

        // If the writer is mid-block with a different schema, flush first.
        if (writer.schema_id) |current| {
            if (current != schema_id) {
                try writer.flushBlock();
                writer.resetSchema();
            }
        } else if (writer.columns.items.len > 0) {
            // After a flush, column buffers may be stale from a previous schema.
            writer.resetSchema();
        }
        try writer.appendRow(schema_id, columns);
    }

    /// Flush buffered rows to disk.
    pub fn flush(self: *Table) !void {
        const writer = self.fs_writer orelse return error.ReadOnly;
        try writer.flushBlock();
    }

    /// Write a tombstone row (dedup_column + ts_column only).
    /// Uses the policy's tombstone_schema_id.
    pub fn deleteTombstone(self: *Table, pk_hash: u64, ts: i64) !void {
        const tombstone_schema = self.policy.tombstone_schema_id orelse return error.NoTombstoneSchema;
        var pk_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &pk_bytes, pk_hash, .little);
        var ts_bytes: [8]u8 = undefined;
        std.mem.writeInt(i64, &ts_bytes, ts, .little);

        const columns = [_]ColumnValue{
            .{
                .column_id = self.policy.dedup_column_id,
                .shape = .SCALAR,
                .phys_type = .U64,
                .encoding = .RAW,
                .dims = 1,
                .data = &pk_bytes,
            },
            .{
                .column_id = self.policy.ts_column_id,
                .shape = .SCALAR,
                .phys_type = .I64,
                .encoding = .RAW,
                .dims = 1,
                .data = &ts_bytes,
            },
        };

        try self.appendRow(tombstone_schema, &columns);
    }

    /// Scan with filtering, dedup, tombstone handling, and pagination.
    pub fn scan(self: *Table, alloc: Allocator, params: ScanParams) !ScanResult {
        _ = try self.fs_reader.refreshIfChanged();

        // Fetch one extra row beyond the requested limit to determine has_more.
        const requested_limit: u32 = if (params.limit == 0) max_unlimited_rows else params.limit;
        const effective_limit: u32 = if (requested_limit < std.math.maxInt(u32)) requested_limit + 1 else requested_limit;
        const now_ms = std.time.milliTimestamp();

        // Step 1: Collect tombstones if configured (requires dedup column for PK matching).
        var tombstones = std.AutoHashMap(u64, i64).init(alloc);
        defer tombstones.deinit();
        if (params.delete_schema_id) |del_schema| {
            if (params.dedup_column_id) |dedup_col| {
                try self.collectTombstones(alloc, del_schema, dedup_col, params.ts_column_id, &tombstones);
            }
        }

        // Step 2: Scan blocks for matching rows.
        var seen = std.AutoHashMap(u64, void).init(alloc);
        defer seen.deinit();

        var results = std.ArrayList(Row).empty;
        errdefer {
            for (results.items) |row| freeRow(alloc, row);
            results.deinit(alloc);
        }

        var limit_reached = false;

        const blocks = try self.fs_reader.getBlocks(alloc);
        defer alloc.free(blocks);

        if (params.reverse) {
            // Iterate blocks in reverse (newest first).
            var block_idx: usize = blocks.len;
            while (block_idx > 0) {
                block_idx -= 1;
                limit_reached = try self.processBlock(
                    alloc,
                    blocks[block_idx],
                    params,
                    effective_limit,
                    now_ms,
                    &tombstones,
                    &seen,
                    &results,
                );
                if (limit_reached) break;
            }
        } else {
            for (blocks) |block| {
                limit_reached = try self.processBlock(
                    alloc,
                    block,
                    params,
                    effective_limit,
                    now_ms,
                    &tombstones,
                    &seen,
                    &results,
                );
                if (limit_reached) break;
            }
        }

        var rows = try results.toOwnedSlice(alloc);
        const has_more = rows.len > requested_limit;
        if (has_more) {
            // Free the extra sentinel row and trim.
            freeRow(alloc, rows[requested_limit]);
            rows = try alloc.realloc(rows, requested_limit);
        }

        return .{
            .rows = rows,
            .has_more = has_more,
        };
    }

    /// Point lookup by primary hash. Returns the newest non-tombstoned,
    /// non-expired row, or null.
    pub fn get(
        self: *Table,
        alloc: Allocator,
        schema_id: u16,
        pk_hash: u64,
        legacy_hash: ?u64,
    ) !?Row {
        _ = try self.fs_reader.refreshIfChanged();

        // Collect tombstones for this schema.
        var tombstones = std.AutoHashMap(u64, i64).init(alloc);
        defer tombstones.deinit();
        if (self.policy.tombstone_schema_id) |del_schema| {
            try self.collectTombstones(
                alloc,
                del_schema,
                self.policy.dedup_column_id,
                self.policy.ts_column_id,
                &tombstones,
            );
        }

        const now_ms = std.time.milliTimestamp();
        var best: ?Row = null;
        var best_ts: i64 = std.math.minInt(i64);

        const blocks = try self.fs_reader.getBlocks(alloc);
        defer alloc.free(blocks);

        // Iterate blocks in reverse (newest first).
        var block_idx: usize = blocks.len;
        while (block_idx > 0) {
            block_idx -= 1;
            const block = blocks[block_idx];

            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, alloc);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != schema_id) continue;

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer alloc.free(descs);

            const pk_desc = findColumn(descs, self.policy.dedup_column_id) orelse continue;
            const ts_desc = findColumn(descs, self.policy.ts_column_id) orelse continue;
            const payload_desc = findColumn(descs, self.policy.dedup_column_id);
            _ = payload_desc;
            const pay_desc = findColumn(descs, 20) orelse continue;

            const pk_bytes = reader.readColumnData(block.offset, pk_desc, alloc) catch continue;
            defer alloc.free(pk_bytes);
            const ts_bytes = reader.readColumnData(block.offset, ts_desc, alloc) catch continue;
            defer alloc.free(ts_bytes);

            // Check TTL column if configured.
            var ttl_bytes: ?[]const u8 = null;
            defer if (ttl_bytes) |b| alloc.free(b);
            if (self.policy.ttl_column_id) |ttl_col| {
                if (findColumn(descs, ttl_col)) |ttl_desc| {
                    ttl_bytes = reader.readColumnData(block.offset, ttl_desc, alloc) catch null;
                }
            }

            var payload_buffers = readVarBytesBuffers(file, block.offset, pay_desc, row_count, alloc) catch continue;
            defer payload_buffers.deinit(alloc);

            // Scan rows in reverse within block.
            var row_idx: usize = row_count;
            while (row_idx > 0) {
                row_idx -= 1;

                const row_pk = readU64At(pk_bytes, row_idx) catch continue;
                const matches = (row_pk == pk_hash) or
                    (if (legacy_hash) |lh| row_pk == lh else false);
                if (!matches) continue;

                const row_ts = readI64At(ts_bytes, row_idx) catch continue;

                // Skip if tombstoned.
                if (tombstones.get(row_pk)) |del_ts| {
                    if (del_ts >= row_ts) continue;
                }
                // Also check legacy hash in tombstones.
                if (legacy_hash) |lh| {
                    if (tombstones.get(lh)) |del_ts| {
                        if (del_ts >= row_ts) continue;
                    }
                }

                // Skip if TTL expired. An expired newer version tombstones
                // all older versions of this key.
                if (ttl_bytes) |tb| {
                    const ttl_val = readI64At(tb, row_idx) catch continue;
                    if (ttl_val > 0 and ttl_val < now_ms) {
                        try tombstones.put(row_pk, std.math.maxInt(i64));
                        continue;
                    }
                }

                if (row_ts <= best_ts) continue;

                // Read payload.
                const payload_raw = payload_buffers.sliceForRow(row_idx) catch continue;

                // Read all scalar columns.
                const scalars = try readAllScalars(alloc, reader, block.offset, descs, row_idx);

                const row = Row{
                    .scalars = scalars,
                    .payload = try alloc.dupe(u8, payload_raw),
                };

                if (best) |old| freeRow(alloc, old);
                best = row;
                best_ts = row_ts;
            }

            // If we found a row in the newest block, we can stop early —
            // any older blocks can only have older versions.
            if (best != null) break;
        }

        return best;
    }

    // =========================================================================
    // Internal: block processing for scan
    // =========================================================================

    fn processBlock(
        self: *Table,
        alloc: Allocator,
        block: db_reader.BlockRef,
        params: ScanParams,
        effective_limit: u32,
        now_ms: i64,
        tombstones: *std.AutoHashMap(u64, i64),
        seen: *std.AutoHashMap(u64, void),
        results: *std.ArrayList(Row),
    ) !bool {
        var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch return false;
        defer file.close();

        const reader = block_reader.BlockReader.init(file, alloc);
        const header = reader.readHeader(block.offset) catch return false;
        if (!matchesSchema(header.schema_id, params.schema_id, params.additional_schema_ids)) return false;

        const row_count = header.row_count;
        if (row_count == 0) return false;

        const descs = reader.readColumnDirectory(header, block.offset) catch return false;
        defer alloc.free(descs);

        // Read dedup (PK) column if dedup is enabled.
        var pk_bytes: ?[]const u8 = null;
        defer if (pk_bytes) |b| alloc.free(b);
        if (params.dedup_column_id) |dedup_col| {
            const pk_desc = findColumn(descs, dedup_col) orelse return false;
            pk_bytes = reader.readColumnData(block.offset, pk_desc, alloc) catch return false;
        }

        // Read timestamp column (always required).
        const ts_desc = findColumn(descs, params.ts_column_id) orelse return false;
        const ts_bytes = reader.readColumnData(block.offset, ts_desc, alloc) catch return false;
        defer alloc.free(ts_bytes);

        // Lazily read filter columns.
        var filter_cols = try alloc.alloc(?[]const u8, params.filters.len);
        defer {
            for (filter_cols) |opt| {
                if (opt) |data| alloc.free(data);
            }
            alloc.free(filter_cols);
        }
        for (params.filters, 0..) |filter, i| {
            const is_dedup = if (params.dedup_column_id) |dc| filter.column_id == dc else false;
            if (is_dedup) {
                filter_cols[i] = null; // Use pk_bytes directly
            } else if (filter.column_id == params.ts_column_id) {
                filter_cols[i] = null; // Use ts_bytes directly
            } else if (findColumn(descs, filter.column_id)) |desc| {
                filter_cols[i] = reader.readColumnData(block.offset, desc, alloc) catch null;
            } else {
                filter_cols[i] = null;
            }
        }

        // Read TTL column if configured.
        var ttl_bytes: ?[]const u8 = null;
        defer if (ttl_bytes) |b| alloc.free(b);
        if (params.ttl_column_id) |ttl_col| {
            if (findColumn(descs, ttl_col)) |ttl_desc| {
                ttl_bytes = reader.readColumnData(block.offset, ttl_desc, alloc) catch null;
            }
        }

        // Read extra scalar columns.
        var extra_col_data = try alloc.alloc(?[]const u8, params.extra_columns.len);
        defer {
            for (extra_col_data) |opt| {
                if (opt) |data| alloc.free(data);
            }
            alloc.free(extra_col_data);
        }
        for (params.extra_columns, 0..) |col_id, i| {
            // Skip if already read as dedup, ts, or filter column.
            const is_dedup = if (params.dedup_column_id) |dc| col_id == dc else false;
            if (is_dedup or col_id == params.ts_column_id) {
                extra_col_data[i] = null;
            } else if (findColumn(descs, col_id)) |desc| {
                extra_col_data[i] = reader.readColumnData(block.offset, desc, alloc) catch null;
            } else {
                extra_col_data[i] = null;
            }
        }

        // Read payload column (lazily — only decode for surviving rows).
        const pay_desc = findColumn(descs, params.payload_column_id) orelse return false;
        var payload_buffers = readVarBytesBuffers(file, block.offset, pay_desc, row_count, alloc) catch return false;
        defer payload_buffers.deinit(alloc);

        // Iterate rows.
        if (params.reverse) {
            var row_idx: usize = row_count;
            while (row_idx > 0) {
                row_idx -= 1;
                const hit_limit = try self.processRow(
                    alloc,
                    row_idx,
                    pk_bytes,
                    ts_bytes,
                    ttl_bytes,
                    filter_cols,
                    extra_col_data,
                    &payload_buffers,
                    params,
                    effective_limit,
                    now_ms,
                    tombstones,
                    seen,
                    results,
                );
                if (hit_limit) return true;
            }
        } else {
            for (0..row_count) |row_idx| {
                const hit_limit = try self.processRow(
                    alloc,
                    row_idx,
                    pk_bytes,
                    ts_bytes,
                    ttl_bytes,
                    filter_cols,
                    extra_col_data,
                    &payload_buffers,
                    params,
                    effective_limit,
                    now_ms,
                    tombstones,
                    seen,
                    results,
                );
                if (hit_limit) return true;
            }
        }

        return false;
    }

    fn processRow(
        self: *Table,
        alloc: Allocator,
        row_idx: usize,
        pk_bytes: ?[]const u8,
        ts_bytes: []const u8,
        ttl_bytes: ?[]const u8,
        filter_cols: []?[]const u8,
        extra_col_data: []?[]const u8,
        payload_buffers: *VarBytesBuffers,
        params: ScanParams,
        effective_limit: u32,
        now_ms: i64,
        tombstones: *std.AutoHashMap(u64, i64),
        seen: *std.AutoHashMap(u64, void),
        results: *std.ArrayList(Row),
    ) !bool {
        _ = self;

        const pk_val: ?u64 = if (pk_bytes) |pb| readU64At(pb, row_idx) catch null else null;
        const ts_val = readI64At(ts_bytes, row_idx) catch return false;

        // Dedup: skip if already seen (newer version already collected).
        if (params.dedup_column_id != null) {
            if (pk_val) |pv| {
                if (seen.get(pv) != null) return false;
            }
        }

        // Tombstone check (requires dedup/PK column).
        if (pk_val) |pv| {
            if (tombstones.get(pv)) |del_ts| {
                if (del_ts >= ts_val) return false;
            }
        }

        // TTL check.
        if (ttl_bytes) |tb| {
            const ttl_val = readI64At(tb, row_idx) catch return false;
            if (ttl_val > 0 and ttl_val < now_ms) return false;
        }

        // Cursor filter.
        if (params.cursor_ts) |cts| {
            if (ts_val > cts) return false;
            if (ts_val == cts) {
                // Hash tiebreak only when dedup is enabled.
                if (params.dedup_column_id != null) {
                    if (params.cursor_hash) |ch| {
                        if (pk_val) |pv| {
                            if (pv >= ch) return false;
                        }
                    }
                }
            }
        }

        // Apply column filters.
        for (params.filters, 0..) |filter, i| {
            const col_val: u64 = blk: {
                const is_dedup = if (params.dedup_column_id) |dc| filter.column_id == dc else false;
                if (is_dedup) {
                    break :blk pk_val orelse return false;
                } else if (filter.column_id == params.ts_column_id) {
                    break :blk @bitCast(ts_val);
                } else if (filter_cols[i]) |data| {
                    break :blk readU64At(data, row_idx) catch return false;
                } else {
                    return false;
                }
            };

            if (!evalFilter(filter.op, col_val, filter.value)) return false;
        }

        // Read payload.
        const payload_raw = payload_buffers.sliceForRow(row_idx) catch return false;

        // Build scalar list.
        var scalar_list = std.ArrayList(ColumnData).empty;
        errdefer scalar_list.deinit(alloc);

        // Add PK column if dedup is enabled.
        if (params.dedup_column_id) |dedup_col| {
            if (pk_val) |pv| {
                try scalar_list.append(alloc, .{ .column_id = dedup_col, .value_u64 = pv });
            }
        }
        // Always add timestamp.
        try scalar_list.append(alloc, .{ .column_id = params.ts_column_id, .value_u64 = @bitCast(ts_val) });

        // Add filter column values (skip dedup/ts already added).
        for (params.filters, 0..) |filter, idx| {
            const is_dedup = if (params.dedup_column_id) |dc| filter.column_id == dc else false;
            if (is_dedup or filter.column_id == params.ts_column_id) continue;
            if (filter_cols[idx]) |data| {
                const v = readU64At(data, row_idx) catch continue;
                try scalar_list.append(alloc, .{ .column_id = filter.column_id, .value_u64 = v });
            }
        }

        // Add extra columns requested by caller.
        for (params.extra_columns, 0..) |col_id, idx| {
            const is_dedup = if (params.dedup_column_id) |dc| col_id == dc else false;
            if (is_dedup or col_id == params.ts_column_id) continue;
            // Check if already added as a filter column.
            var already = false;
            for (params.filters) |f| {
                if (f.column_id == col_id) {
                    already = true;
                    break;
                }
            }
            if (already) continue;
            if (extra_col_data[idx]) |data| {
                const v = readU64At(data, row_idx) catch continue;
                try scalar_list.append(alloc, .{ .column_id = col_id, .value_u64 = v });
            }
        }

        const row = Row{
            .scalars = try scalar_list.toOwnedSlice(alloc),
            .payload = try alloc.dupe(u8, payload_raw),
        };

        try results.append(alloc, row);

        // Mark PK as seen for dedup.
        if (params.dedup_column_id != null) {
            if (pk_val) |pv| {
                try seen.put(pv, {});
            }
        }

        // Check limit.
        if (results.items.len >= effective_limit) return true;
        return false;
    }

    // =========================================================================
    // Internal: tombstone collection
    // =========================================================================

    fn collectTombstones(
        self: *Table,
        alloc: Allocator,
        del_schema: u16,
        pk_col: u32,
        ts_col: u32,
        tombstones: *std.AutoHashMap(u64, i64),
    ) !void {
        const blocks = try self.fs_reader.getBlocks(alloc);
        defer alloc.free(blocks);

        for (blocks) |block| {
            var file = self.fs_reader.dir.openFile(block.path, .{ .mode = .read_only }) catch continue;
            defer file.close();

            const reader = block_reader.BlockReader.init(file, alloc);
            const header = reader.readHeader(block.offset) catch continue;
            if (header.schema_id != del_schema) continue;

            const row_count = header.row_count;
            if (row_count == 0) continue;

            const descs = reader.readColumnDirectory(header, block.offset) catch continue;
            defer alloc.free(descs);

            const pk_desc = findColumn(descs, pk_col) orelse continue;
            const ts_desc = findColumn(descs, ts_col) orelse continue;

            const pk_bytes = reader.readColumnData(block.offset, pk_desc, alloc) catch continue;
            defer alloc.free(pk_bytes);
            const ts_bytes = reader.readColumnData(block.offset, ts_desc, alloc) catch continue;
            defer alloc.free(ts_bytes);

            for (0..row_count) |row_idx| {
                const pk_val = readU64At(pk_bytes, row_idx) catch continue;
                const ts_val = readI64At(ts_bytes, row_idx) catch continue;

                const existing = tombstones.get(pk_val);
                if (existing == null or existing.? < ts_val) {
                    tombstones.put(pk_val, ts_val) catch continue;
                }
            }
        }
    }
};

// =============================================================================
// Helpers
// =============================================================================

fn matchesSchema(block_schema: u16, primary: u16, additional: ?[]const u16) bool {
    if (block_schema == primary) return true;
    if (additional) |ids| {
        for (ids) |id| {
            if (block_schema == id) return true;
        }
    }
    return false;
}

fn evalFilter(op: FilterOp, col_val: u64, filter_val: u64) bool {
    return switch (op) {
        .eq => col_val == filter_val,
        .ne => col_val != filter_val,
        .lt => col_val < filter_val,
        .le => col_val <= filter_val,
        .gt => col_val > filter_val,
        .ge => col_val >= filter_val,
    };
}

pub fn findColumn(descs: []const types.ColumnDesc, col_id: u32) ?types.ColumnDesc {
    for (descs) |d| {
        if (d.column_id == col_id) return d;
    }
    return null;
}

pub fn readU64At(bytes: []const u8, row_idx: usize) !u64 {
    const offset = row_idx * 8;
    if (offset + 8 > bytes.len) return error.OutOfBounds;
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

pub fn readI64At(bytes: []const u8, row_idx: usize) !i64 {
    const offset = row_idx * 8;
    if (offset + 8 > bytes.len) return error.OutOfBounds;
    return std.mem.readInt(i64, bytes[offset..][0..8], .little);
}

pub const VarBytesBuffers = struct {
    data: []u8,
    offsets: []u32,
    lengths: []u32,

    pub fn deinit(self: *VarBytesBuffers, allocator: Allocator) void {
        allocator.free(self.data);
        allocator.free(self.offsets);
        allocator.free(self.lengths);
    }

    pub fn sliceForRow(self: VarBytesBuffers, row_idx: usize) ![]const u8 {
        if (row_idx >= self.offsets.len or row_idx >= self.lengths.len) return error.InvalidColumnData;
        const offset = self.offsets[row_idx];
        const length = self.lengths[row_idx];
        const start = @as(usize, offset);
        const end = start + @as(usize, length);
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

pub fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: types.ColumnDesc,
    row_count: u32,
    allocator: Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = block_reader.BlockReader.init(file, allocator);
    const data = try reader.readColumnData(block_offset, desc, allocator);
    errdefer allocator.free(data);

    const offsets = try readU32Array(file, block_offset + @as(u64, desc.offsets_off), row_count, allocator);
    errdefer allocator.free(offsets);

    const lengths = try readU32Array(file, block_offset + @as(u64, desc.lengths_off), row_count, allocator);

    return .{ .data = data, .offsets = offsets, .lengths = lengths };
}

fn readU32Array(file: std.fs.File, offset: u64, count: u32, allocator: Allocator) ![]u32 {
    const total_bytes = @as(usize, count) * @sizeOf(u32);
    const buffer = try allocator.alloc(u8, total_bytes);
    defer allocator.free(buffer);

    const read_len = try file.preadAll(buffer, offset);
    if (read_len != buffer.len) return error.UnexpectedEof;

    const values = try allocator.alloc(u32, count);
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        const start = i * 4;
        values[i] = std.mem.readInt(u32, buffer[start..][0..4], .little);
    }
    return values;
}

fn readAllScalars(
    alloc: Allocator,
    reader: block_reader.BlockReader,
    block_offset: u64,
    descs: []types.ColumnDesc,
    row_idx: usize,
) ![]ColumnData {
    var list = std.ArrayList(ColumnData).empty;
    errdefer list.deinit(alloc);

    for (descs) |desc| {
        const shape: types.ColumnShape = @enumFromInt(desc.shape);
        if (shape != .SCALAR) continue;

        const data = reader.readColumnData(block_offset, desc, alloc) catch continue;
        defer alloc.free(data);

        const val = readU64At(data, row_idx) catch continue;
        try list.append(alloc, .{ .column_id = desc.column_id, .value_u64 = val });
    }

    return list.toOwnedSlice(alloc);
}

pub fn freeRow(alloc: Allocator, row: Row) void {
    alloc.free(row.scalars);
    alloc.free(row.payload);
}

pub fn freeRows(alloc: Allocator, rows: []Row) void {
    for (rows) |row| freeRow(alloc, row);
    alloc.free(rows);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

/// Build a simple row with pk_hash (col 1, U64), ts (col 2, I64),
/// and payload (col 20, VARBYTES).
fn buildTestRow(
    pk_hash: *u64,
    ts: *i64,
    payload: []const u8,
) [3]ColumnValue {
    return .{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.asBytes(pk_hash),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.asBytes(ts),
        },
        .{
            .column_id = 20,
            .shape = .VARBYTES,
            .phys_type = .BINARY,
            .encoding = .RAW,
            .dims = 0,
            .data = payload,
        },
    };
}

/// Build a row with an extra filter column (col 3, U64).
fn buildTestRowWithFilter(
    pk_hash: *u64,
    ts: *i64,
    filter_val: *u64,
    payload: []const u8,
) [4]ColumnValue {
    return .{
        .{
            .column_id = 1,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.asBytes(pk_hash),
        },
        .{
            .column_id = 2,
            .shape = .SCALAR,
            .phys_type = .I64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.asBytes(ts),
        },
        .{
            .column_id = 3,
            .shape = .SCALAR,
            .phys_type = .U64,
            .encoding = .RAW,
            .dims = 1,
            .data = std.mem.asBytes(filter_val),
        },
        .{
            .column_id = 20,
            .shape = .VARBYTES,
            .phys_type = .BINARY,
            .encoding = .RAW,
            .dims = 0,
            .data = payload,
        },
    };
}

const test_schema: u16 = 100;
const test_delete_schema: u16 = 101;

fn testPolicy() CompactionPolicy {
    return .{
        .active_schema_ids = &[_]u16{test_schema},
        .tombstone_schema_id = test_delete_schema,
        .dedup_column_id = 1,
        .ts_column_id = 2,
    };
}

test "write_and_get" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    var pk: u64 = 42;
    var ts: i64 = 1000;
    const cols = buildTestRow(&pk, &ts, "hello world");
    try table.appendRow(test_schema, &cols);
    try table.flush();

    const row = try table.get(alloc, test_schema, 42, null);
    try testing.expect(row != null);
    defer freeRow(alloc, row.?);

    try testing.expectEqualStrings("hello world", row.?.payload);
}

test "scan_basic" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    var pk1: u64 = 1;
    var ts1: i64 = 100;
    const c1 = buildTestRow(&pk1, &ts1, "row-1");
    try table.appendRow(test_schema, &c1);

    var pk2: u64 = 2;
    var ts2: i64 = 200;
    const c2 = buildTestRow(&pk2, &ts2, "row-2");
    try table.appendRow(test_schema, &c2);

    var pk3: u64 = 3;
    var ts3: i64 = 300;
    const c3 = buildTestRow(&pk3, &ts3, "row-3");
    try table.appendRow(test_schema, &c3);
    try table.flush();

    const result = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, result.rows);

    try testing.expectEqual(@as(usize, 3), result.rows.len);
    try testing.expect(!result.has_more);

    // Reverse order: newest first.
    try testing.expectEqualStrings("row-3", result.rows[0].payload);
    try testing.expectEqualStrings("row-2", result.rows[1].payload);
    try testing.expectEqualStrings("row-1", result.rows[2].payload);
}

test "scan_with_filter" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    var pk1: u64 = 1;
    var ts1: i64 = 100;
    var group1: u64 = 10;
    const c1 = buildTestRowWithFilter(&pk1, &ts1, &group1, "group-10");
    try table.appendRow(test_schema, &c1);

    var pk2: u64 = 2;
    var ts2: i64 = 200;
    var group2: u64 = 20;
    const c2 = buildTestRowWithFilter(&pk2, &ts2, &group2, "group-20");
    try table.appendRow(test_schema, &c2);

    var pk3: u64 = 3;
    var ts3: i64 = 300;
    var group3: u64 = 10;
    const c3 = buildTestRowWithFilter(&pk3, &ts3, &group3, "group-10-b");
    try table.appendRow(test_schema, &c3);
    try table.flush();

    const filters = [_]ColumnFilter{
        .{ .column_id = 3, .op = .eq, .value = 10 },
    };
    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .filters = &filters,
    });
    defer freeRows(alloc, result.rows);

    try testing.expectEqual(@as(usize, 2), result.rows.len);
    try testing.expectEqualStrings("group-10-b", result.rows[0].payload);
    try testing.expectEqualStrings("group-10", result.rows[1].payload);
}

test "scan_range_filter" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    for (0..5) |i| {
        var pk: u64 = @intCast(i + 1);
        var ts: i64 = @intCast((i + 1) * 100);
        var val: u64 = @intCast((i + 1) * 10);
        const cols = buildTestRowWithFilter(&pk, &ts, &val, "row");
        try table.appendRow(test_schema, &cols);
    }
    try table.flush();

    // Filter: col 3 > 30 (should match rows with val=40, val=50)
    const filters = [_]ColumnFilter{
        .{ .column_id = 3, .op = .gt, .value = 30 },
    };
    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .filters = &filters,
    });
    defer freeRows(alloc, result.rows);

    try testing.expectEqual(@as(usize, 2), result.rows.len);
}

test "dedup_keeps_latest" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write 3 versions of the same PK.
    var pk: u64 = 42;
    var ts1: i64 = 100;
    const c1 = buildTestRow(&pk, &ts1, "version-1");
    try table.appendRow(test_schema, &c1);

    var ts2: i64 = 200;
    const c2 = buildTestRow(&pk, &ts2, "version-2");
    try table.appendRow(test_schema, &c2);

    var ts3: i64 = 300;
    const c3 = buildTestRow(&pk, &ts3, "version-3");
    try table.appendRow(test_schema, &c3);
    try table.flush();

    const result = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, result.rows);

    // Only the latest version should be returned.
    try testing.expectEqual(@as(usize, 1), result.rows.len);
    try testing.expectEqualStrings("version-3", result.rows[0].payload);
}

test "tombstone_hides_row" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    var pk: u64 = 42;
    var ts: i64 = 100;
    const cols = buildTestRow(&pk, &ts, "to-be-deleted");
    try table.appendRow(test_schema, &cols);

    // Write tombstone at ts=200 (after the row).
    try table.deleteTombstone(42, 200);
    try table.flush();

    // get() should return null.
    const row = try table.get(alloc, test_schema, 42, null);
    try testing.expect(row == null);

    // scan() should return empty.
    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .delete_schema_id = test_delete_schema,
    });
    defer freeRows(alloc, result.rows);
    try testing.expectEqual(@as(usize, 0), result.rows.len);
}

test "tombstone_only_hides_older" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write tombstone first at ts=100.
    try table.deleteTombstone(42, 100);

    // Then write the row at ts=200 (after the tombstone).
    var pk: u64 = 42;
    var ts: i64 = 200;
    const cols = buildTestRow(&pk, &ts, "alive");
    try table.appendRow(test_schema, &cols);
    try table.flush();

    // The row should be visible because it's newer than the tombstone.
    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .delete_schema_id = test_delete_schema,
    });
    defer freeRows(alloc, result.rows);
    try testing.expectEqual(@as(usize, 1), result.rows.len);
    try testing.expectEqualStrings("alive", result.rows[0].payload);
}

test "cursor_pagination" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write 6 rows with different timestamps.
    for (0..6) |i| {
        var pk: u64 = @intCast(i + 1);
        var ts: i64 = @intCast((i + 1) * 100);
        const cols = buildTestRow(&pk, &ts, "row");
        try table.appendRow(test_schema, &cols);
    }
    try table.flush();

    // Page 1: limit=3, no cursor.
    const page1 = try table.scan(alloc, .{
        .schema_id = test_schema,
        .limit = 3,
    });
    defer freeRows(alloc, page1.rows);
    try testing.expectEqual(@as(usize, 3), page1.rows.len);
    try testing.expect(page1.has_more);

    // Extract cursor from last row of page 1.
    const last_row = page1.rows[2];
    var cursor_ts_val: i64 = undefined;
    var cursor_hash_val: u64 = undefined;
    for (last_row.scalars) |s| {
        if (s.column_id == 2) cursor_ts_val = @bitCast(s.value_u64);
        if (s.column_id == 1) cursor_hash_val = s.value_u64;
    }

    // Page 2: limit=3, with cursor.
    const page2 = try table.scan(alloc, .{
        .schema_id = test_schema,
        .limit = 3,
        .cursor_ts = cursor_ts_val,
        .cursor_hash = cursor_hash_val,
    });
    defer freeRows(alloc, page2.rows);
    try testing.expectEqual(@as(usize, 3), page2.rows.len);
    try testing.expect(!page2.has_more);
}

test "additional_schema_ids" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const legacy_schema: u16 = 99;
    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write a row with the legacy schema.
    var pk: u64 = 1;
    var ts: i64 = 100;
    const c1 = buildTestRow(&pk, &ts, "legacy");
    try table.appendRow(legacy_schema, &c1);

    // Write a row with the primary schema.
    var pk2: u64 = 2;
    var ts2: i64 = 200;
    const c2 = buildTestRow(&pk2, &ts2, "modern");
    try table.appendRow(test_schema, &c2);
    try table.flush();

    // Scan with primary schema only — should find only modern.
    const result1 = try table.scan(alloc, .{
        .schema_id = test_schema,
    });
    defer freeRows(alloc, result1.rows);
    try testing.expectEqual(@as(usize, 1), result1.rows.len);
    try testing.expectEqualStrings("modern", result1.rows[0].payload);

    // Scan with additional_schema_ids — should find both.
    const additional = [_]u16{legacy_schema};
    const result2 = try table.scan(alloc, .{
        .schema_id = test_schema,
        .additional_schema_ids = &additional,
    });
    defer freeRows(alloc, result2.rows);
    try testing.expectEqual(@as(usize, 2), result2.rows.len);
}

test "legacy_hash_get" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write a row with pk_hash=42.
    var pk: u64 = 42;
    var ts: i64 = 100;
    const cols = buildTestRow(&pk, &ts, "legacy-row");
    try table.appendRow(test_schema, &cols);
    try table.flush();

    // get() with wrong primary hash but correct legacy hash.
    const row = try table.get(alloc, test_schema, 999, 42);
    try testing.expect(row != null);
    defer freeRow(alloc, row.?);
    try testing.expectEqualStrings("legacy-row", row.?.payload);
}

test "ttl_expiration" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const ttl_policy = CompactionPolicy{
        .active_schema_ids = &[_]u16{test_schema},
        .tombstone_schema_id = test_delete_schema,
        .dedup_column_id = 1,
        .ts_column_id = 2,
        .ttl_column_id = 3,
    };
    var table = try Table.open(alloc, path, "test", ttl_policy);
    defer table.deinit();

    // Write a row with TTL in the past (expired).
    var pk: u64 = 1;
    var ts: i64 = 100;
    var ttl_expired: u64 = 1; // TTL = 1ms, definitely expired
    const c1 = buildTestRowWithFilter(&pk, &ts, &ttl_expired, "expired");
    try table.appendRow(test_schema, &c1);

    // Write a row with TTL=0 (never expires).
    var pk2: u64 = 2;
    var ts2: i64 = 200;
    var ttl_never: u64 = 0;
    const c2 = buildTestRowWithFilter(&pk2, &ts2, &ttl_never, "permanent");
    try table.appendRow(test_schema, &c2);
    try table.flush();

    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .ttl_column_id = 3,
    });
    defer freeRows(alloc, result.rows);

    // Only the non-expired row should be visible.
    try testing.expectEqual(@as(usize, 1), result.rows.len);
    try testing.expectEqualStrings("permanent", result.rows[0].payload);
}

test "empty_table" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    const row = try table.get(alloc, test_schema, 42, null);
    try testing.expect(row == null);

    const result = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, result.rows);
    try testing.expectEqual(@as(usize, 0), result.rows.len);
    try testing.expect(!result.has_more);
}

test "multi_schema_write" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write with schema A.
    var pk1: u64 = 1;
    var ts1: i64 = 100;
    const c1 = buildTestRow(&pk1, &ts1, "schema-a");
    try table.appendRow(test_schema, &c1);

    // Write tombstone (different schema) — should flush schema A first.
    try table.deleteTombstone(99, 200);

    // Write another row with schema A.
    var pk2: u64 = 2;
    var ts2: i64 = 300;
    const c2 = buildTestRow(&pk2, &ts2, "schema-a-2");
    try table.appendRow(test_schema, &c2);
    try table.flush();

    // Both schema A rows should be scannable.
    const result = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, result.rows);
    try testing.expectEqual(@as(usize, 2), result.rows.len);
}

test "scan_limit_has_more" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    for (0..5) |i| {
        var pk: u64 = @intCast(i + 1);
        var ts: i64 = @intCast((i + 1) * 100);
        const cols = buildTestRow(&pk, &ts, "row");
        try table.appendRow(test_schema, &cols);
    }
    try table.flush();

    const result = try table.scan(alloc, .{
        .schema_id = test_schema,
        .limit = 2,
    });
    defer freeRows(alloc, result.rows);
    try testing.expectEqual(@as(usize, 2), result.rows.len);
    try testing.expect(result.has_more);
}

test "appendRow rejects missing dedup column" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Row with only ts (col 2) and payload (col 20), missing dedup (col 1).
    var ts: i64 = 100;
    const columns = [_]ColumnValue{
        .{ .column_id = 2, .shape = .SCALAR, .phys_type = .I64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&ts) },
        .{ .column_id = 20, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = "payload" },
    };

    try testing.expectError(error.MissingDedupColumn, table.appendRow(test_schema, &columns));
}

test "appendRow rejects missing timestamp column" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Row with only dedup (col 1) and payload (col 20), missing ts (col 2).
    var pk: u64 = 42;
    const columns = [_]ColumnValue{
        .{ .column_id = 1, .shape = .SCALAR, .phys_type = .U64, .encoding = .RAW, .dims = 1, .data = std.mem.asBytes(&pk) },
        .{ .column_id = 20, .shape = .VARBYTES, .phys_type = .BINARY, .encoding = .RAW, .dims = 0, .data = "payload" },
    };

    try testing.expectError(error.MissingTimestampColumn, table.appendRow(test_schema, &columns));
}

test "scan_without_dedup returns all rows including duplicates" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    // Write 3 rows with the same PK but different timestamps.
    var pk: u64 = 42;
    var ts1: i64 = 100;
    const c1 = buildTestRow(&pk, &ts1, "v1");
    try table.appendRow(test_schema, &c1);

    var ts2: i64 = 200;
    const c2 = buildTestRow(&pk, &ts2, "v2");
    try table.appendRow(test_schema, &c2);

    var ts3: i64 = 300;
    const c3 = buildTestRow(&pk, &ts3, "v3");
    try table.appendRow(test_schema, &c3);
    try table.flush();

    // With dedup: only latest version.
    const deduped = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, deduped.rows);
    try testing.expectEqual(@as(usize, 1), deduped.rows.len);

    // Without dedup: all 3 versions.
    const all = try table.scan(alloc, .{
        .schema_id = test_schema,
        .dedup_column_id = null,
    });
    defer freeRows(alloc, all.rows);
    try testing.expectEqual(@as(usize, 3), all.rows.len);
}

test "openReadOnly with policy supports tombstones in get" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    // Write data with a read-write table, then read back with read-only.
    {
        var table = try Table.open(alloc, path, "test", testPolicy());
        defer table.deinit();

        var pk: u64 = 42;
        var ts: i64 = 100;
        const cols = buildTestRow(&pk, &ts, "hello");
        try table.appendRow(test_schema, &cols);

        try table.deleteTombstone(42, 200);
        try table.flush();
    }

    // Open read-only WITH policy — should see tombstone.
    var ro = try Table.openReadOnly(alloc, path, "test", testPolicy());
    defer ro.deinit();

    const row = try ro.get(alloc, test_schema, 42, null);
    try testing.expect(row == null);
}

test "scan_extra_columns includes requested scalars" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    var table = try Table.open(alloc, path, "test", testPolicy());
    defer table.deinit();

    var pk: u64 = 1;
    var ts: i64 = 100;
    var extra_val: u64 = 999;
    const cols = buildTestRowWithFilter(&pk, &ts, &extra_val, "payload");
    try table.appendRow(test_schema, &cols);
    try table.flush();

    // Scan without extra_columns: only pk + ts in scalars.
    const r1 = try table.scan(alloc, .{ .schema_id = test_schema });
    defer freeRows(alloc, r1.rows);
    try testing.expectEqual(@as(usize, 1), r1.rows.len);
    try testing.expectEqual(@as(usize, 2), r1.rows[0].scalars.len);

    // Scan with extra_columns = col 3: pk + ts + col_3 in scalars.
    const extra = [_]u32{3};
    const r2 = try table.scan(alloc, .{
        .schema_id = test_schema,
        .extra_columns = &extra,
    });
    defer freeRows(alloc, r2.rows);
    try testing.expectEqual(@as(usize, 1), r2.rows.len);
    try testing.expectEqual(@as(usize, 3), r2.rows[0].scalars.len);

    // Verify the extra column value.
    var found_extra = false;
    for (r2.rows[0].scalars) |s| {
        if (s.column_id == 3) {
            try testing.expectEqual(@as(u64, 999), s.value_u64);
            found_extra = true;
        }
    }
    try testing.expect(found_extra);
}

test "persistPolicy_roundtrip" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    // Create namespace directory (normally done by Writer.open).
    const ns_path = try std.fmt.allocPrint(alloc, "{s}/ns", .{path});
    defer alloc.free(ns_path);
    try std.fs.cwd().makePath(ns_path);

    const policy = CompactionPolicy{
        .active_schema_ids = &[_]u16{ 10, 20 },
        .tombstone_schema_id = 30,
        .dedup_column_id = 1,
        .ts_column_id = 2,
        .ttl_column_id = 5,
    };

    try persistPolicy(alloc, path, "ns", policy);

    const loaded = (try loadPersistedPolicy(alloc, path, "ns")) orelse return error.TestUnexpectedResult;
    defer freeLoadedPolicy(alloc, loaded);

    try testing.expectEqual(@as(usize, 2), loaded.policy.active_schema_ids.len);
    try testing.expectEqual(@as(u16, 10), loaded.policy.active_schema_ids[0]);
    try testing.expectEqual(@as(u16, 20), loaded.policy.active_schema_ids[1]);
    try testing.expectEqual(@as(?u16, 30), loaded.policy.tombstone_schema_id);
    try testing.expectEqual(@as(u32, 1), loaded.policy.dedup_column_id);
    try testing.expectEqual(@as(u32, 2), loaded.policy.ts_column_id);
    try testing.expectEqual(@as(?u32, 5), loaded.policy.ttl_column_id);
}

test "loadPersistedPolicy_returns_null_when_missing" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const result = try loadPersistedPolicy(alloc, path, "ns");
    try testing.expect(result == null);
}

test "open_persists_policy_on_first_open_and_loads_on_second" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const policy = CompactionPolicy{
        .active_schema_ids = &[_]u16{ test_schema },
        .tombstone_schema_id = test_delete_schema,
        .ttl_column_id = 9,
    };

    // First open — should persist the policy.
    {
        var table = try Table.open(alloc, path, "test", policy);
        defer table.deinit();

        // Policy should match what we provided.
        try testing.expectEqual(@as(?u32, 9), table.policy.ttl_column_id);
    }

    // Verify policy.json was written under the namespace directory.
    const loaded = (try loadPersistedPolicy(alloc, path, "test")) orelse return error.TestUnexpectedResult;
    defer freeLoadedPolicy(alloc, loaded);
    try testing.expectEqual(@as(?u16, test_delete_schema), loaded.policy.tombstone_schema_id);
    try testing.expectEqual(@as(?u32, 9), loaded.policy.ttl_column_id);

    // Second open with a DIFFERENT policy — should load persisted one, ignoring the argument.
    {
        const different_policy = CompactionPolicy{
            .active_schema_ids = &[_]u16{99},
            .tombstone_schema_id = 88,
        };
        var table = try Table.open(alloc, path, "test", different_policy);
        defer table.deinit();

        // Should have loaded the persisted policy, not the different one.
        try testing.expectEqual(@as(?u16, test_delete_schema), table.policy.tombstone_schema_id);
        try testing.expectEqual(@as(?u32, 9), table.policy.ttl_column_id);
    }
}

test "openReadOnly_loads_persisted_policy" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const policy = CompactionPolicy{
        .active_schema_ids = &[_]u16{test_schema},
        .tombstone_schema_id = test_delete_schema,
        .ttl_column_id = 7,
    };

    // Write some data and persist policy via read-write open.
    {
        var table = try Table.open(alloc, path, "test", policy);
        defer table.deinit();

        var pk: u64 = 1;
        var ts: i64 = 100;
        const cols = buildTestRow(&pk, &ts, "data");
        try table.appendRow(test_schema, &cols);
        try table.flush();
    }

    // Open read-only with empty policy — should load persisted one.
    var ro = try Table.openReadOnly(alloc, path, "test", .{ .active_schema_ids = &.{} });
    defer ro.deinit();

    try testing.expectEqual(@as(?u16, test_delete_schema), ro.policy.tombstone_schema_id);
    try testing.expectEqual(@as(?u32, 7), ro.policy.ttl_column_id);
}

test "open_with_empty_active_schema_ids_does_not_persist_policy" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    // Simulate a DELETE-before-POST: open with a dummy policy that has
    // empty active_schema_ids. This must NOT be persisted to disk.
    {
        const dummy_policy = CompactionPolicy{
            .active_schema_ids = &.{},
            .tombstone_schema_id = 0,
        };
        var table = try Table.open(alloc, path, "test", dummy_policy);
        defer table.deinit();
    }

    // Verify no policy.json was written.
    const loaded = try loadPersistedPolicy(alloc, path, "test");
    try testing.expect(loaded == null);

    // Now open with a real policy — it should be persisted.
    const real_policy = CompactionPolicy{
        .active_schema_ids = &[_]u16{test_schema},
        .tombstone_schema_id = test_delete_schema,
        .ttl_column_id = 9,
    };
    {
        var table = try Table.open(alloc, path, "test", real_policy);
        defer table.deinit();

        try testing.expectEqual(@as(?u16, test_delete_schema), table.policy.tombstone_schema_id);
        try testing.expectEqual(@as(?u32, 9), table.policy.ttl_column_id);
    }

    // Confirm the real policy was persisted.
    const real_loaded = (try loadPersistedPolicy(alloc, path, "test")) orelse return error.TestUnexpectedResult;
    defer freeLoadedPolicy(alloc, real_loaded);
    try testing.expectEqual(@as(usize, 1), real_loaded.policy.active_schema_ids.len);
    try testing.expectEqual(@as(?u16, test_delete_schema), real_loaded.policy.tombstone_schema_id);
}

test "different_namespaces_get_separate_policies" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realpathAlloc(alloc, ".");
    defer alloc.free(path);

    const policy_a = CompactionPolicy{
        .active_schema_ids = &[_]u16{10},
        .tombstone_schema_id = 11,
        .ttl_column_id = 5,
    };
    const policy_b = CompactionPolicy{
        .active_schema_ids = &[_]u16{20},
        .tombstone_schema_id = 21,
        .ttl_column_id = 7,
    };

    // Open two tables with different namespaces and policies on the same db_root.
    {
        var table_a = try Table.open(alloc, path, "users", policy_a);
        defer table_a.deinit();
        try testing.expectEqual(@as(?u16, 11), table_a.policy.tombstone_schema_id);
    }
    {
        var table_b = try Table.open(alloc, path, "logs", policy_b);
        defer table_b.deinit();
        try testing.expectEqual(@as(?u16, 21), table_b.policy.tombstone_schema_id);
    }

    // Re-open both and verify each loads its own persisted policy.
    {
        const dummy = CompactionPolicy{ .active_schema_ids = &.{} };
        var table_a = try Table.open(alloc, path, "users", dummy);
        defer table_a.deinit();
        try testing.expectEqual(@as(?u16, 11), table_a.policy.tombstone_schema_id);
        try testing.expectEqual(@as(?u32, 5), table_a.policy.ttl_column_id);
    }
    {
        const dummy = CompactionPolicy{ .active_schema_ids = &.{} };
        var table_b = try Table.open(alloc, path, "logs", dummy);
        defer table_b.deinit();
        try testing.expectEqual(@as(?u16, 21), table_b.policy.tombstone_schema_id);
        try testing.expectEqual(@as(?u32, 7), table_b.policy.ttl_column_id);
    }
}
