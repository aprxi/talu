//! Query execution — scanning, filtering, parallel content matching.
//!
//! All functions take an `fs_reader` parameter instead of `*TableAdapter`, so
//! they are fully decoupled from the adapter struct.

const std = @import("std");
const kvbuf = @import("../../../io/kvbuf/root.zig");
const db_reader = @import("../../reader.zig");
const db_blob_store = @import("../../blob/store.zig");
const block_reader = @import("../../block_reader.zig");
const types = @import("../../types.zig");
const parallel = @import("../../../compute/parallel.zig");

const codec = @import("codec.zig");
const search = @import("search.zig");
const helpers = @import("helpers.zig");

const Allocator = std.mem.Allocator;
const ScannedSessionRecord = codec.ScannedSessionRecord;

const schema_items: u16 = 3;
const schema_sessions: u16 = 4;
const schema_sessions_kvbuf: u16 = 5;
const record_json_trigram_bloom_bytes: usize = 64;

const col_session_hash: u32 = 3;
const col_ts: u32 = 2;
const col_group_hash: u32 = 6;
const col_payload: u32 = 20;

fn isSessionSchema(schema_id: u16) bool {
    return schema_id == schema_sessions or schema_id == schema_sessions_kvbuf;
}

// ============================================================================
// ScanParams
// ============================================================================

/// Parameters for scanSessions pagination and filtering.
pub const ScanParams = struct {
    /// Single-session lookup (hash-based early exit).
    target_hash: ?u64 = null,
    /// Composite cursor: timestamp component (0 = no cursor).
    before_ts: ?i64 = null,
    /// Composite cursor: session hash component.
    before_session_hash: ?u64 = null,
    /// Group filter: scalar prune on col_group_hash.
    target_group_hash: ?u64 = null,
    /// Group filter: exact string verification after deserialization.
    target_group_id: ?[]const u8 = null,
    /// Maximum number of results (0 = no limit).
    limit: u32 = 0,
    /// Case-insensitive substring filter on session metadata and item content.
    search_query: ?[]const u8 = null,
    /// Session hashes with item-content matches, mapping hash → snippet text.
    /// Internal field — set automatically by scanSessionsFiltered, not by callers.
    content_match_hashes: ?*const std.AutoHashMap(u64, []const u8) = null,
    /// Maximum rows to decode before stopping (0 = unlimited).
    /// Prevents unbounded blocking on large DBs when search_query is selective.
    max_scan: u32 = 0,
    /// Tags filter: space-separated tags for exact word matching on tags_text.
    /// Matches sessions whose tags_text contains ALL specified tags (AND logic).
    /// Use tags_filter_any for OR logic.
    tags_filter: ?[]const u8 = null,
    /// Tags filter: space-separated tags for OR matching.
    /// Matches sessions whose tags_text contains ANY of the specified tags.
    /// Mutually exclusive with tags_filter (tags_filter takes precedence).
    tags_filter_any: ?[]const u8 = null,

    // --- Additional filters for /v1/search ---

    /// Marker filter: exact match on session marker field.
    marker_filter: ?[]const u8 = null,
    /// Marker filter: space-separated markers for OR matching.
    /// Matches sessions whose marker equals ANY of the specified values.
    marker_filter_any: ?[]const u8 = null,
    /// Model filter: case-insensitive prefix/substring match.
    /// Supports wildcards: "qwen*" matches "qwen3-0.6b", "*llama*" matches "meta-llama-3".
    model_filter: ?[]const u8 = null,
    /// Created after timestamp filter (inclusive, milliseconds).
    created_after_ms: ?i64 = null,
    /// Created before timestamp filter (exclusive, milliseconds).
    created_before_ms: ?i64 = null,
    /// Updated after timestamp filter (inclusive, milliseconds).
    updated_after_ms: ?i64 = null,
    /// Updated before timestamp filter (exclusive, milliseconds).
    updated_before_ms: ?i64 = null,
    /// Has tags filter: true = must have at least one tag, false = must have no tags.
    has_tags: ?bool = null,
    /// Source document ID filter: exact match on source_doc_id field.
    /// Matches sessions that were created from this prompt document.
    source_doc_id: ?[]const u8 = null,

    /// Build scan params from pagination and filter arguments.
    /// Translates optional cursor/filter slices into hash-based params.
    pub fn fromArgs(
        limit: u32,
        before_updated_at_ms: i64,
        before_session_id: ?[]const u8,
        group_id: ?[]const u8,
    ) ScanParams {
        var params = ScanParams{ .limit = limit };
        if (before_updated_at_ms != 0) {
            params.before_ts = before_updated_at_ms;
            if (before_session_id) |sid| {
                params.before_session_hash = codec.computeSessionHash(sid);
            }
        }
        if (group_id) |gid| {
            params.target_group_hash = codec.computeGroupHash(gid);
            params.target_group_id = gid;
        }
        return params;
    }
};

// ============================================================================
// Core scan implementation
// ============================================================================

/// Scan session records from the database, returning only the latest
/// non-deleted version of each session. Standalone version (no self).
pub fn scanSessionsFiltered(
    fs_reader: *db_reader.Reader,
    blob_store_opt: ?*db_blob_store.BlobStore,
    alloc: Allocator,
    params: ScanParams,
) ![]ScannedSessionRecord {
    // Pre-compute content matches when search_query is active.
    var content_hashes: ?std.AutoHashMap(u64, []const u8) = null;
    defer if (content_hashes) |*h| {
        var it = h.valueIterator();
        while (it.next()) |v| alloc.free(v.*);
        h.deinit();
    };

    if (params.search_query) |query| {
        if (query.len > 0) {
            content_hashes = try collectContentMatchHashes(fs_reader, blob_store_opt, alloc, query);
        }
    }

    var effective_params = params;
    if (content_hashes) |*h| {
        effective_params.content_match_hashes = h;
    }

    var seen = std.AutoHashMap(u64, void).init(alloc);
    defer seen.deinit();

    var results = std.ArrayList(ScannedSessionRecord).empty;
    errdefer {
        for (results.items) |*record| {
            codec.freeScannedSessionRecord(alloc, record);
        }
        results.deinit(alloc);
    }

    var found_target = false;
    const limit_reached = &found_target;
    var scanned: u32 = 0;

    // STEP 1: Scan current.talu (newest data, iterate blocks in reverse)
    if (fs_reader.current_path) |current_path| {
        const file = fs_reader.dir.openFile(current_path, .{ .mode = .read_only }) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (file) |f| {
            defer f.close();

            const stat = try f.stat();
            const reader = block_reader.BlockReader.init(f, alloc);

            const block_index = try reader.getBlockIndex(stat.size);
            defer alloc.free(block_index);

            // Iterate in REVERSE order (newest blocks first)
            var idx: usize = block_index.len;
            while (idx > 0) {
                idx -= 1;
                const entry = block_index[idx];

                if (!isSessionSchema(entry.schema_id)) continue;

                limit_reached.* = try processBlock(f, reader, entry.block_off, alloc, effective_params, &seen, &results, &scanned);
                if (limit_reached.*) break;
            }
        }
    }

    if (limit_reached.*) return results.toOwnedSlice(alloc);

    // STEP 2: Scan sealed segments (REVERSE order - newest segments first)
    if (fs_reader.manifest_data) |manifest| {
        var seg_idx: usize = manifest.segments.len;
        while (seg_idx > 0) {
            seg_idx -= 1;
            const segment = manifest.segments[seg_idx];

            var file = fs_reader.dir.openFile(segment.path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            defer file.close();

            const stat = try file.stat();
            const reader = block_reader.BlockReader.init(file, alloc);

            const block_index = try reader.getBlockIndex(stat.size);
            defer alloc.free(block_index);

            var blk_idx: usize = block_index.len;
            while (blk_idx > 0) {
                blk_idx -= 1;
                const entry = block_index[blk_idx];

                if (!isSessionSchema(entry.schema_id)) continue;

                limit_reached.* = try processBlock(file, reader, entry.block_off, alloc, effective_params, &seen, &results, &scanned);
                if (limit_reached.*) break;
            }

            if (limit_reached.*) break;
        }
    }

    return results.toOwnedSlice(alloc);
}

/// Process a single block for session records.
/// Returns true if target was found, limit was reached, or scan budget exhausted (early exit).
fn processBlock(
    file: std.fs.File,
    reader: block_reader.BlockReader,
    block_offset: u64,
    alloc: Allocator,
    params: ScanParams,
    seen: *std.AutoHashMap(u64, void),
    results: *std.ArrayList(ScannedSessionRecord),
    scanned: *u32,
) !bool {
    const header = try reader.readHeader(block_offset);
    if (header.row_count == 0) return false;

    const descs = try reader.readColumnDirectory(header, block_offset);
    defer alloc.free(descs);

    // 1. Find column descriptors for scalar-first pruning
    const hash_desc = helpers.findColumn(descs, col_session_hash) orelse return error.MissingColumn;
    const payload_desc = helpers.findColumn(descs, col_payload) orelse return error.MissingColumn;

    // 2. Read scalar columns
    const hash_bytes = try reader.readColumnData(block_offset, hash_desc, alloc);
    defer alloc.free(hash_bytes);
    const rows = try helpers.checkedRowCount(header.row_count, hash_bytes.len, 8);

    // Read ts column if cursor or group filtering is needed
    const needs_ts = params.before_ts != null;
    const needs_group_hash = params.target_group_hash != null;

    const ts_bytes: ?[]const u8 = if (needs_ts) blk: {
        const ts_desc = helpers.findColumn(descs, col_ts) orelse return error.MissingColumn;
        break :blk try reader.readColumnData(block_offset, ts_desc, alloc);
    } else null;
    defer if (ts_bytes) |b| alloc.free(b);

    const group_hash_bytes: ?[]const u8 = if (needs_group_hash) blk: {
        const gh_desc = helpers.findColumn(descs, col_group_hash) orelse return error.MissingColumn;
        break :blk try reader.readColumnData(block_offset, gh_desc, alloc);
    } else null;
    defer if (group_hash_bytes) |b| alloc.free(b);

    var new_indices = std.ArrayList(usize).empty;
    defer new_indices.deinit(alloc);

    // 3. Scalar-first pruning, rows iterated in REVERSE (newest first)
    var row_idx: usize = rows;
    while (row_idx > 0) {
        row_idx -= 1;

        const hash = try helpers.readU64At(hash_bytes, row_idx);

        // Session hash filter (single-session lookup)
        if (params.target_hash) |target| {
            if (hash != target) continue;
        }

        // Composite cursor filter (before_ts, before_session_hash)
        if (params.before_ts) |cursor_ts| {
            const ts = try helpers.readI64At(ts_bytes.?, row_idx);
            if (ts > cursor_ts) continue; // strictly newer → skip
            if (ts == cursor_ts) {
                if (params.before_session_hash) |cursor_hash| {
                    if (hash >= cursor_hash) continue; // tie-break: skip >= cursor hash
                }
            }
        }

        // Group hash filter (scalar prune)
        if (params.target_group_hash) |gh| {
            const row_gh = try helpers.readU64At(group_hash_bytes.?, row_idx);
            if (row_gh != gh) continue;
        }

        const gop = try seen.getOrPut(hash);
        if (!gop.found_existing) {
            try new_indices.append(alloc, row_idx);
        }
    }

    if (new_indices.items.len == 0) return false;

    // 4. Read PAYLOAD column only for surviving rows
    var payload_buffers = try helpers.readVarBytesBuffers(file, block_offset, payload_desc, header.row_count, alloc);
    defer payload_buffers.deinit(alloc);

    for (new_indices.items) |idx| {
        const payload = try payload_buffers.sliceForRow(idx);
        // Detect format: KvBuf (Schema 5) vs msgpack (Schema 4)
        var record = if (kvbuf.isKvBuf(payload))
            try codec.decodeSessionRecordKvBuf(alloc, payload)
        else
            try codec.decodeSessionRecordMsgpack(alloc, payload);
        errdefer codec.freeScannedSessionRecord(alloc, &record);

        // 5. Scan budget: count every decoded row (before any post-decode filter)
        scanned.* += 1;

        // 6. Tombstone check: skip if deleted
        if (record.marker) |marker| {
            if (std.mem.eql(u8, marker, "deleted")) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 7. Group ID exact string verification (hash collision safety)
        if (params.target_group_id) |target_gid| {
            const record_gid = record.group_id orelse "";
            if (!std.mem.eql(u8, record_gid, target_gid)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8. Text search filter: metadata fields + pre-computed item content matches
        if (params.search_query) |query| {
            const row_hash = try helpers.readU64At(hash_bytes, idx);
            const content_snippet = if (params.content_match_hashes) |set| set.get(row_hash) else null;
            if (content_snippet) |snippet| {
                // Content matched — attach snippet to the record.
                record.search_snippet = try alloc.dupe(u8, snippet);
            } else {
                const match_title = if (record.title) |t| search.textContainsInsensitive(t, query) else false;
                const match_model = if (record.model) |m| search.textContainsInsensitive(m, query) else false;
                const match_prompt = if (record.system_prompt) |s| search.textContainsInsensitive(s, query) else false;
                const match_tags = if (record.tags_text) |t| search.textContainsInsensitive(t, query) else false;
                if (!match_title and !match_model and !match_prompt and !match_tags) {
                    codec.freeScannedSessionRecord(alloc, &record);
                    if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                    continue;
                }
                // Metadata match — search_snippet stays null.
            }
        }

        // 8b. Tags filter: exact word matching (AND or OR logic)
        if (params.tags_filter) |filter| {
            const tags = record.tags_text orelse "";
            if (!search.tagsMatchAll(tags, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        } else if (params.tags_filter_any) |filter| {
            const tags = record.tags_text orelse "";
            if (!search.tagsMatchAny(tags, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8c. Marker filter: exact match or OR logic
        if (params.marker_filter) |filter| {
            const marker = record.marker orelse "";
            if (!std.mem.eql(u8, marker, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        } else if (params.marker_filter_any) |filter| {
            const marker = record.marker orelse "";
            if (!search.markerMatchAny(marker, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8d. Model filter: case-insensitive with optional wildcard
        if (params.model_filter) |filter| {
            const model = record.model orelse "";
            if (!search.modelMatchesFilter(model, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8e. Timestamp filters
        if (params.created_after_ms) |ts| {
            if (record.created_at_ms < ts) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }
        if (params.created_before_ms) |ts| {
            if (record.created_at_ms >= ts) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }
        if (params.updated_after_ms) |ts| {
            if (record.updated_at_ms < ts) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }
        if (params.updated_before_ms) |ts| {
            if (record.updated_at_ms >= ts) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8f. Has tags filter
        if (params.has_tags) |want_tags| {
            const has = if (record.tags_text) |t| t.len > 0 else false;
            if (has != want_tags) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        // 8g. Source document ID filter: exact match
        if (params.source_doc_id) |filter| {
            const doc_id = record.source_doc_id orelse "";
            if (!std.mem.eql(u8, doc_id, filter)) {
                codec.freeScannedSessionRecord(alloc, &record);
                if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
                continue;
            }
        }

        try results.append(alloc, record);

        // 9. Early exit: single-session lookup
        if (params.target_hash != null) return true;

        // 10. Limit check
        if (params.limit > 0 and results.items.len >= params.limit) return true;

        // 11. Scan budget check (after successful append — budget may expire mid-block)
        if (params.max_scan > 0 and scanned.* >= params.max_scan) return true;
    }

    return false;
}

// ============================================================================
// Parallel Content Search
// ============================================================================

/// Work item for parallel content scanning — one block to process.
const BlockWork = struct {
    file_idx: u16,
    block_offset: u64,
};

/// Shared context for parallel scan workers. All fields are read-only
/// from the workers' perspective except `maps` (one per work item, no contention).
const ParallelScanCtx = struct {
    opened_files: []std.fs.File,
    work_items: []const BlockWork,
    query: []const u8,
    blob_store_opt: ?*db_blob_store.BlobStore,
    maps: []std.AutoHashMap(u64, []const u8),
    alloc: Allocator,
};

/// Worker function for parallelFor — processes a range of BlockWork items.
/// Each work item gets its own hashmap, so no write contention between threads.
fn parallelScanWorker(start: usize, end: usize, ctx: *ParallelScanCtx) void {
    for (start..end) |i| {
        const work = ctx.work_items[i];
        const file = ctx.opened_files[work.file_idx];
        const reader = block_reader.BlockReader.init(file, ctx.alloc);
        scanItemBlockForContent(
            file,
            reader,
            work.block_offset,
            ctx.alloc,
            ctx.query,
            ctx.blob_store_opt,
            &ctx.maps[i],
        ) catch continue;
    }
}

/// Scan a single item block for content matches.
/// For each row where the payload contains the query (case-insensitive),
/// extracts a snippet and stores it keyed by session_hash (first match wins).
///
/// For KvBuf payloads: reads only the content_text field (zero-copy, no JSON parsing).
/// For legacy JSON payloads: falls back to raw substring scan + text extraction.
///
/// Thread safety: safe for concurrent calls on different maps sharing the same
/// file descriptor (all I/O uses preadAll).
fn scanItemBlockForContent(
    file: std.fs.File,
    reader: block_reader.BlockReader,
    block_offset: u64,
    alloc: Allocator,
    query: []const u8,
    blob_store_opt: ?*db_blob_store.BlobStore,
    matches: *std.AutoHashMap(u64, []const u8),
) !void {
    const header = try reader.readHeader(block_offset);
    if (header.row_count == 0) return;

    const descs = try reader.readColumnDirectory(header, block_offset);
    defer alloc.free(descs);

    const session_desc = helpers.findColumn(descs, col_session_hash) orelse return error.MissingColumn;
    const payload_desc = helpers.findColumn(descs, col_payload) orelse return error.MissingColumn;

    const session_bytes = try reader.readColumnData(block_offset, session_desc, alloc);
    defer alloc.free(session_bytes);
    const rows = try helpers.checkedRowCount(header.row_count, session_bytes.len, 8);

    var payload_buffers = try helpers.readVarBytesBuffers(file, block_offset, payload_desc, header.row_count, alloc);
    defer payload_buffers.deinit(alloc);

    for (0..rows) |row_idx| {
        const payload = payload_buffers.sliceForRow(row_idx) catch continue;

        const hash = helpers.readU64At(session_bytes, row_idx) catch continue;
        // First match per session wins — skip if already seen.
        if (matches.contains(hash)) continue;

        if (kvbuf.isKvBuf(payload)) {
            // KvBuf path: zero-copy field access — scan only content bytes.
            const kvbuf_reader = kvbuf.KvBufReader.init(payload) catch continue;
            if (kvbuf_reader.get(kvbuf.FieldIds.content_text)) |content_text| {
                if (try search.extractSnippet(content_text, query, alloc)) |snippet| {
                    try matches.put(hash, snippet);
                }
                continue;
            }
            if (kvbuf_reader.get(kvbuf.FieldIds.record_json)) |record_json| {
                if (search.textFindInsensitive(record_json, query) == null) continue;
                const clean_text = try search.extractTextFromPayload(record_json, alloc);
                defer if (clean_text) |t| alloc.free(t);
                if (clean_text) |text| {
                    if (try search.extractSnippet(text, query, alloc)) |snippet| {
                        try matches.put(hash, snippet);
                    }
                }
                continue;
            }
            if (kvbuf_reader.get(kvbuf.FieldIds.record_json_ref)) |record_json_ref| {
                const blob_store = blob_store_opt orelse continue;
                const trigram_bloom = kvbuf_reader.get(kvbuf.FieldIds.record_json_trigram_bloom);
                if (!mayContainSubstringByTrigramBloom(trigram_bloom, query)) continue;

                const loaded_json_opt = try readBlobForSearch(blob_store, record_json_ref, alloc);
                const loaded_json = loaded_json_opt orelse continue;
                defer alloc.free(loaded_json);
                if (search.textFindInsensitive(loaded_json, query) == null) continue;

                const clean_text = try search.extractTextFromPayload(loaded_json, alloc);
                defer if (clean_text) |t| alloc.free(t);
                if (clean_text) |text| {
                    if (try search.extractSnippet(text, query, alloc)) |snippet| {
                        try matches.put(hash, snippet);
                    }
                }
            }
        } else {
            // Legacy JSON path: raw substring pre-filter + JSON text extraction.
            if (search.textFindInsensitive(payload, query) == null) continue;

            const clean_text = try search.extractTextFromPayload(payload, alloc);
            defer if (clean_text) |t| alloc.free(t);

            if (clean_text) |text| {
                if (try search.extractSnippet(text, query, alloc)) |snippet| {
                    try matches.put(hash, snippet);
                }
            }
        }
    }
}

fn mayContainSubstringByTrigramBloom(bloom_opt: ?[]const u8, query: []const u8) bool {
    if (query.len < 3) return true;
    const bloom = bloom_opt orelse return true;
    if (bloom.len != record_json_trigram_bloom_bytes) return true;

    var i: usize = 0;
    while (i + 2 < query.len) : (i += 1) {
        const t0 = std.ascii.toLower(query[i]);
        const t1 = std.ascii.toLower(query[i + 1]);
        const t2 = std.ascii.toLower(query[i + 2]);
        const h1, const h2 = trigramHashes(t0, t1, t2);
        if (!isBloomBitSet(bloom, h1) or !isBloomBitSet(bloom, h2)) return false;
    }
    return true;
}

fn readBlobForSearch(
    blob_store: *db_blob_store.BlobStore,
    blob_ref: []const u8,
    alloc: Allocator,
) !?[]u8 {
    return blob_store.readAll(blob_ref, alloc) catch |err| switch (err) {
        // Search should stay best-effort for missing/corrupt refs.
        error.FileNotFound, error.InvalidBlobRef => null,
        // Unexpected failures (for example, OOM) should not be hidden.
        else => return err,
    };
}

fn trigramHashes(t0: u8, t1: u8, t2: u8) struct { usize, usize } {
    const trigram = [_]u8{ t0, t1, t2 };
    const bit_count = record_json_trigram_bloom_bytes * 8;
    const h1: usize = @intCast(std.hash.Wyhash.hash(0, &trigram) % bit_count);
    const h2: usize = @intCast(std.hash.Wyhash.hash(0x9e3779b97f4a7c15, &trigram) % bit_count);
    return .{ h1, h2 };
}

fn isBloomBitSet(bloom: []const u8, bit_idx: usize) bool {
    const byte_idx = bit_idx / 8;
    const bit_mask: u8 = @as(u8, 1) << @as(u3, @intCast(bit_idx % 8));
    return (bloom[byte_idx] & bit_mask) != 0;
}

/// Scan all item blocks and collect session_hash values where any item's
/// raw payload contains the search query (case-insensitive).
/// Standalone version (no self).
pub fn collectContentMatchHashes(
    fs_reader: *db_reader.Reader,
    blob_store_opt: ?*db_blob_store.BlobStore,
    alloc: Allocator,
    query: []const u8,
) !std.AutoHashMap(u64, []const u8) {
    // Phase 1: Inventory — open files and collect item block offsets.
    var opened_files = std.ArrayList(std.fs.File).empty;
    defer {
        for (opened_files.items) |f| f.close();
        opened_files.deinit(alloc);
    }

    var work_items = std.ArrayList(BlockWork).empty;
    defer work_items.deinit(alloc);

    // current.talu
    if (fs_reader.current_path) |current_path| {
        const file = fs_reader.dir.openFile(current_path, .{ .mode = .read_only }) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        if (file) |f| {
            const file_idx: u16 = @intCast(opened_files.items.len);
            try opened_files.append(alloc, f);

            const stat = try f.stat();
            const reader = block_reader.BlockReader.init(f, alloc);
            const block_index = try reader.getBlockIndex(stat.size);
            defer alloc.free(block_index);

            for (block_index) |entry| {
                if (entry.schema_id != schema_items) continue;
                try work_items.append(alloc, .{
                    .file_idx = file_idx,
                    .block_offset = entry.block_off,
                });
            }
        }
    }

    // Sealed segments
    if (fs_reader.manifest_data) |manifest| {
        for (manifest.segments) |segment| {
            const file = fs_reader.dir.openFile(segment.path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            const file_idx: u16 = @intCast(opened_files.items.len);
            try opened_files.append(alloc, file);

            const stat = try file.stat();
            const reader = block_reader.BlockReader.init(file, alloc);
            const block_index = try reader.getBlockIndex(stat.size);
            defer alloc.free(block_index);

            for (block_index) |entry| {
                if (entry.schema_id != schema_items) continue;
                try work_items.append(alloc, .{
                    .file_idx = file_idx,
                    .block_offset = entry.block_off,
                });
            }
        }
    }

    const n_work = work_items.items.len;
    if (n_work == 0) return std.AutoHashMap(u64, []const u8).init(alloc);

    // Phase 2: Allocate per-work-item hashmaps and run parallel scan.
    const maps = try alloc.alloc(std.AutoHashMap(u64, []const u8), n_work);
    defer alloc.free(maps);
    for (maps) |*m| m.* = std.AutoHashMap(u64, []const u8).init(alloc);

    // Cleanup: on error, free all snippets in per-work-item maps.
    var maps_owned = true;
    defer if (maps_owned) {
        for (maps) |*m| {
            var it = m.valueIterator();
            while (it.next()) |v| alloc.free(v.*);
            m.deinit();
        }
    };

    var ctx = ParallelScanCtx{
        .opened_files = opened_files.items,
        .work_items = work_items.items,
        .query = query,
        .blob_store_opt = blob_store_opt,
        .maps = maps,
        .alloc = alloc,
    };

    parallel.global().parallelFor(n_work, parallelScanWorker, &ctx);

    // Phase 3: Merge per-work-item maps into final result.
    var matches = std.AutoHashMap(u64, []const u8).init(alloc);
    errdefer {
        var it = matches.valueIterator();
        while (it.next()) |v| alloc.free(v.*);
        matches.deinit();
    }

    for (maps) |*m| {
        var it = m.iterator();
        while (it.next()) |entry| {
            const gop = try matches.getOrPut(entry.key_ptr.*);
            if (!gop.found_existing) {
                // Transfer ownership — don't free from per-work-item map.
                gop.value_ptr.* = entry.value_ptr.*;
            } else {
                // Duplicate session hash — free the snippet we won't use.
                alloc.free(entry.value_ptr.*);
            }
        }
        // Deinit the per-work-item map without freeing values (transferred or freed above).
        m.deinit();
    }
    maps_owned = false;

    return matches;
}
