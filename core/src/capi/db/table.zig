//! DB C-API: Table operations.
//!
//! Canonical exports use `talu_db_table_*` and include document APIs plus
//! session/tag table operations.

const std = @import("std");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const docs = @import("../documents_impl.zig");
const responses_mod = @import("../responses.zig");
const db = @import("../../db/root.zig");
const log = @import("../../log.zig");
const helpers = @import("helpers.zig");
const ops = @import("ops.zig");

const allocator = std.heap.c_allocator;
const optSlice = helpers.optSlice;
const validateDbPath = helpers.validateDbPath;
const validateRequiredArg = helpers.validateRequiredArg;
const setArgError = helpers.setArgError;
const ChatHandle = responses_mod.ChatHandle;
const Chat = @import("../../responses/root.zig").Chat;

// =============================================================================
// Type re-exports: Documents
// =============================================================================

pub const CDocumentRecord = docs.CDocumentRecord;
pub const CDocumentSummary = docs.CDocumentSummary;
pub const CDocumentList = docs.CDocumentList;
pub const CStringList = docs.CStringList;
pub const CSearchResult = docs.CSearchResult;
pub const CSearchResultList = docs.CSearchResultList;
pub const CChangeRecord = docs.CChangeRecord;
pub const CChangeList = docs.CChangeList;
pub const CDeltaChain = docs.CDeltaChain;
pub const CCompactionStats = docs.CCompactionStats;

// =============================================================================
// Type re-exports: Sessions
// =============================================================================

pub const CSessionRecord = ops.CSessionRecord;

pub const CSessionList = extern struct {
    sessions: ?[*]CSessionRecord,
    count: usize,
    total: usize,
    _allocator: ?*anyopaque,
    _arena: ?*anyopaque,
};

// =============================================================================
// Type definitions: Tags
// =============================================================================

pub const CTagRecord = extern struct {
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    created_at_ms: i64,
    updated_at_ms: i64,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

pub const CTagList = extern struct {
    tags: ?[*]CTagRecord,
    count: usize,
    _arena: ?*anyopaque,
};

pub const CRelationStringList = extern struct {
    strings: ?[*]?[*:0]const u8,
    count: usize,
    _arena: ?*anyopaque,
};

/// Batch tag lookup result — flat parallel arrays of (session_id, tag_id) pairs.
pub const CSessionTagBatch = extern struct {
    /// session_ids[i] is paired with tag_ids[i]. Sentinel-terminated strings.
    session_ids: ?[*]?[*:0]const u8,
    /// tag_ids[i] belongs to session_ids[i]. Sentinel-terminated strings.
    tag_ids: ?[*]?[*:0]const u8,
    count: usize,
    _arena: ?*anyopaque,
};

// =============================================================================
// Documents
// =============================================================================

pub export fn talu_db_table_create(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    title: ?[*:0]const u8,
    doc_json: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    parent_id: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    owner_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_create(
        db_path,
        doc_id,
        doc_type,
        title,
        doc_json,
        tags_text,
        parent_id,
        marker,
        group_id,
        owner_id,
    );
}

pub export fn talu_db_table_get(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_doc: *docs.CDocumentRecord,
) callconv(.c) i32 {
    return docs.talu_documents_get(db_path, doc_id, out_doc);
}

pub export fn talu_db_table_get_blob_ref(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_blob_ref: ?[*]u8,
    out_blob_ref_capacity: usize,
    out_has_external_ref: ?*bool,
) callconv(.c) i32 {
    return docs.talu_documents_get_blob_ref(
        db_path,
        doc_id,
        out_blob_ref,
        out_blob_ref_capacity,
        out_has_external_ref,
    );
}

pub export fn talu_db_table_update(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    doc_json: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    marker: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_update(db_path, doc_id, title, doc_json, tags_text, marker);
}

pub export fn talu_db_table_delete(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_delete(db_path, doc_id);
}

pub export fn talu_db_table_delete_batch(
    db_path: ?[*:0]const u8,
    doc_ids: ?[*]const ?[*:0]const u8,
    doc_ids_count: usize,
    doc_type: ?[*:0]const u8,
    out_deleted_count: ?*usize,
) callconv(.c) i32 {
    return docs.talu_documents_delete_batch(
        db_path,
        doc_ids,
        doc_ids_count,
        doc_type,
        out_deleted_count,
    );
}

pub export fn talu_db_table_set_marker_batch(
    db_path: ?[*:0]const u8,
    doc_ids: ?[*]const ?[*:0]const u8,
    doc_ids_count: usize,
    marker: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    out_updated_count: ?*usize,
) callconv(.c) i32 {
    return docs.talu_documents_set_marker_batch(
        db_path,
        doc_ids,
        doc_ids_count,
        marker,
        doc_type,
        out_updated_count,
    );
}

pub export fn talu_db_table_list(
    db_path: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    owner_id: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*docs.CDocumentList,
) callconv(.c) i32 {
    return docs.talu_documents_list(db_path, doc_type, group_id, owner_id, marker, limit, out_list);
}

pub export fn talu_db_table_free_list(list: ?*docs.CDocumentList) callconv(.c) void {
    docs.talu_documents_free_list(list);
}

pub export fn talu_db_table_search(
    db_path: ?[*:0]const u8,
    query: ?[*:0]const u8,
    doc_type: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*docs.CSearchResultList,
) callconv(.c) i32 {
    return docs.talu_documents_search(db_path, query, doc_type, limit, out_list);
}

pub export fn talu_db_table_free_search_results(list: ?*docs.CSearchResultList) callconv(.c) void {
    docs.talu_documents_free_search_results(list);
}

pub export fn talu_db_table_search_batch(
    db_path: ?[*:0]const u8,
    queries_json: ?[*]const u8,
    queries_len: usize,
    out_results_json: ?*?[*]u8,
    out_results_len: ?*usize,
) callconv(.c) i32 {
    return docs.talu_documents_search_batch(
        db_path,
        queries_json,
        queries_len,
        out_results_json,
        out_results_len,
    );
}

pub export fn talu_db_table_free_json(ptr: ?[*]u8, len: usize) callconv(.c) void {
    docs.talu_documents_free_json(ptr, len);
}

pub export fn talu_db_table_get_changes(
    db_path: ?[*:0]const u8,
    since_seq: u64,
    group_id: ?[*:0]const u8,
    limit: u32,
    out_list: ?*?*docs.CChangeList,
) callconv(.c) i32 {
    return docs.talu_documents_get_changes(db_path, since_seq, group_id, limit, out_list);
}

pub export fn talu_db_table_free_changes(list: ?*docs.CChangeList) callconv(.c) void {
    docs.talu_documents_free_changes(list);
}

pub export fn talu_db_table_set_ttl(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    ttl_seconds: u64,
) callconv(.c) i32 {
    return docs.talu_documents_set_ttl(db_path, doc_id, ttl_seconds);
}

pub export fn talu_db_table_count_expired(
    db_path: ?[*:0]const u8,
    out_count: ?*usize,
) callconv(.c) i32 {
    return docs.talu_documents_count_expired(db_path, out_count);
}

pub export fn talu_db_table_create_delta(
    db_path: ?[*:0]const u8,
    base_doc_id: ?[*:0]const u8,
    new_doc_id: ?[*:0]const u8,
    delta_json: ?[*:0]const u8,
    title: ?[*:0]const u8,
    tags_text: ?[*:0]const u8,
    marker: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_create_delta(
        db_path,
        base_doc_id,
        new_doc_id,
        delta_json,
        title,
        tags_text,
        marker,
    );
}

pub export fn talu_db_table_get_delta_chain(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_chain: ?*?*docs.CDeltaChain,
) callconv(.c) i32 {
    return docs.talu_documents_get_delta_chain(db_path, doc_id, out_chain);
}

pub export fn talu_db_table_free_delta_chain(chain: ?*docs.CDeltaChain) callconv(.c) void {
    docs.talu_documents_free_delta_chain(chain);
}

pub export fn talu_db_table_is_delta(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_is_delta: ?*bool,
) callconv(.c) i32 {
    return docs.talu_documents_is_delta(db_path, doc_id, out_is_delta);
}

pub export fn talu_db_table_get_base_id(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_base_id: ?[*]u8,
    out_base_id_len: usize,
) callconv(.c) i32 {
    return docs.talu_documents_get_base_id(db_path, doc_id, out_base_id, out_base_id_len);
}

pub export fn talu_db_table_get_compaction_stats(
    db_path: ?[*:0]const u8,
    out_stats: ?*docs.CCompactionStats,
) callconv(.c) i32 {
    return docs.talu_documents_get_compaction_stats(db_path, out_stats);
}

pub export fn talu_db_table_purge_expired(
    db_path: ?[*:0]const u8,
    out_count: ?*usize,
) callconv(.c) i32 {
    return docs.talu_documents_purge_expired(db_path, out_count);
}

pub export fn talu_db_table_get_garbage_candidates(
    db_path: ?[*:0]const u8,
    out_ids: ?*?*docs.CStringList,
) callconv(.c) i32 {
    return docs.talu_documents_get_garbage_candidates(db_path, out_ids);
}

pub export fn talu_db_table_add_tag(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_add_tag(db_path, doc_id, tag_id, group_id);
}

pub export fn talu_db_table_remove_tag(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return docs.talu_documents_remove_tag(db_path, doc_id, tag_id, group_id);
}

pub export fn talu_db_table_get_tags(
    db_path: ?[*:0]const u8,
    doc_id: ?[*:0]const u8,
    out_tag_ids: ?*?*docs.CStringList,
) callconv(.c) i32 {
    return docs.talu_documents_get_tags(db_path, doc_id, out_tag_ids);
}

pub export fn talu_db_table_get_by_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_doc_ids: ?*?*docs.CStringList,
) callconv(.c) i32 {
    return docs.talu_documents_get_by_tag(db_path, tag_id, out_doc_ids);
}

pub export fn talu_db_table_free_string_list(list: ?*docs.CStringList) callconv(.c) void {
    docs.talu_documents_free_string_list(list);
}

// =============================================================================
// Private helpers: Session/Tag internals
// =============================================================================

fn copyToStaticBuf(buf: []u8, src: []const u8) ?[*:0]const u8 {
    if (src.len == 0) return null;
    const len = @min(src.len, buf.len - 1);
    @memcpy(buf[0..len], src[0..len]);
    buf[len] = 0;
    return @ptrCast(buf.ptr);
}

fn copyToStaticBufOpt(buf: []u8, src: ?[]const u8) ?[*:0]const u8 {
    const s = src orelse return null;
    return copyToStaticBuf(buf, s);
}

fn populateHashSet(session_ids: []const []const u8, out: *std.AutoHashMap(u64, void)) void {
    for (session_ids) |sid| {
        out.put(db.table.sessions.computeSessionHash(sid), {}) catch continue;
    }
}

fn intersectHashSet(target: *std.AutoHashMap(u64, void), session_ids: []const []const u8) void {
    var new_set = std.AutoHashMap(u64, void).init(allocator);
    defer new_set.deinit();
    for (session_ids) |sid| {
        new_set.put(db.table.sessions.computeSessionHash(sid), {}) catch continue;
    }
    var removals = std.ArrayList(u64).empty;
    defer removals.deinit(allocator);
    var it = target.keyIterator();
    while (it.next()) |k| {
        if (!new_set.contains(k.*)) removals.append(allocator, k.*) catch continue;
    }
    for (removals.items) |k| {
        _ = target.remove(k);
    }
}

fn buildSessionScanParams(
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
    project_id_null: i32,
) db.table.sessions.TableAdapter.ScanParams {
    var params = db.table.sessions.TableAdapter.ScanParams.fromArgs(
        limit,
        before_updated_at_ms,
        optSlice(before_session_id),
        optSlice(group_id),
    );
    if (project_id_null != 0) params.target_project_null = true;
    if (optSlice(project_id)) |pid| {
        params.target_project_hash = db.table.sessions.computeGroupHash(pid);
        params.target_project_id = pid;
    }
    return params;
}

fn listSessionsImpl(
    db_path_slice: []const u8,
    params: db.table.sessions.TableAdapter.ScanParams,
    out_sessions: *?*CSessionList,
) i32 {
    const records = db.table.sessions.listSessions(allocator, db_path_slice, params) catch |err| {
        capi_error.setError(err, "failed to scan sessions", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.sessions.freeScannedSessionRecords(allocator, records);
    const list = buildSessionList(records) catch |err| {
        capi_error.setError(err, "failed to build session list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_sessions.* = list;
    return 0;
}

fn buildSessionList(records: []db.table.sessions.ScannedSessionRecord) !*CSessionList {
    const list = allocator.create(CSessionList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();
    const arena = arena_ptr.allocator();
    const sessions_buf = arena.alloc(CSessionRecord, records.len) catch return error.OutOfMemory;
    for (records, 0..) |record, i| {
        sessions_buf[i] = std.mem.zeroes(CSessionRecord);
        sessions_buf[i].session_id = if (record.session_id.len > 0)
            (arena.dupeZ(u8, record.session_id) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].model = if (record.model) |m| (arena.dupeZ(u8, m) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].title = if (record.title) |t| (arena.dupeZ(u8, t) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].system_prompt = if (record.system_prompt) |s| (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].config_json = if (record.config_json) |c| (arena.dupeZ(u8, c) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].marker = if (record.marker) |s| (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].parent_session_id = if (record.parent_session_id) |p| (arena.dupeZ(u8, p) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].group_id = if (record.group_id) |g| (arena.dupeZ(u8, g) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].head_item_id = record.head_item_id;
        sessions_buf[i].ttl_ts = record.ttl_ts;
        sessions_buf[i].metadata_json = if (record.metadata_json) |m| (arena.dupeZ(u8, m) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].search_snippet = if (record.search_snippet) |s| (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].source_doc_id = if (record.source_doc_id) |d| (arena.dupeZ(u8, d) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].project_id = if (record.project_id) |p| (arena.dupeZ(u8, p) catch return error.OutOfMemory).ptr else null;
        sessions_buf[i].created_at_ms = record.created_at_ms;
        sessions_buf[i].updated_at_ms = record.updated_at_ms;
    }
    list.* = .{
        .sessions = if (records.len > 0) sessions_buf.ptr else null,
        .count = records.len,
        .total = records.len,
        ._allocator = null,
        ._arena = @ptrCast(arena_ptr),
    };
    log.debug("capi", "buildSessionList", .{
        .list_ptr = @intFromPtr(list),
        .count = records.len,
        .arena_ptr = @intFromPtr(arena_ptr),
    }, @src());
    return list;
}

fn buildTagList(records: []db.table.tags.TagRecord) !*CTagList {
    const list = allocator.create(CTagList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();
    const arena = arena_ptr.allocator();
    const tags_buf = arena.alloc(CTagRecord, records.len) catch return error.OutOfMemory;
    for (records, 0..) |record, i| {
        tags_buf[i] = std.mem.zeroes(CTagRecord);
        tags_buf[i].tag_id = (arena.dupeZ(u8, record.tag_id) catch return error.OutOfMemory).ptr;
        tags_buf[i].name = (arena.dupeZ(u8, record.name) catch return error.OutOfMemory).ptr;
        tags_buf[i].color = if (record.color) |c| (arena.dupeZ(u8, c) catch return error.OutOfMemory).ptr else null;
        tags_buf[i].description = if (record.description) |d| (arena.dupeZ(u8, d) catch return error.OutOfMemory).ptr else null;
        tags_buf[i].group_id = if (record.group_id) |g| (arena.dupeZ(u8, g) catch return error.OutOfMemory).ptr else null;
        tags_buf[i].created_at_ms = record.created_at_ms;
        tags_buf[i].updated_at_ms = record.updated_at_ms;
    }
    list.* = .{
        .tags = if (records.len > 0) tags_buf.ptr else null,
        .count = records.len,
        ._arena = @ptrCast(arena_ptr),
    };
    return list;
}

fn buildRelationStringList(strings: [][]const u8) !*CRelationStringList {
    const list = allocator.create(CRelationStringList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);
    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();
    const arena = arena_ptr.allocator();
    const strings_buf = arena.alloc(?[*:0]const u8, strings.len) catch return error.OutOfMemory;
    for (strings, 0..) |s, i| {
        strings_buf[i] = (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr;
    }
    list.* = .{
        .strings = if (strings.len > 0) strings_buf.ptr else null,
        .count = strings.len,
        ._arena = @ptrCast(arena_ptr),
    };
    return list;
}

fn populateSessionRecord(out: *CSessionRecord, record: db.table.sessions.ScannedSessionRecord) void {
    const S = struct {
        threadlocal var session_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var model_buf: [256]u8 = .{0} ** 256;
        threadlocal var title_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var system_prompt_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var config_json_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var marker_buf: [64]u8 = .{0} ** 64;
        threadlocal var parent_session_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var group_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var metadata_json_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var search_snippet_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var source_doc_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var project_id_buf: [256]u8 = .{0} ** 256;
    };
    out.session_id = copyToStaticBuf(&S.session_id_buf, record.session_id);
    out.model = copyToStaticBufOpt(&S.model_buf, record.model);
    out.title = copyToStaticBufOpt(&S.title_buf, record.title);
    out.system_prompt = copyToStaticBufOpt(&S.system_prompt_buf, record.system_prompt);
    out.config_json = copyToStaticBufOpt(&S.config_json_buf, record.config_json);
    out.marker = copyToStaticBufOpt(&S.marker_buf, record.marker);
    out.parent_session_id = copyToStaticBufOpt(&S.parent_session_id_buf, record.parent_session_id);
    out.group_id = copyToStaticBufOpt(&S.group_id_buf, record.group_id);
    out.head_item_id = record.head_item_id;
    out.ttl_ts = record.ttl_ts;
    out.metadata_json = copyToStaticBufOpt(&S.metadata_json_buf, record.metadata_json);
    out.search_snippet = copyToStaticBufOpt(&S.search_snippet_buf, record.search_snippet);
    out.source_doc_id = copyToStaticBufOpt(&S.source_doc_id_buf, record.source_doc_id);
    out.project_id = copyToStaticBufOpt(&S.project_id_buf, record.project_id);
    out.created_at_ms = record.created_at_ms;
    out.updated_at_ms = record.updated_at_ms;
}

fn populateTagRecord(out: *CTagRecord, record: db.table.tags.TagRecord) void {
    const S = struct {
        threadlocal var tag_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var name_buf: [256]u8 = .{0} ** 256;
        threadlocal var color_buf: [32]u8 = .{0} ** 32;
        threadlocal var description_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var group_id_buf: [256]u8 = .{0} ** 256;
    };
    out.tag_id = copyToStaticBuf(&S.tag_id_buf, record.tag_id);
    out.name = copyToStaticBuf(&S.name_buf, record.name);
    out.color = copyToStaticBufOpt(&S.color_buf, record.color);
    out.description = copyToStaticBufOpt(&S.description_buf, record.description);
    out.group_id = copyToStaticBufOpt(&S.group_id_buf, record.group_id);
    out.created_at_ms = record.created_at_ms;
    out.updated_at_ms = record.updated_at_ms;
}

// Tag impl helpers

fn listTagsImpl(db_path_slice: []const u8, group_id: ?[]const u8, out_tags: *?*CTagList) i32 {
    const records = db.table.tags.listTags(allocator, db_path_slice, group_id) catch |err| {
        capi_error.setError(err, "failed to scan tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.tags.freeTagRecords(allocator, records);
    const list = buildTagList(records) catch |err| {
        capi_error.setError(err, "failed to build tag list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_tags.* = list;
    return 0;
}

fn getTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, out_tag: *CTagRecord) i32 {
    var record = db.table.tags.getTag(allocator, db_path_slice, tag_id_slice) catch |err| {
        capi_error.setError(err, "tag not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    } orelse {
        capi_error.setErrorWithCode(.tag_not_found, "tag not found", .{});
        return @intFromEnum(error_codes.ErrorCode.tag_not_found);
    };
    defer record.deinit(allocator);
    populateTagRecord(out_tag, record);
    return 0;
}

fn getTagByNameImpl(db_path_slice: []const u8, name_slice: []const u8, group_id: ?[]const u8, out_tag: *CTagRecord) i32 {
    var record = db.table.tags.getTagByName(allocator, db_path_slice, name_slice, group_id) catch |err| {
        capi_error.setError(err, "tag not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    } orelse {
        capi_error.setErrorWithCode(.tag_not_found, "tag not found", .{});
        return @intFromEnum(error_codes.ErrorCode.tag_not_found);
    };
    defer record.deinit(allocator);
    populateTagRecord(out_tag, record);
    return 0;
}

fn createTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, name_slice: []const u8, color: ?[]const u8, description: ?[]const u8, group_id: ?[]const u8) i32 {
    const now_ms = std.time.milliTimestamp();
    const record = db.table.tags.TagRecord{
        .tag_id = tag_id_slice,
        .name = name_slice,
        .color = color,
        .description = description,
        .group_id = group_id,
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };
    db.table.tags.createTag(allocator, db_path_slice, record) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to create tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

fn updateTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, name: ?[]const u8, color: ?[]const u8, description: ?[]const u8) i32 {
    db.table.tags.updateTag(allocator, db_path_slice, tag_id_slice, name, color, description) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to update tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

fn deleteTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.deleteTagAndAssociations(allocator, db_path_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to delete tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

fn addConversationTagImpl(db_path_slice: []const u8, session_id_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.addConversationTag(allocator, db_path_slice, session_id_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to add conversation tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

fn removeConversationTagImpl(db_path_slice: []const u8, session_id_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.removeConversationTag(allocator, db_path_slice, session_id_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to remove conversation tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

fn getConversationTagsImpl(db_path_slice: []const u8, session_id_slice: []const u8, out_tag_ids: *?*CRelationStringList) i32 {
    const tag_ids = db.table.tags.getConversationTagIds(allocator, db_path_slice, session_id_slice) catch |err| {
        capi_error.setError(err, "failed to get conversation tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer {
        for (tag_ids) |id| allocator.free(id);
        allocator.free(tag_ids);
    }
    const list = buildRelationStringList(tag_ids) catch |err| {
        capi_error.setError(err, "failed to build string list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_tag_ids.* = list;
    return 0;
}

fn getTagConversationsImpl(db_path_slice: []const u8, tag_id_slice: []const u8, out_session_ids: *?*CRelationStringList) i32 {
    const session_ids = db.table.tags.getTagConversationIds(allocator, db_path_slice, tag_id_slice) catch |err| {
        capi_error.setError(err, "failed to get tag conversations", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer {
        for (session_ids) |id| allocator.free(id);
        allocator.free(session_ids);
    }
    const list = buildRelationStringList(session_ids) catch |err| {
        capi_error.setError(err, "failed to build string list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_session_ids.* = list;
    return 0;
}

fn buildEmptySessionTagBatch(out: *?*CSessionTagBatch) i32 {
    const batch = allocator.create(CSessionTagBatch) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate batch", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    batch.* = std.mem.zeroes(CSessionTagBatch);
    out.* = batch;
    return 0;
}

fn getSessionsTagsBatchImpl(
    db_path_slice: []const u8,
    ids_ptr: [*]const ?[*:0]const u8,
    session_count: u32,
    out: *?*CSessionTagBatch,
) i32 {
    // Convert C string array to Zig slices.
    var zig_ids = allocator.alloc([]const u8, session_count) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate id array", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    defer allocator.free(zig_ids);

    for (0..session_count) |i| {
        const c_str = ids_ptr[i] orelse {
            capi_error.setErrorWithCode(.invalid_argument, "session_ids contains null", .{});
            return @intFromEnum(error_codes.ErrorCode.invalid_argument);
        };
        zig_ids[i] = std.mem.span(c_str);
    }

    var result = db.table.tags.getConversationTagIdsBatch(allocator, db_path_slice, zig_ids) catch |err| {
        capi_error.setError(err, "failed to batch get session tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer result.deinit(allocator);

    const batch = allocator.create(CSessionTagBatch) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate batch", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(batch);

    const count = result.session_ids.len;
    if (count == 0) {
        batch.* = std.mem.zeroes(CSessionTagBatch);
        out.* = batch;
        return 0;
    }

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate arena", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();
    const arena = arena_ptr.allocator();

    const c_sids = arena.alloc(?[*:0]const u8, count) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate session_ids", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    const c_tids = arena.alloc(?[*:0]const u8, count) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate tag_ids", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };

    for (0..count) |i| {
        c_sids[i] = (arena.dupeZ(u8, result.session_ids[i]) catch {
            capi_error.setErrorWithCode(.out_of_memory, "failed to copy session_id", .{});
            return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        }).ptr;
        c_tids[i] = (arena.dupeZ(u8, result.tag_ids[i]) catch {
            capi_error.setErrorWithCode(.out_of_memory, "failed to copy tag_id", .{});
            return @intFromEnum(error_codes.ErrorCode.out_of_memory);
        }).ptr;
    }

    batch.* = .{
        .session_ids = c_sids.ptr,
        .tag_ids = c_tids.ptr,
        .count = count,
        ._arena = @ptrCast(arena_ptr),
    };
    out.* = batch;
    return 0;
}

// =============================================================================
// Sessions: List / Get / Update / Fork / Delete
// =============================================================================

pub export fn talu_db_table_session_list(
    db_path: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
    project_id_null: i32,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, group_id, project_id, project_id_null);
    params.search_query = optSlice(search_query);
    if (params.search_query != null) params.max_scan = 5000;
    var allowed_hashes = std.AutoHashMap(u64, void).init(allocator);
    defer allowed_hashes.deinit();
    const group_slice = optSlice(group_id);
    if (optSlice(tags_filter)) |f| {
        const sids = db.table.tags.resolveTagFilterSessionIds(allocator, db_path_slice, f, true, group_slice) catch |err| {
            capi_error.setError(err, "failed to resolve tag filter", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, sids);
        populateHashSet(sids, &allowed_hashes);
        params.allowed_session_hashes = &allowed_hashes;
    } else if (optSlice(tags_filter_any)) |f| {
        const sids = db.table.tags.resolveTagFilterSessionIds(allocator, db_path_slice, f, false, group_slice) catch |err| {
            capi_error.setError(err, "failed to resolve tag filter", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, sids);
        populateHashSet(sids, &allowed_hashes);
        params.allowed_session_hashes = &allowed_hashes;
    }
    return listSessionsImpl(db_path_slice, params, out);
}

pub export fn talu_db_table_session_list_ex(
    db_path: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    marker_filter: ?[*:0]const u8,
    marker_filter_any: ?[*:0]const u8,
    model_filter: ?[*:0]const u8,
    created_after_ms: i64,
    created_before_ms: i64,
    updated_after_ms: i64,
    updated_before_ms: i64,
    has_tags: i32,
    source_doc_id: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
    project_id_null: i32,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, group_id, project_id, project_id_null);
    params.search_query = optSlice(search_query);
    params.marker_filter = optSlice(marker_filter);
    params.marker_filter_any = optSlice(marker_filter_any);
    params.model_filter = optSlice(model_filter);
    params.created_after_ms = if (created_after_ms != 0) created_after_ms else null;
    params.created_before_ms = if (created_before_ms != 0) created_before_ms else null;
    params.updated_after_ms = if (updated_after_ms != 0) updated_after_ms else null;
    params.updated_before_ms = if (updated_before_ms != 0) updated_before_ms else null;
    params.source_doc_id = optSlice(source_doc_id);
    if (params.search_query != null) params.max_scan = 5000;
    var allowed_hashes = std.AutoHashMap(u64, void).init(allocator);
    defer allowed_hashes.deinit();
    var excluded_hashes = std.AutoHashMap(u64, void).init(allocator);
    defer excluded_hashes.deinit();
    var use_allowed = false;
    var use_excluded = false;
    const group_slice = optSlice(group_id);
    if (optSlice(tags_filter)) |f| {
        const sids = db.table.tags.resolveTagFilterSessionIds(allocator, db_path_slice, f, true, group_slice) catch |err| {
            capi_error.setError(err, "failed to resolve tag filter", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, sids);
        populateHashSet(sids, &allowed_hashes);
        use_allowed = true;
    }
    if (optSlice(tags_filter_any)) |f| {
        const sids = db.table.tags.resolveTagFilterSessionIds(allocator, db_path_slice, f, false, group_slice) catch |err| {
            capi_error.setError(err, "failed to resolve tag filter", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, sids);
        if (use_allowed) {
            intersectHashSet(&allowed_hashes, sids);
        } else {
            populateHashSet(sids, &allowed_hashes);
            use_allowed = true;
        }
    }
    if (has_tags == 1 or has_tags == 0) {
        const all_tagged = db.table.tags.collectAllTaggedSessionIds(allocator, db_path_slice) catch |err| {
            capi_error.setError(err, "failed to collect tagged sessions", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, all_tagged);
        if (has_tags == 1) {
            if (use_allowed) {
                intersectHashSet(&allowed_hashes, all_tagged);
            } else {
                populateHashSet(all_tagged, &allowed_hashes);
                use_allowed = true;
            }
        } else {
            populateHashSet(all_tagged, &excluded_hashes);
            use_excluded = true;
        }
    }
    if (use_allowed) params.allowed_session_hashes = &allowed_hashes;
    if (use_excluded) params.excluded_session_hashes = &excluded_hashes;
    return listSessionsImpl(db_path_slice, params, out);
}

pub export fn talu_db_table_session_list_by_source(
    db_path: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const source_doc_slice = validateRequiredArg(source_doc_id, "source_doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, null, null, 0);
    params.source_doc_id = source_doc_slice;
    return listSessionsImpl(db_path_slice, params, out);
}

pub export fn talu_db_table_session_list_batch(
    db_path: ?[*:0]const u8,
    offset: u32,
    limit: u32,
    group_id: ?[*:0]const u8,
    marker_filter: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
    project_id_null: i32,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = db.table.sessions.TableAdapter.ScanParams{};
    if (project_id_null != 0) params.target_project_null = true;
    if (optSlice(group_id)) |gid| {
        params.target_group_hash = db.table.sessions.computeGroupHash(gid);
        params.target_group_id = gid;
    }
    if (optSlice(project_id)) |pid| {
        params.target_project_hash = db.table.sessions.computeGroupHash(pid);
        params.target_project_id = pid;
    }
    params.marker_filter = optSlice(marker_filter);
    params.search_query = optSlice(search_query);
    if (params.search_query != null) params.max_scan = 5000;
    var allowed_hashes = std.AutoHashMap(u64, void).init(allocator);
    defer allowed_hashes.deinit();
    if (optSlice(tags_filter_any)) |f| {
        const group_slice = optSlice(group_id);
        const sids = db.table.tags.resolveTagFilterSessionIds(allocator, db_path_slice, f, false, group_slice) catch |err| {
            capi_error.setError(err, "failed to resolve tag filter", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        };
        defer db.table.tags.freeStringSlice(allocator, sids);
        populateHashSet(sids, &allowed_hashes);
        params.allowed_session_hashes = &allowed_hashes;
    }
    const records = db.table.sessions.listSessions(allocator, db_path_slice, params) catch |err| {
        capi_error.setError(err, "failed to scan sessions", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.sessions.freeScannedSessionRecords(allocator, records);
    const total = records.len;
    const start = @min(@as(usize, offset), total);
    const end = if (limit > 0) @min(start + @as(usize, limit), total) else total;
    const page = records[start..end];
    const list = buildSessionList(page) catch |err| {
        capi_error.setError(err, "failed to build session list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    list.total = total;
    out.* = list;
    return 0;
}

pub export fn talu_db_table_session_get(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_session: *CSessionRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_session.* = std.mem.zeroes(CSessionRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var adapter = db.table.sessions.TableAdapter.initReadOnly(allocator, db_path_slice) catch |err| {
        capi_error.setError(err, "failed to open database for reading", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.deinitReadOnly();
    const record = adapter.lookupSession(allocator, session_id_slice) catch |err| {
        capi_error.setError(err, "session not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.sessions.freeScannedSessionRecord(allocator, @constCast(&record));
    populateSessionRecord(out_session, record);
    return 0;
}

pub export fn talu_db_table_session_update(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();
    adapter.updateSession(allocator, session_id_slice, optSlice(title), optSlice(marker), optSlice(metadata_json)) catch |err| {
        capi_error.setError(err, "failed to update session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_db_table_session_update_ex(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
    project_id: ?[*:0]const u8,
    clear_project_id: i32,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();
    adapter.updateSessionEx(allocator, session_id_slice, optSlice(title), optSlice(marker), optSlice(metadata_json), optSlice(source_doc_id), optSlice(project_id), clear_project_id != 0) catch |err| {
        capi_error.setError(err, "failed to update session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_db_table_session_fork(
    db_path: ?[*:0]const u8,
    source_session_id: ?[*:0]const u8,
    target_item_id: u64,
    new_session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const source_slice = validateRequiredArg(source_session_id, "source_session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const new_slice = validateRequiredArg(new_session_id, "new_session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    db.table.sessions.forkSession(allocator, db_path_slice, source_slice, target_item_id, new_slice) catch |err| {
        capi_error.setError(err, "failed to fork session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_db_table_session_delete(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();
    adapter.deleteSession(session_id_slice) catch |err| {
        capi_error.setError(err, "failed to delete session", .{});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    return 0;
}

pub export fn talu_db_table_session_load_conversation(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) ?*anyopaque {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return null;
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return null;
    const conv = db.table.sessions.loadConversation(allocator, db_path_slice, session_id_slice) catch |err| {
        capi_error.setError(err, "failed to load conversation", .{});
        return null;
    };
    return @ptrCast(conv);
}

pub export fn talu_db_table_session_free_list(sessions: ?*CSessionList) callconv(.c) void {
    capi_error.clearError();
    const list = sessions orelse return;
    log.debug("capi", "talu_db_table_session_free_list", .{
        .list_ptr = @intFromPtr(list),
        .count = list.count,
        .arena_ptr = if (list._arena) |a| @intFromPtr(a) else 0,
    }, @src());
    if (list._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }
    allocator.destroy(list);
}

// =============================================================================
// Tags: CRUD
// =============================================================================

pub export fn talu_db_table_tag_list(
    db_path: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tags: ?*?*CTagList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tags orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tags is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return listTagsImpl(db_path_slice, optSlice(group_id), out);
}

pub export fn talu_db_table_tag_get(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_tag: *CTagRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_tag.* = std.mem.zeroes(CTagRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagImpl(db_path_slice, tag_id_slice, out_tag);
}

pub export fn talu_db_table_tag_get_by_name(
    db_path: ?[*:0]const u8,
    name: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tag: *CTagRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_tag.* = std.mem.zeroes(CTagRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const name_slice = validateRequiredArg(name, "name") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagByNameImpl(db_path_slice, name_slice, optSlice(group_id), out_tag);
}

pub export fn talu_db_table_tag_create(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const name_slice = validateRequiredArg(name, "name") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return createTagImpl(db_path_slice, tag_id_slice, name_slice, optSlice(color), optSlice(description), optSlice(group_id));
}

pub export fn talu_db_table_tag_update(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return updateTagImpl(db_path_slice, tag_id_slice, optSlice(name), optSlice(color), optSlice(description));
}

pub export fn talu_db_table_tag_delete(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return deleteTagImpl(db_path_slice, tag_id_slice);
}

pub export fn talu_db_table_tag_free_list(tags: ?*CTagList) callconv(.c) void {
    capi_error.clearError();
    const list = tags orelse return;
    if (list._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }
    allocator.destroy(list);
}

// =============================================================================
// Session-Tag junction
// =============================================================================

pub export fn talu_db_table_session_add_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return addConversationTagImpl(db_path_slice, session_id_slice, tag_id_slice);
}

pub export fn talu_db_table_session_remove_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return removeConversationTagImpl(db_path_slice, session_id_slice, tag_id_slice);
}

pub export fn talu_db_table_session_get_tags(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_tag_ids: ?*?*CRelationStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tag_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tag_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getConversationTagsImpl(db_path_slice, session_id_slice, out);
}

pub export fn talu_db_table_tag_get_conversations(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_session_ids: ?*?*CRelationStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_session_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_session_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagConversationsImpl(db_path_slice, tag_id_slice, out);
}

pub export fn talu_db_table_free_relation_string_list(list: ?*CRelationStringList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;
    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }
    allocator.destroy(l);
}

/// Get tag IDs for multiple sessions in a single scan.
///
/// Returns flat parallel arrays: out_result.session_ids[i] paired with
/// out_result.tag_ids[i]. Free with talu_db_table_free_session_tag_batch.
pub export fn talu_db_table_sessions_get_tags_batch(
    db_path: ?[*:0]const u8,
    session_ids: ?[*]const ?[*:0]const u8,
    session_count: u32,
    out_result: ?*?*CSessionTagBatch,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_result orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_result is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    if (session_count == 0) {
        return buildEmptySessionTagBatch(out);
    }
    const ids_ptr = session_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    return getSessionsTagsBatchImpl(db_path_slice, ids_ptr, session_count, out);
}

pub export fn talu_db_table_free_session_tag_batch(batch: ?*CSessionTagBatch) callconv(.c) void {
    capi_error.clearError();
    const b = batch orelse return;
    if (b._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }
    allocator.destroy(b);
}

// =============================================================================
// Tag inheritance
// =============================================================================

pub export fn talu_chat_inherit_tags(
    chat_handle: ?*ChatHandle,
    db_path: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse return setArgError("chat_handle is null")));
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const prompt_id = chat.getPromptId() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no prompt_id set (nothing to inherit)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const session_id = chat.session_id orelse chat.conv.session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no session_id set (cannot tag non-persisted session)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const doc_tag_ids = db.table.document_tags.getDocumentTagIds(allocator, db_path_slice, prompt_id) catch |err| {
        capi_error.setError(err, "failed to get document tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.document_tags.freeStringSlice(allocator, @constCast(doc_tag_ids));
    for (doc_tag_ids) |tag_id| {
        const rc = addConversationTagImpl(db_path_slice, session_id, tag_id);
        if (rc != 0) log.debug("capi", "inherit_tags_failed", .{ .rc = rc }, @src());
    }
    return 0;
}
