//! Plane-specific DB C-API: Table.
//!
//! Canonical exports use `talu_db_table_*` and include document APIs plus
//! session/tag table operations.

const docs = @import("../documents_impl.zig");
const legacy = @import("../db_impl.zig");

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

// -----------------------------------------------------------------------------
// Documents
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Sessions + tags tables
// -----------------------------------------------------------------------------

pub export fn talu_db_table_session_list(
    db_path: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    out_sessions: ?*?*legacy.CSessionList,
) callconv(.c) i32 {
    return legacy.talu_storage_list_sessions(
        db_path,
        limit,
        before_updated_at_ms,
        before_session_id,
        group_id,
        search_query,
        tags_filter,
        tags_filter_any,
        out_sessions,
    );
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
    out_sessions: ?*?*legacy.CSessionList,
) callconv(.c) i32 {
    return legacy.talu_storage_list_sessions_ex(
        db_path,
        limit,
        before_updated_at_ms,
        before_session_id,
        group_id,
        search_query,
        tags_filter,
        tags_filter_any,
        marker_filter,
        marker_filter_any,
        model_filter,
        created_after_ms,
        created_before_ms,
        updated_after_ms,
        updated_before_ms,
        has_tags,
        source_doc_id,
        out_sessions,
    );
}

pub export fn talu_db_table_session_list_by_source(
    db_path: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    out_sessions: ?*?*legacy.CSessionList,
) callconv(.c) i32 {
    return legacy.talu_storage_list_sessions_by_source(
        db_path,
        source_doc_id,
        limit,
        before_updated_at_ms,
        before_session_id,
        out_sessions,
    );
}

pub export fn talu_db_table_session_list_batch(
    db_path: ?[*:0]const u8,
    offset: u32,
    limit: u32,
    group_id: ?[*:0]const u8,
    marker_filter: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    out_sessions: ?*?*legacy.CSessionList,
) callconv(.c) i32 {
    return legacy.talu_storage_list_sessions_batch(
        db_path,
        offset,
        limit,
        group_id,
        marker_filter,
        search_query,
        tags_filter_any,
        out_sessions,
    );
}

pub export fn talu_db_table_session_get(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_session: *legacy.CSessionRecord,
) callconv(.c) i32 {
    return legacy.talu_storage_get_session_info(db_path, session_id, out_session);
}

pub export fn talu_db_table_session_update(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_update_session(db_path, session_id, title, marker, metadata_json);
}

pub export fn talu_db_table_session_update_ex(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_update_session_ex(
        db_path,
        session_id,
        title,
        marker,
        metadata_json,
        source_doc_id,
    );
}

pub export fn talu_db_table_session_fork(
    db_path: ?[*:0]const u8,
    source_session_id: ?[*:0]const u8,
    target_item_id: u64,
    new_session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_fork_session(db_path, source_session_id, target_item_id, new_session_id);
}

pub export fn talu_db_table_session_delete(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_delete_session(db_path, session_id);
}

pub export fn talu_db_table_session_load_conversation(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) ?*anyopaque {
    const ptr = legacy.talu_storage_load_conversation(db_path, session_id) orelse return null;
    return @ptrCast(ptr);
}

pub export fn talu_db_table_session_free_list(sessions: ?*legacy.CSessionList) callconv(.c) void {
    legacy.talu_storage_free_sessions(sessions);
}

pub export fn talu_db_table_tag_list(
    db_path: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tags: ?*?*legacy.CTagList,
) callconv(.c) i32 {
    return legacy.talu_storage_list_tags(db_path, group_id, out_tags);
}

pub export fn talu_db_table_tag_get(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_tag: *legacy.CTagRecord,
) callconv(.c) i32 {
    return legacy.talu_storage_get_tag(db_path, tag_id, out_tag);
}

pub export fn talu_db_table_tag_get_by_name(
    db_path: ?[*:0]const u8,
    name: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tag: *legacy.CTagRecord,
) callconv(.c) i32 {
    return legacy.talu_storage_get_tag_by_name(db_path, name, group_id, out_tag);
}

pub export fn talu_db_table_tag_create(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_create_tag(db_path, tag_id, name, color, description, group_id);
}

pub export fn talu_db_table_tag_update(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_update_tag(db_path, tag_id, name, color, description);
}

pub export fn talu_db_table_tag_delete(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_delete_tag(db_path, tag_id);
}

pub export fn talu_db_table_tag_free_list(tags: ?*legacy.CTagList) callconv(.c) void {
    legacy.talu_storage_free_tags(tags);
}

pub export fn talu_db_table_session_add_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_add_conversation_tag(db_path, session_id, tag_id);
}

pub export fn talu_db_table_session_remove_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_storage_remove_conversation_tag(db_path, session_id, tag_id);
}

pub export fn talu_db_table_session_get_tags(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_tag_ids: ?*?*legacy.CStringList,
) callconv(.c) i32 {
    return legacy.talu_storage_get_conversation_tags(db_path, session_id, out_tag_ids);
}

pub export fn talu_db_table_tag_get_conversations(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_session_ids: ?*?*legacy.CStringList,
) callconv(.c) i32 {
    return legacy.talu_storage_get_tag_conversations(db_path, tag_id, out_session_ids);
}

pub export fn talu_db_table_free_relation_string_list(list: ?*legacy.CStringList) callconv(.c) void {
    legacy.talu_storage_free_string_list(list);
}
