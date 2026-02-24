//! TaluDB storage management.
//!
//! Provides safe Rust wrappers for TaluDB storage operations including
//! session listing, querying, and deletion.
//!
//! # Concurrency Model
//!
//! TaluDB uses granular locking - locks are acquired only during actual I/O
//! operations (microseconds) rather than for the duration of a session.
//! Multiple processes can safely share the same storage directory.
//!
//! # Example
//!
//! ```no_run
//! use talu::storage::{StorageHandle, StorageError};
//!
//! let handle = StorageHandle::open("./taludb")?;
//! let sessions = handle.list_sessions(Some(50))?;
//! for session in sessions {
//!     println!("{}: {}", session.session_id, session.title.unwrap_or_default());
//! }
//!
//! // Delete a session (blocks briefly if another process is writing)
//! handle.delete_session("old-session")?;
//! # Ok::<(), StorageError>(())
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};

use talu_sys::{CRelationStringList, CSessionList, CSessionRecord, CTagList, CTagRecord};

/// Error codes from the C API (must match error_codes.zig)
const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_STORAGE_ERROR: i32 = 700;
#[allow(dead_code)] // Reserved for future use (matches C API error_codes.zig)
const ERROR_CODE_RESOURCE_BUSY: i32 = 701;
const ERROR_CODE_ITEM_NOT_FOUND: i32 = 702;
const ERROR_CODE_SESSION_NOT_FOUND: i32 = 703;
const ERROR_CODE_TAG_NOT_FOUND: i32 = 704;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;
const ERROR_CODE_IO_FILE_NOT_FOUND: i32 = 500;

/// Error types for storage operations.
#[derive(Debug)]
pub enum StorageError {
    /// Session not found or has been deleted.
    SessionNotFound(String),

    /// Item not found within a session.
    ItemNotFound(String),

    /// Tag not found.
    TagNotFound(String),

    /// Storage path does not exist or is not a valid TaluDB.
    StorageNotFound(PathBuf),

    /// Invalid argument provided.
    InvalidArgument(String),

    /// Storage is corrupted or invalid.
    StorageCorrupted(String),

    /// Generic I/O error.
    IoError(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::SessionNotFound(s) => write!(f, "Session not found: {s}"),
            StorageError::ItemNotFound(s) => write!(f, "Item not found: {s}"),
            StorageError::TagNotFound(s) => write!(f, "Tag not found: {s}"),
            StorageError::StorageNotFound(p) => write!(f, "Storage not found: {}", p.display()),
            StorageError::InvalidArgument(s) => write!(f, "Invalid argument: {s}"),
            StorageError::StorageCorrupted(s) => write!(f, "Storage error: {s}"),
            StorageError::IoError(s) => write!(f, "I/O error: {s}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl StorageError {
    /// Translate C API error code to StorageError.
    fn from_code(code: i32, context: &str) -> Self {
        // Get detailed error message from C API
        let detail = crate::error::last_error_message().unwrap_or_default();

        match code {
            ERROR_CODE_STORAGE_ERROR => {
                if detail.contains("not found") {
                    StorageError::SessionNotFound(context.to_string())
                } else {
                    StorageError::StorageCorrupted(detail)
                }
            }
            ERROR_CODE_SESSION_NOT_FOUND => StorageError::SessionNotFound(context.to_string()),
            ERROR_CODE_ITEM_NOT_FOUND => StorageError::ItemNotFound(context.to_string()),
            ERROR_CODE_TAG_NOT_FOUND => StorageError::TagNotFound(context.to_string()),
            ERROR_CODE_INVALID_ARGUMENT => StorageError::InvalidArgument(detail),
            ERROR_CODE_IO_FILE_NOT_FOUND => StorageError::StorageNotFound(PathBuf::from(context)),
            _ => StorageError::IoError(detail),
        }
    }
}

/// Handle for TaluDB storage directory.
///
/// # Concurrency Model
///
/// TaluDB uses granular locking - write operations acquire locks only for
/// the brief duration of actual I/O (microseconds to milliseconds). This
/// allows multiple processes to safely share the same storage directory
/// with minimal contention.
///
/// - Read operations (`list_sessions`, `get_session`) are lock-free
/// - Write operations (`delete_session`) block briefly if another write is in progress
///
/// No `ResourceBusy` errors - operations simply wait their turn.
#[derive(Debug)]
pub struct StorageHandle {
    path: PathBuf,
    path_cstr: CString,
}

impl StorageHandle {
    /// Open a TaluDB storage at path.
    ///
    /// This does not acquire any locks. Locks are acquired granularly
    /// during individual operations.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::StorageNotFound` if the path doesn't exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_string_lossy();

        let path_cstr = CString::new(path_str.as_ref())
            .map_err(|_| StorageError::InvalidArgument("Path contains null bytes".to_string()))?;

        // Verify path exists
        if !path.exists() {
            return Err(StorageError::StorageNotFound(path));
        }

        Ok(Self { path, path_cstr })
    }

    /// List sessions with optional limit.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum sessions to return (None = all)
    ///
    /// # Memory Safety
    ///
    /// The returned Vec is fully owned by Rust. The underlying C
    /// memory (CSessionList) is freed before this function returns.
    ///
    /// # Performance Warning
    ///
    /// If limit is None or > 1000, logs a warning about memory usage.
    pub fn list_sessions(&self, limit: Option<usize>) -> Result<Vec<SessionRecord>, StorageError> {
        // Warn about large requests
        if limit.map_or(true, |l| l > 1000) {
            eprintln!("Warning: Loading many sessions. This may use significant memory.");
            eprintln!("Consider using a smaller --limit for pagination.");
        }

        let mut c_list: *mut CSessionList = std::ptr::null_mut();

        // Call C API with no cursor/group/search/tags filter (legacy compatibility)
        // SAFETY: path_cstr is valid, c_list is a valid output pointer
        let result = unsafe {
            talu_sys::talu_db_table_session_list(
                self.path_cstr.as_ptr(),
                0,                // no limit (applied in Rust below)
                0,                // no cursor
                std::ptr::null(), // no cursor session_id
                std::ptr::null(), // no group_id
                std::ptr::null(), // no search_query
                std::ptr::null(), // no tags_filter
                std::ptr::null(), // no tags_filter_any
                std::ptr::null(), // no project_id
                0,                // no project_id_null
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(Vec::new());
        }

        // Convert to Rust (copy strings into Rust-owned memory)
        // SAFETY: c_list is non-null and was returned by talu_db_table_session_list
        let rust_vec = unsafe { Self::convert_session_list(c_list, limit) };

        // Free C memory BEFORE returning - arena handles all cleanup
        // SAFETY: c_list was returned by talu_db_table_session_list
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        Ok(rust_vec)
    }

    /// Get details of a specific session (O(1) average case).
    ///
    /// # Errors
    ///
    /// Returns `StorageError::SessionNotFound` if the session doesn't exist.
    pub fn get_session(&self, session_id: &str) -> Result<SessionRecord, StorageError> {
        let session_id_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let mut c_record = CSessionRecord::default();

        // Call C API
        // SAFETY: path_cstr and session_id_cstr are valid, c_record is output
        let result = unsafe {
            talu_sys::talu_db_table_session_get(
                self.path_cstr.as_ptr(),
                session_id_cstr.as_ptr(),
                &mut c_record as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        // Convert to Rust
        // SAFETY: c_record was filled by talu_db_table_session_get
        Ok(unsafe { Self::convert_session_record(&c_record) })
    }

    /// Get session with all fields (system_prompt, config, source_doc_id, etc.).
    pub fn get_session_full(&self, session_id: &str) -> Result<SessionRecordFull, StorageError> {
        let session_id_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let mut c_record = CSessionRecord::default();

        // Call C API
        // SAFETY: path_cstr and session_id_cstr are valid, c_record is output
        let result = unsafe {
            talu_sys::talu_db_table_session_get(
                self.path_cstr.as_ptr(),
                session_id_cstr.as_ptr(),
                &mut c_record as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        // Convert to Rust (full record)
        // SAFETY: c_record was filled by talu_db_table_session_get
        Ok(unsafe { Self::convert_session_record_full(&c_record) })
    }

    /// Get total session count (for pagination UI).
    ///
    /// Note: This currently loads all sessions to count them.
    /// A more efficient implementation would add a dedicated C API function.
    pub fn session_count(&self) -> Result<usize, StorageError> {
        let mut c_list: *mut CSessionList = std::ptr::null_mut();

        // SAFETY: path_cstr is valid, c_list is a valid output pointer
        let result = unsafe {
            talu_sys::talu_db_table_session_list(
                self.path_cstr.as_ptr(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(), // no search_query
                std::ptr::null(), // no tags_filter
                std::ptr::null(), // no tags_filter_any
                std::ptr::null(), // no project_id
                0,                // no project_id_null
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(0);
        }

        // SAFETY: c_list is non-null and was returned by talu_db_table_session_list
        let count = unsafe { (*c_list).count };

        // Free C memory
        // SAFETY: c_list was returned by talu_db_table_session_list
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        Ok(count)
    }

    /// List sessions with offset-based pagination and total count.
    ///
    /// Pushes filtering, offset, and limit down to the Zig scan layer so
    /// only the requested page crosses the FFI boundary.
    pub fn list_sessions_batch(
        &self,
        offset: usize,
        limit: usize,
        group_id: Option<&str>,
        marker: Option<&str>,
        search: Option<&str>,
        tags_any: Option<&str>,
        project_id: Option<&str>,
        project_id_null: bool,
    ) -> Result<SessionBatchResult, StorageError> {
        let limit = limit.clamp(1, 100);

        let group_cstr = group_id
            .map(|g| CString::new(g))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("group_id contains null bytes".into()))?;
        let marker_cstr = marker
            .map(|m| CString::new(m))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("marker contains null bytes".into()))?;
        let search_cstr = search
            .map(|s| CString::new(s))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("search contains null bytes".into()))?;
        let tags_cstr = tags_any
            .map(|t| CString::new(t))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("tags_any contains null bytes".into()))?;
        let project_cstr = project_id
            .map(|p| CString::new(p))
            .transpose()
            .map_err(|_| {
                StorageError::InvalidArgument("project_id contains null bytes".into())
            })?;

        let group_ptr = group_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let marker_ptr = marker_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let search_ptr = search_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let tags_ptr = tags_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let project_ptr = project_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());

        let mut c_list: *mut CSessionList = std::ptr::null_mut();

        // SAFETY: path_cstr is valid, all pointers are valid or null
        let result = unsafe {
            talu_sys::talu_db_table_session_list_batch(
                self.path_cstr.as_ptr(),
                offset as u32,
                limit as u32,
                group_ptr,
                marker_ptr,
                search_ptr,
                tags_ptr,
                project_ptr,
                project_id_null as std::os::raw::c_int,
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(SessionBatchResult {
                sessions: Vec::new(),
                total: 0,
                has_more: false,
            });
        }

        // SAFETY: c_list is non-null and was returned by talu_db_table_session_list_batch
        let (sessions, total) = unsafe {
            let list = &*c_list;
            let records = Self::convert_session_list_full(c_list);
            (records, list.total)
        };

        // Free C memory
        // SAFETY: c_list was returned by talu_db_table_session_list_batch
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        let has_more = (offset + limit) < total;

        Ok(SessionBatchResult {
            sessions,
            total,
            has_more,
        })
    }

    /// Delete a session using Dual-Delete Protocol.
    ///
    /// Writes both:
    /// - Session tombstone (Schema 4) - hides from list
    /// - Clear marker (Schema 2) - hides items from restore
    ///
    /// # Concurrency
    ///
    /// Uses granular locking - blocks briefly (microseconds) if another
    /// process is currently writing. Multiple concurrent deletes are safe.
    pub fn delete_session(&self, session_id: &str) -> Result<(), StorageError> {
        let session_id_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        // Call C API
        // SAFETY: path_cstr and session_id_cstr are valid
        let result = unsafe {
            talu_sys::talu_db_table_session_delete(
                self.path_cstr.as_ptr(),
                session_id_cstr.as_ptr(),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        Ok(())
    }

    /// Load the full item history for a session.
    ///
    /// Returns a read-only snapshot of the session items. The returned handle
    /// is independent of the storage - items are fully materialized in memory.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::StorageCorrupted` if loading fails.
    /// Returns an empty handle if the session exists but has no items.
    pub fn load_session(
        &self,
        session_id: &str,
    ) -> Result<crate::responses::ResponsesHandle, StorageError> {
        let session_id_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        // SAFETY: path_cstr and session_id_cstr are valid CStrings.
        let ptr = unsafe {
            talu_sys::talu_db_table_session_load_conversation(
                self.path_cstr.as_ptr(),
                session_id_cstr.as_ptr(),
            )
        };

        if ptr.is_null() {
            return Err(StorageError::from_code(
                ERROR_CODE_STORAGE_ERROR,
                session_id,
            ));
        }

        // SAFETY: ptr is a non-null ResponsesHandle allocated by the C API.
        // from_raw_owned takes ownership and will free on drop.
        Ok(unsafe {
            crate::responses::ResponsesHandle::from_raw_owned(ptr as *mut talu_sys::ResponsesHandle)
        })
    }

    /// Convert CSessionList to Vec<SessionRecord>, applying limit.
    ///
    /// # Safety
    ///
    /// `c_list` must be a valid pointer returned by `talu_db_table_session_list`.
    unsafe fn convert_session_list(
        c_list: *const CSessionList,
        limit: Option<usize>,
    ) -> Vec<SessionRecord> {
        if c_list.is_null() {
            return Vec::new();
        }

        let list = &*c_list;
        if list.sessions.is_null() || list.count == 0 {
            return Vec::new();
        }

        let count = limit.map_or(list.count, |l| l.min(list.count));
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let c_record = &*list.sessions.add(i);
            result.push(Self::convert_session_record(c_record));
        }

        result
    }

    /// Convert a single CSessionRecord to SessionRecord.
    ///
    /// # Safety
    ///
    /// `c_record` must be a valid pointer to a CSessionRecord.
    unsafe fn convert_session_record(c_record: &CSessionRecord) -> SessionRecord {
        SessionRecord {
            session_id: cstr_to_string(c_record.session_id),
            title: optional_cstr_to_string(c_record.title),
            model: optional_cstr_to_string(c_record.model),
            created_at: c_record.created_at_ms,
            updated_at: c_record.updated_at_ms,
            marker: optional_cstr_to_string(c_record.marker),
        }
    }

    /// Convert a single CSessionRecord to SessionRecordFull (all fields).
    ///
    /// # Safety
    ///
    /// `c_record` must be a valid pointer to a CSessionRecord.
    unsafe fn convert_session_record_full(c_record: &CSessionRecord) -> SessionRecordFull {
        SessionRecordFull {
            session_id: cstr_to_string(c_record.session_id),
            model: optional_cstr_to_string(c_record.model),
            title: optional_cstr_to_string(c_record.title),
            system_prompt: optional_cstr_to_string(c_record.system_prompt),
            config_json: optional_cstr_to_string(c_record.config_json),
            marker: optional_cstr_to_string(c_record.marker),
            parent_session_id: optional_cstr_to_string(c_record.parent_session_id),
            group_id: optional_cstr_to_string(c_record.group_id),
            head_item_id: c_record.head_item_id,
            ttl_ts: c_record.ttl_ts,
            metadata_json: optional_cstr_to_string(c_record.metadata_json),
            search_snippet: optional_cstr_to_string(c_record.search_snippet),
            source_doc_id: optional_cstr_to_string(c_record.source_doc_id),
            project_id: optional_cstr_to_string(c_record.project_id),
            created_at: c_record.created_at_ms,
            updated_at: c_record.updated_at_ms,
        }
    }

    /// List sessions with cursor-based pagination and extended filters.
    ///
    /// Returns a `SessionListResult` with `has_more` and optional `next_cursor`
    /// for stable pagination. Determines `has_more` by requesting `limit + 1`
    /// rows from the C API.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum sessions to return (clamped to 1..=100)
    /// * `cursor` - Continue from a previous page's cursor position
    /// * `group_id` - Filter by group (multi-tenant); uses scalar column pruning
    /// * `params` - Extended search parameters (text, tags, markers, model, dates, etc.)
    pub fn list_sessions_paginated_ex(
        &self,
        limit: usize,
        cursor: Option<&SessionCursor>,
        group_id: Option<&str>,
        params: &SearchParams<'_>,
    ) -> Result<SessionListResult, StorageError> {
        let limit = limit.clamp(1, 100);
        // Request one extra to detect has_more
        let c_limit = (limit + 1) as u32;

        let (before_ts, cursor_session_cstr) = match cursor {
            Some(c) => {
                let cstr = CString::new(c.session_id.as_str()).map_err(|_| {
                    StorageError::InvalidArgument(
                        "Cursor session_id contains null bytes".to_string(),
                    )
                })?;
                (c.updated_at_ms, Some(cstr))
            }
            None => (0i64, None),
        };

        // Helper to convert Option<&str> to Option<CString>
        fn opt_cstring(s: Option<&str>, name: &str) -> Result<Option<CString>, StorageError> {
            match s {
                Some(v) => Ok(Some(CString::new(v).map_err(|_| {
                    StorageError::InvalidArgument(format!("{name} contains null bytes"))
                })?)),
                None => Ok(None),
            }
        }

        let group_id_cstr = opt_cstring(group_id, "group_id")?;
        let query_cstr = opt_cstring(params.query, "search query")?;
        let tags_filter_cstr = opt_cstring(params.tags_filter, "tags_filter")?;
        let tags_filter_any_cstr = opt_cstring(params.tags_filter_any, "tags_filter_any")?;
        let marker_filter_cstr = opt_cstring(params.marker_filter, "marker_filter")?;
        let marker_filter_any_cstr = opt_cstring(params.marker_filter_any, "marker_filter_any")?;
        let model_filter_cstr = opt_cstring(params.model_filter, "model_filter")?;
        let source_doc_id_cstr = opt_cstring(params.source_doc_id, "source_doc_id")?;
        let project_id_cstr = opt_cstring(params.project_id, "project_id")?;

        let cursor_ptr = cursor_session_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let group_ptr = group_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let query_ptr = query_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let tags_filter_ptr = tags_filter_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let tags_filter_any_ptr = tags_filter_any_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let marker_filter_ptr = marker_filter_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let marker_filter_any_ptr = marker_filter_any_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let model_filter_ptr = model_filter_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let source_doc_id_ptr = source_doc_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let project_id_ptr = project_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());

        // Timestamps: 0 means no filter
        let created_after_ms = params.created_after_ms.unwrap_or(0);
        let created_before_ms = params.created_before_ms.unwrap_or(0);
        let updated_after_ms = params.updated_after_ms.unwrap_or(0);
        let updated_before_ms = params.updated_before_ms.unwrap_or(0);

        // has_tags: -1 = no filter, 0 = false, 1 = true
        let has_tags_int: i32 = match params.has_tags {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };

        let mut c_list: *mut CSessionList = std::ptr::null_mut();

        // SAFETY: path_cstr is valid, all pointers are valid or null
        let result = unsafe {
            talu_sys::talu_db_table_session_list_ex(
                self.path_cstr.as_ptr(),
                c_limit,
                before_ts,
                cursor_ptr,
                group_ptr,
                query_ptr,
                tags_filter_ptr,
                tags_filter_any_ptr,
                marker_filter_ptr,
                marker_filter_any_ptr,
                model_filter_ptr,
                created_after_ms,
                created_before_ms,
                updated_after_ms,
                updated_before_ms,
                has_tags_int,
                source_doc_id_ptr,
                project_id_ptr,
                params.project_id_null as std::os::raw::c_int,
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(SessionListResult {
                sessions: Vec::new(),
                has_more: false,
                next_cursor: None,
            });
        }

        // SAFETY: c_list is non-null and was returned by talu_db_table_session_list_ex
        let all_records = unsafe { Self::convert_session_list_full(c_list) };

        // Free C memory before returning
        // SAFETY: c_list was returned by talu_db_table_session_list_ex
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        let has_more = all_records.len() > limit;
        let sessions: Vec<SessionRecordFull> = all_records.into_iter().take(limit).collect();

        let next_cursor = if has_more {
            sessions.last().map(|last| SessionCursor {
                updated_at_ms: last.updated_at,
                session_id: last.session_id.clone(),
            })
        } else {
            None
        };

        Ok(SessionListResult {
            sessions,
            has_more,
            next_cursor,
        })
    }

    /// List sessions with cursor-based pagination.
    ///
    /// Returns a `SessionListResult` with `has_more` and optional `next_cursor`
    /// for stable pagination. Determines `has_more` by requesting `limit + 1`
    /// rows from the C API.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum sessions to return (clamped to 1..=100)
    /// * `cursor` - Continue from a previous page's cursor position
    /// * `group_id` - Filter by group (multi-tenant); uses scalar column pruning
    /// * `query` - Case-insensitive substring search across title, model, system_prompt
    /// * `tags_filter` - Space-separated tags for AND matching (sessions must have ALL tags)
    /// * `tags_filter_any` - Space-separated tags for OR matching (sessions must have ANY tag)
    pub fn list_sessions_paginated(
        &self,
        limit: usize,
        cursor: Option<&SessionCursor>,
        group_id: Option<&str>,
        query: Option<&str>,
        tags_filter: Option<&str>,
        tags_filter_any: Option<&str>,
    ) -> Result<SessionListResult, StorageError> {
        let limit = limit.clamp(1, 100);
        // Request one extra to detect has_more
        let c_limit = (limit + 1) as u32;

        let (before_ts, cursor_session_cstr) = match cursor {
            Some(c) => {
                let cstr = CString::new(c.session_id.as_str()).map_err(|_| {
                    StorageError::InvalidArgument(
                        "Cursor session_id contains null bytes".to_string(),
                    )
                })?;
                (c.updated_at_ms, Some(cstr))
            }
            None => (0i64, None),
        };

        let group_id_cstr = match group_id {
            Some(g) => Some(CString::new(g).map_err(|_| {
                StorageError::InvalidArgument("group_id contains null bytes".to_string())
            })?),
            None => None,
        };

        let query_cstr = match query {
            Some(q) => Some(CString::new(q).map_err(|_| {
                StorageError::InvalidArgument("search query contains null bytes".to_string())
            })?),
            None => None,
        };

        let tags_filter_cstr = match tags_filter {
            Some(t) => Some(CString::new(t).map_err(|_| {
                StorageError::InvalidArgument("tags_filter contains null bytes".to_string())
            })?),
            None => None,
        };

        let tags_filter_any_cstr = match tags_filter_any {
            Some(t) => Some(CString::new(t).map_err(|_| {
                StorageError::InvalidArgument("tags_filter_any contains null bytes".to_string())
            })?),
            None => None,
        };

        let cursor_ptr = cursor_session_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let group_ptr = group_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let query_ptr = query_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let tags_filter_ptr = tags_filter_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let tags_filter_any_ptr = tags_filter_any_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());

        let mut c_list: *mut CSessionList = std::ptr::null_mut();

        // SAFETY: path_cstr is valid, all pointers are valid or null
        let result = unsafe {
            talu_sys::talu_db_table_session_list(
                self.path_cstr.as_ptr(),
                c_limit,
                before_ts,
                cursor_ptr,
                group_ptr,
                query_ptr,
                tags_filter_ptr,
                tags_filter_any_ptr,
                std::ptr::null(), // no project_id
                0,                // no project_id_null
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(SessionListResult {
                sessions: Vec::new(),
                has_more: false,
                next_cursor: None,
            });
        }

        // SAFETY: c_list is non-null and was returned by talu_db_table_session_list
        let all_records = unsafe { Self::convert_session_list_full(c_list) };

        // Free C memory before returning
        // SAFETY: c_list was returned by talu_db_table_session_list
        unsafe { talu_sys::talu_db_table_session_free_list(c_list) };

        let has_more = all_records.len() > limit;
        let sessions: Vec<SessionRecordFull> = all_records.into_iter().take(limit).collect();

        let next_cursor = if has_more {
            sessions.last().map(|last| SessionCursor {
                updated_at_ms: last.updated_at,
                session_id: last.session_id.clone(),
            })
        } else {
            None
        };

        Ok(SessionListResult {
            sessions,
            has_more,
            next_cursor,
        })
    }

    /// Update session metadata (PATCH semantics).
    ///
    /// Uses read-modify-write internally: reads the current session head,
    /// merges non-None fields, writes a new complete record. Only provided
    /// fields are changed; `None` fields are preserved.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::SessionNotFound` if the session doesn't exist.
    pub fn update_session(
        &self,
        session_id: &str,
        updates: &SessionUpdate,
    ) -> Result<(), StorageError> {
        let session_id_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let title_cstr = match &updates.title {
            Some(t) => Some(CString::new(t.as_str()).map_err(|_| {
                StorageError::InvalidArgument("Title contains null bytes".to_string())
            })?),
            None => None,
        };
        let marker_cstr = match &updates.marker {
            Some(s) => Some(CString::new(s.as_str()).map_err(|_| {
                StorageError::InvalidArgument("Marker contains null bytes".to_string())
            })?),
            None => None,
        };
        let metadata_cstr = match &updates.metadata_json {
            Some(m) => Some(CString::new(m.as_str()).map_err(|_| {
                StorageError::InvalidArgument("Metadata JSON contains null bytes".to_string())
            })?),
            None => None,
        };
        let source_doc_id_cstr = match &updates.source_doc_id {
            Some(s) => Some(CString::new(s.as_str()).map_err(|_| {
                StorageError::InvalidArgument("Source doc ID contains null bytes".to_string())
            })?),
            None => None,
        };
        let project_id_cstr = match &updates.project_id {
            Some(p) => Some(CString::new(p.as_str()).map_err(|_| {
                StorageError::InvalidArgument("Project ID contains null bytes".to_string())
            })?),
            None => None,
        };

        let title_ptr = title_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let marker_ptr = marker_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let metadata_ptr = metadata_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let source_doc_id_ptr = source_doc_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());
        let project_id_ptr = project_id_cstr
            .as_ref()
            .map_or(std::ptr::null(), |c| c.as_ptr());

        // SAFETY: all pointers are valid CStrings or null
        let result = unsafe {
            talu_sys::talu_db_table_session_update_ex(
                self.path_cstr.as_ptr(),
                session_id_cstr.as_ptr(),
                title_ptr,
                marker_ptr,
                metadata_ptr,
                source_doc_id_ptr,
                project_id_ptr,
                if updates.clear_project_id { 1 } else { 0 },
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        Ok(())
    }

    /// Fork a session at a specific item, creating a new session.
    ///
    /// Creates a new session containing items up to and including `target_item_id`
    /// from the source session. The new session's `parent_session_id` references
    /// the source. Items have `origin_session_id` and `origin_item_id` set for
    /// lineage tracking.
    ///
    /// # Arguments
    ///
    /// * `source_session_id` - Session to fork from
    /// * `target_item_id` - Item ID to fork at (inclusive)
    /// * `new_session_id` - ID for the new forked session
    ///
    /// # Errors
    ///
    /// Returns `StorageError::SessionNotFound` if the source session doesn't exist.
    /// Returns `StorageError::ItemNotFound` if `target_item_id` doesn't exist in the source.
    pub fn fork_session(
        &self,
        source_session_id: &str,
        target_item_id: u64,
        new_session_id: &str,
    ) -> Result<(), StorageError> {
        let source_cstr = CString::new(source_session_id).map_err(|_| {
            StorageError::InvalidArgument("Source session ID contains null bytes".to_string())
        })?;
        let new_cstr = CString::new(new_session_id).map_err(|_| {
            StorageError::InvalidArgument("New session ID contains null bytes".to_string())
        })?;

        // SAFETY: all pointers are valid CStrings
        let result = unsafe {
            talu_sys::talu_db_table_session_fork(
                self.path_cstr.as_ptr(),
                source_cstr.as_ptr(),
                target_item_id,
                new_cstr.as_ptr(),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, source_session_id));
        }

        Ok(())
    }

    /// Convert CSessionList to Vec<SessionRecordFull> (all fields).
    ///
    /// # Safety
    ///
    /// `c_list` must be a valid pointer returned by `talu_db_table_session_list`.
    unsafe fn convert_session_list_full(c_list: *const CSessionList) -> Vec<SessionRecordFull> {
        if c_list.is_null() {
            return Vec::new();
        }

        let list = &*c_list;
        if list.sessions.is_null() || list.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(list.count);
        for i in 0..list.count {
            let c_record = &*list.sessions.add(i);
            result.push(Self::convert_session_record_full(c_record));
        }

        result
    }

    // =========================================================================
    // Tag Operations
    // =========================================================================

    /// List all tags, optionally filtered by group.
    pub fn list_tags(&self, group_id: Option<&str>) -> Result<Vec<TagRecord>, StorageError> {
        let group_cstr = group_id.map(|g| CString::new(g)).transpose().map_err(|_| {
            StorageError::InvalidArgument("Group ID contains null bytes".to_string())
        })?;

        let mut c_list: *mut CTagList = std::ptr::null_mut();

        let result = unsafe {
            talu_sys::talu_db_table_tag_list(
                self.path_cstr.as_ptr(),
                group_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(
                result,
                &self.path.to_string_lossy(),
            ));
        }

        if c_list.is_null() {
            return Ok(Vec::new());
        }

        let rust_vec = unsafe { Self::convert_tag_list(c_list) };

        // Free C memory
        unsafe { talu_sys::talu_db_table_tag_free_list(c_list) };

        Ok(rust_vec)
    }

    /// Get a tag by ID.
    pub fn get_tag(&self, tag_id: &str) -> Result<TagRecord, StorageError> {
        let tag_id_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let mut c_record = CTagRecord::default();

        let result = unsafe {
            talu_sys::talu_db_table_tag_get(
                self.path_cstr.as_ptr(),
                tag_id_cstr.as_ptr(),
                &mut c_record as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, tag_id));
        }

        Ok(unsafe { Self::convert_tag_record(&c_record) })
    }

    /// Get a tag by name (within optional group).
    pub fn get_tag_by_name(
        &self,
        name: &str,
        group_id: Option<&str>,
    ) -> Result<TagRecord, StorageError> {
        let name_cstr = CString::new(name).map_err(|_| {
            StorageError::InvalidArgument("Tag name contains null bytes".to_string())
        })?;

        let group_cstr = group_id.map(|g| CString::new(g)).transpose().map_err(|_| {
            StorageError::InvalidArgument("Group ID contains null bytes".to_string())
        })?;

        let mut c_record = CTagRecord::default();

        let result = unsafe {
            talu_sys::talu_db_table_tag_get_by_name(
                self.path_cstr.as_ptr(),
                name_cstr.as_ptr(),
                group_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                &mut c_record as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, name));
        }

        Ok(unsafe { Self::convert_tag_record(&c_record) })
    }

    /// Create a new tag.
    pub fn create_tag(&self, tag: &TagCreate) -> Result<(), StorageError> {
        let tag_id_cstr = CString::new(tag.tag_id.as_str())
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let name_cstr = CString::new(tag.name.as_str()).map_err(|_| {
            StorageError::InvalidArgument("Tag name contains null bytes".to_string())
        })?;

        let color_cstr = tag
            .color
            .as_ref()
            .map(|c| CString::new(c.as_str()))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("Color contains null bytes".to_string()))?;

        let desc_cstr = tag
            .description
            .as_ref()
            .map(|d| CString::new(d.as_str()))
            .transpose()
            .map_err(|_| {
                StorageError::InvalidArgument("Description contains null bytes".to_string())
            })?;

        let group_cstr = tag
            .group_id
            .as_ref()
            .map(|g| CString::new(g.as_str()))
            .transpose()
            .map_err(|_| {
                StorageError::InvalidArgument("Group ID contains null bytes".to_string())
            })?;

        let result = unsafe {
            talu_sys::talu_db_table_tag_create(
                self.path_cstr.as_ptr(),
                tag_id_cstr.as_ptr(),
                name_cstr.as_ptr(),
                color_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                desc_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                group_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, &tag.tag_id));
        }

        Ok(())
    }

    /// Update an existing tag.
    pub fn update_tag(&self, tag_id: &str, update: &TagUpdate) -> Result<(), StorageError> {
        let tag_id_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let name_cstr = update
            .name
            .as_ref()
            .map(|n| CString::new(n.as_str()))
            .transpose()
            .map_err(|_| {
                StorageError::InvalidArgument("Tag name contains null bytes".to_string())
            })?;

        let color_cstr = update
            .color
            .as_ref()
            .map(|c| CString::new(c.as_str()))
            .transpose()
            .map_err(|_| StorageError::InvalidArgument("Color contains null bytes".to_string()))?;

        let desc_cstr = update
            .description
            .as_ref()
            .map(|d| CString::new(d.as_str()))
            .transpose()
            .map_err(|_| {
                StorageError::InvalidArgument("Description contains null bytes".to_string())
            })?;

        let result = unsafe {
            talu_sys::talu_db_table_tag_update(
                self.path_cstr.as_ptr(),
                tag_id_cstr.as_ptr(),
                name_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                color_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                desc_cstr.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, tag_id));
        }

        Ok(())
    }

    /// Delete a tag.
    pub fn delete_tag(&self, tag_id: &str) -> Result<(), StorageError> {
        let tag_id_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let result = unsafe {
            talu_sys::talu_db_table_tag_delete(self.path_cstr.as_ptr(), tag_id_cstr.as_ptr())
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, tag_id));
        }

        Ok(())
    }

    /// Add a tag to a session.
    pub fn add_session_tag(&self, session_id: &str, tag_id: &str) -> Result<(), StorageError> {
        let session_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let tag_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let result = unsafe {
            talu_sys::talu_db_table_session_add_tag(
                self.path_cstr.as_ptr(),
                session_cstr.as_ptr(),
                tag_cstr.as_ptr(),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        Ok(())
    }

    /// Remove a tag from a session.
    pub fn remove_session_tag(&self, session_id: &str, tag_id: &str) -> Result<(), StorageError> {
        let session_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let tag_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let result = unsafe {
            talu_sys::talu_db_table_session_remove_tag(
                self.path_cstr.as_ptr(),
                session_cstr.as_ptr(),
                tag_cstr.as_ptr(),
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        Ok(())
    }

    /// Get all tag IDs for a session.
    pub fn get_session_tags(&self, session_id: &str) -> Result<Vec<String>, StorageError> {
        let session_cstr = CString::new(session_id).map_err(|_| {
            StorageError::InvalidArgument("Session ID contains null bytes".to_string())
        })?;

        let mut c_list: *mut talu_sys::CRelationStringList = std::ptr::null_mut();

        let result = unsafe {
            talu_sys::talu_db_table_session_get_tags(
                self.path_cstr.as_ptr(),
                session_cstr.as_ptr(),
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, session_id));
        }

        if c_list.is_null() {
            return Ok(Vec::new());
        }

        let rust_vec = unsafe { Self::convert_relation_string_list(c_list) };

        // Free C memory
        unsafe { talu_sys::talu_db_table_free_relation_string_list(c_list) };

        Ok(rust_vec)
    }

    /// Get all session IDs that have a specific tag.
    pub fn get_tag_sessions(&self, tag_id: &str) -> Result<Vec<String>, StorageError> {
        let tag_cstr = CString::new(tag_id)
            .map_err(|_| StorageError::InvalidArgument("Tag ID contains null bytes".to_string()))?;

        let mut c_list: *mut talu_sys::CRelationStringList = std::ptr::null_mut();

        let result = unsafe {
            talu_sys::talu_db_table_tag_get_conversations(
                self.path_cstr.as_ptr(),
                tag_cstr.as_ptr(),
                &mut c_list as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, tag_id));
        }

        if c_list.is_null() {
            return Ok(Vec::new());
        }

        let rust_vec = unsafe { Self::convert_relation_string_list(c_list) };

        // Free C memory
        unsafe { talu_sys::talu_db_table_free_relation_string_list(c_list) };

        Ok(rust_vec)
    }

    /// Get tags for multiple sessions in a single scan.
    ///
    /// Returns a map from session_id to list of tag_ids. Sessions with no tags
    /// are omitted from the result.
    pub fn get_sessions_tags_batch(
        &self,
        session_ids: &[&str],
    ) -> Result<std::collections::HashMap<String, Vec<String>>, StorageError> {
        use std::collections::HashMap;

        if session_ids.is_empty() {
            return Ok(HashMap::new());
        }

        // Build null-terminated C strings for each session ID.
        let c_strings: Vec<CString> = session_ids
            .iter()
            .map(|s| {
                CString::new(*s).map_err(|_| {
                    StorageError::InvalidArgument(
                        "Session ID contains null bytes".to_string(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let c_ptrs: Vec<*const std::os::raw::c_char> =
            c_strings.iter().map(|c| c.as_ptr()).collect();

        let mut c_batch: *mut talu_sys::CSessionTagBatch = std::ptr::null_mut();

        let result = unsafe {
            talu_sys::talu_db_table_sessions_get_tags_batch(
                self.path_cstr.as_ptr(),
                c_ptrs.as_ptr(),
                session_ids.len() as u32,
                &mut c_batch as *mut _,
            )
        };

        if result != ERROR_CODE_OK {
            return Err(StorageError::from_code(result, "batch_tags"));
        }

        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        if !c_batch.is_null() {
            let batch = unsafe { &*c_batch };
            if batch.count > 0 && !batch.session_ids.is_null() && !batch.tag_ids.is_null() {
                for i in 0..batch.count {
                    let sid_ptr = unsafe { *batch.session_ids.add(i) };
                    let tid_ptr = unsafe { *batch.tag_ids.add(i) };
                    if !sid_ptr.is_null() && !tid_ptr.is_null() {
                        let sid = unsafe { std::ffi::CStr::from_ptr(sid_ptr) }
                            .to_string_lossy()
                            .into_owned();
                        let tid = unsafe { std::ffi::CStr::from_ptr(tid_ptr) }
                            .to_string_lossy()
                            .into_owned();
                        map.entry(sid).or_default().push(tid);
                    }
                }
            }
            unsafe { talu_sys::talu_db_table_free_session_tag_batch(c_batch) };
        }

        Ok(map)
    }

    // =========================================================================
    // Tag Conversion Helpers
    // =========================================================================

    /// Convert CTagList to Vec<TagRecord>.
    unsafe fn convert_tag_list(c_list: *const CTagList) -> Vec<TagRecord> {
        if c_list.is_null() {
            return Vec::new();
        }

        let list = &*c_list;
        if list.tags.is_null() || list.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(list.count);
        for i in 0..list.count {
            let c_record = &*list.tags.add(i);
            result.push(Self::convert_tag_record(c_record));
        }

        result
    }

    /// Convert a single CTagRecord to TagRecord.
    unsafe fn convert_tag_record(c_record: &CTagRecord) -> TagRecord {
        TagRecord {
            tag_id: cstr_to_string(c_record.tag_id),
            name: cstr_to_string(c_record.name),
            color: optional_cstr_to_string(c_record.color),
            description: optional_cstr_to_string(c_record.description),
            group_id: optional_cstr_to_string(c_record.group_id),
            created_at: c_record.created_at_ms,
            updated_at: c_record.updated_at_ms,
        }
    }

    /// Convert CRelationStringList to Vec<String>.
    unsafe fn convert_relation_string_list(c_list: *const CRelationStringList) -> Vec<String> {
        if c_list.is_null() {
            return Vec::new();
        }

        let list = &*c_list;
        if list.strings.is_null() || list.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(list.count);
        let strings_ptr = list.strings;
        for i in 0..list.count {
            let str_ptr = *strings_ptr.add(i);
            if !str_ptr.is_null() {
                result.push(CStr::from_ptr(str_ptr).to_string_lossy().into_owned());
            }
        }

        result
    }
}

/// Session metadata (owned strings, safe to use after C memory freed).
///
/// Note: This is a subset of CSessionRecord fields. Additional fields
/// (system_prompt, config_json, parent_session_id, group_id, head_item_id,
/// ttl_ts, metadata_json) are available via the full CSessionRecord but
/// omitted here for CLI simplicity.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    /// Unique session identifier.
    pub session_id: String,
    /// Human-readable session title.
    pub title: Option<String>,
    /// Model identifier used for this session.
    pub model: Option<String>,
    /// Creation timestamp (milliseconds since epoch).
    pub created_at: i64,
    /// Last update timestamp (milliseconds since epoch).
    pub updated_at: i64,
    /// Session marker (e.g., "pinned", "archived", "deleted").
    pub marker: Option<String>,
}

/// Full session record with all fields from CSessionRecord.
///
/// Use this when you need complete session metadata.
#[derive(Debug, Clone)]
pub struct SessionRecordFull {
    pub session_id: String,
    pub model: Option<String>,
    pub title: Option<String>,
    pub system_prompt: Option<String>,
    pub config_json: Option<String>,
    pub marker: Option<String>,
    pub parent_session_id: Option<String>,
    pub group_id: Option<String>,
    pub head_item_id: u64,
    pub ttl_ts: i64,
    pub metadata_json: Option<String>,
    pub search_snippet: Option<String>,
    /// Source document ID for lineage tracking.
    /// Links this session to the prompt/persona document that spawned it.
    pub source_doc_id: Option<String>,
    /// Project identifier for multi-project session organization.
    pub project_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Opaque cursor for composite pagination.
///
/// Encodes `(updated_at_ms, session_id)` — the session_id is passed as a
/// string to the C API, which hashes it internally (hash algorithm stays
/// private to core).
#[derive(Debug, Clone)]
pub struct SessionCursor {
    pub updated_at_ms: i64,
    pub session_id: String,
}

/// Paginated list result with cursor metadata.
#[derive(Debug)]
pub struct SessionListResult {
    pub sessions: Vec<SessionRecordFull>,
    pub has_more: bool,
    pub next_cursor: Option<SessionCursor>,
}

/// Batch list result with total count for offset-based pagination.
#[derive(Debug)]
pub struct SessionBatchResult {
    /// Sessions in the current page.
    pub sessions: Vec<SessionRecordFull>,
    /// Total matching sessions (before offset/limit).
    pub total: usize,
    /// Whether there are more sessions after this page.
    pub has_more: bool,
}

/// Partial update for PATCH semantics (top-level replacement, not deep merge).
///
/// Only provided fields are changed; `None` fields are preserved.
/// `metadata_json` replaces the entire metadata object when present.
#[derive(Debug, Default)]
pub struct SessionUpdate {
    pub title: Option<String>,
    pub marker: Option<String>,
    pub metadata_json: Option<String>,
    /// Source document ID for lineage tracking.
    pub source_doc_id: Option<String>,
    /// Project ID for multi-project session organization.
    /// Set to `Some(id)` to update, `None` to leave unchanged.
    pub project_id: Option<String>,
    /// When true, clears the project_id field (sets it to null in storage).
    pub clear_project_id: bool,
}

// =============================================================================
// Search Types
// =============================================================================

/// Extended search parameters for `list_sessions_paginated_ex`.
#[derive(Debug, Default)]
pub struct SearchParams<'a> {
    /// Case-insensitive substring search across title, model, system_prompt, content.
    pub query: Option<&'a str>,
    /// Space-separated tags for AND matching (sessions must have ALL tags).
    pub tags_filter: Option<&'a str>,
    /// Space-separated tags for OR matching (sessions must have ANY tag).
    pub tags_filter_any: Option<&'a str>,
    /// Marker exact match filter.
    pub marker_filter: Option<&'a str>,
    /// Space-separated markers for OR matching.
    pub marker_filter_any: Option<&'a str>,
    /// Model filter (case-insensitive, supports wildcard suffix like "qwen*").
    pub model_filter: Option<&'a str>,
    /// Created after timestamp (inclusive, milliseconds).
    pub created_after_ms: Option<i64>,
    /// Created before timestamp (exclusive, milliseconds).
    pub created_before_ms: Option<i64>,
    /// Updated after timestamp (inclusive, milliseconds).
    pub updated_after_ms: Option<i64>,
    /// Updated before timestamp (exclusive, milliseconds).
    pub updated_before_ms: Option<i64>,
    /// Has tags filter: true = must have at least one tag, false = must have no tags.
    pub has_tags: Option<bool>,
    /// Source document ID filter: exact match on sessions created from this prompt document.
    pub source_doc_id: Option<&'a str>,
    /// Project ID filter: exact match on sessions belonging to this project.
    pub project_id: Option<&'a str>,
    /// Project null filter: include only sessions with no project_id.
    pub project_id_null: bool,
}

// =============================================================================
// Tag Types
// =============================================================================

/// Tag record from TaluDB.
#[derive(Debug, Clone)]
pub struct TagRecord {
    /// Unique tag identifier (UUID).
    pub tag_id: String,
    /// Tag name (unique within group).
    pub name: String,
    /// Optional hex color (e.g., "#4a90d9").
    pub color: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Optional group ID for multi-tenant isolation.
    pub group_id: Option<String>,
    /// Creation timestamp (milliseconds since epoch).
    pub created_at: i64,
    /// Last update timestamp (milliseconds since epoch).
    pub updated_at: i64,
}

/// Parameters for creating a new tag.
#[derive(Debug)]
pub struct TagCreate {
    /// Unique tag identifier (UUID, caller-generated).
    pub tag_id: String,
    /// Tag name (unique within group).
    pub name: String,
    /// Optional hex color (e.g., "#4a90d9").
    pub color: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Optional group ID for multi-tenant isolation.
    pub group_id: Option<String>,
}

/// Partial update for tag (only provided fields are changed).
#[derive(Debug, Default)]
pub struct TagUpdate {
    /// New name (None = keep existing).
    pub name: Option<String>,
    /// New color (None = keep existing).
    pub color: Option<String>,
    /// New description (None = keep existing).
    pub description: Option<String>,
}

/// Convert a C string pointer to an owned Rust String.
///
/// Returns an empty string if the pointer is null.
fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        // SAFETY: Caller ensures ptr is valid if non-null
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

/// Convert an optional C string pointer to Option<String>.
///
/// Returns None if the pointer is null.
fn optional_cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: Caller ensures ptr is valid if non-null
        Some(unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cstr_to_string_null() {
        let result = cstr_to_string(std::ptr::null());
        assert_eq!(result, "");
    }

    #[test]
    fn test_optional_cstr_to_string_null() {
        let result = optional_cstr_to_string(std::ptr::null());
        assert!(result.is_none());
    }

    #[test]
    fn test_storage_error_display() {
        let err = StorageError::SessionNotFound("test-session".to_string());
        assert_eq!(format!("{}", err), "Session not found: test-session");
    }

    #[test]
    fn test_storage_error_item_not_found_display() {
        let err = StorageError::ItemNotFound("item-42".to_string());
        assert_eq!(format!("{}", err), "Item not found: item-42");
    }

    #[test]
    fn test_session_cursor_clone() {
        let cursor = SessionCursor {
            updated_at_ms: 1700000000000,
            session_id: "sess-abc".to_string(),
        };
        let cloned = cursor.clone();
        assert_eq!(cloned.updated_at_ms, 1700000000000);
        assert_eq!(cloned.session_id, "sess-abc");
    }

    #[test]
    fn test_session_update_default() {
        let update = SessionUpdate::default();
        assert!(update.title.is_none());
        assert!(update.marker.is_none());
        assert!(update.metadata_json.is_none());
    }

    #[test]
    fn test_session_list_result_empty() {
        let result = SessionListResult {
            sessions: Vec::new(),
            has_more: false,
            next_cursor: None,
        };
        assert!(result.sessions.is_empty());
        assert!(!result.has_more);
        assert!(result.next_cursor.is_none());
    }

    #[test]
    fn test_storage_open_nonexistent_path() {
        let result = StorageHandle::open("/nonexistent/path/to/db");
        assert!(result.is_err());
        match result.unwrap_err() {
            StorageError::StorageNotFound(p) => {
                assert_eq!(p, PathBuf::from("/nonexistent/path/to/db"));
            }
            other => panic!("Expected StorageNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_tag_record_fields() {
        let tag = TagRecord {
            tag_id: "tag-123".to_string(),
            name: "test-tag".to_string(),
            color: Some("#ff0000".to_string()),
            description: Some("A test tag".to_string()),
            group_id: Some("group-1".to_string()),
            created_at: 1700000000000,
            updated_at: 1700000001000,
        };
        assert_eq!(tag.tag_id, "tag-123");
        assert_eq!(tag.name, "test-tag");
        assert_eq!(tag.color.as_deref(), Some("#ff0000"));
        assert_eq!(tag.description.as_deref(), Some("A test tag"));
        assert_eq!(tag.group_id.as_deref(), Some("group-1"));
        assert_eq!(tag.created_at, 1700000000000);
        assert_eq!(tag.updated_at, 1700000001000);
    }

    #[test]
    fn test_tag_record_optional_fields() {
        let tag = TagRecord {
            tag_id: "tag-456".to_string(),
            name: "minimal-tag".to_string(),
            color: None,
            description: None,
            group_id: None,
            created_at: 1700000000000,
            updated_at: 1700000000000,
        };
        assert!(tag.color.is_none());
        assert!(tag.description.is_none());
        assert!(tag.group_id.is_none());
    }

    #[test]
    fn test_tag_create_fields() {
        let create = TagCreate {
            tag_id: "new-tag".to_string(),
            name: "my-tag".to_string(),
            color: Some("#00ff00".to_string()),
            description: None,
            group_id: Some("tenant-1".to_string()),
        };
        assert_eq!(create.tag_id, "new-tag");
        assert_eq!(create.name, "my-tag");
        assert_eq!(create.color.as_deref(), Some("#00ff00"));
        assert!(create.description.is_none());
        assert_eq!(create.group_id.as_deref(), Some("tenant-1"));
    }

    #[test]
    fn test_tag_update_fields() {
        let update = TagUpdate {
            name: Some("renamed-tag".to_string()),
            color: Some("#0000ff".to_string()),
            description: Some("Updated description".to_string()),
        };
        assert_eq!(update.name.as_deref(), Some("renamed-tag"));
        assert_eq!(update.color.as_deref(), Some("#0000ff"));
        assert_eq!(update.description.as_deref(), Some("Updated description"));
    }

    #[test]
    fn test_tag_update_partial() {
        let update = TagUpdate {
            name: None,
            color: Some("#ffffff".to_string()),
            description: None,
        };
        assert!(update.name.is_none());
        assert_eq!(update.color.as_deref(), Some("#ffffff"));
        assert!(update.description.is_none());
    }

    #[test]
    fn test_storage_error_tag_not_found_display() {
        let err = StorageError::TagNotFound("tag-abc".to_string());
        assert_eq!(format!("{}", err), "Tag not found: tag-abc");
    }
}
