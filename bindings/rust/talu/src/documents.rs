//! Document storage and management.
//!
//! Provides safe Rust wrappers for TaluDB document operations including
//! CRUD, search, tagging, TTL, versioning, and compaction.
//!
//! # Example
//!
//! ```no_run
//! use talu::documents::{DocumentsHandle, DocumentRecord, DocumentError};
//!
//! let handle = DocumentsHandle::open("./taludb")?;
//!
//! // Create a document
//! handle.create(
//!     "doc-123",
//!     "prompt",
//!     "My Prompt",
//!     r#"{"content": "Hello"}"#,
//!     None, None, None, None, None,
//! )?;
//!
//! // Get document
//! if let Some(doc) = handle.get("doc-123")? {
//!     println!("{}: {}", doc.doc_id, doc.title);
//! }
//!
//! // Search documents
//! let results = handle.search("hello", Some("prompt"), 10)?;
//! # Ok::<(), DocumentError>(())
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::ptr;

/// Error codes from the C API (must match error_codes.zig)
const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_STORAGE_ERROR: i32 = 700;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;

/// Error types for document operations.
#[derive(Debug, thiserror::Error)]
pub enum DocumentError {
    /// Document not found.
    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    /// Storage path does not exist or is not a valid TaluDB.
    #[error("Storage not found: {0}")]
    StorageNotFound(PathBuf),

    /// Invalid argument provided.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    StorageError(String),
}

impl DocumentError {
    fn from_code(code: i32, context: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_default();

        match code {
            ERROR_CODE_STORAGE_ERROR => {
                if detail.contains("not found") {
                    DocumentError::DocumentNotFound(context.to_string())
                } else {
                    DocumentError::StorageError(detail)
                }
            }
            ERROR_CODE_INVALID_ARGUMENT => DocumentError::InvalidArgument(detail),
            _ => DocumentError::StorageError(detail),
        }
    }
}

/// A document record (full details).
#[derive(Debug, Clone)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub doc_type: String,
    pub title: String,
    pub tags_text: Option<String>,
    pub doc_json: String,
    pub parent_id: Option<String>,
    pub marker: Option<String>,
    pub group_id: Option<String>,
    pub owner_id: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub expires_at_ms: i64,
    pub content_hash: u64,
    pub seq_num: u64,
}

/// A document summary (for list results).
#[derive(Debug, Clone)]
pub struct DocumentSummary {
    pub doc_id: String,
    pub doc_type: String,
    pub title: String,
    pub marker: Option<String>,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

/// A search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub doc_id: String,
    pub doc_type: String,
    pub title: String,
    pub snippet: String,
}

/// A change record for CDC.
#[derive(Debug, Clone)]
pub struct ChangeRecord {
    pub seq_num: u64,
    pub doc_id: String,
    pub action: ChangeAction,
    pub timestamp_ms: i64,
    pub doc_type: Option<String>,
    pub title: Option<String>,
}

/// Change action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeAction {
    Create = 1,
    Update = 2,
    Delete = 3,
}

impl From<u8> for ChangeAction {
    fn from(v: u8) -> Self {
        match v {
            1 => ChangeAction::Create,
            2 => ChangeAction::Update,
            3 => ChangeAction::Delete,
            _ => ChangeAction::Update, // Default
        }
    }
}

/// Compaction statistics.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    pub total_documents: usize,
    pub active_documents: usize,
    pub expired_documents: usize,
    pub deleted_documents: usize,
    pub tombstone_count: usize,
    pub delta_versions: usize,
    pub estimated_garbage_bytes: u64,
}

/// Handle for document operations.
#[derive(Debug)]
pub struct DocumentsHandle {
    path: PathBuf,
    path_cstr: CString,
}

impl DocumentsHandle {
    /// Open a TaluDB storage for document operations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DocumentError> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_string_lossy();

        let path_cstr = CString::new(path_str.as_ref())
            .map_err(|_| DocumentError::InvalidArgument("Path contains null bytes".to_string()))?;

        Ok(Self { path, path_cstr })
    }

    /// Get the storage path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    // =========================================================================
    // CRUD Operations
    // =========================================================================

    /// Create a new document.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        doc_id: &str,
        doc_type: &str,
        title: &str,
        doc_json: &str,
        tags_text: Option<&str>,
        parent_id: Option<&str>,
        marker: Option<&str>,
        group_id: Option<&str>,
        owner_id: Option<&str>,
    ) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let doc_type_c = to_cstring(doc_type)?;
        let title_c = to_cstring(title)?;
        let doc_json_c = to_cstring(doc_json)?;
        let tags_c = to_optional_cstring(tags_text)?;
        let parent_c = to_optional_cstring(parent_id)?;
        let marker_c = to_optional_cstring(marker)?;
        let group_c = to_optional_cstring(group_id)?;
        let owner_c = to_optional_cstring(owner_id)?;

        let code = unsafe {
            talu_sys::talu_documents_create(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                doc_type_c.as_ptr(),
                title_c.as_ptr(),
                doc_json_c.as_ptr(),
                opt_ptr(&tags_c),
                opt_ptr(&parent_c),
                opt_ptr(&marker_c),
                opt_ptr(&group_c),
                opt_ptr(&owner_c),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// Get a document by ID.
    pub fn get(&self, doc_id: &str) -> Result<Option<DocumentRecord>, DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;

        let mut out_doc = talu_sys::CDocumentRecord::default();

        let code = unsafe {
            talu_sys::talu_documents_get(self.path_cstr.as_ptr(), doc_id_c.as_ptr(), &mut out_doc)
        };

        if code == 1 {
            // Not found
            return Ok(None);
        }
        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }

        Ok(Some(document_from_c(&out_doc)))
    }

    /// Update an existing document.
    pub fn update(
        &self,
        doc_id: &str,
        title: Option<&str>,
        doc_json: Option<&str>,
        tags_text: Option<&str>,
        marker: Option<&str>,
    ) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let title_c = to_optional_cstring(title)?;
        let doc_json_c = to_optional_cstring(doc_json)?;
        let tags_c = to_optional_cstring(tags_text)?;
        let marker_c = to_optional_cstring(marker)?;

        let code = unsafe {
            talu_sys::talu_documents_update(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                opt_ptr(&title_c),
                opt_ptr(&doc_json_c),
                opt_ptr(&tags_c),
                opt_ptr(&marker_c),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// Delete a document.
    pub fn delete(&self, doc_id: &str) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;

        let code =
            unsafe { talu_sys::talu_documents_delete(self.path_cstr.as_ptr(), doc_id_c.as_ptr()) };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// List documents with optional filters.
    /// Returns summaries (not full records) for efficiency.
    pub fn list(
        &self,
        doc_type: Option<&str>,
        group_id: Option<&str>,
        owner_id: Option<&str>,
        marker: Option<&str>,
        limit: u32,
    ) -> Result<Vec<DocumentSummary>, DocumentError> {
        let type_c = to_optional_cstring(doc_type)?;
        let group_c = to_optional_cstring(group_id)?;
        let owner_c = to_optional_cstring(owner_id)?;
        let marker_c = to_optional_cstring(marker)?;

        let mut out_list: *mut talu_sys::CDocumentList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_list(
                self.path_cstr.as_ptr(),
                opt_ptr(&type_c),
                opt_ptr(&group_c),
                opt_ptr(&owner_c),
                opt_ptr(&marker_c),
                limit,
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "list"));
        }

        let result = if out_list.is_null() {
            Vec::new()
        } else {
            unsafe {
                let list = &*out_list;
                let mut docs = Vec::with_capacity(list.count);
                if !list.items.is_null() {
                    for i in 0..list.count {
                        let item = &*list.items.add(i);
                        docs.push(summary_from_c(item));
                    }
                }
                talu_sys::talu_documents_free_list(out_list);
                docs
            }
        };

        Ok(result)
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Search documents by content.
    pub fn search(
        &self,
        query: &str,
        doc_type: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SearchResult>, DocumentError> {
        let query_c = to_cstring(query)?;
        let type_c = to_optional_cstring(doc_type)?;

        let mut out_list: *mut talu_sys::CSearchResultList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_search(
                self.path_cstr.as_ptr(),
                query_c.as_ptr(),
                opt_ptr(&type_c),
                limit,
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, query));
        }

        let result = if out_list.is_null() {
            Vec::new()
        } else {
            unsafe {
                let list = &*out_list;
                let mut results = Vec::with_capacity(list.count);
                if !list.items.is_null() {
                    for i in 0..list.count {
                        let item = &*list.items.add(i);
                        results.push(SearchResult {
                            doc_id: cstr_to_string(item.doc_id),
                            doc_type: cstr_to_string(item.doc_type),
                            title: cstr_to_string(item.title),
                            snippet: cstr_to_string(item.snippet),
                        });
                    }
                }
                talu_sys::talu_documents_free_search_results(out_list);
                results
            }
        };

        Ok(result)
    }

    // =========================================================================
    // Tag Operations
    // =========================================================================

    /// Add a tag to a document.
    pub fn add_tag(
        &self,
        doc_id: &str,
        tag_id: &str,
        group_id: Option<&str>,
    ) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let tag_id_c = to_cstring(tag_id)?;
        let group_c = to_optional_cstring(group_id)?;

        let code = unsafe {
            talu_sys::talu_documents_add_tag(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                tag_id_c.as_ptr(),
                opt_ptr(&group_c),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// Remove a tag from a document.
    pub fn remove_tag(
        &self,
        doc_id: &str,
        tag_id: &str,
        group_id: Option<&str>,
    ) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let tag_id_c = to_cstring(tag_id)?;
        let group_c = to_optional_cstring(group_id)?;

        let code = unsafe {
            talu_sys::talu_documents_remove_tag(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                tag_id_c.as_ptr(),
                opt_ptr(&group_c),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// Get tags for a document.
    pub fn get_tags(&self, doc_id: &str) -> Result<Vec<String>, DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;

        let mut out_list: *mut talu_sys::CStringList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_get_tags(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }

        let result = extract_string_list(out_list);
        if !out_list.is_null() {
            unsafe { talu_sys::talu_documents_free_string_list(out_list) };
        }

        Ok(result)
    }

    /// Get documents by tag.
    pub fn get_by_tag(&self, tag_id: &str) -> Result<Vec<String>, DocumentError> {
        let tag_id_c = to_cstring(tag_id)?;

        let mut out_list: *mut talu_sys::CStringList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_get_by_tag(
                self.path_cstr.as_ptr(),
                tag_id_c.as_ptr(),
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, tag_id));
        }

        let result = extract_string_list(out_list);
        if !out_list.is_null() {
            unsafe { talu_sys::talu_documents_free_string_list(out_list) };
        }

        Ok(result)
    }

    // =========================================================================
    // TTL Operations
    // =========================================================================

    /// Set TTL for a document.
    /// ttl_seconds: Time-to-live in seconds from now. 0 = remove TTL (never expires).
    pub fn set_ttl(&self, doc_id: &str, ttl_seconds: u64) -> Result<(), DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;

        let code = unsafe {
            talu_sys::talu_documents_set_ttl(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                ttl_seconds,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(())
    }

    /// Count expired documents.
    pub fn count_expired(&self) -> Result<usize, DocumentError> {
        let mut count: usize = 0;

        let code = unsafe {
            talu_sys::talu_documents_count_expired(
                self.path_cstr.as_ptr(),
                &mut count as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "count_expired"));
        }
        Ok(count)
    }

    // =========================================================================
    // CDC Operations
    // =========================================================================

    /// Get changes since a sequence number.
    pub fn get_changes(
        &self,
        since_seq: u64,
        group_id: Option<&str>,
        limit: u32,
    ) -> Result<Vec<ChangeRecord>, DocumentError> {
        let group_c = to_optional_cstring(group_id)?;

        let mut out_list: *mut talu_sys::CChangeList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_get_changes(
                self.path_cstr.as_ptr(),
                since_seq,
                opt_ptr(&group_c),
                limit,
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "get_changes"));
        }

        let result = if out_list.is_null() {
            Vec::new()
        } else {
            unsafe {
                let list = &*out_list;
                let mut changes = Vec::with_capacity(list.count);
                if !list.items.is_null() {
                    for i in 0..list.count {
                        let item = &*list.items.add(i);
                        changes.push(ChangeRecord {
                            seq_num: item.seq_num,
                            doc_id: cstr_to_string(item.doc_id),
                            action: ChangeAction::from(item.action),
                            timestamp_ms: item.timestamp_ms,
                            doc_type: cstr_to_opt_string(item.doc_type),
                            title: cstr_to_opt_string(item.title),
                        });
                    }
                }
                talu_sys::talu_documents_free_changes(out_list);
                changes
            }
        };

        Ok(result)
    }

    // =========================================================================
    // Delta Versioning Operations
    // =========================================================================

    /// Create a delta version of a document.
    #[allow(clippy::too_many_arguments)]
    pub fn create_delta(
        &self,
        base_doc_id: &str,
        new_doc_id: &str,
        delta_json: &str,
        title: Option<&str>,
        tags_text: Option<&str>,
        marker: Option<&str>,
    ) -> Result<(), DocumentError> {
        let base_c = to_cstring(base_doc_id)?;
        let new_c = to_cstring(new_doc_id)?;
        let json_c = to_cstring(delta_json)?;
        let title_c = to_optional_cstring(title)?;
        let tags_c = to_optional_cstring(tags_text)?;
        let marker_c = to_optional_cstring(marker)?;

        let code = unsafe {
            talu_sys::talu_documents_create_delta(
                self.path_cstr.as_ptr(),
                base_c.as_ptr(),
                new_c.as_ptr(),
                json_c.as_ptr(),
                opt_ptr(&title_c),
                opt_ptr(&tags_c),
                opt_ptr(&marker_c),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, base_doc_id));
        }
        Ok(())
    }

    /// Check if a document is a delta version.
    pub fn is_delta(&self, doc_id: &str) -> Result<bool, DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let mut is_delta: bool = false;

        let code = unsafe {
            talu_sys::talu_documents_is_delta(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                &mut is_delta as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }
        Ok(is_delta)
    }

    /// Get the base document ID for a delta.
    /// Returns None if not a delta version.
    pub fn get_base_id(&self, doc_id: &str) -> Result<Option<String>, DocumentError> {
        let doc_id_c = to_cstring(doc_id)?;
        let mut buf = [0u8; 256];

        let code = unsafe {
            talu_sys::talu_documents_get_base_id(
                self.path_cstr.as_ptr(),
                doc_id_c.as_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
            )
        };

        if code == 1 {
            // Not a delta
            return Ok(None);
        }
        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, doc_id));
        }

        let base_id = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .to_string();

        if base_id.is_empty() {
            Ok(None)
        } else {
            Ok(Some(base_id))
        }
    }

    // =========================================================================
    // Compaction Operations
    // =========================================================================

    /// Get compaction statistics.
    pub fn get_compaction_stats(&self) -> Result<CompactionStats, DocumentError> {
        let mut stats = talu_sys::CCompactionStats {
            total_documents: 0,
            active_documents: 0,
            expired_documents: 0,
            deleted_documents: 0,
            tombstone_count: 0,
            delta_versions: 0,
            estimated_garbage_bytes: 0,
        };

        let code = unsafe {
            talu_sys::talu_documents_get_compaction_stats(self.path_cstr.as_ptr(), &mut stats)
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "compaction_stats"));
        }

        Ok(CompactionStats {
            total_documents: stats.total_documents,
            active_documents: stats.active_documents,
            expired_documents: stats.expired_documents,
            deleted_documents: stats.deleted_documents,
            tombstone_count: stats.tombstone_count,
            delta_versions: stats.delta_versions,
            estimated_garbage_bytes: stats.estimated_garbage_bytes,
        })
    }

    /// Purge expired documents (write tombstones).
    /// Returns the number of documents purged.
    pub fn purge_expired(&self) -> Result<usize, DocumentError> {
        let mut count: usize = 0;

        let code = unsafe {
            talu_sys::talu_documents_purge_expired(
                self.path_cstr.as_ptr(),
                &mut count as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "purge_expired"));
        }
        Ok(count)
    }

    /// Get garbage collection candidates.
    pub fn get_garbage_candidates(&self) -> Result<Vec<String>, DocumentError> {
        let mut out_list: *mut talu_sys::CStringList = ptr::null_mut();

        let code = unsafe {
            talu_sys::talu_documents_get_garbage_candidates(
                self.path_cstr.as_ptr(),
                &mut out_list as *mut _ as *mut std::ffi::c_void,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(DocumentError::from_code(code, "garbage_candidates"));
        }

        let result = extract_string_list(out_list);
        if !out_list.is_null() {
            unsafe { talu_sys::talu_documents_free_string_list(out_list) };
        }

        Ok(result)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn to_cstring(s: &str) -> Result<CString, DocumentError> {
    CString::new(s)
        .map_err(|_| DocumentError::InvalidArgument("String contains null bytes".to_string()))
}

fn to_optional_cstring(s: Option<&str>) -> Result<Option<CString>, DocumentError> {
    match s {
        Some(s) => Ok(Some(to_cstring(s)?)),
        None => Ok(None),
    }
}

fn opt_ptr(cs: &Option<CString>) -> *const c_char {
    match cs {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
    }
}

fn cstr_to_opt_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string())
    }
}

fn document_from_c(c: &talu_sys::CDocumentRecord) -> DocumentRecord {
    DocumentRecord {
        doc_id: cstr_to_string(c.doc_id),
        doc_type: cstr_to_string(c.doc_type),
        title: cstr_to_string(c.title),
        tags_text: cstr_to_opt_string(c.tags_text),
        doc_json: cstr_to_string(c.doc_json),
        parent_id: cstr_to_opt_string(c.parent_id),
        marker: cstr_to_opt_string(c.marker),
        group_id: cstr_to_opt_string(c.group_id),
        owner_id: cstr_to_opt_string(c.owner_id),
        created_at_ms: c.created_at_ms,
        updated_at_ms: c.updated_at_ms,
        expires_at_ms: c.expires_at_ms,
        content_hash: c.content_hash,
        seq_num: c.seq_num,
    }
}

fn summary_from_c(c: &talu_sys::CDocumentSummary) -> DocumentSummary {
    DocumentSummary {
        doc_id: cstr_to_string(c.doc_id),
        doc_type: cstr_to_string(c.doc_type),
        title: cstr_to_string(c.title),
        marker: cstr_to_opt_string(c.marker),
        created_at_ms: c.created_at_ms,
        updated_at_ms: c.updated_at_ms,
    }
}

fn extract_string_list(list: *mut talu_sys::CStringList) -> Vec<String> {
    if list.is_null() {
        return Vec::new();
    }

    unsafe {
        let l = &*list;
        let mut result = Vec::with_capacity(l.count);
        if !l.items.is_null() {
            // CStringList.items is a pointer to array of null-terminated strings
            // Cast to correct type: array of c_char pointers
            let items_ptr = l.items as *const *const c_char;
            for i in 0..l.count {
                let item = *items_ptr.add(i);
                if !item.is_null() {
                    result.push(CStr::from_ptr(item).to_string_lossy().to_string());
                }
            }
        }
        result
    }
}
