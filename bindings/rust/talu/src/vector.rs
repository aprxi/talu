//! Vector storage for TaluDB.
//!
//! Provides safe Rust wrappers for the TaluDB vector store — a flat-file
//! storage engine for dense embeddings with brute-force search.
//!
//! # Example
//!
//! ```no_run
//! use talu::vector::VectorStore;
//!
//! let store = VectorStore::open("./taludb")?;
//! store.append(&[1, 2, 3], &[
//!     0.1, 0.2, 0.3, 0.4,  // id=1
//!     0.5, 0.6, 0.7, 0.8,  // id=2
//!     0.9, 1.0, 0.0, 0.1,  // id=3
//! ], 4)?;
//!
//! let results = store.search(&[0.1, 0.2, 0.3, 0.4], 2)?;
//! assert_eq!(results.ids[0], 1);
//! # Ok::<(), talu::vector::VectorError>(())
//! ```

use std::ffi::CString;
use std::os::raw::c_void;

use crate::error;

/// Error types for vector store operations.
#[derive(Debug)]
pub enum VectorError {
    /// Invalid argument (dimension mismatch, null path, etc.).
    InvalidArgument(String),

    /// Storage I/O or corruption error.
    StoreError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::InvalidArgument(s) => write!(f, "Invalid argument: {s}"),
            VectorError::StoreError(s) => write!(f, "Vector store error: {s}"),
        }
    }
}

impl std::error::Error for VectorError {}

impl VectorError {
    fn from_last(fallback: &str) -> Self {
        let code = unsafe { talu_sys::talu_last_error_code() };
        let detail = error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match talu_sys::ErrorCode::from(code) {
            talu_sys::ErrorCode::InvalidArgument => VectorError::InvalidArgument(detail),
            _ => VectorError::StoreError(detail),
        }
    }
}

/// Search results returned by [`VectorStore::search`].
///
/// Contains parallel arrays of IDs and scores, sorted by descending score.
/// All data is Rust-owned; C memory is freed before this struct is returned.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector IDs, ordered by descending similarity.
    pub ids: Vec<u64>,
    /// Similarity scores corresponding to each ID.
    pub scores: Vec<f32>,
}

/// Batch search results returned by [`VectorStore::search_batch`].
///
/// Contains flat arrays of IDs and scores for all queries.
/// For query `q` (0-indexed), results are at indices
/// `[q * count_per_query .. (q+1) * count_per_query]`.
/// All data is Rust-owned; C memory is freed before this struct is returned.
#[derive(Debug, Clone)]
pub struct SearchBatchResult {
    /// Vector IDs for all queries, packed contiguously.
    pub ids: Vec<u64>,
    /// Similarity scores for all queries, packed contiguously.
    pub scores: Vec<f32>,
    /// Number of results per query (may be less than requested k).
    pub count_per_query: u32,
    /// Number of queries.
    pub query_count: u32,
}

/// Loaded vectors returned by [`VectorStore::load`].
///
/// All data is Rust-owned; C memory is freed before this struct is returned.
#[derive(Debug, Clone)]
pub struct LoadResult {
    /// Vector IDs.
    pub ids: Vec<u64>,
    /// Flat array of vector components (length = ids.len() * dims).
    pub vectors: Vec<f32>,
    /// Dimensionality of each vector.
    pub dims: u32,
}

/// Delete summary returned by [`VectorStore::delete`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteResult {
    pub deleted_count: usize,
    pub not_found_count: usize,
}

/// Fetch-by-id result returned by [`VectorStore::fetch`].
#[derive(Debug, Clone)]
pub struct FetchResult {
    pub ids: Vec<u64>,
    pub vectors: Option<Vec<f32>>,
    pub missing_ids: Vec<u64>,
    pub dims: u32,
}

/// State statistics returned by [`VectorStore::stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatsResult {
    pub visible_count: usize,
    pub tombstone_count: usize,
    pub segment_count: usize,
    pub total_count: usize,
    pub manifest_generation: u64,
    pub index_ready_segments: usize,
    pub index_pending_segments: usize,
    pub index_failed_segments: usize,
}

/// Compaction result returned by [`VectorStore::compact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactResult {
    pub kept_count: usize,
    pub removed_tombstones: usize,
}

/// ANN index build summary returned by [`VectorStore::build_indexes_with_generation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexBuildResult {
    pub built_segments: usize,
    pub failed_segments: usize,
    pub pending_segments: usize,
}

/// Vector mutation operation type.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ChangeOp {
    Append = 1,
    Upsert = 2,
    Delete = 3,
    Compact = 4,
}

impl TryFrom<u8> for ChangeOp {
    type Error = VectorError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(ChangeOp::Append),
            2 => Ok(ChangeOp::Upsert),
            3 => Ok(ChangeOp::Delete),
            4 => Ok(ChangeOp::Compact),
            _ => Err(VectorError::StoreError(format!(
                "unknown change op: {value}"
            ))),
        }
    }
}

/// A single vector mutation event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeEvent {
    pub seq: u64,
    pub op: ChangeOp,
    pub id: u64,
    pub timestamp: i64,
}

/// Change feed page returned by [`VectorStore::changes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangesResult {
    pub events: Vec<ChangeEvent>,
    pub has_more: bool,
    pub next_since: u64,
}

/// RAII handle for a TaluDB vector store.
///
/// Thread safety: NOT thread-safe (single-writer semantics).
/// Each `VectorStore` instance must be used from a single thread.
pub struct VectorStore {
    ptr: *mut c_void,
}

impl VectorStore {
    /// Open (or create) a vector store at the given database root.
    ///
    /// The path should be a directory. TaluDB creates the necessary
    /// files inside it.
    pub fn open(db_path: &str) -> Result<Self, VectorError> {
        let path_cstr = CString::new(db_path)
            .map_err(|_| VectorError::InvalidArgument("Path contains null bytes".into()))?;

        let mut ptr: *mut c_void = std::ptr::null_mut();

        // SAFETY: path_cstr is valid, ptr is a valid output location.
        let rc = unsafe {
            talu_sys::talu_vector_store_init(path_cstr.as_ptr(), &mut ptr as *mut _ as *mut c_void)
        };

        if rc != 0 {
            return Err(VectorError::from_last("failed to initialize vector store"));
        }
        if ptr.is_null() {
            return Err(VectorError::StoreError("init returned null handle".into()));
        }

        Ok(Self { ptr })
    }

    /// Append a batch of vectors.
    ///
    /// # Arguments
    ///
    /// * `ids` — Unique identifier per vector.
    /// * `vectors` — Flat f32 array of length `ids.len() * dims`.
    /// * `dims` — Dimensionality of each vector.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::InvalidArgument` if `vectors.len() != ids.len() * dims`.
    pub fn append(&self, ids: &[u64], vectors: &[f32], dims: u32) -> Result<(), VectorError> {
        self.append_with_options(ids, vectors, dims, false, false)
    }

    /// Append with explicit options.
    pub fn append_with_options(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dims: u32,
        normalize: bool,
        reject_existing: bool,
    ) -> Result<(), VectorError> {
        let expected = ids.len() * dims as usize;
        if vectors.len() != expected {
            return Err(VectorError::InvalidArgument(format!(
                "vectors.len() = {} but ids.len() * dims = {} * {} = {}",
                vectors.len(),
                ids.len(),
                dims,
                expected,
            )));
        }

        if ids.is_empty() {
            return Ok(());
        }

        // SAFETY: ptr is valid (created by init, not yet freed).
        // ids and vectors slices are valid for their stated lengths.
        // The C API reads ids as *const u64 and vectors as *const f32.
        let rc = unsafe {
            talu_sys::talu_vector_store_append_ex(
                self.ptr,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len(),
                dims,
                normalize,
                reject_existing,
            )
        };

        if rc != 0 {
            return Err(VectorError::from_last("failed to append vectors"));
        }
        Ok(())
    }

    /// Append with idempotency semantics.
    pub fn append_idempotent_with_options(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dims: u32,
        normalize: bool,
        reject_existing: bool,
        key_hash: u64,
        request_hash: u64,
    ) -> Result<(), VectorError> {
        let expected = ids.len() * dims as usize;
        if vectors.len() != expected {
            return Err(VectorError::InvalidArgument(format!(
                "vectors.len() = {} but ids.len() * dims = {} * {} = {}",
                vectors.len(),
                ids.len(),
                dims,
                expected,
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }

        let rc = unsafe {
            talu_sys::talu_vector_store_append_idempotent_ex(
                self.ptr,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len(),
                dims,
                normalize,
                reject_existing,
                key_hash,
                request_hash,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to append vectors"));
        }
        Ok(())
    }

    /// Upsert a batch of vectors.
    pub fn upsert(&self, ids: &[u64], vectors: &[f32], dims: u32) -> Result<(), VectorError> {
        self.upsert_with_options(ids, vectors, dims, false)
    }

    /// Upsert with explicit options.
    pub fn upsert_with_options(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dims: u32,
        normalize: bool,
    ) -> Result<(), VectorError> {
        let expected = ids.len() * dims as usize;
        if vectors.len() != expected {
            return Err(VectorError::InvalidArgument(format!(
                "vectors.len() = {} but ids.len() * dims = {} * {} = {}",
                vectors.len(),
                ids.len(),
                dims,
                expected,
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }

        let rc = unsafe {
            talu_sys::talu_vector_store_upsert_ex(
                self.ptr,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len(),
                dims,
                normalize,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to upsert vectors"));
        }
        Ok(())
    }

    /// Upsert with idempotency semantics.
    pub fn upsert_idempotent_with_options(
        &self,
        ids: &[u64],
        vectors: &[f32],
        dims: u32,
        normalize: bool,
        key_hash: u64,
        request_hash: u64,
    ) -> Result<(), VectorError> {
        let expected = ids.len() * dims as usize;
        if vectors.len() != expected {
            return Err(VectorError::InvalidArgument(format!(
                "vectors.len() = {} but ids.len() * dims = {} * {} = {}",
                vectors.len(),
                ids.len(),
                dims,
                expected,
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }

        let rc = unsafe {
            talu_sys::talu_vector_store_upsert_idempotent_ex(
                self.ptr,
                ids.as_ptr(),
                vectors.as_ptr(),
                ids.len(),
                dims,
                normalize,
                key_hash,
                request_hash,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to upsert vectors"));
        }
        Ok(())
    }

    /// Tombstone-delete vectors by ID.
    pub fn delete(&self, ids: &[u64]) -> Result<DeleteResult, VectorError> {
        let mut deleted_count: usize = 0;
        let mut not_found_count: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_delete(
                self.ptr,
                ids.as_ptr() as *mut c_void,
                ids.len(),
                &mut deleted_count as *mut usize as *mut c_void,
                &mut not_found_count as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to delete vectors"));
        }

        Ok(DeleteResult {
            deleted_count,
            not_found_count,
        })
    }

    /// Tombstone-delete vectors by ID with idempotency semantics.
    pub fn delete_idempotent(
        &self,
        ids: &[u64],
        key_hash: u64,
        request_hash: u64,
    ) -> Result<DeleteResult, VectorError> {
        let mut deleted_count: usize = 0;
        let mut not_found_count: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_delete_idempotent(
                self.ptr,
                ids.as_ptr(),
                ids.len(),
                key_hash,
                request_hash,
                &mut deleted_count as *mut usize as *mut c_void,
                &mut not_found_count as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to delete vectors"));
        }

        Ok(DeleteResult {
            deleted_count,
            not_found_count,
        })
    }

    /// Fetch current visible vectors by ID.
    pub fn fetch(&self, ids: &[u64], include_values: bool) -> Result<FetchResult, VectorError> {
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_vectors: *mut f32 = std::ptr::null_mut();
        let mut out_found_count: usize = 0;
        let mut out_dims: u32 = 0;
        let mut out_missing_ids: *mut u64 = std::ptr::null_mut();
        let mut out_missing_count: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_fetch(
                self.ptr,
                ids.as_ptr() as *mut c_void,
                ids.len(),
                include_values,
                &mut out_ids as *mut *mut u64 as *mut c_void,
                &mut out_vectors as *mut *mut f32 as *mut c_void,
                &mut out_found_count as *mut usize as *mut c_void,
                &mut out_dims as *mut u32 as *mut c_void,
                &mut out_missing_ids as *mut *mut u64 as *mut c_void,
                &mut out_missing_count as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to fetch vectors"));
        }

        let found_ids = if out_ids.is_null() || out_found_count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(out_ids, out_found_count) }.to_vec()
        };
        let vectors = if include_values {
            let total = out_found_count * out_dims as usize;
            let vec = if out_vectors.is_null() || total == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(out_vectors, total) }.to_vec()
            };
            Some(vec)
        } else {
            None
        };
        let missing_ids = if out_missing_ids.is_null() || out_missing_count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(out_missing_ids, out_missing_count) }.to_vec()
        };

        unsafe {
            talu_sys::talu_vector_store_free_fetch(
                out_ids as *mut c_void,
                out_vectors,
                out_found_count,
                out_dims,
                out_missing_ids as *mut c_void,
                out_missing_count,
            );
        }

        Ok(FetchResult {
            ids: found_ids,
            vectors,
            missing_ids,
            dims: out_dims,
        })
    }

    /// Read persisted vector mutation statistics.
    pub fn stats(&self) -> Result<StatsResult, VectorError> {
        let mut visible_count: usize = 0;
        let mut tombstone_count: usize = 0;
        let mut segment_count: usize = 0;
        let mut total_count: usize = 0;
        let mut manifest_generation: u64 = 0;
        let mut index_ready_segments: usize = 0;
        let mut index_pending_segments: usize = 0;
        let mut index_failed_segments: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_stats(
                self.ptr,
                &mut visible_count as *mut usize as *mut c_void,
                &mut tombstone_count as *mut usize as *mut c_void,
                &mut segment_count as *mut usize as *mut c_void,
                &mut total_count as *mut usize as *mut c_void,
                &mut manifest_generation as *mut u64 as *mut c_void,
                &mut index_ready_segments as *mut usize as *mut c_void,
                &mut index_pending_segments as *mut usize as *mut c_void,
                &mut index_failed_segments as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to read vector stats"));
        }

        Ok(StatsResult {
            visible_count,
            tombstone_count,
            segment_count,
            total_count,
            manifest_generation,
            index_ready_segments,
            index_pending_segments,
            index_failed_segments,
        })
    }

    /// Compact physical vector segments using visible rows only.
    pub fn compact(&self, dims: u32) -> Result<CompactResult, VectorError> {
        let mut kept_count: usize = 0;
        let mut removed_tombstones: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_compact(
                self.ptr,
                dims,
                &mut kept_count as *mut usize as *mut c_void,
                &mut removed_tombstones as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to compact vectors"));
        }

        Ok(CompactResult {
            kept_count,
            removed_tombstones,
        })
    }

    /// Compact physical vector segments with idempotency semantics.
    pub fn compact_idempotent(
        &self,
        dims: u32,
        key_hash: u64,
        request_hash: u64,
    ) -> Result<CompactResult, VectorError> {
        let mut kept_count: usize = 0;
        let mut removed_tombstones: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_compact_idempotent(
                self.ptr,
                dims,
                key_hash,
                request_hash,
                &mut kept_count as *mut usize as *mut c_void,
                &mut removed_tombstones as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to compact vectors"));
        }

        Ok(CompactResult {
            kept_count,
            removed_tombstones,
        })
    }

    /// Read mutation events with cursor pagination.
    pub fn changes(&self, since: u64, limit: usize) -> Result<ChangesResult, VectorError> {
        let mut out_seqs: *mut u64 = std::ptr::null_mut();
        let mut out_ops: *mut u8 = std::ptr::null_mut();
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_timestamps: *mut i64 = std::ptr::null_mut();
        let mut out_count: usize = 0;
        let mut out_has_more: bool = false;
        let mut out_next_since: u64 = since;

        let rc = unsafe {
            talu_sys::talu_vector_store_changes(
                self.ptr,
                since,
                limit,
                &mut out_seqs as *mut *mut u64 as *mut c_void,
                &mut out_ops as *mut *mut u8 as *mut c_void,
                &mut out_ids as *mut *mut u64 as *mut c_void,
                &mut out_timestamps as *mut *mut i64 as *mut c_void,
                &mut out_count as *mut usize as *mut c_void,
                &mut out_has_more as *mut bool as *mut c_void,
                &mut out_next_since as *mut u64 as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last("failed to read vector changes"));
        }

        let mut events = Vec::with_capacity(out_count);
        for idx in 0..out_count {
            let seq = unsafe { *out_seqs.add(idx) };
            let op_raw = unsafe { *out_ops.add(idx) };
            let id = unsafe { *out_ids.add(idx) };
            let timestamp = unsafe { *out_timestamps.add(idx) };
            events.push(ChangeEvent {
                seq,
                op: ChangeOp::try_from(op_raw)?,
                id,
                timestamp,
            });
        }

        unsafe {
            talu_sys::talu_vector_store_free_changes(
                out_seqs as *mut c_void,
                out_ops as *mut c_void,
                out_ids as *mut c_void,
                out_timestamps as *mut c_void,
                out_count,
            );
        }

        Ok(ChangesResult {
            events,
            has_more: out_has_more,
            next_since: out_next_since,
        })
    }

    /// Search for the `k` nearest vectors to `query`.
    ///
    /// Uses dot-product similarity. Results are sorted by descending score.
    ///
    /// # Arguments
    ///
    /// * `query` — Query vector (length must match the stored dimensionality).
    /// * `k` — Maximum number of results to return.
    pub fn search(&self, query: &[f32], k: u32) -> Result<SearchResult, VectorError> {
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_scores: *mut f32 = std::ptr::null_mut();
        let mut out_count: usize = 0;

        // SAFETY: ptr is valid. Output pointers are valid stack locations.
        // The C API writes newly allocated arrays into out_ids/out_scores.
        let rc = unsafe {
            talu_sys::talu_vector_store_search(
                self.ptr,
                query.as_ptr(),
                query.len(),
                k,
                &mut out_ids as *mut *mut u64 as *mut c_void,
                &mut out_scores as *mut *mut f32 as *mut c_void,
                &mut out_count as *mut usize as *mut c_void,
            )
        };

        if rc != 0 {
            return Err(VectorError::from_last("search failed"));
        }

        // Copy into Rust-owned Vecs, then free C memory immediately.
        let ids = if out_ids.is_null() || out_count == 0 {
            Vec::new()
        } else {
            // SAFETY: out_ids points to out_count u64s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_ids, out_count) }.to_vec()
        };

        let scores = if out_scores.is_null() || out_count == 0 {
            Vec::new()
        } else {
            // SAFETY: out_scores points to out_count f32s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_scores, out_count) }.to_vec()
        };

        // SAFETY: Freeing buffers allocated by talu_vector_store_search.
        unsafe {
            talu_sys::talu_vector_store_free_search(out_ids as *mut c_void, out_scores, out_count);
        }

        Ok(SearchResult { ids, scores })
    }

    /// Search for the `k` nearest vectors to each query in a batch.
    ///
    /// Processes all queries in a single pass (SIMD-optimized in Core).
    /// Results are packed contiguously: query 0's results first, then query 1, etc.
    ///
    /// # Arguments
    ///
    /// * `queries` — Flat f32 array of all query vectors (`queries.len() == dims * query_count`).
    /// * `dims` — Dimensionality of each query vector.
    /// * `k` — Maximum results per query.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::InvalidArgument` if `queries.len()` is not a multiple of `dims`.
    pub fn search_batch(
        &self,
        queries: &[f32],
        dims: u32,
        k: u32,
    ) -> Result<SearchBatchResult, VectorError> {
        self.search_batch_with_options(queries, dims, k, false, false)
    }

    /// Search batch with explicit options.
    pub fn search_batch_with_options(
        &self,
        queries: &[f32],
        dims: u32,
        k: u32,
        normalize_queries: bool,
        approximate: bool,
    ) -> Result<SearchBatchResult, VectorError> {
        if dims == 0 {
            return Err(VectorError::InvalidArgument("dims must be > 0".into()));
        }
        if queries.len() % dims as usize != 0 {
            return Err(VectorError::InvalidArgument(format!(
                "queries.len() = {} is not a multiple of dims = {}",
                queries.len(),
                dims,
            )));
        }
        let query_count = (queries.len() / dims as usize) as u32;

        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_scores: *mut f32 = std::ptr::null_mut();
        let mut out_count_per_query: u32 = 0;

        // SAFETY: ptr is valid. Output pointers are valid stack locations.
        let rc = unsafe {
            talu_sys::talu_vector_store_search_batch_ex(
                self.ptr,
                queries.as_ptr(),
                queries.len(),
                dims,
                query_count,
                k,
                normalize_queries,
                approximate,
                &mut out_ids as *mut *mut u64 as *mut c_void,
                &mut out_scores as *mut *mut f32 as *mut c_void,
                &mut out_count_per_query as *mut u32 as *mut c_void,
            )
        };

        if rc != 0 {
            return Err(VectorError::from_last("search_batch failed"));
        }

        let total = out_count_per_query as usize * query_count as usize;

        let ids = if out_ids.is_null() || total == 0 {
            Vec::new()
        } else {
            // SAFETY: out_ids points to `total` u64s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_ids, total) }.to_vec()
        };

        let scores = if out_scores.is_null() || total == 0 {
            Vec::new()
        } else {
            // SAFETY: out_scores points to `total` f32s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_scores, total) }.to_vec()
        };

        // SAFETY: Freeing buffers allocated by talu_vector_store_search_batch.
        unsafe {
            talu_sys::talu_vector_store_free_search_batch(
                out_ids as *mut c_void,
                out_scores,
                out_count_per_query,
                query_count,
            );
        }

        Ok(SearchBatchResult {
            ids,
            scores,
            count_per_query: out_count_per_query,
            query_count,
        })
    }

    /// Compact vectors only if manifest generation matches `expected_generation`.
    pub fn compact_with_generation(
        &self,
        dims: u32,
        expected_generation: u64,
    ) -> Result<CompactResult, VectorError> {
        let mut kept_count: usize = 0;
        let mut removed_tombstones: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_compact_with_generation(
                self.ptr,
                dims,
                expected_generation,
                &mut kept_count as *mut usize as *mut c_void,
                &mut removed_tombstones as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last(
                "failed to compact vectors with generation",
            ));
        }
        Ok(CompactResult {
            kept_count,
            removed_tombstones,
        })
    }

    /// Build pending ANN indexes only if manifest generation matches `expected_generation`.
    pub fn build_indexes_with_generation(
        &self,
        expected_generation: u64,
        max_segments: usize,
    ) -> Result<IndexBuildResult, VectorError> {
        let mut built_segments: usize = 0;
        let mut failed_segments: usize = 0;
        let mut pending_segments: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_build_indexes_with_generation(
                self.ptr,
                expected_generation,
                max_segments,
                &mut built_segments as *mut usize as *mut c_void,
                &mut failed_segments as *mut usize as *mut c_void,
                &mut pending_segments as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last(
                "failed to build vector indexes with generation",
            ));
        }
        Ok(IndexBuildResult {
            built_segments,
            failed_segments,
            pending_segments,
        })
    }

    /// Compact vectors when tombstones older than `max_age_ms` exist.
    pub fn compact_expired_tombstones(
        &self,
        dims: u32,
        now_ms: i64,
        max_age_ms: i64,
    ) -> Result<CompactResult, VectorError> {
        let mut kept_count: usize = 0;
        let mut removed_tombstones: usize = 0;

        let rc = unsafe {
            talu_sys::talu_vector_store_compact_expired_tombstones(
                self.ptr,
                dims,
                now_ms,
                max_age_ms,
                &mut kept_count as *mut usize as *mut c_void,
                &mut removed_tombstones as *mut usize as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(VectorError::from_last(
                "failed to compact expired tombstones",
            ));
        }
        Ok(CompactResult {
            kept_count,
            removed_tombstones,
        })
    }

    /// Load all vectors from the store.
    ///
    /// Returns owned copies of all IDs and vector data.
    pub fn load(&self) -> Result<LoadResult, VectorError> {
        let mut out_ids: *mut u64 = std::ptr::null_mut();
        let mut out_vectors: *mut f32 = std::ptr::null_mut();
        let mut out_count: usize = 0;
        let mut out_dims: u32 = 0;

        // SAFETY: ptr is valid. Output pointers are valid stack locations.
        let rc = unsafe {
            talu_sys::talu_vector_store_load(
                self.ptr,
                &mut out_ids as *mut *mut u64 as *mut c_void,
                &mut out_vectors as *mut *mut f32 as *mut c_void,
                &mut out_count as *mut usize as *mut c_void,
                &mut out_dims as *mut u32 as *mut c_void,
            )
        };

        if rc != 0 {
            return Err(VectorError::from_last("load failed"));
        }

        let total_floats = out_count * out_dims as usize;

        let ids = if out_ids.is_null() || out_count == 0 {
            Vec::new()
        } else {
            // SAFETY: out_ids points to out_count u64s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_ids, out_count) }.to_vec()
        };

        let vectors = if out_vectors.is_null() || total_floats == 0 {
            Vec::new()
        } else {
            // SAFETY: out_vectors points to total_floats f32s allocated by Core.
            unsafe { std::slice::from_raw_parts(out_vectors, total_floats) }.to_vec()
        };

        // SAFETY: Freeing buffers allocated by talu_vector_store_load.
        unsafe {
            talu_sys::talu_vector_store_free_load(
                out_ids as *mut c_void,
                out_vectors,
                out_count,
                out_dims,
            );
        }

        Ok(LoadResult {
            ids,
            vectors,
            dims: out_dims,
        })
    }

    /// Simulates a process crash for testing.
    ///
    /// Closes all file descriptors (releasing flocks) WITHOUT flushing
    /// pending data or deleting the WAL file. After calling this, the
    /// store handle is invalidated and Drop becomes a no-op.
    pub fn simulate_crash(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr is non-null (checked above) and was created by talu_vector_store_init.
            unsafe { talu_sys::talu_vector_store_simulate_crash(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }

    /// Sets the write durability mode.
    ///
    /// Controls whether writes are fsynced to disk on every append
    /// (full durability) or buffered in the OS page cache (async durability).
    pub fn set_durability(&self, mode: crate::Durability) -> std::result::Result<(), VectorError> {
        // SAFETY: ptr is valid (created by talu_vector_store_init, not yet freed).
        let rc = unsafe { talu_sys::talu_vector_store_set_durability(self.ptr, mode as u8) };
        if rc != 0 {
            return Err(VectorError::from_last("failed to set durability"));
        }
        Ok(())
    }
}

impl Drop for VectorStore {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr was created by talu_vector_store_init and not yet freed.
            unsafe { talu_sys::talu_vector_store_free(self.ptr) };
        }
    }
}
