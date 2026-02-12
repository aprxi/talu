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
#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    /// Invalid argument (dimension mismatch, null path, etc.).
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Storage I/O or corruption error.
    #[error("Vector store error: {0}")]
    StoreError(String),
}

impl VectorError {
    fn from_last(fallback: &str) -> Self {
        let detail = error::last_error_message().unwrap_or_else(|| fallback.to_string());
        VectorError::StoreError(detail)
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
            talu_sys::talu_vector_store_append(
                self.ptr,
                ids.as_ptr() as *mut c_void,
                vectors.as_ptr(),
                ids.len(),
                dims,
            )
        };

        if rc != 0 {
            return Err(VectorError::from_last("failed to append vectors"));
        }
        Ok(())
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
            talu_sys::talu_vector_store_search_batch(
                self.ptr,
                queries.as_ptr(),
                queries.len(),
                dims,
                query_count,
                k,
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
