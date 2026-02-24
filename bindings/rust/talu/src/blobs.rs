//! Raw blob storage wrappers for TaluDB content-addressable storage.
//!
//! Provides safe Rust access to `talu_db_blob_*` C-API functions.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

/// Error codes from the C API (must match error_codes.zig).
const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_IO_FILE_NOT_FOUND: i32 = 500;
const ERROR_CODE_IO_PERMISSION_DENIED: i32 = 501;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;
const ERROR_CODE_RESOURCE_EXHAUSTED: i32 = 905;
const DEFAULT_GC_MIN_BLOB_AGE_SECONDS: u64 = 15 * 60;

/// Error type for blob operations.
#[derive(Debug)]
pub enum BlobError {
    InvalidArgument(String),
    NotFound(String),
    PermissionDenied(String),
    ResourceExhausted(String),
    StorageError(String),
}

/// Blob mark-and-sweep statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobGcStats {
    pub referenced_blob_count: usize,
    pub total_blob_files: usize,
    pub deleted_blob_files: usize,
    pub reclaimed_bytes: u64,
    pub invalid_reference_count: usize,
    pub skipped_invalid_entries: usize,
    pub skipped_recent_blob_files: usize,
}

impl From<talu_sys::BlobGcStats> for BlobGcStats {
    fn from(value: talu_sys::BlobGcStats) -> Self {
        Self {
            referenced_blob_count: value.referenced_blob_count,
            total_blob_files: value.total_blob_files,
            deleted_blob_files: value.deleted_blob_files,
            reclaimed_bytes: value.reclaimed_bytes,
            invalid_reference_count: value.invalid_reference_count,
            skipped_invalid_entries: value.skipped_invalid_entries,
            skipped_recent_blob_files: value.skipped_recent_blob_files,
        }
    }
}

impl std::fmt::Display for BlobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobError::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            BlobError::NotFound(msg) => write!(f, "Blob not found: {msg}"),
            BlobError::PermissionDenied(msg) => write!(f, "Permission denied: {msg}"),
            BlobError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            BlobError::StorageError(msg) => write!(f, "Storage error: {msg}"),
        }
    }
}

impl std::error::Error for BlobError {}

impl BlobError {
    fn from_code(code: i32, fallback_context: &str) -> Self {
        let detail =
            crate::error::last_error_message().unwrap_or_else(|| fallback_context.to_string());
        match code {
            ERROR_CODE_INVALID_ARGUMENT => BlobError::InvalidArgument(detail),
            ERROR_CODE_IO_FILE_NOT_FOUND => BlobError::NotFound(detail),
            ERROR_CODE_IO_PERMISSION_DENIED => BlobError::PermissionDenied(detail),
            ERROR_CODE_RESOURCE_EXHAUSTED => BlobError::ResourceExhausted(detail),
            _ => BlobError::StorageError(detail),
        }
    }
}

/// Handle for blob storage operations.
#[derive(Debug)]
pub struct BlobsHandle {
    path: PathBuf,
    path_cstr: CString,
}

impl BlobsHandle {
    /// Open a blob handle for a TaluDB bucket path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, BlobError> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_string_lossy();
        let path_cstr = CString::new(path_str.as_ref())
            .map_err(|_| BlobError::InvalidArgument("Path contains null bytes".to_string()))?;
        Ok(Self { path, path_cstr })
    }

    /// Get the storage path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Store bytes as content-addressed blob and return `sha256:<hex>` or `multi:<hex>`.
    pub fn put(&self, bytes: &[u8]) -> Result<String, BlobError> {
        let mut blob_ref_buf = [0u8; 129];
        let bytes_ptr = if bytes.is_empty() {
            ptr::null()
        } else {
            bytes.as_ptr()
        };

        // SAFETY: `path_cstr` and `blob_ref_buf` are valid for the duration of the call.
        // `bytes_ptr` is either null with len=0 or points to `bytes` memory with `bytes.len()`.
        let code = unsafe {
            talu_sys::talu_db_blob_put(
                self.path_cstr.as_ptr(),
                bytes_ptr,
                bytes.len(),
                blob_ref_buf.as_mut_ptr() as *const u8,
                blob_ref_buf.len(),
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob put failed"));
        }

        // SAFETY: C API guarantees NUL-terminated output on success.
        let blob_ref = unsafe { CStr::from_ptr(blob_ref_buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .to_string();
        if blob_ref.is_empty() {
            return Err(BlobError::StorageError(
                "blob put succeeded but returned empty reference".to_string(),
            ));
        }
        Ok(blob_ref)
    }

    /// Check if a blob reference currently resolves to stored bytes.
    pub fn contains(&self, blob_ref: &str) -> Result<bool, BlobError> {
        let blob_ref_c = CString::new(blob_ref)
            .map_err(|_| BlobError::InvalidArgument("blob_ref contains null bytes".to_string()))?;
        let mut out_exists = false;

        // SAFETY: `path_cstr` and `blob_ref_c` are valid C strings, and `out_exists`
        // points to writable memory for the output flag.
        let code = unsafe {
            talu_sys::talu_db_blob_exists(
                self.path_cstr.as_ptr(),
                blob_ref_c.as_ptr(),
                &mut out_exists as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob exists check failed"));
        }
        Ok(out_exists)
    }

    /// List stored physical blob references (`sha256:<hex>`).
    ///
    /// `limit=None` means no limit.
    pub fn list(&self, limit: Option<usize>) -> Result<Vec<String>, BlobError> {
        let mut out_list: *mut talu_sys::CStringList = ptr::null_mut();
        let limit = limit.unwrap_or(0);

        // SAFETY: `path_cstr` is valid and `out_list` points to writable memory.
        let code = unsafe {
            talu_sys::talu_db_blob_list(self.path_cstr.as_ptr(), limit, &mut out_list as *mut _)
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob list failed"));
        }

        if out_list.is_null() {
            return Ok(Vec::new());
        }
        let refs = extract_string_list(out_list);
        // SAFETY: list handle is owned by this function on success.
        unsafe { talu_sys::talu_db_blob_free_string_list(out_list) };
        Ok(refs)
    }

    /// Run blob mark-and-sweep using the default GC grace period.
    pub fn gc(&self) -> Result<BlobGcStats, BlobError> {
        self.gc_with_min_age(DEFAULT_GC_MIN_BLOB_AGE_SECONDS)
    }

    /// Run blob mark-and-sweep with an explicit grace period.
    pub fn gc_with_min_age(&self, min_blob_age_seconds: u64) -> Result<BlobGcStats, BlobError> {
        let mut stats = talu_sys::BlobGcStats::default();

        // SAFETY: `path_cstr` is a valid C string and `stats` points to writable output memory.
        let code = unsafe {
            talu_sys::talu_db_blob_gc(
                self.path_cstr.as_ptr(),
                min_blob_age_seconds,
                &mut stats as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob gc failed"));
        }
        Ok(stats.into())
    }

    /// Open a streaming writer for incremental blob uploads.
    pub fn open_write_stream(&self) -> Result<BlobWriteStream, BlobError> {
        let mut stream_handle: *mut c_void = ptr::null_mut();

        // SAFETY: `path_cstr` is a valid C string and `stream_handle` points to writable memory.
        let code = unsafe {
            talu_sys::talu_db_blob_open_write_stream(
                self.path_cstr.as_ptr(),
                &mut stream_handle as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob write stream open failed"));
        }
        if stream_handle.is_null() {
            return Err(BlobError::StorageError(
                "blob write stream open succeeded but returned null handle".to_string(),
            ));
        }
        Ok(BlobWriteStream {
            handle: stream_handle,
        })
    }

    /// Open a streaming reader for a blob reference.
    pub fn open_stream(&self, blob_ref: &str) -> Result<BlobReadStream, BlobError> {
        let blob_ref_c = CString::new(blob_ref)
            .map_err(|_| BlobError::InvalidArgument("blob_ref contains null bytes".to_string()))?;
        let mut stream_handle: *mut c_void = ptr::null_mut();

        // SAFETY: `path_cstr` and `blob_ref_c` are valid C strings, and `stream_handle`
        // points to writable memory for the output handle.
        let code = unsafe {
            talu_sys::talu_db_blob_open_stream(
                self.path_cstr.as_ptr(),
                blob_ref_c.as_ptr(),
                &mut stream_handle as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, blob_ref));
        }
        if stream_handle.is_null() {
            return Err(BlobError::StorageError(
                "blob stream open succeeded but returned null handle".to_string(),
            ));
        }

        Ok(BlobReadStream {
            handle: stream_handle,
        })
    }

    /// Convenience helper that reads an entire blob by streaming.
    pub fn read_all(&self, blob_ref: &str) -> Result<Vec<u8>, BlobError> {
        let mut stream = self.open_stream(blob_ref)?;
        let size = stream.total_size().unwrap_or(0);
        let mut out = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
        let mut buf = [0u8; 64 * 1024];

        loop {
            let read_len = stream.read(&mut buf)?;
            if read_len == 0 {
                break;
            }
            out.extend_from_slice(&buf[..read_len]);
        }
        Ok(out)
    }
}

/// Owned blob stream handle.
#[derive(Debug)]
pub struct BlobReadStream {
    handle: *mut c_void,
}

// SAFETY: The handle is uniquely owned by `BlobReadStream`, all operations require `&mut self`,
// and drop closes the underlying stream exactly once. Moving ownership across threads is safe.
unsafe impl Send for BlobReadStream {}

impl BlobReadStream {
    /// Read bytes into `buffer`. Returns 0 on EOF.
    pub fn read(&mut self, buffer: &mut [u8]) -> Result<usize, BlobError> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let mut out_read: usize = 0;

        // SAFETY: `self.handle` is a valid handle created by `talu_db_blob_open_stream`.
        // `buffer` is writable and alive for the call.
        let code = unsafe {
            talu_sys::talu_db_blob_stream_read(
                self.handle,
                buffer.as_mut_ptr() as *const u8,
                buffer.len(),
                &mut out_read as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob stream read failed"));
        }
        Ok(out_read)
    }

    /// Return the total blob size in bytes.
    pub fn total_size(&self) -> Result<u64, BlobError> {
        let mut size: u64 = 0;
        // SAFETY: `self.handle` is valid and `size` points to writable memory.
        let code = unsafe {
            talu_sys::talu_db_blob_stream_total_size(
                self.handle,
                &mut size as *mut _ as *mut c_void,
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob stream total_size failed"));
        }
        Ok(size)
    }

    /// Seek to an absolute byte offset from stream start.
    pub fn seek(&mut self, offset_bytes: u64) -> Result<(), BlobError> {
        // SAFETY: `self.handle` is valid and owned by this stream.
        let code = unsafe { talu_sys::talu_db_blob_stream_seek(self.handle, offset_bytes) };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob stream seek failed"));
        }
        Ok(())
    }
}

impl Drop for BlobReadStream {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` was returned by `talu_db_blob_open_stream` and is owned by this struct.
        unsafe { talu_sys::talu_db_blob_stream_close(self.handle) };
        self.handle = ptr::null_mut();
    }
}

/// Owned blob write stream handle.
#[derive(Debug)]
pub struct BlobWriteStream {
    handle: *mut c_void,
}

// SAFETY: The handle is uniquely owned by `BlobWriteStream`, all operations require `&mut self`,
// and drop closes the underlying stream exactly once. Moving ownership across threads is safe.
unsafe impl Send for BlobWriteStream {}

impl BlobWriteStream {
    /// Write a chunk to the blob stream.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), BlobError> {
        let bytes_ptr = if bytes.is_empty() {
            ptr::null()
        } else {
            bytes.as_ptr()
        };
        // SAFETY: `handle` is valid and `bytes_ptr` is null for len=0 or points to `bytes`.
        let code = unsafe {
            talu_sys::talu_db_blob_write_stream_write(self.handle, bytes_ptr, bytes.len())
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(code, "blob write stream write failed"));
        }
        Ok(())
    }

    /// Finalize stream and return content-addressed blob reference.
    ///
    /// This consumes the stream so callers cannot write after finish.
    pub fn finish(self) -> Result<String, BlobError> {
        let mut blob_ref_buf = [0u8; 129];
        // SAFETY: `handle` is valid and output buffer is writable.
        let code = unsafe {
            talu_sys::talu_db_blob_write_stream_finish(
                self.handle,
                blob_ref_buf.as_mut_ptr() as *const u8,
                blob_ref_buf.len(),
            )
        };
        if code != ERROR_CODE_OK {
            return Err(BlobError::from_code(
                code,
                "blob write stream finish failed",
            ));
        }
        // SAFETY: C API guarantees NUL-terminated output on success.
        let blob_ref = unsafe { CStr::from_ptr(blob_ref_buf.as_ptr() as *const c_char) }
            .to_string_lossy()
            .to_string();
        if blob_ref.is_empty() {
            return Err(BlobError::StorageError(
                "blob write stream finish returned empty reference".to_string(),
            ));
        }
        Ok(blob_ref)
    }
}

impl Drop for BlobWriteStream {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `handle` was returned by `talu_db_blob_open_write_stream` and is owned here.
        unsafe { talu_sys::talu_db_blob_write_stream_close(self.handle) };
        self.handle = ptr::null_mut();
    }
}

fn extract_string_list(list: *mut talu_sys::CStringList) -> Vec<String> {
    if list.is_null() {
        return Vec::new();
    }

    unsafe {
        let l = &*list;
        let mut out = Vec::with_capacity(l.count);
        if !l.items.is_null() {
            let items_ptr = l.items;
            for i in 0..l.count {
                let item = *items_ptr.add(i);
                if !item.is_null() {
                    out.push(CStr::from_ptr(item).to_string_lossy().to_string());
                }
            }
        }
        out
    }
}
