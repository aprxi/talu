//! Safe wrapper for repository pin metadata (repo_meta namespace).

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};

use crate::error;

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_RESOURCE_BUSY: i32 = 701;

unsafe extern "C" {
    #[link_name = "talu_repo_meta_init"]
    fn talu_repo_meta_init_raw(db_path: *const c_char, out_handle: *mut *mut c_void) -> c_int;
    #[link_name = "talu_repo_meta_free"]
    fn talu_repo_meta_free_raw(handle: *mut c_void);
    #[link_name = "talu_repo_meta_pin"]
    fn talu_repo_meta_pin_raw(handle: *mut c_void, model_uri: *const c_char) -> c_int;
    #[link_name = "talu_repo_meta_unpin"]
    fn talu_repo_meta_unpin_raw(handle: *mut c_void, model_uri: *const c_char) -> c_int;
    #[link_name = "talu_repo_meta_update_size"]
    fn talu_repo_meta_update_size_raw(
        handle: *mut c_void,
        model_uri: *const c_char,
        size_bytes: u64,
    ) -> c_int;
    #[link_name = "talu_repo_meta_clear_size"]
    fn talu_repo_meta_clear_size_raw(handle: *mut c_void, model_uri: *const c_char) -> c_int;
    #[link_name = "talu_repo_meta_list_pins"]
    fn talu_repo_meta_list_pins_raw(
        handle: *mut c_void,
        out_list: *mut talu_sys::CPinList,
    ) -> c_int;
    #[link_name = "talu_repo_meta_free_list"]
    fn talu_repo_meta_free_list_raw(list: *mut talu_sys::CPinList);
}

/// Error from repo metadata operations.
#[derive(Debug, Clone)]
pub struct RepoMetaError {
    pub code: i32,
    pub message: String,
}

impl RepoMetaError {
    fn from_last_or(fallback: &str) -> Self {
        let code = unsafe { talu_sys::talu_last_error_code() };
        let message = error::last_error_message().unwrap_or_else(|| fallback.to_string());
        Self { code, message }
    }

    pub fn is_busy(&self) -> bool {
        self.code == ERROR_CODE_RESOURCE_BUSY
    }
}

impl std::fmt::Display for RepoMetaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "repo meta error (code {}): {}", self.code, self.message)
    }
}

impl std::error::Error for RepoMetaError {}

/// Result alias for repo metadata operations.
pub type RepoMetaResult<T> = std::result::Result<T, RepoMetaError>;

/// Pinned model metadata entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoPinEntry {
    pub model_uri: String,
    pub pinned_at_ms: i64,
    pub size_bytes: Option<u64>,
    pub size_updated_at_ms: Option<i64>,
}

/// RAII handle for repo pin metadata.
///
/// Thread safety: NOT thread-safe.
pub struct RepoMetaStore {
    handle: *mut c_void,
}

impl RepoMetaStore {
    /// Open or create repository metadata store rooted at `db_root`.
    pub fn open(db_root: &str) -> RepoMetaResult<Self> {
        let c_root = CString::new(db_root)
            .map_err(|_| RepoMetaError::from_last_or("path contains null byte"))?;
        let mut handle: *mut c_void = std::ptr::null_mut();

        let rc = unsafe { talu_repo_meta_init_raw(c_root.as_ptr(), &mut handle) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or(
                "failed to initialize repo metadata",
            ));
        }
        if handle.is_null() {
            return Err(RepoMetaError::from_last_or(
                "repo metadata init returned null handle",
            ));
        }

        Ok(Self { handle })
    }

    pub fn pin(&self, model_uri: &str) -> RepoMetaResult<()> {
        let c_uri = CString::new(model_uri)
            .map_err(|_| RepoMetaError::from_last_or("model_uri contains null byte"))?;
        let rc = unsafe { talu_repo_meta_pin_raw(self.handle, c_uri.as_ptr()) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or("failed to pin model"));
        }
        Ok(())
    }

    pub fn unpin(&self, model_uri: &str) -> RepoMetaResult<()> {
        let c_uri = CString::new(model_uri)
            .map_err(|_| RepoMetaError::from_last_or("model_uri contains null byte"))?;
        let rc = unsafe { talu_repo_meta_unpin_raw(self.handle, c_uri.as_ptr()) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or("failed to unpin model"));
        }
        Ok(())
    }

    pub fn update_size(&self, model_uri: &str, size_bytes: u64) -> RepoMetaResult<()> {
        let c_uri = CString::new(model_uri)
            .map_err(|_| RepoMetaError::from_last_or("model_uri contains null byte"))?;
        let rc = unsafe { talu_repo_meta_update_size_raw(self.handle, c_uri.as_ptr(), size_bytes) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or("failed to update model size"));
        }
        Ok(())
    }

    pub fn clear_size(&self, model_uri: &str) -> RepoMetaResult<()> {
        let c_uri = CString::new(model_uri)
            .map_err(|_| RepoMetaError::from_last_or("model_uri contains null byte"))?;
        let rc = unsafe { talu_repo_meta_clear_size_raw(self.handle, c_uri.as_ptr()) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or("failed to clear model size"));
        }
        Ok(())
    }

    pub fn list_pins(&self) -> RepoMetaResult<Vec<RepoPinEntry>> {
        let mut list = talu_sys::CPinList::default();
        let rc = unsafe { talu_repo_meta_list_pins_raw(self.handle, &mut list) };
        if rc != ERROR_CODE_OK {
            return Err(RepoMetaError::from_last_or("failed to list pinned models"));
        }
        if list.items.is_null() || list.count == 0 {
            unsafe { talu_repo_meta_free_list_raw(&mut list) };
            return Ok(Vec::new());
        }

        let mut out = Vec::new();
        unsafe {
            let items = std::slice::from_raw_parts(list.items, list.count);
            out.reserve(items.len());
            for item in items {
                let model_uri = if item.model_uri.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(item.model_uri)
                        .to_string_lossy()
                        .into_owned()
                };

                out.push(RepoPinEntry {
                    model_uri,
                    pinned_at_ms: item.pinned_at_ms,
                    size_bytes: if item.has_size_bytes {
                        Some(item.size_bytes)
                    } else {
                        None
                    },
                    size_updated_at_ms: if item.has_size_updated_at_ms {
                        Some(item.size_updated_at_ms)
                    } else {
                        None
                    },
                });
            }
            talu_repo_meta_free_list_raw(&mut list);
        }

        Ok(out)
    }
}

impl Drop for RepoMetaStore {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        unsafe { talu_repo_meta_free_raw(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}
