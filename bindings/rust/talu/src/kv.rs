//! Generic KV-plane wrappers for `talu_db_kv_*`.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_RESOURCE_BUSY: i32 = 701;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;

unsafe extern "C" {
    #[link_name = "talu_db_kv_init"]
    fn talu_db_kv_init_raw(
        db_path: *const c_char,
        namespace: *const c_char,
        out_handle: *mut *mut c_void,
    ) -> c_int;
    #[link_name = "talu_db_kv_free"]
    fn talu_db_kv_free_raw(handle: *mut c_void);
    #[link_name = "talu_db_kv_put"]
    fn talu_db_kv_put_raw(
        handle: *mut c_void,
        key: *const c_char,
        value: *const u8,
        value_len: usize,
    ) -> c_int;
    #[link_name = "talu_db_kv_get"]
    fn talu_db_kv_get_raw(
        handle: *mut c_void,
        key: *const c_char,
        out_value: *mut talu_sys::CKvValue,
    ) -> c_int;
    #[link_name = "talu_db_kv_delete"]
    fn talu_db_kv_delete_raw(
        handle: *mut c_void,
        key: *const c_char,
        out_deleted: *mut bool,
    ) -> c_int;
    #[link_name = "talu_db_kv_list"]
    fn talu_db_kv_list_raw(handle: *mut c_void, out_list: *mut talu_sys::CKvList) -> c_int;
    #[link_name = "talu_db_kv_free_value"]
    fn talu_db_kv_free_value_raw(value: *mut talu_sys::CKvValue);
    #[link_name = "talu_db_kv_free_list"]
    fn talu_db_kv_free_list_raw(list: *mut talu_sys::CKvList);
    #[link_name = "talu_db_kv_flush"]
    fn talu_db_kv_flush_raw(handle: *mut c_void) -> c_int;
    #[link_name = "talu_db_kv_compact"]
    fn talu_db_kv_compact_raw(handle: *mut c_void) -> c_int;
}

/// Error type for generic KV operations.
#[derive(Debug, Clone)]
pub enum KvError {
    InvalidArgument(String),
    Busy(String),
    Storage(String),
}

impl KvError {
    fn from_code(code: i32, fallback: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match code {
            ERROR_CODE_INVALID_ARGUMENT => Self::InvalidArgument(detail),
            ERROR_CODE_RESOURCE_BUSY => Self::Busy(detail),
            _ => Self::Storage(detail),
        }
    }
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::Busy(msg) => write!(f, "Resource busy: {msg}"),
            Self::Storage(msg) => write!(f, "Storage error: {msg}"),
        }
    }
}

impl std::error::Error for KvError {}

/// Value returned by KV get operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvValue {
    pub data: Vec<u8>,
    pub updated_at_ms: i64,
}

/// Entry returned by KV list operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub updated_at_ms: i64,
}

/// RAII handle for one KV namespace.
#[derive(Debug)]
pub struct KvHandle {
    handle: *mut c_void,
}

impl KvHandle {
    /// Open or create a KV namespace rooted at `db_root`.
    pub fn open(db_root: &str, namespace: &str) -> Result<Self, KvError> {
        let c_root = CString::new(db_root)
            .map_err(|_| KvError::InvalidArgument("db_root contains null byte".to_string()))?;
        let c_namespace = CString::new(namespace)
            .map_err(|_| KvError::InvalidArgument("namespace contains null byte".to_string()))?;

        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: pointers are valid for this call, and `handle` is writable.
        let rc = unsafe { talu_db_kv_init_raw(c_root.as_ptr(), c_namespace.as_ptr(), &mut handle) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to initialize kv namespace"));
        }
        if handle.is_null() {
            return Err(KvError::Storage(
                "kv init succeeded but returned null handle".to_string(),
            ));
        }
        Ok(Self { handle })
    }

    /// Put `value` for `key` (upsert).
    pub fn put(&self, key: &str, value: &[u8]) -> Result<(), KvError> {
        let c_key = CString::new(key)
            .map_err(|_| KvError::InvalidArgument("key contains null byte".to_string()))?;
        let value_ptr = if value.is_empty() {
            std::ptr::null()
        } else {
            value.as_ptr()
        };
        // SAFETY: handle is valid; key is valid C string; value pointer matches len.
        let rc = unsafe { talu_db_kv_put_raw(self.handle, c_key.as_ptr(), value_ptr, value.len()) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to put kv entry"));
        }
        Ok(())
    }

    /// Get value for `key`.
    pub fn get(&self, key: &str) -> Result<Option<KvValue>, KvError> {
        let c_key = CString::new(key)
            .map_err(|_| KvError::InvalidArgument("key contains null byte".to_string()))?;
        let mut out = talu_sys::CKvValue::default();
        // SAFETY: handle and key are valid; out points to writable memory.
        let rc = unsafe { talu_db_kv_get_raw(self.handle, c_key.as_ptr(), &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to get kv entry"));
        }

        let result = if !out.found {
            None
        } else {
            if out.data.is_null() && out.len > 0 {
                // SAFETY: `out` originates from kv_get and can be freed here.
                unsafe { talu_db_kv_free_value_raw(&mut out) };
                return Err(KvError::Storage(
                    "kv get returned invalid value buffer".to_string(),
                ));
            }
            let data = if out.len == 0 {
                Vec::new()
            } else {
                // SAFETY: buffer is valid for `len` bytes until free_value.
                unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec()
            };
            Some(KvValue {
                data,
                updated_at_ms: out.updated_at_ms,
            })
        };

        // SAFETY: `out` was returned by kv_get.
        unsafe { talu_db_kv_free_value_raw(&mut out) };
        Ok(result)
    }

    /// Delete `key`. Returns whether it existed.
    pub fn delete(&self, key: &str) -> Result<bool, KvError> {
        let c_key = CString::new(key)
            .map_err(|_| KvError::InvalidArgument("key contains null byte".to_string()))?;
        let mut deleted = false;
        // SAFETY: handle and key are valid; deleted points to writable output memory.
        let rc = unsafe { talu_db_kv_delete_raw(self.handle, c_key.as_ptr(), &mut deleted) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to delete kv entry"));
        }
        Ok(deleted)
    }

    /// List all entries in key-sorted order.
    pub fn list(&self) -> Result<Vec<KvEntry>, KvError> {
        let mut out = talu_sys::CKvList::default();
        // SAFETY: handle is valid; out points to writable output memory.
        let rc = unsafe { talu_db_kv_list_raw(self.handle, &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to list kv entries"));
        }

        let mut entries = Vec::with_capacity(out.count);
        if !out.items.is_null() && out.count > 0 {
            // SAFETY: items points to `count` entries until free_list.
            let items = unsafe { std::slice::from_raw_parts(out.items, out.count) };
            for item in items {
                let key = if item.key.is_null() {
                    String::new()
                } else {
                    // SAFETY: key points to a valid null-terminated string while list is alive.
                    unsafe { CStr::from_ptr(item.key as *const c_char) }
                        .to_string_lossy()
                        .into_owned()
                };

                let value = if item.value.is_null() || item.value_len == 0 {
                    Vec::new()
                } else {
                    // SAFETY: value points to `value_len` bytes while list is alive.
                    unsafe { std::slice::from_raw_parts(item.value, item.value_len) }.to_vec()
                };

                entries.push(KvEntry {
                    key,
                    value,
                    updated_at_ms: item.updated_at_ms,
                });
            }
        }

        // SAFETY: `out` was returned by kv_list.
        unsafe { talu_db_kv_free_list_raw(&mut out) };
        Ok(entries)
    }

    /// Flush the namespace.
    pub fn flush(&self) -> Result<(), KvError> {
        // SAFETY: handle is valid while `self` is alive.
        let rc = unsafe { talu_db_kv_flush_raw(self.handle) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to flush kv namespace"));
        }
        Ok(())
    }

    /// Compact the namespace.
    pub fn compact(&self) -> Result<(), KvError> {
        // SAFETY: handle is valid while `self` is alive.
        let rc = unsafe { talu_db_kv_compact_raw(self.handle) };
        if rc != ERROR_CODE_OK {
            return Err(KvError::from_code(rc, "failed to compact kv namespace"));
        }
        Ok(())
    }
}

impl Drop for KvHandle {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: handle was returned by `talu_db_kv_init` and is owned by this struct.
        unsafe { talu_db_kv_free_raw(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::KvHandle;

    #[test]
    fn kv_roundtrip_put_get_list_delete() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let root = tmp.path().to_string_lossy().to_string();

        let kv = KvHandle::open(&root, "test_ns").expect("open kv");
        kv.put("alpha", b"one").expect("put alpha");
        kv.put("beta", b"two").expect("put beta");

        let alpha = kv.get("alpha").expect("get alpha").expect("alpha present");
        assert_eq!(alpha.data, b"one");
        assert!(alpha.updated_at_ms > 0);

        let entries = kv.list().expect("list");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, "alpha");
        assert_eq!(entries[0].value, b"one");
        assert_eq!(entries[1].key, "beta");
        assert_eq!(entries[1].value, b"two");

        assert!(kv.delete("alpha").expect("delete alpha"));
        assert!(!kv.delete("alpha").expect("delete alpha again"));
        assert!(kv.get("alpha").expect("get missing").is_none());
    }

    #[test]
    fn kv_open_rejects_nul_namespace() {
        let err = KvHandle::open("/tmp", "bad\0ns").expect_err("expected invalid namespace");
        assert!(err.to_string().contains("null byte"));
    }
}
