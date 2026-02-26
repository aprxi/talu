//! Safe wrapper for SQL querying over TaluDB virtual tables.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

use crate::error;

const ERROR_CODE_OK: i32 = 0;

/// Typed SQL parameter for binding to prepared statements.
/// Layout matches CSqlParam in core/src/capi/sql.zig exactly.
#[repr(C)]
#[derive(Clone)]
pub struct SqlParam {
    tag: u8,
    int_val: i64,
    float_val: f64,
    ptr: *const u8,
    len: usize,
}

// SqlParam is Send because the ptr is only read during the FFI call (which is
// synchronous) and the caller guarantees the pointed-to data outlives the call.
unsafe impl Send for SqlParam {}

impl SqlParam {
    /// Create a NULL parameter.
    pub fn null() -> Self {
        Self {
            tag: 0,
            int_val: 0,
            float_val: 0.0,
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Create an integer parameter.
    pub fn int(val: i64) -> Self {
        Self {
            tag: 1,
            int_val: val,
            float_val: 0.0,
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Create a float parameter.
    pub fn float(val: f64) -> Self {
        Self {
            tag: 2,
            int_val: 0,
            float_val: val,
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Create a text parameter. The caller must ensure `text` outlives the FFI call.
    pub fn text(text: &[u8]) -> Self {
        Self {
            tag: 3,
            int_val: 0,
            float_val: 0.0,
            ptr: text.as_ptr(),
            len: text.len(),
        }
    }

    /// Create a blob parameter. The caller must ensure `data` outlives the FFI call.
    pub fn blob(data: &[u8]) -> Self {
        Self {
            tag: 4,
            int_val: 0,
            float_val: 0.0,
            ptr: data.as_ptr(),
            len: data.len(),
        }
    }
}

unsafe extern "C" {
    #[link_name = "talu_db_sql_query"]
    fn talu_sql_query_raw(
        db_path: *const c_char,
        query: *const c_char,
        out_json: *mut *mut c_char,
    ) -> c_int;
    #[link_name = "talu_db_sql_query_params"]
    fn talu_sql_query_params_raw(
        db_path: *const c_char,
        query: *const c_char,
        params: *const SqlParam,
        num_params: u32,
        out_json: *mut *mut c_char,
    ) -> c_int;
    #[link_name = "talu_db_sql_query_free"]
    fn talu_sql_query_free_raw(ptr: *mut c_char);
}

/// SQL query error.
#[derive(Debug, Clone)]
pub struct SqlError {
    pub code: i32,
    pub message: String,
}

impl SqlError {
    fn from_last_or(fallback: &str) -> Self {
        let code = unsafe { talu_sys::talu_last_error_code() };
        let message = error::last_error_message().unwrap_or_else(|| fallback.to_string());
        Self { code, message }
    }
}

impl std::fmt::Display for SqlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sql error (code {}): {}", self.code, self.message)
    }
}

impl std::error::Error for SqlError {}

/// Result alias for SQL operations.
pub type SqlResult<T> = std::result::Result<T, SqlError>;

/// Stateless SQL execution entry point.
pub struct SqlEngine;

impl SqlEngine {
    /// Execute a SQL query without parameters. Returns raw JSON array.
    pub fn query_json(db_root: &str, query: &str) -> SqlResult<String> {
        let c_root = CString::new(db_root)
            .map_err(|_| SqlError::from_last_or("db_root contains null byte"))?;
        let c_query =
            CString::new(query).map_err(|_| SqlError::from_last_or("query contains null byte"))?;

        let mut out_json: *mut c_char = std::ptr::null_mut();
        let rc = unsafe { talu_sql_query_raw(c_root.as_ptr(), c_query.as_ptr(), &mut out_json) };
        if rc != ERROR_CODE_OK {
            return Err(SqlError::from_last_or("failed to execute SQL query"));
        }
        if out_json.is_null() {
            return Err(SqlError::from_last_or(
                "SQL query succeeded but returned null output",
            ));
        }

        let result = unsafe { CStr::from_ptr(out_json) }
            .to_string_lossy()
            .into_owned();
        unsafe { talu_sql_query_free_raw(out_json) };
        Ok(result)
    }

    /// Execute a parameterized SQL query.
    /// Returns structured JSON: {"columns":[...],"rows":[...],"row_count":N}
    pub fn query_params(db_root: &str, query: &str, params: &[SqlParam]) -> SqlResult<String> {
        let c_root = CString::new(db_root)
            .map_err(|_| SqlError::from_last_or("db_root contains null byte"))?;
        let c_query =
            CString::new(query).map_err(|_| SqlError::from_last_or("query contains null byte"))?;

        let params_ptr = if params.is_empty() {
            std::ptr::null()
        } else {
            params.as_ptr()
        };
        let num_params = params.len() as u32;

        let mut out_json: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            talu_sql_query_params_raw(
                c_root.as_ptr(),
                c_query.as_ptr(),
                params_ptr,
                num_params,
                &mut out_json,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(SqlError::from_last_or(
                "failed to execute parameterized SQL query",
            ));
        }
        if out_json.is_null() {
            return Err(SqlError::from_last_or(
                "SQL query succeeded but returned null output",
            ));
        }

        let result = unsafe { CStr::from_ptr(out_json) }
            .to_string_lossy()
            .into_owned();
        unsafe { talu_sql_query_free_raw(out_json) };
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::SqlEngine;

    #[test]
    fn query_json_rejects_nul_db_root() {
        let err = SqlEngine::query_json("bad\0path", "SELECT 1").expect_err("must fail");
        assert!(err.message.contains("db_root contains null byte"));
    }

    #[test]
    fn query_json_rejects_nul_query() {
        let err = SqlEngine::query_json("/tmp", "SELECT\01").expect_err("must fail");
        assert!(err.message.contains("query contains null byte"));
    }
}
