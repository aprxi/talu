//! Safe wrapper for SQL querying over TaluDB virtual tables.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

use crate::error;

const ERROR_CODE_OK: i32 = 0;

unsafe extern "C" {
    #[link_name = "talu_db_sql_query"]
    fn talu_sql_query_raw(
        db_path: *const c_char,
        query: *const c_char,
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
    /// Execute a SQL query against TaluDB virtual tables and return a JSON array.
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
