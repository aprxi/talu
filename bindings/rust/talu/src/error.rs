//! Error types for the talu SDK.

use std::ffi::CStr;
use talu_sys;

#[derive(Debug)]
pub enum Error {
    Talu(String),
    NulError(std::ffi::NulError),
    Generic(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Talu(msg) => f.write_str(msg),
            Error::NulError(_) => f.write_str("null string conversion"),
            Error::Generic(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::NulError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::NulError(e)
    }
}

impl Error {
    pub fn talu(msg: impl Into<String>) -> Self {
        Self::Talu(msg.into())
    }

    pub fn generic(msg: impl Into<String>) -> Self {
        Self::Generic(msg.into())
    }
}

/// Retrieves and clears the last error message from the talu C API.
pub fn last_error_message() -> Option<String> {
    // SAFETY: talu_last_error() returns a valid C string pointer or null.
    let ptr = unsafe { talu_sys::talu_last_error() };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null and points to a valid null-terminated C string
    // owned by the thread-local error buffer.
    let msg = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string();
    // SAFETY: Clears the thread-local error buffer. No preconditions.
    unsafe { talu_sys::talu_clear_error() };
    Some(msg)
}

/// Creates an Error from the last talu error, or uses a fallback message.
pub fn error_from_last_or(fallback: &str) -> Error {
    last_error_message()
        .map(Error::talu)
        .unwrap_or_else(|| Error::generic(fallback))
}
