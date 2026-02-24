//! Safe wrappers for talu logging configuration.
//!
//! Log levels match Zig's std.log.Level values:
//! - TRACE (1): Most verbose, includes code locations
//! - DEBUG (5): Includes code locations
//! - INFO (9): Progress, status
//! - WARN (13): Default - silent for normal operation
//! - ERROR (17): Failures only
//! - FATAL (21): Unrecoverable errors
//! - OFF (255): Completely silent

use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};

/// Log level for talu library.
///
/// Values match Zig's std.log.Level for direct C API compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LogLevel {
    /// Most verbose, includes code locations
    Trace = 1,
    /// Includes code locations
    Debug = 5,
    /// Progress, status
    Info = 9,
    /// Default - silent for normal operation
    Warn = 13,
    /// Failures only
    Error = 17,
    /// Unrecoverable errors
    Fatal = 21,
    /// Completely silent
    Off = 255,
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LogFormat {
    /// JSON (OpenTelemetry compliant, machine-readable)
    Json = 0,
    /// Colored, human-readable
    Human = 1,
}

/// Structured core log record delivered via `set_core_log_callback`.
#[derive(Debug, Clone)]
pub struct CoreLogRecord {
    pub level: LogLevel,
    pub scope: String,
    pub message: String,
    pub attrs_json: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
}

/// Callback for core log bridge records.
pub type CoreLogCallback = fn(&CoreLogRecord);

static CORE_LOG_CALLBACK: OnceLock<RwLock<Option<CoreLogCallback>>> = OnceLock::new();

fn callback_slot() -> &'static RwLock<Option<CoreLogCallback>> {
    CORE_LOG_CALLBACK.get_or_init(|| RwLock::new(None))
}

extern "C" fn log_callback_trampoline(
    level: i32,
    scope_ptr: *const u8,
    scope_len: usize,
    body_ptr: *const u8,
    body_len: usize,
    attrs_json_ptr: *const u8,
    attrs_json_len: usize,
    file_ptr: *const u8,
    file_len: usize,
    line: u32,
    _user_data: *mut c_void,
) {
    let cb = callback_slot().read().ok().and_then(|guard| *guard);
    let Some(callback) = cb else {
        return;
    };

    let scope = if scope_ptr.is_null() || scope_len == 0 {
        String::new()
    } else {
        // SAFETY: Pointer/length are provided by the core callback and are valid
        // only during this call. We copy into owned Strings before returning.
        let bytes = unsafe { std::slice::from_raw_parts(scope_ptr, scope_len) };
        String::from_utf8_lossy(bytes).to_string()
    };

    let message = if body_ptr.is_null() || body_len == 0 {
        String::new()
    } else {
        // SAFETY: Same lifetime contract as `scope_ptr/scope_len`.
        let bytes = unsafe { std::slice::from_raw_parts(body_ptr, body_len) };
        String::from_utf8_lossy(bytes).to_string()
    };

    let attrs_json = if attrs_json_ptr.is_null() || attrs_json_len == 0 {
        None
    } else {
        // SAFETY: Same lifetime contract as `scope_ptr/scope_len`.
        let bytes = unsafe { std::slice::from_raw_parts(attrs_json_ptr, attrs_json_len) };
        let text = String::from_utf8_lossy(bytes).to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    };

    let file = if file_ptr.is_null() || file_len == 0 {
        None
    } else {
        // SAFETY: Same lifetime contract as `scope_ptr/scope_len`.
        let bytes = unsafe { std::slice::from_raw_parts(file_ptr, file_len) };
        Some(String::from_utf8_lossy(bytes).to_string())
    };

    let level = match level {
        1 => LogLevel::Trace,
        5 => LogLevel::Debug,
        9 => LogLevel::Info,
        13 => LogLevel::Warn,
        17 => LogLevel::Error,
        21 => LogLevel::Fatal,
        255 => LogLevel::Off,
        _ => LogLevel::Warn,
    };

    let record = CoreLogRecord {
        level,
        scope,
        message,
        attrs_json,
        file,
        line: if line == 0 { None } else { Some(line) },
    };
    callback(&record);
}

/// Sets the global log level for the talu library.
pub fn set_log_level(level: LogLevel) {
    // SAFETY: talu_set_log_level is a simple setter with no preconditions.
    // It accepts any i32 value and clamps internally if needed.
    unsafe { talu_sys::talu_set_log_level(level as i32) };
}

/// Sets a scope filter for the talu library logs.
///
/// The filter is matched against the fully-qualified scope (e.g. "core::inference").
/// Supports trailing `*` for prefix match: "core::*" matches all core scopes.
/// Empty string or "*" disables filtering.
pub fn set_log_filter(filter: &str) {
    // SAFETY: talu_set_log_filter copies the data internally.
    unsafe { talu_sys::talu_set_log_filter(filter.as_ptr(), filter.len()) };
}

/// Sets the global log format for the talu library.
pub fn set_log_format(format: LogFormat) {
    // SAFETY: talu_set_log_format is a simple setter with no preconditions.
    // It accepts any i32 value and clamps internally if needed.
    unsafe { talu_sys::talu_set_log_format(format as i32) };
}

/// Gets the current log level.
pub fn get_log_level() -> LogLevel {
    // SAFETY: talu_get_log_level is a simple getter with no preconditions.
    let level = unsafe { talu_sys::talu_get_log_level() };
    match level {
        1 => LogLevel::Trace,
        5 => LogLevel::Debug,
        9 => LogLevel::Info,
        13 => LogLevel::Warn,
        17 => LogLevel::Error,
        21 => LogLevel::Fatal,
        255 => LogLevel::Off,
        _ => LogLevel::Warn, // Default
    }
}

/// Gets the current log format.
pub fn get_log_format() -> LogFormat {
    // SAFETY: talu_get_log_format is a simple getter with no preconditions.
    let format = unsafe { talu_sys::talu_get_log_format() };
    match format {
        0 => LogFormat::Json,
        _ => LogFormat::Human,
    }
}

/// Install or clear the core log callback bridge.
///
/// The callback receives copied (owned) data and may outlive the FFI frame.
pub fn set_core_log_callback(callback: Option<CoreLogCallback>) {
    if let Ok(mut slot) = callback_slot().write() {
        *slot = callback;
    }
    let raw_callback: *mut c_void = if callback.is_some() {
        (log_callback_trampoline as *const ()) as *mut c_void
    } else {
        std::ptr::null_mut()
    };

    // SAFETY: Setter stores callback pointer and opaque user_data only.
    unsafe {
        talu_sys::talu_set_log_callback(raw_callback, std::ptr::null_mut());
    }
}
