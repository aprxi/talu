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

/// Sets the global log level for the talu library.
pub fn set_log_level(level: LogLevel) {
    // SAFETY: talu_set_log_level is a simple setter with no preconditions.
    // It accepts any i32 value and clamps internally if needed.
    unsafe { talu_sys::talu_set_log_level(level as i32) };
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
