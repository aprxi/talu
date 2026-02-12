//! Centralized JSON parsing with security hardening.
//!
//! All JSON parsing in core MUST use these helpers. Direct use of
//! std.json.parseFromSlice is prohibited outside this module.
//!
//! Responsibilities:
//! - Size limit enforcement (caller specifies max_size_bytes)
//! - Consistent error mapping (ParseError)
//! - Rejection logging (scope, size, reason)
//! - Correct parsing via std.json (no heuristic parsers)

const parser = @import("parser.zig");

pub const ParseOptions = parser.ParseOptions;
pub const ParseError = parser.ParseError;
pub const parseValue = parser.parseValue;
pub const parseStruct = parser.parseStruct;
pub const parseStructFromValue = parser.parseStructFromValue;
pub const extractStringField = parser.extractStringField;
