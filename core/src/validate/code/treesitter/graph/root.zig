//! Call graph analysis module.
//!
//! Extracts callable definitions, call sites, and import aliases from
//! source code using tree-sitter ASTs. Supports Python, Rust, and
//! JavaScript/TypeScript.

pub const types = @import("types.zig");
pub const paths = @import("paths.zig");
pub const extract = @import("extract.zig");
pub const python = @import("python.zig");
pub const rust_lang = @import("rust_lang.zig");
pub const javascript = @import("javascript.zig");
pub const json_output = @import("json_output.zig");

// Re-export commonly used types
pub const CallableDefinitionInfo = types.CallableDefinitionInfo;
pub const CallSiteDetail = types.CallSiteDetail;
pub const AliasInfo = types.AliasInfo;
pub const ParamDetail = types.ParamDetail;
pub const Argument = types.Argument;
pub const Span = types.Span;
pub const Visibility = types.Visibility;
pub const ArgumentSource = types.ArgumentSource;
pub const ImportEntry = types.ImportEntry;

// Re-export top-level functions
pub const extractCallablesAndAliases = extract.extractCallablesAndAliases;
pub const extractCallSites = extract.extractCallSites;
pub const extractCallablesToJson = extract.extractCallablesToJson;
pub const extractCallSitesToJson = extract.extractCallSitesToJson;
pub const callablesToJson = json_output.callablesToJson;
pub const callSitesToJson = json_output.callSitesToJson;
pub const aliasesToJson = json_output.aliasesToJson;
pub const extractionToJson = json_output.extractionToJson;

// Force compilation/testing of all submodules
comptime {
    _ = types;
    _ = paths;
    _ = extract;
    _ = python;
    _ = rust_lang;
    _ = javascript;
    _ = json_output;
}
