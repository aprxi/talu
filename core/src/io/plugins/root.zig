//! Plugin discovery — scans ~/.talu/plugins/ for UI plugins.
//!
//! Each plugin is either:
//! - A single .js/.ts file (single-file plugin)
//! - A directory containing talu.json manifest and/or index.ts/index.js
//!
//! The scanner returns raw manifest JSON (or inferred defaults) and
//! the entry path for each discovered plugin. It does not parse manifests
//! into structured types — that is the binding layer's responsibility.

pub const scanner = @import("scanner.zig");

// Re-export public types and functions.
pub const PluginInfo = scanner.PluginInfo;
pub const ScanError = scanner.ScanError;
pub const scanPlugins = scanner.scanPlugins;
pub const freePluginList = scanner.freePluginList;
pub const getPluginsDir = scanner.getPluginsDir;
