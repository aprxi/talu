//! Filesystem primitives for agent runtime features.
//!
//! This module centralizes workspace path sandboxing and typed file operations
//! so tool wrappers and C API entrypoints can reuse one implementation.

pub const path = @import("path.zig");
pub const operations = @import("operations.zig");
