//! I/O Subsystem - model file formats, storage, and transport.
//!
//! This module is intentionally model-agnostic. It provides container and
//! transport primitives consumed by higher-level graph/inference code.

/// Model repository format (HF-style structure).
pub const repository = @import("repository/root.zig");

/// Transport layer (HTTP, HuggingFace Hub API).
pub const transport = @import("transport/root.zig");

/// Plugin discovery (UI plugin scanner).
pub const plugins = @import("plugins/root.zig");

/// SafeTensors format parsing.
pub const safetensors = struct {
    pub const root = @import("safetensors/root.zig");
    pub const names = @import("safetensors/names.zig");
    pub const norm_loader = @import("safetensors/norm_loader.zig");
};

/// JSON value extraction helpers.
pub const json_helpers = @import("json_helpers.zig");

/// JSON parsing with centralized size limits and error mapping.
pub const json = @import("json/root.zig");

/// KvBuf (Key-Value Buffer) binary format for zero-copy field access.
pub const kvbuf = @import("kvbuf/root.zig");
