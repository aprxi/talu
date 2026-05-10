//! I/O Subsystem - model file formats, storage, and transport.
//!
//! This module is intentionally model-agnostic. It provides container and
//! transport primitives consumed by higher-level graph/inference code.

/// Model repository format (HF-style structure).
pub const repository = struct {
    pub const root = @import("repository/root.zig");
    pub const scheme = @import("repository/scheme.zig");
    pub const talu_cache = @import("repository/talu_cache.zig");
};

/// Transport layer (HTTP, HuggingFace Hub API).
pub const transport = @import("transport/root.zig");

/// HuggingFace model config fetch and minimal metadata view.
pub const model_config = @import("model_config.zig");

/// SafeTensors format parsing.
pub const safetensors = struct {
    pub const root = @import("safetensors/root.zig");
    pub const reader = @import("safetensors/reader.zig");
    pub const writer = @import("safetensors/writer.zig");
    pub const sharded = @import("safetensors/sharded.zig");
    pub const names = @import("safetensors/names.zig");
    pub const norm_loader = @import("safetensors/norm_loader.zig");
};

/// JSON value extraction helpers.
pub const json_helpers = @import("json_helpers.zig");

/// JSON parsing with centralized size limits and error mapping.
pub const json = @import("json/root.zig");
