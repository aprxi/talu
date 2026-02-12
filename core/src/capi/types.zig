//! C API shared types.
//!
//! Defines extern structs and enums used across multiple C API modules.
const std = @import("std");

// Internal Zig Enum (for logic, not ABI)
pub const BackendType = enum(c_int) {
    Unspecified = -1,
    Local = 0,
    OpenAICompatible = 1,
};

pub const LocalConfig = extern struct {
    gpu_layers: c_int,
    use_mmap: u8,
    num_threads: c_int,
    _reserved: [32]u8,
};

pub const OpenAICompatibleConfig = extern struct {
    base_url: ?[*:0]const u8,
    api_key: ?[*:0]const u8,
    org_id: ?[*:0]const u8,
    timeout_ms: c_int,
    max_retries: c_int,
    /// Custom HTTP headers as JSON object string.
    /// Format: {"Header-Name": "value", "Another-Header": "value2"}
    /// These are added to every request to this backend.
    custom_headers_json: ?[*:0]const u8,
    _reserved: [24]u8, // Reduced from 32 to accommodate pointer
};

pub const BackendUnion = extern union {
    local: LocalConfig,
    openai_compat: OpenAICompatibleConfig,
};

pub const TaluModelSpec = extern struct {
    abi_version: u32,
    struct_size: u32,
    ref: ?[*:0]const u8,
    backend_type_raw: c_int,
    backend_config: BackendUnion,
};

pub const TaluCapabilities = extern struct {
    abi_version: u32,
    struct_size: u32,
    streaming: u8,
    tool_calling: u8,
    logprobs: u8,
    embeddings: u8,
    json_schema: u8,
    _reserved: [32]u8,
};

test "TaluModelSpec ABI layout" {
    var offset: usize = 0;
    offset = std.mem.alignForward(usize, offset, @alignOf(u32));
    try std.testing.expectEqual(offset, @offsetOf(TaluModelSpec, "abi_version"));
    offset += @sizeOf(u32);

    offset = std.mem.alignForward(usize, offset, @alignOf(u32));
    try std.testing.expectEqual(offset, @offsetOf(TaluModelSpec, "struct_size"));
    offset += @sizeOf(u32);

    offset = std.mem.alignForward(usize, offset, @alignOf(?[*:0]const u8));
    try std.testing.expectEqual(offset, @offsetOf(TaluModelSpec, "ref"));
    offset += @sizeOf(?[*:0]const u8);

    offset = std.mem.alignForward(usize, offset, @alignOf(c_int));
    try std.testing.expectEqual(offset, @offsetOf(TaluModelSpec, "backend_type_raw"));
    offset += @sizeOf(c_int);

    offset = std.mem.alignForward(usize, offset, @alignOf(BackendUnion));
    try std.testing.expectEqual(offset, @offsetOf(TaluModelSpec, "backend_config"));
}
