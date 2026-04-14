//! Package root for named `io_pkg` imports.

pub const transport = @import("io/transport/root.zig");
pub const plugins = @import("io/plugins/root.zig");
pub const safetensors = struct {
    pub const root = @import("io/safetensors/root.zig");
    pub const reader = @import("io/safetensors/reader.zig");
    pub const writer = @import("io/safetensors/writer.zig");
    pub const sharded = @import("io/safetensors/sharded.zig");
    pub const names = @import("io/safetensors/names.zig");
    pub const norm_loader = @import("io/safetensors/norm_loader.zig");
};
pub const repository = struct {
    pub const root = @import("io/repository/root.zig");
    pub const scheme = @import("io/repository/scheme.zig");
};
pub const json_helpers = @import("io/json_helpers.zig");
pub const json = @import("io/json/root.zig");
pub const kvbuf = @import("io/kvbuf/root.zig");
