//! Development/test module root.
//!
//! This mirrors the production C-API root (`lib.zig`) and adds broad module
//! exports used by integration tests and internal tooling.

const prod = @import("lib.zig");

pub const capi = prod.capi;
pub const ABI_VERSION = prod.ABI_VERSION;
pub const talu_get_abi_version = prod.talu_get_abi_version;

pub const core = @import("root.zig");
pub const tokenizer = @import("tokenizer/root.zig");
pub const template = @import("template/root.zig");
pub const io = @import("io_pkg");
pub const inference = @import("inference_pkg");
pub const responses = @import("responses/root.zig");
pub const router = @import("router/root.zig");
pub const generation_config = @import("config/generation.zig");
pub const converter = @import("converter/root.zig");
pub const compute = @import("compute_pkg");
pub const tensor = @import("tensor_pkg");
pub const xray = @import("xray_pkg");
pub const validate = @import("validate_pkg");
pub const db = @import("db/root.zig");
pub const collab = @import("collab/root.zig");
pub const policy = @import("agent/policy/root.zig");
pub const dump = @import("xray_pkg").dump.root;
pub const agent = @import("agent/root.zig");
pub const train = @import("train/root.zig");

pub const models = struct {
    pub const dispatcher = @import("models_pkg");
};

pub const nn = struct {
    pub const sampling = inference.sampling;
};
