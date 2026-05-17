//! Metal backend interface surface used by bridge and transport.
//!
//! Local stage execution is intentionally fail-closed until the Metal stage
//! executor implementation lands.

pub const stage_executor = @import("stage_executor.zig");
pub const transport_endpoint = @import("transport_endpoint.zig");
