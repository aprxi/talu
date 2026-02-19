//! Metal graph bridge for inference execution.
//!
//! This keeps a backend-local `graph` module in parity with CPU backend root.

const compute = @import("../../../compute/root.zig");

pub const Runtime = compute.metal.graph;
pub const Cache = Runtime.Cache;
pub const ShortConvCache = Runtime.ShortConvCache;
pub const ArrayHandle = Runtime.ArrayHandle;
