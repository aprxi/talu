//! Metal cache kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");
const runtime_graph = @import("../runtime_graph.zig");

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
const ArrayHandle = compute.metal.graph.ArrayHandle;

pub const KVCache = struct {
    /// Canonical kernel-call contract for backend parity checks.
    pub const ForwardParams = struct {
        cache_index: usize,
        key_input: ArrayHandle,
        value_input: ArrayHandle,
    };

    cache: *Cache,

    pub fn forward(
        self: *KVCache,
        cache_index: usize,
        key_input: ArrayHandle,
        value_input: ArrayHandle,
    ) void {
        _ = self.cache.updateAndFetch(cache_index, key_input, value_input);
    }
};

test {
    _ = KVCache;
}
