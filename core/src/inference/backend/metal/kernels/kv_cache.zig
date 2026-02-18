//! Metal cache kernel surface.

pub const supported = true;

const compute = @import("../../../../compute/root.zig");
const cache = @import("../executor/runtime.zig");

pub const Cache = cache.Cache;
pub const ShortConvCache = cache.ShortConvCache;
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
