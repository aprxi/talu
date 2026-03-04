const runtime = @import("executor/runtime.zig");
const kv_cache = @import("kernels/kv_cache.zig");

pub const RuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    layered_cache: *kv_cache.LayeredBatchedKVCache,
    scratch: *runtime.ScratchBuffer,
    slot_index: usize,
};

pub const KvRuntimeState = RuntimeState;
pub const RecurrentRuntimeState = RuntimeState;
pub const ScratchRuntimeState = RuntimeState;
