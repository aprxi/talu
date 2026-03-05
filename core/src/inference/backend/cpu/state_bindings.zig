const kv_cache = @import("kernels/kv_cache.zig");
const runtime = @import("executor/runtime.zig");

pub const RuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
};

pub const KvRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    layered_cache: *kv_cache.LayeredBatchedKVCache,
    scratch: *runtime.ScratchBuffer,
    slot_index: usize,
};

pub const RecurrentRuntimeState = extern struct {
    runtime_kind: u8,
    _pad: [7]u8 = [_]u8{0} ** 7,
    scratch: *runtime.ScratchBuffer,
    slot_index: usize,
};
pub const ScratchRuntimeState = RuntimeState;
