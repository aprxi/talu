//! Inference-owned runtime state wrappers for Metal MLX execution.
//!
//! Compute owns primitive array/lazy graph operations. Inference owns cache
//! state containers and runtime decode state lifecycles.

const compute = @import("../../../compute/root.zig");

const mlx_graph = compute.metal.graph;

pub const ArrayHandle = mlx_graph.ArrayHandle;
pub const CacheHandle = ?*anyopaque;
pub const ShortConvCacheHandle = ?*anyopaque;
pub const MambaCacheHandle = ?*anyopaque;

extern fn mlx_cache_create(n_layers: usize, max_seq_len: usize) CacheHandle;
extern fn mlx_cache_create_bfloat16(n_layers: usize, max_seq_len: usize) CacheHandle;
extern fn mlx_cache_free(cache: CacheHandle) void;
extern fn mlx_cache_update_and_fetch_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_new: ArrayHandle,
    v_new: ArrayHandle,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
    is_prefill_out: *bool,
) void;
extern fn mlx_cache_get_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
) void;
extern fn mlx_cache_set_full_bfloat16(
    cache: CacheHandle,
    layer_idx: usize,
    k_full: ArrayHandle,
    v_full: ArrayHandle,
) void;
extern fn mlx_cache_eval_all(cache: CacheHandle, n_layers: usize) void;
extern fn mlx_cache_update_and_fetch(
    cache: CacheHandle,
    layer_idx: usize,
    k_new: ArrayHandle,
    v_new: ArrayHandle,
    k_out: *ArrayHandle,
    v_out: *ArrayHandle,
    is_prefill_out: *bool,
) void;
extern fn mlx_cache_get_quantized(
    cache: CacheHandle,
    layer_idx: usize,
    k_weights_out: *ArrayHandle,
    k_scales_out: *ArrayHandle,
    k_biases_out: *ArrayHandle,
    v_weights_out: *ArrayHandle,
    v_scales_out: *ArrayHandle,
    v_biases_out: *ArrayHandle,
) void;

extern fn mlx_shortconv_cache_create(n_layers: usize) ShortConvCacheHandle;
extern fn mlx_shortconv_cache_reset(cache: ShortConvCacheHandle) void;
extern fn mlx_shortconv_cache_free(cache: ShortConvCacheHandle) void;
extern fn mlx_mamba_cache_create(n_layers: usize) MambaCacheHandle;
extern fn mlx_mamba_cache_reset(cache: MambaCacheHandle) void;
extern fn mlx_mamba_cache_free(cache: MambaCacheHandle) void;

pub const Cache = struct {
    handle: CacheHandle,
    use_bfloat16: bool,

    pub fn init(n_layers: usize, use_bfloat16: bool, max_seq_len: usize) Cache {
        const handle = if (use_bfloat16)
            mlx_cache_create_bfloat16(n_layers, max_seq_len)
        else
            mlx_cache_create(n_layers, max_seq_len);
        return .{ .handle = handle, .use_bfloat16 = use_bfloat16 };
    }

    pub fn disabled(use_bfloat16: bool) Cache {
        return .{ .handle = null, .use_bfloat16 = use_bfloat16 };
    }

    pub fn deinit(self: Cache) void {
        if (self.handle == null) return;
        mlx_cache_free(self.handle);
    }

    pub fn updateAndFetch(self: Cache, layer_idx: usize, k_new: ArrayHandle, v_new: ArrayHandle) struct { k: ArrayHandle, v: ArrayHandle, is_prefill: bool } {
        if (self.handle == null) return .{ .k = k_new, .v = v_new, .is_prefill = false };
        var k_cache: ArrayHandle = null;
        var v_cache: ArrayHandle = null;
        var is_prefill: bool = false;

        if (self.use_bfloat16) {
            mlx_cache_update_and_fetch_bfloat16(self.handle, layer_idx, k_new, v_new, &k_cache, &v_cache, &is_prefill);
        } else {
            mlx_cache_update_and_fetch(self.handle, layer_idx, k_new, v_new, &k_cache, &v_cache, &is_prefill);
        }
        return .{ .k = k_cache, .v = v_cache, .is_prefill = is_prefill };
    }

    pub fn get(self: Cache, layer_idx: usize) struct { k: ArrayHandle, v: ArrayHandle } {
        if (self.handle == null) return .{ .k = null, .v = null };
        var k_cache: ArrayHandle = null;
        var v_cache: ArrayHandle = null;
        mlx_cache_get_bfloat16(self.handle, layer_idx, &k_cache, &v_cache);
        return .{ .k = k_cache, .v = v_cache };
    }

    pub fn setFull(self: Cache, layer_idx: usize, k_full: ArrayHandle, v_full: ArrayHandle) void {
        if (self.handle == null) return;
        mlx_cache_set_full_bfloat16(self.handle, layer_idx, k_full, v_full);
    }

    pub fn evalAll(self: Cache, n_layers: usize) void {
        if (self.handle == null) return;
        mlx_cache_eval_all(self.handle, n_layers);
    }

    pub fn getQuantized(self: Cache, layer_idx: usize) struct {
        k_weights: ArrayHandle,
        k_scales: ArrayHandle,
        k_biases: ArrayHandle,
        v_weights: ArrayHandle,
        v_scales: ArrayHandle,
        v_biases: ArrayHandle,
    } {
        if (self.handle == null) {
            return .{
                .k_weights = null,
                .k_scales = null,
                .k_biases = null,
                .v_weights = null,
                .v_scales = null,
                .v_biases = null,
            };
        }
        var k_w: ArrayHandle = null;
        var k_s: ArrayHandle = null;
        var k_b: ArrayHandle = null;
        var v_w: ArrayHandle = null;
        var v_s: ArrayHandle = null;
        var v_b: ArrayHandle = null;
        mlx_cache_get_quantized(self.handle, layer_idx, &k_w, &k_s, &k_b, &v_w, &v_s, &v_b);
        return .{
            .k_weights = k_w,
            .k_scales = k_s,
            .k_biases = k_b,
            .v_weights = v_w,
            .v_scales = v_s,
            .v_biases = v_b,
        };
    }
};

pub const ShortConvCache = struct {
    handle: ShortConvCacheHandle,

    pub fn init(n_layers: usize) ShortConvCache {
        return .{ .handle = mlx_shortconv_cache_create(n_layers) };
    }

    pub fn disabled() ShortConvCache {
        return .{ .handle = null };
    }

    pub fn reset(self: ShortConvCache) void {
        if (self.handle == null) return;
        mlx_shortconv_cache_reset(self.handle);
    }

    pub fn deinit(self: ShortConvCache) void {
        if (self.handle == null) return;
        mlx_shortconv_cache_free(self.handle);
    }
};

pub const MambaCache = struct {
    handle: MambaCacheHandle,

    pub fn init(n_layers: usize) MambaCache {
        return .{ .handle = mlx_mamba_cache_create(n_layers) };
    }

    pub fn disabled() MambaCache {
        return .{ .handle = null };
    }

    pub fn reset(self: MambaCache) void {
        if (self.handle == null) return;
        mlx_mamba_cache_reset(self.handle);
    }

    pub fn deinit(self: MambaCache) void {
        if (self.handle == null) return;
        mlx_mamba_cache_free(self.handle);
    }
};
