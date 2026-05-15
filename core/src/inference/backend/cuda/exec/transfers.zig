const std = @import("std");

const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

pub fn uploadCpuKvToMirrors(
    gpu_backend: anytype,
    cpu_backend: anytype,
    slot_index: usize,
    pos_start: usize,
    n_positions: usize,
) !void {
    const GpuType = @TypeOf(gpu_backend.*);
    if (comptime !@hasField(GpuType, "block_runtime")) return;
    const BrtType = @TypeOf(gpu_backend.block_runtime);
    if (comptime !@hasField(BrtType, "replicated_kv_sources")) return;
    const replicated = gpu_backend.block_runtime.replicated_kv_sources;
    if (replicated.len == 0) return;

    const mirrors = gpu_backend.block_runtime.mirror_kv;
    const head_dim = gpu_backend.head_dim;
    const allocator = gpu_backend.allocator;

    // Only f16 KV cache is supported for cross-device mirrors.
    if (gpu_backend.kv_cache_dtype != .f16) return error.UnsupportedModel;

    for (replicated, 0..) |src, mi| {
        const mk = &mirrors[mi];
        const cpu_kv = cpu_backend.kv_cache.getLayer(src.global_layer_idx);
        const n_kv_heads = src.kv_dim / head_dim;
        const kv_dim = src.kv_dim;
        const total_elems = n_positions * kv_dim;
        const total_bytes = total_elems * @sizeOf(f16);

        const staging_f16 = try allocator.alloc(f16, total_elems);
        defer allocator.free(staging_f16);

        // Convert + transpose K: CPU [slot][head][pos][dim] → GPU [pos][head*dim]
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_k = cpu_kv.getK(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_k[di]);
                }
            }
        }
        const gpu_k_offset = pos_start * kv_dim * @sizeOf(f16);
        const k_slice = try bufferSlice(&mk.k, gpu_k_offset, total_bytes);
        try k_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));

        // Convert + transpose V.
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_v = cpu_kv.getV(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_v[di]);
                }
            }
        }
        const gpu_v_offset = pos_start * kv_dim * @sizeOf(f16);
        const v_slice = try bufferSlice(&mk.v, gpu_v_offset, total_bytes);
        try v_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));
    }
}
