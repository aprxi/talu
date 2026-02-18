//! KV cache write/copy primitives for CPU.

const std = @import("std");

/// Copy KV data from a single-cache layout into a slotted batched-cache layout.
///
/// Source layout:
/// - `[kv_head, seq_pos, head_dim]` with stride `src_seq_stride` for `seq_pos`.
///
/// Destination layout (flattened):
/// - `[slot, kv_head, seq_pos, head_dim]` with `dst_slot_stride` and `dst_head_stride`.
pub fn copyFromSingleCache(
    dst_key_cache: []f32,
    dst_value_cache: []f32,
    dst_slot_stride: usize,
    dst_head_stride: usize,
    slot_index: usize,
    src_key_cache: []const f32,
    src_value_cache: []const f32,
    src_seq_stride: usize,
    kv_head_count: usize,
    head_dim: usize,
    seq_len: usize,
) void {
    const src_needed = kv_head_count * src_seq_stride * head_dim;
    std.debug.assert(src_key_cache.len >= src_needed);
    std.debug.assert(src_value_cache.len >= src_needed);
    const dst_slot_base = slot_index * dst_slot_stride;
    std.debug.assert(dst_key_cache.len >= dst_slot_base + kv_head_count * dst_head_stride);
    std.debug.assert(dst_value_cache.len >= dst_slot_base + kv_head_count * dst_head_stride);

    for (0..kv_head_count) |kv_head_idx| {
        for (0..seq_len) |token_pos| {
            const src_offset = kv_head_idx * src_seq_stride * head_dim + token_pos * head_dim;
            const dst_offset = dst_slot_base + kv_head_idx * dst_head_stride + token_pos * head_dim;
            @memcpy(dst_key_cache[dst_offset .. dst_offset + head_dim], src_key_cache[src_offset .. src_offset + head_dim]);
            @memcpy(dst_value_cache[dst_offset .. dst_offset + head_dim], src_value_cache[src_offset .. src_offset + head_dim]);
        }
    }
}

/// Append one token worth of K/V vectors into a slotted KV cache.
pub fn appendTokenKV(
    key_cache: []f32,
    value_cache: []f32,
    slot_stride: usize,
    head_stride: usize,
    head_dim: usize,
    n_kv_heads: usize,
    slot_index: usize,
    slot_position: usize,
    k_data: []const f32,
    v_data: []const f32,
) void {
    const kv_values_per_token = n_kv_heads * head_dim;
    std.debug.assert(k_data.len == kv_values_per_token);
    std.debug.assert(v_data.len == kv_values_per_token);
    const slot_base = slot_index * slot_stride;
    std.debug.assert(key_cache.len >= slot_base + n_kv_heads * head_stride);
    std.debug.assert(value_cache.len >= slot_base + n_kv_heads * head_stride);

    for (0..n_kv_heads) |kv_head| {
        const src_k = k_data[kv_head * head_dim ..][0..head_dim];
        const src_v = v_data[kv_head * head_dim ..][0..head_dim];
        const dst_offset = slot_base + kv_head * head_stride + slot_position * head_dim;
        @memcpy(key_cache[dst_offset .. dst_offset + head_dim], src_k);
        @memcpy(value_cache[dst_offset .. dst_offset + head_dim], src_v);
    }
}

/// Append a prefill batch `[seq_len, n_kv_heads, head_dim]` into slotted KV cache.
pub fn appendBatchKV(
    key_cache: []f32,
    value_cache: []f32,
    slot_stride: usize,
    head_stride: usize,
    head_dim: usize,
    n_kv_heads: usize,
    slot_index: usize,
    start_position: usize,
    k_data: []const f32,
    v_data: []const f32,
    seq_len: usize,
) void {
    const kv_values_per_token = n_kv_heads * head_dim;
    std.debug.assert(k_data.len == seq_len * kv_values_per_token);
    std.debug.assert(v_data.len == seq_len * kv_values_per_token);
    const slot_base = slot_index * slot_stride;
    std.debug.assert(key_cache.len >= slot_base + n_kv_heads * head_stride);
    std.debug.assert(value_cache.len >= slot_base + n_kv_heads * head_stride);

    for (0..n_kv_heads) |kv_head| {
        for (0..seq_len) |token_index| {
            const src_offset = token_index * kv_values_per_token + kv_head * head_dim;
            const dst_offset = slot_base + kv_head * head_stride + (start_position + token_index) * head_dim;
            @memcpy(
                key_cache[dst_offset .. dst_offset + head_dim],
                k_data[src_offset .. src_offset + head_dim],
            );
            @memcpy(
                value_cache[dst_offset .. dst_offset + head_dim],
                v_data[src_offset .. src_offset + head_dim],
            );
        }
    }
}

/// Populate MLA caches from prefill buffers.
pub fn populateMLACache(
    key_cache: []f32,
    value_cache: []f32,
    rope_key_cache: []f32,
    max_seq_len: usize,
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    cache_position: usize,
    kv_expanded: []const f32,
    kv_compressed: []const f32,
    seq_len: usize,
) void {
    const kv_exp_dim = qk_nope_head_dim + v_head_dim;
    const kv_comp_dim = kv_lora_rank + qk_rope_head_dim;
    std.debug.assert(kv_expanded.len >= seq_len * n_heads * kv_exp_dim);
    std.debug.assert(kv_compressed.len >= seq_len * kv_comp_dim);

    for (0..seq_len) |token_idx| {
        const cache_pos = cache_position + token_idx;
        std.debug.assert(cache_pos < max_seq_len);

        for (0..n_heads) |head_idx| {
            const exp_offset = token_idx * n_heads * kv_exp_dim + head_idx * kv_exp_dim;
            const k_cache_offset = head_idx * max_seq_len * qk_head_dim + cache_pos * qk_head_dim;
            const v_cache_offset = head_idx * max_seq_len * v_head_dim + cache_pos * v_head_dim;

            @memcpy(
                key_cache[k_cache_offset .. k_cache_offset + qk_nope_head_dim],
                kv_expanded[exp_offset .. exp_offset + qk_nope_head_dim],
            );
            @memcpy(
                value_cache[v_cache_offset .. v_cache_offset + v_head_dim],
                kv_expanded[exp_offset + qk_nope_head_dim .. exp_offset + kv_exp_dim],
            );
        }

        const rope_src_offset = token_idx * kv_comp_dim + kv_lora_rank;
        const rope_cache_offset = cache_pos * qk_rope_head_dim;
        @memcpy(
            rope_key_cache[rope_cache_offset .. rope_cache_offset + qk_rope_head_dim],
            kv_compressed[rope_src_offset .. rope_src_offset + qk_rope_head_dim],
        );
    }
}

/// Append one MLA token into caches during decode.
pub fn appendMLAToken(
    key_cache: []f32,
    value_cache: []f32,
    rope_key_cache: []f32,
    max_seq_len: usize,
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    cache_position: usize,
    kv_expanded_token: []const f32,
    kv_compressed_token: []const f32,
) void {
    std.debug.assert(cache_position < max_seq_len);
    const kv_exp_dim = qk_nope_head_dim + v_head_dim;
    const kv_comp_dim = kv_lora_rank + qk_rope_head_dim;
    std.debug.assert(kv_expanded_token.len >= n_heads * kv_exp_dim);
    std.debug.assert(kv_compressed_token.len >= kv_comp_dim);

    for (0..n_heads) |head_idx| {
        const exp_offset = head_idx * kv_exp_dim;
        const k_cache_offset = head_idx * max_seq_len * qk_head_dim + cache_position * qk_head_dim;
        const v_cache_offset = head_idx * max_seq_len * v_head_dim + cache_position * v_head_dim;

        @memcpy(
            key_cache[k_cache_offset .. k_cache_offset + qk_nope_head_dim],
            kv_expanded_token[exp_offset .. exp_offset + qk_nope_head_dim],
        );
        @memcpy(
            value_cache[v_cache_offset .. v_cache_offset + v_head_dim],
            kv_expanded_token[exp_offset + qk_nope_head_dim .. exp_offset + kv_exp_dim],
        );
    }

    const rope_src_offset = kv_lora_rank;
    const rope_cache_offset = cache_position * qk_rope_head_dim;
    @memcpy(
        rope_key_cache[rope_cache_offset .. rope_cache_offset + qk_rope_head_dim],
        kv_compressed_token[rope_src_offset .. rope_src_offset + qk_rope_head_dim],
    );
}

test "copyFromSingleCache copies kv rows into slot layout" {
    const kv_head_count: usize = 1;
    const head_dim: usize = 2;
    const seq_len: usize = 2;
    const src_seq_stride: usize = 4;
    const dst_slot_stride: usize = 8;
    const dst_head_stride: usize = 4;

    // Source with extra capacity stride (4 positions, use first 2)
    const src_k = [_]f32{
        10, 11, // pos0
        20, 21, // pos1
        30, 31, // pos2
        40, 41, // pos3
    };
    const src_v = [_]f32{
        110, 111,
        120, 121,
        130, 131,
        140, 141,
    };
    var dst_k = [_]f32{0} ** 16;
    var dst_v = [_]f32{0} ** 16;

    copyFromSingleCache(
        &dst_k,
        &dst_v,
        dst_slot_stride,
        dst_head_stride,
        1,
        &src_k,
        &src_v,
        src_seq_stride,
        kv_head_count,
        head_dim,
        seq_len,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 10), dst_k[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), dst_k[9], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20), dst_k[10], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 21), dst_k[11], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 110), dst_v[8], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 121), dst_v[11], 1e-6);
}

test "appendTokenKV writes one token per head" {
    var key_cache = [_]f32{0} ** 32;
    var value_cache = [_]f32{0} ** 32;
    const k = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const v = [_]f32{ 11, 12, 13, 14, 15, 16 };

    appendTokenKV(&key_cache, &value_cache, 16, 8, 3, 2, 1, 1, &k, &v);

    try std.testing.expectApproxEqAbs(@as(f32, 1), key_cache[19], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), key_cache[29], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), value_cache[19], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 16), value_cache[29], 1e-6);
}

test "appendMLAToken writes K/V and rope slices" {
    const max_seq_len: usize = 4;
    const n_heads: usize = 2;
    const qk_nope_head_dim: usize = 2;
    const qk_head_dim: usize = 4;
    const v_head_dim: usize = 2;
    const kv_lora_rank: usize = 2;
    const qk_rope_head_dim: usize = 2;

    var key_cache = [_]f32{0} ** (n_heads * max_seq_len * qk_head_dim);
    var value_cache = [_]f32{0} ** (n_heads * max_seq_len * v_head_dim);
    var rope_cache = [_]f32{0} ** (max_seq_len * qk_rope_head_dim);

    const kv_expanded = [_]f32{
        1, 2, 11, 12, // head 0: k_nope, v
        3, 4, 13, 14, // head 1: k_nope, v
    };
    const kv_compressed = [_]f32{
        100, 101, // lora
        201, 202, // rope
    };

    appendMLAToken(
        &key_cache,
        &value_cache,
        &rope_cache,
        max_seq_len,
        n_heads,
        qk_nope_head_dim,
        qk_head_dim,
        v_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        1,
        &kv_expanded,
        &kv_compressed,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 1), key_cache[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), key_cache[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), key_cache[20], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4), key_cache[21], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11), value_cache[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14), value_cache[11], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 201), rope_cache[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 202), rope_cache[3], 1e-6);
}
