//! Metric calculations for training benchmarks.
//!
//! Computes GFLOP/s and throughput from timing data.

/// Compute GFLOP/s from total FLOP count and elapsed nanoseconds.
pub fn gflops(flops: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(elapsed_ns));
}

/// Estimate FLOPs for a matmul: A[M,K] @ B[K,N] = 2*M*K*N (multiply + add).
pub fn matmulFlops(m: usize, k: usize, n: usize) u64 {
    return 2 * @as(u64, m) * @as(u64, k) * @as(u64, n);
}

/// Estimate FLOPs for a single transformer layer forward pass.
/// Q/K/V projections + O projection + gate/up/down FFN + attention.
pub fn layerForwardFlops(bs: usize, d: usize, nh: usize, nkv: usize, hd: usize, ff: usize, s: usize) u64 {
    const b: u64 = @intCast(bs / s);
    const s64: u64 = @intCast(s);
    const d64: u64 = @intCast(d);
    const nh64: u64 = @intCast(nh);
    const nkv64: u64 = @intCast(nkv);
    const hd64: u64 = @intCast(hd);
    const ff64: u64 = @intCast(ff);
    const bs64: u64 = @intCast(bs);

    // Q/K/V projections: 3 matmuls [bs, d] @ [out, d]^T
    var flops: u64 = 0;
    flops += matmulFlops(bs64, d64, nh64 * hd64); // Q
    flops += matmulFlops(bs64, d64, nkv64 * hd64); // K
    flops += matmulFlops(bs64, d64, nkv64 * hd64); // V

    // Attention: QK^T and softmax@V per head per batch
    // QK^T: [s, hd] @ [hd, s] per head per batch
    // attn@V: [s, s] @ [s, hd] per head per batch
    flops += b * nh64 * matmulFlops(s64, hd64, s64); // QK^T
    flops += b * nh64 * matmulFlops(s64, s64, hd64); // attn@V

    // O projection: [bs, nh*hd] @ [d, nh*hd]^T
    flops += matmulFlops(bs64, nh64 * hd64, d64);

    // FFN: gate + up + down projections
    flops += matmulFlops(bs64, d64, ff64); // gate
    flops += matmulFlops(bs64, d64, ff64); // up
    flops += matmulFlops(bs64, ff64, d64); // down

    return flops;
}

/// Estimate total forward pass FLOPs (all layers + embedding + LM head).
pub fn forwardFlops(bs: usize, d: usize, nh: usize, nkv: usize, hd: usize, ff: usize, s: usize, num_layers: usize, vocab_size: usize) u64 {
    var flops: u64 = 0;

    // Per-layer
    flops += @as(u64, num_layers) * layerForwardFlops(bs, d, nh, nkv, hd, ff, s);

    // LM head: [bs, d] @ [vocab, d]^T
    flops += matmulFlops(@intCast(bs), @intCast(d), @intCast(vocab_size));

    return flops;
}

test "matmulFlops basic" {
    // 2x3 @ 3x4 = 2*2*3*4 = 48
    try @import("std").testing.expectEqual(@as(u64, 48), matmulFlops(2, 3, 4));
}
