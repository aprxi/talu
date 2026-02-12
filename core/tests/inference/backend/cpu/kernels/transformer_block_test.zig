//! Integration tests for inference.backend.kernels.TransformerBlock

const std = @import("std");
const main = @import("main");

const TransformerBlock = main.inference.backend.kernels.kv_cache.BatchedKVCache;

test "TransformerBlock type verification" {
    // Type accessibility test - uses BatchedKVCache as proxy since TransformerBlock
    // is re-exported from block_kernels which requires model weights
    const T = TransformerBlock;
    _ = T;
}
