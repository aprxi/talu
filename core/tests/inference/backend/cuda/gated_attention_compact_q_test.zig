//! Integration tests for CUDA query-gated attention compaction plumbing.

const main = @import("main");

test "cuda gated-attention compact-q primitive is exported" {
    _ = main.compute.cuda.gated_attention_compact_q;
}
