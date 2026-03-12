//! Integration tests for CUDA gated-delta query/key normalization plumbing.

const main = @import("main");

test "cuda gated-delta qk norm primitive is exported" {
    _ = main.compute.cuda.gated_delta_qk_norm;
}
