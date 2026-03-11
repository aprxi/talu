//! Integration tests for CUDA gated-delta conv plumbing.

const main = @import("main");

test "cuda gated-delta conv primitive is exported" {
    _ = main.compute.cuda.gated_delta_conv;
}
