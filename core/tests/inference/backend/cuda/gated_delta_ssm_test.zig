//! Integration tests for CUDA gated-delta state-space plumbing.

const main = @import("main");

test "cuda gated-delta ssm primitive is exported" {
    _ = main.compute.cuda.gated_delta_ssm;
}
