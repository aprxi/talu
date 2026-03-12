//! Integration tests for CUDA query-gated attention output gating plumbing.

const main = @import("main");

test "cuda gated-attention output-gate primitive is exported" {
    _ = main.compute.cuda.gated_attention_output_gate;
}
