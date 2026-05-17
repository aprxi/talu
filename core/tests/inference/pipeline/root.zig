//! Integration tests for the inference pipeline module.

test {
    _ = @import("local_stage_contract_bundle_test.zig");
    _ = @import("local_stage_chain_test.zig");
    _ = @import("local_pipeline_runtime_test.zig");
    _ = @import("local_pipeline_stage_adapter_test.zig");
}
