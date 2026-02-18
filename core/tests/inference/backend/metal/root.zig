//! Integration tests for the inference/backend/metal module.

test {
    _ = @import("executor_block_test.zig");
    _ = @import("executor_model_test.zig");
    _ = @import("executor_weights_test.zig");
    _ = @import("metal_backend_test.zig");
}
