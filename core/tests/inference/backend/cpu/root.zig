//! Integration tests for the inference/backend/cpu module.

test {
    _ = @import("interface_stage_executor_test.zig");
    _ = @import("interface_transport_endpoint_test.zig");
    _ = @import("kernels/root.zig");
}
