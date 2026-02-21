//! Integration tests for the inference/backend module.

test {
    _ = @import("backend_test.zig");
    _ = @import("kernel_symmetry_test.zig");
    _ = @import("model_contract_conformance_test.zig");
    _ = @import("cpu/root.zig");
    _ = @import("cuda/root.zig");
    _ = @import("metal/root.zig");
}
