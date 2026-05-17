//! Integration tests for the inference/transport module.

test {
    _ = @import("cuda_activation_test.zig");
    _ = @import("cuda_kv_mirror_test.zig");
}
