//! Integration tests for the inference/backend module.

test {
    _ = @import("backend_test.zig");
    _ = @import("cpu/root.zig");
    _ = @import("metal/root.zig");
}
