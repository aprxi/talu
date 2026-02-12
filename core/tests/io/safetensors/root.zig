//! io/safetensors integration test module

const std = @import("std");

comptime {
    _ = @import("safe_tensors_test.zig");
    _ = @import("sharded_safe_tensors_test.zig");
    _ = @import("unified_safe_tensors_test.zig");
    _ = @import("builder_test.zig");
}

test {
    std.testing.refAllDecls(@This());
}
