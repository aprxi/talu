//! graph/loader integration test module

const std = @import("std");

comptime {
    _ = @import("loaded_model_test.zig");
    _ = @import("reporter_test.zig");
}

test {
    std.testing.refAllDecls(@This());
}
