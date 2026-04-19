//! Converter integration test module

const std = @import("std");

comptime {
    _ = @import("f32_result_test.zig");
    _ = @import("weight_layout_map_test.zig");
}

test {
    std.testing.refAllDecls(@This());
}
