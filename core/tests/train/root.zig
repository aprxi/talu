//! Integration tests for the train module.

const std = @import("std");

pub const grad_tensor = @import("grad_tensor_test.zig");
pub const lora_adapter = @import("lora_adapter_test.zig");

test {
    std.testing.refAllDecls(@This());
}
