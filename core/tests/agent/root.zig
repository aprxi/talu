//! Integration tests for the agent module.

test {
    _ = @import("tool_registry_test.zig");
    _ = @import("context_test.zig");
    _ = @import("tools_test.zig");
    _ = @import("memory_test.zig");
}
