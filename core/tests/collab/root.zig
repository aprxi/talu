//! Integration tests for collaboration module.

test {
    _ = @import("resource_store_test.zig");
    _ = @import("session_store_test.zig");
    _ = @import("text_crdt_test.zig");
}
