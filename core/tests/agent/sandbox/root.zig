//! Integration tests for agent sandbox module.

test {
    _ = @import("runtime_test.zig");
    _ = @import("profile_test.zig");
    _ = @import("detect_test.zig");
    _ = @import("cgroups_test.zig");
    _ = @import("probe_test.zig");
}

