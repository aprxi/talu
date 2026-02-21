//! Integration tests for the models module.

test {
    _ = @import("architecture_test.zig");
    _ = @import("metadata_contract_test.zig");
    _ = @import("onboarding_contract_test.zig");
    _ = @import("report.zig");
}
