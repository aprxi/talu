//! Template Integration Tests
//!
//! Integration tests for the template engine public API.
//! Tests types exported from core/src/template/root.zig.

const std = @import("std");

// Template engine types
pub const template_parser = @import("template_parser_test.zig");
pub const template_input = @import("template_input_test.zig");
pub const template_evaluator = @import("template_evaluator_test.zig");
pub const template_parser_internal = @import("template_parser_internal_test.zig");
pub const template_lexer = @import("template_lexer_test.zig");
pub const validation_result = @import("validation_result_test.zig");
pub const custom_filter_set = @import("custom_filter_set_test.zig");
pub const render_result = @import("render_result_test.zig");
pub const render_debug_result = @import("render_debug_result_test.zig");
pub const c_output_span_list = @import("c_output_span_list_test.zig");

test {
    std.testing.refAllDecls(@This());
}
