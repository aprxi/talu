//! Image integration tests.

const std = @import("std");

pub const decode = @import("decode_test.zig");
pub const convert = @import("convert_test.zig");
pub const model_input = @import("model_input_test.zig");
pub const encode = @import("encode_test.zig");
pub const preprocess = @import("preprocess_test.zig");
pub const capi = @import("capi_image_test.zig");
pub const capi_file = @import("capi_file_test.zig");

test {
    std.testing.refAllDecls(@This());
}
