//! Metal MLA kernel surface.

pub const supported = false;

pub const UnsupportedError = error{
    MLANotSupportedOnMetal,
};

pub fn unsupported() UnsupportedError {
    return error.MLANotSupportedOnMetal;
}

pub fn requireImplemented(comptime requested_by: []const u8) void {
    @compileError("Metal kernel 'mla_attention' is not implemented (requested by " ++ requested_by ++ ")");
}

test "unsupported returns explicit MLA unsupported error" {
    try @import("std").testing.expectError(error.MLANotSupportedOnMetal, unsupported());
}

test "requireImplemented symbol is present for compile-time enforcement" {
    try @import("std").testing.expect(@hasDecl(@This(), "requireImplemented"));
    const req = requireImplemented;
    _ = req;
}
