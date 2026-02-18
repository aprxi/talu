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
