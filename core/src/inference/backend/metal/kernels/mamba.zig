//! Metal Mamba kernel surface.

pub const supported = false;

pub const UnsupportedError = error{
    MambaNotSupportedOnMetal,
};

pub fn unsupported() UnsupportedError {
    return error.MambaNotSupportedOnMetal;
}

pub fn requireImplemented(comptime requested_by: []const u8) void {
    @compileError("Metal kernel 'mamba' is not implemented (requested by " ++ requested_by ++ ")");
}
