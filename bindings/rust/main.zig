const std = @import("std");

extern fn talu_cli_main() c_int;

pub fn main() void {
    const code = talu_cli_main();
    if (code != 0) {
        const exit_code: u8 = if (code < 0) 1 else if (code > 255) 255 else @intCast(code);
        std.process.exit(exit_code);
    }
}
