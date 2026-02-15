const std = @import("std");
const miniz_port = @import("../miniz/build.zig");

pub const Spng = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    miniz: miniz_port.Miniz,
) Spng {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addCMacro("SPNG_USE_MINIZ", "");
    mod.addCMacro("SPNG_STATIC", "");

    const lib = b.addLibrary(.{
        .name = "spng",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.linkLibrary(miniz.lib);
    lib.addIncludePath(b.path("deps/spng/spng"));
    lib.addIncludePath(miniz.include_dir);
    lib.addCSourceFiles(.{
        .files = &.{"deps/spng/spng/spng.c"},
    });

    return .{ .lib = lib, .include_dir = b.path("deps/spng/spng") };
}
