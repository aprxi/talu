const std = @import("std");

pub const Sqlite3 = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Sqlite3 {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Compute/query focused embed:
    // - JSON1 needed for querying payload fields in SQL.
    // - No loadable extensions for safety/reproducibility.
    // - Single-threaded invocation model.
    mod.addCMacro("SQLITE_ENABLE_JSON1", "1");
    mod.addCMacro("SQLITE_OMIT_LOAD_EXTENSION", "1");
    mod.addCMacro("SQLITE_OMIT_SHARED_CACHE", "1");
    mod.addCMacro("SQLITE_THREADSAFE", "0");
    mod.addCMacro("SQLITE_DQS", "0");
    mod.addCMacro("SQLITE_DEFAULT_MEMSTATUS", "0");

    const lib = b.addLibrary(.{
        .name = "sqlite3",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.addIncludePath(b.path("deps/sqlite"));
    lib.addCSourceFiles(.{
        .files = &.{"deps/sqlite/sqlite3.c"},
    });

    return .{
        .lib = lib,
        .include_dir = b.path("deps/sqlite"),
    };
}
