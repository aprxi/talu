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

    // Strip unused features for minimal binary size.
    // PinStore only uses: open/close/exec/prepare/step/bind/column/changes/errmsg.
    mod.addCMacro("SQLITE_OMIT_LOAD_EXTENSION", "1");
    mod.addCMacro("SQLITE_OMIT_JSON", "1");
    mod.addCMacro("SQLITE_OMIT_FTS3", "1");
    mod.addCMacro("SQLITE_OMIT_FTS4", "1");
    mod.addCMacro("SQLITE_OMIT_FTS5", "1");
    mod.addCMacro("SQLITE_OMIT_RTREE", "1");
    mod.addCMacro("SQLITE_OMIT_GEOPOLY", "1");
    mod.addCMacro("SQLITE_OMIT_DECLTYPE", "1");
    mod.addCMacro("SQLITE_OMIT_DEPRECATED", "1");
    mod.addCMacro("SQLITE_OMIT_SHARED_CACHE", "1");
    mod.addCMacro("SQLITE_OMIT_UTF16", "1");
    mod.addCMacro("SQLITE_OMIT_DESERIALIZE", "1");
    mod.addCMacro("SQLITE_OMIT_COMPLETE", "1");
    mod.addCMacro("SQLITE_OMIT_EXPLAIN", "1");
    mod.addCMacro("SQLITE_OMIT_AUTHORIZATION", "1");
    mod.addCMacro("SQLITE_OMIT_TRACE", "1");
    mod.addCMacro("SQLITE_OMIT_PROGRESS_CALLBACK", "1");
    mod.addCMacro("SQLITE_OMIT_GET_TABLE", "1");
    mod.addCMacro("SQLITE_THREADSAFE", "1");
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

    return .{ .lib = lib, .include_dir = b.path("deps/sqlite") };
}
