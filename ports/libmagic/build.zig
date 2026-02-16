const std = @import("std");

pub const LibMagic = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) LibMagic {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addCMacro("HAVE_CONFIG_H", "");
    mod.addCMacro("_GNU_SOURCE", "");

    const lib = b.addLibrary(.{
        .name = "magic",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();

    // Generate config.h and magic.h via WriteFiles (libmagic uses autotools, no cmake template).
    // magic.h.in only needs X.YY replaced with the version string.
    // config.h defines POSIX features available on both Linux glibc 2.28+ and macOS.
    const generated = b.addWriteFiles();
    _ = generated.add("config.h",
        \\#ifndef CONFIG_H
        \\#define CONFIG_H
        \\#define HAVE_STDINT_H 1
        \\#define HAVE_INTTYPES_H 1
        \\#define HAVE_UNISTD_H 1
        \\#define HAVE_STDLIB_H 1
        \\#define HAVE_STRING_H 1
        \\#define HAVE_SYS_STAT_H 1
        \\#define HAVE_SYS_TYPES_H 1
        \\#define HAVE_FCNTL_H 1
        \\#define HAVE_LIMITS_H 1
        \\#define HAVE_SIGNAL_H 1
        \\#define HAVE_SYS_WAIT_H 1
        \\#define HAVE_VASPRINTF 1
        \\#define HAVE_ASPRINTF 1
        \\#define HAVE_DPRINTF 1
        \\#define HAVE_STRNDUP 1
        \\#define HAVE_PREAD 1
        \\#define HAVE_MMAP 1
        \\#define HAVE_GETLINE 1
        \\#define HAVE_STRCASESTR 1
        \\#define HAVE_CTIME_R 1
        \\#define HAVE_ASCTIME_R 1
        \\#define HAVE_GMTIME_R 1
        \\#define HAVE_LOCALTIME_R 1
        \\#define HAVE_FORK 1
        \\#define HAVE_VISIBILITY 1
        \\#define VERSION "5.46"
        \\#endif
    );
    lib.addIncludePath(generated.getDirectory());
    lib.addIncludePath(b.path("deps/file/src"));

    lib.addCSourceFiles(.{
        .files = &.{
            "deps/file/src/magic.c",
            "deps/file/src/apprentice.c",
            "deps/file/src/softmagic.c",
            "deps/file/src/ascmagic.c",
            "deps/file/src/encoding.c",
            "deps/file/src/compress.c",
            "deps/file/src/is_csv.c",
            "deps/file/src/is_json.c",
            "deps/file/src/is_simh.c",
            "deps/file/src/is_tar.c",
            "deps/file/src/readelf.c",
            "deps/file/src/print.c",
            "deps/file/src/fsmagic.c",
            "deps/file/src/funcs.c",
            "deps/file/src/buffer.c",
            "deps/file/src/cdf.c",
            "deps/file/src/cdf_time.c",
            "deps/file/src/readcdf.c",
            "deps/file/src/der.c",
            // fmtcheck not in glibc 2.28
            "deps/file/src/fmtcheck.c",
        },
    });

    // strlcpy/strlcat: only needed on Linux (glibc doesn't have them).
    // macOS provides these natively and defines them as fortified macros.
    if (target.result.os.tag == .linux) {
        lib.addCSourceFiles(.{
            .files = &.{
                "deps/file/src/strlcpy.c",
                "deps/file/src/strlcat.c",
            },
        });
    }

    return .{ .lib = lib, .include_dir = b.path("deps/file/src") };
}
