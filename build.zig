const std = @import("std");
const builtin = @import("builtin");

// =============================================================================
// FailStep - fails the build with a message
// =============================================================================

const FailStep = struct {
    step: std.Build.Step,
    message: []const u8,

    fn create(b: *std.Build, message: []const u8) *FailStep {
        const self = b.allocator.create(FailStep) catch @panic("OOM");
        self.* = .{
            .step = std.Build.Step.init(.{
                .id = .custom,
                .name = "fail",
                .owner = b,
                .makeFn = make,
            }),
            .message = message,
        };
        return self;
    }

    fn make(step: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
        const self: *FailStep = @fieldParentPtr("step", step);
        std.debug.print("{s}", .{self.message});
        return error.BuildFailed;
    }
};

// =============================================================================
// Version extraction from VERSION
// =============================================================================

fn getVersion(b: *std.Build) []const u8 {
    const content = std.fs.cwd().readFileAlloc(
        b.allocator,
        "VERSION",
        1024,
    ) catch return "0.0.0";

    const trimmed = std.mem.trim(u8, content, " \t\r\n");
    if (trimmed.len == 0) {
        return "0.0.0";
    }
    return b.allocator.dupe(u8, trimmed) catch return "0.0.0";
}

// =============================================================================
// PCRE2 dependency
// =============================================================================

const Pcre2 = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addPcre2(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Pcre2 {
    const config_header = b.addConfigHeader(
        .{
            .style = .{ .cmake = b.path("deps/pcre2/src/config-cmake.h.in") },
            .include_path = "config.h",
        },
        .{
            .HAVE_ASSERT_H = true,
            .HAVE_UNISTD_H = target.result.os.tag != .windows,
            .HAVE_WINDOWS_H = target.result.os.tag == .windows,
            .HAVE_ATTRIBUTE_UNINITIALIZED = true,
            .HAVE_BUILTIN_MUL_OVERFLOW = true,
            .HAVE_BUILTIN_UNREACHABLE = true,
            .SUPPORT_PCRE2_8 = true,
            .SUPPORT_PCRE2_16 = false,
            .SUPPORT_PCRE2_32 = false,
            .SUPPORT_UNICODE = true,
            .SUPPORT_JIT = false,
            .PCRE2_EXPORT = null,
            .PCRE2_LINK_SIZE = 2,
            .PCRE2_HEAP_LIMIT = 20000000,
            .PCRE2_MATCH_LIMIT = 10000000,
            .PCRE2_MATCH_LIMIT_DEPTH = "MATCH_LIMIT",
            .PCRE2_MAX_VARLOOKBEHIND = 255,
            .NEWLINE_DEFAULT = 2,
            .PCRE2_PARENS_NEST_LIMIT = 250,
            .PCRE2GREP_BUFSIZE = 20480,
            .PCRE2GREP_MAX_BUFSIZE = 1048576,
        },
    );

    const header_dir = b.addWriteFiles();
    _ = header_dir.addCopyFile(b.path("deps/pcre2/src/pcre2.h.generic"), "pcre2.h");

    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addCMacro("HAVE_CONFIG_H", "");
    mod.addCMacro("PCRE2_CODE_UNIT_WIDTH", "8");
    mod.addCMacro("PCRE2_STATIC", "");

    const lib = b.addLibrary(.{
        .name = "pcre2-8",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.addConfigHeader(config_header);
    lib.addIncludePath(header_dir.getDirectory());
    lib.addIncludePath(b.path("deps/pcre2/src"));

    const chartables = b.addWriteFiles();
    const chartables_file = chartables.addCopyFile(b.path("deps/pcre2/src/pcre2_chartables.c.dist"), "pcre2_chartables.c");
    lib.addCSourceFile(.{ .file = chartables_file });

    lib.addCSourceFiles(.{
        .files = &.{
            "deps/pcre2/src/pcre2_auto_possess.c",
            "deps/pcre2/src/pcre2_chkdint.c",
            "deps/pcre2/src/pcre2_compile.c",
            "deps/pcre2/src/pcre2_compile_cgroup.c",
            "deps/pcre2/src/pcre2_compile_class.c",
            "deps/pcre2/src/pcre2_config.c",
            "deps/pcre2/src/pcre2_context.c",
            "deps/pcre2/src/pcre2_convert.c",
            "deps/pcre2/src/pcre2_dfa_match.c",
            "deps/pcre2/src/pcre2_error.c",
            "deps/pcre2/src/pcre2_extuni.c",
            "deps/pcre2/src/pcre2_find_bracket.c",
            "deps/pcre2/src/pcre2_jit_compile.c",
            "deps/pcre2/src/pcre2_maketables.c",
            "deps/pcre2/src/pcre2_match.c",
            "deps/pcre2/src/pcre2_match_data.c",
            "deps/pcre2/src/pcre2_match_next.c",
            "deps/pcre2/src/pcre2_newline.c",
            "deps/pcre2/src/pcre2_ord2utf.c",
            "deps/pcre2/src/pcre2_pattern_info.c",
            "deps/pcre2/src/pcre2_script_run.c",
            "deps/pcre2/src/pcre2_serialize.c",
            "deps/pcre2/src/pcre2_string_utils.c",
            "deps/pcre2/src/pcre2_study.c",
            "deps/pcre2/src/pcre2_substitute.c",
            "deps/pcre2/src/pcre2_substring.c",
            "deps/pcre2/src/pcre2_tables.c",
            "deps/pcre2/src/pcre2_ucd.c",
            "deps/pcre2/src/pcre2_valid_utf.c",
            "deps/pcre2/src/pcre2_xclass.c",
        },
    });

    return .{ .lib = lib, .include_dir = header_dir.getDirectory() };
}

// =============================================================================
// SQLite3 dependency (amalgamation, minimal build)
// =============================================================================

const Sqlite3 = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addSqlite3(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Sqlite3 {
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

// =============================================================================
// Miniz dependency (zip/deflate, amalgamation)
// =============================================================================

const Miniz = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addMiniz(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Miniz {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    // Static build — MINIZ_EXPORT is empty (normally generated by CMake/Meson)
    mod.addCMacro("MINIZ_EXPORT", "");

    const lib = b.addLibrary(.{
        .name = "miniz",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.addIncludePath(b.path("deps/miniz"));
    lib.addCSourceFiles(.{
        .files = &.{
            "deps/miniz/miniz.c",
            "deps/miniz/miniz_tdef.c",
            "deps/miniz/miniz_tinfl.c",
            "deps/miniz/miniz_zip.c",
        },
    });

    return .{ .lib = lib, .include_dir = b.path("deps/miniz") };
}

// =============================================================================
// JPEG (libjpeg-turbo), PNG (libspng), and WebP (libwebp)
// =============================================================================

const JpegTurbo = struct {
    lib: *std.Build.Step.Compile,
    jpeg12_lib: *std.Build.Step.Compile,
    jpeg16_lib: *std.Build.Step.Compile,
    turbojpeg12_lib: *std.Build.Step.Compile,
    turbojpeg16_lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
    generated_include_dir: std.Build.LazyPath,
};

fn addJpegTurbo(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) JpegTurbo {
    const generated = b.addWriteFiles();
    _ = generated.add("jconfig.h",
        \\/* Version ID for the JPEG library. */
        \\#define JPEG_LIB_VERSION  62
        \\/* libjpeg-turbo version */
        \\#define LIBJPEG_TURBO_VERSION  3.1.3
        \\/* libjpeg-turbo version in integer form */
        \\#define LIBJPEG_TURBO_VERSION_NUMBER  3001003
        \\/* Support arithmetic encoding when using 8-bit samples */
        \\#define C_ARITH_CODING_SUPPORTED 1
        \\/* Support arithmetic decoding when using 8-bit samples */
        \\#define D_ARITH_CODING_SUPPORTED 1
        \\/* Support in-memory source/destination managers */
        \\#define MEM_SRCDST_SUPPORTED  1
        \\/* Use accelerated SIMD routines when using 8-bit samples */
        \\/* #undef WITH_SIMD */
        \\#ifndef BITS_IN_JSAMPLE
        \\#define BITS_IN_JSAMPLE  8
        \\#endif
        \\#ifdef _WIN32
        \\#undef RIGHT_SHIFT_IS_UNSIGNED
        \\#ifndef __RPCNDR_H__
        \\typedef unsigned char boolean;
        \\#endif
        \\#define HAVE_BOOLEAN
        \\#if !(defined(_BASETSD_H_) || defined(_BASETSD_H))
        \\typedef short INT16;
        \\typedef signed int INT32;
        \\#endif
        \\#define XMD_H
        \\#else
        \\/* #undef RIGHT_SHIFT_IS_UNSIGNED */
        \\#endif
    );
    _ = generated.add("jconfigint.h",
        \\/* libjpeg-turbo build number */
        \\#define BUILD  "20260215"
        \\/* How to hide global symbols. */
        \\#define HIDDEN  __attribute__((visibility("hidden")))
        \\/* Compiler's inline keyword */
        \\#undef inline
        \\/* How to obtain function inlining. */
        \\#define INLINE  __inline__ __attribute__((always_inline))
        \\/* How to obtain thread-local storage */
        \\#define THREAD_LOCAL  __thread
        \\/* Define to the full name of this package. */
        \\#define PACKAGE_NAME  "libjpeg-turbo"
        \\/* Version number of package */
        \\#define VERSION  "3.1.3"
        \\/* The size of `size_t', as computed by sizeof. */
        \\#define SIZEOF_SIZE_T  8
        \\/* Define if compiler has __builtin_ctzl() and sizeof(unsigned long) == sizeof(size_t). */
        \\#define HAVE_BUILTIN_CTZL
        \\/* Define to 1 if you have the <intrin.h> header file. */
        \\/* #undef HAVE_INTRIN_H */
        \\#if defined(_MSC_VER) && defined(HAVE_INTRIN_H)
        \\#if (SIZEOF_SIZE_T == 8)
        \\#define HAVE_BITSCANFORWARD64
        \\#elif (SIZEOF_SIZE_T == 4)
        \\#define HAVE_BITSCANFORWARD
        \\#endif
        \\#endif
        \\#if defined(__has_attribute)
        \\#if __has_attribute(fallthrough)
        \\#define FALLTHROUGH  __attribute__((fallthrough));
        \\#else
        \\#define FALLTHROUGH
        \\#endif
        \\#else
        \\#define FALLTHROUGH
        \\#endif
        \\#ifndef BITS_IN_JSAMPLE
        \\#define BITS_IN_JSAMPLE  8
        \\#endif
        \\#undef C_ARITH_CODING_SUPPORTED
        \\#undef D_ARITH_CODING_SUPPORTED
        \\#undef WITH_SIMD
        \\#if BITS_IN_JSAMPLE == 8
        \\#define C_ARITH_CODING_SUPPORTED 1
        \\#define D_ARITH_CODING_SUPPORTED 1
        \\/* #undef WITH_SIMD */
        \\#endif
    );
    _ = generated.add("jversion.h",
        \\#if JPEG_LIB_VERSION >= 80
        \\#define JVERSION        "8d  15-Jan-2012"
        \\#elif JPEG_LIB_VERSION >= 70
        \\#define JVERSION        "7  27-Jun-2009"
        \\#else
        \\#define JVERSION        "6b  27-Mar-1998"
        \\#endif
        \\#define JCOPYRIGHT1 \
        \\  "Copyright (C) 2009-2024 D. R. Commander\n" \
        \\  "Copyright (C) 2015, 2020 Google, Inc.\n" \
        \\  "Copyright (C) 2019-2020 Arm Limited\n" \
        \\  "Copyright (C) 2015-2016, 2018 Matthieu Darbois\n" \
        \\  "Copyright (C) 2011-2016 Siarhei Siamashka\n" \
        \\  "Copyright (C) 2015 Intel Corporation\n"
        \\#define JCOPYRIGHT2 \
        \\  "Copyright (C) 2013-2014 Linaro Limited\n" \
        \\  "Copyright (C) 2013-2014 MIPS Technologies, Inc.\n" \
        \\  "Copyright (C) 2009, 2012 Pierre Ossman for Cendio AB\n" \
        \\  "Copyright (C) 2009-2011 Nokia Corporation and/or its subsidiary(-ies)\n" \
        \\  "Copyright (C) 1999-2006 MIYASAKA Masaru\n" \
        \\  "Copyright (C) 1999 Ken Murchison\n" \
        \\  "Copyright (C) 1991-2020 Thomas G. Lane, Guido Vollbeding\n"
        \\#define JCOPYRIGHT_SHORT \
        \\  "Copyright (C) 1991-2024 The libjpeg-turbo Project and many others"
    );

    const generated_include_dir = generated.getDirectory();
    const include_dir = b.path("deps/jpeg-turbo/src");

    const turbo_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const lib = b.addLibrary(.{
        .name = "turbojpeg",
        .root_module = turbo_mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.addIncludePath(generated_include_dir);
    lib.addIncludePath(include_dir);
    lib.addCSourceFiles(.{
        .files = &.{
            // Common JPEG sources (8-bit precision build)
            "deps/jpeg-turbo/src/jcapistd.c",
            "deps/jpeg-turbo/src/jccolor.c",
            "deps/jpeg-turbo/src/jcdiffct.c",
            "deps/jpeg-turbo/src/jclossls.c",
            "deps/jpeg-turbo/src/jcmainct.c",
            "deps/jpeg-turbo/src/jcprepct.c",
            "deps/jpeg-turbo/src/jcsample.c",
            "deps/jpeg-turbo/src/jdapistd.c",
            "deps/jpeg-turbo/src/jdcolor.c",
            "deps/jpeg-turbo/src/jddiffct.c",
            "deps/jpeg-turbo/src/jdlossls.c",
            "deps/jpeg-turbo/src/jdmainct.c",
            "deps/jpeg-turbo/src/jdpostct.c",
            "deps/jpeg-turbo/src/jdsample.c",
            "deps/jpeg-turbo/src/jutils.c",
            "deps/jpeg-turbo/src/jccoefct.c",
            "deps/jpeg-turbo/src/jcdctmgr.c",
            "deps/jpeg-turbo/src/jdcoefct.c",
            "deps/jpeg-turbo/src/jddctmgr.c",
            "deps/jpeg-turbo/src/jdmerge.c",
            "deps/jpeg-turbo/src/jfdctfst.c",
            "deps/jpeg-turbo/src/jfdctint.c",
            "deps/jpeg-turbo/src/jidctflt.c",
            "deps/jpeg-turbo/src/jidctfst.c",
            "deps/jpeg-turbo/src/jidctint.c",
            "deps/jpeg-turbo/src/jidctred.c",
            "deps/jpeg-turbo/src/jquant1.c",
            "deps/jpeg-turbo/src/jquant2.c",
            "deps/jpeg-turbo/src/jcapimin.c",
            "deps/jpeg-turbo/src/jchuff.c",
            "deps/jpeg-turbo/src/jcicc.c",
            "deps/jpeg-turbo/src/jcinit.c",
            "deps/jpeg-turbo/src/jclhuff.c",
            "deps/jpeg-turbo/src/jcmarker.c",
            "deps/jpeg-turbo/src/jcmaster.c",
            "deps/jpeg-turbo/src/jcomapi.c",
            "deps/jpeg-turbo/src/jcparam.c",
            "deps/jpeg-turbo/src/jcphuff.c",
            "deps/jpeg-turbo/src/jctrans.c",
            "deps/jpeg-turbo/src/jdapimin.c",
            "deps/jpeg-turbo/src/jdatadst.c",
            "deps/jpeg-turbo/src/jdatasrc.c",
            "deps/jpeg-turbo/src/jdhuff.c",
            "deps/jpeg-turbo/src/jdicc.c",
            "deps/jpeg-turbo/src/jdinput.c",
            "deps/jpeg-turbo/src/jdlhuff.c",
            "deps/jpeg-turbo/src/jdmarker.c",
            "deps/jpeg-turbo/src/jdmaster.c",
            "deps/jpeg-turbo/src/jdphuff.c",
            "deps/jpeg-turbo/src/jdtrans.c",
            "deps/jpeg-turbo/src/jerror.c",
            "deps/jpeg-turbo/src/jfdctflt.c",
            "deps/jpeg-turbo/src/jmemmgr.c",
            "deps/jpeg-turbo/src/jmemnobs.c",
            "deps/jpeg-turbo/src/jpeg_nbits.c",
            "deps/jpeg-turbo/src/jaricom.c",
            "deps/jpeg-turbo/src/jcarith.c",
            "deps/jpeg-turbo/src/jdarith.c",
            // TurboJPEG API + helpers
            "deps/jpeg-turbo/src/turbojpeg.c",
            "deps/jpeg-turbo/src/transupp.c",
            "deps/jpeg-turbo/src/jdatadst-tj.c",
            "deps/jpeg-turbo/src/jdatasrc-tj.c",
            "deps/jpeg-turbo/src/rdbmp.c",
            "deps/jpeg-turbo/src/rdppm.c",
            "deps/jpeg-turbo/src/wrbmp.c",
            "deps/jpeg-turbo/src/wrppm.c",
        },
        .flags = &.{
            "-fPIC",
            "-DBMP_SUPPORTED",
            "-DPPM_SUPPORTED",
        },
    });

    const jpeg12_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const jpeg12_lib = b.addLibrary(.{
        .name = "jpeg12",
        .root_module = jpeg12_mod,
        .linkage = .static,
    });
    jpeg12_lib.linkLibC();
    jpeg12_lib.addIncludePath(generated_include_dir);
    jpeg12_lib.addIncludePath(include_dir);
    jpeg12_lib.addCSourceFiles(.{
        .files = &.{
            "deps/jpeg-turbo/src/jcapistd.c",
            "deps/jpeg-turbo/src/jccolor.c",
            "deps/jpeg-turbo/src/jcdiffct.c",
            "deps/jpeg-turbo/src/jclossls.c",
            "deps/jpeg-turbo/src/jcmainct.c",
            "deps/jpeg-turbo/src/jcprepct.c",
            "deps/jpeg-turbo/src/jcsample.c",
            "deps/jpeg-turbo/src/jdapistd.c",
            "deps/jpeg-turbo/src/jdcolor.c",
            "deps/jpeg-turbo/src/jddiffct.c",
            "deps/jpeg-turbo/src/jdlossls.c",
            "deps/jpeg-turbo/src/jdmainct.c",
            "deps/jpeg-turbo/src/jdpostct.c",
            "deps/jpeg-turbo/src/jdsample.c",
            "deps/jpeg-turbo/src/jutils.c",
            "deps/jpeg-turbo/src/jccoefct.c",
            "deps/jpeg-turbo/src/jcdctmgr.c",
            "deps/jpeg-turbo/src/jdcoefct.c",
            "deps/jpeg-turbo/src/jddctmgr.c",
            "deps/jpeg-turbo/src/jdmerge.c",
            "deps/jpeg-turbo/src/jfdctfst.c",
            "deps/jpeg-turbo/src/jfdctint.c",
            "deps/jpeg-turbo/src/jidctflt.c",
            "deps/jpeg-turbo/src/jidctfst.c",
            "deps/jpeg-turbo/src/jidctint.c",
            "deps/jpeg-turbo/src/jidctred.c",
            "deps/jpeg-turbo/src/jquant1.c",
            "deps/jpeg-turbo/src/jquant2.c",
        },
        .flags = &.{
            "-fPIC",
            "-DBITS_IN_JSAMPLE=12",
        },
    });

    const jpeg16_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const jpeg16_lib = b.addLibrary(.{
        .name = "jpeg16",
        .root_module = jpeg16_mod,
        .linkage = .static,
    });
    jpeg16_lib.linkLibC();
    jpeg16_lib.addIncludePath(generated_include_dir);
    jpeg16_lib.addIncludePath(include_dir);
    jpeg16_lib.addCSourceFiles(.{
        .files = &.{
            "deps/jpeg-turbo/src/jcapistd.c",
            "deps/jpeg-turbo/src/jccolor.c",
            "deps/jpeg-turbo/src/jcdiffct.c",
            "deps/jpeg-turbo/src/jclossls.c",
            "deps/jpeg-turbo/src/jcmainct.c",
            "deps/jpeg-turbo/src/jcprepct.c",
            "deps/jpeg-turbo/src/jcsample.c",
            "deps/jpeg-turbo/src/jdapistd.c",
            "deps/jpeg-turbo/src/jdcolor.c",
            "deps/jpeg-turbo/src/jddiffct.c",
            "deps/jpeg-turbo/src/jdlossls.c",
            "deps/jpeg-turbo/src/jdmainct.c",
            "deps/jpeg-turbo/src/jdpostct.c",
            "deps/jpeg-turbo/src/jdsample.c",
            "deps/jpeg-turbo/src/jutils.c",
        },
        .flags = &.{
            "-fPIC",
            "-DBITS_IN_JSAMPLE=16",
        },
    });

    const turbojpeg12_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const turbojpeg12_lib = b.addLibrary(.{
        .name = "turbojpeg12",
        .root_module = turbojpeg12_mod,
        .linkage = .static,
    });
    turbojpeg12_lib.linkLibC();
    turbojpeg12_lib.addIncludePath(generated_include_dir);
    turbojpeg12_lib.addIncludePath(include_dir);
    turbojpeg12_lib.addCSourceFiles(.{
        .files = &.{
            "deps/jpeg-turbo/src/rdppm.c",
            "deps/jpeg-turbo/src/wrppm.c",
        },
        .flags = &.{
            "-fPIC",
            "-DBITS_IN_JSAMPLE=12",
            "-DPPM_SUPPORTED",
        },
    });

    const turbojpeg16_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const turbojpeg16_lib = b.addLibrary(.{
        .name = "turbojpeg16",
        .root_module = turbojpeg16_mod,
        .linkage = .static,
    });
    turbojpeg16_lib.linkLibC();
    turbojpeg16_lib.addIncludePath(generated_include_dir);
    turbojpeg16_lib.addIncludePath(include_dir);
    turbojpeg16_lib.addCSourceFiles(.{
        .files = &.{
            "deps/jpeg-turbo/src/rdppm.c",
            "deps/jpeg-turbo/src/wrppm.c",
        },
        .flags = &.{
            "-fPIC",
            "-DBITS_IN_JSAMPLE=16",
            "-DPPM_SUPPORTED",
        },
    });

    lib.linkLibrary(jpeg12_lib);
    lib.linkLibrary(jpeg16_lib);
    lib.linkLibrary(turbojpeg12_lib);
    lib.linkLibrary(turbojpeg16_lib);

    return .{
        .lib = lib,
        .jpeg12_lib = jpeg12_lib,
        .jpeg16_lib = jpeg16_lib,
        .turbojpeg12_lib = turbojpeg12_lib,
        .turbojpeg16_lib = turbojpeg16_lib,
        .include_dir = include_dir,
        .generated_include_dir = generated_include_dir,
    };
}

const Spng = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addSpng(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    miniz: Miniz,
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

const Webp = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addWebp(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) Webp {
    const target_arch = target.result.cpu.arch;
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (target_arch == .x86_64) {
        mod.addCMacro("WEBP_HAVE_SSE2", "1");
        mod.addCMacro("WEBP_HAVE_SSE41", "1");
    } else if (target_arch == .aarch64) {
        mod.addCMacro("WEBP_HAVE_NEON", "1");
    }

    const lib = b.addLibrary(.{
        .name = "webp",
        .root_module = mod,
        .linkage = .static,
    });
    lib.linkLibC();
    lib.addIncludePath(b.path("deps/webp"));
    lib.addIncludePath(b.path("deps/webp/src"));

    lib.addCSourceFiles(.{
        .files = &.{
            "deps/webp/src/dec/alpha_dec.c",
            "deps/webp/src/dec/buffer_dec.c",
            "deps/webp/src/dec/frame_dec.c",
            "deps/webp/src/dec/idec_dec.c",
            "deps/webp/src/dec/io_dec.c",
            "deps/webp/src/dec/quant_dec.c",
            "deps/webp/src/dec/tree_dec.c",
            "deps/webp/src/dec/vp8_dec.c",
            "deps/webp/src/dec/vp8l_dec.c",
            "deps/webp/src/dec/webp_dec.c",
            "deps/webp/src/dsp/alpha_processing.c",
            "deps/webp/src/dsp/cpu.c",
            "deps/webp/src/dsp/dec.c",
            "deps/webp/src/dsp/dec_clip_tables.c",
            "deps/webp/src/dsp/filters.c",
            "deps/webp/src/dsp/lossless.c",
            "deps/webp/src/dsp/rescaler.c",
            "deps/webp/src/dsp/upsampling.c",
            "deps/webp/src/dsp/yuv.c",
            "deps/webp/src/utils/bit_reader_utils.c",
            "deps/webp/src/utils/color_cache_utils.c",
            "deps/webp/src/utils/filters_utils.c",
            "deps/webp/src/utils/huffman_utils.c",
            "deps/webp/src/utils/palette.c",
            "deps/webp/src/utils/quant_levels_dec_utils.c",
            "deps/webp/src/utils/random_utils.c",
            "deps/webp/src/utils/rescaler_utils.c",
            "deps/webp/src/utils/thread_utils.c",
            "deps/webp/src/utils/utils.c",
        },
    });

    if (target_arch == .x86_64) {
        lib.addCSourceFiles(.{
            .files = &.{
                "deps/webp/src/dsp/alpha_processing_sse2.c",
                "deps/webp/src/dsp/dec_sse2.c",
                "deps/webp/src/dsp/filters_sse2.c",
                "deps/webp/src/dsp/lossless_sse2.c",
                "deps/webp/src/dsp/rescaler_sse2.c",
                "deps/webp/src/dsp/upsampling_sse2.c",
                "deps/webp/src/dsp/yuv_sse2.c",
            },
            .flags = &.{"-msse2"},
        });
        lib.addCSourceFiles(.{
            .files = &.{
                "deps/webp/src/dsp/alpha_processing_sse41.c",
                "deps/webp/src/dsp/dec_sse41.c",
                "deps/webp/src/dsp/lossless_sse41.c",
                "deps/webp/src/dsp/upsampling_sse41.c",
                "deps/webp/src/dsp/yuv_sse41.c",
            },
            .flags = &.{"-msse4.1"},
        });
    } else if (target_arch == .aarch64) {
        lib.addCSourceFiles(.{
            .files = &.{
                "deps/webp/src/dsp/alpha_processing_neon.c",
                "deps/webp/src/dsp/dec_neon.c",
                "deps/webp/src/dsp/filters_neon.c",
                "deps/webp/src/dsp/lossless_neon.c",
                "deps/webp/src/dsp/rescaler_neon.c",
                "deps/webp/src/dsp/upsampling_neon.c",
                "deps/webp/src/dsp/yuv_neon.c",
            },
        });
    }

    return .{ .lib = lib, .include_dir = b.path("deps/webp/src") };
}

// =============================================================================
// Libmagic dependency (file type detection)
// =============================================================================

const LibMagic = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

fn addLibMagic(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) LibMagic {
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
            // Compat implementations for functions not in glibc 2.28
            "deps/file/src/fmtcheck.c",
            "deps/file/src/strlcpy.c",
            "deps/file/src/strlcat.c",
        },
    });

    return .{ .lib = lib, .include_dir = b.path("deps/file/src") };
}

// =============================================================================
// Helper to add C dependencies to a module
// =============================================================================

fn addCDependencies(
    b: *std.Build,
    mod: *std.Build.Module,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
) void {
    mod.addIncludePath(b.path("src/text"));
    mod.addIncludePath(b.path("include"));
    mod.addIncludePath(b.path("deps/utf8proc"));
    mod.addIncludePath(pcre2.include_dir);
    mod.addIncludePath(b.path("deps/pcre2/src"));
    mod.addIncludePath(b.path("deps/curl/include"));
    mod.addIncludePath(miniz.include_dir);
    mod.addIncludePath(libmagic.include_dir);
    mod.addIncludePath(jpeg_turbo.generated_include_dir);
    mod.addIncludePath(jpeg_turbo.include_dir);
    mod.addIncludePath(spng.include_dir);
    mod.addIncludePath(webp.include_dir);

    // Mozilla CA certificates bundle for HTTPS - generated during `make deps`
    mod.addImport("cacert", b.createModule(.{
        .root_source_file = b.path("deps/cacert.zig"),
    }));

    // Compiled magic database for file type detection - generated during `make deps`
    mod.addImport("magic_db", b.createModule(.{
        .root_source_file = b.path("deps/magic_db.zig"),
    }));
}

fn linkCDependencies(
    b: *std.Build,
    artifact: *std.Build.Step.Compile,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
    comptime skip_external_archives: bool,
) void {
    artifact.linkLibC();
    artifact.linkLibrary(pcre2.lib);
    artifact.linkLibrary(miniz.lib);
    artifact.linkLibrary(libmagic.lib);
    artifact.linkLibrary(jpeg_turbo.lib);
    artifact.linkLibrary(jpeg_turbo.jpeg12_lib);
    artifact.linkLibrary(jpeg_turbo.jpeg16_lib);
    artifact.linkLibrary(jpeg_turbo.turbojpeg12_lib);
    artifact.linkLibrary(jpeg_turbo.turbojpeg16_lib);
    artifact.linkLibrary(spng.lib);
    artifact.linkLibrary(webp.lib);

    // For static libraries, skip external .a archives to avoid nested archive warnings.
    // The final executable/shared lib will link them directly.
    if (!skip_external_archives) {
        // Link pre-built libcurl static library
        artifact.addObjectFile(b.path("deps/curl/build/lib/libcurl.a"));

        // Link mbedTLS for HTTPS support (used on all platforms)
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedtls.a"));
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedx509.a"));
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedcrypto.a"));

        const target_os = artifact.rootModuleTarget().os.tag;
        if (target_os == .macos) {
            artifact.linkFramework("CoreFoundation");
            artifact.linkFramework("SystemConfiguration");
            artifact.linkFramework("Security");
        }
        artifact.linkSystemLibrary("pthread");
    }

    artifact.addCSourceFiles(.{
        .files = &.{"deps/utf8proc/utf8proc.c"},
        .flags = &.{"-std=gnu11"},
    });

    // Signal guard for graceful SIGBUS handling
    const target_os = artifact.rootModuleTarget().os.tag;
    if (target_os == .linux or target_os == .macos) {
        artifact.addCSourceFiles(.{
            .files = &.{"core/src/capi/signal_guard.c"},
            .flags = &.{"-std=gnu11"},
        });
    }
}

/// Link only external archives (.a files) and system libraries.
/// Use this when C sources are already compiled into a static lib being linked.
fn linkExternalArchives(
    b: *std.Build,
    artifact: *std.Build.Step.Compile,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
) void {
    artifact.linkLibC();
    artifact.linkLibrary(pcre2.lib);
    artifact.linkLibrary(miniz.lib);
    artifact.linkLibrary(libmagic.lib);
    artifact.linkLibrary(jpeg_turbo.lib);
    artifact.linkLibrary(jpeg_turbo.jpeg12_lib);
    artifact.linkLibrary(jpeg_turbo.jpeg16_lib);
    artifact.linkLibrary(jpeg_turbo.turbojpeg12_lib);
    artifact.linkLibrary(jpeg_turbo.turbojpeg16_lib);
    artifact.linkLibrary(spng.lib);
    artifact.linkLibrary(webp.lib);

    // Link pre-built libcurl static library
    artifact.addObjectFile(b.path("deps/curl/build/lib/libcurl.a"));

    // Link mbedTLS for HTTPS support (used on all platforms)
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedtls.a"));
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedx509.a"));
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedcrypto.a"));

    const target_os = artifact.rootModuleTarget().os.tag;
    if (target_os == .macos) {
        artifact.linkFramework("CoreFoundation");
        artifact.linkFramework("SystemConfiguration");
        artifact.linkFramework("Security");
    }
    artifact.linkSystemLibrary("pthread");
}

// =============================================================================
// Metal GPU support (macOS only)
// =============================================================================

fn addMetalSupport(
    b: *std.Build,
    mod: *std.Build.Module,
    artifact: *std.Build.Step.Compile,
    enable_metal: bool,
) void {
    if (artifact.rootModuleTarget().os.tag != .macos) return;
    if (!enable_metal) return;

    mod.addIncludePath(b.path("core/src/compute/metal"));
    mod.addIncludePath(b.path("deps/mlx/include"));

    artifact.linkFramework("Metal");
    artifact.linkFramework("MetalPerformanceShaders");
    artifact.linkFramework("Foundation");
    artifact.linkFramework("Accelerate");

    artifact.addObjectFile(b.path("deps/mlx/lib/libmlx.a"));

    artifact.addCSourceFiles(.{
        .files = &.{
            "core/src/compute/metal/device.m",
            "core/src/compute/metal/matmul.m",
        },
        .flags = &.{
            "-std=c11",
            "-fobjc-arc",
            "-fno-objc-exceptions",
        },
    });

    artifact.addCSourceFiles(.{
        .files = &.{
            "core/src/compute/metal/mlx/array_pool.cpp",
            "core/src/compute/metal/mlx/ops.cpp",
            "core/src/compute/metal/mlx/fused_ops.cpp",
            "core/src/compute/metal/mlx/cache.cpp",
            "core/src/compute/metal/mlx/model_dense.cpp",
            "core/src/compute/metal/mlx/model_quantized.cpp",
        },
        .flags = &.{
            "-std=c++17",
        },
    });

    artifact.linkLibCpp();
}

// =============================================================================
// Embedded Graphs Generator
// =============================================================================

/// Generate embedded_graphs.zig module from JSON files in tools/archs/_graphs/
/// This embeds architecture definitions directly into the binary for distribution.
fn generateEmbeddedGraphs(b: *std.Build) *std.Build.Module {
    const graphs_dir_path = "tools/archs/_graphs";

    // Use WriteFiles to generate the embedded_graphs.zig source
    const gen_step = b.addWriteFiles();

    // Build the generated Zig source code using ArrayListUnmanaged
    var code = std.ArrayListUnmanaged(u8){};

    // Write module header
    code.appendSlice(b.allocator,
        \\//! Embedded Architecture Graphs
        \\//!
        \\//! Auto-generated from bindings/python/talu/_graphs/*.json
        \\//! Do not edit manually - regenerate with `zig build`.
        \\
        \\const std = @import("std");
        \\
        \\/// Map of architecture names to their JSON definitions.
        \\/// Use `get()` for O(1) lookups.
        \\pub const graphs = std.StaticStringMap([]const u8).initComptime(.{
        \\
    ) catch @panic("OOM");

    // Scan the _graphs directory and embed each JSON file
    var graphs_found: usize = 0;
    if (std.fs.cwd().openDir(graphs_dir_path, .{ .iterate = true })) |dir| {
        var graphs_dir = dir;
        defer graphs_dir.close();

        var iter = graphs_dir.iterate();
        while (iter.next() catch null) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

            // Read the JSON file content
            const json_content = graphs_dir.readFileAlloc(b.allocator, entry.name, 1024 * 1024) catch continue;

            // Get the architecture name (filename without .json extension)
            const arch_name = entry.name[0 .. entry.name.len - 5]; // strip ".json"

            // Write the map entry with escaped JSON content using multiline string literal
            const entry_start = std.fmt.allocPrint(b.allocator, "    .{{ \"{s}\", \n        \\\\", .{arch_name}) catch @panic("OOM");
            code.appendSlice(b.allocator, entry_start) catch @panic("OOM");

            // Write JSON content as Zig multiline string literal
            // Only newlines need special handling - backslashes stay raw in multiline strings
            for (json_content) |c| {
                switch (c) {
                    '\n' => code.appendSlice(b.allocator, "\n        \\\\") catch @panic("OOM"),
                    else => code.append(b.allocator, c) catch @panic("OOM"),
                }
            }
            code.appendSlice(b.allocator, "\n    },\n") catch @panic("OOM");
            graphs_found += 1;
        }
    } else |_| {
        // Directory doesn't exist - emit empty map (allows build without graphs)
        std.debug.print("Warning: {s} not found, building without embedded graphs\n", .{graphs_dir_path});
    }

    // Close the map and add helper functions
    code.appendSlice(b.allocator,
        \\});
        \\
        \\/// Get an embedded graph by name.
        \\pub fn get(name: []const u8) ?[]const u8 {
        \\    return graphs.get(name);
        \\}
        \\
        \\/// Returns true if any graphs are embedded.
        \\pub fn hasGraphs() bool {
        \\
    ) catch @panic("OOM");

    const has_graphs_str = if (graphs_found > 0) "    return true;\n" else "    return false;\n";
    code.appendSlice(b.allocator, has_graphs_str) catch @panic("OOM");

    code.appendSlice(b.allocator,
        \\}
        \\
    ) catch @panic("OOM");

    // Add the generated source to the WriteFiles step
    const source_file = gen_step.add("embedded_graphs.zig", code.items);

    // Create and return the module
    return b.createModule(.{
        .root_source_file = source_file,
    });
}

// =============================================================================
// Main build function
// =============================================================================

pub fn build(b: *std.Build) void {
    // --------------------------------------------------------------------------
    // TARGET CONFIGURATION
    // --------------------------------------------------------------------------
    // Use host platform by default, but enforce GLIBC 2.28 on Linux for distro
    // compatibility (RHEL 8, Ubuntu 18.04, etc.).
    //
    // On x86_64 Linux, use -Dcpu=x86_64_v3 (via Makefile) for AVX2 support.
    // This fixes "couldn't allocate output register" errors with SIMD code.
    var target = b.standardTargetOptions(.{});

    // For Linux targets: enforce GLIBC 2.28 for broad distro compatibility
    if (target.result.os.tag == .linux) {
        var query = target.query;
        query.glibc_version = .{ .major = 2, .minor = 28, .patch = 0 };
        if (query.abi == null) query.abi = .gnu;
        target = b.resolveTargetQuery(query);

        // Debug output for Linux builds
        const glibc = target.result.os.version_range.linux.glibc;
        const cpu_model_name = target.result.cpu.model.name;
        std.debug.print("Build Target: {s}-{s}-{s} (GLIBC {d}.{d}) [CPU: {s}]\n", .{
            @tagName(target.result.cpu.arch),
            @tagName(target.result.os.tag),
            @tagName(target.result.abi),
            glibc.major,
            glibc.minor,
            cpu_model_name,
        });
    }

    // --------------------------------------------------------------------------
    // OPTIMIZATION MODE
    // --------------------------------------------------------------------------
    // -Drelease triggers ReleaseFast mode (handled by standardOptimizeOption).
    // Default is ReleaseFast when neither flag is provided.
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Warn loudly about Debug builds - agents should always use release
    if (optimize == .Debug) {
        std.debug.print(
            \\
            \\╔══════════════════════════════════════════════════════════════════════╗
            \\║  WARNING: Debug build detected!                                      ║
            \\║  Debug builds are 10-100x slower and should NOT be used.             ║
            \\║                                                                      ║
            \\║  Use one of these instead:                                           ║
            \\║    make                          # recommended                       ║
            \\║    zig build release -Drelease  # full build                         ║
            \\║    zig build -Drelease          # library only                       ║
            \\╚══════════════════════════════════════════════════════════════════════╝
            \\
        , .{});
    }

    const enable_metal = b.option(bool, "metal", "Enable Metal GPU support (macOS only)") orelse true;
    const debug_matmul = b.option(bool, "debug-matmul", "Enable matmul debug instrumentation (slow)") orelse false;
    const dump_tensors = b.option(bool, "dump-tensors", "Enable full tensor dump (for debugging, produces talu-dump binary)") orelse false;
    const version = getVersion(b);

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_metal", enable_metal);
    build_options.addOption(bool, "debug_matmul", debug_matmul);
    build_options.addOption(bool, "dump_tensors", dump_tensors);
    build_options.addOption([]const u8, "version", version);

    // Build dependencies
    const pcre2 = addPcre2(b, target, optimize);
    const sqlite3 = addSqlite3(b, target, optimize);
    const miniz = addMiniz(b, target, optimize);
    const libmagic = addLibMagic(b, target, optimize);
    const jpeg_turbo = addJpegTurbo(b, target, optimize);
    const spng = addSpng(b, target, optimize, miniz);
    const webp = addWebp(b, target, optimize);

    // ==========================================================================
    // Generate embedded_graphs.zig from bindings/python/talu/_graphs/*.json
    // ==========================================================================
    const embedded_graphs_mod = generateEmbeddedGraphs(b);

    // ==========================================================================
    // Native shared library
    // ==========================================================================
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_mod.addOptions("build_options", build_options);
    lib_mod.addImport("embedded_graphs", embedded_graphs_mod);
    addCDependencies(b, lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp);

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "talu",
        .root_module = lib_mod,
    });
    linkCDependencies(b, lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, false);
    addMetalSupport(b, lib_mod, lib, enable_metal);

    b.installArtifact(lib);

    // ==========================================================================
    // Python install step - copies library to bindings/python/talu/
    // ==========================================================================
    const python_install_step = b.step("python-install", "Build and copy shared library to Python bindings");
    python_install_step.dependOn(b.getInstallStep());

    // Determine platform-specific library name
    const lib_name = switch (target.result.os.tag) {
        .macos => "libtalu.dylib",
        .windows => "talu.dll",
        else => "libtalu.so",
    };

    const copy_to_python = b.addInstallFile(
        lib.getEmittedBin(),
        b.fmt("../bindings/python/talu/{s}", .{lib_name}),
    );
    python_install_step.dependOn(&copy_to_python.step);

    // ==========================================================================
    // Native static library
    // ==========================================================================
    const static_lib_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    static_lib_mod.addOptions("build_options", build_options);
    static_lib_mod.addImport("embedded_graphs", embedded_graphs_mod);
    addCDependencies(b, static_lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp);

    const static_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "talu",
        .root_module = static_lib_mod,
    });
    // Skip external .a archives for static lib - they'll be linked by the final executable
    linkCDependencies(b, static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, true);
    addMetalSupport(b, static_lib_mod, static_lib, enable_metal);

    const static_step = b.step("static", "Build static library");
    static_step.dependOn(&b.addInstallArtifact(static_lib, .{}).step);

    // ==========================================================================
    // Native CLI
    // ==========================================================================
    const cli_mod = b.createModule(.{
        .root_source_file = b.path("bindings/rust/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "talu",
        .root_module = cli_mod,
    });
    const cargo_cmd = b.addSystemCommand(&.{ "cargo", "build", "--release", "--manifest-path", "bindings/rust/Cargo.toml" });
    exe.step.dependOn(&cargo_cmd.step);
    exe.linkLibrary(static_lib);
    // Link all deps including external .a archives (curl, mbedtls)
    linkCDependencies(b, exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, false);
    exe.addObjectFile(b.path("bindings/rust/target/release/libtalu_cli.a"));

    // Link platform-specific runtime libraries for Rust CLI
    const cli_target_os = exe.rootModuleTarget().os.tag;
    if (cli_target_os == .linux) {
        exe.linkSystemLibrary("unwind");
        exe.linkSystemLibrary("gcc_s");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("pthread");
        exe.linkSystemLibrary("m");
    }
    exe.linkLibrary(sqlite3.lib);
    // macOS frameworks already linked via linkCDependencies

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the CLI executable");
    run_step.dependOn(&run_cmd.step);

    // ==========================================================================
    // Release step - recommended full build
    // ==========================================================================
    // Builds library, copies to Python, and builds CLI - all in ReleaseFast.
    // This is the recommended build command for development and agents.
    const release_step = b.step("release", "Build library + CLI and copy to Python (recommended)");
    release_step.dependOn(python_install_step);
    release_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    // Enforce release mode for the release step
    if (optimize == .Debug) {
        const fail_step = FailStep.create(b,
            \\
            \\ERROR: 'release' step requires -Drelease flag.
            \\
            \\Use: zig build release -Drelease
            \\  or: make
            \\
        );
        release_step.dependOn(&fail_step.step);
    }

    // ==========================================================================
    // Dump binary - tensor capture for debugging new models
    // ==========================================================================
    // Special binary that captures full tensors at trace points for offline comparison.
    // This is dev-only tooling, NOT a product feature.
    //
    // Usage: zig build dump -Drelease
    //        ./zig-out/bin/talu-dump --model path/to/model --prompt "Hello" -o /tmp/talu.npz
    const dump_step = b.step("dump", "Build tensor dump binary for debugging (requires -Drelease)");

    // Enforce release mode - dump binary is useless in debug (too slow)
    if (optimize == .Debug) {
        const dump_fail_step = FailStep.create(b,
            \\
            \\ERROR: 'dump' step requires -Drelease flag.
            \\
            \\Dump binary MUST be release build - debug is too slow for model debugging.
            \\
            \\Use: zig build dump -Drelease
            \\
        );
        dump_step.dependOn(&dump_fail_step.step);
    } else {
        // Create separate build_options with dump_tensors=true
        const dump_build_options = b.addOptions();
        dump_build_options.addOption(bool, "enable_metal", enable_metal);
        dump_build_options.addOption(bool, "debug_matmul", debug_matmul);
        dump_build_options.addOption(bool, "dump_tensors", true); // Always true for dump binary
        dump_build_options.addOption([]const u8, "version", version);

        // Static library with dump instrumentation
        const dump_lib_mod = b.createModule(.{
            .root_source_file = b.path("core/src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        dump_lib_mod.addOptions("build_options", dump_build_options);
        dump_lib_mod.addImport("embedded_graphs", embedded_graphs_mod);
        addCDependencies(b, dump_lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp);

        const dump_static_lib = b.addLibrary(.{
            .linkage = .static,
            .name = "talu-dump-core",
            .root_module = dump_lib_mod,
        });
        // Add C sources to static lib with skip_external_archives=true (matches main build pattern)
        linkCDependencies(b, dump_static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, true);
        addMetalSupport(b, dump_lib_mod, dump_static_lib, enable_metal);

        // Dump CLI executable - only links static lib and external archives (no C sources)
        const dump_cli_mod = b.createModule(.{
            .root_source_file = b.path("core/src/xray/dump/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        dump_cli_mod.addImport("lib", dump_lib_mod); // Allow main.zig to import core modules

        const dump_exe = b.addExecutable(.{
            .name = "talu-dump",
            .root_module = dump_cli_mod,
        });
        dump_exe.linkLibrary(dump_static_lib);
        // Only link external archives - C sources are in the static lib
        linkExternalArchives(b, dump_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp);
        addMetalSupport(b, dump_cli_mod, dump_exe, enable_metal);

        dump_step.dependOn(&b.addInstallArtifact(dump_exe, .{}).step);
    }

    // ==========================================================================
    // Tests
    // ==========================================================================
    const test_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addOptions("build_options", build_options);
    test_mod.addImport("embedded_graphs", embedded_graphs_mod);
    addCDependencies(b, test_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });
    linkCDependencies(b, tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, false);
    addMetalSupport(b, test_mod, tests, enable_metal);

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    const integration_test_mod = b.createModule(.{
        .root_source_file = b.path("core/tests/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "main", .module = test_mod },
        },
    });

    const integration_tests = b.addTest(.{
        .root_module = integration_test_mod,
    });

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // ==========================================================================
    // Compliance (only if tests directory exists)
    // ==========================================================================
    const compliance_path = "core/tests/helpers/enforce_test_mapping.zig";
    if (std.fs.cwd().access(compliance_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});
        const compliance_mod = b.createModule(.{
            .root_source_file = b.path(compliance_path),
            .target = host_target,
            .optimize = .ReleaseFast,
        });
        const compliance_exe = b.addExecutable(.{
            .name = "enforce_test_mapping",
            .root_module = compliance_mod,
        });
        b.installArtifact(compliance_exe);
        const run_compliance = b.addRunArtifact(compliance_exe);
        run_compliance.addArg("core/src");
        run_compliance.addArg("--test-dir");
        run_compliance.addArg("tests");
        const compliance_step = b.step("check-compliance", "Check that all pub fn have corresponding tests");
        compliance_step.dependOn(&run_compliance.step);
    } else |_| {
        // core/tests/ directory not present (e.g., sdist build) - skip compliance tool
    }

    // ==========================================================================
    // Policy Linter (core/tests/helpers/lint/)
    // ==========================================================================
    const lint_path = "core/tests/helpers/lint/root.zig";
    if (std.fs.cwd().access(lint_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});

        // Lint executable
        const lint_mod = b.createModule(.{
            .root_source_file = b.path(lint_path),
            .target = host_target,
            .optimize = .ReleaseFast,
        });
        const lint_exe = b.addExecutable(.{
            .name = "enforce_lint",
            .root_module = lint_mod,
        });
        b.installArtifact(lint_exe);

        // Run lint
        const run_lint = b.addRunArtifact(lint_exe);
        run_lint.addArg("core/src");
        if (b.args) |args| {
            for (args) |arg| {
                run_lint.addArg(arg);
            }
        }
        const lint_step = b.step("lint", "Run policy compliance linter");
        lint_step.dependOn(&run_lint.step);

        // Lint tests
        const lint_test_mod = b.createModule(.{
            .root_source_file = b.path(lint_path),
            .target = host_target,
            .optimize = optimize,
        });
        const lint_tests = b.addTest(.{
            .root_module = lint_test_mod,
        });
        const run_lint_tests = b.addRunArtifact(lint_tests);
        const lint_test_step = b.step("test-lint", "Run lint module tests");
        lint_test_step.dependOn(&run_lint_tests.step);
    } else |_| {
        // lint module not present - skip
    }

    // ==========================================================================
    // Python Binding Generator (core/tests/helpers/gen_bindings.zig)
    // ==========================================================================
    // Python bindings generator
    const gen_bindings_path = "core/tests/helpers/gen_bindings.zig";
    if (std.fs.cwd().access(gen_bindings_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});

        const gen_mod = b.createModule(.{
            .root_source_file = b.path(gen_bindings_path),
            .target = host_target,
            .optimize = .ReleaseFast,
        });
        const gen_exe = b.addExecutable(.{
            .name = "gen_bindings",
            .root_module = gen_mod,
        });
        b.installArtifact(gen_exe);

        // Run generator - output to zig-out/lib/ (build_hook.py copies to package)
        const run_gen = b.addRunArtifact(gen_exe);
        run_gen.addArg("."); // project root
        run_gen.addArg("zig-out/lib/_native.py"); // output path (same pattern as .so)

        // Format the generated file with ruff (auto-generated code, use defaults + line-length)
        const format_cmd = b.addSystemCommand(&.{
            "uvx", "ruff", "format", "--line-length", "100", "zig-out/lib/_native.py",
        });
        format_cmd.step.dependOn(&run_gen.step);

        // Copy generated _native.py to Python package (best-effort: dir may not exist in sdist builds)
        const copy_native = b.addSystemCommand(&.{
            "sh", "-c", "[ -d bindings/python/talu ] && cp zig-out/lib/_native.py bindings/python/talu/_native.py || true",
        });
        copy_native.step.dependOn(&format_cmd.step);

        const gen_step = b.step("gen-bindings", "Generate Python ctypes bindings from C API");
        gen_step.dependOn(&copy_native.step);

        // Release step generates _native.py but skips the cp to bindings/python/
        // (build_hook.py handles that copy — the bindings dir doesn't exist in sdist builds)
        release_step.dependOn(&format_cmd.step);
    } else |_| {
        // gen_bindings module not present - skip
    }

    // Rust bindings generator
    const gen_bindings_rust_path = "core/tests/helpers/gen_bindings_rust.zig";
    if (std.fs.cwd().access(gen_bindings_rust_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});

        const gen_rust_mod = b.createModule(.{
            .root_source_file = b.path(gen_bindings_rust_path),
            .target = host_target,
            .optimize = .ReleaseFast,
        });
        const gen_rust_exe = b.addExecutable(.{
            .name = "gen_bindings_rust",
            .root_module = gen_rust_mod,
        });
        b.installArtifact(gen_rust_exe);

        // Run generator
        const run_gen_rust = b.addRunArtifact(gen_rust_exe);
        run_gen_rust.addArg("."); // project root
        run_gen_rust.addArg("bindings/rust/talu-sys/src/lib.rs"); // output path
        const gen_rust_step = b.step("gen-bindings-rust", "Generate Rust FFI bindings from C API");
        gen_rust_step.dependOn(&run_gen_rust.step);
    } else |_| {
        // gen_bindings_rust module not present - skip
    }
}
