const std = @import("std");

pub const JpegTurbo = struct {
    lib: *std.Build.Step.Compile,
    jpeg12_lib: *std.Build.Step.Compile,
    jpeg16_lib: *std.Build.Step.Compile,
    turbojpeg12_lib: *std.Build.Step.Compile,
    turbojpeg16_lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
    generated_include_dir: std.Build.LazyPath,
};

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) JpegTurbo {
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
