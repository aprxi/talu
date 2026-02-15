const std = @import("std");

pub const Webp = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(
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
