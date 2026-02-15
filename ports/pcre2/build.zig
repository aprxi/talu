const std = @import("std");

pub const Pcre2 = struct {
    lib: *std.Build.Step.Compile,
    include_dir: std.Build.LazyPath,
};

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Pcre2 {
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
