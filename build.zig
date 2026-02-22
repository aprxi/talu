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
// Dependencies — each built from ports/<dep>/build.zig
// =============================================================================

const pcre2_port = @import("ports/pcre2/build.zig");
const sqlite_port = @import("ports/sqlite/build.zig");
const miniz_port = @import("ports/miniz/build.zig");
const jpeg_turbo_port = @import("ports/jpeg-turbo/build.zig");
const spng_port = @import("ports/spng/build.zig");
const webp_port = @import("ports/webp/build.zig");
const libmagic_port = @import("ports/libmagic/build.zig");
const tree_sitter_port = @import("ports/tree-sitter/build.zig");

const Pcre2 = pcre2_port.Pcre2;
const Sqlite3 = sqlite_port.Sqlite3;
const Miniz = miniz_port.Miniz;
const JpegTurbo = jpeg_turbo_port.JpegTurbo;
const Spng = spng_port.Spng;
const Webp = webp_port.Webp;
const LibMagic = libmagic_port.LibMagic;
const TreeSitter = tree_sitter_port.TreeSitter;

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
    tree_sitter: TreeSitter,
) void {
    mod.addIncludePath(b.path("src/text"));
    mod.addIncludePath(b.path("include"));
    mod.addIncludePath(b.path("deps/utf8proc"));
    mod.addIncludePath(pcre2.include_dir);
    mod.addIncludePath(b.path("deps/pcre2/src"));
    mod.addIncludePath(b.path("deps/curl/include"));
    mod.addIncludePath(b.path("deps/pdfium/public"));
    mod.addIncludePath(miniz.include_dir);
    mod.addIncludePath(libmagic.include_dir);
    mod.addIncludePath(jpeg_turbo.generated_include_dir);
    mod.addIncludePath(jpeg_turbo.include_dir);
    mod.addIncludePath(spng.include_dir);
    mod.addIncludePath(webp.include_dir);
    mod.addIncludePath(tree_sitter.include_dir);

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
    tree_sitter: TreeSitter,
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
    artifact.linkLibrary(tree_sitter.lib);

    // For static libraries, skip external .a archives to avoid nested archive warnings.
    // The final executable/shared lib will link them directly.
    if (!skip_external_archives) {
        // Link pre-built libcurl static library
        artifact.addObjectFile(b.path("deps/curl/build/lib/libcurl.a"));

        // Link mbedTLS for HTTPS support (used on all platforms)
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedtls.a"));
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedx509.a"));
        artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedcrypto.a"));

        // Link PDFium for PDF rendering
        artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libpdfium.a"));
        artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libagg.a"));
        artifact.addObjectFile(b.path("deps/pdfium/cmake-build/liblcms.a"));
        artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libopenjpeg.a"));
        artifact.addObjectFile(b.path("deps/freetype/build/libfreetype.a"));
        artifact.linkLibCpp();

        const target_os = artifact.rootModuleTarget().os.tag;
        if (target_os == .macos) {
            artifact.linkFramework("CoreFoundation");
            artifact.linkFramework("SystemConfiguration");
            artifact.linkFramework("Security");
        }
        artifact.linkSystemLibrary("pthread");

        // PDFium ICU shim: u_tolower, u_isalpha, etc. backed by utf8proc.
        artifact.addCSourceFiles(.{
            .files = &.{"ports/pdfium/icu_shim.c"},
            .flags = &.{ "-std=gnu11", "-Ideps/utf8proc" },
        });
        // PDFium zlib shim: inflate, compress, etc. forwarded to miniz.
        // Must define MINIZ_NO_ZLIB_COMPATIBLE_NAMES to avoid conflicting
        // inline definitions from miniz.h.
        artifact.addCSourceFiles(.{
            .files = &.{"ports/pdfium/zlib_shim.c"},
            .flags = &.{ "-std=gnu11", "-DMINIZ_NO_ZLIB_COMPATIBLE_NAMES" },
        });
        artifact.addIncludePath(miniz.include_dir);

        artifact.addCSourceFiles(.{
            .files = &.{"deps/utf8proc/utf8proc.c"},
            .flags = &.{"-std=gnu11"},
        });
    }

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
    tree_sitter: TreeSitter,
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
    artifact.linkLibrary(tree_sitter.lib);

    // Link pre-built libcurl static library
    artifact.addObjectFile(b.path("deps/curl/build/lib/libcurl.a"));

    // Link mbedTLS for HTTPS support (used on all platforms)
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedtls.a"));
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedx509.a"));
    artifact.addObjectFile(b.path("deps/mbedtls/build/library/libmbedcrypto.a"));

    // Link PDFium for PDF rendering
    artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libpdfium.a"));
    artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libagg.a"));
    artifact.addObjectFile(b.path("deps/pdfium/cmake-build/liblcms.a"));
    artifact.addObjectFile(b.path("deps/pdfium/cmake-build/libopenjpeg.a"));
    artifact.addObjectFile(b.path("deps/freetype/build/libfreetype.a"));
    artifact.linkLibCpp();

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
    mod.addIncludePath(b.path("core/src/compute/metal/mlx"));
    mod.addIncludePath(b.path("core/src/inference/backend/metal/mlx"));
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
            "core/src/inference/backend/metal/mlx/cache.cpp",
            "core/src/inference/backend/metal/mlx/fused_ops.cpp",
            "core/src/inference/backend/metal/mlx/decode_model.cpp",
            "core/src/inference/backend/metal/mlx/model_dense.cpp",
            "core/src/inference/backend/metal/mlx/model_quantized.cpp",
        },
        .flags = &.{
            "-std=c++17",
        },
    });

    artifact.linkLibCpp();
}

/// Link only the minimal Metal primitives used by compute-level sanity tools.
/// This intentionally excludes MLX C++ sources to avoid eager MLX runtime init.
fn addMetalCoreSupport(
    b: *std.Build,
    mod: *std.Build.Module,
    artifact: *std.Build.Step.Compile,
    enable_metal: bool,
) void {
    if (artifact.rootModuleTarget().os.tag != .macos) return;
    if (!enable_metal) return;

    mod.addIncludePath(b.path("core/src/compute/metal"));

    artifact.linkFramework("Metal");
    artifact.linkFramework("MetalPerformanceShaders");
    artifact.linkFramework("Foundation");

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
    const enable_cuda = b.option(bool, "cuda", "Enable CUDA backend scaffold (Linux/Windows)") orelse true;
    const debug_matmul = b.option(bool, "debug-matmul", "Enable matmul debug instrumentation (slow)") orelse false;
    const dump_tensors = b.option(bool, "dump-tensors", "Enable full tensor dump (for debugging, produces talu-dump binary)") orelse false;
    const version = getVersion(b);

    const gen_ptx_step = b.step("gen-ptx", "Generate CUDA PTX assets (requires nvcc)");
    {
        const gen_fallback_ptx = b.addSystemCommand(&.{
            "nvcc",
            "-ptx",
            "-arch=sm_80",
            "core/src/compute/cuda/kernels/kernels.cu",
            "-o",
            "core/src/compute/cuda/kernels/kernels.ptx",
        });
        gen_ptx_step.dependOn(&gen_fallback_ptx.step);
    }

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_metal", enable_metal);
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption(bool, "debug_matmul", debug_matmul);
    build_options.addOption(bool, "dump_tensors", dump_tensors);
    build_options.addOption([]const u8, "version", version);

    // Build dependencies
    const pcre2 = pcre2_port.add(b, target, optimize);
    const sqlite3 = sqlite_port.add(b, target, optimize);
    const miniz = miniz_port.add(b, target, optimize);
    const libmagic = libmagic_port.add(b, target, optimize);
    const jpeg_turbo = jpeg_turbo_port.add(b, target, optimize);
    const spng = spng_port.add(b, target, optimize, miniz);
    const webp = webp_port.add(b, target, optimize);
    const tree_sitter = tree_sitter_port.add(b, target, optimize);

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
    addCDependencies(b, lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "talu",
        .root_module = lib_mod,
    });
    linkCDependencies(b, lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);
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
    addCDependencies(b, static_lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);

    const static_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "talu",
        .root_module = static_lib_mod,
    });
    // Skip external .a archives for static lib - they'll be linked by the final executable
    linkCDependencies(b, static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, true);
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
    linkCDependencies(b, exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);
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
        dump_build_options.addOption(bool, "enable_cuda", enable_cuda);
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
        addCDependencies(b, dump_lib_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);

        const dump_static_lib = b.addLibrary(.{
            .linkage = .static,
            .name = "talu-dump-core",
            .root_module = dump_lib_mod,
        });
        // Add C sources to static lib with skip_external_archives=true (matches main build pattern)
        linkCDependencies(b, dump_static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, true);
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
        linkExternalArchives(b, dump_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);
        addMetalSupport(b, dump_cli_mod, dump_exe, enable_metal);

        dump_step.dependOn(&b.addInstallArtifact(dump_exe, .{}).step);
    }

    // ==========================================================================
    // Tests
    // ==========================================================================
    // Keep default unit tests CPU-only to avoid MLX/Metal runtime coupling in
    // aggregate runners. Metal coverage is exercised via targeted Metal tests.
    const unit_test_build_options = b.addOptions();
    unit_test_build_options.addOption(bool, "enable_metal", false);
    unit_test_build_options.addOption(bool, "enable_cuda", enable_cuda);
    unit_test_build_options.addOption(bool, "debug_matmul", debug_matmul);
    unit_test_build_options.addOption(bool, "dump_tensors", dump_tensors);
    unit_test_build_options.addOption([]const u8, "version", version);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addOptions("build_options", unit_test_build_options);
    addCDependencies(b, test_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });
    linkCDependencies(b, tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // Build integration tests against a separate copy of core/src/lib.zig.
    // Keep integration on CPU-only to avoid MLX/Metal runtime coupling and
    // ensure deterministic behavior across host GPU setups.
    const integration_build_options = b.addOptions();
    integration_build_options.addOption(bool, "enable_metal", false);
    integration_build_options.addOption(bool, "enable_cuda", enable_cuda);
    integration_build_options.addOption(bool, "debug_matmul", debug_matmul);
    integration_build_options.addOption(bool, "dump_tensors", dump_tensors);
    integration_build_options.addOption([]const u8, "version", version);

    const integration_main_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    integration_main_mod.addOptions("build_options", integration_build_options);
    addCDependencies(b, integration_main_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter);

    const integration_test_mod = b.createModule(.{
        .root_source_file = b.path("core/tests/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "main", .module = integration_main_mod },
        },
    });

    const integration_tests = b.addTest(.{
        .root_module = integration_test_mod,
    });
    linkCDependencies(b, integration_tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Models metadata report command:
    //   zig build models-report -- registry
    //   zig build models-report -- metadata
    const models_report_path = "core/tests/models/report.zig";
    if (std.fs.cwd().access(models_report_path, .{})) |_| {
        const models_report_mod = b.createModule(.{
            .root_source_file = b.path(models_report_path),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        const models_report_exe = b.addExecutable(.{
            .name = "models_report",
            .root_module = models_report_mod,
        });
        linkCDependencies(b, models_report_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);
        const run_models_report = b.addRunArtifact(models_report_exe);
        if (b.args) |args| {
            for (args) |arg| run_models_report.addArg(arg);
        } else {
            run_models_report.addArg("registry");
        }
        const models_report_step = b.step("models-report", "Print deterministic models metadata reports");
        models_report_step.dependOn(&run_models_report.step);
    } else |_| {
        // report helper not present - skip
    }

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
    // Model change policy checker (core/tests/helpers/model_policy/)
    // ==========================================================================
    const model_policy_path = "core/tests/helpers/model_policy/root.zig";
    if (std.fs.cwd().access(model_policy_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});

        const policy_mod = b.createModule(.{
            .root_source_file = b.path(model_policy_path),
            .target = host_target,
            .optimize = .ReleaseFast,
        });
        const policy_exe = b.addExecutable(.{
            .name = "model_policy",
            .root_module = policy_mod,
        });
        b.installArtifact(policy_exe);

        const run_policy = b.addRunArtifact(policy_exe);
        if (b.args) |args| {
            for (args) |arg| {
                run_policy.addArg(arg);
            }
        }
        const policy_step = b.step("model-policy", "Run model-change policy checks");
        policy_step.dependOn(&run_policy.step);

        const policy_tests = b.addTest(.{
            .root_module = policy_mod,
        });
        const run_policy_tests = b.addRunArtifact(policy_tests);
        const policy_test_step = b.step("test-model-policy", "Run model-change policy tests");
        policy_test_step.dependOn(&run_policy_tests.step);
    } else |_| {
        // model policy checker not present - skip
    }

    // ==========================================================================
    // Perf/lifecycle sanity checker (core/tests/helpers/perf_sanity/)
    // ==========================================================================
    const perf_sanity_path = "core/tests/helpers/perf_sanity/root.zig";
    if (std.fs.cwd().access(perf_sanity_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});
        const perf_exe_build_options = b.addOptions();
        perf_exe_build_options.addOption(bool, "enable_metal", enable_metal);
        perf_exe_build_options.addOption(bool, "enable_cuda", enable_cuda);

        const perf_exe_mod = b.createModule(.{
            .root_source_file = b.path(perf_sanity_path),
            .target = host_target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        perf_exe_mod.addOptions("build_options", perf_exe_build_options);
        const perf_exe = b.addExecutable(.{
            .name = "perf_sanity",
            .root_module = perf_exe_mod,
        });
        linkCDependencies(b, perf_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);
        addMetalCoreSupport(b, perf_exe_mod, perf_exe, enable_metal);
        b.installArtifact(perf_exe);

        const run_perf = b.addRunArtifact(perf_exe);
        if (b.args) |args| {
            for (args) |arg| {
                run_perf.addArg(arg);
            }
        }
        const perf_step = b.step("perf-sanity", "Run perf/lifecycle sanity checks");
        perf_step.dependOn(&run_perf.step);

        const perf_test_build_options = b.addOptions();
        perf_test_build_options.addOption(bool, "enable_metal", false);
        perf_test_build_options.addOption(bool, "enable_cuda", enable_cuda);

        const perf_test_mod = b.createModule(.{
            .root_source_file = b.path(perf_sanity_path),
            .target = host_target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        perf_test_mod.addOptions("build_options", perf_test_build_options);

        const perf_tests = b.addTest(.{
            .root_module = perf_test_mod,
        });
        linkCDependencies(b, perf_tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, tree_sitter, false);
        const run_perf_tests = b.addRunArtifact(perf_tests);
        const perf_test_step = b.step("test-perf-sanity", "Run perf/lifecycle sanity tests");
        perf_test_step.dependOn(&run_perf_tests.step);
        // Keep perf/determinism guardrails part of integration validation.
        integration_test_step.dependOn(&run_perf_tests.step);
    } else |_| {
        // perf sanity checker not present - skip
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
