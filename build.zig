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

fn pathExists(path: []const u8) bool {
    if (std.fs.path.isAbsolute(path)) {
        std.fs.accessAbsolute(path, .{}) catch return false;
        return true;
    }
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn findMlxMetallib(b: *std.Build) ?[]const u8 {
    const env_path = std.process.getEnvVarOwned(b.allocator, "MLX_METALLIB") catch null;
    if (env_path) |path| {
        if (pathExists(path)) {
            if (std.fs.path.isAbsolute(path)) return path;
            const abs_path = std.fs.cwd().realpathAlloc(b.allocator, path) catch {
                b.allocator.free(path);
                return null;
            };
            b.allocator.free(path);
            return abs_path;
        }
        b.allocator.free(path);
    }

    const candidates = [_][]const u8{
        "deps/mlx/lib/mlx.metallib",
        "deps/mlx-src/build/mlx/backend/metal/kernels/mlx.metallib",
        "/opt/homebrew/bin/mlx.metallib",
        "/usr/local/bin/mlx.metallib",
    };
    for (candidates) |candidate| {
        if (pathExists(candidate)) {
            if (std.fs.path.isAbsolute(candidate)) {
                return b.allocator.dupe(u8, candidate) catch @panic("OOM");
            }
            return std.fs.cwd().realpathAlloc(b.allocator, candidate) catch @panic("OOM");
        }
    }
    return null;
}

fn hashBytesFnv1a(seed: u64, bytes: []const u8) u64 {
    var h = seed;
    for (bytes) |b| {
        h ^= @as(u64, b);
        h *%= 1099511628211;
    }
    return h;
}

fn hasSuffixAny(name: []const u8, suffixes: []const []const u8) bool {
    for (suffixes) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return true;
    }
    return false;
}

fn mlxIncludeFingerprintFlag(b: *std.Build) []const u8 {
    const dir_path = "core/src/compute/metal/mlx";
    const tracked_suffixes = [_][]const u8{ ".inc", ".h" };

    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch {
        return b.fmt("-DTALU_MLX_INCLUDE_FP=0x0", .{});
    };
    defer dir.close();

    var names = std.ArrayList([]const u8){};
    defer names.deinit(b.allocator);

    var it = dir.iterate();
    while (it.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!hasSuffixAny(entry.name, tracked_suffixes[0..])) continue;
        names.append(b.allocator, b.dupe(entry.name)) catch @panic("OOM");
    }

    std.sort.block([]const u8, names.items, {}, struct {
        fn lessThan(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.order(u8, a, c) == .lt;
        }
    }.lessThan);

    var h: u64 = 14695981039346656037;
    for (names.items) |name| {
        h = hashBytesFnv1a(h, name);
        const full_path = b.fmt("{s}/{s}", .{ dir_path, name });
        const contents = std.fs.cwd().readFileAlloc(b.allocator, full_path, 16 * 1024 * 1024) catch continue;
        defer b.allocator.free(contents);
        h = hashBytesFnv1a(h, contents);
    }

    return b.fmt("-DTALU_MLX_INCLUDE_FP=0x{x}", .{h});
}

fn isAllowedInferenceBoundaryImport(rel_path: []const u8, import_path: []const u8) bool {
    if (std.mem.eql(u8, rel_path, "router/inference_bridge.zig")) {
        return std.mem.eql(u8, import_path, "inference_pkg");
    }
    if (std.mem.eql(u8, rel_path, "converter/calibration_capture.zig")) {
        return std.mem.eql(u8, import_path, "inference_pkg");
    }
    if (std.mem.eql(u8, rel_path, "lib_dev.zig")) {
        return std.mem.eql(u8, import_path, "inference_pkg");
    }
    return false;
}

fn validateInferenceBoundaryImports(b: *std.Build) void {
    var root_dir = std.fs.cwd().openDir("core/src", .{ .iterate = true }) catch |err| {
        std.debug.panic("failed to open core/src for boundary validation: {s}", .{@errorName(err)});
    };
    defer root_dir.close();

    var walker = root_dir.walk(b.allocator) catch |err| {
        std.debug.panic("failed to walk core/src for boundary validation: {s}", .{@errorName(err)});
    };
    defer walker.deinit();

    var has_violations = false;
    while (walker.next() catch |err| {
        std.debug.panic("failed during boundary validation walk: {s}", .{@errorName(err)});
    }) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".zig")) continue;
        if (std.mem.startsWith(u8, entry.path, "inference/")) continue;

        const abs_path = b.fmt("core/src/{s}", .{entry.path});
        const source = std.fs.cwd().readFileAlloc(b.allocator, abs_path, 16 * 1024 * 1024) catch |err| {
            std.debug.panic("failed reading {s} for boundary validation: {s}", .{ abs_path, @errorName(err) });
        };
        defer b.allocator.free(source);

        const import_prefix = "@import(\"";
        var search_start: usize = 0;
        while (std.mem.indexOfPos(u8, source, search_start, import_prefix)) |match_start| {
            const path_start = match_start + import_prefix.len;
            const path_end = std.mem.indexOfPos(u8, source, path_start, "\"") orelse break;
            search_start = path_end + 1;

            const import_path = source[path_start..path_end];
            const targets_inference = std.mem.indexOf(u8, import_path, "inference/") != null or
                std.mem.eql(u8, import_path, "inference_pkg");
            if (!targets_inference) continue;
            if (isAllowedInferenceBoundaryImport(entry.path, import_path)) continue;

            has_violations = true;
            std.debug.print(
                "inference boundary violation: core/src/{s} imports {s}; route via inference_pkg bridge\n",
                .{ entry.path, import_path },
            );
        }
    }

    if (has_violations) {
        std.debug.panic("inference import boundary validation failed", .{});
    }
}

// =============================================================================
// Dependencies — each built from ports/<dep>/build.zig
// =============================================================================

const pcre2_port = @import("ports/pcre2/build.zig");
const miniz_port = @import("ports/miniz/build.zig");
const jpeg_turbo_port = @import("ports/jpeg-turbo/build.zig");
const spng_port = @import("ports/spng/build.zig");
const webp_port = @import("ports/webp/build.zig");
const libmagic_port = @import("ports/libmagic/build.zig");
const tree_sitter_port = @import("ports/tree-sitter/build.zig");
const sqlite_port = @import("ports/sqlite/build.zig");

const Pcre2 = pcre2_port.Pcre2;
const Miniz = miniz_port.Miniz;
const JpegTurbo = jpeg_turbo_port.JpegTurbo;
const Spng = spng_port.Spng;
const Webp = webp_port.Webp;
const LibMagic = libmagic_port.LibMagic;
const TreeSitter = tree_sitter_port.TreeSitter;
const Sqlite3 = sqlite_port.Sqlite3;

const CorePackages = struct {
    models_pkg: *std.Build.Module,
    tensor_pkg: *std.Build.Module,
    compute_pkg: *std.Build.Module,
    log_pkg: *std.Build.Module,
    io_pkg: *std.Build.Module,
    dtype_pkg: *std.Build.Module,
    xray_pkg: *std.Build.Module,
    runtime_contract_pkg: *std.Build.Module,
    progress_pkg: *std.Build.Module,
    validate_pkg: *std.Build.Module,
    image_pkg: *std.Build.Module,
    error_context_pkg: *std.Build.Module,

    fn init(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) CorePackages {
        return .{
            .models_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/models/root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .tensor_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/tensor.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .compute_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/compute_pkg.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .log_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/log.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .io_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/io_pkg.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .dtype_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/dtype.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .xray_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/xray/root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .runtime_contract_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/runtime_contract/root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .progress_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/progress.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .validate_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/validate/root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .image_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/image/root.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
            .error_context_pkg = b.createModule(.{
                .root_source_file = b.path("core/src/error_context.zig"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            }),
        };
    }
};

fn addCorePackageImports(mod: *std.Build.Module, pkgs: *const CorePackages) void {
    mod.addImport("models_pkg", pkgs.models_pkg);
    if (mod != pkgs.tensor_pkg) mod.addImport("tensor_pkg", pkgs.tensor_pkg);
    mod.addImport("compute_pkg", pkgs.compute_pkg);
    if (mod != pkgs.log_pkg) mod.addImport("log_pkg", pkgs.log_pkg);
    if (mod != pkgs.io_pkg) mod.addImport("io_pkg", pkgs.io_pkg);
    if (mod != pkgs.dtype_pkg) mod.addImport("dtype_pkg", pkgs.dtype_pkg);
    mod.addImport("xray_pkg", pkgs.xray_pkg);
    if (mod != pkgs.runtime_contract_pkg) mod.addImport("runtime_contract_pkg", pkgs.runtime_contract_pkg);
    if (mod != pkgs.progress_pkg) mod.addImport("progress_pkg", pkgs.progress_pkg);
    if (mod != pkgs.validate_pkg) mod.addImport("validate_pkg", pkgs.validate_pkg);
    if (mod != pkgs.image_pkg) mod.addImport("image_pkg", pkgs.image_pkg);
    if (mod != pkgs.error_context_pkg) mod.addImport("error_context_pkg", pkgs.error_context_pkg);
}

fn wireCorePackageImports(pkgs: *const CorePackages) void {
    addCorePackageImports(pkgs.models_pkg, pkgs);
    addCorePackageImports(pkgs.tensor_pkg, pkgs);
    addCorePackageImports(pkgs.compute_pkg, pkgs);
    addCorePackageImports(pkgs.log_pkg, pkgs);
    addCorePackageImports(pkgs.io_pkg, pkgs);
    addCorePackageImports(pkgs.dtype_pkg, pkgs);
    addCorePackageImports(pkgs.xray_pkg, pkgs);
    addCorePackageImports(pkgs.runtime_contract_pkg, pkgs);
    addCorePackageImports(pkgs.progress_pkg, pkgs);
    addCorePackageImports(pkgs.validate_pkg, pkgs);
    addCorePackageImports(pkgs.image_pkg, pkgs);
    addCorePackageImports(pkgs.error_context_pkg, pkgs);
}

fn addCorePackageBuildImportsAndDeps(
    b: *std.Build,
    pkgs: *const CorePackages,
    build_options_mod: *std.Build.Module,
    cacert_mod: *std.Build.Module,
    magic_db_mod: *std.Build.Module,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
    sqlite3: Sqlite3,
    tree_sitter: TreeSitter,
) void {
    const modules = [_]*std.Build.Module{
        pkgs.models_pkg,
        pkgs.tensor_pkg,
        pkgs.compute_pkg,
        pkgs.log_pkg,
        pkgs.io_pkg,
        pkgs.dtype_pkg,
        pkgs.xray_pkg,
        pkgs.runtime_contract_pkg,
        pkgs.progress_pkg,
        pkgs.validate_pkg,
        pkgs.image_pkg,
        pkgs.error_context_pkg,
    };
    inline for (modules) |mod| {
        mod.addImport("build_options", build_options_mod);
        addCDependencies(b, mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);
    }
}

// =============================================================================
// Helper to add C dependencies to a module
// =============================================================================

fn addCDependencies(
    b: *std.Build,
    mod: *std.Build.Module,
    cacert_mod: *std.Build.Module,
    magic_db_mod: *std.Build.Module,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
    sqlite3: Sqlite3,
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
    mod.addIncludePath(sqlite3.include_dir);
    mod.addIncludePath(tree_sitter.include_dir);

    // Mozilla CA certificates bundle for HTTPS - generated during `make deps`
    mod.addImport("cacert", cacert_mod);

    // Compiled magic database for file type detection - generated during `make deps`
    mod.addImport("magic_db", magic_db_mod);
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
    sqlite3: Sqlite3,
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
    artifact.linkLibrary(sqlite3.lib);
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
    sqlite3: Sqlite3,
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
    artifact.linkLibrary(sqlite3.lib);
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
// Per-module unit test helper
// =============================================================================

const UnitTestCfg = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    build_options: *std.Build.Step.Options,
    cuda_assets_mod: *std.Build.Module,
    inference_pkg_mod: *std.Build.Module,
    cacert_mod: *std.Build.Module,
    magic_db_mod: *std.Build.Module,
    core_pkgs: *const CorePackages,
    pcre2: Pcre2,
    miniz: Miniz,
    libmagic: LibMagic,
    jpeg_turbo: JpegTurbo,
    spng: Spng,
    webp: Webp,
    sqlite3: Sqlite3,
    tree_sitter: TreeSitter,

    fn addLazy(
        self: UnitTestCfg,
        comptime name: []const u8,
        root_source_file: std.Build.LazyPath,
        filters: []const []const u8,
    ) void {
        const mod = self.b.createModule(.{
            .root_source_file = root_source_file,
            .target = self.target,
            .optimize = self.optimize,
            .link_libc = true,
        });
        mod.addImport("cuda_assets", self.cuda_assets_mod);
        mod.addImport("inference_pkg", self.inference_pkg_mod);
        addCorePackageImports(mod, self.core_pkgs);
        mod.addOptions("build_options", self.build_options);
        addCDependencies(self.b, mod, self.cacert_mod, self.magic_db_mod, self.pcre2, self.miniz, self.libmagic, self.jpeg_turbo, self.spng, self.webp, self.sqlite3, self.tree_sitter);

        const test_artifact = self.b.addTest(.{
            .root_module = mod,
            .filters = filters,
        });
        linkCDependencies(self.b, test_artifact, self.pcre2, self.miniz, self.libmagic, self.jpeg_turbo, self.spng, self.webp, self.sqlite3, self.tree_sitter, false);
        const run = self.b.addRunArtifact(test_artifact);
        const step = self.b.step("test-" ++ name, "Run " ++ name ++ " unit tests");
        step.dependOn(&run.step);
    }

    fn add(self: UnitTestCfg, comptime name: []const u8, comptime root_path: []const u8) void {
        self.addLazy(name, self.b.path(root_path), &.{});
    }
};

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
    mod.addIncludePath(b.path("core/src/inference/backend/metal/mlx_bridge"));
    mod.addIncludePath(b.path("deps/mlx/include"));
    artifact.addIncludePath(b.path("core/src/compute/metal"));
    artifact.addIncludePath(b.path("core/src/compute/metal/mlx"));
    artifact.addIncludePath(b.path("core/src/inference/backend/metal/mlx_bridge"));
    artifact.addIncludePath(b.path("deps/mlx/include"));

    artifact.linkFramework("Metal");
    artifact.linkFramework("MetalPerformanceShaders");
    artifact.linkFramework("Foundation");
    artifact.linkFramework("Accelerate");

    artifact.addObjectFile(b.path("deps/mlx/lib/libmlx.a"));
    const mlx_include_fp_flag = mlxIncludeFingerprintFlag(b);

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
            "core/src/compute/metal/mlx/cache.cpp",
            "core/src/compute/metal/mlx/fused_ops.cpp",
            "core/src/inference/backend/metal/mlx_bridge/config_parse.cpp",
            "core/src/inference/backend/metal/mlx_bridge/bridge.cpp",
        },
        .flags = &.{
            "-std=c++17",
            mlx_include_fp_flag,
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
    // Default to x86_64_v3 (AVX2) on x86_64 Linux for consistent codegen.
    // Without this, `zig build` without -Dcpu uses native CPU detection, which
    // can silently enable AVX-512 and produce code that (a) won't run on AVX2-
    // only machines and (b) gives misleading benchmark results. Users can still
    // override via -Dcpu=native or any other value.
    var target = b.standardTargetOptions(.{});

    // For Linux x86_64: default to x86_64_v3 when no -Dcpu was provided.
    // Without this, `zig build -Drelease` uses native CPU detection, which
    // can silently enable AVX-512 and produce binaries that won't run on
    // AVX2-only machines (and give misleading benchmark results).
    if (target.result.os.tag == .linux and target.result.cpu.arch == .x86_64 and
        target.query.cpu_model == .determined_by_arch_os)
    {
        var query = target.query;
        query.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
        target = b.resolveTargetQuery(query);
    }

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
    const cuda_startup_selftests = b.option(bool, "cuda-startup-selftests", "Run CUDA startup smoke/parity checks in backend init (slow)") orelse false;
    const dump_tensors = b.option(bool, "dump-tensors", "Enable full tensor dump (for debugging, produces talu-dump binary)") orelse false;
    const xray_bridge_default = switch (optimize) {
        .Debug => true,
        else => false,
    };
    const xray_bridge = b.option(bool, "xray_bridge", "Enable xray bridge instrumentation hooks (default: on in Debug, off in Release)") orelse xray_bridge_default;
    const version = getVersion(b);

    // Enforce inference modular boundary for non-inference modules.
    validateInferenceBoundaryImports(b);

    const gen_cuda_kernels_step = b.step("gen-cuda-kernels", "Generate CUDA kernel module assets (requires nvcc)");
    {
        const ensure_cuda_assets_dir = b.addSystemCommand(&.{
            "mkdir",
            "-p",
            "core/assets/cuda",
        });
        const gen_kernel_module = b.addSystemCommand(&.{
            "nvcc",
            "--fatbin",
            "-arch=all-major",
            "core/src/compute/cuda/kernels/kernels.cu",
            "-o",
            "core/assets/cuda/kernels.fatbin",
        });
        gen_kernel_module.step.dependOn(&ensure_cuda_assets_dir.step);
        gen_cuda_kernels_step.dependOn(&gen_kernel_module.step);
    }

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_metal", enable_metal);
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
    build_options.addOption(bool, "debug_matmul", debug_matmul);
    build_options.addOption(bool, "dump_tensors", dump_tensors);
    build_options.addOption(bool, "xray_bridge", xray_bridge);
    build_options.addOption([]const u8, "version", version);
    const build_options_mod = build_options.createModule();

    const cacert_mod = b.createModule(.{
        .root_source_file = b.path("deps/cacert.zig"),
    });
    const magic_db_mod = b.createModule(.{
        .root_source_file = b.path("deps/magic_db.zig"),
    });
    const inference_pkg_mod = b.createModule(.{
        .root_source_file = b.path("core/src/inference/abi.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const cuda_assets_mod = b.createModule(.{
        .root_source_file = b.path("core/cuda_assets.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Build dependencies
    const pcre2 = pcre2_port.add(b, target, optimize);
    const miniz = miniz_port.add(b, target, optimize);
    const libmagic = libmagic_port.add(b, target, optimize);
    const jpeg_turbo = jpeg_turbo_port.add(b, target, optimize);
    const spng = spng_port.add(b, target, optimize, miniz);
    const webp = webp_port.add(b, target, optimize);
    const sqlite3 = sqlite_port.add(b, target, optimize);
    const tree_sitter = tree_sitter_port.add(b, target, optimize);

    var core_pkgs = CorePackages.init(b, target, optimize);
    wireCorePackageImports(&core_pkgs);
    addCorePackageBuildImportsAndDeps(
        b,
        &core_pkgs,
        build_options_mod,
        cacert_mod,
        magic_db_mod,
        pcre2,
        miniz,
        libmagic,
        jpeg_turbo,
        spng,
        webp,
        sqlite3,
        tree_sitter,
    );
    addCorePackageImports(inference_pkg_mod, &core_pkgs);
    inference_pkg_mod.addImport("build_options", build_options_mod);
    addCDependencies(b, inference_pkg_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

    // ==========================================================================
    // Internal inference library (inference + models + compute)
    // ==========================================================================
    const inference_mod = b.createModule(.{
        .root_source_file = b.path("core/src/inference/boundary.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    inference_mod.addImport("cuda_assets", cuda_assets_mod);
    inference_mod.addImport("inference_pkg", inference_pkg_mod);
    addCorePackageImports(inference_mod, &core_pkgs);
    inference_mod.addImport("build_options", build_options_mod);
    addCDependencies(b, inference_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

    const inference_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "talu-inference",
        .root_module = inference_mod,
    });
    // Keep this archive self-contained for internal linking; external archives
    // are linked by final binaries/shared libs.
    linkCDependencies(b, inference_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, true);
    addMetalSupport(b, inference_mod, inference_lib, enable_metal);

    const inference_step = b.step("inference", "Build internal inference library (inference + models + compute)");
    inference_step.dependOn(&b.addInstallArtifact(inference_lib, .{}).step);

    // ==========================================================================
    // Native shared library
    // ==========================================================================
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    lib_mod.addImport("cuda_assets", cuda_assets_mod);
    lib_mod.addImport("inference_pkg", inference_pkg_mod);
    addCorePackageImports(lib_mod, &core_pkgs);
    lib_mod.addImport("build_options", build_options_mod);
    addCDependencies(b, lib_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

    const lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "talu",
        .root_module = lib_mod,
    });
    linkCDependencies(b, lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
    addMetalSupport(b, lib_mod, lib, enable_metal);

    const install_lib = b.addInstallArtifact(lib, .{});
    b.getInstallStep().dependOn(&install_lib.step);

    // ==========================================================================
    // Python install step - copies library to bindings/python/talu/
    // ==========================================================================
    const python_install_step = b.step("python-install", "Build and copy shared library to Python bindings");
    python_install_step.dependOn(&install_lib.step);

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

    const mlx_metallib_path = if (target.result.os.tag == .macos) findMlxMetallib(b) else null;
    if (mlx_metallib_path) |path| {
        // MLX resolves the default metallib relative to the loaded shared
        // library image path (libtalu.dylib), so keep mlx.metallib colocated
        // with libtalu in zig-out/lib.
        const copy_mlx_metallib_lib = b.addInstallFileWithDir(
            .{ .cwd_relative = path },
            .lib,
            "mlx.metallib",
        );
        install_lib.step.dependOn(&copy_mlx_metallib_lib.step);

        // Keep Python wheel/runtime layout self-contained as well.
        const copy_mlx_metallib_python = b.addInstallFile(
            .{ .cwd_relative = path },
            "../bindings/python/talu/mlx.metallib",
        );
        python_install_step.dependOn(&copy_mlx_metallib_python.step);
    }

    // ==========================================================================
    // Native static library
    // ==========================================================================
    const static_lib_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    static_lib_mod.addImport("cuda_assets", cuda_assets_mod);
    static_lib_mod.addImport("inference_pkg", inference_pkg_mod);
    addCorePackageImports(static_lib_mod, &core_pkgs);

    static_lib_mod.addImport("build_options", build_options_mod);
    addCDependencies(b, static_lib_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

    const static_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "talu",
        .root_module = static_lib_mod,
    });
    // Skip external .a archives for static lib - they'll be linked by the final executable
    linkCDependencies(b, static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, true);
    addMetalSupport(b, static_lib_mod, static_lib, enable_metal);

    const static_step = b.step("static", "Build static library");
    static_step.dependOn(&b.addInstallArtifact(static_lib, .{}).step);
    const core_step = b.step("core", "Build core shared library/runtime artifact");
    core_step.dependOn(&install_lib.step);

    // ==========================================================================
    // Native CLI (requires cargo/Rust toolchain)
    // ==========================================================================
    const has_cargo = if (b.findProgram(&.{"cargo"}, &.{})) |_| true else |_| false;
    var install_exe_step: ?*std.Build.Step = null;

    if (has_cargo) {
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
        // Link CLI against the shared core library to avoid compiling core twice
        // (shared + static) in the release path.
        exe.linkLibrary(lib);
        // Link all deps including external .a archives (curl, mbedtls)
        linkCDependencies(b, exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        exe.addObjectFile(b.path("bindings/rust/target/release/libtalu_cli.a"));

        // Link platform-specific runtime libraries for Rust CLI
        const cli_target_os = exe.rootModuleTarget().os.tag;
        if (cli_target_os == .linux) {
            exe.root_module.addRPathSpecial("$ORIGIN/../lib");
            exe.linkSystemLibrary("unwind");
            exe.linkSystemLibrary("gcc_s");
            exe.linkSystemLibrary("dl");
            exe.linkSystemLibrary("pthread");
            exe.linkSystemLibrary("m");
        } else if (cli_target_os == .macos) {
            exe.root_module.addRPathSpecial("@executable_path/../lib");
        }
        // macOS frameworks already linked via linkCDependencies

        const install_exe = b.addInstallArtifact(exe, .{});
        b.getInstallStep().dependOn(&install_exe.step);
        install_exe_step = &install_exe.step;

        if (mlx_metallib_path) |path| {
            const copy_mlx_metallib_bin = b.addInstallFileWithDir(
                .{ .cwd_relative = path },
                .bin,
                "mlx.metallib",
            );
            // Keep mlx.metallib next to CLI executable for direct colocated
            // lookup paths as well.
            install_exe.step.dependOn(&copy_mlx_metallib_bin.step);
        }

        const run_cmd = b.addSystemCommand(&.{b.getInstallPath(.bin, "talu")});
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            for (args) |arg| {
                run_cmd.addArg(arg);
            }
        }
        const run_step = b.step("run", "Run the CLI executable");
        run_step.dependOn(&run_cmd.step);
    } else {
        std.debug.print("NOTE: cargo not found — skipping CLI build (library still builds)\n", .{});
    }

    // ==========================================================================
    // Release step - recommended full build
    // ==========================================================================
    // Builds library, copies to Python, and builds CLI - all in ReleaseFast.
    // This is the recommended build command for development and agents.
    const release_step = b.step("release", "Build library + CLI and copy to Python (recommended)");
    release_step.dependOn(inference_step);
    release_step.dependOn(core_step);
    release_step.dependOn(python_install_step);
    if (install_exe_step) |step| {
        release_step.dependOn(step);
    }

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
        dump_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
        dump_build_options.addOption(bool, "debug_matmul", debug_matmul);
        dump_build_options.addOption(bool, "dump_tensors", true); // Always true for dump binary
        dump_build_options.addOption(bool, "xray_bridge", xray_bridge);
        dump_build_options.addOption([]const u8, "version", version);

        // Static library with dump instrumentation
        const dump_lib_mod = b.createModule(.{
            .root_source_file = b.path("core/src/lib_dev.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        dump_lib_mod.addImport("cuda_assets", cuda_assets_mod);
        dump_lib_mod.addImport("inference_pkg", inference_pkg_mod);
        addCorePackageImports(dump_lib_mod, &core_pkgs);
        dump_lib_mod.addOptions("build_options", dump_build_options);
        addCDependencies(b, dump_lib_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

        const dump_static_lib = b.addLibrary(.{
            .linkage = .static,
            .name = "talu-dump-core",
            .root_module = dump_lib_mod,
        });
        // Add C sources to static lib with skip_external_archives=true (matches main build pattern)
        linkCDependencies(b, dump_static_lib, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, true);
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
        linkExternalArchives(b, dump_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);
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
    unit_test_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
    unit_test_build_options.addOption(bool, "debug_matmul", debug_matmul);
    unit_test_build_options.addOption(bool, "dump_tensors", dump_tensors);
    unit_test_build_options.addOption(bool, "xray_bridge", xray_bridge);
    unit_test_build_options.addOption([]const u8, "version", version);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&FailStep.create(b,
        \\
        \\`zig build test` is disabled — the monolithic binary exceeds LLVM
        \\release-mode memory limits. Use per-module test steps instead:
        \\
        \\  zig build test-<module> -Drelease
        \\
        \\Available modules:
        \\  tokenizer
        \\  validate
        \\  io
        \\  db
        \\  collab
        \\  template
        \\  policy
        \\  models
        \\  responses
        \\  converter
        \\  xray
        \\  image
        \\  compute
        \\  inference
        \\  inference-cpu
        \\  inference-metal
        \\  inference-cuda
        \\  agent
        \\
        \\Example: zig build test-db -Drelease
        \\
    ).step);

    // Per-module test steps — each module compiled as a separate LLVM unit
    // so release-mode optimization stays within memory limits.
    // `zig build test-<name> -Drelease` to run a single module.
    //
    // Excluded from per-module (debug-only via `zig build test`):
    //   capi   — aggregates all modules
    //   router — imports inference → full backend chain
    const ut = UnitTestCfg{
        .b = b,
        .target = target,
        .optimize = optimize,
        .build_options = unit_test_build_options,
        .cuda_assets_mod = cuda_assets_mod,
        .inference_pkg_mod = inference_pkg_mod,
        .cacert_mod = cacert_mod,
        .magic_db_mod = magic_db_mod,
        .core_pkgs = &core_pkgs,
        .pcre2 = pcre2,
        .miniz = miniz,
        .libmagic = libmagic,
        .jpeg_turbo = jpeg_turbo,
        .spng = spng,
        .webp = webp,
        .sqlite3 = sqlite3,
        .tree_sitter = tree_sitter,
    };
    ut.addLazy("tokenizer", b.path("core/src/lib_dev.zig"), &.{
        "wordpiece",
        "tokenizer",
        "SentencePiece",
    });
    ut.addLazy("validate", b.path("core/src/lib_dev.zig"), &.{
        "sampler",
        "mask ",
        "grammar",
    });
    ut.addLazy("io", b.path("core/src/lib_dev.zig"), &.{
        "Http",
        "fetch ",
        "download",
    });
    ut.addLazy("db", b.path("core/src/lib_dev.zig"), &.{
        "index build",
        "vector",
        "bench",
    });
    ut.addLazy("collab", b.path("core/src/lib_dev.zig"), &.{
        "ResourceStore",
        "SessionStore",
        "OperationEnvelope",
        "StorageLane",
        "TextCrdt",
        "LamportClock",
        "collab",
    });
    ut.addLazy("template", b.path("core/src/lib_dev.zig"), &.{
        "TemplateInput",
        "TemplateTokenizer",
        "template",
    });
    ut.addLazy("policy", b.path("core/src/lib_dev.zig"), &.{
        "parsePolicy",
        "globMatch",
        "Policy.evaluate",
        "evaluate ",
    });
    ut.addLazy("models", b.path("core/src/lib_dev.zig"), &.{
        "registry",
        "compileLayerProgram",
        "plan",
    });
    ut.addLazy("responses", b.path("core/src/lib_dev.zig"), &.{
        "TableAdapter",
        "responses",
        "storage",
        "toolsToGrammarSchema",
        "normalizeToolsJson",
        "generateCallId",
        "parseToolCall",
    });
    ut.addLazy("converter", b.path("core/src/lib_dev.zig"), &.{
        "QuantConfig",
        "ConvertOptions",
        "converter",
    });
    ut.addLazy("xray", b.path("core/src/lib_dev.zig"), &.{
        "estimatePerf",
        "LayerGeometry",
        "xray",
    });
    ut.addLazy("image", b.path("core/src/lib_dev.zig"), &.{
        "stripAlpha",
        "compositeRgbaToRgb",
        "image",
    });
    ut.addLazy("compute", b.path("core/src/lib_dev.zig"), &.{
        "validateArgs",
        "mmap",
        "compute",
    });
    // The inference subtree imports shared modules via `../../..` paths.
    // Building tests from `core/src/inference/root.zig` as module root trips
    // Zig's module-path guard ("import of file outside module path").
    // Run inference tests from `core/src/lib_dev.zig` with focused filters instead.
    ut.addLazy("inference", b.path("core/src/lib_dev.zig"), &.{
        "inference",
        "Scheduler",
        "Vision",
        "layer program",
    });
    ut.addLazy("inference-cpu", b.path("core/src/lib_dev.zig"), &.{
        "inference.backend.cpu",
    });
    ut.addLazy("inference-metal", b.path("core/src/lib_dev.zig"), &.{
        "inference.backend.metal",
    });
    ut.addLazy("inference-cuda", b.path("core/src/lib_dev.zig"), &.{
        "inference.backend.cuda",
    });
    ut.addLazy("agent", b.path("core/src/lib_dev.zig"), &.{
        "ToolRegistry",
        "MessageBus",
        "buildSystemPrompt",
        "ProcessSession",
        "ShellSession",
        "checkCommand",
        "compactTurns",
        "CapabilityReport",
        "parseKernelVersion",
        "CgroupConfig",
        "ProbeReport",
        "TaluCapabilityReport",
        "validate_strict_ext",
    });
    ut.addLazy("train", b.path("core/src/lib_dev.zig"), &.{
        "GradTensor",
        "LoraLayer",
        "LoraAdapter",
        "gradWeight",
        "gradInput",
        "gradBias",
        "crossEntropy",
        "embedding",
        "rmsnorm",
        "silu",
        "gelu",
        "swiglu",
        "rope",
        "attention",
        "AdamW",
        "ParamState",
        "Scheduler",
        "TrainableParam",
        "TrainableParams",
        "routerBackward",
        "expertWeightGrad",
        "expertOutputGrad",
        "ssmBackward",
        "conv1d",
        "DataLoader",
        "Checkpoint",
        "clipGrad",
        "TrainingConfig",
        "StepMetrics",
        "TrainingSession",
        "capi_bridge",
    });
    // Build integration tests against a separate copy of core/src/lib_dev.zig.
    // Keep integration on CPU-only to avoid MLX/Metal runtime coupling and
    // ensure deterministic behavior across host GPU setups.
    const integration_build_options = b.addOptions();
    integration_build_options.addOption(bool, "enable_metal", false);
    integration_build_options.addOption(bool, "enable_cuda", enable_cuda);
    integration_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
    integration_build_options.addOption(bool, "debug_matmul", debug_matmul);
    integration_build_options.addOption(bool, "dump_tensors", dump_tensors);
    integration_build_options.addOption(bool, "xray_bridge", xray_bridge);
    integration_build_options.addOption([]const u8, "version", version);

    const integration_main_mod = b.createModule(.{
        .root_source_file = b.path("core/src/lib_dev.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    integration_main_mod.addImport("cuda_assets", cuda_assets_mod);
    integration_main_mod.addImport("inference_pkg", inference_pkg_mod);
    addCorePackageImports(integration_main_mod, &core_pkgs);
    integration_main_mod.addOptions("build_options", integration_build_options);
    addCDependencies(b, integration_main_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

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
    linkCDependencies(b, integration_tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);

    const run_integration_tests = b.addRunArtifact(integration_tests);
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_integration_tests.step);

    // Metal-focused inference integration tests (opt-in lane).
    const integration_metal_step = b.step(
        "test-integration-inference-metal",
        "Run inference backend Metal integration tests",
    );
    if (target.result.os.tag == .macos and enable_metal) {
        const integration_metal_build_options = b.addOptions();
        integration_metal_build_options.addOption(bool, "enable_metal", true);
        integration_metal_build_options.addOption(bool, "enable_cuda", enable_cuda);
        integration_metal_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
        integration_metal_build_options.addOption(bool, "debug_matmul", debug_matmul);
        integration_metal_build_options.addOption(bool, "dump_tensors", dump_tensors);
        integration_metal_build_options.addOption(bool, "xray_bridge", xray_bridge);
        integration_metal_build_options.addOption([]const u8, "version", version);

        const integration_metal_main_mod = b.createModule(.{
            .root_source_file = b.path("core/src/lib_dev.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        integration_metal_main_mod.addImport("cuda_assets", cuda_assets_mod);
        integration_metal_main_mod.addImport("inference_pkg", inference_pkg_mod);
        addCorePackageImports(integration_metal_main_mod, &core_pkgs);
        integration_metal_main_mod.addOptions("build_options", integration_metal_build_options);
        addCDependencies(b, integration_metal_main_mod, cacert_mod, magic_db_mod, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter);

        const integration_metal_test_mod = b.createModule(.{
            .root_source_file = b.path("core/tests/inference/backend/metal/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "main", .module = integration_metal_main_mod },
            },
        });

        const integration_metal_tests = b.addTest(.{
            .name = "test-integration-inference-metal",
            .root_module = integration_metal_test_mod,
        });
        linkCDependencies(b, integration_metal_tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        addMetalSupport(b, integration_metal_main_mod, integration_metal_tests, true);
        const install_integration_metal_tests = b.addInstallArtifact(integration_metal_tests, .{});

        const run_integration_metal_tests = b.addSystemCommand(&.{
            b.getInstallPath(.bin, "test-integration-inference-metal"),
        });
        run_integration_metal_tests.step.dependOn(&install_integration_metal_tests.step);
        if (mlx_metallib_path) |path| {
            run_integration_metal_tests.setEnvironmentVariable("MLX_METALLIB", path);
        }
        // Harden allocator diagnostics for intermittent heap corruption on Metal paths.
        run_integration_metal_tests.setEnvironmentVariable("MallocScribble", "1");
        run_integration_metal_tests.setEnvironmentVariable("MallocGuardEdges", "1");
        run_integration_metal_tests.setEnvironmentVariable("MallocCheckHeapStart", "1");
        run_integration_metal_tests.setEnvironmentVariable("MallocCheckHeapEach", "1");
        integration_metal_step.dependOn(&run_integration_metal_tests.step);
    } else {
        integration_metal_step.dependOn(&FailStep.create(
            b,
            "test-integration-inference-metal requires macOS with Metal enabled (-Dmetal=true)\n",
        ).step);
    }

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
        linkCDependencies(b, models_report_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
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
        perf_exe_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
        perf_exe_build_options.addOption(bool, "xray_bridge", xray_bridge);

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
        linkCDependencies(b, perf_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
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
        perf_test_build_options.addOption(bool, "cuda_startup_selftests", cuda_startup_selftests);
        perf_test_build_options.addOption(bool, "xray_bridge", xray_bridge);

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
        linkCDependencies(b, perf_tests, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        const run_perf_tests = b.addRunArtifact(perf_tests);
        const perf_test_step = b.step("test-perf-sanity", "Run perf/lifecycle sanity tests");
        perf_test_step.dependOn(&run_perf_tests.step);
        // Keep perf/determinism guardrails part of integration validation.
        integration_test_step.dependOn(&run_perf_tests.step);
    } else |_| {
        // perf sanity checker not present - skip
    }

    // ==========================================================================
    // CPU compute benchmark harness (core/bench/compute/cpu/)
    // ==========================================================================
    const bench_cpu_compute_path = "core/bench/compute/cpu/root.zig";
    if (std.fs.cwd().access(bench_cpu_compute_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});
        const bench_cpu_compute_mod = b.createModule(.{
            .root_source_file = b.path(bench_cpu_compute_path),
            .target = host_target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        const bench_cpu_compute_exe = b.addExecutable(.{
            .name = "bench-cpu-compute",
            .root_module = bench_cpu_compute_mod,
        });
        linkCDependencies(b, bench_cpu_compute_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        addMetalSupport(b, bench_cpu_compute_mod, bench_cpu_compute_exe, enable_metal);
        b.installArtifact(bench_cpu_compute_exe);

        const run_bench_cpu_compute = b.addRunArtifact(bench_cpu_compute_exe);
        if (b.args) |args| {
            for (args) |arg| {
                run_bench_cpu_compute.addArg(arg);
            }
        }
        const bench_cpu_compute_step = b.step("bench-cpu-compute", "Run CPU compute benchmark harness");
        bench_cpu_compute_step.dependOn(&run_bench_cpu_compute.step);
    } else |_| {
        // CPU compute benchmark harness not present - skip
    }

    // ==========================================================================
    // Metal compute benchmark harness (core/bench/compute/metal/)
    // ==========================================================================
    const bench_metal_compute_path = "core/bench/compute/metal/root.zig";
    if (std.fs.cwd().access(bench_metal_compute_path, .{})) |_| {
        const host_target = b.resolveTargetQuery(.{});
        const bench_metal_compute_mod = b.createModule(.{
            .root_source_file = b.path(bench_metal_compute_path),
            .target = host_target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        const bench_metal_compute_exe = b.addExecutable(.{
            .name = "bench-metal-compute",
            .root_module = bench_metal_compute_mod,
        });
        addMetalSupport(b, bench_metal_compute_mod, bench_metal_compute_exe, enable_metal);
        const install_bench_metal_compute = b.addInstallArtifact(bench_metal_compute_exe, .{});
        const run_bench_metal_compute = b.addSystemCommand(&.{b.getInstallPath(.bin, "bench-metal-compute")});
        run_bench_metal_compute.step.dependOn(&install_bench_metal_compute.step);
        if (mlx_metallib_path) |path| {
            const copy_bench_mlx_metallib = b.addInstallFileWithDir(
                .{ .cwd_relative = path },
                .bin,
                "mlx.metallib",
            );
            run_bench_metal_compute.step.dependOn(&copy_bench_mlx_metallib.step);
        }
        if (b.args) |args| {
            for (args) |arg| {
                run_bench_metal_compute.addArg(arg);
            }
        }
        const bench_metal_compute_step = b.step("bench-metal-compute", "Run Metal compute benchmark harness");
        bench_metal_compute_step.dependOn(&run_bench_metal_compute.step);
    } else |_| {
        // compute benchmark harness not present - skip
    }

    // ==========================================================================
    // Tokenizer benchmark harness (core/bench/tokenizer/)
    // ==========================================================================
    const bench_tokenizer_path = "core/bench/tokenizer/root.zig";
    if (std.fs.cwd().access(bench_tokenizer_path, .{})) |_| {
        const host_target_tok = b.resolveTargetQuery(.{});
        const bench_tokenizer_mod = b.createModule(.{
            .root_source_file = b.path(bench_tokenizer_path),
            .target = host_target_tok,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        const bench_tokenizer_exe = b.addExecutable(.{
            .name = "bench-tokenizer",
            .root_module = bench_tokenizer_mod,
        });
        linkCDependencies(b, bench_tokenizer_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        b.installArtifact(bench_tokenizer_exe);

        const run_bench_tokenizer = b.addRunArtifact(bench_tokenizer_exe);
        if (b.args) |args| {
            for (args) |arg| {
                run_bench_tokenizer.addArg(arg);
            }
        }
        const bench_tokenizer_step = b.step("bench-tokenizer", "Run tokenizer benchmark harness");
        bench_tokenizer_step.dependOn(&run_bench_tokenizer.step);
    } else |_| {
        // tokenizer benchmark harness not present - skip
    }

    // ==========================================================================
    // Training benchmark harness (core/bench/train/)
    // ==========================================================================
    const bench_train_path = "core/bench/train/root.zig";
    if (std.fs.cwd().access(bench_train_path, .{})) |_| {
        const host_target_train = b.resolveTargetQuery(.{});
        const bench_train_mod = b.createModule(.{
            .root_source_file = b.path(bench_train_path),
            .target = host_target_train,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "main", .module = integration_main_mod },
            },
        });
        const bench_train_exe = b.addExecutable(.{
            .name = "bench-train",
            .root_module = bench_train_mod,
        });
        linkCDependencies(b, bench_train_exe, pcre2, miniz, libmagic, jpeg_turbo, spng, webp, sqlite3, tree_sitter, false);
        b.installArtifact(bench_train_exe);

        const run_bench_train = b.addRunArtifact(bench_train_exe);
        if (b.args) |args| {
            for (args) |arg| {
                run_bench_train.addArg(arg);
            }
        }
        const bench_train_step = b.step("bench-train", "Run training benchmark harness");
        bench_train_step.dependOn(&run_bench_train.step);
    } else |_| {
        // training benchmark harness not present - skip
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
