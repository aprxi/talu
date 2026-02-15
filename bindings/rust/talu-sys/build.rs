use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));

    // --- VERSION -> TALU_VERSION ---
    // In the monorepo, vendor/VERSION is at ../vendor/VERSION relative to talu-sys/.
    // On crates.io the monorepo structure doesn't exist, so fall back to CARGO_PKG_VERSION.
    let version = manifest_dir
        .parent()
        .map(|rust_root| rust_root.join("vendor").join("VERSION"))
        .filter(|p| p.exists())
        .and_then(|p| fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into()));

    println!("cargo:rustc-env=TALU_VERSION={}", version);

    // --- Locate the shared library ---
    // 1. Explicit env var (TALU_LIB_DIR) takes priority.
    // 2. Monorepo detection: navigate up from talu-sys/ to find zig-out/lib/.
    // 3. External consumer: download pre-built library from GitHub Releases.
    let lib_dir = if let Some(lib_dir) = env::var("TALU_LIB_DIR")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
    {
        // Explicit env var pointing to an existing directory.
        lib_dir
    } else if let Some(lib_dir) = find_monorepo_lib(&manifest_dir) {
        // Monorepo: zig-out/lib/ found relative to talu-sys/.
        lib_dir
    } else {
        // External consumer (crates.io): download from GitHub Releases.
        download_from_github()
    };

    // --- Link to the shared library ---
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=talu");

    // --- Expose lib dir to downstream crates ---
    // DEP_TALU_LIB_DIR: Downstream build scripts (talu, cli) read this to
    // embed rpath in their binaries/tests. Uses the DEP_<LINKS>_<KEY>
    // mechanism (links = "talu" -> DEP_TALU_LIB_DIR).
    println!("cargo:lib_dir={}", lib_dir.display());
}

/// Walk up from talu-sys/Cargo.toml to detect the monorepo and return
/// the zig-out/lib/ path. Returns None outside the monorepo (crates.io).
///
/// Note: the library file may not exist yet when build.rs runs — the Zig
/// build installs it after invoking cargo. The linker only needs it when
/// the final binary links, by which point the file is in place.
fn find_monorepo_lib(manifest_dir: &PathBuf) -> Option<PathBuf> {
    // talu-sys -> rust -> bindings -> repo root
    let repo_root = manifest_dir.parent()?.parent()?.parent()?;
    if repo_root.join("build.zig").exists() {
        Some(repo_root.join("zig-out").join("lib"))
    } else {
        None
    }
}

/// Download the pre-built libtalu archive from the matching GitHub Release
/// and extract it into OUT_DIR. Returns the directory containing the library.
fn download_from_github() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let pkg_version = env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION not set");
    let target = env::var("TARGET").expect("TARGET not set");

    let lib_name = if target.contains("apple") {
        "libtalu.dylib"
    } else if target.contains("windows") {
        "talu.dll"
    } else {
        "libtalu.so"
    };

    // Skip download if the library already exists in OUT_DIR (cached from a previous build).
    if out_dir.join(lib_name).exists() {
        return out_dir.clone();
    }

    let archive_name = format!("libtalu-{}.tar.gz", target);
    let url = format!(
        "https://github.com/aprxi/talu/releases/download/v{}/{}",
        pkg_version, archive_name
    );
    let archive_path = out_dir.join(&archive_name);

    // --- Download ---
    eprintln!("Downloading {} ...", url);
    let curl_status = Command::new("curl")
        .args([
            "-sfL",
            "-o",
            archive_path.to_str().expect("non-UTF-8 OUT_DIR"),
            &url,
        ])
        .status()
        .expect("failed to run curl — is it installed?");

    if !curl_status.success() {
        panic!(
            "Failed to download libtalu from {}.\n\
             Ensure a GitHub Release tagged v{} exists with the asset '{}'.\n\
             For local development, set TALU_LIB_DIR to point at your zig-out/lib/ directory.",
            url, pkg_version, archive_name
        );
    }

    // --- Extract ---
    let tar_status = Command::new("tar")
        .args([
            "xzf",
            archive_path.to_str().unwrap(),
            "-C",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run tar — is it installed?");

    if !tar_status.success() {
        panic!(
            "Failed to extract {}. The archive may be corrupt.",
            archive_path.display()
        );
    }

    // Clean up the archive.
    let _ = fs::remove_file(&archive_path);

    assert!(
        out_dir.join(lib_name).exists(),
        "Archive extracted but {} not found in {}. \
         Expected the archive to contain '{}' at the root.",
        lib_name,
        out_dir.display(),
        lib_name
    );

    out_dir
}
