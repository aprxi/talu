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
    let lib_dir = match env::var("TALU_LIB_DIR") {
        Ok(dir) => {
            // Local / monorepo development: use the directory as-is.
            let lib_dir = PathBuf::from(dir);
            if !lib_dir.exists() {
                panic!(
                    "TALU_LIB_DIR is set to '{}' but that directory does not exist. \
                     Build libtalu first (zig build release -Drelease).",
                    lib_dir.display()
                );
            }
            lib_dir
        }
        Err(_) => {
            // crates.io / external consumer: download pre-built library from GitHub.
            download_from_github()
        }
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
