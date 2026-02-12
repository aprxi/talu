use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    // bindings/rust/talu-sys -> bindings/rust
    let rust_root = manifest_dir
        .parent()
        .expect("Could not find rust root")
        .to_path_buf();

    // --- VERSION -> TALU_VERSION ---
    // VERSION file is at bindings/rust/vendor/VERSION (self-contained)
    let version_path = rust_root.join("vendor").join("VERSION");
    println!("cargo:rerun-if-changed={}", version_path.display());

    let content = fs::read_to_string(&version_path).expect("Failed to read vendor/VERSION");
    let version = content.trim();
    let version = if version.is_empty() { "0.0.0" } else { version };
    println!("cargo:rustc-env=TALU_VERSION={}", version);

    // --- Ensure Zig has built the shared lib (libtalu.so) ---
    // bindings/rust -> bindings -> repo root
    let repo_root = rust_root
        .parent()
        .and_then(|p| p.parent())
        .expect("Could not find repo root");
    let zig_out_lib = repo_root.join("zig-out").join("lib");
    let so = zig_out_lib.join(if cfg!(target_os = "macos") {
        "libtalu.dylib"
    } else if cfg!(target_os = "windows") {
        "talu.dll"
    } else {
        "libtalu.so"
    });

    println!("cargo:rerun-if-changed={}", so.display());

    // --- Link to the shared library ---
    println!("cargo:rustc-link-search=native={}", zig_out_lib.display());
    println!("cargo:rustc-link-lib=dylib=talu");

    // --- Expose lib dir to downstream crates ---
    // DEP_TALU_LIB_DIR: Downstream build scripts (talu, cli) read this to
    // embed rpath in their binaries/tests. Uses the DEP_<LINKS>_<KEY>
    // mechanism (links = "talu" -> DEP_TALU_LIB_DIR).
    println!("cargo:lib_dir={}", zig_out_lib.display());
}
