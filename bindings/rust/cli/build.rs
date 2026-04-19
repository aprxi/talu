use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    // bindings/rust/cli -> bindings/rust
    let rust_root = manifest_dir
        .parent()
        .expect("Could not find rust root")
        .to_path_buf();

    // --- VERSION -> TALU_VERSION ---
    // VERSION file is at bindings/rust/vendor/VERSION (self-contained)
    let version_path = rust_root.join("vendor").join("VERSION");
    println!("cargo:rerun-if-changed={}", version_path.display());

    let content = std::fs::read_to_string(&version_path).expect("Failed to read vendor/VERSION");
    let version = content.trim();
    let version = if version.is_empty() { "0.0.0" } else { version };
    println!("cargo:rustc-env=TALU_VERSION={}", version);
}
