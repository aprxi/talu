#[path = "../build_support/version.rs"]
mod version_support;

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
    let repo_root = rust_root
        .parent()
        .and_then(|path| path.parent())
        .expect("Could not find repo root")
        .to_path_buf();
    println!("cargo:rerun-if-changed={}", version_path.display());
    println!("cargo:rerun-if-env-changed=TALU_VERSION_OVERRIDE");
    println!("cargo:rerun-if-env-changed=GITHUB_ACTIONS");
    version_support::emit_git_rerun_hints(&repo_root);

    let version = env::var("TALU_VERSION_OVERRIDE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let base_version = version_support::read_base_version(&version_path);
            version_support::local_compiled_version(&repo_root, &base_version)
        });
    println!("cargo:rustc-env=TALU_VERSION={}", version);
}
