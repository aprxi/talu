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
    // bindings/rust -> bindings -> repo root
    let repo_root = rust_root
        .parent()
        .expect("Could not find bindings dir")
        .parent()
        .expect("Could not find repo root")
        .to_path_buf();

    // --- Bundled UI detection ---
    // Set cfg(bundled_ui) when ui/dist/ contains the required files.
    // This lets http.rs conditionally compile include_bytes!() for the
    // console UI, so a fresh clone without `make ui` still builds.
    let ui_dist = repo_root.join("ui").join("dist");
    println!("cargo:rerun-if-changed={}", ui_dist.display());
    // Track individual UI files so cargo rebuilds when their contents change.
    for file in ["index.html", "main.js", "style.css"] {
        println!("cargo:rerun-if-changed={}", ui_dist.join(file).display());
    }
    println!("cargo:rustc-check-cfg=cfg(bundled_ui)");
    if has_ui_files(&ui_dist) {
        println!("cargo:rustc-cfg=bundled_ui");
    }

    // --- VERSION -> TALU_VERSION ---
    // VERSION file is at bindings/rust/vendor/VERSION (self-contained)
    let version_path = rust_root.join("vendor").join("VERSION");
    println!("cargo:rerun-if-changed={}", version_path.display());

    let content = std::fs::read_to_string(&version_path).expect("Failed to read vendor/VERSION");
    let version = content.trim();
    let version = if version.is_empty() { "0.0.0" } else { version };
    println!("cargo:rustc-env=TALU_VERSION={}", version);
}

/// Check whether `ui/dist/` contains any of the required UI files.
fn has_ui_files(dist_path: &std::path::Path) -> bool {
    if !dist_path.is_dir() {
        return false;
    }
    for file in ["index.html", "main.js", "style.css"] {
        if dist_path.join(file).is_file() {
            return true;
        }
    }
    false
}
