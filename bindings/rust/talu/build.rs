use std::env;
use std::path::PathBuf;

fn main() {
    let lib_dir = match env::var("DEP_TALU_LIB_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return, // talu-sys not yet built; nothing to do
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    // OUT_DIR is e.g. target/debug/build/talu-<hash>/out.
    // Test binaries live in target/debug/deps/.
    // Walk up to target/<profile>/ then into deps/.
    let deps_dir = out_dir
        .ancestors() // .../out -> .../talu-<hash> -> .../build -> .../<profile>
        .nth(3)
        .expect("could not resolve target profile dir from OUT_DIR")
        .join("deps");

    let lib_name = if cfg!(target_os = "macos") {
        "libtalu.dylib"
    } else if cfg!(target_os = "windows") {
        "talu.dll"
    } else {
        "libtalu.so"
    };

    let src = lib_dir.join(lib_name);
    let dst = deps_dir.join(lib_name);

    // Re-run this build script whenever the native library changes so the
    // copy in target/<profile>/deps/ stays in sync with zig-out/lib/.
    println!("cargo:rerun-if-changed={}", src.display());
    // Cargo does not otherwise know that the Rust tests depend on Zig sources.
    // Track the core tokenizer sources too so Zig edits trigger a fresh copy.
    println!("cargo:rerun-if-changed=../../core/src");

    if src.exists() {
        // Always refresh the test-side copy when this build script runs.
        // Zig can preserve the installed library's mtime across rebuilds, which
        // makes an mtime-only check stale even when the contents changed.
        std::fs::copy(&src, &dst).expect("failed to copy libtalu into deps/");
    }

    // $ORIGIN: the dynamic linker looks in the directory containing the
    // executable. Works for any machine, any checkout location, and on
    // crates.io once libtalu.so is placed next to the binary.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,$ORIGIN");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg-tests=-Wl,-rpath,@executable_path");
    }
}
