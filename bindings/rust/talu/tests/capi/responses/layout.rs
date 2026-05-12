//! Test module layout invariants for `capi::responses`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn nested_response_test_files_are_declared_by_parent_modules() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/capi/responses");
    let files = rust_files(&root);

    for file in files {
        if file.file_name().and_then(|name| name.to_str()) == Some("mod.rs") {
            continue;
        }
        if file.parent() == Some(root.as_path()) {
            continue;
        }

        let module_name = file.file_stem().unwrap().to_str().unwrap();
        let parent_mod = file.parent().unwrap().join("mod.rs");
        let declarations = module_declarations(&parent_mod);
        assert!(
            declarations.contains(module_name),
            "{} is not declared by {}",
            file.display(),
            parent_mod.display()
        );
    }
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_rust_files(root, &mut out);
    out.sort();
    out
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn module_declarations(path: &Path) -> BTreeSet<String> {
    let source = fs::read_to_string(path).unwrap();
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let declaration = line
                .strip_prefix("mod ")
                .or_else(|| line.strip_prefix("pub mod "))?;
            declaration
                .strip_suffix(';')
                .map(|name| name.trim().to_string())
        })
        .collect()
}
