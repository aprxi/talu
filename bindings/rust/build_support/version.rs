use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

pub struct VersionSelection<'a> {
    pub override_version: Option<&'a str>,
    pub base_version: &'a str,
    pub github_actions: bool,
    pub short_commit: Option<&'a str>,
    pub dirty: bool,
    pub exact_version_tag: bool,
    pub diff_hash: Option<&'a str>,
}

pub fn read_base_version(version_path: &Path) -> String {
    let content = fs::read_to_string(version_path).expect("Failed to read vendor/VERSION");
    let version = content.trim();
    if version.is_empty() {
        "0.0.0".to_string()
    } else {
        version.to_string()
    }
}

pub fn git_stdout(repo_root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn tracked_dirty(repo_root: &Path) -> bool {
    let status = Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .current_dir(repo_root)
        .status();
    matches!(status, Ok(status) if !status.success())
}

pub fn tracked_diff_hash(repo_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["diff", "--binary", "HEAD", "--"])
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }

    let mut hash: u64 = 14695981039346656037;
    for byte in output.stdout {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(1099511628211);
    }

    Some(format!("{:08x}", hash as u32))
}

pub fn exact_version_tag(repo_root: &Path, version: &str) -> bool {
    let expected = format!("v{version}");
    git_stdout(
        repo_root,
        &[
            "describe",
            "--tags",
            "--exact-match",
            "--match",
            &expected,
            "HEAD",
        ],
    )
    .as_deref()
        == Some(expected.as_str())
}

pub fn emit_git_rerun_hints(repo_root: &Path) {
    let git_dir = repo_root.join(".git");
    let head_path = git_dir.join("HEAD");
    if !head_path.exists() {
        return;
    }

    println!("cargo:rerun-if-changed={}", head_path.display());
    if let Ok(head) = fs::read_to_string(&head_path) {
        if let Some(reference) = head.trim().strip_prefix("ref: ") {
            println!(
                "cargo:rerun-if-changed={}",
                git_dir.join(reference).display()
            );
        }
    }
}

pub fn select_compiled_version(selection: VersionSelection<'_>) -> String {
    if let Some(version) = selection
        .override_version
        .filter(|value| !value.trim().is_empty())
    {
        return version.to_string();
    }

    if selection.github_actions {
        return selection.base_version.to_string();
    }

    let short_commit = selection.short_commit.unwrap_or("unknown");
    if !selection.dirty && selection.exact_version_tag {
        return selection.base_version.to_string();
    }
    if !selection.dirty {
        return format!("{}-local.{short_commit}", selection.base_version);
    }

    match selection.diff_hash {
        Some(diff_hash) => format!(
            "{}-local.{short_commit}.{diff_hash}",
            selection.base_version
        ),
        None => format!("{}-local.{short_commit}.dirty", selection.base_version),
    }
}

pub fn local_compiled_version(repo_root: &Path, base_version: &str) -> String {
    let github_actions = env::var("GITHUB_ACTIONS").ok().as_deref() == Some("true");
    let override_version = env::var("TALU_VERSION_OVERRIDE").ok();
    let short_commit = git_stdout(repo_root, &["rev-parse", "--short=10", "HEAD"]);
    let dirty = tracked_dirty(repo_root);
    let exact_version_tag = !dirty && exact_version_tag(repo_root, base_version);
    let diff_hash = if dirty {
        tracked_diff_hash(repo_root)
    } else {
        None
    };

    select_compiled_version(VersionSelection {
        override_version: override_version.as_deref(),
        base_version,
        github_actions,
        short_commit: short_commit.as_deref(),
        dirty,
        exact_version_tag,
        diff_hash: diff_hash.as_deref(),
    })
}
