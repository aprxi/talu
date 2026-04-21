#[allow(dead_code)]
#[path = "../../build_support/version.rs"]
mod version_support;

use version_support::{select_compiled_version, VersionSelection};

#[test]
fn github_actions_preserves_override_version() {
    let version = select_compiled_version(VersionSelection {
        override_version: Some("0.0.3-post.202604211530"),
        base_version: "0.0.3",
        github_actions: true,
        short_commit: Some("c361c6f01f"),
        dirty: true,
        exact_version_tag: false,
        diff_hash: Some("9061425d"),
    });

    assert_eq!(version, "0.0.3-post.202604211530");
}

#[test]
fn github_actions_preserves_base_release_version_without_local_suffix() {
    let version = select_compiled_version(VersionSelection {
        override_version: None,
        base_version: "0.0.3",
        github_actions: true,
        short_commit: Some("c361c6f01f"),
        dirty: true,
        exact_version_tag: false,
        diff_hash: Some("9061425d"),
    });

    assert_eq!(version, "0.0.3");
}
