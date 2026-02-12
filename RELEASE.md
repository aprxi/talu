# Release Process

## How It Works

Every merge to `main` triggers a build. The type of release depends on whether a GitHub release for the current `VERSION` already exists:

| VERSION release exists? | Result |
|------------------------|--------|
| No | Versioned release (`v0.0.4`) |
| Yes | Post-release build (`post-202601211430`) |

Post-release builds are marked as prereleases on GitHub.

## Creating a Versioned Release

1. Bump the version in `VERSION` (e.g., `0.0.3` → `0.0.4`)
2. Create a PR to `main`
3. Merge the PR

The workflow will automatically:
- Build for all platforms
- Create tag `v0.0.4`
- Create a GitHub release

All artifacts get the clean version (`0.0.4`):
- Python wheel: `talu-0.0.4-py3-none-....whl`
- Zig/Rust CLI: `talu -V` → `0.0.4`
- Python: `talu.__version__` → `0.0.4`

## Post-Release Builds

If you merge without changing `VERSION`, a post-release build is created with a timestamp:

| Artifact | Version Format | Example |
|----------|---------------|---------|
| GitHub release tag | `post-YYYYMMDDHHMM` | `post-202601211430` |
| Python wheel | PEP 440 | `0.0.3.post202601211430` |
| Zig/Rust CLI | SemVer | `0.0.3-post.202601211430` |

The timestamp `202601211430` means 2026-01-21 14:30 UTC.

## Version Queries

```bash
# Python (PEP 440 format)
python -c "import talu; print(talu.__version__)"
# Output: 0.0.3.post202601211430

# CLI - Zig or Rust (SemVer format)
talu -V
# Output: 0.0.3-post.202601211430
```

Note: Python uses PEP 440 format (`.post`) while CLI uses SemVer format (`-post.`). Both represent the same build.

## Removing a Bad Release

If a release needs to be replaced:

1. Delete the GitHub release (UI or `gh release delete v0.0.4`)
2. Delete the tag: `git push --delete origin v0.0.4`
3. Merge a fix to `main`

The next merge will recreate the release from the new commit.

Note: Only the GitHub release is checked, not the tag. Deleting just the tag won't trigger a new versioned release.

## Publishing to PyPI

After a versioned release is created, manually trigger the "Publish to PyPI" workflow with the release tag.
