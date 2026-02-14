#!/usr/bin/env bash
#
# Publish talu-sys and talu crates to crates.io.
#
# Reads the version from TALU_RELEASE_VERSION (e.g. "0.0.1-post.202602140912").
# Patches Cargo.toml files with the version, publishes, then reverts.
#
# Usage:
#   TALU_RELEASE_VERSION=0.0.1-post.202602140912 .github/scripts/publish-crates.sh
#
# Works both locally and in GitHub Actions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUST_DIR="$REPO_ROOT/bindings/rust"
cd "$RUST_DIR"

# --- Validate version ---
if [ -z "${TALU_RELEASE_VERSION:-}" ]; then
    echo "ERROR: TALU_RELEASE_VERSION is not set." >&2
    echo "Usage: TALU_RELEASE_VERSION=x.y.z ./publish.sh" >&2
    exit 1
fi

VERSION="$TALU_RELEASE_VERSION"

# Basic semver sanity check (digits.digits.digits with optional pre-release).
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
    echo "ERROR: TALU_RELEASE_VERSION='$VERSION' does not look like a valid semver version." >&2
    exit 1
fi

echo "Publishing version: $VERSION"

# --- Save originals for revert ---
cp Cargo.toml Cargo.toml.publish-backup
cp talu-sys/Cargo.toml talu-sys/Cargo.toml.publish-backup
cp talu/Cargo.toml talu/Cargo.toml.publish-backup

cleanup() {
    echo "Reverting Cargo.toml files..."
    mv -f Cargo.toml.publish-backup Cargo.toml 2>/dev/null || true
    mv -f talu-sys/Cargo.toml.publish-backup talu-sys/Cargo.toml 2>/dev/null || true
    mv -f talu/Cargo.toml.publish-backup talu/Cargo.toml 2>/dev/null || true
}
trap cleanup EXIT

# --- Patch versions ---
echo "Patching Cargo.toml files with version $VERSION..."

# 1. Workspace Cargo.toml: 0.0.0 -> $VERSION
sed -i "s/^version = \"0\.0\.0\"/version = \"$VERSION\"/" Cargo.toml

# 2. talu-sys dep in talu/Cargo.toml: add version pin
#    { path = "../talu-sys" } -> { version = "=$VERSION", path = "../talu-sys" }
sed -i "s|talu-sys = { path = \"../talu-sys\" }|talu-sys = { version = \"=$VERSION\", path = \"../talu-sys\" }|" talu/Cargo.toml

# Verify patches applied.
if grep -q 'version = "0\.0\.0"' Cargo.toml; then
    echo "ERROR: Failed to patch workspace version in Cargo.toml" >&2
    exit 1
fi
if ! grep -q "version = \"=$VERSION\"" talu/Cargo.toml; then
    echo "ERROR: Failed to patch talu-sys dependency version in talu/Cargo.toml" >&2
    exit 1
fi

echo "Patched. Workspace version: $VERSION"

# --- Publish talu-sys ---
echo ""
echo "Publishing talu-sys@$VERSION..."
cargo publish -p talu-sys --allow-dirty

# --- Wait for crates.io index ---
echo ""
echo "Waiting for talu-sys@$VERSION to appear in crates.io index..."
for i in $(seq 1 30); do
    if cargo search talu-sys 2>/dev/null | grep -q "$VERSION"; then
        echo "Found talu-sys@$VERSION in index."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: talu-sys@$VERSION not found in crates.io index after 60s." >&2
        echo "It may still be propagating. Try publishing talu manually:" >&2
        echo "  cargo publish -p talu --allow-dirty" >&2
        exit 1
    fi
    sleep 2
done

# --- Publish talu ---
echo ""
echo "Publishing talu@$VERSION..."
cargo publish -p talu --allow-dirty

echo ""
echo "Done. Published talu-sys@$VERSION and talu@$VERSION."
