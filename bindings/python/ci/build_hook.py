"""Hatch build hook for compiling the native Zig library and CLI binary."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class ZigBuildHook(BuildHookInterface):
    """Build hook that compiles the Zig native library and CLI binary before packaging."""

    PLUGIN_NAME = "zig-build"

    def initialize(self, version: str, build_data: dict) -> None:
        """Run Zig build before packaging."""
        package_root = Path(self.root)

        # Sync VERSION to _version.py (for builds from sdist)
        self._sync_version(package_root)

        if self.target_name == "sdist":
            # Don't build for sdist - just include source
            return

        zig_root = self._resolve_zig_root(package_root)
        lib_name = self._get_lib_name()
        bin_name = self._get_bin_name()
        target_lib = package_root / "talu" / lib_name
        target_metallib = package_root / "talu" / "mlx.metallib"

        # Determine where the binary is
        built_bin = zig_root / "zig-out" / "bin" / bin_name
        prebuilt_bin = package_root / "bin" / bin_name

        # Check if library already exists (pre-built in CI)
        if target_lib.exists() and (built_bin.exists() or prebuilt_bin.exists()):
            self._log(f"Using existing {lib_name} and {bin_name}")
            # Add binary to shared_scripts for installation to bin/
            bin_path = prebuilt_bin if prebuilt_bin.exists() else built_bin
            build_data["shared_scripts"] = {str(bin_path): bin_name}
            # Add metallib if present (for macOS Metal GPU support)
            if target_metallib.exists():
                build_data["shared_scripts"][str(target_metallib)] = "mlx.metallib"
                self._log("Added mlx.metallib to shared_scripts")
            return

        # Build from source
        self._log("Building native library and CLI from source...")
        self._run_build(zig_root)

        # Copy library to package
        built_lib = zig_root / "zig-out" / "lib" / lib_name
        if not built_lib.exists():
            raise RuntimeError(f"Build failed: {built_lib} not found")

        shutil.copy2(built_lib, target_lib)
        self._log(f"Copied {lib_name} to talu/")

        # Copy _native.py to package (same pattern as .so)
        built_native = zig_root / "zig-out" / "lib" / "_native.py"
        target_native = package_root / "talu" / "_native.py"
        if built_native.exists():
            shutil.copy2(built_native, target_native)
            self._log("Copied _native.py to talu/")

        # Verify CLI binary exists
        if not built_bin.exists():
            raise RuntimeError(f"Build failed: {built_bin} not found")

        # Add binary to shared_scripts for installation to bin/
        # This puts it in .data/scripts/ which pip installs to the bin directory
        build_data["shared_scripts"] = {str(built_bin): bin_name}
        self._log(f"Added {bin_name} to shared_scripts")

        # Copy Metal library on macOS (required for GPU acceleration)
        # metallib goes to shared_scripts so it's installed next to the binary
        if platform.system() == "Darwin" and not os.environ.get("TALU_DISABLE_METAL"):
            self._copy_metallib(zig_root, target_metallib)
            if target_metallib.exists():
                build_data["shared_scripts"][str(target_metallib)] = "mlx.metallib"
                self._log("Added mlx.metallib to shared_scripts")

    def _get_lib_name(self) -> str:
        """Get platform-specific library name."""
        system = platform.system()
        if system == "Darwin":
            return "libtalu.dylib"
        elif system == "Windows":
            return "talu.dll"
        else:
            return "libtalu.so"

    def _get_bin_name(self) -> str:
        """Get platform-specific binary name."""
        if platform.system() == "Windows":
            return "talu.exe"
        return "talu"

    def _resolve_zig_root(self, package_root: Path) -> Path:
        """Resolve Zig project root for builds."""
        vendor_root = package_root / "vendor" / "talu"
        if (vendor_root / "build.zig").exists() and (vendor_root / "core/src").exists():
            return vendor_root

        for candidate in [package_root, *package_root.parents]:
            if (candidate / "build.zig").exists() and (candidate / "core/src").exists():
                return candidate

        raise RuntimeError(
            "Unable to locate Zig sources. "
            "Set TALU_ZIG_ROOT or populate bindings/python/vendor/talu."
        )

    def _run_build(self, zig_root: Path) -> None:
        """Run the build commands."""
        env = os.environ.copy()

        # Check for required tools
        if not shutil.which("zig"):
            raise RuntimeError("Zig compiler not found. Install from https://ziglang.org/download/")
        if not shutil.which("cmake"):
            raise RuntimeError("CMake not found. Install cmake to build from source.")
        if not shutil.which("cargo"):
            raise RuntimeError("Cargo not found. Install Rust from https://rustup.rs/")

        # Metal is enabled by default on macOS for GPU acceleration
        # Set TALU_DISABLE_METAL=1 to build CPU-only if needed
        if platform.system() == "Darwin" and os.environ.get("TALU_DISABLE_METAL"):
            self._log("Building CPU-only (TALU_DISABLE_METAL is set)")

        # Run make cli (includes deps, builds library and CLI binary)
        self._log("Building with make cli...")
        subprocess.run(
            ["make", "build"],
            cwd=zig_root,
            env=env,
            check=True,
        )

        # Strip binaries
        if shutil.which("strip"):
            lib_path = zig_root / "zig-out" / "lib" / self._get_lib_name()
            bin_path = zig_root / "zig-out" / "bin" / self._get_bin_name()
            self._log("Stripping binaries...")
            # macOS strip needs -x for dylibs: full strip fails on
            # indirect symbol table entries (Metal, CoreFoundation, etc.)
            is_macos = platform.system() == "Darwin"
            if lib_path.exists():
                strip_cmd = ["strip", "-x", str(lib_path)] if is_macos else ["strip", str(lib_path)]
                subprocess.run(strip_cmd, check=False)
            if bin_path.exists():
                subprocess.run(["strip", str(bin_path)], check=False)

    def _copy_metallib(self, zig_root: Path, target: Path) -> None:
        """Copy Metal library on macOS if available."""
        if platform.system() != "Darwin":
            return
        if target.exists():
            return

        # Check possible metallib locations
        metallib_paths = [
            zig_root
            / "deps"
            / "mlx-src"
            / "build"
            / "mlx"
            / "backend"
            / "metal"
            / "kernels"
            / "mlx.metallib",
        ]
        for src in metallib_paths:
            if src.exists():
                shutil.copy2(src, target)
                self._log(f"Copied mlx.metallib ({src.stat().st_size // 1024}KB)")
                return

    def _sync_version(self, package_root: Path) -> None:
        """Sync VERSION_PYTHON file to _version.py.

        Always generates a fresh version — never reuses an existing _version.py.

        Resolution order:
        1. vendor/talu/VERSION_PYTHON (CI sdist, pre-stamped PEP 440)
        2. Repo-root VERSION + fresh timestamp (local dev builds)
        """
        from datetime import datetime, timezone

        # CI path: pre-stamped version in sdist
        version_file = package_root / "vendor" / "talu" / "VERSION_PYTHON"
        if version_file.exists():
            version = version_file.read_text().strip()
        else:
            # Local dev: find VERSION at repo root and generate a timestamped post-release
            version_base = self._find_version_base(package_root)
            if version_base is None:
                raise RuntimeError(
                    "Cannot determine package version: no VERSION file found. "
                    "Ensure the repo root contains a VERSION file or build from an sdist."
                )
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            version = f"{version_base}.post{ts}"

        version_py = package_root / "talu" / "_version.py"
        content = f'"""Version from VERSION file."""\n__version__ = "{version}"\n'
        version_py.write_text(content)
        self._log(f"Synced version {version} to _version.py")

    def _find_version_base(self, package_root: Path) -> str | None:
        """Find the base version string from VERSION file."""
        # Check vendor first (sdist), then walk up to repo root
        for candidate in [
            package_root / "vendor" / "talu" / "VERSION",
            *(p / "VERSION" for p in [package_root, *package_root.parents]),
        ]:
            if candidate.exists():
                return candidate.read_text().strip()
        return None

    def _log(self, msg: str) -> None:
        """Log build progress."""
        print(f"[zig-build] {msg}", file=sys.stderr)
