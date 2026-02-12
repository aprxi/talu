#!/usr/bin/env python3
"""
Test Coverage Compliance Checker.

Verifies that source files have corresponding test files.
This is a developer-facing tool (not an auditor tool).

Usage:
    uv run python tests/check_coverage.py        # Check all
    uv run python tests/check_coverage.py -v     # Verbose

Exit codes:
    0 = All source files have tests
    1 = Missing test files detected
"""

from __future__ import annotations

import argparse
import sys

from discovery import (  # noqa: E402
    REFERENCE_ONLY_MODULES,
    TALU_SRC,
    TESTS_DIR,
    get_expected_test_file,
    get_source_files_for_module,
    get_source_modules,
)

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def supports_color() -> bool:
    """Check if terminal supports color."""
    import os

    return (
        hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    )


def colored(text: str, color: str) -> str:
    """Apply color if supported."""
    if supports_color():
        return f"{color}{text}{RESET}"
    return text


def check_coverage(verbose: bool = False) -> list[tuple[str, str]]:
    """Check that all source files have corresponding test files.

    Returns list of (source_file, expected_test_file) tuples for missing tests.
    """
    missing: list[tuple[str, str]] = []

    # Check modules have test directories
    for module_name, _ in get_source_modules():
        if module_name.startswith("_"):
            continue

        if module_name in REFERENCE_ONLY_MODULES:
            ref_dir = TESTS_DIR / "reference" / module_name
            if not ref_dir.exists():
                missing.append((f"talu/{module_name}/", f"tests/reference/{module_name}/"))
            continue

        test_dir = TESTS_DIR / module_name
        if not test_dir.exists():
            missing.append((f"talu/{module_name}/", f"tests/{module_name}/"))

    # Check source files have test files
    for subdir in TALU_SRC.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("_"):
            continue
        if not (subdir / "__init__.py").exists():
            continue

        module_name = subdir.name
        if module_name in REFERENCE_ONLY_MODULES:
            continue

        test_dir = TESTS_DIR / module_name
        if not test_dir.exists():
            continue

        for file_stem, _ in get_source_files_for_module(subdir):
            expected = get_expected_test_file(file_stem)
            test_file = test_dir / expected
            test_files_in_subdirs = list(test_dir.rglob(expected))

            if not test_file.exists() and not test_files_in_subdirs:
                missing.append(
                    (
                        f"talu/{module_name}/{file_stem}.py",
                        f"tests/{module_name}/{expected}",
                    )
                )

    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check test coverage compliance.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    print(colored("Test Coverage Check", BOLD))
    print("=" * 40)

    missing = check_coverage(args.verbose)

    if missing:
        print(f"\n{colored('MISSING', RED)}: {len(missing)} source file(s) without tests:\n")
        for source, expected_test in missing:
            print(f"  {source}")
            print(f"    -> {expected_test}")
        print()
        return 1
    else:
        print(f"\n{colored('OK', GREEN)}: All source files have corresponding tests.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
