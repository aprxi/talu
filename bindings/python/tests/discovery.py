"""
Zero-maintenance public API discovery.

================================================================================
IMPORTANT: THIS MODULE MUST BE 100% AUTO-GENERATED
================================================================================

DO NOT add manual lists, registries, or configuration to this module.
All discovery MUST be automatic based on:

1. Filesystem structure (talu/*.py, talu/*/)
2. Python conventions (__all__ exports in __init__.py)
3. Runtime introspection (classes, methods, properties)

A subpackage is public if and only if:
- It's a directory with __init__.py (standard Python package)
- It has __all__ defined and non-empty (Python export convention)
- It doesn't start with underscore (Python private convention)

If you need to exclude something, use Python conventions:
- Prefix with underscore for private modules
- Don't define __all__ for internal-only packages
- Use __all__ = [] for packages with no public API

DO NOT create manual lists like PUBLIC_SUBPACKAGES or similar.
================================================================================

Shared by:
- docs/scripts/docgen.py - documentation generation
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# Project Paths
# =============================================================================

# bindings/python/tests/discovery.py -> tests -> python -> bindings -> repo_root
PYTHON_ROOT = Path(__file__).resolve().parent.parent
TALU_SRC = PYTHON_ROOT / "talu"
TESTS_DIR = PYTHON_ROOT / "tests"

# =============================================================================
# Constants (derived from Python/filesystem constraints, not configurable)
# =============================================================================

# Source files that don't need direct test coverage (structural Python files)
EXCLUDED_SOURCE_FILES = frozenset(
    {
        "__init__.py",  # Package markers
        "__main__.py",  # CLI entry point
        "_lib.py",  # C library loader
    }
)

# Module-internal files that don't need direct tests (tested via public API)
EXCLUDED_MODULE_FILES = frozenset(
    {
        "_bindings.py",  # C bindings
        "_errors.py",  # Internal error helpers
        "base.py",  # Abstract base classes
    }
)

# No directory overrides allowed - strict 1:1 mapping enforced.
# If a module name conflicts with stdlib, rename the SOURCE file.
# Example: talu/_types.py should be talu/_config.py to map to tests/config/

# Modules tested only in tests/reference/ (require reference implementations)
REFERENCE_ONLY_MODULES = frozenset(
    {
        "ops",  # Tensor operations require PyTorch comparison
    }
)

# =============================================================================
# Module Discovery
# =============================================================================


def get_source_modules() -> list[tuple[str, Path]]:
    """Discover all source modules from filesystem.

    Returns
    -------
    list[tuple[str, Path]]
        List of (module_name, path) tuples.
    """
    modules = []

    # Top-level .py files
    for py_file in TALU_SRC.glob("*.py"):
        if py_file.name in EXCLUDED_SOURCE_FILES:
            continue
        modules.append((py_file.stem, py_file))

    # Subdirectories (packages)
    for subdir in TALU_SRC.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("__"):
            continue
        if (subdir / "__init__.py").exists():
            modules.append((subdir.name, subdir))

    return modules


def get_source_files_for_module(module_path: Path) -> list[tuple[str, Path]]:
    """Discover all source files in a module from filesystem.

    Parameters
    ----------
    module_path
        Path to the module directory.

    Returns
    -------
    list[tuple[str, Path]]
        List of (file_stem, path) tuples.
    """
    files = []

    for py_file in module_path.glob("*.py"):
        if py_file.name in EXCLUDED_SOURCE_FILES:
            continue
        if py_file.name in EXCLUDED_MODULE_FILES:
            continue
        if py_file.name.startswith("_"):
            continue

        files.append((py_file.stem, py_file))

    return files


def get_public_subpackages() -> list[str]:
    """Discover public subpackages automatically.

    A subpackage is public if:
    1. It's a directory under talu/ with __init__.py
    2. It doesn't start with underscore (not private)
    3. It has __all__ defined in __init__.py

    Returns
    -------
    list[str]
        List of full module names (e.g., "talu.tokenizer").
    """
    public = []

    for name, path in get_source_modules():
        # Must be a directory (package), not a single file
        if not path.is_dir():
            continue

        # Skip private packages
        if name.startswith("_"):
            continue

        # Check if it has __all__ (the marker of a public package)
        full_module = f"talu.{name}"
        try:
            module = importlib.import_module(full_module)
            if hasattr(module, "__all__") and module.__all__:
                public.append(full_module)
        except ImportError:
            continue

    return public


# =============================================================================
# Test Path Derivation
# =============================================================================


def get_expected_test_dir(module_name: str) -> str:
    """Derive test directory from source module name.

    Strict 1:1 mapping: talu/_foo.py -> tests/foo/
    No overrides allowed - if stdlib conflicts, rename the source file.

    Parameters
    ----------
    module_name
        Source module name (e.g., "_log", "tokenizer").

    Returns
    -------
    str
        Test directory name (e.g., "log", "tokenizer").
    """
    if module_name.startswith("_"):
        return module_name[1:]
    return module_name


def get_expected_test_file(source_file_stem: str) -> str:
    """Derive test file name from source file name.

    Parameters
    ----------
    source_file_stem
        Source file stem (e.g., "chat", "tokenizer").

    Returns
    -------
    str
        Test file name (e.g., "test_chat.py").
    """
    return f"test_{source_file_stem}.py"


def is_reference_only(module_name: str) -> bool:
    """Check if module requires reference tests only.

    Parameters
    ----------
    module_name
        Module name without 'talu.' prefix.

    Returns
    -------
    bool
        True if module should be tested in tests/reference/.
    """
    return module_name in REFERENCE_ONLY_MODULES


# =============================================================================
# Class/Symbol Discovery (via introspection)
# =============================================================================


def get_module_exports(module_name: str) -> list[str]:
    """Get __all__ exports from a module.

    Parameters
    ----------
    module_name
        Full module name (e.g., "talu.tokenizer").

    Returns
    -------
    list[str]
        List of exported symbol names.
    """
    try:
        module = importlib.import_module(module_name)
        return list(getattr(module, "__all__", []))
    except ImportError:
        return []


def get_exported_classes(module_name: str) -> list[type]:
    """Get all exported classes from a module.

    Parameters
    ----------
    module_name
        Full module name (e.g., "talu.tokenizer").

    Returns
    -------
    list[type]
        List of class objects.
    """
    try:
        module = importlib.import_module(module_name)
        exports = getattr(module, "__all__", [])
        return [
            getattr(module, name)
            for name in exports
            if isinstance(getattr(module, name, None), type)
        ]
    except ImportError:
        return []


def get_public_methods(cls: type) -> list[str]:
    """Get public method names from a class.

    Parameters
    ----------
    cls
        The class to inspect.

    Returns
    -------
    list[str]
        List of public method names.
    """
    methods = []
    for name in dir(cls):
        if name.startswith("_") and name != "__call__":
            continue
        obj = getattr(cls, name, None)
        if callable(obj) and not isinstance(obj, type):
            methods.append(name)
    return methods


def get_public_properties(cls: type) -> list[str]:
    """Get public property names from a class.

    Parameters
    ----------
    cls
        The class to inspect.

    Returns
    -------
    list[str]
        List of public property names.
    """
    properties = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        try:
            obj = inspect.getattr_static(cls, name)
            if isinstance(obj, property):
                properties.append(name)
        except AttributeError:
            continue
    return properties


def iter_all_public_classes() -> Iterator[tuple[str, type]]:
    """Iterate over all public classes in talu.

    Yields
    ------
    tuple[str, type]
        (module_name, class_object) for each public class.
    """
    # Root package
    for cls in get_exported_classes("talu"):
        yield "talu", cls

    # Subpackages
    for subpkg in get_public_subpackages():
        module_name = f"talu.{subpkg}"
        for cls in get_exported_classes(module_name):
            yield module_name, cls


def is_exception_class(cls: type) -> bool:
    """Check if a class is an exception.

    Parameters
    ----------
    cls
        Class to check.

    Returns
    -------
    bool
        True if cls is a subclass of Exception.
    """
    try:
        return issubclass(cls, Exception)
    except TypeError:
        return False


def is_protocol_class(cls: type) -> bool:
    """Check if a class is a Protocol.

    Protocol classes are interface specifications, not concrete implementations.
    They should be excluded from certain validations (e.g., Raises sections)
    since their methods are not meant to be called directly.

    Parameters
    ----------
    cls
        Class to check.

    Returns
    -------
    bool
        True if cls is a Protocol subclass.
    """
    from typing import Protocol

    # Check if it's a Protocol by looking at the MRO
    try:
        return Protocol in cls.__mro__ and cls is not Protocol
    except (TypeError, AttributeError):
        return False


# =============================================================================
# Documentation Helpers
# =============================================================================


def get_doc_page_name(module_name: str) -> str:
    """Derive documentation page name from module name.

    Strict 1:1 mapping - same as test directories.

    Parameters
    ----------
    module_name
        Module name (e.g., "chat_session", "_config").

    Returns
    -------
    str
        Doc page name (e.g., "chat-session", "config").
    """
    if module_name.startswith("_"):
        base = module_name[1:]
    else:
        base = module_name

    # Convert underscores to hyphens for URL-friendly names
    return base.replace("_", "-")


def should_have_doc_page(cls: type) -> bool:
    """Check if a class should have a documentation page.

    All public classes get documented. This function exists for
    future extensibility but currently returns True for all non-exceptions.

    Parameters
    ----------
    cls
        Class to check.

    Returns
    -------
    bool
        True if class should have a dedicated doc page.
    """
    # Exceptions are documented in the errors page, not individual pages
    if is_exception_class(cls):
        return False

    # All other public classes get documented
    return True
