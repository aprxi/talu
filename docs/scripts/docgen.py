"""
Generate API reference documentation from Python source.

This script:
1. Discovers all public modules and classes from talu/
2. Validates docstrings and type annotations (fails on violations)
3. Generates one reference page per module

All validation rules are defined in docs/POLICY.md.
Discovery is shared with tests via scripts/discovery.py.
"""

import importlib
import inspect
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygments import highlight as _pygments_highlight
from pygments.formatters import HtmlFormatter as _PygFormatter
from pygments.lexers import PythonConsoleLexer as _PyConLexer, PythonLexer as _PyLexer
from pygments import token as _T

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bindings" / "python" / "tests"))

# Import shared discovery (from bindings/python/tests/)
from discovery import (
    get_doc_page_name,
    get_module_exports,
    get_public_subpackages,
    get_source_modules,
)


# =============================================================================
# Naming Helpers
# =============================================================================


def snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase.

    Examples:
        chat_session -> Chat
        local_engine -> LocalEngine
        _api -> Api
        _errors -> Errors
    """
    # Strip leading underscore for private modules
    if name.startswith("_"):
        name = name[1:]

    # Split on underscores and capitalize each part
    return "".join(word.capitalize() for word in name.split("_"))


# =============================================================================
# Configuration (auto-generated from discovery)
# =============================================================================


def _get_module_description(module_name: str) -> str:
    """Extract the first sentence of a module's docstring."""
    try:
        mod = importlib.import_module(module_name)
        doc = inspect.getdoc(mod)
        if doc:
            # First line, strip trailing period for consistency
            first_line = doc.strip().split("\n")[0].rstrip(".")
            return first_line
    except ImportError:
        pass
    return module_name


def _build_module_pages() -> dict:
    """Build MODULE_PAGES dynamically from source discovery.

    Returns dict mapping module names to page configuration.
    """
    pages = {}

    # Root module - top-level functions only
    pages["talu"] = {
        "page": "api",
        "title": "Top-Level API",
        "description": _get_module_description("talu"),
        "include_classes": False,
    }

    # Discover top-level .py files (single-file modules)
    for module_name, path in get_source_modules():
        if path.is_dir():
            continue  # Skip packages, handled below
        if module_name.startswith("_"):
            continue  # Skip private modules

        full_module = f"talu.{module_name}"
        page_name = get_doc_page_name(module_name)
        title = snake_to_camel(module_name)

        # Get classes from the module's exports
        exports = get_module_exports(full_module)

        # Auto-detect if module exports exceptions by checking the first export
        include_exceptions = False
        if exports:
            try:
                mod = importlib.import_module(full_module)
                first_export = getattr(mod, exports[0], None)
                if first_export is not None and isinstance(first_export, type):
                    include_exceptions = issubclass(first_export, Exception)
            except (ImportError, TypeError):
                pass

        pages[full_module] = {
            "page": page_name,
            "title": title,
            "description": _get_module_description(full_module),
            "classes": exports if exports else None,
            "include_exceptions": include_exceptions,
        }

    # Modules excluded from docs entirely (no page, no search, no auto-links)
    _EXCLUDED = {"xray"}

    # Discover subpackages (get_public_subpackages returns full names like "talu.tokenizer")
    for full_module in get_public_subpackages():
        # Extract short name (e.g., "tokenizer" from "talu.tokenizer")
        subpkg = full_module.split(".")[-1]
        if subpkg in _EXCLUDED:
            continue
        page_name = get_doc_page_name(subpkg)
        title = snake_to_camel(subpkg)

        # Auto-detect if package exports exceptions
        include_exceptions = False
        exports = get_module_exports(full_module)
        if exports:
            try:
                mod = importlib.import_module(full_module)
                first_export = getattr(mod, exports[0], None)
                if first_export is not None and isinstance(first_export, type):
                    include_exceptions = issubclass(first_export, Exception)
            except (ImportError, TypeError):
                pass

        config = {
            "page": page_name,
            "title": title,
            "description": _get_module_description(full_module),
            # No "classes" key = use __all__ from package
        }
        if include_exceptions:
            config["include_exceptions"] = True
        pages[full_module] = config

    return pages


# Build MODULE_PAGES dynamically
MODULE_PAGES = _build_module_pages()

# Display order for reference modules (by importance).
# Modules not listed here are excluded from nav/index (e.g. xray).
# Three tiers: primary (class-based), secondary (function-based utilities), supporting.
MODULE_ORDER = ["chat", "tokenizer", "template", "validate", "db", "converter", "repository", "client", "profile", "types", "router", "exceptions"]

# Tier boundaries for index page and sidebar separators.
_SECONDARY_MODULES = {"converter", "repository"}
_SUPPORTING_MODULES = {"client", "profile", "types", "router", "exceptions"}

# Symbols excluded from docs per module.  These are internal building blocks
# whose structure is shown inline in their parent type's docstring (e.g.
# ItemRecord's docstring shows the variant/content structures with examples).
_EXCLUDE_SYMBOLS: dict[str, set[str]] = {
    "talu.types": {
        # Record Literal type aliases (string enums for storage)
        "RecordItemType", "RecordItemStatus", "RecordMessageRole", "RecordContentType",
        # Record content part TypedDicts (nested inside item variants)
        "InputTextContent", "InputImageContent", "InputAudioContent",
        "InputVideoContent", "InputFileContent", "OutputTextContent",
        "RefusalContent", "TextContent", "ReasoningTextContent",
        "SummaryTextContent", "RecordUnknownContent", "RecordContentPart",
        # Record item variant TypedDicts (nested inside ItemRecord)
        "MessageItemVariant", "FunctionCallVariant", "FunctionCallOutputVariant",
        "ReasoningVariant", "ItemReferenceVariant",
    },
}



# =============================================================================
# Error Collection
# =============================================================================


@dataclass
class DocError:
    """A documentation validation error."""

    location: str  # e.g., "Tokenizer.encode()"
    message: str  # e.g., "Parameter 'text' missing type annotation"

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass
class DocValidator:
    """Collects validation errors during doc generation."""

    errors: list[DocError] = field(default_factory=list)

    def error(self, location: str, message: str) -> None:
        """Record an error."""
        self.errors.append(DocError(location, message))

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def report(self) -> str:
        """Generate error report."""
        lines = [f"ERROR: {e}" for e in self.errors]
        lines.append(f"\nDocumentation generation failed: {len(self.errors)} error(s)")
        return "\n".join(lines)


# Global validator instance
validator = DocValidator()


# =============================================================================
# Module Discovery
# =============================================================================


@dataclass
class ClassInfo:
    """Information about a documented class."""

    cls: type
    name: str
    module: str
    docstring: str
    is_exception: bool = False


@dataclass
class FunctionInfo:
    """Information about a documented function."""

    func: Any
    name: str
    module: str
    docstring: str


@dataclass
class TypeAliasInfo:
    """Information about a documented type alias (union, Literal, etc.)."""

    name: str
    module: str
    members: list[str]  # Member type names


@dataclass
class ModuleInfo:
    """Information about a documented module."""

    name: str
    page: str
    title: str
    description: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    type_aliases: list[TypeAliasInfo] = field(default_factory=list)


def discover_modules() -> list[ModuleInfo]:
    """Discover all modules and their public symbols.

    Handles two types of modules:
    1. Single-file modules (talu/engine.py) - uses explicit "classes" list from config
    2. Package modules (talu/tokenizer/) - uses __all__ from package __init__.py

    Validates:
    - Package modules have __all__
    - Package __all__ is non-empty
    - All specified classes exist and have docstrings
    """
    modules = []

    for module_name, config in MODULE_PAGES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            validator.error(module_name, f"Cannot import module: {e}")
            continue

        info = ModuleInfo(
            name=module_name,
            page=config["page"],
            title=config["title"],
            description=config["description"],
        )

        # Determine symbols to document
        explicit_classes = config.get("classes")

        if explicit_classes is not None:
            # Single-file module with explicit class list
            symbols = explicit_classes
        elif module_name == "talu":
            # Root module - get from __all__ for functions
            symbols = getattr(module, "__all__", [])
        else:
            # Package module - require __all__
            if not hasattr(module, "__all__"):
                validator.error(module_name, "Missing __all__")
                continue
            symbols = module.__all__
            if not symbols:
                validator.error(module_name, "Empty __all__")
                continue

        include_classes = config.get("include_classes", True)
        include_exceptions = config.get("include_exceptions", False)
        exclude = _EXCLUDE_SYMBOLS.get(module_name, set())

        for symbol_name in symbols:
            if symbol_name in exclude:
                continue

            obj = getattr(module, symbol_name, None)
            if obj is None:
                validator.error(f"{module_name}.{symbol_name}", "Symbol not found in module")
                continue

            # Skip non-types for class discovery
            if isinstance(obj, type):
                is_exception = issubclass(obj, Exception)

                # Skip exceptions unless explicitly included
                if is_exception and not include_exceptions:
                    continue

                # Skip classes if not included
                if not is_exception and not include_classes:
                    continue

                docstring = inspect.getdoc(obj)
                if not docstring:
                    validator.error(symbol_name, "Missing class docstring")
                    docstring = ""

                info.classes.append(
                    ClassInfo(
                        cls=obj,
                        name=symbol_name,
                        module=module_name,
                        docstring=docstring,
                        is_exception=is_exception,
                    )
                )

            elif hasattr(obj, "__args__"):
                # Union type or generic alias (e.g. ContentPart = A | B | C)
                members = [
                    f'"{a}"' if isinstance(a, str) else getattr(a, "__name__", str(a))
                    for a in obj.__args__
                ]
                info.type_aliases.append(
                    TypeAliasInfo(
                        name=symbol_name,
                        module=module_name,
                        members=members,
                    )
                )

            elif callable(obj) and not isinstance(obj, type):
                # It's a function
                docstring = inspect.getdoc(obj)
                if not docstring:
                    validator.error(f"{module_name}.{symbol_name}()", "Missing docstring")
                    docstring = ""

                info.functions.append(
                    FunctionInfo(
                        func=obj,
                        name=symbol_name,
                        module=module_name,
                        docstring=docstring,
                    )
                )

        modules.append(info)

    return modules


# =============================================================================
# Validation
# =============================================================================


def validate_class(cls: type, class_name: str) -> None:
    """Validate a class has proper docstrings and type annotations.

    Only validates methods/properties defined directly on the class,
    not inherited ones. This avoids false positives for methods like
    Exception.add_note() or Protocol.__init__().
    """
    # Validate class-level docstring section order
    class_doc = inspect.getdoc(cls)
    if class_doc:
        _validate_section_order(class_doc, class_name)

    # Skip Protocol classes - they don't need __init__ docs
    is_protocol = getattr(cls, "_is_protocol", False)

    # Check constructor if it exists and is custom (defined on this class)
    if "__init__" in cls.__dict__ and not is_protocol:
        init = cls.__init__
        validate_method(init, "__init__", class_name, skip_return=True)

    # Check all public methods and properties DEFINED ON THIS CLASS
    for name, obj in cls.__dict__.items():
        if name.startswith("_") and name != "__call__":
            continue

        if isinstance(obj, property):
            validate_property(obj, name, class_name)
        elif callable(obj) and not isinstance(obj, type):
            # Skip staticmethod/classmethod wrappers - get the underlying function
            if isinstance(obj, staticmethod):
                validate_method(obj.__func__, name, class_name)
            elif isinstance(obj, classmethod):
                validate_method(obj.__func__, name, class_name)
            else:
                validate_method(obj, name, class_name)


def _validate_args_docs(
    doc: str, params: list[inspect.Parameter], location: str,
) -> None:
    """Validate that Args section entries match signature parameters."""
    # Skip __init__ — constructor params are documented at the class level
    # via Attributes sections, not in __init__'s own Args.
    if ".__init__()" in location:
        return

    named = [
        p for p in params
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not named:
        return

    parsed = parse_docstring(doc)
    entries = parsed["args"]

    if not entries:
        validator.error(location, "Has parameters but missing Args/Parameters section")
        return

    normalized = {k.lstrip("*"): k for k in entries}
    for p in named:
        if p.name not in normalized:
            validator.error(location, f"Parameter '{p.name}' not documented in Args")

    all_names = {p.name for p in params}
    for entry_name in entries:
        if entry_name.lstrip("*") not in all_names:
            validator.error(location, f"Args documents '{entry_name}' which is not a parameter")


# Canonical order for structural sections — note/custom are informational and float freely.
_STRUCTURAL_ORDER = ["args", "returns", "raises", "example"]


def _validate_section_order(doc: str, location: str) -> None:
    """Validate that structural docstring sections appear in canonical order."""
    parsed = parse_docstring(doc)
    sections = parsed["sections"]
    if len(sections) < 2:
        return

    max_rank = -1
    max_name = ""
    for section in sections:
        kind = section["kind"]
        if kind not in _STRUCTURAL_ORDER:
            continue
        rank = _STRUCTURAL_ORDER.index(kind)
        if rank < max_rank:
            validator.error(
                location,
                f"Section '{section['name']}' ({kind}) appears after {max_name}"
                f" — expected order: {' > '.join(_STRUCTURAL_ORDER)}",
            )
            return
        max_rank = rank
        max_name = section["name"]


def validate_method(
    method: Any, method_name: str, class_name: str, skip_return: bool = False
) -> None:
    """Validate a method has docstring and type annotations."""
    location = f"{class_name}.{method_name}()"

    # Check docstring
    doc = inspect.getdoc(method)
    if not doc:
        validator.error(location, "Missing docstring")

    # Check type annotations
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())

        # Skip self
        if params and params[0].name == "self":
            params = params[1:]

        for p in params:
            if p.annotation == inspect.Parameter.empty:
                validator.error(location, f"Parameter '{p.name}' missing type annotation")

        # Check return type (skip for __init__)
        if not skip_return and sig.return_annotation == inspect.Signature.empty:
            validator.error(location, "Missing return type annotation")

        # Validate Args docs match signature
        if doc:
            _validate_args_docs(doc, params, location)
            _validate_section_order(doc, location)

    except (ValueError, TypeError):
        pass  # Some builtins don't have signatures


def validate_property(prop: Any, prop_name: str, class_name: str) -> None:
    """Validate a property has docstring and return type."""
    location = f"{class_name}.{prop_name}"

    if not hasattr(prop, "fget") or not prop.fget:
        validator.error(location, "Property has no getter")
        return

    # Check docstring
    doc = inspect.getdoc(prop.fget)
    if not doc:
        validator.error(location, "Missing docstring")

    # Check return type
    hints = getattr(prop.fget, "__annotations__", {})
    if "return" not in hints:
        validator.error(location, "Missing return type annotation")


def validate_function(func: Any, func_name: str, module_name: str = "talu") -> None:
    """Validate a function has docstring and type annotations."""
    location = f"{module_name}.{func_name}()"

    # Check docstring
    doc = inspect.getdoc(func)
    if not doc:
        validator.error(location, "Missing docstring")

    # Check type annotations
    try:
        sig = inspect.signature(func)

        for p in sig.parameters.values():
            if p.annotation == inspect.Parameter.empty:
                validator.error(location, f"Parameter '{p.name}' missing type annotation")

        if sig.return_annotation == inspect.Signature.empty:
            validator.error(location, "Missing return type annotation")

        # Validate Args docs match signature
        if doc:
            params = list(sig.parameters.values())
            _validate_args_docs(doc, params, location)
            _validate_section_order(doc, location)

    except (ValueError, TypeError):
        pass


# =============================================================================
# Markdown Generation
# =============================================================================


def format_type(type_str: str) -> str:
    """Clean up type annotation for display.

    Removes internal module prefixes to show cleaner type names.
    Uses regex to handle any talu.* prefix automatically.
    """
    # Remove any talu.* module prefixes (auto-handles all submodules)
    type_str = re.sub(r"talu\.[a-z_]+\.[a-z_]+\.", "", type_str)
    type_str = re.sub(r"talu\.[a-z_]+\.", "", type_str)

    # Remove typing module prefix
    type_str = type_str.replace("typing.", "")

    # Remove class wrapper syntax
    type_str = type_str.replace("<class '", "").replace("'>", "")

    return type_str


def get_signature(
    method: Any, method_name: str, *, is_classmethod: bool = False,
) -> tuple[list[tuple[str, str, str | None]], str]:
    """Get method signature as (params, return_type).

    Returns:
        params: List of (name, type, default_or_none)
        return_type: Return type string
    """
    params = []
    ret_type = ""

    try:
        sig = inspect.signature(method)
        sig_params = list(sig.parameters.values())

        if sig_params and sig_params[0].name in ("self", "cls"):
            # Keep self/cls — matches actual source code
            sig_params[0] = sig_params[0].replace(annotation=inspect.Parameter.empty)
        elif is_classmethod:
            # inspect.signature strips cls from classmethods — add it back
            cls_param = inspect.Parameter("cls", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            sig_params.insert(0, cls_param)

        for p in sig_params:
            type_str = ""
            if p.annotation != inspect.Parameter.empty:
                type_str = format_type(str(p.annotation))

            default = None
            if p.default != inspect.Parameter.empty:
                if p.default is None:
                    default = "None"
                elif isinstance(p.default, str):
                    default = repr(p.default)
                else:
                    default = str(p.default)

            params.append((p.name, type_str, default))

        if sig.return_annotation != inspect.Signature.empty:
            ret_type = format_type(str(sig.return_annotation))

    except (ValueError, TypeError):
        pass

    return params, ret_type


def _dedent_lines(lines: list[str]) -> str:
    """Remove common leading whitespace from lines, then strip outer blank lines."""
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return ""
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    return "\n".join(l[min_indent:] for l in lines).strip()


def parse_docstring(doc: str) -> dict:
    """Parse NumPy/Google-style docstring into structured sections.

    Returns dict with:
        description: Text before the first section header.
        args: Dict mapping parameter names to {"type": str, "desc": str}.
        sections: Ordered list of section dicts with keys:
            name: Section header text (e.g. "Returns", "Example - Basic usage").
            kind: One of "args", "returns", "raises", "example", "note", "custom".
            content: Raw section text.
            entries: Dict of name→{"type", "desc"} (only for args/raises/attributes kinds).
    """
    if not doc:
        return {"description": "", "args": {}, "sections": []}

    # Section header classification
    _KIND_MAP = {
        "Args": "args", "Arguments": "args", "Parameters": "args",
        "Attributes": "args",
        "Returns": "returns", "Return": "returns", "Yields": "returns",
        "Raises": "raises", "Raise": "raises",
        "Example": "example", "Examples": "example",
        "Note": "note", "Notes": "note",
        "See Also": "custom",
    }

    lines = doc.strip().split("\n")
    raw_sections: list[tuple[str, str, list[str]]] = []  # (name, kind, lines)
    current_name = ""
    current_kind = "description"
    current_lines: list[str] = []
    pending_numpy: str | None = None

    def _flush():
        raw_sections.append((current_name, current_kind, current_lines))

    def _classify(header: str) -> tuple[str, str]:
        """Return (display_name, kind) for a section header."""
        # Titled examples: "Example - Title" or "Examples - Title"
        if re.match(r"Examples?\s*[-–—]", header):
            return (header, "example")
        # Known header
        if header in _KIND_MAP:
            return (header, _KIND_MAP[header])
        return (header, "custom")

    for line in lines:
        stripped = line.strip()

        # NumPy-style underline confirms pending header
        if pending_numpy is not None and re.fullmatch(r"-{3,}", stripped):
            _flush()
            current_name, current_kind = _classify(pending_numpy)
            current_lines = []
            pending_numpy = None
            continue
        elif pending_numpy is not None:
            current_lines.append(pending_numpy)
            pending_numpy = None

        # Google-style header: "Args:" or "Example - Title:"
        if stripped.endswith(":") and not stripped.startswith(">>>"):
            bare = stripped[:-1]
            name, kind = _classify(bare)
            if kind != "custom" or (
                # Accept custom Google-style headers only at reasonable indent
                len(line) - len(line.lstrip()) <= 8
                and re.fullmatch(r"[A-Z][A-Za-z0-9 \-–—()]+", bare)
                and len(bare.split()) <= 6
            ):
                _flush()
                current_name = bare
                current_kind = kind
                current_lines = []
                continue

        # Bare word that could be a NumPy header — wait for underline
        if (
            current_kind != "example"
            and re.fullmatch(r"[A-Z][A-Za-z0-9 \-–—()]+", stripped)
            and len(stripped.split()) <= 6
        ):
            pending_numpy = stripped
            continue

        current_lines.append(line)

    if pending_numpy is not None:
        current_lines.append(pending_numpy)
    _flush()

    # Build result
    description = ""
    args: dict[str, str] = {}
    sections: list[dict] = []

    for name, kind, sec_lines in raw_sections:
        content = _dedent_lines(sec_lines)

        if kind == "description":
            description = content
            continue

        section: dict = {"name": name, "kind": kind, "content": content}

        # Parse key:value entries for args/attributes/raises
        if kind in ("args", "raises"):
            entries = _parse_entries(sec_lines)
            section["entries"] = entries
            # Top-level args for backward compat
            if kind == "args" and name in ("Args", "Arguments", "Parameters"):
                args = entries

        sections.append(section)

    return {"description": description, "args": args, "sections": sections}


def _parse_entries(lines: list[str]) -> dict[str, dict]:
    """Parse key:value entries from an Args/Raises/Attributes section.

    Returns dict mapping name → {"type": str, "desc": str}.

    Handles three formats:
    - NumPy typed: ``param : type`` on key line, description on indented continuation
    - Google-style: ``param: description`` (no space before colon)
    - NumPy bare: ``param`` alone, description indented on next lines
    """
    entries: dict[str, dict] = {}
    current_key: str | None = None
    current_type: str = ""
    current_desc_lines: list[str] = []
    entry_indent: int | None = None
    # None = bare-name, True = NumPy typed (" : "), False = Google (":")
    typed_format: bool | None = None

    _BARE_NAME = re.compile(r"\*{0,2}\w+(?:\s*\([^)]*\))?$")

    def _flush() -> None:
        nonlocal current_key, current_type, current_desc_lines
        if current_key:
            desc = _dedent_lines(current_desc_lines) if current_desc_lines else ""
            entries[current_key] = {"type": current_type, "desc": desc}
        current_key = None
        current_type = ""
        current_desc_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Preserve blank lines as paragraph breaks in description
            if current_key and current_desc_lines:
                current_desc_lines.append("")
            continue
        indent = len(line) - len(line.lstrip())

        is_new_entry = False

        if entry_indent is None:
            # Detect format from first entry
            if " : " in stripped and not stripped.startswith((">>>", "-", "*")):
                entry_indent, typed_format, is_new_entry = indent, True, True
            elif ":" in stripped and not stripped.startswith((">>>", "-", "*")):
                entry_indent, typed_format, is_new_entry = indent, False, True
            elif _BARE_NAME.match(stripped):
                entry_indent, typed_format, is_new_entry = indent, None, True
        elif indent <= entry_indent:
            if typed_format is True:
                if (" : " in stripped or _BARE_NAME.match(stripped)) and not stripped.startswith((">>>", "-", "*")):
                    is_new_entry = True
            elif typed_format is False:
                if ":" in stripped and not stripped.startswith((">>>", "-", "*")):
                    is_new_entry = True
            else:
                if _BARE_NAME.match(stripped):
                    is_new_entry = True

        if is_new_entry:
            _flush()
            if typed_format is True and " : " in stripped:
                key_part, _, type_part = stripped.partition(" : ")
                current_key = key_part.strip().lstrip("*")
                current_type = type_part.strip()
            elif typed_format is False and ":" in stripped:
                key_part, _, desc_part = stripped.partition(":")
                current_key = key_part.strip()
                current_type = ""
                if desc_part.strip():
                    current_desc_lines = [desc_part]
            else:
                m = re.match(r"(\*{0,2}\w+)", stripped)
                current_key = m.group(1) if m else stripped
                current_type = ""
        elif current_key:
            current_desc_lines.append(line)

    _flush()
    return entries


def _default_token(value: str) -> str:
    """Return the hl-* class for a default value literal."""
    if value in ("None", "True", "False"):
        return "hl-keyword"
    if value.lstrip("-").isdigit() or value.replace(".", "", 1).lstrip("-").isdigit():
        return "hl-number"
    return "hl-string"


def _highlight_param(pname: str, ptype: str, pdefault: str) -> str:
    """Build a syntax-highlighted parameter fragment."""
    name_cls = "hl-builtin-name" if pname in ("self", "cls") else "hl-property-name"
    parts = [f'<span class="{name_cls}">{pname}</span>']
    if ptype:
        parts.append(f'<span class="hl-punctuation">: </span><span class="hl-type-name">{ptype}</span>')
    if pdefault:
        tok = _default_token(pdefault)
        parts.append(f'<span class="hl-operator"> = </span><span class="{tok}">{pdefault}</span>')
    return "".join(parts)


def _highlight_return(ret_type: str) -> str:
    """Build a syntax-highlighted return type fragment."""
    return f'<span class="hl-punctuation"> → </span><span class="hl-type-name">{ret_type}</span>'


def _html_escape(text: str) -> str:
    """Escape HTML special characters (< and & only; > is safe in content)."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


def _inline_markup(text: str) -> str:
    """Convert RST/docstring inline markup to HTML.

    Handles ``code`` (double backticks), **bold**, and *emphasis*.
    """
    text = re.sub(r"``([^`]+)``", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


# =============================================================================
# Structured Docstring Rendering
# =============================================================================

# Pygments hl-* token map (same mapping as build.py's UnifiedHtmlFormatter)
_DOC_HL_MAP = {
    _T.Comment:             "hl-comment",
    _T.Keyword.Type:        "hl-keyword-type",
    _T.Keyword.Declaration: "hl-keyword-declaration",
    _T.Keyword.Namespace:   "hl-keyword-namespace",
    _T.Keyword:             "hl-keyword",
    _T.Literal.String:      "hl-string",
    _T.Literal.Number:      "hl-number",
    _T.Name.Function:       "hl-function-name",
    _T.Name.Class:          "hl-class-name",
    _T.Name.Builtin:        "hl-builtin-name",
    _T.Name.Decorator:      "hl-decorator-name",
    _T.Operator:            "hl-operator",
    _T.Punctuation:         "hl-punctuation",
    _T.Name:                "",
    _T.Text:                "",
    _T.Other:               "",
}


def _doc_hl_class(ttype):
    """Map a Pygments token to a hl-* class name."""
    while ttype:
        if ttype in _DOC_HL_MAP:
            return _DOC_HL_MAP[ttype]
        ttype = ttype.parent
    return ""


class _DocFormatter(_PygFormatter):
    """HtmlFormatter emitting unified hl-* classes for docstring examples."""

    def _get_css_class(self, ttype):
        return _doc_hl_class(ttype)


_PYCON_LEXER = _PyConLexer()
_PY_LEXER = _PyLexer()
_DOC_FMT = _DocFormatter()


def _highlight_code(code: str) -> str:
    """Syntax-highlight a Python/console code block."""
    if not code.strip():
        return ""
    lexer = _PYCON_LEXER if ">>>" in code else _PY_LEXER
    return _pygments_highlight(code, lexer, _DOC_FMT)


def _is_code_block(lines: list[str]) -> bool:
    """Check if a group of lines forms a code block."""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(">>>") or s.startswith("..."):
            return True
        indent = len(line) - len(line.lstrip())
        if indent >= 4 and not s.startswith(("-", "*", "+")):
            return True
        return False  # first meaningful line determines
    return False


def _render_content(text: str) -> str:
    """Render docstring text as HTML paragraphs, lists, and highlighted code blocks."""
    if not text:
        return ""
    raw_blocks = re.split(r"\n{2,}", text.strip())
    parts = []
    for block in raw_blocks:
        if not block.strip():
            continue
        lines = block.split("\n")
        if _is_code_block(lines):
            parts.append(_highlight_code(_dedent_lines(lines)))
        elif any(l.strip().startswith("- ") for l in lines):
            # Bullet list block
            items: list[str] = []
            current: list[str] = []
            for l in lines:
                s = l.strip()
                if not s:
                    continue
                if s.startswith("- "):
                    if current:
                        items.append(" ".join(current))
                    current = [s[2:]]
                else:
                    current.append(s)
            if current:
                items.append(" ".join(current))
            li_tags = "\n".join(
                f"<li>{_inline_markup(_html_escape(item))}</li>" for item in items
            )
            parts.append(f"<ul>\n{li_tags}\n</ul>")
        else:
            joined = " ".join(l.strip() for l in lines if l.strip())
            # Strip RST code-block indicator (trailing ::)
            if joined.endswith("::"):
                joined = joined[:-1]
            parts.append(f"<p>{_inline_markup(_html_escape(joined))}</p>")
    return "\n".join(parts)


def _render_description(text: str) -> str:
    """Render description text, handling RST directives."""
    if not text:
        return ""
    # Fast path: no directives
    if "\n.. " not in f"\n{text}":
        return _render_content(text)

    result_lines: list[str] = []
    output_parts: list[str] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        m = re.match(r"\s*\.\. ([\w-]+)::", lines[i])
        if m:
            dtype = m.group(1)
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            dir_lines: list[str] = []
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                dir_lines.append(lines[i])
                i += 1
            if dtype == "warning":
                pre_text = "\n".join(result_lines).strip()
                if pre_text:
                    output_parts.append(_render_content(pre_text))
                    result_lines = []
                inner = _render_content(_dedent_lines(dir_lines))
                output_parts.append(
                    f'<div class="note warning"><div class="note-label">Warning</div>{inner}</div>'
                )
            # else: strip (design-decision, etc.)
        else:
            result_lines.append(lines[i])
            i += 1

    remaining = "\n".join(result_lines).strip()
    if remaining:
        output_parts.append(_render_content(remaining))
    return "\n".join(output_parts)


def _render_args_section(section: dict) -> str:
    """Render Args/Attributes section as a definition list."""
    entries = section.get("entries", {})
    label = "Attributes" if section["name"] in ("Attributes",) else "Parameters"
    parts = [f'<div class="ds-section">', f'<h5 class="ds-label">{label}</h5>']
    if entries:
        parts.append("<dl>")
        for name, entry in entries.items():
            type_str = entry["type"] if isinstance(entry, dict) else ""
            desc = entry["desc"] if isinstance(entry, dict) else entry
            type_html = (
                f' <span class="hl-punctuation">:</span>'
                f' <span class="hl-type-name">{_html_escape(type_str)}</span>'
                if type_str else ""
            )
            parts.append(f"<dt><code>{_html_escape(name)}</code>{type_html}</dt>")
            parts.append(f"<dd>{_render_content(desc)}</dd>")
        parts.append("</dl>")
    else:
        parts.append(_render_content(section["content"]))
    parts.append("</div>")
    return "\n".join(parts)


def _render_returns_section(section: dict) -> str:
    """Render Returns/Yields section."""
    label = "Yields" if section["name"] in ("Yields",) else "Returns"
    return "\n".join([
        '<div class="ds-section">',
        f'<h5 class="ds-label">{label}</h5>',
        _render_content(section["content"]),
        '</div>',
    ])


def _render_raises_section(section: dict) -> str:
    """Render Raises section as a definition list."""
    entries = section.get("entries", {})
    parts = ['<div class="ds-section">', '<h5 class="ds-label">Raises</h5>']
    if entries:
        parts.append("<dl>")
        for name, entry in entries.items():
            desc = entry["desc"] if isinstance(entry, dict) else entry
            parts.append(f"<dt><code>{_html_escape(name)}</code></dt>")
            parts.append(f"<dd>{_render_content(desc)}</dd>")
        parts.append("</dl>")
    else:
        parts.append(_render_content(section["content"]))
    parts.append("</div>")
    return "\n".join(parts)


def _render_example_section(section: dict) -> str:
    """Render Example section with syntax-highlighted code."""
    label = _html_escape(section["name"])
    code = section["content"]
    parts = [
        '<div class="ds-section">',
        f'<h5 class="ds-label">{label}</h5>',
    ]
    # Detect whether content has prose mixed with code
    lines = code.split("\n")
    has_prose = any(
        l.strip()
        and not l.strip().startswith(">>>")
        and not l.strip().startswith("...")
        and not l.startswith("    ")
        and not l.strip().startswith("#")
        for l in lines
    )
    if has_prose:
        parts.append(_render_content(code))
    else:
        parts.append(_highlight_code(code))
    parts.append("</div>")
    return "\n".join(parts)


def _render_note_section(section: dict) -> str:
    """Render Note section as a callout."""
    inner = _render_content(section["content"])
    return (
        '<div class="ds-section">'
        f'<div class="note"><div class="note-label">Note</div>{inner}</div>'
        '</div>'
    )


def _render_custom_section(section: dict) -> str:
    """Render a custom section (Mutability, Concurrency, etc.)."""
    return "\n".join([
        '<div class="ds-section">',
        f'<h5 class="ds-label">{_html_escape(section["name"])}</h5>',
        _render_content(section["content"]),
        '</div>',
    ])


_SECTION_RENDERERS = {
    "args": _render_args_section,
    "returns": _render_returns_section,
    "raises": _render_raises_section,
    "example": _render_example_section,
    "note": _render_note_section,
    "custom": _render_custom_section,
}


def _render_docstring(parsed: dict) -> str:
    """Render a parsed docstring as structured HTML.

    Converts parse_docstring() output into semantic HTML with:
    - Description paragraphs with inline markup
    - Definition lists for Args/Raises
    - Syntax-highlighted example code
    - Note callouts
    - Custom section headers
    """
    parts: list[str] = []

    if parsed["description"]:
        parts.append(_render_description(parsed["description"]))

    for section in parsed["sections"]:
        renderer = _SECTION_RENDERERS.get(section["kind"], _render_custom_section)
        parts.append(renderer(section))

    return "\n".join(p for p in parts if p)


def generate_method_markdown(
    name: str, method: Any, class_name: str,
    is_property: bool = False, is_classmethod: bool = False,
) -> str:
    """Generate markdown for a method or property."""
    lines = []

    doc = inspect.getdoc(method)
    parsed = parse_docstring(doc or "")

    # Anchor — scope to class to avoid duplicate IDs across classes
    member_id = name.lower().replace("_", "-")
    class_prefix = class_name.lower().replace("_", "-")
    anchor_id = f"{class_prefix}-{member_id}"
    lines.append(f'<div class="api-method" id="{anchor_id}">')

    if is_property:
        # Property — show as typed attribute: "name: type"
        # method is already the fget function (passed from caller)
        hints = getattr(method, "__annotations__", {})
        ret_type = format_type(str(hints.get("return", "")))
        type_suffix = (
            f'<span class="hl-punctuation">: </span><span class="hl-type-name">{ret_type}</span>'
            if ret_type else ""
        )
        lines.append(
            f'<h4><code><span class="hl-property-name">{name}</span>'
            f'{type_suffix}</code></h4>'
        )
    else:
        # Method signature
        params, ret_type = get_signature(method, name, is_classmethod=is_classmethod)
        ret_suffix = _highlight_return(ret_type) if ret_type else ""
        kw = '<span class="hl-keyword-declaration">def </span>'

        if not params:
            lines.append(
                f'<h4><code>{kw}<span class="hl-function-name">{name}</span>'
                f'<span class="hl-punctuation">()</span>{ret_suffix}</code></h4>'
            )
        else:
            param_strs = [_highlight_param(pn, pt, pd) for pn, pt, pd in params]
            comma = '<span class="hl-punctuation">, </span>'

            # Short signatures (≤2 params) stay on one line
            if len(params) <= 2:
                lines.append(
                    f'<h4><code>{kw}<span class="hl-function-name">{name}</span>'
                    f'<span class="hl-punctuation">(</span>{comma.join(param_strs)}'
                    f'<span class="hl-punctuation">)</span>{ret_suffix}</code></h4>'
                )
            else:
                # Multi-line: name( on first line, indented params, ) → type
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;"
                sig_parts = [
                    f'{kw}<span class="hl-function-name">{name}</span>'
                    f'<span class="hl-punctuation">(</span>'
                ]
                for i, ps in enumerate(param_strs):
                    trail = '<span class="hl-punctuation">,</span>' if i < len(param_strs) - 1 else ""
                    sig_parts.append(f"<br>{indent}{ps}{trail}")
                sig_parts.append(f'<br><span class="hl-punctuation">)</span>{ret_suffix}')
                lines.append(f'<h4><code>{"".join(sig_parts)}</code></h4>')

    # Structured docstring rendering
    rendered = _render_docstring(parsed)
    if rendered:
        lines.append(rendered)

    lines.append("</div>\n")

    return "\n".join(lines)


def _get_own_members(cls: type) -> list[str]:
    """Get public member names defined on cls itself, not inherited from stdlib bases."""
    own = set()
    for klass in cls.__mro__:
        if klass is cls:
            own.update(klass.__dict__)
        elif klass.__name__ in ("object", "dict", "list", "tuple", "set", "int", "float", "str", "bytes", "Enum"):
            break
        else:
            own.update(klass.__dict__)
    return [
        n for n in dir(cls)
        if (not n.startswith("_") or n == "__call__") and n in own
    ]


def generate_class_markdown(info: ClassInfo) -> str:
    """Generate markdown for a class."""
    cls = info.cls
    name = info.name
    lines = []

    # Class header — raw HTML to control ID and add keyword prefix
    anchor = name.lower().replace("_", "-")
    lines.append(f'<h2 id="{anchor}"><span class="hl-class-keyword">class </span><span class="hl-class-name">{name}</span></h2>\n')

    # Constructor signature (if custom) — right after class header
    if "__init__" in cls.__dict__:
        init = cls.__init__
        params, _ = get_signature(init, "__init__")

        if params:
            param_strs = [_highlight_param(pn, pt, pd) for pn, pt, pd in params]
            comma = '<span class="hl-punctuation">, </span>'

            if len(params) <= 2:
                sig_html = (
                    f'<span class="hl-class-name">{name}</span>'
                    f'<span class="hl-punctuation">(</span>{comma.join(param_strs)}'
                    f'<span class="hl-punctuation">)</span>'
                )
            else:
                indent = "&nbsp;&nbsp;&nbsp;&nbsp;"
                parts = [
                    f'<span class="hl-class-name">{name}</span>'
                    f'<span class="hl-punctuation">(</span>'
                ]
                for i, ps in enumerate(param_strs):
                    trail = '<span class="hl-punctuation">,</span>' if i < len(param_strs) - 1 else ""
                    parts.append(f"<br>{indent}{ps}{trail}")
                parts.append(f'<br><span class="hl-punctuation">)</span>')
                sig_html = "".join(parts)
            lines.append(f'<h4><code>{sig_html}</code></h4>\n')

    # Structured docstring — directly under class header (no card wrapper)
    if info.docstring:
        parsed = parse_docstring(info.docstring)
        rendered = _render_docstring(parsed)
        if rendered:
            lines.append(rendered)

    # Collect members
    properties = []
    methods = []

    for member_name in _get_own_members(cls):
        static_attr = inspect.getattr_static(cls, member_name)
        obj = getattr(cls, member_name)

        if isinstance(static_attr, property):
            properties.append((member_name, static_attr))
        elif callable(obj) and not isinstance(obj, type):
            is_cm = isinstance(static_attr, classmethod)
            methods.append((member_name, obj, is_cm))

    # Quick reference table
    if properties or methods:
        lines.append('<h3 class="ref-label">Quick Reference</h3>\n')

        if properties:
            lines.append("**Properties**\n")
            lines.append("| Name | Type |")
            lines.append("|------|------|")
            for prop_name, prop in sorted(properties):
                if hasattr(prop, "fget") and prop.fget:
                    hints = getattr(prop.fget, "__annotations__", {})
                    ret = format_type(str(hints.get("return", "")))
                    lines.append(
                        f"| [`{prop_name}`](#{name.lower().replace('_', '-')}-{prop_name.lower().replace('_', '-')}) | `{ret}` |"
                    )
            lines.append("")

        if methods:
            lines.append("**Methods**\n")
            lines.append("| Method | Description |")
            lines.append("|--------|-------------|")
            for method_name, method, _is_cm in sorted(methods):
                doc = inspect.getdoc(method) or ""
                summary = doc.split("\n")[0].split(". ")[0] if doc else ""
                if len(summary) > 50:
                    summary = summary[:47] + "..."
                lines.append(
                    f"| [`{method_name}()`](#{name.lower().replace('_', '-')}-{method_name.lower().replace('_', '-')}) | {summary} |"
                )
            lines.append("")

    # Properties section
    if properties:
        lines.append('<h3 class="ref-label">Properties</h3>\n')
        for prop_name, prop in sorted(properties):
            if hasattr(prop, "fget") and prop.fget:
                lines.append(generate_method_markdown(prop_name, prop.fget, name, is_property=True))

    # Methods section
    if methods:
        lines.append('<h3 class="ref-label">Methods</h3>\n')
        for method_name, method, is_cm in sorted(methods):
            lines.append(generate_method_markdown(method_name, method, name, is_classmethod=is_cm))

    return "\n".join(lines)


def generate_function_markdown(info: FunctionInfo) -> str:
    """Generate markdown for a function."""
    lines = []

    parsed = parse_docstring(info.docstring)
    params, ret_type = get_signature(info.func, info.name)

    # Qualified name (e.g. "talu.convert" or "talu.converter.verify")
    prefix = info.module

    # Function header — raw HTML to control ID and add keyword prefix
    anchor = info.name.lower().replace("_", "-")
    lines.append(f'<h2 id="{anchor}"><span class="hl-keyword-declaration">def </span><span class="hl-function-name">{info.name}</span></h2>\n')

    # Signature — raw HTML with hl-* syntax tokens (no bg-box)
    ret_suffix = _highlight_return(ret_type) if ret_type else ""
    prefix_html = f'<span class="hl-type-name">{prefix}</span><span class="hl-punctuation">.</span>'

    if not params:
        sig_html = (
            f'{prefix_html}<span class="hl-function-name">{info.name}</span>'
            f'<span class="hl-punctuation">()</span>{ret_suffix}'
        )
    else:
        param_strs = [_highlight_param(pn, pt, pd) for pn, pt, pd in params]
        comma = '<span class="hl-punctuation">, </span>'

        if len(params) <= 2:
            sig_html = (
                f'{prefix_html}<span class="hl-function-name">{info.name}</span>'
                f'<span class="hl-punctuation">(</span>{comma.join(param_strs)}'
                f'<span class="hl-punctuation">)</span>{ret_suffix}'
            )
        else:
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;"
            parts = [
                f'{prefix_html}<span class="hl-function-name">{info.name}</span>'
                f'<span class="hl-punctuation">(</span>'
            ]
            for i, ps in enumerate(param_strs):
                trail = '<span class="hl-punctuation">,</span>' if i < len(param_strs) - 1 else ""
                parts.append(f"<br>{indent}{ps}{trail}")
            parts.append(f'<br><span class="hl-punctuation">)</span>{ret_suffix}')
            sig_html = "".join(parts)
    lines.append(f'<h4><code>{sig_html}</code></h4>\n')

    # Full structured docstring
    rendered = _render_docstring(parsed)
    if rendered:
        lines.append(rendered)

    return "\n".join(lines)


def generate_module_page(module: ModuleInfo) -> str:
    """Generate a complete reference page for a module."""
    lines = []

    # Header — use raw HTML with a distinct ID to avoid collisions
    # with class names (e.g. "Chat" module title vs "Chat" class)
    title_id = f"module-{module.title.lower()}"
    lines.append(f'<h1 id="{title_id}">{module.title}</h1>\n')
    lines.append(f"{module.description}\n")

    # Classes overview
    if module.classes:
        lines.append("## Classes\n")
        lines.append("| Class | Description |")
        lines.append("|-------|-------------|")
        for info in module.classes:
            desc = info.docstring.split("\n")[0] if info.docstring else ""
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(f"| [{info.name}](#{info.name.lower()}) | {desc} |")
        lines.append("")

    # Functions overview
    if module.functions:
        lines.append("## Functions\n")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")
        for info in module.functions:
            desc = info.docstring.split("\n")[0] if info.docstring else ""
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(f"| [{info.name}()](#{info.name.lower()}) | {desc} |")
        lines.append("")

    lines.append("---\n")

    # Class documentation
    for info in module.classes:
        lines.append(generate_class_markdown(info))
        lines.append("---\n")

    # Function documentation
    for info in module.functions:
        lines.append(generate_function_markdown(info))
        lines.append("---\n")

    # Type alias documentation
    for alias in module.type_aliases:
        alias_anchor = alias.name.lower().replace("_", "-")
        lines.append(f'<h2 id="{alias_anchor}"><span class="hl-type-name">{alias.name}</span></h2>\n')
        lines.append(f"```\n{alias.name} = {' | '.join(alias.members)}\n```\n")
        lines.append("---\n")

    return "\n".join(lines)


def generate_index_page(modules: list[ModuleInfo]) -> str:
    """Generate the reference index page."""
    lines = []

    lines.append("# API Reference\n")
    page_to_module = {m.page: m for m in modules}

    def _render_tier(heading: str, pages: list[str]) -> None:
        lines.append(f"## {heading}\n")
        lines.append("| Module | Description |")
        lines.append("|--------|-------------|")
        for page in pages:
            module = page_to_module.get(page)
            if module is None:
                continue
            lines.append(f"| [{module.title}]({module.page}.html) | {module.description} |")
        lines.append("")

    primary = [p for p in MODULE_ORDER if p not in _SECONDARY_MODULES and p not in _SUPPORTING_MODULES]
    secondary = [p for p in MODULE_ORDER if p in _SECONDARY_MODULES]
    supporting = [p for p in MODULE_ORDER if p in _SUPPORTING_MODULES]

    _render_tier("Modules", primary)
    _render_tier("Utilities", secondary)
    _render_tier("Supporting", supporting)

    lines.append("")

    # Top-level functions
    api_module = next((m for m in modules if m.page == "api"), None)
    if api_module and api_module.functions:
        lines.append("## Convenience Functions\n")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")
        for func in api_module.functions:
            desc = func.docstring.split("\n")[0] if func.docstring else ""
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(f"| `talu.{func.name}()` | {desc} |")
        lines.append("")

    return "\n".join(lines)


def build_symbol_registry(modules: list[ModuleInfo]) -> dict[str, dict]:
    """Build a registry mapping exported symbol names to their doc URLs.

    Includes classes, top-level functions, and type aliases from __all__.
    Used for auto-linking in code blocks and search.
    """
    registry: dict[str, dict] = {}

    for module in modules:
        page = module.page

        for cls_info in module.classes:
            name = cls_info.name
            doc = cls_info.docstring.split("\n")[0] if cls_info.docstring else ""
            if len(doc) > 80:
                doc = doc[:77] + "..."
            registry[name] = {
                "url": f"reference/{page}.html#{name.lower()}",
                "type": "class",
                "doc": doc,
            }

        for func_info in module.functions:
            name = func_info.name
            doc = func_info.docstring.split("\n")[0] if func_info.docstring else ""
            if len(doc) > 80:
                doc = doc[:77] + "..."
            registry[name] = {
                "url": f"reference/{page}.html#{name.lower()}",
                "type": "function",
                "doc": doc,
            }

    # Type aliases (union types, Literal aliases) — these have headings now
    for module in modules:
        page = module.page
        for alias in module.type_aliases:
            name = alias.name
            doc = " | ".join(alias.members)
            if len(doc) > 80:
                doc = doc[:77] + "..."
            registry[name] = {
                "url": f"reference/{page}.html#{name.lower()}",
                "type": "type",
                "doc": doc,
            }

    return registry


# Python builtins and typing names that should never be flagged
_BUILTIN_TYPES = frozenset({
    # Builtin types
    "str", "int", "float", "bool", "bytes", "None", "list", "dict", "set",
    "tuple", "type", "object", "complex", "frozenset", "bytearray",
    "memoryview", "range", "slice", "property", "classmethod", "staticmethod",
    # Builtin exceptions
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "AttributeError", "RuntimeError", "OSError", "IOError", "StopIteration",
    # Typing constructs
    "Any", "Optional", "Union", "Iterator", "AsyncIterator", "Generator",
    "AsyncGenerator", "Callable", "Iterable", "AsyncIterable", "Sequence",
    "Mapping", "MutableMapping", "MutableSequence", "MutableSet",
    "Awaitable", "Coroutine", "TypeVar", "Generic", "Protocol",
    "Literal", "Final", "ClassVar", "Self", "TypeAlias",
    # Common stdlib types that appear in signatures
    "Path", "TextIO", "BinaryIO",
    # External types used in TYPE_CHECKING annotations (not talu-owned)
    "BaseModel",
})

# Regex to extract identifiers from type annotation strings
_TYPE_NAME_RE = re.compile(r'\b([A-Z][A-Za-z0-9]+)\b')


def validate_symbol_coverage(
    modules: list[ModuleInfo],
    registry: dict[str, dict],
) -> None:
    """Warn about talu types used in public signatures but not exported.

    Flags CamelCase type names that appear in parameter/return annotations
    but are not in the symbol registry (i.e., not exported via __all__).
    Python builtins and typing constructs are excluded.

    Prints warnings but does not block the build.
    """
    def _extract_type_names(annotation_str: str) -> set[str]:
        """Extract CamelCase type names from an annotation string."""
        return set(_TYPE_NAME_RE.findall(annotation_str))

    warnings: list[str] = []

    def _check_annotation(annotation_str: str, location: str) -> None:
        """Check a single annotation string for unexported types."""
        for name in _extract_type_names(annotation_str):
            if name in _BUILTIN_TYPES or name in registry:
                continue
            warnings.append(
                f"  WARN: {location}: Type '{name}' used in signature "
                f"but not exported via __all__"
            )

    def _check_method(method: Any, method_name: str, class_name: str) -> None:
        """Check all annotations in a method signature."""
        location = f"{class_name}.{method_name}()"
        try:
            sig = inspect.signature(method)
            for p in sig.parameters.values():
                if p.name == "self" or p.annotation == inspect.Parameter.empty:
                    continue
                _check_annotation(format_type(str(p.annotation)), location)
            if sig.return_annotation != inspect.Signature.empty:
                _check_annotation(format_type(str(sig.return_annotation)), location)
        except (ValueError, TypeError):
            pass

    for module in modules:
        for cls_info in module.classes:
            cls = cls_info.cls

            # Check constructor
            if "__init__" in cls.__dict__:
                _check_method(cls.__init__, "__init__", cls_info.name)

            # Check public methods and properties
            for name in dir(cls):
                if name.startswith("_") and name != "__call__":
                    continue
                static_attr = inspect.getattr_static(cls, name)
                obj = getattr(cls, name)
                if isinstance(static_attr, property):
                    if hasattr(static_attr, "fget") and static_attr.fget:
                        hints = getattr(static_attr.fget, "__annotations__", {})
                        ret = hints.get("return", "")
                        if ret:
                            _check_annotation(
                                format_type(str(ret)), f"{cls_info.name}.{name}"
                            )
                elif callable(obj) and not isinstance(obj, type):
                    _check_method(obj, name, cls_info.name)

        for func_info in module.functions:
            location = f"{func_info.module}.{func_info.name}()"
            try:
                sig = inspect.signature(func_info.func)
                for p in sig.parameters.values():
                    if p.annotation == inspect.Parameter.empty:
                        continue
                    _check_annotation(format_type(str(p.annotation)), location)
                if sig.return_annotation != inspect.Signature.empty:
                    _check_annotation(
                        format_type(str(sig.return_annotation)), location
                    )
            except (ValueError, TypeError):
                pass

    if warnings:
        print(f"Symbol coverage: {len(warnings)} warning(s)")
        for w in warnings:
            print(w)


# =============================================================================
# Main
# =============================================================================


def generate_reference_pages() -> tuple[list[tuple[str, str]], dict[str, dict]] | None:
    """Generate reference markdown in memory.

    Returns:
        Tuple of (pages, symbol_registry), or None on validation failure.
        pages is a list of (page_name, markdown_content) tuples.
        page_name is e.g. "chat", "tokenizer", "index".
    """
    print("Discovering modules...")
    modules = discover_modules()

    # Print API surface stats
    n_modules = sum(1 for m in modules if m.classes or m.functions or m.type_aliases)
    n_classes = sum(len(m.classes) for m in modules)
    n_functions = sum(len(m.functions) for m in modules)
    n_type_aliases = sum(len(m.type_aliases) for m in modules)
    n_methods = 0
    n_properties = 0
    for module in modules:
        for cls_info in module.classes:
            for name in _get_own_members(cls_info.cls):
                static_attr = inspect.getattr_static(cls_info.cls, name)
                obj = getattr(cls_info.cls, name)
                if isinstance(static_attr, property):
                    n_properties += 1
                elif callable(obj) and not isinstance(obj, type):
                    n_methods += 1
    print(
        f"API surface: {n_modules} modules, {n_classes} classes, "
        f"{n_methods} methods, {n_properties} properties, "
        f"{n_functions} functions, {n_type_aliases} type aliases"
    )

    print("Validating documentation...")
    for module in modules:
        for cls_info in module.classes:
            validate_class(cls_info.cls, cls_info.name)
        for func_info in module.functions:
            validate_function(func_info.func, func_info.name, module.name)

    if validator.has_errors():
        print(validator.report())
        return None

    pages = []

    print("Generating reference pages...")
    for module in modules:
        if not module.classes and not module.functions:
            continue
        pages.append((module.page, generate_module_page(module)))

    pages.append(("index", generate_index_page(modules)))

    registry = build_symbol_registry(modules)

    print("Validating symbol coverage...")
    validate_symbol_coverage(modules, registry)

    return pages, registry
