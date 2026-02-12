# Documentation System Gaps

Issues discovered during docs system analysis (2024-12-30).

## Issue 1: talu.ops not documented

**Status:** RESOLVED

The `talu.ops` module is now auto-discovered because it has `__all__` defined.
Public subpackages are auto-discovered based on having `__all__` in their `__init__.py`.

---

## Issue 2: Exception classes not documented

**Severity:** High

Exception classes are exported in `__all__` but silently excluded from documentation:
- `TemplateSyntaxError`, `UndefinedError` (from `talu.template`)
- `ConvertError` (from `talu.converter`)
- All ops errors

**Fix required:**
- Decision: Document exceptions OR explicitly exclude with explanation
- If documenting: Update `docgen.py` to handle exception classes
- If excluding: Add comment explaining why in `get_all_public_classes()`

---

## Issue 3: talu.cli not in docs

**Status:** RESOLVED (by design)

The `talu/cli/` module has no `__all__` defined, so it is correctly excluded from
the public API documentation. CLI modules are internal implementation and not
part of the public Python API. This is the correct Pythonic approach - if a module
doesn't define `__all__`, it's not part of the public API.
