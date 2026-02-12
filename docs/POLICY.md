# Documentation Policy

How `docs/` works: what's hand-authored, what's generated, and how the build pipeline fits together.

---

## Directory Layout

```
docs/
├── pyproject.toml          # Self-contained deps (markdown, pygments, talu as path dep)
├── .python-version         # Python version for uv
├── template.html           # Base HTML template (all pages share this)
├── pages/                  # Hand-authored HTML pages (all content lives here)
│   ├── index.html          # Landing page
│   ├── python.html         # Python library guide
│   ├── cli.html            # CLI guide
│   ├── building.html       # Building from source
│   └── getting-started/
│       ├── index.html
│       ├── quickstart.html
│       └── supported-models.html
├── styles/                 # Docs-specific CSS overrides only
│   └── docs/               # Layered on top of ui/src/styles shared base
├── js/                     # Hand-authored JavaScript
│   └── nav.js
├── scripts/
│   ├── build.py            # Main build script (orchestrates everything)
│   └── docgen.py           # API reference: discovery, validation, markdown generation
└── dist/                   # AUTO-GENERATED (gitignored) — final HTML output
```

### What's auto-generated (do not edit)

`docs/dist/` — the only generated artifact. `build.py` wipes and recreates it entirely each build. Everything goes straight to HTML in dist — no intermediate files anywhere in `docs/`.

- `dist/reference/` — API reference HTML (from Python source introspection via `docgen.py`)
- `dist/examples/` — example pages HTML (from `examples/*.py` source files)
- `dist/*.html` — hand-authored pages
- `dist/style.css` — merged CSS
- `dist/nav.js` — copied JS

### Everything else is hand-authored source

All hand-authored content is HTML. No markdown source files.

- `pages/` — all content pages
- `template.html` — shared HTML shell
- `styles/docs/`, `js/` — docs-specific assets
- `scripts/build.py`, `scripts/docgen.py` — build tooling

---

## Build Pipeline

```
make docs
  └─ cd docs && uv run python scripts/build.py
       │
       ├─ 1. build.py: merge CSS, copy JS
       │
       ├─ 2. build.py → docgen.py: build API reference
       │     - Discovers public modules via bindings/python/tests/discovery.py
       │     - Validates docstrings and type annotations (fails on violations)
       │     - Generates markdown in memory, converts to HTML, writes to dist/
       │
       ├─ 3. build.py: process pages/ → dist/
       │     - Extracts <article> content from each HTML page
       │     - Wraps in template.html with navigation, prev/next links
       │     - Handles subdirectories (e.g. pages/getting-started/)
       │
       ├─ 4. build.py: build examples from examples/*.py
       │     - Reads source files from repo-root examples/
       │     - Generates HTML pages, category indexes, and index grid
       │
       └─ Output: docs/dist/
```

The docs directory is self-contained. It has its own `pyproject.toml` with only three dependencies: `talu` (path dep to `../bindings/python`), `markdown`, and `pygments`.

```bash
# From repo root
make docs

# Or directly from docs/
cd docs && uv run python scripts/build.py
```

---

## Build-time Validation

`docgen.py` validates the Python public API during reference generation. These checks ensure the generated docs are complete — if any fail, the build stops:

| Check | Error |
|-------|-------|
| Subpackage has non-empty `__all__` | `Module 'talu.foo' missing __all__` |
| Class has docstring | `ClassName: Missing class docstring` |
| Method/property has docstring | `ClassName.method(): Missing docstring` |
| Parameters have type annotations | `ClassName.method(): Parameter 'x' missing type annotation` |
| Return has type annotation | `ClassName.method(): Missing return type annotation` |

These are documentation completeness checks — they ensure every public symbol has enough information to generate a useful reference page. Code quality enforcement (linting, type checking, testing) is owned by `bindings/python/`.

---

## Adding Content

**New page:**
1. Create `docs/pages/<page>.html` (or `docs/pages/<section>/<page>.html`) with an `<article class="markdown-body">` element
2. Add navigation entries in `build_main_nav()` in `build.py`
3. `build.py` extracts the article, wraps it in the template, and adds navigation

**New class in existing module:**
Nothing to do here — `make docs` regenerates the reference page from source.

**New module:**
1. Create `talu/<module>/__init__.py` with `__all__` defined
2. Add docstrings and types to exported classes
3. `make docs` auto-discovers and generates the reference page

**New example:**
1. Create `examples/<category>/<name>.py` (supports nesting, e.g. `examples/developers/chat/`)
2. Start with a docstring (becomes the example title)
3. `make docs` picks it up automatically

---

## Examples

Runnable examples live in `examples/` (repo root). `build.py` reads them directly and generates HTML pages in `dist/examples/`. Categories are discovered automatically from the directory structure — any directory containing `.py` files becomes a category.

### Example Format

```python
"""
Basic Chat - Send a message and stream the response.
"""

from talu import Chat

session = Chat("Qwen/Qwen3-0.6B")

for chunk in session.send("Hello!"):
    print(chunk, end="", flush=True)

print()
```

The docstring becomes the example title. The code is rendered with syntax highlighting.
