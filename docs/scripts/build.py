#!/usr/bin/env python3
"""
Documentation build script.

Builds all pages into docs/dist/:
1. Merges CSS, copies JS
2. Generates API reference via docgen.py (Python source introspection → HTML)
3. Processes hand-authored HTML pages from docs/pages/
4. Builds example pages from examples/*.py

All validation rules are defined in docs/POLICY.md.
"""

import re
import shutil
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter
from pygments import token as T

DOCS_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = DOCS_DIR / "scripts"
CSS_DIR = DOCS_DIR / "styles"
UI_STYLES_DIR = DOCS_DIR.parent / "ui" / "src" / "styles"
JS_DIR = DOCS_DIR / "js"
OUTPUT_DIR = DOCS_DIR / "dist"
EXAMPLES_DIR = DOCS_DIR.parent / "examples"

# Pygments formatter that outputs unified hl-* classes (shared with UI)
_HL_MAP = {
    T.Comment:          "hl-comment",
    T.Keyword.Type:     "hl-keyword-type",
    T.Keyword.Declaration: "hl-keyword-declaration",
    T.Keyword.Namespace: "hl-keyword-namespace",
    T.Keyword:          "hl-keyword",
    T.Literal.String:   "hl-string",
    T.Literal.Number:   "hl-number",
    T.Name.Function:    "hl-function-name",
    T.Name.Class:       "hl-class-name",
    T.Name.Enum:        "hl-enum-name",
    T.Name.Exception:   "hl-type-name",
    T.Name.Decorator:   "hl-decorator-name",
    T.Name.Builtin:     "hl-builtin-name",
    T.Name.Namespace:   "hl-namespace-name",
    T.Name.Attribute:   "hl-property-name",
    T.Name.Property:    "hl-property-name",
    T.Name.Constant:    "hl-constant-name",
    T.Operator:         "hl-operator",
    T.Punctuation:      "hl-punctuation",
    T.Name:             "",
    T.Text:             "",
    T.Other:            "",
}


def _hl_class(ttype):
    """Map a Pygments token type to a unified hl-* class name."""
    while ttype:
        if ttype in _HL_MAP:
            return _HL_MAP[ttype]
        ttype = ttype.parent
    return ""


class UnifiedHtmlFormatter(HtmlFormatter):
    """HtmlFormatter that emits hl-* classes instead of Pygments short names."""

    def _get_css_class(self, ttype):
        return _hl_class(ttype)


# Module-level symbol registry (set by build_reference, used by convert_markdown)
_symbol_registry: dict[str, dict] = {}

# Regex for splitting code blocks on span tags
_SPAN_TAG_RE = re.compile(r'(<span[^>]*>|</span>)')
_SKIP_CLASS_RE = re.compile(r'<span class="hl-(string|comment)[^"]*">')


def auto_link_symbols(html: str, asset_prefix: str, current_page: str) -> str:
    """Replace known symbol names in code blocks with clickable links.

    Processes text nodes inside <div class="highlight"> blocks, skipping
    text inside string and comment spans to avoid false matches.
    """
    if not _symbol_registry:
        return html

    # Build regex matching any registry key as a whole word (longest first)
    names = sorted(_symbol_registry.keys(), key=len, reverse=True)
    name_pattern = re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b')

    def _linkify_text(text: str) -> str:
        """Replace registry matches in a text node."""
        def _replacer(m: re.Match) -> str:
            name = m.group(1)
            info = _symbol_registry[name]
            url = asset_prefix + info["url"]
            return f'<a href="{url}" class="api-link">{name}</a>'
        return name_pattern.sub(_replacer, text)

    def _process_block(block_match: re.Match) -> str:
        """Process a single highlight block."""
        block = block_match.group(0)
        parts = _SPAN_TAG_RE.split(block)
        skip_depth = 0
        result = []

        for part in parts:
            if _SKIP_CLASS_RE.match(part):
                skip_depth += 1
                result.append(part)
            elif part == '</span>' and skip_depth > 0:
                skip_depth -= 1
                result.append(part)
            elif skip_depth == 0 and not part.startswith('<'):
                result.append(_linkify_text(part))
            else:
                result.append(part)

        return ''.join(result)

    # Pass 1: code blocks (highlight divs) — needs span-aware parsing
    html = re.sub(
        r'<div class="highlight">.*?</div>',
        _process_block,
        html,
        flags=re.DOTALL,
    )

    # Pass 2: inline <code> tags (e.g. property type columns in tables)
    def _process_inline_code(m: re.Match) -> str:
        inner = m.group(1)
        # Skip if already contains a link
        if '<a ' in inner:
            return m.group(0)
        linked = _linkify_text(inner)
        if linked == inner:
            return m.group(0)
        return f'<code>{linked}</code>'

    html = re.sub(r'<code>([^<]+)</code>', _process_inline_code, html)

    # Pass 3: hl-type-name spans in method/property signatures
    def _process_type_name(m: re.Match) -> str:
        inner = m.group(1)
        if '<a ' in inner:
            return m.group(0)
        linked = _linkify_text(inner)
        if linked == inner:
            return m.group(0)
        return f'<span class="hl-type-name">{linked}</span>'

    html = re.sub(
        r'<span class="hl-type-name">([^<]+)</span>', _process_type_name, html
    )

    return html


# Markdown processor with extensions
md = markdown.Markdown(
    extensions=[
        "tables",
        "fenced_code",
        "toc",
        "attr_list",
        "codehilite",
    ],
    extension_configs={
        "codehilite": {
            "css_class": "highlight",
            "guess_lang": False,
            "pygments_formatter": UnifiedHtmlFormatter,
        }
    },
)


# =============================================================================
# Navigation
# =============================================================================


def snake_to_camel(name: str) -> str:
    """Convert snake_case or kebab-case to CamelCase.

    Examples:
        chat_session -> Chat
        chat-session -> Chat
        local_engine -> LocalEngine
    """
    # Normalize: replace hyphens with underscores
    name = name.replace("-", "_")

    # Strip leading underscore for private modules
    if name.startswith("_"):
        name = name[1:]

    # Split on underscores and capitalize each part
    return "".join(word.capitalize() for word in name.split("_"))


def title_from_path(path: Path) -> str:
    """Convert path to title using CamelCase for API-related pages.

    Reference and Examples pages use CamelCase since they document Python classes.
    Other pages (guides, getting-started) use Title Case with spaces.
    """
    name = path.stem if path.is_file() else path.name
    if name == "index":
        name = path.parent.name

    # Reference and Examples pages use CamelCase (they document Python classes)
    # Auto-detect by checking if path contains "reference" or "examples"
    if "reference" in str(path) or "examples" in str(path):
        return snake_to_camel(name)

    # For other pages like "getting-started", use Title Case with spaces
    return name.replace("-", " ").replace("_", " ").title()



def get_module_all_exports(module_name: str) -> list[tuple[str, str | None]]:
    """Get __all__ exports from a module, preserving exact order.

    Parses the __init__.py to extract items and comments in order.
    Returns list of tuples:
      - ("name", None) for export names
      - (None, "comment text") for comment lines
    """
    import importlib

    try:
        if module_name == "talu":
            mod = importlib.import_module("talu")
        elif module_name.startswith("talu."):
            mod = importlib.import_module(module_name)
        else:
            mod = importlib.import_module(f"talu.{module_name}")
    except ImportError:
        return []

    if not hasattr(mod, "__all__"):
        return []

    # Parse __all__ block preserving exact order including comments
    try:
        import inspect
        source = inspect.getsource(mod)

        items = []
        in_all_block = False

        for line in source.split("\n"):
            stripped = line.strip()

            if "__all__" in line and "=" in line:
                in_all_block = True
                continue

            if in_all_block:
                if stripped.startswith("]"):
                    break

                # Comment line
                if stripped.startswith("#"):
                    comment = stripped[1:].strip()
                    if comment:  # Skip empty comments
                        items.append((None, comment))
                    continue

                # Export name
                if '"' in stripped or "'" in stripped:
                    for quote in ['"', "'"]:
                        if quote in stripped:
                            start = stripped.index(quote) + 1
                            end = stripped.index(quote, start)
                            name = stripped[start:end]
                            if name in mod.__all__:
                                items.append((name, None))
                            break

        # Fallback if parsing failed
        if not items:
            return [(name, None) for name in mod.__all__]

        return items

    except Exception:
        return [(name, None) for name in mod.__all__]


def build_symbol_to_page_index() -> dict[str, str]:
    """Build index mapping symbol names to their documentation page.

    Returns dict like {"Chat": "chat", "Tokenizer": "tokenizer"}.
    Uses MODULE_PAGES from docgen to avoid reading intermediate files.
    """
    from docgen import MODULE_PAGES

    index = {}
    for full_module, config in MODULE_PAGES.items():
        page = config["page"]
        # Get explicit classes list or module exports
        classes = config.get("classes")
        if classes:
            for name in classes:
                index[name] = page
        elif full_module != "talu":
            # Use module's __all__
            exports = get_module_all_exports(full_module)
            for name, _comment in exports:
                if name is not None:
                    index[name] = page

    return index



def build_main_nav(docs_dir: Path, current_path: str) -> list:
    """Build the fixed sidebar navigation (shown on every page)."""
    nav = []

    # Getting Started (always expanded)
    get_started = {
        "title": "Getting Started",
        "path": "getting-started.html",
        "children": [
            {"title": "Install", "path": "install.html", "children": []},
            {"title": "Command Line", "path": "cli.html", "children": []},
            {"title": "Python", "path": "examples/python.html", "children": []},
        ],
    }
    nav.append(get_started)

    # Reference — always expanded
    from docgen import MODULE_PAGES, MODULE_ORDER, _SECONDARY_MODULES, _SUPPORTING_MODULES
    ref_children = []
    page_to_config = {c["page"]: c for c in MODULE_PAGES.values()}
    prev_tier = None
    for page in MODULE_ORDER:
        config = page_to_config.get(page)
        if config is None:
            continue
        # Determine tier and insert separator on tier change
        if page in _SUPPORTING_MODULES:
            tier = "supporting"
        elif page in _SECONDARY_MODULES:
            tier = "secondary"
        else:
            tier = "primary"
        if prev_tier is not None and tier != prev_tier:
            ref_children.append({"is_separator": True})
        prev_tier = tier
        ref_children.append({
            "title": config["title"],
            "path": f"reference/{page}.html",
            "children": [],
        })

    nav.append({"title": "Reference", "path": "reference/index.html", "children": ref_children})

    return nav


def build_examples_nav(docs_dir: Path, current_path: str) -> list:
    """Build Examples-only navigation.

    Shows categories that contain .py files. Scans source examples/ at repo root.
    """
    if not EXAMPLES_DIR.exists():
        return []

    # Find current category path and whether we're on an individual example page
    current_cat = None
    is_individual = False
    if current_path.startswith("examples/") and current_path != "examples/index.html":
        # Strip "examples/" prefix and ".html" suffix
        inner = current_path.replace("examples/", "").replace(".html", "")
        # Check if this is an individual page (the category index .html won't have
        # a matching directory with .py files — it's the category itself)
        candidate_dir = EXAMPLES_DIR / inner
        if candidate_dir.is_dir() and any(candidate_dir.glob("*.py")):
            # e.g. examples/basics/01_chat.html — inner is "basics/01_chat"
            # but basics/01_chat is not a dir. Let's check parent.
            pass
        # The category path is everything except the last segment for individual pages
        parts = inner.split("/")
        cat_dir = EXAMPLES_DIR / inner
        if cat_dir.is_dir() and any(cat_dir.glob("*.py")):
            # This is a category index page (e.g. examples/basics.html doesn't match,
            # but something like examples/developers/chat.html could)
            current_cat = inner
        else:
            # Individual example page — category is the parent path
            current_cat = "/".join(parts[:-1]) if len(parts) > 1 else parts[0]
            is_individual = True

    nav = []

    # Add back link when viewing individual example pages
    if is_individual and current_cat:
        nav.append({
            "title": "« Examples",
            "path": f"examples/{current_cat}.html",
            "children": [],
            "is_back": True,
        })

    # Only show python examples
    python_dir = EXAMPLES_DIR / "python"
    if python_dir.is_dir() and any(python_dir.glob("*.py")):
        nav.append({
            "title": "Python",
            "path": "examples/python.html",
            "children": [],
            "is_current_module": current_cat == "python",
        })

    return nav


def build_nav_tree(docs_dir: Path, current_path: str = "") -> list:
    """Build navigation tree for the sidebar."""
    return build_main_nav(docs_dir, current_path)


def render_nav(nav: list, current_path: str, level: int = 0) -> str:
    """Render navigation as HTML.

    Rustdoc-style: all sections always expanded, no toggle arrows.
    Categories rendered as section headers.
    """
    html = []

    for item in nav:
        # Handle separator
        if item.get("is_separator"):
            html.append('<li class="nav-separator"><hr></li>')
            continue

        # Exact match, section match, or child match
        item_path = item["path"]
        is_current = item_path == current_path
        if not is_current and item_path.endswith("/index.html"):
            section = item_path.rsplit("/", 1)[0] + "/"
            is_current = current_path.startswith(section)
        if not is_current and item["children"]:
            child_paths = [c["path"] for c in item["children"] if not c.get("is_separator")]
            is_current = current_path in child_paths
        has_children = bool(item["children"])
        is_label = item.get("is_label", False)
        is_back = item.get("is_back", False)
        is_current_module = item.get("is_current_module", False)
        is_current_example = item.get("is_current_example", False)

        classes = []
        if is_current or is_current_module or is_current_example:
            classes.append("current")
        if has_children:
            # Always open - rustdoc style
            classes.append("has-children")
            classes.append("open")
        if is_label:
            classes.append("is-label")
        if is_back:
            classes.append("is-back")
        if is_current_module:
            classes.append("current-module")

        class_attr = f' class="{" ".join(classes)}"' if classes else ""

        # Calculate relative path
        depth = current_path.count("/")
        prefix = "../" * depth if depth > 0 else ""
        href = prefix + item["path"]

        html.append(f"<li{class_attr}>")

        # Labels are spans, not links
        if is_label:
            html.append(f'<span class="nav-label">{item["title"]}</span>')
        else:
            html.append(f'<a href="{href}">{item["title"]}</a>')

        if has_children:
            html.append("<ul>")
            html.append(render_nav(item["children"], current_path, level + 1))
            html.append("</ul>")

        html.append("</li>")

    return "\n".join(html)


def get_page_order() -> list:
    """Get ordered list of all pages for prev/next navigation."""
    pages = [
        ("getting-started.html", "Getting Started"),
        ("install.html", "Install"),
        ("cli.html", "Command Line"),
    ]

    # Reference - from MODULE_PAGES
    from docgen import MODULE_PAGES

    pages.append(("reference/index.html", "API Reference"))
    for config in sorted(MODULE_PAGES.values(), key=lambda c: c["page"]):
        page = config["page"]
        if page == "api":
            continue  # Top-level API is not in the page order
        pages.append((f"reference/{page}.html", config["title"]))

    return pages


def generate_prev_next(html_path: str, asset_prefix: str) -> str:
    """Generate prev/next navigation HTML."""
    pages = get_page_order()
    page_paths = [p[0] for p in pages]

    if html_path not in page_paths:
        return ""

    idx = page_paths.index(html_path)
    prev_page = pages[idx - 1] if idx > 0 else None
    next_page = pages[idx + 1] if idx < len(pages) - 1 else None

    html = ['<nav class="page-nav">']

    if prev_page:
        html.append(f'<a href="{asset_prefix}{prev_page[0]}" class="page-nav-prev">')
        html.append('<span class="page-nav-label">← Previous</span>')
        html.append(f'<span class="page-nav-title">{prev_page[1]}</span>')
        html.append("</a>")

    if next_page:
        html.append(f'<a href="{asset_prefix}{next_page[0]}" class="page-nav-next">')
        html.append('<span class="page-nav-label">Next →</span>')
        html.append(f'<span class="page-nav-title">{next_page[1]}</span>')
        html.append("</a>")

    html.append("</nav>")
    return "\n".join(html)


def generate_toc(body: str) -> str:
    """Generate table of contents from headings.

    Disabled — sidebar navigation is sufficient. Kept as a stub
    so callers don't need to change.
    """
    return ""


# =============================================================================
# Markdown Conversion
# =============================================================================


def escape_markdown_for_html(text: str) -> str:
    """Escape markdown for embedding in HTML script tag."""
    return text.replace("</script>", "<\\/script>")


def convert_markdown(content: str, html_path: str, template: str) -> str:
    """Convert markdown content to HTML page.

    Args:
        content: Markdown source text.
        html_path: Virtual output path (e.g. "examples/chat.html") for nav/asset resolution.
        template: HTML template string.
    """
    markdown_escaped = escape_markdown_for_html(content)

    # Reset markdown processor state
    md.reset()

    # Convert markdown to HTML
    body = md.convert(content)

    # Convert .md links to .html
    body = re.sub(r'href="([^"]+)\.md"', r'href="\1.html"', body)

    # Auto-link known symbols in code blocks
    depth = html_path.count("/")
    asset_prefix_for_links = "../" * depth if depth > 0 else ""
    body = auto_link_symbols(body, asset_prefix_for_links, html_path)

    # Extract title from first h1
    title_match = re.search(r"<h1[^>]*>(.+?)</h1>", body)
    title = title_match.group(1) if title_match else html_path.replace(".html", "").split("/")[-1]

    # Build navigation tree for this specific page
    nav = build_nav_tree(DOCS_DIR, html_path)
    nav_html = render_nav(nav, html_path)

    # Calculate asset prefix
    depth = html_path.count("/")
    asset_prefix = "../" * depth if depth > 0 else ""

    # Generate prev/next navigation
    prev_next = generate_prev_next(html_path, asset_prefix)

    # Generate table of contents
    toc = generate_toc(body)

    # Examples link for reference pages
    examples_link = ""
    if html_path.startswith("reference/") and html_path != "reference/index.html":
        module_name = html_path.replace("reference/", "").replace(".html", "")
        examples_dir = EXAMPLES_DIR / module_name.replace("-", "_")
        if examples_dir.exists() and any(examples_dir.glob("*.py")):
            examples_link = f'<a href="{asset_prefix}examples/{module_name.replace("-", "_")}.html" class="examples-link">📖 Examples</a>'

    # Fill template
    html = template.replace("{{title}}", title)
    html = html.replace("{{body_class}}", "")
    html = html.replace("{{nav}}", nav_html)
    html = html.replace("{{ref_nav}}", "")
    html = html.replace("{{content}}", body + prev_next)
    html = html.replace("{{toc}}", toc)
    html = html.replace("{{asset_prefix}}", asset_prefix)
    html = html.replace("{{markdown}}", markdown_escaped)
    html = html.replace("{{examples_link}}", examples_link)

    return html


# =============================================================================
# CSS Merging
# =============================================================================


def merge_css_files() -> str:
    """Merge shared UI CSS + docs overrides into a single docs/dist/style.css."""
    css_order = [
        # Shared (single source of truth: ui/src/styles/)
        (UI_STYLES_DIR / "colors.css", "ui/src/styles/colors.css"),
        (UI_STYLES_DIR / "syntax.css", "ui/src/styles/syntax.css"),
        (UI_STYLES_DIR / "base.css", "ui/src/styles/base.css"),
        (UI_STYLES_DIR / "components.css", "ui/src/styles/components.css"),
        (UI_STYLES_DIR / "markdown.css", "ui/src/styles/markdown.css"),
        # Docs-specific overrides
        (CSS_DIR / "docs" / "base.css", "docs/styles/docs/base.css"),
        (CSS_DIR / "docs" / "layout.css", "docs/styles/docs/layout.css"),
        (CSS_DIR / "docs" / "typography.css", "docs/styles/docs/typography.css"),
        (CSS_DIR / "docs" / "navigation.css", "docs/styles/docs/navigation.css"),
        (CSS_DIR / "docs" / "components.css", "docs/styles/docs/components.css"),
        (CSS_DIR / "docs" / "shared-overrides.css", "docs/styles/docs/shared-overrides.css"),
        (CSS_DIR / "docs" / "pages.css", "docs/styles/docs/pages.css"),
    ]

    merged = []
    merged.append(
        "/* ============================================================================="
    )
    merged.append("   TALU DOCS - Merged CSS (auto-generated by build.py)")
    merged.append("   Do not edit directly - modify ui/src/styles/ or docs/styles/docs/")
    merged.append(
        "   ============================================================================= */\n"
    )

    for css_path, label in css_order:
        if css_path.exists():
            content = css_path.read_text()
            merged.append(f"\n/* --- {label} --- */\n")
            merged.append(content)
        else:
            print(f"  Warning: {label} not found")

    return "\n".join(merged)


# =============================================================================
# Main Build
# =============================================================================


def build_reference(template: str) -> int | None:
    """Build API reference HTML directly from Python source introspection.

    Calls docgen to discover, validate, and generate markdown in memory,
    then converts to HTML and writes to dist/reference/.

    Returns:
        Number of pages built, or None on validation failure.
    """
    from docgen import generate_reference_pages

    global _symbol_registry

    print("Generating API reference...")
    result = generate_reference_pages()
    if result is None:
        return None

    pages, _symbol_registry = result
    print(f"Symbol registry: {len(_symbol_registry)} symbols")

    print("Building reference HTML...")
    ref_dir = OUTPUT_DIR / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for page_name, md_content in pages:
        html_path = f"reference/{page_name}.html"
        html = convert_markdown(md_content, html_path, template)
        (ref_dir / f"{page_name}.html").write_text(html)
        count += 1
        print(f"  reference/{page_name}.html")

    return count


def _parse_example_title(content: str) -> str:
    """Extract title (first line of docstring) from an example file."""
    doc_content = content
    if content.startswith("#!"):
        newline_idx = content.find("\n")
        if newline_idx != -1:
            doc_content = content[newline_idx + 1 :].lstrip()

    for quote in ('"""', "'''"):
        if doc_content.startswith(quote):
            end = doc_content.find(quote, 3)
            if end != -1:
                return doc_content[3:end].strip().split("\n", 1)[0].strip()

    return ""


def _discover_example_categories(root: Path) -> list[dict]:
    """Scan examples/python/ for .py files.

    Returns a list with a single "python" category containing its example files.
    """
    python_dir = root / "python"
    if not python_dir.is_dir():
        return []

    py_files = sorted(python_dir.glob("*.py"))
    if not py_files:
        return []

    examples = []
    for f in py_files:
        content = f.read_text()
        title = _parse_example_title(content) or snake_to_camel(f.stem)
        examples.append({
            "name": f.stem,
            "title": title,
            "content": content,
            "file": f.name,
        })

    return [{"path": "python", "examples": examples}]


def build_examples(template: str) -> int:
    """Build examples HTML directly from examples/*.py source files.

    Reads from repo-root examples/, converts to HTML, writes to dist/examples/.
    Handles nested structures (e.g. developers/chat/).

    Returns:
        Number of pages built.
    """
    if not EXAMPLES_DIR.exists():
        return 0

    print("Building examples...")
    count = 0

    all_categories = _discover_example_categories(EXAMPLES_DIR)

    # Individual example pages
    for cat_data in all_categories:
        cat_path = cat_data["path"]
        cat_dir = OUTPUT_DIR / "examples" / cat_path
        cat_dir.mkdir(parents=True, exist_ok=True)

        for example in cat_data["examples"]:
            md_lines = [f"# {example['title']}\n"]
            md_lines.append("```python")
            md_lines.append(example["content"].strip())
            md_lines.append("```\n")

            html_path = f"examples/{cat_path}/{example['name']}.html"
            html = convert_markdown("\n".join(md_lines), html_path, template)
            (cat_dir / f"{example['name']}.html").write_text(html)
            count += 1

        print(f"  examples/{cat_path}/ ({len(cat_data['examples'])} files)")

    # Category index pages
    for cat_data in all_categories:
        cat_path = cat_data["path"]
        title = " / ".join(snake_to_camel(p) for p in cat_path.split("/"))

        md_lines = [f"# {title} Examples\n"]
        # Links are relative to the page's directory, so use last path segment
        cat_last = cat_path.split("/")[-1]
        for example in cat_data["examples"]:
            link_path = f"{cat_last}/{example['name']}.html"
            md_lines.append(f"### [{example['file']}]({link_path})")
            md_lines.append(f"{example['title']}\n")

        html_path = f"examples/{cat_path}.html"
        out_path = OUTPUT_DIR / "examples" / f"{cat_path}.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        html = convert_markdown("\n".join(md_lines), html_path, template)
        out_path.write_text(html)
        count += 1
        print(f"  examples/{cat_path}.html")

    (OUTPUT_DIR / "examples").mkdir(parents=True, exist_ok=True)

    return count


def build() -> bool:
    """Build all documentation.

    Returns:
        True if successful, False if any validation failed.
    """
    print("Building HTML documentation...")

    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Load template
    template = (DOCS_DIR / "template.html").read_text()

    # Merge CSS files
    print("Merging CSS files...")
    merged_css = merge_css_files()
    (OUTPUT_DIR / "style.css").write_text(merged_css)

    # Copy JS
    shutil.copy(JS_DIR / "nav.js", OUTPUT_DIR / "nav.js")

    # Build API reference (validates + generates HTML directly to dist)
    ref_count = build_reference(template)
    if ref_count is None:
        return False

    # Process hand-authored HTML pages from docs/pages/
    count = ref_count
    pages_src = DOCS_DIR / "pages"
    if pages_src.exists():
        for html_file in sorted(pages_src.rglob("*.html")):
            content = html_file.read_text()
            rel_path = html_file.relative_to(pages_src)
            page_path = str(rel_path)

            # Extract title
            title_match = re.search(r"<title>([^<]+)</title>", content)
            title = (
                title_match.group(1).replace(" - talu", "")
                if title_match
                else html_file.stem.title()
            )

            # Extract article content
            article_match = re.search(r"<article[^>]*>(.*?)</article>", content, re.DOTALL)
            article_content = article_match.group(1).strip() if article_match else ""

            # Remove article-actions that we'll add via template
            article_content = re.sub(
                r'<div class="article-actions">.*?</div>', "", article_content, flags=re.DOTALL
            )

            # Extract page-specific scripts
            script_matches = re.findall(r"<script>([^<]+)</script>", content)
            page_scripts = "\n".join(script_matches) if script_matches else ""

            # Build and render navigation for this page
            nav = build_nav_tree(DOCS_DIR, page_path)
            nav_html = render_nav(nav, page_path)

            # Calculate asset prefix for subdirectory pages
            depth = page_path.count("/")
            asset_prefix = "../" * depth if depth > 0 else ""

            # Generate prev/next navigation
            prev_next = generate_prev_next(page_path, asset_prefix)

            # Build the page using template
            html = template.replace("{{title}}", title)
            html = html.replace("{{body_class}}", "")
            html = html.replace("{{nav}}", nav_html)
            html = html.replace("{{ref_nav}}", "")
            html = html.replace("{{asset_prefix}}", asset_prefix)
            html = html.replace("{{examples_link}}", "")
            html = html.replace("{{toc}}", "")
            html = html.replace("{{markdown}}", "")
            html = html.replace("{{content}}", article_content + prev_next)

            # Add page-specific scripts
            if page_scripts:
                html = html.replace("</body>", f"  <script>\n{page_scripts}\n  </script>\n</body>")

            out_path = OUTPUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html)
            print(f"  {page_path}")
            count += 1

    # Build examples directly from examples/*.py source files
    count += build_examples(template)

    # index.html → redirect to getting-started.html
    (OUTPUT_DIR / "index.html").write_text(
        '<!DOCTYPE html>\n'
        '<html><head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta http-equiv="refresh" content="0;url=getting-started.html">\n'
        '  <title>talu</title>\n'
        '</head><body>\n'
        '  <p>Redirecting to <a href="getting-started.html">Getting Started</a>...</p>\n'
        '</body></html>\n'
    )

    # Write search index
    if _symbol_registry:
        import json
        search_index = [
            {"name": name, "url": info["url"], "type": info["type"], "doc": info["doc"]}
            for name, info in sorted(_symbol_registry.items())
        ]
        (OUTPUT_DIR / "search-index.json").write_text(json.dumps(search_index))
        print(f"Search index: {len(search_index)} entries")

    print(f"\nBuilt {count} pages → {OUTPUT_DIR}")
    return True


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
