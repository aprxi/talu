"""
Custom Filters - Use register_filter() to add domain-specific formatting.

Primary API: talu.PromptTemplate.register_filter, talu.TemplateEnvironment.register_filter
Scope: Single

Register custom Python functions as Jinja2 filters to add domain-specific
formatting logic directly in your templates.

Two ways to register filters:
- template.register_filter() - for a single template
- env.register_filter() - for all templates from that environment (see environment.py)

Related:
- examples/developers/template/environment.py
"""

import talu

# =============================================================================
# Basic Custom Filters
# =============================================================================

# Create a template with a custom filter
template = talu.PromptTemplate("{{ name | shout }}")

# Register the filter - it receives the piped value as first argument
template.register_filter("shout", lambda s: s.upper() + "!!!")

print("Basic filter:")
print(template(name="hello"))
print()

# =============================================================================
# Filters with Arguments
# =============================================================================

# Filters can take additional arguments from the template
template = talu.PromptTemplate("{{ amount | currency('EUR') }}")


def format_currency(amount: float, symbol: str = "USD") -> str:
    """Format a number as currency."""
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return f"{symbols.get(symbol, symbol)}{amount:,.2f}"


template.register_filter("currency", format_currency)

print("Filter with arguments:")
print(template(amount=1234.56))
print()

# =============================================================================
# Chaining Filters
# =============================================================================

# Multiple custom filters can be chained together
template = (
    talu.PromptTemplate("{{ text | clean | truncate(20) | emphasize }}")
    .register_filter("clean", lambda s: s.strip().replace("\n", " "))
    .register_filter("truncate", lambda s, n: s[:n] + "..." if len(s) > n else s)
    .register_filter("emphasize", lambda s: f"**{s}**")
)

print("Chained filters:")
print(template(text="  This is a very long piece of text that needs processing  "))
print()

# =============================================================================
# Structured Data Formatting
# =============================================================================

report_template = talu.PromptTemplate("""
{{ user | format_user }}

Stats: {{ stats | format_stats }}

Notes: {{ notes | redact_pii }}
""")


def format_user(record: dict) -> str:
    """Format user record."""
    return f"{record['name']} (ID: {record['id']})"


def format_stats(stats: dict) -> str:
    """Format statistics as key-value pairs."""
    return " | ".join(f"{k}: {v}" for k, v in stats.items())


def redact_pii(text: str) -> str:
    """Redact personally identifiable information."""
    import re

    # Redact phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    # Redact emails
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
    return text


report_template.register_filter("format_user", format_user)
report_template.register_filter("format_stats", format_stats)
report_template.register_filter("redact_pii", redact_pii)

print("Structured data formatting:")
print(
    report_template(
        user={"name": "Alice", "id": "U12345"},
        stats={"logins": 42, "posts": 15, "score": 89.5},
        notes="Contact at 555-123-4567 or alice@example.com for details.",
    )
)
print()

# =============================================================================
# Override Built-in Filters
# =============================================================================

# Custom filters take precedence over built-ins
template = talu.PromptTemplate("{{ items | join(', ') }}")

# Override the built-in join to add "and" before last item
def oxford_join(items: list, sep: str = ", ") -> str:
    """Join with Oxford comma."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{sep.join(items[:-1])}{sep}and {items[-1]}"


template.register_filter("join", oxford_join)

print("Custom join with Oxford comma:")
print(template(items=["apples", "oranges", "bananas"]))
print()

# =============================================================================
# Chat Message Formatting
# =============================================================================

chat_template = talu.PromptTemplate("""
{% for msg in messages %}
{{ msg.role | role_icon }} {{ msg.content | wrap_content }}
{% endfor %}
""")

chat_template.register_filter(
    "role_icon",
    lambda r: {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(r, "❓"),
)
chat_template.register_filter("wrap_content", lambda s: f"「{s}」")

print("Chat message formatting:")
print(
    chat_template(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    )
)

"""
Topics covered:
* template.render
* template.control.flow
"""
