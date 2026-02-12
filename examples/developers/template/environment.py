"""
Template Environment - Share filters and globals across multiple templates.

Primary API: talu.TemplateEnvironment, talu.template.TemplateEnvironment
Scope: Single

When you have 50+ templates in an application, you don't want to register
the same filters and pass the same globals to every single one. The
TemplateEnvironment class solves this by providing shared configuration.

Benefits:
- Register filters once, use in all templates
- Define global variables (app name, version, feature flags)
- Set default strict mode for all templates
- No boilerplate factory functions

Related:
- examples/developers/template/custom_filters.py
"""

from talu import TemplateEnvironment

# =============================================================================
# The Problem: Repetitive Configuration
# =============================================================================

# Without environment, you repeat yourself constantly:

# t1 = PromptTemplate("{{ price | currency }}")
# t1.register_filter("currency", format_currency)
#
# t2 = PromptTemplate("{{ amount | currency }}")
# t2.register_filter("currency", format_currency)  # Again!
#
# # Or you write a factory function:
# def make_template(source):
#     t = PromptTemplate(source)
#     t.register_filter("currency", format_currency)
#     t.register_filter("date", format_date)
#     return t

# =============================================================================
# Solution: TemplateEnvironment
# =============================================================================

# Create an environment with shared configuration
env = TemplateEnvironment(strict=True)  # All templates are strict by default

# Register filters once
env.register_filter("currency", lambda x: f"${x:,.2f}")
env.register_filter("percentage", lambda x: f"{x:.1%}")
env.register_filter("shout", lambda s: s.upper() + "!")
env.register_filter("truncate_words", lambda s, n=10: " ".join(s.split()[:n]) + "...")

# Set global variables available to all templates
env.globals["app_name"] = "MyAssistant"
env.globals["version"] = "2.1"
env.globals["support_email"] = "help@myapp.com"

# Create templates from the environment - they all have access to filters and globals
t1 = env.from_string("Total: {{ amount | currency }}")
t2 = env.from_string("Growth: {{ rate | percentage }}")

print("Environment filters:")
print(f"  {t1(amount=1234.56)}")
print(f"  {t2(rate=0.156)}")

# Templates automatically have access to globals
greeting = env.from_string("""
Welcome to {{ app_name }} v{{ version }}!
For help, contact {{ support_email }}.
""".strip())

print("\nEnvironment globals:")
print(greeting())  # No variables needed!

# =============================================================================
# Multiple Environments for Different Contexts
# =============================================================================

# Admin environment with extra capabilities
admin_env = TemplateEnvironment(strict=True)
admin_env.globals["user_level"] = "admin"
admin_env.register_filter("redact", lambda s: "[REDACTED]")

# User environment with different defaults
user_env = TemplateEnvironment(strict=False)  # Lenient for user-facing
user_env.globals["user_level"] = "standard"
user_env.register_filter("redact", lambda s: "***")  # Less scary redaction

admin_template = admin_env.from_string("Data: {{ data | redact }} ({{ user_level }})")
user_template = user_env.from_string("Data: {{ data | redact }} ({{ user_level }})")

print("\nDifferent environments:")
print(f"  Admin: {admin_template(data='secret123')}")
print(f"  User:  {user_template(data='secret123')}")

# =============================================================================
# Variable Precedence
# =============================================================================

# When the same variable exists in multiple places:
# 1. Render-time variables (highest priority)
# 2. Partial variables
# 3. Environment globals (lowest priority)

env = TemplateEnvironment()
env.globals["name"] = "Default Name"

t = env.from_string("Hello {{ name }}!")

print("\nVariable precedence:")
print(f"  Environment default: {t()}")
print(f"  Render override:     {t(name='Override')}")

# With partial:
partial_t = t.partial(name="Partial Name")
print(f"  Partial:             {partial_t()}")
print(f"  Partial+render:      {partial_t(name='Final Override')}")

# =============================================================================
# Loading Templates from Files
# =============================================================================

# Environment can load templates from files with shared config
# env.from_file("prompts/greeting.j2")
# env.from_file("prompts/rag_context.j2")

# All loaded templates have access to environment filters and globals

# =============================================================================
# Real-World Pattern: Application Setup
# =============================================================================

print("\n--- Real-World Pattern ---")


def setup_app_templates():
    """Setup templates for an application at startup."""
    env = TemplateEnvironment(strict=True)

    # App metadata
    env.globals["app_name"] = "AI Assistant"
    env.globals["model_version"] = "v3"
    env.globals["max_context_length"] = 4096

    # Common filters
    env.register_filter("truncate", lambda s, n=100: s[:n] + "..." if len(s) > n else s)
    env.register_filter("quote", lambda s: f'"{s}"')
    env.register_filter("bullet", lambda items: "\n".join(f"- {item}" for item in items))

    return env


# At app startup
env = setup_app_templates()

# Throughout the codebase, create templates easily
rag_template = env.from_string("""
{{ app_name }} ({{ model_version }})

Context:
{{ documents | bullet }}

Question: {{ question | quote }}
""".strip())

result = rag_template(
    documents=["Paris is in France.", "The Eiffel Tower is in Paris."],
    question="Where is the Eiffel Tower?",
)
print(result)

# =============================================================================
# Filter Precedence (Environment vs Instance)
# =============================================================================

# Instance filters override environment filters:
env = TemplateEnvironment()
env.register_filter("fmt", lambda x: f"env({x})")

t = env.from_string("{{ x | fmt }}")
print(f"\nEnvironment filter: {t(x='test')}")  # env()

t.register_filter("fmt", lambda x: f"instance({x})")
print(f"Instance override:  {t(x='test')}")  # instance() wins

"""
Topics covered:
* template.render
* template.control.flow
"""
