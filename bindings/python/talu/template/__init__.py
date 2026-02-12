"""
User-defined prompt template engine for LLM applications.

Provides the PromptTemplate class for creating reusable prompt templates with
Jinja2 syntax. Templates compile once and render quickly with different
variables.

This module is for *user-defined* templates (few-shot, RAG, custom formats).
For applying a model's built-in chat template from ``tokenizer_config.json``,
see ``talu.tokenizer.apply_chat_template`` instead.

Common Use Cases
----------------

**Few-shot learning**::

    template = PromptTemplate('''
    {% for example in examples %}
    Input: {{example.input}}
    Output: {{example.output}}
    {% endfor %}
    Input: {{query}}
    Output:''')

**RAG (Retrieval-Augmented Generation)**::

    template = PromptTemplate('''
    Context:
    {% for doc in documents %}
    [{{doc.source}}]: {{doc.content}}
    {% endfor %}

    Question: {{question}}
    Answer:''')

**Chat Formatting**::

    template = PromptTemplate.from_preset("chatml")
    prompt = template.apply([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ])

Quick Start
-----------

Create and use a template::

    from talu import PromptTemplate

    # Create template
    t = PromptTemplate("Hello {{ name }}!")

    # Render (three equivalent ways)
    t(name="World")           # Callable (recommended)
    t.format(name="World")    # Like str.format()
    t.render(name="World")    # Like Jinja2

See Also
--------
talu.Chat : Use templates with LLM chat.
"""

from .config import config as config
from .environment import TemplateEnvironment
from .prompt import PromptTemplate
from .results import DebugResult, DebugSpan, TemplateValueError, ValidationResult

# =============================================================================
# Public API - See talu/__init__.py for documentation mapping guidelines
# =============================================================================
__all__ = [
    # Core
    "PromptTemplate",
    "TemplateEnvironment",
    # Errors
    "TemplateValueError",
    # Results
    "DebugResult",
    "DebugSpan",
    "ValidationResult",
]
