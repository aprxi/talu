"""
Prompt templates with Jinja2 syntax for LLM workflows.

Provides the PromptTemplate class for Jinja2-compatible prompt templating with
introspection, partial application, and strict validation modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, overload

from ..exceptions import (
    TemplateError,
    TemplateSyntaxError,
    TemplateUndefinedError,
    ValidationError,
)
from . import _bindings as _c
from .loaders import get_chat_template_source, resolve_model_path
from .presets import PRESETS
from .results import DebugResult, DebugSpan, TemplateValueError, ValidationResult


class PromptTemplate:
    r"""
    Prompt template with Jinja2 syntax for AI/LLM workflows.

    A PromptTemplate compiles once and can be rendered many times with different
    variables. Supports standard Jinja2 syntax including control flow,
    filters, and functions - enabling seamless migration from existing
    Jinja2 templates.

    Features for Prompt Engineering
    -------------------------------

    **Introspection** - Discover required variables without running:

        >>> t = PromptTemplate("Hello {{ name }}, you are {{ age }} years old")
        >>> t.input_variables
        {'name', 'age'}

    **Partial Application** - Pre-fill variables for pipelines:

        >>> system = PromptTemplate("System: {{ persona }}\\nUser: {{ query }}")
        >>> chat = system.partial(persona="You are a helpful assistant")
        >>> chat(query="Hello!")  # Only need to provide query
        'System: You are a helpful assistant\nUser: Hello!'

    **Custom Filters** - Register Python functions as template filters:

        >>> t = PromptTemplate("{{ name | shout }}")
        >>> t.register_filter("shout", lambda s: s.upper() + "!!!")
        >>> t(name="hello")
        'HELLO!!!'

    Chat Formatting
    ---------------

    PromptTemplate includes presets for common chat formats:

        >>> t = PromptTemplate.chatml()
        >>> prompt = t.apply([
        ...     {"role": "user", "content": "Hello!"},
        ... ])

    Available presets: ``chatml()``, ``llama2()``, ``alpaca()``, ``vicuna()``, ``zephyr()``

    Basic Usage
    -----------

        >>> from talu import PromptTemplate

        # Create template
        >>> t = PromptTemplate("Hello {{ name }}!")

        # Render with variables (three equivalent ways)
        >>> t(name="World")           # Callable (recommended)
        'Hello World!'
        >>> t.format(name="World")    # Like str.format()
        'Hello World!'
        >>> t.render(name="World")    # Like Jinja2
        'Hello World!'

        # RAG example
        >>> rag = PromptTemplate('''
        ... Context:
        ... {% for doc in docs %}
        ... - {{ doc.content }}
        ... {% endfor %}
        ...
        ... Question: {{ question }}
        ... ''')
        >>> rag(docs=[{"content": "Paris is in France."}],
        ...      question="Where is Paris?")

    Supported Jinja2 Features
    -------------------------

    **Fully Supported:**
        - Variables: ``{{ name }}``, ``{{ user.name }}``
        - Control flow: ``{% if %}``, ``{% for %}``, ``{% set %}``, ``{% macro %}``
        - Filters: ``| upper``, ``| lower``, ``| join``, ``| default``, etc.
        - Operators: ``+``, ``-``, ``*``, ``/``, ``in``, ``not in``, ``is``, ``and``, ``or``
        - Functions: ``range()``, ``dict()``, ``namespace()``
        - Comments: ``{# comment #}``
        - Whitespace control: ``{{- name -}}``, ``{%- if -%}``
        - Custom Python filters via ``register_filter()``
        - Template composition via ``{% include %}`` (see below)

    **Template Composition:**
        The ``{% include %}`` tag allows including other templates dynamically.
        The argument is an **expression** that evaluates to a template string
        at runtime - it does **not** load from the filesystem:

            >>> t = PromptTemplate("{% include header %}Body")
            >>> t(header="=== {{ title }} ===", title="My Doc")
            '=== My Doc ===Body'

        This design differs from standard Jinja2 (which loads files) for
        security reasons - templates cannot access the filesystem. Pass
        template strings as variables instead.

        Included templates have access to the parent template's context,
        including macros defined in previously included templates:

            >>> t = PromptTemplate("{% include utils %}{{ greet('World') }}")
            >>> t(utils="{% macro greet(n) %}Hello {{ n }}!{% endmacro %}")
            'Hello World!'

    **Not Supported:**
        - Template inheritance (``{% extends %}``, ``{% block %}``) - use
          ``{% include %}`` for composition instead
        - File-based includes - pass template strings as variables for security

    Args:
        source: The template string with Jinja2 syntax.

    Raises
    ------
        TemplateSyntaxError: If the template has invalid syntax.
    """

    __slots__ = (
        "_source",
        "_strict",
        "_partial_vars",
        "_cached_input_vars",
        "_custom_filters",
        "_filter_callbacks",
        "_env_globals",
    )

    def __init__(self, source: str, *, strict: bool = True):
        """
        Create a new template.

        Args:
            source: Template string with Jinja2 syntax.
            strict: If True (default), undefined variables raise TemplateUndefinedError.
                If False, undefined variables render as empty string (Jinja2 default).

        Raises
        ------
            TemplateSyntaxError: If template syntax is invalid.

        Note:
            Unlike standard Jinja2, Talu defaults to strict mode to catch missing
            variables early. Use ``{{ var | default('') }}`` for optional variables.
        """
        if not isinstance(source, str):
            raise ValidationError(
                f"Template source must be str, got {type(source).__name__}",
                code="INVALID_ARGUMENT",
                details={"param": "source", "type": type(source).__name__},
            )

        self._source = source
        self._strict = strict
        self._partial_vars: dict[str, Any] = {}
        self._cached_input_vars: set[str] | None = None
        # Custom filters: name -> (python_func, ctypes_callback)
        # We keep both to prevent garbage collection of the callback
        self._custom_filters: dict[str, tuple[Any, Any]] = {}
        # Keep references to callback objects to prevent GC
        self._filter_callbacks: list[Any] = []
        # Environment globals (set by TemplateEnvironment.from_string())
        self._env_globals: dict[str, Any] = {}
        self._validate_syntax()

    @classmethod
    def from_file(cls, path: str) -> PromptTemplate:
        """
        Load a template from a file.

        Args:
            path: Path to the template file.

        Returns
        -------
            A new PromptTemplate instance.

        Raises
        ------
            FileNotFoundError: If the file doesn't exist.
            TemplateSyntaxError: If template syntax is invalid.

        Example:
            >>> template = PromptTemplate.from_file("prompts/rag.j2")
            >>> result = template(documents=docs, question="...")
        """
        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        return cls(template_path.read_text())

    @classmethod
    def from_chat_template(cls, model: str, *, strict: bool = False) -> PromptTemplate:
        """
        Load a model's chat template as an inspectable PromptTemplate.

        Args:
            model: Model path or HuggingFace ID (e.g., "Qwen/Qwen3-0.6B").
            strict: If True, undefined variables raise errors.

        Returns
        -------
            PromptTemplate with the model's chat template.

        Raises
        ------
            FileNotFoundError: If model not found or has no chat template.

        Example:
            >>> t = PromptTemplate.from_chat_template("Qwen/Qwen3-0.6B")
            >>> t.input_variables
            {'messages', 'add_generation_prompt', ...}
        """
        # Resolve model path (handles local paths and HF IDs)
        model_path = resolve_model_path(model)

        # Get the chat template source
        source = get_chat_template_source(model_path)

        return cls(source, strict=strict)

    def _validate_syntax(self) -> None:
        """Validate template syntax on construction."""
        # Basic Python-side validation for quick errors
        # Check for unclosed tags (simple heuristic)
        if "{{" in self._source and "}}" not in self._source:
            raise TemplateSyntaxError("Syntax error: unclosed variable tag, expected }}")
        if "{%" in self._source and "%}" not in self._source:
            raise TemplateSyntaxError("Syntax error: unclosed block tag, expected %}")

        # Count opening/closing tags, accounting for delimiters inside strings
        # We can't do a simple count because }} or %} might appear inside string literals
        # Instead, skip this check if the template looks complex (has quotes)
        # and let the Zig compiler do proper validation
        has_string_literals = '"' in self._source or "'" in self._source
        if not has_string_literals:
            opens = self._source.count("{{") + self._source.count("{%")
            closes = self._source.count("}}") + self._source.count("%}")

            if opens != closes:
                raise TemplateSyntaxError(
                    f"Syntax error: unclosed tag, {opens} opening vs {closes} closing"
                )

        # Validate via Zig compiler
        self._compile_check()

    def _compile_check(self) -> None:
        """Validate syntax by doing a test parse with empty context."""
        from .._bindings import check

        code = _c.template_compile_check(self._source)

        if code != 0:
            # Re-raise syntax errors, but allow undefined variable errors
            # (they're expected when compiling without variables)
            try:
                check(code)
            except TemplateSyntaxError:
                raise  # Syntax errors should fail compilation
            except TemplateUndefinedError:
                pass  # Undefined vars are OK at compile time (no vars provided)
            except TemplateError:
                pass  # Other template errors may be OK at compile time

    # =========================================================================
    # Introspection
    # =========================================================================

    @property
    def input_variables(self) -> set[str]:
        """
        Return the set of variable names required by this template.

        This enables building tools on top of templates (CLIs, web UIs, validators)
        that need to know what inputs are expected without executing the template.

        Returns
        -------
            Set of variable names found in the template source.

        Note:
            This uses **AST-based extraction** via the Zig template engine,
            which accurately detects all required variables including:

            - Simple variables: ``{{ name }}``
            - Attribute access: ``{{ user.name }}`` (extracts ``user``)
            - Array indexing: ``{{ items[0] }}`` (extracts ``items``)
            - Dictionary access: ``{{ data['key'] }}`` (extracts ``data``)
            - Complex expressions: ``{{ a if b else c }}``
            - Nested in filters: ``{{ name | default(fallback) }}``
            - In conditions: ``{% if show %}``
            - In loops (the iterable): ``{% for x in items %}``

            **Automatically excluded:**
                - Loop iteration variables: ``x`` in ``{% for x in items %}``
                - Set variables: ``x`` in ``{% set x = ... %}``
                - Macro parameters
                - Built-ins: ``loop``, ``true``, ``false``, ``none``, ``range``, etc.

        Example:
            >>> t = PromptTemplate("Hello {{ name }}, you are {{ age }} years old")
            >>> t.input_variables
            {'name', 'age'}

            >>> rag = PromptTemplate('''
            ... {% for doc in docs %}
            ... {{ doc.content }}
            ... {% endfor %}
            ... Question: {{ question }}
            ... ''')
            >>> rag.input_variables
            {'docs', 'question'}
        """
        if self._cached_input_vars is not None:
            return self._cached_input_vars

        # Use Zig AST-based extraction via validation dry-run.
        # When we validate with no variables, all required variables
        # appear in the "missing" set - this is 100% accurate because
        # it uses the actual parser/AST, not regex.

        # Only pass partial vars (if any) - everything else will be "missing"
        json_vars = json.dumps(self._partial_vars)

        code, result_json = _c.template_validate_raw(self._source, json_vars)

        if code != 0:
            # On error, fall back to empty set (syntax errors handled elsewhere)
            self._cached_input_vars = set()
            return self._cached_input_vars

        if result_json:
            result_data = json.loads(result_json)
            # Combine required and optional - both are "input variables"
            # required = variables used without default()
            # optional = variables used with default()
            required = set(result_data.get("required", []))
            optional = set(result_data.get("optional", []))
            self._cached_input_vars = required | optional
        else:
            self._cached_input_vars = set()

        return self._cached_input_vars

    # =========================================================================
    # Partial Application
    # =========================================================================

    def partial(self, **kwargs: Any) -> PromptTemplate:
        r"""
        Return a new PromptTemplate with some variables pre-filled.

        Args:
            **kwargs: Variables to pre-fill in the new template.

        Returns
        -------
            A new PromptTemplate with the variables baked in.

        Example:
            >>> t = PromptTemplate("{{ persona }}\\n{{ query }}")
            >>> chat = t.partial(persona="You are helpful")
            >>> chat(query="Hello!")
            'You are helpful\nHello!'
        """
        # Validate that all values are JSON-serializable
        self._validate_json_serializable(kwargs)

        # Create new template with same source and settings
        new_template = PromptTemplate.__new__(PromptTemplate)
        new_template._source = self._source
        new_template._strict = self._strict
        new_template._cached_input_vars = None  # Reset cache

        # Merge partial vars (existing + new)
        new_template._partial_vars = {**self._partial_vars, **kwargs}

        # Copy custom filters
        new_template._custom_filters = self._custom_filters.copy()
        new_template._filter_callbacks = self._filter_callbacks.copy()

        # Copy environment globals (reference, not copy - changes to env affect all templates)
        new_template._env_globals = self._env_globals

        return new_template

    # =========================================================================
    # Custom Filters
    # =========================================================================

    def register_filter(self, name: str, func: Any) -> PromptTemplate:
        """
        Register a custom Python filter function.

        Args:
            name: Filter name to use in templates (e.g., ``{{ x | name }}``).
            func: Callable that takes the piped value and returns a result.

        Returns
        -------
            Self, for method chaining.

        Example:
            >>> t = PromptTemplate("{{ name | shout }}")
            >>> t.register_filter("shout", lambda s: s.upper() + "!")
            >>> t(name="hello")
            'HELLO!'
        """
        if not callable(func):
            raise ValidationError(
                f"Filter function must be callable, got {type(func).__name__}",
                code="INVALID_ARGUMENT",
                details={"param": "func", "type": type(func).__name__},
            )

        # Create a wrapper that handles JSON serialization/deserialization
        def make_callback(py_func: Any) -> Any:
            """Create a ctypes callback that wraps the Python function."""

            def callback(
                value_json: bytes | None,
                args_json: bytes | None,
                user_data: Any,
            ) -> int | None:
                try:
                    # Deserialize input value
                    if value_json:
                        value = json.loads(value_json.decode("utf-8"))
                    else:
                        value = None

                    # Deserialize arguments
                    if args_json:
                        args = json.loads(args_json.decode("utf-8"))
                        if not isinstance(args, list):
                            args = []
                    else:
                        args = []

                    # Call the Python function
                    result = py_func(value, *args)

                    # Serialize result back to JSON
                    result_str = json.dumps(result)
                    result_bytes = result_str.encode("utf-8")

                    # Allocate C memory for the result (Zig will free with c_allocator)
                    return _c.alloc_string_for_callback(result_bytes)
                except (json.JSONDecodeError, TypeError, ValueError, OSError, MemoryError):
                    # Return None (NULL) to signal error to Zig
                    return None

            return _c.CustomFilterFunc(callback)

        # Create the callback and store both to prevent garbage collection
        ctypes_callback = make_callback(func)
        self._custom_filters[name] = (func, ctypes_callback)
        self._filter_callbacks.append(ctypes_callback)

        return self

    def _validate_json_serializable(self, values: dict[str, Any]) -> None:
        """
        Validate that all values can be serialized to JSON.

        Args:
            values: Dictionary of variable values to validate.

        Raises
        ------
            TemplateValueError: If any value cannot be serialized to JSON.
        """
        for key, value in values.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError) as e:
                raise TemplateValueError(
                    f"Variable '{key}' cannot be serialized to JSON: {e}\n"
                    f"Value type: {type(value).__name__}\n"
                    f"Templates only support JSON-compatible types: "
                    f"str, int, float, bool, None, list, dict"
                ) from e

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, **kwargs: Any) -> ValidationResult:
        """
        Validate inputs and prepare for efficient rendering.

        Args:
            **kwargs: Variables to validate against template requirements.

        Returns
        -------
            ValidationResult with ``is_valid``, ``required``, ``optional``, ``extra``.
            Use ``result.render()`` to render without re-serializing variables.

        Example:
            >>> t = PromptTemplate("Hello {{ name }}, age {{ age }}")
            >>> result = t.validate(name="Alice")
            >>> result.is_valid
            False
            >>> result.required
            {'age'}
        """
        # Check for non-JSON-serializable values first
        invalid: dict[str, str] = {}
        serializable_vars: dict[str, Any] = {}

        for key, value in kwargs.items():
            try:
                json.dumps(value)
                serializable_vars[key] = value
            except (TypeError, ValueError) as e:
                invalid[key] = str(e)

        # Merge: env globals < partial vars < render-time variables
        merged_vars = {**self._env_globals, **self._partial_vars, **serializable_vars}

        # Use Zig AST-based validation
        json_vars = json.dumps(merged_vars)

        result_json = _c.template_validate(self._source, json_vars)

        if result_json is None:
            # Fallback to regex-based validation if Zig validation fails
            return self._validate_regex(kwargs, invalid, json_vars)

        result_data = json.loads(result_json)

        required = set(result_data.get("required", []))
        optional = set(result_data.get("optional", []))
        extra = set(result_data.get("extra", []))

        return ValidationResult(
            required=required,
            optional=optional,
            extra=extra,
            invalid=invalid,
            _template=self,
            _json_vars=json_vars,
            _strict=self._strict,
        )

    def _validate_regex(
        self, kwargs: dict[str, Any], invalid: dict[str, str], json_vars: str
    ) -> ValidationResult:
        """Fallback regex-based validation (legacy behavior, no optional detection)."""
        # Get required variables (excluding those already partially applied)
        all_vars = self.input_variables
        provided = set(kwargs.keys())

        # Find missing variables (all considered required in fallback mode)
        required = all_vars - provided

        # Find extra variables (not required by template)
        # Include partial vars in the "known" set
        known_vars = all_vars | set(self._partial_vars.keys())
        extra = provided - known_vars

        # Fallback can't distinguish optional vs required, but still supports render()
        return ValidationResult(
            required=required,
            optional=set(),
            extra=extra,
            invalid=invalid,
            _template=self,
            _json_vars=json_vars,
            _strict=self._strict,
        )

    # =========================================================================
    # Rendering
    # =========================================================================

    @overload
    def __call__(
        self, *, debug: Literal[False] = False, strict: bool | None = None, **variables: Any
    ) -> str: ...

    @overload
    def __call__(
        self, *, debug: Literal[True], strict: bool | None = None, **variables: Any
    ) -> DebugResult: ...

    @overload
    def __call__(
        self, *, debug: bool = False, strict: bool | None = None, **variables: Any
    ) -> str | DebugResult: ...

    def __call__(
        self, *, debug: bool = False, strict: bool | None = None, **variables: Any
    ) -> str | DebugResult:
        """
        Render the template with the given variables.

        Args:
            debug: If True, return DebugResult with span tracking.
            strict: Override strict mode for this render only.
            **variables: Variables to substitute in the template.

        Returns
        -------
            Rendered string, or DebugResult if debug=True.

        Example:
            >>> t = PromptTemplate("Hello {{ name }}!")
            >>> t(name="World")
            'Hello World!'
        """
        import os

        use_strict = self._strict if strict is None else strict

        if debug or os.environ.get("TALU_DEBUG_TEMPLATES") == "1":
            return self._render_debug(variables, use_strict)
        return self._render(variables, use_strict)

    def render(self, *, strict: bool | None = None, **variables: Any) -> str:
        """
        Render the template with the given variables.

        Familiar API for Jinja2 users.

        Args:
            strict: Override the instance's strict mode for this render only.
                If None (default), uses the instance's strict setting.
            **variables: Variables to substitute in the template.

        Returns
        -------
            The rendered template string.

        Example:
            >>> t = PromptTemplate("Hello {{ name }}!")
            >>> t.render(name="World")
            'Hello World!'
        """
        use_strict = self._strict if strict is None else strict
        return self._render(variables, use_strict)

    def format(self, **variables: Any) -> str:
        """
        Render the template with the given variables.

        Familiar API for str.format() users.

        Args:
            **variables: Variables to substitute in the template.

        Returns
        -------
            The rendered template string.

        Example:
            >>> t = PromptTemplate("Hello {{ name }}!")
            >>> t.format(name="World")
            'Hello World!'
        """
        return self._render(variables, self._strict)

    def _render(self, variables: dict[str, Any], strict: bool) -> str:
        """Render the template with the given variables dictionary."""
        # Validate JSON serializability of input variables
        self._validate_json_serializable(variables)

        # Merge: env globals < partial vars < render-time variables
        # Later values take precedence over earlier
        merged_vars = {**self._env_globals, **self._partial_vars, **variables}

        # Convert merged variables to JSON
        json_vars = json.dumps(merged_vars)

        # Use custom filters API if any filters are registered
        if self._custom_filters:
            text = _c.template_render_with_filters(
                self._source, json_vars, strict, self._custom_filters
            )
        else:
            text = _c.template_render(self._source, json_vars, strict)

        if text is None:
            raise TemplateError("Template rendering returned no result")

        return text

    def _render_from_json(self, json_vars: str, strict: bool) -> str:
        """
        Render the template with pre-serialized JSON variables.

        This is an internal optimization used by ValidationResult.render()
        to avoid re-serializing variables that were already serialized
        during validation.

        Args:
            json_vars: Pre-serialized JSON string of variables.
            strict: Whether to use strict mode.

        Returns
        -------
            The rendered template string.
        """
        # Use custom filters API if any filters are registered
        if self._custom_filters:
            text = _c.template_render_with_filters(
                self._source, json_vars, strict, self._custom_filters
            )
        else:
            text = _c.template_render(self._source, json_vars, strict)

        if text is None:
            raise TemplateError("Template rendering returned no result")

        return text

    def _render_debug(self, variables: dict[str, Any], strict: bool) -> DebugResult:
        """Render with debug span tracking."""
        # Validate JSON serializability of input variables
        self._validate_json_serializable(variables)

        # Merge: env globals < partial vars < render-time variables
        merged_vars = {**self._env_globals, **self._partial_vars, **variables}

        # Convert merged variables to JSON
        json_vars = json.dumps(merged_vars)

        output, raw_spans = _c.template_render_debug(self._source, json_vars, strict)

        if output is None:
            raise TemplateError("Template debug rendering returned no result")

        # Convert tuples to DebugSpan objects
        spans = [
            DebugSpan(start=start, end=end, source=source, text=text)
            for start, end, source, text in raw_spans
        ]

        return DebugResult(output=output, spans=spans)

    @property
    def source(self) -> str:
        """The original template source string."""
        return self._source

    @property
    def strict(self) -> bool:
        """Whether strict mode is enabled (undefined variables raise errors)."""
        return self._strict

    @property
    def supports_system_role(self) -> bool:
        """
        Check if this chat template explicitly handles the 'system' role.

        Returns True if the template source contains explicit handling for
        system messages (e.g., ``message.role == 'system'`` or ``role == "system"``).

        This helps Chat implementations decide whether to:
        - Pass system messages directly (if supported)
        - Prepend system content to the first user message (if not supported)

        Returns
        -------
            True if template handles system role, False otherwise.

        Example:
            >>> t = PromptTemplate.chatml()
            >>> t.supports_system_role  # ChatML handles any role
            True

            >>> t = PromptTemplate("{{ messages[0].content }}")
            >>> t.supports_system_role  # No role handling
            False

        Note:
            This is a heuristic based on source inspection. Templates that
            use ``{{ message.role }}`` directly (like ChatML) will return True
            because they handle any role generically.
        """
        source = self._source.lower()
        # Check for explicit system role handling
        if "'system'" in source or '"system"' in source:
            return True
        # Check for generic role handling (like ChatML's {{ message.role }})
        # Match common loop variable patterns: message.role, msg.role, m.role
        if ".role" in source:
            return True
        # Check for <<SYS>> style (Llama2)
        if "<<sys>>" in source or "<|system|>" in source:
            return True
        return False

    @property
    def supports_tools(self) -> bool:
        """
        Check if this chat template has built-in tool/function calling support.

        Returns True if the template source contains handling for tools,
        functions, or available_tools variables (common in HuggingFace templates
        for models like Llama-3, Qwen, etc.).

        This helps Chat implementations decide whether to:
        - Let the template handle tool formatting (if supported)
        - Handle tool formatting separately (if not supported)

        When a template has built-in tool support but the Chat module also
        handles tools, pass an empty tools list to the template to avoid
        conflicts.

        Returns
        -------
            True if template has tool handling, False otherwise.

        Example:
            >>> t = PromptTemplate.chatml()
            >>> t.supports_tools  # Basic ChatML has no tool handling
            False

        Note:
            This is a heuristic. Templates with tool support typically check
            for ``tools``, ``available_tools``, or ``functions`` variables.
        """
        source = self._source.lower()
        # Check for common tool-related variables
        if "tools" in source or "functions" in source:
            return True
        # Check for tool_call handling
        if "tool_call" in source:
            return True
        return False

    # =========================================================================
    # Chat Template Presets
    # =========================================================================

    @classmethod
    def from_preset(cls, name: str) -> PromptTemplate:
        r"""
        Create a PromptTemplate from a built-in chat format preset.

        Available presets: ``chatml``, ``llama2``, ``alpaca``, ``vicuna``,
        ``zephyr``.

        Args:
            name: Preset name (case-sensitive).

        Returns
        -------
            PromptTemplate configured for the specified format.

        Raises
        ------
        ValueError
            If *name* is not a known preset.

        Example:
            >>> t = PromptTemplate.from_preset("chatml")
            >>> t.apply([{"role": "user", "content": "Hi"}])
            '<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\n'
        """
        try:
            source = PRESETS[name]
        except KeyError:
            available = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset {name!r}. Available presets: {available}") from None
        return cls(source)

    # =========================================================================
    # Chat Apply Method
    # =========================================================================

    def apply(
        self,
        messages: list[dict[str, Any]],
        *,
        debug: bool = False,
        strict: bool | None = None,
        add_generation_prompt: bool = True,
        bos_token: str = "",
        eos_token: str = "",
        **kwargs: Any,
    ) -> str | DebugResult:
        """
        Render chat messages using this template.

        Convenience wrapper around ``__call__()`` with named parameters for
        common chat template variables.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            debug: If True, return DebugResult with span tracking.
            strict: Override strict mode for this render only.
            add_generation_prompt: Add assistant marker at end (default True).
            bos_token: Beginning of sequence token (model-specific).
            eos_token: End of sequence token (model-specific).
            **kwargs: Additional template variables (e.g., ``tools``).

        Returns
        -------
            Formatted prompt string, or DebugResult if debug=True.

        Example:
            >>> t = PromptTemplate.chatml()
            >>> prompt = t.apply([
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"},
            ... ])
        """
        return self(
            messages=messages,
            debug=debug,
            strict=strict,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    def __repr__(self) -> str:
        preview = self._source[:50]
        if len(self._source) > 50:
            preview += "..."
        return f"PromptTemplate({preview!r})"

    def __str__(self) -> str:
        return self._source
