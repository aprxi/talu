"""
Result types for template validation and debug rendering.

Contains dataclasses for validation results and debug span tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..exceptions import StateError, TemplateError

if TYPE_CHECKING:
    pass


class TemplateValueError(TypeError):
    """Raised when template input cannot be serialized to JSON."""

    pass


@dataclass
class ValidationResult:
    """
    Result of template input validation.

    Distinguishes between required and optional variables. Required variables
    are used "naked" in the template (``{{ name }}``), while optional variables
    have a ``default()`` filter (``{{ context | default('') }}``).

    This distinction helps when loading third-party templates via
    ``PromptTemplate.from_chat_template()`` - you know which variables will break
    the template vs which will safely fall back to defaults.

    The ValidationResult can also render the template directly, avoiding
    re-serialization of the variables that were already validated::

        result = template.validate(documents=large_docs)
        if result.is_valid:
            output = result.render()  # No re-serialization!

    Attributes
    ----------
        required: Variables required by the template (no default) but not provided.
            Missing these will likely break template logic or produce empty output.
        optional: Variables with default() filter that are not provided.
            These are safe to omit - the template handles missing values gracefully.
        extra: Variables provided but not used by the template.
            These are warnings, not errors - extra variables are ignored.
        invalid: Dictionary mapping variable names to error messages for
            values that cannot be serialized to JSON.

    Example:
        >>> t = PromptTemplate("Hello {{ name }}! Context: {{ context | default('N/A') }}")
        >>> result = t.validate(name="Alice")
        >>> result.is_valid
        True  # Only 'required' matters for validity
        >>> result.required
        set()  # name was provided
        >>> result.optional
        {'context'}  # context has default, safe to omit

        # Efficient validate-then-render pattern:
        >>> result = t.validate(name="Alice")
        >>> if result:
        ...     output = result.render()  # Reuses serialized variables
    """

    required: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    extra: set[str] = field(default_factory=set)
    invalid: dict[str, str] = field(default_factory=dict)

    # Private fields for render optimization (not part of public API)
    _template: Any = field(default=None, repr=False, compare=False)
    _json_vars: str | None = field(default=None, repr=False, compare=False)
    _strict: bool = field(default=False, repr=False, compare=False)

    @property
    def is_valid(self) -> bool:
        """
        Whether validation passed.

        Returns True only if there are no missing required variables and no
        invalid (non-JSON-serializable) values.

        Note:
            Optional and extra variables do NOT cause validation to fail.
            Only required variables and invalid values matter for validity.
        """
        return not self.required and not self.invalid

    @property
    def summary(self) -> str:
        """
        Human-readable summary of validation issues.

        Returns empty string if validation passed.

        Example:
            >>> result.summary
            'Missing required variables: name. Missing optional variables (have defaults): context'
        """
        parts = []

        if self.required:
            vars_str = ", ".join(sorted(self.required))
            parts.append(f"Missing required variables: {vars_str}")

        if self.invalid:
            invalid_str = ", ".join(f"{k} ({v})" for k, v in sorted(self.invalid.items()))
            parts.append(f"Invalid values: {invalid_str}")

        if self.optional:
            vars_str = ", ".join(sorted(self.optional))
            parts.append(f"Missing optional variables (have defaults): {vars_str}")

        if self.extra:
            vars_str = ", ".join(sorted(self.extra))
            parts.append(f"Extra variables (will be ignored): {vars_str}")

        return ". ".join(parts)

    def render(self, *, strict: bool | None = None) -> str:
        """
        Render the template using the validated variables.

        This method reuses the JSON serialization from validation, avoiding
        the overhead of re-serializing large variables like RAG documents.

        Args:
            strict: Override strict mode for this render only.
                If None (default), uses the mode from validation.

        Returns
        -------
            The rendered template string.

        Raises
        ------
            ValueError: If validation failed or result wasn't created by validate().
            TemplateError: If rendering fails.

        Example:
            Efficient validate-then-render for large documents::

                >>> docs = load_large_documents()  # 10MB of data
                >>> result = template.validate(documents=docs, question="What is...?")
                >>> if result.is_valid:
                ...     output = result.render()  # No re-serialization!

            With strict mode override::

                >>> result = template.validate(name="Alice")
                >>> output = result.render(strict=True)  # Force strict for this render
        """
        if self._template is None or self._json_vars is None:
            raise StateError(
                "ValidationResult.render() requires a result created by "
                "PromptTemplate.validate(). This result has no template context.",
                code="STATE_INVALID_CONTEXT",
            )

        if not self.is_valid:
            raise TemplateError(
                f"Cannot render invalid template. {self.summary}",
                code="TEMPLATE_VALIDATION_FAILED",
            )

        use_strict = self._strict if strict is None else strict
        return self._template._render_from_json(self._json_vars, use_strict)

    def __bool__(self) -> bool:
        """
        Allow using ValidationResult in boolean context.

        Returns True if validation passed (is_valid).

        Example:
            >>> if t.validate(name="Alice", age=30):
            ...     result = t(name="Alice", age=30)
        """
        return self.is_valid


@dataclass
class DebugSpan:
    """
    Span in rendered template output for debug visualization.

    Tracks which parts of the output came from variables vs static template text.
    This enables debugging "why did the model hallucinate?" by showing exactly
    which variable produced which part of the output.

    Attributes
    ----------
        start: Start position in output (inclusive).
        end: End position in output (exclusive).
        source: What produced this span - "static", variable path
            (e.g., "name", "user.email"), or "expression".
        text: The actual text content of this span.
    """

    start: int
    end: int
    source: str  # "static", variable path, or "expression"
    text: str

    @property
    def is_static(self) -> bool:
        """Whether this span is static template text."""
        return self.source == "static"

    @property
    def is_variable(self) -> bool:
        """Whether this span came from a variable substitution."""
        return self.source not in ("static", "expression")

    @property
    def is_expression(self) -> bool:
        """Whether this span came from a complex expression."""
        return self.source == "expression"


@dataclass
class DebugResult:
    """
    Result of rendering with debug mode enabled.

    Contains the rendered output plus detailed span information showing
    which parts came from variables vs static text.

    Attributes
    ----------
        output: The rendered template string.
        spans: List of DebugSpan showing the source of each part.
    """

    output: str
    spans: list[DebugSpan]

    def format_ansi(self) -> str:
        """
        Format output with ANSI colors for terminal display.

        - Static text: normal (no color)
        - Variables: cyan with variable name in parentheses
        - Expressions: yellow

        Note:
            Empty variable values produce no output, so they don't appear
            in the span list. Use ``PromptTemplate.validate()`` before rendering
            to detect empty or missing values.

        Returns
        -------
            Formatted string with ANSI escape codes.
        """
        result = []
        for span in self.spans:
            if span.is_static:
                result.append(span.text)
            elif span.is_expression:
                # Yellow for expressions
                result.append(f"\033[33m{span.text}\033[0m")
            else:
                # Cyan for variables, with variable name in dim grey
                result.append(f"\033[36m{span.text}\033[0m(\033[90m{span.source}\033[0m)")
        return "".join(result)

    def format_plain(self) -> str:
        """
        Format output with plain text markers (no ANSI).

        - Static text: unchanged
        - Variables: «value» (var_name)
        - Expressions: «value»

        Note:
            Empty variable values produce no output, so they don't appear
            in the span list. Use ``PromptTemplate.validate()`` before rendering
            to detect empty or missing values.

        Returns
        -------
            Formatted string with plain text markers.
        """
        result = []
        for span in self.spans:
            if span.is_static:
                result.append(span.text)
            elif span.is_expression:
                result.append(f"«{span.text}»")
            else:
                # Variable with source name in parentheses
                result.append(f"«{span.text}» ({span.source})")
        return "".join(result)

    def __str__(self) -> str:
        """Return ANSI formatted string for terminal display."""
        return self.format_ansi()
