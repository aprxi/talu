"""
Template environment for shared configuration.

Provides TemplateEnvironment class for managing shared filters and globals
across multiple templates.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import ValidationError

if TYPE_CHECKING:
    from .prompt import PromptTemplate


class TemplateEnvironment:
    """
    Shared configuration for a group of templates.

    Use an environment when you have multiple templates that share:

    - **Custom filters**: Date formatting, text processing, domain logic
    - **Global variables**: App name, version, feature flags, constants
    - **Default settings**: Strict mode, undefined behavior

    This eliminates boilerplate when you have many templates that need
    the same configuration.

    Parameters
    ----------
        strict: Default strict mode for templates. When True (default),
            undefined variables raise errors instead of rendering as empty.
            This prevents silent failures in LLM prompts.

    Example:
        Set up an environment for your application::

            >>> from talu import TemplateEnvironment
            >>> env = TemplateEnvironment()  # strict=True by default
            >>> env.globals["app_name"] = "MyAssistant"
            >>> env.globals["version"] = "2.0"
            >>> env.register_filter("currency", lambda x: f"${x:,.2f}")

        Create templates from the environment::

            >>> greeting = env.from_string("Welcome to {{ app_name }} v{{ version }}!")
            >>> greeting()
            'Welcome to MyAssistant v2.0!'

            >>> invoice = env.from_string("Total: {{ amount | currency }}")
            >>> invoice(amount=99.50)
            'Total: $99.50'

        Load templates from files::

            >>> prompt = env.from_file("prompts/rag_context.j2")

    See Also
    --------
        PromptTemplate : For standalone templates without shared config.
    """

    def __init__(self, *, strict: bool = True):
        self._strict = strict
        self._filters: dict[str, Callable] = {}
        self._globals: dict[str, Any] = {}

    @property
    def strict(self) -> bool:
        """Default strict mode for templates created from this environment."""
        return self._strict

    @property
    def filters(self) -> dict[str, Callable]:
        """
        Custom filters for this environment.

        Modify directly to add/remove filters, or use ``register_filter()``.

        Example:
            >>> env.filters["shout"] = lambda s: s.upper() + "!"
        """
        return self._filters

    @property
    def globals(self) -> dict[str, Any]:
        """
        Global variables available to all templates in this environment.

        These are merged with render-time variables, with render-time
        variables taking precedence.

        Example:
            >>> env.globals["app_name"] = "MyApp"
            >>> env.globals["debug"] = True
        """
        return self._globals

    def register_filter(self, name: str, func: Callable) -> TemplateEnvironment:
        """
        Register a filter for this environment.

        Args:
            name: Filter name to use in templates.
            func: Filter function.

        Returns
        -------
            Self, for method chaining.

        Raises
        ------
            ValidationError: If func is not callable.

        Example:
            >>> env.register_filter("upper", str.upper).register_filter("trim", str.strip)
        """
        if not callable(func):
            raise ValidationError(
                f"Filter must be callable, got {type(func).__name__}",
                code="INVALID_ARGUMENT",
                details={"param": "func", "type": type(func).__name__},
            )
        self._filters[name] = func
        return self

    def from_string(self, source: str, *, strict: bool | None = None) -> PromptTemplate:
        """
        Create a template from a string with this environment's configuration.

        Args:
            source: Template source string (Jinja2 syntax).
            strict: Override the environment's default strict mode.

        Returns
        -------
            A PromptTemplate with this environment's filters and globals.

        Example:
            >>> t = env.from_string("Hello {{ name | upper }}!")
            >>> t(name="world")
            'Hello WORLD!'
        """
        # Import here to avoid circular import
        from .prompt import PromptTemplate

        use_strict = self._strict if strict is None else strict
        t = PromptTemplate(source, strict=use_strict)

        # Register environment filters
        for name, func in self._filters.items():
            t.register_filter(name, func)

        # Store environment globals reference
        t._env_globals = self._globals

        return t

    def from_file(self, path: str | Path, *, strict: bool | None = None) -> PromptTemplate:
        """
        Load a template from a file with this environment's configuration.

        Args:
            path: Path to template file.
            strict: Override the environment's default strict mode.

        Returns
        -------
            A PromptTemplate with this environment's filters and globals.

        Raises
        ------
            FileNotFoundError: If the template file doesn't exist.

        Example:
            >>> t = env.from_file("prompts/greeting.j2")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")
        return self.from_string(path.read_text(encoding="utf-8"), strict=strict)
