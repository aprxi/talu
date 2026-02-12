"""
Template module configuration.

Provides runtime configuration for template behavior. Settings can be
modified programmatically without environment variables.

Example:
    >>> from talu.template import config
    >>> config.debug = True  # Enable debug output for all templates
"""

from ..exceptions import ValidationError


class _TemplateConfig:
    """
    Singleton configuration for template module settings.

    This is a singleton - import and modify `config` directly:

        from talu.template import config
        config.debug = True

    Attributes
    ----------
        debug: When True, enables verbose output during template operations.
            Useful for debugging template rendering issues.
    """

    __slots__ = ("_debug",)

    def __init__(self) -> None:
        self._debug = False

    @property
    def debug(self) -> bool:
        """Enable debug output for template operations."""
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValidationError(
                f"debug must be bool, got {type(value).__name__}",
                code="INVALID_ARGUMENT",
                details={"param": "debug", "type": type(value).__name__},
            )
        self._debug = value

    def __repr__(self) -> str:
        return f"TemplateConfig(debug={self._debug})"


# Module-level singleton
config = _TemplateConfig()
