"""Schema conversion for structured output (Python types to JSON Schema)."""

from .convert import (
    AmbiguousUnionWarning as AmbiguousUnionWarning,
)
from .convert import (
    dataclass_to_schema as dataclass_to_schema,
)
from .convert import (
    normalize_response_format as normalize_response_format,
)

__all__: list[str] = []
