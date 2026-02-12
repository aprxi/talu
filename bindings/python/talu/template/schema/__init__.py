"""Schema-to-prompt generation for structured output.

Converts JSON Schema into model-appropriate prompt text (TypeScript,
JSON Schema dump, or XML) that instructs the model how to respond.
"""

from .generators import (
    JsonSchemaGenerator as JsonSchemaGenerator,
)
from .generators import (
    PromptGenerator as PromptGenerator,
)
from .generators import (
    TypeScriptGenerator as TypeScriptGenerator,
)
from .generators import (
    XmlGenerator as XmlGenerator,
)
from .injection import (
    get_generator as get_generator,
)
from .injection import (
    schema_to_prompt_description as schema_to_prompt_description,
)

__all__: list[str] = []
