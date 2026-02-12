"""Schema injection with strategy pattern for prompt generation."""

from __future__ import annotations

from typing import Any, Literal

from ...exceptions import ValidationError
from .generators import (
    JsonSchemaGenerator,
    PromptGenerator,
    TypeScriptGenerator,
    XmlGenerator,
)

PromptStrategy = Literal["auto", "typescript", "json_schema", "xml_schema"]

_GENERATORS: dict[str, type[PromptGenerator]] = {
    "typescript": TypeScriptGenerator,
    "json_schema": JsonSchemaGenerator,
    "xml_schema": XmlGenerator,
}

_MODEL_TYPE_STRATEGY_MAP: dict[str, PromptStrategy] = {
    "qwen2": "typescript",
    "qwen3": "typescript",
    "deepseek_v2": "typescript",
    "starcoder": "typescript",
    "llama": "typescript",
    "mistral": "typescript",
    "gemma": "typescript",
    "gemma2": "typescript",
    "phi": "typescript",
    "phi3": "typescript",
    "gpt2": "json_schema",
    "bloom": "json_schema",
}

_MODEL_NAME_STRATEGY_MAP: dict[str, PromptStrategy] = {
    "qwen": "typescript",
    "deepseek": "typescript",
    "codellama": "typescript",
    "llama": "typescript",
    "mistral": "typescript",
    "gemma": "typescript",
    "phi": "typescript",
    "gpt-3.5": "json_schema",
}


def auto_select_strategy(
    model_name: str | None = None,
    model_type: str | None = None,
) -> PromptStrategy:
    """Auto-select prompt strategy based on model architecture."""
    if model_type is not None:
        normalized_type = model_type.lower()
        if normalized_type in _MODEL_TYPE_STRATEGY_MAP:
            return _MODEL_TYPE_STRATEGY_MAP[normalized_type]

    if model_name is not None:
        normalized_name = model_name.lower()
        for pattern, strategy in _MODEL_NAME_STRATEGY_MAP.items():
            if pattern in normalized_name:
                return strategy

    return "typescript"


def get_generator(
    strategy: PromptStrategy | None,
    model_name: str | None = None,
    model_type: str | None = None,
) -> PromptGenerator:
    """Get a prompt generator by strategy name."""
    if strategy is None or strategy == "auto":
        strategy = auto_select_strategy(model_name=model_name, model_type=model_type)

    if strategy not in _GENERATORS:
        raise ValidationError(
            f"Unknown prompt strategy: {strategy}. Valid options: {list(_GENERATORS.keys())}",
            code="INVALID_ARGUMENT",
            details={"param": "strategy", "value": strategy, "allowed": list(_GENERATORS.keys())},
        )

    return _GENERATORS[strategy]()


def schema_to_prompt_description(
    schema: dict[str, Any],
    allow_thinking: bool = False,
    strategy: PromptStrategy | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
) -> str:
    """Convert JSON Schema to prompt description using the specified strategy."""
    generator = get_generator(strategy, model_name=model_name, model_type=model_type)
    return generator.generate(schema, allow_thinking=allow_thinking)
