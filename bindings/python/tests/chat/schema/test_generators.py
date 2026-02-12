"""Tests for prompt generation strategy pattern."""

from typing import Literal

import pytest
from pydantic import BaseModel

from talu.template.schema.generators import (
    JsonSchemaGenerator,
    TypeScriptGenerator,
    XmlGenerator,
)
from talu.template.schema.injection import get_generator


class Weather(BaseModel):
    """Weather response."""

    location: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]


class TestTypeScriptGenerator:
    """Test default TypeScript interface generation."""

    def test_basic_output(self):
        gen = TypeScriptGenerator()
        schema = Weather.model_json_schema()
        result = gen.generate(schema)

        assert "```typescript" in result
        assert "interface Response {" in result
        assert "location: string;" in result
        assert '"celsius" | "fahrenheit"' in result

    def test_with_thinking_instructions(self):
        gen = TypeScriptGenerator()
        schema = Weather.model_json_schema()
        result = gen.generate(schema, allow_thinking=True)

        assert "<think>" in result
        assert "Then output your response as JSON" in result


class TestJsonSchemaGenerator:
    """Test JSON Schema dump generation for older models."""

    def test_basic_output(self):
        gen = JsonSchemaGenerator()
        schema = Weather.model_json_schema()
        result = gen.generate(schema)

        assert "```json" in result
        assert '"type": "object"' in result
        assert '"properties"' in result

    def test_human_readable_intro(self):
        gen = JsonSchemaGenerator()
        schema = Weather.model_json_schema()
        result = gen.generate(schema)

        assert "Respond with JSON matching this schema:" in result


class TestXmlGenerator:
    """Test XML schema generation for Anthropic-style models."""

    def test_basic_output(self):
        gen = XmlGenerator()
        schema = Weather.model_json_schema()
        result = gen.generate(schema)

        assert "<schema>" in result or "<tool_def>" in result
        assert "location" in result
        assert "</schema>" in result or "</tool_def>" in result


class TestStrategySelection:
    """Test strategy pattern selection."""

    def test_default_is_typescript(self):
        gen = get_generator(None)
        assert isinstance(gen, TypeScriptGenerator)

    def test_string_selection(self):
        assert isinstance(get_generator("typescript"), TypeScriptGenerator)
        assert isinstance(get_generator("json_schema"), JsonSchemaGenerator)
        assert isinstance(get_generator("xml_schema"), XmlGenerator)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            get_generator("invalid_strategy")


class TestAutoStrategy:
    """Test auto-strategy selection based on model architecture."""

    def test_auto_selects_typescript_for_qwen(self):
        gen = get_generator("auto", model_name="Foo/Bar-0B")
        assert isinstance(gen, TypeScriptGenerator)

    def test_auto_selects_typescript_for_llama(self):
        gen = get_generator("auto", model_name="meta-llama/Llama-3-8B")
        assert isinstance(gen, TypeScriptGenerator)

    def test_auto_selects_json_schema_for_older_models(self):
        gen = get_generator("auto", model_name="gpt-3.5-turbo")
        assert isinstance(gen, JsonSchemaGenerator)

    def test_auto_defaults_to_typescript_for_unknown(self):
        gen = get_generator("auto", model_name="some-unknown-model")
        assert isinstance(gen, TypeScriptGenerator)

    def test_none_strategy_uses_auto(self):
        gen = get_generator(None, model_name="Foo/Bar-0B")
        assert isinstance(gen, TypeScriptGenerator)

    def test_explicit_strategy_overrides_auto(self):
        gen = get_generator("json_schema", model_name="Foo/Bar-0B")
        assert isinstance(gen, JsonSchemaGenerator)


class TestAutoStrategyWithModelType:
    """Test auto-strategy selection using model_type from config.json."""

    def test_model_type_takes_precedence_over_name(self):
        gen = get_generator(
            "auto",
            model_name="Foo/Bar-0B",
            model_type="gpt2",
        )
        assert isinstance(gen, JsonSchemaGenerator)

    def test_model_type_qwen2(self):
        gen = get_generator("auto", model_type="qwen2")
        assert isinstance(gen, TypeScriptGenerator)

    def test_model_type_llama(self):
        gen = get_generator("auto", model_type="llama")
        assert isinstance(gen, TypeScriptGenerator)

    def test_model_type_gpt2(self):
        gen = get_generator("auto", model_type="gpt2")
        assert isinstance(gen, JsonSchemaGenerator)

    def test_model_type_bloom(self):
        gen = get_generator("auto", model_type="bloom")
        assert isinstance(gen, JsonSchemaGenerator)

    def test_unknown_model_type_falls_back_to_name(self):
        gen = get_generator(
            "auto",
            model_name="qwen/some-model",
            model_type="unknown_architecture",
        )
        assert isinstance(gen, TypeScriptGenerator)

    def test_renamed_model_still_works(self):
        gen = get_generator(
            "auto",
            model_name="./my-finetune-v2",
            model_type="qwen2",
        )
        assert isinstance(gen, TypeScriptGenerator)

    def test_gguf_filename_works(self):
        gen = get_generator(
            "auto",
            model_name="model-gaf4_64.gguf",
            model_type="mistral",
        )
        assert isinstance(gen, TypeScriptGenerator)
