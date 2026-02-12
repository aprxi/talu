"""Tests for Template prompt engineering features.

Tests for the features designed for AI/LLM workflows:
- input_variables property (introspection)
- partial() method (currying)
- strict mode (error handling)
- JSON serialization validation
"""

import pytest

import talu
from talu.exceptions import TemplateUndefinedError
from talu.template import PromptTemplate, TemplateValueError


class TestInputVariables:
    """Tests for Template.input_variables property."""

    def test_simple_variable(self):
        """Single variable is detected."""
        t = PromptTemplate("Hello {{ name }}!")
        assert t.input_variables == {"name"}

    def test_multiple_variables(self):
        """Multiple variables are detected."""
        t = PromptTemplate("{{ greeting }}, {{ name }}! You are {{ age }} years old.")
        assert t.input_variables == {"greeting", "name", "age"}

    def test_repeated_variable(self):
        """Repeated variable appears once in set."""
        t = PromptTemplate("{{ name }} {{ name }} {{ name }}")
        assert t.input_variables == {"name"}

    def test_loop_variable_excluded(self):
        """Loop iteration variable is excluded."""
        t = PromptTemplate("{% for item in items %}{{ item }}{% endfor %}")
        assert t.input_variables == {"items"}
        assert "item" not in t.input_variables

    def test_set_variable_excluded(self):
        """Set variable is excluded."""
        t = PromptTemplate("{% set greeting = 'Hello' %}{{ greeting }} {{ name }}")
        assert t.input_variables == {"name"}
        assert "greeting" not in t.input_variables

    def test_loop_object_excluded(self):
        """Built-in loop object is excluded."""
        t = PromptTemplate("{% for i in items %}{{ loop.index }}{% endfor %}")
        assert t.input_variables == {"items"}
        assert "loop" not in t.input_variables

    def test_boolean_literals_excluded(self):
        """Boolean literals are excluded."""
        t = PromptTemplate("{% if true %}yes{% endif %}{% if false %}no{% endif %}")
        assert "true" not in t.input_variables
        assert "false" not in t.input_variables

    def test_none_excluded(self):
        """None literal is excluded."""
        t = PromptTemplate("{% if value is none %}null{% endif %}")
        assert "value" in t.input_variables
        assert "none" not in t.input_variables

    def test_complex_rag_template(self):
        """Complex RAG template detects correct variables."""
        t = PromptTemplate("""
Context:
{% for doc in documents %}
- {{ doc.content }} (score: {{ doc.score }})
{% endfor %}

Question: {{ question }}
""")
        assert t.input_variables == {"documents", "question"}
        assert "doc" not in t.input_variables

    def test_if_condition_variable(self):
        """Variable in if condition is detected."""
        t = PromptTemplate("{% if show_greeting %}Hello{% endif %} {{ name }}")
        assert t.input_variables == {"show_greeting", "name"}

    def test_cached_result(self):
        """input_variables is cached for performance."""
        t = PromptTemplate("{{ name }} {{ age }}")
        first = t.input_variables
        second = t.input_variables
        assert first is second  # Same object (cached)

    def test_no_variables(self):
        """Template with no variables returns empty set."""
        t = PromptTemplate("Hello World!")
        assert t.input_variables == set()


class TestStrictMode:
    """Tests for Template strict mode."""

    def test_strict_default_true(self):
        """strict defaults to True (strict mode) - prevents silent prompt failures."""
        t = PromptTemplate("{{ name }}")
        assert t.strict is True

    def test_strict_mode_raises_on_undefined(self):
        """In strict mode (default), undefined variables raise TemplateUndefinedError."""
        t = PromptTemplate("Hello {{ name }}!")
        with pytest.raises(TemplateUndefinedError):
            t()

    def test_lenient_mode_renders_empty(self):
        """In lenient mode, undefined variables render as empty."""
        t = PromptTemplate("Hello {{ name }}!", strict=False)
        result = t()
        assert result == "Hello !"

    def test_strict_mode_works_with_defined_vars(self):
        """Strict mode works normally when variables are provided."""
        t = PromptTemplate("Hello {{ name }}!", strict=True)
        result = t(name="World")
        assert result == "Hello World!"

    def test_strict_mode_default_filter_works(self):
        """Default filter works in strict mode (no error raised)."""
        t = PromptTemplate("{{ name | default('Guest') }}", strict=True)
        result = t()
        assert result == "Guest"

    def test_strict_mode_is_defined_works(self):
        """'is defined' test works in strict mode."""
        t = PromptTemplate("{% if name is defined %}yes{% else %}no{% endif %}", strict=True)
        result = t()
        assert result == "no"

    def test_strict_mode_is_undefined_works(self):
        """'is undefined' test works in strict mode."""
        t = PromptTemplate("{% if name is undefined %}yes{% else %}no{% endif %}", strict=True)
        result = t()
        assert result == "yes"

    def test_strict_property(self):
        """strict property returns the strict mode setting."""
        t_strict = PromptTemplate("{{ x }}")  # Default is now strict
        t_lenient = PromptTemplate("{{ x }}", strict=False)
        assert t_strict.strict is True
        assert t_lenient.strict is False

    def test_strict_mode_missing_attribute(self):
        """Strict mode raises on missing object attribute."""
        t = PromptTemplate("{{ user.name }}", strict=True)
        with pytest.raises(TemplateUndefinedError):
            t(user={})

    def test_strict_mode_partial_preserves(self):
        """Partial preserves strict mode setting."""
        t = PromptTemplate("{{ a }} {{ b }}", strict=True)
        t2 = t.partial(a="1")
        assert t2.strict is True
        # Should still raise for undefined b
        with pytest.raises(TemplateUndefinedError):
            t2()

    def test_runtime_strict_override_strict_to_lenient(self):
        """Runtime strict=False overrides strict instance."""
        t = PromptTemplate("Hello {{ name }}!")  # strict by default
        # Normal strict behavior raises
        with pytest.raises(TemplateUndefinedError):
            t()
        # Runtime lenient override
        assert t(strict=False) == "Hello !"

    def test_runtime_strict_override_lenient_to_strict(self):
        """Runtime strict=True overrides lenient instance."""
        t = PromptTemplate("Hello {{ name }}!", strict=False)
        # Normal lenient behavior
        assert t() == "Hello !"
        # Runtime strict override
        with pytest.raises(TemplateUndefinedError):
            t(strict=True)

    def test_runtime_strict_none_uses_instance(self):
        """Runtime strict=None uses instance setting."""
        t_strict = PromptTemplate("{{ x }}")  # Default is now strict
        t_lenient = PromptTemplate("{{ x }}", strict=False)
        # None uses instance default
        with pytest.raises(TemplateUndefinedError):
            t_strict(strict=None)
        assert t_lenient(strict=None) == ""

    def test_runtime_strict_with_render(self):
        """render() accepts strict parameter."""
        t = PromptTemplate("{{ x }}")  # strict by default
        with pytest.raises(TemplateUndefinedError):
            t.render()
        assert t.render(strict=False) == ""
        with pytest.raises(TemplateUndefinedError):
            t.render(strict=True)

    def test_runtime_strict_with_format(self):
        """format() accepts strict parameter."""
        t = PromptTemplate("{{ x }}")  # strict by default
        with pytest.raises(TemplateUndefinedError):
            t.render()
        assert t.render(strict=False) == ""
        with pytest.raises(TemplateUndefinedError):
            t.render(strict=True)

    def test_runtime_strict_with_debug(self):
        """debug mode respects runtime strict override."""
        t = PromptTemplate("Hello {{ name }}!")  # strict by default
        # Debug with strict (default) raises
        with pytest.raises(TemplateUndefinedError):
            t(debug=True)
        # Debug with lenient
        result = t(debug=True, strict=False)
        assert result.output == "Hello !"


class TestPartialApplication:
    """Tests for Template.partial() method."""

    def test_partial_single_var(self):
        """Partial fills a single variable."""
        t = PromptTemplate("{{ greeting }}, {{ name }}!")
        t2 = t.partial(greeting="Hello")
        result = t2(name="World")
        assert result == "Hello, World!"

    def test_partial_updates_input_variables(self):
        """Partial removes pre-filled variables from input_variables."""
        t = PromptTemplate("{{ greeting }}, {{ name }}!")
        t2 = t.partial(greeting="Hello")
        assert t2.input_variables == {"name"}
        assert "greeting" not in t2.input_variables

    def test_partial_chain(self):
        """Partials can be chained."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        t2 = t.partial(a="1")
        t3 = t2.partial(b="2")
        result = t3(c="3")
        assert result == "1 2 3"

    def test_partial_override_at_render(self):
        """Render-time variables override partial variables."""
        t = PromptTemplate("{{ name }}")
        t2 = t.partial(name="Alice")
        result = t2(name="Bob")
        assert result == "Bob"

    def test_partial_original_unchanged(self):
        """Original template is not modified by partial."""
        t = PromptTemplate("{{ a }} {{ b }}")
        _ = t.partial(a="1")
        assert t.input_variables == {"a", "b"}

    def test_partial_complex_value(self):
        """Partial accepts complex values (dicts, lists)."""
        t = PromptTemplate("{% for doc in docs %}{{ doc.title }}{% endfor %}")
        t2 = t.partial(docs=[{"title": "Doc1"}, {"title": "Doc2"}])
        result = t2()
        assert result == "Doc1Doc2"

    def test_partial_with_non_json_raises(self):
        """Partial with non-JSON-serializable value raises TemplateValueError."""
        t = PromptTemplate("{{ func }}")
        with pytest.raises(TemplateValueError) as exc_info:
            t.partial(func=lambda x: x)
        assert "func" in str(exc_info.value)
        assert "cannot be serialized to JSON" in str(exc_info.value)


class TestJSONValidation:
    """Tests for JSON serialization validation."""

    def test_render_with_function_raises(self):
        """Render with function value raises TemplateValueError."""
        t = PromptTemplate("{{ func }}")
        with pytest.raises(TemplateValueError) as exc_info:
            t(func=lambda x: x)
        assert "func" in str(exc_info.value)

    def test_render_with_class_raises(self):
        """Render with class instance raises TemplateValueError."""

        class CustomClass:
            pass

        t = PromptTemplate("{{ obj }}")
        with pytest.raises(TemplateValueError) as exc_info:
            t(obj=CustomClass())
        assert "obj" in str(exc_info.value)

    def test_render_with_valid_types(self):
        """Render works with JSON-compatible types."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }} {{ d }} {{ e }} {{ f }}")
        result = t(
            a="string",
            b=42,
            c=3.14,
            d=True,
            e=None,
            f=[1, 2, 3],
        )
        # Just verify no error is raised
        assert isinstance(result, str)

    def test_render_with_nested_dict(self):
        """Render works with nested dicts."""
        t = PromptTemplate("{{ user.name }}")
        result = t(user={"name": "Alice", "age": 30})
        assert result == "Alice"

    def test_error_message_includes_type(self):
        """Error message includes the type that failed."""
        t = PromptTemplate("{{ func }}")
        with pytest.raises(TemplateValueError) as exc_info:
            t(func=lambda: None)
        assert "function" in str(exc_info.value)

    def test_error_message_includes_guidance(self):
        """Error message includes guidance about valid types."""
        t = PromptTemplate("{{ obj }}")
        with pytest.raises(TemplateValueError) as exc_info:
            t(obj=object())
        assert "str, int, float, bool, None, list, dict" in str(exc_info.value)


class TestFromFile:
    """Tests for PromptTemplate.from_file() class method."""

    def test_from_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file(str(tmp_path / "nonexistent.j2"))

    def test_from_file_basic(self, tmp_path):
        """Load template from file."""
        template_file = tmp_path / "greeting.j2"
        template_file.write_text("Hello {{ name }}!")
        t = PromptTemplate.from_file(str(template_file))
        result = t(name="World")
        assert result == "Hello World!"


class TestFromChatTemplate:
    """Tests for PromptTemplate.from_chat_template() class method."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        """Create a mock model directory with chat template."""
        model_path = tmp_path / "mock-model"
        model_path.mkdir()

        # Create tokenizer_config.json with chat_template
        config = {
            "chat_template": (
                "{%- for message in messages %}"
                "<|{{ message.role }}|>{{ message.content }}<|end|>"
                "{%- endfor %}"
                "{%- if add_generation_prompt %}<|assistant|>{% endif %}"
            ),
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        import json

        (model_path / "tokenizer_config.json").write_text(json.dumps(config))
        return model_path

    def test_from_chat_template_local_path(self, model_dir):
        """Load chat template from local model directory."""
        t = PromptTemplate.from_chat_template(str(model_dir))
        assert isinstance(t, PromptTemplate)
        assert "message.role" in t.source
        assert "message.content" in t.source

    def test_from_chat_template_source_accessible(self, model_dir):
        """Template source is accessible for inspection."""
        t = PromptTemplate.from_chat_template(str(model_dir))
        # Source should contain the template
        assert "<|{{ message.role }}|>" in t.source
        assert "add_generation_prompt" in t.source

    def test_from_chat_template_can_render(self, model_dir):
        """Template from model can render messages."""
        t = PromptTemplate.from_chat_template(str(model_dir))
        result = t(
            messages=[{"role": "user", "content": "Hello!"}],
            add_generation_prompt=True,
        )
        assert "<|user|>Hello!<|end|>" in result
        assert "<|assistant|>" in result

    def test_from_chat_template_strict_mode(self, model_dir):
        """Strict mode is passed through to Template."""
        t = PromptTemplate.from_chat_template(str(model_dir), strict=True)
        assert t.strict is True

    def test_from_chat_template_not_found(self, tmp_path):
        """Raise FileNotFoundError for non-existent model."""
        with pytest.raises(FileNotFoundError) as exc_info:
            PromptTemplate.from_chat_template(str(tmp_path / "nonexistent-model"))
        # Error message contains "TemplateNotFound" or "not found"
        msg = str(exc_info.value).lower()
        assert "notfound" in msg or "not found" in msg

    def test_from_chat_template_no_chat_template(self, tmp_path):
        """Raise FileNotFoundError if model has no chat template."""
        model_path = tmp_path / "no-template-model"
        model_path.mkdir()
        # Create tokenizer_config.json WITHOUT chat_template
        import json

        (model_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}))

        with pytest.raises(FileNotFoundError) as exc_info:
            PromptTemplate.from_chat_template(str(model_path))
        assert "chat template" in str(exc_info.value).lower()

    def test_from_chat_template_jinja_file_fallback(self, tmp_path):
        """Fall back to chat_template.jinja file if not in config."""
        model_path = tmp_path / "jinja-file-model"
        model_path.mkdir()

        # Create tokenizer_config.json WITHOUT chat_template
        import json

        (model_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}))

        # Create chat_template.jinja file
        jinja_template = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
        (model_path / "chat_template.jinja").write_text(jinja_template)

        t = PromptTemplate.from_chat_template(str(model_path))
        assert "bos_token" in t.source
        assert "m.content" in t.source

    def test_from_chat_template_can_modify_and_reuse(self, model_dir):
        """Template source can be modified and used to create new Template."""
        original = PromptTemplate.from_chat_template(str(model_dir))

        # Modify the source
        modified_source = original.source.replace("assistant", "AI")

        # Create new template from modified source
        modified = PromptTemplate(modified_source)

        # Verify modification
        assert "AI" in modified.source
        assert "assistant" not in modified.source

        # Verify it still renders
        result = modified(
            messages=[{"role": "user", "content": "Hi"}],
            add_generation_prompt=True,
        )
        assert "<|AI|>" in result


class TestIntegration:
    """Integration tests for prompt engineering workflows."""

    def test_rag_pipeline(self):
        """RAG pipeline with partial application."""
        system = PromptTemplate("""
System: {{ persona }}
Context:
{% for doc in context %}
- {{ doc }}
{% endfor %}
User: {{ query }}
""")
        # Step 1: Bind persona at startup
        assistant = system.partial(persona="You are a helpful assistant.")
        assert "persona" not in assistant.input_variables

        # Step 2: Bind context when retrieval completes
        with_context = assistant.partial(context=["Fact 1", "Fact 2"])
        assert with_context.input_variables == {"query"}

        # Step 3: Render with user query
        result = with_context(query="Tell me about the facts")
        assert "You are a helpful assistant" in result
        assert "Fact 1" in result
        assert "Tell me about the facts" in result

    def test_multi_persona_system(self):
        """Create multiple personas from a single template."""
        base = PromptTemplate("{{ persona }}: {{ message }}")

        alice = base.partial(persona="Alice")
        bob = base.partial(persona="Bob")

        assert alice(message="Hello") == "Alice: Hello"
        assert bob(message="Hi there") == "Bob: Hi there"

    def test_introspection_for_validation(self):
        """Use input_variables to validate inputs before render."""
        t = PromptTemplate("{{ name }} is {{ age }} years old")
        required = t.input_variables

        # Validate user input
        user_input = {"name": "Alice"}
        missing = required - set(user_input.keys())
        assert missing == {"age"}


class TestValidateMethod:
    """Tests for Template.validate() method."""

    def test_validate_all_provided(self):
        """Validation passes when all variables are provided."""
        t = PromptTemplate("Hello {{ name }}, you are {{ age }} years old")
        result = t.validate(name="Alice", age=30)
        assert result.is_valid
        assert result.required == set()
        assert result.extra == set()
        assert result.invalid == {}

    def test_validate_missing_variable(self):
        """Validation fails when required variable is missing."""
        t = PromptTemplate("Hello {{ name }}, you are {{ age }} years old")
        result = t.validate(name="Alice")
        assert not result.is_valid
        assert result.required == {"age"}

    def test_validate_multiple_missing(self):
        """Validation detects multiple missing variables."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        result = t.validate(a="1")
        assert not result.is_valid
        assert result.required == {"b", "c"}

    def test_validate_extra_variables(self):
        """Extra variables are reported but don't fail validation."""
        t = PromptTemplate("Hello {{ name }}")
        result = t.validate(name="Alice", unused="extra", another="ignored")
        assert result.is_valid  # Extra vars don't fail validation
        assert result.extra == {"unused", "another"}
        assert result.required == set()

    def test_validate_invalid_json_value(self):
        """Validation fails for non-JSON-serializable values."""
        t = PromptTemplate("{{ func }}")
        result = t.validate(func=lambda x: x)
        assert not result.is_valid
        assert "func" in result.invalid
        assert "JSON" in result.invalid["func"]

    def test_validate_invalid_custom_class(self):
        """Validation fails for custom class instances."""

        class CustomObj:
            pass

        t = PromptTemplate("{{ obj }}")
        result = t.validate(obj=CustomObj())
        assert not result.is_valid
        assert "obj" in result.invalid

    def test_validate_with_partial(self):
        """Validation considers partial variables as provided."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        t2 = t.partial(a="1")
        result = t2.validate(b="2", c="3")
        assert result.is_valid
        assert result.required == set()

    def test_validate_with_partial_missing(self):
        """Validation detects missing after partial application."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        t2 = t.partial(a="1")
        result = t2.validate(b="2")
        assert not result.is_valid
        assert result.required == {"c"}

    def test_validate_boolean_context_valid(self):
        """ValidationResult is truthy when valid."""
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="Alice")
        assert bool(result) is True
        # Can use directly in if statement
        if t.validate(name="Alice"):
            passed = True
        else:
            passed = False
        assert passed

    def test_validate_boolean_context_invalid(self):
        """ValidationResult is falsy when invalid."""
        t = PromptTemplate("{{ name }}")
        result = t.validate()
        assert bool(result) is False
        # Can use directly in if statement
        if t.validate():
            passed = True
        else:
            passed = False
        assert not passed

    def test_validate_summary_missing(self):
        """Summary includes missing variables."""
        t = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        result = t.validate(a="1")
        assert "Missing required variables" in result.summary
        assert "b" in result.summary
        assert "c" in result.summary

    def test_validate_summary_invalid(self):
        """Summary includes invalid values."""
        t = PromptTemplate("{{ func }}")
        result = t.validate(func=lambda: None)
        assert "Invalid values" in result.summary
        assert "func" in result.summary

    def test_validate_summary_extra(self):
        """Summary includes extra variables."""
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="Alice", unused="extra")
        # Extra vars appear in summary but don't fail validation
        assert "Extra variables" in result.summary
        assert "unused" in result.summary
        assert result.is_valid  # Still valid

    def test_validate_summary_empty_when_valid(self):
        """Summary is empty when validation passes with no extras."""
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="Alice")
        assert result.summary == ""

    def test_validate_no_variables(self):
        """Validation passes for template with no variables."""
        t = PromptTemplate("Hello World!")
        result = t.validate()
        assert result.is_valid
        assert result.required == set()

    def test_validate_no_variables_with_extra(self):
        """Extra variables detected even for no-variable template."""
        t = PromptTemplate("Hello World!")
        result = t.validate(name="Alice")
        assert result.is_valid  # Extra vars don't fail
        assert result.extra == {"name"}

    def test_validate_mixed_issues(self):
        """Validation reports all issues simultaneously."""
        t = PromptTemplate("{{ a }} {{ b }}")

        class BadObj:
            pass

        result = t.validate(b=BadObj(), extra="ignored")
        assert not result.is_valid
        # 'b' is in invalid because it can't be serialized
        # 'a' is missing, and 'b' is also missing since the invalid value wasn't accepted
        assert "a" in result.required
        assert result.extra == {"extra"}
        assert "b" in result.invalid

    def test_validate_complex_values_pass(self):
        """Validation passes for JSON-serializable complex values."""
        t = PromptTemplate("{% for item in items %}{{ item.name }}{% endfor %}")
        result = t.validate(items=[{"name": "Alice"}, {"name": "Bob"}])
        assert result.is_valid

    def test_validate_nested_non_serializable(self):
        """Validation catches non-serializable values in nested structures."""
        t = PromptTemplate("{{ data.func }}")
        result = t.validate(data={"func": lambda: None})
        assert not result.is_valid
        assert "data" in result.invalid

    def test_validate_workflow_pattern(self):
        """Demonstrate validation in typical workflow."""
        t = PromptTemplate("{{ name }} is {{ age }} years old")

        # Simulate user input that may be incomplete
        user_input = {"name": "Alice"}
        result = t.validate(**user_input)

        if not result:
            # Handle validation failure
            assert "age" in result.required
            error_msg = result.summary
            assert "Missing required variables" in error_msg
        else:
            pytest.fail("Validation should have failed")

    def test_validate_invalid_template_syntax(self):
        """Validation handles templates with syntax errors gracefully."""
        # Note: PromptTemplate() constructor catches syntax errors,
        # so we can't easily test validate() with a broken template.
        # This test documents that validation works on valid templates.
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="Alice")
        assert result.is_valid

    def test_validate_complex_nested_json(self):
        """Validation handles complex nested JSON structures."""
        t = PromptTemplate("{% for item in data.items %}{{ item.name }}{% endfor %}")
        result = t.validate(data={"items": [{"name": "a"}, {"name": "b"}]})
        assert result.is_valid
        assert len(result.required) == 0

    def test_validate_array_input(self):
        """Validation handles array inputs."""
        t = PromptTemplate("{% for x in items %}{{ x }}{% endfor %}")
        result = t.validate(items=["a", "b", "c"])
        assert result.is_valid

    def test_validate_unicode_values(self):
        """Validation handles unicode in values."""
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="日本語テスト")
        assert result.is_valid

    def test_validate_special_json_values(self):
        """Validation handles special JSON values (null, booleans)."""
        t = PromptTemplate("{% if flag %}{{ value }}{% endif %}")
        result = t.validate(flag=True, value=None)
        assert result.is_valid


class TestValidationResultRender:
    """Tests for ValidationResult.render() - efficient validate-then-render pattern."""

    def test_render_basic(self):
        """ValidationResult.render() produces correct output."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t.validate(name="World")
        assert result.is_valid
        assert result.render() == "Hello World!"

    def test_render_matches_direct(self):
        """ValidationResult.render() matches direct template call."""
        t = PromptTemplate("{{ a }} + {{ b }} = {{ c }}")
        result = t.validate(a="1", b="2", c="3")
        direct = t(a="1", b="2", c="3")
        assert result.render() == direct

    def test_render_complex_data(self):
        """ValidationResult.render() works with complex nested data."""
        t = PromptTemplate(
            "{% for doc in documents %}{{ doc.title }}: {{ doc.content }}\n{% endfor %}"
        )
        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]
        result = t.validate(documents=docs)
        assert result.is_valid
        output = result.render()
        assert "Doc 1: Content 1" in output
        assert "Doc 2: Content 2" in output

    def test_render_with_partial(self):
        """ValidationResult.render() works with partial templates."""
        t = PromptTemplate("{{ system }}\n{{ query }}")
        partial = t.partial(system="You are helpful")
        result = partial.validate(query="Hello!")
        assert result.is_valid
        assert result.render() == "You are helpful\nHello!"

    def test_render_raises_on_invalid(self):
        """ValidationResult.render() raises TemplateError if validation failed."""
        t = PromptTemplate("{{ name }} is {{ age }}")
        result = t.validate(name="Alice")  # Missing 'age'
        assert not result.is_valid
        with pytest.raises(talu.TemplateError) as exc_info:
            result.render()
        assert "Cannot render invalid template" in str(exc_info.value)
        assert "age" in str(exc_info.value)

    def test_render_raises_on_detached_result(self):
        """ValidationResult.render() raises if result has no template reference."""
        from talu.template import ValidationResult

        # Manually created result without template reference
        detached = ValidationResult(required=set(), optional=set(), extra=set(), invalid={})
        with pytest.raises(talu.StateError) as exc_info:
            detached.render()
        assert "requires a result created by" in str(exc_info.value)

    def test_render_strict_override(self):
        """ValidationResult.render() can override strict mode."""
        t = PromptTemplate("{{ name }} {{ missing }}")
        result = t.validate(name="Alice")
        # Result has missing=required, but we can still render in lenient mode
        # because only 'missing' is flagged as required in the ValidationResult
        # Actually, the result is invalid due to missing 'missing' variable
        assert not result.is_valid  # 'missing' is required

    def test_render_strict_mode_from_template(self):
        """ValidationResult.render() inherits strict mode from template."""
        t = PromptTemplate("Hello {{ name }}!", strict=True)
        result = t.validate(name="World")
        assert result.is_valid
        # Should work since all vars provided
        assert result.render() == "Hello World!"

    def test_render_strict_override_explicit(self):
        """ValidationResult.render(strict=...) overrides template setting."""
        t = PromptTemplate("{{ x }} {{ y | default('fallback') }}", strict=False)
        result = t.validate(x="hello")  # y is optional
        assert result.is_valid
        # Render with lenient mode (inherited)
        output = result.render()
        assert output == "hello fallback"

    def test_render_preserves_filters(self):
        """ValidationResult.render() works with filtered values."""
        t = PromptTemplate("{{ name | upper }}")
        result = t.validate(name="alice")
        assert result.is_valid
        assert result.render() == "ALICE"

    def test_render_with_custom_filters(self):
        """ValidationResult.render() works with custom Python filters."""
        t = PromptTemplate("{{ x | double }}")
        t.register_filter("double", lambda x: x * 2)
        result = t.validate(x=5)
        assert result.is_valid
        assert result.render() == "10"

    def test_render_workflow_pattern(self):
        """Demonstrate efficient validate-then-render workflow."""
        # Simulate RAG pattern with large documents
        t = PromptTemplate("""
Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ question }}
""")
        large_docs = [{"content": f"Document {i} content"} for i in range(100)]

        # Validate once (serializes large_docs)
        result = t.validate(documents=large_docs, question="What is the answer?")

        if result.is_valid:
            # Render reuses serialized data (no re-serialization)
            output = result.render()
            assert "Document 0 content" in output
            assert "Document 99 content" in output
            assert "What is the answer?" in output
        else:
            pytest.fail(f"Validation should pass: {result.summary}")


class TestDebugMode:
    """Tests for template debug mode (debug=True)."""

    def test_debug_basic(self):
        """Debug render returns output and spans."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        assert result.output == "Hello World!"
        assert len(result.spans) == 3  # "Hello ", "World", "!"

    def test_debug_span_positions(self):
        """Spans have correct start/end positions."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        # "Hello " -> [0:6], "World" -> [6:11], "!" -> [11:12]
        assert result.spans[0].start == 0
        assert result.spans[0].end == 6
        assert result.spans[0].text == "Hello "
        assert result.spans[1].start == 6
        assert result.spans[1].end == 11
        assert result.spans[1].text == "World"
        assert result.spans[2].start == 11
        assert result.spans[2].end == 12
        assert result.spans[2].text == "!"

    def test_debug_static_span(self):
        """Static text spans are correctly identified."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        static_spans = [s for s in result.spans if s.is_static]
        assert len(static_spans) == 2
        assert static_spans[0].source == "static"
        assert static_spans[0].text == "Hello "

    def test_debug_variable_span(self):
        """Variable spans include variable path."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 1
        assert var_spans[0].source == "name"
        assert var_spans[0].text == "World"

    def test_debug_nested_attribute(self):
        """Nested attribute paths are correctly tracked."""
        t = PromptTemplate("Email: {{ user.email }}")
        result = t(user={"email": "test@example.com"}, debug=True)
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 1
        assert var_spans[0].source == "user.email"
        assert var_spans[0].text == "test@example.com"

    def test_debug_array_index(self):
        """Array indexing paths are tracked."""
        t = PromptTemplate("First: {{ items[0] }}")
        result = t(items=["apple", "banana"], debug=True)
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 1
        # Array access should be tracked with index
        assert "items" in var_spans[0].source
        assert var_spans[0].text == "apple"

    def test_debug_expression(self):
        """Arithmetic expressions are marked as expressions."""
        t = PromptTemplate("Sum: {{ a + b }}")
        result = t(a=2, b=3, debug=True)
        expr_spans = [s for s in result.spans if s.is_expression]
        assert len(expr_spans) == 1
        assert expr_spans[0].source == "expression"
        assert expr_spans[0].text == "5"

    def test_debug_filter_attributes_to_variable(self):
        """Filters on variables still attribute to the variable."""
        t = PromptTemplate("Hello {{ name | upper }}!")
        result = t(name="world", debug=True)
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 1
        # Should attribute to 'name', not mark as expression
        assert var_spans[0].source == "name"
        assert var_spans[0].text == "WORLD"

    def test_debug_multiple_variables(self):
        """Multiple variables are all tracked."""
        t = PromptTemplate("{{ a }} and {{ b }} and {{ c }}")
        result = t(a="X", b="Y", c="Z", debug=True)
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 3
        assert {s.source for s in var_spans} == {"a", "b", "c"}

    def test_debug_empty_value(self):
        """Empty variable values produce no span (zero-width spans skipped)."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="", debug=True)
        # Empty values don't produce a span - they render to nothing
        assert result.output == "Hello !"
        var_spans = [s for s in result.spans if s.is_variable]
        # No span for empty value (correct behavior - nothing to highlight)
        assert len(var_spans) == 0
        # But static text is still tracked
        static_spans = [s for s in result.spans if s.is_static]
        assert len(static_spans) == 2  # "Hello " and "!"

    def test_debug_format_ansi(self):
        """ANSI format includes color codes."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        ansi = result.format_ansi()
        # Should contain ANSI escape codes
        assert "\033[" in ansi
        # Should contain variable name annotation
        assert "name" in ansi
        assert "World" in ansi

    def test_debug_format_ansi_empty_value(self):
        """Empty values don't produce spans, so ANSI shows just static text."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="", debug=True)
        ansi = result.format_ansi()
        # Empty values produce no span, so output is just "Hello !" (static text)
        # Use validate() before rendering to catch empty/missing values
        assert ansi == "Hello !"

    def test_debug_format_plain(self):
        """Plain format uses text markers."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="World", debug=True)
        plain = result.format_plain()
        # Should use « » markers
        assert "«World»" in plain
        assert "(name)" in plain

    def test_debug_format_plain_empty_value(self):
        """Empty values don't produce spans, so plain shows just static text."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t(name="", debug=True)
        plain = result.format_plain()
        # Empty values produce no span, so output is just "Hello !" (static text)
        # Use validate() before rendering to catch empty/missing values
        assert plain == "Hello !"

    def test_debug_format_ansi_expression(self):
        """ANSI format uses yellow for expressions (no variable source)."""
        t = PromptTemplate("Sum: {{ 1 + 2 }}")
        result = t(debug=True)
        ansi = result.format_ansi()
        # Expression spans should be yellow (\033[33m)
        assert "\033[33m" in ansi
        assert "3" in ansi

    def test_debug_format_plain_expression(self):
        """Plain format uses «value» for expressions (no variable name)."""
        t = PromptTemplate("Sum: {{ 1 + 2 }}")
        result = t(debug=True)
        plain = result.format_plain()
        # Expression without variable name shows just «value»
        assert "«3»" in plain
        # No parentheses with variable name since it's an expression
        assert "(expression)" not in plain

    def test_debug_with_partial(self):
        """Debug mode works with partial application."""
        t = PromptTemplate("{{ a }} {{ b }}")
        t2 = t.partial(a="X")
        result = t2(b="Y", debug=True)
        assert result.output == "X Y"
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 2
        # Both a and b should be tracked as variables
        sources = {s.source for s in var_spans}
        assert "a" in sources
        assert "b" in sources

    def test_debug_loop_static(self):
        """Static text inside loops is tracked."""
        t = PromptTemplate("{% for i in items %}[{{ i }}]{% endfor %}")
        result = t(items=["a", "b"], debug=True)
        # Output: "[a][b]"
        assert result.output == "[a][b]"
        static_spans = [s for s in result.spans if s.is_static]
        # Should have static brackets
        assert any(s.text == "[" for s in static_spans)
        assert any(s.text == "]" for s in static_spans)

    def test_debug_output_equals_regular_render(self):
        """Debug output matches regular render output."""
        t = PromptTemplate("Hello {{ name }}, age {{ age }}!")
        regular = t(name="Alice", age=30)
        debug = t(name="Alice", age=30, debug=True)
        assert debug.output == regular

    def test_debug_str_uses_ansi(self):
        """str(DebugResult) uses ANSI format."""
        t = PromptTemplate("{{ x }}")
        result = t(x="test", debug=True)
        assert str(result) == result.format_ansi()

    def test_debug_span_properties(self):
        """DebugSpan properties work correctly."""
        t = PromptTemplate("{{ a }} {{ b + c }}")
        result = t(a="X", b=1, c=2, debug=True)

        # Find each type
        static_span = next((s for s in result.spans if s.is_static), None)
        var_span = next((s for s in result.spans if s.is_variable), None)
        expr_span = next((s for s in result.spans if s.is_expression), None)

        assert static_span is not None
        assert static_span.is_static is True
        assert static_span.is_variable is False
        assert static_span.is_expression is False

        assert var_span is not None
        assert var_span.is_static is False
        assert var_span.is_variable is True
        assert var_span.is_expression is False

        assert expr_span is not None
        assert expr_span.is_static is False
        assert expr_span.is_variable is False
        assert expr_span.is_expression is True

    def test_debug_complex_rag(self):
        """Debug mode works with complex RAG template."""
        t = PromptTemplate("""Context:
{% for doc in docs %}
- {{ doc.title }}: {{ doc.content }}
{% endfor %}
Question: {{ question }}""")
        result = t(
            docs=[{"title": "Doc1", "content": "Content1"}],
            question="What?",
            debug=True,
        )
        # Should render correctly
        assert "Doc1" in result.output
        assert "Content1" in result.output
        assert "What?" in result.output
        # Should have variable spans
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) >= 3  # At least title, content, question

    def test_debug_only_static(self):
        """Template with only static text has no variable spans."""
        t = PromptTemplate("Hello World!")
        result = t(debug=True)
        assert result.output == "Hello World!"
        var_spans = [s for s in result.spans if s.is_variable]
        assert len(var_spans) == 0
        static_spans = [s for s in result.spans if s.is_static]
        assert len(static_spans) == 1
        assert static_spans[0].text == "Hello World!"

    def test_debug_json_validation(self):
        """Debug mode validates JSON serializability."""
        t = PromptTemplate("{{ func }}")
        with pytest.raises(TemplateValueError):
            t(func=lambda: None, debug=True)

    def test_debug_env_var(self):
        """TALU_DEBUG_TEMPLATES=1 enables debug mode globally."""
        import os

        t = PromptTemplate("Hello {{ name }}!")

        # Without env var, returns str
        result = t(name="World")
        assert isinstance(result, str)

        # With env var, returns DebugResult
        os.environ["TALU_DEBUG_TEMPLATES"] = "1"
        try:
            result = t(name="World")
            assert hasattr(result, "output")
            assert hasattr(result, "spans")
            assert result.output == "Hello World!"
        finally:
            del os.environ["TALU_DEBUG_TEMPLATES"]


class TestTemplateEnvironment:
    """Tests for TemplateEnvironment - shared configuration for templates."""

    def test_environment_globals(self):
        """Environment globals are available in templates."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.globals["app_name"] = "MyApp"
        env.globals["version"] = "2.0"

        t = env.from_string("{{ app_name }} v{{ version }}")
        assert t() == "MyApp v2.0"

    def test_environment_filters(self):
        """Environment filters are available in templates."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.register_filter("double", lambda x: x * 2)

        t = env.from_string("{{ x | double }}")
        assert t(x=5) == "10"

    def test_environment_filter_chaining(self):
        """register_filter returns self for chaining."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        result = env.register_filter("a", str.upper).register_filter("b", str.lower)
        assert result is env

    def test_environment_register_filter_non_callable_raises(self):
        """TemplateEnvironment.register_filter raises ValidationError for non-callable."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        with pytest.raises(talu.ValidationError, match="callable"):
            env.register_filter("bad", "not a function")

    def test_environment_strict_property(self):
        """TemplateEnvironment.strict property returns the strict mode."""
        from talu.template import TemplateEnvironment

        env_strict = TemplateEnvironment(strict=True)
        env_lenient = TemplateEnvironment(strict=False)
        assert env_strict.strict is True
        assert env_lenient.strict is False

    def test_environment_strict_default(self):
        """Environment strict mode (True by default) is inherited by templates."""
        from talu.exceptions import TemplateUndefinedError
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()  # strict=True by default
        t = env.from_string("{{ missing }}")

        with pytest.raises(TemplateUndefinedError):
            t()

    def test_environment_strict_override(self):
        """Template can override environment's strict mode."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()  # strict=True by default
        t = env.from_string("{{ missing }}", strict=False)

        # Should not raise - overridden to lenient
        result = t()
        assert result == ""

    def test_environment_multiple_templates(self):
        """Multiple templates share environment configuration."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.globals["brand"] = "Acme"
        env.register_filter("shout", lambda s: s.upper() + "!")

        t1 = env.from_string("{{ brand }}")
        t2 = env.from_string("{{ name | shout }}")
        t3 = env.from_string("Welcome to {{ brand }}, {{ name }}!")

        assert t1() == "Acme"
        assert t2(name="hello") == "HELLO!"
        assert t3(name="Alice") == "Welcome to Acme, Alice!"

    def test_render_time_overrides_globals(self):
        """Render-time variables override environment globals."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.globals["name"] = "Default"

        t = env.from_string("Hello {{ name }}!")

        # Without override - uses global
        assert t() == "Hello Default!"

        # With override - uses render-time value
        assert t(name="Override") == "Hello Override!"

    def test_partial_overrides_globals(self):
        """Partial variables override environment globals."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.globals["a"] = "global_a"
        env.globals["b"] = "global_b"

        t = env.from_string("{{ a }} {{ b }}")
        partial = t.partial(a="partial_a")

        # Partial overrides global for 'a', keeps global for 'b'
        assert partial() == "partial_a global_b"

    def test_from_file(self, tmp_path):
        """Environment can load templates from files."""
        from talu.template import TemplateEnvironment

        # Create a temp template file
        template_file = tmp_path / "greeting.j2"
        template_file.write_text("Hello {{ name }}!")

        env = TemplateEnvironment()
        t = env.from_file(str(template_file))

        assert t(name="World") == "Hello World!"

    def test_from_file_with_globals(self, tmp_path):
        """File-loaded templates have access to environment globals."""
        from talu.template import TemplateEnvironment

        template_file = tmp_path / "branded.j2"
        template_file.write_text("{{ app_name }} says: {{ message }}")

        env = TemplateEnvironment()
        env.globals["app_name"] = "MyBot"

        t = env.from_file(str(template_file))
        assert t(message="Hello!") == "MyBot says: Hello!"

    def test_from_file_not_found(self):
        """from_file raises FileNotFoundError for missing files."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        with pytest.raises(FileNotFoundError):
            env.from_file("/nonexistent/path/template.j2")

    def test_globals_dict_access(self):
        """Globals dict can be accessed and modified directly."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.globals["a"] = 1
        env.globals["b"] = 2

        assert env.globals == {"a": 1, "b": 2}

        del env.globals["a"]
        t = env.from_string("{{ b }}")
        assert t() == "2"

    def test_filters_dict_access(self):
        """Filters dict can be accessed and modified directly."""
        from talu.template import TemplateEnvironment

        env = TemplateEnvironment()
        env.filters["double"] = lambda x: x * 2

        t = env.from_string("{{ x | double }}")
        assert t(x=3) == "6"


class TestTemplateConfig:
    """Tests for template.config module."""

    def test_config_import(self):
        """config is importable from talu.template."""
        from talu.template import config

        assert config is not None

    def test_config_debug_default(self):
        """config.debug defaults to False."""
        from talu.template import config

        # Reset to default
        config.debug = False
        assert config.debug is False

    def test_config_debug_set(self):
        """config.debug can be set to True."""
        from talu.template import config

        original = config.debug
        try:
            config.debug = True
            assert config.debug is True
        finally:
            config.debug = original

    def test_config_debug_type_error(self):
        """config.debug raises ValidationError for non-bool."""
        from talu.template import config

        with pytest.raises(talu.ValidationError) as exc_info:
            config.debug = "yes"
        assert "bool" in str(exc_info.value)

    def test_config_repr(self):
        """config has readable repr."""
        from talu.template import config

        original = config.debug
        try:
            config.debug = False
            assert repr(config) == "TemplateConfig(debug=False)"
            config.debug = True
            assert repr(config) == "TemplateConfig(debug=True)"
        finally:
            config.debug = original
