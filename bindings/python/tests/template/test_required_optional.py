"""Tests for required vs optional variable validation.

Variables used "naked" ({{ name }}) are required - missing them fails validation.
Variables with default() filter ({{ bio | default('') }}) are optional - safe to omit.
"""

from talu.template import PromptTemplate


class TestRequiredVariables:
    """Tests for required (naked) variable detection."""

    def test_single_required_variable(self):
        """Single naked variable is required."""
        t = PromptTemplate("Hello {{ name }}!")
        result = t.validate()
        assert not result.is_valid
        assert result.required == {"name"}
        assert result.optional == set()

    def test_multiple_required_variables(self):
        """Multiple naked variables are all required."""
        t = PromptTemplate("{{ first }} {{ last }} is {{ age }} years old")
        result = t.validate()
        assert not result.is_valid
        assert result.required == {"first", "last", "age"}
        assert result.optional == set()

    def test_required_in_expression(self):
        """Variables in expressions are required."""
        t = PromptTemplate("{{ user.name }} - {{ items | length }}")
        result = t.validate()
        assert result.required == {"user", "items"}

    def test_required_in_condition(self):
        """Variables in if conditions are required."""
        t = PromptTemplate("{% if active %}Active{% endif %}")
        result = t.validate()
        assert result.required == {"active"}

    def test_required_in_for_loop(self):
        """Loop source variable is required."""
        t = PromptTemplate("{% for item in items %}{{ item }}{% endfor %}")
        result = t.validate()
        assert result.required == {"items"}
        assert "item" not in result.required  # Loop variable is defined


class TestOptionalVariables:
    """Tests for optional (with default filter) variable detection."""

    def test_default_filter_makes_optional(self):
        """Variable with default() filter is optional."""
        t = PromptTemplate("{{ bio | default('No bio') }}")
        result = t.validate()
        assert result.is_valid
        assert result.required == set()
        assert result.optional == {"bio"}

    def test_d_filter_alias(self):
        """The d() filter alias also makes variable optional."""
        t = PromptTemplate("{{ role | d('user') }}")
        result = t.validate()
        assert result.is_valid
        assert result.required == set()
        assert result.optional == {"role"}

    def test_default_empty_string(self):
        """Default to empty string makes variable optional."""
        t = PromptTemplate("{{ context | default('') }}")
        result = t.validate()
        assert result.is_valid
        assert result.optional == {"context"}

    def test_default_false_value(self):
        """Default to false/none makes variable optional."""
        t = PromptTemplate("{{ enabled | default(false) }}")
        result = t.validate()
        assert result.is_valid
        assert result.optional == {"enabled"}

    def test_multiple_optional_variables(self):
        """Multiple variables with defaults are all optional."""
        t = PromptTemplate("""
            {{ a | default(1) }}
            {{ b | d(2) }}
            {{ c | default('three') }}
        """)
        result = t.validate()
        assert result.is_valid
        assert result.required == set()
        assert result.optional == {"a", "b", "c"}


class TestMixedRequiredOptional:
    """Tests for templates with both required and optional variables."""

    def test_mixed_required_and_optional(self):
        """Template with both required and optional variables."""
        t = PromptTemplate("""
            Name: {{ name }}
            Bio: {{ bio | default('No bio') }}
            Role: {{ role | d('user') }}
        """)
        result = t.validate()
        assert not result.is_valid
        assert result.required == {"name"}
        assert result.optional == {"bio", "role"}

    def test_mixed_validation_with_required_provided(self):
        """Providing only required variables passes validation."""
        t = PromptTemplate("""
            Name: {{ name }}
            Bio: {{ bio | default('No bio') }}
        """)
        result = t.validate(name="Alice")
        assert result.is_valid
        assert result.required == set()
        assert result.optional == {"bio"}

    def test_mixed_validation_all_provided(self):
        """Providing all variables (required + optional) passes."""
        t = PromptTemplate("""
            Name: {{ name }}
            Bio: {{ bio | default('No bio') }}
        """)
        result = t.validate(name="Alice", bio="Developer")
        assert result.is_valid
        assert result.required == set()
        assert result.optional == set()

    def test_complex_mixed_template(self):
        """Complex template with multiple required and optional."""
        # Note: system_context is used naked in the body, making it required
        # even though it has default() in the condition
        t = PromptTemplate("""
            {% for doc in documents %}
            - {{ doc.content }}
            {% endfor %}
            Question: {{ question }}
            {% if system_context | default('') %}
            System: {{ system_context }}
            {% endif %}
        """)
        result = t.validate()
        # All three are required because system_context is used naked in body
        assert result.required == {"documents", "question", "system_context"}
        assert result.optional == set()


class TestVariableUsedBothWays:
    """Tests for edge case: variable used both naked and with default."""

    def test_naked_and_default_is_required(self):
        """Variable used both naked AND with default is required."""
        # If a variable is used naked anywhere, it's required
        t = PromptTemplate("{{ x }} - {{ x | default('fallback') }}")
        result = t.validate()
        assert not result.is_valid
        assert result.required == {"x"}
        assert result.optional == set()  # Not optional since used naked

    def test_default_first_naked_later(self):
        """Order doesn't matter - naked usage makes it required."""
        t = PromptTemplate("{{ y | default('a') }} then {{ y }}")
        result = t.validate()
        assert result.required == {"y"}
        assert result.optional == set()

    def test_multiple_defaults_still_optional(self):
        """Variable with default in multiple places stays optional."""
        t = PromptTemplate("""
            {{ ctx | default('a') }}
            {{ ctx | d('b') }}
        """)
        result = t.validate()
        assert result.is_valid
        assert result.optional == {"ctx"}
        assert result.required == set()


class TestInputVariablesProperty:
    """Tests for input_variables property combining required + optional."""

    def test_input_variables_includes_both(self):
        """input_variables returns union of required and optional."""
        t = PromptTemplate("""
            {{ name }}
            {{ bio | default('') }}
        """)
        assert t.input_variables == {"name", "bio"}

    def test_input_variables_only_required(self):
        """input_variables works with only required variables."""
        t = PromptTemplate("{{ a }} {{ b }}")
        assert t.input_variables == {"a", "b"}

    def test_input_variables_only_optional(self):
        """input_variables works with only optional variables."""
        t = PromptTemplate("{{ x | default(1) }} {{ y | d(2) }}")
        assert t.input_variables == {"x", "y"}


class TestIsValidProperty:
    """Tests for is_valid checking only required variables."""

    def test_is_valid_with_missing_required(self):
        """is_valid is False when required variable is missing."""
        t = PromptTemplate("{{ name }}")
        result = t.validate()
        assert not result.is_valid

    def test_is_valid_with_missing_optional(self):
        """is_valid is True even when optional variable is missing."""
        t = PromptTemplate("{{ bio | default('') }}")
        result = t.validate()
        assert result.is_valid

    def test_is_valid_ignores_extra_variables(self):
        """is_valid is True even with extra variables."""
        t = PromptTemplate("{{ name }}")
        result = t.validate(name="Alice", unused="extra")
        assert result.is_valid

    def test_is_valid_false_for_invalid_values(self):
        """is_valid is False when values are non-serializable."""
        t = PromptTemplate("{{ func }}")
        result = t.validate(func=lambda: None)
        assert not result.is_valid


class TestSummaryProperty:
    """Tests for summary message content."""

    def test_summary_mentions_required(self):
        """Summary mentions missing required variables."""
        t = PromptTemplate("{{ name }} {{ age }}")
        result = t.validate()
        assert "Missing required variables" in result.summary
        assert "name" in result.summary
        assert "age" in result.summary

    def test_summary_mentions_optional_missing(self):
        """Summary mentions missing optional variables."""
        t = PromptTemplate("{{ name }} {{ bio | default('') }}")
        result = t.validate(name="Alice")
        assert "Optional variables" in result.summary or "bio" in result.summary

    def test_summary_empty_when_all_provided(self):
        """Summary is empty when everything is provided."""
        t = PromptTemplate("{{ name }} {{ bio | default('') }}")
        result = t.validate(name="Alice", bio="Dev")
        assert result.summary == ""


class TestPartialWithRequiredOptional:
    """Tests for partial() with required/optional tracking."""

    def test_partial_makes_required_provided(self):
        """Partial application satisfies required variables."""
        t = PromptTemplate("{{ name }} {{ bio | default('') }}")
        t2 = t.partial(name="Alice")
        result = t2.validate()
        assert result.is_valid
        assert result.required == set()

    def test_partial_with_optional(self):
        """Partial can also provide optional variables."""
        t = PromptTemplate("{{ name }} {{ bio | default('') }}")
        t2 = t.partial(bio="Developer")
        result = t2.validate()
        assert not result.is_valid  # Still missing required 'name'
        assert result.required == {"name"}
        assert result.optional == set()  # bio was provided


class TestConditionalOptional:
    """Tests for optional variables in conditional blocks."""

    def test_optional_in_if_condition(self):
        """Variable with default in if condition is optional."""
        t = PromptTemplate("""
            {% if show_header | default(true) %}
            Header
            {% endif %}
        """)
        result = t.validate()
        assert result.is_valid
        assert result.optional == {"show_header"}

    def test_optional_in_conditional_body_naked_is_required(self):
        """Variable used naked in body is required, even if guarded by default."""
        # The naked usage {{ system }} makes it required, regardless of
        # the default() in the condition
        t = PromptTemplate("""
            {% if system | default('') %}
            System: {{ system }}
            {% endif %}
        """)
        result = t.validate()
        assert not result.is_valid
        # 'system' is required because it's used naked in body
        assert "system" in result.required

    def test_truly_optional_conditional(self):
        """Variable only used with default is truly optional."""
        # To make it truly optional, use default in both places
        t = PromptTemplate("""
            {% if ctx | default('') %}
            Context: {{ ctx | default('none') }}
            {% endif %}
        """)
        result = t.validate()
        assert result.is_valid
        assert result.optional == {"ctx"}


class TestRealWorldPatterns:
    """Tests for common real-world template patterns."""

    def test_chat_template_pattern_naked_body(self):
        """Chat template where body uses variables naked (common pattern)."""
        # When body uses naked {{ system_message }}, it becomes required
        t = PromptTemplate("""
            {% if system_message | default('') %}
            <|system|>{{ system_message }}<|end|>
            {% endif %}
            {% for msg in messages %}
            <|{{ msg.role }}|>{{ msg.content }}<|end|>
            {% endfor %}
            {% if add_generation_prompt | default(true) %}
            <|assistant|>
            {% endif %}
        """)
        result = t.validate()
        # system_message is used naked in body, so it's required
        assert result.required == {"messages", "system_message"}
        # add_generation_prompt is ONLY used with default, so it's optional
        assert result.optional == {"add_generation_prompt"}

    def test_chat_template_truly_optional(self):
        """Chat template where optional vars use default everywhere."""
        # To make system_message truly optional, use default in body too
        t = PromptTemplate("""
            {% if system_message | default('') %}
            <|system|>{{ system_message | default('') }}<|end|>
            {% endif %}
            {% for msg in messages %}
            <|{{ msg.role }}|>{{ msg.content }}<|end|>
            {% endfor %}
        """)
        result = t.validate()
        assert result.required == {"messages"}
        assert result.optional == {"system_message"}

    def test_rag_template_pattern_naked_body(self):
        """RAG template where optional instruction is used naked in body."""
        t = PromptTemplate("""
            Context:
            {% for doc in documents %}
            - {{ doc.content }}
            {% endfor %}

            Question: {{ question }}

            {% if instructions | default('') %}
            Instructions: {{ instructions }}
            {% endif %}
        """)
        result = t.validate()
        # instructions is used naked in body
        assert result.required == {"documents", "question", "instructions"}
        assert result.optional == set()

    def test_rag_template_truly_optional(self):
        """RAG template with truly optional instructions."""
        t = PromptTemplate("""
            Context:
            {% for doc in documents %}
            - {{ doc.content }}
            {% endfor %}

            Question: {{ question }}

            {% if instructions | default('') %}
            Instructions: {{ instructions | d('') }}
            {% endif %}
        """)
        result = t.validate()
        assert result.required == {"documents", "question"}
        assert result.optional == {"instructions"}

    def test_function_calling_pattern_for_loop(self):
        """Function calling - for loop over default makes var required."""
        # The for loop iterates over 'tools', making it required
        t = PromptTemplate("""
            {% if tools | default([]) | length > 0 %}
            Available tools:
            {% for tool in tools %}
            - {{ tool.name }}: {{ tool.description }}
            {% endfor %}
            {% endif %}

            {{ prompt }}
        """)
        result = t.validate()
        # 'tools' is used naked in for loop
        assert result.required == {"prompt", "tools"}

    def test_function_calling_truly_optional(self):
        """Function calling with truly optional tools."""
        # Use default on the for loop source to make it optional
        t = PromptTemplate("""
            {% if tools | default([]) | length > 0 %}
            Available tools:
            {% for tool in tools | default([]) %}
            - {{ tool.name }}: {{ tool.description }}
            {% endfor %}
            {% endif %}

            {{ prompt }}
        """)
        result = t.validate()
        assert result.required == {"prompt"}
        assert result.optional == {"tools"}


class TestEdgeCases:
    """Edge cases and corner cases."""

    def test_empty_template(self):
        """Empty template has no variables."""
        t = PromptTemplate("")
        result = t.validate()
        assert result.is_valid
        assert result.required == set()
        assert result.optional == set()

    def test_static_template(self):
        """Template with no variables."""
        t = PromptTemplate("Hello, World!")
        result = t.validate()
        assert result.is_valid
        assert result.required == set()
        assert result.optional == set()

    def test_nested_default_with_accessor(self):
        """Parenthesized expression with accessor - edge case behavior."""
        # This is a known limitation: parenthesized expressions with accessors
        # don't extract variables correctly. The workaround is to use
        # a simpler pattern or assign to a variable first.
        t = PromptTemplate("{{ (data | default({})).value }}")
        result = t.validate()
        # Currently returns no variables (edge case)
        # This test documents current behavior
        assert result.is_valid  # Passes because no required vars detected
        assert result.required == set()

    def test_nested_access_without_parens(self):
        """Access on variable without parentheses works correctly."""
        t = PromptTemplate("{{ data.value }}")
        result = t.validate()
        assert result.required == {"data"}

    def test_simple_default_is_optional(self):
        """Simple default filter makes variable optional."""
        t = PromptTemplate("{{ data | default({}) }}")
        result = t.validate()
        assert result.optional == {"data"}
        assert result.required == set()

    def test_chained_filters_with_default(self):
        """Default filter in chain of filters."""
        t = PromptTemplate("{{ name | default('anon') | upper }}")
        result = t.validate()
        assert result.optional == {"name"}

    def test_default_in_filter_argument(self):
        """Default used in filter argument."""
        t = PromptTemplate("{{ items | join(sep | default(', ')) }}")
        result = t.validate()
        assert result.required == {"items"}
        assert result.optional == {"sep"}
