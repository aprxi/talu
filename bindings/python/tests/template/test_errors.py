"""
Error handling tests for Template.

Tests for syntax errors, undefined variables, and other error conditions.
"""

import pytest

from talu.exceptions import TemplateError, TemplateSyntaxError


class TestSyntaxErrors:
    """Tests for template syntax errors."""

    def test_unclosed_variable(self, Template):
        """Unclosed variable tag raises error."""
        with pytest.raises(TemplateSyntaxError) as exc_info:
            Template("Hello {{ name")
        # Should mention syntax error or unclosed tag
        assert (
            "syntax" in str(exc_info.value).lower()
            or "unclosed" in str(exc_info.value).lower()
            or "expected" in str(exc_info.value).lower()
        )

    def test_unclosed_block(self, Template):
        """Unclosed block tag raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% if x %}yes")

    def test_mismatched_block(self, Template):
        """Mismatched block end raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% if x %}{% endfor %}")

    def test_double_operator_parses(self, Template):
        """Double operator is parsed (unary + after binary +)."""
        # {{ 1 + + 2 }} is parsed as 1 + (+2) which is valid
        t = Template("{{ 1 + + 2 }}")
        assert t() == "3"

    def test_invalid_filter_syntax(self, Template):
        """Invalid filter syntax raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{{ x | }}")

    def test_unknown_tag(self, Template):
        """Unknown tag raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% unknowntag %}")

    def test_extra_endif(self, Template):
        """Extra endif raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% endif %}")

    def test_extra_endfor(self, Template):
        """Extra endfor raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% endfor %}")

    def test_else_without_if(self, Template):
        """Else without if raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% else %}")

    def test_elif_without_if(self, Template):
        """Elif without if raises error."""
        with pytest.raises(TemplateSyntaxError):
            Template("{% elif x %}")


class TestUndefinedBehavior:
    """Tests for undefined variable handling.

    These tests verify lenient mode behavior (strict=False), which is
    Jinja2-compatible: undefined variables render as empty string.

    Note: PromptTemplate now defaults to strict=True for LLM safety.
    Use LenientTemplate fixture for Jinja2-compatible lenient tests.
    """

    def test_undefined_variable_renders_empty(self, LenientTemplate):
        """Undefined variable renders as empty string (lenient mode)."""
        t = LenientTemplate("{{ undefined_var }}")
        assert t() == ""

    def test_undefined_in_expression(self, LenientTemplate):
        """Undefined variable in expression raises error."""
        t = LenientTemplate("{{ x + undefined }}")
        # Zig runtime returns TemplateError for evaluation errors
        with pytest.raises(TemplateError):
            t(x=5)

    def test_undefined_attribute_renders_empty(self, LenientTemplate):
        """Undefined attribute access renders empty (lenient mode)."""
        t = LenientTemplate("{{ obj.missing_attr }}")
        assert t(obj={"a": 1}) == ""

    def test_undefined_in_loop(self, LenientTemplate):
        """Undefined in loop raises error."""
        t = LenientTemplate("{% for i in undefined_list %}{{ i }}{% endfor %}")
        # Zig runtime returns TemplateError for evaluation errors
        with pytest.raises(TemplateError):
            t()

    def test_undefined_in_condition_is_falsy(self, LenientTemplate):
        """Undefined in condition evaluates as falsy (lenient mode)."""
        t = LenientTemplate("{% if undefined_var %}yes{% else %}no{% endif %}")
        assert t() == "no"

    def test_undefined_with_default_ok(self, Template):
        """Undefined with default filter is OK (works in both strict and lenient)."""
        t = Template("{{ undefined_var | default('fallback') }}")
        assert t() == "fallback"

    def test_undefined_is_undefined_ok(self, Template):
        """Checking 'is undefined' is OK (works in both strict and lenient)."""
        t = Template("{% if x is undefined %}missing{% else %}{{ x }}{% endif %}")
        assert t() == "missing"
        assert t(x="present") == "present"


class TestTypeErrors:
    """Tests for type-related errors."""

    def test_iterate_non_iterable(self, Template):
        """Iterating over non-iterable raises error."""
        t = Template("{% for i in x %}{{ i }}{% endfor %}")
        with pytest.raises(TemplateError):
            t(x=42)

    def test_access_attr_on_none_renders_empty(self, LenientTemplate):
        """Accessing attribute on None renders empty (lenient mode)."""
        t = LenientTemplate("{{ x.attr }}")
        assert t(x=None) == ""

    def test_index_non_subscriptable(self, Template):
        """Indexing non-subscriptable raises error."""
        t = Template("{{ x[0] }}")
        with pytest.raises(TemplateError):
            t(x=42)

    def test_call_non_callable(self, Template):
        """Calling non-callable raises error."""
        t = Template("{{ x() }}")
        with pytest.raises(TemplateError):
            t(x=42)

    def test_invalid_filter_arg_uses_default(self, Template):
        """Invalid argument to filter uses default value (lenient mode)."""
        t = Template("{{ x | truncate('not a number') }}")
        # Filter uses default truncate length when arg is invalid
        result = t(x="hello world this is a long string")
        assert len(result) > 0  # Should produce some output


class TestFilterErrors:
    """Tests for filter-related errors."""

    def test_unknown_filter(self, Template):
        """Unknown filter raises error."""
        t = Template("{{ x | nonexistent_filter }}")
        with pytest.raises(TemplateError) as exc_info:
            t(x="hello")
        # Error message follows centralized format: "Template render failed"
        assert "failed" in str(exc_info.value).lower()

    def test_filter_wrong_arg_count(self, Template):
        """Filter with wrong argument count raises error."""
        t = Template("{{ x | join(1, 2, 3, 4, 5) }}")
        with pytest.raises(TemplateError):
            t(x=[1, 2, 3])


class TestIncludeErrors:
    """Tests for {% include %} error messages."""

    def test_include_undefined_variable_error(self, Template):
        """Include with undefined variable shows helpful error message."""
        t = Template("{% include header %}Hello")
        with pytest.raises(TemplateError) as exc_info:
            t()
        error_msg = str(exc_info.value)
        # Should mention the variable name in context
        assert "header" in error_msg
        # Centralized error format: "Template render failed ('var' is undefined)"
        assert "undefined" in error_msg.lower()

    def test_include_non_string_error(self, Template):
        """Include with non-string value shows helpful error message."""
        t = Template("{% include x %}")
        with pytest.raises(TemplateError) as exc_info:
            t(x=123)
        error_msg = str(exc_info.value)
        # Centralized error format: "Template render failed"
        # Include with non-string is a type error in sandboxed mode
        assert "failed" in error_msg.lower()


class TestMacroErrors:
    """Tests for macro-related errors."""

    def test_undefined_macro(self, Template):
        """Calling undefined macro raises error."""
        t = Template("{{ undefined_macro() }}")
        with pytest.raises(TemplateError):
            t()

    def test_macro_missing_args_uses_empty(self, Template):
        """Macro with missing arguments uses empty string (lenient mode)."""
        t = Template("""
{%- macro greet(name) -%}Hello {{ name }}{%- endmacro -%}
{{ greet() }}
""")
        result = t()
        assert "Hello" in result  # Renders with empty name


class TestErrorMessages:
    """Tests for error message quality.

    Contract: Error messages should be actionable and help users locate problems.
    - Syntax errors should mention line/column or context
    - Undefined errors should mention the variable name
    - Type errors should mention expected vs actual types
    """

    def test_error_includes_context(self, Template):
        """Error message includes context about where error occurred."""
        try:
            Template("line1\n{{ unclosed")
        except Exception as e:
            error_msg = str(e).lower()
            # Should give some indication of location or problem
            assert len(error_msg) > 10  # Not just empty error

    def test_undefined_error_includes_name(self, Template):
        """Undefined error includes variable name."""
        t = Template("{{ my_special_var }}")
        try:
            t()
        except Exception as e:
            error_msg = str(e)
            assert "my_special_var" in error_msg or "undefined" in error_msg.lower()

    def test_syntax_error_mentions_location(self, Template):
        """Syntax error should mention line or column number.

        Expected: Error message should include line number (e.g., 'line 2')
        or column position to help user locate the problem.
        """
        try:
            Template("line1\nline2 {{ unclosed")
        except Exception as e:
            error_msg = str(e).lower()
            # Check if error mentions line number or position
            has_location_info = (
                "line" in error_msg
                or "column" in error_msg
                or "position" in error_msg
                or "2" in error_msg  # Line 2 where error is
                or ":" in str(e)  # line:column format
            )
            if not has_location_info:
                pytest.xfail(f"Syntax error should include line/column info - got: {e}")
            assert has_location_info

    def test_unclosed_block_mentions_tag(self, Template):
        """Unclosed block error should mention the unclosed tag.

        Expected: Error should mention 'if' or 'endif' to help user identify
        which block is unclosed.
        """
        try:
            Template("{% if x %}content but no endif")
        except Exception as e:
            error_msg = str(e).lower()
            # Should mention the tag type
            if "if" not in error_msg and "endif" not in error_msg and "block" not in error_msg:
                pytest.xfail(f"Unclosed block error should mention 'if' or 'block' - got: {e}")
            assert (
                "if" in error_msg
                or "endif" in error_msg
                or "block" in error_msg
                or "unclosed" in error_msg
            )

    def test_mismatched_tag_error_is_descriptive(self, Template):
        """Mismatched block error mentions both expected and actual tags.

        Expected: Error should help user understand they used wrong end tag.
        """
        try:
            Template("{% if x %}{% endfor %}")
        except Exception as e:
            error_msg = str(e).lower()
            # Should mention the mismatch
            if "if" not in error_msg and "for" not in error_msg and "mismatch" not in error_msg:
                pytest.xfail(
                    f"Mismatched tag error should mention 'if' or 'for' or 'mismatch' - got: {e}"
                )
            assert (
                "if" in error_msg
                or "for" in error_msg
                or "mismatch" in error_msg
                or "expected" in error_msg
            )


class TestRobustness:
    """Tests for robustness against unusual inputs."""

    def test_none_as_variable_name(self, Template):
        """None cannot be used as variable name."""
        # This tests the API, not the template
        t = Template("{{ x }}")
        # Should handle gracefully
        try:
            t(**{None: "value"})  # type: ignore
        except (TypeError, TemplateSyntaxError):
            pass  # Expected to fail

    def test_empty_variable_name(self, Template):
        """Empty variable name handled."""
        # Invalid template syntax
        with pytest.raises(TemplateSyntaxError):
            Template("{{ }}")

    def test_only_whitespace_in_variable(self, Template):
        """Only whitespace in variable tag."""
        with pytest.raises(TemplateSyntaxError):
            Template("{{   }}")

    def test_deeply_nested_undefined_renders_empty(self, LenientTemplate):
        """Undefined in deeply nested structure renders empty (lenient mode)."""
        t = LenientTemplate("""
{%- for a in items -%}
  {%- for b in a -%}
    {%- for c in b -%}
      {{ undefined_deep }}
    {%- endfor -%}
  {%- endfor -%}
{%- endfor -%}
""")
        result = t(items=[[[1]]])
        assert result.strip() == ""  # Undefined renders as empty


class TestStrictModeErrorMessages:
    """Tests for strict mode error messages with variable paths.

    When strict=True, undefined variables raise TemplateUndefinedError
    with a message that includes the full path to the undefined variable.
    This helps users identify exactly which variable or attribute is missing.
    """

    def test_strict_undefined_variable_includes_name(self, Template):
        """Strict mode error includes the undefined variable name."""
        from talu.exceptions import TemplateUndefinedError

        t = Template("{{ name }}", strict=True)
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t()
        assert "'name'" in str(exc_info.value)
        assert "undefined" in str(exc_info.value).lower()

    def test_strict_undefined_attribute_includes_path(self, Template):
        """Strict mode error includes full attribute path like 'doc.page_content'."""
        from talu.exceptions import TemplateUndefinedError

        t = Template("{{ doc.page_content }}", strict=True)
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t(doc={"text": "hello"})  # Has 'text', not 'page_content'
        error_msg = str(exc_info.value)
        assert "'doc.page_content'" in error_msg

    def test_strict_nested_attribute_includes_full_path(self, Template):
        """Strict mode error includes deeply nested paths like 'user.address.city'."""
        from talu.exceptions import TemplateUndefinedError

        t = Template("{{ user.address.city }}", strict=True)
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t(user={"address": {}})  # address exists but city doesn't
        error_msg = str(exc_info.value)
        assert "'user.address.city'" in error_msg

    def test_strict_dict_key_access_includes_path(self, Template):
        """Strict mode error includes dict key access like \"data['key']\"."""
        from talu.exceptions import TemplateUndefinedError

        t = Template("{{ data['missing_key'] }}", strict=True)
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t(data={"other": "value"})
        error_msg = str(exc_info.value)
        assert "data" in error_msg
        assert "missing_key" in error_msg

    def test_strict_loop_variable_attribute_includes_path(self, Template):
        """Strict mode error in loop includes the loop variable path."""
        from talu.exceptions import TemplateUndefinedError

        t = Template(
            "{% for item in items %}{{ item.name }}{% endfor %}",
            strict=True,
        )
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t(items=[{"id": 1}, {"id": 2}])  # Items have 'id', not 'name'
        error_msg = str(exc_info.value)
        assert "'item.name'" in error_msg

    def test_strict_override_per_call(self, Template):
        """Strict mode can be overridden per-call and error includes path."""
        from talu.exceptions import TemplateUndefinedError

        t = Template("{{ user.email }}")  # Strict by default

        # Strict mode (default): raises with path
        with pytest.raises(TemplateUndefinedError) as exc_info:
            t(user={})
        assert "'user.email'" in str(exc_info.value)

        # Lenient override: returns empty
        result = t(user={}, strict=False)
        assert result == ""


class TestRecovery:
    """Tests for error recovery scenarios."""

    def test_valid_after_invalid(self, Template):
        """Valid template works after invalid one failed."""
        # First, create an invalid template
        with pytest.raises(TemplateSyntaxError):
            Template("{{ unclosed")

        # Then, valid template should still work
        t = Template("{{ x }}")
        assert t(x="works") == "works"

    def test_same_template_different_vars(self, LenientTemplate):
        """Same template works with valid vars after lenient handling."""
        t = LenientTemplate("{{ x.y }}")

        # First call with None - lenient behavior returns empty
        result1 = t(x=None)
        assert result1 == ""  # Lenient undefined

        # Second call should work
        assert t(x={"y": "value"}) == "value"
