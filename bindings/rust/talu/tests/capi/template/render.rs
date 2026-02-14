//! Tests for `talu_template_render`.
//!
//! Validates Jinja2 template rendering with variable substitution,
//! control flow, strict mode, and error handling.

use crate::capi::template::common::render_template;

// ===========================================================================
// Basic rendering
// ===========================================================================

/// Simple variable substitution: "Hello {{ name }}" + {"name":"World"}.
#[test]
fn render_simple_variable() {
    let result = render_template("Hello {{ name }}", r#"{"name":"World"}"#, false).unwrap();
    assert_eq!(result, "Hello World");
}

/// Multiple variables in one template.
#[test]
fn render_multiple_variables() {
    let result = render_template(
        "{{ greeting }}, {{ name }}!",
        r#"{"greeting":"Hi","name":"Alice"}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "Hi, Alice!");
}

/// Empty template renders to empty string.
#[test]
fn render_empty_template() {
    let result = render_template("", "{}", false).unwrap();
    assert_eq!(result, "");
}

/// Template with no variables renders as-is.
#[test]
fn render_static_text() {
    let result = render_template("Just static text.", "{}", false).unwrap();
    assert_eq!(result, "Just static text.");
}

// ===========================================================================
// Control flow
// ===========================================================================

/// For-loop: iterate over a list.
#[test]
fn render_for_loop() {
    let result = render_template(
        "{% for x in items %}{{ x }}{% endfor %}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "abc");
}

/// Conditional: if/else.
#[test]
fn render_conditional_true() {
    let result = render_template(
        "{% if flag %}yes{% else %}no{% endif %}",
        r#"{"flag":true}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "yes");
}

/// Conditional: false branch.
#[test]
fn render_conditional_false() {
    let result = render_template(
        "{% if flag %}yes{% else %}no{% endif %}",
        r#"{"flag":false}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "no");
}

/// Nested access: "{{ user.name }}".
#[test]
fn render_nested_object() {
    let result = render_template(
        "{{ user.name }}",
        r#"{"user":{"name":"Bob"}}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "Bob");
}

// ===========================================================================
// Strict mode
// ===========================================================================

/// Strict mode: undefined variable returns error code 601.
#[test]
fn strict_undefined_var_errors() {
    let err = render_template("Hello {{ missing }}", "{}", true).unwrap_err();
    assert_eq!(err, 601, "expected template_undefined_var (601)");
}

/// Non-strict mode: undefined variable renders as empty.
#[test]
fn non_strict_undefined_var_ok() {
    let result = render_template("Hello {{ missing }}", "{}", false).unwrap();
    assert_eq!(result, "Hello ");
}

// ===========================================================================
// Error handling
// ===========================================================================

/// Invalid JSON variables returns error code 605.
#[test]
fn invalid_json_vars_errors() {
    let err = render_template("{{ x }}", "not json", false).unwrap_err();
    assert_eq!(err, 605, "expected template_invalid_json (605)");
}

/// Syntax error in template returns error code 600.
#[test]
fn syntax_error_in_template() {
    let err = render_template("{% if %}", "{}", false).unwrap_err();
    assert_eq!(err, 600, "expected template_syntax_error (600)");
}

/// Unclosed block returns syntax error (600).
#[test]
fn unclosed_block_errors() {
    let err = render_template("{% for x in items %}{{ x }}", r#"{"items":[1]}"#, false)
        .unwrap_err();
    assert_eq!(err, 600, "expected template_syntax_error (600)");
}

// ===========================================================================
// Unicode content
// ===========================================================================

/// Unicode in variable value passes through correctly.
#[test]
fn render_unicode_variable() {
    let result = render_template(
        "{{ text }}",
        r#"{"text":"日本語 🎉 café"}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "日本語 🎉 café");
}

/// Special characters in variable value are preserved.
#[test]
fn render_special_chars() {
    let result = render_template(
        "{{ text }}",
        r#"{"text":"<b>bold</b> & \"quoted\""}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "<b>bold</b> & \"quoted\"");
}

// ===========================================================================
// Whitespace control
// ===========================================================================

/// Trim left whitespace with `{%-`.
#[test]
fn whitespace_trim_left() {
    let result = render_template("  {%- if true %} yes {%- endif %}", "{}", false).unwrap();
    assert_eq!(result, " yes");
}

/// Trim right whitespace with `-%}`.
#[test]
fn whitespace_trim_right() {
    let result = render_template("{% if true -%}  yes  {% endif %}", "{}", false).unwrap();
    assert_eq!(result, "yes  ");
}

/// Trim both sides in a for-loop produces compact output.
#[test]
fn whitespace_trim_for_loop() {
    let tmpl = concat!(
        "{%- for x in items -%}",
        "{{ x }}",
        "{%- endfor -%}",
    );
    let result =
        render_template(tmpl, r#"{"items":["a","b","c"]}"#, false).unwrap();
    assert_eq!(result, "abc");
}

/// Expression trim: `{{-` and `-}}`.
#[test]
fn whitespace_trim_expression() {
    let result = render_template("  {{- name -}}  ", r#"{"name":"X"}"#, false).unwrap();
    assert_eq!(result, "X");
}

// ===========================================================================
// Filters
// ===========================================================================

/// `upper` filter converts to uppercase.
#[test]
fn filter_upper() {
    let result = render_template("{{ name | upper }}", r#"{"name":"hello"}"#, false).unwrap();
    assert_eq!(result, "HELLO");
}

/// `lower` filter converts to lowercase.
#[test]
fn filter_lower() {
    let result = render_template("{{ name | lower }}", r#"{"name":"HELLO"}"#, false).unwrap();
    assert_eq!(result, "hello");
}

/// `default` filter provides fallback for undefined variables.
#[test]
fn filter_default() {
    let result =
        render_template("{{ missing | default('N/A') }}", "{}", false).unwrap();
    assert_eq!(result, "N/A");
}

/// `default` filter does nothing when variable is defined.
#[test]
fn filter_default_defined() {
    let result = render_template(
        "{{ val | default('N/A') }}",
        r#"{"val":"present"}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "present");
}

/// `trim` filter strips leading/trailing whitespace.
#[test]
fn filter_trim() {
    let result =
        render_template("{{ text | trim }}", r#"{"text":"  hi  "}"#, false).unwrap();
    assert_eq!(result, "hi");
}

/// `length` filter returns collection size.
#[test]
fn filter_length() {
    let result = render_template(
        "{{ items | length }}",
        r#"{"items":[1,2,3]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "3");
}

/// `join` filter concatenates list elements.
#[test]
fn filter_join() {
    let result = render_template(
        "{{ items | join(', ') }}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "a, b, c");
}

/// `capitalize` filter capitalizes first character.
#[test]
fn filter_capitalize() {
    let result =
        render_template("{{ text | capitalize }}", r#"{"text":"hello world"}"#, false).unwrap();
    assert_eq!(result, "Hello world");
}

/// `title` filter capitalizes each word.
#[test]
fn filter_title() {
    let result =
        render_template("{{ text | title }}", r#"{"text":"hello world"}"#, false).unwrap();
    assert_eq!(result, "Hello World");
}

/// `replace` filter substitutes text.
#[test]
fn filter_replace() {
    let result = render_template(
        "{{ text | replace('world', 'earth') }}",
        r#"{"text":"hello world"}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "hello earth");
}

/// Filter chain: multiple filters applied in sequence.
#[test]
fn filter_chain() {
    let result = render_template(
        "{{ text | trim | upper }}",
        r#"{"text":"  hello  "}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "HELLO");
}

/// `first` and `last` filters on a list.
#[test]
fn filter_first_last() {
    let vars = r#"{"items":["alpha","beta","gamma"]}"#;
    let first = render_template("{{ items | first }}", vars, false).unwrap();
    let last = render_template("{{ items | last }}", vars, false).unwrap();
    assert_eq!(first, "alpha");
    assert_eq!(last, "gamma");
}

/// `reverse` filter reverses a list.
#[test]
fn filter_reverse() {
    let result = render_template(
        "{{ items | reverse | join(',') }}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "c,b,a");
}

/// `int` filter converts string to integer.
#[test]
fn filter_int() {
    let result =
        render_template("{{ val | int + 1 }}", r#"{"val":"41"}"#, false).unwrap();
    assert_eq!(result, "42");
}

// ===========================================================================
// Loop variables
// ===========================================================================

/// `loop.index` is 1-based.
#[test]
fn loop_index() {
    let result = render_template(
        "{% for x in items %}{{ loop.index }}{% endfor %}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "123");
}

/// `loop.index0` is 0-based.
#[test]
fn loop_index0() {
    let result = render_template(
        "{% for x in items %}{{ loop.index0 }}{% endfor %}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "012");
}

/// `loop.first` and `loop.last` boolean flags.
#[test]
fn loop_first_last() {
    let tmpl = concat!(
        "{% for x in items %}",
        "{% if loop.first %}[{% endif %}",
        "{{ x }}",
        "{% if loop.last %}]{% endif %}",
        "{% endfor %}",
    );
    let result =
        render_template(tmpl, r#"{"items":["a","b","c"]}"#, false).unwrap();
    assert_eq!(result, "[abc]");
}

/// `loop.length` returns total iteration count.
#[test]
fn loop_length() {
    let result = render_template(
        "{% for x in items %}{{ loop.length }}{% endfor %}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "333");
}

/// Comma-separated output using `loop.last`.
#[test]
fn loop_comma_separated() {
    let tmpl = "{% for x in items %}{{ x }}{% if not loop.last %}, {% endif %}{% endfor %}";
    let result =
        render_template(tmpl, r#"{"items":["a","b","c"]}"#, false).unwrap();
    assert_eq!(result, "a, b, c");
}

// ===========================================================================
// Set statements
// ===========================================================================

/// `{% set %}` assigns a local variable.
#[test]
fn set_statement() {
    let tmpl = "{% set greeting = 'hi' %}{{ greeting }}";
    let result = render_template(tmpl, "{}", false).unwrap();
    assert_eq!(result, "hi");
}

/// `{% set %}` with expression.
#[test]
fn set_with_expression() {
    let tmpl = "{% set total = a + b %}{{ total }}";
    let result = render_template(tmpl, r#"{"a":3,"b":4}"#, false).unwrap();
    assert_eq!(result, "7");
}

/// Namespace for mutable state across scopes.
#[test]
fn namespace_mutable_state() {
    let tmpl = concat!(
        "{% set ns = namespace(count=0) %}",
        "{% for x in items %}{% set ns.count = ns.count + 1 %}{% endfor %}",
        "{{ ns.count }}",
    );
    let result =
        render_template(tmpl, r#"{"items":["a","b","c"]}"#, false).unwrap();
    assert_eq!(result, "3");
}

// ===========================================================================
// Macros
// ===========================================================================

/// Basic macro definition and call.
#[test]
fn macro_basic() {
    let tmpl = concat!(
        "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}",
        "{{ greet('World') }}",
    );
    let result = render_template(tmpl, "{}", false).unwrap();
    assert_eq!(result, "Hello World!");
}

/// Macro with default parameter.
#[test]
fn macro_default_param() {
    let tmpl = concat!(
        "{% macro greet(name='stranger') %}Hi {{ name }}{% endmacro %}",
        "{{ greet() }} {{ greet('Alice') }}",
    );
    let result = render_template(tmpl, "{}", false).unwrap();
    assert_eq!(result, "Hi stranger Hi Alice");
}

// ===========================================================================
// Complex expressions
// ===========================================================================

/// Logical `and` in conditional.
#[test]
fn expr_logical_and() {
    let result = render_template(
        "{% if a and b %}both{% else %}not both{% endif %}",
        r#"{"a":true,"b":true}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "both");
}

/// Logical `or` in conditional.
#[test]
fn expr_logical_or() {
    let result = render_template(
        "{% if a or b %}either{% else %}neither{% endif %}",
        r#"{"a":false,"b":true}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "either");
}

/// `not` operator.
#[test]
fn expr_not() {
    let result = render_template(
        "{% if not flag %}negated{% endif %}",
        r#"{"flag":false}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "negated");
}

/// Ternary (inline if) expression.
#[test]
fn expr_ternary() {
    let result = render_template(
        "{{ 'yes' if flag else 'no' }}",
        r#"{"flag":true}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "yes");
}

/// `in` operator for membership test.
#[test]
fn expr_in_operator() {
    let result = render_template(
        "{% if 'b' in items %}found{% else %}missing{% endif %}",
        r#"{"items":["a","b","c"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "found");
}

/// `not in` operator.
#[test]
fn expr_not_in() {
    let result = render_template(
        "{% if 'z' not in items %}absent{% endif %}",
        r#"{"items":["a","b"]}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "absent");
}

/// String concatenation with `~` operator.
#[test]
fn expr_string_concat() {
    let result = render_template(
        "{{ first ~ ' ' ~ last }}",
        r#"{"first":"John","last":"Doe"}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "John Doe");
}

/// Arithmetic in expressions.
#[test]
fn expr_arithmetic() {
    let result = render_template(
        "{{ (a + b) * 2 }}",
        r#"{"a":3,"b":4}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "14");
}

/// Comparison operators in conditional.
#[test]
fn expr_comparison() {
    let result = render_template(
        "{% if x > 5 %}big{% elif x == 5 %}five{% else %}small{% endif %}",
        r#"{"x":10}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "big");
}

/// Elif chain.
#[test]
fn elif_chain() {
    let tmpl = concat!(
        "{% if val == 1 %}one",
        "{% elif val == 2 %}two",
        "{% elif val == 3 %}three",
        "{% else %}other{% endif %}",
    );
    assert_eq!(render_template(tmpl, r#"{"val":2}"#, false).unwrap(), "two");
    assert_eq!(render_template(tmpl, r#"{"val":3}"#, false).unwrap(), "three");
    assert_eq!(render_template(tmpl, r#"{"val":9}"#, false).unwrap(), "other");
}

// ===========================================================================
// String methods
// ===========================================================================

/// `.upper()` string method.
#[test]
fn method_upper() {
    let result =
        render_template("{{ name.upper() }}", r#"{"name":"hello"}"#, false).unwrap();
    assert_eq!(result, "HELLO");
}

/// `.lower()` string method.
#[test]
fn method_lower() {
    let result =
        render_template("{{ name.lower() }}", r#"{"name":"HELLO"}"#, false).unwrap();
    assert_eq!(result, "hello");
}

/// `.strip()` string method.
#[test]
fn method_strip() {
    let result =
        render_template("{{ text.strip() }}", r#"{"text":"  hi  "}"#, false).unwrap();
    assert_eq!(result, "hi");
}

// ===========================================================================
// Comments
// ===========================================================================

/// Comments are stripped from output.
#[test]
fn comments_stripped() {
    let result = render_template(
        "before{# this is a comment #}after",
        "{}",
        false,
    )
    .unwrap();
    assert_eq!(result, "beforeafter");
}

/// Multi-line comment is stripped.
#[test]
fn multiline_comment() {
    let tmpl = "A{# line1\nline2\nline3 #}B";
    let result = render_template(tmpl, "{}", false).unwrap();
    assert_eq!(result, "AB");
}

// ===========================================================================
// Built-in functions
// ===========================================================================

/// `range()` generates a number sequence.
#[test]
fn builtin_range() {
    let result = render_template(
        "{% for i in range(4) %}{{ i }}{% endfor %}",
        "{}",
        false,
    )
    .unwrap();
    assert_eq!(result, "0123");
}

/// `range(start, stop)` with explicit bounds.
#[test]
fn builtin_range_start_stop() {
    let result = render_template(
        "{% for i in range(2, 5) %}{{ i }}{% endfor %}",
        "{}",
        false,
    )
    .unwrap();
    assert_eq!(result, "234");
}

// ===========================================================================
// Nested control flow
// ===========================================================================

/// Nested for loops.
#[test]
fn nested_for_loops() {
    let tmpl = concat!(
        "{% for row in matrix %}",
        "[{% for val in row %}{{ val }}{% if not loop.last %},{% endif %}{% endfor %}]",
        "{% endfor %}",
    );
    let vars = r#"{"matrix":[[1,2],[3,4]]}"#;
    let result = render_template(tmpl, vars, false).unwrap();
    assert_eq!(result, "[1,2][3,4]");
}

/// If inside for loop.
#[test]
fn conditional_inside_loop() {
    let tmpl = "{% for x in items %}{% if x > 2 %}{{ x }}{% endif %}{% endfor %}";
    let result =
        render_template(tmpl, r#"{"items":[1,2,3,4]}"#, false).unwrap();
    assert_eq!(result, "34");
}

/// For loop with inline condition filter.
#[test]
fn for_with_condition() {
    let tmpl = "{% for x in items if x > 2 %}{{ x }}{% endfor %}";
    let result =
        render_template(tmpl, r#"{"items":[1,2,3,4]}"#, false).unwrap();
    assert_eq!(result, "34");
}

// ===========================================================================
// Type tests
// ===========================================================================

/// `is defined` test.
#[test]
fn test_is_defined() {
    let tmpl = "{% if val is defined %}yes{% else %}no{% endif %}";
    assert_eq!(render_template(tmpl, r#"{"val":1}"#, false).unwrap(), "yes");
    assert_eq!(render_template(tmpl, "{}", false).unwrap(), "no");
}

/// `is string` / `is number` type tests.
#[test]
fn test_is_type() {
    assert_eq!(
        render_template("{{ val is string }}", r#"{"val":"hi"}"#, false).unwrap(),
        "True"
    );
    assert_eq!(
        render_template("{{ val is number }}", r#"{"val":42}"#, false).unwrap(),
        "True"
    );
}

// ===========================================================================
// Dictionary iteration
// ===========================================================================

/// Iterate over dictionary items.
#[test]
fn dict_iteration() {
    let tmpl = "{% for k, v in data | dictsort %}{{ k }}={{ v }} {% endfor %}";
    let vars = r#"{"data":{"b":2,"a":1}}"#;
    let result = render_template(tmpl, vars, false).unwrap();
    assert_eq!(result, "a=1 b=2 ");
}

// ===========================================================================
// Filter blocks
// ===========================================================================

/// `{% filter upper %}` block applies filter to enclosed content.
#[test]
fn filter_block() {
    let result = render_template(
        "{% filter upper %}hello world{% endfilter %}",
        "{}",
        false,
    )
    .unwrap();
    assert_eq!(result, "HELLO WORLD");
}

// ===========================================================================
// Input sanitization & adversarial templates
// ===========================================================================

/// Very long variable value (10K chars) renders without crash.
#[test]
fn sanitize_long_variable_value() {
    let long_val = "x".repeat(10_000);
    let vars = format!(r#"{{"text":"{}"}}"#, long_val);
    let result = render_template("{{ text }}", &vars, false).unwrap();
    assert_eq!(result.len(), 10_000);
}

/// Very long template literal (10K chars) renders correctly.
#[test]
fn sanitize_long_template_literal() {
    let long_tmpl = "A".repeat(10_000);
    let result = render_template(&long_tmpl, "{}", false).unwrap();
    assert_eq!(result.len(), 10_000);
}

/// Large loop (1000 iterations) completes without crash.
#[test]
fn sanitize_large_loop() {
    let items: Vec<String> = (0..1000).map(|i| i.to_string()).collect();
    let vars = format!(r#"{{"items":[{}]}}"#, items.join(","));
    let result = render_template(
        "{% for x in items %}{{ x }}{% endfor %}",
        &vars,
        false,
    )
    .unwrap();
    // Should contain all numbers 0-999.
    assert!(result.contains("0"));
    assert!(result.contains("999"));
}

/// Deeply nested conditionals (10 levels) parse and evaluate.
#[test]
fn sanitize_nested_conditionals() {
    // Build 10 levels of nested if/endif.
    let mut tmpl = String::new();
    for _ in 0..10 {
        tmpl.push_str("{% if true %}");
    }
    tmpl.push_str("deep");
    for _ in 0..10 {
        tmpl.push_str("{% endif %}");
    }
    let result = render_template(&tmpl, "{}", false).unwrap();
    assert_eq!(result, "deep");
}

/// Template with Jinja2 delimiters in variable value (not interpreted).
#[test]
fn sanitize_delimiters_in_value() {
    let vars = r#"{"text":"{% if true %}injected{% endif %}"}"#;
    let result = render_template("{{ text }}", vars, false).unwrap();
    // The value is literal text, NOT interpreted as Jinja2.
    assert_eq!(result, "{% if true %}injected{% endif %}");
}

/// HTML-like content in variable value passes through unescaped.
#[test]
fn sanitize_html_in_value() {
    let vars = r#"{"text":"<script>alert('xss')</script>"}"#;
    let result = render_template("{{ text }}", vars, false).unwrap();
    assert_eq!(result, "<script>alert('xss')</script>");
}

/// JSON with null value renders as empty or "None".
#[test]
fn sanitize_null_json_value() {
    let result = render_template(
        "{{ val }}",
        r#"{"val":null}"#,
        false,
    )
    .unwrap();
    // Null renders as empty string in non-strict mode.
    assert!(result.is_empty() || result == "None", "got: {result:?}");
}

/// Very large number in JSON.
#[test]
fn sanitize_large_number() {
    let result = render_template(
        "{{ val + 1 }}",
        r#"{"val":999999999}"#,
        false,
    )
    .unwrap();
    assert_eq!(result, "1000000000");
}

/// Boolean values render correctly.
#[test]
fn sanitize_boolean_values() {
    assert_eq!(
        render_template("{{ val }}", r#"{"val":true}"#, false).unwrap(),
        "True"
    );
    assert_eq!(
        render_template("{{ val }}", r#"{"val":false}"#, false).unwrap(),
        "False"
    );
}

/// Empty JSON object renders variables as empty (non-strict).
#[test]
fn sanitize_empty_vars_non_strict() {
    let result = render_template(
        "{{ a }}{{ b }}{{ c }}",
        "{}",
        false,
    )
    .unwrap();
    assert_eq!(result, "");
}

/// Multiple errors: bad template with bad JSON still returns an error.
#[test]
fn sanitize_multiple_errors() {
    let err = render_template("{% if %}", "not json", false).unwrap_err();
    // Should get some error (template or JSON parse), not crash.
    assert_ne!(err, 0);
}

// ===========================================================================
// Error code consistency
// ===========================================================================

/// Different syntax errors all return error code 600.
#[test]
fn error_code_consistency_syntax() {
    let syntax_errors = [
        "{% if %}",
        "{% for %}",
        "{% endfor %}",
        "{{ }}",
        "{%  %}",
    ];
    for tmpl in syntax_errors {
        let err = render_template(tmpl, "{}", false).unwrap_err();
        assert_eq!(err, 600, "expected 600 for template: {tmpl:?}, got {err}");
    }
}

/// Different invalid JSON inputs all return error code 605.
#[test]
fn error_code_consistency_json() {
    let bad_json = [
        "not json",
        "{invalid}",
        r#"{"key":}"#, // missing value
    ];
    for json in bad_json {
        let err = render_template("{{ x }}", json, false).unwrap_err();
        assert_eq!(err, 605, "expected 605 for JSON: {json:?}, got {err}");
    }
}

// ===========================================================================
// `not X is defined` operator precedence
// ===========================================================================
//
// In Jinja2, `is` has higher precedence than `not`:
//   `not X is defined` = `not (X is defined)`
//
// If the engine incorrectly parses it as `(not X) is defined`, the result
// is always true (since `not X` produces a defined boolean value), which
// causes `{% set X = false %}` to execute even when X is already defined.
// Affects: deepseek-ai/DeepSeek-R1 templates (8 chat_template failures)
// where add_generation_prompt gets overridden to false.

/// `not X is defined` must parse as `not (X is defined)`.
///
/// When `x` is defined, `not x is defined` must be false. If the engine
/// parses it as `(not x) is defined`, the result is always true because
/// `not x` produces a defined value.
#[test]
fn not_is_defined_precedence() {
    // x=true is defined, so `not x is defined` should be `not true` = false.
    // The template should NOT execute the set block → output "true".
    let tmpl = concat!(
        "{% if not x is defined %}{% set x = false %}{% endif %}",
        "{{ x }}",
    );
    let result = render_template(tmpl, r#"{"x": true}"#, false).unwrap();
    assert_eq!(
        result.trim(), "True",
        "`not x is defined` must parse as `not (x is defined)`, \
         but x was overridden to false, got: {result:?}"
    );
}

/// `not X is defined` with false value: variable is still defined.
///
/// Even when `x=false`, `x` is defined. `not x is defined` should be
/// `not (false is defined)` = `not true` = false. The set block must not
/// execute.
#[test]
fn not_is_defined_false_value_still_defined() {
    let tmpl = concat!(
        "{% if not x is defined %}{% set x = 42 %}{% endif %}",
        "{{ x }}",
    );
    let result = render_template(tmpl, r#"{"x": false}"#, false).unwrap();
    assert_eq!(
        result.trim(), "False",
        "x=false is defined; `not x is defined` must be false, got: {result:?}"
    );
}

// ===========================================================================
// `is string` test
// ===========================================================================
//
// Jinja2's `is string` test checks if a value is a string type.
// Templates like Qwen3-VL use `{% if content is string %}` to distinguish
// plain text from structured content arrays.
// Affects: Qwen3-VL-Thinking template rendering (macro dispatches on string
// vs array content).

/// `is string` must return true for string values.
#[test]
fn is_string_true_for_strings() {
    let tmpl = "{% if val is string %}yes{% else %}no{% endif %}";
    let result = render_template(tmpl, r#"{"val": "hello"}"#, false).unwrap();
    assert_eq!(result, "yes", "`is string` must be true for string values, got: {result:?}");
}

/// `is string` must return false for non-string values.
#[test]
fn is_string_false_for_non_strings() {
    let tmpl = "{% if val is string %}yes{% else %}no{% endif %}";
    // Array
    let result = render_template(tmpl, r#"{"val": [1, 2, 3]}"#, false).unwrap();
    assert_eq!(result, "no", "`is string` must be false for arrays, got: {result:?}");
    // Number
    let result = render_template(tmpl, r#"{"val": 42}"#, false).unwrap();
    assert_eq!(result, "no", "`is string` must be false for numbers, got: {result:?}");
}

// ===========================================================================
// Slice reversal: `list[::-1]`
// ===========================================================================
//
// Jinja2 supports Python-style slice reversal. Templates like Qwen3-VL
// iterate messages in reverse to find the last user query index:
//   `{% for message in messages[::-1] %}`
// Affects: Qwen3-VL-Thinking template (reverse loop for multi-step tool
// detection).

/// `list[::-1]` must reverse the list.
#[test]
fn slice_reversal() {
    let tmpl = concat!(
        "{%- for item in items[::-1] -%}",
        "{{ item }}",
        "{%- if not loop.last %},{% endif -%}",
        "{%- endfor -%}",
    );
    let result = render_template(tmpl, r#"{"items": ["a", "b", "c"]}"#, false).unwrap();
    assert_eq!(result, "c,b,a", "[::-1] must reverse the list, got: {result:?}");
}

// ===========================================================================
// String methods: split, startswith, endswith, strip
// ===========================================================================
//
// Templates use Python string methods on template values:
//   content.split('</think>')[-1]  — split and negative indexing
//   content.startswith('<tool_response>')
//   content.strip('\n')
// Affects: Qwen3-VL-Thinking and DeepSeek-R1 templates.

/// `str.split(sep)[-1]` must return the last segment.
#[test]
fn string_split_negative_index() {
    let tmpl = "{{ text.split('|')[-1] }}";
    let result = render_template(tmpl, r#"{"text": "a|b|c"}"#, false).unwrap();
    assert_eq!(result, "c", "split('|')[-1] must return last segment, got: {result:?}");
}

/// `str.startswith(prefix)` must test the prefix.
#[test]
fn string_startswith() {
    let tmpl = "{% if text.startswith('hello') %}yes{% else %}no{% endif %}";
    let result = render_template(tmpl, r#"{"text": "hello world"}"#, false).unwrap();
    assert_eq!(result, "yes", "startswith must match prefix, got: {result:?}");
    let result = render_template(tmpl, r#"{"text": "world hello"}"#, false).unwrap();
    assert_eq!(result, "no", "startswith must not match non-prefix, got: {result:?}");
}

/// `str.endswith(suffix)` must test the suffix.
#[test]
fn string_endswith() {
    let tmpl = "{% if text.endswith('world') %}yes{% else %}no{% endif %}";
    let result = render_template(tmpl, r#"{"text": "hello world"}"#, false).unwrap();
    assert_eq!(result, "yes", "endswith must match suffix, got: {result:?}");
}

/// `str.strip(chars)` must remove specified characters from both ends.
#[test]
fn string_strip_with_argument() {
    let tmpl = "{{ text.strip('\\n') }}";
    let result = render_template(tmpl, r#"{"text": "\nhello\n"}"#, false).unwrap();
    assert_eq!(result, "hello", "strip('\\n') must remove newlines, got: {result:?}");
}
