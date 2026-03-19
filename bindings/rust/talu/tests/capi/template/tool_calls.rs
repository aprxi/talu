//! Tests for chat template rendering with tool definitions (Qwen3.5 XML format).
//!
//! Validates that chat templates correctly render tool definitions and format
//! instructions for models that use XML-based tool call syntax.

use crate::capi::template::common::render_template;

// ===========================================================================
// Qwen3.5-style template fixture
// ===========================================================================

/// Simplified Qwen3.5 template with tool support.
///
/// When `tools` is provided and non-empty, renders a system message with:
/// - Tool definitions inside `<tools>...</tools>` block
/// - XML format instructions for `<tool_call>` syntax
const QWEN35_TOOLS_TEMPLATE: &str = concat!(
    "{%- if tools and tools is iterable and tools is not mapping %}",
    "{{- '<|im_start|>system\\n' }}",
    "{{- '# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>' }}",
    "{%- for tool in tools %}",
    "{{- '\\n' }}",
    "{{- tool | tojson }}",
    "{%- endfor %}",
    "{{- '\\n</tools>' }}",
    "{{- '\\n\\nIf you choose to call a function ONLY reply in the following format:\\n\\n",
    "<tool_call>\\n<function=example_function_name>\\n",
    "<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n",
    "</function>\\n</tool_call>' }}",
    "{%- if messages[0].role == 'system' %}",
    "{%- set content = messages[0].content | trim %}",
    "{%- if content %}",
    "{{- '\\n\\n' + content }}",
    "{%- endif %}",
    "{%- endif %}",
    "{{- '<|im_end|>\\n' }}",
    "{%- else %}",
    "{%- if messages[0].role == 'system' %}",
    "{{- '<|im_start|>system\\n' + messages[0].content | trim + '<|im_end|>\\n' }}",
    "{%- endif %}",
    "{%- endif %}",
    // Render non-system messages.
    "{%- for message in messages %}",
    "{%- if message.role != 'system' or (not tools) %}",
    "{%- if not (message.role == 'system' and tools) %}",
    "{{- '<|im_start|>' + message.role + '\\n' }}",
    "{{- message.content }}",
    "{{- '<|im_end|>\\n' }}",
    "{%- endif %}",
    "{%- endif %}",
    "{%- endfor %}",
    "{%- if add_generation_prompt %}",
    "{{- '<|im_start|>assistant\\n' }}",
    "{%- endif %}",
);

// ===========================================================================
// Tool definition rendering
// ===========================================================================

/// Template with tools renders `<tools>` section containing tool definitions.
#[test]
fn tools_rendered_in_system_message() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "What is the area of a triangle with base 10 and height 5?"}],
        "tools": [
            {"type": "function", "function": {"name": "calculate_area", "parameters": {"type": "object", "properties": {"base": {"type": "number"}, "height": {"type": "number"}}}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        result.contains("<tools>"),
        "must contain <tools> section, got: {result:?}"
    );
    assert!(
        result.contains("</tools>"),
        "must contain closing </tools>, got: {result:?}"
    );
    assert!(
        result.contains("calculate_area"),
        "must contain tool name, got: {result:?}"
    );
}

/// Template with tools includes XML format instructions.
#[test]
fn tools_include_format_instructions() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        result.contains("<tool_call>"),
        "must contain <tool_call> format instruction, got: {result:?}"
    );
    assert!(
        result.contains("<function=example_function_name>"),
        "must contain <function=...> format instruction, got: {result:?}"
    );
    assert!(
        result.contains("<parameter=example_parameter_1>"),
        "must contain <parameter=...> format instruction, got: {result:?}"
    );
}

/// Multiple tools are each rendered inside the `<tools>` block.
#[test]
fn multiple_tools_rendered() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object"}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        result.contains("get_weather"),
        "must contain first tool, got: {result:?}"
    );
    assert!(
        result.contains("search_web"),
        "must contain second tool, got: {result:?}"
    );
}

/// System message content is appended after tool definitions.
#[test]
fn system_message_appended_after_tools() {
    let vars = r#"{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    // System content should appear in the same system message block, after tools.
    assert!(
        result.contains("You are a helpful assistant."),
        "must contain system content, got: {result:?}"
    );
    // The tools section should appear before the system content.
    let tools_pos = result.find("</tools>").expect("must have </tools>");
    let system_pos = result
        .find("You are a helpful assistant.")
        .expect("must have system content");
    assert!(
        tools_pos < system_pos,
        "tools must appear before system content"
    );
}

// ===========================================================================
// Without tools
// ===========================================================================

/// Template without tools renders normally (no `<tools>` section).
#[test]
fn no_tools_renders_normally() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        !result.contains("<tools>"),
        "must not contain <tools> when no tools, got: {result:?}"
    );
    assert!(
        result.contains("Hello"),
        "must contain user message, got: {result:?}"
    );
}

/// Empty tools array renders normally (no `<tools>` section).
#[test]
fn empty_tools_renders_normally() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        !result.contains("<tools>"),
        "must not contain <tools> for empty tools array, got: {result:?}"
    );
}

// ===========================================================================
// Generation prompt with tools
// ===========================================================================

/// With tools and add_generation_prompt, assistant header is appended.
#[test]
fn tools_with_generation_prompt() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    assert!(
        result.ends_with("<|im_start|>assistant\n"),
        "must end with assistant generation prompt, got: ...{}",
        &result[result.len().saturating_sub(60)..],
    );
}

// ===========================================================================
// Flat → nested tool format normalization
// ===========================================================================

/// Flat-format tools (our API format) must produce the same template output
/// as nested OpenAI format, since the model was trained on nested format.
///
/// The server normalizes flat → nested in `buildEffectiveContext` before
/// passing tools to the template.  This test verifies that the *nested*
/// format renders the expected `<tools>` block content — i.e. the structure
/// the model was trained on.
#[test]
fn nested_format_tools_render_function_key() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    // Template renders `{{ tool | tojson }}` which includes the nested "function" key.
    assert!(
        result.contains("\"function\""),
        "nested format must contain \"function\" key in rendered output, got: {result:?}"
    );
    assert!(
        result.contains("get_weather"),
        "must contain tool name"
    );
}

/// Flat-format tools render WITHOUT the "function" wrapper — confirming that
/// the template does NOT normalize on its own.  The engine must normalize
/// before the template sees the tools.
#[test]
fn flat_format_tools_lack_function_wrapper() {
    let vars = r#"{
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [
            {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
        ],
        "add_generation_prompt": true
    }"#;
    let result = render_template(QWEN35_TOOLS_TEMPLATE, vars, false).unwrap();
    // Without normalization the template renders flat JSON (no "function" wrapper).
    assert!(
        result.contains("get_weather"),
        "flat format must still contain tool name"
    );
    // The rendered JSON has "name" at the top level (flat), not nested under
    // a "function" key.  `"function": {` would indicate the nested wrapper.
    assert!(
        !result.contains("\"function\": {") && !result.contains("\"function\":{"),
        "flat format must NOT contain nested \"function\" wrapper (template doesn't normalize), got: {result:?}"
    );
}
