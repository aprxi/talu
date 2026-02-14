//! Tests for `talu_apply_chat_template_string`.
//!
//! Validates chat template rendering with various message configurations,
//! generation prompts, BOS/EOS tokens, and error handling.

use crate::capi::template::common::{
    apply_chat_template, BOS_EOS_TEMPLATE, GENERATION_TEMPLATE, MULTITURN_MSGS,
    SIMPLE_TEMPLATE, SINGLE_USER_MSG,
};

// ===========================================================================
// Basic message rendering
// ===========================================================================

/// Single user message renders correctly.
#[test]
fn single_user_message() {
    let result = apply_chat_template(SIMPLE_TEMPLATE, SINGLE_USER_MSG, false, "", "").unwrap();
    assert!(result.contains("user: Hello"), "got: {result:?}");
}

/// Multi-turn conversation renders all messages in order.
#[test]
fn multiturn_conversation() {
    let result = apply_chat_template(SIMPLE_TEMPLATE, MULTITURN_MSGS, false, "", "").unwrap();
    assert!(result.contains("system: You are helpful."), "missing system msg in: {result:?}");
    assert!(result.contains("user: Hi"), "missing first user msg in: {result:?}");
    assert!(result.contains("assistant: Hello!"), "missing assistant msg in: {result:?}");
    assert!(result.contains("user: How are you?"), "missing second user msg in: {result:?}");
}

// ===========================================================================
// Generation prompt
// ===========================================================================

/// add_generation_prompt=true appends assistant header.
#[test]
fn add_generation_prompt_true() {
    let result =
        apply_chat_template(GENERATION_TEMPLATE, SINGLE_USER_MSG, true, "", "").unwrap();
    assert!(
        result.ends_with("<|assistant|>"),
        "expected assistant header at end, got: {result:?}"
    );
}

/// add_generation_prompt=false does not append assistant header.
#[test]
fn add_generation_prompt_false() {
    let result =
        apply_chat_template(GENERATION_TEMPLATE, SINGLE_USER_MSG, false, "", "").unwrap();
    assert!(
        !result.ends_with("<|assistant|>"),
        "should not end with assistant header, got: {result:?}"
    );
}

// ===========================================================================
// BOS/EOS tokens
// ===========================================================================

/// BOS token appears at start of output.
#[test]
fn bos_token_injected() {
    let result =
        apply_chat_template(BOS_EOS_TEMPLATE, SINGLE_USER_MSG, false, "<s>", "</s>").unwrap();
    assert!(result.starts_with("<s>"), "expected BOS at start, got: {result:?}");
}

/// EOS token appears between messages.
#[test]
fn eos_token_between_messages() {
    let msgs = r#"[{"role":"user","content":"A"},{"role":"assistant","content":"B"}]"#;
    let result = apply_chat_template(BOS_EOS_TEMPLATE, msgs, false, "", "</s>").unwrap();
    assert!(result.contains("</s>"), "expected EOS between messages, got: {result:?}");
}

// ===========================================================================
// Unicode in messages
// ===========================================================================

/// Unicode content in messages passes through correctly.
#[test]
fn unicode_in_message_content() {
    let msgs = r#"[{"role":"user","content":"日本語 🎉 café"}]"#;
    let result = apply_chat_template(SIMPLE_TEMPLATE, msgs, false, "", "").unwrap();
    assert!(result.contains("日本語 🎉 café"), "got: {result:?}");
}

/// Special characters in content are preserved.
#[test]
fn special_chars_in_content() {
    let msgs = r#"[{"role":"user","content":"line1\nline2\ttab"}]"#;
    let result = apply_chat_template(SIMPLE_TEMPLATE, msgs, false, "", "").unwrap();
    assert!(result.contains("line1\nline2\ttab"), "got: {result:?}");
}

// ===========================================================================
// Edge cases
// ===========================================================================

/// Empty messages array produces output (template-dependent).
#[test]
fn empty_messages_array() {
    let result = apply_chat_template(SIMPLE_TEMPLATE, "[]", false, "", "").unwrap();
    // With no messages, for-loop body never executes → empty string.
    assert_eq!(result, "");
}

/// Empty content in message renders role but no content text.
#[test]
fn empty_content_in_message() {
    let msgs = r#"[{"role":"user","content":""}]"#;
    let result = apply_chat_template(SIMPLE_TEMPLATE, msgs, false, "", "").unwrap();
    // SIMPLE_TEMPLATE: "{{ msg.role }}: {{ msg.content }}\n" → "user: \n"
    assert_eq!(result, "user: \n");
}

// ===========================================================================
// Error handling
// ===========================================================================

/// Invalid messages JSON returns internal_error (999).
#[test]
fn invalid_messages_json_errors() {
    let err = apply_chat_template(SIMPLE_TEMPLATE, "not json", false, "", "").unwrap_err();
    assert_eq!(err, 999, "expected internal_error (999) for malformed messages JSON");
}

// ===========================================================================
// Realistic chat template formats
// ===========================================================================

/// ChatML-style template: `<|im_start|>role\ncontent<|im_end|>`.
///
/// `{%- endfor %}` trims the `\n` before it; `{%- endif %}` trims likewise.
#[test]
fn chatml_style_template() {
    let tmpl = concat!(
        "{%- for msg in messages %}",
        "<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}",
    );
    let msgs = r#"[{"role":"user","content":"Hi"}]"#;
    let result = apply_chat_template(tmpl, msgs, true, "", "").unwrap();
    // \n before {%- endfor %} is trimmed; \n before {%- endif %} is trimmed.
    assert_eq!(result, "<|im_start|>user\nHi<|im_end|><|im_start|>assistant");
}

/// Llama-style template with header IDs and BOS.
#[test]
fn llama_style_template() {
    let tmpl = concat!(
        "{{ bos_token }}",
        "{%- for msg in messages %}",
        "<|start_header_id|>{{ msg.role }}<|end_header_id|>\n\n",
        "{{ msg.content }}<|eot_id|>",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "{%- endif %}",
    );
    let msgs = r#"[{"role":"system","content":"Be helpful."},{"role":"user","content":"Hi"}]"#;
    let result = apply_chat_template(tmpl, msgs, true, "<|begin_of_text|>", "").unwrap();
    assert!(result.starts_with("<|begin_of_text|>"), "missing BOS: {result:?}");
    assert!(result.contains("<|start_header_id|>system<|end_header_id|>"), "missing system header: {result:?}");
    assert!(result.contains("Be helpful.<|eot_id|>"), "missing system content: {result:?}");
    // {%- endif %} trims the trailing \n\n from the generation prompt.
    assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>"), "missing gen prompt: {result:?}");
}

// ===========================================================================
// Whitespace control in chat templates
// ===========================================================================

/// Whitespace-trimmed template produces compact output.
#[test]
fn whitespace_trimmed_chat() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "[{{ msg.role }}]{{ msg.content }}",
        "{%- endfor -%}",
    );
    let msgs = r#"[{"role":"user","content":"A"},{"role":"assistant","content":"B"}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    assert_eq!(result, "[user]A[assistant]B");
}

// ===========================================================================
// Conditional logic on message roles
// ===========================================================================

/// Template uses if/elif to render different markup per role.
#[test]
fn role_conditional_rendering() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{%- if msg.role == 'system' -%}",
        "<<SYS>>{{ msg.content }}<</SYS>>",
        "{%- elif msg.role == 'user' -%}",
        "[INST]{{ msg.content }}[/INST]",
        "{%- elif msg.role == 'assistant' -%}",
        "{{ msg.content }}",
        "{%- endif -%}",
        "{%- endfor -%}",
    );
    let msgs = r#"[
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"Hi"},
        {"role":"assistant","content":"Hello!"},
        {"role":"user","content":"Bye"}
    ]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    assert_eq!(result, "<<SYS>>You are helpful.<</SYS>>[INST]Hi[/INST]Hello![INST]Bye[/INST]");
}

/// Only the first system message gets special treatment.
///
/// All `{%-`/`-%}` trim aggressively — the `\n` before `{%- else -%}`,
/// `{%- endif -%}`, and `{%- endfor -%}` are consumed.
#[test]
fn first_message_system_check() {
    let tmpl = concat!(
        "{%- for msg in messages %}",
        "{%- if loop.first and msg.role == 'system' %}",
        "SYSTEM: {{ msg.content }}\n",
        "{%- else %}",
        "{{ msg.role }}: {{ msg.content }}\n",
        "{%- endif %}",
        "{%- endfor %}",
    );
    let msgs = r#"[
        {"role":"system","content":"Be brief."},
        {"role":"user","content":"Hi"}
    ]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // \n before {%- else %}, {%- endif %}, {%- endfor %} are trimmed.
    assert_eq!(result, "SYSTEM: Be brief.user: Hi");
}

// ===========================================================================
// Namespace for state across messages
// ===========================================================================

/// Template uses namespace to count messages.
#[test]
fn namespace_message_counter() {
    let tmpl = concat!(
        "{%- set ns = namespace(n=0) -%}",
        "{%- for msg in messages -%}",
        "{%- set ns.n = ns.n + 1 -%}",
        "{%- endfor -%}",
        "{{ ns.n }} messages",
    );
    let result = apply_chat_template(tmpl, MULTITURN_MSGS, false, "", "").unwrap();
    assert_eq!(result, "4 messages");
}

/// Namespace tracks whether system prompt was seen.
#[test]
fn namespace_system_flag() {
    let tmpl = concat!(
        "{%- set ns = namespace(has_system=false) -%}",
        "{%- for msg in messages -%}",
        "{%- if msg.role == 'system' -%}",
        "{%- set ns.has_system = true -%}",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{{ 'yes' if ns.has_system else 'no' }}",
    );
    assert_eq!(
        apply_chat_template(tmpl, MULTITURN_MSGS, false, "", "").unwrap(),
        "yes"
    );
    assert_eq!(
        apply_chat_template(tmpl, SINGLE_USER_MSG, false, "", "").unwrap(),
        "no"
    );
}

// ===========================================================================
// Loop variables in chat context
// ===========================================================================

/// `loop.last` controls separator between messages.
///
/// `{%- if not loop.last %}` trims preceding whitespace; the `\n` before
/// `{%- endif -%}` is also consumed, so separators lose their trailing newline.
#[test]
fn loop_last_separator() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{{ msg.content }}",
        "{%- if not loop.last %}\n---\n{%- endif -%}",
        "{%- endfor -%}",
    );
    let msgs = r#"[{"role":"user","content":"A"},{"role":"assistant","content":"B"},{"role":"user","content":"C"}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // {%- endif -%} trims \n before it; -%} trims whitespace into the next
    // iteration, consuming the \n after "---" as well.
    assert_eq!(result, "A---B---C");
}

/// `loop.index` used for message numbering.
///
/// `{%- endfor -%}` trims `\n` before it.
#[test]
fn loop_index_numbering() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{{ loop.index }}. {{ msg.content }}\n",
        "{%- endfor -%}",
    );
    let msgs = r#"[{"role":"user","content":"A"},{"role":"assistant","content":"B"}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // {%- endfor -%} trims \n at end of each iteration.
    assert_eq!(result, "1. A2. B");
}

// ===========================================================================
// Filters in chat context
// ===========================================================================

/// Template applies `upper` filter to role.
#[test]
fn filter_on_role() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{{ msg.role | upper }}: {{ msg.content }}\n",
        "{%- endfor -%}",
    );
    let result = apply_chat_template(tmpl, SINGLE_USER_MSG, false, "", "").unwrap();
    // {%- endfor -%} trims the trailing \n.
    assert_eq!(result, "USER: Hello");
}

/// `trim` filter on message content.
#[test]
fn filter_trim_content() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{{ msg.content | trim }}\n",
        "{%- endfor -%}",
    );
    let msgs = r#"[{"role":"user","content":"  hello  "}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    assert_eq!(result, "hello");
}

// ===========================================================================
// Generation prompt variations
// ===========================================================================

/// Generation prompt with custom prefix and BOS.
///
/// The `-%}` on for/if trims whitespace after tags; `{%-` trims before.
/// `\n` before `{%- endfor -%}` is consumed.
#[test]
fn generation_prompt_with_bos() {
    let tmpl = concat!(
        "{{ bos_token }}",
        "{%- for msg in messages -%}",
        "{{ msg.role }}: {{ msg.content }}\n",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}",
        "assistant: ",
        "{%- endif -%}",
    );
    let result =
        apply_chat_template(tmpl, SINGLE_USER_MSG, true, "<s>", "").unwrap();
    // \n before {%- endfor -%} trimmed; trailing space before {%- endif -%} trimmed.
    assert_eq!(result, "<s>user: Helloassistant:");
}

/// Without generation prompt, output ends after last message.
#[test]
fn no_generation_prompt_ends_at_last() {
    let tmpl = concat!(
        "{%- for msg in messages -%}",
        "{{ msg.content }}",
        "{%- if not loop.last %} | {% endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt %} [GEN]{% endif -%}",
    );
    let msgs = r#"[{"role":"user","content":"A"},{"role":"user","content":"B"}]"#;
    let with = apply_chat_template(tmpl, msgs, true, "", "").unwrap();
    let without = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    assert_eq!(with, "A | B [GEN]");
    assert_eq!(without, "A | B");
}

// ===========================================================================
// Many messages stress
// ===========================================================================

// ===========================================================================
// lstrip_blocks: leading whitespace before block tags must be stripped
// ===========================================================================
//
// HuggingFace chat templates run with lstrip_blocks=True by default.
// When a line starts with a block tag ({% ... %}), all leading whitespace
// on that line is stripped. Without this, indented templates produce
// unwanted spaces in the output.
// Affects: nvidia/NVIDIA-Nemotron (3 chat_template failures).

/// Indented block tags must not produce leading whitespace in output.
///
/// With lstrip_blocks enabled, the spaces before `{% if %}` and `{% endif %}`
/// are stripped, so the output has no leading whitespace.
#[test]
fn lstrip_blocks_strips_indentation() {
    // Template with indented block tags (like Nemotron's template).
    // The 8-space indent before {% if %} / {% endif %} must be stripped.
    let tmpl = concat!(
        "{%- for msg in messages %}\n",
        "<|im_start|>{{ msg.role }}\n",
        "{{ msg.content }}<|im_end|>\n",
        "        {% if msg.role == 'user' %}\n",
        "MARKER\n",
        "        {% endif %}\n",
        "{%- endfor -%}",
    );
    let msgs = r#"[{"role":"user","content":"Hi"}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // With lstrip_blocks, the "        " before {% if %} and {% endif %} is stripped.
    // Without lstrip_blocks, "        " appears before MARKER.
    assert!(
        !result.contains("        MARKER"),
        "lstrip_blocks must strip indentation before block tags, got: {result:?}"
    );
}

/// Realistic Nemotron-style template: multi-turn with indented conditionals.
///
/// The template uses indented `{% if %}` blocks for role-specific handling.
/// lstrip_blocks must strip the leading whitespace so the output matches.
#[test]
fn lstrip_blocks_multiturn_indented_template() {
    // Simplified Nemotron pattern: indented conditionals within for loop.
    let tmpl = concat!(
        "{%- for msg in messages %}\n",
        "<|im_start|>{{ msg.role }}\n",
        "        {% if msg.role == 'assistant' %}\n",
        "<think></think>{{ msg.content }}\n",
        "        {% else %}\n",
        "{{ msg.content }}\n",
        "        {% endif %}\n",
        "<|im_end|>\n",
        "{%- endfor -%}",
    );
    let msgs = r#"[
        {"role":"user","content":"Hello"},
        {"role":"assistant","content":"Hi there!"}
    ]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // The assistant content must NOT have leading spaces from template indentation.
    assert!(
        !result.contains("        <think>"),
        "lstrip_blocks must strip indentation before conditionals, got: {result:?}"
    );
}


// ===========================================================================
// {% set %} reassignment inside for-loop
// ===========================================================================
//
// In Jinja2, {% set content = message['content'] %} inside a for-loop must
// update the variable on every iteration. Without correct scoping, the
// variable "sticks" at the first iteration's value and all subsequent
// iterations render stale content.
// Affects: Qwen/Qwen3-VL-Thinking models (10 chat_template failures).

/// Macro parameter scope must not leak across for-loops.
///
/// Qwen3-VL-Thinking templates use two sequential for-loops that both
/// call the same macro and assign `{% set content = render_content(...) %}`.
/// The macro parameter `content` must be freshly bound on each call.
///
/// Bug: `callMacro()` binds parameters to the global context, but
/// `getVar()` searches local scopes first. A stale local `content` from
/// a previous loop iteration is found instead of the macro's parameter,
/// causing all iterations to render the first message's content.
#[test]
fn macro_param_scope_across_two_loops() {
    // Macro with a parameter named `content` (same as the loop variable).
    // First loop: iterates in reverse (like Qwen3-VL's multi-step tool check).
    // Second loop: iterates forward and renders each message's content.
    let tmpl = concat!(
        "{%- macro render_content(content) -%}",
        "{%- if content is string -%}",
        "{{ content }}",
        "{%- endif -%}",
        "{%- endmacro -%}",
        // First loop: sets `content` via macro call (reverse order).
        "{%- for message in messages -%}",
        "{%- set content = render_content(message['content']) -%}",
        "{%- endfor -%}",
        // Second loop: sets `content` via same macro call (forward order).
        "{%- for message in messages -%}",
        "{%- set content = render_content(message['content']) -%}",
        "<|im_start|>{{ message['role'] }}\n{{ content }}<|im_end|>\n",
        "{%- endfor -%}",
    );
    let msgs = r#"[
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"Hi"},
        {"role":"assistant","content":"Hello!"}
    ]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    // Each message must render its own content, not the first message's.
    assert!(
        result.contains("user\nHi<|im_end|>"),
        "user message must contain 'Hi', not system content, got: {result:?}"
    );
    assert!(
        result.contains("assistant\nHello!<|im_end|>"),
        "assistant message must contain 'Hello!', not system content, got: {result:?}"
    );
}

// ===========================================================================
// add_generation_prompt with namespace and compound conditional
// ===========================================================================
//
// DeepSeek-R1 templates use:
//   {% set ns = namespace(is_tool=false) %}
//   {% if add_generation_prompt and not ns.is_tool %}
//     <|Assistant|><think>
//   {% endif %}
//
// The generation prompt must be appended when add_generation_prompt=true
// and ns.is_tool is false.
// Affects: deepseek-ai/DeepSeek-R1 (8 chat_template failures).

/// `add_generation_prompt and not ns.attr` must evaluate correctly.
///
/// When `add_generation_prompt` is true and `ns.is_tool` is false (its
/// default), the compound condition `add_generation_prompt and not ns.is_tool`
/// must be true and the generation prompt block must render.
#[test]
fn generation_prompt_with_namespace_compound_conditional() {
    // Use {{ '...' }} expression to output the generation prompt, matching the
    // real DeepSeek-R1 template pattern. This prevents {%- endif %} from
    // trimming the trailing newline (which is inside an expression, not raw text).
    let tmpl = concat!(
        "{%- set ns = namespace(is_tool=false) -%}",
        "{%- for message in messages -%}",
        "<|{{ message['role'] }}|>{{ message['content'] }}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt and not ns.is_tool %}",
        "{{'<|Assistant|><think>\n'}}",
        "{%- endif -%}",
    );
    let msgs = r#"[{"role":"user","content":"Hi"}]"#;
    let result = apply_chat_template(tmpl, msgs, true, "", "").unwrap();
    assert!(
        result.ends_with("<|Assistant|><think>\n"),
        "generation prompt must be appended when add_generation_prompt=true and ns.is_tool=false, got: {result:?}"
    );
}

/// `add_generation_prompt=false` must suppress generation prompt even with namespace.
#[test]
fn generation_prompt_false_with_namespace_suppresses() {
    let tmpl = concat!(
        "{%- set ns = namespace(is_tool=false) -%}",
        "{%- for message in messages -%}",
        "<|{{ message['role'] }}|>{{ message['content'] }}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt and not ns.is_tool -%}",
        "<|Assistant|><think>\n",
        "{%- endif -%}",
    );
    let msgs = r#"[{"role":"user","content":"Hi"}]"#;
    let result = apply_chat_template(tmpl, msgs, false, "", "").unwrap();
    assert!(
        !result.contains("<|Assistant|>"),
        "generation prompt must not appear when add_generation_prompt=false, got: {result:?}"
    );
}

// ===========================================================================
// Many messages stress
// ===========================================================================

/// 20 messages render correctly and completely.
#[test]
fn twenty_messages() {
    let msgs: Vec<String> = (0..20)
        .map(|i| {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            format!(r#"{{"role":"{}","content":"msg{}"}}"#, role, i)
        })
        .collect();
    let json = format!("[{}]", msgs.join(","));
    let result = apply_chat_template(SIMPLE_TEMPLATE, &json, false, "", "").unwrap();
    for i in 0..20 {
        assert!(
            result.contains(&format!("msg{i}")),
            "missing msg{i} in output"
        );
    }
}
