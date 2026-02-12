"""
Debug Mode - Use debug=True to inspect which variables produced what output.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

When the model gives unexpected outputs, debug mode shows exactly which
variables produced which parts of the prompt. Essential for debugging
"why did the model hallucinate?" (often: a variable was empty).

Enable debug mode with:
  - template(debug=True) for a single call
  - TALU_DEBUG_TEMPLATES=1 environment variable for global debug

Related:
- examples/developers/template/prompt_logging.py
"""

import talu

# A RAG prompt with potential issues
rag = talu.PromptTemplate("""
Context: {{ context }}

Question: {{ question }}
Answer:""")

# Debug to see variable contributions
result = rag(
    context="Paris is the capital of France.",
    question="What is the capital of France?",
    debug=True,
)

print("=== Rendered Output ===")
print(result.output)

print("\n=== Variable Map ===")
for span in result.spans:
    if span.is_variable:
        print(f"  {span.source}: '{span.text}'")


# Check if context was empty (empty values produce no span)
print("\n=== Check for Empty Variables ===")
result = rag(context="", question="What is the capital?", debug=True)
context_span = next((s for s in result.spans if s.source == "context"), None)
if context_span is None:
    print("context was empty")

# Spot empty variables (common cause of hallucinations)
result = rag(context="", question="What is the capital?", debug=True)

print("\n=== Empty Context Debug ===")
print(f"Output: {result.output!r}")
for span in result.spans:
    if span.is_variable and not span.text:
        print(f"  WARNING: '{span.source}' is empty!")

# Plain text format for logs
print("\n=== Log Format ===")
result = rag(context="Some docs", question="A question", debug=True)
print(result.format_plain())

# Debug nested attributes
user_prompt = talu.PromptTemplate("Hello {{ user.name }}, your email is {{ user.email }}")
result = user_prompt(user={"name": "Alice", "email": "alice@example.com"}, debug=True)

print("\n=== Nested Attributes ===")
for span in result.spans:
    if span.is_variable:
        print(f"  Path: {span.source} -> '{span.text}'")

# Debug expressions vs variables
calc = talu.PromptTemplate("Sum: {{ a + b }}, Name: {{ name }}")
result = calc(a=2, b=3, name="Test", debug=True)

print("\n=== Expressions vs Variables ===")
for span in result.spans:
    if span.is_expression:
        print(f"  Expression: '{span.text}'")
    elif span.is_variable:
        print(f"  Variable {span.source}: '{span.text}'")

# Global debug via environment variable
print("\n=== Environment Variable ===")
print("Set TALU_DEBUG_TEMPLATES=1 to enable debug mode for all template renders")
print("Example: TALU_DEBUG_TEMPLATES=1 python my_script.py")

"""
Topics covered:
* template.render
* template.control.flow
"""
