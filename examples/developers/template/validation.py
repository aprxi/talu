"""
Validation - Use validate() to catch missing variables before rendering.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

Bad prompts waste API calls and produce garbage outputs. Validate
inputs before rendering to catch missing variables and invalid data.

Validation distinguishes between:
- Required variables: Used "naked" ({{ name }}) - must be provided
- Optional variables: Have defaults ({{ name | default('') }}) - safe to omit

Related:
- examples/basics/10_prompt_templates.py
"""

import talu
from talu.exceptions import TemplateUndefinedError

# =============================================================================
# Required vs Optional Variables
# =============================================================================

# Template with both required and optional variables
template = talu.PromptTemplate("""
User: {{ name }}
Bio: {{ bio | default('No bio provided') }}
Role: {{ role | d('user') }}
""")

# Check what variables are needed
print("All input variables:", template.input_variables)

# Validate with only required variable provided
result = template.validate(name="Alice")
print(f"\nValidation with only 'name':")
print(f"  is_valid: {result.is_valid}")  # True - optional vars have defaults
print(f"  required (missing): {result.required}")  # empty - name provided
print(f"  optional (missing): {result.optional}")  # {bio, role}

# Validate with nothing provided
result = template.validate()
print(f"\nValidation with nothing:")
print(f"  is_valid: {result.is_valid}")  # False - name is required
print(f"  required (missing): {result.required}")  # {name}
print(f"  optional (missing): {result.optional}")  # {bio, role}

# =============================================================================
# Production RAG Template
# =============================================================================

rag = talu.PromptTemplate("""
Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ question }}
""")

print(f"\nRAG template inputs: {rag.input_variables}")


# Validate before calling the model
def generate_answer(documents, question):
    """Generate answer with input validation.

    Uses ValidationResult.render() to avoid re-serializing documents twice.
    This is especially important for large RAG contexts (10MB+ of documents).
    """
    # Validate once - this serializes kwargs to JSON
    result = rag.validate(documents=documents, question=question)

    if not result.is_valid:
        raise ValueError(f"Invalid inputs: {result.summary}")

    # Render reuses the serialized JSON - no double serialization!
    return result.render()


# Good inputs
try:
    prompt = generate_answer(
        documents=[{"content": "Paris is the capital of France."}],
        question="What is the capital of France?",
    )
    print("Valid prompt generated")
except ValueError as e:
    print(f"Error: {e}")

# Missing required input
result = rag.validate(documents=[])  # Missing 'question'
if not result.is_valid:
    print(f"\nValidation failed: {result.summary}")

# =============================================================================
# Detecting Invalid Values
# =============================================================================

# Detect non-serializable values (common bug: passing functions)
template = talu.PromptTemplate("{{ callback }}")
result = template.validate(callback=lambda x: x)
print(f"\nLambda valid? {result.is_valid}")
print(f"Invalid values: {result.invalid}")

# =============================================================================
# Strict Mode (Default)
# =============================================================================

# Strict mode is now the default - prevents silent failures in LLM prompts.
# Missing variables raise errors instead of rendering as empty strings.
template = talu.PromptTemplate("Hello {{ name }}!")  # strict by default

try:
    template()  # Missing 'name' - will raise
except TemplateUndefinedError:
    print("\nStrict mode caught missing variable!")

# With default filter, strict mode is satisfied - use for optional variables
optional_template = talu.PromptTemplate("""
User: {{ name }}
Role: {{ role | default('user') }}
""")

print(optional_template(name="Alice"))  # Works - role has default

# Runtime strict override - switch modes per-call without new instance
# Use strict=False for debugging to see partial output

# Lenient mode (for debugging): see partial output
print(f"\nLenient (debug): {template(strict=False)!r}")  # 'Hello !'

# Explicit strict (redundant but allowed)
try:
    template(strict=True)  # Explicitly strict
except TemplateUndefinedError:
    print("Strict caught missing variable!")

# Switch strict->lenient at runtime (e.g., for debugging)
print(f"Override strict->lenient: {template(strict=False)!r}")  # 'Hello !'

# =============================================================================
# Template-Level Validation with raise_exception()
# =============================================================================

# Use raise_exception() inside templates for custom validation logic
# Error messages are preserved and surfaced to Python

validated_template = talu.PromptTemplate("""
{%- if not documents -%}
{{ raise_exception("At least one document is required for RAG") }}
{%- endif -%}
{%- if documents | length > 10 -%}
{{ raise_exception("Too many documents (max 10), got " ~ documents | length) }}
{%- endif -%}
{%- if not question or question | trim == '' -%}
{{ raise_exception("Question cannot be empty") }}
{%- endif -%}

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ question }}
""")

print("\n--- Template-Level Validation ---")

# Valid inputs
try:
    prompt = validated_template(
        documents=[{"content": "Paris is in France."}],
        question="Where is Paris?",
    )
    print("Valid: Generated prompt successfully")
except Exception as e:
    print(f"Error: {e}")

# Empty documents
try:
    validated_template(documents=[], question="What?")
except Exception as e:
    print(f"Empty docs error: {e}")

# Empty question
try:
    validated_template(documents=[{"content": "test"}], question="   ")
except Exception as e:
    print(f"Empty question error: {e}")

# Too many documents
try:
    validated_template(
        documents=[{"content": f"doc {i}"} for i in range(15)], question="What?"
    )
except Exception as e:
    print(f"Too many docs error: {e}")

# =============================================================================
# Third-Party Templates (from_chat_template)
# =============================================================================

# When loading templates you didn't write, validation helps understand requirements
print("\n--- Understanding Third-Party Templates ---")

# Simulate a complex template from a model
complex_template = talu.PromptTemplate("""
{% if system_message | default('') %}
<|im_start|>system
{{ system_message }}
<|im_end|>
{% endif %}
{% for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}
<|im_end|>
{% endfor %}
{% if add_generation_prompt | default(true) %}
<|im_start|>assistant
{% endif %}
""")

result = complex_template.validate(messages=[{"role": "user", "content": "Hi"}])
print(f"Required: {result.required}")  # What MUST be provided
print(f"Optional: {result.optional}")  # What CAN be omitted (has defaults)
print(f"Valid: {result.is_valid}")

# =============================================================================
# Efficient Validate-Then-Render Pattern
# =============================================================================

# For large data (RAG documents, conversation history), avoid serializing twice.
# ValidationResult.render() reuses the JSON from validation.

print("\n--- Efficient Validate-Then-Render ---")

large_documents = [{"content": f"Document {i} with lots of content..."} for i in range(100)]

# Old pattern (serializes twice):
#   result = template.validate(documents=large_documents)  # serialize #1
#   if result.is_valid:
#       output = template(documents=large_documents)       # serialize #2 (wasteful!)

# New pattern (serializes once):
result = rag.validate(documents=large_documents, question="What is in the documents?")
if result.is_valid:
    output = result.render()  # Reuses JSON from validation - no re-serialization!
    print(f"Rendered {len(output)} chars from {len(large_documents)} documents")
else:
    print(f"Validation failed: {result.summary}")

"""
Topics covered:
* template.render
* template.control.flow
"""
