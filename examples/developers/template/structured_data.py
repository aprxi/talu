"""
JSON Schema Inference - Generate schemas from example data with json_schema filter.

Primary API: talu.PromptTemplate, talu.template.PromptTemplate
Scope: Single

The json_schema filter infers a JSON Schema from any data structure.
This is useful for structured output prompts where you want the model
to respond in a specific format.

Benefits:
- No manual schema writing
- Schema always matches your data types
- Works with nested objects and arrays
- Combine with tojson for pretty output

Related:
- examples/developers/chat/structured_output.py
"""

import talu

# =============================================================================
# Basic Type Inference
# =============================================================================

print("=== Basic Types ===")
t = talu.PromptTemplate("{{ data | json_schema | tojson }}")

print(f"String:  {t(data='hello')}")
print(f"Integer: {t(data=42)}")
print(f"Float:   {t(data=3.14)}")
print(f"Boolean: {t(data=True)}")
print(f"Null:    {t(data=None)}")

# =============================================================================
# Object Schema Inference
# =============================================================================

print("\n=== Object Schema ===")
t = talu.PromptTemplate("{{ data | json_schema | tojson(2) }}")

user_example = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com",
}

print(t(data=user_example))

# =============================================================================
# Array Schema Inference
# =============================================================================

print("\n=== Array Schema ===")

# Array of strings
print("String array:")
print(t(data=["apple", "banana", "cherry"]))

# Array of objects
print("\nObject array:")
print(t(data=[{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]))

# =============================================================================
# Nested Structures
# =============================================================================

print("\n=== Nested Structures ===")

complex_example = {
    "user": {
        "name": "Alice",
        "profile": {
            "bio": "Software engineer",
            "links": ["github.com", "linkedin.com"],
        },
    },
    "settings": {
        "theme": "dark",
        "notifications": True,
    },
}

print(t(data=complex_example))

# =============================================================================
# Structured Output Prompts
# =============================================================================

print("\n=== Structured Output Prompt ===")

# Define your expected response format with an example
response_example = {
    "sentiment": "positive",
    "confidence": 0.95,
    "keywords": ["happy", "excited"],
    "summary": "Brief summary here",
}

prompt = talu.PromptTemplate("""
Analyze the following text and respond with JSON matching this schema:

{{ example | json_schema | tojson(2) }}

Text: {{ text }}

Respond with valid JSON only.
""".strip())

print(prompt(
    example=response_example,
    text="I'm so happy about this amazing product! It exceeded all my expectations.",
))

# =============================================================================
# RAG with Schema-Constrained Output
# =============================================================================

print("\n\n=== RAG with Schema ===")

answer_format = {
    "answer": "The direct answer",
    "sources": [1, 2],  # Document indices used
    "confidence": "high",  # high, medium, low
}

rag_prompt = talu.PromptTemplate("""
Context documents:
{% for doc in documents %}
[{{ loop.index }}] {{ doc }}
{% endfor %}

Question: {{ question }}

Respond with JSON matching this schema:
{{ format | json_schema | tojson(2) }}
""".strip())

print(rag_prompt(
    documents=[
        "Paris is the capital of France.",
        "France is in Western Europe.",
        "The Eiffel Tower is in Paris.",
    ],
    question="What is the capital of France?",
    format=answer_format,
))

# =============================================================================
# Entity Extraction
# =============================================================================

print("\n\n=== Entity Extraction ===")

entity_format = {
    "people": [{"name": "John", "role": "CEO"}],
    "organizations": ["Acme Corp"],
    "locations": ["New York"],
    "dates": ["2024-01-15"],
}

extraction_prompt = talu.PromptTemplate("""
Extract entities from the text below.

Output schema:
{{ format | json_schema | tojson(2) }}

Text: {{ text }}

JSON:
""".strip())

print(extraction_prompt(
    format=entity_format,
    text="John Smith, CEO of Acme Corp, announced the merger in New York on January 15, 2024.",
))

# =============================================================================
# Combining with Partial Application
# =============================================================================

print("\n\n=== Partial Application ===")

# Create a reusable structured output template
base_template = talu.PromptTemplate("""
{{ instruction }}

Output schema:
{{ schema | json_schema | tojson(2) }}

Input: {{ input }}
Output:
""".strip())

# Pre-configure for sentiment analysis
sentiment_template = base_template.partial(
    instruction="Analyze the sentiment of the following text.",
    schema={"sentiment": "positive", "score": 0.9},
)

# Pre-configure for summarization
summary_template = base_template.partial(
    instruction="Summarize the following text.",
    schema={"summary": "Brief summary", "word_count": 50},
)

print("Sentiment template:")
print(sentiment_template(input="I love this product!"))

print("\n\nSummary template:")
print(summary_template(input="Long article text here..."))

"""
Topics covered:
* template.render
* structured.output
"""
