"""Structured Output - Get typed, validated responses from the model.

Primary API: talu.Chat
Scope: Single

This example shows the complete end-to-end workflow:
1. Define a Pydantic model
2. Pass it to chat.send(..., response_format=...)
3. Get back a validated, typed Python object

The library handles:
- Converting Pydantic models to JSON Schema
- Injecting schema into prompts (TypeScript/JSON Schema/XML format)
- Parsing and validating the model's JSON response
- Returning a typed Pydantic object via response.parsed

Key points:
- response_format is the only argument you need for most use cases
- Schema strategy is auto-selected based on model architecture
- Use GenerationConfig to customize behavior if needed
- For union types, add a Literal discriminator field for reliable disambiguation

Related:
    - examples/basics/15_structured_output.py
"""

import json
from pydantic import BaseModel, Field

import talu
from talu.router import GenerationConfig
from talu.router.schema import schema_to_prompt_description


# =============================================================================
# Step 1: Define Your Schema with Pydantic
# =============================================================================

class Answer(BaseModel):
    """A simple answer with reasoning."""
    value: int
    reasoning: str


# You can inspect the generated JSON Schema using Pydantic's built-in method:
schema = Answer.model_json_schema()
print("Generated JSON Schema:")
print(json.dumps(schema, indent=2))

# And see how it will be injected into the prompt:
prompt_text = schema_to_prompt_description(schema)
print("\nPrompt injection (TypeScript format):")
print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)


# =============================================================================
# Step 2: Send Request with response_format
# =============================================================================

chat = talu.Chat("Qwen/Qwen3-0.6B")

# The clean API - just specify what you want, not how
response = chat.send("What is 15 + 27?", response_format=Answer)

# The raw text is valid JSON
print(f"\nRaw response text: {response.text}")


# =============================================================================
# Step 3: Access the Parsed, Validated Result
# =============================================================================

# response.parsed returns a validated Pydantic object
result = response.parsed
print(f"\nParsed result type: {type(result).__name__}")
print(f"Answer: {result.value}")
print(f"Reasoning: {result.reasoning}")

# You can also get the raw dict without Pydantic validation
raw_dict = response.parsed_dict
print(f"Raw dict: {raw_dict}")


# =============================================================================
# Complex Schemas - Nested models, arrays, unions
# =============================================================================

class Address(BaseModel):
    street: str
    city: str
    country: str = "USA"


class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150, description="Age in years")
    email: str | None = None
    addresses: list[Address] = []


chat2 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat2.send(
    "Create a profile for a software engineer named Alice, age 28, "
    "living in San Francisco with a vacation home in Lake Tahoe.",
    response_format=Person,
)

person = response.parsed
print(f"\nPerson: {person.name}, {person.age} years old")
for addr in person.addresses:
    print(f"  - {addr.city}, {addr.country}")


# =============================================================================
# Discriminated Unions - When the model must choose between types
# =============================================================================

# When you have a union of types, add a discriminator field so the model
# can reliably indicate which type it's returning. Without a discriminator,
# Talu will emit an AmbiguousUnionWarning.

from typing import Literal


class Refund(BaseModel):
    """Refund action - return money to customer."""
    kind: Literal["refund"] = "refund"  # Discriminator field
    transaction_id: str
    amount: float


class Escalate(BaseModel):
    """Escalate to human support."""
    kind: Literal["escalate"] = "escalate"  # Discriminator field
    reason: str
    urgency: int = Field(ge=1, le=5, description="1=low, 5=critical")


# The union type - model will output one or the other
CustomerAction = Refund | Escalate

chat_support = talu.Chat("Qwen/Qwen3-0.6B")
response = chat_support.send(
    "Customer says: 'I was charged twice for order #12345, please fix this!'",
    response_format=CustomerAction,
)

action = response.parsed
print(f"\nAction type: {type(action).__name__}")
print(f"Discriminator: {action.kind}")

if isinstance(action, Refund):
    print(f"Refunding ${action.amount} for transaction {action.transaction_id}")
elif isinstance(action, Escalate):
    print(f"Escalating: {action.reason} (urgency: {action.urgency})")


# =============================================================================
# Thinking Mode - Let the model reason before answering
# =============================================================================

class Solution(BaseModel):
    answer: int
    steps: list[str]


# Enable thinking via GenerationConfig
config = GenerationConfig(
    allow_thinking=True,       # Enable <think>...</think> block
    max_thinking_tokens=256,   # Limit thinking length
)

chat3 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat3.send(
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    response_format=Solution,
    config=config,
)

print(f"\nAnswer: {response.parsed.answer}")
print("Steps:")
for step in response.parsed.steps:
    print(f"  - {step}")

# Access the thinking block if present
if response.thinking:
    print(f"\nModel's reasoning:\n{response.thinking}")


# =============================================================================
# Expert Control - Override schema injection strategy
# =============================================================================

# For most users, auto-selection works great. But if you're experimenting
# with a new model or know a specific format works better:

expert_config = GenerationConfig(
    schema_strategy="json_schema",   # Force JSON Schema format
    inject_schema_prompt=True,       # Auto-inject (default)
    max_tokens=200,
)

response = chat.send(
    "What is 2 + 2?",
    response_format=Answer,
    config=expert_config,
)
print(f"\nWith JSON Schema strategy: {response.parsed.value}")


# =============================================================================
# Manual Schema Injection - For prompt engineers
# =============================================================================

# If you've carefully crafted your own prompt with the schema included,
# disable auto-injection to avoid duplication:

manual_config = GenerationConfig(
    inject_schema_prompt=False,  # Don't auto-inject, I did it myself
)

chat4 = talu.Chat(
    "Qwen/Qwen3-0.6B",
    system="""You are a math tutor. Always respond with JSON in this format:
{"value": <integer>, "reasoning": "<explanation>"}"""
)

response = chat4.send(
    "What is 10 * 5?",
    response_format=Answer,
    config=manual_config,
)
print(f"\nManual injection: {response.parsed.value}")


# =============================================================================
# Streaming with Structured Output
# =============================================================================

class Story(BaseModel):
    title: str
    content: str
    moral: str


chat5 = talu.Chat("Qwen/Qwen3-0.6B")

# Streaming works with structured output too
print("\nStreaming story generation:")
for chunk in chat5("Tell me a short fable", response_format=Story):
    print(chunk, end="", flush=True)

# After iteration completes, access the parsed result via last_response
print(f"\n\nTitle: {chat5.last_response.parsed.title}")
print(f"Moral: {chat5.last_response.parsed.moral}")


# =============================================================================
# Error Handling - When validation fails
# =============================================================================

from talu.exceptions import SchemaValidationError

class StrictPerson(BaseModel):
    name: str
    age: int = Field(ge=0, le=120)  # Age must be 0-120


# If the model returns invalid data, you can still recover it
chat6 = talu.Chat("Qwen/Qwen3-0.6B")
try:
    response = chat6.send(
        "Create a profile for an ancient wizard who is 500 years old",
        response_format=StrictPerson,
    )
    print(f"\nParsed: {response.parsed}")
except SchemaValidationError as e:
    # Validation failed, but we can still access the raw data
    print(f"\nValidation failed: {e}")
    print(f"Raw data that failed validation: {e.data}")
    # You might salvage what you can or retry with different constraints


# =============================================================================
# Using Raw JSON Schema (without Pydantic)
# =============================================================================

# You can also pass a raw JSON Schema dict instead of a Pydantic model
raw_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "required": ["name", "score"],
}

chat7 = talu.Chat("Qwen/Qwen3-0.6B")
response = chat7.send(
    "Rate the movie 'Inception' out of 100",
    response_format=raw_schema,
)

# With raw schema, parsed_dict returns a dict (no Pydantic model)
print(f"\nRaw schema result: {response.parsed_dict}")

"""
Topics covered:
* structured.output
* schema.pydantic
* schema.union
* parsing.typed

Related:
* examples/developers/chat/precompiled_grammar.py (for zero-latency reuse)
"""
