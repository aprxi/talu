"""Pre-compiled Grammar - Zero-latency structured output.

Primary API: talu.router.Grammar
Scope: Single

This example demonstrates how to pre-compile JSON schemas into reusable grammar
handles for zero-latency structured output. This is useful when:

1. Using the same schema across many requests (e.g., API servers)
2. Validating schemas at startup time instead of at request time
3. Minimizing per-request overhead in high-throughput scenarios

Key benefits:
- Schema validation happens once at Grammar creation time
- Subsequent generations skip schema parsing/compilation entirely
- Grammar instances can be shared across multiple Chat sessions
- Invalid schemas fail fast at startup, not at request time

Related:
- examples/basics/15_structured_output.py
"""

from dataclasses import dataclass

import talu
from talu.router import Grammar


# =============================================================================
# Step 1: Define Your Schema
# =============================================================================

@dataclass
class Answer:
    """A structured answer with value and reasoning."""
    value: int
    reasoning: str


@dataclass
class Person:
    """A person with name and age."""
    name: str
    age: int


# =============================================================================
# Step 2: Pre-compile at Startup
# =============================================================================

# Compile grammars once at module/application startup
# If the schema is invalid, StructuredOutputError is raised HERE, not during generation
ANSWER_GRAMMAR = Grammar(Answer)
PERSON_GRAMMAR = Grammar(Person)

print("Grammars pre-compiled successfully!")
print(f"  Answer schema: {ANSWER_GRAMMAR.schema}")
print(f"  Person schema: {PERSON_GRAMMAR.schema}")


# =============================================================================
# Step 3: Use in Requests (Zero Compilation Overhead)
# =============================================================================

chat = talu.Chat("Qwen/Qwen3-0.6B")

# Each call reuses the pre-compiled grammar - no schema parsing needed
response1 = chat.send("What is 2 + 2?", response_format=ANSWER_GRAMMAR, stream=False)
print(f"\nAnswer 1: {response1.parsed.value}")

response2 = chat.send("What is 10 * 5?", response_format=ANSWER_GRAMMAR, stream=False)
print(f"Answer 2: {response2.parsed.value}")


# =============================================================================
# Production Pattern: Global Grammar Registry
# =============================================================================

# For production applications, define grammars at module level
# This ensures validation at import time and enables sharing across requests

class GrammarRegistry:
    """Central registry for pre-compiled grammars."""

    ANSWER = Grammar(Answer)
    PERSON = Grammar(Person)

    # You can add more as needed
    # PRODUCT = Grammar(Product)
    # ORDER = Grammar(Order)


# Use in request handlers
def handle_math_question(question: str) -> int:
    """Handle a math question using the pre-compiled Answer grammar."""
    chat = talu.Chat("Qwen/Qwen3-0.6B")
    response = chat.send(question, response_format=GrammarRegistry.ANSWER, stream=False)
    return response.parsed.value


result = handle_math_question("What is 7 * 8?")
print(f"\nRegistry pattern result: {result}")


# =============================================================================
# Error Handling: Fail Fast at Startup
# =============================================================================

from talu.exceptions import StructuredOutputError

# Invalid schemas fail at Grammar() construction, not at generation time
try:
    invalid_grammar = Grammar({"type": "invalid_type"})
except StructuredOutputError as e:
    print(f"\nSchema validation caught at startup: {e}")


# =============================================================================
# Comparing: With vs Without Pre-compiled Grammar
# =============================================================================

# WITHOUT pre-compiled grammar (schema compiled on each call):
# response = chat.send("What is 1+1?", response_format=Answer)
#                                      ^^^^^^^^^^^^^^^^^^^^^^
#                                      Schema parsed and compiled here

# WITH pre-compiled grammar (schema already compiled):
# response = chat.send("What is 1+1?", response_format=ANSWER_GRAMMAR)
#                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                      Just uses the pre-compiled handle

print("\nPre-compiled grammars provide:")
print("  - Zero compilation latency per request")
print("  - Fail-fast schema validation at startup")
print("  - Grammar reuse across multiple Chat sessions")


# =============================================================================
# Sharing Grammar Across Multiple Chats
# =============================================================================

# The same grammar can be used with different Chat instances
chat_a = talu.Chat("Qwen/Qwen3-0.6B", system="You are a math tutor.")
chat_b = talu.Chat("Qwen/Qwen3-0.6B", system="You are a trivia host.")

# Both use the same pre-compiled grammar
response_a = chat_a.send("What is 100 / 4?", response_format=ANSWER_GRAMMAR, stream=False)
response_b = chat_b.send("How many legs does a spider have?", response_format=ANSWER_GRAMMAR, stream=False)

print(f"\nMath tutor: {response_a.parsed.value}")
print(f"Trivia host: {response_b.parsed.value}")


"""
Topics covered:
* structured.output
* parsing.typed
* model.reuse

Related:
* examples/basics/15_structured_output.py
"""
