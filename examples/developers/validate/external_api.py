"""External API Integration - Validate responses from LLM providers.

Primary API: talu.Validator
Scope: Single

When using external LLM providers (OpenAI, Anthropic, etc.),
streaming validation enables early abort on schema violations,
saving API costs and reducing latency.

Related:
    - examples/developers/validate/streaming_validation.py
"""

from talu import Validator


# =============================================================================
# The Problem: External APIs and Structured Output
# =============================================================================

# When you request structured output from an external LLM API:
# 1. You send a prompt asking for JSON in a specific format
# 2. The model streams back tokens
# 3. You wait until completion to validate
# 4. If invalid, you've paid for all those tokens for nothing
#
# With streaming validation, you can abort mid-stream when
# a violation is detected.


# =============================================================================
# Example: Validating OpenAI-style Streaming Response
# =============================================================================

# Define the schema you requested from the LLM
response_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "sources": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["answer", "confidence"],
}


def validate_streaming_response(stream, schema, on_chunk=None):
    """
    Validate a streaming JSON response with early abort.

    Args:
        stream: Iterator yielding string chunks
        schema: JSON schema dict or Pydantic model
        on_chunk: Optional callback for each valid chunk

    Returns:
        Complete validated JSON string

    Raises:
        ValueError: On schema violation or incomplete JSON
    """
    validator = Validator(schema)
    buffer = []

    for chunk in stream:
        # Validate this chunk
        if not validator.feed(chunk):
            received = "".join(buffer)
            raise ValueError(
                f"Schema violation at byte {validator.position}.\n"
                f"Received: {received!r}\n"
                f"Failed on: {chunk!r}"
            )

        buffer.append(chunk)

        # Optional: notify caller of progress
        if on_chunk:
            on_chunk(chunk)

    if not validator.is_complete:
        raise ValueError(
            f"Incomplete JSON. Received: {''.join(buffer)!r}"
        )

    return "".join(buffer)


# Simulate valid streaming response
valid_chunks = [
    '{"answer": "',
    'The capital of France is Paris.',
    '", "confidence": ',
    '0.95, "sources": ["wikipedia"]}',
]

result = validate_streaming_response(valid_chunks, response_schema)
print(f"Valid response: {result[:50]}...")


# Simulate invalid response (confidence as string instead of number)
invalid_chunks = [
    '{"answer": "Test", "confidence": "',
    'high"',  # Should be number!
]

try:
    validate_streaming_response(invalid_chunks, response_schema)
except ValueError as e:
    print(f"Caught violation: {e}")


# =============================================================================
# Pattern: Wrapper for External API Client
# =============================================================================

class ValidatedStreamingClient:
    """
    Wrapper that adds streaming validation to any LLM client.

    This pattern works with OpenAI, Anthropic, or any provider
    that returns streaming responses.
    """

    def __init__(self, client, schema):
        """
        Args:
            client: The underlying LLM client
            schema: Schema to validate responses against
        """
        self.client = client
        self.schema = schema

    def stream(self, prompt, **kwargs):
        """
        Stream a response with validation.

        Yields chunks as they arrive, aborts on violation.
        """
        validator = Validator(self.schema)
        buffer = []

        # In real code, this would be:
        # for chunk in self.client.chat.completions.create(
        #     messages=[{"role": "user", "content": prompt}],
        #     stream=True,
        #     **kwargs
        # ):
        #     content = chunk.choices[0].delta.content or ""

        # Simulated response for this example
        simulated_response = [
            '{"answer": "',
            f'Response to: {prompt[:20]}...',
            '", "confidence": 0.9}',
        ]

        for content in simulated_response:
            if not content:
                continue

            if not validator.feed(content):
                raise ValueError(
                    f"Schema violation at byte {validator.position}"
                )

            buffer.append(content)
            yield content

        if not validator.is_complete:
            raise ValueError("Incomplete JSON response")


# Usage
# client = ValidatedStreamingClient(openai_client, response_schema)
# for chunk in client.stream("What is the capital of France?"):
#     print(chunk, end="", flush=True)


# =============================================================================
# Cost Savings Example
# =============================================================================

def estimate_savings(total_chunks, failure_point, cost_per_chunk):
    """
    Estimate cost savings from early abort.

    Args:
        total_chunks: How many chunks the full response would have
        failure_point: Chunk index where violation was detected
        cost_per_chunk: Cost per chunk (e.g., based on tokens)

    Returns:
        Tuple of (cost_without_validation, cost_with_validation, savings)
    """
    cost_without = total_chunks * cost_per_chunk
    cost_with = failure_point * cost_per_chunk
    savings = cost_without - cost_with
    return cost_without, cost_with, savings


# Example: Model outputs 100 chunks, but fails schema at chunk 10
without, with_val, saved = estimate_savings(
    total_chunks=100,
    failure_point=10,
    cost_per_chunk=0.001,  # $0.001 per chunk
)

print(f"\nCost without validation: ${without:.3f}")
print(f"Cost with validation: ${with_val:.3f}")
print(f"Savings: ${saved:.3f} ({saved/without*100:.0f}%)")


"""
Topics covered:
* validate.streaming
* validate.api
"""
