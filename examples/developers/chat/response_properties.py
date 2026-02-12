"""Response Properties - Multimodal content and prompt audit trail.

Primary API: talu.chat.Response, talu.types.ContentType, talu.types.OutputText
Scope: Single

This example demonstrates the Response.content and Response.prompt properties
for multimodal output symmetry and debugging audit trails.

Key points:
- Response.content returns list[ContentPart] for forward-compatible multimodal output
- Response.prompt contains the fully rendered prompt for debugging template issues
- Both properties work with streaming and async variants
- content is the source of truth; .text is a convenience wrapper

Related:
    - examples/developers/chat/configuration.py (generation config)
"""

from talu import Chat
from talu.types import ContentType, OutputText

# =============================================================================
# Response.content - Multimodal Output Symmetry
# =============================================================================

print("=== Response.content ===")

chat = Chat("Qwen/Qwen3-0.6B", system="You are helpful.")
response = chat.send("What is 2+2?", max_tokens=10)

# Access structured content parts
content = response.content
print(f"Content parts: {len(content)}")

# For text responses, content contains OutputText
for part in content:
    print(f"  Type: {part.type.name}")
    if isinstance(part, OutputText):
        print(f"  Text: {part.text}")

# Verify content matches text property
assert content[0].text == response.text
print(f"\nContent matches .text: {content[0].text == response.text}")


# =============================================================================
# Why Content Matters: Future-Proofing
# =============================================================================

print("\n=== Future-Proofing Pattern ===")

# Today: Text-only responses
# Future: Multimodal responses (images, audio, etc.)
#
# This code will work unchanged when models return multimodal output:
#
# for part in response.content:
#     match part.type:
#         case ContentType.OUTPUT_TEXT:
#             display_text(part.text)
#         case ContentType.OUTPUT_IMAGE:
#             display_image(part.data, part.media_type)
#         case ContentType.OUTPUT_AUDIO:
#             play_audio(part.data, part.format)

# For now, check the type
part = content[0]
if part.type == ContentType.OUTPUT_TEXT:
    print(f"Text output: {part.text[:50]}...")


# =============================================================================
# Response.prompt - Audit Trail
# =============================================================================

print("\n=== Response.prompt ===")

chat = Chat("Qwen/Qwen3-0.6B", system="You are a geography expert.")
response = chat.send("What is the capital of France?", max_tokens=10)

# Access the fully rendered prompt
prompt = response.prompt
if prompt is not None:
    print("Rendered prompt:")
    print("-" * 40)
    # Show first 500 chars to avoid overwhelming output
    print(prompt[:500] if len(prompt) > 500 else prompt)
    print("-" * 40)

    # Useful assertions for debugging
    assert "geography expert" in prompt, "System prompt missing!"
    assert "France" in prompt, "User message missing!"
    print("\nPrompt contains system and user messages: OK")


# =============================================================================
# Debugging Template Issues
# =============================================================================

print("\n=== Debugging Use Case ===")

# Common debugging pattern: verify template rendering
def debug_chat_template(chat: Chat, message: str) -> None:
    """Debug helper to verify chat template rendering."""
    response = chat.send(message, max_tokens=5)

    if response.prompt is None:
        print("Warning: Prompt not captured (remote API?)")
        return

    print(f"User said: {message!r}")
    print(f"Prompt length: {len(response.prompt)} chars")

    # Check for common template markers
    markers = ["<|im_start|>", "<|im_end|>", "[INST]", "<s>", "</s>"]
    found = [m for m in markers if m in response.prompt]
    if found:
        print(f"Template markers: {found}")

    # Verify message appears in prompt
    if message in response.prompt:
        print("Message correctly included in prompt")
    else:
        print("WARNING: Message not found in rendered prompt!")


debug_chat = Chat("Qwen/Qwen3-0.6B")
debug_chat_template(debug_chat, "Hello world")


# =============================================================================
# Streaming Response Properties
# =============================================================================

print("\n=== Streaming Response ===")

chat = Chat("Qwen/Qwen3-0.6B")
streaming = chat("Tell me a short joke", max_tokens=30)

# Consume the stream
print("Streaming: ", end="", flush=True)
full_text = ""
for token in streaming:
    print(token, end="", flush=True)
    full_text += str(token)
print()

# After iteration, content and prompt are available
print(f"\nContent type: {streaming.content[0].type.name}")
print(f"Content matches accumulated: {streaming.content[0].text == full_text}")

if streaming.prompt is not None:
    print(f"Prompt captured: {len(streaming.prompt)} chars")


# =============================================================================
# Multi-turn Conversation Prompt
# =============================================================================

print("\n=== Multi-turn Prompt ===")

chat = Chat("Qwen/Qwen3-0.6B", system="You remember everything.")
r1 = chat.send("My favorite color is blue.", max_tokens=5)
r2 = r1.append("What is my favorite color?", max_tokens=10)

# The prompt for r2 contains the full conversation
if r2.prompt is not None:
    print("Second response prompt contains:")
    print(f"  - System: {'remember' in r2.prompt}")
    print(f"  - First user: {'blue' in r2.prompt}")
    print(f"  - First assistant: {r1.text[:20] in r2.prompt if r1.text else 'N/A'}")
    print(f"  - Second user: {'favorite color' in r2.prompt}")


"""
Topics covered:
* response.content
* response.prompt
* multimodal.symmetry
* debug.template
* streaming.properties

Related:
* examples/developers/chat/history_access.py
"""
