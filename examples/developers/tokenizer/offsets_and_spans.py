"""Token Offsets - Map tokens back to source text positions.

Primary API: talu.Tokenizer
Scope: Single

TokenArray provides byte-level offsets that map each token back to its
position in the original text. This is essential for:
- Highlighting which part of text a token represents
- Building token-level annotations
- Debugging tokenization issues with Unicode
- NER and other span-based tasks

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Basic offset access
# =============================================================================

text = "Hello world"
tokens = tokenizer.encode(text, special_tokens=False)

print(f"Text: '{text}'")
print(f"Tokens: {tokens.tolist()}")
print(f"Token strings: {tokenizer.tokenize(text)}")

# Offsets are (start, end) byte positions into the source text
print("\nToken offsets:")
for i, offset in enumerate(tokens.offsets):
    # Use slice() to extract the text span (handles Unicode correctly)
    span = offset.slice(text)
    print(f"  Token {i}: {offset} -> '{span}'")

# =============================================================================
# Unicode handling
# =============================================================================

# Offsets are UTF-8 byte indices, not character indices
# The slice() method handles this automatically

text_unicode = "Hello 🎉 world"
tokens = tokenizer.encode(text_unicode, special_tokens=False)

print(f"\nUnicode text: '{text_unicode}'")
print(f"Tokens: {tokens.tolist()}")

print("\nOffset mapping (note: emoji is 4 bytes in UTF-8):")
for i, offset in enumerate(tokens.offsets):
    span = offset.slice(text_unicode)
    print(f"  Token {i}: bytes {offset.start}-{offset.end} -> '{span}'")

# =============================================================================
# Handling byte-level BPE edge cases
# =============================================================================

# Some tokenizers (GPT-2, Qwen) use byte-level BPE which can split
# multi-byte UTF-8 characters across tokens. Use errors="replace"
# to handle partial sequences gracefully.

text_complex = "café naïve"
tokens = tokenizer.encode(text_complex, special_tokens=False)

print(f"\nComplex text: '{text_complex}'")
for i, offset in enumerate(tokens.offsets):
    # Safe extraction with replacement for partial UTF-8 sequences
    span = offset.slice(text_complex, errors="replace")
    print(f"  Token {i}: {offset} -> '{span}'")

# =============================================================================
# Special tokens have (0, 0) offsets
# =============================================================================

text = "Hello"
tokens = tokenizer.encode(text, special_tokens=True)  # With BOS/EOS

print(f"\nWith special tokens:")
print(f"Tokens: {tokens.tolist()}")
for i, offset in enumerate(tokens.offsets):
    if offset == (0, 0):
        print(f"  Token {i}: {offset} -> <special token>")
    else:
        span = offset.slice(text)
        print(f"  Token {i}: {offset} -> '{span}'")

# =============================================================================
# Token-level annotation example
# =============================================================================


def highlight_tokens(text: str, tokenizer: talu.Tokenizer) -> str:
    """Add markers around each token in the text."""
    tokens = tokenizer.encode(text, special_tokens=False)

    # Build highlighted text by inserting markers
    result = []
    last_end = 0
    text_bytes = text.encode("utf-8")

    for offset in tokens.offsets:
        # Add any text between tokens (shouldn't happen normally)
        if offset.start > last_end:
            result.append(text_bytes[last_end : offset.start].decode("utf-8"))

        # Add marked token
        token_text = text_bytes[offset.start : offset.end].decode("utf-8")
        result.append(f"[{token_text}]")
        last_end = offset.end

    return "".join(result)


highlighted = highlight_tokens("Hello world!", tokenizer)
print(f"\nHighlighted: {highlighted}")

# =============================================================================
# NER-style span extraction
# =============================================================================


def get_token_spans(text: str, tokenizer: talu.Tokenizer) -> list[tuple[int, int, str]]:
    """Get character spans for each token (not byte spans)."""
    tokens = tokenizer.encode(text, special_tokens=False)
    text_bytes = text.encode("utf-8")

    spans = []
    for i, offset in enumerate(tokens.offsets):
        # Convert byte offsets to character offsets
        char_start = len(text_bytes[: offset.start].decode("utf-8"))
        char_end = len(text_bytes[: offset.end].decode("utf-8"))
        token_str = tokenizer.id_to_token(tokens[i])
        spans.append((char_start, char_end, token_str))

    return spans


text = "The café is nice"
spans = get_token_spans(text, tokenizer)
print(f"\nCharacter spans for '{text}':")
for start, end, token in spans:
    print(f"  chars {start}-{end}: '{text[start:end]}' (token: {token!r})")

# =============================================================================
# Tuple unpacking
# =============================================================================

# TokenOffset supports tuple unpacking for convenience
offset = tokens.offsets[0]
start, end = offset  # Unpack like a tuple
print(f"\nTuple unpacking: start={start}, end={end}")

# Also supports comparison with tuples
print(f"offset == (0, 3): {offset == (0, 3)}")

"""
Topics covered:
* tokenizer.encode
* tokenizer.decode
"""
