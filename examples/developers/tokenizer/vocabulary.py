"""Vocabulary - Inspect and work with the token vocabulary.

Primary API: talu.Tokenizer
Scope: Single

Every model has a vocabulary mapping token strings to IDs. Understanding
the vocabulary helps with debugging tokenization issues and building
token-level features like logit bias.

Related:
    - examples/basics/03_tokenize_text.py
"""

import talu

tokenizer = talu.Tokenizer("Qwen/Qwen3-0.6B")

# =============================================================================
# Basic vocabulary info
# =============================================================================

print(f"Vocabulary size: {tokenizer.vocab_size:,}")

# =============================================================================
# Token ID <-> String conversion
# =============================================================================

# Get the string for a token ID
token_str = tokenizer.id_to_token(9707)
print(f"\nToken ID 9707 -> '{token_str}'")

# Get the ID for a token string
token_id = tokenizer.token_to_id("Hello")
print(f"Token 'Hello' -> {token_id}")

# Check if a string is a single token
if "Hello" in tokenizer:
    print("'Hello' is a single token")
else:
    print("'Hello' is split into multiple tokens")

# =============================================================================
# Batch conversion
# =============================================================================

# Convert multiple IDs to tokens
ids = [9707, 1879, 151645]
tokens = tokenizer.convert_ids_to_tokens(ids)
print(f"\nIDs {ids} -> {tokens}")

# Convert multiple tokens to IDs
strings = ["Hello", "World", "<|im_end|>"]
ids = tokenizer.convert_tokens_to_ids(strings)
print(f"Tokens {strings} -> {ids}")

# =============================================================================
# See how text is tokenized
# =============================================================================

# tokenize() returns the token strings (not IDs)
text = "Hello, world! How are you?"
pieces = tokenizer.tokenize(text)
print(f"\nTokenization of '{text}':")
print(f"  Pieces: {pieces}")
print(f"  Count: {len(pieces)}")

# Compare with encode (which returns IDs)
ids = tokenizer.encode(text, special_tokens=False)
print(f"  IDs: {ids.tolist()}")

# =============================================================================
# Debug tokenization issues
# =============================================================================

# Sometimes you need to see exact bytes (for unicode edge cases)
text = "caf\u00e9"  # café with combining character
pieces_str = tokenizer.tokenize(text, return_bytes=False)
pieces_bytes = tokenizer.tokenize(text, return_bytes=True)

print(f"\nTokenizing '{text}':")
print(f"  As strings: {pieces_str}")
print(f"  As bytes: {pieces_bytes}")

# =============================================================================
# Get full vocabulary (memory intensive)
# =============================================================================

# Warning: This loads the entire vocabulary into memory
# Only do this if you need to search/filter tokens
vocab = tokenizer.get_vocab()
print(f"\nFull vocabulary loaded: {len(vocab):,} entries")

# Find tokens containing a substring
python_tokens = [tok for tok in vocab if "python" in tok.lower()]
print(f"Tokens containing 'python': {python_tokens[:10]}")

# Find tokens in an ID range
sample_tokens = {tok: id for tok, id in vocab.items() if 100 <= id < 110}
print(f"Tokens with IDs 100-109: {sample_tokens}")

# =============================================================================
# Build logit bias (for constrained generation)
# =============================================================================


def build_logit_bias(
    tokenizer: talu.Tokenizer,
    boost_words: list[str],
    suppress_words: list[str],
    boost_value: float = 5.0,
    suppress_value: float = -100.0,
) -> dict[int, float]:
    """Build a logit bias dict for generation."""
    bias = {}

    for word in boost_words:
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            bias[token_id] = boost_value

    for word in suppress_words:
        token_id = tokenizer.token_to_id(word)
        if token_id is not None:
            bias[token_id] = suppress_value

    return bias


# Example: boost "yes"/"no" tokens, suppress profanity
logit_bias = build_logit_bias(
    tokenizer,
    boost_words=["Yes", "No", "yes", "no"],
    suppress_words=["badword"],  # Add actual words to suppress
)
print(f"\nLogit bias: {logit_bias}")

# =============================================================================
# Token frequency analysis
# =============================================================================


def analyze_tokenization(tokenizer: talu.Tokenizer, text: str) -> dict:
    """Analyze how text is tokenized."""
    tokens = tokenizer.encode(text, special_tokens=False)
    pieces = tokenizer.tokenize(text)

    # Character to token ratio (lower = more efficient)
    char_per_token = len(text) / len(tokens) if tokens else 0

    return {
        "text_length": len(text),
        "token_count": len(tokens),
        "chars_per_token": round(char_per_token, 2),
        "tokens": pieces[:10],  # First 10 for preview
    }


# Compare tokenization efficiency for different content
samples = [
    "Hello, how are you today?",  # Common English
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",  # Code
    "日本語のテキスト",  # Japanese
    "https://example.com/path?query=value",  # URL
]

print("\nTokenization efficiency:")
for sample in samples:
    analysis = analyze_tokenization(tokenizer, sample)
    print(f"  '{sample[:30]}...' -> {analysis['chars_per_token']} chars/token")

"""
Topics covered:
* tokenizer.encode
* tokenizer.decode
"""
