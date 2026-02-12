"""
Tokenizer-specific fixtures.

Provides fixtures for tokenizer testing including:
- Model registry with supported models
- Tokenizer and transformers fixtures
- Test string collections
- Minimal test tokenizer for unit testing without model files
"""

import pytest

from tests.conftest import MODEL_REGISTRY
from tests.fixtures import find_cached_model_path

# =============================================================================
# Minimal Test Tokenizer JSON
# =============================================================================

# Minimal BPE tokenizer JSON for unit testing.
# Uses byte-level vocabulary (printable ASCII) plus special tokens.
# This allows testing tokenizer functionality without needing model files.
MINIMAL_TOKENIZER_JSON = """{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "<pad>": 0,
      "<s>": 1,
      "</s>": 2,
      "<unk>": 3,
      " ": 4,
      "!": 5,
      "\\"": 6,
      "#": 7,
      "$": 8,
      "%": 9,
      "&": 10,
      "'": 11,
      "(": 12,
      ")": 13,
      "*": 14,
      "+": 15,
      ",": 16,
      "-": 17,
      ".": 18,
      "/": 19,
      "0": 20,
      "1": 21,
      "2": 22,
      "3": 23,
      "4": 24,
      "5": 25,
      "6": 26,
      "7": 27,
      "8": 28,
      "9": 29,
      ":": 30,
      ";": 31,
      "<": 32,
      "=": 33,
      ">": 34,
      "?": 35,
      "@": 36,
      "A": 37,
      "B": 38,
      "C": 39,
      "D": 40,
      "E": 41,
      "F": 42,
      "G": 43,
      "H": 44,
      "I": 45,
      "J": 46,
      "K": 47,
      "L": 48,
      "M": 49,
      "N": 50,
      "O": 51,
      "P": 52,
      "Q": 53,
      "R": 54,
      "S": 55,
      "T": 56,
      "U": 57,
      "V": 58,
      "W": 59,
      "X": 60,
      "Y": 61,
      "Z": 62,
      "[": 63,
      "\\\\": 64,
      "]": 65,
      "^": 66,
      "_": 67,
      "`": 68,
      "a": 69,
      "b": 70,
      "c": 71,
      "d": 72,
      "e": 73,
      "f": 74,
      "g": 75,
      "h": 76,
      "i": 77,
      "j": 78,
      "k": 79,
      "l": 80,
      "m": 81,
      "n": 82,
      "o": 83,
      "p": 84,
      "q": 85,
      "r": 86,
      "s": 87,
      "t": 88,
      "u": 89,
      "v": 90,
      "w": 91,
      "x": 92,
      "y": 93,
      "z": 94,
      "{": 95,
      "|": 96,
      "}": 97,
      "~": 98
    },
    "merges": []
  },
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true},
    {"id": 3, "content": "<unk>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "post_processor": null,
  "decoder": {"type": "ByteLevel"}
}"""

# MODEL_REGISTRY is imported from tests.conftest (centralized model config)


# =============================================================================
# Test String Collections (from scripts/validate_tokenizer.py)
# =============================================================================

BASIC_STRINGS = [
    "Hello, world!",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog.",
]

NUMBER_STRINGS = [
    "What is 2+2?",
    "The answer is 42.",
    "3.14159 is approximately pi.",
    "100,000 people attended.",
]

PUNCTUATION_STRINGS = [
    "Wait... what?!",
    'He said: "Hello!"',
    "It's a test's test.",
    "email@example.com",
    "https://example.com/path?query=value&other=123",
]

CONTRACTION_STRINGS = [
    "I'm going to the store.",
    "We've been waiting.",
    "They're not here.",
    "It's John's book.",
    "I'd like that.",
    "We'll see.",
]

WHITESPACE_STRINGS = [
    "Multiple   spaces   here",
    "Tabs\there\tand\tthere",
    "Line\nbreaks\nincluded",
]

UNICODE_STRINGS = [
    "Cafe resume naive",  # ASCII approximation
    "Cafe\u0301 re\u0301sume\u0301 nai\u0308ve",  # With combining chars
    "Cafe\u0301",  # Accent aigu
]

MULTILINGUAL_STRINGS = [
    ("japanese", "日本語テスト"),
    ("chinese", "中文测试"),
    ("korean", "한국어 테스트"),
    ("russian", "Привет мир"),
    ("arabic", "مرحبا بالعالم"),
    ("emoji", "🎉 Emoji test 🚀"),
    ("mixed_scripts", "Hello 世界!"),
    ("mixed_numbers", "Testing 123 日本語"),
]

CODE_STRINGS = [
    "def foo(): return 42",
    "if (x > 0) { print(x); }",
    "<html><body>Test</body></html>",
]

EDGE_CASE_STRINGS = [
    "",  # Empty string
    " ",  # Single space
    "   ",  # Multiple spaces
    "a",  # Single char
    "A" * 100,  # Repeated char
]

SPECIAL_TOKEN_STRINGS = [
    "<|endoftext|>",
    "[CLS] test [SEP]",
    "<s>test</s>",
    "<bos>test<eos>",
]

# Model-specific special tokens
GRANITE_SPECIAL_TOKENS = [
    "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>",
    "<|start_of_role|>assistant<|end_of_role|>",
]

PHI_SPECIAL_TOKENS = [
    "<|user|>",
    "<|user|>Hi",
    "<|user|>Hi<|end|><|assistant|>",
    "<|assistant|>Hello!<|end|>",
    "<|system|>You are helpful.<|end|>",
]

# All test strings combined
ALL_TEST_STRINGS = (
    BASIC_STRINGS
    + NUMBER_STRINGS
    + PUNCTUATION_STRINGS
    + CONTRACTION_STRINGS
    + WHITESPACE_STRINGS
    + UNICODE_STRINGS
    + [s for _, s in MULTILINGUAL_STRINGS]
    + CODE_STRINGS
    + EDGE_CASE_STRINGS
    + SPECIAL_TOKEN_STRINGS
    + GRANITE_SPECIAL_TOKENS
    + PHI_SPECIAL_TOKENS
)


# =============================================================================
# Helper Functions
# =============================================================================

# find_cached_model_path is imported from tests.fixtures


def get_available_models() -> list[tuple[str, str, str]]:
    """
    Get list of available models for testing.

    Returns:
        List of (model_name, hf_id, local_path) tuples
    """
    available = []
    for name, info in MODEL_REGISTRY.items():
        local_path = find_cached_model_path(info["hf_id"])
        if local_path:
            available.append((name, info["hf_id"], local_path))
    return available


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def available_models() -> list[tuple[str, str, str]]:
    """Get list of available models (name, hf_id, path)."""
    return get_available_models()


@pytest.fixture(scope="session")
def model_registry() -> dict:
    """Get the full model registry."""
    return MODEL_REGISTRY


@pytest.fixture(scope="session")
def transformers():
    """Import transformers library."""
    pytest.importorskip("transformers")
    import transformers

    return transformers


@pytest.fixture(scope="session")
def hf_tokenizer_cache(transformers) -> dict:
    """
    Shared cache for HuggingFace tokenizers.

    Avoids reloading tokenizers multiple times.
    """
    return {}


def load_hf_tokenizer(model_path: str, cache: dict, transformers):
    """Load HuggingFace tokenizer with caching.

    Resolves model URIs (e.g., "Qwen/Qwen3-0.6B-GAF4") to local paths
    via talu's repository before passing to AutoTokenizer.
    """
    if model_path not in cache:
        from talu.repository import resolve_path

        local_path = resolve_path(model_path)
        cache[model_path] = transformers.AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    return cache[model_path]


@pytest.fixture(scope="session")
def hf_tokenizer(test_model_path, hf_tokenizer_cache, transformers):
    """Get HuggingFace tokenizer for the test model."""
    return load_hf_tokenizer(test_model_path, hf_tokenizer_cache, transformers)


@pytest.fixture(scope="session")
def tokenizer(talu, test_model_path):
    """Get talu Tokenizer for the test model.

    Scoped to session to avoid reloading tokenizer files.
    The Tokenizer is stateless for encode/decode, so sharing is safe.
    """
    return talu.Tokenizer(test_model_path)


@pytest.fixture(scope="session")
def tokenizer_cache(talu) -> dict:
    """
    Shared cache for talu Tokenizers across all models.

    Use with load_tokenizer() to avoid reloading models.
    """
    return {}


def load_tokenizer(model_path: str, cache: dict, talu):
    """Load talu Tokenizer with caching."""
    if model_path not in cache:
        cache[model_path] = talu.Tokenizer(model_path)
    return cache[model_path]


# =============================================================================
# Test String Fixtures
# =============================================================================


@pytest.fixture
def basic_strings():
    """Basic test strings."""
    return BASIC_STRINGS


@pytest.fixture
def number_strings():
    """Number and math test strings."""
    return NUMBER_STRINGS


@pytest.fixture
def punctuation_strings():
    """Punctuation-heavy test strings."""
    return PUNCTUATION_STRINGS


@pytest.fixture
def contraction_strings():
    """Contraction test strings."""
    return CONTRACTION_STRINGS


@pytest.fixture
def whitespace_strings():
    """Whitespace variation test strings."""
    return WHITESPACE_STRINGS


@pytest.fixture
def unicode_strings():
    """Unicode test strings."""
    return UNICODE_STRINGS


@pytest.fixture
def multilingual_strings():
    """Multilingual test strings (name, string) tuples."""
    return MULTILINGUAL_STRINGS


@pytest.fixture
def code_strings():
    """Code-like test strings."""
    return CODE_STRINGS


@pytest.fixture
def edge_case_strings():
    """Edge case test strings."""
    return EDGE_CASE_STRINGS


@pytest.fixture
def special_token_strings():
    """Special token test strings."""
    return SPECIAL_TOKEN_STRINGS


@pytest.fixture
def all_test_strings():
    """All test strings combined."""
    return ALL_TEST_STRINGS


# =============================================================================
# Minimal Test Tokenizer Fixtures
# =============================================================================


@pytest.fixture
def minimal_tokenizer_json():
    """Get the minimal test tokenizer JSON string."""
    return MINIMAL_TOKENIZER_JSON


@pytest.fixture(scope="session")
def minimal_tokenizer(talu):
    """
    Get a minimal tokenizer created from JSON for unit testing.

    This tokenizer does NOT require model files and can be used for:
    - Testing tokenizer API without model dependencies
    - Testing error paths and edge cases
    - Fast unit tests that don't need real model behavior

    The tokenizer uses byte-level BPE with printable ASCII characters.
    Each ASCII character (32-126) maps to its own token ID.

    Note: This tokenizer has no merges, so it tokenizes to individual bytes.
    """
    return talu.Tokenizer.from_json(MINIMAL_TOKENIZER_JSON)
