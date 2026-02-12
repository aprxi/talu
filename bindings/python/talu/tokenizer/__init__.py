"""
Tokenizer module - Text encoding and decoding.

Provides:
- Tokenizer: Text-to-token encoding and token-to-text decoding
- TokenArray: Zero-copy token container
- TokenArrayView: View into a TokenArray slice
- TokenOffset: Byte offset pair for token-to-text mapping
- BatchEncoding: Batch tokenization container
"""

from .batch import BatchEncoding
from .template import ChatTemplate as ChatTemplate
from .template import apply_chat_template as apply_chat_template
from .token_array import TokenArray, TokenArrayView, TokenOffset
from .tokenizer import Tokenizer

# =============================================================================
# Public API - See talu/__init__.py for documentation mapping guidelines
# =============================================================================
__all__ = [
    # Core
    "Tokenizer",
    # Token Containers
    "TokenArray",
    "TokenArrayView",
    "TokenOffset",
    # Batch Encoding
    "BatchEncoding",
]
