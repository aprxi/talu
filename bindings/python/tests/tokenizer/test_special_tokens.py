"""Smoke tests for tokenizer special tokens module.

Maps to: talu/tokenizer/special_tokens.py
"""

from talu.tokenizer.special_tokens import SpecialTokensMixin


def test_special_tokens_mixin_export() -> None:
    assert SpecialTokensMixin.__name__ == "SpecialTokensMixin"

