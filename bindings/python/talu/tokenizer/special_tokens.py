"""
Special token handling for tokenizers.

Provides properties and methods for accessing special tokens (EOS, BOS, PAD, UNK).
This is a mixin class used by Tokenizer.
"""

from __future__ import annotations


class SpecialTokensMixin:
    """
    Special token properties and methods for Tokenizer.

    Requires the following attributes on the implementing class:
    - _eos_tokens: tuple[int, ...]
    - _bos_token_id: int | None
    - _unk_token_id: int | None
    - _pad_token_id: int | None
    - id_to_token: Callable[[int], str | None]
    """

    # Type hints for attributes provided by Tokenizer
    _eos_tokens: tuple[int, ...]
    _bos_token_id: int | None
    _unk_token_id: int | None
    _pad_token_id: int | None

    def id_to_token(self, token_id: int) -> str | None:
        """Get the string representation of a token ID (provided by Tokenizer)."""
        ...

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """
        Token IDs that signal end of generation.

        Returns an immutable, ordered tuple of deduplicated EOS token IDs.
        Many models have multiple EOS tokens (e.g., Qwen, Llama 3, Gemma 3).

        Returns
        -------
            Tuple of EOS token IDs. May be empty if no EOS tokens are configured.

        Example:
            >>> tokenizer.eos_token_ids
            (151643, 151644, 151645)
        """
        return self._eos_tokens

    @property
    def eos_tokens(self) -> list[str]:
        """
        String representations of all EOS tokens.

        Returns
        -------
            List of EOS token strings.

        Example:
            >>> tokenizer.eos_tokens
            ['<|endoftext|>', '<|im_end|>', '<|end|>']
        """
        result = []
        for tid in self._eos_tokens:
            token = self.id_to_token(tid)
            if token is not None:
                result.append(token)
        return result

    @property
    def bos_token_id(self) -> int | None:
        """
        Beginning-of-sequence token ID.

        Returns None if the model doesn't use a BOS token (e.g., Qwen3).

        Example:
            >>> tokenizer.bos_token_id  # Llama 3.2
            128000
            >>> tokenizer.bos_token_id  # Qwen3
            None
        """
        return self._bos_token_id

    @property
    def bos_token(self) -> str | None:
        """
        String representation of the BOS token.

        Example:
            >>> tokenizer.bos_token  # Llama 3.2
            '<|begin_of_text|>'
        """
        if self._bos_token_id is None:
            return None
        return self.id_to_token(self._bos_token_id)

    @property
    def unk_token_id(self) -> int | None:
        """Unknown token ID."""
        return self._unk_token_id

    @property
    def unk_token(self) -> str | None:
        """String representation of the unknown token."""
        if self._unk_token_id is None:
            return None
        return self.id_to_token(self._unk_token_id)

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        return self._pad_token_id

    @property
    def pad_token(self) -> str | None:
        """String representation of the padding token."""
        if self._pad_token_id is None:
            return None
        return self.id_to_token(self._pad_token_id)

    @property
    def special_ids(self) -> frozenset[int]:
        """
        Immutable set of all special token IDs.

        Includes EOS, BOS, UNK, and PAD tokens. Use for fast O(1) membership testing.

        Returns
        -------
            Frozen set of all special token IDs.

        Example:
            >>> 128001 in tokenizer.special_ids
            True
        """
        ids: set[int] = set(self._eos_tokens)
        if self._bos_token_id is not None:
            ids.add(self._bos_token_id)
        if self._unk_token_id is not None:
            ids.add(self._unk_token_id)
        if self._pad_token_id is not None:
            ids.add(self._pad_token_id)
        return frozenset(ids)

    def is_special_id(self, token_id: int) -> bool:
        """
        Check if a token ID is a special token.

        Args:
            token_id: The token ID to check.

        Returns
        -------
            True if the token ID is a special token.

        Example:
            >>> tokenizer.is_special_id(128001)  # EOS token
            True
        """
        if token_id in self._eos_tokens:
            return True
        if self._bos_token_id is not None and token_id == self._bos_token_id:
            return True
        if self._unk_token_id is not None and token_id == self._unk_token_id:
            return True
        if self._pad_token_id is not None and token_id == self._pad_token_id:
            return True
        return False

    def primary_eos_token_id(self) -> int | None:
        """
        Get the primary EOS token ID for insertion.

        Use this when inserting an EOS token. For detection/stopping,
        use ``token_id in eos_token_ids`` instead.

        Returns
        -------
            The primary EOS token ID, or None if no EOS tokens are configured.

        Example:
            >>> eos_id = tokenizer.primary_eos_token_id()
            >>> tokens.append(eos_id)
        """
        if self._eos_tokens:
            return self._eos_tokens[0]
        return None
