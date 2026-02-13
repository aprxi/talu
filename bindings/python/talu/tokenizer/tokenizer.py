"""
Text encoding and decoding.

Provides the Tokenizer class for encoding text to tokens and decoding tokens
back to text.
"""

from typing import Any

from .._bindings import get_last_error
from .._logging import scoped_logger
from .._native import DecodeOptionsC as DecodeOptions
from .._native import EncodeOptions
from ..exceptions import ModelError, StateError, TokenizerError, ValidationError
from ._bindings import (
    call_apply_chat_template_string,
    call_buffer_create_from_owned,
    call_buffer_get_data_ptr,
    call_encode_result_free,
    call_resolve_model_path,
    call_tokenizer_create,
    call_tokenizer_create_from_json,
    call_tokenizer_decode,
    call_tokenizer_encode,
    call_tokenizer_encode_batch,
    call_tokenizer_free,
    call_tokenizer_get_eos_tokens,
    call_tokenizer_get_model_max_length,
    call_tokenizer_get_special_tokens,
    call_tokenizer_get_vocab,
    call_tokenizer_get_vocab_size,
    call_tokenizer_id_to_token,
    call_tokenizer_token_to_id,
    call_tokenizer_tokenize,
    call_tokenizer_tokenize_bytes,
    free_tokenize_bytes_result,
    free_tokenize_result,
)
from .batch import BatchEncoding
from .special_tokens import SpecialTokensMixin
from .template import ChatTemplate
from .token_array import TokenArray, TokenOffset

logger = scoped_logger("tokenizer")


class Tokenizer(SpecialTokensMixin):
    """
    Text-to-token encoder and token-to-text decoder.

    Converts text into token IDs from the model's vocabulary and back.
    Thread-safe after construction.

    Attributes
    ----------
    model_path : str
        Resolved path to the model directory.
    vocab_size : int
        Number of tokens in the vocabulary.
    eos_token_ids : list[int]
        Token IDs that signal end of generation.
    bos_token_id : int | None
        Beginning-of-sequence token ID.
    pad_token_id : int | None
        Padding token ID.
    padding_side : str
        Side for padding ("left" or "right").
    
    Example:
        >>> tokenizer = Tokenizer("Qwen/Qwen3-0.6B")
        >>> tokens = tokenizer.encode("Hello world")
        >>> text = tokenizer.decode(tokens)
    """

    __slots__ = (
        "_model_dir",
        "_ptr",
        "_eos_tokens",
        "_bos_token_id",
        "_unk_token_id",
        "_pad_token_id",
        "_chat_template",
        "_chat_template_str",  # For from_json() with custom template
        "_bos_token_str",  # For from_json() with custom tokens
        "_eos_token_str",  # For from_json() with custom tokens
        "_padding_side",
        "_truncation_side",
    )

    def __init__(
        self,
        model: str,
        *,
        padding_side: str = "left",
        truncation_side: str = "right",
    ):
        """
        Create a tokenizer for a model.

        Args:
            model: Path to model directory or model ID (e.g., "Qwen/Qwen3-0.6B").
            padding_side: Default padding side ("left" or "right"). Default "left"
                for generation models.
            truncation_side: Default truncation side ("left" or "right"). Default
                "right" keeps beginning of text. Use "left" for RAG to keep recent
                context.

        Raises
        ------
            TokenizerError: If the model path is invalid or tokenizer files are missing.
            ValidationError: If padding_side or truncation_side is not "left" or "right".
            ModelError: If the model path cannot be resolved.
        """
        if padding_side not in ("left", "right"):
            from ..exceptions import ValidationError

            raise ValidationError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
        if truncation_side not in ("left", "right"):
            from ..exceptions import ValidationError

            raise ValidationError(
                f"truncation_side must be 'left' or 'right', got {truncation_side!r}"
            )

        logger.debug("Creating tokenizer", extra={"model": model})
        self._model_dir = self._resolve_model_path(model)
        self._ptr = self._create_handle()
        if not self._ptr:
            error = get_last_error() or "unknown error"
            raise TokenizerError(f"Failed to load tokenizer from '{self._model_dir}': {error}")
        self._load_eos_tokens()
        self._load_special_tokens()
        self._chat_template = ChatTemplate(self._model_dir)
        self._chat_template_str = None  # Not used for model-loaded tokenizers
        self._bos_token_str = None  # Not used for model-loaded tokenizers
        self._eos_token_str = None  # Not used for model-loaded tokenizers
        self._padding_side = padding_side
        self._truncation_side = truncation_side
        logger.debug("Tokenizer loaded", extra={"path": self._model_dir})

    @classmethod
    def from_json(
        cls,
        json_content: str | bytes,
        *,
        chat_template: str | None = None,
        bos_token: str = "",
        eos_token: str = "",
        padding_side: str = "left",
        truncation_side: str = "right",
    ) -> "Tokenizer":
        """
        Create a tokenizer directly from JSON content.

        Creates a standalone tokenizer without needing a model directory.
        Useful for custom tokenizers, serverless deployments, or testing.

        Args:
            json_content: The tokenizer.json content as string or bytes.
            chat_template: Optional Jinja2 chat template string. If provided,
                enables apply_chat_template() on this tokenizer.
            bos_token: Beginning-of-sequence token string for chat templates.
            eos_token: End-of-sequence token string for chat templates.
            padding_side: Default padding side ("left" or "right"). Default "left".
            truncation_side: Default truncation side ("left" or "right"). Default "right".

        Returns
        -------
            A new Tokenizer instance.

        Raises
        ------
            TokenizerError: If the JSON content is invalid.
            ValidationError: If padding_side or truncation_side is invalid.

        Example
        -------
            >>> json = '{"version": "1.0", "model": {"type": "BPE", ...}}'
            >>> template = "{% for m in messages %}{{ m.content }}{% endfor %}"
            >>> tok = Tokenizer.from_json(json, chat_template=template)
            >>> prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
        """
        if padding_side not in ("left", "right"):
            raise ValidationError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
        if truncation_side not in ("left", "right"):
            raise ValidationError(
                f"truncation_side must be 'left' or 'right', got {truncation_side!r}"
            )

        # Convert string to bytes if needed
        if isinstance(json_content, str):
            json_bytes = json_content.encode("utf-8")
        else:
            json_bytes = json_content

        logger.debug("Creating tokenizer from JSON", extra={"json_len": len(json_bytes)})

        # Create tokenizer handle from JSON
        code, ptr = call_tokenizer_create_from_json(json_bytes)
        if code != 0 or not ptr:
            error = get_last_error() or "invalid JSON content"
            raise TokenizerError(f"Failed to create tokenizer from JSON: {error}")

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._model_dir = ""  # No model directory for JSON-created tokenizers
        instance._ptr = ptr
        instance._load_eos_tokens()
        instance._load_special_tokens()
        instance._chat_template = None  # No ChatTemplate object
        instance._chat_template_str = chat_template  # Store template string for apply_chat_template
        instance._bos_token_str = bos_token
        instance._eos_token_str = eos_token
        instance._padding_side = padding_side
        instance._truncation_side = truncation_side

        logger.debug("Tokenizer created from JSON", extra={"vocab_size": instance.vocab_size})
        return instance

    def _resolve_model_path(self, model: str) -> str:
        """Resolve model path (local or remote)."""
        code, model_dir = call_resolve_model_path(model.encode("utf-8"))
        if code != 0 or model_dir is None:
            error = get_last_error() or "path not found and not a valid model ID"
            raise ModelError(f"Failed to resolve model '{model}': {error}")
        return model_dir

    def _create_handle(self):
        """Create the internal handle."""
        code, ptr = call_tokenizer_create(self._model_dir.encode("utf-8"))
        if code != 0:
            from .._bindings import check
            from ..exceptions import TaluError

            try:
                check(code)
            except TaluError as exc:
                message = str(exc)
                normalized = message.lower()
                if (
                    "resolve" not in normalized
                    and "not found" not in normalized
                    and "weights" not in normalized
                ):
                    message = f"Failed to resolve model '{self._model_dir}': {message}"
                raise type(exc)(message, code=exc.code) from exc
        return ptr

    @property
    def _handle(self) -> int:
        """Get the internal handle, raising if closed."""
        if self._ptr is None:
            raise StateError("Tokenizer is closed")
        return self._ptr

    def _load_eos_tokens(self):
        """Load EOS token IDs from the model.

        Stores as ordered, deduplicated tuple for immutability.
        """
        eos_tokens, _ = call_tokenizer_get_eos_tokens(self._handle)
        self._eos_tokens = eos_tokens

    def _load_special_tokens(self):
        """Load special token IDs from the model."""
        bos, unk, pad = call_tokenizer_get_special_tokens(self._handle)
        self._bos_token_id = bos
        self._unk_token_id = unk
        self._pad_token_id = pad

    def _free_handle(self):
        """Free the internal handle."""
        if hasattr(self, "_ptr") and self._ptr:
            call_tokenizer_free(self._ptr)
            self._ptr = None

    def close(self) -> None:
        """
        Release native tokenizer resources.

        After calling close(), the tokenizer cannot be used. Safe to call
        multiple times (idempotent).

        Raises
        ------
            None. This method never raises.
        """
        self._free_handle()

    def __enter__(self) -> "Tokenizer":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self):
        try:
            self._free_handle()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"Tokenizer({self._model_dir!r})"

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model_path(self) -> str:
        """The resolved path to the model directory."""
        return self._model_dir

    @property
    def padding_side(self) -> str:
        """Side for padding: "left" (default, for generation) or "right".

        Raises
        ------
        ValidationError
            If set to a value other than ``"left"`` or ``"right"``.
        """
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        if value not in ("left", "right"):
            from ..exceptions import ValidationError

            raise ValidationError(f"padding_side must be 'left' or 'right', got {value!r}")
        self._padding_side = value

    @property
    def truncation_side(self) -> str:
        """Side for truncation: "right" (default, keeps beginning) or "left" (keeps end).

        Raises
        ------
        ValidationError
            If set to a value other than ``"left"`` or ``"right"``.
        """
        return self._truncation_side

    @truncation_side.setter
    def truncation_side(self, value: str) -> None:
        if value not in ("left", "right"):
            from ..exceptions import ValidationError

            raise ValidationError(f"truncation_side must be 'left' or 'right', got {value!r}")
        self._truncation_side = value

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return call_tokenizer_get_vocab_size(self._handle)

    @property
    def model_max_length(self) -> int:
        """
        Maximum sequence length the model supports.

        This value is read from tokenizer_config.json (model_max_length field).
        It represents the maximum context length the model was trained with.

        When truncation=True is used without an explicit max_length, this value
        is used as the default truncation limit.

        Returns 0 if the model does not specify a maximum length.

        Example:
            >>> tokenizer.model_max_length
            32768
        """
        return call_tokenizer_get_model_max_length(self._handle)

    # =========================================================================
    # Core Encoding/Decoding
    # =========================================================================

    def encode(
        self,
        text: str | list[str],
        special_tokens: bool | set[str] = True,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        truncation_side: str | None = None,
    ) -> TokenArray | BatchEncoding:
        """
        Convert text to token IDs.

        Args:
            text: Text to tokenize. String returns TokenArray, list returns BatchEncoding.
            special_tokens: Control special token insertion. Can be:
                - True: Add all special tokens (BOS, EOS) - default
                - False: No special tokens (raw tokenization)
                - {"bos"}: Add only BOS token
                - {"eos"}: Add only EOS token
                - {"bos", "eos"}: Add both (same as True)
            truncation: If True, truncate to max_length. When max_length is not
                specified, uses model_max_length from tokenizer_config.json.
            max_length: Maximum sequence length. If None and truncation=True,
                uses model_max_length.
            truncation_side: "left" or "right". Overrides tokenizer default.
                - "right" (default): Keep beginning, truncate end
                - "left": Keep end, truncate beginning (useful for RAG)

        Returns
        -------
            TokenArray for single string, BatchEncoding for list.

        Raises
        ------
            ValidationError: If special_tokens is not bool or set[str], or
                if text is not str or list[str].
            TokenizerError: If encoding fails.
            MemoryError: If buffer allocation fails.

        Note:
            Special token behavior depends on the model's tokenizer configuration:

            - Models with postprocessor (BERT, RoBERTa): BOS/EOS are added via
              the tokenizer's postprocessor when special_tokens=True.

            - Chat models (Llama 3, Qwen3, Gemma): Special tokens are typically
              added via chat templates, not the postprocessor. For these models,
              special_tokens=True may not add BOS/EOS to raw text. Use
              apply_chat_template() for proper formatting.

            - If bos_token_id is None (e.g., Qwen3), requesting BOS is a no-op.
              Check tokenizer.bos_token_id to verify special token availability.

            For batch encoding, padding_side is inherited from the tokenizer's
            ``padding_side`` property (default "left" for generation models).
            To override for a specific batch, set the property on the result::

                batch = tokenizer.encode(["Hello", "World"])
                batch.padding_side = "right"  # Override before converting
                tensor = torch.from_dlpack(batch)

        Example:
            >>> # Default: add all special tokens
            >>> tokens = tokenizer.encode("Hello world")

            >>> # No special tokens
            >>> tokens = tokenizer.encode("Hello world", special_tokens=False)

            >>> # BOS only (useful for document snippets)
            >>> tokens = tokenizer.encode("Document...", special_tokens={"bos"})

            >>> # With truncation (uses model_max_length if no explicit max_length)
            >>> tokens = tokenizer.encode(long_text, truncation=True)

            >>> # With explicit max_length
            >>> tokens = tokenizer.encode(text, truncation=True, max_length=512)

            >>> # Left truncation for RAG (keep recent context)
            >>> tokens = tokenizer.encode(text, truncation=True, max_length=512,
            ...                           truncation_side="left")

            >>> # Check if model has BOS token
            >>> if tokenizer.bos_token_id is not None:
            ...     print(f"BOS token: {tokenizer.bos_token}")
        """
        # Resolve special_tokens to add_bos/add_eos flags
        if special_tokens is True:
            add_bos, add_eos = True, True
        elif special_tokens is False:
            add_bos, add_eos = False, False
        elif isinstance(special_tokens, set):
            add_bos = "bos" in special_tokens
            add_eos = "eos" in special_tokens
        else:
            raise ValidationError(
                f"special_tokens must be bool or set[str], got {type(special_tokens).__name__}",
                code="INVALID_ARGUMENT",
                details={"param": "special_tokens", "type": type(special_tokens).__name__},
            )

        # Resolve max_length: explicit > model_max_length > no truncation
        effective_max_length = max_length if max_length is not None else 0
        if truncation and max_length is None:
            model_limit = self.model_max_length
            if model_limit > 0:
                effective_max_length = model_limit

        # Resolve truncation_side: method arg > instance default
        eff_trunc_side = truncation_side if truncation_side else self._truncation_side

        if isinstance(text, list):
            return self._encode_batch(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                truncation=truncation,
                max_length=effective_max_length,
                truncation_side=eff_trunc_side,
            )
        if isinstance(text, str):
            return self._encode_single(
                text,
                add_bos=add_bos,
                add_eos=add_eos,
                truncation=truncation,
                max_length=effective_max_length,
                truncation_side=eff_trunc_side,
            )
        raise ValidationError(
            f"text must be str or list[str], got {type(text).__name__}",
            code="INVALID_ARGUMENT",
            details={"param": "text", "type": type(text).__name__},
        )

    def _encode_single(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        truncation: bool = False,
        max_length: int = 0,
        truncation_side: str = "right",
    ) -> TokenArray:
        """Encode a single string to TokenArray with refcounted buffer."""
        text_bytes = text.encode("utf-8")

        # Build options struct
        options = EncodeOptions(
            add_bos=1 if add_bos else 0,
            add_eos=1 if add_eos else 0,
            truncation=1 if truncation else 0,
            truncation_side=1 if truncation_side == "left" else 0,
            max_length=max_length,
        )
        result = call_tokenizer_encode(self._handle, text_bytes, options)

        if result.error_msg:
            raise TokenizerError(f"Encode failed: {result.error_msg.decode('utf-8')}")

        # Handle empty result
        if result.num_tokens == 0:
            return TokenArray(None, 0, _offsets=[])

        num_tokens = result.num_tokens

        # Extract offsets into Python objects
        offsets = [
            TokenOffset(result.offsets[i].start, result.offsets[i].end)
            for i in range(num_tokens)
        ]

        # Transfer ids ownership to SharedBuffer
        buffer_handle = call_buffer_create_from_owned(result.ids, num_tokens)
        if not buffer_handle:
            call_encode_result_free(result)
            raise MemoryError("Failed to create SharedBuffer for TokenArray")

        data_ptr = call_buffer_get_data_ptr(buffer_handle)

        # Null out ids (now owned by SharedBuffer), free remaining arrays
        result.ids = None
        call_encode_result_free(result)

        return TokenArray(
            data_ptr,
            num_tokens,
            _offsets=offsets,
            _buffer_handle=buffer_handle,
        )

    def _encode_batch(
        self,
        texts: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        truncation: bool = False,
        max_length: int = 0,
        truncation_side: str = "right",
    ) -> BatchEncoding:
        """Encode a list of strings using parallel Zig thread pool."""
        num_texts = len(texts)

        if num_texts == 0:
            return BatchEncoding(
                ids_ptr=None,
                offsets_ptr=None,
                total_tokens=0,
                num_sequences=0,
                padding_side=self._padding_side,
                pad_token_id=self._pad_token_id,
            )

        # Marshal Python strings to C
        text_bytes_list = [t.encode("utf-8") for t in texts]

        # Build options struct
        options = EncodeOptions(
            add_bos=1 if add_bos else 0,
            add_eos=1 if add_eos else 0,
            truncation=1 if truncation else 0,
            truncation_side=1 if truncation_side == "left" else 0,
            max_length=max_length,
        )
        result = call_tokenizer_encode_batch(self._handle, text_bytes_list, options)

        if result.error_msg:
            raise TokenizerError(f"Batch encode failed: {result.error_msg.decode('utf-8')}")

        return BatchEncoding(
            ids_ptr=result.ids,
            offsets_ptr=result.offsets,
            total_tokens=result.total_tokens,
            num_sequences=result.num_sequences,
            padding_side=self._padding_side,
            pad_token_id=self._pad_token_id,
        )

    def decode(
        self,
        tokens: TokenArray | list[int],
        num_tokens: int | None = None,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: Token IDs to decode.
            num_tokens: Number of tokens (only for raw pointers).
            skip_special_tokens: If True, omit special tokens from output.

        Returns
        -------
            Decoded text string.

        Raises
        ------
            ValidationError: If num_tokens is required but not provided.
            TokenizerError: If decoding fails.
        """
        if isinstance(tokens, TokenArray):
            token_list = list(tokens)
        elif isinstance(tokens, list):
            token_list = tokens
        else:
            # Raw pointer case - not supported without ctypes
            if num_tokens is None:
                raise ValidationError("num_tokens required when passing raw pointer")
            # Convert raw pointer to list - this requires accessing the array
            # For now, raise an error as raw pointer decode should use TokenArray
            raise ValidationError("Raw pointer decode not supported. Use TokenArray or list[int].")

        options = DecodeOptions(skip_special_tokens=1 if skip_special_tokens else 0)
        text, error = call_tokenizer_decode(self._handle, token_list, options)

        if error:
            raise TokenizerError(f"Decode failed: {error}")

        return text or ""

    def tokenize(self, text: str, *, return_bytes: bool = False) -> list[str] | list[bytes]:
        r"""
        Split text into token strings.

        This is useful for debugging tokenization - seeing exactly how text
        is segmented before being converted to token IDs.

        Args:
            text: Text to tokenize.
            return_bytes: If True, return raw bytes instead of strings.
                Use this for debugging when you need to see exact byte
                representations (e.g., for tokens with invalid UTF-8 or
                special byte sequences).

        Returns
        -------
            List of token strings (default) or bytes (if return_bytes=True).

        Raises
        ------
            TokenizerError: If tokenization fails.

        Example:
            >>> tokenizer.tokenize("Hello world")
            ['Hello', ' world']

            >>> # Debug mode - see raw bytes
            >>> tokenizer.tokenize("Hello", return_bytes=True)
            [b'Hello']

            >>> # Useful for debugging unicode edge cases
            >>> tokenizer.tokenize("café", return_bytes=True)
            [b'caf', b'\\xc3\\xa9']  # Shows UTF-8 encoding
        """
        text_bytes = text.encode("utf-8")

        if return_bytes:
            # Use bytes API for exact byte representation
            result = call_tokenizer_tokenize_bytes(self._handle, text_bytes)

            if result.error_msg:
                raise TokenizerError(f"Tokenize failed: {result.error_msg.decode('utf-8')}")

            if result.num_tokens == 0:
                free_tokenize_bytes_result(result)
                return []

            # Extract tokens from CSR format
            tokens: list[bytes] = []
            for i in range(result.num_tokens):
                start = result.offsets[i]
                end = result.offsets[i + 1]
                token_bytes_chunk = bytes(result.data[start:end])
                tokens.append(token_bytes_chunk)

            free_tokenize_bytes_result(result)
            return tokens
        else:
            # Use string API for normal use
            result = call_tokenizer_tokenize(self._handle, text_bytes)

            if result.error_msg:
                raise TokenizerError(f"Tokenize failed: {result.error_msg.decode('utf-8')}")

            if not result.tokens or result.num_tokens == 0:
                return []

            tokens_str: list[str] = []
            for i in range(result.num_tokens):
                token_ptr = result.tokens[i]
                if token_ptr:
                    # Use 'replace' to handle invalid UTF-8 gracefully
                    tokens_str.append(token_ptr.decode("utf-8", errors="replace"))
                else:
                    tokens_str.append("")

            free_tokenize_result(result)
            return tokens_str

    def count_tokens(self, text: str, special_tokens: bool | set[str] = True) -> int:
        """
        Count the number of tokens in text.

        This returns the exact token count that would be used in generation,
        including BOS/EOS tokens by default. Use this to check if prompts fit
        within context windows.

        Args:
            text: Text to count tokens for.
            special_tokens: Control special token counting. Can be:
                - True (default): Include all special tokens (matches generation)
                - False: Count only content tokens
                - {"bos"}, {"eos"}, {"bos", "eos"}: Include specific tokens

        Returns
        -------
            Number of tokens.

        Example:
            >>> # Check if prompt fits in context window
            >>> tokens = tokenizer.count_tokens(prompt)
            >>> if tokens > 4096:
            ...     print("Prompt too long!")

            >>> # Count without special tokens
            >>> content_tokens = tokenizer.count_tokens(text, special_tokens=False)
        """
        return len(self.encode(text, special_tokens=special_tokens))

    # =========================================================================
    # Vocabulary Access
    # =========================================================================

    def get_vocab(self) -> dict[str, int]:
        """
        Get the complete vocabulary as a dictionary.

        Returns
        -------
            Dictionary mapping token strings to their IDs.

        Raises
        ------
            TokenizerError: If vocabulary retrieval fails.
        """
        vocab, error = call_tokenizer_get_vocab(self._handle)
        if error:
            raise TokenizerError(f"get_vocab failed: {error}")
        return vocab

    def id_to_token(self, token_id: int) -> str | None:
        """
        Get the string representation of a token ID.

        Args:
            token_id: The token ID to convert.

        Returns
        -------
            The token string, or None if the ID is invalid.

        Example:
            >>> tokenizer.id_to_token(9707)
            'Hello'
        """
        return call_tokenizer_id_to_token(self._handle, token_id)

    def token_to_id(self, token: str) -> int | None:
        """
        Get the ID of a token string.

        Args:
            token: The token string to convert.

        Returns
        -------
            The token ID, or None if the token is not in the vocabulary.

        Example:
            >>> tokenizer.token_to_id('Hello')
            9707
        """
        result = call_tokenizer_token_to_id(self._handle, token.encode("utf-8"))
        return result if result >= 0 else None

    def __contains__(self, token: object) -> bool:
        """Check if a string exists as a single token in the vocabulary."""
        if not isinstance(token, str):
            return False
        return self.token_to_id(token) is not None

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        """Convert a list of token IDs to their string representations.

        Args:
            ids: Token IDs to convert.
        """
        return [self.id_to_token(token_id) for token_id in ids]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int | None]:
        """Convert a list of token strings to their IDs.

        Args:
            tokens: Token strings to convert.
        """
        return [self.token_to_id(token) for token in tokens]

    # =========================================================================
    # Chat Templates
    # =========================================================================

    def apply_chat_template(
        self,
        messages: list,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str | TokenArray | BatchEncoding:
        """
        Format a conversation using the model's chat template.

        .. design-decision:: Tokenizer owns Chat Templates
            :status: DELIBERATE
            :rationale: Industry standard - HuggingFace bundles chat_template in tokenizer.json.

            In the LLM world, tokenizers own chat templates. HuggingFace's transformers
            library bundles ``chat_template`` inside ``tokenizer_config.json``, and users
            expect ``tokenizer.apply_chat_template()`` to work. Splitting templating
            into a separate class (e.g., PromptTemplate) would force users to load two
            files for one standard operation and create friction ("Why can't I just use
            the tokenizer like in HuggingFace?").

            For power users who want template inspection/customization, use:
            - ``PromptTemplate.from_chat_template(model)`` to load and inspect a model's template
            - ``PromptTemplate.from_file(path)`` to load a custom template
            - ``chat.chat_template`` to access the Chat's template

            But Tokenizer remains the "one-stop-shop" for text-to-ids conversion,
            which includes templating as an integrated step.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            add_generation_prompt: If True, add assistant turn marker.
            tokenize: If True, return TokenArray instead of string.

        Returns
        -------
            Formatted prompt string, or TokenArray if tokenize=True.

        Raises
        ------
            StateError: If no chat template is available (neither from model nor from_json()).

        Note:
            When tokenize=True, BOS tokens are handled automatically:

            - If the chat template includes BOS in output, it won't be doubled
            - EOS is not added (chat templates use turn markers instead)

            This prevents the "double-BOS" bug where models like Llama-3
            would receive two BOS tokens (one from the template, one from
            encode), which can degrade generation quality.
        """
        import json

        # Determine which template source to use
        if self._chat_template_str is not None:
            # from_json() tokenizer with explicit template string
            messages_json = json.dumps(messages).encode("utf-8")
            bos_token = (self._bos_token_str or "").encode("utf-8")
            eos_token = (self._eos_token_str or "").encode("utf-8")

            code, text = call_apply_chat_template_string(
                self._chat_template_str.encode("utf-8"),
                messages_json,
                add_generation_prompt,
                bos_token,
                eos_token,
            )
            if code != 0 or text is None:
                error = get_last_error() or "template rendering failed"
                raise TokenizerError(f"apply_chat_template failed: {error}")
        elif self._chat_template is not None:
            # Model-loaded tokenizer with ChatTemplate object
            text = self._chat_template.apply(messages, add_generation_prompt)
        else:
            raise StateError(
                "No chat template available. For from_json() tokenizers, "
                "provide chat_template parameter."
            )

        if not tokenize:
            return text

        # Auto-detect if template already added BOS to prevent double-BOS.
        # This is model-agnostic: we inspect the rendered output, not the model name.
        # - Llama-3: template adds "<|begin_of_text|>..." → add_bos=False
        # - Mistral: template outputs "User:..." → add_bos=True
        # - Qwen-3: bos_token is None → add_bos=False (no-op anyway)
        add_bos = not (self.bos_token and text.lstrip().startswith(self.bos_token))

        return self.encode(text, special_tokens={"bos"} if add_bos else False)

    # =========================================================================
    # HuggingFace-style __call__
    # =========================================================================

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | list[str] | None = None,
        special_tokens: bool | set[str] = True,
        truncation: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> "BatchEncoding":
        """
        Callable interface for tokenization with zero-copy tensor access.

        Returns a BatchEncoding object that provides dict-like access to
        input_ids and attention_mask via the DLPack protocol. This enables
        zero-copy transfer to PyTorch, JAX, or NumPy.

        Padding is applied automatically when exporting to tensors. Control
        the padding side via ``tokenizer.padding_side`` or ``batch.padding_side``.

        Args:
            text: Text(s) to tokenize. Single string is wrapped as batch of 1.
            text_pair: Not supported.
            special_tokens: Control special token insertion. Can be:
                - True: Add all special tokens (BOS, EOS) - default
                - False: No special tokens
                - {"bos"}, {"eos"}, {"bos", "eos"}: Granular control
            truncation: False or True.
            max_length: Maximum sequence length.
            return_tensors: Ignored (use DLPack instead).

        Returns
        -------
            BatchEncoding with dict-like interface for zero-copy tensor access:
            - batch["input_ids"] → DLPack-compatible accessor
            - batch["attention_mask"] → DLPack-compatible accessor

        Raises
        ------
            NotImplementedError: If text_pair is provided.
            ValidationError: If truncation=True but no max_length specified.

        Example:
            >>> batch = tokenizer(["Hello", "World!"])
            >>> input_ids = torch.from_dlpack(batch["input_ids"])
            >>> attention_mask = torch.from_dlpack(batch["attention_mask"])
            >>> model(input_ids, attention_mask=attention_mask)

        Note:
            For Python list output (debugging only), use batch.to_list().
        """
        if text_pair is not None:
            raise NotImplementedError("text_pair is not yet supported. Concatenate texts manually.")

        # Ignore return_tensors - we always return BatchEncoding
        if return_tensors is not None:
            logger.debug(
                "return_tensors ignored, use torch.from_dlpack(batch['input_ids']) instead",
                extra={"return_tensors": return_tensors},
            )

        for key in kwargs:
            logger.warning("Unknown argument ignored in tokenizer()", extra={"argument": key})

        # Resolve max_length: explicit > model_max_length > error
        effective_max_length = max_length
        if truncation and truncation is not False and max_length is None:
            # Use model_max_length as default if available
            model_limit = self.model_max_length
            if model_limit > 0:
                effective_max_length = model_limit
            else:
                from ..exceptions import ValidationError

                raise ValidationError(
                    "truncation=True requires max_length. "
                    "This model does not specify model_max_length in tokenizer_config.json. "
                    "Example: tokenizer(text, truncation=True, max_length=512)"
                )

        # Normalize single string to list for consistent BatchEncoding return
        texts = [text] if isinstance(text, str) else text

        # Encode to BatchEncoding
        # texts is always a list at this point, so encode returns BatchEncoding
        result = self.encode(
            texts,
            special_tokens=special_tokens,
            truncation=bool(truncation),
            max_length=effective_max_length or 0,
        )

        # Configure padding on the BatchEncoding
        # padding_side and pad_token_id are already inherited from tokenizer
        # User can override: batch.padding_side = "right"

        # At this point, result is guaranteed to be BatchEncoding since texts is a list
        assert isinstance(result, BatchEncoding)
        return result
