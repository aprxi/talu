"""
Batch tokenization container.

Wraps batch tokenization results, providing lazy access to individual
sequences and support for padded tensor conversion.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..exceptions import InteropError, StateError, ValidationError
from ._bindings import (
    PaddedTensorOptions,
    PaddedTensorResult,
    call_batch_encode_result_free,
    call_batch_mask_to_dlpack,
    call_batch_to_dlpack,
    call_batch_to_padded_tensor,
    call_padded_tensor_result_free,
    create_dlpack_capsule,
)

if TYPE_CHECKING:
    from .token_array import TokenArrayView


class _DLPackAccessor:
    """
    Internal DLPack protocol wrapper for a single tensor from BatchEncoding.

    BatchEncoding holds multiple tensors (input_ids, attention_mask). Since
    ``__dlpack__`` can only export one tensor, each property returns a
    ``_DLPackAccessor`` bound to the appropriate export function.

    Users interact with this transparently via ``torch.from_dlpack(batch.input_ids)``
    or ``np.from_dlpack(batch["attention_mask"])`` — they never need to name this type.
    """

    __slots__ = ("_parent", "_export_fn", "_name")

    def __init__(self, parent: BatchEncoding, export_fn: Any, name: str):
        self._parent = parent
        self._export_fn = export_fn
        self._name = name

    def __dlpack__(self, *, stream=None):
        """DLPack protocol — export as 2D tensor [num_sequences, padded_length]."""
        return self._export_fn()

    def __dlpack_device__(self):
        """Return device tuple for DLPack protocol (CPU)."""
        return (1, 0)  # kDLCPU = 1, device_id = 0

    def __repr__(self) -> str:
        return f"{self._name}(num_sequences={self._parent._num_sequences})"


class BatchEncoding:
    """
    Lazy container for batch tokenization results.

    Holds all token IDs in a single contiguous memory block, with lazy
    creation of TokenArray views when accessing individual sequences.

    Key Features
    ------------

    **List-like interface** - Works like a list of TokenArrays:

        >>> batch = tokenizer.encode(["Hello", "World"])
        >>> len(batch)
        2
        >>> batch[0]  # Returns lazy TokenArray view
        TokenArray([...], len=...)

    **Lazy evaluation** - TokenArray views are created on-demand:

        >>> # Only creates view when accessed
        >>> first = batch[0]

    **Dictionary-like access** - Compatible with HuggingFace patterns:

        >>> input_ids = torch.from_dlpack(batch["input_ids"])
        >>> attention_mask = torch.from_dlpack(batch["attention_mask"])

    ML Framework Integration
    ------------------------

    To convert to PyTorch or JAX, use the **specific tensor accessors**.
    Do NOT try to convert the batch object itself - it contains multiple
    tensors and will raise an InteropError.

    **Correct usage**:

        >>> import torch
        >>> input_ids = torch.from_dlpack(batch.input_ids)
        >>> attention_mask = torch.from_dlpack(batch.attention_mask)

    **Incorrect usage** (raises InteropError):

        >>> tensor = torch.from_dlpack(batch)  # Error: Ambiguous - which tensor?

    Memory Management
    -----------------

    BatchEncoding owns the underlying memory and frees it when garbage
    collected. The TokenArray views returned by indexing are lightweight
    wrappers that point into this shared memory - they become invalid
    if the BatchEncoding is deleted.
    """

    __slots__ = (
        "_ids_ptr",
        "_offsets_ptr",
        "_total_tokens",
        "_num_sequences",
        "_padding_side",
        "_pad_token_id",
    )

    def __init__(
        self,
        ids_ptr: Any = None,
        offsets_ptr: Any = None,
        total_tokens: int = 0,
        num_sequences: int = 0,
        *,
        padding_side: str = "left",
        pad_token_id: int | None = None,
    ):
        self._ids_ptr = ids_ptr
        self._offsets_ptr = offsets_ptr
        self._total_tokens = total_tokens
        self._num_sequences = num_sequences
        self._padding_side = padding_side
        self._pad_token_id = pad_token_id

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, key: int | str) -> TokenArrayView | _DLPackAccessor:
        """
        Get item by index (list-like) or key (dict-like).

        Integer indexing returns individual sequences as TokenArrayView:
            >>> batch[0]  # First sequence
            >>> batch[-1]  # Last sequence

        String keys provide DLPack-compatible accessors:
            >>> torch.from_dlpack(batch["input_ids"])
            >>> torch.from_dlpack(batch["attention_mask"])

        Args:
            key: Integer index for sequence access, or string key for tensor access.

        Returns
        -------
            - For int: TokenArrayView of that sequence
            - For "input_ids": DLPack-compatible accessor
            - For "attention_mask": DLPack-compatible accessor

        Raises
        ------
            IndexError: If integer index is out of range.
            KeyError: If string key is not "input_ids" or "attention_mask".
        """
        if isinstance(key, str):
            if key == "input_ids":
                return _DLPackAccessor(self, self._export_input_ids_dlpack, "InputIdsAccessor")
            elif key == "attention_mask":
                return _DLPackAccessor(self, self._export_mask_dlpack, "AttentionMaskAccessor")
            else:
                raise KeyError(f"Unknown key {key!r}. Valid keys: 'input_ids', 'attention_mask'")

        # Integer indexing for sequence access
        idx = key
        if idx < 0:
            idx = self._num_sequences + idx
        if idx < 0 or idx >= self._num_sequences:
            raise IndexError(f"Sequence index {idx} out of range [0, {self._num_sequences})")

        if self._offsets_ptr is None or self._ids_ptr is None:
            raise StateError(
                "BatchEncoding not initialized properly. "
                "Ensure it was created via Tokenizer.encode().",
                code="STATE_UNINITIALIZED",
            )
        start = self._offsets_ptr[idx]
        end = self._offsets_ptr[idx + 1]
        from .token_array import TokenArrayView

        return TokenArrayView(self._ids_ptr, start, end - start, self)

    def __contains__(self, key: str) -> bool:
        """Check if key is a valid tensor key."""
        return key in ("input_ids", "attention_mask")

    def keys(self) -> list[str]:
        """Return available tensor keys (dict-like interface)."""
        return ["input_ids", "attention_mask"]

    def __iter__(self) -> Iterator[TokenArrayView]:
        if self._offsets_ptr is None or self._ids_ptr is None:
            return  # Empty batch
        from .token_array import TokenArrayView

        for i in range(self._num_sequences):
            start = self._offsets_ptr[i]
            end = self._offsets_ptr[i + 1]
            yield TokenArrayView(self._ids_ptr, start, end - start, self)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all sequences."""
        return self._total_tokens

    def lengths(self) -> list[int]:
        """Get the length of each sequence in the batch."""
        if self._offsets_ptr is None:
            return []
        return [self._offsets_ptr[i + 1] - self._offsets_ptr[i] for i in range(self._num_sequences)]

    def max_length(self) -> int:
        """Get the maximum sequence length in the batch."""
        if self._num_sequences == 0:
            return 0
        return max(self.lengths())

    @property
    def padding_side(self) -> str:
        """
        Side for padding: "left" (default for generation) or "right".

        This property is inherited from the tokenizer when the batch is created.
        Set this property to override padding behavior before calling
        ``to_list()`` or using ``torch.from_dlpack()``.

        Example:
            >>> batch = tokenizer.encode(["Hello", "World"])
            >>> batch.padding_side
            'left'
            >>> batch.padding_side = "right"  # Override for this batch
            >>> tensor = torch.from_dlpack(batch)  # Uses right padding
        """
        return self._padding_side

    @padding_side.setter
    def padding_side(self, value: str) -> None:
        if value not in ("left", "right"):
            raise ValidationError(
                f"padding_side must be 'left' or 'right', got {value!r}.",
                code="INVALID_ARGUMENT",
                details={"param": "padding_side", "value": value, "allowed": ["left", "right"]},
            )
        self._padding_side = value

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID used for this batch."""
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value: int | None) -> None:
        self._pad_token_id = value

    @property
    def input_ids(self) -> _DLPackAccessor:
        """
        DLPack interface for the input_ids tensor.

        Returns an accessor object that implements the DLPack protocol.
        Use with ``torch.from_dlpack()`` or ``np.from_dlpack()`` to get
        a 2D tensor of shape (num_sequences, padded_length).

        NOTE: Each export allocates a new padded buffer. The internal CSR
        storage is materialized to dense format for ML framework consumption.

        Example:
            >>> batch = tokenizer.encode(["Hello", "World!"])
            >>> input_ids = torch.from_dlpack(batch.input_ids)
            >>> attention_mask = torch.from_dlpack(batch.attention_mask)
            >>> model(input_ids, attention_mask=attention_mask)
        """
        return _DLPackAccessor(self, self._export_input_ids_dlpack, "InputIdsAccessor")

    @property
    def attention_mask(self) -> _DLPackAccessor:
        """
        DLPack interface for the attention mask tensor.

        Returns an accessor object that implements the DLPack protocol.
        Use with ``torch.from_dlpack()`` or ``np.from_dlpack()`` to get
        a 2D tensor of shape (num_sequences, padded_length) with values:
        - 1 for real tokens
        - 0 for padding tokens

        NOTE: Each export allocates a new mask buffer computed from sequence
        lengths and padding configuration.

        Example:
            >>> batch = tokenizer.encode(["Hello", "World!"])
            >>> input_ids = torch.from_dlpack(batch.input_ids)
            >>> attention_mask = torch.from_dlpack(batch.attention_mask)
            >>> model(input_ids, attention_mask=attention_mask)
        """
        return _DLPackAccessor(self, self._export_mask_dlpack, "AttentionMaskAccessor")

    def _export_input_ids_dlpack(self):
        """Export the input_ids via DLPack C-API."""
        if self._num_sequences == 0:
            raise InteropError(
                "Cannot export empty BatchEncoding via DLPack. "
                "Encode at least one sequence before exporting.",
                code="EMPTY_BATCH",
            )

        # Resolve defaults
        final_side = self._padding_side
        final_pad_id = self._pad_token_id or 0

        dlpack_ptr = call_batch_to_dlpack(
            self._ids_ptr,
            self._offsets_ptr,
            self._num_sequences,
            final_pad_id,
            0,  # max_length (0 = use longest sequence)
            final_side == "left",
        )

        if not dlpack_ptr:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise InteropError(
                f"Failed to create input_ids DLPack capsule: {error}",
                code="DLPACK_EXPORT_FAILED",
            )

        return create_dlpack_capsule(dlpack_ptr)

    def _export_mask_dlpack(self):
        """Export the attention mask via DLPack C-API."""
        if self._num_sequences == 0:
            raise InteropError(
                "Cannot export empty BatchEncoding via DLPack. "
                "Encode at least one sequence before exporting.",
                code="EMPTY_BATCH",
            )

        dlpack_ptr = call_batch_mask_to_dlpack(
            self._ids_ptr,
            self._offsets_ptr,
            self._num_sequences,
            0,  # max_length (0 = use longest sequence)
            self._padding_side == "left",
        )

        if not dlpack_ptr:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise InteropError(
                f"Failed to create attention mask DLPack capsule: {error}",
                code="DLPACK_EXPORT_FAILED",
            )

        return create_dlpack_capsule(dlpack_ptr)

    def _pad_to_zig_result(
        self,
        pad_id: int,
        padding_side: str,
        max_length: int | None,
        truncation: bool,
        return_attention_mask: bool,
    ) -> PaddedTensorResult:
        """Call Zig to get padded tensor result. Caller must free the result."""
        options = PaddedTensorOptions(
            pad_id=pad_id,
            padding_side=1 if padding_side == "left" else 0,
            max_length=max_length or 0,
            truncate=truncation,
            return_attention_mask=return_attention_mask,
        )

        result = call_batch_to_padded_tensor(
            self._ids_ptr, self._offsets_ptr, self._num_sequences, options
        )

        if result.error_msg:
            from ..exceptions import TaluError

            raise TaluError(
                result.error_msg.decode("utf-8"),
                code="PADDING_FAILED",
            )

        return result

    def to_list(
        self,
        padding: bool = True,
        pad_id: int | None = None,
        padding_side: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
        return_attention_mask: bool = True,
    ) -> dict[str, list[list[int]]]:
        """
        Convert batch to padded Python lists.

        This is primarily useful for debugging or when you need plain Python
        data structures. For ML workloads, use ``torch.from_dlpack(batch.input_ids)``
        and ``torch.from_dlpack(batch.attention_mask)`` directly.

        Args:
            padding: If True (default), pad shorter sequences.
            pad_id: Token ID to use for padding. Defaults to stored pad_token_id
                from tokenizer, or 0 if not set.
            padding_side: Where to add padding tokens. Defaults to value passed
                to encode(), which defaults to tokenizer.padding_side.
                - "right": Pad at end (encoder models)
                - "left": Pad at start (decoder/generation models)
            max_length: Maximum length to pad to. If None, uses longest sequence.
            truncation: If True, truncate sequences longer than max_length.
            return_attention_mask: If True (default), include attention_mask.

        Returns
        -------
            Dictionary with:
            - "input_ids": 2D list of padded token IDs
            - "attention_mask": 2D list of masks (1=real, 0=padding)

        Raises
        ------
            ValidationError: If padding_side is not 'left' or 'right'.
        """
        # Resolve defaults: use arg if provided, else stored config
        final_side = padding_side if padding_side is not None else self._padding_side
        final_pad_id = pad_id if pad_id is not None else (self._pad_token_id or 0)

        if final_side not in ("left", "right"):
            from ..exceptions import ValidationError

            raise ValidationError(f"padding_side must be 'left' or 'right', got {final_side!r}")

        if self._num_sequences == 0:
            output: dict[str, list[list[int]]] = {"input_ids": []}
            if return_attention_mask:
                output["attention_mask"] = []
            return output

        if not padding:
            seq_lengths = self.lengths()
            if len(set(seq_lengths)) > 1:
                from ..exceptions import ValidationError

                raise ValidationError(
                    f"Sequences have different lengths {set(seq_lengths)}. "
                    "Use padding=True to pad to uniform length."
                )

        result = self._pad_to_zig_result(
            final_pad_id, final_side, max_length, truncation, return_attention_mask
        )

        # Convert flat arrays to 2D lists
        num_seq = result.num_sequences
        padded_len = result.padded_length

        input_ids = [
            [result.input_ids[i * padded_len + j] for j in range(padded_len)]
            for i in range(num_seq)
        ]

        output = {"input_ids": input_ids}

        if return_attention_mask and result.attention_mask:
            attention_mask = [
                [result.attention_mask[i * padded_len + j] for j in range(padded_len)]
                for i in range(num_seq)
            ]
            output["attention_mask"] = attention_mask

        # Free the Zig-allocated memory
        call_padded_tensor_result_free(result.input_ids, result.attention_mask, num_seq, padded_len)

        return output

    def close(self) -> None:
        """
        Release the native batch encoding memory.

        After calling close(), the BatchEncoding cannot be used for data
        access, DLPack export, or iteration. Safe to call multiple times
        (idempotent).
        """
        if hasattr(self, "_ids_ptr") and self._ids_ptr:
            call_batch_encode_result_free(
                self._ids_ptr,
                self._offsets_ptr,
                self._total_tokens,
                self._num_sequences,
            )
            self._ids_ptr = None
            self._offsets_ptr = None

    def __enter__(self) -> BatchEncoding:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self):
        """Free the batch encoding memory when garbage collected.

        Robust to interpreter shutdown - silently ignores errors when
        module globals may be unavailable.
        """
        try:
            self.close()
        except Exception:
            # Ignore errors during interpreter shutdown when globals may be cleared
            pass

    def __repr__(self) -> str:
        return (
            f"BatchEncoding(num_sequences={self._num_sequences}, total_tokens={self._total_tokens})"
        )

    # =========================================================================
    # DLPack Protocol (zero-copy PyTorch/JAX interop)
    # =========================================================================

    def __dlpack__(self, *, stream=None):
        """
        Raise InteropError because BatchEncoding contains multiple tensors.

        BatchEncoding holds both input_ids and attention_mask. Calling
        `torch.from_dlpack(batch)` is ambiguous. Use the explicit accessors:

            input_ids = torch.from_dlpack(batch.input_ids)
            attention_mask = torch.from_dlpack(batch.attention_mask)

        Raises
        ------
            InteropError: Always, with guidance on using explicit accessors.
        """
        raise InteropError(
            "BatchEncoding contains multiple tensors. "
            "Export specific fields:\n"
            "  input_ids = torch.from_dlpack(batch.input_ids)\n"
            "  attention_mask = torch.from_dlpack(batch.attention_mask)",
            code="AMBIGUOUS_EXPORT",
        )

    def __dlpack_device__(self):
        """
        Return device tuple for DLPack protocol.

        Returns (device_type, device_id) where device_type 1 = CPU.

        Returns
        -------
            Tuple (1, 0) indicating CPU device 0.
        """
        return (1, 0)  # kDLCPU = 1, device_id = 0
