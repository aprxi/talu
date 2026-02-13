"""
Token containers - TokenArray, TokenArrayView, and TokenOffset.

Provides efficient token storage with zero-copy NumPy/DLPack support.

Memory Safety Contract:
- DLPack export shares storage with the TokenArray
- TokenArray remains valid after DLPack export (multiple exports are safe)
- If a consumer mutates the exported tensor, the underlying storage is mutated
- TokenArray APIs treat tokens as immutable; mutation from consumers is allowed
  but not recommended
- If isolation is needed, clone/copy in the consumer framework

Slicing:
- `tokens[a:b]` returns a *copy* (new buffer), not a view
- `__dlpack__` is zero-copy for a TokenArray instance
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

from ..exceptions import InteropError, StateError
from ._bindings import (
    call_buffer_create_from_copy,
    call_buffer_release,
    call_buffer_to_dlpack,
    call_list_concat_with_tokens,
    call_tokens_concat,
    call_tokens_concat_with_list,
    create_dlpack_capsule,
    get_ptr_address,
    get_ptr_address_offset,
)

if TYPE_CHECKING:
    from .batch import BatchEncoding


class TokenOffset:
    """
    Byte offset pair mapping a token to its position in source text.

    Represents a (start, end) byte range in the original UTF-8 encoded text.
    Offsets are byte indices, not character indices, enabling O(1) slicing
    regardless of Unicode content.

    Methods
    -------
    slice(text, errors="strict")
        Extracts the text span for this token. This is the recommended way
        to use offsets as it handles byte-to-string conversion automatically.

    Attributes
    ----------
    start : int
        Start byte offset in source text (inclusive).
    end : int
        End byte offset in source text (exclusive).

    Notes
    -----
    - Offsets are UTF-8 byte indices, not character indices. Do NOT use them
      directly on Python strings (e.g., ``text[start:end]``). Use the
      ``slice()`` method instead.
    - For byte-level BPE tokenizers (GPT-2, Qwen), individual tokens may
      split multi-byte UTF-8 sequences. Use ``errors="replace"`` in ``slice()``
      to handle these gracefully.
    - Special tokens (BOS, EOS) that don't correspond to source text
      are assigned (0, 0).

    Examples
    --------
    Extract text using slice() method (recommended):
        >>> text = "Hello 🎉 world"
        >>> tokens = tokenizer.encode(text)
        >>> tokens.offsets[-1].slice(text)
        ' world'

    Handle byte-level BPE tokens that split UTF-8 sequences:

        >>> tokens.offsets[1].slice(text, errors="replace")
        ' \ufffd\ufffd'  # Replacement chars for partial emoji bytes

    Tuple unpacking for raw byte indices:

        >>> offset = TokenOffset(0, 5)
        >>> start, end = offset
        >>> print(start, end)
        0 5

    Comparing with tuples:

        >>> offset = TokenOffset(0, 5)
        >>> offset == (0, 5)
        True
    """

    __slots__ = ("start", "end")

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def slice(self, text: str | bytes, errors: str = "strict") -> str:
        r"""
        Extract the text span corresponding to this offset.

        This is the recommended way to use offsets. It handles the conversion
        between byte offsets and Python string indexing automatically.

        Args:
            text: The original source text (str or bytes).
            errors: How to handle decode errors. Default "strict" raises
                UnicodeDecodeError. Use "replace" to substitute invalid bytes
                with the replacement character. This can happen with byte-level
                BPE tokenizers that split multi-byte UTF-8 sequences.

        Returns
        -------
            The substring corresponding to this token.

        Raises
        ------
            UnicodeDecodeError: If the byte span is not valid UTF-8 and
                errors="strict" (default). This can happen with byte-level
                BPE tokenizers (GPT-2, Qwen) that split multi-byte characters
                across tokens.

        Example:
            >>> tokens = tokenizer.encode("Hello 🎉 world")
            >>> tokens.offsets[-1].slice("Hello 🎉 world")
            ' world'

            For byte-level BPE tokens that may split UTF-8 sequences:

            >>> tokens.offsets[1].slice(text, errors="replace")
            ' \ufffd\ufffd'  # Replacement characters for partial bytes
        """
        if isinstance(text, str):
            text = text.encode("utf-8")
        return text[self.start : self.end].decode("utf-8", errors=errors)

    def __repr__(self) -> str:
        return f"({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TokenOffset):
            return self.start == other.start and self.end == other.end
        if isinstance(other, tuple) and len(other) == 2:
            return self.start == other[0] and self.end == other[1]
        return NotImplemented

    def __iter__(self):
        """Allow tuple unpacking: start, end = offset."""
        yield self.start
        yield self.end


class TokenArray(Sequence[int]):
    """
    Sequence of token IDs with zero-copy NumPy and DLPack support.

    Returned by ``Tokenizer.encode()`` and provides efficient access to
    token data. The underlying memory is managed by a refcounted buffer
    in the Zig runtime. Implements ``collections.abc.Sequence[int]``.

    Key Features
    ------------

    **Zero-copy NumPy conversion** - No data copying when converting to NumPy:

        >>> import numpy as np
        >>> tokens = tokenizer.encode("Hello world")
        >>> arr = np.asarray(tokens)  # Zero-copy view
        >>> print(arr.dtype)
        uint32

    **Safe DLPack export** - Zero-copy export to PyTorch/JAX without invalidation:

        >>> tensor = torch.from_dlpack(tokens)  # Zero-copy!
        >>> len(tokens)  # Still valid! TokenArray not invalidated
        2
        >>> tensor2 = torch.from_dlpack(tokens)  # Multiple exports safe

    **Standard sequence operations** - Works like a Python list:

        >>> len(tokens)
        2
        >>> tokens[0]
        9707
        >>> tokens[-1]  # Negative indexing
        1879

    **Convert to list** - When you need a regular Python list:

        >>> tokens.tolist()
        [9707, 1879]

    **Token offset mapping** - Map tokens back to source text positions:

        >>> text = "Hello world"
        >>> tokens = tokenizer.encode(text)
        >>> tokens.offsets  # Lazy, computed on first access
        [(0, 5), (5, 11)]
        >>> tokens.offsets[0].slice(text)  # Recommended
        'Hello'

    Memory Management
    -----------------

    TokenArray uses a refcounted buffer. The buffer is freed when all
    references are released (TokenArray deleted + all DLPack exports consumed).

    NumPy arrays from `np.asarray()` are views - they become invalid if the
    TokenArray is deleted AND no DLPack exports exist:

        >>> tokens = tokenizer.encode("Hello")
        >>> arr = np.asarray(tokens)
        >>> del tokens  # Only safe if no other references exist!
        >>> arr[0]      # May be undefined if buffer was freed

    To keep the data safely, either:
    1. Keep the TokenArray alive while using the NumPy view
    2. Copy it: `arr = np.array(tokens)`
    3. Use DLPack: `tensor = torch.from_dlpack(tokens)` (keeps buffer alive)

    See Also
    --------
    Tokenizer.encode : Creates TokenArrays from text.
    Tokenizer.decode : Converts TokenArrays back to text.
    TokenOffset : The offset type returned by the offsets property.
    """

    def __init__(
        self,
        tokens_ptr: Any,
        num_tokens: int,
        *,
        _offsets: list[TokenOffset] | None = None,
        _buffer_handle: Any = None,
    ) -> None:
        """
        Initialize from a Zig-allocated token pointer.

        This is an internal constructor. Users should get TokenArrays from
        `Tokenizer.encode()`, not create them directly.

        Args:
            tokens_ptr: Pointer to uint32 token IDs (from buffer or legacy).
            num_tokens: Number of tokens in the array.
            _offsets: Pre-calculated token offsets (from encode).
            _buffer_handle: SharedBuffer handle (if using refcounted buffer).
        """
        self._ptr = tokens_ptr
        self._num_tokens = num_tokens
        self._buffer_handle = _buffer_handle  # SharedBuffer for refcounting
        self._offsets = _offsets
        # Future-proofing for views: offset into buffer (always 0 for now)
        self._offset_elems = 0

    def __len__(self) -> int:
        """
        Return the number of tokens.

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> len(tokens)
            2
        """
        return self._num_tokens

    @overload
    def __getitem__(self, idx: int) -> int: ...

    @overload
    def __getitem__(self, idx: slice) -> "TokenArray": ...

    def __getitem__(self, idx: int | slice) -> "int | TokenArray":
        """
        Get a token ID by index or a slice of tokens.

        Supports negative indexing (e.g., -1 for last token) and slicing.

        Args:
            idx: Token index (0-based, or negative from end), or slice.

        Returns
        -------
            int: The token ID at that position (for single index).
            TokenArray: A new TokenArray containing the slice (for slice).

        Raises
        ------
            IndexError: If index is out of range.

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> tokens[0]      # First token
            9707
            >>> tokens[-1]     # Last token
            1879
            >>> tokens[:5]     # First 5 tokens (returns TokenArray)
            TokenArray([...], len=5)
            >>> tokens[1:3]    # Slice
            TokenArray([...], len=2)
        """
        if self._ptr is None:
            raise StateError(
                "TokenArray not initialized properly. "
                "Ensure it was created via Tokenizer.encode().",
                code="STATE_UNINITIALIZED",
            )

        # Handle slice
        if isinstance(idx, slice):
            # Convert slice to range indices
            start, stop, step = idx.indices(self._num_tokens)
            indices = range(start, stop, step)
            token_list = [self._ptr[i] for i in indices]
            return self._from_list(token_list)

        # Handle single index
        if idx < 0:
            idx = self._num_tokens + idx
        if idx < 0 or idx >= self._num_tokens:
            raise IndexError(f"Token index {idx} out of range [0, {self._num_tokens})")
        return self._ptr[idx]

    @classmethod
    def _from_list(cls, token_list: list[int]) -> "TokenArray":
        """Create a TokenArray from a Python list (internal helper).

        Uses SharedBuffer for refcounted memory management.
        """
        if not token_list:
            # Empty array - create with null ptr and 0 length
            return cls(None, 0)

        buffer_handle, data_ptr = call_buffer_create_from_copy(token_list)
        if not buffer_handle:
            raise MemoryError("Failed to allocate SharedBuffer for TokenArray")

        return cls(data_ptr, len(token_list), _buffer_handle=buffer_handle)

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over token IDs.

        Example:
            >>> tokens = tokenizer.encode("Hi")
            >>> for token_id in tokens:
            ...     print(token_id)
        """
        if self._ptr is None:
            return
        for i in range(self._num_tokens):
            yield self._ptr[i]

    def tolist(self) -> list[int]:
        """
        Convert to a Python list.

        This copies the data into a new Python list. Use this when you
        need a regular list, or when you need to keep the data after
        the TokenArray is deleted.

        Returns
        -------
            A new list containing the token IDs.

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> token_list = tokens.tolist()
            >>> print(token_list)
            [9707, 1879]
            >>> type(token_list)
            <class 'list'>
        """
        if self._ptr is None:
            return []
        return [self._ptr[i] for i in range(self._num_tokens)]

    def __contains__(self, value: object) -> bool:
        """
        Check if a token ID is in the array.

        Args:
            value: Token ID to search for (must be int).

        Returns
        -------
            True if value is in the array, False otherwise.

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> 9707 in tokens  # Check if token ID is present
            True
            >>> 99999 in tokens
            False
        """
        if not isinstance(value, int):
            return False
        if self._ptr is None:
            return False
        for i in range(self._num_tokens):
            if self._ptr[i] == value:
                return True
        return False

    def index(self, value: int, start: int = 0, stop: int | None = None) -> int:
        """
        Return index of first occurrence of value.

        Args:
            value: Token ID to search for.
            start: Start index for search (default 0).
            stop: Stop index for search (default end of array).

        Returns
        -------
            Index of first occurrence of value.

        Raises
        ------
            ValueError: If value is not in the array.

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> tokens.index(9707)  # Find first occurrence
            0
        """
        if self._ptr is None:
            raise ValueError(f"{value} is not in TokenArray")
        if stop is None:
            stop = self._num_tokens
        for i in range(start, stop):
            if self._ptr[i] == value:
                return i
        raise ValueError(f"{value} is not in TokenArray")

    def count(self, value: int) -> int:
        """
        Return number of occurrences of value.

        Args:
            value: Token ID to count.

        Returns
        -------
            Number of times value appears in the array.

        Example:
            >>> tokens = tokenizer.encode("hello hello hello")
            >>> tokens.count(9707)  # Count occurrences
            3
        """
        if self._ptr is None:
            return 0
        count = 0
        for i in range(self._num_tokens):
            if self._ptr[i] == value:
                count += 1
        return count

    def __mul__(self, n: int) -> "TokenArray":
        """
        Repeat the token array n times.

        Args:
            n: Number of times to repeat.

        Returns
        -------
            New TokenArray with contents repeated n times.

        Example:
            >>> tokens = tokenizer.encode("Hi")
            >>> repeated = tokens * 3  # Repeat 3 times
            >>> len(repeated) == len(tokens) * 3
            True
        """
        if not isinstance(n, int):
            return NotImplemented
        if n <= 0 or self._ptr is None:
            return self._from_list([])
        if n == 1:
            return self._from_list(self.tolist())
        # Repeat the list n times
        token_list = self.tolist() * n
        return self._from_list(token_list)

    def __rmul__(self, n: int) -> "TokenArray":
        """
        Repeat the token array n times (right-hand multiplication).

        Args:
            n: Number of times to repeat.

        Returns
        -------
            New TokenArray with contents repeated n times.

        Example:
            >>> tokens = tokenizer.encode("Hi")
            >>> repeated = 3 * tokens  # Repeat 3 times
            >>> len(repeated) == len(tokens) * 3
            True
        """
        return self.__mul__(n)

    @property
    def offsets(self) -> list[TokenOffset]:
        """
        Byte offsets mapping tokens back to source text.

        Each offset is a (start, end) pair of UTF-8 byte indices into the
        original text. Computed during encoding and returned directly.

        Special tokens (BOS, EOS, PAD) that don't correspond to source text
        are assigned (0, 0).

        Returns
        -------
            List of TokenOffset objects, one per token.

        Raises
        ------
            StateError: If the TokenArray was not created via Tokenizer.encode()
                (e.g., created from a list or slice).

        Example:
            >>> text = "Hello 🎉 world"
            >>> tokens = tokenizer.encode(text)
            >>> tokens.offsets
            [(0, 5), (5, 10), ...]

            Use the slice() helper to extract text (handles Unicode correctly):

            >>> tokens.offsets[0].slice(text)
            'Hello'
            >>> tokens.offsets[-1].slice(text)
            ' world'
        """
        if self._offsets is not None:
            return self._offsets

        raise StateError(
            "Offset mapping requires source text. "
            "Ensure the TokenArray was created via Tokenizer.encode().",
            code="STATE_MISSING_SOURCE",
        )

    @property
    def __array_interface__(self) -> dict:
        """
        NumPy array interface for zero-copy access.

        This allows NumPy to create an array view directly over the
        underlying memory without copying. Use `np.asarray(tokens)`.

        Returns
        -------
            Dictionary conforming to NumPy's array interface protocol.

        Example:
            >>> import numpy as np
            >>> tokens = tokenizer.encode("Hello world")
            >>> arr = np.asarray(tokens)  # Zero-copy!
            >>> print(arr.shape, arr.dtype)
            (2,) uint32
        """
        if self._ptr is None:
            raise StateError(
                "TokenArray not initialized properly. "
                "Ensure it was created via Tokenizer.encode().",
                code="STATE_UNINITIALIZED",
            )
        return {
            "version": 3,
            "shape": (self._num_tokens,),
            "typestr": "<u4",  # uint32 little-endian
            "data": (get_ptr_address(self._ptr), False),
            "strides": (4,),  # 4 bytes per uint32
        }

    def close(self) -> None:
        """
        Release the underlying native buffer.

        After calling close(), the TokenArray cannot be used for data access,
        DLPack export, or NumPy conversion. Safe to call multiple times
        (idempotent).
        """
        if hasattr(self, "_buffer_handle") and self._buffer_handle:
            call_buffer_release(self._buffer_handle)
            self._buffer_handle = None
            self._ptr = None

    def __enter__(self) -> "TokenArray":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self):
        """Release the buffer reference when garbage collected.

        Robust to interpreter shutdown - silently ignores errors when
        module globals may be unavailable.
        """
        try:
            self.close()
        except Exception:
            # Ignore errors during interpreter shutdown when globals may be cleared
            pass

    def __repr__(self) -> str:
        """Return string representation showing token IDs."""
        if self._ptr is None:
            return "TokenArray([], len=0)"
        if self._num_tokens <= 10:
            tokens_str = str(self.tolist())
        else:
            first = [self._ptr[i] for i in range(5)]
            last = [self._ptr[i] for i in range(self._num_tokens - 3, self._num_tokens)]
            tokens_str = f"[{', '.join(map(str, first))}, ..., {', '.join(map(str, last))}]"
        return f"TokenArray({tokens_str}, len={self._num_tokens})"

    def __eq__(self, other) -> bool:
        """Compare two TokenArrays for equality."""
        if isinstance(other, TokenArray):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(len(self)))
        if isinstance(other, list):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(len(self)))
        return NotImplemented

    def __add__(self, other: "TokenArray | list[int]") -> "TokenArray":
        """
        Concatenate with another TokenArray or list.

        Uses Zig for efficient memory allocation and copying.

        Args:
            other: TokenArray or list of token IDs to append.

        Returns
        -------
            New TokenArray containing tokens from both arrays.

        Example:
            >>> tokens1 = tokenizer.encode("Hello")
            >>> tokens2 = tokenizer.encode("World")
            >>> combined = tokens1 + tokens2
            >>> combined = tokens1 + [1, 2, 3]  # Also works with lists
        """
        if isinstance(other, TokenArray):
            # TokenArray + TokenArray: use Zig concat directly
            result_ptr = call_tokens_concat(
                self._ptr,
                self._num_tokens,
                other._ptr,
                other._num_tokens,
            )
            if not result_ptr:
                raise MemoryError("Failed to allocate memory for token concatenation")
            return TokenArray(result_ptr, self._num_tokens + other._num_tokens)

        elif isinstance(other, list):
            # TokenArray + list: convert list to array, then concat
            result_ptr = call_tokens_concat_with_list(self._ptr, self._num_tokens, other)
            if not result_ptr:
                raise MemoryError("Failed to allocate memory for token concatenation")
            return TokenArray(result_ptr, self._num_tokens + len(other))

        return NotImplemented

    def __radd__(self, other: list[int]) -> "TokenArray":
        """
        Concatenate when list is on the left side.

        Args:
            other: List of token IDs to prepend.

        Returns
        -------
            New TokenArray containing tokens from both.

        Example:
            >>> tokens = tokenizer.encode("World")
            >>> combined = [1, 2, 3] + tokens
        """
        if isinstance(other, list):
            result_ptr = call_list_concat_with_tokens(other, self._ptr, self._num_tokens)
            if not result_ptr:
                raise MemoryError("Failed to allocate memory for token concatenation")
            return TokenArray(result_ptr, len(other) + self._num_tokens)

        return NotImplemented

    # =========================================================================
    # DLPack Protocol (zero-copy PyTorch/JAX interop)
    # =========================================================================

    def __dlpack__(self, *, stream=None):
        """
        DLPack protocol for zero-copy export to PyTorch/JAX.

        Returns a PyCapsule containing a DLManagedTensor that can be consumed
        by `torch.from_dlpack()` or `jax.dlpack.from_dlpack()`.

        REFCOUNTED SEMANTICS:
        - The buffer's refcount is incremented (TokenArray remains valid)
        - Multiple exports from the same TokenArray are safe
        - The consumer (PyTorch/JAX) will call the deleter which decrements refcount
        - Buffer is freed only when all references are released

        SHARED STORAGE WARNING:
        - Exported tensors share storage with this TokenArray
        - If the consumer mutates the tensor, the underlying storage is mutated
        - TokenArray APIs treat tokens as immutable; mutation is allowed but not recommended
        - If you need isolation, clone/copy in the consumer framework

        Args:
            stream: CUDA stream (ignored, CPU-only).

        Returns
        -------
            PyCapsule containing DLManagedTensor*.

        Raises
        ------
            InteropError: If array is empty (code="EMPTY_EXPORT") or
                DLPack export fails (code="DLPACK_EXPORT_FAILED").
            StateError: If buffer is not initialized (code="STATE_UNINITIALIZED").

        Example:
            >>> tokens = tokenizer.encode("Hello world")
            >>> tensor = torch.from_dlpack(tokens)  # Zero-copy!
            >>> len(tokens)  # Still valid!
            2
            >>> tensor2 = torch.from_dlpack(tokens)  # Multiple exports safe!
        """
        if self._num_tokens == 0:
            raise InteropError(
                "Cannot export empty TokenArray via DLPack. "
                "Encode at least one token before exporting.",
                code="EMPTY_EXPORT",
            )

        if not self._buffer_handle:
            raise StateError(
                "TokenArray has no buffer (not properly initialized). "
                "Ensure it was created via Tokenizer.encode().",
                code="STATE_UNINITIALIZED",
            )

        dlpack_ptr = call_buffer_to_dlpack(
            self._buffer_handle,
            self._offset_elems,
            self._num_tokens,
        )

        if not dlpack_ptr:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise InteropError(
                f"Failed to create DLPack capsule: {error}",
                code="DLPACK_EXPORT_FAILED",
            )

        return create_dlpack_capsule(dlpack_ptr)

    def __dlpack_device__(self):
        """
        Return device tuple for DLPack protocol.

        Returns (device_type, device_id) where device_type 1 = CPU.

        Returns
        -------
            Tuple (1, 0) indicating CPU device 0.
        """
        return (1, 0)  # kDLCPU = 1, device_id = 0


class TokenArrayView:
    """
    Lightweight view into a BatchEncoding's contiguous buffer.

    Zero-copy view that slices into the parent's memory. The parent
    BatchEncoding must remain alive while this view is used.

    Users interact with this class when indexing into a BatchEncoding:

        >>> batch = tokenizer.encode(["Hello", "World"])
        >>> view = batch[0]  # Returns TokenArrayView
        >>> list(view)
        [9707]
    """

    __slots__ = ("_ids_ptr", "_start", "_length", "_parent")

    def __init__(self, ids_ptr: Any, start: int, length: int, parent: "BatchEncoding"):
        self._ids_ptr = ids_ptr
        self._start = start
        self._length = length
        self._parent = parent

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int | slice) -> int | list[int]:
        """Get token by index or slice."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._length)
            return [self._ids_ptr[self._start + i] for i in range(start, stop, step)]

        if idx < 0:
            idx = self._length + idx
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Token index {idx} out of range [0, {self._length})")
        return self._ids_ptr[self._start + idx]

    def __iter__(self) -> Iterator[int]:
        for i in range(self._length):
            yield self._ids_ptr[self._start + i]

    def __contains__(self, value: int) -> bool:
        """Check if a token ID is in the view."""
        for i in range(self._length):
            if self._ids_ptr[self._start + i] == value:
                return True
        return False

    def __eq__(self, other: object) -> bool:
        """Compare with another TokenArrayView or list."""
        if isinstance(other, TokenArrayView):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(len(self)))
        if isinstance(other, list):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(len(self)))
        return NotImplemented

    def index(self, value: int, start: int = 0, stop: int | None = None) -> int:
        """Return index of first occurrence of value.

        Args:
            value: Token ID to search for.
            start: Start index for search.
            stop: Stop index for search, or None for end of view.
        """
        if stop is None:
            stop = self._length
        for i in range(start, stop):
            if self._ids_ptr[self._start + i] == value:
                return i
        raise ValueError(f"{value} is not in TokenArrayView")

    def count(self, value: int) -> int:
        """Return number of occurrences of value.

        Args:
            value: Token ID to count.
        """
        count = 0
        for i in range(self._length):
            if self._ids_ptr[self._start + i] == value:
                count += 1
        return count

    def tolist(self) -> list[int]:
        """Convert to a Python list (copies data)."""
        return [self._ids_ptr[self._start + i] for i in range(self._length)]

    @property
    def __array_interface__(self) -> dict:
        """NumPy array interface for zero-copy access to this slice."""
        slice_addr = get_ptr_address_offset(self._ids_ptr, self._start * 4)
        return {
            "version": 3,
            "shape": (self._length,),
            "typestr": "<u4",
            "data": (slice_addr, False),
            "strides": (4,),
        }

    def __repr__(self) -> str:
        if self._length <= 10:
            tokens_str = str(self.tolist())
        else:
            first = [self._ids_ptr[self._start + i] for i in range(5)]
            last = [self._ids_ptr[self._start + self._length - 3 + i] for i in range(3)]
            tokens_str = f"[{', '.join(map(str, first))}, ..., {', '.join(map(str, last))}]"
        return f"TokenArrayView({tokens_str}, len={self._length})"
