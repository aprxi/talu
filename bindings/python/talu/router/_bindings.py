"""
FFI bindings for router module.

Provides SamplingParams with Python defaults, StopFlag for cooperative
cancellation, RouterGenerateConfig and GrammarConfigC with custom __init__
methods that handle Python-to-C conversion with proper reference counting
for stop sequences and logit bias arrays. CBackendUnion (ctypes.Union) for
model spec backend configuration. Model spec FFI functions (build_c_spec,
config_canonicalize, config_get_view, backend_get_capabilities).
Response format FFI functions. Router generation and iterator FFI functions.
"""

import ctypes
import json
from typing import Any

from .._bindings import check, get_lib
from .._native import (
    BackendCreateOptions,
    CLogitBiasEntry,
)
from .._native import (
    LocalConfig as CLocalConfig,
)
from .._native import (
    OpenAICompatibleConfig as COpenAICompatibleConfig,
)
from .._native import (
    SamplingParams as _CSamplingParams,
)

# Get the library handle (signatures are set up by _native.py at import time)
_lib = get_lib()


def get_router_lib():
    """Get the router library with all signatures configured.

    Note: Signatures are automatically set up by _native.py (auto-generated
    from Zig C API) when the library is first loaded via get_lib().
    """
    return _lib


# =============================================================================
# Sampling Parameters with Python Defaults
# =============================================================================


class SamplingStrategy:
    """Sampling strategy enum."""

    GREEDY = 0
    TOP_K = 1
    TOP_P = 2


class SamplingParams(_CSamplingParams):
    """Sampling configuration with Python default values.

    Extends the auto-generated SamplingParams struct with a custom __init__
    that provides sensible defaults for common use cases.
    """

    def __init__(
        self,
        strategy: int = SamplingStrategy.GREEDY,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        seed: int = 0,
    ):
        """Initialize sampling parameters with defaults."""
        super().__init__()
        self.strategy = strategy
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed


# =============================================================================
# Stop Flag for Async Cancellation
# =============================================================================


class StopFlag:
    """Thread-safe stop flag for cancelling generation.

    This class wraps a ctypes bool that can be passed to the Zig core for
    cooperative cancellation. When signal() is called, the Zig generation
    loop will detect the flag on its next iteration and stop gracefully.

    Usage:
        stop_flag = StopFlag()

        # In generation thread or async task
        for token in router.stream(chat, msg, stop_flag=stop_flag):
            yield token

        # In cancellation handler (e.g., asyncio.CancelledError)
        stop_flag.signal()

    The stop flag can be reused after calling reset().
    """

    def __init__(self):
        """Create a new stop flag (initially False)."""
        # Use c_bool which is compatible with Zig's std.atomic.Value(bool)
        # ctypes ensures proper memory layout for C ABI
        self._flag = ctypes.c_bool(False)

    def signal(self) -> None:
        """Signal cancellation (set flag to True).

        This is thread-safe - can be called from any thread.
        """
        self._flag.value = True

    def reset(self) -> None:
        """Reset the flag to False for reuse."""
        self._flag.value = False

    def is_set(self) -> bool:
        """Check if the stop flag has been signalled."""
        return self._flag.value

    @property
    def ptr(self) -> int:
        """Get the pointer address for passing to C API.

        Returns an integer that can be cast to c_void_p.
        """
        return ctypes.addressof(self._flag)

    def __bool__(self) -> bool:
        """Allow using stop flag in boolean context."""
        return self._flag.value


# =============================================================================
# Structs with Custom Initialization (keep full definitions here)
# =============================================================================


class CContentPart(ctypes.Structure):
    """
    C-compatible content part structure for router API.

    Matches GenerateContentPart in Zig's router/capi_bridge.zig (32 bytes).
    Used for multimodal input (text, images, etc.).

    Note: This has a different layout than the CContentPart in responses.zig.
    """

    _fields_ = [
        ("content_type", ctypes.c_uint8),  # 0=text, 1=image, 2=audio, 3=video
        ("_padding", ctypes.c_uint8 * 7),  # Align data_ptr to 8 bytes
        ("data_ptr", ctypes.POINTER(ctypes.c_char)),
        ("data_len", ctypes.c_size_t),
        ("mime_ptr", ctypes.c_char_p),  # MIME type (null for text)
    ]


class RouterGenerateConfig(ctypes.Structure):
    """Configuration for router generation (C struct).

    Must match CGenerateConfig in core/src/router/capi_bridge.zig exactly.
    """

    _fields_ = [
        ("max_tokens", ctypes.c_size_t),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_size_t),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("stop_sequences", ctypes.POINTER(ctypes.c_char_p)),  # Array of null-terminated strings
        ("stop_sequence_count", ctypes.c_size_t),
        ("logit_bias", ctypes.POINTER(CLogitBiasEntry)),  # Array of logit bias entries
        ("logit_bias_count", ctypes.c_size_t),
        ("seed", ctypes.c_uint64),  # Random seed for reproducibility (0 = don't reseed)
        ("template_override", ctypes.c_char_p),  # Custom chat template (null = use model's)
        ("extra_context_json", ctypes.c_char_p),  # Extra context JSON object (null = none)
        # Tool calling fields (Story 1)
        ("tools_json", ctypes.c_char_p),  # Tool definitions as JSON array
        (
            "tool_choice",
            ctypes.c_char_p,
        ),  # Tool choice: "auto", "required", "none", or function name
        # Cancellation support - pointer to atomic bool, set to true to stop generation
        ("stop_flag", ctypes.c_void_p),  # Pointer to std.atomic.Value(bool)
        # Extra body JSON for remote API requests (provider-specific parameters)
        ("extra_body_json", ctypes.c_char_p),
        # Preserve raw model output (no reasoning-tag filtering)
        ("raw_output", ctypes.c_uint8),
        # Prefill progress callback: fn(completed_layers, total_layers, userdata)
        ("prefill_progress_fn", ctypes.c_void_p),
        ("prefill_progress_data", ctypes.c_void_p),
    ]

    def __init__(
        self,
        max_tokens: int = 0,
        temperature: float = -1.0,
        top_k: int = 0,
        top_p: float = -1.0,
        min_p: float = -1.0,
        repetition_penalty: float = 0.0,
        stop_sequences: list[str] | None = None,
        logit_bias: dict[int, float] | None = None,
        seed: int = 0,
        chat_template: str | None = None,
        extra_context: dict | None = None,
        tools_json: str | None = None,
        tool_choice: str | None = None,
        stop_flag: StopFlag | None = None,
        extra_body: dict | None = None,
        raw_output: bool = False,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed

        # Store stop sequences - we need to keep references alive
        self._stop_sequence_strs: list[bytes] = []
        self._stop_sequence_array: ctypes.Array | None = None

        if stop_sequences:
            # Convert to bytes and create C array
            self._stop_sequence_strs = [s.encode("utf-8") for s in stop_sequences]
            arr_type = ctypes.c_char_p * len(self._stop_sequence_strs)
            self._stop_sequence_array = arr_type(*self._stop_sequence_strs)
            self.stop_sequences = ctypes.cast(
                self._stop_sequence_array, ctypes.POINTER(ctypes.c_char_p)
            )
            self.stop_sequence_count = len(self._stop_sequence_strs)
        else:
            self.stop_sequences = None
            self.stop_sequence_count = 0

        # Store logit bias - we need to keep references alive
        self._logit_bias_entries: list[CLogitBiasEntry] = []
        self._logit_bias_array: ctypes.Array | None = None

        if logit_bias:
            # Convert dict to array of CLogitBiasEntry
            self._logit_bias_entries = [
                CLogitBiasEntry(token_id=token_id, bias=bias)
                for token_id, bias in logit_bias.items()
            ]
            arr_type = CLogitBiasEntry * len(self._logit_bias_entries)
            self._logit_bias_array = arr_type(*self._logit_bias_entries)
            self.logit_bias = ctypes.cast(self._logit_bias_array, ctypes.POINTER(CLogitBiasEntry))
            self.logit_bias_count = len(self._logit_bias_entries)
        else:
            self.logit_bias = None
            self.logit_bias_count = 0

        # Store chat template - keep reference alive
        # Note: C struct field is named template_override for Zig compatibility
        self._chat_template_bytes: bytes | None = None
        if chat_template:
            self._chat_template_bytes = chat_template.encode("utf-8")
            self.template_override = self._chat_template_bytes  # C struct field name
        else:
            self.template_override = None

        # Store extra context as JSON - keep reference alive
        self._extra_context_bytes: bytes | None = None
        if extra_context:
            self._extra_context_bytes = json.dumps(extra_context).encode("utf-8")
            self.extra_context_json = self._extra_context_bytes
        else:
            self.extra_context_json = None

        # Store tools JSON - keep reference alive
        self._tools_json_bytes: bytes | None = None
        if tools_json:
            self._tools_json_bytes = tools_json.encode("utf-8")
            self.tools_json = self._tools_json_bytes
        else:
            self.tools_json = None

        # Store tool choice - keep reference alive
        self._tool_choice_bytes: bytes | None = None
        if tool_choice:
            self._tool_choice_bytes = tool_choice.encode("utf-8")
            self.tool_choice = self._tool_choice_bytes
        else:
            self.tool_choice = None

        # Store stop flag reference and set pointer
        self._stop_flag_ref: StopFlag | None = stop_flag
        if stop_flag is not None:
            self.stop_flag = ctypes.c_void_p(stop_flag.ptr)
        else:
            self.stop_flag = None

        # Store extra body as JSON - keep reference alive
        self._extra_body_bytes: bytes | None = None
        if extra_body:
            self._extra_body_bytes = json.dumps(extra_body).encode("utf-8")
            self.extra_body_json = self._extra_body_bytes
        else:
            self.extra_body_json = None

        self.raw_output = 1 if raw_output else 0


class GrammarConfigC(ctypes.Structure):
    """Configuration for grammar-constrained decoding (C struct).

    This extends ValidateConfigC with a custom __init__ for Python convenience.
    """

    _fields_ = [
        ("allow_thinking", ctypes.c_bool),
        ("max_thinking_tokens", ctypes.c_size_t),
        ("start_marker", ctypes.c_char_p),
        ("soft_limit_ratio", ctypes.c_float),
        ("soft_limit_bias", ctypes.c_float),
    ]

    def __init__(
        self,
        allow_thinking: bool = False,
        max_thinking_tokens: int = 512,
        *,
        start_marker: str | None = None,
        soft_limit_ratio: float = 0.9,
        soft_limit_bias: float = -2.0,
    ) -> None:
        super().__init__(
            allow_thinking,
            max_thinking_tokens,
            start_marker.encode("utf-8") if start_marker else None,
            soft_limit_ratio,
            soft_limit_bias,
        )


# =============================================================================
# Model Spec System
# =============================================================================


# CBackendUnion and CTaluModelSpec must be manual because the union type
# requires explicit struct composition, not just void pointers
class CBackendUnion(ctypes.Union):
    _fields_ = [
        ("local", CLocalConfig),
        ("openai_compat", COpenAICompatibleConfig),
    ]


class CTaluModelSpec(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("ref", ctypes.c_char_p),
        ("backend_type_raw", ctypes.c_int),
        ("backend_config", CBackendUnion),
    ]


TaluCanonicalSpecHandle = ctypes.c_void_p
TaluInferenceBackendHandle = ctypes.c_void_p


def get_spec_lib():
    """Get the library handle for spec operations."""
    return _lib


# =============================================================================
# Model Spec FFI Functions
# =============================================================================


def _zero_struct(value: ctypes.Structure) -> None:
    """Zero-initialize a ctypes Structure."""
    ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))


def _make_cstr(value: str, buffers: list[ctypes.Array]) -> ctypes.c_char_p:
    """Create a C string buffer, storing reference in buffers to prevent GC."""
    buf = ctypes.create_string_buffer(value.encode("utf-8"))
    buffers.append(buf)
    return ctypes.cast(buf, ctypes.c_char_p)


def build_c_spec(
    ref: str,
    backend_type_raw: int,
    local_config: tuple[int, bool, int] | None,
    openai_config: tuple[str | None, str | None, str | None, int, int, str | None] | None,
) -> tuple["CTaluModelSpec", list[ctypes.Array]]:
    """Build a C model spec struct from Python values.

    Args:
        ref: Model reference string.
        backend_type_raw: Integer backend type (-1=unspecified, 0=local, 1=openai).
        local_config: (gpu_layers, use_mmap, num_threads) or None.
        openai_config: (base_url, api_key, org_id, timeout_ms, max_retries, headers_json) or None.

    Returns
    -------
        Tuple of (CTaluModelSpec, buffers) where buffers must be kept alive.
    """
    buffers: list[ctypes.Array] = []
    spec = CTaluModelSpec()
    _zero_struct(spec)
    spec.abi_version = 1
    spec.struct_size = ctypes.sizeof(CTaluModelSpec)
    spec.ref = _make_cstr(ref, buffers)
    spec.backend_type_raw = backend_type_raw

    if local_config is not None:
        gpu_layers, use_mmap, num_threads = local_config
        spec.backend_config.local = CLocalConfig(
            gpu_layers=gpu_layers,
            use_mmap=1 if use_mmap else 0,
            num_threads=num_threads,
            _reserved=(ctypes.c_uint8 * 32)(),
        )
    elif openai_config is not None:
        base_url, api_key, org_id, timeout_ms, max_retries, headers_json = openai_config
        spec.backend_config.openai_compat = COpenAICompatibleConfig(
            base_url=_make_cstr(base_url, buffers) if base_url is not None else None,
            api_key=_make_cstr(api_key, buffers) if api_key is not None else None,
            org_id=_make_cstr(org_id, buffers) if org_id is not None else None,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            custom_headers_json=_make_cstr(headers_json, buffers)
            if headers_json is not None
            else None,
            _reserved=(ctypes.c_uint8 * 24)(),
        )

    return spec, buffers


def config_canonicalize(
    c_spec: "CTaluModelSpec",
    model_ref: str,
) -> "TaluCanonicalSpecHandle":
    """Call talu_config_canonicalize C API.

    Args:
        c_spec: The C model spec struct.
        model_ref: Model reference for error context.

    Returns
    -------
        Canonical spec handle.
    """
    handle = TaluCanonicalSpecHandle()
    code = get_spec_lib().talu_config_canonicalize(ctypes.byref(c_spec), ctypes.byref(handle))
    check(code, context={"model": model_ref})
    return handle


def config_get_view(handle: "TaluCanonicalSpecHandle") -> "CTaluModelSpec":
    """Call talu_config_get_view C API.

    Args:
        handle: Canonical spec handle.

    Returns
    -------
        Model spec view struct.
    """
    view = CTaluModelSpec()
    _zero_struct(view)
    code = get_spec_lib().talu_config_get_view(handle, ctypes.byref(view))
    check(code)
    return view


def backend_get_capabilities(
    view: "CTaluModelSpec",
) -> tuple[bool, bool, bool, bool, bool]:
    """Call talu_backend_get_capabilities C API.

    Args:
        view: Model spec view from config_get_view.

    Returns
    -------
        Tuple of (streaming, tool_calling, logprobs, embeddings, json_schema).
    """
    from .._native import TaluCapabilities as CTaluCapabilities

    caps = CTaluCapabilities()
    _zero_struct(caps)
    code = get_spec_lib().talu_backend_get_capabilities(
        view.backend_type_raw,
        ctypes.byref(view.backend_config),
        ctypes.byref(caps),
    )
    check(code)
    return (
        bool(caps.streaming),
        bool(caps.tool_calling),
        bool(caps.logprobs),
        bool(caps.embeddings),
        bool(caps.json_schema),
    )


def config_free(handle: "TaluCanonicalSpecHandle") -> None:
    """Call talu_config_free C API."""
    get_spec_lib().talu_config_free(handle)


# =============================================================================
# Grammar/Response Format FFI Functions
# =============================================================================


def create_uint32_array(values: list[int]) -> tuple[Any, int]:
    """Create a ctypes c_uint32 array from a list of integers.

    Returns
    -------
        Tuple of (array or None, length).
    """
    if not values:
        return None, 0
    array_type = ctypes.c_uint32 * len(values)
    return array_type(*values), len(values)


def set_response_format_handle(
    lib: Any,
    chat_ptr: Any,
    grammar_handle: Any,
    config_c: "GrammarConfigC",
    stop_tokens: list[int],
    prefix_ids: list[int],
) -> int:
    """Call talu_set_response_format_handle C API.

    Returns
    -------
        Return code from C API.
    """
    stop_array, stop_len = create_uint32_array(stop_tokens)
    prefix_array, prefix_len = create_uint32_array(prefix_ids)
    return lib.talu_set_response_format_handle(
        chat_ptr,
        grammar_handle,
        ctypes.byref(config_c),
        stop_array,
        stop_len,
        prefix_array,
        prefix_len,
    )


def set_response_format(
    lib: Any,
    chat_ptr: Any,
    schema_json: bytes,
    config_c: "GrammarConfigC",
    stop_tokens: list[int],
    prefix_ids: list[int],
) -> int:
    """Call talu_set_response_format C API.

    Returns
    -------
        Return code from C API.
    """
    stop_array, stop_len = create_uint32_array(stop_tokens)
    prefix_array, prefix_len = create_uint32_array(prefix_ids)
    return lib.talu_set_response_format(
        chat_ptr,
        schema_json,
        ctypes.byref(config_c),
        stop_array,
        stop_len,
        prefix_array,
        prefix_len,
    )


def validate_response_format(
    lib: Any,
    chat_ptr: Any,
) -> tuple[bool, str | None]:
    """Call talu_validate_response_format C API.

    Returns
    -------
        Tuple of (is_valid, error_message or None).
    """
    from .._native import SemanticValidationResultC

    result = SemanticValidationResultC()
    rc = lib.talu_validate_response_format(chat_ptr, ctypes.byref(result))
    if rc == 0 and not result.is_valid:
        path = result.path.decode("utf-8") if result.path else "$"
        msg = result.message.decode("utf-8") if result.message else "Semantic validation failed"
        return False, f"{msg} at {path}"
    return True, None


def clear_response_format(lib: Any, chat_ptr: Any) -> None:
    """Call talu_clear_response_format C API."""
    lib.talu_clear_response_format(chat_ptr)


# =============================================================================
# Router Generation FFI Functions
# =============================================================================


def _read_c_string_ptr(ptr: Any) -> str | None:
    """Read a null-terminated C string from a pointer."""
    if not ptr:
        return None
    return ptr.decode("utf-8", errors="replace")


def router_create_backend(
    lib: Any,
    canonical_handle: "TaluCanonicalSpecHandle",
) -> "TaluInferenceBackendHandle":
    """Call talu_backend_create_from_canonical C API.

    Args:
        lib: The loaded talu shared library.
        canonical_handle: Canonical spec handle from config_canonicalize.

    Returns
    -------
        Backend handle.

    Raises
    ------
        TaluError: If backend creation fails.
    """
    backend_handle = TaluInferenceBackendHandle()
    options = BackendCreateOptions()
    code = lib.talu_backend_create_from_canonical(
        canonical_handle, options, ctypes.byref(backend_handle)
    )
    if code != 0:
        check(code)
    return backend_handle


def router_generate_with_backend(
    lib: Any,
    chat_ptr: Any,
    content_parts: "ctypes.Array",
    parts_count: int,
    backend_handle: "TaluInferenceBackendHandle",
    config: "RouterGenerateConfig | None",
) -> Any:
    """Call talu_router_generate_with_backend C API.

    Args:
        lib: The loaded talu shared library.
        chat_ptr: Pointer to the chat handle.
        content_parts: CContentPart array.
        parts_count: Number of content parts.
        backend_handle: Backend handle.
        config: Optional RouterGenerateConfig.

    Returns
    -------
        RouterResult struct from C API.
    """
    return lib.talu_router_generate_with_backend(
        chat_ptr,
        content_parts,
        parts_count,
        backend_handle,
        ctypes.byref(config) if config else None,
    )


def router_result_extract_text(result: Any) -> str:
    """Extract text from RouterResult struct.

    Args:
        result: RouterResult struct from C API.

    Returns
    -------
        Decoded text string.
    """
    if not result.text:
        return ""
    text_bytes = ctypes.cast(result.text, ctypes.c_char_p).value
    return text_bytes.decode("utf-8", errors="replace") if text_bytes else ""


def router_result_extract_tool_calls(result: Any) -> list[dict[str, str]] | None:
    """Extract tool calls from RouterResult struct.

    Args:
        result: RouterResult struct from C API.

    Returns
    -------
        List of tool call dicts (id, name, arguments) or None.
    """
    if not result.tool_calls or result.tool_call_count == 0:
        return None

    calls = []
    for i in range(result.tool_call_count):
        c_call = result.tool_calls[i]
        calls.append(
            {
                "id": _read_c_string_ptr(c_call.call_id) or "",
                "name": _read_c_string_ptr(c_call.name) or "",
                "arguments": _read_c_string_ptr(c_call.arguments) or "",
            }
        )
    return calls


def router_result_free(lib: Any, result: Any) -> None:
    """Free RouterResult memory.

    Args:
        lib: The loaded talu shared library.
        result: RouterResult struct to free.
    """
    lib.talu_router_result_free(ctypes.byref(result))


def build_router_content_parts(
    parts: list[dict[str, Any]],
) -> tuple["ctypes.Array", list[bytes]]:
    """Build CContentPart array from Python content parts for router.

    Args:
        parts: List of content part dicts in Open Responses format.

    Returns
    -------
        Tuple of (CContentPart array, list of data references to keep alive).
    """
    # Content type mapping (Open Responses format)
    type_map = {
        "input_text": 0,
        "input_image": 1,
        "input_audio": 2,
        "input_video": 3,
        "text": 0,
        "image": 1,
        "audio": 2,
        "video": 3,
    }

    # Keep references to data alive
    data_refs: list[bytes] = []

    # Build array
    arr_type = CContentPart * len(parts)
    c_parts = arr_type()

    for i, part in enumerate(parts):
        part_type = part.get("type", "text")
        content_type = type_map.get(part_type, 0)
        c_parts[i].content_type = content_type

        # Get data based on content type
        if content_type == 0:  # text / input_text
            data = part.get("text", "").encode("utf-8")
        elif content_type == 1:  # image / input_image
            data = part.get("image_url") or part.get("data", "")
            if isinstance(data, str):
                data = data.encode("utf-8")
        elif content_type == 2:  # audio / input_audio
            data = part.get("audio_data") or part.get("data", "")
            if isinstance(data, str):
                data = data.encode("utf-8")
        elif content_type == 3:  # video / input_video
            data = part.get("video_url") or part.get("data", "")
            if isinstance(data, str):
                data = data.encode("utf-8")
        else:
            data = part.get("data", b"")
            if isinstance(data, str):
                data = data.encode("utf-8")

        data_refs.append(data)
        c_parts[i].data_ptr = ctypes.cast(ctypes.c_char_p(data), ctypes.POINTER(ctypes.c_char))
        c_parts[i].data_len = len(data)

        # MIME type (for non-text content)
        mime = part.get("mime")
        if mime:
            mime_bytes = mime.encode("utf-8")
            data_refs.append(mime_bytes)
            c_parts[i].mime_ptr = mime_bytes
        else:
            c_parts[i].mime_ptr = None

    return c_parts, data_refs


def router_embed(
    lib: Any,
    model: str,
    text: str,
    pooling_value: int,
    normalize: bool,
) -> list[float]:
    """Call talu_router_embed C API.

    Args:
        lib: The loaded talu shared library.
        model: Model name.
        text: Text to embed.
        pooling_value: Pooling strategy (0=last, 1=mean, 2=first).
        normalize: Whether to normalize the embedding.

    Returns
    -------
        List of floats representing the embedding vector.

    Raises
    ------
        GenerationError: If embedding extraction fails.
    """
    from talu._bindings import get_last_error
    from talu.exceptions import GenerationError

    # Output pointers
    out_embedding = ctypes.POINTER(ctypes.c_float)()
    out_dim = ctypes.c_size_t(0)

    # Call Zig C API
    result = lib.talu_router_embed(
        model.encode("utf-8"),
        text.encode("utf-8"),
        pooling_value,
        normalize,
        ctypes.byref(out_embedding),
        ctypes.byref(out_dim),
    )

    if result != 0:
        zig_error = get_last_error()
        if zig_error:
            raise GenerationError(
                f"Embedding extraction failed: {zig_error}",
                code="GENERATION_FAILED",
            )

        error_messages = {
            -1: "Invalid model",
            -2: "Invalid text",
            -4: "Engine creation failed",
            -6: "Embedding extraction failed",
            -8: "Memory allocation failed",
        }
        msg = error_messages.get(result, f"Unknown error: {result}")
        raise GenerationError(
            f"Embedding extraction failed: {msg}",
            code="GENERATION_FAILED",
            details={"error_code": result},
        )

    # Convert to Python list
    dim = out_dim.value
    embedding = [out_embedding[i] for i in range(dim)]

    # Free the C memory
    lib.talu_router_embedding_free(out_embedding, dim)

    return embedding


# =============================================================================
# Iterator API (Pull-based Streaming)
# =============================================================================

# Opaque handle type for token iterator
TaluTokenIteratorHandle = ctypes.c_void_p


def iterator_create(
    lib: Any,
    chat_ptr: Any,
    content_parts: "ctypes.Array",
    parts_count: int,
    backend_handle: "TaluInferenceBackendHandle",
    config: "RouterGenerateConfig | None",
) -> TaluTokenIteratorHandle:
    """Create a token iterator for pull-based streaming.

    Args:
        lib: The loaded talu shared library.
        chat_ptr: Pointer to the chat handle.
        content_parts: CContentPart array.
        parts_count: Number of content parts.
        backend_handle: Backend handle.
        config: Optional RouterGenerateConfig.

    Returns
    -------
        Iterator handle, or None on error.
    """
    return lib.talu_router_create_iterator(
        chat_ptr,
        content_parts,
        parts_count,
        backend_handle,
        ctypes.byref(config) if config else None,
    )


def iterator_next(lib: Any, iterator: TaluTokenIteratorHandle) -> str | None:
    """Get the next token from the iterator.

    Blocks until a token is available or generation completes.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        Token string, or None when generation is complete.
    """
    result = lib.talu_router_iterator_next(iterator)
    if result:
        return ctypes.string_at(result).decode("utf-8", errors="replace")
    return None


def iterator_has_error(lib: Any, iterator: TaluTokenIteratorHandle) -> bool:
    """Check if the iterator encountered an error.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        True if an error occurred.
    """
    return lib.talu_router_iterator_has_error(iterator)


def iterator_error_code(lib: Any, iterator: TaluTokenIteratorHandle) -> int:
    """Get the error code from the iterator.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        Error code (0 = no error).
    """
    return lib.talu_router_iterator_error_code(iterator)


def iterator_cancel(lib: Any, iterator: TaluTokenIteratorHandle) -> None:
    """Cancel generation early.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.
    """
    lib.talu_router_iterator_cancel(iterator)


def iterator_free(lib: Any, iterator: TaluTokenIteratorHandle) -> None:
    """Free the iterator and all associated resources.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.
    """
    lib.talu_router_iterator_free(iterator)


def iterator_item_type(lib: Any, iterator: TaluTokenIteratorHandle) -> int:
    """Get the item type of the most recently returned token.

    Returns an ItemType discriminator (u8).
    Values: 0=message, 1=function_call, 3=reasoning, 255=unknown.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        Item type discriminator.
    """
    return lib.talu_router_iterator_item_type(iterator)


def iterator_content_type(lib: Any, iterator: TaluTokenIteratorHandle) -> int:
    """Get the content type of the most recently returned token.

    Returns a ContentType discriminator (u8).
    Values: 5=output_text, 8=reasoning_text, 255=unknown.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        Content type discriminator.
    """
    return lib.talu_router_iterator_content_type(iterator)


def iterator_finish_reason(lib: Any, iterator: TaluTokenIteratorHandle) -> int:
    """Get the finish reason after generation completes.

    Returns a FinishReason discriminator (u8).
    Values: 0=eos_token, 1=length, 2=stop_sequence, 3=tool_calls,
            4=content_filter, 5=cancelled, 255=not_finished.

    Args:
        lib: The loaded talu shared library.
        iterator: Iterator handle.

    Returns
    -------
        Finish reason discriminator.
    """
    return lib.talu_router_iterator_finish_reason(iterator)
