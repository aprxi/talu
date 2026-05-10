"""
FFI bindings for router module.

Provides SamplingParams with Python defaults, StopFlag for cooperative
cancellation, RouterGenerateConfig with custom __init__ methods that handle
Python-to-C conversion with proper reference counting for stop sequences and
logit bias arrays. CBackendUnion (ctypes.Union) for model spec backend
configuration. Model spec FFI functions (build_c_spec, config_canonicalize,
config_get_view, backend_get_capabilities). Response format FFI functions.
Router generation and streaming FFI functions.
"""

import ctypes
import json
from typing import Any

from .._bindings import check, get_lib, take_last_error
from .._native import (
    CBatchResult,
    CEvent,
    BackendCreateOptions,
    CGenerateConfig,
    CLogitBiasEntry,
    CToolCallRef,
)
from .._native import (
    LocalConfig as CLocalConfig,
)
from ..exceptions import GenerationError

try:
    from .._native import OpenAICompatibleConfig as COpenAICompatibleConfig
except ImportError:
    # OpenAI-compatible backend config was removed from core CAPI.
    # Keep a local struct definition so Python imports remain stable while
    # remote backend paths are cleaned up.
    class COpenAICompatibleConfig(ctypes.Structure):
        _fields_ = [
            ("base_url", ctypes.c_char_p),
            ("api_key", ctypes.c_char_p),
            ("org_id", ctypes.c_char_p),
            ("timeout_ms", ctypes.c_int),
            ("max_retries", ctypes.c_int),
            ("custom_headers_json", ctypes.c_char_p),
            ("_reserved", ctypes.c_uint8 * 24),
        ]


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


class _CSamplingParams(ctypes.Structure):
    _fields_ = [
        ("strategy", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("seed", ctypes.c_uint64),
    ]


class SamplingParams(_CSamplingParams):
    """Sampling configuration with Python default values."""

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


class RouterGenerateConfig(CGenerateConfig):
    """Configuration for router generation (C struct).

    Must match CGenerateConfig in core/src/responses/capi_bridge.zig exactly.
    """

    def __init__(
        self,
        max_tokens: int = 0,
        max_completion_tokens: int = 0,
        max_reasoning_tokens: int | None = None,
        temperature: float = -1.0,
        top_k: int = 0,
        top_p: float = -1.0,
        min_p: float = -1.0,
        repetition_penalty: float = -1.0,
        presence_penalty: float = -1.0,
        frequency_penalty: float = -1.0,
        stop_sequences: list[str] | None = None,
        logit_bias: dict[int, float] | None = None,
        seed: int = 0,
        chat_template: str | None = None,
        extra_context: dict | None = None,
        reasoning_effort: str | None = None,
        tools_json: str | None = None,
        tool_choice: str | None = None,
        stop_flag: StopFlag | None = None,
        extra_body: dict | None = None,
        raw_output: bool = False,
        completions_mode: bool = False,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_reasoning_tokens = (
            ctypes.c_size_t(-1).value if max_reasoning_tokens is None else max_reasoning_tokens
        )
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed

        # Store stop sequences - we need to keep references alive
        self._stop_sequence_strs: list[bytes] = []
        self._stop_sequence_array: ctypes.Array | None = None

        if stop_sequences:
            # Convert to bytes and create C array
            self._stop_sequence_strs = [s.encode("utf-8") for s in stop_sequences]
            arr_type = ctypes.c_char_p * len(self._stop_sequence_strs)
            self._stop_sequence_array = arr_type(*self._stop_sequence_strs)
            self.stop_sequences = ctypes.cast(self._stop_sequence_array, ctypes.c_void_p)
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

        # Store reasoning effort - keep reference alive
        self._reasoning_effort_bytes: bytes | None = None
        if reasoning_effort:
            self._reasoning_effort_bytes = reasoning_effort.encode("utf-8")
            self.reasoning_effort = self._reasoning_effort_bytes
        else:
            self.reasoning_effort = None

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
        self.completions_mode = 1 if completions_mode else 0


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


def _raise_generation_error_from_last(message: str) -> None:
    """Raise GenerationError with the latest Zig error context."""
    code, detail = take_last_error()
    details = {"zig_code": code} if code != 0 else {}
    full_message = f"{message}: {detail}" if detail else message
    raise GenerationError(
        full_message,
        code="GENERATION_FAILED",
        details=details,
        original_code=code if code != 0 else None,
    )


def router_append_user_message(
    lib: Any,
    chat_ptr: Any,
    content: str,
    *,
    context: str = "Router.generate()",
) -> None:
    """Append a user message to a chat before batch generation."""
    conversation_ptr = lib.talu_chat_get_conversation(chat_ptr)
    if not conversation_ptr:
        _raise_generation_error_from_last(f"{context} failed: chat has no conversation")

    content_bytes = content.encode("utf-8")
    content_ptr = ctypes.c_char_p(content_bytes)
    result = lib.talu_responses_append_message(
        conversation_ptr,
        1,
        content_ptr,
        len(content_bytes),
    )
    if result < 0:
        _raise_generation_error_from_last(f"{context} failed: failed to append user message")


RunLoopCallbackType = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(CEvent), ctypes.c_size_t, ctypes.c_void_p
)


def router_generate_batch_final(
    lib: Any,
    chat_ptr: Any,
    backend_handle: "TaluInferenceBackendHandle",
    config: "RouterGenerateConfig | None",
) -> dict[str, Any]:
    """Run local final-only generation through the batch C API."""
    batch_handle = lib.talu_batch_create(backend_handle, None)
    if not batch_handle:
        _raise_generation_error_from_last("Router.generate() failed: batch creation failed")

    try:
        request_id = lib.talu_batch_submit(
            batch_handle,
            chat_ptr,
            ctypes.byref(config) if config else None,
        )
        if request_id == 0:
            _raise_generation_error_from_last("Router.generate() failed: batch submit failed")

        pending = ctypes.c_bool(False)

        @RunLoopCallbackType
        def _ignore_events(_events: Any, _count: int, _userdata: Any) -> None:
            return None

        rc = lib.talu_batch_run_loop_final_only(
            batch_handle,
            ctypes.byref(pending),
            ctypes.cast(_ignore_events, ctypes.c_void_p),
            None,
        )
        if rc != 0:
            _raise_generation_error_from_last("Router.generate() failed: batch run loop failed")

        result_ptr = lib.talu_batch_take_result(batch_handle, request_id)
        if not result_ptr:
            _raise_generation_error_from_last(
                "Router.generate() failed: batch generation completed without a result"
            )

        try:
            result = ctypes.cast(result_ptr, ctypes.POINTER(CBatchResult)).contents
            if result.error_code != 0:
                _raise_generation_error_from_last("Router.generate() failed")
            text = batch_result_extract_text(result)
            tool_calls = batch_result_extract_tool_calls(result)
            return {
                "text": text,
                "token_count": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "prefill_ns": result.prefill_ns,
                "generation_ns": result.generation_ns,
                "ttft_ns": result.ttft_ns,
                "finish_reason": result.finish_reason,
                "tool_calls": tool_calls,
            }
        finally:
            lib.talu_batch_result_free(result_ptr)
    finally:
        lib.talu_batch_destroy(batch_handle)


def router_generate_batch_streaming(
    lib: Any,
    chat_ptr: Any,
    backend_handle: "TaluInferenceBackendHandle",
    config: "RouterGenerateConfig",
    callback: Any,
) -> dict[str, Any]:
    """Run local streaming generation through the batch C API."""
    batch_handle = lib.talu_batch_create(backend_handle, None)
    if not batch_handle:
        _raise_generation_error_from_last("Router.stream() failed: batch creation failed")

    try:
        request_id = lib.talu_batch_submit(batch_handle, chat_ptr, ctypes.byref(config))
        if request_id == 0:
            _raise_generation_error_from_last("Router.stream() failed: batch submit failed")

        pending = ctypes.c_bool(False)
        callback_stopped = False

        @RunLoopCallbackType
        def _on_events(events: Any, count: int, _userdata: Any) -> None:
            nonlocal callback_stopped
            if not events or count == 0 or callback_stopped:
                return None

            for idx in range(count):
                event = events[idx]
                if event.request_id != request_id:
                    continue
                if event.event_type != 0 or event.text_len == 0:
                    continue

                text = ctypes.string_at(event.text_ptr, event.text_len).decode(
                    "utf-8", errors="replace"
                )
                keep_going = callback(
                    text,
                    event.item_type,
                    event.content_type,
                    event.tokens_generated,
                    event.timestamp_ns,
                )
                if not keep_going:
                    callback_stopped = True
                    pending.value = True
                    break
            return None

        rc = lib.talu_batch_run_loop(
            batch_handle,
            ctypes.byref(pending),
            ctypes.cast(_on_events, ctypes.c_void_p),
            None,
        )
        if rc != 0:
            _raise_generation_error_from_last("Router.stream() failed: batch run loop failed")

        if callback_stopped and lib.talu_batch_has_active(batch_handle):
            drain_pending = ctypes.c_bool(False)

            @RunLoopCallbackType
            def _ignore_events(_events: Any, _count: int, _userdata: Any) -> None:
                return None

            rc = lib.talu_batch_run_loop_final_only(
                batch_handle,
                ctypes.byref(drain_pending),
                ctypes.cast(_ignore_events, ctypes.c_void_p),
                None,
            )
            if rc != 0:
                _raise_generation_error_from_last(
                    "Router.stream() failed: batch drain loop failed"
                )

        result_ptr = lib.talu_batch_take_result(batch_handle, request_id)
        if not result_ptr:
            _raise_generation_error_from_last(
                "Router.stream() failed: batch generation completed without a result"
            )

        try:
            result = ctypes.cast(result_ptr, ctypes.POINTER(CBatchResult)).contents
            if result.error_code != 0:
                _raise_generation_error_from_last("Router.stream() failed")
            return {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "prefill_ns": result.prefill_ns,
                "generation_ns": result.generation_ns,
                "ttft_ns": result.ttft_ns,
                "finish_reason": result.finish_reason,
            }
        finally:
            lib.talu_batch_result_free(result_ptr)
    finally:
        lib.talu_batch_destroy(batch_handle)


def batch_result_extract_text(result: Any) -> str:
    """Extract text from a CBatchResult struct."""
    if not result.text:
        return ""
    text_bytes = ctypes.cast(result.text, ctypes.c_char_p).value
    return text_bytes.decode("utf-8", errors="replace") if text_bytes else ""


def batch_result_extract_tool_calls(result: Any) -> list[dict[str, str]] | None:
    """Extract tool calls from a CBatchResult struct."""
    if not result.tool_calls or result.tool_call_count == 0:
        return None

    tool_calls_ptr = ctypes.cast(result.tool_calls, ctypes.POINTER(CToolCallRef))
    calls = []
    for i in range(result.tool_call_count):
        c_call = tool_calls_ptr[i]
        calls.append(
            {
                "id": _read_c_string_ptr(c_call.call_id) or "",
                "name": _read_c_string_ptr(c_call.name) or "",
                "arguments": _read_c_string_ptr(c_call.arguments) or "",
            }
        )
    return calls


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
