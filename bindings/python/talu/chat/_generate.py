"""Shared generation logic for Chat and AsyncChat.

This module contains helper functions that prepare and finalize generation,
reducing duplication between sync and async code paths.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from talu.router.config import Grammar

from ..exceptions import StructuredOutputError

if TYPE_CHECKING:
    from talu.router.config import GenerationConfig, SchemaStrategy


SCHEMA_PLACEHOLDER = "{{ schema }}"


@dataclass
class GenerationContext:
    """Context prepared before generation."""

    effective_config: GenerationConfig
    hooks: Any  # GenerationHooks | None
    generation_start_time: float
    allow_thinking: bool
    max_thinking_tokens: int
    inject_schema_prompt: bool
    schema_strategy: SchemaStrategy
    model_name: str | None
    model_type: str | None
    schema_prompt: str | None
    schema_tokens: int
    messages_for_submit: list[dict[str, Any]] | None
    stop_tokens: set[int]
    prefill_prefix: str | None
    grammar_cleanup: Callable[[], str | None] | None
    actual_response_format: type | dict | None
    use_submit: bool
    notify_storage: Callable[[str], None]


def prepare_generation(
    chat: Any,  # Chat or AsyncChat
    message: str | list[dict[str, Any]],
    config: GenerationConfig | None,
    response_format: type | dict | Grammar | None,
    **kwargs: Any,
) -> GenerationContext:
    """Prepare all context needed for generation.

    This is the common setup code shared between sync and async generation.
    """
    import time as _time

    from talu.types import normalize_message_input

    # Normalize message input
    message = normalize_message_input(message)

    # Build effective config
    effective_config = chat._build_effective_config(config, **kwargs)

    # Get hooks from client
    hooks = chat._client._hooks if chat._client is not None else None
    generation_start_time = _time.perf_counter()

    # Dispatch generation start hook
    input_text = message if isinstance(message, str) else str(message)
    if hooks:
        hooks.dispatch_start(chat, input_text, effective_config)

    # Extract structured output settings
    allow_thinking = effective_config.allow_thinking
    max_thinking_tokens = effective_config.max_thinking_tokens
    inject_schema_prompt = effective_config.inject_schema_prompt
    schema_strategy = effective_config.schema_strategy

    model_name = chat._router.default_model if chat._router else None
    model_type = _get_model_type(model_name, response_format, schema_strategy)

    schema_prompt, schema_tokens = _build_schema_prompt(
        message,
        response_format,
        inject_schema_prompt,
        allow_thinking,
        schema_strategy,
        model_name,
        model_type,
        chat._apply_numeric_const,
    )

    # Build notify_storage callback
    notify_storage = _make_notify_storage(chat, message)

    # Prepare messages for submit
    messages_for_submit = None
    if isinstance(message, str) and response_format is not None:
        messages_for_submit = chat._prepare_messages(
            message,
            response_format=response_format,
            allow_thinking=allow_thinking,
            inject_schema_prompt=inject_schema_prompt,
            schema_strategy=schema_strategy,
            model_name=model_name,
            model_type=model_type,
        )

    stop_tokens = chat._resolve_stop_tokens(model_name) if response_format is not None else set()
    prefill_prefix = chat._detect_prefill_prefix(allow_thinking) if response_format else None

    grammar_cleanup = _setup_grammar(
        chat,
        response_format,
        message,
        stop_tokens,
        prefill_prefix,
        model_name,
        allow_thinking,
        max_thinking_tokens,
    )

    # Extract actual response format for Response.parsed hydration
    actual_response_format = (
        response_format.response_format if isinstance(response_format, Grammar) else response_format
    )

    use_submit = (
        response_format is not None
        and messages_for_submit is not None
        and hasattr(chat._router, "submit")
        and not hasattr(chat._router, "_lib")
    )

    return GenerationContext(
        effective_config=effective_config,
        hooks=hooks,
        generation_start_time=generation_start_time,
        allow_thinking=allow_thinking,
        max_thinking_tokens=max_thinking_tokens,
        inject_schema_prompt=inject_schema_prompt,
        schema_strategy=schema_strategy,
        model_name=model_name,
        model_type=model_type,
        schema_prompt=schema_prompt,
        schema_tokens=schema_tokens,
        messages_for_submit=messages_for_submit,
        stop_tokens=stop_tokens,
        prefill_prefix=prefill_prefix,
        grammar_cleanup=grammar_cleanup,
        actual_response_format=actual_response_format,
        use_submit=use_submit,
        notify_storage=notify_storage,
    )


def _get_model_type(
    model_name: str | None,
    response_format: type | dict | Grammar | None,
    schema_strategy: SchemaStrategy,
) -> str | None:
    """Get model type for schema strategy decisions."""
    if response_format is None or schema_strategy != "auto":
        return None
    if not model_name:
        return None
    try:
        from ..converter import describe

        return describe(model_name).model_type
    except (ImportError, ValueError, RuntimeError, OSError):
        return None


def _build_schema_prompt(
    message: str | list[dict[str, Any]],
    response_format: type | dict | Grammar | None,
    inject_schema_prompt: bool,
    allow_thinking: bool,
    schema_strategy: SchemaStrategy,
    model_name: str | None,
    model_type: str | None,
    apply_numeric_const: Callable[[dict, str | None], dict],
) -> tuple[str | None, int]:
    """Build schema prompt and count tokens."""
    if (
        response_format is None
        or not inject_schema_prompt
        or not isinstance(message, str)
        or isinstance(response_format, Grammar)
    ):
        return None, 0

    from talu.router.schema.convert import normalize_response_format
    from talu.template.schema.injection import schema_to_prompt_description

    schema_dict = normalize_response_format(response_format)
    schema_dict = apply_numeric_const(schema_dict, message)
    schema_prompt = schema_to_prompt_description(
        schema_dict,
        allow_thinking=allow_thinking,
        strategy=schema_strategy,
        model_name=model_name,
        model_type=model_type,
    )

    schema_tokens = 0
    if model_name:
        try:
            from ..tokenizer import Tokenizer

            tokenizer = Tokenizer(model_name)
            schema_tokens = len(tokenizer.encode(schema_prompt, special_tokens=False))
        except (ImportError, ValueError, RuntimeError, OSError):
            pass

    return schema_prompt, schema_tokens


def _make_notify_storage(
    chat: Any,
    message: str | list[dict[str, Any]],  # noqa: ARG001
) -> Callable[[str], None]:
    """Create storage notification callback.

    Note: Storage events are handled by TaluDB when a Database with talu://
    location is used. This callback is a no-op but kept for interface consistency.
    """

    def notify_storage(assistant_text: str) -> None:  # noqa: ARG001
        # Storage events are handled by TaluDB directly.
        # This is a no-op kept for interface consistency with session.py.
        pass

    return notify_storage


def _setup_grammar(
    chat: Any,
    response_format: type | dict | Grammar | None,
    message: str | list[dict[str, Any]],
    stop_tokens: set[int],
    prefill_prefix: str | None,
    model_name: str | None,
    allow_thinking: bool,
    max_thinking_tokens: int,
) -> Callable[[], str | None] | None:
    """Set up grammar constraints for structured output."""
    if response_format is None or chat._router is None or not hasattr(chat._router, "_lib"):
        return None

    from talu.router._bindings import (
        GrammarConfigC,
        clear_response_format,
        set_response_format,
        set_response_format_handle,
        validate_response_format,
    )
    from talu.router.schema.convert import normalize_response_format

    from .._bindings import get_last_error

    stop_list = sorted(stop_tokens)

    prefix_ids: list[int] = []
    if prefill_prefix and model_name:
        try:
            from ..tokenizer import Tokenizer
            from ..tokenizer.token_array import TokenArray

            tokenizer = Tokenizer(model_name)
            encoded = tokenizer.encode(prefill_prefix, special_tokens=False)
            assert isinstance(encoded, TokenArray)
            prefix_ids = list(encoded)
        except (ImportError, ValueError, RuntimeError, OSError):
            pass

    config_c = GrammarConfigC(
        allow_thinking=allow_thinking,
        max_thinking_tokens=max_thinking_tokens,
    )

    if isinstance(response_format, Grammar):
        rc = set_response_format_handle(
            chat._router._lib,
            chat._chat_ptr,
            response_format._handle,
            config_c,
            stop_list,
            prefix_ids,
        )
        if rc != 0:
            err = get_last_error()
            raise StructuredOutputError(
                f"talu_set_response_format_handle failed: {err}"
                if err
                else "talu_set_response_format_handle failed"
            )
    else:
        schema_dict = normalize_response_format(response_format)
        schema_dict = chat._apply_numeric_const(
            schema_dict, message if isinstance(message, str) else None
        )
        schema_json = json.dumps(schema_dict).encode("utf-8")

        rc = set_response_format(
            chat._router._lib,
            chat._chat_ptr,
            schema_json,
            config_c,
            stop_list,
            prefix_ids,
        )
        if rc != 0:
            err = get_last_error()
            raise StructuredOutputError(
                f"talu_set_response_format failed: {err}"
                if err
                else "talu_set_response_format failed"
            )

    def _validate_and_clear_grammar() -> str | None:
        """Validate semantic constraints and clear grammar."""
        error_message: str | None = None
        if chat._router is not None:
            is_valid, err_msg = validate_response_format(chat._router._lib, chat._chat_ptr)
            if not is_valid:
                error_message = err_msg
            clear_response_format(chat._router._lib, chat._chat_ptr)
        return error_message

    return _validate_and_clear_grammar


def extract_json_from_response(text: str) -> str:
    """Extract JSON from response text, skipping thinking blocks."""
    # Find last occurrence of </think> or <|/think|>
    think_end = max(
        text.rfind("</think>"),
        text.rfind("<|/think|>"),
    )
    if think_end != -1:
        # Find the end of the tag
        tag_end = text.find(">", think_end)
        if tag_end != -1:
            text = text[tag_end + 1 :].strip()

    return text


def build_response(
    chat: Any,
    result: dict,
    effective_config: Any,
    response_class: type,
    *,
    _stream_mode: bool = False,
    _response_format: type | dict | Grammar | None = None,
    _prompt: str | None = None,
) -> Any:
    """Build a Response or AsyncResponse from router generation result.

    Args:
        chat: Chat or AsyncChat instance
        result: Generation result dict from router
        effective_config: GenerationConfig used for this generation
        response_class: Response or AsyncResponse class to instantiate
        _stream_mode: Whether this was a streaming response
        _response_format: Schema used for structured output
        _prompt: Rendered prompt for audit trail
    """
    from .response import ResponseMetadata, Timings, Usage

    completion_tokens = result.get("completion_tokens", result.get("token_count", 0))
    prompt_tokens = result.get("prompt_tokens", 0)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    prefill_ns = result.get("prefill_ns", 0)
    generation_ns = result.get("generation_ns", 0)
    timings = Timings.from_ns(prefill_ns, generation_ns, completion_tokens)

    finish_reason = result.get("finish_reason")
    if finish_reason is None:
        max_tokens = effective_config.max_tokens if effective_config else 0
        if max_tokens > 0 and completion_tokens >= max_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

    return response_class(
        text=result["text"],
        tokens=result.get("tokens"),
        finish_reason=finish_reason,
        tool_calls=result.get("tool_calls"),
        usage=usage,
        timings=timings,
        model=chat._router.default_model if chat._router else None,
        logprobs=result.get("logprobs"),
        chat=chat,
        metadata=ResponseMetadata(
            finish_reason=finish_reason,
            schema_tokens=result.get("schema_tokens", 0),
            schema_injection=result.get("schema_injection"),
            grammar_gbnf=result.get("grammar_gbnf"),
            grammar_trace=result.get("grammar_trace"),
            prefill_success=result.get("prefill_success"),
        ),
        _stream_mode=_stream_mode,
        _response_format=_response_format,
        _prompt=_prompt,
    )
