"""
Model-native chat template formatting.

Applies the Jinja2 chat template embedded in a model's ``tokenizer_config.json``.
This is the low-level interface for rendering prompts exactly as the model expects.

For user-defined prompt templates (few-shot, RAG, custom formats), see
``talu.template.PromptTemplate`` instead.
"""

import json
from typing import Any

from ..exceptions import TemplateError
from ._bindings import call_apply_chat_template

# Type alias for messages
Message = dict[str, str | list | dict]


def apply_chat_template(
    model_path: str,
    messages: list[Message] | Any,  # Also accepts MessageList or any sequence
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply a model's chat template with a list of messages.

    Supports multi-turn conversations, tool calls, and assistant prefill.

    Args:
        model_path: Path to model directory containing tokenizer_config.json
        messages: List of message dicts with 'role' and 'content' keys.
                  Roles can be: 'system', 'user', 'assistant', 'tool'
        add_generation_prompt: Whether to add the assistant prompt marker at the end

    Returns
    -------
        Formatted prompt string

    Raises
    ------
        TaluError: If the C API call fails.
        TemplateError: If template rendering fails or returns empty result.

    Example:
        >>> prompt = apply_chat_template(
        ...     "models/qwen",
        ...     messages=[
        ...         {"role": "system", "content": "You are helpful."},
        ...         {"role": "user", "content": "Hello!"},
        ...         {"role": "assistant", "content": "Hi there!"},
        ...         {"role": "user", "content": "How are you?"},
        ...     ],
        ... )
    """
    from .._bindings import check

    # Convert MessageList to list if needed
    messages_list: list[Any]
    if hasattr(messages, "to_list"):
        messages_list = messages.to_list()  # type: ignore[union-attr]
    elif not isinstance(messages, list):
        messages_list = list(messages)
    else:
        messages_list = messages

    messages_json = json.dumps(messages_list)

    code, result = call_apply_chat_template(
        model_path.encode("utf-8"),
        messages_json.encode("utf-8"),
        add_generation_prompt,
    )

    if code != 0:
        check(code)

    if result is None:
        raise TemplateError(
            f"Failed to apply chat template for model '{model_path}'. "
            "Ensure the model has a valid chat_template in tokenizer_config.json.",
            code="TEMPLATE_RENDER_FAILED",
        )

    return result


class ChatTemplate:
    """
    Chat template formatter for a specific model.

    Loads the chat template from the model's ``tokenizer_config.json``
    and formats message lists into model-native prompts.

    Example:
        >>> template = ChatTemplate("models/qwen")

        # Multi-turn conversation
        >>> prompt = template.apply([
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi!"},
        ...     {"role": "user", "content": "How are you?"},
        ... ])
    """

    def __init__(self, model_path: str):
        """
        Initialize chat template for a model.

        Args:
            model_path: Path to model directory
        """
        self._model_path = model_path

    def apply(
        self,
        messages: list[Message],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply the chat template with a list of messages.

        Supports multi-turn conversations, tool calls, and assistant prefill.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            add_generation_prompt: Whether to add assistant prompt at end.

        Returns
        -------
            Formatted prompt string
        """
        return apply_chat_template(self._model_path, messages, add_generation_prompt)

    def __repr__(self) -> str:
        return f"ChatTemplate({self._model_path!r})"
