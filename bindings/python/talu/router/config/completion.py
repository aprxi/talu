"""Raw-completion-specific options."""

from dataclasses import dataclass

__all__ = ["CompletionOptions"]


@dataclass
class CompletionOptions:
    """
    Raw-completion-specific options.

    These options don't make sense with chat-formatted prompts
    and are ONLY available via ``raw_complete()``.

    Use this for technical use cases where you need low-level
    token control.

    Attributes
    ----------
        token_ids: Pre-tokenized input. Bypasses tokenizer.
            Use case: External tokenization, token-level testing, sending
            tokens directly from another process. Does NOT make sense
            with role-based chat formatting (user/assistant markers).

        continue_from_token_id: Continue generation from a specific token ID.
            Use case: Autocomplete systems, prefix completion, debugging
            token alignment. Does NOT make sense with chat-formatted
            prompts that expect conversation turns.

        echo_prompt: Return input prompt plus generated completion combined.
            Use case: Auto-regressive training data generation, completion-style
            debugging where you need to verify input+output pairing.
            Chat applications already have full conversation history in
            ``messages`` list, so explicit echo doesn't make sense.

    Example:
        >>> from talu.router import CompletionOptions

        >>> # Pre-tokenized input
        >>> opts = CompletionOptions(
        ...     token_ids=[1234, 5678, 9012],
        ...     continue_from_token_id=151645
        ... )
        >>> response = talu.raw_complete(
        ...     "Qwen/Qwen3-0.6B",
        ...     "Continue: ",
        ...     completion_opts=opts
        ... )

        >>> # Echo mode
        >>> opts = CompletionOptions(echo_prompt=True)
        >>> response = talu.raw_complete(
        ...     "Qwen/Qwen3-0.6B",
        ...     "Hello",
        ...     completion_opts=opts
        ... )
    """

    token_ids: list[int] | None = None
    continue_from_token_id: int | None = None
    echo_prompt: bool = False
