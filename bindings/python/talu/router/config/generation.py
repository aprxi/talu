"""Configuration for text generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from .._bindings import SamplingParams, SamplingStrategy

if TYPE_CHECKING:
    from talu.chat.response.format import ResponseFormat

    from ...template import PromptTemplate

# Schema injection strategy type
SchemaStrategy = Literal["auto", "typescript", "json_schema", "xml_schema"]

__all__ = ["GenerationConfig", "SchemaStrategy"]


@dataclass
class GenerationConfig:
    r"""
    Configuration for text generation.

    This dataclass groups all sampling and generation parameters into a single
    object, making it easy to reuse configurations across multiple calls and
    reducing argument bloat in method signatures.

    Mutability
    ----------

    GenerationConfig is mutable, allowing in-place modification of session defaults:

        >>> chat = Chat("model", config=GenerationConfig(temperature=0.7))
        >>> chat.config.temperature = 1.2  # Modify for subsequent calls
        >>> chat.config.max_tokens = 500

    For creating variations without modifying the original, use ``.override()``:

        >>> base = GenerationConfig(temperature=0.7)
        >>> creative = base.override(temperature=1.2)  # New config, base unchanged

    Design Note
    -----------

    GenerationConfig serves as "Request Configuration" - not just sampling
    parameters, but all per-request settings including prompting concerns
    (chat_template, extra_context, schema_strategy, stop_sequences).

    This design enables:

    1. Session-level defaults via ``Chat(config=GenerationConfig(...))``
    2. In-place modification via ``chat.config.temperature = 0.5``
    3. Per-request overrides via ``chat.send(..., schema_strategy="typescript")``
    4. A/B testing by passing different configs to the same Chat

    Parameters like schema_strategy live here (not on Chat) because:

    - "auto" handles 99% of cases without user intervention
    - Session defaults work via ``Chat(config=...)``
    - Per-request flexibility is preserved for experimentation
    - ``Chat.__init__`` stays clean (identity only: model, system, storage)

    Attributes
    ----------
        max_tokens: Maximum number of tokens to generate. Default is 256.
            One token is roughly 4 characters or 0.75 words in English.

        temperature: Controls randomness in generation. Default is 0.7.
            - 0.0: Deterministic (greedy decoding)
            - 0.1-0.5: Focused and consistent
            - 0.7-1.0: Balanced creativity
            - 1.0-2.0: More creative and varied

        top_k: Limits token selection to the k most likely. Default is 50.
            Set to 0 to disable top-k filtering.

        top_p: Nucleus sampling threshold. Default is 0.9.
            Selects from smallest set of tokens whose cumulative probability
            exceeds this threshold.

        min_p: Minimum probability threshold. Default is 0.0 (disabled).
            Tokens with probability below min_p * max_prob are excluded.
            Modern alternative to top_p for some use cases.

        repetition_penalty: Penalty for repeating tokens. Default is 1.0.
            Values > 1.0 discourage repetition, < 1.0 encourage it.

        stop_sequences: List of strings that stop generation. Default is None.
            When any of these strings is generated, generation stops immediately.
            The stop sequence itself is NOT included in the output.

            Multi-token sequences are fully supported. Each stop sequence is
            tokenized and the full token sequence is matched during generation.
            For example, "User:" will only trigger when the complete sequence
            is generated, not when just "User" appears.

        stop_token_ids: List of token IDs that stop generation. Default is None.
            When any of these token IDs is generated, generation stops immediately.

            This overrides the model's default EOS tokens for this request only.
            If you want to ADD to the default EOS tokens (not replace them),
            include the model's eos_token_ids in your list.

            Use cases:
            - Override model's EOS tokens for specific prompts
            - Add additional stop tokens (like newline) for line-by-line generation
            - Implement custom stop logic without string matching overhead

            Example: stop_token_ids=[151645, 198]  # EOS + newline for Qwen

        seed: Random seed for reproducibility. Default is None (random).
            Set to a fixed value for deterministic outputs.

        response_format: Structured output format specification. Default is None.
            Use ResponseFormat to constrain output to JSON or regex patterns.
            Note: Not yet implemented in the runtime.

        logprobs: Whether to return token log probabilities. Default is False.
            When True, GenerationResult.logprobs will contain log probabilities
            for each generated token.

        top_logprobs: Number of top token alternatives to return. Default is None.
            When set (1-20), returns log probabilities for the top N most likely
            tokens at each position. Requires logprobs=True.

        logit_bias: Dictionary mapping token IDs to bias values. Default is None.
            Positive values increase the likelihood of a token being sampled,
            negative values decrease it. Use -100 or lower to effectively ban
            a token from appearing.

            Example: {1234: -100}  # Ban token 1234
            Example: {5678: 5.0}   # Strongly prefer token 5678

        chat_template: Custom chat template for this request. Default is None.
            When set, uses this template instead of the session's chat_template
            or the model's built-in template. Accepts either a template string
            or a PromptTemplate object.

            This follows standard configuration precedence:
            - GenerationConfig.chat_template (per-request) takes priority over
            - Chat.chat_template (session-level) which takes priority over
            - Model's default template (from tokenizer_config.json)

            The template uses Jinja2 syntax with standard variables:
            - messages: List of message dicts with 'role' and 'content'
            - bos_token, eos_token: Special tokens from model config
            - add_generation_prompt: Whether to append assistant prompt

            Example: "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"

        extra_context: Additional template variables. Default is None.
            Dictionary of extra variables to inject into the template context.
            These become available alongside standard variables (messages, etc.).

            Use cases:
            - Tool definitions: {"tools": [{"name": "search", ...}]}
            - System metadata: {"date": "2024-01-15", "user_name": "Alice"}
            - Custom flags: {"enable_thinking": True}

            Example:
                >>> config = GenerationConfig(
                ...     extra_context={"tools": [{"name": "calculator"}]}
                ... )

        tools_json: Tool definitions as a JSON array string. Default is None.
            This is set internally by Chat when tool calling is enabled.

        tool_choice: Tool choice directive ("auto", "required", "none", or function name).
            Default is None and set internally by Chat when tool calling is enabled.

        schema_strategy: Strategy for injecting schema into prompts. Default is "auto".
            Controls how JSON schemas are formatted when using response_format.
            - "auto": Automatically select based on model architecture (recommended)
            - "typescript": TypeScript interface syntax (best for code-trained models)
            - "json_schema": Raw JSON Schema (for older/simpler models)
            - "xml_schema": XML-wrapped schema (for Anthropic-style models)

            Most users should leave this as "auto". Override only when experimenting
            with new models or when you know a specific format works better.

        inject_schema_prompt: Whether to inject schema into the prompt. Default is True.
            When True and response_format is provided, the schema is automatically
            injected into the system prompt or user message.

            Set to False if you've manually included the schema in your prompt
            and want grammar enforcement without auto-injection.

        allow_thinking: Enable chain-of-thought reasoning mode. Default is False.
            When True, allows the model to output a <think>...</think> block
            before the structured response. Useful for complex reasoning tasks.

        max_thinking_tokens: Maximum tokens for thinking block. Default is 512.
            Only applies when allow_thinking=True. Limits the reasoning output
            to prevent excessive token usage.

        validation_retries: Number of automatic retries on SchemaValidationError. Default is 0.
            When using response_format with Pydantic validators, the grammar ensures
            syntactically valid JSON but cannot enforce semantic constraints (e.g.,
            field_validator that requires age < 120).

            When > 0, if parsing raises SchemaValidationError, the Chat automatically:
            1. Appends the error message to conversation history
            2. Regenerates with the same grammar constraint
            3. Repeats until validation passes or retries exhausted

            This closes the loop between grammar (syntactic) and Pydantic (semantic).
            Set to 1-3 for most use cases; higher values rarely help.

        extra_body: Extra parameters for remote API requests. Default is None.
            Dictionary of provider-specific parameters that are merged into the
            request body for OpenAI-compatible APIs. This is the "escape hatch"
            for using new or provider-specific features not yet in GenerationConfig.

            For local inference, this parameter is ignored.

            Example: extra_body={"repetition_penalty": 1.1, "top_a": 0.5}

    Example:
        >>> # Creative writing config
        >>> creative = GenerationConfig(
        ...     temperature=1.2,
        ...     top_p=0.95,
        ...     max_tokens=500
        ... )

        >>> # Precise/deterministic config
        >>> precise = GenerationConfig(
        ...     temperature=0.0,
        ...     max_tokens=100
        ... )

        >>> # JSON extraction config
        >>> json_config = GenerationConfig(
        ...     temperature=0.0,
        ...     stop_sequences=["}"],
        ...     max_tokens=200
        ... )

        >>> # With logprobs
        >>> config = GenerationConfig(
        ...     logprobs=True,
        ...     top_logprobs=5,
        ...     max_tokens=50
        ... )

        >>> # With logit bias (ban specific tokens)
        >>> config = GenerationConfig(
        ...     logit_bias={1234: -100, 5678: -100},  # Ban tokens 1234 and 5678
        ...     max_tokens=100
        ... )

        >>> # Use with Chat
        >>> chat = Chat("model", config=precise)
        >>> chat.send("What is 2+2?")  # Uses precise config
        >>> chat.send("Write a poem", config=creative)  # Override for this call
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop_sequences: list[str] | None = None
    stop_token_ids: list[int] | None = None
    seed: int | None = None
    response_format: ResponseFormat | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    logit_bias: dict[int, float] | None = None
    chat_template: PromptTemplate | str | None = None
    extra_context: dict | None = None
    tools_json: str | None = None
    tool_choice: str | None = None
    # Structured output controls
    schema_strategy: SchemaStrategy = "auto"
    inject_schema_prompt: bool = True
    # Thinking mode (chain-of-thought)
    allow_thinking: bool = False
    max_thinking_tokens: int = 512
    # Validation retry (self-correction loop)
    validation_retries: int = 0
    # Escape hatch for remote API parameters
    extra_body: dict | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of config fields."""
        from talu.chat.response.format import ResponseFormat

        response_format: object | None = None
        if self.response_format is not None:
            if isinstance(self.response_format, ResponseFormat):
                response_format = {
                    "type": self.response_format.type,
                    "json_schema": self.response_format.json_schema,
                }
            else:
                response_format = self.response_format

        chat_template: object | None = self.chat_template
        if chat_template is not None:
            from ...template import PromptTemplate

            if isinstance(chat_template, PromptTemplate):
                chat_template = chat_template.source

        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "stop_sequences": self.stop_sequences,
            "stop_token_ids": self.stop_token_ids,
            "seed": self.seed,
            "response_format": response_format,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
            "logit_bias": self.logit_bias,
            "chat_template": chat_template,
            "extra_context": self.extra_context,
            "tools_json": self.tools_json,
            "tool_choice": self.tool_choice,
            "schema_strategy": self.schema_strategy,
            "inject_schema_prompt": self.inject_schema_prompt,
            "allow_thinking": self.allow_thinking,
            "max_thinking_tokens": self.max_thinking_tokens,
            "validation_retries": self.validation_retries,
            "extra_body": self.extra_body,
        }

    def override(self, **kwargs: object) -> GenerationConfig:
        """
        Create a new config with specified fields overridden.

        Returns a new GenerationConfig with the specified fields changed.
        The original config is unchanged.

        For in-place modification, assign directly: ``config.temperature = 0.5``

        Args:
            **kwargs: Fields to override. Must be valid GenerationConfig fields.

        Returns
        -------
            New GenerationConfig with the specified fields changed.

        Example:
            >>> config = GenerationConfig(temperature=0.7, max_tokens=100)
            >>> creative = config.override(temperature=1.2)
            >>> creative.temperature
            1.2
            >>> config.temperature  # Original unchanged
            0.7
        """
        return replace(self, **kwargs)

    def __or__(self, other: GenerationConfig) -> GenerationConfig:
        """
        Merge two configs with the pipe operator (other wins).

        Creates a new GenerationConfig where fields from `other` override
        fields from `self`, but only if the field in `other` differs from
        its default value. This allows composing partial configs.

        Args:
            other: Config whose non-default values will override self.

        Returns
        -------
            New GenerationConfig with merged fields.

        Example:
            >>> creative = GenerationConfig(temperature=1.2, top_p=0.95)
            >>> json_mode = GenerationConfig(stop_sequences=["}"])
            >>> merged = creative | json_mode  # Both temperature and stop_sequences
            >>> merged.temperature
            1.2
            >>> merged.stop_sequences
            ['}']

        Example - Layer configs for different aspects:
            >>> sampling = GenerationConfig(temperature=0.7, top_k=40)
            >>> limits = GenerationConfig(max_tokens=500)
            >>> full = sampling | limits  # Combines both
        """
        if not isinstance(other, GenerationConfig):
            return NotImplemented

        # Get default values from the class
        defaults = GenerationConfig()

        # Build kwargs from other's non-default values
        merged_kwargs: dict = {}
        for field_name in [
            "max_tokens",
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "stop_sequences",
            "stop_token_ids",
            "seed",
            "response_format",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "chat_template",
            "extra_context",
            "tools_json",
            "tool_choice",
            "schema_strategy",
            "inject_schema_prompt",
            "allow_thinking",
            "max_thinking_tokens",
            "validation_retries",
            "extra_body",
        ]:
            other_val = getattr(other, field_name)
            default_val = getattr(defaults, field_name)
            self_val = getattr(self, field_name)

            # Use other's value if it differs from default, otherwise use self's
            if other_val != default_val:
                merged_kwargs[field_name] = other_val
            else:
                merged_kwargs[field_name] = self_val

        return GenerationConfig(**merged_kwargs)

    def __ror__(self, other: GenerationConfig) -> GenerationConfig:
        """Support other | self when other is also a GenerationConfig."""
        if not isinstance(other, GenerationConfig):
            return NotImplemented
        return other.__or__(self)

    def _to_sampling_params(self) -> SamplingParams:
        """
        Convert to C-compatible SamplingParams struct.

        Returns
        -------
            SamplingParams struct for passing to C-API.
        """
        # Determine strategy based on temperature
        if self.temperature == 0.0:
            strategy = SamplingStrategy.GREEDY
        else:
            strategy = SamplingStrategy.TOP_K

        return SamplingParams(
            strategy=strategy,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if self.seed is not None else 0,
        )
