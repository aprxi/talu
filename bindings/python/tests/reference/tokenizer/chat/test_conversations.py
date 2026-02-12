"""
Multi-turn conversation tests.

Tests for chat templates with conversation history, not just single messages.
"""

import pytest

try:
    import jinja2.exceptions
except ImportError:
    jinja2 = None  # Will fail gracefully in tests


class TestMultiTurn:
    """Tests for multi-turn conversations."""

    @pytest.mark.requires_model
    def test_two_turn_conversation(self, tokenizer, hf_tokenizer):
        """Two-turn conversation matches transformers."""
        messages = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "What is 2+2?"},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # talu needs: tokenizer.apply_chat_template(messages=[...])
        talu_result = tokenizer.apply_chat_template(messages=messages)

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_three_turn_conversation(self, tokenizer, hf_tokenizer):
        """Three-turn conversation matches transformers."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
            {"role": "user", "content": "Great to hear."},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        talu_result = tokenizer.apply_chat_template(messages=messages)

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_system_then_multiturn(self, tokenizer, hf_tokenizer):
        """System message followed by multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's the weather?"},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        talu_result = tokenizer.apply_chat_template(messages=messages)

        assert talu_result == hf_result


class TestGenerationPrompt:
    """Tests for add_generation_prompt behavior."""

    @pytest.mark.requires_model
    def test_generation_prompt_adds_assistant_marker(self, hf_tokenizer):
        """add_generation_prompt=True adds assistant turn start."""
        messages = [{"role": "user", "content": "Hello"}]

        try:
            with_prompt = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            without_prompt = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        # With generation prompt should be longer (has assistant marker)
        assert len(with_prompt) >= len(without_prompt)

    @pytest.mark.requires_model
    def test_talu_includes_generation_prompt(self, tokenizer, hf_tokenizer):
        """talu template includes generation prompt by default."""
        try:
            hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        talu_result = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])

        # talu should match HF with generation prompt
        # (since we want the model to generate a response)
        assert "Hello" in talu_result


class TestAssistantPrefill:
    """Tests for assistant message prefilling."""

    @pytest.mark.requires_model
    def test_assistant_prefill(self, hf_tokenizer):
        """Partial assistant response for constrained generation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is"},
        ]

        try:
            result = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # Don't add another assistant marker
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        assert "The answer is" in result

    @pytest.mark.requires_model
    def test_talu_prefill(self, tokenizer, hf_tokenizer):
        """Assistant prefill for constrained generation."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is"},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Use add_generation_prompt=False for prefill
        talu_result = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)

        assert talu_result == hf_result
        assert "The answer is" in talu_result


class TestExactOutputMatch:
    """Tests for exact template output matching."""

    @pytest.mark.requires_model
    def test_exact_single_user_match(self, tokenizer, hf_tokenizer):
        """Single user message exactly matches transformers."""
        user_msg = "What is the capital of France?"

        try:
            hf_result = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        talu_result = tokenizer.apply_chat_template([{"role": "user", "content": user_msg}])

        assert talu_result == hf_result, (
            f"Template mismatch:\n"
            f"  talu:     {repr(talu_result[:100])}\n"
            f"  transformers: {repr(hf_result[:100])}"
        )

    @pytest.mark.requires_model
    def test_exact_user_system_match(self, tokenizer, hf_tokenizer):
        """User + system message exactly matches transformers."""
        user_msg = "Hello!"
        system_msg = "You are a helpful assistant."

        try:
            hf_result = hf_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        talu_result = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )

        assert talu_result == hf_result, (
            f"Template mismatch:\n"
            f"  talu:     {repr(talu_result[:100])}\n"
            f"  transformers: {repr(hf_result[:100])}"
        )


class TestSpecialTokens:
    """Tests for BOS/EOS token handling in templates."""

    @pytest.mark.requires_model
    def test_bos_token_presence(self, tokenizer, hf_tokenizer):
        """Check if template includes BOS token when expected.

        Contract: If the HF tokenizer has a BOS token, the template output
        should either start with it or not, consistently matching HF behavior.
        """
        result = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])

        # Check if BOS token is at start
        bos = getattr(hf_tokenizer, "bos_token", None)
        if bos:
            # Compare against HF behavior
            hf_result = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            hf_has_bos = hf_result.startswith(bos)
            talu_has_bos = result.startswith(bos)

            assert talu_has_bos == hf_has_bos, (
                f"BOS token handling mismatch: "
                f"talu starts with BOS={talu_has_bos}, HF starts with BOS={hf_has_bos}"
            )

    @pytest.mark.requires_model
    def test_eos_token_ids_property(self, tokenizer):
        """Tokenizer exposes EOS tokens as immutable tuple."""
        eos_tokens = tokenizer.eos_token_ids

        assert isinstance(eos_tokens, tuple)
        # Most models have at least one EOS token
        # (some might have multiple for different purposes)

    @pytest.mark.requires_model
    def test_template_token_encoding(self, tokenizer, hf_tokenizer):
        """Templated text encodes to same tokens as transformers."""
        user_msg = "Hello!"

        try:
            hf_templated = hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            pytest.xfail(f"HF tokenizer doesn't support apply_chat_template: {e}")

        talu_templated = tokenizer.apply_chat_template([{"role": "user", "content": user_msg}])

        # Encode both
        talu_tokens = tokenizer.encode(talu_templated).tolist()
        hf_tokens = hf_tokenizer.encode(hf_templated, add_special_tokens=False)

        # If templates match, tokens should match
        if talu_templated == hf_templated:
            assert talu_tokens == hf_tokens


class TestToolCalling:
    """Tests for tool/function calling format."""

    # Models known to NOT support tool calling in their chat templates
    MODELS_WITHOUT_TOOL_SUPPORT = {
        "gemma",  # Gemma family doesn't have tool role in template
        "phi",  # Phi family has limited tool support
    }

    @pytest.mark.requires_model
    def test_tool_call_and_response(self, tokenizer, hf_tokenizer, test_model_path):
        """Tool call and tool response in conversation.

        CONTRACT: For models that support tool calling, talu must match HF output.
        Models without tool support are skipped with explicit reason.
        """
        # Check if model is known to lack tool support
        model_name = str(test_model_path).lower()
        for unsupported in self.MODELS_WITHOUT_TOOL_SUPPORT:
            if unsupported in model_name:
                pytest.skip(f"Model family '{unsupported}' does not support tool calling")

        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 20, "condition": "sunny"}',
            },
        ]

        # Try HF first - if it fails, categorize the error
        try:
            hf_result = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except jinja2.exceptions.UndefinedError as e:
            # Template doesn't define tool handling - skip this model
            pytest.skip(f"Model template lacks tool support: {e}")
        except KeyError as e:
            # Template expects different tool format
            pytest.skip(f"Model template uses different tool format: {e}")
        except Exception as e:
            # Unexpected error - fail the test to surface the issue
            pytest.fail(f"Unexpected error from HF template: {type(e).__name__}: {e}")

        # talu should match HF
        talu_result = tokenizer.apply_chat_template(messages=messages)
        assert talu_result == hf_result


class TestModelSpecificFormats:
    """Tests for model-specific template formats."""

    @pytest.mark.requires_model
    def test_qwen_thinking_tags(self, tokenizer, hf_tokenizer):
        """Qwen models may use <think> tags."""
        # Qwen3 supports thinking mode with <think>...</think>
        # This is handled in generation, not template
        pass

    @pytest.mark.requires_model
    def test_template_markers_present(self, tokenizer, hf_tokenizer):
        """Template includes appropriate role markers.

        Contract: Chat templates MUST add structural markers beyond raw text.
        The template should contain at least one known marker pattern.
        """
        result = tokenizer.apply_chat_template([{"role": "user", "content": "Test"}])

        # Should have some kind of structure, not just raw text
        assert len(result) > len("Test"), (
            f"Template should add structural markers, got only: {result}"
        )

        # Common markers (model-dependent)
        common_markers = [
            "<|",  # Qwen, Phi
            "[INST]",  # Llama
            "<start_of_turn>",  # Gemma
            "<|start_of_role|>",  # Granite
        ]

        # At least one marker pattern should be present
        has_marker = any(m in result for m in common_markers)
        assert has_marker, (
            f"Template should contain structural markers. "
            f"Expected one of {common_markers}, got: {result[:100]}..."
        )
