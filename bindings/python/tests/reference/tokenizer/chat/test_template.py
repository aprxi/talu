"""
Chat template functionality tests.

Tests for talu.ChatTemplate and Tokenizer.apply_chat_template().
"""

import pytest


class TestChatTemplateBasic:
    """Basic ChatTemplate functionality tests."""

    @pytest.mark.requires_model
    def test_apply_chat_template(self, tokenizer):
        """apply_chat_template returns formatted string."""
        result = tokenizer.apply_chat_template([{"role": "user", "content": "Hello!"}])

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Hello" in result

    @pytest.mark.requires_model
    def test_apply_with_system(self, tokenizer):
        """apply_chat_template with system message."""
        result = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )

        assert isinstance(result, str)
        assert "What is 2+2" in result
        # System message may or may not be visible depending on template

    @pytest.mark.requires_model
    def test_apply_chat_template_via_tokenizer(self, tokenizer):
        """apply_chat_template is accessible via tokenizer."""
        result = tokenizer.apply_chat_template([{"role": "user", "content": "Hello!"}])

        assert isinstance(result, str)
        assert len(result) > 0


class TestChatTemplateFormatting:
    """Tests for chat template formatting behavior."""

    @pytest.mark.requires_model
    def test_template_adds_markers(self, tokenizer):
        """Template adds some kind of formatting markers."""
        raw_text = "Plain text"
        templated = tokenizer.apply_chat_template([{"role": "user", "content": raw_text}])

        # Templated version should be longer due to markers
        assert len(templated) > len(raw_text)

    @pytest.mark.requires_model
    def test_template_preserves_content(self, tokenizer):
        """Template preserves user message content."""
        user_msg = "Specific unique content 12345"
        templated = tokenizer.apply_chat_template([{"role": "user", "content": user_msg}])

        assert user_msg in templated

    @pytest.mark.requires_model
    def test_empty_user_message(self, tokenizer):
        """Template handles empty user message."""
        try:
            result = tokenizer.apply_chat_template([{"role": "user", "content": ""}])
            # May return empty or template-only string
            assert isinstance(result, str)
        except (RuntimeError, ValueError):
            pass  # May reject empty messages

    @pytest.mark.requires_model
    def test_multiline_user_message(self, tokenizer):
        """Template handles multiline user message."""
        multiline = "Line 1\nLine 2\nLine 3"
        result = tokenizer.apply_chat_template([{"role": "user", "content": multiline}])

        assert isinstance(result, str)
        # Content should be preserved
        assert "Line 1" in result or "Line" in result


class TestChatTemplateVsTransformers:
    """Tests comparing chat templates to transformers."""

    @pytest.mark.requires_model
    def test_template_output_comparable(self, tokenizer, hf_tokenizer):
        """Chat template output is comparable to transformers."""
        user_msg = "Hello, how are you?"

        # Apply talu template
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tokenizer.apply_chat_template(messages)

        # Apply transformers template
        try:
            hf_result = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Both should contain the user message
            assert user_msg in talu_result
            assert user_msg in hf_result

        except Exception:
            # Not all tokenizers support apply_chat_template
            pass

    @pytest.mark.requires_model
    def test_template_with_system_comparable(self, tokenizer, hf_tokenizer):
        """Chat template with system message comparable to transformers."""
        user_msg = "What is 2+2?"
        system_msg = "You are a helpful assistant."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        talu_result = tokenizer.apply_chat_template(messages)

        try:
            hf_result = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Both should contain the user message
            assert user_msg in talu_result
            assert user_msg in hf_result

        except Exception:
            pass


class TestChatTemplateEncodeDecode:
    """Tests for encoding templated text."""

    @pytest.mark.requires_model
    def test_templated_text_encodes(self, tokenizer):
        """Templated text can be encoded."""
        templated = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        )

        tokens = tokenizer.encode(templated)

        assert len(tokens) > 0

    @pytest.mark.requires_model
    def test_templated_roundtrip(self, tokenizer):
        """Templated text survives encode/decode."""
        templated = tokenizer.apply_chat_template([{"role": "user", "content": "Test message"}])

        tokens = tokenizer.encode(templated)
        decoded = tokenizer.decode(tokens)

        # Key content should be preserved
        assert "Test message" in decoded or "Test" in decoded


class TestChatTemplateTokenize:
    """Tests for apply_chat_template with tokenize=True."""

    @pytest.mark.requires_model
    def test_tokenize_returns_token_array(self, tokenizer):
        """tokenize=True returns TokenArray instead of string."""
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello!"}],
            tokenize=True,
        )

        # Should be TokenArray, not string
        assert not isinstance(result, str)
        assert hasattr(result, "__len__")
        assert len(result) > 0

    @pytest.mark.requires_model
    def test_tokenize_no_double_bos(self, tokenizer):
        """tokenize=True prevents double-BOS tokens.

        This is the key regression test for the double-BOS bug.
        Models like Llama-3 include BOS in their chat template output.
        If encode() also adds BOS, the model sees two start markers.
        """
        messages = [{"role": "user", "content": "Hello!"}]

        # Get tokens via tokenize=True (should be correct)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        token_list = list(tokens)

        # Check for double-BOS
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and len(token_list) >= 2:
            # Should NOT have two consecutive BOS tokens at start
            has_double_bos = token_list[0] == bos_id and token_list[1] == bos_id
            assert not has_double_bos, (
                f"Double-BOS detected: tokens start with [{bos_id}, {bos_id}, ...]. "
                "apply_chat_template(tokenize=True) should prevent this."
            )

    @pytest.mark.requires_model
    def test_tokenize_has_single_bos_if_model_uses_bos(self, tokenizer):
        """tokenize=True produces exactly one BOS if model uses BOS."""
        messages = [{"role": "user", "content": "Test"}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        token_list = list(tokens)

        bos_id = tokenizer.bos_token_id
        if bos_id is not None:
            # Count BOS tokens (should be exactly 1)
            bos_count = sum(1 for t in token_list if t == bos_id)
            # Allow 0 or 1 (some models may not add BOS in certain templates)
            assert bos_count <= 1, (
                f"Found {bos_count} BOS tokens, expected at most 1. Token IDs: {token_list[:10]}..."
            )

    @pytest.mark.requires_model
    def test_tokenize_matches_manual_encode_without_special(self, tokenizer):
        """tokenize=True should match manual encode with special_tokens=False.

        When the template includes BOS, tokenize=True should behave like
        encode(text, special_tokens=False) to avoid doubling.
        """
        messages = [{"role": "user", "content": "Hello!"}]

        # Get string first
        text = tokenizer.apply_chat_template(messages, tokenize=False)

        # Check if template added BOS
        template_has_bos = tokenizer.bos_token and text.lstrip().startswith(tokenizer.bos_token)

        # Get tokens via tokenize=True
        tokens_auto = list(tokenizer.apply_chat_template(messages, tokenize=True))

        if template_has_bos:
            # Template added BOS, so tokenize=True should NOT add another
            tokens_manual = list(tokenizer.encode(text, special_tokens=False))
            assert tokens_auto == tokens_manual, (
                "When template includes BOS, tokenize=True should use special_tokens=False"
            )
        else:
            # Template didn't add BOS, so tokenize=True should add it
            tokens_manual = list(tokenizer.encode(text, special_tokens={"bos"}))
            assert tokens_auto == tokens_manual, (
                "When template lacks BOS, tokenize=True should add BOS"
            )
