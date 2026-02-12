"""
Model-specific chat template tests.

Tests chat template behavior for each supported model family,
comparing exact output against transformers.
"""

import re

import pytest

from tests.tokenizer.conftest import (
    MODEL_REGISTRY,
    find_cached_model_path,
    load_hf_tokenizer,
    load_tokenizer,
)


def normalize_dates(text: str) -> str:
    """Normalize dynamic dates in chat templates for comparison.

    Talu uses UTC while HuggingFace transformers uses local time for strftime_now.
    This can cause off-by-one day differences near midnight. Normalize all date
    patterns to a fixed placeholder so comparisons are timezone-independent.

    Patterns handled:
    - "20 Jan 2026" -> "[DATE]" (Llama3 style)
    - "January 20, 2026" -> "[DATE]" (Granite style)
    - "2026-01-20" -> "[DATE]" (ISO style)
    """
    # Llama3 style: "20 Jan 2026"
    text = re.sub(
        r"\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}", "[DATE]", text
    )
    # Granite style: "January 20, 2026"
    text = re.sub(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}",
        "[DATE]",
        text,
    )
    # ISO style: "2026-01-20"
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "[DATE]", text)
    return text


# =============================================================================
# Models that can't be compared against HuggingFace transformers
# =============================================================================
#
# ministral3:
#   - TokenizersBackend requires transformers v5 (not yet released)
#   - Talu chat template works fine, just can't compare against HF
#
# llama3:
#   - Some multi-turn tests have known differences
#   - LLaMA3 tokenizer uses special handling for turn boundaries
#
SKIP_MODELS_FOR_HF_COMPARISON = {"ministral3"}

# =============================================================================
# Known issues for xfail markers
# =============================================================================
#
# Format: reason string with context
#
# Ministral/Mistral: TokenizersBackend requires transformers v5
#   - Current transformers (v4.x) doesn't support Ministral tokenizer backend
#   - Talu implementation works correctly, just can't compare
#
XFAIL_MINISTRAL_V5 = "TokenizersBackend requires transformers v5 (not yet released)"


def get_model_tokenizers(model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
    """Get both talu and HF tokenizers for a model."""
    model_info = MODEL_REGISTRY.get(model_name)
    if not model_info:
        pytest.skip(
            f"Unknown model '{model_name}' not in MODEL_REGISTRY - add entry to conftest.py"
        )

    local_path = find_cached_model_path(model_info["hf_id"])
    if not local_path:
        pytest.skip(
            f"Model not cached: {model_info['hf_id']} - run: huggingface-cli download {model_info['hf_id']}"
        )

    tok = load_tokenizer(local_path, tokenizer_cache, talu)
    hf_tok = load_hf_tokenizer(local_path, hf_tokenizer_cache, transformers)

    return tok, hf_tok, model_info


class TestQwenChatTemplate:
    """Tests for Qwen chat template (Qwen3, Qwen2.5, Qwen2)."""

    @pytest.mark.requires_model
    def test_qwen_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Qwen template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "qwen3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result, (
            f"Qwen template mismatch:\n"
            f"  talu: {repr(talu_result[:100])}\n"
            f"  transformers: {repr(hf_result[:100])}"
        )

    @pytest.mark.requires_model
    def test_qwen_with_system_exact_match(
        self, talu, transformers, tokenizer_cache, hf_tokenizer_cache
    ):
        """Qwen template with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "qwen3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_qwen_markers(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Qwen uses <|im_start|> / <|im_end|> markers."""
        tok, _, _ = get_model_tokenizers(
            "qwen3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        result = tok.apply_chat_template([{"role": "user", "content": "Hello"}])

        assert "<|im_start|>" in result
        assert "<|im_end|>" in result
        assert "user" in result
        assert "assistant" in result  # Generation prompt


class TestLlamaChatTemplate:
    """Tests for Llama chat template (Llama2, Llama3, TinyLlama)."""

    @pytest.mark.requires_model
    def test_llama3_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Llama3 template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "llama3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Normalize dates (Talu uses UTC, HF uses local time)
        assert normalize_dates(talu_result) == normalize_dates(hf_result)

    @pytest.mark.requires_model
    def test_llama2_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Llama2/TinyLlama template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "llama2", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "Hello"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_llama3_with_system(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Llama3 with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "llama3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Normalize dates (Talu uses UTC, HF uses local time)
        assert normalize_dates(talu_result) == normalize_dates(hf_result)


class TestGemmaChatTemplate:
    """Tests for Gemma chat template (Gemma2, Gemma3)."""

    @pytest.mark.requires_model
    def test_gemma_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Gemma template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "gemma3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_gemma_with_system(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Gemma with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "gemma3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_gemma_markers(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Gemma uses <start_of_turn> / <end_of_turn> markers."""
        tok, _, _ = get_model_tokenizers(
            "gemma3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        result = tok.apply_chat_template([{"role": "user", "content": "Hello"}])

        assert "<start_of_turn>" in result
        assert "<end_of_turn>" in result


class TestPhiChatTemplate:
    """Tests for Phi chat template (Phi-3, Phi-4)."""

    @pytest.mark.requires_model
    def test_phi_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Phi template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "phi4", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_phi_with_system(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Phi with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "phi4", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    def test_phi_markers(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Phi uses <|user|> / <|assistant|> / <|end|> markers."""
        tok, _, _ = get_model_tokenizers(
            "phi4", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        result = tok.apply_chat_template([{"role": "user", "content": "Hello"}])

        # Phi-4 uses these markers
        assert "<|user|>" in result or "user" in result.lower()


class TestGraniteChatTemplate:
    """Tests for Granite chat template (IBM Granite)."""

    @pytest.mark.requires_model
    def test_granite_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Granite template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "granite3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Normalize dates (Talu uses UTC, HF uses local time)
        assert normalize_dates(talu_result) == normalize_dates(hf_result)

    @pytest.mark.requires_model
    def test_granite_with_system(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Granite with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "granite3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Normalize dates (Talu uses UTC, HF uses local time)
        assert normalize_dates(talu_result) == normalize_dates(hf_result)

    @pytest.mark.requires_model
    def test_granite_markers(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Granite uses <|start_of_role|> / <|end_of_role|> markers."""
        tok, _, _ = get_model_tokenizers(
            "granite3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        result = tok.apply_chat_template([{"role": "user", "content": "Hello"}])

        assert "<|start_of_role|>" in result
        assert "<|end_of_role|>" in result


class TestMistralChatTemplate:
    """Tests for Mistral/Ministral chat template."""

    @pytest.mark.requires_model
    @pytest.mark.xfail(reason=XFAIL_MINISTRAL_V5)
    def test_mistral_exact_match(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Mistral template exactly matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "ministral3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        user_msg = "What is 2+2?"
        messages = [{"role": "user", "content": user_msg}]
        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result

    @pytest.mark.requires_model
    @pytest.mark.xfail(reason=XFAIL_MINISTRAL_V5)
    def test_mistral_with_system(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Mistral with system message matches transformers."""
        tok, hf_tok, _ = get_model_tokenizers(
            "ministral3", talu, transformers, tokenizer_cache, hf_tokenizer_cache
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        talu_result = tok.apply_chat_template(messages)

        hf_result = hf_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        assert talu_result == hf_result


class TestAllModelsExactMatch:
    """Cross-model validation tests."""

    # Models with known template/tokenizer issues that can't be compared to HF
    SKIP_MODELS = {"llama3", "ministral3"}

    @pytest.mark.requires_model
    def test_all_models_user_only(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """All models match transformers for user-only message."""
        user_msg = "What is the capital of France?"
        failures = []

        for model_name in MODEL_REGISTRY:
            if model_name in self.SKIP_MODELS:
                continue
            try:
                tok, hf_tok, info = get_model_tokenizers(
                    model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache
                )

                messages = [{"role": "user", "content": user_msg}]
                talu_result = tok.apply_chat_template(messages)
                hf_result = hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Normalize dates (Talu uses UTC, HF uses local time)
                if normalize_dates(talu_result) != normalize_dates(hf_result):
                    failures.append(
                        {
                            "model": model_name,
                            "talu": talu_result[:80],
                            "hf": hf_result[:80],
                        }
                    )
            except Exception as e:
                if "not cached" not in str(e):
                    failures.append({"model": model_name, "error": str(e)})

        if failures:
            msg = f"{len(failures)} model(s) failed:\n"
            for f in failures:
                if "error" in f:
                    msg += f"  {f['model']}: {f['error']}\n"
                else:
                    msg += f"  {f['model']}:\n    tok: {f['talu'][:50]}...\n    hf:  {f['hf'][:50]}...\n"
            pytest.fail(msg)

    @pytest.mark.requires_model
    def test_all_models_user_and_system(
        self, talu, transformers, tokenizer_cache, hf_tokenizer_cache
    ):
        """All models match transformers for user+system message."""
        user_msg = "Hello"
        system_msg = "You are a helpful assistant."
        failures = []

        for model_name in MODEL_REGISTRY:
            if model_name in self.SKIP_MODELS:
                continue
            try:
                tok, hf_tok, info = get_model_tokenizers(
                    model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache
                )

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                talu_result = tok.apply_chat_template(messages)
                hf_result = hf_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Normalize dates (Talu uses UTC, HF uses local time)
                if normalize_dates(talu_result) != normalize_dates(hf_result):
                    failures.append(
                        {
                            "model": model_name,
                            "talu": talu_result[:80],
                            "hf": hf_result[:80],
                        }
                    )
            except Exception as e:
                if "not cached" not in str(e):
                    failures.append({"model": model_name, "error": str(e)})

        if failures:
            msg = f"{len(failures)} model(s) failed:\n"
            for f in failures:
                if "error" in f:
                    msg += f"  {f['model']}: {f['error']}\n"
                else:
                    msg += f"  {f['model']}:\n    tok: {f['talu'][:50]}...\n    hf:  {f['hf'][:50]}...\n"
            pytest.fail(msg)


class TestEdgeCases:
    """Edge cases tested across models."""

    # Models with known template/tokenizer issues that can't be compared to HF
    SKIP_MODELS = {"llama3", "ministral3"}

    @pytest.mark.requires_model
    def test_special_characters(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Special characters in messages."""
        special_msg = 'Hello! How\'s the <weather>? "Nice" day & stuff...'
        failures = []

        for model_name in MODEL_REGISTRY:
            if model_name in self.SKIP_MODELS:
                continue

            # get_model_tokenizers handles missing cache via pytest.skip
            tok, hf_tok, _ = get_model_tokenizers(
                model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache
            )

            messages = [{"role": "user", "content": special_msg}]
            talu_result = tok.apply_chat_template(messages)
            hf_result = hf_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Normalize dates (Talu uses UTC, HF uses local time)
            if normalize_dates(talu_result) != normalize_dates(hf_result):
                failures.append(model_name)

        assert not failures, f"Special character handling failed for: {failures}"

    @pytest.mark.requires_model
    def test_unicode_content(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Unicode in messages."""
        unicode_msg = "Hello 世界! 🎉 Привет"
        failures = []

        for model_name in MODEL_REGISTRY:
            if model_name in self.SKIP_MODELS:
                continue

            # get_model_tokenizers handles missing cache via pytest.skip
            tok, hf_tok, _ = get_model_tokenizers(
                model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache
            )

            messages = [{"role": "user", "content": unicode_msg}]
            talu_result = tok.apply_chat_template(messages)
            hf_result = hf_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Normalize dates (Talu uses UTC, HF uses local time)
            if normalize_dates(talu_result) != normalize_dates(hf_result):
                failures.append(model_name)

        assert not failures, f"Unicode content handling failed for: {failures}"

    @pytest.mark.requires_model
    def test_code_content(self, talu, transformers, tokenizer_cache, hf_tokenizer_cache):
        """Code snippets in messages."""
        code_msg = "def hello():\n    print('Hello!')\n\nhello()"
        failures = []

        for model_name in MODEL_REGISTRY:
            if model_name in self.SKIP_MODELS:
                continue

            # get_model_tokenizers handles missing cache via pytest.skip
            tok, hf_tok, _ = get_model_tokenizers(
                model_name, talu, transformers, tokenizer_cache, hf_tokenizer_cache
            )

            messages = [{"role": "user", "content": code_msg}]
            talu_result = tok.apply_chat_template(messages)
            hf_result = hf_tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Normalize dates (Talu uses UTC, HF uses local time)
            if normalize_dates(talu_result) != normalize_dates(hf_result):
                failures.append(model_name)

        assert not failures, f"Code content handling failed for: {failures}"
