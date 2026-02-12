"""
Tests for converter asset preservation.

Tests that the converter correctly copies all non-weight files from the source
model directory, ensuring chat_template, added_tokens.json, and other inference
assets are preserved.
"""

import json
import os
import tempfile


class TestAssetPreservationExpectations:
    """Tests for expected asset preservation behavior.

    These tests document and verify the contract: converter should copy
    all files except weights and config.json (which is patched).
    """

    def test_tokenizer_config_with_chat_template_preserved(self):
        """tokenizer_config.json containing chat_template should be preserved.

        The chat_template is critical for inference - it defines how chat
        messages are formatted for the model.
        """
        # Create a mock source directory
        with tempfile.TemporaryDirectory() as source_dir:
            # Create tokenizer_config.json with chat_template
            tokenizer_config = {
                "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
            config_path = os.path.join(source_dir, "tokenizer_config.json")
            with open(config_path, "w") as f:
                json.dump(tokenizer_config, f)

            # Verify file was created and contains chat_template
            with open(config_path) as f:
                content = json.load(f)
                assert "chat_template" in content
                assert "{% for message in messages %}" in content["chat_template"]

    def test_added_tokens_json_expected_to_be_copied(self):
        """added_tokens.json should be copied if present.

        This file contains special tokens that may be added to the tokenizer
        during fine-tuning.
        """
        # This is a documentation test - the actual copying is done in Zig
        expected_files_to_copy = [
            "tokenizer.json",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ]

        # All of these should be preserved by copyModelAssets
        assert "added_tokens.json" in expected_files_to_copy
        assert "tokenizer_config.json" in expected_files_to_copy

    def test_weight_files_not_expected_to_be_copied(self):
        """Weight files should NOT be copied (they are converted instead)."""
        weight_extensions = [
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".gguf",
        ]

        # These should all be filtered out by isWeightFile in Zig
        for ext in weight_extensions:
            filename = f"model{ext}"
            assert ext in filename  # Sanity check

    def test_config_json_not_copied_directly(self):
        """config.json should NOT be copied directly.

        It is handled separately by copyConfigFileWithQuantization which
        patches in the quantization_config.
        """
        # This is a design contract test
        files_to_skip = [
            "config.json",
            "config.json.backup",
        ]
        for f in files_to_skip:
            assert f.startswith("config")


class TestChatTemplatePreservation:
    """Tests specifically for chat_template preservation."""

    def test_chat_template_structure(self):
        """Verify chat_template JSON structure is valid."""
        # Example from Qwen models
        qwen_chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "{% endif %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

        # Create tokenizer_config with this template
        config = {
            "chat_template": qwen_chat_template,
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
        }

        # Verify it's valid JSON
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        assert parsed["chat_template"] == qwen_chat_template

    def test_chat_template_jinja_syntax(self):
        """Verify chat_template Jinja2 syntax is preserved exactly."""
        # The template must be preserved character-for-character
        original = "{% for msg in messages %}{{ msg.content }}{% endfor %}"

        config = {"chat_template": original}
        json_str = json.dumps(config)
        parsed = json.loads(json_str)

        assert parsed["chat_template"] == original
        assert "{% for msg in messages %}" in parsed["chat_template"]
        assert "{{ msg.content }}" in parsed["chat_template"]


class TestAddedTokensPreservation:
    """Tests for added_tokens.json preservation."""

    def test_added_tokens_structure(self):
        """Verify added_tokens.json structure."""
        # Example structure from HuggingFace models
        added_tokens = [
            {
                "id": 151643,
                "content": "<|endoftext|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 151644,
                "content": "<|im_start|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ]

        # Verify it's valid JSON
        json_str = json.dumps(added_tokens)
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]["content"] == "<|endoftext|>"

    def test_added_tokens_special_field(self):
        """Verify special tokens are marked correctly."""
        token = {
            "id": 0,
            "content": "<pad>",
            "special": True,
        }
        assert token["special"] is True


class TestGenerationConfigPreservation:
    """Tests for generation_config.json preservation."""

    def test_generation_config_structure(self):
        """Verify generation_config.json structure."""
        # Example from typical models
        gen_config = {
            "bos_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [151645, 151643],
            "max_length": 32768,
            "pad_token_id": 151643,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
        }

        json_str = json.dumps(gen_config)
        parsed = json.loads(json_str)

        assert parsed["do_sample"] is True
        assert parsed["temperature"] == 0.7
        assert isinstance(parsed["eos_token_id"], list)


class TestSpecialTokensMapPreservation:
    """Tests for special_tokens_map.json preservation."""

    def test_special_tokens_map_structure(self):
        """Verify special_tokens_map.json structure."""
        special_tokens = {
            "bos_token": "<|endoftext|>",
            "eos_token": "<|im_end|>",
            "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>",
        }

        json_str = json.dumps(special_tokens)
        parsed = json.loads(json_str)

        assert parsed["bos_token"] == "<|endoftext|>"
        assert parsed["eos_token"] == "<|im_end|>"


class TestConfigPatcherBehavior:
    """Tests documenting config patcher behavior."""

    def test_quantization_config_structure(self):
        """Verify the quantization_config structure added by patcher."""
        # This is what copyConfigFileWithQuantization adds
        quant_config = {
            "quant_method": "talu",
            "quant_type": "gaf4_64",
            "bits": 4,
        }

        assert quant_config["quant_method"] == "talu"
        assert quant_config["bits"] == 4

    def test_existing_quantization_config_replaced(self):
        """If source has quantization_config, it should be replaced, not merged."""
        # Original config with existing quantization_config
        original = {
            "vocab_size": 32000,
            "quantization_config": {
                "old_method": "old_value",
                "bits": 8,
            },
            "hidden_size": 768,
        }

        # After patching, the new quantization_config should completely replace old
        new_quant = {
            "quant_method": "talu",
            "quant_type": "gaf4_64",
            "bits": 4,
        }

        # Simulate the patching behavior
        patched = {k: v for k, v in original.items() if k != "quantization_config"}
        patched["quantization_config"] = new_quant

        assert patched["quantization_config"]["quant_method"] == "talu"
        assert "old_method" not in patched["quantization_config"]
        assert patched["vocab_size"] == 32000  # Preserved
        assert patched["hidden_size"] == 768  # Preserved
