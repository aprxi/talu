"""
Configuration struct reference lifetime tests.

Verifies that GenerationConfig and similar structs keep
Python objects alive while Zig may access them.
"""

import gc


class TestGenerationConfigReferences:
    """GenerationConfig field reference tests."""

    def test_stop_sequences_kept_alive(self):
        """Stop sequence strings survive config lifetime."""
        from talu import GenerationConfig

        # Create config with stop sequences
        config = GenerationConfig(
            stop_sequences=["</s>", "<|end|>", "STOP"],
        )

        # Force GC
        gc.collect()
        gc.collect()
        gc.collect()

        # Config should still have valid stop sequences
        assert config.stop_sequences is not None
        assert len(config.stop_sequences) == 3
        assert "</s>" in config.stop_sequences
        assert "<|end|>" in config.stop_sequences
        assert "STOP" in config.stop_sequences

    def test_logit_bias_entries_kept_alive(self):
        """Logit bias entries survive config lifetime."""
        from talu import GenerationConfig

        config = GenerationConfig(
            logit_bias={100: 1.5, 200: -1.0, 300: 0.5},
        )

        gc.collect()
        gc.collect()
        gc.collect()

        assert config.logit_bias is not None
        assert len(config.logit_bias) == 3
        assert config.logit_bias[100] == 1.5
        assert config.logit_bias[200] == -1.0
        assert config.logit_bias[300] == 0.5

    def test_config_with_none_fields_safe(self):
        """Config with None/empty fields doesn't crash."""
        from talu import GenerationConfig

        config = GenerationConfig(
            stop_sequences=None,
            logit_bias=None,
        )

        gc.collect()
        gc.collect()
        gc.collect()

        # Should not crash accessing fields
        assert config.stop_sequences is None or config.stop_sequences == []
        assert config.logit_bias is None or config.logit_bias == {}

    def test_config_merge_preserves_references(self):
        """Merged configs preserve all field references."""
        from talu import GenerationConfig

        config1 = GenerationConfig(
            temperature=0.7,
            stop_sequences=["END1"],
        )

        config2 = GenerationConfig(
            max_tokens=100,
            stop_sequences=["END2"],
        )

        # Merge
        merged = config1 | config2

        gc.collect()
        gc.collect()
        gc.collect()

        # Merged config should have config2's stop sequences (right wins)
        assert merged.temperature == 0.7
        assert merged.max_tokens == 100
        assert "END2" in merged.stop_sequences

    def test_config_copy_independent(self):
        """Copied config has independent field storage."""
        from talu import GenerationConfig

        original = GenerationConfig(
            stop_sequences=["STOP"],
            logit_bias={100: 1.0},
        )

        # Create new config with same values
        copied = GenerationConfig(
            stop_sequences=list(original.stop_sequences),
            logit_bias=dict(original.logit_bias),
        )

        # Delete original
        del original
        gc.collect()
        gc.collect()
        gc.collect()

        # Copied should still be valid
        assert copied.stop_sequences == ["STOP"]
        assert copied.logit_bias == {100: 1.0}


class TestConfigMemoryLeaks:
    """Config memory leak tests."""

    def test_many_configs_no_leak(self, memory_tracker):
        """Creating many configs doesn't leak memory."""
        from talu import GenerationConfig

        memory_tracker.capture_baseline()

        for i in range(500):
            config = GenerationConfig(
                temperature=0.7,
                max_tokens=100,
                stop_sequences=[f"STOP{i}", f"END{i}"],
                logit_bias={i: 1.0, i + 1: -1.0},
            )
            del config

        memory_tracker.assert_no_leak(threshold_mb=5, context="500 GenerationConfig cycles")

    def test_config_merge_chain_no_leak(self, memory_tracker):
        """Chained config merges don't leak memory."""
        from talu import GenerationConfig

        memory_tracker.capture_baseline()

        for _ in range(200):
            base = GenerationConfig(temperature=0.5)
            a = GenerationConfig(max_tokens=50)
            b = GenerationConfig(stop_sequences=["STOP"])
            c = GenerationConfig(logit_bias={100: 1.0})

            merged = base | a | b | c
            _ = merged.temperature
            del base, a, b, c, merged

        memory_tracker.assert_no_leak(threshold_mb=5, context="200 config merge chains")


class TestChatConfigReferences:
    """Chat-specific config reference tests."""

    def test_chat_with_config_keeps_references(self):
        """Chat keeps config references alive."""
        from talu import Chat, GenerationConfig

        config = GenerationConfig(
            temperature=0.7,
            stop_sequences=["END"],
        )

        chat = Chat()

        # Store config reference (simulating usage)
        chat_config = config

        # Force GC
        del config
        gc.collect()
        gc.collect()
        gc.collect()

        # Config should still be valid via chat_config
        assert chat_config.temperature == 0.7
        assert "END" in chat_config.stop_sequences

        chat.close()

    def test_chat_default_config_no_leak(self, memory_tracker):
        """Chat with default config doesn't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for _ in range(100):
            chat = Chat()
            chat.append("user", "hello")
            chat.close()

        memory_tracker.assert_no_leak(threshold_mb=10, context="100 Chat with default config")
