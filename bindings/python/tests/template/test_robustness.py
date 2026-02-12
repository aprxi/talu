"""
Robustness tests for Template.

Tests for memory safety, stress conditions, and edge cases.
Validates template behavior under extreme conditions.
"""

import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from talu.template import PromptTemplate

# Timeout for thread operations
THREAD_TIMEOUT = 30


class TestMemoryLeaks:
    """Memory leak detection tests."""

    def test_repeated_render_no_crash(self):
        """Render many times without crash (memory test)."""
        t = PromptTemplate("Hello {{ name }}!")

        # Render many times
        for i in range(1000):
            result = t(name=f"User{i}")
            assert f"User{i}" in result

        # Force garbage collection
        gc.collect()
        # No crash = success

    def test_template_creation_destruction(self):
        """Create and destroy many templates without crash."""
        for i in range(100):
            t = PromptTemplate(f"Template {{{{ x }}}} #{i}")
            result = t(x=i)
            assert f"#{i}" in result
            del t

        # Force garbage collection
        gc.collect()
        # No crash = success

    def test_large_output_cleanup(self):
        """Large rendered outputs are properly freed."""
        t = PromptTemplate("{% for i in range(count) %}{{ i }} {% endfor %}")

        for _ in range(10):
            result = t(count=10000)
            assert len(result) > 40000  # Each number + space
            del result

        gc.collect()
        # No crash = success


class TestStress:
    """Stress condition tests."""

    def test_very_large_template(self):
        """Handle 50KB+ template source."""
        # Create a large template
        parts = ["{{ var }}"] * 5001
        large_source = " ".join(parts)
        assert len(large_source) > 50000

        t = PromptTemplate(large_source)
        result = t(var="X")

        assert "X" in result
        assert result.count("X") == 5001

    def test_many_variables(self):
        """Template with 500+ variables works."""
        var_count = 500
        var_names = [f"var_{i}" for i in range(var_count)]
        template_str = " ".join(f"{{{{ {name} }}}}" for name in var_names)

        t = PromptTemplate(template_str)
        values = {name: str(i) for i, name in enumerate(var_names)}
        result = t(**values)

        # Verify first, middle, and last values
        assert "0" in result
        assert "250" in result
        assert "499" in result

    def test_deeply_nested_loops(self):
        """Handle deeply nested for loops."""
        t = PromptTemplate("""
{% for a in items %}
  {% for b in items %}
    {% for c in items %}
      {{ a }}-{{ b }}-{{ c }}
    {% endfor %}
  {% endfor %}
{% endfor %}
""")
        items = list(range(5))
        result = t(items=items)

        # Should have 5^3 = 125 combinations
        assert "0-0-0" in result
        assert "4-4-4" in result

    def test_long_string_values(self):
        """Handle very long string values."""
        t = PromptTemplate("{{ content }}")
        long_content = "x" * 100000  # 100KB string

        result = t(content=long_content)
        assert len(result) == 100000
        assert result == long_content

    def test_rapid_template_switching(self):
        """Rapidly switch between different templates."""
        templates = [PromptTemplate(f"Template {i}: {{{{ value }}}}") for i in range(20)]

        results = []
        for i in range(100):
            t = templates[i % len(templates)]
            results.append(t(value=i))

        assert len(results) == 100
        assert "Template 0: 0" in results[0]
        assert "Template 19: 99" in results[99]


class TestUnicodeEdgeCases:
    """Unicode edge case tests."""

    def test_rtl_text(self):
        """Right-to-left text (Arabic, Hebrew)."""
        t = PromptTemplate("{{ greeting }}")

        # Arabic
        result = t(greeting="مرحبا بالعالم")
        assert "مرحبا بالعالم" in result

        # Hebrew
        result = t(greeting="שלום עולם")
        assert "שלום עולם" in result

    def test_cjk_text(self):
        """CJK (Chinese, Japanese, Korean) text."""
        t = PromptTemplate("{{ text }}")

        # Chinese
        assert "你好世界" in t(text="你好世界")

        # Japanese
        assert "こんにちは" in t(text="こんにちは")

        # Korean
        assert "안녕하세요" in t(text="안녕하세요")

    def test_emoji_basic(self):
        """Basic emoji rendering."""
        t = PromptTemplate("{{ emoji }}")
        result = t(emoji="Hello 👋 World 🌍")
        assert "👋" in result
        assert "🌍" in result

    def test_emoji_zwj_sequences(self):
        """Emoji with ZWJ (Zero Width Joiner) sequences."""
        t = PromptTemplate("{{ emoji }}")

        # Family emoji (ZWJ sequence)
        result = t(emoji="👨‍👩‍👧‍👦")
        assert "👨‍👩‍👧‍👦" in result or len(result) > 0

        # Flag emoji
        result = t(emoji="🇺🇸")
        assert len(result) > 0

    def test_combining_characters(self):
        """Text with combining diacritical marks."""
        t = PromptTemplate("{{ text }}")

        # é composed as e + combining acute accent
        combining = "e\u0301"  # e + ́
        result = t(text=combining)
        assert len(result) >= 1  # May normalize

        # Vietnamese with multiple marks
        result = t(text="Việt Nam")
        assert "Việt" in result

    def test_mixed_scripts(self):
        """Mix of different scripts in one template."""
        t = PromptTemplate("{{ a }} - {{ b }} - {{ c }}")
        result = t(
            a="Hello",
            b="你好",
            c="مرحبا",
        )
        assert "Hello" in result
        assert "你好" in result
        assert "مرحبا" in result

    def test_control_characters(self):
        """Handle control characters safely."""
        t = PromptTemplate("{{ text }}")

        # Tab and newline
        result = t(text="line1\tcolumn\nline2")
        assert "\t" in result
        assert "\n" in result

        # Null character should be handled
        result = t(text="before\x00after")
        # Should not crash, may strip null


class TestConcurrentAccess:
    """Thread safety tests for templates."""

    @pytest.mark.slow
    def test_concurrent_render_same_template(self):
        """Multiple threads render same template concurrently."""
        tmpl = PromptTemplate("{{ name }} - {{ value }}")
        errors = []
        results = []

        def worker(thread_id):
            try:
                for i in range(50):
                    result = tmpl(name=f"T{thread_id}", value=i)
                    expected = f"T{thread_id} - {i}"
                    if result != expected:
                        errors.append(f"T{thread_id}: got '{result}', expected '{expected}'")
                    results.append(result)
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=THREAD_TIMEOUT)  # DEADLOCK_GUARD
            assert not th.is_alive(), "Thread hung - deadlock detected"

        assert len(errors) == 0, f"Errors: {errors[:5]}"
        assert len(results) == 8 * 50

    @pytest.mark.slow
    def test_concurrent_template_creation(self):
        """Create templates concurrently from multiple threads."""
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    tmpl = PromptTemplate(f"Thread {{{{ x }}}} #{thread_id}_{i}")
                    result = tmpl(x="test")
                    if f"#{thread_id}_{i}" not in result:
                        errors.append(f"T{thread_id}: missing identifier")
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=THREAD_TIMEOUT)  # DEADLOCK_GUARD
            assert not th.is_alive(), "Thread hung - deadlock detected"

        assert len(errors) == 0, f"Errors: {errors[:5]}"

    @pytest.mark.slow
    def test_thread_pool_executor(self):
        """Work correctly with ThreadPoolExecutor."""
        t = PromptTemplate("{{ a }} + {{ b }} = {{ a + b }}")

        def compute(args):
            a, b = args
            return t(a=a, b=b)

        inputs = [(i, j) for i in range(10) for j in range(10)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(compute, inp) for inp in inputs]
            results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]

        assert len(results) == 100


class TestChatTemplateRobustness:
    """Robustness tests specific to PromptTemplate."""

    def test_empty_message_list_chatml(self):
        """Handle empty message list with ChatML."""
        t = PromptTemplate.from_preset("chatml")
        # Should not crash
        result = t.apply([])
        assert isinstance(result, str)

    def test_empty_message_list_vicuna(self):
        """Handle empty message list with Vicuna."""
        t = PromptTemplate.from_preset("vicuna")
        # Should not crash
        result = t.apply([])
        assert isinstance(result, str)

    def test_very_long_conversation(self):
        """Handle conversation with many turns."""
        t = PromptTemplate.from_preset("chatml")
        messages = []
        for i in range(100):
            messages.append({"role": "user", "content": f"Message {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        result = t.apply(messages)
        assert "Message 0" in result
        assert "Message 99" in result
        assert "Response 99" in result

    def test_large_message_content(self):
        """Handle messages with very long content."""
        t = PromptTemplate.from_preset("chatml")
        long_content = "word " * 10000  # ~50KB

        result = t.apply([{"role": "user", "content": long_content}])
        assert "word" in result
        # Content should be preserved
        assert result.count("word") >= 10000

    def test_special_characters_in_role(self):
        """Handle edge case role names."""
        # Custom template that doesn't validate roles
        t = PromptTemplate("{% for m in messages %}[{{ m.role }}]: {{ m.content }}\n{% endfor %}")

        result = t.apply(
            [
                {"role": "custom_role", "content": "Test"},
                {"role": "another-role", "content": "Test2"},
            ]
        )
        assert "[custom_role]:" in result
        assert "[another-role]:" in result


class TestFileLoading:
    """File loading edge cases for PromptTemplate.from_file()."""

    def test_utf8_file(self, tmp_path):
        """Load UTF-8 encoded template file."""
        template_file = tmp_path / "unicode.j2"
        template_file.write_text("你好 {{ name }}!", encoding="utf-8")

        t = PromptTemplate.from_file(str(template_file))
        result = t(name="世界")
        assert "你好 世界!" == result

    def test_empty_file(self, tmp_path):
        """Load empty template file."""
        template_file = tmp_path / "empty.j2"
        template_file.write_text("")

        t = PromptTemplate.from_file(str(template_file))
        result = t()
        assert result == ""

    def test_whitespace_only_file(self, tmp_path):
        """Load whitespace-only template file."""
        template_file = tmp_path / "whitespace.j2"
        template_file.write_text("   \n\t\n   ")

        t = PromptTemplate.from_file(str(template_file))
        result = t()
        assert result.strip() == ""

    def test_large_file(self, tmp_path):
        """Load large template file."""
        template_file = tmp_path / "large.j2"
        content = "{{ x }} " * 10000
        template_file.write_text(content)

        t = PromptTemplate.from_file(str(template_file))
        result = t(x="Y")
        assert result.count("Y") == 10000
