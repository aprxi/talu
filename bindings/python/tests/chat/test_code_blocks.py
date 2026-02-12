"""Tests for CodeBlock type and code_blocks in OutputText."""

from __future__ import annotations

import pytest

from talu.types import CodeBlock, OutputText


class TestCodeBlock:
    """Tests for CodeBlock dataclass."""

    def test_create_code_block(self) -> None:
        """Create a CodeBlock with all fields."""
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=30,
            language_start=3,
            language_end=9,
            content_start=10,
            content_end=25,
            complete=True,
        )
        assert block.index == 0
        assert block.fence_start == 0
        assert block.fence_end == 30
        assert block.language_start == 3
        assert block.language_end == 9
        assert block.content_start == 10
        assert block.content_end == 25
        assert block.complete is True

    def test_from_dict(self) -> None:
        """Create CodeBlock from dictionary."""
        data = {
            "index": 1,
            "fence_start": 50,
            "fence_end": 100,
            "language_start": 53,
            "language_end": 57,
            "content_start": 58,
            "content_end": 95,
            "complete": True,
        }
        block = CodeBlock.from_dict(data)
        assert block.index == 1
        assert block.fence_start == 50
        assert block.fence_end == 100
        assert block.language_start == 53
        assert block.language_end == 57
        assert block.content_start == 58
        assert block.content_end == 95
        assert block.complete is True

    def test_from_dict_with_defaults(self) -> None:
        """from_dict handles missing fields with defaults."""
        data = {"complete": True}
        block = CodeBlock.from_dict(data)
        assert block.index == 0
        assert block.fence_start == 0
        assert block.complete is True

    def test_get_language(self) -> None:
        """get_language extracts language from source text."""
        # Text: "```python\nprint('hello')\n```"
        text = "```python\nprint('hello')\n```"
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=28,
            language_start=3,
            language_end=9,
            content_start=10,
            content_end=25,
            complete=True,
        )
        assert block.get_language(text) == "python"

    def test_get_language_empty(self) -> None:
        """get_language returns empty string when no language."""
        # Text: "```\ncode\n```"
        text = "```\ncode\n```"
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=12,
            language_start=3,
            language_end=3,  # Empty range
            content_start=4,
            content_end=8,
            complete=True,
        )
        assert block.get_language(text) == ""

    def test_get_content(self) -> None:
        """get_content extracts code from source text."""
        # Text: "```python\nprint('hello')\n```"
        text = "```python\nprint('hello')\n```"
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=28,
            language_start=3,
            language_end=9,
            content_start=10,
            content_end=25,
            complete=True,
        )
        assert block.get_content(text) == "print('hello')\n"

    def test_get_content_multiline(self) -> None:
        """get_content works with multiline code."""
        text = '```rust\nfn main() {\n    println!("hi");\n}\n```'
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=45,
            language_start=3,
            language_end=7,
            content_start=8,
            content_end=41,
            complete=True,
        )
        assert block.get_language(text) == "rust"
        assert "fn main()" in block.get_content(text)
        assert "println!" in block.get_content(text)

    def test_incomplete_block(self) -> None:
        """CodeBlock can represent incomplete (unclosed) blocks."""
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=20,
            language_start=3,
            language_end=9,
            content_start=10,
            content_end=20,
            complete=False,
        )
        assert block.complete is False

    def test_frozen_immutable(self) -> None:
        """CodeBlock is frozen (immutable)."""
        block = CodeBlock(
            index=0,
            fence_start=0,
            fence_end=10,
            language_start=3,
            language_end=6,
            content_start=7,
            content_end=9,
            complete=True,
        )
        with pytest.raises(AttributeError):
            block.index = 1  # type: ignore[misc]


class TestOutputTextCodeBlocks:
    """Tests for code_blocks field in OutputText."""

    def test_output_text_without_code_blocks(self) -> None:
        """OutputText with no code blocks."""
        output = OutputText(text="Hello, world!")
        assert output.text == "Hello, world!"
        assert output.code_blocks is None

    def test_output_text_with_code_blocks(self) -> None:
        """OutputText with code blocks."""
        text = "Here's some code:\n```python\nprint('hi')\n```"
        blocks = [
            CodeBlock(
                index=0,
                fence_start=18,
                fence_end=44,
                language_start=21,
                language_end=27,
                content_start=28,
                content_end=40,
                complete=True,
            )
        ]
        output = OutputText(text=text, code_blocks=blocks)
        assert output.code_blocks is not None
        assert len(output.code_blocks) == 1
        assert output.code_blocks[0].get_language(text) == "python"
        assert output.code_blocks[0].get_content(text) == "print('hi')\n"

    def test_output_text_multiple_code_blocks(self) -> None:
        """OutputText with multiple code blocks."""
        text = "```python\na()\n```\n\n```rust\nb()\n```"
        blocks = [
            CodeBlock(
                index=0,
                fence_start=0,
                fence_end=17,
                language_start=3,
                language_end=9,
                content_start=10,
                content_end=14,
                complete=True,
            ),
            CodeBlock(
                index=1,
                fence_start=19,
                fence_end=34,
                language_start=22,
                language_end=26,
                content_start=27,
                content_end=31,
                complete=True,
            ),
        ]
        output = OutputText(text=text, code_blocks=blocks)
        assert output.code_blocks is not None
        assert len(output.code_blocks) == 2
        assert output.code_blocks[0].get_language(text) == "python"
        assert output.code_blocks[1].get_language(text) == "rust"
