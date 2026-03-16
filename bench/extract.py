"""Deterministic answer extraction for multiple-choice evaluation."""

from __future__ import annotations

import re

# Strip reasoning model think blocks.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Strip markdown bold/italic around letters: **C** → C, *B* → B.
_MD_BOLD_RE = re.compile(r"\*{1,2}([A-E])\*{1,2}")

# Priority-ordered extraction patterns.
_PATTERNS = [
    re.compile(r"(?i)(?:the\s+)?answer\s+is\s*:?\s*\(?([A-E])\)?"),
    re.compile(r"(?i)(?:correct\s+)?(?:answer|option)\s*:\s*\(?([A-E])\)?"),
    # Markdown bold answer: **B**
    re.compile(r"\*{1,2}([A-E])\*{1,2}"),
]

# Trailing letter on any line: "B." or "B)" or bare "B" at end.
_TRAILING_RE = re.compile(r"([A-E])\s*[.)]*\s*$", re.MULTILINE)

# First character: starts with A-E followed by non-alpha.
_FIRST_RE = re.compile(r"^([A-E])(?=[^a-zA-Z]|$)")


def extract_answer(text: str) -> str | None:
    """Extract a single letter answer (A-E) from model output.

    Returns the uppercase letter, or None if no answer can be extracted.
    """
    # Strip think blocks from reasoning models, then markdown bold/italic.
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _MD_BOLD_RE.sub(r"\1", cleaned).strip()
    if not cleaned:
        return None

    # Priority 1: explicit answer patterns.
    for pat in _PATTERNS:
        m = pat.search(cleaned)
        if m:
            return m.group(1).upper()

    # Priority 2: trailing letter on any line.
    m = _TRAILING_RE.search(cleaned)
    if m:
        return m.group(1).upper()

    # Priority 3: single-letter response.
    if len(cleaned) == 1 and cleaned.upper() in "ABCDE":
        return cleaned.upper()

    # Priority 4: first character is A-E followed by non-alpha.
    m = _FIRST_RE.match(cleaned)
    if m:
        return m.group(1).upper()

    return None
