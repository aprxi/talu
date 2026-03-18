"""IFEval instruction verification.

Implements the 25 instruction checkers from the IFEval benchmark
(Zhou et al., 2023).  Based on the reference implementation by Google
Research (Apache 2.0 license).

Dependencies (installed via bench pyproject.toml):
    nltk>=3.9       — sentence tokenization (punkt)
    langdetect>=1.0.9 — language detection
"""

from __future__ import annotations

import json
import re
from typing import Sequence

# ---------------------------------------------------------------------------
# Optional deps — degrade gracefully.
# ---------------------------------------------------------------------------

try:
    import nltk

    # Ensure punkt_tab data is available (download once, silently).
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    def _count_sentences(text: str) -> int:
        return len(nltk.sent_tokenize(text))

    def _count_words(text: str) -> int:
        from nltk.tokenize import RegexpTokenizer

        return len(RegexpTokenizer(r"\w+").tokenize(text))

    def _word_tokenize(text: str) -> list[str]:
        return nltk.word_tokenize(text)

except ImportError:
    nltk = None  # type: ignore[assignment]

    def _count_sentences(text: str) -> int:  # type: ignore[misc]
        return len(re.split(r"(?<=[.!?])\s+", text.strip())) if text.strip() else 0

    def _count_words(text: str) -> int:  # type: ignore[misc]
        return len(re.findall(r"\w+", text))

    def _word_tokenize(text: str) -> list[str]:  # type: ignore[misc]
        return re.findall(r"\w+", text)


try:
    import langdetect as _langdetect
except ImportError:
    _langdetect = None  # type: ignore[assignment]


def _detect_language(text: str) -> str | None:
    """Detect language, return ISO 639-1 code or None on failure."""
    if _langdetect is None:
        return None
    try:
        return _langdetect.detect(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Language codes (subset used by IFEval).
# ---------------------------------------------------------------------------

LANGUAGE_CODES: dict[str, str] = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "ar": "Arabic",
    "hi": "Hindi", "fr": "French", "ru": "Russian", "de": "German",
    "ja": "Japanese", "it": "Italian", "bn": "Bengali", "uk": "Ukrainian",
    "th": "Thai", "ur": "Urdu", "ta": "Tamil", "te": "Telugu",
    "bg": "Bulgarian", "ko": "Korean", "pl": "Polish", "he": "Hebrew",
    "fa": "Persian", "vi": "Vietnamese", "ne": "Nepali", "sw": "Swahili",
    "kn": "Kannada", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi",
    "ml": "Malayalam", "fi": "Finnish",
}

# Reverse mapping: "English" → "en", case-insensitive.
_LANG_NAME_TO_CODE: dict[str, str] = {
    name.lower(): code for code, name in LANGUAGE_CODES.items()
}

# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------

_COMPARISON_RELATION = ("less than", "at least")


def _compare(value: int, relation: str, threshold: int) -> bool:
    if relation == "less than":
        return value < threshold
    elif relation == "at least":
        return value >= threshold
    raise ValueError(f"Unknown relation: {relation!r}")


# ---------------------------------------------------------------------------
# Base class.
# ---------------------------------------------------------------------------

class Instruction:
    """Base class for IFEval instruction checkers."""

    def __init__(self, instruction_id: str) -> None:
        self.id = instruction_id

    def build_description(self, **kwargs: object) -> str:
        raise NotImplementedError

    def get_instruction_args(self) -> dict | None:
        return None

    def get_instruction_args_keys(self) -> list[str]:
        return []

    def check_following(self, value: str) -> bool:
        raise NotImplementedError


# ===================================================================
# Keyword checkers.
# ===================================================================

class KeywordChecker(Instruction):
    """All specified keywords must appear (case-insensitive)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._keywords: list[str] = []

    def build_description(self, *, keywords: Sequence[str] | None = None, **kw: object) -> str:
        if keywords is not None:
            self._keywords = list(keywords)
        return f"Include keywords: {self._keywords}"

    def get_instruction_args(self) -> dict | None:
        return {"keywords": self._keywords} if self._keywords else None

    def get_instruction_args_keys(self) -> list[str]:
        return ["keywords"]

    def check_following(self, value: str) -> bool:
        lower = value.lower()
        return all(kw.lower() in lower for kw in self._keywords)


class KeywordFrequencyChecker(Instruction):
    """A keyword appears < or >= N times (case-insensitive)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._keyword: str = ""
        self._frequency: int = 0
        self._relation: str = "at least"

    def build_description(
        self,
        *,
        keyword: str | None = None,
        frequency: int | None = None,
        relation: str | None = None,
        **kw: object,
    ) -> str:
        if keyword is not None:
            self._keyword = keyword
        if frequency is not None:
            self._frequency = int(frequency)
        if relation is not None:
            self._relation = relation
        return f"Keyword '{self._keyword}' {self._relation} {self._frequency} times"

    def get_instruction_args(self) -> dict | None:
        return {"keyword": self._keyword, "frequency": self._frequency, "relation": self._relation}

    def get_instruction_args_keys(self) -> list[str]:
        return ["keyword", "frequency", "relation"]

    def check_following(self, value: str) -> bool:
        count = len(re.findall(re.escape(self._keyword), value, re.IGNORECASE))
        return _compare(count, self._relation, self._frequency)


class ForbiddenWords(Instruction):
    """None of the forbidden words appear (word-boundary, case-insensitive)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._forbidden: list[str] = []

    def build_description(self, *, forbidden_words: Sequence[str] | None = None, **kw: object) -> str:
        if forbidden_words is not None:
            self._forbidden = list(forbidden_words)
        return f"Forbidden words: {self._forbidden}"

    def get_instruction_args(self) -> dict | None:
        return {"forbidden_words": self._forbidden} if self._forbidden else None

    def get_instruction_args_keys(self) -> list[str]:
        return ["forbidden_words"]

    def check_following(self, value: str) -> bool:
        lower = value.lower()
        for word in self._forbidden:
            if re.search(r"\b" + re.escape(word.lower()) + r"\b", lower):
                return False
        return True


class LetterFrequencyChecker(Instruction):
    """A specific letter appears < or >= N times."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._letter: str = ""
        self._frequency: int = 0
        self._relation: str = "at least"

    def build_description(
        self,
        *,
        letter: str | None = None,
        let_frequency: int | None = None,
        let_relation: str | None = None,
        **kw: object,
    ) -> str:
        if letter is not None:
            self._letter = letter
        if let_frequency is not None:
            self._frequency = int(let_frequency)
        if let_relation is not None:
            self._relation = let_relation
        return f"Letter '{self._letter}' {self._relation} {self._frequency} times"

    def get_instruction_args(self) -> dict | None:
        return {"letter": self._letter, "let_frequency": self._frequency, "let_relation": self._relation}

    def get_instruction_args_keys(self) -> list[str]:
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value: str) -> bool:
        count = value.lower().count(self._letter.lower())
        return _compare(count, self._relation, self._frequency)


# ===================================================================
# Language checker.
# ===================================================================

class ResponseLanguageChecker(Instruction):
    """Entire response is in the specified language."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._language: str = ""

    def build_description(self, *, language: str | None = None, **kw: object) -> str:
        if language is not None:
            self._language = _LANG_NAME_TO_CODE.get(language.lower(), language.lower())
        return f"Response in language: {self._language}"

    def get_instruction_args(self) -> dict | None:
        return {"language": self._language} if self._language else None

    def get_instruction_args_keys(self) -> list[str]:
        return ["language"]

    def check_following(self, value: str) -> bool:
        detected = _detect_language(value)
        if detected is None:
            return True  # Lenient on detection failure (matches reference).
        return detected == self._language


# ===================================================================
# Length constraint checkers.
# ===================================================================

class NumberOfSentences(Instruction):
    """Sentence count satisfies relation (< or >= N)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_sentences: int = 0
        self._relation: str = "at least"

    def build_description(
        self,
        *,
        num_sentences: int | None = None,
        relation: str | None = None,
        **kw: object,
    ) -> str:
        if num_sentences is not None:
            self._num_sentences = int(num_sentences)
        if relation is not None:
            self._relation = relation
        return f"Sentences {self._relation} {self._num_sentences}"

    def get_instruction_args(self) -> dict | None:
        return {"num_sentences": self._num_sentences, "relation": self._relation}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_sentences", "relation"]

    def check_following(self, value: str) -> bool:
        return _compare(_count_sentences(value), self._relation, self._num_sentences)


class ParagraphChecker(Instruction):
    """Exactly N paragraphs (separated by ***)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_paragraphs: int = 0

    def build_description(self, *, num_paragraphs: int | None = None, **kw: object) -> str:
        if num_paragraphs is not None:
            self._num_paragraphs = int(num_paragraphs)
        return f"Exactly {self._num_paragraphs} paragraphs"

    def get_instruction_args(self) -> dict | None:
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_paragraphs"]

    def check_following(self, value: str) -> bool:
        parts = re.split(r"\s?\*\*\*\s?", value)
        paragraphs = [p.strip() for p in parts if p.strip()]
        return len(paragraphs) == self._num_paragraphs


class NumberOfWords(Instruction):
    """Word count satisfies relation (< or >= N)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_words: int = 0
        self._relation: str = "at least"

    def build_description(
        self,
        *,
        num_words: int | None = None,
        relation: str | None = None,
        **kw: object,
    ) -> str:
        if num_words is not None:
            self._num_words = int(num_words)
        if relation is not None:
            self._relation = relation
        return f"Words {self._relation} {self._num_words}"

    def get_instruction_args(self) -> dict | None:
        return {"num_words": self._num_words, "relation": self._relation}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_words", "relation"]

    def check_following(self, value: str) -> bool:
        return _compare(_count_words(value), self._relation, self._num_words)


class ParagraphFirstWordCheck(Instruction):
    """Response has exactly N paragraphs (\\n\\n), Nth starts with a specific word."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_paragraphs: int = 0
        self._nth_paragraph: int = 1
        self._first_word: str = ""

    def build_description(
        self,
        *,
        num_paragraphs: int | None = None,
        nth_paragraph: int | None = None,
        first_word: str | None = None,
        **kw: object,
    ) -> str:
        if num_paragraphs is not None:
            self._num_paragraphs = int(num_paragraphs)
        if nth_paragraph is not None:
            self._nth_paragraph = int(nth_paragraph)
        if first_word is not None:
            self._first_word = first_word
        return (
            f"Exactly {self._num_paragraphs} paragraphs, "
            f"paragraph {self._nth_paragraph} starts with '{self._first_word}'"
        )

    def get_instruction_args(self) -> dict | None:
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value: str) -> bool:
        paragraphs = [p.strip() for p in value.split("\n\n") if p.strip()]
        if len(paragraphs) != self._num_paragraphs:
            return False
        idx = self._nth_paragraph - 1
        if idx < 0 or idx >= len(paragraphs):
            return False
        first = paragraphs[idx].split()[0] if paragraphs[idx].split() else ""
        return first.lower() == self._first_word.lower()


# ===================================================================
# Detectable content checkers.
# ===================================================================

class PlaceholderChecker(Instruction):
    """At least N [placeholder] patterns present."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_placeholders: int = 0

    def build_description(self, *, num_placeholders: int | None = None, **kw: object) -> str:
        if num_placeholders is not None:
            self._num_placeholders = int(num_placeholders)
        return f"At least {self._num_placeholders} placeholders"

    def get_instruction_args(self) -> dict | None:
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_placeholders"]

    def check_following(self, value: str) -> bool:
        return len(re.findall(r"\[.*?\]", value)) >= self._num_placeholders


class PostscriptChecker(Instruction):
    """Response ends with a P.S. or P.P.S postscript."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._postscript_marker: str = "P.S."

    def build_description(self, *, postscript_marker: str | None = None, **kw: object) -> str:
        if postscript_marker is not None:
            self._postscript_marker = postscript_marker
        return f"End with postscript: {self._postscript_marker}"

    def get_instruction_args(self) -> dict | None:
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self) -> list[str]:
        return ["postscript_marker"]

    def check_following(self, value: str) -> bool:
        # Check last few lines for the postscript marker.
        return self._postscript_marker.lower() in value.lower().rstrip().split("\n")[-1].lower()


# ===================================================================
# Detectable format checkers.
# ===================================================================

class BulletListChecker(Instruction):
    """Exactly N bullet points (lines starting with * or -)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_bullets: int = 0

    def build_description(self, *, num_bullets: int | None = None, **kw: object) -> str:
        if num_bullets is not None:
            self._num_bullets = int(num_bullets)
        return f"Exactly {self._num_bullets} bullet points"

    def get_instruction_args(self) -> dict | None:
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_bullets"]

    def check_following(self, value: str) -> bool:
        star_bullets = re.findall(r"^\s*\*[^\*].*$", value, re.MULTILINE)
        dash_bullets = re.findall(r"^\s*-.*$", value, re.MULTILINE)
        return len(star_bullets) + len(dash_bullets) == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Response contains 'My answer is yes/no/maybe.'"""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Response must contain 'My answer is yes/no/maybe.'"

    def check_following(self, value: str) -> bool:
        lower = value.strip().lower()
        return any(
            phrase in lower
            for phrase in ("my answer is yes.", "my answer is no.", "my answer is maybe.")
        )


class HighlightSectionChecker(Instruction):
    """At least N highlighted sections (*text* or **text**)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_highlights: int = 0

    def build_description(self, *, num_highlights: int | None = None, **kw: object) -> str:
        if num_highlights is not None:
            self._num_highlights = int(num_highlights)
        return f"At least {self._num_highlights} highlighted sections"

    def get_instruction_args(self) -> dict | None:
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_highlights"]

    def check_following(self, value: str) -> bool:
        highlights = re.findall(r"\*[^\n\*]+\*", value)
        return len(highlights) >= self._num_highlights


class SectionChecker(Instruction):
    """At least N sections with section markers."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._num_sections: int = 0
        self._section_splitter: str = "Section"

    def build_description(
        self,
        *,
        num_sections: int | None = None,
        section_spliter: str | None = None,
        **kw: object,
    ) -> str:
        if num_sections is not None:
            self._num_sections = int(num_sections)
        if section_spliter is not None:
            self._section_splitter = section_spliter
        return f"At least {self._num_sections} sections"

    def get_instruction_args(self) -> dict | None:
        return {"num_sections": self._num_sections, "section_spliter": self._section_splitter}

    def get_instruction_args_keys(self) -> list[str]:
        return ["num_sections", "section_spliter"]

    def check_following(self, value: str) -> bool:
        parts = value.split(self._section_splitter)
        return len(parts) - 1 >= self._num_sections


class JsonFormat(Instruction):
    """Entire response is valid JSON (strips markdown code fences)."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Response must be valid JSON"

    def check_following(self, value: str) -> bool:
        stripped = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


class TitleChecker(Instruction):
    """Response contains a <<title>> format title."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Response must contain a <<title>>"

    def check_following(self, value: str) -> bool:
        return bool(re.search(r"<<[^>]+>>", value))


# ===================================================================
# Combination checkers.
# ===================================================================

class TwoResponsesChecker(Instruction):
    """Exactly 2 different responses separated by ******."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Two different responses separated by ******"

    def check_following(self, value: str) -> bool:
        parts = value.split("******")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) != 2:
            return False
        return parts[0] != parts[1]


class RepeatPromptThenAnswer(Instruction):
    """Response starts with the original prompt repeated."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._prompt: str = ""

    def build_description(
        self, *, prompt_to_repeat: str | None = None, **kw: object
    ) -> str:
        if prompt_to_repeat is not None:
            self._prompt = prompt_to_repeat
        return "Repeat the prompt then answer"

    def get_instruction_args(self) -> dict | None:
        return {"prompt_to_repeat": self._prompt} if self._prompt else None

    def get_instruction_args_keys(self) -> list[str]:
        return ["prompt_to_repeat"]

    def check_following(self, value: str) -> bool:
        if not self._prompt:
            return True
        # Case-insensitive, whitespace-tolerant prefix match.
        prompt_norm = " ".join(self._prompt.lower().split())
        value_norm = " ".join(value.lower().split())
        return value_norm.startswith(prompt_norm)


# ===================================================================
# Start/end checkers.
# ===================================================================

class EndChecker(Instruction):
    """Response ends with a specific phrase."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._end_phrase: str = ""

    def build_description(self, *, end_phrase: str | None = None, **kw: object) -> str:
        if end_phrase is not None:
            self._end_phrase = end_phrase
        return f"End with: '{self._end_phrase}'"

    def get_instruction_args(self) -> dict | None:
        return {"end_phrase": self._end_phrase} if self._end_phrase else None

    def get_instruction_args_keys(self) -> list[str]:
        return ["end_phrase"]

    def check_following(self, value: str) -> bool:
        # Strip trailing whitespace and quotes.
        stripped = value.rstrip().rstrip('"').rstrip("'").rstrip()
        return stripped.lower().endswith(self._end_phrase.lower())


class QuotationChecker(Instruction):
    """Entire response wrapped in double quotation marks."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Wrap response in double quotation marks"

    def check_following(self, value: str) -> bool:
        stripped = value.strip()
        return stripped.startswith('"') and stripped.endswith('"')


# ===================================================================
# Case checkers.
# ===================================================================

class CapitalWordFrequencyChecker(Instruction):
    """ALL-CAPS words appear < or >= N times."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)
        self._capital_frequency: int = 0
        self._capital_relation: str = "at least"

    def build_description(
        self,
        *,
        capital_frequency: int | None = None,
        capital_relation: str | None = None,
        **kw: object,
    ) -> str:
        if capital_frequency is not None:
            self._capital_frequency = int(capital_frequency)
        if capital_relation is not None:
            self._capital_relation = capital_relation
        return f"ALL-CAPS words {self._capital_relation} {self._capital_frequency}"

    def get_instruction_args(self) -> dict | None:
        return {
            "capital_frequency": self._capital_frequency,
            "capital_relation": self._capital_relation,
        }

    def get_instruction_args_keys(self) -> list[str]:
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value: str) -> bool:
        tokens = _word_tokenize(value)
        cap_count = sum(1 for t in tokens if t.isupper() and t.isalpha())
        return _compare(cap_count, self._capital_relation, self._capital_frequency)


class CapitalLettersEnglishChecker(Instruction):
    """Entire response is English AND all uppercase."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Response must be entirely uppercase English"

    def check_following(self, value: str) -> bool:
        # Check uppercase.
        alpha = "".join(c for c in value if c.isalpha())
        if not alpha or not alpha.isupper():
            return False
        # Check English.
        detected = _detect_language(value)
        if detected is None:
            return True  # Lenient on detection failure.
        return detected == "en"


class LowercaseLettersEnglishChecker(Instruction):
    """Entire response is English AND all lowercase."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Response must be entirely lowercase English"

    def check_following(self, value: str) -> bool:
        alpha = "".join(c for c in value if c.isalpha())
        if not alpha or not alpha.islower():
            return False
        detected = _detect_language(value)
        if detected is None:
            return True
        return detected == "en"


# ===================================================================
# Punctuation checker.
# ===================================================================

class CommaChecker(Instruction):
    """No commas anywhere in the response."""

    def __init__(self, instruction_id: str) -> None:
        super().__init__(instruction_id)

    def build_description(self, **kw: object) -> str:
        return "Do not use any commas"

    def check_following(self, value: str) -> bool:
        return "," not in value


# ===================================================================
# Registry.
# ===================================================================

INSTRUCTION_DICT: dict[str, type[Instruction]] = {
    "keywords:existence": KeywordChecker,
    "keywords:frequency": KeywordFrequencyChecker,
    "keywords:forbidden_words": ForbiddenWords,
    "keywords:letter_frequency": LetterFrequencyChecker,
    "language:response_language": ResponseLanguageChecker,
    "length_constraints:number_sentences": NumberOfSentences,
    "length_constraints:number_paragraphs": ParagraphChecker,
    "length_constraints:number_words": NumberOfWords,
    "length_constraints:nth_paragraph_first_word": ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": PlaceholderChecker,
    "detectable_content:postscript": PostscriptChecker,
    "detectable_format:number_bullet_lists": BulletListChecker,
    "detectable_format:constrained_response": ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": HighlightSectionChecker,
    "detectable_format:multiple_sections": SectionChecker,
    "detectable_format:json_format": JsonFormat,
    "detectable_format:title": TitleChecker,
    "combination:two_responses": TwoResponsesChecker,
    "combination:repeat_prompt": RepeatPromptThenAnswer,
    "startend:end_checker": EndChecker,
    "startend:quotation": QuotationChecker,
    "change_case:capital_word_frequency": CapitalWordFrequencyChecker,
    "change_case:english_capital": CapitalLettersEnglishChecker,
    "change_case:english_lowercase": LowercaseLettersEnglishChecker,
    "punctuation:no_comma": CommaChecker,
}


# ===================================================================
# Evaluation helpers.
# ===================================================================

def _build_checker(instruction_id: str, kwargs: dict, prompt: str = "") -> Instruction:
    """Instantiate and configure a checker for the given instruction."""
    cls = INSTRUCTION_DICT.get(instruction_id)
    if cls is None:
        raise ValueError(f"Unknown instruction ID: {instruction_id!r}")
    checker = cls(instruction_id)
    # Filter None values from kwargs (dataset stores all fields, most are None).
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    checker.build_description(**filtered)
    # Re-inject prompt if the checker needs it (RepeatPromptThenAnswer).
    if "prompt_to_repeat" in checker.get_instruction_args_keys():
        if not filtered.get("prompt_to_repeat") and prompt:
            checker.build_description(prompt_to_repeat=prompt)
    return checker


def evaluate_strict(
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
    prompt: str = "",
) -> list[bool]:
    """Evaluate each instruction strictly against the raw response.

    Returns a list of booleans, one per instruction.
    """
    results: list[bool] = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        checker = _build_checker(inst_id, kw, prompt)
        if response.strip():
            results.append(checker.check_following(response))
        else:
            results.append(False)
    return results
