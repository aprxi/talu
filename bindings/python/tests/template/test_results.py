"""Smoke tests for template results module.

Maps to: talu/template/results.py
"""

from talu.template.results import DebugResult, DebugSpan, ValidationResult


def test_results_module_exports() -> None:
    assert ValidationResult.__name__ == "ValidationResult"
    assert DebugSpan.__name__ == "DebugSpan"
    assert DebugResult.__name__ == "DebugResult"

