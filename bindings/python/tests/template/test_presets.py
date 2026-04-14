"""Smoke tests for template presets module.

Maps to: talu/template/presets.py
"""

from talu.template.presets import PRESETS, preset_names


def test_presets_module_exports() -> None:
    assert isinstance(PRESETS, dict)
    assert "chatml" in PRESETS
    assert "chatml" in preset_names()

