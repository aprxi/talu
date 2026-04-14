"""Smoke tests for template environment module.

Maps to: talu/template/environment.py
"""

from talu.template.environment import TemplateEnvironment


def test_environment_module_exports() -> None:
    env = TemplateEnvironment()
    assert env.strict is True

