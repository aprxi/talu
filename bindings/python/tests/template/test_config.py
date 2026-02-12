"""Tests for talu.template.config - Template module configuration."""

import pytest

from talu.exceptions import ValidationError
from talu.template.config import _TemplateConfig, config


class TestTemplateConfig:
    """Test _TemplateConfig class."""

    def test_debug_default_false(self):
        cfg = _TemplateConfig()
        assert cfg.debug is False

    def test_debug_setter_accepts_bool(self):
        cfg = _TemplateConfig()
        cfg.debug = True
        assert cfg.debug is True
        cfg.debug = False
        assert cfg.debug is False

    def test_debug_setter_rejects_non_bool(self):
        cfg = _TemplateConfig()
        with pytest.raises(ValidationError) as exc_info:
            cfg.debug = "true"  # type: ignore[assignment]

        assert exc_info.value.code == "INVALID_ARGUMENT"
        assert "debug must be bool" in str(exc_info.value)

    def test_debug_setter_rejects_int(self):
        cfg = _TemplateConfig()
        with pytest.raises(ValidationError):
            cfg.debug = 1  # type: ignore[assignment]

    def test_repr(self):
        cfg = _TemplateConfig()
        assert repr(cfg) == "TemplateConfig(debug=False)"
        cfg.debug = True
        assert repr(cfg) == "TemplateConfig(debug=True)"


class TestModuleSingleton:
    """Test module-level config singleton."""

    def test_config_is_template_config(self):
        assert isinstance(config, _TemplateConfig)

    def test_config_modifications_persist(self):
        original = config.debug
        try:
            config.debug = True
            assert config.debug is True
        finally:
            config.debug = original
