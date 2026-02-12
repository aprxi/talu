"""
Tests for talu.converter.verification module.

Tests for VerificationResult class and verify() function.
"""

import pytest

from talu.converter import VerificationResult, verify


class TestVerificationResult:
    """Tests for VerificationResult class."""

    def test_success_result(self):
        """VerificationResult can represent success."""
        result = VerificationResult(
            success=True,
            model_path="/path/to/model",
            output="Paris",
            tokens_generated=5,
        )

        assert result.success is True
        assert result.model_path == "/path/to/model"
        assert result.output == "Paris"
        assert result.tokens_generated == 5
        assert result.error is None

    def test_failure_result(self):
        """VerificationResult can represent failure."""
        result = VerificationResult(
            success=False,
            model_path="/path/to/model",
            error="Model file not found",
        )

        assert result.success is False
        assert result.error == "Model file not found"
        assert result.output == ""
        assert result.tokens_generated == 0

    def test_bool_true_on_success(self):
        """bool(result) returns True on success."""
        result = VerificationResult(success=True, model_path="/test")

        assert bool(result) is True
        assert result  # Can use in if statement

    def test_bool_false_on_failure(self):
        """bool(result) returns False on failure."""
        result = VerificationResult(success=False, model_path="/test", error="Failed")

        assert bool(result) is False
        assert not result  # Can use in if statement

    def test_repr_success(self):
        """repr() shows success info."""
        result = VerificationResult(
            success=True,
            model_path="/test",
            tokens_generated=10,
        )

        repr_str = repr(result)

        assert "success=True" in repr_str
        assert "tokens=10" in repr_str

    def test_repr_failure(self):
        """repr() shows error info on failure."""
        result = VerificationResult(
            success=False,
            model_path="/test",
            error="Timeout",
        )

        repr_str = repr(result)

        assert "success=False" in repr_str
        assert "Timeout" in repr_str

    def test_slots(self):
        """VerificationResult uses __slots__ for efficiency."""
        result = VerificationResult(success=True, model_path="/test")

        # __slots__ prevents __dict__
        assert not hasattr(result, "__dict__")

    def test_default_values(self):
        """VerificationResult has sensible defaults."""
        result = VerificationResult(success=True, model_path="/test")

        assert result.output == ""
        assert result.tokens_generated == 0
        assert result.error is None


class TestVerifyModel:
    """Tests for verify() function."""

    def test_verify_nonexistent_model(self):
        """verify() returns failure for nonexistent model."""
        result = verify("/nonexistent/model/path")

        assert result.success is False
        assert result.error is not None
        assert result.model_path == "/nonexistent/model/path"

    def test_verify_returns_verification_result(self):
        """verify() returns VerificationResult."""
        result = verify("/nonexistent/model")

        assert isinstance(result, VerificationResult)

    def test_verify_accepts_custom_prompt(self):
        """verify() accepts prompt parameter."""
        # This will fail because model doesn't exist, but validates signature
        result = verify("/nonexistent/model", prompt="2 + 2 =")

        assert isinstance(result, VerificationResult)
        assert result.success is False

    def test_verify_accepts_max_tokens(self):
        """verify() accepts max_tokens parameter."""
        result = verify("/nonexistent/model", max_tokens=10)

        assert isinstance(result, VerificationResult)

    def test_verify_with_all_params(self):
        """verify() accepts all parameters together."""
        result = verify(
            "/nonexistent/model",
            prompt="Hello",
            max_tokens=3,
        )

        assert isinstance(result, VerificationResult)


@pytest.mark.requires_model
class TestVerifyModelIntegration:
    """Integration tests for verify() requiring a real model."""

    def test_verify_valid_model(self, test_model_path):
        """verify() succeeds for valid model."""
        result = verify(test_model_path, max_tokens=3)

        assert result.success is True
        assert result.tokens_generated >= 0

    def test_verify_with_custom_prompt(self, test_model_path):
        """verify() uses custom prompt."""
        result = verify(test_model_path, prompt="2 + 2 =", max_tokens=3)

        assert result.success is True
        assert result.tokens_generated >= 0

    def test_verify_result_can_be_used_in_condition(self, test_model_path):
        """VerificationResult works in if statements."""
        result = verify(test_model_path, max_tokens=3)

        if result:
            passed = True
        else:
            passed = False

        assert passed is True
