"""Test fixtures for structured output tests."""

import pytest


class MockRouter:
    """Simulates the Zig engine for testing Python logic."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.last_schema = None

    def submit(
        self,
        messages,
        config=None,
        response_format=None,
        **kwargs,
    ):
        """Simulate generation."""
        self.last_schema = response_format

        if response_format is not None:
            return {
                "text": '{"location": "London", "temperature": 15.5}',
                "finish_reason": "stop",
                "tokens": [1, 2, 3],
            }

        return {
            "text": "Normal text response",
            "finish_reason": "stop",
            "tokens": [1, 2, 3],
        }


@pytest.fixture
def mock_router():
    return MockRouter()
