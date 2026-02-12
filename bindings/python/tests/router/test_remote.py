"""
Unit tests for remote inference utilities.

Tests the HTTP client functions for OpenAI-compatible endpoints,
mocking network requests to avoid real HTTP calls.
"""

import json
import urllib.error
from unittest.mock import patch

import pytest

from talu.exceptions import IOError, ValidationError
from talu.router.remote import (
    RemoteModelInfo,
    check_endpoint,
    get_model_ids,
    list_endpoint_models,
)


class MockResponse:
    """Mock HTTP response object."""

    def __init__(self, data: dict | str | bytes, status: int = 200):
        if isinstance(data, dict):
            self._data = json.dumps(data).encode("utf-8")
        elif isinstance(data, str):
            self._data = data.encode("utf-8")
        else:
            self._data = data
        self.status = status

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class TestRemoteModelInfo:
    """Tests for RemoteModelInfo dataclass."""

    def test_create_with_required_fields(self):
        """RemoteModelInfo can be created with just id."""
        info = RemoteModelInfo(id="test-model")
        assert info.id == "test-model"
        assert info.object == "model"
        assert info.created is None
        assert info.owned_by == ""

    def test_create_with_all_fields(self):
        """RemoteModelInfo can be created with all fields."""
        info = RemoteModelInfo(
            id="test-model",
            object="model",
            created=1234567890,
            owned_by="test-org",
        )
        assert info.id == "test-model"
        assert info.object == "model"
        assert info.created == 1234567890
        assert info.owned_by == "test-org"

    def test_frozen_dataclass(self):
        """RemoteModelInfo is immutable."""
        info = RemoteModelInfo(id="test-model")
        with pytest.raises(AttributeError):
            info.id = "new-id"


class TestListModels:
    """Tests for list_endpoint_models() function."""

    def test_list_endpoint_models_basic(self):
        """list_endpoint_models returns parsed model info."""
        mock_response = MockResponse(
            {
                "data": [
                    {"id": "model-1", "object": "model", "created": 123, "owned_by": "org"},
                    {"id": "model-2"},
                ]
            }
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = list_endpoint_models("http://localhost:8000")

        assert len(models) == 2
        assert models[0].id == "model-1"
        assert models[0].object == "model"
        assert models[0].created == 123
        assert models[0].owned_by == "org"
        assert models[1].id == "model-2"

    def test_list_endpoint_models_url_normalization_adds_v1(self):
        """list_endpoint_models adds /v1 if missing."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            list_endpoint_models("http://localhost:8000")

        # Check the URL was normalized
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v1/models" in req.full_url

    def test_list_endpoint_models_url_normalization_trailing_slash(self):
        """list_endpoint_models handles trailing slashes."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            list_endpoint_models("http://localhost:8000/")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v1/models" in req.full_url

    def test_list_endpoint_models_url_already_has_v1(self):
        """list_endpoint_models doesn't double /v1."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            list_endpoint_models("http://localhost:8000/v1")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "http://localhost:8000/v1/models"

    def test_list_endpoint_models_with_api_key(self):
        """list_endpoint_models includes Authorization header when api_key provided."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            list_endpoint_models("http://localhost:8000", api_key="sk-test-key")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test-key"

    def test_list_endpoint_models_connection_error(self):
        """list_endpoint_models raises IOError on connection failure."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(IOError) as exc_info:
                list_endpoint_models("http://localhost:9999")

        assert "Failed to connect" in str(exc_info.value)
        assert exc_info.value.code == "CONNECTION_ERROR"

    def test_list_endpoint_models_invalid_json(self):
        """list_endpoint_models raises ValidationError on invalid JSON."""
        mock_response = MockResponse(b"not json")

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValidationError) as exc_info:
                list_endpoint_models("http://localhost:8000")

        assert "Invalid JSON" in str(exc_info.value)

    def test_list_endpoint_models_not_dict_response(self):
        """list_endpoint_models raises ValidationError if response is not a dict."""
        mock_response = MockResponse(b"[]")  # JSON array instead of object

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValidationError) as exc_info:
                list_endpoint_models("http://localhost:8000")

        assert "Expected dict" in str(exc_info.value)

    def test_list_endpoint_models_skips_invalid_items(self):
        """list_endpoint_models skips items without id or non-dict items."""
        mock_response = MockResponse(
            {
                "data": [
                    {"id": "valid-model"},
                    {"no_id": "missing"},  # No id field
                    "not a dict",  # Not a dict
                    None,  # None item
                ]
            }
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = list_endpoint_models("http://localhost:8000")

        assert len(models) == 1
        assert models[0].id == "valid-model"

    def test_list_endpoint_models_empty_data(self):
        """list_endpoint_models returns empty list when data is empty."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = list_endpoint_models("http://localhost:8000")

        assert models == []

    def test_list_endpoint_models_missing_data_key(self):
        """list_endpoint_models returns empty list when 'data' key is missing."""
        mock_response = MockResponse({"models": []})

        with patch("urllib.request.urlopen", return_value=mock_response):
            models = list_endpoint_models("http://localhost:8000")

        assert models == []


class TestCheckEndpoint:
    """Tests for check_endpoint() function."""

    def test_check_endpoint_available(self):
        """check_endpoint returns True when endpoint is available."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response):
            assert check_endpoint("http://localhost:8000") is True

    def test_check_endpoint_connection_error(self):
        """check_endpoint returns False on connection error."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert check_endpoint("http://localhost:9999") is False

    def test_check_endpoint_validation_error(self):
        """check_endpoint returns False on invalid response."""
        mock_response = MockResponse(b"not json")

        with patch("urllib.request.urlopen", return_value=mock_response):
            assert check_endpoint("http://localhost:8000") is False

    def test_check_endpoint_timeout(self):
        """check_endpoint returns False on timeout."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            assert check_endpoint("http://localhost:8000") is False

    def test_check_endpoint_with_api_key(self):
        """check_endpoint passes api_key to list_endpoint_models."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            check_endpoint("http://localhost:8000", api_key="sk-test")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test"


class TestGetModelIds:
    """Tests for get_model_ids() function."""

    def test_get_model_ids_returns_ids(self):
        """get_model_ids returns list of model ID strings."""
        mock_response = MockResponse(
            {
                "data": [
                    {"id": "model-1"},
                    {"id": "model-2"},
                    {"id": "model-3"},
                ]
            }
        )

        with patch("urllib.request.urlopen", return_value=mock_response):
            ids = get_model_ids("http://localhost:8000")

        assert ids == ["model-1", "model-2", "model-3"]

    def test_get_model_ids_empty(self):
        """get_model_ids returns empty list when no models."""
        mock_response = MockResponse({"data": []})

        with patch("urllib.request.urlopen", return_value=mock_response):
            ids = get_model_ids("http://localhost:8000")

        assert ids == []

    def test_get_model_ids_with_api_key(self):
        """get_model_ids passes api_key through."""
        mock_response = MockResponse({"data": [{"id": "test"}]})

        with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            get_model_ids("http://localhost:8000", api_key="sk-key")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-key"

    def test_get_model_ids_propagates_errors(self):
        """get_model_ids propagates errors from list_endpoint_models."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(IOError):
                get_model_ids("http://localhost:8000")
