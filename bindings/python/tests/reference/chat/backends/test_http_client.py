"""
Tests for talu/chat/backends/remote.py.

Tests for remote inference utilities (OpenAI-compatible endpoints).
"""

import json
from unittest.mock import Mock, patch

import pytest

from talu.router.remote import RemoteModelInfo, check_endpoint, get_model_ids, list_endpoint_models

# =============================================================================
# RemoteModelInfo Tests
# =============================================================================


class TestRemoteModelInfo:
    """Tests for RemoteModelInfo dataclass."""

    def test_minimal_info(self):
        """RemoteModelInfo with minimal required fields."""
        info = RemoteModelInfo(id="model-id")
        assert info.id == "model-id"
        assert info.object == "model"
        assert info.created is None
        assert info.owned_by == ""

    def test_full_info(self):
        """RemoteModelInfo with all fields."""
        info = RemoteModelInfo(
            id="model-id",
            object="model",
            created=1234567890,
            owned_by="organization",
        )
        assert info.id == "model-id"
        assert info.object == "model"
        assert info.created == 1234567890
        assert info.owned_by == "organization"

    def test_frozen_dataclass(self):
        """RemoteModelInfo is frozen (immutable)."""
        info = RemoteModelInfo(id="model-id")
        # Creating a new instance with same id shows immutability
        info2 = RemoteModelInfo(id="model-id")
        assert info == info2
        # Frozen dataclass with slots cannot be modified directly
        # (would require complex __setattr__ bypass)

    def test_custom_object_field(self):
        """Custom object field can be set."""
        info = RemoteModelInfo(id="model-id", object="custom-object")
        assert info.object == "custom-object"


# =============================================================================
# list_endpoint_models Tests
# =============================================================================


class TestListModels:
    """Tests for list_endpoint_models function."""

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_successful_list(self, mock_urlopen):
        """Successfully list models from endpoint."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": "model-1", "object": "model", "owned_by": "org1"},
                    {"id": "model-2", "object": "model", "owned_by": "org2"},
                ],
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        models = list_endpoint_models("http://localhost:8000")

        assert len(models) == 2
        assert models[0].id == "model-1"
        assert models[0].owned_by == "org1"
        assert models[1].id == "model-2"
        assert models[1].owned_by == "org2"

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_url_normalization_adds_v1(self, mock_urlopen):
        """URL normalization adds /v1 if missing."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        list_endpoint_models("http://localhost:8000")

        # Should have added /v1
        call_args = mock_urlopen.call_args[0][0]
        assert "/v1/models" in call_args.full_url

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_url_with_v1_not_duplicated(self, mock_urlopen):
        """URL already containing /v1 is not modified."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        list_endpoint_models("http://localhost:8000/v1")

        # Should not duplicate /v1
        call_args = mock_urlopen.call_args[0][0]
        url = call_args.full_url
        assert url.count("/v1") == 1

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_api_key_in_headers(self, mock_urlopen):
        """API key is included in headers."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        list_endpoint_models("http://localhost:8000", api_key="test-key")

        call_args = mock_urlopen.call_args[0][0]
        assert "Authorization" in call_args.headers
        assert call_args.headers["Authorization"] == "Bearer test-key"

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_no_api_key_no_auth_header(self, mock_urlopen):
        """No API key means no Authorization header."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        list_endpoint_models("http://localhost:8000")

        call_args = mock_urlopen.call_args[0][0]
        assert "Authorization" not in call_args.headers

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_connection_error_raises_io_error(self, mock_urlopen):
        """Connection errors raise IOError."""
        import urllib.error

        from talu.exceptions import IOError as TaluIOError

        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        with pytest.raises(TaluIOError, match="Failed to connect"):
            list_endpoint_models("http://localhost:8000")

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_invalid_json_raises_validation_error(self, mock_urlopen):
        """Invalid JSON response raises ValidationError."""
        from talu.exceptions import ValidationError

        mock_response = Mock()
        mock_response.read.return_value = b"not json"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with pytest.raises(ValidationError, match="Invalid JSON"):
            list_endpoint_models("http://localhost:8000")

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_non_dict_response_raises_validation_error(self, mock_urlopen):
        """Non-dict response raises ValidationError."""
        from talu.exceptions import ValidationError

        mock_response = Mock()
        mock_response.read.return_value = json.dumps([1, 2, 3]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with pytest.raises(ValidationError, match="Expected dict"):
            list_endpoint_models("http://localhost:8000")

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_missing_data_field_returns_empty_list(self, mock_urlopen):
        """Missing 'data' field returns empty list."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        models = list_endpoint_models("http://localhost:8000")
        assert models == []

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_model_without_id_field_skipped(self, mock_urlopen):
        """Models without 'id' field are skipped."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(
            {
                "data": [
                    {"id": "valid-model"},
                    {"no_id": True},
                    {"id": "another-valid-model"},
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        models = list_endpoint_models("http://localhost:8000")
        assert len(models) == 2
        assert models[0].id == "valid-model"
        assert models[1].id == "another-valid-model"

    @patch("talu.router.remote.urllib.request.urlopen")
    def test_custom_timeout(self, mock_urlopen):
        """Custom timeout is passed to urlopen."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        list_endpoint_models("http://localhost:8000", timeout=30.0)

        # Check that timeout was passed
        call_kwargs = mock_urlopen.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 30.0


# =============================================================================
# check_endpoint Tests
# =============================================================================


class TestCheckEndpoint:
    """Tests for check_endpoint function."""

    @patch("talu.router.remote.list_endpoint_models")
    def test_available_endpoint_returns_true(self, mock_list_endpoint_models):
        """Available endpoint returns True."""
        mock_list_endpoint_models.return_value = []

        result = check_endpoint("http://localhost:8000")

        assert result is True

    @patch("talu.router.remote.list_endpoint_models")
    def test_connection_error_returns_false(self, mock_list_endpoint_models):
        """Connection error returns False."""
        from talu.exceptions import IOError as TaluIOError

        mock_list_endpoint_models.side_effect = TaluIOError("Connection failed")

        result = check_endpoint("http://localhost:8000")

        assert result is False

    @patch("talu.router.remote.list_endpoint_models")
    def test_invalid_json_returns_false(self, mock_list_endpoint_models):
        """Invalid JSON response returns False."""
        from talu.exceptions import ValidationError

        mock_list_endpoint_models.side_effect = ValidationError("Invalid JSON")

        result = check_endpoint("http://localhost:8000")

        assert result is False

    @patch("talu.router.remote.list_endpoint_models")
    def test_timeout_returns_false(self, mock_list_endpoint_models):
        """Timeout returns False."""
        mock_list_endpoint_models.side_effect = TimeoutError()

        result = check_endpoint("http://localhost:8000")

        assert result is False

    @patch("talu.router.remote.list_endpoint_models")
    def test_custom_timeout_passed_to_list_endpoint_models(self, mock_list_endpoint_models):
        """Custom timeout is passed to list_endpoint_models."""
        mock_list_endpoint_models.return_value = []

        check_endpoint("http://localhost:8000", timeout=15.0)

        mock_list_endpoint_models.assert_called_once_with("http://localhost:8000", None, 15.0)


# =============================================================================
# get_model_ids Tests
# =============================================================================


class TestGetModelIds:
    """Tests for get_model_ids function."""

    @patch("talu.router.remote.list_endpoint_models")
    def test_returns_only_ids(self, mock_list_endpoint_models):
        """Returns only model ID strings."""
        mock_list_endpoint_models.return_value = [
            RemoteModelInfo(id="model-1", owned_by="org1"),
            RemoteModelInfo(id="model-2", owned_by="org2"),
            RemoteModelInfo(id="model-3", owned_by="org3"),
        ]

        ids = get_model_ids("http://localhost:8000")

        assert ids == ["model-1", "model-2", "model-3"]

    @patch("talu.router.remote.list_endpoint_models")
    def test_empty_list(self, mock_list_endpoint_models):
        """Empty list returns empty list of IDs."""
        mock_list_endpoint_models.return_value = []

        ids = get_model_ids("http://localhost:8000")

        assert ids == []

    @patch("talu.router.remote.list_endpoint_models")
    def test_single_model(self, mock_list_endpoint_models):
        """Single model returns list with one ID."""
        mock_list_endpoint_models.return_value = [RemoteModelInfo(id="only-model")]

        ids = get_model_ids("http://localhost:8000")

        assert ids == ["only-model"]

    @patch("talu.router.remote.list_endpoint_models")
    def test_passes_api_key(self, mock_list_endpoint_models):
        """API key is passed to list_endpoint_models."""
        mock_list_endpoint_models.return_value = []

        get_model_ids("http://localhost:8000", api_key="secret-key")

        mock_list_endpoint_models.assert_called_once_with(
            "http://localhost:8000", "secret-key", 10.0
        )

    @patch("talu.router.remote.list_endpoint_models")
    def test_passes_timeout(self, mock_list_endpoint_models):
        """Custom timeout is passed to list_endpoint_models."""
        mock_list_endpoint_models.return_value = []

        get_model_ids("http://localhost:8000", timeout=20.0)

        mock_list_endpoint_models.assert_called_once_with("http://localhost:8000", None, 20.0)
