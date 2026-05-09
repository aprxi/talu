"""Tests for endpoint discovery helpers."""

import json
import urllib.error
from unittest.mock import patch

import pytest

from talu.exceptions import IOError, ValidationError
from talu.router.remote import check_endpoint, get_model_ids, list_endpoint_models


class _Response:
    def __init__(self, payload: object):
        self._payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_list_endpoint_models_normalizes_url_and_headers() -> None:
    payload = {
        "data": [
            {"id": "model-a", "created": 123, "owned_by": "team"},
            {"id": "model-b"},
            {"object": "model"},
            "invalid",
        ]
    }

    with patch("urllib.request.urlopen", return_value=_Response(payload)) as urlopen:
        models = list_endpoint_models("http://localhost:8000", api_key="token", timeout=3.5)

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://localhost:8000/v1/models"
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") == "Bearer token"
    assert urlopen.call_args.kwargs["timeout"] == 3.5
    assert [m.id for m in models] == ["model-a", "model-b"]
    assert models[0].created == 123
    assert models[0].owned_by == "team"


def test_list_endpoint_models_rejects_non_object_response() -> None:
    with patch("urllib.request.urlopen", return_value=_Response([])):
        with pytest.raises(ValidationError, match="Expected dict response"):
            list_endpoint_models("http://localhost:8000/v1")


def test_list_endpoint_models_wraps_url_errors() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(IOError, match="Failed to connect"):
            list_endpoint_models("http://localhost:8000")


def test_check_endpoint_returns_false_for_discovery_errors() -> None:
    with patch("talu.router.remote.list_endpoint_models", side_effect=ValidationError("bad")):
        assert check_endpoint("http://localhost:8000") is False


def test_get_model_ids_returns_ids_only() -> None:
    payload = {"data": [{"id": "model-a"}, {"id": "model-b"}]}

    with patch("urllib.request.urlopen", return_value=_Response(payload)):
        assert get_model_ids("http://localhost:8000") == ["model-a", "model-b"]
