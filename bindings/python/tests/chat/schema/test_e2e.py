"""End-to-end tests with mocked Zig router."""

from unittest.mock import patch

from pydantic import BaseModel

from talu import Chat
from talu.chat.response import Response


class ToolCall(BaseModel):
    name: str
    arguments: dict


class Weather(BaseModel):
    location: str
    temperature: float


class TestPromptInjection:
    """Test that schema is injected into prompts."""

    def test_schema_injected_into_system(self):
        """response_format should inject TypeScript into system message."""
        chat = Chat(system="You are helpful.")

        messages = chat._prepare_messages(
            "What's the weather?",
            response_format=Weather,
            inject_schema_prompt=True,
        )

        system_content = messages[0]["content"]
        assert "interface Response {" in system_content
        assert "location: string;" in system_content
        assert "temperature: number;" in system_content

    def test_thinking_instruction_added(self):
        """allow_thinking should add think instruction."""
        chat = Chat(system="You are helpful.")

        messages = chat._prepare_messages(
            "What tool?",
            response_format=ToolCall,
            allow_thinking=True,
            inject_schema_prompt=True,
        )

        system_content = messages[0]["content"]
        assert "<think>" in system_content
        assert "Then output your response as JSON" in system_content

    def test_injection_opt_out(self):
        """inject_schema_prompt=False should not modify prompt."""
        chat = Chat(system="You know the schema already.")

        messages = chat._prepare_messages(
            "Return user data",
            response_format=Weather,
            inject_schema_prompt=False,
        )

        system_content = messages[0]["content"]
        assert "interface Response" not in system_content
        assert system_content == "You know the schema already."


class TestMockedGeneration:
    """Test full flow with mocked router."""

    @patch("talu.router.router.Router.submit")
    def test_structured_response(self, mock_submit):
        """Test that structured response is properly wrapped."""
        mock_submit.return_value = {
            "text": '{"location": "London", "temperature": 15.5}',
            "finish_reason": "stop",
            "tokens": [1, 2, 3],
        }

        resp = Response(
            text='{"location": "London", "temperature": 15.5}',
            finish_reason="stop",
            _response_format=Weather,
        )

        weather = resp.parsed
        assert weather.location == "London"
        assert weather.temperature == 15.5
