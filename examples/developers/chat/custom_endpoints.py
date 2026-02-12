"""Custom endpoints - Override default API endpoints.

Primary API: talu.Client, talu.router.ModelSpec, talu.router.OpenAICompatibleBackend
Scope: Single

Use OpenAICompatibleBackend with base_url to connect to any server
that exposes an OpenAI-compatible API (vLLM, llama.cpp, Ollama, etc.).
"""

from talu import Client
from talu.router import ModelSpec, OpenAICompatibleBackend

# OpenAI API (default base_url)
openai_spec = ModelSpec(
    ref="gpt-4o",
    backend=OpenAICompatibleBackend(api_key="sk-..."),
)
client = Client(openai_spec)
chat = client.chat()
response = chat("Hello!")
print(response)
client.close()

# Custom endpoint - any OpenAI-compatible server
# Works with vLLM, llama.cpp, Ollama, LiteLLM, etc.
custom_spec = ModelSpec(
    ref="my-model",
    backend=OpenAICompatibleBackend(
        base_url="http://localhost:8000/v1",
    ),
)
client = Client(custom_spec)
chat = client.chat()
response = chat("Hello!")
print(response)
client.close()

"""
Topics covered:
* chat.session
* client.ask
"""
