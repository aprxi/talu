"""Mixed engines - Combine native and API-based models.

Primary API: talu.Client, talu.router.ModelSpec, talu.router.OpenAICompatibleBackend
Scope: Single

ModelSpec allows explicit backend configuration for each model.
Local models omit backend (uses native inference).
Remote APIs use OpenAICompatibleBackend with base_url and/or api_key.
"""

from talu import Client
from talu.router import ModelSpec, OpenAICompatibleBackend

# Multiple backends in one client using ModelSpec
client = Client([
    # Native local inference (no backend specified)
    ModelSpec(ref="Qwen/Qwen3-0.6B"),

    # OpenAI API
    ModelSpec(
        ref="gpt-4o",
        backend=OpenAICompatibleBackend(api_key="sk-..."),
    ),

    # Local server with OpenAI-compatible API (vLLM, llama.cpp, etc.)
    ModelSpec(
        ref="local-server",
        backend=OpenAICompatibleBackend(
            base_url="http://localhost:8000/v1",
        ),
    ),
])

chat = client.chat(system="You are helpful.")

# Use native backend (default, first in list)
response = chat("What is 2+2?")
print(f"Native: {response}")

# Switch to OpenAI API
response = chat("What is 2+2?", model="gpt-4o")
print(f"gpt-4o: {response}")

# Switch to local server
response = chat("What is 2+2?", model="local-server")
print(f"local-server: {response}")

# Continue conversation on a different model
response = response.append("Why?", model="gpt-4o")
print(f"gpt-4o: {response}")

client.close()

"""
Topics covered:
* chat.session
* client.shared.model
"""
