"""Use a remote vLLM, Ollama, or OpenAI-compatible server.

This example shows:
- Checking if a remote endpoint is available
- Listing available models on the server
- Using the flattened API (base_url=, api_key=) for simple configuration
- Using ModelSpec with OpenAICompatibleBackend for advanced configuration
"""

from talu import Client
from talu.router import (
    ModelSpec,
    OpenAICompatibleBackend,
    check_endpoint,
    get_capabilities,
    get_model_ids,
    list_endpoint_models,
)

# Default endpoint (vLLM typically runs on 8000)
ENDPOINT = "http://localhost:8000"

# Check if the server is running
if not check_endpoint(ENDPOINT):
    print(f"Server not available at {ENDPOINT}")
    print("Start a vLLM server with:")
    print("  vllm serve Qwen/Qwen3-4B-Instruct --port 8000")
    print("\nOr an Ollama server:")
    print("  ollama serve")
    exit(1)

print(f"Connected to {ENDPOINT}")

# List available models
print("\nAvailable models:")
models = list_endpoint_models(ENDPOINT)
for m in models:
    print(f"  - {m.id} (owned_by: {m.owned_by})")

# Get just the model IDs
model_ids = get_model_ids(ENDPOINT)
print(f"\nModel IDs: {model_ids}")

# Pick the first available model
if not model_ids:
    print("No models available!")
    exit(1)

model_id = model_ids[0]
print(f"\nUsing model: {model_id}")

# =============================================================================
# Option 1: Flattened API (recommended for simple cases)
# =============================================================================

# Create a Client with base_url - no need to import ModelSpec or backend types!
client = Client(
    model_id,
    base_url=f"{ENDPOINT}/v1",
    timeout_ms=60000,  # 60 second timeout
)
print(f"\nClient created with flattened API: {client}")

# Get capabilities
caps = client.capabilities()
print(f"Capabilities: streaming={caps.streaming}, tools={caps.tool_calling}")

# Clean up
client.close()

# =============================================================================
# Option 2: ModelSpec API (for advanced configuration)
# =============================================================================

# For more complex scenarios (custom headers, multiple backends, etc.),
# use ModelSpec with OpenAICompatibleBackend explicitly
backend = OpenAICompatibleBackend(
    base_url=f"{ENDPOINT}/v1",
    timeout_ms=60000,
    max_retries=2,
)
spec = ModelSpec(ref=model_id, backend=backend)

print(f"\nModelSpec backend: {spec.backend}")

# Get capabilities from spec
caps = get_capabilities(spec)
print(f"Capabilities: streaming={caps.streaming}, tools={caps.tool_calling}")

print("\nRemote inference configured successfully!")
print("Use `list_endpoint_models()` to discover models on any OpenAI-compatible server.")

"""
Topics covered:

* client.ask
* chat.streaming
* model.reuse

Related:

* examples/developers/router/remote_backends.py
"""
