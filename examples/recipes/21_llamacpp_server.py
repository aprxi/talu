"""Use a llama.cpp server for inference.

Job: Connect to a llama.cpp server and run chat completions.
Prereqs: A running llama.cpp server
Failure mode: Prints "Server not available" and exits with code 1.

This example shows:
- Checking if a llama.cpp server is available
- Listing available models on the server
- Using the flattened API (base_url=) for simple remote inference
- Creating a ModelSpec for llama.cpp inference (advanced)

llama.cpp server is OpenAI-compatible, so we use the same remote inference
utilities as for vLLM, Ollama, and other OpenAI-compatible servers.

Start a llama.cpp server with:
    llama-server -m model.gguf --port 8080

Or download and build from:
    https://github.com/ggml-org/llama.cpp
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

# llama.cpp default port is 8080
ENDPOINT = "http://localhost:8080"

# Check if the server is running
if not check_endpoint(ENDPOINT):
    print(f"Server not available at {ENDPOINT}")
    print("\nStart a llama.cpp server with:")
    print("  llama-server -m model.gguf --port 8080")
    print("\nOr download llama.cpp from:")
    print("  https://github.com/ggml-org/llama.cpp")
    exit(1)

print(f"Connected to llama.cpp server at {ENDPOINT}")

# List available models
# llama.cpp typically reports the loaded model
print("\nAvailable models:")
models = list_endpoint_models(ENDPOINT)
for m in models:
    print(f"  - {m.id} (owned_by: {m.owned_by})")

# Get just the model IDs
model_ids = get_model_ids(ENDPOINT)
print(f"\nModel IDs: {model_ids}")

if not model_ids:
    print("No models loaded!")
    print("Make sure to load a model when starting the server:")
    print("  llama-server -m model.gguf --port 8080")
    exit(1)

# Use the first (usually only) model
model_id = model_ids[0]
print(f"\nUsing model: {model_id}")

# =============================================================================
# Option 1: Flattened API (recommended)
# =============================================================================

# Create a Client with base_url - simple and Pythonic!
client = Client(
    model_id,
    base_url=f"{ENDPOINT}/v1",
    timeout_ms=60000,  # 60 second timeout (llama.cpp can be slow on CPU)
)
print(f"\nClient: {client}")

# Get capabilities
caps = client.capabilities()
print(f"Capabilities: streaming={caps.streaming}, tools={caps.tool_calling}")

# Clean up
client.close()

# =============================================================================
# Option 2: ModelSpec API (for advanced configuration)
# =============================================================================

# For more control, use ModelSpec with OpenAICompatibleBackend
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

# Check the server health (llama.cpp specific endpoint)
import json
import urllib.request

try:
    with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=5) as resp:
        health = json.loads(resp.read().decode("utf-8"))
        print(f"\nServer health: {health.get('status', 'unknown')}")
except Exception as e:
    print(f"\nHealth check failed: {e}")

# Example: Direct HTTP chat completion (using Python stdlib)
print("\n--- Example Chat Completion ---")

payload = {
    "model": model_id,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be brief."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    "max_tokens": 50,
    "temperature": 0.7,
}

req = urllib.request.Request(
    f"{ENDPOINT}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        print(f"Response: {content}")
except Exception as e:
    print(f"Chat failed: {e}")

print("\nllama.cpp server configured successfully!")
print("Use the same remote utilities for vLLM, Ollama, and other OpenAI-compatible servers.")

"""
Topics covered:

* client.ask
* chat.streaming
* model.reuse

Related:

* examples/basics/24_remote_inference.py - Generic remote inference example
* examples/recipes/vllm_proxy_server.py - Multi-backend proxy server
"""
