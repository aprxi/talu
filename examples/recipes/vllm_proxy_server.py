"""
vLLM Proxy Server - Aggregate multiple vLLM instances behind a single API.

Job: Proxy chat completions to one or more vLLM servers with load balancing.
Prereqs: fastapi, uvicorn, httpx
Failure mode: Prints "Setup: pip install fastapi uvicorn httpx" and exits with code 1.

This example shows how to:
- Discover models from multiple vLLM/Ollama servers
- Route requests to the appropriate backend
- Expose a unified OpenAI-compatible API
- Handle failover between servers

Use cases:
- Aggregate multiple GPU servers
- Add a caching layer in front of vLLM
- Route different models to different servers
- Add authentication/rate limiting

Requirements:
    pip install fastapi uvicorn httpx

Run:
    python examples/recipes/vllm_proxy_server.py

    Note: Start your vLLM servers first:
    vllm serve Qwen/Qwen3-4B --port 8001
    vllm serve Llama-3-8B --port 8002
"""

from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from talu.router import check_endpoint, list_endpoint_models

# FastAPI imports
try:
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    print("Setup: pip install fastapi uvicorn httpx")
    raise SystemExit(1)


# =============================================================================
# Configuration
# =============================================================================

# Configure your backend servers here
BACKEND_SERVERS = [
    os.environ.get("VLLM_SERVER_1", "http://localhost:8000"),
    os.environ.get("VLLM_SERVER_2", "http://localhost:8001"),
]

# Timeout for backend requests (seconds)
REQUEST_TIMEOUT = 120.0


# =============================================================================
# Backend Registry
# =============================================================================


@dataclass
class BackendServer:
    """A backend vLLM/Ollama server."""

    url: str
    models: list[str] = field(default_factory=list)
    available: bool = False


class BackendRegistry:
    """Registry of backend servers and their models."""

    def __init__(self, server_urls: list[str]):
        self.servers = [BackendServer(url=url) for url in server_urls]
        self._model_to_server: dict[str, BackendServer] = {}

    def refresh(self) -> None:
        """Refresh the list of available models from all servers."""
        self._model_to_server.clear()

        for server in self.servers:
            try:
                if check_endpoint(server.url, timeout=5.0):
                    server.available = True
                    server.models = [m.id for m in list_endpoint_models(server.url)]
                    for model_id in server.models:
                        # First server wins for duplicate models
                        if model_id not in self._model_to_server:
                            self._model_to_server[model_id] = server
                else:
                    server.available = False
                    server.models = []
            except Exception:
                server.available = False
                server.models = []

    def get_server_for_model(self, model_id: str) -> BackendServer | None:
        """Get the backend server that has a specific model."""
        return self._model_to_server.get(model_id)

    def get_all_models(self) -> list[dict]:
        """Get all available models across all servers."""
        models = []
        for model_id, server in self._model_to_server.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": server.url,
            })
        return models


# Global registry
registry = BackendRegistry(BACKEND_SERVERS)


# =============================================================================
# FastAPI Application
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Refresh backend registry on startup."""
    print("Discovering backend servers...")
    registry.refresh()

    available = [s for s in registry.servers if s.available]
    print(f"Found {len(available)} available server(s)")
    for server in available:
        print(f"  - {server.url}: {len(server.models)} model(s)")

    all_models = registry.get_all_models()
    print(f"Total models available: {len(all_models)}")
    for m in all_models:
        print(f"  - {m['id']}")

    yield

    print("Shutting down proxy server")


app = FastAPI(
    title="vLLM Proxy Server",
    description="Proxy for multiple vLLM/Ollama servers",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/v1/models")
async def list_models_handler():
    """List all available models across all backend servers."""
    return {"object": "list", "data": registry.get_all_models()}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Proxy chat completion requests to the appropriate backend.

    Routes based on the model ID to the server that has that model.
    """
    # Find the backend server for this model
    server = registry.get_server_for_model(request.model)
    if server is None:
        # Try refreshing the registry
        registry.refresh()
        server = registry.get_server_for_model(request.model)

    if server is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found on any backend server",
        )

    # Build the request payload
    payload = request.model_dump(exclude_none=True)

    # Proxy to the backend
    backend_url = f"{server.url}/v1/chat/completions"

    if request.stream:
        # Streaming response - proxy the SSE stream
        async def stream_response() -> AsyncIterator[bytes]:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    backend_url,
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )

    # Non-streaming response
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(backend_url, json=payload)
        response.raise_for_status()
        return response.json()


@app.post("/v1/completions")
async def completions(request: Request):
    """
    Proxy legacy completion requests.

    Note: Not all vLLM servers support /v1/completions.
    """
    body = await request.json()
    model_id = body.get("model")

    server = registry.get_server_for_model(model_id)
    if server is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    backend_url = f"{server.url}/v1/completions"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(backend_url, json=body)
        response.raise_for_status()
        return response.json()


@app.get("/health")
async def health_check():
    """Health check with backend status."""
    return {
        "status": "healthy",
        "backends": [
            {
                "url": s.url,
                "available": s.available,
                "models": len(s.models),
            }
            for s in registry.servers
        ],
        "total_models": len(registry.get_all_models()),
    }


@app.post("/refresh")
async def refresh_backends():
    """Manually refresh the backend registry."""
    registry.refresh()
    return {
        "status": "refreshed",
        "backends": [
            {
                "url": s.url,
                "available": s.available,
                "models": s.models,
            }
            for s in registry.servers
        ],
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting vLLM Proxy Server...")
    print(f"Backend servers: {BACKEND_SERVERS}")
    print("API docs available at: http://localhost:9000/docs")
    uvicorn.run(app, host="0.0.0.0", port=9000)

"""
Topics covered:
* client.ask
* workflow.end.to.end
* chat.streaming
"""
