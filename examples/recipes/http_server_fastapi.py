"""
FastAPI Integration - Building an async chat API server with SSE streaming.

Job: Serve chat completions via FastAPI with SSE streaming and session management.
Prereqs: fastapi, uvicorn, sse-starlette
Failure mode: Prints "Setup: pip install fastapi uvicorn sse-starlette" and exits with code 1.

This example shows how to integrate talu with FastAPI to build
a production-ready chat API with:
- Non-blocking async inference
- Server-Sent Events (SSE) for streaming
- Session management for multi-turn conversations
- Proper resource lifecycle management
- Client disconnect handling (cancellation)

Requirements:
    pip install fastapi uvicorn sse-starlette

Run:
    python examples/recipes/http_server_fastapi.py
    uvicorn examples.recipes.http_server_fastapi:app --reload
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from talu import AsyncClient

# FastAPI imports - install with: pip install fastapi uvicorn sse-starlette
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    print("Setup: pip install fastapi uvicorn sse-starlette")
    raise SystemExit(1)


# =============================================================================
# Application State
# =============================================================================

# Global client - initialized on startup, closed on shutdown
client: AsyncClient | None = None

# In-memory session storage (use Redis/database in production)
sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage AsyncClient lifecycle with FastAPI lifespan."""
    global client

    # Startup: Initialize the client
    print("Loading model...")
    client = AsyncClient("Qwen/Qwen3-0.6B")
    print("Model loaded!")

    yield

    # Shutdown: Clean up resources
    if client:
        await client.close()
        print("Model unloaded.")


app = FastAPI(
    title="Talu Chat API",
    description="Async LLM chat API powered by talu",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""

    message: str
    session_id: str | None = None
    system: str | None = None
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""

    session_id: str
    message: str
    tokens_used: int
    finish_reason: str


class CompletionRequest(BaseModel):
    """Request body for one-shot completion."""

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def get_or_create_session(session_id: str | None, system: str | None) -> tuple[str, dict]:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = {
        "chat": client.chat(system=system or "You are a helpful assistant."),
        "system": system,
    }
    sessions[new_id] = session
    return new_id, session


# =============================================================================
# API Endpoints
# =============================================================================


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message and get a response.

    Supports multi-turn conversations via session_id.
    First request creates a new session, subsequent requests continue it.
    """
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    session_id, session = get_or_create_session(request.session_id, request.system)
    chat = session["chat"]

    # Generate response
    response = await chat.send(
        request.message,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return ChatResponse(
        session_id=session_id,
        message=str(response),
        tokens_used=response.usage.total_tokens,
        finish_reason=response.finish_reason,
    )


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Stream a chat response using Server-Sent Events (SSE).

    Returns tokens as they are generated for real-time UI updates.

    Client disconnect handling:
    - When client closes connection, asyncio.CancelledError is raised
    - talu automatically stops generation when the async iterator is cancelled
    - No "zombie" inference continues after client disconnects
    """
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    session_id, session = get_or_create_session(request.session_id, request.system)
    chat = session["chat"]

    async def generate_stream() -> AsyncIterator[dict]:
        """Generate SSE events for streaming response.

        When the client disconnects, EventSourceResponse cancels this generator,
        which propagates to the async for loop. talu's async streaming catches
        the CancelledError and signals the stop flag to halt generation cleanly.
        """
        # Send session ID first
        yield {"event": "session", "data": session_id}

        # Stream tokens - cancellation is handled automatically
        response = await chat(
            request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        async for token in response:
            yield {"event": "token", "data": token}

        # Send completion event with metadata
        yield {
            "event": "done",
            "data": {
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.finish_reason,
            },
        }

    return EventSourceResponse(generate_stream())


@app.post("/complete")
async def complete_endpoint(request: CompletionRequest):
    """
    One-shot completion without session state.

    Use this for single prompts that don't need conversation history.
    Supports streaming with automatic client disconnect handling.
    """
    if client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.stream:
        # Streaming completion with cancellation support
        async def generate_stream() -> AsyncIterator[dict]:
            chat = client.chat()
            response = await chat(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            # Client disconnect cancels this loop, stopping generation
            async for token in response:
                yield {"event": "token", "data": token}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(generate_stream())

    # Non-streaming completion
    response = await client.ask(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "text": str(response),
        "tokens_used": response.usage.total_tokens,
        "finish_reason": response.finish_reason,
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session info and conversation history."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    chat = session["chat"]

    return {
        "session_id": session_id,
        "system": session["system"],
        "message_count": len(chat.messages),
        "messages": chat.messages.to_list(),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": client is not None,
        "active_sessions": len(sessions),
    }


# =============================================================================
# Main - Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting Talu Chat API server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Topics covered:
* chat.session
* chat.streaming
* workflow.end.to.end
"""
