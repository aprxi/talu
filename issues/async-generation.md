# Async Generation Support

## Summary

Add async/await support for text generation to enable concurrent inference in web frameworks (FastAPI, Django, Starlette) and async applications.

## Motivation

Currently, all talu generation APIs are synchronous and blocking:

```python
# Current API - blocks the event loop
session = Chat("Qwen/Qwen3-0.6B")
response = session("What is 2+2?")  # Blocks until complete
```

This is problematic for:

1. **Web Frameworks**: FastAPI, Django 4+, Starlette, and Quart all use async. A blocking call ties up the entire worker.
2. **Concurrent Requests**: A server handling 100 concurrent users would need 100 threads/processes.
3. **Cancellation**: No way to cancel a long-running generation mid-stream.
4. **Backpressure**: No way to slow down generation if the client disconnects.

## Proposed API

### 1. Async Streaming (Primary Use Case)

```python
from talu import Chat

async with Chat("Qwen/Qwen3-0.6B") as session:
    async for chunk in session.stream_async("What is 2+2?", max_tokens=100):
        print(chunk, end="", flush=True)
```

### 2. Async Collect (Convenience)

```python
async with Chat("Qwen/Qwen3-0.6B") as session:
    result = await session.stream_async("What is 2+2?", max_tokens=100).collect()
    print(result.text)
```

### 3. Top-Level Async Functions

```python
import talu

# Streaming
async for chunk in talu.stream_async("Qwen/Qwen3-0.6B", "Hello"):
    print(chunk)

# One-shot
response = await talu.generate_async("Qwen/Qwen3-0.6B", "Hello")
```

### 4. FastAPI Integration Example

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from talu import Chat

app = FastAPI()
session = Chat("Qwen/Qwen3-0.6B")

@app.post("/chat")
async def chat(prompt: str):
    async def generate():
        async for chunk in session.stream_async(prompt, max_tokens=500):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
```

## Implementation Strategy

### Option A: Thread Pool Executor (Recommended for v1)

Wrap synchronous generation in `asyncio.to_thread()` or `loop.run_in_executor()`.

**Pros:**
- Minimal changes to existing code
- Works immediately with current Zig backend
- No changes to C API required

**Cons:**
- Still uses threads under the hood (not true async I/O)
- Thread pool size limits concurrency

```python
# Implementation sketch
import asyncio
from typing import AsyncIterator

class AsyncGenerationStream:
    def __init__(self, sync_stream):
        self._sync_stream = sync_stream
        self._queue = asyncio.Queue()
        self._task = None

    async def __aenter__(self):
        loop = asyncio.get_event_loop()
        self._task = loop.run_in_executor(None, self._producer)
        return self

    def _producer(self):
        """Runs in thread pool, pushes chunks to queue."""
        try:
            for chunk in self._sync_stream:
                asyncio.run_coroutine_threadsafe(
                    self._queue.put(chunk),
                    asyncio.get_event_loop()
                )
        finally:
            asyncio.run_coroutine_threadsafe(
                self._queue.put(None),  # Sentinel
                asyncio.get_event_loop()
            )

    async def __anext__(self):
        chunk = await self._queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk
```

### Option B: Native Async in Zig (Future)

Modify the Zig backend to use non-blocking I/O and integrate with Python's event loop via file descriptors or callbacks.

**Pros:**
- True async (no thread overhead)
- Better resource utilization
- Enables cancellation at the Zig level

**Cons:**
- Significant Zig changes required
- Complex integration with Python event loop
- May require platform-specific code (epoll/kqueue)

### Option C: Callback-Based API (Alternative)

Expose a callback-based API from Zig that Python wraps in async.

```c
// C API addition
typedef void (*talu_chunk_callback)(const char* chunk, size_t len, void* userdata);

int talu_generate_with_callback(
    talu_session_t session,
    const char* prompt,
    talu_generate_options_t options,
    talu_chunk_callback callback,
    void* userdata
);
```

**Pros:**
- Clean separation of concerns
- Callback can post to any async framework

**Cons:**
- Requires C API changes
- Callback management complexity in Python

## Recommended Implementation Plan

### Phase 1: Thread Pool Wrapper (Week 1)

1. Add `stream_async()` method to `Chat` using `asyncio.to_thread()`
2. Add `AsyncGenerationStream` class with `__aiter__`/`__anext__`
3. Add `collect()` async method for convenience
4. Add top-level `talu.stream_async()` and `talu.generate_async()`
5. Document thread pool size tuning

### Phase 2: Cancellation Support (Week 2)

1. Add `cancel()` method to `AsyncGenerationStream`
2. Propagate cancellation to Zig via a flag in session state
3. Handle `asyncio.CancelledError` properly
4. Add timeout parameter: `stream_async(..., timeout=30.0)`

### Phase 3: Context Manager Async (Week 2)

1. Add `async with Chat(...)` support via `__aenter__`/`__aexit__`
2. Ensure proper cleanup on cancellation
3. Add connection pooling for model sessions

### Phase 4: Native Async (Future)

1. Evaluate whether thread pool is sufficient for production
2. If not, implement callback-based Zig API
3. Integrate with `asyncio` via `add_reader()`/`add_writer()`

## API Reference

### Chat Methods

```python
class Chat:
    def stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        system: str | None = None,
        chat: bool = True,
        timeout: float | None = None,
    ) -> AsyncGenerationStream:
        """
        Stream generation asynchronously.

        Returns an async iterator that yields text chunks.
        Use .collect() to get the full GenerationOutput.

        Example:
            async for chunk in session.stream_async("Hello"):
                print(chunk, end="")
        """
        ...

    async def generate_async(
        self,
        prompt: str,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate text asynchronously (convenience wrapper).

        Equivalent to: await session.stream_async(prompt, **kwargs).collect()
        """
        ...
```

### AsyncGenerationStream

```python
class AsyncGenerationStream:
    """Async iterator for streaming generation."""

    def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over text chunks."""
        ...

    async def __anext__(self) -> str:
        """Get next chunk."""
        ...

    async def collect(self) -> GenerationOutput:
        """Consume stream and return complete output."""
        ...

    def cancel(self) -> None:
        """Cancel generation (best-effort)."""
        ...

    @property
    def cancelled(self) -> bool:
        """Whether generation was cancelled."""
        ...
```

### Top-Level Functions

```python
async def stream_async(
    model: str,
    prompt: str,
    **kwargs,
) -> AsyncGenerationStream:
    """
    Stream generation from a model asynchronously.

    Creates or reuses a cached Chat.
    """
    ...

async def generate_async(
    model: str,
    prompt: str,
    **kwargs,
) -> GenerationOutput:
    """
    Generate text from a model asynchronously.

    Convenience wrapper around stream_async().collect().
    """
    ...
```

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_stream_async_yields_chunks(synthetic_session):
    chunks = []
    async for chunk in synthetic_session.stream_async("Hello", max_tokens=5):
        chunks.append(chunk)
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_stream_async_collect(synthetic_session):
    result = await synthetic_session.stream_async("Hello", max_tokens=5).collect()
    assert result.text
    assert result.generated_len <= 5

@pytest.mark.asyncio
async def test_stream_async_cancellation(synthetic_session):
    stream = synthetic_session.stream_async("Tell me a long story", max_tokens=1000)
    async for i, chunk in enumerate(stream):
        if i > 5:
            stream.cancel()
            break
    assert stream.cancelled

@pytest.mark.asyncio
async def test_concurrent_streams(synthetic_model):
    async with Chat(str(synthetic_model.path)) as session:
        streams = [
            session.stream_async(f"Count to {i}", max_tokens=10)
            for i in range(5)
        ]
        results = await asyncio.gather(*[s.collect() for s in streams])
        assert len(results) == 5
```

### Integration Tests

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_fastapi_streaming():
    """Test FastAPI integration."""
    from fastapi.testclient import TestClient
    # ... test streaming endpoint
```

## Performance Considerations

1. **Thread Pool Size**: Default `ThreadPoolExecutor` uses `min(32, cpu_count + 4)` threads. For inference-heavy workloads, consider tuning:
   ```python
   import asyncio
   import concurrent.futures

   executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
   asyncio.get_event_loop().set_default_executor(executor)
   ```

2. **Memory**: Each concurrent generation holds model activations. Monitor memory with many concurrent streams.

3. **Batching**: Future optimization - batch multiple async requests into single forward passes.

## Migration Guide

### From Sync to Async

```python
# Before (sync)
session = Chat("model")
for chunk in session.send("Hello"):
    print(chunk)

# After (async)
async with Chat("model") as session:
    async for chunk in session.send_async("Hello"):
        print(chunk)
```

### Gradual Migration

The sync API remains unchanged. Async is opt-in:

```python
# Sync still works
response = session("Hello")

# Async is additive
response = await session.generate_async("Hello")
```

## Dependencies

- Python 3.10+ (for `asyncio.to_thread`)
- No new external dependencies
- Optional: `pytest-asyncio` for testing

## Open Questions

1. **Should `Chat` itself be async-constructable?**
   - Model loading is slow - could benefit from async
   - But complicates the API significantly

2. **Connection pooling?**
   - Multiple sessions for the same model?
   - Automatic session recycling?

3. **Backpressure handling?**
   - What if consumer is slower than producer?
   - Queue size limits? Drop policy?

4. **Error propagation?**
   - How to surface Zig-level errors in async context?
   - Should cancellation raise or return partial result?

## References

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [FastAPI Streaming Responses](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [PEP 525 - Asynchronous Generators](https://peps.python.org/pep-0525/)
- [anyio - Async compatibility layer](https://anyio.readthedocs.io/)
