# TaluDB Storage Walkthroughs

This directory contains in-depth Python walkthroughs for TaluDB storage backends.

## Walkthroughs

| File | Description |
|------|-------------|
| [`talu_storage.py`](talu_storage.py) | Built-in TaluDB storage with `Database` class |
| [`vector_storage.py`](vector_storage.py) | VectorStore for high-throughput embedding storage |

## Quick Start

```bash
cd examples/walkthrough

# TaluDB storage demo
uv run python talu_storage.py --mode demo

# Vector storage demo
uv run python vector_storage.py --mode demo
```

## Architecture Overview

### Chat Storage

TaluDB provides built-in persistence for chat conversations:

```
Zig Memory (:memory:)        TaluDB Storage
=====================================================================
[item finalized]   →         Written to WAL (crash-safe)
[session restore]  ←         Loaded from TaluDB files
```

- **Zig memory** is the source of truth during runtime
- **TaluDB** provides native persistence with automatic WAL writes
- **Session restore** hydrates Zig memory from storage on startup

### Storage Options

1. **Database (built-in)**: Uses TaluDB's native Zig-backed storage
   ```python
   from talu.db import Database
   chat = Chat(storage=Database("talu://./my-db"), session_id="demo")
   ```

2. **VectorStore**: High-throughput embedding storage (separate from chat)
   ```python
   from talu.db import VectorStore
   store = VectorStore("./vectors")
   store.append_batch(ids, vectors, dims=384)
   ```

## CLI Alternative

For command-line workflows, see [`../cli/`](../cli/):

```bash
# Initialize storage
talu db init ./mydb

# Generate with persistence
talu generate --db ./mydb "Hello world"

# List and inspect sessions
talu db list ./mydb
talu db show <session-id> ./mydb
```

The CLI uses the same TaluDB storage format, so databases created by Python or CLI are interoperable.

## Data Files

| File | Description |
|------|-------------|
| `chat.json` | Sample chat export in Open Responses format |
| `vector.json` | Sample vector data |

## See Also

- [`../cli/`](../cli/) - CLI workflow examples
- [`../basics/`](../basics/) - Python API basics
- [TaluDB format documentation](../../env/features/db/) - Internal format details
