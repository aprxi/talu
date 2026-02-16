# Talu

**Full-Stack Inference for Local AI.**

Talu is a local-first inference engine built from scratch in Zig — no PyTorch, Transformers, or other heavyweight runtime dependencies. Download a model from HuggingFace, optionally quantize it, and run it from the command line, Python, or the built-in web interface.

It provides structured outputs, multi-turn sessions with persistent profiles, embeddings with vector search, and tool calling. While optimized for local models, talu can also route requests to remote OpenAI-compatible endpoints using the same API contracts.

Distributed as a single binary, talu exposes thin bindings over a shared core runtime. The Web UI runs on the same server APIs and is designed for plugin-based extension.

- **CLI** — download, quantize, chat, serve, and inspect models
- **Python API** — sync and async, multi-turn chat, streaming, embeddings, tool calling, hooks
- **HTTP server** — built-in chat UI, OpenResponses-compatible API, document storage, full-text search
- **Web UI** — extensible via plugins (document upload, analysis, and more planned)
- **Structured output** — precompiled grammars with streaming validation
- **Quantization** — built-in 4-bit and 8-bit grouped affine schemes
- **Backends** — CPU and Metal (CUDA planned); local models and remote OpenAI-compatible endpoints

## Install

```bash
pip install talu
```

Requires Python 3.10+.

## Quick Start (CLI)

### Download and list cached models

```bash
talu get LiquidAI/LFM2-350M
talu ls
```

### Ask a question directly with `-m`

```bash
talu ask -m LiquidAI/LFM2-350M "What is 2+2?"
```

### Set a default model so `-m` is optional

```bash
talu set LiquidAI/LFM2-350M
talu set show
talu ask "Tell me a short joke."
```

### Use implicit `ask` mode (stdin)

When no subcommand is provided, talu implicitly runs `ask`.

```bash
# Equivalent to: talu ask -m LiquidAI/LFM2-350M "hello"
echo "hello" | talu -m LiquidAI/LFM2-350M

# Equivalent to: talu ask "hello"
echo "hello" | talu
```

For multimodal models, piped image bytes are treated as an image input. If no
text prompt is provided, talu uses `Describe this image.` automatically.

```bash
cat test.jpeg | talu ask -m Qwen/Qwen3-VL-2B-Instruct
cat test.jpeg | talu ask -m Qwen/Qwen3-VL-2B-Instruct "What objects are visible?"
```

### Quantize a model

```bash
talu convert LiquidAI/LFM2-350M
talu set LiquidAI/LFM2-350M-GAF4
talu ask "Explain quantization in one sentence."
```

Converts the model to 4-bit (default scheme: `gaf4_64`). Converted models are saved with a `-GAF4` suffix; the original remains available.

Available schemes (Grouped Affine):

| Scheme | Description |
|---|---|
| `gaf4_32` | 4-bit, group 32 — highest accuracy, largest |
| `gaf4_64` | 4-bit, group 64 — balanced (default) |
| `gaf4_128` | 4-bit, group 128 — smallest 4-bit |
| `gaf8_32` | 8-bit, group 32 — near-original quality |
| `gaf8_64` | 8-bit, group 64 |
| `gaf8_128` | 8-bit, group 128 |

4-bit reduces model size ~4x with some quality loss. 8-bit preserves more quality at ~2x reduction. Smaller group sizes improve accuracy but increase file size.

### Start the HTTP server

```bash
talu serve
```

By default, the server listens on `http://127.0.0.1:8258`:
- Console UI: `http://127.0.0.1:8258/`
- OpenResponses-compatible API: `http://127.0.0.1:8258/v1`
- Override port with: `talu serve --port 9000`

API compatibility target: [OpenResponses](https://openresponses.org).

More at [docs.talu.dev](https://docs.talu.dev).

## Script/CI Flow

For machine-friendly stdout contracts, use the CLI output flags:

```bash
MODEL_URI=$(talu get --model-uri LiquidAI/LFM2-350M)
CONVERTED_URI=$(talu convert --model-uri "$MODEL_URI")
SET_URI=$(talu set --model-uri "$CONVERTED_URI")
SESSION_ID=$(talu ask --session-id "Start incident analysis")
talu ask --session "$SESSION_ID" -q "List likely causes"
```

See `examples/cli/README.md` for broader interactive and CI patterns.

## Python

Basic chat:

```python
from talu import Chat

chat = Chat("LiquidAI/LFM2-350M", system="You are helpful.")
response = chat("What is the capital of France?")
print(response)

response = response.append("Now answer in one sentence.")
print(response)
```

Shared client for multiple chats:

```python
from talu import Client

client = Client("LiquidAI/LFM2-350M")
alice = client.chat(system="You are concise.")
bob = client.chat(system="You are detailed.")

print(alice("Explain recursion."))
print(bob("Explain recursion."))
client.close()
```

Persistent profile-backed sessions:

```python
import talu

profile = talu.Profile("work")
chat = talu.Chat("LiquidAI/LFM2-350M", profile=profile)
chat("Draft release notes.", stream=False)

print(talu.list_sessions(profile="work", limit=1))
```

Python docs: `bindings/python/README.md`

## Key Commands

| Command | Purpose |
|---|---|
| `talu get <model>` | Download/cache a model |
| `talu ls` | List cached models |
| `talu ls <model>` | List files for one cached model |
| `talu set <model>` | Set default model for active profile |
| `talu set show` | Show active profile configuration |
| `talu ask ...` | Ask a model and manage sessions |
| `talu convert <model> ...` | Quantize a model |
| `talu rm <model>` | Remove a cached model |
| `talu tokenize <model> "text"` | Inspect tokenization |
| `talu serve` | Start chat HTTP server + OpenResponses API (default: `127.0.0.1:8258`) |

## Profiles and Persistence

- Profiles are selected with `TALU_PROFILE` (default: `default`)
- Profile config stores defaults such as selected model
- Conversation sessions are persisted and can be resumed across CLI, Python, and server workflows when using the same profile/storage

## Architecture

Talu is organized as one inference/runtime core with thin interface layers:

- `core/`: Zig inference engine and C API boundary (`core/src/capi/`)
- `bindings/python/`: Python package (`import talu`)
- `bindings/rust/talu-sys/`: generated low-level Rust FFI bindings
- `bindings/rust/talu/`: safe Rust API over FFI
- `bindings/rust/cli/`: CLI and chat HTTP server (OpenResponses-compatible API)
- `ui/`: browser client served by `talu serve` (or `--html-dir`)

Request flow:

```text
CLI / Python / UI
       |
       v
Surface layer (CLI, Python binding, or HTTP server)
       |
       v
Rust safe wrapper / C API boundary
       |
       v
Core Zig runtime (model + inference + sessions)
       |
       v
Local model cache + profile/session storage
```

## Repository Layout

```text
core/               Zig inference engine + C API
bindings/python/    Python package
bindings/rust/      Rust FFI, safe wrapper, CLI/server
ui/                 Web UI source and build output
docs/               Documentation site source
examples/           Runnable CLI/Python/server examples
```

## Supported Models

Models are downloaded from HuggingFace on first use. The models below have been verified. Other sizes and variants based on the same architecture are expected to work as well. This list is updated as coverage expands — the objective is to support all major model architectures (see [Roadmap](#roadmap)).

**Qwen**
- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`

**LLaMA**
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-1B-Instruct`

**Mistral**
- `mistralai/Ministral-3B-Instruct`

**Gemma**
- `google/gemma-3-270m-it`
- `google/gemma-3-1b-it`

**Phi**
- `microsoft/Phi-3-mini-128k-instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `microsoft/Phi-4-mini-instruct`
- `microsoft/Phi-4-mini-reasoning`

**Granite**
- `ibm-granite/granite-4.0-h-350m`
- `ibm-granite/granite-4.0-micro`

**LFM**
- `LiquidAI/LFM2-350M`
- `LiquidAI/LFM2-1.2B`
- `LiquidAI/LFM2-2.6B`
- `LiquidAI/LFM2.5-1.2B-Instruct`
- `LiquidAI/LFM2.5-1.2B-Thinking`

## Build from Source

Requires [Zig 0.15.2](https://ziglang.org/download/), [Cargo (Rust)](https://www.rust-lang.org/tools/install), [Bun](https://bun.sh/), [uv](https://docs.astral.sh/uv/), and CMake.

```bash
git clone https://github.com/aprxi/talu.git
cd talu
make
```

`make` builds:
- CLI binary: `./zig-out/bin/talu`
- Native Python library:
  - Linux: `bindings/python/talu/libtalu.so`
  - macOS: `bindings/python/talu/libtalu.dylib`

This source build does not produce a Python wheel/package.

## Python Package Build (Short Version)

For a local package build with a timestamped `post` version:

```bash
cd bindings/python
make build
```

If native artifacts are missing (for example after `make clean` at repo root), this target bootstraps the required root build automatically.

Artifacts are written to `bindings/python/dist/` as:
- `talu-<base>.post<timestamp>.tar.gz`
- `talu-<base>.post<timestamp>-py3-none-any.whl`

Optional: set `BUILD_TS` for reproducible local version stamps, e.g. `make build BUILD_TS=202602121015`.

## Roadmap
This blueprint outlines the major components of the project and how they fit together. It serves as a shared high-level reference for the system’s structure and direction.

Development follows this structure: the architecture evolves incrementally while individual subsystems are expanded and refined. Features are introduced progressively, and implementation depth varies across components.

The roadmap below highlights the primary areas of focus for upcoming releases. In parallel, we continue ongoing work on bug fixes, performance improvements, compatibility updates, and documentation.

![Blueprint](./docs/images/talu-blueprint.svg)


**Core**
- Expand model coverage across supported HuggingFace architectures
- Add support for vision and speech models
- Maintain CPU and Metal as primary backends; introduce initial CUDA support
- Implement structured output validation for popular programming languages (syntax-aware validation engine)
- Add a policy firewall to enforce safety constraints for both local and remote models
- Integrate a VectorDB-backed chat memory layer (long-term memory / full RAG workflows)
- Add optional S3-compatible storage backend

**Bindings**
- Maintain Python as the primary, fully supported binding
- Complete Rust bindings beyond the current minimal FFI layer
- Introduce initial TypeScript bindings

**Server**
- Extend OpenResponses-compatible API with a Completions interface (to support alternative UIs and tooling)

**Web**
- Add agentic execution module
- Add plugin system (e.g. document upload and analysis)


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Area-specific policies are in `core/POLICY.md` (Zig) and `bindings/python/POLICY.md` (Python).

## License

MIT
