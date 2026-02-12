# Talu Python Bindings

Python interface to the talu inference engine. Run LLMs locally with a single `pip install` — no PyTorch, no runtime dependencies.

Models are downloaded from HuggingFace automatically on first use.

```bash
pip install talu
```

```python
from talu import Chat

chat = Chat("LiquidAI/LFM2-350M")
for token in chat("Tell me a short joke"):
    print(token, end="", flush=True)
```

## What's Included

- **Chat** — multi-turn conversations, streaming, system prompts, branching
- **Client** — shared model serving multiple independent chats
- **AsyncChat / AsyncClient** — async equivalents for FastAPI, aiohttp, etc.
- **Tokenizer** — encode/decode text without loading model weights
- **PromptTemplate** — Jinja2-style chat templates
- **Converter** — quantize models (e.g. GAF4) for faster inference
- **Profiles** — persist and resume chat sessions
- **Embeddings** — vector embeddings from any loaded model
- **Structured output** — JSON schema-constrained generation
- **Validator** — streaming JSON schema validation

## Examples

The [`examples/python/`](https://github.com/aprxi/talu/tree/main/examples/python) directory has runnable scripts covering the main features:

| # | Script | Topic |
|---|--------|-------|
| 00 | `fetch_model` | Fetch a model before first use |
| 01 | `chat` | Chat basics, multi-turn, and follow-ups |
| 02 | `streaming` | Stream tokens to stdout |
| 03 | `tokenize_text` | Encode, decode, and inspect tokens |
| 04 | `prompt_templates` | Build prompts with templates |
| 05 | `manage_models` | Cache, download, and inspect model files |
| 06 | `manage_client` | Reuse one model across multiple chats |
| 07 | `client_ask` | One-shot tasks and shared models |
| 08 | `generation_settings` | Temperature and limits |
| 09 | `convert_model` | Convert a model for local use |
| 10 | `manage_profiles` | Persistent chat sessions with profiles |
| 11 | `save_restore` | Save and restore a chat session |
| 12 | `embeddings` | Text embeddings for similarity and search |
| 13 | `chat_template` | Format messages with a chat template |
| 14 | `batch_chat` | Run multiple prompts and save results |
| 15 | `token_budget` | Manage a token budget with truncation |
| 16 | `structured_output` | Get structured JSON responses |
| 17 | `chat_history` | Inspect and manage chat history |
| 18 | `generation_settings` | Adjust basic generation settings |
| 19 | `branching` | Branch conversations to explore alternatives |
| 20 | `streaming_validation` | Validate JSON as it streams in |

## Documentation

Full guide at [docs.talu.dev](https://docs.talu.dev) — CLI usage, supported models, build from source, and API reference.

## License

MIT
