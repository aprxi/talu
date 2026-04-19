# Talu CLI Examples (Interactive + CI)

This guide tracks the current CLI surface for inference-only talu.

## Quick Start

```bash
# Download and cache a model
talu get LiquidAI/LFM2-350M

# Set default model (optional)
talu set LiquidAI/LFM2-350M

# Ask a prompt
talu ask "What is the capital of France?"
```

## Core Commands

| Area | Commands |
|------|----------|
| Model lifecycle | `get`, `ls`, `convert`, `rm`, `set` |
| Inference | `ask`, `eval`, `describe`, `tokenize`, `xray` |
| Server | `serve` |

## Interactive Usage

```bash
# Browse/download
talu get
talu get LiquidAI/LFM2-350M
talu ls
talu ls LiquidAI/LFM2-350M

# Quantize
talu convert LiquidAI/LFM2-350M
talu convert LiquidAI/LFM2-350M --scheme tq8

# Ask
talu ask "Hello"
talu ask --session <session_id>
talu ask --session <session_id> "Follow-up"
talu ask --session <session_id> --delete

# Inspect
talu describe LiquidAI/LFM2-350M
talu xray LiquidAI/LFM2-350M
talu tokenize LiquidAI/LFM2-350M "Hello world"
```

## Persistence (`TALU_DB_HOST`)

Chat/session persistence is externalized through Talupi APIs. Use the same
`TALU_DB_HOST` for `talu ask` and Talupi UI to read/write the same sessions.

```bash
# Create a persisted session ID
SESSION_ID=$(TALU_DB_HOST=localhost:7258 talu ask --session-id "Start analysis")

# Continue persisted session
TALU_DB_HOST=localhost:7258 talu ask --session "$SESSION_ID" "Follow-up"
```

## Script/CI Usage

Stdout contracts:
- `--model-uri` prints model URI only
- `--session-id` prints session ID only
- `-q` prints response text only
- `--format json` prints OpenResponses JSON
- `-s` suppresses stdout (exit code only)

```bash
# Contracted model URI flows
MODEL_URI=$(talu get --model-uri LiquidAI/LFM2-350M)
CONVERTED_URI=$(talu convert --model-uri "$MODEL_URI")
SET_URI=$(talu set --model-uri "$CONVERTED_URI")
talu rm --model-uri "$MODEL_URI"

# Output controls
ANSWER=$(talu ask -q "What is 2+2?")
talu ask --format json "Summarize this log block"
talu ask --output /tmp/answer.txt "Write one sentence"
talu ask -s "Run background check"
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_URI` | Per-command model override | `LiquidAI/LFM2-350M` |
| `SESSION_ID` | Target a session without `--session` | `sess_...` |
| `TALU_DB_HOST` | Talupi chat persistence host | `localhost:7258` |
| `TOKENS` | Max output tokens (default: `1024`) | `100` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `SEED` | Deterministic seed | `42` |
| `HF_ENDPOINT` | HuggingFace endpoint override | `https://huggingface.co` |

## Full Command Reference

| Command | Purpose | Mode |
|---------|---------|------|
| `talu set` | Set default model (interactive picker) | Interactive |
| `talu set <model-uri>` | Set default model directly | Both |
| `talu set --model-uri <model-uri>` | Set default and print URI only | Script/CI |
| `talu ask "prompt"` | Ask model | Both |
| `talu ask --session <session_id>` | Show transcript | Both |
| `talu ask --session <session_id> "prompt"` | Append to session | Both |
| `talu ask --session <session_id> --delete` | Delete session | Both |
| `talu ask --new` | Create empty persisted session and print UUID | Script/CI |
| `talu ask --session-id ["prompt"]` | Print session ID only | Script/CI |
| `talu ask -q "prompt"` | Response text only | Script/CI |
| `talu ask --raw "prompt"` | Preserve raw model output | Both |
| `talu ask --hide-thinking "prompt"` | Hide reasoning content | Both |
| `talu ask -s "prompt"` | Silent stdout | Script/CI |
| `talu ask --format json "prompt"` | OpenResponses JSON output | Script/CI |
| `talu ask --output <path> "prompt"` | Write response to file | Script/CI |
| `talu get` | Interactive HuggingFace browser | Interactive |
| `talu get <model-uri>` | Download model | Both |
| `talu get --model-uri <model-uri>` | Download/check and print URI only | Script/CI |
| `talu ls` | List cached models | Both |
| `talu rm <model-uri>` | Remove cached model | Both |
| `talu rm --model-uri <model-uri>` | Remove with no stdout | Script/CI |
| `talu convert <model-uri>` | Quantize model | Both |
| `talu convert --model-uri <model-uri>` | Convert and print URI only | Script/CI |
| `talu describe <model-uri>` | Model architecture info | Both |
| `talu xray <model-uri>` | Kernel profiling | Both |
| `talu tokenize <model-uri>` | Tokenize text | Both |
| `talu eval <model-uri>` | Run quality evaluations | Both |
| `talu serve` | Run HTTP server | Advanced |
