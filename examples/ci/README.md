# Talu CLI Examples

This directory documents two usage tracks in one place:
- Interactive CLI (human-readable)
- Script/CI CLI (machine-friendly stdout contracts)

## Getting Started

```bash
# 1) Fetch a model into cache

talu get LiquidAI/LFM2-350M

# 2) Set default model for future commands
talu set LiquidAI/LFM2-350M

# 3) Ask a question (new session by default)
talu ask "What is the capital of France?"

# 4) List and continue sessions
talu ask
talu ask --session <session_id> "Tell me more"
```

## Mode Split

Use Interactive mode for manual usage and readable output.
Use Script/CI mode when you need stable stdout contracts and exit-code checks.

## Global Flags

These apply to all commands.

```bash
-v            # INFO logs
-vv           # DEBUG logs
-vvv          # TRACE logs
--log-format human|json
```

## Quick Index

| Area | Commands |
|------|----------|
| Model lifecycle | `get`, `ls`, `convert`, `rm`, `set` |
| Chat/session | `ask`, `search` |
| Tooling/analysis | `shell`, `describe`, `xray`, `tokenize` |
| Storage admin | `db <subcmd>` |
| Runtime service | `serve` |

## Interactive Guide

### 1) Model Lifecycle

```bash
# Browse and pick from HuggingFace interactively

talu get

# Download directly by URI
talu get LiquidAI/LFM2-350M

# Manage profile-local pinned models
talu get --add-pin LiquidAI/LFM2-350M
talu get --remove-pin LiquidAI/LFM2-350M
talu ls -P
talu get --sync-pins

# Run raw non-stream sample inference across pinned models
talu sample

# List cached models/files
talu ls
talu ls LiquidAI/LFM2-350M

# Convert quantization
talu convert LiquidAI/LFM2-350M
talu convert LiquidAI/LFM2-350M --scheme gaf8_64

# Remove from cache
talu rm LiquidAI/LFM2-350M
```

### 2) Set Defaults

```bash
# Interactive picker
talu set

# Explicit default
talu set LiquidAI/LFM2-350M

# Show current config
talu set show
```

### 3) Ask and Sessions

`ask` creates a new session unless you pass `--session` or `SESSION_ID`.

```bash
# Ask in a new session
talu ask "Hello"

# List sessions (includes stats)
talu ask

# View transcript
talu ask --session <session_id>

# Continue session
talu ask --session <session_id> "Follow-up"

# Delete session
talu ask --session <session_id> --delete

# Ephemeral query (no persistence)
talu ask --no-bucket "Quick check"
```

### 4) Agent

```bash
# Tool-calling agent with confirmation
talu agent "Find large files in this repo"

# Custom policy/tools
talu agent --policy ./policy.json --tools ./tools/ "Run diagnostics"
```

### 5) Inspect and Analyze

```bash
# Architecture summary
talu describe LiquidAI/LFM2-350M

# Kernel profiling
talu xray LiquidAI/LFM2-350M

# Tokenization
talu tokenize LiquidAI/LFM2-350M "Hello world"
```

### 6) Search Conversations

```bash
# Interactive conversation search (from storage bucket)
talu search
```

## Script/CI Guide

Stdout contracts in this mode:
- `--model-uri` modes print only the URI/result string on success.
- `--session-id` prints only session ID on success.
- `-q` prints response text only.
- `--format json` prints OpenResponses JSON.
- `-s` prints nothing to stdout.
- Logs remain on stderr (`-v`/`-vv`/`-vvv`).

### 1) Model URI Contracted Flows

```bash
# Download/check and return URI only
MODEL_URI=$(talu get --model-uri LiquidAI/LFM2-350M)

# Convert and return converted URI only
CONVERTED_URI=$(talu convert --model-uri "$MODEL_URI")

# Set default and echo selected URI only
SET_URI=$(talu set --model-uri "$CONVERTED_URI")

# Remove model with no stdout (exit code only)
talu rm --model-uri "$MODEL_URI"
```

### 2) Session-ID Contracted Flows

```bash
# New session ID only (with or without prompt)
SESSION_ID=$(talu ask --session-id)
SESSION_ID=$(talu ask --session-id "Start incident analysis")

# Continue session explicitly
talu ask --session "$SESSION_ID" "List likely causes"
```

### 3) Output Controls

```bash
# Response text only
ANSWER=$(talu ask -q "What is 2+2?")

# OpenResponses JSON
talu ask --format json "Summarize this log block"

# File output
talu ask --output /tmp/answer.txt "Write one sentence"

# Silent stdout (exit-code based)
talu ask -s "Run background check"
```

### 4) Use-Case Scripts in This Directory

- `examples/cli/ask/incident_update.sh`
- `examples/cli/ask/transcript_json.sh`

## Implicit Ask Mode

When no subcommand is provided, Talu can implicitly run `ask`.

```bash
# Equivalent to: talu ask -m <model-uri> "hello"
echo "hello" | talu -m LiquidAI/LFM2-350M

# Equivalent to: talu ask "hello"
echo "hello" | talu
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MODEL_URI` | Per-command model override | `LiquidAI/LFM2-350M` |
| `SESSION_ID` | Target a session without `--session` | UUID |
| `TOKENS` | Max output tokens (default: `1024`) | `100` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `SEED` | Deterministic seed control | `42` |
| `TALU_PROFILE` | Storage profile name | `default` |
| `TALU_BUCKET` | Storage bucket path override | `/path/to/db` |
| `TOOL_POLICY` | Shell policy path | `/path/to/policy.json` |
| `HF_ENDPOINT` | HuggingFace endpoint override | `https://huggingface.co` |

## Advanced: DB and Serve

### `talu db`

```bash
# Initialize custom storage
talu db init ./mydb

# List sessions (table/json/csv)
talu db list ./mydb
talu db list ./mydb --format json
talu db list ./mydb --format csv

# Show session
talu db show <session_id> ./mydb
talu db show <session_id> ./mydb --format json --raw

# Delete session
talu db delete <session_id> ./mydb
```

### `talu serve`

```bash
# Run HTTP server (see `talu serve --help` for options)
talu serve
```

## Full Command Reference

| Command | Purpose | Mode |
|---------|---------|------|
| `talu set` | Set default model (interactive picker) | Interactive |
| `talu set <model-uri>` | Set default model directly | Both |
| `talu set --model-uri <model-uri>` | Set default and print URI only | Script/CI |
| `talu ask "prompt"` | Ask model (new session by default) | Both |
| `talu ask` | List sessions (with stats) | Both |
| `talu ask --session <session_id>` | Show transcript | Both |
| `talu ask --session <session_id> "prompt"` | Append to session | Both |
| `talu ask --session <session_id> --delete` | Delete session | Both |
| `talu ask --new` | Create empty session and print UUID | Script/CI |
| `talu ask --session-id ["prompt"]` | Create session and print session ID only | Script/CI |
| `talu ask -q "prompt"` | Response text only | Script/CI |
| `talu ask --raw "prompt"` | Preserve raw model output (no reasoning-tag filtering) | Both |
| `talu ask --hide-thinking "prompt"` | Hide reasoning/thinking content from output | Both |
| `talu ask -s "prompt"` | Silent stdout | Script/CI |
| `talu ask --format json "prompt"` | OpenResponses JSON output | Script/CI |
| `talu ask --output <path> "prompt"` | Write output to file | Script/CI |
| `talu ask --no-bucket "prompt"` | Ephemeral query | Both |
| `talu agent "prompt"` | Interactive tool-calling agent | Interactive |
| `talu search` | Interactive conversation search | Interactive |
| `talu get` | Interactive HuggingFace browser | Interactive |
| `talu get <model-uri>` | Download model | Both |
| `talu get --model-uri <model-uri>` | Download/check and print URI only | Script/CI |
| `talu get --add-pin <model-uri>` | Add model URI to profile pin list | Both |
| `talu get --remove-pin <model-uri>` | Remove model URI from profile pin list | Both |
| `talu get --sync-pins` | Sync/download missing pinned models | Both |
| `talu ls -P` | Quick list of pinned models for profile | Both |
| `talu sample [--max-models N]` | Run raw non-stream samples across pinned models | Both |
| `talu ls` | List cached models | Both |
| `talu rm <model-uri>` | Remove cached model | Both |
| `talu rm --model-uri <model-uri>` | Remove with no stdout | Script/CI |
| `talu convert <model-uri>` | Quantize model | Both |
| `talu convert --model-uri <model-uri>` | Convert and print URI only | Script/CI |
| `talu describe <model-uri>` | Model architecture info | Both |
| `talu xray <model-uri>` | Kernel profiling | Both |
| `talu tokenize <model-uri>` | Tokenize text | Both |
| `talu db <subcmd>` | Storage admin operations | Advanced |
| `talu serve` | Run HTTP server | Advanced |
