#!/usr/bin/env bash
#
# Talu CLI testable example (script/CI friendly)
#
# What this script does:
#   1) get model URI
#   2) convert model URI
#   3) set default model URI
#   4) run ask modes (-q, --session-id/--session, --format json, --output, -s)
#   5) verify env var generation settings in JSON output
#   6) remove model URI
#
# Usage:
#   bash examples/cli/tests/test_cli.sh
#
# Env (optional):
#   MODEL_URI=LiquidAI/LFM2-350M
#   SKIP_CONVERT=1
#   TOKENS=100
#   JSON_PREVIEW_CHARS=240
#   SYSTEM_PROMPT="You are a helpful assistant."
#   SESSION_SYSTEM_PROMPT="You are a helpful assistant. Always return at least one visible character."

set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Error: run this script with bash: bash examples/cli/tests/test_cli.sh" >&2
  exit 1
fi

# --- Output helpers -----------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() {
  echo -e "${GREEN}OK${NC}: $1"
}

show_output() {
  echo -e "${YELLOW}OUTPUT${NC}: $1"
}

show_json_preview() {
  local json_text="$1"
  local max_chars="$2"
  if [[ "${#json_text}" -le "$max_chars" ]]; then
    show_output "$json_text"
  else
    show_output "${json_text:0:max_chars}..."
  fi
}

error_exit() {
  echo -e "${RED}Error${NC}: $1" >&2
  exit 1
}

# --- User-configurable inputs -------------------------------------------------
MODEL_URI="${MODEL_URI:-LiquidAI/LFM2-350M}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
TOKENS="${TOKENS:-100}"
JSON_PREVIEW_CHARS="${JSON_PREVIEW_CHARS:-240}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"
SESSION_SYSTEM_PROMPT="${SESSION_SYSTEM_PROMPT:-You are a helpful assistant. Always return at least one visible character in every reply.}"

# Exactly two prompts (main + follow-up).
PROMPT_MAIN="What is the weather in Paris today? If you are unsure, say you are unsure."
PROMPT_FOLLOW_UP="Based on your previous answer, summarize in one short sentence."

# Internal state:
EXPECTED_MODEL_URI="$MODEL_URI"
# QUANT_URI is the model URI used after conversion. If conversion is skipped,
# it stays equal to MODEL_URI.
QUANT_URI="$MODEL_URI"

# --- Preconditions ------------------------------------------------------------
if ! type -P talu >/dev/null 2>&1; then
  error_exit "talu not found"
fi
if ! type -P jq >/dev/null 2>&1; then
  error_exit "jq not found"
fi

echo -e "${CYAN}Using talu:${NC} talu"
echo -e "${CYAN}Using MODEL_URI:${NC} $EXPECTED_MODEL_URI"

# ==============================================================================
# Step 1: talu get --model-uri
# Contract: stdout is exactly the requested model URI.
# ==============================================================================
MODEL_URI="$(talu get --model-uri "$EXPECTED_MODEL_URI")"
[[ "$MODEL_URI" == "$EXPECTED_MODEL_URI" ]] || {
  echo "expected: $EXPECTED_MODEL_URI" >&2
  echo "actual:   $MODEL_URI" >&2
  error_exit "get --model-uri mismatch"
}

ok "Step 1: Get model URI"

# ==============================================================================
# Step 2: talu convert --model-uri
# Contract: stdout is non-empty converted model URI/path.
# ==============================================================================
if [[ "$SKIP_CONVERT" != "1" ]]; then
  CONVERT_URI="$(talu convert --model-uri "$MODEL_URI")"
  [[ -n "$CONVERT_URI" ]] || {
    error_exit "convert --model-uri returned empty stdout"
  }
  QUANT_URI="$CONVERT_URI"
  ok "Step 2: Convert model URI"
  show_output "QUANT_URI=$QUANT_URI"
else
  echo -e "${CYAN}Skip:${NC} convert (SKIP_CONVERT=1)"
fi

# ==============================================================================
# Step 3: talu set --model-uri
# Contract: stdout is exactly the URI that was set.
# ==============================================================================
SET_URI="$(talu set --model-uri "$QUANT_URI")"
[[ "$SET_URI" == "$QUANT_URI" ]] || {
  echo "expected: $QUANT_URI" >&2
  echo "actual:   $SET_URI" >&2
  error_exit "set --model-uri mismatch"
}

ok "Step 3: Set default model URI"

# --- Shared env for ask steps -------------------------------------------------
# Script is linear, so set common generation env once.
export MODEL_URI="$QUANT_URI"
export TOKENS="$TOKENS"
export SEED=42
export TEMPERATURE=0.7

# ==============================================================================
# Step 4: talu ask -q
# Contract: returns non-empty model text.
# ==============================================================================
ASK_Q="$(
  talu ask \
    -S "$SYSTEM_PROMPT" \
    -q "$PROMPT_MAIN"
)"
[[ -n "$ASK_Q" ]] || {
  error_exit "ask -q returned empty output"
}

ok "Step 4: Ask quiet mode"
show_output "${ASK_Q:0:120}"

# ==============================================================================
# Step 5: talu ask --session-id + --session (two-turn flow)
# Contract:
#   - --session-id returns a non-empty session ID
#   - two prompts are appended into the same session history
# ==============================================================================
SESSION_ID="$(
  talu ask --session-id | tail -n 1 | tr -d '\r\n'
)"
[[ -n "$SESSION_ID" ]] || {
  error_exit "ask --session-id returned empty output"
}

SESSION_MAIN="$(
  talu ask \
    --session "$SESSION_ID" \
    --model-uri "$QUANT_URI" \
    -S "$SESSION_SYSTEM_PROMPT" \
    -q "$PROMPT_MAIN"
)"

FOLLOW_UP="$(
  talu ask \
    --session "$SESSION_ID" \
    --model-uri "$QUANT_URI" \
    -S "$SESSION_SYSTEM_PROMPT" \
    -q "$PROMPT_FOLLOW_UP"
)"

SESSION_JSON="$(
  talu ask \
    --session "$SESSION_ID" \
    --format json
)"
echo "$SESSION_JSON" | python3 -m json.tool >/dev/null 2>&1 || {
  error_exit "ask --session --format json produced invalid JSON"
}
echo "$SESSION_JSON" | jq -e \
  --arg main "$PROMPT_MAIN" \
  --arg follow "$PROMPT_FOLLOW_UP" '
  (any(.[]; .type=="message" and .role=="user" and any(.content[]?; .text? == $main)))
  and (any(.[]; .type=="message" and .role=="user" and any(.content[]?; .text? == $follow)))
  and ([.[] | select(.type=="message" and .role=="user")] | length >= 2)
' >/dev/null || {
  echo "$SESSION_JSON" | jq '[.[] | select(.type=="message")] | map({role, content})' >&2 || true
  error_exit "session transcript does not contain both user prompts"
}

ok "Step 5: Session flow (2 prompts)"
if [[ -n "$FOLLOW_UP" ]]; then
  show_output "${FOLLOW_UP:0:120}"
else
  show_output "(empty model text; transcript validation passed)"
fi

# ==============================================================================
# Step 6: talu ask --format json
# Contract: valid JSON + assistant generation metadata matches request settings.
# ==============================================================================
ASK_JSON="$(
  talu ask \
    --model-uri "$QUANT_URI" \
    -S "$SYSTEM_PROMPT" \
    --format json "$PROMPT_MAIN"
)"
echo "$ASK_JSON" | python3 -m json.tool >/dev/null 2>&1 || {
  error_exit "ask --format json produced invalid JSON"
}

echo "$ASK_JSON" | jq -e --arg uri "$QUANT_URI" --argjson expected_tokens "$TOKENS" '
  ([.[] | select(.type=="message" and .role=="assistant" and (.generation | type == "object"))] | last) as $a
  | ($a != null)
  and ((($a.generation.temperature - 0.7) | if . < 0 then -. else . end) < 0.000001)
  and ($a.generation.max_tokens == $expected_tokens)
  and ($a.generation.seed == 42)
  and ($a.generation.model | endswith($uri))
' >/dev/null || {
  echo "$ASK_JSON" | jq '[.[] | select(.type=="message" and .role=="assistant")] | map({id, generation})' >&2 || true
  error_exit "ask --format json generation metadata mismatch"
}

ok "Step 6: Ask JSON mode"
show_json_preview "$ASK_JSON" "$JSON_PREVIEW_CHARS"

# ==============================================================================
# Step 7: talu ask --output
# Contract: file is created and non-empty.
# ==============================================================================
TMP_FILE="$(mktemp)"
talu ask \
  --model-uri "$QUANT_URI" \
  -S "$SYSTEM_PROMPT" \
  -q \
  --output "$TMP_FILE" \
  "$PROMPT_MAIN" >/dev/null
[[ -s "$TMP_FILE" ]] || {
  rm -f "$TMP_FILE"
  error_exit "ask --output produced empty file"
}
rm -f "$TMP_FILE"

ok "Step 7: Ask output file mode"

# ==============================================================================
# Step 8: talu ask -s
# Contract: -s suppresses stdout entirely.
# ==============================================================================
ASK_SILENT="$(
  talu ask \
    --model-uri "$QUANT_URI" \
    -S "$SYSTEM_PROMPT" \
    -s "$PROMPT_MAIN"
)"
[[ -z "$ASK_SILENT" ]] || {
  echo "stdout: $ASK_SILENT" >&2
  error_exit "ask -s produced stdout"
}

ok "Step 8: Ask silent mode"

# ==============================================================================
# Step 9: env vars honored in JSON generation metadata
# Contract: MODEL_URI/TOKENS/TEMPERATURE/SEED env values are reflected in JSON.
# ==============================================================================
ENV_JSON="$(
  TEMPERATURE=0.5 \
    talu ask \
      -S "$SYSTEM_PROMPT" \
      --format json "$PROMPT_FOLLOW_UP"
)"
echo "$ENV_JSON" | python3 -m json.tool >/dev/null 2>&1 || {
  error_exit "env var override JSON check produced invalid JSON"
}
echo "$ENV_JSON" | jq -e --arg uri "$QUANT_URI" --argjson expected_tokens "$TOKENS" '
  ([.[] | select(.type=="message" and .role=="assistant" and (.generation | type == "object"))] | last) as $a
  | ($a != null)
  and ((($a.generation.temperature - 0.5) | if . < 0 then -. else . end) < 0.000001)
  and ($a.generation.max_tokens == $expected_tokens)
  and ($a.generation.seed == 42)
  and ($a.generation.model | endswith($uri))
' >/dev/null || {
  echo "$ENV_JSON" | jq '{
    assistant: [.[] | select(.type=="message" and .role=="assistant")
      | {id, finish_reason, generation}],
    finish_reasons: [.[] | select(.finish_reason != null)
      | {type, id, finish_reason}]
  }' >&2 || true
  echo "Hint: if assistant generation is missing with finish_reason=length, increase TOKENS." >&2
  error_exit "env var override generation metadata mismatch"
}

ok "Step 9: Env overrides reflected in JSON metadata"
show_json_preview "$ENV_JSON" "$JSON_PREVIEW_CHARS"

# ==============================================================================
# Step 10: talu rm --model-uri
# Contract: script-friendly remove emits no stdout on success.
# ==============================================================================
RM_OUT="$(talu rm --model-uri "$QUANT_URI")"
[[ -z "$RM_OUT" ]] || {
  echo "stdout: $RM_OUT" >&2
  error_exit "rm --model-uri produced stdout"
}

ok "Step 10: Remove model URI"
echo -e "${CYAN}Summary:${NC}"
echo "  MODEL_URI=$EXPECTED_MODEL_URI"
echo "  QUANT_URI=$QUANT_URI"
echo "  TOKENS=$TOKENS"
echo "  SESSION_ID=$SESSION_ID"
echo -e "${GREEN}All checks passed.${NC}"
