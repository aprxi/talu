#!/bin/bash
set -euo pipefail

# Create a session and export the transcript as OpenResponses JSON.
SESSION_ID=$(talu ask --new)
MODEL_URI="Qwen/Qwen3-0.6B"
OUTPUT_FILE="/tmp/talu_transcript.json"

SESSION_ID=$SESSION_ID MODEL_URI=$MODEL_URI talu ask -s "What is 2+2?"
SESSION_ID=$SESSION_ID MODEL_URI=$MODEL_URI talu ask -s "What is 3+3?"

SESSION_ID=$SESSION_ID talu ask --format json --session "$SESSION_ID" > "$OUTPUT_FILE"

echo "Wrote transcript to $OUTPUT_FILE"

