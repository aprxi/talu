#!/usr/bin/env bash
set -euo pipefail

# Use a known model for repeatable results.
MODEL_URI="Qwen/Qwen3-0.6B"
talu set "$MODEL_URI" >/dev/null

# Start a session and capture only the session ID.
SESSION_ID=$(talu ask --session-id "Summarize this incident: API latency spiked to 2s at 12:04 UTC. Error rate peaked at 8%. Mitigation: scaled cache layer. Provide a short summary.")

# Follow-ups in the same session.
talu ask --session "$SESSION_ID" "List immediate next steps for investigation."
talu ask --session "$SESSION_ID" "Draft a short status update for stakeholders."
