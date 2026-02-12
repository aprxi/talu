#!/usr/bin/env bash
# TaluDB CLI Demo
#
# This script demonstrates the full TaluDB workflow:
# 1. Initialize storage
# 2. Generate conversations with persistence
# 3. List and inspect sessions
# 4. Export and delete sessions
#
# Prerequisites:
#   - talu CLI built and in PATH (or ./zig-out/bin/talu)
#   - A model available (set with `talu set` or MODEL_URI env var)
#
# Usage:
#   ./demo_taludb.sh

set -e

# Configuration
TALU="${TALU:-talu}"
DB_PATH="${DB_PATH:-/tmp/talu_demo_db}"
export TALU_LOG_LEVEL="${TALU_LOG_LEVEL:-off}"
export TOKENS="${TOKENS:-20}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "\n${CYAN}=== $1 ===${NC}\n"
}

echo_cmd() {
    echo -e "${YELLOW}\$ $1${NC}"
}

# Cleanup any previous demo
rm -rf "$DB_PATH"

# Step 1: Initialize storage
echo_step "Step 1: Initialize TaluDB Storage"
echo_cmd "talu db init $DB_PATH"
$TALU db init "$DB_PATH"

# Step 2: Generate first conversation
echo_step "Step 2: Generate First Conversation"
echo_cmd "talu generate --db $DB_PATH \"What is the capital of France?\""
SESSION1=$($TALU generate --db "$DB_PATH" "What is the capital of France?" 2>&1 | grep "Session ID:" | awk '{print $3}')
echo -e "${GREEN}Created session: $SESSION1${NC}"

# Step 3: Generate second conversation with custom session ID
echo_step "Step 3: Generate Second Conversation (Custom Session ID)"
echo_cmd "talu generate --db $DB_PATH --session my-math-chat \"What is 2 + 2?\""
$TALU generate --db "$DB_PATH" --session my-math-chat "What is 2 + 2?"

# Step 4: List all sessions
echo_step "Step 4: List All Sessions"
echo_cmd "talu db list $DB_PATH"
$TALU db list "$DB_PATH"

# Step 5: Show session details with transcript
echo_step "Step 5: Show Session Details and Transcript"
echo_cmd "talu db show $SESSION1 $DB_PATH"
$TALU db show "$SESSION1" "$DB_PATH"

# Step 6: Show session with raw JSON
echo_step "Step 6: Export Session as JSON"
echo_cmd "talu db show --format json --raw my-math-chat $DB_PATH"
$TALU db show --format json --raw my-math-chat "$DB_PATH"

# Step 7: Delete a session
echo_step "Step 7: Delete a Session"
echo_cmd "talu db delete --force my-math-chat $DB_PATH"
$TALU db delete --force my-math-chat "$DB_PATH"

# Step 8: Verify deletion
echo_step "Step 8: Verify Deletion"
echo_cmd "talu db list $DB_PATH"
$TALU db list "$DB_PATH"

# Summary
echo_step "Demo Complete!"
echo -e "Storage location: ${GREEN}$DB_PATH${NC}"
echo -e "Remaining sessions: ${GREEN}1${NC}"
echo ""
echo "Next steps:"
echo "  - Continue the conversation: talu generate --db $DB_PATH --session $SESSION1 \"Tell me more\""
echo "  - Clean up: rm -rf $DB_PATH"
