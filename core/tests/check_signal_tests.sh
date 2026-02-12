#!/usr/bin/env bash
# =============================================================================
# Signal Guard Test Runner
# =============================================================================
#
# Runs standalone tests for signal handling (SIGBUS recovery).
#
# These tests cannot run in the Zig test framework because:
#   - Zig test framework has its own signal handlers
#   - sigsetjmp/siglongjmp require being the active signal handler
#
# This script builds and runs standalone executables that verify:
#   - Normal callbacks work through the signal guard
#   - SIGBUS signals are caught and recovered from
#   - Multiple guard cycles work correctly
#   - Return values propagate correctly
#
# Usage:
#   ./core/tests/check_signal_tests.sh           # Run all signal tests
#   ./core/tests/check_signal_tests.sh -v        # Verbose output
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_ROOT/core/tests/helpers/signal"
BUILD_DIR="$PROJECT_ROOT/zig-out/bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
VERBOSE=false
for arg in "$@"; do
    case $arg in
        -v|--verbose)
            VERBOSE=true
            ;;
    esac
done

# Platform check
OS="$(uname -s)"
if [[ "$OS" != "Linux" && "$OS" != "Darwin" ]]; then
    echo -e "${YELLOW}Signal guard tests skipped on $OS (only Linux/macOS supported)${NC}"
    exit 0
fi

echo "=== Signal Guard Tests ==="
echo ""

mkdir -p "$BUILD_DIR"

PASS=0
FAIL=0

# -----------------------------------------------------------------------------
# Test 1: Zig standalone test
# -----------------------------------------------------------------------------
echo -n "Building Zig signal test... "
if zig build-exe \
    "$TEST_DIR/signal_guard_test.zig" \
    "$PROJECT_ROOT/core/src/capi/signal_guard.c" \
    -lc \
    -femit-bin="$BUILD_DIR/signal_guard_test" \
    2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    FAIL=$((FAIL + 1))
fi

if [[ -f "$BUILD_DIR/signal_guard_test" ]]; then
    echo -n "Running Zig signal test... "
    if OUTPUT=$("$BUILD_DIR/signal_guard_test" 2>&1); then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
        if $VERBOSE; then
            echo "$OUTPUT" | sed 's/^/  /'
        fi
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
        echo "$OUTPUT" | sed 's/^/  /'
    fi
fi

# -----------------------------------------------------------------------------
# Test 2: C standalone test
# -----------------------------------------------------------------------------
echo -n "Building C signal test... "
if gcc -o "$BUILD_DIR/signal_guard_test_c" \
    "$TEST_DIR/signal_guard_test.c" \
    "$PROJECT_ROOT/core/src/capi/signal_guard.c" \
    2>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    FAIL=$((FAIL + 1))
fi

if [[ -f "$BUILD_DIR/signal_guard_test_c" ]]; then
    echo -n "Running C signal test... "
    if OUTPUT=$("$BUILD_DIR/signal_guard_test_c" 2>&1); then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS + 1))
        if $VERBOSE; then
            echo "$OUTPUT" | sed 's/^/  /'
        fi
    else
        echo -e "${RED}FAIL${NC}"
        FAIL=$((FAIL + 1))
        echo "$OUTPUT" | sed 's/^/  /'
    fi
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo -e "Signal Tests: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo "================================================================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

echo ""
echo "All signal tests passed!"
