#!/usr/bin/env bash
# =============================================================================
# C API Specification Generator
# =============================================================================
#
# Scans core/src/capi/*.zig files and generates a markdown API reference.
# Uses grep and awk to extract exported functions and their documentation.
#
# Usage:
#   generate_capi_spec.sh                    # Full spec (all modules)
#   generate_capi_spec.sh --category tokenizer  # Only tokenizer module
#   generate_capi_spec.sh -c chat            # Short form
#   generate_capi_spec.sh --list             # List available categories
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CAPI_DIR="$PROJECT_ROOT/core/src/capi"

# Parse arguments
CATEGORY=""
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        --list|-l)
            LIST_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--category|-c MODULE] [--list|-l]" >&2
            exit 1
            ;;
    esac
done

# Define modules in order (must be before any usage)
declare -A MODULE_NAMES
MODULE_NAMES[error]="Error"
MODULE_NAMES[tensor]="Tensor"
MODULE_NAMES[dlpack]="DLPack"
MODULE_NAMES[buffer]="Buffer"
MODULE_NAMES[tokenizer]="Tokenizer"
MODULE_NAMES[template]="Template"
MODULE_NAMES[session]="Session"
MODULE_NAMES[architecture]="Architecture"
MODULE_NAMES[repository]="Repository"
MODULE_NAMES[converter]="Converter"
MODULE_NAMES[chat]="Chat"
MODULE_NAMES[messages]="Messages"
MODULE_NAMES[persistence]="Persistence"
MODULE_NAMES[validate]="Validate"
MODULE_NAMES[xray]="X-Ray"

declare -A MODULE_DESCS
MODULE_DESCS[error]="Error handling and retrieval"
MODULE_DESCS[tensor]="Tensor creation and inspection"
MODULE_DESCS[dlpack]="DLPack interoperability"
MODULE_DESCS[buffer]="Shared buffer management"
MODULE_DESCS[tokenizer]="Text tokenization"
MODULE_DESCS[template]="Jinja2 template rendering"
MODULE_DESCS[session]="Model resolution and utilities"
MODULE_DESCS[architecture]="Model architecture registry"
MODULE_DESCS[repository]="HuggingFace model cache"
MODULE_DESCS[converter]="Model format conversion"
MODULE_DESCS[chat]="Chat state and generation"
MODULE_DESCS[messages]="Conversation history"
MODULE_DESCS[persistence]="Message persistence"
MODULE_DESCS[validate]="Structured output validation"
MODULE_DESCS[xray]="Tensor inspection during inference"

# Module order
MODULES=(error tensor dlpack buffer tokenizer template session architecture repository converter chat messages persistence validate xray)

# Handle --list option
if [[ "$LIST_ONLY" == true ]]; then
    echo "Available categories:"
    for mod in "${MODULES[@]}"; do
        printf "  %-12s - %s\n" "$mod" "${MODULE_DESCS[$mod]}"
    done
    exit 0
fi

# Validate category if specified
if [[ -n "$CATEGORY" ]]; then
    if [[ -z "${MODULE_NAMES[$CATEGORY]+x}" ]]; then
        echo "Error: Unknown category '$CATEGORY'" >&2
        echo "Use --list to see available categories" >&2
        exit 1
    fi
fi

# Header
if [[ -z "$CATEGORY" ]]; then
    cat << 'EOF'
# Talu C API Reference

Auto-generated from `core/src/capi/` source files.

## Overview

The Talu C API provides FFI bindings for Python and other languages. All functions follow these conventions:

- **Naming**: All functions are prefixed with `talu_`
- **Error handling**: Functions returning `i32` return 0 on success, non-zero on error
- **Memory management**: Functions ending with `_free` release memory allocated by the library
- **Handles**: Opaque pointers (like `TokenizerHandle`) represent internal state

## Table of Contents

EOF

    # Print TOC
    for mod in "${MODULES[@]}"; do
        name="${MODULE_NAMES[$mod]}"
        desc="${MODULE_DESCS[$mod]}"
        anchor=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
        echo "- [$name](#$anchor) - $desc"
    done

    echo ""
    echo "---"
    echo ""
else
    echo "# Talu C API: ${MODULE_NAMES[$CATEGORY]}"
    echo ""
fi

# Function to extract exports from a file
extract_exports() {
    local file="$1"
    local mod_name="$2"
    local mod_desc="$3"

    echo "## $mod_name"
    echo ""
    echo "$mod_desc"
    echo ""

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        echo "*File not found*"
        echo ""
        return
    fi

    # Extract exported functions
    # Pattern: pub export fn talu_*
    local funcs
    funcs=$(grep -E "^pub export fn talu_" "$file" 2>/dev/null | \
            sed 's/pub export fn //' | \
            sed 's/ callconv(.c)//' | \
            sed 's/{.*//' || true)

    if [[ -z "$funcs" ]]; then
        echo "*No exported functions*"
        echo ""
        return
    fi

    echo "### Functions"
    echo ""

    # For each function, extract name and try to get doc comment
    while IFS= read -r func_sig; do
        # Extract function name
        func_name=$(echo "$func_sig" | cut -d'(' -f1 | tr -d ' ')

        echo "#### \`$func_name\`"
        echo ""

        # Try to get doc comment (lines starting with /// before the function)
        local doc
        doc=$(grep -B 20 "^pub export fn $func_name" "$file" 2>/dev/null | \
              tac | \
              awk '/^pub export fn/ { found=1; next } found && /^\/\/\// { print; next } found { exit }' | \
              tac | \
              sed 's/^\/\/\/ //' || true)

        if [[ -n "$doc" ]]; then
            echo "$doc"
            echo ""
        fi

        # Print simplified signature
        echo '```c'
        echo "$func_sig;"
        echo '```'
        echo ""
    done <<< "$funcs"
}

# Process modules
if [[ -n "$CATEGORY" ]]; then
    # Single category mode
    name="${MODULE_NAMES[$CATEGORY]}"
    desc="${MODULE_DESCS[$CATEGORY]}"
    file="$CAPI_DIR/${CATEGORY}.zig"
    extract_exports "$file" "$name" "$desc"
else
    # All modules
    for mod in "${MODULES[@]}"; do
        name="${MODULE_NAMES[$mod]}"
        desc="${MODULE_DESCS[$mod]}"
        file="$CAPI_DIR/${mod}.zig"
        extract_exports "$file" "$name" "$desc"
    done

    # Also process ops submodule
    echo "## Ops"
    echo ""
    echo "Tensor operations"
    echo ""

    # Get all ops from ops/root.zig re-exports
    OPS_ROOT="$CAPI_DIR/ops/root.zig"
    if [[ -f "$OPS_ROOT" ]]; then
        echo "### Functions"
        echo ""

        # Extract function names from re-exports
        grep -E "^pub const talu_" "$OPS_ROOT" 2>/dev/null | \
            sed 's/pub const //' | \
            sed 's/ =.*//' | \
            while read -r func_name; do
                echo "- \`$func_name\`"
            done
        echo ""
    fi
fi

# Footer with error codes (only for full spec)
if [[ -z "$CATEGORY" ]]; then
    cat << 'EOF'

---

## Error Codes

All functions returning `i32` use these error codes:

| Code | Name | Description |
|------|------|-------------|
| 0 | Success | Operation completed successfully |
| 1 | InvalidArgument | Invalid parameter value |
| 2 | OutOfMemory | Memory allocation failed |
| 3 | IOError | File or network I/O error |
| 4 | NotFound | Resource not found |
| 5 | ParseError | Failed to parse input |
| 6 | InternalError | Internal error |

Use `talu_last_error()` to get detailed error message after a non-zero return.

## Memory Management

- **Handles** (`TokenizerHandle`, `ChatHandle`, etc.): Created by `*_create` functions, freed by `*_free` functions
- **Result structs**: Memory owned by the library until freed with corresponding `*_free` function
- **Strings**: `const char*` returns point to internal memory - copy if needed for longer lifetime

## Thread Safety

- Handles are NOT thread-safe - do not share across threads
- Read-only operations on shared data are safe
- Use separate handles for concurrent operations
EOF
fi
