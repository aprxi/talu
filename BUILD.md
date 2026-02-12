# BUILD.md

## Core (Zig)

    zig build release -Drelease           # build library + CLI + copy to Python
    zig build test -Drelease              # run unit tests
    zig build test-integration -Drelease  # run integration tests

See `core/POLICY.md` for test requirements and conventions.

## Bindings

### Python

Run from `bindings/python/`:

    uv sync                                              # install deps
    uv run pytest tests/<module>/                        # test a module
    uv run pytest tests/<module>/test_<file>.py          # test a file
    uv run pytest tests/ --ignore=tests/reference        # API tests
    uv run pytest tests/reference/                       # reference tests (vs PyTorch)
    uv run pytest tests/                                 # all tests (slow, use sparingly)

Match test scope to your changes. Changed `talu/chat/`? Run `tests/chat/`.

See `bindings/python/POLICY.md` for test requirements and conventions.
