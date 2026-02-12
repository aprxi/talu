.PHONY: format test test-reference test-unit build sync sync-version clean

# Detect platform-specific library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
else
	LIB_EXT := so
endif

# Path to Zig build output (relative to repo root)
ZIG_OUT := ../../zig-out/lib
BUILD_TS ?= $(shell date +%Y%m%d%H%M)
VERSION_BASE := $(shell cat ../../VERSION)
VERSION_SEMVER := $(VERSION_BASE)-post.$(BUILD_TS)
VERSION_PYTHON := $(VERSION_BASE).post$(BUILD_TS)
UV_BUILD_PYTHON := $(shell command -v python3 2>/dev/null)
ifeq ($(UV_BUILD_PYTHON),)
UV_BUILD_PYTHON := python3
endif

# =============================================================================
# Environment Configuration
# =============================================================================
# Store venv and caches outside project directory. This allows:
# - Read-only mounts
# - Clean project directory (no .venv/.ruff_cache/.pytest_cache clutter)
export UV_PROJECT_ENVIRONMENT := $(HOME)/.cache/uv/venvs/talu-python
export RUFF_CACHE_DIR := $(HOME)/.cache/ruff

# Pytest cache dir (passed via --override-ini, no env var support)
PYTEST_CACHE := --override-ini="cache_dir=$(HOME)/.cache/pytest"

all: build

# =============================================================================
# Lint (Static Quality)
# =============================================================================

lint:
	uv run ruff check talu/ tests/
	uv run ruff format --check talu/ tests/
	uv run pyright talu/

format:
	uv run ruff format talu/ tests/
	uv run ruff check --fix talu/ tests/

# =============================================================================
# Testing
# =============================================================================

# -n 4: run tests in parallel (4 workers)
PYTEST_OPTS := -q -n 4 $(PYTEST_CACHE)

test:
	TALU_LOG_LEVEL=off uv run pytest tests/ --ignore=tests/compat $(PYTEST_OPTS)

test-reference:
	TALU_LOG_LEVEL=off uv run pytest tests/reference/ $(PYTEST_OPTS)

test-api:
	TALU_LOG_LEVEL=off uv run pytest tests/ --ignore=tests/compat --ignore=tests/reference $(PYTEST_OPTS)

# =============================================================================
# Build and Package
# =============================================================================

# Sync dependencies
sync:
	uv sync

# Stamp local package versions as post releases.
# This keeps local wheels distinct from official tagged versions.
sync-version:
	@mkdir -p vendor/talu vendor/talu/bindings/rust/vendor
	@printf '%s\n' "$(VERSION_SEMVER)" > vendor/talu/VERSION
	@printf '%s\n' "$(VERSION_PYTHON)" > vendor/talu/VERSION_PYTHON
	@printf '%s\n' "$(VERSION_SEMVER)" > vendor/talu/bindings/rust/vendor/VERSION
	@printf '"""Version from VERSION file."""\n\n__version__ = "%s"\n' "$(VERSION_PYTHON)" > talu/_version.py
	@echo "Stamped local package version: $(VERSION_PYTHON)"

# Build wheel for distribution (bundles the native library)
build:
	@if [ ! -f $(ZIG_OUT)/libtalu.$(LIB_EXT) ]; then \
		echo "Native library missing at $(ZIG_OUT)/libtalu.$(LIB_EXT)"; \
		echo "Bootstrapping from repository root..."; \
		$(MAKE) -C ../..; \
	fi
	@if [ ! -f $(ZIG_OUT)/libtalu.$(LIB_EXT) ]; then \
		echo "Error: Native library still missing at $(ZIG_OUT)/libtalu.$(LIB_EXT)"; \
		echo "Run 'make' from the repository root and resolve build errors."; \
		exit 1; \
	fi
	@$(MAKE) sync-version BUILD_TS=$(BUILD_TS)
	cp $(ZIG_OUT)/libtalu.$(LIB_EXT) talu/
	@echo "Validating ABI compatibility..."
	TALU_LOG_LEVEL=off uv run pytest tests/compat/abi/ -q $(PYTEST_CACHE)
	uv build --python $(UV_BUILD_PYTHON) --sdist --wheel --out-dir dist

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
	rm -rf talu/__pycache__ tests/__pycache__
	rm -rf .coverage htmlcov dist
	rm -f talu/libtalu.so talu/libtalu.dylib talu/libtalu.dll
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
