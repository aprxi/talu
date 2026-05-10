.PHONY: all deps build core inference static cuda clean clean-deps test docs mlx-build gen-bindings-python

WINDOWS := $(filter Windows_NT,$(OS))
CURDIR_POSIX := $(subst \,/,$(CURDIR))
WINDOWS_GIT_ROOT_NATIVE := $(ProgramFiles)\Git
WINDOWS_LOCAL_ZIG := $(CURDIR_POSIX)/.tools/zig-x86_64-windows-0.15.2/zig.exe

ifeq ($(WINDOWS),Windows_NT)
export PATH := $(WINDOWS_GIT_ROOT_NATIVE)\usr\bin;$(WINDOWS_GIT_ROOT_NATIVE)\bin;$(WINDOWS_GIT_ROOT_NATIVE)\cmd;$(PATH)
SHELL := sh.exe
UNAME_S := Windows
UNAME_M := x86_64
else
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
endif

ZIG ?= zig
CMAKE ?= cmake
BUILD_JOBS := $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

ifeq ($(WINDOWS),Windows_NT)
ifneq ($(wildcard $(WINDOWS_LOCAL_ZIG)),)
ZIG := $(WINDOWS_LOCAL_ZIG)
endif
endif

# Detect platform-specific settings
MLX_VERSION ?= v0.31.1
MLX_REPO ?= https://github.com/ml-explore/mlx.git

ifeq ($(WINDOWS),Windows_NT)
	LIB_EXT := dll
	PYTHON_LIB_NAME := talu.dll
	ZIG_BUILD_FLAGS := -Drelease -Dcpu=x86_64_v3
	ZIG_CC_FLAGS :=
	SED_INPLACE := sed -i
else ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
	PYTHON_LIB_NAME := libtalu.dylib
	# macOS: no special CPU flags needed, native detection works
	ZIG_BUILD_FLAGS := -Drelease
	ZIG_CC_FLAGS := -fno-sanitize=all -DNDEBUG -O3
	# BSD sed requires '' after -i for in-place edit without backup
	SED_INPLACE := sed -i ''
	# Metal GPU is enabled by default. Set TALU_DISABLE_METAL=1 for CPU-only builds.
	ifdef TALU_DISABLE_METAL
		ZIG_BUILD_FLAGS += -Dmetal=false
	endif
else
	LIB_EXT := so
	PYTHON_LIB_NAME := libtalu.so
	# Linux x86_64: target GLIBC 2.28 for distro compatibility (RHEL 8, Ubuntu 18.04, etc.)
	# Use x86_64_v3 for AVX2 support (fixes register allocation with SIMD code)
	ZIG_BUILD_FLAGS := -Drelease -Dcpu=x86_64_v3
	ZIG_CC_FLAGS := -target x86_64-linux-gnu.2.28 -mcpu=x86_64_v3 -fno-sanitize=all -DNDEBUG -O3
	# GNU sed uses -i without argument
	SED_INPLACE := sed -i
endif

ifdef TALU_ENABLE_CUDA
	ZIG_BUILD_FLAGS += -Dcuda=true
endif
ifdef TALU_CUDA_STARTUP_SELFTESTS
	ZIG_BUILD_FLAGS += -Dcuda-startup-selftests=true
endif

CUDA_BUILD_FLAGS := $(ZIG_BUILD_FLAGS)
ifeq ($(findstring -Dcuda=true,$(CUDA_BUILD_FLAGS)),)
	CUDA_BUILD_FLAGS += -Dcuda=true
endif

CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
ifneq ("$(wildcard /usr/local/cuda/bin/nvcc)","")
	CUDA_NVCC := /usr/local/cuda/bin/nvcc
endif
endif

CUDA_BIN_DIR := $(dir $(CUDA_NVCC))

all: build

check-zig-version:
	@actual="$$("$(ZIG)" version)"; \
	if [ "$$actual" != "0.15.2" ]; then \
		echo "Error: talu currently requires Zig 0.15.2, found $$actual at $(ZIG)." >&2; \
		echo "Set ZIG=/path/to/zig-0.15.2/zig or place zig 0.15.2 at .tools/zig-x86_64-windows-0.15.2/zig.exe." >&2; \
		exit 1; \
	fi

deps: check-zig-version
	@test -d deps/utf8proc || git clone --branch v2.11.2 --depth 1 https://github.com/JuliaStrings/utf8proc.git deps/utf8proc
	@test -f deps/pcre2/deps/sljit/sljit_src/sljitLir.c || (rm -rf deps/pcre2 && git clone --branch pcre2-10.47 --depth 1 --recurse-submodules https://github.com/PCRE2Project/pcre2.git deps/pcre2)
	@test -f deps/cacert.pem || curl -sL https://curl.se/ca/cacert.pem -o deps/cacert.pem
	@printf '%s\n%s\n' '//! Mozilla CA certificates - auto-generated, do not edit' 'pub const data = @embedFile("cacert.pem");' > deps/cacert.zig
ifeq ($(UNAME_S),Darwin)
	@{ test -f deps/mlx/lib/libmlx.a && test -f deps/mlx/lib/mlx.metallib; } || $(MAKE) mlx-build
endif

mlx-build:
	@echo "Building MLX static library..."
	@if [ ! -d deps/mlx-src/.git ]; then \
		git clone --branch $(MLX_VERSION) --depth 1 $(MLX_REPO) deps/mlx-src; \
	else \
		cd deps/mlx-src && \
		git fetch origin --tags --force && \
		git checkout -f $(MLX_VERSION) && \
		git reset --hard $(MLX_VERSION); \
	fi
	@rm -rf deps/mlx-src/build
	@mkdir -p deps/mlx-src/build
	@cd deps/mlx-src/build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_SHARED_LIBS=OFF \
		-DMLX_BUILD_TESTS=OFF \
		-DMLX_BUILD_EXAMPLES=OFF \
		-DMLX_BUILD_BENCHMARKS=OFF \
		-DMLX_BUILD_PYTHON_BINDINGS=OFF \
		-DMLX_BUILD_METAL=ON \
		-DMLX_BUILD_CPU=ON \
		-DMLX_METAL_JIT=ON
	@cd deps/mlx-src/build && cmake --build . --config Release -j$$(sysctl -n hw.ncpu)
	@echo "Installing MLX to deps/mlx..."
	@mkdir -p deps/mlx/lib deps/mlx/include
	@cp deps/mlx-src/build/libmlx.a deps/mlx/lib/
	@cp deps/mlx-src/build/mlx/backend/metal/kernels/mlx.metallib deps/mlx/lib/
	@rm -rf deps/mlx/include/mlx
	@cp -r deps/mlx-src/mlx deps/mlx/include/
	@echo "MLX $(MLX_VERSION) (JIT mode) installed to deps/mlx/"

build: deps sync-version
	$(ZIG) build release $(ZIG_BUILD_FLAGS)

core: deps sync-version
	$(ZIG) build core $(ZIG_BUILD_FLAGS)

inference: deps sync-version
	$(ZIG) build inference $(ZIG_BUILD_FLAGS)

cuda: deps sync-version
	@if [ -z "$(CUDA_NVCC)" ]; then \
		echo "Error: nvcc not found. Install CUDA toolkit or add nvcc to PATH." >&2; \
		exit 1; \
	fi
	PATH="$(CUDA_BIN_DIR):$$PATH" $(ZIG) build gen-cuda-kernels $(CUDA_BUILD_FLAGS)
	PATH="$(CUDA_BIN_DIR):$$PATH" $(ZIG) build release $(CUDA_BUILD_FLAGS)

# Generate Python ctypes bindings from Zig C API
gen-bindings-python: check-zig-version
	$(ZIG) build gen-bindings-python -Drelease
	$(ZIG) build gen-bindings-rust -Drelease

# Sync VERSION to binding vendor directories
sync-version:
	@mkdir -p bindings/python/vendor/talu bindings/rust/vendor
	@cp VERSION bindings/rust/vendor/

static: deps
	$(ZIG) build static $(ZIG_BUILD_FLAGS)

test: deps
	TALU_LOG_LEVEL=off $(ZIG) build test $(ZIG_BUILD_FLAGS)
	TALU_LOG_LEVEL=off $(ZIG) build test-integration $(ZIG_BUILD_FLAGS)
	./core/tests/check_signal_tests.sh

docs:
	cd docs && uv run python scripts/build.py

clean:
	rm -rf zig-out .zig-cache .zig-cache-global
	cargo clean --manifest-path bindings/rust/Cargo.toml 2>/dev/null || true
	rm -f bindings/python/talu/$(PYTHON_LIB_NAME)
	rm -rf docs/dist .venv
	rm -rf .pytest_cache .ruff_cache
	rm -rf deps/mlx-src/build

clean-deps:
	rm -rf deps/
