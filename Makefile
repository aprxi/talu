.PHONY: all deps build core inference static cuda clean clean-deps test docs curl-build mlx-build mbedtls-build gen-bindings-python

WINDOWS := $(filter Windows_NT,$(OS))
CURDIR_POSIX := $(subst \,/,$(CURDIR))
WINDOWS_GIT_ROOT_NATIVE := $(ProgramFiles)\Git
WINDOWS_CMAKE_EXE :=
WINDOWS_LOCAL_ZIG := $(CURDIR_POSIX)/.tools/zig-x86_64-windows-0.15.2/zig.exe

ifeq ($(WINDOWS),Windows_NT)
WINDOWS_CMAKE_EXE := $(strip $(shell powershell -NoProfile -ExecutionPolicy Bypass -File "$(CURDIR)\ports\windows\find-cmake.ps1"))
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
SQLITE_TMP_DIR := $(if $(WINDOWS),$(CURDIR_POSIX)/.zig-cache/sqlite-dl,/tmp/sqlite-dl)

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
	MBEDTLS_ARCHIVE := deps/mbedtls/build/library/Release/mbedtls.lib
	CURL_ARCHIVE := deps/curl/build/lib/Release/libcurl.lib
	ifneq ($(WINDOWS_CMAKE_EXE),)
		CMAKE := $(WINDOWS_CMAKE_EXE)
	endif
else ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
	PYTHON_LIB_NAME := libtalu.dylib
	# macOS: no special CPU flags needed, native detection works
	ZIG_BUILD_FLAGS := -Drelease
	ZIG_CC_FLAGS := -fno-sanitize=all -DNDEBUG -O3
	# BSD sed requires '' after -i for in-place edit without backup
	SED_INPLACE := sed -i ''
	MBEDTLS_ARCHIVE := deps/mbedtls/build/library/libmbedtls.a
	CURL_ARCHIVE := deps/curl/build/lib/libcurl.a
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
	MBEDTLS_ARCHIVE := deps/mbedtls/build/library/libmbedtls.a
	CURL_ARCHIVE := deps/curl/build/lib/libcurl.a
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
	@test -d deps/curl || git clone --branch curl-8_17_0 --depth 1 https://github.com/curl/curl.git deps/curl
	@test -d deps/mbedtls || (git clone --branch v3.6.2 --depth 1 --recurse-submodules https://github.com/Mbed-TLS/mbedtls.git deps/mbedtls)
	@test -d deps/miniz || git clone --branch 3.1.1 --depth 1 https://github.com/richgel999/miniz.git deps/miniz
	@test -f deps/miniz/miniz_export.h || printf '#ifndef MINIZ_EXPORT\n#define MINIZ_EXPORT\n#endif\n' > deps/miniz/miniz_export.h
	@test -d deps/sqlite || (mkdir -p "$(SQLITE_TMP_DIR)" && \
		curl -sL "https://sqlite.org/2026/sqlite-amalgamation-3510200.zip" \
		-o "$(SQLITE_TMP_DIR)/sqlite.zip" && \
		unzip -qo "$(SQLITE_TMP_DIR)/sqlite.zip" -d "$(SQLITE_TMP_DIR)" && \
		mkdir -p deps/sqlite && \
		cp "$(SQLITE_TMP_DIR)"/sqlite-amalgamation-*/sqlite3.c \
		   "$(SQLITE_TMP_DIR)"/sqlite-amalgamation-*/sqlite3.h deps/sqlite/ && \
		rm -rf "$(SQLITE_TMP_DIR)")
	@test -f deps/cacert.pem || curl -sL https://curl.se/ca/cacert.pem -o deps/cacert.pem
	@printf '%s\n%s\n' '//! Mozilla CA certificates - auto-generated, do not edit' 'pub const data = @embedFile("cacert.pem");' > deps/cacert.zig
	@test -f $(MBEDTLS_ARCHIVE) || $(MAKE) mbedtls-build
	@test -f $(CURL_ARCHIVE) || $(MAKE) curl-build
ifeq ($(UNAME_S),Darwin)
	@{ test -f deps/mlx/lib/libmlx.a && test -f deps/mlx/lib/mlx.metallib; } || $(MAKE) mlx-build
endif

mbedtls-build:
	@echo "Building mbedTLS static library..."
	@rm -rf deps/mbedtls/build
	@mkdir -p deps/mbedtls/build
ifeq ($(UNAME_S),Windows)
	@cd deps/mbedtls/build && "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DENABLE_TESTING=OFF \
		-DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF
else ifeq ($(UNAME_S),Darwin)
	@cd deps/mbedtls/build && "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DENABLE_TESTING=OFF \
		-DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF
else
	@cd deps/mbedtls/build && \
	CC="$(ZIG) cc $(ZIG_CC_FLAGS)" "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DENABLE_TESTING=OFF \
		-DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF
endif
	@cd deps/mbedtls/build && "$(CMAKE)" --build . --config Release -j$(BUILD_JOBS)
	@echo "mbedTLS installed to deps/mbedtls/build/library/"

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

curl-build:
	@echo "Building libcurl with CMake (HTTP-only, minimal)..."
	@rm -rf deps/curl/build
	@mkdir -p deps/curl/build
ifeq ($(UNAME_S),Windows)
	@cd deps/curl/build && "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_CURL_EXE=OFF \
		-DBUILD_TESTING=OFF \
		-DHTTP_ONLY=ON \
		-DCURL_USE_OPENSSL=OFF \
		-DCURL_USE_MBEDTLS=ON \
		-DMBEDTLS_INCLUDE_DIRS=$(CURDIR_POSIX)/deps/mbedtls/include \
		-DMBEDTLS_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/Release/mbedtls.lib \
		-DMBEDX509_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/Release/mbedx509.lib \
		-DMBEDCRYPTO_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/Release/mbedcrypto.lib \
		-DCURL_USE_LIBPSL=OFF \
		-DCURL_USE_LIBSSH2=OFF \
		-DCURL_ZLIB=OFF \
		-DUSE_LIBIDN2=OFF \
		-DUSE_NGHTTP2=OFF \
		-DCURL_BROTLI=OFF \
		-DCURL_ZSTD=OFF \
		-DCURL_DISABLE_ALTSVC=ON \
		-DCURL_DISABLE_COOKIES=OFF \
		-DCURL_DISABLE_DOH=ON \
		-DCURL_DISABLE_GETOPTIONS=ON \
		-DCURL_DISABLE_HSTS=ON \
		-DCURL_DISABLE_MIME=ON \
		-DCURL_DISABLE_NETRC=ON \
		-DCURL_DISABLE_NTLM=ON \
		-DCURL_DISABLE_PROGRESS_METER=ON \
		-DCURL_DISABLE_PROXY=OFF \
		-DCURL_DISABLE_VERBOSE_STRINGS=ON \
		-DCURL_DISABLE_WEBSOCKETS=ON \
		-DCURL_DISABLE_IPFS=ON \
		-DCURL_DISABLE_FORM_API=ON \
		-DCURL_DISABLE_HEADERS_API=ON \
		-DCURL_DISABLE_BINDLOCAL=ON \
		-DCURL_DISABLE_DIGEST_AUTH=ON \
		-DCURL_DISABLE_BEARER_AUTH=ON \
		-DCURL_DISABLE_KERBEROS_AUTH=ON \
		-DCURL_DISABLE_NEGOTIATE_AUTH=ON \
		-DCURL_DISABLE_AWS=ON \
		-DCURL_DISABLE_SRP=ON
else ifeq ($(UNAME_S),Darwin)
	@cd deps/curl/build && "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_CURL_EXE=OFF \
		-DBUILD_TESTING=OFF \
		-DHTTP_ONLY=ON \
		-DCURL_USE_OPENSSL=OFF \
		-DCURL_USE_MBEDTLS=ON \
		-DMBEDTLS_INCLUDE_DIRS=$(CURDIR_POSIX)/deps/mbedtls/include \
		-DMBEDTLS_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedcrypto.a \
		-DCURL_USE_LIBPSL=OFF \
		-DCURL_USE_LIBSSH2=OFF \
		-DCURL_ZLIB=OFF \
		-DUSE_LIBIDN2=OFF \
		-DUSE_NGHTTP2=OFF \
		-DCURL_BROTLI=OFF \
		-DCURL_ZSTD=OFF \
		-DCURL_DISABLE_ALTSVC=ON \
		-DCURL_DISABLE_COOKIES=OFF \
		-DCURL_DISABLE_DOH=ON \
		-DCURL_DISABLE_GETOPTIONS=ON \
		-DCURL_DISABLE_HSTS=ON \
		-DCURL_DISABLE_MIME=ON \
		-DCURL_DISABLE_NETRC=ON \
		-DCURL_DISABLE_NTLM=ON \
		-DCURL_DISABLE_PROGRESS_METER=ON \
		-DCURL_DISABLE_PROXY=OFF \
		-DCURL_DISABLE_VERBOSE_STRINGS=ON \
		-DCURL_DISABLE_WEBSOCKETS=ON \
		-DCURL_DISABLE_IPFS=ON \
		-DCURL_DISABLE_FORM_API=ON \
		-DCURL_DISABLE_HEADERS_API=ON \
		-DCURL_DISABLE_BINDLOCAL=ON \
		-DCURL_DISABLE_DIGEST_AUTH=ON \
		-DCURL_DISABLE_BEARER_AUTH=ON \
		-DCURL_DISABLE_KERBEROS_AUTH=ON \
		-DCURL_DISABLE_NEGOTIATE_AUTH=ON \
		-DCURL_DISABLE_AWS=ON \
		-DCURL_DISABLE_SRP=ON
else
	@cd deps/curl/build && \
	CC="$(ZIG) cc $(ZIG_CC_FLAGS)" "$(CMAKE)" .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_CURL_EXE=OFF \
		-DBUILD_TESTING=OFF \
		-DHTTP_ONLY=ON \
		-DCURL_USE_OPENSSL=OFF \
		-DCURL_USE_MBEDTLS=ON \
		-DMBEDTLS_INCLUDE_DIRS=$(CURDIR_POSIX)/deps/mbedtls/include \
		-DMBEDTLS_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(CURDIR_POSIX)/deps/mbedtls/build/library/libmbedcrypto.a \
		-DCURL_USE_LIBPSL=OFF \
		-DCURL_USE_LIBSSH2=OFF \
		-DCURL_ZLIB=OFF \
		-DUSE_LIBIDN2=OFF \
		-DUSE_NGHTTP2=OFF \
		-DCURL_BROTLI=OFF \
		-DCURL_ZSTD=OFF \
		-DCURL_DISABLE_ALTSVC=ON \
		-DCURL_DISABLE_COOKIES=OFF \
		-DCURL_DISABLE_DOH=ON \
		-DCURL_DISABLE_GETOPTIONS=ON \
		-DCURL_DISABLE_HSTS=ON \
		-DCURL_DISABLE_MIME=ON \
		-DCURL_DISABLE_NETRC=ON \
		-DCURL_DISABLE_NTLM=ON \
		-DCURL_DISABLE_PROGRESS_METER=ON \
		-DCURL_DISABLE_PROXY=OFF \
		-DCURL_DISABLE_VERBOSE_STRINGS=ON \
		-DCURL_DISABLE_WEBSOCKETS=ON \
		-DCURL_DISABLE_IPFS=ON \
		-DCURL_DISABLE_FORM_API=ON \
		-DCURL_DISABLE_HEADERS_API=ON \
		-DCURL_DISABLE_BINDLOCAL=ON \
		-DCURL_DISABLE_DIGEST_AUTH=ON \
		-DCURL_DISABLE_BEARER_AUTH=ON \
		-DCURL_DISABLE_KERBEROS_AUTH=ON \
		-DCURL_DISABLE_NEGOTIATE_AUTH=ON \
		-DCURL_DISABLE_AWS=ON \
		-DCURL_DISABLE_SRP=ON
endif
	@cd deps/curl/build && "$(CMAKE)" --build . --config Release -j$(BUILD_JOBS)

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
	rm -rf deps/curl/build
	rm -rf deps/mlx-src/build
	rm -rf deps/mbedtls/build

clean-deps:
	rm -rf deps/
