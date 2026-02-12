.PHONY: all deps build static clean clean-deps test graphs docs curl-build mlx-build mbedtls-build gen-bindings ui

# Detect platform-specific settings
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
	# macOS: no special CPU flags needed, native detection works
	ZIG_BUILD_FLAGS := -Drelease
	ZIG_CC_FLAGS := -fno-sanitize=all -DNDEBUG -O3
	# Metal GPU is enabled by default. Set TALU_DISABLE_METAL=1 for CPU-only builds.
	ifdef TALU_DISABLE_METAL
		ZIG_BUILD_FLAGS += -Dmetal=false
	endif
else
	LIB_EXT := so
	# Linux x86_64: target GLIBC 2.28 for distro compatibility (RHEL 8, Ubuntu 18.04, etc.)
	# Use x86_64_v3 for AVX2 support (fixes register allocation with SIMD code)
	ZIG_BUILD_FLAGS := -Drelease -Dcpu=x86_64_v3
	ZIG_CC_FLAGS := -target x86_64-linux-gnu.2.28 -mcpu=x86_64_v3 -fno-sanitize=all -DNDEBUG -O3
endif

all: build

deps:
	# macOS: requires `brew install cmake` (Xcode Command Line Tools for Metal)
	@test -d deps/utf8proc || git clone --branch v2.11.2 --depth 1 https://github.com/JuliaStrings/utf8proc.git deps/utf8proc
	@test -d deps/pcre2 || git clone --branch pcre2-10.47 --depth 1 https://github.com/PCRE2Project/pcre2.git deps/pcre2
	@test -d deps/sqlite || (mkdir -p /tmp/sqlite-dl && \
		curl -sL "https://sqlite.org/2026/sqlite-amalgamation-3510200.zip" \
		-o /tmp/sqlite-dl/sqlite.zip && \
		unzip -qo /tmp/sqlite-dl/sqlite.zip -d /tmp/sqlite-dl && \
		mkdir -p deps/sqlite && \
		cp /tmp/sqlite-dl/sqlite-amalgamation-*/sqlite3.c \
		   /tmp/sqlite-dl/sqlite-amalgamation-*/sqlite3.h deps/sqlite/ && \
		rm -rf /tmp/sqlite-dl)
	@test -d deps/curl || git clone --branch curl-8_17_0 --depth 1 https://github.com/curl/curl.git deps/curl
	@test -d deps/mbedtls || (git clone --branch v3.6.2 --depth 1 --recurse-submodules https://github.com/Mbed-TLS/mbedtls.git deps/mbedtls)
	@test -f deps/cacert.pem || curl -sL https://curl.se/ca/cacert.pem -o deps/cacert.pem
	@printf '%s\n%s\n' '//! Mozilla CA certificates - auto-generated, do not edit' 'pub const data = @embedFile("cacert.pem");' > deps/cacert.zig
	@test -f deps/mbedtls/build/library/libmbedtls.a || $(MAKE) mbedtls-build
	@test -f deps/curl/build/lib/libcurl.a || $(MAKE) curl-build
ifeq ($(UNAME_S),Darwin)
	@test -f deps/mlx/lib/libmlx.a || $(MAKE) mlx-build
endif

mbedtls-build:
	@echo "Building mbedTLS static library (GLIBC 2.28 compat)..."
	@rm -rf deps/mbedtls/build
	@mkdir -p deps/mbedtls/build
ifeq ($(UNAME_S),Darwin)
	@cd deps/mbedtls/build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DENABLE_TESTING=OFF \
		-DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF
else
	@cd deps/mbedtls/build && \
	CC="zig cc $(ZIG_CC_FLAGS)" cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DENABLE_TESTING=OFF \
		-DENABLE_PROGRAMS=OFF \
		-DMBEDTLS_FATAL_WARNINGS=OFF
endif
	@cd deps/mbedtls/build && cmake --build . --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)
	@echo "mbedTLS installed to deps/mbedtls/build/library/"

mlx-build:
	@echo "Building MLX static library..."
	@test -d deps/mlx-src || git clone --branch v0.30.1 --depth 1 https://github.com/ml-explore/mlx.git deps/mlx-src
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
	@rm -rf deps/mlx/include/mlx
	@cp -r deps/mlx-src/mlx deps/mlx/include/
	@echo "MLX v0.30.1 (JIT mode) installed to deps/mlx/"

curl-build:
	@echo "Building libcurl with CMake (HTTP-only, minimal)..."
	@rm -rf deps/curl/build
	@mkdir -p deps/curl/build
ifeq ($(UNAME_S),Darwin)
	@cd deps/curl/build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_CURL_EXE=OFF \
		-DBUILD_TESTING=OFF \
		-DHTTP_ONLY=ON \
		-DCURL_USE_OPENSSL=OFF \
		-DCURL_USE_MBEDTLS=ON \
		-DMBEDTLS_INCLUDE_DIRS=$(CURDIR)/deps/mbedtls/include \
		-DMBEDTLS_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedcrypto.a \
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
	CC="zig cc $(ZIG_CC_FLAGS)" cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DBUILD_CURL_EXE=OFF \
		-DBUILD_TESTING=OFF \
		-DHTTP_ONLY=ON \
		-DCURL_USE_OPENSSL=OFF \
		-DCURL_USE_MBEDTLS=ON \
		-DMBEDTLS_INCLUDE_DIRS=$(CURDIR)/deps/mbedtls/include \
		-DMBEDTLS_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedtls.a \
		-DMBEDX509_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedx509.a \
		-DMBEDCRYPTO_LIBRARY=$(CURDIR)/deps/mbedtls/build/library/libmbedcrypto.a \
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
	@cd deps/curl/build && cmake --build . --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)

build: deps sync-version gen-bindings ui
	zig build release $(ZIG_BUILD_FLAGS)

# Generate Python ctypes bindings from Zig C API
gen-bindings:
	zig build gen-bindings -Drelease

# Sync VERSION to binding vendor directories
sync-version:
	@mkdir -p bindings/python/vendor/talu bindings/rust/vendor
	@cp VERSION bindings/rust/vendor/

static: deps
	zig build static $(ZIG_BUILD_FLAGS)

test: deps
	TALU_LOG_LEVEL=off zig build test $(ZIG_BUILD_FLAGS)
	TALU_LOG_LEVEL=off zig build test-integration $(ZIG_BUILD_FLAGS)
	./core/tests/check_signal_tests.sh


# Compute graph files in tools/archs/_graphs/
# Generated from readable Python models, embedded into binary at build time
GRAPHS_DIR = tools/archs/_graphs

graphs:
	@echo "Generating graph files from Python models..."
	cd tools/archs && uv run python -m trace --all
	@echo ""
	@ls -la $(GRAPHS_DIR)/*.json

docs:
	cd docs && uv run python scripts/build.py

ui:
	@if [ -f ui/dist/index.html ]; then \
		echo "Using pre-built UI assets"; \
	elif command -v bun >/dev/null 2>&1; then \
		cd ui && bun install && bun run build; \
	else \
		echo "Error: bun is required to build UI. Install from https://bun.sh" >&2; exit 1; \
	fi

clean:
	rm -rf zig-out .zig-cache .zig-cache-global
	rm -f bindings/python/talu/libtalu.$(LIB_EXT)
	rm -rf docs/dist .venv
	rm -rf .pytest_cache .ruff_cache
	rm -rf deps/curl/build
	rm -rf deps/mlx-src/build
	rm -rf deps/mbedtls/build
	rm -rf ui/node_modules ui/dist

clean-deps:
	rm -rf deps/
