.PHONY: all deps build static cuda clean clean-deps test docs curl-build mlx-build mbedtls-build freetype-build pdfium-build gen-bindings ui

# Detect platform-specific settings
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
	LIB_EXT := dylib
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
	@test -d deps/miniz || git clone --branch 3.1.1 --depth 1 https://github.com/richgel999/miniz.git deps/miniz
	@test -f deps/miniz/miniz_export.h || printf '#ifndef MINIZ_EXPORT\n#define MINIZ_EXPORT\n#endif\n' > deps/miniz/miniz_export.h
	@test -d deps/file || git clone --branch FILE5_46 --depth 1 https://github.com/file/file.git deps/file
	@test -d deps/jpeg-turbo || git clone --branch 3.1.3 --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo.git deps/jpeg-turbo
	@test -d deps/spng || git clone --branch v0.7.4 --depth 1 https://github.com/randy408/libspng.git deps/spng
	@test -d deps/webp || git clone --branch v1.6.0 --depth 1 https://github.com/webmproject/libwebp.git deps/webp
	@test -f deps/file/src/magic.h || sed 's/X\.YY/5.46/' deps/file/src/magic.h.in > deps/file/src/magic.h
	@test -f deps/file/magic.mgc || cp /usr/share/file/magic.mgc deps/file/magic.mgc
	@test -f deps/cacert.pem || curl -sL https://curl.se/ca/cacert.pem -o deps/cacert.pem
	@printf '%s\n%s\n' '//! Mozilla CA certificates - auto-generated, do not edit' 'pub const data = @embedFile("cacert.pem");' > deps/cacert.zig
	@printf '%s\n%s\n' '//! Compiled magic database - auto-generated, do not edit' 'pub const data = @embedFile("file/magic.mgc");' > deps/magic_db.zig
	@test -d deps/freetype || git clone --branch VER-2-13-3 --depth 1 https://github.com/freetype/freetype.git deps/freetype
	@test -d deps/pdfium || git clone --depth 1 https://pdfium.googlesource.com/pdfium deps/pdfium
	@test -d deps/pdfium/third_party/fast_float/src || \
		git clone --branch v8.2.3 --depth 1 https://github.com/fastfloat/fast_float.git deps/pdfium/third_party/fast_float/src
	@# Tree-sitter core runtime + language grammars
	@test -d deps/tree-sitter || git clone --branch v0.26.5 --depth 1 https://github.com/tree-sitter/tree-sitter.git deps/tree-sitter
	@test -d deps/tree-sitter-python || git clone --branch v0.23.6 --depth 1 https://github.com/tree-sitter/tree-sitter-python.git deps/tree-sitter-python
	@test -d deps/tree-sitter-javascript || git clone --branch v0.23.1 --depth 1 https://github.com/tree-sitter/tree-sitter-javascript.git deps/tree-sitter-javascript
	@test -d deps/tree-sitter-typescript || git clone --branch v0.23.2 --depth 1 https://github.com/tree-sitter/tree-sitter-typescript.git deps/tree-sitter-typescript
	@test -d deps/tree-sitter-rust || git clone --branch v0.23.2 --depth 1 https://github.com/tree-sitter/tree-sitter-rust.git deps/tree-sitter-rust
	@test -d deps/tree-sitter-go || git clone --branch v0.23.4 --depth 1 https://github.com/tree-sitter/tree-sitter-go.git deps/tree-sitter-go
	@test -d deps/tree-sitter-c || git clone --branch v0.23.5 --depth 1 https://github.com/tree-sitter/tree-sitter-c.git deps/tree-sitter-c
	@test -d deps/tree-sitter-zig || git clone --branch v1.1.2 --depth 1 https://github.com/tree-sitter-grammars/tree-sitter-zig.git deps/tree-sitter-zig
	@test -d deps/tree-sitter-json || git clone --branch v0.24.8 --depth 1 https://github.com/tree-sitter/tree-sitter-json.git deps/tree-sitter-json
	@test -d deps/tree-sitter-html || git clone --branch v0.23.2 --depth 1 https://github.com/tree-sitter/tree-sitter-html.git deps/tree-sitter-html
	@test -d deps/tree-sitter-css || git clone --branch v0.23.1 --depth 1 https://github.com/tree-sitter/tree-sitter-css.git deps/tree-sitter-css
	@test -d deps/tree-sitter-bash || git clone --branch v0.23.3 --depth 1 https://github.com/tree-sitter/tree-sitter-bash.git deps/tree-sitter-bash
	@test -f deps/mbedtls/build/library/libmbedtls.a || $(MAKE) mbedtls-build
	@test -f deps/curl/build/lib/libcurl.a || $(MAKE) curl-build
	@test -f deps/freetype/build/libfreetype.a || $(MAKE) freetype-build
	@test -f deps/pdfium/cmake-build/libpdfium.a || $(MAKE) pdfium-build
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

freetype-build:
	@echo "Building FreeType static library..."
	@rm -rf deps/freetype/build
	@mkdir -p deps/freetype/build
ifeq ($(UNAME_S),Darwin)
	@cd deps/freetype/build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DFT_DISABLE_BZIP2=ON \
		-DFT_DISABLE_BROTLI=ON \
		-DFT_DISABLE_HARFBUZZ=ON \
		-DFT_DISABLE_PNG=ON \
		-DFT_DISABLE_ZLIB=ON
else
	@cd deps/freetype/build && \
	CC="zig cc $(ZIG_CC_FLAGS)" cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DFT_DISABLE_BZIP2=ON \
		-DFT_DISABLE_BROTLI=ON \
		-DFT_DISABLE_HARFBUZZ=ON \
		-DFT_DISABLE_PNG=ON \
		-DFT_DISABLE_ZLIB=ON
endif
	@cd deps/freetype/build && cmake --build . --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)
	@echo "FreeType installed to deps/freetype/build/"

pdfium-build: freetype-build
	@echo "Building PDFium static library..."
	# Copy our CMake build and shim headers into PDFium source tree
	@cp ports/pdfium/CMakeLists.txt deps/pdfium/
	@mkdir -p deps/pdfium/build
	@cp ports/pdfium/buildflag.h deps/pdfium/build/
	@cp ports/pdfium/build_config.h deps/pdfium/build/
	# Patch out abseil dependency (2 files)
	@cd deps/pdfium && $(SED_INPLACE) \
		-e 's|#include "third_party/abseil-cpp/absl/container/inlined_vector.h"|#include <vector>|' \
		-e 's|absl::InlinedVector<float, 16, FxAllocAllocator<float>>|std::vector<float, FxAllocAllocator<float>>|' \
		-e 's|absl::InlinedVector<uint32_t, 16, FxAllocAllocator<uint32_t>>|std::vector<uint32_t, FxAllocAllocator<uint32_t>>|' \
		core/fpdfapi/page/cpdf_sampledfunc.cpp
	@cd deps/pdfium && $(SED_INPLACE) \
		-e 's|#include "third_party/abseil-cpp/absl/container/flat_hash_set.h"|#include <unordered_set>|' \
		-e 's|absl::flat_hash_set|std::unordered_set|g' \
		core/fpdfdoc/cpdf_nametree.cpp
	# Build with zig cc/c++ for glibc 2.28 compatibility
	@rm -rf deps/pdfium/cmake-build
	@mkdir -p deps/pdfium/cmake-build
ifeq ($(UNAME_S),Darwin)
	@cd deps/pdfium/cmake-build && cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DFREETYPE_INCLUDE_DIR=$(CURDIR)/deps/freetype/include \
		-DFREETYPE_LIBRARY=$(CURDIR)/deps/freetype/build/libfreetype.a \
		-DJPEG_INCLUDE_DIR="$(CURDIR)/deps/jpeg-turbo/src;$(CURDIR)/ports/pdfium/jpeg_compat" \
		-DICU_INCLUDE_DIR=$(CURDIR)/ports/pdfium/icu_stubs \
		-DZLIB_INCLUDE_DIR=$(CURDIR)/ports/pdfium/zlib_compat
else
	@cd deps/pdfium/cmake-build && \
	CC="zig cc $(ZIG_CC_FLAGS)" CXX="zig c++ $(ZIG_CC_FLAGS)" cmake .. \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DFREETYPE_INCLUDE_DIR=$(CURDIR)/deps/freetype/include \
		-DFREETYPE_LIBRARY=$(CURDIR)/deps/freetype/build/libfreetype.a \
		-DJPEG_INCLUDE_DIR="$(CURDIR)/deps/jpeg-turbo/src;$(CURDIR)/ports/pdfium/jpeg_compat" \
		-DICU_INCLUDE_DIR=$(CURDIR)/ports/pdfium/icu_stubs \
		-DZLIB_INCLUDE_DIR=$(CURDIR)/ports/pdfium/zlib_compat
endif
	@cd deps/pdfium/cmake-build && cmake --build . --config Release -j$$(nproc 2>/dev/null || sysctl -n hw.ncpu)
	@echo "PDFium installed to deps/pdfium/cmake-build/"

build: deps sync-version gen-bindings ui
	zig build release $(ZIG_BUILD_FLAGS)

cuda: deps sync-version gen-bindings ui
	@if [ -z "$(CUDA_NVCC)" ]; then \
		echo "Error: nvcc not found. Install CUDA toolkit or add nvcc to PATH." >&2; \
		exit 1; \
	fi
	PATH="$(CUDA_BIN_DIR):$$PATH" zig build gen-cuda-kernels $(CUDA_BUILD_FLAGS)
	PATH="$(CUDA_BIN_DIR):$$PATH" zig build release $(CUDA_BUILD_FLAGS)

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
