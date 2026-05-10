# Windows build notes

This file records the generic steps and code changes needed to make `talu` build and run on Windows with `make`, while keeping the changes isolated so Linux and macOS behavior stays unchanged unless explicitly gated.

## Required tools

- GNU Make
- Git for Windows
  - `make` recipes use POSIX shell tools such as `rm`, `cp`, `mkdir -p`, and `test`
- Zig `0.15.2`
  - The codebase is currently pinned to this Zig line
- Rust toolchain / Cargo
  - Needed for the CLI build in `bindings/rust`
- CMake
  - Used to build `mbedtls` and `curl`

## Build steps

1. Use Zig `0.15.2` instead of Zig `0.16.x`.
2. Make sure `make` runs recipes through a POSIX shell on Windows.
3. Ensure `cmake.exe` is on PATH, or pass an explicit path with `CMAKE=/path/to/cmake`.
4. Build third-party dependencies with Windows archive names:
   - `deps/mbedtls/build/library/Release/*.lib`
   - `deps/curl/build/lib/Release/libcurl.lib`
5. Build the project with `make`.
6. For runtime testing in PowerShell, set environment variables with `$env:NAME = "value"`.

## Code changes kept for Windows support

- Added `core/src/env.zig` so code can read environment variables without relying on POSIX-only APIs.
- Added Windows-aware home-directory resolution for repository caches:
  - `HOME`
  - `USERPROFILE`
  - `HOMEDRIVE` + `HOMEPATH`
- Added Windows file mapping for safetensor reads.
- Added Windows build/link handling in `build.zig`:
  - correct static archive names
  - required Windows system libraries
  - Windows Python binding copy step
  - MSVC ABI preference on Windows
- Added Windows `Makefile` handling:
  - use Git for Windows `sh.exe` through PATH
  - use `CMAKE`/PATH for `cmake.exe`
  - use repo-local temp storage instead of `/tmp`
  - clean the generated Windows Python DLL
- Added Windows CUDA robustness fixes:
  - dynamic loading of cuBLAS/cuBLASLt from standard CUDA locations
  - FP8 KV cache fallback on unsupported GPUs
  - NVFP4 safeguards for Windows pre-SM89 GPUs

## Runtime notes

PowerShell does not support Unix-style inline env assignment. Use:

```powershell
$env:BACKEND = "cpu"
echo helo | .\zig-out\bin\talu.exe -m Qwen/Qwen3.5-0.8B
```

For CUDA:

```powershell
$env:BACKEND = "cuda"
echo helo | .\zig-out\bin\talu.exe -m Qwen/Qwen3.5-2B-NVFP4
```

If CUDA DLLs are installed in a nonstandard location, set:

```powershell
$env:TALU_CUDA_DLL_DIR = "C:\path\to\cuda\bin"
```

## Scope guardrails

- Keep Windows-only behavior behind platform checks.
- Avoid hardcoded user paths, machine names, or repo-local bootstrap binaries.
- Prefer host-platform detection for build helper commands.
- Do not change Linux or macOS behavior unless the same path is clearly safer everywhere.
