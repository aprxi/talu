# CUDA Kernels Layout

- Build entrypoint: `kernels.cu`
- Implementation units: `ops/*.cu`
- Shared device helpers: `ops/*.cuh`

`kernels.cu` should remain a thin aggregation unit compiled by `zig build gen-cuda-kernels`.
Add new kernels in a focused `ops/` file and include it from `kernels.cu`.

This keeps symbol generation stable while preventing a single monolithic source file.
