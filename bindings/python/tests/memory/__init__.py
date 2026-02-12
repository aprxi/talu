"""
Memory safety tests for FFI boundary.

Tests FFI memory ownership and lifecycle:
- Callback lifetime (segfault prevention)
- Buffer ownership (use-after-free prevention)
- Config field lifetimes (data corruption prevention)
- Result cleanup completeness (leak prevention)
- Error path cleanup (leak-on-failure prevention)
- Stress tests (slow leak detection)
"""
