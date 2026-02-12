"""Tests for talu.chat.spec - Model specification types.

This file consolidates spec-related tests. Implementation details are split across:
- test_spec_python.py: Python-level ModelSpec tests (same directory)
- core/test_spec_ffi.py: C API function tests
- tests/compat/abi/test_sizes.py: C ABI structure size validation (moved to compat)

This file imports those tests to satisfy the source-test mirroring requirement
(talu/chat/spec.py -> tests/chat/config/test_model_spec.py).
"""

# Re-export all tests from the detailed test files
# Note: ABI tests moved to tests/compat/abi/test_sizes.py for centralized validation
from tests.chat.config.test_spec_python import *  # noqa: F401, F403
from tests.router.test_spec import *  # noqa: F401, F403
