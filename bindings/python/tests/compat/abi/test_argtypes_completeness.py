"""
Tests for C API binding completeness.

This module ensures that all C API functions used in the Python bindings have
proper argtypes/restype configured. Missing argtypes can cause silent pointer
corruption on 64-bit systems.

This test acts as a safety net - if a new C function is added to Zig and used
in Python without adding argtypes, this test will catch it.
"""

import ctypes
import re

from talu._bindings import get_lib as get_chat_lib


def _ensure_all_bindings_loaded(lib) -> None:
    """Load all binding modules to ensure argtypes are configured.

    Note: Since argtypes are now set up automatically by _native.py at library
    load time, this function no longer needs to call setup functions.
    It's kept for consistency with the test structure.
    """
    # Argtypes are configured automatically by _native.py when the library loads.
    # No additional imports needed - all structs are defined in _native.py.
    pass


def get_all_talu_function_calls_in_module(module_path: str) -> set[str]:
    """Extract all talu_* function calls from a Python source file."""
    with open(module_path) as f:
        content = f.read()

    # Match patterns like: self._lib.talu_xxx( or lib.talu_xxx(
    pattern = r"(?:self\._lib|lib)\.(\w+)\s*\("
    matches = re.findall(pattern, content)

    # Filter to only talu_* functions
    return {m for m in matches if m.startswith("talu_")}


class TestCAPIBindingsComplete:
    """Verify all used C API functions have argtypes configured."""

    def test_session_py_functions_have_argtypes(self):
        """All talu_* functions used in session.py must have argtypes."""
        import talu.chat.session as session_module

        module_path = session_module.__file__

        used_functions = get_all_talu_function_calls_in_module(module_path)
        lib = get_chat_lib()
        _ensure_all_bindings_loaded(lib)

        missing_argtypes = []
        for func_name in used_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                if not hasattr(func, "argtypes") or func.argtypes is None:
                    missing_argtypes.append(func_name)

        assert not missing_argtypes, (
            f"The following C functions are used in session.py but have no argtypes configured:\n"
            f"  {missing_argtypes}\n"
            f"Add argtypes in talu/chat/items.py or talu/chat/_bindings.py"
        )

    def test_items_py_functions_have_argtypes(self):
        """All talu_* functions used in items.py must have argtypes."""
        import talu.chat.items as items_module

        module_path = items_module.__file__

        used_functions = get_all_talu_function_calls_in_module(module_path)
        lib = get_chat_lib()

        missing_argtypes = []
        for func_name in used_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                if not hasattr(func, "argtypes") or func.argtypes is None:
                    missing_argtypes.append(func_name)

        assert not missing_argtypes, (
            f"The following C functions are used in items.py but have no argtypes configured:\n"
            f"  {missing_argtypes}\n"
            f"Add argtypes in talu/chat/items.py"
        )

    def test_router_py_functions_have_argtypes(self):
        """All talu_* functions used in router.py must have argtypes."""
        import talu.router.router as router_module

        module_path = router_module.__file__

        used_functions = get_all_talu_function_calls_in_module(module_path)
        lib = get_chat_lib()

        missing_argtypes = []
        for func_name in used_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                if not hasattr(func, "argtypes") or func.argtypes is None:
                    missing_argtypes.append(func_name)

        assert not missing_argtypes, (
            f"The following C functions are used in router.py but have no argtypes configured:\n"
            f"  {missing_argtypes}\n"
            f"Add argtypes in talu/chat/_bindings.py"
        )


class TestKnownCriticalFunctions:
    """Explicitly test that known critical functions have correct signatures."""

    def test_pointer_returning_functions_return_void_p(self):
        """Functions returning pointers must have c_void_p restype."""
        lib = get_chat_lib()

        pointer_functions = [
            "talu_chat_create",
            "talu_chat_create_with_system",
            "talu_chat_create_with_session",
            "talu_chat_create_with_system_and_session",
            "talu_chat_get_conversation",
            "talu_chat_get_messages",
            "talu_chat_get_system",
        ]

        for func_name in pointer_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                assert func.restype == ctypes.c_void_p, (
                    f"{func_name} should return c_void_p but returns {func.restype}"
                )

    def test_hidden_append_functions_have_bool_arg(self):
        """Functions with hidden flag must have c_bool in argtypes."""
        lib = get_chat_lib()

        hidden_functions = [
            ("talu_responses_append_message_hidden", 4),  # 5th arg (index 4)
            ("talu_responses_insert_message_hidden", 5),  # 6th arg (index 5)
        ]

        for func_name, bool_index in hidden_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                assert func.argtypes is not None, f"{func_name} has no argtypes"
                assert len(func.argtypes) > bool_index, (
                    f"{func_name} argtypes too short: {len(func.argtypes)}"
                )
                assert func.argtypes[bool_index] == ctypes.c_bool, (
                    f"{func_name} arg {bool_index} should be c_bool but is {func.argtypes[bool_index]}"
                )

    def test_i64_returning_functions_have_correct_restype(self):
        """Functions returning i64 must have c_int64 restype."""
        lib = get_chat_lib()

        i64_functions = [
            "talu_responses_append_message",
            "talu_responses_append_message_hidden",
            "talu_responses_insert_message",
            "talu_responses_insert_message_hidden",
            "talu_responses_append_function_call",
            "talu_responses_append_function_call_output",
        ]

        for func_name in i64_functions:
            if hasattr(lib, func_name):
                func = getattr(lib, func_name)
                assert func.restype == ctypes.c_int64, (
                    f"{func_name} should return c_int64 but returns {func.restype}"
                )
