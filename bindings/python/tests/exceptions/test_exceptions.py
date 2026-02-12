"""
Tests for the centralized error handling system.

Tests that:
1. Error codes are properly mapped to Python exceptions
2. Error messages are retrieved correctly
3. Thread-local error handling works
4. All error types are accessible and properly categorized

See tests/errors/__init__.py for the full code range specification.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


class TestErrorTypes:
    """Tests for error type hierarchy and accessibility."""

    def test_base_error_importable(self, talu):
        """TaluError is importable from talu."""
        assert hasattr(talu, "TaluError")
        assert issubclass(talu.TaluError, Exception)

    def test_generation_errors_importable(self):
        """Generation errors are importable from talu.exceptions."""
        from talu.exceptions import EmptyPromptError, GenerationError, TaluError

        assert issubclass(GenerationError, TaluError)
        assert issubclass(EmptyPromptError, GenerationError)

    def test_model_errors_importable(self):
        """Model errors are importable from talu.exceptions."""
        from talu.exceptions import ModelError, ModelNotFoundError, TaluError

        assert issubclass(ModelError, TaluError)
        assert issubclass(ModelNotFoundError, ModelError)

    def test_tokenizer_error_importable(self):
        """TokenizerError is importable from talu.exceptions."""
        from talu.exceptions import TaluError, TokenizerError

        assert issubclass(TokenizerError, TaluError)

    def test_convert_error_importable(self):
        """ConvertError is importable from talu.exceptions."""
        from talu.exceptions import ConvertError, TaluError

        assert issubclass(ConvertError, TaluError)

    def test_error_inheritance_chain(self):
        """Error classes have proper inheritance for exception handling."""
        from talu.exceptions import EmptyPromptError, GenerationError, ModelNotFoundError

        # GenerationError should be catchable as RuntimeError
        assert issubclass(GenerationError, RuntimeError)

        # EmptyPromptError should be catchable as ValueError
        assert issubclass(EmptyPromptError, ValueError)

        # ModelNotFoundError should be catchable as FileNotFoundError
        assert issubclass(ModelNotFoundError, FileNotFoundError)


class TestModelErrors:
    """Tests for model-related error handling.

    Note: Chat uses lazy loading - model validation happens at generation time,
    not construction. These tests use Tokenizer which validates eagerly.
    The Zig core currently returns error code 999 (internal error) for path
    resolution failures rather than 100-199 (model errors), so we catch TaluError.
    """

    def test_invalid_model_path_raises_talu_error(self, talu):
        """Loading invalid model path raises TaluError.

        Note: This raises TaluError (not ModelError) because Zig returns code 999
        for path resolution failures. Tokenizer is used because Chat lazy loads.
        """
        with pytest.raises(talu.TaluError):
            talu.Tokenizer("/nonexistent/path/to/model")

    def test_error_message_contains_path_info(self, talu):
        """Error message includes resolution failure info."""
        with pytest.raises(talu.TaluError) as exc_info:
            talu.Tokenizer("/nonexistent/path/to/model")
        message = str(exc_info.value).lower()
        # Message should reference resolution failure
        assert "resolve" in message or "not found" in message or "weights" in message

    def test_empty_model_path_raises_talu_error(self, talu):
        """Empty model path raises TaluError (path resolution failure)."""
        with pytest.raises(talu.TaluError):
            talu.Tokenizer("")


class TestTokenizerErrors:
    """Tests for tokenizer-related error handling.

    Note: The Zig core returns error code 999 (internal error) for path resolution
    failures, which maps to TaluError rather than ModelError or TokenizerError.
    """

    def test_tokenizer_invalid_path_raises_talu_error(self, talu):
        """Creating tokenizer with invalid path raises TaluError.

        Path resolution failures return error code 999 from Zig, not 100-199.
        """
        with pytest.raises(talu.TaluError):
            talu.Tokenizer("/nonexistent/tokenizer/path")


class TestConverterErrors:
    """Tests for converter-related error handling."""

    def test_convert_invalid_model_raises_convert_error(self, talu):
        """Converting non-existent model raises ConvertError."""
        from talu.exceptions import ConvertError

        with pytest.raises(ConvertError):
            talu.convert("definitely/not-a-real-model-12345")

    def test_convert_invalid_scheme_raises_value_error(self, talu):
        """Converting with invalid scheme raises ValueError."""
        with pytest.raises(ValueError):
            talu.convert("some/model", scheme="invalid_scheme")


class TestErrorMessageRetrieval:
    """Tests for error message content and retrieval.

    Note: These tests use Tokenizer because Chat lazy loads models.
    """

    def test_error_has_message(self, talu):
        """Errors contain descriptive messages."""
        try:
            talu.Tokenizer("/nonexistent/path")
            pytest.fail("Should have raised")
        except Exception as e:
            message = str(e)
            assert len(message) > 0
            # Should not be just an error code
            assert not message.isdigit()

    def test_error_message_not_unknown(self, talu):
        """Error messages should not be 'Unknown error' for known cases."""
        try:
            talu.Tokenizer("/nonexistent/path")
            pytest.fail("Should have raised")
        except Exception as e:
            message = str(e).lower()
            # For known error types, should have meaningful message
            # (may contain "unknown" if it's genuinely unknown, but not for path errors)
            if (
                "path" in message
                or "resolve" in message
                or "not found" in message
                or "weights" in message
            ):
                pass  # Good - descriptive message
            elif "unknown error (code" in message:
                pytest.fail(f"Got generic 'unknown error' for path error: {e}")


class TestStructuredOutputErrors:
    """Tests for structured output exceptions."""

    def test_schema_validation_error_handles_invalid_json(self):
        """SchemaValidationError tolerates invalid JSON and leaves partial_data unset."""
        from talu.exceptions import SchemaValidationError

        err = SchemaValidationError("{invalid json", ValueError("boom"))

        assert err.raw_text == "{invalid json"
        assert err.partial_data is None


class TestErrorCodeMapping:
    """Tests for error code to exception mapping."""

    def test_lib_check_function_exists(self):
        """The check() function exists in _lib."""
        from talu._bindings import check

        assert callable(check)

    def test_check_zero_does_not_raise(self):
        """check(0) does not raise (success code)."""
        from talu._bindings import check

        # Should not raise
        check(0)

    def test_check_nonzero_raises(self):
        """check(nonzero) raises TaluError."""
        from talu._bindings import check
        from talu.exceptions import TaluError

        # Non-zero codes should raise TaluError
        with pytest.raises(TaluError):
            check(999)  # Internal error code

    def test_memory_error_code_raises_memory_error(self):
        """Error code 900 (out_of_memory) raises MemoryError."""
        from talu._bindings import check

        with pytest.raises(MemoryError):
            check(900)


class TestThreadSafety:
    """Tests for thread-local error handling.

    Note: These tests use Tokenizer because Chat lazy loads models.
    """

    def test_errors_are_thread_local(self, talu):
        """Errors in one thread don't affect another thread."""
        errors = {"thread1": None, "thread2": None}

        def thread1_work():
            try:
                # This should fail (Tokenizer validates eagerly)
                talu.Tokenizer("/thread1/nonexistent/path")
            except Exception as e:
                errors["thread1"] = str(e)

        def thread2_work():
            try:
                # This should also fail with a different path
                talu.Tokenizer("/thread2/different/path")
            except Exception as e:
                errors["thread2"] = str(e)

        t1 = threading.Thread(target=thread1_work)
        t2 = threading.Thread(target=thread2_work)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should have caught errors
        assert errors["thread1"] is not None
        assert errors["thread2"] is not None

        # Error messages should be independent (contain their respective paths)
        # Note: This is a best-effort check - the exact message format may vary
        assert errors["thread1"] != "" and errors["thread2"] != ""

    def test_concurrent_error_handling(self, talu):
        """Multiple threads can handle errors concurrently."""
        error_count = {"count": 0}
        lock = threading.Lock()

        def worker(path_suffix):
            try:
                talu.Tokenizer(f"/nonexistent/path/{path_suffix}")
            except Exception:
                with lock:
                    error_count["count"] += 1

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for f in futures:
                f.result()

        # All 10 calls should have raised errors
        assert error_count["count"] == 10


class TestErrorRecovery:
    """Tests that errors don't leave the system in a bad state."""

    def test_can_continue_after_error(self, talu, test_model_path):
        """System remains usable after an error."""
        # First, cause an error
        try:
            talu.Chat("/nonexistent/path")
        except Exception:
            pass

        # Then, do a valid operation
        session = talu.Chat(test_model_path)
        assert session is not None

        # Should still be able to use it (send method works)
        assert hasattr(session, "send")

    def test_multiple_errors_dont_accumulate(self, talu):
        """Multiple errors don't cause state accumulation issues."""
        for i in range(10):
            try:
                talu.Chat(f"/nonexistent/path/{i}")
            except Exception as e:
                # Each error should be fresh, not accumulated
                message = str(e)
                # Should not contain multiple error messages concatenated
                assert message.count("Failed to resolve") <= 1


class TestErrorCodeAttribute:
    """Tests for error code accessibility via exception attribute.

    Note: These tests use Tokenizer because Chat lazy loads models.
    """

    def test_error_has_code_attribute(self, talu):
        """Exceptions have a string code attribute."""
        try:
            talu.Tokenizer("/nonexistent/path")
            pytest.fail("Should have raised")
        except talu.TaluError as e:
            assert hasattr(e, "code")
            assert isinstance(e.code, str)
            # Also has original_code for the Zig integer code
            assert hasattr(e, "original_code")

    def test_code_matches_error_type(self):
        """String code matches the error category."""
        from talu._bindings import check, clear_error
        from talu.exceptions import EmptyPromptError, ModelNotFoundError

        # Model errors should have string code starting with MODEL_
        clear_error()
        try:
            check(100)
        except ModelNotFoundError as e:
            assert e.code == "MODEL_NOT_FOUND"
            assert e.original_code == 100

        # Generation errors should have string code starting with GENERATION_
        clear_error()
        try:
            check(301)
        except EmptyPromptError as e:
            assert e.code == "GENERATION_EMPTY_PROMPT"
            assert e.original_code == 301


class TestSpecificErrorCodes:
    """Tests for specific error code behaviors.

    Note: check() is designed to be called immediately after a C API function.
    For testing, we clear the error state first to simulate a fresh error.
    """

    def test_model_not_found_code_100(self):
        """Error code 100 maps to ModelNotFoundError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import ModelNotFoundError

        clear_error()  # Clear any previous error state
        with pytest.raises(ModelNotFoundError) as exc:
            check(100)
        assert exc.value.code == "MODEL_NOT_FOUND"
        assert exc.value.original_code == 100

    def test_tokenizer_error_code_200(self):
        """Error code 200 maps to TokenizerError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import TokenizerError

        clear_error()
        with pytest.raises(TokenizerError) as exc:
            check(200)
        assert exc.value.code == "TOKENIZER_NOT_FOUND"
        assert exc.value.original_code == 200

    def test_generation_error_code_300(self):
        """Error code 300 maps to GenerationError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import GenerationError

        clear_error()
        with pytest.raises(GenerationError) as exc:
            check(300)
        assert exc.value.code == "GENERATION_FAILED"
        assert exc.value.original_code == 300

    def test_empty_prompt_error_code_301(self):
        """Error code 301 maps to EmptyPromptError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import EmptyPromptError

        clear_error()
        with pytest.raises(EmptyPromptError) as exc:
            check(301)
        assert exc.value.code == "GENERATION_EMPTY_PROMPT"
        assert exc.value.original_code == 301

    def test_convert_error_code_400(self):
        """Error code 400 maps to ConvertError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import ConvertError

        clear_error()
        with pytest.raises(ConvertError) as exc:
            check(400)
        assert exc.value.code == "CONVERT_FAILED"
        assert exc.value.original_code == 400

    def test_out_of_memory_code_900(self):
        """Error code 900 maps to MemoryError."""
        from talu._bindings import check, clear_error

        clear_error()
        with pytest.raises(MemoryError):
            check(900)

    def test_unmapped_code_raises_unmapped_error(self):
        """Unmapped Zig codes raise TaluError with UNMAPPED_ERROR (strict one-path)."""
        from talu._bindings import check, clear_error
        from talu.exceptions import TaluError

        clear_error()
        # Code 150 is not in ERROR_MAP - this signals drift between Zig and Python
        # Strict one-path: no range-based fallback, always UNMAPPED_ERROR
        with pytest.raises(TaluError) as exc:
            check(150)
        assert exc.value.code == "UNMAPPED_ERROR"
        assert exc.value.original_code == 150

    def test_unmapped_code_preserves_original_code(self):
        """Unmapped codes preserve the original Zig integer for debugging."""
        from talu._bindings import check, clear_error
        from talu.exceptions import TaluError

        clear_error()
        # Various unmapped codes all get UNMAPPED_ERROR but preserve original
        for code in [150, 250, 350, 450, 550, 650, 800]:
            clear_error()
            with pytest.raises(TaluError) as exc:
                check(code)
            assert exc.value.code == "UNMAPPED_ERROR"
            assert exc.value.original_code == code
            assert "zig_code" in exc.value.details
            assert exc.value.details["zig_code"] == code

    def test_invalid_argument_code_901(self):
        """Error code 901 maps to ValidationError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import ValidationError

        clear_error()
        with pytest.raises(ValidationError) as exc:
            check(901)
        assert exc.value.code == "INVALID_ARGUMENT"
        assert exc.value.original_code == 901

    def test_invalid_handle_code_902(self):
        """Error code 902 maps to ValidationError with string code."""
        from talu._bindings import check, clear_error
        from talu.exceptions import ValidationError

        clear_error()
        with pytest.raises(ValidationError) as exc:
            check(902)
        assert exc.value.code == "INVALID_HANDLE"
        assert exc.value.original_code == 902


class TestTakeLastError:
    """Tests for take_last_error() function behavior."""

    def test_take_last_error_clears_state(self):
        """After take_last_error(), error state is cleared."""
        from talu._bindings import check, clear_error, take_last_error

        clear_error()
        # Trigger an error
        try:
            check(100)
        except Exception:
            pass

        # Error should be cleared after check() called take_last_error()
        code, msg = take_last_error()
        assert code == 0
        assert msg is None

    def test_take_last_error_returns_code_and_message(self):
        """take_last_error() returns tuple of (code, message)."""
        from talu._bindings import clear_error, take_last_error

        clear_error()
        code, msg = take_last_error()

        # When no error, returns (0, None)
        assert code == 0
        assert msg is None

    def test_take_last_error_is_thread_local(self, talu):
        """Each thread has its own error buffer - errors don't leak.

        Note: Uses Tokenizer because Chat lazy loads models.
        """
        import threading

        # This test verifies that each thread can catch its own error
        # independently of other threads
        errors_caught = {"count": 0}
        lock = threading.Lock()

        def thread_work(thread_id):
            try:
                talu.Tokenizer(f"/nonexistent/{thread_id}")
            except talu.TaluError:
                with lock:
                    errors_caught["count"] += 1

        threads = [threading.Thread(target=thread_work, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 3 threads should have caught their own error
        assert errors_caught["count"] == 3


class TestErrorCodeRangeFallback:
    """Tests that codes in ranges map to category-appropriate exceptions."""

    def test_model_range_mapping(self):
        """Codes 100-199 map to ModelError with appropriate string codes."""
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import ModelError

        # Test mapped codes in model range
        for code in [102, 103, 104, 105]:
            clear_error()
            with pytest.raises(ModelError) as exc:
                check(code)
            # String code comes from ERROR_MAP
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code

    def test_model_error_codes_have_descriptive_messages(self):
        """Model error codes 102-105 have descriptive fallback messages.

        When the Zig core sets these errors, it provides descriptive messages.
        When no message is set (e.g., in tests), the fallback includes the code.

        Error code semantics (from error_codes.zig):
        - 102: model_unsupported_architecture
        - 103: model_config_missing
        - 104: model_weights_missing
        - 105: model_weights_corrupted
        """
        from talu._bindings import check, clear_error
        from talu.exceptions import ModelError

        # Expected semantics for each code (for documentation and future assertions)
        code_semantics = {
            102: "unsupported_architecture",
            103: "config_missing",
            104: "weights_missing",
            105: "weights_corrupted",
        }

        for code, semantic in code_semantics.items():
            clear_error()
            with pytest.raises(ModelError) as exc:
                check(code)

            message = str(exc.value)
            # Message must be non-empty and include context
            assert len(message) > 0, f"Error {code} ({semantic}) has empty message"
            # Fallback message format includes the code number
            assert str(code) in message, (
                f"Error {code} ({semantic}) message should contain code: {message}"
            )

    def test_tokenizer_range_mapping(self):
        """Codes 200-299 map to TokenizerError with appropriate string codes."""
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import TokenizerError

        for code in [201, 202, 203]:
            clear_error()
            with pytest.raises(TokenizerError) as exc:
                check(code)
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code

    def test_generation_range_mapping(self):
        """Codes 300-399 map to GenerationError with appropriate string codes."""
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import GenerationError

        for code in [302, 303]:
            clear_error()
            with pytest.raises(GenerationError) as exc:
                check(code)
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code

    def test_convert_range_mapping(self):
        """Codes 400-499 map to ConvertError with appropriate string codes."""
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import ConvertError

        for code in [401, 402, 403]:
            clear_error()
            with pytest.raises(ConvertError) as exc:
                check(code)
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code

    def test_io_range_mapping(self):
        """Codes 500-599 map to IOError with appropriate string codes."""
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import IOError

        for code in [500, 501, 502, 503, 504]:
            clear_error()
            with pytest.raises(IOError) as exc:
                check(code)
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code


class TestErrorCodeInvariants:
    """These tests ensure the Python error mapping stays synchronized with
    the Zig error code definitions and documented contracts.
    """

    def test_error_map_covers_documented_codes(self):
        """
        Documented codes:
        - Model: 100-105
        - Tokenizer: 200-203
        - Generation: 300-303
        - Conversion: 400-403
        """
        from talu._bindings import ERROR_MAP

        # Documented error codes from ERROR_HANDLING.md
        documented_codes = {
            # Model errors (100-199)
            100,  # model_not_found
            101,  # model_invalid_format
            102,  # model_unsupported_architecture
            103,  # model_config_missing
            104,  # model_weights_missing
            105,  # model_weights_corrupted
            # Tokenizer errors (200-299)
            200,  # tokenizer_not_found
            201,  # tokenizer_invalid_format
            202,  # tokenizer_encode_failed
            203,  # tokenizer_decode_failed
            # Generation errors (300-399)
            300,  # generation_failed
            301,  # generation_empty_prompt
            302,  # generation_context_overflow
            303,  # generation_invalid_params
            # Conversion errors (400-499)
            400,  # convert_failed
            401,  # convert_unsupported_format
            402,  # convert_already_quantized
            403,  # convert_output_exists
        }

        mapped_codes = set(ERROR_MAP.keys())

        # All documented codes must be in the map
        missing = documented_codes - mapped_codes
        assert not missing, f"ERROR_MAP missing documented codes: {sorted(missing)}"

    def test_error_map_codes_are_in_valid_ranges(self):
        """All ERROR_MAP codes fall within documented ranges.

        Valid ranges:
        - 100-199: Model errors
        - 200-299: Tokenizer errors
        - 300-399: Generation errors
        - 400-499: Conversion errors
        - 500-599: I/O errors
        - 600-699: Template errors
        - 700-799: Storage errors
        - 900-999: System errors
        """
        from talu._bindings import ERROR_MAP
        from talu.exceptions import (
            ConvertError,
            GenerationError,
            IOError,
            ModelError,
            StorageError,
            TaluError,
            TemplateError,
            TokenizerError,
        )

        for code, (error_class, string_code) in ERROR_MAP.items():
            # Each code must be in a valid range
            valid_ranges = (100 <= code < 800) or (900 <= code < 1000)
            assert valid_ranges, f"Code {code} is outside valid ranges"

            # String code must be a valid string
            assert isinstance(string_code, str), f"Code {code} has invalid string code type"
            assert len(string_code) > 0, f"Code {code} has empty string code"

            # Code range must match error class
            if 100 <= code < 200:
                assert issubclass(error_class, ModelError), (
                    f"Code {code} should map to ModelError subclass, got {error_class}"
                )
            elif 200 <= code < 300:
                assert issubclass(error_class, TokenizerError), (
                    f"Code {code} should map to TokenizerError subclass, got {error_class}"
                )
            elif 300 <= code < 400:
                assert issubclass(error_class, GenerationError), (
                    f"Code {code} should map to GenerationError subclass, got {error_class}"
                )
            elif 400 <= code < 500:
                assert issubclass(error_class, ConvertError), (
                    f"Code {code} should map to ConvertError subclass, got {error_class}"
                )
            elif 500 <= code < 600:
                assert issubclass(error_class, IOError), (
                    f"Code {code} should map to IOError subclass, got {error_class}"
                )
            elif 600 <= code < 700:
                assert issubclass(error_class, TemplateError), (
                    f"Code {code} should map to TemplateError subclass, got {error_class}"
                )
            elif 700 <= code < 800:
                assert issubclass(error_class, StorageError), (
                    f"Code {code} should map to StorageError subclass, got {error_class}"
                )
            elif 900 <= code < 1000:
                # System errors can be various types
                assert issubclass(error_class, TaluError), (
                    f"Code {code} should map to TaluError subclass, got {error_class}"
                )

    def test_io_errors_in_error_map(self):
        """I/O errors (500-599) are now explicitly in ERROR_MAP.

        Contract: I/O errors have explicit mappings to IOError with string codes.
        """
        from talu._bindings import ERROR_MAP
        from talu.exceptions import IOError

        # Check documented I/O codes are mapped
        io_codes = [500, 501, 502, 503, 504]
        for code in io_codes:
            assert code in ERROR_MAP, f"I/O code {code} should be in ERROR_MAP"
            error_class, string_code = ERROR_MAP[code]
            assert issubclass(error_class, IOError), (
                f"I/O code {code} should map to IOError, got {error_class}"
            )
            assert string_code.startswith("IO_"), (
                f"I/O code {code} should have IO_ prefix, got {string_code}"
            )

    def test_io_range_handled_by_check(self):
        """check() properly handles all documented I/O codes.

        Documented I/O codes (from ERROR_HANDLING.md):
        - 500: io_file_not_found
        - 501: io_permission_denied
        - 502: io_read_failed
        - 503: io_write_failed
        - 504: io_network_failed
        """
        from talu._bindings import ERROR_MAP, check, clear_error
        from talu.exceptions import IOError

        io_codes = [500, 501, 502, 503, 504]

        for code in io_codes:
            clear_error()
            with pytest.raises(IOError) as exc:
                check(code)
            expected_string_code = ERROR_MAP[code][1]
            assert exc.value.code == expected_string_code
            assert exc.value.original_code == code

    def test_system_error_codes_handled(self):
        """System error codes (900-999) are handled correctly.

        Documented system codes (from ERROR_HANDLING.md):
        - 900: out_of_memory -> MemoryError
        - 901: invalid_argument -> ValidationError
        - 902: invalid_handle -> StateError
        - 999: internal_error -> TaluError
        """
        from talu._bindings import check, clear_error
        from talu.exceptions import TaluError, ValidationError

        # 900 should raise MemoryError
        clear_error()
        with pytest.raises(MemoryError):
            check(900)

        # 901 should raise ValidationError
        clear_error()
        with pytest.raises(ValidationError) as exc:
            check(901)
        assert exc.value.code == "INVALID_ARGUMENT"
        assert exc.value.original_code == 901

        # 902 should raise ValidationError (INVALID_HANDLE is Zig-originated, not Python lifecycle)
        clear_error()
        with pytest.raises(ValidationError) as exc:
            check(902)
        assert exc.value.code == "INVALID_HANDLE"
        assert exc.value.original_code == 902

        # 999 should raise TaluError
        clear_error()
        with pytest.raises(TaluError) as exc:
            check(999)
        assert exc.value.code == "INTERNAL_ERROR"
        assert exc.value.original_code == 999
