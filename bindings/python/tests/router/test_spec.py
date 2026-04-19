import ctypes
from pathlib import Path

from talu._bindings import get_lib
from talu._native import BackendCreateOptions
from talu.router import _bindings as spec

# Centralized error codes (from error_codes.zig)
OK = 0
MODEL_NOT_FOUND = 100
INVALID_ARGUMENT = 901


def _zero_struct(value: ctypes.Structure) -> None:
    ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))


def _make_cstr(value: str, buffers: list[ctypes.Array]) -> ctypes.c_char_p:
    buf = ctypes.create_string_buffer(value.encode("utf-8"))
    buffers.append(buf)
    return ctypes.cast(buf, ctypes.c_char_p)


def _build_local_spec(path: str) -> tuple[spec.CTaluModelSpec, list[ctypes.Array]]:
    buffers: list[ctypes.Array] = []
    s = spec.CTaluModelSpec()
    _zero_struct(s)
    s.abi_version = 1
    s.struct_size = ctypes.sizeof(spec.CTaluModelSpec)
    s.ref = _make_cstr(path, buffers)
    s.backend_type_raw = 0  # LOCAL
    s.backend_config.local = spec.CLocalConfig(
        gpu_layers=-1,
        use_mmap=1,
        num_threads=0,
        _reserved=(ctypes.c_uint8 * 32)(),
    )
    return s, buffers


def test_canonical_view_lifetime() -> None:
    lib = get_lib()
    s, _buffers = _build_local_spec(".")

    handle = spec.TaluCanonicalSpecHandle()
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == OK

    view = spec.CTaluModelSpec()
    code = lib.talu_config_get_view(handle, ctypes.byref(view))
    assert code == OK

    ref_copy = ctypes.string_at(view.ref).decode("utf-8")

    lib.talu_config_free(handle)

    assert ref_copy == "."


def test_invalid_enum_rejected(tmp_path: Path) -> None:
    lib = get_lib()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "model.bin"
    model_path.write_text("ok")

    s = spec.CTaluModelSpec()
    _zero_struct(s)
    s.abi_version = 1
    s.struct_size = ctypes.sizeof(spec.CTaluModelSpec)
    ref_buf = ctypes.create_string_buffer(str(model_path).encode("utf-8"))
    s.ref = ctypes.cast(ref_buf, ctypes.c_char_p)
    s.backend_type_raw = 99

    handle = spec.TaluCanonicalSpecHandle(0x1234)
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == INVALID_ARGUMENT
    assert handle.value is None


def test_backend_detection_rules(tmp_path: Path) -> None:
    lib = get_lib()
    missing_path = tmp_path / "missing-model"

    # Unspecified + missing local path -> ModelNotFound
    # (local paths like /tmp/.../missing-model are detected as local scheme,
    # so missing path returns ModelNotFound rather than AmbiguousBackend)
    s = spec.CTaluModelSpec()
    _zero_struct(s)
    s.abi_version = 1
    s.struct_size = ctypes.sizeof(spec.CTaluModelSpec)
    ref_buf = ctypes.create_string_buffer(str(missing_path).encode("utf-8"))
    s.ref = ctypes.cast(ref_buf, ctypes.c_char_p)
    s.backend_type_raw = -1  # UNSPECIFIED

    handle = spec.TaluCanonicalSpecHandle(0x1234)
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == MODEL_NOT_FOUND
    assert handle.value is None

    # Local + missing path -> ModelNotFound (via canonicalize)
    s.backend_type_raw = 0  # LOCAL
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == MODEL_NOT_FOUND


def test_out_pointer_safety(tmp_path: Path) -> None:
    lib = get_lib()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "model.bin"
    model_path.write_text("ok")

    s, _buffers = _build_local_spec(str(model_path))

    null_out = ctypes.cast(0, ctypes.POINTER(spec.TaluCanonicalSpecHandle))
    code = lib.talu_config_canonicalize(ctypes.byref(s), null_out)
    assert code == INVALID_ARGUMENT

    handle = spec.TaluCanonicalSpecHandle(0x1234)
    s.backend_type_raw = 99
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == INVALID_ARGUMENT
    assert handle.value is None


def test_truncated_struct_size_rejected(tmp_path: Path) -> None:
    lib = get_lib()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "model.bin"
    model_path.write_text("ok")

    min_header_size = spec.CTaluModelSpec.backend_type_raw.offset + ctypes.sizeof(ctypes.c_int)
    raw = ctypes.create_string_buffer(min_header_size)
    spec_p = ctypes.cast(raw, ctypes.POINTER(spec.CTaluModelSpec))

    ref_buf = ctypes.create_string_buffer(str(model_path).encode("utf-8"))
    spec_p.contents.abi_version = 1
    spec_p.contents.struct_size = min_header_size
    spec_p.contents.ref = ctypes.cast(ref_buf, ctypes.c_char_p)
    spec_p.contents.backend_type_raw = 0  # LOCAL

    handle = spec.TaluCanonicalSpecHandle(0x1234)
    code = lib.talu_config_canonicalize(spec_p, ctypes.byref(handle))
    assert code == INVALID_ARGUMENT
    assert handle.value is None


def test_backend_lifecycle_local_model(test_model_path) -> None:
    lib = get_lib()
    model_path = Path(test_model_path)

    s, _buffers = _build_local_spec(str(model_path))
    handle = spec.TaluCanonicalSpecHandle()
    code = lib.talu_config_canonicalize(ctypes.byref(s), ctypes.byref(handle))
    assert code == OK

    backend = spec.TaluInferenceBackendHandle()
    options = BackendCreateOptions()
    code = lib.talu_backend_create_from_canonical(handle, options, ctypes.byref(backend))
    assert code == OK

    lib.talu_config_free(handle)
    lib.talu_backend_free(backend)
