"""
FFI bindings for full (from-scratch) training sessions.

Provides ctypes Structure definitions mirroring Zig extern structs
(CTransformerConfig, CFullSessionConfig, CFullSessionInfo), and call
wrappers for talu_train_full_* C API functions.

Reuses CStepMetrics and CStepCallback from _bindings.py.
"""

import array
import ctypes
import enum
from dataclasses import dataclass
from typing import Any

from .._bindings import check, get_lib
from ._bindings import CStepCallback, CStepMetrics, StepMetrics, from_c_step_metrics

# =============================================================================
# ctypes Structures (mirror Zig extern structs from capi/train_full.zig)
# =============================================================================


class CTransformerConfig(ctypes.Structure):
    """C-compatible transformer architecture configuration."""

    _fields_ = [
        ("vocab_size", ctypes.c_uint32),
        ("d_model", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("num_heads", ctypes.c_uint32),
        ("num_kv_heads", ctypes.c_uint32),
        ("d_ff", ctypes.c_uint32),
        ("seq_len", ctypes.c_uint32),
        ("rope_theta", ctypes.c_float),
        ("norm_eps", ctypes.c_float),
    ]


class CFullSessionConfig(ctypes.Structure):
    """C-compatible full training session hyperparameters."""

    _fields_ = [
        ("learning_rate", ctypes.c_float),
        ("min_learning_rate", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("max_grad_norm", ctypes.c_float),
        ("batch_size", ctypes.c_uint32),
        ("warmup_steps", ctypes.c_uint64),
        ("total_steps", ctypes.c_uint64),
    ]


class CFullSessionInfo(ctypes.Structure):
    """C-compatible full training session info."""

    _fields_ = [
        ("current_step", ctypes.c_uint64),
        ("total_steps", ctypes.c_uint64),
        ("total_params", ctypes.c_uint64),
        ("batch_size", ctypes.c_uint32),
        ("state", ctypes.c_uint8),
        ("_padding", ctypes.c_uint8 * 3),
    ]


# =============================================================================
# Python Dataclasses (user-facing, no ctypes)
# =============================================================================


class FullSessionState(enum.IntEnum):
    """Full training session state."""

    CREATED = 0
    INITIALIZED = 1
    CONFIGURED = 2
    DATA_LOADED = 3
    TRAINING = 4
    COMPLETED = 5


@dataclass
class TransformerConfig:
    """Transformer architecture configuration.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key/value heads (defaults to num_heads).
        d_ff: Feed-forward intermediate dimension (defaults to 4 * d_model).
        seq_len: Maximum sequence length.
        rope_theta: RoPE frequency base.
        norm_eps: RMSNorm epsilon.

    Example:
        >>> config = TransformerConfig(
        ...     vocab_size=32000, d_model=256, num_layers=4,
        ...     num_heads=4, d_ff=1024, seq_len=256,
        ... )
    """

    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None = None
    d_ff: int | None = None
    seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


@dataclass
class FullSessionConfig:
    """Training hyperparameters for from-scratch training.

    Args:
        learning_rate: Peak learning rate for cosine schedule.
        min_learning_rate: Minimum learning rate at end of cosine decay.
        weight_decay: AdamW weight decay coefficient.
        beta1: Adam first moment decay rate.
        beta2: Adam second moment decay rate.
        epsilon: Adam numerical stability constant.
        batch_size: Number of sequences per training step.
        warmup_steps: Linear warmup steps before cosine decay.
        total_steps: Total training steps.
        max_grad_norm: Maximum gradient norm for clipping.

    Example:
        >>> config = FullSessionConfig(learning_rate=3e-4, total_steps=10000)
    """

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    batch_size: int = 32
    warmup_steps: int = 500
    total_steps: int = 10000
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class FullSessionInfo:
    """Full training session state snapshot (immutable).

    Attributes:
        current_step: Current training step.
        total_steps: Total configured training steps.
        total_params: Total number of model parameters.
        batch_size: Configured batch size.
        state: Current session state.
    """

    current_step: int
    total_steps: int
    total_params: int
    batch_size: int
    state: FullSessionState


# =============================================================================
# Conversion Helpers
# =============================================================================


def to_c_transformer_config(config: TransformerConfig) -> CTransformerConfig:
    """Convert Python TransformerConfig to C struct."""
    num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
    d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model
    return CTransformerConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        seq_len=config.seq_len,
        rope_theta=config.rope_theta,
        norm_eps=config.norm_eps,
    )


def to_c_full_session_config(config: FullSessionConfig) -> CFullSessionConfig:
    """Convert Python FullSessionConfig to C struct."""
    return CFullSessionConfig(
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        max_grad_norm=config.max_grad_norm,
        batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
    )


def from_c_full_session_info(info: CFullSessionInfo) -> FullSessionInfo:
    """Convert C session info to Python dataclass."""
    return FullSessionInfo(
        current_step=info.current_step,
        total_steps=info.total_steps,
        total_params=info.total_params,
        batch_size=info.batch_size,
        state=FullSessionState(info.state),
    )


# =============================================================================
# Callback Wrapper
# =============================================================================


def wrap_full_callback(
    py_callback: "Any",
) -> tuple[CStepCallback | None, ctypes.c_void_p]:
    """Wrap a Python callback into a C-compatible step callback.

    The Python callback receives StepMetrics and returns:
    - True or None: continue training
    - False: cancel training

    Returns:
        Tuple of (c_callback, c_user_data). Both are None/null if no callback.
    """
    if py_callback is None:
        return (None, ctypes.c_void_p(0))

    def c_callback_fn(
        metrics_ptr: "Any", user_data: ctypes.c_void_p
    ) -> int:
        metrics = from_c_step_metrics(metrics_ptr.contents)
        result = py_callback(metrics)
        if result is False:
            return 1
        return 0

    c_callback = CStepCallback(c_callback_fn)
    return (c_callback, ctypes.c_void_p(0))


# =============================================================================
# FFI Call Wrappers
# =============================================================================

_lib = get_lib()


def call_train_full_create() -> tuple[int, int]:
    """Create a full training session.

    Returns:
        Tuple of (error_code, handle_ptr). Handle is 0 on failure.
    """
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_train_full_create(ctypes.byref(out_ptr))
    if code != 0:
        return (code, 0)
    return (0, out_ptr.value or 0)


def call_train_full_destroy(ptr: int) -> None:
    """Destroy a full training session. Null-safe."""
    _lib.talu_train_full_destroy(ptr)


def call_train_full_init_model(
    ptr: int, config: TransformerConfig, seed: int
) -> int:
    """Initialize model with random weights.

    Returns:
        Error code (0 = success).
    """
    c_config = to_c_transformer_config(config)
    return _lib.talu_train_full_init_model(
        ptr, ctypes.byref(c_config), ctypes.c_uint64(seed)
    )


def call_train_full_configure(ptr: int, config: FullSessionConfig) -> int:
    """Set training hyperparameters.

    Returns:
        Error code (0 = success).
    """
    c_config = to_c_full_session_config(config)
    return _lib.talu_train_full_configure(ptr, ctypes.byref(c_config))


def call_train_full_set_data(ptr: int, tokens: list[int]) -> tuple[int, Any]:
    """Set tokenized training data from a list of token IDs.

    Returns:
        Tuple of (error_code, retained_buffer).
    """
    arr = (ctypes.c_uint32 * len(tokens))(*tokens)
    return (_lib.talu_train_full_set_data(ptr, arr, len(tokens)), arr)


def call_train_full_load_data(ptr: int, data_path: bytes) -> int:
    """Load tokenized training data from a file.

    Returns:
        Error code (0 = success).
    """
    return _lib.talu_train_full_load_data(ptr, data_path)


def call_train_full_step(ptr: int) -> tuple[int, StepMetrics | None]:
    """Run one training step.

    Returns:
        Tuple of (error_code, step_metrics). Metrics is None on failure.
    """
    c_metrics = CStepMetrics()
    code = _lib.talu_train_full_step(ptr, ctypes.byref(c_metrics))
    if code != 0:
        return (code, None)
    return (0, from_c_step_metrics(c_metrics))


def call_train_full_run(
    ptr: int,
    c_callback: CStepCallback | None,
    c_user_data: ctypes.c_void_p,
) -> int:
    """Run the full training loop.

    Returns:
        Error code (0 = success).
    """
    return _lib.talu_train_full_run(ptr, c_callback, c_user_data)


def call_train_full_get_info(ptr: int) -> tuple[int, FullSessionInfo | None]:
    """Query full training session state.

    Returns:
        Tuple of (error_code, session_info). Info is None on failure.
    """
    c_info = CFullSessionInfo()
    code = _lib.talu_train_full_get_info(ptr, ctypes.byref(c_info))
    if code != 0:
        return (code, None)
    return (0, from_c_full_session_info(c_info))


def call_train_full_copy_weights_f32(
    ptr: int, count: int
) -> tuple[int, array.array | None]:
    """Copy flat f32 model weights from a full training session.

    Returns:
        Tuple of (error_code, array('f')). Weights are None on failure.
    """
    if count == 0:
        code = _lib.talu_train_full_copy_weights_f32(ptr, None, 0)
        if code != 0:
            return (code, None)
        return (0, array.array("f"))

    weights = array.array("f", [0.0]) * count
    c_weights = (ctypes.c_float * count).from_buffer(weights)
    code = _lib.talu_train_full_copy_weights_f32(ptr, c_weights, count)
    if code != 0:
        return (code, None)
    return (0, weights)


def call_train_full_copy_optimizer_state_f32(
    ptr: int, count: int
) -> tuple[int, array.array | None]:
    """Copy flat f32 optimizer state from a full training session."""
    if count == 0:
        code = _lib.talu_train_full_copy_optimizer_state_f32(ptr, None, 0)
        if code != 0:
            return (code, None)
        return (0, array.array("f"))

    state = array.array("f", [0.0]) * count
    c_state = (ctypes.c_float * count).from_buffer(state)
    code = _lib.talu_train_full_copy_optimizer_state_f32(ptr, c_state, count)
    if code != 0:
        return (code, None)
    return (0, state)


def call_train_full_load_weights_f32(
    ptr: int, weights: array.array | list[float], step: int
) -> int:
    """Load flat f32 model weights into a full training session."""
    if isinstance(weights, array.array):
        data = weights
    else:
        data = array.array("f", weights)

    if len(data) == 0:
        return _lib.talu_train_full_load_weights_f32(ptr, None, 0, ctypes.c_uint64(step))

    c_weights = (ctypes.c_float * len(data)).from_buffer(data)
    return _lib.talu_train_full_load_weights_f32(
        ptr, c_weights, len(data), ctypes.c_uint64(step)
    )


def call_train_full_load_optimizer_state_f32(
    ptr: int, state: array.array | list[float]
) -> int:
    """Load flat f32 optimizer state into a full training session."""
    if isinstance(state, array.array):
        data = state
    else:
        data = array.array("f", state)

    if len(data) == 0:
        return _lib.talu_train_full_load_optimizer_state_f32(ptr, None, 0)

    c_state = (ctypes.c_float * len(data)).from_buffer(data)
    return _lib.talu_train_full_load_optimizer_state_f32(ptr, c_state, len(data))

