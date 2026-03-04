"""
FFI bindings for training sessions.

Justification: Provides ctypes Structure definitions mirroring Zig extern structs
(CLoraConfig, CTrainingConfig, CStepMetrics, CTrainingInfo), CFUNCTYPE for step
callbacks, and call wrappers for talu_train_* C API functions.
"""

import ctypes
import enum
from dataclasses import dataclass
from typing import Any

from .._bindings import check, get_lib

# =============================================================================
# ctypes Structures (mirror Zig extern structs from capi_bridge.zig)
# =============================================================================


class CLoraConfig(ctypes.Structure):
    """C-compatible LoRA adapter configuration."""

    _fields_ = [
        ("rank", ctypes.c_uint32),
        ("alpha", ctypes.c_float),
    ]


class CTrainingConfig(ctypes.Structure):
    """C-compatible training hyperparameters."""

    _fields_ = [
        ("learning_rate", ctypes.c_float),
        ("min_learning_rate", ctypes.c_float),
        ("weight_decay", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("batch_size", ctypes.c_uint32),
        ("seq_len", ctypes.c_uint32),
        ("warmup_steps", ctypes.c_uint64),
        ("total_steps", ctypes.c_uint64),
        ("max_grad_norm", ctypes.c_float),
        ("gradient_accumulation_steps", ctypes.c_uint32),
        ("log_interval", ctypes.c_uint32),
        ("save_interval", ctypes.c_uint32),
    ]


class CStepMetrics(ctypes.Structure):
    """C-compatible per-step training metrics."""

    _fields_ = [
        ("step", ctypes.c_uint64),
        ("loss", ctypes.c_float),
        ("learning_rate", ctypes.c_float),
        ("grad_norm", ctypes.c_float),
    ]


class CTrainingInfo(ctypes.Structure):
    """C-compatible training session info."""

    _fields_ = [
        ("current_step", ctypes.c_uint64),
        ("total_steps", ctypes.c_uint64),
        ("trainable_params", ctypes.c_uint64),
        ("adapter_layers", ctypes.c_uint32),
        ("state", ctypes.c_uint8),
        ("_padding", ctypes.c_uint8 * 3),
    ]


# Step callback: (metrics_ptr, user_data) -> i32
# Returns 0 to continue, non-zero to cancel.
CStepCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.POINTER(CStepMetrics),
    ctypes.c_void_p,
)


# =============================================================================
# Python Dataclasses (user-facing, no ctypes)
# =============================================================================


class TrainingState(enum.IntEnum):
    """Training session state."""

    CREATED = 0
    MODEL_LOADED = 1
    CONFIGURED = 2
    DATA_LOADED = 3
    TRAINING = 4
    COMPLETED = 5


@dataclass
class LoraConfig:
    """LoRA adapter configuration.

    Args:
        rank: Rank of the low-rank matrices. Higher rank = more parameters.
        alpha: Scaling factor. Effective scale is alpha / rank.

    Example:
        >>> config = LoraConfig(rank=8, alpha=16.0)
    """

    rank: int = 16
    alpha: float = 32.0


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Default values match PyTorch/tinyllm conventions.

    Args:
        learning_rate: Peak learning rate for cosine schedule.
        min_learning_rate: Minimum learning rate at end of cosine decay.
        weight_decay: AdamW weight decay coefficient.
        beta1: Adam first moment decay rate.
        beta2: Adam second moment decay rate.
        epsilon: Adam numerical stability constant.
        batch_size: Number of sequences per training step.
        seq_len: Sequence length for each training example.
        warmup_steps: Linear warmup steps before cosine decay.
        total_steps: Total training steps.
        max_grad_norm: Maximum gradient norm for clipping.
        gradient_accumulation_steps: Accumulate gradients over N micro-steps.
        log_interval: Log metrics every N steps.
        save_interval: Save checkpoints every N steps (0 = disabled).

    Example:
        >>> config = TrainingConfig(learning_rate=1e-4, total_steps=500)
    """

    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    batch_size: int = 4
    seq_len: int = 128
    warmup_steps: int = 100
    total_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    save_interval: int = 0


@dataclass(frozen=True)
class StepMetrics:
    """Per-step training metrics (immutable).

    Attributes:
        step: Current training step.
        loss: Loss value at this step.
        learning_rate: Current learning rate.
        grad_norm: Gradient norm before clipping.
    """

    step: int
    loss: float
    learning_rate: float
    grad_norm: float


@dataclass(frozen=True)
class TrainingInfo:
    """Training session state snapshot (immutable).

    Attributes:
        current_step: Current training step.
        total_steps: Total configured training steps.
        trainable_params: Number of trainable parameters.
        adapter_layers: Number of LoRA adapter layers.
        state: Current session state.
    """

    current_step: int
    total_steps: int
    trainable_params: int
    adapter_layers: int
    state: TrainingState


# =============================================================================
# Conversion Helpers
# =============================================================================


def to_c_lora_config(config: LoraConfig) -> CLoraConfig:
    """Convert Python LoraConfig to C struct."""
    return CLoraConfig(rank=config.rank, alpha=config.alpha)


def to_c_training_config(config: TrainingConfig) -> CTrainingConfig:
    """Convert Python TrainingConfig to C struct."""
    return CTrainingConfig(
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        max_grad_norm=config.max_grad_norm,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
    )


def from_c_step_metrics(m: CStepMetrics) -> StepMetrics:
    """Convert C step metrics to Python dataclass."""
    return StepMetrics(
        step=m.step,
        loss=m.loss,
        learning_rate=m.learning_rate,
        grad_norm=m.grad_norm,
    )


def from_c_training_info(info: CTrainingInfo) -> TrainingInfo:
    """Convert C training info to Python dataclass."""
    return TrainingInfo(
        current_step=info.current_step,
        total_steps=info.total_steps,
        trainable_params=info.trainable_params,
        adapter_layers=info.adapter_layers,
        state=TrainingState(info.state),
    )


# =============================================================================
# Callback Wrapper
# =============================================================================


def wrap_callback(
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
        # True or None → continue (0), False → cancel (1)
        if result is False:
            return 1
        return 0

    c_callback = CStepCallback(c_callback_fn)
    return (c_callback, ctypes.c_void_p(0))


# =============================================================================
# FFI Call Wrappers
# =============================================================================

_lib = get_lib()


def call_train_create() -> tuple[int, int]:
    """Create a training session.

    Returns:
        Tuple of (error_code, handle_ptr). Handle is 0 on failure.
    """
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_train_create(ctypes.byref(out_ptr))
    if code != 0:
        return (code, 0)
    return (0, out_ptr.value or 0)


def call_train_destroy(ptr: int) -> None:
    """Destroy a training session. Null-safe."""
    _lib.talu_train_destroy(ptr)


def call_train_load_model(
    ptr: int,
    model_path: bytes,
    lora_config: LoraConfig,
    target_modules: list[str] | None,
) -> int:
    """Load model and configure LoRA adapter.

    Returns:
        Error code (0 = success).
    """
    c_lora = to_c_lora_config(lora_config)

    if target_modules:
        targets_bytes = [t.encode("utf-8") for t in target_modules]
        c_targets = (ctypes.c_char_p * len(targets_bytes))(
            *targets_bytes
        )
        num_targets = len(targets_bytes)
    else:
        c_targets = None
        num_targets = 0

    return _lib.talu_train_load_model(
        ptr,
        model_path,
        ctypes.byref(c_lora),
        c_targets,
        num_targets,
    )


def call_train_configure(ptr: int, config: TrainingConfig) -> int:
    """Set training hyperparameters.

    Returns:
        Error code (0 = success).
    """
    c_config = to_c_training_config(config)
    return _lib.talu_train_configure(ptr, ctypes.byref(c_config))


def call_train_load_data(ptr: int, data_path: bytes) -> int:
    """Load tokenized training data.

    Returns:
        Error code (0 = success).
    """
    return _lib.talu_train_load_data(ptr, data_path)


def call_train_run(
    ptr: int,
    c_callback: CStepCallback | None,
    c_user_data: ctypes.c_void_p,
) -> int:
    """Run the training loop.

    Returns:
        Error code (0 = success).
    """
    return _lib.talu_train_run(ptr, c_callback, c_user_data)


def call_train_save_checkpoint(ptr: int, output_path: bytes) -> int:
    """Save adapter weights to file.

    Returns:
        Error code (0 = success).
    """
    return _lib.talu_train_save_checkpoint(ptr, output_path)


def call_train_get_info(ptr: int) -> tuple[int, TrainingInfo | None]:
    """Query training session state.

    Returns:
        Tuple of (error_code, training_info). Info is None on failure.
    """
    c_info = CTrainingInfo()
    code = _lib.talu_train_get_info(ptr, ctypes.byref(c_info))
    if code != 0:
        return (code, None)
    return (0, from_c_training_info(c_info))
