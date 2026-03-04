"""
LoRA fine-tuning session.

Provides the TrainingSession class for managing the lifecycle of a training
session: load model, configure hyperparameters, load data, run training
with per-step callbacks, and save checkpoints.
"""

from collections.abc import Callable
from typing import Any

from .._bindings import check
from ..exceptions import StateError
from ._bindings import (
    LoraConfig,
    StepMetrics,
    TrainingConfig,
    TrainingInfo,
    call_train_configure,
    call_train_create,
    call_train_destroy,
    call_train_get_info,
    call_train_load_data,
    call_train_load_model,
    call_train_run,
    call_train_save_checkpoint,
    wrap_callback,
)


class TrainingSession:
    """LoRA fine-tuning session.

    Manages the lifecycle of a training session: load model, configure
    hyperparameters, load data, run training with callbacks, save checkpoints.

    Implements the standard talu lifecycle pattern: explicit ``close()``,
    context manager (``with``), and fail-safe ``__del__``.

    Args:
        None. Call ``load_model()`` after construction to begin setup.

    Example:
        >>> with TrainingSession() as session:
        ...     session.load_model("./my-model", lora=LoraConfig(rank=8))
        ...     session.configure(TrainingConfig(learning_rate=1e-4, total_steps=500))
        ...     session.load_data("./tokens.bin")
        ...     session.run(callback=lambda m: print(f"step {m.step}: loss={m.loss:.4f}"))
        ...     session.save_checkpoint("./adapter.safetensors")
    """

    __slots__ = ("_ptr", "_callback_ref")

    def __init__(self) -> None:
        code, ptr = call_train_create()
        check(code)
        self._ptr: int | None = ptr
        # prevent GC of C callback during training
        self._callback_ref: Any = None

    @property
    def _handle(self) -> int:
        """Get the internal handle, raising if closed."""
        if self._ptr is None:
            raise StateError("TrainingSession is closed")
        return self._ptr

    def close(self) -> None:
        """Release native training session resources.

        After calling close(), the session cannot be used. Safe to call
        multiple times (idempotent).
        """
        if self._ptr is not None:
            call_train_destroy(self._ptr)
            self._ptr = None
            self._callback_ref = None

    def __enter__(self) -> "TrainingSession":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._ptr is None else "open"
        return f"TrainingSession({state})"

    def __copy__(self) -> None:
        raise TypeError("TrainingSession cannot be copied")

    def __reduce__(self) -> None:
        raise TypeError("TrainingSession cannot be pickled")

    # =========================================================================
    # Session Setup
    # =========================================================================

    def load_model(
        self,
        model_path: str,
        *,
        lora: LoraConfig,
        target_modules: list[str] | None = None,
    ) -> None:
        """Load a model and create a LoRA adapter.

        Args:
            model_path: Path to the model directory.
            lora: LoRA adapter configuration (rank, alpha).
            target_modules: List of weight ID substrings to adapt
                (e.g., ``["q_proj", "v_proj"]``). If None, uses architecture
                defaults.

        Raises:
            TrainingError: If model loading fails.
            StateError: If the session is closed.

        Example:
            >>> session.load_model(
            ...     "./my-model",
            ...     lora=LoraConfig(rank=8, alpha=16.0),
            ...     target_modules=["q_proj", "v_proj"],
            ... )
        """
        code = call_train_load_model(
            self._handle,
            model_path.encode("utf-8"),
            lora,
            target_modules,
        )
        check(code)

    def configure(self, config: TrainingConfig) -> None:
        """Set training hyperparameters.

        Must be called after ``load_model()`` and before ``load_data()``.

        Args:
            config: Training hyperparameters.

        Raises:
            TrainingError: If configuration fails (e.g., invalid state).
            StateError: If the session is closed.

        Example:
            >>> session.configure(TrainingConfig(
            ...     learning_rate=1e-4,
            ...     total_steps=500,
            ...     batch_size=4,
            ...     seq_len=128,
            ... ))
        """
        code = call_train_configure(self._handle, config)
        check(code)

    def load_data(self, data_path: str) -> None:
        """Load tokenized training data from a file.

        The file must contain flat binary u32 tokens (native endianness).
        Must be called after ``configure()``.

        Args:
            data_path: Path to the tokenized data file.

        Raises:
            TrainingError: If data loading fails.
            StateError: If the session is closed.

        Example:
            >>> session.load_data("./train_tokens.bin")
        """
        code = call_train_load_data(self._handle, data_path.encode("utf-8"))
        check(code)

    # =========================================================================
    # Training
    # =========================================================================

    def run(
        self,
        *,
        callback: Callable[[StepMetrics], bool | None] | None = None,
    ) -> None:
        """Run the training loop.

        Must be called after ``load_data()``.

        Args:
            callback: Optional function called after each optimizer step.
                Receives a ``StepMetrics`` object. Return ``False`` to cancel
                training; return ``True`` or ``None`` to continue.

        Raises:
            TrainingError: If training fails or is cancelled.
            StateError: If the session is closed.

        Example:
            >>> def on_step(metrics):
            ...     print(f"step {metrics.step}: loss={metrics.loss:.4f}")
            ...     if metrics.loss < 0.1:
            ...         return False  # stop early
            >>> session.run(callback=on_step)
        """
        c_callback, c_user_data = wrap_callback(callback)
        # prevent GC of the C callback closure during training
        self._callback_ref = c_callback
        try:
            code = call_train_run(self._handle, c_callback, c_user_data)
            check(code)
        finally:
            self._callback_ref = None

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self, output_path: str) -> None:
        """Save adapter weights to a file.

        Args:
            output_path: Path for the output checkpoint file.

        Raises:
            TrainingError: If checkpoint saving fails.
            StateError: If the session is closed.

        Example:
            >>> session.save_checkpoint("./adapter_weights.safetensors")
        """
        code = call_train_save_checkpoint(
            self._handle, output_path.encode("utf-8")
        )
        check(code)

    # =========================================================================
    # Info
    # =========================================================================

    @property
    def info(self) -> TrainingInfo:
        """Query current training session state.

        Returns:
            TrainingInfo with current step, total steps, parameter counts,
            adapter layer count, and session state.

        Raises:
            TrainingError: If the query fails.
            StateError: If the session is closed.

        Example:
            >>> info = session.info
            >>> print(f"Step {info.current_step}/{info.total_steps}")
            >>> print(f"State: {info.state.name}")
        """
        code, info = call_train_get_info(self._handle)
        check(code)
        assert info is not None  # pragma: no cover - check() raises on error
        return info
