"""
From-scratch transformer training session.

Provides the FullTrainingSession class for training small transformer
models from scratch using talu's Zig training engine.
"""

from collections.abc import Callable
from typing import Any

from .._bindings import check
from ..exceptions import StateError
from ._bindings import StepMetrics
from ._full_bindings import (
    FullSessionConfig,
    FullSessionInfo,
    TransformerConfig,
    call_train_full_configure,
    call_train_full_create,
    call_train_full_destroy,
    call_train_full_get_info,
    call_train_full_init_model,
    call_train_full_load_data,
    call_train_full_run,
    call_train_full_set_data,
    call_train_full_step,
    wrap_full_callback,
)


class FullTrainingSession:
    """From-scratch transformer training session.

    Manages the lifecycle of a full training session: initialize model
    with random weights, configure hyperparameters, load data, run
    training with per-step callbacks.

    Implements the standard talu lifecycle pattern: explicit ``close()``,
    context manager (``with``), and fail-safe ``__del__``.

    Args:
        None. Call ``init_model()`` after construction to begin setup.

    Example:
        >>> with FullTrainingSession() as session:
        ...     session.init_model(TransformerConfig(
        ...         vocab_size=32000, d_model=256,
        ...         num_layers=4, num_heads=4, d_ff=1024,
        ...     ))
        ...     session.configure(FullSessionConfig(
        ...         learning_rate=3e-4, total_steps=10000,
        ...     ))
        ...     session.load_data("./tokens.bin")
        ...     session.run(callback=lambda m: print(f"step {m.step}: loss={m.loss:.4f}"))
    """

    __slots__ = ("_ptr", "_callback_ref")

    def __init__(self) -> None:
        code, ptr = call_train_full_create()
        check(code)
        self._ptr: int | None = ptr
        self._callback_ref: Any = None

    @property
    def _handle(self) -> int:
        """Get the internal handle, raising if closed."""
        if self._ptr is None:
            raise StateError("FullTrainingSession is closed")
        return self._ptr

    def close(self) -> None:
        """Release native training session resources.

        After calling close(), the session cannot be used. Safe to call
        multiple times (idempotent).
        """
        if self._ptr is not None:
            call_train_full_destroy(self._ptr)
            self._ptr = None
            self._callback_ref = None

    def __enter__(self) -> "FullTrainingSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "closed" if self._ptr is None else "open"
        return f"FullTrainingSession({state})"

    def __copy__(self) -> None:
        raise TypeError("FullTrainingSession cannot be copied")

    def __reduce__(self) -> None:
        raise TypeError("FullTrainingSession cannot be pickled")

    # =========================================================================
    # Model Setup
    # =========================================================================

    def init_model(
        self,
        config: TransformerConfig,
        *,
        seed: int = 42,
    ) -> None:
        """Initialize model with random weights.

        Args:
            config: Transformer architecture configuration.
            seed: Random seed for weight initialization.

        Raises:
            TaluError: If initialization fails.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> session.init_model(TransformerConfig(
            ...     vocab_size=32000, d_model=256,
            ...     num_layers=4, num_heads=4,
            ... ))
        """
        code = call_train_full_init_model(self._handle, config, seed)
        check(code)

    def configure(self, config: FullSessionConfig) -> None:
        """Set training hyperparameters.

        Must be called after ``init_model()`` and before ``load_data()``.

        Args:
            config: Training hyperparameters.

        Raises:
            TaluError: If configuration fails.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> session.configure(FullSessionConfig(
            ...     learning_rate=3e-4,
            ...     total_steps=10000,
            ...     batch_size=32,
            ... ))
        """
        code = call_train_full_configure(self._handle, config)
        check(code)

    # =========================================================================
    # Data Loading
    # =========================================================================

    def set_data(self, tokens: list[int]) -> None:
        """Set tokenized training data from a list of token IDs.

        The tokens are copied to native memory. Must be called after
        ``configure()``.

        Args:
            tokens: List of token IDs (u32 values).

        Raises:
            TaluError: If data loading fails.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> session.set_data([0, 1, 2, 3, 4, 5, 6, 7])
        """
        code = call_train_full_set_data(self._handle, tokens)
        check(code)

    def load_data(self, data_path: str) -> None:
        """Load tokenized training data from a flat binary file.

        The file must contain flat binary u32 tokens (native endianness).
        Must be called after ``configure()``.

        Args:
            data_path: Path to the tokenized data file.

        Raises:
            TaluError: If data loading fails.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> session.load_data("./train_tokens.bin")
        """
        code = call_train_full_load_data(
            self._handle, data_path.encode("utf-8")
        )
        check(code)

    # =========================================================================
    # Training
    # =========================================================================

    def step(self) -> StepMetrics:
        """Run one training step: forward, backward, clip, optimizer.

        Returns:
            StepMetrics with loss, learning rate, gradient norm, and step number.

        Raises:
            TaluError: If the step fails.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> metrics = session.step()
            >>> print(f"loss={metrics.loss:.4f}")
        """
        code, metrics = call_train_full_step(self._handle)
        check(code)
        assert metrics is not None
        return metrics

    def run(
        self,
        *,
        callback: Callable[[StepMetrics], bool | None] | None = None,
    ) -> None:
        """Run the full training loop.

        Must be called after ``load_data()``.

        Args:
            callback: Optional function called after each step.
                Receives a ``StepMetrics`` object. Return ``False`` to cancel
                training; return ``True`` or ``None`` to continue.

        Raises:
            TaluError: If training fails or is cancelled.
            StateError: If the session is closed or in wrong state.

        Example:
            >>> def on_step(metrics):
            ...     print(f"step {metrics.step}: loss={metrics.loss:.4f}")
            ...     if metrics.loss < 0.1:
            ...         return False  # stop early
            >>> session.run(callback=on_step)
        """
        c_callback, c_user_data = wrap_full_callback(callback)
        self._callback_ref = c_callback
        try:
            code = call_train_full_run(self._handle, c_callback, c_user_data)
            check(code)
        finally:
            self._callback_ref = None

    # =========================================================================
    # Info
    # =========================================================================

    @property
    def info(self) -> FullSessionInfo:
        """Query current training session state.

        Returns:
            FullSessionInfo with current step, total steps, parameter count,
            batch size, and session state.

        Raises:
            TaluError: If the query fails.
            StateError: If the session is closed.

        Example:
            >>> info = session.info
            >>> print(f"Step {info.current_step}/{info.total_steps}")
            >>> print(f"Params: {info.total_params:,}")
        """
        code, info = call_train_full_get_info(self._handle)
        check(code)
        assert info is not None
        return info
