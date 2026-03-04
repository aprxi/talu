"""Tests for talu.train.session — TrainingSession lifecycle and config types."""

import gc

import pytest

from talu._bindings import ERROR_MAP
from talu.exceptions import StateError, TrainingError
from talu.train import (
    LoraConfig,
    StepMetrics,
    TrainingConfig,
    TrainingInfo,
    TrainingSession,
    TrainingState,
)
from talu.train._bindings import (
    CLoraConfig,
    CStepMetrics,
    CTrainingConfig,
    CTrainingInfo,
    from_c_step_metrics,
    from_c_training_info,
    to_c_lora_config,
    to_c_training_config,
    wrap_callback,
)


# =============================================================================
# TrainingState enum
# =============================================================================


class TestTrainingState:
    """TrainingState enum values match Zig session state machine."""

    def test_values_match_zig(self):
        assert TrainingState.CREATED == 0
        assert TrainingState.MODEL_LOADED == 1
        assert TrainingState.CONFIGURED == 2
        assert TrainingState.DATA_LOADED == 3
        assert TrainingState.TRAINING == 4
        assert TrainingState.COMPLETED == 5

    def test_all_members_exist(self):
        expected = {"CREATED", "MODEL_LOADED", "CONFIGURED", "DATA_LOADED", "TRAINING", "COMPLETED"}
        actual = {m.name for m in TrainingState}
        assert actual == expected

    def test_from_int(self):
        assert TrainingState(0) is TrainingState.CREATED
        assert TrainingState(5) is TrainingState.COMPLETED


# =============================================================================
# Config Dataclasses
# =============================================================================


class TestLoraConfig:
    """LoraConfig dataclass with defaults."""

    def test_defaults(self):
        config = LoraConfig()
        assert config.rank == 16
        assert config.alpha == 32.0

    def test_custom_values(self):
        config = LoraConfig(rank=8, alpha=16.0)
        assert config.rank == 8
        assert config.alpha == 16.0

    def test_to_c_struct(self):
        config = LoraConfig(rank=4, alpha=8.0)
        c = to_c_lora_config(config)
        assert isinstance(c, CLoraConfig)
        assert c.rank == 4
        assert pytest.approx(c.alpha) == 8.0


class TestTrainingConfig:
    """TrainingConfig dataclass with PyTorch-convention defaults."""

    def test_defaults(self):
        config = TrainingConfig()
        assert config.learning_rate == pytest.approx(1e-4)
        assert config.min_learning_rate == pytest.approx(1e-6)
        assert config.weight_decay == pytest.approx(0.01)
        assert config.beta1 == pytest.approx(0.9)
        assert config.beta2 == pytest.approx(0.999)
        assert config.epsilon == pytest.approx(1e-8)
        assert config.batch_size == 4
        assert config.seq_len == 128
        assert config.warmup_steps == 100
        assert config.total_steps == 1000
        assert config.max_grad_norm == pytest.approx(1.0)
        assert config.gradient_accumulation_steps == 1
        assert config.log_interval == 10
        assert config.save_interval == 0

    def test_custom_values(self):
        config = TrainingConfig(learning_rate=5e-5, total_steps=500, batch_size=8)
        assert config.learning_rate == pytest.approx(5e-5)
        assert config.total_steps == 500
        assert config.batch_size == 8

    def test_to_c_struct(self):
        config = TrainingConfig(learning_rate=2e-4, total_steps=200)
        c = to_c_training_config(config)
        assert isinstance(c, CTrainingConfig)
        assert pytest.approx(c.learning_rate) == 2e-4
        assert c.total_steps == 200
        assert c.batch_size == 4  # default preserved


class TestStepMetrics:
    """StepMetrics is frozen (immutable)."""

    def test_frozen(self):
        m = StepMetrics(step=10, loss=2.5, learning_rate=1e-4, grad_norm=0.5)
        with pytest.raises(AttributeError):
            m.step = 20  # type: ignore[misc]

    def test_values(self):
        m = StepMetrics(step=42, loss=1.23, learning_rate=5e-5, grad_norm=0.8)
        assert m.step == 42
        assert m.loss == pytest.approx(1.23)
        assert m.learning_rate == pytest.approx(5e-5)
        assert m.grad_norm == pytest.approx(0.8)

    def test_from_c_struct(self):
        c = CStepMetrics(step=5, loss=3.14, learning_rate=1e-3, grad_norm=0.1)
        m = from_c_step_metrics(c)
        assert isinstance(m, StepMetrics)
        assert m.step == 5
        assert m.loss == pytest.approx(3.14)


class TestTrainingInfo:
    """TrainingInfo is frozen with TrainingState enum."""

    def test_frozen(self):
        info = TrainingInfo(
            current_step=0, total_steps=100,
            trainable_params=1000, adapter_layers=4,
            state=TrainingState.CREATED,
        )
        with pytest.raises(AttributeError):
            info.current_step = 1  # type: ignore[misc]

    def test_state_is_enum(self):
        info = TrainingInfo(
            current_step=50, total_steps=100,
            trainable_params=2000, adapter_layers=8,
            state=TrainingState.TRAINING,
        )
        assert info.state is TrainingState.TRAINING
        assert info.state == 4

    def test_from_c_struct(self):
        c = CTrainingInfo(
            current_step=10, total_steps=500,
            trainable_params=4096, adapter_layers=16,
            state=3,
        )
        info = from_c_training_info(c)
        assert isinstance(info, TrainingInfo)
        assert info.current_step == 10
        assert info.total_steps == 500
        assert info.state is TrainingState.DATA_LOADED


# =============================================================================
# Error Mapping
# =============================================================================


class TestErrorMapping:
    """Training error codes 1000-1010 are all mapped to TrainingError."""

    @pytest.mark.parametrize("code", range(1000, 1011))
    def test_code_mapped(self, code):
        assert code in ERROR_MAP, f"Error code {code} not in ERROR_MAP"

    @pytest.mark.parametrize("code", range(1000, 1011))
    def test_maps_to_training_error(self, code):
        exc_class, _ = ERROR_MAP[code]
        assert exc_class is TrainingError

    def test_string_codes_unique(self):
        training_codes = {v[1] for k, v in ERROR_MAP.items() if 1000 <= k <= 1010}
        assert len(training_codes) == 11

    def test_specific_codes(self):
        assert ERROR_MAP[1000] == (TrainingError, "TRAIN_INVALID_STATE")
        assert ERROR_MAP[1010] == (TrainingError, "TRAIN_CANCELLED")


# =============================================================================
# Callback Wrapper
# =============================================================================


class TestCallbackWrapper:
    """wrap_callback converts Python callables to C-compatible callbacks."""

    def test_none_callback(self):
        c_cb, c_ud = wrap_callback(None)
        assert c_cb is None

    def test_wraps_callable(self):
        c_cb, c_ud = wrap_callback(lambda m: None)
        assert c_cb is not None

    def test_continue_on_true(self):
        c_cb, _ = wrap_callback(lambda m: True)
        metrics = CStepMetrics(step=1, loss=1.0, learning_rate=1e-4, grad_norm=0.5)
        import ctypes
        result = c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))
        assert result == 0  # 0 = continue

    def test_cancel_on_false(self):
        c_cb, _ = wrap_callback(lambda m: False)
        metrics = CStepMetrics(step=1, loss=1.0, learning_rate=1e-4, grad_norm=0.5)
        import ctypes
        result = c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))
        assert result == 1  # 1 = cancel

    def test_continue_on_none_return(self):
        c_cb, _ = wrap_callback(lambda m: None)
        metrics = CStepMetrics(step=1, loss=1.0, learning_rate=1e-4, grad_norm=0.5)
        import ctypes
        result = c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))
        assert result == 0  # None → continue

    def test_callback_receives_step_metrics(self):
        received = []

        def cb(m):
            received.append(m)

        c_cb, _ = wrap_callback(cb)
        metrics = CStepMetrics(step=42, loss=2.5, learning_rate=3e-4, grad_norm=0.7)
        import ctypes
        c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))

        assert len(received) == 1
        assert isinstance(received[0], StepMetrics)
        assert received[0].step == 42
        assert received[0].loss == pytest.approx(2.5)


# =============================================================================
# TrainingSession Lifecycle
# =============================================================================


class TestTrainingSessionLifecycle:
    """TrainingSession follows Holy Trinity lifecycle pattern."""

    def test_create_and_destroy(self):
        session = TrainingSession()
        assert session._ptr is not None
        session.close()
        assert session._ptr is None

    def test_context_manager(self):
        with TrainingSession() as session:
            assert session._ptr is not None
        assert session._ptr is None

    def test_double_close_is_noop(self):
        session = TrainingSession()
        session.close()
        session.close()  # should not raise

    def test_use_after_close_raises_state_error(self):
        session = TrainingSession()
        session.close()
        with pytest.raises(StateError, match="closed"):
            session.info

    def test_repr_open(self):
        with TrainingSession() as session:
            assert "open" in repr(session)

    def test_repr_closed(self):
        session = TrainingSession()
        session.close()
        assert "closed" in repr(session)

    def test_copy_raises(self):
        import copy

        with TrainingSession() as session:
            with pytest.raises(TypeError):
                copy.copy(session)

    def test_pickle_raises(self):
        import pickle

        with TrainingSession() as session:
            with pytest.raises(TypeError):
                pickle.dumps(session)

    def test_gc_cleanup(self):
        """Session cleanup via __del__ does not raise."""
        session = TrainingSession()
        ptr = session._ptr
        assert ptr is not None
        del session
        gc.collect()
        # If we get here without error, __del__ worked

    def test_info_on_fresh_session(self):
        """Fresh session reports CREATED state."""
        with TrainingSession() as session:
            info = session.info
            assert info.state is TrainingState.CREATED
            assert info.current_step == 0

    def test_load_model_stub_returns_error(self):
        """load_model is a Zig stub — verify error propagation."""
        with TrainingSession() as session:
            with pytest.raises(TrainingError) as exc_info:
                session.load_model(
                    "./nonexistent",
                    lora=LoraConfig(rank=8),
                )
            assert exc_info.value.code == "TRAIN_MODEL_LOAD_FAILED"

    def test_configure_without_model_raises(self):
        """configure requires model_loaded state."""
        with TrainingSession() as session:
            with pytest.raises(TrainingError):
                session.configure(TrainingConfig())
