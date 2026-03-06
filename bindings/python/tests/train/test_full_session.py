"""Tests for talu.train.full_session — FullTrainingSession lifecycle and config types."""

import copy
import ctypes
import gc
import pickle

import pytest

from talu.exceptions import StateError, TrainingError
from talu.train import (
    FullSessionConfig,
    FullSessionInfo,
    FullSessionState,
    FullTrainingSession,
    StepMetrics,
    TransformerConfig,
)
from talu.train._bindings import CStepMetrics
from talu.train._full_bindings import (
    CFullSessionConfig,
    CFullSessionInfo,
    CTransformerConfig,
    from_c_full_session_info,
    to_c_full_session_config,
    to_c_transformer_config,
    wrap_full_callback,
)


# =============================================================================
# FullSessionState enum
# =============================================================================


class TestFullSessionState:
    """FullSessionState enum values match Zig session state machine."""

    def test_values_match_zig(self):
        assert FullSessionState.CREATED == 0
        assert FullSessionState.INITIALIZED == 1
        assert FullSessionState.CONFIGURED == 2
        assert FullSessionState.DATA_LOADED == 3
        assert FullSessionState.TRAINING == 4
        assert FullSessionState.COMPLETED == 5

    def test_all_members_exist(self):
        expected = {"CREATED", "INITIALIZED", "CONFIGURED", "DATA_LOADED", "TRAINING", "COMPLETED"}
        actual = {m.name for m in FullSessionState}
        assert actual == expected

    def test_from_int(self):
        assert FullSessionState(0) is FullSessionState.CREATED
        assert FullSessionState(5) is FullSessionState.COMPLETED


# =============================================================================
# Config Dataclasses
# =============================================================================


class TestTransformerConfig:
    """TransformerConfig with defaults and auto-derived fields."""

    def test_minimal(self):
        config = TransformerConfig(
            vocab_size=32000, d_model=256, num_layers=4, num_heads=4,
        )
        assert config.vocab_size == 32000
        assert config.d_model == 256
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.num_kv_heads is None
        assert config.d_ff is None
        assert config.seq_len == 256
        assert config.rope_theta == pytest.approx(10000.0)
        assert config.norm_eps == pytest.approx(1e-5)

    def test_to_c_struct_defaults(self):
        """num_kv_heads defaults to num_heads, d_ff defaults to 4*d_model."""
        config = TransformerConfig(
            vocab_size=1000, d_model=64, num_layers=2, num_heads=2,
        )
        c = to_c_transformer_config(config)
        assert isinstance(c, CTransformerConfig)
        assert c.num_kv_heads == 2  # defaults to num_heads
        assert c.d_ff == 256  # defaults to 4 * d_model

    def test_to_c_struct_explicit(self):
        config = TransformerConfig(
            vocab_size=32000, d_model=256, num_layers=4,
            num_heads=4, num_kv_heads=2, d_ff=1024, seq_len=512,
            rope_theta=50000.0, norm_eps=1e-6,
        )
        c = to_c_transformer_config(config)
        assert c.vocab_size == 32000
        assert c.d_model == 256
        assert c.num_layers == 4
        assert c.num_heads == 4
        assert c.num_kv_heads == 2
        assert c.d_ff == 1024
        assert c.seq_len == 512
        assert c.rope_theta == pytest.approx(50000.0)
        assert c.norm_eps == pytest.approx(1e-6)


class TestFullSessionConfig:
    """FullSessionConfig dataclass with training defaults."""

    def test_defaults(self):
        config = FullSessionConfig()
        assert config.learning_rate == pytest.approx(3e-4)
        assert config.min_learning_rate == pytest.approx(3e-5)
        assert config.weight_decay == pytest.approx(0.1)
        assert config.beta1 == pytest.approx(0.9)
        assert config.beta2 == pytest.approx(0.95)
        assert config.epsilon == pytest.approx(1e-8)
        assert config.batch_size == 32
        assert config.warmup_steps == 500
        assert config.total_steps == 10000
        assert config.max_grad_norm == pytest.approx(1.0)

    def test_custom_values(self):
        config = FullSessionConfig(learning_rate=1e-3, total_steps=500)
        assert config.learning_rate == pytest.approx(1e-3)
        assert config.total_steps == 500

    def test_to_c_struct(self):
        config = FullSessionConfig(learning_rate=5e-4, batch_size=16, total_steps=2000)
        c = to_c_full_session_config(config)
        assert isinstance(c, CFullSessionConfig)
        assert c.learning_rate == pytest.approx(5e-4)
        assert c.batch_size == 16
        assert c.total_steps == 2000
        assert c.warmup_steps == 500  # default preserved


class TestFullSessionInfo:
    """FullSessionInfo is frozen with FullSessionState enum."""

    def test_frozen(self):
        info = FullSessionInfo(
            current_step=0, total_steps=100,
            total_params=1000, batch_size=32,
            state=FullSessionState.CREATED,
        )
        with pytest.raises(AttributeError):
            info.current_step = 1  # type: ignore[misc]

    def test_state_is_enum(self):
        info = FullSessionInfo(
            current_step=50, total_steps=100,
            total_params=2000, batch_size=16,
            state=FullSessionState.TRAINING,
        )
        assert info.state is FullSessionState.TRAINING
        assert info.state == 4

    def test_from_c_struct(self):
        c = CFullSessionInfo(
            current_step=10, total_steps=500,
            total_params=4096, batch_size=8,
            state=3,
        )
        info = from_c_full_session_info(c)
        assert isinstance(info, FullSessionInfo)
        assert info.current_step == 10
        assert info.total_steps == 500
        assert info.total_params == 4096
        assert info.batch_size == 8
        assert info.state is FullSessionState.DATA_LOADED


# =============================================================================
# Struct Sizes
# =============================================================================


class TestStructSizes:
    """Verify ctypes struct sizes match Zig extern struct sizes."""

    def test_transformer_config_size(self):
        # 7 × u32 (28) + 2 × f32 (8) = 36
        assert ctypes.sizeof(CTransformerConfig) == 36

    def test_full_session_config_size(self):
        # 7 × f32 (28) + 1 × u32 (4) + 2 × u64 (16) = 48
        assert ctypes.sizeof(CFullSessionConfig) == 48

    def test_full_session_info_size(self):
        # 3 × u64 (24) + 1 × u32 (4) + 1 × u8 (1) + 3 × padding (3) = 32
        assert ctypes.sizeof(CFullSessionInfo) == 32


# =============================================================================
# Callback Wrapper
# =============================================================================


class TestFullCallbackWrapper:
    """wrap_full_callback converts Python callables to C-compatible callbacks."""

    def test_none_callback(self):
        c_cb, c_ud = wrap_full_callback(None)
        assert c_cb is None

    def test_wraps_callable(self):
        c_cb, c_ud = wrap_full_callback(lambda m: None)
        assert c_cb is not None

    def test_continue_on_true(self):
        c_cb, _ = wrap_full_callback(lambda m: True)
        metrics = CStepMetrics(step=1, loss=1.0, learning_rate=1e-4, grad_norm=0.5)
        result = c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))
        assert result == 0

    def test_cancel_on_false(self):
        c_cb, _ = wrap_full_callback(lambda m: False)
        metrics = CStepMetrics(step=1, loss=1.0, learning_rate=1e-4, grad_norm=0.5)
        result = c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))
        assert result == 1

    def test_callback_receives_step_metrics(self):
        received = []
        c_cb, _ = wrap_full_callback(lambda m: received.append(m))
        metrics = CStepMetrics(step=42, loss=2.5, learning_rate=3e-4, grad_norm=0.7)
        c_cb(ctypes.pointer(metrics), ctypes.c_void_p(0))

        assert len(received) == 1
        assert isinstance(received[0], StepMetrics)
        assert received[0].step == 42
        assert received[0].loss == pytest.approx(2.5)


# =============================================================================
# FullTrainingSession Lifecycle
# =============================================================================


class TestFullTrainingSessionLifecycle:
    """FullTrainingSession follows Holy Trinity lifecycle pattern."""

    def test_create_and_destroy(self):
        session = FullTrainingSession()
        assert session._ptr is not None
        session.close()
        assert session._ptr is None

    def test_context_manager(self):
        with FullTrainingSession() as session:
            assert session._ptr is not None
        assert session._ptr is None

    def test_double_close_is_noop(self):
        session = FullTrainingSession()
        session.close()
        session.close()

    def test_use_after_close_raises_state_error(self):
        session = FullTrainingSession()
        session.close()
        with pytest.raises(StateError, match="closed"):
            session.info

    def test_repr_open(self):
        with FullTrainingSession() as session:
            assert "open" in repr(session)

    def test_repr_closed(self):
        session = FullTrainingSession()
        session.close()
        assert "closed" in repr(session)

    def test_copy_raises(self):
        with FullTrainingSession() as session:
            with pytest.raises(TypeError):
                copy.copy(session)

    def test_pickle_raises(self):
        with FullTrainingSession() as session:
            with pytest.raises(TypeError):
                pickle.dumps(session)

    def test_gc_cleanup(self):
        """Session cleanup via __del__ does not raise."""
        session = FullTrainingSession()
        assert session._ptr is not None
        del session
        gc.collect()

    def test_info_on_fresh_session(self):
        """Fresh session reports CREATED state."""
        with FullTrainingSession() as session:
            info = session.info
            assert info.state is FullSessionState.CREATED
            assert info.current_step == 0
            assert info.total_params == 0

class TestFullTrainingSessionWeightExport:
    """FullTrainingSession can export flat model weights for checkpointing."""

    def test_export_weights_matches_parameter_count(self):
        config = TransformerConfig(
            vocab_size=32,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            seq_len=8,
        )
        with FullTrainingSession() as session:
            session.init_model(config, seed=123)

            weights = session.export_weights_f32()
            assert len(weights) == session.info.total_params
            assert any(value != 0.0 for value in weights)

            token_embedding_size = config.vocab_size * config.d_model
            assert weights[token_embedding_size] == pytest.approx(1.0)

    def test_export_weights_before_init_raises_training_error(self):
        with FullTrainingSession() as session:
            with pytest.raises(TrainingError) as exc_info:
                session.export_weights_f32()
        assert exc_info.value.code == "TRAIN_INVALID_STATE"

    def test_import_weights_restores_exported_values_and_step(self):
        config = TransformerConfig(
            vocab_size=32,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            seq_len=8,
        )
        with FullTrainingSession() as source:
            source.init_model(config, seed=123)
            exported = source.export_weights_f32()

        with FullTrainingSession() as restored:
            restored.init_model(config, seed=999)
            restored.import_weights_f32(exported, step=17)
            roundtrip = restored.export_weights_f32()
            assert list(roundtrip) == list(exported)
            assert restored.info.current_step == 17

    def test_import_optimizer_state_restores_exported_values(self):
        config = TransformerConfig(
            vocab_size=32,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            seq_len=8,
        )
        training = FullSessionConfig(total_steps=4, batch_size=1)
        tokens = [0, 1, 2, 3, 4, 5, 6, 7, 0]

        with FullTrainingSession() as source:
            source.init_model(config, seed=123)
            source.configure(training)
            source.set_data(tokens)
            source.step()
            exported_weights = source.export_weights_f32()
            exported_opt = source.export_optimizer_state_f32()

        with FullTrainingSession() as restored:
            restored.init_model(config, seed=999)
            restored.configure(training)
            restored.set_data(tokens)
            restored.import_weights_f32(exported_weights, step=1)
            restored.import_optimizer_state_f32(exported_opt)
            assert list(restored.export_optimizer_state_f32()) == list(exported_opt)

    def test_resume_next_step_matches_uninterrupted_training(self):
        config = TransformerConfig(
            vocab_size=32,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            seq_len=4,
        )
        training = FullSessionConfig(total_steps=6, batch_size=1)
        tokens = [
            0, 1, 2, 3, 4,
            5, 6, 7, 8, 9,
            10, 11, 12, 13, 14,
            15, 16, 17, 18, 19,
            20, 21, 22, 23, 24,
        ]

        with FullTrainingSession() as uninterrupted:
            uninterrupted.init_model(config, seed=123)
            uninterrupted.configure(training)
            uninterrupted.set_data(tokens)
            uninterrupted.step()
            exported_weights = uninterrupted.export_weights_f32()
            exported_opt = uninterrupted.export_optimizer_state_f32()
            second_live = uninterrupted.step()
            final_weights_live = uninterrupted.export_weights_f32()
            final_opt_live = uninterrupted.export_optimizer_state_f32()

        with FullTrainingSession() as resumed:
            resumed.init_model(config, seed=999)
            resumed.configure(training)
            resumed.set_data(tokens)
            resumed.import_weights_f32(exported_weights, step=1)
            resumed.import_optimizer_state_f32(exported_opt)
            second_resumed = resumed.step()
            final_weights_resumed = resumed.export_weights_f32()
            final_opt_resumed = resumed.export_optimizer_state_f32()

        assert second_resumed.step == second_live.step
        assert second_resumed.loss == pytest.approx(second_live.loss, abs=0.0)
        assert second_resumed.learning_rate == pytest.approx(second_live.learning_rate, abs=0.0)
        assert second_resumed.grad_norm == pytest.approx(second_live.grad_norm, abs=0.0)
        assert list(final_weights_resumed) == list(final_weights_live)
        assert list(final_opt_resumed) == list(final_opt_live)
